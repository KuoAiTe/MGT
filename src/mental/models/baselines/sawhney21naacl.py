import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .. import BaseModel, ModelOutput
from ...utils.dataprocessing import prepare_text_inputs, prepare_batch_text_inputs

class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward (self, x, adj):
        n = x.size(0)
        # n x 1 x d
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        # 1 x n x d
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)

        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat).squeeze()
        att_adj = F.sigmoid(att_adj)
        att_adj = torch.mul(adj.to_dense(), att_adj)
        return att_adj

def arcosh(x):
    return Arcosh.apply(x)


def cosh(x, clamp = 7):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=7):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=7):
    return x.clamp(-clamp, clamp).tanh()


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-7, 1 - 1e-7)
        ctx.save_for_backward(x)
        z = x.float()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.float()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-7).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-7)
        ctx.save_for_backward(x)
        z = x.float()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-7).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5


def get_dim_act_curv(feat_dim, dim, num_layers, act):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    acts = [act] * (num_layers - 1)
    dims = [feat_dim] + ([dim] * (num_layers - 1))
    #n_curvatures = num_layers - 1
    n_curvatures = num_layers
    curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    #curvatures = [torch.Tensor([1.]).to('mps') for _ in range(n_curvatures)]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(nn.Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = False
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class PoincareManifold(nn.Module):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature. 
    """

    def __init__(self, ):
        super(PoincareManifold, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-7
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        sqrt_c = c.clamp(1e-7) ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c.clamp(1e-7) ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c.clamp(1e-7) ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c.clamp(1e-7) ** 0.5
        #print('sqrt_c', sqrt_c, c)
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c.clamp(1e-7) ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c.clamp(1e-7) ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

class HGCN(nn.Module):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, num_layers):
        super(HGCN, self).__init__()
        self.manifold = PoincareManifold()
        dropout = 0.1
        bias = 1
        use_att = 1
        local_agg = 1
        self.c = nn.Parameter(torch.Tensor([1.]))
        dims, acts, self.curvatures = get_dim_act_curv(feat_dim = 256, dim = 256, num_layers = 3, act= F.relu)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, dropout, act, bias, use_att, local_agg
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def reset_parameters(self):
        pass
            
    def forward(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        #print('x_tan', x_tan)
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        #print('self.manifold.expmap0', x_hyp)
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        #print('self.manifold.proj', x_hyp)
        return self.layers((x_hyp, adj))


class HawkesAttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HawkesAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.LSTM(input_size, hidden_size, bidirectional = True, batch_first = True)
        self.tweet_history_sentence_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.tweet_history_sentence_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.beta = 1e-3
        self.eps = 1e-2

        self.context_vector = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, hidden_states, time_gap):
        # Apply the bidirectional GRU to the input
        f_output, _ = self.gru(hidden_states)  # f_output shape: (sequence_length, 2 * sent_hidden_size)
        u_i = torch.tanh(f_output @ self.tweet_history_sentence_weight + self.tweet_history_sentence_bias)
        attention_weights = u_i @ self.context_vector  # attention_weights shape: (sequence_length, 1)
        
        attention_weights = F.softmax(attention_weights, dim = 1)  # attention_weights shape: (sequence_length, , 1)
        lambda_i = f_output * attention_weights
        
        labmda_prime_i = lambda_i.relu()
        # alpha: self.eps * lambda^{prime}_{i} as the amount of excitement the post at time step i contributes to the current decision
        alpha = self.eps * labmda_prime_i

        # lambda_t shape: (batch_size, length, dim)
        lambda_t = lambda_i + alpha * torch.exp(-self.beta * time_gap).unsqueeze(-1).repeat(1, 1, lambda_i.shape[-1])
        
        # Compute the history-level embedding  
        # v is the user embedding that summarizes all the information of tweets of a user's history.
        # size: (batch_size, dim)
        v = torch.sum(lambda_t, dim = 1)

        return v

class Sawhney_NAACL_21(BaseModel):
    def __init__(self, args, data_info):
        super().__init__()
        input_size = args.input_size
        config = args.rnn_config
        hidden_size = config.hidden_size
        self.hawkes = HawkesAttentionLayer(input_size, hidden_size)
        self.hyperbolic_gcn = HGCN(num_layers = 3)
        
        self.depression_prediction_head = nn.Linear(2 * hidden_size, 1)
        self.allowed_inputs = ['text_data']
        self.reset_parameters()


    def reset_parameters(self):
        self.hyperbolic_gcn.reset_parameters()
        self.hawkes.reset_parameters()
        self.depression_prediction_head.reset_parameters()

    def forward(self, text_data) -> Tensor:
        logits = []

        for user in text_data:
            user_history = user['user_embeddings'][:-1].unsqueeze(0)
            user_tweets_timestamps = user['user_tweets_timestamps'][:-1].unsqueeze(0)
            user_tweets_timegap = torch.roll(user_tweets_timestamps , -1) - user_tweets_timestamps
            user_tweets_timegap[-1] = 0
            # We assume each user has the same number of tweets, which is pre-determined and confirmed in the preprocessing.
            # Otherwise, this needs to be checked before being fed into the rnn.
            # Num_Friend * Tweets * Embedding_Dim 
            friend_history  = user['friend_embeddings'].reshape(-1, user['user_embeddings'].shape[0], user['user_embeddings'].shape[-1])
            friend_tweets_timestamps  = user['friend_tweets_timestamps'].reshape(friend_history.shape[:2])
            friend_tweets_timegap = torch.roll(friend_tweets_timestamps , -1) - friend_tweets_timestamps
            friend_tweets_timegap[:, -1] = 0

            user_out = self.hawkes(user_history, user_tweets_timegap)
            #print('user_history', user_history)
            #print('user_out', user_out)
            friend_out = self.hawkes(friend_history, friend_tweets_timegap)
            #print('friend_history', friend_history)
            #print('friend_out', friend_out)
            
            # concatenate user and friend embeddings for node attributes
            # Shape: [1 + num_friends, dim]
            features = torch.cat([user_out, friend_out], dim = 0)
            #print('features', features.shape)
            # Adj: Custom adj. 
            num_nodes = features.shape[0]
            # Self-loop
            A = torch.diag(torch.ones((num_nodes), device = features.device, requires_grad = False))
            # User to all, friends are not connected.
            A[0, :] = 1
            A[:, 0] = 1
            #A[:] = 1
        
            # Compute degree matrix
            D = torch.diag(torch.sum(A, dim=1))

            # Compute degree matrix with inverse square root
            D_sqrt_inv = torch.diag(torch.pow(torch.sum(A, dim=1), -0.5))

            # Compute normalized adjacency matrix
            A_norm = torch.matmul(torch.matmul(D_sqrt_inv, A), D_sqrt_inv)

            #print('Before features', features)
            hgcn_out, adj = self.hyperbolic_gcn(features, A_norm)
            #print('hgcn_out', hgcn_out)
            #friend_history_tweets = user['friend_embeddings'].reshape(-1, user['user_embeddings'].shape[0],  user['user_embeddings'].shape[-1])
            #time_gap = torch.arange(friend_history_tweets.shape[1], device = history_tweets.device).unsqueeze(-1).repeat(1, 9) * 100000
            #print(time_gap.shape)
            #heat_out = self.hawkes(friend_history_tweets, time_gap)
            #print(friend_history_tweets.shape)
            #exit()
            user_node = hgcn_out[0]
            
            logits.append(user_node)
        logits = torch.stack(logits)
        cls_logits = self.depression_prediction_head(logits)
        prediction_scores = torch.sigmoid(cls_logits)

        return ModelOutput(
            logits = logits,
            cls_logits = cls_logits,
            prediction_scores = prediction_scores,
        )
    

    #def prepare_inputs(self, inputs):
    #    return prepare_text_inputs(inputs)
    def prepare_inputs(self, inputs):
        return prepare_text_inputs(inputs)
    
    def prepare_batch_inputs(self, inputs):
        return prepare_batch_text_inputs(inputs)
    

