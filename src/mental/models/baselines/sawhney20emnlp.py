import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from .. import BaseModel, ModelOutput
from ...utils.dataprocessing import prepare_text_inputs, prepare_batch_text_inputs

class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Network parameters of the subspace decomposition
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_d = nn.Parameter(torch.Tensor(hidden_size))
        # Network parameters of the forget GATE
        self.W_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # Network parameters of the input gate
        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Network parameters of the output gate
        self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # Network parameters of the candidate memory
        self.W_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def heuristic_decaying_function(self, elapsed_time):
        # The original paper selected g(k) = 1/ ∆t. It however won't work in our settings. We used another suggested function.
        # g(∆t) = 1/ log (e + ∆t ) [25] is preferred for datasets with large elapsed times
        # g(∆t) = 1 / ∆t can be chosen for datasets with small amount of elapsed time
        return 1 / math.log(math.e + elapsed_time)

    def forward(self, inputs, created_at):
        device = inputs.device
        # Initial states and memories are zero tensors.
        prev_hidden_state = torch.zeros(self.hidden_size, device = device)
        prev_memory = torch.zeros(self.hidden_size, device = device)
        outputs = []
        for t in range(inputs.shape[0]):
            x_t = inputs[t]
            # ...First is zero.
            elapsed_time = created_at[t] - created_at[t - 1] if t != 0 else created_at[0] * 0
            # (Short-term memory C^{S}_{t-1})
            short_term_memory = torch.tanh(prev_memory @ self.W_d + self.b_d)

            # (Discounted short-term memory)
            discounted_short_term_memory = short_term_memory * self.heuristic_decaying_function(elapsed_time)
            # (Long-term memory)
            long_term_memory = prev_memory - short_term_memory

            # (Adjusted previous memory)
            adjusted_previous_memory = long_term_memory + discounted_short_term_memory
            # Input gate
            input_gate = torch.sigmoid(x_t @ self.W_i + prev_hidden_state @ self.U_i + self.b_i)

            # Forget Gate
            forget_gate = torch.sigmoid(x_t @ self.W_f + prev_hidden_state @ self.U_f + self.b_f)

            # Output Gate
            output_gate = torch.sigmoid(x_t @ self.W_o  + prev_hidden_state @ self.U_o + self.b_o)

            # Candidate Memory Cell
            candidate_memory_cell = torch.tanh(x_t @ self.W_c + prev_hidden_state @ self.U_c + self.b_c)

            # Current Memory cell
            current_memory = forget_gate * adjusted_previous_memory + input_gate * candidate_memory_cell
            current_hidden_state = output_gate * torch.tanh(prev_memory)
            outputs.append(current_hidden_state)

            # The current state becomes the previous state when t moves foward to the next timestamp.
            prev_hidden_state = current_hidden_state
            prev_memory = current_memory

        outputs = torch.stack(outputs, dim = 0)
        return outputs


class Sawhney_EMNLP_20(BaseModel):
    def __init__(self, args, data_info):
        super().__init__()
        input_size = args.input_size
        config = args.rnn_config
        hidden_size = config.hidden_size
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc_concat = nn.Linear(2 * hidden_size, hidden_size)
        self.tlstm = TimeLSTM(input_size, hidden_size)
        self.depression_prediction_head = nn.Linear(hidden_size, 1)
        self.act = F.relu
        self.allowed_inputs = ['text_data']
        self.reset_parameters()


    def reset_parameters(self):
        self.fc.reset_parameters()
        self.fc_concat.reset_parameters()
        self.tlstm.reset_parameters()
        self.depression_prediction_head.reset_parameters()

    def forward(self, text_data) -> Tensor:
        logits = []
        for user in text_data:
            raw_tweet_features = user['user_embeddings'][-1]
            tweet_features = self.fc(raw_tweet_features)
            tweet_features = self.act(tweet_features)

            history_tweet_features = user['user_embeddings'][:-1]
            history_tweet_created_at = user['user_tweets_timestamps'][:-1]

            lstm_out = self.tlstm(history_tweet_features, history_tweet_created_at)
            # The figure in the paper seems to use the last hidden state. However, the implementation seems to use mean pooling.
            # We here follow the code in their implementation
            # Reference: https://github.com/midas-research/STATENet_Time_Aware_Suicide_Assessment/blob/master/model/model.py
            pooled_lstm_out = torch.mean(lstm_out, dim = 0)
            combined_features = torch.cat([tweet_features, pooled_lstm_out], dim = -1)
            combined_features = F.dropout(combined_features, training = self.training)
            out = self.fc_concat(combined_features)
            out = self.act(out)
            out = F.dropout(out, training = self.training)

            logits.append(out)
        logits = torch.stack(logits)
        cls_logits = self.depression_prediction_head(logits)
        prediction_scores = torch.sigmoid(cls_logits)

        return ModelOutput(
            logits = logits,
            cls_logits = cls_logits,
            prediction_scores = prediction_scores,
        )
    

    def prepare_inputs(self, inputs):
        return prepare_text_inputs(inputs)
    
    def prepare_batch_inputs(self, inputs):
        return prepare_batch_text_inputs(inputs)

