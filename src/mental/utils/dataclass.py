import numpy as np
from dataclasses import dataclass, field
from typing import List
from ..models import (
    GCNWrapper, GraphSAGEWrapper, GATWrapper,
    DySAT, DyHAN,
    MentalNet, MentalNet_Original, MentalNet_SAGE, MentalNet_GAT, MentalNet_GAT2,
    MentalNetNaive,
    MentalPlus, MentalPlus_Base, MentalPlus_Without_Transformer, MentalPlus_HOMO,MentalPlus_SUP,
    MentalPlus_NO_GNN, MentalPlus_BatchNorm,
    MentalPlus_NO_TIMEFRAME_CUTOUT, MentalPlus_NO_CONTENT_ATTENTION, MentalPlus_NO_INTERACTION_CUTOUT,
    ContrastEgo, ContrastEgo_Base,
    UGformer, EvolveGCN,
    DynamicGNNWrapper, DynamicGAT, DynamicGCN, DynamicSAGE,
    CNNWithMax, CNNWithMaxPlus,
    UsmanBiLSTM, UsmanBiLSTMPlus,
    Sawhney_NAACL_21, Sawhney_EMNLP_20
)


@dataclass
class BaselineModel:
    GCN: object = GCNWrapper
    GAT: object = GATWrapper
    GraphSAGE: object = GraphSAGEWrapper
    DynamicGAT: object = DynamicGAT
    DynamicGCN: object = DynamicGCN
    DynamicSAGE: object = DynamicSAGE
    MentalNet: object = MentalNet
    MentalNet_Original: object = MentalNet_Original
    MentalNet_SAGE: object = MentalNet_SAGE
    MentalNet_GAT: object = MentalNet_GAT
    MentalNet_GAT2: object = MentalNet_GAT2
    MentalNetNaive: object = MentalNetNaive
    DySAT: object = DySAT
    DyHAN: object = DyHAN
    UGformer: object = UGformer
    EvolveGCN: object = EvolveGCN
    UsmanBiLSTM: object = UsmanBiLSTM
    UsmanBiLSTMPlus: object = UsmanBiLSTMPlus
    Sawhney_EMNLP_20: object = Sawhney_EMNLP_20
    Sawhney_NAACL_21: object = Sawhney_NAACL_21
    CNNWithMax: object = CNNWithMax
    CNNWithMaxPlus: object = CNNWithMaxPlus
    MentalPlus: object = MentalPlus
    MentalPlus_HOMO: object = MentalPlus_HOMO
    MentalPlus_SUP: object = MentalPlus_SUP
    MentalPlus_NO_TIMEFRAME_CUTOUT: object = MentalPlus_NO_TIMEFRAME_CUTOUT
    MentalPlus_NO_INTERACTION_CUTOUT: object = MentalPlus_NO_INTERACTION_CUTOUT
    MentalPlus_NO_CONTENT_ATTENTION: object = MentalPlus_NO_CONTENT_ATTENTION
    MentalPlus_Without_Transformer: object = MentalPlus_Without_Transformer
    MentalPlus_BatchNorm: object = MentalPlus_BatchNorm
    MentalPlus_NO_GNN: object = MentalPlus_NO_GNN
    MentalPlus_Base: object = MentalPlus_Base
    ContrastEgo: object = ContrastEgo
    ContrastEgo_Base: object = ContrastEgo_Base
    
@dataclass
class EvaluationResult:
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    num_depressed: int = 0
    num_control: int = 0
    precision_depressed: float = 0.0
    recall_depressed: float = 0.0
    f1_depressed: float = 0.0
    acc_depressed: float = 0.0
    precision_control: float = 0.0
    recall_control: float = 0.0
    f1_control: float = 0.0
    acc_control: float = 0.0
    auc_roc_macro: float = 0.0
    auc_roc_micro: float = 0.0
    def __str__(self):
        depressed_indices = (self.labels == 1)
        control_indices = ~depressed_indices
        return f"""
        Depressed(1): [{self.num_depressed:03} users] Precision = {self.precision_depressed:.3f}, Recall = {self.recall_depressed:.3f}, F1 = {self.f1_depressed:.3f}, Acc = {self.acc_depressed:.3f} AUC_ROC: {self.auc_roc_macro:.3f}
        True Labels: {np.array2string(self.labels[depressed_indices], separator = '')}
        Predictions: {np.array2string(self.predictions[depressed_indices], separator = '')}
        Healthy (0): [{self.num_control:03} users] Precision = {self.precision_control:.3f}, Recall = {self.recall_control:.3f}, F1 = {self.f1_control:.3f}, Acc = {self.acc_control:.3f} AUC_ROC: {self.auc_roc_macro:.3f}
        True Labels: {np.array2string(self.labels[control_indices], separator = '')}
        Predictions: {np.array2string(self.predictions[control_indices], separator = '')}\n"""
@dataclass
class TrainingArguments:
    train_test_split: float = field(default=0.3, metadata={"help": "Train/ tset split"})
    learning_rate: float = field(default=0.001, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0005, metadata={"help": "Weight decay for AdamW if we apply some."})
    train_batch_size: int = field(
        default=32, metadata={"help": "Batch size for training."}
    )
    test_batch_size: int = field(
        default=32, metadata={"help": "Batch size for evaluation."}
    )
    num_train_epochs: int = field(
        default=50, metadata={"help": "Total number of training epochs to perform."}
    )

@dataclass
class GraphType:
    STATIC_BY_USER: str = 'STATIC_BY_USER'
    STATIC_USER_ONLY: str = 'STATIC_USER_ONLY'
    DYNAMIC_BY_USER: str = 'DYNAMIC_BY_USER'

@dataclass
class HeteroGNNArguments:
    gnn_hidden_channel: int = 64
    gnn_out_channel: int = 64
    gnn_num_layer: int = 4
    gnn_num_head: int = 1
    dropout_prob: float = 0.5
    hetero_relations: List[str] = field(default_factory=lambda: ['mention', 'reply', 'quote'])

@dataclass
class MentalNetArguments:
    conv1_channel: int = 32
    conv2_channel: int = 64
    kernel_size: int = 5
    k: int = 15
    dropout_prob: float = 0.5
    heterognn: HeteroGNNArguments = HeteroGNNArguments()
    def out_dim(self):
        conv_out_dim = int((self.k - 2) / 2 + 1)
        conv_out_dim = (conv_out_dim - self.kernel_size + 1) * self.conv2_channel
        return conv_out_dim
@dataclass
class TemporalAttentionArguments:
    num_layers: int = 1
    num_attention_heads: int = 2 # number _heads
    hidden_size: int = 64
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12

@dataclass
class RNNArguments:
    hidden_size: int = 128
    num_layers: int = 1

@dataclass
class ModelArguments:
    input_size: int = field(default = 768, metadata={"help": ""})
    graph_type: str = field(default=GraphType.DYNAMIC_BY_USER, metadata={"help": ""})
    structural_num_heads: List[int] = field(default_factory=lambda: [8, 8], metadata={"help": "# attention heads in each GAT layer"})
    structural_out_channels: List[int] = field(default_factory=lambda: [768], metadata={"help": "# units in each GAT layer"})
    temporal_num_heads: List[int] = field(default_factory=lambda: [8, 8], metadata={"help": "# attention heads in temporal attention layer"})
    structural_attention_dropout_prob: float = field(default=0.1, metadata={"help": ""})
    temporal_attention_dropout_prob: float = field(default=0.5, metadata={"help": ""})
    mlp_num_layers: int = field(default=3, metadata={"help": ""})
    mlp_out_channels: int = field(default=2, metadata={"help": ""})
    mlp_hidden_channels: int = field(default=128, metadata={"help": ""} )
    gnn_num_layers: int = field(default=3, metadata={"help": ""})
    gnn_hidden_channels: int = field(default=64, metadata={"help": ""})
    gnn_out_channels: int = field(default = 1, metadata={"help": ""})
    gnn_dropout_prob: float = field(default = 0.5, metadata={"help": ""})
    mental_net: MentalNetArguments  = MentalNetArguments()
    temporal_attention_config: TemporalAttentionArguments = TemporalAttentionArguments()
    rnn_config: RNNArguments = RNNArguments()

@dataclass
class DatasetInfo:
    tweet_processing_model_name: str = field(default = '', metadata={"help": ""})
    num_features: List[int] = field(default = 768, metadata = {"help": "# dims in each tweet embe"})
    dataset_location: str = field(default = '', metadata={"help": ""})
    dataset_name: str = field(default = 'default', metadata={"help": ""})
    num_tweets_per_period: int = field(default = 20, metadata={"help": "The number of tweets per period per user."})
    max_num_friends: int = field(default = 50, metadata={"help": "The maximum number of friends per period."})
    periods_in_months: int = field(default = 3, metadata={"help": "The window of a graph snapshot."})
    period_length: int = field(default = 6, metadata={"help": "The window of a graph snapshot."})
    max_period_length: int = field(default = 10, metadata={"help": "The window of a graph snapshot."})
    random_state: int = field(default = 42, metadata={"help": "Random State"})
