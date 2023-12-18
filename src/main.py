import argparse
import pathlib
from mental.utils.dataclass import DatasetInfo
from mental.utils.dataclass import BaselineModel
from mental.pipeline.executor import PipelineExecutor
import yaml
from box import Box

root_dir = pathlib.Path(__file__).parent.parent.resolve()
with open(root_dir / 'src' / 'config.yaml', 'r') as file:
    config = Box(yaml.safe_load(file))


#dataset_info -> current location
# https://www.kaggle.com/datasets/rrmartin/twitter-depression-tweets-and-musics
#dataset_name = 
dataset_names = ['twitter-roberta-base-2022-154m']#'twitter-roberta-base-emotion']#, ]
#dataset_names = ['nov3']#'twitter-roberta-base-emotion']#, ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model training')
    parser.add_argument('--dataset_name', type=str, default='nov15', help='The name of the dataset.')
    parser.add_argument('--model_name', type=str, default='MentalPlus', help='The name of the model.')
    parser.add_argument('--accelerator', type=str, default='gpu', help='GPU/CPU.')
    parser.add_argument('--num_tweets', type=int, default=4, help='Number of tweets per period')
    parser.add_argument('--num_friends', type=int, default=2, help='Number of friends per period')
    parser.add_argument('--periods_in_months', type=int, default=6, help='The length of each period in months.')
    parser.add_argument('--num_snapshot', type=int, default=10, help='The number of the snapshots.')
    parser.add_argument('--random_state', type=int, default=42, help='The randomness of the seed.')
    # Add more arguments as needed
    args = parser.parse_args()
    dataset_info = DatasetInfo(
        num_tweets_per_period = str(args.num_tweets),
        max_num_friends = str(args.num_friends),
        periods_in_months = str(args.periods_in_months),
        period_length = str(args.num_snapshot),
        dataset_location = root_dir,
        dataset_name = args.dataset_name,
        random_state = args.random_state
    )
    try:
        baseline = getattr(BaselineModel, args.model_name)
    except:
        raise ValueError("Wrong baseline.")
    pe = PipelineExecutor(config, dataset_info)
    pe.change_accelerator(args.accelerator)
    pe.register_model_class(baseline)
    pe.run()



baselines = [
    BaselineModel.GCN,
    #BaselineModel.Sawhney_NAACL_21,
    #BaselineModel.Sawhney_EMNLP_20,
    #BaselineModel.UsmanBiLSTM,
    #BaselineModel.UsmanBiLSTMPlus,
    #BaselineModel.MLP,
    #BaselineModel.UGformer,
    #BaselineModel.MentalPlus_Base,
    #BaselineModel.MentalPlus_HOMO,
    #BaselineModel.MentalPlus_NO_TIMEFRAME_CUTOUT,
    #BaselineModel.MentalPlus_NO_CONTENT_ATTENTION,
    #BaselineModel.MentalPlus_NO_INTERACTION_CUTOUT,
    #BaselineModel.MentalPlus_Without_Transformer,
    BaselineModel.MentalPlus,
    #BaselineModel.MentalPlus_NO_GNN,
    #BaselineModel.MentalPlus_Without_Transformer,
    #BaselineModel.ContrastEgo_Base,
    #BaselineModel.ContrastEgo,
    BaselineModel.GAT,
    BaselineModel.GraphSAGE,
    #BaselineModel.DynamicGCN,
    #BaselineModel.DynamicGAT,
    #BaselineModel.DynamicSAGE,
    #BaselineModel.EvolveGCN,
    #BaselineModel.CNNWithMax,
    #BaselineModel.CNNWithMaxPlus,
    #BaselineModel.DySAT,
    #BaselineModel.DyHAN,
    BaselineModel.MentalNet,
    #BaselineModel.MentalNet_Original,
    #BaselineModel.MentalNetNaive,
    #BaselineModel.MentalNet_SAGE,
    #BaselineModel.MentalNet_GAT,
    #BaselineModel.MentalNet_GAT2,
    #BaselineModel.MentalNetDySAT_SimSiam,
    #BaselineModel.MentalNet_DySAT,
    #BaselineModel.MentalPlus_NO_POSITION,
    #BaselineModel.MentalPlus_NO_POSITION_CLS_POOLING,
    #BaselineModel.MentalPlus_USE_NODE,
    #BaselineModel.MentalPlus_USE_GRAPH,
    #BaselineModel.MentalPlus_SimSiam,
    #BaselineModel.MentalPlus_GCN,
    #BaselineModel.MentalPlus_GAT,
]
