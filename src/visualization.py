import argparse
import pathlib
from mental.utils.dataclass import DatasetInfo
from mental.utils.dataclass import BaselineModel
from mental.pipeline.executor import DepressionVisualizer
import yaml
from box import Box

root_dir = pathlib.Path(__file__).parent.parent.resolve()
with open(root_dir / 'src' / 'config.yaml', 'r') as file:
    config = Box(yaml.safe_load(file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model training')
    parser.add_argument('--dataset_name', type=str, default='nov31', help='The name of the dataset.')
    parser.add_argument('--model_name', type=str, default='MentalPlus', help='The name of the model.')
    parser.add_argument('--num_tweets', type=int, default=4 , help='Number of tweets per period')
    parser.add_argument('--num_friends', type=int, default=4, help='Number of friends per period')
    parser.add_argument('--periods_in_months', type=int, default=3, help='The length of each period in months.')
    parser.add_argument('--random_state', type=int, default=5, help='The randomness of the seed.')
    # Add more arguments as needed
    args = parser.parse_args()
    dataset_info = DatasetInfo(
        num_tweets_per_period = str(args.num_tweets),
        max_num_friends = str(args.num_friends),
        periods_in_months = str(args.periods_in_months),
        period_length = str(4),
        dataset_location = root_dir,
        dataset_name = args.dataset_name,
        random_state = args.random_state
    )
    try:
        baseline = getattr(BaselineModel, args.model_name)
    except:
        raise ValueError("Wrong baseline.")
    pe = DepressionVisualizer(config, dataset_info)
    pe.register_model_class(baseline)
    pe.run()