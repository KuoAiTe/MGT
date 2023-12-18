methods=(
    "GCN"
    "GAT"
    "UsmanBiLSTM"
    "UsmanBiLSTMPlus"
    "Sawhney_EMNLP_20"
    "Sawhney_NAACL_21"
    "DySAT"
    "DyHAN"
    #BaselineModel.,
    #BaselineModel.,
    #BaselineModel.,
    #BaselineModel.,
    "MentalPlus"
    "MentalNet"
)
dataset_name="nov9"
num_tweets_per_period_list=(6 4 2)
num_friends_per_period_list=(2 4 6)
periods_in_months_list=(3 6 12)
random_state=42
for num_tweets in "${num_tweets_per_period_list[@]}"; do
    for num_friends in "${num_friends_per_period_list[@]}"; do
        for periods_in_months in "${periods_in_months_list[@]}"; do
            for method in "${methods[@]}"; do
                    command="python3 test.py --dataset_name $dataset_name --model_name $method --num_tweets $num_tweets --num_friends $num_friends --periods_in_months $periods_in_months --random_state $random_state"
                    echo $command
                    $command
            done
        done
    done
done