import subprocess
import concurrent.futures
from datetime import datetime

methods = [
    #"GCN",
    #"GAT",
    #"GraphSAGE",
    #"UsmanBiLSTM",
    #"UsmanBiLSTMPlus",
    #"Sawhney_EMNLP_20",
    #"Sawhney_NAACL_21",
    #"DySAT",
    #"DyHAN",
    #"MentalNet",
    #BaselineModel.,
    #BaselineModel.,
    #BaselineModel.,
    #BaselineModel.,
    #"CNNWithMax",
    #"ContrastEgo",
    "MentalPlus",
    #"MentalPlus_SUP",
    #"MentalPlus_Base",
    #"MentalPlus_BatchNorm",
    #"MentalPlus_HOMO",
    #"MentalPlus_NO_GNN",
    #"MentalPlus_Without_Transformer",
    #"MentalPlus_NO_CONTENT_ATTENTION",
    #"MentalPlus_NO_TIMEFRAME_CUTOUT",
    #"MentalPlus_NO_INTERACTION_CUTOUT",
]
dataset_name = "nov30"
num_tweets_per_period_list = [2, 4, 6]
default_num_tweets_per_period = 4
num_friends_per_period_list = [2, 4, 6]
periods_in_months_list = [3]
default_num_friends_per_period = 4
default_periods_in_months= 3
num_snapshots = [8, 4, 1, -1, -2, -4]
default_num_snapshots = 4
random_state = 42
accelerator = "cpu"
MAX_WORKERS = 8
stdout = subprocess.DEVNULL
stdout = subprocess.DEVNULL
# Create a list to store subprocess objects
processes = []

commands = []


for num_tweets in num_tweets_per_period_list:
    for method in methods:
        commands.append(f"python3 src/main.py --dataset_name {dataset_name} --model_name {method} --num_tweets {num_tweets} --num_friends {default_num_friends_per_period} --periods_in_months {default_periods_in_months} --num_snapshot {default_num_snapshots} --random_state {random_state} --accelerator {accelerator}")

for num_friends in num_friends_per_period_list:
    for method in methods:
        commands.append(f"python3 src/main.py --dataset_name {dataset_name} --model_name {method} --num_tweets {default_num_tweets_per_period} --num_friends {num_friends} --periods_in_months {default_periods_in_months} --num_snapshot {default_num_snapshots} --random_state {random_state} --accelerator {accelerator}")

for num_snapshot in num_snapshots:
    for method in methods:
        commands.append(f"python3 src/main.py --dataset_name {dataset_name} --model_name {method} --num_tweets {default_num_tweets_per_period} --num_friends {default_num_friends_per_period} --periods_in_months {default_periods_in_months} --num_snapshot {num_snapshot} --random_state {random_state} --accelerator {accelerator}")
for _ in commands:
    print(_)
commands = commands * 100
counter = 0
# Function to execute a command and return its exit code
def run_command(command):
    global counter
    counter += 1
    print(f"{datetime.now()} | Run #{counter}: {command}")
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True)
    return result.returncode

while True:
    # Create a ThreadPoolExecutor with a specified number of threads (adjust as needed)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each command for execution in parallel
        future_to_command = {executor.submit(run_command, cmd): cmd for cmd in commands}

        # Wait for all commands to complete and collect their exit codes
        exit_codes = [future.result() for future in concurrent.futures.as_completed(future_to_command)]

    # Process exit codes or perform other actions as needed
    for cmd, exit_code in zip(commands, exit_codes):
        print(f"{datetime.now()} | Command '{cmd}' finished with exit code {exit_code}")


exit()
for command in commands_to_run:
    send_command_to_tmux(method, 0, command)
exit()
all_tmux_sessions_finished()
exit()
destroy_tmux_session(method)


   
    