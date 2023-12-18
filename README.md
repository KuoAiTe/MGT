conda create --name env python=3.9
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.3

# Install PyTorch 2.0 (py3.9_cuda11.8_cudnn8.7.0_0)
conda install pytorch torchvision torchaudio -c pytorch
# Install PyTorch Lightning
conda install -c conda-forge pytorch-lightning

conda install pyg -c pyg
pip install torch_geometric
pip install pyg_lib
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html



pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

conda install -c conda-forge pytorch-lightning
conda install -c conda-forge transformers
conda install pandas

pip install -e .

pip install package

# Use
python3 src/main.py --dataset_name {dataset_name} --model_name {method} --num_tweets {num_tweets} --num_friends {default_num_friends_per_period} --periods_in_months {default_periods_in_months} --num_snapshot {default_num_snapshots} --random_state 42 --accelerator {accelerator}