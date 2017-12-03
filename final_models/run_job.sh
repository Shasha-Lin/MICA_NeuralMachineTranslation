#!/bin/bash
#
#SBATCH --job-name=mica_trials
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --output=slurm-%A.out
#SBATCH --error=slurm-%A.err

# All the load commands below may not be required by everyone.
# The commands below can be modified/changed/removed  based on the python version you have
module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9
module load nltk/python3.5/3.2.4 # It is possible that you need to download nltk_data first on prince /home/ directory.

model_type=$1
learning_rate=$2
embed_size=$3
hidden_size=$4
kmax=$5
n_layers=$6
code_dir=$7
attention=$8
experiment=$9
python3 -m pip install comet_ml --user

python3 -u  $code_dir --USE_CUDA --learning_rate $learning_rate --model_type "$model_type" --attention "$attention" --embed_size $embed_size --hidden_size $hidden_size  --kmax $kmax --experiment $experiment

