#!/bin/bash
#
#SBATCH --job-name=mica_edu_akash
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
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
embed_size=$4
hidden_size=$5
kmax=$6
n_layers=$7
code_dir=$8
attention=$9

python3 -m pip install comet_ml --user

python3 -u ../RunPrince/main_model_BS_all.py $code_dir --USE_CUDA --learning_rate $learning_rate --model_type "$model_type" --attention "$attention" --embed_size $embed_size --hidden_size $hidden_size  --kmax $kmax --main_data_dir "/scratch/eff254/NLP/Data/Model_ready/" --out_dir "/scratch/ak6201/NLP/ModelOutputs/"