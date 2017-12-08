#!/bin/bash
#
#SBATCH --job-name=mica_trials_continued
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --output=cont_results/slurm-%A.out
#SBATCH --error=cont_results/slurm-%A.err

# All the load commands below may not be required by everyone.
# The commands below can be modified/changed/removed  based on the python version you have
module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9
module load nltk/python3.5/3.2.4 # It is possible that you need to download nltk_data first on prince /home/ directory.

model_type=$1
learning_rate=$2
hidden_size=$3
kmax=$4
n_layers=$5
code_dir=$6
attention=$7
experiment=$8
enc_cp=$9
dec_cp=${10}
epochs=${11}
enc_opt=${12}
dec_opt=${13}



python3 -m pip install comet_ml --user

python3 -u  $code_dir --print_every 1 --batch_size 32 --USE_CUDA --learning_rate $learning_rate --model_type "$model_type" --attention "$attention" --hidden_size $hidden_size --kmax $kmax --experiment $experiment --checkpoint_enc "$enc_cp" --checkpoint_dec "$dec_cp" --epoch_continue $epochs --checkpoint_enc_optim $enc_opt --checkpoint_dec_optim $dec_opt

