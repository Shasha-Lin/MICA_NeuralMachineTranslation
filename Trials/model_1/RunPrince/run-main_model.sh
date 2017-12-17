#!/bin/bash
#
#SBATCH --job-name=mica_edu_akash
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --mem=40GB
#SBATCH --output=slurm-%A.out
#SBATCH --error=slurm-%A.err

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
# module load torchvision/0.1.8
module load torchvision/python3.5/0.1.9
# module load nltk/python3.6
module load nltk/python3.5/3.2.4 # It is possible that you need to download nltk_data first on prince /home/ directory.

python3 -m pip install comet_ml --user

python3 -u main_model_BS_gpu.py --USE_CUDA --n_epochs 800000 --learning_rate 0.001 --teacher_forcing_ratio 0.9  --model_type "bpe2bpe" --main_data_dir "/scratch/eff254/NLP/Data/Model_ready/" --out_dir "/scratch/ak6201/NLP/ModelOutputs/"
