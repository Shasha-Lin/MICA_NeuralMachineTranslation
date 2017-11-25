#!/bin/bash
#
#SBATCH --job-name=Mica
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --mem=40GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err

module purge
module load python3/intel/3.5.3
module load nltk/python3.5/3.2.4 # It is possible that you need to download nltk_data first on prince /home/ directory. 

python3 -m pip install comet_ml --user

python3 -u main_model_BS.py --use_cuda --n_iters 2000000 --learning_rate 0.001 --teacher_forcing_ratio 0.9  --main_data_dir "/scratch/eff254/NLP/Data/Model_ready/" --out_dir "/scratch/eff254/NLP/ModelOutputs/"