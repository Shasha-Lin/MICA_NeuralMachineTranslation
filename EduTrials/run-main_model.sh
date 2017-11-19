#!/bin/bash
#
#SBATCH --job-name=mica_edu2
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --mem=40GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err

module purge
module load python3/intel/3.5.3
module load nltk/python3.5/3.2.4 # It is possible that you need to download nltk_data first on prince /home/ directory. 

python3 -m pip install comet_ml --user

python3 -u main_model.py --use_cuda --n_iters 800000 --learning_rate 0.0001 --teacher_forcing_ratio 1  --main_data_dir "/scratch/eff254/NLP/Data/Model_ready/" --out_dir "/scratch/eff254/NLP/ModelOutputs/"

# On local machine:
# python main_model.py --learning_rate 0.01 --main_data_dir "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Model_ready" --MIN_LENGTH_INPUT 5 --MIN_LENGTH_TARGET 5 --MAX_LENGTH_INPUT 10 --MAX_LENGTH_TARGET 10 --out_dir /Users/eduardofierro/Desktop/
