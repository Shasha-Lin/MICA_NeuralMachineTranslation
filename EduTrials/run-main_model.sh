#!/bin/bash
#
#SBATCH --job-name=mica_edu2
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --mem=10GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err

module purge
module load python3/intel/3.5.3

python3 -m pip install comet_ml --user

python3 -u main_model.py  --n_iters 200000 --learning_rate 0.001  --main_data_dir "/scratch/eff254/NLP/Data/Model_ready/" --out_dir "/scratch/eff254/NLP/ModelOutputs/"
