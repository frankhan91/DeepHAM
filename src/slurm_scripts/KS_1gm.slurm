#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem-per-cpu=60G
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00 # 6 hours
#SBATCH -o KS50agt%j.out
#SBATCH -e KS50agt%j.err

export PYTHONUNBUFFERED=TRUE
python train_KS.py -c ./configs/KS/game_nn_n50_0fm1gm.json -n 1gm
