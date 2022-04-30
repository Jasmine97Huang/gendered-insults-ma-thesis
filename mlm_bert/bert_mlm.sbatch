#!/bin/bash
#SBATCH --job-name=bert_mlm
#SBATCH --output=sbatch.out
#SBATCH --error=sbatch.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=20 # num cores to drive each gpu
#SBATCH --cpus-per-task=4   # set this to the desired number of threads
#SBATCH --gres=gpu:4
#SBATCH --account=soci40133

module load python
module load cuda

python run ./main.py data/sp_bb_metro_matched.csv
python run  ./train.py output/inputs.npy 2
