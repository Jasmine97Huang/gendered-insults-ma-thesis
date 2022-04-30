#!/bin/bash
#SBATCH --job-name=bert_viz
#SBATCH --output=viz_sbatch.out
#SBATCH --error=viz_sbatch.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=15 # num cores to drive each gpu
#SBATCH --cpus-per-task=3   # set this to the desired number of threads
#SBATCH --gres=gpu:2
#SBATCH --account=macs30123

module load python
module load cuda

python ./viz.py /data/bad-words.txt "bitch" /data/bitch_df.csv ./lyrics-bert2
