#!/bin/bash
#SBATCH -p gpu-train
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH -t 2-00:00:00

json_path=$1

apptainer exec --nv /software/containers/users/akubaney/mpnn.sif python ./na_run.py $json_path

