#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 0-02:00:00

input_csv_path=$1
output_directory=$2
modulo=$((SLURM_ARRAY_TASK_MAX + 1))
remainder=$SLURM_ARRAY_TASK_ID

apptainer exec /software/containers/users/akubaney/mpnn.sif python ./preprocess_dataset.py $input_csv_path $output_directory $modulo $remainder