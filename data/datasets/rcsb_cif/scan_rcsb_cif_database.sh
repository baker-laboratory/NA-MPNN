#!/bin/bash
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32g
#SBATCH --time=05:00:00
#SBATCH --output=tmp.out
#SBATCH --array=0-1999

number_of_array_tasks=$(($SLURM_ARRAY_TASK_MAX + 1))
number_of_pdbs=$(find /databases/rcsb/cif -type f | wc -l)
pdbs_per_array_task=$((($number_of_pdbs + $number_of_array_tasks - 1) / $number_of_array_tasks))

start_idx=$(($SLURM_ARRAY_TASK_ID * $pdbs_per_array_task))
end_idx=$(($start_idx + $pdbs_per_array_task))

log_file_path="./pdb_content/"$start_idx"_"$end_idx".log"

apptainer exec /software/containers/users/akubaney/mpnn_train.sif python ./scan_rcsb_cif_database.py $start_idx $end_idx > $log_file_path
