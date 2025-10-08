#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

fasta_splits_directory=$1
family_label_output_directory=$2

# Fasta path.
fasta_path=$fasta_splits_directory"/all_protein_sequences_"$SLURM_ARRAY_TASK_ID".fa"

# Output path.
output_path=$family_label_output_directory"/family_label_"$SLURM_ARRAY_TASK_ID".csv"

# Run InterProScan on the fasta.
/home/akubaney/software/interproscan/interproscan.sh -i $fasta_path -f tsv -o $output_path -appl Pfam
