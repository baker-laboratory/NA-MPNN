#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a4000:1
#SBATCH --mem=32g
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --job-name=process_designs

SPECIFIED_DIRECTORY=$1
OUTPUT_DIR=$2

# 1) sanity checks
if [[ ! -d "$SPECIFIED_DIRECTORY" ]]; then
    echo "Directory '$SPECIFIED_DIRECTORY' not found!" >&2
    exit 1
fi
if ! command -v jq &>/dev/null; then
    echo "Error: jq required but not on PATH." >&2
    exit 1
fi

# Get the list of JSON files.
shopt -s nullglob
json_files=( "$SPECIFIED_DIRECTORY"/*/design_json/*.json )
total_json=${#json_files[@]}
if (( total_json == 0 )); then
    echo "No JSON files found under $SPECIFIED_DIRECTORY/*/design_json/." >&2
    exit 1
fi

# Number of data rows (excluding the header)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total_json + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total_json )) && END_IDX=$(( total_json - 1 ))

# Process the assigned chunk
for idx in $(seq "$START_IDX" "$END_IDX"); do
    json_file=${json_files[idx]}

    python /home/akubaney/projects/na_mpnn/evaluation/na_eval_utils.py \
        --function_name "process_design_monomer_rna" \
        --subject_path "$json_file" \
        --overall_output_directory "$OUTPUT_DIR"
done