#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --job-name=design_sequences

CSV_FILE=$1
OUTPUT_DIR=$2
METHOD=$3
NUM_SAMPLES=$4
TEMPERATURE=${5:-}
NA_MPNN_MODEL_PATH=${6:-}

# 1) sanity check
if [[ ! -f "$CSV_FILE" ]]; then
    echo "CSV file '$CSV_FILE' not found!" >&2
    exit 1
fi

# 2) read all structure_path values via Python csv.DictReader
mapfile -t STRUCTURE_PATHS < <(
    python - "$CSV_FILE" <<'PYCODE'
import sys, pandas as pd

df = pd.read_csv(sys.argv[1])

for p in df['structure_path']:
    print(p)
PYCODE
)

total=${#STRUCTURE_PATHS[@]}
if (( total == 0 )); then
    echo "No data rows found in CSV." >&2
    exit 1
fi

# 3) compute chunking based on SLURM array
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
CHUNK_SIZE=$(( (total + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))
(( END_IDX >= total )) && END_IDX=$(( total - 1 ))

# 4) process this shard
for (( idx=START_IDX; idx<=END_IDX; idx++ )); do
    structure_path=${STRUCTURE_PATHS[idx]}

    cmd=(
        python /home/akubaney/projects/na_mpnn/evaluation/na_eval_utils.py
        --function_name "design_nucleic_acid_sequence"
        --structure_path "$structure_path"
        --overall_output_directory "$OUTPUT_DIR"
        --num_samples "$NUM_SAMPLES"
        --method "$METHOD"
    )

    if [[ -n "$TEMPERATURE" ]]; then
        cmd+=(--temperature "$TEMPERATURE")
    fi

    if [[ -n "$NA_MPNN_MODEL_PATH" ]]; then
        cmd+=(--na_mpnn_model_path "$NA_MPNN_MODEL_PATH")
    fi

    # Execute the command
    "${cmd[@]}"
done
