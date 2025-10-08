#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --job-name=score_native_specificities
#SBATCH --time=00:10:00

CSV_FILE=$1
SPECIFIED_DIRECTORY=$2
OUTPUT_DIR=$3

# 1) sanity checks
if [[ ! -f "$CSV_FILE" ]]; then
    echo "CSV file '$CSV_FILE' not found!" >&2
    exit 1
fi
if [[ ! -d "$SPECIFIED_DIRECTORY" ]]; then
    echo "Directory '$SPECIFIED_DIRECTORY' not found!" >&2
    exit 1
fi
if ! command -v jq &>/dev/null; then
    echo "Error: jq required but not on PATH." >&2
    exit 1
fi

declare -A structure_path_to_ppm_paths
while IFS=$'\t' read -r structure_path ppm_paths; do
    structure_path_to_ppm_paths["$structure_path"]="$ppm_paths"
done < <(
    python - "$CSV_FILE" <<'PYCODE'
import sys
import pandas as pd

# read the CSV into a DataFrame
df = pd.read_csv(sys.argv[1])

# iterate over the two columns and print them tab‑separated
for structure_path, ppm_paths in zip(df['structure_path'], df['ppm_paths']):
    print(f"{structure_path}\t{ppm_paths}")
PYCODE
)

# Count total lines in the CSV (including the header)
TOTAL_LINES=$(awk 'END{print NR}' "$CSV_FILE")
if (( TOTAL_LINES < 2 )); then
    echo "CSV file does not contain any data rows."
    exit 1
fi

# Get the list of JSON files.
shopt -s nullglob
json_files=( "$SPECIFIED_DIRECTORY"/*/specificity_json/*.json )
total_json=${#json_files[@]}
if (( total_json == 0 )); then
    echo "No JSON files found under $SPECIFIED_DIRECTORY/*/specificity_json/." >&2
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
    
    structure_path=$(jq -r '.original_input_structure_path' "$json_file")
    [[ -z "$structure_path" || "$structure_path" == "null" ]] && {
        echo "Skipping $json_file: no original_input_structure_path" >&2
        continue
    }
    
    ppm_paths=${structure_path_to_ppm_paths["$structure_path"]}
    [[ -z "$ppm_paths" ]] && \
        echo "Warning: no ppm_paths for $structure_path" >&2

    python /home/akubaney/projects/na_mpnn/evaluation/na_eval_utils.py \
        --function_name "score_specificity_prediction" \
        --reference_ppms_list_str "$ppm_paths" \
        --subject_path "$json_file" \
        --overall_output_directory "$OUTPUT_DIR"
done