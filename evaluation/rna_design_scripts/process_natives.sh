#!/bin/bash

CSV_FILE=$1
PROCESSED_REF_DIR=$2

# 1) Load the "pdb_path" column from the CSV into a Bash array
mapfile -t PDB_PATHS < <(
    python - "$CSV_FILE" <<'PYCODE'
import sys, pandas as pd

df = pd.read_csv(sys.argv[1])

for p in df['structure_path']:
    print(p)
PYCODE
)

# 2) Loop over each path and invoke the Python processor
for pdb_path in "${PDB_PATHS[@]}"; do
    echo $pdb_path
    python /home/akubaney/projects/na_mpnn/evaluation/na_eval_utils.py \
        --function_name process_reference_monomer_rna \
        --reference_structure_path "$pdb_path" \
        --overall_output_directory "$PROCESSED_REF_DIR"
done