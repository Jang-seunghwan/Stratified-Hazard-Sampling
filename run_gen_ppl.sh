#!/usr/bin/env bash
# Table 1 (3B column): Gen PPL / Entropy of Standard (ancestral) and SHS samples,
# seed 0, 1000 samples per run, NFE 4-128. One json per (mode, NFE, seed) in $OUTPUT_DIR.
#
#   bash run_gen_ppl.sh
#   MODES="ancestral adaptive shs" NFES="4 8" bash run_gen_ppl.sh
#
# Extra arguments are passed to eval_gen_ppl.py. Existing result files are skipped.
set -euo pipefail

PYTHON=${PYTHON:-python}
MODES=${MODES:-"ancestral shs"}
NFES=${NFES:-"4 8 16 32 64 128"}
SEEDS=${SEEDS:-"0"}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/gen_ppl}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for mode in $MODES; do
  for nfe in $NFES; do
    for seed in $SEEDS; do
      out="$OUTPUT_DIR/${mode}_nfe${nfe}_seed${seed}.json"
      if [ -f "$out" ]; then
        echo "[skip] $out exists"
        continue
      fi
      "$PYTHON" "$SCRIPT_DIR/eval_gen_ppl.py" \
        --mode "$mode" --nfe "$nfe" --seed "$seed" \
        --num_samples "$NUM_SAMPLES" --output_dir "$OUTPUT_DIR" "$@"
    done
  done
done
