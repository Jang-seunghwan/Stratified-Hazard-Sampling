#!/bin/bash
# Paper grid for GIDD (Table 1): modes x NFEs x seeds, 1024 samples per run.
# Usage:
#   bash scripts/run_gidd_eval.sh                 # both modes (default, shs)
#   bash scripts/run_gidd_eval.sh shs             # one mode
# Optional env vars: PYTHON, OUTPUT_DIR, NFES, SEEDS, NUM_SAMPLES, BATCH_SIZE
# Each run writes ${OUTPUT_DIR}/<mode>/seed<S>_nfe<N>.json; existing runs are skipped.
set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
NFES="${NFES:-4 8 16 32 64 128 256 512}"
SEEDS="${SEEDS:-1 2 3 4 5}"
NUM_SAMPLES="${NUM_SAMPLES:-1024}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MODES="${*:-default shs}"

for MODE in ${MODES}; do
  for NFE in ${NFES}; do
    for SEED in ${SEEDS}; do
      JSON="${OUTPUT_DIR}/${MODE}/seed${SEED}_nfe${NFE}.json"
      if [ -f "${JSON}" ]; then
        echo ">>> SKIP (exists): ${JSON}"
        continue
      fi
      echo ">>> mode=${MODE} NFE=${NFE} seed=${SEED} [$(date)]"
      "${PYTHON}" -u eval_gen_ppl.py \
        --mode "${MODE}" \
        --seeds "${SEED}" \
        --nfes "${NFE}" \
        --num_samples "${NUM_SAMPLES}" \
        --batch_size "${BATCH_SIZE}" \
        --output_dir "${OUTPUT_DIR}"
    done
  done
done
