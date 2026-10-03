#!/usr/bin/env bash
# Paper grid for DFM (Tables 1 and 2): {Standard, SHS} x NFE {4, 8, 16, 32, 64, 128}
# x seeds {1, 2, 3}, 1024 samples per run, per-step jump probability p = min(h*lambda, 1).
#
# Usage:
#   CKPT=/path/to/checkpoint.pth bash scripts/run_dfm_eval.sh [OUT_DIR]
#
# Optional environment variables: PYTHON, CACHE_DIR, NFES, SEEDS, SAMPLERS,
# N_SAMPLES, BATCH_SIZE. Select the GPU with CUDA_VISIBLE_DEVICES if needed.
# Each run writes <run>.json (metrics) and <run>.txt (samples) to OUT_DIR/<run>/;
# finished runs are skipped. A mean/std table over seeds is printed at the end.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${CKPT:-$ROOT/checkpoints/checkpoint.pth}"
OUT="${1:-$ROOT/results/dfm}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache_dir}"
PYTHON="${PYTHON:-python}"
NFES="${NFES:-4 8 16 32 64 128}"
SEEDS="${SEEDS:-1 2 3}"
SAMPLERS="${SAMPLERS:-standard shs}"
N_SAMPLES="${N_SAMPLES:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

for nfe in $NFES; do
  for seed in $SEEDS; do
    for sampler in $SAMPLERS; do
      shs_flag=""
      [ "$sampler" = "shs" ] && shs_flag="--use-shs"
      run="dfm_${sampler}_dtmc_nfe${nfe}_seed${seed}"
      wd="$OUT/$run"
      if [ -f "$wd/$run.json" ]; then
        echo "[skip] $run"
        continue
      fi
      "$PYTHON" "$ROOT/fs_dfm/run_eval.py" \
        --work_dir "$wd" \
        --pre_trained_model_path "$CKPT" \
        --cache_dir "$CACHE_DIR" \
        --teacher_model --dtmc $shs_flag \
        --eval_perplexity \
        --sampling_steps "$nfe" \
        --seed "$seed" \
        --perplexity_n_samples "$N_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --ngpus 1
    done
  done
done

"$PYTHON" - "$OUT" <<'EOF'
import glob, json, os, sys
from collections import defaultdict

import numpy as np

cells = defaultdict(list)
for path in glob.glob(os.path.join(sys.argv[1], "*", "*.json")):
    r = json.load(open(path))
    cells[(r["nfe"], r["sampler"])].append((r["gen_ppl"], r["entropy"]))

print(f"{'NFE':>5} {'sampler':>9} {'seeds':>5} {'Gen PPL':>18} {'Entropy':>16}")
for (nfe, sampler), v in sorted(cells.items()):
    ppl, ent = np.array(v).T
    print(f"{nfe:>5} {sampler:>9} {len(v):>5} {ppl.mean():>9.2f} ± {ppl.std():<6.2f} "
          f"{ent.mean():>7.3f} ± {ent.std():<6.3f}")
EOF
