#!/bin/bash
# GPU 3: SHS sampler (DFM + FS-DFM)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3

PROJ_DIR="/data1/seunghwan/fs-dfm"
CONDA_ENV="seunghwan_fs-dfm"
DFM_CKPT="${PROJ_DIR}/checkpoints/DFM_checkpoint.pth"
FSDFM_CKPT="${PROJ_DIR}/checkpoints/FSDFM_checkpoint.pth"
RESULT_BASE="${PROJ_DIR}/results"
NGPUS=1
N_SAMPLES=320
SAMPLING_STEPS=1024
SEED=42

echo "=== [GPU3] DFM + SHS ==="
conda run -n ${CONDA_ENV} --no-capture-output \
  python ${PROJ_DIR}/fs_dfm/run_eval.py \
    --work_dir "${RESULT_BASE}/dfm_shs" \
    --pre_trained_model_path "${DFM_CKPT}" \
    --ngpus ${NGPUS} \
    --perplexity_n_samples ${N_SAMPLES} \
    --sampling_steps ${SAMPLING_STEPS} \
    --seed ${SEED} \
    --eval_perplexity \
    --teacher_model \
    --use-shs \
  2>&1 | tee "${RESULT_BASE}/dfm_shs.log"

echo "=== [GPU3] FS-DFM + SHS ==="
conda run -n ${CONDA_ENV} --no-capture-output \
  python ${PROJ_DIR}/fs_dfm/run_eval.py \
    --work_dir "${RESULT_BASE}/fsdfm_shs" \
    --pre_trained_model_path "${FSDFM_CKPT}" \
    --ngpus ${NGPUS} \
    --perplexity_n_samples ${N_SAMPLES} \
    --sampling_steps ${SAMPLING_STEPS} \
    --seed ${SEED} \
    --eval_perplexity \
    --use-shs \
  2>&1 | tee "${RESULT_BASE}/fsdfm_shs.log"

echo "=== [GPU3] All SHS done ==="
