#!/bin/bash
# ============================================================
# SHS on DFM / FS-DFM: 4-way evaluation
#   1) DFM  + Standard
#   2) DFM  + SHS
#   3) FS-DFM + Standard
#   4) FS-DFM + SHS
# ============================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

# --- Paths ---
PROJ_DIR="/data1/seunghwan/fs-dfm"
CONDA_ENV="seunghwan_fs-dfm"
DFM_CKPT="${PROJ_DIR}/checkpoints/DFM_checkpoint.pth"
FSDFM_CKPT="${PROJ_DIR}/checkpoints/FSDFM_checkpoint.pth"
RESULT_BASE="${PROJ_DIR}/results"

# --- Eval settings ---
NGPUS=1
N_SAMPLES=320          # perplexity samples (원제님 세팅과 동일)
SAMPLING_STEPS=1024    # 전체 스텝 (내부에서 2^i 순회)
SEED=42

# ============================================================
# 1) DFM + Standard (teacher model, baseline solver)
# ============================================================
echo "=== [1/4] DFM + Standard ==="
conda run -n ${CONDA_ENV} --no-capture-output \
  python ${PROJ_DIR}/fs_dfm/run_eval.py \
    --work_dir "${RESULT_BASE}/dfm_standard" \
    --pre_trained_model_path "${DFM_CKPT}" \
    --ngpus ${NGPUS} \
    --perplexity_n_samples ${N_SAMPLES} \
    --sampling_steps ${SAMPLING_STEPS} \
    --seed ${SEED} \
    --eval_perplexity \
    --teacher_model \
  2>&1 | tee "${RESULT_BASE}/dfm_standard.log"

# ============================================================
# 2) DFM + SHS (teacher model, SHS solver)
# ============================================================
echo "=== [2/4] DFM + SHS ==="
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

# ============================================================
# 3) FS-DFM + Standard (student model, baseline solver)
# ============================================================
echo "=== [3/4] FS-DFM + Standard ==="
conda run -n ${CONDA_ENV} --no-capture-output \
  python ${PROJ_DIR}/fs_dfm/run_eval.py \
    --work_dir "${RESULT_BASE}/fsdfm_standard" \
    --pre_trained_model_path "${FSDFM_CKPT}" \
    --ngpus ${NGPUS} \
    --perplexity_n_samples ${N_SAMPLES} \
    --sampling_steps ${SAMPLING_STEPS} \
    --seed ${SEED} \
    --eval_perplexity \
  2>&1 | tee "${RESULT_BASE}/fsdfm_standard.log"

# ============================================================
# 4) FS-DFM + SHS (student model, SHS solver)
# ============================================================
echo "=== [4/4] FS-DFM + SHS ==="
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

echo "=== All 4 evaluations complete ==="
echo "Results in: ${RESULT_BASE}/"
