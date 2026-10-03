#!/bin/bash
# Gen PPL / entropy of UDLM (kuleshov-group/udlm-lm1b, LM1B, 128 tokens)
# with the Standard (`default`) and SHS (`shs`) samplers.
# The defaults reproduce the paper grid:
#   2 samplers x 9 NFEs x 5 seeds, 1024 samples (32 batches x 32) per run.
#
# Usage (from any directory, with the `discdiff` env activated):
#   bash scripts/eval_udlm_gen_ppl.sh
#   MODES="shs" NFES="128" SEEDS="1" bash scripts/eval_udlm_gen_ppl.sh
#   SAVE_JUMP_STATS=True bash scripts/eval_udlm_gen_ppl.sh
# Extra arguments are passed to `main.py` as Hydra overrides.
#
# Environment variables (default):
#   MODES               ("default shs")
#   NFES                ("4 8 16 32 64 128 256 512 1024")
#   SEEDS               ("1 2 3 4 5")
#   NUM_SAMPLE_BATCHES  (32)
#   BATCH_SIZE          (32)
#   SAVE_JUMP_STATS     (False)  also write <json name>_jump_stats.npz
#   OUTPUT_DIR          (outputs/lm1b/udlm-hf)
# One json per run: ${OUTPUT_DIR}/samples-lm1b-mode-<mode>_T-<nfe>_seed-<seed>.json
# Runs whose json already exists are skipped.

set -e
cd "$(dirname "$0")/.." || exit
export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

MODES="${MODES:-default shs}"
NFES="${NFES:-4 8 16 32 64 128 256 512 1024}"
SEEDS="${SEEDS:-1 2 3 4 5}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_JUMP_STATS="${SAVE_JUMP_STATS:-False}"
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/outputs/lm1b/udlm-hf}"
mkdir -p "${OUTPUT_DIR}"

for MODE in ${MODES}; do
  for NFE in ${NFES}; do
    for SEED in ${SEEDS}; do
      JSON="${OUTPUT_DIR}/samples-lm1b-mode-${MODE}_T-${NFE}_seed-${SEED}.json"
      if [ -f "${JSON}" ]; then
        echo ">>> Skip (exists): ${JSON}"
        continue
      fi
      echo ">>> mode=${MODE} NFE=${NFE} seed=${SEED}"
      python -u -m main \
        hydra.output_subdir=null \
        hydra.run.dir="${PWD}" \
        hydra/job_logging=disabled \
        hydra/hydra_logging=disabled \
        mode=gen_ppl_eval \
        seed="${SEED}" \
        data=lm1b \
        backbone=hf_dit \
        model=hf \
        model.pretrained_model_name_or_path=kuleshov-group/udlm-lm1b \
        model.length=128 \
        training.guidance=null \
        parameterization=d3pm \
        diffusion=uniform \
        time_conditioning=True \
        T=0 \
        sampling.num_sample_batches="${NUM_SAMPLE_BATCHES}" \
        sampling.batch_size="${BATCH_SIZE}" \
        sampling.steps="${NFE}" \
        sampling.use_cache=False \
        sampling.use_float64=False \
        sampling.sampling_mode="${MODE}" \
        sampling.save_jump_stats="${SAVE_JUMP_STATS}" \
        eval.generated_samples_path="${JSON}" \
        +eval.generative_ppl_model_name_or_path=gpt2-large \
        "$@"
    done
  done
done
