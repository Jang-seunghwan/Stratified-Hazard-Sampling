#!/bin/bash
#SBATCH -o ../watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 96:00:00                    # Time limit (hh:mm:ss)
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

<<comment
#  Usage:
cd scripts/

# Basic usage (default SHS sampling):
SAMPLING_STEPS=128 bash eval_lm1b_gen_ppl_hf.sh

# With specific sampling mode (default/shs/shs_safe):
SAMPLING_MODE=shs SAMPLING_STEPS=128 bash eval_lm1b_gen_ppl_hf.sh

# With random blacklist (safe-word generation research):
SAMPLING_MODE=shs_safe BLACKLIST_PERCENT=10 BLACKLIST_SEED=42 bash eval_lm1b_gen_ppl_hf.sh

# With specific banned words (comma-separated):
SAMPLING_MODE=shs_safe BLACKLIST_WORDS="hate,kill,violence" bash eval_lm1b_gen_ppl_hf.sh

# Combined: random blacklist + specific words:
SAMPLING_MODE=shs_safe BLACKLIST_PERCENT=5 BLACKLIST_WORDS="hate,kill" bash eval_lm1b_gen_ppl_hf.sh

# Full example:
SAMPLING_MODE=shs_safe BLACKLIST_PERCENT=10 BLACKLIST_SEED=42 SAMPLING_STEPS=128 SEED=1 bash eval_lm1b_gen_ppl_hf.sh

# Or with sbatch:
sbatch \
  --export=ALL,SAMPLING_MODE=shs_safe,BLACKLIST_PERCENT=10 \
  --job-name=eval_lm1b_gen_ppl_hf_safe \
  eval_lm1b_gen_ppl_hf.sh
comment

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

# Default values
if [ -z "${SAMPLING_STEPS}" ]; then
  SAMPLING_STEPS=128
fi
if [ -z "${SEED}" ]; then
  SEED=1
fi
if [ -z "${USE_FLOAT64}" ]; then
  USE_FLOAT64=False
fi
if [ -z "${NUM_SAMPLE_BATCHES}" ]; then
  NUM_SAMPLE_BATCHES=32
fi
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE=32
fi

# Sampling mode: default, shs, shs_safe
if [ -z "${SAMPLING_MODE}" ]; then
  SAMPLING_MODE=shs
fi

# Blacklist settings (for safe-word generation research)
# BLACKLIST_PERCENT: percentage of vocabulary to blacklist (0-100)
# BLACKLIST_SEED: seed for blacklist generation (for reproducibility)
# BLACKLIST_WORDS: comma-separated list of specific words to ban (e.g., "hate,kill,violence")
if [ -z "${BLACKLIST_PERCENT}" ]; then
  BLACKLIST_PERCENT=0
fi
if [ -z "${BLACKLIST_SEED}" ]; then
  BLACKLIST_SEED=42
fi
# BLACKLIST_WORDS is optional, no default value

# UDLM config for HuggingFace model
parameterization="d3pm"
diffusion="uniform"
TRAIN_T=0
time_conditioning=True
sampling_use_cache=False

# Output path - includes sampling mode and blacklist info
OUTPUT_DIR="${PWD}/outputs/lm1b/udlm-hf"
mkdir -p "${OUTPUT_DIR}"

# Build output filename based on blacklist settings
BLACKLIST_SUFFIX=""
if [ "${BLACKLIST_PERCENT}" != "0" ]; then
  BLACKLIST_SUFFIX="_bl-${BLACKLIST_PERCENT}pct_blseed-${BLACKLIST_SEED}"
fi
if [ -n "${BLACKLIST_WORDS}" ]; then
  # Create short hash of word list for filename
  WORDS_HASH=$(echo -n "${BLACKLIST_WORDS}" | md5sum | cut -c1-8)
  BLACKLIST_SUFFIX="${BLACKLIST_SUFFIX}_words-${WORDS_HASH}"
fi

if [ -n "${BLACKLIST_SUFFIX}" ]; then
  generated_seqs_path="${OUTPUT_DIR}/samples-lm1b-mode-${SAMPLING_MODE}${BLACKLIST_SUFFIX}_T-${SAMPLING_STEPS}_seed-${SEED}.json"
else
  generated_seqs_path="${OUTPUT_DIR}/samples-lm1b-mode-${SAMPLING_MODE}_T-${SAMPLING_STEPS}_seed-${SEED}.json"
fi

echo "=== Gen-PPL Evaluation with HuggingFace UDLM-LM1B ==="
echo "Sampling mode: ${SAMPLING_MODE}"
echo "Sampling steps: ${SAMPLING_STEPS}"
echo "Seed: ${SEED}"
if [ "${BLACKLIST_PERCENT}" != "0" ]; then
  echo "Blacklist (random): ${BLACKLIST_PERCENT}% (seed=${BLACKLIST_SEED})"
fi
if [ -n "${BLACKLIST_WORDS}" ]; then
  echo "Blacklist (words): ${BLACKLIST_WORDS}"
fi
echo "Output: ${generated_seqs_path}"

# Build blacklist args
BLACKLIST_ARGS=""
if [ "${BLACKLIST_PERCENT}" != "0" ]; then
  BLACKLIST_ARGS="+sampling.blacklist_percent=${BLACKLIST_PERCENT} +sampling.blacklist_seed=${BLACKLIST_SEED}"
fi
if [ -n "${BLACKLIST_WORDS}" ]; then
  BLACKLIST_ARGS="${BLACKLIST_ARGS} +sampling.blacklist_words='${BLACKLIST_WORDS}'"
fi

# shellcheck disable=SC2086
python -u -m main \
    hydra.output_subdir=null \
    hydra.run.dir="${PWD}" \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    seed=${SEED} \
    mode="gen_ppl_eval" \
    data=lm1b \
    backbone=hf_dit \
    model=hf \
    model.pretrained_model_name_or_path="kuleshov-group/udlm-lm1b" \
    model.length=128 \
    training.guidance=null \
    parameterization=${parameterization} \
    diffusion=${diffusion} \
    time_conditioning=${time_conditioning} \
    T=${TRAIN_T} \
    sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
    sampling.batch_size=${BATCH_SIZE} \
    sampling.steps=${SAMPLING_STEPS} \
    sampling.use_cache=${sampling_use_cache} \
    sampling.use_float64=${USE_FLOAT64} \
    +sampling.sampling_mode=${SAMPLING_MODE} \
    ${BLACKLIST_ARGS} \
    eval.generated_samples_path=${generated_seqs_path} \
    +eval.generative_ppl_model_name_or_path="gpt2-large"
