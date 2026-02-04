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
SAMPLING_STEPS=128 bash eval_lm1b_gen_ppl_hf.sh

# Or with sbatch:
sbatch \
  --export=ALL \
  --job-name=eval_lm1b_gen_ppl_hf \
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
if [ -z "${USE_SHS}" ]; then
  USE_SHS=True
fi

# UDLM config for HuggingFace model
parameterization="d3pm"
diffusion="uniform"
TRAIN_T=0
time_conditioning=True
sampling_use_cache=False

# Output path
OUTPUT_DIR="${PWD}/outputs/lm1b/udlm-hf"
mkdir -p "${OUTPUT_DIR}"
generated_seqs_path="${OUTPUT_DIR}/samples-lm1b-gen-ppl-eval-shs-${USE_SHS}_float64-${USE_FLOAT64}_T-${SAMPLING_STEPS}_seed-${SEED}.json"

echo "=== Gen-PPL Evaluation with HuggingFace UDLM-LM1B ==="
echo "Sampling steps: ${SAMPLING_STEPS}"
echo "Seed: ${SEED}"
echo "Use SHS: ${USE_SHS}"
echo "Output: ${generated_seqs_path}"

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
    +sampling.use_shs=${USE_SHS} \
    eval.generated_samples_path=${generated_seqs_path} \
    +eval.generative_ppl_model_name_or_path="gpt2-large"
