# Systematic Hazard Sampling (SHS) — UDLM

Project page: https://jang-seunghwan.github.io/Systematic-Hazard-Sampling/

Official code for **Systematic Hazard Sampling: Minimal-Variance Inference for Discrete Diffusion and Flow Models** (NeurIPS 2026).

**Seunghwan Jang**<sup>1,†</sup>, **Wonje Jeung**<sup>2</sup>, **SooJean Han**<sup>3</sup>  
<sup>1</sup>Nanyang Technological University &nbsp; <sup>2</sup>University of Michigan &nbsp; <sup>3</sup>KAIST  
Work done while Seunghwan Jang was at KAIST and Wonje Jeung was at Yonsei University.  
<sup>†</sup>Corresponding author: seunghwa001@e.ntu.edu.sg

[[arXiv]](https://arxiv.org/abs/2601.02799) (an earlier version of the paper appeared on arXiv under the title *Stratified Hazard Sampling*)

SHS is a training-free, hyperparameter-free drop-in replacement for the per-step stay-vs.-replace decisions of CTMC/DTMC samplers.
Instead of drawing an independent Bernoulli change decision at every step, each position accumulates its jump mass
`S_i = sum_k p_ik` and jumps whenever `S_i` crosses `theta_i + k` (`k = 0, 1, ...`), with a single random phase `theta_i ~ U(0,1)` per position.
This keeps the expected number of jumps and the destination distribution unchanged while minimizing the jump-count variance (at most 1/4 for a fixed cumulative mass).

This branch applies SHS to **UDLM (uniform-noise discrete diffusion, `kuleshov-group/udlm-lm1b`, LM1B, 128 tokens)**.

| Branch | Model | Base code |
|---|---|---|
| [`main`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/main) | UDLM (`kuleshov-group/udlm-lm1b`) | [kuleshov-group/discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance) |
| [`gidd`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/gidd) | GIDD (`dvruette/gidd-base-p_unif-0.2`) | [dvruette/gidd](https://github.com/dvruette/gidd) |
| [`gidd-easydel`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/gidd-easydel) | 3B uniform diffusion (`dvruette/gidd-unif-3b`) | [dvruette/gidd-easydel](https://github.com/dvruette/gidd-easydel) |
| [`fs-dfm`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/fs-dfm) | DFM 1.3B (Apple FS-DFM release) | [apple/ml-fs-dfm](https://github.com/apple/ml-fs-dfm) |

## Setup

```bash
conda env create -f requirements.yaml
conda activate discdiff
```

The environment is the one of the base repository (Python 3.9, `torch==2.2.2`, `transformers==4.38.2`; `diffusion.py` imports `mamba_ssm`, so `mamba-ssm` must be installed).
No training is needed. The following are downloaded from the Hugging Face Hub on first use
(set `HF_HOME` to choose the cache location):

- the UDLM checkpoint [`kuleshov-group/udlm-lm1b`](https://huggingface.co/kuleshov-group/udlm-lm1b) (loaded with `trust_remote_code=True`),
- the `bert-base-uncased` tokenizer (the UDLM / LM1B tokenizer),
- `gpt2-large` (the Gen PPL evaluator).

## Quick start

The sampler is selected with the config key `sampling.sampling_mode`:
`default` (Standard: the base repository's per-step sampler, which is also the config default) or `shs` (Systematic Hazard Sampling).
Generate 8 samples with each sampler at 16 NFE and evaluate them:

```bash
MODES="default shs" NFES=16 SEEDS=1 NUM_SAMPLE_BATCHES=1 BATCH_SIZE=8 \
OUTPUT_DIR=outputs/quickstart bash scripts/eval_udlm_gen_ppl.sh
```

This writes `outputs/quickstart/samples-lm1b-mode-{default,shs}_T-16_seed-1.json`.
In Python, `Diffusion.sample(sampling_mode='shs')` returns a `(batch_size, model.length)` tensor of token ids
(`sampling_mode=None` uses the config value).

## Reproducing the paper results

**Table 1 (Gen PPL and Entropy, UDLM).** The default settings of the script run the full paper grid:
samplers `default` and `shs` x NFE `4 8 16 32 64 128 256 512 1024` x seeds `1 2 3 4 5`, with 1024 samples per run (32 batches of 32), sequence length 128,
`T=0`, `time_conditioning=True`, `sampling.use_cache=False`, `sampling.use_float64=False`, no guidance.

```bash
bash scripts/eval_udlm_gen_ppl.sh
# or split the grid, e.g. one sampler per GPU:
CUDA_VISIBLE_DEVICES=0 MODES=default bash scripts/eval_udlm_gen_ppl.sh
CUDA_VISIBLE_DEVICES=1 MODES=shs     bash scripts/eval_udlm_gen_ppl.sh
```

Each run (`python -m main mode=gen_ppl_eval ...`, seeded with `seed=<seed>` via `L.seed_everything`) writes one json,
`outputs/lm1b/udlm-hf/samples-lm1b-mode-<mode>_T-<nfe>_seed-<seed>.json`, with the keys
`generative_ppl`, `entropy`, `sampling_mode`, `steps`, `seed` and `generated_seqs` (the 1024 decoded samples).
Table 1 reports the mean and standard deviation over the 5 seeds.

- **Gen PPL**: the samples are re-tokenized with the GPT-2 tokenizer (truncated to 128 tokens) and scored by GPT-2 Large;
  the reported value is the corpus-level perplexity `exp(total NLL / total number of tokens)` over all 1024 samples
  (`eval_utils.compute_generative_ppl`, unchanged from the base repository).
- **Entropy**: the samples are re-encoded with the BERT tokenizer (`max_length=128`, padded to 128, padding included);
  the reported value is the unigram entropy (in nats) of the pooled token counts of all 1024 samples (`main._gen_ppl_eval`, unchanged from the base repository).

**Table 2 (diversity).** Distinct-n and Self-BLEU are computed on the seed-1 runs from the `generated_seqs` field of the jsons above
(the diversity script is not part of this repository).

**Jump statistics.** For the jump-count variance table and the count-deviation figure, add `SAVE_JUMP_STATS=True`
(config key `sampling.save_jump_stats`, default `False`):

```bash
SAVE_JUMP_STATS=True bash scripts/eval_udlm_gen_ppl.sh
```

Each run then also writes `<json name>_jump_stats.npz` with arrays of shape `(1024, 128)`:
`jump_counts` (`J`, number of jumps per position), `cumulative_mass` (`S`, float32, the per-position sum of the per-step jump masses `p = 1 - q(x_s = x_t | x_t)`)
and `token_ids` (the final samples). For the Standard sampler, `J` counts the steps at which the token changed;
for SHS, `J` counts the triggered jumps. Saving the statistics does not change the samples.

## What SHS changes in this codebase

Relative to [kuleshov-group/discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance):

- `diffusion.py`
  - `Diffusion._diffusion_sample_shs` (new): the SHS sampler for unconditional sampling (`guidance=null`).
  - `Diffusion.sample`: new arguments `sampling_mode` (`default` / `shs`) and `return_jump_stats`.
  - `Diffusion._diffusion_sample` (Standard): optional jump statistics (`return_jump_stats`); the sampling itself is unchanged.
- `main.py` (`_gen_ppl_eval`): uses `sampling.sampling_mode`, adds `sampling_mode`, `steps` and `seed` to the output json, and optionally saves the jump statistics.
- `configs/config.yaml`: new keys `sampling.sampling_mode` (default `default`) and `sampling.save_jump_stats` (default `False`).
- `scripts/eval_udlm_gen_ppl.sh` (new): the paper grid.

## Citation

```bibtex
@inproceedings{jang2026systematic,
  title     = {Systematic Hazard Sampling: Minimal-Variance Inference for Discrete Diffusion and Flow Models},
  author    = {Jang, Seunghwan and Jeung, Wonje and Han, SooJean},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

We thank Albert No and Yair Schiff for helpful discussions and advice.

This repository is built on [kuleshov-group/discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance)
(Schiff et al., *Simple Guidance Mechanisms for Discrete Diffusion Models*), which builds on [MDLM](https://github.com/kuleshov-group/mdlm) and [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion),
and uses their released UDLM checkpoint. All other code (training, guidance, other datasets) is kept as in the base repository; see its README for those experiments.
The base repository's license (Apache License 2.0, [`LICENSE`](./LICENSE)) is kept.
