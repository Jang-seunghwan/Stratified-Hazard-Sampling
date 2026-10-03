# Systematic Hazard Sampling (SHS) — DFM

Official code for **Systematic Hazard Sampling: Minimal-Variance Inference for Discrete Diffusion and Flow Models** (NeurIPS 2026).

**Seunghwan Jang**<sup>1,†</sup>, **Wonje Jeung**<sup>2</sup>, **SooJean Han**<sup>1</sup>  
<sup>1</sup>KAIST &nbsp; <sup>2</sup>Yonsei University  
Seunghwan Jang is now at Nanyang Technological University, and Wonje Jeung is now at the University of Michigan.  
<sup>†</sup>Corresponding author: seunghwa001@e.ntu.edu.sg

[[arXiv]](https://arxiv.org/abs/2601.02799) (an earlier version of the paper appeared on arXiv under the title *Stratified Hazard Sampling*)

SHS is a training-free, hyperparameter-free drop-in replacement for the per-step stay-vs.-replace decisions of CTMC/DTMC samplers.
Instead of drawing an independent Bernoulli change decision at every step, each position accumulates its jump mass
`S_i = sum_k p_ik` and jumps whenever `S_i` crosses `theta_i + k` (`k = 0, 1, ...`), with a single random phase `theta_i ~ U(0,1)` per position.
This keeps the expected number of jumps and the destination distribution unchanged while minimizing the jump-count variance (at most 1/4 for a fixed cumulative mass).

This branch applies SHS to **the 1.3B discrete flow matching (DFM) language model released with FS-DFM (Apple), with uniform source distribution, 1024 tokens; the per-step change mass is p = min(h*lambda, 1) (`--dtmc`) in the main results and p = 1 - exp(-h*lambda) in the appendix comparison**.

| Branch | Model | Base code |
|---|---|---|
| [`main`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/main) | UDLM (`kuleshov-group/udlm-lm1b`) | [kuleshov-group/discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance) |
| [`gidd`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/gidd) | GIDD (`dvruette/gidd-base-p_unif-0.2`) | [dvruette/gidd](https://github.com/dvruette/gidd) |
| [`gidd-easydel`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/gidd-easydel) | 3B uniform diffusion (`dvruette/gidd-unif-3b`) | [dvruette/gidd-easydel](https://github.com/dvruette/gidd-easydel) |
| [`fs-dfm`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/fs-dfm) | DFM 1.3B (Apple FS-DFM release) | [apple/ml-fs-dfm](https://github.com/apple/ml-fs-dfm) |

## Setup

```bash
conda env create -f fsdfm_environment.yml
conda activate FSDFM
pip install -e .
```

The paper results were produced with Python 3.9, PyTorch 2.2.2 (CUDA 12.1) and Transformers 4.38.2 on a single NVIDIA RTX 4090 (24 GB).

**Checkpoint.** Download the DFM checkpoint released with FS-DFM (about 16 GB; it also contains the optimizer state) and place it at `checkpoints/checkpoint.pth`:

```bash
mkdir -p checkpoints
wget -O checkpoints/checkpoint.pth https://ml-site.cdn-apple.com/models/fs-dfm/checkpoint.pth
```

Any other location works as well; pass it with `--pre_trained_model_path` (or `CKPT=...` for the grid script).
The model configuration of this checkpoint is `fs_dfm/configs/config_1.3b.yaml` (the default of `--config`).
The GPT-2 tokenizer is downloaded on the first run and cached under `--cache_dir` (default `./cache_dir`);
the Gen PPL evaluator [`gpt2-large`](https://huggingface.co/openai-community/gpt2-large) is downloaded from the HuggingFace Hub on first use.

## Quick start

One run = one sampler, one NFE, one seed. Standard vs. SHS at NFE 16 with 64 samples:

```bash
python fs_dfm/run_eval.py --teacher_model --dtmc --eval_perplexity \
    --pre_trained_model_path checkpoints/checkpoint.pth --work_dir results/quick \
    --sampling_steps 16 --seed 1 --perplexity_n_samples 64

python fs_dfm/run_eval.py --teacher_model --dtmc --use-shs --eval_perplexity \
    --pre_trained_model_path checkpoints/checkpoint.pth --work_dir results/quick \
    --sampling_steps 16 --seed 1 --perplexity_n_samples 64
```

| Flag | Meaning |
|---|---|
| `--teacher_model` | Sample from the DFM model (always set for the paper results). |
| `--use-shs` | Systematic Hazard Sampling; without it, the Standard sampler (independent per-step decisions). |
| `--dtmc` | Per-step change mass `p = min(h*lambda, 1)` (paper main results). Without it, `p = 1 - exp(-h*lambda)` (appendix comparison). |
| `--sampling_steps` | NFE. |
| `--perplexity_n_samples`, `--batch_size` | Number of samples (default 1024) and sampling batch size (default 4). |
| `--seed` | Seed (`torch.manual_seed` once at the start of the run). |

The four samplers are also available by name through `flow_matching.solver.get_solver_by_name`:
`mixture_euler` (Standard, `1 - exp(-h*lambda)`), `mixture_euler_dtmc` (Standard, `min(h*lambda, 1)`),
`mixture_euler_shs` (SHS, `1 - exp(-h*lambda)`), `mixture_euler_shs_dtmc` (SHS, `min(h*lambda, 1)`);
see `fs_dfm/logic/generate.py:generate_samples` for how a solver is built and called.

## Reproducing the paper results

Table 1 (DFM rows: Gen PPL and Entropy, Standard vs. SHS, NFE 4–128, 3 seeds x 1024 samples):

```bash
CKPT=checkpoints/checkpoint.pth bash scripts/run_dfm_eval.sh results/dfm
```

The script runs `fs_dfm/run_eval.py --teacher_model --dtmc [--use-shs] --eval_perplexity` for the samplers Standard and SHS, NFE `4 8 16 32 64 128` and seeds `1 2 3`,
with 1024 samples per run and batch size 4 (the paper setting; the samples of a seed depend on the batch size), skips runs whose json already exists,
and finally prints the mean and standard deviation (ddof 0) over seeds for every (NFE, sampler) cell.
Environment variables `PYTHON`, `CACHE_DIR`, `NFES`, `SEEDS`, `SAMPLERS`, `N_SAMPLES`, `BATCH_SIZE` override the defaults.
The paper's main results use `--dtmc`; omitting it gives the `p = 1 - exp(-h*lambda)` variant of the appendix comparison.

- **Gen PPL**: GPT-2 large (`gpt2-large`, fp32) scores the generated token ids directly (no decoding and re-tokenization, no attention mask, batch size 4); Gen PPL = exp(sum of token NLLs / number of scored tokens) over all positions 2..1024 of the 1024 samples of a run (`compute_gen_ppl_nll` in `fs_dfm/logic/evaluate.py`).
- **Entropy**: unigram entropy (log2) of the token ids of each generated sequence, averaged over the 1024 sequences (`compute_entropy` in `fs_dfm/logic/evaluate.py`).
- Table 2 (diversity) is computed from the samples of the seed-1 runs; the diversity script is not included here.

Each run writes, under `results/dfm/<run>/` with `<run> = dfm_<standard|shs>_dtmc_nfe<N>_seed<S>`:
`<run>.json` (`gen_ppl`, `entropy`, `n_samples`, sampler settings and the path of the samples file) and
`<run>.txt` (all generated samples, one per line; backslashes, newlines and carriage returns inside a sample are written as `\\`, `\n`, `\r`).
The paper samples were generated on an NVIDIA RTX 4090; other GPUs or library versions can produce different samples because of floating-point differences.

## What SHS changes in this codebase

Relative to [apple/ml-fs-dfm](https://github.com/apple/ml-fs-dfm):

- `flow_matching/solver/discrete_solver_fsdfm.py`
  - `MixtureDiscreteEulerSolverSHS` (SHS) and `MixtureDiscreteEulerSolverSHS_DTMC`, registered as `mixture_euler_shs` and `mixture_euler_shs_dtmc`.
  - `MixtureDiscreteEulerSolver` (Standard) gets `p_jump_mode`: `"ctmc"` (Poisson jump test as upstream) or `"dtmc"` (Bernoulli with `p = min(h*lambda, 1)`, `MixtureDiscreteEulerSolver_DTMC` / `mixture_euler_dtmc`). The destination distribution is the rate vector normalized without the upstream `1e-9` floor, so tokens with zero rate are never drawn.
  - Last step: upstream resamples every position from `p_{1|t}` at the last step. Here the last step uses the sampler's own rule with change mass `p = 1 - p_{1|t}(x_t)` and destination `p_{1|t}` without the current token: a Bernoulli decision for Standard (equal in distribution to the upstream resampling) and the SHS schedule for SHS. The masked-source path is unchanged.
- `fs_dfm/logic/generate.py`: the solver is selected by name.
- `fs_dfm/eval.py`, `fs_dfm/run_eval.py`: one NFE per run, unconditional generation only; flags `--use-shs`, `--dtmc`, `--config`, `--cache_dir`; Gen PPL and Entropy over all samples, one json and one samples file per run. The validation data is only loaded for `--eval_elbo`.
- `fs_dfm/logic/evaluate.py`: `compute_gen_ppl_nll` (Gen PPL above).
- `fs_dfm/utils/checkpointing.py`: also loads the DFM release checkpoint (`model` key); the checkpoint is loaded on the CPU first and DDP is used only with more than one GPU.
- `fs_dfm/configs/config_1.3b.yaml`: configuration of the released 1.3B checkpoints.
- `scripts/run_dfm_eval.sh`: paper grid.

The FS-DFM student path (`fs_dfm/run_eval.py` without `--teacher_model`, with the FS-DFM checkpoint) and the training code are kept from upstream but are not used in the paper; see the [upstream README](https://github.com/apple/ml-fs-dfm) for training.

## Citation
```bibtex
@inproceedings{jang2026systematic,
  title     = {Systematic Hazard Sampling: Minimal-Variance Inference for Discrete Diffusion and Flow Models},
  author    = {Jang, Seunghwan and Jeung, Wonje and Han, SooJean},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

```bibtex
@article{monsefi2025fsdfm,
  title   = {FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models},
  author  = {Monsefi, Amin Karimi and Bhendawade, Nikhil and Ciosici, Manuel Rafael and Culver, Dominic and Zhang, Yizhe and Belousova, Irina},
  journal = {arXiv preprint arXiv:2509.20624},
  year    = {2025}
}

@inproceedings{gat2024discrete,
  title     = {Discrete Flow Matching},
  author    = {Gat, Itai and Remez, Tal and Shaul, Neta and Kreuk, Felix and Chen, Ricky T. Q. and Synnaeve, Gabriel and Adi, Yossi and Lipman, Yaron},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

## Acknowledgements

We thank Albert No and Yair Schiff for helpful discussions and advice.

This code is built on [apple/ml-fs-dfm](https://github.com/apple/ml-fs-dfm), the official implementation of *FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models* (Monsefi et al., 2025),
which in turn builds on [Flow Matching](https://github.com/facebookresearch/flow_matching) from Meta; the DFM checkpoint is the one released by Apple with FS-DFM.
The Apple license is kept in [`LICENSE`](LICENSE) and the third-party notices (Flow Matching, CC BY-NC 4.0) in [`ACKNOWLEDGMENTS`](ACKNOWLEDGMENTS).
