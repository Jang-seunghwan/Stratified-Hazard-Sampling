# Systematic Hazard Sampling (SHS) — GIDD

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

This branch applies SHS to **GIDD (`dvruette/gidd-base-p_unif-0.2`, hybrid masking + uniform noise with p_u = 0.2, OpenWebText, 512 tokens); SHS schedules the replacement events of non-[MASK] tokens, while [MASK] tokens are unmasked by the standard posterior sampler**.

| Branch | Model | Base code |
|---|---|---|
| [`main`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/main) | UDLM (`kuleshov-group/udlm-lm1b`) | [kuleshov-group/discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance) |
| [`gidd`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/gidd) | GIDD (`dvruette/gidd-base-p_unif-0.2`) | [dvruette/gidd](https://github.com/dvruette/gidd) |
| [`gidd-easydel`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/gidd-easydel) | 3B uniform diffusion (`dvruette/gidd-unif-3b`) | [dvruette/gidd-easydel](https://github.com/dvruette/gidd-easydel) |
| [`fs-dfm`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/fs-dfm) | DFM 1.3B (Apple FS-DFM release) | [apple/ml-fs-dfm](https://github.com/apple/ml-fs-dfm) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

The paper results were produced with PyTorch 2.5.1 and Transformers 4.45.1 (the versions pinned in `requirements.txt`).
The GIDD checkpoint [`dvruette/gidd-base-p_unif-0.2`](https://huggingface.co/dvruette/gidd-base-p_unif-0.2) and the Gen PPL evaluator [`gpt2-large`](https://huggingface.co/openai-community/gpt2-large) are downloaded from the HuggingFace Hub on first use.

## Quick start

```python
import torch
from gidd import GiddPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = GiddPipeline.from_pretrained("dvruette/gidd-base-p_unif-0.2", trust_remote_code=True)
pipe.to(device)

texts_std = pipe.generate(num_samples=4, num_inference_steps=128)                # Standard sampler
texts_shs = pipe.generate(num_samples=4, num_inference_steps=128, use_shs=True)  # SHS
```

Command line (one seed, one NFE, 64 samples):

```bash
python eval_gen_ppl.py --mode default --seeds 1 --nfes 128 --num_samples 64
python eval_gen_ppl.py --mode shs     --seeds 1 --nfes 128 --num_samples 64
```

## Reproducing the paper results

Table 1 (GIDD rows: Gen PPL and Entropy, Standard vs. SHS, NFE 4–512, 5 seeds x 1024 samples):

```bash
bash scripts/run_gidd_eval.sh            # modes default + shs
bash scripts/run_gidd_eval.sh shs        # a single mode
```

The script loops over modes `default` (Standard) and `shs`, NFE `4 8 16 32 64 128 256 512`, seeds `1 2 3 4 5`, with 1024 samples per run and batch size 16, and skips runs whose json already exists
(environment variables `PYTHON`, `OUTPUT_DIR`, `NFES`, `SEEDS`, `NUM_SAMPLES`, `BATCH_SIZE` override the defaults).
The defaults of `eval_gen_ppl.py` are the paper protocol, so `python eval_gen_ppl.py --mode shs` runs the whole SHS grid in one process; the seed is reset before every (seed, NFE) run, so the samples do not depend on how the grid is split.

- **Gen PPL**: GPT-2 large (`gpt2-large`, bf16) scores the decoded samples (GPT-2 tokenizer, `max_length=512` with truncation, batch size 16); Gen PPL = exp of the token-averaged NLL over all scored tokens of the 1024 samples of a run.
- **Entropy**: the samples are re-tokenized with the GIDD tokenizer (`add_special_tokens=False`, `max_length=512`, truncation, `padding="max_length"`), pad tokens (id 50256) are dropped, and the unigram entropy (nats) of all remaining tokens of the run is computed (`compute_entropy` in `eval_gen_ppl.py`).
- Table 1 reports the mean and standard deviation over seeds 1–5. Self-correction is not used in the paper (`--self_correction` enables GIDD's self-correction step).
- Table 2 (diversity) is computed from the `generated_seqs` of the seed-1 runs; the diversity script is not included here.

Each run writes `outputs/<mode>/seed<S>_nfe<N>.json` with `ppl`, `avg_nll`, `median_nll`, `acc`, `tokens`, `entropy` and all generated texts (`generated_seqs`);
`outputs/<mode>/summary.json` lists the metrics of all runs in that directory.
The paper samples were generated on an NVIDIA RTX 4090; other GPUs or library versions can produce different samples because of floating-point differences.

## What SHS changes in this codebase

- `gidd/sampling.py`: new `GiddSamplerSHS` (SHS for non-[MASK] tokens, standard posterior sampling for [MASK] tokens, same posterior as `GiddSampler`). The Standard sampler `GiddSampler` is unchanged.
- `gidd/pipeline.py`: `GiddPipeline` builds both samplers; `generate(..., use_shs=True)` selects SHS.
- `eval_gen_ppl.py`: evaluation entry point (generation, Gen PPL, Entropy; one json per run).
- `scripts/run_gidd_eval.sh`: paper grid.

The training and evaluation code of GIDD is otherwise unchanged; see the [upstream README](https://github.com/dvruette/gidd) for training and the original evaluation scripts.

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
@inproceedings{vonrutte2025generalized,
  title     = {Generalized Interpolating Discrete Diffusion},
  author    = {von R{\"u}tte, Dimitri and Fluri, Janis and Ding, Yuhui and Orvieto, Antonio and Sch{\"o}lkopf, Bernhard and Hofmann, Thomas},
  booktitle = {International Conference on Machine Learning},
  year      = {2025}
}
```

## Acknowledgements

We thank Albert No and Yair Schiff for helpful discussions and advice.

This code is built on [dvruette/gidd](https://github.com/dvruette/gidd), the official implementation of *Generalized Interpolating Discrete Diffusion* (von Rütte et al., ICML 2025), and uses its pretrained checkpoint `dvruette/gidd-base-p_unif-0.2`.
The upstream MIT license is kept in [`LICENSE`](LICENSE).
