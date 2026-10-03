# Systematic Hazard Sampling (SHS) — 3B uniform diffusion (GIDD-EasyDeL)

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

This branch applies SHS to **the 3B uniform-noise diffusion language model `dvruette/gidd-unif-3b` of von Rütte et al. (2025) (block-wise generation, 256 tokens, PyTorch inference path)**.

| Branch | Model | Base code |
|---|---|---|
| [`main`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/main) | UDLM (`kuleshov-group/udlm-lm1b`) | [kuleshov-group/discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance) |
| [`gidd`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/gidd) | GIDD (`dvruette/gidd-base-p_unif-0.2`) | [dvruette/gidd](https://github.com/dvruette/gidd) |
| [`gidd-easydel`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/gidd-easydel) | 3B uniform diffusion (`dvruette/gidd-unif-3b`) | [dvruette/gidd-easydel](https://github.com/dvruette/gidd-easydel) |
| [`fs-dfm`](https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling/tree/fs-dfm) | DFM 1.3B (Apple FS-DFM release) | [apple/ml-fs-dfm](https://github.com/apple/ml-fs-dfm) |

## Setup

All results of the paper for this model use the PyTorch inference path, which only needs PyTorch and `transformers`
(tested with Python 3.11, `torch==2.5.1`, `transformers==4.57.6`, one RTX 4090):

```bash
git clone -b gidd-easydel https://github.com/Jang-seunghwan/Systematic-Hazard-Sampling.git
cd Systematic-Hazard-Sampling
pip install torch==2.5.1 transformers==4.57.6 accelerate numpy tqdm
```

Weights, config and tokenizer are downloaded from the HuggingFace Hub ([`dvruette/gidd-unif-3b`](https://huggingface.co/dvruette/gidd-unif-3b); the paper used revision `b0357ea`),
and the Gen PPL evaluator from [`gpt2-large`](https://huggingface.co/gpt2-large).
The model *code* is not taken from the Hub: the Hub's remote code does not contain SHS, so the model class is loaded from the
local file `gidd_easydel/model/modeling_gidd_hf.py` (no `trust_remote_code`).

The JAX / EasyDeL training and evaluation code of the base repository (`main*.py`, `eval_ray.py`, `gidd_easydel/`) is kept unchanged;
see [dvruette/gidd-easydel](https://github.com/dvruette/gidd-easydel) for its setup (JAX, EasyDeL/eformer forks) and usage.

## Quick start

```python
import torch
from eval_gen_ppl import load_gidd_model

model, tokenizer = load_gidd_model("dvruette/gidd-unif-3b", torch.bfloat16, "cuda")
inputs = torch.full((4, 1), tokenizer.bos_token_id, dtype=torch.long, device="cuda")  # unconditional

for method in ["ancestral", "shs"]:  # Standard vs. SHS
    torch.manual_seed(0)
    ids = model.generate(inputs=inputs, max_length=256, block_length=128, steps=16,
                         sampling_method=method, temperature=1.0)
    print(method, tokenizer.batch_decode(ids, skip_special_tokens=True)[0][:300])
```

`sampling_method` is one of `"ancestral"` (Standard), `"shs"` (Systematic Hazard Sampling) and `"adaptive"`
(the confidence-based decoding of the base repository); `steps` is the number of denoising steps (NFE) per block.

A small end-to-end run of the evaluation (16 samples, NFE 16):

```bash
python eval_gen_ppl.py --mode shs --nfe 16 --num_samples 16
```

## Reproducing the paper results

Table 1 (3B column, Gen PPL and Entropy of Standard and SHS for NFE 4, 8, 16, 32, 64, 128; seed 0; 1000 samples each):

```bash
bash run_gen_ppl.sh
```

which runs, for `mode` in `ancestral shs` and `nfe` in `4 8 16 32 64 128`,

```bash
python eval_gen_ppl.py --mode $mode --nfe $nfe --seed 0
```

The defaults of `eval_gen_ppl.py` are the paper protocol: 1000 unconditional samples per run, generated from the BOS token
with `max_length=256` (two blocks of `block_length=128`, `--nfe` steps per block), temperature 1.0, model in bf16, batch size 4,
with `torch.manual_seed(seed)` set once before generation.
Adaptive decoding can be added with `MODES="ancestral adaptive shs" bash run_gen_ppl.sh`;
`run_gen_ppl.sh` also reads `NFES`, `SEEDS`, `NUM_SAMPLES`, `OUTPUT_DIR` and `PYTHON` from the environment.

- **Gen PPL**: GPT-2 Large (fp32) on the non-empty samples (batch 8, right-padded, truncated to 512 tokens); exp of the token-averaged NLL pooled over all samples (scored positions: `attention_mask[:, :-1]`). No sample is removed.
- **Entropy**: entropy (nats) of the unigram distribution of the GPT-2 token ids pooled over all non-empty samples.

Each run writes `outputs/gen_ppl/{mode}_nfe{nfe}_seed{seed}.json` with the settings, `gen_ppl`, `avg_nll`, `entropy`,
the number of scored texts/tokens and the generated `texts`.
On one RTX 4090, a run of 1000 samples takes about 0.5 h at NFE 4 and about 13 h at NFE 128.

Reference values (seed 0, computed with `eval_gen_ppl.py` on the samples used in the paper):

| NFE | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| Standard Gen PPL | 936.3 | 443.4 | 221.4 | 150.2 | 113.7 | 104.9 |
| SHS Gen PPL | 869.5 | 417.2 | 218.1 | 142.0 | 113.0 | 99.6 |
| Standard Entropy | 6.758 | 6.660 | 6.612 | 6.649 | 6.654 | 6.649 |
| SHS Entropy | 6.634 | 6.536 | 6.554 | 6.588 | 6.619 | 6.618 |

Samples are reproduced token-for-token only with the same GPU type and library versions; elsewhere, floating-point differences give different (statistically equivalent) samples.

## What SHS changes in this codebase

- `gidd_easydel/model/modeling_gidd_hf.py` (PyTorch model, used for all results):
  - `GiddForDiffusionLM._sample_shs`: the SHS step (new).
  - `GiddForDiffusionLM.generate`: `sampling_method="shs"`; the SHS state (jump mass `S`, jump count `k`, phase `theta`) is reset for every block.
  - `GiddForDiffusionLM._sample_ancestral` (Standard): the posterior is clamped to be non-negative and renormalized, and rows without probability mass keep the current token.
  - `"adaptive"`: `tokens_per_step` defaults to `ceil(block_length / steps)` (was 1).
- `gidd_easydel/sampling.py` (JAX / EasyDeL inference): `shs_sampling_step` and `generate(sampler="shs")`. Not used for the paper results.
- `eval_gen_ppl.py`, `run_gen_ppl.sh`: Gen PPL / Entropy evaluation (new).

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
@article{von2025scaling,
  title={Scaling Behavior of Discrete Diffusion Language Models},
  author={von R{\"u}tte, Dimitri and Fluri, Janis and Pooladzandi, Omead and Sch{\"o}lkopf, Bernhard and Hofmann, Thomas and Orvieto, Antonio},
  journal={arXiv preprint arXiv:2512.10858},
  year={2025}
}
```

## Acknowledgements

We thank Albert No and Yair Schiff for helpful discussions and advice.

This branch is built on [dvruette/gidd-easydel](https://github.com/dvruette/gidd-easydel), the code of
"Scaling Behavior of Discrete Diffusion Language Models" (von Rütte et al., 2025), and uses their released `gidd-unif-3b` checkpoint.
The code is released under the Apache License 2.0 of the base repository; see [LICENSE](LICENSE).
