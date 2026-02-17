# SHS Sampling for GIDD (Generalized Interpolating Discrete Diffusion)

This repository extends [GIDD](https://github.com/dvruette/gidd) with
**Stratified Hazard Sampling (SHS)** as an additional inference-time sampling method.
Training code is unchanged; SHS is available alongside the original diffusion sampler
via `use_shs=True` in `GiddPipeline.generate()`.

## SHS Paper
- Title: **Stratified Hazard Sampling: Minimal-Variance Event Scheduling for CTMC/DTMC Discrete Diffusion and Flow Models**
- Authors: **Seunghwan Jang, SooJean Han**
- arXiv: **https://arxiv.org/abs/2601.02799**

## Sampling Modes

| Mode | Description |
|------|-------------|
| `default` | Standard GIDD diffusion sampling |
| `shs` | Stratified Hazard Sampling (variance reduction for non-[MASK] tokens) |

SHS applies hazard-based jump events to non-[MASK] tokens while [MASK] tokens
use standard probabilistic unmasking, unchanged from the original sampler.

---

## Original Paper

By Dimitri von Rütte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Schölkopf, Thomas Hofmann

[![arXiv](https://img.shields.io/badge/arXiv-2503.04482-d22c2c.svg)](https://arxiv.org/abs/2503.04482)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xv4RyZhXHkIpIZeMYahl_4kMthLxKdg_?usp=sharing)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-GIDD-f59a0c)](https://huggingface.co/collections/dvruette/generalized-interpolating-discrete-diffusion-67c6fc45663eafb85c6487af)

---

![animation](animation.gif)

We present Generalized Interpolating Discrete Diffusion (GIDD), a novel framework for training discrete diffusion models.
GIDD can be seen as a generalization of the popular masked diffusion paradigm (MDM) to any diffusion process that can be written as a linear interpolation between a data distribution and some (time-variable) mixing distribution.
We demonstrate the flexibility of GIDD by training models on a hybrid diffusion process that combines masking and uniform noise.
The model therefore is trained to not only "fill in the blanks" (i.e. the masked tokens), but also to consider the correctness of already-filled-in tokens and, if necessary, replace incorrect tokens with more plausible ones.
We show that GIDD models trained on hybrid noise have better sample quality (generative PPL) than mask-only models, and that they are able to identify and correct their own mistakes in generated samples through a self-correction step.
This repository contains all training and evaluation code necessary for reproducing the results in the paper.



### Pretrained Models
Our trained checkpoints are available on HuggingFace under the following links. All of them have been trained on 131B tokens from the [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset with the [GPT-2 tokenizer](https://huggingface.co/openai-community/gpt2).

| Model | Small (169.6M) | Base (424.5M) |
|-------|-------|------|
| GIDD+ ($p_u = 0.0$) | [dvruette/gidd-small-p_unif-0.0](https://huggingface.co/dvruette/gidd-small-p_unif-0.0) | [dvruette/gidd-base-p_unif-0.0](https://huggingface.co/dvruette/gidd-base-p_unif-0.0) |
| GIDD+ ($p_u = 0.1$) | [dvruette/gidd-small-p_unif-0.1](https://huggingface.co/dvruette/gidd-small-p_unif-0.1) | [dvruette/gidd-base-p_unif-0.1](https://huggingface.co/dvruette/gidd-base-p_unif-0.1) |
| GIDD+ ($p_u = 0.2$) | [dvruette/gidd-small-p_unif-0.2](https://huggingface.co/dvruette/gidd-small-p_unif-0.2) | [dvruette/gidd-base-p_unif-0.2](https://huggingface.co/dvruette/gidd-base-p_unif-0.2) |


## Quick Start

0. Clone the repository (recommended version: [v1.1.0](https://github.com/dvruette/gidd/tree/v1.1.0)

1. Set up the environment:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

2. For quickly downloading a trained model and playing around with it, the `GiddPipeline` class is most convenient:

```python
from gidd import GiddPipeline

# Download a pretrained model from HuggingFace
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = GiddPipeline.from_pretrained("dvruette/gidd-base-p_unif-0.2", trust_remote_code=True)
pipe.to(device)

# Generate samples with SHS
texts = pipe.generate(num_samples=4, num_inference_steps=128, use_shs=True)

# Generate samples with default sampler
texts = pipe.generate(num_samples=4, num_inference_steps=128, use_shs=False)

# Run self-correction step
corrected_texts = pipe.self_correction(texts, num_inference_steps=128, early_stopping=True, temperature=0.1)

print(corrected_texts)
```

## Gen-PPL Evaluation

Compare default vs SHS sampling with generative PPL measurement.

```bash
# SHS sampling
python eval_gen_ppl.py --mode shs --seeds 0 1 2 --nfes 128

# Default sampling
python eval_gen_ppl.py --mode default --seeds 0 1 2 --nfes 128

# Multiple NFE values
python eval_gen_ppl.py --mode shs --seeds 0 1 2 --nfes 16 32 64 128 256

# Skip self-correction / PPL
python eval_gen_ppl.py --mode shs --no_self_correction --no_ppl
```

### Gen-PPL Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `default` | `default` / `shs` |
| `--seeds` | `0 1 2` | Random seeds |
| `--nfes` | `128` | NFE (num denoising steps) values |
| `--num_samples` | `64` | Number of samples per configuration |
| `--batch_size` | `16` | Batch size for generation and PPL |
| `--model` | `dvruette/gidd-base-p_unif-0.2` | HuggingFace model |
| `--ppl_model` | `gpt2-large` | Reference LM for PPL measurement |
| `--no_self_correction` | | Skip self-correction step |
| `--no_ppl` | | Skip PPL computation |

## Code Structure (SHS additions)

```
gidd/sampling.py
├── GiddSamplerSHS              # SHS sampler class (non-[MASK]: hazard-based, [MASK]: standard)
└── GiddSampler                 # Original diffusion sampler (unchanged)

gidd/pipeline.py
├── GiddPipeline.__init__()     # Creates both sampler and sampler_shs
└── GiddPipeline.generate()     # use_shs=True selects SHS sampler

eval_gen_ppl.py                         # Gen-PPL evaluation (default vs shs comparison)
```

For training, setup, and original evaluation scripts, see the original repository: [https://github.com/dvruette/gidd](https://github.com/dvruette/gidd)

## Citation

```bibtex
@misc{jang2026stratifiedhazardsamplingminimalvariance,
  title         = {Stratified Hazard Sampling: Minimal-Variance Event Scheduling for CTMC/DTMC Discrete Diffusion and Flow Models},
  author        = {Seunghwan Jang and SooJean Han},
  year          = {2026},
  eprint        = {2601.02799},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2601.02799},
}
```

```bibtex
@article{vonruette2025gidd,
  title={Generalized Interpolating Discrete Diffusion},
  author={von R{\"u}tte, Dimitri and Fluri, Janis and Ding, Yuhui and Orvieto, Antonio and Sch{\"o}lkopf, Bernhard and Hofmann, Thomas},
  journal={arXiv preprint arXiv:2503.04482},
  year={2025}
}
```

## Acknowledgements
This code is adapted from:
- **Generalized Interpolating Discrete Diffusion** [https://github.com/dvruette/gidd]
