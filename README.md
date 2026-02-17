# SHS Sampling for GIDD (Scaling Discrete Diffusion LMs)

This repository extends [GIDD-EasyDeL](https://github.com/dvruette/gidd-easydel) with
**Stratified Hazard Sampling (SHS)** as an additional inference-time sampling method.
Training code is unchanged; SHS is available alongside the original `ancestral` and `adaptive`
methods via `sampling_method="shs"`.

## SHS Paper
- Title: **Stratified Hazard Sampling: Minimal-Variance Event Scheduling for CTMC/DTMC Discrete Diffusion and Flow Models**
- Authors: **Seunghwan Jang, SooJean Han**
- arXiv: **https://arxiv.org/abs/2601.02799**

## Sampling Modes

| Mode | Description |
|------|-------------|
| `ancestral` | Standard ancestral sampling (original GIDD) |
| `adaptive` | Score-based adaptive token selection |
| `shs` | Stratified Hazard Sampling (variance reduction) |

---

## Original Paper

Dimitri von Rütte, Janis Fluri, Antonio Orvieto, Omead Pooladzandi, Bernhard Schölkopf, Thomas Hofmann


[![arXiv](https://img.shields.io/badge/arXiv-2512.10858-d22c2c.svg)](https://arxiv.org/abs/2512.10858)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Scaling%20GIDD-f59a0c)](https://huggingface.co/collections/dvruette/scaling-behavior-of-discrete-diffusion-language-models)

This repository contains the code to reproduce the experiments from the paper "Scaling Behavior of Discrete Diffusion Language Models".
It includes implementations of the model architecture, training procedures, and evaluation code used in the study.
The implementation is based on [EasyDeL](https://github.com/erfanzar/EasyDeL), a JAX-based framework for training and running LLMs at scale.

In our paper, we investigate the scaling behavior of discrete diffusion language models (DLMs) for different noise types (masking, uniform, and hybrid-noise), finding that all of them scale well in compute-bound settings and especially in token-bound settings, with uniform noise coming out on top for the latter.
To confirm these findings, we train scaled-up models to compute optimality.
Specifically, we train two 3B models (masked and uniform diffusion) as well as a 10B parameter uniform diffusion model, which, to the best of our knowledge, is the largest public uniform diffusion model to date.
Below we plot the compute-bound and token-bound scaling laws for all investigated noise types with the scaled-up runs (3B and 10B) overlayed as circles.

[![Scaling laws of discrete diffusion language models](thumbnail.png)](https://arxiv.org/abs/2512.10858)

| Model | Size | Train. PPL | Diffusion type | HuggingFace link |
|:------|-----:|-----------:|:---------------|:-----------------|
| `gidd-unif-10b` | 10B | 9.15 | uniform | https://huggingface.co/dvruette/gidd-unif-10b |
| `gidd-mask-3b` | 3B | 11.3 | masked | https://huggingface.co/dvruette/gidd-mask-3b |
| `gidd-unif-3b` | 3B | 11.7 | uniform | https://huggingface.co/dvruette/gidd-unif-3b |

## Quick Start

The 3B and 10B models are available as converted PyTorch models on HuggingFace and can be used as follows:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "dvruette/gidd-unif-10b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
model.eval().to(device)

prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[:, :-1].to(device)

# SHS sampling
generated_ids = model.generate(
    inputs=inputs,
    max_length=128,
    block_length=128,
    steps=256,
    sampling_method="shs",       # or "ancestral", "adaptive"
    temperature=0.0,
    show_progress=True,
)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])
```

## Gen-PPL Evaluation

Two-phase workflow: (1) generate unconditional samples with GIDD, (2) measure PPL with a reference AR model.

```bash
# All three sampling modes, NFE=128, 3 seeds, 1000 samples each
python eval_gen_ppl.py

# Quick test
python eval_gen_ppl.py --num_samples 16 --nfes 64 --seeds 0

# SHS only
python eval_gen_ppl.py --modes shs --nfes 128 --seeds 0 1 2

# Generation only (PPL measured later)
python eval_gen_ppl.py --phase gen

# PPL only (using previously generated samples)
python eval_gen_ppl.py --phase ppl
```

### Gen-PPL Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--modes` | `ancestral adaptive shs` | Sampling methods to evaluate |
| `--nfes` | `128` | NFE (number of function evaluations / denoising steps) |
| `--seeds` | `0 1 2` | Random seeds |
| `--num_samples` | `1000` | Number of unconditional samples per mode |
| `--ppl_model` | `Qwen/Qwen2.5-7B` | Reference AR model for PPL measurement |
| `--phase` | `both` | `gen` / `ppl` / `both` |
| `--temperature` | `1.0` | Sampling temperature |
| `--model_name` | `dvruette/gidd-unif-3b` | GIDD model to evaluate |

## GSM8K Evaluation

Few-shot math reasoning benchmark evaluation.

```bash
# SHS sampling
python eval_gsm8k.py --sampling_method shs --steps 256

# Ancestral sampling (default)
python eval_gsm8k.py --sampling_method ancestral --steps 256

# Adaptive sampling with temperature
python eval_gsm8k.py --sampling_method adaptive --temperature 0.7

# Manual batch size
python eval_gsm8k.py --sampling_method shs --batch_size 4
```

### GSM8K Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sampling_method` | `ancestral` | `ancestral` / `adaptive` / `shs` |
| `--steps` | `256` | Number of denoising steps per block |
| `--temperature` | `1.0` | Sampling temperature (0.0 = greedy) |
| `--n_shots` | `8` | Number of few-shot examples (0-8) |
| `--batch_size` | auto | Batch size (auto-detected if not set) |
| `--model_name` | `dvruette/gidd-unif-3b` | GIDD model to evaluate |

## Code Structure (SHS additions)

```
gidd_easydel/model/modeling_gidd_hf.py
├── _sample_shs()              # SHS sampling (PyTorch/HuggingFace)
└── generate()                 # Updated: sampling_method="shs" option added

gidd_easydel/sampling.py
├── shs_sampling_step()        # SHS sampling (JAX, for training)
└── generate()                 # Updated: SHS state init + sampling call

eval_gen_ppl.py                # Gen-PPL evaluation (ancestral/adaptive/shs)
eval_gsm8k.py                  # GSM8K benchmark evaluation
```

For training, setup, and original evaluation scripts, see the original repository: [https://github.com/dvruette/gidd-easydel](https://github.com/dvruette/gidd-easydel)

## Citation
If you find this work useful in your research, please consider citing:

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
@article{von2025scaling,
  title={Scaling Behavior of Discrete Diffusion Language Models},
  author={von R{\"u}tte, Dimitri and Fluri, Janis and Pooladzandi, Omead and Sch{\"o}lkopf, Bernhard and Hofmann, Thomas and Orvieto, Antonio},
  journal={arXiv preprint arXiv:2512.10858},
  year={2025}
}
```

## Acknowledgements
This code is adapted from:
- **Scaling Behavior of Discrete Diffusion Language Models** [https://github.com/dvruette/gidd-easydel]

## License
This repository is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
