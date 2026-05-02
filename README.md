# SHS on FS-DFM / DFM

This repository applies **Stratified Hazard Sampling (SHS)** to the
[FS-DFM](https://arxiv.org/abs/2509.20624) discrete flow matching codebase
from Apple. Training code is unchanged; only the inference-time sampling rule
is swapped to measure the effect of SHS on generation quality (Gen. PPL).

## Paper

- **Stratified Hazard Sampling: Minimal-Variance Event Scheduling for
  CTMC/DTMC Discrete Diffusion and Flow Models**
- Authors: Seunghwan Jang, SooJean Han
- arXiv: <https://arxiv.org/abs/2601.02799>

## What This Branch Contains

| Component | Description |
|-----------|-------------|
| `flow_matching/solver/discrete_solver.py` | Original Meta flow-matching Euler solver |
| `flow_matching/solver/discrete_solver_fsdfm.py` | FS-DFM solvers + **`MixtureDiscreteEulerSolverSHS`** |
| `fs_dfm/` | FS-DFM evaluation & training code (Apple) |
| `pre_training/` | DFM pre-training code (Apple) |

### SHS Integration

SHS is implemented as a drop-in solver that replaces the standard
Euler sampler's independent Bernoulli/Poisson jump decisions with
stratified event scheduling in cumulative hazard space.

### Three Independent Axes

Experiments are controlled by three orthogonal choices:

| Axis | Options | Flag |
|------|---------|------|
| **Model** | DFM (teacher) / FS-DFM (student) | `--teacher_model` (DFM) or omit (FS-DFM) |
| **Probability** | CTMC / DTMC | `--dtmc` (DTMC) or omit (CTMC) |
| **Sampling rule** | Standard / SHS | `--use-shs` (SHS) or omit (Standard) |

**Probability mode** controls how per-step jump probability is computed:
- **CTMC** (default): `p_jump = 1 - exp(-h * lambda)` — exact CTMC holding-time probability
- **DTMC**: `p_jump = min(h * lambda, 1)` — tau-leap linear approximation (larger p_jump)

**Sampling rule** controls how p_jump is used:
- **Standard** (default): independent Bernoulli/Poisson per step
- **SHS**: stratified cumulative hazard with single random phase per position

### Usage Examples (2 x 2 x 2 = 8 combinations)

```bash
COMMON="--ngpus 1 --perplexity_n_samples 320 --eval_perplexity --sampling_steps 32"

# DFM + Standard + CTMC (baseline)
python fs_dfm/run_eval.py $COMMON --teacher_model \
    --pre_trained_model_path checkpoints/DFM_checkpoint.pth

# DFM + SHS + DTMC
python fs_dfm/run_eval.py $COMMON --teacher_model --use-shs --dtmc \
    --pre_trained_model_path checkpoints/DFM_checkpoint.pth

# FS-DFM + Standard + DTMC
python fs_dfm/run_eval.py $COMMON --dtmc \
    --pre_trained_model_path checkpoints/FSDFM_checkpoint.pth

# FS-DFM + SHS + DTMC
python fs_dfm/run_eval.py $COMMON --use-shs --dtmc \
    --pre_trained_model_path checkpoints/FSDFM_checkpoint.pth
```

### Solver Registry

| Name | Sampling | Probability | Class |
|------|----------|-------------|-------|
| `mixture_euler` | Standard | CTMC | `MixtureDiscreteEulerSolver` |
| `mixture_euler_dtmc` | Standard | DTMC | `MixtureDiscreteEulerSolver_DTMC` |
| `mixture_euler_shs` | SHS | CTMC | `MixtureDiscreteEulerSolverSHS` |
| `mixture_euler_shs_dtmc` | SHS | DTMC | `MixtureDiscreteEulerSolverSHS_DTMC` |

## Setup

```bash
conda env create -f fsdfm_environment.yml
conda activate FSDFM
pip install -e .
```

## Checkpoints

Pretrained checkpoints from Apple ML Research:

| Model | Size | Source | Notes | URL |
|-------|-----:|--------|-------|-----|
| FS-DFM | 1.3B | uniform | RK4 teacher distilled | [Download](https://ml-site.cdn-apple.com/models/fs-dfm/checkpoint_step_190000.pth) |
| DFM | 1.3B | uniform | DFM pretrained initialization | [Download](https://ml-site.cdn-apple.com/models/fs-dfm/checkpoint.pth) |

## Project Structure

```
.
├── flow_matching/
│   └── solver/
│       ├── discrete_solver.py            # Original Meta Euler solver
│       └── discrete_solver_fsdfm.py      # FS-DFM solvers + SHS
├── fs_dfm/
│   ├── configs/config.yaml               # Hydra configuration
│   ├── run_eval.py                       # Evaluation entry point (--use-shs)
│   ├── eval.py                           # Evaluation orchestration
│   └── logic/
│       └── generate.py                   # Sample generation (solver selection)
└── pre_training/                         # DFM pre-training utilities
```

## Acknowledgements

This code is adapted from:
- **FS-DFM** (Apple): <https://github.com/apple/ml-fs-dfm>
- **Flow Matching** (Meta): <https://github.com/facebookresearch/flow_matching>

## Citation

```bibtex
@misc{jang2026stratifiedhazardsamplingminimalvariance,
  title         = {Stratified Hazard Sampling: Minimal-Variance Event Scheduling
                   for CTMC/DTMC Discrete Diffusion and Flow Models},
  author        = {Seunghwan Jang and SooJean Han},
  year          = {2026},
  eprint        = {2601.02799},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2601.02799},
}

@article{fsdfm2025,
  title   = {FS-DFM: Fast and Accurate Long Text Generation with
             Few-Step Diffusion Language Models},
  author  = {Amin Karimi Monsefi and Nikhil Bhendawade and
             Manuel Rafael Ciosici and Dominic Culver and
             Yizhe Zhang and Irina Belousova},
  year    = {2025},
}
```

## License

See [LICENSE](./LICENSE) for the Apple license terms.
