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
stratified event scheduling in cumulative hazard space. The solver is
selectable at evaluation time via `--use-shs`:

```bash
# Baseline (standard Euler)
python fs_dfm/run_eval.py \
    --work_dir results/baseline \
    --ngpus 1 \
    --perplexity_n_samples 320 \
    --eval_perplexity \
    --pre_trained_model_path DFM_checkpoint.pth \
    --teacher_model

# SHS
python fs_dfm/run_eval.py \
    --work_dir results/shs \
    --ngpus 1 \
    --perplexity_n_samples 320 \
    --eval_perplexity \
    --pre_trained_model_path DFM_checkpoint.pth \
    --teacher_model \
    --use-shs
```

### Solver Registry

| Name | Class | Description |
|------|-------|-------------|
| `mixture_euler` | `MixtureDiscreteEulerSolver` | Standard Poisson-jump Euler (baseline) |
| `mixture_euler_shs` | `MixtureDiscreteEulerSolverSHS` | **Stratified Hazard Sampling** |
| `mixture_euler_with_cumulative_scalar` | `MixtureDiscreteEleurSolverWithCumulativeScalar` | Cumulative-scalar variant |

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
