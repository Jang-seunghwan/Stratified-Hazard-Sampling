# SHS Sampling for Discrete Diffusion (UDLM)

This repository modifies UDLM-based discrete diffusion models to apply
**Stratified Hazard Sampling (SHS)** only during **evaluation (gen-PPL)**.
Training code remains unchanged; only the evaluation sampling path is
switched to SHS to measure gen-PPL.

Key Features
- SHS is enabled only during **evaluation (gen-PPL)**.
- The original sampling path (`_diffusion_sample`) is preserved.
- SHS can be enabled via the `sample(use_shs=True)` interface.
- Currently, SHS is applied automatically when `mode=gen_ppl_eval`.

## Paper
- Title: **Stratified Hazard Sampling: Minimal-Variance Event Scheduling for CTMC/DTMC Discrete Diffusion and Flow Models**
- Authors: **Seunghwan Jang, SooJean Han**
- arXiv: **https://arxiv.org/abs/2601.02799**

## Quickstart
```bash
conda env create -f requirements.yaml
conda activate discdiff
mkdir -p outputs watch_folder
```

## Gen-PPL Evaluation (LM1B + UDLM)

### Run
```bash
cd /discrete-diffusion-guidance
MODEL=udlm SAMPLING_STEPS=128 bash scripts/eval_lm1b_gen_ppl.sh
```

- In `mode=gen_ppl_eval`, `main.py` calls `sample(use_shs=True)`,
  which uses the `_diffusion_sample_shs` path.
- The output JSON path is defined by `generated_seqs_path` in
  `scripts/eval_lm1b_gen_ppl.sh`.

## SHS Integration (Summary)
- `diffusion.py`: add `sample(..., use_shs=False)` and branch by flag
- `main.py`: call `sample(use_shs=True)` in `mode=gen_ppl_eval`

If needed, you can extend this to a config flag (e.g., `sampling.use_shs`)
to enable SHS in other evaluation paths.

## Code Structure
- `diffusion.py`: sampling routines (`_diffusion_sample`, `_diffusion_sample_shs`)
- `main.py`: gen-PPL evaluation entry point
- `scripts/`: run scripts for experiments/evaluation
- `configs/`: hyperparameters and dataset settings

## Acknowledgements
This code is adapted from:
- **Simple Guidance Mechanisms for Discrete Diffusion Models**<br>
[https://github.com/kuleshov-group/discrete-diffusion-guidance]

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
