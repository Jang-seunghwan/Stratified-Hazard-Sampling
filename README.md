# SHS Sampling for Discrete Diffusion (UDLM)

This repository modifies UDLM-based discrete diffusion models to apply
**Stratified Hazard Sampling (SHS)** only during **evaluation (gen-PPL)**.
Training code remains unchanged; only the evaluation sampling path is
switched to SHS to measure gen-PPL.

## Key Features
- **Three sampling modes**: `default` (tau-leap), `shs` (SHS), `shs_safe` (SHS-Safe)
- **Blacklist-based safe-word generation**: Block specific tokens during sampling
- **HuggingFace model support**: Use pre-trained models from HuggingFace Hub
- SHS can be enabled via `sample(sampling_mode='shs')` interface

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

## Gen-PPL Evaluation

### Using HuggingFace Pre-trained Model (Recommended)

```bash
cd /data1/seunghwan/discrete-diffusion-guidance

# Basic SHS sampling
SAMPLING_MODE=shs bash scripts/eval_lm1b_gen_ppl_hf.sh

# Default (tau-leap) sampling
SAMPLING_MODE=default bash scripts/eval_lm1b_gen_ppl_hf.sh

# SHS-Safe with blacklist
SAMPLING_MODE=shs_safe BLACKLIST_PERCENT=10 BLACKLIST_SEED=42 bash scripts/eval_lm1b_gen_ppl_hf.sh
```

### Using Local Checkpoint

```bash
cd /data1/seunghwan/discrete-diffusion-guidance
MODEL=udlm SAMPLING_STEPS=128 bash scripts/eval_lm1b_gen_ppl.sh
```

## Sampling Modes

| Mode | Description |
|------|-------------|
| `default` | Standard tau-leap sampling (original UDLM) |
| `shs` | Stratified Hazard Sampling (variance reduction) |
| `shs_safe` | SHS with blacklist-aware lambda allocation |

## Safe-Word Generation (Blacklist)

For research on controlled text generation with vocabulary constraints.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLING_MODE` | `shs` | `default` / `shs` / `shs_safe` |
| `BLACKLIST_PERCENT` | `0` | Percentage of vocabulary to blacklist (0-100) |
| `BLACKLIST_SEED` | `42` | Seed for random blacklist generation (reproducibility) |
| `BLACKLIST_WORDS` | `""` | Comma-separated list of specific words to ban |
| `SAMPLING_STEPS` | `128` | Number of sampling steps |
| `SEED` | `1` | General random seed |

### How Blacklist Works

#### Random Blacklist (Percentage-Based)

1. **Blacklist Generation**: Randomly select N% of vocabulary as blacklist
   - Special tokens (PAD, BOS, EOS, CLS, SEP, MASK) are excluded
   - Use `BLACKLIST_SEED` to fix the blacklist across experiments

#### Word-Based Blacklist (Specific Words)

1. **Word Mapping**: Each word is converted to subword tokens using BERT tokenizer
   - Example: "uncomfortable" → `['un', '##com', '##for', '##table']`
   - All subword tokens are added to blacklist

2. **Token Resolution**: Script shows exactly which tokens are banned:
   ```
   [Blacklist] Word-based: 3 words -> 8 tokens blacklisted
     - 'hate' -> token 5765 ('hate')
     - 'kill' -> token 4590 ('kill')
     - 'violence' -> token 5996 ('violence')
   ```

#### Combined Blacklist

Both methods can be combined - all tokens from both will be blacklisted:
```bash
BLACKLIST_PERCENT=5 BLACKLIST_WORDS="hate,kill" bash scripts/eval_lm1b_gen_ppl_hf.sh
```

### Blacklist Effect on Sampling

1. **Initial Noise**: Blacklisted tokens are excluded from initial sampling

2. **Probability Scaling**: At each step, blacklisted token probabilities are zeroed and safe token probabilities are scaled by `p(V)/p(V_safe)` to preserve total transition rate

### SHS-Safe Mode

The `shs_safe` mode extends SHS with blacklist-aware lambda allocation:

1. Pre-sample all lambdas (jump thresholds) and sort in ascending order
2. At each step, compute `p(V_black)/p(V)` for each token position
3. Assign smaller lambdas to positions with higher blacklist affinity
4. **Deferred allocation**: Only assign lambdas up to the last jumping token; defer the rest

This encourages "dangerous" tokens (high blacklist probability) to jump first, giving them more opportunities to transition away from blacklisted states.

### Apple-to-Apple Comparison

Fix `BLACKLIST_SEED` to compare different sampling seeds with the same blacklist:

```bash
# Same blacklist (seed=42), different sampling seeds
SAMPLING_MODE=shs_safe BLACKLIST_PERCENT=10 BLACKLIST_SEED=42 SEED=1 bash scripts/eval_lm1b_gen_ppl_hf.sh
SAMPLING_MODE=shs_safe BLACKLIST_PERCENT=10 BLACKLIST_SEED=42 SEED=2 bash scripts/eval_lm1b_gen_ppl_hf.sh
SAMPLING_MODE=shs_safe BLACKLIST_PERCENT=10 BLACKLIST_SEED=42 SEED=3 bash scripts/eval_lm1b_gen_ppl_hf.sh
```

### Word-Based Blacklist Examples

Ban specific offensive words from generation:

```bash
# Ban specific words
SAMPLING_MODE=shs_safe BLACKLIST_WORDS="hate,kill,violence" bash scripts/eval_lm1b_gen_ppl_hf.sh

# Combine random blacklist + specific words
SAMPLING_MODE=shs_safe BLACKLIST_PERCENT=5 BLACKLIST_WORDS="hate,kill" bash scripts/eval_lm1b_gen_ppl_hf.sh

# Multiple offensive terms
SAMPLING_MODE=shs_safe BLACKLIST_WORDS="hate,violence,abuse,harm,attack" bash scripts/eval_lm1b_gen_ppl_hf.sh
```

Note: Subword tokenization means banning "hate" will also affect "hated", "hatred", etc. if they share subword tokens.

## Code Structure

```
diffusion.py
├── _create_blacklist()             # Random N% blacklist from vocab
├── _create_blacklist_from_words()  # Word-based blacklist (specific words)
├── _apply_blacklist_to_probs()     # Zero blacklist, scale safe probs
├── _sample_prior()                 # Initial noise (blacklist-aware)
├── sample()                        # Main entry (supports sampling_mode + blacklist)
├── _diffusion_sample()             # Default tau-leap sampling
├── _diffusion_sample_shs()         # SHS sampling
└── _diffusion_sample_shs_safe()    # SHS-Safe with deferred lambda

main.py
└── _gen_ppl_eval()                 # Gen-PPL evaluation entry point

scripts/
├── eval_lm1b_gen_ppl.sh            # Local checkpoint evaluation
└── eval_lm1b_gen_ppl_hf.sh         # HuggingFace model evaluation
```

## Output Format

Results are saved as JSON:

```json
{
  "generative_ppl": 12.345,
  "entropy": 5.678,
  "generated_seqs": ["sample1...", "sample2...", ...]
}
```

Output filename includes sampling configuration:
- Without blacklist: `samples-lm1b-mode-{MODE}_T-{STEPS}_seed-{SEED}.json`
- With blacklist: `samples-lm1b-mode-{MODE}_bl-{PERCENT}pct_blseed-{BLSEED}_T-{STEPS}_seed-{SEED}.json`

## Acknowledgements
This code is adapted from:
- **Simple Guidance Mechanisms for Discrete Diffusion Models** [https://github.com/kuleshov-group/discrete-diffusion-guidance]

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
