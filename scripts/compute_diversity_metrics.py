"""Compute Distinct-n and Self-BLEU for generated text samples.

Works with:
  1. .pt diagnostics files (from any project, reads 'samples' key)
  2. .json files (reads 'generated_seqs' or 'samples' key)
  3. .txt files (one sentence per line)

Usage:
  python compute_diversity_metrics.py \
    --inputs path/to/diagnostics_default_T16.pt path/to/diagnostics_shs_T16.pt \
    --labels "Standard T=16" "SHS T=16" \
    --output_csv results.csv

  # Or batch mode for a project:
  python compute_diversity_metrics.py \
    --input_dir /path/to/outputs/ \
    --pattern "diagnostics_*.pt" \
    --output_csv results.csv
"""
import argparse
import glob
import json
import os
from collections import Counter
from itertools import combinations

import numpy as np
import torch
from tqdm import tqdm


# ============================================================
#  Distinct-n
# ============================================================
def compute_distinct_n(texts, n=1):
  """Compute Distinct-n: ratio of unique n-grams to total n-grams.

  Higher is better (more lexical diversity).
  """
  all_ngrams = []
  for text in texts:
    tokens = text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    all_ngrams.extend(ngrams)

  if not all_ngrams:
    return 0.0

  return len(set(all_ngrams)) / len(all_ngrams)


# ============================================================
#  Self-BLEU
# ============================================================
def _compute_bleu_single(hypothesis_tokens, reference_tokens, max_n=4):
  """Compute BLEU score of a single hypothesis against a single reference."""
  scores = []
  for n in range(1, max_n + 1):
    hyp_ngrams = Counter(
      tuple(hypothesis_tokens[i:i+n]) for i in range(len(hypothesis_tokens) - n + 1))
    ref_ngrams = Counter(
      tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens) - n + 1))

    clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
    total = sum(hyp_ngrams.values())

    if total == 0:
      scores.append(0.0)
    else:
      scores.append(clipped / total)

  # Geometric mean with brevity penalty
  if min(scores) == 0:
    return 0.0

  log_avg = sum(np.log(s) for s in scores) / len(scores)

  # Brevity penalty
  bp = min(1.0, np.exp(1 - len(reference_tokens) / max(len(hypothesis_tokens), 1)))

  return bp * np.exp(log_avg)


def _self_bleu_worker(args):
  """Worker for parallel Self-BLEU computation."""
  i, tokenized, max_n, num_refs = args
  hyp = tokenized[i]
  if len(hyp) < 2:
    return None
  n = len(tokenized)
  ref_indices = np.random.choice(
    [j for j in range(n) if j != i],
    size=min(num_refs, n - 1), replace=False)
  bleu_scores = []
  for j in ref_indices:
    ref = tokenized[j]
    if len(ref) < 2:
      continue
    bleu_scores.append(_compute_bleu_single(hyp, ref, max_n=max_n))
  return float(np.mean(bleu_scores)) if bleu_scores else None


def compute_self_bleu(texts, max_n=4, num_samples=500, num_workers=16):
  """Compute Self-BLEU: average BLEU of each sentence against all others.

  Lower is better (more diversity — sentences are less similar to each other).
  Uses multiprocessing for speed.
  """
  from multiprocessing import Pool

  tokenized = [text.split() for text in texts]
  n = len(tokenized)

  if n < 2:
    return 0.0

  if n > num_samples:
    indices = np.random.choice(n, num_samples, replace=False)
    tokenized = [tokenized[i] for i in indices]
    n = len(tokenized)

  args_list = [(i, tokenized, max_n, 50) for i in range(n)]

  with Pool(processes=num_workers) as pool:
    results = list(tqdm(
      pool.imap(_self_bleu_worker, args_list, chunksize=16),
      total=n, desc="Self-BLEU", leave=False))

  scores = [r for r in results if r is not None]
  return float(np.mean(scores)) if scores else 0.0


# ============================================================
#  Load texts from various formats
# ============================================================
def load_texts(path):
  """Load generated texts from .pt, .json, or .txt files."""
  if path.endswith('.pt'):
    data = torch.load(path, map_location='cpu', weights_only=False)
    if 'samples' in data:
      return data['samples']
    elif 'generated_seqs' in data:
      return data['generated_seqs']
    else:
      raise ValueError(f"No 'samples' or 'generated_seqs' key in {path}")
  elif path.endswith('.json'):
    with open(path) as f:
      data = json.load(f)
    if isinstance(data, list):
      return data
    if 'generated_seqs' in data:
      return data['generated_seqs']
    if 'samples' in data:
      return data['samples']
    if 'texts' in data:
      return data['texts']
    raise ValueError(f"Cannot find text list in {path}")
  elif path.endswith('.txt'):
    with open(path) as f:
      return [line.strip() for line in f if line.strip()]
  else:
    raise ValueError(f"Unsupported file type: {path}")


# ============================================================
#  Main
# ============================================================
def main():
  parser = argparse.ArgumentParser(description="Compute Distinct-n and Self-BLEU")
  parser.add_argument("--inputs", nargs="+", help="Input files (.pt, .json, .txt)")
  parser.add_argument("--labels", nargs="+", help="Labels for each input file")
  parser.add_argument("--input_dir", help="Directory to scan for files")
  parser.add_argument("--pattern", default="diagnostics_*.pt",
                      help="Glob pattern for --input_dir mode")
  parser.add_argument("--output_csv", default=None, help="Save results to CSV")
  parser.add_argument("--self_bleu_samples", type=int, default=500,
                      help="Max samples for Self-BLEU computation")
  args = parser.parse_args()

  # Collect input files
  if args.input_dir:
    files = sorted(glob.glob(os.path.join(args.input_dir, "**", args.pattern), recursive=True))
    labels = [os.path.basename(f).replace("diagnostics_", "").replace(".pt", "") for f in files]
  elif args.inputs:
    files = args.inputs
    labels = args.labels or [os.path.basename(f) for f in files]
  else:
    parser.error("Provide --inputs or --input_dir")

  if not files:
    print("No files found.")
    return

  print(f"Computing metrics for {len(files)} files...\n")

  results = []
  for path, label in zip(files, labels):
    print(f"--- {label} ({path}) ---")
    texts = load_texts(path)
    print(f"  Loaded {len(texts)} texts")

    # Filter empty
    texts = [t for t in texts if t.strip()]

    # Distinct-n
    d1 = compute_distinct_n(texts, n=1)
    d2 = compute_distinct_n(texts, n=2)
    d3 = compute_distinct_n(texts, n=3)

    # Self-BLEU
    sb = compute_self_bleu(texts, num_samples=args.self_bleu_samples)

    result = {
      'label': label,
      'file': path,
      'num_texts': len(texts),
      'distinct_1': d1,
      'distinct_2': d2,
      'distinct_3': d3,
      'self_bleu': sb,
    }
    results.append(result)
    print(f"  Distinct-1: {d1:.4f}")
    print(f"  Distinct-2: {d2:.4f}")
    print(f"  Distinct-3: {d3:.4f}")
    print(f"  Self-BLEU:  {sb:.4f}")
    print()

  # Print summary table
  print("=" * 80)
  print(f"{'Label':<40} {'D-1':>7} {'D-2':>7} {'D-3':>7} {'S-BLEU':>8} {'N':>6}")
  print("=" * 80)
  for r in results:
    print(f"{r['label']:<40} {r['distinct_1']:>7.4f} {r['distinct_2']:>7.4f} "
          f"{r['distinct_3']:>7.4f} {r['self_bleu']:>8.4f} {r['num_texts']:>6}")
  print("=" * 80)

  # Save CSV
  if args.output_csv:
    import csv
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=results[0].keys())
      writer.writeheader()
      writer.writerows(results)
    print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
  main()
