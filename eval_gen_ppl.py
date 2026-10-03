#!/usr/bin/env python3
"""Gen PPL and Entropy of unconditional GIDD samples (SHS paper, Table 1, 3B column).

One run = one (mode, NFE, seed): generate `--num_samples` unconditional samples (starting from BOS)
with the PyTorch model code in `gidd_easydel/model/modeling_gidd_hf.py` and the HuggingFace weights,
score them with GPT-2 Large, and write a single json with the metrics and the generated texts.

Usage:
    python eval_gen_ppl.py --mode shs --nfe 128 --seed 0
    python eval_gen_ppl.py --mode ancestral --nfe 128 --seed 0
"""

import argparse
import gc
import importlib
import json
import os
import sys
import time
import types
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────
#  GIDD model
# ──────────────────────────────────────────────

def import_gidd_modeling():
    """Import `gidd_easydel/model/modeling_gidd_hf.py` without importing the JAX package `gidd_easydel.model`."""
    package = "gidd_torch"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [os.path.join(REPO_DIR, "gidd_easydel", "model")]
        sys.modules[package] = module
    return importlib.import_module(f"{package}.modeling_gidd_hf")


def load_gidd_model(model_name, torch_dtype=torch.bfloat16, device="cuda"):
    """Load config and weights from the hub into the local `GiddForDiffusionLM` class."""
    modeling = import_gidd_modeling()

    # All weights come from the checkpoint; skip transformers' init of "missing" keys,
    # which the model's `_init_weights` does not support.
    initialize_missing_keys = PreTrainedModel._initialize_missing_keys
    PreTrainedModel._initialize_missing_keys = lambda self, *args, **kwargs: None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = modeling.GiddForDiffusionLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    finally:
        PreTrainedModel._initialize_missing_keys = initialize_missing_keys

    model = model.eval().to(device)
    return model, tokenizer


def generate_samples(
    model, tokenizer, device,
    num_samples: int, gen_batch_size: int,
    sampling_method: str, steps: int,
    max_length: int = 256, block_length: int = 128,
    temperature: float = 1.0,
) -> list[str]:
    """Generate unconditional text samples (prompt = BOS token only)."""
    all_texts = []
    num_batches = (num_samples + gen_batch_size - 1) // gen_batch_size

    for _ in tqdm(range(num_batches), desc=f"Generating ({sampling_method}, NFE={steps})"):
        batch_n = min(gen_batch_size, num_samples - len(all_texts))
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
        inputs = torch.full((batch_n, 1), bos_id, dtype=torch.long, device=device)

        generated_ids = model.generate(
            inputs=inputs,
            max_length=max_length,
            block_length=block_length,
            steps=steps,
            sampling_method=sampling_method,
            temperature=temperature,
            show_progress=False,
        )
        all_texts.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    return all_texts[:num_samples]


# ──────────────────────────────────────────────
#  Gen PPL / Entropy
# ──────────────────────────────────────────────

def load_eval_model(eval_model_name, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(eval_model_name, torch_dtype=torch.float32)
    return model.eval().to(device), tokenizer


@torch.no_grad()
def compute_gen_ppl(texts, model, tokenizer, batch_size=8, max_length=512, device="cuda"):
    """Gen PPL = exp(token-averaged NLL under the evaluator, pooled over all texts).

    Texts are right-padded and truncated to `max_length` tokens; prediction positions are
    masked with `attention_mask[:, :-1]`.
    """
    total_nll, total_tokens = 0.0, 0
    for i in tqdm(range(0, len(texts), batch_size), desc="Gen PPL"):
        enc = tokenizer(
            texts[i:i + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, :-1]
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, :-1]

        nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction="none").view_as(labels)
        total_nll += (nll * mask).sum().item()
        total_tokens += mask.sum().item()

    avg_nll = total_nll / total_tokens
    return {"gen_ppl": float(np.exp(avg_nll)), "avg_nll": float(avg_nll), "num_scored_tokens": int(total_tokens)}


def compute_entropy(texts, tokenizer):
    """Entropy (nats) of the pooled unigram distribution of the tokenizer ids of all texts
    (no special tokens, no padding, no truncation)."""
    counts = Counter()
    for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
        counts.update(ids)
    freqs = np.array(list(counts.values()), dtype=np.float64)
    probs = freqs / freqs.sum()
    return float(-(probs * np.log(probs)).sum())


def compute_metrics(texts, model, tokenizer, batch_size=8, max_length=512, device="cuda"):
    """Gen PPL and Entropy of the non-empty texts (empty / whitespace-only texts are dropped)."""
    texts = [t for t in texts if t.strip()]
    metrics = compute_gen_ppl(texts, model, tokenizer, batch_size=batch_size, max_length=max_length, device=device)
    metrics["entropy"] = compute_entropy(texts, tokenizer)
    metrics["num_scored_texts"] = len(texts)
    return metrics


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Gen PPL / Entropy of unconditional GIDD samples")

    # GIDD model
    parser.add_argument("--model_name", type=str, default="dvruette/gidd-unif-3b")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    # Generation
    parser.add_argument("--mode", type=str, default="shs", choices=["ancestral", "adaptive", "shs"],
                        help="ancestral = Standard, shs = Systematic Hazard Sampling, adaptive = confidence-based")
    parser.add_argument("--nfe", type=int, default=128, help="denoising steps per block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Evaluator
    parser.add_argument("--eval_model", type=str, default="gpt2-large")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_max_length", type=int, default=512)

    parser.add_argument("--output_dir", type=str, default="outputs/gen_ppl")
    args = parser.parse_args()

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = os.path.join(args.output_dir, f"{args.mode}_nfe{args.nfe}_seed{args.seed}.json")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Generation
    torch.set_float32_matmul_precision("high")
    model, tokenizer = load_gidd_model(args.model_name, torch_dtype, device)

    set_seed(args.seed)
    t0 = time.time()
    texts = generate_samples(
        model, tokenizer, device,
        num_samples=args.num_samples,
        gen_batch_size=args.gen_batch_size,
        sampling_method=args.mode,
        steps=args.nfe,
        max_length=args.max_length,
        block_length=args.block_length,
        temperature=args.temperature,
    )
    gen_time = time.time() - t0

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 2) Gen PPL / Entropy (evaluator in full fp32)
    torch.set_float32_matmul_precision("highest")
    eval_model, eval_tokenizer = load_eval_model(args.eval_model, device)
    metrics = compute_metrics(
        texts, eval_model, eval_tokenizer,
        batch_size=args.eval_batch_size,
        max_length=args.eval_max_length,
        device=device,
    )

    result = {
        "model_name": args.model_name,
        "mode": args.mode,
        "nfe": args.nfe,
        "seed": args.seed,
        "num_samples": len(texts),
        "gen_batch_size": args.gen_batch_size,
        "max_length": args.max_length,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "dtype": args.dtype,
        "eval_model": args.eval_model,
        **metrics,
        "gen_time_s": round(gen_time, 1),
        "texts": texts,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print(f"mode={args.mode} NFE={args.nfe} seed={args.seed}: "
          f"Gen PPL={metrics['gen_ppl']:.2f}  Entropy={metrics['entropy']:.3f}  "
          f"({metrics['num_scored_texts']}/{len(texts)} non-empty texts) -> {out_path}")


if __name__ == "__main__":
    main()
