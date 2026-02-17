#!/usr/bin/env python3
"""Generative PPL evaluation for GIDD diffusion language models.

Two-phase workflow:
  Phase 1: GIDD 모델로 각 sampling method별 1000개씩 unconditional 생성 → 저장
  Phase 2: GIDD 해제 → Gemma 2 9B 한 번 로드 → 모든 샘플에 대해 PPL/Entropy 측정

Usage:
    # 기본: ancestral/adaptive/shs 3개 × 1000샘플, Gemma 2 9B
    python eval_gen_ppl.py

    # 빠른 테스트
    python eval_gen_ppl.py --num_samples 16 --nfes 64 --seeds 0

    # 생성만 (PPL은 나중에)
    python eval_gen_ppl.py --phase gen

    # 이미 생성된 샘플로 PPL만 측정
    python eval_gen_ppl.py --phase ppl
"""

import argparse
import json
import os
import gc
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ──────────────────────────────────────────────
#  GIDD Model Loading
# ──────────────────────────────────────────────

def load_gidd_model(model_name, torch_dtype=torch.bfloat16):
    """Load GIDD model with transformers compatibility patch."""
    from transformers.modeling_utils import PreTrainedModel
    _orig = PreTrainedModel._initialize_missing_keys
    PreTrainedModel._initialize_missing_keys = lambda self, *a, **kw: None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    PreTrainedModel._initialize_missing_keys = _orig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    return model, tokenizer, device


# ──────────────────────────────────────────────
#  Phase 1: Sample Generation
# ──────────────────────────────────────────────

def generate_samples(
    model, tokenizer, device,
    num_samples: int, gen_batch_size: int,
    sampling_method: str, steps: int,
    max_length: int = 256, block_length: int = 128,
    temperature: float = 0.0,
) -> list[str]:
    """Generate unconditional text samples from GIDD model."""
    all_texts = []
    num_batches = (num_samples + gen_batch_size - 1) // gen_batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"  Generating ({sampling_method}, NFE={steps})"):
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
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_texts.extend(texts)

    return all_texts[:num_samples]


def run_generation_phase(args, torch_dtype):
    """Phase 1: Generate all samples with GIDD, then free the model."""
    print("=" * 60)
    print("  Phase 1: Sample Generation")
    print("=" * 60)

    print(f"\nLoading GIDD model: {args.model_name}")
    gidd_model, gidd_tokenizer, device = load_gidd_model(args.model_name, torch_dtype)

    # {(mode, nfe, seed): text_path}
    generated = {}

    for mode in args.modes:
        mode_dir = os.path.join(args.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        for nfe in args.nfes:
            for seed in args.seeds:
                tag = f"mode={mode}, NFE={nfe}, seed={seed}"
                text_path = os.path.join(mode_dir, f"seed{seed}_nfe{nfe}_samples.json")

                # Skip if already generated
                if os.path.exists(text_path):
                    print(f"\n[skip] {tag} — already exists: {text_path}")
                    generated[(mode, nfe, seed)] = text_path
                    continue

                print(f"\n[gen] {tag}")

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                t0 = time.time()
                texts = generate_samples(
                    gidd_model, gidd_tokenizer, device,
                    num_samples=args.num_samples,
                    gen_batch_size=args.gen_batch_size,
                    sampling_method=mode,
                    steps=nfe,
                    max_length=args.max_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                )
                gen_time = time.time() - t0

                # Save as JSON (machine-readable)
                sample_data = {
                    "mode": mode, "nfe": nfe, "seed": seed,
                    "num_samples": len(texts),
                    "gen_time_s": round(gen_time, 1),
                    "texts": texts,
                }
                with open(text_path, "w") as f:
                    json.dump(sample_data, f, ensure_ascii=False)

                generated[(mode, nfe, seed)] = text_path
                print(f"  {len(texts)} samples in {gen_time:.1f}s → {text_path}")

    # Free GIDD model completely
    del gidd_model, gidd_tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("\nGIDD model released from VRAM.")

    return generated


# ──────────────────────────────────────────────
#  Phase 2: PPL Measurement (single Gemma load)
# ──────────────────────────────────────────────

def compute_ppl_for_texts(texts, ref_model, ref_tokenizer, ppl_batch_size, max_length, device):
    """Compute PPL/Entropy metrics for a list of texts using an already-loaded reference model.

    Returns dict with aggregate metrics AND per-sample NLLs for downstream std computation.
    """
    total_acc = 0.0
    total_tokens = 0
    per_sample_nlls = []  # per-sample average NLL
    per_sample_token_counts = []  # per-sample token count

    for i in range(0, len(texts), ppl_batch_size):
        batch_texts = texts[i:i + ppl_batch_size]

        batch = ref_tokenizer(
            batch_texts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        attn_mask = batch["attention_mask"]
        logits = ref_model(
            input_ids=batch["input_ids"],
            attention_mask=attn_mask,
            use_cache=False,
        ).logits[:, :-1]

        labels = batch["input_ids"][:, 1:]
        loss_mask = attn_mask[:, :-1]

        nll = F.cross_entropy(
            logits.flatten(0, 1).float(),
            labels.flatten(0, 1),
            reduction="none",
        ).view_as(labels)

        # Per-sample NLL: average NLL per sample
        sample_nll_sum = (nll * loss_mask).sum(dim=-1)   # (batch,)
        sample_token_count = loss_mask.sum(dim=-1)        # (batch,)
        sample_avg_nll = sample_nll_sum / sample_token_count.clamp(min=1)
        per_sample_nlls.extend(sample_avg_nll.float().cpu().numpy().tolist())
        per_sample_token_counts.extend(sample_token_count.cpu().numpy().tolist())

        acc = (logits.argmax(-1) == labels).float()
        total_acc += (acc * loss_mask).sum().item()
        total_tokens += sample_token_count.sum().item()

    per_sample_nlls = np.array(per_sample_nlls)
    per_sample_token_counts = np.array(per_sample_token_counts)

    return {
        "per_sample_nlls": per_sample_nlls,
        "per_sample_token_counts": per_sample_token_counts,
        "total_acc": total_acc,
        "total_tokens": total_tokens,
    }


def aggregate_metrics(per_sample_nlls, per_sample_token_counts, total_acc, total_tokens):
    """Compute aggregate metrics from per-sample data."""
    per_sample_ppls = np.exp(per_sample_nlls)

    # Token-weighted average NLL
    weights = per_sample_token_counts / per_sample_token_counts.sum()
    avg_nll = float(np.sum(per_sample_nlls * weights))
    ppl = float(np.exp(avg_nll))
    acc = total_acc / total_tokens if total_tokens > 0 else 0.0

    return {
        "ppl": ppl,
        "avg_nll": avg_nll,
        "median_nll": float(np.median(per_sample_nlls)),
        "entropy_nats": avg_nll,
        "entropy_bits": avg_nll / np.log(2),
        "acc": float(acc),
        "tokens": int(total_tokens),
        "ppl_std": float(np.std(per_sample_ppls)),
        "nll_std": float(np.std(per_sample_nlls)),
        "num_samples": len(per_sample_nlls),
    }


def run_ppl_phase(args):
    """Phase 2: Load reference model once, measure PPL for all saved samples.

    All seeds for the same (mode, nfe) are pooled together as one big sample set.
    Stats (mean, std) are computed across individual samples, not across seeds.
    """
    print("\n" + "=" * 60)
    print("  Phase 2: PPL Measurement with", args.ppl_model)
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading reference model: {args.ppl_model}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ppl_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    ref_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model)
    if ref_tokenizer.pad_token_id is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    print("  Reference model loaded.")

    all_results = []

    for mode in args.modes:
        mode_dir = os.path.join(args.output_dir, mode)

        for nfe in args.nfes:
            tag = f"mode={mode}, NFE={nfe}"
            result_path = os.path.join(mode_dir, f"nfe{nfe}_metrics.json")

            # Skip if already measured
            if os.path.exists(result_path):
                print(f"\n[skip] {tag} — already measured")
                with open(result_path) as f:
                    all_results.append(json.load(f))
                continue

            # Collect texts from ALL seeds
            all_texts = []
            all_gen_times = []
            for seed in args.seeds:
                text_path = os.path.join(mode_dir, f"seed{seed}_nfe{nfe}_samples.json")
                if not os.path.exists(text_path):
                    print(f"\n[miss] {tag}, seed={seed} — samples not found: {text_path}")
                    continue
                with open(text_path) as f:
                    sample_data = json.load(f)
                all_texts.extend(sample_data["texts"])
                if sample_data.get("gen_time_s"):
                    all_gen_times.append(sample_data["gen_time_s"])

            if not all_texts:
                print(f"\n[miss] {tag} — no samples found for any seed")
                continue

            os.makedirs(mode_dir, exist_ok=True)
            print(f"\n[ppl] {tag} — {len(all_texts)} samples pooled from {len(args.seeds)} seed(s)")

            with torch.no_grad():
                raw = compute_ppl_for_texts(
                    all_texts, ref_model, ref_tokenizer,
                    ppl_batch_size=args.ppl_batch_size,
                    max_length=512,
                    device=device,
                )

            per_sample_nlls = raw["per_sample_nlls"]
            per_sample_tc = raw["per_sample_token_counts"]
            per_sample_ppls = np.exp(per_sample_nlls)

            # Raw metrics (all samples)
            raw_metrics = aggregate_metrics(per_sample_nlls, per_sample_tc, raw["total_acc"], raw["total_tokens"])

            # Remove the single worst outlier (highest PPL sample)
            worst_idx = int(np.argmax(per_sample_ppls))
            worst_ppl = float(per_sample_ppls[worst_idx])
            n_outliers = 1
            print(f"  [outlier] Removing worst sample: idx={worst_idx}, PPL={worst_ppl:.1f}")

            keep = np.ones(len(per_sample_ppls), dtype=bool)
            keep[worst_idx] = False
            filtered_nlls = per_sample_nlls[keep]
            filtered_tc = per_sample_tc[keep]
            filtered_total_tokens = int(filtered_tc.sum())
            filtered_acc = raw["total_acc"] * (filtered_total_tokens / raw["total_tokens"]) if raw["total_tokens"] > 0 else 0.0
            filtered_metrics = aggregate_metrics(filtered_nlls, filtered_tc, filtered_acc, filtered_total_tokens)

            result = {
                "mode": mode, "nfe": nfe,
                "seeds": args.seeds,
                "num_samples": len(all_texts),
                "gen_time_s": sum(all_gen_times) if all_gen_times else None,
                "ppl_model": args.ppl_model,
                # Raw (all samples)
                "raw_ppl": raw_metrics["ppl"],
                "raw_ppl_std": raw_metrics["ppl_std"],
                "raw_num_samples": raw_metrics["num_samples"],
                # Filtered (outliers removed)
                "n_outliers": n_outliers,
                "outlier_ppl": float(worst_ppl),
                **filtered_metrics,
            }

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

            if n_outliers > 0:
                print(f"  Raw   PPL: {raw_metrics['ppl']:.2f} ± {raw_metrics['ppl_std']:.2f}  ({raw_metrics['num_samples']} samples)")
                print(f"  Clean PPL: {filtered_metrics['ppl']:.2f} ± {filtered_metrics['ppl_std']:.2f}  ({filtered_metrics['num_samples']} samples, {n_outliers} outliers removed)")
            else:
                print(f"  PPL: {filtered_metrics['ppl']:.2f} ± {filtered_metrics['ppl_std']:.2f} | "
                      f"Entropy: {filtered_metrics['entropy_nats']:.4f} ± {filtered_metrics['nll_std']:.4f} nats | "
                      f"Acc: {filtered_metrics['acc']:.4f} (no outliers)")
            all_results.append(result)

    # Free reference model
    del ref_model, ref_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return all_results


# ──────────────────────────────────────────────
#  Summary
# ──────────────────────────────────────────────

def print_summary(all_results, output_dir):
    """Print and save summary. Stats are per-sample (across all pooled samples), not across seeds."""
    if not all_results:
        return

    # Save raw
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Table
    print(f"\n{'='*120}")
    print(f"{'Mode':<12} {'NFE':<6} {'#Samp':<7} {'Out':<5} {'RawGenPPL':<12} {'GenPPL (mean±std)':<24} {'Entropy nats (mean±std)':<28} {'Acc':<8}")
    print(f"{'='*120}")
    for r in sorted(all_results, key=lambda x: (x["mode"], x["nfe"])):
        n_out = r.get("n_outliers", 0)
        raw_ppl = r.get("raw_ppl", r["ppl"])
        ppl_str = f"{r['ppl']:>8.2f} ± {r.get('ppl_std', 0):<8.2f}"
        ent_str = f"{r['entropy_nats']:>8.4f} ± {r.get('nll_std', 0):<8.4f}"
        print(f"{r['mode']:<12} {r['nfe']:<6} {r['num_samples']:<7} {n_out:<5} {raw_ppl:<12.2f} {ppl_str:<24} {ent_str:<28} {r['acc']:<8.4f}")
    print(f"{'='*120}")

    # Save summary CSV
    csv_path = os.path.join(output_dir, "summary_table.csv")
    with open(csv_path, "w") as f:
        f.write("mode,nfe,num_samples,n_outliers,raw_ppl,ppl_mean,ppl_std,entropy_nats_mean,nll_std,entropy_bits_mean,acc\n")
        for r in sorted(all_results, key=lambda x: (x["mode"], x["nfe"])):
            f.write(f"{r['mode']},{r['nfe']},{r['num_samples']},{r.get('n_outliers', 0)},"
                    f"{r.get('raw_ppl', r['ppl']):.4f},{r['ppl']:.4f},{r.get('ppl_std', 0):.4f},"
                    f"{r['entropy_nats']:.4f},{r.get('nll_std', 0):.4f},"
                    f"{r['entropy_bits']:.4f},{r['acc']:.4f}\n")
    print(f"\nSummary CSV: {csv_path}")
    print(f"Full results: {summary_path}")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generative PPL evaluation for GIDD")

    # GIDD model
    parser.add_argument("--model_name", type=str, default="dvruette/gidd-unif-3b")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    # Generation
    parser.add_argument("--modes", type=str, nargs="+", default=["ancestral", "adaptive", "shs"],
                        choices=["ancestral", "adaptive", "shs"])
    parser.add_argument("--nfes", type=int, nargs="+", default=[128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)

    # PPL reference model
    parser.add_argument("--ppl_model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--ppl_batch_size", type=int, default=4)

    # Phase control
    parser.add_argument("--phase", type=str, default="both", choices=["both", "gen", "ppl"],
                        help="Run both phases, generation only, or PPL only")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs_gen_ppl")

    args = parser.parse_args()
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.set_float32_matmul_precision("high")

    print(f"=== Generative PPL Evaluation ===")
    print(f"GIDD Model:  {args.model_name}")
    print(f"PPL Model:   {args.ppl_model}")
    print(f"Modes:       {args.modes}")
    print(f"NFEs:        {args.nfes}")
    print(f"Seeds:       {args.seeds}")
    print(f"Samples:     {args.num_samples}")
    print(f"Phase:       {args.phase}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 1: Generate all samples
    if args.phase in ("both", "gen"):
        run_generation_phase(args, torch_dtype)

    # Phase 2: Measure PPL (Gemma loaded once)
    all_results = []
    if args.phase in ("both", "ppl"):
        all_results = run_ppl_phase(args)

    # Summary
    if all_results:
        print_summary(all_results, args.output_dir)


if __name__ == "__main__":
    main()
