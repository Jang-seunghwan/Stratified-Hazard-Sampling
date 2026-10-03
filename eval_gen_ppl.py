from gidd import GiddPipeline
import torch
import torch.nn.functional as F
import os
import glob
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="GIDD sampling: Generative PPL and Entropy (Standard vs. SHS)")
    parser.add_argument("--mode", type=str, default="default", choices=["default", "shs"],
                        help="Sampling mode: 'default' (Standard) or 'shs' (Systematic Hazard Sampling)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Seeds to run (e.g., --seeds 1 2 3 4 5)")
    parser.add_argument("--nfes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256, 512],
                        help="NFE (num_inference_steps) values to run (e.g., --nfes 64 128 256)")
    parser.add_argument("--num_samples", type=int, default=1024,
                        help="Number of samples to generate per configuration")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation and PPL computation")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for results")
    parser.add_argument("--model", type=str, default="dvruette/gidd-base-p_unif-0.2",
                        help="HuggingFace model name or path")
    parser.add_argument("--self_correction", action="store_true",
                        help="Run the GIDD self-correction step on the samples before evaluation (off in the paper)")

    # Generative PPL options
    parser.add_argument("--ppl_model", type=str, default="gpt2-large",
                        help="Model for computing generative PPL (e.g., gpt2-large, google/gemma-2-9b)")
    parser.add_argument("--no_ppl", action="store_true",
                        help="Skip generative PPL computation")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_generative_ppl(texts, ppl_model_name, batch_size=4, device="cuda"):
    """Compute generative PPL using a reference LM."""
    print(f"\nLoading PPL model: {ppl_model_name}")
    model = AutoModelForCausalLM.from_pretrained(ppl_model_name, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_nll = 0
    total_acc = 0
    total_tokens = 0
    all_nlls = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing PPL"):
            batch_texts = texts[i:i + batch_size]
            batch = tokenizer(batch_texts, padding=True, return_tensors="pt", truncation=True, max_length=512)
            batch = {k: v.to(device) for k, v in batch.items()}

            attn_mask = batch["attention_mask"]
            logits = model(input_ids=batch["input_ids"], attention_mask=attn_mask, use_cache=False).logits[:, :-1]

            labels = batch["input_ids"][:, 1:]
            loss_mask = attn_mask[:, :-1]

            nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction='none').view_as(labels)
            all_nlls.extend(nll[loss_mask == 1].float().cpu().numpy().tolist())
            total_nll += (nll * loss_mask).sum().item()

            acc = (logits.argmax(-1) == labels).float()
            total_acc += (acc * loss_mask).sum().item()
            total_tokens += loss_mask.sum().item()

    avg_nll = total_nll / total_tokens
    ppl = np.exp(avg_nll)
    acc = total_acc / total_tokens

    # Clean up
    del model
    torch.cuda.empty_cache()

    return {
        "ppl": ppl,
        "avg_nll": avg_nll,
        "median_nll": float(np.median(all_nlls)),
        "acc": acc,
        "tokens": total_tokens,
    }


def compute_entropy(texts, tokenizer, max_length=512):
    """Pooled unigram entropy (nats) of all samples of a run.

    The texts are re-tokenized with the GIDD tokenizer (no special tokens, truncated/padded to
    `max_length`), pad tokens are dropped, and the entropy of the token frequencies is computed
    over all remaining tokens.
    """
    ids = tokenizer(texts, return_tensors="pt", add_special_tokens=False, max_length=max_length,
                    padding="max_length", truncation=True)["input_ids"].reshape(-1)
    ids = ids[ids != tokenizer.pad_token_id]
    _, counts = torch.unique(ids, return_counts=True)
    return float(torch.special.entr(counts.float() / counts.sum()).sum())


def write_summary(output_dir):
    """Collect the metrics (without texts) of all runs in `output_dir` into summary.json."""
    runs = []
    for path in glob.glob(f"{output_dir}/seed*_nfe*.json"):
        with open(path) as f:
            run = json.load(f)
        run.pop("generated_seqs", None)
        runs.append(run)
    runs.sort(key=lambda r: (r["nfe"], r["seed"]))
    summary_path = f"{output_dir}/summary.json"
    with open(summary_path, "w") as f:
        json.dump(runs, f, indent=2)
    return summary_path, runs


def main():
    args = parse_args()

    use_shs = (args.mode == "shs")
    mode_str = "shs" if use_shs else "default"

    # Output directory with mode suffix
    output_dir = f"{args.output_dir}/{mode_str}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== GIDD Experiment ===")
    print(f"Mode: {mode_str}")
    print(f"Seeds: {args.seeds}")
    print(f"NFEs: {args.nfes}")
    print(f"Samples per config: {args.num_samples}")
    print(f"Self-correction: {args.self_correction}")
    print(f"PPL Model: {args.ppl_model}")
    print(f"Output: {output_dir}/")
    print()

    # Load GIDD model (once)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = GiddPipeline.from_pretrained(args.model, trust_remote_code=True)
    pipe.to(device)

    # Run experiments: seed x NFE
    for seed in args.seeds:
        for nfe in args.nfes:
            print(f"\n{'='*60}")
            print(f"Running: mode={mode_str}, seed={seed}, NFE={nfe}")
            print(f"{'='*60}")

            set_seed(seed)

            # Generate samples (in batches)
            all_texts = []
            num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
            for batch_idx in range(num_batches):
                batch_samples = min(args.batch_size, args.num_samples - len(all_texts))
                texts = pipe.generate(
                    num_samples=batch_samples,
                    num_inference_steps=nfe,
                    use_shs=use_shs,
                    show_progress=(batch_idx == 0)
                )
                all_texts.extend(texts)

            print(f"Generated {len(all_texts)} samples")

            # Self-correction (optional, not used in the paper)
            if args.self_correction:
                print("Running self-correction...")
                corrected_texts = []
                for i in range(0, len(all_texts), args.batch_size):
                    batch = all_texts[i:i + args.batch_size]
                    corrected = pipe.self_correction(
                        batch,
                        num_inference_steps=nfe,
                        early_stopping=True,
                        temperature=0.1,
                        show_progress=(i == 0)
                    )
                    corrected_texts.extend(corrected)
            else:
                corrected_texts = all_texts

            result = {
                "mode": mode_str,
                "seed": seed,
                "nfe": nfe,
                "num_samples": len(corrected_texts),
                "self_correction": args.self_correction,
            }

            # Generative PPL
            if not args.no_ppl:
                ppl_metrics = compute_generative_ppl(
                    corrected_texts,
                    args.ppl_model,
                    batch_size=args.batch_size,
                    device=device
                )
                result.update(ppl_metrics)
                print(f"\n>>> PPL: {ppl_metrics['ppl']:.2f} | NLL: {ppl_metrics['avg_nll']:.4f} | Acc: {ppl_metrics['acc']:.4f}")

            # Entropy
            result["entropy"] = compute_entropy(corrected_texts, pipe.tokenizer)
            print(f">>> Entropy: {result['entropy']:.4f}")

            result["generated_seqs"] = corrected_texts

            result_path = f"{output_dir}/seed{seed}_nfe{nfe}.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved: {result_path}")

    # Metrics of all runs in the output directory
    summary_path, runs = write_summary(output_dir)
    print(f"\n{'='*60}")
    print(f"Summary of all runs in {output_dir}: {summary_path}")
    print(f"{'='*60}")
    print(f"{'Mode':<10} {'Seed':<6} {'NFE':<6} {'PPL':<10} {'NLL':<10} {'Acc':<10} {'Entropy':<10}")
    print(f"{'='*60}")
    for r in runs:
        ppl = f"{r['ppl']:<10.2f} {r['avg_nll']:<10.4f} {r['acc']:<10.4f}" if "ppl" in r else f"{'-':<10} {'-':<10} {'-':<10}"
        ent = f"{r['entropy']:<10.4f}" if "entropy" in r else "-"
        print(f"{r['mode']:<10} {r['seed']:<6} {r['nfe']:<6} {ppl} {ent}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
