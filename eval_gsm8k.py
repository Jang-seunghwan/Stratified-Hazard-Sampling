#!/usr/bin/env python3
"""GSM8K evaluation script for GIDD diffusion language models.

Supports both SHS (Stratified Hazard Sampling) and standard (ancestral/adaptive) sampling,
with automatic VRAM-aware batch sizing.

Usage:
    # Ancestral sampling (default)
    python eval_gsm8k.py --sampling_method ancestral --steps 256

    # SHS sampling
    python eval_gsm8k.py --sampling_method shs --steps 256

    # Adaptive sampling with temperature
    python eval_gsm8k.py --sampling_method adaptive --temperature 0.7

    # Manual batch size (skip auto-detection)
    python eval_gsm8k.py --batch_size 4

    # 8-shot evaluation
    python eval_gsm8k.py --n_shots 8
"""

import argparse
import json
import re
import time
import gc
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ──────────────────────────────────────────────
#  GSM8K Answer Extraction
# ──────────────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from GSM8K format.

    GSM8K answers end with '#### <number>'.
    Generated answers may contain the number after #### or as the last number in text.
    """
    # Try to find #### pattern first
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: find the last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def normalize_answer(answer: str | None) -> str | None:
    """Normalize numeric answer for comparison."""
    if answer is None:
        return None
    answer = answer.replace(",", "").strip()
    try:
        val = float(answer)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return answer


# ──────────────────────────────────────────────
#  Few-Shot Prompt Construction
# ──────────────────────────────────────────────

GSM8K_FEWSHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 2 + 2 = 4 more toys. 5 + 4 = 9. #### 9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. #### 29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### 8",
    },
]


def build_prompt(question: str, n_shots: int = 8) -> str:
    """Build a few-shot prompt for GSM8K evaluation."""
    examples = GSM8K_FEWSHOT_EXAMPLES[:n_shots]

    parts = []
    for ex in examples:
        parts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")

    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────
#  VRAM-Aware Batch Size Detection
# ──────────────────────────────────────────────

def get_free_vram_mb() -> float:
    """Get free VRAM in MB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return free / (1024 ** 2)


def find_max_batch_size(
    model,
    tokenizer,
    max_length: int,
    block_length: int,
    steps: int,
    sampling_method: str,
    temperature: float,
    max_batch: int = 32,
    test_prompt: str = "Question: What is 2 + 2?\nAnswer:",
) -> int:
    """Find maximum batch size via binary search with OOM detection.

    Tries progressively smaller batch sizes until one works without OOM.
    """
    device = next(model.parameters()).device

    # Start from max_batch and halve on OOM
    batch = max_batch
    while batch >= 1:
        torch.cuda.empty_cache()
        gc.collect()

        try:
            inputs = tokenizer(
                [test_prompt] * batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).input_ids.to(device)

            _ = model.generate(
                inputs=inputs,
                max_length=max_length,
                block_length=block_length,
                steps=steps,
                sampling_method=sampling_method,
                temperature=temperature,
                show_progress=False,
            )

            torch.cuda.empty_cache()
            gc.collect()
            print(f"  Batch size {batch}: OK (free VRAM: {get_free_vram_mb():.0f} MB)")
            return batch

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  Batch size {batch}: OOM")
            batch //= 2

    return 1


# ──────────────────────────────────────────────
#  Main Evaluation
# ──────────────────────────────────────────────

def evaluate_gsm8k(
    model_name: str,
    sampling_method: str = "ancestral",
    steps: int = 256,
    block_length: int = 128,
    max_length: int = 512,
    temperature: float = 0.0,
    n_shots: int = 8,
    batch_size: int | None = None,
    max_batch: int = 16,
    num_samples: int | None = None,
    output_file: str | None = None,
    dtype: str = "bf16",
    show_progress: bool = True,
):
    """Run GSM8K evaluation."""

    # ── Load model & tokenizer ──
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # The remote model's _init_weights has bugs (references non-existent config.initializer_range
    # and module.weight on container modules). Since we're loading pretrained weights,
    # skip the re-initialization step entirely.
    from transformers.modeling_utils import PreTrainedModel
    _orig_init_missing = PreTrainedModel._initialize_missing_keys
    PreTrainedModel._initialize_missing_keys = lambda self, *a, **kw: None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    PreTrainedModel._initialize_missing_keys = _orig_init_missing

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    print(f"  Device: {device}, dtype: {torch_dtype}")

    if torch.cuda.is_available():
        print(f"  Free VRAM: {get_free_vram_mb():.0f} MB")

    # ── Load GSM8K ──
    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    print(f"  {len(dataset)} examples")

    # ── Determine batch size ──
    if batch_size is None:
        print(f"\nAuto-detecting batch size (sampling_method={sampling_method}, steps={steps})...")
        batch_size = find_max_batch_size(
            model, tokenizer,
            max_length=max_length,
            block_length=block_length,
            steps=steps,
            sampling_method=sampling_method,
            temperature=temperature,
            max_batch=max_batch,
        )
    print(f"  Using batch_size={batch_size}")

    # ── Build prompts ──
    prompts = []
    gold_answers = []
    for example in dataset:
        question = example["question"]
        answer_text = example["answer"]
        gold = extract_answer(answer_text)
        prompts.append(build_prompt(question, n_shots=n_shots))
        gold_answers.append(normalize_answer(gold))

    # ── Run inference ──
    print(f"\nRunning GSM8K evaluation...")
    print(f"  sampling_method={sampling_method}, steps={steps}, temperature={temperature}")
    print(f"  block_length={block_length}, max_length={max_length}, n_shots={n_shots}")

    all_predictions = []
    correct = 0
    total = 0
    t_start = time.time()

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_gold = gold_answers[batch_start:batch_end]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).input_ids.to(device)

        # Generate
        try:
            generated_ids = model.generate(
                inputs=inputs,
                max_length=max_length,
                block_length=block_length,
                steps=steps,
                sampling_method=sampling_method,
                temperature=temperature,
                show_progress=False,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  OOM at batch [{batch_start}:{batch_end}], falling back to batch_size=1")
            # Fallback: process one by one
            for i in range(len(batch_prompts)):
                single_input = tokenizer(
                    batch_prompts[i],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).input_ids.to(device)
                single_out = model.generate(
                    inputs=single_input,
                    max_length=max_length,
                    block_length=block_length,
                    steps=steps,
                    sampling_method=sampling_method,
                    temperature=temperature,
                    show_progress=False,
                )
                gen_text = tokenizer.decode(single_out[0], skip_special_tokens=True)
                # Extract only the generated part after the prompt
                prompt_text = batch_prompts[i]
                if gen_text.startswith(prompt_text):
                    gen_text = gen_text[len(prompt_text):]

                pred = normalize_answer(extract_answer(gen_text))
                is_correct = pred == batch_gold[i]
                correct += int(is_correct)
                total += 1
                all_predictions.append({
                    "idx": batch_start + i,
                    "pred": pred,
                    "gold": batch_gold[i],
                    "correct": is_correct,
                    "generated": gen_text[:500],
                })
            continue

        # Decode and evaluate batch
        for i in range(len(batch_prompts)):
            gen_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)

            # Extract only the generated part
            prompt_text = batch_prompts[i]
            if gen_text.startswith(prompt_text):
                gen_text = gen_text[len(prompt_text):]

            pred = normalize_answer(extract_answer(gen_text))
            is_correct = pred == batch_gold[i]
            correct += int(is_correct)
            total += 1

            all_predictions.append({
                "idx": batch_start + i,
                "pred": pred,
                "gold": batch_gold[i],
                "correct": is_correct,
                "generated": gen_text[:500],
            })

        elapsed = time.time() - t_start
        acc = correct / total if total > 0 else 0
        if show_progress:
            print(f"  [{total}/{len(prompts)}] Accuracy: {acc:.4f} ({correct}/{total}) | {elapsed:.1f}s")

    # ── Results ──
    elapsed = time.time() - t_start
    final_acc = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"GSM8K Results")
    print(f"{'='*60}")
    print(f"  Model:           {model_name}")
    print(f"  Sampling Method: {sampling_method}")
    print(f"  Steps:           {steps}")
    print(f"  Temperature:     {temperature}")
    print(f"  Block Length:    {block_length}")
    print(f"  Max Length:      {max_length}")
    print(f"  N-shots:         {n_shots}")
    print(f"  Batch Size:      {batch_size}")
    print(f"  Dtype:           {dtype}")
    print(f"{'='*60}")
    print(f"  Accuracy:        {final_acc:.4f} ({correct}/{total})")
    print(f"  Time:            {elapsed:.1f}s ({elapsed/total:.2f}s/example)")
    print(f"{'='*60}")

    # ── Save results ──
    if output_file is None:
        output_file = f"gsm8k_{sampling_method}_steps{steps}.json"

    results = {
        "model": model_name,
        "sampling_method": sampling_method,
        "steps": steps,
        "temperature": temperature,
        "block_length": block_length,
        "max_length": max_length,
        "n_shots": n_shots,
        "batch_size": batch_size,
        "dtype": dtype,
        "accuracy": final_acc,
        "correct": correct,
        "total": total,
        "elapsed_seconds": elapsed,
        "predictions": all_predictions,
    }

    output_path = Path(output_file)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {output_path}")

    return final_acc


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GSM8K evaluation for GIDD diffusion LM")
    parser.add_argument("--model_name", type=str, default="dvruette/gidd-unif-3b",
                        help="HuggingFace model name or local path")
    parser.add_argument("--sampling_method", type=str, default="ancestral",
                        choices=["ancestral", "adaptive", "shs"],
                        help="Sampling method: ancestral, adaptive, or shs")
    parser.add_argument("--steps", type=int, default=256,
                        help="Number of denoising steps per block")
    parser.add_argument("--block_length", type=int, default=128,
                        help="Block length for generation")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length for generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (1.0 = raw posterior, 0.0 = greedy)")
    parser.add_argument("--n_shots", type=int, default=8,
                        help="Number of few-shot examples (0-8)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (None = auto-detect)")
    parser.add_argument("--max_batch", type=int, default=16,
                        help="Max batch size for auto-detection")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of test examples (None = all 1319)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Model dtype")
    parser.add_argument("--no_progress", action="store_true",
                        help="Disable progress output")

    args = parser.parse_args()

    evaluate_gsm8k(
        model_name=args.model_name,
        sampling_method=args.sampling_method,
        steps=args.steps,
        block_length=args.block_length,
        max_length=args.max_length,
        temperature=args.temperature,
        n_shots=args.n_shots,
        batch_size=args.batch_size,
        max_batch=args.max_batch,
        num_samples=args.num_samples,
        output_file=args.output,
        dtype=args.dtype,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
