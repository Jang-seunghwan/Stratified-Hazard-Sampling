#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import argparse
import os

import torch.multiprocessing as mp

from eval import run_mp_eval

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "config_1.3b.yaml"
)


def main(args: argparse.Namespace):
    port = 12346

    assert args.perplexity_n_samples % args.ngpus == 0
    assert args.batch_size % args.ngpus == 0

    if args.ngpus == 1:
        run_mp_eval(
            rank=0,
            world_size=1,
            seed=args.seed,
            work_dir=args.work_dir,
            pre_trained_model_path=args.pre_trained_model_path,
            batch_size=args.batch_size // args.ngpus,
            sampling_steps=args.sampling_steps,
            eval_elbo=args.eval_elbo,
            eval_perplexity=args.eval_perplexity,
            elbo_data=args.elbo_data,
            perplexity_n_samples=args.perplexity_n_samples // args.ngpus,
            port=port,
            teacher_model=args.teacher_model,
            do_dynamic_step=args.do_dynamic_step,
            use_shs=args.use_shs,
            use_dtmc=args.dtmc,
            config_path=args.config,
            cache_dir=args.cache_dir,
        )
    else:
        mp.set_start_method("forkserver")

        mp.spawn(
            run_mp_eval,
            args=(
                args.ngpus,
                args.seed,
                args.work_dir,
                args.pre_trained_model_path,
                args.batch_size // args.ngpus,
                args.sampling_steps,
                args.eval_elbo,
                args.eval_perplexity,
                args.elbo_data,
                args.perplexity_n_samples // args.ngpus,
                port,
                args.teacher_model,
                args.do_dynamic_step,
                args.use_shs,
                args.dtmc,
                args.config,
                args.cache_dir,
            ),
            nprocs=args.ngpus,
            join=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--pre_trained_model_path", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Model/flow config of the checkpoint (default: released 1.3B models).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Overrides data.cache_dir (GPT-2 tokenizer and ELBO data cache).",
    )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument(
        "--teacher_model",
        action="store_true",
        help="Evaluate the DFM (teacher) model; omit for the FS-DFM student.",
    )

    parser.add_argument("--eval_elbo", action="store_true")
    parser.add_argument("--eval_perplexity", action="store_true")
    parser.add_argument("--do_dynamic_step", action="store_true")

    # Perplexity parameters
    parser.add_argument("--sampling_steps", type=int, default=1024)
    parser.add_argument("--perplexity_n_samples", type=int, default=1024)

    # ELBO parameters
    parser.add_argument("--elbo_data", type=str, default="wikitext103")

    # Sampler
    parser.add_argument(
        "--use-shs",
        action="store_true",
        help="Use Systematic Hazard Sampling (SHS) instead of the Standard sampler.",
    )
    parser.add_argument(
        "--dtmc",
        action="store_true",
        help="Per-step jump probability p = min(h*lambda, 1) instead of 1 - exp(-h*lambda).",
    )

    args = parser.parse_args()
    main(args)
