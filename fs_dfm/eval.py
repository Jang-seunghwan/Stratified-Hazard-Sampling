#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn

from data import data
from flow_matching.loss import MixturePathGeneralizedKL

from logic import evaluate, flow, generate
from logic.state import WrappedModel

from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from utils import checkpointing, logging


def generate_samples_teacher_model(
    perplexity_n_samples,
    batch_size,
    model,
    work_dirs,
    vocab_size,
    tokenizer,
    rank,
    device,
    path,
    source_distribution,
    cfg,
    time_epsilon,
    step,
    dataloader,
    solver_name: str = "mixture_euler",
):
    samples = []
    for _ in range(perplexity_n_samples // batch_size):
        samples.append(
            generate.generate_samples(
                model=WrappedModel(model=model),
                step=step,
                sample_dir=None,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                rank=rank,
                device=device,
                path=path,
                source_distribution=source_distribution,
                sample_batch_size=batch_size,
                sequence_length=cfg.model.length,
                sampling_steps=step,
                time_epsilon=time_epsilon,
                solver_name=solver_name,
            )
        )
        # Avoid allocator fragmentation over many batches
        torch.cuda.empty_cache()
        dist.barrier()

    samples = torch.cat(samples, dim=0)
    return samples


def do_generation(
    model,
    perplexity_n_samples,
    step,
    work_dirs,
    vocab_size,
    tokenizer,
    rank,
    device,
    path,
    source_distribution,
    cfg,
    time_epsilon,
    controlled_unmasking,
    dataloader,
    controller_mode,
    controller_left_k,
    controller_pct,
    return_metrics,
    logger,
    do_dynamic_step,
    grid,
):

    samples, metrics = generate.generate_few_steps_samples_with_dataset(
        wrapped_model=WrappedModel(model=model),
        step=step,
        sample_dir=work_dirs.samples,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        rank=rank,
        device=device,
        path=path,
        source_distribution=source_distribution,
        sequence_length=cfg.model.length,
        sampling_steps=step,
        time_epsilon=time_epsilon,
        student_solver=cfg.flow.student_solver,
        unmask_change=cfg.training.unmask_change,
        controlled_unmasking=controlled_unmasking,
        can_apply_dt=cfg.training.can_apply_dt,
        dataloader=dataloader,
        controller_mode=controller_mode,
        controller_left_k=controller_left_k,
        controller_pct=controller_pct,
        return_metrics=return_metrics,
        perplexity_n_samples=perplexity_n_samples,
        do_dynamic_step=do_dynamic_step,
        grid=grid,
    )
    num_predicted_tokens = metrics["num_predicted_tokens"]
    num_correct_predicted = metrics["num_correct_predicted"]
    value = controller_pct
    if controller_mode == "left_k":
        value = controller_left_k

    perplexity = evaluate.compute_perplexity(
        samples=samples,
        perplexity_batch_size=cfg.eval.perplexity_batch_size,
    )
    dist.all_reduce(perplexity, dist.ReduceOp.AVG)

    entropy = evaluate.compute_entropy(samples=samples)
    dist.all_reduce(entropy, dist.ReduceOp.AVG)

    logger.log_metric(
        value=num_correct_predicted.item() / num_predicted_tokens.item(),
        name=f"accuracy_{controller_mode}_{value}",
        stage="Evaluation",
        step=step,
    )
    logger.log_metric(
        value=perplexity.item(),
        name=f"perplexity_{controller_mode}_{value}",
        stage="Evaluation",
        step=step,
    )
    logger.log_metric(
        value=entropy.item(),
        name=f"entropy_{controller_mode}_{value}",
        stage="Evaluation",
        step=step,
    )


def generate_samples_student_model(
    perplexity_n_samples,
    batch_size,
    model,
    work_dirs,
    vocab_size,
    tokenizer,
    rank,
    device,
    path,
    source_distribution,
    cfg,
    time_epsilon,
    step,
    dataloader,
    logger,
    do_dynamic_step,
    grid,
    solver_name: str = None,
):
    samples = []
    controlled_unmasking = cfg.training.controlled_unmasking
    if controlled_unmasking and cfg.training.controlled_unmasking_type == "Training":
        controlled_unmasking = False
    _solver = solver_name if solver_name is not None else cfg.flow.student_solver
    for _ in range(perplexity_n_samples // batch_size):
        samples.append(
            generate.generate_few_steps_samples(
                model=WrappedModel(model=model),
                step=step,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                rank=rank,
                device=device,
                path=path,
                source_distribution=source_distribution,
                sample_batch_size=batch_size,
                sequence_length=cfg.model.length,
                sampling_steps=step,
                time_epsilon=time_epsilon,
                sample_dir=None,
                student_solver=_solver,
                unmask_change=cfg.training.unmask_change,
                controlled_unmasking=controlled_unmasking,
                can_apply_dt=cfg.training.can_apply_dt,
                do_dynamic_step=do_dynamic_step,
                grid=grid,
            )
        )

    samples = torch.cat(samples, dim=0)
    dist.barrier()
    return samples


def calculate_perplexity(
    perplexity_n_samples: int,
    batch_size: int,
    teacher_model: bool,
    model: nn.Module,
    vocab_size: int,
    work_dirs,
    tokenizer,
    rank,
    device,
    path,
    source_distribution,
    cfg,
    time_epsilon,
    logger,
    sampling_steps,
    dataloader,
    do_dynamic_step,
    solver_name: str = "mixture_euler",
    output_dir: str = None,
    run_name: str = "eval",
    run_info: dict = None,
):
    assert perplexity_n_samples // batch_size > 0

    step = sampling_steps
    if teacher_model:
        samples = generate_samples_teacher_model(
            perplexity_n_samples,
            batch_size,
            model,
            work_dirs,
            vocab_size,
            tokenizer,
            rank,
            device,
            path,
            source_distribution,
            cfg,
            time_epsilon,
            step=step,
            dataloader=dataloader,
            solver_name=solver_name,
        )
    else:
        samples = generate_samples_student_model(
            perplexity_n_samples,
            batch_size,
            model,
            work_dirs,
            vocab_size,
            tokenizer,
            rank,
            device,
            path,
            source_distribution,
            cfg,
            time_epsilon,
            step=step,
            dataloader=dataloader,
            logger=logger,
            do_dynamic_step=do_dynamic_step,
            grid=None,
            solver_name=solver_name,
        )

    # Gen PPL: corpus-level exp(sum NLL / #tokens) under GPT-2 Large
    model.cpu()
    torch.cuda.empty_cache()
    nll_sum, n_tokens = evaluate.compute_gen_ppl_nll(samples=samples, batch_size=4)
    dist.all_reduce(nll_sum, dist.ReduceOp.SUM)
    dist.all_reduce(n_tokens, dist.ReduceOp.SUM)
    gen_ppl = torch.exp(nll_sum / n_tokens)
    model.to(samples.device)

    entropy = evaluate.compute_entropy(samples=samples)
    dist.all_reduce(entropy, dist.ReduceOp.AVG)

    all_samples = _gather_from_ranks(samples)

    if rank == 0:
        print(f"Step {step} -> Gen PPL: {gen_ppl:.3f}, Entropy: {entropy:.3f}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            texts = tokenizer.batch_decode(all_samples)
            txt_path = os.path.join(output_dir, f"{run_name}.txt")
            _write_samples_one_per_line(texts, txt_path)

            result = dict(run_info or {})
            result.update(
                {
                    "nfe": step,
                    "n_samples": len(texts),
                    "batch_size": batch_size,
                    "gen_ppl": gen_ppl.item(),
                    "gen_ppl_eval_model": "gpt2-large",
                    "entropy": entropy.item(),
                    "samples_file": os.path.basename(txt_path),
                }
            )
            json_path = os.path.join(output_dir, f"{run_name}.json")
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[Eval] Saved {len(texts)} samples to {txt_path}")
            print(f"[Eval] Saved results to {json_path}")

    logger.log_metric(value=gen_ppl.item(), name="GenPPL", stage="Evaluation", step=step)
    logger.log_metric(value=entropy.item(), name="Entropy", stage="Evaluation", step=step)


def _gather_from_ranks(x: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    if world_size == 1:
        return x
    parts = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(parts, x)
    return torch.cat(parts, dim=0)


def _write_samples_one_per_line(texts, path):
    """One sample per line; backslashes and in-sample newlines are escaped."""
    with open(path, "w") as f:
        for t in texts:
            line = t.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")
            f.write(line + "\n")


def get_dt_grid(iloc, cfg, device, rev=False):
    step_sizes = cfg.flow.step_sizes
    grid = []
    for i in range(iloc):
        grid.append(step_sizes[i + 1])
    grid.append(step_sizes[iloc])
    if rev:
        grid.reverse()
    grid = torch.tensor(grid).to(device)
    return grid


def run_eval(
    rank: int,
    seed: int,
    work_dir: str,
    pre_trained_model_path: str,
    batch_size: int,
    perplexity_n_samples: int,
    sampling_steps: int,
    eval_perplexity: bool,
    eval_elbo: bool,
    elbo_data: str,
    world_size: int,
    n_discretization: float = 1024,
    teacher_model: bool = True,
    do_dynamic_step: bool = True,
    use_shs: bool = False,
    use_dtmc: bool = False,
    config_path: str = None,
    cache_dir: str = None,
) -> None:
    torch.manual_seed(seed + rank)

    # Logging and configuration
    os.makedirs(work_dir, exist_ok=True)
    work_dirs = checkpointing.get_work_dirs(work_dir=work_dir, rank=rank)
    work_dirs.checkpoint = Path(pre_trained_model_path)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if config_path is None:
        cfg = checkpointing.load_cfg_from_path(work_dir=work_dirs.checkpoint)
    else:
        cfg = OmegaConf.load(config_path)
    if cache_dir is not None:
        cfg.data.cache_dir = cache_dir
    logger = logging.TrainLogger(log_dir=work_dirs.root, rank=rank, cfg=cfg)
    logger.info(work_dirs)
    logger.info(cfg)
    logger.log_devices(device=device, logger=logger)

    # Data
    save_path = os.path.join(cfg.data.cache_dir, "processed_data", "tokenizer_dir")
    if rank == 0 and not os.path.exists(save_path):
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.save_pretrained(save_path)

    dist.barrier()
    tokenizer = GPT2TokenizerFast.from_pretrained(save_path)
    vocab_size = tokenizer.vocab_size
    logger.info(f"vocab_size is {vocab_size}")

    # Flow matching
    path = flow.get_path(
        scheduler_type=cfg.flow.scheduler_type, exponent=cfg.flow.exponent
    )
    if teacher_model:
        loss_fn = flow.get_loss_function(
            loss_function=cfg.flow.teacher_loss_function, path=path
        )
    else:
        loss_fn = flow.get_loss_function(
            loss_function=cfg.flow.student_loss_function, path=path
        )
    # Elbo may have singularity at 1
    time_epsilon = 1e-3 if isinstance(loss_fn, MixturePathGeneralizedKL) else 0.0

    source_distribution = flow.get_source_distribution(
        source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size
    )

    model, missing_keys, unexpected_keys = checkpointing.load_model_from_path(
        work_dir=work_dirs.checkpoint,
        device=device,
        source_distribution=source_distribution,
        cfg=cfg,
        vocab_size=vocab_size,
        teacher_model=teacher_model,
    )
    model.eval()
    logger.info(model)
    logger.info("****************")
    logger.info(f"⚠️  missing_keys is: {missing_keys}")
    logger.info(f"⚠️  unexpected_keys is: {unexpected_keys}")
    logger.info("****************")

    if cfg.model.compile:
        model = torch.compile(model)
        torch.set_float32_matmul_precision("high")

    # The validation data is only needed for the ELBO
    dataloader = None
    if eval_elbo:
        data_state = data._get_dataset(
            name=elbo_data,
            mode="validation",
            cache_dir=cfg.data.cache_dir,
            block_size=cfg.model.length,
            num_proc=cfg.data.num_workers,
            batch_size=batch_size,
            ngpus=world_size,
            force_process=cfg.data.force_process,
        )

        dataloader = DataLoader(
            data_state.dataset,
            batch_size=batch_size,
            sampler=data_state.sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.sampler is None),
        )

    # Sampler: {Standard, SHS} x per-step jump probability {CTMC, DTMC}
    if use_shs and use_dtmc:
        solver_name = "mixture_euler_shs_dtmc"
    elif use_shs:
        solver_name = "mixture_euler_shs"
    elif use_dtmc:
        solver_name = "mixture_euler_dtmc"
    else:
        solver_name = "mixture_euler"
    if rank == 0:
        print(f"[Eval] Solver: {solver_name}")

    sampler = "shs" if use_shs else "standard"
    p_jump_mode = "dtmc" if use_dtmc else "ctmc"
    model_name = "dfm" if teacher_model else "fsdfm"
    run_name = f"{model_name}_{sampler}_{p_jump_mode}_nfe{sampling_steps}_seed{seed}"
    run_info = {
        "model": model_name,
        "sampler": sampler,
        "p_jump_mode": p_jump_mode,
        "solver": solver_name,
        "seed": seed,
        "world_size": world_size,
    }

    if eval_perplexity:
        calculate_perplexity(
            perplexity_n_samples=perplexity_n_samples,
            batch_size=batch_size,
            teacher_model=teacher_model,
            model=model,
            vocab_size=vocab_size,
            work_dirs=work_dirs,
            tokenizer=tokenizer,
            rank=rank,
            device=device,
            path=path,
            source_distribution=source_distribution,
            cfg=cfg,
            time_epsilon=time_epsilon,
            logger=logger,
            sampling_steps=sampling_steps,
            dataloader=dataloader,
            do_dynamic_step=do_dynamic_step,
            solver_name=solver_name,
            output_dir=work_dir,
            run_name=run_name,
            run_info=run_info,
        )

    if eval_elbo:
        elbo, num_elements = evaluate.estimate_likelihood(
            model=model,
            dataloader=dataloader,
            source_distribution=source_distribution,
            n_discretization=n_discretization,
            device=device,
            batch_size=batch_size,
            path=path,
        )
        dist.barrier()

        dist.all_reduce(elbo, dist.ReduceOp.SUM)
        dist.all_reduce(num_elements, dist.ReduceOp.SUM)

        logger.log_metric(
            value=torch.exp(elbo / num_elements).item(),
            name=f"ELBO",
            stage="Evaluation",
            step=0,
        )

        if rank == 0:
            print(f"ELBO: {torch.exp(elbo / num_elements).item():.2f}")


def setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(port))

    torch.cuda.set_device(rank)

    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)


def cleanup() -> None:
    dist.destroy_process_group()


def run_mp_eval(
    rank: int,
    world_size: int,
    seed: int,
    work_dir: str,
    pre_trained_model_path: str,
    batch_size: int,
    sampling_steps: int,
    eval_elbo: bool,
    eval_perplexity: bool,
    elbo_data: str,
    perplexity_n_samples: int,
    port: int,
    teacher_model: bool = True,
    do_dynamic_step: bool = False,
    use_shs: bool = False,
    use_dtmc: bool = False,
    config_path: str = None,
    cache_dir: str = None,
) -> None:
    try:
        setup(rank=rank, world_size=world_size, port=port)
        run_eval(
            rank=rank,
            seed=seed,
            work_dir=work_dir,
            pre_trained_model_path=pre_trained_model_path,
            batch_size=batch_size,
            sampling_steps=sampling_steps,
            eval_elbo=eval_elbo,
            eval_perplexity=eval_perplexity,
            elbo_data=elbo_data,
            world_size=world_size,
            perplexity_n_samples=perplexity_n_samples,
            teacher_model=teacher_model,
            do_dynamic_step=do_dynamic_step,
            use_shs=use_shs,
            use_dtmc=use_dtmc,
            config_path=config_path,
            cache_dir=cache_dir,
        )
    finally:
        cleanup()
