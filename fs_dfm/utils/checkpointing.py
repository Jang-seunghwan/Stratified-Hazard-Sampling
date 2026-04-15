#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
from logic.flow import SourceDistribution
from model import Transformer
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def load_cfg_from_path(work_dir: str) -> OmegaConf:
    work_dir = Path(work_dir)

    root_dir = work_dir if work_dir.is_dir() else work_dir.parents[1]

    cfg_path = root_dir / ".hydra/config.yaml"

    return OmegaConf.load(cfg_path)


def load_model_from_path(
    work_dir: str,
    source_distribution: SourceDistribution,
    device: torch.device,
    vocab_size: int,
    cfg: OmegaConf,
    teacher_model: bool = True,
) -> nn.Module:
    work_dir = Path(work_dir)

    if work_dir.is_dir():
        ckpt_dir = work_dir / "checkpoints" / "checkpoint.pth"
    else:
        ckpt_dir = work_dir

    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)

    # Support both checkpoint formats:
    #   Apple release: {"model": state_dict, "optimizer": ..., "step": ...}
    #   FS-DFM trained: {"teacher_model": state_dict, "student_model": state_dict, ...}
    use_ddp = dist.is_initialized() and dist.get_world_size() > 1

    if teacher_model:
        model = Transformer(
            config=cfg.model,
            vocab_size=vocab_size,
            masked=source_distribution.masked,
            dt_conditioned=False,
        ).to(device)
        if "teacher_model" in loaded_state:
            state_dict = loaded_state["teacher_model"]
        elif "model" in loaded_state:
            state_dict = loaded_state["model"]
        else:
            raise KeyError(f"Checkpoint has no 'teacher_model' or 'model' key. Keys: {list(loaded_state.keys())}")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)
        if use_ddp:
            model = DDP(model, device_ids=[device])
        print("teacher_model is loaded!!!")
    else:
        model = Transformer(
            config=cfg.model,
            vocab_size=vocab_size,
            masked=source_distribution.masked,
            dt_conditioned=True,
        ).to(device)
        if "student_model" in loaded_state:
            state_dict = loaded_state["student_model"]
        elif "model" in loaded_state:
            state_dict = loaded_state["model"]
        else:
            raise KeyError(f"Checkpoint has no 'student_model' or 'model' key. Keys: {list(loaded_state.keys())}")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)
        if use_ddp:
            model = DDP(model, device_ids=[device])
        print("student_model is loaded!!!")

    return model, missing_keys, unexpected_keys


@dataclass
class WorkDirectory:
    root: Path = field(metadata={"help": "Root work directory"})
    checkpoint: Path = field(metadata={"help": "Checkpoint directory"})
    samples: Path = field(metadata={"help": "Samples directory"})


def get_work_dirs(work_dir: str, rank: int) -> WorkDirectory:
    work_dir = Path(work_dir)

    sample_dir = work_dir / "samples"
    checkpoint_dir = work_dir / "checkpoints" / "checkpoint.pth"

    if rank == 0:
        sample_dir.mkdir(exist_ok=True)
        checkpoint_dir.parents[0].mkdir(exist_ok=True)

    return WorkDirectory(root=work_dir, checkpoint=checkpoint_dir, samples=sample_dir)
