from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm

from gidd.diffusion_process import NoiseSchedule
from gidd.utils import sample_categorical


class Sampler(nn.Module):
    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps: float = 1e-4):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.t_eps = t_eps

    @abstractmethod
    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, num_samples=1, num_denoising_steps=1000, max_length=None, decode=True, show_progress=True):
        max_length = max_length or self.model.config.max_seq_len
        device = next(self.model.parameters()).device

        z_t = self._do_generate(num_samples, num_denoising_steps, max_length, show_progress=show_progress, device=device)

        if decode:
            texts = self.tokenizer.batch_decode(z_t, skip_special_tokens=True)
            return texts
        else:
            return z_t


class GiddSampler(Sampler):
    class DenoisingStep(nn.Module):
        def __init__(self, model, noise_schedule, tokenizer, min_p=0.0):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.tokenizer = tokenizer
            self.min_p = min_p

        def forward(self, z_t, t, s):
            logits = self.model(z_t, t)
            logits[..., self.tokenizer.mask_token_id] = -1e6

            # if i > 0:
            q_s = self.noise_schedule.probs_at_t(logits.softmax(-1), s)
            q_t = self.noise_schedule.probs_at_t(logits.softmax(-1), t)
            q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

            alpha_t, beta_pi_t = self.noise_schedule.get_alpha_betapi(t)
            alpha_s, beta_pi_s = self.noise_schedule.get_alpha_betapi(s)

            alpha_ts = alpha_t / alpha_s
            beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

            vz_t = F.one_hot(z_t, num_classes=len(self.tokenizer))
            beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
            q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt)

            q_st = q_ts * q_s / q_zt
            if self.min_p > 0.0:
                is_small = (q_st < self.min_p).float()
                q_st = (1 - is_small) * q_st
                q_st = q_st / q_st.sum(-1, keepdim=True)
            return sample_categorical(q_st)

    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_step = self.DenoisingStep(model, noise_schedule, tokenizer, min_p=min_p)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):

        ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
        ts = (1 - 2 * self.t_eps) * ts + self.t_eps

        # zt = sample_categorical(p_zt)
        z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)
        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)]).clone()
        return z_t


class GiddSamplerSHS(Sampler):
    """GIDD Sampler with Stratified Hazard Sampling (SHS) for uniform noise part only.

    - [MASK] tokens: Standard probabilistic unmasking (unchanged)
    - non-[MASK] tokens: SHS applied (hazard accumulation -> jump when threshold crossed)
    """

    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=False, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.min_p = min_p
        self.mask_id = tokenizer.mask_token_id
        # Note: compile_step=False by default for SHS since the loop has dynamic control flow

    def _compute_q_st(self, z_t, t, s):
        """Compute posterior q(z_s | z_t) - same as GiddSampler.DenoisingStep"""
        logits = self.model(z_t, t)
        logits[..., self.mask_id] = -1e6

        q_s = self.noise_schedule.probs_at_t(logits.softmax(-1), s)
        q_t = self.noise_schedule.probs_at_t(logits.softmax(-1), t)
        q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

        alpha_t, beta_pi_t = self.noise_schedule.get_alpha_betapi(t)
        alpha_s, beta_pi_s = self.noise_schedule.get_alpha_betapi(s)

        alpha_ts = alpha_t / alpha_s
        beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

        vz_t = F.one_hot(z_t, num_classes=len(self.tokenizer))
        beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
        q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt)

        q_st = q_ts * q_s / q_zt

        if self.min_p > 0.0:
            is_small = (q_st < self.min_p).float()
            q_st = (1 - is_small) * q_st
            q_st = q_st / q_st.sum(-1, keepdim=True)

        return q_st

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
        ts = (1 - 2 * self.t_eps) * ts + self.t_eps

        # Initialize z_t from prior (all [MASK] tokens)
        z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)

        # === SHS state initialization (for non-[MASK] tokens only) ===
        hazard_dtype = torch.float64
        S = torch.zeros((num_samples, max_length), dtype=hazard_dtype, device=device)  # Cumulative hazard
        k = torch.zeros((num_samples, max_length), dtype=torch.long, device=device)    # Jump counter
        theta = torch.rand((num_samples, max_length), dtype=hazard_dtype, device=device)  # Random phase U(0,1)

        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating (SHS)", disable=not show_progress, dynamic_ncols=True):
            t, s = ts[i], ts[max(0, i-1)]

            # Compute posterior q(z_s | z_t)
            q_st = self._compute_q_st(z_t, t, s)

            # Identify token states
            is_mask = (z_t == self.mask_id)  # (batch, seq_len)

            # === [MASK] tokens: Standard probabilistic sampling ===
            mask_samples = sample_categorical(q_st)

            # === non-[MASK] tokens: SHS ===
            # p_stay = probability of staying at current token
            p_stay = q_st.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
            p_jump = (1.0 - p_stay).clamp(0.0, 1.0).to(hazard_dtype)

            # Only accumulate hazard for non-[MASK] positions
            p_jump_masked = p_jump.masked_fill(is_mask, 0.0)
            S = S + p_jump_masked

            # Trigger jump when S crosses threshold (theta + k)
            threshold = theta + k.to(hazard_dtype)
            jump_mask = (S >= threshold) & (p_jump_masked > 0) & (~is_mask)

            # Sample new tokens for jumped positions (excluding current token)
            shs_samples = z_t.clone()
            if jump_mask.any():
                # Flatten for indexing
                flat_jump = jump_mask.reshape(-1)
                jump_idx = flat_jump.nonzero(as_tuple=False).squeeze(-1)

                q_sel = q_st.reshape(-1, q_st.shape[-1]).index_select(0, jump_idx)
                zt_sel = z_t.reshape(-1).index_select(0, jump_idx)

                # Remove current token from distribution and renormalize
                q_sel = q_sel.clone()
                q_sel.scatter_(1, zt_sel.unsqueeze(1), 0.0)
                q_sel = q_sel / q_sel.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                new_tokens = sample_categorical(q_sel)

                shs_flat = shs_samples.reshape(-1)
                shs_flat[jump_idx] = new_tokens
                shs_samples = shs_flat.view_as(shs_samples)

                # Update jump counter
                k_flat = k.reshape(-1)
                k_flat[jump_idx] += 1
                k = k_flat.view_as(k)

            # === Combine: [MASK] uses standard sampling, non-[MASK] uses SHS ===
            z_t = torch.where(is_mask, mask_samples, shs_samples)

        return z_t


class MDLMSampler(Sampler):
    class DenoisingStep(nn.Module):
        def __init__(self, model, noise_schedule, mask_id, min_p=0.0):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.mask_id = mask_id
            self.min_p = min_p

        def get_sigmas(self, t, eps=1e-4):
            dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
            sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
            return dsigma, sigma

        def forward(self, z_t, t, tm1, i=None, eps=1e-4):
            logits = self.model(z_t, t)
            logits[..., self.mask_id] = -1e6

            if i == 0:
                z_tm1 = logits.argmax(-1)
            else:
                _, sigma_t = self.get_sigmas(t, eps=eps)
                _, sigma_tm1 = self.get_sigmas(tm1, eps=eps)

                move_chance_t = 1 - torch.exp(-sigma_t)
                move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
                move_chance_t = move_chance_t[:, None, None]
                move_chance_tm1 = move_chance_tm1[:, None, None]
                probs = logits.softmax(-1) * (move_chance_t - move_chance_tm1)
                probs[:, :, self.mask_id] = move_chance_tm1[:, :, 0]
                probs /= move_chance_t
                if self.min_p > 0.0:
                    is_small = (probs < self.min_p).float()
                    probs = (1 - is_small) * probs
                    probs = probs / probs.sum(-1, keepdim=True)
                z_tm1 = sample_categorical(probs)
                # z_tm1 = torch.distributions.Categorical(probs=probs).sample()
                # z_tm1 = _sample_categorical(probs)

            copy_flag = (z_t != self.mask_id).to(z_t.dtype)
            z_t = copy_flag * z_t + (1 - copy_flag) * z_tm1
            return z_t

    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_step = self.DenoisingStep(model, noise_schedule, tokenizer.mask_token_id, min_p=min_p)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)

        ts = torch.linspace(self.t_eps, 1 - self.t_eps, num_denoising_steps + 1, device=device).unsqueeze(-1)

        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress):
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], i=i, eps=self.t_eps).clone()

        return z_t


class AutoregressiveSampler(Sampler):
    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, compile_step=True):
        super().__init__(model, tokenizer, noise_schedule)
        if compile_step:
            self.model = torch.compile(model)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        bos_token_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

        input_ids = torch.full((num_samples, max_length), eos_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((num_samples, max_length), dtype=torch.long, device=device)
        input_ids[:, 0] = bos_token_id
        attention_mask[:, 0] = 1

        done = torch.zeros(num_samples, device=device)
        for i in tqdm.trange(1, max_length, desc="Generating samples", disable=not show_progress):
            logits = self.model(input_ids, use_cache=False).logits[:, i-1]
            probs = logits.softmax(-1)
            next_x = (1 - done) * sample_categorical(probs) + done * self.tokenizer.pad_token_id
            input_ids[:, i] = next_x.to(input_ids.dtype)
            done += (1 - done) * (next_x == eos_token_id).to(done.dtype)
            if (done == 1).all():
                break

        return input_ids


def get_sampler(config, model, tokenizer, noise_schedule: NoiseSchedule, compile_step=True, min_p=0.0):
    if config.model.type == "diffusion":
        if config.model.diffusion_process == "gidd":
            return GiddSampler(model, tokenizer, noise_schedule, t_eps=config.model.t_eps, compile_step=compile_step, min_p=min_p)
        elif config.model.diffusion_process == "mdlm":
            return MDLMSampler(model, tokenizer, noise_schedule, t_eps=config.model.t_eps, compile_step=compile_step, min_p=min_p)
        else:
            raise ValueError(f"Unsupported forward process: {config.model.diffusion_process}")
    elif config.model.type == "autoregressive":
        return AutoregressiveSampler(model, tokenizer, noise_schedule, compile_step=True)
    else:
        raise ValueError(f"Unsupported model type: {config.model.type}")
