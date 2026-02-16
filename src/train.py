"""Training pipeline for the ARC transformer model.

Contains: optimizer building (including NorMuon), training loop, validation, checkpointing.
For model/data building, see build.py.
"""

import argparse
import math
import numbers
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, TextIO, Tuple

import torch
from torch import nn
from torch.optim import AdamW

from common import (
    ARCExampleDataset,
    MAX_SEQ_LEN,
    capture_rng_state,
    create_dataloader,
)
from tinytransformer import RMSNorm, TinyTransformer


# =============================================================================
# NorMuon Optimizer (Muon + AdamW hybrid)
# =============================================================================
# Taken from the official implementation: https://github.com/zichongli5/NorMuon

def _zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = torch.lerp(grad, momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
    if original_shape is not None:
        update = update.reshape(original_shape)
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean.to(second_momentum.dtype), 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10)))
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def _adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceNorMuonWithAuxAdam(torch.optim.Optimizer):
    """NorMuon optimizer: Muon for linear weights, AdamW for everything else."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.1)
                assert set(group.keys()) == {
                    "params", "lr", "momentum", "beta2", "weight_decay", "use_muon",
                }
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.1)
                assert set(group.keys()) == {
                    "params", "lr", "betas", "eps", "weight_decay", "use_muon",
                }
        super().__init__(param_groups, dict())

        self._normuon_update_fn: Callable[..., torch.Tensor] = _normuon_update
        self._adam_update_fn: Callable[..., torch.Tensor] = _adam_update
        self._compiled_update_kernels = False
        self._zero_scalars: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

        disable_compile = str(
            os.environ.get("MDLARC_DISABLE_OPTIMIZER_COMPILE", "0")
        ).lower() in {"1", "true", "yes", "on"}
        has_cuda_params = any(
            p.is_cuda for group in param_groups for p in group["params"]
        )
        if has_cuda_params and hasattr(torch, "compile") and not disable_compile:
            try:
                self._normuon_update_fn = torch.compile(
                    _normuon_update,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True,
                )
                self._adam_update_fn = torch.compile(
                    _adam_update,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True,
                )
                self._compiled_update_kernels = True
                print("Compiled NorMuon optimizer update kernels.")
            except Exception as exc:
                print(f"Skipping NorMuon optimizer kernel compile ({exc}).")

    def _get_zero_grad(self, p: torch.Tensor) -> torch.Tensor:
        key = (p.device, p.dtype)
        zero = self._zero_scalars.get(key)
        if zero is None:
            zero = torch.zeros((), device=p.device, dtype=p.dtype)
            self._zero_scalars[key] = zero
        return zero.expand_as(p)

    def _call_normuon_update(
        self,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        second_momentum: torch.Tensor,
        beta: float,
        beta2: float,
    ) -> torch.Tensor:
        if not self._compiled_update_kernels:
            return _normuon_update(grad, momentum, second_momentum, beta=beta, beta2=beta2)
        try:
            return self._normuon_update_fn(
                grad, momentum, second_momentum, beta=beta, beta2=beta2
            )
        except Exception as exc:
            print(
                "Compiled NorMuon kernels failed during step "
                f"({exc}); falling back to eager optimizer updates."
            )
            self._compiled_update_kernels = False
            self._normuon_update_fn = _normuon_update
            self._adam_update_fn = _adam_update
            return _normuon_update(grad, momentum, second_momentum, beta=beta, beta2=beta2)

    def _call_adam_update(
        self,
        grad: torch.Tensor,
        buf1: torch.Tensor,
        buf2: torch.Tensor,
        step: int,
        betas: Tuple[float, float],
        eps: float,
    ) -> torch.Tensor:
        if not self._compiled_update_kernels:
            return _adam_update(grad, buf1, buf2, step, betas, eps)
        try:
            return self._adam_update_fn(grad, buf1, buf2, step, betas, eps)
        except Exception as exc:
            print(
                "Compiled NorMuon kernels failed during step "
                f"({exc}); falling back to eager optimizer updates."
            )
            self._compiled_update_kernels = False
            self._normuon_update_fn = _normuon_update
            self._adam_update_fn = _adam_update
            return _adam_update(grad, buf1, buf2, step, betas, eps)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                lr = float(group["lr"])
                beta = float(group["momentum"])
                beta2 = float(group["beta2"])
                weight_decay = float(group["weight_decay"])

                buckets: Dict[
                    Tuple[Tuple[int, ...], torch.dtype, torch.device], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]
                ] = {}
                for p in group["params"]:
                    grad = p.grad
                    had_grad = grad is not None
                    if grad is None:
                        grad = self._get_zero_grad(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                    key = (tuple(p.shape), p.dtype, p.device)
                    bucket = buckets.setdefault(key, [])
                    bucket.append(
                        (
                            p,
                            grad,
                            state["momentum_buffer"],
                            state["second_momentum_buffer"],
                            had_grad,
                        )
                    )

                params_to_update: List[torch.Tensor] = []
                updates: List[torch.Tensor] = []
                params_with_grad: List[torch.Tensor] = []

                for bucket in buckets.values():
                    if len(bucket) == 1:
                        p, grad, momentum_buffer, second_momentum_buffer, had_grad = bucket[0]
                        update = self._call_normuon_update(
                            grad,
                            momentum_buffer,
                            second_momentum_buffer,
                            beta,
                            beta2,
                        )
                        params_to_update.append(p)
                        updates.append(update.reshape_as(p))
                        if weight_decay and had_grad:
                            params_with_grad.append(p)
                        continue

                    grad_batch = torch.stack([item[1] for item in bucket], dim=0)
                    momentum_batch = torch.stack([item[2] for item in bucket], dim=0)
                    second_momentum_batch = torch.stack(
                        [item[3] for item in bucket], dim=0
                    )
                    update_batch = self._call_normuon_update(
                        grad_batch,
                        momentum_batch,
                        second_momentum_batch,
                        beta,
                        beta2,
                    )

                    for idx, (
                        p,
                        _grad,
                        momentum_buffer,
                        second_momentum_buffer,
                        had_grad,
                    ) in enumerate(bucket):
                        momentum_buffer.copy_(momentum_batch[idx])
                        second_momentum_buffer.copy_(second_momentum_batch[idx])
                        params_to_update.append(p)
                        updates.append(update_batch[idx].reshape_as(p))
                        if weight_decay and had_grad:
                            params_with_grad.append(p)

                if weight_decay and params_with_grad:
                    torch._foreach_mul_(params_with_grad, 1 - lr * weight_decay)
                if params_to_update:
                    torch._foreach_add_(params_to_update, updates, alpha=-lr)
            else:
                lr = float(group["lr"])
                betas = group["betas"]
                beta1 = float(betas[0])
                beta2 = float(betas[1])
                eps = float(group["eps"])
                weight_decay = float(group["weight_decay"])

                params: List[torch.Tensor] = []
                grads: List[torch.Tensor] = []
                exp_avgs: List[torch.Tensor] = []
                exp_avg_sqs: List[torch.Tensor] = []
                steps: List[int] = []
                params_with_grad: List[torch.Tensor] = []

                for p in group["params"]:
                    grad = p.grad
                    had_grad = grad is not None
                    if grad is None:
                        grad = self._get_zero_grad(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    step = int(state["step"])

                    params.append(p)
                    grads.append(grad)
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    steps.append(step)
                    if weight_decay and had_grad:
                        params_with_grad.append(p)

                if not params:
                    continue

                all_same_step = all(step == steps[0] for step in steps)
                if all_same_step:
                    torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
                    torch._foreach_mul_(exp_avg_sqs, beta2)
                    torch._foreach_addcmul_(
                        exp_avg_sqs,
                        grads,
                        grads,
                        value=1 - beta2,
                    )

                    step_value = steps[0]
                    bias_correction1 = 1 - beta1**step_value
                    bias_correction2 = 1 - beta2**step_value

                    updates = torch._foreach_div(exp_avgs, bias_correction1)
                    denoms = torch._foreach_sqrt(exp_avg_sqs)
                    torch._foreach_div_(denoms, math.sqrt(bias_correction2))
                    torch._foreach_add_(denoms, eps)
                    updates = torch._foreach_div(updates, denoms)
                else:
                    updates = []
                    for grad, exp_avg, exp_avg_sq, step in zip(
                        grads, exp_avgs, exp_avg_sqs, steps
                    ):
                        updates.append(
                            self._call_adam_update(
                                grad,
                                exp_avg,
                                exp_avg_sq,
                                step,
                                betas,
                                eps,
                            )
                        )

                if weight_decay and params_with_grad:
                    torch._foreach_mul_(params_with_grad, 1 - lr * weight_decay)
                torch._foreach_add_(params, updates, alpha=-lr)
        return loss


# =============================================================================
# Training One Epoch
# =============================================================================

def _emit_log(message: str, log_location: str, log_handle: Optional[TextIO]) -> None:
    if log_location in ("terminal", "both"):
        print(message)
    if log_location in ("file", "both") and log_handle is not None:
        log_handle.write(message + "\n")
        log_handle.flush()


def train_one_epoch(
    model: TinyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    gradient_accumulation_steps: int = 1,
    start_step: int = 0,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    epoch: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    train_log_mode: str = "10_steps",
    log_location: str = "both",
    log_handle: Optional[TextIO] = None,
) -> int:
    model.train()
    step = start_step
    use_amp = device.type == "cuda"

    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)
    if steps_per_epoch is not None and steps_per_epoch <= 0:
        steps_per_epoch = None

    accum_steps = max(1, int(gradient_accumulation_steps or 1))
    accum_index = 0
    accum_target = accum_steps
    dataloader_length = steps_per_epoch if steps_per_epoch is not None else None
    optimizer_step = 0

    window_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    window_input_sum = torch.zeros((), device=device, dtype=torch.float32)
    window_output_sum = torch.zeros((), device=device, dtype=torch.float32)
    window_count = 0

    epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    epoch_input_sum = torch.zeros((), device=device, dtype=torch.float32)
    epoch_output_sum = torch.zeros((), device=device, dtype=torch.float32)
    epoch_count = 0

    def maybe_log_window(force: bool = False) -> None:
        nonlocal window_count
        should_log = False
        if train_log_mode == "step":
            should_log = True
        elif train_log_mode == "10_steps":
            should_log = (optimizer_step % 10 == 0) or force
        if not should_log or window_count == 0:
            return

        avg_loss = (window_loss_sum / window_count).item()
        avg_inp = (window_input_sum / window_count).item()
        avg_out = (window_output_sum / window_count).item()
        current_lr = (
            scheduler.get_last_lr()[0]
            if scheduler
            else optimizer.param_groups[0]["lr"]
        )
        if epoch is None:
            prefix = f"step={optimizer_step}"
        else:
            prefix = f"epoch={epoch + 1} step={optimizer_step}"
        log_msg = (
            f"{prefix} lr={current_lr:.2e} losses: avg={avg_loss:.4f} "
            f"inp={avg_inp:.4f} out={avg_out:.4f}"
        )
        _emit_log(log_msg, log_location, log_handle)
        window_loss_sum.zero_()
        window_input_sum.zero_()
        window_output_sum.zero_()
        window_count = 0

    last_batch_idx = None
    for batch_idx, batch in enumerate(dataloader):
        last_batch_idx = batch_idx
        step += 1
        input_ids = batch["input_ids"].to(device)
        sep_indices_cpu = batch.get("sep_indices")
        sep_indices = (
            sep_indices_cpu.to(device) if sep_indices_cpu is not None else None
        )
        cu_seqlens_cpu = batch.get("cu_seqlens")
        if cu_seqlens_cpu is not None:
            cu_seqlens = cu_seqlens_cpu.to(device=device, dtype=torch.int32)
            max_seqlen_raw = batch.get("max_seqlen")
            if max_seqlen_raw is None:
                raise ValueError("Packed batches must include max_seqlen.")
            max_seqlen = (
                int(max_seqlen_raw.item())
                if torch.is_tensor(max_seqlen_raw)
                else int(max_seqlen_raw)
            )
            attention_mask = None
        else:
            cu_seqlens = None
            max_seqlen = None
            has_padding = bool(batch.get("has_padding", True))
            if not has_padding:
                attention_mask = None
            else:
                attention_mask = batch["attention_mask"].to(device)
        example_ids = batch["example_ids"].to(device)
        dihedral_ids = batch["dihedral_ids"].to(device)
        positions_3d = batch["positions_3d"].to(device)
        if accum_index == 0:
            optimizer.zero_grad(set_to_none=True)
            if dataloader_length is not None:
                remaining = dataloader_length - batch_idx
                accum_target = min(accum_steps, remaining)
            else:
                accum_target = accum_steps

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            outputs = model(
                input_ids,
                example_ids,
                dihedral_ids,
                attention_mask=attention_mask,
                sep_indices=sep_indices,
                compute_input_loss=False,
                positions_3d=positions_3d,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            loss = outputs["output_loss"]
            inp_loss = outputs.get("input_loss")
            out_loss = outputs.get("output_loss")

        batch_loss = loss.detach().float()
        window_loss_sum += batch_loss
        epoch_loss_sum += batch_loss
        if inp_loss is not None:
            batch_inp_loss = inp_loss.detach().float()
            window_input_sum += batch_inp_loss
            epoch_input_sum += batch_inp_loss
        if out_loss is not None:
            batch_out_loss = out_loss.detach().float()
            window_output_sum += batch_out_loss
            epoch_output_sum += batch_out_loss
        window_count += 1
        epoch_count += 1

        loss = loss / accum_target
        loss.backward()
        accum_index += 1

        if accum_index >= accum_target:
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                if epoch is None or steps_per_epoch is None:
                    scheduler.step()
                else:
                    epoch_progress = float(epoch) + (batch_idx + 1) / float(
                        steps_per_epoch
                    )
                    scheduler.step(epoch_progress)
            accum_index = 0
            optimizer_step += 1
            maybe_log_window()
    if accum_index > 0:
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            if epoch is None or steps_per_epoch is None or last_batch_idx is None:
                scheduler.step()
            else:
                epoch_progress = float(epoch) + (last_batch_idx + 1) / float(
                    steps_per_epoch
                )
                scheduler.step(epoch_progress)
        optimizer_step += 1
        maybe_log_window()

    if train_log_mode == "10_steps":
        maybe_log_window(force=True)
    elif train_log_mode == "epoch" and epoch_count > 0:
        avg_loss = (epoch_loss_sum / epoch_count).item()
        avg_inp = (epoch_input_sum / epoch_count).item()
        avg_out = (epoch_output_sum / epoch_count).item()
        current_lr = (
            scheduler.get_last_lr()[0]
            if scheduler
            else optimizer.param_groups[0]["lr"]
        )
        if epoch is None:
            prefix = "epoch"
        else:
            prefix = f"epoch={epoch + 1}"
        log_msg = (
            f"{prefix} lr={current_lr:.2e} losses: avg={avg_loss:.4f} "
            f"inp={avg_inp:.4f} out={avg_out:.4f}"
        )
        _emit_log(log_msg, log_location, log_handle)
    return step


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate_one_epoch(
    model: TinyTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Calculates validation loss (Output Loss) on the test set."""
    model.eval()
    use_amp = device.type == "cuda"
    total_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    total_tokens = torch.zeros((), device=device, dtype=torch.float32)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        sep_indices_cpu = batch.get("sep_indices")
        sep_indices = (
            sep_indices_cpu.to(device) if sep_indices_cpu is not None else None
        )
        cu_seqlens_cpu = batch.get("cu_seqlens")
        if cu_seqlens_cpu is not None:
            cu_seqlens = cu_seqlens_cpu.to(device=device, dtype=torch.int32)
            max_seqlen_raw = batch.get("max_seqlen")
            if max_seqlen_raw is None:
                raise ValueError("Packed batches must include max_seqlen.")
            max_seqlen = (
                int(max_seqlen_raw.item())
                if torch.is_tensor(max_seqlen_raw)
                else int(max_seqlen_raw)
            )
            attention_mask = None
        else:
            cu_seqlens = None
            max_seqlen = None
            has_padding = bool(batch.get("has_padding", True))
            if not has_padding:
                attention_mask = None
            else:
                attention_mask = batch["attention_mask"].to(device)
        example_ids = batch["example_ids"].to(device)
        dihedral_ids = batch["dihedral_ids"].to(device)
        positions_3d = batch["positions_3d"].to(device)

        if not any(batch["has_output"]):
            continue

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            outputs = model(
                input_ids,
                example_ids,
                dihedral_ids,
                attention_mask=attention_mask,
                sep_indices=sep_indices,
                compute_input_loss=False,
                positions_3d=positions_3d,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        out_loss = outputs.get("output_loss")
        num_tokens = outputs.get("num_output_tokens")

        if out_loss is not None and num_tokens is not None:
            n = num_tokens.detach().to(dtype=torch.float32)
            total_loss_sum += out_loss.detach().float() * n
            total_tokens += n

    return (total_loss_sum / total_tokens.clamp_min(1.0)).item()


# =============================================================================
# Optimizer Building
# =============================================================================

def _is_norm_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            RMSNorm,
        ),
    )


def _is_positional_embedding_param(name: str) -> bool:
    lowered = name.lower()
    return (
        "pos_embed" in lowered
        or "position_embedding" in lowered
        or "positional_embedding" in lowered
    )


def _is_attention_param(name: str) -> bool:
    parts = name.split(".")
    return "attention" in parts


def _is_muon_candidate(
    module: nn.Module, name: str, param: nn.Parameter
) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if name.startswith("lm_head."):
        return False
    if not name.endswith(".weight"):
        return False
    return param.ndim == 2


def _normuon_supported(device: torch.device) -> Tuple[bool, str]:
    if device.type != "cuda":
        return False, "NorMuon requires CUDA."
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(bf16_supported) and not bf16_supported():
        return False, "NorMuon requires CUDA bfloat16 support"
    return True, ""


def _collect_param_groups(
    model: nn.Module, *, include_muon: bool
) -> Dict[str, list[nn.Parameter]]:
    groups: Dict[str, list[nn.Parameter]] = {
        "decay": [],
        "attention": [],
        "token_embed": [],
        "task_embed": [],
        "no_decay": [],
        "muon": [],
        "muon_attention": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = model.get_submodule(module_name) if module_name else model
        is_attention = _is_attention_param(name)

        if include_muon and _is_muon_candidate(module, name, param):
            if is_attention:
                groups["muon_attention"].append(param)
            else:
                groups["muon"].append(param)
            continue

        if name.endswith(".bias"):
            groups["no_decay"].append(param)
            continue

        if _is_norm_module(module) or _is_positional_embedding_param(name):
            groups["no_decay"].append(param)
            continue

        if isinstance(module, nn.Embedding):
            if name.startswith("token_embedding."):
                groups["token_embed"].append(param)
            elif name.startswith("example_embedding.") or name.startswith(
                "dihedral_embedding."
            ):
                groups["task_embed"].append(param)
            else:
                groups["no_decay"].append(param)
            continue

        if is_attention:
            groups["attention"].append(param)
            continue

        if isinstance(module, nn.Linear):
            groups["decay"].append(param)
        else:
            groups["no_decay"].append(param)

    return groups


def _build_param_groups(
    model: nn.Module,
    weight_decay: float,
    attention_weight_decay: float,
    token_embedding_weight_decay: float,
    task_embedding_weight_decay: float,
) -> Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]:
    """Split params for Muon (linear weights only) and AdamW (everything else)."""
    groups = _collect_param_groups(model, include_muon=True)

    muon_groups = []
    if groups["muon"]:
        muon_groups.append({"params": groups["muon"], "weight_decay": weight_decay})
    if groups["muon_attention"]:
        muon_groups.append(
            {
                "params": groups["muon_attention"],
                "weight_decay": attention_weight_decay,
            }
        )

    adamw_groups = []
    if groups["decay"]:
        adamw_groups.append({"params": groups["decay"], "weight_decay": weight_decay})
    if groups["attention"]:
        adamw_groups.append(
            {"params": groups["attention"], "weight_decay": attention_weight_decay}
        )
    if groups["token_embed"]:
        adamw_groups.append(
            {
                "params": groups["token_embed"],
                "weight_decay": token_embedding_weight_decay,
            }
        )
    if groups["task_embed"]:
        adamw_groups.append(
            {
                "params": groups["task_embed"],
                "weight_decay": task_embedding_weight_decay,
            }
        )
    if groups["no_decay"]:
        adamw_groups.append({"params": groups["no_decay"], "weight_decay": 0.0})

    return muon_groups, adamw_groups


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state_dict: Dict[str, Any],
    device: torch.device,
) -> bool:
    def _is_torch_state_dict(value: Any) -> bool:
        return isinstance(value, dict) and "param_groups" in value

    candidate_state: Optional[Dict[str, Any]] = None
    if _is_torch_state_dict(state_dict):
        candidate_state = state_dict
    else:
        state_key = optimizer.__class__.__name__.lower()
        sub_state = state_dict.get(state_key)
        if _is_torch_state_dict(sub_state):
            candidate_state = sub_state

    if candidate_state is None:
        return False

    try:
        optimizer.load_state_dict(candidate_state)
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"Skipping optimizer state restore ({exc}).")
        return False
    _move_optimizer_state(optimizer, device)
    return True


def _optimizer_identity(optimizer: torch.optim.Optimizer) -> str:
    return optimizer.__class__.__name__.lower()


_OPTIMIZER_HPARAMS_IGNORE_KEYS = {
    "params",
    "fused",
    "foreach",
    "capturable",
    "differentiable",
}


def _sanitize_optimizer_value(value: Any) -> Optional[Any]:
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Number):
        return float(value)
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        sanitized = []
        for item in value:
            item_sanitized = _sanitize_optimizer_value(item)
            if item_sanitized is None and item is not None:
                return None
            sanitized.append(item_sanitized)
        return sanitized
    if isinstance(value, dict):
        sanitized_dict = {}
        for key, item in value.items():
            item_sanitized = _sanitize_optimizer_value(item)
            if item_sanitized is None and item is not None:
                return None
            sanitized_dict[key] = item_sanitized
        return sanitized_dict
    return None


def _param_groups_snapshot(
    param_groups: Sequence[Dict[str, Any]],
) -> Sequence[Dict[str, Any]]:
    snapshot = []
    for group in param_groups:
        entry: Dict[str, Any] = {}
        for key, value in group.items():
            if key in _OPTIMIZER_HPARAMS_IGNORE_KEYS:
                continue
            sanitized = _sanitize_optimizer_value(value)
            if sanitized is None and value is not None:
                continue
            entry[key] = sanitized
        snapshot.append(entry)
    return snapshot


def _checkpoint_optimizer_hparams(checkpoint: Dict[str, Any]) -> Optional[Any]:
    state_dict = checkpoint.get("optimizer_state")
    if isinstance(state_dict, dict) and "param_groups" in state_dict:
        return _param_groups_snapshot(state_dict.get("param_groups", []))
    return None


def _optimizer_values_match(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_optimizer_values_match(lv, rv) for lv, rv in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(
            _optimizer_values_match(left[key], right[key]) for key in left.keys()
        )
    return left == right


def _optimizer_hparams_changed(
    optimizer: torch.optim.Optimizer, checkpoint: Dict[str, Any]
) -> bool:
    saved = checkpoint.get("optimizer_hparams")
    if saved is None:
        saved = _checkpoint_optimizer_hparams(checkpoint)
    if saved is None:
        return False
    current = _param_groups_snapshot(optimizer.param_groups)
    return not _optimizer_values_match(current, saved)


def _apply_param_group_hparams(
    param_groups: Sequence[Dict[str, Any]],
    hparams: Optional[Sequence[Dict[str, Any]]],
) -> None:
    if not isinstance(hparams, Sequence):
        return
    for group, desired in zip(param_groups, hparams):
        if not isinstance(desired, dict):
            continue
        for key, value in desired.items():
            group[key] = value


def _optimizer_switch_detected(
    optimizer: torch.optim.Optimizer,
    checkpoint: Dict[str, Any],
) -> bool:
    checkpoint_name = checkpoint.get("optimizer_name")
    if checkpoint_name:
        return str(checkpoint_name).lower() != _optimizer_identity(optimizer)
    state_dict = checkpoint.get("optimizer_state")
    if isinstance(state_dict, dict) and "muon" in state_dict and "adamw" in state_dict:
        return True
    return False


def _build_optimizer(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    attention_weight_decay: float,
    token_embedding_weight_decay: float,
    task_embedding_weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer_name = str(getattr(args, "optimizer", "adamw")).lower()
    use_fused = device.type == "cuda"

    muon_groups, adamw_groups = _build_param_groups(
        model,
        args.weight_decay,
        attention_weight_decay,
        token_embedding_weight_decay,
        task_embedding_weight_decay,
    )

    if optimizer_name not in {"normuon"}:
        return AdamW(list(muon_groups) + list(adamw_groups), lr=args.adamw_lr, fused=use_fused)

    supported, reason = _normuon_supported(device)
    if not supported:
        print(f"NorMuon unavailable ({reason}); falling back to AdamW.")
        return AdamW(list(muon_groups) + list(adamw_groups), lr=args.adamw_lr, fused=use_fused)

    if not muon_groups:
        print("NorMuon requested but no eligible linear weights found; using AdamW.")
        return AdamW(list(adamw_groups), lr=args.adamw_lr, fused=use_fused)

    normuon_lr = getattr(args, "normuon_lr", None)
    normuon_lr = 0.02 if normuon_lr is None else float(normuon_lr)
    normuon_momentum = float(getattr(args, "normuon_momentum", 0.95))
    normuon_beta2 = float(getattr(args, "normuon_beta2", 0.95))

    adamw_lr = getattr(args, "adamw_lr", None)
    adamw_lr = 3e-4 if adamw_lr is None else float(adamw_lr)
    normuon_param_groups = []
    for group in muon_groups:
        normuon_group = dict(group)
        normuon_group["use_muon"] = True
        normuon_group["lr"] = normuon_lr
        normuon_group["momentum"] = normuon_momentum
        normuon_group["beta2"] = normuon_beta2
        normuon_param_groups.append(normuon_group)
    for group in adamw_groups:
        adam_group = dict(group)
        adam_group["use_muon"] = False
        adam_group["lr"] = adamw_lr
        normuon_param_groups.append(adam_group)
    return SingleDeviceNorMuonWithAuxAdam(normuon_param_groups)


# =============================================================================
# Checkpointing
# =============================================================================

def maybe_save_model(
    model: TinyTransformer,
    dataset: ARCExampleDataset,
    data_path: Path,
    save_path: Optional[Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    global_step: Optional[int] = None,
    epoch: Optional[int] = None,
    rng_state: Optional[Dict[str, Any]] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(model.config),
        "task_ids": list(dataset.task_ids),
        "data_path": str(data_path),
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
        checkpoint["optimizer_name"] = _optimizer_identity(optimizer)
        checkpoint["optimizer_hparams"] = _param_groups_snapshot(optimizer.param_groups)
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if global_step is not None:
        checkpoint["global_step"] = int(global_step)
    if epoch is not None:
        checkpoint["epoch"] = int(epoch)
    if rng_state is not None:
        checkpoint["rng_state"] = rng_state
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def _normalize_checkpoint_epochs(
    checkpoint_epochs: Optional[object],
    total_epochs: int,
) -> Optional[Set[int]]:
    if checkpoint_epochs is None:
        return None
    if isinstance(checkpoint_epochs, bool):
        raise TypeError("checkpoint_epochs must be an int or list of ints, not bool.")
    if isinstance(checkpoint_epochs, int):
        interval = int(checkpoint_epochs)
        if interval <= 0:
            return None
        return set(range(interval, total_epochs + 1, interval))
    if isinstance(checkpoint_epochs, Sequence) and not isinstance(
        checkpoint_epochs, (str, bytes)
    ):
        epochs: Set[int] = set()
        for item in checkpoint_epochs:
            if isinstance(item, bool):
                raise TypeError(
                    "checkpoint_epochs must be an int or list of ints, not bool."
                )
            try:
                epoch = int(item)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "checkpoint_epochs must be an int or list of ints."
                ) from exc
            if epoch <= 0:
                raise ValueError("checkpoint_epochs entries must be positive.")
            if epoch <= total_epochs:
                epochs.add(epoch)
        return epochs or None
    raise TypeError("checkpoint_epochs must be an int or list of ints.")


def _checkpoint_path_for_epoch(
    save_path: Path,
    epoch: int,
    total_epochs: int,
) -> Path:
    width = max(2, len(str(total_epochs)))
    suffix = save_path.suffix
    stem = save_path.stem
    return save_path.with_name(f"{stem}.epoch{epoch:0{width}d}{suffix}")


# =============================================================================
# Main Training Loop
# =============================================================================

def train_model(
    args: argparse.Namespace,
    model: TinyTransformer,
    dataloader: torch.utils.data.DataLoader,
    dataset: ARCExampleDataset,
    device: torch.device,
    data_path: Path,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> None:
    """Run the training loop only (no evaluation)."""
    if checkpoint is None:
        checkpoint = getattr(model, "_loaded_checkpoint", None)

    do_validate = getattr(args, "do_validate", True)
    val_dataloader = None

    if do_validate:
        val_batch_size = getattr(args, "val_batch_size", args.batch_size)
        print(f"Building validation dataloader (batch_size={val_batch_size})...")
        print("Building validation dataloader (reading hidden solutions)...")
        val_dataset = ARCExampleDataset(
            json_path=data_path,
            splits=("test",),
            include_outputs=True,
            load_test_solutions=True,
            max_seq_len=MAX_SEQ_LEN,
            task_whitelist=dataset.task_ids,
        )

        val_dataloader = create_dataloader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
        )
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        print("Validation disabled (skipping solutions.json load).")

    log_file = getattr(args, "train_log_file", None)
    if log_file is not None and not isinstance(log_file, Path):
        log_file = Path(log_file)
    train_log_mode = str(getattr(args, "train_log_mode", "10_steps"))
    log_location = str(getattr(args, "log_location", "both"))
    log_handle = None
    if log_location in ("file", "both") and log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("a")

    save_path = getattr(args, "save_path", None)
    if save_path is not None and not isinstance(save_path, Path):
        save_path = Path(save_path)
    checkpoint_schedule = _normalize_checkpoint_epochs(
        getattr(args, "checkpoint_epochs", None), args.epochs
    )

    attention_weight_decay = getattr(args, "attention_weight_decay", args.weight_decay)
    token_embedding_weight_decay = getattr(args, "token_embedding_weight_decay", 0.0)
    task_embedding_weight_decay = getattr(args, "task_embedding_weight_decay", 0.0)

    optimizer = _build_optimizer(
        args,
        model,
        device,
        attention_weight_decay,
        token_embedding_weight_decay,
        task_embedding_weight_decay,
    )
    desired_optimizer_hparams = _param_groups_snapshot(optimizer.param_groups)

    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "drop_last"):
        batch_sampler.drop_last = True

    step = int(checkpoint.get("global_step", 0)) if checkpoint else 0
    steps_per_epoch = len(dataloader)

    start_epoch = 0
    if checkpoint:
        saved_epoch = checkpoint.get("epoch")
        if saved_epoch is None:
            saved_epoch = checkpoint.get("epochs_completed")
        if saved_epoch is not None:
            start_epoch = int(saved_epoch)
        elif step > 0:
            if steps_per_epoch > 0:
                start_epoch = step // steps_per_epoch
        if start_epoch > args.epochs:
            print(
                f"Checkpoint has {start_epoch} epochs completed; "
                f"configured epochs={args.epochs}. Nothing left to train."
            )
            start_epoch = args.epochs

    if checkpoint and step > 0:
        print(f"Resuming training from global_step={step}.")
    if checkpoint and start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch + 1}/{args.epochs}.")

    resume_warmup = False
    if checkpoint and _optimizer_switch_detected(optimizer, checkpoint):
        resume_warmup = True
        print("Detected optimizer change; resetting LR schedule.")
    optimizer_hparams_changed = False
    if checkpoint and _optimizer_hparams_changed(optimizer, checkpoint):
        optimizer_hparams_changed = True
        resume_warmup = True
        print("Detected optimizer hyperparameter change; applying new settings.")

    if checkpoint and "optimizer_state" in checkpoint:
        restored = _load_optimizer_state(
            optimizer, checkpoint["optimizer_state"], device
        )
        if restored:
            if optimizer_hparams_changed:
                _apply_param_group_hparams(optimizer.param_groups, desired_optimizer_hparams)
                for group in optimizer.param_groups:
                    if "initial_lr" in group and "lr" in group:
                        group["initial_lr"] = group.get("lr", group["initial_lr"])
            print("Restored optimizer state from checkpoint.")
        else:
            resume_warmup = True
            print("Skipping optimizer state restore (incompatible optimizer).")
    elif checkpoint:
        resume_warmup = True
        print("No optimizer state in checkpoint; starting with fresh optimizer.")

    # Linear warmup + WSD schedule
    total_epochs = max(0.0, float(args.epochs))
    steps_per_epoch = max(1, steps_per_epoch)
    warmup_pct = float(getattr(args, "warmup_pct", 0.02))
    warmup_pct = max(0.0, min(1.0, warmup_pct))
    warmup_epochs = total_epochs * warmup_pct
    warmup_epochs = max(0.0, min(total_epochs, warmup_epochs))
    min_lr_factor = float(getattr(args, "lr_floor", 0.01))
    min_lr_factor = max(0.0, min(1.0, min_lr_factor))
    decay_start_pct = float(getattr(args, "wsd_decay_start_pct", 0.8))
    decay_start_pct = max(0.0, min(1.0, decay_start_pct))
    decay_start_epoch = total_epochs * decay_start_pct
    decay_start_epoch = max(decay_start_epoch, warmup_epochs)
    decay_start_epoch = min(decay_start_epoch, total_epochs)
    decay_epochs = max(1e-8, total_epochs - decay_start_epoch)

    def base_lr_lambda(current_epoch: float):
        if warmup_epochs > 0 and current_epoch < warmup_epochs:
            return float(current_epoch) / float(warmup_epochs)
        if decay_start_epoch >= total_epochs or current_epoch < decay_start_epoch:
            return 1.0
        progress = float(current_epoch - decay_start_epoch) / float(decay_epochs)
        progress = max(0.0, min(1.0, progress))
        linear = 1.0 - progress
        return min_lr_factor + (1.0 - min_lr_factor) * linear

    lr_lambda = base_lr_lambda
    resume_start_epoch = float(start_epoch)
    if resume_warmup and resume_start_epoch > 0:
        remaining_epochs = max(0.0, total_epochs - resume_start_epoch)
        if remaining_epochs > 0:
            resume_warmup_epochs = remaining_epochs * 0.02
            min_warmup_epochs = 1.0 / float(steps_per_epoch)
            resume_warmup_epochs = max(min_warmup_epochs, resume_warmup_epochs)
            resume_warmup_epochs = min(resume_warmup_epochs, remaining_epochs)
            resume_warmup_start = resume_start_epoch
            resume_warmup_end = resume_start_epoch + resume_warmup_epochs
            resume_target = base_lr_lambda(resume_warmup_end)

            def lr_lambda(current_epoch: float):
                if current_epoch < resume_warmup_start:
                    return base_lr_lambda(current_epoch)
                if current_epoch < resume_warmup_end:
                    progress = float(current_epoch - resume_warmup_start) / float(
                        resume_warmup_epochs
                    )
                    return resume_target * progress
                return base_lr_lambda(current_epoch)

            print(
                f"Applying resume LR warmup for {resume_warmup_epochs:.3f} epochs."
            )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler_restored = False
    if checkpoint and "scheduler_state" in checkpoint and not resume_warmup:
        scheduler_state = checkpoint["scheduler_state"]
        last_epoch = (
            scheduler_state.get("last_epoch")
            if isinstance(scheduler_state, dict)
            else None
        )
        if isinstance(last_epoch, (int, float)) and abs(
            float(last_epoch) - float(start_epoch)
        ) > 1.0:
            print(
                "Skipping scheduler state restore; using epoch-based schedule from args."
            )
        else:
            scheduler.load_state_dict(scheduler_state)
            scheduler_restored = True
            print("Restored scheduler state from checkpoint.")
    elif checkpoint and "scheduler_state" in checkpoint and resume_warmup:
        print(
            "Skipping scheduler state restore; recomputing schedule from args."
        )
    if not scheduler_restored and start_epoch > 0:
        if resume_warmup:
            resume_base_factor = base_lr_lambda(resume_start_epoch)
            for base_lr, group in zip(scheduler.base_lrs, optimizer.param_groups):
                group["lr"] = base_lr * resume_base_factor
        else:
            scheduler.step(float(start_epoch))

    # Compile model for training speedup
    if hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model for training speedup...")
        training_model = torch.compile(model)
    else:
        training_model = model

    augmentor = getattr(dataloader, "augmentor", None)

    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            if augmentor is not None:
                augmentor.set_epoch(epoch)

            step = train_one_epoch(
                model=training_model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                grad_clip=args.grad_clip,
                gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
                start_step=step,
                scheduler=scheduler,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
                train_log_mode=train_log_mode,
                log_location=log_location,
                log_handle=log_handle,
            )

            if val_dataloader is not None:
                val_loss = validate_one_epoch(
                    model=model,
                    dataloader=val_dataloader,
                    device=device,
                )

                val_msg = (
                    f"Epoch {epoch + 1} finished. Validation Output Loss: {val_loss:.4f}"
                )
                _emit_log(val_msg, log_location, log_handle)

            if checkpoint_schedule and save_path is not None:
                epoch_num = epoch + 1
                if epoch_num in checkpoint_schedule:
                    rng_state = capture_rng_state(device)
                    epoch_save_path = _checkpoint_path_for_epoch(
                        save_path, epoch_num, args.epochs
                    )
                    maybe_save_model(
                        model,
                        dataset,
                        data_path,
                        epoch_save_path,
                        optimizer=optimizer,
                        global_step=step,
                        epoch=epoch_num,
                        rng_state=rng_state,
                        scheduler=scheduler,
                    )

        rng_state = capture_rng_state(device)
        maybe_save_model(
            model,
            dataset,
            data_path,
            save_path,
            optimizer=optimizer,
            global_step=step,
            epoch=args.epochs,
            rng_state=rng_state,
            scheduler=scheduler,
        )
    finally:
        if log_handle is not None:
            log_handle.close()
