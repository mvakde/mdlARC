"""Training pipeline for the ARC transformer model.

Contains: optimizer building (including NorMuon), training loop, and checkpointing.
For model/data building, see build.py.
"""

import argparse
import json
import math
import numbers
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, TextIO, Tuple

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import distributed_training as distributed
import parameter_efficient

from checkpoint_io import save_checkpoint_atomically, save_rolling_checkpoint, validate_checkpoint_keep_last
from build import should_reset_training_state
from common import (
    ARCExampleDataset,
    capture_rng_state,
    sequence_format_for_attention,
    GRID_CHAIN_SEQUENCE_FORMAT,
    SUPERVISED_GRID_CHAIN_SEQUENCE_FORMAT,
)
from tinytransformer import (
    RMSNorm,
    TinyTransformer,
)


_BATCH_DEVICE_DTYPES: Dict[str, torch.dtype] = {
    # Keep reduced integer dtypes from collation and let the model upcast only
    # where it actually needs to. The explicit bool cast is still useful for
    # any non-packed path that carries an attention mask.
    "attention_mask": torch.bool,
}


def _enable_compile_scalar_capture() -> None:
    compiler = getattr(torch, "compiler", None)
    compiler_config = getattr(compiler, "config", None)
    if compiler_config is not None and hasattr(compiler_config, "capture_scalar_outputs"):
        compiler_config.capture_scalar_outputs = True
    if compiler_config is not None and hasattr(
        compiler_config, "capture_dynamic_output_shape_ops"
    ):
        compiler_config.capture_dynamic_output_shape_ops = True
    dynamo = getattr(torch, "_dynamo", None)
    dynamo_config = getattr(dynamo, "config", None)
    if dynamo_config is not None and hasattr(dynamo_config, "capture_scalar_outputs"):
        dynamo_config.capture_scalar_outputs = True
    if dynamo_config is not None and hasattr(
        dynamo_config, "capture_dynamic_output_shape_ops"
    ):
        dynamo_config.capture_dynamic_output_shape_ops = True


def _move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        target_dtype = _BATCH_DEVICE_DTYPES.get(key)
        if key == "target_demo_layout":
            moved[key] = value.to(device, non_blocking=non_blocking)
        elif isinstance(value, torch.Tensor):
            if target_dtype is None:
                moved[key] = value.to(device=device, non_blocking=non_blocking)
            else:
                moved[key] = value.to(
                    device=device,
                    dtype=target_dtype,
                    non_blocking=non_blocking,
                )
        else:
            moved[key] = value
    return moved


class _DeviceBatchIterator:
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> None:
        self._iter = iter(dataloader)
        self._device = device
        self._max_batches = None if max_batches is None else max(0, int(max_batches))
        self._loaded_batches = 0
        self._stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._next_batch: Optional[Dict[str, Any]] = None
        self._preload()

    def _preload(self) -> None:
        if self._max_batches is not None and self._loaded_batches >= self._max_batches:
            self._next_batch = None
            return
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return
        self._loaded_batches += 1
        if self._stream is None:
            self._next_batch = _move_batch_to_device(batch, self._device)
            return
        with torch.cuda.stream(self._stream):
            self._next_batch = _move_batch_to_device(batch, self._device)

    def __iter__(self) -> "_DeviceBatchIterator":
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._next_batch is None:
            raise StopIteration
        if self._stream is not None:
            current_stream = torch.cuda.current_stream(device=self._device)
            current_stream.wait_stream(self._stream)
            def record(value):
                if isinstance(value, torch.Tensor):
                    value.record_stream(current_stream)
                elif isinstance(value, dict):
                    for child in value.values():
                        record(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        record(child)
            record(self._next_batch)
        batch = self._next_batch
        self._preload()
        return batch


def _expand_lr_factors(value: Any, expected: int) -> List[float]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        factors = [float(item) for item in value]
        if len(factors) != expected:
            raise ValueError(
                f"LR schedule returned {len(factors)} factors for "
                f"{expected} optimizer groups."
            )
        return factors
    return [float(value)] * expected


def _scale_lr_factors(value: Any, scale: float) -> Any:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [float(item) * scale for item in value]
    return float(value) * scale


class _FractionalLambdaLR:
    """Minimal LambdaLR replacement that accepts fractional progress values."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_lambda: Callable[[float], Any],
    ) -> None:
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.base_lrs = []
        for group in self.optimizer.param_groups:
            initial_lr = float(group.get("initial_lr", group["lr"]))
            group["initial_lr"] = initial_lr
            self.base_lrs.append(initial_lr)
        self.last_epoch = -1.0
        # Apply progress zero immediately.  Unlike torch's LambdaLR, this
        # fractional replacement does not have a base-class constructor that
        # performs the initial scheduler step.  Leaving the optimizer at its
        # base LR makes the first update run at full strength before warmup.
        self._last_lr: List[float] = []
        self._apply_lrs(0.0)

    def _apply_lrs(self, progress: float) -> None:
        factors = _expand_lr_factors(
            self.lr_lambda(float(progress)), len(self.optimizer.param_groups)
        )
        self._last_lr = []
        for base_lr, group, factor in zip(
            self.base_lrs, self.optimizer.param_groups, factors
        ):
            lr = base_lr * factor
            group["lr"] = lr
            self._last_lr.append(lr)

    def step(self, epoch: Optional[float] = None) -> None:
        progress = 0.0 if epoch is None and self.last_epoch < 0 else epoch
        if progress is None:
            progress = self.last_epoch + 1.0
        self.last_epoch = float(progress)
        self._apply_lrs(self.last_epoch)

    def get_last_lr(self) -> List[float]:
        return list(self._last_lr)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "base_lrs": list(self.base_lrs),
            "last_epoch": float(self.last_epoch),
            "_last_lr": list(self._last_lr),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        base_lrs = state_dict.get("base_lrs")
        if isinstance(base_lrs, Sequence):
            self.base_lrs = [float(value) for value in base_lrs]
        self.last_epoch = float(state_dict.get("last_epoch", self.last_epoch))
        last_lr = state_dict.get("_last_lr")
        if isinstance(last_lr, Sequence):
            self._last_lr = [float(value) for value in last_lr]
        else:
            factors = _expand_lr_factors(
                self.lr_lambda(self.last_epoch), len(self.base_lrs)
            )
            self._last_lr = [
                base_lr * factor
                for base_lr, factor in zip(self.base_lrs, factors)
            ]
        for group, base_lr, lr in zip(
            self.optimizer.param_groups, self.base_lrs, self._last_lr
        ):
            group["initial_lr"] = base_lr
            group["lr"] = lr


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


def _normuon_update(
    grad,
    momentum,
    second_momentum,
    beta=0.95,
    beta2=0.95,
    shape_scale=1.0,
    ns_steps=5,
    nesterov=True,
):
    momentum.lerp_(grad, 1 - beta)
    update = torch.lerp(grad, momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4 and not (
        update.size(-3) == 3 and update.size(-2) == update.size(-1)
    ):
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
    update.mul_(shape_scale)
    return update


def _muon_shape_scale(grad: torch.Tensor) -> float:
    rows = int(grad.size(-2))
    cols = int(grad.size(-1))
    return max(1.0, math.sqrt(float(rows) / float(cols)))


def _reshape_qkv_for_muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    second_momentum: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = grad.size(-2)
    cols = grad.size(-1)
    if rows % 3 != 0 or rows // 3 != cols:
        raise ValueError(
            "QKV split Muon expects a packed [3 * d_model, d_model] weight."
        )
    qkv_shape = (*grad.shape[:-2], 3, cols, cols)
    second_momentum_shape = (*second_momentum.shape[:-2], 3, cols, 1)
    return (
        grad.reshape(qkv_shape),
        momentum.reshape(qkv_shape),
        second_momentum.reshape(second_momentum_shape),
    )


def _adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


def get_muon_momentum(
    step: int,
    total_steps: Optional[int] = None,
    muon_warmup_steps: int = 300,
    muon_cooldown_steps: int = 50,
    momentum_min: float = 0.85,
    momentum_max: float = 0.95,
) -> float:
    """Return the scheduled NorMuon momentum for a 1-based optimizer step."""
    step = max(1, int(step))
    if momentum_max < momentum_min:
        momentum_min, momentum_max = momentum_max, momentum_min

    warmup_steps = max(0, int(muon_warmup_steps))
    cooldown_steps = max(0, int(muon_cooldown_steps))
    total = None if total_steps is None else max(0, int(total_steps))

    factor = 1.0
    if warmup_steps > 0 and step <= warmup_steps:
        if warmup_steps == 1:
            factor = 0.0
        else:
            factor = min(factor, float(step - 1) / float(warmup_steps - 1))

    if total is not None and cooldown_steps > 0:
        cooldown_start = max(1, total - cooldown_steps + 1)
        if step >= cooldown_start:
            if cooldown_steps == 1:
                factor = 0.0
            else:
                factor = min(
                    factor,
                    float(max(0, total - step)) / float(cooldown_steps - 1),
                )

    factor = max(0.0, min(1.0, factor))
    return float(momentum_min + (momentum_max - momentum_min) * factor)


class SingleDeviceNorMuonWithAuxAdam(torch.optim.Optimizer):
    """NorMuon optimizer: Muon for matrix weights, AdamW for everything else."""

    def __init__(self, param_groups, normuon_bucket_size: int = 0):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["momentum_min"] = group.get(
                    "momentum_min", min(0.85, float(group["momentum"]))
                )
                group["muon_warmup_steps"] = int(group.get("muon_warmup_steps", 300))
                group["muon_cooldown_steps"] = int(group.get("muon_cooldown_steps", 50))
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.1)
                group["qkv_split"] = bool(group.get("qkv_split", False))
                group["group_name"] = str(group.get("group_name", "muon"))
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "momentum",
                    "momentum_min",
                    "muon_warmup_steps",
                    "muon_cooldown_steps",
                    "beta2",
                    "weight_decay",
                    "qkv_split",
                    "group_name",
                    "use_muon",
                }
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.1)
                group["group_name"] = str(group.get("group_name", "adamw"))
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "group_name",
                    "use_muon",
                }
        super().__init__(param_groups, dict())

        self._normuon_bucket_size = max(0, int(normuon_bucket_size or 0))
        self._normuon_update_fn: Callable[..., torch.Tensor] = _normuon_update
        self._adam_update_fn: Callable[..., torch.Tensor] = _adam_update
        self._compiled_update_kernels = False
        self._zero_scalars: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._scalar_buffers: Dict[Tuple[str, torch.device, torch.dtype], torch.Tensor] = {}
        self._has_muon = any(group["use_muon"] for group in param_groups)
        self._muon_step = 0
        self._muon_total_steps: Optional[int] = None

        disable_compile = str(
            os.environ.get("MDLARC_DISABLE_OPTIMIZER_COMPILE", "0")
        ).lower() in {"1", "true", "yes", "on"}
        has_cuda_params = any(
            p.is_cuda for group in param_groups for p in group["params"]
        )
        if has_cuda_params and not disable_compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("NorMuon optimizer compilation requires torch.compile.")
            _enable_compile_scalar_capture()
            self._normuon_update_fn = torch.compile(
                _normuon_update,
                mode="default",
                fullgraph=False,
                dynamic=True,
            )
            self._adam_update_fn = torch.compile(
                _adam_update,
                mode="default",
                fullgraph=False,
                dynamic=True,
            )
            self._compiled_update_kernels = True
            print("Enabled compiled NorMuon optimizer update path.")

    def _get_zero_grad(self, p: torch.Tensor) -> torch.Tensor:
        key = (p.device, p.dtype)
        zero = self._zero_scalars.get(key)
        if zero is None:
            zero = torch.zeros((), device=p.device, dtype=p.dtype)
            self._zero_scalars[key] = zero
        return zero.expand_as(p)

    def _get_scalar_buffer(
        self, name: str, reference: torch.Tensor, value: float
    ) -> torch.Tensor:
        key = (name, reference.device, reference.dtype)
        scalar = self._scalar_buffers.get(key)
        if scalar is None:
            scalar = torch.empty((), device=reference.device, dtype=reference.dtype)
            self._scalar_buffers[key] = scalar
        scalar.fill_(value)
        return scalar

    def configure_muon_schedule(
        self, total_steps: Optional[int], start_step: int = 0
    ) -> None:
        self._muon_total_steps = (
            None if total_steps is None else max(0, int(total_steps))
        )
        self._muon_step = max(0, int(start_step))

    def _call_normuon_update(
        self,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        second_momentum: torch.Tensor,
        beta: float,
        beta2: float,
        shape_scale: float,
    ) -> torch.Tensor:
        # Keep scheduled scalars as tensors so the compiled kernel input types stay stable.
        beta_tensor = self._get_scalar_buffer("momentum", grad, beta)
        beta2_tensor = self._get_scalar_buffer("beta2", grad, beta2)
        shape_scale_tensor = self._get_scalar_buffer("shape_scale", grad, shape_scale)
        if not self._compiled_update_kernels:
            return _normuon_update(
                grad,
                momentum,
                second_momentum,
                beta=beta_tensor,
                beta2=beta2_tensor,
                shape_scale=shape_scale_tensor,
            )
        return self._normuon_update_fn(
            grad,
            momentum,
            second_momentum,
            beta=beta_tensor,
            beta2=beta2_tensor,
            shape_scale=shape_scale_tensor,
        )

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
        return self._adam_update_fn(grad, buf1, buf2, step, betas, eps)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        current_muon_step: Optional[int] = None
        if self._has_muon:
            self._muon_step += 1
            current_muon_step = self._muon_step

        for group in self.param_groups:
            if group["use_muon"]:
                lr = float(group["lr"])
                momentum_max = float(group["momentum"])
                beta = get_muon_momentum(
                    step=current_muon_step if current_muon_step is not None else 1,
                    total_steps=self._muon_total_steps,
                    muon_warmup_steps=int(group["muon_warmup_steps"]),
                    muon_cooldown_steps=int(group["muon_cooldown_steps"]),
                    momentum_min=float(group["momentum_min"]),
                    momentum_max=momentum_max,
                )
                beta2 = float(group["beta2"])
                weight_decay = float(group["weight_decay"])
                qkv_split = bool(group["qkv_split"])
                normuon_bucket_size = self._normuon_bucket_size

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

                def apply_muon_updates(
                    params_to_update: List[torch.Tensor],
                    updates: List[torch.Tensor],
                    params_with_grad: List[torch.Tensor],
                ) -> None:
                    if weight_decay and params_with_grad:
                        torch._foreach_mul_(params_with_grad, 1 - lr * weight_decay)
                    if params_to_update:
                        torch._foreach_add_(params_to_update, updates, alpha=-lr)

                def process_muon_bucket(
                    bucket: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]
                ) -> None:
                    if len(bucket) == 1:
                        p, grad, momentum_buffer, second_momentum_buffer, had_grad = (
                            bucket[0]
                        )
                        if qkv_split:
                            grad, momentum_buffer, second_momentum_buffer = (
                                _reshape_qkv_for_muon_update(
                                    grad,
                                    momentum_buffer,
                                    second_momentum_buffer,
                                )
                            )
                        shape_scale = _muon_shape_scale(grad)
                        update = self._call_normuon_update(
                            grad,
                            momentum_buffer,
                            second_momentum_buffer,
                            beta,
                            beta2,
                            shape_scale,
                        )
                        apply_muon_updates(
                            [p],
                            [update.reshape_as(p)],
                            [p] if weight_decay and had_grad else [],
                        )
                        return

                    grad_batch = torch.stack([item[1] for item in bucket], dim=0)
                    momentum_batch = torch.stack([item[2] for item in bucket], dim=0)
                    second_momentum_batch = torch.stack(
                        [item[3] for item in bucket], dim=0
                    )
                    if qkv_split:
                        grad_batch, momentum_batch, second_momentum_batch = (
                            _reshape_qkv_for_muon_update(
                                grad_batch,
                                momentum_batch,
                                second_momentum_batch,
                            )
                        )
                    shape_scale = _muon_shape_scale(grad_batch)
                    update_batch = self._call_normuon_update(
                        grad_batch,
                        momentum_batch,
                        second_momentum_batch,
                        beta,
                        beta2,
                        shape_scale,
                    )

                    params_to_update: List[torch.Tensor] = []
                    updates: List[torch.Tensor] = []
                    params_with_grad: List[torch.Tensor] = []
                    for idx, (
                        p,
                        _grad,
                        momentum_buffer,
                        second_momentum_buffer,
                        had_grad,
                    ) in enumerate(bucket):
                        momentum_buffer.copy_(
                            momentum_batch[idx].reshape_as(momentum_buffer)
                        )
                        second_momentum_buffer.copy_(
                            second_momentum_batch[idx].reshape_as(
                                second_momentum_buffer
                            )
                        )
                        params_to_update.append(p)
                        updates.append(update_batch[idx].reshape_as(p))
                        if weight_decay and had_grad:
                            params_with_grad.append(p)

                    apply_muon_updates(params_to_update, updates, params_with_grad)

                for bucket in buckets.values():
                    bucket_size = (
                        len(bucket) if normuon_bucket_size <= 0 else normuon_bucket_size
                    )
                    for start in range(0, len(bucket), bucket_size):
                        process_muon_bucket(bucket[start : start + bucket_size])
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
    if distributed.rank() != 0:
        return
    if log_location in ("terminal", "both"):
        print(message)
    if log_location in ("file", "both") and log_handle is not None:
        log_handle.write(message + "\n")
        log_handle.flush()


def _full_accumulation_minibatches(
    minibatch_steps: int,
    gradient_accumulation_steps: int,
) -> int:
    accum_steps = max(1, int(gradient_accumulation_steps))
    minibatch_steps = max(0, int(minibatch_steps))
    return (minibatch_steps // accum_steps) * accum_steps


def _format_log_prefix(log_prefix: str, message: str) -> str:
    prefix = str(log_prefix or "").strip()
    if not prefix:
        return message
    return f"{prefix} {message}"


def _format_accuracy(correct: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * float(correct) / float(total):.2f}%"


def _mtp_loss_multiplier(progress: float, decay: Sequence[float]) -> float:
    """Scale auxiliary loss from 1 to the final multiplier over the full run."""
    start, end, final_multiplier = decay
    if progress <= start:
        return 1.0
    if progress >= end:
        return float(final_multiplier)
    return final_multiplier + (1.0 - final_multiplier) * (end - progress) / (end - start)


def _snapshot_residual_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    model = distributed.unwrap(model)
    return {
        f"{idx}.{name}": projection.weight.detach().clone()
        for idx, block in enumerate(model.blocks)
        for name, projection in (("attention", block.attention.out_proj),
                                 ("ff", block.ff.fc_out))
    }


@torch.no_grad()
def _log_depth_diagnostics(model, before, grad_norm, step, location, handle, prefix):
    model = distributed.unwrap(model)
    names, values = [], []
    for idx, block in enumerate(model.blocks):
        for name, projection in (("attention", block.attention.out_proj),
                                 ("ff", block.ff.fc_out)):
            key = f"{idx}.{name}"
            old = before[key]
            old_norm = old.float().norm()
            delta_norm = (projection.weight.detach() - old).float().norm()
            names.extend((f"{key}.weight_norm", f"{key}.update_norm", f"{key}.update_ratio"))
            values.extend((old_norm, delta_norm, delta_norm / old_norm.clamp_min(1e-12)))
        for name in ("hidden_state_scale", "token_input_skip_scale", "example_input_skip_scale"):
            names.append(f"{idx}.{name}")
            values.append(getattr(block, name).detach().float())
        if hasattr(block, "dihedral_input_skip_scale"):
            names.append(f"{idx}.dihedral_input_skip_scale")
            values.append(block.dihedral_input_skip_scale.detach().float())
    if grad_norm is not None:
        names.append("grad_norm_before_clip")
        values.append(grad_norm.float())
    stats = torch.stack(values)
    if not bool(torch.isfinite(stats).all()):
        raise FloatingPointError(f"Nonfinite depth diagnostics at minibatch {step}.")
    record = dict(zip(names, stats.tolist()))
    _emit_log(_format_log_prefix(prefix, f"depth_diagnostics step={step} " + json.dumps(record)),
              location, handle)


def train_one_epoch(
    model: TinyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    gradient_accumulation_steps: int = 1,
    start_step: int = 0,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    train_log_mode: str = "10_steps",
    log_location: str = "both",
    log_handle: Optional[TextIO] = None,
    log_prefix: str = "",
    total_epochs: Optional[int] = None,
    mtp_decay: Sequence[float] = (0.8, 0.9, 1.0),
    depth_diagnostics_every: int = 0,
) -> int:
    model.train()
    step = start_step
    use_amp = device.type == "cuda"

    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)

    accum_steps = max(1, int(gradient_accumulation_steps or 1))
    if not getattr(getattr(dataloader, "batch_sampler", None), "preserve_all_pairs", False):
        steps_per_epoch = _full_accumulation_minibatches(steps_per_epoch, accum_steps)
    if steps_per_epoch <= 0:
        return step

    accum_index = 0
    accum_target = accum_steps
    dataloader_length = steps_per_epoch
    optimizer_step = 0

    window_output_sum = torch.zeros((), device=device, dtype=torch.float32)
    window_count = 0

    epoch_output_sum = torch.zeros((), device=device, dtype=torch.float32)
    epoch_count = 0
    config = getattr(distributed.unwrap(model), "config", None)
    use_mtp = (
        getattr(config, "n_predict", 1) > 1
        and getattr(config, "mtp_weight", 0.0) > 0
    )
    track_mtp = (
        use_mtp
        and train_log_mode != "never"
        and log_location != "none"
    )
    track_next_latent = (
        getattr(config, "next_latent_prediction", False)
        and (getattr(config, "next_latent_regression_weight", 1.0)
             or getattr(config, "next_latent_kl_weight", 1.0))
        and train_log_mode != "never" and log_location != "none"
    )
    track_aux = track_mtp or track_next_latent
    window_mtp_sum = torch.zeros_like(window_output_sum) if track_mtp else None
    window_total_sum = torch.zeros_like(window_output_sum) if track_aux else None
    epoch_mtp_sum = torch.zeros_like(epoch_output_sum) if track_mtp else None
    epoch_total_sum = torch.zeros_like(epoch_output_sum) if track_aux else None
    window_latent_sums = torch.zeros(3, device=device) if track_next_latent else None
    epoch_latent_sums = torch.zeros(3, device=device) if track_next_latent else None
    report_epoch_metrics = (
        train_log_mode != "never" and log_location != "none"
    )
    epoch_metric_counts = torch.zeros(6, device=device, dtype=torch.long)

    def loss_summary(output_sum, mtp_sum, total_sum, count, latent_sums=None) -> str:
        names, sums = ["loss"], [output_sum]
        if track_mtp:
            names.append("mtp_loss")
            sums.append(mtp_sum)
        if track_next_latent:
            names.extend(("next_latent_loss", "next_latent_regression_loss", "next_latent_kl_loss"))
            sums.extend(latent_sums.unbind())
        if track_aux:
            names.append("total_loss")
            sums.append(total_sum)
        # Synchronize only when logging, with one read for every loss.
        averages = distributed.summed(torch.stack(sums)) / (count * distributed.world_size())
        return " ".join(f"{name}={value:.4f}" for name, value in zip(names, averages.tolist()))

    def maybe_log_window(force: bool = False) -> None:
        nonlocal window_count
        should_log = False
        if train_log_mode == "step":
            should_log = True
        elif train_log_mode == "10_steps":
            should_log = (optimizer_step % 10 == 0) or force
        if not should_log or window_count == 0:
            return

        losses = loss_summary(window_output_sum, window_mtp_sum, window_total_sum, window_count, window_latent_sums)
        current_lr = (
            scheduler.get_last_lr()[0]
            if scheduler
            else optimizer.param_groups[0]["lr"]
        )
        if epoch is None:
            prefix = f"step={optimizer_step}"
        else:
            prefix = f"epoch={epoch + 1} step={optimizer_step}"
        log_msg = _format_log_prefix(
            log_prefix,
            f"{prefix} lr={current_lr:.2e} {losses}",
        )
        _emit_log(log_msg, log_location, log_handle)
        window_output_sum.zero_()
        if track_mtp:
            window_mtp_sum.zero_()
        if track_aux:
            window_total_sum.zero_()
        if track_next_latent:
            window_latent_sums.zero_()
        window_count = 0

    last_batch_idx = None
    for batch_idx, batch in enumerate(
        _DeviceBatchIterator(dataloader, device, max_batches=dataloader_length)
    ):
        last_batch_idx = batch_idx
        step += 1
        input_ids = batch["input_ids"]
        sep_indices = batch["sep_indices"]
        cu_seqlens = batch.get("cu_seqlens")
        if cu_seqlens is not None:
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
                attention_mask = batch["attention_mask"]
        example_ids = batch["example_ids"]
        positions_3d = batch["positions_3d"]
        if accum_index == 0:
            optimizer.zero_grad(set_to_none=True)
            if dataloader_length is not None:
                remaining = dataloader_length - batch_idx
                accum_target = min(accum_steps, remaining)
            else:
                accum_target = accum_steps

        sync_context = (
            model.no_sync() if isinstance(model, DistributedDataParallel)
            and accum_index + 1 < accum_target else nullcontext()
        )
        with sync_context:
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                outputs = model(
                    input_ids,
                    example_ids,
                    dihedral_ids=batch.get("dihedral_ids"),
                    pair_identity_ids=batch.get("pair_identity_ids"),
                    **({"target_demo_layout": batch["target_demo_layout"]}
                       if "target_demo_layout" in batch else {}),
                    attention_mask=attention_mask,
                    sep_indices=sep_indices,
                    positions_3d=positions_3d,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    compute_training_metrics=report_epoch_metrics,
                )
                outputs = distributed.normalize_losses(outputs)
                loss = outputs["loss"]
                if distributed.active():
                    loss = outputs["output_loss"]
                    if use_mtp:
                        loss = loss + config.mtp_weight * outputs["mtp_loss"]
                    if outputs.get("next_latent_loss") is not None:
                        loss = loss + outputs["next_latent_loss"]
                    if outputs.get("loop_exit_loss") is not None:
                        loss = loss + config.loop_exit_weight * outputs["loop_exit_loss"]

            if use_mtp and total_epochs is not None:
                # Use fractional epoch progress at the minibatch's start, including
                # completed epochs on resume. Keep changing scalars outside the
                # compiled model so the decay does not trigger recompilation.
                epoch_progress = float(epoch or 0) + batch_idx / float(steps_per_epoch)
                progress = epoch_progress / total_epochs
                multiplier = _mtp_loss_multiplier(progress, mtp_decay)
                loss = (
                    outputs["output_loss"]
                    + config.mtp_weight * multiplier * outputs["mtp_loss"]
                )
                if outputs.get("next_latent_loss") is not None:
                    loss = loss + outputs["next_latent_loss"]
                if outputs.get("loop_exit_loss") is not None:
                    loss = loss + config.loop_exit_weight * outputs["loop_exit_loss"]

            batch_loss = outputs["output_loss"].detach().float()
            window_output_sum += batch_loss
            epoch_output_sum += batch_loss
            if track_mtp:
                batch_mtp = outputs["mtp_loss"].detach().float()
                window_mtp_sum.add_(batch_mtp)
                epoch_mtp_sum.add_(batch_mtp)
            if track_aux:
                batch_total = loss.detach().float()
                window_total_sum.add_(batch_total)
                epoch_total_sum.add_(batch_total)
            if track_next_latent:
                batch_latent = torch.stack([outputs[key].detach().float() for key in (
                    "next_latent_loss", "next_latent_regression_loss", "next_latent_kl_loss")])
                window_latent_sums.add_(batch_latent)
                epoch_latent_sums.add_(batch_latent)
            window_count += 1
            epoch_count += 1
            if report_epoch_metrics:
                metric_counts = outputs.get("training_metric_counts")
                if metric_counts is None:
                    raise RuntimeError("Training metrics were requested but not returned.")
                epoch_metric_counts.add_(metric_counts.detach())

            loss = loss / accum_target
            loss.backward()
        accum_index += 1

        if accum_index >= accum_target:
            diagnostic_due = depth_diagnostics_every > 0 and (
                step <= accum_steps or step % (depth_diagnostics_every * accum_steps) == 0
            )
            if diagnostic_due and not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(f"Nonfinite loss at minibatch {step}.")
            before = _snapshot_residual_weights(model) if diagnostic_due else None
            grad_norm = None
            if grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip,
                    error_if_nonfinite=depth_diagnostics_every > 0,
                )
            if scheduler is not None:
                if epoch is None or steps_per_epoch is None:
                    scheduler.step()
                else:
                    epoch_progress = float(epoch) + (batch_idx + 1) / float(
                        steps_per_epoch
                    )
                    scheduler.step(epoch_progress)
            optimizer.step()
            if os.environ.pop("MDLARC_REPORT_FIRST_UPDATE", None) == "1" and distributed.rank() == 0:
                print("MDLARC_STARTUP_EVENT first_optimizer_step", flush=True)
            if before is not None:
                _log_depth_diagnostics(model, before, grad_norm, step,
                                       log_location, log_handle, log_prefix)
            accum_index = 0
            optimizer_step += 1
            maybe_log_window()
    if accum_index > 0:
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scheduler is not None:
            if epoch is None or steps_per_epoch is None or last_batch_idx is None:
                scheduler.step()
            else:
                epoch_progress = float(epoch) + (last_batch_idx + 1) / float(
                    steps_per_epoch
                )
                scheduler.step(epoch_progress)
        optimizer.step()
        if os.environ.pop("MDLARC_REPORT_FIRST_UPDATE", None) == "1" and distributed.rank() == 0:
            print("MDLARC_STARTUP_EVENT first_optimizer_step", flush=True)
        optimizer_step += 1
        maybe_log_window()

    if train_log_mode == "10_steps":
        maybe_log_window(force=True)
    if report_epoch_metrics and epoch_count > 0:
        losses = loss_summary(epoch_output_sum, epoch_mtp_sum, epoch_total_sum, epoch_count, epoch_latent_sums)
        (
            color_correct,
            color_total,
            special_correct,
            special_total,
            exact_correct,
            exact_total,
        ) = distributed.summed(epoch_metric_counts).tolist()
        current_lr = (
            scheduler.get_last_lr()[0]
            if scheduler
            else optimizer.param_groups[0]["lr"]
        )
        if epoch is None:
            prefix = "epoch"
        else:
            prefix = f"epoch={epoch + 1}"
        log_msg = _format_log_prefix(
            log_prefix,
            (
                f"{prefix} lr={current_lr:.2e} {losses} "
                f"color_accuracy={_format_accuracy(color_correct, color_total)} "
                "special_token_accuracy="
                f"{_format_accuracy(special_correct, special_total)} "
                "exact_output_accuracy="
                f"{_format_accuracy(exact_correct, exact_total)}"
            ),
        )
        _emit_log(log_msg, log_location, log_handle)
    return step


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


def _is_qkv_projection_param(name: str, param: nn.Parameter) -> bool:
    if not name.endswith(".attention.qkv_proj.weight"):
        return False
    if param.ndim != 2:
        return False
    return param.size(0) % 3 == 0 and param.size(0) // 3 == param.size(1)


def _is_muon_candidate(
    module: nn.Module, name: str, param: nn.Parameter
) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if name.startswith(("lm_head.", "lm_head_mtp.")):
        return False
    if not name.endswith(".weight"):
        return False
    return param.ndim == 2


def _is_example_value_latent_matrix_param(
    module: nn.Module, name: str, param: nn.Parameter
) -> bool:
    if not isinstance(module, nn.Embedding):
        return False
    if param.ndim != 2:
        return False
    return name == "example_embedding.latent_embedding.weight" or (
        name.startswith("value_embeddings.")
        and name.endswith(".latent_embedding.weight")
    )


def _normuon_supported(device: torch.device) -> Tuple[bool, str]:
    if device.type != "cuda":
        return False, "NorMuon requires CUDA."
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(bf16_supported) and not bf16_supported():
        return False, "NorMuon requires CUDA bfloat16 support"
    return True, ""


def _collect_param_groups(
    model: nn.Module, *, include_muon: bool, separate_attention_output: bool = False,
    separate_loop: bool = False,
) -> Dict[str, list[nn.Parameter]]:
    groups: Dict[str, list[nn.Parameter]] = {
        "decay": [],
        "attention": [],
        "token_embed": [],
        "task_factor": [],
        "example_value_latent": [],
        "dihedral_embed": [],
        "no_decay": [],
        "loop": [],
        "muon_qkv": [],
        "muon_attention": [],
        "muon_attention_output": [],
        "muon_fc_in": [],
        "muon_fc_out": [],
        "muon_example_value_latent": [],
        "muon_other": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = model.get_submodule(module_name) if module_name else model
        is_attention = _is_attention_param(name)

        if separate_loop and name.startswith("loop_"):
            groups["loop"].append(param)
            continue

        if include_muon and _is_example_value_latent_matrix_param(
            module, name, param
        ):
            groups["muon_example_value_latent"].append(param)
            continue

        if include_muon and _is_muon_candidate(module, name, param):
            if _is_qkv_projection_param(name, param):
                groups["muon_qkv"].append(param)
            elif separate_attention_output and name.endswith(".attention.out_proj.weight"):
                groups["muon_attention_output"].append(param)
            elif is_attention:
                groups["muon_attention"].append(param)
            elif name.endswith(".ff.fc_in.weight"):
                groups["muon_fc_in"].append(param)
            elif name.endswith(".ff.fc_out.weight"):
                groups["muon_fc_out"].append(param)
            else:
                groups["muon_other"].append(param)
            continue

        if param.ndim <= 1:
            groups["no_decay"].append(param)
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
            elif name.startswith("task_factor_embedding."):
                groups["task_factor"].append(param)
            elif name.startswith("value_embeddings."):
                groups["example_value_latent"].append(param)
            elif name.startswith("example_embedding."):
                groups["example_value_latent"].append(param)
            elif name.startswith("dihedral_embedding."):
                groups["dihedral_embed"].append(param)
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
    dihedral_embedding_weight_decay: float = 0.0,
    *,
    include_muon: bool = True,
    separate_attention_output: bool = False,
    separate_loop: bool = False,
) -> Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]:
    """Split params for Muon matrix weights and AdamW-managed parameters."""
    groups = _collect_param_groups(
        model, include_muon=include_muon,
        separate_attention_output=separate_attention_output,
        separate_loop=separate_loop,
    )

    muon_groups = []
    for group_name, params, group_weight_decay, qkv_split in (
        # Keep packed QKV projections as a single [3 * d_model, d_model] matrix
        # so Muon orthogonalizes the concatenated projection instead of Q/K/V
        # independently.
        ("muon_qkv", groups["muon_qkv"], attention_weight_decay, False),
        ("muon_attention", groups["muon_attention"], attention_weight_decay, False),
        ("muon_attention_output", groups["muon_attention_output"], attention_weight_decay, False),
        ("muon_fc_in", groups["muon_fc_in"], weight_decay, False),
        ("muon_fc_out", groups["muon_fc_out"], weight_decay, False),
        (
            "muon_example_value_latent",
            groups["muon_example_value_latent"],
            task_embedding_weight_decay,
            False,
        ),
        ("muon_other", groups["muon_other"], weight_decay, False),
    ):
        if params:
            muon_groups.append(
                {
                    "params": params,
                    "weight_decay": group_weight_decay,
                    "group_name": group_name,
                    "qkv_split": qkv_split,
                }
            )

    adamw_groups = []
    if groups["decay"]:
        adamw_groups.append(
            {
                "params": groups["decay"],
                "weight_decay": weight_decay,
                "group_name": "adamw_decay",
            }
        )
    if groups["attention"]:
        adamw_groups.append(
            {
                "params": groups["attention"],
                "weight_decay": attention_weight_decay,
                "group_name": "adamw_attention",
            }
        )
    if groups["token_embed"]:
        adamw_groups.append(
            {
                "params": groups["token_embed"],
                "weight_decay": token_embedding_weight_decay,
                "group_name": "adamw_token_embed",
            }
        )
    if groups["task_factor"]:
        adamw_groups.append(
            {
                "params": groups["task_factor"],
                "weight_decay": task_embedding_weight_decay,
                "group_name": "adamw_task_factor",
            }
        )
    if groups["example_value_latent"]:
        adamw_groups.append(
            {
                "params": groups["example_value_latent"],
                "weight_decay": task_embedding_weight_decay,
                "group_name": "adamw_example_value_latent",
            }
        )
    if groups["dihedral_embed"]:
        adamw_groups.append(
            {
                "params": groups["dihedral_embed"],
                "weight_decay": dihedral_embedding_weight_decay,
                "group_name": "adamw_dihedral_embed",
            }
        )
    if groups["no_decay"]:
        adamw_groups.append(
            {
                "params": groups["no_decay"],
                "weight_decay": 0.0,
                "group_name": "adamw_no_decay",
            }
        )
    if groups["loop"]:
        adamw_groups.append(
            {
                "params": groups["loop"],
                "weight_decay": 0.0,
                "group_name": "adamw_loop",
            }
        )

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
) -> None:
    optimizer.load_state_dict(state_dict)
    _move_optimizer_state(optimizer, device)


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
    optimizer: torch.optim.Optimizer, checkpoint: Dict[str, Any], *,
    allow_lr_change: bool = False,
) -> bool:
    current = _param_groups_snapshot(optimizer.param_groups)
    # Saved groups include scheduler-added initial_lr and a possibly decayed
    # lr. Compare configured base LRs; an explicit resume can allow new LRs
    # while keeping all other optimizer settings strict. This also accepts
    # checkpoints saved before this normalization.
    def configured_groups(groups):
        result = []
        for original in groups:
            group = dict(original)
            group["lr"] = group.pop("initial_lr", group["lr"])
            if allow_lr_change:
                group.pop("lr")
            result.append(group)
        return result

    return not _optimizer_values_match(
        configured_groups(current), configured_groups(checkpoint["optimizer_hparams"])
    )


def _optimizer_switch_detected(
    optimizer: torch.optim.Optimizer,
    checkpoint: Dict[str, Any],
) -> bool:
    checkpoint_name = str(checkpoint["optimizer_name"]).lower()
    return checkpoint_name != _optimizer_identity(optimizer)


_NORMUON_GROUP_LR_MULTIPLIER_KEYS: Dict[str, Tuple[str, ...]] = {
    "muon_qkv": ("normuon_qkv_lr_multiplier",),
    "muon_attention": ("normuon_attention_lr_multiplier",),
    "muon_attention_output": ("normuon_attention_lr_multiplier",),
    "muon_fc_in": (
        "normuon_fc_in_lr_multiplier",
        "normuon_ff_in_lr_multiplier",
        "normuon_mlp_lr_multiplier",
    ),
    "muon_fc_out": (
        "normuon_fc_out_lr_multiplier",
        "normuon_ff_out_lr_multiplier",
        "normuon_mlp_lr_multiplier",
    ),
    "muon_example_value_latent": (
        "normuon_example_value_latent_lr_multiplier",
    ),
    "muon_other": ("normuon_other_lr_multiplier", "normuon_muon_lr_multiplier"),
}


_ADAMW_GROUP_LR_MULTIPLIER_KEYS: Dict[str, Tuple[str, ...]] = {
    "adamw_decay": ("adamw_decay_lr_multiplier",),
    "adamw_attention": ("adamw_attention_lr_multiplier",),
    "adamw_token_embed": (
        "adamw_token_embedding_lr_multiplier",
        "adamw_token_embed_lr_multiplier",
    ),
    "adamw_task_factor": ("adamw_task_factor_lr_multiplier",),
    "adamw_example_value_latent": ("adamw_example_value_latent_lr_multiplier",),
    "adamw_dihedral_embed": ("adamw_dihedral_embedding_lr_multiplier", "adamw_dihedral_embed_lr_multiplier"),
    "adamw_no_decay": ("adamw_no_decay_lr_multiplier",),
    "adamw_loop": ("adamw_loop_lr_multiplier",),
}


_NORMUON_GROUP_LR_FLOOR_KEYS: Dict[str, Tuple[str, ...]] = {
    "muon_qkv": ("normuon_qkv_lr_floor", "muon_qkv_lr_floor"),
    "muon_attention": ("normuon_attention_lr_floor", "muon_attention_lr_floor"),
    "muon_attention_output": ("normuon_attention_lr_floor", "muon_attention_lr_floor"),
    "muon_fc_in": (
        "normuon_fc_in_lr_floor",
        "normuon_ff_in_lr_floor",
        "normuon_mlp_lr_floor",
        "muon_fc_in_lr_floor",
    ),
    "muon_fc_out": (
        "normuon_fc_out_lr_floor",
        "normuon_ff_out_lr_floor",
        "normuon_mlp_lr_floor",
        "muon_fc_out_lr_floor",
    ),
    "muon_example_value_latent": (
        "normuon_example_value_latent_lr_floor",
        "muon_example_value_latent_lr_floor",
    ),
    "muon_other": (
        "normuon_other_lr_floor",
        "normuon_muon_lr_floor",
        "muon_other_lr_floor",
    ),
}


_ADAMW_GROUP_LR_FLOOR_KEYS: Dict[str, Tuple[str, ...]] = {
    "adamw_decay": ("adamw_decay_lr_floor",),
    "adamw_attention": ("adamw_attention_lr_floor",),
    "adamw_token_embed": (
        "adamw_token_embedding_lr_floor",
        "adamw_token_embed_lr_floor",
    ),
    "adamw_task_factor": ("adamw_task_factor_lr_floor",),
    "adamw_example_value_latent": ("adamw_example_value_latent_lr_floor",),
    "adamw_dihedral_embed": ("adamw_dihedral_embedding_lr_floor", "adamw_dihedral_embed_lr_floor"),
    "adamw_no_decay": ("adamw_no_decay_lr_floor",),
    "adamw_loop": ("adamw_no_decay_lr_floor",),
}


def _optional_float_arg(
    args: argparse.Namespace, *names: str
) -> Optional[float]:
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            return float(value)
    return None


def _clamp_lr_floor(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lr_floor_for_group(args: argparse.Namespace, group: Dict[str, Any]) -> float:
    group_name = str(group.get("group_name", ""))
    is_muon_group = bool(group.get("use_muon", False)) or group_name.startswith(
        "muon_"
    )

    if is_muon_group:
        floor = _optional_float_arg(
            args, *_NORMUON_GROUP_LR_FLOOR_KEYS.get(group_name, ())
        )
        if floor is None:
            floor = _optional_float_arg(args, "normuon_lr_floor", "muon_lr_floor")
    else:
        floor = _optional_float_arg(
            args, *_ADAMW_GROUP_LR_FLOOR_KEYS.get(group_name, ())
        )
        if floor is None:
            floor = _optional_float_arg(args, "adamw_lr_floor")

    if floor is None:
        floor = _optional_float_arg(args, "lr_floor")
    if floor is None:
        floor = 0.01
    return _clamp_lr_floor(floor)


def _group_lr_with_multiplier(
    args: argparse.Namespace,
    group_name: str,
    base_lr: float,
    multiplier_keys: Dict[str, Tuple[str, ...]],
) -> float:
    multiplier = _optional_float_arg(args, *multiplier_keys.get(group_name, ()))
    if multiplier is None:
        multiplier = 1.0
    return base_lr * multiplier


def _normuon_group_lr(
    args: argparse.Namespace, group_name: str, base_lr: float
) -> float:
    lr = _group_lr_with_multiplier(
        args, group_name, base_lr, _NORMUON_GROUP_LR_MULTIPLIER_KEYS
    )
    if group_name in {"muon_attention_output", "muon_fc_out"}:
        lr *= _residual_output_lr_multiplier(args)
    return lr


def _residual_output_lr_multiplier(args: argparse.Namespace) -> float:
    value = float(getattr(args, "normuon_residual_output_lr_multiplier", 1.0))
    if not math.isfinite(value) or value < 0:
        raise ValueError("normuon_residual_output_lr_multiplier must be finite and nonnegative.")
    return value


def _loop_lr_multiplier(args: argparse.Namespace) -> float:
    value = float(getattr(args, "adamw_loop_lr_multiplier", None) or 1.0)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("adamw_loop_lr_multiplier must be finite and positive.")
    return value


def _adamw_group_lr(
    args: argparse.Namespace, group_name: str, base_lr: float
) -> float:
    return _group_lr_with_multiplier(
        args, group_name, base_lr, _ADAMW_GROUP_LR_MULTIPLIER_KEYS
    )


def _with_adamw_group_lrs(
    args: argparse.Namespace,
    groups: Sequence[Dict[str, Any]],
    base_lr: float,
) -> Sequence[Dict[str, Any]]:
    param_groups = []
    for group in groups:
        adam_group = dict(group)
        group_name = str(adam_group.get("group_name", "adamw"))
        adam_group["group_name"] = group_name
        adam_group["lr"] = _adamw_group_lr(args, group_name, base_lr)
        param_groups.append(adam_group)
    return param_groups


def _build_optimizer(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    attention_weight_decay: float,
    token_embedding_weight_decay: float,
    task_embedding_weight_decay: float,
) -> torch.optim.Optimizer:
    if getattr(args, "finetune_method", None) == "lora":
        # Adapters only, with the 600M LoRA sweeps' optimizer (src/checkpoint_adaptation.py).
        adapters = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW([{"params": adapters, "group_name": "adamw_decay"}], lr=float(args.lora_lr),
                                 weight_decay=float(args.lora_weight_decay), betas=(0.9, 0.999))
    optimizer_name = str(getattr(args, "optimizer", "normuon")).lower()
    adamw_lr = getattr(args, "adamw_lr", None)
    adamw_lr = 3e-4 if adamw_lr is None else float(adamw_lr)

    if optimizer_name not in {"normuon"}:
        raise ValueError("optimizer must be 'normuon'.")

    muon_groups, adamw_groups = _build_param_groups(
        model,
        args.weight_decay,
        attention_weight_decay,
        token_embedding_weight_decay,
        task_embedding_weight_decay,
        dihedral_embedding_weight_decay=getattr(args, "dihedral_embedding_weight_decay", 0.0),
        include_muon=True,
        # Keep historical group ordering/topology unless the ablation is active.
        separate_attention_output=_residual_output_lr_multiplier(args) != 1.0,
        # Loop-extension scalars get their own AdamW group only when re-scaled,
        # keeping the historical optimizer topology otherwise.
        separate_loop=_loop_lr_multiplier(args) != 1.0,
    )

    supported, reason = _normuon_supported(device)
    if not supported:
        raise RuntimeError(reason)

    if not muon_groups:
        raise RuntimeError("NorMuon requested but no eligible linear weights were found.")

    normuon_lr = getattr(args, "normuon_lr", None)
    if normuon_lr is None:
        normuon_lr = 0.02
    else:
        normuon_lr = float(normuon_lr)
    normuon_momentum = float(getattr(args, "normuon_momentum", 0.95))
    normuon_momentum_min = float(
        getattr(args, "normuon_momentum_min", min(0.85, normuon_momentum))
    )
    normuon_warmup_steps = max(0, int(getattr(args, "normuon_warmup_steps", 300)))
    normuon_cooldown_steps = max(
        0, int(getattr(args, "normuon_cooldown_steps", 50))
    )
    normuon_beta2 = float(getattr(args, "normuon_beta2", 0.95))
    normuon_bucket_size = max(0, int(getattr(args, "normuon_bucket_size", 0) or 0))

    normuon_param_groups = []
    for group in muon_groups:
        normuon_group = dict(group)
        group_name = str(normuon_group.get("group_name", "muon_other"))
        normuon_group["use_muon"] = True
        normuon_group["lr"] = _normuon_group_lr(args, group_name, normuon_lr)
        normuon_group["momentum"] = normuon_momentum
        normuon_group["momentum_min"] = normuon_momentum_min
        normuon_group["muon_warmup_steps"] = normuon_warmup_steps
        normuon_group["muon_cooldown_steps"] = normuon_cooldown_steps
        normuon_group["beta2"] = normuon_beta2
        normuon_param_groups.append(normuon_group)
    for group in adamw_groups:
        adam_group = dict(group)
        adam_group["use_muon"] = False
        group_name = str(adam_group.get("group_name", "adamw"))
        adam_group["group_name"] = group_name
        adam_group["lr"] = _adamw_group_lr(args, group_name, adamw_lr)
        normuon_param_groups.append(adam_group)
    return SingleDeviceNorMuonWithAuxAdam(
        normuon_param_groups,
        normuon_bucket_size=normuon_bucket_size,
    )


# =============================================================================
# Checkpointing
# =============================================================================

def _unwrap_state_model(model: nn.Module) -> nn.Module:
    return distributed.unwrap(model)


def maybe_save_model(
    model: TinyTransformer,
    dataset: ARCExampleDataset,
    data_path: Path,
    save_path: Optional[Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    global_step: Optional[int] = None,
    epoch: Optional[int] = None,
    rng_state: Optional[Dict[str, Any]] = None,
    rng_state_by_rank: Optional[List[Dict[str, Any]]] = None,
    scheduler: Optional[Any] = None,
    checkpoint_keep_last: Optional[int] = None,
    checkpoint_base_path: Optional[Path] = None,
    rolling: bool = False,
) -> None:
    if save_path is None:
        return
    if distributed.rank() != 0:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state_model = _unwrap_state_model(model)
    if parameter_efficient.has_lora(state_model):
        # Merged weights only: an ordinary inference checkpoint, not an adapter-training resume point.
        optimizer = scheduler = None
    checkpoint = {
        "sequence_format": (SUPERVISED_GRID_CHAIN_SEQUENCE_FORMAT if state_model.config.predict_intermediate_grids else
                            GRID_CHAIN_SEQUENCE_FORMAT if getattr(dataset, "has_intermediate_grids", False)
                            else sequence_format_for_attention(state_model.config.attention_pattern)),
        "augmentation_sampling_mode": getattr(dataset, "augmentation_sampling_mode", "full_task_prefix"),
        "model_state": parameter_efficient.merged_model_state(state_model),
        "config": asdict(state_model.config),
        "task_ids": list(dataset.task_ids),
        "embedding_task_ids": list(dataset.embedding_task_ids),
        "data_path": str(data_path),
    }
    if hasattr(dataset, "variation_index"):
        checkpoint["task_variations"] = dataset.checkpoint_metadata(
            enable_aug=bool(getattr(dataset, "variation_augmentation_enabled", False)),
        )
    else:
        source_checkpoint = getattr(state_model, "_loaded_checkpoint", None)
        if isinstance(source_checkpoint, dict) and "task_variations" in source_checkpoint:
            checkpoint["task_variations"] = source_checkpoint["task_variations"]
    if hasattr(dataset, "training_data_metadata"):
        checkpoint["task_data"] = dataset.training_data_metadata
    else:
        source_checkpoint = getattr(state_model, "_loaded_checkpoint", None)
        if isinstance(source_checkpoint, dict) and "task_data" in source_checkpoint:
            checkpoint["task_data"] = source_checkpoint["task_data"]
    for attr_name, key in (
        ("rearc_source_path", "rearc_data_path"),
        ("nvarc_training_dir", "nvarc_training_path"),
        ("nvarc_full_dir", "nvarc_full_path"),
    ):
        path = getattr(dataset, attr_name, None)
        if path is not None:
            checkpoint[key] = str(path)
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
    if rng_state_by_rank is not None:
        checkpoint["rng_state_by_rank"] = rng_state_by_rank
        checkpoint["distributed_world_size"] = len(rng_state_by_rank)
    if rolling:
        # Local resume point only: never queued, uploaded or counted toward retention.
        save_rolling_checkpoint(save_path, lambda temporary: torch.save(checkpoint, temporary))
    else:
        save_checkpoint_atomically(
            save_path, lambda temporary: torch.save(checkpoint, temporary),
            checkpoint_keep_last=checkpoint_keep_last, checkpoint_base_path=checkpoint_base_path,
        )
    print(f"Saved {'rolling ' if rolling else ''}checkpoint to {save_path}")


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


def _rolling_checkpoint_path(save_path: Path) -> Path:
    return save_path.with_name(f"{save_path.stem}.rolling{save_path.suffix}")


def _kept_checkpoint_epochs(every: int, total_epochs: int) -> Set[int]:
    """Epochs kept as standalone checkpoints; the final epoch is the main checkpoint."""
    if isinstance(every, bool) or not isinstance(every, int) or every < 0:
        raise ValueError("kept_checkpoint_every must be an integer >= 0.")
    return set(range(every, total_epochs, every)) if every else set()


def _save_checkpoint(
    model: TinyTransformer,
    dataset: ARCExampleDataset,
    data_path: Path,
    save_path: Optional[Path],
    *,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    scheduler: Any,
    device: torch.device,
    checkpoint_keep_last: Optional[int] = None,
    checkpoint_base_path: Optional[Path] = None,
    rolling: bool = False,
) -> None:
    if save_path is None:
        return
    rng_state = capture_rng_state(device)
    rank_states = distributed.gather_checkpoint_rng_state(rng_state)
    maybe_save_model(
        model,
        dataset,
        data_path,
        save_path,
        optimizer=optimizer,
        global_step=global_step,
        epoch=epoch,
        rng_state=rng_state,
        rng_state_by_rank=rank_states,
        scheduler=scheduler,
        checkpoint_keep_last=checkpoint_keep_last,
        checkpoint_base_path=checkpoint_base_path,
        rolling=rolling,
    )
    if distributed.active():
        # No worker may resume/read the checkpoint before rank zero publishes it.
        distributed.checkpoint_barrier()


# =============================================================================
# Main Training Loop
# =============================================================================

def _base_lr_schedule(args, optimizer):
    """Fresh-run LR policy shared with bounded full-epoch benchmarks."""
    total_epochs = max(0.0, float(args.epochs))
    warmup_pct = float(getattr(args, "warmup_pct", 0.02))
    warmup_pct = max(0.0, min(1.0, warmup_pct))
    warmup_epochs = total_epochs * warmup_pct
    warmup_epochs = max(0.0, min(total_epochs, warmup_epochs))
    min_lr_factors = [
        _lr_floor_for_group(args, group) for group in optimizer.param_groups
    ]
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
        return [
            min_lr_factor + (1.0 - min_lr_factor) * linear
            for min_lr_factor in min_lr_factors
        ]

    return base_lr_lambda


def _train_single_model(
    args: argparse.Namespace,
    model: TinyTransformer,
    dataloader: torch.utils.data.DataLoader,
    dataset: ARCExampleDataset,
    device: torch.device,
    data_path: Path,
    *,
    save_path: Optional[Path],
    checkpoint_schedule: Optional[Set[int]],
    checkpoint: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_keep_last = getattr(args, "checkpoint_keep_last", None)
    validate_checkpoint_keep_last(checkpoint_keep_last)
    if checkpoint is None:
        checkpoint = getattr(model, "_loaded_checkpoint", None)
    if should_reset_training_state(args):
        checkpoint = None
    step = int(checkpoint["global_step"]) if checkpoint else 0
    start_epoch = int(checkpoint["epoch"]) if checkpoint else 0
    if checkpoint and start_epoch >= args.epochs:
        raise ValueError(
            f"Checkpoint has {start_epoch} epochs completed; configured epochs={args.epochs} "
            "must be greater than the completed epoch to resume training."
        )
    rolling_every = getattr(args, "rolling_checkpoint_every", 0)
    if isinstance(rolling_every, bool) or not isinstance(rolling_every, int) or rolling_every < 0:
        raise ValueError("rolling_checkpoint_every must be an integer >= 0.")
    rolling_path = _rolling_checkpoint_path(save_path) if save_path is not None and rolling_every else None

    log_file = getattr(args, "train_log_file", None)
    if log_file is not None and not isinstance(log_file, Path):
        log_file = Path(log_file)
    train_log_mode = str(getattr(args, "train_log_mode", "10_steps"))
    log_location = str(getattr(args, "log_location", "both"))
    log_handle = None
    if distributed.rank() == 0 and log_location in ("file", "both") and log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("a")

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

    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "drop_last"):
        batch_sampler.drop_last = not getattr(batch_sampler, "preserve_all_pairs", False)
    # The training loop finalizes drop_last after DataLoader construction.
    # Context color plans must follow these actual, potentially sparse visits
    # and the resulting unchanged geometry schedule, before workers start.
    planned_augmentor = getattr(dataloader, "augmentor", None)
    if hasattr(planned_augmentor, "prepare_training_schedule"):
        planned_augmentor.prepare_training_schedule(batch_sampler)

    if checkpoint and step > 0:
        print(f"Resuming training from global_step={step}.")
    if checkpoint and start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch + 1}/{args.epochs}.")

    gradient_accumulation_steps = max(
        1, int(getattr(args, "gradient_accumulation_steps", 1))
    )

    epoch_batch_sampler = getattr(dataloader, "batch_sampler", None)
    preserve_all_pairs = getattr(epoch_batch_sampler, "preserve_all_pairs", False)
    aligned_minibatches = (lambda n, _: n) if preserve_all_pairs else _full_accumulation_minibatches
    epoch_lengths: List[int]
    if epoch_batch_sampler is not None and hasattr(epoch_batch_sampler, "set_epoch"):
        previous_epoch = int(getattr(epoch_batch_sampler, "_epoch", 0))
        epoch_lengths = []
        for sample_epoch in range(max(0, int(args.epochs))):
            epoch_batch_sampler.set_epoch(sample_epoch)
            epoch_lengths.append(
                aligned_minibatches(
                    len(dataloader),
                    gradient_accumulation_steps,
                )
            )
        epoch_batch_sampler.set_epoch(previous_epoch)
    else:
        aligned_epoch_length = aligned_minibatches(
            len(dataloader),
            gradient_accumulation_steps,
        )
        epoch_lengths = [aligned_epoch_length] * max(0, int(args.epochs))

    default_steps_per_epoch = aligned_minibatches(
        len(dataloader),
        gradient_accumulation_steps,
    )
    if epoch_lengths:
        steps_per_epoch = epoch_lengths[min(start_epoch, len(epoch_lengths) - 1)]
    else:
        steps_per_epoch = default_steps_per_epoch

    total_muon_steps = sum(
        (math.ceil(epoch_steps / gradient_accumulation_steps) if preserve_all_pairs else epoch_steps // gradient_accumulation_steps)
        for epoch_steps in epoch_lengths
    )
    completed_muon_steps = sum(
        (math.ceil(epoch_steps / gradient_accumulation_steps) if preserve_all_pairs else epoch_steps // gradient_accumulation_steps)
        for epoch_steps in epoch_lengths[:start_epoch]
    )
    completed_muon_steps = min(completed_muon_steps, total_muon_steps)
    if isinstance(optimizer, SingleDeviceNorMuonWithAuxAdam):
        optimizer.configure_muon_schedule(
            total_steps=total_muon_steps,
            start_step=completed_muon_steps,
        )

    optimizer_switched = bool(
        checkpoint and _optimizer_switch_detected(optimizer, checkpoint)
    )
    if optimizer_switched:
        raise ValueError("Checkpoint optimizer does not match the configured optimizer.")
    configured_base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    resume_lr_changed = bool(checkpoint and not _optimizer_values_match(
        configured_base_lrs,
        [float(group.get("initial_lr", group["lr"]))
         for group in checkpoint["optimizer_hparams"]],
    ))
    if checkpoint and _optimizer_hparams_changed(optimizer, checkpoint, allow_lr_change=True):
        raise ValueError(
            "Checkpoint optimizer hyperparameters do not match the configured optimizer."
        )

    if checkpoint:
        _load_optimizer_state(optimizer, checkpoint["optimizer_state"], device)
        print("Restored optimizer state from checkpoint.")
        if resume_lr_changed:
            for group, base_lr in zip(optimizer.param_groups, configured_base_lrs):
                group["lr"] = base_lr
                group["initial_lr"] = base_lr
            print(f"Applying configured resume base LRs: {configured_base_lrs}; optimizer buffers preserved.")

    total_epochs = max(0.0, float(args.epochs))
    steps_per_epoch = max(1, steps_per_epoch)
    base_lr_lambda = _base_lr_schedule(args, optimizer)

    resume_warmup_pct = float(getattr(args, "resume_warmup_pct", 0.0) or 0.0)
    resume_warmup_pct = max(0.0, min(1.0, resume_warmup_pct))
    resume_warmup = bool(checkpoint and start_epoch > 0 and resume_warmup_pct > 0.0)

    lr_lambda = base_lr_lambda
    resume_start_epoch = float(start_epoch)
    if resume_warmup and resume_start_epoch > 0:
        remaining_epochs = max(0.0, total_epochs - resume_start_epoch)
        if remaining_epochs > 0:
            resume_warmup_epochs = remaining_epochs * resume_warmup_pct
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
                    return _scale_lr_factors(resume_target, progress)
                return base_lr_lambda(current_epoch)

            print(f"Applying resume LR warmup for {resume_warmup_epochs:.3f} epochs.")

    scheduler = _FractionalLambdaLR(optimizer, lr_lambda)

    scheduler_restored = False
    if checkpoint and not resume_warmup:
        scheduler_state = checkpoint["scheduler_state"]
        last_epoch = (
            scheduler_state.get("last_epoch")
            if isinstance(scheduler_state, dict)
            else None
        )
        if isinstance(last_epoch, (int, float)) and abs(
            float(last_epoch) - float(start_epoch)
        ) > 1.0:
            raise ValueError(
                "Checkpoint scheduler epoch does not match checkpoint epoch: "
                f"scheduler last_epoch={last_epoch}, checkpoint epoch={start_epoch}."
            )
        scheduler.load_state_dict(scheduler_state)
        if resume_lr_changed:
            scheduler.base_lrs = configured_base_lrs
            scheduler.step(scheduler.last_epoch)
        scheduler_restored = True
        print("Restored scheduler state from checkpoint.")
    if not scheduler_restored and start_epoch > 0:
        if resume_warmup:
            resume_base_factor = base_lr_lambda(resume_start_epoch)
            resume_base_factors = _expand_lr_factors(
                resume_base_factor, len(optimizer.param_groups)
            )
            for base_lr, group, factor in zip(
                scheduler.base_lrs, optimizer.param_groups, resume_base_factors
            ):
                group["lr"] = base_lr * factor
            scheduler._last_lr = [
                base_lr * factor
                for base_lr, factor in zip(
                    scheduler.base_lrs, resume_base_factors
                )
            ]
        else:
            scheduler.step(float(start_epoch))

    if hasattr(torch, "compile") and device.type == "cuda":
        _enable_compile_scalar_capture()
        print("Compiling model for training speedup...")
        training_model = torch.compile(model)
    else:
        training_model = model

    if distributed.active():
        training_model = DistributedDataParallel(
            training_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
        if not checkpoint:
            # Model initialization stays identical; training dropout differs by rank.
            from common import set_seed
            set_seed(int(args.seed) + distributed.rank())

    augmentor = getattr(dataloader, "augmentor", None)
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            if augmentor is not None:
                augmentor.set_epoch(epoch)
            if epoch_batch_sampler is not None and hasattr(epoch_batch_sampler, "set_epoch"):
                epoch_batch_sampler.set_epoch(epoch)
            epoch_steps = (
                epoch_lengths[epoch]
                if epoch < len(epoch_lengths)
                else default_steps_per_epoch
            )

            step = train_one_epoch(
                model=training_model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                grad_clip=args.grad_clip,
                gradient_accumulation_steps=gradient_accumulation_steps,
                start_step=step,
                scheduler=scheduler,
                epoch=epoch,
                steps_per_epoch=epoch_steps,
                total_epochs=args.epochs,
                mtp_decay=getattr(args, "mtp_decay", (0.8, 0.9, 1.0)),
                depth_diagnostics_every=getattr(args, "depth_diagnostics_every", 0),
                train_log_mode=train_log_mode,
                log_location=log_location,
                log_handle=log_handle,
            )

            if checkpoint_schedule and save_path is not None:
                epoch_num = epoch + 1
                if epoch_num in checkpoint_schedule and not (
                    epoch_num == args.epochs and checkpoint_keep_last == 1
                ):
                    epoch_save_path = _checkpoint_path_for_epoch(
                        save_path, epoch_num, args.epochs
                    )
                    _save_checkpoint(
                        model,
                        dataset,
                        data_path,
                        epoch_save_path,
                        optimizer=optimizer,
                        global_step=step,
                        epoch=epoch_num,
                        scheduler=scheduler,
                        device=device,
                        checkpoint_keep_last=checkpoint_keep_last,
                        checkpoint_base_path=save_path,
                    )
            # Overwritten every N epochs so a crashed or stopped run can resume locally.
            if rolling_path is not None and (epoch + 1) % rolling_every == 0 and epoch + 1 < args.epochs:
                _save_checkpoint(
                    model,
                    dataset,
                    data_path,
                    rolling_path,
                    optimizer=optimizer,
                    global_step=step,
                    epoch=epoch + 1,
                    scheduler=scheduler,
                    device=device,
                    rolling=True,
                )

        _save_checkpoint(
            model,
            dataset,
            data_path,
            save_path,
            optimizer=optimizer,
            global_step=step,
            epoch=args.epochs,
            scheduler=scheduler,
            device=device,
            checkpoint_keep_last=checkpoint_keep_last,
            checkpoint_base_path=save_path,
        )
        if rolling_path is not None and distributed.rank() == 0:
            # The final checkpoint supersedes the resume point.
            rolling_path.unlink(missing_ok=True)
    finally:
        if log_handle is not None:
            log_handle.close()


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
    save_path = getattr(args, "save_path", None)
    if save_path is not None and not isinstance(save_path, Path):
        save_path = Path(save_path)
    checkpoint_schedule = _normalize_checkpoint_epochs(
        getattr(args, "checkpoint_epochs", None), args.epochs
    )
    kept = _kept_checkpoint_epochs(getattr(args, "kept_checkpoint_every", 0), args.epochs)
    if kept:
        checkpoint_schedule = (checkpoint_schedule or set()) | kept
    _train_single_model(
        args,
        model,
        dataloader,
        dataset,
        device,
        data_path,
        save_path=save_path,
        checkpoint_schedule=checkpoint_schedule,
        checkpoint=checkpoint,
    )
