"""Cross-entropy without the saturated-target or distillation cancellation bugs.

The hard-label path saves its analytic gradient, like ordinary CE saves log
probabilities. This keeps first-order training to a rowwise forward plus a
pointwise backward under torch.compile, without differentiating a logsumexp /
softplus expression. Low-precision logits accumulate in FP32; FP64 is retained
for reference checks. No attention kernels or model parameters are changed.

Reference: https://jonbarron.info/softmaxfix/ (Barron et al., 2026).
"""

import math

import torch
from torch.autograd.function import once_differentiable


_EXPM1_COEFFICIENTS = tuple(1 / math.factorial(n) for n in range(2, 21))


def _accumulate(logits):
    return logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits


def _reduce(losses, reduction, valid=None):
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        if valid is None:
            return losses.mean()
        count = valid.sum()
        # Match PyTorch's NaN value but zero gradient for an all-ignored mean.
        return torch.where(count > 0, losses.sum() / count.clamp(min=1), float("nan"))
    raise ValueError("reduction must be 'none', 'sum', or 'mean'.")


class _HardTargetCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        z = _accumulate(logits)
        valid = targets != ignore_index
        index = targets.masked_fill(~valid, 0).unsqueeze(-1)
        shifted = z - z.amax(dim=-1, keepdim=True)
        is_target = torch.arange(z.shape[-1], device=z.device) == index
        exponentials = shifted.exp()
        others = exponentials.masked_fill(is_target, 0)
        other_sum = others.sum(dim=-1, keepdim=True)
        target_shift = shifted.gather(-1, index)
        denominator = other_sum + exponentials.gather(-1, index)
        # If the target is a maximum, log(1 + tiny) must use log1p(tiny).
        # Otherwise the subtraction below is well conditioned, even when the
        # target exponential underflows. No exp(-target_shift) can overflow.
        losses = torch.where(
            target_shift == 0,
            other_sum.log1p(),
            denominator.log() - target_shift,
        ).squeeze(-1).masked_fill(~valid, 0)
        if ctx.needs_input_grad[0]:
            # Never form p_target - 1. The other classes supply its negative
            # mass even when adding that mass to 1 rounded it away.
            gradient = (others / denominator).scatter(-1, index, -other_sum / denominator)
            gradient = gradient.masked_fill(~valid.unsqueeze(-1), 0)
            ctx.save_for_backward(gradient)
            ctx.input_dtype = logits.dtype
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (gradient,) = ctx.saved_tensors
        return (gradient * grad_output.unsqueeze(-1)).to(ctx.input_dtype), None, None


def hard_target_cross_entropy(logits, targets, *, ignore_index=-100, reduction="none"):
    """CE for logits [..., classes] and integer targets [...].

    Supports first-order training, arbitrary leading dimensions, and the usual
    none/sum/mean reductions. As in PyTorch CE, mean over no valid labels is NaN;
    model callers use sum/none and their existing clamped supervision counts.
    """
    if logits.ndim < 1 or logits.shape[-1] < 1 or targets.shape != logits.shape[:-1]:
        raise ValueError("Expected logits [..., classes] and matching targets [...].")
    if targets.dtype != torch.long:
        raise TypeError("Hard targets must have dtype torch.long.")
    losses = _HardTargetCE.apply(logits, targets, ignore_index)
    return _reduce(losses, reduction, targets != ignore_index if reduction == "mean" else None)


def _log_softmax(logits):
    """Keep the winning log probability accurate even when its exp rounds to 1."""
    index = logits.argmax(dim=-1, keepdim=True)
    shifted = logits - logits.gather(-1, index)
    is_max = torch.arange(logits.shape[-1], device=logits.device) == index
    other_sum = shifted.exp().masked_fill(is_max, 0).sum(dim=-1, keepdim=True)
    return shifted - other_sum.log1p()


def _expm1_bounded(x):
    """Accurate expm1 on [-1, 1], including under native compilation.

    Inductor's CPU vector lowering can replace expm1(x) by exp(x)-1, undoing
    the cancellation fix. CUDA lowers to libdevice.expm1; retain that fast path.
    Horner evaluation avoids the CPU lowering entirely.
    Degree 12 (FP32) / 20 (FP64) truncation is below the respective roundoff;
    adding the correction to x also avoids adding tiny numbers to 1.
    """
    if x.device.type == "cuda":
        return x.expm1()
    coefficients = _EXPM1_COEFFICIENTS if x.dtype == torch.float64 else _EXPM1_COEFFICIENTS[:11]
    correction = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        correction = coefficient + x * correction
    return x + x.square() * correction


class _SoftTargetCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_logits, teacher_logits):
        z, teacher = _accumulate(student_logits), _accumulate(teacher_logits)
        dtype = torch.promote_types(z.dtype, teacher.dtype)
        z, teacher = z.to(dtype), teacher.to(dtype)
        log_q, log_p = _log_softmax(z), _log_softmax(teacher)
        q, p = log_q.exp(), log_p.exp()
        loss = -(p * log_q).sum(dim=-1)
        if ctx.needs_input_grad[0]:
            delta = z - teacher
            # Remove a common logit offset exactly before the weighted center.
            delta = delta - delta[..., :1]
            centered = delta - (p * delta).sum(dim=-1, keepdim=True)
            # The logit-difference expression is essential near equality. Far
            # apart, log_q - log_p avoids expm1 overflow before the per-entry
            # ordinary-difference fallback. Keep both torch.where arms finite.
            near = centered.abs().amax(dim=-1, keepdim=True) < 1
            small = torch.where(near, centered, 0)
            log_ratio_near = small - (p * _expm1_bounded(small)).sum(dim=-1, keepdim=True).log1p()
            log_ratio = torch.where(near, log_ratio_near, log_q - log_p)
            difference = torch.where(
                log_ratio.abs() < 1,
                p * _expm1_bounded(log_ratio.clamp(min=-1, max=1)),
                q - p,
            )
            gradient = difference - q * difference.sum(dim=-1, keepdim=True)
            ctx.save_for_backward(gradient)
            ctx.input_dtype = student_logits.dtype
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (gradient,) = ctx.saved_tensors
        return (gradient * grad_output.unsqueeze(-1)).to(ctx.input_dtype), None


def soft_target_cross_entropy(student_logits, teacher_logits, *, reduction="none"):
    """Distillation CE accepting detached teacher logits, not rounded targets.

    Equal logits give exactly zero student gradient (the CE value remains the
    teacher entropy). Temperature scaling, if desired, is applied by the caller
    to both logits and the resulting loss. This API does no work in hard-label
    model training and supports first-order student gradients only.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 1 or student_logits.shape[-1] < 1:
        raise ValueError("Student and teacher logits must have the same [..., classes] shape.")
    if student_logits.device != teacher_logits.device:
        raise ValueError("Student and teacher logits must have the same device.")
    return _reduce(_SoftTargetCE.apply(student_logits, teacher_logits.detach()), reduction)
