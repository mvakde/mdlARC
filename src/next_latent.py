"""Recursive NextLat on causal target outputs (arXiv:2511.05963, v4).

The residual dynamics MLP follows the authors' reference implementation.
Prefix states and transitions across packed examples never become targets.
"""
import torch
from torch import nn
from torch.nn import functional as F

from common import END_PAIR_TOKEN_ID


class NextLatentDynamics(nn.Module):
    def __init__(self, width, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 2 * width
        self.input_norm = nn.LayerNorm(2 * width, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * width, hidden_dim, bias=False), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False), nn.GELU(),
            nn.Linear(hidden_dim, width, bias=False),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, current_states, next_token_embeds):
        inputs = self.input_norm(torch.cat((next_token_embeds, current_states), dim=-1))
        return current_states + self.mlp(inputs)


@torch.compiler.disable
def _transition_indices(input_ids, valid_output, cu_seqlens):
    """Select only output transitions before the compiled, expensive MLP.

    nonzero has a data-dependent shape, like the existing eager attention-layout
    preparation. Keep this small selection outside Dynamo rather than running
    the dynamics network on thousands of unsupervised prefix tokens.
    """
    if input_ids.dim() == 2:
        valid = F.pad(valid_output, (0, 1)).reshape(-1)
        positions = torch.arange(input_ids.numel(), device=input_ids.device)
        same_sequence = positions.remainder(input_ids.size(1)) < input_ids.size(1) - 1
    else:
        valid = valid_output
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        ends = torch.repeat_interleave(cu_seqlens[1:].long(), lengths)
        same_sequence = torch.arange(input_ids.numel(), device=input_ids.device) + 1 < ends
    next_tokens = input_ids.reshape(-1).roll(-1)
    # END has no following output distribution and is not a latent target.
    sources = (valid & same_sequence & (next_tokens != END_PAIR_TOKEN_ID)).nonzero().flatten()
    return sources, valid[sources + 1]


def next_latent_losses(dynamics, hidden_states, input_ids, valid_output,
                       token_embedding, lm_head, *, cu_seqlens=None, source_indices=None,
                       horizon=1):
    if horizon > 1:
        return _multi_step_losses(dynamics, hidden_states, input_ids, valid_output,
                                  token_embedding, lm_head, cu_seqlens=cu_seqlens,
                                  source_indices=source_indices, horizon=horizon)
    if source_indices is None:
        sources, kl_valid = _transition_indices(input_ids, valid_output, cu_seqlens)
        regression_valid = torch.ones_like(kl_valid)
    else:
        # Native packed attention already selected the output-token indices.
        # Reuse them without a second nonzero / GPU synchronization. The final
        # END transition is cheap to evaluate and receives exactly zero weight.
        sources = source_indices
        following_tokens = input_ids[sources + 1]
        regression_valid = valid_output[sources] & (following_tokens != END_PAIR_TOKEN_ID)
        kl_valid = regression_valid & valid_output[sources + 1]
    states = hidden_states.reshape(-1, hidden_states.size(-1))
    following = sources + 1
    next_embeds = token_embedding(input_ids.reshape(-1)[following])
    predicted = dynamics(states[sources], next_embeds)
    target = states[following].detach()
    # Accumulate both losses in float32 under BF16 autocast.
    regression = F.smooth_l1_loss(predicted.float(), target.float(), reduction="none").mean(-1)
    regression = (regression * regression_valid).sum() / regression_valid.sum().clamp(min=1)
    # Detach the head for this branch only. Gradients still flow through the
    # prediction to the dynamics, current hidden state and token embedding.
    student_logits = F.linear(predicted, lm_head.weight.detach()).float()
    teacher_logits = F.linear(target, lm_head.weight.detach()).float().detach()
    pointwise = F.kl_div(F.log_softmax(student_logits, dim=-1),
                         F.log_softmax(teacher_logits, dim=-1),
                         reduction="none", log_target=True).sum(-1)
    kl = (pointwise * kl_valid).sum() / kl_valid.sum().clamp(min=1)
    return regression, kl, regression_valid.sum()


def _multi_step_losses(dynamics, hidden_states, input_ids, valid_output,
                       token_embedding, lm_head, *, cu_seqlens, source_indices, horizon):
    """Unroll one shared dynamics model; supervise every predicted state.

    Each step consumes a teacher-forced token and the previous *predicted*
    state, retaining gradients through the rollout. Average the masked mean
    losses over the configured horizon, following paper equations (3)-(4).
    A rollout stops contributing at END, ignored labels, or example boundaries.
    Keep the selected source shape fixed so compiled CUDA avoids a data-dependent
    index selection / synchronization at each future step.
    """
    sources = source_indices
    if sources is None:
        sources, _ = _transition_indices(input_ids, valid_output, cu_seqlens)
    tokens = input_ids.reshape(-1)
    states = hidden_states.reshape(-1, hidden_states.size(-1))
    if input_ids.dim() == 2:
        valid = F.pad(valid_output, (0, 1)).reshape(-1)
        ends = (sources // input_ids.size(1) + 1) * input_ids.size(1)
    else:
        valid = valid_output
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        ends = torch.repeat_interleave(cu_seqlens[1:].long(), lengths,
                                      output_size=tokens.numel())[sources]
    predicted = states[sources]
    active = torch.ones_like(sources, dtype=torch.bool)
    regression = states.new_zeros((), dtype=torch.float32)
    kl = states.new_zeros((), dtype=torch.float32)
    count = sources.new_zeros(())
    for offset in range(1, horizon + 1):
        following = sources + offset
        safe = following.clamp(max=tokens.numel() - 1)
        active = (active & (following < ends) & valid[safe - 1]
                  & (tokens[safe] != END_PAIR_TOKEN_ID))
        predicted = dynamics(predicted, token_embedding(tokens[safe]))
        target = states[safe].detach()
        terms = F.smooth_l1_loss(predicted.float(), target.float(), reduction="none").mean(-1)
        regression = regression + (terms * active).sum() / active.sum().clamp(min=1)
        kl_valid = active & valid[safe]
        student = F.linear(predicted, lm_head.weight.detach()).float()
        teacher = F.linear(target, lm_head.weight.detach()).float().detach()
        terms = F.kl_div(F.log_softmax(student, dim=-1), F.log_softmax(teacher, dim=-1),
                         reduction="none", log_target=True).sum(-1)
        kl = kl + (terms * kl_valid).sum() / kl_valid.sum().clamp(min=1)
        count = count + active.sum()
    return regression / horizon, kl / horizon, count
