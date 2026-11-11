import dataclasses
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError:  # pragma: no cover - flash-attn optional at import time
    flash_attn_varlen_qkvpacked_func = None

try:
    from flash_attn import flash_attn_varlen_kvpacked_func
except ImportError:  # pragma: no cover - flash-attn optional at import time
    flash_attn_varlen_kvpacked_func = None

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:  # pragma: no cover - flash-attn optional at import time
    flash_attn_varlen_func = None

try:
    import flash_attn_interface as flash_attn3_interface
except ImportError:  # pragma: no cover - flash-attn-3 optional at import time
    flash_attn3_varlen_func = None
else:  # pragma: no cover - depends on optional flash-attn-3 package
    flash_attn3_varlen_func = getattr(
        flash_attn3_interface,
        "flash_attn_varlen_func",
        None,
    )

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
except ImportError:  # pragma: no cover - flex-attention optional at import time
    create_block_mask = None
    flex_attention = None
    BlockMask = None

from attention_runtime import load_fa4_varlen_func
from stable_losses import hard_target_cross_entropy
from common import (
    END_PAIR_TOKEN_ID,
    IGNORE_INDEX,
    MAX_SEQ_LEN,
    START_TOKEN_ID,
    VOCAB_SIZE,
    compute_positions_3d,
    normalize_attention_pattern,
)
from group_attention import padded_prefix_structure, role_positions
from pair_identity import simplex_table
from target_demo_attention import (
    cached_attention, packed_attention, packed_prefix_lm_layout_fields, padded_attention,
    split_cache_attention,
)
from next_latent import NextLatentDynamics, next_latent_losses
from research_options import loop_dense_pairs, loop_extension_shapes, validate_research_options


def _cache_query_positions(cache_position: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """Absolute cache slots, shared [S] or independent [B, S] per row."""
    if cache_position.dim() == 2:
        if cache_position.shape != (batch_size, seq_len):
            raise ValueError("cache_position must match [batch, seq_len].")
        return cache_position
    if cache_position.dim() != 1 or cache_position.numel() not in (1, seq_len):
        raise ValueError("cache_position must be [1], [seq_len], or [batch, seq_len].")
    positions = cache_position
    if cache_position.numel() == 1 and seq_len > 1:
        positions = cache_position + torch.arange(seq_len, device=cache_position.device)
    return positions.view(1, -1).expand(batch_size, -1)


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())
    return False


def _compile_safe_assert(condition: Any, message: str) -> None:
    if torch.is_tensor(condition) and condition.device.type == "cuda":
        # Device-side check. Reading the condition on the host (``torch._assert``
        # or ``.item()``) stalls the CPU until the GPU drains its queue, three
        # times per training step; the async form costs one tiny kernel and
        # surfaces a failure as a CUDA assertion instead of a ValueError.
        torch._assert_async(condition, message)
        return
    if _is_torch_compiling() and hasattr(torch, "_assert"):
        torch._assert(condition, message)
        return
    if torch.is_tensor(condition):
        if bool(condition.item()):
            return
        raise ValueError(message)
    if bool(condition):
        return
    raise ValueError(message)


def normalize_flash_attention_version(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("FA must be 2 or 3, not a boolean.")
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("FA must be 2, 3 or 4.") from exc
    if version not in (2, 3, 4):
        raise ValueError("FA must be 2, 3 or 4.")
    return version


@dataclass
class TinyTransformerConfig:
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    dropout: float = 0.1
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.1
    num_examples: int = 1280
    num_task_embedding_latents: int = 400
    task_embeddings: bool = True
    dihedral_embeddings: bool = False
    FA: int = 2
    # Legacy checkpoints without these fields remain next-token models.
    n_predict: int = 1
    mtp_weight: float = 0.3
    # Physical layers stay unique; only this [start, end) span is repeated.
    loop_iterations: int = 1
    loop_start_layer: Optional[int] = None
    loop_end_layer: Optional[int] = None
    loop_input_injection: bool = True
    loop_dense_mixing: bool = False
    loop_iteration_embedding: bool = False
    loop_gate: str = "none"
    loop_gate_init: float = 1.0
    loop_ffn: str = "all"
    loop_iteration_norms: bool = False
    loop_streams: int = 0
    loop_stream_gate: str = "dynamic"
    loop_exit_weight: float = 0.0
    loop_exit_iterations: Optional[Tuple[int, ...]] = None
    next_latent_prediction: bool = False
    next_latent_horizon: int = 1
    next_latent_hidden_dim: Optional[int] = None
    next_latent_regression_weight: float = 1.0
    next_latent_kl_weight: float = 1.0
    input_shortcuts: str = "learned"
    rope_spatial_base: float = 1000.0
    attention_pattern: str = "dense"
    split_target_demo_attention: bool = False
    pair_identity_vertices: int = 0
    pair_identity_strength: float = 0.25
    # Checkpoint identity, not a configurable architecture switch. Old unmarked
    # checkpoints used a 10000 role base unless they explicitly saved another.
    architecture_version: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.pair_identity_vertices, bool) or not isinstance(self.pair_identity_vertices, int) or self.pair_identity_vertices < 0 or self.pair_identity_vertices == 1:
            raise ValueError("pair_identity_vertices must be 0 or an integer >= 2")
        if not math.isfinite(self.pair_identity_strength) or self.pair_identity_strength <= 0:
            raise ValueError("pair_identity_strength must be positive and finite")
        if self.pair_identity_vertices and (self.attention_pattern != "pair_group" or self.d_model < self.pair_identity_vertices - 1):
            raise ValueError("Simplex requires pair_group and width >= vertices-1")
        self.FA = normalize_flash_attention_version(self.FA)
        self.attention_pattern = normalize_attention_pattern(self.attention_pattern)
        if not isinstance(self.split_target_demo_attention, bool):
            raise ValueError("split_target_demo_attention must be boolean.")
        if self.split_target_demo_attention and self.attention_pattern != "pair_group":
            raise ValueError("Split target/demo attention requires attention_pattern='pair_group'.")
        if not isinstance(self.task_embeddings, bool):
            raise ValueError("task_embeddings must be a boolean.")
        if not isinstance(self.dihedral_embeddings, bool):
            raise ValueError("dihedral_embeddings must be a boolean.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        validate_research_options(vars(self), n_layers=self.n_layers, require_complete=True)
        if self.loop_exit_iterations is not None:
            self.loop_exit_iterations = tuple(sorted(int(item) for item in self.loop_exit_iterations))
        if self.loop_iterations > 1 and (self.loop_start_layer is None or self.loop_end_layer is None):
            raise ValueError("Looping requires explicit loop_start_layer and loop_end_layer.")
        if self.input_shortcuts not in ("learned", "fixed"):
            raise ValueError("input_shortcuts must be learned or fixed.")
        if (isinstance(self.rope_spatial_base, bool)
                or not isinstance(self.rope_spatial_base, (int, float))
                or not math.isfinite(self.rope_spatial_base)
                or self.rope_spatial_base <= 1):
            raise ValueError("rope_spatial_base must be finite and > 1.")
        if (
            isinstance(self.n_predict, bool)
            or not isinstance(self.n_predict, int)
            or self.n_predict < 1
        ):
            raise ValueError("n_predict must be an integer >= 1.")
        if (
            isinstance(self.mtp_weight, bool)
            or not isinstance(self.mtp_weight, (int, float))
            or not math.isfinite(self.mtp_weight)
            or self.mtp_weight < 0
        ):
            raise ValueError("mtp_weight must be finite and >= 0.")
        if self.num_examples < 1:
            raise ValueError("num_examples must be >= 1.")
        if self.num_task_embedding_latents is None:
            self.num_task_embedding_latents = 400
        self.num_task_embedding_latents = int(self.num_task_embedding_latents)
        if self.num_task_embedding_latents < 1:
            raise ValueError("num_task_embedding_latents must be >= 1.")


# Task-indexed tables that exist only when ``task_embeddings`` is on.
# Checkpoints saved before they were dropped still carry them unused.
TASK_TABLE_MODULES = ("task_factor_embedding", "example_embedding", "value_embeddings")


def _has_value_embedding(layer_idx: int, n_layers: int) -> bool:
    return layer_idx % 2 == (n_layers - 1) % 2


class TaskFactorizedEmbedding(nn.Module):
    def __init__(self, num_latents: int, d_model: int) -> None:
        super().__init__()
        self.latent_embedding = nn.Embedding(num_latents, d_model)

    @property
    def weight(self) -> torch.Tensor:
        return self.latent_embedding.weight

    def project(self, task_factors: torch.Tensor) -> torch.Tensor:
        return torch.matmul(
            task_factors.to(dtype=self.latent_embedding.weight.dtype),
            self.latent_embedding.weight,
        )


@dataclass
class PackedPrefixLMLayout:
    prefix_token_indices: torch.Tensor
    prefix_cu_seqlens: torch.Tensor
    max_prefix_len: int
    split_target_demo_layout: Optional[dict] = None
    prefix_kv_indices: Optional[torch.Tensor] = None
    prefix_kv_cu_seqlens: Optional[torch.Tensor] = None
    max_prefix_kv_len: int = 0
    padded_token_indices: Optional[torch.Tensor] = None
    output_token_indices: Optional[torch.Tensor] = None
    output_cu_seqlens: Optional[torch.Tensor] = None
    output_kv_indices: Optional[torch.Tensor] = None
    output_kv_cu_seqlens: Optional[torch.Tensor] = None
    max_output_len: int = 0
    max_output_kv_len: int = 0
    output_uses_full_kv: bool = False


_LAYOUT_FIELDS = tuple(field.name for field in dataclasses.fields(PackedPrefixLMLayout))


def _build_padded_output_targets_and_mask(
    input_ids: torch.Tensor,
    sep_indices: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if input_ids.dim() != 2:
        raise ValueError("Padded supervision expects input_ids with shape [batch, seq_len].")
    device = input_ids.device
    batch_size, _seq_len = input_ids.shape
    if sep_indices.dim() != 1 or sep_indices.size(0) != batch_size:
        raise ValueError("sep_indices must have shape [batch_size].")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    else:
        attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    if targets is None:
        targets = input_ids

    shift_targets = targets[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_targets = shift_targets.masked_fill(~shift_mask, IGNORE_INDEX)

    valid_mask = shift_targets != IGNORE_INDEX
    shift_len = shift_targets.size(1)
    sep_positions = sep_indices.clamp(min=0, max=shift_len).unsqueeze(1)
    positions = torch.arange(shift_len, device=device).unsqueeze(0)
    is_output_phase = positions >= sep_positions
    valid_output = valid_mask & is_output_phase
    return shift_targets, valid_output


def _build_varlen_output_targets_and_mask(
    input_ids: torch.Tensor,
    sep_indices: torch.Tensor,
    *,
    targets: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if input_ids.dim() != 1:
        raise ValueError("Packed supervision expects input_ids with shape [total_tokens].")
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for packed supervision.")
    if sep_indices.dim() != 1:
        raise ValueError("sep_indices must have shape [batch_size].")

    device = input_ids.device
    total_tokens = int(input_ids.size(0))
    batch_size = int(sep_indices.size(0))
    seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.long)
    if targets is None:
        targets = torch.roll(input_ids, shifts=-1).clone()
        sequence_ends = cu_seqlens[1:].to(dtype=torch.long) - 1
        targets[sequence_ends] = IGNORE_INDEX

    valid_mask = targets != IGNORE_INDEX
    seq_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=torch.long),
        seq_lengths,
        output_size=total_tokens,
    )
    local_positions = torch.arange(total_tokens, device=device, dtype=torch.long)
    local_positions = local_positions - torch.repeat_interleave(
        cu_seqlens[:-1].to(dtype=torch.long),
        seq_lengths,
        output_size=total_tokens,
    )
    is_output_phase = local_positions >= sep_indices[seq_ids]
    valid_output = valid_mask & is_output_phase
    return targets, valid_output


def build_output_targets_and_mask(
    input_ids: torch.Tensor,
    sep_indices: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return next-token targets and the subset mask for output-region supervision."""
    if input_ids.dim() == 2:
        return _build_padded_output_targets_and_mask(
            input_ids,
            sep_indices,
            attention_mask=attention_mask,
            targets=targets,
        )
    if input_ids.dim() == 1:
        return _build_varlen_output_targets_and_mask(
            input_ids,
            sep_indices,
            targets=targets,
            cu_seqlens=cu_seqlens,
        )
    raise ValueError("input_ids must have shape [B, S] or [total_tokens].")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rms = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(rms + self.eps)
        return hidden_states * self.weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TinyTransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim**-0.5
        self._has_flash_attn_varlen = flash_attn_varlen_qkvpacked_func is not None
        self._has_flash_attn_varlen_kvpacked = (
            flash_attn_varlen_kvpacked_func is not None
        )
        self._has_flash_attn_varlen_func = flash_attn_varlen_func is not None
        self._has_flash_attn3_varlen_func = flash_attn3_varlen_func is not None
        self._has_flex_attention = flex_attention is not None
        self.flash_attention_version = config.FA

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value_embed_gate_channels = min(32, config.d_model)
        self.value_embed_gate = (
            nn.Linear(self.value_embed_gate_channels, self.n_heads, bias=False)
            if _has_value_embedding(layer_idx, config.n_layers)
            else None
        )
        # Attention dropout is intentionally separate from hidden dropout.
        self.dropout_p = float(config.attention_dropout)
        self.split_target_demo_attention = config.split_target_demo_attention
        # Construct the gate after shared model initialization, preserving the
        # exact baseline initial weights and RNG stream for a matched seed.
        self.target_demo_gate = None

        self.rope = RotaryEmbedding3D(
            self.head_dim, spatial_base=config.rope_spatial_base,
        )

    def _apply_value_embedding(
        self,
        values: torch.Tensor,
        hidden_states: torch.Tensor,
        value_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if value_embed is None or self.value_embed_gate is None:
            return values

        gate = 2.0 * torch.sigmoid(
            self.value_embed_gate(hidden_states[..., : self.value_embed_gate_channels])
        )
        gate = gate.to(dtype=values.dtype)
        value_embed = value_embed.to(dtype=values.dtype)
        if hidden_states.dim() == 2:
            value_embed = value_embed.reshape(hidden_states.size(0), self.n_heads, self.head_dim)
            return values + gate.unsqueeze(-1) * value_embed

        if value_embed.dim() == 2:
            value_embed = value_embed.unsqueeze(1)
        value_embed = value_embed.reshape(
            hidden_states.size(0),
            value_embed.size(1),
            self.n_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        return values + gate.permute(0, 2, 1).unsqueeze(-1) * value_embed

    def _build_sdpa_attn_bias(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor],
        sdpa_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if sdpa_mask is not None:
            return sdpa_mask

        batch_size = queries.size(0)
        query_len = queries.size(2)
        key_len = keys.size(2)
        attn_bias = None

        if causal_mask is not None:
            attn_bias = torch.zeros(
                (1, 1, query_len, key_len), device=queries.device, dtype=queries.dtype
            )
            if causal_mask.size(-2) == query_len and causal_mask.size(-1) == key_len:
                attn_bias = attn_bias.masked_fill(causal_mask, float("-inf"))
            else:
                q_positions = torch.arange(query_len, device=queries.device).view(-1, 1)
                kv_positions = torch.arange(key_len, device=queries.device).view(1, -1)
                generated_causal = q_positions < kv_positions
                attn_bias = attn_bias.masked_fill(
                    generated_causal[None, None, :, :], float("-inf")
                )

        if attention_mask is not None:
            if attention_mask.size(1) < key_len:
                raise ValueError(
                    "attention_mask length must cover all key/value positions."
                )
            if attn_bias is None:
                attn_bias = torch.zeros(
                    (batch_size, 1, query_len, key_len),
                    device=queries.device,
                    dtype=queries.dtype,
                )
            key_mask = ~attention_mask[:, None, None, :key_len]
            attn_bias = attn_bias.masked_fill(key_mask, float("-inf"))

        return attn_bias

    def _apply_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor],
        sdpa_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        # Fast path: no explicit attn mask so SDPA can dispatch flash kernels.
        use_builtin_causal = bool(is_causal or causal_mask is not None)
        if sdpa_mask is None and attention_mask is None:
            return F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=use_builtin_causal,
            )

        attn_bias = self._build_sdpa_attn_bias(
            queries=queries,
            keys=keys,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            sdpa_mask=sdpa_mask,
        )
        return F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_bias,
            dropout_p=dropout_p,
            is_causal=False,
        )

    def _apply_flex_decode_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        decode_block_mask: Optional[object] = None,
        query_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._has_flex_attention:
            raise RuntimeError("flex_attention is not available.")
        if attention_mask.dim() != 2:
            raise ValueError("Decode attention_mask must have shape [batch, key_len].")
        if attention_mask.size(0) != queries.size(0):
            raise ValueError("Decode attention_mask batch size must match query batch size.")
        if attention_mask.size(1) < keys.size(2):
            raise ValueError("Decode attention_mask must cover all key/value positions.")

        key_len = keys.size(2)
        mask = attention_mask[:, :key_len]

        if decode_block_mask is not None:
            return flex_attention(
                queries,
                keys,
                values,
                block_mask=decode_block_mask,
                scale=self.scale,
            )

        # Fallback path for environments where create_block_mask is unavailable.
        def decode_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del h
            is_valid = mask[b, kv_idx]
            if query_positions is not None:
                is_valid = is_valid & (kv_idx <= query_positions[b, q_idx])
            return torch.where(is_valid, score, score.new_full((), float("-inf")))

        return flex_attention(
            queries,
            keys,
            values,
            score_mod=decode_score_mod,
            scale=self.scale,
        )

    @property
    def _has_flash_attn4_varlen_func(self) -> bool:
        # FA4 (CuTe DSL) is imported only when a model selects FA=4.
        return self.flash_attention_version == 4 and load_fa4_varlen_func() is not None

    def _has_selected_varlen_flash_attention(self) -> bool:
        if self.flash_attention_version == 4:
            return self._has_flash_attn4_varlen_func
        if self.flash_attention_version == 3:
            return self._has_flash_attn3_varlen_func
        return (
            self._has_flash_attn_varlen_func
            or self._has_flash_attn_varlen
            or self._has_flash_attn_varlen_kvpacked
        )

    def _apply_varlen_flash_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        if queries.device.type != "cuda":
            raise RuntimeError("flash-attn varlen kernels require CUDA tensors.")
        if queries.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "flash-attn varlen expects fp16/bf16 qkv tensors. "
                "Run this path under CUDA autocast."
            )
        if cu_seqlens_q.device != queries.device or cu_seqlens_q.dtype != torch.int32:
            cu_seqlens_q = cu_seqlens_q.to(device=queries.device, dtype=torch.int32)
        if cu_seqlens_k.device != queries.device or cu_seqlens_k.dtype != torch.int32:
            cu_seqlens_k = cu_seqlens_k.to(device=queries.device, dtype=torch.int32)

        if self.flash_attention_version == 4:
            if not self._has_flash_attn4_varlen_func:
                raise RuntimeError(
                    "FA=4 requires flash_attn.cute.interface.flash_attn_varlen_func "
                    "(flash-attn-4 with nvidia-cutlass-dsl) or use FA=2."
                )
            if float(dropout_p) != 0.0:
                raise RuntimeError(
                    "FA=4 varlen attention does not expose attention dropout. "
                    "Set attention_dropout=0.0 or use FA=2 to preserve dropout behavior."
                )
            result = load_fa4_varlen_func()(
                queries.contiguous(),
                keys.contiguous(),
                values.contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=int(max_seqlen_q),
                max_seqlen_k=int(max_seqlen_k),
                softmax_scale=None,
                causal=is_causal,
            )
            # The CuTe interface returns (output, lse) even without return_lse.
            return result[0] if isinstance(result, tuple) else result

        if self.flash_attention_version == 3:
            if not self._has_flash_attn3_varlen_func:
                raise RuntimeError(
                    "FA=3 requires flash_attn_interface.flash_attn_varlen_func. "
                    "Install FlashAttention-3 from the hopper package or use FA=2."
                )
            if float(dropout_p) != 0.0:
                raise RuntimeError(
                    "FA=3 varlen attention does not expose attention dropout. "
                    "Set attention_dropout=0.0 or use FA=2 to preserve dropout behavior."
                )
            result = flash_attn3_varlen_func(
                queries.contiguous(),
                keys.contiguous(),
                values.contiguous(),
                cu_seqlens_q,
                cu_seqlens_k,
                int(max_seqlen_q),
                int(max_seqlen_k),
                softmax_scale=None,
                causal=is_causal,
            )
            # Current hopper builds return the output alone; older ones (out, lse).
            return result[0] if isinstance(result, tuple) else result

        if self._has_flash_attn_varlen_func:
            return flash_attn_varlen_func(
                queries.contiguous(),
                keys.contiguous(),
                values.contiguous(),
                cu_seqlens_q,
                cu_seqlens_k,
                int(max_seqlen_q),
                int(max_seqlen_k),
                dropout_p=dropout_p,
                softmax_scale=None,
                causal=is_causal,
            )

        same_layout = (
            int(max_seqlen_q) == int(max_seqlen_k)
            and cu_seqlens_q.shape == cu_seqlens_k.shape
            and torch.equal(cu_seqlens_q, cu_seqlens_k)
        )
        if same_layout and self._has_flash_attn_varlen:
            qkv = torch.stack((queries, keys, values), dim=1).contiguous()
            return flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens_q,
                int(max_seqlen_q),
                dropout_p=dropout_p,
                softmax_scale=None,
                causal=is_causal,
            )
        if self._has_flash_attn_varlen_kvpacked:
            kv = torch.stack((keys, values), dim=1).contiguous()
            return flash_attn_varlen_kvpacked_func(
                queries.contiguous(),
                kv,
                cu_seqlens_q,
                cu_seqlens_k,
                int(max_seqlen_q),
                int(max_seqlen_k),
                dropout_p=dropout_p,
                softmax_scale=None,
                causal=is_causal,
            )
        raise RuntimeError(
            "flash-attn varlen attention requires flash_attn_varlen_func, "
            "flash_attn_varlen_qkvpacked_func, or flash_attn_varlen_kvpacked_func."
        )

    def _apply_varlen_prefix_lm_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        prefix_lm_layout: PackedPrefixLMLayout,
        dropout_p: float,
        gate_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prefix_lm_layout.split_target_demo_layout is not None:
            return packed_attention(self, queries, keys, values,
                prefix_lm_layout.split_target_demo_layout, gate_logits, dropout_p)
        attn_output = torch.empty_like(queries)
        prefix_keys = prefix_lm_layout.prefix_kv_indices
        if prefix_keys is None:
            prefix_keys = prefix_lm_layout.prefix_token_indices
        prefix_kv_cu = prefix_lm_layout.prefix_kv_cu_seqlens
        if prefix_kv_cu is None:
            prefix_kv_cu = prefix_lm_layout.prefix_cu_seqlens
        prefix_attn = self._apply_varlen_flash_attention(
            queries=queries.index_select(0, prefix_lm_layout.prefix_token_indices),
            keys=keys.index_select(0, prefix_keys),
            values=values.index_select(0, prefix_keys),
            cu_seqlens_q=prefix_lm_layout.prefix_cu_seqlens,
            cu_seqlens_k=prefix_kv_cu,
            max_seqlen_q=prefix_lm_layout.max_prefix_len,
            max_seqlen_k=prefix_lm_layout.max_prefix_kv_len or prefix_lm_layout.max_prefix_len,
            dropout_p=dropout_p,
            is_causal=False,
        )
        attn_output.index_copy_(0, prefix_lm_layout.prefix_token_indices, prefix_attn)

        if prefix_lm_layout.output_token_indices is not None:
            # Prefix-LM is exactly a bidirectional pass on the prefix followed by
            # causal suffix queries over the full sequence K/V for each example.
            output_keys = keys
            output_values = values
            if not prefix_lm_layout.output_uses_full_kv:
                output_keys = keys.index_select(0, prefix_lm_layout.output_kv_indices)
                output_values = values.index_select(
                    0, prefix_lm_layout.output_kv_indices
                )
            output_attn = self._apply_varlen_flash_attention(
                queries=queries.index_select(0, prefix_lm_layout.output_token_indices),
                keys=output_keys,
                values=output_values,
                cu_seqlens_q=prefix_lm_layout.output_cu_seqlens,
                cu_seqlens_k=prefix_lm_layout.output_kv_cu_seqlens,
                max_seqlen_q=prefix_lm_layout.max_output_len,
                max_seqlen_k=prefix_lm_layout.max_output_kv_len,
                dropout_p=dropout_p,
                is_causal=True,
            )
            attn_output.index_copy_(
                0,
                prefix_lm_layout.output_token_indices,
                output_attn,
            )

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        value_embed: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        pos_xyz: Optional[torch.Tensor] = None,
        sdpa_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        prefix_lm_layout: Optional[PackedPrefixLMLayout] = None,
        rope_tables: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 2:
            if cu_seqlens is None:
                raise ValueError(
                    "cu_seqlens is required for packed varlen attention inputs."
                )
            if max_seqlen is None:
                raise ValueError(
                    "max_seqlen is required for packed varlen attention inputs."
                )
            if attention_mask is not None or causal_mask is not None or sdpa_mask is not None:
                raise ValueError(
                    "Packed varlen attention does not support explicit attention masks."
                )

            total_tokens, dim = hidden_states.shape
            qkv = self.qkv_proj(hidden_states)
            qkv = qkv.view(total_tokens, 3, self.n_heads, self.head_dim)
            queries, keys, values = qkv.unbind(1)
            qkv_dtype = queries.dtype

            if pos_xyz is not None or rope_tables is not None:
                if rope_tables is None:
                    if pos_xyz.dim() != 2 or pos_xyz.size(0) != total_tokens:
                        raise ValueError(
                            "Packed pos_xyz must have shape [total_tokens, 3]."
                        )
                    rope_tables = self.rope.tables(pos_xyz)
                cos, sin = rope_tables
                if cos.dim() != 2 or cos.size(0) != total_tokens:
                    raise ValueError(
                        "Packed rope tables must have shape [total_tokens, head_dim]."
                    )
                # Rotate in fp32 directly in the [T, H, D] layout; the shared
                # tables broadcast over heads, so no transposes are needed.
                q_f32, k_f32 = self.rope.apply_rotary(
                    queries.float(), keys.float(),
                    tables=(cos.unsqueeze(1), sin.unsqueeze(1)),
                )
                queries = q_f32.to(dtype=qkv_dtype)
                keys = k_f32.to(dtype=qkv_dtype)

            values = self._apply_value_embedding(
                values,
                hidden_states,
                value_embed,
            )

            if prefix_lm_layout is not None:
                attn_output = self._apply_varlen_prefix_lm_attention(
                    queries=queries,
                    keys=keys,
                    values=values,
                    prefix_lm_layout=prefix_lm_layout,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    gate_logits=self.target_demo_gate(hidden_states) if self.split_target_demo_attention else None,
                )
            else:
                attn_output = self._apply_varlen_flash_attention(
                    queries=queries,
                    keys=keys,
                    values=values,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=is_causal,
                )
            attn_output = attn_output.contiguous().view(total_tokens, dim)
            return self.out_proj(attn_output)

        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must be rank-2 or rank-3.")
        batch_size, seq_len, dim = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)
        qkv_dtype = queries.dtype

        if pos_xyz is not None:
            # Match cache path semantics: rotate in fp32 for stability, then
            # cast back so q/k/v dtypes stay aligned for SDPA.
            q_f32 = queries.float()
            k_f32 = keys.float()
            q_f32, k_f32 = self.rope.apply_rotary(q_f32, k_f32, pos_xyz)
            queries = q_f32.to(dtype=qkv_dtype)
            keys = k_f32.to(dtype=qkv_dtype)

        values = self._apply_value_embedding(
            values,
            hidden_states,
            value_embed,
        )

        if self.split_target_demo_attention:
            attn_output = padded_attention(queries, keys, values, sdpa_mask,
                pos_xyz[..., 2], self.target_demo_gate(hidden_states),
                self.dropout_p if self.training else 0.0)
        else:
            attn_output = self._apply_attention(
                queries, keys, values, attention_mask=attention_mask,
                causal_mask=causal_mask, sdpa_mask=sdpa_mask,
                dropout_p=self.dropout_p if self.training else 0.0, is_causal=is_causal)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        )
        return self.out_proj(attn_output)

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        pos_xyz: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        decode_block_mask: Optional[object] = None,
        value_embed: Optional[torch.Tensor] = None,
        sdpa_mask: Optional[torch.Tensor] = None,
        prefix_lm_layout: Optional[PackedPrefixLMLayout] = None,
        demo_attention_mask: Optional[torch.Tensor] = None,
        split_cache: Optional[tuple] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        batch_size, seq_len, dim = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)
        qkv_dtype = queries.dtype

        if pos_xyz is not None:
            # Cast to fp32 for rotation precision
            q_f32 = queries.float()
            k_f32 = keys.float()
            q_f32, k_f32 = self.rope.apply_rotary(q_f32, k_f32, pos_xyz)
            # Cast back to original dtype (e.g. bfloat16) for storage
            queries = q_f32.to(dtype=qkv_dtype)
            keys = k_f32.to(dtype=qkv_dtype)

        values = self._apply_value_embedding(
            values,
            hidden_states,
            value_embed,
        )

        # ------------------------------------------------------------------
        # PATH S: DECODE over a split demo/target cache (split attention only)
        # ------------------------------------------------------------------
        if split_cache is not None:
            if not self.split_target_demo_attention:
                raise ValueError("A split decode cache requires split target/demo attention.")
            demo_keys, demo_values, target_keys, target_values, demo_lengths, target_lengths, kernel = split_cache
            attn_output = split_cache_attention(
                queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                (demo_keys, demo_values, target_keys, target_values), demo_lengths, target_lengths,
                self.target_demo_gate(hidden_states), kernel)
            return self.out_proj(attn_output.reshape(batch_size, seq_len, dim)), None

        # ------------------------------------------------------------------
        # PATH A: DECODE (We have a cache buffer)
        # ------------------------------------------------------------------
        if past_key_value is not None:
            past_keys, past_values = past_key_value
            key_layer = past_keys
            value_layer = past_values
            if cache_position is not None:
                positions = _cache_query_positions(cache_position, batch_size, seq_len)
                if cache_position.dim() == 2:
                    # index_put_ stays in place under torch.compile (a scatter_
                    # into the input cache became a full copy per layer and call).
                    rows = torch.arange(batch_size, device=keys.device)[:, None, None]
                    heads = torch.arange(keys.size(1), device=keys.device)[None, :, None]
                    slots = positions[:, None, :]
                    past_keys.index_put_((rows, heads, slots), keys)
                    past_values.index_put_((rows, heads, slots), values)
                else:
                    past_keys.index_copy_(2, positions[0], keys)
                    past_values.index_copy_(2, positions[0], values)

                # Use the FULL buffer for attention (masking handles the future tokens)
                key_layer = past_keys
                value_layer = past_values

            # A key-validity mask alone is sufficient for one-token decoding.
            # Block verification additionally masks later speculative keys.
            query_positions = None
            if seq_len > 1 or (cache_position is not None and cache_position.dim() == 2):
                if cache_position is None:
                    raise ValueError("Block verification requires cache_position.")
                query_positions = _cache_query_positions(cache_position, batch_size, seq_len)

            use_flex_decode = (
                self._has_flex_attention
                and queries.device.type == "cuda"
                and attention_mask is not None
            )
            if self.split_target_demo_attention:
                if demo_attention_mask is None or attention_mask is None:
                    raise ValueError("Split cached decoding requires demo_attention_mask and attention_mask.")
                attn_output = cached_attention(queries, key_layer, value_layer,
                    attention_mask, demo_attention_mask, query_positions,
                    self.target_demo_gate(hidden_states))
            elif use_flex_decode:
                attn_output = self._apply_flex_decode_attention(
                    queries,
                    key_layer,
                    value_layer,
                    attention_mask=attention_mask,
                    decode_block_mask=decode_block_mask,
                    query_positions=query_positions,
                )
            else:
                verification_bias = None
                if query_positions is not None:
                    allowed = torch.arange(key_layer.size(2), device=queries.device)[None, None, :] <= query_positions[:, :, None]
                    if attention_mask is not None:
                        allowed = allowed & attention_mask[:, None, :key_layer.size(2)]
                    verification_bias = torch.zeros_like(allowed, dtype=queries.dtype).masked_fill(~allowed, float("-inf")).unsqueeze(1)
                attn_output = self._apply_attention(
                    queries,
                    key_layer,
                    value_layer,
                    attention_mask=attention_mask,
                    causal_mask=None,
                    sdpa_mask=verification_bias,
                    dropout_p=0.0,
                    is_causal=False,
                )

            attn_output = (
                attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
            )

            # --- Return ONLY the output. Cache is already updated. ---
            return self.out_proj(attn_output), None

        # ------------------------------------------------------------------
        # PATH B: PROMPT (We are initializing)
        # ------------------------------------------------------------------
        if prefix_lm_layout is not None:
            valid = prefix_lm_layout.padded_token_indices
            def pack(tensor):
                return tensor.transpose(1, 2).reshape(-1, self.n_heads, self.head_dim).index_select(0, valid)
            packed_output = self._apply_varlen_prefix_lm_attention(
                pack(queries), pack(keys), pack(values), prefix_lm_layout, dropout_p=0.0,
                gate_logits=self.target_demo_gate(hidden_states).reshape(-1, self.n_heads).index_select(0, valid)
                    if self.split_target_demo_attention else None,
            )
            flat_output = torch.zeros((batch_size * seq_len, self.n_heads, self.head_dim),
                                      device=queries.device, dtype=queries.dtype)
            flat_output = flat_output.index_copy(0, valid, packed_output)
            attn_output = flat_output.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        elif self.split_target_demo_attention:
            attn_output = padded_attention(queries, keys, values, sdpa_mask,
                pos_xyz[..., 2], self.target_demo_gate(hidden_states), 0.0)
        else:
            attn_output = self._apply_attention(
                queries,
                keys,
                values,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                sdpa_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=causal_mask is not None,
            )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        )

        # --- Return output AND the new KVs so we can build the buffer ---
        return self.out_proj(attn_output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.d_model, config.d_ff * 2)
        self.fc_out = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.fc_in(hidden_states).chunk(2, dim=-1)
        hidden_states = hidden_states * F.silu(gate)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return self.dropout(hidden_states)


def mark_block_compile_region() -> bool:
    """Compile the repeated transformer block once and stamp it out per application.

    With 38 block applications per forward, tracing and compiling the unrolled
    schedule took 5-9 minutes per run. torch.compiler.nested_compile_region
    compiles the marked region on first encounter and reuses the compiled code
    for every later call with matching inputs, while the loop glue, exits and
    loss stay in the outer graph. Idempotent and process-wide: it wraps the
    class method, so every block instance shares the region. Returns whether
    this PyTorch supports it.
    """
    region = getattr(torch.compiler, "nested_compile_region", None)
    if region is None:
        return False
    if not getattr(TransformerBlock, "_compile_region_marked", False):
        TransformerBlock._eager_forward = TransformerBlock.forward
        TransformerBlock.forward = region(TransformerBlock.forward)
        TransformerBlock._compile_region_marked = True
    return True


def unmark_block_compile_region() -> None:
    if getattr(TransformerBlock, "_compile_region_marked", False):
        TransformerBlock.forward = TransformerBlock._eager_forward
        TransformerBlock._compile_region_marked = False


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyTransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(config, layer_idx)
        self.ln_2 = RMSNorm(config.d_model)
        self.ff = FeedForward(config)
        self.hidden_state_scale = nn.Parameter(torch.tensor(1.0))
        self.token_input_skip_scale = nn.Parameter(torch.tensor(0.0))
        self.example_input_skip_scale = nn.Parameter(torch.tensor(0.0))
        if config.dihedral_embeddings:
            self.dihedral_input_skip_scale = nn.Parameter(torch.tensor(0.0))
        if config.input_shortcuts == "fixed":
            for scale in (self.hidden_state_scale, self.token_input_skip_scale,
                          self.example_input_skip_scale):
                scale.requires_grad_(False)
            if config.dihedral_embeddings:
                self.dihedral_input_skip_scale.requires_grad_(False)

    def _apply_input_shortcut(
        self,
        hidden_states: torch.Tensor,
        token_input_residual: torch.Tensor,
        example_input_residual: torch.Tensor,
        dihedral_input_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_scale = self.hidden_state_scale.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        token_skip_scale = self.token_input_skip_scale.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        example_skip_scale = self.example_input_skip_scale.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        result = (
            hidden_states * hidden_scale
            + token_input_residual * token_skip_scale
            + example_input_residual * example_skip_scale
        )
        if dihedral_input_residual is not None:
            result = result + dihedral_input_residual * self.dihedral_input_skip_scale.to(
                device=hidden_states.device, dtype=hidden_states.dtype,
            )
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_input_residual: torch.Tensor,
        example_input_residual: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor],
        pos_xyz: Optional[torch.Tensor],
        sdpa_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        prefix_lm_layout: Optional[PackedPrefixLMLayout] = None,
        value_embed: Optional[torch.Tensor] = None,
        dihedral_input_residual: Optional[torch.Tensor] = None,
        skip_ff: bool = False,
        norm_gains: Optional[torch.Tensor] = None,
        rope_tables: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = self._apply_input_shortcut(
            hidden_states,
            token_input_residual=token_input_residual,
            example_input_residual=example_input_residual,
            dihedral_input_residual=dihedral_input_residual,
        )
        attn_input = self.ln_1(hidden_states)
        if norm_gains is not None:
            attn_input = attn_input * norm_gains[0].to(attn_input.dtype)
        attn_output = self.attention(
            attn_input,
            value_embed=value_embed,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            pos_xyz=pos_xyz,
            sdpa_mask=sdpa_mask,
            is_causal=is_causal,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            prefix_lm_layout=prefix_lm_layout,
            rope_tables=rope_tables,
        )
        hidden_states = hidden_states + attn_output

        if not skip_ff:
            ff_input = self.ln_2(hidden_states)
            if norm_gains is not None:
                ff_input = ff_input * norm_gains[1].to(ff_input.dtype)
            hidden_states = hidden_states + self.ff(ff_input)
        return hidden_states

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        token_input_residual: torch.Tensor,
        example_input_residual: torch.Tensor,
        pos_xyz: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        decode_block_mask: Optional[object] = None,
        value_embed: Optional[torch.Tensor] = None,
        dihedral_input_residual: Optional[torch.Tensor] = None,
        sdpa_mask: Optional[torch.Tensor] = None,
        prefix_lm_layout: Optional[PackedPrefixLMLayout] = None,
        demo_attention_mask: Optional[torch.Tensor] = None,
        skip_ff: bool = False,
        norm_gains: Optional[torch.Tensor] = None,
        split_cache: Optional[tuple] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        hidden_states = self._apply_input_shortcut(
            hidden_states,
            token_input_residual=token_input_residual,
            example_input_residual=example_input_residual,
            dihedral_input_residual=dihedral_input_residual,
        )
        attn_input = self.ln_1(hidden_states)
        if norm_gains is not None:
            attn_input = attn_input * norm_gains[0].to(attn_input.dtype)
        attn_output, present_key_value = (
            self.attention.forward_with_cache(
                attn_input,
                pos_xyz=pos_xyz,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                decode_block_mask=decode_block_mask,
                value_embed=value_embed,
                sdpa_mask=sdpa_mask,
                prefix_lm_layout=prefix_lm_layout,
                demo_attention_mask=demo_attention_mask,
                split_cache=split_cache,
            )
        )
        hidden_states = hidden_states + attn_output

        if not skip_ff:
            ff_input = self.ln_2(hidden_states)
            if norm_gains is not None:
                ff_input = ff_input * norm_gains[1].to(ff_input.dtype)
            hidden_states = hidden_states + self.ff(ff_input)

        if attention_mask is not None:
            seq_len = hidden_states.size(1)
            if cache_position is not None:
                # Build positions for the currently generated tokens so we mask the right slot
                positions = _cache_query_positions(cache_position, hidden_states.size(0), seq_len)
                positions = positions.clamp(max=attention_mask.size(1) - 1)
            else:
                positions = torch.arange(
                    attention_mask.size(1) - seq_len,
                    attention_mask.size(1),
                    device=hidden_states.device,
                ).view(1, -1)
                positions = positions.expand(attention_mask.size(0), -1)

            token_mask = attention_mask.gather(1, positions)
            hidden_states = hidden_states * token_mask.unsqueeze(-1)
        return hidden_states, present_key_value


class TinyTransformer(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        # Training-loss option (not architecture, not saved in the config): the
        # saturation-safe hard-target CE of stable_losses.py instead of F.cross_entropy.
        self.stable_cross_entropy = False
        if config.pair_identity_vertices:
            self.register_buffer("pair_identity_table", simplex_table(config.pair_identity_vertices, config.d_model), persistent=False)

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.task_factor_embedding = nn.Embedding(
            config.num_examples,
            config.num_task_embedding_latents,
        )
        self.example_embedding = TaskFactorizedEmbedding(
            config.num_task_embedding_latents,
            config.d_model,
        )
        self.value_embedding_layers = tuple(
            _has_value_embedding(i, config.n_layers) for i in range(config.n_layers)
        )
        self.value_embeddings = nn.ModuleDict(
            {
                str(i): TaskFactorizedEmbedding(
                    config.num_task_embedding_latents,
                    config.d_model,
                )
                for i in range(config.n_layers)
                if self.value_embedding_layers[i]
            }
        )
        self.example_gate_logit = nn.Parameter(torch.tensor(-2.0))
        if config.dihedral_embeddings:
            self.dihedral_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.input_norm = RMSNorm(config.d_model)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        nn.init.normal_(
            self.task_factor_embedding.weight,
            mean=0.0,
            std=config.num_task_embedding_latents ** -0.5,
        )
        for block in self.blocks:
            if block.attention.value_embed_gate is not None:
                nn.init.zeros_(block.attention.value_embed_gate.weight)

        # Initialize on a fork of the CPU RNG so embedding and MTP ablations
        # keep all shared initial weights equal, including the restored table.
        if config.dihedral_embeddings:
            with torch.random.fork_rng(devices=[]):
                self.dihedral_embedding = nn.Embedding(8, config.d_model)
                self.dihedral_embedding.apply(self._init_weights)

        # Initialize after the existing model so adding heads does not change
        # the trunk/next-token head's initial weights for a given seed.
        self.lm_head_mtp = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.vocab_size, bias=False)
                for _ in range(config.n_predict - 1)
            ]
        )
        self.lm_head_mtp.apply(self._init_weights)

        # Optional modules must not perturb shared initialization or dropout RNG
        # in the matched baseline. Old checkpoints have neither of these keys.
        with torch.random.fork_rng(devices=[]):
            self.next_latent_dynamics = (
                NextLatentDynamics(config.d_model, config.next_latent_hidden_dim)
                if config.next_latent_prediction else None
            )
        if config.loop_iterations > 1 and config.loop_input_injection:
            self.loop_input_scale = nn.Parameter(torch.tensor(0.0))
        start, end = config.loop_start_layer, config.loop_end_layer
        self.layer_schedule = tuple(range(config.n_layers)) if config.loop_iterations == 1 else (
            tuple(range(start)) + tuple(range(start, end)) * config.loop_iterations
            + tuple(range(end, config.n_layers))
        )
        self._loop_restart_steps = () if config.loop_iterations == 1 else tuple(
            start + (end - start) * iteration for iteration in range(1, config.loop_iterations)
        )
        self._init_loop_extensions(config)

        if config.split_target_demo_attention:
            with torch.random.fork_rng(devices=[]):
                for block in self.blocks:
                    gate = nn.Linear(config.d_model, config.n_heads, bias=True)
                    nn.init.zeros_(gate.weight)
                    nn.init.zeros_(gate.bias)
                    block.attention.target_demo_gate = gate

        # Without task embeddings the task tables would have no lookup,
        # gradient or optimizer state, so they are deleted. They are still
        # built and initialized above, so every other weight starts the same
        # for a given seed. The small gates stay in the state dict, frozen.
        if not config.task_embeddings:
            for name in TASK_TABLE_MODULES:
                delattr(self, name)
            self.example_gate_logit.requires_grad_(False)
            for block in self.blocks:
                block.example_input_skip_scale.requires_grad_(False)
                if block.attention.value_embed_gate is not None:
                    block.attention.value_embed_gate.requires_grad_(False)

        # Cache structural mask metadata by device, capacity, and KV block.
        self._decode_block_mask_cache: Dict[tuple, object] = {}
        self._decode_block_size = 128

    @property
    def conditioning_dtype(self) -> torch.dtype:
        """Dtype of the example/dihedral conditioning stream."""
        table = self.example_embedding if self.config.task_embeddings else self.token_embedding
        return table.weight.dtype

    @property
    def cache_layer_count(self) -> int:
        """One KV cache per logical invocation, even when weights are shared."""
        return len(self.layer_schedule)

    def _init_loop_extensions(self, config) -> None:
        """Static per-logical-step loop metadata and identity-initialized parameters."""
        total = len(self.layer_schedule)
        iteration_of, application = [0] * total, [-1] * total
        self._loop_iteration_end: Dict[int, int] = {}
        self._loop_dense_pairs: Dict[Tuple[int, int], int] = {}
        self._loop_span = 0
        if config.loop_iterations > 1:
            start, end = config.loop_start_layer, config.loop_end_layer
            span = end - start
            self._loop_span = span
            for k in range(config.loop_iterations):
                for position in range(span):
                    iteration_of[start + k * span + position] = k + 1
                    application[start + k * span + position] = k * span + position
                self._loop_iteration_end[start + span * (k + 1) - 1] = k + 1
            self._loop_dense_pairs = {pair: i for i, pair in enumerate(loop_dense_pairs(config.loop_iterations))}
        self._loop_iteration_of = tuple(iteration_of)
        self._loop_application = tuple(application)
        # Deep supervision exits: every iteration but the last unless restricted.
        self._loop_exit_iterations = frozenset(
            config.loop_exit_iterations if config.loop_exit_iterations is not None
            else range(1, config.loop_iterations))
        for name, shape in loop_extension_shapes(config).items():
            if name == "loop_gate_bias":
                half = config.loop_gate_init / 2.0
                value = torch.full(shape, math.log(half / (1.0 - half)))
            elif name in ("loop_norm_gains", "loop_stream_write"):
                value = torch.ones(shape)
            elif name == "loop_stream_read":
                # All streams start equal; distinct one-hot readers break symmetry.
                value = torch.zeros(shape)
                for row in range(config.loop_iterations):
                    value[row, min(row, config.loop_streams - 1)] = 1.0
                value[config.loop_iterations, config.loop_streams - 1] = 1.0
            else:
                value = torch.zeros(shape)
            setattr(self, name, nn.Parameter(value))

    def _loop_stream_read(self, streams, row):
        weights = self.loop_stream_read[row].to(streams.dtype)
        return torch.tensordot(weights, streams, dims=1)

    def _loop_input(self, hidden_states, state, logical_idx):
        """Prepare a logical block's input; state carries per-token loop latents.

        Every operation is position-wise, so cached decoding recomputes the
        same state for new tokens without any cache beyond per-logical-layer KV.
        """
        config = self.config
        if config.loop_iterations == 1:
            return hidden_states, state
        if state is None:
            state = {}
        width = (config.d_model,)
        if logical_idx == config.loop_start_layer:
            state["prelude"] = hidden_states
            state["latents"] = [hidden_states]
            if config.loop_streams:
                state["streams"] = hidden_states.unsqueeze(0).expand(config.loop_streams, *hidden_states.shape)
                hidden_states = self._loop_stream_read(state["streams"], 0)
            state["iteration_input"] = hidden_states
        elif logical_idx in self._loop_restart_steps:
            iteration = self._loop_iteration_of[logical_idx]
            dtype = hidden_states.dtype
            if config.loop_streams:
                hidden_states = self._loop_stream_read(state["streams"], iteration - 1)
            # Stream updates are measured from the read, so the injected terms
            # below stay in the residual exactly as they do in the plain loop.
            state["iteration_input"] = hidden_states
            if config.loop_input_injection:
                injected = F.rms_norm(state["prelude"], width)
                hidden_states = hidden_states + self.loop_input_scale.to(dtype) * injected
            if config.loop_dense_mixing:
                for earlier in range(1, iteration - 1):
                    scale = self.loop_dense_scales[self._loop_dense_pairs[(iteration, earlier)]]
                    hidden_states = hidden_states + scale.to(dtype) * F.rms_norm(state["latents"][earlier], width)
            if config.loop_iteration_embedding:
                hidden_states = hidden_states + self.loop_iteration_embeddings[iteration - 2].to(dtype)
        return hidden_states, state

    def _loop_block_options(self, logical_idx) -> dict:
        iteration = self._loop_iteration_of[logical_idx]
        if iteration < 2 or self.config.loop_iterations == 1:
            return {}
        options = {}
        if self.config.loop_ffn == "first":
            options["skip_ff"] = True
        if self.config.loop_iteration_norms:
            options["norm_gains"] = self.loop_norm_gains[self._loop_application[logical_idx] - self._loop_span]
        return options

    def _loop_gate(self, block_input, hidden_states, logical_idx):
        """Scale a looped block's residual update by a learned gate in (0, 2)."""
        application = self._loop_application[logical_idx]
        if self.config.loop_gate == "none" or application < 0 or self.config.loop_iterations == 1:
            return hidden_states
        logit = self.loop_gate_bias[application].to(block_input.dtype)
        if self.config.loop_gate == "token":
            weight = self.loop_gate_weight[application].to(block_input.dtype)
            # 1/sqrt(width) keeps AdamW's per-coordinate steps from saturating the gate.
            dynamic = F.rms_norm(block_input, (self.config.d_model,)) @ weight
            logit = logit + dynamic.unsqueeze(-1) * self.config.d_model ** -0.5
        gate = 2.0 * torch.sigmoid(logit)
        return block_input + gate * (hidden_states - block_input)

    def _loop_output(self, hidden_states, state, logical_idx):
        """Record an iteration's output latent and update the residual streams."""
        iteration = self._loop_iteration_end.get(logical_idx)
        if iteration is None or self.config.loop_iterations == 1:
            return hidden_states, state
        config = self.config
        width = (config.d_model,)
        dtype = hidden_states.dtype
        state["latents"].append(hidden_states)
        if config.loop_streams:
            update_input = state["iteration_input"]
            shape = (config.loop_streams,) + (1,) * hidden_states.dim()
            logit = self.loop_stream_gate_bias[iteration - 1].to(dtype).view(shape)
            if config.loop_stream_gate == "dynamic":
                weight = self.loop_stream_gate_weight[iteration - 1].to(dtype)
                dynamic = (F.rms_norm(update_input, width) @ weight.t()) * config.d_model ** -0.5
                logit = logit + dynamic.movedim(-1, 0).unsqueeze(-1)
            keep = 2.0 * torch.sigmoid(logit)
            write = self.loop_stream_write[iteration - 1].to(dtype).view(shape)
            delta = (hidden_states - update_input).unsqueeze(0)
            state["streams"] = keep * state["streams"] + write * delta
        if iteration == config.loop_iterations:
            if config.loop_streams:
                hidden_states = self._loop_stream_read(state["streams"], config.loop_iterations)
            if config.loop_dense_mixing:
                for earlier in range(config.loop_iterations):
                    scale = self.loop_coda_scales[earlier].to(dtype)
                    hidden_states = hidden_states + scale * F.rms_norm(state["latents"][earlier], width)
        return hidden_states, state

    def _loop_exit_due(self, logical_idx) -> bool:
        """Deep supervision: an early exit after each selected loop iteration (never the last)."""
        iteration = self._loop_iteration_end.get(logical_idx)
        return (self.training and self.config.loop_exit_weight > 0 and self.config.loop_iterations > 1
                and iteration is not None and iteration in self._loop_exit_iterations)

    def _loop_exit_states(self, exit_states, run_block):
        """Run the coda (blocks after the loop span) and the final norm on each early exit."""
        coda_start = self.config.loop_start_layer + self._loop_span * self.config.loop_iterations if exit_states else 0
        finished = []
        for states in exit_states:
            for layer_idx in self.layer_schedule[coda_start:]:
                states = run_block(layer_idx, states)
            finished.append(self.norm(states))
        return finished

    def _cross_entropy(self, logits, targets, *, ignore_index, reduction):
        if self.stable_cross_entropy:
            return hard_target_cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)
        return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)

    def _loop_exit_loss(self, exit_states, logits_fn, targets, valid_output, num_output_tokens):
        if not exit_states:
            return None
        losses = []
        for states in exit_states:
            # Same CE as the final head, so exits match its numerics.
            raw = self._cross_entropy(logits_fn(states), targets, ignore_index=IGNORE_INDEX, reduction="none")
            losses.append((raw * valid_output).sum() / num_output_tokens.clamp(min=1))
        return torch.stack(losses).mean()

    def _next_latent_output(self, hidden_states, input_ids, valid_output, *,
                            compute_next_latent_loss=True, cu_seqlens=None, source_indices=None):
        result = dict(next_latent_loss=None, next_latent_regression_loss=None,
                      next_latent_kl_loss=None, num_next_latent_tokens=None)
        if (not compute_next_latent_loss or self.next_latent_dynamics is None
                or not (self.config.next_latent_regression_weight or self.config.next_latent_kl_weight)):
            return result
        regression, kl, count = next_latent_losses(
            self.next_latent_dynamics, hidden_states, input_ids, valid_output,
            self.token_embedding, self.lm_head, cu_seqlens=cu_seqlens, source_indices=source_indices,
            horizon=self.config.next_latent_horizon,
        )
        result.update(
            next_latent_loss=self.config.next_latent_regression_weight * regression
                + self.config.next_latent_kl_weight * kl,
            next_latent_regression_loss=regression,
            next_latent_kl_loss=kl,
            num_next_latent_tokens=count,
        )
        return result

    def _add_pair_identities(self, token_embeds, pair_identity_ids):
        if not self.config.pair_identity_vertices:
            return token_embeds
        if pair_identity_ids is None:
            raise ValueError("Simplex model requires explicit per-token pair_identity_ids")
        if pair_identity_ids.shape != token_embeds.shape[:-1]:
            raise ValueError("pair_identity_ids must match input token shape")
        ids = pair_identity_ids.to(device=token_embeds.device, dtype=torch.long)
        valid = torch.all((ids >= 0) & (ids <= self.config.pair_identity_vertices))
        if _is_torch_compiling():
            # Data checks must remain tensor operations in full-graph cached
            # decoding; a synchronous torch._assert requires a scalar guard.
            torch._assert_async(valid, "Pair ID outside simplex palette")
        else:
            _compile_safe_assert(valid, "Pair ID outside simplex palette")
        return token_embeds + self.config.pair_identity_strength * F.embedding(ids, self.pair_identity_table).to(token_embeds.dtype)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        return mask[None, None, :, :]

    def _expand_conditioning_embeds(
        self,
        token_embeds: torch.Tensor,
        conditioning_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeds.dim() == conditioning_embeds.dim() + 1:
            conditioning_embeds = conditioning_embeds.unsqueeze(-2)
        return conditioning_embeds

    def _normalize_example_ids(
        self,
        example_ids: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        example_ids = example_ids.to(device=device, dtype=torch.int32)
        if example_ids.dim() == 1 and example_ids.size(0) == batch_size:
            return example_ids
        if (
            example_ids.dim() == 2
            and example_ids.size(0) == batch_size
            and example_ids.size(1) == 2
        ):
            return example_ids
        raise ValueError("example_ids must have shape [batch_size] or [batch_size, 2].")

    def _lookup_example_embeds(self, example_ids: torch.Tensor) -> torch.Tensor:
        if not self.config.task_embeddings:
            return self.token_embedding.weight.new_zeros((example_ids.size(0), self.config.d_model))
        return self._lookup_task_indexed_embeds(self.example_embedding, example_ids)

    def _lookup_dihedral_embeds(
        self,
        dihedral_ids: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.config.dihedral_embeddings:
            return None
        if dihedral_ids is None:
            raise ValueError("dihedral_ids are required when dihedral_embeddings is enabled.")
        if dihedral_ids.dim() != 1 or dihedral_ids.size(0) != batch_size:
            raise ValueError("dihedral_ids must have shape [batch_size].")
        if dihedral_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("dihedral_ids must contain integer transform indices.")
        # Embedding lookup enforces the range [0, 7] without a host-side read
        # or synchronization in the compiled training/decode paths.
        return self.dihedral_embedding(dihedral_ids.to(device=device))

    def _lookup_task_indexed_embeds(
        self, embedding: TaskFactorizedEmbedding, example_ids: torch.Tensor
    ) -> torch.Tensor:
        task_factors = self.task_factor_embedding(example_ids.clamp(min=0))
        example_embeds = embedding.project(task_factors)
        if example_ids.dim() == 1:
            return example_embeds
        example_mask = (example_ids >= 0).to(dtype=example_embeds.dtype).unsqueeze(-1)
        return (example_embeds * example_mask).sum(dim=1)

    def _lookup_value_embedding(
        self,
        layer_idx: int,
        example_ids: torch.Tensor,
        *,
        seq_lengths: Optional[torch.Tensor] = None,
        total_tokens: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if not self.config.task_embeddings or not self.value_embedding_layers[layer_idx]:
            return None
        value_embed = self._lookup_task_indexed_embeds(
            self.value_embeddings[str(layer_idx)],
            example_ids,
        )
        if seq_lengths is not None:
            # A known output size keeps repeat_interleave from reading the
            # lengths back to the host inside the compiled forward.
            value_embed = torch.repeat_interleave(value_embed, seq_lengths, dim=0, output_size=total_tokens)
        return value_embed

    def compute_value_embeds(
        self,
        example_ids: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        device = example_ids.device
        batch_size = int(example_ids.size(0))
        example_ids = self._normalize_example_ids(
            example_ids,
            batch_size=batch_size,
            device=device,
        )
        return tuple(
            self._lookup_value_embedding(layer_idx, example_ids)
            for layer_idx in range(self.config.n_layers)
        )

    def _value_embedding_for_layer(
        self,
        layer_idx: int,
        example_ids: torch.Tensor,
        value_embeds: Optional[Tuple[Optional[torch.Tensor], ...]],
    ) -> Optional[torch.Tensor]:
        if not self.config.task_embeddings:
            return None
        if value_embeds is None:
            return self._lookup_value_embedding(layer_idx, example_ids)
        if len(value_embeds) != self.config.n_layers:
            raise ValueError(
                "value_embeds must contain one entry per transformer layer."
            )
        return value_embeds[layer_idx]

    def _embed_inputs(
        self,
        token_embeds: torch.Tensor,
        *,
        conditioning_embeds: torch.Tensor,
    ) -> torch.Tensor:
        conditioning_embeds = self._expand_conditioning_embeds(
            token_embeds,
            conditioning_embeds,
        )

        hidden_states = token_embeds + conditioning_embeds
        return self.input_norm(hidden_states)

    def _conditioning_embeds(
        self,
        example_embeds: torch.Tensor,
        dihedral_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        example_stream, dihedral_stream = self._conditioning_streams(
            example_embeds, dihedral_embeds,
        )
        return example_stream if dihedral_stream is None else example_stream + dihedral_stream

    def _conditioning_streams(
        self,
        example_embeds: torch.Tensor,
        dihedral_embeds: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        g_example = torch.sigmoid(self.example_gate_logit).to(
            device=example_embeds.device,
            dtype=example_embeds.dtype,
        )
        dihedral_stream = None
        if dihedral_embeds is not None:
            dihedral_stream = torch.sigmoid(self.dihedral_gate_logit).to(
                device=dihedral_embeds.device, dtype=dihedral_embeds.dtype,
            ) * dihedral_embeds
        example_stream = g_example * example_embeds if self.config.task_embeddings else torch.zeros_like(example_embeds)
        return example_stream, dihedral_stream

    def _prepare_conditioning_streams(
        self,
        token_embeds: torch.Tensor,
        *,
        example_embeds: torch.Tensor,
        dihedral_embeds: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        example_stream, dihedral_stream = self._conditioning_streams(
            example_embeds, dihedral_embeds,
        )
        example_stream = self._expand_conditioning_embeds(token_embeds, example_stream)
        if dihedral_stream is not None:
            dihedral_stream = self._expand_conditioning_embeds(token_embeds, dihedral_stream)
        return example_stream, dihedral_stream

    def _prepare_input_streams(
        self,
        token_embeds: torch.Tensor,
        *,
        example_embeds: torch.Tensor,
        dihedral_embeds: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        example_stream, dihedral_stream = self._prepare_conditioning_streams(
            token_embeds,
            example_embeds=example_embeds,
            dihedral_embeds=dihedral_embeds,
        )
        hidden_states = self._embed_inputs(
            token_embeds,
            conditioning_embeds=(example_stream if dihedral_stream is None else example_stream + dihedral_stream),
        )
        return hidden_states, token_embeds, example_stream, dihedral_stream

    def compute_conditioning_embeds(
        self,
        example_ids: torch.Tensor,
        dihedral_ids: Optional[torch.Tensor] = None,
        *,
        example_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = example_ids.device
        batch_size = int(example_ids.size(0))
        example_ids = self._normalize_example_ids(
            example_ids,
            batch_size=batch_size,
            device=device,
        )
        if example_embeds is None:
            example_embeds = self._lookup_example_embeds(example_ids)
        else:
            example_embeds = example_embeds.to(
                device=device, dtype=self.conditioning_dtype
            )
        dihedral_embeds = self._lookup_dihedral_embeds(
            dihedral_ids, batch_size=batch_size, device=device,
        )
        return self._conditioning_embeds(example_embeds, dihedral_embeds)

    def _build_prefix_lm_sdpa_mask(
        self,
        attention_mask: torch.Tensor,
        prefix_lengths: torch.Tensor,
        dtype: torch.dtype,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device

        prefix_lengths = prefix_lengths.to(device=device)
        prefix_lengths = prefix_lengths.clamp(min=1, max=seq_len)

        q_positions = torch.arange(seq_len, device=device).view(1, seq_len, 1)
        k_positions = torch.arange(seq_len, device=device).view(1, 1, seq_len)
        prefix_limits = prefix_lengths.view(batch_size, 1, 1)

        prefix_queries = q_positions < prefix_limits
        prefix_invalid = k_positions >= prefix_limits
        output_invalid = k_positions > q_positions
        invalid = torch.where(prefix_queries, prefix_invalid, output_invalid)
        if self.config.attention_pattern == "pair_group":
            if input_ids is None:
                raise ValueError("Pair-group attention requires input_ids to identify pairs.")
            pair_ids, hub = padded_prefix_structure(input_ids, attention_mask, prefix_lengths)
            same_pair = pair_ids[:, :, None] == pair_ids[:, None, :]
            prefix_allowed = hub[:, :, None] | hub[:, None, :] | same_pair
            invalid = invalid | (prefix_queries & ~prefix_allowed)
        invalid = invalid | (~attention_mask[:, None, :])

        sdpa_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len), device=device, dtype=dtype
        )
        return sdpa_mask.masked_fill(invalid.unsqueeze(1), float("-inf"))

    @torch.compiler.disable
    def _build_packed_prefix_lm_layout(
        self,
        cu_seqlens: torch.Tensor,
        total_tokens: int,
        prefix_lengths: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        precomputed: Optional[dict] = None,
    ) -> PackedPrefixLMLayout:
        """Assemble the batch layout eagerly, then resume compilation in the layers.

        Scalar/dynamic-shape capture cannot safely trace the layout's Python
        branches, so the whole layout stays outside Dynamo. A layout collated on
        the CPU is adopted when it matches this model's attention pattern;
        otherwise it is rebuilt on the device.
        """
        with torch.profiler.record_function("packed_layout"):
            if precomputed is not None:
                layout = self._adopt_precomputed_layout(precomputed, cu_seqlens.device)
                if layout is not None:
                    return layout
            return PackedPrefixLMLayout(**packed_prefix_lm_layout_fields(
                cu_seqlens, total_tokens, prefix_lengths, input_ids,
                attention_pattern=self.config.attention_pattern,
                split_target_demo=self.config.split_target_demo_attention,
            ))

    def _adopt_precomputed_layout(self, precomputed: dict, device) -> Optional[PackedPrefixLMLayout]:
        if precomputed.get("attention_pattern") != self.config.attention_pattern:
            return None
        split_layout = precomputed.get("split_target_demo_layout")
        if self.config.split_target_demo_attention and split_layout is None:
            return None

        def on_device(value):
            if isinstance(value, torch.Tensor):
                return value.to(device=device)
            if isinstance(value, dict):
                return {key: on_device(item) for key, item in value.items()}
            return value

        fields = {name: on_device(precomputed[name]) for name in _LAYOUT_FIELDS if name in precomputed}
        # A model without split attention must not route through the split path.
        fields["split_target_demo_layout"] = on_device(split_layout) if self.config.split_target_demo_attention else None
        return PackedPrefixLMLayout(**fields)

    def _get_decode_block_mask(
        self, cache_position: int, kv_len: int, device: torch.device
    ) -> Optional[object]:
        if create_block_mask is None or kv_len < 1:
            return None

        block_idx = max(int(cache_position), 0) // self._decode_block_size
        device_index = -1 if device.index is None else int(device.index)
        cache_key = (device.type, device_index, int(kv_len), block_idx)
        cached = self._decode_block_mask_cache.get(cache_key)
        if cached is not None:
            return cached

        block_end = min(((block_idx + 1) * self._decode_block_size) - 1, kv_len - 1)

        def block_superset_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h, q_idx
            return kv_idx <= block_end

        build_kwargs: Dict[str, Any] = {
            "mask_mod": block_superset_mask,
            "B": None,
            "H": None,
            "Q_LEN": 1,
            "KV_LEN": int(kv_len),
            "device": device,
            "BLOCK_SIZE": self._decode_block_size,
        }
        try:
            block_mask = create_block_mask(_compile=True, **build_kwargs)
        except TypeError:
            try:
                block_mask = create_block_mask(**build_kwargs)
            except TypeError:
                # Backward-compatible fallback for older signatures.
                block_mask = create_block_mask(
                    block_superset_mask,
                    1,
                    1,
                    1,
                    int(kv_len),
                    device=device,
                    BLOCK_SIZE=self._decode_block_size,
                )

        self._decode_block_mask_cache[cache_key] = block_mask
        return block_mask

    def _has_varlen_flash_attention(self) -> bool:
        return any(
            block.attention._has_selected_varlen_flash_attention()
            for block in self.blocks
        )

    def _forward_varlen_via_padded(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        dihedral_ids: Optional[torch.Tensor],
        sep_indices: torch.Tensor,
        targets: Optional[torch.Tensor],
        positions_3d: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int],
        compute_training_metrics: bool = False,
        compute_mtp_loss: bool = True,
        compute_next_latent_loss: bool = True,
        pair_identity_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        device = input_ids.device
        batch_size = int(example_ids.size(0))
        seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(
            device=device, dtype=torch.long
        )
        max_seqlen_resolved = (
            int(max_seqlen) if max_seqlen is not None else int(seq_lengths.max().item())
        )

        padded_input_ids = torch.full(
            (batch_size, max_seqlen_resolved),
            END_PAIR_TOKEN_ID,
            dtype=input_ids.dtype,
            device=device,
        )
        padded_positions_3d = torch.zeros(
            (batch_size, max_seqlen_resolved, 3),
            dtype=positions_3d.dtype,
            device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_seqlen_resolved),
            dtype=torch.bool,
            device=device,
        )
        padded_pair_ids = torch.zeros_like(padded_input_ids) if pair_identity_ids is not None else None
        padded_targets = None
        if targets is not None:
            padded_targets = torch.full(
                (batch_size, max_seqlen_resolved),
                IGNORE_INDEX,
                dtype=targets.dtype,
                device=device,
            )

        for batch_idx in range(batch_size):
            start = int(cu_seqlens[batch_idx].item())
            end = int(cu_seqlens[batch_idx + 1].item())
            seq_len = end - start
            padded_input_ids[batch_idx, :seq_len] = input_ids[start:end]
            padded_positions_3d[batch_idx, :seq_len] = positions_3d[start:end]
            if padded_pair_ids is not None:
                padded_pair_ids[batch_idx, :seq_len] = pair_identity_ids[start:end]
            attention_mask[batch_idx, :seq_len] = True
            if padded_targets is not None:
                # Packed labels are already next-token aligned; padded forward
                # shifts token-aligned labels once. Never shift them twice.
                padded_targets[batch_idx, 1:seq_len] = targets[start:end - 1]

        outputs = self._forward_padded(
            pair_identity_ids=padded_pair_ids,
            input_ids=padded_input_ids,
            example_ids=example_ids,
            dihedral_ids=dihedral_ids,
            attention_mask=attention_mask,
            sep_indices=sep_indices,
            targets=padded_targets,
            positions_3d=padded_positions_3d,
            compute_training_metrics=compute_training_metrics,
            compute_mtp_loss=compute_mtp_loss,
            compute_next_latent_loss=compute_next_latent_loss,
        )

        outputs["logits"] = outputs["logits"][attention_mask]
        return outputs

    def build_decode_block_mask_for_step(
        self,
        attention_mask: Optional[torch.Tensor],
        cache_position: Optional[torch.Tensor],
        kv_len: int,
        device: torch.device,
    ) -> Optional[object]:
        if attention_mask is None or cache_position is None:
            return None
        if attention_mask.dim() != 2:
            return None
        if attention_mask.size(1) < kv_len:
            return None

        if cache_position.dim() == 2:
            if BlockMask is None:
                return None
            query_len = cache_position.size(1)
            block_size = self._decode_block_size
            last_block = int(cache_position.max().item()) // block_size
            device_index = -1 if device.index is None else int(device.index)
            cache_key = ("verify", device.type, device_index, kv_len, query_len, last_block)
            topology = self._decode_block_mask_cache.get(cache_key)
            if topology is None:
                num_blocks = min(last_block + 1, (kv_len + block_size - 1) // block_size)
                num_q_blocks = (query_len + block_size - 1) // block_size
                counts = torch.full((1, 1, num_q_blocks), num_blocks, dtype=torch.int32, device=device)
                indices = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, 1, 1, -1).expand(1, 1, num_q_blocks, -1).contiguous()
                # All included blocks are partial: row lengths, left padding,
                # and causal boundaries must always run the elementwise mask.
                topology = (counts, indices)
                self._decode_block_mask_cache[cache_key] = topology
            key_mask = attention_mask[:, :kv_len]

            def verify_mask_mod(b, h, q_idx, kv_idx):
                del h
                q = q_idx.clamp(max=query_len - 1)
                k = kv_idx.clamp(max=kv_len - 1)
                return (q_idx < query_len) & (kv_idx < kv_len) & key_mask[b, k] & (kv_idx <= cache_position[b, q])

            kwargs = dict(kv_num_blocks=topology[0], kv_indices=topology[1],
                          BLOCK_SIZE=block_size, mask_mod=verify_mask_mod,
                          compute_q_blocks=False)
            try:
                return BlockMask.from_kv_blocks(seq_lengths=(query_len, kv_len), **kwargs)
            except TypeError:  # Older PyTorch versions infer rounded lengths.
                return BlockMask.from_kv_blocks(**kwargs)

        cache_pos = int(cache_position.reshape(-1)[0].item())
        block_mask = self._get_decode_block_mask(cache_pos, kv_len=kv_len, device=device)
        if block_mask is None:
            return None

        key_mask = attention_mask[:, :kv_len]
        q_offset = cache_position.reshape(-1)[0]

        def decode_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del h
            causal_ok = q_idx + q_offset >= kv_idx
            key_ok = key_mask[b, kv_idx]
            return causal_ok & key_ok

        try:
            block_mask.mask_mod = decode_mask_mod
        except Exception:
            return None
        return block_mask

    def _compute_training_metric_counts(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_output: torch.Tensor,
        sequence_ids: Optional[torch.Tensor] = None,
        num_sequences: Optional[int] = None,
    ) -> torch.Tensor:
        """Count teacher-forced color, structural-token, and exact-output accuracy."""
        predictions = logits.detach().argmax(dim=-1)
        is_color = valid_output & (targets >= 0) & (targets < START_TOKEN_ID)
        is_special = valid_output & (targets >= START_TOKEN_ID)
        correct = predictions == targets
        incorrect = valid_output & ~correct
        if targets.dim() == 2:
            has_output = valid_output.any(dim=-1)
            exact = has_output & ~incorrect.any(dim=-1)
        else:
            if sequence_ids is None or num_sequences is None:
                raise ValueError("Packed metrics require sequence_ids and num_sequences.")
            incorrect_counts = torch.zeros(
                num_sequences, dtype=torch.long, device=targets.device
            ).scatter_add(0, sequence_ids, incorrect.to(dtype=torch.long))
            output_counts = torch.zeros_like(incorrect_counts).scatter_add(
                0, sequence_ids, valid_output.to(dtype=torch.long)
            )
            has_output = output_counts > 0
            exact = has_output & (incorrect_counts == 0)

        # An episode without supervised output must not count as a correct grid.
        return torch.stack(
            (
                (correct & is_color).sum(),
                is_color.sum(),
                (correct & is_special).sum(),
                is_special.sum(),
                exact.sum(),
                has_output.sum(),
            )
        )

    def _compute_mtp_loss(
        self,
        hidden_states: torch.Tensor,
        next_targets: torch.Tensor,
        valid_output: torch.Tensor,
        *,
        cu_seqlens: Optional[torch.Tensor] = None,
        return_token_counts: bool = False,
    ):
        """Offsets 2..K share the trunk and supervise only output-side queries.

        No extra transformer passes or host reads; auxiliary logits are not returned.
        Labels reuse the next-token alignment, including caller ignore masks.
        With ``return_token_counts`` each head's valid-label count is returned too
        (DDP renormalizes the per-head means over the global batch).
        """
        if not self.lm_head_mtp or self.config.mtp_weight == 0:
            return (None, None, None) if return_token_counts else (None, None)

        remaining = None
        if hidden_states.dim() == 2:
            total_tokens = hidden_states.size(0)
            seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.long)
            sequence_ids = torch.repeat_interleave(
                torch.arange(seq_lengths.size(0), device=hidden_states.device),
                seq_lengths,
                output_size=total_tokens,
            )
            # Absolute ends minus absolute query positions: computed once for
            # all heads, so offsets cannot cross packed-example boundaries.
            remaining = cu_seqlens[1:][sequence_ids] - torch.arange(
                total_tokens, device=hidden_states.device
            )

        losses = []
        token_counts = []
        for offset, head in enumerate(self.lm_head_mtp, start=2):
            if remaining is None:
                length = max(next_targets.size(1) - offset + 1, 0)
                head_targets = next_targets[:, offset - 1:offset - 1 + length]
                valid = valid_output[:, :length] & (head_targets != IGNORE_INDEX)
                # Project the contiguous trunk once, then slice the small logits.
                # Slicing [B, S, D] first can copy the much larger trunk per head.
                head_logits = head(hidden_states)[:, :length]
            else:
                head_targets = torch.roll(next_targets, shifts=1 - offset)
                valid = valid_output & (remaining > offset) & (head_targets != IGNORE_INDEX)
                head_logits = head(hidden_states)
            head_targets = head_targets.masked_fill(~valid, IGNORE_INDEX)
            # Sum + a clamped count remains finite for short/empty outputs.
            # Under training autocast, cross_entropy accumulates in float32.
            raw = self._cross_entropy(
                head_logits.reshape(-1, self.config.vocab_size),
                head_targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            count = valid.sum()
            losses.append(raw / count.clamp(min=1))
            token_counts.append(count)
        head_losses = torch.stack(losses)
        if return_token_counts:
            return head_losses.mean(), head_losses, torch.stack(token_counts)
        return head_losses.mean(), head_losses

    def forward(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sep_indices: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        positions_3d: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        compute_training_metrics: bool = False,
        compute_mtp_loss: bool = True,
        compute_next_latent_loss: bool = True,
        dihedral_ids: Optional[torch.Tensor] = None,
        pair_identity_ids: Optional[torch.Tensor] = None,
        prefix_layout: Optional[dict] = None,
    ) -> dict:
        if input_ids.dim() == 1:
            return self._forward_varlen(
                input_ids=input_ids,
                example_ids=example_ids,
                dihedral_ids=dihedral_ids,
                sep_indices=sep_indices,
                targets=targets,
                positions_3d=positions_3d,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                compute_training_metrics=compute_training_metrics,
                compute_mtp_loss=compute_mtp_loss,
                compute_next_latent_loss=compute_next_latent_loss,
                pair_identity_ids=pair_identity_ids,
                prefix_layout=prefix_layout,
            )
        if input_ids.dim() == 2:
            return self._forward_padded(
                input_ids=input_ids,
                example_ids=example_ids,
                dihedral_ids=dihedral_ids,
                attention_mask=attention_mask,
                sep_indices=sep_indices,
                targets=targets,
                positions_3d=positions_3d,
                compute_training_metrics=compute_training_metrics,
                compute_mtp_loss=compute_mtp_loss,
                compute_next_latent_loss=compute_next_latent_loss,
                pair_identity_ids=pair_identity_ids,
            )
        raise ValueError("input_ids must have shape [B, S] or [total_tokens].")

    def _forward_padded(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        dihedral_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        sep_indices: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
        positions_3d: Optional[torch.Tensor],
        compute_training_metrics: bool = False,
        compute_mtp_loss: bool = True,
        compute_next_latent_loss: bool = True,
        pair_identity_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        batch_size, seq_len = input_ids.size()
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model capacity ({self.config.max_seq_len})."
            )

        device = input_ids.device
        if sep_indices is None:
            raise ValueError("sep_indices are required for padded forward inputs.")
        if sep_indices.device != device or sep_indices.dtype != torch.int32:
            sep_indices = sep_indices.to(device=device, dtype=torch.int32)
        if sep_indices.dim() != 1 or sep_indices.size(0) != batch_size:
            raise ValueError("sep_indices must have shape [batch_size].")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        else:
            if attention_mask.device != device or attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        if positions_3d is not None and positions_3d.shape[:2] != input_ids.shape:
            raise ValueError("positions_3d must match [batch, seq_len] of input_ids.")
        example_ids = self._normalize_example_ids(
            example_ids,
            batch_size=batch_size,
            device=device,
        )

        if targets is None:
            targets = input_ids

        pos_xyz = (
            self._compute_positions_3d(input_ids, attention_mask)
            if positions_3d is None
            else positions_3d.to(device=device, dtype=torch.int32)
        )
        if self.config.attention_pattern == "pair_group":
            target_z = pos_xyz[..., 2].gather(1, sep_indices.long()[:, None])
            pos_xyz = role_positions(pos_xyz, target_z)
        token_embeds = self.token_embedding(input_ids)
        token_embeds = self._add_pair_identities(token_embeds, pair_identity_ids)
        (
            hidden_states,
            token_input_residual,
            example_input_residual,
            dihedral_input_residual,
        ) = self._prepare_input_streams(
            token_embeds,
            example_embeds=self._lookup_example_embeds(example_ids),
            dihedral_embeds=self._lookup_dihedral_embeds(
                dihedral_ids, batch_size=batch_size, device=device,
            ),
        )
        hidden_states = self.embedding_dropout(hidden_states)

        prefix_lengths = sep_indices.clamp(min=0, max=seq_len - 1) + 1
        # The bidirectional prefix breaks the simple right-padded causal mask.
        prefix_lm_sdpa_mask = self._build_prefix_lm_sdpa_mask(
            attention_mask=attention_mask,
            prefix_lengths=prefix_lengths,
            dtype=hidden_states.dtype,
            input_ids=input_ids,
        )

        prelude_states = None
        exit_states = []
        for logical_idx, layer_idx in enumerate(self.layer_schedule):
            block = self.blocks[layer_idx]
            hidden_states, prelude_states = self._loop_input(hidden_states, prelude_states, logical_idx)
            block_input = hidden_states
            hidden_states = block(
                hidden_states,
                token_input_residual=token_input_residual,
                example_input_residual=example_input_residual,
                dihedral_input_residual=dihedral_input_residual,
                attention_mask=None,
                causal_mask=None,
                pos_xyz=pos_xyz,
                sdpa_mask=prefix_lm_sdpa_mask,
                is_causal=False,
                value_embed=self._lookup_value_embedding(layer_idx, example_ids),
                **self._loop_block_options(logical_idx),
            )
            hidden_states = self._loop_gate(block_input, hidden_states, logical_idx)
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            hidden_states, prelude_states = self._loop_output(hidden_states, prelude_states, logical_idx)
            if self._loop_exit_due(logical_idx):
                exit_states.append(hidden_states)

        exit_states = self._loop_exit_states(exit_states, lambda layer_idx, states: self.blocks[layer_idx](
            states, token_input_residual=token_input_residual, example_input_residual=example_input_residual,
            dihedral_input_residual=dihedral_input_residual, attention_mask=None, causal_mask=None,
            pos_xyz=pos_xyz, sdpa_mask=prefix_lm_sdpa_mask, is_causal=False,
            value_embed=self._lookup_value_embedding(layer_idx, example_ids),
        ) * attention_mask.unsqueeze(-1))
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        output_loss = None
        mtp_loss = None
        mtp_losses = None
        mtp_token_counts = None
        num_output_tokens = None
        training_metric_counts = None
        loop_exit_loss = None

        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets, valid_output = build_output_targets_and_mask(
                input_ids,
                sep_indices,
                attention_mask=attention_mask,
                targets=targets,
            )

            # 1. Calculate per-token loss (reduction='none')
            raw_losses = self._cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view(batch_size, -1)

            num_output_tokens = valid_output.sum()
            output_loss = (raw_losses * valid_output).sum() / num_output_tokens.clamp(
                min=1
            )
            loop_exit_loss = self._loop_exit_loss(
                exit_states, lambda states: self.lm_head(states)[:, :-1, :].reshape(-1, self.config.vocab_size),
                shift_targets.view(-1), valid_output.reshape(-1), num_output_tokens,
            )
            if compute_training_metrics:
                training_metric_counts = self._compute_training_metric_counts(
                    shift_logits, shift_targets, valid_output
                )

            if compute_mtp_loss:
                mtp_loss, mtp_losses, mtp_token_counts = self._compute_mtp_loss(
                    hidden_states, shift_targets, valid_output, return_token_counts=True,
                )

        loss = output_loss
        if mtp_loss is not None:
            loss = output_loss + self.config.mtp_weight * mtp_loss
        if loop_exit_loss is not None:
            loss = loss + self.config.loop_exit_weight * loop_exit_loss

        latent = self._next_latent_output(
            hidden_states, input_ids, valid_output,
            compute_next_latent_loss=compute_next_latent_loss,
        )
        if latent["next_latent_loss"] is not None:
            loss = loss + latent["next_latent_loss"]

        return {
            **latent,
            "logits": logits,
            "loss": loss,
            "output_loss": output_loss,
            "mtp_loss": mtp_loss,
            "mtp_losses": mtp_losses,
            "mtp_token_counts": mtp_token_counts,
            "num_output_tokens": num_output_tokens if targets is not None else None,
            "training_metric_counts": training_metric_counts,
            "loop_exit_loss": loop_exit_loss,
        }

    def _forward_varlen(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        dihedral_ids: Optional[torch.Tensor],
        sep_indices: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
        positions_3d: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
        compute_training_metrics: bool = False,
        compute_mtp_loss: bool = True,
        compute_next_latent_loss: bool = True,
        pair_identity_ids: Optional[torch.Tensor] = None,
        prefix_layout: Optional[dict] = None,
    ) -> dict:
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for packed varlen inputs.")
        if positions_3d is None:
            raise ValueError("positions_3d is required for packed varlen inputs.")

        device = input_ids.device
        total_tokens = int(input_ids.size(0))
        batch_size = int(example_ids.size(0))
        example_ids = self._normalize_example_ids(
            example_ids,
            batch_size=batch_size,
            device=device,
        )
        if positions_3d.shape != (total_tokens, 3):
            raise ValueError("positions_3d must match packed shape [total_tokens, 3].")

        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
        if cu_seqlens.dim() != 1 or cu_seqlens.size(0) != batch_size + 1:
            raise ValueError("cu_seqlens must have shape [batch_size + 1].")
        # Each check forces a GPU->CPU sync; training batches come from the collate function that builds
        # cu_seqlens, so validate only outside training.
        if not self.training:
            _compile_safe_assert(cu_seqlens[0] == 0, "cu_seqlens must start at 0.")
            _compile_safe_assert(
                cu_seqlens[-1] == total_tokens,
                "cu_seqlens[-1] must equal the packed token count in input_ids.",
            )

        seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.long)
        if not self.training:
            _compile_safe_assert(
                torch.all(seq_lengths > 0),
                "All sequences must have positive length in cu_seqlens.",
            )

        if max_seqlen is None:
            max_seqlen_value = seq_lengths.max()
            _compile_safe_assert(
                max_seqlen_value <= self.config.max_seq_len,
                f"Sequence length exceeds model capacity ({self.config.max_seq_len}).",
            )
            max_seqlen_resolved = (
                self.config.max_seq_len
                if _is_torch_compiling()
                else int(max_seqlen_value.item())
            )
        else:
            max_seqlen_resolved = int(max_seqlen)
            if max_seqlen_resolved > self.config.max_seq_len:
                raise ValueError(
                    f"Sequence length {max_seqlen_resolved} exceeds model capacity ({self.config.max_seq_len})."
                )

        if sep_indices is None:
            raise ValueError("sep_indices are required for packed forward inputs.")
        if sep_indices.device != device or sep_indices.dtype != torch.int32:
            sep_indices = sep_indices.to(device=device, dtype=torch.int32)
        if sep_indices.dim() != 1 or sep_indices.size(0) != batch_size:
            raise ValueError("sep_indices must have shape [batch_size].")

        if targets is not None:
            if targets.dim() != 1 or targets.size(0) != total_tokens:
                raise ValueError("targets must have shape [total_tokens] for varlen inputs.")
            if targets.device != device or targets.dtype != torch.long:
                targets = targets.to(device=device, dtype=torch.long)

        if not self._has_varlen_flash_attention():
            if self.config.FA == 4 and device.type == "cuda":
                raise RuntimeError(
                    "FA=4 requires flash_attn.cute.interface.flash_attn_varlen_func "
                    "(flash-attn-4 with nvidia-cutlass-dsl) or use FA=2."
                )
            if self.config.FA == 3 and device.type == "cuda":
                raise RuntimeError(
                    "FA=3 requires flash_attn_interface.flash_attn_varlen_func. "
                    "Install FlashAttention-3 from the hopper package or use FA=2."
                )
            return self._forward_varlen_via_padded(
                input_ids=input_ids,
                example_ids=example_ids,
                dihedral_ids=dihedral_ids,
                sep_indices=sep_indices,
                targets=targets,
                positions_3d=positions_3d.to(device=device, dtype=torch.int32),
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen_resolved,
                compute_training_metrics=compute_training_metrics,
                compute_mtp_loss=compute_mtp_loss,
                compute_next_latent_loss=compute_next_latent_loss,
                pair_identity_ids=pair_identity_ids,
            )

        pos_xyz = positions_3d.to(device=device, dtype=torch.int32)
        if self.config.attention_pattern == "pair_group":
            target_indices = cu_seqlens[:-1].long() + sep_indices.long()
            target_z = torch.repeat_interleave(pos_xyz[target_indices, 2], seq_lengths, output_size=total_tokens)
            pos_xyz = role_positions(pos_xyz, target_z)
        # One gather of the fp32 cos/sin tables serves every block application.
        rope_tables = self.blocks[0].attention.rope.tables(pos_xyz)
        token_embeds = self.token_embedding(input_ids)
        token_embeds = self._add_pair_identities(token_embeds, pair_identity_ids)
        example_sequence_embeds = torch.repeat_interleave(
            self._lookup_example_embeds(example_ids),
            seq_lengths,
            dim=0,
            output_size=total_tokens,
        )
        dihedral_sequence_embeds = self._lookup_dihedral_embeds(
            dihedral_ids, batch_size=batch_size, device=device,
        )
        if dihedral_sequence_embeds is not None:
            dihedral_sequence_embeds = torch.repeat_interleave(
                dihedral_sequence_embeds, seq_lengths, dim=0, output_size=total_tokens,
            )
        (
            hidden_states,
            token_input_residual,
            example_input_residual,
            dihedral_input_residual,
        ) = self._prepare_input_streams(
            token_embeds,
            example_embeds=example_sequence_embeds,
            dihedral_embeds=dihedral_sequence_embeds,
        )
        hidden_states = self.embedding_dropout(hidden_states)

        prefix_lengths = sep_indices.clamp(min=0) + 1
        prefix_lm_layout = self._build_packed_prefix_lm_layout(
            cu_seqlens=cu_seqlens,
            total_tokens=total_tokens,
            prefix_lengths=prefix_lengths,
            input_ids=input_ids,
            precomputed=prefix_layout,
        )

        prelude_states = None
        exit_states = []
        for logical_idx, layer_idx in enumerate(self.layer_schedule):
            block = self.blocks[layer_idx]
            hidden_states, prelude_states = self._loop_input(hidden_states, prelude_states, logical_idx)
            block_input = hidden_states
            hidden_states = block(
                hidden_states,
                token_input_residual=token_input_residual,
                example_input_residual=example_input_residual,
                dihedral_input_residual=dihedral_input_residual,
                attention_mask=None,
                causal_mask=None,
                pos_xyz=pos_xyz,
                sdpa_mask=None,
                is_causal=False,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen_resolved,
                prefix_lm_layout=prefix_lm_layout,
                rope_tables=rope_tables,
                value_embed=self._lookup_value_embedding(
                    layer_idx,
                    example_ids,
                    seq_lengths=seq_lengths,
                    total_tokens=total_tokens,
                ),
                **self._loop_block_options(logical_idx),
            )
            hidden_states = self._loop_gate(block_input, hidden_states, logical_idx)
            hidden_states, prelude_states = self._loop_output(hidden_states, prelude_states, logical_idx)
            if self._loop_exit_due(logical_idx):
                exit_states.append(hidden_states)

        exit_states = self._loop_exit_states(exit_states, lambda layer_idx, states: self.blocks[layer_idx](
            states, token_input_residual=token_input_residual, example_input_residual=example_input_residual,
            dihedral_input_residual=dihedral_input_residual, attention_mask=None, causal_mask=None,
            pos_xyz=pos_xyz, sdpa_mask=None, is_causal=False, cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen_resolved, prefix_lm_layout=prefix_lm_layout, rope_tables=rope_tables,
            value_embed=self._lookup_value_embedding(layer_idx, example_ids, seq_lengths=seq_lengths,
                                                     total_tokens=total_tokens),
        ))
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        targets, valid_output = build_output_targets_and_mask(
            input_ids,
            sep_indices,
            targets=targets,
            cu_seqlens=cu_seqlens,
        )
        raw_losses = self._cross_entropy(
            logits,
            targets,
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        num_output_tokens = valid_output.sum()

        output_loss = (raw_losses * valid_output).sum() / num_output_tokens.clamp(min=1)
        mtp_loss = None
        mtp_losses = None
        mtp_token_counts = None
        if compute_mtp_loss:
            mtp_loss, mtp_losses, mtp_token_counts = self._compute_mtp_loss(
                hidden_states, targets, valid_output, cu_seqlens=cu_seqlens, return_token_counts=True,
            )
        loss = output_loss
        if mtp_loss is not None:
            loss = output_loss + self.config.mtp_weight * mtp_loss
        loop_exit_loss = self._loop_exit_loss(exit_states, self.lm_head, targets, valid_output, num_output_tokens)
        if loop_exit_loss is not None:
            loss = loss + self.config.loop_exit_weight * loop_exit_loss
        latent_sources = None
        if self.next_latent_dynamics is not None and compute_next_latent_loss:
            latent_sources = (prefix_lm_layout.output_token_indices - 1
                              if prefix_lm_layout.output_token_indices is not None
                              else input_ids.new_empty(0))
        latent = self._next_latent_output(
            hidden_states, input_ids, valid_output, cu_seqlens=cu_seqlens,
            compute_next_latent_loss=compute_next_latent_loss,
            source_indices=latent_sources,
        )
        if latent["next_latent_loss"] is not None:
            loss = loss + latent["next_latent_loss"]
        training_metric_counts = None
        if compute_training_metrics:
            sequence_ids = torch.repeat_interleave(
                torch.arange(batch_size, device=device, dtype=torch.long),
                seq_lengths,
                output_size=total_tokens,
            )
            training_metric_counts = self._compute_training_metric_counts(
                logits, targets, valid_output, sequence_ids, batch_size
            )

        return {
            **latent,
            "logits": logits,
            "loss": loss,
            "output_loss": output_loss,
            "mtp_loss": mtp_loss,
            "mtp_losses": mtp_losses,
            "mtp_token_counts": mtp_token_counts,
            "num_output_tokens": num_output_tokens,
            "training_metric_counts": training_metric_counts,
            "loop_exit_loss": loop_exit_loss,
        }

    def predict_mtp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy offsets 2..K from one normalized anchor per batch row."""
        return torch.stack([head(hidden_states).argmax(dim=-1) for head in self.lm_head_mtp], dim=-1)

    def predict_next_latent(self, hidden_states: torch.Tensor, pending: torch.Tensor) -> torch.Tensor:
        """Draft future tokens recursively from a verified state and next token.

        The pending token is already selected by the main transformer head.
        Consume it in the dynamics before predicting the first proposal; each
        later transition consumes the preceding proposal. Only verification
        may commit these proposals to the output or transformer KV cache.
        """
        if self.next_latent_dynamics is None:
            raise ValueError("NextLat drafting requires a trained dynamics model.")
        proposals = []
        for _ in range(self.config.next_latent_horizon):
            hidden_states = self.next_latent_dynamics(hidden_states, self.token_embedding(pending))
            pending = self.lm_head(hidden_states).argmax(dim=-1)
            proposals.append(pending)
        return torch.stack(proposals, dim=-1)

    def forward_verify(self, **kwargs) -> dict:
        """Separate compile entry so block shapes don't consume decode's cache."""
        return self.forward_generate(return_hidden_states=True, **kwargs)

    def forward_generate(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        positions_3d: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        example_embeds: Optional[torch.Tensor] = None,
        conditioning_embeds: Optional[torch.Tensor] = None,
        value_embeds: Optional[Tuple[Optional[torch.Tensor], ...]] = None,
        decode_block_mask: Optional[object] = None,
        dihedral_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        demo_attention_mask: Optional[torch.Tensor] = None,
        pair_identity_ids: Optional[torch.Tensor] = None,
        split_cache: Optional[tuple] = None,
    ) -> dict:
        """Forward used for autoregressive generation with a KV cache.

        When `past_key_values` is None, the call is treated as a full prompt
        pass and the method returns per-layer key/value tensors that
        represent the entire prefix. When `past_key_values` is provided,
        `input_ids` and `positions_3d` should contain only the newly
        generated tokens, and the cache is updated accordingly. `attention_mask`
        can be provided to mask padded tokens during the initial prompt pass,
        and to mask cached keys during incremental decoding.
        `value_embeds` can be provided to reuse per-layer value embeddings
        across prompt and decode calls for the same generation batch.

        Split target/demo models can instead decode over `split_cache` =
        (layers, demo_lengths, target_lengths, kernel) from
        `target_demo_attention.split_prompt_cache`: per-layer (demo_k, demo_v,
        target_k, target_v) buffers and int32 [B] filled lengths. Each call
        writes its tokens at target_lengths[b] + i; the caller then advances
        target_lengths. No attention mask or cache position is used.
        """
        batch_size, seq_len = input_ids.size()
        decoding = past_key_values is not None or split_cache is not None
        # Cached decode/verification contains target tokens only. Its zero ID
        # is fixed; the demo identities are already stored in the KV cache.
        if self.config.pair_identity_vertices and pair_identity_ids is None and decoding:
            pair_identity_ids = torch.zeros_like(input_ids)
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model capacity ({self.config.max_seq_len})."
            )

        if positions_3d is not None and positions_3d.shape[:2] != input_ids.shape:
            raise ValueError("positions_3d must match [batch, seq_len] of input_ids.")

        device = input_ids.device
        example_ids = self._normalize_example_ids(
            example_ids,
            batch_size=batch_size,
            device=device,
        )
        if attention_mask is not None:
            if attention_mask.device != device or attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        if conditioning_embeds is not None and example_embeds is not None:
            raise ValueError(
                "Pass only one of conditioning_embeds or example_embeds."
            )

        if conditioning_embeds is not None and not self.config.task_embeddings:
            conditioning_embeds = self.compute_conditioning_embeds(example_ids, dihedral_ids)

        if not decoding and attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        if positions_3d is None:
            if decoding:
                raise ValueError("positions_3d must be provided when using past_key_values.")
            pos_xyz = self._compute_positions_3d(input_ids, attention_mask)
        else:
            pos_xyz = positions_3d.to(device=device, dtype=torch.int32)
        if self.config.attention_pattern == "pair_group":
            if not decoding:
                # Left-padded prompts end at the target input's separator.
                pos_xyz = role_positions(pos_xyz, pos_xyz[:, -1, 2:3])
            else:
                # Cached calls contain only target-output tokens, including
                # speculative verification blocks. Preserve spatial state.
                pos_xyz = torch.cat((pos_xyz[..., :2], torch.full_like(pos_xyz[..., 2:], 4)), dim=-1)
        token_embeds = self.token_embedding(input_ids)
        token_embeds = self._add_pair_identities(token_embeds, pair_identity_ids)
        dihedral_embeds = self._lookup_dihedral_embeds(
            dihedral_ids, batch_size=batch_size, device=device,
        )
        if conditioning_embeds is not None:
            if conditioning_embeds.shape[0] != input_ids.size(0):
                raise ValueError(
                    "conditioning_embeds must have batch dimension matching input_ids."
                )
            conditioning_embeds = conditioning_embeds.to(
                device=device, dtype=self.conditioning_dtype
            )
            if self.config.dihedral_embeddings:
                # Keep the historical separate shortcut streams even when the
                # sum used at the input is cached for the generation batch.
                example_input_residual, dihedral_input_residual = self._prepare_conditioning_streams(
                    token_embeds,
                    example_embeds=self._lookup_example_embeds(example_ids),
                    dihedral_embeds=dihedral_embeds,
                )
            else:
                example_input_residual = self._expand_conditioning_embeds(
                    token_embeds, conditioning_embeds,
                )
                dihedral_input_residual = None
            hidden_states = self._embed_inputs(
                token_embeds,
                conditioning_embeds=conditioning_embeds,
            )
            token_input_residual = token_embeds
        else:
            if example_embeds is None:
                example_embeds = self._lookup_example_embeds(example_ids)
            else:
                if example_embeds.shape[0] != input_ids.size(0):
                    raise ValueError(
                        "example_embeds must have batch dimension matching input_ids."
                    )
                example_embeds = example_embeds.to(
                    device=device, dtype=self.conditioning_dtype
                )
            (
                hidden_states,
                token_input_residual,
                example_input_residual,
                dihedral_input_residual,
            ) = self._prepare_input_streams(
                token_embeds,
                example_embeds=example_embeds,
                dihedral_embeds=dihedral_embeds,
            )
        # hidden_states = self.dropout(hidden_states)

        # Initial prompt: no cache yet, compute 3D positions and use the
        # input-side prompt as a bidirectional prefill.
        if not decoding:
            prefill_mask = None if bool(attention_mask.all()) else attention_mask
            prefill_layout = None
            prefill_bias = None
            if self.config.attention_pattern == "pair_group":
                if self._has_varlen_flash_attention():
                    valid = attention_mask.flatten().nonzero().flatten()
                    lengths = attention_mask.sum(-1).to(torch.int32)
                    cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(torch.int32)
                    prefill_layout = self._build_packed_prefix_lm_layout(
                        cu, valid.numel(), lengths, input_ids=input_ids.flatten()[valid],
                    )
                    prefill_layout.padded_token_indices = valid
                else:
                    prefill_bias = self._build_prefix_lm_sdpa_mask(
                        attention_mask, torch.full((batch_size,), seq_len, device=device),
                        hidden_states.dtype, input_ids=input_ids,
                    )
            past_key_values_out: List[Tuple[torch.Tensor, torch.Tensor]] = []
            prelude_states = None
            for logical_idx, layer_idx in enumerate(self.layer_schedule):
                block = self.blocks[layer_idx]
                hidden_states, prelude_states = self._loop_input(hidden_states, prelude_states, logical_idx)
                block_input = hidden_states
                hidden_states, present_kv = (
                    block.forward_with_cache(
                        hidden_states,
                        token_input_residual=token_input_residual,
                        example_input_residual=example_input_residual,
                        dihedral_input_residual=dihedral_input_residual,
                        pos_xyz=pos_xyz,
                        attention_mask=prefill_mask,
                        causal_mask=None,
                        past_key_value=None,
                        sdpa_mask=prefill_bias,
                        prefix_lm_layout=prefill_layout,
                        value_embed=self._value_embedding_for_layer(
                            layer_idx,
                            example_ids,
                            value_embeds,
                        ),
                        **self._loop_block_options(logical_idx),
                    )
                )
                hidden_states = self._loop_gate(block_input, hidden_states, logical_idx)
                hidden_states, prelude_states = self._loop_output(hidden_states, prelude_states, logical_idx)
                if present_kv is None:
                    raise RuntimeError("Prompt KV build requires present_key_value.")
                past_key_values_out.append(present_kv)

            hidden_states = self.norm(hidden_states)
            # Prefill only needs next-token logits; avoid projecting the full prompt.
            logits = self.lm_head(hidden_states[:, -1:, :])
            result = {"logits": logits, "past_key_values": tuple(past_key_values_out)}
            if return_hidden_states:
                result["hidden_states"] = hidden_states[:, -1:, :]
            return result

        if split_cache is not None:
            split_layers, demo_lengths, target_lengths, kernel = split_cache
            if len(split_layers) != self.cache_layer_count:
                raise ValueError("split_cache must have one cache per logical layer invocation.")
            if attention_mask is not None:
                raise ValueError("Split-cache decoding takes no attention_mask.")
        elif len(past_key_values) != self.cache_layer_count:
            raise ValueError("past_key_values must have one cache per logical layer invocation.")
        prelude_states = None
        for logical_idx, i in enumerate(self.layer_schedule):
            block = self.blocks[i]
            hidden_states, prelude_states = self._loop_input(hidden_states, prelude_states, logical_idx)
            block_input = hidden_states
            hidden_states, _present_kv = (
                block.forward_with_cache(
                    hidden_states,
                    token_input_residual=token_input_residual,
                    example_input_residual=example_input_residual,
                    dihedral_input_residual=dihedral_input_residual,
                    pos_xyz=pos_xyz,
                    attention_mask=attention_mask,
                    past_key_value=None if split_cache is not None else past_key_values[logical_idx],
                    cache_position=cache_position,
                    decode_block_mask=decode_block_mask,
                    demo_attention_mask=demo_attention_mask,
                    split_cache=None if split_cache is None else (
                        *split_layers[logical_idx], demo_lengths, target_lengths, kernel),
                    value_embed=self._value_embedding_for_layer(
                        i,
                        example_ids,
                        value_embeds,
                    ),
                    **self._loop_block_options(logical_idx),
                )
            )
            hidden_states = self._loop_gate(block_input, hidden_states, logical_idx)
            hidden_states, prelude_states = self._loop_output(hidden_states, prelude_states, logical_idx)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # The cache was updated in place; verification also needs its anchor.
        result = {"logits": logits}
        if return_hidden_states:
            result["hidden_states"] = hidden_states
        return result

    # ------------------------ 3D RoPE utilities ------------------------
    def _compute_positions_3d(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute 3D positions on CPU, then move them to the target device."""
        pos_cpu = compute_positions_3d(
            input_ids=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
        )
        return pos_cpu.to(device=input_ids.device, dtype=torch.int32)


class RotaryEmbedding3D(nn.Module):
    """3D grid RoPE with a configurable spatial base and fixed role base.

    Splits head channels across column, row and input/output role. The role
    frequencies use the R1 1k/1k base of 1000.
    """

    def __init__(self, head_dim: int, *, spatial_base: float = 1000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        self.head_dim = head_dim
        self.spatial_base = spatial_base

        # Distribute pairs across 3 axes as evenly as possible
        n_pairs = head_dim // 2
        px = n_pairs // 3
        py = n_pairs // 3
        pz = n_pairs - px - py
        self.d_x = px * 2
        self.d_y = py * 2
        self.d_z = pz * 2

        # Dense contexts use z=1..6; pair-group uses fixed roles z=1..4.
        self.max_x = 32
        self.max_y = 32
        self.max_z = 7

        # Precompute and register caches
        self._register_cache("x", self.d_x, self.max_x)
        self._register_cache("y", self.d_y, self.max_y)
        self._register_cache("z", self.d_z, self.max_z)

    def _build_inv_freq(self, dim: int, base: float) -> torch.Tensor:
        if dim <= 0:
            return torch.empty(0)
        return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    def _register_cache(self, name: str, dim: int, max_pos: int) -> None:
        if dim <= 0:
            # Empty buffers for unused dimensions (keeps state_dict clean)
            self.register_buffer(f"cos_{name}_cache", torch.empty(0), persistent=False)
            self.register_buffer(f"sin_{name}_cache", torch.empty(0), persistent=False)
            return

        inv_freq = self._build_inv_freq(dim, self.spatial_base if name in ("x", "y") else 1000.0)
        # Generate all possible positions [0, 1, ... max_pos-1]
        pos = torch.arange(max_pos).float()

        # Outer product: [max_pos, dim/2]
        t = pos.unsqueeze(-1) * inv_freq

        # Compute cos/sin and repeat_interleave to match [max_pos, dim]
        # We do the interleave here once, so we don't do it at every forward step.
        cos = torch.cos(t).repeat_interleave(2, dim=-1)
        sin = torch.sin(t).repeat_interleave(2, dim=-1)

        self.register_buffer(f"cos_{name}_cache", cos, persistent=False)
        self.register_buffer(f"sin_{name}_cache", sin, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # pairwise rotate: (x0,x1,x2,x3,...) -> (-x1, x0, -x3, x2, ...)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.stack((-x2, x1), dim=-1)
        return out.flatten(-2)

    def tables(self, pos_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather fp32 cos/sin tables for integer positions [..., 3] -> two [..., D] tensors.

        The tables depend only on positions, so a forward pass gathers them once
        and every block application (including repeated loop blocks) reuses them.
        """
        # Ensure indices are within bounds (clamping protects against out-of-bounds crash)
        # pos_xyz is expected to be an integer tensor suitable for indexing.
        pos_x = pos_xyz[..., 0].clamp(0, self.max_x - 1)
        pos_y = pos_xyz[..., 1].clamp(0, self.max_y - 1)
        pos_z = pos_xyz[..., 2].clamp(0, self.max_z - 1)

        parts_cos = []
        parts_sin = []

        # Gather cached Cos/Sin tables based on positions.
        # Note: self.cos_x_cache is [MaxPos, dx] -> Gather creates [B, S, dx]
        if self.d_x > 0:
            parts_cos.append(self.cos_x_cache[pos_x])
            parts_sin.append(self.sin_x_cache[pos_x])

        if self.d_y > 0:
            parts_cos.append(self.cos_y_cache[pos_y])
            parts_sin.append(self.sin_y_cache[pos_y])

        if self.d_z > 0:
            parts_cos.append(self.cos_z_cache[pos_z])
            parts_sin.append(self.sin_z_cache[pos_z])

        # Concatenate tables to form the full [..., D] embedding mask.
        # This 'cat' is on non-gradient tensors, which is cheap in the backward pass.
        return torch.cat(parts_cos, dim=-1), torch.cat(parts_sin, dim=-1)

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_xyz: Optional[torch.Tensor] = None,
        tables: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D RoPE to the first d_x, d_y, d_z channels respectively.

        Default layout: q, k [B, H, S, D] with pos_xyz [B, S, 3] integer (x, y, z).
        With ``tables`` (cos, sin already broadcastable to q and k) no gather runs
        and any layout works; packed training passes [T, 1, D] tables for
        [T, H, D] queries and keys. Both routes compute identical values.
        """
        if tables is None:
            if pos_xyz is None:
                raise ValueError("apply_rotary needs pos_xyz or precomputed tables.")
            cos, sin = self.tables(pos_xyz)
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # [B, 1, S, D]
        else:
            cos, sin = tables

        # Apply standard RoPE arithmetic on the full tensors.
        # This keeps q and k contiguous and avoids slicing activations.
        q_out = (q * cos) + (self._rotate_half(q) * sin)
        k_out = (k * cos) + (self._rotate_half(k) * sin)

        return q_out, k_out
