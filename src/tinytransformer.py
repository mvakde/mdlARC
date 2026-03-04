from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError:  # pragma: no cover - flash-attn optional at import time
    flash_attn_varlen_qkvpacked_func = None

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # pragma: no cover - flex-attention optional at import time
    create_block_mask = None
    flex_attention = None

from common import (
    IGNORE_INDEX,
    IO_SEPARATOR_TOKEN_ID,
    MAX_SEQ_LEN,
    VOCAB_SIZE,
)


@dataclass
class TinyTransformerConfig:
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    dropout: float = 0.1
    attention_dropout: Optional[float] = None
    num_examples: int = 1280
    num_dihedrals: int = 8

    def __post_init__(self) -> None:
        if self.attention_dropout is None:
            # Backwards compatibility: old configs used a single dropout knob.
            self.attention_dropout = self.dropout
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        if self.num_examples < 1:
            raise ValueError("num_examples must be >= 1.")
        if self.num_dihedrals != 8:
            raise ValueError("num_dihedrals must be exactly 8.")


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
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim**-0.5
        self._has_flash_attn_varlen = flash_attn_varlen_qkvpacked_func is not None
        self._has_flex_attention = flex_attention is not None

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        # Attention dropout is intentionally separate from hidden dropout.
        self.dropout_p = float(config.attention_dropout)

        # 1D RoPE setup (sequence index based)
        self.rope = RotaryEmbedding1D(self.head_dim, config.max_seq_len)

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
            del h, q_idx
            is_valid = mask[b, kv_idx]
            return torch.where(is_valid, score, score.new_full((), float("-inf")))

        return flex_attention(
            queries,
            keys,
            values,
            score_mod=decode_score_mod,
            scale=self.scale,
        )

    def _apply_varlen_flash_attention(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        if not self._has_flash_attn_varlen:
            raise RuntimeError(
                "flash-attn varlen attention requested, but flash-attn is not installed."
            )
        if qkv.device.type != "cuda":
            raise RuntimeError("flash-attn varlen kernels require CUDA tensors.")
        if qkv.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "flash-attn varlen expects fp16/bf16 qkv tensors. "
                "Run this path under CUDA autocast."
            )
        if cu_seqlens.device != qkv.device or cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(device=qkv.device, dtype=torch.int32)
        return flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            int(max_seqlen),
            dropout_p=dropout_p,
            softmax_scale=None,
            causal=is_causal,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        sdpa_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
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

            if positions is not None:
                if positions.dim() == 2 and positions.size(0) == 1:
                    positions = positions.squeeze(0)
                if positions.dim() != 1 or positions.size(0) != total_tokens:
                    raise ValueError(
                        "Packed positions must have shape [total_tokens]."
                    )
                q_f32 = queries.transpose(0, 1).unsqueeze(0).float()
                k_f32 = keys.transpose(0, 1).unsqueeze(0).float()
                q_f32, k_f32 = self.rope.apply_rotary(
                    q_f32, k_f32, positions.unsqueeze(0)
                )
                queries = q_f32.squeeze(0).transpose(0, 1).to(dtype=qkv_dtype)
                keys = k_f32.squeeze(0).transpose(0, 1).to(dtype=qkv_dtype)

            qkv_packed = torch.stack((queries, keys, values), dim=1).contiguous()
            attn_output = self._apply_varlen_flash_attention(
                qkv=qkv_packed,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
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

        if positions is not None:
            # Match cache path semantics: rotate in fp32 for stability, then
            # cast back so q/k/v dtypes stay aligned for SDPA.
            q_f32 = queries.float()
            k_f32 = keys.float()
            q_f32, k_f32 = self.rope.apply_rotary(q_f32, k_f32, positions)
            queries = q_f32.to(dtype=qkv_dtype)
            keys = k_f32.to(dtype=qkv_dtype)

        attn_output = self._apply_attention(
            queries,
            keys,
            values,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            sdpa_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        )
        return self.out_proj(attn_output)

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        decode_block_mask: Optional[object] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)
        qkv_dtype = queries.dtype

        if positions is not None:
            # Cast to fp32 for rotation precision
            q_f32 = queries.float()
            k_f32 = keys.float()
            q_f32, k_f32 = self.rope.apply_rotary(q_f32, k_f32, positions)
            # Cast back to original dtype (e.g. bfloat16) for storage
            queries = q_f32.to(dtype=qkv_dtype)
            keys = k_f32.to(dtype=qkv_dtype)

        # ------------------------------------------------------------------
        # PATH A: DECODE (We have a cache buffer)
        # ------------------------------------------------------------------
        if past_key_value is not None:
            past_keys, past_values = past_key_value
            key_layer = past_keys
            value_layer = past_values
            if cache_position is not None:
                # We update the buffer directly using index_copy_ (in-place).
                # keys is [B, H, 1, D], past_keys is [B, H, MaxLen, D]
                # cache_position is Tensor([step])
                past_keys.index_copy_(2, cache_position, keys)
                past_values.index_copy_(2, cache_position, values)

                # Use the FULL buffer for attention (masking handles the future tokens)
                key_layer = past_keys
                value_layer = past_values

            use_flex_decode = (
                self._has_flex_attention
                and queries.device.type == "cuda"
                and attention_mask is not None
            )
            if use_flex_decode:
                attn_output = self._apply_flex_decode_attention(
                    queries,
                    key_layer,
                    value_layer,
                    attention_mask=attention_mask,
                    decode_block_mask=decode_block_mask,
                )
            else:
                attn_output = self._apply_attention(
                    queries,
                    key_layer,
                    value_layer,
                    attention_mask=attention_mask,
                    causal_mask=None,
                    sdpa_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )

            attn_output = (
                attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
            )

            # --- Return ONLY the output. Cache is already updated. ---
            return self.out_proj(attn_output)

        # ------------------------------------------------------------------
        # PATH B: PROMPT (We are initializing)
        # ------------------------------------------------------------------
        attn_output = self._apply_attention(
            queries,
            keys,
            values,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            sdpa_mask=None,
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


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(config)
        self.ln_2 = RMSNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        sdpa_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        attn_input = self.ln_1(hidden_states)
        attn_output = self.attention(
            attn_input,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            positions=positions,
            sdpa_mask=sdpa_mask,
            is_causal=is_causal,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = hidden_states + attn_output

        ff_input = self.ln_2(hidden_states)
        ff_output = self.ff(ff_input)
        hidden_states = hidden_states + ff_output
        return hidden_states

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        decode_block_mask: Optional[object] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_input = self.ln_1(hidden_states)
        # Check if we are in Decode mode or Prompt mode
        if past_key_value is not None:
            # --- DECODE MODE ---
            # Attention returns ONLY tensor
            attn_output = self.attention.forward_with_cache(
                attn_input,
                positions=positions,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                decode_block_mask=decode_block_mask,
            )
            present_key_value = None  # No return value needed
        else:
            # --- PROMPT MODE ---
            # Attention returns Tensor + KV Tuple
            attn_output, present_key_value = self.attention.forward_with_cache(
                attn_input,
                positions=positions,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=None,
                cache_position=None,
            )
        hidden_states = hidden_states + attn_output

        ff_input = self.ln_2(hidden_states)
        ff_output = self.ff(ff_input)
        hidden_states = hidden_states + ff_output

        if attention_mask is not None:
            seq_len = hidden_states.size(1)
            if cache_position is not None:
                # Build positions for the currently generated tokens so we mask the right slot
                positions = cache_position.view(1, -1) + torch.arange(
                    seq_len, device=cache_position.device
                ).view(1, -1)
                positions = positions.clamp(max=attention_mask.size(1) - 1)
                positions = positions.expand(attention_mask.size(0), -1)
            else:
                positions = torch.arange(
                    attention_mask.size(1) - seq_len,
                    attention_mask.size(1),
                    device=hidden_states.device,
                ).view(1, -1)
                positions = positions.expand(attention_mask.size(0), -1)

            token_mask = attention_mask.gather(1, positions)
            hidden_states = hidden_states * token_mask.unsqueeze(-1)
        if past_key_value is not None:
            return hidden_states
        else:
            return hidden_states, present_key_value


class TinyTransformer(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.example_embedding = nn.Embedding(config.num_examples, config.d_model)
        self.dihedral_embedding = nn.Embedding(config.num_dihedrals, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Decode block-mask cache keyed by (device_type, device_index, kv_len, block_idx).
        self._decode_block_mask_cache: Dict[Tuple[str, int, int, int], object] = {}
        self._decode_block_size = 128

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

    def _build_positions(
        self,
        seq_len: int,
        device: torch.device,
        batch_size: int,
        start_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        if start_position is not None:
            if not torch.is_tensor(start_position):
                start_position = torch.tensor(
                    start_position, device=device, dtype=torch.long
                )
            else:
                start_position = start_position.to(device=device, dtype=torch.long)
            if start_position.dim() == 0:
                start_position = start_position.view(1, 1)
            elif start_position.dim() == 1:
                start_position = start_position.view(-1, 1)
            else:
                raise ValueError("start_position must be a scalar or 1D tensor.")
            positions = positions.view(1, -1) + start_position
            if positions.size(0) == 1 and batch_size > 1:
                positions = positions.expand(batch_size, -1)
        else:
            positions = positions.view(1, -1)
            if batch_size > 1:
                positions = positions.expand(batch_size, -1)
        return positions

    def _prepare_positions(
        self,
        positions: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device,
        start_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if positions is None:
            return self._build_positions(
                seq_len=seq_len,
                device=device,
                batch_size=batch_size,
                start_position=start_position,
            )

        if positions.device != device or positions.dtype != torch.long:
            positions = positions.to(device=device, dtype=torch.long)

        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        if positions.dim() != 2 or positions.size(1) != seq_len:
            raise ValueError("positions must have shape [batch, seq_len] or [seq_len].")
        if positions.size(0) == 1 and batch_size > 1:
            positions = positions.expand(batch_size, -1)
        if positions.size(0) != batch_size:
            raise ValueError("positions batch size must match input_ids batch size.")
        return positions

    @staticmethod
    def _build_varlen_positions(
        cu_seqlens: torch.Tensor,
        total_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.long)
        starts = cu_seqlens[:-1].to(dtype=torch.long)
        return torch.arange(total_tokens, device=device, dtype=torch.long) - torch.repeat_interleave(
            starts,
            seq_lengths,
        )

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

    def forward(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        dihedral_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sep_indices: Optional[torch.Tensor] = None,
        compute_input_loss: bool = True,
        targets: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> dict:
        if input_ids.dim() == 1:
            return self._forward_varlen(
                input_ids=input_ids,
                example_ids=example_ids,
                dihedral_ids=dihedral_ids,
                sep_indices=sep_indices,
                compute_input_loss=compute_input_loss,
                targets=targets,
                positions=positions,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        if input_ids.dim() == 2:
            return self._forward_padded(
                input_ids=input_ids,
                example_ids=example_ids,
                dihedral_ids=dihedral_ids,
                attention_mask=attention_mask,
                sep_indices=sep_indices,
                compute_input_loss=compute_input_loss,
                targets=targets,
                positions=positions,
            )
        raise ValueError("input_ids must have shape [B, S] or [total_tokens].")

    def _forward_padded(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        dihedral_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        sep_indices: Optional[torch.Tensor],
        compute_input_loss: bool,
        targets: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
    ) -> dict:
        batch_size, seq_len = input_ids.size()
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model capacity ({self.config.max_seq_len})."
            )

        device = input_ids.device
        if sep_indices is not None:
            if sep_indices.device != device or sep_indices.dtype != torch.long:
                sep_indices = sep_indices.to(device=device, dtype=torch.long)
            if sep_indices.dim() != 1 or sep_indices.size(0) != batch_size:
                raise ValueError("sep_indices must have shape [batch_size].")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        else:
            if attention_mask.device != device or attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        if positions is not None and positions.shape[-1] != input_ids.shape[1]:
            raise ValueError("positions must match seq_len of input_ids.")
        if example_ids.device != device or example_ids.dtype != torch.long:
            example_ids = example_ids.to(device=device, dtype=torch.long)
        if example_ids.dim() != 1 or example_ids.size(0) != batch_size:
            raise ValueError("example_ids must have shape [batch_size].")
        if dihedral_ids.device != device or dihedral_ids.dtype != torch.long:
            dihedral_ids = dihedral_ids.to(device=device, dtype=torch.long)
        if dihedral_ids.dim() != 1 or dihedral_ids.size(0) != batch_size:
            raise ValueError("dihedral_ids must have shape [batch_size].")

        if targets is None:
            targets = input_ids

        token_embeds = self.token_embedding(input_ids)
        example_embeds = self.example_embedding(example_ids)  # [B, D]
        dihedral_embeds = self.dihedral_embedding(dihedral_ids)  # [B, D]
        # Add the per-example embedding to every token in the sequence.
        hidden_states = (
            token_embeds + example_embeds.unsqueeze(1) + dihedral_embeds.unsqueeze(1)
        )
        hidden_states = self.dropout(hidden_states)

        # Compute or reuse 1D positions (sequence indices).
        pos_1d = self._prepare_positions(
            positions=positions,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

        # Training batches are right-padded; causal attention is sufficient to
        # prevent valid tokens from attending to padded keys. Keeping attn_mask
        # unset allows SDPA to dispatch flash kernels.
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                causal_mask=None,
                positions=pos_1d,
                sdpa_mask=None,
                is_causal=True,
            )
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        input_loss = None
        output_loss = None
        num_output_tokens = None

        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            shift_targets = shift_targets.masked_fill(~shift_mask, IGNORE_INDEX)

            # 1. Calculate per-token loss (reduction='none')
            raw_losses = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view(batch_size, -1)

            # 2. Identify which tokens are valid (not ignored)
            valid_mask = shift_targets != IGNORE_INDEX
            total_valid = valid_mask.sum()

            # 3. Standard total loss (for backprop)
            # Use clamp(min=1) to avoid division by zero if a batch is entirely padding
            loss = raw_losses.sum() / total_valid.clamp(min=1)

            # 4. Separate Input vs Output portions
            # The output region starts where the current input token is the separator.
            if sep_indices is not None:
                shift_len = shift_targets.size(1)
                sep_positions = sep_indices.clamp(min=0, max=shift_len).unsqueeze(1)
                positions = torch.arange(shift_len, device=device).unsqueeze(0)
                is_output_phase = positions >= sep_positions
            else:
                shift_input_ids = input_ids[:, :-1]
                # Fallback for callsites that do not provide sep_indices.
                is_output_phase = (shift_input_ids == IO_SEPARATOR_TOKEN_ID).cumsum(
                    dim=1
                ) >= 1
            # Calculate specific losses
            valid_output = valid_mask & is_output_phase

            num_output_tokens = valid_output.sum()

            if compute_input_loss:
                valid_input = valid_mask & (~is_output_phase)
                input_loss = (raw_losses * valid_input).sum() / valid_input.sum().clamp(
                    min=1
                )
            output_loss = (raw_losses * valid_output).sum() / num_output_tokens.clamp(
                min=1
            )

        return {
            "logits": logits,
            "loss": loss,
            "input_loss": input_loss,
            "output_loss": output_loss,
            "num_output_tokens": num_output_tokens if targets is not None else None,
        }

    def _forward_varlen(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        dihedral_ids: torch.Tensor,
        sep_indices: Optional[torch.Tensor],
        compute_input_loss: bool,
        targets: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
    ) -> dict:
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for packed varlen inputs.")
        if example_ids.dim() != 1:
            raise ValueError("example_ids must have shape [batch_size] for varlen inputs.")
        if dihedral_ids.dim() != 1:
            raise ValueError("dihedral_ids must have shape [batch_size] for varlen inputs.")

        device = input_ids.device
        total_tokens = int(input_ids.size(0))
        batch_size = int(example_ids.size(0))
        if int(dihedral_ids.size(0)) != batch_size:
            raise ValueError("dihedral_ids must match example_ids shape [batch_size].")

        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
        if cu_seqlens.dim() != 1 or cu_seqlens.size(0) != batch_size + 1:
            raise ValueError("cu_seqlens must have shape [batch_size + 1].")
        if int(cu_seqlens[0].item()) != 0:
            raise ValueError("cu_seqlens must start at 0.")
        if int(cu_seqlens[-1].item()) != total_tokens:
            raise ValueError(
                "cu_seqlens[-1] must equal the packed token count in input_ids."
            )

        seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.long)
        if torch.any(seq_lengths <= 0):
            raise ValueError("All sequences must have positive length in cu_seqlens.")

        max_seqlen_resolved = (
            int(seq_lengths.max().item()) if max_seqlen is None else int(max_seqlen)
        )
        if max_seqlen_resolved > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {max_seqlen_resolved} exceeds model capacity ({self.config.max_seq_len})."
            )

        if sep_indices is not None:
            if sep_indices.device != device or sep_indices.dtype != torch.long:
                sep_indices = sep_indices.to(device=device, dtype=torch.long)
            if sep_indices.dim() != 1 or sep_indices.size(0) != batch_size:
                raise ValueError("sep_indices must have shape [batch_size].")

        if targets is not None:
            if targets.dim() != 1 or targets.size(0) != total_tokens:
                raise ValueError("targets must have shape [total_tokens] for varlen inputs.")
            if targets.device != device or targets.dtype != torch.long:
                targets = targets.to(device=device, dtype=torch.long)
        else:
            targets = torch.roll(input_ids, shifts=-1).clone()
            sequence_ends = cu_seqlens[1:].to(dtype=torch.long) - 1
            targets[sequence_ends] = IGNORE_INDEX

        example_ids = example_ids.to(device=device, dtype=torch.long)
        dihedral_ids = dihedral_ids.to(device=device, dtype=torch.long)
        token_embeds = self.token_embedding(input_ids)

        # Build per-sequence conditioning once, then broadcast to packed tokens.
        sequence_embeds = self.example_embedding(example_ids) + self.dihedral_embedding(
            dihedral_ids
        )
        token_sequence_embeds = torch.repeat_interleave(
            sequence_embeds,
            seq_lengths,
            dim=0,
        )
        hidden_states = token_embeds + token_sequence_embeds
        hidden_states = self.dropout(hidden_states)

        seq_local_positions = self._build_varlen_positions(
            cu_seqlens=cu_seqlens,
            total_tokens=total_tokens,
            device=device,
        )
        if positions is None:
            rope_positions = seq_local_positions
        else:
            if positions.device != device or positions.dtype != torch.long:
                positions = positions.to(device=device, dtype=torch.long)
            if positions.dim() == 2 and positions.size(0) == 1:
                positions = positions.squeeze(0)
            if positions.dim() != 1 or positions.size(0) != total_tokens:
                raise ValueError("positions must have shape [total_tokens] for varlen inputs.")
            rope_positions = positions

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                causal_mask=None,
                positions=rope_positions,
                sdpa_mask=None,
                is_causal=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen_resolved,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        raw_losses = F.cross_entropy(
            logits,
            targets,
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        valid_mask = targets != IGNORE_INDEX
        total_valid = valid_mask.sum()
        loss = raw_losses.sum() / total_valid.clamp(min=1)

        seq_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=torch.long),
            seq_lengths,
        )

        if sep_indices is not None:
            is_output_phase = seq_local_positions >= sep_indices[seq_ids]
        else:
            is_output_phase = torch.zeros_like(valid_mask)
            is_sep = input_ids == IO_SEPARATOR_TOKEN_ID
            for seq_idx in range(batch_size):
                start = int(cu_seqlens[seq_idx].item())
                end = int(cu_seqlens[seq_idx + 1].item())
                is_output_phase[start:end] = is_sep[start:end].cumsum(dim=0) >= 1

        valid_output = valid_mask & is_output_phase
        num_output_tokens = valid_output.sum()

        input_loss = None
        if compute_input_loss:
            valid_input = valid_mask & (~is_output_phase)
            input_loss = (raw_losses * valid_input).sum() / valid_input.sum().clamp(
                min=1
            )
        output_loss = (raw_losses * valid_output).sum() / num_output_tokens.clamp(min=1)

        return {
            "logits": logits,
            "loss": loss,
            "input_loss": input_loss,
            "output_loss": output_loss,
            "num_output_tokens": num_output_tokens,
        }

    def forward_generate(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        dihedral_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        example_embeds: Optional[torch.Tensor] = None,
        decode_block_mask: Optional[object] = None,
    ) -> dict:
        """Forward used for autoregressive generation with a KV cache.

        When `past_key_values` is None, the call is treated as a full prompt
        pass and the method returns per-layer key/value tensors that
        represent the entire prefix. When `past_key_values` is provided,
        `input_ids` and `positions` should contain only the newly
        generated tokens, and the cache is updated accordingly. `attention_mask`
        can be provided to mask padded tokens during the initial prompt pass,
        and to mask cached keys during incremental decoding.
        """
        batch_size, seq_len = input_ids.size()
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model capacity ({self.config.max_seq_len})."
            )

        if positions is not None and positions.shape[-1] != input_ids.shape[1]:
            raise ValueError("positions must match seq_len of input_ids.")

        device = input_ids.device
        example_ids = example_ids.to(device=device, dtype=torch.long)
        dihedral_ids = dihedral_ids.to(device=device, dtype=torch.long)
        if example_ids.dim() != 1 or example_ids.size(0) != batch_size:
            raise ValueError("example_ids must have shape [batch_size].")
        if dihedral_ids.dim() != 1 or dihedral_ids.size(0) != batch_size:
            raise ValueError("dihedral_ids must have shape [batch_size].")
        if attention_mask is not None:
            if attention_mask.device != device or attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        if example_embeds is not None:
            if example_embeds.shape[0] != input_ids.size(0):
                raise ValueError(
                    "example_embeds must have batch dimension matching input_ids."
                )
        else:
            example_embeds = self.example_embedding(example_ids)
        dihedral_embeds = self.dihedral_embedding(dihedral_ids)

        token_embeds = self.token_embedding(input_ids)

        # During generation, also broadcast the example embedding across
        # all tokens in the (prompt or incremental) sequence.
        hidden_states = (
            token_embeds + example_embeds.unsqueeze(1) + dihedral_embeds.unsqueeze(1)
        )
        # hidden_states = self.dropout(hidden_states)

        pos_1d = positions.to(device=device, dtype=torch.long) if positions is not None else None

        # Initial prompt: no cache yet, compute 1D positions and use the
        # exact same masking behavior as the standard forward pass.
        if past_key_values is None:
            if attention_mask is None:
                attention_mask = torch.ones_like(
                    input_ids, dtype=torch.bool, device=device
                )

            pos_1d = self._prepare_positions(
                positions=pos_1d,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
            )
            causal_mask = self._build_causal_mask(seq_len, device)

            past_key_values_out: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for block in self.blocks:
                hidden_states, present_kv = block.forward_with_cache(
                    hidden_states,
                    positions=pos_1d,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    past_key_value=None,
                )
                past_key_values_out.append(present_kv)

            hidden_states = self.norm(hidden_states)
            # Prefill only needs next-token logits; avoid projecting the full prompt.
            logits = self.lm_head(hidden_states[:, -1:, :])
            return {"logits": logits, "past_key_values": tuple(past_key_values_out)}

        if pos_1d is None and cache_position is None:
            raise ValueError(
                "cache_position must be provided when using past_key_values without explicit positions."
            )

        pos_1d = self._prepare_positions(
            positions=pos_1d,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            start_position=cache_position,
        )
        # We iterate by index to access the specific layer's static buffer in past_key_values
        for i, block in enumerate(self.blocks):
            # Note:
            # 1. We pass past_key_values[i] (the static buffer).
            # 2. We do NOT capture a second return value (present_kv) because
            #    TransformerBlock.forward_with_cache no longer returns it in this mode.
            hidden_states = block.forward_with_cache(
                hidden_states,
                positions=pos_1d,
                attention_mask=attention_mask,
                past_key_value=past_key_values[i],
                cache_position=cache_position,
                decode_block_mask=decode_block_mask,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Return ONLY logits.
        # The cache update is done, and we don't need to pass it back.
        return {"logits": logits}


class RotaryEmbedding1D(nn.Module):
    """1D Rotary Positional Embedding applied to Q/K using precomputed lookups."""

    def __init__(
        self, head_dim: int, max_position: int, base: float = 10000.0
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        if max_position <= 0:
            raise ValueError("max_position must be positive.")
        self.head_dim = head_dim
        self.base = base
        self.max_position = max_position

        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(self.max_position).float()
        t = pos.unsqueeze(-1) * inv_freq

        cos = torch.cos(t).repeat_interleave(2, dim=-1)
        sin = torch.sin(t).repeat_interleave(2, dim=-1)

        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.stack((-x2, x1), dim=-1)
        return out.flatten(-2)

    def apply_rotary(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 1D RoPE using sequence indices.

        q, k: [B, H, S, D]
        positions: [B, S] or [S]
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        if positions.dim() != 2:
            raise ValueError("positions must have shape [batch, seq_len] or [seq_len].")

        pos = positions.clamp(0, self.max_position - 1)
        cos = self.cos_cache[pos].to(dtype=q.dtype).unsqueeze(1)
        sin = self.sin_cache[pos].to(dtype=q.dtype).unsqueeze(1)

        q_out = (q * cos) + (self._rotate_half(q) * sin)
        k_out = (k * cos) + (self._rotate_half(k) * sin)

        return q_out, k_out
