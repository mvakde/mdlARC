from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.attention.flex_attention import (
        and_masks,
        create_block_mask,
        flex_attention,
    )
except ImportError:  # pragma: no cover - older torch versions
    and_masks = None
    create_block_mask = None
    flex_attention = None

from common import (
    IGNORE_INDEX,
    IO_SEPARATOR_TOKEN_ID,
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    compute_positions_3d,
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
        self._has_flex_attention = (
            flex_attention is not None and create_block_mask is not None
        )

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        # Attention dropout is intentionally separate from hidden dropout.
        self.dropout_p = float(config.attention_dropout)

        # 3D RoPE setup
        self.rope = RotaryEmbedding3D(self.head_dim)

    def can_use_flex_attention(self, device: torch.device, dropout_p: float) -> bool:
        del device, dropout_p
        # Force SDPA (flash kernels) across training/inference paths.
        return False

    def _build_flex_mask_mod(
        self,
        attention_mask: Optional[torch.Tensor],
        use_causal: bool,
    ) -> Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ]:
        if attention_mask is None and not use_causal:
            return None

        def causal_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            return q_idx >= kv_idx

        if attention_mask is None:
            return causal_mod

        def padding_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del h, q_idx
            return attention_mask[b, kv_idx]

        if use_causal:
            if and_masks is not None:
                return and_masks(causal_mod, padding_mod)

            def causal_and_padding_mod(
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ) -> torch.Tensor:
                del h
                return (q_idx >= kv_idx) & attention_mask[b, kv_idx]

            return causal_and_padding_mod

        return padding_mod

    def _build_flex_score_mod(
        self,
        attention_mask: Optional[torch.Tensor],
        use_causal: bool,
        neg_inf: torch.Tensor,
        key_len: int,
    ) -> Optional[
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ]
    ]:
        if attention_mask is not None and attention_mask.size(1) < key_len:
            raise ValueError(
                "attention_mask length must cover all key/value positions."
            )

        if attention_mask is None and not use_causal:
            return None

        if attention_mask is None:

            def score_mod(
                score: torch.Tensor,
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ) -> torch.Tensor:
                del b, h
                return torch.where(q_idx >= kv_idx, score, neg_inf)

            return score_mod

        if use_causal:

            def score_mod(
                score: torch.Tensor,
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ) -> torch.Tensor:
                del h
                allowed = (q_idx >= kv_idx) & attention_mask[b, kv_idx]
                return torch.where(allowed, score, neg_inf)

            return score_mod

        def score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del h, q_idx
            return torch.where(attention_mask[b, kv_idx], score, neg_inf)

        return score_mod

    def build_flex_block_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        use_causal: bool,
        query_len: int,
        key_len: int,
        device: torch.device,
        dropout_p: float,
    ) -> Optional[object]:
        if not self.can_use_flex_attention(device, dropout_p):
            return None
        if torch.compiler.is_compiling():
            # Build-time block-mask creation can trigger graph breaks under
            # fullgraph compile; use score_mod inside flex_attention instead.
            return None

        if attention_mask is not None and attention_mask.size(1) < key_len:
            raise ValueError(
                "attention_mask length must cover all key/value positions."
            )

        mask_mod = self._build_flex_mask_mod(attention_mask, use_causal)
        if mask_mod is None:
            return None

        batch_dim = attention_mask.size(0) if attention_mask is not None else None
        return create_block_mask(
            mask_mod,
            B=batch_dim,
            H=None,
            Q_LEN=query_len,
            KV_LEN=key_len,
            device=device,
        )

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
        flex_block_mask: Optional[object],
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        del flex_block_mask

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        pos_xyz: Optional[torch.Tensor] = None,
        sdpa_mask: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
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

        attn_output = self._apply_attention(
            queries,
            keys,
            values,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            sdpa_mask=sdpa_mask,
            flex_block_mask=flex_block_mask,
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
        pos_xyz: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
    ) -> torch.Tensor:
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

            attn_output = self._apply_attention(
                queries,
                key_layer,
                value_layer,
                attention_mask=attention_mask,
                causal_mask=None,
                sdpa_mask=None,
                flex_block_mask=flex_block_mask,
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
            flex_block_mask=flex_block_mask,
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
        pos_xyz: Optional[torch.Tensor],
        sdpa_mask: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        attn_input = self.ln_1(hidden_states)
        attn_output = self.attention(
            attn_input,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            pos_xyz=pos_xyz,
            sdpa_mask=sdpa_mask,
            flex_block_mask=flex_block_mask,
            is_causal=is_causal,
        )
        hidden_states = hidden_states + attn_output

        ff_input = self.ln_2(hidden_states)
        ff_output = self.ff(ff_input)
        hidden_states = hidden_states + ff_output
        return hidden_states

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        pos_xyz: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_input = self.ln_1(hidden_states)
        # Check if we are in Decode mode or Prompt mode
        if past_key_value is not None:
            # --- DECODE MODE ---
            # Attention returns ONLY tensor
            attn_output = self.attention.forward_with_cache(
                attn_input,
                pos_xyz=pos_xyz,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                flex_block_mask=flex_block_mask,
            )
            present_key_value = None  # No return value needed
        else:
            # --- PROMPT MODE ---
            # Attention returns Tensor + KV Tuple
            attn_output, present_key_value = self.attention.forward_with_cache(
                attn_input,
                pos_xyz=pos_xyz,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=None,
                cache_position=None,
                flex_block_mask=flex_block_mask,
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
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

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

    def forward(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sep_indices: Optional[torch.Tensor] = None,
        compute_input_loss: bool = True,
        targets: Optional[torch.Tensor] = None,
        positions_3d: Optional[torch.Tensor] = None,
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

        if positions_3d is not None and positions_3d.shape[:2] != input_ids.shape:
            raise ValueError("positions_3d must match [batch, seq_len] of input_ids.")

        if targets is None:
            targets = input_ids

        token_embeds = self.token_embedding(input_ids)
        example_embeds = self.example_embedding(example_ids)  # [B, D]
        # Add the per-example embedding to every token in the sequence.
        hidden_states = token_embeds + example_embeds.unsqueeze(1)
        hidden_states = self.dropout(hidden_states)

        # Compute or reuse 3D positions per token.
        if positions_3d is None:
            pos_xyz = self._compute_positions_3d(input_ids, attention_mask)
        else:
            pos_xyz = positions_3d.to(device=device, dtype=torch.long)

        # Training batches are right-padded; causal attention is sufficient to
        # prevent valid tokens from attending to padded keys. Keeping attn_mask
        # unset allows SDPA to dispatch flash kernels.
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                causal_mask=None,
                pos_xyz=pos_xyz,
                sdpa_mask=None,
                flex_block_mask=None,
                is_causal=True,
            )
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        input_loss = None
        output_loss = None

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

    def forward_generate(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        positions_3d: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        example_embeds: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward used for autoregressive generation with a KV cache.

        When `past_key_values` is None, the call is treated as a full prompt
        pass and the method returns per-layer key/value tensors that
        represent the entire prefix. When `past_key_values` is provided,
        `input_ids` and `positions_3d` should contain only the newly
        generated tokens, and the cache is updated accordingly. `attention_mask`
        can be provided to mask padded tokens during the initial prompt pass,
        and to mask cached keys during incremental decoding.
        """
        batch_size, seq_len = input_ids.size()
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model capacity ({self.config.max_seq_len})."
            )

        if positions_3d is not None and positions_3d.shape[:2] != input_ids.shape:
            raise ValueError("positions_3d must match [batch, seq_len] of input_ids.")

        device = input_ids.device
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

        token_embeds = self.token_embedding(input_ids)

        # During generation, also broadcast the example embedding across
        # all tokens in the (prompt or incremental) sequence.
        hidden_states = token_embeds + example_embeds.unsqueeze(1)
        # hidden_states = self.dropout(hidden_states)

        pos_xyz = (
            positions_3d.to(device=device, dtype=torch.long)
            if positions_3d is not None
            else None
        )

        # Initial prompt: no cache yet, compute 3D positions and use the
        # exact same masking behavior as the standard forward pass.
        if past_key_values is None:
            if attention_mask is None:
                attention_mask = torch.ones_like(
                    input_ids, dtype=torch.bool, device=device
                )

            if pos_xyz is None:
                pos_xyz = self._compute_positions_3d(input_ids, attention_mask)
            causal_mask = self._build_causal_mask(seq_len, device)

            past_key_values_out: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for block in self.blocks:
                hidden_states, present_kv = block.forward_with_cache(
                    hidden_states,
                    pos_xyz,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    past_key_value=None,
                    flex_block_mask=None,
                )
                past_key_values_out.append(present_kv)

            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return {"logits": logits, "past_key_values": tuple(past_key_values_out)}

        if pos_xyz is None:
            raise ValueError(
                "positions_3d must be provided when using past_key_values."
            )
        # We iterate by index to access the specific layer's static buffer in past_key_values
        for i, block in enumerate(self.blocks):
            # Note:
            # 1. We pass past_key_values[i] (the static buffer).
            # 2. We do NOT capture a second return value (present_kv) because
            #    TransformerBlock.forward_with_cache no longer returns it in this mode.
            hidden_states = block.forward_with_cache(
                hidden_states,
                pos_xyz,
                attention_mask=attention_mask,
                past_key_value=past_key_values[i],
                cache_position=cache_position,
                flex_block_mask=None,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Return ONLY logits.
        # The cache update is done, and we don't need to pass it back.
        return {"logits": logits}

    # ------------------------ 3D RoPE utilities ------------------------
    def _compute_positions_3d(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute 3D positions on CPU, then move them to the target device."""
        pos_cpu = compute_positions_3d(
            input_ids=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
        )
        return pos_cpu.to(device=input_ids.device, dtype=torch.long)


class RotaryEmbedding3D(nn.Module):
    """3D Rotary Positional Embedding applied to Q/K using precomputed lookups.

    Splits head_dim into three even slices (x,y,z). Precomputes cos/sin tables
    for fixed coordinate ranges (x:0-32, y:0-32, z:0-8) to speed up inference.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        self.head_dim = head_dim
        self.base = base

        # Distribute pairs across 3 axes as evenly as possible
        n_pairs = head_dim // 2
        px = n_pairs // 3
        py = n_pairs // 3
        pz = n_pairs - px - py
        self.d_x = px * 2
        self.d_y = py * 2
        self.d_z = pz * 2

        # Define bounds based on grid constraints (30x30 max, 5 z-levels).
        # Using slightly higher powers-of-2-ish bounds for safety.
        self.max_x = 32
        self.max_y = 32
        self.max_z = 8

        # Precompute and register caches
        self._register_cache("x", self.d_x, self.max_x)
        self._register_cache("y", self.d_y, self.max_y)
        self._register_cache("z", self.d_z, self.max_z)

    def _build_inv_freq(self, dim: int) -> torch.Tensor:
        if dim <= 0:
            return torch.empty(0)
        return 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))

    def _register_cache(self, name: str, dim: int, max_pos: int) -> None:
        if dim <= 0:
            # Empty buffers for unused dimensions (keeps state_dict clean)
            self.register_buffer(f"cos_{name}_cache", torch.empty(0), persistent=False)
            self.register_buffer(f"sin_{name}_cache", torch.empty(0), persistent=False)
            return

        inv_freq = self._build_inv_freq(dim)
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

    def apply_rotary(
        self, q: torch.Tensor, k: torch.Tensor, pos_xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D RoPE to the first d_x, d_y, d_z channels respectively.

        q, k: [B, H, S, D]
        pos_xyz: [B, S, 3] with integer coordinates (x, y, z)
        """
        # B, H, S, D = q.shape
        # assert D == self.head_dim

        # Ensure indices are within bounds (clamping protects against out-of-bounds crash)
        # pos_xyz is expected to be LongTensor suitable for indexing.
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

        # Concatenate tables to form the full [B, S, D] embedding mask.
        # This 'cat' is on non-gradient tensors, which is cheap in the backward pass.
        cos = torch.cat(parts_cos, dim=-1).unsqueeze(1)  # [B, 1, S, D]
        sin = torch.cat(parts_sin, dim=-1).unsqueeze(1)  # [B, 1, S, D]

        # Apply standard RoPE arithmetic on the full tensors.
        # This keeps q and k contiguous and avoids slicing activations.
        q_out = (q * cos) + (self._rotate_half(q) * sin)
        k_out = (k * cos) + (self._rotate_half(k) * sin)

        return q_out, k_out
