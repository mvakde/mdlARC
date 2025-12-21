from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    IGNORE_INDEX,
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    compute_positions_3d,
    IO_SEPARATOR_TOKEN_ID,
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
    num_examples: int = 1280

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        if self.num_examples < 1:
            raise ValueError("num_examples must be >= 1.")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout_p = config.dropout  # Store float for functional call

        # 3D RoPE setup
        self.rope = RotaryEmbedding3D(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        pos_xyz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)

        if pos_xyz is not None:
            # pos_xyz: [B, S, 3] (x, y, z)
            queries, keys = self.rope.apply_rotary(queries, keys, pos_xyz)

        # Construct a combined attention bias for SDPA
        # SDPA handles broadcasting, but constructing the mask explicitly ensures
        # correctness with your specific causal + padding setup.
        attn_bias = None

        # 1. Start with Causal Mask if present
        if causal_mask is not None:
            # causal_mask is usually boolean [1, 1, S, S] where True means "Mask out"
            # We convert to float: 0.0 for keep, -inf for mask
            attn_bias = torch.zeros(
                (1, 1, seq_len, seq_len), device=queries.device, dtype=queries.dtype
            )
            attn_bias = attn_bias.masked_fill(causal_mask, float("-inf"))

        # 2. Apply Padding Mask (attention_mask)
        if attention_mask is not None:
            # attention_mask is [B, S] where True means "Keep", False means "Pad"
            # We need to mask out keys where attention_mask is False.
            if attn_bias is None:
                attn_bias = torch.zeros(
                    (batch_size, 1, seq_len, seq_len),
                    device=queries.device,
                    dtype=queries.dtype,
                )

            # Broadcast attention_mask to [B, 1, 1, S]
            key_mask = ~attention_mask[:, None, None, :]
            attn_bias = attn_bias.masked_fill(key_mask, float("-inf"))

        # Fused Flash Attention execution
        # Note: We pass is_causal=False because we manually constructed the causal mask into attn_bias above
        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
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
    ) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)

        if pos_xyz is not None:
            # Cast to fp32 for rotation precision
            q_f32 = queries.float()
            k_f32 = keys.float()
            q_f32, k_f32 = self.rope.apply_rotary(q_f32, k_f32, pos_xyz)
            # Cast back to original dtype (e.g. bfloat16) for storage
            queries = q_f32.to(dtype=hidden_states.dtype)
            keys = k_f32.to(dtype=hidden_states.dtype)

        # ------------------------------------------------------------------
        # PATH A: DECODE (We have a cache buffer)
        # ------------------------------------------------------------------
        if past_key_value is not None:
            past_keys, past_values = past_key_value
            if cache_position is not None:
                # We update the buffer directly using index_copy_ (in-place).
                # keys is [B, H, 1, D], past_keys is [B, H, MaxLen, D]
                # cache_position is Tensor([step])
                past_keys.index_copy_(2, cache_position, keys)
                past_values.index_copy_(2, cache_position, values)

                # Use the FULL buffer for attention (masking handles the future tokens)
                key_layer = past_keys
                value_layer = past_values

                # Masking setup for static buffer
            attn_bias = None
            if attention_mask is not None:
                attn_bias = torch.zeros(
                    (batch_size, 1, seq_len, key_layer.size(2)),
                    device=queries.device,
                    dtype=queries.dtype,
                )
                key_mask = ~attention_mask[:, None, None, :]
                attn_bias = attn_bias.masked_fill(key_mask, float("-inf"))

            attn_output = F.scaled_dot_product_attention(
                queries,
                key_layer,
                value_layer,
                attn_mask=attn_bias,
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
        attn_bias = None
        if causal_mask is not None:
            attn_bias = torch.zeros(
                (1, 1, seq_len, seq_len), device=queries.device, dtype=queries.dtype
            )
            attn_bias = attn_bias.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            if attn_bias is None:
                attn_bias = torch.zeros(
                    (batch_size, 1, seq_len, seq_len),
                    device=queries.device,
                    dtype=queries.dtype,
                )
            key_mask = ~attention_mask[:, None, None, :]
            attn_bias = attn_bias.masked_fill(key_mask, float("-inf"))

        attn_output = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        )

        # --- Return output AND the new KVs so we can build the buffer ---
        return self.out_proj(attn_output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal_mask: torch.Tensor,
        pos_xyz: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_input = self.ln_1(hidden_states)
        attn_output = self.attention(
            attn_input,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            pos_xyz=pos_xyz,
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
        self.norm = nn.LayerNorm(config.d_model)
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
        targets: Optional[torch.Tensor] = None,
        positions_3d: Optional[torch.Tensor] = None,
    ) -> dict:
        batch_size, seq_len = input_ids.size()
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model capacity ({self.config.max_seq_len})."
            )

        device = input_ids.device
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

        causal_mask = self._build_causal_mask(seq_len, device)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, causal_mask, pos_xyz)
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
            # The input sequence (shift_logits input) is input_ids[:, :-1]
            shift_input_ids = input_ids[:, :-1]

            # Find where the Output phase starts.
            # If the current input token is IO_SEPARATOR, it is predicting the first output token.
            # So, the "Output Loss" region starts wherever SEP or subsequent tokens appear.
            # cumsum >= 1 creates a mask that turns True from the Separator onwards.
            is_output_phase = (shift_input_ids == IO_SEPARATOR_TOKEN_ID).cumsum(
                dim=1
            ) >= 1
            is_input_phase = ~is_output_phase

            # Calculate specific losses
            valid_input = valid_mask & is_input_phase
            valid_output = valid_mask & is_output_phase

            num_output_tokens = valid_output.sum()

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
