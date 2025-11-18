from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    IGNORE_INDEX,
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    START_TOKEN_ID,
    NEXT_LINE_TOKEN_ID,
    IO_SEPARATOR_TOKEN_ID,
    END_TOKEN_ID,
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
    num_examples: int = 1024

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
        self.dropout = nn.Dropout(config.dropout)

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

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            key_mask = ~attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, values)
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
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Self-attention variant that also exposes a KV cache.

        When `past_key_value` is None, this reduces to the normal attention
        computation (including causal and padding masks) while returning
        the per-layer keys/values for caching. When `past_key_value` is
        provided, `hidden_states` and `pos_xyz` should contain only the
        newly generated tokens and no masking is applied, since there are
        no future positions during autoregressive decoding.
        """
        batch_size, seq_len, dim = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)

        if pos_xyz is not None:
            queries, keys = self.rope.apply_rotary(queries, keys, pos_xyz)

        # Incremental decoding branch: concatenate cached K/V and attend
        # from the new tokens only. No causal mask is needed because there
        # are no future positions beyond the newly appended tokens.
        if past_key_value is not None:
            past_keys, past_values = past_key_value
            keys = torch.cat([past_keys, keys], dim=2)
            values = torch.cat([past_values, values], dim=2)

            attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, values)
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, dim)
            )
            attn_output = self.out_proj(attn_output)
            present_key_value = (keys, values)
            return attn_output, present_key_value

        # Full-sequence branch used for the initial prompt: this mirrors
        # the standard attention forward (including causal and padding
        # masks) while also returning K/V for caching.
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            key_mask = ~attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, values)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        )
        attn_output = self.out_proj(attn_output)
        present_key_value = (keys, values)
        return attn_output, present_key_value


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
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_input = self.ln_1(hidden_states)
        attn_output, present_key_value = self.attention.forward_with_cache(
            attn_input,
            pos_xyz=pos_xyz,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            past_key_value=past_key_value,
        )
        hidden_states = hidden_states + attn_output

        ff_input = self.ln_2(hidden_states)
        ff_output = self.ff(ff_input)
        hidden_states = hidden_states + ff_output

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
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
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        if targets is None:
            targets = input_ids

        token_embeds = self.token_embedding(input_ids)
        example_embeds = self.example_embedding(example_ids)  # [B, D]
        # Add the per-example embedding to every token in the sequence.
        hidden_states = token_embeds + example_embeds.unsqueeze(1)
        hidden_states = self.dropout(hidden_states)

        # Compute 3D positions per token based on token semantics.
        pos_xyz = self._compute_positions_3d(input_ids, attention_mask)

        causal_mask = self._build_causal_mask(seq_len, device)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, causal_mask, pos_xyz)
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            shift_targets = shift_targets.masked_fill(~shift_mask, IGNORE_INDEX)
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return {"logits": logits, "loss": loss}

    def forward_generate(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        positions_3d: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward used for autoregressive generation with a KV cache.

        When `past_key_values` is None, the call is treated as a full prompt
        pass and the method returns per-layer key/value tensors that
        represent the entire prefix. When `past_key_values` is provided,
        `input_ids` and `positions_3d` should contain only the newly
        generated tokens, and the cache is updated accordingly.
        """
        batch_size, seq_len = input_ids.size()
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model capacity ({self.config.max_seq_len})."
            )

        device = input_ids.device

        token_embeds = self.token_embedding(input_ids)
        example_embeds = self.example_embedding(example_ids)

        # During generation, also broadcast the example embedding across
        # all tokens in the (prompt or incremental) sequence.
        hidden_states = token_embeds + example_embeds.unsqueeze(1)
        hidden_states = self.dropout(hidden_states)

        # Initial prompt: no cache yet, compute 3D positions and use the
        # exact same masking behavior as the standard forward pass.
        if past_key_values is None:
            attention_mask = torch.ones_like(
                input_ids, dtype=torch.bool, device=device
            )
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

        if positions_3d is None:
            raise ValueError("positions_3d must be provided when using past_key_values.")

        if len(past_key_values) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} past key/value pairs, "
                f"got {len(past_key_values)}."
            )

        past_key_values_out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for block, layer_past in zip(self.blocks, past_key_values):
            hidden_states, present_kv = block.forward_with_cache(
                hidden_states, positions_3d, past_key_value=layer_past
            )
            past_key_values_out.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return {"logits": logits, "past_key_values": tuple(past_key_values_out)}

    # ------------------------ 3D RoPE utilities ------------------------
    def _compute_positions_3d(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Map sequence tokens to 3D coordinates (x,y,z) per user spec.

        Grid extents: x in [0, 29], y in [0, 30], z in {0..4}.
        Mapping rules:
          - <start> at (0,0,0)
          - Input grid in z=1 layer; rows along y, columns along x; <next_line> occupies a cell
          - <input_output_separator> at (0,0,2)
          - Output grid in z=3 layer; same layout as input
          - <end> at (0,0,4)
        """
        B, S = input_ids.shape
        device = input_ids.device
        pos = torch.zeros((B, S, 3), dtype=torch.long, device=device)

        # Iterate per batch element to respect per-sequence semantics
        for b in range(B):
            x = 0
            y = 0
            z = 1  # start in input layer after <start>
            seen_sep = False
            for t in range(S):
                if not attention_mask[b, t]:
                    # beyond real tokens; leave zeros
                    continue
                tok = int(input_ids[b, t].item())
                if t == 0 and tok == START_TOKEN_ID:
                    pos[b, t, 0] = 0
                    pos[b, t, 1] = 0
                    pos[b, t, 2] = 0
                    # Do not change grid counters
                    continue

                if tok == IO_SEPARATOR_TOKEN_ID:
                    pos[b, t, 0] = 0
                    pos[b, t, 1] = 0
                    pos[b, t, 2] = 2
                    # Switch to output grid after separator
                    x, y = 0, 0
                    z = 3
                    seen_sep = True
                    continue

                if tok == END_TOKEN_ID:
                    pos[b, t, 0] = 0
                    pos[b, t, 1] = 0
                    pos[b, t, 2] = 4
                    continue

                # Regular grid tokens (digits 0-9 and <next_line>)
                # Clamp to grid bounds (30x31) defensively
                px = min(max(x, 0), 29)
                py = min(max(y, 0), 30)
                pos[b, t, 0] = px
                pos[b, t, 1] = py
                pos[b, t, 2] = z

                if tok == NEXT_LINE_TOKEN_ID:
                    x = 0
                    y += 1
                else:
                    x += 1

        return pos


class RotaryEmbedding3D(nn.Module):
    """3D Rotary Positional Embedding applied to Q/K.

    Splits head_dim into three even slices (x,y,z), applies standard RoPE to
    each slice using the token's grid coordinate along that axis.
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
        # Precompute inverse frequency for each axis slice
        self.register_buffer("inv_freq_x", self._build_inv_freq(self.d_x), persistent=False)
        self.register_buffer("inv_freq_y", self._build_inv_freq(self.d_y), persistent=False)
        self.register_buffer("inv_freq_z", self._build_inv_freq(self.d_z), persistent=False)

    def _build_inv_freq(self, dim: int) -> torch.Tensor:
        if dim <= 0:
            return torch.empty(0)
        # Standard RoPE frequency schedule
        return 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # pairwise rotate: (x0,x1,x2,x3,...) -> (-x1, x0, -x3, x2, ...)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.stack((-x2, x1), dim=-1)
        return out.flatten(-2)

    def _build_cos_sin(
        self, pos: torch.Tensor, inv_freq: torch.Tensor, dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dim == 0:
            shape = (*pos.shape, 0)
            zero = torch.zeros(shape, dtype=pos.dtype, device=pos.device)
            return zero, zero
        # pos: [B, S]; inv_freq: [dim/2]
        t = pos.float().unsqueeze(-1) * inv_freq  # [B, S, dim/2]
        cos = torch.cos(t).repeat_interleave(2, dim=-1)  # [B, S, dim]
        sin = torch.sin(t).repeat_interleave(2, dim=-1)
        return cos, sin

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_xyz: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D RoPE to the first d_x, d_y, d_z channels respectively.

        q, k: [B, H, S, D]
        pos_xyz: [B, S, 3] with integer coordinates (x, y, z)
        """
        B, H, S, D = q.shape
        assert D == self.head_dim

        # Build cos/sin for each axis and broadcast over heads
        pos_x = pos_xyz[..., 0]
        pos_y = pos_xyz[..., 1]
        pos_z = pos_xyz[..., 2]
        cos_x, sin_x = self._build_cos_sin(pos_x, self.inv_freq_x, self.d_x)
        cos_y, sin_y = self._build_cos_sin(pos_y, self.inv_freq_y, self.d_y)
        cos_z, sin_z = self._build_cos_sin(pos_z, self.inv_freq_z, self.d_z)
        # [B, 1, S, dim]
        cos_x = cos_x.unsqueeze(1)
        sin_x = sin_x.unsqueeze(1)
        cos_y = cos_y.unsqueeze(1)
        sin_y = sin_y.unsqueeze(1)
        cos_z = cos_z.unsqueeze(1)
        sin_z = sin_z.unsqueeze(1)

        # Slices
        dx, dy, dz = self.d_x, self.d_y, self.d_z
        s0 = 0
        s1 = s0 + dx
        s2 = s1 + dy
        s3 = s2 + dz

        q_x, q_y, q_z = q[..., s0:s1], q[..., s1:s2], q[..., s2:s3]
        k_x, k_y, k_z = k[..., s0:s1], k[..., s1:s2], k[..., s2:s3]

        if dx > 0:
            q_x = q_x * cos_x + self._rotate_half(q_x) * sin_x
            k_x = k_x * cos_x + self._rotate_half(k_x) * sin_x
        if dy > 0:
            q_y = q_y * cos_y + self._rotate_half(q_y) * sin_y
            k_y = k_y * cos_y + self._rotate_half(k_y) * sin_y
        if dz > 0:
            q_z = q_z * cos_z + self._rotate_half(q_z) * sin_z
            k_z = k_z * cos_z + self._rotate_half(k_z) * sin_z

        # Concatenate back, leaving any remaining tail dims (if any) unchanged
        if s3 < D:
            q_tail = q[..., s3:]
            k_tail = k[..., s3:]
            q = torch.cat([q[..., :s0], q_x, q_y, q_z, q_tail], dim=-1)
            k = torch.cat([k[..., :s0], k_x, k_y, k_z, k_tail], dim=-1)
        else:
            q = torch.cat([q[..., :s0], q_x, q_y, q_z], dim=-1)
            k = torch.cat([k[..., :s0], k_x, k_y, k_z], dim=-1)

        return q, k
