"""Common utilities shared across build, train, and evaluate modules.

Contains: vocabulary definitions, tokenization, grid transforms, dataset classes,
dataloader creation, 3D position computation, and RNG/device handling.
"""

import functools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from numba import njit
from torch.utils.data import DataLoader, Dataset, Sampler


# =============================================================================
# Vocabulary Definitions
# =============================================================================

VOCAB_SIZE = 14

SPECIAL_TOKENS = ["<start>", "<next_line>", "<input_output_separator>", "<end>"]
TOKEN_TO_ID: Dict[str, int] = {str(i): i for i in range(10)}
for offset, token in enumerate(SPECIAL_TOKENS, start=10):
    TOKEN_TO_ID[token] = offset

ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

START_TOKEN_ID = TOKEN_TO_ID["<start>"]
NEXT_LINE_TOKEN_ID = TOKEN_TO_ID["<next_line>"]
IO_SEPARATOR_TOKEN_ID = TOKEN_TO_ID["<input_output_separator>"]
END_TOKEN_ID = TOKEN_TO_ID["<end>"]

MAX_SEQ_LEN = 1863
IGNORE_INDEX = -100


# =============================================================================
# Color Extraction
# =============================================================================

def _extract_task_colors(task: Dict[str, object], key: str) -> Set[int]:
    colors: Set[int] = set()
    for split in ("train", "test"):
        pairs = task.get(split, [])
        for pair in pairs:
            grid = pair.get(key) or []
            for row in grid:
                for val in row:
                    val_i = int(val)
                    if 1 <= val_i <= 9:
                        colors.add(val_i)
    return colors


def extract_task_input_colors(task: Dict[str, object]) -> List[int]:
    """Return sorted colors (1-9) eligible for permutation, excluding output-only colors."""
    input_colors = _extract_task_colors(task, "input")
    output_colors = _extract_task_colors(task, "output")
    output_only = output_colors - input_colors
    return [color for color in range(1, 10) if color not in output_only]


# =============================================================================
# Dihedral Transforms (8 geometric operations)
# =============================================================================

_DIHEDRAL_TRANSFORM_NAMES = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_horizontal",
    "flip_vertical",
    "flip_main_diagonal",
    "flip_anti_diagonal",
]


def _dihedral_copy(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(row) for row in grid]


def _dihedral_rot90(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    if not grid:
        return []
    return [list(row) for row in zip(*grid[::-1])]


def _dihedral_rot180(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(reversed(row)) for row in reversed(grid)]


def _dihedral_rot270(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    if not grid:
        return []
    return [list(row) for row in zip(*grid)][::-1]


def _dihedral_flip_horizontal(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(reversed(row)) for row in grid]


def _dihedral_flip_vertical(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(row) for row in reversed(grid)]


def _dihedral_flip_main_diagonal(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    if not grid:
        return []
    return [list(row) for row in zip(*grid)]


def _dihedral_flip_anti_diagonal(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    return _dihedral_flip_vertical(_dihedral_rot90(grid))


_DIHEDRAL_TRANSFORMS = {
    "identity": _dihedral_copy,
    "rot90": _dihedral_rot90,
    "rot180": _dihedral_rot180,
    "rot270": _dihedral_rot270,
    "flip_horizontal": _dihedral_flip_horizontal,
    "flip_vertical": _dihedral_flip_vertical,
    "flip_main_diagonal": _dihedral_flip_main_diagonal,
    "flip_anti_diagonal": _dihedral_flip_anti_diagonal,
}


def apply_dihedral_transform(
    grid: Sequence[Sequence[int]], transform_index: int
) -> List[List[int]]:
    if transform_index < 0:
        raise ValueError("transform_index must be non-negative.")
    transform_name = _DIHEDRAL_TRANSFORM_NAMES[transform_index % 8]
    return _DIHEDRAL_TRANSFORMS[transform_name](grid)


_DIHEDRAL_INVERSES = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_horizontal": "flip_horizontal",
    "flip_vertical": "flip_vertical",
    "flip_main_diagonal": "flip_main_diagonal",
    "flip_anti_diagonal": "flip_anti_diagonal",
}


def apply_inverse_dihedral_transform(
    grid: Sequence[Sequence[int]], transform_index: int
) -> List[List[int]]:
    """Undo a dihedral transform using the known augmentation index (mod 8)."""
    if transform_index < 0:
        raise ValueError("transform_index must be non-negative.")
    transform_name = _DIHEDRAL_TRANSFORM_NAMES[transform_index % 8]
    inverse_name = _DIHEDRAL_INVERSES[transform_name]
    return _DIHEDRAL_TRANSFORMS[inverse_name](grid)


# =============================================================================
# Color Permutation
# =============================================================================

def apply_color_permutation_to_tokens(
    tokens: Sequence[int], mapping: Sequence[int]
) -> List[int]:
    """Apply a color permutation mapping to a token list (keeps specials/0 fixed)."""
    return [int(mapping[tok] if 0 <= tok < len(mapping) else tok) for tok in tokens]


def apply_color_permutation_to_grid(
    grid: Sequence[Sequence[int]], mapping: Sequence[int]
) -> List[List[int]]:
    return [
        [int(mapping[val] if 0 <= val < len(mapping) else val) for val in row]
        for row in grid
    ]


# =============================================================================
# Grid/Token Conversion
# =============================================================================

def _value_to_token_id(value: int) -> int:
    if value not in range(10):
        raise ValueError(f"Grid values must be digits in [0, 9], received {value}")
    return value


def grid_to_tokens(grid: Iterable[Iterable[int]]) -> List[int]:
    """Flattens a 2D grid into a token list, inserting <next_line> after each row."""
    tokens: List[int] = []
    for row in grid:
        for value in row:
            tokens.append(_value_to_token_id(int(value)))
        tokens.append(NEXT_LINE_TOKEN_ID)
    return tokens


def encode_example(
    input_grid: Iterable[Iterable[int]],
    output_grid: Optional[Iterable[Iterable[int]]] = None,
    include_output: bool = True,
    append_end: bool = True,
) -> List[int]:
    """Serializes an ARC pair into a single token stream."""
    tokens = [START_TOKEN_ID]
    tokens.extend(grid_to_tokens(input_grid))
    tokens.append(IO_SEPARATOR_TOKEN_ID)
    if include_output and output_grid is not None:
        tokens.extend(grid_to_tokens(output_grid))
    if append_end:
        tokens.append(END_TOKEN_ID)
    return tokens


def load_challenges(json_path: Path) -> Dict[str, dict]:
    with Path(json_path).open("r") as handle:
        return json.load(handle)


def tokens_to_grid(tokens: Sequence[int]) -> List[List[int]]:
    """Converts a flat sequence of tokens into a grid (list of rows)."""
    rows: List[List[int]] = []
    current_row: List[int] = []
    for token in tokens:
        if token == NEXT_LINE_TOKEN_ID:
            if current_row:
                rows.append(current_row)
                current_row = []
            continue
        if 0 <= token <= 9:
            current_row.append(token)
        else:
            break
    if current_row:
        rows.append(current_row)
    return rows


def extract_output_tokens(sequence: Sequence[int]) -> List[int]:
    """Returns the tokens that appear after the <input_output_separator> marker."""
    after_separator = False
    outputs: List[int] = []
    for token in sequence:
        if not after_separator:
            if token == IO_SEPARATOR_TOKEN_ID:
                after_separator = True
            continue
        if token == END_TOKEN_ID:
            break
        outputs.append(token)
    return outputs


def tokens_to_string(tokens: Sequence[int]) -> str:
    """Convert a token id sequence into a space-delimited string."""
    parts: List[str] = []
    for tok in tokens:
        parts.append(ID_TO_TOKEN.get(int(tok), str(int(tok))))
    return " ".join(parts)


def split_grids_from_tokens(tokens: Sequence[int]) -> List[List[List[int]]]:
    """Split a token sequence into multiple grids."""
    grids: List[List[List[int]]] = []
    current_grid: List[List[int]] = []
    current_row: List[int] = []

    for tok in tokens:
        if tok == END_TOKEN_ID:
            break
        if tok == START_TOKEN_ID:
            continue
        if tok == IO_SEPARATOR_TOKEN_ID:
            if current_row:
                current_grid.append(current_row)
                current_row = []
            if current_grid:
                grids.append(current_grid)
            current_grid = []
            continue
        if tok == NEXT_LINE_TOKEN_ID:
            if current_row:
                current_grid.append(current_row)
                current_row = []
            continue
        if 0 <= tok <= 9:
            current_row.append(int(tok))
        else:
            break

    if current_row:
        current_grid.append(current_row)
    if current_grid:
        grids.append(current_grid)
    return grids


def is_rectangular_grid(grid: Sequence[Sequence[int]]) -> bool:
    """Return True if all rows have the same non-zero length."""
    if not grid:
        return False
    first_row_len = len(grid[0])
    if first_row_len == 0:
        return False
    return all(len(row) == first_row_len for row in grid)


# =============================================================================
# 3D Position Computation (Numba-accelerated)
# =============================================================================

@njit
def _fill_3d_positions_numba(ids, mask, out, start_id, sep_id, end_id, nl_id):
    B, S = ids.shape
    for b in range(B):
        x = 0
        y = 0
        z = 1
        for t in range(S):
            if not mask[b, t]:
                continue

            val = ids[b, t]

            if val == start_id:
                out[b, t, 0] = 0
                out[b, t, 1] = 0
                out[b, t, 2] = 0
                x = 0
                y = 0
                z = 1
                continue

            if val == sep_id:
                out[b, t, 0] = 0
                out[b, t, 1] = 0
                out[b, t, 2] = 2
                x = 0
                y = 0
                z = 3
                continue

            if val == end_id:
                out[b, t, 0] = 0
                out[b, t, 1] = 0
                out[b, t, 2] = 4
                continue

            px = x
            if px < 0:
                px = 0
            if px > 30:
                px = 30

            py = y
            if py < 0:
                py = 0
            if py > 29:
                py = 29

            out[b, t, 0] = px
            out[b, t, 1] = py
            out[b, t, 2] = z

            if val == nl_id:
                x = 0
                y += 1
            else:
                x += 1


def compute_positions_3d(
    input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute per-token 3D grid coordinates on CPU (using numba)."""
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape [batch, seq_len].")

    ids_cpu = input_ids.detach().cpu()
    ids_np = ids_cpu.numpy()

    if attention_mask is None:
        mask_np = np.ones_like(ids_np, dtype=bool)
    else:
        mask_np = attention_mask.detach().cpu().numpy().astype(bool)

    B, S = ids_np.shape
    pos_np = np.zeros((B, S, 3), dtype=np.int64)

    _fill_3d_positions_numba(
        ids_np,
        mask_np,
        pos_np,
        START_TOKEN_ID,
        IO_SEPARATOR_TOKEN_ID,
        END_TOKEN_ID,
        NEXT_LINE_TOKEN_ID,
    )

    return torch.from_numpy(pos_np).to(device=input_ids.device)


# =============================================================================
# Dataset Classes
# =============================================================================

@dataclass
class SequenceExample:
    tokens: torch.LongTensor
    example_id: int
    task_id: str
    split: str
    pair_index: int
    has_output: bool
    seq_len: int
    tokens_by_dihedral: Optional[List[torch.LongTensor]] = None
    cached_positions_by_dihedral: Optional[List[torch.LongTensor]] = None
    seq_len_by_dihedral: Optional[List[int]] = None


class ARCExampleDataset(Dataset):
    """Dataset that turns ARC tasks into autoregressive token sequences."""

    def __init__(
        self,
        json_path: Path,
        splits: Sequence[str] = ("train", "test"),
        include_outputs: bool = True,
        max_seq_len: int = MAX_SEQ_LEN,
        drop_long_sequences: bool = False,
        task_whitelist: Optional[Sequence[str]] = None,
        load_test_solutions: bool = False,
    ) -> None:
        available_splits = {"train", "test"}
        for split in splits:
            if split not in available_splits:
                raise ValueError(
                    f"Unsupported split '{split}'. Expected values in {available_splits}."
                )

        self.source_path = Path(json_path)
        self.max_seq_len = max_seq_len
        self.drop_long_sequences = drop_long_sequences
        self.include_outputs = include_outputs

        challenges = load_challenges(self.source_path)

        solutions_map = {}
        if load_test_solutions:
            sol_path = self.source_path.with_name("solutions.json")
            if sol_path.exists():
                with sol_path.open("r") as handle:
                    solutions_map = json.load(handle)
            else:
                print(f"Warning: solutions.json not found at {sol_path}")
        if task_whitelist is not None:
            task_ids = list(task_whitelist)
            missing = [task_id for task_id in task_ids if task_id not in challenges]
            if missing:
                raise ValueError(f"Task ids {missing} were not found in {json_path}.")
        else:
            task_ids = sorted(challenges.keys())

        self.examples: List[SequenceExample] = []
        self.task_id_to_example_id: Dict[str, int] = {}
        self.indices_by_split: Dict[str, List[int]] = {split: [] for split in splits}
        self.task_ids = task_ids
        self.sequence_lengths: List[int] = []
        self.task_input_colors: Dict[str, List[int]] = {}

        for example_id, task_id in enumerate(task_ids):
            self.task_id_to_example_id[task_id] = example_id
            task = challenges[task_id]
            self.task_input_colors[task_id] = extract_task_input_colors(task)
            for split in splits:
                pairs = task.get(split, [])
                for pair_index, pair in enumerate(pairs):
                    input_grid = pair["input"]
                    output_grid = pair.get("output")

                    if split == "test" and load_test_solutions:
                        if task_id in solutions_map:
                            task_sols = solutions_map[task_id]
                            if pair_index < len(task_sols):
                                output_grid = task_sols[pair_index]

                    has_output = output_grid is not None
                    include_output_tokens = include_outputs and has_output
                    append_end = include_output_tokens

                    tokens_by_dihedral: List[torch.Tensor] = []
                    seq_len_by_dihedral: List[int] = []
                    skip_example = False
                    for transform_index in range(8):
                        dihedral_input = apply_dihedral_transform(
                            input_grid, transform_index
                        )
                        dihedral_output = (
                            apply_dihedral_transform(output_grid, transform_index)
                            if output_grid is not None
                            else None
                        )
                        tokens = encode_example(
                            dihedral_input,
                            dihedral_output,
                            include_output=include_output_tokens,
                            append_end=append_end,
                        )
                        if len(tokens) > max_seq_len:
                            if drop_long_sequences:
                                skip_example = True
                                break
                            raise ValueError(
                                f"Sequence length {len(tokens)} exceeds max_seq_len={max_seq_len} "
                                f"for task {task_id} ({split} pair {pair_index}) dihedral {transform_index}."
                            )
                        tensor = torch.tensor(tokens, dtype=torch.long)
                        tokens_by_dihedral.append(tensor)
                        seq_len_by_dihedral.append(len(tokens))
                    if skip_example:
                        continue

                    tensor = tokens_by_dihedral[0]
                    seq_len = seq_len_by_dihedral[0]
                    example = SequenceExample(
                        tokens=tensor,
                        example_id=example_id,
                        task_id=task_id,
                        split=split,
                        pair_index=pair_index,
                        has_output=has_output,
                        seq_len=seq_len,
                        tokens_by_dihedral=tokens_by_dihedral,
                        seq_len_by_dihedral=seq_len_by_dihedral,
                    )
                    self.indices_by_split.setdefault(split, []).append(
                        len(self.examples)
                    )
                    self.examples.append(example)
                    self.sequence_lengths.append(max(seq_len_by_dihedral))

        self.num_examples = len(self.task_id_to_example_id)

        print("Precomputing 3D positions...")
        for ex in self.examples:
            if ex.tokens_by_dihedral:
                cached_positions_by_dihedral: List[torch.Tensor] = []
                for tokens in ex.tokens_by_dihedral:
                    fake_batch = tokens.unsqueeze(0)
                    mask = torch.ones_like(fake_batch, dtype=torch.bool)
                    pos = compute_positions_3d(fake_batch, mask)
                    cached_positions_by_dihedral.append(pos.squeeze(0))
                ex.cached_positions_by_dihedral = cached_positions_by_dihedral
                ex.cached_positions = cached_positions_by_dihedral[0]
            else:
                fake_batch = ex.tokens.unsqueeze(0)
                mask = torch.ones_like(fake_batch, dtype=torch.bool)
                pos = compute_positions_3d(fake_batch, mask)
                ex.cached_positions = pos.squeeze(0)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SequenceExample:
        return self.examples[idx]

    def get_task_example_id(self, task_id: str) -> int:
        return self.task_id_to_example_id[task_id]

    def iter_examples(
        self, split: Optional[str] = None, has_output: Optional[bool] = None
    ) -> Iterable[SequenceExample]:
        for example in self.examples:
            if split is not None and example.split != split:
                continue
            if has_output is not None and example.has_output != has_output:
                continue
            yield example


# =============================================================================
# Batch Sampler and DataLoader
# =============================================================================

class LengthBucketBatchSampler(Sampler[List[int]]):
    """Group indices with similar sequence lengths to limit padding within a batch."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_size: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        bucket_size = bucket_size or batch_size * 4
        self.bucket_size = max(bucket_size, batch_size)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if not self.lengths:
            return iter(())

        if self.shuffle:
            indices = torch.randperm(len(self.lengths)).tolist()
        else:
            indices = sorted(
                range(len(self.lengths)),
                key=lambda idx: self.lengths[idx],
                reverse=True,
            )

        batches: List[List[int]] = []
        if self.shuffle:
            for start in range(0, len(indices), self.bucket_size):
                bucket = indices[start : start + self.bucket_size]
                bucket.sort(key=lambda idx: self.lengths[idx], reverse=True)
                for bucket_start in range(0, len(bucket), self.batch_size):
                    batch = bucket[bucket_start : bucket_start + self.batch_size]
                    if len(batch) == self.batch_size or not self.drop_last:
                        batches.append(batch)
            if len(batches) > 1:
                order = torch.randperm(len(batches)).tolist()
                batches = [batches[i] for i in order]
        else:
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        return iter(batches)


def collate_examples(
    batch: List[SequenceExample],
    pad_token_id: int = END_TOKEN_ID,
    augment_selector: Optional[
        Callable[[SequenceExample], Tuple[Optional[torch.Tensor], Optional[int]]]
    ] = None,
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Empty batch encountered during collation.")

    selected: List[
        Tuple[
            SequenceExample,
            torch.Tensor,
            Optional[torch.Tensor],
            int,
            Optional[torch.Tensor],
        ]
    ] = []
    for example in batch:
        tokens = example.tokens
        cached_positions = getattr(example, "cached_positions", None)
        mapping: Optional[torch.Tensor] = None
        transform_index: Optional[int] = None
        if augment_selector is not None:
            mapping, transform_index = augment_selector(example)
        if transform_index is not None:
            tokens_by_dihedral = getattr(example, "tokens_by_dihedral", None)
            if tokens_by_dihedral:
                if transform_index < 0 or transform_index >= len(tokens_by_dihedral):
                    raise ValueError(
                        f"Invalid dihedral index {transform_index} for example {example.task_id}."
                    )
                tokens = tokens_by_dihedral[transform_index]
                cached_by_dihedral = getattr(example, "cached_positions_by_dihedral", None)
                if cached_by_dihedral:
                    cached_positions = cached_by_dihedral[transform_index]
        seq_len = int(tokens.size(0))
        selected.append((example, tokens, cached_positions, seq_len, mapping))

    batch_size = len(selected)
    max_len = max(item[3] for item in selected)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    example_ids = torch.zeros(batch_size, dtype=torch.long)
    positions_3d = torch.zeros((batch_size, max_len, 3), dtype=torch.long)

    for idx, (example, tokens, cached_positions, seq_len, mapping) in enumerate(
        selected
    ):
        if mapping is not None:
            tokens = mapping[tokens]
        input_ids[idx, :seq_len] = tokens
        attention_mask[idx, :seq_len] = True
        example_ids[idx] = example.example_id
        if cached_positions is None:
            fake_batch = tokens.unsqueeze(0)
            mask = torch.ones_like(fake_batch, dtype=torch.bool)
            cached_positions = compute_positions_3d(fake_batch, mask).squeeze(0)
        positions_3d[idx, :seq_len] = cached_positions

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "example_ids": example_ids,
        "positions_3d": positions_3d,
        "task_ids": [example.task_id for example in batch],
        "splits": [example.split for example in batch],
        "has_output": [example.has_output for example in batch],
    }


def create_dataloader(
    dataset: ARCExampleDataset,
    batch_size: int,
    shuffle: bool = True,
    bucket_size_multiplier: int = 4,
    augment_selector: Optional[
        Callable[[SequenceExample], Tuple[Optional[torch.Tensor], Optional[int]]]
    ] = None,
) -> DataLoader:
    lengths = getattr(dataset, "sequence_lengths", None)
    if lengths is None:
        lengths = [len(dataset[i].tokens) for i in range(len(dataset))]

    bucket_size = max(batch_size * max(1, bucket_size_multiplier), batch_size)
    batch_sampler = LengthBucketBatchSampler(
        lengths=lengths, batch_size=batch_size, shuffle=shuffle, bucket_size=bucket_size
    )
    if augment_selector is not None:
        collate_fn = functools.partial(
            collate_examples,
            augment_selector=augment_selector,
        )
    else:
        collate_fn = collate_examples
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )


# =============================================================================
# RNG and Device Handling
# =============================================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    device_str = str(device_str or "cuda").lower()
    if not device_str.startswith("cuda"):
        raise ValueError("Only CUDA is supported. Set device='cuda'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device(device_str)


def capture_rng_state(device: torch.device) -> Dict[str, Any]:
    """Capture Python/numpy/torch RNG states so training can resume deterministically."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    state["torch"] = torch.get_rng_state()
    if torch.cuda.is_available() and device.type == "cuda":
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def restore_rng_state(state: Optional[Dict[str, Any]], device: torch.device) -> None:
    """Restore RNG state saved in a checkpoint; safe to call with None."""
    if not state:
        return
    try:
        random.setstate(state["python"])
    except Exception:
        pass
    try:
        np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        torch.set_rng_state(state["torch"])
    except Exception:
        pass
    if "cuda" in state and torch.cuda.is_available() and device.type == "cuda":
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass
