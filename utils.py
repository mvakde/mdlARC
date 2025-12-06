import json
import random
import math
import functools
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from numba import njit

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except Exception:  # pragma: no cover
    plt = None
    ListedColormap = None

# Vocabulary definitions ----------------------------------------------------

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


def generate_color_permutations(
    max_permutations: int, seed: int
) -> List[Tuple[int, ...]]:
    """Return up to `max_permutations` unique shuffles of colors 1-9.

    Identity permutation is always included first (index 0 / permutation 1).
    """
    if max_permutations <= 0:
        return []
    rng = random.Random(seed)
    digits = list(range(1, 10))
    identity = tuple(digits)
    permutations: List[Tuple[int, ...]] = [identity]
    seen = {identity}
    limit = math.factorial(9)
    target = min(max_permutations, limit)
    if target == 1:
        return permutations

    if target == limit:
        # Generate all, shuffle, then force identity to stay first.
        all_perms = list(itertools.permutations(digits))
        rng.shuffle(all_perms)
        deduped = [identity]
        for perm in all_perms:
            if perm == identity:
                continue
            deduped.append(perm)
        return deduped[:target]

    while len(permutations) < target:
        perm = tuple(rng.sample(digits, len(digits)))
        if perm in seen:
            continue
        seen.add(perm)
        permutations.append(perm)
    return permutations


def color_permutation_to_mapping(perm: Sequence[int]) -> torch.Tensor:
    """Build a token-id mapping tensor for a specific color permutation."""
    mapping = torch.arange(VOCAB_SIZE, dtype=torch.long)
    mapping[1:10] = torch.tensor(list(perm), dtype=torch.long)
    return mapping


def generate_color_mapping_tensors(
    max_permutations: int, seed: int
) -> List[torch.Tensor]:
    perms = generate_color_permutations(max_permutations, seed)
    return [color_permutation_to_mapping(perm) for perm in perms]


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


class ColorAugmentor:
    """Holds a deterministic list of color mappings and exposes epoch-based selection."""

    def __init__(
        self, mappings: Sequence[torch.Tensor], apply_to_test_split: bool = False
    ) -> None:
        self.mappings = list(mappings)
        self.apply_to_test_split = apply_to_test_split
        self._index = 0

    @property
    def num_permutations(self) -> int:
        return len(self.mappings)

    @property
    def current_index(self) -> int:
        if self.num_permutations == 0:
            return 0
        return self._index % self.num_permutations

    def set_index(self, index: int) -> None:
        if self.num_permutations == 0:
            return
        self._index = max(0, int(index))

    def mapping_for_split(self, split: str) -> Optional[torch.Tensor]:
        if not self.mappings:
            return None
        if split == "test" and not self.apply_to_test_split:
            return None
        idx = self._index % self.num_permutations
        return self.mappings[idx]


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
            # Stop decoding when we hit an unexpected special token.
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
    """Convert a token id sequence into a space-delimited string.

    - Digits 0-9 remain as their numeric character.
    - Special tokens use their literal names, e.g. "<start>".
    """
    parts: List[str] = []
    for tok in tokens:
        parts.append(ID_TO_TOKEN.get(int(tok), str(int(tok))))
    return " ".join(parts)


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
    """Compute per-token 3D grid coordinates on CPU. (using numba)

    Expects 2D input tensors shaped [batch, seq_len]; padding should be
    indicated via `attention_mask`.
    """
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape [batch, seq_len].")

    # Convert inputs to numpy for Numba
    ids_cpu = input_ids.detach().cpu()
    ids_np = ids_cpu.numpy()

    if attention_mask is None:
        mask_np = np.ones_like(ids_np, dtype=bool)
    else:
        mask_np = attention_mask.detach().cpu().numpy().astype(bool)

    B, S = ids_np.shape
    # Pre-allocate output array
    pos_np = np.zeros((B, S, 3), dtype=np.int64)

    # Call the JIT-compiled helper
    _fill_3d_positions_numba(
        ids_np,
        mask_np,
        pos_np,
        START_TOKEN_ID,
        IO_SEPARATOR_TOKEN_ID,
        END_TOKEN_ID,
        NEXT_LINE_TOKEN_ID,
    )

    # Convert back to torch tensor and move to original device
    return torch.from_numpy(pos_np).to(device=input_ids.device)


def split_grids_from_tokens(tokens: Sequence[int]) -> List[List[List[int]]]:
    """Split a token sequence into multiple grids.

    Rules:
    - Digits 0-9 map to cell values in the current row.
    - <next_line> ends the current row and starts a new row.
    - <input_output_separator> closes the current grid and starts a new one.
    - <start> is ignored; <end> stops parsing.
    """
    grids: List[List[List[int]]] = []
    current_grid: List[List[int]] = []
    current_row: List[int] = []

    for tok in tokens:
        if tok == END_TOKEN_ID:
            break
        if tok == START_TOKEN_ID:
            # ignore start markers
            continue
        if tok == IO_SEPARATOR_TOKEN_ID:
            # close current row if non-empty
            if current_row:
                current_grid.append(current_row)
                current_row = []
            # close current grid if non-empty
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
            # unknown special; stop
            break

    if current_row:
        current_grid.append(current_row)
    if current_grid:
        grids.append(current_grid)
    return grids


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


def is_rectangular_grid(grid: Sequence[Sequence[int]]) -> bool:
    """Return True if all rows have the same non-zero length."""
    if not grid:
        return False
    first_row_len = len(grid[0])
    if first_row_len == 0:
        return False
    return all(len(row) == first_row_len for row in grid)


def apply_inverse_dihedral_transform(
    grid: Sequence[Sequence[int]], transform_index: int
) -> List[List[int]]:
    """Undo a dihedral transform using the known augmentation index (mod 8)."""
    if transform_index < 0:
        raise ValueError("transform_index must be non-negative.")
    transform_name = _DIHEDRAL_TRANSFORM_NAMES[transform_index % 8]
    inverse_name = _DIHEDRAL_INVERSES[transform_name]
    return _DIHEDRAL_TRANSFORMS[inverse_name](grid)


def _grid_to_tuple(grid: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(val) for val in row) for row in grid)


def _tuple_to_grid(grid_tuple: Tuple[Tuple[int, ...], ...]) -> List[List[int]]:
    return [list(row) for row in grid_tuple]


@dataclass
class AAIVRSelection:
    task_id: str
    original_pair_index: int
    selected_outputs: List[List[List[int]]]
    ranked_candidates: List[Dict[str, object]]
    num_generated: int
    num_valid: int
    discarded_non_rectangular: int
    discarded_input_copies: int
    target_grid: Optional[List[List[int]]] = None
    pass_at_k: Optional[bool] = None


def run_aaivr_on_results(
    results: Sequence[Dict[str, object]],
    top_k: int = 2,
    discard_input_copies: bool = True,
    rng: Optional[random.Random] = None,
) -> List[AAIVRSelection]:
    """Aggregate augmented predictions via AAIVR voting. (automated augmentation inverse)

    The function assumes pair_index encodes augmentation order (mod 8). It
    returns up to `top_k` most common inverse-transformed outputs per
    original test input.
    """
    rng = rng if rng is not None else random
    case_map: Dict[Tuple[str, int], Dict[str, object]] = {}

    for res in results:
        task_id = res.get("task_id")
        pair_index = res.get("pair_index")
        if task_id is None or pair_index is None:
            continue

        base_pair_index = int(pair_index) // 8
        transform_index = int(pair_index) % 8
        predicted_grid = res.get("output_grid", [])
        prompt_tokens = res.get("prompt_tokens", [])
        input_grids = split_grids_from_tokens(prompt_tokens)
        input_grid = input_grids[0] if input_grids else []

        key = (task_id, base_pair_index)
        if key not in case_map:
            case_map[key] = {
                "counts": {},
                "generated": 0,
                "valid": 0,
                "dropped_rect": 0,
                "dropped_input": 0,
                "target_grid": None,
            }
        stats = case_map[key]
        stats["generated"] += 1

        if not is_rectangular_grid(predicted_grid):
            stats["dropped_rect"] += 1
            continue
        if discard_input_copies and input_grid and predicted_grid == input_grid:
            stats["dropped_input"] += 1
            continue

        try:
            normalized_grid = apply_inverse_dihedral_transform(
                predicted_grid, transform_index
            )
        except Exception:
            stats["dropped_rect"] += 1
            continue

        if not is_rectangular_grid(normalized_grid):
            stats["dropped_rect"] += 1
            continue

        stats["valid"] += 1
        grid_key = _grid_to_tuple(normalized_grid)
        counts: Dict[Tuple[Tuple[int, ...], ...], int] = stats["counts"]
        counts[grid_key] = counts.get(grid_key, 0) + 1

        target_grid = res.get("target_grid", [])
        if stats["target_grid"] is None and is_rectangular_grid(target_grid):
            normalized_target = apply_inverse_dihedral_transform(
                target_grid, transform_index
            )
            stats["target_grid"] = (
                normalized_target if is_rectangular_grid(normalized_target) else None
            )

    selections: List[AAIVRSelection] = []
    for (task_id, base_idx), stats in sorted(case_map.items()):
        items = list(stats["counts"].items())
        if items:
            rng.shuffle(items)  # tie-break randomly before sorting by count
            items.sort(key=lambda pair: pair[1], reverse=True)
        ranked_candidates = [
            {"grid": _tuple_to_grid(grid_key), "count": count}
            for grid_key, count in items
        ]
        selected_outputs = [entry["grid"] for entry in ranked_candidates[:top_k]]

        target_grid = stats.get("target_grid")
        pass_at_k = None
        if target_grid is not None:
            pass_at_k = any(grid == target_grid for grid in selected_outputs)

        selections.append(
            AAIVRSelection(
                task_id=task_id,
                original_pair_index=base_idx,
                selected_outputs=selected_outputs,
                ranked_candidates=ranked_candidates,
                num_generated=stats["generated"],
                num_valid=stats["valid"],
                discarded_non_rectangular=stats["dropped_rect"],
                discarded_input_copies=stats["dropped_input"],
                target_grid=target_grid,
                pass_at_k=pass_at_k,
            )
        )

    return selections


def summarize_aaivr_pass_at_k(selections: Sequence[AAIVRSelection]) -> Dict[str, int]:
    """Return counts for how many selections have the target in top-k."""
    evaluated = [sel for sel in selections if sel.pass_at_k is not None]
    hits = [sel for sel in evaluated if sel.pass_at_k]
    return {"evaluated": len(evaluated), "hits": len(hits)}


DEFAULT_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 fuchsia
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 aqua
    "#B10DC9",  # 9 purple
]


def plot_grids(
    grids: List[List[List[int]]], title: Optional[str] = None, figsize=(4, 4)
) -> None:
    """Plot one or more integer grids using a fixed 10-color palette.

    Each grid is shown as a separate subplot in a single row.
    """
    if plt is None or ListedColormap is None:
        raise RuntimeError("matplotlib is not available in this environment.")

    n = max(1, len(grids))
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
    if n == 1:
        axes = [axes]
    cmap = ListedColormap(DEFAULT_COLORS)

    for ax, grid in zip(axes, grids):
        if not grid:
            ax.axis("off")
            continue
        arr = np.array(grid, dtype=int)
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


@dataclass
class SequenceExample:
    tokens: torch.LongTensor
    example_id: int
    task_id: str
    split: str
    pair_index: int
    has_output: bool
    seq_len: int


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

        for example_id, task_id in enumerate(task_ids):
            self.task_id_to_example_id[task_id] = example_id
            task = challenges[task_id]
            for split in splits:
                pairs = task.get(split, [])
                for pair_index, pair in enumerate(pairs):
                    input_grid = pair["input"]
                    output_grid = pair.get(
                        "output"
                    )  # Valid for 'train', usually None for 'test'

                    # Explicitly fetch test outputs from solutions_map if allowed
                    if split == "test" and load_test_solutions:
                        if task_id in solutions_map:
                            task_sols = solutions_map[task_id]
                            if pair_index < len(task_sols):
                                output_grid = task_sols[pair_index]

                    # Standard logic follows
                    has_output = output_grid is not None
                    include_output_tokens = include_outputs and has_output
                    append_end = include_output_tokens

                    tokens = encode_example(
                        input_grid,
                        output_grid,
                        include_output=include_output_tokens,
                        append_end=append_end,
                    )
                    if len(tokens) > max_seq_len:
                        if drop_long_sequences:
                            continue
                        raise ValueError(
                            f"Sequence length {len(tokens)} exceeds max_seq_len={max_seq_len} "
                            f"for task {task_id} ({split} pair {pair_index})."
                        )
                    tensor = torch.tensor(tokens, dtype=torch.long)
                    seq_len = len(tokens)
                    example = SequenceExample(
                        tokens=tensor,
                        example_id=example_id,
                        task_id=task_id,
                        split=split,
                        pair_index=pair_index,
                        has_output=has_output,
                        seq_len=seq_len,
                    )
                    self.indices_by_split.setdefault(split, []).append(
                        len(self.examples)
                    )
                    self.examples.append(example)
                    self.sequence_lengths.append(seq_len)

        self.num_examples = len(self.task_id_to_example_id)

        print("Precomputing 3D positions...")
        for ex in self.examples:
            # We treat a single example as a batch of 1 to reuse your existing function
            # or refactor the function to handle 1D tensors.
            # Using your existing function for minimal code changes:
            fake_batch = ex.tokens.unsqueeze(0)  # [1, seq_len]
            mask = torch.ones_like(fake_batch, dtype=torch.bool)

            # This is slow, but it only happens ONCE during startup
            pos = compute_positions_3d(fake_batch, mask)

            # Store the result (remove batch dim)
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
        # Bucket size controls how tightly we cluster similar lengths before batching.
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
            # Within each bucket, sort by length so batches group similar sequence sizes.
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
    color_mapper: Optional[Callable[[str], Optional[torch.Tensor]]] = None,
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Empty batch encountered during collation.")

    batch_size = len(batch)
    max_len = max(example.seq_len for example in batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    example_ids = torch.zeros(batch_size, dtype=torch.long)
    positions_3d = torch.zeros((batch_size, max_len, 3), dtype=torch.long)

    for idx, example in enumerate(batch):
        seq_len = example.seq_len
        tokens = example.tokens
        if color_mapper is not None:
            mapping = color_mapper(example.split)
            if mapping is not None:
                tokens = mapping[tokens]
        input_ids[idx, :seq_len] = tokens
        attention_mask[idx, :seq_len] = True
        example_ids[idx] = example.example_id
        positions_3d[idx, :seq_len] = example.cached_positions

    # positions_3d = compute_positions_3d(input_ids, attention_mask)

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
    num_workers: int = 0,
    bucket_size_multiplier: int = 4,
    color_mapper: Optional[Callable[[str], Optional[torch.Tensor]]] = None,
) -> DataLoader:
    lengths = getattr(dataset, "sequence_lengths", None)
    if lengths is None:
        lengths = [len(dataset[i].tokens) for i in range(len(dataset))]

    bucket_size = max(batch_size * max(1, bucket_size_multiplier), batch_size)
    batch_sampler = LengthBucketBatchSampler(
        lengths=lengths, batch_size=batch_size, shuffle=shuffle, bucket_size=bucket_size
    )
    collate_fn = (
        functools.partial(collate_examples, color_mapper=color_mapper)
        if color_mapper is not None
        else collate_examples
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
