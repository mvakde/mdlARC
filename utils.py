import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

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


def compute_positions_3d(
    input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute per-token 3D grid coordinates on CPU.

    Expects 2D input tensors shaped [batch, seq_len]; padding should be
    indicated via `attention_mask`.
    """
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape [batch, seq_len].")

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)

    ids_cpu = input_ids.detach().cpu()
    mask_cpu = attention_mask.detach().cpu()
    B, S = ids_cpu.shape
    pos = torch.zeros((B, S, 3), dtype=torch.long)

    for b in range(B):
        x = 0
        y = 0
        z = 1
        for t in range(S):
            if not mask_cpu[b, t]:
                continue
            tok = int(ids_cpu[b, t].item())
            if t == 0 and tok == START_TOKEN_ID:
                pos[b, t, 0] = 0
                pos[b, t, 1] = 0
                pos[b, t, 2] = 0
                continue

            if tok == IO_SEPARATOR_TOKEN_ID:
                pos[b, t, 0] = 0
                pos[b, t, 1] = 0
                pos[b, t, 2] = 2
                x, y = 0, 0
                z = 3
                continue

            if tok == END_TOKEN_ID:
                pos[b, t, 0] = 0
                pos[b, t, 1] = 0
                pos[b, t, 2] = 4
                continue

            px = min(max(x, 0), 30)
            py = min(max(y, 0), 29)
            pos[b, t, 0] = px
            pos[b, t, 1] = py
            pos[b, t, 2] = z

            if tok == NEXT_LINE_TOKEN_ID:
                x = 0
                y += 1
            else:
                x += 1

    return pos


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

        for example_id, task_id in enumerate(task_ids):
            self.task_id_to_example_id[task_id] = example_id
            task = challenges[task_id]
            for split in splits:
                pairs = task.get(split, [])
                for pair_index, pair in enumerate(pairs):
                    has_output = "output" in pair and pair["output"] is not None
                    include_output_tokens = include_outputs and has_output
                    append_end = include_output_tokens
                    tokens = encode_example(
                        pair["input"],
                        pair.get("output"),
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
                    example = SequenceExample(
                        tokens=tensor,
                        example_id=example_id,
                        task_id=task_id,
                        split=split,
                        pair_index=pair_index,
                        has_output=has_output,
                    )
                    self.indices_by_split.setdefault(split, []).append(
                        len(self.examples)
                    )
                    self.examples.append(example)

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


def collate_examples(
    batch: List[SequenceExample], pad_token_id: int = END_TOKEN_ID
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Empty batch encountered during collation.")

    batch_size = len(batch)
    max_len = max(example.tokens.size(0) for example in batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    example_ids = torch.zeros(batch_size, dtype=torch.long)
    positions_3d = torch.zeros((batch_size, max_len, 3), dtype=torch.long)

    for idx, example in enumerate(batch):
        seq_len = example.tokens.size(0)
        input_ids[idx, :seq_len] = example.tokens
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
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_examples,
    )
