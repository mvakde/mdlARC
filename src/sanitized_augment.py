from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from utils import IO_SEPARATOR_TOKEN_ID, VOCAB_SIZE, SequenceExample


def _extract_input_tokens(tokens: Sequence[int]) -> List[int]:
    input_tokens: List[int] = []
    for tok in tokens:
        val = int(tok)
        if val == IO_SEPARATOR_TOKEN_ID:
            break
        input_tokens.append(val)
    return input_tokens


def _hash_tokens(tokens: Sequence[int]) -> int:
    digest = hashlib.sha256(bytes(tokens)).digest()
    return int.from_bytes(digest[:8], "little")


def _derive_order_seed(seed: int, key: Tuple[str, str, int]) -> int:
    payload = f"{int(seed)}|{key[0]}|{key[1]}|{int(key[2])}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _cycle_seed(order_seed: int, mode: int, cycle: int) -> int:
    payload = f"{int(order_seed)}|{int(mode)}|{int(cycle)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _shuffled_indices(indices: Sequence[int], seed: int) -> List[int]:
    order = list(indices)
    rng = random.Random(int(seed))
    rng.shuffle(order)
    return order


def _index_for_epoch(
    order_seed: int, indices: Sequence[int], epoch: int, mode: int
) -> int:
    if len(indices) == 1:
        return int(indices[0])
    epoch_i = max(0, int(epoch))
    cycle_len = len(indices)
    cycle = epoch_i // cycle_len
    offset = epoch_i % cycle_len
    order = _shuffled_indices(indices, _cycle_seed(order_seed, mode, cycle))
    if cycle > 0:
        prev_order = _shuffled_indices(
            indices, _cycle_seed(order_seed, mode, cycle - 1)
        )
        if order == prev_order:
            order = order[1:] + order[:1]
    return int(order[offset])


def _colors_from_tokens(tokens: Sequence[int]) -> List[int]:
    return sorted({int(tok) for tok in tokens if 1 <= int(tok) <= 9})


def _max_injections(n_a: int, n_b: int) -> int:
    return math.factorial(n_a + n_b) // math.factorial(n_b)


def _sample_injections(
    colors_a: Sequence[int],
    colors_b: Sequence[int],
    target_count: int,
    rng: random.Random,
) -> List[Tuple[int, ...]]:
    n_a = len(colors_a)
    if n_a == 0:
        return [tuple()]

    max_count = _max_injections(n_a, len(colors_b))
    target_count = min(int(target_count), max_count)
    identity = tuple(int(c) for c in colors_a)
    injections: List[Tuple[int, ...]] = [identity]
    seen = {identity}
    if target_count <= 1:
        return injections

    pool = [int(c) for c in list(colors_a) + list(colors_b)]
    max_attempts = max(100, target_count * 50)
    attempts = 0
    while len(injections) < target_count and attempts < max_attempts:
        attempts += 1
        rng.shuffle(pool)
        target = tuple(pool[:n_a])
        if target in seen:
            continue
        seen.add(target)
        injections.append(target)
    return injections


def _build_color_mapping(
    colors_a: Sequence[int], colors_b: Sequence[int], targets: Sequence[int]
) -> List[int]:
    mapping = list(range(VOCAB_SIZE))
    target_to_source: Dict[int, int] = {}
    for src, dst in zip(colors_a, targets):
        src_i = int(src)
        dst_i = int(dst)
        mapping[src_i] = dst_i
        target_to_source[dst_i] = src_i
    for color in colors_b:
        color_i = int(color)
        mapping[color_i] = target_to_source.get(color_i, color_i)
    return mapping


@dataclass
class SanitizedAugments:
    color_maps: torch.Tensor
    dihedral_indices: List[int]
    color_map_indices: List[int]
    identity_tuple_index: int
    color_identity_indices: List[int]
    dihedral_identity_indices: List[int]
    order_seed: int = 0


class SanitizedAugmentor:
    def __init__(
        self,
        augments_by_key: Dict[Tuple[str, str, int], SanitizedAugments],
        *,
        color_apply_to_test_split: bool,
        dihedral_apply_to_test_split: bool,
    ) -> None:
        self.augments_by_key = augments_by_key
        self.color_apply_to_test_split = bool(color_apply_to_test_split)
        self.dihedral_apply_to_test_split = bool(dihedral_apply_to_test_split)
        self._enabled = True
        self._color_enabled = True
        self._dihedral_enabled = True
        self._epoch = 0

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def set_color_enabled(self, enabled: bool) -> None:
        self._color_enabled = bool(enabled)

    def set_dihedral_enabled(self, enabled: bool) -> None:
        self._dihedral_enabled = bool(enabled)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def _resolve_flags(self, split: str) -> Tuple[bool, bool]:
        color_enabled = self._color_enabled
        dihedral_enabled = self._dihedral_enabled
        if split == "test":
            color_enabled = color_enabled and self.color_apply_to_test_split
            dihedral_enabled = dihedral_enabled and self.dihedral_apply_to_test_split
        return color_enabled, dihedral_enabled

    def _select_index_for_epoch(
        self, augments: SanitizedAugments, *, color_enabled: bool, dihedral_enabled: bool
    ) -> int:
        if not color_enabled and not dihedral_enabled:
            return augments.identity_tuple_index
        if not color_enabled:
            candidates = augments.color_identity_indices
        elif not dihedral_enabled:
            candidates = augments.dihedral_identity_indices
        else:
            candidates = list(range(len(augments.dihedral_indices)))
        if not candidates:
            return augments.identity_tuple_index
        if len(candidates) == 1:
            return candidates[0]
        mode = (1 if color_enabled else 0) | (2 if dihedral_enabled else 0)
        return _index_for_epoch(augments.order_seed, candidates, self._epoch, mode)

    def select_for_example(
        self, example: SequenceExample
    ) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        if not self._enabled:
            return None, None
        key = (example.task_id, example.split, example.pair_index)
        augments = self.augments_by_key.get(key)
        if augments is None:
            return None, None

        color_enabled, dihedral_enabled = self._resolve_flags(example.split)
        idx = self._select_index_for_epoch(
            augments, color_enabled=color_enabled, dihedral_enabled=dihedral_enabled
        )

        color_idx = augments.color_map_indices[idx]
        dihedral_idx = augments.dihedral_indices[idx]
        mapping = augments.color_maps[color_idx]
        return mapping, dihedral_idx


def build_sanitized_augmentor(
    dataset: Iterable[SequenceExample],
    task_input_colors: Dict[str, Sequence[int]],
    *,
    max_color_augments: int,
    enable_color: bool,
    enable_dihedral: bool,
    seed: int,
    color_apply_to_test_split: bool,
    dihedral_apply_to_test_split: bool,
) -> SanitizedAugmentor:
    rng = random.Random(int(seed))
    max_color_augments = int(max_color_augments)
    if max_color_augments <= 0:
        max_color_augments = 1
    if not enable_color:
        max_color_augments = 1

    dihedral_count = 8 if enable_dihedral else 1
    max_total_augments = max_color_augments * dihedral_count

    examples_by_task: Dict[str, List[Tuple[str, str, int]]] = {}
    input_tokens_by_key: Dict[Tuple[str, str, int], List[List[int]]] = {}
    input_colors_by_key: Dict[Tuple[str, str, int], List[int]] = {}

    for example in dataset:
        key = (example.task_id, example.split, example.pair_index)
        tokens_by_dihedral = example.tokens_by_dihedral or [example.tokens]
        num_dihedral = 8 if enable_dihedral else 1
        input_tokens_by_dihedral: List[List[int]] = []
        for d in range(num_dihedral):
            tokens = tokens_by_dihedral[d]
            tokens_list = (
                tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
            )
            input_tokens_by_dihedral.append(_extract_input_tokens(tokens_list))
        input_tokens_by_key[key] = input_tokens_by_dihedral
        input_colors_by_key[key] = _colors_from_tokens(input_tokens_by_dihedral[0])
        examples_by_task.setdefault(example.task_id, []).append(key)

    augments_by_key: Dict[Tuple[str, str, int], SanitizedAugments] = {}
    stats: List[int] = []

    for task_id, keys in examples_by_task.items():
        seen_hashes: Set[int] = set()
        for key in keys:
            identity_tokens = input_tokens_by_key[key][0]
            seen_hashes.add(_hash_tokens(identity_tokens))

        allowed_colors = [
            int(c) for c in task_input_colors.get(task_id, []) if 1 <= int(c) <= 9
        ]
        allowed_set = set(allowed_colors)

        for key in keys:
            colors_a = input_colors_by_key[key]
            colors_b = sorted(allowed_set - set(colors_a))
            target_injections = _sample_injections(
                colors_a, colors_b, max_color_augments, rng
            )

            identity_map = list(range(VOCAB_SIZE))
            color_maps: List[List[int]] = [identity_map]
            color_map_lookup = {tuple(identity_map): 0}

            dihedral_indices: List[int] = [0]
            color_map_indices: List[int] = [0]

            dihedral_range = range(8) if enable_dihedral else range(1)
            input_tokens_by_dihedral = input_tokens_by_key[key]

            for targets in target_injections:
                mapping = _build_color_mapping(colors_a, colors_b, targets)
                mapping_key = tuple(mapping)
                map_idx = color_map_lookup.get(mapping_key)
                if map_idx is None:
                    map_idx = len(color_maps)
                    color_map_lookup[mapping_key] = map_idx
                    color_maps.append(mapping)

                for d in dihedral_range:
                    if len(dihedral_indices) >= max_total_augments:
                        break
                    tokens = input_tokens_by_dihedral[d]
                    mapped_tokens = [
                        mapping[tok] if 0 <= tok < len(mapping) else tok
                        for tok in tokens
                    ]
                    hashed = _hash_tokens(mapped_tokens)
                    if hashed in seen_hashes:
                        continue
                    dihedral_indices.append(int(d))
                    color_map_indices.append(int(map_idx))
                    seen_hashes.add(hashed)
                if len(dihedral_indices) >= max_total_augments:
                    break

            color_maps_tensor = torch.tensor(color_maps, dtype=torch.long)
            color_identity_indices = [
                idx for idx, c in enumerate(color_map_indices) if c == 0
            ]
            dihedral_identity_indices = [
                idx for idx, d in enumerate(dihedral_indices) if d == 0
            ]

            augments = SanitizedAugments(
                color_maps=color_maps_tensor,
                dihedral_indices=dihedral_indices,
                color_map_indices=color_map_indices,
                identity_tuple_index=0,
                color_identity_indices=color_identity_indices,
                dihedral_identity_indices=dihedral_identity_indices,
                order_seed=_derive_order_seed(seed, key),
            )
            augments_by_key[key] = augments
            stats.append(len(dihedral_indices))

    if stats:
        avg = sum(stats) / len(stats)
        print(
            "Sanitized augmentation: "
            f"{len(stats)} sequences, avg augments={avg:.1f}, "
            f"min={min(stats)}, max={max(stats)}"
        )

    return SanitizedAugmentor(
        augments_by_key,
        color_apply_to_test_split=color_apply_to_test_split,
        dihedral_apply_to_test_split=dihedral_apply_to_test_split,
    )
