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


def _max_color_permutations(n_colors: int, k: int) -> int:
    if k <= 0 or n_colors <= 0 or k > n_colors:
        return 1
    return math.factorial(n_colors) // math.factorial(n_colors - k)


def _unrank_permutation(
    domain: Sequence[int], index: int, factorials: Sequence[int]
) -> Tuple[int, ...]:
    items = [int(c) for c in domain]
    result: List[int] = []
    idx = int(index)
    for size in range(len(items), 0, -1):
        fact = factorials[size - 1]
        pos = idx // fact
        idx %= fact
        result.append(int(items.pop(pos)))
    return tuple(result)


def _generate_task_permutations(
    domain: Sequence[int], max_permutations: int, rng: random.Random
) -> List[Tuple[int, ...]]:
    domain_list = [int(c) for c in domain]
    n = len(domain_list)
    if n == 0:
        return [tuple()]

    total = math.factorial(n)
    target = min(max(1, int(max_permutations)), total)
    if target == 1:
        return [tuple(domain_list)]

    factorials = [1]
    for i in range(1, n + 1):
        factorials.append(factorials[-1] * i)

    if target >= total:
        indices = list(range(total))
        rng.shuffle(indices)
    else:
        indices = rng.sample(range(1, total), target - 1)
        indices.append(0)
        rng.shuffle(indices)

    return [_unrank_permutation(domain_list, idx, factorials) for idx in indices]


def _mapping_from_permutation(
    domain: Sequence[int], perm: Sequence[int]
) -> List[int]:
    mapping = list(range(VOCAB_SIZE))
    for src, dst in zip(domain, perm):
        mapping[int(src)] = int(dst)
    return mapping


@dataclass
class Augments:
    color_maps: torch.Tensor
    dihedral_indices: List[int]
    color_map_indices: List[int]
    identity_tuple_index: int
    color_identity_indices: List[int]
    dihedral_identity_indices: List[int]
    order_seed: int = 0


class Augmentor:
    def __init__(
        self,
        augments_by_key: Dict[Tuple[str, str, int], Augments],
        *,
        color_apply_to_test_split: bool,
        dihedral_apply_to_test_split: bool,
        max_augments: int = 0,
    ) -> None:
        self.augments_by_key = augments_by_key
        self.color_apply_to_test_split = bool(color_apply_to_test_split)
        self.dihedral_apply_to_test_split = bool(dihedral_apply_to_test_split)
        self.max_augments = int(max_augments)
        self._enabled = True
        self._epoch = 0

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def _resolve_flags(self, split: str) -> Tuple[bool, bool]:
        if split == "test":
            return self.color_apply_to_test_split, self.dihedral_apply_to_test_split
        return True, True

    def _select_index_for_epoch(
        self, augments: Augments, *, color_enabled: bool, dihedral_enabled: bool
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


def build_augmentor(
    dataset: Iterable[SequenceExample],
    task_input_colors: Dict[str, Sequence[int]],
    *,
    max_augments: int,
    enable_color: bool,
    enable_dihedral: bool,
    seed: int,
    color_apply_to_test_split: bool,
    dihedral_apply_to_test_split: bool,
) -> Augmentor:
    rng = random.Random(int(seed))
    max_augments = int(max_augments)
    if max_augments <= 0:
        max_augments = 1
    max_dihedral = 8 if enable_dihedral else 1

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

    augments_by_key: Dict[Tuple[str, str, int], Augments] = {}
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
        task_input_set: Set[int] = set()
        for key in keys:
            task_input_set.update(input_colors_by_key[key])
        colors_a = sorted(task_input_set)
        if not allowed_set:
            allowed_set = set(colors_a)
        colors_b = sorted(allowed_set - set(colors_a))
        domain_colors = colors_a + colors_b
        identity_perm = tuple(domain_colors)

        identity_map = list(range(VOCAB_SIZE))
        sequence_states: Dict[Tuple[str, str, int], Dict[str, object]] = {}
        remaining = 0
        for key in keys:
            input_colors = input_colors_by_key[key]
            max_color_x = _max_color_permutations(len(domain_colors), len(input_colors))
            if enable_color:
                max_possible = max_color_x * max_dihedral
            else:
                max_possible = max_dihedral
            augment_limit = min(max_augments, max_possible)
            if augment_limit <= 0:
                augment_limit = 1
            order_seed = _derive_order_seed(seed, key)
            dihedral_order = list(range(max_dihedral))
            if len(dihedral_order) > 1:
                dihedral_order = _shuffled_indices(dihedral_order, order_seed)
            identity_signature = tuple(input_colors)
            state: Dict[str, object] = {
                "input_colors": input_colors,
                "input_tokens_by_dihedral": input_tokens_by_key[key],
                "color_maps": [identity_map],
                "dihedral_indices": [0],
                "color_map_indices": [0],
                "augment_limit": int(augment_limit),
                "augment_count": 1,
                "seen_signatures": {identity_signature},
                "seen_pairs": {(identity_signature, 0)},
                "mapping_index_by_key": {tuple(identity_map): 0},
                "mapping_signatures": [identity_signature],
                "dihedral_order": dihedral_order,
                "dihedral_cursor": 0,
                "order_seed": order_seed,
            }
            if state["augment_count"] < state["augment_limit"]:
                remaining += 1
            sequence_states[key] = state

        if enable_color and remaining > 0:
            # Task-level permutations over A+B (C stays fixed), filtered by per-sequence hashes.
            task_permutations = _generate_task_permutations(
                domain_colors, 5000, rng
            )
            for perm in task_permutations:
                if remaining == 0:
                    break
                if perm == identity_perm:
                    continue
                mapping = _mapping_from_permutation(domain_colors, perm)
                mapping_key = tuple(mapping)

                for key in keys:
                    state = sequence_states[key]
                    if state["augment_count"] >= state["augment_limit"]:
                        continue
                    input_colors = state["input_colors"]
                    signature = tuple(mapping[color] for color in input_colors)
                    if signature in state["seen_signatures"]:
                        continue
                    input_tokens_by_dihedral = state["input_tokens_by_dihedral"]
                    dihedral_order = state["dihedral_order"]
                    if not dihedral_order:
                        continue
                    dihedral_cursor = int(state["dihedral_cursor"])
                    order_len = len(dihedral_order)
                    selected_idx: Optional[int] = None
                    selected_hash: Optional[int] = None
                    for offset in range(order_len):
                        d_idx = int(
                            dihedral_order[(dihedral_cursor + offset) % order_len]
                        )
                        if (signature, d_idx) in state["seen_pairs"]:
                            continue
                        tokens = input_tokens_by_dihedral[d_idx]
                        mapped_tokens = [
                            mapping[tok] if 0 <= tok < len(mapping) else tok
                            for tok in tokens
                        ]
                        hashed = _hash_tokens(mapped_tokens)
                        state["seen_pairs"].add((signature, d_idx))
                        if hashed in seen_hashes:
                            continue
                        selected_idx = d_idx
                        selected_hash = hashed
                        state["dihedral_cursor"] = (
                            dihedral_cursor + offset + 1
                        ) % order_len
                        break

                    if selected_idx is None or selected_hash is None:
                        state["seen_signatures"].add(signature)
                        continue

                    map_idx = state["mapping_index_by_key"].get(mapping_key)
                    if map_idx is None:
                        map_idx = len(state["color_maps"])
                        state["color_maps"].append(mapping)
                        state["mapping_index_by_key"][mapping_key] = map_idx
                        state["mapping_signatures"].append(signature)
                    state["dihedral_indices"].append(int(selected_idx))
                    state["color_map_indices"].append(int(map_idx))
                    state["augment_count"] += 1
                    state["seen_signatures"].add(signature)
                    seen_hashes.add(selected_hash)
                    if state["augment_count"] >= state["augment_limit"]:
                        remaining -= 1
                        if remaining == 0:
                            break

        if enable_dihedral and remaining > 0:
            for pass_idx in range(max_dihedral):
                if remaining == 0:
                    break
                for key in keys:
                    state = sequence_states[key]
                    if state["augment_count"] >= state["augment_limit"]:
                        continue
                    input_tokens_by_dihedral = state["input_tokens_by_dihedral"]
                    dihedral_order = list(range(max_dihedral))
                    if len(dihedral_order) > 1:
                        dihedral_order = _shuffled_indices(
                            dihedral_order,
                            _cycle_seed(state["order_seed"], 3, pass_idx),
                        )
                    for map_idx, signature in enumerate(state["mapping_signatures"]):
                        if state["augment_count"] >= state["augment_limit"]:
                            break
                        mapping = state["color_maps"][map_idx]
                        for d in dihedral_order:
                            if state["augment_count"] >= state["augment_limit"]:
                                break
                            if (signature, d) in state["seen_pairs"]:
                                continue
                            tokens = input_tokens_by_dihedral[d]
                            mapped_tokens = [
                                mapping[tok] if 0 <= tok < len(mapping) else tok
                                for tok in tokens
                            ]
                            hashed = _hash_tokens(mapped_tokens)
                            state["seen_pairs"].add((signature, d))
                            if hashed in seen_hashes:
                                continue
                            state["dihedral_indices"].append(int(d))
                            state["color_map_indices"].append(int(map_idx))
                            state["augment_count"] += 1
                            seen_hashes.add(hashed)
                            if state["augment_count"] >= state["augment_limit"]:
                                remaining -= 1
                            break
                    if remaining == 0:
                        break

        for key in keys:
            state = sequence_states[key]
            color_maps_tensor = torch.tensor(state["color_maps"], dtype=torch.long)
            color_identity_indices = [
                idx for idx, c in enumerate(state["color_map_indices"]) if c == 0
            ]
            dihedral_identity_indices = [
                idx for idx, d in enumerate(state["dihedral_indices"]) if d == 0
            ]

            augments = Augments(
                color_maps=color_maps_tensor,
                dihedral_indices=state["dihedral_indices"],
                color_map_indices=state["color_map_indices"],
                identity_tuple_index=0,
                color_identity_indices=color_identity_indices,
                dihedral_identity_indices=dihedral_identity_indices,
                order_seed=_derive_order_seed(seed, key),
            )
            augments_by_key[key] = augments
            stats.append(len(state["dihedral_indices"]))

    if stats:
        avg = sum(stats) / len(stats)
        print(
            "Augmentation stats: "
            f"{len(stats)} sequences, avg augments={avg:.1f}, "
            f"min={min(stats)}, max={max(stats)}"
        )

    return Augmentor(
        augments_by_key,
        color_apply_to_test_split=color_apply_to_test_split,
        dihedral_apply_to_test_split=dihedral_apply_to_test_split,
        max_augments=max_augments,
    )
