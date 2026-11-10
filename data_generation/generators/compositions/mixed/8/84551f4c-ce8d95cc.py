"""Exact-contact rod closure followed by consecutive-vector contraction.

No files, parent modules, training examples, or generation state are used by solve.
Red (1) is the fixed trigger; the constant top row identifies the background.
"""
from __future__ import annotations

from collections import deque
import hashlib

import numpy as np


HOST_SEEDS = (0, 3414688, 125962335, 420589976, 749987450, 435026478,
              923469552, 988924193, 499062964, 702493815, 277071981,
              438462053, 398354155, 680083130, 624463908, 285673441,
              42450400, 440420718, 138018078, 227432197, 200513855,
              794718156, 615996047, 864023210, 332583544, 533419494,
              79531308, 911680060, 185658946, 677708329, 2203950,
              454515827, 372516044, 338148786, 561742554, 241942305,
              357040137, 262258458, 702191607, 600004128, 660266385,
              322926927)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _parse(value):
    a = _grid(value)
    background = int(a[0, 0])
    if background == 1 or not np.all(a[0] == background):
        raise ValueError("A blank top row must identify a non-red background")
    rods = []
    for x in range(a.shape[1]):
        rows = np.flatnonzero(a[:, x] != background)
        if not len(rows):
            continue
        h = len(rows)
        color = int(a[-1, x])
        if (h < 2 or not np.array_equal(rows, np.arange(a.shape[0] - h, a.shape[0]))
                or not np.all(a[rows, x] == color) or x + h > a.shape[1]):
            raise ValueError("Rods must be monochromatic, bottom aligned, and fit when fallen")
        if rods and x < rods[-1][0] + rods[-1][1]:
            raise ValueError("A rod may not lie inside another rod's landing interval")
        rods.append((x, h, color))
    if not rods or not any(c == 1 for _, _, c in rods):
        raise ValueError("At least one red trigger rod is required")
    return a, background, rods


def _active(rods, *, one_hop=False, bridge_gap=False, starts=None):
    lookup = {x: i for i, (x, _, _) in enumerate(rods)}
    active = ({i for i, (_, _, c) in enumerate(rods) if c == 1}
              if starts is None else set(starts))
    queue = deque(sorted(active))
    while queue:
        i = queue.popleft()
        x, h, _ = rods[i]
        for target in ((x + h, x + h + 1) if bridge_gap else (x + h,)):
            j = lookup.get(target)
            if j is not None and j not in active:
                active.add(j)
                if not one_hop:
                    queue.append(j)
    return active


def _land(a, background, rods, active):
    canvas = a.copy()
    for i in sorted(active):
        x, h, _ = rods[i]
        canvas[-h:, x] = background
    for i in sorted(active):
        x, h, c = rods[i]
        canvas[-1, x:x + h] = c
    return canvas


def _compress(a):
    """Retain first vectors of adjacent equality runs, including colored runs."""
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    reduced = a[rows]
    columns = np.r_[True, np.any(reduced[:, 1:] != reduced[:, :-1], axis=0)]
    return reduced[:, columns].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, background, rods = _parse(input_grid)
    return _compress(_land(a, background, rods, _active(rods)))


def _longest_chain(rods):
    lookup = {x: i for i, (x, _, _) in enumerate(rods)}
    longest = 0
    for i, (_, _, c) in enumerate(rods):
        if c != 1:
            continue
        length = 0
        while i is not None:
            length += 1
            x, h, _ = rods[i]
            i = lookup.get(x + h)
        longest = max(longest, length)
    return longest


def _passive_depth(rods):
    """Number of successive passive activations since the most recent trigger."""
    pending, best = {}, 0
    for x, h, color in rods:
        depth = 0 if color == 1 else pending.get(x)
        if depth is not None:
            best = max(best, depth)
            pending[x + h] = depth + 1
    return best


def _make_pair(rods, height, width, background):
    a = np.full((height, width), background, dtype=np.int8)
    for x, h, color in rods:
        a[-h:, x] = color
    return {"input": a, "output": solve(a)}


def _anchors(rng, background, palette):
    # Matched columns: changing only the first trigger's height changes contact.
    for _ in range(128):
        short = int(rng.integers(2, 5))
        passive_h = int(rng.integers(2, 5))
        chain_n = int(rng.integers(3, 5))
        heights = sorted({short, short + 1, passive_h})
        rods = []
        start = int(rng.integers(0, 5))
        if rng.random() < 0.55:
            lead_h = int(rng.choice(heights))
            lead_x = int(rng.integers(0, 3))
            rods.append((lead_x, lead_h, int(rng.choice(palette))))
            start = lead_x + lead_h + int(rng.integers(0, 3))
        first = len(rods)
        rods.append((start, short + 1, 1))
        x = start + short + 1
        for _ in range(chain_n - 1):
            rods.append((x, passive_h, palette[0]))
            x += passive_h
        # An independent trigger is essential evidence even when train_count=2.
        x += int(rng.integers(1, 4))
        tail_h = int(rng.choice(heights))
        rods.append((x, tail_h, 1))
        end = x + tail_h
        if end > 29:
            continue
        width = end + int(rng.integers(0, min(4, 30 - end) + 1))
        height = max(h for _, h, _ in rods) + int(rng.integers(1, 5))
        miss = list(rods)
        miss[first] = (start, short, 1)
        pairs = [_make_pair(miss, height, width, background),
                 _make_pair(rods, height, width, background)]
        if any(np.array_equal(_compress(p["input"]), p["output"]) for p in pairs):
            continue
        rng.shuffle(pairs)
        return pairs, heights, chain_n
    raise RuntimeError("Bounded anchor construction exhausted")


def _general_pair(rng, background, palette, heights, chain_limit=None):
    for _ in range(192):
        count = int(rng.integers(2, 8))
        hs = [int(rng.choice(heights)) for _ in range(count)]
        colors = [1 if rng.random() < 0.34 else int(rng.choice(palette))
                  for _ in range(count)]
        if 1 not in colors:
            colors[int(rng.integers(count))] = 1
        if all(c == 1 for c in colors):
            colors[int(rng.integers(count))] = int(rng.choice(palette))
        x = int(rng.integers(0, 5))
        rods = []
        for h, color in zip(hs, colors):
            rods.append((x, h, color))
            x += h + int(rng.choice((0, 0, 0, 1, 1, 2, 4)))
        end = rods[-1][0] + hs[-1]
        if end > 30 or (chain_limit is not None and _longest_chain(rods) > chain_limit):
            continue
        width = end + int(rng.integers(0, min(6, 30 - end) + 1))
        height = max(hs) + int(rng.integers(1, 7))
        pair = _make_pair(rods, height, width, background)
        if not np.array_equal(_compress(pair["input"]), pair["output"]):
            return pair
    raise RuntimeError("Bounded general construction exhausted")


def _long_pair(rng, background, palette, heights, chain_n, allow_last_trigger=True):
    for _ in range(192):
        rods = []
        start = int(rng.integers(0, 5))
        if rng.random() < 0.6:
            lead_h = int(rng.choice(heights))
            lead_x = int(rng.integers(0, 3))
            rods.append((lead_x, lead_h, int(rng.choice(palette))))
            start = lead_x + lead_h + int(rng.integers(0, 4))
        x = start
        for i in range(chain_n):
            h = int(rng.choice(heights))
            color = 1 if i == 0 else int(rng.choice(palette))
            if i == chain_n - 1 and rng.random() < 0.3 and allow_last_trigger:
                color = 1  # A chain may contact an independently active trigger.
            rods.append((x, h, color))
            x += h
        if x > 30:
            continue
        if rng.random() < 0.5:
            tail_x = x + int(rng.integers(1, 4))
            tail_h = int(rng.choice(heights))
            if tail_x + tail_h <= 30:
                color = 1 if rng.random() < 0.5 else int(rng.choice(palette))
                rods.append((tail_x, tail_h, color))
                x = tail_x + tail_h
        width = x + int(rng.integers(0, min(5, 30 - x) + 1))
        height = max(h for _, h, _ in rods) + int(rng.integers(1, 7))
        return _make_pair(rods, height, width, background)
    raise RuntimeError("Bounded long-chain construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    background = int(rng.choice([0, 2, 3, 4, 5, 6, 7, 8, 9]))
    palette = [int(c) for c in rng.choice([c for c in range(10) if c not in (background, 1)],
                                         size=int(rng.integers(1, 4)), replace=False)]
    anchors, heights, chain_n = _anchors(rng, background, palette)
    puzzle = {"train": anchors, "test": []}
    used = {_key(p["input"]) for p in anchors}
    for _ in range(num_train - 2):
        for _ in range(64):
            pair = _general_pair(rng, background, palette, heights, chain_n)
            key = _key(pair["input"])
            if key not in used:
                used.add(key)
                puzzle["train"].append(pair)
                break
        else:
            raise RuntimeError("Distinct training construction exhausted")
    # Every test height and passive pigment has already occurred in training.
    observed = set()
    for pair in puzzle["train"]:
        observed.update(c for _, _, c in _parse(pair["input"])[2] if c != 1)
    test_palette = sorted(observed)
    for i in range(num_test):
        for _ in range(64):
            if i == 0 or rng.random() < 0.45:
                pair = _long_pair(rng, background, test_palette, heights, chain_n + 1,
                                  allow_last_trigger=i != 0)
            else:
                pair = _general_pair(rng, background, test_palette, heights)
            key = _key(pair["input"])
            if key not in used:
                used.add(key)
                puzzle["test"].append(pair)
                break
        else:
            raise RuntimeError("Distinct test construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _global_unique(a):
    """Deliberately wrong alternative: delete even separated recurrences."""
    def firsts(vectors):
        seen, keep = set(), []
        for i, vector in enumerate(vectors):
            key = vector.tobytes()
            if key not in seen:
                keep.append(i)
                seen.add(key)
        return keep
    reduced = a[firsts(a)]
    return reduced[:, firsts(reduced.T)]


def _ablation_failures(a, background, rods, output):
    active = _active(rods)
    final = _land(a, background, rods, active)
    triggers = {i for i, (_, _, c) in enumerate(rods) if c == 1}
    row_keep = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    early = a[row_keep]
    early_rods = [(x, int(np.count_nonzero(early[:, x] != background)), c)
                  for x, _, c in rods]
    alternatives = {
        "omit_A": _compress(a),
        "omit_B": final,
        "triggers_only": _compress(_land(a, background, rods, triggers)),
        "one_hop": _compress(_land(a, background, rods, _active(rods, one_hop=True))),
        "all_fall": _compress(_land(a, background, rods, set(range(len(rods))))),
        "bridge_one_cell_gap": _compress(_land(a, background, rods, _active(rods, bridge_gap=True))),
        "global_deduplication": _global_unique(final),
        "compress_rows_first": _compress(_land(early, background, early_rods, _active(early_rods))),
        "leftmost_trigger_only": _compress(_land(a, background, rods,
                                                _active(rods, starts=[min(triggers)]))),
        "rows_only": final[np.r_[True, np.any(final[1:] != final[:-1], axis=1)]],
        "columns_only": final[:, np.r_[True, np.any(final[:, 1:] != final[:, :-1], axis=0)]],
    }
    return {name for name, candidate in alternatives.items() if not np.array_equal(candidate, output)}


def _height_contrast(analyses):
    for i, (a, _, rods, out) in enumerate(analyses):
        for b, _, other, result in analyses[i + 1:]:
            if a.shape != b.shape or len(rods) != len(other) or np.array_equal(out, result):
                continue
            differences = [(r, s) for r, s in zip(rods, other) if r != s]
            if len(differences) != 1:
                continue
            r, s = differences[0]
            if r[0] == s[0] and r[2] == s[2] == 1 and abs(r[1] - s[1]) == 1:
                return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    analyses = {"train": [], "test": []}
    seen, backgrounds, failures = set(), set(), set()
    train_heights, test_heights, train_colors, test_colors = set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, background, rods = _parse(pair["input"])
            out = _grid(pair["output"])
            key = _key(a)
            assert key not in seen
            seen.add(key)
            backgrounds.add(background)
            active = _active(rods)
            final = _land(a, background, rods, active)
            assert np.array_equal(_compress(final), out)
            assert np.array_equal(_compress(final.T).T, out)
            assert out.shape[0] <= a.shape[0] and out.shape[1] < a.shape[1]
            assert not np.array_equal(_compress(a), out), "Toppling must remain observable"
            assert not np.any(np.all(out[1:] == out[:-1], axis=1))
            assert not np.any(np.all(out[:, 1:] == out[:, :-1], axis=0))
            # Reconstruct the FINAL canvas using its lossless row/column run lengths.
            row_starts = np.r_[0, np.flatnonzero(np.any(final[1:] != final[:-1], axis=1)) + 1]
            col_starts = np.r_[0, np.flatnonzero(np.any(final[:, 1:] != final[:, :-1], axis=0)) + 1]
            row_runs = np.diff(np.r_[row_starts, a.shape[0]])
            col_runs = np.diff(np.r_[col_starts, a.shape[1]])
            assert np.array_equal(np.repeat(np.repeat(out, row_runs, axis=0), col_runs, axis=1), final)
            for i, (x, h, color) in enumerate(rods):
                if i in active:
                    assert np.all(final[-1, x:x + h] == color)
                    assert np.all(final[-h:-1, x] == background)
                else:
                    assert color != 1 and np.all(final[-h:, x] == color)
            analyses[split].append((a, background, rods, out))
            (train_heights if split == "train" else test_heights).update(h for _, h, _ in rods)
            (train_colors if split == "train" else test_colors).update(c for _, _, c in rods)
            if split == "train":
                failures.update(_ablation_failures(a, background, rods, out))
    assert len(backgrounds) == 1
    assert any(out.shape[0] < a.shape[0] for a, _, _, out in analyses["train"])
    assert test_heights <= train_heights and test_colors <= train_colors
    assert _height_contrast(analyses["train"])
    assert failures == {"omit_A", "omit_B", "triggers_only", "one_hop", "all_fall",
                        "bridge_one_cell_gap", "global_deduplication", "compress_rows_first",
                        "leftmost_trigger_only", "rows_only", "columns_only"}
    assert max(_longest_chain(r) for _, _, r, _ in analyses["test"]) > max(
        _longest_chain(r) for _, _, r, _ in analyses["train"])
    assert _passive_depth(analyses["test"][0][2]) > max(
        _passive_depth(r) for _, _, r, _ in analyses["train"])
    return True


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    # Hand-calculated fixtures, including the one-cell halt before an upright rod.
    a = np.zeros((5, 11), dtype=np.int8)
    for x, h, c in ((0, 3, 1), (3, 2, 2), (5, 2, 2), (8, 3, 2)):
        a[-h:, x] = c
    expected = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [1, 2, 0, 2, 0]], dtype=np.int8)
    assert np.array_equal(solve(a), expected)
    miss = a.copy()
    miss[-3, 0] = 0
    expected_miss = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 2, 0],
                              [0, 0, 2, 0, 2, 0, 2, 0],
                              [1, 0, 2, 0, 2, 0, 2, 0]], dtype=np.int8)
    assert np.array_equal(solve(miss), expected_miss)
    # Passive palette relabeling and repeated blank top padding preserve the rule.
    recolored = a.copy()
    recolored[a == 0] = 7
    recolored[a == 2] = 4
    target = expected.copy()
    target[expected == 0] = 7
    target[expected == 2] = 4
    assert np.array_equal(solve(recolored), target)
    assert np.array_equal(solve(np.pad(a, ((3, 0), (0, 0)))), expected)
    tall = np.zeros((30, 30), dtype=np.int8)
    tall[1:, 0] = 1
    assert np.array_equal(solve(tall), np.array([[0, 0], [1, 0]], dtype=np.int8))
    pigments = [1] + [2 + i % 8 for i in range(14)]
    dense = np.zeros((3, 30), dtype=np.int8)
    for i, pigment in enumerate(pigments):
        dense[1:, 2 * i] = pigment
    assert np.array_equal(solve(dense), np.array([[0] * 15, pigments], dtype=np.int8))
    input_shapes, output_shapes, layouts, palettes, heights, chains, rod_counts = (set() for _ in range(7))
    puzzle_hashes = set()
    pair_count = leading_passive = fully_active = multicolor = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _digest(puzzle)
        assert signature == _digest(generate_puzzle(seed))
        puzzle_hashes.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, background, rods = _parse(pair["input"])
                pair_count += 1
                input_shapes.add(a.shape)
                output_shapes.add(pair["output"].shape)
                palettes.add((background, tuple(sorted({c for _, _, c in rods}))))
                layouts.add((a.shape, tuple((x, h, c == 1) for x, h, c in rods)))
                heights.update(h for _, h, _ in rods)
                chains.add(_longest_chain(rods))
                rod_counts.add(len(rods))
                leading_passive += rods[0][2] != 1
                fully_active += len(_active(rods)) == len(rods)
                multicolor += len({c for _, _, c in rods if c != 1}) > 1
                original = a.copy()
                assert np.array_equal(solve(a.tolist()), pair["output"])
                assert np.array_equal(a, original)
    assert len(puzzle_hashes) == 42
    assert len(input_shapes) >= 50 and len(output_shapes) >= 12
    assert len(layouts) >= 120 and len(palettes) >= 25
    assert {2, 3, 4, 5} <= heights and {3, 4, 5} <= chains
    assert len(rod_counts) >= 5 and leading_passive >= 20 and fully_active >= 20
    assert multicolor >= 10
    count_calls = 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(0, nt, ns)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ns
            count_calls += 1
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    return {"seeds": len(HOST_SEEDS), "pairs": pair_count, "distinct_puzzles": len(puzzle_hashes),
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes),
            "layouts": len(layouts), "palettes": len(palettes), "rod_heights": sorted(heights),
            "chain_lengths": sorted(chains), "rod_counts": sorted(rod_counts),
            "leading_passive_pairs": leading_passive, "fully_active_pairs": fully_active,
            "multicolor_passive_pairs": multicolor, "legal_count_calls": count_calls,
            "semantic_verification": "Generator-owned invariants and hand-calculated fixtures only"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
