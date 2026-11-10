"""Bottom-anchored row shear, then lossless-in-value consecutive-run collapse.

Generation makes equality appear and disappear under the shear.  Solving has
no knowledge of the sampler, seed, palettes, or example split.  No disk writes.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


VERIFICATION_SEEDS = (
    0, 165454504, 244142500, 768961730, 896093258, 924520295,
    204841292, 148271332, 305943638, 942877333, 308111101, 586075485,
    43166544, 28402683, 888431956, 584940031, 592713267, 290645488,
    60858472, 975629196, 681405104, 39628111, 318511694, 19056122,
    217984969, 391950048, 936622699, 478482744, 858536439, 397602535,
    70319166, 103594113, 233395559, 675716105, 217032311, 880306840,
    891755765, 936370533, 67213646, 237883293, 359436186, 731914891,
)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _shear(a):
    h, w = a.shape
    s = np.zeros((h, h + w - 1), dtype=np.int8)
    for r in range(h):
        start = h - 1 - r
        s[r, start:start + w] = a[r]
    return s


def _run_info(a, axis):
    vectors = a if axis == 0 else a.T
    starts = [0]
    for i in range(1, len(vectors)):
        if not np.array_equal(vectors[i], vectors[i - 1]):
            starts.append(i)
    lengths = np.diff(starts + [len(vectors)]).tolist()
    return starts, lengths


def _collapse(a):
    rows, _ = _run_info(a, 0)
    cols, _ = _run_info(a, 1)
    return a[np.ix_(rows, cols)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the fixed rule to any admissible rectangle, including zero data."""
    a = _grid(input_grid)
    if sum(a.shape) - 1 > 30:
        raise ValueError("the uncompressed shear canvas must fit in 30 columns")
    return _collapse(_shear(a))


def _reference(a):
    """Coordinate-copy and tuple-run check, separate from solve's helpers."""
    h, w = a.shape
    s = [[0] * (h + w - 1) for _ in range(h)]
    for r in range(h):
        for c in range(w):
            s[r][h - 1 - r + c] = int(a[r, c])
    rows = [s[0]]
    for row in s[1:]:
        if row != rows[-1]:
            rows.append(row)
    columns = list(zip(*rows))
    kept = [columns[0]]
    for column in columns[1:]:
        if column != kept[-1]:
            kept.append(column)
    return np.array(list(zip(*kept)), dtype=np.int8)


def _key(a):
    return a.shape, a.tobytes()


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _normalized_key(a):
    mapping = {0: 0}
    normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in a.flat)
    return a.shape, hashlib.sha256(normalized).digest()


def _composition(rng, total, parts):
    """A uniformly selected positive integer composition."""
    if parts == 1:
        return [total]
    cuts = sorted(int(c) for c in rng.choice(np.arange(1, total), parts - 1, replace=False))
    return np.diff([0] + cuts + [total]).tolist()


def _palette(rng, low=2, high=5):
    return rng.choice(np.arange(1, 10, dtype=np.int8), int(rng.integers(low, high + 1)), replace=False)


def _unshear(s, w):
    """Construct source diagonals from legal repeated sheared profiles."""
    h = len(s)
    a = np.empty((h, w), dtype=np.int8)
    for r in range(h):
        start = h - 1 - r
        assert np.all(s[r, :start] == 0) and np.all(s[r, start + w:] == 0)
        a[r] = s[r, start:start + w]
    return a


def _merging_anchor(rng):
    # Repeated nonzero profiles P,Q,P; unequal bands; nonlocal repeats survive.
    row_runs = [int(rng.integers(2, 4)), 1, int(rng.integers(2, 4))]
    col_runs = [int(rng.integers(2, 5)), int(rng.integers(1, 3)),
                int(rng.integers(2, 5)), int(rng.integers(1, 3)),
                int(rng.integers(2, 5))]
    h, span = sum(row_runs), sum(col_runs)
    extra = int(rng.integers(0, min(3, 30 - (2 * h - 2 + span)) + 1))
    left = int(rng.integers(0, extra + 1))
    a, b = (int(c) for c in _palette(rng, 2, 2))
    base = np.array([[a, a, 0, b, a], [b, a, 0, b, b], [a, a, 0, b, a]], dtype=np.int8)
    motif = np.repeat(np.repeat(base, row_runs, axis=0), col_runs, axis=1)
    w = h - 1 + span + extra
    s = np.zeros((h, h + w - 1), dtype=np.int8)
    start = h - 1 + left
    s[:, start:start + span] = motif
    return _unshear(s, w)


def _splitting_anchor(rng, other_height=None):
    h = int(rng.integers(2, 7))
    if h == other_height:
        h = 2
    colors = _palette(rng, 2, 3)
    runs = [int(rng.integers(h + 1, h + 5)), int(rng.integers(2, 5))]
    if len(colors) == 3:
        runs.append(int(rng.integers(1, 4)))
    row = np.repeat(colors, runs)
    if rng.integers(2):
        row = row[::-1]
    return np.repeat(row[None, :], h, axis=0).copy()


def _common_profiles(rng, h, w=None, wide=False):
    if w is None:
        span = int(rng.integers(4, min(14, 32 - 2 * h) + 1))
        extra = int(rng.integers(0, min(3, 32 - 2 * h - span) + 1))
        w = h - 1 + span + extra
    else:
        extra = int(rng.integers(0, min(2, w - h - 2) + 1))
        span = w - h + 1 - extra
    left = int(rng.integers(0, extra + 1))
    k = int(rng.integers(min(6 if wide else 2, span), min(9 if wide else 6, span) + 1))
    groups = 1 if h == 2 else int(rng.integers(2, min(6, h)))
    row_runs = _composition(rng, h, groups)
    col_runs = _composition(rng, span, k)
    palette = np.concatenate((np.array([0], dtype=np.int8), _palette(rng)))
    base = rng.choice(palette, size=(groups, k))
    if groups >= 3 and rng.random() < 0.65:
        base[-1] = base[0]
    if not np.any(base):
        base[0, 0] = palette[1]
    motif = np.repeat(np.repeat(base, row_runs, axis=0), col_runs, axis=1)
    s = np.zeros((h, h + w - 1), dtype=np.int8)
    s[:, h - 1 + left:h - 1 + left + span] = motif
    return _unshear(s, w)


def _local_profiles(rng, h, w):
    """Different row groups occupy different feasible canvas intervals.

    Global column bands guarantee exact vector repeats without requiring a
    common overlap across the whole height.  A blank border group keeps both
    axes of collapse observable even in a 30-by-1 source.
    """
    n = h + w - 1
    cuts = [0] + [i for i in range(1, n) if rng.random() < 0.75] + [n]
    bands = list(zip(cuts[:-1], cuts[1:]))
    blank = int(rng.integers(2, min(4, h - 1) + 1))
    blank_top = bool(rng.integers(2))
    first, stop = (blank, h) if blank_top else (0, h - blank)
    s = np.zeros((h, n), dtype=np.int8)
    palette = _palette(rng, 1, 5)
    r = first
    while r < stop:
        count = int(rng.integers(1, min(4, w, stop - r) + 1))
        lo, hi = h - 1 - r, h - 1 - (r + count - 1) + w
        available = [(a, b) for a, b in bands if lo <= a and b <= hi]
        for a, b in available:
            if rng.random() < 0.68:
                s[r:r + count, a:b] = rng.choice(palette)
        r += count
    if not np.any(s):
        # One legal data cell, preserving the reserved blank border group.
        r = first
        s[r, h - 1 - r + int(rng.integers(w))] = palette[0]
    return _unshear(s, w)


def _fallback(serial, split):
    # 15 geometrically distinct candidates suffice: at most 13 earlier inputs.
    # All preserve the test dimension shift.  Used only after random collisions.
    if split == "train":
        h, w = 8, 8 + serial
    else:
        h, w = 9 + serial, min(5, 22 - serial)
    a = np.zeros((h, w), dtype=np.int8)
    for r in range(2, h):
        a[r, (2 * r + serial) % w] = 1 + (r + serial) % 9
    return a


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for attempt in range(47):
                max_width = max((p["input"].shape[1] for p in puzzle["train"]), default=1)
                if attempt >= 32:
                    a = _fallback(attempt - 32, split)
                elif split == "train" and index == 0:
                    a = _merging_anchor(rng)
                elif split == "train" and index == 1:
                    a = _splitting_anchor(rng, puzzle["train"][0]["input"].shape[0])
                elif split == "train":
                    mode = int(rng.integers(3))
                    if mode == 0:
                        a = _common_profiles(rng, int(rng.integers(2, 9)))
                    elif mode == 1:
                        a = _local_profiles(rng, int(rng.integers(3, 9)), int(rng.integers(1, 18)))
                    else:
                        a = _splitting_anchor(rng)
                else:
                    mode = int(rng.integers(3))
                    if mode == 0:
                        a = _common_profiles(rng, int(rng.integers(9, 13)))
                    elif mode == 1:
                        h = int(rng.integers(9, 31))
                        w = int(rng.integers(1, min(11, 31 - h) + 1))
                        a = _local_profiles(rng, h, w)
                    else:
                        h = int(rng.integers(2, 5))
                        w = int(rng.integers(max_width + 1, 32 - h))
                        a = _common_profiles(rng, h, w, wide=True)
                if _key(a) not in used:
                    used.add(_key(a))
                    puzzle[split].append({"input": a, "output": solve(a)})
                    break
            else:
                raise RuntimeError("bounded unique-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _separated_repeat(a, axis):
    vectors = a if axis == 0 else a.T
    starts, _ = _run_info(a, axis)
    seen = set()
    for i in starts:
        key = vectors[i].tobytes()
        if np.any(vectors[i]) and key in seen:
            return True
        seen.add(key)
    return False


def validate_generated_puzzle(puzzle):
    """Check returned values, equality changes, reconstruction and teaching cases."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used, train_shapes = set(), set()
    merge = split_rows = both_axes = separate_rows = separate_cols = False
    omit_a = omit_b = wrong_order = False
    max_h = max(np.asarray(p["input"]).shape[0] for p in puzzle["train"])
    max_w = max(np.asarray(p["input"]).shape[1] for p in puzzle["train"])
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            h, w = a.shape
            assert h + w - 1 <= 30 and _key(a) not in used and np.any(a)
            used.add(_key(a))
            s = _shear(a)
            rows, rr = _run_info(s, 0)
            cols, cc = _run_info(s, 1)
            assert out.shape == (len(rows), len(cols))
            assert np.array_equal(out, s[np.ix_(rows, cols)])
            assert np.array_equal(out, _reference(a))
            assert np.array_equal(out, solve(a))
            assert np.array_equal(np.repeat(np.repeat(out, rr, axis=0), cc, axis=1), s)
            assert all(n == 1 for n in _run_info(out, 0)[1] + _run_info(out, 1)[1])
            # Removing a duplicate vector cannot alter equality on the other axis.
            column_reduced = s[:, cols]
            column_first_rows, _ = _run_info(column_reduced, 0)
            column_first = column_reduced[column_first_rows]
            assert np.array_equal(out, column_first)
            if split == "train":
                train_shapes.add(a.shape)
                both_axes |= len(rows) < h and len(cols) < s.shape[1]
                separate_rows |= _separated_repeat(s, 0)
                separate_cols |= _separated_repeat(s, 1)
                for r in range(1, h):
                    equal_in = np.array_equal(a[r], a[r - 1])
                    equal_shear = np.array_equal(s[r], s[r - 1])
                    merge |= not equal_in and equal_shear and bool(np.any(s[r]))
                    split_rows |= equal_in and not equal_shear
                omit_a |= not np.array_equal(_collapse(a), out)
                omit_b |= not np.array_equal(s, out)
                wrong_order |= not np.array_equal(_shear(_collapse(a)), out)
            else:
                assert h > max_h or w > max_w, "test must extend a demonstrated dimension"
    assert len(train_shapes) >= 2
    assert merge and split_rows and both_axes and separate_rows and separate_cols
    assert omit_a and omit_b and wrong_order


def self_test():
    """Bounded in-memory checks over 42 seeds and every legal explicit count."""
    # These small expected answers are hand-computable, independent fixtures.
    fixtures = (
        ([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], [[0, 1, 0]]),
        ([[1], [2], [3]], [[0, 0, 1], [0, 2, 0], [3, 0, 0]]),
        ([[0, 0, 2, 2, 0, 2, 2, 0, 0]], [[0, 2, 0, 2, 0]]),
        ([[0, 0], [0, 0]], [[0]]),
        ([[8]], [[8]]),
    )
    for inp, expected in fixtures:
        assert np.array_equal(solve(np.array(inp)), np.array(expected))
    shapes, output_shapes, structures, palettes, puzzles = set(), set(), set(), set(), set()
    merging_pairs = splitting_pairs = both = one = no_op = boundary = narrow = 0
    max_area = 0
    for seed in VERIFICATION_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        puzzles.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                s = _shear(a)
                shapes.add(a.shape)
                output_shapes.add(out.shape)
                structures.add(_normalized_key(a))
                palettes.add(tuple(int(c) for c in np.unique(a)))
                max_area = max(max_area, a.size, s.size, out.size)
                shrink_r, shrink_c = out.shape[0] < s.shape[0], out.shape[1] < s.shape[1]
                both += shrink_r and shrink_c
                one += shrink_r != shrink_c
                no_op += not shrink_r and not shrink_c
                boundary += s.shape[1] == 30
                narrow += a.shape[1] <= 2
                merging_pairs += any(np.any(s[r]) and np.array_equal(s[r], s[r - 1])
                                     and not np.array_equal(a[r], a[r - 1]) for r in range(1, len(a)))
                splitting_pairs += any(np.array_equal(a[r], a[r - 1])
                                       and not np.array_equal(s[r], s[r - 1]) for r in range(1, len(a)))
    assert len(puzzles) == 42 and len(shapes) >= 60 and len(structures) >= 150
    assert len(output_shapes) >= 30 and len(palettes) >= 40
    assert merging_pairs >= 42 and splitting_pairs >= 42 and both >= 42
    assert one >= 42 and boundary >= 2 and narrow >= 1
    legal_count_calls = 0
    for n_train in range(2, 11):
        for n_test in range(1, min(4, n_train - 1) + 1):
            p = generate_puzzle(1234, n_train, n_test)
            assert (len(p["train"]), len(p["test"])) == (n_train, n_test)
            legal_count_calls += 1
    invalid = ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
               (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None))
    for counts in invalid:
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for bad_seed in (-1, True, 1.2, None):
        try:
            generate_puzzle(bad_seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # All 1..30 boundary aspect ratios, with data zeros and arbitrary colors.
    rng = np.random.default_rng(7301)
    for h in range(1, 31):
        a = rng.integers(0, 10, (h, 31 - h), dtype=np.int8)
        assert np.array_equal(solve(a), _reference(a))
    for split in ("train", "test"):
        fallback_keys = set()
        for serial in range(15):
            a = _fallback(serial, split)
            fallback_keys.add(_key(a))
            assert np.array_equal(solve(a), _reference(a))
            assert a.shape[0] <= 8 if split == "train" else a.shape[0] >= 9
        assert len(fallback_keys) == 15
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(puzzles),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "normalized_input_structures": len(structures), "palettes": len(palettes),
            "merging_pairs": merging_pairs, "splitting_pairs": splitting_pairs,
            "both_axes_collapse": both, "one_axis_collapse": one, "no_op_collapse": no_op,
            "width_30_canvases": boundary, "narrow_inputs": narrow,
            "max_grid_cells": max_area, "legal_count_calls": legal_count_calls}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
