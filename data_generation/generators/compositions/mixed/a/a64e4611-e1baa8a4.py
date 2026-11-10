"""Logical-scale empty neighborhoods after whole-line run compression.

Background is always 0; the introduced highlight is always 3. No files, parent
modules, seeds, or puzzle-level parameters are consulted by solve().
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


HOST_SEEDS = (0, 244298914, 196492641, 796898412, 36241455, 857182309,
              25305517, 242934403, 224583242, 93515267, 898121674,
              677151937, 785146410, 600323704, 427601177, 193159431,
              845098559, 682237478, 906507787, 649011517, 172184199,
              312743763, 996018344, 573906214, 1323626, 656781589,
              42445658, 417328807, 103417182, 50545832, 833458496,
              355367221, 771765723, 153851555, 130101523, 239211085,
              442682982, 533063609, 39323095, 890433383, 165076971,
              292404592)
FOREGROUND_COLORS = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=np.int8)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("a grid must be an integer rectangle, 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _starts(a, axis):
    different = np.any(np.diff(a, axis=axis) != 0, axis=1 - axis)
    return np.r_[0, np.flatnonzero(different) + 1]


def _compress(a):
    return a[np.ix_(_starts(a, 0), _starts(a, 1))].copy()


def _runs(a, axis):
    return tuple(int(x) for x in np.diff(np.r_[_starts(a, axis), a.shape[axis]]))


def _expand(a, rows, cols):
    return np.repeat(np.repeat(a, rows, axis=0), cols, axis=1)


def _empty_mask(a, diagonal=True):
    h, w = a.shape
    padded = np.pad(a != 0, 1, constant_values=False)
    near = np.zeros((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            if diagonal or dy == 1 or dx == 1:
                near |= padded[dy:dy + h, dx:dx + w]
    return ~near


def _paint(a):
    out = a.copy()
    out[_empty_mask(a)] = 3
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compress original full-line runs once, then mark all clear centers."""
    a = _grid(input_grid)
    if np.any(a == 3):
        raise ValueError("3 is reserved for the introduced output highlight")
    return _paint(_compress(a))


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _sample_runs(rng, n, expand, nonuniform=False):
    if not expand:
        return (1,) * n
    mode = int(rng.integers(5))
    if mode == 0:
        values = np.full(n, int(rng.integers(2, min(4, 30 // n) + 1)))
    elif mode == 1:
        values = np.ones(n, dtype=int)
        values[int(rng.integers(n))] = int(rng.integers(7, min(17, 31 - n)))
    else:
        values = rng.integers(1, 5, size=n)
        if mode == 2:
            values.sort()
        elif mode == 3:
            values = np.sort(values)[::-1].copy()
    # At most 60 iterations (n <= 15, initial values <= 4, except one wide run).
    while int(values.sum()) > 30:
        indices = np.flatnonzero(values > 1)
        values[int(rng.choice(indices))] -= 1
    if values.max() == 1:
        values[0] = 2
    if nonuniform and np.all(values == values[0]):
        values[int(rng.integers(n))] = 1
    return tuple(int(v) for v in values)


def _layout(rng, h, w, mode, palette, diagnostic=None):
    """Sample one logical scene; reject unobservable full-line boundaries later."""
    transpose = bool(rng.integers(2))
    if transpose:
        h, w = w, h
    density = float(rng.uniform(.18, .40))
    a = np.where(rng.random((h, w)) < density,
                 rng.choice(palette, size=(h, w)), 0).astype(np.int8)
    if mode == 0:  # Several small openings in irregular texture.
        for _ in range(int(rng.integers(1, 4))):
            r, c = int(rng.integers(h - 2)), int(rng.integers(w - 2))
            a[r:r + 3, c:c + 3] = 0
    elif mode == 1:  # Stepped corridor; exterior cells retain line identity.
        c = int(rng.integers(1, w - 3))
        for r in range(h):
            if rng.random() < .35:
                c = int(np.clip(c + rng.choice([-1, 1]), 1, w - 4))
            a[r, c:c + 3] = 0
    elif mode == 2:  # Broad opening, sometimes with a foreground island.
        ph = int(rng.integers(3, max(3, h - 4) + 1))
        pw = int(rng.integers(3, max(3, w - 4) + 1))
        r, c = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
        a[r:r + ph, c:c + pw] = 0
        if ph >= 5 and pw >= 5 and rng.random() < .65:
            a[r + ph // 2, c + pw // 2] = rng.choice(palette)
    elif mode == 3:  # Short obstacle streaks and a bounded clear patch.
        for _ in range(int(rng.integers(2, 6))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            if rng.random() < .5:
                a[r, c:min(w, c + int(rng.integers(2, 5)))] = rng.choice(palette)
            else:
                a[r:min(h, r + int(rng.integers(2, 5))), c] = rng.choice(palette)
        a[h // 2 - 1:h // 2 + 2, w // 2 - 1:w // 2 + 2] = 0
    elif mode == 4:  # Thin crossing slit plus an unrelated qualifying pocket.
        a[1:h - 1, w // 2:w // 2 + 2] = 0
        a[0:3, 0:3] = 0
    else:  # Two separated clear pockets, not one selected object.
        a[:3, :3] = 0
        a[h - 3:, w - 3:] = 0

    if diagnostic == "corner":
        # Three-cell clear-center component and a diagonal-only blocker at (1,1).
        a[:4, :4] = rng.choice(palette)
        a[:3, :3] = 0
        a[2, 2] = rng.choice(palette)
        # Separated identical vectors must survive compression.
        if rng.random() < .5:
            a[-1, :] = a[0, :]
        else:
            a[:, -1] = a[:, 0]
    elif diagnostic == "pocket":
        r, c = int(rng.integers(1, h - 4)), int(rng.integers(1, w - 4))
        a[r:r + 5, c:c + 5] = rng.choice(palette)
        a[r + 1:r + 4, c + 1:c + 4] = 0
    # Isometries apply to the whole scene, including the diagnostic.
    if rng.random() < .5:
        a = a[::-1]
    if rng.random() < .5:
        a = a[:, ::-1]
    return a.T.copy() if transpose else a.copy()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    sizes = []
    while remaining:
        todo = [remaining.pop()]
        size = 0
        while todo:
            r, c = todo.pop()
            size += 1
            for pos in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if pos in remaining:
                    remaining.remove(pos)
                    todo.append(pos)
        sizes.append(size)
    return sizes


def _separated_repeat(a):
    return (any(np.array_equal(a[r], a[s]) for r in range(a.shape[0])
                for s in range(r + 2, a.shape[0]))
            or any(np.array_equal(a[:, c], a[:, d]) for c in range(a.shape[1])
                   for d in range(c + 2, a.shape[1])))


def _key(a):
    return a.shape, a.tobytes()


def _different(a, b):
    return a.shape != b.shape or not np.array_equal(a, b)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    tr = _count("num_train", num_train, 2, 10)
    te = _count("num_test", num_test, 1, 4)
    if tr <= te:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    # Test expands the logical extent on at least one axis beyond all training.
    growth = int(rng.integers(3))
    for split, count in (("train", tr), ("test", te)):
        for index in range(count):
            diagnostic = ("corner" if index == 0 else "pocket" if index == 1 else None) if split == "train" else None
            h, w = (int(rng.integers(7, 11)), int(rng.integers(7, 11)))
            if split == "test":
                if growth in (0, 2):
                    h = int(rng.integers(11, 16))
                if growth in (1, 2):
                    w = int(rng.integers(11, 16))
            color_count = 1 if diagnostic == "corner" else int(rng.integers(2, 5))
            palette = rng.choice(FOREGROUND_COLORS, size=color_count, replace=False)
            mode = int(rng.integers(6))
            axis_mode = (0 if diagnostic == "corner" else
                         int(rng.integers(1, 3)) if diagnostic == "pocket" else
                         int(rng.integers(3)))
            for _ in range(256):
                latent = _layout(rng, h, w, mode, palette, diagnostic)
                if _compress(latent).shape != latent.shape:
                    continue
                if diagnostic == "pocket" and len(np.unique(latent)) < 3:
                    continue
                clear = _empty_mask(latent)
                if not clear.any() or not np.any((latent == 0) & ~clear):
                    continue
                rows = _sample_runs(rng, h, axis_mode != 2, diagnostic == "corner")
                cols = _sample_runs(rng, w, axis_mode != 1, diagnostic == "corner")
                inp = _expand(latent, rows, cols)
                out = _paint(latent)
                # An actual order counterexample, not just different output dimensions.
                if not _different(_compress(_paint(inp)), out) or _key(inp) in seen:
                    continue
                seen.add(_key(inp))
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError(f"bounded construction exhausted for {split}[{index}]")
    validate_generated_puzzle(puzzle)
    return puzzle


def _reference(a):
    """Scalar specification check, implemented separately from the fast solver."""
    rows = [r for r in range(len(a)) if r == 0 or list(a[r]) != list(a[r - 1])]
    cols = [c for c in range(a.shape[1])
            if c == 0 or list(a[:, c]) != list(a[:, c - 1])]
    latent = a[np.ix_(rows, cols)]
    out = latent.copy()
    h, w = latent.shape
    for r in range(h):
        for c in range(w):
            if all(latent[y, x] == 0
                   for y in range(max(0, r - 1), min(h, r + 2))
                   for x in range(max(0, c - 1), min(w, c + 2))):
                out[r, c] = 3
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, train_shapes = set(), []
    witnesses = Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert not np.any(inp == 3) and np.any(inp == 0) and np.any(inp != 0)
            latent = _compress(inp)
            rows, cols = _runs(inp, 0), _runs(inp, 1)
            assert np.array_equal(_expand(latent, rows, cols), inp)
            assert out.shape == latent.shape and inp.shape != out.shape
            assert np.array_equal(out, _reference(inp))
            assert np.array_equal(out, solve(inp))
            assert np.all(out[latent != 0] == latent[latent != 0])
            mask = _empty_mask(latent)
            assert mask.any() and np.any((latent == 0) & ~mask)
            assert np.array_equal(out == 3, mask)
            assert _different(out, _paint(inp)), "omitting B must change the answer"
            assert _different(out, latent), "omitting A must change the answer"
            assert _different(out, _compress(_paint(inp))), "order must matter"
            assert not np.array_equal(_empty_mask(inp), _expand(mask, rows, cols))
            if split == "train":
                train_shapes.append(out.shape)
                witnesses["both_nonuniform"] += (max(rows) > 1 and max(cols) > 1
                                                and len(set(rows)) > 1 and len(set(cols)) > 1)
                witnesses["one_axis"] += (max(rows) == 1) != (max(cols) == 1)
                witnesses["corner"] += bool(mask[0, 0] or mask[0, -1] or mask[-1, 0] or mask[-1, -1])
                witnesses["diagonal"] += bool(np.any(_empty_mask(latent, False) & ~mask))
                witnesses["small_component"] += any(n <= 4 for n in _components(mask))
                witnesses["singleton"] += 1 in _components(mask)
                witnesses["separated_repeat"] += _separated_repeat(latent)
                witnesses["multiple_foreground_values"] += len(np.unique(inp)) >= 3
    assert all(witnesses[k] for k in ("both_nonuniform", "one_axis", "corner", "diagonal",
                                     "small_component", "singleton", "separated_repeat",
                                     "multiple_foreground_values")), witnesses
    for pair in puzzle["test"]:
        h, w = np.asarray(pair["output"]).shape
        assert h > max(s[0] for s in train_shapes) or w > max(s[1] for s in train_shapes)


def self_test():
    """Bounded, streaming 42-seed audit; returns only small aggregate metrics."""
    shapes, physical, structures, palettes, schedules, fingerprints = (set() for _ in range(6))
    counts = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                latent = _compress(inp)
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                shapes.add(out.shape)
                physical.add(inp.shape)
                structures.add((latent.shape, (latent != 0).tobytes()))
                palettes.add(tuple(int(x) for x in np.unique(inp)))
                schedules.add((_runs(inp, 0), _runs(inp, 1)))
                counts["pairs"] += 1
                counts["single_axis"] += (inp.shape[0] == out.shape[0]) != (inp.shape[1] == out.shape[1])
                counts["full_axis_30"] += 30 in inp.shape
                counts["multiple_components"] += len(_components(out == 3)) > 1
                counts["nonadjacent_repeats"] += _separated_repeat(latent)
                assert np.array_equal(solve(inp.T), out.T)
                assert np.array_equal(solve(inp[::-1, ::-1]), out[::-1, ::-1])
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42
    assert len(shapes) >= 30 and len(physical) >= 70
    assert len(structures) >= 150 and len(palettes) >= 25 and len(schedules) >= 100
    assert all(counts[k] for k in ("single_axis", "full_axis_30", "multiple_components", "nonadjacent_repeats"))
    # Solver also handles thin and degenerate family members not sampled above.
    for a, expected in (([[0]], [[3]]), ([[2]], [[2]]),
                        ([[0, 0, 2, 0, 0]], [[0, 2, 0]]),
                        ([[0, 1, 0], [0, 0, 0], [0, 1, 0]],
                         [[0, 1, 0], [0, 0, 0], [0, 1, 0]])):
        assert np.array_equal(solve(np.array(a)), np.array(expected))
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(73, tr, te)
            assert (len(p["train"]), len(p["test"])) == (tr, te)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {tr}/{te}")
    a, b = generate_puzzle(1234), generate_puzzle(1234)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
               for k in ("input", "output"))
    return {"seeds": 42, "distinct_puzzles": len(fingerprints),
            "logical_shapes": len(shapes), "physical_shapes": len(physical),
            "foreground_structures": len(structures), "palettes": len(palettes),
            "band_schedules": len(schedules), **dict(counts)}
