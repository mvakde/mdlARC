"""Repair a reflected enlarged mosaic, then collapse equal adjacent vectors.

Black (0) is missing, never payload. Folds maximize compatible in-bounds
overlap. All inference below receives only the current input.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _axis_candidates(grid, dimension):
    """(reflection sum, overlap) for folds with visible unanimous comparisons."""
    a = grid if dimension == 0 else grid.T
    n = len(a)
    result = []
    for s in range(1, 2 * n - 2):
        first = np.arange(max(0, s - n + 1), min(n, (s + 1) // 2))
        x, y = a[first], a[s - first]
        visible = (x != 0) & (y != 0)
        if visible.any() and np.all(x[visible] == y[visible]):
            result.append((s, n - abs(s - n + 1)))
    return result


def _infer_axes(grid):
    axes = []
    for dimension in (0, 1):
        candidates = _axis_candidates(grid, dimension)
        if not candidates:
            raise ValueError("No supported reflection fold")
        largest = max(overlap for _, overlap in candidates)
        best = [s for s, overlap in candidates if overlap == largest]
        if len(best) != 1:
            raise ValueError("Ambiguous largest-overlap reflection fold")
        axes.append(best[0])
    return tuple(axes)


def _orbits(shape, axes):
    h, w = shape
    a, b = axes
    seen = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rows = sorted({r, a - r} & set(range(h)))
            cols = sorted({c, b - c} & set(range(w)))
            orbit = tuple((y, x) for y in rows for x in cols)
            seen.update(orbit)
            yield orbit


def _repair(grid, axes):
    result = grid.copy()
    for orbit in _orbits(grid.shape, axes):
        values = {int(grid[p]) for p in orbit if grid[p] != 0}
        if len(values) != 1:
            raise ValueError("Empty or contradictory reflection orbit")
        color = values.pop()
        for p in orbit:
            result[p] = color
    return result


def _collapse(grid):
    rows = np.r_[0, np.flatnonzero(np.any(grid[1:] != grid[:-1], axis=1)) + 1]
    cols = np.r_[0, np.flatnonzero(np.any(grid[:, 1:] != grid[:, :-1], axis=0)) + 1]
    return grid[np.ix_(rows, cols)].copy(), rows, cols


def _analyze(grid):
    axes = _infer_axes(grid)
    clean = _repair(grid, axes)
    out, rows, cols = _collapse(clean)
    rb = tuple(int(x) for x in np.diff(np.r_[rows, grid.shape[0]]))
    cb = tuple(int(x) for x in np.diff(np.r_[cols, grid.shape[1]]))
    return out, clean, axes, rows, cols, rb, cb


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore unanimous reflection orbits before comparing complete vectors."""
    return _analyze(_grid(input_grid))[0]


def _visible_boundaries(grid):
    return (np.all(np.any(grid[1:] != grid[:-1], axis=1))
            and np.all(np.any(grid[:, 1:] != grid[:, :-1], axis=0)))


def _hidden_seam(grid):
    # One vector boundary is invisible at some cells, exposed at just one other.
    return any(np.any(np.count_nonzero(a[1:] != a[:-1], axis=1) == 1)
               for a in (grid, grid.T))


def _axis_plan(rng, half=None, fringe_max=2):
    k = int(rng.integers(1, 4)) if half is None else half
    fringe = int(rng.integers(fringe_max + 1))
    before = fringe if rng.random() < .5 else 0
    after = fringe - before
    n = before + 2 * k + 1 + after
    return n, 2 * before + 2 * k


def _scales(rng, n, fold, profile):
    groups = []
    seen = set()
    for i in range(n):
        if i not in seen:
            group = sorted({i, fold - i} & set(range(n)))
            groups.append(group)
            seen.update(group)
    for _ in range(64):
        if profile == "moderate":
            values = rng.integers(2, 5, len(groups))
        else:
            values = (np.full(len(groups), int(rng.integers(2, 5)))
                      if rng.random() < .18 else rng.integers(1, 6, len(groups)))
        if profile == "extreme":
            i, j = rng.choice(len(groups), 2, replace=False)
            values[i], values[j] = 1, int(rng.integers(5, 8))
        bands = np.empty(n, dtype=int)
        for value, group in zip(values, groups):
            bands[group] = value
        if bands.sum() <= 30 and max(bands) >= 2:
            return bands
    # A bounded, symmetry-compatible profile always fits n <= 9.
    bands = np.full(n, 2, dtype=int)
    if profile == "extreme":
        bands[groups[0]], bands[groups[-1]] = 1, 5
    return bands


def _logical(rng, shape, axes, seam):
    orbits = list(_orbits(shape, axes))
    k = int(rng.integers(2, min(3 if seam else 9, len(orbits)) + 1))
    colors = rng.choice(np.arange(1, 10, dtype=np.int8), k, replace=False)
    style = int(rng.integers(4))
    for _ in range(96):
        if seam or style == 0:
            values = rng.integers(k, size=len(orbits))
        else:
            points = np.array([o[0] for o in orbits])
            r, c = points.T
            if style == 1:
                values = (r + c + int(rng.integers(k))) % k
            elif style == 2:
                values = (np.maximum(r, c) + int(rng.integers(k))) % k
            else:
                values = (2 * r + c // 2 + int(rng.integers(k))) % k
            noise = rng.random(len(orbits)) < .25
            values[noise] = rng.integers(k, size=int(noise.sum()))
        values[rng.permutation(len(orbits))[:k]] = np.arange(k)
        out = np.empty(shape, dtype=np.int8)
        for value, orbit in zip(values, orbits):
            for p in orbit:
                out[p] = colors[value]
        if _visible_boundaries(out) and (not seam or _hidden_seam(out)):
            return out
    return None


def _raw_mask(rng, shape):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    kind = int(rng.integers(5))
    if kind == 0:
        return rng.random(shape) < rng.uniform(.02, .10)
    for _ in range(int(rng.integers(1, 4))):
        bh, bw = int(rng.integers(1, max(2, h // 3))), int(rng.integers(1, max(2, w // 3)))
        if kind == 1:
            bh = 1
        elif kind == 2:
            bw = 1
        r, c = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
        for dr in range(bh):
            reach = bw if kind != 3 else 1 + dr * (bw - 1) // max(1, bh - 1)
            mask[r + dr, c:c + reach] = True
    return mask


def _make_input(rng, latent, logical_axes, rb, cb, hard, extra=True):
    clean = np.repeat(np.repeat(latent, rb, axis=0), cb, axis=1)
    rs, cs = np.r_[0, np.cumsum(rb)], np.r_[0, np.cumsum(cb)]
    # The first and last pixels of a reflected pair determine its pixel fold.
    pixel_axes = []
    for starts, fold in ((rs, logical_axes[0]), (cs, logical_axes[1])):
        i = max(0, fold - (len(starts) - 2))
        j = fold - i
        pixel_axes.append(int(starts[i] + starts[j + 1] - 1))
    axes = tuple(pixel_axes)
    mask = _raw_mask(rng, clean.shape) if extra else np.zeros_like(clean, dtype=bool)
    four = [o for o in _orbits(latent.shape, logical_axes) if len(o) == 4]
    block_orbit = four[int(rng.integers(len(four)))]
    chosen = rng.permutation(4)
    for index, p in enumerate(block_orbit):
        r, c = p
        mask[rs[r]:rs[r + 1], cs[c]:cs[c + 1]] = bool(index in chosen[:3 if hard else 1])
    # A partial gap splits genuinely duplicated pixel rows AND columns.
    partial = [(r, c) for r in range(latent.shape[0]) for c in range(latent.shape[1])
               if rb[r] >= 2 and cb[c] >= 2 and (r, c) not in block_orbit]
    if not partial:
        return None
    r, c = partial[int(rng.integers(len(partial)))]
    mask[rs[r], cs[c]] = True
    for orbit in _orbits(clean.shape, axes):
        if all(mask[p] for p in orbit):
            # The forced whole-block orbit already has a complete survivor.
            mask[orbit[int(rng.integers(len(orbit)))]] = False
    inp = clean.copy()
    inp[mask] = 0
    return inp, clean, axes


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _sample_pair(rng, role, split, shift, shift_axis, exclude_shape=None):
    hard = role == "hard" or split == "test" or (role == "ordinary" and rng.random() < .35)
    for _ in range(128):
        plans = []
        for axis in (0, 1):
            half = 1 if role == "seam" else None
            fringe_max = 1 if role == "seam" else 2
            if shift == 0 and axis == shift_axis:
                half = 1 if split == "train" else int(rng.integers(2, 4))
                fringe_max = 1 if split == "train" else 2
            plans.append(_axis_plan(rng, half, fringe_max))
        shape = tuple(p[0] for p in plans)
        axes = tuple(p[1] for p in plans)
        if shape == exclude_shape:
            continue
        latent = _logical(rng, shape, axes, role == "seam")
        if latent is None:
            continue
        bands = []
        for axis in (0, 1):
            profile = ("moderate" if split == "train" else "extreme") if shift == 1 and axis == shift_axis else "free"
            bands.append(_scales(rng, shape[axis], axes[axis], profile))
        if role == "seam" and all(len(set(b)) == 1 for b in bands):
            continue
        made = _make_input(rng, latent, axes, *bands, hard)
        if made is None:
            continue
        inp, clean, pixel_axes = made
        try:
            predicted, repaired, inferred, _, _, _, _ = _analyze(inp)
        except ValueError:
            continue
        raw = _collapse(inp)[0]
        if (inferred == pixel_axes and np.array_equal(predicted, latent)
                and np.array_equal(repaired, clean)
                and all(raw.shape[d] > latent.shape[d] for d in (0, 1))):
            return {"input": inp, "output": latent.copy()}
    return _fallback_pair(rng, hard, split, shift, shift_axis, exclude_shape)


def _fallback_pair(rng, hard, split, shift, shift_axis, exclude_shape):
    """Constructive centered guard, with the same rule and held-out axis."""
    n = 5 if shift == 0 and split == "test" else 3
    m = 3 if (n, 5) == (exclude_shape if shift_axis == 0 else tuple(reversed(exclude_shape or ()))) else 5
    # Rows differ only at the middle column; columns still differ globally.
    latent = np.empty((n, m), dtype=np.int8)
    for r in range(n):
        for c in range(m):
            latent[r, c] = (c % 2) if c != m // 2 else (2 if r % 2 else 0)
    if shift_axis == 1:
        latent = latent.T.copy()
    colors = rng.choice(np.arange(1, 10, dtype=np.int8), 3, replace=False)
    latent = colors[latent]
    axes = tuple(x - 1 for x in latent.shape)
    bands = [np.full(x, 2, dtype=int) for x in latent.shape]
    if shift == 1 and split == "test":
        bands[shift_axis][0] = bands[shift_axis][-1] = 1
        bands[shift_axis][len(bands[shift_axis]) // 2] = 5
    else:
        bands[shift_axis][len(bands[shift_axis]) // 2] = 3
    made = _make_input(rng, latent, axes, *bands, hard, extra=False)
    assert made is not None
    inp, clean, pixel_axes = made
    assert np.array_equal(solve(inp), latent)
    assert all(_collapse(inp)[0].shape[d] > latent.shape[d] for d in (0, 1))
    return {"input": inp, "output": latent}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("Require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    shift, shift_axis = int(rng.integers(2)), int(rng.integers(2))
    roles = ["seam", "hard"] + ["ordinary"] * (num_train - 2)
    rng.shuffle(roles)
    puzzle, seen = {"train": [], "test": []}, set()
    first_shape = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            role = roles[index] if split == "train" else "hard"
            for _ in range(16):
                pair = _sample_pair(rng, role, split, shift, shift_axis,
                                    first_shape if split == "train" and index == 1 else None)
                key = _signature(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Distinct-input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                first_shape = pair["output"].shape
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, analysis):
    out, clean, (a, b), rows, cols, rb, cb = analysis
    h, w = inp.shape
    mask = inp == 0
    diagonal = top_bottom = left_right = fringe = 0
    for r, c in zip(*np.nonzero(mask)):
        rr, cc = a - r, b - c
        tb = 0 <= rr < h and rr != r and inp[rr, c] != 0
        lr = 0 <= cc < w and cc != c and inp[r, cc] != 0
        dg = 0 <= rr < h and 0 <= cc < w and inp[rr, cc] != 0
        diagonal += not tb and not lr and dg
        top_bottom += not lr
        left_right += not tb
        fringe += not 0 <= rr < h or not 0 <= cc < w
    full_gaps = 0
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            full_gaps += bool(np.all(mask[r:r + rb[i], c:c + cb[j]]))
    return {"diagonal_only": int(diagonal), "needs_top_bottom": int(top_bottom),
            "needs_left_right": int(left_right), "fringe_erased": int(fringe),
            "full_macro_gaps": int(full_gaps), "erased": int(mask.sum()),
            "offset_sign": (int(np.sign(a - h + 1)), int(np.sign(b - w + 1))),
            "parity": (a % 2, b % 2), "hidden_seam": bool(_hidden_seam(out)),
            "uneven": len(set(rb)) > 1 or len(set(cb)) > 1}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_features, shapes, profiles = set(), [], {"train": [], "test": []}, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, target = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(inp)
            assert key not in seen
            seen.add(key)
            analysis = _analyze(inp)
            out, clean, axes, rows, cols, rb, cb = analysis
            assert np.array_equal(out, target), "Incorrect repaired/compressed output"
            assert np.any(inp == 0) and not np.any(out == 0)
            assert 2 <= len(np.unique(out)) <= 9 and _visible_boundaries(out)
            assert np.array_equal(clean[inp != 0], inp[inp != 0])
            assert np.array_equal(np.repeat(np.repeat(out, rb, axis=0), cb, axis=1), clean)
            assert _infer_axes(clean) == axes
            assert all(out.shape[d] < inp.shape[d] for d in (0, 1))
            # Direct compression breaks both axes, proving the order matters.
            raw = _collapse(inp)[0]
            assert all(raw.shape[d] > out.shape[d] for d in (0, 1))
            f = _features(inp, analysis)
            assert f["full_macro_gaps"] > 0, "Include a gap without local block witnesses"
            if split == "train":
                train_features.append(f)
            else:
                assert f["diagonal_only"] > 0, "Tests must exercise both folds"
            shapes[split].append(out.shape)
            profiles[split].append((rb, cb))
    assert len(set(shapes["train"])) >= 2
    assert any(f["hidden_seam"] and f["uneven"] for f in train_features)
    assert any(f["diagonal_only"] > 0 for f in train_features)
    size_shift = any(min(s[d] for s in shapes["test"]) > max(s[d] for s in shapes["train"]) for d in (0, 1))
    scale_shift = any(all(min(p[d]) >= 2 and max(p[d]) <= 4 for p in profiles["train"])
                      and all(min(p[d]) == 1 and max(p[d]) >= 5 for p in profiles["test"])
                      for d in (0, 1))
    assert size_shift or scale_shift, "Missing documented train/test shift"


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 streaming seeds, semantic ablations, count boundaries; no files."""
    shapes, output_shapes, structures, offsets, parity, palettes, profiles = (set() for _ in range(7))
    signatures, features = set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                analysis = _analyze(inp)
                out, clean, axes, rows, cols, rb, cb = analysis
                assert np.array_equal(solve(inp.copy()), pair["output"])
                assert not np.array_equal(clean, out)  # omit A
                assert not np.array_equal(_collapse(inp)[0], out)  # omit B
                # Separated recurrences are retained, unlike global deduplication.
                assert len({row.tobytes() for row in out}) < len(out)
                assert len({col.tobytes() for col in out.T}) < out.shape[1]
                f = _features(inp, analysis)
                f["split"] = split
                features.append(f)
                offsets.add(f["offset_sign"])
                parity.add(f["parity"])
                shapes.add(inp.shape)
                output_shapes.add(out.shape)
                palettes.add(len(np.unique(out)))
                profiles.add((rb, cb))
                colors = {}
                normalized = bytes(colors.setdefault(int(v), len(colors)) for v in inp.flat)
                structures.add((inp.shape, hashlib.sha256(normalized).hexdigest()))
    assert len(signatures) == 42 and len(shapes) >= 50 and len(output_shapes) >= 12
    assert len(structures) >= 180 and len(profiles) >= 150
    assert len(offsets) == 9 and len(parity) == 4 and len(palettes) >= 6
    assert any(f["fringe_erased"] for f in features)
    assert any(f["diagonal_only"] == 0 for f in features if f["split"] == "train")
    assert _puzzle_digest(generate_puzzle(17)) == _puzzle_digest(generate_puzzle(17))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(31000 + 10 * nt + ns, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Exercise the constructive guard in both shift regimes and orientations.
    for shift in (0, 1):
        for axis in (0, 1):
            for split in ("train", "test"):
                pair = _fallback_pair(np.random.default_rng(123), True, split, shift, axis, None)
                assert np.array_equal(solve(pair["input"]), pair["output"])
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted a damaged output")
    return {"seeds": 42, "pairs": len(features), "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "normalized_inputs": len(structures), "scale_profiles": len(profiles),
            "offset_signs": len(offsets), "fold_parities": len(parity),
            "payload_cardinalities": sorted(palettes),
            "diagonal_only_pairs": sum(f["diagonal_only"] > 0 for f in features),
            "fringe_erasure_pairs": sum(f["fringe_erased"] > 0 for f in features),
            "semantic_verification": "generator-owned checks, no independent solver"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
