"""Infer a damaged field's perpendicular folds, repair it, then shear its rows.

Only standard library and NumPy are used. No evidence files are loaded at runtime.
The erasure color is also the padding color; all three roles (color and two
fold sums) must be uniquely recoverable from each input itself.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256

import numpy as np


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected a rectangular integer ARC grid")
    return grid.astype(np.int8, copy=True)


@lru_cache(maxsize=32)
def _axis_pairs(length):
    result = []
    # Fold at sum/2. Each candidate has at least one pair of distinct indices.
    for total in range(1, 2 * length - 2):
        left = np.array([p for p in range(length) if p < total - p < length])
        result.append((total, left, total - left))
    return tuple(result)


def _axis_candidates(grid, erase, dimension):
    view = grid if dimension == 0 else grid.T
    result = []
    for total, left, right in _axis_pairs(len(view)):
        a, b = view[left], view[right]
        visible = (a != erase) & (b != erase)
        # No score, centered-axis preference, minimum-overlap filter or tie break.
        if np.all(a[visible] == b[visible]):
            result.append(total)
    return result


@lru_cache(maxsize=128)
def _orbits(height, width, row_sum, col_sum):
    seen, result = set(), []
    for r in range(height):
        for c in range(width):
            if (r, c) in seen:
                continue
            rr, cc = row_sum - r, col_sum - c
            rows = {r, rr} if 0 <= rr < height else {r}
            cols = {c, cc} if 0 <= cc < width else {c}
            orbit = tuple(sorted((y, x) for y in rows for x in cols))
            seen.update(orbit)
            result.append(orbit)
    return tuple(result)


def _complete(grid, erase, row_sum, col_sum):
    restored = grid.copy()
    for orbit in _orbits(*grid.shape, row_sum, col_sum):
        values = {int(grid[p]) for p in orbit if grid[p] != erase}
        if len(values) != 1:
            return None
        color = next(iter(values))
        for p in orbit:
            restored[p] = color
    return restored


def _infer(grid):
    """Exhaust every present marker and every nontrivial in-bounds fold pair."""
    answer = None
    for erase in (int(c) for c in np.unique(grid)):
        row_axes = _axis_candidates(grid, erase, 0)
        col_axes = _axis_candidates(grid, erase, 1)
        for row_sum in row_axes:
            for col_sum in col_axes:
                restored = _complete(grid, erase, row_sum, col_sum)
                if restored is None:
                    continue
                if answer is not None:
                    raise ValueError("ambiguous erasure color or reflection folds")
                answer = erase, row_sum, col_sum, restored
    if answer is None:
        raise ValueError("no recoverable perpendicular-fold interpretation")
    return answer


def _shear(restored, padding):
    height, width = restored.shape
    output = np.full((height, height + width - 1), padding, dtype=np.int8)
    for r in range(height):
        start = height - 1 - r
        output[r, start:start + width] = restored[r]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only prediction; never modifies the caller's grid."""
    grid = _grid(input_grid)
    if min(grid.shape) < 2 or sum(grid.shape) - 1 > 30:
        raise ValueError("two folds and an ARC-sized sheared canvas are required")
    erase, _, _, restored = _infer(grid)
    return _shear(restored, erase)


def _shape(rng, large):
    # Tests extrapolate canvas width and stretch aspect ratios under the same rule.
    if large:
        total = int(rng.integers(27, 32))
        height = int(rng.integers(6, total - 5))
        return height, total - height
    height = int(rng.integers(6, 13))
    width = int(rng.integers(6, 13))
    return height, width


def _geometry(rng, shape, require_offset=False):
    h, w = shape
    dr, dc = (int(x) for x in rng.integers(-2, 3, size=2))
    if require_offset and dr == dc == 0:
        dr = int(rng.choice([-2, -1, 1, 2]))
    return h - 1 + dr, w - 1 + dc


def _field(rng, shape, axes, colors, style):
    """Assign texture to orbit representatives, not to a parent puzzle output."""
    orbits = _orbits(*shape, *axes)
    reps = np.array([o[0] for o in orbits])
    r, c = reps.T
    k = len(colors)
    bh, bw = (int(x) for x in rng.integers(1, 4, size=2))
    phase = int(rng.integers(k))
    if style == 0:
        values = rng.integers(k, size=len(orbits))
    elif style == 1:
        blocks = rng.integers(k, size=shape)
        values = blocks[r // bh, c // bw]
    elif style == 2:
        values = (r // bh + c // bw + phase) % k
    elif style == 3:
        center = reps[int(rng.integers(len(reps)))]
        values = (np.maximum(abs(r - center[0]), abs(c - center[1])) // bh + phase) % k
    elif style == 4:
        values = ((r + int(rng.choice([-2, -1, 1, 2])) * c) // bw + phase) % k
    else:
        values = np.full(len(orbits), phase, dtype=int)
        for _ in range(int(rng.integers(3, 8))):
            r0, c0 = reps[int(rng.integers(len(reps)))]
            rh, cw = (int(x) for x in rng.integers(1, 5, size=2))
            region = (r >= r0) & (r < r0 + rh) & (c >= c0) & (c < c0 + cw)
            values[region] = int(rng.integers(k))
    # Noise is optional; a clean motif can survive without any alterations.
    if style and rng.random() < 0.4:
        speckles = rng.random(len(orbits)) < rng.uniform(0.04, 0.18)
        values[speckles] = rng.integers(k, size=int(speckles.sum()))
    field = np.empty(shape, dtype=np.int8)
    for orbit, index in zip(orbits, values):
        for p in orbit:
            field[p] = colors[int(index)]
    return field


def _mask(rng, field, axes, difficulty):
    h, w = field.shape
    family = int(rng.integers(5))
    mask = np.zeros((h, w), dtype=bool)
    if family == 0:
        mask = rng.random((h, w)) < rng.uniform(0.04, 0.16)
    else:
        for _ in range(int(rng.integers(1, 4))):
            r0, c0 = int(rng.integers(h)), int(rng.integers(w))
            rh, cw = int(rng.integers(1, max(3, h // 3 + 1))), int(rng.integers(1, max(3, w // 3 + 1)))
            if family == 2:
                rh, cw = 1, int(rng.integers(3, max(4, w // 2 + 1)))
            elif family == 3:
                rh, cw = int(rng.integers(3, max(4, h // 2 + 1))), 1
            if family == 4:
                for dr in range(min(rh, h - r0)):
                    mask[r0 + dr, c0:min(w, c0 + 1 + dr)] = True
            else:
                mask[r0:min(h, r0 + rh), c0:min(w, c0 + cw)] = True
    orbits = _orbits(h, w, *axes)
    for orbit in orbits:
        points = [p for p in orbit if mask[p]]
        if difficulty == 0 and len(points) > 1:
            keep = points[int(rng.integers(len(points)))]
            for p in points:
                mask[p] = p == keep
        if all(mask[p] for p in orbit):
            mask[orbit[int(rng.integers(len(orbit)))]] = False
    four = [o for o in orbits if len(o) == 4]
    # Two different damaged colors prevent swapping the payload/marker roles.
    order = list(rng.permutation(len(four)))
    selected = []
    for i in order:
        orbit = four[int(i)]
        if not selected or field[orbit[0]] != field[selected[0][0]]:
            selected.append(orbit)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        return None
    if difficulty == 2:
        remaining = [four[int(i)] for i in order if four[int(i)] not in selected]
        selected.extend(remaining[:min(len(remaining), int(rng.integers(1, 4)))])
    for orbit in selected:
        survivor = orbit[int(rng.integers(4))]
        single = orbit[(orbit.index(survivor) + 1) % 4]
        for p in orbit:
            mask[p] = p != survivor if difficulty else p == single
    # Include one-fold fringe or through-axis repair when available, without
    # forcing boundary contact or any particular corner to reveal the marker.
    two = [o for o in orbits if len(o) == 2]
    if two and rng.random() < 0.65:
        orbit = two[int(rng.integers(len(two)))]
        mask[orbit[0]], mask[orbit[1]] = True, False
    return mask


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _make_pair(rng, shape, axes, colors, erase, difficulty, seen):
    style = int(rng.integers(6))
    # Retries retain the requested geometry and difficulty. Later attempts use
    # independent orbit colors, still subject to exactly the same evidence rules.
    for attempt in range(128):
        field = _field(rng, shape, axes, colors, style if attempt < 24 else 0)
        if any(_axis_candidates(field, erase, d) != [axes[d]] for d in (0, 1)):
            continue
        mask = _mask(rng, field, axes, difficulty)
        if mask is None:
            continue
        inp = field.copy()
        inp[mask] = erase
        if _signature(inp) in seen:
            continue
        try:
            marker, a, b, repaired = _infer(inp)
        except ValueError:
            continue
        if (marker, a, b) != (erase, *axes) or not np.array_equal(repaired, field):
            continue
        # Strong, multicolor surviving evidence is a generation quality check,
        # never a solver filter that could conceal a weak competing fold.
        if not _multicolor_evidence(inp, erase, axes):
            continue
        seen.add(_signature(inp))
        return {"input": inp, "output": _shear(field, erase)}
    raise RuntimeError("bounded construction exhausted without unique visible folds")


def _multicolor_evidence(grid, erase, axes):
    for dimension, total in enumerate(axes):
        view = grid if dimension == 0 else grid.T
        _, left, right = next(p for p in _axis_pairs(len(view)) if p[0] == total)
        a, b = view[left], view[right]
        visible = (a != erase) & (b != erase)
        if np.count_nonzero(visible) < 4 or len(np.unique(a[visible])) < 2:
            return False
    return True


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    num_train = _integer("num_train", num_train, 2, 10)
    num_test = _integer("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    erase = int(rng.integers(10))
    payload = [c for c in range(10) if c != erase]
    # Optional test-local colors always retain their own witnesses in the input.
    held_out = [int(rng.choice(payload))] if rng.random() < 0.45 else []
    train_pool = [c for c in payload if c not in held_out]
    puzzle, seen = {"train": [], "test": []}, set()
    training_geometry = []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            shape = _shape(rng, split == "test")
            if split == "train" and index == 1:
                old_h, old_w = puzzle["train"][0]["input"].shape
                h, w = shape
                shape = (6 + (h - 5) % 7 if h == old_h else h,
                         6 + (w - 5) % 7 if w == old_w else w)
            axes = _geometry(rng, shape, require_offset=split == "train" and index == 1)
            if split == "train" and index == 1:
                previous = training_geometry[0]
                if (axes[0] - shape[0] + 1, axes[1] - shape[1] + 1) == previous:
                    axes = (shape[0] - 1 + (1 if previous[0] != 1 else -1), axes[1])
            pool = train_pool if split == "train" else payload
            color_count = int(rng.integers(2, len(pool) + 1))
            colors = rng.choice(pool, size=color_count, replace=False)
            if split == "test" and held_out:
                others = [c for c in pool if c != held_out[0]]
                colors = np.array([held_out[0], *rng.choice(others, size=color_count - 1, replace=False)])
            difficulty = 2 if split == "test" else (0 if index == 0 else 1)
            pair = _make_pair(rng, shape, axes, colors, erase, difficulty, seen)
            puzzle[split].append(pair)
            if split == "train":
                training_geometry.append((axes[0] - shape[0] + 1, axes[1] - shape[1] + 1))
    # Presentation order carries no semantic information.
    rng.shuffle(puzzle["train"])
    return puzzle


def _features(grid, erase, axes):
    mask = grid == erase
    h, w = grid.shape
    a, b = axes
    diagonal = fringe = through = horizontal_needed = vertical_needed = 0
    for r, c in zip(*np.nonzero(mask)):
        rr, cc = a - r, b - c
        horizontal = 0 <= rr < h and not mask[rr, c]
        vertical = 0 <= cc < w and not mask[r, cc]
        diagonal += (not horizontal and not vertical and 0 <= rr < h and 0 <= cc < w
                     and not mask[rr, cc])
        horizontal_needed += not vertical
        vertical_needed += not horizontal
        fringe += not (0 <= rr < h and 0 <= cc < w)
        through += rr == r or cc == c
    return {"erased": int(mask.sum()), "diagonal_only": int(diagonal),
            "horizontal_needed": int(horizontal_needed), "vertical_needed": int(vertical_needed),
            "fringe_erased": int(fringe), "axis_erased": int(through),
            "edges": tuple(bool(x) for x in (mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, markers, training_shapes, training_offsets = set(), set(), set(), set()
    hard_demo = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            h, w = inp.shape
            assert min(h, w) >= 6 and h + w - 1 <= 30
            assert _signature(inp) not in seen
            seen.add(_signature(inp))
            erase, a, b, restored = _infer(inp)
            markers.add(erase)
            assert _multicolor_evidence(inp, erase, (a, b))
            assert out.shape == (h, h + w - 1)
            assert np.any(inp == erase) and not np.any(restored == erase)
            assert 2 <= len(np.unique(restored)) <= 9
            assert np.array_equal(restored[inp != erase], inp[inp != erase])
            assert len(np.unique(restored[inp == erase])) >= 2
            for r in range(h):
                start = h - 1 - r
                assert np.all(out[r, :start] == erase)
                assert np.all(out[r, start + w:] == erase)
                assert np.array_equal(out[r, start:start + w], restored[r])
            for orbit in _orbits(h, w, a, b):
                assert len({int(inp[p]) for p in orbit if inp[p] != erase}) == 1
                if len(orbit) == 1:
                    assert inp[orbit[0]] != erase
            if split == "train":
                training_shapes.add((h, w))
                training_offsets.add((a - h + 1, b - w + 1))
                hard_demo |= _features(inp, erase, (a, b))["diagonal_only"] > 0
    assert len(markers) == 1
    assert hard_demo and len({h for h, w in training_shapes}) >= 2
    assert len({w for h, w in training_shapes}) >= 2 and len(training_offsets) >= 2


def _puzzle_digest(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _direct_prediction(grid, erase, a, b):
    """Alternate coordinate implementation, using only three direct lookups."""
    h, w = grid.shape
    rows = []
    for r in range(h):
        row = []
        for c in range(w):
            color = int(grid[r, c])
            if color == erase:
                values = {int(grid[y, x]) for y, x in ((a - r, c), (r, b - c), (a - r, b - c))
                          if 0 <= y < h and 0 <= x < w and grid[y, x] != erase}
                assert len(values) == 1
                color = values.pop()
            row.append(color)
        rows.append([erase] * (h - 1 - r) + row + [erase] * r)
    return np.array(rows, dtype=np.int8)


def self_test():
    """42 streamed seeds; bounded caches, no saved payloads or disk writes."""
    digests, shapes, offsets, parities, markers, palettes, normalized = (set() for _ in range(7))
    totals = {s: {k: 0 for k in ("pairs", "diagonal_only", "fringe_erased", "axis_erased", "interior")}
              for s in ("train", "test")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        assert max(sum(p["input"].shape) for p in puzzle["train"]) < min(sum(p["input"].shape) for p in puzzle["test"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                erase, a, b, restored = _infer(inp)
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
                assert np.array_equal(_direct_prediction(inp, erase, a, b), out)
                # Each parent is causally necessary in every generated example.
                assert not np.array_equal(_shear(inp, erase), out)
                assert out.shape != restored.shape
                h, w = inp.shape
                shapes.add((h, w))
                offsets.add((int(np.sign(a - h + 1)), int(np.sign(b - w + 1))))
                parities.add((a % 2, b % 2))
                markers.add(erase)
                palettes.add(len(np.unique(restored)))
                role_map = {}
                normalized.add((inp.shape, bytes(role_map.setdefault(int(c), len(role_map)) for c in inp.flat)))
                f = _features(inp, erase, (a, b))
                totals[split]["pairs"] += 1
                for name in ("diagonal_only", "fringe_erased", "axis_erased"):
                    totals[split][name] += f[name] > 0
                totals[split]["interior"] += not any(f["edges"])
    assert len(digests) == 42 and len(normalized) == 210
    assert len(shapes) >= 50 and len(markers) >= 8 and len(palettes) >= 6
    assert len(offsets) == 9 and len(parities) == 4
    assert totals["train"]["diagonal_only"] >= 42 and totals["test"]["diagonal_only"] == 42
    for split in totals:
        assert totals[split]["fringe_erased"] >= 5
        assert totals[split]["axis_erased"] >= 5
        assert totals[split]["interior"] >= 1
    assert _puzzle_digest(generate_puzzle(0)) == _puzzle_digest(generate_puzzle(0))
    for nt, ns in ((2, 1), (10, 4), (5, 3)):
        puzzle = generate_puzzle(1234, nt, ns)
        validate_generated_puzzle(puzzle)
        assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(0)
    # A color permutation must also permute the inferred padding role.
    permutation = np.random.default_rng(456).permutation(10)
    pair = sample["test"][0]
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    ambiguous = np.ones((6, 6), dtype=np.int8)
    ambiguous[1, 2] = 0
    for invalid in (ambiguous, np.zeros((6, 6), dtype=np.int8)):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous or witness-free input accepted")
    for point in ((0, 0), (pair["output"].shape[0] - 1, 0)):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
        bad["test"][0]["output"][point] = (int(bad["test"][0]["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted corrupted padding or payload")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "markers": len(markers),
            "palette_sizes": sorted(palettes), "offset_signs": len(offsets), "parities": len(parities),
            "normalized_structures": len(normalized), "features": totals}
