"""Recover a crop-selected 20x20 reference, then apply a bottom-anchored shear.

Inputs visibly contain the reference and crop. No learned image, examples, RNG,
or generation metadata is used by solve. All diagnostics are bounded and in
memory; this module writes no artifacts or execution traces.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _d4(grid, index):
    # Four rotations followed by the four rotations of a left/right reflection.
    return np.rot90(grid if index < 4 else np.fliplr(grid), index % 4).copy()


def _connected(mask):
    points = set(zip(*np.nonzero(mask)))
    if not points:
        return False
    todo = [points.pop()]
    while todo:
        r, c = todo.pop()
        for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if point in points:
                points.remove(point)
                todo.append(point)
    return not points


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _rectangular_frames(grid):
    """Find monochrome 4-connected components that are exact hollow frames."""
    found = []
    for color in np.unique(grid):
        remaining = set(zip(*np.nonzero(grid == color)))
        while remaining:
            first = remaining.pop()
            todo, cells = [first], [first]
            while todo:
                r, c = todo.pop()
                for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if point in remaining:
                        remaining.remove(point)
                        todo.append(point)
                        cells.append(point)
            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells) + 1
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells) + 1
            h, w = r1 - r0, c1 - c0
            if (h >= 4 and w >= 4 and len(cells) == 2 * h + 2 * w - 4
                    and all(r in (r0, r1 - 1) or c in (c0, c1 - 1) for r, c in cells)):
                found.append((int(color), r0, r1, c0, c1))
    return found


def _decode(input_grid):
    grid = _grid(input_grid)
    frames = _rectangular_frames(grid)
    interpretations = []
    for primary in frames:
        marker, r0, r1, c0, c1 = primary
        if (r1 - r0, c1 - c0) != (22, 22):
            continue
        for secondary in frames:
            m, a, b, c, d = secondary
            h, w = b - a - 2, d - c - 2
            if (secondary == primary or m != marker or not (2 <= h <= 20 and 2 <= w <= 20)
                    or min(h, w) > 5):
                continue
            if not (b < r0 or r1 < a or d < c0 or c1 < c):
                continue  # A full paper row/column separates the two frames.
            outside = np.ones(grid.shape, dtype=bool)
            outside[r0:r1, c0:c1] = False
            outside[a:b, c:d] = False
            paper_colors = np.unique(grid[outside])
            ref = grid[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
            patch = grid[a + 1:b - 1, c + 1:d - 1]
            ref_colors, patch_colors = np.unique(ref), np.unique(patch)
            if len(paper_colors) != 1 or len(ref_colors) != 2 or len(patch_colors) != 2:
                continue
            roles = [marker, int(paper_colors[0]), *map(int, ref_colors), *map(int, patch_colors)]
            if len(set(roles)) != 6:
                continue
            background = int(ref[0, 0])
            if not (np.all(ref[0] == background) and np.all(ref[-1] == background)
                    and np.all(ref[:, 0] == background) and np.all(ref[:, -1] == background)):
                continue
            binary = (ref != background).astype(np.int8)
            if not _connected(binary):
                continue
            interpretations.append((binary, patch.copy(), tuple(map(int, patch_colors))))
    if len(interpretations) != 1:
        raise ValueError("input must have one reference frame and one separated crop frame")
    return interpretations[0]


def _locations(full, patch):
    h, w = patch.shape
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    return np.argwhere(np.all(windows == patch, axis=(-2, -1)))


def _matches(reference, patch):
    matches = []
    for orientation in range(8):
        full = _d4(reference, orientation)
        for swap in (0, 1):
            for r, c in _locations(full, patch if swap == 0 else 1 - patch):
                matches.append((orientation, swap, int(r), int(c)))
    return matches


def _recover(input_grid):
    reference, patch, colors = _decode(input_grid)
    normalized = (patch == colors[1]).astype(np.int8)
    matches = _matches(reference, normalized)
    if len(matches) != 1:
        raise ValueError("crop must have one orientation, role assignment, and offset")
    orientation, swap, r, c = matches[0]
    roles = colors if swap == 0 else colors[::-1]
    return reference, _d4(reference, orientation), patch, roles, (orientation, r, c)


def _shear(mask, anchor=None, direction=-1, rate=1):
    rows, cols = np.nonzero(mask)
    if anchor is None:
        anchor = int(rows.max())
    destination = cols + direction * rate * (anchor - rows)
    keep = (destination >= 0) & (destination < mask.shape[1])
    out = np.zeros(mask.shape, dtype=np.int8)
    out[rows[keep], destination[keep]] = 1
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Match the visible swatch, complete the board, then shear its foreground."""
    _, complete, _, (background, foreground), _ = _recover(input_grid)
    return np.where(_shear(complete), foreground, background).astype(np.int8)


def _line(mask, start, end):
    r0, c0 = start
    r1, c1 = end
    mask[min(r0, r1):max(r0, r1) + 1, c0] = 1
    mask[r1, min(c0, c1):max(c0, c1) + 1] = 1


def _ring(mask, r0, r1, c0, c1):
    mask[r0:r1 + 1, c0] = mask[r0:r1 + 1, c1] = 1
    mask[r0, c0:c1 + 1] = mask[r1, c0:c1 + 1] = 1


def _structured_shape(rng, h, w, style):
    a = np.zeros((h, w), dtype=np.int8)
    if style == 0:  # An unequal loop network with an inner branch/ladder.
        _ring(a, 0, h - 1, 0, w - 1)
        spine = int(rng.integers(2, w - 1))
        a[1:h - 1, spine] = 1
        for r in range(2, h - 1, int(rng.integers(2, 4))):
            a[r, :spine + 1] = 1
    elif style == 1:  # A tree/comb with independently sized bars and tips.
        spine = int(rng.integers(1, w - 1))
        a[:, spine] = 1
        for r in range(0, h, int(rng.integers(1, 3))):
            lo = int(rng.integers(0, spine + 1))
            hi = int(rng.integers(spine + 1, w + 1))
            a[r, lo:hi] = 1
        a[h - 1, int(rng.integers(spine + 1)):spine + 1] = 1
    elif style == 2:  # A thick winding ribbon, with connected carved holes.
        previous = int(rng.integers(1, w - 2))
        for r in range(h):
            center = int(np.clip(previous + rng.integers(-2, 3), 1, w - 2))
            lo = max(0, min(previous, center) - int(rng.integers(0, 3)))
            hi = min(w, max(previous, center) + int(rng.integers(2, 5)))
            a[r, lo:hi] = 1
            previous = center
        for r, c in rng.integers([1, 1], [h - 1, w - 1], size=(h, 2)):
            if a[r, c]:
                a[r, c] = 0
                if not _connected(a):
                    a[r, c] = 1
    elif style == 3:  # Several irregularly placed loops joined by bent paths.
        origin = (int(rng.integers(h)), int(rng.integers(w)))
        a[origin] = 1
        for _ in range(int(rng.integers(2, 5))):
            r0 = int(rng.integers(h - 2))
            c0 = int(rng.integers(w - 2))
            r1 = int(rng.integers(r0 + 2, h))
            c1 = int(rng.integers(c0 + 2, w))
            _line(a, origin, (r0, c0))
            _ring(a, r0, r1, c0, c1)
            origin = (r1, c1)
    elif style == 4:  # A stepped backbone with unequal side branches.
        previous = int(rng.integers(w))
        a[0, previous] = 1
        for r in range(1, h):
            center = int(np.clip(previous + rng.choice((-1, 1)), 0, w - 1))
            _line(a, (r - 1, previous), (r, center))
            if rng.random() < .7:
                _line(a, (r, center), (r, int(rng.integers(w))))
            previous = center
    else:  # Bars, a stem, a loop and a hooked extremity in one figure.
        a[:, 1] = 1
        for r in range(0, h, int(rng.integers(2, 4))):
            a[r, 1:int(rng.integers(3, w + 1))] = 1
        r0 = int(rng.integers(0, h - 3))
        _ring(a, r0, h - 2, 1, w - 1)
        _line(a, (h - 2, w - 1), (h - 1, max(0, w - 3)))
    rows, cols = np.nonzero(a)
    return a[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _reference(rng):
    """Bounded construction with both clipped and retained adjacent rotations."""
    for _ in range(96):
        h, w = int(rng.integers(6, 14)), int(rng.integers(6, 15))
        a = _structured_shape(rng, h, w, int(rng.integers(6)))
        h, w = a.shape
        if min(h, w) < 4 or not _connected(a):
            continue
        top, left = int(rng.integers(1, 20 - h)), int(rng.integers(1, 20 - w))
        full = np.zeros((20, 20), dtype=np.int8)
        full[top:top + h, left:left + w] = a
        orbit = [_d4(full, i) for i in range(8)]
        if len({_key(g) for g in orbit}) != 8:
            continue
        mass = int(full.sum())
        sheared = [_shear(g) for g in orbit]
        if any(g.sum() < max(10, .30 * mass) or np.array_equal(g, f)
               for f, g in zip(orbit, sheared)):
            continue
        clipped = [int(g.sum()) < mass for g in sheared]
        choices = [(base, quarter) for base in range(4) for quarter in (1, 3)
                   if clipped[base] != clipped[(base + quarter) % 4]]
        if choices:
            base, quarter = choices[int(rng.integers(len(choices)))]
            return orbit[base], quarter
    return _fallback_reference(rng)


def _fallback_reference(rng):
    # A visible geometric fallback, not a hidden lookup image used by solve.
    # The randomized interior preserves the scaffold's identifying asymmetry.
    full = np.zeros((20, 20), dtype=np.int8)
    _ring(full, 4, 11, 9, 17)
    full[6:12, 12] = 1
    full[8, 9:13] = 1
    full[4:7, 15] = 1
    full[10, int(rng.integers(13, 16)):18] = 1
    return full, 1


def _choose_crop(rng, reference, orientation, training, include_anchor, seen):
    full = _d4(reference, orientation)
    bottom = int(np.nonzero(full)[0].max())
    expected = _shear(full)
    for attempt in range(192):
        short = int(rng.choice((4, 5) if training else (3, 4)))
        long = int(rng.integers(16, 21) if training else rng.integers(10, 16))
        h, w = (short, long) if rng.random() < .5 else (long, short)
        r, c = int(rng.integers(21 - h)), int(rng.integers(21 - w))
        patch = full[r:r + h, c:c + w].copy()
        if len(np.unique(patch)) != 2:
            continue
        observed = np.zeros((20, 20), dtype=np.int8)
        observed[r:r + h, c:c + w] = patch
        visible_bottom = int(np.nonzero(observed)[0].max())
        if include_anchor is not None and (visible_bottom == bottom) != include_anchor:
            continue
        # At least three surviving foreground cells must actually be completed.
        if int(np.count_nonzero(expected & (1 - _shear(observed, anchor=bottom)))) < 3:
            continue
        if np.array_equal(expected, _shear(observed)):
            continue
        key = (orientation, r, c, h, w)
        if key in seen:
            continue
        matches = _matches(reference, patch)
        if matches == [(orientation, 0, r, c)]:
            seen.add(key)
            return patch
    return None


def _frame(core, marker):
    result = np.full((core.shape[0] + 2, core.shape[1] + 2), marker, dtype=np.int8)
    result[1:-1, 1:-1] = core
    return result


def _pack(rng, reference, patch, colors):
    paper, marker, rb, rf, qb, qf = colors
    primary = _frame(np.where(reference, rf, rb), marker)
    secondary = _frame(np.where(patch, qf, qb), marker)
    sh, sw = secondary.shape
    if sh <= 7:
        canvas = np.full((23 + sh, 22), paper, dtype=np.int8)
        c = int(rng.integers(23 - sw))
        if rng.random() < .5:
            canvas[:sh, c:c + sw] = secondary
            canvas[sh + 1:sh + 23, :] = primary
        else:
            canvas[:22, :] = primary
            canvas[23:23 + sh, c:c + sw] = secondary
    else:
        canvas = np.full((22, 23 + sw), paper, dtype=np.int8)
        r = int(rng.integers(23 - sh))
        if rng.random() < .5:
            canvas[r:r + sh, :sw] = secondary
            canvas[:, sw + 1:sw + 23] = primary
        else:
            canvas[:, :22] = primary
            canvas[r:r + sh, 23:23 + sw] = secondary
    return canvas


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    # Retry the shared reference, never store an inconsistent partial puzzle.
    for _ in range(32):
        reference, quarter = _reference(rng)
        first = [(0, True), (quarter, False)]
        if rng.random() < .5:
            first = [(0, False), (quarter, True)]
        if nt >= 3:
            first.append((int(rng.integers(4, 8)), None))
        pool = list(range(4 if nt == 2 else 8))
        train_specs = first + [(int(rng.choice(pool)), None) for _ in range(nt - len(first))]
        rng.shuffle(train_specs)
        demonstrated = {i for i, _ in train_specs}
        held_out = [i for i in pool if i not in demonstrated]
        test_pool = held_out if held_out else pool
        test_specs = [(int(rng.choice(test_pool)), False) for _ in range(nq)]
        puzzle, used_crops, used_inputs = {"train": [], "test": []}, set(), set()
        fixed = tuple(map(int, rng.permutation(10)[:6]))
        stable = bool(rng.integers(2))
        complete = True
        for index, (orientation, anchor_status) in enumerate(train_specs + test_specs):
            training = index < nt
            patch = _choose_crop(rng, reference, orientation, training, anchor_status, used_crops)
            if patch is None:
                complete = False
                break
            if stable:
                colors = fixed
            else:
                # Paper and marker stay fixed; both two-color palettes may vary.
                available = [c for c in range(10) if c not in fixed[:2]]
                colors = (*fixed[:2], *map(int, rng.choice(available, 4, replace=False)))
            inp = _pack(rng, reference, patch, colors)
            if _key(inp) in used_inputs:
                complete = False
                break
            used_inputs.add(_key(inp))
            result = np.where(_shear(_d4(reference, orientation)), colors[5], colors[4]).astype(np.int8)
            puzzle["train" if training else "test"].append({"input": inp, "output": result})
        if complete:
            validate_generated_puzzle(puzzle)
            return puzzle
    raise RuntimeError("bounded reference/crop search exhausted")


def _facts(pair):
    reference, full, patch, roles, (orientation, r, c) = _recover(pair["input"])
    h, w = patch.shape
    observed = np.zeros((20, 20), dtype=np.int8)
    observed[r:r + h, c:c + w] = (patch == roles[1])
    rows, cols = np.nonzero(full)
    bottom = int(rows.max())
    expected = np.full((20, 20), roles[0], dtype=np.int8)
    # Cell-by-cell oracle for the shear, separate from its vectorized constructor.
    for rr, cc in zip(rows, cols):
        target = int(cc) - (bottom - int(rr))
        if target >= 0:
            expected[int(rr), target] = roles[1]
    out = _grid(pair["output"])
    assert np.array_equal(out, expected), "wrong recovered-and-sheared answer"
    mask = (out == roles[1]).astype(np.int8)
    assert np.array_equal(mask[bottom], full[bottom]), "lowest occupied row must stay fixed"
    assert np.all(mask[bottom + 1:] == 0)
    assert not np.array_equal(mask, full), "omitting B must change the answer"
    assert not np.array_equal(mask, _shear(observed)), "omitting A must change the answer"
    assert int(np.count_nonzero(mask & (1 - _shear(observed, anchor=bottom)))) >= 3
    assert not np.array_equal(mask, _shear(full, anchor=19)), "canvas bottom is the wrong anchor"
    assert not np.array_equal(mask, _shear(full, anchor=int(rows.min()), direction=1))
    assert not np.array_equal(mask, _shear(full, direction=1)), "right shear must differ"
    assert not np.array_equal(mask, _shear(full, rate=2)), "unit rate must be exposed"
    visible_bottom = int(np.nonzero(observed)[0].max())
    if visible_bottom != bottom:
        assert not np.array_equal(mask, _shear(full, anchor=visible_bottom)), "the crop cannot supply the anchor"
    assert int(mask.sum()) >= max(10, .30 * int(full.sum()))
    assert np.array_equal(solve(pair["input"]), out)
    return {"reference": _key(reference), "orientation": orientation, "crop": (h, w),
            "offset": (r, c), "anchor": bottom, "clipped": int(full.sum() - mask.sum()),
            "anchor_visible": visible_bottom == bottom,
            "palette": roles, "shape": pair["input"].shape,
            "reference_mass": int(full.sum())}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, references, facts = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert out.shape == (20, 20) and _key(inp) not in seen
            seen.add(_key(inp))
            fact = _facts({"input": inp, "output": out})
            references.add(fact["reference"])
            facts[split].append(fact)
    assert len(references) == 1, "each puzzle has one shared visible reference"
    train, test = facts["train"], facts["test"]
    assert any(f["clipped"] for f in train) and any(not f["clipped"] for f in train)
    assert {f["anchor_visible"] for f in train} == {False, True}
    assert all(not f["anchor_visible"] for f in test)
    assert max(h * w for h, w in (f["crop"] for f in test)) < min(h * w for h, w in (f["crop"] for f in train))
    assert len({f["orientation"] for f in train}) >= 2
    assert nt != 2 or all(f["orientation"] < 4 for f in train + test)


def _signature(puzzle):
    return tuple(_key(p[field]) for s in ("train", "test") for p in puzzle[s]
                 for field in ("input", "output"))


def self_test():
    """42 seeds; sequential, no grids/logs written, no retained full corpus."""
    signatures, references, silhouettes, crop_shapes, input_shapes = set(), set(), set(), set(), set()
    offsets, orientations, anchors, palettes, clip_states = set(), set(), set(), set(), set()
    losses, masses, densities = set(), set(), set()
    test_novel = clipped_tests = hidden_train = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = _signature(p)
        signatures.add(hashlib.sha256(repr(signature).encode()).hexdigest())
        if seed in (0, 21, 41):
            assert signature == _signature(generate_puzzle(seed))
        train_orientation = {_facts(pair)["orientation"] for pair in p["train"]}
        for split in ("train", "test"):
            for pair in p[split]:
                f = _facts(pair)
                ref, _, _ = _decode(pair["input"])
                references.add(min(_key(_d4(ref, i)) for i in range(8)))
                rr, cc = np.nonzero(ref)
                trimmed = ref[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
                silhouettes.add(min(_key(_d4(trimmed, i)) for i in range(8)))
                crop_shapes.add(f["crop"])
                input_shapes.add(f["shape"])
                offsets.add(f["offset"])
                orientations.add(f["orientation"])
                anchors.add(f["anchor"])
                palettes.add(f["palette"])
                clip_states.add(bool(f["clipped"]))
                losses.add(f["clipped"])
                masses.add(f["reference_mass"])
                densities.add(round(float(ref.mean()), 3))
                test_novel += split == "test" and f["orientation"] not in train_orientation
                clipped_tests += split == "test" and f["clipped"] > 0
                hidden_train += split == "train" and not f["anchor_visible"]
    assert len(signatures) == len(references) == 42
    assert len(silhouettes) >= 35, "reference variation must change shapes, not just placement"
    assert len(crop_shapes) >= 30 and len(input_shapes) == 6
    assert len(offsets) >= 50 and orientations == set(range(8))
    assert len(anchors) >= 7 and len(palettes) >= 20
    assert clip_states == {False, True} and len(losses) >= 15
    assert len(masses) >= 20 and len(densities) >= 20
    assert test_novel >= 30 and 0 < clipped_tests < 42 and hidden_train >= 42
    # Count boundaries, invalid types, and corrupted answers/ambiguous swatches.
    for nt, nq in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(193862666, nt, nq)
        assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    out = bad["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt answer accepted")
    blank = np.zeros((20, 20), dtype=np.int8)
    assert len(_matches(blank, np.zeros((3, 10), dtype=np.int8))) > 1
    # A color bijection may reverse every numeric ordering used in decoding.
    pair = generate_puzzle(21)["test"][0]
    permutation = np.arange(9, -1, -1, dtype=np.int8)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    return {"seeds": 42, "distinct_references": len(references), "silhouettes": len(silhouettes), "crop_shapes": len(crop_shapes),
            "input_shapes": len(input_shapes), "offsets": len(offsets), "anchors": len(anchors),
            "palettes": len(palettes), "loss_counts": len(losses), "reference_masses": len(masses),
            "novel_orientation_tests": test_novel, "clipped_tests": clipped_tests,
            "semantic_verification": "generator-owned invariants, not independent approval"}
