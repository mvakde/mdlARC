"""Recognize a reference crop, complete its 20x20 image, then contract runs.

The reference travels visibly with the query, because compressed demonstration
outputs cannot teach the reference's original run lengths. No parent is imported.
Only the standard library and NumPy are used; generation performs no file I/O.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0, 582826990, 15735970, 767452475, 25268898, 68356163,
              178519152, 209397527, 575814562, 462233965, 180501723,
              303659749, 208212264, 106980677, 491290446, 292570405,
              387621655, 169318207, 981900586, 528349633, 871797582,
              186374027, 674867593, 650414097, 9077881, 980728796,
              279688061, 461525724, 813777882, 681654318, 684222376,
              161334568, 868401104, 681449571, 105697454, 794643641,
              703310229, 671690477, 708487575, 340531561, 548532713,
              270931665)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _transform(grid, index):
    # 0..3: counterclockwise turns; 4..7: left-right reflection then turns.
    return np.rot90(np.fliplr(grid) if index >= 4 else grid, index % 4).copy()


def _starts(grid, axis):
    vectors = grid if axis == 0 else grid.T
    return np.r_[0, np.flatnonzero(np.any(vectors[1:] != vectors[:-1], axis=1)) + 1]


def _contract(grid):
    """Keep one vector per adjacent equality run, never per global color/shape."""
    # Both equivalence relations refer to COMPLETE vectors of the same image.
    return grid[np.ix_(_starts(grid, 0), _starts(grid, 1))].copy()


def _runs(grid, axis):
    return np.diff(np.r_[_starts(grid, axis), grid.shape[axis]])


def _separated_repeat(grid, axis):
    vectors = grid if axis == 0 else grid.T
    return len({_key(v.reshape(1, -1)) for v in vectors}) < len(vectors)


def _locations(full, patch):
    h, w = patch.shape
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    return np.argwhere(np.all(windows == patch, axis=(-2, -1)))


def _matches(reference, patch):
    """Return completion classes, with all equivalent offsets kept together."""
    matches = []
    for i in range(8):
        full = _transform(reference, i)
        for swap in (0, 1):
            offsets = _locations(full, patch if swap == 0 else 1 - patch)
            if len(offsets):
                matches.append((i, swap, offsets))
    return matches


def _parse(input_grid):
    """Find the large reference and small query on either side of a separator."""
    inp = _grid(input_grid)
    found = []
    for turn in range(4):
        a = np.rot90(inp, turn)
        if a.shape[1] != 20 or not 24 <= a.shape[0] <= 30:
            continue
        ref, divider, lower = a[:20], a[20], a[21:]
        marker = int(divider[0])
        if not np.all(divider == marker) or marker in ref:
            continue
        ref_colors = np.unique(ref)
        if len(ref_colors) != 2:
            continue
        rr, cc = np.nonzero(lower != marker)
        if not len(rr):
            continue
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        # No blank rows between the separator and crop, or after the crop.
        if r0 != 0 or r1 != len(lower) or not 3 <= c1 - c0 <= 20:
            continue
        crop = lower[:, c0:c1]
        colors = np.unique(crop)
        if (len(colors) != 2 or marker in crop
                or set(map(int, colors)) & set(map(int, ref_colors))):
            continue
        found.append(((ref == ref_colors[1]).astype(np.int8),
                      (crop == colors[1]).astype(np.int8), colors,
                      turn, marker, c0))
    if len(found) != 1:
        raise ValueError("input must visibly specify one reference/crop pair")
    return found[0]


def _recover(input_grid):
    ref, patch, colors, turn, marker, padding = _parse(input_grid)
    matches = _matches(ref, patch)
    # Different offsets are harmless if the entire completed image is the same.
    answers = {}
    for i, swap, offsets in matches:
        binary = _transform(ref, i) ^ swap
        answers[_key(binary)] = (binary, i, swap, offsets)
    if len(answers) != 1:
        raise ValueError("crop has no unique full 20x20 completion")
    full, i, swap, offsets = next(iter(answers.values()))
    return colors[full], (ref, patch, turn, i, swap, offsets, marker, padding)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer entirely from the supplied, visibly encoded input."""
    full, info = _recover(input_grid)
    # Undo only the rotation used to place the reference above its query.
    return np.rot90(_contract(full), -info[2]).copy()


def _local_traps(full, r, c, h, w):
    patch = full[r:r + h, c:c + w]
    rows = (np.all(patch[1:] == patch[:-1], axis=1)
            & np.any(full[r + 1:r + h] != full[r:r + h - 1], axis=1))
    cols = (np.all(patch[:, 1:] == patch[:, :-1], axis=0)
            & np.any(full[:, c + 1:c + w] != full[:, c:c + w - 1], axis=0))
    return bool(np.any(rows)), bool(np.any(cols))


def _junctions(grid):
    # Counts literal filled and unfilled intersections with four filled arms.
    arms = ((grid[:-2, 1:-1] == 1) & (grid[2:, 1:-1] == 1)
            & (grid[1:-1, :-2] == 1) & (grid[1:-1, 2:] == 1))
    return tuple(int(np.sum(arms & (grid[1:-1, 1:-1] == value))) for value in (0, 1))


def _artwork(rng):
    """Orthogonal motifs, branches, staircases and deliberately separated repeats."""
    h, w = map(int, rng.integers(9, 16, size=2))
    a = np.zeros((h, w), dtype=np.int8)
    for _ in range(int(rng.integers(5, 11))):
        r0, r1 = sorted(map(int, rng.choice(h, 2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(w, 2, replace=False)))
        kind = int(rng.integers(5))
        if kind == 0:  # loop
            a[r0, c0:c1 + 1] = a[r1, c0:c1 + 1] = 1
            a[r0:r1 + 1, c0] = a[r0:r1 + 1, c1] = 1
        elif kind == 1:  # elbow or T
            a[r0:r1 + 1, c0] = a[r1, c0:c1 + 1] = 1
            if rng.random() < .5:
                a[r0, c0:c1 + 1] = 1
        elif kind == 2:  # crossing; its center may remain a hole
            r, c = (r0 + r1) // 2, (c0 + c1) // 2
            a[r, c0:c1 + 1] = a[r0:r1 + 1, c] = 1
            a[r, c] = int(rng.integers(2))
        elif kind == 3:  # comb of uneven length
            a[r0:r1 + 1, c0] = 1
            for r in range(r0, r1 + 1, 2):
                a[r, c0:int(rng.integers(c0 + 1, c1 + 2))] = 1
        else:  # staircase, with each step joined orthogonally
            for r in range(r0, r1 + 1):
                c = c0 + (c1 - c0) * (r - r0) // max(1, r1 - r0)
                a[r, c0:c + 1] = 1
    # Copy separated whole vectors, rather than deleting all equal vectors.
    a[-2] = a[1]
    a[:, -2] = a[:, 1]
    core = _contract(a)
    if (min(core.shape) < 6 or not all(_separated_repeat(core, ax) for ax in (0, 1))
            or sum(_junctions(core)) == 0):
        return None
    if len({_key(_transform(core, i) ^ s) for i in range(8) for s in (0, 1)}) != 16:
        return None
    lengths = []
    for n in core.shape:
        weights = np.ones(n)
        mode = int(rng.integers(3))
        if mode:
            target = int(rng.choice([0, n - 1])) if mode == 1 else int(rng.integers(1, n - 1))
            weights[target] = n
        lengths.append(1 + rng.multinomial(20 - n, weights / weights.sum()))
    full = np.repeat(np.repeat(core, lengths[0], axis=0), lengths[1], axis=1)
    return full.astype(np.int8)


def _crop(rng, reference, orientation, test, trap):
    full = _transform(reference, orientation)
    # Every test has smaller area than every training crop: <=66 versus >=70.
    for _ in range(320):
        h = int(rng.integers(3, 7) if test else rng.integers(7, 10))
        w = int(rng.integers(5, 12) if test else rng.integers(10, 21))
        r, c = int(rng.integers(21 - h)), int(rng.integers(21 - w))
        patch = full[r:r + h, c:c + w]
        if len(np.unique(patch)) != 2:
            continue
        if trap is not None and not _local_traps(full, r, c, h, w)[trap]:
            continue
        # Reject local-only answers, including accidental equality in shape.
        if np.array_equal(_contract(patch), _contract(full)):
            continue
        matches = _matches(reference, patch)
        if len(matches) == 1 and matches[0][0:2] == (orientation, 0):
            return patch.copy()
    return None


def _make_input(rng, reference, patch, swap, layout_turn):
    palette = rng.permutation(10)[:5].astype(np.int8)
    ref_colors, crop_colors = np.sort(palette[:2]), np.sort(palette[2:4])
    if swap:
        crop_colors = crop_colors[::-1]
    marker = int(palette[4])
    h, w = patch.shape
    left = int(rng.integers(21 - w))
    inp = np.full((21 + h, 20), marker, dtype=np.int8)
    inp[:20] = ref_colors[reference]
    inp[21:, left:left + w] = crop_colors[patch]
    return np.rot90(inp, layout_turn).copy(), crop_colors


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # A failed reference/crop search is discarded in memory, never logged to disk.
    for _ in range(160):
        reference = _artwork(rng)
        if reference is None:
            continue
        quarter = int(rng.choice([1, 3]))
        train_orientations = [0, quarter]
        if nt >= 3:
            train_orientations.append(int(rng.integers(4, 8)))
        pool = list(range(8 if nt >= 3 else 4))
        train_orientations += list(map(int, rng.choice(pool, nt - len(train_orientations))))
        # Do not require a new geometric operation: the shift is less crop evidence.
        test_orientations = list(map(int, rng.choice(train_orientations, nq)))
        puzzle, seen = {"train": [], "test": []}, set()
        success = True
        for split, schedule in (("train", train_orientations), ("test", test_orientations)):
            for j, orientation in enumerate(schedule):
                trap = j if split == "train" and j < 2 else None
                patch = _crop(rng, reference, orientation, split == "test", trap)
                if patch is None:
                    success = False
                    break
                swap = j % 2 if split == "train" and j < 2 else int(rng.integers(2))
                layout = int(rng.integers(4))
                for _ in range(24):
                    inp, colors = _make_input(rng, reference, patch, swap, layout)
                    if _key(inp) not in seen:
                        break
                else:
                    success = False
                    break
                full = colors[_transform(reference, orientation)]
                out = np.rot90(_contract(full), layout).copy()
                puzzle[split].append({"input": inp, "output": out})
                seen.add(_key(inp))
            if not success:
                break
        if success:
            validate_generated_puzzle(puzzle)
            return puzzle
    raise RuntimeError("bounded template/crop construction exhausted")


def _analyze(pair):
    if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
        raise ValueError("pair must have exactly input and output")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    full, info = _recover(inp)
    ref, patch, turn, orientation, swap, offsets, marker, padding = info
    expected = np.rot90(_contract(full), -turn)
    assert np.array_equal(out, expected), "incorrect output"
    reduced = _contract(full)
    assert full.shape == (20, 20) and max(reduced.shape) < 20
    assert all(_separated_repeat(reduced, ax) for ax in (0, 1))
    assert np.array_equal(_contract(reduced), reduced)
    # Compare the independent full-vector definition with either sequential
    # implementation. Deleting duplicate vectors must not merge distinct ones.
    row_first = full[_starts(full, 0)]
    row_first = row_first[:, _starts(row_first, 1)]
    column_first = full[:, _starts(full, 1)]
    column_first = column_first[_starts(column_first, 0)]
    assert np.array_equal(row_first, reduced)
    assert np.array_equal(column_first, reduced)
    binary_reduced = (reduced == np.unique(reduced)[1]).astype(np.int8)
    assert len({_key(_transform(binary_reduced, i) ^ s)
                for i in range(8) for s in (0, 1)}) == 16
    # Exact expansion proves that each retained cell, including crossings,
    # is copied rather than overwritten by a horizontal/vertical priority rule.
    assert np.array_equal(np.repeat(np.repeat(reduced, _runs(full, 0), axis=0),
                                    _runs(full, 1), axis=1), full)
    colored_patch = np.unique(full)[patch]
    assert not np.array_equal(_contract(colored_patch), reduced), "A is inert"
    assert not np.array_equal(full, reduced), "B is inert"
    traps = [False, False]
    for r, c in offsets:
        row, col = _local_traps(full, int(r), int(c), *patch.shape)
        traps[0] |= row
        traps[1] |= col
    assert sum(_junctions(_contract(ref))) > 0
    return {"reference": ref, "patch_shape": patch.shape, "traps": traps,
            "orientation": orientation, "swap": swap, "turn": turn,
            "run_lengths": (tuple(map(int, _runs(full, 0))), tuple(map(int, _runs(full, 1)))),
            "output_shape": out.shape, "offsets": offsets, "junctions": _junctions(_contract(ref))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _analyze(pair)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            analyses[split].append(analysis)
    reference = analyses["train"][0]["reference"]
    assert all(np.array_equal(reference, a["reference"])
               for split in analyses.values() for a in split), "reference changed within puzzle"
    assert len({_key(_transform(reference, i) ^ s) for i in range(8) for s in (0, 1)}) == 16
    assert all(any(a["traps"][ax] for a in analyses["train"]) for ax in (0, 1))
    assert {0, 1} <= {a["swap"] for a in analyses["train"]}
    assert len({a["orientation"] for a in analyses["train"]}) >= 2
    assert {a["orientation"] for a in analyses["test"]} <= {a["orientation"] for a in analyses["train"]}
    assert max(np.prod(a["patch_shape"]) for a in analyses["test"]) < min(
        np.prod(a["patch_shape"]) for a in analyses["train"])


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Streaming 42-seed check; returns small actual-grid diversity measurements."""
    references, shapes, crops, palettes, layouts, runs, offsets, orientations = (set() for _ in range(8))
    pairs = row_traps = col_traps = filled = hollow = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed == 0:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        ref = _parse(puzzle["train"][0]["input"])[0]
        references.add(min(_key(_transform(ref, i) ^ s) for i in range(8) for s in (0, 1)))
        # Numeric color order cannot encode the role correspondence. This
        # permutation reverses some pairs and preserves others, including zero.
        relabel = np.array([8, 2, 9, 0, 6, 4, 1, 7, 5, 3], dtype=np.int8)
        anchor = puzzle["train"][0]
        assert np.array_equal(solve(relabel[anchor["input"]]), relabel[anchor["output"]])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                # Geometric covariance: panel placement is never an output code.
                assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
                shapes.add(pair["input"].shape)
                crops.add(a["patch_shape"])
                palettes.add(tuple(map(int, np.unique(pair["output"]))))
                layouts.add(a["turn"])
                runs.add(a["run_lengths"])
                offsets.update(map(tuple, a["offsets"].tolist()))
                orientations.add(a["orientation"])
                pairs += 1
                row_traps += a["traps"][0]
                col_traps += a["traps"][1]
                hollow += a["junctions"][0] > 0
                filled += a["junctions"][1] > 0
    assert len(references) == 42 and len(crops) >= 40 and len(runs) >= 80
    assert len(offsets) >= 30 and len(palettes) >= 30
    assert layouts == set(range(4)) and orientations == set(range(8))
    assert row_traps >= 42 and col_traps >= 42 and filled > 0 and hollow > 0
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(12345, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject a false answer and a query which contains no distinguishing data.
    bad = generate_puzzle(1)
    out = bad["test"][0]["output"]
    out[0, 0] = int(next(v for v in np.unique(out) if v != out[0, 0]))
    try:
        validate_generated_puzzle(bad)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    try:
        solve(np.zeros((25, 20), dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("unencoded input accepted")
    # A genuine two-color crop can still permit incompatible completions.
    ref = _parse(generate_puzzle(0)["train"][0]["input"])[0]
    ambiguous = None
    for r in range(18):
        for c in range(18):
            patch = ref[r:r + 3, c:c + 3]
            if len(np.unique(patch)) == 2 and len(_matches(ref, patch)) > 1:
                ambiguous = patch
                break
        if ambiguous is not None:
            break
    assert ambiguous is not None
    inp, _ = _make_input(np.random.default_rng(9000), ref, ambiguous, 0, 0)
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous crop accepted")
    return {"seeds": 42, "pairs": pairs, "reference_classes": len(references),
            "input_shapes": len(shapes), "crop_shapes": len(crops), "run_layouts": len(runs),
            "crop_offsets": len(offsets), "output_palettes": len(palettes),
            "row_local_equality_traps": row_traps, "column_local_equality_traps": col_traps,
            "filled_junction_pairs": filled, "hollow_junction_pairs": hollow}
