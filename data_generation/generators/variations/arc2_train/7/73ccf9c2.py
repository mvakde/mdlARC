"""Vertical mirror-symmetry outlier scenes."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    (21, 21, "95dee9c6d7a2c0ae08dd0006142b31aff5eff38c5848bf9f76276a493dcf62b9"),
    (23, 23, "bf8c291aaaef376ccc5a06c2f1f5ae8770b04330bd2d632115583faca4591970"),
    (20, 20, "2e7615e6384a7e17052ffb2176618a3793d7389db1e272c1406156ab50a89eb5"),
    (22, 22, "125587599390adef778adafa08c6a25ae616935f3ff1d46ed9de8620db469888"),
}


def _checked_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, seed, num_train, num_test):
    train = None if num_train is None else _checked_count(num_train, "num_train", 2, 10)
    test = None if num_test is None else _checked_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        train = 2 + seed % 9
        test = int(rng.integers(1, min(4, train - 1) + 1))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _connected(mask, diagonal=True):
    points = np.argwhere(mask)
    if not len(points):
        return False
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    start = tuple(int(v) for v in points[0])
    seen[start] = True
    stack = [start]
    while stack:
        row, col = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr == 0 and dc == 0) or (not diagonal and abs(dr) + abs(dc) != 1):
                    continue
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
    return int(seen.sum()) == int(mask.sum())


def _tight(mask):
    return mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()


def _multi_run(mask):
    return any(np.count_nonzero(np.diff(np.pad(row.astype(np.int8), (1, 1))) == 1) > 1 for row in mask)


def _symmetric_mask(rng, height, width, sparse):
    """Grow mirrored arms from a spine; arms may join diagonally."""
    center = width // 2
    if not sparse and rng.random() < 0.38:
        mask = np.zeros((height, width), dtype=bool)
        extent_max = center if width % 2 else center - 1
        extents = rng.integers(0, extent_max + 1, size=height)
        extents[int(rng.integers(height))] = extent_max
        if np.all(extents == extent_max):
            extents[int(rng.integers(height))] = 0
        for row, extent in enumerate(extents):
            extent = int(extent)
            mask[row, center - extent - (width % 2 == 0):center + extent + 1] = True
        return mask
    for _ in range(90):
        mask = np.zeros((height, width), dtype=bool)
        if width % 2:
            mask[:, center] = True
            inner = center - 1
        else:
            mask[:, center - 1:center + 1] = True
            inner = center - 2
        for col in range(inner, -1, -1):
            candidates = [r for r in range(height)
                          if mask[max(0, r - 1):min(height, r + 2), col + 1].any()]
            row = int(candidates[int(rng.integers(len(candidates)))])
            mask[row, col] = mask[row, width - 1 - col] = True
        density = float(rng.uniform(0.33, 0.55) if sparse else rng.uniform(0.58, 0.83))
        for _ in range(height * width * 2):
            if mask.mean() >= density:
                break
            row = int(rng.integers(height))
            col = int(rng.integers((width + 1) // 2))
            if mask[row, col]:
                continue
            if mask[max(0, row - 1):min(height, row + 2),
                    max(0, col - 1):min(width, col + 2)].any():
                mask[row, col] = mask[row, width - 1 - col] = True
        if (not mask.all() and _tight(mask) and _connected(mask)
                and (not sparse or width < 5 or _multi_run(mask))):
            return mask
    raise RuntimeError("could not grow a symmetric figure")


def _asymmetric_mask(rng, height, width, near):
    """Edit a mirrored figure, retaining its box and eight-connectivity."""
    for attempt in range(100):
        mask = _symmetric_mask(rng, height, width, sparse=attempt % 3 != 0)
        edits = 1 if near else int(rng.integers(3, 7))
        cells = [(r, c) for r in range(height) for c in range(width)
                 if c != width - 1 - c]
        for _ in range(edits):
            rng.shuffle(cells)
            for r, c in cells:
                trial = mask.copy()
                trial[r, c] = not trial[r, c]
                if not trial.all() and _tight(trial) and _connected(trial):
                    mask = trial
                    break
        mismatch = int(np.count_nonzero(mask != np.fliplr(mask)))
        if (mismatch == 2 if near else mismatch >= 4):
            return mask
    raise RuntimeError("could not grow an asymmetric figure")


def _palette(rng, style, previous):
    if style == "constant":
        return previous[0]
    if rng.random() < 0.23:
        background, foreground = previous[int(rng.integers(len(previous)))]
        return foreground, background
    if rng.random() < 0.25:
        return previous[int(rng.integers(len(previous)))]
    background = int(rng.integers(10))
    foreground = int(rng.integers(9))
    if foreground >= background:
        foreground += 1
    return background, foreground


def _layout(rng, figure_count, fixed):
    if fixed:
        return 2, 2
    options = {3: ((2, 2), (1, 3), (3, 1), (2, 3)),
               4: ((2, 2), (2, 3), (3, 2)),
               5: ((2, 3), (3, 2)), 6: ((2, 3), (3, 2))}[figure_count]
    return options[int(rng.integers(len(options)))]


def _canvas_shape(rng, rows, cols, fixed):
    min_height = max(24 if fixed else 17, rows * 7 + 2)
    min_width = max(24 if fixed else 17, cols * 7 + 2)
    if rng.random() < 0.36:
        side = int(rng.integers(max(min_height, min_width), 31))
        return side, side
    return int(rng.integers(min_height, 31)), int(rng.integers(min_width, 31))


def _slots(rng, height, width, rows, cols):
    row_edges = [0] + [int(round(height * k / rows)) + int(rng.integers(-1, 2))
                       for k in range(1, rows)] + [height]
    col_edges = [0] + [int(round(width * k / cols)) + int(rng.integers(-1, 2))
                       for k in range(1, cols)] + [width]
    result = []
    for row in range(rows):
        for col in range(cols):
            top = row_edges[row] + (1 if row else 0)
            bottom = row_edges[row + 1] - (1 if row < rows - 1 else 0)
            left = col_edges[col] + (1 if col else 0)
            right = col_edges[col + 1] - (1 if col < cols - 1 else 0)
            result.append((top, bottom, left, right))
    return result


def _dimensions(rng, slot, regime, index, portrait):
    top, bottom, left, right = slot
    hmax, wmax = min(9, bottom - top), min(9, right - left)
    if regime == "source_train" and index == 0:
        h = int(rng.integers(4, min(7, hmax) + 1))
        w = int(rng.integers(4, min(7, wmax) + 1))
    elif regime == "source_train" and index == 1:
        h = int(rng.integers(4, min(8, hmax) + 1))
        w = int(rng.integers(3, min(8, wmax) + 1))
    elif portrait and hmax >= 7:
        h = int(rng.integers(7, hmax + 1))
        w = int(rng.integers(3, min(6, wmax, h - 1) + 1))
    else:
        h = int(rng.integers(3, hmax + 1))
        w = int(rng.integers(3, wmax + 1))
    return h, w


def _position(rng, slot, dims, canvas_shape, edge_bias):
    top, bottom, left, right = slot
    h, w = dims
    if edge_bias and top == 0 and rng.random() < 0.5:
        row = 0
    elif edge_bias and bottom == canvas_shape[0] and rng.random() < 0.5:
        row = bottom - h
    else:
        row = int(rng.integers(top, bottom - h + 1))
    if edge_bias and left == 0 and rng.random() < 0.5:
        col = 0
    elif edge_bias and right == canvas_shape[1] and rng.random() < 0.5:
        col = right - w
    else:
        col = int(rng.integers(left, right - w + 1))
    return row, col


def _make_pair(rng, regime, index, palette, opposite_phase, near_phase):
    fixed = regime == "source_train" and index < 2
    figure_count = 4 if fixed else int(rng.integers(3, 7))
    rows, cols = _layout(rng, figure_count, fixed)
    height, width = _canvas_shape(rng, rows, cols, fixed)
    slots = _slots(rng, height, width, rows, cols)
    if fixed:
        target_slot = opposite_phase if index == 0 else 3 - opposite_phase
        selected = list(range(4))
    else:
        selected = sorted(int(x) for x in rng.choice(len(slots), size=figure_count, replace=False))
        target_slot = selected[int(rng.integers(len(selected)))]
    background, foreground = palette
    canvas = np.full((height, width), background, dtype=np.int8)
    portrait = regime == "target_test" and rng.random() < 0.38
    target_dims = _dimensions(rng, slots[target_slot], regime, index, portrait)
    near = ((regime == "source_train" and index == near_phase) or
            (regime == "target_test" and index == 0))
    if regime == "source_train" and index > 1:
        near = rng.random() < 0.45
    elif regime == "target_test" and index > 0:
        near = rng.random() < 0.65
    target_mask = _asymmetric_mask(rng, *target_dims, near=near)
    target_box = None
    control_index = 0
    for slot_index in [target_slot] + [s for s in selected if s != target_slot]:
        slot = slots[slot_index]
        target = slot_index == target_slot
        if target:
            dims, mask = target_dims, target_mask
        else:
            hmax, wmax = min(9, slot[1] - slot[0]), min(9, slot[3] - slot[2])
            if fixed and index == 0 and control_index == 0:
                dims = (min(hmax, target_dims[0] + 1), min(wmax, target_dims[1] + 1))
            else:
                dims = (int(rng.integers(3, hmax + 1)), int(rng.integers(3, wmax + 1)))
            sparse = (fixed and control_index < 2) or rng.random() < (0.68 if regime == "target_test" else 0.57)
            mask = _symmetric_mask(rng, *dims, sparse=sparse)
            control_index += 1
        row, col = _position(rng, slot, dims, (height, width), rng.random() < 0.38)
        canvas[row:row + dims[0], col:col + dims[1]][mask] = foreground
        if target:
            target_box = (row, row + dims[0], col, col + dims[1])
    r0, r1, c0, c1 = target_box
    return {"input": canvas, "output": canvas[r0:r1, c0:c1].copy()}


def _grid_key(grid):
    return (grid.shape, grid.astype(np.uint8).tobytes())


def _pair_digest(pair):
    return hashlib.sha256(pair["input"].astype(np.uint8).tobytes() +
                          pair["output"].astype(np.uint8).tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    style = "constant" if rng.random() < 0.25 else "varied"
    first_background = int(rng.integers(10))
    first_foreground = int(rng.integers(9))
    if first_foreground >= first_background:
        first_foreground += 1
    first_palette = (first_background, first_foreground)
    palettes = [first_palette]
    opposite_phase = int(rng.integers(4))
    near_phase = int(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count, split in (("source_train", train_count, "train"),
                                 ("target_test", test_count, "test")):
        for index in range(count):
            palette = first_palette if not seen else _palette(rng, style, palettes)
            palettes.append(palette)
            for _ in range(15):
                pair_rng = np.random.default_rng(int(rng.integers(0, 1 << 64, dtype=np.uint64)))
                pair = _make_pair(pair_rng, regime, index, palette, opposite_phase, near_phase)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make a unique scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    height, width = mask.shape
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        seen[row, col] = True
        stack = [(row, col)]
        points = []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
        result.append(points)
    return result


def _analyze_pair(pair):
    canvas, output = pair["input"], pair["output"]
    colors, counts = np.unique(canvas, return_counts=True)
    assert len(colors) == 2 and counts[0] != counts[1], "binary scene with clear background required"
    background = int(colors[int(np.argmax(counts))])
    foreground = int(colors[int(np.argmin(counts))])
    assert set(int(x) for x in np.unique(output)) == {background, foreground}
    components = _components(canvas == foreground)
    assert 3 <= len(components) <= 6
    records = []
    for points in components:
        rows, cols = zip(*points)
        r0, r1, c0, c1 = min(rows), max(rows) + 1, min(cols), max(cols) + 1
        mask = canvas[r0:r1, c0:c1] == foreground
        assert int(mask.sum()) == len(points), "bounding boxes must not enclose other figures"
        assert _tight(mask) and _connected(mask)
        records.append({"box": (r0, r1, c0, c1), "mask": mask,
                        "symmetric": np.array_equal(mask, np.fliplr(mask))})
    for i, first in enumerate(records):
        a, b, c, d = first["box"]
        for other in records[i + 1:]:
            e, f, g, h = other["box"]
            assert b <= e or f <= a or d <= g or h <= c, "bounding boxes overlap"
    outliers = [record for record in records if not record["symmetric"]]
    assert len(outliers) == 1
    r0, r1, c0, c1 = outliers[0]["box"]
    assert np.array_equal(output, canvas[r0:r1, c0:c1]), "incorrect unchanged crop"
    return {"background": background, "foreground": foreground,
            "records": records, "target": outliers[0]}


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
        seen_inputs, seen_pairs = set(), set()
        analyses = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in pair.values():
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert all(1 <= size <= 30 for size in grid.shape)
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert np.all((0 <= grid) & (grid <= 9))
                input_key = _grid_key(pair["input"])
                pair_key = (input_key, _grid_key(pair["output"]))
                assert input_key not in seen_inputs and pair_key not in seen_pairs
                seen_inputs.add(input_key)
                seen_pairs.add(pair_key)
                assert (pair["input"].shape + (_pair_digest(pair),)) not in _SOURCE_DIGESTS
                assert not np.array_equal(pair["input"], pair["output"])
                analyses[split].append(_analyze_pair(pair))
        # Opposite target positions and both mismatch severities are visible in training.
        centers = []
        for pair, analysis in zip(puzzle["train"][:2], analyses["train"][:2]):
            assert len(analysis["records"]) == 4
            r0, r1, c0, c1 = analysis["target"]["box"]
            centers.append(((r0 + r1) / (2 * pair["input"].shape[0]),
                            (c0 + c1) / (2 * pair["input"].shape[1])))
        assert (centers[0][0] < 0.5) != (centers[1][0] < 0.5)
        assert (centers[0][1] < 0.5) != (centers[1][1] < 0.5)
        mismatches = [int(np.count_nonzero(a["target"]["mask"] !=
                                           np.fliplr(a["target"]["mask"])))
                      for a in analyses["train"][:2]]
        assert 2 in mismatches and max(mismatches) >= 4
        assert int(np.count_nonzero(analyses["test"][0]["target"]["mask"] !=
                                    np.fliplr(analyses["test"][0]["target"]["mask"]))) == 2
        first_target = analyses["train"][0]["target"]["mask"].shape
        assert any(record["symmetric"] and
                   record["mask"].shape[0] >= first_target[0] and
                   record["mask"].shape[1] >= first_target[1]
                   for record in analyses["train"][0]["records"])
        assert any(_multi_run(record["mask"]) for analysis in analyses["train"][:2]
                   for record in analysis["records"] if record["symmetric"])
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError(f"invalid generated puzzle: {error}") from error


def _puzzle_key(puzzle):
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None), (6, 3)):
        puzzle = generate_puzzle(160 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(19, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a = generate_puzzle(991, 7, 2)
    assert _puzzle_key(a) == _puzzle_key(generate_puzzle(991, 7, 2))
    train_counts, test_counts, shapes, output_shapes = set(), set(), set(), set()
    backgrounds, palettes, counts, square, edges = set(), set(), set(), 0, 0
    multi_run, disconnected4, all_odd, all_even, near, portrait = 0, 0, 0, 0, 0, 0
    location, signatures = set(), set()
    changing_backgrounds = constant_palettes = swapped_roles = reused_palettes = 0
    varying_counts = row_convex_pairs = combined_cases = 0
    train_shapes, test_shapes = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_key(puzzle))
        puzzle_backgrounds, puzzle_palettes, puzzle_counts = set(), [], set()
        split_palettes = {"train": set(), "test": set()}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_pair(pair)
                puzzle_backgrounds.add(analysis["background"])
                palette = (analysis["background"], analysis["foreground"])
                puzzle_palettes.append(palette)
                split_palettes[split].add(palette)
                puzzle_counts.add(len(analysis["records"]))
                (train_shapes if split == "train" else test_shapes).add(pair["output"].shape)
                shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                backgrounds.add(analysis["background"])
                palettes.add((analysis["background"], analysis["foreground"]))
                counts.add(len(analysis["records"]))
                square += pair["input"].shape[0] == pair["input"].shape[1]
                edges += any(r["box"][0] == 0 or r["box"][1] == pair["input"].shape[0]
                             or r["box"][2] == 0 or r["box"][3] == pair["input"].shape[1]
                             for r in analysis["records"])
                multi_run += any(_multi_run(r["mask"]) for r in analysis["records"])
                row_convex_pairs += not any(_multi_run(r["mask"]) for r in analysis["records"])
                disconnected4 += any(not _connected(r["mask"], diagonal=False)
                                     for r in analysis["records"])
                parities = {r["mask"].shape[1] % 2 for r in analysis["records"]
                            if r["symmetric"]}
                all_odd += parities == {1}
                all_even += parities == {0}
                mismatch = np.count_nonzero(analysis["target"]["mask"] !=
                                            np.fliplr(analysis["target"]["mask"]))
                near += mismatch == 2
                portrait += pair["output"].shape[0] > pair["output"].shape[1]
                r0, r1, c0, c1 = analysis["target"]["box"]
                location.add(((r0 + r1) < pair["input"].shape[0],
                              (c0 + c1) < pair["input"].shape[1]))
                combined_cases += (len(analysis["records"]) != 4 and
                                   pair["input"].shape[0] == pair["input"].shape[1] and
                                   any(_multi_run(r["mask"]) for r in analysis["records"]))
        changing_backgrounds += len(puzzle_backgrounds) > 1
        constant_palettes += len(set(puzzle_palettes)) == 1
        swapped_roles += any((fg, bg) in puzzle_palettes for bg, fg in puzzle_palettes)
        reused_palettes += bool(split_palettes["train"] & split_palettes["test"])
        varying_counts += len(puzzle_counts) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 25 and len(output_shapes) >= 15
    assert len(backgrounds) >= 6 and len(palettes) >= 20 and counts == {3, 4, 5, 6}
    assert square >= 20 and edges >= 20 and multi_run >= 50 and disconnected4 >= 15
    assert all_odd >= 3 and all_even >= 3 and near >= 42 and portrait >= 25
    assert len(location) == 4
    assert row_convex_pairs >= 2 and combined_cases >= 10
    assert len(train_shapes & test_shapes) >= 10
    assert changing_backgrounds >= 10 and constant_palettes >= 3
    assert swapped_roles >= 5 and reused_palettes >= 5 and varying_counts >= 20
