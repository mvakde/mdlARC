import numpy as np


_SOURCE_INPUTS = {
    ("000000000", "000000000", "000040000", "000000000", "000000000", "008000000", "088000000", "008800000", "000000000"),
    ("000004000", "008000000", "880000000", "880000000", "000004000", "000000000", "000400000", "000000000", "000000000"),
    ("088000000", "880040000", "080000000", "000000400", "000000000", "040000000", "000000000", "000000000", "000004000"),
    ("000000000", "000004000", "000000000", "088800000", "008000000", "000000400", "000000000", "000400000", "000000000"),
    ("000000000", "008800000", "088800400", "008000000", "000000000", "000000400", "000000000", "000000000", "000000000"),
    ("000400000", "400000000", "000000000", "000008000", "040088800", "000008000", "000000000", "000000000", "000040000"),
    ("000008800", "000008080", "000008800", "000000000", "400000000", "000000000", "000400000", "000000000", "000000000"),
}
_SIDES = ("top", "bottom", "left", "right")


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + (int(seed) % 4)
        minimum = max(2, test + 1)
        train = minimum + ((int(seed) // 4) % (11 - minimum))
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + (int(seed) % (11 - minimum))
    else:
        test = 1 + (int(seed) % min(4, train - 1))
    return train, test


def _symmetry(mask):
    return bool(np.array_equal(mask, np.fliplr(mask)) or
                np.array_equal(mask, np.flipud(mask)) or
                np.array_equal(mask, np.rot90(mask, 2)))


def _catalog():
    result = {side: [] for side in (None,) + _SIDES}
    for bits in range(1, 512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        rows, cols = np.where(mask)
        box = (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))
        if box == (0, 2, 0, 2):
            result[None].append(mask)
        elif box == (1, 2, 0, 2):
            result["top"].append(mask)
        elif box == (0, 1, 0, 2):
            result["bottom"].append(mask)
        elif box == (0, 2, 1, 2):
            result["left"].append(mask)
        elif box == (0, 2, 0, 1):
            result["right"].append(mask)
    return result


_PATTERNS = _catalog()


def _sample_pattern(rng, side, regime, exclude=None, nonfull=False):
    candidates = _PATTERNS[side]
    if nonfull:
        candidates = [p for p in candidates if int(p.sum()) < 9]
    if exclude is not None:
        different = [p for p in candidates if tuple(p.flat) != exclude]
        if different:
            candidates = different
    densities = sorted({int(p.sum()) for p in candidates})
    if regime == "target_test" and side is None and rng.random() < 0.3:
        dense = [d for d in densities if d >= 6]
        if dense:
            densities = dense
    density = densities[int(rng.integers(len(densities)))]
    bucket = [p for p in candidates if int(p.sum()) == density]
    want_symmetry = rng.random() < 0.38
    matching = [p for p in bucket if _symmetry(p) == want_symmetry]
    if matching:
        bucket = matching
    return bucket[int(rng.integers(len(bucket)))].copy()


def _choose_dimensions(rng, count):
    bands = ((5, 7), (8, 13), (14, 20), (21, 30))
    height_band = int(rng.choice(4, p=(0.18, 0.39, 0.26, 0.17)))
    width_band = (height_band if rng.random() < 0.40
                  else int(rng.choice(4, p=(0.18, 0.39, 0.26, 0.17))))
    height = int(rng.integers(bands[height_band][0], bands[height_band][1] + 1))
    width = int(rng.integers(bands[width_band][0], bands[width_band][1] + 1))
    if count >= 7 and height * width < 65:
        if rng.random() < 0.5:
            height = int(rng.integers(10, 19))
        else:
            width = int(rng.integers(10, 19))
    return height, width


def _choose_origin(rng, height, width, regime):
    row = int(rng.integers(height - 2))
    column = int(rng.integers(width - 2))
    if rng.random() < 0.27:
        if regime == "source_train":
            column = int(rng.integers(max(1, (width - 2) // 2)))
        else:
            column = int(rng.integers((width - 3) // 2, width - 2))
    return row, column


def _marker_positions(rng, height, width, origin, count):
    row0, col0 = origin
    eligible = [(r, c) for r in range(height) for c in range(width)
                if not (row0 - 1 <= r <= row0 + 3 and col0 - 1 <= c <= col0 + 3)]
    mode = int(rng.integers(7))
    for attempt in range(3):
        pool = eligible.copy()
        rng.shuffle(pool)
        if attempt == 0 and mode < 4:
            if mode == 0:
                pool.sort(key=lambda rc: (rc[0] > row0 - 2, rng.random()))
            elif mode == 1:
                pool.sort(key=lambda rc: (rc[0] < row0 + 4, rng.random()))
            elif mode == 2:
                pool.sort(key=lambda rc: (rc[1] > col0 - 2, rng.random()))
            else:
                pool.sort(key=lambda rc: (rc[1] < col0 + 4, rng.random()))
        elif attempt == 0 and mode == 4:
            line = int(rng.integers(height))
            pool.sort(key=lambda rc: (rc[0] != line, rng.random()))
        elif attempt == 0 and mode == 5:
            line = int(rng.integers(width))
            pool.sort(key=lambda rc: (rc[1] != line, rng.random()))
        chosen = []
        for cell in pool:
            if all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return tuple(sorted(chosen))
    return None


def _source_copy(grid):
    return grid.shape == (9, 9) and tuple("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUTS


def _make_pair(rng, regime, colors, mask, count):
    background, payload, marker = colors
    tile = np.where(mask, payload, background).astype(np.int8)
    for attempt in range(180):
        if attempt < 100:
            height, width = _choose_dimensions(rng, count)
        else:
            height = int(rng.integers(12, 31))
            width = int(rng.integers(12, 31))
        origin = _choose_origin(rng, height, width, regime)
        positions = _marker_positions(rng, height, width, origin, count)
        if positions is None:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        r, c = origin
        grid[r:r + 3, c:c + 3] = tile
        for mr, mc in positions:
            grid[mr, mc] = marker
        if np.count_nonzero(grid == background) <= max(int(mask.sum()), count):
            continue
        if _source_copy(grid):
            continue
        return {"input": grid, "output": np.tile(tile, (1, count))}
    raise RuntimeError("Could not place tile and markers")


def _train_counts(rng, amount):
    counts = [1, 4]
    if amount >= 3:
        counts.append(3)
    while len(counts) < amount:
        counts.append(int(rng.integers(1, 11)))
    return counts


def _test_count(rng):
    if rng.random() < 0.64:
        return int(rng.integers(1, 5))
    return int(rng.integers(5, 11))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_amount, test_amount = _resolve_counts(seed, num_train, num_test)
    marker = int(rng.integers(10))
    other_colors = [c for c in range(10) if c != marker]
    first_four = rng.choice(other_colors, size=4, replace=False)
    padding = None if rng.random() < 0.43 else _SIDES[int(rng.integers(4))]
    train = []
    test = []
    used_inputs = set()
    masks = []
    counts = _train_counts(rng, train_amount)
    for index in range(train_amount + test_amount):
        regime = "source_train" if index < train_amount else "target_test"
        local = index if regime == "source_train" else index - train_amount
        if index < 2:
            background, payload = (int(first_four[2 * index]), int(first_four[2 * index + 1]))
        else:
            background, payload = (int(v) for v in rng.choice(other_colors, size=2, replace=False))
        if regime == "source_train":
            count = counts[local]
            side = padding if padding is not None and local == 0 else (
                padding if padding is not None and local > 2 and rng.random() < 0.22 else None)
            if local == 2:
                mask = masks[0].copy()
            else:
                previous = tuple(masks[0].flat) if local == 1 else None
                mask = _sample_pattern(rng, side, regime, previous, nonfull=(local == 1))
            masks.append(mask)
        else:
            count = _test_count(rng)
            side = padding if padding is not None and (local == 0 or rng.random() < 0.22) else None
            if side is not None and rng.random() < 0.23:
                mask = masks[0].copy()
            elif side is None and rng.random() < 0.16:
                tight = [p for p in masks if tuple(np.where(p)[0]) and
                         np.where(p)[0].min() == 0 and np.where(p)[0].max() == 2 and
                         np.where(p)[1].min() == 0 and np.where(p)[1].max() == 2]
                mask = tight[int(rng.integers(len(tight)))].copy() if tight else _sample_pattern(rng, None, regime)
            else:
                mask = _sample_pattern(rng, side, regime)
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        for retry in range(5):
            pair = _make_pair(pair_rng, regime, (background, payload, marker), mask, count)
            key = _array_key(pair["input"])
            if key not in used_inputs:
                used_inputs.add(key)
                break
        else:
            raise RuntimeError("Repeated input construction")
        (train if regime == "source_train" else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _array_key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(p["input"]), _array_key(p["output"])) for p in puzzle[part])
                 for part in ("train", "test"))


def _roles_from_train(train):
    candidates = None
    for pair in train:
        absent = set(np.unique(pair["input"]).tolist()) - set(np.unique(pair["output"]).tolist())
        candidates = absent if candidates is None else candidates & absent
    assert candidates is not None and len(candidates) == 1
    return next(iter(candidates))


def _pair_geometry(pair, marker):
    grid, output = pair["input"], pair["output"]
    colors = set(np.unique(grid).tolist())
    assert marker in colors and len(colors) == 3
    remaining = colors - {marker}
    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    background = max(remaining, key=lambda color: counts[color])
    payload = next(iter(remaining - {background}))
    assert counts[background] > max(counts[payload], counts[marker])
    assert set(np.unique(output).tolist()).issubset({background, payload})
    points = np.argwhere(grid == payload)
    r0, r1 = int(points[:, 0].min()), int(points[:, 0].max())
    c0, c1 = int(points[:, 1].min()), int(points[:, 1].max())
    assert 1 <= r1 - r0 <= 2 and 1 <= c1 - c0 <= 2
    assert (r1 - r0, c1 - c0) in {(2, 2), (1, 2), (2, 1)}
    return background, payload, (r0, r1, c0, c1)


def _origin_for_box(box, side):
    r0, r1, c0, c1 = box
    if (r1 - r0, c1 - c0) == (2, 2):
        return r0, c0
    assert side is not None
    if (r1 - r0, c1 - c0) == (1, 2):
        assert side in ("top", "bottom")
        return (r0 - 1 if side == "top" else r0), c0
    assert side in ("left", "right")
    return r0, (c0 - 1 if side == "left" else c0)


def _analyze_pair(pair, marker, side):
    grid, output = pair["input"], pair["output"]
    background, payload, box = _pair_geometry(pair, marker)
    row, col = _origin_for_box(box, side)
    assert 0 <= row <= grid.shape[0] - 3 and 0 <= col <= grid.shape[1] - 3
    tile = grid[row:row + 3, col:col + 3]
    assert set(np.unique(tile).tolist()).issubset({background, payload})
    markers = [(int(r), int(c)) for r, c in np.argwhere(grid == marker)]
    assert 1 <= len(markers) <= 10
    for i, (r, c) in enumerate(markers):
        assert not (row - 1 <= r <= row + 3 and col - 1 <= c <= col + 3)
        for rr, cc in markers[i + 1:]:
            assert max(abs(r - rr), abs(c - cc)) > 1
    reconstructed = np.full(grid.shape, background, dtype=grid.dtype)
    reconstructed[row:row + 3, col:col + 3] = tile
    for r, c in markers:
        reconstructed[r, c] = marker
    assert np.array_equal(reconstructed, grid)
    assert output.shape == (3, 3 * len(markers))
    assert np.array_equal(output, np.tile(tile, (1, len(markers))))
    parities = {(r % 2, c % 2) for r, c in markers}
    return {"tile": tile.copy(), "mask": tile == payload, "count": len(markers),
            "shape": grid.shape, "origin": (row, col), "markers": markers,
            "parities": len(parities), "background": background, "payload": payload,
            "margin": box != (row, row + 2, col, col + 2)}


def _infer_padding(train, marker):
    inferred = None
    for pair in train:
        _, _, box = _pair_geometry(pair, marker)
        extent = (box[1] - box[0], box[3] - box[2])
        if extent == (2, 2):
            continue
        options = ("top", "bottom") if extent == (1, 2) else ("left", "right")
        valid = []
        for side in options:
            try:
                _analyze_pair(pair, marker, side)
                valid.append(side)
            except AssertionError:
                pass
        assert len(valid) == 1
        if inferred is None:
            inferred = valid[0]
        else:
            assert inferred == valid[0]
    return inferred


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert all(1 <= d <= 30 for d in grid.shape)
            assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
        input_key = _array_key(pair["input"])
        pair_key = (input_key, _array_key(pair["output"]))
        assert input_key not in inputs and pair_key not in pairs
        assert not _source_copy(pair["input"])
        inputs.add(input_key)
        pairs.add(pair_key)
    marker = _roles_from_train(puzzle["train"])
    side = _infer_padding(puzzle["train"], marker)
    train_metadata = [_analyze_pair(p, marker, side) for p in puzzle["train"]]
    test_metadata = [_analyze_pair(p, marker, side) for p in puzzle["test"]]
    assert all(marker not in {m["background"], m["payload"]} for m in train_metadata + test_metadata)
    assert len({m["count"] for m in train_metadata}) >= 2
    assert any(int(m["mask"].sum()) < 9 for m in train_metadata)
    assert side is not None or not any(m["margin"] for m in test_metadata)
    assert len({(m["shape"], m["origin"], tuple(m["markers"])) for m in train_metadata + test_metadata}) == len(train_metadata) + len(test_metadata)


def _component_count(mask):
    remaining = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    result = 0
    while remaining:
        result += 1
        stack = [remaining.pop()]
        while stack:
            row, col = stack.pop()
            neighbors = {(row + dr, col + dc) for dr in (-1, 0, 1)
                         for dc in (-1, 0, 1) if dr or dc}
            found = neighbors & remaining
            remaining -= found
            stack.extend(found)
    return result


def self_test() -> None:
    source_train_four = np.array([[int(c) for c in row] for row in
        ("000000000", "000004000", "000000000", "088800000", "008000000",
         "000000400", "000000000", "000400000", "000000000")], dtype=np.int8)
    source_tile = np.array([[int(c) for c in row] for row in
        ("000", "888", "080")], dtype=np.int8)
    source_pair = {"input": source_train_four, "output": np.tile(source_tile, (1, 3))}
    assert _analyze_pair(source_pair, 4, "top")["margin"]
    assert all(any(np.array_equal(pattern, wanted) for pattern in _PATTERNS[side])
               for side, wanted in ((None, np.array([[0, 8, 0], [8, 8, 8], [0, 8, 0]], dtype=bool)),
                                    (None, np.array([[0, 8, 8], [8, 8, 8], [0, 8, 0]], dtype=bool))))
    seen = {"train_counts": set(), "test_counts": set(), "shapes": set(),
            "masks": set(), "palettes": set(), "counts": set(), "sides": set(),
            "parities": set(), "puzzles": set()}
    padded = 0
    filled = 0
    wide = 0
    tall = 0
    repeat_count_tile = 0
    varying_palettes = 0
    same_count_different_tile = 0
    mixed_train_density = 0
    mixed_train_symmetry = 0
    reverse_shift = 0
    little_shift = 0
    category = {"train_density": set(), "test_density": set(),
                "train_symmetry": set(), "test_symmetry": set(),
                "train_position": set(), "test_position": set(),
                "train_count": set(), "test_count": set(),
                "marker_sides": set()}
    special = {"aligned": 0, "disconnected": 0, "internal_blank": 0,
               "at_boundary": 0, "several_marker_sides": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        seen["puzzles"].add(_puzzle_key(puzzle))
        seen["train_counts"].add(len(puzzle["train"]))
        seen["test_counts"].add(len(puzzle["test"]))
        marker = _roles_from_train(puzzle["train"])
        side = _infer_padding(puzzle["train"], marker)
        seen["sides"].add(side)
        metas = [_analyze_pair(p, marker, side) for p in puzzle["train"] + puzzle["test"]]
        train_metas = metas[:len(puzzle["train"])]
        test_metas = metas[len(puzzle["train"]):]
        mixed_train_density += any(int(m["mask"].sum()) <= 5 for m in train_metas) and any(int(m["mask"].sum()) >= 6 for m in train_metas)
        mixed_train_symmetry += any(_symmetry(m["mask"]) for m in train_metas) and any(not _symmetry(m["mask"]) for m in train_metas)
        train_left = sum(m["origin"][1] < m["shape"][1] / 2 for m in train_metas) / len(train_metas)
        test_left = sum(m["origin"][1] < m["shape"][1] / 2 for m in test_metas) / len(test_metas)
        reverse_shift += train_left < .5 and test_left > .5
        little_shift += abs(train_left - test_left) < .26
        palettes = {(m["background"], m["payload"]) for m in metas}
        varying_palettes += len(palettes) > 1
        paired = [(tuple(m["mask"].flat), m["count"]) for m in metas]
        repeat_count_tile += any(a[0] == b[0] and a[1] != b[1]
                                 for i, a in enumerate(paired) for b in paired[i + 1:])
        same_count_different_tile += any(a[0] != b[0] and a[1] == b[1]
                                         for i, a in enumerate(paired) for b in paired[i + 1:])
        for index, m in enumerate(metas):
            split = "train" if index < len(train_metas) else "test"
            category[split + "_density"].add(int(m["mask"].sum()))
            category[split + "_symmetry"].add(_symmetry(m["mask"]))
            category[split + "_count"].add(m["count"])
            column = m["origin"][1]
            width = m["shape"][1]
            category[split + "_position"].add(
                "left" if column < (width - 3) / 3 else
                "right" if column > 2 * (width - 3) / 3 else "middle")
            row = m["origin"][0]
            height = m["shape"][0]
            if row == 0 or column == 0 or row == height - 3 or column == width - 3:
                special["at_boundary"] += 1
            if not m["mask"][1].any() and m["mask"][0].any() and m["mask"][2].any():
                special["internal_blank"] += 1
            if _component_count(m["mask"]) > 1:
                special["disconnected"] += 1
            marker_rows = {r for r, c in m["markers"]}
            marker_columns = {c for r, c in m["markers"]}
            if m["count"] >= 2 and (len(marker_rows) == 1 or len(marker_columns) == 1):
                special["aligned"] += 1
            sides = set()
            for mr, mc in m["markers"]:
                if mr < row - 1: sides.add("above")
                if mr > row + 3: sides.add("below")
                if mc < column - 1: sides.add("left")
                if mc > column + 3: sides.add("right")
            category["marker_sides"].update(sides)
            special["several_marker_sides"] += len(sides) >= 2
            seen["shapes"].add(m["shape"])
            seen["masks"].add(tuple(m["mask"].flat))
            seen["palettes"].add((m["background"], m["payload"], marker))
            seen["counts"].add(m["count"])
            seen["parities"].add(m["parities"])
            padded += m["margin"]
            filled += int(m["mask"].sum()) == 9
            tall += m["shape"][0] >= 2 * m["shape"][1]
            wide += m["shape"][1] >= 2 * m["shape"][0]
        assert _puzzle_key(generate_puzzle(seed)) == _puzzle_key(puzzle)
    assert len(seen["puzzles"]) == 42
    assert len(seen["train_counts"]) >= 4 and len(seen["test_counts"]) >= 3
    assert len(seen["shapes"]) >= 45 and len(seen["masks"]) >= 60
    assert len(seen["palettes"]) >= 35 and seen["counts"] == set(range(1, 11))
    assert len(seen["sides"]) == 5 and max(seen["parities"]) >= 3
    assert padded >= 10 and filled >= 3 and tall >= 3 and wide >= 3
    assert repeat_count_tile >= 15 and varying_palettes == 42
    assert same_count_different_tile >= 20
    assert mixed_train_density >= 20 and mixed_train_symmetry >= 20
    assert reverse_shift >= 1 and little_shift >= 5
    assert category["train_density"] == set(range(2, 10))
    assert category["test_density"] == set(range(2, 10))
    assert category["train_count"] == set(range(1, 11))
    assert category["test_count"] == set(range(1, 11))
    assert category["train_symmetry"] == {False, True}
    assert category["test_symmetry"] == {False, True}
    assert category["train_position"] == {"left", "middle", "right"}
    assert category["test_position"] == {"left", "middle", "right"}
    assert category["marker_sides"] == {"above", "below", "left", "right"}
    assert all(value >= 10 for value in special.values())
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        result = generate_puzzle(1000 + i, train, test)
        assert (len(result["train"]), len(result["test"])) == (train, test)
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        result = generate_puzzle(2000 + i, train, test)
        assert train is None or len(result["train"]) == train
        assert test is None or len(result["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 3.0},
                   {"num_test": True}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted illegal counts: {kwargs}")
