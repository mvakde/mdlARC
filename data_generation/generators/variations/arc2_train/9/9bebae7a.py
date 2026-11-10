import numpy as np


_DIRS = ("up", "right", "down", "left")
_DELTAS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
_CCW_SIDE = {"up": "left", "right": "up", "down": "right", "left": "down"}
_SOURCE_INPUT_SIGNATURES = frozenset((
    "0000000000/0000044000/0000440000/0000404000/0000040000/0000000000/0000000000/0666000000/0060000000/0060000000/0000000000",
    "0000000000/0000040000/0004440000/0044444000/0004400000/0000000000/0000000000/0000000000/0000600000/0066660000/0000600000/0000000000",
    "0000000000/0000000000/0040000000/0004000000/0440000000/0444000000/0000000000/0000066600/0000006000/0000006000",
    "0000000000000/0000000000000/0000000000000/0000000000000/0000040400000/0004044440000/0000444400000/0000000000000/0000000006000/0000000066660/0000000006000",
    "0004444400/0044440440/0004000000/0000006000/0000666600/0000006000",
    "000000000000/000000400000/000004440000/000000040000/000000444000/000000440000/000000000000/006000000000/006000000000/066600000000/000000000000",
))


def _validate_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _validate_count(num_train, "num_train", 2, 10)
    num_test = _validate_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _components(mask, diagonal=False):
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    remaining = set(cells)
    groups = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    while remaining:
        start = remaining.pop()
        stack = [start]
        group = {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _make_motif(rng, height, width, kind):
    area = height * width
    for _ in range(350):
        mask = np.zeros((height, width), dtype=bool)
        center = (int(rng.integers(height)), int(rng.integers(width)))
        anchors = [(0, int(rng.integers(width))),
                   (height - 1, int(rng.integers(width))),
                   (int(rng.integers(height)), 0),
                   (int(rng.integers(height)), width - 1)]
        for start in anchors:
            r, c = start
            mask[r, c] = True
            while (r, c) != center:
                vertical, horizontal = r != center[0], c != center[1]
                if kind == "diagonal" and vertical and horizontal and rng.random() < 0.84:
                    r += 1 if center[0] > r else -1
                    c += 1 if center[1] > c else -1
                elif vertical and (not horizontal or rng.random() < 0.5):
                    r += 1 if center[0] > r else -1
                else:
                    c += 1 if center[1] > c else -1
                mask[r, c] = True
        if kind == "compact":
            target = int(rng.integers(max(5, int(0.34 * area)),
                                      max(6, int(0.67 * area)) + 1))
            while int(mask.sum()) < target:
                frontier = [(r, c) for r in range(height) for c in range(width)
                            if not mask[r, c] and any(
                                0 <= r + dr < height and 0 <= c + dc < width
                                and mask[r + dr, c + dc] for dr, dc in _DELTAS.values())]
                if not frontier:
                    break
                mask[frontier[int(rng.integers(len(frontier)))]] = True
        elif kind == "diagonal":
            # Sparse side branches preserve diagonal links and internal gaps.
            target = int(rng.integers(max(5, int(0.20 * area)),
                                      max(6, int(0.40 * area)) + 1))
            while int(mask.sum()) < target:
                frontier = [(r, c) for r in range(height) for c in range(width)
                            if not mask[r, c] and any(
                                0 <= r + dr < height and 0 <= c + dc < width
                                and mask[r + dr, c + dc]
                                for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)]
                if not frontier:
                    break
                mask[frontier[int(rng.integers(len(frontier)))]] = True
        else:
            mask |= np.fliplr(mask)
        occupied = int(mask.sum())
        if (5 <= occupied < area and occupied <= int(0.72 * area)
                and all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()))
                and len(_components(mask, diagonal=True)) == 1
                and (kind != "diagonal" or len(_components(mask)) >= 2)
                and (kind != "symmetric" or np.array_equal(mask, np.fliplr(mask)))):
            return mask
    raise RuntimeError("could not construct motif")


def _marker_lengths(rng, direction):
    long_length = int(rng.integers(2, 6))
    lengths = {d: int(rng.integers(1, min(3, long_length - 1) + 1))
               for d in _DIRS}
    lengths[direction] = long_length
    if rng.random() < 0.48:
        options = [d for d in _DIRS if d != direction]
        lengths[options[int(rng.integers(len(options)))]] = 0
    return lengths


def _marker_cells(junction, lengths):
    jr, jc = junction
    cells = {(jr, jc)}
    for direction, length in lengths.items():
        dr, dc = _DELTAS[direction]
        cells.update((jr + dr * i, jc + dc * i) for i in range(1, length + 1))
    return cells


def _decode_marker(mask):
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    candidates = []
    for jr, jc in cells:
        lengths = {}
        reconstructed = {(jr, jc)}
        for direction, (dr, dc) in _DELTAS.items():
            length = 0
            while (jr + dr * (length + 1), jc + dc * (length + 1)) in cells:
                length += 1
                reconstructed.add((jr + dr * length, jc + dc * length))
            lengths[direction] = length
        active = [d for d in _DIRS if lengths[d]]
        maximum = max(lengths.values())
        longest = [d for d in _DIRS if lengths[d] == maximum]
        if (reconstructed == cells and len(active) in (3, 4)
                and maximum >= 2 and len(longest) == 1
                and all(lengths[d] < maximum for d in active if d != longest[0])):
            candidates.append(((jr, jc), longest[0], lengths))
    if len(candidates) != 1:
        raise AssertionError("marker needs one unambiguous junction and longest arm")
    return candidates[0]


def _rectangle(bbox, side):
    r0, c0, h, w = bbox
    if side == "left":
        return {(r, c) for r in range(r0, r0 + h) for c in range(c0 - w, c0)}
    if side == "right":
        return {(r, c) for r in range(r0, r0 + h) for c in range(c0 + w, c0 + 2 * w)}
    if side == "up":
        return {(r, c) for r in range(r0 - h, r0) for c in range(c0, c0 + w)}
    return {(r, c) for r in range(r0 + h, r0 + 2 * h) for c in range(c0, c0 + w)}


def _fits(bbox, side, rows, cols):
    return all(0 <= r < rows and 0 <= c < cols for r, c in _rectangle(bbox, side))


def _reflection_coordinates(payload_cells, bbox, side):
    r0, c0, h, w = bbox
    reflected = set()
    for r, c in payload_cells:
        lr, lc = r - r0, c - c0
        if side == "right":
            reflected.add((r, c0 + 2 * w - 1 - lc))
        elif side == "left":
            reflected.add((r, c0 - 1 - lc))
        elif side == "down":
            reflected.add((r0 + 2 * h - 1 - lr, c))
        else:
            reflected.add((r0 - 1 - lr, c))
    return reflected


def _grid_signature(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _dimensions(rng, size_target=None, reference_area=None):
    for _ in range(100):
        shape_mode = int(rng.integers(6))
        if shape_mode == 0:
            h, w = int(rng.integers(3, 5)), int(rng.integers(5, 8))
        elif shape_mode == 1:
            h, w = int(rng.integers(5, 8)), int(rng.integers(3, 5))
        elif shape_mode == 2:
            h, w = int(rng.integers(3, 6)), int(rng.integers(6, 10))
        elif shape_mode == 3:
            h, w = int(rng.integers(6, 10)), int(rng.integers(3, 6))
        else:
            h, w = int(rng.integers(3, 10)), int(rng.integers(3, 10))
        area = h * w
        if reference_area is None or size_target is None:
            return h, w
        if ((size_target == "smaller" and area <= 0.8 * reference_area)
                or (size_target == "larger" and area >= 1.2 * reference_area)
                or (size_target == "comparable" and 0.8 * reference_area < area < 1.2 * reference_area)):
            return h, w
    return h, w


def _generate_pair(rng, direction, colors, kind=None, marker_place=None,
                   room=None, size_target=None, reference_area=None):
    side = _CCW_SIDE[direction]
    background, payload_color, marker_color = colors
    for _ in range(700):
        h, w = _dimensions(rng, size_target, reference_area)
        motif_kind = kind if kind is not None else str(rng.choice(("compact", "diagonal", "symmetric"), p=(0.43, 0.49, 0.08)))
        try:
            motif = _make_motif(rng, h, w, motif_kind)
        except RuntimeError:
            continue
        lengths = _marker_lengths(rng, direction)
        if sum(lengths.values()) + 1 >= int(motif.sum()):
            continue
        # In an input with changing palettes, geometry must identify the marker.
        try:
            _decode_marker(motif)
        except AssertionError:
            pass
        else:
            continue
        layout = room if room is not None else str(rng.choice(("tight", "two", "roomy"), p=(0.36, 0.42, 0.22)))
        vertical = side in ("up", "down")
        min_rows = max(5, 2 * h if vertical else h + 2,
                       2 * h if layout != "tight" else 0)
        min_cols = max(5, 2 * w if not vertical else w + 2,
                       2 * w if layout != "tight" else 0)
        if layout == "roomy":
            min_rows, min_cols = max(min_rows, 3 * h), max(min_cols, 3 * w)
        if min_rows > 30 or min_cols > 30:
            continue
        slack = 2 if layout == "tight" else 7
        rows = int(rng.integers(min_rows, min(30, min_rows + slack) + 1))
        cols = int(rng.integers(min_cols, min(30, min_cols + slack) + 1))
        positions = []
        for r0 in range(rows - h + 1):
            for c0 in range(cols - w + 1):
                bbox = (r0, c0, h, w)
                fits = [d for d in _DIRS if _fits(bbox, d, rows, cols)]
                if side in fits and len(fits) >= (3 if layout == "roomy" else 2 if layout == "two" else 1):
                    positions.append((bbox, len(fits)))
        if not positions:
            continue
        bbox, fit_count = positions[int(rng.integers(len(positions)))]
        r0, c0, _, _ = bbox
        payload = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(motif)}
        original_box = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
        destination = _rectangle(bbox, side)
        reflected = _reflection_coordinates(payload, bbox, side)
        moat = {(r + dr, c + dc) for r, c in payload for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)}
        pools = {"off": [], "blank": [], "overlap": []}
        for jr in range(rows):
            for jc in range(cols):
                cells = _marker_cells((jr, jc), lengths)
                if (any(not (0 <= r < rows and 0 <= c < cols) for r, c in cells)
                        or not cells.isdisjoint(original_box)
                        or not cells.isdisjoint(moat)):
                    continue
                placement = ("off" if cells.isdisjoint(destination) else
                             "overlap" if not cells.isdisjoint(reflected) else "blank")
                pools[placement].append(cells)
        placement = marker_place if marker_place is not None else str(rng.choice(("off", "blank", "overlap"), p=(0.55, 0.20, 0.25)))
        if not pools[placement]:
            continue
        marker = pools[placement][int(rng.integers(len(pools[placement])))]
        inp = np.full((rows, cols), background, dtype=np.int8)
        out = np.full((rows, cols), background, dtype=np.int8)
        for r, c in payload:
            inp[r, c] = payload_color
            out[r, c] = payload_color
        for r, c in marker:
            inp[r, c] = marker_color
        for r, c in reflected:
            out[r, c] = payload_color
        if (np.count_nonzero(inp == background) <= inp.size // 2
                or np.count_nonzero(out == background) <= out.size // 2
                or _grid_signature(inp) in _SOURCE_INPUT_SIGNATURES):
            continue
        return inp, out
    raise RuntimeError("could not place payload and marker")


def _array_key(array):
    return (array.shape, array.dtype.str, array.tobytes())


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    input_values = {int(x) for x in np.unique(inp)}
    output_values = {int(x) for x in np.unique(out)}
    assert len(input_values) == 3 and len(output_values) == 2 and output_values < input_values
    marker_color = next(iter(input_values - output_values))
    counts = {color: int(np.count_nonzero(inp == color)) for color in input_values}
    background = max(counts, key=counts.get)
    assert counts[background] > inp.size // 2 and background in output_values
    payload_color = next(iter(output_values - {background}))
    assert np.count_nonzero(out == background) > out.size // 2
    marker_mask, payload_mask = inp == marker_color, inp == payload_color
    junction, direction, lengths = _decode_marker(marker_mask)
    side = _CCW_SIDE[direction]
    positions = np.argwhere(payload_mask)
    assert len(positions) > int(marker_mask.sum())
    r0, c0 = map(int, positions.min(axis=0))
    r1, c1 = map(int, positions.max(axis=0))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    assert 3 <= h <= 9 and 3 <= w <= 9
    crop = payload_mask[r0:r1 + 1, c0:c1 + 1]
    assert (crop[0].any() and crop[-1].any() and crop[:, 0].any() and crop[:, -1].any()
            and 5 <= int(crop.sum()) < crop.size and len(_components(crop, True)) == 1)
    try:
        _decode_marker(payload_mask)
    except AssertionError:
        pass
    else:
        raise AssertionError("payload looks like a marker")
    bbox = (r0, c0, h, w)
    payload = {tuple(map(int, p)) for p in positions}
    marker = {tuple(map(int, p)) for p in np.argwhere(marker_mask)}
    assert all(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in marker)
    assert all(max(abs(mr - pr), abs(mc - pc)) > 1 for mr, mc in marker for pr, pc in payload)
    assert _fits(bbox, side, *inp.shape)
    reflected = _reflection_coordinates(payload, bbox, side)
    expected = np.full_like(inp, background)
    for r, c in payload | reflected:
        expected[r, c] = payload_color
    assert np.array_equal(out, expected) and not np.array_equal(inp, out)
    destination = _rectangle(bbox, side)
    marker_place = ("off" if marker.isdisjoint(destination) else
                    "overlap" if not marker.isdisjoint(reflected) else "blank")
    return {"roles": (background, payload_color, marker_color),
            "direction": direction, "side": side, "bbox": bbox,
            "motif": (h, w, crop.tobytes()), "components4": len(_components(crop)),
            "marker_place": marker_place, "fit_count": sum(_fits(bbox, d, *inp.shape) for d in _DIRS),
            "short_lengths": tuple(lengths[d] for d in _DIRS if d != direction and lengths[d]),
            "arm_count": sum(bool(v) for v in lengths.values()),
            "long_length": lengths[direction], "junction": junction}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    analyses = {"train": [], "test": []}
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            inpkey = _array_key(pair["input"])
            pairkey = (inpkey, _array_key(pair["output"]))
            assert inpkey not in inputs and pairkey not in pairs
            assert _grid_signature(pair["input"]) not in _SOURCE_INPUT_SIGNATURES
            inputs.add(inpkey)
            pairs.add(pairkey)
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))
    directions = {a["direction"] for a in analyses["train"]}
    assert len(directions) >= 2
    assert analyses["train"][0]["components4"] >= 2
    assert analyses["train"][1]["components4"] == 1
    # Two distinct observed quarter-turn mappings reveal the sole direction convention.
    assert len({a["side"] for a in analyses["train"]}) >= 2
    assert all(a["marker_place"] == "off" and a["fit_count"] >= 2
               for a in analyses["train"][:2])
    # Either the palette is fixed or training visibly shows that colors can trade roles.
    roles = [a["roles"] for a in analyses["train"] + analyses["test"]]
    if len(set(a["roles"] for a in analyses["train"])) == 1:
        assert len(set(roles)) == 1


def _direction_plan(rng, num_train, num_test):
    order = list(_DIRS)
    rng.shuffle(order)
    modes = ["heldout", "shared"]
    if num_train >= 3 and num_test >= 2:
        modes.append("mixed")
    if num_train >= 4:
        modes.append("full")
    mode = str(rng.choice(modes))
    if mode == "heldout":
        train_basis = order[1:4]
        train = train_basis[:min(3, num_train)]
        test = [order[0]] * num_test
    elif mode == "full":
        train = order[:4]
        test = order[:min(num_test, 4)]
    elif mode == "mixed":
        train = order[:min(3, num_train)]
        test = [order[3], order[0]]
        while len(test) < num_test:
            test.append(order[int(rng.integers(4))])
    else:
        train = order[:min(3, num_train)]
        test = order[:min(num_test, len(train))]
        while len(test) < num_test:
            test.append(train[int(rng.integers(len(train)))])
    while len(train) < num_train:
        train.append(train[int(rng.integers(len(train)))])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _pair_colors(rng, base, palette_mode, index):
    if palette_mode == "stable" or index == 0:
        return base
    if index == 1:
        return base[1], base[2], base[0]
    if rng.random() < 0.45:
        values = rng.choice(10, size=3, replace=False)
        return tuple(map(int, values))
    values = list(base)
    rng.shuffle(values)
    return tuple(values)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    base = tuple(map(int, rng.choice(10, size=3, replace=False)))
    palette_mode = str(rng.choice(("stable", "varying")))
    train_directions, test_directions = _direction_plan(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    used = set()
    train_areas = []
    for index, direction in enumerate(train_directions + test_directions):
        split = "train" if index < num_train else "test"
        local_index = index if split == "train" else index - num_train
        colors = _pair_colors(rng, base, palette_mode, index)
        kind = "diagonal" if index == 0 else "compact" if index == 1 else None
        marker_place = "off" if index < 2 else None
        room = "two" if index < 2 else None
        if split == "test":
            size_target = ("smaller", "comparable", "larger")[(int(seed) + local_index) % 3]
            reference_area = float(np.median(train_areas))
        else:
            size_target = reference_area = None
        for _ in range(80):
            child = np.random.default_rng(int(rng.integers(0, 2**63)))
            inp, out = _generate_pair(child, direction, colors, kind, marker_place,
                                      room, size_target, reference_area)
            key = _array_key(inp)
            if key not in used:
                used.add(key)
                puzzle[split].append({"input": inp, "output": out})
                if split == "train":
                    p = np.argwhere(inp == colors[1])
                    train_areas.append(int(np.prod(p.max(axis=0) - p.min(axis=0) + 1)))
                break
        else:
            raise RuntimeError("could not generate unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(p["input"]), _array_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    a, b = generate_puzzle(81723, 7, 3), generate_puzzle(81723, 7, 3)
    assert _puzzle_key(a) == _puzzle_key(b)
    assert _puzzle_key(generate_puzzle(314159)) == _puzzle_key(generate_puzzle(314159))
    for seed, nt, ne in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (1101, None, 1), (1102, None, 4),
                         (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        assert nt is None or len(puzzle["train"]) == nt
        assert ne is None or len(puzzle["test"]) == ne
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    keys, train_counts, test_counts = set(), set(), set()
    motif_keys, layouts, shapes, marker_variants = set(), set(), set(), set()
    palettes, directions, placements, size_relations = set(), set(), set(), set()
    diagonal_count = short_varied = ambiguous_off = varied_puzzles = 0
    heldout_puzzles = shared_puzzles = full_train_puzzles = mixed_test_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        varied_puzzles += len({x["roles"] for x in train_info}) > 1
        train_dirs = {x["direction"] for x in train_info}
        test_dirs = {x["direction"] for x in test_info}
        heldout_puzzles += bool(test_dirs - train_dirs)
        shared_puzzles += bool(test_dirs & train_dirs)
        full_train_puzzles += len(train_dirs) == 4
        mixed_test_puzzles += len(test_dirs) >= 2
        median_area = float(np.median([x["bbox"][2] * x["bbox"][3] for x in train_info]))
        for info in test_info:
            ratio = info["bbox"][2] * info["bbox"][3] / median_area
            size_relations.add("smaller" if ratio <= 0.8 else "larger" if ratio >= 1.2 else "comparable")
        directions.add((tuple(sorted({x["direction"] for x in train_info})),
                        tuple(sorted({x["direction"] for x in test_info}))))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = _analyze_pair(pair["input"], pair["output"])
                diagonal_count += x["components4"] > 1
                short_varied += any(length > 1 for length in x["short_lengths"])
                ambiguous_off += x["marker_place"] == "off" and x["fit_count"] >= 2
                motif_keys.add(x["motif"])
                layouts.add((pair["input"].shape, x["bbox"], x["junction"]))
                shapes.add(pair["input"].shape)
                marker_variants.add((x["direction"], x["arm_count"], x["long_length"], x["short_lengths"]))
                palettes.add(x["roles"])
                placements.add(x["marker_place"])
    assert len(keys) >= 40 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(motif_keys) >= 120 and len(layouts) >= 180
    assert len(marker_variants) >= 20 and len(palettes) >= 25 and len(directions) >= 8
    assert diagonal_count >= 40 and short_varied >= 30 and ambiguous_off >= 50
    assert varied_puzzles >= 8 and placements == {"off", "blank", "overlap"}
    assert heldout_puzzles >= 5 and shared_puzzles >= 5
    assert full_train_puzzles >= 3 and mixed_test_puzzles >= 5
    assert size_relations == {"smaller", "comparable", "larger"}
