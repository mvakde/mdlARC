"""Generate control-strip puzzles with a top-relative cyclic row shift."""

import numpy as np


_SOURCE_PAIR_KEYS = frozenset({
    ("2111198/6439998/6439388/6433388/6488555/6455533",
     "111198/439998/439388/433388/488555/455533"),
    ("6888444/6998344/2998333/6911153/6441555/6441555",
     "441555/441555/888444/998344/998333/911153"),
    ("6888444/6898491/6899991/2553331/6553431/6553444",
     "553331/553431/553444/888444/898491/899991"),
    ("6588899/2544449/6531149/6531148/6133388/6113889",
     "113889/588899/544449/531149/531148/133388"),
})


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _source_key(input_grid, output_grid):
    return (
        "/".join("".join(str(int(v)) for v in row) for row in input_grid),
        "/".join("".join(str(int(v)) for v in row) for row in output_grid),
    )


def _strip_signature(column):
    height = len(column)
    if height < 4:
        return None
    values, counts = np.unique(column, return_counts=True)
    if len(values) != 2 or sorted(int(c) for c in counts) != [1, height - 1]:
        return None
    marker = int(values[np.argmin(counts)])
    rail = int(values[np.argmax(counts)])
    offset = int(np.flatnonzero(column == marker)[0])
    return rail, marker, offset


def _dimensions(rng):
    if rng.random() < 0.14:
        return 6, 6
    height_ranges = ((5, 8), (8, 13), (13, 21), (21, 30))
    width_ranges = ((2, 6), (6, 11), (11, 19), (19, 30))
    height_group = int(rng.choice(4, p=(0.28, 0.30, 0.24, 0.18)))
    width_group = int(rng.choice(4, p=(0.26, 0.35, 0.25, 0.14)))
    h0, h1 = height_ranges[height_group]
    w0, w1 = width_ranges[width_group]
    return int(rng.integers(h0, h1)), int(rng.integers(w0, w1))


def _make_payload(rng, height, width, palette):
    colors = np.asarray(palette, dtype=np.int8)
    count = len(colors)
    mode = int(rng.integers(0, 5))
    rr = np.arange(height)[:, None]
    cc = np.arange(width)[None, :]
    if mode == 0:  # Dense, irregular cells and patches.
        indices = rng.integers(0, count, size=(height, width))
        for _ in range(int(rng.integers(2, 7))):
            r = int(rng.integers(height))
            c = int(rng.integers(width))
            rh = int(rng.integers(1, min(5, height - r) + 1))
            cw = int(rng.integers(1, min(5, width - c) + 1))
            indices[r:r + rh, c:c + cw] = int(rng.integers(count))
    elif mode == 1:  # Several rectangular regions.
        indices = np.full((height, width), int(rng.integers(count)), dtype=np.int64)
        for _ in range(int(rng.integers(4, 11))):
            r = int(rng.integers(height))
            c = int(rng.integers(width))
            rh = int(rng.integers(1, height - r + 1))
            cw = int(rng.integers(1, width - c + 1))
            indices[r:r + rh, c:c + cw] = int(rng.integers(count))
    elif mode == 2:  # Repeated horizontal bands with row landmarks.
        band = int(rng.integers(2, min(6, height) + 1))
        row_codes = rng.integers(0, count, size=(height + band - 1) // band)
        indices = np.repeat(row_codes, band)[:height, None] * np.ones((1, width), dtype=np.int64)
        for _ in range(int(rng.integers(2, 7))):
            r = int(rng.integers(height))
            c = int(rng.integers(width))
            span = int(rng.integers(1, min(5, width - c) + 1))
            indices[r, c:c + span] = int(rng.integers(count))
    elif mode == 3:  # Sparse marks on a quiet field.
        indices = np.zeros((height, width), dtype=np.int64)
        marks = max(3, int(height * width * rng.uniform(0.035, 0.16)))
        for _ in range(marks):
            r = int(rng.integers(height))
            c = int(rng.integers(width))
            indices[r, c] = int(rng.integers(1, count))
        if width >= 3:
            r = int(rng.integers(height))
            c = int(rng.integers(width - 1))
            indices[r, c:c + 2] = int(rng.integers(1, count))
    else:  # Row and column interactions.
        rows = rng.integers(0, count, size=(height, 1))
        columns = rng.integers(0, count, size=(1, width))
        indices = (rows + columns + rr * cc * int(rng.integers(1, count))) % count
        for _ in range(int(rng.integers(1, 5))):
            r = int(rng.integers(height))
            indices[r, :] = int(rng.integers(count))

    grid = colors[indices].copy()
    if np.unique(grid).size < 2:
        grid[0, 0] = colors[0]
        grid[-1, -1] = colors[1]
    # No payload column may look like the one exceptional full-height strip.
    for c in range(width):
        signature = _strip_signature(grid[:, c])
        if signature is not None:
            rail, marker, marked_row = signature
            other_row = (marked_row + max(1, height // 2)) % height
            grid[other_row, c] = marker
    return grid


def _color_roles(rng, fixed_roles):
    if fixed_roles is None:
        colors = rng.permutation(10)
        rail, marker = int(colors[0]), int(colors[1])
        available = colors[2:]
    else:
        rail, marker = fixed_roles
        available = np.asarray([v for v in range(10) if v not in fixed_roles])
        available = rng.permutation(available)
    size = int(rng.choice(np.arange(2, 9), p=(0.19, 0.18, 0.16, 0.14, 0.12, 0.11, 0.10)))
    palette = tuple(int(v) for v in available[:size])
    return rail, marker, palette


def _make_pair(rng, height, width, offset, side, fixed_roles, training):
    rail, marker, palette = _color_roles(rng, fixed_roles)
    for attempt in range(48):
        payload = _make_payload(rng, height, width, palette)
        shifted = np.roll(payload, offset, axis=0)
        if any(np.array_equal(payload, np.roll(payload, k, axis=0)) for k in range(1, height)):
            continue
        if training and offset and np.array_equal(shifted, np.roll(payload, -offset, axis=0)):
            continue
        break
    else:
        payload = np.full((height, width), palette[0], dtype=np.int8)
        payload[0:2, :] = palette[1]
        shifted = np.roll(payload, offset, axis=0)
    control = np.full((height, 1), rail, dtype=np.int8)
    control[offset, 0] = marker
    input_grid = np.concatenate((control, payload), axis=1) if side == "left" else np.concatenate((payload, control), axis=1)
    return {"input": input_grid.astype(np.int8), "output": shifted.astype(np.int8)}


def _training_offset(rng, index, height):
    if index == 0:
        return 1
    if index == 1:
        return 2
    draw = rng.random()
    if draw < 0.17:
        offset = 0
    elif draw < 0.40:
        offset = height - 1
    else:
        offset = int(rng.integers(1, height))
    if height % 2 == 0 and offset == height // 2:
        offset = height - 1
    return offset


def _test_offset(rng, height, familiar, train_offsets):
    if familiar:
        return int(rng.choice([k for k in train_offsets if 0 < k < height]))
    choices = [k for k in range(1, height) if k not in train_offsets]
    if not choices:
        choices = list(range(1, height))
    if height - 1 in choices and rng.random() < 0.40:
        return height - 1
    return int(rng.choice(choices))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # The top-relative downward rule is the only learned convention. Colors
    # can be pair-local; fixed roles occur in a subset for source-like tasks.
    fixed_roles = tuple(int(v) for v in rng.permutation(10)[:2]) if rng.random() < 0.45 else None
    first_side = "left" if rng.random() < 0.5 else "right"
    train_sides_mixed = rng.random() < 0.75
    other_side = "right" if first_side == "left" else "left"
    train = []
    test = []
    seen = set()

    def add(section, height, width, offset, side):
        for _ in range(64):
            pair = _make_pair(rng, height, width, offset, side, fixed_roles, section == "train")
            key = _grid_key(pair["input"])
            if key in seen or _source_key(pair["input"], pair["output"]) in _SOURCE_PAIR_KEYS:
                continue
            seen.add(key)
            (train if section == "train" else test).append(pair)
            return
        # A bounded fallback supplies many distinct aperiodic two-color
        # payloads even if random construction repeatedly collides.
        rail, marker, palette = _color_roles(rng, fixed_roles)
        for code in range(64):
            payload = np.full((height, width), palette[0], dtype=np.int8)
            payload[:2, :] = palette[1]
            for bit in range(min(6, (height - 2) * width)):
                if code & (1 << bit):
                    row = 2 + bit // width
                    column = bit % width
                    payload[row, column] = palette[1]
            if any(_strip_signature(payload[:, c]) is not None for c in range(width)):
                continue
            if any(np.array_equal(payload, np.roll(payload, k, axis=0)) for k in range(1, height)):
                continue
            control = np.full((height, 1), rail, dtype=np.int8)
            control[offset, 0] = marker
            input_grid = np.concatenate((control, payload), axis=1) if side == "left" else np.concatenate((payload, control), axis=1)
            pair = {"input": input_grid, "output": np.roll(payload, offset, axis=0)}
            key = _grid_key(input_grid)
            if key in seen or _source_key(input_grid, pair["output"]) in _SOURCE_PAIR_KEYS:
                continue
            seen.add(key)
            (train if section == "train" else test).append(pair)
            return
        raise RuntimeError("could not produce a unique pair")

    train_offsets = []
    for i in range(train_count):
        height, width = _dimensions(rng)
        offset = _training_offset(rng, i, height)
        if not train_sides_mixed or i == 0:
            side = first_side
        elif i == 1:
            side = other_side
        else:
            side = "left" if rng.random() < 0.5 else "right"
        add("train", height, width, offset, side)
        train_offsets.append(offset)
    order = rng.permutation(train_count)
    train = [train[int(i)] for i in order]

    for i in range(test_count):
        height, width = _dimensions(rng)
        familiar = (i == 0) if test_count >= 2 else bool(rng.integers(0, 2))
        if not familiar and all(k in train_offsets for k in range(1, height)):
            height = 13
        offset = _test_offset(rng, height, familiar, train_offsets)
        side = (other_side if i == 0 and not train_sides_mixed and rng.random() < 0.75
                else ("left" if rng.random() < 0.5 else "right"))
        add("test", height, width, offset, side)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    height, input_width = input_grid.shape
    _require(height >= 5 and output_grid.shape == (height, input_width - 1), "incorrect input/output dimensions")
    signatures = [(c, _strip_signature(input_grid[:, c])) for c in range(input_width)]
    candidates = [(c, sig) for c, sig in signatures if sig is not None]
    _require(len(candidates) == 1, "control strip must be the unique exceptional column")
    column, (rail, marker, offset) = candidates[0]
    _require(column in (0, input_width - 1), "control strip must attach to an edge")
    side = "left" if column == 0 else "right"
    payload = input_grid[:, 1:] if side == "left" else input_grid[:, :-1]
    _require(not np.any(payload == rail) and not np.any(payload == marker), "control colors must be absent from payload")
    palette = frozenset(int(v) for v in np.unique(payload))
    _require(2 <= len(palette) <= 8, "payload requires 2-8 colors")
    _require(np.array_equal(output_grid, np.roll(payload, offset, axis=0)), "incorrect cyclic downward shift")
    _require(all(not np.array_equal(payload, np.roll(payload, k, axis=0)) for k in range(1, height)), "payload has ambiguous cyclic order")
    return {"height": height, "width": payload.shape[1], "offset": offset,
            "side": side, "rail": rail, "marker": marker, "palette": palette,
            "payload": payload, "output": output_grid}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "sections must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "invalid pair counts")
    infos = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D ndarray")
                _require(all(1 <= d <= 30 for d in grid.shape), "grid dimension outside 1-30")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "grid color outside 0-9")
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(_source_key(pair["input"], pair["output"]) not in _SOURCE_PAIR_KEYS, "source pair copied")
            infos[section].append(_inspect_pair(pair))
    train_offsets = {info["offset"] for info in infos["train"]}
    _require(1 in train_offsets and 2 in train_offsets, "training must show two nonzero marker distances")
    _require(any(not np.array_equal(info["output"], np.roll(info["payload"], -info["offset"], axis=0))
                 for info in infos["train"] if info["offset"]), "training does not establish downward direction")
    _require(all(info["offset"] > 0 for info in infos["test"]), "test must have visible nonzero shift")
    if test_count >= 2:
        _require(any(info["offset"] in train_offsets for info in infos["test"]), "multi-test puzzle lacks familiar shift")
        _require(any(info["offset"] not in train_offsets for info in infos["test"]), "multi-test puzzle lacks held-out shift")


def _puzzles_equal(left, right):
    return all(len(left[s]) == len(right[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(left[s], right[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, fingerprints, test_offsets, sides = set(), set(), set(), set()
    tall_narrow = wide_short = bottom_markers = sparse = dense = source_size = 0
    training_side_swaps = unseen_test_sides = role_changes = mixed_test_offsets = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_inspect_pair(p) for p in puzzle["train"]]
        test_info = [_inspect_pair(p) for p in puzzle["test"]]
        training_side_swaps += len({x["side"] for x in train_info}) > 1
        unseen_test_sides += bool({x["side"] for x in test_info} - {x["side"] for x in train_info})
        role_changes += len({(x["rail"], x["marker"]) for x in train_info + test_info}) > 1
        mixed_test_offsets += len({x["offset"] for x in test_info}) > 1
        for section, info_list in (("train", train_info), ("test", test_info)):
            for pair, info in zip(puzzle[section], info_list):
                height, width, offset = info["height"], info["width"], info["offset"]
                shapes.add((height, width))
                sides.add(info["side"])
                fingerprints.add(_grid_key(pair["input"]))
                tall_narrow += height >= 15 and width <= 5
                wide_short += height <= 7 and width >= 15
                bottom_markers += offset == height - 1
                sparse += len(info["palette"]) <= 3
                dense += len(info["palette"]) >= 5
                source_size += (height, width) == (6, 6)
                if section == "test":
                    test_offsets.add(offset)
                payload = pair["input"][:, 1:] if info["side"] == "left" else pair["input"][:, :-1]
                _require(np.array_equal(pair["output"], np.concatenate((payload[-offset:], payload[:-offset]), axis=0))
                         if offset else np.array_equal(pair["output"], payload), "self-test row movement")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack variety")
    _require(len(shapes) >= 50 and len(fingerprints) >= 150, "insufficient structural variety")
    _require(sides == {"left", "right"} and training_side_swaps >= 10, "side swaps lack coverage")
    _require(unseen_test_sides >= 3, "unseen test-side swaps lack coverage")
    _require(role_changes >= 10, "pair-local color roles lack coverage")
    _require(len(test_offsets) >= 8 and mixed_test_offsets >= 5, "test shifts lack coverage")
    _require(min(tall_narrow, wide_short, bottom_markers, sparse, dense, source_size) >= 2,
             "geometry, palette, or marker extremes lack coverage")
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + i, *counts)
        validate_generated_puzzle(puzzle)
        _require((len(puzzle["train"]), len(puzzle["test"])) == counts, "explicit counts ignored")
        _require(_puzzles_equal(puzzle, generate_puzzle(1000 + i, *counts)), "nondeterministic explicit puzzle")
    for i, kwargs in enumerate(({"num_train": None, "num_test": 1}, {"num_train": None, "num_test": 4},
                                {"num_train": 2, "num_test": None}, {"num_train": 10, "num_test": None})):
        puzzle = generate_puzzle(2000 + i, **kwargs)
        validate_generated_puzzle(puzzle)
        for name, count in kwargs.items():
            if count is not None:
                _require(len(puzzle["train" if name == "num_train" else "test"]) == count, "partial count ignored")
    for i, kwargs in enumerate(({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                                {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 3},
                                {"num_train": 3, "num_test": 4})):
        try:
            generate_puzzle(3000 + i, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_puzzles_equal(generate_puzzle(8675309), generate_puzzle(8675309)), "nondeterministic automatic puzzle")
