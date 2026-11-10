"""Solved puzzles with a reflected copy and a source-side color exchange."""

import hashlib

import numpy as np


_DIRECTIONS = (
    ("vertical", "left"),
    ("vertical", "right"),
    ("horizontal", "above"),
    ("horizontal", "below"),
)
_SOURCE_PAIR_DIGESTS = frozenset(
    (
        "2198c9a066ed493ab40b495612bcc8d90c6e99bec5a50ffe9cf9d349cb840d77",
        "fb59235a27ba122a043c4773ed2a18c25514024af644a3e7e7fa31592aec9d14",
        "d0cba1cec51d319699245c5316692f6ad4777f5bfa9eb0788ca5e3180ab30581",
        "de3dbf577b6f1fb6e13181f77251297e70a77bd1721061f2c7c327ea2c83664b",
    )
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _normalize_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _normalize_count("num_train", num_train, 2, 10)
    test = _normalize_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _choice(rng: np.random.Generator, values: list | tuple):
    return values[int(rng.integers(len(values)))]


def _sample_plan(rng: np.random.Generator, train_count: int, test_count: int) -> dict:
    direction_mode = _choice(rng, ("source", "mixed", "single", "held_out"))
    if direction_mode == "source":
        train_dirs = [
            (("vertical", "right"), ("horizontal", "above"), ("horizontal", "below"))[i % 3]
            for i in range(train_count)
        ]
        test_dirs = [("vertical", "left")] * test_count
    elif direction_mode == "single":
        orientation = _choice(rng, ("vertical", "horizontal"))
        sides = ("left", "right") if orientation == "vertical" else ("above", "below")
        first = int(rng.integers(2))
        train_dirs = [(orientation, sides[(first + i) % 2]) for i in range(train_count)]
        test_dirs = [(orientation, _choice(rng, sides)) for _ in range(test_count)]
    elif direction_mode == "held_out":
        withheld = _choice(rng, _DIRECTIONS)
        eligible = [direction for direction in _DIRECTIONS if direction != withheld]
        first_orientation = "horizontal" if withheld[0] == "vertical" else "vertical"
        first = _choice(rng, [d for d in eligible if d[0] == first_orientation])
        second = _choice(rng, [d for d in eligible if d[0] != first_orientation])
        train_dirs = [first, second] + [_choice(rng, eligible) for _ in range(train_count - 2)]
        test_dirs = [withheld] * test_count
    else:
        vertical = _choice(rng, _DIRECTIONS[:2])
        horizontal = _choice(rng, _DIRECTIONS[2:])
        train_dirs = [vertical, horizontal] + [
            _choice(rng, _DIRECTIONS) for _ in range(train_count - 2)
        ]
        rng.shuffle(train_dirs)
        test_dirs = [_choice(rng, _DIRECTIONS) for _ in range(test_count)]

    palette_mode = _choice(rng, ("stable", "overlap", "recombine", "fresh"))
    role_mode = _choice(rng, ("stable", "vary"))
    colors = [int(x) for x in rng.permutation(10)]
    fixed_roles = (colors[0], colors[1])
    pool = colors[2:] if role_mode == "stable" else colors

    if palette_mode == "stable":
        content = [tuple(pool[:2])] * (train_count + test_count)
    elif palette_mode == "overlap":
        anchor = pool[0]
        others = pool[1:]
        content = [(anchor, others[i % len(others)]) for i in range(train_count)]
        content.extend(
            (anchor, _choice(rng, others)) for _ in range(test_count)
        )
    elif palette_mode == "recombine":
        first = (pool[0], pool[1])
        second = (pool[2], pool[3])
        content = [first, second]
        for _ in range(train_count - 2):
            content.append(tuple(int(x) for x in rng.choice(pool, size=2, replace=False)))
        content.extend(
            (_choice(rng, first), _choice(rng, second)) for _ in range(test_count)
        )
    else:
        content = [
            tuple(int(x) for x in rng.choice(pool, size=2, replace=False))
            for _ in range(train_count + test_count)
        ]

    roles = []
    previous_content = None
    for i, pair in enumerate(content):
        if role_mode == "stable":
            roles.append(fixed_roles)
        else:
            available = [c for c in range(10) if c not in pair]
            if previous_content is not None and i % 3 == 1:
                reusable = [c for c in previous_content if c in available]
                background = _choice(rng, reusable) if reusable else _choice(rng, available)
            else:
                background = _choice(rng, available)
            divider = _choice(rng, [c for c in available if c != background])
            roles.append((background, divider))
        previous_content = pair

    complex_motifs = bool(rng.integers(2))
    motif_modes = []
    for i in range(train_count + test_count):
        if complex_motifs:
            modes = ("irregular", "solid", "path", "fragments", "band", "sparse")
        else:
            modes = ("irregular", "solid", "path", "band")
        motif_modes.append(_choice(rng, modes))
    motif_modes[0] = "irregular"
    if complex_motifs:
        motif_modes[1] = "fragments"

    return {
        "train_dirs": train_dirs,
        "test_dirs": test_dirs,
        "content": content,
        "roles": roles,
        "motif_modes": motif_modes,
        "complex_motifs": complex_motifs,
    }


def _connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(int(x) for x in points[0])
    seen = {start}
    stack = [start]
    while stack:
        row, col = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (row + dr, col + dc)
            if (0 <= neighbor[0] < mask.shape[0] and
                    0 <= neighbor[1] < mask.shape[1] and
                    bool(mask[neighbor]) and neighbor not in seen):
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(points)


def _sample_size(rng: np.random.Generator, orientation: str, regime: str) -> tuple[int, int]:
    form = _choice(rng, ("small", "medium", "large", "tall", "wide"))
    if form == "small":
        height, width = int(rng.integers(1, 5)), int(rng.integers(1, 6))
    elif form == "medium":
        height, width = int(rng.integers(3, 8)), int(rng.integers(3, 8))
    elif form == "large":
        height, width = int(rng.integers(6, 11)), int(rng.integers(6, 11))
    elif form == "tall":
        height, width = int(rng.integers(6, 13)), int(rng.integers(1, 4))
    else:
        height, width = int(rng.integers(1, 4)), int(rng.integers(6, 13))
    if regime == "target_test" and rng.random() < 0.32:
        height = max(height, int(rng.integers(5, 10)))
        width = max(width, int(rng.integers(5, 10)))
    if orientation == "vertical":
        width = min(width, 9)
    else:
        height = min(height, 9)
    if height * width == 1:
        width = 2
    return height, width


def _sample_mask(
    rng: np.random.Generator, height: int, width: int, mode: str
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if mode == "solid":
        mask[:] = True
    elif mode == "band":
        if height == 1 or (width > 1 and rng.random() < 0.5):
            mask[int(rng.integers(height)), :] = True
            if height > 1:
                mask[:, int(rng.integers(width))] = True
        else:
            mask[:, int(rng.integers(width))] = True
            if width > 1:
                mask[int(rng.integers(height)), :] = True
    elif mode == "path":
        row, col = int(rng.integers(height)), int(rng.integers(width))
        mask[row, col] = True
        steps = max(3, int(rng.integers(height + width, 3 * (height + width) + 1)))
        for _ in range(steps):
            neighbors = [
                (row + dr, col + dc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= row + dr < height and 0 <= col + dc < width
            ]
            if neighbors:
                row, col = _choice(rng, neighbors)
                mask[row, col] = True
    else:
        density = {
            "irregular": (0.48, 0.82),
            "fragments": (0.16, 0.42),
            "sparse": (0.20, 0.48),
        }[mode]
        mask = rng.random((height, width)) < rng.uniform(*density)
    if np.count_nonzero(mask) < 2:
        mask[0, 0] = True
        mask[-1, -1] = True
        if height * width == 1:
            raise ValueError("motif needs two cells")
    if mode == "fragments" and _connected(mask):
        mask[:] = False
        mask[0, 0] = True
        mask[-1, -1] = True
        if height == 1 and width == 2:
            mask[0, 1] = True
    rows, cols = np.where(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _inspect_input(grid: np.ndarray) -> dict:
    height, width = grid.shape
    values = [int(x) for x in np.unique(grid)]
    _require(len(values) == 4, "input must have exactly four colors")
    counts = np.bincount(grid.astype(np.int64, copy=False).ravel(), minlength=10)
    background = int(np.argmax(counts))
    _require(all(counts[background] > counts[c] for c in values if c != background),
             "background must be uniquely most frequent")
    lines = []
    for row in range(height):
        color = int(grid[row, 0])
        if color != background and np.all(grid[row, :] == color):
            lines.append(("horizontal", row, color))
    for col in range(width):
        color = int(grid[0, col])
        if color != background and np.all(grid[:, col] == color):
            lines.append(("vertical", col, color))
    _require(len(lines) == 1, "input needs one unambiguous full-length divider")
    orientation, axis, divider = lines[0]
    line_length = width if orientation == "horizontal" else height
    _require(counts[divider] == line_length, "divider color occurs outside divider")
    colors = tuple(c for c in values if c not in (background, divider))
    _require(len(colors) == 2, "motif needs exactly two colors")
    rows, cols = np.where(np.isin(grid, colors))
    _require(len(rows) >= 2, "motif needs at least two cells")
    if orientation == "vertical":
        if np.all(cols <= axis - 2):
            side = "left"
            gap = axis - int(cols.max()) - 1
        elif np.all(cols >= axis + 2):
            side = "right"
            gap = int(cols.min()) - axis - 1
        else:
            raise AssertionError("motif must lie on one side of divider with a gap")
        mirror_rows, mirror_cols = rows, 2 * axis - cols
    else:
        if np.all(rows <= axis - 2):
            side = "above"
            gap = axis - int(rows.max()) - 1
        elif np.all(rows >= axis + 2):
            side = "below"
            gap = int(rows.min()) - axis - 1
        else:
            raise AssertionError("motif must lie on one side of divider with a gap")
        mirror_rows, mirror_cols = 2 * axis - rows, cols
    _require(gap >= 1, "motif must have a background gap from divider")
    _require(np.all((0 <= mirror_rows) & (mirror_rows < height)) and
             np.all((0 <= mirror_cols) & (mirror_cols < width)),
             "reflected cells must fit")
    _require(np.all(grid[mirror_rows, mirror_cols] == background),
             "reflected cells must land on background")
    bbox = (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))
    mask = np.zeros((bbox[1] - bbox[0] + 1, bbox[3] - bbox[2] + 1), dtype=bool)
    mask[rows - bbox[0], cols - bbox[2]] = True
    return {
        "background": background, "divider": divider, "colors": colors,
        "orientation": orientation, "side": side, "axis": axis, "gap": gap,
        "rows": rows, "cols": cols, "mirror_rows": mirror_rows,
        "mirror_cols": mirror_cols, "bbox": bbox, "mask": mask,
        "connected": _connected(mask),
        "counts": tuple(int(counts[c]) for c in colors),
    }


def _derive_output(grid: np.ndarray, meta: dict | None = None) -> np.ndarray:
    if meta is None:
        meta = _inspect_input(grid)
    result = grid.copy()
    a, b = meta["colors"]
    for row, col, mr, mc in zip(meta["rows"], meta["cols"],
                                meta["mirror_rows"], meta["mirror_cols"]):
        color = int(grid[row, col])
        result[mr, mc] = color
        result[row, col] = b if color == a else a
    return result


def _generate_pair(
    rng: np.random.Generator, direction: tuple[str, str],
    colors: tuple[int, int], roles: tuple[int, int], mode: str,
    regime: str, first_train: bool,
) -> dict:
    orientation, side = direction
    for _ in range(64):
        mh, mw = _sample_size(rng, orientation, regime)
        if mode == "fragments":
            mh, mw = max(mh, 2), max(mw, 2)
        if first_train:
            if orientation == "vertical":
                mw = max(mw, 3)
            else:
                mh = max(mh, 3)
        mask = _sample_mask(rng, mh, mw, mode)
        mh, mw = mask.shape
        if first_train:
            flipped = np.fliplr(mask) if orientation == "vertical" else np.flipud(mask)
            if np.array_equal(mask, flipped):
                continue
        perp = mw if orientation == "vertical" else mh
        along = mh if orientation == "vertical" else mw
        maximum_gap = min(4, (29 - 2 * perp) // 2)
        if maximum_gap < 1:
            continue
        gap = int(rng.integers(1, maximum_gap + 1))
        base = 2 * (perp + gap) + 1
        outer_slack = int(rng.integers(0, min(30 - base, 11) + 1))
        if rng.random() < 0.12:
            outer_slack = 30 - base
        across_size = base + outer_slack
        near_margin = int(rng.integers(outer_slack + 1))
        axis = near_margin + perp + gap
        if rng.random() < 0.12:
            along_size = 30
        else:
            along_size = along + int(rng.integers(1, min(17, 31 - along)))
        along_margin = int(rng.integers(along_size - along + 1))
        if orientation == "vertical":
            height, width = along_size, across_size
            row0 = along_margin
            col0 = near_margin if side == "left" else axis + gap + 1
        else:
            height, width = across_size, along_size
            row0 = near_margin if side == "above" else axis + gap + 1
            col0 = along_margin
        background, divider = roles
        grid = np.full((height, width), background, dtype=np.int8)
        if orientation == "vertical":
            grid[:, axis] = divider
        else:
            grid[axis, :] = divider
        locations = np.argwhere(mask)
        n = len(locations)
        if rng.random() < 0.30:
            labels = np.zeros(n, dtype=np.int8)
            labels[int(rng.integers(n))] = 1
            if rng.random() < 0.5:
                labels = 1 - labels
        else:
            labels = rng.integers(0, 2, size=n, dtype=np.int8)
            labels[0], labels[-1] = 0, 1
        for (r, c), label in zip(locations, labels):
            grid[row0 + int(r), col0 + int(c)] = colors[int(label)]
        try:
            meta = _inspect_input(grid)
        except AssertionError:
            continue
        if first_train:
            observed = meta["mask"]
            flipped = np.fliplr(observed) if orientation == "vertical" else np.flipud(observed)
            if np.array_equal(observed, flipped):
                continue
        return {"input": grid, "output": _derive_output(grid, meta)}
    raise ValueError("could not construct a valid pair")


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _pair_digest(pair: dict) -> str:
    digest = hashlib.sha256()
    for name in ("input", "output"):
        grid = pair[name]
        digest.update(str(grid.shape).encode())
        digest.update(grid.astype(np.int8, copy=False).tobytes())
    return digest.hexdigest()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    plan = _sample_plan(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count, directions, start in (
        ("train", train_count, plan["train_dirs"], 0),
        ("test", test_count, plan["test_dirs"], train_count),
    ):
        for local_index in range(count):
            index = start + local_index
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(16):
                pair = _generate_pair(
                    pair_rng, directions[local_index], plan["content"][index],
                    plan["roles"][index], plan["motif_modes"][index],
                    "source_train" if split == "train" else "target_test",
                    split == "train" and local_index == 0,
                )
                key = _grid_key(pair["input"])
                if key not in seen and _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not construct distinct pairs")
    if (any(not _inspect_input(pair["input"])["connected"] for pair in puzzle["test"])
            and all(_inspect_input(pair["input"])["connected"] for pair in puzzle["train"])):
        old_key = _grid_key(puzzle["train"][1]["input"])
        seen.remove(old_key)
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        for _ in range(16):
            pair = _generate_pair(
                pair_rng, plan["train_dirs"][1], plan["content"][1],
                plan["roles"][1], "fragments", "source_train", False,
            )
            key = _grid_key(pair["input"])
            if key not in seen and _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS:
                puzzle["train"][1] = pair
                break
        else:
            raise ValueError("could not construct fragment demonstration")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    inputs, pairs = set(), set()
    train_meta, test_meta = [], []
    for split, destination in (("train", train_meta), ("test", test_meta)):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be two-dimensional NumPy arrays")
                _require(all(1 <= size <= 30 for size in grid.shape),
                         "grid dimensions out of range")
                _require(np.issubdtype(grid.dtype, np.integer) and
                         np.all((0 <= grid) & (grid <= 9)), "invalid grid cells")
            key = _grid_key(pair["input"])
            pair_key = _pair_key(pair)
            _require(key not in inputs and pair_key not in pairs, "duplicate pair")
            inputs.add(key)
            pairs.add(pair_key)
            _require(_pair_digest(pair) not in _SOURCE_PAIR_DIGESTS,
                     "exact original pair copied")
            source = pair["input"]
            output = pair["output"]
            _require(source.shape == output.shape and not np.array_equal(source, output),
                     "dimensions changed or output unchanged")
            meta = _inspect_input(source)
            _require(np.array_equal(_derive_output(source, meta), output),
                     "output violates reflected-copy/source-swap rule")
            _require(set(int(x) for x in np.unique(source)) ==
                     set(int(x) for x in np.unique(output)),
                     "output color roles changed")
            destination.append(meta)
    first = train_meta[0]
    mirror = np.fliplr(first["mask"]) if first["orientation"] == "vertical" else np.flipud(first["mask"])
    _require(not np.array_equal(first["mask"], mirror),
             "training must distinguish reflection from translation")
    if any(not m["connected"] for m in test_meta):
        _require(any(not m["connected"] for m in train_meta),
                 "fragmented tests need a fragmented demonstration")
    _require(all((m["orientation"], m["side"]) in _DIRECTIONS
                 for m in train_meta + test_meta), "illegal direction")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(_pair_key(pair) for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    for seed, train, test in ((700, 2, 1), (701, 10, 1), (702, 10, 4),
                              (703, 7, 3), (800, None, 1), (801, None, 4),
                              (802, 2, None), (803, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            _require(len(puzzle["train"]) == train, "explicit train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "explicit test count ignored")
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(42, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_puzzle_key(generate_puzzle(9001, 8, 3)) ==
             _puzzle_key(generate_puzzle(9001, 8, 3)), "generation not deterministic")
    signatures, train_counts, test_counts = set(), set(), set()
    directions, backgrounds, dimensions, motif_shapes = set(), set(), set(), set()
    singleton, disconnected, solid, source_edge, mirror_edge = 0, 0, 0, 0, 0
    black, size30, thin, sparse, small_test, large_train = 0, 0, 0, 0, 0, 0
    mixed_directions, mixed_roles, stable_roles = 0, 0, 0
    overlapping_palettes, stable_content, fresh_test = 0, 0, 0
    within_gaps, within_sizes, reused_roles = 0, 0, 0
    train_directions, test_directions, gaps = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        metadata = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, output = pair["input"], pair["output"]
                meta = _inspect_input(source)
                metadata.append(meta)
                a, b = meta["colors"]
                expected = source.copy()
                for row, col in zip(meta["rows"], meta["cols"]):
                    mr = 2 * meta["axis"] - row if meta["orientation"] == "horizontal" else row
                    mc = 2 * meta["axis"] - col if meta["orientation"] == "vertical" else col
                    expected[mr, mc] = source[row, col]
                    expected[row, col] = b if source[row, col] == a else a
                _require(np.array_equal(expected, output), "independent coordinate check failed")
                directions.add((meta["orientation"], meta["side"]))
                (train_directions if split == "train" else test_directions).add(
                    (meta["orientation"], meta["side"])
                )
                backgrounds.add(meta["background"])
                dimensions.add(source.shape)
                motif_shapes.add((meta["mask"].shape, meta["mask"].tobytes()))
                gaps.add(meta["gap"])
                singleton += min(meta["counts"]) == 1
                disconnected += not meta["connected"]
                solid += bool(np.all(meta["mask"]))
                sparse += float(np.mean(meta["mask"])) < 0.5
                thin += min(meta["mask"].shape) == 1
                source_edge += bool(np.any(
                    (meta["rows"] == 0) | (meta["rows"] == source.shape[0] - 1) |
                    (meta["cols"] == 0) | (meta["cols"] == source.shape[1] - 1)
                ))
                mirror_edge += bool(np.any(
                    (meta["mirror_rows"] == 0) |
                    (meta["mirror_rows"] == source.shape[0] - 1) |
                    (meta["mirror_cols"] == 0) |
                    (meta["mirror_cols"] == source.shape[1] - 1)
                ))
                black += meta["background"] == 0
                size30 += 30 in source.shape
                small_test += split == "test" and max(source.shape) <= 18
                large_train += split == "train" and (
                    meta["mask"].shape[0] >= 5 or meta["mask"].shape[1] >= 6
                )
        mixed_directions += len({(m["orientation"], m["side"]) for m in metadata}) > 1
        mixed_roles += len({(m["background"], m["divider"]) for m in metadata}) > 1
        stable_roles += len({(m["background"], m["divider"]) for m in metadata}) == 1
        within_gaps += len({m["gap"] for m in metadata}) > 1
        within_sizes += len({m["mask"].shape for m in metadata}) > 1
        reused_roles += any(
            metadata[i]["background"] in metadata[i - 1]["colors"] or
            metadata[i]["divider"] in metadata[i - 1]["colors"]
            for i in range(1, len(metadata))
        )
        train_sets = [set(_inspect_input(p["input"])["colors"]) for p in puzzle["train"]]
        test_sets = [set(_inspect_input(p["input"])["colors"]) for p in puzzle["test"]]
        stable_content += len({frozenset(z) for z in train_sets + test_sets}) == 1
        fresh_test += any(
            not z.issubset(set.union(*train_sets)) for z in test_sets
        )
        overlapping_palettes += any(
            len(train_sets[i] & train_sets[j]) == 1
            for i in range(len(train_sets)) for j in range(i + 1, len(train_sets))
        )
    _require(len(signatures) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3,
             "insufficient whole-puzzle/count diversity")
    _require(len(directions) == 4 and len(backgrounds) == 10,
             "directions or background colors missing")
    _require(len(dimensions) >= 35 and len(motif_shapes) >= 80,
             "insufficient geometry diversity")
    _require(len(train_directions) == 4 and len(test_directions) == 4 and
             len(gaps) >= 3, "direction or spacing coverage is too narrow")
    _require(min(singleton, disconnected, solid, source_edge, mirror_edge,
                 black, size30, thin, sparse, small_test, large_train) > 0,
             "a requested shape, edge, or color case is missing")
    _require(min(mixed_directions, mixed_roles, stable_roles, overlapping_palettes,
                 stable_content, fresh_test, within_gaps, within_sizes,
                 reused_roles) > 0, "within-puzzle or palette variation missing")
