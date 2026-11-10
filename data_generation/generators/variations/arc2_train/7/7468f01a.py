"""Single filled-panel ARC tasks: crop the panel and reverse its columns."""

import numpy as np


# Exact source pairs, so source dimensions and palettes remain available to sample.
_SOURCE_PAIRS = {
    (
        ("0000000000000000", "0004444400000000", "0004444400000000",
         "0004114400000000", "0004411400000000", "0004414400000000",
         "0000000000000000", "0000000000000000", "0000000000000000",
         "0000000000000000", "0000000000000000", "0000000000000000"),
        ("44444", "44444", "44114", "41144", "44144"),
    ),
    (
        ("000000000000000", "000000000000000", "000000000000000",
         "000000000000000", "000000000000000", "008888882800000",
         "008228888800000", "008822888800000", "008888888800000",
         "000000000000000"),
        ("82888888", "88888228", "88882288", "88888888"),
    ),
    (
        ("00000000000000000", "00000000000000000", "00000000000000000",
         "00000000000000000", "00000000000000000", "00000000000000000",
         "00000000000000000", "00666366000000000", "00633366000000000",
         "00666636000000000", "00666636000000000", "00000000000000000",
         "00000000000000000", "00000000000000000", "00000000000000000"),
        ("663666", "663336", "636666", "636666"),
    ),
    (
        ("00000000000000000", "00000000000000000", "00000000000000000",
         "00011118111000000", "00011111881000000", "00011111111000000",
         "00088811111000000", "00011111111000000", "00011888111000000",
         "00011111111000000", "00000000000000000", "00000000000000000",
         "00000000000000000", "00000000000000000", "00000000000000000",
         "00000000000000000", "00000000000000000"),
        ("11181111", "18811111", "11111111", "11111888", "11111111",
         "11188811", "11111111"),
    ),
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + (5 * seed + 1) % (10 - test)
    elif test is None:
        test = 1 + (7 * seed + 3) % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _panel_shape(rng, regime):
    # The same geometry families occur on both sides of the split. A mild
    # test bias toward larger general panels reproduces, but does not force,
    # the size shift in the original task.
    aspect = int(rng.integers(0, 5))
    if aspect == 0:
        height = int(rng.integers(9, 28))
        width = int(rng.integers(4, min(13, height - 1)))
    elif aspect == 1:
        width = int(rng.integers(9, 28))
        height = int(rng.integers(4, min(13, width - 1)))
    else:
        high = 24 if regime == "target_test" and rng.random() < 0.4 else 21
        height = int(rng.integers(4, high))
        width = int(rng.integers(4, high))
    return height, width


def _grow_cluster(rng, height, width, size):
    start = (int(rng.integers(height)), int(rng.integers(width)))
    cells = {start}
    mode = int(rng.integers(4))
    while len(cells) < size:
        frontier = set()
        for row, col in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (row + dr, col + dc)
                if (0 <= neighbor[0] < height and 0 <= neighbor[1] < width
                        and neighbor not in cells):
                    frontier.add(neighbor)
        if not frontier:
            break
        options = sorted(frontier)
        weights = np.ones(len(options), dtype=float)
        if mode == 0:  # long bar with occasional bends
            weights *= [5 if row == start[0] else 1 for row, col in options]
        elif mode == 1:
            weights *= [5 if col == start[1] else 1 for row, col in options]
        elif mode == 2:  # compact block, often with protrusions
            weights *= [1 / (1 + abs(row - start[0]) + abs(col - start[1]))
                        for row, col in options]
        weights /= weights.sum()
        cells.add(options[int(rng.choice(len(options), p=weights))])
    return cells


def _motif_mask(rng, height, width, symmetric=False):
    area = height * width
    for _ in range(80):
        separate = not symmetric and rng.random() < 0.52
        fraction = float(rng.uniform(0.05, 0.24 if separate else 0.39))
        target = max(2, min(int(round(area * fraction)), (area - 1) // 2))
        if separate and target >= 5:
            clusters = int(rng.integers(2, min(7, target // 2 + 1)))
            sizes = 2 + rng.multinomial(target - 2 * clusters,
                                        np.ones(clusters) / clusters)
        else:
            clusters = int(rng.integers(1, min(7, max(2, target // 2 + 1))))
            weights = rng.dirichlet(np.ones(clusters) * 0.9)
            sizes = 1 + rng.multinomial(max(0, target - clusters), weights)
        mask = np.zeros((height, width), dtype=bool)
        for size in sizes:
            for _ in range(80):
                cells = _grow_cluster(rng, height, width, int(size))
                if not separate:
                    break
                # Orthogonal separation leaves diagonal contacts possible.
                if all(not mask[row, col] and
                       all(not (0 <= row + dr < height and 0 <= col + dc < width
                                and mask[row + dr, col + dc])
                           for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                       for row, col in cells):
                    break
            else:
                break
            for row, col in cells:
                mask[row, col] = True
        else:
            if symmetric:
                mask |= mask[:, ::-1]
            if 2 <= int(mask.sum()) < area * 0.45:
                if symmetric or _reflection_distinct(mask):
                    return mask
    raise RuntimeError("could not make a suitable motif")


def _reflection_distinct(panel):
    mirror = panel[:, ::-1]
    return all(not np.array_equal(mirror, other) for other in
               (panel, panel[::-1, :], panel[::-1, ::-1]))


def _palette(rng, history):
    colors = [int(x) for x in rng.choice(10, size=4, replace=False)]
    background, fill, motif, extra = colors
    if history:
        previous = history[int(rng.integers(len(history)))]
        roll = rng.random()
        if roll < 0.22:
            fill, motif = previous[2], previous[1]
        elif roll < 0.42:
            fill = previous[int(rng.integers(1, 3))]
        elif roll < 0.58:
            motif = previous[int(rng.integers(1, 3))]
        # Local colors must still be distinct. Background is free to change.
        if fill == background:
            fill = int(rng.choice([x for x in range(10) if x != background]))
        if motif in (background, fill):
            motif = int(rng.choice([x for x in range(10) if x not in (background, fill)]))
        available = [x for x in range(10) if x not in (background, fill, motif)]
        extra = int(rng.choice(available))
    return background, fill, motif, extra


def _placement(rng, height, width):
    margins = []
    for length in (height, width):
        maximum = 30 - length
        cap = maximum if rng.random() < 0.18 else min(maximum, 13)
        margins.append(int(rng.integers(1, cap + 1)))
    vertical, horizontal = margins
    mode = float(rng.random())
    if mode < 0.53:
        vertical = max(2, vertical)
        horizontal = max(2, horizontal)
        top = int(rng.integers(1, vertical))
        left = int(rng.integers(1, horizontal))
    elif mode < 0.84:
        if rng.random() < 0.5:
            top = int(rng.choice((0, vertical)))
            horizontal = max(2, horizontal)
            left = int(rng.integers(1, horizontal))
        else:
            left = int(rng.choice((0, horizontal)))
            vertical = max(2, vertical)
            top = int(rng.integers(1, vertical))
    else:
        top = int(rng.choice((0, vertical)))
        left = int(rng.choice((0, horizontal)))
    return height + vertical, width + horizontal, top, left


def _make_pair(rng, regime, palette, three_colors, symmetric):
    height, width = _panel_shape(rng, regime)
    background, fill, motif, extra = palette
    mask = _motif_mask(rng, height, width, symmetric)
    panel = np.full((height, width), fill, dtype=np.int8)
    panel[mask] = motif
    if three_colors:
        cells = np.argwhere(mask)
        chosen_count = max(1, min(len(cells) - 1, int(round(len(cells) * rng.uniform(0.18, 0.48)))))
        for index in rng.choice(len(cells), size=chosen_count, replace=False):
            row, col = cells[int(index)]
            panel[row, col] = extra
    if not symmetric and not _reflection_distinct(panel):
        raise RuntimeError("motif did not show horizontal reflection")
    canvas_height, canvas_width, top, left = _placement(rng, height, width)
    input_grid = np.full((canvas_height, canvas_width), background, dtype=np.int8)
    input_grid[top:top + height, left:left + width] = panel
    return {"input": input_grid, "output": panel[:, ::-1].copy()}


def _signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_signature(pair):
    return tuple(tuple("".join(str(int(x)) for x in row) for row in pair[name])
                 for name in ("input", "output"))


def _crop(input_grid):
    # At most two adjacent canvas edges may be touched. Three or four
    # corners therefore identify the uniform exterior, even at an edge.
    corners = [int(input_grid[r, c]) for r in (0, -1) for c in (0, -1)]
    background = max(set(corners), key=corners.count)
    rows, cols = np.nonzero(input_grid != background)
    if len(rows) == 0:
        raise AssertionError("no panel")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    return background, (top, bottom, left, right), input_grid[top:bottom, left:right]


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask)))
    sizes = []
    while pending:
        stack = [pending.pop()]
        size = 0
        while stack:
            row, col = stack.pop()
            size += 1
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def _rule_candidates(train):
    candidates = {"crop", "left_right", "up_down", "half_turn",
                  "transpose", "anti_transpose", "quarter_cw", "quarter_ccw"}
    for pair in train:
        _, _, panel = _crop(pair["input"])
        variants = {"crop": panel, "left_right": panel[:, ::-1],
                    "up_down": panel[::-1, :], "half_turn": panel[::-1, ::-1],
                    "transpose": panel.T, "anti_transpose": panel[::-1, ::-1].T,
                    "quarter_cw": np.rot90(panel, -1),
                    "quarter_ccw": np.rot90(panel, 1)}
        candidates = {name for name in candidates
                      if np.array_equal(variants[name], pair["output"])}
    return candidates


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    seen = set()
    train_shapes = set()
    train_palettes = set()
    train_three = False
    test_three = False
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(1 <= min(grid.shape) and max(grid.shape) <= 30,
                         "grid dimension outside 1..30")
                _require(np.issubdtype(grid.dtype, np.integer)
                         and int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid values must be ARC colors")
            signature = _signature(pair["input"])
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            _require(_source_signature(pair) not in _SOURCE_PAIRS, "source pair copied")
            background, (top, bottom, left, right), panel = _crop(pair["input"])
            corners = [int(pair["input"][r, c]) for r in (0, -1) for c in (0, -1)]
            _require(corners.count(background) >= 3, "exterior corners are ambiguous")
            _require(bottom - top < pair["input"].shape[0]
                     and right - left < pair["input"].shape[1],
                     "panel must have exterior background on both axes")
            _require(np.all(panel != background), "panel has background holes")
            _require(np.array_equal(pair["output"], panel[:, ::-1]),
                     "output is not the reflected tight crop")
            values, counts = np.unique(panel, return_counts=True)
            _require(len(values) in (2, 3), "panel must have two or three colors")
            fill_index = int(np.argmax(counts))
            _require(int(counts[fill_index]) > int(counts.sum() - counts[fill_index]),
                     "fill must dominate all motif colors together")
            if split == "train":
                _require(_reflection_distinct(panel), "training reflection ambiguous")
                train_shapes.add(panel.shape)
                train_palettes.add((background, *[int(x) for x in values]))
                train_three |= len(values) == 3
            else:
                test_three |= len(values) == 3
    _require(len(train_shapes) >= 2, "training panel shapes must vary")
    _require(len(train_palettes) >= 2, "training colors must vary")
    _require(not test_three or train_three, "third test color lacks training evidence")
    _require(_rule_candidates(train) == {"left_right"},
             "training does not uniquely establish left-right reflection")
    # Prediction uses only the demonstrations and each test input.
    for pair in test:
        _, _, panel = _crop(pair["input"])
        prediction = panel[:, ::-1]
        _require(np.array_equal(prediction, pair["output"]),
                 "test output cannot be reconstructed")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # The only task-wide convention is the fixed left-right operation.
    # A three-color training example is scheduled before any such test.
    allow_three = bool(rng.random() < 0.48)
    puzzle = {"train": [], "test": []}
    history = []
    seen = set()
    for split, regime, count in (("train", "source_train", train_count),
                                 ("test", "target_test", test_count)):
        for index in range(count):
            for _ in range(120):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                palette = _palette(pair_rng, history)
                if split == "train" and index == 1 and history:
                    if palette[0] == history[0][0]:
                        continue
                three = allow_three and (split == "train" and index == 0
                                         or pair_rng.random() < 0.32)
                symmetric = split == "test" and not three and pair_rng.random() < 0.09
                try:
                    pair = _make_pair(pair_rng, regime, palette, three, symmetric)
                except RuntimeError:
                    continue
                signature = _signature(pair["input"])
                if signature in seen or _source_signature(pair) in _SOURCE_PAIRS:
                    continue
                if split == "train" and index == 1:
                    _, _, first_panel = _crop(puzzle["train"][0]["input"])
                    _, _, panel = _crop(pair["input"])
                    if panel.shape == first_panel.shape:
                        continue
                puzzle[split].append(pair)
                history.append(palette)
                seen.add(signature)
                break
            else:
                raise RuntimeError("could not construct a distinct valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    def equal(a, b):
        return all(np.array_equal(x[key], y[key])
                   for split in ("train", "test")
                   for x, y in zip(a[split], b[split])
                   for key in ("input", "output"))

    _require(equal(generate_puzzle(12345), generate_puzzle(12345)),
             "automatic generation is not deterministic")
    _require(equal(generate_puzzle(54321, 7, 3), generate_puzzle(54321, 7, 3)),
             "explicit generation is not deterministic")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(6000 + index, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts ignored")
    for seed, train, test in ((7000, None, 1), (7001, None, 4),
                              (7002, 2, None), (7003, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            _require(len(puzzle["train"]) == train, "partial train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "partial test count ignored")
    for train, test in ((1, None), (11, None), (None, 0),
                        (None, 5), (2, 2), (2, 4)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    observed = {name: set() for name in ("canvas", "panel", "palette", "layout",
                                         "train_count", "test_count", "cluster_size")}
    portrait = edge = corner = three = symmetric_test = 0
    larger = similar = smaller = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        observed["train_count"].add(len(puzzle["train"]))
        observed["test_count"].add(len(puzzle["test"]))
        train_areas = []
        test_areas = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                background, (top, bottom, left, right), panel = _crop(pair["input"])
                observed["canvas"].add(pair["input"].shape)
                observed["panel"].add(panel.shape)
                observed["palette"].add(tuple(int(x) for x in np.unique(pair["input"])))
                observed["layout"].add((pair["input"].shape, panel.shape, top, left))
                colors, counts = np.unique(panel, return_counts=True)
                fill = int(colors[int(np.argmax(counts))])
                observed["cluster_size"].update(_components(panel != fill))
                portrait += panel.shape[0] > panel.shape[1]
                edges = sum((top == 0, left == 0, bottom == pair["input"].shape[0],
                             right == pair["input"].shape[1]))
                edge += edges >= 1
                corner += edges == 2
                three += len(colors) == 3
                if split == "test":
                    symmetric_test += np.array_equal(panel, panel[:, ::-1])
                    test_areas.append(panel.size)
                else:
                    train_areas.append(panel.size)
        comparison = np.mean(test_areas) / np.mean(train_areas)
        larger += comparison > 1.2
        similar += 0.8 <= comparison <= 1.2
        smaller += comparison < 0.8
    _require(len(observed["train_count"]) >= 5 and len(observed["test_count"]) == 4,
             "automatic pair counts lack variety")
    _require(len(observed["canvas"]) >= 80 and len(observed["panel"]) >= 80
             and len(observed["layout"]) >= 200, "geometry lacks variety")
    _require(max(observed["cluster_size"]) >= 10 and portrait >= 30
             and edge >= 30 and corner >= 10 and three >= 20,
             "review variation axes lack coverage")
    _require(larger > 0 and similar > 0 and smaller > 0 and symmetric_test > 0,
             "test regimes lack variety")
