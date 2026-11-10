import numpy as np


_ORIGINAL_INPUTS = {
    ("00000000", "00000000", "00222000", "00222000", "00000000", "03333300", "00030000", "00030000"),
    ("007700", "000700", "077770", "000000", "004400", "000400"),
    ("000000", "080800", "088800", "000000", "099900", "090900"),
    ("0000000000", "0055000000", "0555000000", "0000000000", "0000000000", "0000000000", "0000000000", "0000000000", "0777000000", "0777000000", "0777000000"),
    ("000", "222", "000", "080", "080", "080"),
    ("00000000", "04400000", "04400000", "00000000", "00000000", "06666666", "06666666", "06666666"),
}


def _connected(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(int(v) for v in points[0])
    seen, pending = {start}, [start]
    height, width = mask.shape
    while pending:
        row, col = pending.pop()
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                pending.append((nr, nc))
    return len(seen) == len(points)


def _bbox(mask):
    rows, cols = np.where(mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _shape(rng, kind, height=None, width=None):
    if kind == "line":
        length = int(rng.integers(3, 11))
        height, width = (1, length) if rng.random() < 0.5 else (length, 1)
    else:
        if height is None:
            height = int(rng.integers(3 if kind in ("T", "U", "ring", "branch") else 2, 9))
        if width is None:
            width = int(rng.integers(3 if kind in ("T", "U", "ring", "branch") else 2, 10))
    mask = np.zeros((height, width), dtype=bool)
    if kind in ("solid", "line"):
        mask[:] = True
    elif kind == "L":
        mask[-1, :] = True
        mask[:, 0] = True
    elif kind == "T":
        mask[0, :] = True
        mask[:, int(rng.integers(1, width - 1))] = True
    elif kind == "U":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif kind == "ring":
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    elif kind == "branch":
        moves = [0] * (height - 1) + [1] * (width - 1)
        rng.shuffle(moves)
        row = col = 0
        mask[row, col] = True
        for move in moves:
            if move == 0:
                row += 1
            else:
                col += 1
            mask[row, col] = True
        low = max(int(mask.sum()), height * width // 4)
        high = max(low, height * width * 3 // 5)
        target = int(rng.integers(low, high + 1))
        while int(mask.sum()) < target:
            neighbors = []
            for r, c in np.argwhere(mask):
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        neighbors.append((nr, nc))
            nr, nc = neighbors[int(rng.integers(len(neighbors)))]
            mask[nr, nc] = True
    elif kind == "notched":
        mask[:] = True
        wanted = int(rng.integers(1, max(2, height * width // 3) + 1))
        removed = 0
        for flat in rng.permutation(height * width):
            r, c = divmod(int(flat), width)
            mask[r, c] = False
            if _connected(mask) and _bbox(mask) == (0, height - 1, 0, width - 1):
                removed += 1
                if removed >= wanted:
                    break
            else:
                mask[r, c] = True
        assert removed > 0
    else:
        raise ValueError("unknown shape kind")
    if kind in ("T", "L", "U", "ring"):
        mask = np.rot90(mask, int(rng.integers(0, 4)))
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
    assert _connected(mask)
    assert _bbox(mask) == (0, mask.shape[0] - 1, 0, mask.shape[1] - 1)
    return mask


def _random_shape(rng, regime):
    kinds = ("solid", "line", "T", "L", "U", "ring", "branch", "notched")
    weights = ((0.22, 0.09, 0.15, 0.13, 0.13, 0.08, 0.14, 0.06)
               if regime == "source_train" else
               (0.32, 0.17, 0.11, 0.10, 0.10, 0.07, 0.10, 0.03))
    return _shape(rng, str(rng.choice(kinds, p=weights)))


def _analyze_input(grid):
    colors = [int(c) for c in np.unique(grid)]
    assert len(colors) == 3
    candidates = []
    for background in colors:
        objects = []
        for color in colors:
            if color == background:
                continue
            mask = grid == color
            if not _connected(mask):
                break
            top, bottom, left, right = _bbox(mask)
            objects.append({"color": color, "top": top, "bottom": bottom,
                            "left": left, "right": right, "pixels": int(mask.sum())})
        if len(objects) != 2:
            continue
        objects.sort(key=lambda obj: obj["top"])
        upper, lower = objects
        gap = lower["top"] - upper["bottom"] - 1
        if gap >= 1 and np.all(grid[upper["bottom"] + 1:lower["top"]] == background):
            candidates.append((background, upper, lower, gap))
    assert len(candidates) == 1
    return candidates[0]


def _lower_crop(grid):
    _, _, lower, _ = _analyze_input(grid)
    return grid[lower["top"]:lower["bottom"] + 1,
                lower["left"]:lower["right"] + 1].copy()


def _compose(rng, upper, lower, roles, gap, bottom_padding):
    background, upper_color, lower_color = roles
    assert len({background, upper_color, lower_color}) == 3
    available = 30 - upper.shape[0] - lower.shape[0] - gap - bottom_padding
    assert available >= 0
    top_padding = int(rng.integers(0, min(4, available) + 1))
    height = top_padding + upper.shape[0] + gap + lower.shape[0] + bottom_padding
    width = max(upper.shape[1], lower.shape[1]) + int(rng.integers(0, 11))
    assert height <= 30 and width <= 30
    upper_left = int(rng.integers(0, width - upper.shape[1] + 1))
    lower_left = int(rng.integers(0, width - lower.shape[1] + 1))
    lower_top = top_padding + upper.shape[0] + gap
    grid = np.full((height, width), background, dtype=np.int8)
    upper_region = grid[top_padding:top_padding + upper.shape[0], upper_left:upper_left + upper.shape[1]]
    lower_region = grid[lower_top:lower_top + lower.shape[0], lower_left:lower_left + lower.shape[1]]
    upper_region[upper] = upper_color
    lower_region[lower] = lower_color
    return grid, _lower_crop(grid)


def _roles(rng):
    background = 0 if rng.random() < 0.3 else int(rng.integers(0, 10))
    objects = rng.choice([c for c in range(10) if c != background], size=2, replace=False)
    return background, int(objects[0]), int(objects[1])


def _valid_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is not None:
        num_train = _valid_count(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _valid_count(num_test, "num_test", 1, 4)
    if num_train is None:
        num_train = int(rng.integers(2 if num_test is None else num_test + 1, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    source_shape_shift = num_train >= 3 and rng.random() < 0.16
    first_roles = _roles(rng)
    second_background = int(rng.choice([c for c in range(10) if c not in first_roles]))
    second_roles = (second_background, first_roles[2], first_roles[1])
    anchors = [
        (_shape(rng, "solid", int(rng.integers(5, 8)), int(rng.integers(5, 8))),
         _shape(rng, str(rng.choice(("T", "L", "U", "branch"))),
                int(rng.integers(3, 5)), int(rng.integers(3, 5))),
         first_roles, int(rng.integers(1, 3)), 0),
        (_shape(rng, str(rng.choice(("L", "T", "U", "notched"))), 3, 3),
         _shape(rng, str(rng.choice(("T", "L", "U", "branch"))) if source_shape_shift else "solid",
                int(rng.integers(5, 7)) if source_shape_shift else int(rng.integers(4, 7)),
                int(rng.integers(5, 8))),
         second_roles, int(rng.integers(3, 7)), int(rng.integers(1, 4))),
    ]
    rng.shuffle(anchors)
    puzzle = {"train": [], "test": []}
    used = set()
    for label, count in (("train", num_train), ("test", num_test)):
        regime = "source_train" if label == "train" else "target_test"
        for index in range(count):
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                if label == "train" and index < 2:
                    upper, lower, roles, gap, bottom_padding = anchors[index]
                else:
                    upper = _random_shape(pair_rng, regime)
                    if source_shape_shift:
                        kinds = (("T", "L", "U", "ring", "branch", "notched")
                                 if label == "train" else ("solid", "line"))
                        lower = _shape(pair_rng, str(pair_rng.choice(kinds)))
                    else:
                        lower = _random_shape(pair_rng, regime)
                    roles = _roles(pair_rng)
                    long_gap = pair_rng.random() < (0.42 if label == "train" else 0.58)
                    gap = int(pair_rng.integers(3, 8) if long_gap else pair_rng.integers(1, 3))
                    bottom_padding = (0 if pair_rng.random() < (0.48 if label == "train" else 0.43)
                                      else int(pair_rng.integers(1, 4)))
                inp, out = _compose(pair_rng, upper, lower, roles, gap, bottom_padding)
                key = (inp.shape, inp.tobytes())
                rows = tuple("".join(str(int(v)) for v in row) for row in inp)
                if key not in used and rows not in _ORIGINAL_INPUTS:
                    used.add(key)
                    puzzle[label].append({"input": inp, "output": out})
                    break
            else:
                raise RuntimeError("could not create a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(inp, out):
    background, upper, lower, gap = _analyze_input(inp)
    crop = inp[lower["top"]:lower["bottom"] + 1,
               lower["left"]:lower["right"] + 1]
    assert np.array_equal(out, crop)
    assert np.all(np.isin(out, (background, lower["color"])))
    return background, upper, lower, gap


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    input_keys, pair_keys, train_features = set(), set(), []
    for label, sequence in (("train", train), ("test", test)):
        for pair in sequence:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            ikey = (inp.shape, inp.astype(np.int16).tobytes())
            pkey = (ikey, out.shape, out.astype(np.int16).tobytes())
            assert ikey not in input_keys and pkey not in pair_keys
            input_keys.add(ikey)
            pair_keys.add(pkey)
            rows = tuple("".join(str(int(v)) for v in row) for row in inp)
            assert rows not in _ORIGINAL_INPUTS and not np.array_equal(inp, out)
            background, upper, lower, gap = _inspect_pair(inp, out)
            if label == "train":
                train_features.append((background, upper, lower, gap, inp, out))
    assert len({f[0] for f in train_features}) >= 2
    assert len({f[2]["color"] for f in train_features}) >= 2
    assert any(np.any(f[5] == f[0]) for f in train_features)
    assert (any(np.all(f[5] == f[2]["color"]) for f in train_features)
            or any(np.any(f[4][f[1]["top"]:f[1]["bottom"] + 1,
                                   f[1]["left"]:f[1]["right"] + 1] == f[0])
                   for f in train_features))
    assert any(f[3] <= 2 for f in train_features) and any(f[3] >= 3 for f in train_features)
    assert any(f[4].shape[0] - f[2]["bottom"] - 1 > 0 for f in train_features)
    assert any(f[1]["pixels"] > f[2]["pixels"] for f in train_features)
    assert any(f[1]["pixels"] < f[2]["pixels"] for f in train_features)
    assert any(f[1]["right"] - f[1]["left"] > f[2]["right"] - f[2]["left"] for f in train_features)
    assert any(f[1]["right"] - f[1]["left"] < f[2]["right"] - f[2]["left"] for f in train_features)


def _fingerprint(puzzle):
    return tuple((label, pair["input"].shape, pair["input"].tobytes(),
                  pair["output"].shape, pair["output"].tobytes())
                 for label in ("train", "test") for pair in puzzle[label])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    backgrounds, input_sizes, output_sizes, layouts = set(), set(), set(), set()
    keys = ("train_solid", "train_irregular", "test_solid", "test_irregular",
            "train_thin", "test_thin", "train_short", "train_long", "test_short",
            "test_long", "train_padded", "test_padded", "train_hole", "test_hole")
    tallies = {key: 0 for key in keys}
    within_width = beyond_width = 0
    source_shape_shift_puzzles = 0
    varying_background_puzzles = 0
    size_reversal_puzzles = 0
    equal_span_pairs = 0
    horizontal_offsets = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_info = [_inspect_pair(p["input"], p["output"])
                      for p in puzzle["train"]]
        source_shape_shift_puzzles += int(
            all(np.any(p["output"] == f[0]) for p, f in zip(puzzle["train"], train_info))
            and all(np.unique(p["output"]).size == 1 for p in puzzle["test"])
        )
        varying_background_puzzles += int(len({f[0] for f in train_info}) >= 2)
        size_reversal_puzzles += int(
            any(f[1]["pixels"] > f[2]["pixels"] for f in train_info)
            and any(f[1]["pixels"] < f[2]["pixels"] for f in train_info)
        )
        widest_train = max(pair["output"].shape[1] for pair in puzzle["train"])
        for label in ("train", "test"):
            for pair in puzzle[label]:
                inp, out = pair["input"], pair["output"]
                background, upper, lower, gap = _inspect_pair(inp, out)
                backgrounds.add(background)
                input_sizes.add(inp.shape)
                output_sizes.add(out.shape)
                layouts.add((inp.shape, upper["top"], upper["left"],
                             lower["top"], lower["left"], gap))
                equal_span_pairs += int(upper["right"] - upper["left"] ==
                                        lower["right"] - lower["left"])
                horizontal_offsets += int(upper["left"] != lower["left"])
                tallies[label + ("_solid" if np.all(out == lower["color"]) else "_irregular")] += 1
                tallies[label + ("_short" if gap <= 2 else "_long")] += 1
                tallies[label + "_thin"] += int(min(out.shape) == 1)
                tallies[label + "_hole"] += int(np.any(out[1:-1, 1:-1] == background))
                tallies[label + "_padded"] += int(lower["bottom"] < inp.shape[0] - 1)
                if label == "test":
                    within_width += int(out.shape[1] <= widest_train)
                    beyond_width += int(out.shape[1] > widest_train)
    assert len({_fingerprint(p) for p in puzzles}) >= 40
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) >= 8 and len(input_sizes) >= 30
    assert len(output_sizes) >= 20 and len(layouts) >= 100
    assert all(tallies[key] > 0 for key in tallies)
    assert within_width > 0 and beyond_width > 0
    assert source_shape_shift_puzzles > 0
    assert varying_background_puzzles == 42 and size_reversal_puzzles == 42
    assert equal_span_pairs > 0 and horizontal_offsets > 0

    requests = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                (2, None), (10, None), (6, 3))
    for index, (nt, ns) in enumerate(requests):
        first = generate_puzzle(1000 + index, nt, ns)
        second = generate_puzzle(1000 + index, nt, ns)
        assert _fingerprint(first) == _fingerprint(second)
        if nt is not None:
            assert len(first["train"]) == nt
        if ns is not None:
            assert len(first["test"]) == ns
    assert _fingerprint(generate_puzzle(314159)) == _fingerprint(generate_puzzle(314159))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (4, 4), (3, 4)):
        try:
            generate_puzzle(7, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {nt}, {ns}")
