import numpy as np


_MIXED_CONNECTED = ("l", "cross", "zigzag", "ring")
_MIXED_OTHER = ("random", "complete_checker", "partial_checker", "clusters")
_FAMILIES = (
    "single", "row", "column", "solid", "l", "cross", "zigzag",
    "ring", "clusters", "random", "complete_checker", "partial_checker",
)
_TEST_FAMILIES = (
    "solid", "l", "cross", "zigzag", "ring", "clusters", "random",
    "complete_checker", "complete_checker", "partial_checker", "partial_checker",
)


def _literal_grid(rows):
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _literal_grid(
            [
                "000000000",
                "000040000",
                "000440000",
                "000004000",
                "000000000",
                "000000000",
                "000000000",
                "000000000",
                "000000000",
            ]
        ),
        _literal_grid(
            [
                "004400",
                "004400",
                "444400",
                "444400",
                "000044",
                "000044",
            ]
        ),
    ),
    (
        _literal_grid(
            [
                "000000000",
                "000000000",
                "004400000",
                "044440000",
                "004400000",
                "000000000",
                "000000000",
                "000000000",
                "000000000",
            ]
        ),
        _literal_grid(
            [
                "00444400",
                "00444400",
                "44444444",
                "44444444",
                "00444400",
                "00444400",
            ]
        ),
    ),
    (
        _literal_grid(
            [
                "000000000",
                "000000000",
                "000000000",
                "000000000",
                "004000000",
                "044000000",
                "004000000",
                "004400000",
                "000000000",
            ]
        ),
        _literal_grid(
            [
                "004400",
                "004400",
                "444400",
                "444400",
                "004400",
                "004400",
                "004444",
                "004444",
            ]
        ),
    ),
    (
        _literal_grid(
            [
                "000000000",
                "000040400",
                "000404040",
                "000040400",
                "000000000",
                "000000000",
                "000000000",
                "000000000",
                "000000000",
            ]
        ),
        _literal_grid(
            [
                "0044004400",
                "0044004400",
                "4400440044",
                "4400440044",
                "0044004400",
                "0044004400",
            ]
        ),
    ),
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _normalize_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _normalize_count(num_train, "num_train", 2, 10)
    num_test = _normalize_count(num_test, "num_test", 1, 4)
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


def _dimensions(rng, family, modest=False):
    if family == "single":
        return (1, 1)
    if family == "row":
        return (1, int(rng.integers(2, 16)))
    if family == "column":
        return (int(rng.integers(2, 16)), 1)
    low = 4 if family == "partial_checker" else 3 if family in ("cross", "ring") else 2
    ceiling = 9 if modest else 16
    mode = int(rng.integers(0, 4))
    if mode == 0:
        side = int(rng.integers(low, ceiling))
        return side, side
    if mode == 1:
        return int(rng.integers(max(low + 1, 4), ceiling)), int(rng.integers(low, min(7, ceiling)))
    if mode == 2:
        return int(rng.integers(low, min(7, ceiling))), int(rng.integers(max(low + 1, 4), ceiling))
    return int(rng.integers(low, ceiling)), int(rng.integers(low, ceiling))


def _preserves_bounds(mask, row, col):
    return (np.count_nonzero(mask[0]) > 1 or row != 0) and (np.count_nonzero(mask[-1]) > 1 or row != mask.shape[0] - 1) and (np.count_nonzero(mask[:, 0]) > 1 or col != 0) and (np.count_nonzero(mask[:, -1]) > 1 or col != mask.shape[1] - 1)


def _motif(rng, family, height, width):
    mask = np.zeros((height, width), dtype=bool)
    if family in ("single", "row", "column", "solid"):
        mask[:] = True
    elif family == "l":
        mask[:, 0] = True
        mask[-1, :] = True
        if rng.random() < 0.5:
            mask = np.flip(mask, axis=0)
        if rng.random() < 0.5:
            mask = np.flip(mask, axis=1)
    elif family == "cross":
        mask[int(rng.integers(1, height - 1)), :] = True
        mask[:, int(rng.integers(1, width - 1))] = True
    elif family == "zigzag":
        row = col = 0
        mask[row, col] = True
        while row < height - 1 or col < width - 1:
            if row == height - 1:
                col += 1
            elif col == width - 1:
                row += 1
            elif rng.random() < 0.5:
                row += 1
            else:
                col += 1
            mask[row, col] = True
    elif family == "ring":
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        if height > 4 and width > 4 and rng.random() < 0.5:
            mask[int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1))] = True
    elif family in ("random", "clusters"):
        probability = float(rng.uniform(0.25, 0.75)) if family == "random" else float(rng.uniform(0.08, 0.30))
        mask = rng.random((height, width)) < probability
        mask[0, int(rng.integers(width))] = True
        mask[-1, int(rng.integers(width))] = True
        mask[int(rng.integers(height)), 0] = True
        mask[int(rng.integers(height)), -1] = True
    elif family in ("complete_checker", "partial_checker"):
        rows, cols = np.indices((height, width))
        parity = 1 if family == "complete_checker" and (height, width) == (3, 5) else int(rng.integers(2))
        mask = (rows + cols) % 2 == parity
        if family == "partial_checker":
            order = rng.permutation(height * width)
            to_remove = int(rng.integers(1, max(2, int(mask.sum() * 0.5))))
            removed = 0
            for index in order:
                row, col = divmod(int(index), width)
                if mask[row, col] and _preserves_bounds(mask, row, col):
                    mask[row, col] = False
                    removed += 1
                    if removed >= to_remove:
                        break
            if removed == 0:
                # The sampled dimensions are at least 4 by 4, so an interior
                # checker cell is removable while retaining all four edges.
                for row in range(1, height - 1):
                    for col in range(1, width - 1):
                        if mask[row, col]:
                            mask[row, col] = False
                            removed = 1
                            break
                    if removed:
                        break
    else:
        raise ValueError("Unknown motif family")
    _require(mask.any(), "Empty motif")
    _require(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any(), "Motif bounds are not tight")
    return mask


def _palette(rng, base, mode, section, index):
    if mode == "fixed":
        return base
    if section == "train" and index == 0:
        return base
    if section == "train" and index == 1 and mode == "swap":
        return base[1], base[0]
    if mode == "swap" and rng.random() < 0.55:
        return base if rng.random() < 0.5 else (base[1], base[0])
    colors = rng.choice(10, size=2, replace=False)
    roles = int(colors[0]), int(colors[1])
    if section == "train" and index == 1 and roles == base:
        return base[1], base[0]
    return roles


def _padding(rng, height, width, touch):
    extra_h = int(rng.integers(1 if touch in ("top", "bottom") else 2, min(10, 30 - height) + 1))
    extra_w = int(rng.integers(1 if touch in ("left", "right") else 2, min(10, 30 - width) + 1))
    if touch == "top":
        top = 0
    elif touch == "bottom":
        top = extra_h
    else:
        top = int(rng.integers(1, extra_h))
    if touch == "left":
        left = 0
    elif touch == "right":
        left = extra_w
    else:
        left = int(rng.integers(1, extra_w))
    return extra_h, extra_w, top, left


def _make_pair(rng, family, dimensions, colors, require_padding=False):
    height, width = dimensions
    background, foreground = colors
    mask = _motif(rng, family, height, width)
    touch = "none" if require_padding else (("none",) * 12 + ("top", "bottom", "left", "right"))[int(rng.integers(16))]
    extra_h, extra_w, top, left = _padding(rng, height, width, touch)
    grid = np.full((height + extra_h, width + extra_w), background, dtype=np.int8)
    region = grid[top:top + height, left:left + width]
    region[mask] = foreground
    crop = grid[top:top + height, left:left + width]
    output = np.repeat(np.repeat(crop, 2, axis=0), 2, axis=1)
    return {"input": grid, "output": output}


def _grid_key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _matches_source_pair(input_grid, output_grid):
    return any(np.array_equal(input_grid, original_input) and np.array_equal(output_grid, original_output) for original_input, original_output in _SOURCE_PAIRS)


def _background(grid):
    corners = (int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1]))
    colors, counts = np.unique(corners, return_counts=True)
    background = int(colors[np.argmax(counts)])
    _require(max(counts) >= 3, "No clear background at the corners")
    return background


def _pair_info(pair):
    grid, output = pair["input"], pair["output"]
    background = _background(grid)
    colors = np.unique(grid)
    _require(len(colors) == 2 and background in colors, "Input must have a background and one foreground")
    foreground = int(colors[colors != background][0])
    coordinates = np.argwhere(grid == foreground)
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0)
    top, bottom, left, right = map(int, (top, bottom, left, right))
    crop = grid[top:bottom + 1, left:right + 1]
    _require(top > 0 or bottom < grid.shape[0] - 1, "No vertical background padding")
    _require(left > 0 or right < grid.shape[1] - 1, "No horizontal background padding")
    touches = sum((top == 0, bottom == grid.shape[0] - 1, left == 0, right == grid.shape[1] - 1))
    _require(touches <= 1, "More than one canvas edge touched")
    expected = np.repeat(np.repeat(crop, 2, axis=0), 2, axis=1)
    _require(np.array_equal(output, expected), "Output must be the exact tight crop enlarged 2x")
    _require(not np.array_equal(grid, output), "Input and output coincide")
    mask = crop == foreground
    return {"background": background, "foreground": foreground, "roles": (background, foreground), "dimensions": tuple(crop.shape), "mask": mask, "bounds": (top, bottom, left, right), "touches": touches, "density": float(mask.mean())}


def _motif_key(info):
    return info["dimensions"], info["mask"].tobytes()


def _test_dimensions(rng, challenge, train_infos):
    if challenge == "same_size":
        eligible = [a["dimensions"] for a in train_infos if min(a["dimensions"]) > 1]
        return eligible[int(rng.integers(len(eligible)))]
    if challenge == "taller":
        height = int(rng.integers(5, 16))
        return height, int(rng.integers(2, min(7, height)))
    if challenge == "larger":
        median_area = float(np.median([a["dimensions"][0] * a["dimensions"][1] for a in train_infos]))
        for _ in range(24):
            dimensions = int(rng.integers(5, 16)), int(rng.integers(5, 16))
            if dimensions[0] * dimensions[1] > median_area:
                return dimensions
        return (15, 15)
    family = _FAMILIES[int(rng.integers(len(_FAMILIES)))]
    return _dimensions(rng, family)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    colors = rng.choice(10, size=2, replace=False)
    base = int(colors[0]), int(colors[1])
    palette_mode = ("fixed", "varied", "swap")[int(rng.integers(3))]
    dense_holdout = rng.random() < 0.25
    puzzle = {"train": [], "test": []}
    seen = set()
    train_infos = []
    train_motifs = set()
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if section == "train" and index < 2:
                families = _MIXED_CONNECTED if index == 0 else _MIXED_OTHER
                family = families[int(rng.integers(len(families)))]
                dimensions = _dimensions(rng, family, modest=True)
                if index == 1:
                    for _ in range(12):
                        if dimensions != train_infos[0]["dimensions"]:
                            break
                        dimensions = _dimensions(rng, family, modest=True)
                    else:
                        dimensions = (5, 6) if train_infos[0]["dimensions"] != (5, 6) else (6, 5)
            else:
                if section == "train":
                    choices = _MIXED_CONNECTED + _MIXED_OTHER if dense_holdout else _FAMILIES
                    family = choices[int(rng.integers(len(choices)))]
                    dimensions = _dimensions(rng, family)
                else:
                    challenge = "dense" if dense_holdout and index == 0 else ("larger", "same_size", "taller", "dense", "free", "free", "one_dim")[int(rng.integers(7))]
                    if challenge == "one_dim":
                        family = ("single", "single", "row", "column")[int(rng.integers(4))]
                        dimensions = _dimensions(rng, family)
                    else:
                        family = _TEST_FAMILIES[int(rng.integers(len(_TEST_FAMILIES)))]
                        if challenge == "free":
                            dimensions = _dimensions(rng, family)
                        elif challenge == "dense":
                            family = "solid" if dense_holdout and index == 0 else ("solid", "l", "cross", "ring", "random")[int(rng.integers(5))]
                            dimensions = _dimensions(rng, family)
                        else:
                            dimensions = _test_dimensions(rng, challenge, train_infos)
                    if min(dimensions) == 1 and dimensions in {a["dimensions"] for a in train_infos}:
                        used_dimensions = {a["dimensions"] for a in train_infos}
                        choices = [(1, length) for length in range(2, 16)] + [(length, 1) for length in range(2, 16)]
                        choices = [shape for shape in choices if shape not in used_dimensions]
                        dimensions = choices[int(rng.integers(len(choices)))]
                        family = "row" if dimensions[0] == 1 else "column"
                    if min(dimensions) < 3 and family in ("cross", "ring"):
                        family = "l"
                    if min(dimensions) < 4 and family == "partial_checker":
                        family = "complete_checker"
                if (section == "test" or index >= 2) and rng.random() < 0.07:
                    family, dimensions = "complete_checker", (3, 5)
            palette = _palette(rng, base, palette_mode, section, index)
            for attempt in range(80):
                if attempt == 24 and section == "train" and (index < 2 or dense_holdout):
                    family = "ring"
                    dimensions = (5, 6) if not train_infos or train_infos[0]["dimensions"] != (5, 6) else (6, 5)
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _make_pair(pair_rng, family, dimensions, palette, section == "train" and index == 0)
                key = _grid_key(pair["input"])
                if key in seen or _matches_source_pair(pair["input"], pair["output"]):
                    continue
                info = _pair_info(pair)
                if section == "train" and (index < 2 or dense_holdout) and not np.any(~info["mask"]):
                    continue
                if section == "test" and _motif_key(info) in train_motifs:
                    if attempt >= 4:
                        family = _FAMILIES[int(rng.integers(len(_FAMILIES)))]
                        dimensions = _dimensions(rng, family)
                    if attempt == 40:
                        family, dimensions = "random", (7, 8)
                    continue
                seen.add(key)
                puzzle[section].append(pair)
                if section == "train":
                    train_infos.append(info)
                    train_motifs.add(_motif_key(info))
                break
            else:
                raise RuntimeError("Unable to construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "Sections must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"]), "Pair counts")
    seen_inputs = set()
    seen_pairs = set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Pair schema")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer), "Grid type")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "Grid dimensions")
                _require(0 <= int(grid.min()) and int(grid.max()) <= 9, "Grid colors")
            input_key = _grid_key(pair["input"])
            pair_key = input_key, _grid_key(pair["output"])
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "Duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(not _matches_source_pair(pair["input"], pair["output"]), "Source pair copy")
            infos[section].append(_pair_info(pair))
    train_infos = infos["train"]
    _require(train_infos[0]["touches"] == 0, "First demonstration needs full padding")
    _require(all(np.any(~info["mask"]) for info in train_infos[:2]), "First two demonstrations must show internal background")
    _require(train_infos[0]["dimensions"] != train_infos[1]["dimensions"], "Demonstrations need different crop sizes")
    train_motifs = {_motif_key(info) for info in train_infos}
    _require(all(_motif_key(info) not in train_motifs for info in infos["test"]), "Test repeats a training motif")
    # All color choices are visible in their own input; the only shared
    # convention inferred from training is tight cropping followed by 2x scale.
    for info in train_infos + infos["test"]:
        _require(info["touches"] <= 1, "Ambiguous background")


def _puzzles_equal(first, second):
    return all(len(first[section]) == len(second[section]) and all(np.array_equal(a[name], b[name]) for a, b in zip(first[section], second[section]) for name in ("input", "output")) for section in ("train", "test"))


def _expect_value_error(seed, num_train, num_test):
    try:
        generate_puzzle(seed, num_train, num_test)
    except ValueError:
        return
    raise AssertionError("Illegal count accepted")


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return count


def _observable_features(info):
    mask = info["mask"]
    height, width = mask.shape
    rows, cols = np.indices(mask.shape)
    parity = (rows + cols) % 2
    one_parity = np.all(parity[mask] == parity[mask][0])
    complete = one_parity and np.array_equal(mask, parity == parity[mask][0])
    return {
        "single": height == width == 1,
        "row": height == 1 and width > 1,
        "column": width == 1 and height > 1,
        "solid": bool(mask.all()) and height > 1 and width > 1,
        "complete_checker": complete and height > 1 and width > 1,
        "partial_checker": one_parity and not complete and height > 1 and width > 1,
        "connected": _components(mask) == 1 and height > 1 and width > 1,
        "separated": _components(mask) > 1,
        "ring": height >= 3 and width >= 3 and mask[0].all() and mask[-1].all() and mask[:, 0].all() and mask[:, -1].all() and np.any(~mask[1:-1, 1:-1]),
        "mixed": bool(np.any(~mask)),
        "edge_touch": info["touches"] == 1,
        "square": height == width,
        "tall": height > width,
        "wide": width > height,
        "large": height >= 10 or width >= 10,
    }


def _visible_crop(grid):
    background = _background(grid)
    positions = np.argwhere(grid != background)
    low = positions.min(axis=0)
    high = positions.max(axis=0) + 1
    return grid[low[0]:high[0], low[1]:high[1]]


def _predict_from_training(train_pairs, test_input):
    possible_scales = [
        scale for scale in range(1, 6)
        if all(
            np.array_equal(
                pair["output"],
                np.repeat(np.repeat(_visible_crop(pair["input"]), scale, axis=0), scale, axis=1),
            )
            for pair in train_pairs
        )
    ]
    _require(possible_scales == [2], "Training does not uniquely show the scale")
    crop = _visible_crop(test_input)
    return np.repeat(np.repeat(crop, possible_scales[0], axis=0), possible_scales[0], axis=1)


def self_test() -> None:
    for seed, train_count, test_count in ((12001, None, None), (12002, 7, 3), (13001, 2, 1), (13002, 10, 1), (13003, 10, 4), (14001, None, 1), (14002, None, 4), (14003, 2, None), (14004, 10, None)):
        first = generate_puzzle(seed, train_count, test_count)
        second = generate_puzzle(seed, train_count, test_count)
        _require(_puzzles_equal(first, second), "Nondeterministic generation")
        validate_generated_puzzle(first)
        if train_count is not None:
            _require(len(first["train"]) == train_count, "Train count changed")
        if test_count is not None:
            _require(len(first["test"]) == test_count, "Test count changed")
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (3.5, 1), (3, False), (3, 1.5)):
        _expect_value_error(16001, train_count, test_count)

    train_counts, test_counts, signatures, dimensions, input_shapes = set(), set(), set(), set(), set()
    feature_counts = {section: {name: 0 for name in _observable_features(_pair_info(generate_puzzle(200, 2, 1)["train"][0]))} for section in ("train", "test")}
    palette_fixed = palette_varied = palette_swapped = 0
    same_size_tests = wider_tests = taller_tests = denser_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple((section, _grid_key(pair["input"]), _grid_key(pair["output"])) for section in ("train", "test") for pair in puzzle[section]))
        train_infos = [_pair_info(pair) for pair in puzzle["train"]]
        test_infos = [_pair_info(pair) for pair in puzzle["test"]]
        roles = [info["roles"] for info in train_infos + test_infos]
        palette_fixed += len(set(roles)) == 1
        palette_varied += len(set(roles)) > 1
        palette_swapped += any((a[1], a[0]) in roles for a in roles)
        for section, infos in (("train", train_infos), ("test", test_infos)):
            for pair, info in zip(puzzle[section], infos):
                dimensions.add(info["dimensions"])
                input_shapes.add(tuple(pair["input"].shape))
                for name, value in _observable_features(info).items():
                    feature_counts[section][name] += int(value)
                # Independently infer the scale from demonstrations and use
                # only the test input for a prediction check.
                if section == "test":
                    predicted = _predict_from_training(puzzle["train"], pair["input"].copy())
                    _require(np.array_equal(predicted, pair["output"]), "Training-only test prediction failed")
        train_dims = {info["dimensions"] for info in train_infos}
        same_size_tests += any(info["dimensions"] in train_dims for info in test_infos)
        wider_tests += any(info["dimensions"][1] > max(a["dimensions"][1] for a in train_infos) for info in test_infos)
        taller_tests += any(info["dimensions"][0] > info["dimensions"][1] for info in test_infos)
        denser_tests += any(info["density"] > max(a["density"] for a in train_infos) for info in test_infos)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "Automatic counts lack range")
    _require(len(signatures) >= 40 and len(dimensions) >= 35 and len(input_shapes) >= 35, "Structural diversity too narrow")
    _require(palette_fixed >= 4 and palette_varied >= 4 and palette_swapped >= 2, "Palette diversity too narrow")
    _require(same_size_tests >= 3 and wider_tests >= 3 and taller_tests >= 3 and denser_tests >= 3, "Test challenge diversity too narrow")
    for section in ("train", "test"):
        for name in ("single", "row", "column", "solid", "complete_checker", "partial_checker", "connected", "separated", "ring", "edge_touch", "square", "tall", "wide", "large"):
            _require(feature_counts[section][name] >= 1, f"Missing {section} feature: {name}")
