import numpy as np


_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_SOURCE_SPECS = (
    (8, 11, (1, 8), (5, 1)),
    (10, 12, (2, 1), (8, 9)),
    (12, 11, (10, 2), (1, 8)),
    (12, 13, (8, 3), (2, 11)),
)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _optional_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _optional_count("num_train", num_train, 2, 10)
    test = _optional_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _draw_path(grid: np.ndarray, source: tuple[int, int], target: tuple[int, int], connector: int) -> np.ndarray:
    sr, sc = source
    tr, tc = target
    if sr == tr or sc == tc:
        raise ValueError("markers must occupy different rows and columns")
    output = grid.copy()
    rs = 1 if tr > sr else -1
    cs = 1 if tc > sc else -1
    for row in range(sr + rs, tr + rs, rs):
        output[row, sc] = connector
    for col in range(sc + cs, tc, cs):
        output[tr, col] = connector
    return output


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, grid.tobytes())


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def _check_grid(grid: np.ndarray, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} must be an ndarray")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(all(1 <= size <= 30 for size in grid.shape), f"{label} has invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must use integers")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), f"{label} has invalid colors")


def _analyze_pair(inp: np.ndarray, out: np.ndarray) -> dict:
    _require(inp.shape == out.shape, "input/output dimensions differ")
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 3, "input requires one background and two distinct markers")
    backgrounds = [int(color) for color, count in zip(colors, counts) if count == inp.size - 2]
    _require(len(backgrounds) == 1, "input background is not unambiguous")
    background = backgrounds[0]
    markers = sorted(set(int(x) for x in colors) - {background})
    _require(all(np.count_nonzero(inp == color) == 1 for color in markers), "markers must be singletons")
    added = set(int(x) for x in np.unique(out)) - set(int(x) for x in colors)
    _require(len(added) == 1, "output must add one connector color")
    connector = next(iter(added))
    _require(len(np.unique(out)) == 4, "output must retain all input colors")
    candidates = []
    for source_color, target_color in ((markers[0], markers[1]), (markers[1], markers[0])):
        source = tuple(int(x) for x in np.argwhere(inp == source_color)[0])
        target = tuple(int(x) for x in np.argwhere(inp == target_color)[0])
        if source[0] == target[0] or source[1] == target[1]:
            continue
        expected = _draw_path(inp, source, target, connector)
        if np.array_equal(expected, out):
            candidates.append((source_color, target_color, source, target))
    _require(len(candidates) == 1, "output does not uniquely identify the vertical-first route")
    source_color, target_color, source, target = candidates[0]
    row_gap = abs(source[0] - target[0])
    col_gap = abs(source[1] - target[1])
    _require(np.count_nonzero(out == connector) == row_gap + col_gap - 1,
             "connector cell count is wrong")
    _require(not np.array_equal(inp, out), "input must change")
    orientation = (1 if target[0] > source[0] else -1,
                   1 if target[1] > source[1] else -1)
    height, width = inp.shape
    edge = any(r in (0, height - 1) or c in (0, width - 1) for r, c in (source, target))
    corner = any(r in (0, height - 1) and c in (0, width - 1) for r, c in (source, target))
    return {"background": background, "source_color": source_color,
            "target_color": target_color, "connector": connector,
            "source": source, "target": target, "orientation": orientation,
            "row_gap": row_gap, "col_gap": col_gap, "shape": inp.shape,
            "edge": edge, "corner": corner}


def _matches_source(inp: np.ndarray, out: np.ndarray) -> bool:
    for height, width, source, target in _SOURCE_SPECS:
        if inp.shape != (height, width):
            continue
        original = np.zeros((height, width), dtype=np.int8)
        original[source] = 8
        original[target] = 2
        if np.array_equal(inp, original) and np.array_equal(out, _draw_path(original, source, target, 4)):
            return True
    return False


def _infer_roles(train: list) -> tuple[tuple[int, int, int], set[int]]:
    analyses = [_analyze_pair(pair["input"], pair["output"]) for pair in train]
    roles = {(x["source_color"], x["target_color"], x["connector"]) for x in analyses}
    _require(len(roles) == 1, "training contradicts marker or connector roles")
    return next(iter(roles)), {x["background"] for x in analyses}


def _predict_from_training(train: list, test_input: np.ndarray) -> np.ndarray:
    (source_color, target_color, connector), backgrounds = _infer_roles(train)
    colors, counts = np.unique(test_input, return_counts=True)
    candidates = [int(color) for color, count in zip(colors, counts)
                  if count == test_input.size - 2]
    _require(len(candidates) == 1 and candidates[0] in backgrounds,
             "test background was not established by training")
    background = candidates[0]
    _require(set(int(x) for x in colors) == {background, source_color, target_color},
             "test marker roles differ from training")
    _require(np.count_nonzero(test_input == source_color) == 1 and
             np.count_nonzero(test_input == target_color) == 1,
             "test markers must be singletons")
    source = tuple(int(x) for x in np.argwhere(test_input == source_color)[0])
    target = tuple(int(x) for x in np.argwhere(test_input == target_color)[0])
    return _draw_path(test_input, source, target, connector)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} needs exactly input and output")
            inp, out = pair["input"], pair["output"]
            _check_grid(inp, label + ".input")
            _check_grid(out, label + ".output")
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(not _matches_source(inp, out), "exact source-task copy")
            analyses[split].append(_analyze_pair(inp, out))
    train_info, test_info = analyses["train"], analyses["test"]
    learned, backgrounds = _infer_roles(puzzle["train"])
    _require(any(x["row_gap"] >= 3 and x["col_gap"] >= 3 for x in train_info),
             "training lacks a clear long-bend demonstration")
    train_directions = {x["orientation"] for x in train_info}
    for pair, info in zip(puzzle["test"], test_info):
        _require((info["source_color"], info["target_color"], info["connector"]) == learned,
                 "test changes learned marker or connector roles")
        _require(info["background"] in backgrounds, "test background is unseen")
        if info["orientation"] not in train_directions:
            _require(len(train_directions) >= 2,
                     "unseen direction needs multiple training directions")
        prediction = _predict_from_training(puzzle["train"], pair["input"])
        _require(np.array_equal(prediction, pair["output"]),
                 "test is not recoverable from training and its input")


def _sample_gap(rng: np.random.Generator, regime: str, clear: bool = False) -> int:
    if clear:
        return int(rng.integers(3, 12))
    weights = (0.10, 0.30, 0.40, 0.20) if regime == "source_train" else (0.16, 0.26, 0.31, 0.27)
    bucket = int(rng.choice(4, p=weights))
    low, high = ((1, 2), (2, 5), (5, 11), (11, 21))[bucket]
    return int(rng.integers(low, high))


def _sample_size(rng: np.random.Generator, gap: int, regime: str) -> tuple[int, int]:
    max_extra = 29 - gap
    weights = (0.24, 0.45, 0.31) if regime == "source_train" else (0.20, 0.38, 0.42)
    mode = int(rng.choice(3, p=weights))
    if mode == 0:
        low, high = 0, min(max_extra, 3)
    elif mode == 1:
        low, high = min(max_extra, 3), min(max_extra, 11)
    else:
        low, high = min(max_extra, 11), max_extra
    extra = int(rng.integers(low, high + 1))
    return gap + 1 + extra, extra


def _margin(rng: np.random.Generator, extra: int, direction: int,
            anchor: str, axis: str) -> int:
    if anchor == "source_corner" or (anchor == "source_edge" and axis == "row"):
        return 0 if direction == 1 else extra
    if anchor == "target_corner" or (anchor == "target_edge" and axis == "row"):
        return extra if direction == 1 else 0
    if extra >= 2:
        return int(rng.integers(1, extra))
    return int(rng.integers(0, extra + 1))


def _make_pair(rng: np.random.Generator, gaps: tuple[int, int],
               orientation: tuple[int, int], colors: tuple[int, int, int, int],
               regime: str, fallback: int = -1) -> dict:
    row_gap, col_gap = gaps
    dr, dc = orientation
    if fallback >= 0:
        height, width = 30, 30
        row_extra, col_extra = 29 - row_gap, 29 - col_gap
        row_margin = fallback % (row_extra + 1)
        col_margin = (fallback * 3 + 1) % (col_extra + 1)
    else:
        height, row_extra = _sample_size(rng, row_gap, regime)
        width, col_extra = _sample_size(rng, col_gap, regime)
        anchor = str(rng.choice(("free", "source_edge", "target_edge",
                                 "source_corner", "target_corner"),
                                p=(0.70, 0.10, 0.10, 0.05, 0.05)))
        row_margin = _margin(rng, row_extra, dr, anchor, "row")
        col_margin = _margin(rng, col_extra, dc, anchor, "col")
    top, left = row_margin, col_margin
    source = (top if dr == 1 else top + row_gap,
              left if dc == 1 else left + col_gap)
    target = (top + row_gap if dr == 1 else top,
              left + col_gap if dc == 1 else left)
    background, source_color, target_color, connector = colors
    inp = np.full((height, width), background, dtype=np.int8)
    inp[source] = source_color
    inp[target] = target_color
    return {"input": inp, "output": _draw_path(inp, source, target, connector)}


def _direction_plan(rng: np.random.Generator, nt: int, ne: int) -> tuple[list, list]:
    mode = int(rng.choice(4, p=(0.20, 0.30, 0.30, 0.20)))
    directions = [tuple(_DIRECTIONS[int(x)]) for x in rng.permutation(4)]
    if mode == 0:
        return [directions[0]] * nt, [directions[0]] * ne
    if mode == 2:
        unseen = directions[0]
        train = [directions[1], directions[2]]
        train.extend(directions[1 + int(rng.integers(0, 3))] for _ in range(nt - 2))
        rng.shuffle(train)
        test = [unseen] + [directions[int(rng.integers(0, 4))] for _ in range(ne - 1)]
        return train, test
    train = [directions[0], directions[1]]
    train.extend(directions[int(rng.integers(0, 4))] for _ in range(nt - 2))
    rng.shuffle(train)
    if mode == 1:
        test = [train[int(rng.integers(0, nt))] for _ in range(ne)]
    else:
        test = [directions[int(rng.integers(0, 4))] for _ in range(ne)]
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt, ne = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = [int(x) for x in rng.choice(10, size=3, replace=False)]
    available = [x for x in range(10) if x not in roles]
    if rng.random() < 0.35:
        background_count = 1
    else:
        background_count = int(rng.integers(2, min(4, nt) + 1))
    backgrounds = [int(x) for x in rng.choice(available, size=background_count, replace=False)]
    train_backgrounds = backgrounds + [backgrounds[int(rng.integers(0, background_count))]
                                       for _ in range(nt - background_count)]
    rng.shuffle(train_backgrounds)
    train_dirs, test_dirs = _direction_plan(rng, nt, ne)
    train_gaps = [(_sample_gap(rng, "source_train", clear=True),
                   _sample_gap(rng, "source_train", clear=True))]
    train_gaps.extend((_sample_gap(rng, "source_train"),
                       _sample_gap(rng, "source_train")) for _ in range(nt - 1))
    test_gaps = []
    for _ in range(ne):
        mode = int(rng.choice(3, p=(0.34, 0.34, 0.32)))
        if mode == 0:
            gaps = train_gaps[int(rng.integers(0, nt))]
        elif mode == 1:
            known_rows = {x[0] for x in train_gaps}
            known_cols = {x[1] for x in train_gaps}
            if rng.random() < 0.5:
                choices = [x for x in range(1, 21) if x not in known_rows]
                gaps = (int(rng.choice(choices)), _sample_gap(rng, "target_test"))
            else:
                choices = [x for x in range(1, 21) if x not in known_cols]
                gaps = (_sample_gap(rng, "target_test"), int(rng.choice(choices)))
        else:
            gaps = (_sample_gap(rng, "target_test"), _sample_gap(rng, "target_test"))
        test_gaps.append(gaps)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, directions, gaps_list, bg_list in (
        ("train", train_dirs, train_gaps, train_backgrounds),
        ("test", test_dirs, test_gaps,
         [backgrounds[int(rng.integers(0, background_count))] for _ in range(ne)]),
    ):
        regime = "source_train" if split == "train" else "target_test"
        for index, (direction, gaps, background) in enumerate(zip(directions, gaps_list, bg_list)):
            colors = (background, roles[0], roles[1], roles[2])
            for attempt in range(48):
                pair = _make_pair(rng, gaps, direction, colors, regime,
                                  fallback=attempt - 24 if attempt >= 24 else -1)
                key = _grid_key(pair["input"])
                if key not in seen and not _matches_source(pair["input"], pair["output"]):
                    break
            else:
                raise AssertionError("deterministic unique-layout fallback exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    train_counts, test_counts, puzzles = set(), set(), set()
    palettes, layouts, shapes, gap_pairs = set(), set(), set(), set()
    directions, easy, unseen, repeated, novel = set(), 0, 0, 0, 0
    short, edge, corner, mixed_bg = 0, 0, 0, 0
    source_edge, target_edge, source_corner, target_corner = 0, 0, 0, 0
    combined = 0
    width_less, width_equal, width_more = 0, 0, 0
    aspect_tall, aspect_wide, roomy_short = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_analyze_pair(x["input"], x["output"]) for x in puzzle["train"]]
        test_info = [_analyze_pair(x["input"], x["output"]) for x in puzzle["test"]]
        info = train_info + test_info
        train_dirs = {x["orientation"] for x in train_info}
        train_gaps = {(x["row_gap"], x["col_gap"]) for x in train_info}
        train_rows = {x["row_gap"] for x in train_info}
        train_cols = {x["col_gap"] for x in train_info}
        train_widths = {x["shape"][1] for x in train_info}
        easy += len({x["orientation"] for x in info}) == 1
        mixed_bg += len({x["background"] for x in train_info}) > 1
        combined += (len(train_dirs) > 1 and
                     len({x["background"] for x in train_info}) > 1 and
                     any(x["edge"] and (x["row_gap"] == 1 or x["col_gap"] == 1)
                         for x in info))
        palettes.add((train_info[0]["source_color"], train_info[0]["target_color"],
                      train_info[0]["connector"]))
        for pair, x in zip(puzzle["test"], test_info):
            _require(np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                                    pair["output"]), "training-only prediction failed")
            unseen += x["orientation"] not in train_dirs
            repeated += (x["row_gap"], x["col_gap"]) in train_gaps
            novel += x["row_gap"] not in train_rows or x["col_gap"] not in train_cols
            width_less += x["shape"][1] < max(train_widths)
            width_equal += x["shape"][1] in train_widths
            width_more += x["shape"][1] > max(train_widths)
        for x in info:
            directions.add(x["orientation"])
            shapes.add(x["shape"])
            gap_pairs.add((x["row_gap"], x["col_gap"], x["orientation"]))
            layouts.add((x["shape"], x["source"], x["target"]))
            short += x["row_gap"] == 1 or x["col_gap"] == 1
            edge += x["edge"]
            corner += x["corner"]
            height, width = x["shape"]
            sr, sc = x["source"]
            tr, tc = x["target"]
            source_edge += sr in (0, height - 1) or sc in (0, width - 1)
            target_edge += tr in (0, height - 1) or tc in (0, width - 1)
            source_corner += sr in (0, height - 1) and sc in (0, width - 1)
            target_corner += tr in (0, height - 1) and tc in (0, width - 1)
            aspect_tall += x["shape"][0] > x["shape"][1]
            aspect_wide += x["shape"][1] > x["shape"][0]
            roomy_short += (max(x["row_gap"], x["col_gap"]) <= 5 and
                            (x["shape"][0] >= 15 or x["shape"][1] >= 15))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    _require(len(puzzles) >= 40 and len(layouts) >= 100 and len(shapes) >= 50,
             "structures or dimensions lack diversity")
    _require(len(palettes) >= 20 and len(gap_pairs) >= 80 and directions == set(_DIRECTIONS),
             "colors, legs, or orientations lack diversity")
    for name, value in (("easy", easy), ("unseen", unseen), ("repeated", repeated),
                        ("novel", novel), ("short", short), ("edge", edge),
                        ("corner", corner), ("mixed background", mixed_bg),
                        ("narrower test", width_less), ("equal test width", width_equal),
                        ("wider test", width_more), ("tall", aspect_tall),
                        ("wide", aspect_wide), ("roomy short L", roomy_short)):
        _require(value >= 3, f"insufficient {name} variation")
    _require(min(source_edge, target_edge) >= 10 and
             min(source_corner, target_corner) >= 3 and combined >= 3,
             "endpoint roles or combined variation lack coverage")
    for index, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(7000 + index, nt, ne)
        _require((len(p["train"]), len(p["test"])) == (nt, ne), "explicit counts ignored")
    for index, (nt, ne) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(8000 + index, nt, ne)
        _require((nt is None or len(p["train"]) == nt) and
                 (ne is None or len(p["test"]) == ne), "partial counts ignored")
    _require(_puzzle_key(generate_puzzle(12345, 7, 3)) ==
             _puzzle_key(generate_puzzle(12345, 7, 3)), "generation is nondeterministic")
    _require(_puzzle_key(generate_puzzle(12345, 7, 3)) !=
             _puzzle_key(generate_puzzle(12346, 7, 3)), "seeds fail to change puzzles")
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(10, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request was accepted")
