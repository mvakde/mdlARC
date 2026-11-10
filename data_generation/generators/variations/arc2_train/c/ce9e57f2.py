"""Generate separated boundary-aligned bars and recolor their near halves."""

import numpy as np


_ORIENTATIONS = ("bottom", "top", "left", "right")
_SOURCE_LAYOUTS = {
    (8, (7, 5, 3, 6)),
    (7, (6, 5, 4, 3)),
    (9, (7, 3, 4, 8)),
    (11, (10, 9, 2, 5)),
}


def _validated_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _validated_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    num_train = _validated_count("num_train", num_train, 2, 10)
    num_test = _validated_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = max(2 + (seed // 4) % 9, num_test + 1)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("pair counts require 2 <= train <= 10, 1 <= test <= 4, train > test")
    return num_train, num_test


def _to_bottom(grid: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "bottom":
        return grid
    if orientation == "top":
        return np.flipud(grid)
    if orientation == "left":
        return np.rot90(grid, 1)
    if orientation == "right":
        return np.rot90(grid, 3)
    raise ValueError("unknown orientation")


def _from_bottom(grid: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "bottom":
        return grid
    if orientation == "top":
        return np.flipud(grid)
    if orientation == "left":
        return np.rot90(grid, 3)
    if orientation == "right":
        return np.rot90(grid, 1)
    raise ValueError("unknown orientation")


def _sample_geometry(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    for _ in range(40):
        margins = [int(rng.integers(1, 5)), int(rng.integers(1, 5))]
        if rng.random() < 0.2:
            gaps = [int(rng.integers(1, 5))] * (count - 1)
        else:
            gaps = [int(rng.integers(1, 5)) for _ in range(count - 1)]
        width = sum(margins) + count + sum(gaps)
        if width <= 30:
            columns = [margins[0]]
            for gap in gaps:
                columns.append(columns[-1] + gap + 1)
            return width, tuple(columns)
    # The independent draws above almost always fit; this guarantees a legal end.
    return 2 * count + 1, tuple(range(1, 2 * count, 2))


def _sample_lengths(rng: np.random.Generator, height: int, count: int,
                    allow_one: bool, anchors: tuple[int, ...]) -> tuple[int, ...]:
    profile = str(rng.choice(("mixed", "even", "odd", "equal", "cluster", "wide"),
                             p=(0.30, 0.14, 0.14, 0.12, 0.18, 0.12)))
    choices = list(range(2, height))
    if profile == "even":
        choices = [v for v in choices if v % 2 == 0]
    elif profile == "odd":
        choices = [v for v in choices if v % 2 == 1]
    elif profile == "cluster":
        center = int(rng.integers(3, height))
        choices = [v for v in choices if abs(v - center) <= 2]
    if profile == "equal":
        lengths = [int(rng.choice(choices))] * count
    elif profile == "wide":
        lengths = [int(rng.choice(range(2, min(5, height)))),
                   int(rng.choice(range(max(3, height - 3), height)))]
        lengths.extend(int(rng.choice(choices)) for _ in range(count - 2))
        rng.shuffle(lengths)
    else:
        lengths = [int(rng.choice(choices)) for _ in range(count)]
    if allow_one and not anchors and rng.random() < 0.18:
        lengths[int(rng.integers(count))] = 1
    if anchors:
        slots = rng.choice(count, size=len(anchors), replace=False)
        for slot, value in zip(slots, anchors):
            lengths[int(slot)] = value
    return tuple(lengths)


def _sample_pair(rng: np.random.Generator, regime: str, orientation: str,
                 colors: tuple[int, int, int], allow_one: bool,
                 training_index: int | None) -> dict:
    if regime == "source_train":
        height = int(rng.integers(8 if training_index in (0, 1) else 7, 13))
    elif regime == "target_test":
        height = int(rng.integers(7, 13) if rng.random() < 0.55 else rng.integers(11, 19))
    else:
        raise ValueError("unknown pair regime")
    count = int(rng.choice((2, 3, 4, 5, 6), p=(0.13, 0.18, 0.35, 0.21, 0.13)))
    if training_index == 0 and allow_one:
        count = max(3, count)
    width, columns = _sample_geometry(rng, count)
    anchors = ()
    if training_index == 0:
        even = list(range(4, height, 2))
        selected = rng.choice(even, size=2, replace=False)
        anchors = tuple(int(v) for v in selected)
        if allow_one:
            anchors += (1,)
    elif training_index == 1:
        odd = list(range(5, height, 2))
        selected = rng.choice(odd, size=2, replace=False)
        anchors = tuple(int(v) for v in selected)
    lengths = _sample_lengths(rng, height, count, allow_one, anchors)
    background, foreground, accent = colors
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for column, length in zip(columns, lengths):
        inp[height - length:, column] = foreground
        out[height - length:, column] = foreground
        half = length // 2
        if half:
            out[height - half:, column] = accent
    return {"input": _from_bottom(inp, orientation).copy(),
            "output": _from_bottom(out, orientation).copy()}


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(int(v) for v in grid.shape), tuple(int(v) for v in grid.flat))


def _analyze_pair(pair: dict) -> dict:
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    assert len({int(inp[0, 0]), int(inp[0, -1]), int(inp[-1, 0]), int(inp[-1, -1])}) == 1
    background = int(inp[0, 0])
    input_colors = {int(v) for v in np.unique(inp)}
    assert len(input_colors) == 2 and background in input_colors
    foreground = next(iter(input_colors - {background}))
    matches = []
    for orientation in _ORIENTATIONS:
        canonical = _to_bottom(inp, orientation)
        height, width = canonical.shape
        if not (7 <= height <= 18 and 5 <= width <= 30):
            continue
        if not np.all(canonical[0] == background):
            continue
        if not np.all(canonical[:, 0] == background) or not np.all(canonical[:, -1] == background):
            continue
        columns = tuple(int(v) for v in np.flatnonzero(canonical[-1] == foreground))
        if not (2 <= len(columns) <= 6 and all(b - a >= 2 for a, b in zip(columns, columns[1:]))):
            continue
        lengths = []
        valid = True
        expected_input = np.full(canonical.shape, background, dtype=canonical.dtype)
        for column in columns:
            rows = np.flatnonzero(canonical[:, column] == foreground)
            length = len(rows)
            if not (1 <= length <= height - 1) or not np.array_equal(rows, np.arange(height - length, height)):
                valid = False
                break
            expected_input[height - length:, column] = foreground
            lengths.append(length)
        if valid and np.array_equal(canonical, expected_input) and max(lengths) >= 2:
            matches.append((orientation, canonical, columns, tuple(lengths)))
    assert len(matches) == 1
    orientation, canonical, columns, lengths = matches[0]
    introduced = {int(v) for v in np.unique(out)} - input_colors
    assert len(introduced) == 1
    accent = next(iter(introduced))
    expected = inp.copy()
    expected_canonical = _to_bottom(expected, orientation)
    height, width = canonical.shape
    for column, length in zip(columns, lengths):
        half = length // 2
        if half:
            expected_canonical[height - half:, column] = accent
    assert np.array_equal(out, expected)
    assert {int(v) for v in np.unique(out)} == {background, foreground, accent}
    return {"orientation": orientation, "height": height, "width": width,
            "columns": columns, "lengths": lengths,
            "left_margin": columns[0], "right_margin": width - 1 - columns[-1],
            "gaps": tuple(b - a - 1 for a, b in zip(columns, columns[1:])),
            "headroom": height - max(lengths),
            "roles": (background, foreground, accent)}


def _is_source_copy(info: dict) -> bool:
    return (info["orientation"] == "bottom" and info["width"] == 9
            and info["columns"] == (1, 3, 5, 7)
            and info["roles"] == (0, 2, 8)
            and (info["height"], info["lengths"]) in _SOURCE_LAYOUTS)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validated_seed(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    accent = int(rng.integers(10))
    color_mode = "variable" if rng.random() < 0.58 else "fixed"
    available = [v for v in range(10) if v != accent]
    fixed_roles = tuple(int(v) for v in rng.choice(available, size=2, replace=False))
    allow_one = bool(rng.random() < 0.28)
    orientation_count = int(rng.choice((1, 2, 3), p=(0.42, 0.43, 0.15)))
    orientation_count = min(orientation_count, num_train)
    orientations = tuple(str(v) for v in rng.choice(_ORIENTATIONS,
                                                     size=orientation_count, replace=False))
    child_seeds = rng.integers(0, 2**63, size=num_train + num_test, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count, regime in (("train", num_train, "source_train"),
                                  ("test", num_test, "target_test")):
        for index in range(count):
            child_index = index if split == "train" else num_train + index
            pair_rng = np.random.default_rng(int(child_seeds[child_index]))
            orientation = orientations[index] if split == "train" and index < len(orientations) else str(pair_rng.choice(orientations))
            for attempt in range(80):
                if color_mode == "fixed" or (split == "train" and index == 0):
                    background, foreground = fixed_roles
                else:
                    background, foreground = (int(v) for v in pair_rng.choice(available, size=2, replace=False))
                    if split == "train" and index == 1 and (background, foreground) == fixed_roles:
                        foreground = next(v for v in available if v not in (background, fixed_roles[1]))
                pair = _sample_pair(pair_rng, regime, orientation,
                                    (background, foreground, accent), allow_one,
                                    index if split == "train" else None)
                info = _analyze_pair(pair)
                key = _grid_key(pair["input"])
                if key not in seen and not _is_source_copy(info):
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a unique non-source pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    accents = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp_key = _grid_key(pair["input"])
            pair_key = (inp_key, _grid_key(pair["output"]))
            assert inp_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(inp_key)
            seen_pairs.add(pair_key)
            info = _analyze_pair(pair)
            assert not _is_source_copy(info)
            infos[split].append(info)
            accents.add(info["roles"][2])
    assert len(accents) == 1
    train_orientations = {info["orientation"] for info in infos["train"]}
    assert all(info["orientation"] in train_orientations for info in infos["test"])
    train_lengths = [length for info in infos["train"] for length in info["lengths"]]
    assert len(set(train_lengths)) >= 4
    assert any(length >= 4 and length % 2 == 0 for length in train_lengths)
    assert any(length >= 5 and length % 2 == 1 for length in train_lengths)
    if any(1 in info["lengths"] for info in infos["test"]):
        assert 1 in train_lengths
    for info in infos["train"]:
        assert 7 <= info["height"] <= 12
    for info in infos["test"]:
        assert 7 <= info["height"] <= 18


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    shapes, layouts, roles, structures = set(), set(), set(), set()
    counts, orientations, gap_patterns, headrooms = set(), set(), set(), set()
    same_scale, large_test = 0, 0
    repeated, all_even, all_odd, one, varied_colors, varied_orientations = 0, 0, 0, 0, 0, 0
    varied_width = varied_count = varied_gaps = varied_left = varied_right = 0
    test_equal = test_clustered = test_without_extremes = test_one = regular_four = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_infos = [_analyze_pair(p) for p in puzzle["train"]]
        test_infos = [_analyze_pair(p) for p in puzzle["test"]]
        all_infos = train_infos + test_infos
        varied_colors += len({i["roles"][:2] for i in all_infos}) > 1
        varied_orientations += len({i["orientation"] for i in all_infos}) > 1
        varied_width += len({i["width"] for i in all_infos}) > 1
        varied_count += len({len(i["lengths"]) for i in all_infos}) > 1
        varied_gaps += len({i["gaps"] for i in all_infos}) > 1
        varied_left += len({i["left_margin"] for i in all_infos}) > 1
        varied_right += len({i["right_margin"] for i in all_infos}) > 1
        same_scale += any(i["height"] <= max(t["height"] for t in train_infos) for i in test_infos)
        large_test += any(i["height"] > max(t["height"] for t in train_infos) for i in test_infos)
        for info in test_infos:
            lengths = info["lengths"]
            test_equal += len(set(lengths)) == 1
            test_clustered += max(lengths) - min(lengths) <= 2
            test_without_extremes += min(lengths) >= 4 and info["headroom"] >= 3
            test_one += 1 in lengths
            regular_four += len(lengths) == 4 and len(set(info["gaps"])) == 1
        for pair, info in zip(puzzle["train"] + puzzle["test"], all_infos):
            lengths = info["lengths"]
            shapes.add(pair["input"].shape)
            layouts.add((info["orientation"], info["height"], info["width"],
                         info["columns"], lengths))
            roles.add(info["roles"])
            structures.add((info["orientation"], info["columns"], lengths))
            counts.add(len(lengths))
            orientations.add(info["orientation"])
            gap_patterns.add(info["gaps"])
            headrooms.add(info["headroom"])
            repeated += len(set(lengths)) < len(lengths)
            all_even += all(v % 2 == 0 for v in lengths)
            all_odd += all(v % 2 == 1 for v in lengths)
            one += 1 in lengths
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(shapes) >= 30 and len(layouts) >= 150 and len(structures) >= 150
    assert len(roles) >= 30 and counts == {2, 3, 4, 5, 6}
    assert orientations == set(_ORIENTATIONS) and len(gap_patterns) >= 25
    assert len(headrooms) >= 6 and same_scale >= 10 and large_test >= 10
    assert repeated >= 20 and all_even >= 10 and all_odd >= 10 and one >= 5
    assert varied_colors >= 10 and varied_orientations >= 10
    assert varied_width >= 30 and varied_count >= 30 and varied_gaps >= 30
    assert varied_left >= 30 and varied_right >= 30
    assert test_equal >= 5 and test_clustered >= 10 and test_without_extremes >= 5
    assert test_one >= 3 and regular_four >= 3
    assert _puzzle_key(generate_puzzle(17)) == _puzzle_key(generate_puzzle(17))
    assert _puzzle_key(generate_puzzle(314159, 6, 3)) == _puzzle_key(generate_puzzle(314159, 6, 3))
    for index, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(500 + index, ntrain, ntest)
        assert (len(p["train"]), len(p["test"])) == (ntrain, ntest)
    for seed, ntrain, ntest in ((601, None, 1), (602, None, 4),
                                (603, 2, None), (604, 10, None)):
        p = generate_puzzle(seed, ntrain, ntest)
        assert ntrain is None or len(p["train"]) == ntrain
        assert ntest is None or len(p["test"]) == ntest
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 4}):
        try:
            generate_puzzle(700, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
