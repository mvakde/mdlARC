"""Generate grids in which every shortest horizontal endpoint pair is joined."""

import numpy as np


_SOURCE_PAIRS = (
    (("200002", "000000", "000000", "200002"), ("222222", "000000", "000000", "222222")),
    (("200002", "020020", "002200", "020020"), ("200002", "020020", "002200", "020020")),
    (("000000", "020020", "200002", "020020"), ("000000", "022220", "200002", "022220")),
    (("002200", "020020", "200002", "020020"), ("002200", "020020", "200002", "020020")),
    (("200002", "020020", "020020", "200002"), ("200002", "022220", "022220", "200002")),
    (("200002", "020020", "000000", "200002"), ("200002", "022220", "000000", "200002")),
)


def _checked_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _select_counts(rng: np.random.Generator, num_train: int | None,
                   num_test: int | None) -> tuple[int, int]:
    train = _checked_count(num_train, "num_train")
    test = _checked_count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is None and test is None:
        test = int(rng.integers(1, 5))
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _pair_key(pair: dict) -> tuple:
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _source_copy(pair: dict) -> bool:
    if pair["input"].shape != (4, 6) or pair["output"].shape != (4, 6):
        return False
    grids = tuple(tuple("".join(str(int(x)) for x in row) for row in pair[part])
                  for part in ("input", "output"))
    return grids in _SOURCE_PAIRS


def _sample_palettes(rng: np.random.Generator, train_count: int,
                     test_count: int, source_like: bool) -> list[tuple[int, int]]:
    all_palettes = [(background, marker) for background in range(10)
                    for marker in range(10) if background != marker]
    indices = rng.choice(len(all_palettes), size=min(4, train_count), replace=False)
    pool = [all_palettes[int(i)] for i in indices]
    if source_like or rng.random() < 0.22:
        return [pool[0]] * (train_count + test_count)
    pool = pool[:int(rng.integers(2, len(pool) + 1))]
    train_palettes = [pool[i % len(pool)] for i in range(train_count)]
    test_palettes = [train_palettes[int(rng.integers(0, train_count))]
                     for _ in range(test_count)]
    return train_palettes + test_palettes


def _sample_size(rng: np.random.Generator, source_like: bool) -> tuple[int, int]:
    if source_like:
        return int(rng.integers(3, 7)), int(rng.choice((6, 8)))
    width_roll = rng.random()
    if width_roll < 0.25:
        width = int(rng.integers(6, 10))
    elif width_roll < 0.72:
        width = int(rng.integers(10, 18))
    else:
        width = int(rng.integers(18, 27))
    height_roll = rng.random()
    if height_roll < 0.30:
        height = int(rng.integers(3, 7))
    elif height_roll < 0.72:
        height = int(rng.integers(7, 14))
    else:
        height = int(rng.integers(14, 25))
    return height, width


def _place_pair(rng: np.random.Generator, width: int, gap: int,
                source_like: bool, excluded: set[int]) -> int:
    centered = (width - 1 - gap) // 2
    if source_like:
        return centered
    choices = list(range(width - gap))
    if len(choices) > len(excluded):
        choices = [left for left in choices if left not in excluded]
    if (width - 1 - gap) % 2 == 0 and centered in choices:
        if rng.random() < 0.20:
            return centered
    return choices[int(rng.integers(0, len(choices)))]


def _make_pair(rng: np.random.Generator, mode: str,
               palette: tuple[int, int], source_like: bool) -> dict:
    height, width = _sample_size(rng, source_like)
    centered_pair = source_like or bool(rng.random() < 0.15)
    if centered_pair and mode.startswith("adjacent") and width % 2:
        centered_pair = False
    occupied = int(rng.integers(3, height + 1))
    legal = list(range(1, width))
    if centered_pair:
        legal = [gap for gap in legal if (width - 1 - gap) % 2 == 0]
    if mode.startswith("adjacent"):
        minimum = 1
    else:
        candidates = [gap for gap in legal if 2 <= gap < legal[-1]]
        minimum = int(rng.choice(candidates))
    minimum_count = 1 if mode.endswith("unique") else int(
        rng.integers(2, min(occupied - 1, 5) + 1))
    wider_legal = [gap for gap in legal if gap > minimum]
    wider_count = occupied - minimum_count
    wider = []
    if wider_count >= 2 and len(wider_legal) >= 2 and rng.random() < 0.55:
        wider.extend(int(x) for x in rng.choice(wider_legal, size=2, replace=False))
    wider.extend(int(rng.choice(wider_legal)) for _ in range(wider_count - len(wider)))
    row_gaps: list[int | None] = [minimum] * minimum_count + wider
    row_gaps.extend([None] * (height - occupied))
    rng.shuffle(row_gaps)
    background, marker = palette
    input_grid = np.full((height, width), background, dtype=np.int8)
    endpoints = []
    used_minimum_positions: set[int] = set()
    used_wider_positions: dict[int, set[int]] = {}
    for row, gap in enumerate(row_gaps):
        if gap is None:
            continue
        excluded = used_minimum_positions if gap == minimum else used_wider_positions.setdefault(gap, set())
        left = _place_pair(rng, width, gap, centered_pair, excluded)
        excluded.add(left)
        right = left + gap
        input_grid[row, left] = marker
        input_grid[row, right] = marker
        endpoints.append((row, gap, left, right))
    output_grid = input_grid.copy()
    for row, gap, left, right in endpoints:
        if gap == minimum:
            output_grid[row, left + 1:right] = marker
    return {"input": input_grid, "output": output_grid}


def _fallback_pair(index: int, palette: tuple[int, int], mode: str) -> dict:
    height = 17 + index
    background, marker = palette
    grid = np.full((height, 30), background, dtype=np.int8)
    minimum = 1 if mode.startswith("adjacent") else 2
    rows = (0, 2) if mode.endswith("tie") else (0,)
    for row in rows:
        grid[row, index] = marker
        grid[row, index + minimum] = marker
    grid[1, 0] = marker
    grid[1, 8] = marker
    if len(rows) == 1:
        grid[2, 1] = marker
        grid[2, 10] = marker
    result = grid.copy()
    for row in rows:
        result[row, index + 1:index + minimum] = marker
    return {"input": grid, "output": result}


def _analyze_input(grid: np.ndarray) -> dict:
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2, "input needs one background and one marker"
    assert counts[0] != counts[1], "background must be identifiable"
    background = int(colors[int(np.argmax(counts))])
    marker = int(colors[int(np.argmin(counts))])
    rows = []
    empty = []
    for row_index, row in enumerate(grid):
        positions = np.flatnonzero(row == marker)
        assert len(positions) in (0, 2), "row must have zero or two endpoints"
        if len(positions) == 0:
            empty.append(row_index)
        else:
            left, right = (int(x) for x in positions)
            rows.append((row_index, right - left, left, right))
    assert len(rows) >= 3, "pairs need several rows to compare"
    minimum = min(item[1] for item in rows)
    assert any(item[1] > minimum for item in rows), "pairs need a wider span"
    selected = [item for item in rows if item[1] == minimum]
    expected = grid.copy()
    for row, _, left, right in selected:
        expected[row, left + 1:right] = marker
    return {"palette": (background, marker), "rows": rows, "empty": empty,
            "minimum": minimum, "selected": selected, "expected": expected}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    observed: dict[str, list[dict]] = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and inp.dtype == out.dtype
            assert inp.shape[0] >= 3 and inp.shape[1] >= 6
            key = _grid_key(inp)
            pair_key = _pair_key(pair)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _source_copy(pair)
            info = _analyze_input(inp)
            assert np.array_equal(out, info["expected"]), "incorrect shortest-gap fill"
            assert set(int(x) for x in np.unique(out)) == set(info["palette"])
            observed[split].append(info)
    train = observed["train"]
    assert any(info["minimum"] > 1 and len(info["selected"]) >= 2 for info in train)
    assert any(info["minimum"] > 1 and len(info["selected"]) == 1 for info in train)
    if train_count >= 3:
        assert any(info["minimum"] == 1 and len(info["selected"]) >= 2 for info in train)
    train_palettes = {info["palette"] for info in train}
    assert all(info["palette"] in train_palettes for info in observed["test"])
    if len({info["palette"] for group in observed.values() for info in group}) > 1:
        assert len(train_palettes) > 1


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _select_counts(rng, num_train, num_test)
    source_like = bool(rng.random() < 0.12)
    palettes = _sample_palettes(rng, train_count, test_count, source_like)
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen_inputs: set[tuple] = set()
    for index, pair_seed in enumerate(pair_seeds):
        split = "train" if index < train_count else "test"
        local_index = index if split == "train" else index - train_count
        pair_rng = np.random.default_rng(int(pair_seed))
        if split == "train" and local_index == 0:
            mode = "visible_tie"
        elif split == "train" and local_index == 1:
            mode = "visible_unique"
        elif split == "train" and local_index == 2:
            mode = "adjacent_tie"
        else:
            mode = str(pair_rng.choice(("visible_tie", "visible_unique",
                                        "adjacent_tie", "adjacent_unique"),
                                       p=(0.28, 0.34, 0.22, 0.16)))
        for _ in range(80):
            pair = _make_pair(pair_rng, mode, palettes[index], source_like)
            if _grid_key(pair["input"]) not in seen_inputs and not _source_copy(pair):
                break
        else:
            pair = _fallback_pair(index, palettes[index], mode)
        key = _grid_key(pair["input"])
        assert key not in seen_inputs
        seen_inputs.add(key)
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in automatic}
    test_counts = {len(p["test"]) for p in automatic}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    structures = set()
    features = {name: 0 for name in (
        "offcenter", "offset_tie", "wider_offset_repeat", "width6", "tall",
        "zero_empty_test", "many_empty", "unique_visible_train", "adjacent_tie_train",
        "tie_test", "adjacent_test", "palette_change", "all_centered_compact",
        "mixed_centering_puzzle")}
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        palettes = []
        centered_compact = True
        centered_statuses = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _analyze_input(inp)
                palettes.append(info["palette"])
                assert np.array_equal(out, info["expected"])
                assert not _source_copy(pair)
                structures.add((inp.shape, tuple(tuple(np.flatnonzero(row == info["palette"][1]))
                                                 for row in inp)))
                gaps = [gap for _, gap, _, _ in info["rows"]]
                centered = all(left + right == inp.shape[1] - 1
                               for _, _, left, right in info["rows"])
                centered_statuses.add(centered)
                centered_compact &= centered and inp.shape[1] <= 8 and inp.shape[0] <= 6
                features["offcenter"] += not centered
                selected_lefts = {left for _, _, left, _ in info["selected"]}
                features["offset_tie"] += len(info["selected"]) >= 2 and len(selected_lefts) >= 2
                features["wider_offset_repeat"] += any(
                    len({left for _, g, left, _ in info["rows"] if g == gap}) >= 2
                    for gap in set(gaps) if gap > info["minimum"])
                features["width6"] += inp.shape[1] == 6
                features["tall"] += len(info["rows"]) > 6
                features["many_empty"] += len(info["empty"]) >= 3
                if split == "train":
                    features["unique_visible_train"] += info["minimum"] > 1 and len(info["selected"]) == 1
                    features["adjacent_tie_train"] += info["minimum"] == 1 and len(info["selected"]) >= 2
                else:
                    features["zero_empty_test"] += not info["empty"]
                    features["tie_test"] += len(info["selected"]) >= 2
                    features["adjacent_test"] += info["minimum"] == 1
        features["palette_change"] += len(set(palettes)) > 1
        features["all_centered_compact"] += centered_compact
        features["mixed_centering_puzzle"] += len(centered_statuses) > 1
    assert len(structures) >= 120
    assert all(value > 0 for value in features.values()), features
    assert any(_pair_key(a) != _pair_key(b) for a, b in zip(automatic[0]["train"], automatic[1]["train"]))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                        (2, None), (10, None)):
        puzzle = generate_puzzle(451, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        duplicate = generate_puzzle(451, train, test)
        assert all(_pair_key(left) == _pair_key(right)
                   for split in ("train", "test")
                   for left, right in zip(puzzle[split], duplicate[split]))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(451, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
