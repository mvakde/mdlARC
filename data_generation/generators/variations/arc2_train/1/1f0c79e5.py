import numpy as np


# bit, seed-row offset, seed-column offset, vertical step, horizontal step
_CORNERS = (
    (1, 0, 0, -1, -1), (2, 0, 1, -1, 1),
    (4, 1, 0, 1, -1), (8, 1, 1, 1, 1),
)
_MASKS = tuple(range(1, 15))

# Valid outputs are determined by their inputs, so these exact source inputs
# also identify the five original source input/output pairs.
_SOURCE_INPUTS = frozenset({
    ("000000000", "000000000", "000000000", "000620000",
     "000260000", "000000000", "000000000", "000000000", "000000000"),
    ("000000000", "003300000", "003200000", "000000000",
     "000000000", "000000000", "000000000", "000000000", "000000000"),
    ("000000000", "000000000", "000000000", "000220000",
     "000270000", "000000000", "000000000", "000000000", "000000000"),
    ("000000000", "000000000", "000000000", "000000000",
     "004200000", "004400000", "000000000", "000000000", "000000000"),
    ("000000000", "000000000", "000002200", "000008200",
     "000000000", "000000000", "000000000", "000000000", "000000000"),
})


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _key(grid):
    return (grid.shape, grid.tobytes())


def _rows(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _size(rng, minimum):
    choice = int(rng.integers(0, 12))
    if choice == 0 and minimum <= 9:
        return 9
    if choice == 1:
        return 30
    return int(rng.integers(minimum, 31))


def _axis(rng, maximum, target):
    mode = int(rng.integers(0, 8 if target else 10))
    if mode == 0:
        return 0
    if mode == 1:
        return maximum
    if mode == 2:
        return min(maximum, int(rng.integers(1, 4)))
    if mode == 3:
        return max(0, maximum - int(rng.integers(1, 4)))
    if mode == 4:
        return maximum // 2
    return int(rng.integers(0, maximum + 1))


def _geometry(rng, mask, kind, target):
    if kind == "fringe":
        height, width = _size(rng, 12), _size(rng, 10)
        _, _, _, dy, dx = next(spec for spec in _CORNERS if mask & spec[0])
        col = 3 if dx < 0 else width - 5
        row = (int(rng.integers(5, height - 1)) if dy < 0
               else int(rng.integers(0, height - 5)))
    elif kind == "coverage":
        height, width = _size(rng, 10), _size(rng, 10)
        row = int(rng.integers(3, height - 4))
        col = int(rng.integers(3, width - 4))
    else:
        height, width = _size(rng, 6), _size(rng, 6)
        row = _axis(rng, height - 2, target)
        col = _axis(rng, width - 2, target)
    return height, width, row, col


def _draw(height, width, row, col, mask, background, marker, payload):
    source = np.full((height, width), background, dtype=np.int8)
    result = np.full((height, width), background, dtype=np.int8)
    result[row:row + 2, col:col + 2] = payload
    for bit, dr, dc, dy, dx in _CORNERS:
        source[row + dr, col + dc] = marker if mask & bit else payload
        if mask & bit:
            rr, cc = row + dr, col + dc
            while 0 <= rr < height:
                left, right = max(0, cc - 1), min(width, cc + 2)
                if left < right:
                    result[rr, left:right] = payload
                rr += dy
                cc += dx
    return source, result


def _input_features(grid):
    height, width = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    cells = np.argwhere(grid != background)
    assert cells.shape == (4, 2)
    row, col = int(cells[:, 0].min()), int(cells[:, 1].min())
    assert row + 1 < height and col + 1 < width
    assert np.all(grid[row:row + 2, col:col + 2] != background)
    assert len(np.unique(grid[row:row + 2, col:col + 2])) == 2
    return background, row, col


def _predict(grid, marker, stop_at_side=False):
    """Predict from a visible input and a marker learned from training."""
    background, row, col = _input_features(grid)
    tile = grid[row:row + 2, col:col + 2]
    assert marker != background and np.any(tile == marker)
    other = {int(value) for value in tile.ravel()} - {marker}
    assert len(other) == 1
    payload = other.pop()
    height, width = grid.shape
    output = np.full(grid.shape, background, dtype=np.int8)
    for rr in range(height):
        for cc in range(width):
            if row <= rr <= row + 1 and col <= cc <= col + 1:
                output[rr, cc] = payload
                continue
            for _, dr, dc, dy, dx in _CORNERS:
                mr, mc = row + dr, col + dc
                if int(grid[mr, mc]) != marker:
                    continue
                steps = (rr - mr) * dy
                if steps < 0:
                    continue
                center = mc + dx * steps
                if stop_at_side and not 0 <= center < width:
                    continue
                if abs(cc - center) <= 1:
                    output[rr, cc] = payload
                    break
    return output


def _features(source, result):
    background, row, col = _input_features(source)
    assert source.shape == result.shape
    payload = int(result[row, col])
    assert payload != background
    assert set(int(value) for value in np.unique(result)) == {background, payload}
    assert np.all(result[row:row + 2, col:col + 2] == payload)
    tile = source[row:row + 2, col:col + 2]
    assert payload in tile
    others = {int(value) for value in tile.ravel()} - {payload}
    assert len(others) == 1
    marker = others.pop()
    assert marker != background
    mask = sum(bit for bit, dr, dc, _, _ in _CORNERS
               if int(source[row + dr, col + dc]) == marker)
    assert 1 <= mask <= 14
    assert np.array_equal(result, _predict(source, marker))
    assert not np.array_equal(source, result)
    fringe = not np.array_equal(result, _predict(source, marker, True))
    return {"height": source.shape[0], "width": source.shape[1],
            "row": row, "col": col, "mask": mask, "background": background,
            "marker": marker, "payload": payload, "fringe": fringe}


def _reach(info, bit):
    _, dr, dc, dy, dx = next(spec for spec in _CORNERS if spec[0] == bit)
    rr, cc = info["row"] + dr, info["col"] + dc
    vertical = rr if dy < 0 else info["height"] - 1 - rr
    horizontal = cc if dx < 0 else info["width"] - 1 - cc
    return min(vertical, horizontal) + 1


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            source, result = pair["input"], pair["output"]
            key = _key(source)
            assert key not in seen
            seen.add(key)
            assert _rows(source) not in _SOURCE_INPUTS
            info[split].append(_features(source, result))
    train, test = info["train"], info["test"]
    markers = {item["marker"] for item in train}
    assert len(markers) == 1
    marker = next(iter(markers))
    assert all(item["marker"] == marker for item in test)
    assert np.bitwise_or.reduce(np.array([item["mask"] for item in train])) == 15
    assert any(item["fringe"] for item in train)
    for bit, _, _, _, _ in _CORNERS:
        assert any(item["mask"] & bit and _reach(item, bit) >= 3 for item in train)
    # This prediction sees only the learned train marker and each test input.
    for pair in puzzle["test"]:
        assert np.array_equal(
            _predict(pair["input"].copy(), marker), pair["output"])


def _masks(rng, train_count, test_count):
    single = 1 << int(rng.integers(0, 4))
    train = [single, 15 ^ single]
    if train_count >= 3:
        train.append(int(rng.choice([m for m in _MASKS if m not in train])))
    if train_count >= 4:
        train.append(single)
    while len(train) < train_count:
        unseen = [m for m in _MASKS if m not in train]
        pool = unseen if unseen and rng.random() < 0.75 else _MASKS
        train.append(int(rng.choice(pool)))
    test = []
    for _ in range(test_count):
        familiar = sorted(set(train))
        novel = [m for m in _MASKS if m not in train]
        pool = novel if novel and rng.random() < 0.65 else familiar
        test.append(int(rng.choice(pool)))
    return train, test


def _make_pair(rng, marker, mask, payload, kind, target):
    height, width, row, col = _geometry(rng, mask, kind, target)
    background = int(rng.choice(
        [value for value in range(10) if value not in (marker, payload)]))
    return _draw(height, width, row, col, mask, background, marker, payload)


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
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(0, 10))
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    train_masks, test_masks = _masks(rng, train_count, test_count)
    colors = [value for value in range(10) if value != marker]
    rng.shuffle(colors)
    train_payloads = []
    for index in range(train_count):
        if index == 0:
            payload = colors[0]
        elif index == 1:
            payload = colors[1]
        elif index == 2:
            payload = colors[0]
        elif index == 3:
            payload = colors[2]
        else:
            payload = int(rng.choice(colors[:5]))
        train_payloads.append(int(payload))
    familiar = sorted(set(train_payloads))
    unseen = [value for value in colors if value not in familiar]
    test_payloads = [int(rng.choice(
        unseen if unseen and rng.random() < 0.5 else familiar))
        for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, masks, payloads in (("train", train_masks, train_payloads),
                                   ("test", test_masks, test_payloads)):
        for index, (mask, payload) in enumerate(zip(masks, payloads)):
            kind = ("fringe" if split == "train" and index == 0 else
                    "coverage" if split == "train" and index == 1 else "free")
            for _ in range(40):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                source, result = _make_pair(
                    child, marker, mask, payload, kind, split == "test")
                key = _key(source)
                if key not in seen and _rows(source) not in _SOURCE_INPUTS:
                    break
            else:
                # Bounded deterministic fallback after unlikely collisions.
                height = 16 + index + (0 if split == "train" else 10)
                width = 30 - index - (0 if split == "train" else 4)
                row = col = 5
                if kind == "fringe":
                    _, _, _, dy, dx = next(
                        spec for spec in _CORNERS if mask & spec[0])
                    row = 6 if dy < 0 else 3
                    col = 3 if dx < 0 else width - 5
                background = next(value for value in range(10)
                                  if value not in (marker, payload))
                source, result = _draw(
                    height, width, row, col, mask, background, marker, payload)
                key = _key(source)
                assert key not in seen and _rows(source) not in _SOURCE_INPUTS
            seen.add(key)
            puzzle[split].append({"input": source, "output": result})
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(pa[key], pb[key])
        for pa, pb in zip(a[split], b[split])
        for key in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 3)):
        puzzle = generate_puzzle(1000 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1100 + (train or 0) + (test or 0), train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    assert _equal(generate_puzzle(1201, 7, 3), generate_puzzle(1201, 7, 3))
    assert not _equal(generate_puzzle(1202, 7, 3), generate_puzzle(1203, 7, 3))
    for arguments in ({"num_train": 1}, {"num_train": 11},
                      {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1301, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    all_info = []
    varying_sizes = varying_backgrounds = varying_masks = 0
    familiar_tests = novel_tests = 0
    for puzzle in puzzles:
        train = [_features(p["input"], p["output"]) for p in puzzle["train"]]
        test = [_features(p["input"], p["output"]) for p in puzzle["test"]]
        all_info.extend(train + test)
        varying_sizes += len({(i["height"], i["width"]) for i in train}) > 1
        varying_backgrounds += len({i["background"] for i in train}) > 1
        varying_masks += len({i["mask"] for i in train}) > 1
        train_masks = {i["mask"] for i in train}
        familiar_tests += sum(i["mask"] in train_masks for i in test)
        novel_tests += sum(i["mask"] not in train_masks for i in test)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert varying_sizes >= 38 and varying_backgrounds >= 30
    assert varying_masks == 42 and familiar_tests > 0 and novel_tests > 0
    assert {i["mask"] for i in all_info} == set(_MASKS)
    assert len({(i["height"], i["width"]) for i in all_info}) >= 70
    assert len({(i["height"], i["width"], i["row"], i["col"], i["mask"])
                for i in all_info}) >= 150
    assert min(i["height"] for i in all_info) <= 7
    assert max(i["height"] for i in all_info) == 30
    assert any(i["height"] != i["width"] for i in all_info)
    assert any(i["height"] == i["width"] == 9 for i in all_info)
    assert {i["background"] for i in all_info} == set(range(10))
    assert {i["marker"] for i in all_info} == set(range(10))
    assert {i["payload"] for i in all_info} == set(range(10))
