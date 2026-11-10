import numpy as np


# Exact original pairs are retained solely for regression and copy rejection.
_SOURCE_TEXT = (
    (
        "000000000000/000001101100/000001001000/024000000000/030001100000/000001000000/000000000000/000000000000/000000000000/000000000000",
        "000000000000/000002204400/000002004000/024000000000/030003300000/000003000000/000000000000/000000000000/000000000000/000000000000",
    ),
    (
        "110011000000000000/101010100000000000/011001100000002300/000000000000003030/110000001100000320/101000001010000000/011000000110000000/000000000000000000/000011001100000000/000010101010000000/000001100110000000/000000000000000000",
        "220033000000000000/202030300000000000/022003300000002300/000000000000003030/330000003300000320/303000003030000000/033000000330000000/000000000000000000/000033002200000000/000030302020000000/000003300220000000/000000000000000000",
    ),
    (
        "020000000000/474000000000/303000100000/000001110000/000001010000/000000000000/001000100010/011101110111/010101010101/000000000000/001000000010/011100000111/010100000101/000000000000",
        "020000000000/474000000000/303000200000/000002220000/000002020000/000000000000/004000700040/044407770444/040407070404/000000000000/003000000030/033300000333/030300000303/000000000000",
    ),
)
_SOURCE_SIGNATURES = {(tuple(a.split("/")), tuple(b.split("/"))) for a, b in _SOURCE_TEXT}
_DIMENSIONS = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3))
_BANK = {}


def _grid_signature(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _checked_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    groups = []
    while remaining:
        group = {next(iter(remaining))}
        remaining.difference_update(group)
        stack = list(group)
        while stack:
            r, c = stack.pop()
            for other in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if other in remaining:
                    remaining.remove(other)
                    group.add(other)
                    stack.append(other)
        groups.append(group)
    return groups


def _support_key(mask):
    return (mask.shape, tuple(bool(x) for x in mask.flat))


def _catalog():
    if _BANK:
        return _BANK
    for h, w in _DIMENSIONS:
        categories = {name: [] for name in ("simple", "disconnected", "ring", "full", "thin", "random")}
        for bits in range(1 << (h * w)):
            if bits.bit_count() < 3:
                continue
            mask = np.array([(bits >> i) & 1 for i in range(h * w)], dtype=bool).reshape(h, w)
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            groups = _components(mask)
            n = int(mask.sum())
            categories["random"].append(mask)
            if len(groups) == 1 and n <= 5 and h <= 3 and w <= 3:
                categories["simple"].append(mask)
            if len(groups) >= 2 and n >= 5 and min(map(len, groups)) >= 2:
                diagonal = any(
                    abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1
                    for i, group in enumerate(groups)
                    for other in groups[i + 1:]
                    for a in group for b in other
                )
                if diagonal:
                    categories["disconnected"].append(mask)
            if n == h * w:
                categories["full"].append(mask)
            if h >= 3 and w >= 3 and np.all(mask[0]) and np.all(mask[-1]) and np.all(mask[:, 0]) and np.all(mask[:, -1]) and not mask[1:-1, 1:-1].any():
                categories["ring"].append(mask)
            if min(h, w) == 2 or n <= max(h, w) + 1:
                categories["thin"].append(mask)
        _BANK[h, w] = categories
    return _BANK


def _sample_support(rng, category):
    catalog = _catalog()
    choices = [(h, w) for h, w in _DIMENSIONS if catalog[h, w][category]]
    h, w = choices[int(rng.integers(len(choices)))]
    bank = catalog[h, w][category]
    return bank[int(rng.integers(len(bank)))].copy()


def _choose_labels(rng, support, pool, distinct):
    points = np.argwhere(support)
    chosen = rng.choice(np.array(sorted(pool), dtype=np.int8), size=distinct, replace=False)
    values = np.empty(len(points), dtype=np.int8)
    values[:distinct] = chosen
    if len(points) > distinct:
        values[distinct:] = rng.choice(chosen, size=len(points) - distinct)
    rng.shuffle(values)
    labels = np.full(support.shape, -1, dtype=np.int8)
    for (r, c), color in zip(points, values):
        labels[int(r), int(c)] = color
    return labels


def _layout(rng, support):
    h, w = support.shape
    options = []
    for gy in range(1, 6):
        for gx in range(1, 6):
            eh = (h - 1) * (h + gy) + h
            ew = (w - 1) * (w + gx) + w
            for direction in ("seed_left", "seed_right", "seed_above", "seed_below"):
                if direction in ("seed_left", "seed_right"):
                    major, minor = w + 1 + ew, max(h, eh)
                else:
                    major, minor = h + 1 + eh, max(w, ew)
                if major <= 30 and minor <= 30:
                    options.append((gy, gx, direction))
    gy, gx, direction = options[int(rng.integers(len(options)))]
    eh = (h - 1) * (h + gy) + h
    ew = (w - 1) * (w + gx) + w
    horizontal = direction in ("seed_left", "seed_right")
    major_base = w + ew if horizontal else h + eh
    gap = int(rng.integers(1, min(6, 30 - major_base) + 1))
    major_size = major_base + gap
    leading = int(rng.integers(0, min(5, 30 - major_size) + 1))
    trailing = int(rng.integers(0, min(5, 30 - major_size - leading) + 1))
    major_size += leading + trailing
    minor_base = max(h, eh) if horizontal else max(w, ew)
    minor_slack = int(rng.integers(0, min(10, 30 - minor_base) + 1))
    minor_size = minor_base + minor_slack
    expansion_minor = int(rng.integers(0, minor_size - (eh if horizontal else ew) + 1))
    seed_extent = h if horizontal else w
    expansion_extent = eh if horizontal else ew
    seed_low = max(0, expansion_minor - seed_extent + 1)
    seed_high = min(minor_size - seed_extent, expansion_minor + expansion_extent - 1)
    seed_minor = int(rng.integers(seed_low, seed_high + 1))
    if direction == "seed_left":
        seed_top, seed_left = seed_minor, leading
        expansion_top, expansion_left = expansion_minor, leading + w + gap
    elif direction == "seed_right":
        expansion_top, expansion_left = expansion_minor, leading
        seed_top, seed_left = seed_minor, leading + ew + gap
    elif direction == "seed_above":
        seed_top, seed_left = leading, seed_minor
        expansion_top, expansion_left = leading + h + gap, expansion_minor
    else:
        expansion_top, expansion_left = leading, expansion_minor
        seed_top, seed_left = leading + eh + gap, seed_minor
    canvas = (minor_size, major_size) if horizontal else (major_size, minor_size)
    return canvas, (seed_top, seed_left), (expansion_top, expansion_left), (gy, gx)


def _make_pair(rng, support, label_pool, distinct):
    labels = _choose_labels(rng, support, label_pool, distinct)
    used = set(int(x) for x in labels[support])
    roles = rng.choice(np.array(sorted(set(range(10)) - used), dtype=np.int8), size=2, replace=False)
    background, carrier = map(int, roles)
    canvas, seed_origin, expansion_origin, (gy, gx) = _layout(rng, support)
    input_grid = np.full(canvas, background, dtype=np.int8)
    output_grid = input_grid.copy()
    sh, sw = support.shape
    points = [tuple(map(int, point)) for point in np.argwhere(support)]
    for r, c in points:
        rr, cc = seed_origin[0] + r, seed_origin[1] + c
        input_grid[rr, cc] = output_grid[rr, cc] = labels[r, c]
    for mr, mc in points:
        for r, c in points:
            rr = expansion_origin[0] + mr * (sh + gy) + r
            cc = expansion_origin[1] + mc * (sw + gx) + c
            input_grid[rr, cc] = carrier
            output_grid[rr, cc] = labels[mr, mc]
    return {"input": input_grid, "output": output_grid}


def _recover(input_grid, carrier):
    colors = np.bincount(input_grid.ravel().astype(int), minlength=10)
    other_colors = colors.copy()
    other_colors[carrier] = -1
    background = int(np.argmax(other_colors))
    seed_mask = (input_grid != background) & (input_grid != carrier)
    seed_points = np.argwhere(seed_mask)
    carrier_points = np.argwhere(input_grid == carrier)
    if len(seed_points) < 3 or len(carrier_points) != len(seed_points) ** 2:
        return None
    sr0, sc0 = seed_points.min(axis=0)
    sr1, sc1 = seed_points.max(axis=0)
    er0, ec0 = carrier_points.min(axis=0)
    er1, ec1 = carrier_points.max(axis=0)
    sh, sw = int(sr1 - sr0 + 1), int(sc1 - sc0 + 1)
    if (sh, sw) not in _DIMENSIONS:
        return None
    support = seed_mask[sr0:sr1 + 1, sc0:sc1 + 1]
    labels = input_grid[sr0:sr1 + 1, sc0:sc1 + 1]
    if len(seed_points) != int(support.sum()):
        return None
    dy, dx = int(er1 - er0 + 1 - sh), int(ec1 - ec0 + 1 - sw)
    if dy % (sh - 1) or dx % (sw - 1):
        return None
    py, px = dy // (sh - 1), dx // (sw - 1)
    gy, gx = py - sh, px - sw
    if not (1 <= gy <= 5 and 1 <= gx <= 5):
        return None
    expected = set()
    points = [tuple(map(int, p)) for p in np.argwhere(support)]
    predicted = input_grid.copy()
    for mr, mc in points:
        for r, c in points:
            rr, cc = int(er0) + mr * py + r, int(ec0) + mc * px + c
            expected.add((rr, cc))
            predicted[rr, cc] = labels[mr, mc]
    actual = {tuple(map(int, p)) for p in carrier_points}
    if expected != actual or len(expected) != len(points) ** 2:
        return None
    if sc1 + 1 < ec0:
        placement = "seed_left"
    elif ec1 + 1 < sc0:
        placement = "seed_right"
    elif sr1 + 1 < er0:
        placement = "seed_above"
    elif er1 + 1 < sr0:
        placement = "seed_below"
    else:
        # The original test has overlapping object bounding rectangles;
        # their differently colored occupied cells remain separate.
        placement = "offset"
    margins = (int(min(sr0, er0)), int(min(sc0, ec0)),
               int(input_grid.shape[0] - 1 - max(sr1, er1)),
               int(input_grid.shape[1] - 1 - max(sc1, ec1)))
    return predicted, {
        "background": background, "carrier": carrier, "support": support,
        "support_key": _support_key(support), "height": sh, "width": sw,
        "count": len(points), "labels": frozenset(int(x) for x in labels[support]),
        "gutter": (gy, gx), "placement": placement, "margins": margins,
        "shape": input_grid.shape, "components": len(_components(support)),
        "ring": bool(sh >= 3 and sw >= 3 and np.all(support[0]) and np.all(support[-1])
                     and np.all(support[:, 0]) and np.all(support[:, -1])
                     and not support[1:-1, 1:-1].any()),
        "full": bool(support.all()),
    }


def _solve_visible(input_grid):
    candidates = []
    for carrier in np.unique(input_grid):
        found = _recover(input_grid, int(carrier))
        if found is not None:
            candidates.append(found)
    assert len(candidates) == 1
    return candidates[0]


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    changed = input_grid != output_grid
    assert changed.any()
    carriers = np.unique(input_grid[changed])
    assert len(carriers) == 1
    carrier = int(carriers[0])
    assert np.array_equal(changed, input_grid == carrier)
    recovered = _recover(input_grid, carrier)
    assert recovered is not None
    predicted, info = recovered
    assert np.array_equal(predicted, output_grid)
    visible, visible_info = _solve_visible(input_grid)
    assert np.array_equal(visible, output_grid)
    assert visible_info["support_key"] == info["support_key"]
    return info


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_train = int(rng.choice(np.arange(2, 11), p=np.array([3, 3, 2, 2, 2, 1, 1, 1, 1]) / 16))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # The rule is the sole shared convention. This pool leaves room for tests
    # to introduce multiple unseen literal label colors.
    familiar = set(map(int, rng.choice(10, size=int(rng.integers(5, 8)), replace=False)))
    puzzle = {"train": [], "test": []}
    supports = []
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(120):
                if split == "train" and index == 0:
                    category = "simple"
                elif split == "train" and index == 1:
                    category = "disconnected"
                else:
                    categories = ("disconnected", "ring", "full", "thin", "random")
                    weights = (25, 15, 12, 20, 28) if split == "test" else (18, 10, 10, 23, 39)
                    category = categories[int(rng.choice(5, p=np.array(weights) / 100))]
                if split == "test" and supports and rng.random() < 0.3:
                    support = supports[int(rng.integers(len(supports)))].copy()
                else:
                    support = _sample_support(rng, category)
                n = int(support.sum())
                if split == "train":
                    max_labels = min(n, len(familiar), 7)
                    min_labels = 3 if index == 0 else 1
                    distinct = int(rng.integers(min_labels, max_labels + 1))
                    pool = familiar
                else:
                    train_colors = set().union(*(
                        _analyze_pair(p["input"], p["output"])["labels"] for p in puzzle["train"]
                    ))
                    new_colors = set(range(10)) - train_colors
                    mode = int(rng.integers(4))
                    if mode == 0:
                        pool = train_colors
                        distinct = int(rng.integers(1, min(n, len(pool), 8) + 1))
                    elif mode == 1:
                        pool = new_colors
                        distinct = int(rng.integers(1, min(n, len(pool), 8) + 1))
                    elif mode == 2:
                        fresh = set(map(int, rng.choice(sorted(new_colors), size=min(len(new_colors), int(rng.integers(2, 4))), replace=False)))
                        known = set(map(int, rng.choice(sorted(train_colors), size=min(len(train_colors), int(rng.integers(1, 4))), replace=False)))
                        pool = fresh | known
                        distinct = len(pool)
                        if distinct > min(n, 8):
                            continue
                    else:
                        pool = set(range(10))
                        if n >= 8 and rng.random() < 0.6:
                            distinct = 8
                        else:
                            distinct = int(rng.integers(min(5, n, 8), min(n, 8) + 1))
                pair = _make_pair(rng, support, pool, distinct)
                signature = (_grid_signature(pair["input"]), _grid_signature(pair["output"]))
                if signature in seen or signature in _SOURCE_SIGNATURES:
                    continue
                try:
                    _analyze_pair(pair["input"], pair["output"])
                except AssertionError:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                if split == "train":
                    supports.append(support)
                break
            else:
                raise ValueError("could not construct an unambiguous distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            input_grid, output_grid = pair["input"], pair["output"]
            assert input_grid.shape == output_grid.shape
            signature = (_grid_signature(input_grid), _grid_signature(output_grid))
            assert signature[0] not in inputs and signature not in pairs
            assert signature not in _SOURCE_SIGNATURES
            inputs.add(signature[0])
            pairs.add(signature)
            info[split].append(_analyze_pair(input_grid, output_grid))
    assert info["train"][0]["components"] == 1
    assert info["train"][1]["components"] >= 2
    for item in info["train"] + info["test"]:
        assert 3 <= item["count"] <= 12
        assert 1 <= len(item["labels"]) <= min(item["count"], 8)


def _puzzle_signature(puzzle):
    return tuple(tuple((_grid_signature(p["input"]), _grid_signature(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    for source_input, source_output in _SOURCE_TEXT:
        input_grid = np.array([[int(x) for x in row] for row in source_input.split("/")], dtype=np.int8)
        output_grid = np.array([[int(x) for x in row] for row in source_output.split("/")], dtype=np.int8)
        _analyze_pair(input_grid, output_grid)
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert _puzzle_signature(generate_puzzle(17)) == _puzzle_signature(generate_puzzle(17))
    assert len({_puzzle_signature(p) for p in automatic}) == 42
    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(10000 + index, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for index, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(11000 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(12000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    all_info = [_analyze_pair(pair["input"], pair["output"])
                for puzzle in automatic for split in ("train", "test") for pair in puzzle[split]]
    assert {x["placement"] for x in all_info} == {"seed_left", "seed_right", "seed_above", "seed_below"}
    assert {x["gutter"][0] for x in all_info} >= {1, 2, 3}
    assert {x["gutter"][1] for x in all_info} >= {1, 2, 3}
    assert any(x["ring"] for x in all_info) and any(x["full"] for x in all_info)
    assert any(x["components"] > 1 for x in all_info)
    assert len({x["support_key"] for x in all_info}) >= 35
    assert len({x["shape"] for x in all_info}) >= 30
    assert {(x["height"], x["width"]) for x in all_info} == set(_DIMENSIONS)
    assert {len(x["labels"]) for x in all_info} == set(range(1, 9))
    assert max(x["count"] for x in all_info) >= 10
    assert any(0 in x["margins"] for x in all_info)
    assert sum(len(p["train"]) <= 3 for p in automatic) >= 5
    train_infos = [[_analyze_pair(pair["input"], pair["output"])
                    for pair in puzzle["train"]] for puzzle in automatic]
    test_infos = [[_analyze_pair(pair["input"], pair["output"])
                   for pair in puzzle["test"]] for puzzle in automatic]
    for group in (train_infos, test_infos):
        flat = [item for puzzle in group for item in puzzle]
        assert any(x["components"] > 1 and (x["height"], x["width"], x["count"]) == (3, 3, 6)
                   for x in flat)
        assert any(x["ring"] for x in flat) and any(x["full"] for x in flat)
        assert any(x["components"] > 1 and max(x["gutter"]) >= 3
                   and len(x["labels"]) >= 4 for x in flat)
    assert all(len({x["background"] for x in train + test}) > 1
               and len({x["carrier"] for x in train + test}) > 1
               for train, test in zip(train_infos, test_infos))
    assert sum(bool({x["placement"] for x in train} &
                    {x["placement"] for x in test})
               for train, test in zip(train_infos, test_infos)) >= 10
    assert sum(bool({x["placement"] for x in train} & {"seed_left", "seed_right"})
               and bool({x["placement"] for x in train} & {"seed_above", "seed_below"})
               for train in train_infos) >= 10
    assert any(x["support_key"] in {y["support_key"] for y in train}
               for train, test in zip(train_infos, test_infos) for x in test)
    assert any(x["support_key"] not in {y["support_key"] for y in train}
               for train, test in zip(train_infos, test_infos) for x in test)
    assert any(not (x["labels"] - set().union(*(y["labels"] for y in train)))
               for train, test in zip(train_infos, test_infos) for x in test)
    assert any(len(x["labels"] - set().union(*(y["labels"] for y in train))) >= 2
               for train, test in zip(train_infos, test_infos) for x in test)
