import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "1100",
            "1010",
            "1101",
            "1111",
            "4444",
            "2202",
            "0020",
            "0200",
            "2020",
        ),
        (
            "3303",
            "3030",
            "3303",
            "3333",
        ),
    ),
    (
        (
            "1110",
            "0101",
            "0011",
            "1101",
            "4444",
            "0002",
            "0002",
            "2222",
            "2202",
        ),
        (
            "3333",
            "0303",
            "3333",
            "3303",
        ),
    ),
    (
        (
            "1010",
            "1101",
            "1011",
            "0101",
            "4444",
            "2200",
            "0020",
            "2200",
            "0020",
        ),
        (
            "3330",
            "3333",
            "3333",
            "0333",
        ),
    ),
    (
        (
            "1100",
            "0101",
            "0100",
            "1010",
            "4444",
            "2222",
            "0022",
            "2200",
            "0022",
        ),
        (
            "3333",
            "0333",
            "3300",
            "3033",
        ),
    ),
    (
        (
            "1010",
            "1010",
            "0100",
            "1010",
            "4444",
            "2200",
            "0020",
            "0202",
            "2220",
        ),
        (
            "3330",
            "3030",
            "0303",
            "3330",
        ),
    ),
)


def _rows_key(rows):
    return (
        (len(rows), len(rows[0])),
        tuple(int(character) for row in rows for character in row),
    )


_SOURCE_PAIR_KEYS = frozenset(
    (_rows_key(input_rows), _rows_key(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _grid_key(grid):
    return (
        tuple(int(value) for value in grid.shape),
        tuple(int(value) for value in grid.reshape(-1)),
    )


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _puzzle_key(puzzle):
    return (
        tuple(_pair_key(pair) for pair in puzzle["train"]),
        tuple(_pair_key(pair) for pair in puzzle["test"]),
    )


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _normalize_count(value, name, minimum, maximum):
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")

    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _normalize_count(num_train, "num_train", 2, 10)
    num_test = _normalize_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        # This deterministic schedule guarantees genuine count diversity across
        # consecutive seeds while always producing a legal combination.
        num_test = 1 + seed % 4
        number_of_train_choices = 10 - num_test
        num_train = (
            num_test
            + 1
            + (seed // 4) % number_of_train_choices
        )
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        largest_test_count = min(4, num_train - 1)
        num_test = int(rng.integers(1, largest_test_count + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _size(rng, orientation):
    roll = rng.random()
    if roll < 0.12:
        return 4, 4
    if roll < 0.30:
        return int(rng.integers(2, 5)), int(rng.integers(2, 5))
    if roll < 0.48:
        if orientation == "vertical":
            return int(rng.integers(5, 13)), int(rng.integers(9, 19))
        return int(rng.integers(9, 19)), int(rng.integers(5, 13))
    if roll < 0.58:
        if orientation == "vertical":
            return int(rng.integers(11, 15)), int(rng.integers(3, 11))
        return int(rng.integers(3, 11)), int(rng.integers(11, 15))
    return int(rng.integers(3, 11)), int(rng.integers(3, 11))


def _palette(rng, result, previous_background=None):
    choices = [color for color in range(10) if color != result]
    backgrounds = [color for color in choices if color != previous_background]
    background = int(rng.choice(backgrounds))
    rest = rng.choice([x for x in choices if x != background], size=3,
                      replace=False)
    return (background,) + tuple(int(x) for x in rest)


def _shape(rng, height, width):
    area = height * width
    mask = np.zeros((height, width), dtype=bool)
    kind = int(rng.integers(0, 7))
    if kind == 0:  # irregular cells
        return rng.random((height, width)) < rng.uniform(0.15, 0.57)
    if kind == 1:  # sparse marks
        count = int(rng.integers(1, max(3, area // 5 + 1)))
        mask.flat[rng.choice(area, size=min(count, area - 1), replace=False)] = True
    elif kind in (2, 6):  # patches, including edge-touching ones
        top = int(rng.integers(0, height))
        left = int(rng.integers(0, width))
        bottom = int(rng.integers(top + 1, height + 1))
        right = int(rng.integers(left + 1, width + 1))
        if kind == 6:
            side = int(rng.integers(0, 4))
            if side == 0:
                top = 0
            elif side == 1:
                bottom = height
            elif side == 2:
                left = 0
            else:
                right = width
        mask[top:bottom, left:right] = True
    elif kind == 3:  # strokes
        if rng.random() < 0.5:
            row = int(rng.integers(0, height))
            left = int(rng.integers(0, width))
            right = int(rng.integers(left + 1, width + 1))
            mask[row, left:right] = True
        else:
            col = int(rng.integers(0, width))
            top = int(rng.integers(0, height))
            bottom = int(rng.integers(top + 1, height + 1))
            mask[top:bottom, col] = True
    elif kind == 4 and height >= 3 and width >= 3:  # hollow border
        top = int(rng.integers(0, height - 2))
        left = int(rng.integers(0, width - 2))
        bottom = int(rng.integers(top + 3, height + 1))
        right = int(rng.integers(left + 3, width + 1))
        mask[top:bottom, left] = True
        mask[top:bottom, right - 1] = True
        mask[top, left:right] = True
        mask[bottom - 1, left:right] = True
    elif kind == 5 and height >= 4 and width >= 4:  # interior patch
        top = int(rng.integers(1, height - 1))
        left = int(rng.integers(1, width - 1))
        bottom = int(rng.integers(top + 1, height))
        right = int(rng.integers(left + 1, width))
        mask[top:bottom, left:right] = True
    else:
        mask.flat[int(rng.integers(0, area))] = True
    return mask


def _masks(rng, height, width, relation):
    area = height * width
    for _ in range(24):
        first = _shape(rng, height, width)
        second = _shape(rng, height, width)
        if relation == "equal":
            second = first.copy()
        elif relation == "contain":
            second &= first
            if not second.any() or np.array_equal(first, second):
                points = np.flatnonzero(first)
                if len(points) >= 2:
                    second = np.zeros_like(first)
                    second.flat[points[: max(1, len(points) // 2)]] = True
        elif relation == "disjoint":
            second &= ~first
        elif relation == "full":
            second |= ~first
        elif relation == "heavy":
            second |= first & (rng.random((height, width)) < 0.75)
        states = first.astype(int) + 2 * second.astype(int)
        anchor_good = (np.count_nonzero(states == 1) >= max(2, int(np.ceil(0.1 * area)))
                       and np.count_nonzero(states == 2) >= max(2, int(np.ceil(0.1 * area))))
        if (0 < first.sum() < area and 0 < second.sum() < area
                and (relation not in ("all_four", "anchor") or len(np.unique(states)) == 4)
                and (relation != "anchor" or anchor_good)):
            return first, second
    # Bounded fallback for small grids and rare geometric combinations.
    states = rng.integers(0, 4, size=area, dtype=np.int8)
    positions = rng.permutation(area)
    if relation == "anchor":
        quota = min(max(2, int(np.ceil(0.1 * area))), (area - 2) // 2)
        states[:] = 0
        states[positions[:quota]] = 1
        states[positions[quota:2 * quota]] = 2
        states[positions[2 * quota]] = 3
    elif relation == "equal":
        states[:] = 0
        states[positions[: max(1, area // 3)]] = 3
    elif relation == "contain":
        states[:] = 0
        states[positions[: max(2, area // 2)]] = 1
        states[positions[: max(1, area // 4)]] = 3
    elif relation == "disjoint":
        states[:] = 0
        states[positions[: max(1, area // 3)]] = 1
        states[positions[max(1, area // 3):max(2, 2 * area // 3)]] = 2
    elif relation == "full":
        states[:] = 1
        states[positions[: max(1, area // 3)]] = 2
        states[positions[max(1, area // 3):max(2, area // 2)]] = 3
    else:
        states[positions[:4]] = np.arange(4)
    states = states.reshape(height, width)
    return (states == 1) | (states == 3), (states == 2) | (states == 3)


def _pair(rng, orientation, height, width, palette, result, relation):
    first, second = _masks(rng, height, width, relation)
    background, first_color, second_color, divider = palette
    a = np.where(first, first_color, background).astype(np.int8)
    b = np.where(second, second_color, background).astype(np.int8)
    if orientation == "vertical":
        inp = np.vstack((a, np.full((1, width), divider, np.int8), b))
    else:
        inp = np.hstack((a, np.full((height, 1), divider, np.int8), b))
    out = np.where(first | second, result, background).astype(np.int8)
    return {"input": inp, "output": out}


def _read_input(inp):
    """Recover local roles from only a visible input grid."""
    candidates = []
    rows, cols = inp.shape
    for orientation in ("vertical", "horizontal"):
        if orientation == "vertical":
            if rows < 5 or rows % 2 == 0:
                continue
            middle = rows // 2
            line = inp[middle, :]
            a, b = inp[:middle, :], inp[middle + 1:, :]
        else:
            if cols < 5 or cols % 2 == 0:
                continue
            middle = cols // 2
            line = inp[:, middle]
            a, b = inp[:, :middle], inp[:, middle + 1:]
        divider = int(line[0])
        if not np.all(line == divider) or np.count_nonzero(inp == divider) != line.size:
            continue
        colors_a = set(int(x) for x in np.unique(a))
        colors_b = set(int(x) for x in np.unique(b))
        common = colors_a & colors_b
        if len(colors_a) != 2 or len(colors_b) != 2 or len(common) != 1:
            continue
        background = next(iter(common))
        first_color = next(iter(colors_a - common))
        second_color = next(iter(colors_b - common))
        if len({background, first_color, second_color, divider}) != 4:
            continue
        candidates.append((orientation, background, first_color, second_color,
                           divider, a == first_color, b == second_color))
    assert len(candidates) == 1, "input must have one identifiable aligned divider"
    return candidates[0]


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= x <= 30 for x in grid.shape)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def _predict_from_training(train, test_input):
    """Infer the output from visible demonstrations and a new input."""
    results = set()
    for pair in train:
        _, bg, _, _, _, _, _ = _read_input(pair["input"])
        results.update(int(x) for x in np.unique(pair["output"]) if x != bg)
    assert len(results) == 1
    result = next(iter(results))
    _, bg, _, _, _, a, b = _read_input(test_input)
    assert bg != result
    return np.where(a | b, result, bg).astype(np.int8)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    result_colors = set()
    train_states = set()
    train_orientations = set()
    train_backgrounds = set()
    substantial_train = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            input_key = _grid_key(inp)
            assert input_key not in seen, "duplicate input"
            seen.add(input_key)
            assert _pair_key(pair) not in _SOURCE_PAIR_KEYS, "source copy"
            orientation, bg, fg1, fg2, divider, a, b = _read_input(inp)
            assert out.shape == a.shape == b.shape
            assert not np.array_equal(inp, out)
            assert a.any() and (~a).any() and b.any() and (~b).any()
            states = a.astype(np.int8) + 2 * b.astype(np.int8)
            if section == "train":
                train_states.update(int(x) for x in np.unique(states))
                train_orientations.add(orientation)
                train_backgrounds.add(bg)
                substantial_train |= (np.count_nonzero(states == 1) >= 0.1 * states.size
                                      and np.count_nonzero(states == 2) >= 0.1 * states.size)
                colors = set(int(x) for x in np.unique(out)) - {bg}
                assert len(colors) == 1, "training must show the result color"
                result_colors.update(colors)
            else:
                assert orientation in train_orientations, "test layout unseen in training"
            assert not (result_colors & {bg, fg1, fg2, divider})
    assert train_states == {0, 1, 2, 3}
    assert len(result_colors) == 1, "result color must be stable"
    assert len(train_backgrounds) >= 2, "training must show local palette changes"
    assert substantial_train, "training must show both masks contributing materially"
    result = next(iter(result_colors))
    for pair in puzzle["train"] + puzzle["test"]:
        _, bg, fg1, fg2, divider, a, b = _read_input(pair["input"])
        assert result not in {bg, fg1, fg2, divider}
        assert np.array_equal(pair["output"], np.where(a | b, result, bg))
    visible_train = [{"input": p["input"].copy(), "output": p["output"].copy()}
                     for p in puzzle["train"]]
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_training(visible_train, pair["input"].copy()),
                              pair["output"]), "test not inferable from training"


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    result = int(rng.integers(0, 10))  # one learned puzzle-level convention
    style = rng.choice(("source", "fixed", "mixed"), p=(0.12, 0.38, 0.50))
    fixed_orientation = rng.choice(("vertical", "horizontal"))
    mixed_first = rng.choice(("vertical", "horizontal"))
    puzzle = {"train": [], "test": []}
    used = set()
    previous_background = None
    first_size = None
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _attempt in range(48):
                if style == "source":
                    orientation = "vertical"
                    height, width = 4, 4
                else:
                    if style == "fixed":
                        orientation = fixed_orientation
                    elif section == "train" and index < 2:
                        orientation = mixed_first if index == 0 else (
                            "horizontal" if mixed_first == "vertical" else "vertical")
                    else:
                        orientation = pair_rng.choice(("vertical", "horizontal"))
                    height, width = _size(pair_rng, orientation)
                    if section == "train" and index == 1 and (height, width) == first_size:
                        continue
                palette = _palette(pair_rng, result,
                                   previous_background if section == "train" else None)
                if section == "train" and index == 0:
                    relation = "anchor"
                else:
                    choices = ("all_four", "independent", "equal", "contain",
                               "disjoint", "full", "heavy")
                    probs = ((0.36, 0.18, 0.08, 0.12, 0.10, 0.06, 0.10)
                             if section == "train" else
                             (0.40, 0.17, 0.08, 0.10, 0.10, 0.07, 0.08))
                    relation = pair_rng.choice(choices, p=probs)
                pair = _pair(pair_rng, orientation, height, width, palette, result, relation)
                key = _grid_key(pair["input"])
                if key in used or _pair_key(pair) in _SOURCE_PAIR_KEYS:
                    continue
                try:
                    _read_input(pair["input"])
                except AssertionError:
                    continue
                puzzle[section].append(pair)
                used.add(key)
                if section == "train":
                    previous_background = palette[0]
                    if index == 0:
                        first_size = (height, width)
                break
            else:
                raise ValueError("unable to construct distinct unambiguous pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    signatures = set()
    sizes = set()
    orientations = set()
    results = set()
    test_state_counts = set()
    input_shapes = set()
    input_palettes = set()
    relation_counts = {name: 0 for name in ("equal", "contain", "disjoint", "full")}
    motifs = {name: 0 for name in ("sparse", "rectangle", "stroke", "hollow",
                                      "edge", "interior")}
    small = large = wide = solid_tests = dense_outputs = 0
    heavy_overlap = 0
    overlap_locations = set()
    palettes_in_training = 0
    all_role_changes = 0
    combined = 0
    mixed = 0
    varying_sizes = 0
    source_like = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_grid_key(p["input"]) for p in puzzle["train"] + puzzle["test"]))
        local_sizes = set()
        local_orientations = set()
        train_roles = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                orientation, bg, fg1, fg2, divider, a, b = _read_input(pair["input"])
                local_sizes.add(a.shape)
                local_orientations.add(orientation)
                sizes.add(a.shape)
                orientations.add(orientation)
                input_shapes.add(pair["input"].shape)
                input_palettes.add((bg, fg1, fg2, divider))
                height, width = a.shape
                small += height <= 3 or width <= 3
                large += height >= 11 or width >= 11
                wide += max(height, width) >= 2 * min(height, width)
                dense_outputs += np.mean(a | b) > 0.8
                relation_counts["equal"] += np.array_equal(a, b)
                relation_counts["contain"] += (np.all(a <= b) or np.all(b <= a))
                relation_counts["disjoint"] += not np.any(a & b)
                relation_counts["full"] += np.all(a | b)
                overlap = a & b
                heavy_overlap += (np.count_nonzero(overlap) >=
                                  0.5 * min(np.count_nonzero(a), np.count_nonzero(b)))
                if overlap.any():
                    yy, xx = np.where(overlap)
                    overlap_locations.add((int(2 * np.mean(yy) >= height),
                                           int(2 * np.mean(xx) >= width)))
                for mask in (a, b):
                    yy, xx = np.where(mask)
                    top, bottom = int(yy.min()), int(yy.max()) + 1
                    left, right = int(xx.min()), int(xx.max()) + 1
                    crop = mask[top:bottom, left:right]
                    motifs["sparse"] += np.mean(mask) <= 0.15
                    motifs["rectangle"] += crop.size >= 4 and bool(crop.all())
                    motifs["stroke"] += ((bottom - top == 1 or right - left == 1)
                                         and crop.size >= 3)
                    motifs["hollow"] += (bottom - top >= 3 and right - left >= 3
                                         and crop[0].all() and crop[-1].all()
                                         and crop[:, 0].all() and crop[:, -1].all()
                                         and not crop[1:-1, 1:-1].any())
                    edge = (mask[0].any() or mask[-1].any()
                            or mask[:, 0].any() or mask[:, -1].any())
                    motifs["edge"] += bool(edge)
                    motifs["interior"] += not edge
                if section == "test":
                    test_state_counts.add(len(np.unique(a.astype(int) + 2 * b.astype(int))))
                    solid_tests += bool(np.all(a | b))
                else:
                    results.update(int(x) for x in np.unique(pair["output"]) if x != bg)
                    train_roles.append((bg, fg1, fg2, divider))
        palettes_in_training += len(set(train_roles)) >= 2
        all_role_changes += all(len({role[index] for role in train_roles}) >= 2
                                for index in range(4))
        combined += len(local_orientations) == 2 and len(local_sizes) >= 2 and len(set(train_roles)) >= 2
        mixed += len(local_orientations) == 2
        varying_sizes += len(local_sizes) >= 2
        source_like += local_sizes == {(4, 4)} and local_orientations == {"vertical"}
    assert len(signatures) == 42
    assert len(sizes) >= 30 and orientations == {"vertical", "horizontal"}
    assert len(results) >= 8
    assert mixed >= 10 and varying_sizes >= 25 and source_like >= 1
    assert {2, 3, 4} <= test_state_counts
    assert len(input_shapes) >= 30 and len(input_palettes) >= 100
    assert palettes_in_training == 42 and all_role_changes >= 20 and combined >= 10
    assert small >= 10 and large >= 10 and wide >= 10
    assert all(relation_counts[name] >= 2 for name in relation_counts)
    assert heavy_overlap >= 10 and len(overlap_locations) == 4
    assert solid_tests >= 1 and dense_outputs < 0.25 * sum(
        len(p["train"]) + len(p["test"]) for p in puzzles)
    assert all(motifs[name] >= 5 for name in motifs)
    for seed, ntrain, ntest in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest)
    for seed, ntrain, ntest in ((2001, None, 1), (2002, None, 4),
                               (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 4}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request was accepted")
    first = generate_puzzle(314159, 7, 3)
    second = generate_puzzle(314159, 7, 3)
    assert all(_grid_key(a[k]) == _grid_key(b[k]) for split in ("train", "test")
               for a, b in zip(first[split], second[split]) for k in ("input", "output"))
    auto_first = generate_puzzle(271828)
    auto_second = generate_puzzle(271828)
    assert all(_grid_key(a[k]) == _grid_key(b[k]) for split in ("train", "test")
               for a, b in zip(auto_first[split], auto_second[split])
               for k in ("input", "output"))
    changed = {"train": [dict(p) for p in first["train"]],
               "test": [dict(p) for p in first["test"]]}
    changed["test"][0]["output"] = changed["test"][0]["output"].copy()
    changed["test"][0]["output"][0, 0] = (int(changed["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(changed)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted output")
