"""Generators for the compartment frequency puzzle."""

import numpy as np


_SOURCE = (
    (
        ("00050205200", "20050005002", "00050005000", "55555555555", "20050005000", "20050025000", "00050005020", "55555555555", "00050005000", "20050025002", "00050005000"),
        ("00050005222", "00050005222", "00050005222", "55555555555", "22250005000", "22250005000", "22250005000", "55555555555", "00050005000", "00050005000", "00050005000"),
    ),
    (
        ("00050005000", "10050005010", "00050015000", "55555555555", "00050015000", "00050005010", "00050005000", "55555555555", "00050005100", "01050005001", "00050005000"),
        ("00050005000", "00050005000", "00050005000", "55555555555", "00050005000", "00050005000", "00050005000", "55555555555", "00050005111", "00050005111", "00050005111"),
    ),
    (
        ("33050005000", "00050005030", "00050005000", "55555555555", "00050005000", "03050305000", "00053005000", "55555555555", "00050005000", "03053005330", "00050005003"),
        ("00050005000", "00050005000", "00050005000", "55555555555", "00050005000", "00050005000", "00050005000", "55555555555", "00050005333", "00050005333", "00050005333"),
    ),
    (
        ("44050005000", "04050045440", "40050005040", "55555555555", "00050005040", "40050405404", "00050005000", "55555555555", "00050405004", "40050045040", "00054405000"),
        ("44450005000", "44450005000", "44450005000", "55555555555", "00050005000", "00050005000", "00050005000", "55555555555", "00054445000", "00054445000", "00054445000"),
    ),
)


def _checked_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _checked_optional_count(num_train, "num_train", 2, 10)
    num_test = _checked_optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        first = max(2, num_test + 1)
        num_train = first + (seed // 4) % (11 - first)
    elif num_train is None:
        first = max(2, num_test + 1)
        num_train = first + seed % (11 - first)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


_GEOMETRIES = tuple(
    (rows, cols, height, width)
    for rows in range(2, 8)
    for cols in range(2, 8)
    for height in range(2, 9)
    for width in range(2, 9)
    if rows * height + rows - 1 <= 30
    and cols * width + cols - 1 <= 30
)


def _sample_geometry(rng, exclude=None, minimum_area=4):
    choices = [g for g in _GEOMETRIES if g != exclude and g[2] * g[3] >= minimum_area]
    return choices[int(rng.integers(len(choices)))]


def _sample_maximum(rng, capacity, low=1):
    assert 1 <= low <= capacity
    if low == capacity:
        return low
    branch = int(rng.integers(4))
    if branch == 0:
        return int(rng.integers(low, min(capacity, low + 2) + 1))
    if branch == 1:
        return int(rng.integers(max(low, capacity // 2), capacity + 1))
    return int(rng.integers(low, capacity + 1))


def _sample_winner_count(rng, number, maximum, area, regime):
    upper = number if maximum < area else number - 1
    if regime == "target_test":
        mode = int(rng.integers(5))
        if mode == 0:
            return 1
        if mode == 1:
            return upper
    return int(rng.integers(1, upper + 1))


def _loser_value(rng, maximum):
    if maximum == 1:
        return 0
    mode = int(rng.integers(5))
    if mode == 0:
        return 0
    if mode == 1:
        return 1
    if mode == 2:
        return maximum - 1
    return int(rng.integers(maximum))


def _sample_counts(rng, number, area, maximum, winner_count, forced_loser=None):
    winners = rng.choice(number, size=winner_count, replace=False)
    counts = np.zeros(number, dtype=np.int16)
    counts[winners] = maximum
    losers = [i for i in range(number) if counts[i] != maximum]
    for i in losers:
        counts[i] = _loser_value(rng, maximum)
    if forced_loser is not None:
        assert losers and forced_loser < maximum
        counts[int(rng.choice(losers))] = forced_loser
    if maximum == area and losers and not any(counts[i] > 0 for i in losers):
        counts[int(rng.choice(losers))] = 1
    return counts


def _local_positions(rng, height, width, count, mode):
    cells = [(r, c) for r in range(height) for c in range(width)]
    if not count:
        return []
    if mode == "random":
        selected = rng.choice(len(cells), size=count, replace=False)
        return [cells[int(i)] for i in selected]
    if mode == "compact":
        anchor = cells[int(rng.integers(len(cells)))]
        scores = [(abs(r-anchor[0]) + abs(c-anchor[1]), float(rng.random()), (r, c)) for r, c in cells]
        scores.sort()
        return [item[2] for item in scores[:count]]
    assert mode == "dispersed"
    selected = [cells.pop(int(rng.integers(len(cells))))]
    while len(selected) < count:
        distances = [min(abs(r-a) + abs(c-b) for a, b in selected) for r, c in cells]
        farthest = [i for i, distance in enumerate(distances) if distance == max(distances)]
        selected.append(cells.pop(int(rng.choice(farthest))))
    return selected


def _pair_from_counts(rng, geometry, colors, counts, arrangement=None):
    rows, cols, height, width = geometry
    background, separator, payload = colors
    shape = (rows * height + rows - 1, cols * width + cols - 1)
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for r in range(1, rows):
        inp[r * (height + 1) - 1, :] = separator
        out[r * (height + 1) - 1, :] = separator
    for c in range(1, cols):
        inp[:, c * (width + 1) - 1] = separator
        out[:, c * (width + 1) - 1] = separator
    maximum = max(counts)
    for index, count in enumerate(counts):
        r, c = divmod(index, cols)
        top, left = r * (height + 1), c * (width + 1)
        mode = arrangement.get(index, "random") if arrangement else rng.choice(("random", "compact", "dispersed"))
        for local_r, local_c in _local_positions(rng, height, width, int(count), mode):
            inp[top + local_r, left + local_c] = payload
        if count == maximum:
            out[top:top + height, left:left + width] = payload
    return inp, out


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _source_copy(inp, out):
    rows_in = tuple("".join(str(int(v)) for v in row) for row in inp)
    rows_out = tuple("".join(str(int(v)) for v in row) for row in out)
    return (rows_in, rows_out) in _SOURCE


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    background = int(rng.integers(10))
    palette = [int(x) for x in rng.permutation([c for c in range(10) if c != background])]
    fixed_geometry = bool(rng.random() < 0.16)
    shared = _sample_geometry(rng) if fixed_geometry else None
    total = num_train + num_test
    geometries = []
    for i in range(total):
        if fixed_geometry:
            geometries.append(shared)
        else:
            exclude = geometries[0] if i == 1 else None
            geometries.append(_sample_geometry(rng, exclude=exclude, minimum_area=6 if i == 1 else 4))
    if geometries[1][2] * geometries[1][3] < 6:
        geometries[1] = _sample_geometry(rng, exclude=geometries[0] if not fixed_geometry else None, minimum_area=6)
        if fixed_geometry:
            geometries = [geometries[1]] * total
    cap0 = geometries[0][2] * geometries[0][3]
    cap1 = geometries[1][2] * geometries[1][3]
    first_max = _sample_maximum(rng, min(cap0, cap1 - 1))
    second_max = _sample_maximum(rng, cap1, first_max + 1)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_maxima = []
    for i, geometry in enumerate(geometries):
        regime = "source_train" if i < num_train else "target_test"
        number = geometry[0] * geometry[1]
        area = geometry[2] * geometry[3]
        if i == 0:
            maximum, winner_count = first_max, 1
        elif i == 1:
            maximum = second_max
            winner_count = int(rng.integers(2, number))
        else:
            if regime == "target_test" and rng.random() < 0.32:
                lower = min(area, max(train_maxima))
                maximum = _sample_maximum(rng, area, lower)
            else:
                maximum = _sample_maximum(rng, area)
            winner_count = _sample_winner_count(rng, number, maximum, area, regime)
        if winner_count == number and maximum == area:
            maximum -= 1
        if maximum < 1:
            maximum = 1
        if i < num_train:
            train_maxima.append(maximum)
        if i == 0:
            separator, payload = palette[0], palette[1]
        elif i == 1:
            separator, payload = palette[2], palette[3]
        else:
            separator, payload = [int(x) for x in rng.choice(palette, size=2, replace=False)]
        colors = background, separator, payload
        for _ in range(80):
            counts = _sample_counts(rng, number, area, maximum, winner_count, first_max if i == 1 else None)
            arrangement = None
            if i == 1:
                losers = [j for j, value in enumerate(counts) if value < maximum]
                winners = [j for j, value in enumerate(counts) if value == maximum]
                contrast_loser = next(j for j in losers if counts[j] == first_max)
                arrangement = {contrast_loser: "compact", winners[0]: "compact", winners[1]: "dispersed"}
            inp, out = _pair_from_counts(rng, geometry, colors, counts, arrangement)
            key = _grid_key(inp)
            if key not in seen and not _source_copy(inp, out) and not np.array_equal(inp, out):
                break
        else:
            raise ValueError("could not construct distinct pair")
        seen.add(key)
        puzzle["test" if regime == "target_test" else "train"].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(inp, out, background):
    values = set(int(v) for v in np.unique(inp))
    assert len(values) == 3 and background in values
    candidates = []
    for color in values:
        lines_r = tuple(int(v) for v in np.flatnonzero(np.all(inp == color, axis=1)))
        lines_c = tuple(int(v) for v in np.flatnonzero(np.all(inp == color, axis=0)))
        if lines_r and lines_c:
            candidates.append((color, lines_r, lines_c))
    assert len(candidates) == 1
    separator, lines_r, lines_c = candidates[0]
    assert separator != background
    payload, = values - {separator, background}
    edge_r = (-1,) + lines_r + (inp.shape[0],)
    edge_c = (-1,) + lines_c + (inp.shape[1],)
    heights = [b-a-1 for a, b in zip(edge_r, edge_r[1:])]
    widths = [b-a-1 for a, b in zip(edge_c, edge_c[1:])]
    assert 2 <= len(heights) <= 7 and 2 <= len(widths) <= 7
    assert len(set(heights)) == len(set(widths)) == 1
    assert 2 <= heights[0] <= 8 and 2 <= widths[0] <= 8
    counts = []
    expected = np.full(inp.shape, background, dtype=np.int8)
    expected[list(lines_r), :] = separator
    expected[:, list(lines_c)] = separator
    for a, b in zip(edge_r, edge_r[1:]):
        for c, d in zip(edge_c, edge_c[1:]):
            tile = inp[a+1:b, c+1:d]
            assert tile.shape == (heights[0], widths[0])
            assert np.all((tile == background) | (tile == payload))
            counts.append(int(np.count_nonzero(tile == payload)))
    maximum = max(counts)
    assert maximum > 0
    for i, count in enumerate(counts):
        if count == maximum:
            r, c = divmod(i, len(widths))
            expected[edge_r[r]+1:edge_r[r+1], edge_c[c]+1:edge_c[c+1]] = payload
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    return {"geometry": (len(heights), len(widths), heights[0], widths[0]),
            "separator": separator, "payload": payload, "counts": tuple(counts),
            "maximum": maximum, "winners": counts.count(maximum)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    all_pairs = puzzle["train"] + puzzle["test"]
    for pair in all_pairs:
        assert type(pair) is dict and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= size <= 30 for size in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))
        inp, out = pair["input"], pair["output"]
        assert inp.shape == out.shape
        key = _grid_key(inp)
        pair_key = key, _grid_key(out)
        assert key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(key)
        seen_pairs.add(pair_key)
        assert not _source_copy(inp, out)
    train_nonseparators = []
    for pair in puzzle["train"]:
        inp = pair["input"]
        vals = set(int(v) for v in np.unique(inp))
        separators = {v for v in vals if np.any(np.all(inp == v, axis=1)) and np.any(np.all(inp == v, axis=0))}
        assert len(separators) == 1
        train_nonseparators.append(vals - separators)
    common = set.intersection(*train_nonseparators)
    assert len(common) == 1, "training must identify the background color"
    background, = common
    infos = [_inspect_pair(pair["input"], pair["output"], background) for pair in all_pairs]
    first, second = infos[:2]
    assert first["winners"] == 1 and second["winners"] >= 2
    assert first["maximum"] < second["maximum"]
    assert first["maximum"] in second["counts"]
    assert first["payload"] != second["payload"]
    assert first["separator"] != second["separator"]


def _puzzle_key(puzzle):
    return tuple((_grid_key(p["input"]), _grid_key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def _component_count(mask):
    remaining = set(zip(*np.nonzero(mask)))
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    geometries, maxima_train, maxima_test, tie_counts, role_triples, layouts = set(), set(), set(), set(), set(), set()
    mixed_geometry = mixed_payload = mixed_separator = 0
    all_tied = unique_test = empty_test = smaller_test = larger_test = 0
    no_empty_train = one_loser = near_tie = large_gap = already_solid = 0
    connected_loser_disconnected_winner = different_tied_arrangements = 0
    combinations = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = puzzle["train"]
        common = set.intersection(*(set(np.unique(pair["input"]).tolist()) - {
            next(v for v in np.unique(pair["input"]) if np.any(np.all(pair["input"] == v, axis=1)) and np.any(np.all(pair["input"] == v, axis=0)))} for pair in train))
        background, = common
        infos = [_inspect_pair(p["input"], p["output"], background) for s in ("train", "test") for p in puzzle[s]]
        mixed_geometry += len({x["geometry"] for x in infos}) > 1
        mixed_payload += len({x["payload"] for x in infos}) > 1
        mixed_separator += len({x["separator"] for x in infos}) > 1
        max_train = max(x["maximum"] for x in infos[:len(train)])
        for i, info in enumerate(infos):
            geometries.add(info["geometry"])
            tie_counts.add(info["winners"])
            role_triples.add((background, info["separator"], info["payload"]))
            layouts.add((info["geometry"], info["counts"]))
            combinations.add((info["geometry"], info["maximum"], info["winners"],
                              background, info["separator"], info["payload"]))
            maximum = info["maximum"]
            losers = [count for count in info["counts"] if count < maximum]
            near_tie += maximum - 1 in losers
            large_gap += any(maximum - count >= 3 for count in losers)
            one_loser += len(losers) == 1
            already_solid += maximum == info["geometry"][2] * info["geometry"][3]
            inp = (puzzle["train"] + puzzle["test"])[i]["input"]
            rows, cols, height, width = info["geometry"]
            components = []
            for index, count in enumerate(info["counts"]):
                r, c = divmod(index, cols)
                tile = inp[r * (height + 1):r * (height + 1) + height,
                           c * (width + 1):c * (width + 1) + width]
                components.append((count, _component_count(tile == info["payload"])))
            connected_loser_disconnected_winner += (
                any(count >= 2 and pieces == 1 for count, pieces in components if count < maximum)
                and any(pieces >= 2 for count, pieces in components if count == maximum)
            )
            different_tied_arrangements += (
                info["winners"] >= 2
                and len({pieces for count, pieces in components if count == maximum}) >= 2
            )
            if i < len(train):
                maxima_train.add(info["maximum"])
                no_empty_train += 0 not in info["counts"]
            else:
                maxima_test.add(info["maximum"])
                all_tied += info["winners"] == len(info["counts"])
                unique_test += info["winners"] == 1
                empty_test += 0 in info["counts"]
                smaller_test += info["maximum"] < max_train
                larger_test += info["maximum"] > max_train
    assert len(geometries) >= 25 and len(layouts) >= 100
    assert len(role_triples) >= 30
    assert {1, 2, 3, 4}.issubset(maxima_train | maxima_test)
    assert max(maxima_train | maxima_test) >= 10
    assert max(tie_counts) >= 4
    assert len(combinations) >= 100
    assert mixed_geometry >= 20 and mixed_payload == 42 and mixed_separator == 42
    assert all_tied and unique_test and empty_test and smaller_test and larger_test
    assert no_empty_train and one_loser and near_tie and large_gap and already_solid
    assert connected_loser_disconnected_winner and different_tied_arrangements
    assert _puzzle_key(generate_puzzle(1234)) == _puzzle_key(generate_puzzle(1234))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(5000 + 10 * train_count + test_count, train_count, test_count)
        assert (len(p["train"]), len(p["test"])) == (train_count, test_count)
        validate_generated_puzzle(p)
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(6000 + (train_count or 0) + (test_count or 0), train_count, test_count)
        assert train_count is None or len(p["train"]) == train_count
        assert test_count is None or len(p["test"]) == test_count
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(7000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid pair counts: {kwargs}")
