"""Synthetic instances of the alternating seed propagation ARC task."""

import numpy as np


_SOURCE_PAIRS = (
    (("0400000",) + ("0000000",) * 5,
     ("0400000", "4040000") * 3),
    (("0200080000",) + ("0000000000",) * 5,
     ("0200080000", "2020808000") * 3),
    (("003000600700",) + ("000000000000",) * 5,
     ("003000600700", "030306067070") * 3),
)
_DIRECTIONS = ("top", "bottom", "left", "right")


def _key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _source_copy(inp, out):
    rows = lambda grid: tuple("".join(str(int(x)) for x in row) for row in grid)
    return (rows(inp), rows(out)) in _SOURCE_PAIRS


def _counts(rng, num_train, num_test):
    for value, name, lo, hi in (
        (num_train, "num_train", 2, 10),
        (num_test, "num_test", 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi
        ):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    train = None if num_train is None else int(num_train)
    test = None if num_test is None else int(num_test)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _length(rng, minimum, maximum=30):
    """Sample small, intermediate, and large lengths with broad support."""
    bands = ((3, 7), (8, 17), (18, 30))
    order = rng.permutation(3)
    for index in order:
        low, high = bands[int(index)]
        low, high = max(low, minimum), min(high, maximum)
        if low <= high:
            return int(rng.integers(low, high + 1))
    return int(rng.integers(minimum, maximum + 1))


def _positions(rng, edge, count, style, identifiable):
    if count == 1:
        if identifiable:
            return [int(rng.integers(1, edge - 1))]
        return [int(rng.choice((0, edge - 1)))]
    gap = 2 if style == "dense" else 3
    minimum = 1 + gap * (count - 1)
    slack = edge - minimum
    assert slack >= 0
    if style == "regular":
        spacing = int(rng.integers(gap, (edge - 1) // (count - 1) + 1))
        span = spacing * (count - 1)
        start = int(rng.integers(0, edge - span))
        positions = [start + spacing * i for i in range(count)]
    elif style == "spread":
        extras = rng.multinomial(slack, np.full(count - 1, 1 / (count - 1)))
        positions = [0]
        for extra in extras:
            positions.append(positions[-1] + gap + int(extra))
    elif style == "compact":
        extras = np.zeros(count - 1, dtype=int)
        for _ in range(min(slack, count - 1)):
            extras[int(rng.integers(0, count - 1))] += 1
        span = minimum + int(sum(extras))
        margin = edge - span
        placement = int(rng.integers(0, 3))
        start = (0, margin, margin // 2)[placement]
        positions = [start]
        for extra in extras:
            positions.append(positions[-1] + gap + int(extra))
    elif style == "mixed":
        weights = rng.dirichlet(np.full(count + 1, 0.65))
        extras = rng.multinomial(slack, weights)
        positions = [int(extras[0])]
        for i in range(count - 1):
            positions.append(positions[-1] + gap + int(extras[i + 1]))
    else:  # dense: two-apart tracks, whose intersections share a color
        start = int(rng.integers(0, slack + 1))
        positions = [start + 2 * i for i in range(count)]
    if identifiable and all(p in (0, edge - 1) for p in positions):
        if positions[-1] - positions[-2] >= 3:
            positions[-1] -= 1
        else:
            positions[0] += 1
    assert any(0 < p < edge - 1 for p in positions)
    return positions


def _colors(rng, count, background, style):
    available = [c for c in range(10) if c != background]
    mode = "mono" if style == "dense" else str(rng.choice(("unique", "mono", "repeat")))
    if count == 1:
        return [int(rng.choice(available))]
    if mode == "mono":
        return [int(rng.choice(available))] * count
    if mode == "unique" and count <= 9:
        return [int(c) for c in rng.choice(available, count, replace=False)]
    palette_size = int(rng.integers(2, min(count - 1, 8) + 1)) if count >= 3 else 1
    palette = [int(c) for c in rng.choice(available, palette_size, replace=False)]
    colors = palette + [int(rng.choice(palette)) for _ in range(count - palette_size)]
    rng.shuffle(colors)
    return colors


def _make_pair(rng, direction, count, background, source_like, is_train,
               max_train_edge, allow_corner, require_long, target_shape):
    style = str(rng.choice(("regular", "compact", "spread", "mixed", "dense"),
                           p=(0.23, 0.21, 0.18, 0.27, 0.11)))
    if count == 1:
        style = "single"
    elif count == 2 and style == "regular":
        style = "mixed"
    minimum_edge = max(3, 2 * count - 1 if style == "dense" else 3 * count - 2)
    if not allow_corner and count == 2:
        minimum_edge = max(minimum_edge, 5)
    maximum_edge = 15 if source_like and is_train else 30
    if source_like and not is_train:
        minimum_edge = max(minimum_edge, max_train_edge + 1)
    if target_shape is None:
        edge = _length(rng, minimum_edge, maximum_edge)
        inward = _length(rng, 4 if require_long else 3,
                         14 if source_like and is_train else 30)
    else:
        edge = target_shape[1] if direction in ("top", "bottom") else target_shape[0]
        inward = target_shape[0] if direction in ("top", "bottom") else target_shape[1]
        assert edge >= minimum_edge and inward >= 3
    positions = _positions(rng, edge, count, style, not allow_corner)
    colors = _colors(rng, count, background, style)
    canonical = np.full((inward, edge), background, dtype=np.int8)
    canonical[0, positions] = colors
    result = np.full_like(canonical, background)
    for distance in range(inward):
        for p, color in zip(positions, colors):
            for lateral in ((0,) if distance % 2 == 0 else (-1, 1)):
                q = p + lateral
                if 0 <= q < edge:
                    assert result[distance, q] in (background, color)
                    result[distance, q] = color
    turns = {"top": 0, "left": 1, "bottom": 2, "right": 3}[direction]
    return np.rot90(canonical, turns).copy(), np.rot90(result, turns).copy()


def _direction_schedule(rng, train_count, test_count, mode):
    if mode == "fixed":
        direction = str(rng.choice(_DIRECTIONS))
        return [direction] * train_count, [direction] * test_count
    if mode == "new":
        held = str(rng.choice(_DIRECTIONS))
        shown = [d for d in _DIRECTIONS if d != held]
        training = [str(rng.choice(shown)) for _ in range(train_count)]
        testing = [held] + [str(rng.choice(_DIRECTIONS)) for _ in range(test_count - 1)]
        rng.shuffle(testing)
        return training, testing
    shuffled = list(rng.permutation(_DIRECTIONS))
    training = shuffled[:2] + [str(rng.choice(_DIRECTIONS)) for _ in range(train_count - 2)]
    testing = [str(rng.choice(_DIRECTIONS)) for _ in range(test_count)]
    rng.shuffle(training)
    return training, testing


def _seed_counts(rng, train_count, test_count, source_like):
    if source_like:
        training = [1, 2] + [int(rng.integers(1, 3)) for _ in range(train_count - 2)]
        testing = [int(rng.integers(3, 8)) for _ in range(test_count)]
    else:
        training = [1, int(rng.integers(2, 7))]
        training += [int(rng.integers(1, 11)) for _ in range(train_count - 2)]
        testing = [int(rng.integers(1, 11)) for _ in range(test_count)]
    rng.shuffle(training)
    return training, testing


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    direction_mode = str(rng.choice(("fixed", "mixed", "new"), p=(0.32, 0.38, 0.30)))
    source_like = bool(rng.random() < 0.28)
    fixed_background = bool(rng.random() < 0.25)
    train_directions, test_directions = _direction_schedule(
        rng, train_count, test_count, direction_mode)
    train_seeds, test_seeds = _seed_counts(rng, train_count, test_count, source_like)
    corner_case = (direction_mode == "fixed" and not source_like
                   and bool(rng.random() < 0.55))
    tiny_case = not source_like and bool(rng.random() < 0.18)
    same_shape_case = not source_like and not tiny_case and bool(rng.random() < 0.35)
    if corner_case or same_shape_case or tiny_case:
        test_seeds[0] = 1
    first_background = int(rng.integers(0, 10))
    backgrounds = [first_background] * (train_count + test_count)
    if not fixed_background:
        for i in range(1, len(backgrounds)):
            choices = [c for c in range(10) if c != first_background] if i == 1 else range(10)
            backgrounds[i] = int(rng.choice(choices))
    child_seeds = rng.integers(0, np.iinfo(np.int64).max,
                               size=train_count + test_count, dtype=np.int64)
    used = set()
    puzzle = {"train": [], "test": []}
    max_train_edge = 0
    matching_shape = None
    for split, directions, counts, offset in (
        ("train", train_directions, train_seeds, 0),
        ("test", test_directions, test_seeds, train_count),
    ):
        if split == "test" and same_shape_case:
            matching_shape = next(
                pair["input"].shape for pair, count in zip(puzzle["train"], train_seeds)
                if count == 1)
        elif split == "test" and tiny_case:
            matching_shape = (3, 3)
        for j, (direction, count) in enumerate(zip(directions, counts)):
            pair_rng = np.random.default_rng(int(child_seeds[offset + j]))
            for _ in range(128):
                allow_corner = (split == "test" and direction_mode == "fixed"
                                and count == 1 and
                                ((corner_case and j == 0) or pair_rng.random() < 0.28))
                inp, out = _make_pair(
                    pair_rng, direction, count, backgrounds[offset + j],
                    source_like, split == "train", max_train_edge, allow_corner,
                    split == "train" and j == 0,
                    matching_shape if split == "test" and j == 0 else None)
                if _key(inp) not in used and not _source_copy(inp, out):
                    break
            else:
                raise RuntimeError("unable to construct a unique pair")
            used.add(_key(inp))
            puzzle[split].append({"input": inp, "output": out})
            if split == "train":
                edge = inp.shape[1] if direction in ("top", "bottom") else inp.shape[0]
                max_train_edge = max(max_train_edge, edge)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_input(inp):
    colors, frequencies = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(frequencies)])
    points = np.argwhere(inp != background)
    assert len(points) >= 1
    height, width = inp.shape
    candidates = []
    if np.all(points[:, 0] == 0):
        candidates.append("top")
    if np.all(points[:, 0] == height - 1):
        candidates.append("bottom")
    if np.all(points[:, 1] == 0):
        candidates.append("left")
    if np.all(points[:, 1] == width - 1):
        candidates.append("right")
    assert candidates
    return background, points, candidates


def _prediction(inp, background, direction):
    """Reconstruct from the seed edge in display coordinates."""
    height, width = inp.shape
    result = np.full_like(inp, background)
    if direction in ("top", "bottom"):
        edge = width
        inward = height
        coordinates = [(int(c), int(inp[0 if direction == "top" else height - 1, c]))
                       for c in range(width)
                       if inp[0 if direction == "top" else height - 1, c] != background]
    else:
        edge = height
        inward = width
        coordinates = [(int(r), int(inp[r, 0 if direction == "left" else width - 1]))
                       for r in range(height)
                       if inp[r, 0 if direction == "left" else width - 1] != background]
    assert coordinates
    for distance in range(inward):
        for lateral, color in coordinates:
            for offset in ((0,) if distance % 2 == 0 else (-1, 1)):
                location = lateral + offset
                if not 0 <= location < edge:
                    continue
                if direction == "top":
                    r, c = distance, location
                elif direction == "bottom":
                    r, c = height - 1 - distance, location
                elif direction == "left":
                    r, c = location, distance
                else:
                    r, c = location, width - 1 - distance
                assert result[r, c] in (background, color), "conflicting seed colors"
                result[r, c] = color
    return result


def _features(inp, known_direction=None):
    background, points, candidates = _inspect_input(inp)
    if len(candidates) == 1:
        direction = candidates[0]
    else:
        assert known_direction in candidates, "ambiguous seed-bearing edge"
        direction = known_direction
    edge = inp.shape[1] if direction in ("top", "bottom") else inp.shape[0]
    inward = inp.shape[0] if direction in ("top", "bottom") else inp.shape[1]
    positions = sorted(int(c if direction in ("top", "bottom") else r) for r, c in points)
    seed_colors = [int(inp[0 if direction == "top" else inp.shape[0] - 1, p])
                   if direction in ("top", "bottom")
                   else int(inp[p, 0 if direction == "left" else inp.shape[1] - 1])
                   for p in positions]
    return {
        "background": background, "direction": direction, "edge": edge,
        "inward": inward, "positions": positions, "colors": seed_colors,
        "count": len(positions), "ambiguous": len(candidates) != 1,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    training = []
    for pair in puzzle["train"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp = pair["input"]
        assert isinstance(inp, np.ndarray) and inp.ndim == 2
        background, points, candidates = _inspect_input(inp)
        assert len(candidates) == 1, "training must identify its seed edge"
        training.append(_features(inp))
    known = training[0]["direction"] if len({x["direction"] for x in training}) == 1 else None
    assert any(x["inward"] >= 4 for x in training)
    assert any(x["count"] >= 2 for x in training)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 3 <= grid.shape[0] <= 30 and 3 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert inp.shape == out.shape
            item = (_key(inp), _key(out))
            assert _key(inp) not in seen_inputs and item not in seen_pairs
            seen_inputs.add(_key(inp))
            seen_pairs.add(item)
            assert not _source_copy(inp, out)
            feature = _features(inp, known)
            assert feature["inward"] >= 3 and feature["edge"] >= 3
            assert np.array_equal(out, _prediction(
                inp, feature["background"], feature["direction"]))
            assert not np.array_equal(inp, out)
            positions, colors = feature["positions"], feature["colors"]
            for a, b, left, right in zip(positions, positions[1:], colors, colors[1:]):
                assert b - a >= 2
                assert b - a != 2 or left == right


def _puzzle_key(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def _expect_bad(**kwargs):
    try:
        generate_puzzle(0, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {kwargs}")


def self_test() -> None:
    for source_input, source_output in _SOURCE_PAIRS:
        inp = np.asarray([[int(x) for x in row] for row in source_input], dtype=np.int8)
        out = np.asarray([[int(x) for x in row] for row in source_output], dtype=np.int8)
        assert np.array_equal(_prediction(inp, 0, "top"), out)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1),
                              (1003, 10, 4), (1004, 7, 3),
                              (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}):
        _expect_bad(**kwargs)
    assert _puzzle_key(generate_puzzle(314159, 7, 3)) == _puzzle_key(generate_puzzle(314159, 7, 3))
    backgrounds, directions, counts, shapes, layouts = set(), set(), set(), set(), set()
    varying_backgrounds = mixed_directions = new_directions = 0
    mono = repeated = regular = dense = tall = wide = 0
    coverage = {
        "fixed_background": 0, "fixed_direction": 0, "train_three": 0,
        "test_few": 0, "same_shape": 0, "smaller_test": 0,
        "wider_test": 0, "not_wider_test": 0, "corner": 0,
        "irregular": 0, "compact": 0, "mixed_gaps": 0,
        "both_edges": 0, "neither_edge": 0, "short": 0,
        "long": 0, "edge_small": 0, "edge_large": 0,
        "tiny": 0, "mixed_background_direction": 0,
        "repeated_horizontal": 0,
    }
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_features = [_features(p["input"]) for p in puzzle["train"]]
        known = train_features[0]["direction"] if len({f["direction"] for f in train_features}) == 1 else None
        test_features = [_features(p["input"], known) for p in puzzle["test"]]
        features = train_features + test_features
        varying_backgrounds += len({f["background"] for f in features}) > 1
        mixed_directions += len({f["direction"] for f in features}) > 1
        new_directions += bool({f["direction"] for f in test_features}
                               - {f["direction"] for f in train_features})
        coverage["fixed_background"] += len({f["background"] for f in features}) == 1
        coverage["fixed_direction"] += len({f["direction"] for f in features}) == 1
        coverage["mixed_background_direction"] += (
            len({f["direction"] for f in features}) > 1
            and len({f["background"] for f in features}) > 1)
        coverage["train_three"] += any(f["count"] >= 3 for f in train_features)
        coverage["test_few"] += any(f["count"] <= 2 for f in test_features)
        coverage["same_shape"] += any(
            p["input"].shape in {q["input"].shape for q in puzzle["train"]}
            for p in puzzle["test"])
        coverage["smaller_test"] += any(
            p["input"].size < max(q["input"].size for q in puzzle["train"])
            for p in puzzle["test"])
        coverage["wider_test"] += all(
            f["edge"] > max(t["edge"] for t in train_features) for f in test_features)
        coverage["not_wider_test"] += any(
            f["edge"] <= max(t["edge"] for t in train_features) for f in test_features)
        for f in features:
            backgrounds.add(f["background"])
            directions.add(f["direction"])
            counts.add(f["count"])
            shapes.add((f["edge"], f["inward"]))
            layouts.add((f["edge"], tuple(f["positions"])))
            colors = f["colors"]
            mono += len(colors) >= 2 and len(set(colors)) == 1
            repeated += len(set(colors)) < len(colors) and len(set(colors)) > 1
            gaps = np.diff(f["positions"])
            regular += len(gaps) >= 2 and len(set(int(g) for g in gaps)) == 1
            dense += any(int(g) == 2 for g in gaps)
            tall += f["inward"] >= 2 * f["edge"]
            wide += f["edge"] >= 2 * f["inward"]
            coverage["corner"] += f["ambiguous"]
            coverage["irregular"] += (
                len(gaps) >= 2 and len(set(int(g) for g in gaps)) > 1)
            coverage["both_edges"] += (
                f["positions"][0] == 0 and f["positions"][-1] == f["edge"] - 1)
            coverage["neither_edge"] += (
                f["positions"][0] > 0 and f["positions"][-1] < f["edge"] - 1)
            coverage["short"] += f["inward"] == 3
            coverage["long"] += f["inward"] == 30
            coverage["edge_small"] += f["edge"] == 3
            coverage["edge_large"] += f["edge"] == 30
            coverage["tiny"] += f["edge"] == 3 and f["inward"] == 3
            coverage["repeated_horizontal"] += (
                f["count"] >= 2 and len(set(f["colors"])) < f["count"]
                and f["direction"] in ("left", "right"))
            if f["count"] >= 2:
                coverage["compact"] += (
                    max(f["positions"][0], f["edge"] - 1 - f["positions"][-1]) >= 5
                    and f["positions"][-1] - f["positions"][0] <= f["edge"] // 2)
                coverage["mixed_gaps"] += (
                    len(gaps) >= 2 and int(max(gaps)) - int(min(gaps)) >= 3)
    assert len(backgrounds) == 10 and len(directions) == 4
    assert len(counts) >= 8 and len(shapes) >= 70 and len(layouts) >= 100
    assert varying_backgrounds >= 15 and mixed_directions >= 15 and new_directions >= 5
    assert mono >= 20 and repeated >= 10 and regular >= 5 and dense >= 5
    assert tall >= 5 and wide >= 5
    assert all(coverage[key] >= minimum for key, minimum in {
        "fixed_background": 5, "fixed_direction": 5, "train_three": 10,
        "test_few": 5, "same_shape": 3, "smaller_test": 10,
        "wider_test": 5, "not_wider_test": 5, "corner": 2,
        "irregular": 10, "compact": 5, "mixed_gaps": 5,
        "both_edges": 5, "neither_edge": 5, "short": 5,
        "long": 2, "edge_small": 2, "edge_large": 2,
        "tiny": 2, "mixed_background_direction": 5,
        "repeated_horizontal": 10,
    }.items())
