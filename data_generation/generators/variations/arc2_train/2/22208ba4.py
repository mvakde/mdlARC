"""Four corner blocks: every repeated color advances one block diagonally inward."""

import numpy as np


# Exact descriptions of the five source inputs. Source dimensions remain usable.
_SOURCE_INPUTS = frozenset({
    (16, 16, 1, 7, 2, 2, 2, 2),
    (17, 17, 4, 7, 8, 8, 9, 1),
    (22, 15, 3, 7, 9, 2, 9, 9),
    (7, 7, 2, 7, 5, 5, 9, 5),
    (7, 30, 2, 7, 6, 1, 1, 3),
})


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _integer(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _integer(num_test, "num_test", 1, 4)
    choices = [(a, b) for a in range(2, 11)
               for b in range(1, min(4, a - 1) + 1)
               if (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[seed % len(choices)]


def _pattern(colors):
    counts = {color: colors.count(color) for color in colors}
    counts_sorted = sorted(counts.values(), reverse=True)
    if counts_sorted == [4]:
        return "all_four"
    if counts_sorted == [3, 1]:
        return "triple"
    if counts_sorted == [2, 2]:
        return "two_pairs"
    if counts_sorted == [1, 1, 1, 1]:
        return "all_distinct"
    if counts_sorted == [2, 1, 1]:
        repeated = next(color for color in counts if counts[color] == 2)
        indices = [i for i, color in enumerate(colors) if color == repeated]
        adjacent = (indices[0] // 2 == indices[1] // 2
                    or indices[0] % 2 == indices[1] % 2)
        return "adjacent_pair" if adjacent else "diagonal_pair"
    raise AssertionError("invalid color partition")


def _parse_input(grid):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "input array")
    _require(np.issubdtype(grid.dtype, np.integer), "input dtype")
    height, width = grid.shape
    _require(1 <= height <= 30 and 1 <= width <= 30, "input dimensions")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), "input colors")
    first = int(grid[0, 0])
    changes = np.flatnonzero(grid[0] != first)
    _require(changes.size > 0, "no field after first corner")
    size = int(changes[0])
    _require(height >= 3 * size and width >= 3 * size, "inward step will not fit")
    background = int(grid[size, size])
    starts = ((0, 0), (0, width - size),
              (height - size, 0), (height - size, width - size))
    colors = tuple(int(grid[row, col]) for row, col in starts)
    _require(all(color != background for color in colors), "corner uses background")
    expected = np.full(grid.shape, background, dtype=grid.dtype)
    for (row, col), color in zip(starts, colors):
        expected[row:row + size, col:col + size] = color
    _require(np.array_equal(grid, expected), "not exactly four equal solid corners")
    return height, width, size, background, colors


def _destinations(height, width, size, colors):
    starts = ((0, 0), (0, width - size),
              (height - size, 0), (height - size, width - size))
    return tuple((row + (size if i < 2 else -size),
                  col + (size if i % 2 == 0 else -size), color)
                 for i, ((row, col), color) in enumerate(zip(starts, colors))
                 if colors.count(color) >= 2)


def _collision_free(height, width, size, colors):
    boxes = _destinations(height, width, size, colors)
    return all(a == b or abs(r1 - r2) >= size or abs(c1 - c2) >= size
               for i, (r1, c1, a) in enumerate(boxes)
               for r2, c2, b in boxes[i + 1:])


def _output_from_input(grid, parsed=None):
    height, width, size, background, colors = parsed or _parse_input(grid)
    _require(_collision_free(height, width, size, colors), "different colors collide")
    output = np.full(grid.shape, background, dtype=grid.dtype)
    starts = ((0, 0), (0, width - size),
              (height - size, 0), (height - size, width - size))
    for (row, col), color in zip(starts, colors):
        if colors.count(color) == 1:
            output[row:row + size, col:col + size] = color
    for row, col, color in _destinations(height, width, size, colors):
        output[row:row + size, col:col + size] = color
    return output


def _layout(rng, pattern):
    if pattern == "all_four":
        return (0, 0, 0, 0)
    if pattern == "triple":
        result = [0] * 4
        result[int(rng.integers(4))] = 1
        return tuple(result)
    if pattern == "all_distinct":
        return (0, 1, 2, 3)
    if pattern == "two_pairs":
        options = ((0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0))
        return options[int(rng.integers(3))]
    if pattern == "adjacent_pair":
        options = ((0, 0, 1, 2), (1, 2, 0, 0),
                   (0, 1, 0, 2), (1, 0, 2, 0))
        return options[int(rng.integers(4))]
    if pattern == "diagonal_pair":
        return (0, 1, 2, 0) if int(rng.integers(2)) == 0 else (1, 0, 0, 2)
    raise ValueError("unknown pattern")


def _size(rng, pattern):
    maximum = 7 if pattern == "two_pairs" else 9
    bands = ((1, 2, 3), (2, 3, 4, 5), (4, 5, 6, 7, 8, 9))
    band = bands[int(rng.choice(3, p=(.36, .40, .24)))]
    return int(rng.choice([n for n in band if n <= maximum]))


def _band_possible(size, band):
    return (3 * size <= 30 if band == "overlap"
            else 4 * size <= 30 if band == "touch"
            else 4 * size + 1 <= 30)


def _extent(rng, size, band):
    if band == "overlap":
        low, high = 3 * size, min(4 * size - 1, 30)
    elif band == "touch":
        low = high = 4 * size
    else:
        low, high = 4 * size + 1, 30
    return int(rng.integers(low, high + 1))


def _dimensions(rng, size, pattern):
    bands = [b for b in ("overlap", "touch", "separate") if _band_possible(size, b)]
    chosen = [str(rng.choice(bands)), str(rng.choice(bands))]
    if pattern == "two_pairs":
        # A touching or separated axis prevents different color collisions.
        safe = [b for b in bands if b != "overlap"]
        chosen[int(rng.integers(2))] = str(rng.choice(safe))
    return _extent(rng, size, chosen[0]), _extent(rng, size, chosen[1])


def _colors(rng, layout, background, pool, preferred=None):
    available = [int(color) for color in pool if color != background]
    rng.shuffle(available)
    if preferred is not None and preferred in available:
        available.remove(preferred)
        available.insert(0, preferred)
    mapping = dict(zip(sorted(set(layout)), available))
    return tuple(mapping[label] for label in layout)


def _make_pair(rng, pattern, pool, forbidden_background=None,
               preferred=None, excluded_sizes=()):
    for _ in range(150):
        size = _size(rng, pattern)
        if size in excluded_sizes:
            continue
        height, width = _dimensions(rng, size, pattern)
        background = int(rng.integers(10))
        if background == forbidden_background:
            continue
        colors = _colors(rng, _layout(rng, pattern), background, pool, preferred)
        if not _collision_free(height, width, size, colors):
            continue
        if (height, width, size, background, *colors) in _SOURCE_INPUTS:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        starts = ((0, 0), (0, width - size),
                  (height - size, 0), (height - size, width - size))
        for (row, col), color in zip(starts, colors):
            grid[row:row + size, col:col + size] = color
        return {"input": grid, "output": _output_from_input(grid)}
    raise RuntimeError("could not sample a valid pair")


def _signature(grid):
    return grid.shape, grid.dtype.str, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = ("adjacent_to_diagonal", "diagonal_to_adjacent", "mixed")[int(rng.integers(3))]
    palette = ("independent", "fresh_test", "role_reversal")[int(rng.integers(3))]
    shuffled = [int(x) for x in rng.permutation(10)]
    train_pool = shuffled[:5] if palette == "fresh_test" else list(range(10))
    test_pool = shuffled[5:] if palette == "fresh_test" else list(range(10))
    train_pair = "diagonal_pair" if shift == "diagonal_to_adjacent" else "adjacent_pair"
    test_pair = "adjacent_pair" if shift == "diagonal_to_adjacent" else "diagonal_pair"
    train_patterns = ["two_pairs", train_pair]
    extras = ("all_four", "triple", "adjacent_pair", "diagonal_pair",
              "two_pairs", "all_distinct")
    train_patterns += [str(rng.choice(extras, p=(.19, .22, .19, .17, .15, .08)))
                       for _ in range(num_train - 2)]
    rng.shuffle(train_patterns)
    test_options = (test_pair, "two_pairs", "triple", "all_four",
                    "all_distinct", train_pair)
    test_patterns = [test_pair]
    test_patterns += [str(rng.choice(test_options, p=(.33, .22, .16, .11, .08, .10)))
                      for _ in range(num_test - 1)]
    rng.shuffle(test_patterns)

    puzzle = {"train": [], "test": []}
    used = set()
    first_background = first_size = None
    stationary_colors = []
    for split, patterns, pool in (("train", train_patterns, train_pool),
                                  ("test", test_patterns, test_pool)):
        for index, pattern in enumerate(patterns):
            for _ in range(150):
                forbid = first_background if split == "train" and index == 1 else None
                exclude = (first_size,) if split == "train" and index == 1 else ()
                preferred = (int(rng.choice(stationary_colors))
                             if split == "test" and palette == "role_reversal"
                             and stationary_colors and pattern != "all_distinct" else None)
                pair = _make_pair(rng, pattern, pool, forbid, preferred, exclude)
                signature = _signature(pair["input"])
                if signature in used:
                    continue
                used.add(signature)
                puzzle[split].append(pair)
                _, _, size, background, colors = _parse_input(pair["input"])
                if split == "train":
                    stationary_colors.extend(color for color in colors
                                             if colors.count(color) == 1)
                    if index == 0:
                        first_background, first_size = background, size
                break
            else:
                raise RuntimeError("could not produce unique puzzle inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split type")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest, "pair counts")
    inputs, pairs = set(), set()
    train_patterns, train_sizes, train_backgrounds = [], set(), set()
    test_patterns = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid type")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid shape")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "grid colors")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape and inp.dtype == out.dtype, "shape or dtype changed")
            parsed = _parse_input(inp)
            height, width, size, bg, colors = parsed
            _require((height, width, size, bg, *colors) not in _SOURCE_INPUTS, "source pair copied")
            _require(_collision_free(height, width, size, colors), "color collision")
            _require(np.array_equal(out, _output_from_input(inp, parsed)), "incorrect output")
            pattern = _pattern(colors)
            in_sig, out_sig = _signature(inp), _signature(out)
            _require(in_sig not in inputs and (in_sig, out_sig) not in pairs, "duplicate pair")
            inputs.add(in_sig)
            pairs.add((in_sig, out_sig))
            if split == "train":
                train_patterns.append(pattern)
                train_sizes.add(size)
                train_backgrounds.add(bg)
            else:
                test_patterns.append(pattern)
    _require("two_pairs" in train_patterns, "training must show two moving groups")
    _require(any(p in ("adjacent_pair", "diagonal_pair") for p in train_patterns),
             "training must show moving and stationary corners")
    _require(len(train_sizes) >= 2, "training must show size-relative movement")
    _require(len(train_backgrounds) >= 2, "background must vary by pair")
    _require(any(p in ("adjacent_pair", "diagonal_pair") for p in test_patterns),
             "test must exercise a matching pair")


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and
               all(np.array_equal(x[key], y[key]) for x, y in zip(a[s], b[s])
                   for key in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    def independent_prediction(grid):
        height, width, size, background, colors = _parse_input(grid)
        result = np.full_like(grid, background)
        corners = ((0, 0), (0, width - size),
                   (height - size, 0), (height - size, width - size))
        for row in range(height):
            for col in range(width):
                candidates = set()
                for i, ((r, c), color) in enumerate(zip(corners, colors)):
                    if colors.count(color) >= 2:
                        r += size if i < 2 else -size
                        c += size if i % 2 == 0 else -size
                    if r <= row < r + size and c <= col < c + size:
                        candidates.add(color)
                assert len(candidates) <= 1
                if candidates:
                    result[row, col] = next(iter(candidates))
        return result

    train_counts, test_counts, patterns, sizes, shapes = set(), set(), set(), set(), set()
    geometry, orientations, backgrounds = set(), set(), set()
    geometry_by_pattern, aspects, shifts, color_relations = set(), set(), set(), set()
    mixed_background_puzzles = mixed_aspect_puzzles = source_shape_uses = 0
    size_one_triples = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        observed_movement = {}
        for pair in puzzle["train"]:
            height, width, size, _, colors = _parse_input(pair["input"])
            corners = ((0, 0), (0, width - size),
                       (height - size, 0), (height - size, width - size))
            for (row, col), color in zip(corners, colors):
                stayed = bool(np.all(pair["output"][row:row + size, col:col + size] == color))
                observed_movement.setdefault(colors.count(color), set()).add(not stayed)
        assert observed_movement[1] == {False}
        assert observed_movement[2] == {True}
        assert all(outcome == {True} for count, outcome in observed_movement.items()
                   if count >= 2)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_backgrounds, local_aspects = set(), set()
        train_names, test_names = set(), set()
        train_movers, test_movers, train_stationary = set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(pair["output"], independent_prediction(pair["input"]))
                height, width, size, bg, colors = _parse_input(pair["input"])
                pattern = _pattern(colors)
                (train_names if split == "train" else test_names).add(pattern)
                patterns.add((split, pattern))
                sizes.add((split, size))
                shapes.add((height, width))
                backgrounds.add(bg)
                local_backgrounds.add(bg)
                aspect = ("wide" if width >= 2 * height else
                          "tall" if height >= 2 * width else
                          "square" if height == width else "moderate")
                aspects.add((split, aspect))
                local_aspects.add(aspect)
                source_shape_uses += (height, width) in {
                    (16, 16), (17, 17), (22, 15), (7, 7), (7, 30)}
                if size == 1 and pattern == "triple":
                    size_one_triples.add((height, width))
                for color in colors:
                    if colors.count(color) >= 2:
                        (train_movers if split == "train" else test_movers).add(color)
                    elif split == "train":
                        train_stationary.add(color)
                if pattern == "two_pairs":
                    orientations.add(tuple(i for i, x in enumerate(colors) if x == colors[0]))
                boxes = _destinations(height, width, size, colors)
                for i, (r1, c1, a) in enumerate(boxes):
                    for r2, c2, b in boxes[i + 1:]:
                        if a == b:
                            row_gap, col_gap = abs(r1 - r2), abs(c1 - c2)
                            kind = ("overlap" if row_gap < size and col_gap < size else
                                    "touch" if row_gap <= size and col_gap <= size else "separate")
                            partial_axes = int(0 < row_gap < size) + int(0 < col_gap < size)
                            geometry.add((kind, partial_axes))
                            geometry_by_pattern.add((pattern, kind, partial_axes))
        mixed_background_puzzles += len(local_backgrounds) >= 2
        mixed_aspect_puzzles += len(local_aspects) >= 2
        if "adjacent_pair" in train_names and "diagonal_pair" in test_names:
            shifts.add("adjacent_to_diagonal")
        if "diagonal_pair" in train_names and "adjacent_pair" in test_names:
            shifts.add("diagonal_to_adjacent")
        color_relations.add("shared" if train_movers & test_movers else "disjoint")
        if train_stationary & test_movers:
            color_relations.add("role_reversal")
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 65 and len(backgrounds) == 10
    assert mixed_background_puzzles == 42
    assert mixed_aspect_puzzles >= 30
    assert {aspect for _, aspect in aspects} == {"wide", "tall", "square", "moderate"}
    assert shifts == {"adjacent_to_diagonal", "diagonal_to_adjacent"}
    assert color_relations == {"shared", "disjoint", "role_reversal"}
    assert source_shape_uses > 0
    assert any(shape != (3, 3) for shape in size_one_triples)
    assert {p for s, p in patterns if s == "train"} == {
        "all_four", "triple", "adjacent_pair", "diagonal_pair", "two_pairs", "all_distinct"}
    assert {p for s, p in patterns if s == "test"} >= {
        "adjacent_pair", "diagonal_pair", "two_pairs"}
    assert max(n for s, n in sizes if s == "test") >= 5
    assert len(orientations) == 3
    assert ("overlap", 1) in geometry
    assert any(kind == "overlap" and partial == 2 for kind, partial in geometry)
    assert any(kind == "touch" for kind, _ in geometry)
    assert any(kind == "separate" for kind, _ in geometry)
    assert all(any(name == pattern and kind == "overlap"
                   for name, kind, _ in geometry_by_pattern)
               for pattern in ("all_four", "triple", "two_pairs",
                               "adjacent_pair", "diagonal_pair"))
    for seed, a, b in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, 6, 3),
                       (1101, None, 1), (1102, None, 4), (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        if a is not None:
            assert len(puzzle["train"]) == a
        if b is not None:
            assert len(puzzle["test"]) == b
    assert _same_puzzle(generate_puzzle(1200, 7, 3), generate_puzzle(1200, 7, 3))
    assert not _same_puzzle(generate_puzzle(1201, 7, 3), generate_puzzle(1202, 7, 3))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(1300, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
