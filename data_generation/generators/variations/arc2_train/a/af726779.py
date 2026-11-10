import numpy as np


_SOURCE_INPUT_DESCRIPTORS = frozenset({
    (11, 16, 3, 1, "7373737377373737"),
    (13, 16, 3, 2, "7773737377373777"),
    (9, 16, 3, 4, "3337377737737737"),
    (13, 15, 3, 2, "337377777773733"),
    (21, 17, 3, 2, "73737373737373737"),
    (10, 24, 3, 2, "333777373737737373773733"),
})


def _next_mask(mask):
    result = np.zeros_like(mask, dtype=bool)
    if len(mask) >= 3:
        result[1:-1] = mask[:-2] & ~mask[1:-1] & mask[2:]
    return result


def _generations(mask):
    generations = []
    current = mask.copy()
    while True:
        current = _next_mask(current)
        if not np.any(current):
            return generations
        generations.append(current)
        assert len(generations) <= len(mask)


def _alternating(depth):
    segment = np.zeros(2 * depth + 1, dtype=bool)
    segment[::2] = True
    return segment


def _pack_segments(rng, segments):
    segments = list(segments)
    rng.shuffle(segments)
    gaps = [int(rng.integers(2, 5)) for _ in range(len(segments) - 1)]
    minimum = sum(map(len, segments)) + sum(gaps)
    if minimum > 30:
        gaps = [2] * (len(segments) - 1)
        minimum = sum(map(len, segments)) + sum(gaps)
    assert minimum <= 30
    width = minimum if rng.random() < 0.25 else int(rng.integers(minimum, 31))
    column = int(rng.integers(0, width - minimum + 1))
    mask = np.zeros(width, dtype=bool)
    for index, segment in enumerate(segments):
        mask[column:column + len(segment)] = segment
        column += len(segment)
        if index < len(gaps):
            column += gaps[index]
    return mask


def _make_mask(rng, kind, depth):
    if kind == "pure":
        segment = _alternating(depth)
        span = len(segment)
        width = span if rng.random() < 0.32 else int(rng.integers(span, 31))
        offset = int(rng.integers(0, width - span + 1))
        mask = np.zeros(width, dtype=bool)
        mask[offset:offset + span] = segment
        return mask
    if kind == "one":
        left = int(rng.integers(1, 6))
        right = int(rng.integers(1, 6))
        center = np.asarray([True] * left + [False] + [True] * right)
        pieces = [center]
        if rng.random() < 0.65:
            pieces.append(np.ones(int(rng.integers(1, 6)), dtype=bool))
        return _pack_segments(rng, pieces)
    if kind == "dense":
        width = int(rng.integers(9, 31))
        length = int(rng.integers(5, min(width, 24) + 1))
        offset = int(rng.integers(0, width - length + 1))
        mask = np.zeros(width, dtype=bool)
        mask[offset:offset + length] = True
        candidates = list(range(offset + 1, offset + length - 1))
        rng.shuffle(candidates)
        wanted = int(rng.integers(1, min(4, (length - 1) // 2) + 1))
        holes = []
        for column in candidates:
            if all(abs(column - previous) >= 2 for previous in holes):
                holes.append(column)
                if len(holes) == wanted:
                    break
        mask[holes] = False
        return mask
    if kind == "multi":
        depth = max(depth, 2)
        other_depth = 2 if depth == 2 else int(rng.integers(2, min(4, depth - 1) + 1))
        pieces = [_alternating(depth), _alternating(other_depth)]
        room = 30 - sum(map(len, pieces)) - 4
        if room >= 1 and rng.random() < 0.55:
            pieces.append(np.ones(int(rng.integers(1, min(5, room) + 1)), dtype=bool))
        return _pack_segments(rng, pieces)
    if kind == "mixed":
        solid_length = int(rng.integers(2, min(7, 28 - (2 * depth + 1)) + 1))
        pieces = [_alternating(depth), np.ones(solid_length, dtype=bool)]
        room = 30 - sum(map(len, pieces)) - 4
        if room >= 1 and rng.random() < 0.4:
            pieces.append(np.ones(int(rng.integers(1, min(4, room) + 1)), dtype=bool))
        return _pack_segments(rng, pieces)
    raise ValueError("unknown mask kind")


def _render_output(inp, seed_row, seed_mask, seed_color, odd_color):
    output = inp.copy()
    generations = _generations(seed_mask)
    for depth, mask in enumerate(generations, 1):
        row = seed_row + 2 * depth
        assert row < len(inp)
        output[row, mask] = odd_color if depth % 2 else seed_color
    return output, len(generations)


def _is_count(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(rng, num_train, num_test):
    if num_train is not None and (not _is_count(num_train) or not 2 <= num_train <= 10):
        raise ValueError("num_train must be an integer from 2 through 10")
    if num_test is not None and (not _is_count(num_test) or not 1 <= num_test <= 4):
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _infer_input(inp):
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 2
    background = int(colors[int(np.argmax(counts))])
    assert np.max(counts) > np.min(counts)
    rows = np.flatnonzero(np.any(inp != background, axis=1))
    assert len(rows) == 1
    seed_row = int(rows[0])
    seed_color = int(inp[seed_row, np.flatnonzero(inp[seed_row] != background)[0]])
    seed_mask = inp[seed_row] == seed_color
    assert np.any(~seed_mask)
    assert np.any(_next_mask(seed_mask))
    return background, seed_color, seed_row, seed_mask


def _input_descriptor(inp, background, seed_row):
    return (inp.shape[0], inp.shape[1], background, seed_row,
            "".join(str(int(value)) for value in inp[seed_row]))


def _sample_palette(rng, odd_color, base_background, base_seed, mode,
                    first_background=None, first_seed=None):
    background, seed_color = base_background, base_seed
    if mode in ("background", "both"):
        choices = [color for color in range(10) if color != odd_color
                   and color != base_seed and color != first_background]
        background = int(rng.choice(choices))
    if mode in ("seed", "both"):
        choices = [color for color in range(10) if color != odd_color
                   and color != background and color != first_seed]
        seed_color = int(rng.choice(choices))
    assert len({background, seed_color, odd_color}) == 3
    return background, seed_color


def _generate_pair(rng, regime, palette, demonstration=False):
    if demonstration:
        kind = "mixed"
        target_depth = int(rng.integers(3, 7))
    else:
        choices = ("one", "dense", "pure", "multi", "mixed")
        probabilities = (0.19, 0.19, 0.2, 0.22, 0.2) if regime == "source_train" else (0.14, 0.16, 0.25, 0.27, 0.18)
        kind = str(rng.choice(choices, p=probabilities))
        target_depth = int(rng.integers(1, 7 if regime == "source_train" else 10))
    mask = _make_mask(rng, kind, target_depth)
    depth = len(_generations(mask))
    assert depth >= 1
    max_row = min(10, 28 - 2 * depth)
    seed_row = int(rng.integers(0, max_row + 1))
    max_tail = min(12, 29 - seed_row - 2 * depth)
    tail = int(rng.integers(1, max_tail + 1))
    height = seed_row + 2 * depth + 1 + tail
    background, seed_color, odd_color = palette
    inp = np.full((height, len(mask)), background, dtype=np.int8)
    inp[seed_row, mask] = seed_color
    output, computed_depth = _render_output(inp, seed_row, mask, seed_color, odd_color)
    assert computed_depth == depth
    return {"input": inp, "output": output}


def _grid_signature(grid):
    return (grid.shape, tuple(map(int, grid.ravel())))


def _puzzles_equal(left, right):
    return all(len(left[section]) == len(right[section]) and all(
        np.array_equal(a[key], b[key])
        for a, b in zip(left[section], right[section]) for key in ("input", "output"))
        for section in ("train", "test"))


def _fallback_pair(palette, global_index, seen):
    background, seed_color, odd_color = palette
    height = 15 + global_index
    for row in range(6):
        for shift in range(5):
            mask = np.zeros(29, dtype=bool)
            mask[2 + shift:9 + shift:2] = True
            mask[17:21] = True
            inp = np.full((height, 29), background, dtype=np.int8)
            inp[row, mask] = seed_color
            if _grid_signature(inp) not in seen:
                output, depth = _render_output(
                    inp, row, mask, seed_color, odd_color
                )
                assert depth == 3
                return {"input": inp, "output": output}
    raise RuntimeError("could not construct a unique fallback pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    odd_color, base_background, base_seed = map(int, rng.choice(10, 3, replace=False))
    mode = str(rng.choice(("fixed", "background", "seed", "both"),
                          p=(0.35, 0.3, 0.15, 0.2)))
    puzzle = {"train": [], "test": []}
    seen = set()
    demonstration_index = int(rng.integers(0, train_count))
    first_background = first_seed = None
    for section, count, regime in (("train", train_count, "source_train"),
                                   ("test", test_count, "target_test")):
        for index in range(count):
            for attempt in range(100):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                exclude_background = first_background if section == "train" and index == 1 and mode in ("background", "both") else None
                exclude_seed = first_seed if section == "train" and index == 1 and mode in ("seed", "both") else None
                background, seed_color = _sample_palette(
                    child_rng, odd_color, base_background, base_seed, mode,
                    exclude_background, exclude_seed)
                pair = _generate_pair(child_rng, regime,
                                      (background, seed_color, odd_color),
                                      section == "train" and index == demonstration_index)
                inp = pair["input"]
                _, _, seed_row, _ = _infer_input(inp)
                signature = _grid_signature(inp)
                if signature not in seen and _input_descriptor(inp, background, seed_row) not in _SOURCE_INPUT_DESCRIPTORS:
                    break
            else:
                global_index = index if section == "train" else train_count + index
                pair = _fallback_pair(
                    (background, seed_color, odd_color), global_index, seen
                )
                inp = pair["input"]
                signature = _grid_signature(inp)
            if section == "train" and index == 0:
                first_background, first_seed = background, seed_color
            seen.add(signature)
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    odd_color = None
    training_depths = []
    training_solid = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            input_signature = _grid_signature(inp)
            pair_signature = (input_signature, _grid_signature(out))
            assert input_signature not in seen_inputs
            assert pair_signature not in seen_pairs
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            background, seed_color, seed_row, mask = _infer_input(inp)
            assert _input_descriptor(inp, background, seed_row) not in _SOURCE_INPUT_DESCRIPTORS
            new_colors = set(map(int, np.unique(out))) - {background, seed_color}
            assert len(new_colors) == 1
            pair_odd = new_colors.pop()
            if odd_color is None:
                odd_color = pair_odd
            assert pair_odd == odd_color
            expected, depth = _render_output(inp, seed_row, mask, seed_color, odd_color)
            assert np.array_equal(out, expected)
            assert depth >= 1 and seed_row + 2 * depth < len(inp) - 1
            if section == "train":
                training_depths.append(depth)
                training_solid |= bool(np.any(mask[:-1] & mask[1:]))
    assert max(training_depths) >= 2
    assert training_solid


def _predict_from_training(train, test_input):
    learned_colors = [
        set(map(int, np.unique(pair["output"]))) -
        set(map(int, np.unique(pair["input"])))
        for pair in train
    ]
    assert all(len(colors) == 1 for colors in learned_colors)
    odd_color = next(iter(learned_colors[0]))
    assert all(next(iter(colors)) == odd_color for colors in learned_colors)
    colors, counts = np.unique(test_input, return_counts=True)
    background = int(colors[np.argmax(counts)])
    seed_color = int(colors[np.argmin(counts)])
    seed_row = int(np.flatnonzero(np.any(test_input != background, axis=1))[0])
    active = [cell == seed_color for cell in test_input[seed_row]]
    prediction = test_input.copy()
    generation = 0
    while True:
        following = [False] * len(active)
        for column in range(1, len(active) - 1):
            following[column] = (
                active[column - 1] and not active[column] and active[column + 1]
            )
        if not any(following):
            return prediction
        generation += 1
        row = seed_row + 2 * generation
        assert row < len(prediction)
        for column, present in enumerate(following):
            if present:
                prediction[row, column] = (
                    odd_color if generation % 2 else seed_color
                )
        active = following


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        predictions = [
            _predict_from_training(puzzle["train"], pair["input"])
            for pair in puzzle["test"]
        ]
        assert all(
            np.array_equal(prediction, pair["output"])
            for prediction, pair in zip(predictions, puzzle["test"])
        )
    assert _puzzles_equal(generate_puzzle(90125, 7, 3), generate_puzzle(90125, 7, 3))
    assert _puzzles_equal(generate_puzzle(7123), generate_puzzle(7123))
    assert not _puzzles_equal(generate_puzzle(101), generate_puzzle(102))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (6, 2), (7, 3))):
        puzzle = generate_puzzle(2000 + index, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + index, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4},
                      {"num_train": 2.0, "num_test": 1},
                      {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(17, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count arguments were accepted")
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    widths, depths, shapes, layouts = set(), set(), set(), set()
    one_holes = multi_groups = pure_train = deep_composite_test = 0
    shallow_test = deep_test = edge_contact = long_tail = 0
    variable_backgrounds = variable_seeds = 0
    for puzzle in puzzles:
        backgrounds, seeds = set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp = pair["input"]
                background, seed_color, seed_row, mask = _infer_input(inp)
                generations = _generations(mask)
                depth = len(generations)
                first_count = int(np.count_nonzero(generations[0]))
                shapes.add(inp.shape)
                widths.add(inp.shape[1])
                depths.add(depth)
                layouts.add((inp.shape[1], tuple(np.flatnonzero(mask))))
                backgrounds.add(background)
                seeds.add(seed_color)
                one_holes += first_count == 1
                multi_groups += (len(generations) >= 2 and
                                 np.any(np.diff(np.flatnonzero(generations[1])) > 2))
                edge_contact += bool(mask[0] or mask[-1])
                long_tail += inp.shape[0] - 1 - (seed_row + 2 * depth) >= 8
                pure = (len(np.flatnonzero(mask)) >= 2 and
                        np.all(np.diff(np.flatnonzero(mask)) == 2))
                if section == "train":
                    pure_train += pure
                else:
                    shallow_test += depth == 1
                    deep_test += depth >= 5
                    deep_composite_test += depth >= 5 and not pure
        variable_backgrounds += len(backgrounds) > 1
        variable_seeds += len(seeds) > 1
    assert len(widths) >= 15 and len(shapes) >= 45 and len(layouts) >= 100
    assert min(depths) == 1 and max(depths) >= 7
    assert one_holes >= 15 and multi_groups >= 8
    assert pure_train >= 4 and deep_composite_test >= 3
    assert shallow_test >= 3 and deep_test >= 5
    assert edge_contact >= 20 and long_tail >= 15
    assert variable_backgrounds >= 5 and variable_seeds >= 3
