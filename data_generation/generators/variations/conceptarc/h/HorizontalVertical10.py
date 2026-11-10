import numpy as np


_SOURCE_INPUTS = {
    ("H", 4, 3, 3, 3, (4, 2, 3)),
    ("V", 3, 2, 4, 4, (7, 8, 9, 1)),
    ("H", 5, 0, 4, 5, (3, 4, 5, 1)),
    ("V", 5, 3, 3, 2, (2, 3)),
    ("H", 6, 2, 2, 7, (4, 6)),
}


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError("Invalid pair count")
    if num_train is None:
        low = 2 if num_test is None else max(2, int(num_test) + 1)
        num_train = int(rng.integers(low, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("Training count must exceed test count")
    return int(num_train), int(num_test)


def _input_info(grid):
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    assert int(counts.max()) > grid.size // 2
    points = np.argwhere(grid != background)
    assert len(points) > 0
    top, left = (int(v) for v in points.min(axis=0))
    bottom, right = (int(v) + 1 for v in points.max(axis=0))
    block = grid[top:bottom, left:right]
    assert block.shape[0] >= 2 and block.shape[1] >= 2
    assert len(points) == block.size and np.all(block != background)
    horizontal = bool(np.all(block == block[:, :1]) and
                      len(np.unique(block[:, 0])) >= 2)
    vertical = bool(np.all(block == block[:1, :]) and
                    len(np.unique(block[0, :])) >= 2)
    assert horizontal != vertical
    orientation = "H" if horizontal else "V"
    stripe = block[:, 0] if horizontal else block[0, :]
    bands = tuple(int(v) for i, v in enumerate(stripe)
                  if i == 0 or v != stripe[i - 1])
    assert len(bands) >= 2
    if horizontal:
        assert np.all(grid[:, -1] == background)
    else:
        assert np.all(grid[0, :] == background)
    return (orientation, background, top, left, block.shape[0], block.shape[1],
            bands, tuple(int(v) for v in stripe))


def _make_bands(rng, count, colors, guide, background, regime, train_colors,
                opposite_colors):
    allowed = [c for c in colors if c != background]
    assert len(allowed) >= 3
    bands = []
    for index in range(count):
        options = [c for c in allowed if not bands or c != bands[-1]]
        if index >= 2 and rng.random() < 0.23 and bands[-2] in options:
            bands.append(bands[-2])
        else:
            bands.append(int(rng.choice(options)))
    assignments = []
    if guide in allowed and rng.random() < 0.28:
        assignments.append(guide)
    if regime == "target_test":
        fresh = [c for c in allowed if c not in train_colors]
        if fresh and rng.random() < 0.66:
            assignments.append(int(rng.choice(fresh)))
        reused = [c for c in opposite_colors if c in allowed]
        if reused and rng.random() < 0.57:
            assignments.append(int(rng.choice(reused)))
    slots = list(rng.permutation(count))
    for color, slot in zip(assignments, slots):
        bands[int(slot)] = color
    for index in range(1, count):
        if bands[index] == bands[index - 1]:
            options = [c for c in allowed if c != bands[index - 1] and
                       (index + 1 == count or c != bands[index + 1])]
            bands[index] = int(rng.choice(options))
    return bands


def _draw(rng, orientation, guide, background, train_palette, regime,
          train_colors, opposite_colors, classic=False, fallback_size=None):
    if fallback_size is not None:
        canvas_h, canvas_w = fallback_size, 19
    elif classic:
        canvas_h = canvas_w = 10
    else:
        canvas_h = int(rng.integers(7, 19))
        canvas_w = int(rng.integers(7, 19))
    axis_capacity = canvas_h if orientation == "H" else canvas_w
    count_max = 4 if classic and fallback_size is None else 6
    count = int(rng.integers(2, min(count_max, axis_capacity) + 1))
    if classic or fallback_size is not None:
        runs = [1] * count
    else:
        runs = [int(rng.integers(2, 4)) if rng.random() < 0.16 else 1
                for _ in range(count)]
        while sum(runs) > min(10, axis_capacity):
            longer = [i for i, run in enumerate(runs) if run > 1]
            runs[int(rng.choice(longer))] -= 1
    axis_length = sum(runs)
    cross_capacity = canvas_w - 1 if orientation == "H" else canvas_h - 1
    area_limit = max(2, int(0.42 * canvas_h * canvas_w // axis_length))
    cross_max = min(11, cross_capacity, area_limit)
    if classic and regime == "source_train" and fallback_size is None:
        cross_length = axis_length
    elif classic and regime == "target_test" and fallback_size is None:
        classic_max = min(cross_max, 5 if orientation == "V" else 8)
        cross_length = int(rng.integers(axis_length + 1, classic_max + 1))
    else:
        cross_length = int(rng.integers(2, cross_max + 1))
    height = axis_length if orientation == "H" else cross_length
    width = cross_length if orientation == "H" else axis_length
    if classic and regime == "target_test" and fallback_size is None:
        top = int(rng.integers(5, canvas_h - height + 1))
    else:
        top_low = 0 if orientation == "H" else 1
        top = int(rng.integers(top_low, canvas_h - height + 1))
    left_high = canvas_w - width - (1 if orientation == "H" else 0)
    left = int(rng.integers(0, left_high + 1))
    palette = train_palette if regime == "source_train" else tuple(range(10))
    bands = _make_bands(rng, count, palette, guide, background, regime,
                        train_colors, opposite_colors)
    stripe = [color for color, run in zip(bands, runs) for _ in range(run)]
    inp = np.full((canvas_h, canvas_w), background, dtype=np.int8)
    if orientation == "H":
        for offset, color in enumerate(stripe):
            inp[top + offset, left:left + width] = color
    else:
        for offset, color in enumerate(stripe):
            inp[top:top + height, left + offset] = color
    out = inp.copy()
    if orientation == "H":
        out[:, -1] = guide
    else:
        out[0, :] = guide
    return {"input": inp, "output": out}


def _key(grid):
    return (grid.shape, grid.tobytes())


def _source_copy(info, shape):
    orientation, background, top, left, height, width, _, stripe = info
    return (shape == (10, 10) and background == 0 and
            (orientation, top, left, height, width, stripe) in _SOURCE_INPUTS)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    classic = bool(rng.random() < 0.13)
    guide = 8 if classic or rng.random() < 0.42 else int(rng.choice([1, 2, 3, 4, 5, 6, 7, 9]))
    palette_size = int(rng.integers(5, 8))
    train_palette = tuple(int(v) for v in rng.choice(10, size=palette_size, replace=False))
    if classic:
        train_palette = tuple(range(1, 10))
    first = "H" if rng.random() < 0.5 else "V"
    train_directions = [first, "V" if first == "H" else "H"]
    train_directions += ["H" if rng.random() < 0.5 else "V"
                         for _ in range(num_train - 2)]
    test_directions = ["H" if rng.random() < 0.5 else "V"
                       for _ in range(num_test)]
    if num_test >= 2 and len(set(test_directions)) == 1:
        test_directions[int(rng.integers(num_test))] = (
            "V" if test_directions[0] == "H" else "H")
    puzzle = {"train": [], "test": []}
    seen = set()
    train_colors = set()
    colors_by_direction = {"H": set(), "V": set()}
    serial = 0
    for split, directions in (("train", train_directions), ("test", test_directions)):
        for orientation in directions:
            other = "V" if orientation == "H" else "H"
            for attempt in range(33):
                background = 0 if classic or rng.random() < 0.53 else int(
                    rng.choice([c for c in range(10) if c != guide]))
                if background == guide:
                    background = 0
                pair = _draw(rng, orientation, guide, background, train_palette,
                             "source_train" if split == "train" else "target_test",
                             train_colors, colors_by_direction[other], classic,
                             17 + serial if attempt == 32 else None)
                info = _input_info(pair["input"])
                key = _key(pair["input"])
                if key not in seen and not _source_copy(info, pair["input"].shape):
                    break
            else:
                raise AssertionError("Deterministic fallback failed")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                colors_by_direction[orientation].update(info[6])
                train_colors.update(info[6])
            serial += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict(train_pairs, test_input):
    guides = set()
    demonstrations = set()
    for pair in train_pairs:
        info = _input_info(pair["input"])
        orientation = info[0]
        edge = pair["output"][:, -1] if orientation == "H" else pair["output"][0, :]
        assert np.all(edge == edge[0])
        guides.add(int(edge[0]))
        demonstrations.add(orientation)
    assert len(guides) == 1 and demonstrations == {"H", "V"}
    guide = guides.pop()
    orientation = _input_info(test_input)[0]
    answer = test_input.copy()
    if orientation == "H":
        answer[:, -1] = guide
    else:
        answer[0, :] = guide
    return answer


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    guide_colors = set()
    train_directions = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            assert inp.shape == out.shape
            info = _input_info(inp)
            orientation, background = info[:2]
            guide = int(out[0, -1])
            edge = out[:, -1] if orientation == "H" else out[0, :]
            assert np.all(edge == guide) and guide != background
            guide_colors.add(guide)
            expected = inp.copy()
            if orientation == "H":
                expected[:, -1] = guide
            else:
                expected[0, :] = guide
            assert np.array_equal(out, expected)
            assert not np.array_equal(inp, out)
            assert not _source_copy(info, inp.shape)
            input_key, output_key = _key(inp), _key(out)
            assert input_key not in seen_inputs
            assert (input_key, output_key) not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            if split == "train":
                train_directions.add(orientation)
    assert len(guide_colors) == 1
    assert train_directions == {"H", "V"}
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(puzzle["train"], pair["input"]), pair["output"])


def _equal_puzzles(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[role], b[role])
                   for a, b in zip(first[split], second[split])
                   for role in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, guides = set(), set(), set()
    shapes, backgrounds, layouts, band_counts = set(), set(), set(), set()
    aspect_by_direction = {"H": set(), "V": set()}
    aspect_by_split_direction = {(s, d): set() for s in ("train", "test")
                                 for d in ("H", "V")}
    orientations_by_count = {n: set() for n in range(2, 7)}
    saw_thick = saw_repeat = saw_guide_band = False
    saw_novel_test_color = saw_cross_direction_reuse = False
    saw_square_test = saw_nonsquare_train = False
    saw_10x10_08 = saw_varied_size = saw_varied_background = False
    measured = {name: 0 for name in (
        "thick", "repeat", "guide_band", "new_test_color", "cross_reuse",
        "square_test", "nonsquare_train", "touch_top", "touch_bottom",
        "touch_left", "touch_right", "varied_size", "varied_background",
        "varied_top", "varied_left")}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(tuple(_key(p["input"]) for p in puzzle[s])
                             for s in ("train", "test")))
        train_infos = [_input_info(p["input"]) for p in puzzle["train"]]
        colors_by_direction = {
            d: {c for i in train_infos if i[0] == d for c in i[6]}
            for d in ("H", "V")
        }
        train_colors = colors_by_direction["H"] | colors_by_direction["V"]
        guide = int(puzzle["train"][0]["output"][0, -1])
        guides.add(guide)
        puzzle_shapes, puzzle_backgrounds = set(), set()
        puzzle_tops, puzzle_lefts = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                info = _input_info(inp)
                orientation, bg, top, left, height, width, bands, stripe = info
                assert np.array_equal(_predict(puzzle["train"], inp), pair["output"])
                assert not _source_copy(info, inp.shape)
                shapes.add(inp.shape)
                backgrounds.add(bg)
                layouts.add((inp.shape, top, left, height, width, orientation))
                band_counts.add(len(bands))
                orientations_by_count[len(bands)].add((split, orientation))
                aspect_by_direction[orientation].add(
                    "square" if height == width else "wide" if width > height else "tall")
                aspect_by_split_direction[(split, orientation)].add(
                    "square" if height == width else "wide" if width > height else "tall")
                saw_thick |= len(stripe) > len(bands)
                saw_repeat |= len(bands) > len(set(bands))
                saw_guide_band |= guide in bands
                measured["thick"] += len(stripe) > len(bands)
                measured["repeat"] += len(bands) > len(set(bands))
                measured["guide_band"] += guide in bands
                measured["touch_top"] += top == 0
                measured["touch_bottom"] += top + height == inp.shape[0]
                measured["touch_left"] += left == 0
                measured["touch_right"] += left + width == inp.shape[1]
                saw_10x10_08 |= inp.shape == (10, 10) and bg == 0 and guide == 8
                puzzle_shapes.add(inp.shape)
                puzzle_backgrounds.add(bg)
                puzzle_tops.add(top)
                puzzle_lefts.add(left)
                if split == "train":
                    saw_nonsquare_train |= height != width
                    measured["nonsquare_train"] += height != width
                else:
                    saw_square_test |= height == width
                    measured["square_test"] += height == width
                    saw_novel_test_color |= bool(set(bands) - train_colors)
                    measured["new_test_color"] += bool(set(bands) - train_colors)
                    other = "V" if orientation == "H" else "H"
                    saw_cross_direction_reuse |= bool(
                        (set(bands) & colors_by_direction[other]) -
                        colors_by_direction[orientation])
                    measured["cross_reuse"] += bool(
                        (set(bands) & colors_by_direction[other]) -
                        colors_by_direction[orientation])
        saw_varied_size |= len(puzzle_shapes) > 1
        saw_varied_background |= len(puzzle_backgrounds) > 1
        measured["varied_size"] += len(puzzle_shapes) > 1
        measured["varied_background"] += len(puzzle_backgrounds) > 1
        measured["varied_top"] += len(puzzle_tops) > 1
        measured["varied_left"] += len(puzzle_lefts) > 1
    assert len(signatures) >= 40
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(guides) >= 4 and 8 in guides
    assert len(shapes) >= 35 and len(backgrounds) >= 5 and len(layouts) >= 100
    assert {2, 3, 4, 5, 6} <= band_counts
    assert all({("train", "H"), ("train", "V"), ("test", "H"), ("test", "V")}
               <= orientations_by_count[n] for n in (2, 3, 4))
    assert all({"square", "wide", "tall"} <= aspect_by_direction[d]
               for d in ("H", "V"))
    assert all({"square", "wide", "tall"} <= values
               for values in aspect_by_split_direction.values())
    assert all((saw_thick, saw_repeat, saw_guide_band, saw_novel_test_color,
                saw_cross_direction_reuse, saw_square_test, saw_nonsquare_train,
                saw_10x10_08, saw_varied_size, saw_varied_background))
    minimums = {"thick": 40, "repeat": 40, "guide_band": 40,
                "new_test_color": 20, "cross_reuse": 8,
                "square_test": 4, "nonsquare_train": 60,
                "touch_top": 8, "touch_bottom": 20,
                "touch_left": 20, "touch_right": 8,
                "varied_size": 25, "varied_background": 25,
                "varied_top": 25, "varied_left": 25}
    assert all(measured[name] >= bound for name, bound in minimums.items())
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                    (None, 4), (2, None), (10, None))):
        first = generate_puzzle(100 + index, *counts)
        validate_generated_puzzle(first)
        assert _equal_puzzles(first, generate_puzzle(100 + index, *counts))
        for actual, requested in zip((len(first["train"]), len(first["test"])), counts):
            assert requested is None or actual == requested
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
    demonstrations = generate_puzzle(984, 2, 1)["train"]
    guide = int(demonstrations[0]["output"][0, -1])
    horizontal_tall = np.zeros((5, 3), dtype=np.int8)
    horizontal_tall[1:4, 0:2] = np.array([1, 2, 3], dtype=np.int8)[:, None]
    horizontal_output = horizontal_tall.copy()
    horizontal_output[:, -1] = guide
    vertical_wide = np.zeros((3, 5), dtype=np.int8)
    vertical_wide[1:3, 1:4] = np.array([1, 2, 3], dtype=np.int8)[None, :]
    vertical_output = vertical_wide.copy()
    vertical_output[0, :] = guide
    for inp, out in ((horizontal_tall, horizontal_output),
                     (vertical_wide, vertical_output)):
        validate_generated_puzzle({"train": demonstrations,
                                   "test": [{"input": inp, "output": out}]})
    broken = generate_puzzle(987, 2, 1)
    broken["train"][0]["output"][0, 0] = (
        int(broken["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted an altered output")
