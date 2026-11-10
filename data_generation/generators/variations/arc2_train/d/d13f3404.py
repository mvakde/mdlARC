import numpy as np


_SOURCE_INPUT_SIGNATURES = {
    ((3, 3), (0, 4, 0, 0, 8, 0, 2, 0, 0)),
    ((3, 3), (6, 1, 0, 3, 0, 0, 0, 0, 0)),
    ((3, 3), (0, 0, 6, 1, 3, 0, 0, 0, 0)),
    ((3, 3), (0, 0, 3, 0, 0, 0, 0, 4, 9)),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _signature(grid: np.ndarray) -> tuple:
    return tuple(map(int, grid.shape)), tuple(map(int, grid.flat))


def _pair_signature(pair: dict) -> tuple:
    return _signature(pair["input"]), _signature(pair["output"])


def _background(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    _require(counts[background] > grid.size // 2, "background is not a strict majority")
    return background


def _ray_output(input_grid: np.ndarray, background: int) -> np.ndarray:
    height, width = input_grid.shape
    output = np.full((2 * height, 2 * width), background, dtype=np.int8)
    for row, col in np.argwhere(input_grid != background):
        row, col = int(row), int(col)
        length = min(2 * height - row, 2 * width - col)
        indices = np.arange(length)
        output[row + indices, col + indices] = input_grid[row, col]
    return output


def _independent_output(input_grid: np.ndarray, background: int) -> np.ndarray:
    height, width = input_grid.shape
    result = np.full((2 * height, 2 * width), background, dtype=np.int8)
    for row in range(2 * height):
        for col in range(2 * width):
            for distance in range(min(row, col) + 1):
                source_row, source_col = row - distance, col - distance
                if source_row < height and source_col < width:
                    value = int(input_grid[source_row, source_col])
                    if value != background:
                        result[row, col] = value
                        break
    return result


def _validate_count(value: int | None, name: str, low: int, high: int) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _pair_counts(rng: np.random.Generator, num_train: int | None,
                 num_test: int | None) -> tuple[int, int]:
    _validate_count(num_train, "num_train", 2, 10)
    _validate_count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None:
        if int(num_train) <= int(num_test):
            raise ValueError("num_train must exceed num_test")
        return int(num_train), int(num_test)
    legal = [(t, s) for t in range(2, 11) for s in range(1, min(4, t - 1) + 1)]
    legal = [(t, s) for t, s in legal
             if (num_train is None or t == int(num_train))
             and (num_test is None or s == int(num_test))]
    if not legal:
        raise ValueError("no legal pair count combination")
    return legal[int(rng.integers(len(legal)))]


def _sample_shape(rng: np.random.Generator, forbidden: set[tuple[int, int]],
                  minimum_height: int = 2, minimum_width: int = 2) -> tuple[int, int]:
    for _ in range(64):
        mode = int(rng.choice(7, p=[.12, .16, .30, .20, .145, .05, .025]))
        if mode == 0:
            shape = (3, 3)
        elif mode == 1:
            shape = (int(rng.integers(2, 5)), int(rng.integers(2, 6)))
        elif mode == 2:
            shape = (int(rng.integers(4, 9)), int(rng.integers(4, 9)))
        elif mode == 3:
            short, long = int(rng.integers(2, 6)), int(rng.integers(9, 16))
            shape = (short, long) if rng.random() < .5 else (long, short)
        elif mode == 4:
            shape = (int(rng.integers(9, 16)), int(rng.integers(9, 16)))
        elif mode == 5:
            shape = (int(rng.integers(2, 16)), int(rng.integers(2, 16)))
        else:
            shape = (15, 15)
        if (shape[0] * shape[1] >= 5
                and shape[0] >= minimum_height and shape[1] >= minimum_width
                and shape not in forbidden):
            return shape
    for height in range(minimum_height, 16):
        for width in range(minimum_width, 16):
            if height * width >= 5 and (height, width) not in forbidden:
                return height, width
    raise RuntimeError("no unused shape remains")


def _layout_options(height: int, width: int, kind: str) -> list[tuple[tuple, list]]:
    upper = [(r, c) for r in range(height - 1) for c in range(width)]
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    if kind == "adjacent_bottom":
        return [(((height - 1, c), (height - 1, c + 1)), upper)
                for c in range(width - 1)]
    if kind == "separated_bottom":
        return [(((height - 1, a), (height - 1, b)), upper)
                for a in range(width) for b in range(a + 2, width)]
    if kind == "one_bottom":
        return [(((height - 1, c),), upper) for c in range(width)]
    if kind == "zero_bottom":
        return [((), upper)]
    if kind == "no_top":
        return [((), [(r, c) for r, c in all_cells if r > 0])]
    if kind == "general":
        return [((), all_cells)]
    raise ValueError("unknown layout kind")


def _positions(rng: np.random.Generator, height: int, width: int,
               count: int, kind: str, top: bool = False,
               witness: bool = False, corner: bool = False) -> list[tuple[int, int]]:
    options = _layout_options(height, width, kind)
    requirements = []
    if top:
        requirements.append(lambda cell: cell[0] == 0)
    if witness:
        requirements.append(lambda cell: cell[0] > 0 and cell[1] > 0)
    if corner:
        requirements.append(lambda cell: cell[0] in (0, height - 1)
                            and cell[1] in (0, width - 1))

    for option_index in rng.permutation(len(options)):
        forced, free = options[int(option_index)]
        cells = list(dict.fromkeys(list(forced) + free))
        by_offset = {}
        for cell in cells:
            by_offset.setdefault(cell[1] - cell[0], []).append(cell)
        selected = {cell[1] - cell[0]: cell for cell in forced}
        if len(selected) != len(forced) or len(by_offset) < count:
            continue

        def fill_requirement(index: int) -> bool:
            if index == len(requirements):
                return len(selected) <= count and len(by_offset) >= count
            candidates = [cell for cell in cells if requirements[index](cell)]
            for candidate_index in rng.permutation(len(candidates)):
                cell = candidates[int(candidate_index)]
                offset = cell[1] - cell[0]
                previous = selected.get(offset)
                if previous is not None and previous != cell:
                    continue
                selected[offset] = cell
                if len(selected) <= count and fill_requirement(index + 1):
                    return True
                if previous is None:
                    del selected[offset]
            return False

        if not fill_requirement(0):
            continue
        remaining = [offset for offset in by_offset if offset not in selected]
        extras = count - len(selected)
        if extras:
            for offset in rng.choice(remaining, size=extras, replace=False):
                choices = by_offset[int(offset)]
                selected[int(offset)] = choices[int(rng.integers(len(choices)))]
        return list(selected.values())
    raise RuntimeError("no layout meets its constraints")


def _max_seeds(height: int, width: int, kind: str) -> int:
    capacity = max(len({c - r for r, c in list(forced) + free})
                   for forced, free in _layout_options(height, width, kind))
    return min(16, capacity, (height * width - 1) // 2)


def _sample_count(rng: np.random.Generator, maximum: int, regime: str) -> int:
    if regime == "first_train":
        return int(rng.integers(3, min(6, maximum) + 1))
    draw = rng.random()
    if draw < .18:
        target = 1
    elif draw < .38:
        target = 2
    elif draw < .76:
        target = int(rng.integers(3, 7))
    else:
        target = int(rng.integers(7, 17))
    return min(target, maximum)


def _colors(rng: np.random.Generator, count: int, background: int,
            earlier: list[list[int]], force_zero: bool,
            repeat: bool, reuse: list[int] | None,
            distinct: bool) -> list[int]:
    if reuse is not None and len(reuse) == count:
        colors = list(reuse)
    else:
        available = [color for color in range(10) if color != background]
        if distinct:
            if force_zero:
                colors = [0] + list(map(int, rng.choice(
                    [color for color in available if color != 0],
                    size=count - 1, replace=False)))
            else:
                colors = list(map(int, rng.choice(available, size=count,
                                                  replace=False)))
            rng.shuffle(colors)
            return colors
        if earlier and rng.random() < .55:
            preferred = sorted({color for palette in earlier for color in palette})
            available = preferred + [color for color in available if color not in preferred]
            pool = available[:max(2, min(len(available), int(rng.integers(2, 6))))]
        else:
            pool = list(rng.choice(available, size=int(rng.integers(2, 7)), replace=False))
        colors = [int(pool[int(rng.integers(len(pool)))]) for _ in range(count)]
        if force_zero:
            colors[0] = 0
        if repeat and count >= 2:
            colors[-1] = colors[0]
        if count >= 3 and len(set(colors)) == 1:
            colors[-1] = next(color for color in available if color != colors[0])
    rng.shuffle(colors)
    return colors


def _make_pair(rng: np.random.Generator, shape: tuple[int, int], count: int,
               kind: str, background: int, earlier: list[list[int]],
               used: set, first_train: bool = False,
               reuse: list[int] | None = None) -> dict:
    height, width = shape
    force_zero = first_train and background != 0
    for attempt in range(48):
        try:
            positions = _positions(rng, height, width, count, kind,
                                   top=first_train, witness=first_train,
                                   corner=(not first_train and attempt == 0
                                           and kind in ("general", "zero_bottom", "no_top")
                                           and rng.random() < .32))
        except RuntimeError:
            continue
        colors = _colors(rng, count, background, earlier, force_zero,
                         repeat=(not first_train and rng.random() < .32),
                         reuse=reuse, distinct=first_train)
        input_grid = np.full(shape, background, dtype=np.int8)
        for (row, col), color in zip(positions, colors):
            input_grid[row, col] = color
        signature = _signature(input_grid)
        if signature in used or signature in _SOURCE_INPUT_SIGNATURES:
            continue
        used.add(signature)
        return {"input": input_grid,
                "output": _ray_output(input_grid, background)}
    raise RuntimeError("could not make a unique pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _pair_counts(rng, num_train, num_test)
    background = 0 if rng.random() < .70 else int(rng.integers(1, 10))
    simple = int(seed) % 7 == 0
    used = set()
    train, test = [], []
    train_shapes = set()
    earlier = []
    first_colors = None

    for index in range(train_count):
        if index == 0:
            kind = "general"
        elif index == 1:
            kind = "adjacent_bottom" if train_count >= 3 or rng.random() < .5 else "separated_bottom"
        elif index == 2:
            kind = "separated_bottom"
        else:
            kind = ("general", "zero_bottom", "one_bottom", "no_top",
                    "adjacent_bottom", "separated_bottom")[int(rng.integers(6))]
        if simple:
            shape = (3, 3)
        else:
            shape = _sample_shape(rng, train_shapes if index == 1 else set(),
                                  minimum_height=3 if index == 0 else 2,
                                  minimum_width=3 if index == 0 or kind == "separated_bottom" else 2)
        maximum = _max_seeds(*shape, kind)
        if index == 0:
            count = 3 if simple else _sample_count(rng, maximum, "first_train")
        elif simple:
            count = 3
        elif index == 1:
            count = 2
        elif index == 2 and first_colors is not None and len(first_colors) <= maximum:
            count = len(first_colors)
        else:
            count = _sample_count(rng, maximum, "train")
        if kind in ("adjacent_bottom", "separated_bottom"):
            count = max(2, count)
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        reuse = first_colors if index == 2 and len(first_colors) == count else None
        pair = _make_pair(pair_rng, shape, count, kind, background, earlier, used,
                          first_train=(index == 0), reuse=reuse)
        train.append(pair)
        train_shapes.add(shape)
        colors = [int(v) for v in pair["input"][pair["input"] != background]]
        earlier.append(colors)
        if index == 0:
            first_colors = colors

    for index in range(test_count):
        kind = ("adjacent_bottom", "separated_bottom", "one_bottom",
                "zero_bottom", "no_top")[int(rng.integers(5))]
        if simple:
            shape = (3, 3)
        else:
            shape = _sample_shape(rng, train_shapes,
                                  minimum_width=3 if kind == "separated_bottom" else 2)
        maximum = _max_seeds(*shape, kind)
        count = 3 if simple else _sample_count(rng, maximum, "test")
        if kind in ("adjacent_bottom", "separated_bottom"):
            count = max(2, count)
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        reuse = first_colors if first_colors is not None and len(first_colors) == count and rng.random() < .35 else None
        pair = _make_pair(pair_rng, shape, count, kind, background, earlier, used,
                          reuse=reuse)
        test.append(pair)

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    backgrounds = set()
    training_witness = False
    training_rays = 0
    training_colors = set()
    train_shapes = set()
    test_shapes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain exactly input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must use integers")
                _require(all(1 <= side <= 30 for side in grid.shape), "grid size out of range")
                _require(0 <= int(grid.min()) and int(grid.max()) <= 9,
                         "grid colors out of range")
            height, width = inp.shape
            _require(out.shape == (2 * height, 2 * width), "output must double both dimensions")
            _require(inp.dtype == out.dtype, "input and output dtypes differ")
            background = _background(inp)
            backgrounds.add(background)
            positions = [(int(r), int(c)) for r, c in np.argwhere(inp != background)]
            _require(1 <= len(positions) <= 16, "invalid ray count")
            _require(len(positions) < inp.size - len(positions), "background not dominant")
            offsets = [c - r for r, c in positions]
            _require(len(set(offsets)) == len(offsets), "conflicting rays on a diagonal")
            _require(np.array_equal(out, _independent_output(inp, background)),
                     "output violates southeast forward ray rule")
            _require(not np.array_equal(inp, out), "input equals output")
            input_signature, pair_signature = _signature(inp), _pair_signature(pair)
            _require(input_signature not in seen_inputs and pair_signature not in seen_pairs,
                     "duplicate input or pair")
            _require(input_signature not in _SOURCE_INPUT_SIGNATURES, "exact source copy")
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            if split == "train":
                train_shapes.add(inp.shape)
                training_rays += len(positions)
                training_colors.update(int(inp[row, col]) for row, col in positions)
                for row, col in positions:
                    if row > 0 and col > 0 and out[row - 1, col - 1] == background:
                        training_witness = True
            else:
                test_shapes.add(inp.shape)
    _require(len(backgrounds) == 1, "background convention changes within puzzle")
    _require(training_witness, "training lacks a forward-only witness")
    _require(training_rays >= 4, "training does not show several independent rays")
    _require(len(training_colors) >= 3, "training lacks diverse ray colors")
    background = next(iter(backgrounds))
    if background != 0 and any(np.any(pair["input"] == 0) for pair in puzzle["test"]):
        _require(any(np.any(pair["input"] == 0) for pair in puzzle["train"]),
                 "training does not demonstrate zero as a seed color")
    if len(train_shapes) > 1:
        _require(test_shapes - train_shapes, "variable-size puzzle lacks a novel test size")


def _puzzles_equal(left: dict, right: dict) -> bool:
    return all(len(left[split]) == len(right[split]) and
               all(np.array_equal(a[key], b[key])
                   for a, b in zip(left[split], right[split])
                   for key in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    _require(_puzzles_equal(generate_puzzle(99173, 8, 4), generate_puzzle(99173, 8, 4)),
             "explicit calls are not deterministic")
    _require(_puzzles_equal(generate_puzzle(77231), generate_puzzle(77231)),
             "automatic calls are not deterministic")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(41000 + index, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts changed")
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(51000 + index, train, test)
        _require((train is None or len(puzzle["train"]) == train) and
                 (test is None or len(puzzle["test"]) == test), "partial count changed")
    for index, (train, test) in enumerate(((1, None), (11, None), (None, 0),
                                           (None, 5), (2, 2), (3, 4),
                                           (True, 1), (3, False), (2.5, 1))):
        try:
            generate_puzzle(61000 + index, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal count was accepted")

    all_pairs = [pair for puzzle in puzzles for split in ("train", "test")
                 for pair in puzzle[split]]
    shapes = {pair["input"].shape for pair in all_pairs}
    counts = {int(np.count_nonzero(pair["input"] != _background(pair["input"])))
              for pair in all_pairs}
    layouts = {(pair["input"].shape,
                tuple(map(tuple, np.argwhere(pair["input"] != _background(pair["input"])))))
               for pair in all_pairs}
    _require(len(shapes) >= 20, "too few shapes")
    _require(len(counts) >= 7 and 1 in counts and 2 in counts, "too few ray counts")
    _require(len(layouts) >= 100, "too few layouts")
    _require(any(len({pair["input"].shape for pair in puzzle["train"]}) > 1
                 for puzzle in puzzles), "no within-puzzle shape variation")
    _require(any(len({int(np.count_nonzero(pair["input"] != _background(pair["input"])))
                      for pair in puzzle["train"]}) > 1 for puzzle in puzzles),
             "no within-puzzle count variation")
    _require(any(_background(pair["input"]) != 0 for pair in all_pairs),
             "nonzero backgrounds absent")

    train_bottom, test_bottom = set(), set()
    test_without_top = 0
    repeated_color_pairs = 0
    corner_pairs = interior_pairs = small_pairs = large_pairs = 0
    palette_reuse_puzzles = combined_puzzles = simple_puzzles = 0
    clip_sides = set()
    maximum_density = 0.0
    for puzzle in puzzles:
        all_local = puzzle["train"] + puzzle["test"]
        bg = _background(all_local[0]["input"])
        palettes = []
        local_repeat = False
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        train_counts = {int(np.count_nonzero(pair["input"] != bg))
                        for pair in puzzle["train"]}
        if train_shapes == {(3, 3)} and train_counts == {3} and all(
                pair["input"].shape == (3, 3) for pair in puzzle["test"]):
            simple_puzzles += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                height, width = inp.shape
                positions = [(int(row), int(col))
                             for row, col in np.argwhere(inp != bg)]
                colors = [int(inp[row, col]) for row, col in positions]
                palettes.append(tuple(sorted(set(colors))))
                if len(set(colors)) < len(colors):
                    repeated_color_pairs += 1
                    local_repeat = True
                maximum_density = max(maximum_density,
                                      len(positions) / (height * width))
                small_pairs += height == 2 or width == 2
                large_pairs += height == 15 or width == 15
                corner_pairs += any(row in (0, height - 1)
                                    and col in (0, width - 1)
                                    for row, col in positions)
                interior_pairs += any(0 < row < height - 1
                                      and 0 < col < width - 1
                                      for row, col in positions)
                bottom_cols = sorted(col for row, col in positions
                                     if row == height - 1)
                if not bottom_cols:
                    category = "zero"
                elif len(bottom_cols) == 1:
                    category = "one"
                elif len(bottom_cols) == 2 and bottom_cols[1] == bottom_cols[0] + 1:
                    category = "adjacent"
                else:
                    category = "separated_or_many"
                (train_bottom if split == "train" else test_bottom).add(category)
                if split == "test" and not any(row == 0 for row, _ in positions):
                    test_without_top += 1
                for row, col in positions:
                    vertical, horizontal = 2 * height - row, 2 * width - col
                    clip_sides.add("bottom" if vertical < horizontal else
                                   "right" if horizontal < vertical else "corner")
        if len(set(palettes)) < len(palettes):
            palette_reuse_puzzles += 1
        if bg != 0 and len(train_shapes) > 1 and local_repeat:
            combined_puzzles += 1
    _require(simple_puzzles >= 4, "too few uniform source-scale puzzles")
    _require(train_bottom >= {"adjacent", "separated_or_many"},
             "training lacks both bottom-start layouts")
    _require(test_bottom >= {"zero", "one", "adjacent", "separated_or_many"},
             "test bottom-start layouts lack diversity")
    _require(test_without_top >= 8, "tests rarely omit top-row seeds")
    _require(small_pairs >= 5 and large_pairs >= 5 and maximum_density >= .30,
             "size or density spectrum is too narrow")
    _require(corner_pairs >= 10 and interior_pairs >= 10,
             "corner or interior starts are scarce")
    _require(clip_sides >= {"bottom", "right"}, "one clipping direction is absent")
    _require(repeated_color_pairs >= 15 and palette_reuse_puzzles >= 10,
             "color reuse lacks coverage")
    _require(combined_puzzles >= 2, "compatible variations do not combine")
