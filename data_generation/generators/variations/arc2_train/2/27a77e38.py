import numpy as np


_SOURCE_TEXT_PAIRS = (
    (("36424", "84334", "55555", "00000", "00000"),
     ("36424", "84334", "55555", "00000", "00400")),
    (("223", "555", "000"), ("223", "555", "020")),
    (("1996184", "4678971", "9314136", "5555555", "0000000", "0000000", "0000000"),
     ("1996184", "4678971", "9314136", "5555555", "0000000", "0000000", "0001000")),
    (("912849821", "443127679", "216978436", "986342917", "555555555",
      "000000000", "000000000", "000000000", "000000000"),
     ("912849821", "443127679", "216978436", "986342917", "555555555",
      "000000000", "000000000", "000000000", "000090000")),
)
_SOURCE_PAIRS = frozenset(
    (tuple(tuple(map(int, row)) for row in a), tuple(tuple(map(int, row)) for row in b))
    for a, b in _SOURCE_TEXT_PAIRS
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _signature(grid):
    return tuple(tuple(int(x) for x in row) for row in grid)


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _resolve_counts(seed, num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not lo <= value <= hi):
            raise ValueError(f"{name} outside legal range")
    train, test = num_train, num_test
    code = abs(int(seed))
    if train is None and test is None:
        test = 1 + code % 4
        train = test + 1 + (code // 4) % (10 - test)
    elif train is None:
        train = test + 1 + code % (10 - test)
    elif test is None:
        test = 1 + code % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return int(train), int(test)


def _parse(input_grid):
    """Find the single content/divider/blank decomposition, from input alone."""
    candidates = []
    for rotation in range(4):
        grid = np.rot90(input_grid, rotation)
        height, width = grid.shape
        if width < 3 or width % 2 != 1 or height < 3:
            continue
        for depth in range(1, height - 1):
            divider = int(grid[depth, 0])
            background = int(grid[-1, 0])
            if divider == background or not np.all(grid[depth] == divider):
                continue
            if not np.all(grid[depth + 1:] == background):
                continue
            content = grid[:depth]
            if np.any((content == divider) | (content == background)):
                continue
            colors, counts = np.unique(content, return_counts=True)
            if len(colors) < 2 or np.count_nonzero(counts == max(counts)) != 1:
                continue
            candidates.append((rotation, depth, height - depth - 1, width,
                               background, divider, int(colors[np.argmax(counts)]),
                               colors, counts, content.copy()))
    return candidates


def _counts(rng, cells, palette_size, style):
    styles = (style, "near", "plurality", "dominant")
    for candidate_style in styles:
        choices = []
        if candidate_style == "near":
            margins = (1, 2)
        elif candidate_style == "close_multi":
            margins = (1, 2)
        elif candidate_style == "plurality":
            margins = tuple(range(2, max(3, cells // 9) + 1))
        else:
            margins = tuple(range(2, max(3, cells // 3) + 1))
        for margin in margins:
            for winner in range(2, cells - palette_size + 2):
                second = winner - margin
                remainder = cells - winner - second
                if second < 1:
                    continue
                if palette_size == 2:
                    feasible = remainder == 0
                else:
                    feasible = palette_size - 2 <= remainder <= (palette_size - 2) * second
                if candidate_style == "close_multi" and palette_size >= 3:
                    feasible = feasible and second >= 2 and (
                        remainder - (second - 1) >= palette_size - 3) and (
                        remainder - (second - 1) <= (palette_size - 3) * second)
                if candidate_style == "close_multi" and palette_size < 3:
                    feasible = False
                if candidate_style == "plurality":
                    feasible = feasible and winner * 2 < cells
                if candidate_style == "dominant":
                    feasible = feasible and winner * 2 >= cells
                if feasible:
                    choices.append((winner, second, candidate_style))
        if choices:
            winner, second, used_style = choices[int(rng.integers(len(choices)))]
            others = [1] * (palette_size - 2)
            if used_style == "close_multi":
                others[0] = second - 1
            remaining = cells - winner - second - sum(others)
            while remaining:
                eligible = [i for i, value in enumerate(others) if value < second]
                i = eligible[int(rng.integers(len(eligible)))]
                others[i] += 1
                remaining -= 1
            rng.shuffle(others)
            return np.array([winner, second] + others, dtype=np.int64)
    raise RuntimeError("no unique plurality for geometry and palette")


def _arrange(rng, palette, counts, depth, width, style):
    cells = depth * width
    locations = np.arange(cells)
    result = np.full(cells, -1, dtype=np.int8)
    winner_count = int(counts[0])
    available = locations.copy()
    if style == "loser_patch":
        # The largest loser occupies a contiguous snake at the top; the
        # winner is kept out of the first row to disprove a local-row rule.
        snake = []
        for row in range(depth):
            columns = range(width) if row % 2 == 0 else range(width - 1, -1, -1)
            snake.extend(row * width + col for col in columns)
        patch = np.array(snake[:int(counts[1])], dtype=int)
        result[patch] = palette[1]
        available = locations[result < 0]
        winner_places = available[available >= width]
        if len(winner_places) < winner_count:
            winner_places = available
        winner_places = rng.choice(winner_places, winner_count, replace=False)
    elif style in ("winner_rows", "winner_columns"):
        axis_size = depth if style == "winner_rows" else width
        units = np.arange(axis_size)
        rng.shuffle(units)
        capacity = width if style == "winner_rows" else depth
        unit_count = min(axis_size, max(1, (winner_count + capacity - 1) // capacity))
        if unit_count == axis_size and axis_size > 1 and winner_count <= (axis_size - 1) * capacity:
            unit_count -= 1
        chosen_units = units[:unit_count]
        pool = locations[np.isin(locations // width if style == "winner_rows"
                                 else locations % width, chosen_units)]
        winner_places = rng.choice(pool, winner_count, replace=False)
    elif style == "cluster":
        center_row = int(rng.integers(depth))
        center_col = int(rng.integers(width))
        score = (np.abs(locations // width - center_row) +
                 np.abs(locations % width - center_col) + rng.random(cells) * .3)
        winner_places = np.argsort(score)[:winner_count]
    elif style == "stripes":
        period = int(rng.integers(2, min(5, width + 1)))
        phase = int(rng.integers(period))
        score = ((locations % width - phase) % period + rng.random(cells) * .4)
        winner_places = np.argsort(score)[:winner_count]
    else:
        winner_places = rng.choice(locations, winner_count, replace=False)
    result[winner_places] = palette[0]
    leftovers = np.repeat(np.array(palette[1:], dtype=np.int8), counts[1:])
    if style == "loser_patch":
        leftovers = np.repeat(np.array(palette[2:], dtype=np.int8), counts[2:])
    rng.shuffle(leftovers)
    result[result < 0] = leftovers
    return result.reshape(depth, width)


def _make_pair(rng, width, depth, blank_depth, palette_size, winner,
               orientation, count_style, arrangement, previous_roles=None):
    possible_roles = [c for c in range(10) if c != winner]
    role_pairs = [(a, b) for a in possible_roles for b in possible_roles
                  if a != b and (a, b) != previous_roles]
    background, divider = role_pairs[int(rng.integers(len(role_pairs)))]
    other_colors = [c for c in range(10) if c not in (winner, background, divider)]
    losers = rng.choice(other_colors, palette_size - 1, replace=False).tolist()
    palette = [winner] + losers
    counts = _counts(rng, width * depth, palette_size, count_style)
    content = _arrange(rng, palette, counts, depth, width, arrangement)
    grid = np.full((depth + blank_depth + 1, width), background, dtype=np.int8)
    grid[:depth] = content
    grid[depth] = divider
    output = grid.copy()
    output[-1, width // 2] = winner
    return np.rot90(grid, orientation).copy(), np.rot90(output, orientation).copy()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    source_layout = bool(rng.random() < .22)
    orientation_mode = 0 if source_layout else int(rng.choice(3, p=[.30, .28, .42]))
    if orientation_mode == 0:
        orientations = [0]
    elif orientation_mode == 1:
        orientations = [int(rng.integers(1, 4))]
    else:
        orientations = [0, int(rng.integers(1, 4))]
        if train_count >= 5 and rng.random() < .5:
            orientations.append(next(x for x in (1, 2, 3) if x not in orientations))
    puzzle = {"train": [], "test": []}
    train_dimensions = []
    train_winners = []
    train_palettes = []
    previous_roles = None
    for index in range(train_count + test_count):
        training = index < train_count
        if training:
            local = index
            if local == 0:
                width = int(rng.choice([3, 5, 7]))
                depth = 1 if width == 3 else int(rng.integers(1, 3))
                blank = depth
                style = "near"
                arrangement = "shuffle"
            elif local == 1:
                width = int(rng.choice([7, 9, 11]))
                depth = int(rng.integers(3, 6))
                blank = int(rng.integers(2, 7))
                style = "near"
                arrangement = "loser_patch"
            else:
                width = int(rng.choice(np.arange(3, 26, 2)))
                depth = int(rng.integers(1, min(12, 29 - 1) + 1))
                blank = int(rng.integers(1, min(12, 29 - depth) + 1))
                style = str(rng.choice(["near", "close_multi", "plurality", "dominant"]))
                arrangement = str(rng.choice(["shuffle", "winner_rows", "winner_columns",
                                               "loser_patch", "stripes", "cluster"]))
            if source_layout:
                depth = (width - 1) // 2
                blank = depth
            max_palette = min(8, width * depth - 1)
            if local == 0:
                palette_size = int(rng.integers(2, min(4, max_palette) + 1))
            elif local == 1:
                palette_size = int(rng.integers(3, min(7, max_palette) + 1))
            else:
                palette_size = int(rng.integers(2, max_palette + 1))
            winner_options = [x for x in range(10) if local != 1 or x != train_winners[0]]
            winner = int(rng.choice(winner_options))
            orientation = orientations[local % len(orientations)]
            train_dimensions.append((width, depth, blank))
            train_winners.append(winner)
            train_palettes.append(palette_size)
        else:
            local = index - train_count
            size_regime = str(rng.choice(["large", "familiar", "small"], p=[.42, .38, .20]))
            if size_regime == "familiar":
                width, depth, blank = train_dimensions[int(rng.integers(len(train_dimensions)))]
            elif size_regime == "small":
                width = int(rng.choice([3, 5, 7, 9, 11]))
                depth = int(rng.integers(1, 5))
                blank = int(rng.integers(1, 6))
            else:
                width = int(rng.choice(np.arange(15, 30, 2)))
                depth = int(rng.integers(5, 15))
                blank = int(rng.integers(2, min(14, 29 - depth) + 1))
            if source_layout:
                depth = (width - 1) // 2
                blank = depth
            max_palette = min(8, width * depth - 1)
            palette_size = int(rng.integers(2, max_palette + 1))
            if rng.random() < .48:
                winner = int(rng.choice(train_winners))
            else:
                novel = [x for x in range(10) if x not in train_winners]
                winner = int(rng.choice(novel if novel else train_winners))
            style = str(rng.choice(["near", "close_multi", "plurality", "dominant"]))
            arrangement = str(rng.choice(["shuffle", "winner_rows", "winner_columns",
                                           "loser_patch", "stripes", "cluster"]))
            factor_shift = str(rng.choice(
                ["free", "size_only", "palette_only", "winner_only"],
                p=[.50, .18, .16, .16]))
            if factor_shift == "size_only":
                reference = int(rng.integers(len(train_dimensions)))
                palette_size = train_palettes[reference]
                winner = train_winners[reference]
                if source_layout:
                    width = int(rng.choice([w for w in range(15, 30, 2)
                                            if w > max(x[0] for x in train_dimensions)]))
                    depth = blank = (width - 1) // 2
                else:
                    larger = [(w, d) for w in range(15, 30, 2) for d in range(5, 15)
                              if w * d > max(a * b for a, b, _ in train_dimensions)]
                    width, depth = larger[int(rng.integers(len(larger)))]
                    blank = int(rng.integers(1, min(14, 29 - depth) + 1))
            elif factor_shift == "palette_only":
                width, depth, blank = train_dimensions[1]
                palette_size = int(rng.integers(train_palettes[1] + 1, 9))
                winner = train_winners[1]
            elif factor_shift == "winner_only":
                width, depth, blank = train_dimensions[1]
                palette_size = train_palettes[1]
                novel = [x for x in range(10) if x not in train_winners]
                winner = int(rng.choice(novel if novel else train_winners))
            orientation = orientations[int(rng.integers(len(orientations)))]
        # A changed layout, colors, and count texture are obtained from a fresh
        # RNG state on each attempt. The bounded retry only excludes rare source
        # copies or ambiguous decompositions.
        for _ in range(24):
            input_grid, output_grid = _make_pair(
                rng, width, depth, blank, palette_size, winner,
                orientation, style, arrangement, previous_roles)
            signature = (_signature(input_grid), _signature(output_grid))
            if (len(_parse(input_grid)) == 1 and signature not in _SOURCE_PAIRS and
                all(_signature(p["input"]) != signature[0]
                    for split in puzzle.values() for p in split)):
                break
        else:
            raise RuntimeError("could not construct an unambiguous unique pair")
        canonical = np.rot90(input_grid, -orientation)
        previous_roles = (int(canonical[-1, 0]), int(canonical[depth, 0]))
        puzzle["train" if training else "test"].append(
            {"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split schema")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "pair counts")
    inputs, pairs = set(), set()
    parsed = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                         np.issubdtype(grid.dtype, np.integer), "grid type")
                _require(all(1 <= size <= 30 for size in grid.shape) and
                         np.all((grid >= 0) & (grid <= 9)), "grid bounds")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape, "shape changed")
            input_key, output_key = _signature(inp), _signature(out)
            _require(input_key not in inputs and (input_key, output_key) not in pairs,
                     "duplicate pair")
            _require((input_key, output_key) not in _SOURCE_PAIRS, "source copy")
            inputs.add(input_key)
            pairs.add((input_key, output_key))
            decompositions = _parse(inp)
            _require(len(decompositions) == 1, "ambiguous panel layout")
            rotation, depth, blank, width, background, divider, winner, colors, counts, content = decompositions[0]
            canonical_in = np.rot90(inp, rotation)
            canonical_out = np.rot90(out, rotation)
            expected = canonical_in.copy()
            expected[-1, width // 2] = winner
            _require(np.array_equal(canonical_out, expected), "incorrect plurality transfer")
            _require(np.count_nonzero(inp != out) == 1, "wrong number of changes")
            _require(background != divider and winner not in (background, divider), "role collision")
            _require(blank >= 1 and depth >= 1, "empty panel")
            parsed[split].append({"orientation": (-rotation) % 4,
                                  "depth": depth, "blank": blank, "width": width,
                                  "palette": len(colors), "winner": winner,
                                  "roles": (background, divider), "content": content,
                                  "counts": counts})
    train = parsed["train"]
    test = parsed["test"]
    _require(len({x["winner"] for x in train}) >= 2, "training has one winner")
    _require(len({x["roles"] for x in train + test}) >= 2, "roles never vary")
    _require({x["orientation"] for x in test} <= {x["orientation"] for x in train},
             "test orientation unseen in training")
    _require(any(x["depth"] > 1 and any(
        not np.any(row == x["winner"]) for row in x["content"]) for x in train),
        "training does not rule out every-row cue")


def _largest_patch(mask):
    seen = set()
    largest = 0
    for raw_row, raw_col in zip(*np.where(mask)):
        start = (int(raw_row), int(raw_col))
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        size = 0
        while stack:
            row, col = stack.pop()
            size += 1
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                y, x = neighbor
                if (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and
                        mask[y, x] and neighbor not in seen):
                    seen.add(neighbor)
                    stack.append(neighbor)
        largest = max(largest, size)
    return largest


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, "cross-seed duplicate")
    _require(_puzzle_signature(generate_puzzle(314159, 8, 3)) ==
             _puzzle_signature(generate_puzzle(314159, 8, 3)), "nondeterministic")
    for seed, a, b in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                       (2001, None, 1), (2002, None, 4),
                       (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        _require(a is None or len(puzzle["train"]) == a, "explicit train count")
        _require(b is None or len(puzzle["test"]) == b, "explicit test count")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(3000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    samples = [(_parse(pair["input"])[0], split)
               for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    _require(len({tuple(pair["input"].shape) for p in puzzles for s in p.values()
                  for pair in s}) >= 18, "size diversity")
    _require(len({x[0] for x, _ in samples}) == 4, "orientation diversity")
    _require(len({(x[3], x[1], x[2]) for x, _ in samples}) >= 35,
             "panel geometry diversity")
    _require(len({len(x[7]) for x, _ in samples}) >= 6, "palette diversity")
    _require(any(len(x[7]) >= 7 and split == "train" for x, split in samples),
             "no rich training palette")
    _require(sum(sorted(x[8], reverse=True)[0] - sorted(x[8], reverse=True)[1] == 1
                 for x, _ in samples) >= 50, "too few near ties")
    _require(any(x[8].max() * 2 < x[8].sum() for x, _ in samples),
             "no sub-majority plurality")
    _require(any(x[8].max() * 2 > x[8].sum() for x, _ in samples),
             "no dominant winner")
    _require(len({(x[4], x[5]) for x, _ in samples}) >= 25, "role diversity")
    _require(any(x[1] != x[2] for x, _ in samples), "only equal panel depths")
    _require(any(x[3] != x[1] + x[2] + 1 for x, _ in samples),
             "only square canvases")
    _require(any(x[3] == x[1] + x[2] + 1 and x[1] == x[2] for x, _ in samples),
             "original square equal-depth layout absent")
    _require(sum(len(x[7]) >= 5 and split == "train" for x, split in samples) >= 30,
             "training palettes remain narrow")
    _require(sum(len(x[7]) >= 7 and split == "train" for x, split in samples) >= 15,
             "seven-color training examples absent")

    measured = {name: 0 for name in (
        "train_local_loser", "train_loser_patch", "train_solid", "train_row_absent",
        "train_col_absent", "test_margin_one", "test_familiar_size",
        "test_larger", "test_smaller", "test_familiar_winner", "test_novel_winner",
        "test_richer_palette", "test_only_palette_shift", "mixed_orientation",
        "original_puzzles", "combined", "large_near", "over_sixty_percent")}
    for puzzle in puzzles:
        train = [_parse(pair["input"])[0] for pair in puzzle["train"]]
        test = [_parse(pair["input"])[0] for pair in puzzle["test"]]
        train_dimensions = {(x[3], x[1], x[2]) for x in train}
        train_winners = {x[6] for x in train}
        largest_train_area = max(x[3] * x[1] for x in train)
        largest_train_palette = max(len(x[7]) for x in train)
        measured["mixed_orientation"] += len({x[0] for x in train + test}) > 1
        measured["original_puzzles"] += all(
            x[0] == 0 and x[1] == x[2] and x[3] == x[1] + x[2] + 1
            for x in train + test)
        for x in train:
            content, winner, colors = x[9], x[6], x[7]
            measured["train_row_absent"] += any(winner not in row for row in content)
            measured["train_col_absent"] += any(
                winner not in content[:, col] for col in range(x[3]))
            measured["train_solid"] += any(len(np.unique(row)) == 1 for row in content)
            measured["train_local_loser"] += any(
                np.count_nonzero(row == color) > np.count_nonzero(row == winner)
                for row in content for color in colors if color != winner)
            measured["train_loser_patch"] += max(
                _largest_patch(content == color) for color in colors if color != winner
            ) > _largest_patch(content == winner)
        for x in test:
            frequency = sorted(x[8], reverse=True)
            area = int(sum(frequency))
            familiar_size = (x[3], x[1], x[2]) in train_dimensions
            familiar_winner = x[6] in train_winners
            measured["large_near"] += area >= 100 and frequency[0] - frequency[1] <= 2
            measured["over_sixty_percent"] += frequency[0] * 5 > area * 3
            measured["test_margin_one"] += frequency[0] - frequency[1] == 1
            measured["test_familiar_size"] += familiar_size
            measured["test_larger"] += x[3] * x[1] > largest_train_area
            measured["test_smaller"] += x[3] * x[1] < largest_train_area
            measured["test_familiar_winner"] += familiar_winner
            measured["test_novel_winner"] += not familiar_winner
            measured["test_richer_palette"] += len(x[7]) > largest_train_palette
            measured["test_only_palette_shift"] += (
                familiar_size and familiar_winner and len(x[7]) > largest_train_palette)
            measured["combined"] += (
                frequency[0] - frequency[1] <= 2 and x[0] != 0 and
                x[3] != x[1] + x[2] + 1 and
                any(x[6] not in row for row in x[9]))
    for name, minimum in (
        ("train_local_loser", 30), ("train_loser_patch", 20),
        ("train_solid", 10), ("train_row_absent", 25),
        ("train_col_absent", 25), ("test_margin_one", 15),
        ("test_familiar_size", 15), ("test_larger", 10),
        ("test_smaller", 10), ("test_familiar_winner", 15),
        ("test_novel_winner", 10), ("test_richer_palette", 3),
        ("test_only_palette_shift", 2), ("mixed_orientation", 5),
        ("original_puzzles", 2), ("combined", 2)):
        _require(measured[name] >= minimum, f"insufficient {name} coverage")
    _require(measured["large_near"] >= 5, "large close contests absent")
    _require(measured["over_sixty_percent"] >= 1, "old 60-percent cap remains")
