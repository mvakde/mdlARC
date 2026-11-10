import numpy as np


_KINDS = ("corner", "open_frame", "branch")
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Original inputs are embedded only for the exact-copy check. They are never
# used to choose a shape or to construct an output.
_SOURCE_INPUTS = (
    """33300000000
00300033330
00300030030
00000030030
00000030030
00000000000
33333000000
00300000000
00300000000
00333000000""",
    """000000000000000003300
000000000000000000300
000000000000000000300
000300300000000000000
000333300000333000000
000300000000300000000
000000000000300000000
000000000000333000000
000000000000000000000
033000000000000000000
030000000000000033000
030000000000000003000
030000000000000033000
333300333330000000000
000000300000000000000
000000300000000333000
000000000000000303000
003000000000000303000
033000000000000003000""",
    """000000000000
030000000300
030000000333
030003000000
033333000000
000000000000
000000000000
000000030000
000003330000
000003030000
000003030000
333000000000
003000000303
000000000333""",
    """0303000000003000
0303000000003000
0333000000003330
0000000000003030
0000003300000030
0000003000000000
0003333300333000
0000000000003000
0000000003333000
0000000000000000
0000000000030000
0333333000030000
0000003000033300
0000003000000000""",
)


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _integer(num_train, "num_train", 2, 10)
    test = _integer(num_test, "num_test", 1, 4)
    rng = np.random.default_rng(seed ^ 0xA5B72318)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _grid_key(grid):
    return (tuple(int(x) for x in grid.shape), np.asarray(grid, dtype=np.uint8).tobytes())


def _source_keys():
    result = set()
    for drawing in _SOURCE_INPUTS:
        rows = drawing.splitlines()
        result.add(((len(rows), len(rows[0])), bytes(int(c) for row in rows for c in row)))
    return result


_SOURCE_KEYS = _source_keys()


def _canonical(kind, lengths):
    cells = set()
    if kind == "corner":
        horizontal, vertical = lengths
        cells.update((0, x) for x in range(horizontal + 1))
        cells.update((y, 0) for y in range(vertical + 1))
    elif kind == "open_frame":
        span, first, second = lengths
        cells.update((0, x) for x in range(span + 1))
        cells.update((y, 0) for y in range(first + 1))
        cells.update((y, span) for y in range(second + 1))
    elif kind == "branch":
        left, right, connector, terminal = lengths
        cells.update((0, x) for x in range(-left, right + 1))
        cells.update((y, 0) for y in range(connector + 1))
        cells.update((connector, x) for x in range(terminal + 1))
    else:
        raise ValueError("unknown component kind")
    return cells


def _transform(cells, transform):
    result = set()
    for row, col in cells:
        if transform >= 4:
            col = -col
        for _ in range(transform % 4):
            row, col = col, -row
        result.add((row, col))
    top = min(row for row, _ in result)
    left = min(col for _, col in result)
    return tuple(sorted((row - top, col - left) for row, col in result))


def _sample_lengths(kind, rng):
    if kind == "corner":
        return (int(rng.integers(1, 7)), int(rng.integers(1, 7)))
    if kind == "open_frame":
        return (int(rng.integers(2, 7)), int(rng.integers(1, 7)), int(rng.integers(1, 7)))
    return (int(rng.integers(1, 5)), int(rng.integers(1, 5)),
            int(rng.integers(2, 7)), int(rng.integers(1, 5)))


def _sample_kinds(rng, regime, pair_index, first_train_total):
    if regime == "source_train" and pair_index == 0:
        total = int(rng.integers(3, 7))
        kinds = list(_KINDS)
        kinds.extend(str(x) for x in rng.choice(_KINDS, size=total - 3))
    elif regime == "source_train":
        totals = [n for n in range(2, 10) if n != first_train_total]
        total = int(rng.choice(totals))
        mode = float(rng.random())
        if mode < 0.24:
            kinds = [str(rng.choice(_KINDS))] * total
        elif mode < 0.55:
            chosen = rng.choice(_KINDS, size=2, replace=False)
            kinds = [str(chosen[0]), str(chosen[1])]
            kinds.extend(str(x) for x in rng.choice(chosen, size=total - 2))
        else:
            kinds = [str(x) for x in rng.choice(_KINDS, size=total)]
    else:
        mode = float(rng.random())
        if mode < 0.23:
            kinds = list(_KINDS) * 2
        elif mode < 0.51:
            total = int(rng.integers(1, 8))
            kinds = [str(rng.choice(_KINDS))] * total
        elif mode < 0.77:
            total = int(rng.integers(4, 9))
            dominant = str(rng.choice(_KINDS))
            others = [kind for kind in _KINDS if kind != dominant]
            minority = int(rng.integers(1, min(3, total)))
            kinds = [dominant] * (total - minority)
            kinds.extend(str(x) for x in rng.choice(others, size=minority))
        else:
            total = int(rng.integers(1, 9))
            kinds = [str(x) for x in rng.choice(_KINDS, size=total)]
    order = rng.permutation(len(kinds))
    return [kinds[int(i)] for i in order]


def _sample_shapes(kinds, rng):
    shapes = [None] * len(kinds)
    open_indices = [i for i, kind in enumerate(kinds) if kind == "open_frame"]
    branch_indices = [i for i, kind in enumerate(kinds) if kind == "branch"]
    # This pair has identical area and bounds but different graph topology.
    if open_indices and branch_indices and rng.random() < 0.46:
        left = int(rng.integers(1, 5))
        right = int(rng.integers(1, 5))
        connector = int(rng.integers(2, 7))
        terminal = int(rng.integers(1, min(right, connector, 4) + 1))
        transform = int(rng.integers(0, 8))
        oi = int(rng.choice(open_indices))
        bi = int(rng.choice(branch_indices))
        shapes[oi] = _transform(_canonical("open_frame", (left + right, connector, terminal)), transform)
        shapes[bi] = _transform(_canonical("branch", (left, right, connector, terminal)), transform)
    for i, kind in enumerate(kinds):
        if shapes[i] is None:
            transform = int(rng.integers(0, 8))
            shapes[i] = _transform(_canonical(kind, _sample_lengths(kind, rng)), transform)
    return shapes


def _dimensions(shape):
    return max(row for row, _ in shape) + 1, max(col for _, col in shape) + 1


def _can_place(shape, row, col, occupied):
    height, width = occupied.shape
    for dy, dx in shape:
        y, x = row + dy, col + dx
        if occupied[y, x]:
            return False
        for ny, nx in _DIRS:
            yy, xx = y + ny, x + nx
            if 0 <= yy < height and 0 <= xx < width and occupied[yy, xx]:
                return False
    return True


def _pack(shapes, rng, background, foreground):
    count = len(shapes)
    largest_height = max(_dimensions(shape)[0] for shape in shapes)
    largest_width = max(_dimensions(shape)[1] for shape in shapes)
    base = 10 + 2 * count
    first_height = min(30, max(largest_height + 1, base + int(rng.integers(-4, 6))))
    first_width = min(30, max(largest_width + 1, base + int(rng.integers(-4, 6))))
    order = sorted(range(count), key=lambda i: -(_dimensions(shapes[i])[0] * _dimensions(shapes[i])[1] + rng.random() * 0.1))
    for round_index in range(5):
        height = min(30, first_height + 4 * round_index)
        width = min(30, first_width + 4 * round_index)
        occupied = np.zeros((height, width), dtype=bool)
        placed = []
        success = True
        for index in order:
            shape = shapes[index]
            shape_height, shape_width = _dimensions(shape)
            row_limit = height - shape_height
            col_limit = width - shape_width
            if row_limit < 0 or col_limit < 0:
                success = False
                break
            for _ in range(1200):
                row = int(rng.integers(0, row_limit + 1))
                col = int(rng.integers(0, col_limit + 1))
                if rng.random() < 0.32:
                    side = int(rng.integers(0, 4))
                    if side == 0:
                        row = 0
                    elif side == 1:
                        row = row_limit
                    elif side == 2:
                        col = 0
                    else:
                        col = col_limit
                if _can_place(shape, row, col, occupied):
                    for dy, dx in shape:
                        occupied[row + dy, col + dx] = True
                    placed.append((index, row, col))
                    break
            else:
                success = False
                break
        if success:
            break
    else:
        raise RuntimeError("could not place separated components")

    ys, xs = np.where(occupied)
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())
    box_height, box_width = bottom - top + 1, right - left + 1
    edge_mode = rng.random() < 0.58
    margins = [int(rng.integers(0 if edge_mode else 1, 5)) for _ in range(4)]
    if edge_mode:
        margins[int(rng.integers(0, 4))] = 0
    for first, second, size in ((0, 1, box_height), (2, 3, box_width)):
        while size + margins[first] + margins[second] > 30:
            side = first if margins[first] >= margins[second] else second
            margins[side] -= 1
    while (box_height + margins[0] + margins[1]) * (box_width + margins[2] + margins[3]) <= 2 * int(occupied.sum()):
        choices = [i for i in range(4) if
                   (box_height + margins[0] + margins[1] < 30 if i < 2
                    else box_width + margins[2] + margins[3] < 30)]
        if not choices:
            raise RuntimeError("background cannot remain the majority")
        margins[int(rng.choice(choices))] += 1
    grid = np.full((box_height + margins[0] + margins[1],
                    box_width + margins[2] + margins[3]), background, dtype=np.int8)
    grid[margins[0]:margins[0] + box_height,
         margins[2]:margins[2] + box_width][occupied[top:bottom + 1, left:right + 1]] = foreground
    return grid


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for start_row in range(height):
        for start_col in range(width):
            if not mask[start_row, start_col] or seen[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            seen[start_row, start_col] = True
            cells = []
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for dy, dx in _DIRS:
                    y, x = row + dy, col + dx
                    if 0 <= y < height and 0 <= x < width and mask[y, x] and not seen[y, x]:
                        seen[y, x] = True
                        stack.append((y, x))
            result.append(tuple(sorted(cells)))
    return result


def _classify(component):
    cells = set(component)
    _check(len(cells) >= 3, "component is too small")
    degrees = {}
    neighbors = {}
    bends = []
    for row, col in cells:
        vectors = [(dy, dx) for dy, dx in _DIRS if (row + dy, col + dx) in cells]
        degree = len(vectors)
        _check(1 <= degree <= 3, "component has an invalid cell degree")
        degrees[(row, col)] = degree
        neighbors[(row, col)] = [(row + dy, col + dx) for dy, dx in vectors]
        if degree == 2 and (vectors[0][0] + vectors[1][0], vectors[0][1] + vectors[1][1]) != (0, 0):
            bends.append((row, col))
    _check(sum(degrees.values()) // 2 == len(cells) - 1, "component is not a tree")
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    junctions = [cell for cell, degree in degrees.items() if degree == 3]
    if (len(endpoints), len(junctions), len(bends)) == (2, 0, 1):
        return "corner"
    if (len(endpoints), len(junctions), len(bends)) == (2, 0, 2):
        directions = []
        for endpoint in endpoints:
            previous = None
            current = endpoint
            while current not in bends:
                following = [cell for cell in neighbors[current] if cell != previous]
                _check(len(following) == 1, "open path has an invalid run")
                previous, current = current, following[0]
            directions.append((int(np.sign(endpoint[0] - current[0])),
                               int(np.sign(endpoint[1] - current[1]))))
        if directions[0] == directions[1]:
            return "open_frame"
    if (len(endpoints), len(junctions), len(bends)) == (3, 1, 1):
        junction = junctions[0]
        adjacent = neighbors[junction]
        stem = None
        for candidate in adjacent:
            others = [cell for cell in adjacent if cell != candidate]
            if (others[0][0] + others[1][0], others[0][1] + others[1][1]) == (2 * junction[0], 2 * junction[1]):
                stem = candidate
                break
        _check(stem is not None, "branch has no straight crossbar")
        previous, current = junction, stem
        while degrees[current] == 2 and current not in bends:
            following = [cell for cell in neighbors[current] if cell != previous]
            previous, current = current, following[0]
        if current == bends[0]:
            return "branch"
    raise AssertionError("component is outside the three shape classes")


def _inspect(grid):
    colors, counts = np.unique(grid, return_counts=True)
    _check(len(colors) == 2, "input needs background and one foreground color")
    background_index = int(np.argmax(counts))
    _check(counts[background_index] > counts[1 - background_index], "background is not the unique majority")
    background = int(colors[background_index])
    foreground = int(colors[1 - background_index])
    records = [(cells, _classify(cells)) for cells in _components(grid == foreground)]
    _check(len(records) > 0, "input has no objects")
    return background, foreground, records


def _make_output(grid, background, foreground, labels):
    result = np.full(grid.shape, background, dtype=np.int8)
    for cells in _components(grid == foreground):
        color = labels[_classify(cells)]
        for row, col in cells:
            result[row, col] = color
    return result


def _generate_pair(rng, regime, pair_index, first_train_total, colors):
    for _ in range(12):
        kinds = _sample_kinds(rng, regime, pair_index, first_train_total)
        shapes = _sample_shapes(kinds, rng)
        try:
            grid = _pack(shapes, rng, colors[0], colors[1])
        except RuntimeError:
            continue
        records = _inspect(grid)[2]
        if sorted(kind for _, kind in records) != sorted(kinds):
            continue
        output = _make_output(grid, colors[0], colors[1], dict(zip(_KINDS, colors[2:])))
        return {"input": grid, "output": output}
    raise RuntimeError("could not construct a valid pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = tuple(int(x) for x in rng.choice(10, size=5, replace=False))
    vary_input_colors = bool(rng.random() < 0.45)
    available = [color for color in range(10) if color not in colors[2:]]
    input_role_pairs = [(background, foreground) for background in available
                        for foreground in available if background != foreground]
    rng.shuffle(input_role_pairs)
    puzzle = {"train": [], "test": []}
    used = set()
    first_train_total = None
    pair_number = 0
    for regime, count, split in (("source_train", train_count, "train"), ("target_test", test_count, "test")):
        for pair_index in range(count):
            pair_colors = ((input_role_pairs[pair_number % len(input_role_pairs)] + colors[2:])
                           if vary_input_colors else colors)
            for _ in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0, 1 << 63)))
                pair = _generate_pair(pair_rng, regime, pair_index, first_train_total, pair_colors)
                key = _grid_key(pair["input"])
                if key not in used and key not in _SOURCE_KEYS:
                    break
            else:
                raise RuntimeError("could not make a unique pair")
            used.add(key)
            puzzle[split].append(pair)
            pair_number += 1
            if regime == "source_train" and pair_index == 0:
                first_train_total = len(_inspect(pair["input"])[2])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _check(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "invalid pair counts")
    inputs = set()
    pairs = set()
    learned = {}
    pair_input_roles = []
    train_totals = []
    for split, collection in (("train", train), ("test", test)):
        for index, pair in enumerate(collection):
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            grid, output = pair["input"], pair["output"]
            for name, array in (("input", grid), ("output", output)):
                _check(isinstance(array, np.ndarray) and array.ndim == 2, f"invalid {name} array")
                _check(np.issubdtype(array.dtype, np.integer), f"noninteger {name}")
                _check(1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30, f"invalid {name} size")
                _check(int(array.min()) >= 0 and int(array.max()) <= 9, f"invalid {name} color")
            _check(grid.shape == output.shape and not np.array_equal(grid, output), "wrong output dimensions or unchanged input")
            input_key, output_key = _grid_key(grid), _grid_key(output)
            _check(input_key not in inputs and (input_key, output_key) not in pairs, "duplicate pair")
            _check(input_key not in _SOURCE_KEYS, "exact source input was copied")
            inputs.add(input_key)
            pairs.add((input_key, output_key))
            background, foreground, records = _inspect(grid)
            _check(background != foreground, "ambiguous input colors")
            pair_input_roles.append((background, foreground))
            _check(np.all(output[grid == background] == background), "background changed")
            if split == "train":
                train_totals.append(len(records))
                _check(2 <= len(records) <= 9, "training component count outside regime")
                if index == 0:
                    _check(3 <= len(records) <= 6 and {kind for _, kind in records} == set(_KINDS), "first training pair must teach all classes")
            else:
                _check(1 <= len(records) <= 8, "test component count outside regime")
            for cells, kind in records:
                colors = {int(output[row, col]) for row, col in cells}
                _check(len(colors) == 1, "component not uniformly recolored")
                color = colors.pop()
                _check(color not in {background, foreground}, "label color collides with input role")
                if split == "train":
                    if kind in learned:
                        _check(learned[kind] == color, "inconsistent class label")
                    else:
                        learned[kind] = color
                else:
                    _check(kind in learned and learned[kind] == color, "test label was not learned")
            if split == "test":
                _check(np.array_equal(output, _make_output(grid, background, foreground, learned)), "test output not inferable from training")
    _check(len(set(train_totals)) >= 2, "training counts do not vary within puzzle")
    _check(set(learned) == set(_KINDS) and len(set(learned.values())) == 3, "training does not distinguish labels")
    for background, foreground in pair_input_roles:
        _check(len({background, foreground, *learned.values()}) == 5, "semantic color roles overlap")
    for pair in train:
        grid, output = pair["input"], pair["output"]
        background, foreground, _ = _inspect(grid)
        _check(np.array_equal(output, _make_output(grid, background, foreground, learned)), "training output violates rule")


def _equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[name], b[name])
        for a, b in zip(first[split], second[split]) for name in ("input", "output")
    ) for split in ("train", "test"))


def _bbox(cells):
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def self_test() -> None:
    first = generate_puzzle(777, 7, 3)
    _check(_equal(first, generate_puzzle(777, 7, 3)), "generation is not deterministic")
    _check(not _equal(first, generate_puzzle(778, 7, 3)), "seed had no effect")
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 2), (9, 3), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1000 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        _check(train is None or len(puzzle["train"]) == train, "explicit training count ignored")
        _check(test is None or len(puzzle["test"]) == test, "explicit test count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(1400, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")

    train_counts, test_counts, roles, grids, layouts, profiles = set(), set(), set(), set(), set(), set()
    train_missing = test_single = test_balanced = test_repeated_area = matched = edge_pairs = 0
    varied_input_palettes = 0
    test_totals = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        b, f, first_records = _inspect(puzzle["train"][0]["input"])
        labels = {kind: int(puzzle["train"][0]["output"][cells[0]]) for cells, kind in first_records}
        roles.add((b, f, *(labels[kind] for kind in _KINDS)))
        palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                background, foreground, records = _inspect(grid)
                palettes.add((background, foreground))
                _check(np.array_equal(output, _make_output(grid, background, foreground, labels)), "independent pair rule check failed")
                grids.add(tuple(grid.shape))
                layouts.add((tuple(grid.shape), np.asarray(grid != background, dtype=np.uint8).tobytes()))
                profile = tuple(sorted((kind, len(cells), *_bbox(cells)) for cells, kind in records))
                profiles.add(profile)
                if any(row in (0, grid.shape[0] - 1) or col in (0, grid.shape[1] - 1)
                       for cells, _ in records for row, col in cells):
                    edge_pairs += 1
                if split == "train":
                    train_missing += int({kind for _, kind in records} != set(_KINDS))
                else:
                    test_totals.add(len(records))
                    counts = [sum(kind == name for _, kind in records) for name in _KINDS]
                    test_single += int(sum(count > 0 for count in counts) == 1)
                    test_balanced += int(sorted(counts) == [2, 2, 2])
                    test_repeated_area += int(len({len(cells) for cells, _ in records}) < len(records))
                    open_features = {(len(cells), tuple(sorted(_bbox(cells)))) for cells, kind in records if kind == "open_frame"}
                    branch_features = {(len(cells), tuple(sorted(_bbox(cells)))) for cells, kind in records if kind == "branch"}
                    matched += int(bool(open_features & branch_features))
        varied_input_palettes += int(len(palettes) > 1)
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic pair counts lack diversity")
    _check(len(roles) >= 35 and len(grids) >= 30 and len(layouts) >= 150 and len(profiles) >= 80, "cross-seed diversity too narrow")
    _check(train_missing >= 15 and len(test_totals) >= 6, "component composition too narrow")
    _check(test_single >= 8 and test_balanced >= 5, "test class regimes too narrow")
    _check(test_repeated_area >= 8 and matched >= 5, "repeated or matched component sizes too rare")
    _check(edge_pairs >= 50, "edge placement too rare")
    _check(varied_input_palettes >= 10, "pair-level input color variation too rare")
