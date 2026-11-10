"""Generate ARC puzzles by completing and extending monochrome stencil rays."""

import hashlib

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1))
CARDINAL, DIAGONAL = DIRS[:4], DIRS[4:]
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if r or c)
SOURCE_INPUT_DIGESTS = frozenset((
    "9fd7e540f4e3000b963c8356cbb80fe4b3cc012160de38d4a0f572963b7d6fba",
    "d46a565de049503d468bc7f94e2123a3034bd3e4472b198afd653f9e946baddd",
    "689cb047902a0eb8237cb26d9afee3291c66557ebee75e37853210b7eb94b9c2",
    "a70adca4352900b6cda95d94cbe9ef0a620013a5906e88eb57ac8b55525e21b4",
))


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4 + 3 * seed) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + (seed // 7 + 5 * seed) % (11 - low)
    elif test is None:
        test = 1 + (seed // 11 + 7 * seed) % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _connected(mask):
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        row, col = stack.pop()
        for dr, dc in NEIGHBORS:
            point = (row + dr, col + dc)
            if point in cells and point not in seen:
                seen.add(point)
                stack.append(point)
    return seen == cells


def _asymmetric(mask):
    transforms = (np.rot90(mask, 1), np.rot90(mask, 2), np.rot90(mask, 3),
                  np.flipud(mask), np.fliplr(mask), mask.T, np.flipud(mask.T))
    return all(other.shape != mask.shape or not np.array_equal(mask, other)
               for other in transforms)


def _random_mask(rng, height, width):
    # The initial eight-connected path touches every side of the box.
    mask = np.zeros((height, width), dtype=bool)
    row = col = 0
    mask[0, 0] = True
    while row < height - 1 or col < width - 1:
        choices = []
        if row < height - 1:
            choices.append((1, 0))
        if col < width - 1:
            choices.append((0, 1))
        if row < height - 1 and col < width - 1:
            choices.append((1, 1))
        dr, dc = choices[int(rng.integers(len(choices)))]
        row += dr
        col += dc
        mask[row, col] = True
    goal = int(rng.integers(max(4, int(mask.sum())), height * width))
    while int(mask.sum()) < goal:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBORS:
                rr, cc = int(r) + dr, int(c) + dc
                if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                    frontier.add((rr, cc))
        options = sorted(frontier)
        rr, cc = options[int(rng.integers(len(options)))]
        mask[rr, cc] = True
    return mask


def _sample_mask(rng, height, width, require_asym=False):
    styles = ["irregular"] * 3
    if height >= 3 and width >= 3:
        styles += ["cross", "cross", "ring", "u", "t", "t"]
    if (height, width) == (3, 3):
        styles += ["diagonal_join"] * 2
    for _ in range(100):
        style = styles[int(rng.integers(len(styles)))]
        mask = np.zeros((height, width), dtype=bool)
        if style == "irregular":
            mask = _random_mask(rng, height, width)
        elif style == "cross":
            middle_rows = [(height - 1) // 2, height // 2]
            middle_cols = [(width - 1) // 2, width // 2]
            mask[int(rng.choice(middle_rows)), :] = True
            mask[:, int(rng.choice(middle_cols))] = True
        elif style == "ring":
            mask[0, :] = mask[-1, :] = True
            mask[:, 0] = mask[:, -1] = True
        elif style == "u":
            mask[:, 0] = mask[:, -1] = True
            mask[int(rng.choice([0, height - 1])), :] = True
        elif style == "t":
            if rng.random() < 0.5:
                mask[int(rng.choice([0, height - 1])), :] = True
                mask[:, int(rng.choice([(width - 1) // 2, width // 2]))] = True
            else:
                mask[:, int(rng.choice([0, width - 1]))] = True
                mask[int(rng.choice([(height - 1) // 2, height // 2])), :] = True
        else:
            mask = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool)
            mask = np.rot90(mask, int(rng.integers(4)))
        if (4 <= int(mask.sum()) < height * width and mask[0].any()
                and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
                and _connected(mask) and (not require_asym or _asymmetric(mask))):
            return mask
    raise RuntimeError("could not sample a template")


def _sample_shape(rng, avoid=None, compact=False):
    for _ in range(20):
        if compact:
            height = int(rng.integers(3, 5))
            width = int(rng.integers(3, 5))
        else:
            mode = int(rng.integers(10))
            if mode < 3:
                height = width = 3
            elif mode < 6:
                height = int(rng.integers(3, 6))
                width = int(rng.integers(3, 6))
            elif mode < 8:
                height, width = 2, int(rng.integers(4, 8))
                if rng.random() < 0.5:
                    height, width = width, height
            else:
                height = int(rng.integers(5, 7))
                width = int(rng.integers(5, 8))
        if (height, width) != avoid:
            return height, width
    return (3, 4) if avoid != (3, 4) else (3, 5)


def _min_canvas(mask, directions):
    h, w = mask.shape
    rows = {dr for dr, _ in directions}
    cols = {dc for _, dc in directions}
    up, down = int(-1 in rows), int(1 in rows)
    left, right = int(-1 in cols), int(1 in cols)
    return h * (1 + up + down) + up + down, w * (1 + left + right) + left + right


def _sample_canvas(rng, mask, directions, fixed, demonstration):
    minimum_h, minimum_w = _min_canvas(mask, directions)
    if demonstration:
        h, w = mask.shape
        minimum_h = max(minimum_h, 4 * h + 3 if any(dr for dr, _ in directions) else 6)
        minimum_w = max(minimum_w, 4 * w + 3 if any(dc for _, dc in directions) else 6)
    minimum_h, minimum_w = max(6, minimum_h), max(6, minimum_w)
    if fixed is not None:
        if fixed[0] < minimum_h or fixed[1] < minimum_w:
            raise RuntimeError("rays do not fit the fixed canvas")
        return fixed
    if minimum_h > 30 or minimum_w > 30:
        raise RuntimeError("rays exceed the canvas limit")

    def extent(low):
        if rng.random() < 0.32:
            return low
        if rng.random() < 0.48:
            return int(rng.integers(low, min(30, low + 6) + 1))
        return int(rng.integers(low, 31))

    return extent(minimum_h), extent(minimum_w)


def _visible_count(mask, origin, direction, index, canvas):
    h, w = mask.shape
    row0 = origin[0] + index * direction[0] * (h + 1)
    col0 = origin[1] + index * direction[1] * (w + 1)
    return sum(0 <= row0 + int(r) < canvas[0] and 0 <= col0 + int(c) < canvas[1]
               for r, c in np.argwhere(mask))


def _profile(mask, origin, direction, canvas):
    counts = []
    for index in range(1, 31):
        count = _visible_count(mask, origin, direction, index, canvas)
        if not count:
            break
        counts.append(count)
    return tuple(counts)


def _sample_origin(rng, mask, directions, canvas, demonstration, long_mode=False):
    h, w = mask.shape
    lo_r = h + 1 if any(dr < 0 for dr, _ in directions) else 0
    hi_r = canvas[0] - 2 * h - 1 if any(dr > 0 for dr, _ in directions) else canvas[0] - h
    lo_c = w + 1 if any(dc < 0 for _, dc in directions) else 0
    hi_c = canvas[1] - 2 * w - 1 if any(dc > 0 for _, dc in directions) else canvas[1] - w
    assert lo_r <= hi_r and lo_c <= hi_c
    candidates = [(r, c) for r in range(lo_r, hi_r + 1)
                  for c in range(lo_c, hi_c + 1)]
    if demonstration:
        candidates = [origin for origin in candidates
                      if any(len(_profile(mask, origin, d, canvas)) >= 2
                             for d in directions)]
        if not candidates:
            raise RuntimeError("no repeating demonstration")
    if long_mode:
        def reach(origin, direction):
            dr, dc = direction
            row_reach = (canvas[0]-1-origin[0]) // (h+1) if dr > 0 else (
                (origin[0]+h-1) // (h+1) if dr < 0 else 30)
            col_reach = (canvas[1]-1-origin[1]) // (w+1) if dc > 0 else (
                (origin[1]+w-1) // (w+1) if dc < 0 else 30)
            return min(row_reach, col_reach)
        scores = [max(reach(origin, d) for d in directions) for origin in candidates]
        best = max(scores)
        candidates = [origin for origin, score in zip(candidates, scores) if score == best]
    return candidates[int(rng.integers(len(candidates)))]


def _starter(rng, mask):
    points = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    count = len(points)
    mode = int(rng.integers(6))
    if mode == 0:
        return [points[int(rng.integers(count))]]
    if mode == 1:
        size = count - int(rng.integers(1, min(3, count)))
    elif mode == 2:
        side = int(rng.integers(4))
        points = [p for p in points if
                  (p[0] == 0, p[0] == mask.shape[0] - 1,
                   p[1] == 0, p[1] == mask.shape[1] - 1)[side]]
        size = min(count - 1, len(points))
    elif mode == 3:
        corners = ((0, 0), (0, mask.shape[1] - 1),
                   (mask.shape[0] - 1, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
        corner = corners[int(rng.integers(4))]
        points.sort(key=lambda p: (abs(p[0]-corner[0]) + abs(p[1]-corner[1]), p))
        return points[:int(rng.integers(1, min(4, count)))]
    elif mode == 4:
        size = int(rng.integers(2, count))
        chosen = {points[int(rng.integers(count))]}
        while len(chosen) < size:
            frontier = sorted(p for p in points if p not in chosen and
                              any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1
                                  for q in chosen))
            chosen.add(frontier[int(rng.integers(len(frontier)))])
        return sorted(chosen)
    else:
        size = int(rng.integers(1, count))
    chosen = rng.choice(len(points), size=size, replace=False)
    return [points[int(i)] for i in chosen]


def _colors(rng, background, directions):
    count = len(directions)
    if count == 1:
        ray_count = 1
    else:
        mode = int(rng.integers(10))
        ray_count = (1 if mode < 3 else
                     int(rng.integers(2, count)) if mode < 6 and count >= 3
                     else count)
    available = [v for v in range(10) if v != background]
    chosen = [int(v) for v in rng.choice(available, size=ray_count + 1, replace=False)]
    assignments = chosen[1:] + [chosen[1 + int(rng.integers(ray_count))]
                                 for _ in range(count - ray_count)]
    rng.shuffle(assignments)
    return chosen[0], dict(zip(directions, assignments))


def _draw_ray(grid, mask, origin, direction, color):
    h, w = mask.shape
    for index in range(1, 31):
        row0 = origin[0] + index * direction[0] * (h + 1)
        col0 = origin[1] + index * direction[1] * (w + 1)
        visible = False
        for r, c in np.argwhere(mask):
            rr, cc = row0 + int(r), col0 + int(c)
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                grid[rr, cc] = color
                visible = True
        if not visible:
            break


def _make_pair(rng, directions, settings, demonstration=False,
               avoid_shape=None, avoid_canvas=None, avoid_background=None):
    long_mode = rng.random() < (0.30 if len(directions) == 1 else 0.08)
    if settings["fixed_canvas"] is None and avoid_canvas == (30, 30):
        long_mode = False
    for _ in range(120):
        shape = ((3, 3) if long_mode and len(directions) == 1 and avoid_shape != (3, 3)
                 else _sample_shape(rng, avoid_shape, compact=demonstration or long_mode))
        try:
            mask = _sample_mask(rng, *shape, require_asym=demonstration)
            canvas = ((30, 30) if long_mode and settings["fixed_canvas"] is None else
                      _sample_canvas(rng, mask, directions, settings["fixed_canvas"],
                                     demonstration))
            if settings["fixed_canvas"] is None and canvas == avoid_canvas:
                continue
            origin = _sample_origin(rng, mask, directions, canvas,
                                    demonstration, long_mode)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("could not construct pair geometry")
    if settings["fixed_background"] is None:
        choices = [v for v in range(10) if v != avoid_background]
        background = int(choices[int(rng.integers(len(choices)))])
    else:
        background = settings["fixed_background"]
    template_color, direction_colors = _colors(rng, background, directions)
    base = np.full(canvas, background, dtype=np.int8)
    for r, c in np.argwhere(mask):
        base[origin[0] + int(r), origin[1] + int(c)] = template_color
    h, w = mask.shape
    for attempt in range(31):
        inp = base.copy()
        for direction, color in direction_colors.items():
            row0 = origin[0] + direction[0] * (h + 1)
            col0 = origin[1] + direction[1] * (w + 1)
            points = (_starter(rng, mask) if attempt < 30 else
                      [tuple(map(int, np.argwhere(mask)[0]))])
            for r, c in points:
                inp[row0 + r, col0 + c] = color
        values, counts = np.unique(inp, return_counts=True)
        if int(counts[values == background][0]) > max(
                int(count) for value, count in zip(values, counts)
                if value != background):
            break
    out = inp.copy()
    for direction, color in direction_colors.items():
        _draw_ray(out, mask, origin, direction, color)
    return {"input": inp, "output": out}


def _components(grid, background):
    remaining = {(int(r), int(c)) for r, c in np.argwhere(grid != background)}
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color = int(grid[start])
        cells = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for dr, dc in NEIGHBORS:
                point = (row + dr, col + dc)
                if point in remaining and int(grid[point]) == color:
                    remaining.remove(point)
                    cells.add(point)
                    stack.append(point)
        components.append((color, cells))
    return components


def _analyze_pair(inp, out):
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) >= 3 and list(counts).count(max(counts)) == 1
    background = int(values[int(np.argmax(counts))])
    components = _components(inp, background)
    sizes = [len(cells) for _, cells in components]
    assert sizes and sizes.count(max(sizes)) == 1
    template_color, template_cells = components[int(np.argmax(sizes))]
    full_size = len(template_cells)
    assert full_size >= 4
    assert int(np.count_nonzero(inp == template_color)) == full_size
    row0 = min(r for r, _ in template_cells)
    col0 = min(c for _, c in template_cells)
    h = max(r for r, _ in template_cells) - row0 + 1
    w = max(c for _, c in template_cells) - col0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in template_cells:
        mask[r-row0, c-col0] = True
    assert _connected(mask) and full_size < h * w
    block = inp[row0:row0+h, col0:col0+w]
    assert np.all((block == background) | (block == template_color))
    direction_colors, starter_sizes = {}, {}
    covered = set(template_cells)
    for direction in DIRS:
        first_r = row0 + direction[0] * (h + 1)
        first_c = col0 + direction[1] * (w + 1)
        if not (0 <= first_r <= inp.shape[0] - h and
                0 <= first_c <= inp.shape[1] - w):
            continue
        cells = [(r, c) for r in range(first_r, first_r+h)
                 for c in range(first_c, first_c+w) if inp[r, c] != background]
        if not cells:
            continue
        colors = {int(inp[r, c]) for r, c in cells}
        assert len(colors) == 1 and len(cells) < full_size
        color = colors.pop()
        assert color != template_color
        assert all(mask[r-first_r, c-first_c] for r, c in cells)
        assert not covered.intersection(cells)
        covered.update(cells)
        direction_colors[direction] = color
        starter_sizes[direction] = len(cells)
    assert direction_colors
    assert covered == {(int(r), int(c)) for r, c in np.argwhere(inp != background)}
    expected = inp.copy()
    profiles = {}
    for direction, color in direction_colors.items():
        profile = _profile(mask, (row0, col0), direction, inp.shape)
        assert profile and profile[0] == full_size
        profiles[direction] = profile
        _draw_ray(expected, mask, (row0, col0), direction, color)
    assert np.array_equal(expected, out)
    assert set(np.unique(inp)) == set(np.unique(out))
    return {"background": background, "canvas": inp.shape, "mask": mask,
            "origin": (row0, col0), "directions": frozenset(direction_colors),
            "colors": direction_colors, "starter_sizes": starter_sizes,
            "profiles": profiles}


def _direction_sample(rng, count, pool=DIRS):
    selected = rng.choice(len(pool), size=count, replace=False)
    return tuple(pool[int(index)] for index in selected)


def _free_count(rng, test=False):
    probabilities = ([0.11, 0.16, 0.27, 0.22, 0.13, 0.06, 0.03, 0.02]
                     if test else [0.20, 0.28, 0.22, 0.14, 0.08, 0.04, 0.025, 0.015])
    return int(rng.choice(np.arange(1, 9), p=probabilities))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_u = _seed(seed)
    train_count, test_count = _resolve_counts(seed_u, num_train, num_test)
    rng = np.random.default_rng(seed_u)
    fixed_canvas = None
    if rng.random() < 0.14:
        side = int(rng.choice([21, 25, 30]))
        fixed_canvas = (side, side)
    fixed_background = int(rng.integers(10)) if rng.random() < 0.16 else None
    settings = {"fixed_canvas": fixed_canvas, "fixed_background": fixed_background}
    recombination = rng.random() < 0.18
    if recombination:
        cardinals = _direction_sample(rng, 3, CARDINAL)
        diagonals = _direction_sample(rng, 2, DIAGONAL)
        anchors = [cardinals, diagonals]
    else:
        first = (_direction_sample(rng, 1, CARDINAL)
                 + _direction_sample(rng, 1, DIAGONAL))
        if rng.random() < 0.45:
            first += _direction_sample(rng, 1, tuple(d for d in DIRS if d not in first))
        second = _direction_sample(rng, int(rng.integers(1, 4)))
        anchors = [first, second]
    train_specs = [(anchors[0], True), (anchors[1], False)]
    for _ in range(train_count - 2):
        train_specs.append((_direction_sample(rng, _free_count(rng)), False))
    rng.shuffle(train_specs)
    train_union = {d for spec, _ in train_specs for d in spec}
    train_sets = {frozenset(spec) for spec, _ in train_specs}
    test_specs = []
    for index in range(test_count):
        if recombination and index == 0:
            spec = (_direction_sample(rng, 2, cardinals)
                    + _direction_sample(rng, 1, diagonals))
        else:
            for _ in range(40):
                pool = tuple(sorted(train_union)) if rng.random() < 0.57 else DIRS
                count = min(_free_count(rng, test=True), len(pool))
                spec = _direction_sample(rng, count, pool)
                if frozenset(spec) not in train_sets:
                    break
        test_specs.append(spec)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_shape = first_canvas = first_background = None
    for split, specs in (("train", train_specs),
                         ("test", [(spec, False) for spec in test_specs])):
        for index, (directions, demonstration) in enumerate(specs):
            pair_rng = np.random.default_rng(int(rng.integers(0, (1 << 63)-1)))
            for _ in range(30):
                pair = _make_pair(pair_rng, directions, settings, demonstration,
                                  avoid_shape=first_shape if split == "train" and index == 1 else None,
                                  avoid_canvas=first_canvas if split == "train" and index == 1 else None,
                                  avoid_background=first_background if split == "train" and index == 1 else None)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and index == 0:
                        info = _analyze_pair(pair["input"], pair["output"])
                        first_shape = info["mask"].shape
                        first_canvas = info["canvas"]
                        first_background = info["background"]
                    break
            else:
                raise RuntimeError("could not create distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if inp.shape == (21, 21):
                digest = hashlib.sha256(inp.astype(np.int8, copy=False).tobytes()).hexdigest()
                assert digest not in SOURCE_INPUT_DIGESTS
            metadata[split].append(_analyze_pair(inp, out))
    # These are demonstrated rules, not hidden variable conventions.
    train = metadata["train"]
    assert any(_asymmetric(m["mask"]) and
               any(len(profile) >= 2 for profile in m["profiles"].values())
               for m in train)
    assert any(any(d in CARDINAL for d in m["directions"]) for m in train)
    assert any(any(d in DIAGONAL for d in m["directions"]) for m in train)


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    original = generate_puzzle(9173, 6, 2)
    assert _puzzle_key(original) == _puzzle_key(generate_puzzle(9173, 6, 2))
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (104, None, 1), (105, None, 4),
                              (106, 2, None), (107, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
    train_counts, test_counts, puzzle_keys = set(), set(), set()
    canvases, shapes, masks, backgrounds = set(), set(), set(), set()
    ray_counts, color_counts, direction_sets = set(), set(), set()
    clipped = complete = short = long = symmetric = asymmetric = False
    within_canvas = within_background = within_shape = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        infos = [_analyze_pair(p["input"], p["output"])
                 for split in ("train", "test") for p in puzzle[split]]
        within_canvas += len({m["canvas"] for m in infos}) > 1
        within_background += len({m["background"] for m in infos}) > 1
        within_shape += len({m["mask"].shape for m in infos}) > 1
        for m in infos:
            canvases.add(m["canvas"])
            shapes.add(m["mask"].shape)
            masks.add((m["mask"].shape, m["mask"].tobytes()))
            backgrounds.add(m["background"])
            ray_counts.add(len(m["directions"]))
            color_counts.add(len(set(m["colors"].values())))
            direction_sets.add(m["directions"])
            symmetric |= not _asymmetric(m["mask"])
            asymmetric |= _asymmetric(m["mask"])
            full = int(m["mask"].sum())
            for profile in m["profiles"].values():
                clipped |= any(0 < n < full for n in profile)
                complete |= all(n == full for n in profile)
                short |= len(profile) == 1
                long |= len(profile) >= 4
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) >= 40 and len(canvases) >= 18
    assert len(shapes) >= 12 and len(masks) >= 80
    assert len(backgrounds) == 10 and len(direction_sets) >= 30
    assert {1, 2, 3, 4} <= ray_counts and {1, 2, 3} <= color_counts
    assert all((clipped, complete, short, long, symmetric, asymmetric))
    assert within_canvas >= 25 and within_background >= 25 and within_shape >= 35
