import numpy as np


_DIRECTIONS = ("up", "right", "down", "left")

_STEPS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

_SOURCE_PAIR_ROWS = (
    (
        (
            "555555555",
            "555555555",
            "551111555",
            "551551555",
            "555551555",
            "551111555",
            "555555555",
        ),
        (
            "555555555",
            "555555555",
            "551111555",
            "551221555",
            "222221555",
            "551111555",
            "555555555",
        ),
    ),
    (
        (
            "8888888888",
            "8888888888",
            "8118188888",
            "8188188888",
            "8188188888",
            "8111188888",
            "8888881111",
            "8888881881",
            "8888881881",
            "8888881181",
            "8888888888",
        ),
        (
            "8882888888",
            "8882888888",
            "8112188888",
            "8122188888",
            "8122188888",
            "8111188888",
            "8888881111",
            "8888881221",
            "8888881221",
            "8888881121",
            "8888888828",
        ),
    ),
    (
        (
            "919119999",
            "919919999",
            "919919999",
            "911119999",
            "999999999",
            "999999999",
            "991111999",
            "991991999",
            "991999999",
            "991111999",
            "999999999",
        ),
        (
            "912119999",
            "912219999",
            "912219999",
            "911119999",
            "999999999",
            "999999999",
            "991111999",
            "991221999",
            "991222222",
            "991111999",
            "999999999",
        ),
    ),
    (
        (
            "444444444444444",
            "444444441111444",
            "414114441444444",
            "414414441441444",
            "414414441111444",
            "411114444444444",
            "444444411114444",
            "444444414414444",
            "444444414414444",
            "444444414114444",
            "444444444444444",
        ),
        (
            "442444444444444",
            "442444441111444",
            "412114441222222",
            "412214441221444",
            "412214441111444",
            "411114444444444",
            "444444411114444",
            "444444412214444",
            "444444412214444",
            "444444412114444",
            "444444442444444",
        ),
    ),
)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside {low}..{high}")
    return value


def _counts(rng, train, test):
    train = _count(train, "num_train", 2, 10)
    test = _count(test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _ray(rect, height, width):
    dr, dc = _STEPS[rect["direction"]]
    row, col = rect["gap"]
    result = []
    while 0 <= row < height and 0 <= col < width:
        result.append((row, col))
        row += dr
        col += dc
    return result


def _inside(point, rect):
    row, col = point
    return rect["top"] <= row <= rect["bottom"] and rect["left"] <= col <= rect["right"]


def _separate(a, b):
    return (a["bottom"] + 1 < b["top"] or b["bottom"] + 1 < a["top"]
            or a["right"] + 1 < b["left"] or b["right"] + 1 < a["left"])


def _rectangle(top, left, fh, fw, direction, offset):
    bottom, right = top + fh - 1, left + fw - 1
    if direction == "up":
        gap = (top, left + offset)
    elif direction == "down":
        gap = (bottom, left + offset)
    elif direction == "left":
        gap = (top + offset, left)
    else:
        gap = (top + offset, right)
    return dict(top=top, left=left, bottom=bottom, right=right,
                gap=gap, direction=direction)


def _candidate(rng, height, width, direction, short=False, large=False, fixed_size=None):
    # Height and width are independent, including 3-cell thin and long frames.
    def dimension(limit):
        if short:
            return int(rng.integers(3, min(5, limit) + 1))
        if large:
            return int(rng.integers(8, min(14, limit) + 1))
        band = int(rng.choice(3, p=[0.35, 0.45, 0.20]))
        lo, hi = ((3, 4), (5, 7), (8, 14))[band]
        if lo > limit:
            lo, hi = 3, min(5, limit)
        return int(rng.integers(lo, min(hi, limit) + 1))

    fh, fw = fixed_size if fixed_size is not None else (dimension(height), dimension(width))
    max_distance = (height - fh if direction in ("up", "down") else width - fw)
    if max_distance < 0:
        return None
    selector = int(rng.integers(0, 4))
    if selector == 0:
        distance = 0
    elif selector == 1:
        distance = min(1, max_distance)
    elif selector == 2:
        distance = int(rng.integers(0, min(5, max_distance) + 1))
    else:
        distance = int(rng.integers(0, max_distance + 1))
    if direction == "up":
        top = distance
        left = int(rng.integers(0, width - fw + 1))
    elif direction == "down":
        top = height - fh - distance
        left = int(rng.integers(0, width - fw + 1))
    elif direction == "left":
        left = distance
        top = int(rng.integers(0, height - fh + 1))
    else:
        left = width - fw - distance
        top = int(rng.integers(0, height - fh + 1))
    offset = int(rng.integers(1, (fw if direction in ("up", "down") else fh) - 1))
    return _rectangle(top, left, fh, fw, direction, offset)


def _compatible(rect, others, height, width):
    if any(_inside(p, rect) for p in ((0, 0), (0, width - 1),
                                       (height - 1, 0), (height - 1, width - 1))):
        return False
    for old in others:
        if not _separate(rect, old):
            return False
        if set(_ray(rect, height, width)) & set(_ray(old, height, width)):
            return False
        if any(_inside(p, old) for p in _ray(rect, height, width)[1:]):
            return False
        if any(_inside(p, rect) for p in _ray(old, height, width)[1:]):
            return False
    return True


def _crossing_pair(rng, height, width):
    # A vertical down ray and a horizontal right ray meet in empty background.
    ah = int(rng.integers(3, 6))
    aw = int(rng.integers(3, 6))
    bh = int(rng.integers(3, 6))
    bw = int(rng.integers(3, 6))
    atop = int(rng.integers(1, 3))
    bleft = int(rng.integers(1, 3))
    aleft = int(rng.integers(max(bw + bleft + 2, width // 2), width - aw))
    btop = int(rng.integers(max(atop + ah + 2, height // 2), height - bh))
    a = _rectangle(atop, aleft, ah, aw, "down", int(rng.integers(1, aw - 1)))
    b = _rectangle(btop, bleft, bh, bw, "right", int(rng.integers(1, bh - 1)))
    return [a, b]


def _place(rng, height, width, directions, crossing=False, short=False, large=False):
    if crossing:
        return _crossing_pair(rng, height, width)
    rects = []
    fixed_size = None
    if len(directions) >= 2 and rng.random() < 0.24:
        upper = min(5 if len(directions) >= 4 or short else 8, height, width)
        side = int(rng.integers(3, upper + 1))
        fixed_size = (side, side) if rng.random() < 0.55 else (
            side, int(rng.integers(3, upper + 1)))
    for direction in directions:
        for _ in range(450):
            rect = _candidate(rng, height, width, direction, short, large, fixed_size)
            if rect is not None and _compatible(rect, rects, height, width):
                rects.append(rect)
                break
        else:
            return None
    return rects


def _draw(grid, rect, color, background):
    t, l, b, r = rect["top"], rect["left"], rect["bottom"], rect["right"]
    grid[t, l:r + 1] = color
    grid[b, l:r + 1] = color
    grid[t:b + 1, l] = color
    grid[t:b + 1, r] = color
    grid[rect["gap"]] = background


def _clutter(grid, rect, color, rng):
    t, l, b, r = rect["top"], rect["left"], rect["bottom"], rect["right"]
    if b - t < 4 or r - l < 4:
        return False
    candidates = []
    for row in range(t + 1, b):
        for col in range(l + 1, r):
            if abs(row - rect["gap"][0]) + abs(col - rect["gap"][1]) <= 1:
                continue
            if row in (t + 1, b - 1) or col in (l + 1, r - 1):
                candidates.append((row, col))
    if not candidates:
        return False
    grid[candidates[int(rng.integers(len(candidates)))]] = color
    return True


def _answer(input_grid, rects, signal):
    result = input_grid.copy()
    height, width = result.shape
    for rect in rects:
        result[rect["top"] + 1:rect["bottom"], rect["left"] + 1:rect["right"]] = signal
        for point in _ray(rect, height, width):
            result[point] = signal
    return result


def _size(rng, count, regime, crossing, compact=False, large=False):
    if crossing:
        return int(rng.integers(15, 23)), int(rng.integers(16, 25))
    if compact:
        return int(rng.integers(7, 11)), int(rng.integers(7, 12))
    if large or count >= 5:
        return int(rng.integers(22, 31)), int(rng.integers(22, 31))
    if count >= 4:
        return int(rng.integers(18, 31)), int(rng.integers(18, 31))
    if count == 3:
        low = 13
    elif count == 2:
        low = 9
    else:
        low = 6
    mode = int(rng.choice(3, p=[0.25, 0.45, 0.30] if regime == "source_train" else [0.20, 0.35, 0.45]))
    hi = (13, 21, 30)[mode]
    low = min(low, hi)
    return int(rng.integers(low, hi + 1)), int(rng.integers(low, hi + 1))


def _pair(rng, regime, directions, background, colors, signal, crossing=False,
          clutter=False, compact=False, large=False, visible=False):
    count = len(directions)
    for attempt in range(140):
        fallback = attempt >= 90 and count >= 4 and not crossing
        height, width = ((30, 30) if fallback else
                         _size(rng, count, regime, crossing, compact, large))
        rects = _place(rng, height, width, directions, crossing,
                       compact or fallback, large and not fallback)
        if rects is None:
            continue
        if visible and not any(len(_ray(r, height, width)) > 1 for r in rects):
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for rect, color in zip(rects, colors):
            _draw(grid, rect, color, background)
        if clutter:
            eligible = [i for i, rect in enumerate(rects) if rect["bottom"] - rect["top"] >= 4
                        and rect["right"] - rect["left"] >= 4]
            if not eligible:
                continue
            i = int(rng.choice(eligible))
            _clutter(grid, rects[i], colors[i], rng)
        output = _answer(grid, rects, signal)
        return {"input": grid, "output": output}
    raise RuntimeError("could not construct frame layout")


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.ravel())


def _rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _direction_plan(rng, train_count, test_count, heldout, crossing_train, crossing_test):
    directions = list(_DIRECTIONS)
    rng.shuffle(directions)
    train_allowed = directions[:3] if heldout else directions
    train_counts = [int(rng.choice([1, 2, 3, 4, 5], p=[0.22, 0.30, 0.26, 0.17, 0.05])) for _ in range(train_count)]
    if crossing_train is not None:
        train_counts[crossing_train] = 2
    while sum(train_counts) < len(train_allowed):
        i = int(rng.integers(train_count))
        train_counts[i] += 1
    train_plan = [[None] * count for count in train_counts]
    if crossing_train is not None:
        train_plan[crossing_train] = ["down", "right"]
    missing = [d for d in train_allowed if not any(d in row for row in train_plan)]
    slots = [(i, j) for i, row in enumerate(train_plan) for j, d in enumerate(row) if d is None]
    while len(slots) < len(missing):
        i = min((index for index in range(train_count) if index != crossing_train),
                key=lambda index: len(train_plan[index]))
        train_plan[i].append(None)
        slots.append((i, len(train_plan[i]) - 1))
    rng.shuffle(slots)
    for d in missing:
        i, j = slots.pop()
        train_plan[i][j] = d
    for i, j in slots:
        train_plan[i][j] = str(rng.choice(train_allowed))
    test_plan = []
    for index in range(test_count):
        if index == crossing_test:
            test_plan.append(["down", "right"])
            continue
        count = int(rng.choice([1, 2, 3, 4, 5], p=[0.15, 0.24, 0.28, 0.23, 0.10]))
        row = [str(rng.choice(train_allowed)) for _ in range(count)]
        if heldout and index == 0:
            row[int(rng.integers(count))] = directions[3]
        test_plan.append(row)
    return train_plan, test_plan


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _counts(rng, num_train, num_test)
    palette = list(map(int, rng.permutation(10)))
    signal = palette[0]
    structural_pool = palette[1:4]
    background_pool = palette[4:]
    color_mode = int(rng.integers(3))  # shared, per pair, or several colors per grid
    constant_background = bool(rng.random() < 0.27)
    heldout = bool(rng.random() < 0.15)
    clutter_mode = bool(rng.random() < 0.16)
    crossing_mode = bool(rng.random() < 0.14)
    crossing_train = int(rng.integers(num_train)) if crossing_mode else None
    crossing_test = int(rng.integers(num_test)) if crossing_mode else None
    train_plan, test_plan = _direction_plan(rng, num_train, num_test, heldout,
                                            crossing_train, crossing_test)
    all_plan = train_plan + test_plan
    train_backgrounds = ([background_pool[0]] * num_train if constant_background else
                         [int(rng.choice(background_pool)) for _ in range(num_train)])
    test_backgrounds = ([background_pool[0]] * num_test if constant_background else
                        [int(rng.choice(background_pool if rng.random() < 0.45 else train_backgrounds))
                         for _ in range(num_test)])
    pair_colors = []
    if color_mode == 0:
        pair_colors = [[structural_pool[0]] * len(row) for row in all_plan]
    elif color_mode == 1:
        selected = [structural_pool[0], structural_pool[1]]
        train_choices = [selected[i % 2] for i in range(num_train)]
        rng.shuffle(train_choices)
        pair_colors = [[train_choices[i]] * len(row) for i, row in enumerate(train_plan)]
        pair_colors += [[int(rng.choice(selected))] * len(row) for row in test_plan]
    else:
        selected = [structural_pool[0], structural_pool[1], structural_pool[2]]
        pair_colors = [[int(rng.choice(selected)) for _ in row] for row in all_plan]
        multi_indices = [i for i, row in enumerate(train_plan) if len(row) >= 2]
        if multi_indices:
            i = multi_indices[int(rng.integers(len(multi_indices)))]
            pair_colors[i][0:2] = selected[:2]
        seen = {c for row in pair_colors[:num_train] for c in row}
        for row in pair_colors[num_train:]:
            for j, color in enumerate(row):
                if color not in seen:
                    row[j] = int(rng.choice(tuple(seen)))
    if color_mode == 2 and not any(len(set(row)) > 1 for row in pair_colors[:num_train]):
        color_mode = 1
    pairs = {"train": [], "test": []}
    used = set()
    for split, plans, backgrounds, offset in (("train", train_plan, train_backgrounds, 0),
                                                ("test", test_plan, test_backgrounds, num_train)):
        for i, (directions, background) in enumerate(zip(plans, backgrounds)):
            crossing = crossing_mode and i == (crossing_train if split == "train" else crossing_test)
            clutter = clutter_mode and (i == 0 if split == "train" else rng.random() < 0.45)
            # Compact and large examples occur in both splits; count and shape remain visible in input.
            compact = len(directions) == 1 and rng.random() < 0.32
            large = rng.random() < (0.15 if split == "train" else 0.21)
            if clutter:
                compact = False
            if crossing:
                compact = large = False
            for _ in range(12):
                pair = _pair(rng, "source_train" if split == "train" else "target_test",
                             directions, background, pair_colors[offset + i], signal,
                             crossing, clutter, compact, large, split == "train" and i == 0)
                key = _key(pair["input"])
                if key not in used and (_rows(pair["input"]), _rows(pair["output"])) not in _SOURCE_PAIR_ROWS:
                    used.add(key)
                    pairs[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make a unique pair")
    validate_generated_puzzle(pairs)
    return pairs


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    answer = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            component = []
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                for dr, dc in _STEPS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            answer.append(component)
    return answer


def _inspect_pair(input_grid, output_grid, signal):
    _require(input_grid.shape == output_grid.shape, "shape changed")
    height, width = input_grid.shape
    background = int(input_grid[0, 0])
    _require(all(int(input_grid[p]) == background for p in
                 ((0, width - 1), (height - 1, 0), (height - 1, width - 1))),
             "background corners disagree")
    _require(signal not in set(map(int, np.unique(input_grid))), "signal in input")
    components = _components(input_grid != background)
    _require(bool(components), "no frames")
    rects = []
    colors = []
    clutter = 0
    for component in components:
        color = int(input_grid[component[0]])
        _require(all(int(input_grid[p]) == color for p in component), "multicolor frame")
        t = min(p[0] for p in component)
        b = max(p[0] for p in component)
        l = min(p[1] for p in component)
        r = max(p[1] for p in component)
        _require(b - t >= 2 and r - l >= 2, "frame too small")
        perimeter = ([(t, c) for c in range(l, r + 1)] + [(b, c) for c in range(l, r + 1)] +
                     [(row, l) for row in range(t + 1, b)] + [(row, r) for row in range(t + 1, b)])
        gaps = [p for p in perimeter if int(input_grid[p]) != color]
        _require(len(gaps) == 1 and int(input_grid[gaps[0]]) == background, "not one missing border cell")
        gap = gaps[0]
        _require(gap not in ((t, l), (t, r), (b, l), (b, r)), "corner gap")
        direction = "up" if gap[0] == t else "down" if gap[0] == b else "left" if gap[1] == l else "right"
        rect = dict(top=t, left=l, bottom=b, right=r, gap=gap, direction=direction)
        interior = [(row, col) for row in range(t + 1, b) for col in range(l + 1, r)
                    if int(input_grid[row, col]) != background]
        _require(len(interior) <= 1, "too much interior clutter")
        for p in interior:
            _require(int(input_grid[p]) == color, "foreign interior color")
            _require(abs(p[0] - gap[0]) + abs(p[1] - gap[1]) > 1, "clutter blocks outlet")
            _require(any((abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 and
                          int(input_grid[q]) == color) for q in perimeter), "clutter detached")
        clutter += len(interior)
        rects.append(rect)
        colors.append(color)
    for i, rect in enumerate(rects):
        for old in rects[:i]:
            _require(_separate(rect, old), "frames crowded")
        for point in _ray(rect, height, width)[1:]:
            _require(int(input_grid[point]) == background, "ray hits frame")
    expected = _answer(input_grid, rects, signal)
    _require(np.array_equal(expected, output_grid), "wrong cavity or ray")
    _require(not np.array_equal(input_grid, output_grid), "no transformation")
    rays = [set(_ray(r, height, width)) for r in rects]
    crossings = sum(bool(a & b) for i, a in enumerate(rays) for b in rays[:i])
    return dict(background=background, colors=colors, directions=[r["direction"] for r in rects],
                frames=len(rects), clutter=clutter, crossings=crossings,
                exteriors=[len(ray) - 1 for ray in rays], rects=rects)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "bad counts")
    used = set()
    signal = None
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
        for grid in pair.values():
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid not 2D NumPy")
            _require(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
            _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid dimensions")
            _require(bool(np.all((0 <= grid) & (grid <= 9))), "color range")
        key = _key(pair["input"])
        _require(key not in used, "duplicate input")
        used.add(key)
        _require((_rows(pair["input"]), _rows(pair["output"])) not in _SOURCE_PAIR_ROWS,
                 "source pair copied")
    for pair in train:
        introduced = set(map(int, np.unique(pair["output"]))) - set(map(int, np.unique(pair["input"])))
        _require(len(introduced) == 1, "training signal not identified")
        found = next(iter(introduced))
        if signal is None:
            signal = found
        _require(found == signal, "inconsistent signal")
    train_info = [_inspect_pair(p["input"], p["output"], signal) for p in train]
    test_info = [_inspect_pair(p["input"], p["output"], signal) for p in test]
    train_directions = {d for info in train_info for d in info["directions"]}
    test_directions = {d for info in test_info for d in info["directions"]}
    _require(len(train_directions) >= 3, "insufficient directional evidence")
    _require(len(test_directions - train_directions) <= 1, "too many unshown directions")
    _require(any(e > 0 for info in train_info for e in info["exteriors"]), "no visible train ray")
    train_colors = {c for info in train_info for c in info["colors"]}
    test_colors = {c for info in test_info for c in info["colors"]}
    _require(test_colors <= train_colors, "unshown frame color")
    if any(len(set(info["colors"])) > 1 for info in test_info):
        _require(any(len(set(info["colors"])) > 1 for info in train_info), "multicolor frames unshown")
    if any(info["clutter"] for info in test_info):
        _require(any(info["clutter"] for info in train_info), "clutter overwrite unshown")
    if any(info["crossings"] for info in test_info):
        _require(any(info["crossings"] for info in train_info), "ray crossing unshown")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    for seed, train, test in ((17, None, None), (901, 6, 3), (2100, 2, 1),
                              (2101, 10, 1), (2102, 10, 4), (2103, None, 1),
                              (2104, None, 4), (2105, 2, None), (2106, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "explicit train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "explicit test count ignored")
    a = generate_puzzle(1203, 5, 2)
    _require(_equal(a, generate_puzzle(1203, 5, 2)), "not deterministic")
    _require(not _equal(a, generate_puzzle(1204, 5, 2)), "seeds identical")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(3100, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts, shapes, layouts, palette, inputs = set(), set(), set(), set(), set(), set()
    frames, lengths, directions = set(), set(), set()
    clean, clutter, crossing, multicolor, same_direction = 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for pair in puzzle["train"] + puzzle["test"]:
            grid = pair["input"]
            shapes.add(grid.shape)
            inputs.add(_key(grid))
            introduced = set(map(int, np.unique(pair["output"]))) - set(map(int, np.unique(grid)))
            signal = next(iter(introduced))
            info = _inspect_pair(grid, pair["output"], signal)
            frames.add(info["frames"])
            lengths.update(info["exteriors"])
            directions.update(info["directions"])
            clean += info["clutter"] == 0
            clutter += info["clutter"] > 0
            crossing += info["crossings"] > 0
            multicolor += len(set(info["colors"])) > 1
            same_direction += info["frames"] >= 2 and len(set(info["directions"])) == 1
            palette.add((info["background"], tuple(sorted(set(info["colors"]))), signal))
            layouts.add((grid.shape, tuple(sorted((r["bottom"] - r["top"] + 1,
                       r["right"] - r["left"] + 1, r["top"], r["left"], r["direction"])
                       for r in info["rects"]))))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _require(len(shapes) >= 30 and len(layouts) >= 100, "geometry diversity")
    _require({1, 2, 3, 4} <= frames and {0, 1} <= lengths and len(directions) == 4,
             "frame, ray or direction diversity")
    _require(clean > clutter > 0 and crossing > 0 and multicolor > 0 and same_direction > 0,
             "review feature diversity")
    _require(len(palette) >= 20 and len(inputs) >= 100, "palette/input diversity")
