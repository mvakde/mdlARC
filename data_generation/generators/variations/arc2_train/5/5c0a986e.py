"""Color-coded diagonal extensions from square seeds."""

import numpy as np


QUADRANTS = (
    "upper_above_left_lower", "upper_above_right_lower",
    "lower_above_left_upper", "lower_above_right_upper",
)
ALIGNMENTS = ("same_rows", "overlap_rows", "same_cols", "overlap_cols")
LAYOUTS = QUADRANTS + ALIGNMENTS
SOURCE_POSITIONS = {(7, 6, 0, 2), (2, 2, 6, 4), (5, 3, 2, 5), (3, 6, 5, 2)}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _counts(seed, num_train, num_test):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer between {low} and {high}")
    train, test = num_train, num_test
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + int(seed) % 4
        train = test + 1 + (int(seed) // 4) % (10 - test)
    elif train is None:
        train = test + 1 + int(seed) % (10 - test)
    else:
        test = 1 + int(seed) % min(4, train - 1)
    return int(train), int(test)


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _ray(row, col, size, height, width, role):
    if role == "upper":
        row, col, step = row - 1, col - 1, -1
    else:
        row, col, step = row + size, col + size, 1
    cells = []
    while 0 <= row < height and 0 <= col < width:
        cells.append((row, col))
        row += step
        col += step
    return tuple(cells)


def _exit(row, col, size, height, width, role):
    if role == "upper":
        if min(row, col) == 0:
            return "edge"
        return "corner" if row == col else ("top" if row < col else "left")
    bottom, right = height - row - size, width - col - size
    if min(bottom, right) == 0:
        return "edge"
    return "corner" if bottom == right else ("bottom" if bottom < right else "right")


def _layout(upper, lower, size):
    ur, uc = upper
    lr, lc = lower
    if ur + size <= lr and uc + size <= lc:
        return QUADRANTS[0]
    if ur + size <= lr and lc + size <= uc:
        return QUADRANTS[1]
    if lr + size <= ur and lc + size <= uc:
        return QUADRANTS[2]
    if lr + size <= ur and uc + size <= lc:
        return QUADRANTS[3]
    if ur < lr + size and lr < ur + size:
        return "same_rows" if ur == lr else "overlap_rows"
    return "same_cols" if uc == lc else "overlap_cols"


def _geometry_ok(upper, lower, size):
    ur, uc, uray = upper[:3]
    lr, lc, lray = lower[:3]
    if ur < lr + size and lr < ur + size and uc < lc + size and lc < uc + size:
        return False
    if set(uray).intersection(lray):
        return False
    if any(lr <= r < lr + size and lc <= c < lc + size for r, c in uray):
        return False
    if any(ur <= r < ur + size and uc <= c < uc + size for r, c in lray):
        return False
    return bool(uray or lray)


def _is_source(shape, size, background, upper_color, lower_color, upper, lower):
    return (shape == (10, 10) and size == 2 and background == 0
            and upper_color == 1 and lower_color == 2
            and (upper[0], upper[1], lower[0], lower[1]) in SOURCE_POSITIONS)


def _shape(rng, mode, size):
    if mode == "source":
        return 10, 10
    if mode == "compact":
        side = int(rng.choice((6, 6, 7, 8) if size == 2 else (8, 8, 9)))
        return side, side
    if mode == "square":
        sides = (6, 7, 8, 9, 11, 12, 14, 16, 18, 20) if size == 2 else (8, 9, 11, 12, 14, 16, 18, 20)
        side = int(rng.choice(sides))
        return side, side
    if mode == "large":
        return int(rng.choice((20, 22, 24, 26, 28, 30))), int(rng.choice((20, 22, 24, 26, 28, 30)))
    low = max(6, size + 3)
    for _ in range(20):
        h, w = int(rng.integers(low, 21)), int(rng.integers(low, 21))
        if h != w:
            return h, w
    return low, 10


def _places(height, width, size, role):
    return [(r, c, _ray(r, c, size, height, width, role),
             _exit(r, c, size, height, width, role))
            for r in range(height - size + 1) for c in range(width - size + 1)]


def _make_pair(rng, colors, background, size, shape_mode, layout_target,
               exits, length_target, spacing_target, allow_edge, used_inputs,
               used_specs):
    upper_color, lower_color = colors
    for _ in range(8):
        height, width = _shape(rng, shape_mode, size)
        upper_all = _places(height, width, size, "upper")
        lower_all = _places(height, width, size, "lower")
        if not allow_edge:
            upper_all = [p for p in upper_all if p[2]]
            lower_all = [p for p in lower_all if p[2]]
        upper_wanted = [p for p in upper_all if p[3] == exits[0]] or upper_all
        lower_wanted = [p for p in lower_all if p[3] == exits[1]] or lower_all
        best, best_score = [], float("inf")
        for trial in range(1500):
            # Search all four preference combinations; an exit preference
            # can be geometrically incompatible with the desired layout.
            upool = upper_wanted if trial % 4 >= 2 else upper_all
            lpool = lower_wanted if trial % 4 in (1, 3) else lower_all
            upper = upool[int(rng.integers(len(upool)))]
            lower = lpool[int(rng.integers(len(lpool)))]
            if not _geometry_ok(upper, lower, size):
                continue
            if _is_source((height, width), size, background, upper_color, lower_color, upper, lower):
                continue
            spec = (height, width, size, background, upper_color, lower_color,
                    upper[0], upper[1], lower[0], lower[1])
            if spec in used_specs:
                continue
            ray_total = len(upper[2]) + len(lower[2])
            if max(height, width) >= 18 and ray_total + 2 * size * size < max(14, int(0.05 * height * width)):
                continue
            layout = _layout(upper[:2], lower[:2], size)
            distance = abs(upper[0] - lower[0]) + abs(upper[1] - lower[1])
            span = height + width
            spacing_ok = (distance <= 0.30 * span if spacing_target == "near"
                          else max(abs(upper[0] - lower[0]), abs(upper[1] - lower[1]))
                          >= 0.50 * max(height, width))
            if length_target == "short":
                length_penalty = sum(max(0, len(p[2]) - 2) for p in (upper, lower))
            elif length_target == "long":
                length_penalty = 3 * sum(max(0, 5 - len(p[2])) for p in (upper, lower))
            else:
                length_penalty = sum(abs(len(p[2]) - 4) for p in (upper, lower))
            score = (60 * (layout != layout_target)
                     + 12 * (upper[3] != exits[0]) + 12 * (lower[3] != exits[1])
                     + 8 * (not spacing_ok) + length_penalty)
            if score < best_score:
                best_score, best = score, [(upper, lower)]
            elif score == best_score:
                best.append((upper, lower))
        if not best:
            continue
        upper, lower = best[int(rng.integers(len(best)))]
        inp = np.full((height, width), background, dtype=np.int8)
        inp[upper[0]:upper[0] + size, upper[1]:upper[1] + size] = upper_color
        inp[lower[0]:lower[0] + size, lower[1]:lower[1] + size] = lower_color
        key = _grid_key(inp)
        if key in used_inputs:
            continue
        out = inp.copy()
        for r, c in upper[2]:
            out[r, c] = upper_color
        for r, c in lower[2]:
            out[r, c] = lower_color
        used_inputs.add(key)
        used_specs.add((height, width, size, background, upper_color, lower_color,
                        upper[0], upper[1], lower[0], lower[1]))
        return {"input": inp, "output": out}
    raise RuntimeError("unable to construct a distinct collision-free pair")


def _layout_schedule(rng, train_count, test_count):
    mode = str(rng.choice(("rotating", "familiar", "similar", "mixed")))
    if mode == "rotating":
        holdout = str(rng.choice(QUADRANTS))
        choices = [x for x in QUADRANTS if x != holdout]
        rng.shuffle(choices)
        train = [choices[i % len(choices)] for i in range(train_count)]
        test = [holdout] * test_count
    elif mode == "familiar":
        choices = list(LAYOUTS)
        rng.shuffle(choices)
        train = [choices[i % len(choices)] for i in range(train_count)]
        test = [train[int(rng.integers(len(train)))] for _ in range(test_count)]
    elif mode == "similar":
        common = str(rng.choice(LAYOUTS))
        train, test = [common] * train_count, [common] * test_count
    else:
        choices = list(LAYOUTS)
        rng.shuffle(choices)
        train = [choices[i % len(choices)] for i in range(train_count)]
        test = [str(rng.choice(LAYOUTS)) for _ in range(test_count)]
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    upper_color, lower_color = (int(v) for v in rng.choice(10, size=2, replace=False))
    backgrounds = [c for c in range(10) if c not in (upper_color, lower_color)]
    rng.shuffle(backgrounds)
    if bool(rng.integers(2)):
        train_bgs = [backgrounds[i % len(backgrounds)] for i in range(train_count)]
        test_bgs = [backgrounds[int(rng.integers(len(backgrounds)))] for _ in range(test_count)]
    else:
        train_bgs = [backgrounds[0]] * train_count
        test_bgs = [backgrounds[0]] * test_count
    mixed_scale = bool(rng.integers(2))
    train_sizes = [2] * train_count
    if mixed_scale:
        train_sizes[1] = 3
        for i in range(2, train_count):
            train_sizes[i] = 2 + int(rng.integers(2))
    test_sizes = [2 + int(mixed_scale and rng.integers(2)) for _ in range(test_count)]
    train_layouts, test_layouts = _layout_schedule(rng, train_count, test_count)
    modes = ("source", "compact", "square", "rectangle", "large")
    train_modes = [str(rng.choice(modes, p=(0.25, 0.10, 0.20, 0.35, 0.10))) for _ in range(train_count)]
    if train_modes[0] == train_modes[1]:
        train_modes[1] = "rectangle" if train_modes[0] != "rectangle" else "source"
    test_modes = [str(rng.choice(modes, p=(0.25, 0.12, 0.18, 0.27, 0.18))) for _ in range(test_count)]
    puzzle, used_inputs, used_specs = {"train": [], "test": []}, set(), set()
    for split, count, layouts, sizes, shape_modes, bgs in (
        ("train", train_count, train_layouts, train_sizes, train_modes, train_bgs),
        ("test", test_count, test_layouts, test_sizes, test_modes, test_bgs),
    ):
        for i in range(count):
            allow_edge = (split == "test" or i >= 2) and bool(rng.random() < 0.09)
            upper_exit = ("edge" if allow_edge and bool(rng.integers(2))
                          else str(rng.choice(("top", "left", "corner"), p=(0.39, 0.39, 0.22))))
            lower_exit = ("edge" if allow_edge and upper_exit != "edge"
                          else str(rng.choice(("bottom", "right", "corner"), p=(0.39, 0.39, 0.22))))
            puzzle[split].append(_make_pair(
                rng, (upper_color, lower_color), bgs[i], sizes[i], shape_modes[i],
                layouts[i], (upper_exit, lower_exit),
                str(rng.choice(("short", "medium", "long"), p=(0.25, 0.30, 0.45))),
                str(rng.choice(("near", "far"))), allow_edge, used_inputs,
                used_specs,
            ))
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must have two dimensions")
    _require(all(1 <= n <= 30 for n in grid.shape), f"{label} has invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must have integer cells")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), f"{label} has invalid colors")


def _inspect_input(grid):
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) == 3, "input must have two squares and one background")
    objects = {}
    for raw_color, raw_count in zip(values, counts):
        color, count = int(raw_color), int(raw_count)
        if count not in (4, 9):
            continue
        size = 2 if count == 4 else 3
        points = np.argwhere(grid == color)
        row, col = (int(v) for v in points.min(axis=0))
        if (int(points[:, 0].max()) == row + size - 1
                and int(points[:, 1].max()) == col + size - 1
                and bool(np.all(grid[row:row + size, col:col + size] == color))):
            objects[color] = (row, col, size)
    _require(len(objects) == 2, "input must contain two solid squares")
    _require(len({o[2] for o in objects.values()}) == 1, "both squares must share a size")
    background = next((int(v) for v in values if int(v) not in objects), None)
    _require(background is not None, "background is ambiguous")
    size = next(iter(objects.values()))[2]
    _require(int(np.count_nonzero(grid == background)) == grid.size - 2 * size * size,
             "input contains unrelated cells")
    return background, objects, size


def _inspect_pair(inp, out):
    background, objects, size = _inspect_input(inp)
    colors = tuple(objects)
    solutions = []
    for upper_color, lower_color in ((colors[0], colors[1]), (colors[1], colors[0])):
        ur, uc, _ = objects[upper_color]
        lr, lc, _ = objects[lower_color]
        upper = (ur, uc, _ray(ur, uc, size, *inp.shape, "upper"),
                 _exit(ur, uc, size, *inp.shape, "upper"))
        lower = (lr, lc, _ray(lr, lc, size, *inp.shape, "lower"),
                 _exit(lr, lc, size, *inp.shape, "lower"))
        if not _geometry_ok(upper, lower, size):
            continue
        expected = inp.copy()
        for row, col in upper[2]:
            expected[row, col] = upper_color
        for row, col in lower[2]:
            expected[row, col] = lower_color
        if np.array_equal(expected, out):
            solutions.append((upper_color, lower_color, upper, lower))
    _require(len(solutions) == 1, "pair must uniquely establish roles and exact rays")
    upper_color, lower_color, upper, lower = solutions[0]
    return {"background": background, "upper_color": upper_color,
            "lower_color": lower_color, "size": size, "upper": upper,
            "lower": lower, "layout": _layout(upper[:2], lower[:2], size)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    details = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} needs exactly input and output")
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp, label + ".input")
            _validate_grid(out, label + ".output")
            _require(inp.shape == out.shape and inp.dtype == out.dtype,
                     f"{label} must preserve shape and dtype")
            _require(not np.array_equal(inp, out), f"{label} has no visible change")
            info = _inspect_pair(inp, out)
            _require(not _is_source(inp.shape, info["size"], info["background"],
                                    info["upper_color"], info["lower_color"],
                                    info["upper"], info["lower"]), f"{label} copies a source pair")
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     f"{label} duplicates another pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            details[split].append(info)
    roles = {(d["upper_color"], d["lower_color"]) for ds in details.values() for d in ds}
    _require(len(roles) == 1, "directional colors must persist throughout the puzzle")
    train_sizes = {d["size"] for d in details["train"]}
    _require(all(d["size"] in train_sizes for d in details["test"]),
             "a test square size is absent from training")
    _require(any(d["upper"][2] for d in details["train"])
             and any(d["lower"][2] for d in details["train"]),
             "training must visibly establish both color roles")
    train_bgs = {d["background"] for d in details["train"]}
    if len(train_bgs) == 1:
        _require(all(d["background"] in train_bgs for d in details["test"]),
                 "a test background change is not demonstrated in training")
    profiles = {(p["input"].shape, d["size"], d["upper"][:2], d["lower"][:2])
                for p, d in zip(puzzle["train"], details["train"])}
    _require(len(profiles) >= 2, "training needs structural variation")


def self_test() -> None:
    train_counts, test_counts, puzzle_keys = set(), set(), set()
    layouts, shapes, palettes, profiles = set(), set(), set(), set()
    train_exits, test_exits = set(), set()
    varying_bgs = constant_bgs = mixed_sizes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        info = {split: [_inspect_pair(p["input"], p["output"]) for p in puzzle[split]]
                for split in ("train", "test")}
        varying_bgs += len({d["background"] for d in info["train"]}) > 1
        constant_bgs += len({d["background"] for d in info["train"]}) == 1
        mixed_sizes += len({d["size"] for d in info["train"]}) > 1
        for split in ("train", "test"):
            for pair, d in zip(puzzle[split], info[split]):
                shapes.add(pair["input"].shape)
                layouts.add(d["layout"])
                palettes.add((d["upper_color"], d["lower_color"], d["background"]))
                profiles.add((pair["input"].shape, d["size"], d["layout"],
                              len(d["upper"][2]), len(d["lower"][2])))
                (train_exits if split == "train" else test_exits).update(
                    (d["upper"][3], d["lower"][3]))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    _require(len(puzzle_keys) == 42, "seeds should yield distinct puzzles")
    _require(layouts == set(LAYOUTS), "all quadrant and alignment layouts must occur")
    needed_exits = {"top", "left", "bottom", "right", "corner"}
    _require(needed_exits <= train_exits and needed_exits <= test_exits,
             "both splits need all side and corner exits")
    _require((10, 10) in shapes and (6, 6) in shapes and any(max(s) == 30 for s in shapes),
             "source, compact, and large canvases must occur")
    _require(len(shapes) >= 35 and len(palettes) >= 50 and len(profiles) >= 70,
             "cross-seed structural diversity is too low")
    _require(varying_bgs >= 8 and constant_bgs >= 8 and mixed_sizes >= 8,
             "background or square-size regimes are missing")
    for seed, nt, ns in ((1000, 2, 1), (1001, 10, 1), (1002, 10, 4), (1003, 6, 2)):
        puzzle = generate_puzzle(seed, nt, ns)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns),
                 "explicit counts were ignored")
    for seed, nt, ns in ((1100, None, 1), (1101, None, 4), (1102, 2, None), (1103, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        _require((nt is None or len(puzzle["train"]) == nt)
                 and (ns is None or len(puzzle["test"]) == ns), "partial counts were ignored")
    _require(_puzzle_key(generate_puzzle(1200, 7, 3)) == _puzzle_key(generate_puzzle(1200, 7, 3)),
             "explicit generation must be deterministic")
    _require(_puzzle_key(generate_puzzle(1201)) == _puzzle_key(generate_puzzle(1201)),
             "automatic generation must be deterministic")
    _require(_puzzle_key(generate_puzzle(1202, 6, 2)) != _puzzle_key(generate_puzzle(1203, 6, 2)),
             "different seeds should differ")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3},
                   {"num_train": 3.0, "num_test": 1},
                   {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(1300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
