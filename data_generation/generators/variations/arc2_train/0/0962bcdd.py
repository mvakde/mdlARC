"""Generate complete, independently colored cross-expansion ARC puzzles."""

import numpy as np

CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
SOURCE_SIGNATURES = {
    (12, 12, 0, ((3, 3, 6, 8), (9, 8, 6, 8))),
    (12, 12, 0, ((3, 2, 2, 7), (7, 8, 2, 7))),
    (12, 12, 0, ((2, 8, 4, 3), (7, 2, 4, 3))),
}


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(seed, num_train, num_test):
    if num_train is not None:
        if not _integer(num_train) or not 2 <= num_train <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _integer(num_test) or not 1 <= num_test <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        lower = max(2, num_test + 1)
        num_train = lower + (seed // 4) % (11 - lower)
    elif num_train is None:
        lower = max(2, num_test + 1)
        num_train = lower + seed % (11 - lower)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _choose_count(rng):
    return int(rng.choice(np.arange(1, 9),
                          p=[.17, .25, .17, .14, .10, .08, .05, .04]))


def _shape(rng, count, forced_style=None):
    if forced_style in ("down_right", "down_left"):
        return int(rng.integers(10, 31)), int(rng.integers(10, 31))
    if forced_style == "row":
        return int(rng.integers(5, 31)), int(rng.integers(10, 31))
    if forced_style == "column":
        return int(rng.integers(10, 31)), int(rng.integers(5, 31))
    if count == 1 and rng.random() < .20:
        return 5, 5
    mode = int(rng.integers(0, 6))
    if mode == 0:
        rows = int(rng.integers((count + 5) // 6, min(6, count) + 1))
        cols = (count + rows - 1) // rows
        height = 5 * rows + int(rng.integers(0, min(5, 31 - 5 * rows)))
        width = 5 * cols + int(rng.integers(0, min(5, 31 - 5 * cols)))
        return height, width
    if mode in (1, 2) and count <= 6:
        long_side = int(rng.integers(5 * count, 31))
        short_side = int(rng.integers(5, 10))
        return (short_side, long_side) if mode == 1 else (long_side, short_side)
    if mode == 3:
        return 30, 30
    for _ in range(32):
        height = int(rng.integers(5, 31))
        width = int(rng.integers(5, 31))
        if (height // 5) * (width // 5) >= count:
            return height, width
    return 30, 30


def _slot_offsets(rng, size, slots):
    extra = size - 5 * slots
    return sorted(int(x) for x in rng.integers(0, extra + 1, size=slots))


def _layout(rng, height, width, count, forced_style=None):
    if count == 2 and forced_style in (None, "down_right", "down_left") and rng.random() < .10:
        dense_offsets = [(dr, dc) for dr, dc in
                         ((4, 3), (4, -3), (-4, 3), (-4, -3),
                          (3, 4), (3, -4), (-3, 4), (-3, -4))
                         if height >= abs(dr) + 5 and width >= abs(dc) + 5
                         and (forced_style is None or
                              (dr * dc > 0) == (forced_style == "down_right"))]
        if dense_offsets:
            dr, dc = dense_offsets[int(rng.integers(len(dense_offsets)))]
            row = int(rng.integers(2 + max(0, -dr), height - 2 - max(0, dr)))
            col = int(rng.integers(2 + max(0, -dc), width - 2 - max(0, dc)))
            return tuple(sorted(((row, col), (row + dr, col + dc))))
    rows, cols = height // 5, width // 5
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    if count == 1:
        chosen = [slots[int(rng.integers(len(slots)))]]
    elif count == 2:
        classes = ["free"]
        if cols >= 2:
            classes.append("row")
        if rows >= 2:
            classes.append("column")
        if rows >= 2 and cols >= 2:
            classes.extend(("down_right", "down_left"))
        style = forced_style if forced_style is not None else classes[int(rng.integers(len(classes)))]
        if style == "row":
            r = int(rng.integers(rows))
            cc = rng.choice(cols, size=2, replace=False)
            chosen = [(r, int(c)) for c in cc]
        elif style == "column":
            c = int(rng.integers(cols))
            rr = rng.choice(rows, size=2, replace=False)
            chosen = [(int(r), c) for r in rr]
        elif style in ("down_right", "down_left"):
            rr = sorted(int(x) for x in rng.choice(rows, size=2, replace=False))
            cc = sorted(int(x) for x in rng.choice(cols, size=2, replace=False))
            if style == "down_left":
                cc.reverse()
            chosen = list(zip(rr, cc))
        else:
            chosen = [slots[int(i)] for i in rng.choice(len(slots), size=2, replace=False)]
    else:
        classes = ["free", "free"]
        if cols >= count:
            classes.append("row")
        if rows >= count:
            classes.append("column")
        style = classes[int(rng.integers(len(classes)))]
        if style == "row":
            r = int(rng.integers(rows))
            cc = rng.choice(cols, size=count, replace=False)
            chosen = [(r, int(c)) for c in cc]
        elif style == "column":
            c = int(rng.integers(cols))
            rr = rng.choice(rows, size=count, replace=False)
            chosen = [(int(r), c) for r in rr]
        else:
            chosen = [slots[int(i)] for i in rng.choice(len(slots), size=count, replace=False)]
    row_offsets = _slot_offsets(rng, height, rows)
    col_offsets = _slot_offsets(rng, width, cols)
    return tuple(sorted((5 * r + row_offsets[r] + 2,
                         5 * c + col_offsets[c] + 2) for r, c in chosen))


def _role_pairs(rng, colors, count, style=None, core=None):
    colors = tuple(int(c) for c in colors)
    if style is None:
        style = ("shared", "mixed", "exchange", "disjoint")[int(rng.integers(4))]
    if core is None:
        core = tuple(int(x) for x in rng.choice(colors, size=2, replace=False))
    pairs = [core]
    if count == 1:
        return pairs
    if style == "shared":
        return pairs * count
    if style == "exchange":
        return [core, core[::-1]] + [core if i % 2 == 0 else core[::-1]
                                     for i in range(2, count)]
    if style == "disjoint":
        shuffled = [int(x) for x in rng.permutation(colors)]
        if len(shuffled) >= 4:
            pairs[0] = tuple(shuffled[:2])
            pairs.append(tuple(shuffled[2:4]))
        else:
            pairs.append(core[::-1])
    else:
        candidates = [(a, b) for a in colors for b in colors if a != b and (a, b) != core]
        pairs.append(candidates[int(rng.integers(len(candidates)))])
    while len(pairs) < count:
        candidates = [(a, b) for a in colors for b in colors if a != b]
        pairs.append(candidates[int(rng.integers(len(candidates)))])
    return pairs


def _paint(height, width, background, crosses):
    inp = np.full((height, width), background, dtype=np.int8)
    for row, col, hub, spoke in crosses:
        inp[row, col] = hub
        for dr, dc in CARDINAL:
            inp[row + dr, col + dc] = spoke
    out = inp.copy()
    for row, col, hub, spoke in crosses:
        for dr, dc in CARDINAL:
            out[row + 2 * dr, col + 2 * dc] = spoke
        for distance in (1, 2):
            for dr, dc in DIAGONAL:
                out[row + distance * dr, col + distance * dc] = hub
    return inp, out


def _grid_key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_n, test_n = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # Color pools guide sampling only; each cross displays its own color roles.
    color_order = [int(x) for x in rng.permutation(10)]
    train_pool, new_pool = color_order[:5], color_order[5:]
    core, new_core = tuple(train_pool[:2]), tuple(new_pool[:2])
    count_policy = ("constant", "mixed", "larger_test")[int(rng.choice(3, p=[.16, .54, .30]))]
    test_color_policy = ("new", "familiar", "exchange", "mixed")[int(rng.integers(4))]
    layout_policy = ("source_shift", "reverse_shift", "unchanged", "mixed")[int(rng.integers(4))]
    directions = ("row", "column", "down_right", "down_left")
    if layout_policy == "source_shift":
        forced_layouts = {"train": ("down_right",), "test": ("down_left",)}
    elif layout_policy == "reverse_shift":
        forced_layouts = {"train": ("down_left",), "test": ("down_right",)}
    elif layout_policy == "unchanged":
        style = directions[int(rng.integers(4))]
        forced_layouts = {"train": (style,), "test": (style,)}
    else:
        train_styles = tuple(directions[int(i)] for i in rng.choice(4, size=2, replace=False))
        test_styles = tuple(directions[int(i)] for i in rng.choice(4, size=2, replace=False))
        forced_layouts = {"train": train_styles, "test": test_styles}
    constant_count = _choose_count(rng) if count_policy == "constant" else None
    if constant_count == 1 and test_color_policy == "mixed":
        constant_count = 2
    if count_policy == "constant" and constant_count != 2:
        forced_layouts = {"train": (), "test": ()}
    train_base = int(rng.integers(1, 4)) if count_policy == "larger_test" else None
    used_inputs, used_layouts = set(), set()
    puzzle = {"train": [], "test": []}
    first_background = None
    train_counts = []

    for split, amount in (("train", train_n), ("test", test_n)):
        for index in range(amount):
            if count_policy == "constant":
                count = constant_count
            elif count_policy == "larger_test":
                count = int(rng.integers(1, train_base + 1)) if split == "train" else int(rng.integers(train_base + 1, 9))
            else:
                count = _choose_count(rng)
                if split == "train" and index == 0:
                    count = max(2, count)
                if split == "train" and index == 1 and count == train_counts[0]:
                    count = 1 if count != 1 else 2
            if split == "test" and test_color_policy == "mixed":
                count = max(2, count)
            forced_style = forced_layouts[split][index] if index < len(forced_layouts[split]) else None
            if forced_style is not None:
                count = 2
            if split == "train":
                train_counts.append(count)

            if split == "train":
                role_pairs = _role_pairs(rng, train_pool, count,
                                         style="mixed" if index == 0 and count > 1 else None,
                                         core=core if index == 0 else None)
            elif test_color_policy == "new":
                role_pairs = _role_pairs(rng, new_pool, count, core=new_core)
            elif test_color_policy == "familiar":
                role_pairs = _role_pairs(rng, train_pool, count, core=core)
            elif test_color_policy == "exchange":
                role_pairs = _role_pairs(rng, train_pool, count, style="exchange", core=core[::-1])
            else:
                role_pairs = [core, new_core]
                if count > 2:
                    reserved = next(c for c in color_order if c not in core + new_core)
                    available = [c for c in color_order if c != reserved]
                    role_pairs += _role_pairs(rng, available, count - 2)

            for _ in range(80):
                height, width = _shape(rng, count, forced_style)
                centers = _layout(rng, height, width, count, forced_style)
                layout_key = (height, width, centers)
                if layout_key in used_layouts:
                    continue
                crosses = tuple(sorted((r, c, *role_pairs[i])
                                       for i, (r, c) in enumerate(centers)))
                forbidden = {color for _, _, hub, spoke in crosses for color in (hub, spoke)}
                backgrounds = [c for c in range(10) if c not in forbidden]
                if not backgrounds:
                    continue
                if split == "train" and index == 1 and len(backgrounds) > 1:
                    alternatives = [c for c in backgrounds if c != first_background]
                    if alternatives:
                        backgrounds = alternatives
                background = int(rng.choice(backgrounds))
                inp, out = _paint(height, width, background, crosses)
                pair = {"input": inp, "output": out}
                signature = (height, width, background, crosses)
                if signature in SOURCE_SIGNATURES or _grid_key(inp) in used_inputs:
                    continue
                puzzle[split].append(pair)
                used_inputs.add(_grid_key(inp))
                used_layouts.add(layout_key)
                if split == "train" and index == 0:
                    first_background = background
                break
            else:
                raise ValueError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(inp, out):
    _require(inp.shape == out.shape, "input and output shapes differ")
    height, width = inp.shape
    values, frequencies = np.unique(inp, return_counts=True)
    _require(len(values) >= 3, "input needs background and two object colors")
    largest = int(np.argmax(frequencies))
    _require(np.count_nonzero(frequencies == frequencies[largest]) == 1,
             "background must be uniquely identifiable")
    background = int(values[largest])
    foreground = inp != background
    unseen = {(int(r), int(c)) for r, c in np.argwhere(foreground)}
    crosses = []
    while unseen:
        start = unseen.pop()
        component, queue = {start}, [start]
        while queue:
            row, col = queue.pop()
            for dr, dc in CARDINAL:
                neighbor = (row + dr, col + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        _require(len(component) == 5, "each input component must be a five-cell cross")
        hubs = [(r, c) for r, c in component
                if all((r + dr, c + dc) in component for dr, dc in CARDINAL)]
        _require(len(hubs) == 1, "cross has no unique center")
        row, col = hubs[0]
        hub = int(inp[row, col])
        spokes = {int(inp[row + dr, col + dc]) for dr, dc in CARDINAL}
        _require(len(spokes) == 1, "cross spokes must match")
        spoke = spokes.pop()
        _require(hub != spoke and hub != background and spoke != background,
                 "cross roles must use distinct colors")
        _require(2 <= row < height - 2 and 2 <= col < width - 2,
                 "the entire expansion must fit")
        crosses.append((row, col, hub, spoke))
    crosses.sort()
    _require(1 <= len(crosses) <= 8, "cross count is outside the supported range")
    additions = {}
    for row, col, hub, spoke in crosses:
        for dr, dc in CARDINAL:
            position = (row + 2 * dr, col + 2 * dc)
            _require(position not in additions and not foreground[position],
                     "expansions must not collide with each other or input")
            additions[position] = spoke
        for distance in (1, 2):
            for dr, dc in DIAGONAL:
                position = (row + distance * dr, col + distance * dc)
                _require(position not in additions and not foreground[position],
                         "expansions must not collide with each other or input")
                additions[position] = hub
    expected = inp.copy()
    for position, color in additions.items():
        expected[position] = color
    _require(np.array_equal(out, expected), "output differs from local cross expansion")
    _require(len(additions) == 12 * len(crosses), "incomplete additions")
    return {"background": background, "crosses": tuple(crosses),
            "layout": (height, width, tuple((r, c) for r, c, _, _ in crosses))}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train_n, test_n = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_n <= 10 and 1 <= test_n <= 4 and train_n > test_n,
             "invalid pair counts")
    inputs, pairs, layouts = set(), set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be two-dimensional arrays")
                _require(all(1 <= size <= 30 for size in grid.shape),
                         "grid dimensions outside ARC bounds")
                _require(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
                _require(0 <= int(np.min(grid)) and int(np.max(grid)) <= 9,
                         "grid color outside ARC bounds")
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            _require(input_key not in inputs and pair_key not in pairs,
                     "duplicate input or pair")
            inputs.add(input_key)
            pairs.add(pair_key)
            info = _inspect_pair(pair["input"], pair["output"])
            _require(info["layout"] not in layouts, "duplicate structural layout")
            layouts.add(info["layout"])
            h, w, _ = info["layout"]
            signature = (h, w, info["background"], info["crosses"])
            _require(signature not in SOURCE_SIGNATURES, "exact source pair copied")
            infos[split].append(info)
    _require(infos["train"][0]["crosses"], "training must show a cross")
    for info in infos["test"]:
        _require(info["layout"] not in {x["layout"] for x in infos["train"]},
                 "test repeats a training configuration")


def _fingerprint(puzzle):
    return tuple((split, tuple(_pair_key(p) for p in puzzle[split]))
                 for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_fingerprint(p) for p in puzzles}) == 42,
             "automatic seeds must produce distinct puzzles")
    _require(len({len(p["train"]) for p in puzzles}) >= 4 and
             len({len(p["test"]) for p in puzzles}) >= 3,
             "automatic pair counts lack diversity")
    counts, shapes, backgrounds, layouts, palette_types = set(), set(), set(), set(), set()
    train_mixed_counts = 0
    test_larger = 0
    edge_sides = set()
    arrangements = {"train": set(), "test": set()}
    shift_cases = {"source": 0, "reverse": 0, "unchanged": 0,
                   "mixed_train": 0, "mixed_test": 0}
    edge_classes = {"train": set(), "test": set()}
    color_shifts = set()
    varied_backgrounds = varied_counts = constant_counts = 0
    disjoint_roles = exchanged_roles = dense_boxes = 0
    small_by_split = {"train": 0, "test": 0}
    large_by_split = {"train": 0, "test": 0}
    long_rows = long_columns = 0
    minimum_density, maximum_density = 1.0, 0.0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_counts, test_counts = [], []
        puzzle_arrangements = {"train": set(), "test": set()}
        puzzle_backgrounds = set()
        train_colors = {
            color
            for pair in puzzle["train"]
            for _, _, hub, spoke in _inspect_pair(pair["input"], pair["output"])["crosses"]
            for color in (hub, spoke)
        }
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair["input"], pair["output"])
                cross = info["crosses"]
                count = len(cross)
                counts.add(count)
                (train_counts if split == "train" else test_counts).append(count)
                h, w, centers = info["layout"]
                shapes.add((h, w))
                layouts.add(info["layout"])
                backgrounds.add(info["background"])
                puzzle_backgrounds.add(info["background"])
                roles = {(hub, spoke) for _, _, hub, spoke in cross}
                palette_types.add("shared" if len(roles) == 1 else "mixed")
                train_mixed_counts += int(split == "train" and len(roles) > 1)
                ordered_roles = [(hub, spoke) for _, _, hub, spoke in cross]
                if count >= 2:
                    disjoint_roles += int(set(ordered_roles[0]).isdisjoint(ordered_roles[1]))
                    exchanged_roles += int(ordered_roles[0] == ordered_roles[1][::-1])
                dense_boxes += int(any(abs(r - rr) < 5 and abs(c - cc) < 5
                                       for i, (r, c) in enumerate(centers)
                                       for rr, cc in centers[i + 1:]))
                small_by_split[split] += int((h, w) == (5, 5))
                large_by_split[split] += int((h, w) == (30, 30))
                long_rows += int(h <= 9 and w >= 15)
                long_columns += int(w <= 9 and h >= 15)
                density = 5 * count / (h * w)
                minimum_density = min(minimum_density, density)
                maximum_density = max(maximum_density, density)
                touches_edge = any(r in (2, h - 3) or c in (2, w - 3)
                                   for r, c in centers)
                touches_corner = any(r in (2, h - 3) and c in (2, w - 3)
                                     for r, c in centers)
                edge_classes[split].add("corner" if touches_corner else
                                        "edge" if touches_edge else "none")
                if count == 2:
                    (r1, c1), (r2, c2) = centers
                    arrangement = ("row" if r1 == r2 else
                                   "column" if c1 == c2 else
                                   "down_right" if c1 < c2 else "down_left")
                    arrangements[split].add(arrangement)
                    puzzle_arrangements[split].add(arrangement)
                if split == "test":
                    test_colors = {v for a, b in ordered_roles for v in (a, b)}
                    color_shifts.add("familiar" if test_colors <= train_colors else
                                     "new" if test_colors.isdisjoint(train_colors) else
                                     "mixed")
                for row, col in centers:
                    if row == 2: edge_sides.add("top")
                    if row == h - 3: edge_sides.add("bottom")
                    if col == 2: edge_sides.add("left")
                    if col == w - 3: edge_sides.add("right")
        test_larger += int(max(test_counts) > max(train_counts))
        varied_backgrounds += int(len(puzzle_backgrounds) > 1)
        varied_counts += int(len(set(train_counts + test_counts)) > 1)
        constant_counts += int(len(set(train_counts + test_counts)) == 1)
        shift_cases["source"] += int("down_right" in puzzle_arrangements["train"]
                                     and "down_left" in puzzle_arrangements["test"])
        shift_cases["reverse"] += int("down_left" in puzzle_arrangements["train"]
                                      and "down_right" in puzzle_arrangements["test"])
        shift_cases["unchanged"] += int(bool(puzzle_arrangements["train"]
                                              & puzzle_arrangements["test"]))
        shift_cases["mixed_train"] += int(len(puzzle_arrangements["train"]) > 1)
        shift_cases["mixed_test"] += int(len(puzzle_arrangements["test"]) > 1)
    _require(counts == set(range(1, 9)), "cross-count range was not exercised")
    _require(len(shapes) >= 80 and (30, 30) in shapes and min(min(s) for s in shapes) == 5,
             "size range lacks coverage")
    _require(len(layouts) >= 200 and len(backgrounds) == 10,
             "layout or background range lacks coverage")
    _require(palette_types == {"shared", "mixed"} and train_mixed_counts >= 10,
             "shared and mixed palettes need training coverage")
    _require(edge_sides == {"top", "bottom", "left", "right"} and test_larger >= 3,
             "boundary or larger-test cases lack coverage")
    _require(varied_backgrounds == 42 and varied_counts >= 20 and constant_counts >= 1,
             "within-puzzle background or count policies lack coverage")
    _require(all(v == {"row", "column", "down_right", "down_left"}
                 for v in arrangements.values()), "layout arrangements lack coverage")
    _require(all(value >= 1 for value in shift_cases.values()),
             "source, reverse, unchanged, or mixed layout relationships lack coverage")
    _require(all(v == {"corner", "edge", "none"} for v in edge_classes.values()),
             "both splits must cover edge and margin cases")
    _require(color_shifts == {"new", "familiar", "mixed"} and
             disjoint_roles >= 5 and exchanged_roles >= 5,
             "test color exposure or local palette roles lack coverage")
    _require(dense_boxes >= 1 and sum(small_by_split.values()) >= 5 and
             all(v >= 1 for v in large_by_split.values()),
             "dense, 5x5, or 30x30 cases lack coverage")
    _require(long_rows >= 5 and long_columns >= 5 and
             minimum_density < .02 and maximum_density >= .19,
             "aspect ratio or density range lacks coverage")
    for request in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                    (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(73, *request))
    for request in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (4, 4)):
        try:
            generate_puzzle(73, *request)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair count accepted")
    _require(_fingerprint(generate_puzzle(123)) == _fingerprint(generate_puzzle(123)),
             "generation is not deterministic")
