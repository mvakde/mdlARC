"""Compositional generator for clockwise corner signals and diagonal rays."""

import numpy as np


CORNERS = ((-1, -1), (-1, 2), (2, 2), (2, -1))
DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))

# Source grids are retained only for exact-copy and source-regression checks.
SOURCE_INPUTS = (
    ("000000600700", "000000022000", "800700022000", "022000800300", "022000000000", "600300000000", "000000600700", "000000022000", "000000022000", "000000300800", "000000000000", "000000000000"),
    ("000000800600", "000000022000", "000000022000", "000000300700", "000000006007", "000000000220", "000000000220", "080030003008", "002200000000", "002200000000", "070060000000", "000000000000"),
    ("30070000", "02200000", "02200000", "60080000", "00060070", "00002200", "00002200", "00080030"),
    ("000000000", "000000000", "003006000", "000220000", "000220000", "008007000", "000000000", "000000000", "000000000"),
    ("00300800", "00022000", "00022000", "00700600", "70060000", "02200000", "02200000", "80030000"),
)
SOURCE_OUTPUTS = (
    ("300008800600", "030080022000", "603800022000", "022300300700", "022033000076", "300733000067", "000370300600", "003007022000", "030000722000", "300000870700", "000008007070", "000080000707"),
    ("000030300808", "000003022080", "000000322800", "000000738600", "000007083066", "000070800226", "700708000220", "077080008007", "002200080000", "002200800000", "060038000000", "600083000000"),
    ("60030000", "02200000", "02200000", "80870006", "00087060", "00002200", "00002200", "00030070"),
    ("800000030", "080000300", "008003000", "000220000", "000220000", "007006000", "070000600", "700000060", "000000006"),
    ("00700307", "00022070", "00022700", "00607800", "86070080", "62200008", "02200000", "30060000"),
)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _from_rows(rows):
    return np.asarray([[int(cell) for cell in row] for row in rows], dtype=np.int8)


def _key(grid):
    return (grid.shape, grid.tobytes())


_SOURCE_KEYS = frozenset(_key(_from_rows(rows)) for rows in SOURCE_INPUTS)


def _footprint(anchor):
    r, c = anchor
    return {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1),
            *((r + dr, c + dc) for dr, dc in CORNERS)}


def _parse(grid):
    """Reconstruct local objects from cells, without generation metadata."""
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    biggest = max(counts)
    _check(sum(counts == biggest) == 1, "Background is not unique")
    bg = int(colors[int(np.argmax(counts))])
    candidates = {}
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(grid[r, c])
            if color != bg and np.all(grid[r:r + 2, c:c + 2] == color):
                candidates.setdefault(color, []).append((r, c))
    _check(len(candidates) == 1, "There must be one identifiable core color")
    core, anchors = next(iter(candidates.items()))
    anchors = tuple(sorted(anchors))
    _check(1 <= len(anchors) <= 4, "Object count is outside the family")
    cells = set()
    signal_cells = set()
    signals = {}
    for anchor in anchors:
        r, c = anchor
        _check(1 <= r <= h - 3 and 1 <= c <= w - 3,
               "Core must have four surrounding corners")
        block = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
        corners = tuple((r + dr, c + dc) for dr, dc in CORNERS)
        _check(not cells.intersection(block) and not signal_cells.intersection(corners),
               "Objects overlap")
        cells.update(block)
        signal_cells.update(corners)
        values = tuple(int(grid[p]) for p in corners)
        _check(len(set(values)) == 4 and bg not in values and core not in values,
               "Each core needs four distinct signals")
        signals[anchor] = values
    _check({tuple(p) for p in np.argwhere(grid == core)} == cells,
           "Core cells do not form separate 2 by 2 blocks")
    actual_signals = {tuple(p) for p in np.argwhere((grid != bg) & (grid != core))}
    _check(actual_signals == signal_cells, "Non-background cells must be object features")
    return {"shape": (h, w), "background": bg, "core": core,
            "anchors": anchors, "cores": cells, "signals": signals,
            "corners": signal_cells}


def _ray(info, anchor, corner):
    h, w = info["shape"]
    r = anchor[0] + CORNERS[corner][0]
    c = anchor[1] + CORNERS[corner][1]
    dr, dc = DIRECTIONS[corner]
    while 0 <= r < h and 0 <= c < w:
        if (r, c) in info["cores"]:
            break
        yield r, c
        r += dr
        c += dc


def _blocked(info):
    h, w = info["shape"]
    for anchor in info["anchors"]:
        for corner in range(4):
            ray = list(_ray(info, anchor, corner))
            if ray:
                r, c = ray[-1]
                dr, dc = DIRECTIONS[corner]
                if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) in info["cores"]:
                    yield anchor, corner, (r + dr, c + dc)


def _ambiguous_near_miss(anchors):
    """Exclude the unresolved source-train-2 relative placement and symmetries."""
    for a in anchors:
        for b in anchors:
            if a != b and sorted((abs(a[0] - b[0]), abs(a[1] - b[1]))) == [5, 7]:
                return True
    return False


def _solve(grid):
    info = _parse(grid)
    result = np.full(grid.shape, info["background"], dtype=np.int8)
    for r, c in info["cores"]:
        result[r, c] = info["core"]
    assigned = {}
    for anchor in info["anchors"]:
        original = info["signals"][anchor]
        shifted = (original[3], original[0], original[1], original[2])
        for corner, color in enumerate(shifted):
            for cell in _ray(info, anchor, corner):
                previous = assigned.get(cell)
                _check(previous is None or previous == color,
                       "Incompatible rays meet at one cell")
                assigned[cell] = color
    for cell, color in assigned.items():
        result[cell] = color
    return result


def _palette(rng):
    sample = rng.choice(10, size=10, replace=False).tolist()
    return int(sample[0]), int(sample[1]), tuple(int(x) for x in sample[2:])


def _valid_anchors(h, w):
    return [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]


def _choose_layout(rng, h, w, count, style):
    candidates = _valid_anchors(h, w)
    first = candidates[int(rng.integers(len(candidates)))]
    anchors = [first]
    if style in ("block", "mixed", "near"):
        options = []
        if style == "near":
            offsets = [(sr * a, sc * b) for a, b in ((4, 2), (2, 4))
                       for sr in (-1, 1) for sc in (-1, 1)]
        else:
            offsets = [(sr * a, sc * b)
                       for distance in range(4, 10)
                       for a, b in ((distance, distance - 1),
                                    (distance, distance),
                                    (distance - 1, distance))
                       for sr in (-1, 1) for sc in (-1, 1)]
        for dr, dc in offsets:
            second = (first[0] + dr, first[1] + dc)
            if second in candidates and not _footprint(first).intersection(_footprint(second)):
                options.append(second)
        if not options:
            raise AssertionError("No second anchor fits")
        anchors.append(options[int(rng.integers(len(options)))])
    order = rng.permutation(len(candidates))
    for i in order:
        if len(anchors) == count:
            break
        candidate = candidates[int(i)]
        if all(not _footprint(candidate).intersection(_footprint(old)) for old in anchors):
            anchors.append(candidate)
    _check(len(anchors) == count, "Could not place all objects")
    return tuple(sorted(anchors))


def _compose(rng, h, w, anchors, palette, varied_quartets):
    bg, core, available = palette
    result = np.full((h, w), bg, dtype=np.int8)
    shared = tuple(int(x) for x in rng.choice(available, size=4, replace=False))
    for anchor in anchors:
        r, c = anchor
        result[r:r + 2, c:c + 2] = core
        quartet = (tuple(int(x) for x in rng.choice(available, size=4, replace=False))
                   if varied_quartets else shared)
        colors = rng.permutation(quartet)
        for (dr, dc), color in zip(CORNERS, colors):
            result[r + dr, c + dc] = int(color)
    return result


def _blocking_evidence(info, output):
    h, w = info["shape"]
    for anchor, corner, hit in _blocked(info):
        original = info["signals"][anchor]
        color = (original[3], original[0], original[1], original[2])[corner]
        dr, dc = DIRECTIONS[corner]
        r, c = hit
        while 0 <= r < h and 0 <= c < w and (r, c) in info["cores"]:
            r += dr
            c += dc
        if 0 <= r < h and 0 <= c < w and int(output[r, c]) != color:
            return True
    return False


def _generate_pair(rng, palette, color_mode, style):
    if style == "single":
        count = 1
    elif style in ("block", "near"):
        count = 2
    elif style == "mixed":
        count = int(rng.integers(3, 5))
    else:
        count = int(rng.choice((1, 2, 2, 3, 3, 4)))
    min_size = 8 if count <= 2 else (10 if count == 3 else 12)
    for _ in range(1600):
        h = int(rng.integers(min_size, 19))
        w = int(rng.integers(min_size, 19))
        if rng.random() < 0.22:
            w = h
        if style == "near" and rng.random() < 0.20:
            h = w = 8
        if count == 1 and h * w > 145:
            continue
        if count == 2 and h * w > 260:
            continue
        if style == "single":
            # An interior isolated example makes all four extensions visible.
            h = min(h, 14)
            w = min(w, 14)
        try:
            anchors = _choose_layout(rng, h, w, count, style)
            if _ambiguous_near_miss(anchors):
                continue
            pair_palette = palette if color_mode == "constant" else _palette(rng)
            inp = _compose(rng, h, w, anchors, pair_palette,
                           color_mode == "variable" and count > 1 and rng.random() < 0.32)
            info = _parse(inp)
            if style == "single" and any(len(list(_ray(info, anchors[0], j))) < 2 for j in range(4)):
                continue
            if style in ("block", "mixed") and not any(_blocked(info)):
                continue
            if style == "near" and any(_blocked(info)):
                continue
            out = _solve(inp)
            if style in ("block", "mixed") and not _blocking_evidence(info, out):
                continue
            if _key(inp) in _SOURCE_KEYS or np.array_equal(inp, out):
                continue
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise RuntimeError("Could not construct a valid pair")


def _count(name, value, low, high):
    if value is not None and (isinstance(value, (bool, np.bool_))
                              or not isinstance(value, (int, np.integer))
                              or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer from {low} through {high}")


def _counts(seed, rng, num_train, num_test):
    _count("num_train", num_train, 2, 10)
    _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_n, test_n = _counts(int(seed), rng, num_train, num_test)
    palette = _palette(rng)
    color_mode = "constant" if rng.random() < 0.45 else "variable"
    if color_mode == "constant":
        palette = (palette[0], palette[1], palette[2][:4])
    # The clockwise rule and opaque-core behavior are puzzle-wide. Color roles
    # are read from each input; constant and changing palettes both teach that.
    train_styles = ["single", "block"]
    train_styles += [str(rng.choice(("single", "block", "mixed", "random", "random")))
                     for _ in range(train_n - 2)]
    rng.shuffle(train_styles)
    test_styles = [str(rng.choice(("near", "near", "random", "block", "mixed")))
                   for _ in range(test_n)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, styles in (("train", train_styles), ("test", test_styles)):
        for style in styles:
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(80):
                pair = _generate_pair(pair_rng, palette, color_mode, style)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not generate unique pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, name):
    _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
           f"{name} must be a two-dimensional NumPy array")
    h, w = grid.shape
    _check(1 <= h <= 30 and 1 <= w <= 30, f"{name} dimensions are invalid")
    _check(np.issubdtype(grid.dtype, np.integer), f"{name} must have integer cells")
    _check(0 <= int(np.min(grid)) and int(np.max(grid)) <= 9,
           f"{name} colors are invalid")


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "Puzzle must have train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "Puzzle splits must be lists")
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, "Illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    train_single = False
    train_multi = False
    train_block = False
    directions = set()
    test_block = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "Each pair needs input and output")
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp, "input")
            _validate_grid(out, "output")
            _check(inp.shape == out.shape and not np.array_equal(inp, out),
                   "Output must change cells without changing the canvas")
            key = _key(inp)
            pairkey = (key, _key(out))
            _check(key not in seen_inputs and pairkey not in seen_pairs,
                   "Duplicate input or pair")
            seen_inputs.add(key)
            seen_pairs.add(pairkey)
            _check(key not in _SOURCE_KEYS, "Exact source input copied")
            info = _parse(inp)
            _check(not _ambiguous_near_miss(info["anchors"]),
                   "Unresolved source near-miss is excluded")
            _check(np.array_equal(out, _solve(inp)), "Incorrect clockwise ray output")
            if split == "train":
                train_single |= len(info["anchors"]) == 1
                train_multi |= len(info["anchors"]) >= 2
                train_block |= _blocking_evidence(info, out)
                for anchor in info["anchors"]:
                    old = info["signals"][anchor]
                    clockwise = (old[3], old[0], old[1], old[2])
                    counter = (old[1], old[2], old[3], old[0])
                    observed = tuple(int(out[anchor[0] + dr, anchor[1] + dc])
                                     for dr, dc in CORNERS)
                    _check(observed == clockwise and observed != counter,
                           "Training must distinguish clockwise motion")
                    for j in range(4):
                        if len(list(_ray(info, anchor, j))) >= 2:
                            directions.add(j)
            else:
                test_block |= any(_blocked(info))
    _check(train_single and train_multi and train_block,
           "Training needs an isolated core, multiple cores, and visible blocking")
    _check(directions == {0, 1, 2, 3}, "Training must show all four extended directions")
    # A varying test palette carries no hidden mapping: colors are local
    # identities, while the demonstrated corner motion and stopping stay fixed.
    _check(not test_block or train_block, "Blocked tests require blocking training")


def _same(a, b):
    return (all(len(a[s]) == len(b[s]) for s in ("train", "test"))
            and all(np.array_equal(x[k], y[k]) for s in ("train", "test")
                    for x, y in zip(a[s], b[s]) for k in ("input", "output")))


def self_test() -> None:
    # Direct source regression: four pairs fit the core-only extension, while
    # train pair 2 differs exactly at the two recorded cells of the 7 ray.
    for i, (ir, er) in enumerate(zip(SOURCE_INPUTS, SOURCE_OUTPUTS)):
        inp, observed = _from_rows(ir), _from_rows(er)
        predicted = _solve(inp)
        differences = {tuple(p) for p in np.argwhere(predicted != observed)}
        if i == 1:
            _check(differences == {(8, 1), (9, 0)}, "Source exception changed")
            _check(_ambiguous_near_miss(_parse(inp)["anchors"]),
                   "Source exception must remain excluded")
        else:
            _check(not differences, "A source pair no longer fits the rule")
    _check(not _ambiguous_near_miss(_parse(_from_rows(SOURCE_INPUTS[4]))["anchors"]),
           "The original compact test layout must remain permitted")
    for args in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(710 + (args[0] or 0) * 10 + (args[1] or 0), *args)
        validate_generated_puzzle(puzzle)
        if args[0] is not None:
            _check(len(puzzle["train"]) == args[0], "Explicit train count failed")
        if args[1] is not None:
            _check(len(puzzle["test"]) == args[1], "Explicit test count failed")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(800, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count request accepted")
    a = generate_puzzle(12345, 5, 2)
    _check(_same(a, generate_puzzle(12345, 5, 2)), "Generator is not deterministic")
    shapes, layouts, object_counts = set(), set(), set()
    train_counts, test_counts = set(), set()
    palettes, test_layouts = set(), set()
    rectangular = large = four = variable = near = blocked_test = 0
    constant = compact_near = multiple_blocks = combined = 0
    varied_test_layouts = multi_test_puzzles = 0
    blocker_offsets = set()
    blocker_directions = set()
    blocked_lengths = set()
    training_block_slots = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_palettes = set()
        local_test_layouts = set()
        for split in ("train", "test"):
            for pair_index, pair in enumerate(puzzle[split]):
                inp = pair["input"]
                info = _parse(inp)
                shapes.add(inp.shape)
                layouts.add((inp.shape, info["anchors"]))
                object_counts.add(len(info["anchors"]))
                rectangular += inp.shape[0] != inp.shape[1]
                large += max(inp.shape) >= 15
                four += len(info["anchors"]) == 4
                local_palettes.add((info["background"], info["core"],
                                    tuple(sorted({c for vals in info["signals"].values() for c in vals}))))
                if split == "test":
                    test_layouts.add((inp.shape, info["anchors"]))
                    local_test_layouts.add((inp.shape, info["anchors"]))
                blockers = list(_blocked(info))
                multiple_blocks += len(blockers) >= 2
                combined += len(info["anchors"]) == 4 and bool(blockers)
                if split == "train" and blockers:
                    training_block_slots.add(pair_index)
                for anchor, direction, hit in blockers:
                    blocker_directions.add(direction)
                    blocked_lengths.add(len(list(_ray(info, anchor, direction))))
                    for other in info["anchors"]:
                        if other != anchor and hit in _footprint(other):
                            blocker_offsets.add((other[0] - anchor[0],
                                                 other[1] - anchor[1]))
                            break
                if split == "test":
                    blocked_test += bool(blockers)
                    if len(info["anchors"]) == 2:
                        a, b = info["anchors"]
                        is_near = sorted((abs(a[0]-b[0]), abs(a[1]-b[1]))) == [2, 4]
                        near += is_near
                        compact_near += is_near and inp.shape == (8, 8)
        palettes.update(local_palettes)
        variable += len(local_palettes) > 1
        constant += len(local_palettes) == 1
        if len(puzzle["test"]) > 1:
            multi_test_puzzles += 1
            varied_test_layouts += len(local_test_layouts) > 1
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "Counts lack diversity")
    _check(len(shapes) >= 35 and len(layouts) >= 100 and len(test_layouts) >= 30,
           "Grid geometry lacks diversity")
    _check(object_counts == {1, 2, 3, 4}, "Object counts lack diversity")
    _check(rectangular >= 80 and large >= 35 and four >= 5,
           "Rectangles, large grids, or four-core grids are too rare")
    _check(variable >= 10 and len(palettes) >= 60, "Palette variation is too narrow")
    _check(near >= 8 and blocked_test >= 8, "Test geometry lacks near misses or blockers")
    _check(constant >= 5 and compact_near >= 1,
           "Constant palettes or original-size near misses are missing")
    _check(multi_test_puzzles >= 20 and varied_test_layouts == multi_test_puzzles,
           "Multiple test pairs repeat one geometry")
    _check(len(blocker_offsets) >= 20 and blocker_directions == {0, 1, 2, 3}
           and len(blocked_lengths) >= 4 and multiple_blocks >= 15,
           "Obstacle direction, distance, or multiplicity lacks diversity")
    _check(len(training_block_slots) >= 4 and combined >= 5,
           "Training order or combined four-core cases lack diversity")
