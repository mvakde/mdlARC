"""Generate sparse marker-to-anchor projection puzzles."""

import numpy as np


_DIRECTIONS = ("up", "down", "left", "right")
_LANES = tuple((direction, lane) for direction in _DIRECTIONS for lane in (0, 1))
_SOURCE_ROWS = (
    ("0700008000", "0000000000", "0007033008", "0000033000",
     "0000000000", "0800000000", "0000000080", "0008000000",
     "0700000000", "0000070007"),
    ("1000000000", "0000000060", "0000000000", "0033000010",
     "0033000000", "0000000000", "0000000600", "0000006000",
     "0060000000", "0000100000"),
    ("0000000000", "0000100000", "0100000000", "0000000000",
     "0000000000", "0000000001", "0000330000", "0000330000",
     "0000000000", "0100000000"),
    ("0002000000", "2000000000", "0000000200", "6000000000",
     "0000000600", "0000000000", "6033000000", "0033000000",
     "0000000200", "0006060000"),
)
_SOURCE_INPUTS = tuple(
    np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)
    for rows in _SOURCE_ROWS
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _key(grid):
    return grid.shape, tuple(int(value) for value in grid.flat)


def _is_source(grid):
    return any(grid.shape == old.shape and np.array_equal(grid, old)
               for old in _SOURCE_INPUTS)


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    _require(int(counts[index]) > grid.size // 2,
             "background must be the unique majority color")
    return int(colors[index])


def _anchor(grid, background):
    matches = []
    height, width = grid.shape
    for row in range(height - 1):
        for column in range(width - 1):
            color = int(grid[row, column])
            if color != background and np.all(grid[row:row + 2, column:column + 2] == color):
                matches.append((row, column, color))
    _require(len(matches) == 1, "input must have one solid 2 by 2 anchor")
    return matches[0]


def _alignment(row, column, anchor_row, anchor_column):
    if row in (anchor_row, anchor_row + 1):
        if column < anchor_column:
            return ("left", row - anchor_row,
                    tuple((row, c) for c in range(column + 1, anchor_column)))
        if column > anchor_column + 1:
            return ("right", row - anchor_row,
                    tuple((row, c) for c in range(anchor_column + 2, column)))
    if column in (anchor_column, anchor_column + 1):
        if row < anchor_row:
            return ("up", column - anchor_column,
                    tuple((r, column) for r in range(row + 1, anchor_row)))
        if row > anchor_row + 1:
            return ("down", column - anchor_column,
                    tuple((r, column) for r in range(anchor_row + 2, row)))
    return None


def _orthogonal_neighbors(coordinate):
    row, column = coordinate
    return ((row - 1, column), (row + 1, column),
            (row, column - 1), (row, column + 1))


def _analyze(grid):
    background = _background(grid)
    anchor_row, anchor_column, anchor_color = _anchor(grid, background)
    anchor_cells = {(anchor_row + dr, anchor_column + dc)
                    for dr in (0, 1) for dc in (0, 1)}
    _require(int(np.count_nonzero(grid == anchor_color)) == 4,
             "anchor color must occur only in the anchor")
    markers = [(int(r), int(c)) for r, c in np.argwhere(grid != background)
               if (int(r), int(c)) not in anchor_cells]
    _require(markers, "input needs markers")
    marker_cells = set(markers)
    for coordinate in markers:
        _require(not any(neighbor in anchor_cells for neighbor in _orthogonal_neighbors(coordinate)),
                 "a marker must not touch the anchor orthogonally")
        _require(not any(neighbor in marker_cells for neighbor in _orthogonal_neighbors(coordinate)),
                 "markers must remain separate one-cell objects")

    active, distractors, lanes = [], [], set()
    for row, column in markers:
        alignment = _alignment(row, column, anchor_row, anchor_column)
        if alignment is None:
            distractors.append((row, column))
            continue
        direction, lane, path = alignment
        _require(path, "active marker needs a visible background gap")
        _require((direction, lane) not in lanes, "two markers share an outward lane")
        _require(all(int(grid[point]) == background for point in path),
                 "active gap must be clear")
        lanes.add((direction, lane))
        active.append((row, column, direction, lane, int(grid[row, column]), path))
    _require(active, "at least one active marker is required")
    _require(distractors, "at least one preserved distractor is required")
    collinear = any(a[0] == b[0] or a[1] == b[1]
                    for i, a in enumerate(distractors) for b in distractors[i + 1:])
    return {
        "background": background,
        "anchor": (anchor_row, anchor_column, anchor_color),
        "active": active,
        "distractors": distractors,
        "collinear": collinear,
        "signal_colors": {int(grid[point]) for point in markers},
    }


def _apply_rule(grid):
    analysis = _analyze(grid)
    output = grid.copy()
    for _, _, _, _, color, path in analysis["active"]:
        for point in path:
            output[point] = color
    return output


def _check_grid(grid):
    _require(isinstance(grid, np.ndarray), "grid must be a NumPy array")
    _require(grid.ndim == 2, "grid must be two-dimensional")
    _require(all(1 <= side <= 30 for side in grid.shape), "grid size outside 1 to 30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid cells must be integers")
    _require(0 <= int(grid.min()) and int(grid.max()) <= 9, "grid color outside 0 to 9")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")

    analyses = {"train": [], "test": []}
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            _require(inp.shape == out.shape, "projection must retain dimensions")
            ikey, pkey = _key(inp), (_key(inp), _key(out))
            _require(ikey not in inputs and pkey not in pairs, "duplicate generated pair")
            inputs.add(ikey)
            pairs.add(pkey)
            _require(not _is_source(inp), "generated input copies the source")
            analysis = _analyze(inp)
            _require(len(analysis["distractors"]) >= 2, "need preserved distractors")
            _require(np.array_equal(out, _apply_rule(inp)), "incorrect marker projection")
            _require(not np.array_equal(inp, out), "projection must visibly change the input")
            analyses[split].append(analysis)

    train = analyses["train"]
    test = analyses["test"]
    _require(any(len(a["active"]) == 1 for a in train),
             "training needs a simple one-ray demonstration")
    _require(any(len(a["active"]) == 4
                 and {m[2] for m in a["active"]} == set(_DIRECTIONS)
                 and len({m[4] for m in a["active"]}) >= 2 for a in train),
             "training needs a multicolor four-direction demonstration")
    _require(any(a["collinear"] for a in train),
             "training must show false dot-to-dot alignment")
    if any(len({m[2] for m in a["active"]}) < len(a["active"]) for a in test):
        _require(any(len({m[2] for m in a["active"]}) < len(a["active"]) for a in train),
                 "parallel test lanes need a parallel training example")
    _require({m[2] for a in train for m in a["active"]} == set(_DIRECTIONS),
             "training must show all approach directions")
    train_colors = {m[4] for a in train for m in a["active"]}
    _require(len(train_colors) >= 2, "training must show marker-color inheritance")
    _require(all(len(a["active"]) <= 8 for a in train + test), "at most eight lanes")


def _validate_count(name, value, minimum, maximum):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _counts(rng, num_train, num_test):
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _choose_lanes(rng, count, four_directions=False, parallel=False):
    if four_directions:
        return tuple((direction, int(rng.integers(0, 2))) for direction in _DIRECTIONS)
    if parallel:
        direction = _DIRECTIONS[int(rng.integers(0, 4))]
        chosen = [(direction, 0), (direction, 1)]
        others = [lane for lane in _LANES if lane not in chosen]
        selected = rng.choice(len(others), size=count - 2, replace=False)
        chosen.extend(others[int(i)] for i in selected)
        rng.shuffle(chosen)
        return tuple(chosen)
    chosen = rng.choice(len(_LANES), size=count, replace=False)
    return tuple(_LANES[int(i)] for i in chosen)


def _shape(rng, mode, split, shared):
    if mode == "equal":
        if split == "test" or rng.random() < 0.55:
            return shared
        return (max(8, min(24, shared[0] + int(rng.integers(-3, 4)))),
                max(8, min(24, shared[1] + int(rng.integers(-3, 4)))))
    if mode == "smaller":
        low, high = (13, 22) if split == "train" else (8, 13)
    elif mode == "larger":
        low, high = (8, 15) if split == "train" else (16, 25)
    elif mode == "source10":
        if split == "test":
            return (10, 10)
        low, high = 9, 17
    else:
        low, high = 8, 25
    return int(rng.integers(low, high)), int(rng.integers(low, high))


def _anchor_candidates(height, width, lanes):
    counts = {direction: sum(d == direction for d, _ in lanes)
              for direction in _DIRECTIONS}
    candidates = []
    for row in range(height - 1):
        for column in range(width - 1):
            available = {
                "up": row - 1,
                "down": height - row - 3,
                "left": column - 1,
                "right": width - column - 3,
            }
            if all(available[d] >= counts[d] for d in _DIRECTIONS):
                candidates.append((row, column))
    return candidates


def _active_position(rng, direction, lane, index, anchor_row, anchor_column,
                     height, width):
    if direction == "up":
        return (index, anchor_column + lane)
    if direction == "down":
        return (anchor_row + 3 + index, anchor_column + lane)
    if direction == "left":
        return (anchor_row + lane, index)
    return (anchor_row + lane, anchor_column + 3 + index)


def _place_active(rng, grid, anchor_row, anchor_column, lanes, colors):
    height, width = grid.shape
    by_direction = {d: [(lane, color) for (dd, lane), color in zip(lanes, colors)
                         if dd == d] for d in _DIRECTIONS}
    placed = []
    for direction, entries in by_direction.items():
        if not entries:
            continue
        available_count = {
            "up": anchor_row - 1,
            "down": height - anchor_row - 3,
            "left": anchor_column - 1,
            "right": width - anchor_column - 3,
        }[direction]
        offsets = rng.choice(available_count, size=len(entries), replace=False)
        for (lane, color), index in zip(entries, offsets):
            point = _active_position(rng, direction, lane, int(index),
                                     anchor_row, anchor_column, height, width)
            grid[point] = color
            placed.append(point)
    return placed


def _can_place(point, occupied):
    return not any(neighbor in occupied for neighbor in _orthogonal_neighbors(point))


def _place_distractors(rng, grid, anchor_row, anchor_column, occupied,
                       colors, target, collinear):
    background = int(grid[0, 0]) if (0, 0) not in occupied else _background(grid)
    height, width = grid.shape
    candidates = [(r, c) for r in range(height) for c in range(width)
                  if r not in (anchor_row, anchor_row + 1)
                  and c not in (anchor_column, anchor_column + 1)
                  and int(grid[r, c]) == background
                  and _can_place((r, c), occupied)]
    order = rng.permutation(len(candidates))
    candidates = [candidates[int(i)] for i in order]
    picked = []
    if collinear:
        for i, first in enumerate(candidates):
            for second in candidates[i + 1:]:
                if (first[0] == second[0] or first[1] == second[1]) and _can_place(second, {first}):
                    picked = [first, second]
                    break
            if picked:
                break
        if not picked:
            return False
    for point in picked:
        grid[point] = colors[int(rng.integers(0, len(colors)))]
        occupied.add(point)
    for point in candidates:
        if len(picked) >= target:
            break
        if point in occupied or not _can_place(point, occupied):
            continue
        grid[point] = colors[int(rng.integers(0, len(colors)))]
        occupied.add(point)
        picked.append(point)
    return len(picked) == target


def _make_pair(rng, shape, lanes, palette, background, anchor_color, collinear):
    height, width = shape
    candidates = _anchor_candidates(height, width, lanes)
    if not candidates:
        raise ValueError("shape cannot hold the requested lanes")
    edge = [p for p in candidates
            if min(p[0], p[1], height - p[0] - 2, width - p[1] - 2) <= 2]
    target = int(rng.integers(2, min(10, max(4, height * width // 15)) + 1))
    for desired in range(target, 1, -1):
        for _ in range(18):
            pool = edge if edge and rng.random() < 0.3 else candidates
            anchor_row, anchor_column = pool[int(rng.integers(0, len(pool)))]
            grid = np.full(shape, background, dtype=np.int8)
            grid[anchor_row:anchor_row + 2, anchor_column:anchor_column + 2] = anchor_color
            colors = [palette[i % len(palette)] for i in range(len(lanes))]
            rng.shuffle(colors)
            active = _place_active(rng, grid, anchor_row, anchor_column, lanes, colors)
            occupied = {(anchor_row + dr, anchor_column + dc)
                        for dr in (0, 1) for dc in (0, 1)}
            occupied.update(active)
            if _place_distractors(rng, grid, anchor_row, anchor_column,
                                  occupied, palette, desired, collinear):
                analysis = _analyze(grid)
                if len(analysis["active"]) == len(lanes):
                    return {"input": grid, "output": _apply_rule(grid)}
    raise ValueError("could not place the requested markers")


def _test_palette(rng, active_count, familiar, novel):
    maximum = min(3, active_count)
    category = int(rng.integers(0, 3 if active_count >= 2 else 2))
    if category == 0:
        count = int(rng.integers(1, min(maximum, len(familiar)) + 1))
        pool = familiar
        return tuple(int(x) for x in rng.choice(pool, size=count, replace=False))
    if category == 1:
        count = int(rng.integers(1, min(maximum, len(novel)) + 1))
        return tuple(int(x) for x in rng.choice(novel, size=count, replace=False))
    count = int(rng.integers(2, maximum + 1))
    colors = [int(rng.choice(familiar)), int(rng.choice(novel))]
    remaining = [x for x in familiar + novel if x not in colors]
    if count > 2:
        colors.extend(int(x) for x in rng.choice(remaining, size=count - 2, replace=False))
    rng.shuffle(colors)
    return tuple(colors)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    number_train, number_test = _counts(rng, num_train, num_test)
    permutation = [int(x) for x in rng.permutation(10)]
    training_pool_size = int(rng.integers(2, 5))
    training_pool = tuple(permutation[:training_pool_size])
    novel_pool = tuple(permutation[training_pool_size:training_pool_size + 3])
    structural = tuple(permutation[training_pool_size + 3:])
    background_varies = bool(rng.integers(0, 2))
    anchor_varies = bool(rng.integers(0, 2))
    mode = ("equal", "smaller", "larger", "source10", "mixed")[int(rng.integers(0, 5))]
    shared = (int(rng.integers(9, 19)), int(rng.integers(9, 19)))

    simple = {"lanes": _choose_lanes(rng, 1),
              "palette": (training_pool[int(rng.integers(0, training_pool_size))],)}
    four_colors = tuple(int(x) for x in rng.choice(training_pool, size=2, replace=False))
    four = {"lanes": _choose_lanes(rng, 4, four_directions=True),
            "palette": four_colors}
    train_specs = [simple, four]
    if number_train >= 3:
        count = int(rng.integers(5, 9))
        train_specs.append({"lanes": _choose_lanes(rng, count, parallel=True),
                            "palette": training_pool[:min(len(training_pool), count)]})
    while len(train_specs) < number_train:
        count = int(rng.integers(1, 9))
        if count >= 5:
            lanes = _choose_lanes(rng, count, parallel=True)
        else:
            lanes = _choose_lanes(rng, count)
        palette_count = int(rng.integers(1, min(len(training_pool), count) + 1))
        palette = tuple(int(x) for x in rng.choice(training_pool, size=palette_count,
                                                   replace=False))
        train_specs.append({"lanes": lanes, "palette": palette})
    rng.shuffle(train_specs)
    collinear_train = int(rng.integers(0, number_train))
    puzzle = {"train": [], "test": []}
    seen = set()
    pair_index = 0

    def roles(index):
        bg = structural[index % len(structural)] if background_varies else structural[0]
        if anchor_varies:
            options = [x for x in structural if x != bg]
            anchor_color = options[index % len(options)]
        else:
            anchor_color = structural[1] if structural[1] != bg else structural[0]
            if anchor_color == bg:
                anchor_color = structural[2]
        return bg, anchor_color

    def append_pair(split, lanes, palette, collinear):
        nonlocal pair_index
        background, anchor_color = roles(pair_index)
        for attempt in range(80):
            pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            pair_rng = np.random.default_rng(pair_seed)
            shape = _shape(pair_rng, mode, split, shared)
            try:
                pair = _make_pair(pair_rng, shape, lanes, palette,
                                  background, anchor_color, collinear)
            except (ValueError, AssertionError):
                continue
            key = _key(pair["input"])
            if key in seen or _is_source(pair["input"]):
                continue
            puzzle[split].append(pair)
            seen.add(key)
            pair_index += 1
            return
        raise ValueError("could not generate a unique valid pair")

    for index, spec in enumerate(train_specs):
        append_pair("train", spec["lanes"], spec["palette"],
                    index == collinear_train or bool(rng.integers(0, 2)))
    familiar = tuple(sorted({m[4] for pair in puzzle["train"]
                             for m in _analyze(pair["input"])["active"]}))
    for _ in range(number_test):
        max_count = 8 if number_train >= 3 else 4
        count = int(rng.integers(1, max_count + 1))
        lanes = _choose_lanes(rng, count, parallel=count >= 5)
        if number_train == 2:
            directions = rng.choice(len(_DIRECTIONS), size=count, replace=False)
            lanes = tuple((_DIRECTIONS[int(i)], int(rng.integers(0, 2)))
                          for i in directions)
        palette = _test_palette(rng, count, familiar, novel_pool)
        append_pair("test", lanes, palette, bool(rng.integers(0, 2)))
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, profiles, layouts = set(), set(), set(), set(), set()
    backgrounds, anchor_colors, palette_counts = set(), set(), set()
    parallel_train = parallel_test = collinear_train = collinear_test = 0
    diagonal_pairs = near_edge = source_size_test = 0
    signatures, first_train_rays, test_rays, test_directions = set(), set(), set(), set()
    test_regions = set()
    density = {"train": {"low": 0, "high": 0}, "test": {"low": 0, "high": 0}}
    test_palette_categories = set()
    background_changes = anchor_changes = 0
    size_relations = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_key(puzzle))
        first_train_rays.add(len(_analyze(puzzle["train"][0]["input"])["active"]))
        training_colors = {m[4] for pair in puzzle["train"]
                           for m in _analyze(pair["input"])["active"]}
        puzzle_backgrounds, puzzle_anchors = set(), set()
        split_areas = {}
        for split in ("train", "test"):
            split_areas[split] = []
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                a = _analyze(inp)
                _require(np.array_equal(out, _apply_rule(inp)), "projection mismatch")
                _require(not _is_source(inp), "source input copied")
                shapes.add(inp.shape)
                backgrounds.add(a["background"])
                anchor_colors.add(a["anchor"][2])
                puzzle_backgrounds.add(a["background"])
                puzzle_anchors.add(a["anchor"][2])
                palette_counts.add(len(a["signal_colors"]))
                density[split]["low"] += len(a["distractors"]) <= 3
                density[split]["high"] += len(a["distractors"]) >= 7
                r, c, _ = a["anchor"]
                layouts.add((inp.shape, r, c))
                profiles.add(tuple(sorted((m[2], m[3], len(m[5])) for m in a["active"])))
                split_areas[split].append(inp.size)
                if split == "test" and inp.shape == (10, 10):
                    source_size_test += 1
                if min(r, c, inp.shape[0] - r - 2, inp.shape[1] - c - 2) <= 2:
                    near_edge += 1
                markers = [(int(rr), int(cc)) for rr, cc in np.argwhere(inp != a["background"])
                           if not (r <= rr <= r + 1 and c <= cc <= c + 1)]
                if any(abs(x[0] - y[0]) == 1 and abs(x[1] - y[1]) == 1
                       for i, x in enumerate(markers) for y in markers[i + 1:]):
                    diagonal_pairs += 1
                if split == "train":
                    parallel_train += len({m[2] for m in a["active"]}) < len(a["active"])
                    collinear_train += a["collinear"]
                else:
                    parallel_test += len({m[2] for m in a["active"]}) < len(a["active"])
                    collinear_test += a["collinear"]
                    test_rays.add(len(a["active"]))
                    test_directions.add(tuple(sorted({m[2] for m in a["active"]})))
                    test_regions.add((int(r >= inp.shape[0] / 2),
                                      int(c >= inp.shape[1] / 2)))
                    colors = {m[4] for m in a["active"]}
                    category = ("mixed" if colors & training_colors and colors - training_colors
                                else "familiar" if colors <= training_colors else "new")
                    test_palette_categories.add(category)
        background_changes += len(puzzle_backgrounds) > 1
        anchor_changes += len(puzzle_anchors) > 1
        delta = np.median(split_areas["test"]) - np.median(split_areas["train"])
        size_relations.add("equal" if abs(delta) <= 12 else "larger" if delta > 0 else "smaller")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack range")
    _require(len(signatures) >= 40 and len(shapes) >= 30 and len(layouts) >= 80,
             "structural diversity too narrow")
    _require(len(profiles) >= 60 and len(backgrounds) >= 5 and len(anchor_colors) >= 5,
             "layout or color diversity too narrow")
    _require({1, 2, 3}.issubset(palette_counts), "signal counts lack range")
    _require(background_changes >= 10 and anchor_changes >= 10,
             "within-puzzle color-role variation too narrow")
    _require(len(first_train_rays) >= 5 and {1, 2, 4, 5, 8}.issubset(test_rays),
             "training order or test ray counts too repetitive")
    _require(len(test_directions) >= 10 and len(test_regions) == 4,
             "test directions or anchor regions too repetitive")
    _require(test_palette_categories == {"familiar", "mixed", "new"},
             "test hues lack familiar, mixed, or wholly new cases")
    _require(all(density[split]["low"] >= 10 and density[split]["high"] >= 10
                 for split in ("train", "test")), "split distractor density lacks range")
    _require(parallel_train >= 15 and parallel_test >= 5, "parallel lanes underrepresented")
    _require(collinear_train >= 42 and collinear_test >= 10,
             "false alignments underrepresented")
    _require(diagonal_pairs >= 20 and near_edge >= 20 and source_size_test >= 2,
             "diagonal, edge, or source-size cases underrepresented")
    _require(size_relations == {"equal", "larger", "smaller"}, "size shifts lack range")
    _require(_puzzle_key(generate_puzzle(314159, 8, 3))
             == _puzzle_key(generate_puzzle(314159, 8, 3)), "nondeterministic explicit puzzle")
    _require(_puzzle_key(generate_puzzle(271828))
             == _puzzle_key(generate_puzzle(271828)), "nondeterministic automatic puzzle")
    for index, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(50000 + index, nt, ns)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns),
                 "explicit counts not honored")
    for index, (nt, ns) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(51000 + index, nt, ns)
        _require((nt is None or len(puzzle["train"]) == nt)
                 and (ns is None or len(puzzle["test"]) == ns), "partial count not honored")
    for index, args in enumerate(((1, None), (11, None), (None, 0), (None, 5),
                                  (2, 2), (3, 4), (True, 1))):
        try:
            generate_puzzle(52000 + index, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
