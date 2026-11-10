"""Generate ARC puzzles about an ordered, unbounded square spiral."""

import numpy as np


_SOURCE_INPUT_ROWS = {
    ("00200002000", "00000000000", "00000000002", "02000000200",
     "00000000000", "00000300000", "00000000000", "00000000000",
     "00200000002", "20000000000", "00000200000"),
    ("00000000000", "00000000000", "00000000000", "00000000000",
     "00000300000", "00000000000", "00000000000", "00000000000",
     "00000000000", "00000000000", "00000000000"),
    ("00000020002", "02000000000", "00000000000", "00000000000",
     "00003000020", "02000000000", "00000000200", "00000000000",
     "20020000000", "00000000200", "00000200000"),
    ("00000000002", "02000020000", "00000000000", "02000000000",
     "00000000000", "00020030200", "00000000000", "00000000000",
     "20000000000", "02000000002", "00000020000"),
}

_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_TEST_MODES = (
    "first_step", "after_turn", "short", "mid", "end_segment",
    "late", "reentry_hit", "open_inert", "open_reentry",
)


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _inside(r, c, h, w):
    return 0 <= r < h and 0 <= c < w


def _spiral_steps(h, w, start):
    """List (row, col, segment, offset) until no later side can visit the grid."""
    r, c = start
    steps = []
    for segment in range(1, 4 * (max(h, w) + 1) + 1):
        length = 2 * ((segment + 1) // 2)
        dr, dc = _DIRECTIONS[(segment - 1) % 4]
        for offset in range(1, length + 1):
            r += dr
            c += dc
            steps.append((r, c, segment, offset))
    return steps


def _trace_output(grid, background, path_color, blocker):
    h, w = grid.shape
    location = np.argwhere(grid == path_color)
    if len(location) != 1:
        raise ValueError("input must have exactly one starter")
    start = tuple(int(x) for x in location[0])
    result = grid.copy()
    collision = None
    for index, (r, c, segment, offset) in enumerate(
        _spiral_steps(h, w, start), 1
    ):
        if not _inside(r, c, h, w):
            continue
        if int(grid[r, c]) == blocker:
            collision = (index, segment, offset, r, c)
            break
        if int(grid[r, c]) == background:
            result[r, c] = path_color
    return result, collision


def _path_features(h, w, start):
    steps = _spiral_steps(h, w, start)
    route = {(r, c) for r, c, _, _ in steps if _inside(r, c, h, w)}
    first_reentry = None
    outside = False
    inside_before = 0
    candidates = []
    for i, (r, c, segment, offset) in enumerate(steps):
        if not _inside(r, c, h, w):
            outside = True
            continue
        if outside and first_reentry is None:
            first_reentry = i
        candidates.append((i, segment, offset, inside_before))
        inside_before += 1
    return steps, route, first_reentry, candidates


def _has_reentry_before(grid, path_color, collision):
    h, w = grid.shape
    start = tuple(int(x) for x in np.argwhere(grid == path_color)[0])
    stop = collision[0] - 1 if collision is not None else None
    outside = False
    for i, (r, c, _, _) in enumerate(_spiral_steps(h, w, start)):
        if stop is not None and i >= stop:
            break
        if not _inside(r, c, h, w):
            outside = True
        elif outside:
            return True
    return False


def _has_future_background(grid, background, path_color, collision):
    if collision is None:
        return False
    h, w = grid.shape
    start = tuple(int(x) for x in np.argwhere(grid == path_color)[0])
    for i, (r, c, _, _) in enumerate(_spiral_steps(h, w, start), 1):
        if i > collision[0] and _inside(r, c, h, w):
            if int(grid[r, c]) == background:
                return True
    return False


def _new_grid(h, w, start, colors):
    grid = np.full((h, w), colors[0], dtype=np.int8)
    grid[start] = colors[1]
    return grid


def _random_start(rng, h, w, style=None):
    if style is None:
        style = int(rng.integers(0, 4))
    if style == 0:  # Anywhere, including center.
        return int(rng.integers(h)), int(rng.integers(w))
    if style == 1:  # One of four edges.
        side = int(rng.integers(4))
        if side == 0:
            return 0, int(rng.integers(w))
        if side == 1:
            return h - 1, int(rng.integers(w))
        if side == 2:
            return int(rng.integers(h)), 0
        return int(rng.integers(h)), w - 1
    if style == 2:  # A corner.
        return (int(rng.choice((0, h - 1))),
                int(rng.choice((0, w - 1))))
    # Near a side but usually not on it.
    side = int(rng.integers(4))
    if side < 2:
        return (int(rng.integers(1, min(4, h))) if side == 0
                else h - 1 - int(rng.integers(1, min(4, h)))), int(rng.integers(w))
    return int(rng.integers(h)), (int(rng.integers(1, min(4, w)))
                                  if side == 2 else w - 1 - int(rng.integers(1, min(4, w))))


def _choose_extras(rng, grid, steps, start, target_index, colors,
                   force_later=False, force_dense=False):
    """Scatter blockers without changing the selected first collision."""
    h, w = grid.shape
    target = None if target_index is None else steps[target_index][:2]
    forbidden = {start}
    if target_index is None:
        forbidden.update((r, c) for r, c, _, _ in steps if _inside(r, c, h, w))
    else:
        forbidden.update((r, c) for r, c, _, _ in steps[:target_index]
                         if _inside(r, c, h, w))
        forbidden.add(target)
    safe = [(r, c) for r in range(h) for c in range(w)
            if (r, c) not in forbidden]
    safe_set = set(safe)
    later = [] if target_index is None else [
        (r, c) for r, c, _, _ in steps[target_index + 1:]
        if (r, c) in safe_set
    ]
    # De-duplicate while retaining traversal order.
    later = list(dict.fromkeys(later))
    if force_later and not later:
        return False
    density = 2 if force_dense else int(rng.choice(3, p=(0.27, 0.33, 0.40)))
    low, high = ((0, 3), (4, 8), (9, 16))[density]
    if target_index is None:
        low = max(low, 2)
    if force_later:
        low = max(low, 1)
    desired = min(len(safe), int(rng.integers(low, high + 1)))
    selected = []
    if force_later:
        selected.append(later[int(rng.integers(len(later)))])
    if target is not None and rng.random() < 0.55:
        neighbors = [(target[0] + dr, target[1] + dc)
                     for dr, dc in _DIRECTIONS]
        neighbors = [p for p in neighbors if p in safe and p not in selected]
        if neighbors:
            selected.append(neighbors[int(rng.integers(len(neighbors)))])
    if target_index is None and safe:
        route = {(r, c) for r, c, _, _ in steps if _inside(r, c, h, w)}
        near = [p for p in safe if any((p[0] + dr, p[1] + dc) in route
                                     for dr, dc in _DIRECTIONS)]
        if near:
            selected.append(near[int(rng.integers(len(near)))])
    desired = max(desired, len(selected))
    remainder = [p for p in safe if p not in selected]
    if desired > len(selected):
        indices = rng.choice(len(remainder), size=desired - len(selected),
                             replace=False)
        selected.extend(remainder[int(i)] for i in np.atleast_1d(indices))
    for p in selected:
        grid[p] = colors[2]
    return True


def _candidate_matches(mode, i, segment, offset, prefix, reentry,
                       steps, h, w):
    if mode == "first_step":
        return i == 0
    if mode == "after_turn":
        return segment == 2 and offset == 1
    if mode == "short":
        return segment == 3 and 1 <= offset <= 4 and prefix >= 4
    if mode == "mid":
        return 4 <= segment <= 6 and prefix >= 5
    if mode == "end_segment":
        return 2 <= segment <= 10 and offset == 2 * ((segment + 1) // 2)
    if mode == "late":
        return 7 <= segment <= 14 and prefix >= 9
    if mode in ("reentry_hit", "core_stop"):
        return (reentry is not None and i > reentry and
                (5 <= segment <= 14 if mode == "core_stop" else
                 segment >= 3) and
                prefix >= (7 if mode == "core_stop" else 2))
    return False


def _make_pair(rng, colors, mode):
    for attempt in range(500):
        if mode == "anchor":
            h = int(rng.integers(10, 31))
            w = int(rng.integers(10, 31))
            start = (int(rng.integers(4, h - 2)),
                     int(rng.integers(2, w - 4)))
        else:
            h = int(rng.integers(7, 31))
            w = int(rng.integers(7, 31))
            start = _random_start(rng, h, w)
        steps, route, reentry, candidates = _path_features(h, w, start)
        grid = _new_grid(h, w, start, colors)
        if mode == "anchor":
            output, _ = _trace_output(grid, *colors)
            if np.count_nonzero(grid != output) >= 24:
                return {"input": grid, "output": output}
            continue
        if mode in ("open_inert", "open_reentry"):
            if mode == "open_reentry" and reentry is None:
                continue
            if not _choose_extras(rng, grid, steps, start, None, colors,
                                  force_dense=(rng.random() < 0.4)):
                continue
            output, collision = _trace_output(grid, *colors)
            if collision is None and np.count_nonzero(grid != output) >= 3:
                return {"input": grid, "output": output}
            continue
        choices = [i for i, segment, offset, prefix in candidates
                   if _candidate_matches(mode, i, segment, offset, prefix,
                                         reentry, steps, h, w)]
        if not choices:
            continue
        target_index = choices[int(rng.integers(len(choices)))]
        r, c = steps[target_index][:2]
        grid[r, c] = colors[2]
        later_exists = any(_inside(rr, cc, h, w)
                           for rr, cc, _, _ in steps[target_index + 1:])
        force_later = later_exists and (mode == "core_stop" or
                                         rng.random() < 0.72)
        if not _choose_extras(rng, grid, steps, start, target_index, colors,
                              force_later=force_later,
                              force_dense=(rng.random() < 0.38)):
            continue
        output, collision = _trace_output(grid, *colors)
        if collision is None or collision[0] != target_index + 1:
            continue
        if mode == "core_stop":
            if (not _has_reentry_before(grid, colors[1], collision) or
                    not _has_future_background(grid, colors[0],
                                               colors[1], collision)):
                continue
        return {"input": grid, "output": output}
    raise RuntimeError(f"could not construct {mode} pair")


def _valid_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed_key, num_train, num_test):
    if num_train is not None:
        num_train = _valid_count(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _valid_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed_key % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed_key // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid_key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _is_source_input(grid):
    return grid.shape == (11, 11) and tuple(
        "".join(str(int(x)) for x in row) for row in grid
    ) in _SOURCE_INPUT_ROWS


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_key = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_key)
    num_train, num_test = _resolve_counts(rng, seed_key, num_train, num_test)
    colors = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    used = set()

    def unique(mode):
        for _ in range(30):
            pair = _make_pair(rng, colors, mode)
            key = _grid_key(pair["input"])
            if key not in used and not _is_source_input(pair["input"]):
                used.add(key)
                return pair
        raise RuntimeError("could not make a distinct pair")

    train = [unique("anchor"), unique("core_stop")]
    other_modes = ("mid", "late", "open_inert", "open_reentry",
                   "end_segment", "short", "reentry_hit")
    offset = int(rng.integers(len(other_modes)))
    for i in range(num_train - 2):
        train.append(unique(other_modes[(offset + i) % len(other_modes)]))
    rng.shuffle(train)
    test = [unique(_TEST_MODES[(seed_key + 4 * i) % len(_TEST_MODES)])
            for i in range(num_test)]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_palette(puzzle):
    candidates = []
    for pair in puzzle["train"]:
        values, counts = np.unique(pair["input"], return_counts=True)
        if len(values) == 2 and min(counts) == 1:
            path = int(values[int(np.argmin(counts))])
            background = int(values[int(np.argmax(counts))])
            candidates.append((background, path))
    _check(candidates and len(set(candidates)) == 1,
           "training must uniquely identify background and starter")
    background, path = candidates[0]
    other = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            other.update(int(v) for v in np.unique(pair["input"])
                         if int(v) not in (background, path))
    _check(len(other) == 1, "one consistent blocker color is required")
    return background, path, other.pop()


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "invalid puzzle keys")
    _check(all(isinstance(puzzle[k], list) for k in ("train", "test")),
           "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    all_pairs = puzzle["train"] + puzzle["test"]
    for pair in all_pairs:
        _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
               "invalid pair")
        for name in ("input", "output"):
            grid = pair[name]
            _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                   "grids must be two-dimensional ndarrays")
            h, w = grid.shape
            _check(1 <= h <= 30 and 1 <= w <= 30 and
                   np.issubdtype(grid.dtype, np.integer) and
                   int(grid.min()) >= 0 and int(grid.max()) <= 9,
                   "invalid grid")
        _check(pair["input"].shape == pair["output"].shape,
               "output size changed")
        _check(not _is_source_input(pair["input"]), "source input copied")
    keys = [_grid_key(p["input"]) for p in all_pairs]
    pair_keys = [(_grid_key(p["input"]), _grid_key(p["output"]))
                 for p in all_pairs]
    _check(len(set(keys)) == len(keys) and len(set(pair_keys)) == len(pair_keys),
           "duplicate pair")
    background, path, blocker = _infer_palette(puzzle)
    info = []
    for pair in all_pairs:
        inp, out = pair["input"], pair["output"]
        _check(set(int(v) for v in np.unique(inp)) <=
               {background, path, blocker}, "unexpected input color")
        _check(set(int(v) for v in np.unique(out)) <=
               {background, path, blocker}, "unexpected output color")
        _check(np.count_nonzero(inp == path) == 1 and
               np.count_nonzero(inp == background) >= 1,
               "starter or background count is wrong")
        expected, collision = _trace_output(inp, background, path, blocker)
        _check(np.array_equal(out, expected), "incorrect spiral output")
        changed = inp != out
        _check(np.all(inp[changed] == background) and
               np.all(out[changed] == path), "invalid recoloring")
        _check(np.any(changed) or
               (collision is not None and collision[0] == 1),
               "unchanged pair lacks an immediate blocker")
        info.append(collision)
    train_info = info[:nt]
    anchors = []
    for pair, collision in zip(puzzle["train"], train_info):
        inp = pair["input"]
        if np.count_nonzero(inp == blocker) == 0 and collision is None:
            start = tuple(int(x) for x in np.argwhere(inp == path)[0])
            if (np.count_nonzero(inp != pair["output"]) >= 24 and
                    all(_inside(r, c, *inp.shape)
                        for r, c, segment, _ in _spiral_steps(*inp.shape, start)
                        if segment <= 6)):
                anchors.append(pair)
    _check(anchors, "training needs a clear six-segment spiral")
    _check(any(c is not None and c[1] >= 5 and
               _has_reentry_before(p["input"], path, c) and
               _has_future_background(p["input"], background, path, c)
               for p, c in zip(puzzle["train"], train_info)),
           "training needs a stopped spiral after off-grid re-entry")
    for pair, collision in zip(puzzle["test"], info[nt:]):
        # Test may be open or blocked at any depth; the training witnesses
        # establish the common ordered rule for both.
        _check(collision is None or collision[0] >= 1,
               "invalid test collision")


def _puzzles_equal(a, b):
    return all(len(a[k]) == len(b[k]) and all(
        np.array_equal(x[n], y[n]) and x[n].dtype == y[n].dtype
        for x, y in zip(a[k], b[k]) for n in ("input", "output"))
        for k in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(100, 142)]
    _check(len({len(p["train"]) for p in puzzles}) >= 4 and
           len({len(p["test"]) for p in puzzles}) >= 3,
           "automatic counts lack diversity")
    shapes, positions, palettes, signatures = set(), set(), set(), set()
    test_segments, test_offsets, test_blockers = set(), set(), set()
    test_open = test_reentry = test_first = test_after_turn = 0
    dense = inert = later = adjacent = 0
    side_starts = set()
    corners = near_inert = final_offset = delayed_exit = 0
    dense_late = dense_reentry = mixed_test_puzzles = varied_test_puzzles = 0
    varied_shapes = varied_blockers = 0
    anchor_positions = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        colors = _infer_palette(puzzle)
        palettes.add(colors)
        signatures.add(tuple(_grid_key(p["input"])
                             for k in ("train", "test") for p in puzzle[k]))
        local_shapes, local_blockers, local_test_stops = set(), set(), set()
        anchor_positions.add(next(i for i, p in enumerate(puzzle["train"])
                                  if np.count_nonzero(p["input"] == colors[2]) == 0))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                h, w = inp.shape
                shapes.add((h, w))
                local_shapes.add((h, w))
                start = tuple(int(x) for x in np.argwhere(inp == colors[1])[0])
                positions.add(start)
                if start[0] == 0: side_starts.add("top")
                if start[0] == h - 1: side_starts.add("bottom")
                if start[1] == 0: side_starts.add("left")
                if start[1] == w - 1: side_starts.add("right")
                if start[0] in (0, h - 1) and start[1] in (0, w - 1):
                    corners += 1
                steps, route, _, _ = _path_features(h, w, start)
                blockers = {(int(r), int(c)) for r, c in np.argwhere(inp == colors[2])}
                local_blockers.add(len(blockers))
                if len(blockers) >= 9: dense += 1
                if any(any((r + dr, c + dc) in blockers for dr, dc in _DIRECTIONS)
                       for r, c in blockers): adjacent += 1
                output, collision = _trace_output(inp, *colors)
                _check(np.array_equal(output, pair["output"]), "wrong output")
                if blockers and collision is None: inert += 1
                if collision is None and any(
                    (r + dr, c + dc) in route
                    for r, c in blockers for dr, dc in _DIRECTIONS
                ):
                    near_inert += 1
                if collision is not None:
                    later_route = {(r, c) for r, c, _, _
                                   in steps[collision[0]:] if _inside(r, c, h, w)}
                    if blockers & later_route: later += 1
                if split == "test":
                    test_blockers.add(len(blockers))
                    local_test_stops.add(None if collision is None else collision[1])
                    if collision is None: test_open += 1
                    else:
                        test_segments.add(collision[1])
                        test_offsets.add(collision[2])
                        if collision[0] == 1: test_first += 1
                        if collision[1:3] == (2, 1): test_after_turn += 1
                        if collision[2] == 2 * ((collision[1] + 1) // 2):
                            final_offset += 1
                        if len(blockers) >= 9 and collision[1] >= 7:
                            dense_late += 1
                    if _has_reentry_before(inp, colors[1], collision):
                        test_reentry += 1
                        if collision is not None and len(blockers) >= 9:
                            dense_reentry += 1
                    first_exit = next((i for i, (r, c, _, _) in enumerate(steps)
                                       if not _inside(r, c, h, w)), len(steps))
                    if sum(_inside(r, c, h, w)
                           for r, c, _, _ in steps[:first_exit]) >= 4:
                        delayed_exit += 1
        if len(local_shapes) >= 2: varied_shapes += 1
        if len(local_blockers) >= 2: varied_blockers += 1
        if None in local_test_stops and len(local_test_stops) > 1:
            mixed_test_puzzles += 1
        if len(puzzle["test"]) > 1 and len(local_test_stops) > 1:
            varied_test_puzzles += 1
    _check(len(signatures) >= 40 and len(palettes) >= 18 and
           len(shapes) >= 45 and len(positions) >= 35,
           "structural or palette diversity is too narrow")
    _check(side_starts == {"top", "bottom", "left", "right"},
           "starter does not reach every side")
    _check(corners >= 10 and {0, 1} <= anchor_positions and
           len(anchor_positions) >= 4 and
           varied_shapes == 42 and varied_blockers == 42,
           "within-puzzle layout diversity is too narrow")
    _check(len(test_segments) >= 7 and len(test_offsets) >= 5 and
           test_open >= 3 and test_reentry >= 3 and
           test_first >= 2 and test_after_turn >= 2 and
           final_offset >= 3 and delayed_exit >= 5 and
           mixed_test_puzzles >= 8 and varied_test_puzzles >= 20,
           "test regimes lack coverage")
    _check(dense >= 20 and inert >= 5 and later >= 15 and adjacent >= 10 and
           max(test_blockers) >= 9 and near_inert >= 5 and
           dense_late >= 4 and dense_reentry >= 4,
           "blocker placement lacks diversity")
    _check(_puzzles_equal(generate_puzzle(777, 6, 3),
                          generate_puzzle(777, 6, 3)), "nondeterministic")
    _check(not _puzzles_equal(generate_puzzle(0), generate_puzzle(1)),
           "different seeds match")
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(800 + i, a, b)
        _check((len(p["train"]), len(p["test"])) == (a, b),
               "explicit counts ignored")
    for i, (a, b) in enumerate(((None, 1), (None, 4), (2, None),
                                 (10, None))):
        p = generate_puzzle(900 + i, a, b)
        _check((a is None or len(p["train"]) == a) and
               (b is None or len(p["test"]) == b),
               "partial counts ignored")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(999, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
