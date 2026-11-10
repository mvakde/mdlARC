"""Four-arm midpoint ARC puzzle generator."""

import numpy as np


_SOURCE_INPUTS = {
    "8888888888888888/8888888888888888/8888188888118888/8888188888888888/8888888888881888/8118881188881888/8888888888888888/8888188888888888/8888188818888888/8888888818888888/8888888888888888/8888888888811888/8818888881188888/8818888888888888/8888888888888888/8888888888888888",
    "88888888/88818888/88818811/88888888/11888118/88888888/88818888/88818888",
    "8888888888888888/8888818888888888/8888818888188888/8888888888188888/8888888888888888/8811888118881188/8888888888888888/8888818888188888/8888818888181888/8888888888881888/8881888888888888/8881888881188811/1188811888888888/8888888888881888/8881888888881888/8881888888888888",
    "11888881188888881188/88888888888888188888/88881188881888188818/88818888881888888818/88818881888118881188/88888881888888888888/11888118188888188881/88888888188888188881/88818888888888888888/88818118881188881188/88888888888888888888/88818888181888188888/88818888181888188818/11888118888888888818/88888881188811888888/88818888888888811888/88818811881888188888/11888888881888188888/88811888118888888818/88888888888118888818",
}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _plain_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _centers(grid, background, foreground):
    """Find the complete, fixed-distance four-arm stencil."""
    h, w = grid.shape
    if h < 7 or w < 7:
        return set()
    bg = grid == background
    fg = grid == foreground
    mask = bg[3:-3, 3:-3].copy()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        mask &= bg[3 + dr:h - 3 + dr, 3 + dc:w - 3 + dc]
        for distance in (2, 3):
            mask &= fg[3 + distance * dr:h - 3 + distance * dr,
                       3 + distance * dc:w - 3 + distance * dc]
    return {(int(r) + 3, int(c) + 3) for r, c in np.argwhere(mask)}


def _axis_midpoints(grid, background, foreground, axis):
    h, w = grid.shape
    bg, fg = grid == background, grid == foreground
    result = set()
    if axis == "horizontal":
        for r in range(h):
            for c in range(3, w - 3):
                if (bg[r, c - 1:c + 2].all()
                    and fg[r, c - 3:c - 1].all()
                    and fg[r, c + 2:c + 4].all()):
                    result.add((r, c))
    else:
        for r in range(3, h - 3):
            for c in range(w):
                if (bg[r - 1:r + 2, c].all()
                    and fg[r - 3:r - 1, c].all()
                    and fg[r + 2:r + 4, c].all()):
                    result.add((r, c))
    return result


def _component_sizes(grid, foreground):
    cells = {(int(r), int(c)) for r, c in np.argwhere(grid == foreground)}
    sizes = []
    while cells:
        start = cells.pop()
        component = {start}
        todo = [start]
        while todo:
            r, c = todo.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in cells:
                    cells.remove(neighbor)
                    component.add(neighbor)
                    todo.append(neighbor)
        sizes.append(len(component))
    return sizes


def _domino_components(grid, foreground):
    """Whether each foreground component is exactly one separate domino."""
    return all(size == 2 for size in _component_sizes(grid, foreground))


def _bar_members(grid, foreground):
    h, w = grid.shape
    for rr, cc in np.argwhere(grid == foreground):
        r, c = int(rr), int(cc)
        if not any(0 <= x < h and 0 <= y < w and grid[x, y] == foreground
                   for x, y in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))):
            return False
    return True


def _stencil(r, c):
    background = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
    arms = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        arms.append({(r + 2 * dr, c + 2 * dc), (r + 3 * dr, c + 3 * dc)})
    return background, arms


def _valid_addition(grid, cells, protected, targets, bg, fg, clean):
    h, w = grid.shape
    if (any(r < 0 or c < 0 or r >= h or c >= w for r, c in cells)
        or not cells.isdisjoint(protected)):
        return None
    trial = grid.copy()
    for r, c in cells:
        trial[r, c] = fg
    if _centers(trial, bg, fg) != targets:
        return None
    if clean and not _domino_components(trial, fg):
        return None
    if not clean and max(_component_sizes(trial, fg), default=0) > 28:
        return None
    return trial


def _place_centers(grid, count, bg, fg, rng, clean, close):
    h, w = grid.shape
    possible = [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]
    chosen, protected, occupied = [], set(), set()
    for _ in range(count):
        feasible = []
        for r, c in possible:
            b, arms = _stencil(r, c)
            f = set().union(*arms)
            if b.isdisjoint(occupied) and f.isdisjoint(protected):
                if clean:
                    trial = grid.copy()
                    for rr, cc in f:
                        trial[rr, cc] = fg
                    if not _domino_components(trial, fg):
                        continue
                feasible.append((r, c))
        if not feasible:
            return None
        if chosen and close:
            nearby = [p for p in feasible if min(abs(p[0]-q[0]) + abs(p[1]-q[1])
                                                for q in chosen) <= 8]
            if nearby:
                feasible = nearby
        r, c = feasible[int(rng.integers(len(feasible)))]
        b, arms = _stencil(r, c)
        f = set().union(*arms)
        chosen.append((r, c))
        protected.update(b)
        occupied.update(f)
        for rr, cc in f:
            grid[rr, cc] = fg
    if _centers(grid, bg, fg) != set(chosen):
        return None
    return set(chosen), protected


def _near_miss_cells(r, c, kind, rng):
    _, arms = _stencil(r, c)
    if kind == "horizontal":
        return arms[2] | arms[3]
    if kind == "vertical":
        return arms[0] | arms[1]
    omit = int(rng.integers(4))
    cells = set().union(*(arm for i, arm in enumerate(arms) if i != omit))
    if kind == "omitted":
        return cells
    dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[omit]
    if kind == "gap":
        return cells | {(r + 3 * dr, c + 3 * dc), (r + 4 * dr, c + 4 * dc)}
    # One arm has the right length and gap but is displaced sideways.
    sr, sc = (0, 1) if dr else (1, 0)
    return cells | {(r + k * dr + sr, c + k * dc + sc) for k in (2, 3)}


def _add_near_miss(grid, kind, protected, targets, bg, fg, rng, clean):
    h, w = grid.shape
    positions = [(r, c) for r in range(4, h - 4) for c in range(4, w - 4)]
    for index in rng.permutation(len(positions))[:min(len(positions), 90)]:
        r, c = positions[int(index)]
        if (r, c) in targets:
            continue
        if any(grid[rr, cc] != bg for rr, cc in ((r, c), (r - 1, c),
                  (r + 1, c), (r, c - 1), (r, c + 1))):
            continue
        cells = _near_miss_cells(r, c, kind, rng)
        if not any(grid[rr, cc] == bg for rr, cc in cells):
            continue
        trial = _valid_addition(grid, cells, protected, targets, bg, fg, clean)
        if trial is None:
            continue
        if kind == "horizontal" and (r, c) not in _axis_midpoints(trial, bg, fg, kind):
            continue
        if kind == "vertical" and (r, c) not in _axis_midpoints(trial, bg, fg, kind):
            continue
        grid[:] = trial
        protected.update({(r, c), (r - 1, c), (r + 1, c),
                          (r, c - 1), (r, c + 1)})
        return True
    return False


def _add_bars(grid, goal, protected, targets, bg, fg, rng, clean, seek_overlap):
    h, w = grid.shape
    positions = [((r, c), (r, c + 1)) for r in range(h) for c in range(w - 1)]
    positions += [((r, c), (r + 1, c)) for r in range(h - 1) for c in range(w)]
    accepted = 0
    for _ in range(goal):
        order = rng.permutation(len(positions))
        if seek_overlap and not clean:
            order = sorted(order, key=lambda i: -sum(
                any(0 <= x < h and 0 <= y < w and grid[x, y] == fg
                    for x, y in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)))
                for r, c in positions[int(i)]))
        placed = False
        for raw_index in order[:min(len(order), 180)]:
            cells = set(positions[int(raw_index)])
            if all(grid[r, c] == fg for r, c in cells):
                continue
            trial = _valid_addition(grid, cells, protected, targets, bg, fg, clean)
            if trial is None:
                continue
            grid[:] = trial
            accepted += 1
            placed = True
            break
        if not placed:
            break
    return accepted


def _make_pair(rng, shape, count, colors, clean, density, decoys, close):
    h, w = shape
    bg, fg, marker = colors
    for _ in range(25):
        grid = np.full(shape, bg, dtype=np.int8)
        placed = _place_centers(grid, count, bg, fg, rng, clean, close)
        if placed is None:
            continue
        targets, protected = placed
        required = True
        for kind in decoys:
            if not _add_near_miss(grid, kind, protected, targets, bg, fg, rng, clean):
                required = False
                break
        if not required:
            continue
        max_extra = max(0, int((h * w * density - np.count_nonzero(grid == fg)) / 2))
        goal = min(max_extra, max(1, (h * w) // 13))
        if h <= 9 or w <= 9:
            goal = min(goal, 2)
        _add_bars(grid, goal, protected, targets, bg, fg, rng, clean,
                  seek_overlap=not clean and rng.random() < 0.65)
        if (_centers(grid, bg, fg) != targets
            or not _bar_members(grid, fg)
            or max(_component_sizes(grid, fg), default=0) > 28
            or np.count_nonzero(grid == fg) * 2 >= h * w):
            continue
        output = grid.copy()
        for r, c in targets:
            output[r, c] = marker
        return {"input": grid, "output": output}
    raise ValueError("could not construct the requested pair")


def _counts(rng, seed, num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if value is not None and (not _plain_int(value) or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer from {lo} through {hi}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _shape(rng, regime):
    pick = rng.random()
    if regime == "source_train":
        bucket = "small" if pick < .25 else "medium" if pick < .78 else "large"
    else:
        bucket = "small" if pick < .12 else "medium" if pick < .42 else "large"
    if bucket == "small":
        h = int(rng.integers(8, 13))
        w = int(rng.integers(8, 15))
    elif bucket == "medium":
        h = int(rng.integers(13, 21))
        w = int(rng.integers(12, 22))
    else:
        h = int(rng.integers(20, 31))
        w = int(rng.integers(18, 31))
    if rng.random() < .25:
        w = h
    if rng.random() < .5:
        h, w = w, h
    return h, w


def _palette(rng, marker, fixed=None):
    if fixed is not None:
        return fixed
    choices = [color for color in range(10) if color != marker]
    picked = rng.choice(choices, size=2, replace=False)
    return int(picked[0]), int(picked[1]), marker


def _pair_plan(rng, shape, regime, train_index, allow_overlap):
    h, w = shape
    upper = min(5, max(1, (h * w - 20) // 65))
    count = int(rng.integers(1, upper + 1))
    if train_index == 0:
        count = 1
    if regime == "target_test" and upper >= 4 and rng.random() < .25:
        count = int(rng.integers(4, upper + 1))
    clean = train_index == 0 or not allow_overlap or rng.random() < .52
    density = float(rng.uniform(.115, .245 if regime == "source_train" else .29))
    if train_index == 0:
        density = min(density, .17)
    kinds = ["horizontal", "vertical", "omitted", "shifted", "gap"]
    if train_index == 0 or min(shape) < 12:
        decoys = []
    elif train_index == 1:
        decoys = ["horizontal", "vertical"] if rng.random() < .22 else [str(rng.choice(kinds[:2]))]
    else:
        decoys = [str(rng.choice(kinds)) for _ in range(int(rng.integers(0, 3)))]
        if min(shape) >= 16 and rng.random() < .16:
            decoys = ["horizontal", "vertical"]
    return count, clean, density, decoys, rng.random() < .28


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _plain_int(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, seed, num_train, num_test)
    marker = int(rng.integers(10))
    fixed_palette = _palette(rng, marker)
    varied_palette = rng.random() < .48
    puzzle = {"train": [], "test": []}
    seen = set()
    training_overlap = False
    train_shapes = []
    for split, length in (("train", ntrain), ("test", ntest)):
        for index in range(length):
            regime = "source_train" if split == "train" else "target_test"
            for attempt in range(80):
                if split == "train" and index == 0:
                    shape = (8, 8) if rng.random() < .25 else _shape(rng, regime)
                    if min(shape) > 16:
                        shape = (int(rng.integers(8, 14)), int(rng.integers(8, 15)))
                elif split == "train" and index == 1:
                    shape = (int(rng.integers(15, 23)), int(rng.integers(15, 23)))
                elif split == "test" and train_shapes and rng.random() < .28:
                    shape = train_shapes[int(rng.integers(len(train_shapes)))]
                else:
                    shape = _shape(rng, regime)
                if varied_palette and (split == "test" or index > 0):
                    colors = _palette(rng, marker)
                    if split == "train" and index == 1 and colors == fixed_palette:
                        continue
                else:
                    colors = fixed_palette
                allow_overlap = split == "train" and index > 0 or training_overlap
                count, clean, density, decoys, close = _pair_plan(
                    rng, shape, regime, index if split == "train" else -1, allow_overlap)
                if split == "train" and index == 1 and seed % 2 == 0:
                    clean = False
                try:
                    pair = _make_pair(rng, shape, count, colors, clean, density, decoys, close)
                except ValueError:
                    continue
                key = _grid_key(pair["input"])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_shapes.append(shape)
                    if not _domino_components(pair["input"], colors[1]):
                        training_overlap = True
                break
            else:
                raise ValueError("unable to generate unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles(pair):
    inp, out = pair["input"], pair["output"]
    changed = inp != out
    _require(bool(changed.any()), "each pair must demonstrate a marker")
    old = np.unique(inp[changed])
    new = np.unique(out[changed])
    _require(len(old) == len(new) == 1, "one color change per pair")
    bg, marker = int(old[0]), int(new[0])
    fg = [int(x) for x in np.unique(inp) if int(x) != bg]
    _require(len(fg) == 1 and len({bg, fg[0], marker}) == 3,
             "background, bar, and marker roles must be distinct")
    return bg, fg[0], marker


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    seen, markers = set(), set()
    has_clean_train = has_decoy_train = False
    has_overlap_train = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2D ndarray")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                         and np.issubdtype(grid.dtype, np.integer)
                         and bool(((0 <= grid) & (grid <= 9)).all()),
                         "invalid grid dimensions, dtype, or colors")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape, "shape changed")
            source_encoding = "/".join("".join(str(int(v)) for v in row) for row in inp)
            _require(source_encoding not in _SOURCE_INPUTS, "exact source input copy")
            key = _grid_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            bg, fg, marker = _roles(pair)
            markers.add(marker)
            _require(np.count_nonzero(inp == bg) > np.count_nonzero(inp == fg),
                     "background must be dominant")
            _require(set(np.unique(inp)) == {bg, fg}, "input roles inconsistent")
            centers = _centers(inp, bg, fg)
            expected = inp.copy()
            for r, c in centers:
                expected[r, c] = marker
            _require(np.array_equal(out, expected), "output violates midpoint rule")
            _require(_bar_members(inp, fg), "foreground contains an isolated cell")
            _require(max(_component_sizes(inp, fg), default=0) <= 28,
                     "foreground contains an oversized merged component")
            if split == "train":
                has_clean_train |= _domino_components(inp, fg)
                has_overlap_train |= not _domino_components(inp, fg)
                h_only = _axis_midpoints(inp, bg, fg, "horizontal") - centers
                v_only = _axis_midpoints(inp, bg, fg, "vertical") - centers
                has_decoy_train |= bool(h_only or v_only)
            else:
                _require(_domino_components(inp, fg) or has_overlap_train,
                         "test overlap lacks a training demonstration")
    _require(len(markers) == 1, "marker color must be fixed within a puzzle")
    _require(has_clean_train, "training needs a clean domino scene")
    _require(has_decoy_train, "training needs an unmarked one-axis midpoint")


def self_test() -> None:
    def equal(a, b):
        return all(np.array_equal(x[k], y[k]) for split in ("train", "test")
                   for x, y in zip(a[split], b[split]) for k in ("input", "output"))
    _require(equal(generate_puzzle(314159, 4, 2), generate_puzzle(314159, 4, 2)),
             "generation is not deterministic")
    for seed, nt, ns in ((501, 2, 1), (502, 10, 1), (503, 10, 4),
                         (601, None, 1), (602, None, 4),
                         (603, 2, None), (604, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(p)
        _require((nt is None or len(p["train"]) == nt)
                 and (ns is None or len(p["test"]) == ns), "count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(700, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    trains, tests, shapes, counts, palettes = set(), set(), set(), set(), set()
    clean_pairs = overlap_pairs = h_only_pairs = v_only_pairs = 0
    no_decoy = both_decoys = source_like_small = rectangular = 0
    low_test_counts = high_train_counts = varied_palettes = matched_shapes = 0
    close_layouts = separated_layouts = 0
    signature = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzle_palettes, split_shapes = set(), {"train": set(), "test": set()}
        trains.add(len(p["train"]))
        tests.add(len(p["test"]))
        signature.add(tuple(_grid_key(x["input"]) for s in ("train", "test")
                            for x in p[s]))
        for s in ("train", "test"):
            for pair in p[s]:
                inp = pair["input"]
                bg, fg, marker = _roles(pair)
                centers = _centers(inp, bg, fg)
                horizontal = _axis_midpoints(inp, bg, fg, "horizontal") - centers
                vertical = _axis_midpoints(inp, bg, fg, "vertical") - centers
                shapes.add(inp.shape)
                split_shapes[s].add(inp.shape)
                counts.add(len(centers))
                palettes.add((bg, fg, marker))
                puzzle_palettes.add((bg, fg))
                clean_pairs += _domino_components(inp, fg)
                overlap_pairs += not _domino_components(inp, fg)
                h_only_pairs += bool(horizontal)
                v_only_pairs += bool(vertical)
                no_decoy += not (horizontal or vertical)
                both_decoys += bool(horizontal and vertical)
                source_like_small += inp.shape == (8, 8) and np.count_nonzero(inp == fg) == 10
                rectangular += inp.shape[0] != inp.shape[1]
                low_test_counts += s == "test" and len(centers) in (1, 2)
                high_train_counts += s == "train" and len(centers) >= 3
                if len(centers) >= 2:
                    distances = [abs(r-u)+abs(c-v) for r,c in centers for u,v in centers
                                 if (r,c) != (u,v)]
                    close_layouts += min(distances) <= 8
                    separated_layouts += min(distances) > 8
        varied_palettes += len(puzzle_palettes) > 1
        matched_shapes += bool(split_shapes["train"] & split_shapes["test"])
    _require(len(trains) >= 4 and len(tests) >= 3, "automatic counts lack variation")
    _require(len(shapes) >= 25 and len(counts) >= 3 and len(palettes) >= 20,
             "geometry, counts, or colors lack variation")
    _require(clean_pairs >= 30 and overlap_pairs >= 10, "component modes lack variation")
    _require(h_only_pairs >= 8 and v_only_pairs >= 8, "one-axis decoys lack variation")
    _require(no_decoy >= 8 and both_decoys >= 8,
             "no-decoy and two-axis-decoy scenes lack variation")
    _require(source_like_small >= 2 and rectangular >= 20,
             "source-like small or rectangular geometry is missing")
    _require(low_test_counts >= 5 and high_train_counts >= 5,
             "target counts remain tied to the split")
    _require(varied_palettes >= 5 and matched_shapes >= 5,
             "within-puzzle palette or matched-size variation is missing")
    _require(close_layouts >= 5 and separated_layouts >= 5,
             "center spacing lacks variation")
    _require(len(signature) == 42, "duplicate puzzles across seeds")
