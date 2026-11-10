"""Generate orthogonal wall networks and fill by boundary reachability."""

import hashlib

import numpy as np


_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_SOURCE_INPUTS = {
    ((22, 25), "27b6da0b6fdd19044be2f809a850240461b6ba17dd06a0d1e87bd1e4b4ed6a62"),
    ((23, 23), "c69fbb2a93bc555afd689b75e835dc33b15a0e5810f31eebb3a433fb33b04946"),
    ((24, 21), "bfb22721ea8c952b129a6786c0b31334b90dae4d48aefc8e62daae112039a85c"),
    ((22, 25), "62948c9eb140938de95e36a1c79369bb0030d998ed539b75e2176723886f90e6"),
}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif nt is None:
        nt = int(rng.integers(ns + 1, 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for yy, xx in np.argwhere(mask):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        cells = []
        stack = [(y, x)]
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for dy, dx in _NEIGHBORS:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        result.append(cells)
    return result


def _exterior(background):
    h, w = background.shape
    seen = np.zeros((h, w), dtype=bool)
    stack = []
    for x in range(w):
        stack.extend(((0, x), (h - 1, x)))
    for y in range(h):
        stack.extend(((y, 0), (y, w - 1)))
    while stack:
        y, x = stack.pop()
        if seen[y, x] or not background[y, x]:
            continue
        seen[y, x] = True
        for dy, dx in _NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and background[ny, nx] and not seen[ny, nx]:
                stack.append((ny, nx))
    return seen


def _solve(inp, background, enclosed_color, exterior_color):
    wall = inp != background
    outside = _exterior(~wall)
    out = np.full(inp.shape, enclosed_color, dtype=np.int8)
    out[outside] = exterior_color
    out[wall] = inp[wall]
    return out


def _line(grid, y0, x0, y1, x1, color):
    if y0 == y1:
        grid[y0, min(x0, x1):max(x0, x1) + 1] = color
    elif x0 == x1:
        grid[min(y0, y1):max(y0, y1) + 1, x0] = color
    else:
        raise ValueError("only orthogonal lines are permitted")


def _rectangle(grid, y0, x0, y1, x1, color):
    _line(grid, y0, x0, y0, x1, color)
    _line(grid, y1, x0, y1, x1, color)
    _line(grid, y0, x0, y1, x0, color)
    _line(grid, y0, x1, y1, x1, color)


def _shape(rng):
    if rng.random() < 0.09:
        return ((22, 25), (23, 23), (24, 21))[int(rng.integers(3))]
    buckets = ((11, 17), (17, 26), (26, 31))
    h0, h1 = buckets[int(rng.choice(3, p=(0.27, 0.48, 0.25)))]
    w0, w1 = buckets[int(rng.choice(3, p=(0.27, 0.48, 0.25)))]
    return int(rng.integers(h0, h1)), int(rng.integers(w0, w1))


def _bounds(rng, size, minimum=6):
    for _ in range(30):
        a = int(rng.integers(1, min(7, size - minimum - 1)))
        b = size - 1 - int(rng.integers(1, min(7, size - minimum - a)))
        if b - a >= minimum:
            return a, b
    return 1, size - 2


def _coordinates(rng, start, stop, count):
    span = stop - start
    if span < 3 * count:
        return None
    gaps = (3 + rng.multinomial(span - 3 * count, [1 / count] * count)).tolist()
    result = [start]
    for gap in gaps:
        result.append(result[-1] + int(gap))
    return result


def _connected_cells(rng, rows, cols, count):
    first = (int(rng.integers(rows)), int(rng.integers(cols)))
    cells = {first}
    while len(cells) < count:
        frontier = {
            (r + dr, c + dc)
            for r, c in cells for dr, dc in _NEIGHBORS
            if 0 <= r + dr < rows and 0 <= c + dc < cols and (r + dr, c + dc) not in cells
        }
        cells.add(sorted(frontier)[int(rng.integers(len(frontier)))])
    return cells


def _erase_seam(grid, ys, xs, a, b, background, rng, long_open=False):
    r, c = a
    if b[0] == r:
        fixed = xs[max(c, b[1])]
        lo, hi = ys[r], ys[r + 1]
        vertical = True
    else:
        fixed = ys[max(r, b[0])]
        lo, hi = xs[c], xs[c + 1]
        vertical = False
    if hi - lo < 3:
        return
    if long_open and hi - lo >= 7:
        start, end = lo + 2, hi - 2
    else:
        length = int(rng.integers(1, hi - lo))
        start = int(rng.integers(lo + 1, hi - length + 1))
        end = start + length - 1
    if vertical:
        grid[start:end + 1, fixed] = background
    else:
        grid[fixed, start:end + 1] = background


def _open_outer_wall(grid, ys, xs, cells, background, rng):
    choices = []
    for r, c in cells:
        for dr, dc in _NEIGHBORS:
            if (r + dr, c + dc) in cells:
                continue
            if dr:
                y = ys[r + (dr > 0)]
                choices.extend((y, x) for x in range(xs[c] + 1, xs[c + 1]))
            else:
                x = xs[c + (dc > 0)]
                choices.extend((y, x) for y in range(ys[r] + 1, ys[r + 1]))
    if choices:
        y, x = choices[int(rng.integers(len(choices)))]
        grid[y, x] = background


def _branch(grid, background, color, rng, target, reach_edge=False):
    wall = grid == color
    outside = _exterior(~wall)
    h, w = grid.shape
    candidates = []
    for yy, xx in np.argwhere(wall):
        y, x = int(yy), int(xx)
        for dy, dx in _NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == background:
                if bool(outside[ny, nx]) == (target == "exterior"):
                    candidates.append((y, x, dy, dx))
    if not candidates:
        return
    for _ in range(min(30, len(candidates))):
        y, x, dy, dx = candidates[int(rng.integers(len(candidates)))]
        limit = max(h, w) if reach_edge else int(rng.integers(2, 10))
        path = []
        for step in range(1, limit + 1):
            ny, nx = y + step * dy, x + step * dx
            if not (0 <= ny < h and 0 <= nx < w) or grid[ny, nx] != background:
                break
            path.append((ny, nx))
            if reach_edge and (ny in (0, h - 1) or nx in (0, w - 1)):
                break
        if len(path) < 2:
            continue
        if reach_edge and path[-1][0] not in (0, h - 1) and path[-1][1] not in (0, w - 1):
            continue
        if not reach_edge and len(path) > 3 and rng.random() < 0.55:
            py, px = path[-1]
            turn = (dx, dy) if rng.random() < 0.5 else (-dx, -dy)
            for step in range(1, int(rng.integers(2, 7))):
                ny, nx = py + step * turn[0], px + step * turn[1]
                if not (0 <= ny < h and 0 <= nx < w) or grid[ny, nx] != background:
                    break
                path.append((ny, nx))
        for py, px in path:
            grid[py, px] = color
        return


def _generic_network(grid, background, color, rng, density, force_open=False):
    h, w = grid.shape
    y0, y1 = _bounds(rng, h)
    x0, x1 = _bounds(rng, w)
    max_rows = min(3, (y1 - y0) // 3)
    max_cols = min(3, (x1 - x0) // 3)
    rows = int(rng.integers(1, max_rows + 1))
    cols = int(rng.integers(1, max_cols + 1))
    if force_open and rows * cols == 1:
        rows, cols = max_rows, max_cols
    if density >= 5 and rows * cols < 3 and max_rows * max_cols >= 3:
        rows, cols = max_rows, max_cols
    ys = _coordinates(rng, y0, y1, rows)
    xs = _coordinates(rng, x0, x1, cols)
    if ys is None or xs is None:
        return False
    total = rows * cols
    count = int(rng.integers(1 if total == 1 else 2, min(total, max(2, density + 1)) + 1))
    cells = _connected_cells(rng, rows, cols, count)
    for r, c in cells:
        _rectangle(grid, ys[r], xs[c], ys[r + 1], xs[c + 1], color)
    seams = sorted((a, b) for a in cells for b in cells if b > a and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1)
    rng.shuffle(seams)
    for a, b in seams[:int(rng.integers(0, min(3, len(seams)) + 1))]:
        _erase_seam(grid, ys, xs, a, b, background, rng, long_open=rng.random() < 0.5)
    if len(cells) >= 2 and (force_open or rng.random() < 0.33):
        _open_outer_wall(grid, ys, xs, cells, background, rng)
    return True


def _cap_network(grid, background, color, rng):
    h, w = grid.shape
    if h < 16 or w < 14:
        return False
    y0, y2 = _bounds(rng, h, 10)
    x0, x2 = _bounds(rng, w, 9)
    if y2 - y0 < 10 or x2 - x0 < 9:
        return False
    y1 = y0 + int(rng.integers(3, min(5, y2 - y0 - 7) + 1))
    x1 = int(rng.integers(x0 + 4, x2 - 3))
    ys, xs = [y0, y1, y2], [x0, x1, x2]
    for r in range(2):
        for c in range(2):
            _rectangle(grid, ys[r], xs[c], ys[r + 1], xs[c + 1], color)
    _erase_seam(grid, ys, xs, (1, 0), (1, 1), background, rng, long_open=True)
    if rng.random() < 0.35 and y2 - y1 >= 8:
        yy = int(rng.integers(y1 + 2, y2 - 2))
        _line(grid, yy, x1, yy, x2, color)
    if rng.random() < 0.2:
        _open_outer_wall(grid, ys, xs, {(0, 0), (0, 1), (1, 0), (1, 1)}, background, rng)
    return True


def _valid_structure(inp, background):
    wall = inp != background
    if not np.any(wall) or len(_components(wall)) != 1:
        return False
    if np.any(wall[:-1, :-1] & wall[1:, :-1] & wall[:-1, 1:] & wall[1:, 1:]):
        return False
    outside = _exterior(~wall)
    return bool(np.any(outside) and np.any(~wall & ~outside))


def _has_near_loop(inp, background):
    wall = inp != background
    h, w = wall.shape
    inside = ~wall & ~_exterior(~wall)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if wall[y, x]:
                continue
            if not ((wall[y - 1, x] and wall[y + 1, x]) or
                    (wall[y, x - 1] and wall[y, x + 1])):
                continue
            closed = wall.copy()
            closed[y, x] = True
            newly_enclosed = (~closed & ~_exterior(~closed)) & ~inside
            if np.count_nonzero(newly_enclosed) >= 4:
                return True
    return False


def _make_input(rng, background, color, density, prefer_cap, force_open=False):
    for _ in range(90):
        h, w = _shape(rng)
        grid = np.full((h, w), background, dtype=np.int8)
        if prefer_cap and not force_open and h >= 16 and w >= 14 and rng.random() < 0.75:
            built = _cap_network(grid, background, color, rng)
        else:
            built = _generic_network(grid, background, color, rng, density, force_open)
        if not built:
            continue
        if rng.random() < 0.24:
            rows = [y for y in range(1, h - 1) if np.count_nonzero(grid[y] == color) >= 4]
            cols = [x for x in range(1, w - 1) if np.count_nonzero(grid[:, x] == color) >= 4]
            if rows and (not cols or rng.random() < 0.5):
                grid[rows[int(rng.integers(len(rows)))], :] = color
            elif cols:
                grid[:, cols[int(rng.integers(len(cols)))]] = color
        for _ in range(int(rng.integers(1, 5))):
            _branch(grid, background, color, rng, "enclosed" if rng.random() < 0.5 else "exterior")
        if rng.random() < 0.25:
            _branch(grid, background, color, rng, "exterior", reach_edge=True)
        if _valid_structure(grid, background) and (not force_open or _has_near_loop(grid, background)):
            return grid
    grid = np.full((18, 20), background, dtype=np.int8)
    _rectangle(grid, 3, 3, 14, 16, color)
    _line(grid, 3, 9, 14, 9, color)
    _line(grid, 8, 3, 8, 9, color)
    _branch(grid, background, color, rng, "exterior")
    if force_open:
        grid[3, 6] = background
    return grid


def _key(grid):
    a = np.ascontiguousarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _source_copy(grid):
    a = np.ascontiguousarray(grid, dtype=np.int8)
    return (a.shape, hashlib.sha256(a.tobytes()).hexdigest()) in _SOURCE_INPUTS


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _resolve_counts(rng, num_train, num_test)
    colors = [int(v) for v in rng.permutation(10)]
    background, enclosed_color, exterior_color = colors[:3]
    wall_colors = colors[3:]
    puzzle = {"train": [], "test": []}
    seen = set()
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=nt + ns, dtype=np.int64)
    for i in range(nt + ns):
        split = "train" if i < nt else "test"
        pair_rng = np.random.default_rng(int(pair_seeds[i]))
        color = wall_colors[int(pair_rng.integers(len(wall_colors)))]
        density = int(pair_rng.integers(1, 10 if split == "test" else 9))
        prefer_cap = pair_rng.random() < (0.43 if split == "test" else 0.25)
        for _ in range(50):
            inp = _make_input(pair_rng, background, color, density, prefer_cap, force_open=i == 0)
            key = _key(inp)
            if key not in seen and not _source_copy(inp):
                break
        else:
            raise ValueError("could not make a unique input")
        seen.add(key)
        puzzle[split].append({"input": inp, "output": _solve(inp, background, enclosed_color, exterior_color)})
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles(pair):
    inp, out = pair["input"], pair["output"]
    input_values = set(int(v) for v in np.unique(inp))
    output_values = set(int(v) for v in np.unique(out))
    _require(len(input_values) == 2 and len(output_values) == 3, "incorrect color roles")
    shared = input_values & output_values
    _require(len(shared) == 1, "wall color must be identifiable")
    wall_color = next(iter(shared))
    background = next(iter(input_values - shared))
    outside = _exterior(inp == background)
    inside = (inp == background) & ~outside
    _require(np.any(outside) and np.any(inside), "both background classes required")
    enclosed_colors = set(int(v) for v in np.unique(out[inside]))
    exterior_colors = set(int(v) for v in np.unique(out[outside]))
    _require(len(enclosed_colors) == len(exterior_colors) == 1, "inconsistent fill colors")
    enclosed_color = next(iter(enclosed_colors))
    exterior_color = next(iter(exterior_colors))
    _require(len({background, wall_color, enclosed_color, exterior_color}) == 4, "colors need distinct roles")
    _require(np.array_equal(out, _solve(inp, background, enclosed_color, exterior_color)), "incorrect flood fill")
    return background, enclosed_color, exterior_color


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "bad pair counts")
    seen_inputs, seen_pairs, roles = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be 2D NumPy array")
                h, w = grid.shape
                _require(1 <= h <= 30 and 1 <= w <= 30, "grid dimensions out of range")
                _require(np.issubdtype(grid.dtype, np.integer) and bool(np.all((0 <= grid) & (grid <= 9))), "invalid ARC colors")
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "bad output shape or unchanged grid")
            _require(not _source_copy(inp), "source input copied")
            pair_roles = _roles(pair)
            _require(_valid_structure(inp, pair_roles[0]), "invalid wall network")
            ik, pk = _key(inp), (_key(inp), _key(out))
            _require(ik not in seen_inputs and pk not in seen_pairs, "duplicate pair")
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            roles.add(pair_roles)
    _require(len(roles) == 1, "puzzle color conventions changed")
    first_train = puzzle["train"][0]
    _require(_has_near_loop(first_train["input"], _roles(first_train)[0]), "training lacks an open near-loop")


def _predict(train, inp):
    role_set = {_roles(pair) for pair in train}
    _require(len(role_set) == 1, "ambiguous training roles")
    background, enclosed_color, exterior_color = next(iter(role_set))
    _require(background in set(int(v) for v in np.unique(inp)), "unseen background convention")
    return _solve(inp, background, enclosed_color, exterior_color)


def _signature(puzzle):
    return tuple((split, _key(pair["input"]), _key(pair["output"])) for split in ("train", "test") for pair in puzzle[split])


def _row_regimes(cells):
    by_row = {}
    for y, x in cells:
        by_row.setdefault(y, []).append(x)
    profile = []
    for y in sorted(by_row):
        xs = sorted(by_row[y])
        runs = sum(i == 0 or x > xs[i - 1] + 1 for i, x in enumerate(xs))
        if not profile or profile[-1] != runs:
            profile.append(runs)
    return len(profile)


def self_test() -> None:
    cases = ((12345, None, None), (54321, 6, 3), (61000, 2, 1), (61001, 10, 1), (61002, 10, 4), (62000, None, 1), (62001, None, 4), (62002, 2, None), (62003, 10, None))
    for seed, nt, ns in cases:
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        _require(_signature(puzzle) == _signature(generate_puzzle(seed, nt, ns)), "nondeterministic puzzle")
        if nt is not None:
            _require(len(puzzle["train"]) == nt, "train count not honored")
        if ns is not None:
            _require(len(puzzle["test"]) == ns, "test count not honored")
        for pair in puzzle["test"]:
            _require(np.array_equal(_predict(puzzle["train"], pair["input"]), pair["output"]), "unrecoverable test output")
    for i, (nt, ns) in enumerate(((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4))):
        try:
            generate_puzzle(63000 + i, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_signature(p) for p in puzzles}) == 42, "cross-seed duplicates")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack range")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack range")
    shapes, region_counts, backgrounds, fill_pairs = set(), set(), set(), set()
    full_lines = irregular = near_loops = shallow_large = row_regimes = 0
    small = large = disconnected_exterior = count_overlap = simpler_tests = harder_tests = 0
    source_dimensions = {(22, 25), (23, 23), (24, 21)}
    seen_source_dimensions = set()
    for puzzle in puzzles:
        split_counts = {}
        for split in ("train", "test"):
            counts = set()
            for pair in puzzle[split]:
                inp = pair["input"]
                background, ec, xc = _roles(pair)
                wall = inp != background
                outside = _exterior(~wall)
                inside = ~wall & ~outside
                comps = _components(inside)
                counts.add(len(comps))
                shapes.add(inp.shape)
                seen_source_dimensions.add(inp.shape)
                region_counts.add(len(comps))
                backgrounds.add(background)
                fill_pairs.add((ec, xc))
                small += int(max(inp.shape) <= 16)
                large += int(max(inp.shape) == 30)
                disconnected_exterior += int(len(_components(outside)) >= 2)
                full_lines += int(bool(np.any(np.all(wall, axis=0)) or np.any(np.all(wall, axis=1))))
                if split == "test":
                    near_loops += int(_has_near_loop(inp, background))
                shallow = 0
                big = False
                for comp in comps:
                    ys = [y for y, _ in comp]
                    xs = [x for _, x in comp]
                    irregular += int(len(comp) < (max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1))
                    shallow += int(max(ys) - min(ys) + 1 <= 3)
                    big |= len(comp) >= 20
                    row_regimes += int(len(comp) >= 20 and _row_regimes(comp) >= 3)
                shallow_large += int(shallow >= 2 and big)
            split_counts[split] = counts
        _require(_has_near_loop(puzzle["train"][0]["input"], _roles(puzzle["train"][0])[0]), "training lacks an open near-loop")
        count_overlap += int(bool(split_counts["train"] & split_counts["test"]))
        simpler_tests += int(min(split_counts["test"]) < max(split_counts["train"]))
        harder_tests += int(max(split_counts["test"]) > max(split_counts["train"]))
        for pair in puzzle["test"]:
            _require(np.array_equal(_predict(puzzle["train"], pair["input"]), pair["output"]), "test prediction failed")
    _require(len(shapes) >= 35 and len(region_counts) >= 5, "insufficient geometry diversity")
    _require(len(backgrounds) >= 7 and len(fill_pairs) >= 15, "insufficient palette diversity")
    _require(full_lines >= 10 and irregular >= 5, "missing edge lines or nonrectangular regions")
    _require(small >= 5 and large >= 5 and source_dimensions <= seen_source_dimensions, "size variation is missing")
    _require(disconnected_exterior >= 10 and near_loops >= 3, "edge topology or open loops are missing")
    _require(shallow_large >= 5 and row_regimes >= 5, "source-like changing-width chambers are missing")
    _require(count_overlap >= 5 and simpler_tests >= 5 and harder_tests >= 2, "train/test counts are too constrained")
