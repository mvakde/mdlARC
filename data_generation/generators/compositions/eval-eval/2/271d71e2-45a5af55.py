"""Fuel-limited concentric ring looms. Only NumPy and the standard library."""
from __future__ import annotations

import numpy as np


_DIRECTIONS = ("U", "R", "D", "L")  # turns needed to make the ribbon point up


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return g


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _key(g):
    g = np.asarray(g)
    return g.shape, tuple(map(int, g.flat))


def _components(g, background):
    """Read complete foreground components, without a majority-color heuristic."""
    h, w = g.shape
    visited = np.zeros(g.shape, dtype=bool)
    result = []
    for sy, sx in np.argwhere(g != background):
        sy, sx = int(sy), int(sx)
        if visited[sy, sx]:
            continue
        stack, cells = [(sy, sx)], []
        visited[sy, sx] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w and not visited[yy, xx]
                        and g[yy, xx] != background):
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        ys, xs = zip(*cells)
        t, b, l, r = min(ys), max(ys), min(xs), max(xs)
        if len(cells) != (b - t + 1) * (r - l + 1):
            raise ValueError("each input loom must be a filled rectangular component")
        result.append((t, b, l, r))
    return result


def _unrotate_point(y, x, original_shape, turns):
    h, w = original_shape
    if turns == 0:
        return y, x
    if turns == 1:
        return x, w - 1 - y
    if turns == 2:
        return h - 1 - y, w - 1 - x
    return h - 1 - x, y


def _decode_component(g, box, background):
    t, b, l, r = box
    block = g[t:b + 1, l:r + 1]
    candidates = []
    for turns in range(4):
        v = np.rot90(block, turns)
        length, width = v.shape
        if width < 3 or length < 5:
            continue
        frame = int(v[0, 0])
        if not np.all(v[0] == frame):
            continue
        for start in range(2, length - 2):
            spent, fuel = int(v[start, 0]), int(v[start, -1])
            if len({background, frame, spent, fuel}) != 4:
                continue
            if not (np.all(v[start, 1:-1] == frame)
                    and np.all(v[-1] == frame)
                    and np.all(v[start + 1:, 0] == frame)
                    and np.all(v[start + 1:, -1] == frame)):
                continue
            interior = v[start + 1:-1, 1:-1]
            q = int(np.count_nonzero(interior == spent))
            if not np.array_equal(interior.ravel(),
                                  [spent] * q + [fuel] * (interior.size - q)):
                continue
            stripes = v[1:start]
            if not np.all(stripes == stripes[:, :1]):
                continue
            code = tuple(map(int, stripes[::-1, 0]))
            if set(code) & {background, frame, spent, fuel}:
                continue
            depth = start - 1
            points = [_unrotate_point(y, x, block.shape, turns)
                      for y in (start, length - 1) for x in (0, width - 1)]
            ys = [t + p[0] for p in points]
            xs = [l + p[1] for p in points]
            core = (min(ys), max(ys), min(xs), max(xs))
            ct, cb, cl, cr = core
            # Full advertised growth plus the cap, even if fuel runs out early.
            bounds = (ct - depth - int(turns == 0), cb + depth + int(turns == 2),
                      cl - depth - int(turns == 3), cr + depth + int(turns == 1))
            candidates.append(dict(block=v.copy(), turns=turns, start=start,
                                   depth=depth, code=code, frame=frame,
                                   spent=spent, fuel=fuel, advanced=q,
                                   capacity=int(interior.size),
                                   remaining=int(interior.size - q),
                                   core=core, bounds=bounds, background=background,
                                   rows=interior.shape[0], columns=interior.shape[1]))
    if len(candidates) != 1:
        raise ValueError("loom does not have one unique keyed interpretation")
    return candidates[0]


def _read_scene(input_grid):
    g = _grid(input_grid)
    background = int(g[0, 0])
    if any(int(g[p]) != background for p in
           ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("all input corners must identify the background")
    boxes = _components(g, background)
    if not boxes:
        raise ValueError("input contains no loom")
    looms = [_decode_component(g, box, background) for box in boxes]
    reserved = np.zeros(g.shape, dtype=np.int8)
    for v in looms:
        t, b, l, r = v["bounds"]
        if not (0 <= t <= b < g.shape[0] and 0 <= l <= r < g.shape[1]):
            raise ValueError("advertised shells or cap would leave the canvas")
        reserved[t:b + 1, l:r + 1] += 1
    if np.max(reserved) > 1:
        raise ValueError("advertised growth footprints overlap")
    return looms


def _patch(v, variant="correct"):
    """Render by distance from the central rectangle; keys and source survive."""
    block, d, start = v["block"], v["depth"], v["start"]
    length, width = block.shape
    out = np.full((length + d, width + 2 * d), v["background"], dtype=int)
    out[:length, d:d + width] = block
    code = v["code"]
    steps = min(v["remaining"], d)
    if variant == "ignore_fuel":
        steps = d
    elif variant == "capacity":
        steps = min(v["capacity"], d)
    elif variant == "advanced":
        steps = min(v["advanced"], d)
    elif variant == "reverse":
        code = code[::-1]
    elif variant == "sorted":
        code = tuple(sorted(code))
    elif variant == "collapse":
        code = tuple(c for i, c in enumerate(code) if i == 0 or c != code[i - 1])
        steps = min(v["remaining"], len(code))
    y, x = np.indices(out.shape)
    dy = np.maximum(np.maximum(start - y, y - (length - 1)), 0)
    dx = np.maximum(np.maximum(d - x, x - (d + width - 1)), 0)
    radius = dy + dx if variant == "manhattan" else np.maximum(dy, dx)
    if variant != "ribbon_only":
        for k in range(1, steps + 1):
            out[radius == k] = code[k - 1]
    capacity, q = v["capacity"], min(v["capacity"], v["advanced"] + steps)
    out[start + 1:length - 1, d + 1:d + width - 1] = np.array(
        [v["spent"]] * q + [v["fuel"]] * (capacity - q)
    ).reshape(v["rows"], v["columns"])
    return np.rot90(out, -v["turns"]).copy()


def _transform(g, looms, variant="correct"):
    out = np.asarray(g).copy()
    for v in looms:
        t, b, l, r = v["bounds"]
        patch = _patch(v, variant)
        _need(patch.shape == (b - t + 1, r - l + 1), "footprint mapping failed")
        out[t:b + 1, l:r + 1] = patch
        if variant == "global_scan":
            ct, cb, cl, cr = v["core"]
            counter = np.asarray(g)[ct + 1:cb, cl + 1:cr].copy()
            pending = np.flatnonzero(counter.ravel() == v["fuel"])
            for index in pending[:min(v["remaining"], v["depth"])]:
                counter.flat[int(index)] = v["spent"]
            out[ct + 1:cb, cl + 1:cr] = counter
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver: no training examples, random state, or cached roles."""
    g = _grid(input_grid)
    return _transform(g, _read_scene(g))


def _alternate_patch(v):
    """Authored cross-check: consume single cells and wrap whole borders."""
    block, d, s = v["block"], v["depth"], v["start"]
    core = block[s:].copy()
    counter = core[1:-1, 1:-1].copy()
    steps = 0
    while steps < d:
        pending = np.flatnonzero(counter.ravel() == v["fuel"])
        if not len(pending):
            break
        counter.flat[int(pending[0])] = v["spent"]
        steps += 1
    core[1:-1, 1:-1] = counter
    for color in v["code"][:steps]:
        core = np.pad(core, 1, constant_values=color)
    length, width = block.shape
    result = np.full((length + d, width + 2 * d), v["background"], dtype=int)
    result[:length, d:d + width] = block
    result[s - steps:s - steps + core.shape[0],
           d - steps:d - steps + core.shape[1]] = core
    return np.rot90(result, -v["turns"]).copy()


def _choose(rng, values):
    values = list(values)
    return values[int(rng.integers(len(values)))]


def _palette(rng, base, style):
    return base if style == "shared" else tuple(map(int, rng.permutation(10)))


def _sample_loom(rng, direction, lesson, regime, is_test, compact, palette):
    bg, frame, spent, fuel = palette[:4]
    colors = palette[4:]
    if regime == "depth":
        depths = range(6, 10) if is_test else range(1, 6)
    else:
        depths = range(1, 7)
    if compact:
        depths = range(1, 3 if compact == 2 else 4)
    if lesson == "supply":
        depths = [d for d in depths if 3 <= d <= 5]
    elif lesson == "fuel":
        depths = [d for d in depths if 4 <= d <= 6]
    d = int(_choose(rng, depths))
    max_rows, max_cols = 27 - 2 * d, 28 - 2 * d
    if compact:
        max_rows, max_cols = min(max_rows, 4), min(max_cols, 4)
    elif not is_test and regime == "aspect":
        max_rows, max_cols = min(max_rows, 6), min(max_cols, 6)
    elif lesson:
        max_rows, max_cols = min(max_rows, 6), min(max_cols, 6)
    else:
        max_rows, max_cols = min(max_rows, 15), min(max_cols, 15)
    if is_test and regime == "aspect":
        # The long axis is novel; the ribbon can be on any side of that core.
        if rng.random() < .5:
            sizes = [(a, b) for a in range(9, max_rows + 1) for b in (1, 2)]
        else:
            sizes = [(a, b) for a in (1, 2) for b in range(9, max_cols + 1)]
        if not sizes:
            return None
    else:
        sizes = [(a, b) for a in range(1, max_rows + 1)
                 for b in range(1, max_cols + 1)]
        if lesson:
            sizes = [(a, b) for a, b in sizes if a >= 2 and b >= 2 and a != b]
        elif rng.random() < .5:
            sizes = [(a, b) for a, b in sizes if min(a, b) <= 3]
    if lesson == "supply":
        options = [(a, b, q) for a, b in sizes for q in range(1, a * b - d)
                   if q // b < (q + d) // b and (q + d) % b != 0]
        def exposes_rotation(option):
            a, b, q = option
            canonical = np.array([spent] * q + [fuel] * (a * b - q)).reshape(a, b)
            wrong = np.rot90(canonical, -direction).copy()
            for index in np.flatnonzero(wrong.ravel() == fuel)[:d]:
                wrong.flat[int(index)] = spent
            right = np.array([spent] * (q + d) + [fuel] * (a * b - q - d)).reshape(a, b)
            return not np.array_equal(np.rot90(wrong, direction), right)
        options = [option for option in options if exposes_rotation(option)]
        if not options:
            return None
        rows, columns, q = _choose(rng, options)
        remaining = rows * columns - q
    else:
        if lesson == "fuel":
            sizes = [(a, b) for a, b in sizes if a * b >= 3]
        if not sizes:
            return None
        rows, columns = _choose(rng, sizes)
        capacity = rows * columns
        if lesson == "fuel":
            remaining = int(_choose(rng, [f for f in range(2, min(d - 1, capacity - 1) + 1)
                                         if 2 * f != capacity]))
        else:
            stop = _choose(rng, ("supply", "fuel", "exact", "zero", "free", "free"))
            if stop == "supply" and capacity > d:
                remaining = int(rng.integers(d + 1, capacity + 1))
            elif stop == "fuel" and d > 1:
                remaining = int(rng.integers(1, min(d - 1, capacity) + 1))
            elif stop == "exact" and capacity >= d:
                remaining = d
            elif stop == "zero":
                remaining = 0
            else:
                remaining = int(rng.integers(0, capacity + 1))
        q = capacity - remaining
    capacity = rows * columns
    size = int(rng.integers(1, min(6, d) + 1))
    chosen = tuple(map(int, rng.choice(colors, size=size, replace=False)))
    # Cover the chosen palette, then insert repeats/returns at arbitrary depths.
    # This includes simple unique layers, broad bands and richer recurring codes.
    code = list(map(int, rng.permutation(chosen)))
    while len(code) < d:
        index = int(rng.integers(0, len(code) + 1))
        color = code[max(0, index - 1)] if rng.random() < .4 else _choose(rng, chosen)
        code.insert(index, color)
    if lesson == "supply":
        c1, c2 = map(int, rng.choice(colors, size=2, replace=False))
        code[0] = code[1] = c1
        code[-1] = c2
    elif lesson == "fuel":
        code[0], code[1] = map(int, rng.choice(colors, size=2, replace=False))
        code[2] = code[0]  # a return distinguishes literal order from sorting
    # Core and code form one rectangular foreground component. Two leading
    # corner keys are part of the frame, not a separate legend panel.
    h, w = rows + 2, columns + 2
    block = np.full((h + d + 1, w), frame, dtype=int)
    block[1:d + 1] = np.asarray(code[::-1])[:, None]
    s = d + 1
    block[s, 0], block[s, -1] = spent, fuel
    block[s + 1:-1, 1:-1] = np.asarray(
        [spent] * q + [fuel] * remaining).reshape(rows, columns)
    footprint = np.full((h + 2 * d + 1, w + 2 * d), bg, dtype=int)
    footprint[:len(block), d:d + w] = block
    return np.rot90(footprint, -direction).copy()


def _positions(occupied, h, w):
    H, W = occupied.shape
    if h > H or w > W:
        return []
    # Leave one cell between full growth footprints; no obligatory outer margin.
    padded = np.pad(occupied, 1)
    blocked = np.zeros_like(occupied)
    for y in range(3):
        for x in range(3):
            blocked |= padded[y:y + H, x:x + W]
    integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(sums == 0)


def _make_scene(rng, parameters, lesson, direction, is_test, wanted, seen):
    regime, style, base = parameters["regime"], parameters["style"], parameters["palette"]
    # One pair-level background, but geometric keys permit local role palettes.
    scene_palette = _palette(rng, base, "shared" if style == "shared" else "pair")
    background = scene_palette[0]
    for attempt in range(32):
        fallback = attempt >= 16
        H, W = ((30, 30) if fallback or (is_test and regime == "depth") else
                (int(rng.integers(16, 31)), int(rng.integers(16, 31))))
        if not fallback and not (is_test and regime == "depth") and rng.random() < .22:
            H = W = int(rng.integers(18, 31))
        g = np.full((H, W), background, dtype=int)
        occupied = np.zeros_like(g, dtype=bool)
        okay = True
        for j in range(wanted):
            placed = False
            for trial in range(40):
                k = direction if j == 0 else int(_choose(rng, parameters["test_dirs"]
                                                        if is_test else parameters["train_dirs"]))
                pal = scene_palette
                if style == "local":
                    tail = list(scene_palette[1:])
                    rng.shuffle(tail)
                    pal = (background, *tail)
                compact = 0 if wanted == 1 else (2 if fallback or trial >= 20 else 1)
                footprint = _sample_loom(rng, k, lesson if j == 0 else None,
                                         regime, is_test, compact, pal)
                if footprint is None:
                    continue
                h, w = footprint.shape
                positions = _positions(occupied, h, w)
                if not len(positions):
                    continue
                if rng.random() < .4:
                    edges = [p for p in positions if p[0] == 0 or p[1] == 0
                             or p[0] + h == H or p[1] + w == W]
                    if len(edges):
                        positions = edges
                y, x = map(int, _choose(rng, positions))
                g[y:y + h, x:x + w] = footprint
                occupied[y:y + h, x:x + w] = True
                placed = True
                break
            if not placed:
                okay = False
                break
        if not okay:
            continue
        # Crop empty canvas around the reserved regions, using independently
        # sampled margins. Reserved regions are actual geometry, not slots.
        ys, xs = np.nonzero(occupied)
        top = int(rng.integers(0, int(ys.min()) + 1))
        left = int(rng.integers(0, int(xs.min()) + 1))
        bottom = int(rng.integers(int(ys.max()) + 1, H + 1))
        right = int(rng.integers(int(xs.max()) + 1, W + 1))
        g = g[top:bottom, left:right].copy()
        if _key(g) in seen:
            continue
        try:
            looms = _read_scene(g)
        except ValueError:
            continue
        if is_test and all(v["remaining"] == 0 for v in looms):
            continue
        seen.add(_key(g))
        return {"input": g, "output": _transform(g, looms)}
    raise RuntimeError("bounded scene construction failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count, test_count = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    regime = str(_choose(rng, ("depth", "aspect", "orientation", "count")))
    directions = list(map(int, rng.permutation(4)))
    if directions[0] == 0:
        directions[0], directions[1] = directions[1], directions[0]
    pool = tuple(directions[:2]) if regime == "orientation" else tuple(range(4))
    parameters = dict(regime=regime, train_dirs=pool,
                      test_dirs=tuple(directions[2:]) if regime == "orientation" else pool,
                      style=str(_choose(rng, ("shared", "pair", "local"))),
                      palette=tuple(map(int, rng.permutation(10))))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(train_count):
        lesson = ("supply", "fuel")[i] if i < 2 else None
        direction = directions[i] if i < 2 else int(_choose(rng, pool))
        if regime == "orientation" and i < 2:
            direction = pool[i]
        wanted = 1 if lesson or regime in ("depth", "aspect", "count") else int(_choose(rng, (1, 1, 2, 3)))
        puzzle["train"].append(_make_scene(rng, parameters, lesson, direction, False, wanted, seen))
    rng.shuffle(puzzle["train"])
    for i in range(test_count):
        direction = int(_choose(rng, parameters["test_dirs"]))
        wanted = (int(rng.integers(2, 5)) if regime == "count" else
                  int(_choose(rng, (1, 1, 2, 3))) if regime == "orientation" else 1)
        puzzle["test"].append(_make_scene(rng, parameters, None, direction, True, wanted, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(puzzle):
    return [(split, np.asarray(p["input"]).shape, _read_scene(p["input"]))
            for split in ("train", "test") for p in puzzle[split]]


def _shift_types(records):
    train = [v for s, _, vs in records if s == "train" for v in vs]
    test = [v for s, _, vs in records if s == "test" for v in vs]
    result = set()
    if min(v["depth"] for v in test) > max(v["depth"] for v in train):
        result.add("depth")
    if min(max(v["rows"], v["columns"]) for v in test) > max(max(v["rows"], v["columns"]) for v in train):
        result.add("aspect")
    if {v["turns"] for v in test} - {v["turns"] for v in train}:
        result.add("orientation")
    if min(len(vs) for s, _, vs in records if s == "test") > max(len(vs) for s, _, vs in records if s == "train"):
        result.add("count")
    return result


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, records, alternatives = set(), [], [], {name: False for name in
        ("ignore_fuel", "capacity", "advanced", "ribbon_only", "reverse", "sorted",
         "collapse", "manhattan", "global_scan")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            g, answer = _grid(pair["input"]), _grid(pair["output"])
            _need(g.shape == answer.shape, "canvas dimensions must be preserved")
            _need(_key(g) not in seen, "duplicate input")
            seen.add(_key(g))
            looms = _read_scene(g)
            records.append((split, g.shape, looms))
            expected = _transform(g, looms)
            _need(np.array_equal(expected, answer), "incorrect output")
            alternate = g.copy()
            for v in looms:
                t, b, l, r = v["bounds"]
                alternate[t:b + 1, l:r + 1] = _alternate_patch(v)
                m = min(v["remaining"], v["depth"])
                _need(m <= v["remaining"], "overconsumed fuel")
                actual = np.rot90(answer[t:b + 1, l:r + 1], v["turns"])
                d, s, length, width = v["depth"], v["start"], *v["block"].shape
                counter = actual[s + 1:length - 1, d + 1:d + width - 1]
                _need(np.count_nonzero(counter == v["spent"]) == v["advanced"] + m,
                      "counter did not pay exactly one cell per shell")
                _need(np.array_equal(actual[0, d:d + width], v["block"][0]), "cap changed")
                for k in range(m + 1, d + 1):
                    _need(np.array_equal(actual[s - k, d:d + width], v["block"][s - k]),
                          "unused stripe changed")
            _need(np.array_equal(alternate, answer), "unit ticks and border wrapping disagree")
            if split == "train":
                training.extend(looms)
                for name in alternatives:
                    alternatives[name] |= not np.array_equal(_transform(g, looms, name), answer)
    _need(all(alternatives.values()), "training permits a simpler competing rule")
    _need(any(0 < v["remaining"] < v["depth"] for v in training), "missing fuel-limited demonstration")
    _need(any(v["remaining"] > v["depth"] for v in training), "missing ribbon-limited demonstration")
    active = [v for v in training if min(v["remaining"], v["depth"]) > 0]
    _need(len({v["turns"] for v in active}) >= 2, "training must expose rotation")
    _need(any(v["rows"] > 1 and v["columns"] > 1
              and 0 < v["advanced"] < v["capacity"]
              and v["advanced"] // v["columns"] <
                  (v["advanced"] + min(v["remaining"], v["depth"])) // v["columns"]
              and (v["advanced"] + min(v["remaining"], v["depth"])) % v["columns"] != 0
              for v in active), "missing row-crossing partial-counter evidence")
    _need(bool(_shift_types(records)), "missing observable train/test shift")


def self_test():
    """Bounded 42-seed diversity audit, counts, rotations and corruption probes."""
    signatures, shapes, geometries, roles, colors = set(), set(), set(), set(), set()
    directions, stops, counts, shifts = set(), set(), set(), set()
    features = []
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = tuple(_key(pair["input"]) for split in ("train", "test") for pair in p[split])
        signatures.add(signature)
        if seed in (0, 17, 41):
            repeated = generate_puzzle(seed)
            _need(signature == tuple(_key(pair["input"]) for split in ("train", "test")
                                     for pair in repeated[split]), "nondeterministic puzzle")
            _need(all(np.array_equal(a["output"], b["output"])
                      for split in p for a, b in zip(p[split], repeated[split])), "nondeterministic labels")
        records = _features(p)
        for split, shape, looms in records:
            shapes.add(shape)
            counts.add(len(looms))
            for v in looms:
                directions.add(v["turns"])
                f, d = v["remaining"], v["depth"]
                stops.add("zero" if f == 0 else "fuel" if f < d else "exact" if f == d else "supply")
                geometries.add((v["rows"], v["columns"], d, f, v["turns"]))
                roles.add((v["background"], v["frame"], v["spent"], v["fuel"]))
                colors.update(v["code"])
                features.append(v)
        shifts.update(_shift_types(records))
    _need(len(signatures) == 42, "42 seeds must produce distinct puzzles")
    _need(len(shapes) >= 60 and len(geometries) >= 140 and len(roles) >= 35, "insufficient structural/palette variation")
    _need(directions == set(range(4)) and stops == {"zero", "fuel", "exact", "supply"}, "missing direction/stopping support")
    _need({1, 2, 3, 4} <= counts and shifts == {"depth", "aspect", "orientation", "count"}, "missing count/shift diversity")
    _need(colors == set(range(10)), "program colors lack palette diversity")
    _need(any(v["advanced"] == 0 for v in features) and any(v["remaining"] == 0 for v in features), "uniform counters missing")
    _need(any(min(v["rows"], v["columns"]) == 1 for v in features), "thin counter missing")
    _need(any(v["rows"] == v["columns"] for v in features), "square counter missing")
    _need(any(len(set(v["code"])) == 1 for v in features), "simple monochrome coat missing")
    _need(any(v["code"][0] in v["code"][2:] for v in features), "nonadjacent color return missing")
    _need(max(v["depth"] for v in features) >= 9, "long code missing")
    # Exact boundary and counter fixtures are deliberately built without the
    # random sampler. These probe the full semantic family, not just its profile.
    def fixture(rows, columns, code, remaining):
        d, h, w = len(code), rows + 2, columns + 2
        block = np.full((h + d + 1, w), 9, dtype=int)
        block[1:d + 1] = np.asarray(code[::-1])[:, None]
        block[d + 1, 0], block[d + 1, -1] = 1, 2
        block[d + 2:-1, 1:-1] = np.array(
            [1] * (rows * columns - remaining) + [2] * remaining).reshape(rows, columns)
        g = np.zeros((h + 2 * d + 1, w + 2 * d), dtype=int)
        g[:len(block), d:d + w] = block
        return g

    for rows, columns, code, remaining in (
            (25, 26, (3,), 650), (25, 26, (3,), 0),
            (1, 1, (3,) * 13, 1), (1, 1, (3,) * 13, 0),
            (2, 3, (3, 3, 5), 5), (2, 3, (3, 4, 4, 5), 2),
            (3, 2, (3, 5, 3, 5), 4)):
        g = fixture(rows, columns, code, remaining)
        vs = _read_scene(g)
        expected = np.zeros_like(g)
        v = vs[0]
        t, b, l, r = v["bounds"]
        expected[t:b + 1, l:r + 1] = _alternate_patch(v)
        for turns in range(4):
            _need(np.array_equal(solve(np.rot90(g, turns)), np.rot90(expected, turns)),
                  "boundary/uniform-counter fixture failed")
    g = fixture(2, 3, (3, 3, 5), 5)
    result = solve(g)
    _need(np.array_equal(result[5:7, 4:7], [[1, 1, 1], [1, 2, 2]]),
          "hand-counted raster fixture failed")
    _need(result[1, 0] == 5 and result[2, 1] == 3 and result[3, 2] == 3,
          "hand-counted shell thickness failed")
    for point, color in (((0, 3), 3), ((6, 6), 1)):
        malformed = g.copy()
        malformed[point] = color
        try:
            solve(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("solver accepted a broken cap or nonprefix counter")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1234 + 10 * a + b, a, b)
            _need(len(p["train"]) == a and len(p["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    for pair in p["train"] + p["test"]:
        before = pair["input"].copy()
        _need(np.array_equal(solve(before), pair["output"]), "solver failed")
        _need(np.array_equal(before, pair["input"]), "solver mutated its input")
        for turns in range(4):
            _need(np.array_equal(solve(np.rot90(before, turns)),
                                 np.rot90(pair["output"], turns)), "rotation equivariance failed")
    bad = {s: [{k: np.array(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted shell")
