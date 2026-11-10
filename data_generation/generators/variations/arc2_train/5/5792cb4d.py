"""Solved puzzles whose single induced grid path reverses its color sequence."""

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SOURCE_ROWS = (
    (
        ("8888888888", "8888888888", "8888888888", "8888888888",
         "8848888888", "8871656188", "8888888288", "8888888888",
         "8888888888", "8888888888"),
        ("8888888888", "8888888888", "8888888888", "8888888888",
         "8828888888", "8816561788", "8888888488", "8888888888",
         "8888888888", "8888888888"),
    ),
    (
        ("8888888888", "8818888888", "8878888888", "8896888888",
         "8884778888", "8888818888", "8888848888", "8888842188",
         "8888888888", "8888888888"),
        ("8888888888", "8818888888", "8828888888", "8844888888",
         "8881778888", "8888848888", "8888868888", "8888897188",
         "8888888888", "8888888888"),
    ),
    (
        ("8888888888", "8888889928", "8888889828", "8866185828",
         "8868485848", "8878155848", "8878888848", "8878888818",
         "8811788118", "8888888888"),
        ("8888888888", "8888884168", "8888881868", "8822985868",
         "8828985878", "8848955878", "8848888878", "8848888818",
         "8811188718", "8888888888"),
    ),
)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _source_keys():
    return {
        tuple(_key(np.array([[int(c) for c in row] for row in rows],
                             dtype=np.int8)) for rows in pair)
        for pair in _SOURCE_ROWS
    }


_SOURCE_KEYS = _source_keys()


def _counts(rng, seed, num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)
    _require(len(order) >= 3, "a path and a background need three colors")
    _require(counts[order[-1]] > grid.size // 2,
             "background must occupy a clear majority")
    return int(colors[order[-1]])


def _trace(grid, background):
    points = {tuple(map(int, point)) for point in np.argwhere(grid != background)}
    _require(len(points) >= 7, "path is too short")
    neighbors = {
        p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRS
            if (p[0] + dr, p[1] + dc) in points]
        for p in points
    }
    endpoints = [p for p in points if len(neighbors[p]) == 1]
    _require(len(endpoints) == 2 and
             all(len(ns) in (1, 2) for ns in neighbors.values()),
             "foreground must be one unbranched path")
    path = [min(endpoints)]
    visited = {path[0]}
    previous = None
    while len(path) < len(points):
        options = [p for p in neighbors[path[-1]] if p != previous]
        _require(len(options) == 1 and options[0] not in visited,
                 "foreground path is disconnected or cyclic")
        previous = path[-1]
        path.append(options[0])
        visited.add(options[0])
    _require(set(path) == points and path[-1] in endpoints,
             "path traversal missed foreground cells")
    return path


def _solve(grid):
    path = _trace(grid, _background(grid))
    values = [int(grid[p]) for p in path]
    result = grid.copy()
    for p, value in zip(path, reversed(values)):
        result[p] = value
    return result


def _visible_prediction(grid):
    """A separate graph walk used by the self-test, from the other endpoint."""
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[int(np.argmax(counts))])
    points = [tuple(map(int, p)) for p in np.argwhere(grid != bg)]
    point_set = set(points)
    adj = {p: [q for q in ((p[0]-1, p[1]), (p[0]+1, p[1]),
                           (p[0], p[1]-1), (p[0], p[1]+1)) if q in point_set]
           for p in points}
    endpoints = sorted((p for p in points if len(adj[p]) == 1), reverse=True)
    _require(len(endpoints) == 2, "visible predictor found no unique endpoints")
    path = [endpoints[0]]
    while path[-1] != endpoints[1]:
        candidates = [p for p in adj[path[-1]]
                      if len(path) < 2 or p != path[-2]]
        _require(len(candidates) == 1 and candidates[0] not in path,
                 "visible predictor found an ambiguous route")
        path.append(candidates[0])
    _require(len(path) == len(points), "visible predictor missed cells")
    output = grid.copy()
    for i, p in enumerate(path):
        output[p] = grid[path[-i-1]]
    return output


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "input and output shape differ")
    bg = _background(inp)
    _require(np.array_equal(inp == bg, out == bg),
             "background or path geometry changed")
    path = _trace(inp, bg)
    _require(np.array_equal(_solve(inp), out), "path colors were not reversed")
    _require(not np.array_equal(inp, out), "unchanged path reversal")
    values = [int(inp[p]) for p in path]
    _, color_counts = np.unique(values, return_counts=True)
    alternating = any(
        len(set(values[i:i+6])) == 2 and
        all(values[i+t] == values[i+t+2] for t in range(4))
        for i in range(len(values)-5))
    repeat_motif = any(
        len(values) >= 3*k and
        sum(values[i] == values[i+k] for i in range(len(values)-k)) /
        (len(values)-k) >= 0.65 and
        sum(a == b for a, b in zip(values, values[1:])) /
        (len(values)-1) < 0.3
        for k in range(2, 5))
    moves = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    bends = sum(a != b for a, b in zip(moves, moves[1:]))
    runs = []
    for move in moves:
        if runs and runs[-1][0] == move:
            runs[-1] = (move, runs[-1][1]+1)
        else:
            runs.append((move, 1))
    rows = [p[0] for p in path]
    cols = [p[1] for p in path]
    bbox = (max(rows)-min(rows)+1, max(cols)-min(cols)+1)
    close = sum(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 2
                for i, a in enumerate(path) for b in path[i+3:])
    return {"background": bg, "length": len(path), "bends": bends,
            "bbox": bbox, "endpoints": (path[0], path[-1]),
            "palette": frozenset(values), "equal_ends": values[0] == values[-1],
            "adjacent_equal": sum(a == b for a, b in zip(values, values[1:])),
            "alternating": alternating, "repeat_motif": repeat_motif,
            "uneven_frequency": int(max(color_counts)-min(color_counts)) >= 3,
            "close": close,
            "run_lengths": tuple(n for _, n in runs),
            "long_horizontal": max((n for (dr, dc), n in runs if dc), default=0),
            "long_vertical": max((n for (dr, dc), n in runs if dr), default=0),
            "interior_ends": all(min(cols) < p[1] < max(cols)
                                 for p in (path[0], path[-1])),
            "edge": min(rows) == 0 or min(cols) == 0 or
                    max(rows) == inp.shape[0]-1 or max(cols) == inp.shape[1]-1,
            "monotone": len({m for m in moves if m[0]}) <= 1 and
                        len({m for m in moves if m[1]}) <= 1}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs train and test")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")),
             "train and test must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq,
             "illegal train/test pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_info = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                         np.issubdtype(grid.dtype, np.integer), "invalid grid type")
                _require(all(1 <= n <= 30 for n in grid.shape) and
                         bool(np.all((grid >= 0) & (grid <= 9))),
                         "invalid grid size or colors")
            pair_key = (_key(pair["input"]), _key(pair["output"]))
            _require(pair_key[0] not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate pair or input")
            _require(pair_key not in _SOURCE_KEYS, "copied source pair")
            seen_inputs.add(pair_key[0])
            seen_pairs.add(pair_key)
            info = _analyze(pair)
            if split == "train":
                train_info.append(info)
    _require(any(len(info["palette"]) >= 3 and info["bends"] >= 2
                 for info in train_info),
             "training does not demonstrate multi-color path reversal")
    _require(len({(a["length"], a["bends"], a["bbox"])
                  for a in train_info}) >= 2,
             "training path shapes do not vary")
    mappings = {}
    for pair in puzzle["train"]:
        inp, out = pair["input"], pair["output"]
        for p in _trace(inp, _background(inp)):
            mappings.setdefault(int(inp[p]), set()).add(int(out[p]))
    _require(any(len(outputs) > 1 for outputs in mappings.values()),
             "training can be explained by a fixed color substitution")


def _random_path(length, rng, max_span, folded):
    """Grow an induced self-avoiding path; check all four-neighbor edges."""
    for attempt in range(180):
        span = min(max_span, max(6, int(np.ceil(np.sqrt(length * 2.15))) +
                                 int(rng.integers(0, 3))))
        start = (int(rng.integers(0, span)), int(rng.integers(0, span)))
        path = [start]
        occupied = {start}
        turn_weight = (float(rng.uniform(0.65, 3.8)) if folded else
                       float(rng.uniform(0.25, 1.5)))
        for _ in range(length-1):
            current = path[-1]
            candidates, weights = [], []
            for dr, dc in _DIRS:
                p = (current[0]+dr, current[1]+dc)
                if not (0 <= p[0] < span and 0 <= p[1] < span) or p in occupied:
                    continue
                if sum((p[0]+ar, p[1]+ac) in occupied for ar, ac in _DIRS) != 1:
                    continue
                turn = len(path) > 1 and (dr, dc) != (
                    current[0]-path[-2][0], current[1]-path[-2][1])
                nearby = sum(abs(p[0]-q[0]) + abs(p[1]-q[1]) == 2
                             for q in occupied)
                weight = (turn_weight if turn else 1.0) * (1 + 0.3*nearby)
                candidates.append(p)
                weights.append(weight)
            if not candidates:
                break
            index = int(rng.choice(len(candidates), p=np.array(weights)/sum(weights)))
            path.append(candidates[index])
            occupied.add(path[-1])
        if len(path) != length:
            continue
        moves = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
        bends = sum(a != b for a, b in zip(moves, moves[1:]))
        if bends < (5 if folded else 2):
            continue
        # Rotation/reflection varies the orientation of long runs independently.
        coords = np.array(path, dtype=np.int16)
        if int(rng.integers(0, 2)):
            coords = coords[:, [1, 0]]
        if int(rng.integers(0, 2)):
            coords[:, 0] *= -1
        if int(rng.integers(0, 2)):
            coords[:, 1] *= -1
        coords -= coords.min(axis=0)
        return [tuple(map(int, p)) for p in coords]
    raise ValueError("could not grow a simple path")


def _run_path(length, bends, rng, max_span):
    """Compose unequal orthogonal runs, including offset and nested returns."""
    for _ in range(90):
        cuts = sorted(int(c) for c in rng.choice(
            np.arange(1, length-1), size=bends, replace=False))
        run_lengths = np.diff([0] + cuts + [length-1])
        axis = int(rng.integers(0, 2))
        path = [(0, 0)]
        occupied = {path[0]}
        for run_index, run_length in enumerate(run_lengths):
            sign = 1 if int(rng.integers(0, 2)) else -1
            dr, dc = (sign, 0) if (axis+run_index) % 2 == 0 else (0, sign)
            for _ in range(int(run_length)):
                p = (path[-1][0]+dr, path[-1][1]+dc)
                if p in occupied or sum((p[0]+ar, p[1]+ac) in occupied
                                        for ar, ac in _DIRS) != 1:
                    break
                path.append(p)
                occupied.add(p)
            else:
                continue
            break
        if len(path) != length:
            continue
        coords = np.array(path, dtype=np.int16)
        coords -= coords.min(axis=0)
        if int(coords[:, 0].max()) >= max_span or int(coords[:, 1].max()) >= max_span:
            continue
        if int(rng.integers(0, 2)):
            coords = coords[:, [1, 0]]
        return [tuple(map(int, p)) for p in coords]
    return _random_path(length, rng, max_span, True)


def _palette(rng, background, length, minimum, vocabulary=None, full=False):
    available = [c for c in (vocabulary if vocabulary is not None else range(10))
                 if c != background]
    size = min(len(available), length, int(rng.integers(minimum, 9)))
    if full:
        size = len(available)
    return [int(c) for c in rng.choice(available, size=size, replace=False)]


def _sequence(length, colors, rng, mode, equal_ends):
    colors = list(colors)
    if len(set(colors)) < 2 or len(set(colors)) != len(colors):
        raise ValueError("payload needs at least two distinct colors")
    for _ in range(60):
        if mode == "scatter":
            values = colors + [int(c) for c in rng.choice(colors, size=length-len(colors))]
            rng.shuffle(values)
        elif mode == "blocks":
            order = [int(c) for c in rng.permutation(colors)]
            block_count = min(length, len(colors) + int(rng.integers(0, max(2, length//5))))
            while len(order) < block_count:
                candidate = int(rng.choice(colors))
                if candidate != order[-1]:
                    order.append(candidate)
            sizes = [1] * block_count
            for _ in range(length-block_count):
                sizes[int(rng.integers(0, block_count))] += 1
            values = [c for c, n in zip(order, sizes) for _ in range(n)]
        else:
            motif = [int(c) for c in rng.permutation(colors)]
            if mode == "alternating" and len(motif) == 2:
                motif = motif[:2]
            elif len(motif) > 4:
                mode = "motif"
            values = [motif[i % len(motif)] for i in range(length)]
            if mode == "motif" and length > len(motif)+3:
                values[int(rng.integers(len(motif), length-1))] = int(rng.choice(colors))
        if equal_ends:
            values[-1] = values[0]
        if len(set(values)) == len(colors) and values != values[::-1] and \
           sum(a != b for a, b in zip(values, values[::-1])) >= 3:
            return np.asarray(values, dtype=np.int8)
    # A scattered fallback still uses all selected colors and changes visibly.
    values = colors + [colors[i % len(colors)] for i in range(length-len(colors))]
    if equal_ends:
        values[-1] = values[0]
    if values == values[::-1] or len(set(values)) < len(colors):
        values = colors + [colors[(i+1) % len(colors)] for i in range(length-len(colors))]
    if values == values[::-1] or len(set(values)) < len(colors):
        raise ValueError("could not construct asymmetric payload")
    return np.asarray(values, dtype=np.int8)


def _embed(path, values, background, rng, shared_canvas):
    height = max(p[0] for p in path)+1
    width = max(p[1] for p in path)+1
    if shared_canvas:
        if height > 10 or width > 10:
            raise ValueError("path does not fit shared canvas")
        top = int(rng.integers(0, 11-height))
        left = int(rng.integers(0, 11-width))
        canvas = (10, 10)
    else:
        pads = [int(rng.integers(0, 5)) for _ in range(4)]
        # Keep the modal background unambiguous while preserving zero margins.
        while (height+pads[0]+pads[1]) * (width+pads[2]+pads[3]) <= 2*len(path):
            side = int(rng.integers(0, 4))
            pads[side] += 1
        top, left = pads[0], pads[2]
        canvas = (height+pads[0]+pads[1], width+pads[2]+pads[3])
    grid = np.full(canvas, background, dtype=np.int8)
    for (r, c), value in zip(path, values):
        grid[r+top, c+left] = value
    return grid


def _pair(rng, length, folded, shared_canvas, background, colors, mode,
          equal_ends, target_bends=None):
    path = (_run_path(length, target_bends, rng,
                      10 if shared_canvas else 13) if target_bends is not None
            else _random_path(length, rng, 8 if shared_canvas else 13, folded))
    values = _sequence(length, colors, rng, mode, equal_ends)
    inp = _embed(path, values, background, rng, shared_canvas)
    return {"input": inp, "output": _solve(inp)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, nq = _counts(rng, seed, num_train, num_test)
    # The only learned convention is path-order reversal. Literal colors have
    # no mapping or directional meaning. Some puzzles retain source-like color
    # progression; others demonstrate local background and payload assignment.
    regime = str(rng.choice(("classic", "balanced", "reverse"), p=(0.35, 0.4, 0.25)))
    color_regime = str(rng.choice(("progression", "stable", "local"), p=(0.25, 0.25, 0.5)))
    shared_canvas = bool(rng.random() < 0.28)
    global_bg = int(rng.integers(0, 10))
    global_colors = [c for c in range(10) if c != global_bg]
    rng.shuffle(global_colors)
    vocabulary = global_colors[:int(rng.integers(4, 8))]
    reserved_color = int(rng.integers(0, 10))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_bg = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for attempt in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
                if regime == "classic":
                    low, high = ((8, 19) if split == "train" else (24, 39))
                elif regime == "reverse":
                    low, high = ((8, 34) if split == "train" else (8, 22))
                    if split == "train" and index == 1:
                        low = 23
                else:
                    low, high = ((8, 34) if split == "train" else (9, 38))
                if shared_canvas:
                    high = min(high, 32)
                length = int(pair_rng.integers(low, high+1))
                target_bends = None
                if regime == "classic" and split == "test" and not shared_canvas \
                   and pair_rng.random() < 0.48:
                    if pair_rng.random() < 0.2:
                        length, target_bends = 28, 8
                    else:
                        length = int(pair_rng.integers(24, 34))
                        target_bends = int(pair_rng.integers(6, 11))
                folded_probability = (0.3 if split == "train" else 0.62)
                if regime == "classic":
                    folded_probability = 0.18 if split == "train" else 0.9
                if regime == "reverse" and split == "train" and index == 1:
                    folded_probability = 0.95
                folded = bool(pair_rng.random() < folded_probability)
                if color_regime == "local":
                    choices = list(range(10))
                    if split == "train":
                        choices.remove(reserved_color)
                    if split == "train" and index == 1:
                        choices.remove(first_bg)
                    bg = int(pair_rng.choice(choices))
                    if split == "train":
                        local_colors = [c for c in range(10)
                                        if c != bg and c != reserved_color]
                    else:
                        local_colors = [c for c in range(10) if c != bg]
                    palette = _palette(pair_rng, bg, length,
                                       3 if split == "train" and index == 0 else 2,
                                       local_colors)
                    if split == "test" and pair_rng.random() < 0.55:
                        if reserved_color != bg and reserved_color not in palette:
                            palette[0] = reserved_color
                else:
                    bg = global_bg
                    if color_regime == "progression":
                        if split == "test":
                            palette = list(vocabulary)
                        elif index == 0:
                            palette = list(vocabulary[:-1])
                        elif index == 1:
                            palette = list(vocabulary[1:])
                        else:
                            palette = _palette(pair_rng, bg, length, 2, vocabulary[:-1])
                    else:
                        palette = _palette(pair_rng, bg, length,
                                           3 if split == "train" and index == 0 else 2,
                                           vocabulary)
                if len(palette) > length:
                    continue
                mode = str(pair_rng.choice(("scatter", "blocks", "alternating", "motif"),
                                            p=(0.3, 0.3, 0.18, 0.22)))
                if color_regime == "progression" and len(palette) > 4 and mode == "alternating":
                    mode = "blocks"
                equal_ends = bool(pair_rng.random() < 0.22)
                try:
                    pair = _pair(pair_rng, length, folded, shared_canvas,
                                 bg, palette, mode, equal_ends, target_bends)
                except ValueError:
                    continue
                key = (_key(pair["input"]), _key(pair["output"]))
                if key in _SOURCE_KEYS or key[0] in seen:
                    continue
                if split == "train" and index == 0:
                    info = _analyze(pair)
                    if len(info["palette"]) < 3 or info["bends"] < 2:
                        continue
                    local_maps = {}
                    for p in _trace(pair["input"], bg):
                        local_maps.setdefault(int(pair["input"][p]), set()).add(
                            int(pair["output"][p]))
                    if not any(len(destinations) > 1
                               for destinations in local_maps.values()):
                        continue
                if split == "train" and index == 1:
                    first = _analyze(puzzle["train"][0])
                    other = _analyze(pair)
                    if (first["length"], first["bends"], first["bbox"]) == \
                       (other["length"], other["bends"], other["bbox"]):
                        continue
                puzzle[split].append(pair)
                seen.add(key[0])
                if split == "train" and index == 0:
                    first_bg = bg
                break
            else:
                raise ValueError("could not generate a distinct valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    source_like = _pair(np.random.default_rng(0), 28, True, True, 0,
                        (1, 2, 3, 4, 5, 6, 7), "blocks", False, 8)
    source_like_info = _analyze(source_like)
    _require(source_like["input"].shape == (10, 10) and
             source_like_info["length"] == 28 and
             source_like_info["bends"] == 8,
             "a 28-cell, 8-bend 10x10 path is not constructible")
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3),
                        (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(603 + (train or 0) + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        _require(train is None or len(puzzle["train"]) == train,
                 "explicit train count ignored")
        _require(test is None or len(puzzle["test"]) == test,
                 "explicit test count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (10, 10)):
        try:
            generate_puzzle(9, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal count request accepted")
    a = generate_puzzle(303, 8, 3)
    b = generate_puzzle(303, 8, 3)
    _require(all(np.array_equal(pa[k], pb[k])
                 for split in ("train", "test")
                 for pa, pb in zip(a[split], b[split])
                 for k in ("input", "output")), "seed is not deterministic")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = {"train_counts": set(), "test_counts": set(), "shapes": set(),
               "bgs": set(), "lengths": set(), "bends": set(),
               "endpoint_relations": set(), "edge": set(),
               "equal_ends": set(), "two_colors": set(), "monotone": set(),
               "close": set(), "blocky": set(), "shared_10": set(),
               "variable_background": set(), "test_shorter": set(),
               "novel_test_color": set(), "progression": set(),
               "long_horizontal": set(), "long_vertical": set(),
               "unequal_runs": set(), "interior_ends": set(),
               "folded_train": set(), "modest_test": set(),
               "variable_shape": set(), "variable_palette": set(),
               "combined_edge_fold_block": set(),
               "alternating": set(), "repeat_motif": set(),
               "uneven_frequency": set(), "mixed_payload": set()}
    fingerprints = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        metrics["train_counts"].add(len(puzzle["train"]))
        metrics["test_counts"].add(len(puzzle["test"]))
        fingerprints.add(tuple(tuple((_key(p["input"]), _key(p["output"]))
                                      for p in puzzle[s]) for s in ("train", "test")))
        infos = {s: [_analyze(p) for p in puzzle[s]] for s in ("train", "test")}
        all_pairs = puzzle["train"] + puzzle["test"]
        all_info = infos["train"] + infos["test"]
        metrics["variable_background"].add(len({i["background"] for i in all_info}) > 1)
        metrics["shared_10"].add(all(p["input"].shape == (10, 10) for p in all_pairs))
        metrics["variable_shape"].add(len({p["input"].shape for p in all_pairs}) > 1)
        metrics["variable_palette"].add(len({len(i["palette"]) for i in all_info}) > 1)
        high_adjacency = any(i["adjacent_equal"] >= (i["length"]-1)*0.4
                             for i in all_info)
        low_adjacency = any(i["adjacent_equal"] <= (i["length"]-1)*0.1
                            for i in all_info)
        metrics["mixed_payload"].add(high_adjacency and low_adjacency)
        metrics["folded_train"].add(any(i["bends"] >= 8 for i in infos["train"]))
        metrics["modest_test"].add(any(i["length"] <= 16 for i in infos["test"]))
        train_colors = set.union(*(set(i["palette"]) for i in infos["train"]))
        all_train_colors = set.union(*(set(map(int, np.unique(p["input"])))
                                       for p in puzzle["train"]))
        metrics["novel_test_color"].add(any(set(i["palette"]) - all_train_colors
                                            for i in infos["test"]))
        metrics["progression"].add(
            len({i["background"] for i in all_info}) == 1 and
            len(train_colors) == max(len(i["palette"]) for i in infos["train"]) + 1 and
            all(set(i["palette"]) == train_colors for i in infos["test"]))
        metrics["test_shorter"].add(min(i["length"] for i in infos["test"]) <=
                                    max(i["length"] for i in infos["train"]))
        for pair, info in zip(all_pairs, all_info):
            _require(np.array_equal(pair["output"],
                                    _visible_prediction(pair["input"])),
                     "wrong path-order reversal")
            metrics["shapes"].add(pair["input"].shape)
            metrics["bgs"].add(info["background"])
            metrics["lengths"].add(info["length"])
            metrics["bends"].add(info["bends"])
            a, b = info["endpoints"]
            relation = "row" if a[0] == b[0] else "column" if a[1] == b[1] else "offset"
            metrics["endpoint_relations"].add(relation)
            metrics["edge"].add(info["edge"])
            metrics["equal_ends"].add(info["equal_ends"])
            metrics["two_colors"].add(len(info["palette"]) == 2)
            metrics["monotone"].add(info["monotone"])
            metrics["close"].add(info["close"] > 0)
            metrics["blocky"].add(info["adjacent_equal"] >= (info["length"]-1)//3)
            metrics["long_horizontal"].add(info["long_horizontal"] >= 4)
            metrics["long_vertical"].add(info["long_vertical"] >= 4)
            metrics["unequal_runs"].add(len(set(info["run_lengths"])) >= 3)
            metrics["interior_ends"].add(info["interior_ends"])
            metrics["combined_edge_fold_block"].add(
                info["edge"] and info["bends"] >= 8 and
                info["adjacent_equal"] >= (info["length"]-1)//3)
            for name in ("alternating", "repeat_motif", "uneven_frequency"):
                metrics[name].add(info[name])
    _require(len(fingerprints) >= 40, "cross-seed puzzle diversity is low")
    _require(len(metrics["train_counts"]) >= 4 and len(metrics["test_counts"]) >= 3,
             "automatic counts lack diversity")
    _require(len(metrics["shapes"]) >= 20 and len(metrics["bgs"]) >= 6 and
             len(metrics["lengths"]) >= 25 and len(metrics["bends"]) >= 10,
             "grid and path diversity is low")
    for name in ("edge", "equal_ends", "two_colors", "monotone", "close",
                 "blocky", "shared_10", "variable_background", "test_shorter",
                 "novel_test_color", "progression", "long_horizontal",
                 "long_vertical", "unequal_runs", "interior_ends",
                 "folded_train", "modest_test", "variable_shape",
                 "combined_edge_fold_block", "alternating", "repeat_motif",
                 "uneven_frequency", "mixed_payload"):
        _require(len(metrics[name]) == 2, f"missing diversity for {name}")
    _require(metrics["variable_palette"] == {True},
             "some puzzles do not vary payload palette size")
    _require(metrics["endpoint_relations"] == {"row", "column", "offset"},
             "endpoint arrangement lacks diversity")
