"""Generate center-marked copies of one compact multicolor motif."""

import numpy as np


_SOURCE_ROWS = (
    (("0000600", "0000110", "0000222", "0000000", "0000000", "0500000", "0000000", "0000000"),
     ("0000600", "0000110", "0000222", "0000000", "6000000", "1100000", "2220000", "0000000")),
    (("020000000", "221000000", "013000000", "000000000", "000000000", "000005000", "000000000", "000000000", "000000000"),
     ("020000000", "221000000", "013000000", "000000000", "000002000", "000022100", "000001300", "000000000", "000000000")),
    (("00000000", "00000000", "00005000", "00000000", "00000000", "00000000", "02200000", "00310000", "03310000", "00000000"),
     ("00000000", "00022000", "00003100", "00033100", "00000000", "00000000", "02200000", "00310000", "03310000", "00000000")),
    (("00000000000", "00000000000", "00022000000", "00110000000", "00033000000", "00000000000", "00000000000", "00000000000", "00000050000", "00000000000"),
     ("00000000000", "00000000000", "00022000000", "00110000000", "00033000000", "00000000000", "00000000000", "00000000000", "00000022000", "00000110000", "00000033000")),
)
_SIZES = ((3, 3), (3, 5), (5, 5))
_DIRECTIONS = ("up", "down", "left", "right", "diagonal")
_EDGES = ("interior", "top", "bottom", "left", "right")


def _key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _source_key(rows):
    return (len(rows), len(rows[0])), tuple(int(c) for row in rows for c in row)


_SOURCE_KEYS = frozenset((_source_key(a), _source_key(b)) for a, b in _SOURCE_ROWS)


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} out of range")
    return value


def _counts(rng, seed, num_train, num_test):
    n = _count("num_train", num_train, 2, 10)
    m = _count("num_test", num_test, 1, 4)
    if n is None and m is None:
        m = 1 + int(seed) % 4
        n = m + 1 + (int(seed) // 4) % (10 - m)
    elif n is None:
        n = int(rng.integers(m + 1, 11))
    elif m is None:
        m = int(rng.integers(1, min(4, n - 1) + 1))
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    return n, m


def _connected(mask):
    cells = {tuple(int(v) for v in p) for p in np.argwhere(mask)}
    if not cells:
        return False
    reached = {next(iter(cells))}
    frontier = list(reached)
    while frontier:
        r, c = frontier.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in cells and p not in reached:
                reached.add(p)
                frontier.append(p)
    return reached == cells


def _sample_mask(rng, size, style):
    h, w = size
    if style == "full":
        return np.ones(size, dtype=bool)
    for _ in range(300):
        if style == "bar":
            mask = rng.random(size) < 0.42
            mask[int(rng.integers(h)), :] = True
            mask[:, int(rng.integers(w))] = True
        elif style == "corner":
            mask = rng.random(size) < 0.45
            row = 0 if int(rng.integers(2)) == 0 else h - 1
            col = 0 if int(rng.integers(2)) == 0 else w - 1
            mask[row, :] = True
            mask[:, col] = True
        else:
            mask = rng.random(size) < float(rng.uniform(0.56, 0.81))
        mask[h // 2, w // 2] = True
        filled = int(mask.sum())
        if not max(5, (h * w) // 2) <= filled <= h * w:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if style != "irregular" and not _connected(mask):
            continue
        return mask
    return np.ones(size, dtype=bool)


def _motif(rng, size, background, marker, mask=None, exact=None):
    if exact is not None:
        return exact.copy()
    if mask is None:
        style = ("bar", "corner", "cluster", "irregular", "full")[int(rng.choice(5, p=[.22, .20, .29, .19, .10]))]
        mask = _sample_mask(rng, size, style)
    else:
        mask = mask.copy()
    available = [c for c in range(10) if c not in (background, marker)]
    palette_size = int(rng.integers(2, min(5, int(mask.sum()) + 1)))
    palette = rng.choice(available, size=palette_size, replace=False)
    motif = np.full(size, background, dtype=np.int8)
    coords = np.argwhere(mask)
    rng.shuffle(coords)
    color_style = int(rng.integers(3))
    if color_style == 0:
        labels = rng.integers(0, palette_size, size=len(coords))
    elif color_style == 1:
        labels = np.asarray([int(r) % palette_size for r, _ in coords])
    else:
        labels = np.asarray([int(c) % palette_size for _, c in coords])
    labels[:palette_size] = rng.permutation(palette_size)
    for (r, c), label in zip(coords, labels):
        motif[int(r), int(c)] = int(palette[int(label)])
    return motif


def _overlap(a, b, size):
    h, w = size
    return not (a[0] + h <= b[0] or b[0] + h <= a[0] or a[1] + w <= b[1] or b[1] + w <= a[1])


def _direction(source, destination):
    dr = destination[0] - source[0]
    dc = destination[1] - source[1]
    if dc == 0:
        return "down" if dr > 0 else "up"
    if dr == 0:
        return "right" if dc > 0 else "left"
    return "diagonal"


def _edge(position, grid_size, motif_size):
    r, c = position
    h, w = motif_size
    height, width = grid_size
    sides = []
    if r == 0:
        sides.append("top")
    if r + h == height:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c + w == width:
        sides.append("right")
    return tuple(sides) if sides else ("interior",)


def _dimensions(rng, size, direction):
    h, w = size
    short_min = 8 if size == (3, 3) else 10
    aspect = ("square", "tall", "wide")[int(rng.integers(3))]
    short = int(rng.integers(short_min, 16))
    if aspect == "square":
        height = short
        width = short + int(rng.integers(-1, 2))
    else:
        long = min(23, short + int(rng.integers(3, 9)))
        height, width = (long, short) if aspect == "tall" else (short, long)
    if direction in ("left", "right"):
        width = max(width, 2 * w)
    if direction in ("up", "down"):
        height = max(height, 2 * h)
    return height, width


def _layout(rng, grid_size, size, direction, requested_edge):
    height, width = grid_size
    h, w = size
    destinations = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
    preferred = [p for p in destinations if requested_edge in _edge(p, grid_size, size)]
    fallback = [p for p in destinations if p not in preferred]
    rng.shuffle(preferred)
    rng.shuffle(fallback)
    sources = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
    for destination in preferred + fallback:
        candidates = [p for p in sources if not _overlap(p, destination, size) and _direction(p, destination) == direction]
        if candidates:
            return candidates[int(rng.integers(len(candidates)))], destination
    raise RuntimeError("No source and destination layout fits")


def _pair(rng, size, background, marker, direction, edge, reuse_mask=None, reuse_exact=None):
    grid_size = _dimensions(rng, size, direction)
    source, destination = _layout(rng, grid_size, size, direction, edge)
    motif = _motif(rng, size, background, marker, reuse_mask, reuse_exact)
    h, w = size
    inp = np.full(grid_size, background, dtype=np.int8)
    inp[source[0]:source[0] + h, source[1]:source[1] + w] = motif
    center = (destination[0] + h // 2, destination[1] + w // 2)
    inp[center] = marker
    out = inp.copy()
    out[destination[0]:destination[0] + h, destination[1]:destination[1] + w] = motif
    return {"input": inp, "output": out}


def _analyze(inp, out):
    _check(inp.shape == out.shape, "Dimensions changed")
    values, counts = np.unique(inp, return_counts=True)
    highest = counts.max()
    _check(int(np.count_nonzero(counts == highest)) == 1, "Background is not uniquely modal")
    background = int(values[int(np.argmax(counts))])
    candidates = []
    for color, count in zip(values, counts):
        if int(color) == background or int(count) != 1:
            continue
        marker = int(color)
        position = np.argwhere(inp == marker)[0]
        mr, mc = (int(position[0]), int(position[1]))
        remaining = inp != background
        remaining[mr, mc] = False
        points = np.argwhere(remaining)
        if len(points) < 5:
            continue
        top, left = points.min(axis=0)
        bottom, right = points.max(axis=0)
        source = (int(top), int(left))
        size = (int(bottom - top + 1), int(right - left + 1))
        if size not in _SIZES:
            continue
        h, w = size
        destination = (mr - h // 2, mc - w // 2)
        if destination[0] < 0 or destination[1] < 0 or destination[0] + h > inp.shape[0] or destination[1] + w > inp.shape[1]:
            continue
        if _overlap(source, destination, size):
            continue
        patch = inp[destination[0]:destination[0] + h, destination[1]:destination[1] + w].copy()
        patch[h // 2, w // 2] = background
        if np.any(patch != background):
            continue
        motif = inp[source[0]:source[0] + h, source[1]:source[1] + w]
        if motif[h // 2, w // 2] == background or np.any(motif == marker):
            continue
        if not ((motif[0] != background).any() and (motif[-1] != background).any() and (motif[:, 0] != background).any() and (motif[:, -1] != background).any()):
            continue
        payload = set(int(v) for v in np.unique(motif)) - {background}
        if not 2 <= len(payload) <= 4:
            continue
        expected = inp.copy()
        expected[destination[0]:destination[0] + h, destination[1]:destination[1] + w] = motif
        candidates.append({
            "background": background, "marker": marker, "size": size,
            "grid_size": inp.shape, "source": source, "destination": destination,
            "direction": _direction(source, destination),
            "edge": _edge(destination, inp.shape, size),
            "mask": tuple(bool(v) for v in (motif != background).flat),
            "motif": motif.copy(), "payload": payload,
            "expected": expected,
        })
    _check(len(candidates) == 1, "Input has no unique centered-copy interpretation")
    _check(np.array_equal(candidates[0].pop("expected"), out), "Output is not the exact centered copy")
    _check(not np.array_equal(inp, out), "Input equals output")
    return candidates[0]


def _validate(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Puzzle keys")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "Split lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= n <= 10 and 1 <= m <= 4 and n > m, "Pair counts")
    infos = {"train": [], "test": []}
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Pair keys")
            for grid in pair.values():
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "Grid type or rank")
                _check(np.issubdtype(grid.dtype, np.integer), "Grid dtype")
                _check(all(1 <= d <= 30 for d in grid.shape), "Grid size")
                _check(bool(np.all((grid >= 0) & (grid <= 9))), "Grid color")
            inp, out = pair["input"], pair["output"]
            a, b = _key(inp), _key(out)
            _check(a not in inputs and (a, b) not in pairs, "Duplicate pair")
            _check((a, b) not in _SOURCE_KEYS, "Original source pair repeated")
            inputs.add(a)
            pairs.add((a, b))
            infos[split].append(_analyze(inp, out))
    train, test = infos["train"], infos["test"]
    _check(len({x["direction"] for x in train}) >= 2, "Training directions do not vary")
    _check(any(x["direction"] in ("left", "right") for x in train), "No lateral training example")
    _check(any(x["direction"] in ("up", "down") for x in train), "No vertical training example")
    train_sizes = {x["size"] for x in train}
    _check(all(x["size"] in train_sizes for x in test), "Test motif size unseen in training")
    train_roles = {(x["background"], x["marker"]) for x in train}
    if len(train_roles) == 1:
        _check(all((x["background"], x["marker"]) in train_roles for x in test), "Undemonstrated role change")
    else:
        _check(len({x["background"] for x in train}) > 1 and len({x["marker"] for x in train}) > 1, "Both variable roles must be demonstrated")
    _check(len({(x["grid_size"], x["source"], x["destination"]) for x in train + test}) == n + m, "Repeated layout")
    return infos


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"Malformed puzzle: {exc}") from exc


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n, m = _counts(rng, seed, num_train, num_test)
    order = [int(v) for v in rng.permutation(10)]
    variable_roles = bool(rng.random() < .48)
    repeat = bool(rng.random() < .42)
    size_mode = int(rng.choice(5, p=[.54, .16, .11, .10, .09]))
    size_choices = ((3, 3),) if size_mode == 0 else (
        ((3, 3), (3, 5)) if size_mode == 1 else
        ((3, 3), (5, 5)) if size_mode == 2 else
        (((3, 5),) if size_mode == 3 else ((5, 5),))
    )
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=n + m, dtype=np.int64)
    directions = ["up" if int(rng.integers(2)) == 0 else "down",
                  "left" if int(rng.integers(2)) == 0 else "right"]
    rng.shuffle(directions)
    directions += [str(rng.choice(_DIRECTIONS)) for _ in range(n + m - 2)]
    edge_choices = list(_EDGES)
    rng.shuffle(edge_choices)
    puzzle = {"train": [], "test": []}
    seen = set()
    used_layouts = set()
    first_mask = None
    first_motif = None
    for i in range(n + m):
        split = "train" if i < n else "test"
        local = np.random.default_rng(int(pair_seeds[i]))
        if len(size_choices) == 2:
            size = size_choices[i] if i < 2 else size_choices[int(local.integers(2))]
        else:
            size = size_choices[0]
        if variable_roles:
            if i == 0:
                background, marker = order[0], order[1]
            elif i == 1:
                background, marker = order[1], order[0]
            else:
                colors = local.choice(10, size=2, replace=False)
                background, marker = int(colors[0]), int(colors[1])
        else:
            background, marker = order[0], order[1]
        direction = directions[i]
        if i < 2:
            edge = edge_choices[i]
        else:
            edge = str(local.choice(_EDGES, p=[.28, .18, .18, .18, .18] if split == "train" else [.17, .2075, .2075, .2075, .2075]))
        if direction == "up" and edge == "bottom":
            edge = "top"
        elif direction == "down" and edge == "top":
            edge = "bottom"
        elif direction == "left" and edge == "right":
            edge = "left"
        elif direction == "right" and edge == "left":
            edge = "right"
        reuse_mask = first_mask if repeat and i == 1 and size == size_choices[0] else None
        reuse_exact = first_motif if reuse_mask is not None and not variable_roles else None
        for _ in range(12):
            pair = _pair(local, size, background, marker, direction, edge, reuse_mask, reuse_exact)
            key = _key(pair["input"])
            if key in seen or (_key(pair["input"]), _key(pair["output"])) in _SOURCE_KEYS:
                continue
            info = _analyze(pair["input"], pair["output"])
            layout = (info["grid_size"], info["source"], info["destination"])
            if layout in used_layouts:
                continue
            seen.add(key)
            used_layouts.add(layout)
            puzzle[split].append(pair)
            if i == 0:
                first_mask = (info["motif"] != background)
                first_motif = info["motif"].copy()
            break
        else:
            raise RuntimeError("Unable to construct unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split])) for split in ("train", "test"))


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(12345, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError for {kwargs}")


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    shapes, layouts, masks, roles = set(), set(), set(), set()
    split_edges = {"train": set(), "test": set()}
    split_aspects = {"train": set(), "test": set()}
    split_directions = {"train": set(), "test": set()}
    mixed_roles = repeated_masks = full_windows = large_windows = combined = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        info = _validate(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_signature(puzzle))
        train_roles = {(x["background"], x["marker"]) for x in info["train"]}
        mixed_roles += len(train_roles) > 1
        all_info = info["train"] + info["test"]
        repeated_masks += len({(x["size"], x["mask"]) for x in all_info}) < len(all_info)
        for split in ("train", "test"):
            for x in info[split]:
                h, w = x["grid_size"]
                aspect = "square" if abs(h - w) <= 1 else ("tall" if h > w else "wide")
                split_aspects[split].add(aspect)
                split_directions[split].add(x["direction"])
                split_edges[split].update(x["edge"])
                shapes.add(x["grid_size"])
                layouts.add((x["grid_size"], x["source"], x["destination"]))
                masks.add((x["size"], x["mask"]))
                roles.add((x["background"], x["marker"]))
                full_windows += all(x["mask"])
                large_windows += x["size"] != (3, 3)
                combined += len(train_roles) > 1 and x["direction"] in ("left", "right") and "interior" not in x["edge"]
                _check(x["direction"] == _direction(x["source"], x["destination"]), "Direction analysis")
        _check(all(x["size"] in {y["size"] for y in info["train"]} for x in info["test"]), "Training size coverage")
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "Automatic count diversity")
    _check(len(signatures) == 42 and len(shapes) >= 35 and len(layouts) >= 180 and len(masks) >= 50, "Structural diversity")
    _check(len(roles) >= 20 and mixed_roles >= 8 and repeated_masks >= 8, "Role or repetition diversity")
    _check(full_windows >= 5 and large_windows >= 20 and combined >= 8, "Motif or combined diversity")
    for split in ("train", "test"):
        _check(split_edges[split] == set(_EDGES), f"{split} edge diversity")
        _check(split_aspects[split] == {"square", "tall", "wide"}, f"{split} aspect diversity")
        _check(split_directions[split] == set(_DIRECTIONS), f"{split} direction diversity")
    a = generate_puzzle(314159, 7, 3)
    _check(_signature(a) == _signature(generate_puzzle(314159, 7, 3)), "Nondeterminism")
    for seed, n, m in ((5001, 2, 1), (5002, 10, 1), (5003, 10, 4), (5004, 8, 3)):
        puzzle = generate_puzzle(seed, n, m)
        _check((len(puzzle["train"]), len(puzzle["test"])) == (n, m), "Explicit counts")
    for seed, n, m in ((6001, None, 1), (6002, None, 4), (6003, 2, None), (6004, 10, None)):
        puzzle = generate_puzzle(seed, n, m)
        _check(n is None or len(puzzle["train"]) == n, "Partial train count")
        _check(m is None or len(puzzle["test"]) == m, "Partial test count")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        _expect_value_error(**kwargs)
