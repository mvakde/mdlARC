"""Competing maze blooms: component budgets determine a substituted stencil.

Only standard-library modules and NumPy are used. The solver reads every role
from the input. Generation styles and teaching schedules never affect decoding.
"""
from collections import deque
import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_INF = 999


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((a >= 0) & (a <= 9)), "colors must be 0..9")
    return a.astype(int, copy=False)


def _components(a, background):
    seen = np.zeros(a.shape, bool)
    result = []
    for r, c in np.ndindex(a.shape):
        if seen[r, c] or a[r, c] == background:
            continue
        color = int(a[r, c])
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _NEIGHBORS:
                yy, xx = y + dy, x + dx
                if (0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]
                        and not seen[yy, xx] and a[yy, xx] == color):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        rs, cs = zip(*cells)
        result.append(dict(color=color, cells=cells,
                           box=(min(rs), min(cs), max(rs) + 1, max(cs) + 1)))
    return result


def _contains(outer, inner):
    a, b, c, d = outer
    y, x, z, w = inner
    return a <= y and b <= x and z <= c and w <= d


def _distance(mask, source):
    dist = np.full(mask.shape, _INF, int)
    dist[source] = 0
    todo = deque([source])
    while todo:
        r, c = todo.popleft()
        for dr, dc in _NEIGHBORS:
            y, x = r + dr, c + dc
            if (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
                    and mask[y, x] and dist[y, x] == _INF):
                dist[y, x] = dist[r, c] + 1
                todo.append((y, x))
    return dist


def _owners(distances, counts, mask, mode="normal"):
    d = np.asarray(distances)
    limits = np.asarray(counts)[:, None, None]
    if mode == "unlimited":
        candidates = d
    elif mode == "nearest_first":
        candidates = d
    else:
        candidates = np.where(d < limits, d, _INF)
    best = candidates.min(axis=0)
    unique = (candidates == best).sum(axis=0) == 1
    owner = np.where(mask & unique & (best < _INF), candidates.argmin(axis=0), -1)
    if mode == "nearest_first":
        for i, n in enumerate(counts):
            owner[(owner == i) & (d[i] >= n)] = -1
    ties = mask & (best < _INF) & ~unique
    return owner, ties


def _decode(input_grid):
    """Identify maximal visible frames; exclude their entire support from counts."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(all(int(a[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))),
             "all four corners must show the background")
    candidates = {}
    for part in _components(a, bg):
        r, c, y, x = part["box"]
        if min(y - r, x - c) < 4:
            continue
        color = part["color"]
        b = a[r:y, c:x]
        if (np.all(b[0] == color) and np.all(b[-1] == color)
                and np.all(b[:, 0] == color) and np.all(b[:, -1] == color)):
            candidates[(r, c, y, x)] = color
    # A tile pattern can contain a smaller ring; its outer frame defines support.
    frames = [(box, color) for box, color in candidates.items()
              if not any(box != other and _contains(other, box) for other in candidates)]
    maps = [(box, color) for box, color in frames
            if min(box[2] - box[0], box[3] - box[1]) >= 7]
    _require(len(maps) == 1, "one large framed map required")
    map_box, divider = maps[0]
    templates = sorted((box, color) for box, color in frames if box != map_box)
    _require(2 <= len(templates) <= 3, "two or three tile frames required")
    colors = [color for _, color in templates]
    _require(len(set(colors)) == len(colors) and divider not in colors,
             "distinct local category and map-frame colors")
    tiles, size = [], None
    outside = a.copy()
    occupied = np.zeros(a.shape, bool)
    for box, color in frames:
        r, c, y, x = box
        _require(r > 0 and c > 0 and y < a.shape[0] and x < a.shape[1],
                 "frames need background margins")
        _require(not occupied[r:y, c:x].any(), "frames overlap or touch")
        occupied[r - 1:y + 1, c - 1:x + 1] = True
        outside[r:y, c:x] = bg
    for box, color in templates:
        r, c, y, x = box
        _require(y - r == x - c and 4 <= y - r <= 6, "tile-frame dimensions")
        tile = a[r + 1:y - 1, c + 1:x - 1].copy()
        values = set(map(int, np.unique(tile)))
        _require(len(values) == 2 and color in values, "binary tile including category color")
        _require(size is None or len(tile) == size, "one macro-cell scale per input")
        size = len(tile)
        tiles.append(tile)
    r, c, y, x = map_box
    maze = a[r + 1:y - 1, c + 1:x - 1].copy()
    _require(min(maze.shape) >= 5 and max(maze.shape) * size <= 30, "expanded map bounds")
    route_values = set(map(int, np.unique(maze))) - {bg} - set(colors)
    _require(len(route_values) == 1, "one neutral corridor color required")
    route = route_values.pop()
    _require(route != divider and divider != bg, "map role colors")
    sources = []
    for color in colors:
        positions = np.argwhere(maze == color)
        _require(len(positions) == 1, "exactly one origin per tile frame")
        sources.append(tuple(map(int, positions[0])))
    mask = maze != bg
    parts = _components(outside, bg)
    labels = np.full(a.shape, -1, int)
    for i, part in enumerate(parts):
        _require(1 <= len(part["cells"]) <= 12, "loose object area")
        rr, cc, yy, xx = part["box"]
        _require(max(yy - rr, xx - cc) <= 6, "loose object extent")
        for yy, xx in part["cells"]:
            _require(not occupied[yy, xx], "loose object touches a frame")
            labels[yy, xx] = i
    # Generated scenes separate even diagonally, so edge components are clear.
    for yy, xx in np.argwhere(labels >= 0):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v, u = yy + dy, xx + dx
                if 0 <= v < len(a) and 0 <= u < a.shape[1]:
                    _require(labels[v, u] in (-1, labels[yy, xx]), "loose objects touch")
    counts = [sum(p["color"] == color for p in parts) for color in colors]
    _require(max(counts) <= 5, "budget range is zero through five")
    areas = [sum(len(p["cells"]) for p in parts if p["color"] == color) for color in colors]
    dist = np.asarray([_distance(mask, source) for source in sources])
    owners, ties = _owners(dist, counts, mask)
    return dict(background=bg, divider=divider, route=route, maze=maze, mask=mask,
                map_box=map_box, templates=templates, colors=colors, tiles=tiles,
                size=size, sources=sources, counts=counts, areas=areas, objects=parts,
                distances=dist, owners=owners, ties=ties)


def _render(d, owners=None, exchange=True, flat=False, only_category=False):
    owner = d["owners"] if owners is None else owners
    n = d["size"]
    base = np.where(d["mask"], d["route"], d["background"])
    out = np.repeat(np.repeat(base, n, axis=0), n, axis=1)
    for i, tile in enumerate(d["tiles"]):
        values = np.unique(tile)
        stamp = values.sum() - tile if exchange else tile
        if flat:
            stamp = np.full(tile.shape, d["colors"][i], int)
        if only_category:
            stamp = np.where(stamp == d["colors"][i], stamp, d["route"])
        for r, c in np.argwhere(owner == i):
            out[r * n:(r + 1) * n, c * n:(c + 1) * n] = stamp
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution; no seed, demonstrations, cached palette, or metadata."""
    return _render(_decode(input_grid))


def _inverse_pixels(d):
    """Check complete output pixels without calling the forward block painter."""
    n = d["size"]
    h, w = d["mask"].shape
    out = np.empty((h * n, w * n), int)
    for y, x in np.ndindex(out.shape):
        r, c = y // n, x // n
        i = int(d["owners"][r, c])
        if i < 0:
            out[y, x] = d["route"] if d["mask"][r, c] else d["background"]
        else:
            tile = d["tiles"][i]
            values = np.unique(tile)
            original = int(tile[y % n, x % n])
            out[y, x] = int(values[1] if original == values[0] else values[0])
    return out


def _alternative(d, behavior):
    if behavior == "manhattan":
        rr, cc = np.indices(d["mask"].shape)
        ds = np.asarray([abs(rr - r) + abs(cc - c) for r, c in d["sources"]])
        owners = _owners(ds, d["counts"], d["mask"])[0]
    elif behavior == "area":
        owners = _owners(d["distances"], d["areas"], d["mask"])[0]
    elif behavior in ("unlimited", "nearest_first"):
        owners = _owners(d["distances"], d["counts"], d["mask"], behavior)[0]
    elif behavior in ("extra_layer", "fewer_layers"):
        change = 1 if behavior == "extra_layer" else -1
        limits = [max(0, n + change) for n in d["counts"]]
        owners = _owners(d["distances"], limits, d["mask"])[0]
    elif behavior == "break_ties":
        eligible = np.where(d["distances"] < np.asarray(d["counts"])[:, None, None],
                            d["distances"], _INF)
        owners = np.where(d["mask"] & (eligible.min(axis=0) < _INF), eligible.argmin(axis=0), -1)
    else:
        owners = d["owners"]
    if behavior == "rotate_tile":
        d = dict(d, tiles=[np.rot90(t) for t in d["tiles"]])
    return _render(d, owners, exchange=behavior != "no_exchange", flat=behavior == "flat",
                   only_category=behavior == "only_category")


def _random_mask(rng, h, w):
    style = int(rng.integers(6))
    rr, cc = np.indices((h, w))
    if style in (0, 1):
        mask = rng.random((h, w)) < rng.uniform(.48, .82)
    elif style == 2:
        mask = np.zeros((h, w), bool)
        r, c = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(h * w * 3):
            mask[r, c] = True
            dr, dc = _NEIGHBORS[int(rng.integers(4))]
            r, c = int(np.clip(r + dr, 0, h - 1)), int(np.clip(c + dc, 0, w - 1))
    elif style == 3:
        mask = rr % 2 == 0
        for row in range(1, h, 2):
            mask[row, int(rng.integers(w))] = True
        mask |= rng.random((h, w)) < .07
    elif style == 4:
        mask = np.ones((h, w), bool)
        for _ in range(int(rng.integers(2, 7))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            rh, cw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            mask[r:r + rh, c:c + cw] = False
    else:
        mask = (rr % 3 == 0) | (cc % 3 == 0)
        mask &= rng.random((h, w)) > .08
        mask |= rng.random((h, w)) < .13
    if rng.random() < .5:
        mask = np.fliplr(mask)
    if rng.random() < .5:
        mask = np.flipud(mask)
    return mask.copy()


def _make_maze(rng, size, counts, kind):
    k = len(counts)
    upper = min(10 if kind == "test" else 9, 30 // size)
    for attempt in range(600):
        h, w = int(rng.integers(5, upper + 1)), int(rng.integers(5, upper + 1))
        mask = _random_mask(rng, h, w)
        points = [tuple(map(int, p)) for p in np.argwhere(mask)]
        if len(points) < 12:
            continue
        first = points[int(rng.integers(len(points)))]
        d0 = _distance(mask, first)
        if kind == "first":
            choices = [p for p in points if d0[p] == 2]
            if not choices:
                continue
            sources = [first, choices[int(rng.integers(len(choices)))]]
        elif kind == "second":
            close = [p for p in points if 1 <= d0[p] <= 2]
            far = [p for p in points if 3 <= d0[p] < _INF]
            if not close or not far:
                continue
            sources = [first, far[int(rng.integers(len(far)))], close[int(rng.integers(len(close)))]]
        else:
            remaining = [p for p in points if p != first]
            inds = rng.choice(len(remaining), k - 1, replace=False)
            sources = [first] + [remaining[int(j)] for j in inds]
        dist = np.asarray([_distance(mask, p) for p in sources])
        owners, ties = _owners(dist, counts, mask)
        if kind == "first":
            rr, cc = np.indices(mask.shape)
            md = np.asarray([abs(rr - r) + abs(cc - c) for r, c in sources])
            manhattan = _owners(md, counts, mask)[0]
            finite_cutoff = mask & (owners < 0) & ~ties & (dist.min(axis=0) < _INF)
            if not (ties.any() and finite_cutoff.any()
                    and not np.array_equal(owners, manhattan)
                    and all(np.any(owners == i) for i in range(k))
                    and np.any((owners == 0) & (dist[0] == 1))
                    and np.any((owners == 1) & (dist[1] == counts[1] - 1))
                    and int((owners >= 0).sum()) > sum(counts)):
                continue
        elif kind == "second":
            if owners[sources[2]] != 0 or not np.any(owners == 1):
                continue
        elif kind == "test":
            if not np.any((owners == 0) & (dist[0] == counts[0] - 1)):
                continue
        return mask, sources, dist, owners, ties
    raise AssertionError("bounded maze witness sampling failed")


def _shape(rng, area=None):
    """Geometry is sampled without a category budget; teaching can set area."""
    if area is None:
        area = int(rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)))
    style = int(rng.integers(4))
    if style == 0 and area <= 6:
        cells = {(0, i) for i in range(area)}
    elif style == 1:
        width = int(rng.integers(2, 5)) if area > 1 else 1
        cells = {(i // width, i % width) for i in range(area)}
    else:
        cells = {(1, 1)}
        while len(cells) < area:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _NEIGHBORS
                               if 0 <= r + dr < 4 and 0 <= c + dc < 4} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _binary_tile(rng, n, category, other, witness):
    for _ in range(100):
        bits = rng.random((n, n)) < rng.uniform(.2, .8)
        if not bits.any() or bits.all():
            continue
        if witness and any(np.array_equal(bits, np.rot90(bits, t)) for t in (1, 2, 3)):
            continue
        return np.where(bits, category, other).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, category, other).astype(int)


def _frame(content, color):
    a = np.full((content.shape[0] + 2, content.shape[1] + 2), color, int)
    a[1:-1, 1:-1] = content
    return a


def _pack(rng, rectangles, specs, bg):
    """Free pixel placements with moats; enlarge proposals instead of resizing shapes."""
    for attempt in range(100):
        low = 18 if attempt < 35 else 25 if attempt < 70 else 30
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        grid = np.full((h, w), bg, int)
        blocked = np.zeros((h, w), bool)
        blocked[[0, -1], :] = True
        blocked[:, [0, -1]] = True
        failed = False
        for rectangle in rectangles:
            rh, cw = rectangle.shape
            choices = [(r, c) for r in range(1, h - rh) for c in range(1, w - cw)
                       if not blocked[r:r + rh, c:c + cw].any()]
            if not choices:
                failed = True
                break
            r, c = choices[int(rng.integers(len(choices)))]
            grid[r:r + rh, c:c + cw] = rectangle
            blocked[r - 1:r + rh + 1, c - 1:c + cw + 1] = True
        if failed:
            continue
        order = sorted(range(len(specs)), key=lambda i: (-len(specs[i][1]), float(rng.random())))
        for index in order:
            color, shape = specs[index]
            sh, sw = 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)
            free = np.ones((h - sh + 1, w - sw + 1), bool)
            for dr, dc in shape:
                free &= ~blocked[dr:dr + h - sh + 1, dc:dc + w - sw + 1]
            positions = np.argwhere(free)
            if not len(positions):
                failed = True
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            for dr, dc in shape:
                y, x = r + dr, c + dc
                grid[y, x] = color
                blocked[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)] = True
        if not failed:
            return grid
    raise AssertionError("bounded free placement failed")


def _partition(rng, total, parts):
    cuts = [0] + sorted(map(int, rng.choice(np.arange(1, total), parts - 1, replace=False))) + [total]
    return [b - a for a, b in zip(cuts, cuts[1:])]


def _make_pair(rng, colors, bg, counts, kind, forced, required_distractor=None):
    available = [c for c in range(10) if c != bg and c not in colors]
    divider, route = map(int, rng.choice(available, 2, replace=False))
    n = int(rng.integers(2, 5))
    tiles = []
    for i, color in enumerate(colors):
        others = [c for c in range(10) if c != color and c != divider]
        # Ground and corridor sharing are common, but visible frames retain support.
        other = int(rng.choice([bg, route, int(rng.choice(others))]))
        if kind == "first" and i == 0:
            other = bg  # Teach complete overwrite, including the background role.
        tiles.append(_binary_tile(rng, n, color, other, kind in ("first", "second")))
    mask, sources, dist, owners, ties = _make_maze(rng, n, counts, kind)
    maze = np.where(mask, route, bg)
    for color, source in zip(colors, sources):
        maze[source] = color
    rectangles = [_frame(maze, divider)] + [_frame(t, c) for t, c in zip(tiles, colors)]
    specs = [(color, _shape(rng, forced.get(i, [None] * count)[j]))
             for i, (color, count) in enumerate(zip(colors, counts)) for j in range(count)]
    dcolors = [c for c in range(10) if c != bg and c not in colors]
    if required_distractor is not None:
        specs.append((required_distractor, _shape(rng)))
    for _ in range(int(rng.integers(0, 4))):
        specs.append((int(rng.choice(dcolors)), _shape(rng)))
    inp = _pack(rng, rectangles, specs, bg)
    detail = dict(mask=mask, route=route, background=bg, owners=owners, size=n,
                  tiles=tiles, colors=colors)
    return {"input": inp, "output": _render(detail)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    anchor, novel = map(int, rng.choice(10, 2, replace=False))
    collision_area = int(rng.integers(6, 10))
    first_palette, previous_bg = None, None
    seen, puzzle = set(), {"train": [], "test": []}
    for split, number in (("train", nt), ("test", nq)):
        for index in range(number):
            kind = ("first" if index == 0 else "second" if index == 1 else "ordinary") if split == "train" else "test"
            k = 2 if kind == "first" else 3 if kind == "second" else int(rng.integers(2, 4))
            excluded_bg = {previous_bg}
            if kind in ("first", "second"):
                excluded_bg.add(anchor)
            if kind == "second":
                excluded_bg.add(first_palette[1])
            if split == "test" and index == 0:
                excluded_bg.add(novel)
            bg = int(rng.choice([c for c in range(10) if c not in excluded_bg]))
            pool = [c for c in range(10) if c != bg and (split != "train" or c != novel)
                    and (kind != "second" or c != first_palette[1])]
            if kind in ("first", "second"):
                pool.remove(anchor)
                colors = [anchor] + list(map(int, rng.choice(pool, k - 1, replace=False)))
            elif split == "test" and index == 0:
                pool.remove(novel)
                colors = [novel] + list(map(int, rng.choice(pool, k - 1, replace=False)))
            else:
                colors = list(map(int, rng.choice(pool, k, replace=False)))
            forced, distractor = {}, None
            if kind == "first":
                counts = [2, 3]
                forced = {0: _partition(rng, collision_area, 2), 1: [1, 1, 1]}
                first_palette = colors[:]
            elif kind == "second":
                counts = [3, 1, 0]
                forced = {0: _partition(rng, collision_area, 3)}
                distractor = first_palette[1]
            else:
                counts = list(map(int, rng.integers(0, 4, k)))
                if split == "test":
                    counts[0] = int(rng.integers(4, 6))
                    while sum(counts) > 10:
                        j = 1 + int(rng.integers(k - 1))
                        if counts[j]:
                            counts[j] -= 1
            for _ in range(8):
                pair = _make_pair(rng, colors, bg, counts, kind, forced, distractor)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
            else:
                raise AssertionError("distinct scene sampling failed")
            seen.add(key)
            puzzle[split].append(pair)
            previous_bg = bg
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    details, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = inp.shape, inp.tobytes()
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _decode(inp)
            _require(np.array_equal(out, _inverse_pixels(d)), "wrong territory, tile, or neutral cells")
            # Exact support and both tile roles must be respected, including ties/walls.
            for r, c in np.argwhere(d["ties"]):
                n = d["size"]
                _require(np.all(out[r*n:(r+1)*n, c*n:(c+1)*n] == d["route"]), "tie not neutral")
            details[split].append(d)
    train, test = details["train"], details["test"]
    triples = [(c, n, a) for d in train for c, n, a in zip(d["colors"], d["counts"], d["areas"])]
    _require(any(c == cc and a == aa and n != nn for c, n, a in triples for cc, nn, aa in triples),
             "same-color area collision teaching missing")
    _require(any(n == nn and a != aa for c, n, a in triples for cc, nn, aa in triples),
             "same-count different-area teaching missing")
    _require({0, 1, 2, 3} <= {n for d in train for n in d["counts"]}, "budget progression missing")
    _require(any(d["ties"].any() for d in train), "eligible ties not demonstrated")
    _require(any(int((d["owners"] >= 0).sum()) > sum(d["counts"]) for d in train),
             "branching must distinguish range from number of stamped cells")
    _require(any(np.any(d["mask"] & (d["owners"] < 0) & ~d["ties"]
                               & (d["distances"].min(axis=0) < _INF)) for d in train),
             "reachable range cutoff not demonstrated")
    _require(any(n == 0 and d["owners"][s] >= 0 for d in train
                 for n, s in zip(d["counts"], d["sources"])), "zero origin traversal missing")
    categories = {c for d in train for c in d["colors"]}
    ignored = {p["color"] for d in train for p in d["objects"] if p["color"] not in d["colors"]}
    _require(bool(categories & ignored), "local category-to-distractor switch missing")
    for behavior in ("area", "manhattan", "nearest_first", "unlimited", "no_exchange", "flat",
                     "extra_layer", "fewer_layers", "break_ties", "rotate_tile", "only_category"):
        _require(any(not np.array_equal(_alternative(d, behavior), _render(d)) for d in train),
                 "training permits shortcut: " + behavior)
    _require(any(set(d["colors"]) - categories for d in test), "new test category missing")
    for d in test:
        _require(any(n >= 4 and np.any((d["owners"] == i) & (d["distances"][i] == n - 1))
                     for i, n in enumerate(d["counts"])), "test range shift has no visible effect")


def puzzle_metrics(puzzle):
    """All measurements come from returned inputs, not generation decisions."""
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            d = _decode(pair["input"])
            masks = []
            for p in d["objects"]:
                r, c, _, _ = p["box"]
                masks.append(tuple(sorted((y-r, x-c) for y, x in p["cells"])))
            records.append(dict(split=split, input_shape=np.asarray(pair["input"]).shape,
                                output_shape=np.asarray(pair["output"]).shape,
                                map_shape=d["mask"].shape, maze=d["mask"].tobytes(),
                                categories=len(d["colors"]), size=d["size"], counts=tuple(d["counts"]),
                                background=d["background"], route=d["route"], ties=int(d["ties"].sum()),
                                painted=int((d["owners"] >= 0).sum()), object_masks=masks,
                                object_areas=[len(p["cells"]) for p in d["objects"]],
                                template_boxes=tuple(box for box, _ in d["templates"]),
                                tile_masks=[(t == c).tobytes() for t, c in zip(d["tiles"], d["colors"])],
                                cutoff=bool(np.any(d["mask"] & (d["owners"] < 0) & ~d["ties"])),
                                disconnected=bool(np.any(d["mask"] & np.all(d["distances"] == _INF, axis=0))),
                                shortcuts={b: not np.array_equal(_alternative(d, b), _render(d))
                                           for b in ("manhattan", "nearest_first", "no_exchange", "unlimited", "flat")}))
    return records


def self_test():
    """Bounded semantic, diversity, determinism, count, and corruption checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [r for puzzle in puzzles for r in puzzle_metrics(puzzle)]
    signature = lambda p: tuple((np.asarray(pair["input"]).shape, np.asarray(pair["input"]).tobytes(),
                                 np.asarray(pair["output"]).tobytes())
                                for s in ("train", "test") for pair in p[s])
    _require(len({signature(p) for p in puzzles}) == 42, "cross-seed repetition")
    _require({r["size"] for r in records} == {2, 3, 4}, "tile-scale diversity")
    _require({r["categories"] for r in records} == {2, 3}, "category-count diversity")
    _require({r["background"] for r in records} == set(range(10)), "background palette diversity")
    _require({n for r in records for n in r["counts"]} == set(range(6)), "budget diversity")
    _require(len({r["input_shape"] for r in records}) >= 35, "scene geometry diversity")
    _require(len({r["output_shape"] for r in records}) >= 25, "map geometry diversity")
    _require(len({(r["map_shape"], r["maze"]) for r in records}) >= 150, "topology diversity")
    _require(len({m for r in records for m in r["object_masks"]}) >= 80, "object shape vocabulary")
    _require(max(a for r in records for a in r["object_areas"]) == 12, "large objects absent")
    _require(any(r["disconnected"] for r in records), "unreachable islands absent")
    _require(any(r["ties"] for r in records if r["split"] == "test"), "test seams absent")
    for p in puzzles:
        validate_generated_puzzle(p)
        for s in ("train", "test"):
            for pair in p[s]:
                inp = np.asarray(pair["input"])
                before = inp.copy()
                _require(np.array_equal(solve(inp), pair["output"]), "input-only solve mismatch")
                _require(np.array_equal(inp, before), "solver mutated input")
    _require(signature(generate_puzzle(137, 7, 3)) == signature(generate_puzzle(137, 7, 3)), "determinism")
    # Global relabeling must commute with solve, including the background role.
    witness = puzzles[0]["train"][0]
    relabel = np.random.default_rng(812).permutation(10)
    _require(np.array_equal(solve(relabel[witness["input"]]), relabel[witness["output"]]),
             "literal color dependency")
    # Alter one supported tile bit. Exactly that bit must change in every owned
    # macro-cell, while the competition footprint stays fixed.
    inp = witness["input"].copy()
    d = _decode(inp)
    i = next(i for i in range(len(d["tiles"])) if np.any(d["owners"] == i))
    tile = d["tiles"][i]
    vals, frequencies = np.unique(tile, return_counts=True)
    original = int(vals[int(np.argmax(frequencies))])
    other = int(next(v for v in vals if v != original))
    ty, tx = map(int, np.argwhere(tile == original)[0])
    r, c, _, _ = d["templates"][i][0]
    inp[r + 1 + ty, c + 1 + tx] = other
    changed = solve(inp) != witness["output"]
    expected_changes = np.zeros(changed.shape, bool)
    for y, x in np.argwhere(d["owners"] == i):
        expected_changes[y*d["size"] + ty, x*d["size"] + tx] = True
    _require(np.array_equal(changed, expected_changes), "tile substitution locality")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(700 + 10*a+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted answer accepted")
    return {"seeds": 42, "pairs": len(records),
            "input_shapes": len({r["input_shape"] for r in records}),
            "output_shapes": len({r["output_shape"] for r in records}),
            "maze_topologies": len({(r["map_shape"], r["maze"]) for r in records}),
            "object_shapes": len({m for r in records for m in r["object_masks"]})}


if __name__ == "__main__":
    print(self_test())
