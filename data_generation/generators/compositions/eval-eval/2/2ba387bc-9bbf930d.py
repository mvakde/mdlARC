"""Address-checked cargo delivery: independent shape ranks meet matching rails.

Only standard library and NumPy are used. Construction metadata is discarded;
solve reads every rank, rail, and destination from the supplied pixels.
"""
from __future__ import annotations

import json
from numbers import Integral

import numpy as np


DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "ARC colors required")
    return a


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _tile(size, color, bg, kind):
    a = np.full((size, size), color, dtype=np.int8)
    if kind == "H":
        a[1:-1, 1:-1] = bg
    return a


def _tiles(g, bg):
    """Separate complete square components from thin guide components."""
    # Linear Python integer indices avoid NumPy-scalar tuple arithmetic in
    # the component flood fill. _analyse has checked the background border,
    # so no foreground cell can wrap to another row via a +/-1 neighbor.
    width = g.shape[1]
    todo = set(map(int, np.flatnonzero(g != bg)))
    tiles = []
    mask = np.zeros(g.shape, dtype=bool)
    while todo:
        start = min(todo)
        todo.remove(start)
        top, left = divmod(start, width)
        bottom, right = top + 1, left + 1
        pending = [start]
        while pending:
            p = pending.pop()
            row, col = divmod(p, width)
            top, left = min(top, row), min(left, col)
            bottom, right = max(bottom, row + 1), max(right, col + 1)
            for q in (p-width, p+width, p-1, p+1):
                if q in todo:
                    todo.remove(q)
                    pending.append(q)
        s = bottom - top
        if s != right - left or s < 3 or s % 2 == 0:
            continue
        block = g[top:bottom, left:right]
        color = int(block[0, 0])
        if color == bg:
            continue
        if np.all(block == color):
            kind = "S"
        elif np.array_equal(block, _tile(s, color, bg, "H")):
            kind = "H"
        else:
            continue
        _require(0 < top and bottom < g.shape[0] and 0 < left and right < g.shape[1],
                 "tiles must be inset")
        surround = g[top-1:bottom+1, left-1:right+1].copy()
        surround[1:-1, 1:-1] = bg
        _require(np.all(surround == bg), "one-cell clear tile moat required")
        mask[top:bottom, left:right] = True
        tiles.append(dict(top=top, left=left, size=s, color=color, kind=kind))
    _require(tiles and {t["kind"] for t in tiles} == {"H", "S"}, "both square geometries required")
    _require(len({t["size"] for t in tiles}) == 1, "common square size required")
    tiles.sort(key=lambda t: (t["top"], t["left"]))
    for kind in ("H", "S"):
        queue = [t for t in tiles if t["kind"] == kind]
        _require(len({t["top"] for t in queue}) == len(queue), "same-class top-row tie")
        for rank, t in enumerate(queue):
            t["rank"] = rank
            t["id"] = (t["top"], t["left"])
    return tiles, mask


def _analyse(input_grid, ignore_matching=False, *, _parsed=None):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(np.all(g[0] == bg) and np.all(g[-1] == bg)
             and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg), "clear background border")
    if _parsed is None:
        tiles, tile_mask = _tiles(g, bg)
        tile_at = {}
        for t in tiles:
            for r in range(t["top"], t["top"] + t["size"]):
                for c in range(t["left"], t["left"] + t["size"]):
                    tile_at[r, c] = t
    else:
        # Reuse only a parse of this exact array within the current operation.
        # There is no cross-call cache or dependency in the public solver.
        _require(g is _parsed["grid"], "parsed geometry belongs to another input")
        tiles, tile_mask, tile_at = _parsed["tiles"], _parsed["tile_mask"], _parsed["tile_at"]

    def guide(p):
        return _inside(g, p) and not tile_mask[p] and int(g[p]) != bg

    def flanks(p, d):
        normal = (-d[1], d[0])
        a, b = _add(p, normal), _add(p, (-normal[0], -normal[1]))
        if guide(a) and guide(b):
            return int(g[a]), int(g[b])
        return None

    def paired(p, d):
        colors = flanks(p, d)
        return colors is not None and colors[0] == colors[1]

    routes, inactive = [], []
    for source in [t for t in tiles if t["kind"] == "S"]:
        size = source["size"]
        start = source["top"] + size // 2, source["left"] + size
        entrance, mismatch = None, None
        p = start
        while _inside(g, p) and int(g[p]) == bg:
            colors = flanks(p, (0, 1))
            if colors is not None:
                if colors[0] == colors[1] or ignore_matching:
                    entrance = p
                else:
                    mismatch = (p, colors)
                break  # The FIRST two-sided opening decides admission.
            p = _add(p, (0, 1))
        if entrance is None:
            inactive.append(dict(source=source, mismatch=mismatch))
            continue
        p, d = start, (0, 1)
        path, directions, turns, visited = [p], [], [], set()
        target = None
        while True:
            _require((p, d) not in visited, "route cycle")
            visited.add((p, d))
            q = _add(p, d)
            if not _inside(g, q):
                break
            if int(g[q]) == bg:
                p = q
                path.append(p)
                directions.append(d)
                continue
            if q in tile_at:
                candidate = tile_at[q]
                mid = candidate["size"] // 2
                centers = ((candidate["top"], candidate["left"] + mid),
                           (candidate["top"] + candidate["size"] - 1, candidate["left"] + mid),
                           (candidate["top"] + mid, candidate["left"]),
                           (candidate["top"] + mid, candidate["left"] + candidate["size"] - 1))
                if candidate["kind"] == "H" and q in centers:
                    target = candidate
                break  # A square rim is a dock/blocker, never a turning rail.
            choices = []
            for turn in ((-d[1], d[0]), (d[1], -d[0])):
                q = _add(p, turn)
                if _inside(g, q) and int(g[q]) == bg and paired(q, turn):
                    choices.append(turn)
            _require(len(choices) <= 1, "ambiguous turn")
            if not choices:
                break
            turns.append((p, d, choices[0]))
            d = choices[0]
        _require(len(path) == len(set(path)), "self-intersecting route")
        supports = [paired(q, direction) for q, direction in zip(path[1:], directions)]
        gaps = []
        for i, supported in enumerate(supports):
            if supported or i == 0 or i == len(supports)-1:
                continue
            if supports[i-1]:
                end = i
                while end < len(supports) and not supports[end]:
                    end += 1
                if end < len(supports) and len(set(directions[i-1:end+1])) == 1:
                    gaps.append(end-i)
        last = max((i for i, yes in enumerate(supports) if yes), default=-1)
        routes.append(dict(source=source, target=target, path=path, turns=turns,
                           entrance=entrance, rail=int(g[entrance[0]-1, entrance[1]]),
                           gaps=gaps, tail=len(supports)-1-last))
    paths = [p for route in routes for p in route["path"]]
    _require(len(paths) == len(set(paths)), "routes must have disjoint centerlines")
    receivers = [r["target"]["id"] for r in routes if r["target"] is not None]
    _require(len(receivers) == len(set(receivers)), "duplicate physical receiver")
    return dict(grid=g, background=bg, tiles=tiles, tile_mask=tile_mask, tile_at=tile_at,
                routes=routes, inactive=inactive)


def _deliveries(a, ignore_rank=False):
    return [r for r in a["routes"] if r["target"] is not None
            and (ignore_rank or r["source"]["rank"] == r["target"]["rank"])]


def _answer(a, ignore_rank=False):
    out = a["grid"].copy()
    for route in _deliveries(a, ignore_rank):
        source, target = route["source"], route["target"]
        s = source["size"]
        sr, sc = source["top"], source["left"]
        tr, tc = target["top"], target["left"]
        # Every address and route was computed on the ORIGINAL input.
        out[sr+1:sr+s-1, sc+1:sc+s-1] = a["background"]
        out[tr+1:tr+s-1, tc+1:tc+s-1] = a["grid"][sr+1:sr+s-1, sc+1:sc+s-1]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer square ranks and matching-rail deliveries solely from this grid."""
    return _answer(_analyse(input_grid))


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        if (p[0] == target[0]) == (p[1] == target[1]):
            return None
        d = int(np.sign(target[0]-p[0])), int(np.sign(target[1]-p[1]))
        while p != target:
            p = _add(p, d)
            path.append(p)
    cells = set(path)
    if len(cells) != len(path):
        return None
    for i, p in enumerate(path):
        r, c = p
        degree = ((r-1, c) in cells) + ((r+1, c) in cells) + ((r, c-1) in cells) + ((r, c+1) in cells)
        if degree != (1 if i in (0, len(path)-1) else 2):
            return None
    return path


def _near_tile(p, t, gap=1):
    return (t["top"]-gap <= p[0] < t["top"]+t["size"]+gap
            and t["left"]-gap <= p[1] < t["left"]+t["size"]+gap)


def _render(shape, tiles, routes, bg):
    g = np.full(shape, bg, dtype=np.int8)
    tops = {"H": set(), "S": set()}
    for i, t in enumerate(tiles):
        r, c, s = t["top"], t["left"], t["size"]
        if r in tops[t["kind"]]:
            return None  # The input parser would reject this same-class tie.
        tops[t["kind"]].add(r)
        if not (1 <= r and r+s < shape[0] and 1 <= c and c+s < shape[1]):
            return None
        if any(not (r+s < u["top"] or u["top"]+u["size"] < r
                    or c+s < u["left"] or u["left"]+u["size"] < c) for u in tiles[:i]):
            return None
        g[r:r+s, c:c+s] = _tile(s, t["color"], bg, t["kind"])
    all_paths = [p for route in routes for p in route["path"]]
    if len(all_paths) != len(set(all_paths)):
        return None
    path_cells = set(all_paths)
    if any(not _inside(g, p) or g[p] != bg for p in all_paths):
        return None
    moat = np.zeros(shape, dtype=bool)
    for t in tiles:
        r, c, s = t["top"], t["left"], t["size"]
        moat[r-1:r+s+1, c-1:c+s+1] = True
    walls = {}
    for route in routes:
        path, color = route["path"], route["color"]
        cells = set(path)
        local = {}
        for p in path:
            for d in DIRS:
                q = _add(p, d)
                if (1 <= q[0] < shape[0]-1 and 1 <= q[1] < shape[1]-1
                        and q not in cells and not moat[q]):
                    local[q] = color
        for q in route.get("remove", ()):
            local.pop(q, None)
        if not route["matching"]:
            # Give the initial upper strand a different identity. Retain the
            # cells supporting every outgoing corner, so geometric tracing
            # still reaches the same dock when entrance equality is ablated.
            r, c = path[0]
            first_end = c
            for q in path[1:]:
                if q[0] != r:
                    break
                first_end = q[1]
            bound = first_end-1 if any(q[0] != r for q in path) else first_end+1
            painted = [(r-1, x) for x in range(c, bound) if (r-1, x) in local]
            if not painted:
                return None
            for q in painted:
                local[q] = route["other"]
        for q, value in local.items():
            if q in path_cells or (q in walls and walls[q] != value):
                return None
            walls[q] = value
    for q, value in walls.items():
        g[q] = value
    return g


def _checked_scene(shape, tiles, routes, bg):
    g = _render(shape, tiles, routes, bg)
    if g is None:
        return None
    try:
        a = _analyse(g)
        geometric = _analyse(g, ignore_matching=True, _parsed=a)
    except AssertionError:
        return None
    actual_tiles = {(t["top"], t["left"], t["kind"], t["color"]) for t in a["tiles"]}
    wanted_tiles = {(t["top"], t["left"], t["kind"], t["color"]) for t in tiles}
    if actual_tiles != wanted_tiles:
        return None  # In particular, rails may not masquerade as square tiles.
    wanted = {(r["source"]["top"], r["source"]["left"]): r for r in routes}
    seen = set()
    for r in geometric["routes"]:
        sid = r["source"]["id"]
        expected = wanted.get(sid)
        if (expected is None or r["path"] != expected["path"] or r["target"] is None
                or r["target"]["id"] != (expected["target"]["top"], expected["target"]["left"])):
            return None
        seen.add(sid)
    if seen != set(wanted):
        return None
    active = {r["source"]["id"] for r in a["routes"]}
    if active != {sid for sid, r in wanted.items() if r["matching"]}:
        return None
    return g, a


def _decorate(rng, route, tiles, gap=False):
    path = route["path"]
    directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    eligible = []
    for i in range(3, len(path)-3):
        if len(set(directions[i-2:i+2])) != 1 or any(_near_tile(path[i], t, 2) for t in tiles):
            continue
        d = directions[i]
        n = (-d[1], d[0])
        eligible.append((i, (_add(path[i], n), _add(path[i], (-n[0], -n[1])))))
    remove = set()
    if eligible and (gap or rng.random() < .45):
        start = eligible[int(rng.integers(len(eligible)))][0]
        length = int(rng.integers(1, 4))
        for i, flanks in eligible:
            if start <= i < start+length:
                remove.update(flanks)
    if rng.random() < .6:
        for i in range(len(path)-2, max(2, len(path)-5), -1):
            if len(set(directions[max(0, i-2):i+1])) != 1:
                break
            d = directions[i]
            n = (-d[1], d[0])
            remove.update((_add(path[i], n), _add(path[i], (-n[0], -n[1]))))
    route["remove"] = remove


def _new_tile(top, left, size, color, kind):
    return dict(top=int(top), left=int(left), size=size, color=int(color), kind=kind)


def _new_route(rng, source, target, path, palette, matching):
    color = int(rng.choice(palette))
    return dict(source=source, target=target, path=path, color=color,
                matching=matching, other=int(rng.choice([c for c in palette if c != color])))


def _add_orphan(rng, shape, tiles, routes, bg, palette, size, kind, tops):
    sites = [(r, c) for r in tops for c in range(1, shape[1]-size)]
    for index in rng.permutation(len(sites)):
        r, c = sites[int(index)]
        if any(t["kind"] == kind and t["top"] == r for t in tiles):
            continue
        tile = _new_tile(r, c, size, rng.choice(palette), kind)
        scene = _checked_scene(shape, tiles+[tile], routes, bg)
        if scene is not None:
            return tiles+[tile], scene
    return None


def _witness(rng, bg, palette, address_witness, combined=False):
    """Variable readable demonstrations, with verified counterexamples.

    These are a constructive starting regime, not the general route sampler.
    One demonstration teaches color gating; the other inserts idle objects
    into only one queue's middle and exposes address rejection.
    """
    for _ in range(64):
        s = 3
        w = int(rng.integers(24 if address_witness else 20, 31))
        tiles, routes, top = [], [], 1
        bent = int(rng.integers(3))
        for i in range(3):
            delta = int(rng.choice([-3, 3])) if i == bent else 0
            source_r = top + s//2 + max(0, -delta)
            target_r = source_r + delta
            source_c = int(rng.integers(1, min(6, w-2*s-10)+1))
            target_c = int(rng.integers(source_c+s+9, w-s))
            source = _new_tile(source_r-s//2, source_c, s, rng.choice(palette), "S")
            target = _new_tile(target_r-s//2, target_c, s, rng.choice(palette), "H")
            start, end = (source_r, source_c+s), (target_r, target_c-1)
            if delta:
                bend_c = int(rng.integers(start[1]+4, end[1]-2))
                path = _polyline([start, (source_r, bend_c), (target_r, bend_c), end])
            else:
                path = _polyline([start, end])
            matching = (i != 2) if combined else (address_witness or i != (bent+1)%3)
            route = _new_route(rng, source, target, path, palette, matching)
            _decorate(rng, route, [source, target], gap=i == bent)
            tiles.extend([source, target])
            routes.append(route)
            top += s + abs(delta) + int(rng.integers(2, 4))
        minimum_h = max(t["top"]+s for t in tiles)+2
        if address_witness:
            minimum_h += s+1
        if minimum_h > 30:
            continue
        h = int(rng.integers(minimum_h, 31))
        scene = _checked_scene((h, w), tiles, routes, bg)
        if scene is None:
            continue
        if address_witness:
            hollow_tops = sorted(t["top"] for t in tiles if t["kind"] == "H")
            added = _add_orphan(rng, (h, w), tiles, routes, bg, palette, s, "H",
                                range(hollow_tops[0]+1, hollow_tops[1]))
            if added is None:
                continue
            tiles, scene = added
            solid_tops = sorted(t["top"] for t in tiles if t["kind"] == "S")
            tops = (range(solid_tops[1]+1, solid_tops[2]) if combined
                    else range(solid_tops[-1]+1, h-s))
            added = _add_orphan(rng, (h, w), tiles, routes, bg, palette, s, "S", tops)
            if added is None:
                continue
            tiles, scene = added
        g, a = scene
        if address_witness:
            if not _deliveries(a) or np.array_equal(_answer(a), _answer(a, ignore_rank=True)):
                continue
            if combined and np.array_equal(_answer(a), _answer(_analyse(g, True, _parsed=a))):
                continue
        else:
            if len(_deliveries(a)) != 2 or not any(r["turns"] and r["gaps"] for r in _deliveries(a)):
                continue
            if np.array_equal(_answer(a), _answer(_analyse(g, ignore_matching=True, _parsed=a))):
                continue
        return dict(input=g, output=_answer(a))
    raise AssertionError("bounded witness construction exhausted")


def _random_module(rng, shape, size, palette, max_bends):
    h, w = shape
    row = int(rng.integers(1+size//2, h-size//2-1))
    col = int(rng.integers(size+1, w-size-4))
    source = _new_tile(row-size//2, col-size, size, rng.choice(palette), "S")
    points, direction = [(row, col)], (0, 1)
    bends = int(rng.choice(np.arange(max_bends+1), p=np.array([4]+[2]*max_bends)/(4+2*max_bends)))
    for i in range(bends+1):
        p = points[-1]
        choices = []
        for length in range(4 if i == 0 else 3, 14):
            q = (p[0]+length*direction[0], p[1]+length*direction[1])
            if 1 <= q[0] < h-1 and 1 <= q[1] < w-1:
                choices.append(q)
        if not choices:
            return None
        points.append(choices[int(rng.integers(len(choices)))])
        if i < bends:
            direction = ((-direction[1], direction[0]) if rng.integers(2)
                         else (direction[1], -direction[0]))
    end = points[-1]
    if direction == (0, 1):
        target_pos = end[0]-size//2, end[1]+1
    elif direction == (0, -1):
        target_pos = end[0]-size//2, end[1]-size
    elif direction == (1, 0):
        target_pos = end[0]+1, end[1]-size//2
    else:
        target_pos = end[0]-size, end[1]-size//2
    target = _new_tile(*target_pos, size, rng.choice(palette), "H")
    path = _polyline(points)
    if path is None:
        return None
    route = _new_route(rng, source, target, path, palette, bool(rng.random() < .72))
    _decorate(rng, route, [source, target])
    return source, target, route


def _free_scene(rng, bg, palette, size, is_test):
    for _ in range(18):
        lower = 23 if size == 5 else (23 if is_test else 17)
        h, w = map(int, rng.integers(lower, 31, size=2))
        desired = int(rng.integers(2, 5 if size == 5 else (6 if is_test else 5)))
        tiles, routes = [], []
        scene = None
        for _ in range(180):
            if len(routes) >= desired:
                break
            module = _random_module(rng, (h, w), size, palette, 3 if is_test else 2)
            if module is None:
                continue
            source, target, route = module
            candidate = _checked_scene((h, w), tiles+[source, target], routes+[route], bg)
            if candidate is not None:
                tiles.extend([source, target])
                routes.append(route)
                scene = candidate
        if scene is None or len(routes) < 2:
            continue
        # Idle tiles alter queue addresses without adding a transport channel.
        for _ in range(int(rng.integers(0, 3 if is_test else 2))):
            kind = str(rng.choice(["H", "S"]))
            added = _add_orphan(rng, (h, w), tiles, routes, bg, palette, size, kind, range(1, h-size))
            if added is not None:
                tiles, scene = added
        g, a = scene
        rank_witness = not np.array_equal(_answer(a), _answer(a, True))
        rail_witness = not np.array_equal(_answer(a), _answer(_analyse(g, True, _parsed=a)))
        if _deliveries(a) and (not is_test or rank_witness and rail_witness):
            return dict(input=g, output=_answer(a))
    return None


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    fixed_roles = bool(rng.random() < .3)
    colors = list(map(int, rng.permutation(10)))
    varying_size = num_train >= 3 and bool(rng.random() < .65)
    taught_sizes = set()
    for index in range(num_train+num_test):
        is_test = index >= num_train
        if not fixed_roles:
            colors = list(map(int, rng.permutation(10)))
        bg = colors[0]
        # Tile and rail palettes may overlap. Their visible geometry defines roles.
        palette = colors[1:1+int(rng.integers(3, 10))]
        size = 5 if varying_size and (index == 2 or index > 2 and rng.random() < .4) else 3
        if is_test and size not in taught_sizes:
            size = 3
        for _ in range(12):
            if index < 2:
                pair = _witness(rng, bg, palette, index == 1)
            else:
                pair = _free_scene(rng, bg, palette, size, is_test)
                if pair is None:
                    pair = _witness(rng, bg, palette, is_test, combined=is_test)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise AssertionError("bounded uniqueness construction exhausted")
        seen.add(_key(pair["input"]))
        puzzle["test" if is_test else "train"].append(pair)
        if not is_test:
            taught_sizes.add(_analyse(pair["input"])["tiles"][0]["size"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "legal counts")
    seen, train = set(), []
    taught_sizes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "same canvas")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyse(inp)
            moves = _deliveries(a)
            _require(moves, "each generated example must deliver cargo")
            _require(np.array_equal(out, _answer(a)), "wrong deliveries")
            changed_allowed = np.zeros(inp.shape, dtype=bool)
            for r in moves:
                for t in (r["source"], r["target"]):
                    y, x, s = t["top"], t["left"], t["size"]
                    changed_allowed[y+1:y+s-1, x+1:x+s-1] = True
            _require(np.array_equal(inp[~changed_allowed], out[~changed_allowed]), "rims and rails preserved")
            s = a["tiles"][0]["size"]
            _require(np.count_nonzero(inp != out) == 2*(s-2)**2*len(moves), "only complete interiors move")
            _require(np.array_equal(np.bincount(inp.ravel(), minlength=10),
                                    np.bincount(out.ravel(), minlength=10)), "per-color cargo conservation")
            # A failed address or entrance leaves BOTH corresponding tiles alone.
            moved_ids = {r["source"]["id"] for r in moves} | {r["target"]["id"] for r in moves}
            for t in a["tiles"]:
                if t["id"] not in moved_ids:
                    y, x, s = t["top"], t["left"], t["size"]
                    _require(np.array_equal(inp[y:y+s, x:x+s], out[y:y+s, x:x+s]), "inert tile changed")
            if split == "train":
                train.append(a)
                taught_sizes.add(a["tiles"][0]["size"])
            else:
                _require(a["tiles"][0]["size"] in taught_sizes, "test payload size absent from training")
                _require(not np.array_equal(_answer(a), _answer(a, True)), "test must exercise address rejection")
                _require(not np.array_equal(_answer(a), _answer(_analyse(inp, True, _parsed=a))),
                         "test must exercise rail-color rejection")
    _require(any(not np.array_equal(_answer(a), _answer(a, ignore_rank=True)) for a in train),
             "training must refute ignoring independent ranks")
    _require(any(not np.array_equal(_answer(a), _answer(_analyse(a["grid"], True, _parsed=a))) for a in train),
             "training must refute ignoring rail-color equality")
    _require(any(len(_deliveries(a)) >= 2 for a in train), "simultaneous transfers must be taught")
    _require(any(r["turns"] and r["gaps"] for a in train for r in _deliveries(a)),
             "training must demonstrate bends and directional rail gaps")
    _require(any(not r["mismatch"] for a in train for r in a["inactive"]), "idle solid must be taught")


def measurements(puzzle):
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _analyse(pair["input"])
            geometric = _analyse(a["grid"], True, _parsed=a)
            records.append(dict(split=split, shape=list(a["grid"].shape), background=a["background"],
                                size=a["tiles"][0]["size"],
                                hollow=sum(t["kind"] == "H" for t in a["tiles"]),
                                solid=sum(t["kind"] == "S" for t in a["tiles"]),
                                active=len(a["routes"]), deliveries=len(_deliveries(a)),
                                mismatch=sum(r["mismatch"] is not None for r in a["inactive"]),
                                idle=sum(r["mismatch"] is None for r in a["inactive"]),
                                wrong_address=sum(r["target"] is not None and r["source"]["rank"] != r["target"]["rank"]
                                                  for r in a["routes"]),
                                turns=[len(r["turns"]) for r in geometric["routes"]],
                                lengths=[len(r["path"]) for r in geometric["routes"]],
                                gaps=[gap for r in a["routes"] for gap in r["gaps"]],
                                directions=[list((r["path"][-1][0]-r["path"][-2][0],
                                                  r["path"][-1][1]-r["path"][-2][1])) for r in geometric["routes"]],
                                layout=[(t["top"], t["left"], t["size"], t["kind"]) for t in a["tiles"]],
                                rank_ablation=not np.array_equal(_answer(a), _answer(a, True)),
                                rail_ablation=not np.array_equal(_answer(a), _answer(geometric))))
    return records


def self_test():
    """Bounded 42-seed diversity, contracts, invariants, and negative controls."""
    signatures, records = set(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        # generate_puzzle already invokes the full family validator.
        signatures.add(tuple(_key(x["input"]) for s in p for x in p[s]))
        records.extend(measurements(p))
        for split in p:
            for pair in p[split]:
                before = pair["input"].copy()
                _require(np.array_equal(solve(before), pair["output"]), "input-only prediction")
                _require(np.array_equal(before, pair["input"]), "solve mutated its input")
    _require(len(signatures) == 42, "cross-seed collapse")
    _require(len({tuple(r["shape"]) for r in records}) >= 35, "canvas diversity")
    _require(len({tuple(r["layout"]) for r in records}) >= 180, "geometry diversity")
    _require({r["background"] for r in records} == set(range(10)), "background palette diversity")
    _require({r["size"] for r in records} == {3, 5}, "payload scale diversity")
    _require(len({(r["hollow"], r["solid"]) for r in records}) >= 10, "queue count diversity")
    _require({0, 1, 2, 3} <= {n for r in records for n in r["turns"]}, "route bend diversity")
    _require(len({tuple(d) for r in records for d in r["directions"]}) >= 3, "dock-side diversity")
    _require(any(r["hollow"] > r["solid"] for r in records)
             and any(r["solid"] > r["hollow"] for r in records), "both unmatched-tail directions")
    # The protected host checks all 30 legal combinations. Keep representative
    # boundaries here without repeating that expensive generation matrix.
    for n, m in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(2000+n*10+m, n, m)
        _require(len(p["train"]) == n and len(p["test"]) == m, "explicit counts")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(816424292, 2, 1), generate_puzzle(816424292, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x), "determinism")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"] = damaged["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing delivery accepted")
    # Explicit rule regressions outside the sampler's size/terminal range.
    # Expected arrays are stated directly, not obtained by another solver.
    large = np.zeros((10, 27), dtype=np.int8)
    large[1:8, 1:8] = 2
    large[1:8, 19:26] = _tile(7, 4, 0, "H")
    large[3, 9:18] = large[5, 9:18] = 3
    expected = large.copy()
    expected[2:7, 2:7] = 0
    expected[2:7, 20:25] = 2
    _require(np.array_equal(solve(large), expected), "side-seven cargo regression")
    interrupted = large.copy()
    interrupted[3, 12:15] = interrupted[5, 12:15] = 0
    expected_gap = expected.copy()
    expected_gap[3, 12:15] = expected_gap[5, 12:15] = 0
    _require(np.array_equal(solve(interrupted), expected_gap), "unsealed straight gap regression")
    mismatch = large.copy()
    mismatch[3, 9] = 7
    _require(np.array_equal(solve(mismatch), mismatch), "first mismatch cannot be overridden")
    blocked = large.copy()
    blocked[4, 14] = 3
    _require(np.array_equal(solve(blocked), blocked), "closed nonsocket terminal cannot deliver")
    return dict(seeds=42, pairs=len(records), shapes=len({tuple(r["shape"]) for r in records}),
                layouts=len({tuple(r["layout"]) for r in records}),
                queue_counts=sorted({(r["hollow"], r["solid"]) for r in records}),
                bend_counts=sorted({n for r in records for n in r["turns"]}))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
