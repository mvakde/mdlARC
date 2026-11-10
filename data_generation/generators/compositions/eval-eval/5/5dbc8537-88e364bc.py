"""Clearance docking: inventory-constrained full-body ray projection.

Only Python's standard library and NumPy are required. No file access or mutable
generation state participates in solve. See task_spec.md for the family rule.
"""
from collections import Counter
import hashlib
import numpy as np


CARDINAL = ((0, 1), (-1, 0), (0, -1), (1, 0))
HOST_SEEDS = (0, 534084706, 87336376, 100293840, 605768471, 1968898,
              547446974, 398305742, 417382854, 342351087, 44552795,
              871847381, 418885330, 435803529, 635424458, 561682882,
              301408455, 322666011, 456183631, 827169837, 602061809,
              570453949, 601317601, 884462678, 720257055, 489154953,
              957208194, 399637964, 944974229, 517944878, 202234883,
              735060462, 222103574, 353008451, 728199340, 600445851,
              142472761, 652248355, 863994194, 798980032, 716135421,
              259593608)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a


def _fingerprint(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _components(mask):
    h, w = mask.shape
    seen = ~mask.copy()
    result = []
    for rr, cc in np.argwhere(mask):
        start = int(rr), int(cc)
        if seen[start]:
            continue
        cells, stack = [], [start]
        seen[start] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in CARDINAL:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def _key_interpretations(key, carrier):
    others = set(map(int, np.unique(key))) - {carrier}
    if len(others) != 2:
        return []
    answers = []
    for tip in sorted(others):
        trail = next(iter(others - {tip}))
        tips = set(map(tuple, np.argwhere(key == tip)))
        trails = set(map(tuple, np.argwhere(key == trail)))
        if len(tips) != 2:
            continue
        for dr, dc in CARDINAL:
            used, lengths = set(), []
            for r, c in sorted(tips):
                stroke = []
                r, c = r - dr, c - dc
                while (r, c) in trails:
                    stroke.append((r, c))
                    r, c = r - dr, c - dc
                if len(stroke) not in (2, 3) or used.intersection(stroke):
                    break
                used.update(stroke)
                lengths.append(len(stroke))
            if len(lengths) == 2 and used == trails:
                answers.append((trail, tip, (dr, dc)))
    return answers


def _read_tray(tray, wall):
    # The tray has a background margin; numerical background is not fixed.
    bg = int(tray[0, 0])
    if any(np.any(edge != bg) for edge in (tray[0], tray[-1], tray[:, 0], tray[:, -1])):
        raise ValueError("tray needs a background margin")
    objects = []
    key_record = None
    for cells in _components(tray != bg):
        r0, c0, r1, c1 = _box(cells)
        sub = tray[r0:r1, c0:c1]
        if sub.size != len(cells):
            raise ValueError("tray object is not a filled rectangle")
        if np.any(sub == wall):
            if key_record is not None or any(np.any(e != wall) for e in
                    (sub[0], sub[-1], sub[:, 0], sub[:, -1])):
                raise ValueError("key border does not match arena wall")
            meanings = _key_interpretations(sub, wall)
            if len(meanings) != 1:
                raise ValueError("key is not uniquely interpretable")
            key_record = (meanings[0], sub.shape, (r0, c0))
        else:
            objects.append(sub.copy())
    if key_record is None or not 2 <= len(objects) <= 7:
        raise ValueError("need one key and 2..7 physical stamps")
    (peg, tip, direction), key_shape, key_position = key_record
    return bg, peg, tip, direction, objects, key_shape, key_position


def _panel_candidates(grid):
    for axis in (0, 1):
        for cut in range(1, grid.shape[axis]):
            a, b = (grid[:cut], grid[cut:]) if axis == 0 else (grid[:, :cut], grid[:, cut:])
            for arena, tray, side in ((a, b, "top" if axis == 0 else "left"),
                                      (b, a, "bottom" if axis == 0 else "right")):
                if min(arena.shape) < 4 or min(tray.shape) < 3:
                    continue
                wall = int(arena[0, 0])
                if any(np.any(e != wall) for e in (arena[0], arena[-1], arena[:, 0], arena[:, -1])):
                    continue
                colors = set(map(int, np.unique(arena)))
                if len(colors) != 4:
                    continue
                try:
                    vacancy, peg, tip, direction, tiles, key_shape, key_position = _read_tray(tray, wall)
                except ValueError:
                    continue
                remaining = colors - {wall, vacancy, peg}
                if len(remaining) != 1 or not {wall, vacancy, peg} <= colors or tip in colors:
                    continue
                floor = next(iter(remaining))
                reserved = {floor, wall, vacancy, peg, tip}
                if len(reserved) != 5 or any(set(map(int, t.flat)) & reserved for t in tiles):
                    continue
                pegs = [tuple(map(int, p)) for p in np.argwhere(arena == peg)]
                if not 2 <= len(pegs) <= 5:
                    continue
                if any(max(t.shape) > 5 for t in tiles):
                    continue
                matrices = {}
                consistent = True
                for tile in tiles:
                    key = _fingerprint(tile)
                    if tile.shape in matrices and matrices[tile.shape] != key:
                        consistent = False
                    matrices[tile.shape] = key
                if not consistent:
                    continue
                yield dict(arena=arena, tray=tray, wall=wall, vacancy=vacancy,
                           peg=peg, floor=floor, tip=tip, direction=direction,
                           tiles=tiles, pegs=pegs, mask=arena == vacancy, side=side,
                           key_shape=key_shape, key_position=key_position)


def _parse(grid):
    meanings = list(_panel_candidates(_grid(grid)))
    if len(meanings) != 1:
        raise ValueError("input must have exactly one semantic arena/tray split")
    return meanings[0]


def _groups(tiles):
    groups, lookup = [], {}
    for tile in tiles:
        key = _fingerprint(tile)
        if key not in lookup:
            lookup[key] = len(groups)
            groups.append({"pattern": tile, "count": 0})
        groups[lookup[key]]["count"] += 1
    return groups


def _landing(wall, start, shape, direction):
    """Last legal upper-left coordinate, checking the entire translated body."""
    r, c = start
    h, w = shape
    dr, dc = direction

    def free(y, x):
        return (0 <= y and 0 <= x and y + h <= wall.shape[0]
                and x + w <= wall.shape[1] and not wall[y:y+h, x:x+w].any())

    if not free(r, c):
        return None
    distance = 0
    while free(r + dr, c + dc):
        r, c = r + dr, c + dc
        distance += 1
    return r, c, distance


def _target_index(mask):
    index = np.full(mask.shape, -1, dtype=int)
    index[mask] = np.arange(int(mask.sum()))
    return index


def _placement(index, group, pattern, r, c, **extra):
    h, w = pattern.shape
    cells = tuple(map(int, index[r:r+h, c:c+w].flat))
    bits = sum(1 << i for i in cells)
    return dict(group=group, bits=bits, assignments=tuple(zip(cells, map(int, pattern.flat))),
                rect=(r, c, h, w), **extra)


def _launch_options(info, reverse=False):
    groups = _groups(info["tiles"])
    index = _target_index(info["mask"])
    wall = info["arena"] == info["wall"]
    direction = tuple(-v for v in info["direction"]) if reverse else info["direction"]
    options = []
    for start in info["pegs"]:
        choices = []
        for g, group in enumerate(groups):
            tile = group["pattern"]
            landing = _landing(wall, start, tile.shape, direction)
            if landing is None:
                continue
            r, c, distance = landing
            h, w = tile.shape
            if info["mask"][r:r+h, c:c+w].all():
                choices.append(_placement(index, g, tile, r, c, start=start, distance=distance))
        options.append(choices)
    return groups, options


def _launch_outcomes(info, reverse=False):
    groups, options = _launch_options(info, reverse)
    full = (1 << int(info["mask"].sum())) - 1
    remaining = [g["count"] for g in groups]
    rendering = [-1] * int(info["mask"].sum())
    outcomes, path = {}, []
    nodes = 0

    def search(done, covered):
        nonlocal nodes
        nodes += 1
        # With <=5 pegs and <=7 stamps, even 7^5 branches are bounded.
        if nodes > 200000:
            raise ValueError("launch assignment budget exceeded")
        if len(outcomes) > 1:
            return
        if done == (1 << len(options)) - 1:
            if covered == full:
                outcomes.setdefault(tuple(rendering), tuple(path))
            return
        best, chosen, possible = None, None, covered
        for i, choices in enumerate(options):
            if done & (1 << i):
                continue
            legal = [p for p in choices if remaining[p["group"]] and not covered & p["bits"]]
            if not legal:
                return
            for p in legal:
                possible |= p["bits"]
            if best is None or len(legal) < len(best):
                best, chosen = legal, i
        if possible != full:
            return
        for placement in best:
            g = placement["group"]
            remaining[g] -= 1
            for cell, color in placement["assignments"]:
                rendering[cell] = color
            path.append(placement)
            search(done | (1 << chosen), covered | placement["bits"])
            path.pop()
            for cell, _ in placement["assignments"]:
                rendering[cell] = -1
            remaining[g] += 1
            if len(outcomes) > 1:
                return

    search(0, 0)
    return outcomes, groups, options, nodes


def _infer(grid):
    info = _parse(grid)
    outcomes, groups, options, nodes = _launch_outcomes(info)
    if len(outcomes) != 1:
        raise ValueError("docking must have exactly one rendered answer")
    rendering, placements = next(iter(outcomes.items()))
    output = info["arena"].copy()
    output[output == info["peg"]] = info["floor"]
    output[info["mask"]] = rendering
    info.update(output=output, placements=placements, groups=groups, options=options, nodes=nodes)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve exclusively from this grid; do not mutate the caller's array."""
    return _infer(np.asarray(input_grid))["output"]


def _unrestricted_packings(info, budget=16000):
    """Ablation: remove pegs/key/rays but retain A's finite patterned inventory.

    This is a diagnostic of the omitted-B shortcut, not an independent solver.
    Return at most two rendered outcomes and whether search was conclusive.
    """
    mask, groups = info["mask"], _groups(info["tiles"])
    index = _target_index(mask)
    count = int(mask.sum())
    by_cell = [[] for _ in range(count)]
    for g, group in enumerate(groups):
        pattern = group["pattern"]
        h, w = pattern.shape
        for r in range(mask.shape[0] - h + 1):
            for c in range(mask.shape[1] - w + 1):
                if mask[r:r+h, c:c+w].all():
                    p = _placement(index, g, pattern, r, c)
                    for cell, _ in p["assignments"]:
                        by_cell[cell].append(p)
    full, remaining = (1 << count) - 1, [g["count"] for g in groups]
    rendering, outcomes, dead = [-1] * count, set(), set()
    nodes, exhaustive = 0, True

    def search(covered):
        nonlocal nodes, exhaustive
        if len(outcomes) > 1 or not exhaustive:
            return
        nodes += 1
        if nodes > budget:
            exhaustive = False
            return
        if covered == full:
            outcomes.add(tuple(rendering))
            return
        state = covered, tuple(remaining)
        if state in dead:
            return
        best = None
        for i, candidates in enumerate(by_cell):
            if covered & (1 << i):
                continue
            legal = [p for p in candidates if remaining[p["group"]] and not covered & p["bits"]]
            if not legal:
                dead.add(state)
                return
            if best is None or len(legal) < len(best):
                best = legal
        previous = len(outcomes)
        for p in best:
            remaining[p["group"]] -= 1
            for cell, color in p["assignments"]:
                rendering[cell] = color
            search(covered | p["bits"])
            for cell, _ in p["assignments"]:
                rendering[cell] = -1
            remaining[p["group"]] += 1
            if len(outcomes) > 1 or not exhaustive:
                return
        # Do not memoize successful states: their colored prefixes can differ.
        if len(outcomes) == previous == 0:
            dead.add(state)

    search(0)
    return outcomes, exhaustive


def _point_output(info):
    result = info["arena"].copy()
    result[result == info["peg"]] = info["floor"]
    wall = info["arena"] == info["wall"]
    for peg in info["pegs"]:
        r, c, _ = _landing(wall, peg, (1, 1), info["direction"])
        result[r, c] = info["peg"]
    return result


def _features(info, ablate=False):
    used = Counter(p["group"] for p in info["placements"])
    components = _components(info["mask"])
    nonrect = any((r1-r0)*(c1-c0) != len(cells)
                  for cells in components for r0, c0, r1, c1 in [_box(cells)])
    asymmetric = any(not np.array_equal(g["pattern"], g["pattern"][::-1, ::-1])
                     for i, g in enumerate(info["groups"]) if used[i])
    locally_unused = any(any(p["group"] == i for choices in info["options"] for p in choices)
                         and used[i] < g["count"] for i, g in enumerate(info["groups"]))
    off_axis = False
    wall = info["arena"] == info["wall"]
    dr, dc = info["direction"]
    for p in info["placements"]:
        r, c, h, w = p["rect"]
        # Does another row/column stop the body before its leading corner ray?
        lead = (p["start"][0] + (h-1 if dr > 0 else 0),
                p["start"][1] + (w-1 if dc > 0 else 0))
        point = _landing(wall, lead, (1, 1), (dr, dc))
        off_axis |= point is not None and point[2] > p["distance"]
    b_witness = False
    if ablate:
        outcomes, _ = _unrestricted_packings(info)
        actual = tuple(map(int, info["output"][info["mask"]]))
        b_witness = any(outcome != actual for outcome in outcomes)
    return dict(side=info["side"], direction=info["direction"], shape=info["arena"].shape,
                roles=(info["floor"], info["wall"], info["peg"], info["vacancy"], info["tip"]),
                pegs=len(info["pegs"]), supplied=len(info["tiles"]),
                surplus=len(info["tiles"])-len(info["pegs"]),
                repeats=max(used.values()) > 1, nonrect=nonrect,
                components=len(components), asymmetric=asymmetric,
                local_choices=any(len(x) > 1 for x in info["options"]),
                locally_unused=locally_unused, off_axis=off_axis, b_witness=b_witness,
                distances=tuple(p["distance"] for p in info["placements"]),
                dimensions=tuple(sorted(g["pattern"].shape for g in info["groups"])),
                key_shape=info["key_shape"])


def _wall_mask(rng, h, w, complexity):
    wall = np.zeros((h, w), dtype=bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    if complexity and rng.random() < .65:
        left, right = 1, w-1
        for r in range(1, h-1):
            left = int(np.clip(left + rng.integers(-1, 2), 1, 3))
            right = int(np.clip(right + rng.integers(-1, 2), w-4, w-1))
            wall[r, :left] = True
            wall[r, right:] = True
    for _ in range(complexity):
        th = int(rng.integers(1, max(3, h//2)))
        tw = int(rng.integers(1, min(4, w-3)))
        y = int(rng.integers(1, h-th))
        x = int(rng.integers(3, w-tw))
        if rng.random() < .55:
            y = 0 if rng.random() < .5 else h-th
        wall[y:y+th, x:x+tw] = True
    if complexity >= 3 and rng.random() < .3:
        wall[:, int(rng.integers(5, w-3))] = True
    return wall


def _endpoint_candidates(wall, dimensions):
    result = []
    for h, w in dimensions:
        for y in range(1, wall.shape[0]-h):
            free = ~wall[y:y+h].any(axis=0)
            edges = np.flatnonzero(np.diff(np.r_[False, free, False]))
            for l, end in zip(edges[::2], edges[1::2]):
                if end-l > w:
                    result.append((y, int(end-w), h, w, int(l)))
    return result


def _select_geometry(rng, wall, n, witness=False, repeat=False):
    dimensions = [(h, w) for h in range(1, 5) for w in range(1, 6)]
    candidates = _endpoint_candidates(wall, dimensions)
    if not candidates:
        return None
    selected = []
    if witness:
        # Reserve a freely sampled ambiguous stacked subassembly; other pieces
        # and walls are sampled compositionally rather than from a layout bank.
        lookup = {(r, c, h, w): (r, c, h, w, lo) for r, c, h, w, lo in candidates}
        pairs = []
        for p in candidates:
            r, c, h, w, lo = p
            if h < 2 or w < 2:
                continue
            for h2 in (2, 3, 4):
                q = lookup.get((r+h, c, h2, w))
                if h2 != h and q is not None:
                    pairs.append((p, q))
        if not pairs:
            return None
        selected.extend(pairs[int(rng.integers(len(pairs)))])
    occupied = np.zeros(wall.shape, dtype=bool)
    for r, c, h, w, lo in selected:
        occupied[r:r+h, c:c+w] = True
    while len(selected) < n:
        legal, touching = [], []
        for p in candidates:
            r, c, h, w, lo = p
            if occupied[r:r+h, c:c+w].any():
                continue
            if repeat and len(selected) == 1 and (h, w) != selected[0][2:4]:
                continue
            legal.append(p)
            if occupied[max(0,r-1):r+h+1, max(0,c-1):c+w+1].any():
                touching.append(p)
        choices = touching if touching and (witness or rng.random() < .7) else legal
        if not choices:
            return None
        p = choices[int(rng.integers(len(choices)))]
        selected.append(p)
        r, c, h, w, _ = p
        occupied[r:r+h, c:c+w] = True
    starts = []
    for r, c, h, w, lo in selected:
        choices = [x for x in range(lo, c) if not occupied[r, x] and (r, x) not in starts]
        if not choices:
            return None
        starts.append((r, choices[int(rng.integers(len(choices)))]))
    return selected, starts


def _rotate_rect(rect, shape, k):
    r, c, h, w = rect
    height, width = shape
    for _ in range(k):
        r, c, h, w = width-c-w, r, w, h
        height, width = width, height
    return r, c, h, w


def _pattern(rng, shape, palette, asymmetric=False, base=None):
    a = int(rng.choice(palette)) if base is None else int(base)
    tile = np.full(shape, a, dtype=np.int8)
    if len(palette) == 1:
        return tile
    b = int(rng.choice([x for x in palette if x != a]))
    yy, xx = np.indices(shape)
    mode = int(rng.integers(6))
    if mode == 1:
        tile[yy % 2 == 1] = b
    elif mode == 2:
        tile[xx % 2 == 1] = b
    elif mode == 3:
        tile[(yy+xx) % 2 == 1] = b
    elif mode == 4:
        tile[int(rng.integers(shape[0])), int(rng.integers(shape[1]))] = b
    elif mode == 5:
        tile = rng.choice(palette, size=shape).astype(np.int8)
    if asymmetric:
        tile.fill(a)
        tile[0, 0] = b
    return tile


def _make_key(rng, direction, wall, trail, tip, compact=False):
    length = 2 if compact else int(rng.integers(2, 4))
    gap = 2
    points = []
    dr, dc = direction
    for i in range(2):
        stagger = 0 if compact else int(rng.integers(2))
        for j in range(length+1):
            r, c = (i*gap, 0) if dr == 0 else (0, i*gap)
            points.append((r+(j+stagger)*dr, c+(j+stagger)*dc,
                           tip if j == length else trail))
    r0, c0, r1, c1 = _box([(r, c) for r, c, _ in points])
    pad = [1]*4 if compact else list(map(int, rng.integers(1, 3, size=4)))
    key = np.full((r1-r0+pad[0]+pad[1], c1-c0+pad[2]+pad[3]), wall, dtype=np.int8)
    for r, c, color in points:
        key[r-r0+pad[0], c-c0+pad[2]] = color
    return key


def _scatter(rng, objects, shape, background):
    h, w = shape
    if any(t.shape[0]+2 > h or t.shape[1]+2 > w for t in objects):
        return None
    order = sorted(range(len(objects)), key=lambda i: -objects[i].size)
    for _ in range(10):
        occupied = np.zeros(shape, dtype=bool)
        tray = np.full(shape, background, dtype=np.int8)
        for i in order:
            tile = objects[i]
            th, tw = tile.shape
            windows = np.lib.stride_tricks.sliding_window_view(occupied, (th+2, tw+2))
            valid = np.argwhere(~windows.any(axis=(-1, -2)))
            if not len(valid):
                break
            r, c = map(int, valid[int(rng.integers(len(valid)))]+1)
            occupied[r:r+th, c:c+tw] = True
            tray[r:r+th, c:c+tw] = tile
        else:
            return tray
    # Deterministic shelf attempt is just a bounded fitting fallback for the
    # auxiliary tray. Typical accepted scenes use independent random positions.
    tray = np.full(shape, background, dtype=np.int8)
    y, x, row_h = 1, 1, 0
    for i in sorted(range(len(objects)), key=lambda i: -objects[i].shape[0]):
        tile = objects[i]
        th, tw = tile.shape
        if x+tw > w-1:
            y, x, row_h = y+row_h+1, 1, 0
        if y+th > h-1:
            return None
        tray[y:y+th, x:x+tw] = tile
        x += tw+1
        row_h = max(row_h, th)
    return tray


def _render_candidate(rng, wall_mask, selected, starts, direction, side,
                      surplus, witness, fallback=False, need_body=False):
    k = CARDINAL.index(direction)
    old_shape = wall_mask.shape
    wall_mask = np.rot90(wall_mask, k).copy()
    rects = [_rotate_rect(p[:4], old_shape, k) for p in selected]
    initial = [_rotate_rect((*start, p[2], p[3]), old_shape, k)
               for start, p in zip(starts, selected)]
    if need_body:
        body_witness = False
        dr, dc = direction
        for (r, c, h, w), (sy, sx, _, _) in zip(rects, initial):
            lead = sy+(h-1 if dr > 0 else 0), sx+(w-1 if dc > 0 else 0)
            point = _landing(wall_mask, lead, (1, 1), direction)
            body_witness |= point is not None and point[2] > abs(r-sy)+abs(c-sx)
        if not body_witness:
            return None
    roles = list(map(int, rng.permutation(10)))
    floor, wall, peg, vacancy, tip = roles[:5]
    palette = roles[5:5+int(rng.integers(2 if witness else 1, 6))]
    patterns = {}
    for r, c, h, w in rects:
        if (h, w) not in patterns:
            patterns[h, w] = _pattern(rng, (h, w), palette,
                                       asymmetric=witness and not patterns)
    if fallback and witness:
        # Make the fallback's alternate unrestricted rendering certain rather
        # than relying on two randomly sampled matrices looking different.
        for i, dim in enumerate(patterns):
            patterns[dim] = np.full(dim, palette[i % len(palette)], dtype=np.int8)
        first = next(iter(patterns))
        patterns[first][0, 0] = palette[1]
    arena = np.full(wall_mask.shape, floor, dtype=np.int8)
    arena[wall_mask] = wall
    output = arena.copy()
    mask = np.zeros(arena.shape, dtype=bool)
    tiles = []
    for r, c, h, w in rects:
        tile = patterns[h, w]
        tiles.append(tile.copy())
        mask[r:r+h, c:c+w] = True
        output[r:r+h, c:c+w] = tile
    arena[mask] = vacancy
    for r, c, h, w in initial:
        if mask[r, c] or arena[r, c] == peg:
            return None
        arena[r, c] = peg
    if surplus:
        # A singleton can often dock inside a larger target but cannot replace
        # that target's full area. More varied optional alternatives also occur.
        decoy_dims = [(1, 1)]
        if not fallback and rng.random() < .4 and len(tiles) < 6:
            decoy_dims.append((int(rng.integers(1, 4)), int(rng.integers(1, 4))))
        for dim in decoy_dims:
            if dim in patterns:
                continue
            tile = _pattern(rng, dim, palette)
            tiles.append(tile)
            patterns[dim] = tile
    key = _make_key(rng, direction, wall, peg, tip, fallback)
    common = arena.shape[0] if side in ("left", "right") else arena.shape[1]
    max_other = 30 - (arena.shape[1] if side in ("left", "right") else arena.shape[0])
    other_sizes = list(range(7, max_other+1))
    rng.shuffle(other_sizes)
    for other in other_sizes:
        shape = (common, other) if side in ("left", "right") else (other, common)
        tray = _scatter(rng, tiles+[key], shape, vacancy)
        if tray is None:
            continue
        parts = (arena, tray) if side in ("left", "top") else (tray, arena)
        inp = np.concatenate(parts, axis=1 if side in ("left", "right") else 0)
        pair = {"input": inp, "output": output}
        try:
            info = _infer(inp)
        except (ValueError, AssertionError):
            continue
        if not np.array_equal(info["output"], output):
            continue
        features = _features(info, ablate=witness)
        if any(d < 1 for d in features["distances"]):
            continue
        if need_body and not (features["off_axis"] and features["repeats"]):
            continue
        if witness and not (features["nonrect"] and features["asymmetric"]
                            and features["local_choices"] and features["b_witness"]):
            continue
        if surplus and features["surplus"] == 0:
            continue
        return pair
    return None


def _build_pair(rng, direction, side, n, surplus, witness=False, repeat=False):
    for _ in range(70):
        h, w = int(rng.integers(14, 23)), int(rng.integers(12, 21))
        complexity = int(rng.choice([0, 1, 2, 3, 4], p=[.23, .22, .25, .2, .1]))
        wall = _wall_mask(rng, h, w, complexity)
        geometry = _select_geometry(rng, wall, n, witness, repeat)
        if geometry is None:
            continue
        pair = _render_candidate(rng, wall, *geometry, direction, side, surplus, witness,
                                 need_body=repeat)
        if pair is not None:
            return pair
    # Roomy bounded geometric tail; randomized patterns and tray positions
    # still pass the exact same input-only interpretation and witness checks.
    for _ in range(20):
        wall = np.zeros((18, 16), dtype=bool)
        wall[0] = wall[-1] = True
        wall[:, 0] = wall[:, -1] = True
        dims = ([(2, 3), (3, 3), (2, 4)] if witness else
                [(2, 3), (2, 3)] + [(2, 2+(i % 3)) for i in range(n-2)])
        if witness:
            dims += [(2, 2)] * (n-3)
        selected, starts, y = [], [], 2
        for i, (h, w) in enumerate(dims):
            endpoint = 15-w
            if repeat and i == 0:
                # Place the obstacle away from the leading reference corner's
                # ray, including after changing the projection orientation.
                row = y+h-1 if CARDINAL.index(direction) < 2 else y
                wall[row, 10] = True
                endpoint = 10-w
            selected.append((y, endpoint, h, w, 1))
            starts.append((y, int(rng.integers(1, 4))))
            y += h
        pair = _render_candidate(rng, wall, selected, starts, direction, side,
                                 surplus, witness, fallback=True, need_body=repeat)
        if pair is not None:
            return pair
    raise RuntimeError("bounded construction could not satisfy teaching witnesses")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer(num_train, 2, 10, "num_train")
    nq = _integer(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Perpendicular demonstration directions distinguish horizontal/vertical
    # behavior. Queries either hold out directions or increase the peg count.
    base = int(rng.integers(4))
    train_directions = (CARDINAL[base], CARDINAL[(base+1) % 4])
    held_out = bool(rng.integers(2))
    allow_surplus = rng.random() < .72
    plans = []
    for i in range(nt):
        direction = train_directions[i % 2] if i < 2 else train_directions[int(rng.integers(2))]
        n = 3 if i == 0 else int(rng.integers(2, 5))
        plans.append(("train", direction, n, i == 0, i == 1))
    for _ in range(nq):
        pool = (CARDINAL[(base+2) % 4], CARDINAL[(base+3) % 4]) if held_out else train_directions
        plans.append(("test", pool[int(rng.integers(2))], int(rng.integers(3, 6)) if held_out else 5,
                      False, False))
    puzzle, seen = {"train": [], "test": []}, set()
    for i, (split, direction, n, witness, repeat) in enumerate(plans):
        side = ("left", "right", "top", "bottom")[int(rng.integers(4))]
        surplus = allow_surplus and (i == 0 or rng.random() < .55)
        for _ in range(5):
            pair = _build_pair(rng, direction, side, n, surplus, witness, repeat)
            key = _fingerprint(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("bounded duplicate-input retry exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, features = set(), []
    for i, pair in enumerate(puzzle["train"]+puzzle["test"]):
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
        a, b = _grid(pair["input"]), _grid(pair["output"])
        key = _fingerprint(a)
        _require(key not in seen, "duplicate input")
        seen.add(key)
        try:
            info = _infer(a)
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc
        _require(np.array_equal(info["output"], b), "incorrect docking output")
        arena, mask = info["arena"], info["mask"]
        _require(b.shape == arena.shape and b.size < a.size, "output must be the full cropped arena")
        fixed = ~mask & (arena != info["peg"])
        _require(np.array_equal(b[fixed], arena[fixed]), "walls or unrelated floor changed")
        _require(np.all(b[arena == info["peg"]] == info["floor"]), "pegs were not erased")
        _require(not np.any((b == info["peg"]) | (b == info["vacancy"])), "unfilled docking cells or pegs")
        used = Counter(p["group"] for p in info["placements"])
        _require(sum(used.values()) == len(info["pegs"]), "one stamp is required per peg")
        for g, count in used.items():
            _require(count <= info["groups"][g]["count"], "inventory exhausted")
        for p in info["placements"]:
            r, c, h, w = p["rect"]
            tile = info["groups"][p["group"]]["pattern"]
            _require(np.array_equal(b[r:r+h, c:c+w], tile), "stamp matrix was altered")
            _require(_landing(arena == info["wall"], p["start"], tile.shape,
                              info["direction"]) == (r, c, p["distance"]), "invalid boundary stop")
        _require(not np.array_equal(_point_output(info), b), "point-marker ablation did not differ")
        features.append(_features(info, ablate=i < nt))
    train, test = features[:nt], features[nt:]
    _require(len({f["direction"] for f in train}) >= 2, "need different training directions")
    _require(any(f["nonrect"] and f["asymmetric"] and f["local_choices"] and f["b_witness"]
                 for f in train), "missing coupled packing/clearance teaching witness")
    _require(any(f["repeats"] for f in train), "missing repeated-copy training witness")
    _require(any(f["off_axis"] for f in train), "missing full-body off-axis wall witness")
    _require(not any(f["surplus"] for f in test) or any(f["surplus"] for f in train),
             "test surplus lacks a demonstration")
    train_dirs = {f["direction"] for f in train}
    _require(all(f["direction"] not in train_dirs for f in test)
             or min(f["pegs"] for f in test) > max(f["pegs"] for f in train),
             "missing direction or compositional-load shift")


def self_test():
    """Stream 42 seeds and bounded negative/analytic cases; no artifacts written."""
    signatures, shapes, roles, directions, sides, sizes, key_shapes = (set() for _ in range(7))
    counts = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for pair in puzzle["train"]+puzzle["test"]:
            info = _infer(pair["input"])
            f = _features(info)
            shapes.add(f["shape"])
            roles.add(f["roles"])
            directions.add(f["direction"])
            sides.add(f["side"])
            sizes.update(f["dimensions"])
            key_shapes.add(f["key_shape"])
            for flag in ("nonrect", "repeats", "surplus", "locally_unused", "off_axis"):
                counts[flag] += bool(f[flag])
            counts["separate"] += f["components"] > 1
            counts["all_used"] += f["surplus"] == 0
            counts["five_pegs"] += f["pegs"] == 5
    _require(len(signatures) == 42, "seed diversity")
    _require(len(shapes) >= 25 and len(roles) >= 100, "geometry or local color diversity")
    _require(directions == set(CARDINAL) and len(sides) == 4, "orientation coverage")
    _require(len(sizes) >= 15 and len(key_shapes) >= 10, "stamp/key scale variation")
    _require(all(counts[f] > 0 for f in counts) and counts["off_axis"] >= 5,
             "missing family-specific geometric variation")
    _require(_signature(generate_puzzle(77, 5, 2)) == _signature(generate_puzzle(77, 5, 2)),
             "nondeterministic generation")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # An obstacle outside a reference point's row stops a full rectangle.
    wall = np.zeros((8, 10), dtype=bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    wall[4, 6] = True
    _require(_landing(wall, (3, 1), (2, 3), (0, 1)) == (3, 3, 2), "body clearance")
    _require(_landing(wall, (3, 3), (1, 1), (0, 1)) == (3, 8, 5), "point-ray distinction")
    # Content corruption and loss of a physical copy must be detected.
    puzzle = generate_puzzle(0)
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    info = _infer(bad["test"][0]["input"])
    r, c = next(iter(map(tuple, np.argwhere(info["mask"]))))
    bad["test"][0]["output"][r, c] = info["floor"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    info = _infer(puzzle["train"][1]["input"])
    used = Counter(p["group"] for p in info["placements"])
    repeated = next(g for g, n in used.items() if n > 1)
    tiles = list(info["tiles"])
    target = _fingerprint(info["groups"][repeated]["pattern"])
    tiles.pop(next(i for i, t in enumerate(tiles) if _fingerprint(t) == target))
    altered = dict(info, tiles=tiles)
    outcomes, _, _, _ = _launch_outcomes(altered)
    _require(not outcomes, "missing-copy inventory ablation should destroy the cover")
