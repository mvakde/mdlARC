"""Socket conveyor: an L-key scan feeds solid pieces into later sealed holes.

Only NumPy and the standard library are used. No examples or random state are
consulted by solve(). self_test() keeps its 42-seed measurements in memory.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(((0 <= a) & (a <= 9)).all()), "color bounds")
    return a.astype(np.int8, copy=True)


def _regions(mask, neighbors=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        cells, stack = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(cells)
    return result


def _crop(mask):
    rr, cc = np.nonzero(mask)
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _holes(mask, neighbors=N4):
    padded = np.pad(mask, 1, constant_values=False)
    outside = np.zeros_like(padded)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not padded[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    return (~padded & ~outside)[1:-1, 1:-1]


def _shape_key(mask):
    return mask.shape, mask.astype(np.uint8).tobytes()


def _pockets(mask):
    result = []
    for cells in _regions(_holes(mask)):
        rr, cc = np.asarray(cells).T
        r, c = int(rr.min()), int(cc.min())
        local = np.zeros((int(rr.max()) - r + 1, int(cc.max()) - c + 1), bool)
        local[rr - r, cc - c] = True
        result.append({"r": r, "c": c, "mask": local})
    return sorted(result, key=lambda h: (h["r"], h["c"]))


def _transform(grid, config):
    rotation, mirror = config
    a = np.rot90(grid, rotation)
    return (np.fliplr(a) if mirror else a).copy()


def _inverse(grid, config):
    rotation, mirror = config
    return np.rot90(np.fliplr(grid) if mirror else grid, -rotation).copy()


def _read(input_grid):
    """Locate the visibly notched fast arm, then parse input pixels only."""
    grid = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        for mirror in (False, True):
            config = rotation, mirror
            a = _inverse(grid, config)
            if min(a.shape) < 6:
                continue
            bg, origin, fast, slow = map(int, (a[-1, -1], a[0, 0], a[0, 1], a[1, 0]))
            if len({bg, origin, fast, slow}) != 4:
                continue
            top = np.full(a.shape[1] - 1, fast, np.int8)
            top[1] = bg  # the fast arm has a gap two steps from the origin
            if (np.array_equal(a[0, 1:], top) and np.all(a[1:, 0] == slow)
                    and np.all(a[-1, 1:] == bg) and np.all(a[1:, -1] == bg)):
                candidates.append((a, config, (bg, origin, fast, slow)))
    _require(len(candidates) == 1, "one L-key with a notched fast arm required")
    a, config, roles = candidates[0]
    bg = roles[0]
    body = a != bg
    body[0] = False
    body[:, 0] = False
    objects = []
    _require(len(_regions(body)) == len(_regions(body, N8)), "diagonal object contact")
    for cells in _regions(body):
        rr, cc = np.asarray(cells).T
        r, c = int(rr.min()), int(cc.min())
        h, w = int(rr.max()) - r + 1, int(cc.max()) - c + 1
        _require(r >= 2 and c >= 2 and r + h < len(a) and c + w < a.shape[1], "clear payload margins")
        colors = np.unique(a[rr, cc])
        _require(len(colors) == 1, "each input object is monochrome")
        mask = np.zeros((h, w), bool)
        mask[rr - r, cc - c] = True
        _require(np.array_equal(_holes(mask), _holes(mask, N8)), "ambiguous diagonal enclosure")
        objects.append({"r": r, "c": c, "mask": mask, "color": int(colors[0])})
    objects.sort(key=lambda o: (o["r"], o["c"]))
    _require(bool(objects), "at least one object required")
    lanes = sorted({o["r"] for o in objects})
    for i, row in enumerate(lanes):
        band = [o for o in objects if o["r"] == row]
        end = max(row + o["mask"].shape[0] for o in band)
        if i + 1 < len(lanes):
            _require(lanes[i + 1] >= end + 1, "lanes overlap or have staggered tops")
        for x, y in zip(band, band[1:]):
            _require(y["c"] >= x["c"] + x["mask"].shape[1] + 1, "objects overlap within lane")
    # Input enclosures must belong to one original component, never a collective
    # arrangement or a nested foreground object.
    owned = np.zeros_like(body)
    for o in objects:
        h, w = o["mask"].shape
        local = _holes(o["mask"])
        owned[o["r"]:o["r"] + h, o["c"]:o["c"] + w] |= local
        hs = _pockets(o["mask"])
        for i, x in enumerate(hs):
            for y in hs[i + 1:]:
                _require(x["r"] + x["mask"].shape[0] <= y["r"]
                         or y["r"] + y["mask"].shape[0] <= x["r"]
                         or x["c"] + x["mask"].shape[1] <= y["c"]
                         or y["c"] + y["mask"].shape[1] <= x["c"], "overlapping cavity boxes")
    _require(np.array_equal(owned, _holes(body)), "collective or nested enclosure")
    return {"grid": a, "config": config, "roles": roles, "objects": objects,
            "lanes": tuple(sum(o["r"] == r for o in objects) for r in lanes)}


def _allocate(objects, order=None, policy="fifo", socket_lists=None):
    """FIFO by exact oriented shape; empty sockets do not wait for later pieces."""
    waiting, moves = [], []
    if socket_lists is None:
        socket_lists = [_pockets(o["mask"]) for o in objects]
    if policy == "lookahead":
        waiting = [i for i, holes in enumerate(socket_lists) if not holes]
    if order is None:
        order = range(len(objects))
    previous_lane = None
    for target in order:
        obj = objects[target]
        if policy == "lane_reset" and previous_lane != obj["r"]:
            waiting.clear()
            previous_lane = obj["r"]
        sockets = socket_lists[target]
        if not sockets:
            if policy != "lookahead":
                waiting.append(target)
            continue
        for socket in sockets:
            key = _shape_key(socket["mask"])
            indices = range(len(waiting) - 1, -1, -1) if policy == "lifo" else range(len(waiting))
            for at in indices:
                source = waiting[at]
                donor = objects[source]
                match = _shape_key(donor["mask"]) == key
                if policy == "rotations":
                    match = any(_shape_key(np.rot90(donor["mask"], k)) == key for k in range(4))
                if policy == "color_match":
                    match &= donor["color"] == obj["color"]
                if policy == "ignore_shape" or match:
                    waiting.pop(at)
                    moves.append((source, target, socket))
                    break
    return moves


def _draw_result(canonical, objects, moves, bg):
    out = canonical.copy()
    for source, target, socket in moves:
        donor, host = objects[source], objects[target]
        r, c = donor["r"], donor["c"]
        h, w = donor["mask"].shape
        out[r:r + h, c:c + w][donor["mask"]] = bg
        r, c = host["r"] + socket["r"], host["c"] + socket["c"]
        h, w = socket["mask"].shape
        out[r:r + h, c:c + w][socket["mask"]] = donor["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone, including the guide's geometric convention."""
    d = _read(input_grid)
    result = _draw_result(d["grid"], d["objects"], _allocate(d["objects"]), d["roles"][0])
    return _transform(result, d["config"])


def _catalog():
    # A varied supply of compact connected shapes; the solver does not use it.
    bases = (('1',), ('11',), ('111',), ('11', '11'), ('11', '10'),
             ('111', '100'), ('111', '110'), ('111', '010'),
             ('110', '011'), ('101', '111'), ('111', '111'),
             ('110', '010', '011'), ('111', '100', '100'),
             ('010', '111', '010'), ('111', '010', '010'))
    unique = {}
    for rows in bases:
        base = np.array([[c == '1' for c in row] for row in rows], bool)
        for rotation in range(4):
            for flip in (False, True):
                shape = np.rot90(base, rotation)
                shape = _crop(np.fliplr(shape) if flip else shape)
                if len(_regions(shape)) == 1 and not _holes(shape).any():
                    unique[_shape_key(shape)] = shape
    return tuple(unique[k] for k in sorted(unique))


SHAPES = _catalog()


def _small(rng, exclude=None, asymmetric=False):
    choices = [m for m in SHAPES if _shape_key(m) != exclude and
               (not asymmetric or (m.shape[0] > 1 and m.shape[1] > 1
                                    and not np.array_equal(m, m[::-1])
                                    and not np.array_equal(m, m[:, ::-1])))]
    return choices[int(rng.integers(len(choices)))].copy()


def _host(rng, shapes, opened=False, aligned=False):
    """Carve exact negative shapes into uneven walls, optionally with one door."""
    vertical = len(shapes) > 1 and bool(rng.integers(2))
    top = 1 if opened else int(rng.integers(1, 3))
    bottom, left, right = map(int, rng.integers(1, 3, size=3))
    if opened:
        bottom = left = right = 1
    wall = int(rng.integers(1, 3))
    if vertical:
        h = sum(m.shape[0] for m in shapes) + wall * (len(shapes) - 1) + top + bottom
        w = max(m.shape[1] for m in shapes) + left + right
    else:
        h = max(m.shape[0] for m in shapes) + top + bottom
        w = sum(m.shape[1] for m in shapes) + wall * (len(shapes) - 1) + left + right
    mask = np.ones((h, w), bool)
    cursor = top if vertical else left
    for m in shapes:
        if vertical:
            r, c = cursor, left + (0 if aligned else int(rng.integers(w - left - right - m.shape[1] + 1)))
            cursor += m.shape[0] + wall
        else:
            r, c = top + (0 if aligned else int(rng.integers(h - top - bottom - m.shape[0] + 1))), cursor
            cursor += m.shape[1] + wall
        # Open contrasts use a cavity that reaches the first interior row.
        if opened:
            r = 1
        mask[r:r + m.shape[0], c:c + m.shape[1]] &= ~m
    sealed = _holes(mask)
    # Independent exterior silhouette variation without changing a socket.
    # Bounding dimensions stay fixed, which also bounds fallback packing.
    for corner in rng.permutation(4)[:int(rng.integers(3))]:
        candidate = mask.copy()
        rh, cw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        rs = slice(0, rh) if corner < 2 else slice(h - rh, h)
        cs = slice(0, cw) if corner % 2 == 0 else slice(w - cw, w)
        candidate[rs, cs] = False
        if (len(_regions(candidate)) == 1 and _crop(candidate).shape == mask.shape
                and np.array_equal(_holes(candidate), sealed)
                and np.array_equal(_holes(candidate, N8), sealed)):
            mask = candidate
    if opened:
        # All four doorway directions are eligible. With one-cell walls a
        # single boundary-cell change distinguishes the open and closed forms.
        candidates = []
        for r, c in np.argwhere(sealed):
            if r == 1:
                candidates.append((0, int(c)))
            if r == h - 2:
                candidates.append((h - 1, int(c)))
            if c == 1:
                candidates.append((int(r), 0))
            if c == w - 2:
                candidates.append((int(r), w - 1))
        rng.shuffle(candidates)
        for r, c in candidates:
            candidate = mask.copy()
            candidate[r, c] = False
            if (len(_regions(candidate)) == 1 and not _holes(candidate).any()
                    and not _holes(candidate, N8).any() and _crop(candidate).shape == mask.shape):
                mask = candidate
                break
        else:
            raise AssertionError("no connected one-cell doorway")
    _require(len(_regions(mask)) == 1, "host construction disconnected")
    _require(len(_pockets(mask)) == (0 if opened else len(shapes)), "host topology construction")
    return mask


def _events(rng, palette, kind, max_holes):
    p = _small(rng, asymmetric=True)
    q = _small(rng, exclude=_shape_key(p))
    colors = list(map(int, rng.permutation(palette)))
    def obj(mask, color=None):
        return {"mask": mask.copy(), "color": int(rng.choice(palette)) if color is None else color}
    if kind == 0:
        # Two equal pieces, an open distractor before the true molds, and a
        # piece arriving too late for its otherwise compatible mold.
        return [obj(p, colors[0]), obj(p, colors[1]), obj(_host(rng, [p], True), colors[2]),
                obj(_host(rng, [p]), colors[2]), obj(_host(rng, [p]), colors[2]),
                obj(_host(rng, [q]), colors[2]), obj(q, colors[3])]
    if kind == 1:
        q = next(np.rot90(p, k).copy() for k in (1, 2, 3) if _shape_key(np.rot90(p, k)) != _shape_key(p))
        shapes = [p, q]
        if max_holes == 3:
            others = [m for m in SHAPES if _shape_key(m) not in {_shape_key(p), _shape_key(q)}]
            shapes.append(others[int(rng.integers(len(others)))].copy())
        # Reverse the supply order relative to cavity order: take a matching
        # shape rather than blindly taking the next color in the queue.
        donors = [obj(m, colors[i % len(colors)]) for i, m in enumerate(shapes)]
        return donors[::-1] + [obj(_host(rng, shapes, aligned=True)), obj(p, colors[3]),
                               obj(_host(rng, [p])), obj(_host(rng, [q], True))]
    bank = [p, q, _small(rng)]
    events = [obj(p), obj(_host(rng, [p]))]
    for _ in range(int(rng.integers(2, 6))):
        events.append(obj(bank[int(rng.integers(len(bank)))]))
    for i in range(int(rng.integers(0, 3))):
        count = int(rng.integers(1, max_holes + 1)) if i == 0 else 1
        holes = [bank[int(rng.integers(len(bank)))] for _ in range(count)]
        events.append(obj(_host(rng, holes)))
    if len(events) < 9 and rng.random() < .5:
        events.append(obj(_host(rng, [q], True)))
    rng.shuffle(events)
    if not _allocate(events):
        # The guaranteed p piece and p host exist. Move one p piece before
        # its host without imposing an absolute first/last object template.
        donor = next(i for i, o in enumerate(events) if _shape_key(o["mask"]) == _shape_key(p))
        host = next(i for i, o in enumerate(events)
                    if any(_shape_key(h["mask"]) == _shape_key(p) for h in _pockets(o["mask"])))
        if donor > host:
            events[donor], events[host] = events[host], events[donor]
    return events


def _partitions(events, margin, gap):
    n = len(events)
    result = []
    for bits in range(1 << (n - 1)):
        groups, start = [], 0
        for j in range(n - 1):
            if bits & (1 << j):
                groups.append(list(range(start, j + 1)))
                start = j + 1
        groups.append(list(range(start, n)))
        widths = [sum(events[i]["mask"].shape[1] for i in group) + gap * (len(group) - 1)
                  for group in groups]
        heights = [max(events[i]["mask"].shape[0] for i in group) for group in groups]
        w = max(widths) + 2 * margin
        h = sum(heights) + gap * (len(groups) - 1) + 2 * margin
        if w <= 29 and h <= 29:
            result.append((groups, widths, heights, h, w))
    return result


def _pack(rng, events, roles, force_witness=False):
    bg, origin, fast, slow = roles
    margin, gap = int(rng.integers(1, 3)), int(rng.integers(1, 4))
    choices = _partitions(events, margin, gap)
    if not choices:
        margin, gap = 1, 1
        choices = _partitions(events, margin, gap)
    if not choices:
        return None
    if force_witness:
        # Bounded layout fallback for the seven-object first teaching scene.
        margin, gap = 1, 1
        groups = [list(range(3)), list(range(3, 7))]
        widths = [sum(events[i]["mask"].shape[1] for i in g) + len(g) - 1 for g in groups]
        heights = [max(events[i]["mask"].shape[0] for i in g) for g in groups]
        w, h = 29, sum(heights) + 3
        _require(max(widths) <= 27 and h <= 29, "teaching fallback fit")
    else:
        groups, widths, heights, h, w = choices[int(rng.integers(len(choices)))]
        h += int(rng.integers(min(4, 29 - h) + 1))
        w += int(rng.integers(min(6, 29 - w) + 1))
    canvas = np.full((h + 1, w + 1), bg, np.int8)
    canvas[0, 0], canvas[0, 1:], canvas[1:, 0] = origin, fast, slow
    canvas[0, 2] = bg
    spare_y = h - (sum(heights) + gap * (len(groups) - 1) + 2 * margin)
    y_extra = rng.multinomial(spare_y, np.full(len(groups) + 1, 1 / (len(groups) + 1)))
    row = 1 + margin + int(y_extra[0])
    placed = []
    for gi, group in enumerate(groups):
        spare = w - 2 * margin - widths[gi]
        extras = rng.multinomial(spare, np.full(len(group) + 1, 1 / (len(group) + 1)))
        if force_witness:
            extras[:] = 0
            extras[0 if gi == 0 else -1] = spare
        col = 1 + margin + int(extras[0])
        for j, index in enumerate(group):
            o = dict(events[index], r=row, c=col)
            mh, mw = o["mask"].shape
            canvas[row:row + mh, col:col + mw][o["mask"]] = o["color"]
            placed.append(o)
            col += mw + gap + int(extras[j + 1])
        row += heights[gi] + gap + int(y_extra[gi + 1])
    return canvas, placed


def _alternative_orders(objects):
    return {"reverse_fast": sorted(range(len(objects)), key=lambda i: (objects[i]["r"], -objects[i]["c"])),
            "reverse_slow": sorted(range(len(objects)), key=lambda i: (-objects[i]["r"], objects[i]["c"])),
            "fast_major": sorted(range(len(objects)), key=lambda i: (objects[i]["c"], objects[i]["r"]))}


def _distinguishes(canvas, objects, bg):
    socket_lists = [_pockets(o["mask"]) for o in objects]
    expected = _draw_result(canvas, objects, _allocate(objects, socket_lists=socket_lists), bg)
    result = {name: not np.array_equal(expected, _draw_result(canvas, objects, _allocate(objects, order, socket_lists=socket_lists), bg))
              for name, order in _alternative_orders(objects).items()}
    for name in ("lifo", "ignore_shape", "lookahead", "lane_reset", "color_match", "rotations"):
        result[name] = not np.array_equal(expected, _draw_result(canvas, objects, _allocate(objects, policy=name, socket_lists=socket_lists), bg))
    return result


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    fast, slow = map(int, rng.choice(10, 2, replace=False))
    max_holes = int(rng.integers(2, 4))
    first = int(rng.integers(4)), bool(rng.integers(2))
    taught = [first, ((first[0] + 1) % 4, not first[1])]
    held = [(r, m) for r in range(4) for m in (False, True) if (r, m) not in taught]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            kind = i if split == "train" and i < 2 else 2
            config = taught[i % 2] if split == "train" else held[int(rng.integers(len(held)))]
            local = [c for c in range(10) if c not in (fast, slow)]
            bg, origin = map(int, rng.choice(local, 2, replace=False))
            pool = [c for c in range(10) if c != bg]
            palette = list(map(int, rng.choice(pool, int(rng.integers(4, 10)), replace=False)))
            for attempt in range(40):
                events = _events(rng, palette, kind, max_holes)
                packed = _pack(rng, events, (bg, origin, fast, slow), force_witness=(kind == 0 and attempt == 39))
                if packed is None:
                    continue
                canvas, objects = packed
                if kind == 0:
                    evidence = _distinguishes(canvas, objects, bg)
                    if not all(value for name, value in evidence.items() if name != "rotations"):
                        continue
                inp = _transform(canvas, config)
                key = _shape_key(inp)
                if key in seen:
                    continue
                moves = _allocate(objects)
                if not moves:
                    continue
                out = _transform(_draw_result(canvas, objects, moves, bg), config)
                puzzle[split].append({"input": inp, "output": out})
                seen.add(key)
                break
            else:
                raise RuntimeError("bounded scene construction failed")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _door_witness(objects):
    """A solid with a one-cell doorway conceals a compatible closed socket."""
    preceding = []
    for o in objects:
        mask = o["mask"]
        if _pockets(mask):
            continue
        h, w = mask.shape
        for r, c in np.argwhere(~mask):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            closed = mask.copy()
            closed[r, c] = True
            for pocket in _pockets(closed):
                if _shape_key(pocket["mask"]) in preceding:
                    return True
        preceding.append(_shape_key(mask))
    return False


def _facts(d):
    objects = d["objects"]
    sockets = [_pockets(o["mask"]) for o in objects]
    moves = _allocate(objects)
    source_ids = {m[0] for m in moves}
    empty_later_match = False
    allocated = {(t, h["r"], h["c"]) for _, t, h in moves}
    for t, holes in enumerate(sockets):
        for h in holes:
            if (t, h["r"], h["c"]) not in allocated:
                empty_later_match |= any(not sockets[s] and s not in source_ids and
                                        _shape_key(objects[s]["mask"]) == _shape_key(h["mask"])
                                        for s in range(t + 1, len(objects)))
    return {"objects": len(objects), "lanes": d["lanes"], "moves": len(moves),
            "holes": sum(map(len, sockets)), "max_holes": max(map(len, sockets)),
            "open_door": _door_witness(objects), "late_piece": bool(empty_later_match),
            "piece_shapes": {_shape_key(o["mask"]) for o, hs in zip(objects, sockets) if not hs},
            "unfilled": sum(map(len, sockets)) - len(moves)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "list splits")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, guides, training, configs = set(), set(), [], {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions changed")
            key = _shape_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _read(inp)
            guides.add(d["roles"][2:])
            configs[split].add(d["config"])
            objects, bg = d["objects"], d["roles"][0]
            moves = _allocate(objects)
            _require(bool(moves), "generated pair must demonstrate a transfer")
            _require(len({s for s, _, _ in moves}) == len(moves), "piece reused")
            _require(all(s < t for s, t, _ in moves), "future piece consumed")
            expected = _transform(_draw_result(d["grid"], objects, moves, bg), d["config"])
            _require(np.array_equal(expected, out), "incorrect ordered socket transfers")
            _require(np.array_equal(np.bincount(inp.ravel(), minlength=10), np.bincount(out.ravel(), minlength=10)),
                     "per-color cell conservation failed")
            canonical_out = _inverse(out, d["config"])
            _require(np.array_equal(canonical_out[0], d["grid"][0])
                     and np.array_equal(canonical_out[:, 0], d["grid"][:, 0]), "key changed")
            for o in objects:
                if _pockets(o["mask"]):
                    r, c = o["r"], o["c"]
                    h, w = o["mask"].shape
                    _require(np.all(canonical_out[r:r + h, c:c + w][o["mask"]] == o["color"]),
                             "original hollow body changed")
            if split == "train":
                training.append((d, _facts(d)))
    _require(len(guides) == 1, "guide palette inconsistent")
    _require(not configs["train"] & configs["test"], "orientation combination shift missing")
    _require(any(f["open_door"] and f["late_piece"] for _, f in training), "open/closed and causal evidence missing")
    _require(any(f["max_holes"] >= 2 for _, f in training), "multi-cavity evidence missing")
    alternatives = [_distinguishes(d["grid"], d["objects"], d["roles"][0]) for d, _ in training]
    for name in ("reverse_fast", "reverse_slow", "fast_major", "lifo", "ignore_shape", "lookahead",
                 "lane_reset", "color_match", "rotations"):
        _require(any(row[name] for row in alternatives),
                 "shortcut undistinguished: " + name)
    # At least one repeated shape is supplied in two colors and used twice.
    fifo = False
    for d, _ in training:
        moved = [d["objects"][s] for s, _, _ in _allocate(d["objects"])]
        fifo |= any(_shape_key(a["mask"]) == _shape_key(b["mask"]) and a["color"] != b["color"]
                    for i, a in enumerate(moved) for b in moved[i + 1:])
    _require(fifo, "FIFO color competition not demonstrated")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded suite: 210 default pairs, count boundaries and small regressions."""
    dimensions, layouts, shapes, configurations, palettes, puzzles = set(), set(), set(), set(), set(), set()
    populations, moves, hole_counts = set(), Counter(), set()
    open_count = late_count = unfilled_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_digest(puzzle))
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), "nondeterministic seed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _read(pair["input"])
                f = _facts(d)
                dimensions.add(pair["input"].shape)
                layouts.add((d["grid"].shape, tuple((o["r"], o["c"], _shape_key(o["mask"])) for o in d["objects"])))
                configurations.add(d["config"])
                palettes.add(d["roles"])
                shapes.update(f["piece_shapes"])
                populations.add(f["lanes"])
                moves[f["moves"]] += 1
                hole_counts.add(f["max_holes"])
                open_count += f["open_door"]
                late_count += f["late_piece"]
                unfilled_count += bool(f["unfilled"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "input-only solve disagreement")
                # Whole-scene symmetries must carry the output with them.
                if split == "test":
                    for config in ((1, False), (0, True)):
                        _require(np.array_equal(solve(_transform(pair["input"], config)),
                                                _transform(pair["output"], config)), "symmetry inconsistency")
    _require(len(puzzles) == 42 and len(layouts) >= 200, "cross-seed structure variation")
    _require(len(dimensions) >= 35 and len(shapes) >= 35 and len(populations) >= 15, "geometric variation")
    _require(len(configurations) == 8 and len(palettes) >= 100, "key variation")
    _require({1, 2, 3} <= hole_counts and len(moves) >= 4, "allocation variation")
    _require(open_count >= 42 and late_count >= 42 and unfilled_count >= 42, "topology/causality witnesses")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(42000 + nt * 10 + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Hand-built tiny allocation regression: shape equality is oriented;
    # repeated same-shape pieces use FIFO, and a late piece cannot fill back.
    p = np.array([[1, 1], [1, 0]], bool)
    q = np.ones((1, 2), bool)
    rng = np.random.default_rng(123)
    events = [{"mask": p, "color": 3}, {"mask": p, "color": 7},
              {"mask": _host(rng, [p], True), "color": 5},
              {"mask": _host(rng, [p]), "color": 5},
              {"mask": _host(rng, [q]), "color": 5},
              {"mask": _host(rng, [p]), "color": 5}, {"mask": q, "color": 8}]
    allocated = _allocate(events)
    _require([(s, t) for s, t, _ in allocated] == [(0, 3), (1, 5)], "FIFO/open/late regression")
    _require(_door_witness(events), "open door regression")
    _require(_shape_key(p) != _shape_key(np.rot90(p)), "oriented shape regression")
    damaged = generate_puzzle(9123, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "default_pairs": 210, "distinct_puzzles": len(puzzles),
            "input_shapes": len(dimensions), "layouts": len(layouts), "solid_shapes": len(shapes),
            "lane_populations": len(populations), "key_configurations": len(configurations),
            "role_palettes": len(palettes), "move_histogram": dict(sorted(moves.items())),
            "max_holes_per_object": sorted(hole_counts), "open_door_pairs": open_count,
            "late_piece_pairs": late_count, "unfilled_socket_pairs": unfilled_count}
