"""Finite relay printers. Only standard library and NumPy are required.

No parent imports, saved examples, hidden palette maps, or file I/O. The only
execution history retained by the simulator is an in-memory list of station IDs.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    pending = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while pending:
        start = min(pending)
        pending.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in pending:
                    pending.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _crop(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "empty stencil")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _port_side(port, h, w):
    sides = [d for d, ok in enumerate((all(r == 0 for r, c in port),
             all(c == w - 1 for r, c in port), all(r == h - 1 for r, c in port),
             all(c == 0 for r, c in port))) if ok]
    _require(len(sides) == 1, "ambiguous port side")
    d = sides[0]
    span = sorted(c if d % 2 == 0 else r for r, c in port)
    _require(span == list(range(span[0], span[0] + len(span))), "disconnected port")
    _require(0 < span[0] and span[-1] < (w if d % 2 == 0 else h) - 1, "corner port")
    return d


def _parse(input_grid):
    """Recover roles and local instructions using the input alone."""
    inp = _grid(input_grid)
    h, w = inp.shape
    bg = int(inp[0, 0])
    _require(all(int(inp[q]) == bg for q in ((0, w-1), (h-1, 0), (h-1, w-1))), "background corners disagree")
    boundary = [(r, c) for r, c in np.argwhere(inp != bg)
                if r in (0, h-1) or c in (0, w-1)]
    _require(len(boundary) == 1, "expected one boundary launcher")
    start = tuple(map(int, boundary[0]))
    initial = 2 if start[0] == 0 else 0 if start[0] == h-1 else 1 if start[1] == 0 else 3
    dr, dc = DIRS[initial]
    primary = int(inp[start])
    tail, q = [], start
    while 0 <= q[0] < h and 0 <= q[1] < w and int(inp[q]) == primary:
        tail.append(q)
        q = (q[0]+dr, q[1]+dc)
    _require(2 <= len(tail) <= 4 and 0 <= q[0] < h and 0 <= q[1] < w, "invalid tail")
    cap, secondary = q, int(inp[q])
    _require(secondary not in (bg, primary), "missing cap swatch")
    after = (q[0]+dr, q[1]+dc)
    _require(0 <= after[0] < h and 0 <= after[1] < w and int(inp[after]) == bg, "launcher needs a gap")
    parses = []
    for frame in sorted(set(map(int, np.unique(inp))) - {bg, primary, secondary}):
        try:
            rooms, used_ports, stencil_colors, counter_colors = [], set(), set(), set()
            for comp in _components((inp == frame) | (inp == secondary)):
                if not any(int(inp[q]) == frame for q in comp):
                    continue
                r0, r1 = min(r for r, c in comp), max(r for r, c in comp) + 1
                c0, c1 = min(c for r, c in comp), max(c for r, c in comp) + 1
                _require(0 < r0 < r1 < h and 0 < c0 < c1 < w, "station touches boundary")
                box = inp[r0:r1, c0:c1].copy()
                bh, bw = box.shape
                _require(min(bh, bw) >= 5, "station too small")
                cuts = [(False, k) for k in range(2, bw-2) if np.all(box[1:-1, k] == frame)]
                cuts += [(True, k) for k in range(2, bh-2) if np.all(box[k, 1:-1] == frame)]
                _require(len(cuts) == 1, "station needs one chamber wall")
                transposed, k = cuts[0]
                scaffold = np.zeros_like(box, dtype=bool)
                scaffold[[0, -1], :] = True
                scaffold[:, [0, -1]] = True
                if transposed:
                    scaffold[k, :] = True
                    bounds = ((1, k, 1, bw-1), (k+1, bh-1, 1, bw-1))
                else:
                    scaffold[:, k] = True
                    bounds = ((1, bh-1, 1, k), (1, bh-1, k+1, bw-1))
                _require(np.array_equal((box == frame) | (box == secondary), scaffold), "broken station frame")
                port = sorted(map(tuple, np.argwhere(box == secondary)))
                _require(bool(port), "station lacks a port")
                direction = _port_side(port, bh, bw)
                panels = [box[a:b, c:d] for a, b, c, d in bounds]
                inks = []
                for panel in panels:
                    colors = set(map(int, np.unique(panel))) - {bg}
                    _require(len(colors) == 1, "chamber needs one ink")
                    inks.append(next(iter(colors)))
                _require(len({bg, primary, secondary, frame, *inks}) == 6, "roles must be distinct")
                tile = _crop(panels[0] == inks[0])
                _require(len(_components(tile)) == 1, "stencil must be connected")
                counter = panels[1] == inks[1]
                count = len(_components(counter))
                _require(count == len(_components(counter, True)), "diagonal counter ambiguity")
                _require(len(port) == tile.shape[1 if direction % 2 == 0 else 0], "port width and stencil disagree")
                rooms.append(dict(grid=box, top=int(r0), left=int(c0), port=port,
                                  out_dir=direction, tile=tile, count=count,
                                  counter_bounds=bounds[1], stencil_bounds=bounds[0],
                                  area=int(counter.sum()), transposed=transposed))
                used_ports.update((r0+r, c0+c) for r, c in port)
                stencil_colors.add(inks[0])
                counter_colors.add(inks[1])
            _require(bool(rooms) and len(stencil_colors) == len(counter_colors) == 1, "inconsistent station inks")
            _require(used_ports | {cap} == set(map(tuple, np.argwhere(inp == secondary))), "stray port ink")
            palette = (bg, primary, secondary, frame, next(iter(stencil_colors)), next(iter(counter_colors)))
            occupied = set(tail) | {cap}
            for room in rooms:
                rr, cc = room['top'], room['left']
                bh, bw = room['grid'].shape
                _require(not any(rr-1 <= r <= rr+bh and cc-1 <= c <= cc+bw
                                 for r, c in tail+[cap]), "launcher touches a station")
                occupied.update((rr+r, cc+c) for r, c in np.argwhere(room['grid'] != bg))
            _require(occupied == set(map(tuple, np.argwhere(inp != bg))), "unexplained ink")
            for i, room in enumerate(rooms):
                _require(all(_apart(room, other) for other in rooms[i+1:]), "stations touch")
            parses.append(dict(input=inp, palette=palette, rooms=rooms, tail=tail,
                               cap=cap, initial=initial))
        except (ValueError, IndexError):
            continue
    _require(len(parses) == 1, "no unique station parse")
    return parses[0]


def _apart(a, b):
    ar, ac, ah, aw = a['top'], a['left'], *a['grid'].shape
    br, bc, bh, bw = b['top'], b['left'], *b['grid'].shape
    return ar+ah < br or br+bh < ar or ac+aw < bc or bc+bw < ac


def _length(room):
    axial = room['tile'].shape[0 if room['out_dir'] % 2 == 0 else 1]
    return room['count'] * (axial+1) - 1


def _run(scene, *, unlimited=False, one_copy=False, straight=False, rotate=False):
    """Paint one slice at a time, querying original station rectangles.

    Optional switches are deliberately wrong models used only by validation.
    No output or generation state is needed for traversal.
    """
    inp, rooms = scene['input'], scene['rooms']
    bg, primary, secondary = scene['palette'][:3]
    out = np.full_like(inp, bg)
    owner = np.full(inp.shape, -1, dtype=np.int16)
    for i, room in enumerate(rooms):
        r, c = room['top'], room['left']
        h, w = room['grid'].shape
        owner[r:r+h, c:c+w] = i
    for q in scene['tail'] + [scene['cap']]:
        out[q] = primary
    protected = set(scene['tail']) | {scene['cap']}
    origins, direction, tile, budget = [scene['cap']], scene['initial'], None, max(inp.shape)
    reached, segments = [], []
    for _ in range(len(rooms)+1):
        dr, dc = DIRS[direction]
        hit, painted = None, 0
        for step in range(1, budget+1):
            front = [(r+step*dr, c+step*dc) for r, c in origins]
            if not all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] for r, c in front):
                return out, dict(reached=reached, segments=segments+[painted], stop='boundary')
            targets = {int(owner[q]) for q in front if owner[q] >= 0}
            if targets:
                _require(len(targets) == 1 and all(int(owner[q]) == next(iter(targets)) for q in front), "partial collision")
                hit = next(iter(targets))
                _require(hit not in reached, "station cycle")
                break
            if tile is None:
                values = [primary]
            else:
                axial = tile.shape[0 if direction % 2 == 0 else 1]
                phase = (step-1) % (axial+1)
                if phase == axial:
                    values = [secondary] * len(front)
                else:
                    index = axial-1-phase if direction in (0, 3) else phase
                    ink = tile[index, :] if direction % 2 == 0 else tile[:, index]
                    values = np.where(ink, primary, secondary)
            for q, color in zip(front, values):
                if q not in protected:
                    out[q] = color
            painted += 1
        segments.append(painted)
        if hit is None:
            return out, dict(reached=reached, segments=segments, stop='budget')
        reached.append(hit)
        room = rooms[hit]
        r, c = room['top'], room['left']
        h, w = room['grid'].shape
        region = out[r:r+h, c:c+w]
        region[room['grid'] != bg] = primary
        outgoing = direction if straight else room['out_dir']
        if straight and outgoing != room['out_dir']:
            # A counterfactual straight beam uses a same-width centered exit.
            span = room['tile'].shape[1 if outgoing % 2 == 0 else 0]
            extent = w if outgoing % 2 == 0 else h
            start = (extent-span)//2
            local = [(0 if outgoing == 0 else h-1, start+k) for k in range(span)] if outgoing % 2 == 0 else [(start+k, w-1 if outgoing == 1 else 0) for k in range(span)]
        else:
            local = room['port']
        origins = sorted((r+x, c+y) for x, y in local)
        direction = outgoing
        tile = room['tile'].copy()
        if rotate:
            # A plausible wrong model rotates motifs 180 degrees with travel.
            tile = np.rot90(tile, 2)
        axial = tile.shape[0 if direction % 2 == 0 else 1]
        count = 1 if one_copy else room['count']
        budget = max(inp.shape) if unlimited else count * (axial+1) - 1
    raise ValueError('unterminated route')


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only deterministic solution, including negative-direction prints."""
    return _run(_parse(input_grid))[0]


def _grow(h, w, area, rng):
    mask = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        options = sorted(frontier)
        r, c = options[int(rng.integers(len(options)))]
        frontier.remove((r, c))
        mask[r, c] = True
        for dr, dc in DIRS:
            q = (r+dr, c+dc)
            if 0 <= q[0] < h and 0 <= q[1] < w and not mask[q]:
                frontier.add(q)
    return mask


def _stencil(h, w, rng, witness=False):
    for _ in range(60):
        mode = int(rng.integers(4))
        if mode == 0:
            mask = np.ones((h, w), dtype=bool)
            if min(h, w) > 2:
                mask[1:-1, 1:-1] = False
        elif mode == 1:
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = mask[:, 0] = True
            for r, c in np.argwhere(~mask):
                if rng.random() < .35:
                    mask[r, c] = True
        elif mode == 2:
            mask = _grow(h, w, int(rng.integers(max(h, w), h*w+1)), rng)
        else:
            mask = np.ones((h, w), dtype=bool)
        mask = _crop(mask)
        if mask.shape != (h, w) or len(_components(mask)) != 1:
            continue
        if witness and (mask.all() or any(np.array_equal(mask, np.rot90(mask[:, ::-1] if k >= 4 else mask, k % 4)) for k in range(1, 8))):
            continue
        return mask
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = mask[:, 0] = True
    if min(h, w) > 1:
        mask[-1, 1] = True
    return mask


def _partition(length, count, rng):
    sizes = np.ones(count, dtype=int)
    for _ in range(length-(2*count-1)):
        sizes[int(rng.integers(count))] += 1
    result, offset = [], 0
    for n in sizes:
        result.append((offset, offset+int(n)))
        offset += int(n)+1
    return result


def _counter(h, w, count, rng, area=None):
    options = [(nr, nc) for nr in range(1, (h+1)//2+1) for nc in range(1, (w+1)//2+1)
               if count <= nr*nc <= count+2]
    for _ in range(80):
        nr, nc = options[int(rng.integers(len(options)))]
        boxes = [(r0, r1, c0, c1) for r0, r1 in _partition(h, nr, rng)
                 for c0, c1 in _partition(w, nc, rng)]
        rng.shuffle(boxes)
        boxes = boxes[:count]
        capacities = [(b-a)*(d-c) for a, b, c, d in boxes]
        if area is not None and sum(capacities) < area:
            continue
        target = int(rng.integers(count, sum(capacities)+1)) if area is None else area
        areas = [1]*count
        for _ in range(target-count):
            available = [i for i in range(count) if areas[i] < capacities[i]]
            areas[int(rng.choice(available))] += 1
        mask = np.zeros((h, w), dtype=bool)
        for (a, b, c, d), size in zip(boxes, areas):
            mask[a:b, c:d] = _grow(b-a, d-c, size, rng)
        return mask
    raise ValueError('counter area cannot fit')


def _make_room(rng, palette, direction, count, *, tile=None,
               area=None, transposed=None):
    bg, primary, secondary, frame, ink, tally = palette
    if tile is None:
        h, w = map(int, rng.integers(1, 5, size=2))
        tile = _stencil(h, w, rng)
    if transposed is None:
        transposed = bool(rng.integers(2))
    t = tile.T if transposed else tile
    ih = max(t.shape[0] + int(rng.integers(0, 3)), 5 if count >= 4 else 3)
    iw = t.shape[1] + int(rng.integers(0, 3))
    cw = int(rng.integers(5, 7))
    box = np.full((ih+2, iw+cw+3), bg, dtype=np.int8)
    box[[0, -1], :] = frame
    box[:, [0, -1]] = frame
    wall = iw+1
    box[:, wall] = frame
    r, c = int(rng.integers(ih-t.shape[0]+1)), int(rng.integers(iw-t.shape[1]+1))
    box[1+r:1+r+t.shape[0], 1+c:1+c+t.shape[1]][t] = ink
    cm = _counter(ih, cw, count, rng, area)
    box[1:-1, wall+1:-1][cm] = tally
    sb, cb = (1, ih+1, 1, wall), (1, ih+1, wall+1, box.shape[1]-1)
    if transposed:
        box = box.T.copy()
        sb = (sb[2], sb[3], sb[0], sb[1])
        cb = (cb[2], cb[3], cb[0], cb[1])
    h, w = box.shape
    span = tile.shape[1 if direction % 2 == 0 else 0]
    start = int(rng.integers(1, (w if direction % 2 == 0 else h)-span))
    port = [(0 if direction == 0 else h-1, start+k) for k in range(span)] if direction % 2 == 0 else [(start+k, w-1 if direction == 1 else 0) for k in range(span)]
    for q in port:
        box[q] = secondary
    return dict(grid=box, tile=tile.copy(), count=count, port=port, out_dir=direction,
                counter_bounds=cb, stencil_bounds=sb, area=int(cm.sum()), transposed=transposed)


def _change_count(room, count, rng, palette, area=None):
    result = dict(room, grid=room['grid'].copy(), count=count)
    a, b, c, d = room['counter_bounds']
    cm = _counter(b-a, d-c, count, rng, area)
    result['grid'][a:b, c:d] = np.where(cm, palette[5], palette[0])
    result['area'] = int(cm.sum())
    return result


def _place_next(previous, room, gap, rng):
    d = previous['out_dir']
    port = sorted((previous['top']+r, previous['left']+c) for r, c in previous['port'])
    h, w = room['grid'].shape
    span = len(port)
    extent = w if d % 2 == 0 else h
    if span > extent:
        return None
    entry = int(rng.integers(extent-span+1))
    if d == 0:
        top, left = port[0][0]-gap-h+1, port[0][1]-entry
    elif d == 2:
        top, left = port[0][0]+gap, port[0][1]-entry
    elif d == 1:
        top, left = port[0][0]-entry, port[0][1]+gap
    else:
        top, left = port[0][0]-entry, port[0][1]-gap-w+1
    return dict(room, top=top, left=left)


def _scene(rooms, tail, cap, initial, palette, shape=(30, 30)):
    inp = np.full(shape, palette[0], dtype=np.int8)
    for room in rooms:
        r, c = room['top'], room['left']
        h, w = room['grid'].shape
        inp[r:r+h, c:c+w] = room['grid']
    for q in tail:
        inp[q] = palette[1]
    inp[cap] = palette[2]
    return dict(input=inp, palette=palette, rooms=rooms, tail=tail, cap=cap, initial=initial)


def _pack(rng, palette, rooms, gaps, initial, decoys):
    """Bounded direct placement; no per-seed files or retained retry payloads."""
    for _ in range(350):
        h, w = rooms[0]['grid'].shape
        placed = [dict(rooms[0], top=int(rng.integers(1, 30-h)), left=int(rng.integers(1, 30-w)))]
        for room, gap in zip(rooms[1:], gaps):
            candidate = _place_next(placed[-1], room, gap, rng)
            if candidate is None:
                break
            r, c = candidate['top'], candidate['left']
            bh, bw = candidate['grid'].shape
            if not (1 <= r and r+bh < 30 and 1 <= c and c+bw < 30 and all(_apart(candidate, x) for x in placed)):
                break
            placed.append(candidate)
        if len(placed) != len(rooms):
            continue
        first = placed[0]
        r, c = first['top'], first['left']
        h, w = first['grid'].shape
        length = int(rng.integers(2, 5))
        if initial % 2 == 0:
            line = c+int(rng.integers(1, w-1))
            tail = [(k if initial == 2 else 29-k, line) for k in range(length)]
        else:
            line = r+int(rng.integers(1, h-1))
            tail = [(line, k if initial == 1 else 29-k) for k in range(length)]
        dr, dc = DIRS[initial]
        cap = (tail[-1][0]+dr, tail[-1][1]+dc)
        if any(r <= q[0] < r+h and c <= q[1] < c+w for q in tail+[cap, (cap[0]+dr, cap[1]+dc)]):
            continue
        try:
            if any(x['top']-1 <= q[0] <= x['top']+x['grid'].shape[0] and x['left']-1 <= q[1] <= x['left']+x['grid'].shape[1]
                   for q in tail+[cap] for x in placed):
                continue
            scene = _scene(placed, tail, cap, initial, palette)
            _, facts = _run(scene)
            if facts['reached'] != list(range(len(rooms))):
                continue
            for local in decoys:
                for _ in range(60):
                    bh, bw = local['grid'].shape
                    candidate = dict(local, top=int(rng.integers(1, 30-bh)), left=int(rng.integers(1, 30-bw)))
                    if not all(_apart(candidate, x) for x in placed):
                        continue
                    if any(candidate['top']-1 <= q[0] <= candidate['top']+bh and candidate['left']-1 <= q[1] <= candidate['left']+bw for q in tail+[cap]):
                        continue
                    trial = _scene(placed+[candidate], tail, cap, initial, palette)
                    try:
                        if _run(trial)[1]['reached'] != list(range(len(rooms))):
                            continue
                    except ValueError:
                        continue
                    placed.append(candidate)
                    break
                else:
                    break
            if len(placed) != len(rooms)+len(decoys):
                continue
            return _scene(placed, tail, cap, initial, palette)
        except ValueError:
            continue
    raise ValueError('bounded scene placement exhausted')


def _trim(scene, rng):
    """Vary canvas margins while keeping the launching edge and all stations."""
    inp = scene['input']
    points = np.argwhere(inp != scene['palette'][0])
    lo, hi = points.min(axis=0), points.max(axis=0)+1
    lo = np.maximum(0, lo-rng.integers(1, 5, size=2))
    hi = np.minimum(30, hi+rng.integers(1, 5, size=2))
    r0, c0, r1, c1 = int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])
    rooms = [dict(x, top=x['top']-r0, left=x['left']-c0) for x in scene['rooms']]
    return dict(scene, input=inp[r0:r1, c0:c1].copy(), rooms=rooms,
                tail=[(r-r0, c-c0) for r, c in scene['tail']], cap=(scene['cap'][0]-r0, scene['cap'][1]-c0))


def _twins(rng, palette):
    for _ in range(70):
        d = int(rng.integers(4))
        initial = (d+int(rng.choice([1, 3]))) % 4
        dims = (2, 3) if d % 2 == 0 else (3, 2)
        tile = _stencil(*dims, rng, witness=True)
        a = _make_room(rng, palette, d, 3, tile=tile, area=6, transposed=bool(d % 2))
        d2 = (d+int(rng.choice([1, 3]))) % 4
        b = _make_room(rng, palette, d2, int(rng.integers(1, 5)), transposed=bool(d % 2))
        weak = _change_count(a, 2, rng, palette, area=6)
        gap = int(rng.integers(_length(weak)+1, _length(a)+1))
        decoy = _make_room(rng, palette, int(rng.integers(4)), int(rng.integers(1, 5)))
        try:
            strong = _trim(_pack(rng, palette, [a, b], [gap], initial, [decoy]), rng)
            weak = dict(weak, top=strong['rooms'][0]['top'], left=strong['rooms'][0]['left'])
            weak_scene = _scene([weak]+strong['rooms'][1:], strong['tail'], strong['cap'], initial, palette, strong['input'].shape)
            if _run(weak_scene)[1]['reached'] != [0]:
                continue
            return weak_scene, strong
        except ValueError:
            continue
    raise ValueError('could not construct contrastive demonstrations')


def _ordinary(rng, palette, test=False):
    for _ in range(90):
        n = 3 if test else int(rng.integers(1, 3))
        directions = [int(rng.integers(4))]
        for _ in range(n-1):
            directions.append((directions[-1]+int(rng.choice([-1, 0, 1]))) % 4)
        rooms = []
        for i, d in enumerate(directions):
            count = int(rng.integers(2, 6)) if i < n-1 else int(rng.integers(1, 6))
            dims = tuple(map(int, rng.integers(1 if i == n-1 else 2, 5, size=2)))
            tile = _stencil(*dims, rng)
            rooms.append(_make_room(rng, palette, d, count, tile=tile,
                                    transposed=bool(d % 2) if rng.random() < .75 else None))
        gaps = [int(rng.integers(2, min(7, _length(x))+1)) for x in rooms[:-1]]
        initial = (directions[0]+int(rng.choice([-1, 0, 1]))) % 4
        decoys = [_make_room(rng, palette, int(rng.integers(4)), int(rng.integers(1, 6)))
                  for _ in range(int(rng.integers(0, 3)))]
        try:
            scene = _trim(_pack(rng, palette, rooms, gaps, initial, decoys), rng)
            # Transfer the finite-budget question to a later link in some tests.
            if test and rng.random() < .4:
                gate = scene['rooms'][1]
                low = _change_count(gate, 1, rng, palette)
                if _length(low) < gaps[1]:
                    scene = _scene([scene['rooms'][0], low]+scene['rooms'][2:], scene['tail'], scene['cap'], initial, palette, scene['input'].shape)
            _run(scene)
            return scene
        except ValueError:
            continue
    raise ValueError('could not construct ordinary scene')


def _pair(scene):
    return {'input': scene['input'].copy(), 'output': _run(scene)[0]}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError('illegal pair count')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base = tuple(map(int, rng.choice(10, 6, replace=False)))
    vary_palette = bool(rng.integers(2))
    weak, strong = _twins(rng, base)
    puzzle = {'train': [_pair(weak), _pair(strong)], 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, amount in (('train', int(num_train)-2), ('test', int(num_test))):
        for _ in range(amount):
            for _ in range(12):
                palette = tuple(map(int, rng.choice(10, 6, replace=False))) if vary_palette else base
                pair = _pair(_ordinary(rng, palette, test=split == 'test'))
                if _key(pair['input']) not in seen:
                    break
            else:
                raise ValueError('distinct input budget exhausted')
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _different(scene, expected, **switch):
    try:
        return not np.array_equal(_run(scene, **switch)[0], expected)
    except ValueError:
        # An alternative that loops or has partial contact also fails to explain
        # this demonstration, rather than being silently assigned an answer.
        return True


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, training, test_facts = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "printer output changes canvas")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            scene = _parse(inp)
            prediction, facts = _run(scene)
            _require(np.array_equal(prediction, out), "incorrect print or route")
            bg, primary, secondary = scene['palette'][:3]
            _require(set(map(int, np.unique(out))) <= {bg, primary, secondary}, "output has structural ink")
            _require(bool(facts['reached']), "generated launcher must reach a station")
            for i, room in enumerate(scene['rooms']):
                r, c = room['top'], room['left']
                h, w = room['grid'].shape
                region = out[r:r+h, c:c+w]
                if i in facts['reached']:
                    _require(np.all(region[room['grid'] != bg] == primary), "activated footprint lost")
                    _require(np.all(region[room['grid'] == bg] == bg), "station interior filled")
                else:
                    _require(np.all(region == bg), "dormant station not erased")
                _require(room['count'] >= 1, "empty counter")
            if split == 'train':
                training.append((scene, facts, out))
            else:
                test_facts.append((scene, facts))
    # Inspect observed instructions, not generation-time mode labels.
    _require(any(len(f['reached']) >= 2 for s, f, o in training), "missing chain witness")
    _require(any(len(f['reached']) < len(s['rooms']) for s, f, o in training), "missing erasure witness")
    _require(any(_different(s, o, unlimited=True) for s, f, o in training), "count never affects answer")
    _require(any(_different(s, o, one_copy=True) for s, f, o in training), "repetition not demonstrated")
    _require(any(_different(s, o, straight=True) for s, f, o in training), "ports not demonstrated")
    _require(any(_different(s, o, rotate=True) for s, f, o in training), "motif orientation not demonstrated")
    # The two explanations below must differ on a controlled context, not merely
    # on unrelated examples with different motifs or network layouts.
    gate_contexts = {}
    for scene, facts, _ in training:
        first = scene['rooms'][facts['reached'][0]]
        cleared = scene['input'].copy()
        a, b, c, d = first['counter_bounds']
        r, col = first['top'], first['left']
        cleared[r+a:r+b, col+c:col+d] = scene['palette'][0]
        gate_contexts.setdefault((_key(cleared), first['area']), []).append(
            (first['count'], len(facts['reached'])))
    _require(any(any(a < b and x == 1 and y >= 2 for a, x in values for b, y in values)
                 for values in gate_contexts.values()), "missing controlled finite-reach witness")
    instructions = []
    for scene, facts, _ in training:
        instructions.extend(scene['rooms'][i] for i in facts['reached'])
    _require(any(a['area'] == b['area'] and a['count'] != b['count'] for a in instructions for b in instructions), "area/count confounding")
    _require(any(not r['tile'].all() and _length(r) >= 2*r['tile'].shape[0 if r['out_dir'] % 2 == 0 else 1]+1 for r in instructions), "missing two-color separator witness")
    _require(all(len(s['rooms']) >= 3 and len(f['reached']) >= 2 for s, f in test_facts), "missing later-link test shift")


def self_test():
    """42 fixed seeds, streamed measurements, no saved grids or trace logs."""
    signatures, shapes, motifs, layouts, counts, directions, palettes = set(), set(), set(), set(), set(), set(), set()
    component_sizes, test_active = set(), set()
    holes = solids = one_cell = terminal_budget = clipped = changed = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene = _parse(pair['input'])
                out, facts = _run(scene)
                assert np.array_equal(out, pair['output'])
                shapes.add(pair['input'].shape)
                palettes.add(scene['palette'])
                directions.add(scene['initial'])
                if split == 'test':
                    test_active.add(len(facts['reached']))
                terminal_budget += facts['stop'] == 'budget'
                clipped += facts['stop'] == 'boundary'
                for room in scene['rooms']:
                    motifs.add(_key(room['tile']))
                    counts.add(room['count'])
                    directions.add(room['out_dir'])
                    layouts.add((room['grid'].shape, room['transposed'], tuple(room['port'])))
                    a, b, c, d = room['counter_bounds']
                    cm = room['grid'][a:b, c:d] == scene['palette'][5]
                    component_sizes.update(map(len, _components(cm)))
                    solids += room['tile'].all()
                    one_cell += room['tile'].size == 1
                    holes += len(_components(~np.pad(room['tile'], 1))) > 1
                for value in pair.values():
                    digest.update(str(value.shape).encode())
                    digest.update(value.tobytes())
                changed += not np.array_equal(pair['input'], pair['output'])
        signatures.add(digest.digest())
        if seed in (0, 17, 41):
            other = generate_puzzle(seed)
            assert all(np.array_equal(puzzle[s][i][k], other[s][i][k]) for s in puzzle for i in range(len(puzzle[s])) for k in ('input', 'output'))
    assert len(signatures) == 42 and changed == 210
    assert len(shapes) >= 25 and len(motifs) >= 35 and len(layouts) >= 100
    assert directions == set(range(4)) and counts == {1, 2, 3, 4, 5}
    assert test_active == {2, 3} and len(palettes) >= 30
    assert max(component_sizes) >= 8 and holes > 0 and solids > 0 and one_cell > 0
    assert terminal_budget > 0 and clipped > 0
    # Direct stencil rendering checks for every direction and counts 1..3.
    # They check the negative-direction indexing independently of slice order;
    # they are fixtures, not another general scene solver.
    fixture_rng = np.random.default_rng(9001)
    palette = (0, 1, 2, 8, 3, 4)
    tile = np.array([[True, False, False], [True, True, True]])
    for direction in range(4):
        for count in range(1, 4):
            room = _make_room(fixture_rng, palette, direction, count, tile=tile)
            room = dict(room, top=12 if direction == 0 else 3 if direction == 2 else 10,
                        left=12 if direction == 3 else 3 if direction == 1 else 10)
            initial = 1 if direction % 2 == 0 else 2
            line = room['top']+1 if initial == 1 else room['left']+1
            tail = [(line, 0), (line, 1)] if initial == 1 else [(0, line), (1, line)]
            cap = (line, 2) if initial == 1 else (2, line)
            scene = _scene([room], tail, cap, initial, palette)
            actual, facts = _run(scene)
            assert facts['reached'] == [0]
            colored = np.where(tile, palette[1], palette[2])
            axis = 0 if direction % 2 == 0 else 1
            gap_shape = (1, tile.shape[1]) if axis == 0 else (tile.shape[0], 1)
            gap = np.full(gap_shape, palette[2], dtype=np.int8)
            expected = np.concatenate([p for k in range(count) for p in ([colored] if k == count-1 else [colored, gap])], axis=axis)
            r, c = sorted(room['port'])[0]
            r, c = r+room['top'], c+room['left']
            r0 = r-expected.shape[0] if direction == 0 else r+1 if direction == 2 else r
            c0 = c-expected.shape[1] if direction == 3 else c+1 if direction == 1 else c
            assert np.array_equal(actual[r0:r0+expected.shape[0], c0:c0+expected.shape[1]], expected)
    # Explicit counts include every legal combination, without keeping them all.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(101+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(0, 2, 1)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                motifs=len(motifs), station_layouts=len(layouts), palettes=len(palettes),
                counts=sorted(counts), component_sizes=sorted(component_sizes), test_active=sorted(test_active),
                solid_stencils=int(solids), singleton_stencils=int(one_cell), stencils_with_holes=int(holes),
                budget_stops=int(terminal_budget), boundary_stops=int(clipped))
