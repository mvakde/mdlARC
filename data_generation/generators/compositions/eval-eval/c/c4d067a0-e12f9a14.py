"""Searchlight mosaic: register a partial constellation, then reveal beam paths.

Only Python's standard library and NumPy are used. Generation draws from sampled
logical scenes; solve recovers those scenes entirely from the input pixels.
"""
from __future__ import annotations

from math import gcd
import numpy as np


_DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1))
_KINDS = ("perpendicular", "opposing_diagonals", "staircase")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or
                not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _matches(plan, offsets, colors):
    extent = tuple(max(p[d] for p in offsets) for d in (0, 1))
    return [(r, c) for r in range(plan.shape[0] - extent[0])
            for c in range(plan.shape[1] - extent[1])
            if all(int(plan[r+p[0], c+p[1]]) == color
                   for p, color in zip(offsets, colors))]


def _decode(a):
    """Find isolated swatches and framed cores, then uniquely register them."""
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > np.partition(counts, -2)[-2], "ambiguous background")
    swatches, frames = [], []
    for cells in _components(a != bg):
        if len(cells) == 1:
            swatches.append(cells[0])
            continue
        points = np.array(cells)
        r, c = map(int, points.min(axis=0))
        bottom, right = map(int, points.max(axis=0) + 1)
        n = bottom-r
        _require(n >= 4 and right-c == n, "object is not a framed square")
        patch = a[r:bottom, c:right]
        side = n-2
        color = int(patch[1, 1])
        _require(color != bg and np.all(patch[1:-1, 1:-1] == color), "invalid payload core")
        ring_mask = np.ones(patch.shape, dtype=bool)
        ring_mask[1:-1, 1:-1] = False
        ring = patch[ring_mask]
        structure = set(map(int, ring)) - {bg}
        _require(len(structure) == 1 and color not in structure, "invalid frame color")
        _require(np.count_nonzero(ring == bg) == 1 and len(cells) == n*n-1,
                 "a frame must have exactly one opening")
        gr, gc = map(int, np.argwhere((patch == bg) & ring_mask)[0])
        direction = (-1 if gr == 0 else 1 if gr == n-1 else 0,
                     -1 if gc == 0 else 1 if gc == n-1 else 0)
        frames.append(dict(core=(r+1, c+1), side=side, color=color,
                           structure=next(iter(structure)), gap=(r+gr, c+gc),
                           direction=direction))
    _require(len(swatches) >= 5 and 2 <= len(frames) <= 4, "missing miniature or anchors")
    _require(len({f['side'] for f in frames}) == 1 and
             len({f['structure'] for f in frames}) == 1, "inconsistent frame vocabulary")
    points = np.array(swatches)
    mr, mc = map(int, points.min(axis=0))
    er, ec = map(int, points.max(axis=0) + 1)
    _require(np.all((points - (mr, mc)) % 2 == 0), "miniature pitch is not two")
    plan = a[mr:er:2, mc:ec:2].copy()
    _require(min(plan.shape) >= 3, "miniature must be at least 3 by 3")
    mask = np.zeros((er-mr, ec-mc), dtype=bool)
    mask[::2, ::2] = True
    _require(np.all(a[mr:er, mc:ec][~mask] == bg), "miniature separators changed")
    _require(np.count_nonzero(plan != bg) == len(swatches), "unaccounted swatches")
    ring = frames[0]['structure']
    _require(not np.any(plan == ring), "frame color cannot be a miniature payload")
    _require(all(np.any(np.diff(np.unique(points[:, d])) == 2) for d in (0, 1)),
             "miniature pitch lacks visible witnesses")
    cores = np.array([f['core'] for f in frames])
    differences = [int(x) for d in (0, 1) for x in np.diff(np.unique(cores[:, d])) if x > 0]
    stride = min(differences)
    side = frames[0]['side']
    _require(stride >= side+3, "frames need a separating gutter")
    first = cores.min(axis=0)
    _require(np.all((cores-first) % stride == 0), "cores do not share a square lattice")
    offsets = [tuple(map(int, p)) for p in (cores-first)//stride]
    matches = _matches(plan, offsets, [f['color'] for f in frames])
    _require(len(matches) == 1, "anchor constellation must match exactly once")
    match = matches[0]
    base = tuple(map(int, first - np.array(match)*stride))
    box = (base[0]-1, base[1]-1,
           base[0]+(plan.shape[0]-1)*stride+side+1,
           base[1]+(plan.shape[1]-1)*stride+side+1)
    _require(0 <= box[0] < box[2] <= a.shape[0] and
             0 <= box[1] < box[3] <= a.shape[1], "full mosaic does not fit")
    _require(not _overlap(box, (mr, mc, er, ec)), "miniature overlaps enlarged mosaic")
    for f, p in zip(frames, offsets):
        f['site'] = (match[0]+p[0], match[1]+p[1])
    return dict(background=bg, structure=ring, plan=plan, frames=frames,
                side=side, stride=stride, base=base, miniature=(mr, mc, er, ec), match=match)


def _walk(point, heading, shape, phase='floor'):
    """Major-axis-first raster: floor separately in absolute coordinates."""
    vr, vc = heading
    major = max(abs(vr), abs(vc))
    result = []
    for t in range(3*max(shape)+1):
        er = major-1 if phase == 'ceil' and vr else 0
        ec = major-1 if phase == 'ceil' and vc else 0
        p = (point[0] + (1 if vr >= 0 else -1)*((t*abs(vr)+er)//major),
             point[1] + (1 if vc >= 0 else -1)*((t*abs(vc)+ec)//major))
        if not _inside(p, shape):
            break
        result.append(p)
    return result


def _encounter(aa, bb):
    hits = []
    for ai, a in enumerate(aa):
        cells = {p: k for k, p in enumerate(a)}
        edges = {(p[0]+q[0], p[1]+q[1]): k for k, (p, q) in enumerate(zip(a, a[1:]))
                 if p[0] != q[0] and p[1] != q[1]}
        for bi, b in enumerate(bb):
            for j, p in enumerate(b):
                if p in cells:
                    i = cells[p]
                    hits.append((i-1, j-1, ai, bi, "cell", i, j))
            for j, (p, q) in enumerate(zip(b, b[1:])):
                midpoint = (p[0]+q[0], p[1]+q[1])
                if p[0] != q[0] and p[1] != q[1] and midpoint in edges:
                    i = edges[midpoint]
                    if {a[i], a[i+1]} != {p, q}:
                        hits.append((i, j, ai, bi, "between", i+.5, j+.5))
    return min(hits, key=lambda h: (sum(h[:2]), max(h[:2]), h[2], h[3])) if hits else None


def _kind(u, v):
    _require(max(map(abs, u)) == max(map(abs, v)) == 1,
             "established staircases may not encounter another beam")
    _require(u != v and u != (-v[0], -v[1]), "overlap or cancellation")
    nu, nv = sum(map(abs, u)), sum(map(abs, v))
    if nu == nv == 1:
        return "perpendicular"
    if nu == nv == 2:
        return "opposing_diagonals"
    _require(u[0]*v[0]+u[1]*v[1] > 0, "unsupported obtuse encounter")
    return "staircase"


def _trace(shape, frames):
    """Trace spatial paths in logical tile coordinates, retaining offsets."""
    groups = [dict(heading=f['direction'], members=[(i, f['site'])])
              for i, f in enumerate(frames)]
    anchors = {f['site']: i for i, f in enumerate(frames)}
    owned, paths, events = {}, [[] for _ in frames], []
    segments = set()

    def paint(owner, points):
        for k, p in enumerate(points):
            _require(p not in anchors or anchors[p] == owner, "beam hits another anchor")
            repeated_start = k == 0 and paths[owner] and paths[owner][-1] == p
            _require(p not in owned or (owned[p] == owner and repeated_start), "beam overlap")
            if not repeated_start:
                if paths[owner]:
                    q = paths[owner][-1]
                    if q[0] != p[0] and q[1] != p[1]:
                        mid = (q[0]+p[0], q[1]+p[1])
                        slope = 1 if (q[0]-p[0])*(q[1]-p[1]) > 0 else -1
                        _require((mid, -slope) not in segments, "unresolved crossing")
                        segments.add((mid, slope))
                paths[owner].append(p)
            owned[p] = owner

    for _ in range(len(frames)):
        forecasts = [[_walk(p, group['heading'], shape) for _, p in group['members']]
                     for group in groups]
        candidates = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                hit = _encounter(forecasts[i], forecasts[j])
                if hit is not None:
                    candidates.append((i, j, hit))
        if not candidates:
            for group, pp in zip(groups, forecasts):
                for (owner, _), path in zip(group['members'], pp):
                    paint(owner, path)
            return set(owned), paths, events
        nearest = {}
        for i, j, hit in candidates:
            nearest[i] = min(nearest.get(i, 100), hit[0])
            nearest[j] = min(nearest.get(j, 100), hit[1])
        ready = [e for e in candidates if e[2][0] == nearest[e[0]] and e[2][1] == nearest[e[1]]]
        _require(bool(ready), "cyclic encounter order")
        i, j, hit = min(ready, key=lambda e: (max(e[2][:2]), sum(e[2][:2]), e[0], e[1]))
        for k, l, other in candidates:
            if (k, l) != (i, j):
                _require(not ((k in (i, j) and other[0] == nearest[k]) or
                              (l in (i, j) and other[1] == nearest[l])), "multiway encounter")
        _require(min(hit[:2]) >= 0, "encounter at a beam origin")
        u, v = groups[i]['heading'], groups[j]['heading']
        kind = _kind(u, v)
        total = (u[0]+v[0], u[1]+v[1])
        divisor = gcd(abs(total[0]), abs(total[1]))
        heading = (total[0]//divisor, total[1]//divisor)
        members = []
        for index, cut in ((i, hit[0]), (j, hit[1])):
            for (owner, _), path in zip(groups[index]['members'], forecasts[index]):
                _require(cut < len(path), "bundle clipped before encounter")
                paint(owner, path[:cut+1])
                members.append((owner, path[cut]))
        events.append(dict(kind=kind, intersection=hit[4], unequal=hit[5] != hit[6],
                           incoming=(u, v), heading=heading,
                           owners=tuple(owner for owner, _ in members), starts=tuple(members)))
        groups = [g for k, g in enumerate(groups) if k not in (i, j)]
        groups.append(dict(heading=heading, members=members))
    raise ValueError("too many encounters")


def _straight_sites(shape, frames):
    return {p for f in frames for p in _walk(f['site'], f['direction'], shape)}


def _wrong_bend_sites(shape, paths, events, mode):
    """Two teaching attacks on the single-event scenes sampled by this author."""
    _require(len(events) == 1, "bend attack requires one encounter")
    event = events[0]
    starts = dict(event['starts'])
    heading = event['heading']
    if mode == 'sign':
        heading = tuple(0 if v == 0 else 1 if v > 0 else -1 for v in heading)
    result = set()
    for owner, path in enumerate(paths):
        if owner not in starts:
            result.update(path)
        else:
            point = starts[owner]
            result.update(path[:path.index(point)+1])
            result.update(_walk(point, heading, shape, phase='ceil' if mode == 'phase' else 'floor'))
    return result


def _reveal(a, meta, sites):
    out = a.copy()
    bg, side, stride = meta['background'], meta['side'], meta['stride']
    for r, c in sites:
        color = int(meta['plan'][r, c])
        if color != bg:
            y, x = meta['base'][0]+r*stride, meta['base'][1]+c*stride
            out[y:y+side, x:x+side] = color
    for f in meta['frames']:
        out[f['gap']] = f['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    meta = _decode(a)
    sites, _, _ = _trace(meta['plan'].shape, meta['frames'])
    return _reveal(a, meta, sites)


def _transform_scene(shape, frames, rotation, reflection):
    h, w = shape
    result = []
    for f in frames:
        r, c = f['site']
        dr, dc = f['direction']
        hh, ww = h, w
        for _ in range(rotation):
            r, c, dr, dc = ww-1-c, r, -dc, dr
            hh, ww = ww, hh
        if reflection:
            c, dc = ww-1-c, -dc
        result.append(dict(site=(r, c), direction=(dr, dc)))
    return (h, w) if rotation % 2 == 0 else (w, h), result


def _primary(rng, shape, kind, intersection):
    h, w = shape
    for _ in range(240):
        if kind == 'isolated':
            positions = rng.choice(h*w, 2, replace=False)
            frames = [dict(site=(int(p)//w, int(p)%w),
                           direction=_DIRECTIONS[int(rng.integers(8))]) for p in positions]
        else:
            r = int(rng.integers(1 if kind != 'staircase' else 2, h-1))
            c = int(rng.integers(1, w))
            if kind == 'perpendicular':
                a, b = int(rng.integers(1, c+1)), int(rng.integers(1, h-r))
                if min(a, b) != 1 or a == b:
                    continue
                frames = [dict(site=(r, c-a), direction=(0, 1)),
                          dict(site=(r+b, c), direction=(-1, 0))]
            elif kind == 'staircase':
                a, b = int(rng.integers(1, c+1)), int(rng.integers(1, h-r))
                if abs(a-b) != 1 or r+a >= h:
                    continue
                frames = [dict(site=(r+a, c-a), direction=(-1, 1)),
                          dict(site=(r+b, c), direction=(-1, 0))]
            else:
                a, b = int(rng.integers(0, h)), int(rng.integers(0, h))
                if abs(a-b) != 1:
                    continue
                shift = 1 if intersection == 'between' else 0
                if (intersection == 'cell' and min(a, b) < 1) or c-a < 0 or c+b+shift >= w:
                    continue
                frames = [dict(site=(r+a+shift, c-a), direction=(-1, 1)),
                          dict(site=(r+b+shift, c+b+shift), direction=(-1, -1))]
        if any(not _inside(f['site'], shape) for f in frames):
            continue
        diffs = [x for d in (0, 1) for x in np.diff(sorted({f['site'][d] for f in frames})) if x]
        if min(diffs) != 1:
            continue
        try:
            sites, paths, events = _trace(shape, frames)
            if ((kind == 'isolated' and events) or
                    (kind != 'isolated' and (len(events) != 1 or events[0]['kind'] != kind or
                     events[0]['intersection'] != intersection))):
                continue
            if len(sites)-2 < 4 or h*w-len(sites) < 2:
                continue
            return frames
        except ValueError:
            pass
    # A small constructive fallback uses the same visible rule, with paths long
    # enough to teach holes and changing tile colors. The shape may be larger.
    templates = {
        'isolated': [((h-1, 0), (-1, 0)), ((h-2, 1), (-1, 1))],
        'perpendicular': [((2, 0), (0, 1)), ((3, 2), (-1, 0))],
        'staircase': [((3, 0), (-1, 1)), ((4, 1), (-1, 0))],
        'opposing_diagonals': ([((3, 0), (-1, 1)), ((2, 2), (-1, -1))]
                               if intersection == 'between' else
                               [((2, 0), (-1, 1)), ((3, 3), (-1, -1))])}
    frames = [dict(site=p, direction=d) for p, d in templates[kind]]
    _require(all(_inside(f['site'], shape) for f in frames), "fallback does not fit")
    _trace(shape, frames)
    return frames


def _logical_scene(rng, shape, kind, intersection):
    frames = _primary(rng, shape, kind, intersection)
    target = int(rng.choice((2, 3, 4), p=(.35, .4, .25)))
    for _ in range(60):
        if len(frames) >= target:
            break
        p = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
        if p in {f['site'] for f in frames}:
            continue
        extra = dict(site=p, direction=_DIRECTIONS[int(rng.integers(8))])
        candidate = frames+[extra]
        try:
            sites, _, events = _trace(shape, candidate)
            wanted = 0 if kind == 'isolated' else 1
            if len(events) != wanted or any(len(frames) in e['owners'] for e in events):
                continue
            if len(sites)-len(candidate) >= 4 and np.prod(shape)-len(sites) >= 2:
                frames = candidate
        except ValueError:
            pass
    return _transform_scene(shape, frames, int(rng.integers(4)), bool(rng.integers(2)))


def _geometry_options(kind, test, side_filter=None):
    options = []
    heights = (6,) if test else ((5,) if kind == 'staircase' else (4, 5))
    widths = (4,) if kind == 'opposing_diagonals' else (3, 4)
    for h in heights:
        for w in widths:
            for side in (2, 3, 4):
                if side_filter is not None and side != side_filter:
                    continue
                for gutter in (3, 4, 5):
                    box = tuple((n-1)*(side+gutter)+side+2 for n in (h, w))
                    mini = (2*h-1, 2*w-1)
                    if max(box) <= 30 and any(box[d]+mini[d]+1 <= 30 for d in (0, 1)):
                        options.append((h, w, side, gutter))
    return options


def _plan(rng, shape, frames, paths, sites, bg, palette):
    anchor_sites = {f['site'] for f in frames}
    traversable = [p for path in paths for j, p in enumerate(path[:-1])
                   if p not in anchor_sites and any(q not in anchor_sites for q in path[j+1:])]
    _require(bool(traversable), "no transparent-hole witness")
    first = tuple(min(p[d] for p in anchor_sites) for d in (0, 1))
    offsets = [(f['site'][0]-first[0], f['site'][1]-first[1]) for f in frames]
    for _ in range(200):
        plan = rng.choice(palette, size=shape).astype(np.int8)
        holes = rng.random(shape) < float(rng.uniform(.06, .25))
        for p in anchor_sites:
            holes[p] = False
        holes[traversable[int(rng.integers(len(traversable)))]] = True
        plan[holes] = bg
        if any(np.all(edge == bg) for edge in (plan[0], plan[-1], plan[:, 0], plan[:, -1])):
            continue
        added = {int(plan[p]) for p in sites-anchor_sites} - {bg}
        if len(added) < 2 or not any(plan[p] != bg for p in set(np.ndindex(shape))-sites):
            continue
        colors = [int(plan[f['site']]) for f in frames]
        if _matches(plan, offsets, colors) != [first]:
            continue
        if not any(plan[p] not in (bg, colors[i]) for i, path in enumerate(paths)
                   for p in path if p not in anchor_sites):
            continue
        if any(np.array_equal(plan, v) for v in (plan[::-1], plan[:, ::-1], plan[::-1, ::-1])):
            continue
        if not any(plan[p] == bg and any(plan[q] != bg for q in path[j+1:])
                   for path in paths for j, p in enumerate(path[:-1]) if p not in anchor_sites):
            continue
        return plan
    raise ValueError("could not sample an explanatory registered plan")


def _place(rng, shape, side, stride):
    box = tuple((n-1)*stride+side+2 for n in shape)
    mini = tuple(2*n-1 for n in shape)
    possible = [d for d in (0, 1) if box[d]+mini[d]+1 <= 30]
    d = possible[int(rng.integers(len(possible)))]
    other = 1-d
    gap = int(rng.integers(1, min(4, 30-box[d]-mini[d])+1))
    required = [max(box[k], mini[k]) for k in (0, 1)]
    required[d] = box[d]+mini[d]+gap
    canvas = tuple(int(rng.integers(n, min(30, n+5)+1)) for n in required)
    if rng.random() < .2:
        canvas = (max(canvas), max(canvas))
    anchors, small = [0, 0], [0, 0]
    margin = int(rng.integers(canvas[d]-required[d]+1))
    if rng.integers(2):
        anchors[d], small[d] = margin, margin+box[d]+gap
    else:
        small[d], anchors[d] = margin, margin+mini[d]+gap
    anchors[other] = int(rng.integers(canvas[other]-box[other]+1))
    small[other] = int(rng.integers(canvas[other]-mini[other]+1))
    return canvas, tuple(small), (anchors[0]+1, anchors[1]+1)


def _draw(rng, plan, frames, bg, ring, side, stride):
    canvas, mini, base = _place(rng, plan.shape, side, stride)
    a = np.full(canvas, bg, dtype=np.int8)
    a[mini[0]:mini[0]+2*plan.shape[0]:2, mini[1]:mini[1]+2*plan.shape[1]:2] = plan
    painted = []
    for f in frames:
        r, c = f['site']
        y, x = base[0]+r*stride, base[1]+c*stride
        color = int(plan[r, c])
        a[y-1:y+side+1, x-1:x+side+1] = ring
        a[y:y+side, x:x+side] = color
        dr, dc = f['direction']
        gr = y-1 if dr < 0 else y+side if dr > 0 else y+int(rng.integers(side))
        gc = x-1 if dc < 0 else x+side if dc > 0 else x+int(rng.integers(side))
        a[gr, gc] = bg
        painted.append(dict(f, color=color, gap=(gr, gc)))
    return a, dict(background=bg, structure=ring, plan=plan, frames=painted,
                   side=side, stride=stride, base=base)


def _make_pair(rng, kind, intersection, test, side_filter=None, displaced=False):
    options = _geometry_options(kind, test, side_filter)
    _require(bool(options), "no fitting geometry")
    for _ in range(80):
        h, w, side, gutter = options[int(rng.integers(len(options)))]
        try:
            shape, frames = _logical_scene(rng, (h, w), kind, intersection)
            sites, paths, events = _trace(shape, frames)
            colors = rng.permutation(10)
            bg, ring = map(int, colors[:2])
            palette = colors[2:2+int(rng.integers(2, 6))]
            plan = _plan(rng, shape, frames, paths, sites, bg, palette)
            if kind == 'isolated':
                emission = {sum(map(abs, f['direction'])) for f, path in zip(frames, paths)
                            if any(plan[p] != bg for p in path[1:])}
                if emission != {1, 2}:
                    continue
            a, meta = _draw(rng, plan, frames, bg, ring, side, side+gutter)
            out = _reveal(a, meta, sites)
            decoded = _decode(a)
            if displaced and decoded['match'] == (0, 0):
                continue
            _require(np.array_equal(solve(a), out), "sampled and recovered scenes differ")
            if kind != 'isolated':
                straight = _reveal(a, decoded, _straight_sites(shape, decoded['frames']))
                if np.array_equal(straight, out):
                    continue
            if kind == 'staircase' and any(np.array_equal(
                    out, _reveal(a, meta, _wrong_bend_sites(shape, paths, events, mode)))
                    for mode in ('sign', 'phase')):
                continue
            return dict(input=a, output=out)
        except ValueError:
            continue
    raise RuntimeError(f"exhausted {kind} construction")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    kinds = list(_KINDS)
    rng.shuffle(kinds)
    kinds = kinds[:int(rng.integers(1, min(3, nt-1)+1))]
    phase = 'cell' if rng.integers(2) else 'between'
    modes = ['isolated']+kinds
    while len(modes) < nt:
        modes.append(str(rng.choice(['isolated']+kinds)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            kind = modes[i] if split == 'train' else str(rng.choice(kinds))
            intersection = phase if kind == 'opposing_diagonals' else 'cell'
            # The two mandatory examples demonstrate distinct scales. Returned
            # order is shuffled, and neither alignment nor rule uses an ordinal.
            side = int(rng.choice((3, 4))) if split == 'train' and i == 0 else (
                   2 if split == 'train' and i == 1 else None)
            for _ in range(20):
                pair = _make_pair(rng, kind, intersection, split == 'test', side,
                                  displaced=split == 'train' and i == 1)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a, out=None):
    meta = _decode(_grid(a))
    plan, bg, frames = meta['plan'], meta['background'], meta['frames']
    sites, paths, events = _trace(plan.shape, frames)
    anchors = {f['site'] for f in frames}
    expected = _reveal(_grid(a), meta, sites)
    if out is not None:
        _require(np.array_equal(expected, _grid(out)), "incorrect revelation or preservation")
    added = {int(plan[p]) for p in sites-anchors} - {bg}
    _require(len(added) >= 2, "example must reveal at least two literal colors")
    _require(any(plan[p] != bg for p in set(np.ndindex(plan.shape))-sites),
             "example must leave a colored tile invisible")
    transparent = any(plan[p] == bg and any(plan[q] != bg for q in path[j+1:])
                      for path in paths for j, p in enumerate(path[:-1]) if p not in anchors)
    _require(transparent, "example must show continuation beyond an empty tile")
    carry_difference = any(plan[p] not in (bg, f['color']) for f, path in zip(frames, paths)
                           for p in path if p not in anchors)
    _require(carry_difference, "source-color shortcut not distinguished")
    straight_difference = not np.array_equal(
        expected, _reveal(_grid(a), meta, _straight_sites(plan.shape, frames)))
    _require(not events or straight_difference, "collision must change visible tiles")
    if len(events) == 1 and events[0]['kind'] == 'staircase':
        _require(all(not np.array_equal(expected, _reveal(_grid(a), meta,
                     _wrong_bend_sites(plan.shape, paths, events, mode)))
                     for mode in ('sign', 'phase')), "staircase alternatives are not distinguished")
    return dict(shape=tuple(np.asarray(a).shape), plan_shape=plan.shape, side=meta['side'],
                gutter=meta['stride']-meta['side'], background=bg, structure=meta['structure'],
                anchors=len(frames), constellation=tuple(sorted(f['site'] for f in frames)),
                match=meta['match'], ports=tuple(f['direction'] for f in frames),
                holes=int(np.count_nonzero(plan == bg)), visited=len(sites),
                event_tags=tuple((e['kind'], e['intersection']) for e in events),
                unequal=any(e['unequal'] for e in events),
                emission=tuple(sorted({sum(map(abs, f['direction']))
                                       for f, path in zip(frames, paths)
                                       if any(plan[p] != bg for p in path[1:])})),
                straight_difference=straight_difference, transparent=transparent,
                normalized=tuple(map(tuple, (plan != bg).astype(int).tolist())))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "output dimensions changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            records[split].append(_features(a, b))
            bg = records[split][-1]['background']
            _require(np.array_equal(a[a != bg], b[a != bg]), "input foreground changed")
    taught = {tag for f in records['train'] for tag in f['event_tags']}
    _require(bool(taught) and any(not f['event_tags'] for f in records['train']),
             "training needs interacting and noninteracting searchlights")
    _require({v for f in records['train'] if not f['event_tags'] for v in f['emission']} == {1, 2},
             "training must visibly demonstrate side and corner emission")
    _require(all(set(f['event_tags']) <= taught for f in records['test']), "untaught test interaction")
    _require(len({f['side'] for f in records['train']}) >= 2, "training must establish scale inference")
    _require(any(f['match'] != (0, 0) for f in records['train']), "fixed registration not challenged")
    _require(all(max(f['plan_shape']) == 6 for f in records['test']) and
             all(max(f['plan_shape']) <= 5 for f in records['train']), "logical-size shift missing")


def self_test():
    """Bounded semantic fixtures, 42-seed diversity, counts, and corruption checks."""
    # Unequal-distance perpendicular encounter: turn from (2,1)/(3,2).
    f = [dict(site=(2, 0), direction=(0, 1)), dict(site=(3, 2), direction=(-1, 0))]
    _, paths, events = _trace((4, 4), f)
    assert paths == [[(2, 0), (2, 1), (1, 2), (0, 3)], [(3, 2), (2, 3)]]
    assert events[0]['heading'] == (-1, 1) and events[0]['unequal']
    # A 2:1 staircase must begin with its major-axis step, preserving offsets.
    f = [dict(site=(3, 0), direction=(-1, 1)), dict(site=(4, 1), direction=(-1, 0))]
    _, paths, events = _trace((5, 3), f)
    assert paths == [[(3, 0), (2, 0), (1, 1), (0, 1)],
                     [(4, 1), (3, 1), (2, 1), (1, 2), (0, 2)]]
    assert events[0]['heading'] == (-2, 1)
    f = [dict(site=(3, 0), direction=(-1, 1)), dict(site=(2, 2), direction=(-1, -1))]
    _, paths, events = _trace((4, 4), f)
    assert paths == [[(3, 0), (2, 1), (1, 1), (0, 1)], [(2, 2), (1, 2), (0, 2)]]
    assert events[0]['intersection'] == 'between'
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = {tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s])
                  for puzzle in puzzles}
    assert len(signatures) == 42
    records = [_features(pair['input'], pair['output']) for puzzle in puzzles
               for s in ('train', 'test') for pair in puzzle[s]]
    assert {f['anchors'] for f in records} >= {2, 3, 4}
    assert {f['side'] for f in records} >= {2, 3, 4}
    assert {f['gutter'] for f in records} >= {3, 4, 5}
    assert {tag[0] for f in records for tag in f['event_tags']} == set(_KINDS)
    assert {tag[1] for f in records for tag in f['event_tags']} == {'cell', 'between'}
    assert len({f['plan_shape'] for f in records}) >= 8
    assert len({f['shape'] for f in records}) >= 35
    assert len({f['constellation'] for f in records}) >= 50
    assert len({f['normalized'] for f in records}) >= 100
    assert len({f['background'] for f in records}) == 10
    assert {d for f in records for d in f['ports']} == set(_DIRECTIONS)
    assert all(any(f['straight_difference'] for f in [_features(p['input']) for p in puzzle['train']])
               for puzzle in puzzles)
    replay = generate_puzzle(17)
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(puzzles[17][s], replay[s]) for k in ('input', 'output'))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(314159, nt, ne)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    # A monochrome plan cannot uniquely locate this repeated constellation.
    assert len(_matches(np.ones((4, 4), dtype=np.int8), [(0, 0), (1, 1)], [1, 1])) == 9


if __name__ == '__main__':
    self_test()
    print('Searchlight mosaic self-test passed (42 seeds).')
