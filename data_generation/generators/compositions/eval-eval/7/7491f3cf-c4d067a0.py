"""Split-scale quilts: match a region-dependent seed, then weave two plans.

Only NumPy and the standard library are used. All solver geometry comes from
pixels. Generation draws its answer from sampled plans, not by calling solve.
"""
from __future__ import annotations

from functools import lru_cache
import json
import numpy as np


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((a >= 0) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, bool)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        directions += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in directions:
                yy, xx = y + dy, x + dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _mask(shape, cells):
    a = np.zeros(shape, bool)
    for p in cells:
        a[p] = True
    return a


def _bbox(cells):
    points = np.asarray(cells)
    r, c = points.min(axis=0)
    y, x = points.max(axis=0) + 1
    return int(r), int(c), int(y), int(x)


def _regions(cut):
    """Check a simple digital path and recover its two four-connected sides."""
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {}
    for r, c in cells:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = r + dy, c + dx
                if (dy or dx) and q in cells:
                    # No diagonal shortcut around an orthogonal corner.
                    if dy and dx and ((r + dy, c) in cells or (r, c + dx) in cells):
                        continue
                    neighbors.append(q)
        graph[r, c] = neighbors
    ends = [p for p in graph if len(graph[p]) == 1]
    _require(len(cells) >= 3 and len(ends) == 2
             and all(1 <= len(n) <= 2 for n in graph.values()), "cut branches or loops")
    border = lambda p: p[0] in (0, cut.shape[0]-1) or p[1] in (0, cut.shape[1]-1)
    _require(all(border(p) == (p in ends) for p in cells), "invalid boundary contacts")
    _require(len(_components(cut, True)) == 1, "disconnected cut")
    parts = _components(~cut)
    _require(len(parts) == 2, "cut does not divide into two regions")
    return [_mask(cut.shape, p) for p in parts]


def _near(mask):
    a = mask.copy()
    for r, c in np.argwhere(mask):
        a[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return a


def _extract(inp):
    """Read the three cards, the cut/marker, and the large seed squares."""
    inp = _grid(inp)
    bg = int(inp[0, 0])
    _require(np.all(inp[[0, -1]] == bg) and np.all(inp[:, [0, -1]] == bg),
             "outside margin must be background")
    candidates = []
    for color in np.unique(inp):
        if color == bg:
            continue
        components = _components(inp == color)
        if len(components) != 3:
            continue
        boxes = []
        for comp in components:
            r, c, y, x = _bbox(comp)
            if min(y-r, x-c) < 5 or len(comp) != 2*(y-r+x-c)-4:
                break
            region = inp[r:y, c:x]
            if not (np.all(region[[0, -1]] == color) and np.all(region[:, [0, -1]] == color)):
                break
            boxes.append((r, c, y, x))
        if len(boxes) == 3:
            candidates.append((int(color), boxes))
    _require(len(candidates) == 1, "frame role is ambiguous")
    frame, boxes = candidates[0]
    sources, targets = [], []
    covered = np.zeros(inp.shape, bool)
    for box in boxes:
        r, c, y, x = box
        _require(not np.any(covered[r:y, c:x]), "overlapping cards")
        covered[r:y, c:x] = True
        body = inp[r+1:y-1, c+1:x-1]
        components = _components(body != bg)
        if components and all(len(p) == 1 for p in components):
            _require(all(n % 2 == 1 and n >= 5 for n in body.shape), "invalid miniature extent")
            sampling = np.zeros(body.shape, bool)
            sampling[1:-1:2, 1:-1:2] = True
            _require(np.all(body[~sampling] == bg), "miniature padding or pitch is wrong")
            sources.append((box, body[1:-1:2, 1:-1:2].copy()))
        else:
            targets.append((box, body))
    _require(np.all(inp[~covered] == bg), "unrelated object outside cards")
    _require(len(sources) == 2 and len(targets) == 1, "need two plans and one work card")
    _require(sources[0][1].shape == sources[1][1].shape, "plans must be congruent")
    if sources[0][0][0] == sources[1][0][0]:
        sources.sort(key=lambda p: p[0][1])  # left plan is A
    else:
        _require(sources[0][0][1] == sources[1][0][1], "source rack must be aligned")
        sources.sort(key=lambda p: p[0][0])  # upper plan is A
    a, b = sources[0][1], sources[1][1]
    palette = set(map(int, np.unique(np.concatenate((a, b))))) - {bg}
    _require(palette and frame not in palette, "invalid payload palette")
    box, body = targets[0]
    control_values = set(map(int, np.unique(body))) - palette - {bg}
    _require(len(control_values) == 1, "control color must be separate from payload")
    control = next(iter(control_values))
    _require(control != frame, "control cannot be the frame")
    components = sorted(_components(body == control, True), key=len)
    _require(len(components) == 2 and len(components[0]) == 1, "cut and dot not separated")
    marker = components[0][0]
    cut = _mask(body.shape, components[1])
    regions = _regions(cut)
    _require(not _near(cut)[marker], "marker touches cut")
    side = 0 if regions[0][marker] else 1
    marked, other = regions[side], regions[1-side]
    squares = []
    seed_mask = (body != bg) & (body != control)
    for cells in _components(seed_mask):
        r, c, y, x = _bbox(cells)
        s = y-r
        _require(s >= 2 and x-c == s and len(cells) == s*s, "seed is not a solid square")
        value = int(body[r, c])
        _require(np.all(body[r:y, c:x] == value), "multicolor seed square")
        ma, ob = np.all(marked[r:y, c:x]), np.all(other[r:y, c:x])
        _require(ma or ob, "seed square crosses the cut")
        squares.append((r, c, s, value, 0 if ma else 1))
    _require(len(squares) >= 2, "need at least two seed squares")
    axis = 1 if len({p[0] for p in squares}) == 1 else 0
    _require(len({p[1-axis] for p in squares}) == 1, "seed must be a straight run")
    squares.sort(key=lambda p: p[axis])
    side_length = squares[0][2]
    _require(all(p[2] == side_length for p in squares), "seed sizes differ")
    diffs = np.diff([p[axis] for p in squares])
    stride = int(diffs[0])
    _require(stride > side_length and np.all(diffs == stride), "seed stride is not uniform")
    return dict(inp=inp, bg=bg, frame=frame, control=control, a=a, b=b, box=box,
                body=body, cut=cut, marker=marker, marked=marked, other=other,
                region_side=side, squares=squares, axis=axis, side=side_length,
                stride=stride, sources=sources, seed_mask=seed_mask)


def _matches(d, mode="regional"):
    """All feasible origins; in regional mode each square chooses its own plan."""
    h, w = d['a'].shape
    length, axis, stride = len(d['squares']), d['axis'], d['stride']
    H, W = d['body'].shape
    result = []
    for r in range(h - (length-1 if axis == 0 else 0)):
        for c in range(w - (length-1 if axis == 1 else 0)):
            base = (d['squares'][0][0]-r*stride, d['squares'][0][1]-c*stride)
            if not (0 <= base[0] and 0 <= base[1]
                    and base[0]+(h-1)*stride+d['side'] <= H
                    and base[1]+(w-1)*stride+d['side'] <= W):
                continue
            good = True
            for k, square in enumerate(d['squares']):
                p = r+(axis == 0)*k, c+(axis == 1)*k
                if mode == "geometry":
                    continue
                source = d['a'] if mode == 'a' or (mode == 'regional' and square[4] == 0) else d['b']
                if int(source[p]) != square[3]:
                    good = False
                    break
            if good:
                result.append(((r, c), base))
    return result


def _decode(inp):
    d = _extract(inp)
    matches = _matches(d)
    _require(len(matches) == 1, "regional seed must determine exactly one origin")
    d['match'], d['base'] = matches[0]
    return d


def _layers(d):
    a = np.full(d['body'].shape, d['bg'], np.int8)
    b = a.copy()
    tiles = np.zeros(a.shape, bool)
    for (r, c), value in np.ndenumerate(d['a']):
        y, x = d['base'][0]+r*d['stride'], d['base'][1]+c*d['stride']
        q = np.s_[y:y+d['side'], x:x+d['side']]
        a[q], b[q], tiles[q] = value, d['b'][r, c], True
    return a, b, tiles


def _answer(d, mode="regional"):
    a, b, tiles = _layers(d)
    bg = d['bg']
    result = np.full(d['body'].shape, bg, np.int8)
    if mode == 'a':
        result = a
    elif mode == 'b':
        result = b
    elif mode == 'union':
        result = np.where(a != bg, a, b).astype(np.int8)
    else:
        marked = d['other'] if mode == 'reverse_marker' else d['marked']
        other = d['marked'] if mode == 'reverse_marker' else d['other']
        if mode == 'whole_tile':
            for r, c in np.ndindex(d['a'].shape):
                y, x = d['base'][0]+r*d['stride'], d['base'][1]+c*d['stride']
                center = y+d['side']//2, x+d['side']//2
                value = d['a'][r, c] if marked[center] or d['cut'][center] else d['b'][r, c]
                result[y:y+d['side'], x:x+d['side']] = value
        else:
            result[marked] = a[marked]
            result[other] = b[other]
            seam = np.where(a != bg, a, b)
            if mode == 'reverse_seam':
                seam = np.where(b != bg, b, a)
            elif mode == 'a_seam_only':
                seam = a
            result[d['cut']] = seam[d['cut']]
    out = d['inp'].copy()
    r, c, y, x = d['box']
    out[r+1:y-1, c+1:x-1] = result
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the common lattice by joint matching; consume the work-card guides."""
    return _answer(_decode(input_grid))


_TEACHING = {'marked_foreground', 'marked_blank', 'other_foreground', 'other_blank',
             'seam_priority', 'seam_fallback', 'pixel_split', 'joint_alignment'}


def _evidence(d):
    a, b, tiles = _layers(d)
    bg, m, o, cut = d['bg'], d['marked'], d['other'], d['cut']
    missing = tiles & ~d['seed_mask']
    correct = _answer(d)
    r0, c0, r1, c1 = d['box']
    quilt = correct[r0+1:r1-1, c0+1:c1-1]
    split_tile = False
    for r, c in np.ndindex(d['a'].shape):
        y, x = d['base'][0]+r*d['stride'], d['base'][1]+c*d['stride']
        tile = quilt[y:y+d['side'], x:x+d['side']]
        split_tile |= int(tile.min()) != int(tile.max())
    geometry = _matches(d, 'geometry')
    joint = (len({s[4] for s in d['squares']}) == 2
             and not _matches(d, 'a') and not _matches(d, 'b') and len(geometry) >= 2)
    # Merely having a second geometric fit is insufficient: ignoring the color
    # match must also permit an actually different completed quilt.
    if joint:
        joint = any(base != d['base'] and not np.array_equal(
            _answer(dict(d, match=match, base=base)), correct) for match, base in geometry)
    flags = {
        'marked_foreground': np.any(missing & m & (a != bg)),
        'marked_blank': np.any(missing & m & (a == bg) & (b != bg)),
        'other_foreground': np.any(missing & o & (b != bg)),
        'other_blank': np.any(missing & o & (b == bg) & (a != bg)),
        'seam_priority': np.any(missing & cut & (a != bg) & (b != bg) & (a != b)),
        'seam_fallback': np.any(missing & cut & (a == bg) & (b != bg)),
        'pixel_split': split_tile,
        'joint_alignment': joint,
    }
    return {k for k, v in flags.items() if v}


def _features(d):
    h, w = d['a'].shape
    cut = d['cut']
    orthogonal = len(_components(cut)) == 1
    bends = 0
    if orthogonal:
        for r, c in np.argwhere(cut):
            vertical = (r > 0 and cut[r-1, c]) or (r+1 < cut.shape[0] and cut[r+1, c])
            horizontal = (c > 0 and cut[r, c-1]) or (c+1 < cut.shape[1] and cut[r, c+1])
            bends += int(vertical and horizontal)
    kind = ('diagonal' if not orthogonal else 'straight' if bends == 0 else
            'elbow' if bends == 1 else 'step' if bends == 2 else 'double_step')
    axis, length = d['axis'], len(d['squares'])
    start, extent = d['match'][axis], d['a'].shape[axis]
    segment = 'whole' if length == extent else 'prefix' if start == 0 else 'suffix' if start+length == extent else 'interior'
    return dict(canvas=d['inp'].shape, plan=(h, w), side=d['side'], gutter=d['stride']-d['side'],
                axis=axis, length=length, base=d['base'], match=d['match'], segment=segment,
                region_side=d['region_side'], background=d['bg'], cut_kind=kind,
                orthogonal_bends=bends if orthogonal else None,
                cross_interior=0 < d['match'][1-axis] < d['a'].shape[1-axis]-1,
                rack=0 if d['sources'][0][0][0] == d['sources'][1][0][0] else 1,
                holes=int(np.count_nonzero(d['a'] == d['bg'])+np.count_nonzero(d['b'] == d['bg'])),
                mixed_seed=len({s[4] for s in d['squares']}) == 2,
                target=d['box'], palette=tuple(map(int, np.unique(d['inp']))))


def _border(grid, r, c, h, w, color):
    grid[r:r+h, c] = grid[r:r+h, c+w-1] = color
    grid[r, c:c+w] = grid[r+h-1, c:c+w] = color


def _pack(rng, a, b, body, answer, bg, frame, rack):
    """Two source cards in a rack; work card on either side of that rack."""
    ph, pw = 2*a.shape[0]+3, 2*a.shape[1]+3
    th, tw = body.shape[0]+2, body.shape[1]+2
    gap = 1
    if rack == 0:
        rh, rw = ph, 2*pw+gap
        total = (rh+gap+th, max(rw, tw))
    else:
        rh, rw = 2*ph+gap, pw
        total = (max(rh, th), rw+gap+tw)
    _require(max(total) <= 28, "layout does not fit")
    shape = tuple(n+2+int(rng.integers(min(2, 28-n)+1)) for n in total)
    offset = [int(rng.integers(shape[d]-total[d]-1))+1 for d in (0, 1)]
    rr, rc = offset
    tr, tc = offset
    order = int(rng.integers(2))
    if rack == 0:
        if order:
            rr += th+gap
        else:
            tr += rh+gap
        rc += int(rng.integers(total[1]-rw+1))
        tc += int(rng.integers(total[1]-tw+1))
    else:
        if order:
            rc += tw+gap
        else:
            tc += rw+gap
        rr += int(rng.integers(total[0]-rh+1))
        tr += int(rng.integers(total[0]-th+1))
    inp = np.full(shape, bg, np.int8)
    for k, plan in enumerate((a, b)):
        r, c = rr+k*(ph+gap)*(rack == 1), rc+k*(pw+gap)*(rack == 0)
        _border(inp, r, c, ph, pw, frame)
        inp[r+2:r+ph-2:2, c+2:c+pw-2:2] = plan
    _border(inp, tr, tc, th, tw, frame)
    inp[tr+1:tr+th-1, tc+1:tc+tw-1] = body
    out = inp.copy()
    out[tr+1:tr+th-1, tc+1:tc+tw-1] = answer
    return {'input': inp, 'output': out}


@lru_cache(None)
def _geometry(axis, interior, joint, avoid_side):
    options = []
    for h in range(2, 6):
        for w in range(2, 6):
            if interior and min(h, w) < 3:
                continue
            for side in range(2, 6):
                if side == avoid_side or (joint and side < 3):
                    continue
                for gutter in range(1, 4):
                    stride = side+gutter
                    bh, bw = (h-1)*stride+side, (w-1)*stride+side
                    ph, pw = 2*h+3, 2*w+3
                    for rack in (0, 1):
                        max_h, max_w = (25-ph, 26) if rack == 0 else (26, 25-pw)
                        rack_extent = 2*pw+1 if rack == 0 else 2*ph+1
                        if rack_extent <= 28 and bh+2 <= max_h and bw+2 <= max_w:
                            # Reserve enough translation freedom for color matching.
                            if max_h-bh >= stride+2 or max_w-bw >= stride+2:
                                options.append((h, w, side, gutter, rack, max_h, max_w))
    return tuple(options)


def _cut(rng, shape, kind):
    H, W = shape
    cut = np.zeros(shape, bool)
    if kind == 'straight':
        if rng.integers(2):
            cut[int(rng.integers(2, H-2)), :] = True
        else:
            cut[:, int(rng.integers(2, W-2))] = True
    elif kind == 'diagonal':
        x0, x1 = sorted(int(x) for x in rng.choice(np.arange(1, W-1), 2, replace=False))
        if x1-x0 > H-1:
            x1 = x0+H-1
        cols = np.rint(np.linspace(x0, x1, H)).astype(int)
        cut[np.arange(H), cols] = True
    elif kind == 'elbow':
        r, c = int(rng.integers(2, H-2)), int(rng.integers(2, W-2))
        cut[:r+1, c] = True
        cut[r, c:] = True
    else:
        rows = list(range(2, H-2, 3))
        count = min(2, len(rows)) if kind == 'double_step' else 1
        rows = sorted(int(x) for x in rng.choice(rows, count, replace=False))
        c, last = int(rng.integers(1, W-1)), 0
        for r in rows:
            choices = [x for x in range(1, W-1) if abs(x-c) >= 2]
            if not choices:
                return _cut(rng, shape, 'elbow')
            new = int(rng.choice(choices))
            cut[last:r+1, c] = True
            cut[r, min(c, new):max(c, new)+1] = True
            c, last = new, r
        cut[last:, c] = True
    if rng.integers(2):
        cut = cut[::-1].copy()
    if rng.integers(2):
        cut = cut[:, ::-1].copy()
    return cut


def _sample_plans(rng, shape, bg, palette):
    plans = []
    for _ in range(2):
        style = int(rng.integers(5))
        plan = rng.choice(palette, size=shape).astype(np.int8)
        if style == 1:
            plan[:] = rng.choice(palette, size=(shape[0], 1))
        elif style == 2:
            plan[:] = rng.choice(palette, size=(1, shape[1]))
        elif style == 3 and shape[0] >= 3:
            i, j = rng.choice(shape[0], 2, replace=False)
            plan[i] = plan[j]
        elif style == 4:
            rr, cc = np.indices(shape)
            plan = np.asarray(palette, np.int8)[(rr+cc) % len(palette)]
        density = float(rng.choice([0, .15, .25, .4]))
        plan[rng.random(shape) < density] = bg
        plans.append(plan)
    return plans


def _build(rng, a, b, bg, frame, control, cut, marker, base, side, stride, axis, match, length, rack):
    regions = _regions(cut)
    marked = regions[0] if regions[0][marker] else regions[1]
    # Generation paints tile cells directly, separately from solver rendering.
    result = np.full(cut.shape, bg, np.int8)
    seed = np.zeros(cut.shape, bool)
    for r, c in np.ndindex(a.shape):
        y, x = base[0]+r*stride, base[1]+c*stride
        for dy in range(side):
            for dx in range(side):
                p = y+dy, x+dx
                if cut[p]:
                    value = a[r, c] if a[r, c] != bg else b[r, c]
                else:
                    value = a[r, c] if marked[p] else b[r, c]
                result[p] = value
    for k in range(length):
        r, c = match[0]+(axis == 0)*k, match[1]+(axis == 1)*k
        y, x = base[0]+r*stride, base[1]+c*stride
        seed[y:y+side, x:x+side] = True
    body = np.full(cut.shape, bg, np.int8)
    body[seed] = result[seed]
    body[cut] = control
    body[marker] = control
    return _pack(rng, a, b, body, result, bg, frame, rack)


def _sample_pair(rng, axis, marker_side, kind, *, interior=False, joint=False,
                 boundary=False, avoid_side=-1, required=frozenset(), seen=frozenset()):
    options = _geometry(axis, interior, joint, avoid_side)
    _require(options, "no feasible geometry")
    for attempt in range(320):
        h, w, side, gutter, rack, max_h, max_w = options[int(rng.integers(len(options)))]
        stride = side+gutter
        box = ((h-1)*stride+side, (w-1)*stride+side)
        shape = tuple(int(rng.integers(n+2, min(n+2+2*stride, limit)+1))
                      for n, limit in zip(box, (max_h, max_w)))
        if kind == 'double_step' and shape[0] < 8:
            continue  # Two separated step rows, rather than silently one step.
        base = tuple(int(rng.integers(1, n-b)) for n, b in zip(shape, box))
        if joint:
            # A step occupies a gutter across the clue line, then enters a
            # missing tile. The run itself stays made of intact solid squares.
            ch, cw = (h, w) if axis == 1 else (w, h)
            canonical_shape = shape if axis == 1 else shape[::-1]
            canonical_base = base if axis == 1 else base[::-1]
            row = int(rng.choice([0, ch-1]))
            length = int(rng.integers(2, cw+1))
            start = int(rng.integers(cw-length+1))
            split = int(rng.integers(start, start+length-1))
            gap_col = canonical_base[1]+split*stride+side+int(rng.integers(gutter))
            tile_col = canonical_base[1]+int(rng.integers(cw))*stride+side//2
            between = (0 if row == 0 else ch-2)
            bend_row = canonical_base[0]+between*stride+side+int(rng.integers(gutter))
            cut = np.zeros(canonical_shape, bool)
            upper, lower = (gap_col, tile_col) if row == 0 else (tile_col, gap_col)
            cut[:bend_row+1, upper] = True
            cut[bend_row, min(upper, lower):max(upper, lower)+1] = True
            cut[bend_row:, lower] = True
            cut = cut if axis == 1 else cut.T.copy()
            chosen = ((row, start) if axis == 1 else (start, row), length)
        else:
            cut = _cut(rng, shape, kind)
            chosen = None
        try:
            regions = _regions(cut)
        except AssertionError:
            continue
        colors = [int(x) for x in rng.permutation(10)]
        bg, frame, control = colors[:3]
        palette = colors[3:3+int(rng.integers(2, 6))]
        a, b = _sample_plans(rng, (h, w), bg, palette)
        near = _near(cut)
        possibilities = []
        for length in range(2, (w if axis == 1 else h)+1):
            if interior and length == (w if axis == 1 else h):
                continue
            for r in range(h-(length-1 if axis == 0 else 0)):
                for c in range(w-(length-1 if axis == 1 else 0)):
                    if chosen is not None and ((r, c), length) != chosen:
                        continue
                    if interior and not 0 < (r if axis == 1 else c) < (h if axis == 1 else w)-1:
                        continue
                    if boundary and (r if axis == 1 else c) not in (0, (h if axis == 1 else w)-1):
                        continue
                    valid = True
                    for k in range(length):
                        p = r+(axis == 0)*k, c+(axis == 1)*k
                        y, x = base[0]+p[0]*stride, base[1]+p[1]*stride
                        q = np.s_[y:y+side, x:x+side]
                        region = 0 if np.all(regions[0][q]) else 1 if np.all(regions[1][q]) else -1
                        if region < 0:
                            valid = False
                            break
                        source = a if region == marker_side else b
                        if joint:
                            source[p] = int(rng.choice(palette))
                        if source[p] == bg:
                            valid = False
                            break
                    if valid:
                        possibilities.append(((r, c), length))
        if not possibilities:
            continue
        match, length = possibilities[int(rng.integers(len(possibilities)))]
        seed = np.zeros(shape, bool)
        for k in range(length):
            r, c = match[0]+(axis == 0)*k, match[1]+(axis == 1)*k
            y, x = base[0]+r*stride, base[1]+c*stride
            seed[y:y+side, x:x+side] = True
        sites = np.argwhere(regions[marker_side] & ~near & ~seed)
        if not len(sites):
            continue
        marker = tuple(map(int, sites[int(rng.integers(len(sites)))]))
        pair = _build(rng, a, b, bg, frame, control, cut, marker, base,
                      side, stride, axis, match, length, rack)
        if _key(pair['input']) in seen:
            continue
        try:
            d = _decode(pair['input'])
        except AssertionError:
            continue
        flags = _evidence(d)
        if (required <= flags and (not joint or 'joint_alignment' in flags)
                and not np.array_equal(pair['input'], pair['output'])):
            return pair
    return _fallback(rng, axis, marker_side, joint, avoid_side, seen, interior, kind)


def _fallback(rng, axis, marker_side, joint, avoid_side, seen, interior, kind):
    """Finite constructive fallback, with the same rule and visible evidence.

    A joint fallback uses a gutter-to-tile step. A teaching fallback places the
    three semantic seam states in three miniature entries, not in pixel noise.
    Interior/complex tests have their own constructive geometries, preserving
    the requested shift even when ordinary conditional sampling is exhausted.
    """
    if interior or kind == 'double_step':
        for trial in range(240):
            colors = [int(x) for x in rng.permutation(10)]
            bg, frame, control, ca, cb, cc = colors[:6]
            shape = (16, 16) if interior else (14, 18)
            plan_shape = (3, 3) if interior else (2, 3)
            a = np.full(plan_shape, ca, np.int8)
            b = a.copy()
            row = 1 if interior else 0
            a[row, 1:], b[row, 1:] = [cb, cc], [cb, cc]
            b[-1, 1] = cb
            cut = np.zeros(shape, bool)
            bend = 8 if interior else 4
            cut[:bend+1, 4] = True
            cut[bend, 4:7] = True
            cut[bend:, 6] = True
            if not interior:
                cut[10:, 6] = False
                cut[9, 6:13] = True
                cut[9:, 12] = True
            regions = _regions(cut)
            sites = np.argwhere(regions[marker_side] & ~_near(cut))
            marker = tuple(map(int, sites[-1]))
            pair = _build(rng, a, b, bg, frame, control, cut, marker, (1, 1),
                          3, 4, 1, (row, 1), 2, 0)
            if axis == 0:
                pair = {k: v.T.copy() for k, v in pair.items()}
            d = _decode(pair['input'])
            if _key(pair['input']) not in seen and 'pixel_split' in _evidence(d):
                return pair
        raise AssertionError("shifted fallback could not exclude duplicates")
    side = 3 if avoid_side != 3 else 4
    stride = side+1
    shape = (2*side+7, 3*side+9)
    base = (1, 1)
    for trial in range(240):
        colors = [int(x) for x in rng.permutation(10)]
        bg, frame, control, ca, cb, cc = colors[:6]
        a = np.array([[ca, cb, cc], [ca, bg, cc]], np.int8)
        b = np.array([[cc, ca, cb], [cb, cb, bg]], np.int8)
        cut = np.zeros(shape, bool)
        if joint:
            gap = base[1]+stride+side
            center = base[1]+stride+side//2
            bend = base[0]+side
            cut[:bend+1, gap] = True
            cut[bend, center:gap+1] = True
            cut[bend:, center] = True
            # Either marker side gives a joint sequence present in neither
            # plan by itself, with a different geometry-only alternative.
            a[0], b[0] = [cb, ca, cb], [ca, cb, cc]
        else:
            cut[base[0]+stride+side//2, :] = True
        regions = _regions(cut)
        sites = np.argwhere(regions[marker_side] & ~_near(cut))
        marker = tuple(map(int, sites[-1]))
        pair = _build(rng, a, b, bg, frame, control, cut, marker, base,
                      side, stride, 1, (0, 1), 2, 0)
        if axis == 0:
            pair = {k: v.T.copy() for k, v in pair.items()}
        try:
            d = _decode(pair['input'])
        except AssertionError:
            continue
        if _key(pair['input']) in seen:
            continue
        if joint and 'joint_alignment' not in _evidence(d):
            continue
        if not joint and not (_TEACHING-{'joint_alignment'}) <= _evidence(d):
            continue
        return pair
    raise AssertionError("constructive fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    axis, first_side = int(rng.integers(2)), int(rng.integers(2))
    regime = str(rng.choice(['interior_alignment', 'more_bends']))
    puzzle = {'train': [], 'test': []}
    seen, coverage = set(), set()
    first_scale = None
    for i in range(num_train):
        required = _TEACHING-coverage if i == num_train-1 else set()
        kind = 'straight' if i == 1 else str(rng.choice(['straight', 'elbow', 'step', 'diagonal']))
        pair = _sample_pair(rng, axis, first_side if i % 2 == 0 else 1-first_side, kind,
                            joint=(i == 0), boundary=(regime == 'interior_alignment'),
                            avoid_side=first_scale if i == 1 else -1,
                            required=required, seen=seen)
        d = _decode(pair['input'])
        if i == 0:
            first_scale = d['side']
        coverage |= _evidence(d)
        seen.add(_key(pair['input']))
        puzzle['train'].append(pair)
    for i in range(num_test):
        interior = regime == 'interior_alignment'
        kind = str(rng.choice(['elbow', 'step', 'diagonal'])) if interior else 'double_step'
        pair = _sample_pair(rng, axis, int(rng.integers(2)), kind,
                            interior=interior, required={'pixel_split'}, seen=seen)
        seen.add(_key(pair['input']))
        puzzle['test'].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, flags, sides, scales, axes = set(), set(), set(), set(), set()
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape and _key(inp) not in seen, "size mismatch or duplicate")
            seen.add(_key(inp))
            d = _decode(inp)
            features[split].append(_features(d))
            expected = _answer(d)
            _require(np.array_equal(out, expected), "incorrect tile, guide removal, or preserved cell")
            _require(not np.array_equal(inp, out), "identity pair")
            # Seed squares are correct observations and must survive completion.
            r, c, y, x = d['box']
            _require(np.array_equal(out[r+1:y-1, c+1:x-1][d['seed_mask']],
                                   d['body'][d['seed_mask']]), "seed changed")
            if split == 'train':
                flags |= _evidence(d)
                sides.add(d['region_side'])
                scales.add(d['side'])
                axes.add(d['axis'])
            else:
                _require(d['axis'] in axes, "seed axis was not taught")
                _require('pixel_split' in _evidence(d), "test does not require pixel-scale cutting")
    _require(_TEACHING <= flags, "training misses a rule distinction")
    _require(sides == {0, 1} and len(scales) >= 2, "marker choices or scales not demonstrated")
    train, test = features['train'], features['test']
    alignment_shift = (all(not f['cross_interior'] for f in train)
                       and all(f['cross_interior'] and f['segment'] != 'whole' for f in test))
    bend_shift = (all(f['cut_kind'] != 'double_step' for f in train)
                  and all(f['cut_kind'] == 'double_step' for f in test))
    _require(alignment_shift or bend_shift, "neither documented train/test shift is present")


def _signature(puzzle):
    return tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded positive, negative, ablation and diversity checks over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_signature(p) for p in puzzles}) == 42, "collapsed seed diversity")
    _require(_signature(puzzles[0]) == _signature(generate_puzzle(0)), "not deterministic")
    records = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        ablations = {name: False for name in ('a', 'b', 'union', 'reverse_marker', 'whole_tile',
                                             'reverse_seam', 'a_seam_only')}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = np.asarray(pair['input'])
                frozen = inp.copy()
                _require(np.array_equal(solve(inp), pair['output']), "solver disagrees")
                _require(np.array_equal(inp, frozen), "solver mutated input")
                d = _decode(inp)
                records.append((split, _features(d)))
                if split == 'train':
                    for name in ablations:
                        ablations[name] |= not np.array_equal(_answer(d, name), pair['output'])
        _require(all(ablations.values()), "training admits a simpler compositor")
    features = [r for _, r in records]
    for key, minimum in (('canvas', 20), ('plan', 10), ('side', 4), ('gutter', 3),
                         ('match', 8), ('base', 12), ('target', 20), ('background', 8)):
        _require(len({r[key] for r in features}) >= minimum, f"insufficient variation in {key}")
    _require({r['axis'] for r in features} == {0, 1}, "one seed axis missing")
    _require(sum(r['mixed_seed'] for r in features) >= 42, "joint clues missing")
    _require(sum(r['holes'] >= 3 for r in features) >= 25, "holes missing")
    _require({'whole', 'prefix', 'suffix'} <= {r['segment'] for r in features}, "segment classes missing")
    _require({r['rack'] for r in features} == {0, 1}, "one rack orientation missing")
    _require({'straight', 'elbow', 'step', 'diagonal', 'double_step'} <=
             {r['cut_kind'] for r in features}, "cut variation missing")
    # Color permutation is a genuine semantic symmetry, including background.
    pair = puzzles[0]['train'][0]
    permutation = np.array([6, 2, 9, 0, 4, 8, 1, 7, 5, 3], np.int8)
    _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
             "literal color roles failed")
    # Transpose keeps source order and maps a horizontal rack to a vertical one.
    _require(np.array_equal(solve(pair['input'].T), pair['output'].T), "transpose failed")
    _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), "list input failed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(6000+nt*10+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    corrupted = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed margin was accepted")
    # Ambiguity is tested on the actual input-only matcher, without labels.
    d = _decode(pair['input'])
    ambiguous = dict(d)
    ambiguous['a'] = np.full_like(d['a'], d['squares'][0][3])
    ambiguous['b'] = ambiguous['a'].copy()
    ambiguous['squares'] = [tuple(list(s[:3])+[d['squares'][0][3], s[4]]) for s in d['squares']]
    _require(len(_matches(ambiguous)) == len(_matches(d, 'geometry')), "matching depends on hidden state")
    # A real ambiguous input must be rejected by the public input-only solver.
    joint_pair = next(p for p in puzzles[0]['train'] if 'joint_alignment' in _evidence(_decode(p['input'])))
    j = _decode(joint_pair['input'])
    invalid = joint_pair['input'].copy()
    color = j['squares'][0][3]
    for box, plan in j['sources']:
        r, c, y, x = box
        invalid[r+2:y-2:2, c+2:x-2:2] = color
    r, c, y, x = j['box']
    invalid[r+1:y-1, c+1:x-1][j['seed_mask']] = color
    try:
        solve(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous seed was accepted")
    return {'seeds': 42, 'pairs': len(features), 'distinct_puzzles': 42,
            'canvas_shapes': len({r['canvas'] for r in features}),
            'plan_shapes': len({r['plan'] for r in features}),
            'note': 'Generator-owned semantic checks; no independent solver or blind predictions.'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
