"""Unfold a port-directed journey into a minimal containment itinerary.

Only standard-library modules and NumPy are used. No import-time file access,
parent imports, persistent caches, execution-trace files, or hidden seed state.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _components(mask):
    h, w = mask.shape
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    h, w = padded.shape
    seen = np.zeros((h, w), dtype=bool)
    seen[0, 0] = True
    todo = [(0, 0)]
    while todo:
        r, c = todo.pop()
        for dr, dc in _DIRS:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and not padded[y, x] and not seen[y, x]:
                seen[y, x] = True
                todo.append((y, x))
    return (~padded & ~seen)[1:-1, 1:-1]


def _mask(cells, shape):
    out = np.zeros(shape, dtype=bool)
    for p in cells:
        out[p] = True
    return out


def _bbox(cells):
    a = np.asarray(cells)
    return tuple(map(int, (*a.min(axis=0), *a.max(axis=0))))


def _near(a, b):
    """Any contact, including diagonals, between two boolean masks."""
    padded = np.pad(a, 1)
    h, w = a.shape
    return any(np.any(padded[dy:dy+h, dx:dx+w] & b)
               for dy in range(3) for dx in range(3))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'grid bounds'
    assert np.issubdtype(a.dtype, np.integer), 'integer cells required'
    assert np.all((0 <= a) & (a <= 9)), 'color bounds'
    return a


def _decode(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    bg = int(g[0, 0])
    assert all(int(g[p]) == bg for p in ((0, w-1), (h-1, 0), (h-1, w-1)))
    components = _components(g != bg)
    boundary = [a for a in components if any(r in (0, h-1) or c in (0, w-1) for r, c in a)]
    assert len(boundary) == 1, 'one boundary launcher required'
    launcher = boundary[0]
    ends = [p for p in launcher if p[0] in (0, h-1) or p[1] in (0, w-1)]
    assert len(ends) == 1 and len(launcher) >= 3
    r, c = ends[0]
    initial = 2 if r == 0 else 0 if r == h-1 else 1 if c == 0 else 3
    dr, dc = _DIRS[initial]
    line = [(r+k*dr, c+k*dc) for k in range(len(launcher))]
    assert set(line) == set(launcher), 'straight contiguous launcher'
    signal, pc = int(g[line[0]]), int(g[line[-1]])
    assert signal != pc and all(int(g[p]) == signal for p in line[:-1])
    relays, contours, wall_colors, body_colors = [], [], set(), set()
    masks = []
    for cells in components:
        m = _mask(cells, g.shape)
        masks.append(m)
        if cells is launcher:
            continue
        values, counts = np.unique([int(g[p]) for p in cells], return_counts=True)
        if len(values) == 1:
            hole = _holes(m)
            assert hole.any() and len(_components(hole)) == 1, 'one closed region per contour'
            contours.append(dict(mask=m, hole=hole, cells=cells, bbox=_bbox(cells)))
            wall_colors.add(int(values[0]))
            continue
        assert len(values) == 2 and pc in values
        ports = [p for p in cells if int(g[p]) == pc]
        assert len(ports) == 1 and len(cells) >= 5
        body = m & (g != pc)
        assert len(_components(body)) == 1 and not _holes(m).any(), 'filled relay'
        body_colors.update(int(v) for v in values if v != pc)
        top, left, bottom, right = _bbox(cells)
        pr, px = ports[0]
        sides = [i for i, yes in enumerate((pr == top, px == right, pr == bottom, px == left)) if yes]
        assert len(sides) == 1, 'port on one extreme side, away from corners'
        side = sides[0]
        inward = (pr-_DIRS[side][0], px-_DIRS[side][1])
        assert body[inward], 'port has an inward body neighbor'
        relays.append(dict(cells=cells, mask=m, port=ports[0], direction=side, bbox=(top,left,bottom,right)))
    assert relays and contours and len(wall_colors) == len(body_colors) == 1
    wall, carrier = next(iter(wall_colors)), next(iter(body_colors))
    assert len({bg, signal, pc, wall, carrier}) == 5
    for i, a in enumerate(masks):
        assert all(not _near(a, b) for b in masks[i+1:]), 'separate components need clearance'
    for i, a in enumerate(contours):
        for b in contours[i+1:]:
            ab = a['hole'][b['mask']]
            ba = b['hole'][a['mask']]
            assert ab.all() or not ab.any()
            assert ba.all() or not ba.any()
            assert ab.all() or ba.all() or not np.any(a['hole'] & b['hole']), 'laminar holes'
    for relay in relays:
        ids = []
        for i, contour in enumerate(contours):
            inside = contour['hole'][relay['mask']]
            assert inside.all() or not inside.any(), 'partial enclosure'
            if inside.all():
                ids.append(i)
        relay['stack'] = tuple(sorted(ids, key=lambda i: -int(contours[i]['hole'].sum())))
    return dict(grid=g, background=bg, signal=signal, port_color=pc,
                wall=wall, carrier=carrier, launcher=line, cap=line[-1],
                initial=initial, relays=relays, contours=contours)


def _trace(info):
    """Use original relay footprints; walls and earlier ray segments are transparent."""
    owner = {p: i for i, obj in enumerate(info['relays']) for p in obj['cells']}
    h, w = info['grid'].shape
    origin, direction = info['cap'], info['initial']
    route, transitions = [], []
    for _ in range(len(info['relays']) + 1):
        dr, dc = _DIRS[direction]
        r, c = origin
        while True:
            r, c = r+dr, c+dc
            if not (0 <= r < h and 0 <= c < w):
                return route, transitions
            hit = owner.get((r, c))
            if hit is not None:
                break
        assert hit not in route, 'relay revisits and cycles are outside the family'
        relay = info['relays'][hit]
        route.append(hit)
        transitions.append((direction, relay['direction']))
        origin, direction = relay['port'], relay['direction']
    raise AssertionError('unterminated route')


def _events(stacks):
    """Return (first stop, exclusive last stop, depth, contour identity)."""
    events = []
    def visit(start, end, depth):
        i = start
        while i < end:
            if len(stacks[i]) == depth:
                i += 1
                continue
            contour = stacks[i][depth]
            j = i+1
            while j < end and len(stacks[j]) > depth and stacks[j][depth] == contour:
                j += 1
            events.append((i, j, depth+1, contour))
            visit(i, j, depth+1)
            i = j
    visit(0, len(stacks), 0)
    return events


def _itinerary(stacks):
    assert stacks, 'the generated family has at least one reached relay'
    def draw(start, end, depth):
        tiles, i = [], start
        while i < end:
            if len(stacks[i]) == depth:
                tile = np.ones((1, 1), dtype=bool)
                i += 1
            else:
                contour = stacks[i][depth]
                j = i+1
                while j < end and len(stacks[j]) > depth and stacks[j][depth] == contour:
                    j += 1
                tile = np.pad(draw(i, j, depth+1), 2)
                tile[0, :] = tile[-1, :] = True
                tile[:, 0] = tile[:, -1] = True
                i = j
            tiles.append(tile)
        height = max(t.shape[0] for t in tiles)
        width = sum(t.shape[1] for t in tiles) + len(tiles)-1
        result = np.zeros((height, width), dtype=bool)
        x = 0
        for tile in tiles:
            y = (height-tile.shape[0])//2
            result[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
            x += tile.shape[1]+1
        return result
    return draw(0, len(stacks), 0)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the chronological scope diagram entirely from this input."""
    info = _decode(np.asarray(input_grid))
    route, _ = _trace(info)
    stacks = [info['relays'][i]['stack'] for i in route]
    return np.where(_itinerary(stacks), info['signal'], info['background']).astype(np.int8)


def _facts(info, route, transitions):
    stacks = [info['relays'][i]['stack'] for i in route]
    events = _events(stacks)
    occurrences = Counter(e[3] for e in events)
    spatial = sorted(route, key=lambda i: info['relays'][i]['bbox'][:2])
    ordered_diff = not np.array_equal(_itinerary(stacks), _itinerary([info['relays'][i]['stack'] for i in spatial]))
    return dict(stops=len(route), decoys=len(info['relays'])-len(route),
                contours=len(info['contours']), depth=max(map(len, stacks)),
                frames=len(events), reentry=any(n > 1 for n in occurrences.values()),
                grouped=any(end-start >= 2 for start, end, _, _ in events),
                ordered_diff=ordered_diff, turns=sum(a != b for a, b in transitions),
                straight=sum(a == b for a, b in transitions),
                initial=info['initial'], stacks=stacks, events=events)


def _axis(rng, count, witness):
    for _ in range(80):
        if witness:
            sizes = [3]*count
            gaps = list(map(int, rng.integers(6, 9, size=max(0, count-1))))
            before, after = map(int, rng.integers(8, 10, size=2))
        else:
            sizes = list(map(int, rng.choice([3, 5], size=count, p=[.75, .25])))
            gaps = list(map(int, rng.integers(4, 9, size=max(0, count-1))))
            before, after = map(int, rng.integers(5, 9, size=2))
        length = before+after+sum(sizes)+sum(gaps)
        if length <= 30:
            centers, pos = [], before
            for i, size in enumerate(sizes):
                centers.append(pos+size//2)
                pos += size + (gaps[i] if i < len(gaps) else 0)
            return length, centers, sizes
    sizes, gaps = [3]*count, [4]*(count-1)
    before = 5
    return 10+sum(sizes)+sum(gaps), [before+1+7*i for i in range(count)], sizes


def _find_path(rng, slots, length, require_straight=False):
    """Find a self-avoiding nearest-rook path with boundary entry and exit."""
    neighbors = {}
    for i, (r, c) in enumerate(slots):
        options = {}
        for d, (dr, dc) in enumerate(_DIRS):
            candidates = [((rr-r)*dr+(cc-c)*dc, j) for j, (rr, cc) in enumerate(slots)
                          if (rr-r)*dc-(cc-c)*dr == 0 and (rr-r)*dr+(cc-c)*dc > 0]
            if candidates:
                options[d] = min(candidates)[1]
        neighbors[i] = options
    starts = [(i, d) for i in range(len(slots)) for d in range(4) if (d+2)%4 not in neighbors[i]]
    rng.shuffle(starts)
    budget = [2500]
    def grow(path, directions):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        current = path[-1]
        if len(path) == length:
            exits = [d for d in range(4) if d not in neighbors[current]]
            if require_straight and not any(a == b for a,b in zip([incoming]+directions,directions)):
                exits = [d for d in exits if d == (directions[-1] if directions else incoming)]
            if exits:
                return path, directions+[int(rng.choice(exits))]
            return None
        options = list(neighbors[current].items())
        rng.shuffle(options)
        for d, nxt in options:
            if nxt not in path:
                result = grow(path+[nxt], directions+[d])
                if result is not None:
                    return result
        return None
    for start, incoming in starts:
        result = grow([start], [])
        if result is not None:
            return result[0], result[1], incoming
    return None


def _relay(rng, row, col, height, width, direction):
    top, left = row-height//2, col-width//2
    dr, dc = _DIRS[direction]
    port = (row+dr*(height//2), col+dc*(width//2))
    cells = {(top+r, left+c) for r in range(height) for c in range(width)}
    kind = int(rng.integers(4))
    if kind:
        for r, c in ((top,left), (top,left+width-1), (top+height-1,left), (top+height-1,left+width-1)):
            if kind == 1 or rng.random() < .7:
                cells.discard((r, c))
    if kind == 3:
        # Peeling off-axis boundary cells preserves both center approach lines.
        for p in sorted(cells):
            r, c = p
            if r != row and c != col and rng.random() < .12:
                candidate = cells-{p}
                local = _mask([(y-top, x-left) for y, x in candidate], (height,width))
                carrier = local.copy()
                carrier[port[0]-top,port[1]-left] = False
                if len(_components(local)) == 1 and len(_components(carrier)) == 1 and not _holes(local).any():
                    cells = candidate
    assert port in cells
    return dict(cells=sorted(cells), port=port, direction=direction, center=(row,col))


def _outline(rng, shape, bbox, pad):
    top, left, bottom, right = bbox
    top, left, bottom, right = top-pad, left-pad, bottom+pad, right+pad
    h, w = bottom-top+1, right-left+1
    if top < 1 or left < 1 or bottom >= shape[0]-1 or right >= shape[1]-1:
        return None
    filled = np.ones((h,w), dtype=bool)
    for corner in range(4):
        depth = int(rng.integers(0, min(3, pad+1)))
        for r in range(depth):
            for c in range(depth-r):
                rr = r if corner < 2 else h-1-r
                cc = c if corner%2 == 0 else w-1-c
                filled[rr,cc] = False
    if pad >= 3 and rng.random() < .6:
        side = int(rng.integers(4))
        if side%2 == 0:
            x = int(rng.integers(2, w-2))
            filled[0 if side == 0 else h-1, x:min(x+int(rng.integers(1,4)),w-2)] = False
        else:
            y = int(rng.integers(2, h-2))
            filled[y:min(y+int(rng.integers(1,4)),h-2), 0 if side == 3 else w-1] = False
    padded = np.pad(filled,1)
    interior = np.ones_like(filled)
    for dy in range(3):
        for dx in range(3):
            interior &= padded[dy:dy+h,dx:dx+w]
    local = filled & ~interior
    if len(_components(local)) != 1:
        return None
    hole_local = _holes(local)
    if not hole_local.any() or len(_components(hole_local)) != 1:
        return None
    mask = np.zeros(shape, dtype=bool)
    hole = mask.copy()
    mask[top:bottom+1,left:right+1] = local
    hole[top:bottom+1,left:right+1] = hole_local
    return dict(mask=mask, hole=hole, cells=list(map(tuple,np.argwhere(mask).tolist())), bbox=(top,left,bottom,right))


def _stacks(relays, contours, route):
    return [tuple(sorted((i for i, c in enumerate(contours) if c['hole'][relays[j]['mask']].all()),
                         key=lambda i: -int(contours[i]['hole'].sum()))) for j in route]


def _contour_ok(candidate, contours, relays, launch_mask):
    if candidate is None or _near(candidate['mask'], launch_mask):
        return False
    for relay in relays:
        if _near(candidate['mask'], relay['mask']):
            return False
        v = candidate['hole'][relay['mask']]
        if v.any() and not v.all():
            return False
    for other in contours:
        if _near(candidate['mask'], other['mask']):
            return False
        ab = candidate['hole'][other['mask']]
        ba = other['hole'][candidate['mask']]
        if not (ab.all() or not ab.any()) or not (ba.all() or not ba.any()):
            return False
        if not (ab.all() or ba.all()) and np.any(candidate['hole'] & other['hole']):
            return False
    return True


def _rectangle_groups(slots, relays):
    nr, nc = max(r for r,c in slots)+1, max(c for r,c in slots)+1
    groups = {}
    for r0 in range(nr):
        for r1 in range(r0,nr):
            for c0 in range(nc):
                for c1 in range(c0,nc):
                    ids = tuple(i for i,(r,c) in enumerate(slots) if r0 <= r <= r1 and c0 <= c <= c1)
                    if ids:
                        groups[ids] = _bbox([p for i in ids for p in relays[i]['cells']])
    return list(groups.items())


def _make_pair(rng, length, kind, palette):
    """Fixed intended length; bounded geometry retries with no stored payloads."""
    bg, wall, body, pc, signal = palette
    nested_mode = kind == 'any' and rng.random() < .55
    for attempt in range(180):
        if kind == 'return':
            nr = nc = 2
            extra = 0
        else:
            options = [(r,c) for r in (2,3) for c in (2,3) if r*c >= length+(kind == 'stay')]
            if nested_mode:
                options = [(r,c) for r,c in options if min(r,c) == 2]
            nr, nc = options[int(rng.integers(len(options)))]
            extra = int(rng.integers(1 if kind == 'stay' else 0, min(2,nr*nc-length)+1))
        chosen = sorted(map(int, rng.choice(nr*nc, size=length+extra, replace=False)))
        slots = [(i//nc,i%nc) for i in chosen]
        path = _find_path(rng,slots,length,require_straight=kind == 'return')
        if path is None:
            continue
        route, directions, initial = path
        if nested_mode and ((nr == 3 and initial % 2 == 0) or (nc == 3 and initial % 2 == 1)):
            continue
        h, rows, row_sizes = _axis(rng,nr,kind == 'return' or (nested_mode and nr == 2))
        w, cols, col_sizes = _axis(rng,nc,kind == 'return' or (nested_mode and nc == 2))
        relays = []
        for i,(r,c) in enumerate(slots):
            direction = directions[route.index(i)] if i in route else int(rng.integers(4))
            height = row_sizes[r] if rng.random() < .65 else 3
            width = col_sizes[c] if rng.random() < .65 else 3
            obj = _relay(rng,rows[r],cols[c],height,width,direction)
            obj['mask'] = _mask(obj['cells'],(h,w))
            relays.append(obj)
        rr, cc = relays[route[0]]['center']
        start = (h-1,cc) if initial == 0 else (rr,0) if initial == 1 else (0,cc) if initial == 2 else (rr,w-1)
        dr,dc = _DIRS[initial]
        launcher = [(start[0]+k*dr,start[1]+k*dc) for k in range(3)]
        launch_mask = _mask(launcher,(h,w))
        if any(_near(launch_mask,obj['mask']) for obj in relays):
            continue
        groups = _rectangle_groups(slots,relays)
        rng.shuffle(groups)
        contours = []
        for ids,bounds in groups:
            flags = [i in ids for i in route]
            if not any(flags):
                continue
            if kind == 'return' and sum(flag and (i == 0 or not flags[i-1]) for i,flag in enumerate(flags)) < 2:
                continue
            if kind == 'stay' and not any(a and b for a,b in zip(flags,flags[1:])):
                continue
            candidate = _outline(rng,(h,w),bounds,4 if kind == 'return' or nested_mode else int(rng.integers(2,5)))
            if _contour_ok(candidate,contours,relays,launch_mask):
                if nested_mode:
                    eligible = [i for i in ids if i in route]
                    rng.shuffle(eligible)
                    for i in eligible:
                        inner = _outline(rng,(h,w),_bbox(relays[i]['cells']),2)
                        if _contour_ok(inner,[candidate],relays,launch_mask):
                            proposed = _stacks(relays,[candidate,inner],route)
                            if max(_itinerary(proposed).shape) <= 30:
                                contours = [candidate,inner]
                                break
                    if not contours:
                        continue
                else:
                    contours.append(candidate)
                break
        if not contours:
            continue
        if kind == 'return':
            enclosed = [i for i in route if contours[0]['hole'][relays[i]['mask']].all()]
            rng.shuffle(enclosed)
            for i in enclosed:
                candidate = _outline(rng,(h,w),_bbox(relays[i]['cells']),2)
                if _contour_ok(candidate,contours,relays,launch_mask):
                    contours.append(candidate)
                    break
            if len(contours) < 2:
                continue
        desired = int(rng.integers(len(contours),5))
        for _ in range(16):
            if len(contours) >= desired:
                break
            ids,bounds = groups[int(rng.integers(len(groups)))]
            candidate = _outline(rng,(h,w),bounds,int(rng.integers(2,5)))
            if not _contour_ok(candidate,contours,relays,launch_mask):
                continue
            trial = _stacks(relays,contours+[candidate],route)
            if max(map(len,trial)) > 3 or max(_itinerary(trial).shape) > 30:
                continue
            contours.append(candidate)
        stacks = _stacks(relays,contours,route)
        output_mask = _itinerary(stacks)
        if max(output_mask.shape) > 30:
            continue
        inp = np.full((h,w),bg,dtype=np.int8)
        for c in contours:
            inp[c['mask']] = wall
        for obj in relays:
            inp[obj['mask']] = body
            inp[obj['port']] = pc
        for p in launcher[:-1]:
            inp[p] = signal
        inp[launcher[-1]] = pc
        out = np.where(output_mask,signal,bg).astype(np.int8)
        info = _decode(inp)
        observed, transitions = _trace(info)
        # Construction and decoder may index components differently.
        actual_centers = [info['relays'][i]['bbox'] for i in observed]
        intended = [_bbox(relays[i]['cells']) for i in route]
        if actual_centers != intended:
            continue
        f = _facts(info,observed,transitions)
        if kind == 'return' and not (f['reentry'] and f['depth'] >= 2 and f['ordered_diff'] and f['turns'] >= 2 and f['straight'] >= 1):
            continue
        if kind == 'stay' and not (f['grouped'] and f['decoys'] >= 1 and f['turns'] >= 1):
            continue
        assert np.array_equal(solve(inp),out)
        return {'input':inp,'output':out}
    return _fallback_pair(rng,length,kind,palette,nested_mode)


def _fallback_pair(rng, length, kind, palette, nested_mode=False):
    """Guaranteed packing; retain intended stops and demonstration obligations."""
    bg,wall,body,pc,signal = palette
    if kind == 'return':
        centers = [(9,9),(9,20),(20,20),(20,9)]
        directions = [1,2,3,2]
        boxes = [(4,4,25,14),(6,6,12,12)]
        initial,launcher = 1,[(9,0),(9,1),(9,2)]
    else:
        centers = [(9,6),(9,14),(9,22),(21,22),(21,14),(21,6)]
        directions = [1,1,2,3,3,2]
        directions[length-1] = 0 if length <= 3 else 2
        boxes = [(4,1,14,27)] if nested_mode else [(6,3,12,25)]
        if nested_mode:
            boxes.append((6,11,12,17))
        initial,launcher = 2,[(0,6),(1,6),(2,6)]
    inp = np.full((30,30),bg,dtype=np.int8)
    contours = []
    for top,left,bottom,right in boxes:
        mask = np.zeros(inp.shape,dtype=bool)
        mask[top,left:right+1] = mask[bottom,left:right+1] = True
        mask[top:bottom+1,left] = mask[top:bottom+1,right] = True
        inp[mask] = wall
        contours.append(dict(mask=mask,hole=_holes(mask)))
    relays = []
    for (r,c),direction in zip(centers,directions):
        cells = [(r+dr,c+dc) for dr in (-1,0,1) for dc in (-1,0,1)]
        peel = int(rng.integers(16))
        for bit,p in enumerate(((r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1))):
            if peel & (1 << bit):
                cells.remove(p)
        mask = _mask(cells,inp.shape)
        port = (r+_DIRS[direction][0],c+_DIRS[direction][1])
        inp[mask] = body
        inp[port] = pc
        relays.append(dict(mask=mask))
    for p in launcher[:-1]:
        inp[p] = signal
    inp[launcher[-1]] = pc
    stacks = _stacks(relays,contours,list(range(length)))
    out = np.where(_itinerary(stacks),signal,bg).astype(np.int8)
    # D4 changes source geometry without changing chronological output order.
    inp = np.rot90(inp,int(rng.integers(4))).copy()
    if rng.random() < .5:
        inp = inp[:,::-1].copy()
    info = _decode(inp)
    route,transitions = _trace(info)
    assert len(route) == length
    f = _facts(info,route,transitions)
    if kind == 'return':
        assert f['reentry'] and f['depth'] >= 2 and f['ordered_diff'] and f['straight'] >= 1
    if kind == 'stay':
        assert f['grouped'] and f['decoys'] >= 1 and f['turns'] >= 1
    assert np.array_equal(solve(inp),out)
    return {'input':inp,'output':out}


def _count(value, name, low, high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(grid):
    a = np.asarray(grid,dtype=np.int8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train,'num_train',2,10)
    num_test = _count(num_test,'num_test',1,4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    base = list(map(int,rng.choice(10,5,replace=False)))
    vary_palette = bool(rng.integers(2))
    puzzle, seen = {'train':[],'test':[]},set()
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            kind = 'return' if split == 'train' and index == 0 else 'stay' if split == 'train' and index == 1 else 'any'
            length = 4 if kind == 'return' else 3 if kind == 'stay' else int(rng.integers(2,5) if split == 'train' else rng.integers(5,7))
            palette = list(map(int,rng.choice(10,5,replace=False))) if vary_palette else base
            for _ in range(12):
                pair = _make_pair(rng,length,kind,palette)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('bounded uniqueness retries exhausted')
            seen.add(key)
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def _check_output(out, info, f):
    bg, signal = info['background'],info['signal']
    assert set(map(int,np.unique(out))) <= {bg,signal}
    records, dots, frames = [],[],[]
    for cells in _components(out != bg):
        mask = _mask(cells,out.shape)
        hole = _holes(mask)
        box = _bbox(cells)
        record = dict(mask=mask,hole=hole,bbox=box,cells=cells)
        records.append(record)
        if hole.any():
            top,left,bottom,right = box
            assert len(cells) == 2*(bottom-top+right-left)
            assert mask[top,left:right+1].all() and mask[bottom,left:right+1].all()
            assert mask[top:bottom+1,left].all() and mask[top:bottom+1,right].all()
            frames.append(record)
        else:
            assert len(cells) == 1, 'every output leaf is a dot'
            dots.append(record)
    dots.sort(key=lambda x:x['bbox'][1])
    assert len(dots) == f['stops'] and len(frames) == f['frames']
    assert all(x['bbox'][0] == out.shape[0]//2 for x in dots)
    assert out.shape == (1+4*f['depth'],2*f['stops']-1+4*f['frames'])
    signatures = []
    for frame in frames:
        contained = [i for i,dot in enumerate(dots) if frame['hole'][dot['cells'][0]]]
        assert contained and contained == list(range(min(contained),max(contained)+1))
        depth = 1+sum(other['hole'][frame['mask']].all() for other in frames if other is not frame)
        signatures.append((min(contained),max(contained)+1,depth))
        immediate = [x for x in records if x is not frame and frame['hole'][x['mask']].all()
                     and not any(y is not frame and y is not x and frame['hole'][y['mask']].all()
                                 and y['hole'][x['mask']].all() for y in frames)]
        bounds = [x['bbox'] for x in immediate]
        top,left,bottom,right = frame['bbox']
        assert min(b[0] for b in bounds) == top+2 and min(b[1] for b in bounds) == left+2
        assert max(b[2] for b in bounds) == bottom-2 and max(b[3] for b in bounds) == right-2
    assert sorted(signatures) == sorted((a,b,d) for a,b,d,_ in f['events'])
    assert (out[0] != bg).any() and (out[-1] != bg).any()
    assert (out[:,0] != bg).any() and (out[:,-1] != bg).any()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, facts = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _decode(inp)
            route,transitions = _trace(info)
            assert route, 'nonempty itinerary'
            f = _facts(info,route,transitions)
            assert f['depth'] > 0, 'A must contribute'
            expected = np.where(_itinerary(f['stacks']),info['signal'],info['background'])
            assert np.array_equal(out,expected), 'incorrect itinerary'
            _check_output(out,info,f)
            assert not np.array_equal(_itinerary([()]*f['stops']),out == info['signal']), 'A ablation'
            facts[split].append(f)
    assert any(f['reentry'] and f['depth'] >= 2 and f['ordered_diff'] and f['turns'] >= 2 and f['straight'] >= 1 for f in facts['train']), 'return/nesting/order/straight witness'
    assert any(f['grouped'] and f['decoys'] >= 1 and f['turns'] >= 1 for f in facts['train']), 'shared scope/erasure witness'
    assert max(f['stops'] for f in facts['train']) <= 4
    assert all(f['stops'] in (5,6) for f in facts['test'])
    return None


def self_test():
    """Stream 42 seeds; retain scalar counters and digests, never all puzzles."""
    shapes, structures, outputs, scopes, counts, palettes, initials = set(),set(),set(),set(),set(),set(),set()
    hist = {'stops':Counter(),'depth':Counter(),'contours':Counter(),'decoys':Counter()}
    split_hist = {s:{k:Counter() for k in hist} for s in ('train','test')}
    total = reentries = grouped = ablations = nonrect = 0
    port_changes = wall_redraws = 0
    rng = np.random.default_rng(92071)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp,out = pair['input'],pair['output']
                info = _decode(inp)
                route,transitions = _trace(info)
                f = _facts(info,route,transitions)
                total += 1
                shapes.add(inp.shape)
                counts.add((f['stops'],f['contours'],f['decoys']))
                initials.add(f['initial'])
                palettes.add((info['background'],info['wall'],info['carrier'],info['port_color'],info['signal']))
                mapping = {}
                normalized = bytes(mapping.setdefault(int(v),len(mapping)) for v in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape)+normalized).hexdigest())
                outputs.add(hashlib.sha256(bytes(out.shape)+(out == info['signal']).tobytes()).hexdigest())
                scopes.add(tuple(f['stacks']))
                for name in hist:
                    hist[name][f[name]] += 1
                    split_hist[split][name][f[name]] += 1
                reentries += f['reentry']
                grouped += f['grouped']
                for contour in info['contours']:
                    top,left,bottom,right = contour['bbox']
                    m = contour['mask']
                    nonrect += not (m[top,left:right+1].all() and m[bottom,left:right+1].all()
                                    and m[top:bottom+1,left].all() and m[top:bottom+1,right].all())
                # Color roles are visible and independent of literals.
                perm = rng.permutation(10)
                assert np.array_equal(solve(perm[inp]),perm[out])
                # Rotate the scene, not the chronological time axis.
                if split == 'test':
                    assert np.array_equal(solve(np.rot90(inp,1+seed%3)),out)
                    assert np.array_equal(solve(inp[:,::-1]),out)
                # Remove all walls: same route, but zero frames. Use the
                # geometry tracer directly because wall-free scenes are an ablation.
                no_walls = inp.copy()
                no_walls[no_walls == info['wall']] = info['background']
                owner = {p:i for i,o in enumerate(info['relays']) for p in o['cells']}
                assert all(no_walls[p] != info['background'] for p in owner)
                assert _trace(dict(info,grid=no_walls,contours=[]))[0] == route
                assert not np.array_equal(_itinerary([()]*len(route)),out == info['signal'])
                if f['decoys']:
                    all_ids = sorted(range(len(info['relays'])),key=lambda i:info['relays'][i]['bbox'][:2])
                    all_stacks = [info['relays'][i]['stack'] for i in all_ids]
                    assert not np.array_equal(_itinerary(all_stacks),out == info['signal'])
                    ablations += 1
                # Redraw one irregular wall as its rectangular bounding frame.
                # Accept only a valid redraw with the same actual memberships.
                if split == 'test':
                    for contour in info['contours']:
                        top,left,bottom,right = contour['bbox']
                        rectangular = np.zeros(inp.shape,dtype=bool)
                        rectangular[top,left:right+1] = rectangular[bottom,left:right+1] = True
                        rectangular[top:bottom+1,left] = rectangular[top:bottom+1,right] = True
                        if np.array_equal(rectangular,contour['mask']):
                            continue
                        altered = inp.copy()
                        altered[contour['mask']] = info['background']
                        if np.any(rectangular & (altered != info['background'])):
                            continue
                        altered[rectangular] = info['wall']
                        try:
                            redrawn = _decode(altered)
                        except AssertionError:
                            continue
                        new_route,_ = _trace(redrawn)
                        if [redrawn['relays'][i]['bbox'] for i in new_route] != [info['relays'][i]['bbox'] for i in route]:
                            continue
                        assert np.array_equal(solve(altered),out), 'topology-preserving wall redraw'
                        wall_redraws += 1
                        break
        # A visible port change, with all room walls unchanged, changes the
        # itinerary whenever the alternative route is valid and different.
        if port_changes < 8:
            for pair in puzzle['train']:
                info = _decode(pair['input'])
                route,_ = _trace(info)
                found = False
                for i in route:
                    obj = info['relays'][i]
                    top,left,bottom,right = obj['bbox']
                    center = ((top+bottom)//2,(left+right)//2)
                    for side in range(4):
                        dr,dc = _DIRS[side]
                        p = (top if dr < 0 else bottom if dr > 0 else center[0],
                             left if dc < 0 else right if dc > 0 else center[1])
                        if p == obj['port'] or p not in obj['cells']:
                            continue
                        altered = pair['input'].copy()
                        altered[obj['port']] = info['carrier']
                        altered[p] = info['port_color']
                        try:
                            alternative = solve(altered)
                        except AssertionError:
                            continue
                        if not np.array_equal(alternative,pair['output']):
                            port_changes += 1
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
    assert len(shapes) >= 15 and len(structures) >= 150
    assert len(outputs) >= 35 and len(scopes) >= 35 and len(counts) >= 12
    assert len(palettes) >= 30 and initials == set(range(4))
    assert {2,3,4,5,6} <= set(hist['stops'])
    assert {1,2} <= set(hist['depth']) and len(hist['contours']) >= 3
    assert reentries >= 42 and grouped >= 42 and ablations >= 42 and nonrect >= 100
    assert port_changes >= 4
    assert wall_redraws >= 20
    assert split_hist['test']['depth'][2] >= 10
    # Exercise the constructive path directly, including every intended length
    # and both nesting policies, rather than waiting for stochastic exhaustion.
    fallback_cases = 0
    for seed in range(8):
        for kind,n,nested in [('return',4,False),('stay',3,False)]+[('any',n,b) for n in range(2,7) for b in (False,True)]:
            pair = _fallback_pair(np.random.default_rng(seed),n,kind,(0,1,2,3,4),nested)
            info = _decode(pair['input'])
            route,transitions = _trace(info)
            f = _facts(info,route,transitions)
            _check_output(pair['output'],info,f)
            fallback_cases += 1
    assert _puzzle_digest(generate_puzzle(17,7,3)) == _puzzle_digest(generate_puzzle(17,7,3))
    for a,b in ((2,1),(10,1),(10,4)):
        validate_generated_puzzle(generate_puzzle(801,a,b))
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(0)
    out = broken['test'][0]['output']
    out[0,0] = (int(out[0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    return {'seeds':42,'pairs':total,'input_shapes':len(shapes),'normalized_inputs':len(structures),
            'normalized_outputs':len(outputs),'scope_sequences':len(scopes),'count_combinations':len(counts),
            'palettes':len(palettes),'reentry_pairs':reentries,'grouped_pairs':grouped,
            'decoy_ablation_pairs':ablations,'port_counterfactuals':port_changes,
            'wall_redraws':wall_redraws,'nonrectangular_contours':nonrect,
            'constructive_fallback_cases':fallback_cases,
            'histograms':{k:{str(a):b for a,b in sorted(v.items())} for k,v in hist.items()},
            'split_histograms':{s:{k:{str(a):b for a,b in sorted(v.items())} for k,v in rows.items()}
                                for s,rows in split_hist.items()}}


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train','test'):
        for pair in puzzle[split]:
            for key in ('input','output'):
                a = np.asarray(pair[key],dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
