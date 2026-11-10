"""Competing rectangle collectors; all solving conventions are in the input.

Only NumPy and the Python standard library are used. Ground truth is rendered
from construction descriptors; solve reconstructs descriptors from visible cells.
"""
from __future__ import annotations

import copy
import json
import numpy as np


BG, SOURCE = 0, 5
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((a >= 0) & (a <= 9)), "invalid color")
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    todo.append(p)
        result.append(component)
    return result


def _bounds(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _inside(cell, bounds):
    r, c = cell
    return bounds[0] <= r <= bounds[2] and bounds[1] <= c <= bounds[3]


def _contains(cells, bounds):
    return all(_inside(p, bounds) for p in cells)


def _corner_cells(b, diagonal):
    r, c, rr, cc = b
    if diagonal == 0:
        return {(r, c), (r+1, c), (r, c+1), (rr, cc), (rr-1, cc), (rr, cc-1)}
    return {(r, cc), (r+1, cc), (r, cc-1), (rr, c), (rr-1, c), (rr, c+1)}


def _straight(cells):
    r, c, rr, cc = _bounds(cells)
    return len(cells) in (2, 3) and ((r == rr and cc-c+1 == len(cells)) or
                                           (c == cc and rr-r+1 == len(cells)))


def _connected(mask):
    return len(_components(mask)) == 1


def _token_cells(receiver, index):
    br, bc = receiver['origin']
    dr, dc = receiver['direction']
    tr, tc = receiver['transverse']
    offset = receiver['gap'] + 1 + index * receiver['pitch']
    return {(br+(offset+u)*dr+v*tr, bc+(offset+u)*dc+v*tc)
            for u, v in zip(*np.nonzero(receiver['mask']))}


def _ownership(sources, rows, mode='whole'):
    owners, containers = [], []
    for cells in sources:
        b = _bounds(cells)
        center = ((b[0]+b[2])//2, (b[1]+b[3])//2)
        eligible = []
        for i, row in enumerate(rows):
            if mode == 'whole':
                accepted = _contains(cells, row['bounds'])
            elif mode == 'intersect':
                accepted = any(_inside(p, row['bounds']) for p in cells)
            elif mode == 'center':
                accepted = _inside(center, row['bounds'])
            else:
                accepted = True
            if accepted:
                eligible.append(i)
        owners.append(eligible[0] if eligible else None)
        containers.append(eligible)
    return owners, containers


def _read_receiver(scene, bar, fill, fill_components):
    r, c, rr, cc = _bounds(bar)
    if r == rr:
        origin, transverse, directions = (r, c), (0, 1), ((-1, 0), (1, 0))
    else:
        origin, transverse, directions = (r, c), (1, 0), ((0, -1), (0, 1))
    width = len(bar)
    candidates = []
    for direction in directions:
        dr, dc = direction
        tr, tc = transverse
        projected = []
        for component in fill_components:
            uv = {((y-r)*dr+(x-c)*dc, (y-r)*tr+(x-c)*tc) for y, x in component}
            u0, v0, u1, v1 = _bounds(uv)
            if u0 < 1 or v0 != 0 or v1 != width-1 or not 2 <= u1-u0+1 <= 3:
                continue
            local = {(u-u0, v) for u, v in uv}
            projected.append((u0, u1-u0+1, local, component))
        projected.sort(key=lambda x: x[0])
        if len(projected) < 2 or not 2 <= projected[0][0] <= 4:
            continue
        start, height, local, _ = projected[0]
        pitch = projected[1][0] - start
        if not height+1 <= pitch <= height+3:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for u, v in local:
            mask[u, v] = True
        if not _connected(mask) or not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        copies = []
        for i, (u0, h, footprint, cells) in enumerate(projected):
            if u0 != start+i*pitch or h != height or footprint != local:
                break
            copies.append(cells)
        if len(copies) < 2:
            continue
        occupied = set().union(*copies)
        # The gaps and silhouette holes within the observed receiver are empty.
        readable = True
        for u in range(1, start+(len(copies)-1)*pitch+height):
            for v in range(width):
                p = (r+u*dr+v*tr, c+u*dc+v*tc)
                if not (0 <= p[0] < scene.shape[0] and 0 <= p[1] < scene.shape[1]):
                    readable = False
                elif scene[p] != (fill if p in occupied else BG):
                    readable = False
        if readable:
            candidates.append(dict(origin=origin, transverse=transverse, direction=direction,
                                   gap=start-1, pitch=pitch, mask=mask, initial=len(copies),
                                   cells=occupied, bar=bar))
    _require(len(candidates) == 1, "receiver direction or repetition is ambiguous")
    return candidates[0]


def _parse_candidate(a, table):
    r, c, rr, cc = table
    cues = a[r:rr+1, c].tolist()
    fills = a[r:rr+1, cc].tolist()
    scene = a.copy()
    scene[r:rr+1, c:cc+1] = BG
    components_by_fill = {f: _components(scene == f) for f in set(fills)}
    rows, assigned = [], set()
    for cue, fill in zip(cues, fills):
        components = _components(scene == cue)
        bars = [s for s in components if _straight(s)]
        _require(len(components) == 3 and len(bars) == 1, "cue must have two Ls and one bar")
        bar = bars[0]
        fragments = [s for s in components if s != bar]
        _require(all(len(s) == 3 for s in fragments), "invalid corner fragments")
        corners = set().union(*fragments)
        b = _bounds(corners)
        h, w = b[2]-b[0]+1, b[3]-b[1]+1
        _require(min(h, w) >= 3, "rectangle is too thin")
        diagonal = next((d for d in (0, 1) if corners == _corner_cells(b, d)), None)
        _require(diagonal is not None, "corners do not define one rectangle")
        receiver = _read_receiver(scene, bar, fill, components_by_fill[fill])
        _require(not receiver['cells'] & assigned, "receivers share tokens")
        assigned.update(receiver['cells'])
        rows.append(dict(cue=int(cue), fill=int(fill), bounds=b, corners=corners,
                         diagonal=diagonal, receiver=receiver))
    all_receipt_cells = {tuple(p) for p in np.argwhere(np.isin(scene, fills))}
    _require(assigned == all_receipt_cells, "unassigned receiver pixels")
    sources = _components(scene == SOURCE)
    _require(bool(sources), "no source objects")
    _require(all(max(b[2]-b[0]+1, b[3]-b[1]+1) <= 3 for b in map(_bounds, sources)),
             "source objects exceed the footprint limit")
    owners, containers = _ownership(sources, rows)
    counts = [owners.count(i) for i in range(len(rows))]
    occupied = set()
    for i, row in enumerate(rows):
        dst = row['receiver']
        for k in range(dst['initial'], dst['initial']+counts[i]):
            cells = _token_cells(dst, k)
            _require(all(0 <= y < a.shape[0] and 0 <= x < a.shape[1] for y, x in cells),
                     "receiver extension would clip")
            _require(all(a[p] == BG for p in cells) and not occupied & cells,
                     "receiver extension would collide")
            occupied.update(cells)
    return dict(table=table, rows=rows, sources=sources, owners=owners,
                containers=containers, counts=counts)


def _parse(input_grid):
    a = _grid(input_grid)
    candidates = []
    height, width = a.shape
    # Local table search; there is no fixed side, row, or full-height gutter.
    for c in range(width-1):
        for r in range(height-1):
            for length in range(2, min(4, height-r)+1):
                cues = a[r:r+length, c].tolist()
                fills = a[r:r+length, c+1].tolist()
                if len(set(cues)) != length or BG in cues or SOURCE in cues:
                    continue
                if BG in fills or SOURCE in fills or set(cues) & set(fills):
                    continue
                table = (r, c, r+length-1, c+1)
                moat = a[max(0, r-1):min(height, r+length+1), max(0, c-1):min(width, c+3)]
                if np.count_nonzero(moat) != 2*length:
                    continue
                try:
                    candidates.append(_parse_candidate(a, table))
                except ValueError:
                    pass
    _require(len(candidates) == 1, "input must have one readable palette and scene")
    return a, candidates[0]


def _render_answer(a, rows, sources, owners, counts):
    output = a.copy()
    for cells, owner in zip(sources, owners):
        if owner is not None:
            for p in cells:
                output[p] = rows[owner]['fill']
    for i, row in enumerate(rows):
        dst = row['receiver']
        for p in dst['bar']:
            output[p] = BG
        for k in range(dst['initial'], dst['initial']+counts[i]):
            for p in _token_cells(dst, k):
                output[p] = row['fill']
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the input's palette, components, corners, and repetitions."""
    a, parsed = _parse(input_grid)
    return _render_answer(a, parsed['rows'], parsed['sources'], parsed['owners'], parsed['counts'])


def _mask(rng, source=False, small=False):
    if source:
        h, w = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    else:
        h, w = (2, 2) if small else (int(rng.integers(2, 4)), int(rng.integers(2, 4)))
    mask = np.zeros((h, w), bool)
    # A connected full-span silhouette with randomly extended adjoining rows.
    mask[:, 0] = True
    mask[-1, :] = True
    for y in range(h-1):
        mask[y, :int(rng.integers(1, w+1))] = True
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    if not source and h == w == 3 and rng.random() < .25:
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    return mask.copy()


def _moat(cells, shape):
    return {(r+dr, c+dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if 0 <= r+dr < shape[0] and 0 <= c+dc < shape[1]}


def _place_receivers(rng, shape, targets, initials, small):
    blocked, receivers = set(), []
    for target, initial in zip(targets, initials):
        placed = False
        for _ in range(100):
            mask = _mask(rng, small=small)
            h, w = mask.shape
            pitch, gap = h+int(rng.integers(1, 3)), int(rng.integers(1, 4))
            length = gap+1+(initial+target-1)*pitch+h
            dr, dc = DIRECTIONS[int(rng.integers(4))]
            tr, tc = (-dc, dr)
            offsets = {(u*dr+v*tr, u*dc+v*tc) for u in range(length) for v in range(w)}
            r0, c0, r1, c1 = _bounds(offsets)
            hh, ww = r1-r0+1, c1-c0+1
            if hh > shape[0] or ww > shape[1]:
                continue
            top, left = int(rng.integers(shape[0]-hh+1)), int(rng.integers(shape[1]-ww+1))
            origin = (top-r0, left-c0)
            corridor = {(origin[0]+r, origin[1]+c) for r, c in offsets}
            if corridor & blocked:
                continue
            bar = {(origin[0]+v*tr, origin[1]+v*tc) for v in range(w)}
            receivers.append(dict(origin=origin, direction=(dr, dc), transverse=(tr, tc),
                                  gap=gap, pitch=pitch, mask=mask, initial=initial, bar=bar))
            blocked.update(_moat(corridor, shape))
            placed = True
            break
        if not placed:
            return None
    return receivers, blocked


def _overlap(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _box_contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _sample_rectangles(rng, shape, count, blocked, style):
    objects, used = [], set(blocked)
    for i in range(count):
        placed = False
        for _ in range(100):
            hh = int(rng.integers(5, min(shape[0], 20)+1))
            ww = int(rng.integers(5, min(shape[1], 20)+1))
            r, c = int(rng.integers(shape[0]-hh+1)), int(rng.integers(shape[1]-ww+1))
            b = (r, c, r+hh-1, c+ww-1)
            if i and style == 'nested':
                parent = objects[-1]['bounds']
                ph, pw = parent[2]-parent[0]+1, parent[3]-parent[1]+1
                if min(ph, pw) < 7:
                    continue
                hh, ww = int(rng.integers(4, ph-1)), int(rng.integers(4, pw-1))
                r = int(rng.integers(parent[0]+1, parent[2]-hh+1))
                c = int(rng.integers(parent[1]+1, parent[3]-ww+1))
                b = (r, c, r+hh-1, c+ww-1)
            if any(b == o['bounds'] for o in objects):
                continue
            if i and style == 'disjoint' and any(_overlap(b, o['bounds']) for o in objects):
                continue
            if i and style in ('partial', 'chain'):
                last = objects[-1]['bounds']
                if not _overlap(b, last) or _box_contains(b, last) or _box_contains(last, b):
                    continue
                if style == 'chain' and any(_overlap(b, o['bounds']) for o in objects[:-1]):
                    continue
            diagonal = int(rng.integers(2))
            cells = _corner_cells(b, diagonal)
            if cells & used:
                continue
            objects.append(dict(bounds=b, corners=cells, diagonal=diagonal))
            used.update(_moat(cells, shape))
            placed = True
            break
        if not placed:
            return None
    if style == 'nested' and rng.random() < .75:
        objects.reverse()
    else:
        rng.shuffle(objects)
    return objects, used


def _place_palette(rng, shape, length, blocked):
    candidates = []
    for r in range(shape[0]-length+1):
        for c in range(shape[1]-1):
            cells = {(r+i, c+j) for i in range(length) for j in (0, 1)}
            if not _moat(cells, shape) & blocked:
                candidates.append((r, c, cells))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _place_source(rng, shape, mask, blocked, rectangles, predicate):
    h, w = mask.shape
    offsets = list(zip(*np.nonzero(mask)))
    # Random permutation rather than picking a geometrically preferred edge.
    for flat in rng.permutation((shape[0]-h+1)*(shape[1]-w+1)):
        r, c = divmod(int(flat), shape[1]-w+1)
        cells = {(r+int(u), c+int(v)) for u, v in offsets}
        if cells & blocked:
            continue
        eligible = [i for i, b in enumerate(rectangles) if _contains(cells, b)]
        intersected = [i for i, b in enumerate(rectangles) if any(_inside(p, b) for p in cells)]
        if predicate(cells, eligible, intersected):
            return cells
    return None


def _make_pair(rng, collectors, targets, initials, teaching=False, fallback=False,
               boundary=False, triple=False):
    for attempt in range(220):
        shape = (30, 30) if attempt >= 60 else (int(rng.integers(16, 31)), int(rng.integers(16, 31)))
        placed = _place_receivers(rng, shape, targets, initials, small=attempt >= 80)
        if placed is None:
            continue
        receivers, blocked = placed
        style = 'partial' if teaching else str(rng.choice(['free', 'partial', 'nested', 'chain', 'disjoint']))
        geometries = _sample_rectangles(rng, shape, collectors, blocked, style)
        if geometries is None:
            continue
        rows, blocked = geometries
        palette = _place_palette(rng, shape, collectors, blocked)
        if palette is None:
            continue
        pr, pc, palette_cells = palette
        blocked.update(_moat(palette_cells, shape))
        cues = rng.choice([1, 2, 3, 4, 6, 7, 8, 9], collectors, replace=False).tolist()
        pool = [c for c in range(1, 10) if c != SOURCE and c not in cues]
        fills = rng.choice(pool, collectors, replace=not teaching).tolist()
        for row, cue, fill, receiver in zip(rows, cues, fills, receivers):
            row.update(cue=cue, fill=fill, receiver=receiver)
        sources, owners, left = [], [], list(targets)
        rectangles = [o['bounds'] for o in rows]

        def add(mask, predicate, expected_owner=None):
            cells = _place_source(rng, shape, mask, blocked, rectangles, predicate)
            if cells is None:
                return False
            eligible = [i for i, b in enumerate(rectangles) if _contains(cells, b)]
            owner = eligible[0] if eligible else None
            if expected_owner is not None and owner != expected_owner:
                return False
            if owner is not None:
                if left[owner] <= 0:
                    return False
                left[owner] -= 1
            sources.append(cells)
            owners.append(owner)
            blocked.update(_moat(cells, shape))
            return True

        if teaching or triple:
            required = 3 if triple else 2
            if not add(np.ones((1, 1), bool), lambda s, e, x: len(e) >= required and e[0] == 0, 0):
                continue
        if fallback:
            if not add(np.ones((2, 2), bool), lambda s, e, x: bool(e) and e[0] == 1 and 0 in x, 1):
                continue
        failed = False
        for owner in rng.permutation(collectors):
            owner = int(owner)
            while left[owner]:
                # A large and a single-cell source are both accepted units.
                mask = np.ones((2, 2), bool) if teaching and all(len(s) == 1 for s in sources) else _mask(rng, source=True)
                if not add(mask, lambda s, e, x, j=owner: bool(e) and e[0] == j, owner):
                    failed = True
                    break
            if failed:
                break
        if failed:
            continue
        if teaching or boundary:
            # The floor of the bounding-box center is inside a rectangle, yet
            # the complete object is not. This defeats center-only membership.
            def straddles(s, e, x):
                b = _bounds(s)
                center = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                return not e and bool(x) and any(_inside(center, b) for b in rectangles)
            for _ in range(2 if boundary else 1):
                if not add(np.ones((2, 2), bool), straddles):
                    failed = True
                    break
            if failed:
                continue
        if not add(_mask(rng, source=True), lambda s, e, x: not x):
            continue
        # Some later scenes contain additional outside distractor objects.
        if not teaching and rng.random() < .4:
            add(_mask(rng, source=True), lambda s, e, x: not e)
        a = np.zeros(shape, dtype=int)
        for i, row in enumerate(rows):
            a[pr+i, pc:pc+2] = row['cue'], row['fill']
            for p in row['corners'] | row['receiver']['bar']:
                a[p] = row['cue']
            for k in range(row['receiver']['initial']):
                for p in _token_cells(row['receiver'], k):
                    a[p] = row['fill']
        for cells in sources:
            for p in cells:
                a[p] = SOURCE
        # Counts here are the explicit sampled operand targets, not solve's counts.
        b = _render_answer(a, rows, sources, owners, targets)
        try:
            _, parsed = _parse(a)
        except ValueError:
            continue
        if parsed['counts'] != targets or not np.array_equal(solve(a), b):
            continue
        if teaching and not _teaching_features(parsed)['different_areas']:
            continue
        return {'input': a, 'output': b}
    raise RuntimeError("bounded collector placement exhausted")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError("illegal pair count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(['topology', 'boundary', 'load']))
    # Sample the arithmetic contrasts jointly; no universal operand template.
    while True:
        targets = [rng.integers(1, 4, size=2).tolist() for _ in range(2)]
        initials = [rng.integers(2, 4, size=2).tolist() for _ in range(2)]
        operands = [(s, d) for ss, dd in zip(targets, initials) for s, d in zip(ss, dd)]
        if (sum(targets[0]) != sum(targets[1]) and
                len({d for s, d in operands}) > 1 and
                len({s for s, d in operands}) > 1 and
                len({(s, d) for s, d in operands}) > 2 and
                not all(s == d for s, d in operands)):
            break
    teaching = [_make_pair(rng, 2, targets[i], initials[i], teaching=True, fallback=(i == 1))
                for i in range(2)]
    rng.shuffle(teaching)
    puzzle, seen = {'train': teaching, 'test': []}, {_key(p['input']) for p in teaching}
    for split, count in (('train', num_train-2), ('test', num_test)):
        for index in range(count):
            for _ in range(30):
                is_test = split == 'test'
                collectors = (3 if is_test else 2) if regime == 'topology' else int(rng.integers(2, 4))
                totals = rng.integers(0, 4, size=collectors).tolist()
                if not any(totals):
                    totals[int(rng.integers(collectors))] = 1
                initial = rng.integers(2, 4, size=collectors).tolist()
                if is_test and regime == 'load':
                    totals = rng.integers(3, 5, size=collectors).tolist()
                    totals[int(rng.integers(collectors))] = 4
                    initial = [2]*collectors
                triple = is_test and regime == 'topology' and rng.random() < .4
                if triple:
                    totals[0] = max(1, totals[0])
                pair = _make_pair(rng, collectors, totals, initial,
                                  boundary=is_test and regime == 'boundary', triple=triple)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _teaching_features(p):
    rows, sources, owners = p['rows'], p['sources'], p['owners']
    intersection_owners, _ = _ownership(sources, rows, 'intersect')
    center_owners, _ = _ownership(sources, rows, 'center')
    return dict(contested=any(len(e) > 1 for e in p['containers']),
                ignored=any(owner is None for owner in owners),
                outside=any(not any(_inside(x, row['bounds']) for x in s for row in rows) for s in sources),
                boundary=intersection_owners != owners,
                center=center_owners != owners,
                fallback=any(o is not None and io is not None and o != io for o, io in zip(owners, intersection_owners)),
                different_areas=len({len(s) for s, o in zip(sources, owners) if o is not None}) >= 2,
                mask_difference=any(len(s) != int(rows[o]['receiver']['mask'].sum())
                                    for s, o in zip(sources, owners) if o is not None))


def _shift_features(p):
    straddlers = sum(o is None and any(_inside(x, row['bounds']) for x in s for row in p['rows'])
                     for s, o in zip(p['sources'], p['owners']))
    initial = sum(row['receiver']['initial'] for row in p['rows'])
    return len(p['rows']), straddlers, sum(p['counts']), initial


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_parses, test_parses = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "shape changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            _, p = _parse(a)
            expected = _render_answer(a, p['rows'], p['sources'], p['owners'], p['counts'])
            _require(np.array_equal(b, expected), "incorrect ownership, count, recolor, bar removal, or preservation")
            _require(not np.array_equal(a, b), "vacuous example")
            # Explicit invariants independent of the aggregate array comparison.
            for cells, owner in zip(p['sources'], p['owners']):
                color = SOURCE if owner is None else p['rows'][owner]['fill']
                _require(all(b[x] == color for x in cells), "incorrect processed source")
            for i, row in enumerate(p['rows']):
                dst = row['receiver']
                _require(all(b[x] == row['cue'] for x in row['corners']), "corner changed")
                _require(all(b[x] == BG for x in dst['bar']), "control retained")
                for k in range(dst['initial']+p['counts'][i]):
                    _require(all(b[x] == row['fill'] for x in _token_cells(dst, k)), "wrong chain geometry")
            if split == 'train':
                train_parses.append(p)
            else:
                test_parses.append(p)
    features = [_teaching_features(p) for p in train_parses]
    for name in ('contested', 'ignored', 'outside', 'boundary', 'center', 'fallback',
                 'different_areas', 'mask_difference'):
        _require(any(f[name] for f in features), "missing teaching witness: "+name)
    operands = [(p['counts'][i], row['receiver']['initial'])
                for p in train_parses for i, row in enumerate(p['rows'])]
    _require(len({s for s, d in operands}) >= 2 and len({d for s, d in operands}) >= 2,
             "source and destination operands do not vary independently")
    _require(any(s1*d2 != s2*d1 for s1, d1 in operands for s2, d2 in operands),
             "training admits a destination multiplier")
    _require(len({sum(p['counts']) for p in train_parses}) >= 2, "training admits a fixed total transfer")
    rivals = (lambda s, d: 2*d, lambda s, d: s*d, lambda s, d: max(s, d),
              lambda s, d: s, lambda s, d: d+1)
    for rival in rivals:
        _require(any(rival(s, d) != s+d for s, d in operands), "arithmetic shortcut fits all training")
    _require(any(any(len(e) > 1 and p['rows'][e[0]]['fill'] != p['rows'][e[-1]]['fill']
                         for e in p['containers']) for p in train_parses), "priority has no color witness")
    train_features, test_features = [list(map(_shift_features, ps)) for ps in (train_parses, test_parses)]
    topology_shift = max(f[0] for f in train_features) < min(f[0] for f in test_features)
    boundary_shift = max(f[1] for f in train_features) < min(f[1] for f in test_features)
    load_shift = all(t[2]*s[3] > s[2]*t[3] for t in test_features for s in train_features)
    _require(topology_shift or boundary_shift or load_shift, "no observable train/test shift")


def _four_collector_regression():
    a = np.zeros((30, 30), dtype=int)
    boxes = [(1, 1, 7, 7), (1, 20, 7, 26), (20, 1, 26, 7), (20, 20, 26, 26)]
    points = [(4, 4), (4, 23), (23, 4), (23, 23)]
    rows, sources = [], []
    for i, (b, point) in enumerate(zip(boxes, points)):
        cue, fill = i+1, i+6
        dst = dict(origin=(12, 9+3*i), direction=(-1, 0), transverse=(0, 1),
                   gap=1, pitch=3, mask=np.ones((2, 2), bool), initial=2,
                   bar={(12, 9+3*i), (12, 10+3*i)})
        row = dict(cue=cue, fill=fill, bounds=b, corners=_corner_cells(b, i%2), receiver=dst)
        rows.append(row)
        a[15+i, :2] = cue, fill
        for p in row['corners'] | dst['bar']:
            a[p] = cue
        for k in range(2):
            for p in _token_cells(dst, k):
                a[p] = fill
        a[point] = SOURCE
        sources.append({point})
    expected = _render_answer(a, rows, sources, list(range(4)), [1]*4)
    _require(np.array_equal(solve(a), expected), "four-collector input-only regression failed")


def self_test():
    """Bounded 42-seed structural/semantic sweep, count and mutation checks."""
    shapes, masks, directions, geometries, tables, fingerprints, counts = set(), set(), set(), set(), set(), set(), set()
    topologies, shared, zero, within = set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(_grid(p['input'])) for split in ('train', 'test') for p in puzzle[split]))
        local_geometries = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, p = _parse(pair['input'])
                shapes.add(a.shape)
                tables.add(p['table'][:2])
                counts.add(tuple(p['counts']))
                shared += len({row['fill'] for row in p['rows']}) < len(p['rows'])
                zero += 0 in p['counts']
                boxes = tuple(row['bounds'] for row in p['rows'])
                geometries.add(boxes)
                local_geometries.add(boxes)
                edges = sum(_overlap(x, y) for i, x in enumerate(boxes) for y in boxes[i+1:])
                topologies.add((len(boxes), edges))
                for row in p['rows']:
                    dst = row['receiver']
                    masks.add((dst['mask'].shape, dst['mask'].tobytes()))
                    directions.add(dst['direction'])
        within += len(local_geometries) > 1
    _require(len(fingerprints) == 42 and within == 42, "insufficient puzzle variation")
    _require(len(shapes) >= 35 and len(geometries) >= 150 and len(tables) >= 50, "insufficient spatial variation")
    _require(len(masks) >= 12 and directions == set(DIRECTIONS), "insufficient receiver variation")
    _require(len(counts) >= 30 and shared >= 4 and zero >= 4 and len(topologies) >= 5,
             "insufficient count, overlap, or palette variation")
    a, b = generate_puzzle(97, 2, 1), generate_puzzle(97, 2, 1)
    _require(all(np.array_equal(x[r], y[r]) for s in a for x, y in zip(a[s], b[s])
                 for r in ('input', 'output')), "seed is not deterministic")
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(1000+train*10+test, train, test)
            _require(len(p['train']) == train and len(p['test']) == test, "wrong explicit counts")
    for counts_bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                       (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Deliberate defects must be rejected; each targets a different part of the rule.
    base = generate_puzzle(31, 2, 1)
    for defect in ('sum', 'highlight', 'bar', 'corner', 'ignored', 'priority'):
        broken = copy.deepcopy(base)
        pair = broken['train'][0]
        _, p = _parse(pair['input'])
        if defect == 'sum':
            i = next(i for i, n in enumerate(p['counts']) if n)
            dst = p['rows'][i]['receiver']
            for point in _token_cells(dst, dst['initial']):
                pair['output'][point] = BG
        elif defect in ('highlight', 'ignored'):
            j = next(j for j, owner in enumerate(p['owners']) if (owner is None) == (defect == 'ignored'))
            point = min(p['sources'][j])
            pair['output'][point] = SOURCE if defect == 'highlight' else p['rows'][0]['fill']
        elif defect == 'priority':
            j = next(j for j, eligible in enumerate(p['containers']) if len(eligible) > 1)
            point = min(p['sources'][j])
            pair['output'][point] = p['rows'][p['containers'][j][-1]]['fill']
        else:
            row = p['rows'][0]
            point = min(row['receiver']['bar'] if defect == 'bar' else row['corners'])
            pair['output'][point] = row['cue'] if defect == 'bar' else BG
        try:
            validate_generated_puzzle(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("defect accepted: "+defect)
    _four_collector_regression()
    return dict(seeds=42, shapes=len(shapes), rectangle_layouts=len(geometries),
                receiver_masks=len(masks), palette_positions=len(tables),
                ownership_vectors=len(counts), shared_swatch_pairs=shared, empty_collector_pairs=zero)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
