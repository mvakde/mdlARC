"""Circuit jigsaw: source stencils disambiguate a rigid four-plate assembly.

Only NumPy and the standard library are used. solve reads one input, never
training labels, random state, construction metadata, or parent files.
"""
from __future__ import annotations

import itertools
import math
import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _DIRECTIONS:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(np.asarray(sorted(cells), dtype=int))
    return result


def _box(coords):
    r, c = coords.min(axis=0)
    e, d = coords.max(axis=0) + 1
    return int(r), int(e), int(c), int(d)


def _bits(mask):
    return sum(1 << int(i) for i in np.flatnonzero(mask))


def _bit_array(bits, side):
    return np.array([(bits >> i) & 1 for i in range(side * side)], dtype=bool).reshape(side, side)


def _route_status(route, side):
    """Return (all degrees two, connected), using bounded integer bitsets."""
    if not route:
        return False, False
    full = (1 << (side * side)) - 1
    first = sum(1 << (r * side) for r in range(side))
    last = first << (side - 1)
    neighbors = ((route << side) & full, route >> side,
                 (route << 1) & full & ~first, (route >> 1) & ~last)
    two = three = 0
    for a, b in itertools.combinations(neighbors, 2):
        two |= a & b
    for a, b, c in itertools.combinations(neighbors, 3):
        three |= a & b & c
    degrees = not bool(route & ~two or route & three)
    if not degrees:
        return False, False
    seen = route & -route
    while True:
        more = seen | (((seen << side) | (seen >> side) |
                        ((seen << 1) & ~first) | ((seen >> 1) & ~last)) & route)
        if more == seen:
            return True, seen == route
        seen = more


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "the outer canvas rim must be background")
    # The frame color appears only in four disjoint hollow rectangular rims.
    frame_options = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components(grid == color)
        if len(comps) != 4:
            continue
        boxes = []
        for cells in comps:
            r, e, c, d = _box(cells)
            if e-r < 3 or d-c < 3 or len(cells) != 2*(e-r+d-c)-4:
                break
            if not all(np.all(edge == color) for edge in
                       (grid[r, c:d], grid[e-1, c:d], grid[r:e, c], grid[r:e, d-1])):
                break
            interior = grid[r+1:e-1, c+1:d-1]
            values = set(map(int, np.unique(interior))) - {bg}
            if len(values) != 1 or color in values:
                break
            boxes.append((r, e, c, d))
        if len(boxes) == 4:
            frame_options.append((color, boxes))
    _require(len(frame_options) == 1, "need one identifiable four-card stencil library")
    frame, card_boxes = frame_options[0]
    stencils, source_boxes = {}, {}
    outside = grid != bg
    for r, e, c, d in card_boxes:
        interior = grid[r+1:e-1, c+1:d-1]
        source = next(iter(set(map(int, np.unique(interior))) - {bg}))
        _require(source not in stencils, "each source needs exactly one card")
        stencils[source] = interior == source
        source_boxes[source] = (r, e, c, d)
        outside[r:e, c:d] = False
    pieces, anchors, piece_boxes = [], [], []
    for cells in _components(outside):
        r, e, c, d = _box(cells)
        vals = grid[cells[:, 0], cells[:, 1]]
        colors, counts = np.unique(vals, return_counts=True)
        footprint = np.zeros((e-r, d-c), dtype=bool)
        footprint[cells[:, 0]-r, cells[:, 1]-c] = True
        key = location = None
        if len(colors) == 1:
            source = int(colors[0])
        else:
            _require(len(colors) == 5 and sum(counts == 1) == 4 and sum(counts > 1) == 1,
                     "anchor must contain four singleton key colors and a carrier")
            source = int(colors[counts > 1][0])
            points = cells[np.isin(vals, colors[counts == 1])]
            kr, ke, kc, kd = _box(points)
            _require((ke-kr, kd-kc) == (2, 2) and len(points) == 4,
                     "anchor key must be a complete 2 by 2 patch")
            key = grid[kr:ke, kc:kd].copy()
            _require(len(set(map(int, key.flat))) == 4 and source not in key,
                     "key colors must be distinct from anchor carrier")
            _require(bg not in key and frame not in key, "key conflicts with structural colors")
            location = (kr-r, kc-c)
        _require(source in stencils, "plate carrier has no stencil")
        stencil = stencils[source]
        _require(stencil.shape == footprint.shape, "card extent must match displayed plate extent")
        _require(not np.any(stencil & ~footprint), "stencil marks must lie on the plate")
        piece = dict(source=source, footprint=footprint, stencil=stencil,
                     box=(r, e, c, d), key=key, key_location=location)
        pieces.append(piece)
        piece_boxes.append((r, e, c, d))
        if key is not None:
            anchors.append(piece)
    _require(len(pieces) == 4 and len(anchors) == 1, "need four plates and one anchor")
    _require({p['source'] for p in pieces} == set(stencils), "carrier identities must be distinct")
    all_boxes = card_boxes + piece_boxes
    # Reserve entire rectangles, including blank portions of stencil descriptions.
    for i, (r, e, c, d) in enumerate(all_boxes):
        for a, b, x, y in all_boxes[i+1:]:
            _require(e < a or b < r or d < x or y < c, "objects require a blank-cell gap")
    anchor = anchors[0]
    ordered = [anchor] + [p for p in pieces if p is not anchor]
    total = sum(int(p['footprint'].sum()) for p in pieces)
    side = math.isqrt(total)
    _require(side * side == total and 4 <= side <= 26, "plate areas must define a supported square")
    _require(bool(anchor['footprint'][0, 0]), "anchor needs its top-left bounding corner")
    return dict(background=bg, frame=frame, side=side, pieces=ordered,
                key=anchor['key'], stencil_boxes=source_boxes)


def _candidates(piece, side, corner, anchor=False):
    corners = (0, side-1, side*(side-1), side*side-1)
    result = {}
    for rotation in ([0] if anchor else range(4)):
        f = np.rot90(piece['footprint'], rotation)
        p = np.rot90(piece['stencil'], rotation)
        h, w = f.shape
        if h > side or w > side:
            continue
        r, c = (0 if corner < 2 else side-h), (0 if corner % 2 == 0 else side-w)
        mask = np.zeros((side, side), dtype=bool)
        mask[r:r+h, c:c+w] = f
        support = _bits(mask)
        if not support & (1 << corners[corner]) or sum(bool(support & (1 << i)) for i in corners) != 1:
            continue
        mask[:] = False
        mask[r:r+h, c:c+w] = p
        route = _bits(mask)
        # Footprint symmetries cannot discard different stencil orientations.
        result.setdefault((support, route), (support, route, rotation))
    return list(result.values())


def _assemble(analysis, allow_rotations=True):
    """Exhaust all corner-constrained rigid placements (at most 384)."""
    side, pieces = analysis['side'], analysis['pieces']
    table = {(i, corner): _candidates(pieces[i], side, corner, anchor=i == 0 or not allow_rotations)
             for i in range(4) for corner in range(4)}
    valid, payloads, degree_two = {}, set(), set()
    full = (1 << (side * side)) - 1

    def search(corner, remaining, occupied, feet, paths, order, rotations):
        if corner == 4:
            if occupied != full:
                return
            signature = tuple(paths)
            payloads.add(signature)
            degrees, connected = _route_status(sum(paths), side)
            if degrees:
                degree_two.add(signature)
            if degrees and connected:
                valid.setdefault(signature, dict(feet=tuple(feet), paths=signature,
                                                  order=tuple(order), rotations=tuple(rotations)))
            return
        for i in remaining:
            for foot, path, rotation in table[i, corner]:
                if foot & occupied:
                    continue
                search(corner+1, tuple(j for j in remaining if j != i), occupied | foot,
                       feet+[foot], paths+[path], order+[i], rotations+[rotation])

    for foot, path, rotation in table[0, 0]:
        search(1, (1, 2, 3), foot, [foot], [path], [0], [rotation])
    return dict(solutions=list(valid.values()), payload_candidates=len(payloads),
                degree_two_candidates=len(degree_two))


def _paint(analysis, solution):
    result = np.full((analysis['side'], analysis['side']), analysis['background'], dtype=np.int8)
    for color, path in zip(analysis['key'].flat, solution['paths']):
        result[_bit_array(path, analysis['side'])] = color
    return result


def _analyze(input_grid):
    a = _parse(input_grid)
    a.update(_assemble(a))
    _require(len(a['solutions']) == 1, "input must determine exactly one colored closed circuit")
    a['solution'] = a['solutions'][0]
    a['output'] = _paint(a, a['solution'])
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode local stencils and select the unique full-square single-loop assembly."""
    return _analyze(input_grid)['output'].copy()


def _loop(rng, side, complexity):
    """Grow or retract rectangular detours, preserving an induced simple cycle."""
    low = max(1, side//3)
    r, c = int(rng.integers(low)), int(rng.integers(low))
    e = int(rng.integers(max(r+3, 2*side//3), side))
    d = int(rng.integers(max(c+3, 2*side//3), side))
    route = np.zeros((side, side), dtype=bool)
    route[r, c:d+1] = route[e, c:d+1] = True
    route[r:e+1, c] = route[r:e+1, d] = True
    steps = int(rng.integers(complexity+1))
    for _ in range(steps):
        options = []
        for r in range(side):
            for c in range(side):
                for dr, dc in ((0, 1), (1, 0)):
                    if r+2*dr >= side or c+2*dc >= side:
                        continue
                    points = [(r+i*dr, c+i*dc) for i in range(3)]
                    if not route[points[0]] or not route[points[2]]:
                        continue
                    for sign in (-1, 1):
                        offset = [(a+sign*dc, b+sign*dr) for a, b in points]
                        if any(not (0 <= a < side and 0 <= b < side) for a, b in offset):
                            continue
                        values = [bool(route[p]) for p in offset]
                        if route[points[1]] and not any(values):
                            candidate = route.copy()
                            candidate[points[1]] = False
                            for p in offset:
                                candidate[p] = True
                        elif not route[points[1]] and all(values):
                            candidate = route.copy()
                            candidate[points[1]] = True
                            for p in offset:
                                candidate[p] = False
                        else:
                            continue
                        if _route_status(_bits(candidate), side) == (True, True):
                            options.append(candidate)
        if not options:
            break
        route = options[int(rng.integers(len(options)))]
    return route


def _partition(rng, side, kind):
    corners = ((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))
    if kind == 'equal':
        y = x = side//2
        owner = np.empty((side, side), dtype=np.int8)
        owner[:y, :x], owner[:y, x:] = 0, 1
        owner[y:, :x], owner[y:, x:] = 2, 3
        return owner
    if kind == 'carved':
        large = int(rng.integers(4))
        owner = np.full((side, side), large, dtype=np.int8)
        for i in rng.permutation([j for j in range(4) if j != large]):
            i = int(i)
            height = int(rng.integers(2, side-1))
            width = int(rng.integers(2, side-1))
            for depth in range(height):
                amount = width if depth in (0, height-1) else int(rng.integers(1, width+1))
                if i == 0 and depth < 2:
                    amount = max(2, amount)
                for step in range(amount):
                    r = depth if i < 2 else side-1-depth
                    c = step if i % 2 == 0 else side-1-step
                    if owner[r, c] != large:
                        return None
                    owner[r, c] = i
    else:
        y = int(rng.integers(2, side-1))
        top = int(rng.integers(2, side-1))
        bottom = int(rng.integers(1, side-1))
        owner = np.empty((side, side), dtype=np.int8)
        owner[:y, :top], owner[:y, top:] = 0, 1
        owner[y:, :bottom], owner[y:, bottom:] = 2, 3
    # Boundary exchanges yield tabs, lobes, and narrow necks, without a
    # compulsory straight horizontal seam. Each change preserves connectivity.
    steps = int(rng.integers(2, side*2+1))
    for _ in range(steps):
        r, c = int(rng.integers(side)), int(rng.integers(side))
        if (r, c) in corners:
            continue
        old = int(owner[r, c])
        neighbors = [int(owner[r+dr, c+dc]) for dr, dc in _DIRECTIONS
                     if 0 <= r+dr < side and 0 <= c+dc < side and owner[r+dr, c+dc] != old]
        if not neighbors:
            continue
        new = int(rng.choice(neighbors))
        owner[r, c] = new
        if any(np.count_nonzero(owner == i) < 3 or len(_components(owner == i)) != 1 for i in (old, new)):
            owner[r, c] = old
    return owner


def _tiles(owner, route):
    if owner is None:
        return None
    side = len(owner)
    if tuple(int(owner[p]) for p in ((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))) != (0, 1, 2, 3):
        return None
    pieces = []
    for i in range(4):
        coords = np.argwhere(owner == i)
        if len(coords) < (6 if i == 0 else 3) or len(_components(owner == i)) != 1:
            return None
        r, e, c, d = _box(coords)
        foot = owner[r:e, c:d] == i
        stencil = route[r:e, c:d] & foot
        if not stencil.any():
            return None
        pieces.append(dict(footprint=foot, stencil=stencil))
    key_spots = [(r, c) for r in range(pieces[0]['footprint'].shape[0]-1)
                 for c in range(pieces[0]['footprint'].shape[1]-1)
                 if pieces[0]['footprint'][r:r+2, c:c+2].all()]
    if not key_spots:
        return None
    return pieces, key_spots


def _layout(rng, dimensions):
    """Randomized shelves on either axis; order is independent of all roles."""
    options = []
    for attempt in range(120):
        transpose = bool(attempt % 2)
        dims = [tuple(reversed(d)) if transpose else d for d in dimensions]
        width = int(rng.integers(max(max(w for h, w in dims), 16), 29))
        order = list(map(int, rng.permutation(len(dims))))
        if attempt % 3 == 0:
            order.sort(key=lambda i: -dims[i][0])
        shelves = []
        for i in order:
            h, w = dims[i]
            fits = [j for j, (height, used, group) in enumerate(shelves) if used+1+w <= width and h <= height]
            if fits:
                j = int(rng.choice(fits))
                height, used, group = shelves[j]
                shelves[j] = (height, used+1+w, group+[i])
            else:
                shelves.append((h, w, [i]))
        height = sum(s[0] for s in shelves) + len(shelves)-1
        used_width = max(s[1] for s in shelves)
        if height > 28 or used_width > 28:
            continue
        offsets = [None]*len(dims)
        r = 0
        for h, w, group in shelves:
            c = int(rng.integers(used_width-w+1))
            for i in group:
                rr = r + int(rng.integers(h-dims[i][0]+1))
                offsets[i] = (c, rr) if transpose else (rr, c)
                c += dims[i][1] + 1
            r += h+1
        extent = (used_width, height) if transpose else (height, used_width)
        options.append((offsets, extent))
        if len(options) == 8:
            break
    if not options:
        return None
    offsets, extent = options[int(rng.integers(len(options)))]
    # Random margins retain an entirely blank outer rim. Slack can become
    # unequal shelf gaps and asymmetric canvas padding.
    pads = []
    for d in extent:
        before = int(rng.integers(1, min(4, 29-d)+1))
        after = int(rng.integers(1, min(4, 30-d-before)+1))
        pads.append((before, after))
    shape = tuple(extent[i]+sum(pads[i]) for i in range(2))
    offsets = [(r+pads[0][0], c+pads[1][0]) for r, c in offsets]
    return offsets, shape


def _asymmetric_payload(piece):
    f, p = piece['footprint'], piece['stencil']
    return len({(_signature(np.rot90(f, k)), _signature(np.rot90(p, k))) for k in range(4)}) == 4


def _make_pair(rng, side, kind, overlap, first=False):
    for attempt in range(240):
        selected_kind = kind if attempt < 180 else 'seam'
        if first:
            selected_kind = 'equal'
        owner = _partition(rng, side, selected_kind)
        route = _loop(rng, side, 2*side)
        tiles = _tiles(owner, route)
        if tiles is None:
            continue
        pieces, key_spots = tiles
        if first and not all(_asymmetric_payload(p) for p in pieces):
            continue
        if not first and kind != 'equal' and (len({int(p['footprint'].sum()) for p in pieces}) < 2
                                            or not any(not p['footprint'].all() for p in pieces)):
            continue
        joint = _assemble(dict(side=side, pieces=pieces))
        if len(joint['solutions']) != 1 or (first and joint['payload_candidates'] < 2):
            continue
        colors = list(map(int, rng.permutation(10)))
        bg, frame = colors[:2]
        carriers = colors[2:6]
        if overlap:
            pool = [c for c in colors if c not in (bg, frame, carriers[0])]
            paints = list(map(int, rng.choice(pool, 4, replace=False)))
            if not set(paints) & set(carriers):
                paints[0] = carriers[int(rng.integers(1, 4))]
        else:
            paints = colors[6:10]
        key = np.asarray(paints, dtype=np.int8).reshape(2, 2)
        location = key_spots[int(rng.integers(len(key_spots)))]
        rotations = [0] + ([1, 2, 3] if first else list(map(int, rng.integers(0, 4, 3))))
        plate_arrays, card_arrays = [], []
        for i, p in enumerate(pieces):
            plate = np.where(p['footprint'], carriers[i], bg).astype(np.int8)
            if i == 0:
                r, c = location
                plate[r:r+2, c:c+2] = key
            plate = np.rot90(plate, rotations[i]).copy()
            stencil = np.rot90(p['stencil'], rotations[i])
            card = np.full((stencil.shape[0]+2, stencil.shape[1]+2), frame, dtype=np.int8)
            card[1:-1, 1:-1] = np.where(stencil, carriers[i], bg)
            plate_arrays.append(plate)
            card_arrays.append(card)
        objects = plate_arrays + card_arrays
        layout = _layout(rng, [a.shape for a in objects])
        if layout is None:
            continue
        offsets, shape = layout
        grid = np.full(shape, bg, dtype=np.int8)
        for obj, (r, c) in zip(objects, offsets):
            grid[r:r+obj.shape[0], c:c+obj.shape[1]] = obj
        # Ground truth is built from the accepted square and original owners,
        # not obtained by calling solve. The returned input is then verified.
        output = np.full((side, side), bg, dtype=np.int8)
        output[route] = key.reshape(-1)[owner[route]]
        a = _analyze(grid)
        _require(np.array_equal(a['output'], output), "construction disagrees with visible assembly")
        if first and set(a['solution']['rotations'][1:]) != {1, 2, 3}:
            continue
        return {'input': grid, 'output': output}
    raise RuntimeError("bounded construction exhausted for this sampling plan")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['scale', 'contour', 'palette']))
    base = int(rng.choice([8, 10]))
    if shift == 'scale':
        base = 8
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            first = split == 'train' and index == 0
            if first:
                side, kind = base, 'equal'
            elif split == 'test' and shift == 'scale':
                side, kind = 10, str(rng.choice(['seam', 'carved']))
            else:
                side = int(rng.choice([6, 7, 8] if shift == 'scale' else [6, 7, 8, 9, 10]))
                kind = ('carved' if split == 'test' and shift == 'contour' else
                        str(rng.choice(['seam', 'carved'] if shift != 'contour' else ['seam'])))
            overlap = split == 'test' if shift == 'palette' else (not first and (index == 1 or bool(rng.integers(2))))
            for _ in range(8):
                pair = _make_pair(rng, side, kind, overlap, first)
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema mismatch")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema mismatch")
            grid, out = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(grid)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            a = _analyze(grid)
            _require(np.array_equal(a['output'], out), "incorrect circuit output")
            _require(out.shape == (a['side'], a['side']), "output must retain the full support square")
            active = out != a['background']
            _require(_route_status(_bits(active), a['side']) == (True, True), "output is not one closed loop")
            _require(set(map(int, np.unique(out[active]))) == set(map(int, a['key'].flat)),
                     "each plate must contribute its corner paint color")
            _require(np.any(~active), "support clearing must be visible")
            # Parent B without A would paint whole footprints; that changes
            # every blank support cell. A without B cannot select these
            # transported stencil placements when there is more than one.
            filled = np.full(out.shape, a['background'], dtype=np.int8)
            for color, foot in zip(a['key'].flat, a['solution']['feet']):
                filled[_bit_array(foot, a['side'])] = color
            _require(not np.array_equal(filled, out), "stencils must affect the answer")
            analyses[split].append(a)
    first = analyses['train'][0]
    _require(first['payload_candidates'] > 1, "training must expose stencil-based assembly selection")
    _require(set(first['solution']['rotations'][1:]) == {1, 2, 3}, "first demonstration must teach all quarter turns")
    second = analyses['train'][1]
    _require(len({int(p['footprint'].sum()) for p in second['pieces']}) > 1 and
             any(not p['footprint'].all() for p in second['pieces']), "second demonstration needs unequal irregular plates")


def _metrics(a):
    pieces, solution = a['pieces'], a['solution']
    areas = [int(p['footprint'].sum()) for p in pieces]
    return dict(side=a['side'], background=a['background'], frame=a['frame'],
                areas=tuple(sorted(areas)), area_ratio=max(areas)/min(areas),
                stencil_components=tuple(len(_components(p['stencil'])) for p in pieces),
                rectangles=sum(bool(p['footprint'].all()) for p in pieces),
                rotations=solution['rotations'], payload_candidates=a['payload_candidates'],
                degree_two_candidates=a['degree_two_candidates'],
                palette_overlap=len(set(p['source'] for p in pieces) & set(map(int, a['key'].flat))),
                support=tuple(_bits(p['footprint']) for p in pieces),
                paths=solution['paths'], key_location=pieces[0]['key_location'])


def self_test():
    """Bounded 42-seed structural, semantic, ablation, and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, records = set(), set(), []
    for puzzle in puzzles:
        first = _analyze(puzzle['train'][0]['input'])
        _require(not _assemble(first, allow_rotations=False)['solutions'],
                 "rotation ablation unexpectedly solved the first demonstration")
        signatures.add(tuple(_signature(pair['input']) for s in ('train', 'test') for pair in puzzle[s]))
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                a = _analyze(pair['input'])
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), "input-only solution mismatch")
                shapes.add(pair['input'].shape)
                records.append(_metrics(a))
    _require(len(signatures) == 42 and len(shapes) >= 25, "insufficient puzzle/shape diversity")
    _require({r['side'] for r in records} == {6, 7, 8, 9, 10}, "size vocabulary collapsed")
    _require({r['background'] for r in records} == set(range(10)), "background roles collapsed")
    _require({r['frame'] for r in records} == set(range(10)), "frame roles collapsed")
    _require(any(r['area_ratio'] >= 4 for r in records), "unequal pieces missing")
    _require(sum(max(r['stencil_components']) > 1 for r in records) >= 10, "disconnected stencils missing")
    _require(len({(r['side'], r['support'], r['areas']) for r in records}) >= 100, "support geometry collapsed")
    _require(len({r['paths'] for r in records}) >= 150, "circuit geometry collapsed")
    _require(any(r['palette_overlap'] >= 2 for r in records) and any(r['palette_overlap'] == 0 for r in records),
             "direct-paint palette overlap variation missing")
    again = generate_puzzle(17)
    _require(all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
                 for p, q in zip(puzzles[17][s], again[s]) for k in ('input', 'output')), "nondeterminism")
    # Recoloring every semantic role must commute with the transformation.
    permutation = np.random.default_rng(572).permutation(10)
    for puzzle in puzzles[:3]:
        # Exchange equal-extent first-demonstration cards. This changes the
        # library order and card/plate proximity while preserving the answer.
        pair = puzzle['train'][0]
        a = _parse(pair['input'])
        boxes = list(a['stencil_boxes'].values())
        r, e, c, d = boxes[0]
        rr, ee, cc, dd = boxes[1]
        moved = pair['input'].copy()
        tile = moved[r:e, c:d].copy()
        moved[r:e, c:d] = moved[rr:ee, cc:dd]
        moved[rr:ee, cc:dd] = tile
        _require(np.array_equal(solve(moved), pair['output']), "solver relies on library placement")
        for pair in puzzle['train'] + puzzle['test']:
            _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
                     "solver relies on literal palette identities")
            # Increasing blank margins changes neither size inference nor assembly.
            g = pair['input']
            if max(g.shape) <= 28:
                padded = np.pad(g, 1, constant_values=int(g[0, 0]))
                _require(np.array_equal(solve(padded), pair['output']), "solver relies on canvas size")
    # Test the bridge's clauses separately, without assuming every negative
    # topology is frequent in generated geometric candidates.
    route = np.zeros((9, 9), dtype=bool)
    for r, c in ((0, 0), (5, 5)):
        route[r:r+3, c] = route[r:r+3, c+2] = True
        route[r, c:c+3] = route[r+2, c:c+3] = True
    _require(_route_status(_bits(route), 9) == (True, False), "two loops must fail connectivity")
    route[5:, :] = False
    _require(_route_status(_bits(route), 9) == (True, True), "simple loop rejected")
    opened = route.copy()
    opened[0, 1] = False
    _require(not _route_status(_bits(opened), 9)[0], "open path must fail degree two")
    branch = route.copy()
    branch[0, 3] = True
    _require(not _route_status(_bits(branch), 9)[0], "branch must fail degree two")
    diagonal = np.eye(9, dtype=bool)
    _require(not _route_status(_bits(diagonal), 9)[0], "diagonal contact is not circuit adjacency")
    # Output corruption must be caught by complete semantic reconstruction.
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted corrupted output")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    return dict(seeds=42, pairs=len(records), input_shapes=len(shapes),
                support_geometries=len({(r['side'], r['support'], r['areas']) for r in records}),
                circuit_geometries=len({r['paths'] for r in records}),
                disconnected_stencil_pairs=sum(max(r['stencil_components']) > 1 for r in records),
                geometry_ambiguous_pairs=sum(r['payload_candidates'] > 1 for r in records))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
