"""Reciprocal socket wiring: pool ports, fit scaled silhouettes, close wires.

Only Python's standard library and NumPy are used. solve() is input-only.
"""
from collections import Counter
from math import isqrt
import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _normalise(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _point(p, transform):
    r, c = p
    if transform >= 4:
        c = -c
    for _ in range(transform % 4):
        r, c = -c, r
    return int(r), int(c)


def _orient(shape, transform):
    return _normalise([_point(p, transform) for p in shape])


def _canonical(shape):
    return min(_orient(shape, t) for t in range(8))


def _span(shape):
    return max(r for r, c in shape)+1, max(c for r, c in shape)+1


def _faces(shape):
    cells = set(shape)
    return tuple((r, c, dr, dc) for r, c in shape for dr, dc in _DIRS
                 if (r+dr, c+dc) not in cells)


def _face_map(shape, transform):
    """Canonical faces -> oriented faces, using exactly the mask's transform."""
    raw = [_point(p, transform) for p in shape]
    r0, c0 = min(r for r, c in raw), min(c for r, c in raw)
    result = {}
    for r, c, dr, dc in _faces(shape):
        rr, cc = _point((r, c), transform)
        vr, vc = _point((dr, dc), transform)
        result[(r, c, dr, dc)] = (rr-r0, cc-c0, vr, vc)
    return result


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, comp = [start], []
        while stack:
            r, c = stack.pop()
            comp.append((int(r), int(c)))
            for dr, dc in _DIRS:
                p = (r+dr, c+dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(comp)))
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells)+1,
            min(c for r, c in cells), max(c for r, c in cells)+1)


def _mask(shape):
    a = np.zeros(_span(shape), dtype=bool)
    for p in shape:
        a[p] = True
    return a


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.uint8).tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected an integer ARC grid with sides 1..30 and colors 0..9')
    return a.astype(np.int8, copy=True)


def _decode_blocks(cells, scale):
    r0, r1, c0, c1 = _bounds(cells)
    if (r1-r0) % scale or (c1-c0) % scale or len(cells) % (scale*scale):
        return None
    counts = Counter(((r-r0)//scale, (c-c0)//scale) for r, c in cells)
    if any(n != scale*scale for n in counts.values()):
        return None
    return tuple(sorted(counts))


def _transport_ports(shape, transform, code, top=0, left=0):
    mapping = _face_map(shape, transform)
    return {(r+top, c+left, dr, dc): color
            for face, color in code.items() for r, c, dr, dc in [mapping[face]]}


def _network(owner, holes, codes, ignore_color=False, skip_blockers=False):
    """First-hit reciprocal edges, counted once; paths contain structure only."""
    ports = []
    for hole in holes:
        ports.append(_transport_ports(hole['key'], hole['transform'],
                                      codes[hole['key']], hole['top'], hole['left']))
    wires = {}
    links, open_ports, mismatches, blocked = [], 0, 0, 0
    for i, local in enumerate(ports):
        for (r, c, dr, dc), color in sorted(local.items()):
            rr, cc = r+dr, c+dc
            path = []
            linked = False
            while 0 <= rr < owner.shape[0] and 0 <= cc < owner.shape[1]:
                j = int(owner[rr, cc])
                if j < 0:
                    path.append((rr, cc))
                else:
                    opposite = ports[j].get((rr, cc, -dr, -dc))
                    if j != i and opposite is not None and (ignore_color or opposite == color):
                        if path:
                            linked = True
                            if j > i:
                                links.append((i, j, color, tuple(path)))
                                for p in path:
                                    if p in wires and wires[p] != color:
                                        raise ValueError('differently colored wires cross')
                                    wires[p] = color
                        if not skip_blockers:
                            break
                    else:
                        if opposite is not None and opposite != color:
                            mismatches += 1
                        else:
                            blocked += 1
                        if not skip_blockers:
                            break
                rr, cc = rr+dr, cc+dc
            if not linked:
                open_ports += 1
    return wires, {'links': tuple(links), 'open_ports': open_ports,
                   'mismatches': mismatches, 'blocked': blocked, 'ports': ports}


def _recover_candidate(inp, bg, structural, bounds):
    r0, r1, c0, c1 = bounds
    board = inp[r0:r1, c0:c1].copy()
    if not np.all((board == bg) | (board == structural)):
        raise ValueError('blueprint uses only background and structure')
    cavities = _components(board == bg)
    if not 2 <= len(cavities) <= 6:
        raise ValueError('expected two through six cavities')
    outside = np.ones(inp.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    external = [(color, comp) for color in range(10) if color != bg
                for comp in _components((inp == color) & outside)]
    bodies = [(color, comp) for color, comp in external if len(comp) >= 4]
    signals = [(color, comp[0]) for color, comp in external if len(comp) == 1]
    if len(bodies) != len(cavities) or len(bodies)+len(signals) != len(external):
        raise ValueError('external objects must be macro-pieces or single-dot signals')
    body_colors = {color for color, comp in bodies}
    signal_colors = {color for color, p in signals}
    if (not signals or structural in body_colors | signal_colors
            or not body_colors.isdisjoint(signal_colors)):
        raise ValueError('body, signal, structure and background roles overlap')
    logical_area = sum(len(c) for c in cavities)
    pixel_area = sum(len(c) for color, c in bodies)
    ratio, remainder = divmod(pixel_area, logical_area)
    scale = isqrt(ratio)
    if remainder or scale*scale != ratio or not 2 <= scale <= 4:
        raise ValueError('pieces do not give one uniform integer scale')
    if max(board.shape)*scale > 30:
        raise ValueError('enlarged blueprint exceeds 30')
    holes, counts = [], Counter()
    owner = np.full(board.shape, -1, dtype=int)
    for index, cells in enumerate(cavities):
        shape = _normalise(cells)
        key = _canonical(shape)
        forms = [_orient(key, t) for t in range(8)]
        if len(set(forms)) != 8 or not 4 <= len(shape) <= 9:
            raise ValueError('cavity silhouettes must be fully asymmetric, area 4..9')
        transform = forms.index(shape)
        top, _, left, _ = _bounds(cells)
        holes.append({'key': key, 'transform': transform, 'top': top, 'left': left,
                      'cells': cells})
        counts[key] += 1
        for p in cells:
            owner[p] = index
    if any(n < 2 for n in counts.values()):
        raise ValueError('every shape class needs at least two copies')
    pieces, class_colors, piece_counts, pixel_owner = [], {}, Counter(), {}
    for index, (color, cells) in enumerate(bodies):
        shape = _decode_blocks(cells, scale)
        if shape is None:
            raise ValueError('partially filled macrocell')
        key = _canonical(shape)
        if key not in counts:
            raise ValueError('loose silhouette has no matching cavity')
        if key in class_colors and class_colors[key] != color:
            raise ValueError('interchangeable shapes have different body colors')
        class_colors[key] = color
        piece_counts[key] += 1
        transform = next(t for t in range(8) if _orient(key, t) == shape)
        top, _, left, _ = _bounds(cells)
        inverse = {v: k for k, v in _face_map(key, transform).items()}
        pieces.append({'key': key, 'transform': transform, 'top': top, 'left': left,
                       'cells': cells, 'inverse': inverse, 'observed': set()})
        for p in cells:
            pixel_owner[p] = index
    if piece_counts != counts:
        raise ValueError('piece/cavity class multiplicities disagree')
    # Touching source bodies are forbidden even when their colors differ.
    for p, i in pixel_owner.items():
        for dr, dc in _DIRS:
            j = pixel_owner.get((p[0]+dr, p[1]+dc), i)
            if j != i:
                raise ValueError('loose bodies touch')
    codes = {key: {} for key in counts}
    for color, (r, c) in signals:
        neighbors = [(pixel_owner[(r+dr, c+dc)], r+dr, c+dc, -dr, -dc)
                     for dr, dc in _DIRS if (r+dr, c+dc) in pixel_owner]
        if len(neighbors) != 1:
            raise ValueError('a dot must touch exactly one body pixel')
        index, rr, cc, dr, dc = neighbors[0]
        piece = pieces[index]
        face = ((rr-piece['top'])//scale, (cc-piece['left'])//scale, dr, dc)
        if face not in piece['inverse']:
            raise ValueError('dot is not on an exposed logical face')
        canonical_face = piece['inverse'][face]
        code = codes[piece['key']]
        if canonical_face in code and code[canonical_face] != color:
            raise ValueError('conflicting port observations')
        code[canonical_face] = color
        piece['observed'].add(canonical_face)
    if any(not 2 <= len(code) <= 5 for code in codes.values()):
        raise ValueError('each class needs two through five evidenced ports')
    logical = board.copy()
    for hole in holes:
        for p in hole['cells']:
            logical[p] = class_colors[hole['key']]
    wires, net = _network(owner, holes, codes)
    for p, color in wires.items():
        logical[p] = color
    output = np.repeat(np.repeat(logical, scale, 0), scale, 1).astype(np.int8)
    pooled = any(all(len(p['observed']) < len(codes[key]) for p in pieces if p['key'] == key)
                 for key in counts)
    transformed = any(p['transform'] != h['transform']
                      for p in pieces for h in holes if p['key'] == h['key'])
    reflection_required = any(
        Counter(p['transform']//4 for p in pieces if p['key'] == key)
        != Counter(h['transform']//4 for h in holes if h['key'] == key)
        for key in counts)
    turn_required = any(
        Counter(p['transform'] % 2 for p in pieces if p['key'] == key)
        != Counter(h['transform'] % 2 for h in holes if h['key'] == key)
        for key in counts)
    try:
        transparent, _ = _network(owner, holes, codes, skip_blockers=True)
        occlusion_witness = transparent != wires
    except ValueError:
        occlusion_witness = False
    info = {'background': bg, 'structural': structural, 'scale': scale,
            'board': board, 'bounds': bounds, 'owner': owner, 'holes': holes,
            'pieces': pieces, 'codes': codes, 'class_colors': class_colors,
            'wire_cells': len(wires), 'wire_colors': tuple(sorted(set(wires.values()))),
            'link_count': len(net['links']), 'open_ports': net['open_ports'],
            'mismatches': net['mismatches'], 'blocked': net['blocked'],
            'pooled': pooled, 'transformed': transformed,
            'reflection_required': reflection_required, 'turn_required': turn_required,
            'occlusion_witness': occlusion_witness,
            'shape_classes': tuple(sorted(counts)), 'piece_count': len(pieces),
            'unhinted': sum(not p['observed'] for p in pieces),
            'signal_count': len(signals), 'logical_output': logical}
    return output, info


def _recover(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= inp.size//2:
        raise ValueError('background must occupy more than half the input')
    solutions = []
    for color in range(10):
        if color == bg:
            continue
        for comp in _components(inp == color):
            r0, r1, c0, c1 = _bounds(comp)
            if min(r1-r0, c1-c0) < 5:
                continue
            board = inp[r0:r1, c0:c1]
            if not (np.all(board[0] == color) and np.all(board[-1] == color)
                    and np.all(board[:, 0] == color) and np.all(board[:, -1] == color)):
                continue
            try:
                solutions.append(_recover_candidate(inp, bg, color, (r0, r1, c0, c1)))
            except ValueError:
                continue
    if len(solutions) != 1:
        raise ValueError('expected one readable blueprint and port interpretation')
    return solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover scale, pooling, fit orientations and reciprocal visibility from input."""
    return _recover(input_grid)[0]


def _grow_shape(rng, area):
    for _ in range(160):
        cells = {(0, 0)}
        while len(cells) < area:
            options = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _DIRS}-cells)
            options = [p for p in options if max(_span(_normalise(cells | {p}))) <= 4]
            weights = np.array([1+sum((p[0]+dr, p[1]+dc) in cells for dr, dc in _DIRS)
                                for p in options], dtype=float)
            # Alternate compact and sparse growth; not a finite silhouette bank.
            if rng.random() < .5:
                weights = 1/weights
            weights /= weights.sum()
            cells.add(options[int(rng.choice(len(options), p=weights))])
        key = _canonical(tuple(cells))
        if len({_orient(key, t) for t in range(8)}) == 8:
            return key
    raise ValueError('bounded asymmetric shape growth exhausted')


def _place_holes(rng, shapes, scale):
    limit = 30//scale
    low = 7 if scale == 4 else 8
    if limit < low:
        return None
    for _ in range(16):
        h, w = map(int, rng.integers(low, limit+1, size=2))
        if rng.random() < .2:
            w = h
        owner = np.full((h, w), -1, dtype=int)
        holes = [None]*len(shapes)
        order = sorted(range(len(shapes)), key=lambda i: -len(shapes[i]))
        for i in order:
            shape = shapes[i]
            transform = int(rng.integers(8))
            oriented = _orient(shape, transform)
            sh, sw = _span(oriented)
            origins = [(r, c) for r in range(1, h-sh) for c in range(1, w-sw)]
            rng.shuffle(origins)
            for top, left in origins:
                cells = [(r+top, c+left) for r, c in oriented]
                if all(owner[r, c] < 0 and all(owner[r+dr, c+dc] < 0 for dr, dc in _DIRS)
                       for r, c in cells):
                    for p in cells:
                        owner[p] = i
                    holes[i] = {'key': shape, 'transform': transform, 'top': top,
                                'left': left, 'cells': tuple(cells)}
                    break
            else:
                break
        else:
            return owner, holes
    return None


def _dot_positions(shape, scale, face):
    """All pixels on this face with exactly one neighboring body pixel."""
    r, c, dr, dc = face
    body = {(r*scale+i, c*scale+j) for r, c in shape
            for i in range(scale) for j in range(scale)}
    positions = []
    for offset in range(scale):
        if dr == -1:
            p = (r*scale-1, c*scale+offset)
        elif dr == 1:
            p = ((r+1)*scale, c*scale+offset)
        elif dc == -1:
            p = (r*scale+offset, c*scale-1)
        else:
            p = (r*scale+offset, (c+1)*scale)
        if sum((p[0]+vr, p[1]+vc) in body for vr, vc in _DIRS) == 1:
            positions.append(p)
    return positions


def _visibility_edges(owner, holes, scale):
    all_ports = []
    for h in holes:
        allowed = {p for p in _faces(h['key']) if _dot_positions(h['key'], scale, p)}
        inverse = {v: p for p, v in _face_map(h['key'], h['transform']).items() if p in allowed}
        all_ports.append({(r+h['top'], c+h['left'], dr, dc): p
                          for (r, c, dr, dc), p in inverse.items()})
    edges = []
    for i, ports in enumerate(all_ports):
        for (r, c, dr, dc), p in ports.items():
            rr, cc = r+dr, c+dc
            path = []
            while 0 <= rr < owner.shape[0] and 0 <= cc < owner.shape[1] and owner[rr, cc] < 0:
                path.append((rr, cc))
                rr, cc = rr+dr, cc+dc
            if 0 <= rr < owner.shape[0] and 0 <= cc < owner.shape[1]:
                j = int(owner[rr, cc])
                if j > i and path:
                    q = all_ports[j].get((rr, cc, -dr, -dc))
                    if q is not None:
                        edges.append((i, p, j, q))
    return edges


def _occluded_edges(owner, holes, scale):
    """Potential opposite ports with at least one opaque cell intervening."""
    all_ports = []
    for h in holes:
        inverse = {v: p for p, v in _face_map(h['key'], h['transform']).items()
                   if _dot_positions(h['key'], scale, p)}
        all_ports.append({(r+h['top'], c+h['left'], dr, dc): p
                          for (r, c, dr, dc), p in inverse.items()})
    result = []
    for i, ports in enumerate(all_ports):
        for (r, c, dr, dc), p in ports.items():
            rr, cc, hit = r+dr, c+dc, False
            while 0 <= rr < owner.shape[0] and 0 <= cc < owner.shape[1]:
                j = int(owner[rr, cc])
                if j >= 0:
                    q = all_ports[j].get((rr, cc, -dr, -dc))
                    if hit and j > i and q is not None:
                        result.append((i, p, j, q))
                    hit = True
                rr, cc = rr+dr, cc+dc
    return result


def _make_codes(rng, owner, holes, scale, signals, negative, rich):
    keys = list(dict.fromkeys(h['key'] for h in holes))
    codes = {key: {} for key in keys}
    edges = _visibility_edges(owner, holes, scale)
    if not edges:
        return None
    rng.shuffle(edges)
    i, p, j, q = edges[0]
    codes[holes[i]['key']][p] = signals[0]
    codes[holes[j]['key']][q] = signals[0]
    # A second visible pair with two unequal labels is an explicit negative example.
    if negative:
        for i, p, j, q in edges[1:]:
            k, l = holes[i]['key'], holes[j]['key']
            if k == l and p == q:
                continue
            cp, cq = codes[k].get(p), codes[l].get(q)
            if cp is None or cq is None:
                if cp is None:
                    codes[k][p] = signals[0] if cq != signals[0] else signals[1]
                if cq is None:
                    codes[l][q] = signals[1] if codes[k][p] == signals[0] else signals[0]
                break
        else:
            return None
        # Demonstrate that a matching port beyond the first cavity is blocked.
        occluded = _occluded_edges(owner, holes, scale)
        rng.shuffle(occluded)
        for i, p, j, q in occluded:
            k, l = holes[i]['key'], holes[j]['key']
            cp, cq = codes[k].get(p), codes[l].get(q)
            if cp is not None and cq is not None and cp != cq:
                continue
            color = cp if cp is not None else (cq if cq is not None else signals[1])
            codes[k][p], codes[l][q] = color, color
            if max(map(len, codes.values())) <= 5:
                break
            return None
        else:
            return None
    # Add some more matching pairs; one class's port can serve several sockets.
    for i, p, j, q in edges[1:]:
        k, l = holes[i]['key'], holes[j]['key']
        if rng.random() < (.5 if rich else .2) and len(codes[k]) < 4 and len(codes[l]) < 4:
            cp, cq = codes[k].get(p), codes[l].get(q)
            if cp is None and cq is None:
                color = int(rng.choice(signals))
                codes[k][p], codes[l][q] = color, color
    for key in keys:
        target = int(rng.integers(3 if rich else 2, 6 if rich else 5))
        available = [p for p in _faces(key) if _dot_positions(key, scale, p)]
        rng.shuffle(available)
        for p in available:
            if len(codes[key]) >= target:
                break
            if p not in codes[key]:
                codes[key][p] = int(rng.choice(signals))
        if not 2 <= len(codes[key]) <= 5:
            return None
    try:
        wires, net = _network(owner, holes, codes)
    except ValueError:
        return None
    if not wires or not net['open_ports'] or (negative and not net['mismatches']):
        return None
    if negative:
        try:
            transparent, _ = _network(owner, holes, codes, skip_blockers=True)
        except ValueError:
            return None
        if transparent == wires:
            return None
    return codes


def _annotated_object(rng, shape, transform, code, observations, scale, color, bg):
    oriented = _orient(shape, transform)
    block_mask = np.repeat(np.repeat(_mask(oriented), scale, 0), scale, 1)
    obj = np.full((block_mask.shape[0]+2, block_mask.shape[1]+2), bg, dtype=np.int8)
    obj[1:-1, 1:-1][block_mask] = color
    mapping = _face_map(shape, transform)
    selected = []
    faces = sorted(observations)
    rng.shuffle(faces)
    for p in faces:
        positions = _dot_positions(oriented, scale, mapping[p])
        rng.shuffle(positions)
        for rr, cc in positions:
            spot = rr+1, cc+1
            if all(max(abs(spot[0]-r), abs(spot[1]-c)) > 1 for r, c in selected):
                obj[spot] = code[p]
                selected.append(spot)
                break
        else:
            return None
    return obj


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    summed = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = summed[h:, w:] - summed[:-h, w:] - summed[h:, :-w] + summed[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, objects, bg):
    for attempt in range(36):
        shape = tuple(map(int, rng.integers(22, 31, size=2))) if attempt < 16 else (30, 30)
        canvas = np.full(shape, bg, dtype=np.int8)
        blocked = np.zeros(shape, dtype=bool)
        order = sorted(range(len(objects)), key=lambda i: -objects[i].size)
        for i in order:
            obj = objects[i]
            h, w = obj.shape
            positions = _free_origins(blocked, h, w)
            if not len(positions):
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            canvas[r:r+h, c:c+w] = obj
            blocked[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True
        else:
            if np.count_nonzero(canvas == bg) <= canvas.size//2:
                continue
            # Crop optional empty margins; never change a piece or annotation.
            cells = list(zip(*np.nonzero(canvas != bg)))
            r0, r1, c0, c1 = _bounds(cells)
            r0, c0 = max(0, r0-int(rng.integers(4))), max(0, c0-int(rng.integers(4)))
            r1, c1 = min(shape[0], r1+int(rng.integers(4))), min(shape[1], c1+int(rng.integers(4)))
            crop = canvas[r0:r1, c0:c1]
            if rng.random() < .65 and np.count_nonzero(crop == bg) > crop.size//2:
                canvas = crop.copy()
            return canvas
    return None


def _make_pair(rng, scale, demonstration=None, rich=False, previous_bg=None):
    for attempt in range(400):
        palette = list(map(int, rng.permutation(10)))
        bg, structural = palette[:2]
        if bg == previous_bg:
            continue
        class_count = 2 if demonstration == 'negative' else int(rng.choice([1, 2], p=[.7, .3]))
        if demonstration == 'pool' or scale == 4:
            class_count = 1
        area = int(rng.integers(5, 8)) if class_count == 2 else 4
        keys = []
        for _ in range(100):
            key = _grow_shape(rng, area if class_count == 2 else int(rng.integers(4, 10)))
            if scale == 4:
                key = _canonical(((0, 0), (1, 0), (2, 0), (2, 1)))
            if key not in keys:
                keys.append(key)
            if len(keys) == class_count:
                break
        else:
            continue
        if class_count == 2:
            counts = [2, 2]
            if scale == 2 and demonstration is None and rng.random() < .25:
                counts[int(rng.integers(2))] += 1
        else:
            maximum = 2 if scale == 4 else (4 if scale == 3 else 6)
            counts = [int(rng.integers(2, maximum+1))]
            if demonstration == 'pool':
                counts = [int(rng.integers(2, 4))]
        shapes = [key for key, n in zip(keys, counts) for _ in range(n)]
        placed = _place_holes(rng, shapes, scale)
        if placed is None:
            continue
        owner, holes = placed
        class_colors = dict(zip(keys, palette[2:2+class_count]))
        signal_count = int(rng.integers(2, 5))
        signals = palette[2+class_count:2+class_count+signal_count]
        codes = _make_codes(rng, owner, holes, scale, signals,
                            demonstration == 'negative', rich)
        if codes is None:
            continue
        board = np.full(owner.shape, structural, dtype=np.int8)
        logical = board.copy()
        for hole in holes:
            for p in hole['cells']:
                board[p] = bg
                logical[p] = class_colors[hole['key']]
        observations = [set() for _ in holes]
        allocation = 'split' if demonstration == 'pool' else str(rng.choice(['split', 'full', 'redundant']))
        for key in keys:
            indices = [i for i, h in enumerate(holes) if h['key'] == key]
            rng.shuffle(indices)
            faces = sorted(codes[key])
            rng.shuffle(faces)
            if allocation == 'full':
                observations[indices[0]].update(faces)
            else:
                for j, face in enumerate(faces):
                    observations[indices[j % len(indices)]].add(face)
                if allocation == 'redundant':
                    for face in faces:
                        if rng.random() < .35:
                            observations[int(rng.choice(indices))].add(face)
        objects = [board]
        for i, hole in enumerate(holes):
            transform = int(rng.integers(8))
            # The first demonstration explicitly transports across a reflection.
            if demonstration in ('pool', 'negative'):
                transform = hole['transform']
                if i == 0 and demonstration == 'pool':
                    transform = (transform+4) % 8
                elif i == 0:
                    transform = (transform//4)*4+(transform+1) % 4
            obj = _annotated_object(rng, hole['key'], transform, codes[hole['key']],
                                    observations[i], scale, class_colors[hole['key']], bg)
            if obj is None:
                break
            objects.append(obj)
        else:
            inp = _pack(rng, objects, bg)
            if inp is None:
                continue
            wires, net = _network(owner, holes, codes)
            for p, color in wires.items():
                logical[p] = color
            output = np.repeat(np.repeat(logical, scale, 0), scale, 1).astype(np.int8)
            try:
                expected, info = _recover(inp)
            except ValueError:
                continue
            if not np.array_equal(expected, output):
                raise AssertionError('construction and input reconstruction disagree')
            if demonstration == 'pool' and not info['pooled']:
                continue
            if not info['transformed']:
                continue
            return {'input': inp, 'output': output}
    raise RuntimeError('bounded socket construction exhausted')


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(5))
    if mode == 0:
        train_scales, test_scales = [2]*num_train, [3]*num_test
    elif mode == 1:
        train_scales, test_scales = [3]*num_train, [2]*num_test
    elif mode == 2:
        train_scales = [2, 3]+list(map(int, rng.choice([2, 3], size=num_train-2)))
        test_scales = list(map(int, rng.choice([2, 3], size=num_test)))
    elif mode == 3:
        scale = int(rng.choice([2, 3]))
        train_scales, test_scales = [scale]*num_train, [scale]*num_test
    else:
        train_scales = [2, 3]+list(map(int, rng.choice([2, 3], size=num_train-2)))
        test_scales = [4]*num_test
    puzzle, seen, previous_bg = {'train': [], 'test': []}, set(), None
    for split, scales in (('train', train_scales), ('test', test_scales)):
        for i, scale in enumerate(scales):
            demo = ('pool', 'negative')[i] if split == 'train' and i < 2 else None
            for _ in range(12):
                pair = _make_pair(rng, scale, demo, rich=(split == 'test'), previous_bg=previous_bg)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct input construction exhausted')
            seen.add(_key(pair['input']))
            previous_bg = _recover(pair['input'])[1]['background']
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(expected, out), 'incorrect assembled wiring'
            assert info['wire_cells'] > 0 and info['open_ports'] > 0
            assert info['transformed']
            # These directly catch the two parent-concept ablations.
            plain = info['board'].copy()
            for h in info['holes']:
                for p in h['cells']:
                    plain[p] = info['class_colors'][h['key']]
            plain = np.repeat(np.repeat(plain, info['scale'], 0), info['scale'], 1)
            assert not np.array_equal(plain, out), 'ignoring the port code changed nothing'
            assert out.shape != info['board'].shape, 'ignoring macrocell scale changed nothing'
            assert np.all(out[::info['scale'], ::info['scale']] == info['logical_output'])
            assert np.all(out[:info['scale']] == info['structural'])
            assert np.all(out[-info['scale']:] == info['structural'])
            assert np.all(out[:, :info['scale']] == info['structural'])
            assert np.all(out[:, -info['scale']:] == info['structural'])
            features[split].append(info)
    train = features['train']
    assert any(f['pooled'] for f in train), 'training needs distributed port observations'
    assert any(f['mismatches'] > 0 and len(f['shape_classes']) == 2 for f in train)
    assert any(f['occlusion_witness'] for f in train), 'training must expose first-hit blocking'
    assert any(f['reflection_required'] for f in train)
    assert any(f['turn_required'] for f in train)
    assert len({f['shape_classes'] for f in train}) >= 2
    assert len({f['background'] for f in train}) >= 2
    return None


def self_test():
    """Bounded tests on 42 seeds, metamorphic transforms and negative diagnostics."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, features, normalized = set(), [], set()
    for puzzle in puzzles:
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                output, info = _recover(pair['input'])
                assert np.array_equal(output, pair['output'])
                features.append((split, info, pair['input'].shape, output.shape))
                normalized.add(_key(pair['input'] != info['background']))
    assert len(signatures) == 42 and len(normalized) >= 180
    assert {f['scale'] for _, f, _, _ in features} == {2, 3, 4}
    assert {f['piece_count'] for _, f, _, _ in features} >= {2, 3, 4, 5}
    assert len({a for _, _, a, _ in features}) >= 30
    assert len({b for _, _, _, b in features}) >= 20
    assert len({k for _, f, _, _ in features for k in f['shape_classes']}) >= 25
    assert len({f['background'] for _, f, _, _ in features}) == 10
    assert any(f['unhinted'] for _, f, _, _ in features)
    assert any(len(f['wire_colors']) >= 2 for _, f, _, _ in features)
    assert any(f['link_count'] >= 3 for _, f, _, _ in features)
    assert any(f['blocked'] for _, f, _, _ in features)
    for split in ('train', 'test'):
        fs = [f for s, f, _, _ in features if s == split]
        assert any(f['pooled'] for f in fs) and any(f['unhinted'] for f in fs)
        assert any(f['mismatches'] for f in fs)
        assert any(f['board'].shape[0] < f['board'].shape[1] for f in fs)
        assert any(f['board'].shape[0] > f['board'].shape[1] for f in fs)
    first = generate_puzzle(918273, 6, 3)
    again = generate_puzzle(918273, 6, 3)
    assert all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], again[s])
               for k in ('input', 'output'))
    # Both whole-grid D4 transformations and arbitrary bijective recolorings commute.
    for i, pair in enumerate(first['train']+first['test']):
        recolor = np.random.default_rng(700+i).permutation(10).astype(np.int8)
        inp = recolor[pair['input']]
        out = recolor[pair['output']]
        for t in range(8):
            a = np.rot90(np.fliplr(inp) if t >= 4 else inp, t % 4).copy()
            b = np.rot90(np.fliplr(out) if t >= 4 else out, t % 4).copy()
            assert np.array_equal(solve(a), b)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    # A source body pixel cannot be removed without breaking the common block scale.
    pair = first['train'][0]
    info = _recover(pair['input'])[1]
    corrupt = pair['input'].copy()
    corrupt[info['pieces'][0]['cells'][0]] = info['background']
    try:
        solve(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError('broken macrocell accepted')
    return None


if __name__ == '__main__':
    self_test()
    print('self_test passed')
