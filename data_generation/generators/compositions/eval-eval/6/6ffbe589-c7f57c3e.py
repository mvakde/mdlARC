"""Rotationally addressed, scaffold-registered decoration transport.

Only NumPy and the standard library are used. No parent files, labels, seeds or
generation metadata are consulted by the public input-only solver.
"""
import json
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger cells"
    assert np.all((0 <= a) & (a <= 9)), "invalid colors"
    return a.astype(np.int8)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _signature(mapping):
    return tuple(sorted((int(y), int(x), int(c)) for (y, x), c in mapping.items()))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], [start]
        while stack:
            y, x = stack.pop()
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    group.append(p)
        groups.append(group)
    return groups


def _bounds(points):
    return min(y for y, x in points), min(x for y, x in points), max(y for y, x in points)+1, max(x for y, x in points)+1


def _separated(a, b, gap=1):
    y, x, h, w = a
    v, u, k, l = b
    return y+h+gap <= v or v+k+gap <= y or x+w+gap <= u or u+l+gap <= x


def _ring(n):
    a = np.zeros((n, n), dtype=bool)
    a[0] = a[-1] = True
    a[:, 0] = a[:, -1] = True
    return a


def _carriers(a, bg):
    candidates = []
    for color in sorted(set(map(int, np.unique(a))) - {bg}):
        objects = []
        for group in _components(a == color):
            y, x, end_y, end_x = _bounds(group)
            d = end_y-y
            if d != end_x-x or d < 5:
                break
            mask = a[y:end_y, x:end_x] == color
            s = int(mask.sum(axis=1).min()) // 2
            if s < 1 or d % s or not 5 <= d // s <= 7:
                break
            n = d // s
            wanted = np.repeat(np.repeat(_ring(n), s, 0), s, 1)
            if not np.array_equal(mask, wanted) or len(group) != int(wanted.sum()):
                break
            objects.append({'anchor': (y, x), 'scale': s, 'n': n})
        else:
            if len(objects) >= 2 and len({o['n'] for o in objects}) == 1:
                candidates.append((color, objects))
    assert len(candidates) == 1, "carrier interpretation is not unique"
    return candidates[0]


def _rotate_coordinates(key, bg, counts, direction=1):
    """Solver path uses coordinate arithmetic, not generation's np.rot90."""
    m = len(key)
    out = np.full(key.shape, bg, dtype=np.int8)
    for r, c in np.argwhere(key != bg):
        color = int(key[r, c])
        y, x = int(r), int(c)
        for _ in range((direction * counts.get(color, 0)) % 4):
            y, x = x, m-1-y
        assert out[y, x] == bg, "rotated key masks overlap"
        out[y, x] = color
    return out


def _read_input(input_grid):
    a = _grid(input_grid)
    values, frequencies = np.unique(a, return_counts=True)
    bg = int(values[frequencies.argmax()])
    assert np.count_nonzero(a == bg) * 2 > a.size, "background must be a strict majority"
    carrier, objects = _carriers(a, bg)
    key_colors, states, addresses = set(), set(), {}
    h, w = a.shape
    accounted = np.zeros(a.shape, dtype=bool)
    for obj in objects:
        y, x = obj['anchor']
        s, n = obj['scale'], obj['n']
        crop = a[y:y+n*s, x:x+n*s]
        accounted[y:y+n*s, x:x+n*s] = True
        logical = crop[::s, ::s]
        assert np.array_equal(crop, np.repeat(np.repeat(logical, s, 0), s, 1)), "nonuniform core blocks"
        key = logical[1:-1, 1:-1].copy()
        assert carrier not in key, "carrier color used in key"
        colors = set(map(int, np.unique(key))) - {bg}
        assert 2 <= len(colors) <= 3, "key palette size"
        if key_colors:
            assert colors == key_colors, "inconsistent key palettes"
        key_colors = colors
        decoration = {}
        for r in range(-1, n+1):
            for c in range(-1, n+1):
                if 0 <= r < n and 0 <= c < n:
                    continue
                if not ((r in (-1, n) and 0 <= c < n) or (c in (-1, n) and 0 <= r < n)):
                    continue
                yy, xx = y+r*s, x+c*s
                # Only cells contacting this frame are its decoration. A long
                # receiver scale can place an unrelated counter inside the
                # notional fringe on an unused side; key-colored cells are
                # commands, never decoration, and are read separately below.
                if r in (-1, n):
                    edge_y = y-1 if r == -1 else y+n*s
                    contact = a[edge_y, max(0, xx):min(w, xx+s)] if 0 <= edge_y < h else np.empty(0, dtype=int)
                else:
                    edge_x = x-1 if c == -1 else x+n*s
                    contact = a[max(0, yy):min(h, yy+s), edge_x] if 0 <= edge_x < w else np.empty(0, dtype=int)
                if not np.any(~np.isin(contact, list(colors | {carrier, bg}))):
                    continue
                block = a[max(0, yy):min(h, yy+s), max(0, xx):min(w, xx+s)] if yy < h and xx < w and yy+s > 0 and xx+s > 0 else np.empty((0, 0), dtype=int)
                if not block.size or np.all(block == bg):
                    continue
                assert block.shape == (s, s) and np.all(block == block[0, 0]), "invalid fringe blocks"
                color = int(block[0, 0])
                assert color not in colors | {carrier}, "counter or other carrier in fringe"
                decoration[r, c] = color
                accounted[yy:yy+s, xx:xx+s] = True
        assert decoration, "missing decoration state"
        signature = _signature(decoration)
        states.add(signature)
        ik = _key(key)
        if ik in addresses:
            assert addresses[ik]['signature'] == signature, "inconsistent duplicate address"
        else:
            addresses[ik] = {'key': key, 'decoration': decoration, 'signature': signature}
        obj.update(key=key, key_id=ik, decoration=decoration, signature=signature)
    assert len(states) == 2, "exactly two decoration states required"
    state_maps = [{(r, c): color for r, c, color in t} for t in sorted(states)]
    decoration_colors = {c for t in state_maps for c in t.values()}
    assert not decoration_colors & key_colors, "key/decoration colors overlap"
    n = objects[0]['n']
    union = {(0, 0), (n-1, n-1)} | set(state_maps[0]) | set(state_maps[1])
    lo_y, lo_x, hi_y, hi_x = _bounds(union)
    covered = np.zeros(a.shape, dtype=bool)
    rectangles = []
    for obj in objects:
        y, x = obj['anchor']
        s = obj['scale']
        rect = (y+lo_y*s, x+lo_x*s, (hi_y-lo_y)*s, (hi_x-lo_x)*s)
        r, c, rh, rw = rect
        assert 0 <= r and r+rh <= h and 0 <= c and c+rw <= w, "replacement would be clipped"
        assert all(_separated(rect, old) for old in rectangles), "replacement rectangles collide"
        rectangles.append(rect)
        covered[r:r+rh, c:c+rw] = True
    assert not np.any((a != bg) & covered & ~accounted), "unexplained material in motif region"
    counts, orientations, marker_points = {}, {}, []
    for color in sorted(set(map(int, np.unique(a[~covered]))) - {bg}):
        assert color in key_colors, "unmatched counter color"
        points = np.argwhere((a == color) & ~covered)
        rows, cols = points.T
        horizontal = np.ptp(rows) == 0 and np.ptp(cols)+1 == len(points)
        vertical = np.ptp(cols) == 0 and np.ptp(rows)+1 == len(points)
        assert (horizontal or vertical) and 1 <= len(points) <= 9, "counter must be one unary bar"
        counts[color] = len(points)
        orientations[color] = 'single' if len(points) == 1 else ('horizontal' if horizontal else 'vertical')
        for py, px in points:
            assert all(max(r-py, 0, py-(r+rh-1), c-px, px-(c+rw-1)) >= 2 for r, c, rh, rw in rectangles), "counter touches motif region"
            marker_points.append((int(py), int(px), color))
    assert counts, "missing counters"
    assert all(c == d or max(abs(y-v), abs(x-u)) >= 2 for i, (y, x, c) in enumerate(marker_points) for v, u, d in marker_points[i+1:]), "counters touch"
    routes = {}
    for ik, data in addresses.items():
        rotated = _rotate_coordinates(data['key'], bg, counts)
        target = _key(rotated)
        assert target in addresses, "missing addressed key"
        routes[ik] = target
    assert len(set(routes.values())) == len(addresses), "address action must be a permutation"
    return {'input': a, 'background': bg, 'carrier': carrier, 'objects': objects, 'addresses': addresses,
            'routes': routes, 'states': state_maps, 'counts': counts, 'orientations': orientations,
            'rectangles': rectangles, 'key_colors': key_colors}


def _paint(grid, mapping, anchor, scale):
    for (r, c), color in mapping.items():
        y, x = anchor[0]+r*scale, anchor[1]+c*scale
        grid[y:y+scale, x:x+scale] = color


def _expected(info, routes=None):
    routes = info['routes'] if routes is None else routes
    out = info['input'].copy()
    for obj in info['objects']:
        old = {p: info['background'] for p in obj['decoration']}
        _paint(out, old, obj['anchor'], obj['scale'])
        target = info['addresses'][routes[obj['key_id']]]['decoration']
        _paint(out, target, obj['anchor'], obj['scale'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _expected(_read_input(input_grid))


def _rotate_array(key, bg, counts):
    out = np.full(key.shape, bg, dtype=np.int8)
    used = np.zeros(key.shape, dtype=bool)
    for color in np.unique(key):
        if color == bg:
            continue
        mask = np.rot90(key == color, -counts.get(int(color), 0))
        assert not np.any(used & mask)
        out[mask] = color
        used |= mask
    return out


def _make_key(rng, m, bg, colors):
    remaining = {(r, c) for r in range(m) for c in range(m)}
    orbits = []
    while remaining:
        r, c = min(remaining)
        orbit = list(dict.fromkeys(((r, c), (c, m-1-r), (m-1-r, m-1-c), (m-1-c, r))))
        remaining.difference_update(orbit)
        orbits.append(orbit)
    big = [o for o in orbits if len(o) == 4]
    rng.shuffle(big)
    owners = list(map(int, rng.permutation(len(colors))))
    owners += list(map(int, rng.integers(0, len(colors), len(big)-len(colors))))
    key = np.full((m, m), bg, dtype=np.int8)
    for j, (orbit, owner) in enumerate(zip(big, owners)):
        if j < len(colors):
            # Each color has an observable asymmetric mask in one orbit.
            count = int(rng.integers(1, 4))
            phase = int(rng.integers(4))
            selected = [(phase+k) % 4 for k in range(count)]
        else:
            mode = int(rng.integers(4))
            phase = int(rng.integers(4))
            selected = [phase] if mode == 0 else ([phase, (phase+2) % 4] if mode == 1 else list(range(4)))
        for k in selected:
            key[orbit[int(k)]] = colors[owner]
    if m % 2 and rng.random() < .6:
        key[m//2, m//2] = int(rng.choice(colors))
    return key


def _side_point(side, t, n):
    return [(-1, t), (t, n), (n, t), (t, -1)][side]


def _decorations(rng, n, kind, colors, compact=False):
    side = int(rng.integers(4))
    def patch(s):
        count = int(rng.integers(1, min(n, 4)+1))
        start = int(rng.integers(n-count+1))
        positions = list(range(start, start+count))
        if count >= 3 and rng.random() < .4:
            positions.pop(int(rng.integers(1, count-1)))
        return {_side_point(s, t, n) for t in positions}
    a = patch(side)
    if kind == 'recolor':
        b = set(a)
    elif kind == 'relocate':
        b = patch((side+int(rng.integers(1, 4))) % 4)
    else:
        other = (side+1) % 4 if compact else (side+int(rng.integers(1, 4))) % 4
        common = _side_point(side, int(rng.integers(n)), n)
        a.add(common)
        b = patch(other) | {common}
    maps = []
    for points, palette in zip((a, b), colors):
        ordered = sorted(points)
        rng.shuffle(ordered)
        maps.append({p: palette[i % len(palette)] for i, p in enumerate(ordered)})
    return maps


def _shelf(sizes, width, order, gap):
    result = [None] * len(sizes)
    y, x, tall = 0, 0, 0
    for i in order:
        h, w = sizes[i]
        if x and x+w > width:
            y, x, tall = y+tall+gap, 0, 0
        if w > width or y+h > 28:
            return None
        result[i] = (y, x, h, w)
        x += w+gap
        tall = max(tall, h)
    return result


def _layouts(rng, sizes):
    choices = []
    for transposed in (False, True):
        dims = [(w, h) for h, w in sizes] if transposed else sizes
        orders = [sorted(range(len(dims)), key=lambda i: (dims[i][0], dims[i][1]), reverse=True), list(rng.permutation(len(dims)))]
        for gap in (1, 2):
            for width in range(max(w for h, w in dims), 29):
                for order in orders:
                    p = _shelf(dims, width, order, gap)
                    if p is not None:
                        choices.append([(x, y, w, h) for y, x, h, w in p] if transposed else p)
    return choices


def _markers(rng, grid, bg, rectangles, counts, orientations):
    h, w = grid.shape
    blocked = np.zeros(grid.shape, dtype=bool)
    for y, x, rh, rw in rectangles:
        blocked[max(0, y-1):min(h, y+rh+1), max(0, x-1):min(w, x+rw+1)] = True
    for color, length in sorted(counts.items(), key=lambda item: -item[1]):
        candidates = []
        options = [orientations[color]] if color in orientations else ['horizontal', 'vertical']
        for direction in options:
            dh, dw = (1, length) if direction == 'horizontal' else (length, 1)
            for y in range(h-dh+1):
                for x in range(w-dw+1):
                    if not np.any(blocked[y:y+dh, x:x+dw]):
                        candidates.append((y, x, dh, dw))
        if not candidates:
            return False
        y, x, dh, dw = candidates[int(rng.integers(len(candidates)))]
        grid[y:y+dh, x:x+dw] = color
        blocked[max(0, y-1):min(h, y+dh+1), max(0, x-1):min(w, x+dw+1)] = True
    return True


def _generate_pair(rng, split, index, regime):
    witness = split == 'train' and index < 2
    if witness:
        m, layer_count = (3, 2) if index == 0 else (4, 3)
        lengths = [1, 2] if index == 0 else [3, 4, 0]
        kind = 'combined'
    else:
        m = 5 if split == 'test' and regime == 1 else (3 if regime == 2 else int(rng.integers(3, 6 if split == 'test' else 5)))
        layer_count = 2 if m == 3 else int(rng.integers(2, 4))
        cap = 4 if split == 'train' and regime == 0 else 9
        lengths = [int(rng.integers(0, cap+1)) for _ in range(layer_count)]
        # Nonidentity address action, with both period two and four allowed.
        lengths[0] = int(rng.choice([1, 2, 3])) if split == 'train' else int(rng.choice([1, 2, 3, 5, 6, 7, 9]))
        if split == 'test' and regime == 0:
            lengths[int(rng.integers(layer_count))] = int(rng.integers(5, 10))
        if all(k % 4 == 0 for k in lengths):
            lengths[0] = 5 if split == 'test' and regime == 0 else 2
        kind = str(rng.choice(['recolor', 'relocate', 'combined']))
    roles = list(map(int, rng.permutation(10)))
    bg, carrier = roles[:2]
    key_colors = roles[2:2+layer_count]
    cursor = 2+layer_count
    color_counts = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
    state_colors = [roles[cursor:cursor+color_counts[0]], roles[cursor+color_counts[0]:cursor+sum(color_counts)]]
    counts = {c: k for c, k in zip(key_colors, lengths) if k}
    orientations = {}
    if witness:
        first = 'horizontal' if rng.random() < .5 else 'vertical'
        orientations[key_colors[0]] = first
        orientations[key_colors[1]] = 'vertical' if first == 'horizontal' else 'horizontal'
    n = m+2
    compact = split == 'test' and regime == 2
    templates = _decorations(rng, n, kind, state_colors, compact)
    base = _make_key(rng, m, bg, key_colors)
    keys, current = [], base
    while not keys or not np.array_equal(current, base):
        assert len(keys) < 4, "rotation cycle cannot exceed four"
        keys.append(current)
        current = _rotate_array(current, bg, counts)
    period = len(keys)
    assert period >= 2
    # Optionally a second independent cycle; copies remain one coherent lookup.
    if not witness and m == 3 and not compact and rng.random() < .25:
        known = {_key(k) for k in keys}
        for _ in range(16):
            other = _make_key(rng, m, bg, key_colors)
            cycle, current = [], other
            while not cycle or not np.array_equal(current, other):
                cycle.append(current)
                current = _rotate_array(current, bg, counts)
            if len(cycle) == period and not known & {_key(k) for k in cycle}:
                keys += cycle
                break
    states = []
    for _ in range(len(keys)//period):
        schedule = [0, 1] if period == 2 else ([0, 0, 1, 1] if witness else list(map(int, rng.permutation([0, 0, 1, 1]))))
        states += schedule
    vocabulary = {_key(k): (k, templates[state]) for k, state in zip(keys, states)}
    destinations = {_key(k): _key(_rotate_array(k, bg, counts)) for k in keys}
    changed = [i for i, k in enumerate(keys) if _signature(vocabulary[destinations[_key(k)]][1]) != _signature(templates[states[i]])]
    assert changed
    union = {(0, 0), (n-1, n-1)} | set(templates[0]) | set(templates[1])
    lo_y, lo_x, hi_y, hi_x = _bounds(union)
    logical_shape = hi_y-lo_y, hi_x-lo_x
    layout_options = None
    for attempt in range(18):
        extra = int(rng.integers(0, 4)) if not witness and attempt < 12 else 0
        copies = list(range(len(keys))) + list(map(int, rng.integers(0, len(keys), extra)))
        scales = [1] * len(copies)
        must_scale = witness or (split == 'test' and regime == 2)
        if must_scale or rng.random() < .6:
            i = int(rng.choice(changed))
            scales[i] = 3 if compact else 2
        mass = sum((4*n-4 + np.count_nonzero(keys[address] != bg) + len(templates[states[address]]))*s*s
                   for address, s in zip(copies, scales)) + sum(counts.values())
        if mass >= 450:
            continue
        sizes = [(logical_shape[0]*s, logical_shape[1]*s) for s in scales]
        layout_options = _layouts(rng, sizes)
        if layout_options:
            break
    assert layout_options, "bounded motif schedule cannot fit"
    for attempt in range(18):
        rects = layout_options[int(rng.integers(len(layout_options)))]
        h0, w0 = max(y+h for y, x, h, w in rects), max(x+w for y, x, h, w in rects)
        h = int(rng.integers(h0, min(30, h0+6)+1)) if attempt < 12 else 30
        w = int(rng.integers(w0, min(30, w0+6)+1)) if attempt < 12 else 30
        dy = int(rng.integers(h-h0+1)) if attempt < 12 else 0
        dx = int(rng.integers(w-w0+1)) if attempt < 12 else 0
        rects = [(y+dy, x+dx, rh, rw) for y, x, rh, rw in rects]
        for _ in range(24 if attempt < 12 else 0):
            i = int(rng.integers(len(rects)))
            _, _, rh, rw = rects[i]
            candidate = (int(rng.integers(h-rh+1)), int(rng.integers(w-rw+1)), rh, rw)
            if all(_separated(candidate, r) for j, r in enumerate(rects) if j != i):
                rects[i] = candidate
        inp = np.full((h, w), bg, dtype=np.int8)
        out = inp.copy()
        for i, ((y, x, rh, rw), s) in enumerate(zip(rects, scales)):
            address = copies[i]
            anchor = y-lo_y*s, x-lo_x*s
            core = np.full((n, n), bg, dtype=np.int8)
            core[_ring(n)] = carrier
            core[1:-1, 1:-1] = keys[address]
            yy, xx = anchor
            rendered = np.repeat(np.repeat(core, s, 0), s, 1)
            inp[yy:yy+n*s, xx:xx+n*s] = rendered
            out[yy:yy+n*s, xx:xx+n*s] = rendered
            _paint(inp, templates[states[address]], anchor, s)
            _paint(out, vocabulary[destinations[_key(keys[address])]][1], anchor, s)
        # At 30x30 there are guaranteed clear bottom/right strips because
        # shelf construction reserves at most 28x28. Prefer varied bars first.
        if attempt == 17:
            for i, (color, length) in enumerate(counts.items()):
                orientation = orientations.get(color, 'horizontal' if i != 1 else 'vertical')
                if orientation == 'vertical':
                    col = 29 if i < 2 else 27
                    inp[:length, col] = color
                else:
                    start = 0 if i < 2 else 11
                    inp[29, start:start+length] = color
        elif not _markers(rng, inp, bg, rects, counts, orientations):
            continue
        if np.count_nonzero(inp == bg)*2 <= inp.size:
            continue
        # Preserve command bars, independently of the substituted payloads.
        inside = np.zeros(inp.shape, dtype=bool)
        for y, x, rh, rw in rects:
            inside[y:y+rh, x:x+rw] = True
        out[~inside] = inp[~inside]
        return {'input': inp, 'output': out}
    raise AssertionError("bounded counter packing exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            for _ in range(32):
                pair = _generate_pair(rng, split, index, regime)
                ik = _key(pair['input'])
                if ik not in seen:
                    seen.add(ik)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(info):
    states = {ik: data['signature'] for ik, data in info['addresses'].items()}
    changed = [o for o in info['objects'] if states[o['key_id']] != states[info['routes'][o['key_id']]]]
    retained = [o for o in info['objects'] if states[o['key_id']] == states[info['routes'][o['key_id']]]]
    inverse = {v: k for k, v in info['routes'].items()}
    directional = not np.array_equal(_expected(info), _expected(info, inverse))
    scaled = any(o['scale'] > 1 and any(d['key_id'] == info['routes'][o['key_id']] and d['scale'] != o['scale'] for d in info['objects']) for o in changed)
    a, b = info['states']
    combined = set(a) != set(b) and any(a[p] != b[p] for p in set(a) & set(b))
    return {'changed': bool(changed), 'retained': bool(retained), 'directional': directional, 'scaled': scaled, 'combined': combined}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _read_input(inp)
            assert out.shape == inp.shape
            assert np.array_equal(out, _expected(info)), "incorrect address transport"
            evidence = _evidence(info)
            assert evidence['changed'], "uninformative identity pair"
            if split == 'train':
                training.append((info, evidence))
    for info, evidence in training[:2]:
        assert all(evidence.values()), "missing teaching witness"
        assert len({v % 4 for v in info['counts'].values()}) >= 2, "independent counts not shown"
    assert set(training[0][0]['counts'].values()) == {1, 2}
    assert set(training[1][0]['counts'].values()) == {3, 4}
    assert training[1][0]['key_colors'] - set(training[1][0]['counts']), "absent-color default not shown"


def self_test():
    shapes, geometries, palettes, backgrounds = set(), set(), set(), set()
    counts, scales, multiplicities, states, kinds, cycle_sizes = set(), set(), set(), set(), set(), set()
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        repeat = generate_puzzle(seed)
        fingerprint = tuple(_key(p[r]) for split in ('train', 'test') for p in puzzle[split] for r in ('input', 'output'))
        assert fingerprint == tuple(_key(p[r]) for split in ('train', 'test') for p in repeat[split] for r in ('input', 'output'))
        fingerprints.add(fingerprint)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _read_input(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                shapes.add(pair['input'].shape)
                backgrounds.add(info['background'])
                palettes.add(tuple(map(int, np.unique(pair['input']))))
                geometries.add(tuple(sorted(tuple((k['key'] != info['background']).flat) for k in info['addresses'].values())))
                counts.update(info['counts'].values())
                scales.update(o['scale'] for o in info['objects'])
                multiplicities.add(len(info['objects']))
                cycle_sizes.add(len(info['addresses']))
                states.add(tuple(_signature(t) for t in info['states']))
                a, b = info['states']
                kinds.add((set(a) != set(b), any(a[p] != b[p] for p in set(a) & set(b))))
    assert len(fingerprints) == 42
    assert len(shapes) >= 25 and len(geometries) >= 50 and len(states) >= 100
    assert len(palettes) >= 25 and backgrounds == set(range(10))
    assert counts == set(range(1, 10)) and scales >= {1, 2, 3}
    assert len(multiplicities) >= 4 and cycle_sizes >= {2, 4, 8}
    assert kinds == {(False, True), (True, False), (True, True)}
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1024+nt*10+ns, nt, ns)
            assert len(p['train']) == nt and len(p['test']) == ns
    for nt, ns in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)]:
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    bad = {s: [{r: g.copy() for r, g in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    info = _read_input(p['train'][0]['input'])
    expected = _expected(info)
    # Omitting the address rotation predicts self lookup, hence identity.
    assert not np.array_equal(expected, info['input'])
    # Omitting state substitution and merely rotating visible keys changes
    # protected address material and fails to move the exterior payload.
    keys_only = info['input'].copy()
    for obj in info['objects']:
        y, x = obj['anchor']
        s = obj['scale']
        rotated = _rotate_coordinates(obj['key'], info['background'], info['counts'])
        m = len(rotated)
        keys_only[y+s:y+(m+1)*s, x+s:x+(m+1)*s] = np.repeat(np.repeat(rotated, s, 0), s, 1)
        assert np.array_equal(expected[y+s:y+(m+1)*s, x+s:x+(m+1)*s], info['input'][y+s:y+(m+1)*s, x+s:x+(m+1)*s])
    assert not np.array_equal(keys_only, expected)
    assert np.array_equal(expected == info['carrier'], info['input'] == info['carrier'])
    damaged = info['input'].copy()
    obj = info['objects'][0]
    r, c = np.argwhere(obj['key'] != info['background'])[0]
    y, x = obj['anchor']
    s = obj['scale']
    damaged[y+(r+1)*s:y+(r+2)*s, x+(c+1)*s:x+(c+2)*s] = info['background']
    try:
        _read_input(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("broken address closure accepted")
    return {'seeds': 42, 'shapes': len(shapes), 'key_geometries': len(geometries), 'decoration_designs': len(states),
            'palettes': len(palettes), 'backgrounds': sorted(backgrounds), 'bar_lengths': sorted(counts),
            'scales': sorted(scales), 'motif_counts': sorted(multiplicities), 'address_counts': sorted(cycle_sizes)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
