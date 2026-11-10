"""Nested perforated plates: color-matched subtraction meets material packing.

Only NumPy and the standard library are used. No files, source examples, seeds,
or generation metadata are consulted by solve(). Diagnostics are kept in memory.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for sy, sx in np.argwhere(mask):
        sy, sx = int(sy), int(sx)
        if seen[sy, sx]:
            continue
        seen[sy, sx] = True
        todo, cells = [(sy, sx)], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        result.append(cells)
    return result


def _bounds(cells):
    a = np.asarray(cells)
    y, x = a.min(axis=0)
    b, r = a.max(axis=0) + 1
    return int(y), int(x), int(b), int(r)


def _span(mask):
    return bool(mask[0].any() and mask[-1].any()
                and mask[:, 0].any() and mask[:, -1].any())


def _material(holes):
    keep = np.ones((holes.shape[0] + 2, holes.shape[1] + 2), dtype=bool)
    keep[1:-1, 1:-1] = ~holes
    return keep


def _separate(first, second):
    y, x, b, r = first
    yy, xx, bb, rr = second
    return b < yy or bb < y or r < xx or rr < x


def _parse(input_grid):
    """Read the frame, local palette roles, solid inventory, and stencil registers."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])):
        raise ValueError("The canvas needs a background margin")
    candidates = []
    for marker in (int(c) for c in np.unique(a) if c != bg):
        y, x, b, r = _bounds(np.argwhere(a == marker))
        if min(b - y, r - x) < 6:
            continue
        region = a[y:b, x:r]
        edge = np.ones(region.shape, dtype=bool)
        edge[1:-1, 1:-1] = False
        if not np.all(region[edge] == marker) or np.any(region[~edge] == marker):
            continue
        interior = region[1:-1, 1:-1]
        colors = set(int(c) for c in np.unique(interior))
        if len(colors) != 2 or bg not in colors:
            continue
        vacancy = next(c for c in colors if c != bg)
        outside = np.ones(a.shape, dtype=bool)
        outside[y:b, x:r] = False
        if np.any((a == vacancy) & outside):
            continue
        candidates.append((y, x, b, r, marker, vacancy, outside))
    if len(candidates) != 1:
        raise ValueError("Expected one uniquely framed two-color destination")
    y, x, b, r, marker, vacancy, outside = candidates[0]
    types, boxes = [], [(y, x, b, r)]
    for color in sorted(int(c) for c in np.unique(a[outside]) if c != bg):
        components = _components(outside & (a == color))
        solids = []
        for cells in components:
            yy, xx, bb, rr = _bounds(cells)
            if min(bb - yy, rr - xx) >= 4 and (bb - yy) * (rr - xx) == len(cells):
                solids.append((len(cells), yy, xx, bb, rr))
        if not solids:
            raise ValueError("A stencil has no solid plate of its color")
        area = max(s[0] for s in solids)
        solids = [s for s in solids if s[0] == area]
        shapes = {(bb - yy, rr - xx) for _, yy, xx, bb, rr in solids}
        if len(shapes) != 1:
            raise ValueError("All copies of a plate color must have the same size")
        h, w = next(iter(shapes))
        ink = outside & (a == color)
        origins = []
        for _, yy, xx, bb, rr in solids:
            ink[yy:bb, xx:rr] = False
            boxes.append((yy, xx, bb, rr))
            origins.append((yy, xx))
        if not ink.any():
            raise ValueError("Missing stencil")
        sy, sx, sb, sr = _bounds(np.argwhere(ink))
        if (sb - sy, sr - sx) != (h - 2, w - 2):
            raise ValueError("Stencil bounds must equal the plate inset")
        register = a[sy:sb, sx:sr]
        if not np.all((register == bg) | (register == color)):
            raise ValueError("Mixed stencil register")
        holes = register == color
        if not _span(holes) or not 0 < int(holes.sum()) < holes.size:
            raise ValueError("Stencil must be nonfull, nonempty, and full-span")
        boxes.append((sy, sx, sb, sr))
        types.append(dict(color=color, holes=holes, keep=_material(holes),
                          count=len(solids), origins=origins))
    if not 1 <= len(types) <= 7 or sum(t['count'] for t in types) > 7:
        raise ValueError("Expected one to seven source plates")
    if any(not _separate(p, q) for i, p in enumerate(boxes) for q in boxes[i + 1:]):
        raise ValueError("Source registers and the frame require separating background")
    return dict(grid=a, bg=bg, marker=marker, vacancy=vacancy,
                box=(y, x, b, r), board=a[y + 1:b - 1, x + 1:r - 1].copy(),
                types=types)


def _pack(mask, types, max_nodes=None):
    """Exact cover of material cells. Holes impose no occupancy constraint.

    Return up to two rendered answers, not just the first construction witness.
    Same-color copies share a finite counter. Bounding rectangles may overlap.
    """
    h, w = mask.shape
    coords = [(int(y), int(x)) for y, x in np.argwhere(mask)]
    if sum(int(t['keep'].sum()) * t['count'] for t in types) != len(coords):
        return [], dict(nodes=0, dead_ends=0, placements=0)
    full = sum(1 << (y * w + x) for y, x in coords)
    by_cell = {y * w + x: [] for y, x in coords}
    placements = []
    for ti, t in enumerate(types):
        keep = t['keep']
        ph, pw = keep.shape
        dy, dx = np.nonzero(keep)
        for y in range(h - ph + 1):
            for x in range(w - pw + 1):
                if not np.all(mask[y + dy, x + dx]):
                    continue
                indices = tuple(int(v) for v in (y + dy) * w + x + dx)
                bits = sum(1 << cell for cell in indices)
                pi = len(placements)
                placements.append((ti, y, x, bits, indices))
                for cell in indices:
                    by_cell[cell].append(pi)
    left = [t['count'] for t in types]
    path, answers, dead = [], {}, set()
    nodes = failures = 0

    def search(covered):
        nonlocal nodes, failures
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            raise ValueError("Construction exact-cover budget exhausted")
        if covered == full:
            if any(left):
                return False
            rendered = np.full(mask.shape, -1, dtype=np.int8)
            for pi in path:
                ti, y, x, _, _ = placements[pi]
                keep = types[ti]['keep']
                rendered[y:y + keep.shape[0], x:x + keep.shape[1]][keep] = types[ti]['color']
            key = rendered[mask].tobytes()
            answers.setdefault(key, (rendered, tuple((placements[pi][0], placements[pi][1],
                                                      placements[pi][2]) for pi in path)))
            return True
        state = covered, tuple(left)
        if state in dead:
            return False
        best = None
        for cell, options in by_cell.items():
            if covered & (1 << cell):
                continue
            valid = [pi for pi in options if left[placements[pi][0]]
                     and not covered & placements[pi][3]]
            if not valid:
                dead.add(state)
                failures += 1
                return False
            if best is None or len(valid) < len(best):
                best = valid
                if len(valid) == 1:
                    break
        found = False
        for pi in best or []:
            ti, _, _, bits, _ = placements[pi]
            left[ti] -= 1
            path.append(pi)
            found = search(covered | bits) or found
            path.pop()
            left[ti] += 1
            if len(answers) >= 2:
                return True
        if not found:
            dead.add(state)
        return found

    search(0)
    return list(answers.values()), dict(nodes=nodes, dead_ends=failures,
                                       placements=len(placements))


def _analyze(input_grid, max_nodes=None):
    info = _parse(input_grid)
    mask = info['board'] == info['vacancy']
    answers, stats = _pack(mask, info['types'], max_nodes)
    if len(answers) != 1:
        raise ValueError("Destination must have exactly one rendered exact cover")
    rendering, path = answers[0]
    y, x, b, r = info['box']
    out = info['grid'][y:b, x:r].copy()
    out[1:-1, 1:-1][mask] = rendering[mask]
    info.update(output=out, path=path, stats=stats)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve entirely from visible geometry and within-input color equality."""
    return _analyze(input_grid)['output']


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _symmetries(mask):
    return [np.rot90(mask, i) for i in range(4)] + [
        np.rot90(np.fliplr(mask), i) for i in range(4)]


def _holes(rng, h, w, asymmetric=False):
    kh, kw = h - 2, w - 2
    for _ in range(100):
        mode = int(rng.integers(6))
        m = rng.random((kh, kw)) < float(rng.uniform(.2, .8))
        if mode == 0:
            m[:] = False
            m[0, 0] = m[-1, -1] = True
        elif mode == 1:
            m[:] = True
            m.flat[int(rng.integers(m.size))] = False
        elif mode == 2:
            m[:] = False
            m[int(rng.integers(kh)), :] = True
            m[:, int(rng.integers(kw))] = True
        elif mode == 3:
            m |= m[::-1, ::-1]
        if not _span(m) or not 0 < int(m.sum()) < m.size:
            continue
        if asymmetric and (any(np.array_equal(m, v) for v in _symmetries(m)[1:])
                           or 2 * int(m.sum()) == m.size):
            continue
        return m
    # Valid for all generated witness sizes (at least 5 by 5).
    m = np.zeros((kh, kw), dtype=bool)
    m[0, 0] = m[0, -1] = m[-1, 0] = m[1, -1] = True
    return m


def _attach(rng, rects, shape):
    """Attach to exterior envelopes by a random partial or full shared edge."""
    h, w = shape
    options = []
    for y, x, rh, rw in rects:
        for side in range(4):
            offsets = range(1 - w, rw) if side < 2 else range(1 - h, rh)
            for offset in offsets:
                yy, xx = ((y - h if side == 0 else y + rh, x + offset)
                          if side < 2 else (y + offset, x - w if side == 2 else x + rw))
                if any(yy < ry + sh and yy + h > ry and xx < rx + sw and xx + w > rx
                       for ry, rx, sh, sw in rects):
                    continue
                trial = rects + [(yy, xx, h, w)]
                bh = max(q[0] + q[2] for q in trial) - min(q[0] for q in trial)
                bw = max(q[1] + q[3] for q in trial) - min(q[1] for q in trial)
                if max(bh, bw) <= 15:
                    options.append((yy, xx, h, w))
    return _pick(rng, options) if options else None


def _blueprint(rng, kind, count):
    """Construct material truth directly; packing verification happens afterward."""
    types, placements, exterior = [], [], []
    if kind == 'siblings':
        h, w = int(rng.integers(5, 8)), int(rng.integers(5, 8))
        first = _holes(rng, h, w, asymmetric=True)
        # Near-matched same-size plates make color-to-stencil pairing necessary.
        second = first.copy()
        toggles = []
        for cell in range(second.size):
            m = first.copy()
            m.flat[cell] ^= True
            if _span(m) and 0 < int(m.sum()) < m.size:
                toggles.append(m)
        second = _pick(rng, toggles).copy()
        types = [first, second]
        placements = [(0, 0, 0)]
        exterior = [(0, 0, h, w)]
        q = _attach(rng, exterior, (h, w))
        if q is None:
            return None
        placements.append((1, q[0], q[1]))
        exterior.append(q)
    else:
        rh, rw = int(rng.integers(8, 10)), int(rng.integers(8, 10))
        ch, cw = int(rng.integers(4, 7)), int(rng.integers(4, 7))
        cy = int(rng.integers(1, rh - ch))
        cx = int(rng.integers(1, rw - cw))
        root = rng.random((rh - 2, rw - 2)) < float(rng.uniform(.05, .25))
        root[cy - 1:cy - 1 + ch, cx - 1:cx - 1 + cw] = True
        root[0, 0] = root[-1, -1] = True
        if root.all():
            return None
        child = _holes(rng, ch, cw)
        types = [root, child]
        placements = [(0, 0, 0), (1, cy, cx)]
        exterior = [(0, 0, rh, rw)]
    while len(placements) < count:
        # The nesting demonstration also teaches two identical physical copies.
        repeat = kind == 'repeat' or (len(placements) > 2 and rng.random() < .28)
        if repeat:
            ti = int(rng.integers(1, len(types)))
            h, w = types[ti].shape[0] + 2, types[ti].shape[1] + 2
        else:
            h, w = int(rng.integers(4, 8)), int(rng.integers(4, 8))
            ti = len(types)
        q = _attach(rng, exterior, (h, w))
        if q is None:
            return None
        if not repeat:
            types.append(_holes(rng, h, w))
        exterior.append(q)
        placements.append((ti, q[0], q[1]))
    miny = min(y for _, y, _ in placements)
    minx = min(x for _, _, x in placements)
    # Insets and output margins vary visibly, independent of plate count.
    pady, padx = int(rng.integers(2)), int(rng.integers(2))
    placements = [(ti, y - miny + pady, x - minx + padx) for ti, y, x in placements]
    height = max(y + types[ti].shape[0] + 2 for ti, y, x in placements)
    width = max(x + types[ti].shape[1] + 2 for ti, y, x in placements)
    height += int(rng.integers(min(1, 15 - height) + 1)) if height <= 15 else 0
    width += int(rng.integers(min(1, 15 - width) + 1)) if width <= 15 else 0
    if max(height, width) > 16:
        return None
    return types, placements, (height, width)


def _scene(rng, objects, bg):
    """Place the framed board and all source registers with bounded retries."""
    for attempt in range(28):
        height = int(rng.integers(27, 31)) if attempt < 8 else 30
        width = int(rng.integers(27, 31)) if attempt < 8 else 30
        a = np.full((height, width), bg, dtype=np.int8)
        occupied = np.zeros(a.shape, dtype=bool)
        success = True
        order = [0] + sorted(range(1, len(objects)), key=lambda i: -objects[i].size)
        for oi in order:
            obj = objects[oi]
            h, w = obj.shape
            options = []
            if oi == 0:
                options = [(yy, xx) for yy in (1, height - h - 1)
                           for xx in (1, width - w - 1) if yy >= 1 and xx >= 1]
            else:
                for y in range(1, height - h):
                    for x in range(1, width - w):
                        if not occupied[y - 1:y + h + 1, x - 1:x + w + 1].any():
                            options.append((y, x))
            if not options:
                success = False
                break
            y, x = _pick(rng, options)
            a[y:y + h, x:x + w] = obj
            occupied[y:y + h, x:x + w] = True
        if success:
            y, x, b, r = _bounds(np.argwhere(a != bg))
            return a[y - 1:b + 1, x - 1:r + 1]
    return None


def _make_pair(rng, kind, count):
    for _ in range(96):
        blueprint = _blueprint(rng, kind, count)
        if blueprint is None:
            continue
        masks, placements, shape = blueprint
        palette = [int(c) for c in rng.permutation(10)]
        bg, marker, vacancy = palette[:3]
        colors = palette[3:3 + len(masks)]
        output = np.full((shape[0] + 2, shape[1] + 2), marker, dtype=np.int8)
        output[1:-1, 1:-1] = bg
        owner = np.full(shape, -1, dtype=np.int8)
        objects = []
        for pi, (ti, y, x) in enumerate(placements):
            keep = _material(masks[ti])
            h, w = keep.shape
            if np.any(owner[y:y + h, x:x + w][keep] >= 0):
                raise AssertionError("Construction overlaps material")
            owner[y:y + h, x:x + w][keep] = pi
            output[1 + y:1 + y + h, 1 + x:1 + x + w][keep] = colors[ti]
            objects.append(np.full((h, w), colors[ti], dtype=np.int8))
        board = output.copy()
        board[1:-1, 1:-1][owner >= 0] = vacancy
        for ti, holes in enumerate(masks):
            objects.append(np.where(holes, colors[ti], bg).astype(np.int8))
        # Source placement order has no correspondence to destination positions.
        rng.shuffle(objects)
        a = _scene(rng, [board] + objects, bg)
        if a is None:
            continue
        turns = int(rng.integers(4))
        a, output = np.rot90(a, turns).copy(), np.rot90(output, turns).copy()
        if rng.random() < .5:
            a, output = a[:, ::-1].copy(), output[:, ::-1].copy()
        try:
            analysis = _analyze(a, max_nodes=20000)
        except ValueError:
            continue
        if np.array_equal(analysis['output'], output):
            if kind == 'siblings' and count == 2:
                source_order = [c for _, _, c in sorted((y, x, t['color'])
                                for t in analysis['types'] for y, x in t['origins'])]
                target_order = [analysis['types'][ti]['color'] for ti, y, x in
                                sorted(analysis['path'], key=lambda q: (q[1], q[2]))]
                if source_order == target_order:
                    continue
            return dict(input=a, output=output)
    raise RuntimeError("Exhausted bounded scene/uniqueness attempts")


def _counts(num_train, num_test):
    for name, v, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(nt + ne):
        if i == 0:
            kind, count = 'siblings', 2
        elif i == 1:
            kind, count = 'repeat', 3
        elif i < nt:
            kind, count = _pick(rng, ['siblings', 'nested']), int(rng.integers(2, 4))
        else:
            kind, count = 'nested', int(rng.integers(4, 6))
        for _ in range(8):
            pair = _make_pair(rng, kind, count)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle['train' if i < nt else 'test'].append(pair)
                break
        else:
            raise RuntimeError("Could not obtain distinct inputs")
    return puzzle


def _features(info):
    mask = info['board'] == info['vacancy']
    owner = np.full(mask.shape, -1, dtype=int)
    boxes = []
    for pi, (ti, y, x) in enumerate(info['path']):
        keep = info['types'][ti]['keep']
        h, w = keep.shape
        assert not np.any(owner[y:y + h, x:x + w][keep] >= 0)
        owner[y:y + h, x:x + w][keep] = pi
        boxes.append((y, x, y + h, x + w))
    nested = 0
    for i, (y, x, b, r) in enumerate(boxes):
        for j, (yy, xx, bb, rr) in enumerate(boxes):
            nested += int(i != j and y < yy and x < xx and bb < b and rr < r)
    merged = max(len({int(owner[y, x]) for y, x in comp}) for comp in _components(mask))
    return dict(plates=len(boxes), types=len(info['types']), nested=nested,
                repeated=any(t['count'] > 1 for t in info['types']), merged=merged,
                shape=info['output'].shape, bg=info['bg'],
                source_shapes=tuple(sorted(t['keep'].shape for t in info['types'])),
                holes=sum(int(t['holes'].sum()) * t['count'] for t in info['types']),
                local_placements=info['stats']['placements'],
                nodes=info['stats']['nodes'], dead_ends=info['stats']['dead_ends'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos, features = set(), [], []
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        a, out = _grid(pair['input']), _grid(pair['output'])
        assert _key(a) not in seen
        seen.add(_key(a))
        info = _analyze(a)
        assert np.array_equal(info['output'], out), "Incorrect material packing or frame crop"
        assert not np.any(out == info['vacancy']), "Unfilled vacancy remains"
        assert np.all((info['board'] == info['bg']) == (out[1:-1, 1:-1] == info['bg']))
        assert len(info['path']) == sum(t['count'] for t in info['types'])
        # Ignoring subtraction cannot exactly cover the same vacancy area.
        solid_area = sum(t['keep'].size * t['count'] for t in info['types'])
        assert solid_area > int((info['board'] == info['vacancy']).sum())
        infos.append(info)
        features.append(_features(info))
    assert all(f['merged'] >= 2 for f in features), "Packing never interacts within a component"
    assert features[0]['nested'] == 0 and features[1]['nested'] >= 1
    assert features[1]['repeated'], "Finite-copy demonstration missing"
    assert max(f['plates'] for f in features[:nt]) < min(f['plates'] for f in features[nt:])
    assert all(f['nested'] >= 1 for f in features[nt:])
    # The first demonstration distinguishes same-sized plate identities by holes.
    first = infos[0]
    ts = first['types']
    assert len(ts) == 2 and ts[0]['keep'].shape == ts[1]['keep'].shape
    assert not np.array_equal(ts[0]['holes'], ts[1]['holes'])
    assert first['stats']['placements'] > 2, "No competing local placement is demonstrated"
    source_order = [c for _, _, c in sorted((y, x, t['color'])
                    for t in ts for y, x in t['origins'])]
    target_order = [ts[ti]['color'] for ti, y, x in
                    sorted(first['path'], key=lambda q: (q[1], q[2]))]
    assert source_order != target_order, "Reading-order transfer remains a shortcut"
    assert any(all(not np.array_equal(t['holes'], v) for v in _symmetries(t['holes'])[1:])
               and 2 * int(t['holes'].sum()) != t['holes'].size for t in ts)
    swapped = [dict(t, keep=ts[1 - i]['keep']) for i, t in enumerate(ts)]
    alternatives, _ = _pack(first['board'] == first['vacancy'], swapped)
    assert len(alternatives) == 1
    assert not np.array_equal(alternatives[0][0][first['board'] == first['vacancy']],
                              first['output'][1:-1, 1:-1][first['board'] == first['vacancy']])


def self_test():
    """Stream 42 seeds; retain hashes/features only, and write no artifacts."""
    hashes, structures, input_shapes, output_shapes, source_shapes = set(), set(), set(), set(), set()
    feats, roles = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for pair in puzzle['train'] + puzzle['test']:
            for role in ('input', 'output'):
                shape, payload = _key(pair[role])
                digest.update(bytes(shape) + payload)
            info = _analyze(pair['input'])
            f = _features(info)
            feats.append(f)
            input_shapes.add(pair['input'].shape)
            output_shapes.add(pair['output'].shape)
            source_shapes.update(f['source_shapes'])
            roles.add((info['bg'], info['marker'], info['vacancy']))
            mapping = {}
            normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in pair['input'].flat)
            structures.add(hashlib.sha256(bytes(pair['input'].shape) + normalized).digest())
        hashes.add(digest.digest())
    assert len(hashes) == 42 and len(structures) == 210
    assert len(input_shapes) >= 20 and len(output_shapes) >= 35
    assert len(source_shapes) >= 16 and len(roles) >= 100
    assert {f['plates'] for f in feats} == {2, 3, 4, 5}
    assert {f['nested'] > 0 for f in feats} == {False, True}
    assert any(f['repeated'] for f in feats) and max(f['merged'] for f in feats) >= 4
    assert any(f['local_placements'] > f['plates'] for f in feats)
    # Determinism and every legal explicit count; large bundles remain in memory.
    def signature(p):
        return tuple(_key(g) for s in ('train', 'test') for p0 in p[s] for g in p0.values())
    assert signature(generate_puzzle(123, 2, 1)) == signature(generate_puzzle(123, 2, 1))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            validate_generated_puzzle(p)
    for counts in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)]:
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted illegal counts")
    # Removing a physical copy cannot satisfy the all-inventory area constraint.
    p = generate_puzzle(11, 2, 1)
    info = _parse(p['train'][1]['input'])
    reduced = [dict(t) for t in info['types']]
    next(t for t in reduced if t['count'] > 1)['count'] -= 1
    assert not _pack(info['board'] == info['vacancy'], reduced)[0]
    # Exercise valid topologies outside the sampler: three concentric plates,
    # and a single rigid plate whose retained material has an isolated island.
    outer = np.zeros((10, 10), dtype=bool)
    outer[1:9, 1:9] = True
    outer[0, 0] = outer[-1, -1] = True
    middle = np.zeros((6, 6), dtype=bool)
    middle[1:5, 1:5] = True
    middle[0, 0] = middle[-1, -1] = True
    island = np.ones((5, 5), dtype=bool)
    island[2, 2] = False
    cases = [([outer, middle, np.eye(2, dtype=bool)], [(0, 0), (2, 2), (4, 4)], 12),
             ([island], [(0, 0)], 7)]
    for masks, positions, side in cases:
        expected = np.ones((side + 2, side + 2), dtype=np.int8)
        expected[1:-1, 1:-1] = 0
        objects = []
        for ti, (holes, (y, x)) in enumerate(zip(masks, positions)):
            keep = _material(holes)
            h, w = keep.shape
            expected[y + 1:y + h + 1, x + 1:x + w + 1][keep] = ti + 3
            objects.append(np.full(keep.shape, ti + 3, dtype=np.int8))
            objects.append(np.where(holes, ti + 3, 0).astype(np.int8))
        board = expected.copy()
        board[board >= 3] = 2
        a = _scene(np.random.default_rng(987), [board] + objects, 0)
        assert a is not None
        assert np.array_equal(solve(a), expected)
    # Equal-shaped but differently colored plates must not receive an arbitrary
    # tie break when exchanging them changes the rendered completion.
    keep = _material(np.eye(2, dtype=bool))
    mask = np.concatenate((keep, keep), axis=1)
    ambiguous = [dict(keep=keep, color=c, count=1) for c in (3, 4)]
    assert len(_pack(mask, ambiguous)[0]) == 2
    # A corrupt target cell must fail validation, even if the schema remains valid.
    out = p['test'][0]['output']
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Accepted a corrupted frame")


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
