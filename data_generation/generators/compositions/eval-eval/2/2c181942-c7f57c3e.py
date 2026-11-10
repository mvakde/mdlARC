"""Quarter-turn switches: color docking couples rotation parity to state exchange.

Only standard-library modules and NumPy are used. Templates are inferred from
each input, registering by the dominant-color attachment scaffold. Sentinel 10
is an internal semantic role, never a grid color.
"""
from __future__ import annotations

import math
import numpy as np

_CARRIER = 10
_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_SEEDS = (0, 166379252, 124249024, 171746212, 274265848, 715071638,
          44718609, 179325677, 537792109, 831133124, 133256848, 652049139,
          76049301, 650244596, 98775715, 119664384, 159994046, 267088915,
          697310314, 185737944, 704950113, 970315157, 665447640, 665655737,
          277864604, 687253454, 583866885, 789374521, 831433906, 311827087,
          873315323, 224502464, 291535075, 116303807, 595262533, 614071418,
          402025196, 470035231, 643586353, 160320279, 5289327, 707553813)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _signature(mapping):
    return tuple(sorted((int(r), int(c), int(v)) for (r, c), v in mapping.items()))


def _bounds(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points), max(c for r, c in points))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _NEIGHBORS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    group.add(p)
        result.append(group)
    return result


def _primitive(crop):
    lengths = []
    for a in (crop, crop.T):
        for row in a:
            cuts = [0] + (np.flatnonzero(row[1:] != row[:-1]) + 1).tolist() + [len(row)]
            lengths.extend(b - a for a, b in zip(cuts, cuts[1:]))
    s = math.gcd(*lengths)
    reduced = crop[::s, ::s].copy()
    assert np.array_equal(crop, np.repeat(np.repeat(reduced, s, 0), s, 1))
    return s, reduced


def _attachment_turns(mask):
    candidates = []
    for undo in range(4):
        m = np.rot90(mask, undo)
        h, w = m.shape
        if h < 3 or w < 4 or w % 2:
            continue
        edge = np.zeros(w, dtype=bool)
        edge[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(m[-1], edge):
            candidates.append(undo)
    return candidates


def _socket(small, background):
    for undo in range(4):
        a = np.rot90(small, undo)
        if a.shape != (2, 4):
            continue
        label, wall = int(a[0, 1]), int(a[1, 0])
        if len({label, wall, background}) != 3:
            continue
        if np.array_equal(a, [[background, label, label, background], [wall] * 4]):
            return dict(color=label, wall=wall, q=(-undo) % 4)
    return None


def _module(small, background):
    values, counts = np.unique(small[small != background], return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, 'carrier color must be uniquely dominant'
    carrier = int(values[np.argmax(counts)])
    rr, cc = np.nonzero(small == carrier)
    mask = small[rr.min():rr.max() + 1, cc.min():cc.max() + 1] == carrier
    assert len(_components(mask)) == 1, 'scaffold must be eight-connected'
    turns = _attachment_turns(mask)
    assert len(turns) == 1, 'attachment direction must be unambiguous'
    undo = turns[0]
    a = np.rot90(small, undo).copy()
    rr, cc = np.nonzero(a == carrier)
    r0, c0, r1, c1 = int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())
    scaffold = frozenset((int(r - r0), int(c - c0)) for r, c in zip(rr, cc))
    mapping = {(int(r - r0), int(c - c0)): (_CARRIER if int(a[r, c]) == carrier else int(a[r, c]))
               for r, c in np.argwhere(a != background)}
    return dict(color=carrier, q=(-undo) % 4, template=mapping,
                scaffold=scaffold, core_shape=(r1 - r0 + 1, c1 - c0 + 1))


def _rotate_point(r, c, q):
    for _ in range(q % 4):
        r, c = -c, r
    return int(r), int(c)


def _pixels(mapping, scale, q, carrier):
    result = {}
    for (r, c), value in mapping.items():
        value = carrier if value == _CARRIER else value
        for dr in range(scale):
            for dc in range(scale):
                result[_rotate_point(r * scale + dr, c * scale + dc, q)] = int(value)
    return result


def _dock_pixels(template, core_shape, socket, carrier):
    h, w = core_shape
    s, q = socket['scale'], socket['q']
    neck = {(h - 1, w // 2 - 1): carrier, (h - 1, w // 2): carrier}
    source = _pixels(neck, s, q, carrier)
    dr, dc = _rotate_point(-1, 0, q)
    target = {(r + dr * s, c + dc * s) for r, c in socket['mouth']}
    r0, c0, _, _ = _bounds(source)
    tr, tc, _, _ = _bounds(target)
    shift = tr - r0, tc - c0
    assert {(r + shift[0], c + shift[1]) for r, c in source} == target
    return {(r + shift[0], c + shift[1]): v
            for (r, c), v in _pixels(template, s, q, carrier).items()}


def _read_input(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] * 2 > grid.size, 'background must occupy more than half the input'
    sockets, modules, rectangles = [], [], []
    for points in _components(grid != bg):
        r0, c0, r1, c1 = _bounds(points)
        rect = (r0, c0, r1 - r0 + 1, c1 - c0 + 1)
        assert all(_separated(rect, other) for other in rectangles), 'object boxes need a background gap'
        rectangles.append(rect)
        crop = grid[r0:r1 + 1, c0:c1 + 1]
        assert np.count_nonzero(crop != bg) == len(points), 'interleaved object boxes'
        s, small = _primitive(crop)
        item = _socket(small, bg)
        if item is not None:
            item.update(scale=s, points=points,
                        mouth={p for p in points if int(grid[p]) == item['color']})
            sockets.append(item)
        else:
            item = _module(small, bg)
            item.update(scale=s, points=points)
            modules.append(item)
    assert 1 <= len(sockets) <= 4 and len(modules) >= 2
    assert len({s['wall'] for s in sockets}) == 1
    wall = sockets[0]['wall']
    labels = {s['color'] for s in sockets}
    assert len(labels) == len(sockets), 'socket labels must be distinct'
    assert len({m['scaffold'] for m in modules}) == 1, 'modules must share a scaffold'
    assert all(wall not in m['template'].values() and m['color'] != wall for m in modules)
    accents = set(v for m in modules for v in m['template'].values()) - {_CARRIER}
    assert accents.isdisjoint(labels | {m['color'] for m in modules} | {bg, wall})
    signatures = sorted({_signature(m['template']) for m in modules})
    assert len(signatures) == 2, 'exactly two local states must be visible'
    assert {_signature(m['template']) for m in modules if m['scale'] == 1} == set(signatures), 'both states need unit references'
    templates = [{(r, c): v for r, c, v in sig} for sig in signatures]
    socket_by_color = {s['color']: s for s in sockets}
    matched = []
    used = set()
    for m in modules:
        m['state'] = signatures.index(_signature(m['template']))
        m['socket'] = socket_by_color.get(m['color'])
        if m['socket'] is None:
            m['delta'] = None
            continue
        assert m['color'] not in used, 'two modules cannot claim one socket'
        used.add(m['color'])
        assert m['scale'] == m['socket']['scale'], 'mouth and neck scales must agree'
        m['delta'] = (m['socket']['q'] - m['q']) % 4
        m['next_state'] = m['state'] ^ (m['delta'] % 2)
        matched.append(m)
    assert matched, 'at least one real docking is required'
    # Reserve both possible states, not just the selected answer. Collisions
    # are invalid inputs; they never select a hidden tie-breaking orientation.
    reserved = set(p for socket in sockets for p in socket['points'])
    for m in matched:
        union = set()
        for template in templates:
            union.update(_dock_pixels(template, m['core_shape'], m['socket'], m['color']))
        assert all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in union), 'assembly clipped'
        assert union.isdisjoint(reserved), 'assemblies overlap'
        reserved.update(union)
    return dict(grid=grid, background=bg, wall=wall, sockets=sockets,
                modules=modules, matched=matched, templates=templates)


def _answer(info, mode='parity'):
    out = np.full_like(info['grid'], info['background'])
    for socket in info['sockets']:
        for p in socket['points']:
            out[p] = info['grid'][p]
    for m in info['matched']:
        flip = (m['delta'] % 2) if mode == 'parity' else int(mode == 'always')
        template = info['templates'][m['state'] ^ flip]
        for p, v in _dock_pixels(template, m['core_shape'], m['socket'], m['color']).items():
            assert out[p] == info['background']
            out[p] = v
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from the current input; no demonstrations/state."""
    return _answer(_read_input(input_grid))


def _scaffold(rng, compact=False):
    for _ in range(100):
        h = int(rng.integers(3, 5 if compact else 6))
        w = 4 if compact else int(rng.choice((4, 6)))
        mid = w // 2
        stem = int(rng.integers(1, h - 1))
        body = h - stem
        mask = np.zeros((h, w), dtype=bool)
        mask[:, mid - 1:mid + 1] = True
        kind = int(rng.integers(3))
        if kind == 0:
            mask[:body] = True
        elif kind == 1:
            for r in range(body):
                left = int(rng.integers(mid))
                right = int(rng.integers(mid + 1, w + 1))
                mask[r, left:right] = True
            mask[int(rng.integers(body)), :mid] = True
            mask[int(rng.integers(body)), mid:] = True
        else:
            reach = min(mid - 1, body)
            for r in range(reach):
                mask[r, mid - 1 - reach + r] = True
                mask[r, mid + reach - r] = True
            mask[reach:body, :] = True
        # Holes and asymmetric notches alter the invariant geometry, not just
        # the palette. A central hole also permits an internal state accent.
        for r, c in np.argwhere(mask[:body]):
            if rng.random() < .12:
                mask[r, c] = False
        if not mask[0].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if int(mask.sum()) < 8 or len(_components(mask)) != 1:
            continue
        if _attachment_turns(mask) == [0]:
            return frozenset(map(tuple, np.argwhere(mask))), (h, w)
    # Directional T, with a single centered terminal edge.
    return frozenset([(0, c) for c in range(4)] + [(1, c) for c in range(4)] + [(2, 1), (2, 2)]), (3, 4)


def _decoration(rng, core, shape, side, compact=False):
    h, w = shape
    if side == 0:
        root = sorted(p for p in core if p[0] == 0)[int(rng.integers(sum(r == 0 for r, c in core)))]
        direction, lateral = (-1, 0), (0, 1)
    else:
        col = 0 if side == 1 else w - 1
        options = sorted(p for p in core if p[1] == col and p[0] < h - 1)
        root = options[int(rng.integers(len(options)))]
        direction, lateral = (0, -1 if side == 1 else 1), (1, 0)
    depth = 1 if compact else int(rng.integers(1, 3))
    width = int(rng.integers(1, 3 if compact else 4))
    start = -int(rng.integers(width))
    patch = {(root[0] + direction[0] * d + lateral[0] * t,
              root[1] + direction[1] * d + lateral[1] * t)
             for d in range(1, depth + 1) for t in range(start, start + width)}
    # Keep every decoration strictly above the terminal attachment row.
    return {p for p in patch if p[0] < h - 1}


def _design(rng, accent_colors, kind, compact=False):
    for _ in range(80):
        core, shape = _scaffold(rng, compact)
        sides = list(map(int, rng.choice(3, 2, replace=False)))
        if kind == 'combined':
            sides = [0, int(rng.choice((1, 2)))]
        a = _decoration(rng, core, shape, sides[0], compact)
        b = a if kind == 'recolor' else _decoration(rng, core, shape, sides[1], compact)
        holes = sorted((r, c) for r in range(shape[0] - 1) for c in range(shape[1])
                       if (r, c) not in core and any((r + dr, c + dc) in core for dr, dc in _NEIGHBORS))
        rng.shuffle(holes)
        interior = set(holes[:int(rng.integers(0, min(2, len(holes)) + 1))])
        maps = []
        for state, patch in enumerate((a, b)):
            color = accent_colors[0] if kind == 'relocate' else accent_colors[state]
            mapping = {p: _CARRIER for p in core}
            mapping.update({p: color for p in patch | interior})
            if len(accent_colors) > 2 and interior:
                mapping[min(interior)] = accent_colors[2] if kind == 'relocate' else accent_colors[state]
            maps.append(mapping)
        if not a or not b or maps[0] == maps[1]:
            continue
        if kind == 'combined' and _bounds(maps[0]) == _bounds(maps[1]):
            continue
        if any(max(sum(v == color for v in m.values()) for color in accent_colors) >= len(core) for m in maps):
            continue
        if any(len(_components(_array(m, 0, -1) != -1)) != 1 for m in maps):
            continue
        # Logical block size is always one, hence scale is observable by runs.
        if any(_primitive(_array(m, 0, -1))[0] != 1 for m in maps):
            continue
        return maps, shape
    raise AssertionError('could not create directional two-state design')


def _array(mapping, carrier, background):
    r0, c0, r1, c1 = _bounds(mapping)
    a = np.full((r1 - r0 + 1, c1 - c0 + 1), background, dtype=int)
    for (r, c), v in mapping.items():
        a[r - r0, c - c0] = carrier if v == _CARRIER else v
    return a


def _vocabulary(templates):
    # Carrier is already semantic. Remove accent names and state ordering for
    # a structural novelty check while retaining scaffold registration.
    variants = []
    for order in ((0, 1), (1, 0)):
        colors = {_CARRIER: 0}
        normalized = []
        for i in order:
            normalized.append(tuple((r, c, colors.setdefault(v, len(colors)))
                                    for r, c, v in _signature(templates[i])))
        variants.append(tuple(normalized))
    return min(variants)


def _separated(a, b):
    r, c, h, w = a
    y, x, hh, ww = b
    return r + h < y or y + hh < r or c + w < x or x + ww < c


def _shelves(rng, sizes):
    options = []
    orders = [sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True),
              list(map(int, rng.permutation(len(sizes))))]
    for width in range(max(w for h, w in sizes), 31):
        for order in orders:
            rects, r, c, height = [None] * len(sizes), 0, 0, 0
            for i in order:
                h, w = sizes[i]
                if c and c + w > width:
                    r, c, height = r + height + 1, 0, 0
                if r + h > 30:
                    break
                rects[i] = (r, c, h, w)
                c += w + 1
                height = max(height, h)
            if all(x is not None for x in rects):
                options.append(rects)
    return options[int(rng.integers(len(options)))] if options else None


def _input_modules(rng, grid, arrays):
    blocked = np.zeros(grid.shape, dtype=bool)
    for points in _components(grid != _background(grid)):
        r0, c0, r1, c1 = _bounds(points)
        blocked[max(0, r0 - 1):r1 + 2, max(0, c0 - 1):c1 + 2] = True
    order = sorted(range(len(arrays)), key=lambda i: arrays[i].size, reverse=True)
    placements = []
    bg = _background(grid)
    for i in order:
        a = arrays[i]
        h, w = a.shape
        if h > grid.shape[0] or w > grid.shape[1]:
            return False
        candidates = [(r, c) for r in range(grid.shape[0] - h + 1)
                      for c in range(grid.shape[1] - w + 1)
                      if not blocked[r:r + h, c:c + w].any()]
        if not candidates:
            return False
        r, c = candidates[int(rng.integers(len(candidates)))]
        blocked[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True
        placements.append((r, c, a))
    for r, c, a in placements:
        view = grid[r:r + a.shape[0], c:c + a.shape[1]]
        view[a != bg] = a[a != bg]
    return True


def _background(grid):
    return int(np.argmax(np.bincount(grid.ravel(), minlength=10)))


def _build_pair(rng, config, forbidden_vocab):
    m, n, d = config['matches'], config['sockets'], config['distractors']
    palette = list(map(int, rng.permutation(10)))
    bg, wall = palette[:2]
    labels = palette[2:2 + n]
    extra = palette[2 + n:2 + n + d]
    accents = palette[2 + n + d:]
    assert len(accents) >= 2
    templates, shape = _design(rng, accents[:int(rng.integers(2, min(3, len(accents)) + 1))],
                               config['kind'], config['large_scale'] == 3)
    if _vocabulary(templates) in forbidden_vocab:
        return None
    states, scales, deltas = list(config['states']), list(config['scales']), list(config['deltas'])
    directions = config['directions']
    modules = []
    for i in range(m + d):
        color = labels[i] if i < m else extra[i - m]
        q = (directions[i] - deltas[i]) % 4 if i < m else int(rng.integers(4))
        a = _array(templates[states[i]], color, bg)
        a = np.rot90(np.repeat(np.repeat(a, scales[i], 0), scales[i], 1), q).copy()
        modules.append(a)
    clusters = []
    for i in range(n):
        h, w = shape
        socket_map = {(h, w // 2 - 1): labels[i], (h, w // 2): labels[i]}
        socket_map.update({(h + 1, c): wall for c in range(w // 2 - 2, w // 2 + 2)})
        s = scales[i] if i < m else int(rng.choice((1, 2)))
        q = directions[i]
        socket_pixels = _pixels(socket_map, s, q, labels[i])
        union = set(socket_pixels)
        if i < m:
            union.update(_pixels(templates[0], s, q, labels[i]))
            union.update(_pixels(templates[1], s, q, labels[i]))
        r0, c0, r1, c1 = _bounds(union)
        socket_pixels = {(r - r0, c - c0): v for (r, c), v in socket_pixels.items()}
        answer_pixels = dict(socket_pixels)
        if i < m:
            chosen = templates[states[i] ^ (deltas[i] % 2)]
            answer_pixels.update({(r - r0, c - c0): v for (r, c), v in _pixels(chosen, s, q, labels[i]).items()})
        clusters.append(((r1 - r0 + 1, c1 - c0 + 1), socket_pixels, answer_pixels))
    sizes = [x[0] for x in clusters]
    rects = _shelves(rng, sizes)
    if rects is None:
        return None
    h0 = max(r + h for r, c, h, w in rects)
    w0 = max(c + w for r, c, h, w in rects)
    initial = (max(h0, int(rng.integers(16, 28))), max(w0, int(rng.integers(16, 28))))
    for attempt in range(18):
        height, width = (min(30, initial[0] + attempt // 3), min(30, initial[1] + attempt // 3))
        if attempt >= 12:
            height = width = 30
        dy, dx = int(rng.integers(height - h0 + 1)), int(rng.integers(width - w0 + 1))
        moved = [(r + dy, c + dx, h, w) for r, c, h, w in rects]
        # Shuffle reserved assembly rectangles while respecting both states.
        for _ in range(20):
            i = int(rng.integers(n))
            _, _, h, w = moved[i]
            candidate = (int(rng.integers(height - h + 1)), int(rng.integers(width - w + 1)), h, w)
            if all(_separated(candidate, a) for j, a in enumerate(moved) if j != i):
                moved[i] = candidate
        inp, out = np.full((height, width), bg, dtype=int), np.full((height, width), bg, dtype=int)
        for (r, c, h, w), (_, socket_pixels, answer_pixels) in zip(moved, clusters):
            for (y, x), v in socket_pixels.items():
                inp[r + y, c + x] = v
            for (y, x), v in answer_pixels.items():
                out[r + y, c + x] = v
        if not _input_modules(rng, inp, modules) or np.count_nonzero(inp == bg) * 2 <= inp.size:
            continue
        info = _read_input(inp)
        assert np.array_equal(_answer(info), out), 'renderer and input reader disagree'
        return dict(input=inp, output=out)
    return None


def _configuration(rng, index, split, large_test, witness_directions):
    if split == 'train' and index < 2:
        m, n, d = 3, int(rng.choice((3, 4))), index
        states = [1, 0, int(rng.integers(2))] if index == 0 else [1, 0, int(rng.integers(2)), int(rng.integers(2))]
        scales = [1, 1, 2] + ([int(rng.choice((1, 2)))] if d else [])
        deltas = [3, 1, 2] if index == 0 else [0, 2, int(rng.choice((1, 3)))]
        directions = list(witness_directions[index])
        kind = 'combined' if index == 0 else str(rng.choice(('combined', 'recolor', 'relocate')))
    else:
        m = int(rng.integers(1 if split == 'train' else 2, 5))
        n = int(rng.integers(max(2, m), 5))
        d = int(rng.integers(0, 3))
        if m + d < 3:
            d = 3 - m
        states = [0, 1] + list(map(int, rng.integers(2, size=m + d - 2)))
        scales = [1, 1] + list(map(int, rng.choice((1, 1, 2), size=m + d - 2)))
        deltas = list(map(int, rng.integers(4, size=m)))
        directions = list(map(int, rng.integers(4, size=m)))
        kind = str(rng.choice(('combined', 'recolor', 'relocate')))
    while len(directions) < n:
        directions.append(int(rng.integers(4)))
    large_scale = 3 if split == 'test' and large_test else 2
    if split == 'test':
        # Keep two unit witnesses, and enlarge a matching module. Extra unit
        # modules may supply references when there are only two matches.
        if m == 2:
            states = [int(rng.integers(2)), 0] + states[2:]
            states[2] = 1
            scales[0], scales[1], scales[2] = large_scale, 1, 1
        else:
            scales[2] = large_scale
    return dict(matches=m, sockets=n, distractors=d, states=states, scales=scales,
                deltas=deltas, directions=directions, kind=kind, large_scale=large_scale)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    large_test = bool(rng.random() < .35)
    order = list(map(int, rng.permutation(4)))
    witness_directions = (order[:3], [order[3], order[0], order[1]])
    result, seen, training_vocab = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            config = _configuration(rng, index, split, large_test, witness_directions)
            for _ in range(120):
                pair = _build_pair(rng, config, training_vocab if split == 'test' else set())
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    result[split].append(pair)
                    if split == 'train':
                        training_vocab.add(_vocabulary(_read_input(pair['input'])['templates']))
                    break
            else:
                raise AssertionError('bounded distinct-pair construction exhausted')
    validate_generated_puzzle(result)
    return result


def _features(info):
    a, b = info['templates']
    shared = set(a) & set(b)
    return dict(vocabulary=_vocabulary(info['templates']),
                core=tuple(sorted(info['modules'][0]['scaffold'])),
                kind=('combined' if set(a) != set(b) and any(a[p] != b[p] for p in shared)
                      else 'relocate' if set(a) != set(b) else 'recolor'),
                bbox_shift=_bounds(a) != _bounds(b),
                sides={m['socket']['q'] for m in info['matched']},
                turns={m['delta'] for m in info['matched']},
                odd_states={m['state'] for m in info['matched'] if m['delta'] % 2},
                scales={m['scale'] for m in info['matched']},
                counts=(len(info['sockets']), len(info['matched']), len(info['modules']) - len(info['matched'])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _read_input(inp)
            assert np.array_equal(_answer(info), out), 'wrong conditional docking result'
            assert not np.array_equal(inp, out)
            for socket in info['sockets']:
                assert all(int(out[p]) == int(inp[p]) for p in socket['points'])
            # Pixel scaling, complete scaffold preservation and template erasure
            # are checked by the input-derived full answer, not sample metadata.
            infos[split].append(info)
    f = [_features(i) for i in infos['train']]
    assert set.union(*(a['sides'] for a in f)) == set(range(4))
    assert set.union(*(a['turns'] for a in f)) == set(range(4))
    assert set.union(*(a['odd_states'] for a in f)) == {0, 1}, 'both exchange directions need witnesses'
    assert any(2 in a['scales'] for a in f), 'show scale-aware docking'
    assert any(m['scale'] > 1 and m['delta'] % 2
               for i in infos['train'] for m in i['matched']), 'show enlarged exchange'
    assert any(a['bbox_shift'] for a in f), 'show scaffold rather than box registration'
    assert any(a['counts'][2] > 0 for a in f), 'show color-based filtering'
    assert any(not np.array_equal(_answer(i, 'never'), _answer(i)) for i in infos['train']), 'B omission invisible'
    assert any(not np.array_equal(_answer(i, 'always'), _answer(i)) for i in infos['train']), 'parity shortcut invisible'
    training_vocab = {a['vocabulary'] for a in f}
    assert all(_features(i)['vocabulary'] not in training_vocab for i in infos['test']), 'test must use a fresh local design'


def self_test():
    """Bounded checks of 42 specified seeds, counts, ablations and variation."""
    shapes, cores, vocabularies, kinds, counts, scales, backgrounds = (set() for _ in range(7))
    signatures = set()
    edges = bbox_shifts = 0
    for seed in _SEEDS:
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(pair['input']) for split in p for pair in p[split]))
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _read_input(pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                f = _features(info)
                shapes.add(pair['input'].shape)
                cores.add(f['core'])
                vocabularies.add(f['vocabulary'])
                kinds.add(f['kind'])
                counts.add(f['counts'])
                scales.update(f['scales'])
                backgrounds.add(info['background'])
                bbox_shifts += f['bbox_shift']
                out = pair['output']
                edges += any(np.any(a != info['background']) for a in (out[0], out[-1], out[:, 0], out[:, -1]))
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(cores) >= 20 and len(vocabularies) >= 100
    assert kinds == {'recolor', 'relocate', 'combined'}
    assert len(counts) >= 10 and scales >= {1, 2, 3} and len(backgrounds) == 10
    assert bbox_shifts >= 42 and edges >= 5
    a, b = generate_puzzle(12345, 2, 1), generate_puzzle(12345, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    # A ablation: changing just the color-to-socket correspondence, while
    # retaining templates and scales, must redirect the two unit modules.
    info = _read_input(a['train'][0]['input'])
    units = [m for m in info['matched'] if m['scale'] == 1]
    left, right = units[:2]
    redirected = info['grid'].copy()
    for p in left['socket']['mouth']:
        redirected[p] = right['color']
    for p in right['socket']['mouth']:
        redirected[p] = left['color']
    assert not np.array_equal(solve(redirected), a['train'][0]['output'])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    info = _read_input(damaged['test'][0]['input'])
    mouth = min(info['sockets'][0]['mouth'])
    damaged['test'][0]['output'][mouth] = info['background']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('changed socket accepted')
    malformed = a['train'][0]['input'].copy()
    first_socket = _read_input(malformed)['sockets'][0]
    malformed[min(first_socket['mouth'])] = first_socket['wall']
    try:
        solve(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError('malformed mouth accepted')
    return dict(seeds=42, shapes=len(shapes), scaffolds=len(cores), state_designs=len(vocabularies),
                configurations=len(counts), scales=sorted(scales), edge_examples=edges)


if __name__ == '__main__':
    print(self_test())
