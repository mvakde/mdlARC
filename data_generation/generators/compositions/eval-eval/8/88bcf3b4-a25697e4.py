"""Dock a rigid core, then tension its flexible tether in the transported frame.

Only NumPy and the standard library are used. Prediction reads the input alone.
The obstacle participates in orientation selection, including reflection parity.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not 2 <= a <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(b) or not 1 <= b <= 4 or a <= b:
        raise ValueError("num_test must be in 1..4 and smaller than num_train")
    return int(a), int(b)


def _array(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not np.issubdtype(g.dtype, np.integer)
            or any(not 1 <= d <= 30 for d in g.shape)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid")
    return g


def _signature(g):
    return g.shape, np.asarray(g, dtype=np.int8).tobytes()


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _normal(cells):
    r, _, c, _ = _bbox(cells)
    return frozenset((y - r, x - c) for y, x in cells)


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _op(p, op):
    r, c = p
    return ((r, c), (c, -r), (-r, -c), (-c, r),
            (r, -c), (-c, -r), (-r, c), (c, r))[op]


def _transform(cells, op):
    return frozenset(_op(p, op) for p in cells)


def _canonical(cells):
    return min(tuple(sorted(_normal(_transform(cells, op)))) for op in range(8))


def _components(cells, directions=DIR4):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        comp, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    comp.add(p)
                    stack.append(p)
        result.append(frozenset(comp))
    return result


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _local(p, endpoint, f, l):
    r, c = p[0] - endpoint[0], p[1] - endpoint[1]
    return r * f[0] + c * f[1], r * l[0] + c * l[1]


def _global(p, endpoint, f, l):
    u, v = p
    return (endpoint[0] + u * f[0] + v * l[0],
            endpoint[1] + u * f[1] + v * l[1])


def _route(obstacle, length):
    """A's bounded convex, axis-parallel/45-degree silhouette grammar."""
    if not obstacle or min(u for u, v in obstacle) < 2:
        raise ValueError("obstacle is not beyond the dock")
    peak = max(v for u, v in obstacle)
    tangent = max(v - u for u, v in obstacle)
    if peak < 0 or not (peak == 0 or tangent == 0):
        raise ValueError("unsupported incoming tangent")
    first_peak = min(u for u, v in obstacle if v == peak)
    n = max(length, max(u for u, v in obstacle) + 4)
    profile = [1 if peak == 0 and u <= first_peak else
               max(v + 1 - abs(u - t) for t, v in obstacle)
               for u in range(n)]
    slopes = [b - a for a, b in zip(profile, profile[1:])]
    if profile[0] != 1 or any(s not in (-1, 0, 1) for s in slopes):
        raise ValueError("invalid route attachment or slope")
    if slopes != sorted(slopes, reverse=True):
        raise ValueError("nonconvex exposed route")
    path = tuple(enumerate(profile[:length]))
    if set(path) & set(obstacle):
        raise ValueError("route enters obstacle")
    return path


def _ports(core, slack):
    """Unique lateral contact; the rigid terminal spine defines forward."""
    contacts = [(p, (p[0] + dr, p[1] + dc)) for p in sorted(core)
                for dr, dc in DIR4 if (p[0] + dr, p[1] + dc) in slack]
    if len(contacts) != 1:
        return []
    endpoint, root = contacts[0]
    l = root[0] - endpoint[0], root[1] - endpoint[1]
    ports = []
    for f in ((-l[1], l[0]), (l[1], -l[0])):
        if (endpoint[0] - f[0], endpoint[1] - f[1]) not in core:
            continue
        cl = {_local(p, endpoint, f, l) for p in core}
        sl = {_local(p, endpoint, f, l) for p in slack}
        if (all(u <= 0 and v <= 0 for u, v in cl)
                and all(u <= 0 and v >= 1 for u, v in sl)):
            ports.append((endpoint, f, l))
    return ports


def _inspect(inp):
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("background must be the unique most frequent color")
    cells = set(map(tuple, np.argwhere(inp != bg)))
    keys, loose = [], set()
    for comp in _components(cells, DIR8):
        colors = sorted({int(inp[p]) for p in comp})
        if len(colors) == 1:
            loose.update(comp)
            continue
        if len(colors) != 2:
            raise ValueError("source keys must have exactly two colors")
        regions = tuple(frozenset(p for p in comp if inp[p] == color) for color in colors)
        roles = []
        for pi in (0, 1):
            core, slack = regions[pi], regions[1 - pi]
            if len(core) < 5 or len(slack) < 2 or len(_components(core)) != 1:
                continue
            if len(_components(slack, DIR8)) != 1:
                continue
            for endpoint, f, l in _ports(core, slack):
                roles.append(dict(core=core, slack=slack, endpoint=endpoint, f=f, l=l,
                                  colors=(colors[pi], colors[1 - pi])))
        if not roles:
            raise ValueError("key has no visible rigid/flexible attachment frame")
        keys.append(dict(cells=comp, roles=roles))
    if not keys or len(keys) > 2:
        raise ValueError("expected one or two dockable keys")
    mono = []
    for color in sorted({int(inp[p]) for p in loose}):
        mono.extend(dict(cells=c, color=color)
                    for c in _components({p for p in loose if inp[p] == color}))
    if len(mono) != 3 * len(keys):
        raise ValueError("each key needs two fragments and one obstacle")
    targets = []
    for i, a in enumerate(mono):
        for j in range(i + 1, len(mono)):
            b = mono[j]
            if a['color'] != b['color'] or min(len(a['cells']), len(b['cells'])) < 2:
                continue
            union = a['cells'] | b['cells']
            r0, r1, c0, c1 = _bbox(union)
            rect = frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
            if {p for p in rect if inp[p] != bg} != set(union):
                continue
            slot = rect - union
            if slot and len(_components(slot)) == 1:
                targets.append(dict(parts=(i, j), rectangle=rect, slot=slot))
    return bg, keys, mono, targets


def _predict(inp, swap_policy=(False, True)):
    bg, keys, mono, targets = _inspect(inp)
    fixed = frozenset(p for item in mono for p in item['cells'])
    candidates = []
    shape_fits = []
    for ki, key in enumerate(keys):
        moves, fits = [], []
        for role in key['roles']:
            for ti, target in enumerate(targets):
                sr, _, sc, _ = _bbox(target['slot'])
                for op in range(8):
                    core = _transform(role['core'], op)
                    cr, _, cc, _ = _bbox(core)
                    dr, dc = sr - cr, sc - cc
                    placed_core = _shift(core, dr, dc)
                    if placed_core != target['slot']:
                        continue
                    e = _op(role['endpoint'], op)
                    endpoint = e[0] + dr, e[1] + dc
                    f, l = _op(role['f'], op), _op(role['l'], op)
                    root = _global((0, 1), endpoint, f, l)
                    if not _inside(root, inp.shape) or root in fixed:
                        continue
                    base = dict(key=ki, target=ti, parts=target['parts'], role=role, op=op,
                                core=placed_core, endpoint=endpoint, f=f, l=l)
                    fits.append(base)
                    for oi, obstacle in enumerate(mono):
                        if oi in target['parts']:
                            continue
                        local_obs = {_local(p, endpoint, f, l) for p in obstacle['cells']}
                        try:
                            local_path = _route(local_obs, len(role['slack']))
                        except ValueError:
                            continue
                        path = tuple(_global(p, endpoint, f, l) for p in local_path)
                        visible = frozenset(p for p in path if _inside(p, inp.shape))
                        full = placed_core | visible
                        if full & fixed:
                            continue
                        colors = role['colors'][::-1] if swap_policy[op >= 4] else role['colors']
                        paint = {p: colors[0] for p in placed_core}
                        paint.update({p: colors[1] for p in visible})
                        moves.append(dict(base, obstacle=oi, used=frozenset((*target['parts'], oi)),
                                          path=path, visible=visible, local_obstacle=local_obs,
                                          full=full, paint=paint))
        candidates.append(moves)
        shape_fits.append(fits)
    solutions = {}

    def search(i, used, occupied, chosen):
        if len(solutions) > 1:
            return
        if i == len(keys):
            if len(used) == len(mono):
                sig = tuple(sorted((p, color) for m in chosen for p, color in m['paint'].items()))
                solutions.setdefault(sig, tuple(chosen))
            return
        for move in candidates[i]:
            if move['used'] & used or move['full'] & occupied:
                continue
            search(i + 1, used | move['used'], occupied | move['full'], chosen + [move])

    search(0, frozenset(), frozenset(), [])
    if len(solutions) != 1:
        raise ValueError("docking and obstacle constraints must give one colored answer")
    moves = next(iter(solutions.values()))
    out = np.full(inp.shape, bg, dtype=np.int8)
    for item in mono:
        for p in item['cells']:
            out[p] = item['color']
    for move in moves:
        for p, color in move['paint'].items():
            out[p] = color
    return out, dict(background=bg, keys=keys, mono=mono, targets=targets,
                     moves=moves, shape_fits=shape_fits)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible geometry, local color roles and the fixed family rule."""
    return _predict(_array(input_grid))[0]


def _rigid_tether_ablation(inp):
    """Counterfactual B-only behavior: move the slack shape with the core."""
    out, info = _predict(inp)
    bg = info['background']
    # Rebuild only fixed material, rigid cores, and rigidly transported coils.
    out[:] = bg
    for item in info['mono']:
        for p in item['cells']:
            out[p] = item['color']
    for m in info['moves']:
        colors = m['role']['colors'][::-1] if m['op'] >= 4 else m['role']['colors']
        for p in m['core']:
            out[p] = colors[0]
        for p in m['role']['slack']:
            uv = _local(p, m['role']['endpoint'], m['role']['f'], m['role']['l'])
            q = _global(uv, m['endpoint'], m['f'], m['l'])
            if _inside(q, inp.shape) and out[q] == bg:
                out[q] = colors[1]
    return out


def _core(rng, small=True, symmetric=False):
    for _ in range(100):
        h = int(rng.integers(4 if small else 6, 6 if small else 8))
        w = h if symmetric else int(rng.integers(4 if small else 6, 6 if small else 8))
        rect = {(r, c) for r in range(1 - h, 1) for c in range(1 - w, 1)}
        if symmetric:
            band = 2 if h >= 6 and rng.random() < .4 else 1
            core = {p for p in rect if abs(p[0] - p[1]) <= band}
        else:
            r, c = 0, 0
            core = {(r, c), (-1, 0)}
            r = -1
            while r > 1 - h or c > 1 - w:
                choices = []
                if r > 1 - h:
                    choices.append((r - 1, c))
                if c > 1 - w:
                    choices.append((r, c - 1))
                r, c = choices[int(rng.integers(len(choices)))]
                core.add((r, c))
            for _ in range(int(rng.integers(0, 4))):
                frontier = sorted({(r + dr, c + dc) for r, c in core for dr, dc in DIR4}
                                  & rect - core)
                if frontier:
                    core.add(frontier[int(rng.integers(len(frontier)))])
        fragments = rect - core
        parts = _components(fragments)
        if len(parts) == 2 and min(map(len, parts)) >= 2 and _bbox(fragments) == _bbox(rect):
            return frozenset(core), frozenset(fragments)
    # Procedural symmetric separator; no stored complete examples.
    low = -3 if small else -5
    rect = {(r, c) for r in range(low, 1) for c in range(low, 1)}
    core = {p for p in rect if abs(p[0] - p[1]) <= 1}
    return frozenset(core), frozenset(rect - core)


def _obstacle(rng, small=True, approach=None, witness=False):
    for _ in range(100):
        near = int(rng.integers(2, 4 if small else 6))
        kind = 1 if witness else int(rng.integers(6))
        size = int(rng.integers(2, 4 if small else 5))
        if kind == 0:
            raw = {(0, 0)}
        elif kind == 1:
            raw = {(u, 0) for u in range(size)}
        elif kind == 2:
            width = int(rng.integers(2, 4))
            raw = {(u, v) for u in range(size) for v in range(width)}
        elif kind == 3:
            raw = {(u, v) for u in range(2 * size - 1)
                   for v in range(size) if v <= size - 1 - abs(u - size + 1)}
        elif kind == 4:
            raw = {(u, v) for u in range(size) for v in range(size - u)}
        else:
            # A narrow descending stepped band, with 4-connected joins.
            raw = {(u, size - 1 - u - t) for u in range(size) for t in (0, 1)}
        style = approach or ('parallel' if rng.random() < .35 else 'diagonal')
        offset = (-max(v for u, v in raw) if style == 'parallel'
                  else near - max(v - u for u, v in raw))
        obs = frozenset((u + near, v + offset) for u, v in raw)
        try:
            _route(obs, 2)
            return obs
        except ValueError:
            pass
    raise AssertionError("convex obstacle construction exhausted")


def _slack(rng, n, style=None, diagonal=False):
    # Only the root touches the core orthogonally. All other cells are at v>=2.
    style = int(rng.integers(4)) if style is None else style
    width = int(rng.integers(3, 7))
    depth = max(3, (n - 2) // (width - 1) + 2)
    cells = {(-1, 2)}
    if style == 1:
        u, v = -1, 2
        while len(cells) < n - 1:
            opts = [(u - 1, v), (u, v + 1), (u - 1, v + 1)]
            opts = [p for p in opts if -depth <= p[0] <= -1 and 2 <= p[1] <= width]
            if not opts:
                break
            u, v = opts[int(rng.integers(len(opts)))]
            cells.add((u, v))
    elif style == 2:
        cells = {(-1 - k // (width - 1), 2 + k % (width - 1)) for k in range(n - 1)}
    while len(cells) < n - 1:
        frontier = sorted({(r + dr, c + dc) for r, c in cells
                           for dr, dc in (DIR4 if style == 2 else DIR8)
                           if -depth <= r + dr <= -1 and 2 <= c + dc <= width} - cells)
        weights = np.array([sum((r + dr, c + dc) in cells for dr, dc in DIR8) + 1
                            for r, c in frontier], dtype=float)
        weights = weights ** (2 if style == 0 else -2 if style == 3 else 0)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    cells.add((0, 1))
    if not diagonal and rng.random() < .4 and n >= 3:
        # Change the root-to-coil link to an orthogonal one without changing n.
        cells.remove((0, 1))
        # Keep the root fixed; add (0,2) by deleting a safe remote leaf.
        removable = [p for p in sorted(cells) if p != (-1, 2)
                     and len(_components(cells - {p}, DIR8)) == 1]
        if removable:
            cells.remove(removable[int(rng.integers(len(removable)))])
            cells.add((0, 2))
        cells.add((0, 1))
    return frozenset(cells)


def _system(rng, parity, small=True, lesson=None, reused=None, simple=False):
    core, fragments = reused if reused is not None else _core(rng, small, rng.random() < .5)
    if simple:
        if reused is None:
            low = -3 if small else -5
            rect = {(r, c) for r in range(low, 1) for c in range(low, 1)}
            core = frozenset(p for p in rect if abs(p[0] - p[1]) <= 1)
            fragments = frozenset(rect - core)
        obs = frozenset({(3, 3), (4, 3)}) if lesson != 'clipped' else frozenset({(2, 0), (3, 0)})
    else:
        obs = _obstacle(rng, small, 'diagonal' if lesson == 'visible' else
                        'parallel' if lesson == 'clipped' else None, witness=lesson is not None)
    far, near = max(u for u, v in obs), min(u for u, v in obs)
    if lesson == 'visible':
        n = far + int(rng.integers(3, 5))
    elif lesson == 'clipped':
        n = far + int(rng.integers(7, 10))
    else:
        stop = int(rng.integers(4))
        n = (int(rng.integers(2, near + 1)) if stop == 0 else
             int(rng.integers(near + 1, far + 2)) if stop == 1 else
             far + int(rng.integers(2, 5)) if stop == 2 else
             far + int(rng.integers(5, 8)))
    if simple:
        n = 12 if lesson == 'clipped' else 8
    slack = _slack(rng, n, diagonal=lesson == 'visible')
    path = _route(obs, n)
    clipped = lesson == 'clipped' or (lesson is None and n > far + 2 and rng.random() < .2)
    cut = far + int(rng.integers(1, 3)) if clipped else n - 1
    source_op = int(rng.integers(8))
    dest_op = int(rng.integers(4)) + 4 * ((source_op >= 4) ^ bool(parity))
    src = dict(core=_transform(core, source_op), slack=_transform(slack, source_op))
    dst = dict(core=_transform(core, dest_op), fragments=_transform(fragments, dest_op),
               obstacle=_transform(obs, dest_op), path=_transform(path, dest_op))
    visible = _transform(path[:cut + 1], dest_op)
    # Bounds follow the actual clipping edge, even if the path's departure is
    # negative laterally. Invisible material never influences placement bounds.
    src_cells = src['core'] | src['slack']
    dst_cells = dst['core'] | dst['fragments'] | dst['obstacle'] | visible
    sr0, sr1, sc0, sc1 = _bbox(src_cells)
    dr0, dr1, dc0, dc1 = _bbox(dst_cells)
    src = {k: _shift(v, -sr0, -sc0) for k, v in src.items()}
    dst = {k: _shift(v, -dr0, -dc0) for k, v in dst.items()}
    f = _op((1, 0), dest_op)
    return dict(src=src, dst=dst, source_shape=(sr1 - sr0 + 1, sc1 - sc0 + 1),
                dest_shape=(dr1 - dr0 + 1, dc1 - dc0 + 1), parity=parity,
                edge=f if clipped else None, canonical_core=_canonical(core))


def _pack(rng, shapes, edges, shape):
    height, width = shape
    placed, rectangles = [None] * len(shapes), []
    order = sorted(range(len(shapes)), key=lambda i: (edges[i] is None, -np.prod(shapes[i])))
    for i in order:
        h, w = shapes[i]
        if h > height or w > width:
            return None
        options = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)
                   if (edges[i] is None or
                       (edges[i] == (1, 0) and r + h == height) or
                       (edges[i] == (-1, 0) and r == 0) or
                       (edges[i] == (0, 1) and c + w == width) or
                       (edges[i] == (0, -1) and c == 0))]
        rng.shuffle(options)
        for r, c in options:
            rect = (r, r + h - 1, c, c + w - 1)
            if any(not (rect[1] + 1 < a or b + 1 < rect[0] or
                        rect[3] + 1 < x or y + 1 < rect[2]) for a, b, x, y in rectangles):
                continue
            placed[i] = (r, c)
            rectangles.append(rect)
            break
        else:
            return None
    return placed


def _make_pair(rng, count, parities, small=True, lesson=None, reused=None, palette=None):
    for attempt in range(160):
        systems = [_system(rng, parities[j], small or count > 1,
                           lesson if j == 0 else None, reused if j == 0 else None,
                           simple=attempt >= 120) for j in range(count)]
        if count > 1 and len({s['canonical_core'] for s in systems}) < count:
            # The safe branch varies the second core with an asymmetric cut.
            if attempt >= 120:
                systems[1] = _system(rng, parities[1], True, simple=False)
            if len({s['canonical_core'] for s in systems}) < count:
                continue
        shapes = [shape for s in systems for shape in (s['dest_shape'], s['source_shape'])]
        edges = [edge for s in systems for edge in (s['edge'], None)]
        origins = None
        for trial in range(8):
            shape = ((30, 30) if trial >= 5 or attempt >= 120 else
                     (int(rng.integers(14, 31)), int(rng.integers(14, 31))))
            origins = _pack(rng, shapes, edges, shape)
            if origins is not None:
                break
        if origins is None:
            continue
        colors = list(map(int, rng.permutation(10))) if palette is None else list(palette)
        bg = colors[0]
        inp = np.full(shape, bg, dtype=np.int8)
        out = inp.copy()
        for j, s in enumerate(systems):
            # Same palettes across systems sometimes prevent color matching;
            # independent local palettes sometimes reuse a color across roles.
            local = colors[1:5] if j == 0 or rng.random() < .5 else list(map(int, rng.choice(colors[1:], 4, replace=False)))
            target_color, obstacle_color, core_color, strand_color = local
            dr, dc = origins[2 * j]
            sr, sc = origins[2 * j + 1]
            for role, color in (('fragments', target_color), ('obstacle', obstacle_color)):
                for p in _shift(s['dst'][role], dr, dc):
                    inp[p] = out[p] = color
            for role, color in (('core', core_color), ('slack', strand_color)):
                for p in _shift(s['src'][role], sr, sc):
                    inp[p] = color
            cc, tc = (strand_color, core_color) if s['parity'] else (core_color, strand_color)
            for role, color in (('core', cc), ('path', tc)):
                for p in _shift(s['dst'][role], dr, dc):
                    if _inside(p, shape):
                        if out[p] != bg:
                            raise AssertionError("construction overlap")
                        out[p] = color
        # Retain chosen clipping edges; crop other unused margins with padding.
        used = set(map(tuple, np.argwhere((inp != bg) | (out != bg))))
        r0, r1, c0, c1 = _bbox(used)
        if rng.random() < .8:
            r0, c0 = max(0, r0 - int(rng.integers(4))), max(0, c0 - int(rng.integers(4)))
            r1, c1 = min(shape[0] - 1, r1 + int(rng.integers(4))), min(shape[1] - 1, c1 + int(rng.integers(4)))
            inp, out = inp[r0:r1 + 1, c0:c1 + 1].copy(), out[r0:r1 + 1, c0:c1 + 1].copy()
        try:
            pred, analysis = _predict(inp)
            if not np.array_equal(pred, out):
                continue
            if sorted(m['op'] >= 4 for m in analysis['moves']) != sorted(map(bool, parities)):
                continue
            if lesson:
                feature = _features(inp)
                if not any(a['obstacle_resolves_parity'] for a in feature['assemblies']):
                    continue
                if lesson == 'visible' and not any(a['full_departure'] and a['diagonal_slack'] and not a['clipped'] for a in feature['assemblies']):
                    continue
                if lesson == 'clipped' and not any(a['clipped'] for a in feature['assemblies']):
                    continue
            return dict(input=inp, output=out)
        except ValueError:
            continue
    raise AssertionError("bounded docking construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    stable = rng.random() < .25
    palette = tuple(map(int, rng.permutation(10))) if stable else None
    shared_core = _core(rng, True, symmetric=True)
    puzzle = dict(train=[], test=[])
    seen = set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            lesson = ('visible' if i == 0 else 'clipped') if split == 'train' and i < 2 else None
            n = (1 if lesson or regime == 1 or split == 'train' and regime == 0
                 else 2 if split == 'test' else int(rng.integers(1, 3)))
            parities = ([i] if lesson else [0, 1] if regime == 2 and split == 'test'
                        else [int(rng.integers(2))] * n if regime == 2
                        else list(map(int, rng.integers(2, size=n))))
            for _ in range(20):
                pair = _make_pair(rng, n, parities, small=split == 'train' or regime != 1,
                                  lesson=lesson, reused=shared_core if lesson else None, palette=palette)
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct input construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp):
    _, a = _predict(inp)
    rows = []
    centers = [tuple(np.mean(list(m['core']), axis=0)) for m in a['moves']]
    for i, m in enumerate(a['moves']):
        role = m['role']
        obs = m['local_obstacle']
        far = max(u for u, v in obs)
        same_target = [b for b in a['shape_fits'][m['key']] if b['target'] == m['target']]
        root_frames = {(b['endpoint'], b['f'], b['l'], b['op'] >= 4) for b in same_target}
        kcenter = tuple(np.mean(list(a['keys'][m['key']]['cells']), axis=0))
        distance = lambda p: abs(p[0] - kcenter[0]) + abs(p[1] - kcenter[1])
        lo, hi, left, right = _bbox(obs)
        rows.append(dict(parity=int(m['op'] >= 4), op=m['op'], f=m['f'], l=m['l'],
                         core_size=len(role['core']), length=len(role['slack']),
                         core_shape=_canonical(role['core']), obstacle_shape=_canonical(obs),
                         obstacle_kind=('point' if len(obs) == 1 else 'bar' if lo == hi or left == right else
                                        'rectangle' if len(obs) == (hi - lo + 1) * (right - left + 1) else 'stepped'),
                         approach='parallel' if max(v for u, v in obs) == 0 else 'diagonal',
                         diagonal_slack=len(_components(role['slack'])) > 1,
                         clipped=len(m['visible']) < len(role['slack']),
                         lost=len(role['slack']) - len(m['visible']),
                         full_departure=len(role['slack']) > far + 2,
                         stop='approach' if len(role['slack']) <= lo else 'contour' if len(role['slack']) <= far + 1 else 'departure',
                         obstacle_resolves_parity={b[3] for b in root_frames} == {False, True},
                         size_relation=int(np.sign(len(role['core']) - len(role['slack']))),
                         colors=role['colors'], root_frames=len(root_frames),
                         nearer_wrong=any(distance(p) < distance(centers[i]) for j, p in enumerate(centers) if i != j)))
    return dict(shape=tuple(inp.shape), background=a['background'], count=len(rows), assemblies=rows)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _array(pair['input']), _array(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            sig = _signature(inp)
            assert sig not in seen, "duplicate input"
            seen.add(sig)
            predicted, a = _predict(inp)
            assert np.array_equal(predicted, out), "input-only reconstruction mismatch"
            bg = a['background']
            source = frozenset(p for k in a['keys'] for p in k['cells'])
            final = frozenset(p for m in a['moves'] for p in m['full'])
            for p in source - final:
                assert out[p] == bg, "old flexible/rigid object not erased"
            for item in a['mono']:
                assert all(out[p] == item['color'] for p in item['cells']), "fixed material changed"
            for m in a['moves']:
                target = a['targets'][m['target']]
                assert m['core'] == target['slot']
                assert _bbox(target['rectangle'] - target['slot']) == _bbox(target['rectangle'])
                assert len(_components(target['rectangle'] - target['slot'])) == 2
                assert len(m['path']) == len(m['role']['slack'])
                assert len(set(m['path'])) == len(m['path'])
                assert m['path'][0] == _global((0, 1), m['endpoint'], m['f'], m['l'])
                assert m['visible'] == frozenset(p for p in m['path'] if _inside(p, inp.shape))
                assert m['visible'].isdisjoint(target['rectangle'])
                assert len(m['role']['colors']) == 2 and bg not in m['role']['colors']
            if split == 'train':
                training.extend(_features(inp)['assemblies'])
    assert {a['parity'] for a in training} == {0, 1}, "missing parity contrast"
    assert {a['approach'] for a in training} == {'parallel', 'diagonal'}
    assert any(a['obstacle_resolves_parity'] for a in training), "A does not constrain B's fit"
    assert any(a['clipped'] for a in training), "missing boundary demonstration"
    assert any(a['full_departure'] and not a['clipped'] and a['diagonal_slack'] for a in training)
    assert len({a['length'] for a in training}) >= 2, "length must vary independently of the shared core"
    by_core = {}
    for a in training:
        by_core.setdefault(a['core_shape'], set()).add(a['length'])
    assert any(len(values) >= 2 for values in by_core.values()), "missing shared-core length contrast"
    assert any(not np.array_equal(_rigid_tether_ablation(_array(p['input'])), p['output'])
               for p in puzzle['train']), "rigidly moving the tether is an unchallenged shortcut"
    # Labels distinguish all four simple reflection-to-swap tables. Geometric
    # fitting remains input-only; labels are used only to audit demonstrations.
    compatible = []
    for policy in ((False, False), (False, True), (True, False), (True, True)):
        try:
            if all(np.array_equal(_predict(_array(p['input']), policy)[0], p['output']) for p in puzzle['train']):
                compatible.append(policy)
        except ValueError:
            pass
    assert compatible == [(False, True)], "color convention not distinguished"


def self_test():
    # Small mathematical regressions, not a second purported independent solver.
    assert _route({(2, 0), (3, 0)}, 7) == ((0, 1), (1, 1), (2, 1), (3, 1), (4, 0), (5, -1), (6, -2))
    assert _route({(2, 2), (3, 2)}, 7) == ((0, 1), (1, 2), (2, 3), (3, 3), (4, 2), (5, 1), (6, 0))
    stepped = {(2, 2), (2, 1), (3, 1), (3, 0), (4, 0), (4, -1)}
    box = {(u, v) for u in range(2, 5) for v in range(-1, 3)}
    assert _route(stepped, 7) != _route(box, 7), "silhouette cannot be replaced by its box"
    summary = Counter()
    dimensions, cores, obstacles, palettes, lengths, directions, structures = set(), set(), set(), set(), set(), set(), set()
    puzzles = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.add(tuple(_signature(q['input']) for s in ('train', 'test') for q in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                before = pair['input'].copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input']), "solve mutated its input"
                f = _features(before)
                dimensions.add(f['shape'])
                palettes.add(f['background'])
                summary['single' if f['count'] == 1 else 'multiple'] += 1
                for a in f['assemblies']:
                    cores.add(a['core_shape'])
                    obstacles.add(a['obstacle_shape'])
                    lengths.add(a['length'])
                    directions.add((a['f'], a['l']))
                    structures.add((a['core_shape'], a['obstacle_shape'], a['length'], a['f'], a['l']))
                    for metric in ('diagonal_slack', 'clipped', 'obstacle_resolves_parity', 'nearer_wrong'):
                        summary[metric] += int(a[metric])
                    summary[a['approach']] += 1
                    summary[a['stop']] += 1
                    summary['parity_' + str(a['parity'])] += 1
                    summary['relation_' + str(a['size_relation'])] += 1
                    summary[a['obstacle_kind']] += 1
    assert len(puzzles) == 42 and len(dimensions) >= 40 and len(cores) >= 20
    assert len(obstacles) >= 10 and len(structures) >= 150 and len(palettes) == 10
    assert len(lengths) >= 10 and len(directions) == 8
    assert all(summary[k] > 0 for k in ('single', 'multiple', 'diagonal_slack', 'clipped',
                                      'obstacle_resolves_parity', 'nearer_wrong', 'parallel', 'diagonal',
                                      'approach', 'contour', 'departure', 'parity_0', 'parity_1',
                                      'relation_-1', 'relation_0', 'relation_1', 'point', 'bar', 'rectangle', 'stepped'))
    a, b = generate_puzzle(71, 6, 3), generate_puzzle(71, 6, 3)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
    validate_generated_puzzle({'train': list(reversed(a['train'])), 'test': a['test']})
    for nt, nq in ((2, 1), (10, 4), (10, 1)):
        p = generate_puzzle(197, nt, nq)
        assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Both ablations alter the answer: rigidly moving the slack is not tension;
    # omitting reflection exchange changes the reflected core and route colors.
    assert any(not np.array_equal(_rigid_tether_ablation(p['input']), p['output']) for p in a['train'])
    for pair in a['train']:
        out, info = _predict(pair['input'])
        if any(m['op'] >= 4 for m in info['moves']):
            assert not np.array_equal(out, _predict(pair['input'], (False, False))[0])
            break
    else:
        raise AssertionError("missing reflection ablation witness")
    broken = {s: [{k: v.copy() for k, v in q.items()} for q in a[s]] for s in a}
    _, info = _predict(broken['test'][0]['input'])
    point = min(info['mono'][0]['cells'])
    broken['test'][0]['output'][point] = info['background']
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted erased fixed material")
    # A diagonal point cannot distinguish the transposed frames of a symmetric
    # band core. Both placements are legal and yield different colors/routes.
    ambiguous = np.zeros((28, 28), dtype=np.int8)
    rectangle = {(r, c) for r in range(-3, 1) for c in range(-3, 1)}
    core = {p for p in rectangle if abs(p[0] - p[1]) <= 1}
    coil = {(0, 1), (-1, 2), (-2, 2), (-2, 3), (-2, 4), (-3, 4)}
    for p in _shift(rectangle - core, 14, 12):
        ambiguous[p] = 2
    ambiguous[16, 14] = 5
    for cells, color in ((core, 3), (coil, 4)):
        for p in _shift(cells, 5, 21):
            ambiguous[p] = color
    try:
        solve(ambiguous)
    except ValueError as exc:
        assert 'one colored answer' in str(exc)
    else:
        raise AssertionError("ambiguous obstacle orientation accepted")
    Path('scratch').mkdir(exist_ok=True)
    metrics = dict(seeds=42, dimensions=len(dimensions), core_shapes=len(cores), obstacle_shapes=len(obstacles),
                   combined_structures=len(structures), backgrounds=len(palettes), lengths=sorted(lengths),
                   attachment_frames=len(directions), counts=dict(summary), independent_semantic_check=False)
    Path('scratch/self_test_metrics.json').write_text(json.dumps(metrics, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
