"""Taut embroidery: surface-selected stamps spend a unary material budget.

Only NumPy and the Python standard library are used. All semantic parameters
are visible in the input; solve has no access to the generator's RNG or state.
"""
from __future__ import annotations

import copy
import json
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _connected(cells, directions=D8):
    if not cells:
        return False
    seen = {min(cells)}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in directions:
            q = r + dr, c + dc
            if q in cells and q not in seen:
                seen.add(q)
                todo.append(q)
    return seen == cells


def _global(p, end, forward, lateral):
    u, v = p
    return (end[0] + u * forward[0] + v * lateral[0],
            end[1] + u * forward[1] + v * lateral[1])


def _local(p, end, forward, lateral):
    dr, dc = p[0] - end[0], p[1] - end[1]
    return dr * forward[0] + dc * forward[1], dr * lateral[0] + dc * lateral[1]


def _decode(grid):
    """Read a fully framed library and the separately framed macro-scene."""
    g = _grid(grid)
    sep = int(g[0, 0])
    assert np.all(g[0] == sep)
    full = [r for r in range(1, g.shape[0]) if np.all(g[r] == sep)]
    assert full
    n = full[0] - 1
    assert n in (2, 3)
    assert n + 5 < g.shape[0]
    assert np.all(g[n + 1] == sep) and np.all(g[n + 3] == sep)
    bg = int(g[n + 2, 1])  # The swatch is centered, leaving this slot blank.
    assert bg != sep
    templates, targets, order = {}, {}, []
    j = 0
    while (j + 1) * (n + 1) < g.shape[1]:
        left, right = j * (n + 1), (j + 1) * (n + 1)
        if not (np.all(g[1:n + 1, left] == sep)
                and np.all(g[1:n + 1, right] == sep)):
            break
        panel = g[1:n + 1, left + 1:right]
        colors = np.unique(panel[panel != bg])
        assert len(colors) == 1
        identity = int(colors[0])
        assert identity != sep and identity not in templates
        templates[identity] = panel == identity
        target = int(g[n + 2, left + 1 + n // 2])
        assert target != bg
        targets[identity] = target
        order.append(identity)
        j += 1
    assert 2 <= j <= 3
    library_width = j * (n + 1) + 1
    assert np.all(g[1:n + 1, library_width:] == bg)
    expected_key = np.full(g.shape[1], bg, dtype=np.int8)
    for i, identity in enumerate(order):
        expected_key[i * (n + 1)] = sep
        expected_key[i * (n + 1) + 1 + n // 2] = targets[identity]
    expected_key[j * (n + 1)] = sep
    assert np.array_equal(expected_key, g[n + 2])
    top = n + 4
    assert g[top, 0] == sep
    stop = 0
    while stop < g.shape[1] and g[top, stop] == sep:
        stop += 1
    w = stop - 2
    assert 5 <= w <= 15 and np.all(g[top, stop:] == bg)
    assert np.all(g[-1, :stop] == sep) and np.all(g[-1, stop:] == bg)
    assert np.all(g[top + 1:-1, 0] == sep)
    assert np.all(g[top + 1:-1, stop - 1] == sep)
    assert np.all(g[top + 1:-1, stop:] == bg)
    scene = g[top + 1:-1, 1:stop - 1].copy()
    h = scene.shape[0]
    assert 5 <= h <= 15 and h * n <= 30 and w * n <= 30
    present = set(map(int, np.unique(scene)))
    assert {bg, sep, *order} <= present
    extra = present - {bg, sep, *order}
    assert len(extra) == 1
    stock = extra.pop()
    support = {tuple(map(int, q)) for q in np.argwhere(scene == sep)}
    slack = {tuple(map(int, q)) for q in np.argwhere(scene == stock)}
    obstacle = {tuple(map(int, p)): int(scene[tuple(p)]) for p in np.argwhere(
        np.isin(scene, order))}
    assert len(support) >= 3 and _connected(support, D4)
    assert len(slack) >= 2 and _connected(slack)
    assert _connected(set(obstacle))
    if len({r for r, c in support}) == 1:
        axis = (0, 1)
    else:
        assert len({c for r, c in support}) == 1
        axis = (1, 0)
    # Unique support-relative frame; stock lives behind its attachment.
    candidates = []
    for end, f in ((max(support), axis),
                   (min(support), (-axis[0], -axis[1]))):
        for l in ((f[1], -f[0]), (-f[1], f[0])):
            s = {_local(p, end, f, l) for p in slack}
            o = {_local(p, end, f, l): color for p, color in obstacle.items()}
            if ((0, 1) in s and all(u <= 0 and v >= 1 for u, v in s)
                    and all(u >= 1 for u, v in o)):
                try:
                    _surface(o, 1)
                except AssertionError:
                    continue
                candidates.append((end, f, l, s, o))
    assert len(candidates) == 1, 'ambiguous attachment or winding side'
    end, f, l, local_slack, local_obstacle = candidates[0]
    return dict(n=n, bg=bg, sep=sep, stock=stock, templates=templates,
                targets=targets, order=order, scene=scene, support=support,
                slack=slack, obstacle=obstacle, end=end, f=f, l=l,
                local_slack=local_slack, local_obstacle=local_obstacle)


def _surface(obstacle, count):
    """Return (u,v,identity) on B's exposed digital contour and its provenance.

    A cell (t,z) contributes z+1-|u-t|. Select the greatest contribution,
    then the smallest axial distance, then smaller t. Only an exposed cell
    can win. Parallel incoming segments inherit the first highest surface.
    """
    roof = {}
    for (u, v), color in obstacle.items():
        if u not in roof or v > roof[u][0]:
            roof[u] = v, color
    near, far = min(roof), max(roof)
    assert near >= 1 and set(roof) == set(range(near, far + 1))
    # Every slice is solid and monochrome, but depth may vary independently.
    for u, (v, color) in roof.items():
        cells = [(z, col) for (t, z), col in obstacle.items() if t == u]
        assert {col for z, col in cells} == {color}
        zs = [z for z, col in cells]
        assert set(zs) == set(range(min(zs), v + 1))
    peak = max(v for v, color in roof.values())
    tangent = max(v - u for u, (v, color) in roof.items())
    assert peak == 0 or tangent == 0
    first_peak = min(u for u, (v, color) in roof.items() if v == peak)
    route = []
    for u in range(max(count, far + 4)):
        if peak == 0 and u <= first_peak:
            v, color = 1, roof[first_peak][1]
        else:
            t = max(roof, key=lambda t: (roof[t][0] + 1 - abs(u - t),
                                        -abs(u - t), -t))
            v = roof[t][0] + 1 - abs(u - t)
            color = roof[t][1]
        route.append((u, v, color))
    assert route[0][:2] == (0, 1)
    slopes = [b[1] - a[1] for a, b in zip(route, route[1:])]
    assert all(x in (-1, 0, 1) for x in slopes)
    assert slopes == sorted(slopes, reverse=True), 'unsupported exposed concavity'
    assert not {p[:2] for p in route} & set(obstacle)
    return route[:count]


def _stitches(p, *, unit_cost=False, constant_identity=False, positional_identity=False,
              first_cost=False, straight=False, box=False, budget=None):
    remaining = len(p['slack']) if budget is None else budget
    obstacle = p['local_obstacle']
    if box:
        us, vs = zip(*obstacle)
        # Deliberately wrong bbox alternative, preserving nearest axial label.
        roof_labels = {u: max((v, col) for (t, v), col in obstacle.items()
                              if t == u)[1] for u in set(us)}
        obstacle = {(u, v): roof_labels[u] for u in range(min(us), max(us) + 1)
                    for v in range(min(vs), max(vs) + 1)}
    route = _surface(obstacle, remaining + 1)
    result = []
    positional = {}
    initial_cost = int(p['templates'][route[0][2]].sum())
    for u, v, color in route:
        if remaining == 0:
            break
        if constant_identity:
            color = p['order'][0]
        if positional_identity:
            if color not in positional:
                positional[color] = p['order'][len(positional)]
            color = positional[color]
        cost = 1 if unit_cost else int(p['templates'][color].sum())
        if first_cost:
            cost = initial_cost
        if remaining < cost:
            # Counterfactual renderings can stop before an incomplete tile.
            # In the real family all material fits complete motifs exactly.
            assert unit_cost or constant_identity or positional_identity or first_cost or box or budget is not None
            break
        if straight:
            v = 1
        result.append((_global((u, v), p['end'], p['f'], p['l']), color))
        remaining -= cost
    if not (unit_cost or constant_identity or positional_identity or first_cost or box or budget is not None):
        assert remaining == 0, 'budget does not end on a complete motif'
    return result


def _render(p, stitches, *, motif_transform=0, ignore_key=False):
    n, bg, scene = p['n'], p['bg'], p['scene']
    base = scene.copy()
    base[base == p['stock']] = bg
    out = np.repeat(np.repeat(base, n, axis=0), n, axis=1)
    h, w = scene.shape
    for (r, c), color in stitches:
        if not (0 <= r < h and 0 <= c < w):
            continue
        mask = p['templates'][color]
        if motif_transform >= 4:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, motif_transform % 4)
        target = color if ignore_key else p['targets'][color]
        out[r*n:(r+1)*n, c*n:(c+1)*n] = np.where(mask, target, bg)
    return out.astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve a framed, fully visible family member from this input alone."""
    p = _decode(input_grid)
    stitches = _stitches(p)
    h, w = p['scene'].shape
    assert all(0 <= r < h and 0 <= c < w for (r, c), col in stitches)
    assert all(p['scene'][q] in (p['bg'], p['stock']) for q, col in stitches)
    return _render(p, stitches)


def _motifs(rng, n, labels):
    """Vary area as well as shape; no mandatory geometric template."""
    areas = rng.choice(np.arange(1 if n == 2 else 2, 5 if n == 2 else 6),
                       size=len(labels), replace=False)
    result = {}
    for label, area in zip(labels, areas):
        mask = np.zeros((n, n), dtype=bool)
        style = int(rng.integers(4))
        if style == 0:
            positions = rng.permutation(n * n)
        elif style == 1:
            # Contiguous row-major pieces, with a random rigid presentation.
            positions = np.arange(n * n)
        elif style == 2:
            coords = [(r, c) for r in range(n) for c in range(n)]
            positions = [r*n+c for r, c in sorted(coords,
                         key=lambda q: (abs(q[0]-q[1]), q))]
        else:
            coords = [(r, c) for r in range(n) for c in range(n)]
            positions = [r*n+c for r, c in sorted(coords,
                         key=lambda q: (min(q[0], q[1], n-1-q[0], n-1-q[1]), q))]
        mask.flat[positions[:int(area)]] = True
        mask = np.rot90(mask, int(rng.integers(4)))
        if rng.random() < .5:
            mask = np.fliplr(mask)
        result[int(label)] = mask.copy()
    return result


def _slack(rng, count, depth, width, diagonal):
    cells = {(0, 1)}
    if diagonal:
        # The attachment is connected only diagonally to the remaining stock.
        domain = {(u, v) for u in range(-depth, 0) for v in range(2, width + 1)}
        cells.add((-1, 2))
    else:
        domain = {(u, v) for u in range(-depth, 1) for v in range(1, width + 1)}
    assert count >= len(cells) and len(domain | cells) >= count
    dense = rng.random() < .5
    while len(cells) < count:
        frontier = sorted({(u+du, v+dv) for u, v in cells for du, dv in D8}
                          & domain - cells)
        assert frontier
        weights = np.array([1 + sum((u+du, v+dv) in cells for du, dv in D8)
                            for u, v in frontier], dtype=float)
        weights = weights ** (2 if dense else -1)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    return cells


def _encode(scene, templates, targets, order, bg, sep):
    n = next(iter(templates.values())).shape[0]
    h, w = scene.shape
    width = max(len(order) * (n + 1) + 1, w + 2)
    g = np.full((n + h + 6, width), bg, dtype=np.int8)
    g[[0, n+1, n+3], :] = sep
    for j, color in enumerate(order):
        left, right = j * (n+1), (j+1) * (n+1)
        g[1:n+1, left] = sep
        g[1:n+1, right] = sep
        g[1:n+1, left+1:right] = np.where(templates[color], color, bg)
        g[n+2, left] = sep
        g[n+2, right] = sep
        g[n+2, left+1+n//2] = targets[color]
    g[n+4, :w+2] = sep
    g[-1, :w+2] = sep
    g[n+5:-1, 0] = sep
    g[n+5:-1, w+1] = sep
    g[n+5:-1, 1:w+1] = scene
    return g


def _make_pair(rng, n, k, mode, length_band, lesson=None):
    """Construct one scene jointly with its stock budget, then decode it."""
    limit = 30 // n
    for attempt in range(350):
        palette = list(map(int, rng.permutation(10)))
        bg, sep, stock = palette[:3]
        labels = palette[3:3+k]
        targets = dict(zip(labels, palette[3+k:3+2*k]))
        templates = _motifs(rng, n, labels)
        if lesson is None and rng.random() < .18:
            targets[labels[-1]] = targets[labels[0]]
        order = list(map(int, rng.permutation(labels)))
        # Concave roof slopes: +1, then 0, then -1. Depth is independent.
        approach = mode
        near = int(rng.integers(1, 3))
        rise = int(rng.integers(0, 2)) if approach == 'diagonal' else 0
        plateau = int(rng.integers(max(1, k-rise), max(2, k-rise)+1))
        fall = int(rng.integers(0, 3))
        if lesson == 'full' and approach == 'diagonal':
            fall = max(1, fall)
        roof = list(range(rise)) + [rise] * plateau + list(range(rise-1, rise-fall-1, -1))
        if len(roof) < k or len(roof) > (4 if n == 3 else 6):
            continue
        offset = -max(roof) if approach == 'parallel' else near-max(z-i for i, z in enumerate(roof))
        roof = [v+offset for v in roof]
        far = near + len(roof) - 1
        # Every identity appears on a surface slice. Library order is free.
        surface_labels = list(map(int, rng.permutation(labels)))
        if lesson == 'full' and surface_labels[0] == order[0]:
            surface_labels = surface_labels[1:] + surface_labels[:1]
        while len(surface_labels) < len(roof):
            surface_labels.insert(int(rng.integers(len(surface_labels)+1)), int(rng.choice(labels)))
        obstacle = {}
        for i, (v, color) in enumerate(zip(roof, surface_labels)):
            depth = int(rng.integers(1, 4))
            for z in range(v-depth+1, v+1):
                obstacle[near+i, z] = color
        try:
            route = _surface(obstacle, limit)
        except AssertionError:
            continue
        low, high = length_band
        if lesson == 'full':
            low = max(low, far+2)
        if low > high:
            continue
        steps = int(rng.integers(low, high+1))
        local_route = route[:steps]
        cost = sum(int(templates[col].sum()) for u, v, col in local_route)
        diagonal = lesson == 'full' or rng.random() < .4
        # Solve the packing constraints before growing the stock. Reserve one
        # unused continuation cell in the full lesson, so a unit-cost shortcut
        # cannot be concealed by the outer canvas boundary.
        reserved_route = route[:steps+1] if lesson == 'full' else local_route
        positive_u = max(far, reserved_route[-1][0])
        low_v = min([v for u, v in obstacle] + [v for u, v, col in reserved_route])
        high_v = max([v for u, v in obstacle] + [v for u, v, col in reserved_route])
        packing = []
        for support_length in range(3, 6):
            for depth in range(2, 6 if n == 2 else 4):
                for width in range(3, 8 if n == 2 else 7):
                    capacity = depth*(width-1)+1 if diagonal else (depth+1)*width
                    if (cost <= capacity and positive_u+max(depth, support_length-1)+1 <= limit
                            and max(high_v, width)-min(low_v, 0)+1 <= limit):
                        packing.append((support_length, depth, width))
        if not packing or cost < 2:
            continue
        support_length, depth, width = packing[int(rng.integers(len(packing)))]
        slack = _slack(rng, cost, depth, width, diagonal)
        support = {(-u, 0) for u in range(support_length)}
        all_cells = set(obstacle) | slack | support | {p[:2] for p in reserved_route}
        min_u, max_u = min(u for u, v in all_cells), max(u for u, v in all_cells)
        min_v, max_v = min(v for u, v in all_cells), max(v for u, v in all_cells)
        # Canonical scene: forward is right, exposed side is up.
        height, width = max_v-min_v+1, max_u-min_u+1
        if max(height, width) > limit:
            continue
        spare_h, spare_w = limit-height, limit-width
        mh = int(rng.integers(0, min(2, spare_h)+1))
        mw = int(rng.integers(0, min(2, spare_w)+1))
        top, left = int(rng.integers(mh+1)), int(rng.integers(mw+1))
        scene = np.full((height+mh, width+mw), bg, dtype=np.int8)
        def pos(p):
            u, v = p
            return top+max_v-v, left+u-min_u
        for p in support:
            scene[pos(p)] = sep
        for p, color in obstacle.items():
            assert scene[pos(p)] == bg
            scene[pos(p)] = color
        for p in slack:
            assert scene[pos(p)] == bg
            scene[pos(p)] = stock
        turns = int(rng.integers(4))
        scene = np.rot90(scene, turns)
        if rng.random() < .5:
            scene = np.fliplr(scene)
        inp = _encode(scene, templates, targets, order, bg, sep)
        try:
            p = _decode(inp)
            stitches = _stitches(p)
            out = _render(p, stitches)
            assert len(stitches) == steps
            assert all(p['scene'][q] in (bg, stock) for q, col in stitches)
            assert all(0 <= q[0] < scene.shape[0] and 0 <= q[1] < scene.shape[1]
                       for q, col in stitches)
            if lesson == 'full':
                assert set(col for q, col in stitches) == set(labels)
                assert not _connected(p['local_slack'], D4)
                assert all(_witnesses(p)[name] for name in ('geometry', 'identity', 'budget', 'orientation', 'key'))
                if approach == 'diagonal':
                    assert _witnesses(p)['silhouette']
        except (AssertionError, IndexError):
            continue
        return {'input': inp.copy(), 'output': out}
    raise RuntimeError(f'bounded scene construction exhausted: n={n}, k={k}, {mode}, {length_band}, {lesson}')


def _witnesses(p):
    expected = _render(p, _stitches(p))
    def differs(**kw):
        try:
            return not np.array_equal(expected, _render(p, _stitches(p, **kw)))
        except AssertionError:
            return False
    return dict(identity=differs(constant_identity=True) and differs(positional_identity=True),
                budget=differs(unit_cost=True) and differs(first_cost=True), geometry=differs(straight=True),
                silhouette=differs(box=True),
                key=not np.array_equal(expected, _render(p, _stitches(p), ignore_key=True)),
                orientation=all(not np.array_equal(expected, _render(p, _stitches(p), motif_transform=t))
                                for t in range(1, 8)))


def _check_counts(a, b):
    for name, value, lo, hi in (('num_train', a, 2, 10), ('num_test', b, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if a <= b:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(4))
    # A single deliberate shift per puzzle; all semantic conventions stay fixed.
    if regime == 0:  # Two surface identities -> three, at the same scale.
        n = int(rng.choice([2, 3]))
        training = [(n, 2, (5, 7) if n == 2 else (5, 6))] * num_train
        tests = [(n, 3, (5, 7) if n == 2 else (5, 6))] * num_test
    elif regime == 1:  # Longer material-limited prefixes at scale two.
        training = [(2, 2, (5, 7))] * num_train
        tests = [(2, 2, (8, 10))] * num_test
    elif regime == 2:  # A new micro scale; macro grammar stays familiar.
        training = [(2, 2, (5, 6))] * num_train
        tests = [(3, 2, (5, 6))] * num_test
    else:  # Reverse scale transfer at familiar prefix lengths.
        training = [(3, 2, (5, 6))] * num_train
        tests = [(2, 2, (5, 6))] * num_test
    slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    p = {'train': [], 'test': []}
    seen = set()
    for split, schedule in (('train', training), ('test', tests)):
        for i, (n, k, band) in enumerate(schedule):
            lesson = 'full' if split == 'train' and i == slots[0] else None
            approach = ('diagonal' if i == slots[0] else 'parallel') if split == 'train' and i in slots else str(rng.choice(['diagonal', 'parallel']))
            for _ in range(20):
                pair = _make_pair(rng, n, k, approach, band, lesson)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    p[split].append(pair)
                    break
            else:
                raise RuntimeError('unique input construction exhausted')
    validate_generated_puzzle(p)
    return p


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, demonstrations = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            p = _decode(inp)
            stitches = _stitches(p)
            n, bg = p['n'], p['bg']
            assert out.shape == tuple(n*d for d in p['scene'].shape)
            assert np.array_equal(solve(inp), out)
            # A simple pair may select only area-one motifs. The full training
            # lesson, rather than every incidental pair, must refute unit cost.
            assert len(stitches) <= len(p['slack'])
            assert sum(int(p['templates'][col].sum()) for q, col in stitches) == len(p['slack'])
            route_cells = {q for q, col in stitches}
            assert len(route_cells) == len(stitches)
            assert _global((0, 1), p['end'], p['f'], p['l']) == stitches[0][0]
            assert not route_cells & (p['support'] | set(p['obstacle']))
            # Every original nonstock material cell becomes a solid block.
            for q in p['support'] | set(p['obstacle']):
                r, c = q
                assert np.all(out[r*n:(r+1)*n, c*n:(c+1)*n] == p['scene'][q])
            motif_mass = 0
            for (r, c), col in stitches:
                block = out[r*n:(r+1)*n, c*n:(c+1)*n]
                assert np.array_equal(block != bg, p['templates'][col])
                assert set(map(int, np.unique(block))) <= {bg, p['targets'][col]}
                motif_mass += int(np.sum(block != bg))
            assert motif_mass == len(p['slack'])
            for r in range(p['scene'].shape[0]):
                for c in range(p['scene'].shape[1]):
                    if (r, c) not in route_cells | p['support'] | set(p['obstacle']):
                        assert np.all(out[r*n:(r+1)*n, c*n:(c+1)*n] == bg)
            if split == 'train':
                demonstrations.append(p)
    for axis in ('identity', 'budget', 'geometry', 'silhouette', 'orientation', 'key'):
        assert any(_witnesses(p)[axis] for p in demonstrations), f'missing {axis} evidence'
    assert any(not _connected(p['local_slack'], D4) for p in demonstrations)
    assert {max(v for u, v in p['local_obstacle']) == 0 for p in demonstrations} == {True, False}
    assert any(len(_stitches(p))-1 > max(u for u, v in p['local_obstacle']) for p in demonstrations)


def measure_puzzle(puzzle):
    """Compact measurements derived only from the delivered grids."""
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            p = _decode(pair['input'])
            stitches = _stitches(p)
            obs = p['local_obstacle']
            near, far = min(u for u, v in obs), max(u for u, v in obs)
            roof = tuple(max(v for u, v in obs if u == t) for t in range(near, far+1))
            records.append(dict(split=split, scale=p['n'], identities=len(p['order']),
                input_shape=list(np.shape(pair['input'])), scene_shape=list(p['scene'].shape),
                background=p['bg'], stock=len(p['slack']), steps=len(stitches),
                costs=sorted(int(x.sum()) for x in p['templates'].values()),
                roof=roof, approach='parallel' if max(roof) == 0 else 'diagonal',
                direction=list(p['f']), side=list(p['l']), support=len(p['support']),
                needs_diagonal=not _connected(p['local_slack'], D4),
                selected_identities=len({c for q, c in stitches}),
                witnesses=_witnesses(p)))
    return records


def self_test():
    """Bounded 42-seed audit, route regressions, metamorphic and negative checks."""
    surfaces = [
        ({(2, 0): 1}, [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 0, 1), (4, -1, 1)]),
        ({(2, 1): 1, (2, 2): 1, (3, 1): 2, (3, 2): 2},
         [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 3, 2), (4, 2, 2)]),
        ({(1, 1): 1, (2, 2): 2, (3, 1): 3},
         [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 2, 3), (4, 1, 3)]),
    ]
    for obstacle, expected in surfaces:
        assert _surface(obstacle, len(expected)) == expected
    # Hand-derived mixed-cost scene: 12 cells fund owners 1,1,1,2,2 at
    # costs 2,2,2,3,3. Changing only motif 1 to area one funds a sixth tile.
    hand_scene = np.zeros((10, 12), dtype=np.int8)
    hand_templates = {1: np.array([[1, 0], [1, 0]], dtype=bool),
                      2: np.array([[1, 1], [1, 0]], dtype=bool)}
    hand_targets = {1: 6, 2: 7}
    end, f, l = (7, 4), (0, 1), (-1, 0)
    for u in range(-3, 1):
        hand_scene[_global((u, 0), end, f, l)] = 5
    for u, col in ((2, 1), (3, 2)):
        for v in (1, 2):
            hand_scene[_global((u, v), end, f, l)] = col
    hand_stock = [(0, 1)] + [(u, v) for u in (-1, -2, -3) for v in range(1, 5)]
    for q in hand_stock[:12]:
        hand_scene[_global(q, end, f, l)] = 9
    hand_input = _encode(hand_scene, hand_templates, hand_targets, [2, 1], 0, 5)
    blank = hand_scene.copy()
    blank[blank == 9] = 0
    expected = np.repeat(np.repeat(blank, 2, axis=0), 2, axis=1)
    manual_route = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 3, 2), (4, 2, 2)]
    for u, v, col in manual_route:
        r, c = _global((u, v), end, f, l)
        expected[2*r:2*r+2, 2*c:2*c+2] = np.where(hand_templates[col], hand_targets[col], 0)
    assert np.array_equal(solve(hand_input), expected)
    smaller = {col: mask.copy() for col, mask in hand_templates.items()}
    smaller[1][1, 0] = False
    changed_input = _encode(hand_scene, smaller, hand_targets, [2, 1], 0, 5)
    changed = _decode(changed_input)
    assert len(_stitches(changed)) == 6 and len(changed['slack']) == 12
    assert not np.array_equal(solve(changed_input), expected)
    # A repeated silhouette / equal-area library is legal, although the
    # sampler favors distinct costs. Its keyed identities remain meaningful.
    equal = {1: hand_templates[1], 2: hand_templates[1]}
    equal_scene = hand_scene.copy()
    equal_scene[equal_scene == 9] = 0
    for q in hand_stock[:10]:
        equal_scene[_global(q, end, f, l)] = 9
    equal_input = _encode(equal_scene, equal, hand_targets, [2, 1], 0, 5)
    assert len(_stitches(_decode(equal_input))) == 5
    equal_expected = np.repeat(np.repeat(blank, 2, axis=0), 2, axis=1)
    for u, v, col in manual_route:
        r, c = _global((u, v), end, f, l)
        equal_expected[2*r:2*r+2, 2*c:2*c+2] = np.where(equal[col], hand_targets[col], 0)
    assert np.array_equal(solve(equal_input), equal_expected)
    records, shapes, roofs, motifs, structural = [], set(), set(), set(), set()
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        validate_generated_puzzle(puzzle)
        assert all(np.array_equal(solve(pair['input'].copy()), pair['output'])
                   for split in puzzle for pair in puzzle[split])
        records.extend(measure_puzzle(puzzle))
        for split in puzzle:
            for pair in puzzle[split]:
                p = _decode(pair['input'])
                shapes.add(p['scene'].shape)
                roofs.add(tuple(sorted(p['local_obstacle'])))
                motifs.update((p['n'], mask.tobytes()) for mask in p['templates'].values())
                structural.add((p['n'], p['scene'].shape, tuple(p['f']), tuple(p['l']),
                                tuple(sorted(p['local_slack'])), tuple(sorted(p['local_obstacle']))))
    assert {r['scale'] for r in records} == {2, 3}
    assert {r['identities'] for r in records} == {2, 3}
    assert len(shapes) >= 20 and len(roofs) >= 30 and len(motifs) >= 40
    assert len(structural) >= 180
    assert len({r['stock'] for r in records}) >= 20
    assert len({r['background'] for r in records}) == 10
    assert {tuple(r['direction']) for r in records} == set(D4)
    assert {r['approach'] for r in records} == {'parallel', 'diagonal'}
    assert {r['needs_diagonal'] for r in records} == {True, False}
    # Material conservation is an actual metamorphic behavior: shrinking the
    # stock by the last tile's area shortens the route by exactly one tile.
    p = _decode(puzzles[0]['train'][0]['input'])
    stitches = _stitches(p)
    final_cost = int(p['templates'][stitches[-1][1]].sum())
    short = _stitches(p, budget=len(p['slack'])-final_cost)
    assert short == stitches[:-1]
    # Rearranging the same material within the available rear quadrant must
    # leave the answer unchanged, including when diagonal links disappear.
    new_scene = p['scene'].copy()
    new_scene[new_scene == p['stock']] = p['bg']
    stock_depth = -min(u for u, v in p['local_slack'])
    stock_width = max(v for u, v in p['local_slack'])
    rearranged = _slack(np.random.default_rng(1701), len(p['slack']), stock_depth,
                        stock_width, False)
    for q in rearranged:
        new_scene[_global(q, p['end'], p['f'], p['l'])] = p['stock']
    new_input = _encode(new_scene, p['templates'], p['targets'], p['order'], p['bg'], p['sep'])
    assert np.array_equal(solve(new_input), puzzles[0]['train'][0]['output'])
    # Palette permutation affects no geometric decisions or template costs.
    permutation = np.array([9, 7, 4, 0, 8, 6, 1, 3, 5, 2], dtype=np.int8)
    pair = puzzles[0]['train'][0]
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # Complete determinism, all explicit counts, and bad count types/bounds.
    assert all(np.array_equal(a[k], b[k]) for s in puzzles[0]
               for a, b in zip(puzzles[0][s], generate_puzzle(0)[s]) for k in a)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            pz = generate_puzzle(999+a*10+b, a, b)
            assert len(pz['train']) == a and len(pz['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (3, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = copy.deepcopy(puzzles[0])
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=len(records), scene_shapes=len(shapes),
                obstacle_shapes=len(roofs), motifs=len(motifs), structures=len(structural),
                stock_counts=sorted({r['stock'] for r in records}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
