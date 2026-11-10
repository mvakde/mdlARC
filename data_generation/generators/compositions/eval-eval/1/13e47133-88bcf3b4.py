"""Contour-selective stitching: unary material on a taut, periodically gated route.

Only the standard library and NumPy are used. Nothing is written by generation,
validation or self_test. All roles and the stencil period are read from the input.
"""
import hashlib
import json
import numpy as np


D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0,572440362,819709003,879540914,728828838,285152156,458597273,550815584,22664774,326312443,159022039,607922875,802649145,768922715,538859675,263016998,339157259,116305993,987523806,923593654,539443727,227182603,312333307,632435124,382425240,321883639,255686318,664414922,474999888,439948068,864775365,606165255,708353694,77762778,108239232,180744596,417341976,430913953,181882545,15370607,808357921,857968402)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(a, bg):
    unseen = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    result = []
    while unseen:
        root = min(unseen)
        color = int(a[root])
        cells, todo = {root}, [root]
        unseen.remove(root)
        while todo:
            r, c = todo.pop()
            for dr, dc in D8:
                p = r + dr, c + dc
                if p in unseen and a[p] == color:
                    unseen.remove(p)
                    cells.add(p)
                    todo.append(p)
        result.append((color, cells))
    return result


def _connected(cells, directions):
    if not cells:
        return False
    todo = [min(cells)]
    seen = set(todo)
    while todo:
        r, c = todo.pop()
        for dr, dc in directions:
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _depths(shape, obstacle, diagonal=True):
    """Distance-transform erosion depths; the support is transparent to dye.

    Eight-neighbor distance is Chebyshev distance. A removed pixel is -1, its
    neighbors are 0. Boundary distance already counts the off-canvas exterior.
    """
    rr, cc = np.indices(shape, dtype=np.int16)
    depth = np.minimum.reduce((rr, cc, shape[0] - 1 - rr, shape[1] - 1 - cc))
    for r, c in obstacle:
        dr, dc = np.abs(rr - r), np.abs(cc - c)
        distance = np.maximum(dr, dc) if diagonal else dr + dc
        depth = np.minimum(depth, distance - 1)
    return depth


def _eroded_depths(shape, obstacle):
    """Slower erosion formulation used for validation, not route construction."""
    current = np.ones(shape, dtype=bool)
    for p in obstacle:
        current[p] = False
    result = np.full(shape, -1, dtype=np.int16)
    d = 0
    while current.any():
        padded = np.pad(current, 1)
        inner = current.copy()
        for dr, dc in D8:
            inner &= padded[1 + dr:1 + dr + shape[0], 1 + dc:1 + dc + shape[1]]
        result[current & ~inner] = d
        current = inner
        d += 1
    return result


def _global(p, end, f, lateral):
    return (end[0] + p[0] * f[0] + p[1] * lateral[0],
            end[1] + p[0] * f[1] + p[1] * lateral[1])


def _local(p, end, f, lateral):
    dr, dc = p[0] - end[0], p[1] - end[1]
    return dr * f[0] + dc * f[1], dr * lateral[0] + dc * lateral[1]


def _taut_profile(obstacle, n=32):
    """Source-supported digital tangent/contour: one forward step per site.

    Actual cells, rather than a bounding box, determine the clearance envelope.
    Parallel and 45-degree incoming tangents and convex exposed profiles only.
    """
    near = min(u for u, v in obstacle)
    peak = max(v for u, v in obstacle)
    tangent = max(v - u for u, v in obstacle)
    if near < 1 or peak < 0 or not (tangent == 0 or peak == 0):
        raise ValueError("unsupported obstacle placement")
    first_peak = min(u for u, v in obstacle if v == peak)
    route = [(u, 1 if peak == 0 and u <= first_peak else
              max(v + 1 - abs(u - t) for t, v in obstacle)) for u in range(n)]
    slopes = [b[1] - a[1] for a, b in zip(route, route[1:])]
    if (route[0] != (0, 1) or any(x not in (-1, 0, 1) for x in slopes)
            or slopes != sorted(slopes, reverse=True) or set(route) & obstacle):
        raise ValueError("nonconvex or ambiguous digital tangent")
    return route


def _visible_route(obstacle, end, f, lateral, shape):
    route = []
    # By u=30 the monotonically advancing axis has left every legal grid.
    for p in _taut_profile(obstacle):
        q = _global(p, end, f, lateral)
        if not _inside(q, shape):
            break
        route.append(q)
    return route


def _read_key(a, cue_cells, depth, roles):
    if len(cue_cells) not in (2, 3):
        raise ValueError("one explicit diagonal key of length two or three is required")
    k = len(cue_cells)
    if len({int(a[p]) for p in cue_cells}) != k:
        raise ValueError("key entries must be distinct")
    if any(int(a[p]) in roles for p in cue_cells):
        raise ValueError("key cannot use background, support or peg color")
    readings = set()
    for root in sorted(cue_cells):
        if depth[root] != 0:
            continue
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            positions = tuple((root[0] + d * dr, root[1] + d * dc) for d in range(k))
            if set(positions) != cue_cells:
                continue
            if all(depth[p] == d for d, p in enumerate(positions)):
                # A real convex corner, including the outside of the canvas.
                r, c = root
                outside = lambda p: not _inside(p, a.shape) or depth[p] < 0
                if outside((r - dr, c)) and outside((r, c - dc)):
                    readings.add(tuple(int(a[p]) for p in positions))
    if len(readings) != 1:
        raise ValueError("key must have one contour-depth reading")
    return next(iter(readings))


def _infer(a):
    """Enumerate complete input explanations; no output or sampling state is used."""
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if sum(counts == counts[bg]) != 1:
        raise ValueError("background must be the unique dominant color")
    components = _components(a, bg)
    candidates = []
    for i, (sc, support) in enumerate(components):
        if not 3 <= len(support) <= 8:
            continue
        if len({r for r, c in support}) == 1:
            positive = (0, 1)
        elif len({c for r, c in support}) == 1:
            positive = (1, 0)
        else:
            continue
        if not _connected(support, D4):
            continue
        for end, f in ((max(support), positive), (min(support), (-positive[0], -positive[1]))):
            for lateral in ((f[1], -f[0]), (-f[1], f[0])):
                attachment = _global((0, 1), end, f, lateral)
                for j, (tc, slack) in enumerate(components):
                    if tc == sc or not 4 <= len(slack) <= 18 or attachment not in slack:
                        continue
                    local_slack = {_local(p, end, f, lateral) for p in slack}
                    if any(u > 0 or v < 1 for u, v in local_slack):
                        continue
                    for z, (oc, peg) in enumerate(components):
                        if z in (i, j) or oc in (sc, tc):
                            continue
                        cues = set().union(*(cells for q, (_, cells) in enumerate(components)
                                             if q not in (i, j, z)))
                        if len(cues) not in (2, 3):
                            continue
                        local_peg = {_local(p, end, f, lateral) for p in peg}
                        try:
                            route = _visible_route(local_peg, end, f, lateral, a.shape)
                            if not route or set(route) & (support | peg | cues):
                                continue
                            depth = _depths(a.shape, peg)
                            period = _read_key(a, cues, depth, (bg, sc, oc))
                            if tc not in period or period[int(depth[attachment]) % len(period)] != tc:
                                continue
                            sites = [p for p in route if period[int(depth[p]) % len(period)] == tc]
                            if len(sites) < 2:
                                continue
                        except ValueError:
                            continue
                        candidates.append(dict(bg=bg, sc=sc, tc=tc, oc=oc, support=support,
                                               slack=slack, peg=peg, cues=cues, period=period,
                                               end=end, f=f, lateral=lateral, depth=depth,
                                               route=route, sites=sites, local_peg=local_peg))
    if len(candidates) != 1:
        raise ValueError(f"expected one input-only assembly explanation, got {len(candidates)}")
    return candidates[0]


def _answer(a, info):
    out = a.copy()
    for p in info['slack']:
        out[p] = info['bg']
    for p in info['sites'][:len(info['slack'])]:
        out[p] = info['tc']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    return _answer(a, _infer(a))


def _peg(rng, kind, approach, near):
    if kind == 0:
        raw = {(0, 0)}
    elif kind == 1:
        size = int(rng.integers(2, 6))
        raw = {(u, 0) for u in range(size)}
    elif kind == 2:
        h, w = map(int, rng.integers(1, 5, size=2))
        raw = {(u, v) for u in range(h) for v in range(w)}
    elif kind == 3:
        radius = int(rng.integers(1, 4))
        raw = {(u, radius - abs(u - radius)) for u in range(2 * radius + 1)}
    elif kind == 4:
        radius = int(rng.integers(1, 4))
        raw = {(u, v) for u in range(2 * radius + 1) for v in range(-radius, radius + 1)
               if abs(u - radius) + abs(v) <= radius}
    else:
        ramp, plateau = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        raw = {(u, v) for u in range(ramp + plateau)
               for v in range(ramp - max(0, u - plateau + 1) + 1)}
        if rng.integers(2):
            far = max(u for u, v in raw)
            raw = {(far - u, v) for u, v in raw}
    shift = -max(v for u, v in raw) if approach == 0 else near - max(v - u for u, v in raw)
    result = {(u + near, v + shift) for u, v in raw}
    _taut_profile(result)
    return result


def _slack(rng, n, style, diagonal_lesson=False):
    if diagonal_lesson:
        # Only a diagonal link joins the attachment to the rest of the material.
        rest = _slack(rng, n - 1, style)
        return {(0, 1)} | {(u - 1, v + 1) for u, v in rest}
    cells = {(0, 1)}
    height, width = int(rng.integers(3, 7)), int(rng.integers(3, 8))
    while len(cells) < n:
        dirs = D4 if style == 0 else D8
        frontier = sorted({(u + du, v + dv) for u, v in cells for du, dv in dirs
                           if -height < u + du <= 0 and 1 <= v + dv <= width} - cells)
        if style == 2:
            weights = np.array([1 / (1 + sum((u + du, v + dv) in cells for du, dv in D8)) ** 2
                                for u, v in frontier], dtype=float)
            choice = int(rng.choice(len(frontier), p=weights / weights.sum()))
        else:
            choice = int(rng.integers(len(frontier)))
        cells.add(frontier[choice])
    return cells


def _candidate(rng, k, lesson, axis, bg, geometry_kind=None):
    kind = int(rng.integers(6)) if geometry_kind is None else geometry_kind
    near = int(rng.integers(3 if lesson == 'visible' else 1, 9))
    obstacle = _peg(rng, kind, int(rng.integers(2)), near)
    support = {(-u, 0) for u in range(int(rng.integers(3, 9)))}
    n = int(rng.integers(6 if lesson == 'visible' else 4, 15 if lesson == 'visible' else 19))
    slack = _slack(rng, n, int(rng.integers(3)), lesson == 'visible')
    all_cells = support | obstacle | slack
    lo_u, hi_u = min(u for u, v in all_cells), max(u for u, v in all_cells)
    lo_v, hi_v = min(v for u, v in all_cells), max(v for u, v in all_cells)
    # Generous but independently varied margins expose several contour depths.
    minimum_h, minimum_w = hi_u - lo_u + 3, hi_v - lo_v + 3
    h = int(rng.integers(max(17 if lesson == 'visible' else 10, minimum_h), 31))
    w = int(rng.integers(max(16 if lesson == 'visible' else 10, minimum_w), 31))
    top = int(rng.integers(1, h - (hi_u - lo_u + 1)))
    left = int(rng.integers(1, w - (hi_v - lo_v + 1)))
    end = top - lo_u, left - lo_v
    shift = lambda cells: {(u + end[0], v + end[1]) for u, v in cells}
    support_g, peg_g, slack_g = map(shift, (support, obstacle, slack))
    route = _visible_route(obstacle, end, (1, 0), (0, 1), (h, w))
    depth = _depths((h, w), peg_g)
    active = int(depth[end[0], end[1] + 1]) % k
    palette = [int(c) for c in rng.permutation(10) if c != bg]
    sc, tc, oc = palette[:3]
    period = palette[3:3 + k]
    period[active] = tc
    sites = [p for p in route if int(depth[p]) % k == active]
    if len(sites) < 3:
        raise ValueError('not enough visible stitches')
    if lesson == 'visible':
        # The first lesson must teach actual periodic recurrence, a partial
        # budget, the bend around an irregular peg, and skipped positions.
        selected = sites[:n]
        if (active != 0 or n >= len(sites) or len(selected) != n
                or not any(depth[p] >= k for p in selected)
                or not any(near <= p[0] - end[0] <= max(u for u, v in obstacle)
                           for p in selected)
                or not any(p[0] - end[0] > max(u for u, v in obstacle) + 1 for p in selected)):
            raise ValueError('missing visible curriculum witness')
    elif lesson == 'clipped':
        if n <= len(sites) or active == 0 or not any(depth[p] >= k + active for p in sites):
            raise ValueError('missing clipped, recurring or nonzero-phase witness')
    elif rng.random() < .6:
        # Free examples retain both saturated and genuinely budget-limited cases.
        n = int(rng.integers(4, min(18, len(sites)) + 1)) if len(sites) >= 4 else n
        slack = _slack(rng, n, int(rng.integers(3)))
        slack_g = shift(slack)
        if not all(_inside(p, (h, w)) for p in slack_g):
            raise ValueError('resampled material does not fit')
    used = support_g | peg_g | slack_g | set(route)
    key_options = []
    for root, direction in (((0, 0), (1, 1)), ((0, w - 1), (1, -1)),
                            ((h - 1, 0), (-1, 1)), ((h - 1, w - 1), (-1, -1))):
        positions = [(root[0] + d * direction[0], root[1] + d * direction[1]) for d in range(k)]
        if (all(depth[p] == d for d, p in enumerate(positions))
                and not any((r + dr, c + dc) in used for r, c in positions for dr, dc in ((0, 0),) + D8)):
            key_options.append(positions)
    if not key_options:
        raise ValueError('no separated corner key')
    key = key_options[int(rng.integers(len(key_options)))]
    inp = np.full((h, w), bg, dtype=np.int8)
    for cells, color in ((support_g, sc), (peg_g, oc), (slack_g, tc)):
        for p in cells:
            if inp[p] != bg:
                raise ValueError('input objects overlap')
            inp[p] = color
    for p, color in zip(key, period):
        inp[p] = color
    out = inp.copy()
    for p in slack_g:
        out[p] = bg
    for p in sites[:n]:
        out[p] = tc
    # The axis shift is the only mandatory test novelty. Reflections within
    # each axis vary independently in both splits.
    turns = axis + 2 * int(rng.integers(2))
    inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
    if rng.integers(2):
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    return {'input': inp, 'output': out}


def _sites_with_depth(info, depth, route=None):
    period, color = info['period'], info['tc']
    return [p for p in (info['route'] if route is None else route)
            if depth[p] >= 0 and period[int(depth[p]) % len(period)] == color][:len(info['slack'])]


def _witnesses(info):
    actual = set(info['sites'][:len(info['slack'])])
    raw = set(info['route'][:len(info['slack'])])
    straight = []
    diagonal = []
    for u in range(32):
        p = _global((u, 1), info['end'], info['f'], info['lateral'])
        if not _inside(p, info['depth'].shape):
            break
        if p not in info['peg']:
            straight.append(p)
    for u in range(32):
        p = _global((u, u + 1), info['end'], info['f'], info['lateral'])
        if not _inside(p, info['depth'].shape):
            break
        if p not in info['peg']:
            diagonal.append(p)
    l1 = _depths(info['depth'].shape, info['peg'], diagonal=False)
    r0, r1 = min(r for r, c in info['peg']), max(r for r, c in info['peg'])
    c0, c1 = min(c for r, c in info['peg']), max(c for r, c in info['peg'])
    box = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    box_depth = _depths(info['depth'].shape, box)
    return dict(ignores_A=actual != raw,
                ignores_B=actual != set(_sites_with_depth(info, info['depth'], straight)),
                diagonal_ray=actual != set(_sites_with_depth(info, info['depth'], diagonal)),
                ignores_budget=len(info['slack']) < len(info['sites']),
                manhattan=actual != set(_sites_with_depth(info, l1)),
                bounding_box=actual != set(_sites_with_depth(info, box_depth)),
                recurrence=any(info['depth'][p] >= len(info['period']) for p in actual),
                diagonal=not _connected(info['slack'], D4),
                clipped=len(info['slack']) > len(info['sites']))


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi):
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt, ne = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    axis = int(rng.integers(2))
    backgrounds = list(map(int, rng.permutation(10)))
    for i in range(nt + ne):
        split = 'train' if i < nt else 'test'
        lesson = 'visible' if i == 0 else 'clipped' if i == 1 else None
        k = 2 if i == 0 else 3 if i == 1 else int(rng.integers(2, 4))
        for attempt in range(800):
            try:
                pair = _candidate(rng, k, lesson, axis if split == 'train' else 1 - axis,
                                  backgrounds[i % 10], int(rng.integers(3, 6)) if i == 0 else None)
                a = pair['input']
                info = _infer(a)
                key = (a.shape, a.tobytes())
                if key in seen or not np.array_equal(_answer(a, info), pair['output']):
                    continue
                w = _witnesses(info)
                if i == 0 and not all(w[x] for x in ('ignores_A', 'ignores_B', 'diagonal_ray', 'ignores_budget',
                                                    'manhattan', 'bounding_box', 'recurrence', 'diagonal')):
                    continue
                if i == 1 and not w['clipped']:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(f'bounded construction exhausted at pair {i}')
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, axes = set(), [], {'train': set(), 'test': set()}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            key = a.shape, a.tobytes()
            assert key not in seen
            seen.add(key)
            info = _infer(a)
            assert np.array_equal(_answer(a, info), b), 'input-only solve mismatch'
            assert np.array_equal(info['depth'], _eroded_depths(a.shape, info['peg']))
            assert not np.array_equal(a, b)
            for p in info['support'] | info['peg'] | info['cues']:
                assert a[p] == b[p], 'changed preserved support, obstacle or key'
            drawn = {tuple(map(int, p)) for p in np.argwhere(b == info['tc'])} - info['cues']
            assert drawn == set(info['sites'][:len(info['slack'])])
            assert len(drawn) == min(len(info['slack']), len(info['sites']))
            assert _global((0, 1), info['end'], info['f'], info['lateral']) in drawn
            assert all(info['period'][int(info['depth'][p]) % len(info['period'])] == info['tc'] for p in drawn)
            axes[split].add(0 if info['f'][0] else 1)
            if split == 'train':
                training.append(info)
    assert len(axes['train']) == len(axes['test']) == 1 and axes['train'] != axes['test']
    first = _witnesses(training[0])
    assert all(first[x] for x in ('ignores_A', 'ignores_B', 'diagonal_ray', 'ignores_budget', 'manhattan',
                                'bounding_box', 'recurrence', 'diagonal'))
    assert _witnesses(training[1])['clipped'] and _witnesses(training[1])['recurrence']
    assert any(_local(p, training[0]['end'], training[0]['f'], training[0]['lateral'])[0]
               > max(u for u, v in training[0]['local_peg']) + 1
               for p in training[0]['sites'][:len(training[0]['slack'])])
    assert len(training[0]['period']) == 2 and len(training[1]['period']) == 3
    assert training[1]['period'].index(training[1]['tc']) != 0
    return None


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for p in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(p[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 host seeds, sequential and in memory; aggregate semantic diversity."""
    signatures, shapes, silhouettes, lengths, phases, palettes, structures = (set() for _ in range(7))
    totals = dict(pairs=0, clipped=0, diagonal=0, manhattan=0, bounding_box=0)
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        assert _digest(p) == _digest(generate_puzzle(seed))
        signatures.add(_digest(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                info = _infer(a)
                assert np.array_equal(solve(a.copy()), pair['output'])
                shapes.add(a.shape)
                lengths.add(len(info['slack']))
                phases.add((len(info['period']), info['period'].index(info['tc'])))
                palettes.add((info['bg'], info['sc'], info['tc'], info['oc']))
                u0 = min(u for u, v in info['local_peg'])
                v0 = min(v for u, v in info['local_peg'])
                silhouettes.add(tuple(sorted((u - u0, v - v0) for u, v in info['local_peg'])))
                structures.add((a.shape, info['f'], info['lateral'], len(info['support']),
                                len(info['slack']), tuple(sorted(info['local_peg'])),
                                len(info['period']), info['period'].index(info['tc'])))
                w = _witnesses(info)
                totals['pairs'] += 1
                for name in ('clipped', 'diagonal', 'manhattan', 'bounding_box'):
                    totals[name] += int(w[name])
    assert len(signatures) == 42 and len(shapes) >= 45 and len(silhouettes) >= 15
    assert len(lengths) >= 10 and len(phases) == 5 and len(palettes) >= 100
    assert len(structures) >= 180 and all(totals[x] >= 42 for x in ('clipped', 'diagonal', 'manhattan', 'bounding_box'))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(19, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(1, 2, 1)
    info = _infer(damaged['train'][0]['input'])
    damaged['train'][0]['output'][min(info['peg'])] = info['bg']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted peg corruption')
    return dict(seeds=42, distinct_puzzles=len(signatures), shapes=len(shapes),
                silhouettes=len(silhouettes), material_lengths=sorted(lengths),
                key_phases=sorted(phases), palettes=len(palettes), structures=len(structures), **totals)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
