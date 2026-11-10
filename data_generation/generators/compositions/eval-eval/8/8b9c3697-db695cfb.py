"""Finite diagonal rails enable size-matched docking into diagonal forks.

Only NumPy and the standard library are used. Ground truth is assembled from
constructed footprints; solve reconstructs every role and action from the input.
"""

import json
from itertools import combinations

import numpy as np


TRAIL = 0
DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in unseen:
                        unseen.remove(p)
                        cells.add(p)
                        stack.append(p)
        result.append(cells)
    return result


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _is_bar(cells):
    if len(cells) == 1:
        return True
    ordered = sorted(cells)
    a, b = ordered[0], ordered[-1]
    if b[0] - a[0] != len(cells) - 1 or abs(b[1] - a[1]) != len(cells) - 1:
        return False
    slope = 1 if b[1] > a[1] else -1
    return cells == {(a[0] + t, a[1] + slope * t) for t in range(len(cells))}


def _parse_fork(cells):
    """Recognize a comb on the two perpendicular diagonal lattice axes."""
    parses = []
    for out in DIAGONALS:
        perp = (out[1], -out[0])
        kmin = min(r * out[0] + c * out[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        raw = [(r * out[0] + c * out[1] - kmin,
                r * perp[0] + c * perp[1] - qmin) for r, c in cells]
        if any(k % 2 or q % 2 for k, q in raw):
            continue
        local = {(k // 2, q // 2) for k, q in raw}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        depths = [sum(k > 0 and qq == q for k, qq in local) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, depth in enumerate(depths) for k in range(1, depth + 1)}
        if local != expected:
            continue
        runs, q = [], 0
        while q < width:
            if depths[q] == 0:
                q += 1
                continue
            first, depth = q, depths[q]
            while q < width and depths[q] == depth:
                q += 1
            runs.append((first, q, depth))
        if len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1)):
            continue
        first, last, center = runs[1]
        if first + last != width:
            continue
        parses.append(dict(cells=cells, out=out, perp=perp, kmin=kmin, qmin=qmin,
                           width=width, runs=tuple(runs), span=last - first,
                           q0=first, stop=center + 1,
                           lengths=tuple(run[2] for run in runs)))
    if len(parses) != 1:
        raise ValueError("A fork must have one full base and three positive separated prongs")
    return parses[0]


def _cell(fork, k, q):
    kk, qq = fork['kmin'] + 2 * k, fork['qmin'] + 2 * q
    out, perp = fork['out'], fork['perp']
    return ((kk * out[0] + qq * perp[0]) // 2,
            (kk * out[1] + qq * perp[1]) // 2)


def _candidate(fork, token, grid, background, exact_width=True):
    raw = [(r * fork['out'][0] + c * fork['out'][1] - fork['kmin'],
            r * fork['perp'][0] + c * fork['perp'][1] - fork['qmin'])
           for r, c in token['cells']]
    if any(k % 2 or q % 2 for k, q in raw):
        return None
    local = {(k // 2, q // 2) for k, q in raw}
    ks, qs = {k for k, _ in local}, {q for _, q in local}
    if len(ks) != 1 or qs != set(range(min(qs), max(qs) + 1)):
        return None
    required = set(range(fork['q0'], fork['q0'] + fork['span']))
    if exact_width:
        if qs != required:
            return None
    elif min(qs) + max(qs) != min(required) + max(required):
        return None
    k0 = next(iter(ks))
    if k0 <= fork['stop']:
        return None
    route = {_cell(fork, k, q) for k in range(fork['stop'], k0 + 1) for q in qs}
    if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or
           ((r, c) not in token['cells'] and int(grid[r, c]) != background)
           for r, c in route):
        return None
    final = {_cell(fork, fork['stop'], q) for q in qs}
    return dict(route=route, final=final, distance=k0 - fork['stop'])


def _segment(a, b):
    if a == b or abs(a[0] - b[0]) != abs(a[1] - b[1]):
        raise ValueError("Rail endpoints must lie on a nondegenerate diagonal")
    dr, dc = (1 if b[0] > a[0] else -1), (1 if b[1] > a[1] else -1)
    return tuple((a[0] + i * dr, a[1] + i * dc) for i in range(abs(b[0] - a[0]) + 1))


def _analyze(input_grid):
    grid = np.asarray(input_grid)
    if grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("Input must be a two-dimensional integer grid")
    if any(n < 1 or n > 30 for n in grid.shape) or np.any((grid < 0) | (grid > 9)):
        raise ValueError("Input is outside ARC grid bounds")
    if np.any(grid == TRAIL):
        raise ValueError("Black trail role 0 must be absent from input")
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    if counts.max() * 2 <= grid.size:
        raise ValueError("Background must occupy more than half of the input")
    components = {int(c): _components(grid == c) for c in values if c != background}
    structural = []
    for color, groups in components.items():
        if any(not _is_bar(cells) for cells in groups):
            structural.append(color)
    if len(structural) != 1:
        raise ValueError("Exactly one color must supply forks and rail endpoints")
    structure = structural[0]
    forks, endpoints, tokens = [], [], []
    occupied = set()
    for color, groups in components.items():
        for cells in groups:
            if not occupied.isdisjoint(_halo(cells)):
                raise ValueError("Input objects must be separated under eight-neighbor adjacency")
            occupied |= cells
            if color == structure:
                if len(cells) == 1:
                    endpoints.append(next(iter(cells)))
                else:
                    forks.append(_parse_fork(cells))
            else:
                if not _is_bar(cells):
                    raise ValueError("Every movable token must be a contiguous diagonal bar")
                tokens.append(dict(cells=cells, color=color))
    if not forks or not tokens:
        raise ValueError("At least one fork and one token are required")
    rails, degrees = [], {p: 0 for p in endpoints}
    for a, b in combinations(sorted(endpoints), 2):
        if abs(a[0] - b[0]) == abs(a[1] - b[1]):
            rail = _segment(a, b)
            if len(rail) < 3:
                raise ValueError("Rail length must be at least three lattice cells")
            degrees[a] += 1
            degrees[b] += 1
            rails.append(rail)
    if not rails or any(d > 1 for d in degrees.values()):
        raise ValueError("Rail endpoints need unique diagonal partners")
    structures = set().union(*(f['cells'] for f in forks))
    rail_union = set().union(*(set(r) for r in rails))
    if rail_union & structures:
        raise ValueError("Rails must not pass through fork structure")
    matches, active, geometric = [], {}, {}
    used_forks, used_routes = set(), set()
    for ti, token in enumerate(tokens):
        containing = [ri for ri, rail in enumerate(rails) if token['cells'] <= set(rail[1:-1])]
        if len(containing) > 1:
            raise ValueError("A bar cannot be enabled by two rails")
        candidates = [(fi, candidate) for fi, fork in enumerate(forks)
                      if (candidate := _candidate(fork, token, grid, background)) is not None]
        geometric[ti] = candidates
        if not containing:
            continue
        active[ti] = containing[0]
        if len(candidates) > 1:
            raise ValueError("An enabled token must have at most one matching socket")
        if candidates:
            fi, candidate = candidates[0]
            if fi in used_forks or used_routes & candidate['route']:
                raise ValueError("Successful dockings must have distinct sockets and disjoint sweeps")
            if candidate['route'] & rail_union != token['cells']:
                raise ValueError("A sweep may intersect rails only at its original token footprint")
            used_forks.add(fi)
            used_routes |= candidate['route']
            matches.append(dict(candidate, ti=ti, fi=fi, ri=containing[0]))
    return dict(background=background, structure=structure, forks=forks, tokens=tokens,
                endpoints=endpoints, rails=rails, rail_union=rail_union, active=active,
                matches=matches, geometric=geometric,
                unpaired=[p for p, d in degrees.items() if d == 0])


def _render(grid, analysis):
    output = np.asarray(grid).copy()
    for token in analysis['tokens']:
        _paint(output, token['cells'], analysis['background'])
    _paint(output, analysis['rail_union'], analysis['structure'])
    for match in analysis['matches']:
        _paint(output, match['route'] - match['final'], TRAIL)
        _paint(output, match['final'], analysis['tokens'][match['ti']]['color'])
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, finite gates, socket matches and swept footprints from input."""
    return _render(input_grid, _analyze(input_grid))


def _fork_local(q0, span, branch, travel, center, guards, gap):
    base = branch * (travel + center + 1)
    left, right = q0 - gap - 1, q0 + span + gap
    cells = {(base, q) for q in range(left, right + 1)}
    for qa, qb, length in ((left, left + 1, guards[0]), (q0, q0 + span, center),
                           (right, right + 1, guards[1])):
        cells |= {(base - branch * k, q) for k in range(1, length + 1) for q in range(qa, qb)}
    return cells


def _group(rng, kind, colors, span_hint, long_trip=False, compact=False, shared_count=2):
    """Compose a rail and its sockets in integer perpendicular diagonal coordinates."""
    tokens, forks = [], []
    branch0 = int(rng.choice((-1, 1)))
    if kind == 'finite':
        second = 3 - span_hint if span_hint in (1, 2) else int(rng.integers(1, 3))
        spans = [span_hint, second, int(rng.integers(1, 3))]
        margin = 2 if compact else int(rng.integers(2, 4))
        second_q = span_hint + (3 if compact else int(rng.integers(3, 5)))
        end = second_q + second - 1 + margin
        positions = [0, second_q, end + 3]
        enabled = [True, True, False]
    elif kind == 'shared':
        spans = [int(rng.integers(1, 3 if compact else 4)) for _ in range(shared_count)]
        positions = [0]
        for span in spans[:-1]:
            positions.append(positions[-1] + span + (3 if compact else int(rng.integers(3, 6))))
        enabled = [True] * shared_count
        end = positions[-1] + spans[-1] - 1 + (2 if compact else int(rng.integers(2, 4)))
    else:
        spans, positions, enabled = [span_hint], [0], [True]
        end = span_hint - 1 + (2 if compact else int(rng.integers(2, 5)))
    start = -(2 if compact else int(rng.integers(2, 5)))
    rail = {(0, q) for q in range(start, end + 1)}
    endpoints = {(0, start), (0, end)}
    for i, (q0, span, is_enabled) in enumerate(zip(positions, spans, enabled)):
        branch = branch0 * (-1 if i % 2 else 1)
        travel = (5 if compact else int(rng.integers(5, 8))) if long_trip else (
            2 if compact else int(rng.integers(2, 5)))
        if kind == 'contest' and i == 0:
            if not long_trip:
                travel = 3 if compact else int(rng.integers(3, 5))
            center, guards = 3, (1, 2)
        elif kind == 'finite' and i == 0:
            center, guards = 1, (1, 2)
        else:
            center = 1 if compact else int(rng.integers(1, 4))
            cap = min(5, travel + center - 1)
            guards = (1, 1) if compact else tuple(map(int, rng.integers(1, cap + 1, size=2)))
        gap = 1 if compact else int(rng.integers(1, 3))
        forks.append(_fork_local(q0, span, branch, travel, center, guards, gap))
        cells = {(0, q) for q in range(q0, q0 + span)}
        final = {(branch * travel, q) for q in range(q0, q0 + span)}
        route = {(k, q) for k in range(min(0, branch * travel), max(0, branch * travel) + 1)
                 for q in range(q0, q0 + span)}
        tokens.append(dict(cells=cells, color=int(rng.choice(colors)),
                           final=final if is_enabled else set(), route=route if is_enabled else set()))
        if kind == 'contest':
            # Same axis, opposite side, strictly wider middle prong: width,
            # rather than proximity or one fixed rail side, selects the socket.
            wrong_travel = 2 if compact else int(rng.integers(2, travel))
            forks.append(_fork_local(q0 - 1, span + 2, -branch, wrong_travel,
                                     1, (1, 1), gap))
    return dict(forks=forks, endpoints=endpoints, rails=[rail], tokens=tokens)


def _place_group(rng, group, rotation, canvas=None, offset=None):
    def turn(p):
        k, q = p
        r, c = k + q, k - q
        for _ in range(rotation):
            r, c = c, -r
        return r, c
    all_cells = set().union(*group['forks'], group['endpoints'], *group['rails'],
                           *(t['cells'] | t['route'] for t in group['tokens']))
    turned = [turn(p) for p in all_cells]
    lo_r, lo_c = min(r for r, c in turned), min(c for r, c in turned)
    bh = max(r for r, c in turned) - lo_r + 1
    bw = max(c for r, c in turned) - lo_c + 1
    if max(bh, bw) > 30:
        return None
    if canvas is None:
        h, w = int(rng.integers(bh, min(30, bh + 5) + 1)), int(rng.integers(bw, min(30, bw + 5) + 1))
    else:
        h, w = canvas
    if offset is None:
        rr, cc = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
    else:
        rr, cc = offset
        if rr < 0 or cc < 0 or rr + bh > h or cc + bw > w:
            return None
    def convert(cells):
        return {(r - lo_r + rr, c - lo_c + cc) for r, c in map(turn, cells)}
    placed = dict(forks=[convert(f) for f in group['forks']],
                  endpoints=convert(group['endpoints']), rails=[convert(s) for s in group['rails']],
                  tokens=[dict(cells=convert(t['cells']), final=convert(t['final']),
                               route=convert(t['route']), color=t['color']) for t in group['tokens']])
    return (h, w), placed


def _construct(shape, geometry, background, structure):
    grid = np.full(shape, background, dtype=np.int8)
    for fork in geometry['forks']:
        _paint(grid, fork, structure)
    _paint(grid, geometry['endpoints'], structure)
    for token in geometry['tokens']:
        _paint(grid, token['cells'], token['color'])
    output = np.full(shape, background, dtype=np.int8)
    for fork in geometry['forks']:
        _paint(output, fork, structure)
    _paint(output, geometry['endpoints'], structure)
    for rail in geometry['rails']:
        _paint(output, rail, structure)
    for token in geometry['tokens']:
        if token['route']:
            _paint(output, token['route'] - token['final'], TRAIL)
            _paint(output, token['final'], token['color'])
    return grid, output


def _consistent(shape, geometry, background, structure):
    grid, output = _construct(shape, geometry, background, structure)
    try:
        return np.array_equal(solve(grid), output)
    except ValueError:
        return False


def _decorate(rng, shape, geometry, background, structure, colors):
    """Optional finite rails, inert endpoints and bars; never required corner noise."""
    h, w = shape
    base_grid, _ = _construct(shape, geometry, background, structure)
    base_analysis = _analyze(base_grid)
    required = _witnesses(base_grid, base_analysis)
    # Preserve the actual original counterexamples, not merely a Boolean
    # witness that a later decoration might accidentally replace.
    protected = set().union(*(t['route'] for t in geometry['tokens']))
    for candidates in base_analysis['geometric'].values():
        for _, candidate in candidates:
            protected |= candidate['route']
    for match in base_analysis['matches']:
        token = base_analysis['tokens'][match['ti']]
        for fork in base_analysis['forks']:
            candidate = _candidate(fork, token, base_grid, background, exact_width=False)
            if candidate is not None:
                protected |= candidate['route']

    def acceptable(candidate):
        grid, output = _construct(shape, candidate, background, structure)
        try:
            analysis = _analyze(grid)
            witnesses = _witnesses(grid, analysis)
            return np.array_equal(_render(grid, analysis), output) and all(
                not need or have for need, have in zip(required, witnesses))
        except ValueError:
            return False
    for _ in range(int(rng.integers(0, 3))):
        for _ in range(24):
            slope = int(rng.choice((-1, 1)))
            steps = int(rng.integers(2, min(8, h, w)))
            r, c = int(rng.integers(h - steps)), int(rng.integers(w - steps))
            if slope < 0:
                c += steps
            rail = set(_segment((r, c), (r + steps, c + slope * steps)))
            grid, _ = _construct(shape, geometry, background, structure)
            occupied = set(map(tuple, np.argwhere(grid != background)))
            if rail & (occupied | protected) or {(r, c), (r + steps, c + slope * steps)} & _halo(occupied):
                continue
            candidate = dict(geometry, rails=geometry['rails'] + [rail],
                             endpoints=geometry['endpoints'] | {(r, c), (r + steps, c + slope * steps)})
            if acceptable(candidate):
                geometry = candidate
                break
    for _ in range(int(rng.integers(0, 3))):
        for _ in range(24):
            p = (int(rng.integers(h)), int(rng.integers(w)))
            grid, _ = _construct(shape, geometry, background, structure)
            occupied = set(map(tuple, np.argwhere(grid != background)))
            if p in _halo(occupied) or p in protected:
                continue
            if any(abs(p[0] - q[0]) == abs(p[1] - q[1]) for q in geometry['endpoints']):
                continue
            candidate = dict(geometry, endpoints=geometry['endpoints'] | {p})
            if acceptable(candidate):
                geometry = candidate
                break
    for _ in range(int(rng.integers(0, 4))):
        for _ in range(24):
            span, slope = int(rng.integers(1, 5)), int(rng.choice((-1, 1)))
            r, c = int(rng.integers(h - span + 1)), int(rng.integers(w - span + 1))
            if slope < 0:
                c += span - 1
            cells = {(r + t, c + slope * t) for t in range(span)}
            grid, _ = _construct(shape, geometry, background, structure)
            occupied = set(map(tuple, np.argwhere(grid != background)))
            if cells & (_halo(occupied) | protected):
                continue
            token = dict(cells=cells, route=set(), final=set(), color=int(rng.choice(colors)))
            candidate = dict(geometry, tokens=geometry['tokens'] + [token])
            if acceptable(candidate):
                geometry = candidate
                break
    return geometry


def _grid_key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def _fallback_pair(rng, kind, background, structure, colors, span_hint,
                   rotation, used, long_trip=False, shared_count=2):
    """Exhaust compact placements; at least 25 distinct translations fit.

    Every compact group has both bounding dimensions at most 26. A 30x30
    canvas therefore offers >=25 translations, versus <=13 earlier inputs.
    Compact groups retain the requested teaching witness and shift.
    """
    group = _group(rng, kind, colors, span_hint, long_trip, True, shared_count)
    checked = False
    for rr in range(30):
        for cc in range(30):
            placed = _place_group(rng, group, rotation, canvas=(30, 30), offset=(rr, cc))
            if placed is None:
                continue
            shape, geometry = placed
            if not checked:
                assert _consistent(shape, geometry, background, structure)
                checked = True
            grid, output = _construct(shape, geometry, background, structure)
            if _grid_key(grid) not in used:
                return dict(input=grid, output=output)
    raise AssertionError("Compact translation capacity exceeds the maximum puzzle size")


def _generate_pair(rng, kind, background, structure, colors, span_hint,
                   rotation, used, long_trip=False, shared_count=2):
    for attempt in range(64):
        compact = attempt >= 48
        group = _group(rng, kind, colors, span_hint, long_trip, compact, shared_count)
        placed = _place_group(rng, group, rotation)
        if placed is None:
            continue
        shape, geometry = placed
        if not _consistent(shape, geometry, background, structure):
            continue
        if not compact:
            geometry = _decorate(rng, shape, geometry, background, structure, colors)
        grid, output = _construct(shape, geometry, background, structure)
        if _grid_key(grid) not in used:
            return dict(input=grid, output=output)
    return _fallback_pair(rng, kind, background, structure, colors, span_hint,
                          rotation, used, long_trip, shared_count)


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(np.arange(1, 10))))
    structure = palette[0]
    colors = palette[1:1 + int(rng.integers(1, 4))]
    backgrounds = [c for c in palette if c != structure and c not in colors]
    constant_bg = bool(rng.integers(2))
    shift = int(rng.integers(2))  # More shared triggers, or greater travel.
    rotations = list(map(int, rng.permutation(4)))
    # One of each primary slope in the two indispensable teaching examples.
    rot0 = rotations[0]
    rot1 = (rot0 + int(rng.choice((1, 3)))) % 4
    spans = list(map(int, rng.permutation((1, 2))))
    specifications = [('contest', spans[0], rot0), ('finite', spans[1], rot1)]
    for _ in range(nt - 2):
        kind = str(rng.choice(('single', 'single', 'shared', 'contest', 'finite')))
        specifications.append((kind, int(rng.integers(1, 4)), int(rng.integers(4))))
    rng.shuffle(specifications)
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            if split == 'train':
                kind, span, rotation = specifications[i]
                long_trip, shared_count = False, 2
            else:
                kind = 'shared' if shift == 0 else str(rng.choice(('single', 'contest')))
                span, rotation = int(rng.integers(1, 4)), int(rng.integers(4))
                long_trip, shared_count = shift == 1, 3
            background = backgrounds[0] if constant_bg else int(rng.choice(backgrounds))
            pair = _generate_pair(rng, kind, background, structure, colors, span,
                                  rotation, used, long_trip, shared_count)
            used.add(_grid_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _witnesses(grid, analysis):
    # An off-span token which would dock if the finite gate were omitted.
    finite = any(ti not in analysis['active'] and candidates
                 for ti, candidates in analysis['geometric'].items())
    width = False
    for match in analysis['matches']:
        token = analysis['tokens'][match['ti']]
        accepting = analysis['forks'][match['fi']]
        for fi, fork in enumerate(analysis['forks']):
            candidate = _candidate(fork, token, grid, analysis['background'], exact_width=False)
            if (fork['span'] != len(token['cells']) and candidate is not None
                    and candidate['distance'] < match['distance']
                    and fork['out'] == tuple(-d for d in accepting['out'])):
                width = True
    return finite, width


def _metrics(grid):
    a = _analyze(grid)
    active_per_rail = [sum(m['ri'] == ri for m in a['matches']) for ri in range(len(a['rails']))]
    return dict(shape=list(np.asarray(grid).shape), background=a['background'], structure=a['structure'],
                token_palette=sorted({t['color'] for t in a['tokens']}),
                rails=len(a['rails']), rail_lengths=[len(r) for r in a['rails']],
                slopes=[1 if r[-1][1] > r[0][1] else -1 for r in a['rails']],
                forks=len(a['forks']), fork_widths=[f['width'] for f in a['forks']],
                spoke_lengths=[list(f['lengths']) for f in a['forks']],
                spans=[len(t['cells']) for t in a['tokens']], tokens=len(a['tokens']),
                active=len(a['active']), dockings=len(a['matches']),
                dockings_per_rail=active_per_rail,
                distances=[m['distance'] for m in a['matches']],
                directions=[list(a['forks'][m['fi']]['out']) for m in a['matches']],
                unpaired=len(a['unpaired']), witnesses=list(_witnesses(grid, a)))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs, roles, teaching_slopes, teaching_modes = set(), set(), set(), set()
    backgrounds, token_colors = set(), set()
    finite_witness = width_witness = False
    metrics = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, output = np.asarray(pair['input']), np.asarray(pair['output'])
            for g in (grid, output):
                assert g.ndim == 2 and np.issubdtype(g.dtype, np.integer)
                assert all(1 <= n <= 30 for n in g.shape) and np.all((0 <= g) & (g <= 9))
            assert grid.shape == output.shape
            assert _grid_key(grid) not in inputs
            inputs.add(_grid_key(grid))
            a = _analyze(grid)
            assert np.array_equal(output, _render(grid, a))
            assert not np.array_equal(grid, output) and a['matches']
            assert TRAIL in output and TRAIL not in grid
            roles.add(a['structure'])
            backgrounds.add(a['background'])
            token_colors.update(t['color'] for t in a['tokens'])
            assert all(int(output[p]) == a['structure'] for p in a['endpoints'])
            for f in a['forks']:
                assert min(f['lengths']) >= 1
                assert all(int(output[p]) == a['structure'] for p in f['cells'])
            for m in a['matches']:
                token = a['tokens'][m['ti']]
                assert len(m['final']) == len(token['cells'])
                assert all(int(output[p]) == token['color'] for p in m['final'])
                assert all(int(output[p]) == TRAIL for p in m['route'] - m['final'])
                fork = a['forks'][m['fi']]
                if split == 'train':
                    teaching_slopes.add(1 if a['rails'][m['ri']][-1][1] > a['rails'][m['ri']][0][1] else -1)
                    guards = (fork['lengths'][0], fork['lengths'][2])
                    center = fork['lengths'][1]
                    if center > max(guards):
                        teaching_modes.add('long_center')
                    if center < max(guards):
                        teaching_modes.add('long_guard')
            if split == 'train':
                finite, width = _witnesses(grid, a)
                finite_witness |= finite
                width_witness |= width
            metrics[split].append(_metrics(grid))
    assert len(roles) == 1
    assert backgrounds.isdisjoint(token_colors | roles)
    assert finite_witness and width_witness
    assert teaching_slopes == {-1, 1}
    assert teaching_modes == {'long_center', 'long_guard'}
    # Every generated puzzle has an observable, one-factor geometric shift.
    tr, te = metrics['train'], metrics['test']
    more_shared = min(max(m['dockings_per_rail']) for m in te) > max(max(m['dockings_per_rail']) for m in tr)
    longer_trip = min(min(m['distances']) for m in te) > max(max(m['distances']) for m in tr)
    assert more_shared or longer_trip


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed semantic, ablation, geometry and diversity checks."""
    records, keys = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        keys.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid = pair['input']
                a = _analyze(grid)
                assert np.array_equal(solve(grid.copy()), pair['output'])
                record = _metrics(grid)
                record.update(seed=seed, split=split)
                records.append(record)
                # Remove the B gate: any off-span geometrically matching token
                # now moves, changing both its final footprint and the trail.
                extra = [(ti, fi, candidate) for ti, candidates in a['geometric'].items()
                         if ti not in a['active'] for fi, candidate in candidates]
                if extra:
                    ti, fi, candidate = extra[0]
                    ablated = dict(a, matches=a['matches'] + [dict(candidate, ti=ti, fi=fi)])
                    assert not np.array_equal(_render(grid, ablated), pair['output'])
                # Remove A width matching: an axis-aligned competing fork
                # permits an additional, incompatible docking direction.
                if record['witnesses'][1]:
                    found_difference = False
                    for match in a['matches']:
                        token = a['tokens'][match['ti']]
                        for fi, fork in enumerate(a['forks']):
                            candidate = _candidate(fork, token, grid, a['background'], False)
                            if fork['span'] != len(token['cells']) and candidate is not None:
                                ablated = dict(a, matches=[m for m in a['matches'] if m['ti'] != match['ti']] +
                                               [dict(candidate, ti=match['ti'], fi=fi)])
                                found_difference |= not np.array_equal(_render(grid, ablated), pair['output'])
                    assert found_difference
    assert len(keys) == 42
    assert len({tuple(r['shape']) for r in records}) >= 60
    assert len({r['structure'] for r in records}) >= 7
    assert len({tuple(r['token_palette']) for r in records}) >= 20
    assert len({r['rails'] for r in records}) >= 3
    assert len({r['forks'] for r in records}) >= 3
    assert {1, 2, 3, 4} <= {s for r in records for s in r['spans']}
    assert {2, 3, 4, 5, 6, 7} <= {d for r in records for d in r['distances']}
    assert len({tuple(d) for r in records for d in r['directions']}) == 4
    assert any(len(set(r['slopes'])) == 2 for r in records)
    assert any(max(r['dockings_per_rail']) >= 3 for r in records)
    assert {0, 1, 2} <= {r['unpaired'] for r in records}
    assert _puzzle_key(generate_puzzle(765, 5, 3)) == _puzzle_key(generate_puzzle(765, 5, 3))
    # Force the finite fallback rather than relying on ordinary random seeds
    # to exhaust the normal retry budget. Reuse identical geometry each time.
    used = set()
    for _ in range(14):
        pair = _fallback_pair(np.random.default_rng(2026), 'finite', 7, 3, [2], 3, 1, used)
        used.add(_grid_key(pair['input']))
        assert np.array_equal(solve(pair['input']), pair['output'])
        assert _witnesses(pair['input'], _analyze(pair['input']))[0]
    assert len(used) == 14
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts were accepted')
    bad = generate_puzzle(11, 2, 1)
    a = _analyze(bad['test'][0]['input'])
    p = min(a['matches'][0]['final'])
    bad['test'][0]['output'][p] = TRAIL
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('A corrupted final token was accepted')
    return dict(seeds=42, pairs=len(records), shapes=len({tuple(r['shape']) for r in records}),
                structural_colors=len({r['structure'] for r in records}),
                token_palettes=len({tuple(r['token_palette']) for r in records}),
                rail_counts=sorted({r['rails'] for r in records}),
                fork_counts=sorted({r['forks'] for r in records}),
                shared_docking_counts=sorted({n for r in records for n in r['dockings_per_rail']}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
