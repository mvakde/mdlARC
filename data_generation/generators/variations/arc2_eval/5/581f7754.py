"""Boundary-guided rigid translation, with input-visible local registration roles.

The lone source train-2 along-guide displacement is not part of this rule.
Construction uses placement records; validation reconstructs everything from grids.
"""
import hashlib
import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SOURCE_DIGESTS = frozenset((
    'cdf50f481670d61a0735e703ec3cec4ba2a5aae183141be40b5aeacf7feebbcd',
    'be43d71caf8eceffd644ecdaea8101871e639fd9443d16e19b47532c8c93150c',
    '164cc43f140e509530fb810ebdfa0aba60b79ea2a2b79bad69da53f9775b7146',
    'f6ea3275740271b3d815f751ee1576de4c5b6ce67b18fe8b08a8683b2cff7653',
    'e432ed6e11253412f372a13e880a290bad91cf88d5129997e200f2c1fdbc69bc',
))


def _require(condition, message='invalid generated puzzle'):
    if not condition:
        raise AssertionError(message)


def _normalise_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return int(value)


def _select_counts(rng, num_train, num_test):
    num_train = _normalise_count(num_train, 'num_train', 2, 10)
    num_test = _normalise_count(num_test, 'num_test', 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError('num_train must exceed num_test')
    elif num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _grid_signature(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _pair_digest(pair):
    a, b = pair['input'], pair['output']
    return hashlib.sha256(str(a.shape).encode() + np.asarray(a, dtype=np.int8).tobytes()
                          + np.asarray(b, dtype=np.int8).tobytes()).hexdigest()


def _component_sets(grid, background):
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                neighbor = r + dr, c + dc
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _edge_sides(points, height, width):
    return {side for r, c in points for side, hit in
            (('top', r == 0), ('bottom', r == height - 1),
             ('left', c == 0), ('right', c == width - 1)) if hit}


def _decode_input(grid):
    """No output or construction metadata is used to identify movable objects."""
    height, width = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    _require(int(counts.max()) > grid.size / 2, 'background must be distinguishable')
    components = _component_sets(grid, background)
    anchors, carriers = {}, []
    for component in components:
        if len(component) == 1 and _edge_sides(component, height, width):
            position = next(iter(component))
            anchors.setdefault(int(grid[position]), set()).add(position)
        else:
            carriers.append(component)
    _require(anchors, 'no isolated boundary anchors')
    systems = {}
    for color, points in anchors.items():
        rows, cols = {r for r, _ in points}, {c for _, c in points}
        horizontal = len(rows) == 1 and all(c in (0, width - 1) for _, c in points)
        vertical = len(cols) == 1 and all(r in (0, height - 1) for r, _ in points)
        _require(horizontal != vertical, 'ambiguous or inconsistent guide')
        axis = 0 if horizontal else 1  # coordinate changed by projection
        guide = next(iter(rows if horizontal else cols))
        _require(0 < guide < grid.shape[axis] - 1, 'corner anchors are ambiguous')
        systems[color] = {'axis': axis, 'guide': guide, 'anchors': points,
                          'sides': _edge_sides(points, height, width)}
    movers = []
    for component in carriers:
        colors = {color: sum(int(grid[p]) == color for p in component) for color in systems}
        candidates = [color for color, count in colors.items() if count == 1]
        _require(len(candidates) == 1, 'carrier has no unique local registration cell')
        marker = candidates[0]
        registration = next(p for p in component if int(grid[p]) == marker)
        system = systems[marker]
        delta = [0, 0]
        delta[system['axis']] = system['guide'] - registration[system['axis']]
        target = {(r + delta[0], c + delta[1]): int(grid[r, c]) for r, c in component}
        outside = {p for p in target if not (0 <= p[0] < height and 0 <= p[1] < width)}
        clip_sides = {s for r, c in outside for s, hit in
                      (('top', r < 0), ('bottom', r >= height),
                       ('left', c < 0), ('right', c >= width)) if hit}
        movers.append({'points': component, 'marker': marker, 'registration': registration,
                       'delta': tuple(delta), 'target': target, 'outside': outside,
                       'clip_sides': clip_sides, 'edge_sides': _edge_sides(component, height, width),
                       'role_reuse': any(n > 1 for n in colors.values())})
    _require(all(any(m['marker'] == color for m in movers) for color in systems),
             'anchor has no matching component')
    return background, systems, movers


def _prediction(grid):
    background, systems, movers = _decode_input(grid)
    result = np.full_like(grid, background)
    occupied = set()
    for color, system in systems.items():
        for p in system['anchors']:
            result[p] = color
            occupied.add(p)
    for mover in movers:
        for p, color in mover['target'].items():
            if p in mover['outside']:
                continue
            _require(p not in occupied, 'projected footprints overlap an object or anchor')
            occupied.add(p)
            result[p] = color
    return result


def solve(train_pairs, test_input):
    """Training-only rule check followed by a prediction from the test input.

    This is a local diagnostic solver, not an independent protected reference.
    """
    _require(train_pairs, 'no demonstrations')
    for pair in train_pairs:
        _require(np.array_equal(_prediction(pair['input']), pair['output']),
                 'demonstrations do not obey rigid projection')
    return _prediction(test_input)


def _has_hole(points):
    r0, r1 = min(r for r, _ in points) - 1, max(r for r, _ in points) + 1
    c0, c1 = min(c for _, c in points) - 1, max(c for _, c in points) + 1
    empty = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)} - points
    reached, stack = {(r0, c0)}, [(r0, c0)]
    while stack:
        r, c = stack.pop()
        for dr, dc in _DIRS:
            p = r + dr, c + dc
            if p in empty and p not in reached:
                reached.add(p)
                stack.append(p)
    return len(reached) < len(empty)


def _analyze_pair(input_grid, output_grid):
    _require(input_grid.shape == output_grid.shape, 'input/output dimensions differ')
    background, systems, movers = _decode_input(input_grid)
    _require(np.array_equal(_prediction(input_grid), output_grid),
             'output is not simultaneous rigid projection')
    _require(not np.array_equal(input_grid, output_grid), 'unchanged scene')
    full_witness = any(len(m['points']) > 1 and m['delta'] != (0, 0) and not m['outside']
                       for m in movers)
    reciprocal = set()
    for m in movers:
        for p in m['points']:
            color = int(input_grid[p])
            if color in systems and color != m['marker']:
                reciprocal.add((m['marker'], color))
    return {'background': background, 'systems': systems, 'movers': movers,
            'reciprocal': any((b, a) in reciprocal for a, b in reciprocal),
            'role_reuse': any(m['role_reuse'] for m in movers),
            'full_witness': full_witness,
            'mixed': len({s['axis'] for s in systems.values()}) > 1,
            'clipped': sum(bool(m['outside']) for m in movers),
            'singletons': sum(len(m['points']) == 1 for m in movers)}


def pair_features(pair):
    """Review metrics measured from returned grids, not sampled parameters."""
    grid = pair['input']
    s = _analyze_pair(grid, pair['output'])
    systems, movers = s['systems'], s['movers']
    carriers = [m for m in movers if len(m['points']) > 1]
    sizes = [len(m['points']) for m in movers]
    body_sets, decorations, spans, mark_degrees, asymmetry = [], [], [], [], []
    for m in carriers:
        points, mark = m['points'], m['registration']
        body = [int(grid[p]) for p in points if p != mark]
        body_sets.append(tuple(sorted(set(body))))
        decorations.append(len(set(body)) - 1)
        spans.append((max(r for r, _ in points) - min(r for r, _ in points) + 1,
                      max(c for _, c in points) - min(c for _, c in points) + 1))
        mark_degrees.append(sum((mark[0] + dr, mark[1] + dc) in points for dr, dc in _DIRS))
        offsets = {(r - mark[0], c - mark[1], int(grid[r, c])) for r, c in points}
        # Any failure of a nontrivial D4 symmetry makes rigid orientation observable.
        symmetric = all({(r * a + c * b, r * d + c * e, v) for r, c, v in offsets} == offsets
                        for a, b, d, e in ((0, -1, 1, 0), (-1, 0, 0, 1), (1, 0, 0, -1)))
        asymmetry.append(not symmetric)
    singleton_only = sum(all(len(m['points']) == 1 for m in movers if m['marker'] == color)
                         for color in systems)
    nonconvex = sum(any((r, c1) in m['points'] and (r, c2) in m['points']
                       and any((r, c) not in m['points'] for c in range(c1, c2))
                       for r, c1 in m['points'] for rr, c2 in m['points'] if rr == r and c1 < c2)
                   for m in carriers)
    return {'shape': list(grid.shape), 'background': s['background'],
            'palette': [int(c) for c in np.unique(grid)], 'markers': sorted(systems),
            'systems': len(systems), 'axes': sorted({x['axis'] for x in systems.values()}),
            'mixed': s['mixed'], 'guides': [[x['axis'], x['guide']] for x in systems.values()],
            'sides': sorted(set().union(*(x['sides'] for x in systems.values()))),
            'bilateral': sum(len(x['anchors']) == 2 for x in systems.values()),
            'movers': len(movers), 'movers_per_system': [sum(m['marker'] == c for m in movers) for c in systems],
            'sizes': sizes, 'spans': [list(x) for x in spans], 'singletons': s['singletons'],
            'singleton_only': singleton_only, 'clipped': s['clipped'],
            'clip_sides': sorted(set().union(*(m['clip_sides'] for m in movers))),
            'boundary_carriers': sum(bool(m['edge_sides']) for m in carriers),
            'boundary_sides': sorted(set().union(*(m['edge_sides'] for m in carriers))),
            'aligned': sum(m['delta'] == (0, 0) for m in movers),
            'displacements': [list(m['delta']) for m in movers],
            'reciprocal': s['reciprocal'], 'role_reuse': s['role_reuse'],
            'holes': sum(_has_hole(m['points']) for m in carriers), 'nonconvex': nonconvex,
            'branching': sum(any(sum((r + dr, c + dc) in m['points'] for dr, dc in _DIRS) >= 3
                                 for r, c in m['points']) for m in carriers),
            'simple_bodies': sum(n == 0 for n in decorations), 'decorations': decorations,
            'body_subsets': [list(x) for x in body_sets], 'mark_degrees': mark_degrees,
            'asymmetric': sum(asymmetry), 'full_witness': s['full_witness'],
            'occupancy': float(np.mean(grid != s['background']))}


def _validate_grid(grid):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a 2D NumPy array')
    _require(all(1 <= n <= 30 for n in grid.shape), 'illegal dimensions')
    _require(np.issubdtype(grid.dtype, np.integer), 'grid must contain integers')
    _require(bool(np.all((grid >= 0) & (grid <= 9))), 'illegal colors')


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal pair counts')
    seen, stats = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            for grid in pair.values():
                _validate_grid(grid)
            signature = _grid_signature(pair['input'])
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            _require(_pair_digest(pair) not in _SOURCE_DIGESTS, 'exact source pair copy')
            stats[split].append(_analyze_pair(pair['input'], pair['output']))
    _require(any(s['full_witness'] for s in stats['train']), 'no complete moving carrier demonstration')
    if any(s['role_reuse'] for s in stats['test']):
        _require(any(s['reciprocal'] and s['full_witness'] for s in stats['train']),
                 'test local color roles are not demonstrated reciprocally')
    if any(s['mixed'] for s in stats['test']):
        _require(any(s['mixed'] for s in stats['train']), 'mixed systems lack a training demonstration')
    # Axes, sides, palettes, counts, and clipping are input-visible pair properties.
    # Neither split has the artificial layouts/palettes of the reviewed generator.
    for pair in puzzle['test']:
        _require(np.array_equal(solve(puzzle['train'], pair['input']), pair['output']),
                 'test prediction failed')


def sample_puzzle_parameters(rng):
    return {'palette': tuple(map(int, rng.permutation(10))),
            'stable_palette': bool(rng.random() < .25),
            'axis_mode': int(rng.choice(3, p=[.27, .18, .55])),
            'base_axis': int(rng.integers(2)),
            'reciprocity': bool(rng.random() < .65),
            'training_clipping': bool(rng.random() < .55)}


def _shape(rng, scale):
    """Compositional shape families with variable lengths, holes and branches."""
    kind = int(rng.integers(6))
    if kind == 0:  # connected polyomino, with a finite frontier at each step
        size = int(rng.integers(2, max(4, scale + 1)))
        cells = {(0, 0)}
        for _ in range(size - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    elif kind in (1, 2, 3):
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 7))
        if kind == 1:
            cells = {(r, c) for r in range(h) for c in range(w)}
        else:
            h, w = max(3, h), max(3, w)
            cells = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h - 1) or c in (0, w - 1)}
            if kind == 3:  # open frame, with a random notch
                cut = int(rng.integers(1, w - 1))
                cells.remove((0, cut))
        if rng.random() < .5:
            r = int(rng.integers(h))
            for j in range(1, int(rng.integers(1, 4))):
                cells.add((r, w - 1 + j))
    else:  # elongated spines with independently sized lateral branches
        length = int(rng.integers(3, 10))
        cells = {(r, 0) for r in range(length)}
        for r in rng.choice(length, size=int(rng.integers(1, min(length, 5) + 1)), replace=False):
            sign = int(rng.choice([-1, 1]))
            for j in range(1, int(rng.integers(1, 4))):
                cells.add((int(r), sign * j))
    # No orientation is reserved for a split or motif family.
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    return sorted(cells)


def _colored_shape(rng, marker, markers, payload, scale, body=None, singleton=False,
                   axis=0, guide=1, axis_size=20, clip=False, eccentric=False):
    if singleton:
        return [(0, 0, marker)]
    points = _shape(rng, scale)
    if body in markers and len(points) < 3:
        return None
    candidates = []
    for r, c in points:
        mark = r, c
        projected = [p[axis] - mark[axis] + guide for p in points]
        if (min(projected) < 0 or max(projected) >= axis_size) != clip:
            continue
        if eccentric and all(2 * mark[k] == min(p[k] for p in points) + max(p[k] for p in points)
                             for k in (0, 1)):
            continue
        candidates.append(mark)
    if not candidates:
        return None
    mark = candidates[int(rng.integers(len(candidates)))]
    if body is None:
        body = int(rng.choice(payload))
    colors = {p: body for p in points}
    others = [p for p in points if p != mark]
    # Most bodies are uniform; decorations are sparse and vary locally.
    if len(others) >= 4 and rng.random() < .42:
        alternatives = list(map(int, rng.permutation([c for c in payload if c != body])))
        limit = min(3, len(others) // 3, len(alternatives))
        if limit:
            n = int(rng.integers(1, limit + 1))
            for j in range(n):
                p = others[int(rng.integers(len(others)))]
                colors[p] = alternatives[j]
    # An anchored body color must repeat, leaving one and only one singleton identity.
    if body in markers and sum(v == body for p, v in colors.items() if p != mark) < 2:
        return None
    colors[mark] = marker
    return [(r - mark[0], c - mark[1], int(colors[r, c])) for r, c in points]


def _anchors(height, width, axis, guide, ends):
    if axis == 0:
        return {(guide, 0 if end == 0 else width - 1) for end in ends}
    return {(0 if end == 0 else height - 1, guide) for end in ends}


def _place(rng, shape, system, inp, out, background, aligned=False, edge=False):
    """Enumerate all legal origins, then sample one; no blind placement retry loop."""
    if shape is None:
        return False
    h, w = inp.shape
    offsets = np.array([(r, c) for r, c, _ in shape], dtype=int)
    low, high = offsets.min(axis=0), offsets.max(axis=0)
    if high[0] - low[0] >= h or high[1] - low[1] >= w:
        return False
    rr, cc = np.meshgrid(np.arange(-low[0], h - high[0]),
                         np.arange(-low[1], w - high[1]), indexing='ij')
    rr, cc = rr.ravel(), cc.ravel()
    if not len(rr):
        return False
    axis, guide = system['axis'], system['guide']
    valid = ((rr if axis == 0 else cc) == guide) if aligned else ((rr if axis == 0 else cc) != guide)
    if len(shape) == 1:
        valid &= (rr > 0) & (rr < h - 1) & (cc > 0) & (cc < w - 1)
    blocked = inp != background
    blocked = (blocked | np.pad(blocked[1:], ((0, 1), (0, 0)))
               | np.pad(blocked[:-1], ((1, 0), (0, 0)))
               | np.pad(blocked[:, 1:], ((0, 0), (0, 1)))
               | np.pad(blocked[:, :-1], ((0, 0), (1, 0))))
    for dr, dc, _ in shape:
        valid &= ~blocked[rr + dr, cc + dc]
        tr = np.full_like(rr, guide + dr) if axis == 0 else rr + dr
        tc = cc + dc if axis == 0 else np.full_like(cc, guide + dc)
        inside = (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)
        valid &= ~inside | (out[np.clip(tr, 0, h - 1), np.clip(tc, 0, w - 1)] == background)
    choices = np.flatnonzero(valid)
    if edge:
        touches = ((rr + low[0] == 0) | (rr + high[0] == h - 1)
                   | (cc + low[1] == 0) | (cc + high[1] == w - 1))
        edge_choices = np.flatnonzero(valid & touches)
        if len(edge_choices):
            choices = edge_choices
    if not len(choices):
        return False
    k = int(rng.choice(choices))
    r, c = int(rr[k]), int(cc[k])
    delta = (guide - r, 0) if axis == 0 else (0, guide - c)
    for dr, dc, color in shape:
        inp[r + dr, c + dc] = color
        tr, tc = r + dr + delta[0], c + dc + delta[1]
        if 0 <= tr < h and 0 <= tc < w:
            out[tr, tc] = color
    return True


def _dimensions(rng, min_systems, challenge):
    kind = int(rng.integers(5))
    if kind == 0:
        h, w = int(rng.integers(7, 15)), int(rng.integers(7, 17))
    elif kind == 1:
        h, w = int(rng.integers(7, 12)), int(rng.integers(18, 31))
        if rng.random() < .5:
            h, w = w, h
    elif kind == 2:
        h, w = int(rng.integers(12, 24)), int(rng.integers(12, 25))
    else:
        h, w = int(rng.integers(17, 31)), int(rng.integers(17, 31))
    # Increase available area for compulsory composition, not for a fixed split template.
    if min_systems >= 2 and h * w < 135:
        h += 3
        w += 3
    if challenge and rng.random() < .10:
        h = w = 30
    return h, w


def _scene(rng, params, plan):
    palette = params['palette'] if params['stable_palette'] else tuple(map(int, rng.permutation(10)))
    bg = palette[0]
    minimum = 2 if plan['reciprocal'] or plan['mixed'] else 1
    h, w = _dimensions(rng, minimum, plan['test'])
    max_systems = min(4, max(minimum, h * w // 95 + 1))
    n = int(rng.integers(minimum, max_systems + 1))
    markers = palette[1:n + 1]
    payload = palette[5:] if params['stable_palette'] else palette[n + 1:]
    inp, out = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
    axes = [plan['axis']] * n
    if plan['mixed']:
        axes = [0, 1] + list(map(int, rng.integers(2, size=n - 2)))
        rng.shuffle(axes)
    systems = []
    # Distribute clips independently over systems. Several edges can participate
    # in the same scene; marking one or both ends is an independent draw.
    clips_by_system = [0] * n
    for _ in range(plan['clips']):
        clips_by_system[int(rng.integers(n))] += 1
    for j, marker in enumerate(markers):
        axis = axes[j]
        limit = (h, w)[axis]
        guide_options = list(range(1, limit - 1))
        rng.shuffle(guide_options)
        if clips_by_system[j]:
            near = int(rng.integers(1, min(4, limit - 2)))
            guide_options = [near if rng.random() < .5 else limit - 1 - near]
        ends = (0, 1) if rng.random() < .3 else (int(rng.integers(2)),)
        for guide in guide_options:
            points = _anchors(h, w, axis, guide, ends)
            if any(s['axis'] == axis and s['guide'] == guide for s in systems):
                continue
            if any(inp[r, c] != bg or any(0 <= r + dr < h and 0 <= c + dc < w
                                          and inp[r + dr, c + dc] != bg for dr, dc in _DIRS)
                   for r, c in points):
                continue
            system = {'marker': marker, 'axis': axis, 'guide': guide, 'anchors': points}
            systems.append(system)
            for p in points:
                inp[p] = out[p] = marker
            break
        else:
            return None
    counts = [int(rng.integers(1, 5)) for _ in systems]
    # More room yields more material; small scenes still contain substantial carriers.
    scale = int(np.clip(h * w * rng.uniform(.13, .28) / sum(counts), 4, 25))
    singleton_only = [bool(rng.random() < .17) for _ in systems]
    for j in range(n):
        if clips_by_system[j] or plan['reciprocal'] and j < 2:
            singleton_only[j] = False
        counts[j] = max(counts[j], clips_by_system[j])
    requests = []
    for j in range(n):
        for k in range(counts[j]):
            clipped = k < clips_by_system[j]
            single = singleton_only[j] or (k > 0 and not clipped and rng.random() < .25)
            body = markers[1 - j] if plan['reciprocal'] and j < 2 and k == 0 else None
            if body is not None:
                single = False
            requests.append({'j': j, 'single': single, 'clip': clipped, 'body': body,
                             'required': k == 0 or clipped, 'witness': False})
    # An explicit complete moving witness distinguishes rigid movement from marker-only movement.
    witness = next((r for r in requests if not r['single'] and not r['clip']), None)
    if witness is None:
        witness = {'j': 0, 'single': False, 'clip': False, 'body': None,
                   'required': True, 'witness': True}
        requests.append(witness)
    witness['witness'] = witness['required'] = True
    requests.sort(key=lambda r: (not r['required'], not r['clip']))
    for request in requests:
        j = request['j']
        system = systems[j]
        aligned = bool(not request['required'] and not request['clip'] and rng.random() < .16)
        edge = bool(not request['single'] and rng.random() < .28)
        placed = False
        for attempt in range(24 if request['required'] else 7):
            shape = _colored_shape(rng, markers[j], markers, payload, scale,
                                   body=request['body'], singleton=request['single'],
                                   axis=system['axis'], guide=system['guide'],
                                   axis_size=(h, w)[system['axis']], clip=request['clip'],
                                   eccentric=request['witness'])
            if _place(rng, shape, system, inp, out, bg, aligned=aligned, edge=edge):
                placed = True
                break
        if not placed and request['required']:
            return None
    if np.mean(inp == bg) <= .5:
        return None
    pair = {'input': inp, 'output': out}
    stats = _analyze_pair(inp, out)
    if not stats['full_witness'] or plan['reciprocal'] and not stats['reciprocal']:
        return None
    return pair


def _fallback(params, plan, variant):
    """Guaranteed spacious layout after bounded sampling; same visible conventions.

    Four disjoint lanes leave ample room for three clipped carriers and a complete
    carrier. A second system demonstrates reciprocal or mixed roles when needed.
    """
    palette = params['palette']
    bg, a, b, body = palette[:4]
    inp, out = np.full((30, 30), bg, np.int8), np.full((30, 30), bg, np.int8)
    two = plan['reciprocal'] or plan['mixed']
    systems = [{'marker': a, 'axis': 0, 'guide': 1, 'anchors': {(1, 0)}}]
    if two:
        systems.append({'marker': b, 'axis': 1 if plan['mixed'] else 0,
                        'guide': 28, 'anchors': {(29, 28)} if plan['mixed'] else {(28, 29)}})
    for s in systems:
        for p in s['anchors']:
            inp[p] = out[p] = s['marker']
    # variant changes the complete input even if all random attempts were exhausted.
    for k in range(max(1, plan['clips']) + 1):
        clipped = k < plan['clips']
        mark = (2, 1) if clipped else (0, 1)
        origin = 6 + variant % 14, 2 + 6 * k
        color = b if plan['reciprocal'] and k == 0 else body
        for r in range(3):
            for c in range(3):
                value = a if (r, c) == mark else color
                inp[origin[0] + r, origin[1] + c] = value
                tr = 1 + r - mark[0]
                if tr >= 0:
                    out[tr, origin[1] + c] = value
    if two:
        s = systems[1]
        origin = (25, 12)
        mark = (1, 1) if s['axis'] else (2, 1)
        for r in range(3):
            for c in range(3):
                value = b if (r, c) == mark else (a if plan['reciprocal'] else body)
                inp[origin[0] + r, origin[1] + c] = value
                tr = origin[0] + r if s['axis'] else 28 + r - mark[0]
                tc = 28 + c - mark[1] if s['axis'] else origin[1] + c
                if 0 <= tr < 30 and 0 <= tc < 30:
                    out[tr, tc] = value
    if plan['axis'] == 1 and not plan['mixed']:
        inp, out = inp.T.copy(), out.T.copy()
    return {'input': inp, 'output': out}


def generate_pair(rng, regime, puzzle_parameters, pair_index=0, plan=None):
    if regime not in ('source_train', 'target_test'):
        raise ValueError('unknown regime')
    if plan is None:
        plan = _plan(rng, puzzle_parameters, regime == 'target_test')
    for _ in range(36):
        pair = _scene(rng, puzzle_parameters, plan)
        if pair is not None:
            return pair['input'], pair['output']
    pair = _fallback(puzzle_parameters, plan, pair_index)
    return pair['input'], pair['output']


def _plan(rng, params, test):
    mode = params['axis_mode']
    return {'test': test, 'axis': (1 - params['base_axis'] if test and mode == 1 else
                                 int(rng.integers(2)) if mode == 2 else params['base_axis']),
            'mixed': bool(mode == 2 and rng.random() < .38),
            'reciprocal': bool(params['reciprocity'] and rng.random() < .4),
            'clips': (int(rng.choice(4, p=[.43, .34, .18, .05])) if test else
                      int(rng.choice(3, p=[.64, .28, .08])) if params['training_clipping'] else 0)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    a, b = _select_counts(rng, num_train, num_test)
    params = sample_puzzle_parameters(rng)
    plans = [_plan(rng, params, i >= a) for i in range(a + b)]
    demo = int(rng.integers(a))
    plans[demo]['reciprocal'] = params['reciprocity']
    plans[demo]['mixed'] = params['axis_mode'] == 2
    plans[demo]['clips'] = 0
    if params['training_clipping']:
        clip_demo = (demo + int(rng.integers(1, a))) % a
        plans[clip_demo]['clips'] = int(rng.integers(1, 3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, plan in enumerate(plans):
        split = 'train' if i < a else 'test'
        for retry in range(12):
            inp, out = generate_pair(rng, 'source_train' if split == 'train' else 'target_test',
                                     params, i + retry, plan)
            signature = _grid_signature(inp)
            pair = {'input': inp, 'output': out}
            if signature not in seen and _pair_digest(pair) not in _SOURCE_DIGESTS:
                break
        else:
            # There are at most 13 previous inputs and 14 distinct valid layouts.
            for variant in range(14):
                pair = _fallback(params, plan, variant)
                signature = _grid_signature(pair['input'])
                if signature not in seen:
                    break
        seen.add(signature)
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_fingerprint(puzzle):
    return tuple(tuple((_grid_signature(p['input']), _grid_signature(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts, fingerprints, layouts, dimensions = set(), set(), set(), set(), set()
    features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        fingerprints.add(_puzzle_fingerprint(puzzle))
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = pair_features(pair)
                f['split'], f['seed'] = split, seed
                features.append(f)
                local.append((tuple(f['shape']), f['systems'], tuple(f['sizes']), f['clipped'], tuple(f['axes'])))
                dimensions.add(tuple(f['shape']))
        layouts.add(tuple(local))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, 'count diversity')
    _require(len(fingerprints) == 42 and len(layouts) >= 40 and len(dimensions) >= 80, 'structural diversity')
    for split in ('train', 'test'):
        fs = [f for f in features if f['split'] == split]
        _require({a for f in fs for a in f['axes']} == {0, 1}, 'axis coverage')
        _require(set().union(*(set(f['sides']) for f in fs)) == {'top', 'bottom', 'left', 'right'}, 'anchor sides')
        _require(set().union(*(set(f['clip_sides']) for f in fs)) == {'top', 'bottom', 'left', 'right'}, 'clip sides')
        _require({0, 1, 2} <= {f['clipped'] for f in fs}, 'clipping multiplicity')
        _require({1, 2, 3} <= {f['systems'] for f in fs}, 'guide counts')
        for name in ('mixed', 'reciprocal', 'boundary_carriers', 'singleton_only', 'aligned', 'holes'):
            _require(sum(bool(f[name]) for f in fs) >= 3, f'{split} lacks {name}')
    _require(max(n for f in features for n in f['sizes']) >= 20, 'large carriers')
    _require({1, 2, 3, 4} <= {n for f in features for n in f['sizes']}, 'small carriers')
    _require(max(n for f in features for span in f['spans'] for n in span) >= 9, 'elongation')
    _require({1, 2, 3, 4} <= {n for f in features for n in f['mark_degrees']}, 'registration placement')
    _require(sum(f['simple_bodies'] for f in features) >= 150, 'simple bodies')
    _require(max(n for f in features for n in f['decorations']) >= 2, 'local decoration variety')
    _require(max(f['occupancy'] for f in features) >= .2, 'dense scenes')
    _require(any(min(f['shape']) <= 9 and max(f['shape']) >= 20 for f in features), 'thin scenes')
    _require(any(f['shape'] == [30, 30] for f in features), 'large canvas')
    _require(any(f['bilateral'] and not f['clipped'] for f in features)
             and any(not f['bilateral'] and f['clipped'] for f in features), 'bilateral/clipping coupling')
    varied_palettes = varied_counts = novel_markers = held_out_clips = 0
    same_axis = rotated_test = 0
    for seed in range(42):
        fs = [f for f in features if f['seed'] == seed]
        varied_palettes += len({f['background'] for f in fs}) > 1
        varied_counts += len({f['systems'] for f in fs}) > 1
        train = [f for f in fs if f['split'] == 'train']
        test = [f for f in fs if f['split'] == 'test']
        train_markers = {c for f in train for c in f['markers']}
        novel_markers += bool({c for f in test for c in f['markers']} - train_markers)
        held_out_clips += not any(f['clipped'] for f in train) and any(f['clipped'] for f in test)
        train_axes = {c for f in train for c in f['axes']}
        test_axes = {c for f in test for c in f['axes']}
        same_axis += len(train_axes | test_axes) == 1
        rotated_test += len(train_axes) == len(test_axes) == 1 and train_axes != test_axes
    _require(varied_palettes >= 20 and varied_counts >= 25, 'within-puzzle variation')
    _require(varied_palettes <= 39 and novel_markers >= 5, 'stable palettes or novel identities missing')
    _require(held_out_clips >= 3 and same_axis >= 3 and rotated_test >= 3, 'puzzle regime coverage')
    _require(any(len(f['clip_sides']) > 1 for f in features), 'multiple clipping edges in one scene')
    _require(sum(f['reciprocal'] and f['mixed'] and f['boundary_carriers'] > 0
                 for f in features) >= 5, 'combined variation coverage')
    _require(any(tuple(f['shape']) in {(18, 8), (10, 21), (22, 11), (14, 20), (11, 22)}
                 for f in features), 'original dimensions have been excluded')
    _require({f['background'] for f in features} == set(range(10)), 'background colors')
    _require(sum(f['nonconvex'] for f in features) >= 50
             and sum(f['branching'] for f in features) >= 50, 'structured geometry coverage')
    _require(sum(len({tuple(x) for x in f['body_subsets']}) > 1 for f in features) >= 100,
             'body color subsets are too uniform')
    for seed, a, b in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1101, None, 1),
                       (1102, None, 4), (1103, 2, None), (1104, 10, None)):
        p = generate_puzzle(seed, a, b)
        _require(a is None or len(p['train']) == a, 'explicit train count')
        _require(b is None or len(p['test']) == b, 'explicit test count')
        _require(_puzzle_fingerprint(p) == _puzzle_fingerprint(generate_puzzle(seed, a, b)), 'determinism')
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(99, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')
    # Semantic metamorphisms: palette reassignment per pair and global rotations
    # change no invariant; this specifically guards against the reviewed validator.
    p = generate_puzzle(2026, 3, 1)
    rng = np.random.default_rng(314)
    for turns in (1, 2, 3):
        q = {'train': [], 'test': []}
        for split in q:
            for pair in p[split]:
                colors = rng.permutation(10)
                q[split].append({k: colors[np.rot90(v, turns)].astype(np.int8) for k, v in pair.items()})
        validate_generated_puzzle(q)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    # Directly exercise the bounded fallback, including reciprocal mixed systems.
    params = sample_puzzle_parameters(rng)
    for axis in (0, 1):
        for mixed in (False, True):
            for clips in range(4):
                plan = {'axis': axis, 'mixed': mixed, 'reciprocal': True, 'clips': clips}
                signatures = set()
                for variant in (0, 13):
                    pair = _fallback(params, plan, variant)
                    f = pair_features(pair)
                    _require(f['clipped'] == clips and f['mixed'] == mixed
                             and f['reciprocal'] and f['full_witness'], 'fallback conventions')
                    signatures.add(_grid_signature(pair['input']))
                _require(len(signatures) == 2, 'fallback uniqueness')
    # A repeated anchored body is legal; two singleton anchored colors are not.
    ambiguous = np.zeros((9, 9), np.int8)
    ambiguous[2, 0], ambiguous[6, 0] = 1, 2
    ambiguous[4, 3:6] = [1, 3, 2]
    try:
        _decode_input(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous local registration accepted')
