"""Phase-gated docking: a cyclic rail selects rigid registration projections.

Only standard-library modules and NumPy are used. No files are read or written.
The rail is traced and all docking choices are reconstructed from the input.
"""
import hashlib
import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_EVIDENCE = {'cycling', 'gating', 'multiplicity', 'bend_phase', 'nearest_match',
             'rigid', 'registration'}


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError('counts must be integers')
    a, b = int(num_train), int(num_test)
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    return a, b


def _grid(value):
    grid = np.asarray(value)
    _need(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), 'invalid grid shape')
    _need(np.issubdtype(grid.dtype, np.integer), 'grid must be integer')
    _need(bool(np.all((grid >= 0) & (grid <= 9))), 'colors outside 0..9')
    return grid


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p):
    return [(p[0] + dr, p[1] + dc) for dr, dc in _DIRS]


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, component = [first], {first}
        while stack:
            for q in _neighbors(stack.pop()):
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _word_colors(path, word, reset=False):
    colors, phase, last_direction = [], 0, None
    for i, p in enumerate(path):
        if i:
            direction = (p[0] - path[i-1][0], p[1] - path[i-1][1])
            if reset and last_direction is not None and direction != last_direction:
                phase = 0
            last_direction = direction
        colors.append(word[phase % len(word)])
        phase += 1
    return colors


def _dock(path, colors, registration, marker, first=False):
    candidates = [(abs(p[0]-registration[0]) + abs(p[1]-registration[1]), i)
                  for i, p in enumerate(path)
                  if (p[0] == registration[0] or p[1] == registration[1])
                  and (colors is None or colors[i] == marker)]
    if not candidates:
        return None
    if first:
        return path[min(i for _, i in candidates)]
    candidates.sort()
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return path[candidates[0][1]]


def _decode(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    # The top left corner is part of a blank margin, independent of palette rank.
    background = int(grid[0, 0])
    components = _components(tuple(map(int, p)) for p in np.argwhere(grid != background))
    swatches, others = [], []
    for component in components:
        rows, cols = sorted({r for r, _ in component}), sorted({c for _, c in component})
        if (len(component) == 4 and len(rows) == len(cols) == 2
                and rows[1] == rows[0]+1 and cols[1] == cols[0]+1
                and len({int(grid[p]) for p in component}) == 1):
            swatches.append((rows[0], cols[0], int(grid[rows[0], cols[0]]), component))
        else:
            others.append(component)
    swatches.sort()
    _need(2 <= len(swatches) <= 6, 'need 2..6 separated legend swatches')
    _need(len({s[0] for s in swatches}) == 1, 'legend is a horizontal row')
    _need(swatches[0][0] >= 1, 'legend must be inset')
    _need(all(b[1] >= a[1]+3 for a, b in zip(swatches, swatches[1:])), 'legend gap')
    word = tuple(s[2] for s in swatches)
    _need(len(set(word)) >= 2, 'word needs at least two colors')
    legend = set().union(*(s[3] for s in swatches))
    rails = [c for c in others if any(r in (0, h-1) or col in (0, w-1) for r, col in c)]
    _need(len(rails) == 1, 'exactly one component touches the boundary')
    rail = rails[0]
    heads = [p for p in rail if int(grid[p]) == word[0]]
    _need(len(heads) == 1, 'rail needs one first-color head')
    head = heads[0]
    _need(head[0] == h-1 or head[1] in (0, w-1), 'head on left, right, or bottom')
    _need(all(int(grid[p]) not in set(word) for p in rail - {head}), 'neutral rail')
    neutral = {int(grid[p]) for p in rail - {head}}
    _need(len(neutral) == 1, 'one neutral rail color')
    rail_color = next(iter(neutral))
    _need(rail_color != background, 'rail differs from background')
    degrees = {p: len(set(_neighbors(p)) & rail) for p in rail}
    _need(len(rail) >= 3 and degrees[head] == 1, 'head must be a rail endpoint')
    _need(sum(d == 1 for d in degrees.values()) == 2
          and all(d in (1, 2) for d in degrees.values()), 'rail must be a simple unbranched path')
    path, previous, current = [], None, head
    while True:
        path.append(current)
        choices = [q for q in _neighbors(current) if q in rail and q != previous]
        if not choices:
            break
        previous, current = current, choices[0]
    _need(len(path) == len(rail), 'rail cannot have loops')
    bottom = swatches[0][0]+1
    _need(all(r >= bottom+3 for r, _ in rail), 'rail must be below the legend margin')
    station_colors = _word_colors(path, word)
    movers = []
    for component in others:
        if component is rail:
            continue
        _need(all(0 < r < h-1 and 0 < c < w-1 and r >= bottom+3 for r, c in component),
              'source carriers must be interior and below legend')
        registrations = [p for p in component if int(grid[p]) in word]
        _need(len(registrations) == 1, 'exactly one legend-colored registration per carrier')
        _need(all(int(grid[p]) != rail_color for p in component), 'body cannot use rail color')
        registration = registrations[0]
        marker = int(grid[registration])
        target = _dock(path, station_colors, registration, marker)
        _need(target is not None, 'missing or tied matching dock')
        delta = (target[0]-registration[0], target[1]-registration[1])
        _need(delta[0] == 0 or delta[1] == 0, 'translation must be axial')
        footprint = {(r+delta[0], c+delta[1]): int(grid[r, c]) for r, c in component}
        movers.append({'points': component, 'registration': registration, 'marker': marker,
                       'target': target, 'delta': delta, 'footprint': footprint})
    _need(movers, 'no carriers')
    return {'grid': grid, 'background': background, 'rail_color': rail_color,
            'path': path, 'word': word, 'legend': legend, 'movers': movers,
            'station_colors': station_colors}


def _render(info, marker_only=False):
    # Erase all sources first; destinations may occupy a different source footprint.
    out = info['grid'].copy()
    for mover in info['movers']:
        for p in mover['points']:
            out[p] = info['background']
    occupied = set(info['legend'])
    for mover in info['movers']:
        footprint = ({mover['target']: mover['marker']} if marker_only else mover['footprint'])
        for p, color in footprint.items():
            if _inside(out.shape, p):
                _need(p not in occupied, 'docked carriers overlap each other or the legend')
                occupied.add(p)
                out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Trace virtual phases, choose unique closest matching axial docks, move shapes."""
    return _render(_decode(input_grid))


def _evidence(info):
    word, path = info['word'], info['path']
    alternatives = {'gating': None,
                    'multiplicity': _word_colors(path, tuple(dict.fromkeys(word))),
                    'bend_phase': _word_colors(path, word, reset=True)}
    evidence = {'cycling'} if len(path) >= 2*len(word) else set()
    for m in info['movers']:
        for name, colors in alternatives.items():
            other = _dock(path, colors, m['registration'], m['marker'])
            if other is not None and other != m['target']:
                evidence.add(name)
        other = _dock(path, info['station_colors'], m['registration'], m['marker'], first=True)
        if other is not None and other != m['target']:
            evidence.add('nearest_match')
        if len(m['points']) > 1 and m['delta'] != (0, 0):
            evidence.add('rigid')
            r, c = m['registration']
            if any(2*q != min(p[k] for p in m['points']) + max(p[k] for p in m['points'])
                   for k, q in enumerate((r, c))):
                evidence.add('registration')
    return evidence


def _sample_word(rng, test, witness):
    n = int(rng.integers(4, 7) if test else rng.integers(3, 6) if witness else rng.integers(2, 7))
    d = int(rng.integers(2, min(4, n-1 if witness else n)+1))
    for _ in range(12):
        word = list(range(d)) + list(map(int, rng.integers(d, size=n-d)))
        rng.shuffle(word)
        unique = tuple(dict.fromkeys(word))
        if not witness or any(word[i % n] != unique[i % d] for i in range(2*n)):
            return tuple(word), d
    return tuple([0]*(n-d+1) + list(range(1, d))), d


def _make_rail(rng, h, w, top, legs):
    """Sample a separated polyline; finite trials, no saved search traces."""
    for _ in range(100):
        side = int(rng.integers(3))
        if side == 0:
            start, direction = (int(rng.integers(top+2, h-2)), 0), (0, 1)
        elif side == 1:
            start, direction = (int(rng.integers(top+2, h-2)), w-1), (0, -1)
        else:
            start, direction = (h-1, int(rng.integers(3, w-3))), (-1, 0)
        path, occupied, valid = [start], {start}, True
        for leg in range(legs):
            if leg:
                choices = [(-1, 0), (1, 0)] if direction[1] else [(0, -1), (0, 1)]
                direction = choices[int(rng.integers(2))]
            length = int(rng.integers(3, 9))
            for _ in range(length):
                p = (path[-1][0]+direction[0], path[-1][1]+direction[1])
                if not (top <= p[0] < h-1 and 1 <= p[1] < w-1):
                    break
                if p in occupied or (set(_neighbors(p)) & occupied) != {path[-1]}:
                    break
                occupied.add(p)
                path.append(p)
            else:
                continue
            valid = False
            break
        if valid and len(path) >= (4 if legs == 1 else 12):
            return path
    # Constructive monotone fallback fits even the smallest generated canvas.
    # Randomized knots preserve broad geometry if the free-turn sampler is unlucky.
    row = h-3
    columns = sorted(map(int, rng.choice(np.arange(3, w-2), 3, replace=False)))
    rows = sorted(map(int, rng.choice(np.arange(top, h-4), 2, replace=False)), reverse=True)
    knots = [(row, 0), (row, columns[0]), (rows[0], columns[0]),
             (rows[0], columns[1]), (rows[1], columns[1]), (rows[1], columns[2])]
    path = [knots[0]]
    for end in knots[1:]:
        while path[-1] != end:
            r, c = path[-1]
            path.append((r + int(np.sign(end[0]-r)), c + int(np.sign(end[1]-c))))
    # Reject adjacent parallel legs by using this broad fixed spacing if needed.
    if any(len(set(_neighbors(p)) & set(path)) > 2 for p in path):
        knots = [(h-3, 0), (h-3, 5), (h-8, 5), (h-8, 12), (top, 12), (top, w-3)]
        path = [knots[0]]
        for end in knots[1:]:
            while path[-1] != end:
                r, c = path[-1]
                path.append((r + int(np.sign(end[0]-r)), c + int(np.sign(end[1]-c))))
    return path


def _shape(rng):
    kind = int(rng.integers(5))
    if kind == 0:
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 5))
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        cells = {(r, c) for r in range(3) for c in range(3) if r in (0, 2) or c in (0, 2)}
        if rng.random() < .5:
            cells.remove((0, 1))
    elif kind == 2:
        length = int(rng.integers(3, 7))
        cells = {(r, 0) for r in range(length)} | {(1, 1), (1, 2)}
    else:
        cells = {(0, 0)}
        for _ in range(int(rng.integers(2, 10))):
            frontier = sorted({q for p in cells for q in _neighbors(p)} - cells)
            frontier = [p for p in frontier if max(abs(p[0]), abs(p[1])) <= 2]
            if frontier:
                cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    points = sorted(cells)
    mark = points[int(rng.integers(len(points)))]
    return [(r-mark[0], c-mark[1]) for r, c in points]


def _origin_candidates(path, word, h, w, top):
    actual = _word_colors(path, word)
    unique = _word_colors(path, tuple(dict.fromkeys(word)))
    reset = _word_colors(path, word, reset=True)
    blocked = set(path) | {q for p in path for q in _neighbors(p)}
    rows, columns = {}, {}
    for i, (r, c) in enumerate(path):
        rows.setdefault(r, []).append(i)
        columns.setdefault(c, []).append(i)
    result = []
    for r in range(top, h-1):
        for c in range(1, w-1):
            reg = (r, c)
            if reg in blocked:
                continue
            indices = sorted(set(rows.get(r, []) + columns.get(c, [])))
            if not indices:
                continue
            # Compute docking on this small local list; maintain original order.
            local_path = [path[i] for i in indices]
            for marker in sorted(set(word)):
                colors = [actual[i] for i in indices]
                target = _dock(local_path, colors, reg, marker)
                if target is None:
                    continue
                flags = set()
                for name, alt in (('gating', None), ('multiplicity', unique), ('bend_phase', reset)):
                    other = _dock(local_path, None if alt is None else [alt[i] for i in indices], reg, marker)
                    if other is not None and other != target:
                        flags.add(name)
                if _dock(local_path, colors, reg, marker, first=True) != target:
                    flags.add('nearest_match')
                result.append((reg, marker, target, flags))
    return result


def _build_pair(rng, test, witness, palette, clipping, require_clip):
    for attempt in range(70):
        pattern, distinct = _sample_word(rng, test, witness)
        word = tuple(palette[2+i] for i in pattern)
        min_width = max(18, 3*len(word)+2)
        h = int(rng.integers(23, 31) if test else rng.integers(19, 29))
        w = int(rng.integers(max(min_width, 24) if test else min_width, 31))
        legend_row = int(rng.integers(1, 3))
        top = legend_row+4
        legs = int(rng.integers(5, 8) if test else rng.integers(3, 6) if witness else rng.integers(1, 6))
        path = _make_rail(rng, h, w, top, legs)
        bg, rail = palette[:2]
        inp = np.full((h, w), bg, dtype=np.int8)
        # Unequal gaps and horizontal offsets vary while preserving reading order.
        slack = w - (3*len(word)-1) - 2
        left = int(rng.integers(1, min(4, slack+1)+1))
        remaining = slack-(left-1)
        columns = [left]
        for _ in word[1:]:
            extra = int(rng.integers(min(2, remaining)+1))
            columns.append(columns[-1]+3+extra)
            remaining -= extra
        legend = set()
        for c, color in zip(columns, word):
            inp[legend_row:legend_row+2, c:c+2] = color
            legend |= {(r, q) for r in range(legend_row, legend_row+2) for q in (c, c+1)}
        for p in path:
            inp[p] = rail
        inp[path[0]] = word[0]
        origins = _origin_candidates(path, word, h, w, top)
        if not origins:
            continue
        n = int(rng.integers(6, 8) if test else rng.integers(3, 6))
        source = set(path) | legend
        target_used = set(legend)
        chosen_flags = {'cycling'} if len(path) >= 2*len(word) else set()
        movers, clipped_count = [], 0
        for j in range(n):
            rng.shuffle(origins)
            missing = _EVIDENCE - chosen_flags
            origins.sort(key=lambda x: (len(x[3] & missing),
                                        require_clip and not clipped_count and
                                        min(x[2][0], x[2][1], h-1-x[2][0], w-1-x[2][1]) < 3), reverse=True)
            placed = False
            blocked = source | {q for p in source for q in _neighbors(p)}
            # A bounded set of origins and shape attempts controls runtime.
            for reg, marker, target, flags in origins[:240]:
                if reg in blocked:
                    continue
                for _ in range(5):
                    offsets = _shape(rng)
                    points = {(reg[0]+dr, reg[1]+dc) for dr, dc in offsets}
                    if any(not (top <= r < h-1 and 0 < c < w-1) or (r, c) in blocked for r, c in points):
                        continue
                    destinations = {(target[0]+dr, target[1]+dc) for dr, dc in offsets}
                    inside = {p for p in destinations if _inside((h, w), p)}
                    clipped = len(inside) < len(points)
                    if (clipped and not clipping) or inside & target_used:
                        continue
                    # If clipping is requested, reserve the last placement for it.
                    if require_clip and not clipped_count and j == n-1 and not clipped:
                        continue
                    body = int(rng.choice(palette[2+distinct:]))
                    cells = {}
                    for dr, dc in offsets:
                        color = marker if (dr, dc) == (0, 0) else body
                        if color == body and rng.random() < .13:
                            color = int(rng.choice(palette[2+distinct:]))
                        p = (reg[0]+dr, reg[1]+dc)
                        inp[p] = color
                        cells[p] = color
                    delta = (target[0]-reg[0], target[1]-reg[1])
                    movers.append((cells, delta))
                    source |= points
                    target_used |= inside
                    chosen_flags |= flags | {'rigid'}
                    if any(2*reg[k] != min(p[k] for p in points)+max(p[k] for p in points) for k in (0, 1)):
                        chosen_flags.add('registration')
                    clipped_count += int(clipped)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                break
        if len(movers) != n or (witness and not _EVIDENCE <= chosen_flags) or (require_clip and not clipped_count):
            continue
        # Construct labels directly from sampled placements, independently of solve.
        out = np.full_like(inp, bg)
        for p in legend:
            out[p] = inp[p]
        for p in path:
            out[p] = inp[p]
        for cells, delta in movers:
            for (r, c), color in cells.items():
                p = (r+delta[0], c+delta[1])
                if _inside(out.shape, p):
                    out[p] = color
        info = _decode(inp)
        _need(np.array_equal(out, _render(info)), 'construction/solver mismatch')
        return {'input': inp, 'output': out}
    raise RuntimeError('bounded scene construction exhausted')


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _fingerprint(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if not _integer(seed) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .25)
    palette = tuple(map(int, rng.permutation(10)))
    clipping = bool(rng.random() < .5)
    witnesses = set(map(int, rng.choice(a, 2, replace=False)))
    clip_demo = min(witnesses)
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', a), ('test', b)):
        for i in range(count):
            for _ in range(8):
                colors = palette if stable else tuple(map(int, rng.permutation(10)))
                pair = _build_pair(rng, split == 'test', split == 'train' and i in witnesses,
                                   colors, clipping, clipping and split == 'train' and i == clip_demo)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('could not construct distinct inputs')
            seen.add(key)
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def _clip_count(info):
    return sum(any(not _inside(info['grid'].shape, p) for p in m['footprint']) for m in info['movers'])


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, stats = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'dimensions must be preserved')
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _decode(inp)
            _need(np.array_equal(out, _render(info)), 'incorrect rigid docking output')
            _need(not np.array_equal(inp, out), 'identity pair')
            # Input-only station/color/geometry invariants, including clipping.
            for m in info['movers']:
                index = info['path'].index(m['target'])
                _need(info['word'][index % len(info['word'])] == m['marker'], 'phase mismatch')
                _need(m['delta'] != (0, 0), 'source carrier cannot start on rail')
                _need(out[m['target']] == m['marker'], 'registration must land on dock')
                for p, color in m['footprint'].items():
                    if _inside(inp.shape, p):
                        _need(int(out[p]) == color, 'shape or payload changed')
            if split == 'train':
                evidence |= _evidence(info)
            stats[split].append(info)
    _need(_EVIDENCE <= evidence, 'training misses distinctions: '+str(sorted(_EVIDENCE-evidence)))
    if any(_clip_count(i) for i in stats['test']):
        _need(any(_clip_count(i) for i in stats['train']), 'test clipping lacks a demonstration')


def self_test():
    """Bounded streaming suite: 42 seeds, legal counts, metamorphisms, corruptions."""
    _need(_dock([(2, 2), (2, 3), (2, 4)], [1, 2, 1], (2, 3), 1) is None,
          'equal-distance docks must be rejected')
    folded = [(7, 0), (7, 1), (7, 2), (6, 2), (5, 2), (5, 3)]
    _need(_word_colors(folded, (4, 4, 8)) == [4, 4, 8, 4, 4, 8], 'continuous phase fixture')
    _need(_word_colors(folded, (4, 4, 8), reset=True)[-1] == 4, 'bend-reset counterexample')
    shapes, layouts, patterns, carriers, rail_shapes = set(), set(), set(), set(), set()
    head_sides, directions, clips, backgrounds = set(), set(), set(), set()
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(hashlib.sha256(repr(_fingerprint(puzzle)).encode()).digest())
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                shapes.add(info['grid'].shape)
                background = info['background']
                backgrounds.add(background)
                path = info['path']
                rail_shapes.add(tuple((r-path[0][0], c-path[0][1]) for r, c in path))
                roles = {}
                patterns.add(tuple(roles.setdefault(x, len(roles)) for x in info['word']))
                h, w = info['grid'].shape
                head_sides.add('bottom' if path[0][0] == h-1 else 'left' if path[0][1] == 0 else 'right')
                clips.add(_clip_count(info))
                for m in info['movers']:
                    directions.add(tuple(int(np.sign(x)) for x in m['delta']))
                    r, c = m['registration']
                    carriers.add(tuple(sorted((rr-r, cc-c) for rr, cc in m['points'])))
                local.append((info['grid'].shape, len(path), len(info['movers']), info['word']))
        layouts.add(tuple(local))
    _need(len(fingerprints) == len(layouts) == 42, '42-seed variation')
    _need(len(shapes) >= 35 and len(rail_shapes) >= 100 and len(carriers) >= 70, 'structural variation')
    _need(len(patterns) >= 15 and backgrounds == set(range(10)), 'program/palette variation')
    _need(head_sides == {'left', 'right', 'bottom'}, 'head side coverage')
    _need(directions == {(-1, 0), (1, 0), (0, -1), (0, 1)}, 'movement sign/axis coverage')
    _need(0 in clips and max(clips) >= 1, 'clipping coverage')
    # Every legal explicit count combination; keep only the current puzzle.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(8900+a*10+b, a, b)
            _need(len(p['train']) == a and len(p['test']) == b, 'explicit counts')
    p = generate_puzzle(812, 2, 1)
    _need(_fingerprint(p) == _fingerprint(generate_puzzle(812, 2, 1)), 'determinism')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Palette changes and reflection of the scene with the word kept in its
    # fixed left-to-right reading order preserve the rule.
    permutation = np.random.default_rng(432).permutation(10)
    for pair in p['train'] + p['test']:
        _need(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]), 'palette covariance')
        reflected = pair['input'][:, ::-1].copy()
        expected = pair['output'][:, ::-1].copy()
        info = _decode(pair['input'])
        groups = _components((r, reflected.shape[1]-1-c) for r, c in info['legend'])
        groups.sort(key=lambda cells: min(c for _, c in cells))
        for cells, color in zip(groups, info['word']):
            for q in cells:
                reflected[q] = expected[q] = color
        _need(np.array_equal(solve(reflected), expected), 'spatial reflection covariance')
    bad = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    info = _decode(p['train'][0]['input'])
    _need(not np.array_equal(_render(info, marker_only=True), p['train'][0]['output']), 'marker-only ablation')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
