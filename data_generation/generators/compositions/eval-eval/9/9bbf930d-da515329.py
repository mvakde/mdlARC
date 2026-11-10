"""Summoned pinwheel corridors: colored phase growth controlled by matching gates.

No parent module, saved grid, seed, or demonstration is used by solve().
The standard library and NumPy are the only dependencies.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _boundary(shape, p):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _phase_map(length, shape, center):
    """Procedural infinite-plane winding, with phase IDs N=1,E=2,S=3,W=4.

    Paint clipped segments, never stop an arm on first leaving the canvas.
    This is the stepped recurrence, including the longer-spoke inward jog.
    """
    result = np.zeros(shape, dtype=np.int8)
    radius = max(2, length)
    cy, cx = center

    def paint(p, q, phase):
        y0, x0 = p[0] + cy, p[1] + cx
        y1, x1 = q[0] + cy, q[1] + cx
        if y0 == y1 and 0 <= y0 < shape[0]:
            lo, hi = max(0, min(x0, x1)), min(shape[1] - 1, max(x0, x1))
            if lo <= hi:
                cells = result[y0, lo:hi + 1]
                _require(np.all((cells == 0) | (cells == phase)), "phase collision")
                cells[:] = phase
        elif x0 == x1 and 0 <= x0 < shape[1]:
            lo, hi = max(0, min(y0, y1)), min(shape[0] - 1, max(y0, y1))
            if lo <= hi:
                cells = result[lo:hi + 1, x0]
                _require(np.all((cells == 0) | (cells == phase)), "phase collision")
                cells[:] = phase

    distance = max(cy, cx, shape[0] - 1 - cy, shape[1] - 1 - cx)
    # Four stages displace the canonical start outward by eight cells.
    stages = 4 * (distance // 8 + 3)
    for phase, radial in enumerate(DIRS, 1):
        p = radius * radial[0], radius * radial[1]
        paint(radial, p, phase)
        dy, dx = radial[1], -radial[0]  # clockwise from the spoke
        for stage in range(stages):
            for vy, vx, count in ((dy, dx, 2 + 4 * stage),
                                  (dx, -dy, radius - 2), (dy, dx, radius)):
                q = p[0] + count * vy, p[1] + count * vx
                paint(p, q, phase)
                p = q
            dy, dx = dx, -dy
    return result


def _analytic_phases(length, shape, center):
    """Cell membership from closed-form stage starts; no accumulated path state.

    Used by validation to cross-check the procedural construction.
    """
    radius = max(2, length)
    yy, xx = np.indices(shape)
    yy, xx = yy - center[0], xx - center[1]
    result = np.zeros(shape, dtype=np.int8)
    bound = max(int(np.abs(yy).max()), int(np.abs(xx).max()))
    for phase in range(1, 5):
        hit = (xx == 0) & (-radius <= yy) & (yy <= -1)
        for cycle in range(bound // 8 + 4):
            starts = ((-radius - 8 * cycle, -8 * cycle),
                      (-2 - 8 * cycle, radius + 8 * cycle + 2),
                      (radius + 8 * cycle + 4, 8 * cycle + 4),
                      (8 * cycle + 6, -radius - 8 * cycle - 6))
            for quarter, ((y, x), (dy, dx)) in enumerate(zip(
                    starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along = (yy - y) * dy + (xx - x) * dx
                across = (yy - y) * dx - (xx - x) * dy
                run = 16 * cycle + 4 * quarter + 2
                hit |= (across == 0) & (0 <= along) & (along <= run)
                hit |= (along == run) & (0 <= across) & (across <= radius - 2)
                hit |= (across == radius - 2) & (run <= along) & (along <= run + radius)
        _require(np.all((result[hit] == 0) | (result[hit] == phase)), "analytic collision")
        result[hit] = phase
        yy, xx = -xx, yy  # inverse rotation into the next arm's coordinates
    return result


def _seed_cells(length, center):
    return {(center[0] + n * dy, center[1] + n * dx): phase
            for phase, (dy, dx) in enumerate(DIRS, 1)
            for n in range(1, length + 1)}


def _inner_ends(length, center):
    radius = max(2, length)
    # NW, NE, SE, SW background dead ends, each capped by two seed phases.
    return [(center[0] + dy, center[1] + dx)
            for dy, dx in ((-1, 1 - radius), (1 - radius, 1),
                           (1, radius - 1), (radius - 1, -1))]


def _trace(mask, start, stop_at_boundary):
    """Read a simple background corridor, independently of a sampled template."""
    _require(mask[start] == 0, "corridor start is on a rail")
    path, seen, previous, p = [], set(), None, start
    while True:
        _require(p not in seen, "corridor cycle")
        seen.add(p)
        path.append(p)
        if stop_at_boundary and _boundary(mask.shape, p):
            break
        choices = [_add(p, d) for d in DIRS
                   if _inside(mask.shape, _add(p, d)) and _add(p, d) != previous
                   and mask[_add(p, d)] == 0]
        _require(len(choices) <= 1, "corridor branches")
        if not choices:
            break
        previous, p = p, choices[0]
    return path


def _flanks(p, direction):
    normal = -direction[1], direction[0]
    return _add(p, normal), _add(p, (-normal[0], -normal[1]))


def _lanes(mask, length, center):
    """Get possible edge mouths by tracing out from the four inner dead ends."""
    lanes = []
    for terminal in _inner_ends(length, center):
        outward = _trace(mask, terminal, True)
        if len(outward) < 5 or not _boundary(mask.shape, outward[-1]):
            continue
        path = outward[::-1]
        direction = path[1][0] - path[0][0], path[1][1] - path[0][1]
        a, b = _flanks(path[0], direction)
        if not (_inside(mask.shape, a) and _inside(mask.shape, b)):
            continue  # corner mouths have no two visible flanks
        if not mask[a] or not mask[b] or mask[a] == mask[b]:
            continue
        straight = 1
        while straight < len(path) and path[straight] == (
                path[0][0] + straight * direction[0], path[0][1] + straight * direction[1]):
            straight += 1
        lanes.append(dict(path=path, direction=direction,
                          phases=(int(mask[a]), int(mask[b])), straight=straight))
    return lanes


def _stub(lane, depth, mask):
    cells = set()
    for p in lane['path'][:depth]:
        for q in _flanks(p, lane['direction']):
            _require(_inside(mask.shape, q) and mask[q] in lane['phases'], "invalid gate stub")
            cells.add(q)
    return cells


def _rail_halo(path, phases, mask):
    """Only the bordering strands, including diagonal outside corners at bends."""
    cells = set()
    for y, x in path:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = y + dy, x + dx
                if _inside(mask.shape, q) and int(mask[q]) in phases:
                    cells.add(q)
    return cells


def _turns(path):
    directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _detect(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    _require(np.count_nonzero(a == bg) > a.size // 2, "sparse majority background required")
    edge_only = [int(c) for c in values if c != bg and all(
        _boundary(a.shape, tuple(map(int, p))) for p in np.argwhere(a == c))]
    _require(len(edge_only) == 1, "one edge-only token color required")
    token = edge_only[0]
    candidates = []
    for y, x in np.argwhere(a == bg):
        center = int(y), int(x)
        lengths, colors = [], []
        for direction in DIRS:
            p = _add(center, direction)
            if not _inside(a.shape, p) or int(a[p]) in (bg, token):
                break
            color, n = int(a[p]), 0
            while _inside(a.shape, p) and int(a[p]) == color:
                n += 1
                p = _add(p, direction)
            lengths.append(n)
            colors.append(color)
        if len(lengths) == 4 and len(set(lengths)) == 1 and 1 <= lengths[0] <= 6:
            length = lengths[0]
            radius = max(2, length)
            if min(y, x, a.shape[0] - 1 - y, a.shape[1] - 1 - x) >= radius + 2:
                candidates.append((length, center, tuple(colors)))
    _require(len(candidates) == 1, "one fully visible equal-spoke seed required")
    length, center, colors = candidates[0]
    _require(set(map(int, values)) == {bg, token, *colors}, "unexplained input color")
    return a, bg, token, length, center, colors


def _analyse(input_grid, analytic=False):
    a, bg, token, length, center, colors = _detect(input_grid)
    mask = (_analytic_phases if analytic else _phase_map)(length, a.shape, center)
    seeds = _seed_cells(length, center)
    expected = np.full(a.shape, bg, dtype=np.int8)
    for p, phase in seeds.items():
        expected[p] = colors[phase - 1]
    gates = []
    inner = set(_inner_ends(length, center))
    for position in np.argwhere(a == token):
        start = tuple(map(int, position))
        path = _trace(mask, start, False)
        _require(path[-1] in inner and len(path) >= 5, "gate must lead to a seed dead end")
        _require(not any(_boundary(a.shape, p) for p in path[1:]), "gate route leaves the canvas")
        direction = path[1][0] - start[0], path[1][1] - start[1]
        fa, fb = _flanks(start, direction)
        _require(_inside(a.shape, fa) and _inside(a.shape, fb), "corner gate is ambiguous")
        phases = int(mask[fa]), int(mask[fb])
        _require(phases[0] and phases[1] and phases[0] != phases[1], "two phase rails required")
        straight = 1
        while straight < len(path) and path[straight] == (
                start[0] + straight * direction[0], start[1] + straight * direction[1]):
            straight += 1
        lane = dict(path=path, phases=phases, direction=direction, straight=straight)
        depth = 0
        for i in range(min(3, straight)):
            pair = _flanks(path[i], direction)
            if all(_inside(a.shape, q) and int(mask[q]) in phases and
                   a[q] == colors[int(mask[q]) - 1] for q in pair):
                depth += 1
            else:
                break
        _require(depth >= 1, "visible two-sided gate required")
        cells = _stub(lane, depth, mask)
        _require(all(max(abs(q[0] - s[0]), abs(q[1] - s[1])) >= 2
                     for q in cells for s in seeds), "gate and seed must be separated")
        for q in cells:
            expected[q] = colors[int(mask[q]) - 1]
        expected[start] = token
        lane.update(depth=depth, active=colors[phases[0] - 1] == colors[phases[1] - 1])
        gates.append(lane)
    _require(1 <= len(gates) <= 4, "one to four gates required")
    _require(np.array_equal(a, expected), "input must be exactly seed, gates, and tokens")
    paths = [p for g in gates for p in g['path']]
    _require(len(paths) == len(set(paths)), "gate corridors overlap")
    return dict(input=a, background=bg, token=token, length=length, center=center,
                colors=colors, mask=mask, gates=gates)


def _answer(info, ignore_equality=False, straight_only=False):
    out = info['input'].copy()
    bg, token, colors, mask = (info[k] for k in ('background', 'token', 'colors', 'mask'))
    for gate in info['gates']:
        if not (gate['active'] or ignore_equality):
            continue
        path = gate['path']
        if straight_only:
            # A deliberate ablation: gate selection but no winding growth.
            p, direction = path[0], gate['direction']
            while _inside(out.shape, _add(p, direction)) and info['input'][_add(p, direction)] == bg:
                p = _add(p, direction)
            out[path[0]] = bg
            out[p] = token
            continue
        for q in _rail_halo(path, gate['phases'], mask):
            out[q] = colors[int(mask[q]) - 1]
        out[path[0]] = bg
        out[path[-1]] = token
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only visible seed geometry, phase colors, and boundary gates."""
    return _answer(_analyse(input_grid))


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10),
                                  (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        if not low <= int(value) <= high:
            raise ValueError(f'{name} must be in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _sample_layout(rng, length, low, high, favor_four=False):
    radius = max(2, length)
    minimum = max(low, 2 * radius + 5)
    h, w = map(int, rng.integers(minimum, high + 1, size=2))
    if rng.random() < (.45 if favor_four else .18):
        w = h
    # All feasible placements are available; narrow grids constrain margins.
    if favor_four and rng.random() < .7:
        center = (h // 2, w // 2)
    else:
        center = (int(rng.integers(radius + 2, h - radius - 2)),
                  int(rng.integers(radius + 2, w - radius - 2)))
    mask = _phase_map(length, (h, w), center)
    lanes = _lanes(mask, length, center)
    return (h, w), center, mask, lanes


def _make_input(rng, length, shape, center, mask, lanes, phase_colors, bg, token):
    seeds = _seed_cells(length, center)
    inp = np.full(shape, bg, dtype=np.int8)
    for p, phase in seeds.items():
        inp[p] = phase_colors[phase - 1]
    for lane in lanes:
        maximum = 0
        for p in lane['path'][:min(3, lane['straight'])]:
            if all(_inside(mask.shape, q) and mask[q] in lane['phases']
                   for q in _flanks(p, lane['direction'])):
                maximum += 1
            else:
                break
        depths = list(map(int, rng.permutation(np.arange(1, maximum + 1))))
        for depth in depths:
            cells = _stub(lane, depth, mask)
            if all(max(abs(q[0] - s[0]), abs(q[1] - s[1])) >= 2
                   for q in cells for s in seeds):
                break
        else:
            return None
        for q in cells:
            inp[q] = phase_colors[int(mask[q]) - 1]
        inp[lane['path'][0]] = token
    return inp


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(['extent', 'crowding']))
    fixed_palette = bool(rng.random() < .25)
    palette = list(map(int, rng.permutation(10)))
    train_lengths = [2, 3] + list(map(int, rng.integers(1, 7, size=ntrain - 2)))
    test_lengths = list(map(int, rng.choice(sorted(set(train_lengths)), size=ntest)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for index, length in enumerate(train_lengths + test_lengths):
        is_test = index >= ntrain
        split = 'test' if is_test else 'train'
        # First examples discriminate equality, single-lane growth, shared
        # rails, and changing scale. Later examples need not repeat a recipe.
        requested = (2 if index == 0 else 3 if index == 1 else
                     4 if is_test and regime == 'crowding' else
                     int(rng.integers(1, 4 if regime == 'crowding' else 5)))
        low, high = ((24, 30) if is_test and regime == 'extent' else
                     (11, 22) if regime == 'extent' else (11, 30))
        for attempt in range(800):
            roles = palette if fixed_palette else list(map(int, rng.permutation(10)))
            bg, token, arm_palette = roles[0], roles[1], roles[2:6]
            shape, center, mask, possible = _sample_layout(rng, length, low, high, requested == 4)
            if len(possible) < requested:
                continue
            if index == 0:
                partition = [0, 0, 1, 1]  # two matches, but only one receives a token
            elif index == 1:
                partition = [0, 0, 0, 1]  # two lanes share a phase strand
            elif is_test and regime == 'crowding':
                partition = [0, 0, 0, 0]  # four simultaneous matching gates
            else:
                partition = [0, 0, 0, 0] if rng.random() < .2 else list(map(int, rng.choice(4, size=4)))
            # Rotate/reflection of the color partition does not alter handedness.
            shift = int(rng.integers(4))
            partition = partition[shift:] + partition[:shift]
            if rng.random() < .5:
                partition = partition[::-1]
            phase_colors = tuple(arm_palette[p] for p in partition)
            active = [g for g in possible if phase_colors[g['phases'][0] - 1] ==
                      phase_colors[g['phases'][1] - 1]]
            inactive = [g for g in possible if g not in active]
            if index == 0:
                if len(active) < 2 or not inactive:
                    continue
                selected = [active[int(rng.integers(len(active)))],
                            inactive[int(rng.integers(len(inactive)))]]
            elif index == 1:
                if len(active) < 2 or not inactive:
                    continue
                selected = active[:2] + [inactive[int(rng.integers(len(inactive)))]]
            else:
                chosen = rng.choice(len(possible), requested, replace=False)
                selected = [possible[int(i)] for i in chosen]
                if not any(g in active for g in selected):
                    continue
            inp = _make_input(rng, length, shape, center, mask, selected, phase_colors, bg, token)
            if inp is None or _key(inp) in seen:
                continue
            try:
                info = _analyse(inp)
            except AssertionError:
                continue
            if not all(_turns(g['path']) >= 2 for g in info['gates'] if g['active']):
                continue
            out = _answer(info)
            if index < 2 and not np.any((mask > 0) & (inp == bg) & (out == bg)):
                continue
            puzzle[split].append({'input': inp, 'output': out})
            seen.add(_key(inp))
            break
        else:
            raise AssertionError('bounded layout search exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'dimensions must be preserved')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _analyse(inp, analytic=True)
            _require(np.array_equal(out, _answer(info)), 'incorrect selective completion or transfer')
            bg, token, mask = info['background'], info['token'], info['mask']
            active = [g for g in info['gates'] if g['active']]
            _require(active and all(_turns(g['path']) >= 2 for g in active), 'winding must affect movers')
            _require(np.count_nonzero(inp == token) == np.count_nonzero(out == token), 'token conservation')
            existing_rails = (inp != bg) & (inp != token)
            _require(np.array_equal(inp[existing_rails], out[existing_rails]), 'existing rails changed')
            changed_to_token = set(map(tuple, np.argwhere((inp != token) & (out == token))))
            _require(changed_to_token == {g['path'][-1] for g in active}, 'wrong inner terminals')
            _require(out[info['center']] == bg, 'hollow center must stay empty')
            _require(np.any((inp == bg) & (out != bg) & (out != token)), 'nonempty rail growth')
            _require(not np.array_equal(_answer(info, straight_only=True), out), 'B ablation must fail')
            records[split].append(info)
    first, second = records['train'][:2]
    _require(first['length'] == 2 and second['length'] == 3, 'short and jogged demonstrations')
    for info in (first, second):
        _require(any(not g['active'] for g in info['gates']), 'training must show a mismatch')
        _require(not np.array_equal(_answer(info, ignore_equality=True), _answer(info)), 'A ablation must fail')
        _require(np.any((info['mask'] > 0) & (info['input'] == info['background']) &
                        (_answer(info) == info['background'])), 'unconditional completion shortcut')
    _require(sum(g['active'] for g in first['gates']) == 1, 'single-lane demonstration')
    _require(sum(g['active'] for g in second['gates']) == 2, 'shared-strand demonstration')
    claimed = {g['path'][0] for g in first['gates']}
    first_out = _answer(first)
    unclaimed_matches = [g for g in _lanes(first['mask'], first['length'], first['center'])
                         if g['path'][0] not in claimed and
                         first['colors'][g['phases'][0] - 1] == first['colors'][g['phases'][1] - 1]]
    _require(any(any(first_out[q] == first['background']
                     for q in _rail_halo(g['path'], g['phases'], first['mask']))
                 for g in unclaimed_matches), 'training must distinguish demand from all matching lanes')
    train_lengths = {a['length'] for a in records['train']}
    _require(all(a['length'] in train_lengths for a in records['test']), 'test scale must be demonstrated')
    larger = all(min(a['input'].shape) > max(max(b['input'].shape) for b in records['train'])
                 for a in records['test'])
    crowded = (all(len(a['gates']) < 4 for a in records['train']) and
               all(len(a['gates']) == 4 and all(g['active'] for g in a['gates'])
                   for a in records['test']))
    _require(larger or crowded, 'meaningful extent or simultaneous-gate test shift')


def self_test():
    """Bounded geometric, ablation, mutation, contract, and 42-seed checks."""
    for length in range(1, 7):
        for shape, center in (((30, 30), (length + 2, length + 2)),
                              ((23, 29), (20 - length, 26 - length))):
            procedural = _phase_map(length, shape, center)
            _require(np.array_equal(procedural, _analytic_phases(length, shape, center)), 'phase recurrence')
            rotated_center = shape[1] - 1 - center[1], center[0]
            rotated = _phase_map(length, shape[::-1], rotated_center)
            # A CCW grid rotation sends N->W, E->N, S->E, W->S.
            relabel = np.array([0, 4, 1, 2, 3], dtype=np.int8)
            _require(np.array_equal(relabel[np.rot90(procedural)], rotated), 'quarter-turn covariance')
            big = _analytic_phases(length, (80, 80), (40, 40))
            y, x = 40 - center[0], 40 - center[1]
            _require(np.array_equal(procedural, big[y:y + shape[0], x:x + shape[1]]), 'clipping and reentry')
    _require(np.array_equal(_phase_map(1, (30, 30), (15, 15)),
                            _phase_map(2, (30, 30), (15, 15))), 'short-spoke radius-two convention')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, layouts, palettes, lengths, gate_counts, active_counts, sides, depths, turns = (
        set() for _ in range(9))
    shifts = Counter()
    signatures = set()
    backgrounds, tokens = set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyse(pair['input'])
                _require(np.array_equal(solve(pair['input']), pair['output']), 'input-only solve')
                shapes.add(a['input'].shape)
                palettes.add((a['background'], a['token'], a['colors']))
                backgrounds.add(a['background'])
                tokens.add(a['token'])
                lengths.add(a['length'])
                gate_counts.add(len(a['gates']))
                active_counts.add(sum(g['active'] for g in a['gates']))
                layouts.add((a['length'], a['input'].shape, a['center'],
                             tuple(g['path'][0] for g in a['gates']),
                             tuple(g['active'] for g in a['gates'])))
                for g in a['gates']:
                    depths.add(g['depth'])
                    turns.add(_turns(g['path']))
                    sides.add(g['direction'])
        test = puzzle['test'][0]['input']
        shifts['extent' if min(test.shape) > max(max(p['input'].shape) for p in puzzle['train'])
               else 'crowding'] += 1
    for puzzle in puzzles[:4]:
        pair = puzzle['train'][0]
        for quarter in range(1, 4):
            _require(np.array_equal(solve(np.rot90(pair['input'], quarter)),
                                    np.rot90(pair['output'], quarter)), 'rotated input-only solve')
        permutation = np.array([9, 3, 7, 1, 8, 0, 5, 2, 6, 4], dtype=np.int8)
        _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
                 'literal color permutation')
    _require(len(signatures) == 42 and len(layouts) > 150, 'structural diversity')
    _require(len(shapes) > 65 and len(palettes) > 90, 'geometry and palette diversity')
    _require(lengths == set(range(1, 7)), 'scale diversity')
    _require(gate_counts == {1, 2, 3, 4} and active_counts == {1, 2, 3, 4}, 'gate population diversity')
    _require(sides == set(DIRS) and depths == {1, 2, 3}, 'gate placement and depth diversity')
    _require(backgrounds == tokens == set(range(10)), 'all literals must support both roles')
    _require(max(turns) >= 8 and len(turns) >= 6, 'winding depth diversity')
    _require(set(shifts) == {'extent', 'crowding'}, 'shift diversity')
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(173, n, m)
            q = generate_puzzle(173, n, m)
            _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s])
                         for k in ('input', 'output')), 'count contract or determinism')
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    return {'seeds': 42, 'pairs': sum(len(p[s]) for p in puzzles for s in p),
            'shapes': len(shapes), 'structural_layouts': len(layouts), 'palettes': len(palettes),
            'lengths': sorted(lengths), 'gate_counts': sorted(gate_counts),
            'active_counts': sorted(active_counts), 'turn_counts': sorted(turns), 'shifts': dict(shifts)}
