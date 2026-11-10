"""Docked keys reroute a signal from their translated boundary ports.

Only Python's standard library and NumPy are used. No files are read or written.
The constructor packs local geometry; the solver decodes only the visible input.
"""

import hashlib
import numbers

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _add(p, d, n=1):
    dr, dc = DIRECTIONS[d]
    return p[0] + n * dr, p[1] + n * dc


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            for d in range(4):
                p = _add(stack[-1], d)
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
                    break
            else:
                stack.pop()
        result.append(component)
    return result


def _expand(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _rectangle(cells):
    a, b, c, d = _bbox(cells)
    return len(cells) == (b - a + 1) * (d - c + 1)


def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= s <= 30 for s in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid


def _parse_fork(cells):
    parses = []
    for d, (dr, dc) in enumerate(DIRECTIONS):
        kr = min(r * dr + c * dc for r, c in cells)
        qr = min(r * dc - c * dr for r, c in cells)
        local = {(r * dr + c * dc - kr, r * dc - c * dr - qr)
                 for r, c in cells}
        width = max(q for k, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 31)) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, length in enumerate(lengths)
                     for k in range(1, length + 1)}
        if local != expected:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((start, q, length))
        if len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1)):
            continue
        a, b, center = runs[1]
        if a + b != width:
            continue
        parses.append(dict(cells=cells, d=d, kr=kr, qr=qr, width=width,
                           q0=a, span=b - a, stop=center + 1,
                           lengths=tuple(x[2] for x in runs)))
    assert len(parses) == 1, "receiver must have a unique three-prong interpretation"
    return parses[0]


def _cell(fork, k, q):
    dr, dc = DIRECTIONS[fork['d']]
    k, q = k + fork['kr'], q + fork['qr']
    return k * dr + q * dc, k * dc - q * dr


def _port_direction(cells, port):
    r0, r1, c0, c1 = _bbox(cells)
    sides = [d for d, ok in enumerate((all(r == r0 for r, c in port),
             all(c == c1 for r, c in port), all(r == r1 for r, c in port),
             all(c == c0 for r, c in port))) if ok]
    assert len(sides) == 1, "a port must be on exactly one side, away from corners"
    d = sides[0]
    qs = sorted(c if d % 2 == 0 else r for r, c in port)
    assert qs == list(range(qs[0], qs[0] + len(qs)))
    assert all(_add(p, (d + 2) % 4) in cells - port for p in port)
    return d


def _candidate(fork, token, grid, bg):
    dr, dc = DIRECTIONS[fork['d']]
    local = {(r * dr + c * dc - fork['kr'], r * dc - c * dr - fork['qr'])
             for r, c in token['cells']}
    qs = {q for k, q in local}
    if qs != set(range(fork['q0'], fork['q0'] + fork['span'])):
        return None
    k0, k1 = min(k for k, q in local), max(k for k, q in local)
    if k0 <= fork['stop']:
        return None
    route = {_cell(fork, k, q) for k in range(fork['stop'], k1 + 1) for q in qs}
    if any(p not in token['cells'] and int(grid[p]) != bg for p in route):
        return None
    distance, length = k0 - fork['stop'], k1 - k0 + 1
    final = {_cell(fork, k, q) for k in range(fork['stop'], fork['stop'] + length) for q in qs}
    moved_port = {_add(p, (fork['d'] + 2) % 4, distance) for p in token['port']}
    return dict(route=route, final=final, port=moved_port,
                d=token['d'], distance=distance, length=length)


def _decode(value):
    grid = _grid(value)
    h, w = grid.shape
    bg = int(grid[0, 0])
    assert all(int(grid[p]) == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    components = _components(map(tuple, np.argwhere(grid != bg)))
    boundary = [s for s in components if any(r in (0, h - 1) or c in (0, w - 1) for r, c in s)]
    assert len(boundary) == 1
    launcher = boundary[0]
    ends = [p for p in launcher if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    assert len(ends) == 1
    start = ends[0]
    initial = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    line = [_add(start, initial, k) for k in range(len(launcher))]
    assert set(line) == launcher and len(line) >= 3
    signal, marker = int(grid[start]), int(grid[line[-1]])
    assert len({bg, signal, marker}) == 3
    assert all(int(grid[p]) == signal for p in line[:-1])
    forks, tokens = [], []
    for cells in components:
        if cells == launcher:
            continue
        colors = {int(grid[p]) for p in cells}
        assert not colors & {bg, signal}
        if _rectangle(cells) and marker in colors:
            assert len(colors) == 2
            body = next(iter(colors - {marker}))
            port = {p for p in cells if int(grid[p]) == marker}
            assert len(port) < len(cells - port)
            tokens.append(dict(cells=cells, color=body, port=port,
                               d=_port_direction(cells, port)))
        else:
            assert len(colors) == 1 and marker not in colors
            f = _parse_fork(cells)
            f['color'] = next(iter(colors))
            forks.append(f)
    assert forks and tokens
    occupied = set()
    for cells in components:
        assert not occupied & _expand(cells), "input objects touch"
        occupied |= cells
    matches, assigned = {}, set()
    for fi, fork in enumerate(forks):
        choices = []
        for ti, token in enumerate(tokens):
            candidate = _candidate(fork, token, grid, bg)
            if candidate is not None:
                choices.append((ti, candidate))
        assert len(choices) <= 1, "multiple keys fit one receiver"
        if choices:
            ti, candidate = choices[0]
            assert ti not in assigned, "one key fits multiple receivers"
            assigned.add(ti)
            matches[fi] = dict(candidate, ti=ti)
    return dict(grid=grid, bg=bg, signal=signal, marker=marker, launcher=launcher,
                cap=line[-1], initial=initial, forks=forks, tokens=tokens, matches=matches)


def _first_fork(origins, direction, forks, shape):
    """Used only for counterfactual evidence, not to construct solver answers."""
    front = set(origins)
    for _ in range(max(shape)):
        front = {_add(p, direction) for p in front}
        inside = {p for p in front if 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]}
        if not inside:
            return None
        hits = {i for i, f in enumerate(forks) if inside & f['cells']}
        if hits:
            return min(hits)
    return None


def _execute(a):
    grid, forks, tokens = a['grid'], a['forks'], a['tokens']
    owners = {p: i for i, f in enumerate(forks) for p in f['cells']}
    token_owner = {p: i for i, t in enumerate(tokens) for p in t['cells']}
    trace, beams = set(a['launcher']), set(a['launcher'])
    reached, inserted, moved, final_cells, sources = [], [], set(), set(), []
    origins, direction = {a['cap']}, a['initial']
    terminal = 'boundary'
    while True:
        front = {_add(p, direction) for p in origins}
        while True:
            inside = {p for p in front if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]}
            if not inside:
                front = set()
                break
            assert inside == front, "partial boundary crossing"
            assert not front & final_cells, "beam strikes an inserted key"
            assert not any(token_owner.get(p, -1) not in moved and p in token_owner for p in front), "beam strikes an undocked key"
            hits = {owners[p] for p in front if p in owners}
            if hits:
                assert len(hits) == 1 and all(p in owners for p in front), "partial receiver contact"
                fi = next(iter(hits))
                assert all(owners[p] == fi for p in front) and fi not in reached, "cycle or tied contact"
                reached.append(fi)
                trace |= forks[fi]['cells']
                if fi not in a['matches']:
                    terminal = 'unmatched_receiver'
                    front = set()
                    break
                match = a['matches'][fi]
                ti = match['ti']
                assert ti not in moved and not match['final'] & final_cells
                moved.add(ti)
                inserted.append((fi, match))
                final_cells |= match['final']
                trace |= match['route'] - match['final']
                sources.append((fi, tokens[ti]['port'], match['port'], match['d']))
                origins, direction = match['port'], match['d']
                break
            trace |= front
            beams |= front
            front = {_add(p, direction) for p in front}
        if not front:
            break
    output = np.full(grid.shape, a['bg'], dtype=np.int8)
    _paint(output, trace, a['signal'])
    for fi, match in inserted:
        token = tokens[match['ti']]
        _paint(output, match['final'], token['color'])
        _paint(output, match['port'], a['signal'])
    changed_target = False
    for i, (fi, old, new, d) in enumerate(sources):
        if i + 1 < len(reached):
            changed_target |= _first_fork(old, d, forks, grid.shape) != reached[i + 1]
    facts = dict(reached=tuple(reached), inserted=tuple(fi for fi, m in inserted),
                 terminal=terminal, unpowered_matches=set(a['matches']) - set(reached),
                 relocation_changes_target=bool(changed_target), beams=beams,
                 sources=sources, trace=trace)
    return output, facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode geometry and simulate the powered docking chain using input only."""
    return _execute(_decode(input_grid))[0]


def _template(rng, kind, compact, lateral=False, force_width=None, force_span=None):
    """Construct one receiver and its separate key in a local coordinate frame."""
    span = int(rng.integers(2, 4 if compact else 5)) if force_span is None else force_span
    clearance = 1 if compact else int(rng.integers(1, 3))
    width, q0 = span + 2 * (clearance + 1), clearance + 1
    length = int(rng.integers(3, 5 if compact else 6))
    if force_width is not None:
        length = max(length, force_width + 2)
    left, center, right = map(int, rng.integers(1, 4 if compact else 6, size=3))
    # Lateral ports must sit beyond both guards after insertion. This is a
    # geometric clearance condition, not an assumption that the center is short.
    lateral_options = [(j, pw) for pw in range(1, min(3, length - 2) + 1)
                       for j in range(1, length - pw)
                       if center + 1 + j > max(left, right)]
    forward_options = [(j, pw) for pw in range(1, span - 1)
                       for j in range(1, span - pw)]
    options = [('side', j, pw) for j, pw in lateral_options]
    if not lateral:
        options += [('front', j, pw) for j, pw in forward_options]
    if force_width is not None:
        options = [x for x in options if x[2] == force_width]
    if not options:
        return None
    mode, j, pw = options[int(rng.integers(len(options)))]
    outward = int(rng.integers(4))
    dr, dc = DIRECTIONS[outward]
    local_fork = {(0, q) for q in range(width)}
    for qa, qb, depth in ((0, 1, left), (q0, q0 + span, center), (width - 1, width, right)):
        local_fork |= {(k, q) for k in range(1, depth + 1) for q in range(qa, qb)}
    start = max(left, center, right) + int(rng.integers(2, 5 if compact else 8))
    token_q, token_span = q0, span
    if kind == 'blocked':
        token_q, token_span = q0 - 1, span + 2
    local_token = {(k, q) for k in range(start, start + length)
                   for q in range(token_q, token_q + token_span)}
    if mode == 'side':
        side = int(rng.integers(2))
        port_q = token_q if side == 0 else token_q + token_span - 1
        local_port = {(start + j + z, port_q) for z in range(pw)}
        port_d = (outward + (3 if side == 0 else 1)) % 4
    else:
        # The positive-q direction is (dc,-dr); forward means the open end.
        local_port = {(start + length - 1, token_q + j + z) for z in range(pw)}
        port_d = outward
    final_start = center + 1
    local_final = {(k, q) for k in range(final_start, final_start + length)
                   for q in range(token_q, token_q + token_span)}
    local_route = {(k, q) for k in range(final_start, start + length)
                   for q in range(token_q, token_q + token_span)}
    local_moved = {(k - start + final_start, q) for k, q in local_port}

    def convert(cells):
        return {(k * dr + q * dc, k * dc - q * dr) for k, q in cells}

    return dict(fork=convert(local_fork), token=convert(local_token), port=convert(local_port),
                final=convert(local_final), moved_port=convert(local_moved),
                reserve=convert(local_fork | local_route), d=port_d, kind=kind)


def _shift(module, rr, cc):
    return {k: {(r + rr, c + cc) for r, c in v} if isinstance(v, set) else v
            for k, v in module.items()}


def _entry_windows(cells, d, width):
    dr, dc = DIRECTIONS[d]
    near = {}
    for r, c in cells:
        q, k = r * dc - c * dr, r * dr + c * dc
        near[q] = min(near.get(q, 1000), k)
    return [(q, near[q]) for q in sorted(near)
            if all(q + j in near and near[q + j] == near[q] for j in range(width))]


def _ray_to_edge(origins, d, shape):
    result, front = set(), set(origins)
    for _ in range(max(shape)):
        front = {_add(p, d) for p in front}
        inside = {p for p in front if 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]}
        if not inside:
            return result
        if inside != front:
            return None
        result |= front
    raise AssertionError('ray did not terminate')


def _layout(rng, active, blocked, decoys, require_counterfactual):
    compact = active + blocked + decoys >= 3
    low = 24 if compact else 18
    h, w = map(int, rng.integers(low, 31, size=2))
    initial = int(rng.integers(4))
    axis = w if initial % 2 else h
    cross = int(rng.integers(2, (h if initial % 2 else w) - 2))
    start = (h - 1, cross) if initial == 0 else (cross, 0) if initial == 1 else (0, cross) if initial == 2 else (cross, w - 1)
    tail_length = int(rng.integers(2, min(5, axis - 2)))
    launcher = {_add(start, initial, k) for k in range(tail_length + 1)}
    cap = _add(start, initial, tail_length)
    budget = [350]
    coverage_widths = list(map(int, rng.permutation([1, 2])))
    coverage_spans = list(map(int, rng.permutation([2, 3])))

    def grow(modules, origins, direction, beams):
        index = len(modules)
        if index == active + blocked:
            return modules, beams
        kind = 'blocked' if index == active else 'active'
        old_input = launcher | set().union(*(m['fork'] | m['token'] for m in modules))
        old_forks = set().union(*(m['fork'] for m in modules))
        old_final = set().union(*(m['final'] for m in modules))
        old_reserve = launcher | set().union(*(m['reserve'] for m in modules))
        forbidden_input = _expand(old_input) | beams | old_reserve
        dr, dc = DIRECTIONS[direction]
        origin_k = next(iter(origins))[0] * dr + next(iter(origins))[1] * dc
        origin_q = min(r * dc - c * dr for r, c in origins)
        for _ in range(90):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            coverage = require_counterfactual and index < 2
            m = _template(rng, kind, compact, lateral=require_counterfactual and index == 0,
                          force_width=coverage_widths[index] if coverage else None,
                          force_span=coverage_spans[index] if coverage else None)
            if m is None:
                continue
            windows = _entry_windows(m['fork'], direction, len(origins))
            if not windows:
                continue
            q, k = windows[int(rng.integers(len(windows)))]
            distance = int(rng.integers(2, max(h, w)))
            dk, dq = origin_k + distance - k, origin_q - q
            m = _shift(m, dk * dr + dq * dc, dk * dc - dq * dr)
            r0, r1, c0, c1 = _bbox(m['reserve'])
            if not (1 <= r0 <= r1 < h - 1 and 1 <= c0 <= c1 < w - 1):
                continue
            if (m['fork'] | m['token']) & forbidden_input or m['reserve'] & old_input:
                continue
            if m['final'] & (old_final | beams) or m['reserve'] & old_final:
                continue
            approach = {_add(p, direction, n) for p in origins for n in range(1, distance)}
            if approach & (old_forks | old_final | m['token'] | m['final']):
                continue
            new_beams = beams | approach
            if kind == 'blocked':
                return modules + [m], new_beams
            if index == active + blocked - 1:
                ray = _ray_to_edge(m['moved_port'], m['d'], (h, w))
                if ray is None or ray & (old_forks | m['fork'] | old_final | m['final']):
                    continue
                return modules + [m], new_beams | ray
            result = grow(modules + [m], m['moved_port'], m['d'], new_beams)
            if result is not None:
                return result
        return None

    result = grow([], {cap}, initial, launcher)
    if result is None:
        return None
    modules, beams = result
    for _ in range(decoys):
        placed = False
        old_input = launcher | set().union(*(m['fork'] | m['token'] for m in modules))
        old_reserve = launcher | set().union(*(m['reserve'] for m in modules))
        forbidden = _expand(old_input) | old_reserve | beams
        for _ in range(300):
            m = _template(rng, 'decoy', True)
            if m is None:
                continue
            r0, r1, c0, c1 = _bbox(m['reserve'])
            bh, bw = r1 - r0 + 1, c1 - c0 + 1
            if bh + 2 > h or bw + 2 > w:
                continue
            m = _shift(m, int(rng.integers(1, h - bh)) - r0,
                       int(rng.integers(1, w - bw)) - c0)
            if (m['fork'] | m['token']) & forbidden or m['reserve'] & old_input:
                continue
            modules.append(m)
            placed = True
            break
        if not placed:
            return None
    return (h, w), launcher, cap, modules, beams


def _make_pair(rng, active, blocked, decoys, colors, counterfactual=False):
    bg, signal, marker, structure, *bodies = colors
    # Geometry retries retain the requested complexity and evidence obligations.
    for _ in range(160):
        layout = _layout(rng, active, blocked, decoys, counterfactual)
        if layout is None:
            continue
        shape, launcher, cap, modules, beams = layout
        grid = np.full(shape, bg, dtype=np.int8)
        _paint(grid, launcher, signal)
        grid[cap] = marker
        for i, m in enumerate(modules):
            _paint(grid, m['fork'], structure)
            _paint(grid, m['token'], bodies[i % len(bodies)])
            _paint(grid, m['port'], marker)
        try:
            a = _decode(grid)
            predicted, facts = _execute(a)
            if len(facts['inserted']) != active or len(facts['reached']) != active + blocked:
                continue
            if len(facts['unpowered_matches']) != decoys:
                continue
            if counterfactual and not facts['relocation_changes_target']:
                continue
            assert facts['terminal'] == ('unmatched_receiver' if blocked else 'boundary')
            # Ground truth comes from the packed local footprints and reserved
            # beam strips, not the input decoder's traversal or docking formula.
            output = np.full(shape, bg, dtype=np.int8)
            _paint(output, beams, signal)
            for m in modules[:active]:
                _paint(output, m['reserve'] - m['final'], signal)
            if blocked:
                _paint(output, modules[active]['fork'], signal)
            for i, m in enumerate(modules[:active]):
                _paint(output, m['final'], bodies[i % len(bodies)])
                _paint(output, m['moved_port'], signal)
            assert np.array_equal(output, predicted), 'constructed footprints disagree with input solver'
            return {'input': grid, 'output': output}
        except (AssertionError, IndexError):
            continue
    raise RuntimeError('bounded packing exhausted without reducing the requested complexity')


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            ordinal = len(puzzle['train']) + len(puzzle['test'])
            # Visible role colors change, never a hidden decoding convention.
            colors = list(np.roll(palette, ordinal % 10))[:6]
            if split == 'train' and i == 0:
                active, blocked, decoys, cf = 2, 0, 1, True
            elif split == 'train' and i == 1:
                active, blocked, decoys, cf = 1, 1, 0, False
            elif split == 'train':
                active, blocked, decoys, cf = int(rng.integers(1, 3)), int(rng.integers(2)), int(rng.integers(2)), False
            else:
                active, blocked, decoys, cf = 3, 0, int(rng.integers(2)), False
            for _ in range(8):
                pair = _make_pair(rng, active, blocked, decoys, colors, cf)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('distinct-input budget exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    order = rng.permutation(num_train)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    train_widths, test_widths = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _decode(inp)
            predicted, facts = _execute(a)
            assert np.array_equal(predicted, out) and not np.array_equal(inp, out)
            assert 1 <= len(facts['inserted']) <= 3
            assert np.count_nonzero(inp == a['bg']) > inp.size // 2
            active = set(facts['reached'])
            for fi, f in enumerate(a['forks']):
                assert min(f['lengths']) >= 1
                assert all(int(out[p]) == (a['signal'] if fi in active else a['bg']) for p in f['cells'])
            for fi in facts['inserted']:
                m = a['matches'][fi]
                t = a['tokens'][m['ti']]
                assert m['distance'] > 0 and len(m['final']) == len(t['cells'])
                assert all(int(out[p]) == t['color'] for p in m['final'] - m['port'])
                assert all(int(out[p]) == a['signal'] for p in m['port'])
                assert len(m['port']) == len(t['port'])
            if split == 'train':
                assert len(facts['inserted']) <= 2
                training.append(facts)
            else:
                assert len(facts['inserted']) == 3
            widths = train_widths if split == 'train' else test_widths
            widths.update(len(a['matches'][fi]['port']) for fi in facts['inserted'])
    assert any(f['unpowered_matches'] for f in training), 'show that geometry alone does not power a receiver'
    assert any(f['terminal'] == 'unmatched_receiver' for f in training), 'show that reachability alone does not dock a key'
    assert any(f['relocation_changes_target'] for f in training), 'show that moving the port changes the next target'
    assert {1, 2} <= train_widths and test_widths <= train_widths


def self_test():
    """42 streaming seeds; no grids, diagnostics or traces are saved to disk."""
    puzzles, shapes, structures, directions, spans, lengths, widths, distances = set(), set(), set(), set(), set(), set(), set(), set()
    relations, active_counts, palettes = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _decode(pair['input'])
                answer, facts = _execute(a)
                assert np.array_equal(answer, pair['output'])
                assert np.array_equal(solve(pair['input'].copy()), answer)
                if seed < 3:
                    for rotation in range(4):
                        assert np.array_equal(solve(np.rot90(pair['input'], rotation)), np.rot90(answer, rotation))
                    assert np.array_equal(solve(np.fliplr(pair['input'])), np.fliplr(answer))
                    permutation = np.array([7, 4, 8, 0, 9, 2, 1, 5, 3, 6], dtype=np.int8)
                    assert np.array_equal(solve(permutation[pair['input']]), permutation[answer])
                digest.update(bytes(pair['input'].shape) + pair['input'].tobytes() + answer.tobytes())
                shapes.add(pair['input'].shape)
                directions.add(a['initial'])
                palettes.add((a['bg'], a['signal'], a['marker']))
                active_counts.add(len(facts['inserted']))
                # Color-normalized hash without retaining a duplicate grid.
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in pair['input'].flat)
                structures.add(hashlib.sha256(bytes(pair['input'].shape) + normalized).digest())
                for fi in facts['inserted']:
                    f, m = a['forks'][fi], a['matches'][fi]
                    spans.add(f['span'])
                    lengths.add(m['length'])
                    widths.add(len(m['port']))
                    distances.add(m['distance'])
                    directions.add(m['d'])
                    l, c, r = f['lengths']
                    relations.add('short' if c < min(l, r) else 'long' if c > max(l, r) else 'other')
        puzzles.add(digest.digest())
    assert len(puzzles) == 42 and len(structures) >= 180 and len(shapes) >= 30
    assert directions == set(range(4)) and spans >= {2, 3, 4}
    assert lengths >= {3, 4, 5} and widths >= {1, 2} and len(distances) >= 5
    assert active_counts == {1, 2, 3} and relations >= {'short', 'long', 'other'}
    assert len(palettes) >= 30
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * nt + ne, nt, ne))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    one, two = generate_puzzle(61, 2, 1), generate_puzzle(61, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in one for x, y in zip(one[s], two[s]) for k in ('input', 'output'))
    one['test'][0]['output'][0, 0] = (int(one['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(one)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted answer accepted')
