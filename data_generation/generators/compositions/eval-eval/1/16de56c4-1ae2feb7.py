"""Clocked rails: source run periods measured in a peg-calibrated path clock.

Only standard-library modules and NumPy are used. Generation paints sampled
geometry; solve reconstructs every role and coordinate from its input alone.
"""
import math
import numpy as np


HOST_SEEDS = (0, 725766038, 316535571, 241626642, 188588543, 816039700,
              509172889, 492942827, 954273054, 299201064, 702096627, 851911549,
              784463185, 664817114, 105426809, 972084510, 169903017, 488339116,
              216512824, 672354157, 117088290, 682042815, 856968392, 874198178,
              752114693, 713457579, 810768811, 979074817, 539398429, 577141707,
              261032263, 889325246, 99894915, 899508459, 93874515, 161518039,
              751549977, 380907170, 484074052, 442417515, 41082908, 637832354)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected an integer ARC grid of size 1..30 and hues 0..9')
    return a


def _key(grid):
    a = np.asarray(grid)
    return a.shape, tuple(map(int, a.flat))


def _neighbors(p):
    r, c = p
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component, stack = {first}, [first]
        while stack:
            for p in _neighbors(stack.pop()):
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _trace(first, cells):
    path, seen, cursor = [], set(), first
    while True:
        path.append(cursor)
        seen.add(cursor)
        options = [p for p in _neighbors(cursor) if p in cells and p not in seen]
        if len(options) > 1:
            raise ValueError('an arm branches or has a nonconsecutive contact')
        if not options:
            break
        cursor = options[0]
    # Inspect the complete graph, including contacts with already visited cells.
    for i, p in enumerate(path):
        actual = {q for q in _neighbors(p) if q in cells}
        expected = set(path[max(0, i - 1):i] + path[i + 1:i + 2])
        if actual != expected:
            raise ValueError('arm is not an induced open path')
    return path


def _directions(path):
    return [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]


def _turns(path):
    d = _directions(path)
    return sum(x != y for x, y in zip(d, d[1:]))


def _parse(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[counts.argmax()])
    if counts.max() <= grid.size // 2:
        raise ValueError('background must exceed half the input')
    foreground = set(zip(*np.nonzero(grid != background)))
    if not foreground:
        raise ValueError('no machines')
    machines = []
    for component in _components(foreground):
        squares = []
        for r, c in sorted(component):
            square = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if square <= component and len({int(grid[p]) for p in square}) == 1:
                squares.append(square)
        if len(squares) != 1:
            raise ValueError('each machine needs one unique square dock')
        dock = squares[0]
        dock_color = int(grid[min(dock)])
        rest = component - dock
        starts = sorted({p for d in dock for p in _neighbors(d) if p in rest})
        if len(starts) != 2:
            raise ValueError('dock needs exactly two arm attachments')
        arms = [_trace(p, rest) for p in starts]
        if set(arms[0]) & set(arms[1]) or set(arms[0] + arms[1]) != rest:
            raise ValueError('dock arms must be disjoint and explain all foreground')
        bends = [_turns(p) for p in arms]
        if sum(b > 0 for b in bends) != 1:
            raise ValueError('one arm must bend and the other must be straight')
        track_index = int(bends[1] > 0)
        path, source = arms[track_index], arms[1 - track_index]
        # Opposite dock faces, same line through the square.
        s, t = source[0], path[0]
        if not ((s[0] == t[0] and abs(s[1] - t[1]) == 3)
                or (s[1] == t[1] and abs(s[0] - t[0]) == 3)):
            raise ValueError('arms must attach directly across opposite dock faces')
        rail_values = np.array([grid[p] for p in path])
        hues, frequencies = np.unique(rail_values, return_counts=True)
        rail = int(hues[frequencies.argmax()])
        if frequencies.max() <= len(path) // 2:
            raise ValueError('rail substrate must be locally dominant')
        repeated = [int(c) for c, n in zip(hues, frequencies) if c != rail and n == 2]
        if len(repeated) != 1 or any(n != 1 for c, n in zip(hues, frequencies)
                                     if c not in (rail, repeated[0])):
            raise ValueError('need a unique peg pair and singleton other marks')
        peg = repeated[0]
        if len({background, dock_color, rail, peg}) != 4:
            raise ValueError('structural hues must differ')
        a, b = map(int, np.flatnonzero(rail_values == peg))
        q = b - a
        extras = [i for i, c in enumerate(rail_values) if c not in (rail, peg)]
        aligned = [i for i in extras if (i - a) % q == 0]
        if len(aligned) > 1 or (aligned and a < aligned[0] < b):
            raise ValueError('terminal must be unique and beyond the peg pair')
        terminal = aligned[0] if aligned else None
        forbidden = {background, dock_color, rail, peg}
        if any(int(grid[p]) in forbidden for p in source):
            raise ValueError('source cannot use structural hues')
        if any(int(rail_values[i]) in forbidden for i in extras):
            raise ValueError('marks cannot use structural hues')
        blocks = []
        for p in source:
            color = int(grid[p])
            if blocks and blocks[-1][1] == color:
                blocks[-1] = (blocks[-1][0] + 1, color)
            else:
                blocks.append((1, color))
        machines.append({'dock': dock, 'path': path, 'source': source,
                         'rail': rail, 'peg': peg, 'a': a, 'b': b, 'q': q,
                         'blocks': blocks, 'terminal': terminal,
                         'terminal_color': None if terminal is None else int(rail_values[terminal]),
                         'extras': extras, 'values': rail_values, 'turns': _turns(path)})
    return background, machines


def _predict(input_grid, rival=None):
    grid = _grid(input_grid)
    _, machines = _parse(grid)
    output = grid.copy()
    for m in machines:
        a, q = m['a'], m['q']
        if rival == 'unit_clock':
            q = 1
        if rival == 'dock_phase':
            a = 0
        terminal = m['terminal']
        bound = terminal is not None and rival not in ('no_terminal', 'no_bound')
        lo, hi = (min(m['a'], terminal), max(m['b'], terminal)) if bound else (0, len(m['path']) - 1)
        last_bend = 0
        directions = _directions(m['path'])
        for i, p in enumerate(m['path']):
            if i > 1 and directions[i - 1] != directions[i - 2]:
                last_bend = i - 1
            if int(grid[p]) != m['rail'] or not lo <= i <= hi:
                continue
            relative = i - a
            if rival == 'turn_reset' and last_bend:
                relative = i - last_bend
            if relative % q:
                continue
            n = relative // q
            firing = []
            for rank, (length, color) in enumerate(m['blocks']):
                period = length
                if rival == 'length_plus':
                    period += 1
                elif rival == 'length_minus':
                    period = max(1, period - 1)
                if rival == 'all_ticks' or n % period == 0:
                    firing.append((rank, length, color))
            if not firing:
                continue
            if rival == 'short':
                winner = min(firing, key=lambda z: (z[1], z[0]))
            elif rival == 'long':
                winner = min(firing, key=lambda z: (-z[1], z[0]))
            elif rival == 'far':
                winner = firing[-1]
            else:
                winner = firing[0]
            color = winner[2]
            if terminal is not None and rival not in ('no_terminal', 'no_dye'):
                color = m['terminal_color']
            output[p] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all colors, source order, clocks and bounds solely from input."""
    return _predict(input_grid)


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _source_lengths(rng, maximum, count=None):
    count = int(rng.integers(1, 4)) if count is None else count
    # A finite packing composition; periods may be equal or divide each other.
    remaining, result = 12, []
    for i in range(count):
        limit = min(maximum, remaining - (count - i - 1))
        length = int(rng.integers(1, limit + 1))
        result.append(length)
        remaining -= length
    return result


def _plan(rng, roles, maximum, kind=None, priority=None, test=False):
    data = roles[4:]
    if kind in ('priority_a', 'priority_b'):
        lengths = list(priority if kind == 'priority_a' else priority[::-1])
        colors = list(data[:2] if kind == 'priority_a' else data[1::-1])
        lcm = math.lcm(*lengths)
        q = int(rng.integers(1, 4 if lcm <= 6 else 3))
        minimum = (max(lengths) + lcm + 2) * q + 1
        mode = 'priority'
    elif kind == 'bounded':
        length = int(_choose(rng, priority))
        lengths, colors = [length], [int(data[0])]
        q = int(rng.integers(2, 4))
        minimum = (3 * length + 3) * q + 1
        mode = 'bounded'
    elif kind == 'single':
        lengths, colors = [maximum], [int(_choose(rng, data))]
        q = int(rng.integers(1, 5))
        minimum = (2 * maximum + 3) * q + 1
        mode = 'single'
    else:
        lengths = _source_lengths(rng, maximum, 3 if test and rng.random() < .6 else None)
        colors = []
        # Consecutive equal hues would merge, so choose a different neighbor.
        palette = list(rng.choice(data, int(rng.integers(2, 7)), replace=False))
        for _ in lengths:
            colors.append(int(_choose(rng, [c for c in palette if not colors or c != colors[-1]])))
        q = int(rng.integers(1, 5))
        minimum = (min(lengths) + 3) * q + 1
        mode = 'random'
    return {'lengths': lengths, 'colors': colors, 'q': q,
            'minimum': minimum, 'mode': mode}


def _serpentine(row, col, width, gaps, passes):
    path = [(row, col)]
    cursor_r, cursor_c = row, col
    for j in range(passes):
        target_c = col + width - 1 if j % 2 == 0 else col
        sign = 1 if target_c > cursor_c else -1
        while cursor_c != target_c:
            cursor_c += sign
            path.append((cursor_r, cursor_c))
        if j < passes - 1:
            for _ in range(gaps[j]):
                cursor_r += 1
                path.append((cursor_r, cursor_c))
    return path


def _make_pair(rng, roles, plans, test, rotation, reflected):
    background, dock_color, rail, peg = map(int, roles[:4])
    count = len(plans)
    dock_col = max(sum(p['lengths']) for p in plans) + int(rng.integers(0, 3))
    available_width = 30 - dock_col - 2
    bands, required_width = [], 5
    for p in plans:
        passes = int(rng.integers(4, 7)) if test else int(rng.integers(2, 4))
        # Three passes guarantee space for all curriculum witnesses.
        if not test and p['minimum'] > 2 * available_width + 2:
            passes = 3
        gaps = [2] * (passes - 1)
        width_needed = math.ceil((p['minimum'] + passes - 1 - sum(gaps)) / passes)
        if width_needed > available_width:
            return None
        required_width = max(required_width, width_needed)
        bands.append((passes, gaps))
    separation = int(rng.integers(1, 3))
    height = 2 + sum(1 + sum(g) for _, g in bands) + separation * (count - 1)
    if height > 30:
        return None
    # Add geometric slack independently to pass gaps and margins.
    slack = 30 - height
    for _, gaps in bands:
        for i in range(len(gaps)):
            delta = int(rng.integers(0, min(2, slack) + 1)) if rng.random() < .45 else 0
            gaps[i] += delta
            slack -= delta
            height += delta
    width = int(rng.integers(required_width, available_width + 1))
    right_margin = int(rng.integers(0, min(2, available_width - width) + 1))
    top = int(rng.integers(1, min(3, slack + 1) + 1))
    height += top - 1
    inp = np.full((height, dock_col + 2 + width + right_margin), background, dtype=np.int8)
    out = inp.copy()
    row = top
    for p, (passes, gaps) in zip(plans, bands):
        path = _serpentine(row, dock_col + 2, width, gaps, passes)
        # Tail clipping preserves the last pass and thus the guaranteed turns.
        removable = min(width - 2, len(path) - p['minimum'])
        cut = int(rng.integers(removable + 1)) if removable > 0 else 0
        if cut:
            path = path[:-cut]
        N, q = len(path), p['q']
        mode, lengths = p['mode'], p['lengths']
        terminal = None
        if mode == 'priority':
            low = max(lengths) * q
            high = N - 1 - math.lcm(*lengths) * q
            # A nonzero emission phase refutes a dock-start interpretation.
            candidates = [a for a in range(low, high) if a % (q * lengths[0])]
        elif mode == 'single':
            low, high = lengths[0] * q, N - (lengths[0] + 1) * q
            candidates = [a for a in range(low, high) if a % (q * lengths[0])]
        elif mode == 'bounded':
            L = lengths[0]
            direction = int(_choose(rng, (-1, 1)))
            low = (2 * L + 1) * q if direction < 0 else L * q
            high = min(low + 2 * q, N - (L + 1) * q if direction < 0 else N - (2 * L + 1) * q)
            candidates = [a for a in range(low, high + 1) if a + q < N]
        else:
            candidates = [a for a in range(1, N - q - 1)
                          if any(0 <= a + k * q * min(lengths) < N
                                 for k in (-1, 1))]
        if not candidates:
            return None
        a = int(_choose(rng, candidates))
        b = a + q
        if mode == 'bounded':
            terminal = a + direction * 2 * lengths[0] * q
        elif mode == 'random' and rng.random() < .48:
            options = [s for s in range(N) if (s - a) % q == 0 and (s < a or s > b)]
            if options:
                terminal = int(_choose(rng, options))
        for coord in path:
            inp[coord] = rail
        inp[row:row + 2, dock_col:dock_col + 2] = dock_color
        cursor = dock_col
        for L, color in zip(lengths, p['colors']):
            inp[row, cursor - L:cursor] = color
            cursor -= L
        inp[path[a]] = inp[path[b]] = peg
        marks = []
        remaining_colors = list(map(int, roles[4:]))
        if terminal is not None:
            color_options = [c for c in remaining_colors if c not in p['colors']]
            terminal_color = int(_choose(rng, color_options or remaining_colors))
            inp[path[terminal]] = terminal_color
            remaining_colors.remove(terminal_color)
            marks.append(terminal)
        if q > 1:
            options = [s for s in range(N) if (s - a) % q]
            if mode == 'bounded':
                lo, hi = min(a, terminal), max(b, terminal)
                options = [s for s in options if s < lo or s > hi]
                if not options:
                    return None
                noise_count = int(rng.integers(1, 3))
            else:
                noise_count = int(_choose(rng, (0, 0, 0, 1, 2)))
            chosen = rng.choice(options, min(noise_count, len(options)), replace=False)
            chosen_colors = rng.choice(remaining_colors, len(chosen), replace=False)
            for s, color in zip(chosen, chosen_colors):
                inp[path[int(s)]] = color
                marks.append(int(s))
        # Direct construction by slice-like emissions, far-to-near overwrites.
        # No role parser or solve call is used to construct labels.
        for coord in path:
            out[coord] = inp[coord]
        out[row:row + 2, dock_col:dock_col + 2] = dock_color
        out[row, :dock_col] = inp[row, :dock_col]
        lower, upper = (0, N - 1) if terminal is None else (min(a, terminal), max(b, terminal))
        for L, color in reversed(list(zip(lengths, p['colors']))):
            step = L * q
            start = lower + (a - lower) % step
            for s in range(start, upper + 1, step):
                if inp[path[s]] == rail:
                    out[path[s]] = color if terminal is None else int(inp[path[terminal]])
        row += 1 + sum(gaps) + separation
    if np.count_nonzero(inp == background) <= inp.size // 2:
        return None
    if reflected:
        inp, out = np.fliplr(inp), np.fliplr(out)
    pair = {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
    try:
        _, machines = _parse(pair['input'])
        if any(not any(pair['input'][c] != pair['output'][c] for c in m['path']) for m in machines):
            return None
    except ValueError:
        return None
    return pair


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = list(map(int, rng.permutation(10)))
    maximum = int(rng.integers(3, 6))
    priority = _choose(rng, [(a, b) for a in range(2, maximum)
                             for b in range(a + 1, maximum + 1)
                             if b % a and math.lcm(a, b) <= 12])
    plans = [[] for _ in range(num_train)]
    slots = list(map(int, rng.permutation(num_train)))
    lessons = ['priority_a', 'priority_b', 'bounded', 'single']
    rng.shuffle(lessons)
    for j, kind in enumerate(lessons):
        plans[slots[j % num_train]].append(_plan(rng, roles, maximum, kind, priority))
    base = int(rng.integers(4))
    orientation = int(rng.integers(3))
    reflected = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', num_train), ('test', num_test)):
        for index in range(number):
            test = split == 'test'
            required = [] if test else plans[index]
            target = int(rng.integers(max(1, len(required)), 3 if test else 4))
            local_plans = list(required)
            while len(local_plans) < target:
                local_plans.append(_plan(rng, roles, maximum, priority=priority, test=test))
            rotation = ((base + 2) % 4 if test and orientation == 1 else
                        int(rng.integers(4)) if orientation == 2 else base)
            pair = None
            for attempt in range(160):
                candidate = _make_pair(rng, roles, local_plans, test, rotation, reflected)
                if candidate is not None and _key(candidate['input']) not in seen:
                    pair = candidate
                    break
                # Extra random machines may impose an unnecessarily tight width.
                if attempt == 79:
                    local_plans = list(required) or [_plan(rng, roles, maximum, priority=priority, test=test)]
            if pair is None:
                raise RuntimeError('bounded rail construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


RIVALS = ('unit_clock', 'dock_phase', 'all_ticks', 'length_plus', 'length_minus',
          'short', 'long', 'far', 'no_terminal', 'no_bound', 'no_dye', 'turn_reset')


def _training_evidence(train):
    short_near, long_near, edges, singles = False, False, set(), set()
    terminal_seen, noise_seen = False, False
    for pair in train:
        inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
        _, machines = _parse(inp)
        for m in machines:
            if m['terminal'] is not None:
                terminal_seen = True
                lo, hi = min(m['a'], m['terminal']), max(m['b'], m['terminal'])
                noise_seen |= any((s < lo or s > hi) and (s - m['a']) % m['q']
                                  and out[m['path'][s]] == inp[m['path'][s]] for s in m['extras'])
                continue
            visible = {j: [] for j in range(len(m['blocks']))}
            collisions = []
            for s, coord in enumerate(m['path']):
                if inp[coord] != m['rail'] or (s - m['a']) % m['q']:
                    continue
                n = (s - m['a']) // m['q']
                hits = [j for j, (L, _) in enumerate(m['blocks']) if n % L == 0]
                if hits and out[coord] == m['blocks'][hits[0]][1]:
                    visible[hits[0]].append(n)
                    if len(hits) > 1:
                        collisions.append(hits)
            if len(m['blocks']) == 1 and len(visible[0]) >= 2:
                singles.add(m['blocks'][0][0])
            for hits in collisions:
                near = hits[0]
                for far in hits[1:]:
                    Ln, Cn = m['blocks'][near]
                    Lf, Cf = m['blocks'][far]
                    if visible[far] and Cn != Cf:
                        short_near |= Ln < Lf
                        long_near |= Ln > Lf
                        edges.add((Cn, Cf))
    return {'short_near': short_near, 'long_near': long_near,
            'color_cycle': any((b, a) in edges for a, b in edges),
            'single_periods': singles, 'terminal': terminal_seen, 'noise': noise_seen}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            background, machines = _parse(inp)
            assert np.all(out[inp == background] == background), 'background changed'
            assert np.array_equal(out, solve(inp)), 'incorrect clock, phase, priority, bounds, or color'
            for m in machines:
                assert any(inp[c] != out[c] for c in m['path']), 'inert machine'
                for c in list(m['dock']) + m['source']:
                    assert out[c] == inp[c], 'dock or source changed'
                for s, c in enumerate(m['path']):
                    if inp[c] != m['rail']:
                        assert out[c] == inp[c], 'original annotation changed'
                    if (s - m['a']) % m['q']:
                        assert out[c] == inp[c], 'off-clock cell changed'
                infos[split].append(m)
    evidence = _training_evidence(puzzle['train'])
    assert all(evidence[k] for k in ('short_near', 'long_near', 'color_cycle', 'terminal', 'noise'))
    assert evidence['single_periods'], 'single-source recurrence not taught'
    maximum = max(evidence['single_periods'])
    assert all(L <= maximum for m in infos['test'] for L, _ in m['blocks'])
    for rival in RIVALS:
        assert any(not np.array_equal(p['output'], _predict(p['input'], rival))
                   for p in puzzle['train']), f'training admits shortcut: {rival}'
    assert min(m['turns'] for m in infos['test']) > max(m['turns'] for m in infos['train']), 'missing turn-count shift'


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def _diagnostic(terminal=False, reverse=False):
    inp = np.zeros((7, 19), dtype=np.int8)
    inp[1:3, 5:7] = 9
    # Dock-outward order is [2 blue, 3 red], or [3 red, 2 blue].
    inp[1, :5] = [2, 2, 2, 1, 1] if not reverse else [1, 1, 2, 2, 2]
    path = _serpentine(1, 7, 12, [2, 2], 3)
    for c in path:
        inp[c] = 8
    inp[path[6]] = inp[path[8]] = 7
    terminal_index = 16 if terminal == 'silent' else 26
    if terminal:
        inp[path[terminal_index]], inp[path[5]] = 3, 4
    expected = inp.copy()
    positions = {0: 2, 2: 1, 10: 1, 12: 2, 14: 1, 18: 1,
                 22: 1, 24: 2, 26: 1, 30: 1, 34: 1, 36: 2}
    if reverse:
        positions.update({18: 2, 30: 2})
    for s, color in positions.items():
        if terminal:
            if 6 <= s <= terminal_index and s != terminal_index:
                expected[path[s]] = 3
        else:
            expected[path[s]] = color
    return inp, expected


def self_test():
    """Bounded semantic diagnostics and diversity measurements over 42 seeds."""
    for terminal, reverse in ((False, False), (False, True), (True, False), ('silent', False)):
        inp, expected = _diagnostic(terminal, reverse)
        for turn in range(4):
            for reflected in (False, True):
                a, b = np.rot90(inp, turn), np.rot90(expected, turn)
                if reflected:
                    a, b = np.fliplr(a), np.fliplr(b)
                assert np.array_equal(solve(a), b), 'hand-computed path diagnostic'
                permutation = np.array([6, 9, 0, 7, 3, 2, 8, 4, 1, 5])
                assert np.array_equal(solve(permutation[a]), permutation[b]), 'semantic hue roles'
    # Equal-period occlusion and period-one dominance are legal family cases.
    inp, _ = _diagnostic()
    path = _serpentine(1, 7, 12, [2, 2], 3)
    for source, indices in (([0, 2, 2, 1, 1], (2, 10, 14, 18, 22, 26, 30, 34)),
                            ([2, 2, 2, 2, 1], (0, 2, 4, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36))):
        a = inp.copy()
        a[1, :5] = source
        expected = a.copy()
        for i in indices:
            expected[path[i]] = 1
        assert np.array_equal(solve(a), expected), 'equal or period-one dominance'
    # A non-serpentine rail exercises the documented wider geometric family.
    custom = [(1, c) for c in range(7, 14)]
    custom += [(r, 13) for r in range(2, 5)]
    custom += [(4, c) for c in range(14, 18)]
    custom += [(r, 17) for r in range(5, 11)]
    custom += [(10, c) for c in range(16, 9, -1)]
    custom += [(r, 10) for r in range(11, 15)]
    custom += [(14, c) for c in range(11, 18)]
    custom = custom[:38]
    old_inp, old_out = _diagnostic()
    a, b = np.zeros((16, 20), dtype=np.int8), np.zeros((16, 20), dtype=np.int8)
    a[1:3, 5:7] = b[1:3, 5:7] = 9
    a[1, :5] = b[1, :5] = [2, 2, 2, 1, 1]
    for before, after in zip(path, custom):
        a[after], b[after] = old_inp[before], old_out[before]
    assert np.array_equal(solve(a), b), 'path indexing must not assume serpentines'
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    shapes, structures, clocks, source_counts, periods, bends, lengths, machine_counts = (set() for _ in range(8))
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                shapes.add(inp.shape)
                mapping = {}
                structures.add((inp.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)))
                _, machines = _parse(inp)
                machine_counts.add(len(machines))
                for m in machines:
                    clocks.add(m['q'])
                    source_counts.add(len(m['blocks']))
                    periods.update(L for L, _ in m['blocks'])
                    bends.add(m['turns'])
                    lengths.add(len(m['path']))
    assert len(shapes) >= 35 and len(structures) >= 180
    assert {1, 2, 3, 4} <= clocks and source_counts == {1, 2, 3}
    assert {1, 2, 3, 4, 5} <= periods and machine_counts == {1, 2, 3}
    assert len(bends) >= 5 and len(lengths) >= 40
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1700 + 10 * a + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Minimal curricula must still refute every ablation, not merely label tests.
    for seed in HOST_SEEDS[:8]:
        generate_puzzle(seed, 2, 1)
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
                 for s in ('train', 'test')}
    _, machines = _parse(corrupted['test'][0]['input'])
    c = machines[0]['path'][machines[0]['a']]
    corrupted['test'][0]['output'][c] = (int(corrupted['test'][0]['output'][c]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('changed calibration annotation accepted')
    return {'seeds': 42, 'pairs': 210, 'input_shapes': len(shapes),
            'normalized_inputs': len(structures), 'clock_spacings': sorted(clocks),
            'source_lengths': sorted(periods), 'turn_counts': sorted(bends),
            'path_lengths': len(lengths), 'semantic_checks': 'generator-owned'}
