"""Counted safety tethers: color deletion and gravity under a routing budget.

All roles are recovered from geometry.  The coil is a persistent unary gauge,
not material that is consumed: a retractable cable pays out only what is needed.
The finite, digital peg-clearance convention is specified in task_spec.md.
"""
from __future__ import annotations

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _components(grid, bg, diagonal=True):
    unseen = {tuple(map(int, p)) for p in np.argwhere(grid != bg)}
    result = []
    while unseen:
        start = min(unseen)
        color = int(grid[start])
        todo, cells = [start], {start}
        unseen.remove(start)
        while todo:
            r, c = todo.pop()
            for dr, dc in (_D8 if diagonal else _D4):
                q = r + dr, c + dc
                if q in unseen and int(grid[q]) == color:
                    unseen.remove(q)
                    cells.add(q)
                    todo.append(q)
        result.append((color, cells))
    return result


def _rod_cells(rod):
    orientation, r, c, n, color = rod
    return {(r + (k if orientation == 'V' else 0),
             c + (k if orientation == 'H' else 0)) for k in range(n)}


def _shift(rod, d):
    return rod[0], rod[1] + d, rod[2], rod[3], rod[4]


def _route(origin, side, pegs, socket_row):
    """Exact input-only digital clearance contour, including both endpoints.

    A peg (u,v) requires clearance v+1, decaying by one per axial step.
    Rasterize the resulting 45-degree contour on a four-neighbor lattice:
    outward before down when rising; down before inward when falling.
    """
    a, c = origin
    assert socket_row >= a
    obstacles = [(r - a, side * (x - c)) for r, x in pegs]
    profile = [max([0] + [v + 1 - abs(u - t) for t, v in obstacles])
               for u in range(socket_row - a + 1)]
    assert profile[0] == profile[-1] == 0
    path = [origin]
    for u, (old, new) in enumerate(zip(profile, profile[1:])):
        assert abs(new - old) <= 1
        if new > old:
            path.append((a + u, c + side * new))
        path.append((a + u + 1, c + side * max(old, new)))
        if new < old:
            path.append((a + u + 1, c + side * new))
    assert len(path) == len(set(path)) and not set(path) & pegs
    return path


def _parse(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and min(grid.shape) >= 4
    bg = int(grid[1, 0])
    assert np.all(grid[1] == bg)
    swatches = [int(x) for x in grid[0] if x != bg]
    commands = set(swatches)
    assert 1 <= len(swatches) <= 3 and len(commands) == len(swatches)
    scene = grid.copy()
    scene[:2] = bg
    components = _components(scene, bg)
    candidates = []
    for rope, coil in components:
        if len(coil) < 4:
            continue
        a = max(r for r, c in coil)
        bottom = [p for p in coil if p[0] == a]
        if len(bottom) != 1:
            continue
        origin = bottom[0]
        other_rope = {tuple(map(int, p)) for p in np.argwhere(scene == rope)} - coil
        if len(other_rope) != 1:
            continue
        socket = next(iter(other_rope))
        if socket[1] != origin[1] or socket[0] <= a + 2:
            continue
        for rock, anchor in components:
            if rock == rope or len(anchor) < 3 or {r for r, c in anchor} != {a}:
                continue
            cols = sorted(c for r, c in anchor)
            if cols != list(range(cols[0], cols[-1] + 1)):
                continue
            side = 1 if cols[-1] == origin[1] - 1 else -1 if cols[0] == origin[1] + 1 else 0
            if not side or any(side * (c - origin[1]) <= 0 for r, c in coil - {origin}):
                continue
            if rock in commands or rope in commands:
                continue
            pegs = {tuple(map(int, p)) for p in np.argwhere(scene == rock)} - anchor
            if not pegs or min(r for r, c in pegs) <= a + 1:
                continue
            if socket[0] <= max(r + max(0, side * (c - origin[1]) + 1) for r, c in pegs):
                continue
            candidates.append((bg, commands, rope, rock, coil, anchor, pegs, origin, side, socket))
    assert len(candidates) == 1, 'ambiguous coil/anchor/socket roles'
    bg, commands, rope, rock, coil, anchor, pegs, origin, side, socket = candidates[0]
    mobile = scene.copy()
    mobile[(mobile == rope) | (mobile == rock)] = bg
    rods = []
    for color, cells in _components(mobile, bg, diagonal=False):
        rows, cols = {r for r, c in cells}, {c for r, c in cells}
        assert len(cells) >= 2 and (len(rows) == 1 or len(cols) == 1), 'non-straight rod'
        rod = ('H' if len(rows) == 1 else 'V', min(rows), min(cols), len(cells), color)
        assert _rod_cells(rod) == cells
        rods.append(rod)
    attached = [i for i, p in enumerate(rods)
                if p[0] == 'H' and p[1] == socket[0] + 1
                and socket[1] in (p[2], p[2] + p[3] - 1)]
    assert len(attached) == 1
    main = attached[0]
    beam = rods[main]
    assert beam[4] not in commands
    assert sum(p[4] == beam[4] for p in rods) == 1
    assert socket[1] == (beam[2] if side == 1 else beam[2] + beam[3] - 1)
    # The socket's vertical shaft is free of all other mobile rods.
    assert all(i == main or all(c != socket[1] for r, c in _rod_cells(p))
               for i, p in enumerate(rods))
    state = dict(bg=bg, commands=commands, rope=rope, rock=rock, coil=coil,
                 anchor=anchor, pegs=pegs, origin=origin, side=side, socket=socket,
                 rods=rods, main=main, budget=len(coil), shape=grid.shape)
    initial_path = _route(origin, side, pegs, socket[0])
    assert len(initial_path) <= state['budget']
    assert all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in initial_path)
    assert not (set(initial_path) - {socket, origin}) & set().union(*map(_rod_cells, rods))
    return state


def _settle(state, commands=None, use_tether=True, use_pegs=True):
    """Greatest simultaneously movable set; supports do not fall in scan order."""
    commands = state['commands'] if commands is None else set(commands)
    rods = [p for p in state['rods'] if p[4] not in commands]
    main_color = state['rods'][state['main']][4]
    # The attached beam has a unique color in the generated grammar.
    main = next(i for i, p in enumerate(rods) if p[4] == main_color)
    fixed = state['anchor'] | state['pegs'] | state['coil']
    h, w = state['shape']
    for _ in range(h * (len(rods) + 1)):
        cells = [_rod_cells(p) for p in rods]
        targets = [{(r + 1, c) for r, c in s} for s in cells]
        moving = {i for i, target in enumerate(targets)
                  if all(r < h for r, c in target) and not target & fixed}
        if use_tether and main in moving:
            socket_row = rods[main][1]  # socket after one downward move
            if use_pegs:
                needed = len(_route(state['origin'], state['side'], state['pegs'], socket_row))
            else:
                needed = socket_row - state['origin'][0] + 1
            if needed > state['budget']:
                moving.remove(main)
        while True:
            blocked = {i for i in moving for j in range(len(rods))
                       if j not in moving and targets[i] & cells[j]}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return rods, main
        rods = [_shift(p, 1) if i in moving else p for i, p in enumerate(rods)]
    raise AssertionError('gravity failed to converge')


def _render_output(grid, state, rods, main, draw_cable=True):
    out = np.full(grid.shape, state['bg'], dtype=np.int8)
    out[:2] = grid[:2]
    for color, cells in ((state['rock'], state['anchor'] | state['pegs']),
                         (state['rope'], state['coil'])):
        for p in cells:
            out[p] = color
    for rod in rods:
        for p in _rod_cells(rod):
            assert out[p] == state['bg']
            out[p] = rod[4]
    socket = rods[main][1] - 1, state['socket'][1]
    path = _route(state['origin'], state['side'], state['pegs'], socket[0]) if draw_cable else [socket]
    for p in path:
        assert out[p] in (state['bg'], state['rope'])
        out[p] = state['rope']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict solely from the queried grid; never reads examples or seed state."""
    grid = np.asarray(input_grid)
    state = _parse(grid)
    rods, main = _settle(state)
    return _render_output(grid, state, rods, main)


def _obstacle(rng):
    kind = int(rng.integers(5))
    if kind == 0:
        raw = {(0, 0)}
    elif kind == 1:
        n = int(rng.integers(2, 5))
        raw = {(u, 0) for u in range(n)} if rng.random() < .5 else {(0, v) for v in range(n)}
    elif kind == 2:
        h, w = map(int, rng.integers(1, 4, size=2))
        raw = {(u, v) for u in range(h) for v in range(w)}
    elif kind == 3:
        rad = int(rng.integers(1, 3))
        raw = {(u, v) for u in range(2 * rad + 1) for v in range(2 * rad + 1)
               if abs(u - rad) + abs(v - rad) <= rad}
    else:
        n = int(rng.integers(2, 5))
        raw = {(u, v) for u in range(n) for v in range(n - u)}
        if rng.random() < .5:
            raw = {(n - 1 - u, v) for u, v in raw}
    lateral = int(rng.integers(0, 3))
    near = max(2, max(v + 1 - u for u, v in raw) + lateral + int(rng.integers(0, 3)))
    return {(u + near, v + lateral) for u, v in raw}


def _blueprint(rng, stacks=1, mode='release', long_drop=5):
    """Procedural support geometry; retries only ensure space and readability."""
    for _ in range(160):
        bg, rock, rope, beam, cap, prop_a, prop_b, landing, shaft, absent = map(int, rng.permutation(10))
        a = int(rng.integers(5, 8))
        rail = int(rng.integers(3, 6))
        c = rail + int(rng.integers(0, 3))
        local_pegs = _obstacle(rng)
        # Additional separated pins on the same departing tangent, sometimes.
        if rng.random() < .3:
            u, v = max(local_pegs, key=lambda p: p[0] + p[1])
            local_pegs.add((u + 2, v - 2))
        peak = max(v + 1 for u, v in local_pegs)
        finish = max(u + max(0, v + 1) for u, v in local_pegs)
        socket_u = finish + int(rng.integers(1, 4))
        b = a + socket_u + 1
        n = peak + (int(rng.integers(10, 14)) if stacks == 2 else int(rng.integers(6, 10)))
        gap = long_drop + int(rng.integers(2, 5))
        h = b + gap + 1
        if h > 30 or c + n + 1 > 30:
            continue
        anchor = {(a, x) for x in range(c - rail, c)}
        pegs = {(a + u, c + v) for u, v in local_pegs}
        xs = sorted(map(int, rng.choice(np.arange(c + 1, c + n), size=int(rng.integers(2, 4)), replace=False)))
        # Equal-height temporary legs can share an identity, or be redundant
        # differently colored supports.  A shorter landing is a separate rod.
        rods = [('H', b, c, n, beam)]
        for j, x in enumerate(xs):
            color = prop_b if mode == 'redundant' and j else prop_a
            rods.append(('V', b + 1, x, gap, color))
        if mode == 'post':
            options = [x for x in range(c + 1, c + n) if x not in xs]
            if not options:
                continue
            x = int(rng.choice(options))
            delta = int(rng.integers(1, 3))
            rods.append(('V', b + delta + 1, x, gap - delta, landing))
        first_x = c + peak + 2
        usable = n - peak - 2
        if stacks == 2 and usable < 7:
            continue
        starts = [first_x] if stacks == 1 else [first_x, first_x + usable // 2 + 1]
        for j, left in enumerate(starts):
            right_limit = (starts[j + 1] - 2) if j + 1 < len(starts) else c + n - 1
            length = int(rng.integers(2, min(4, right_limit - left + 1) + 1))
            x = int(rng.integers(left, left + length))
            height = int(rng.integers(2, 5))
            rods += [('V', b - height, x, height, shaft),
                     ('H', b - height - 1, left, length, cap)]
        occupied = set()
        good = True
        for rod in rods:
            cells = _rod_cells(rod)
            if occupied & cells or cells & (pegs | anchor) or min(r for r, x in cells) < 2:
                good = False
                break
            occupied |= cells
        if not good:
            continue
        # Same-color neighbors must remain separate straight input components.
        if any(p[4] == q[4] and any((r + dr, x + dc) in _rod_cells(q)
               for r, x in _rod_cells(p) for dr, dc in _D4)
               for i, p in enumerate(rods) for q in rods[:i]):
            continue
        # Drop additional rods onto the loaded scaffold.  These can bridge
        # payload branches, build deeper stacks, or provide alternate landings;
        # they are not independent side puzzles.  The cable shaft stays clear.
        extras = int(rng.integers(0, 5))
        for _ in range(extras * 5):
            if extras == 0:
                break
            orientation = str(rng.choice(['H', 'V']))
            length = int(rng.integers(2, min(6, usable) + 1))
            x_max = c + n - length if orientation == 'H' else c + n - 1
            x = int(rng.integers(first_x, x_max + 1))
            color = int(rng.choice([cap, shaft, prop_a, prop_b, landing]))
            p = (orientation, a + 2, x, length, color)
            occupied = set().union(*map(_rod_cells, rods))
            if _rod_cells(p) & (occupied | pegs | anchor):
                continue
            while max(r for r, x in _rod_cells(_shift(p, 1))) < h and not _rod_cells(_shift(p, 1)) & (occupied | pegs | anchor):
                p = _shift(p, 1)
            if any(q[4] == color and any((r + dr, x + dc) in _rod_cells(q)
                   for r, x in _rod_cells(p) for dr, dc in _D4) for q in rods):
                continue
            rods.append(p)
            extras -= 1
        base = dict(bg=bg, rock=rock, rope=rope, beam=beam, cap=cap,
                    prop_a=prop_a, prop_b=prop_b, shaft=shaft, absent=absent,
                    anchor=anchor, pegs=pegs, origin=(a, c), rods=rods,
                    socket=(b - 1, c), shape=(h, c + n + int(rng.integers(0, 3))),
                    reflect=bool(rng.integers(2)))
        try:
            path = _route(base['origin'], 1, pegs, b - 1)
            assert not (set(path) - {base['socket'], base['origin']}) & occupied
        except AssertionError:
            continue
        return base
    raise AssertionError('bounded blueprint construction exhausted')


def _coil(rng, origin, count):
    a, c = origin
    min_span = (count - 2) // (a - 2) + 1
    span = int(rng.integers(min_span, min(30 - c - 1, min_span + 4) + 1))
    cells = {origin, (a - 1, c + 1)}
    directions = _D8 if rng.random() < .6 else _D4
    # All stored cells above the socket are on its outward side.  The bottom
    # singleton and its adjacent straight anchor make the role unambiguous.
    while len(cells) < count:
        frontier = sorted({(r + dr, x + dc) for r, x in cells for dr, dc in directions
                           if 2 <= r + dr < a and c + 1 <= x + dc <= c + span} - cells)
        assert frontier
        style = int(rng.integers(3))
        if style == 0:
            weights = np.array([1 + sum((r + dr, x + dc) in cells for dr, dc in _D8)
                                for r, x in frontier], dtype=float) ** 2
            p = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))]
        else:
            p = frontier[int(rng.integers(len(frontier)))]
        cells.add(p)
    return cells


def _make_pair(rng, base, drop, commands):
    path = _route(base['origin'], 1, base['pegs'], base['socket'][0])
    budget = len(path) + drop
    coil = _coil(rng, base['origin'], budget)
    h, w = base['shape']
    w = max(w, max(c for r, c in coil) + 1)
    assert w <= 30
    grid = np.full((h, w), base['bg'], dtype=np.int8)
    positions = rng.choice(w, len(commands), replace=False)
    for x, color in zip(positions, commands):
        grid[0, x] = color
    for color, cells in ((base['rock'], base['anchor'] | base['pegs']),
                         (base['rope'], coil | {base['socket']})):
        for p in cells:
            assert grid[p] == base['bg']
            grid[p] = color
    for rod in base['rods']:
        for p in _rod_cells(rod):
            assert grid[p] == base['bg']
            grid[p] = rod[4]
    if base['reflect']:
        grid = grid[:, ::-1].copy()
    state = _parse(grid)
    # The pictured scaffold starts stable, even before applying the tether.
    initial, _ = _settle(state, commands=set(), use_tether=False)
    assert initial == state['rods']
    return {'input': grid, 'output': solve(grid)}


def _count(value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError('illegal example count')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    seen = set()

    def unique_pair(base, drop, commands):
        for _ in range(40):
            pair = _make_pair(rng, base, drop, commands)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                return pair
        raise AssertionError('bounded distinct-input construction exhausted')

    small, large = int(rng.integers(1, 4)), int(rng.integers(4, 7))
    base = _blueprint(rng, long_drop=large)
    # Two controlled contrasts teach the meter and an independently removable
    # upper support.  Remaining examples vary structure and support outcomes.
    train = [unique_pair(base, small, [base['prop_a'], base['shaft']]),
             unique_pair(base, large, [base['prop_a']])]
    for index in range(2, num_train):
        mode = 'redundant' if index == 2 else 'post' if index == 3 else str(rng.choice(['release', 'redundant', 'post', 'absent', 'upper']))
        drop = int(rng.integers(3, 7))
        b = _blueprint(rng, stacks=int(rng.integers(1, 3)), mode=mode, long_drop=drop)
        cmd = [b['absent']] if mode == 'absent' else [b['shaft']] if mode == 'upper' else [b['prop_a']]
        if mode in ('release', 'post') and rng.random() < .6:
            cmd.append(b['shaft'])
        if rng.random() < .2 and mode not in ('absent', 'redundant'):
            cmd.append(b['absent'])
        train.append(unique_pair(b, drop, cmd))
    tests = []
    # Early stops need visible teaching evidence. With the minimum two
    # demonstrations, tests recombine only their released, cable-limited loads.
    test_modes = ['release', 'release']
    if num_train >= 4:
        test_modes.append('post')
    if num_train >= 3:
        test_modes.append('redundant')
    for index in range(num_test):
        # More payload branches: new connected support topology, same mechanics.
        mode = str(rng.choice(test_modes))
        drop = int(rng.integers(2, 8))
        b = _blueprint(rng, stacks=2, mode=mode, long_drop=drop)
        cmd = [b['prop_a']]
        if mode != 'redundant' and rng.random() < .65:
            cmd.append(b['shaft'])
        if rng.random() < .2:
            cmd.append(b['absent'])
        tests.append(unique_pair(b, drop, cmd))
    rng.shuffle(train)
    rng.shuffle(tests)
    puzzle = {'train': train, 'test': tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def _settle_events(state):
    """Cross-check: lowest rods first, jump to the next collision or cable cap.

    This uses the closed-form length on the vertical tail, instead of invoking
    the route for every hypothetical move.  Only used by the bounded self-test.
    """
    rods = [p for p in state['rods'] if p[4] not in state['commands']]
    color = state['rods'][state['main']][4]
    main = next(i for i, p in enumerate(rods) if p[4] == color)
    h, w = state['shape']
    fixed = state['anchor'] | state['pegs'] | state['coil']
    initial_cost = len(_route(state['origin'], state['side'], state['pegs'], state['socket'][0]))
    max_beam_row = state['socket'][0] + 1 + state['budget'] - initial_cost
    for _ in range(len(rods) + 1):
        changed = False
        for i in sorted(range(len(rods)), key=lambda j: max(r for r, c in _rod_cells(rods[j])), reverse=True):
            blocked = fixed | set().union(*(_rod_cells(p) for j, p in enumerate(rods) if j != i))
            current = _rod_cells(rods[i])
            limit = h - 1 - max(r for r, c in current)
            if i == main:
                limit = min(limit, max_beam_row - rods[i][1])
            drop = 0
            for d in range(1, limit + 1):
                if {(r + d, c) for r, c in current} & blocked:
                    break
                drop = d
            if drop:
                rods[i] = _shift(rods[i], drop)
                changed = True
        if not changed:
            return rods, main
    raise AssertionError('event settlement did not converge')


def _features(pair):
    s = _parse(pair['input'])
    final, main = _settle(s)
    before = s['rods'][s['main']]
    path0 = _route(s['origin'], s['side'], s['pegs'], s['socket'][0])
    path1 = _route(s['origin'], s['side'], s['pegs'], final[main][1] - 1)
    initial = {p: p for p in s['rods'] if p[4] not in s['commands']}
    # Stable rod ordering is retained through deletion and synchronous motion.
    distances = [q[1] - p[1] for p, q in zip(initial, final)]
    infinity, mi = _settle(s, use_tether=False)
    no_pegs, mp = _settle(s, use_pegs=False)
    no_command, mc = _settle(s, commands=set())
    relative_pegs = tuple(sorted((r - s['origin'][0], s['side'] * (c - s['origin'][1])) for r, c in s['pegs']))
    return dict(shape=s['shape'], background=s['bg'], side=s['side'],
                budget=s['budget'], route_initial=len(path0), route_final=len(path1),
                detour=len(path0) - (s['socket'][0] - s['origin'][0] + 1),
                main_drop=final[main][1] - before[1],
                tether_tight=len(path1) == s['budget'],
                rods=len(s['rods']), commands=len(s['commands']),
                distinct_falls=tuple(sorted(set(distances) - {0})),
                cable_matters=infinity[mi][1] != final[main][1],
                pegs_matter=no_pegs[mp][1] != final[main][1],
                deletion_matters=no_command[mc][1] != final[main][1],
                pegs=relative_pegs,
                structure=tuple(p[:4] for p in s['rods']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = (np.asarray(pair[k]) for k in ('input', 'output'))
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            state = _parse(inp)
            assert np.array_equal(inp[:2], out[:2])
            assert all(out[p] == state['rock'] for p in state['anchor'] | state['pegs'])
            assert all(out[p] == state['rope'] for p in state['coil'])
            initial, _ = _settle(state, commands=set(), use_tether=False)
            assert initial == state['rods'], 'input scaffold must be stable'
            rods, main = _settle(state)
            assert all(p[4] not in state['commands'] for p in rods)
            original = [p for p in state['rods'] if p[4] not in state['commands']]
            assert all(p[0] == q[0] and p[2:] == q[2:] and p[1] <= q[1]
                       for p, q in zip(original, rods)), 'rigid vertical translations only'
            route = _route(state['origin'], state['side'], state['pegs'], rods[main][1] - 1)
            assert len(route) <= state['budget']
            assert route[0] == state['origin']
            assert route[-1] == (rods[main][1] - 1, state['socket'][1])
            assert all(abs(r - s) + abs(c - d) == 1
                       for (r, c), (s, d) in zip(route, route[1:])), 'cable must be orthogonal'
            assert all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] for r, c in route)
            assert set(route) & state['coil'] == {state['origin']}
            assert np.count_nonzero(out[2:] == state['rope']) == state['budget'] + len(route) - 1
            assert np.count_nonzero(out[2:] == state['rock']) == len(state['anchor'] | state['pegs'])
            for color in {p[4] for p in state['rods']}:
                assert np.count_nonzero(out[2:] == color) == sum(p[3] for p in rods if p[4] == color), 'rod material changed'
            expected = _render_output(inp, state, rods, main)
            assert np.array_equal(out, expected)
            # A further step is blocked by floor, another rod, or the meter.
            for i, rod in enumerate(rods):
                target = _rod_cells(_shift(rod, 1))
                physically_blocked = (any(r >= inp.shape[0] for r, c in target)
                                      or bool(target & (state['anchor'] | state['pegs'] | state['coil']))
                                      or any(target & _rod_cells(q) for j, q in enumerate(rods) if i != j))
                budget_blocked = i == main and len(_route(state['origin'], state['side'], state['pegs'], rod[1])) > state['budget']
                assert physically_blocked or budget_blocked
            f = _features(pair)
            if split == 'train':
                training.append(f)
            else:
                testing.append(f)
    lessons = [f for f in training if f['cable_matters'] and f['pegs_matter'] and f['deletion_matters']]
    assert len(lessons) >= 2, 'two coupled demonstrations are required'
    assert len({f['main_drop'] for f in lessons}) >= 2
    assert any(len(f['distinct_falls']) >= 2 for f in lessons), 'missing differential gravity lesson'
    if any(not f['tether_tight'] and f['main_drop'] == 0 for f in testing):
        assert any(not f['tether_tight'] and f['main_drop'] == 0 for f in training), 'unshown redundant support'
    if any(not f['tether_tight'] and f['main_drop'] > 0 for f in testing):
        assert any(not f['tether_tight'] and f['main_drop'] > 0 for f in training), 'unshown early landing'


def self_test():
    # A two-cell rectangular peg: rise outside before moving down, and descend
    # before moving inward.  These corners cannot cut through the obstacle.
    route = _route((0, 0), 1, {(2, 1), (3, 1)}, 6)
    assert route == [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2),
                     (3, 2), (4, 2), (4, 1), (5, 1), (5, 0), (6, 0)]
    assert len(route) == 11
    shapes, pegs, structures, budgets, backgrounds, sides, rod_counts = set(), set(), set(), set(), set(), set(), set()
    fingerprints, slack_cases, differential = set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for pair in puzzle['train'] + puzzle['test']:
            assert np.array_equal(solve(pair['input'].copy()), pair['output'])
            state = _parse(pair['input'])
            assert _settle_events(state) == _settle(state)
            f = _features(pair)
            shapes.add(f['shape']); pegs.add(f['pegs']); structures.add((f['pegs'], f['structure']))
            budgets.add(f['budget']); backgrounds.add(f['background']); sides.add(f['side']); rod_counts.add(f['rods'])
            slack_cases += not f['tether_tight']
            differential += len(f['distinct_falls']) > 1
            # Palette identity is entirely geometric, so arbitrary relabeling
            # must commute with the transformation.
            perm = np.roll(np.arange(10, dtype=np.int8), 3)
            assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
            assert np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1])
        # Edit the visible gauge in a taught cable-limited scene. A connected
        # one-cell shortening must stop the SAME beam one row sooner, without
        # regenerating anything or passing the original budget to solve.
        lesson = next(p for p in puzzle['train'] if _features(p)['cable_matters'])
        state = _parse(lesson['input'])
        original_rods, main = _settle(state)
        for p in sorted(state['coil'] - {state['origin']}):
            shorter = lesson['input'].copy()
            shorter[p] = state['bg']
            try:
                decoded = _parse(shorter)
            except AssertionError:
                continue
            if decoded['coil'] != state['coil'] - {p}:
                continue
            prediction = solve(shorter)
            beam_color = state['rods'][state['main']][4]
            beam_rows = np.argwhere(prediction[2:] == beam_color)[:, 0] + 2
            assert np.all(beam_rows == original_rods[main][1] - 1)
            break
        else:
            raise AssertionError('no connected one-cell gauge shortening found')
    assert len(fingerprints) == 42
    assert len(shapes) >= 30 and len(pegs) >= 20 and len(structures) >= 100
    assert len(budgets) >= 10 and len(backgrounds) == 10 and sides == {-1, 1}
    assert len(rod_counts) >= 4 and slack_cases >= 42 and differential >= 42
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(71, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(4)
    p = damaged['test'][0]
    s = _parse(p['input'])
    p['output'][next(iter(s['pegs']))] = s['bg']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError('validator failed to detect damaged fixed peg')
