"""Mobile relay circuit: color-key release, settle, then emit."""
from dataclasses import dataclass, replace
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class Piece:
    kind: str
    r: int
    c: int
    h: int
    w: int
    color: int
    side: int = 0
    offset: int = 1
    key_r: int = 1
    key_c: int = 1

    def cells(self):
        return {(r, c) for r in range(self.r, self.r + self.h)
                for c in range(self.c, self.c + self.w)}

    def port(self):
        if self.side == 0:
            return self.r, self.c + self.offset
        if self.side == 2:
            return self.r + self.h - 1, self.c + self.offset
        if self.side == 1:
            return self.r + self.offset, self.c + self.w - 1
        return self.r + self.offset, self.c


def _settle(pieces, height, anchors=()):
    """Synchronous unit steps; a moving supporter can move with its load."""
    result = dict(pieces)
    anchors = set(anchors)
    for _ in range(height * (len(result) + 1)):
        cells = {i: p.cells() for i, p in result.items()}
        targets = {i: {(r + 1, c) for r, c in s} for i, s in cells.items()}
        moving = {i for i, s in targets.items() if all(r < height for r, c in s) and not s & anchors}
        while True:
            blocked = {i for i in moving if any(targets[i] & cells[j]
                       for j in cells if j not in moving)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        result = {i: replace(p, r=p.r + 1) if i in moving else p
                  for i, p in result.items()}
    raise AssertionError("gravity did not converge")


def _settle_direct(pieces, height, anchors=()):
    """Cross-check by dropping lowest pieces to their next collision."""
    result = dict(pieces)
    anchors = set(anchors)
    for _ in range(height * (len(result) + 1)):
        changed = False
        for i in sorted(result, key=lambda j: result[j].r + result[j].h, reverse=True):
            p = result[i]
            for distance in range(1, height + 1):
                candidate = replace(p, r=p.r + distance)
                if candidate.r + candidate.h > height or candidate.cells() & anchors or any(
                        candidate.cells() & q.cells() for j, q in result.items() if j != i):
                    distance -= 1
                    break
            if distance:
                result[i] = replace(p, r=p.r + distance)
                changed = True
        if not changed:
            return result
    raise AssertionError("direct settling did not converge")


def _hit(pieces, origin, side, shape):
    dr, dc = DIRS[side]
    candidates = []
    for i, p in pieces.items():
        if p.kind != 'relay':
            continue
        for r, c in p.cells():
            distance = (r - origin[0]) * dr + (c - origin[1]) * dc
            if distance > 0 and (r - origin[0]) * dc == (c - origin[1]) * dr:
                candidates.append((distance, i))
    return min(candidates)[1] if candidates else None


def _anchors(shape, cap, initial):
    """The launcher is a fixed solid anchored to its boundary."""
    dr, dc = DIRS[initial]
    cell, cells = cap, set()
    while 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]:
        cells.add(cell)
        cell = cell[0] - dr, cell[1] - dc
    return cells


def _run(pieces, shape, cap, initial, *, gravity=True, stale_port=False):
    state = dict(pieces)
    activated, events = [], []
    origin, side = cap, initial
    while True:
        target = _hit(state, origin, side, shape)
        if target is None or target in activated:
            break
        p = state[target]
        old_port = p.port()
        deleted = [i for i, q in state.items() if q.kind == 'rod' and q.color == p.color]
        before = dict(state)
        state = {i: q for i, q in state.items() if i not in deleted}
        if gravity:
            state = _settle(state, shape[0], _anchors(shape, cap, initial))
        activated.append(target)
        falls = {i: q.r - before[i].r for i, q in state.items()}
        events.append({'target': target, 'key': p.color, 'deleted': deleted,
                       'falls': falls, 'source_fall': falls[target],
                       'old_port': old_port, 'port': state[target].port(),
                       'incoming': side, 'outgoing': state[target].side})
        origin = old_port if stale_port else state[target].port()
        side = state[target].side
    return state, activated, events


def _render(shape, bg, signal, frame, tail, cap, pieces, activated=()):
    grid = np.full(shape, bg, dtype=np.int8)
    for i, p in pieces.items():
        grid[p.r:p.r + p.h, p.c:p.c + p.w] = (
            p.color if p.kind == 'rod' else signal if i in activated else frame)
        if p.kind == 'relay':
            grid[p.r + p.key_r, p.c + p.key_c] = p.color
            grid[p.port()] = signal
    for cell in tail:
        grid[cell] = signal
    grid[cap] = signal if activated else frame
    return grid


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    group.add(nxt)
        result.append(group)
    return result


def _parse(grid):
    grid = np.asarray(grid)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
    h, w = grid.shape
    bg = int(grid[0, 0])
    assert all(grid[cell] == bg for cell in ((0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    boundary = {(r, c) for r in range(h) for c in range(w)
                if (r in (0, h - 1) or c in (0, w - 1)) and grid[r, c] != bg}
    launch_options = []
    for start in sorted(boundary):
        initial = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
        dr, dc = DIRS[initial]
        signal = int(grid[start])
        tail, cell = [], start
        while 0 <= cell[0] < h and 0 <= cell[1] < w and grid[cell] == signal:
            tail.append(cell)
            cell = cell[0] + dr, cell[1] + dc
        if 2 <= len(tail) <= 4 and 0 <= cell[0] < h and 0 <= cell[1] < w:
            frame = int(grid[cell])
            around = [(cell[0] + rr, cell[1] + cc) for rr, cc in DIRS]
            isolated = all(not (0 <= r < h and 0 <= c < w) or
                           (r, c) == tail[-1] or grid[r, c] == bg for r, c in around)
            if frame not in (bg, signal) and isolated:
                launch_options.append((signal, frame, tail, cell, initial))
    assert len(launch_options) == 1, 'launcher must be unique'
    signal, frame, tail, cap, initial = launch_options[0]
    launcher = set(tail) | {cap}
    marked = {tuple(map(int, x)) for x in np.argwhere((grid == frame) | (grid == signal))} - launcher
    pieces, explained = {}, set(launcher)
    for component in _components(marked):
        r0, r1 = min(r for r, c in component), max(r for r, c in component)
        c0, c1 = min(c for r, c in component), max(c for r, c in component)
        assert r1 - r0 >= 2 and c1 - c0 >= 2
        rectangle = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        key = rectangle - component
        assert len(key) == 1
        kr, kc = next(iter(key))
        assert r0 < kr < r1 and c0 < kc < c1
        color = int(grid[kr, kc])
        assert color not in (bg, signal, frame)
        port = [cell for cell in component if grid[cell] == signal]
        assert len(port) == 1
        pr, pc = port[0]
        sides = [d for d, ok in enumerate((pr == r0, pc == c1, pr == r1, pc == c0)) if ok]
        assert len(sides) == 1, 'port must be on exactly one side'
        side = sides[0]
        offset = pc - c0 if side % 2 == 0 else pr - r0
        pieces[len(pieces)] = Piece('relay', r0, c0, r1 - r0 + 1, c1 - c0 + 1,
                                    color, side, offset, kr - r0, kc - c0)
        explained |= rectangle
    assert pieces
    leftover = {tuple(map(int, x)) for x in np.argwhere(grid != bg)} - explained
    for color in sorted({int(grid[cell]) for cell in leftover}):
        assert color not in (bg, signal, frame)
        for component in _components({cell for cell in leftover if grid[cell] == color}):
            r0, r1 = min(r for r, c in component), max(r for r, c in component)
            c0, c1 = min(c for r, c in component), max(c for r, c in component)
            assert (r0 == r1 or c0 == c1) and len(component) >= 2
            assert len(component) == (r1 - r0 + 1) * (c1 - c0 + 1)
            pieces[len(pieces)] = Piece('rod', r0, c0, r1 - r0 + 1, c1 - c0 + 1, color)
    assert np.array_equal(_render(grid.shape, bg, signal, frame, tail, cap, pieces), grid)
    assert _settle(pieces, h, launcher) == pieces, 'input must be gravity stable'
    return grid.shape, bg, signal, frame, tail, cap, initial, pieces


def solve(input_grid: np.ndarray) -> np.ndarray:
    shape, bg, signal, frame, tail, cap, initial, pieces = _parse(input_grid)
    state, activated, _ = _run(pieces, shape, cap, initial)
    out = _render(shape, bg, signal, frame, tail, cap, state, activated)
    out[cap] = signal
    return out


def _drop(p, pieces, height):
    occupied = set().union(*(q.cells() for q in pieces.values())) if pieces else set()
    for r in range(1, height - p.h + 1):
        p = replace(p, r=r)
        if p.cells() & occupied:
            return None if r == 1 else replace(p, r=r - 1)
    return replace(p, r=height - p.h)


def _adjacent(a, b):
    return any((r + dr, c + dc) in b for r, c in a for dr, dc in DIRS)


def _scaffold(rng, count, large, colors):
    bg, signal, frame = colors[:3]
    palette = list(colors[3:3 + int(rng.integers(3, 8))])
    height = int(rng.integers(17 if large else 12, 31))
    width = int(rng.integers(20 if large else 14, 31))
    pieces = {}
    columns = rng.choice(np.arange(3, width - 3), size=min(int(rng.integers(3, 8)), (width - 6)), replace=False)
    for col in columns:
        length = int(rng.integers(2, min(12, height - 6)))
        p = Piece('rod', height - length, int(col), length, 1, 0)
        if any(_adjacent(p.cells(), q.cells()) for q in pieces.values()):
            continue
        pieces[len(pieces)] = p
    # Drop shelves, devices and separators rather than assigning a fixed slot.
    for _ in range(int(rng.integers(0, 4))):
        length = int(rng.integers(3, min(width - 4, 14)))
        col = int(rng.integers(2, width - length - 1))
        p = _drop(Piece('rod', 1, col, 1, length, 0), pieces, height)
        if p is not None and p.r >= 2:
            pieces[len(pieces)] = p
    relay_ids = []
    for _ in range(count):
        for attempt in range(50):
            ph, pw = int(rng.integers(3, 6)), int(rng.integers(3, 7))
            col = int(rng.integers(2, width - pw - 1))
            p = _drop(Piece('relay', 1, col, ph, pw, 0,
                            key_r=int(rng.integers(1, ph - 1)), key_c=int(rng.integers(1, pw - 1))), pieces, height)
            if p is None or p.r < 2:
                continue
            if any(q.kind == 'relay' and _adjacent(p.cells(), q.cells()) for q in pieces.values()):
                # A thin rod explicitly separates otherwise touching devices.
                shelf = _drop(Piece('rod', 1, col, 1, pw, 0), pieces, height)
                if shelf is not None and shelf.r > ph + 2:
                    pieces[len(pieces)] = shelf
                continue
            relay_ids.append(len(pieces))
            pieces[len(pieces)] = p
            break
        else:
            return None
    for _ in range(int(rng.integers(0, 5))):
        vertical = rng.random() < .35
        length = int(rng.integers(2, min(9, height - 3) if vertical else min(14, width - 3)))
        col = int(rng.integers(2, width - (1 if vertical else length) - 1))
        p = _drop(Piece('rod', 1, col, length if vertical else 1, 1 if vertical else length, 0), pieces, height)
        if p is not None and p.r >= 2:
            pieces[len(pieces)] = p
    # Neighboring rods need different colors so their identities are explicit.
    for i, p in list(pieces.items()):
        if p.kind != 'rod':
            continue
        forbidden = {q.color for j, q in pieces.items() if j < i and q.kind == 'rod'
                     and _adjacent(p.cells(), q.cells())}
        choices = [c for c in palette if c not in forbidden]
        if not choices:
            return None
        pieces[i] = replace(p, color=int(rng.choice(choices)))
    rods = [p for p in pieces.values() if p.kind == 'rod']
    for i in relay_ids:
        p = pieces[i]
        under = [q.color for q in rods if {(r + 1, c) for r, c in p.cells()} & q.cells()]
        options = under if under and rng.random() < .65 else [q.color for q in rods]
        key = int(rng.choice(options if options else palette))
        pieces[i] = replace(p, color=key)
    return (height, width), pieces


def _port_options(p):
    return [replace(p, side=side, offset=offset) for side in range(4)
            for offset in range(1, p.w - 1 if side % 2 == 0 else p.h - 1)]


def _route(rng, pieces, shape, depth):
    """Choose visible ports using the evolving scaffold, with bounded backtracking."""
    h, w = shape
    launches = []
    for initial in range(4):
        lanes = range(2, w - 2) if initial % 2 == 0 else range(2, h - 2)
        for lane in lanes:
            start = (h - 1, lane) if initial == 0 else (lane, 0) if initial == 1 else (0, lane) if initial == 2 else (lane, w - 1)
            dr, dc = DIRS[initial]
            tail = [(start[0] + k * dr, start[1] + k * dc) for k in range(2)]
            cap = start[0] + 2 * dr, start[1] + 2 * dc
            launcher = set(tail) | {cap}
            if any(launcher & p.cells() or _adjacent(launcher, p.cells()) for p in pieces.values()):
                continue
            target = _hit(pieces, cap, initial, shape)
            if target is not None:
                launches.append((tail, cap, initial, target))
    rng.shuffle(launches)
    budget = [180]

    def grow(state, original, target, seen, anchors):
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        key = state[target].color
        state = _settle({i: p for i, p in state.items()
                         if p.kind != 'rod' or p.color != key}, h, anchors)
        seen = seen | {target}
        choices = _port_options(state[target])
        rng.shuffle(choices)
        for p in choices:
            nxt = _hit(state, p.port(), p.side, shape)
            final = len(seen) == depth
            if final and nxt is not None and nxt not in seen:
                continue
            if not final and (nxt is None or nxt in seen):
                continue
            copy_original = dict(original)
            copy_original[target] = replace(original[target], side=p.side, offset=p.offset)
            if final:
                return copy_original
            copy_state = dict(state)
            copy_state[target] = p
            result = grow(copy_state, copy_original, nxt, seen, anchors)
            if result is not None:
                return result
        return None

    for tail, cap, initial, target in launches[:40]:
        result = grow(pieces, pieces, target, set(), set(tail) | {cap})
        if result is not None:
            for i, p in list(result.items()):
                if p.kind == 'relay' and i not in _run(result, shape, cap, initial)[1]:
                    result[i] = _port_options(p)[int(rng.integers(len(_port_options(p))))]
            return result, tail, cap, initial
    return None


def _facts(pieces, shape, cap, initial):
    state, activated, events = _run(pieces, shape, cap, initial)
    frozen = _run(pieces, shape, cap, initial, gravity=False)
    stale = _run(pieces, shape, cap, initial, stale_port=True)
    keys = {p.color for p in pieces.values() if p.kind == 'relay'}
    anchors = _anchors(shape, cap, initial)
    all_commands = _settle({i: p for i, p in pieces.items()
                            if p.kind != 'rod' or p.color not in keys}, shape[0], anchors)
    return {'depth': len(activated), 'decoys': sum(p.kind == 'relay' for p in pieces.values()) - len(activated),
            'feedback': activated != frozen[1], 'stale_diff': activated != stale[1],
            'static_mechanical_diff': state != _settle(frozen[0], shape[0], anchors),
            'all_keys_diff': state != all_commands,
            'source_moves': any(e['source_fall'] for e in events),
            'multi_delete': any(len(e['deleted']) >= 2 for e in events),
            'unequal': any(len({d for d in e['falls'].values() if d}) >= 2 for e in events),
            'stationary_emitter': any(not e['source_fall'] and any(e['falls'].values()) for e in events),
            'falls': [d for e in events for d in e['falls'].values() if d],
            'events': events, 'activated': activated, 'state': state}


def _make_pair(rng, colors, depth, requirement=None, large=False):
    for attempt in range(600):
        n = depth + int(rng.integers(1, 3))
        scene = _scaffold(rng, n, large, colors)
        if scene is None:
            continue
        shape, pieces = scene
        route = _route(rng, pieces, shape, depth)
        if route is None:
            continue
        pieces, tail, cap, initial = route
        facts = _facts(pieces, shape, cap, initial)
        if requirement == 'feedback' and not (facts['feedback'] and facts['stale_diff'] and facts['source_moves'] and facts['multi_delete'] and facts['static_mechanical_diff']):
            continue
        if requirement == 'support' and not ((facts['unequal'] or facts['stationary_emitter']) and facts['all_keys_diff']):
            continue
        if large and not facts['source_moves']:
            continue
        bg, signal, frame = map(int, colors[:3])
        inp = _render(shape, bg, signal, frame, tail, cap, pieces)
        try:
            parsed = _parse(inp)[-1]
            assert sorted(parsed.values(), key=repr) == sorted(pieces.values(), key=repr)
        except AssertionError:
            continue
        out = _render(shape, bg, signal, frame, tail, cap, facts['state'], facts['activated'])
        out[cap] = signal
        return {'input': inp, 'output': out}
    raise AssertionError(f'construction budget exhausted: {depth}/{requirement}')


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _legal_count(value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError('illegal example count')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _legal_count(num_train, 2, 10), _legal_count(num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    base = list(map(int, rng.permutation(10)))
    varying = rng.random() < .65
    puzzle, seen = {'train': [], 'test': []}, set()
    depths = [2, 2] + [int(rng.choice([1, 2, 3], p=[.2, .55, .25])) for _ in range(num_train - 2)]
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            depth = depths[index] if split == 'train' else max(depths) + 1
            requirement = ('feedback' if index == 0 else 'support' if index == 1 else None) if split == 'train' else None
            colors = base if not varying or (split == 'train' and index == 0) else list(map(int, np.roll(base, 1))) if split == 'train' and index == 1 else list(map(int, rng.permutation(10)))
            for _ in range(20):
                pair = _make_pair(rng, colors, depth, requirement, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise AssertionError('distinct-input budget exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate visible geometry, dynamics and demonstration coverage."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            shape, bg, signal, frame, tail, cap, initial, pieces = _parse(inp)
            f = _facts(pieces, shape, cap, initial)
            expected = _render(shape, bg, signal, frame, tail, cap, f['state'], f['activated'])
            expected[cap] = signal
            assert np.array_equal(out, expected), 'incorrect final circuit'
            anchors = set(tail) | {cap}
            assert _settle(f['state'], shape[0], anchors) == f['state']
            occupied = set(anchors)
            for i, p in f['state'].items():
                original = pieces[i]
                assert p.r >= original.r
                assert replace(p, r=original.r) == original, 'piece changed shape or moved sideways'
                assert not occupied & p.cells(), 'pieces overlap'
                occupied |= p.cells()
                if p.kind == 'relay':
                    assert out[p.r + p.key_r, p.c + p.key_c] == p.color
                    assert out[p.port()] == signal
            fired_keys = {pieces[i].color for i in f['activated']}
            assert {i for i in pieces if i not in f['state']} == {
                i for i, p in pieces.items() if p.kind == 'rod' and p.color in fired_keys}
            assert all(out[cell] == signal for cell in set(tail) | {cap})
            assert f['depth'] >= 1 and f['decoys'] >= 1
            records[split].append(f)
    train = records['train']
    assert any(f['feedback'] and f['stale_diff'] and f['source_moves']
               and f['multi_delete'] and f['static_mechanical_diff'] for f in train), 'missing timing demonstration'
    assert any((f['unequal'] or f['stationary_emitter']) and f['all_keys_diff']
               for f in train), 'missing support/reachability demonstration'
    assert all(f['depth'] > max(t['depth'] for t in train) for f in records['test']), 'missing route-depth shift'


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def _mechanics_test():
    # Two removable legs and a shorter, persistent landing beneath a relay.
    pieces = {0: Piece('relay', 4, 2, 3, 7, 1, 1, 1),
              1: Piece('rod', 7, 3, 5, 1, 1),
              2: Piece('rod', 7, 7, 5, 1, 2),
              3: Piece('rod', 10, 5, 2, 1, 3)}
    assert _settle(pieces, 12) == pieces
    one_leg = {i: p for i, p in pieces.items() if i != 1}
    assert _settle(one_leg, 12) == one_leg
    both_legs = _settle({i: p for i, p in pieces.items() if i not in (1, 2)}, 12)
    assert both_legs[0].r == 7 and both_legs[3].r == 10
    # A five-row source fall and two-row remote fall change the next target.
    pieces = {0: Piece('relay', 3, 4, 4, 3, 1, 1, 1),
              1: Piece('relay', 7, 10, 3, 3, 2, 1, 1),
              2: Piece('relay', 2, 10, 3, 3, 4, 1, 1),
              3: Piece('rod', 5, 11, 2, 1, 4),
              4: Piece('rod', 7, 5, 5, 1, 1),
              5: Piece('rod', 10, 11, 2, 1, 1)}
    shape, tail, cap, initial = (12, 16), [(4, 0), (4, 1)], (4, 2), 1
    inp = _render(shape, 0, 8, 9, tail, cap, pieces)
    state, active, events = _run(pieces, shape, cap, initial)
    assert active == [0, 1]
    assert [state[i].r for i in range(4)] == [8, 9, 4, 7]
    assert set(events[0]['deleted']) == {4, 5}
    assert _run(pieces, shape, cap, initial, gravity=False)[1] == [0, 2]
    assert _run(pieces, shape, cap, initial, stale_port=True)[1] == [0, 2]
    expected = _render(shape, 0, 8, 9, tail, cap, state, active)
    assert np.array_equal(solve(inp), expected)
    # A bottom launcher is fixed support, not paint over a falling device.
    anchored = {0: Piece('relay', 3, 4, 3, 3, 1, 2, 1),
                1: Piece('rod', 6, 4, 6, 1, 1)}
    anchor_tail, anchor_cap = [(11, 6), (10, 6)], (9, 6)
    anchor_in = _render((12, 12), 0, 8, 9, anchor_tail, anchor_cap, anchored)
    anchor_state, anchor_active, _ = _run(anchored, (12, 12), anchor_cap, 0)
    assert anchor_state[0].r == 6 and anchor_active == [0]
    anchor_out = _render((12, 12), 0, 8, 9, anchor_tail, anchor_cap, anchor_state, anchor_active)
    assert np.array_equal(solve(anchor_in), anchor_out)
    # All-reached, absent-key circuit with a return to a spent relay.
    cycle = {0: Piece('relay', 5, 3, 3, 3, 1, 1, 1),
             1: Piece('relay', 5, 10, 3, 3, 2, 3, 1)}
    cycle_tail, cycle_cap = [(0, 4), (1, 4)], (2, 4)
    cycle_in = _render((8, 16), 0, 8, 9, cycle_tail, cycle_cap, cycle)
    cycle_state, cycle_active, cycle_events = _run(cycle, (8, 16), cycle_cap, 2)
    assert cycle_active == [0, 1] and cycle_state == cycle
    assert all(not e['deleted'] for e in cycle_events)
    assert np.array_equal(solve(cycle_in), _render((8, 16), 0, 8, 9,
                          cycle_tail, cycle_cap, cycle_state, cycle_active))
    # Misses still consume the cap and preserve the unactivated scene.
    miss_tail, miss_cap = [(2, 0), (2, 1)], (2, 2)
    miss_in = _render((8, 16), 0, 8, 9, miss_tail, miss_cap, cycle)
    miss_out = miss_in.copy()
    miss_out[miss_cap] = 8
    assert np.array_equal(solve(miss_in), miss_out)
    return inp, expected


def self_test():
    """Bounded 42-seed study plus explicit geometry and contract diagnostics."""
    _mechanics_test()
    fingerprints, shapes, rods, relays, lengths, directions, fall_distances = set(), set(), set(), set(), set(), set(), set()
    backgrounds, signals, frames, structures = set(), set(), set(), set()
    measures = {name: 0 for name in ('feedback', 'stale_diff', 'unequal', 'stationary_emitter', 'multi_delete', 'all_keys_diff')}
    depths = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                shape, bg, signal, frame, tail, cap, initial, pieces = _parse(inp)
                f = _facts(pieces, shape, cap, initial)
                shapes.add(shape)
                backgrounds.add(bg); signals.add(signal); frames.add(frame)
                rods.add(sum(p.kind == 'rod' for p in pieces.values()))
                relays.add(sum(p.kind == 'relay' for p in pieces.values()))
                lengths.update(max(p.h, p.w) for p in pieces.values() if p.kind == 'rod')
                directions.update(p.side for p in pieces.values() if p.kind == 'relay')
                fall_distances.update(f['falls'])
                depths.add(f['depth'])
                for name in measures:
                    measures[name] += bool(f[name])
                # Geometry diversity with literal color identities removed.
                roles = {}
                structures.add((shape, tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat)))
                assert np.array_equal(solve(inp.copy()), pair['output'])
                before = dict(pieces)
                for event in f['events']:
                    after_deletion = {i: p for i, p in before.items() if i not in event['deleted']}
                    anchors = set(tail) | {cap}
                    synchronous = _settle(after_deletion, shape[0], anchors)
                    assert synchronous == _settle_direct(after_deletion, shape[0], anchors)
                    assert all(synchronous[i].r - before[i].r == d for i, d in event['falls'].items())
                    before = synchronous
    assert len(fingerprints) == 42 and len(structures) >= 190
    assert len(shapes) >= 80 and len(rods) >= 9 and len(relays) >= 4
    assert len(lengths) >= 10 and max(lengths) >= 12
    assert directions == set(range(4)) and {1, 2, 3, 4} <= depths
    assert len(fall_distances) >= 10 and max(fall_distances) >= 12
    assert all(len(s) == 10 for s in (backgrounds, signals, frames))
    assert all(count >= 20 for count in measures.values())
    assert _fingerprint(generate_puzzle(42)) == _fingerprint(generate_puzzle(42))
    for counts in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(314159, *counts)
        assert (len(puzzle['train']), len(puzzle['test'])) == counts
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Palette roles and left/right placement are read from the input.
    inp, out = _mechanics_test()
    permutation = np.array([3, 6, 0, 8, 2, 9, 1, 4, 5, 7], dtype=np.int8)
    assert np.array_equal(solve(permutation[inp]), permutation[out])
    assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
    puzzle = generate_puzzle(0)
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzle[s]] for s in puzzle}
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'pairs': 210, 'shapes': len(shapes), 'structures': len(structures),
            'rod_counts': sorted(rods), 'relay_counts': sorted(relays),
            'rod_lengths': sorted(lengths), 'fall_distances': sorted(fall_distances),
            'depths': sorted(depths), 'mechanism_counts': measures}
