"""Pulse-locked staircases: clocks can change the terrain route they paint.

Only visible input geometry determines roles, directions, periods and gates.
No parent implementation is imported. All randomness is local to generation.
"""
from collections import Counter
import math

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (not _integer(value) or not low <= int(value) <= high):
            raise ValueError("illegal example count")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 1) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _orient(grid, code):
    return np.rot90(grid[:, ::-1] if code >= 4 else grid, code % 4).copy()


def _unorient(grid, code):
    a = np.rot90(grid, -(code % 4))
    return (a[:, ::-1] if code >= 4 else a).copy()


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _grid(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _decode(grid):
    """The exclusive-color capped separator supplies a local coordinate frame."""
    original = _grid(grid)
    candidates = []
    for code in range(8):
        a = _unorient(original, code)
        h, w = a.shape
        for separator in map(int, np.unique(a)):
            rr, cc = np.nonzero(a == separator)
            top, bottom = int(rr.min()), int(rr.max())
            col = int(cc.max())
            if bottom - top < 3 or col < 3 or col >= w - 1:
                continue
            glyph = {(r, col) for r in range(top, bottom + 1)}
            glyph |= {(top, col - 1), (top, col - 2)}
            if set(zip(map(int, rr), map(int, cc))) != glyph:
                continue
            background = int(a[top + 1, col - 1])
            terrain = int(a[bottom, col + 1])
            if len({background, terrain, separator}) != 3:
                continue
            # Bars occur only on the source row; the cap is the only exception.
            left = a[:, :col].copy()
            left[bottom, :] = background
            left[top, col - 2:col] = background
            if np.any(left != background):
                continue
            strip = a[:, col]
            if np.any(strip[:top] != background) or np.any(strip[bottom + 1:] != background):
                continue
            runs, cursor = [], col - 1
            while cursor >= 0:
                color = int(a[bottom, cursor])
                if color == background:
                    cursor -= 1
                    continue
                near = col - cursor
                length = 0
                while cursor >= 0 and a[bottom, cursor] == color:
                    length += 1
                    cursor -= 1
                runs.append((color, length, near))
            if not 1 <= len(runs) <= 4 or any(not 1 <= b[1] <= 7 for b in runs):
                continue
            payload = {b[0] for b in runs}
            if payload & {background, terrain, separator}:
                continue
            if set(map(int, np.unique(a))) != payload | {background, terrain, separator}:
                continue
            gates = {tuple(map(int, p)) for p in np.argwhere(np.isin(a, list(payload))) if p[1] > col}
            if any((r + dr, c + dc) in gates for r, c in gates
                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                continue
            candidates.append(dict(canonical=a, orientation=code, background=background,
                                   terrain=terrain, separator=separator, col=col, top=top,
                                   start=(bottom, col + 1), runs=tuple(runs), gates=gates))
    if len(candidates) != 1:
        raise ValueError("input needs one unambiguous capped launch separator")
    return candidates[0]


def _pulse(runs, tick, alternative=None):
    firing = []
    for i, (color, period, near) in enumerate(runs):
        p = period + 1 if alternative == 'period_plus_one' else period
        offset = near - 1 if alternative == 'source_gap_phase' else 0
        if (tick - offset) % p == 0:
            firing.append(i)
    if not firing:
        return None
    if alternative == 'shortest':
        winner = min(firing, key=lambda i: (runs[i][1], i))
    elif alternative == 'longest':
        winner = min(firing, key=lambda i: (-runs[i][1], i))
    elif alternative == 'farthest':
        winner = firing[-1]
    else:
        winner = firing[0]
    return runs[winner][0]


def _walk(info, alternative=None):
    """Cell walker: unsuccessful probes do not advance either clock."""
    a = info['canonical']
    out = a.copy()
    terrain, background = info['terrain'], info['background']
    direction, tick, current = 0, 0, info['start']
    vectors = ((0, 1), (-1, 0))
    path, ticks, collisions, opened, closed, legs = [], [], [], [], [], [0]

    def permits(p, time):
        if not _inside(a.shape, p):
            return False
        if a[p] == terrain:
            return True
        if p not in info['gates']:
            return False
        if alternative == 'all_open':
            return True
        if alternative == 'all_closed':
            return False
        if alternative == 'any_firing_gate':
            return any(color == int(a[p]) and time % period == 0
                       for color, period, near in info['runs'])
        return _pulse(info['runs'], time, alternative) == int(a[p])

    while True:
        assert current not in path and len(path) <= sum(a.shape)
        path.append(current)
        ticks.append(tick)
        legs[-1] += 1
        color = _pulse(info['runs'], tick, alternative)
        out[current] = terrain if color is None else color
        if current in info['gates']:
            opened.append((current, tick))
        dr, dc = vectors[direction]
        ahead = (current[0] + dr, current[1] + dc)
        if not _inside(a.shape, ahead):
            if alternative == 'turn_after_exit':
                dr, dc = vectors[1 - direction]
                side = (current[0] + dr, current[1] + dc)
                if permits(side, tick + 1):
                    direction = 1 - direction
                    current, tick = side, tick + 1
                    legs.append(0)
                    continue
            end = 'edge'
            break
        if permits(ahead, tick + 1):
            current, tick = ahead, tick + 1
            continue
        out[ahead] = info['separator']
        collisions.append((ahead, tick + 1))
        if ahead in info['gates']:
            closed.append((ahead, tick + 1))
        if alternative == 'straight':
            end = 'blocked'
            break
        next_tick = 0 if alternative == 'reset_at_bend' else tick + 1
        if alternative == 'count_collision':
            next_tick += 1
        direction = 1 - direction
        dr, dc = vectors[direction]
        side = (current[0] + dr, current[1] + dc)
        if not permits(side, next_tick):
            end = 'blocked'
            break
        current, tick = side, next_tick
        legs.append(0)
    return dict(output=_orient(out, info['orientation']), path=tuple(path), ticks=tuple(ticks),
                collisions=tuple(collisions), opened=tuple(opened), closed=tuple(closed),
                legs=tuple(legs), end=end)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _walk(_decode(input_grid))['output']


_ALTERNATIVES = ('shortest', 'longest', 'farthest', 'period_plus_one',
                 'source_gap_phase', 'all_open', 'all_closed', 'reset_at_bend',
                 'count_collision', 'straight', 'any_firing_gate', 'turn_after_exit')


def _evidence(info, record):
    return {name for name in _ALTERNATIVES
            if not np.array_equal(record['output'], _walk(info, name)['output'])}


def _backbone(rng, h, width, start_row, minimum, simple=False):
    """Sample unequal monotone legs, then add terrain independently around them."""
    r, c, direction = start_row, 0, 0
    path, walls, legs = [(r, c)], [], [1]
    limit = 1 if simple else int(rng.integers(3, 9))
    for leg in range(limit):
        room = width - 1 - c if direction == 0 else r
        if room == 0:
            return None
        final = leg == limit - 1
        reach_edge = final and rng.random() < .55
        move = room if reach_edge else int(rng.integers(1, min(room, 7) + 1))
        if not reach_edge and move == room:
            if room == 1:
                reach_edge = True
            else:
                move -= 1
        dr, dc = ((0, 1), (-1, 0))[direction]
        for _ in range(move):
            r, c = r + dr, c + dc
            path.append((r, c))
            legs[-1] += 1
        ahead = (r + dr, c + dc)
        if not _inside((h, width), ahead):
            break
        walls.append(ahead)
        if final:
            dr, dc = ((0, 1), (-1, 0))[1 - direction]
            side = (r + dr, c + dc)
            if _inside((h, width), side):
                walls.append(side)
            break
        direction = 1 - direction
        dr, dc = ((0, 1), (-1, 0))[direction]
        r, c = r + dr, c + dc
        if not _inside((h, width), (r, c)):
            return None
        path.append((r, c))
        legs.append(1)
    if len(path) < minimum or set(path) & set(walls):
        return None
    # Broad independent row profiles and rectangles leave substantial unused terrain.
    mask = np.zeros((h, width), bool)
    knots = np.unique(np.r_[0, rng.choice(np.arange(1, h - 1), min(6, h - 2), replace=False), h - 1])
    left = np.rint(np.interp(np.arange(h), knots, rng.uniform(-width * .4, width * .6, len(knots)))).astype(int)
    right = np.rint(np.interp(np.arange(h), knots, rng.uniform(width * .45, width * 1.3, len(knots)))).astype(int)
    for row in range(h):
        lo = max(0, min(width - 1, left[row]))
        hi = max(lo + 1, min(width, right[row]))
        mask[row, lo:hi] = True
    for _ in range(int(rng.integers(1, 7))):
        rr, cc = int(rng.integers(h)), int(rng.integers(width))
        dh, dw = int(rng.integers(1, 6)), int(rng.integers(1, 7))
        mask[rr:rr + dh, cc:cc + dw] = bool(rng.integers(2))
    for p in path:
        mask[p] = True
    for p in walls:
        mask[p] = False
    # Some edge endings deliberately have available terrain around the corner.
    if not _inside((h, width), ahead):
        dr, dc = ((0, 1), (-1, 0))[1 - direction]
        corner = (path[-1][0] + dr, path[-1][1] + dc)
        if _inside(mask.shape, corner):
            mask[corner] = True
    return mask, path, legs


def _witness_ok(info, record):
    runs = info['runs']
    minimum = math.lcm(*(b[1] for b in runs)) + 2
    if len(record['path']) < minimum or len(record['legs']) < 2:
        return False
    visible = Counter(_pulse(runs, i) for i, p in enumerate(record['path']) if p not in info['gates'])
    if any(visible[b[0]] < 2 for b in runs):
        return False
    if not record['opened'] or not record['closed']:
        return False
    if not any(i > 0 and all(i % b[1] == 0 for b in runs)
               and p not in info['gates'] for i, p in enumerate(record['path'])):
        return False
    required = {'all_open', 'all_closed', 'reset_at_bend', 'count_collision', 'straight',
                'period_plus_one', 'source_gap_phase', 'farthest', 'any_firing_gate',
                'turn_after_exit'}
    return required <= _evidence(info, record)


def _make_pair(rng, lengths, colors, roles, orientation, witness=False, test=False):
    background, terrain, separator = roles
    for attempt in range(400):
        # Packing, not source dimensions, controls the geometry.
        gaps = [int(rng.integers(1, 3)) if witness else int(rng.integers(0, 3))]
        gaps += [int(rng.integers(int(colors[i] == colors[i - 1]), 3)) for i in range(1, len(colors))]
        payload = sum(lengths) + sum(gaps)
        if payload > 18:
            continue
        col = payload + int(rng.integers(0, min(3, 19 - payload)))
        width = int(rng.integers(max(8, 24 - col), 30 - col)) if witness else int(rng.integers(6, 30 - col))
        h = int(rng.integers(22 if witness else 11, 31))
        start_row = int(rng.integers(max(7, h - 6), h))
        minimum = math.lcm(*lengths) + 8 if witness else (12 if test else 5)
        built = _backbone(rng, h, width, start_row, minimum,
                          simple=not witness and not test and rng.random() < .25)
        if built is None:
            continue
        mask, path, legs = built
        a = np.full((h, col + 1 + width), background, dtype=np.int8)
        a[:, col + 1:] = np.where(mask, terrain, background)
        top = int(rng.integers(0, max(1, start_row - 3)))
        a[top:start_row + 1, col] = separator
        a[top, col - 2:col] = separator
        cursor = col
        for length, color, gap in zip(lengths, colors, gaps):
            cursor -= gap
            a[start_row, cursor - length:cursor] = color
            cursor -= length
        runs = tuple((color, length, 1) for color, length in zip(colors, lengths))
        # Open gates are planted at actual pulse times. Closed gates use another
        # color (or a silent time), and can redirect into the surrounding terrain.
        open_options = [i for i in range(2, len(path) - 3) if _pulse(runs, i) is not None]
        if not open_options:
            continue
        first_open = int(rng.choice(open_options))
        chosen = {first_open}
        gate_values = {first_open: _pulse(runs, first_open)}
        if witness:
            common = math.lcm(*lengths)
            closed_options = [i for i in range(common, len(path) - 1)
                              if i % common == 0 and abs(i - first_open) > 1]
            if not closed_options:
                continue
            closed_index = int(rng.choice(closed_options))
            if first_open > closed_index:
                continue
            chosen.add(closed_index)
            gate_values[closed_index] = colors[-1]
        gate_count = int(rng.integers(3, min(7, (len(path) + 1) // 2))) if test else (2 if witness else int(rng.integers(0, 3)))
        if gate_count == 0:
            chosen, gate_values = set(), {}
        options = list(map(int, rng.permutation(np.arange(2, len(path) - 1))))
        for i in options:
            if len(chosen) >= gate_count:
                break
            if all(abs(i - j) > 1 for j in chosen):
                chosen.add(i)
                fired = _pulse(runs, i)
                gate_values[i] = fired if fired is not None and rng.random() < .6 else int(rng.choice(colors))
        if len(chosen) != gate_count:
            continue
        for i in chosen:
            r, c = path[i]
            a[r, c + col + 1] = gate_values[i]
        inp = _orient(a, orientation)
        try:
            info = _decode(inp)
        except ValueError:
            continue
        record = _walk(info)
        if witness and not _witness_ok(info, record):
            continue
        if test and (not record['opened'] or not record['closed']
                     or 'all_open' not in _evidence(info, record)):
            continue
        if len(record['path']) < 3:
            continue
        return dict(input=inp, output=record['output'])
    raise RuntimeError("bounded staircase construction exhausted")


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _puzzle_key(p):
    return tuple(tuple((_key(q['input']), _key(q['output'])) for q in p[s]) for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    nt, ne = _counts(rng, num_train, num_test)
    palette = tuple(map(int, rng.permutation(10)))
    teaching_colors = palette[:2]
    choices = ((2, 3), (2, 5), (3, 4), (3, 5), (2, 7), (4, 6))
    periods = choices[int(rng.integers(len(choices)))]
    base = int(rng.integers(8))
    orientation_mode = int(rng.integers(3))  # stable, mixed, test quarter-turn
    vary_roles = bool(rng.integers(2))
    slots = list(map(int, rng.choice(nt, 2, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            witness = split == 'train' and index in slots
            for retry in range(16):
                if witness:
                    reverse = index == slots[1]
                    lengths = periods[::-1] if reverse else periods
                    colors = teaching_colors[::-1] if reverse else teaching_colors
                else:
                    n = int(rng.choice([1, 2, 3, 4], p=[.15, .25, .40, .20])) if split == 'test' else int(rng.integers(1, 5))
                    lengths = tuple(map(int, rng.integers(1, max(periods) + 1, n)))
                    if sum(lengths) > 14:
                        continue
                    colors = tuple(map(int, rng.choice(palette[:7], n, replace=True)))
                excluded = set(colors) | set(teaching_colors)
                available = [c for c in palette if c not in excluded]
                roles = tuple(map(int, rng.choice(available, 3, replace=False))) if vary_roles else palette[7:10]
                orientation = int(rng.integers(8)) if orientation_mode == 1 else base
                if split == 'test' and orientation_mode == 2:
                    orientation = (base // 4) * 4 + (base + 1) % 4
                try:
                    pair = _make_pair(rng, lengths, colors, roles, orientation,
                                      witness=witness, test=split == 'test')
                except RuntimeError:
                    continue
                if _key(pair['input']) in seen:
                    continue
                puzzle[split].append(pair)
                seen.add(_key(pair['input']))
                break
            else:
                raise RuntimeError("unable to construct distinct informative examples")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    info = _decode(inp)
    record = _walk(info)
    expected = _unorient(out, info['orientation'])
    a = info['canonical']
    assert np.array_equal(out, record['output']), "wrong route, pulse, gate or collision"
    assert record['ticks'] == tuple(range(len(record['path']))), "collision must not consume time"
    assert len(record['path']) == len(set(record['path']))
    footprint = set(record['path']) | {p for p, _ in record['collisions']}
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            p = (r, c)
            if p not in footprint:
                assert expected[p] == a[p], "unvisited cells and source must be preserved"
    # Direct arithmetic on visited indices, rather than reusing pulse selection.
    for tick, p in enumerate(record['path']):
        eligible = [b for b in info['runs'] if tick % b[1] == 0]
        assert expected[p] == (eligible[0][0] if eligible else info['terrain'])
        if p in info['gates']:
            assert eligible and eligible[0][0] == int(a[p])
    for p, next_tick in record['collisions']:
        eligible = [b for b in info['runs'] if next_tick % b[1] == 0]
        assert expected[p] == info['separator']
        assert a[p] == info['background'] or (
            p in info['gates'] and (not eligible or eligible[0][0] != int(a[p])))
    return info, record


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence, priority_edges, period_orders = set(), set(), set(), set()
    train_gates, test_gates = [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info, record = _check_pair(pair)
            key = _key(_grid(pair['input']))
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert not np.array_equal(pair['input'], pair['output'])
            if split == 'train':
                train_gates.append(len(info['gates']))
                evidence |= _evidence(info, record)
                if _witness_ok(info, record) and len(info['runs']) == 2:
                    a, b = info['runs']
                    priority_edges.add((a[0], b[0]))
                    period_orders.add(a[1] < b[1])
            else:
                test_gates.append(len(info['gates']))
    assert set(_ALTERNATIVES) <= evidence, "training leaves a simpler rival unresolved"
    assert period_orders == {True, False}, "show shorter and longer nearest"
    assert any((b, a) in priority_edges for a, b in priority_edges), "fixed color priority remains possible"
    assert min(test_gates) > max(train_gates), "test must recombine more visible gates"


def _regressions():
    a = np.zeros((8, 14), dtype=np.int8)
    a[:7, 6] = 5
    a[0, 4:6] = 5
    a[6, :3] = 3
    a[6, 4:6] = 2
    a[6, 7:12] = 8
    a[3:7, 11] = 8
    a[3, 11:14] = 8
    a[5, 12:14] = 8
    a[4, 13] = a[2, 13] = 8  # unused corners after edge exits
    a[6, 9] = 2  # t=2: opens
    a[4, 11] = 3  # t=6: both clocks fire; nearer 2 wins, so 3 closes
    expected = a.copy()
    for p, color in { (6, 7):2, (6, 8):8, (6, 9):2, (6, 10):3,
                      (6, 11):2, (5, 11):8, (5, 12):2, (5, 13):8,
                      (6, 12):5, (4, 11):5 }.items():
        expected[p] = color
    opened = a.copy()
    opened[4, 11] = 2
    expected_open = opened.copy()
    for p, color in { (6, 7):2, (6, 8):8, (6, 9):2, (6, 10):3,
                      (6, 11):2, (5, 11):8, (4, 11):2, (3, 11):8,
                      (3, 12):2, (3, 13):3, (6, 12):5, (2, 11):5 }.items():
        expected_open[p] = color
    for before, after in ((a, expected), (opened, expected_open)):
        for code in range(8):
            assert np.array_equal(solve(_orient(before, code)), _orient(after, code))
        permutation = np.array([4, 1, 7, 0, 2, 6, 3, 8, 9, 5], dtype=np.int8)
        assert np.array_equal(solve(permutation[before]), permutation[after])


def self_test():
    _regressions()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    features = []
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info, record = _check_pair(pair)
                features.append((pair['input'].shape, info['orientation'], len(info['runs']),
                                 len(info['gates']), record['legs'], tuple(b[1] for b in info['runs']),
                                 info['background'], info['terrain'], record['end']))
    assert len({f[0] for f in features}) >= 50
    assert {f[1] for f in features} == set(range(8))
    assert {f[2] for f in features} == {1, 2, 3, 4}
    assert len({f[4] for f in features}) >= 80
    assert len({f[5] for f in features}) >= 40
    assert len({f[6] for f in features}) == 10 and len({f[7] for f in features}) == 10
    assert {f[8] for f in features} == {'edge', 'blocked'}
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(101 + nt * 4 + ne, nt, ne)
    for counts in ((2, 1), (10, 4), (None, 4), (2, None)):
        a = generate_puzzle(4321, *counts)
        assert _puzzle_key(a) == _puzzle_key(generate_puzzle(4321, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (3, True)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("output mutation accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, gate/clock regressions, counts and mutations')
