"""Marching pulse looms: distributed clocks, keyed correspondence, cyclic motion.

No files, examples, seed, or generation metadata are consulted by solve().
Construction paints pulse events; validation reconstructs inputs and checks cells.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np


_STEPS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _array(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= v <= 30 for v in a.shape), "invalid dimensions"
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid colors"
    return a


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _add(base, u, j, n, d):
    return base[0] + j*u[0] + d*n[0], base[1] + j*u[1] + d*n[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(a, bg):
    remaining = set(map(tuple, np.argwhere(a != bg).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _STEPS:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _frame(template, transform):
    labels = _orient(np.arange(template.size).reshape(template.shape), transform)
    a = tuple(map(int, np.argwhere(labels == 0)[0]))
    b = tuple(map(int, np.argwhere(labels == 1)[0]))
    f = tuple(map(int, np.argwhere(labels == template.shape[1])[0]))
    return a, (b[0]-a[0], b[1]-a[1]), (f[0]-a[0], f[1]-a[1])


def _device(base, u, n, length, shape, source_depth):
    front = _add(base, u, 0, n, 1)
    if n == (1, 0):
        depth = shape[0]-1-front[0]
    elif n == (-1, 0):
        depth = front[0]
    elif n == (0, 1):
        depth = shape[1]-1-front[1]
    else:
        depth = front[1]
    corners = [_add(base, u, j, n, d)
               for j in (0, length-1) for d in (-source_depth, depth+1)]
    if depth < 1 or not all(_inside(p, shape) for p in corners):
        return None
    rs, cs = zip(*corners)
    cells = {_add(base, u, j, n, d) for j in range(length) for d in (0, 1)}
    return {"base": base, "u": u, "n": n, "depth": depth, "cells": cells,
            "rect": (min(rs), min(cs), max(rs), max(cs))}


def _separate(a, b):
    # A blank row/column between the two rectangular working strips.
    return a[2]+1 < b[0] or b[2]+1 < a[0] or a[3]+1 < b[1] or b[3]+1 < a[1]


def _valid_program(code, length):
    assert 1 <= len(code) <= 4
    assert len({e[1] for e in code}) == len(code), "source depths must be distinct"
    occupied = set()
    for slot, start, period, color in code:
        assert 0 <= slot < length and start >= 2 and 1 <= period <= 6
        assert 0 <= color <= 9
        cells = {(start+k, slot) for k in range(period)}
        assert not occupied.intersection(cells), "overlapping virtual source runs"
        assert not any((r+dr, c+dc) in occupied for r, c in cells for dr, dc in _STEPS), "touching source runs"
        occupied.update(cells)


def _decode(grid):
    a = _array(grid)
    colors, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, "background must be unique"
    bg = int(colors[counts.argmax()])
    raw, fragments = [], []
    for cells in _components(a, bg):
        rs, cs = zip(*cells)
        r, c, bottom, right = min(rs), min(cs), max(rs), max(cs)
        patch = a[r:bottom+1, c:right+1]
        if patch.size == len(cells) and min(patch.shape) == 2 and 4 <= max(patch.shape) <= 9:
            raw.append((r, c, patch.copy()))
        else:
            assert len(set(int(a[p]) for p in cells)) == 1 and min(patch.shape) == 1
            assert patch.size == len(cells) and len(cells) <= 6, "source must be a solid straight run"
            fragments.append((cells, int(a[cells[0]])))
    assert 2 <= len(raw) <= 4 and fragments, "need devices and source fragments"
    first = raw[0][2]
    singletons = [int(c) for c, count in zip(*np.unique(first, return_counts=True)) if count == 1]
    assert len(singletons) == 1, "one unique key color in each device"
    key_color = singletons[0]
    forms = [_orient(first, t) for t in range(8)]
    canonical = [x for x in forms if x.shape[0] == 2 and x[1, 0] == key_color]
    assert len(canonical) == 1
    template, devices = canonical[0], []
    length = template.shape[1]
    structural = set(map(int, template.flat))
    assert bg not in structural and 2 <= len(structural) <= 3
    assert len({_key(_orient(template, t)) for t in range(8)}) == 8
    for r, c, patch in raw:
        matches = [t for t in range(8) if np.array_equal(patch, _orient(template, t))]
        assert len(matches) == 1, "copies must have exact colored correspondence"
        corner, u, n = _frame(template, matches[0])
        base = r+corner[0], c+corner[1]
        devices.append({"base": base, "u": u, "n": n})
    occupied = {_add(e['base'], e['u'], j, e['n'], d)
                for e in devices for j in range(length) for d in (0, 1)}
    program, owners = {}, {}
    for cells, color in fragments:
        assert color not in structural and color != bg
        matches = []
        for index, e in enumerate(devices):
            base, u, n = e['base'], e['u'], e['n']
            slots = [(r-base[0])*u[0] + (c-base[1])*u[1] for r, c in cells]
            ds = [-((r-base[0])*n[0] + (c-base[1])*n[1]) for r, c in cells]
            if len(set(slots)) != 1 or not 0 <= slots[0] < length or min(ds) < 2:
                continue
            if sorted(ds) != list(range(min(ds), max(ds)+1)):
                continue
            slot, start = slots[0], min(ds)
            corridor = {_add(base, u, slot, n, -d) for d in range(1, max(ds)+1)}
            if corridor.isdisjoint(occupied):
                matches.append((index, slot, start, len(cells), color))
        assert len(matches) == 1, "source attribution must be unique"
        index, slot, start, period, color = matches[0]
        address = slot, start
        assert address not in program or program[address] == (period, color), "inconsistent repeated fragment"
        program[address] = period, color
        owners.setdefault(address, set()).add(index)
    code = sorted((s, d, p, c) for (s, d), (p, c) in program.items())
    _valid_program(code, length)
    source_depth = max(d+p-1 for _, d, p, _ in code)
    for i, e in enumerate(devices):
        full = _device(e['base'], e['u'], e['n'], length, a.shape, source_depth)
        assert full is not None, "source envelope or outward strip leaves canvas"
        e.update(full)
        assert all(_separate(e['rect'], old['rect']) for old in devices[:i]), "intersecting working strips"
    return {"input": a, "background": bg, "template": template, "length": length,
            "devices": devices, "code": code, "owners": owners, "source_depth": source_depth}


def _prediction(info, alternative=None):
    """Cell-wise modular rule; also supports explicit, named rival hypotheses."""
    a, length, code = info['input'], info['length'], info['code']
    out = a.copy()
    for index, e in enumerate(info['devices']):
        entries = code
        if alternative == 'local_only':
            entries = [x for x in code if index in info['owners'][(x[0], x[1])]]
        for t in range(e['depth']):
            for slot in range(length):
                firing = []
                for source, start, period, color in entries:
                    tick = t-start if alternative == 'gap_phase' else t
                    interval = 1 if alternative == 'unit_period' else period
                    if tick < 0 or tick % interval:
                        continue
                    k = tick//interval
                    target = (source if alternative == 'static' else
                              source-k if alternative == 'reverse_walk' else source+k)
                    if alternative != 'no_wrap':
                        target %= length
                    if target == slot:
                        firing.append((source, start, period, color))
                if not firing:
                    continue
                if alternative == 'far':
                    winner = max(firing, key=lambda x: x[1])
                elif alternative == 'short':
                    winner = min(firing, key=lambda x: (x[2], x[1]))
                elif alternative == 'long':
                    winner = min(firing, key=lambda x: (-x[2], x[1]))
                else:
                    winner = min(firing, key=lambda x: x[1])
                p = _add(e['base'], e['u'], slot, e['n'], t+2)
                assert int(a[p]) == info['background'], "outward strip must start empty"
                out[p] = winner[3]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode visible fragments, pool by device coordinates, and emit moving pulses."""
    return _prediction(_decode(input_grid))


def _paint(template, devices, code, allocations, shape, bg):
    """Construction uses known frames and far-to-near event overwrites."""
    a = np.full(shape, bg, dtype=np.int8)
    for e in devices:
        for row in (0, 1):
            for slot in range(template.shape[1]):
                a[_add(e['base'], e['u'], slot, e['n'], row)] = template[row, slot]
    for entry, indices in zip(code, allocations):
        slot, start, period, color = entry
        for index in indices:
            e = devices[index]
            for k in range(period):
                a[_add(e['base'], e['u'], slot, e['n'], -start-k)] = color
    out = a.copy()
    for source, start, period, color in sorted(code, key=lambda x: -x[1]):
        for e in devices:
            for k, t in enumerate(range(0, e['depth'], period)):
                slot = (source+k) % template.shape[1]
                out[_add(e['base'], e['u'], slot, e['n'], t+2)] = color
    return {'input': a, 'output': out}


def _template(rng, length, palette):
    a = np.full((2, length), palette[0], dtype=np.int8)
    a[1, 0] = palette[1]
    if rng.random() < .55:
        candidates = [i for i in range(2*length) if i != length]
        count = int(rng.integers(2, min(6, 2*length-2)))
        chosen = rng.choice(candidates, count, replace=False)
        a.flat[chosen] = palette[2]
    return a


def _layout(rng, template, transforms, source_depth, min_draw, shape):
    length = template.shape[1]
    options = []
    for t in transforms:
        patch = _orient(template, t)
        corner, u, n = _frame(template, t)
        choices = []
        for r in range(shape[0]-patch.shape[0]+1):
            for c in range(shape[1]-patch.shape[1]+1):
                e = _device((r+corner[0], c+corner[1]), u, n, length, shape, source_depth)
                if e is not None and e['depth'] >= min_draw:
                    choices.append(e)
        rng.shuffle(choices)
        options.append(choices)
    placed, visits = [], 0

    def search(index):
        nonlocal visits
        if index == len(options):
            return list(placed)
        for e in options[index]:
            visits += 1
            if visits > 2400:
                return None
            if all(_separate(e['rect'], old['rect']) for old in placed):
                placed.append(e)
                result = search(index+1)
                if result is not None:
                    return result
                placed.pop()
        return None

    return search(0)


def _fallback(template, count, source_depth, min_draw, lesson):
    length = template.shape[1]
    shape = (30, 30)
    devices = []
    if lesson == 'rotate':
        # Two perpendicular strips in opposite corners, with one blank gap.
        frames = [((30-min_draw-2, 0), (0, 1), (1, 0)),
                  ((length-1, 30-min_draw-2), (-1, 0), (0, 1))]
    else:
        frames = []
        for i in range(count):
            reverse = (i % 2 == 1)
            c = i*(length+1)+(length-1 if reverse else 0)
            frames.append(((source_depth, c), (0, -1) if reverse else (0, 1), (1, 0)))
    for base, u, n in frames:
        e = _device(base, u, n, length, shape, source_depth)
        assert e is not None and e['depth'] >= min_draw
        assert all(_separate(e['rect'], x['rect']) for x in devices)
        devices.append(e)
    return devices, shape


def _allocations(rng, code, count, split, lesson):
    if lesson:
        return [{0}, {1}]
    if split == 'test':
        # At least one entire copy lacks clues; sometimes one copy supplies all.
        eligible = list(map(int, rng.permutation(count)))[:int(rng.integers(1, count))]
    else:
        eligible = list(map(int, rng.permutation(count)))
    mode = int(rng.integers(3))
    answer = [{int(rng.choice(eligible))} for _ in code]
    if mode == 0:
        answer = [{eligible[0]} for _ in code]
    if split == 'train':
        for i, owner in enumerate(eligible):
            answer[i % len(code)].add(owner)
    elif mode == 2:
        answer[0].update(eligible)
    return answer


def _make_pair(rng, split, lesson, witness, reverse_priority):
    fast, slow, witness_colors = witness
    for attempt in range(16):
        if lesson:
            # Enumerate the small feasible geometry set, avoiding a chance of
            # repeatedly sampling an impossible source-depth/length combination.
            collision = math.lcm(fast, slow)
            geometry = []
            for near_candidate in (2, 3):
                far_candidate = near_candidate+max(fast, slow)+1
                ds = (far_candidate, near_candidate) if reverse_priority else (near_candidate, far_candidate)
                extent = max(ds[0]+fast-1, ds[1]+slow-1)
                for v in range(4, 10):
                    draw = max(collision+1, fast*v+1 if lesson == 'reflect' else 1)
                    if (2*v+1 <= 30 and extent+2+draw <= 30 and
                            (lesson != 'rotate' or extent+2+draw+v+1 <= 30)):
                        geometry.append((near_candidate, v))
            assert geometry, 'witness periods must admit a clear lesson'
            near, length = geometry[int(rng.integers(len(geometry)))]
            far = near+max(fast, slow)+1
            periods = (fast, slow)
            starts = (far, near) if reverse_priority else (near, far)
            source_depth = max(starts[i]+periods[i]-1 for i in range(2))
            start_slot = int(rng.integers(length))
            delta = collision//fast-collision//slow
            code = [(start_slot, starts[0], fast, witness_colors[0]),
                    ((start_slot+delta) % length, starts[1], slow, witness_colors[1])]
            count = 2
            min_draw = max(collision+1, fast*length+1 if lesson == 'reflect' else 1)
            payload = list(witness_colors)
        elif attempt == 15:
            # Guaranteed readable general case after the bounded random trials.
            # This is a small construction, not a stored puzzle or solver cache.
            length, count, source_depth, min_draw = 5, 3, 8, 9
            code = [(0, 2, 2, witness_colors[0]), (1, 6, 3, witness_colors[1])]
        else:
            length = int(rng.integers(4, 10))
            count = int(rng.integers(2 if split == 'train' else 3,
                                     min(4, 31//(length+1))+1))
            code_count = int(rng.integers(1 if split == 'train' else 2, 5))
            payload = list(map(int, rng.choice(10, int(rng.integers(2, 5)), replace=False)))
            code = []
            for _ in range(120):
                e = (int(rng.integers(length)), int(rng.integers(2, 11)),
                     int(rng.integers(1, 7)), int(rng.choice(payload)))
                try:
                    _valid_program(code+[e], length)
                except AssertionError:
                    continue
                code.append(e)
                if len(code) == code_count:
                    break
            if len(code) != code_count:
                continue
            source_depth = max(d+p-1 for _, d, p, _ in code)
            min_draw = max(5, max(e[2] for e in code)+1)
        _valid_program(code, length)
        available = [c for c in range(10) if c not in {e[3] for e in code}]
        rng.shuffle(available)
        bg, structural = available[0], available[1:4]
        template = _template(rng, length, structural)
        base = int(rng.integers(8))
        if lesson == 'reflect':
            transforms = [base, base ^ 4]
        elif lesson == 'rotate':
            transforms = [base, (base//4)*4+(base+1) % 4]
        else:
            mode = int(rng.integers(3))
            transforms = ([base]*count if mode == 0 else
                          [base if i % 2 == 0 else base ^ 4 for i in range(count)] if mode == 1 else
                          list(map(int, rng.integers(8, size=count))))
        required = source_depth+2+min_draw
        if required > 30:
            continue
        shape = [int(rng.integers(max(12, required), 31)) for _ in range(2)]
        if lesson == 'reflect' or len({t % 2 for t in transforms}) == 1:
            cross = count*length+count-1
            axis = 1 if base % 2 == 0 else 0
            shape[axis] = int(rng.integers(cross, 31))
        shape = tuple(shape)
        devices = (None if attempt == 15 else
                   _layout(rng, template, transforms, source_depth, min_draw, shape))
        if devices is None:
            devices, shape = _fallback(template, count, source_depth, min_draw, lesson)
        allocations = _allocations(rng, code, count, split, lesson)
        pair = _paint(template, devices, code, allocations, shape, bg)
        # Global isometries vary compass directions even in the fallback.
        t = int(rng.integers(8))
        pair = {role: _orient(grid, t) for role, grid in pair.items()}
        try:
            info = _decode(pair['input'])
            assert np.array_equal(pair['output'], _prediction(info))
            if split == 'test':
                assert not np.array_equal(pair['output'], _prediction(info, 'unit_period'))
                assert not np.array_equal(pair['output'], _prediction(info, 'static'))
                assert not np.array_equal(pair['output'], _prediction(info, 'local_only'))
            return pair
        except AssertionError:
            continue
    raise RuntimeError('bounded layout attempts exhausted')


def _evidence(info, output):
    code, length = info['code'], info['length']
    collided, short, long, edges, wrap = 0, False, False, set(), False
    seen = set()
    for e in info['devices']:
        events = {}
        for i, (source, start, period, color) in enumerate(code):
            for k, t in enumerate(range(0, e['depth'], period)):
                events.setdefault((t, (source+k) % length), []).append((i, k))
        for (t, slot), hits in events.items():
            i, k = min(hits, key=lambda v: code[v[0]][1])
            p = _add(e['base'], e['u'], slot, e['n'], t+2)
            assert output[p] == code[i][3]
            seen.add(i)
            wrap |= source_wrap(code[i][0], k, length)
            for j, _ in hits:
                if j != i and code[i][3] != code[j][3]:
                    collided += 1
                    short |= code[i][2] < code[j][2]
                    long |= code[i][2] > code[j][2]
                    edges.add((code[i][3], code[j][3]))
    allocations = [sum(index in info['owners'][(s, d)] for s, d, _, _ in code)
                   for index in range(len(info['devices']))]
    handedness = {e['u'][0]*e['n'][1]-e['u'][1]*e['n'][0] for e in info['devices']}
    return {'short': short, 'long': long, 'edges': edges, 'wrap': wrap,
            'collision_count': collided, 'visible_clocks': len(seen),
            'pooled': max(allocations) < len(code), 'unhinted': 0 in allocations,
            'reflection': len(handedness) == 2,
            'mixed_axes': len({e['n'][0] == 0 for e in info['devices']}) == 2,
            'allocations': tuple(allocations)}


def source_wrap(source, pulse_index, length):
    return source+pulse_index >= length


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos, evidence = set(), {'train': [], 'test': []}, {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _array(pair['input']), _array(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            expected = _prediction(info)
            assert np.array_equal(out, expected), 'wrong timing, motion, priority or preservation'
            assert not np.array_equal(inp, out)
            assert np.all(out[inp != info['background']] == inp[inp != info['background']]), 'source or device changed'
            ev = _evidence(info, out)
            if split == 'test':
                assert ev['unhinted'], 'test shift needs a wholly unprogrammed copy'
                for rival in ('unit_period', 'static', 'local_only'):
                    assert not np.array_equal(out, _prediction(info, rival)), 'parent omission has no effect'
            else:
                assert not ev['unhinted'], 'training copies must each carry some evidence'
            infos[split].append(info)
            evidence[split].append(ev)
    train = evidence['train']
    assert any(e['pooled'] and e['reflection'] for e in train), 'missing reflected pooling demonstration'
    assert any(e['mixed_axes'] for e in train), 'missing quarter-turn demonstration'
    assert any(e['wrap'] for e in train), 'no visible wrap'
    assert any(e['short'] for e in train) and any(e['long'] for e in train), 'priority teaching gap'
    edges = set().union(*(e['edges'] for e in train))
    assert any((b, a) in edges for a, b in edges), 'fixed color ranking is not refuted'
    for rival in ('unit_period', 'static', 'local_only', 'no_wrap', 'reverse_walk', 'gap_phase', 'far', 'short', 'long'):
        assert any(not np.array_equal(pair['output'], _prediction(info, rival))
                   for pair, info in zip(puzzle['train'], infos['train'])), f'rival {rival} fits training'


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    fast, slow = ((2, 3), (2, 4), (3, 4), (2, 5))[int(rng.integers(4))]
    witness_colors = tuple(map(int, rng.choice(10, 2, replace=False)))
    witness = fast, slow, witness_colors
    polarity = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    # Lesson locations are shuffled; their semantic evidence is checked globally.
    lessons = [('reflect', polarity), ('rotate', not polarity)] + [(None, False)]*(num_train-2)
    rng.shuffle(lessons)
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            lesson, reverse = lessons[index] if split == 'train' else (None, False)
            for _ in range(12):
                pair = _make_pair(rng, split, lesson, witness, reverse)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded uniqueness attempts exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; bounded metamorphic, count, and hand-event checks."""
    shapes, structures, puzzles, lengths, periods, counts, directions = set(), set(), set(), set(), set(), set(), set()
    backgrounds, allocations = set(), set()
    totals = Counter()
    # Hand-computed p=2 at slot 0, p=3 at slot 1 on a length-four loom.
    # At t=6 both reach slot 3; at t=8 the first clock wraps to slot 0.
    template = np.array([[8, 8, 8, 8], [9, 8, 8, 8]], dtype=np.int8)
    shape = (21, 11)
    devices = [_device((8, 0), (0, 1), (1, 0), 4, shape, 8),
               _device((8, 9), (0, -1), (1, 0), 4, shape, 8)]
    code = [(0, 2, 2, 3), (1, 6, 3, 4)]
    fixture = _paint(template, devices, code, [{0}, {1}], shape, 0)
    expected = fixture['input'].copy()
    # Explicit event/color rows, independent of the generation event loop.
    rows = {0: {0: 3, 1: 4}, 2: {1: 3}, 3: {2: 4}, 4: {2: 3},
            6: {3: 3}, 8: {0: 3}, 9: {0: 4}, 10: {1: 3}}
    for e in devices:
        for t, row in rows.items():
            for slot, color in row.items():
                expected[_add(e['base'], e['u'], slot, e['n'], t+2)] = color
    assert np.array_equal(solve(fixture['input']), expected), 'hand event regression'
    reverse = _paint(template, devices, [(0, 6, 2, 3), (1, 2, 3, 4)], [{0}, {1}], shape, 0)
    assert solve(reverse['input'])[16, 3] == 4, 'near slow clock must win at t=6'
    clipped_shape = (11, 11)
    clipped_devices = [_device((8, 0), (0, 1), (1, 0), 4, clipped_shape, 8),
                       _device((8, 9), (0, -1), (1, 0), 4, clipped_shape, 8)]
    clipped = _paint(template, clipped_devices, code, [{0}, {1}], clipped_shape, 0)
    clipped_expected = clipped['input'].copy()
    clipped_expected[10, [0, 9]] = 3
    clipped_expected[10, [1, 8]] = 4
    assert np.array_equal(solve(clipped['input']), clipped_expected), 'one-cell boundary clipping'
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                ev = _evidence(info, pair['output'])
                shapes.add(pair['input'].shape)
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)
                structures.add((pair['input'].shape, normalized))
                lengths.add(info['length'])
                periods.update(e[2] for e in info['code'])
                counts.add((len(info['devices']), len(info['code'])))
                directions.update(e['n'] for e in info['devices'])
                backgrounds.add(info['background'])
                allocations.add(ev['allocations'])
                totals['pairs'] += 1
                for name in ('pooled', 'unhinted', 'reflection', 'mixed_axes', 'wrap'):
                    totals[name] += int(ev[name])
        if seed < 3:
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
        if seed == 0:
            for pair in puzzle['train']+puzzle['test']:
                for t in range(8):
                    assert np.array_equal(solve(_orient(pair['input'], t)), _orient(pair['output'], t))
                mapping = np.array([7, 2, 9, 4, 0, 6, 1, 8, 5, 3], dtype=np.int8)
                assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
            bad = {s: [{r: np.array(a, copy=True) for r, a in p.items()} for p in puzzle[s]] for s in puzzle}
            bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('output corruption accepted')
    assert len(puzzles) == 42 and len(structures) >= 180 and len(shapes) >= 25
    assert len(lengths) >= 5 and periods == set(range(1, 7))
    assert {a for a, _ in counts} == {2, 3, 4} and {b for _, b in counts} == {1, 2, 3, 4}
    assert directions == set(_STEPS) and len(backgrounds) >= 9 and len(allocations) >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(4242+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'distinct_puzzles': len(puzzles), 'shapes': len(shapes),
            'color_normalized_inputs': len(structures), 'lengths': sorted(lengths),
            'periods': sorted(periods), 'device_clock_counts': sorted(counts),
            'clue_allocations': len(allocations), 'features': dict(totals),
            'independent_solver': False, 'blind_model_check': False}
