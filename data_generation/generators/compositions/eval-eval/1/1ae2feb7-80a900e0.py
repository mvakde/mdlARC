"""Two-ended barcode caps: periodic, hollow diagonal emissions.

No parent code or data is imported. All solving conventions are geometric and
input-visible. Generation uses specified barcodes and walking rails; validation
also uses a cell-wise coordinate formula reconstructed from the returned input.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _base(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _outline(center, radius):
    r, c = center
    return [(r + a, c + b) for a in range(-radius, radius + 1)
            for b in range(-radius, radius + 1) if abs(a) + abs(b) == radius]


def _cap(center, radius, direction):
    r, c = center
    a, b = DIRECTIONS[direction]
    return [(r + a * (k + 1), c + b * (radius - k + 1)) for k in range(radius + 1)]


def _runs(sequence):
    return [(int(color), len(list(group))) for color, group in itertools.groupby(sequence)]


def _path(start, vector, shape):
    p = (start[0] + vector[0], start[1] + vector[1])
    result = []
    while _inside(p, shape):
        result.append(p)
        p = (p[0] + vector[0], p[1] + vector[1])
    return result


def _rails(emitter, shape):
    for side, barcode in sorted(emitter['caps'].items()):
        cap = _cap(emitter['center'], emitter['radius'], side)
        for end, sequence in ((0, barcode), (-1, barcode[::-1])):
            yield side, end, _path(cap[end], DIRECTIONS[side], shape), _runs(sequence)


def _analyze(grid):
    """Recover checkerboard, complete outlines, and complete barcode caps."""
    grid = np.asarray(grid)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
    parity = np.indices(grid.shape).sum(axis=0) % 2
    background = []
    for p in (0, 1):
        counts = np.bincount(grid[parity == p].astype(int), minlength=10)
        color = int(counts.argmax())
        assert counts[color] * 2 > int(np.count_nonzero(parity == p)), 'unclear checkerboard'
        background.append(color)
    assert background[0] != background[1]
    base = _base(grid.shape, background)
    foreground = set(map(tuple, np.argwhere(grid != base)))
    emitters, covered = [], set()
    h, w = grid.shape
    for radius in range(2, 6):
        if min(h, w) < 2 * radius + 1:
            continue
        shape = h - 2 * radius, w - 2 * radius
        offsets = _outline((0, 0), radius)
        dr, dc = offsets[0]
        colors = grid[radius + dr:radius + dr + shape[0], radius + dc:radius + dc + shape[1]]
        mask = (colors != background[0]) & (colors != background[1])
        for dr, dc in offsets[1:]:
            mask &= grid[radius + dr:radius + dr + shape[0], radius + dc:radius + dc + shape[1]] == colors
        for rr, cc in np.argwhere(mask):
            center = int(rr) + radius, int(cc) + radius
            cells = set(_outline(center, radius))
            assert not cells & covered, 'overlapping outline interpretations'
            covered.update(cells)
            emitters.append(dict(center=center, radius=radius, struct=int(colors[rr, cc]), caps={}))
    assert 1 <= len(emitters) <= 2
    for e in emitters:
        for side in range(4):
            cells = _cap(e['center'], e['radius'], side)
            if not all(_inside(p, grid.shape) for p in cells):
                continue
            sequence = tuple(int(grid[p]) for p in cells)
            if all(v not in (*background, e['struct']) for v in sequence):
                assert not set(cells) & covered
                assert 1 <= len(_runs(sequence)) <= 4
                e['caps'][side] = sequence
                covered.update(cells)
        assert e['caps'], 'each diamond needs a barcode'
    assert covered == foreground, 'unassigned foreground or incomplete barcode'
    occupied_paths = set()
    for e in emitters:
        for _, _, path, _ in _rails(e, grid.shape):
            assert not set(path) & (covered | occupied_paths), 'rail obstruction or crossing'
            occupied_paths.update(path)
    return dict(background=tuple(background), base=base, emitters=sorted(emitters, key=lambda e: e['center']))


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid)
    info = _analyze(grid)
    output = grid.copy()
    for e in info['emitters']:
        for _, _, path, runs in _rails(e, grid.shape):
            for t, cell in enumerate(path):
                for color, period in runs:
                    if t % period == 0:
                        output[cell] = color
                        break
    return output


def _coordinate_prediction(grid, info):
    """Closed-form membership and modular priority, without walking paths."""
    out = grid.copy()
    rows, cols = np.indices(grid.shape)
    used = np.zeros(grid.shape, dtype=bool)
    for e in info['emitters']:
        for side, barcode in e['caps'].items():
            a, b = DIRECTIONS[side]
            u, v = (rows - e['center'][0]) * a, (cols - e['center'][1]) * b
            t = np.minimum(u, v) - 2
            for delta, sequence in ((-e['radius'], barcode), (e['radius'], barcode[::-1])):
                rail = (u - v == delta) & (t >= 0)
                assert not np.any(used & rail)
                assert np.all(grid[rail] == info['base'][rail])
                used |= rail
                remaining = rail.copy()
                for color, length in _runs(sequence):
                    firing = remaining & (t % length == 0)
                    out[firing] = color
                    remaining &= ~firing
    return out


def _rival(grid, info, variant):
    """Deliberately simpler alternatives used to audit demonstrations."""
    out = grid.copy()
    for e in info['emitters']:
        for side, barcode in e['caps'].items():
            cap = _cap(e['center'], e['radius'], side)
            indices = range(len(cap)) if variant == 'whole_cap' else (0, len(cap) - 1)
            for index in indices:
                sequence = barcode if index <= e['radius'] / 2 else barcode[::-1]
                runs = _runs(sequence)
                if variant == 'far':
                    runs.reverse()
                elif variant == 'short':
                    runs.sort(key=lambda x: x[1])
                elif variant == 'long':
                    runs.sort(key=lambda x: -x[1])
                for t, cell in enumerate(_path(cap[index], DIRECTIONS[side], grid.shape)):
                    if grid[cell] != info['base'][cell]:
                        continue
                    if variant == 'continuous':
                        out[cell] = sequence[0]
                    elif variant == 'tile':
                        out[cell] = sequence[t % len(sequence)]
                    else:
                        for color, period in runs:
                            if (t + int(variant == 'phase')) % period == 0:
                                out[cell] = color
                                break
    return out


def _footprint(e, shape):
    objects = set(_outline(e['center'], e['radius']))
    for side in e['caps']:
        objects.update(_cap(e['center'], e['radius'], side))
    if not all(_inside(p, shape) for p in objects):
        return None
    paths = {p for _, _, path, _ in _rails(e, shape) for p in path}
    return objects, paths


def _render(shape, background, emitters):
    inp = _base(shape, background)
    for e in emitters:
        for p in _outline(e['center'], e['radius']):
            inp[p] = e['struct']
        for side, sequence in e['caps'].items():
            for p, color in zip(_cap(e['center'], e['radius'], side), sequence):
                inp[p] = color
    # Construction uses declared runs and far-to-near slice-like overwrites.
    out = inp.copy()
    for e in emitters:
        for _, _, path, runs in _rails(e, shape):
            for color, length in reversed(runs):
                for cell in path[::length]:
                    out[cell] = color
    info = _analyze(inp)
    assert info['emitters'] == sorted(emitters, key=lambda e: e['center'])
    assert np.array_equal(out, _coordinate_prediction(inp, info))
    return {'input': inp, 'output': out}


def _barcode(rng, length, palette, parts=None):
    if parts is None:
        count = int(rng.integers(1, min(4, length) + 1))
        cuts = sorted(map(int, rng.choice(range(1, length), count - 1, replace=False)))
        stops = [0] + cuts + [length]
        parts = [b - a for a, b in zip(stops, stops[1:])]
    colors, sequence = [], []
    for size in parts:
        color = int(rng.choice([c for c in palette if not colors or c != colors[-1]]))
        colors.append(color)
        sequence.extend([color] * size)
    return tuple(sequence)


def _sample_pair(rng, max_radius, count, background, witness=None, edge=False, min_rail=0):
    """Bounded free placement; retries resample geometry, never the rule."""
    for restart in range(160):
        shape = (int(rng.integers(25, 31)), int(rng.integers(25, 31))) if witness else (
            (int(rng.integers(25, 31)), int(rng.integers(25, 31))) if count == 2 else
            (int(rng.integers(14, 31)), int(rng.integers(14, 31))))
        emitters, objects, paths = [], set(), set()
        for index in range(count):
            radius = witness['radius'] if witness and index == 0 else int(rng.integers(2, max_radius + 1))
            mask = int(rng.integers(1, 16))
            sides = witness['sides'] if witness and index == 0 else [s for s in range(4) if mask & (1 << s)]
            structural = int(rng.choice([c for c in range(10) if c not in background]))
            palette = [c for c in range(10) if c not in (*background, structural)]
            caps = {s: _barcode(rng, radius + 1, palette) for s in sides}
            if witness and index == 0:
                parts = [2, 3] if witness['kind'] == 'priority' else [radius + 1]
                caps[sides[0]] = _barcode(rng, radius + 1, palette, parts)
            elif index == 0:
                # Every ordinary scene contains an informative mixed barcode.
                split = int(rng.integers(1, radius + 1))
                caps[sides[0]] = _barcode(rng, radius + 1, palette, [split, radius + 1 - split])
            for _ in range(100):
                center = tuple(int(rng.integers(radius, n - radius)) for n in shape)
                e = dict(center=center, radius=radius, struct=structural, caps=caps)
                fp = _footprint(e, shape)
                if fp is None:
                    continue
                cells, new_paths = fp
                if cells & (objects | paths) or new_paths & (objects | paths):
                    continue
                lengths = [len(path) for _, _, path, _ in _rails(e, shape)]
                if min(lengths) < (7 if witness else min_rail):
                    continue
                if edge and index == 0 and not (min(lengths) <= 2 and max(lengths) >= 3):
                    continue
                emitters.append(e)
                objects.update(cells)
                paths.update(new_paths)
                break
            else:
                break
        if len(emitters) != count:
            continue
        try:
            pair = _render(shape, background, emitters)
            info = _analyze(pair['input'])
            if any(np.array_equal(pair['output'], _rival(pair['input'], info, v))
                   for v in ('continuous', 'whole_cap')):
                continue
            return pair
        except AssertionError:
            continue
    raise RuntimeError('bounded free-placement search exhausted')


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    max_radius = int(rng.choice([4, 5]))
    count_shift = bool(rng.integers(2))
    stable_palette = bool(rng.integers(2))
    background = tuple(map(int, rng.choice(10, 2, replace=False)))
    first_mask = int(rng.integers(1, 15))
    masks = (first_mask, (15 ^ first_mask) | int(rng.integers(1, 16)))
    slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    witnesses = {slot: dict(radius=radius, kind=kind,
                           sides=[s for s in range(4) if mask & (1 << s)])
                 for slot, radius, kind, mask in zip(slots, (4, max_radius), ('priority', 'single'), masks)}
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = witnesses.get(index) if split == 'train' else None
            n = 1 if witness or (count_shift and split == 'train') else (2 if count_shift else int(rng.integers(1, 3)))
            for _ in range(12):
                colors = background if stable_palette else tuple(map(int, rng.choice(10, 2, replace=False)))
                pair = _sample_pair(rng, max_radius, n, colors, witness,
                                    edge=split == 'test' and not count_shift,
                                    min_rail=4 if split == 'train' else 0)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('unique input attempts exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _teaching(train):
    directions, priority, single_periods, max_radius = set(), False, set(), 0
    for pair in train:
        inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
        info = _analyze(inp)
        for e in info['emitters']:
            max_radius = max(max_radius, e['radius'])
            for side, barcode in e['caps'].items():
                rails = [(path, runs) for s, _, path, runs in _rails(e, inp.shape) if s == side]
                probes = [(p[0] + k * DIRECTIONS[side][0], p[1] + k * DIRECTIONS[side][1])
                          for p in _cap(e['center'], e['radius'], side)[1:-1] for k in (1, 2)]
                hollow = all(_inside(p, inp.shape) and inp[p] == info['base'][p] and out[p] == inp[p]
                             for p in probes)
                if min(len(p) for p, _ in rails) >= 7 and hollow:
                    directions.add(side)
                    lengths = [length for _, length in _runs(barcode)]
                    if lengths in ([2, 3], [3, 2]):
                        priority = True
                    if len(lengths) == 1:
                        period = lengths[0]
                        if all(len(p) > period and out[p[0]] == out[p[period]] == runs[0][0]
                               for p, runs in rails):
                            single_periods.add(period)
                # Other emitters may place a rail inside this hollow strip.
                # Such a side is valid, but is not counted as a clear witness.
    return directions, priority, single_periods, max_radius


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    a, b = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, max_test_radius = set(), 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(out, _coordinate_prediction(inp, info)), 'wrong rail, period, phase, priority, or preserved cell'
            foreground = inp != info['base']
            assert np.array_equal(inp[foreground], out[foreground])
            for variant in ('continuous', 'whole_cap'):
                assert not np.array_equal(out, _rival(inp, info, variant)), f'no visible {variant} counterexample'
            if split == 'test':
                max_test_radius = max(max_test_radius, max(e['radius'] for e in info['emitters']))
    directions, priority, singles, max_taught = _teaching(puzzle['train'])
    assert directions == set(range(4)) and priority and singles, 'missing teaching evidence'
    assert max_taught >= max_test_radius
    for variant in ('short', 'long', 'far', 'phase', 'tile'):
        assert any(not np.array_equal(np.asarray(p['output']), _rival(np.asarray(p['input']), _analyze(np.asarray(p['input'])), variant))
                   for p in puzzle['train']), f'rival {variant} fits demonstrations'
    train_infos = [_analyze(np.asarray(p['input'])) for p in puzzle['train']]
    train_counts = [len(i['emitters']) for i in train_infos]
    minimum_train_rail = min(len(path) for p, i in zip(puzzle['train'], train_infos)
                             for e in i['emitters'] for _, _, path, _ in _rails(e, np.asarray(p['input']).shape))
    for pair in puzzle['test']:
        inp = np.asarray(pair['input'])
        info = _analyze(inp)
        lengths = [len(path) for e in info['emitters'] for _, _, path, _ in _rails(e, inp.shape)]
        assert len(info['emitters']) > max(train_counts) or min(lengths) < minimum_train_rail, 'missing count or boundary shift'


def _features(pair):
    inp = np.asarray(pair['input'])
    info = _analyze(inp)
    es = info['emitters']
    return dict(shape=inp.shape, background=info['background'], count=len(es),
                radii=[e['radius'] for e in es], masks=[sum(1 << s for s in e['caps']) for e in es],
                partitions=[tuple(n for _, n in _runs(seq)) for e in es for seq in e['caps'].values()],
                lengths=[len(p) for e in es for _, _, p, _ in _rails(e, inp.shape)],
                layout=(inp.shape, tuple((e['center'], e['radius'], tuple((s, tuple(n for _, n in _runs(seq)))
                        for s, seq in sorted(e['caps'].items()))) for e in es)))


def self_test():
    # A fixed, hand-calculated two-ended pulse diagnostic (background omitted).
    a = [(7, 2), (8, 3)]
    b = list(reversed(a))
    def pulse(runs):
        return [next((c for c, n in runs if t % n == 0), 0) for t in range(10)]
    assert pulse(a) == [7, 0, 7, 8, 7, 0, 7, 0, 7, 8]
    assert pulse(b) == [8, 0, 7, 8, 7, 0, 8, 0, 7, 8]
    # Hand-placed geometry and hand-enumerated output cells. This diagnostic
    # does not use cap/path/render helpers or the coordinate predictor.
    rows, cols = np.indices((25, 25))
    fixture = ((rows + cols) % 2).astype(np.int8)
    fixture[np.abs(rows - 12) + np.abs(cols - 12) == 4] = 3
    for cell, color in { (13, 17): 7, (14, 16): 7, (15, 15): 8,
                         (16, 14): 8, (17, 13): 8 }.items():
        fixture[cell] = color
    expected = fixture.copy()
    for cell, color in { (14, 18): 7, (16, 20): 7, (17, 21): 8, (18, 22): 7, (20, 24): 7,
                         (18, 14): 8, (20, 16): 7, (21, 17): 8, (22, 18): 7, (24, 20): 8 }.items():
        expected[cell] = color
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(fixture, turn)), np.rot90(expected, turn))
    # One scene at a time: no 42-puzzle payload or diagnostic trace is retained.
    shapes, layouts, radii, masks, partitions, counts, backgrounds, fingerprints = (set() for _ in range(8))
    min_length, max_length, mixed_counts, shifts = 100, 0, 0, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        local_counts = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(f['shape']); layouts.add(f['layout'])
                radii.update(f['radii']); masks.update(f['masks']); partitions.update(f['partitions'])
                counts.add(f['count']); backgrounds.add(f['background']); local_counts.add(f['count'])
                min_length = min(min_length, min(f['lengths']))
                max_length = max(max_length, max(f['lengths']))
                for role in ('input', 'output'):
                    signature.update(str(pair[role].shape).encode())
                    signature.update(pair[role].tobytes())
                inp, out = pair['input'], pair['output']
                assert np.array_equal(solve(inp), out)
                # Rotation and palette equivariance test input-only inference.
                if seed < 4:
                    for turn in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn))
                    mapping = np.arange(9, -1, -1, dtype=np.int8)
                    assert np.array_equal(solve(mapping[inp]), mapping[out])
        fingerprints.add(signature.digest())
        mixed_counts += len(local_counts) > 1
        train_max = max(_features(p)['count'] for p in puzzle['train'])
        shifts.add('count' if _features(puzzle['test'][0])['count'] > train_max else 'edge')
    assert len(fingerprints) == 42 and len(shapes) >= 40 and len(layouts) >= 150
    assert radii == {2, 3, 4, 5} and masks == set(range(1, 16)) and counts == {1, 2}
    assert len(partitions) >= 25 and len(backgrounds) >= 25
    assert min_length == 0 and max_length >= 15 and mixed_counts >= 20 and shifts == {'count', 'edge'}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(300 + 10 * a + b, a, b)
            assert len(puzzle['train']) == a and len(puzzle['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(_key(x[r]) == _key(y[r]) for s in p for x, y in zip(p[s], q[s]) for r in x)
    changed = next(tuple(x) for x in np.argwhere(p['test'][0]['input'] != p['test'][0]['output']))
    p['test'][0]['output'][changed] = p['test'][0]['input'][changed]
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('mutated label accepted')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), shapes=len(shapes), layouts=len(layouts),
                radii=sorted(radii), masks=sorted(masks), partitions=len(partitions),
                ray_length_range=[min_length, max_length], mixed_count_puzzles=mixed_counts,
                shifts=sorted(shifts), retained_puzzle_files=0)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
