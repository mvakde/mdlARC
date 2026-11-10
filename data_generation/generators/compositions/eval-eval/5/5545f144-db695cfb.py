"""ARC family: a rigid dart follows consumable, conditionally active mirrors.

Only NumPy and the standard library are used. Decoding never reads a seed or an
example collection. See task_spec.md for the geometric domain and conventions.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
RING = frozenset((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                 if abs(r) == 1 or abs(c) == 1)


def _turn(p, n):
    r, c = p
    for _ in range(n % 4):
        r, c = c, -r
    return r, c


def _dart(span, shaft, solid, tail):
    cells = {(0, -i) for i in range(shaft)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if solid else (-depth, depth)
        cells.update((r, -shaft + 1 - depth) for r in rows)
    for length in range(1, tail + 1):
        cells.update((r, -shaft + 1 - span - length) for r in (-span, span))
    return tuple(sorted(cells))


MASKS = tuple(dict.fromkeys(_dart(s, e, fill, tail)
                           for s in (1, 2) for e in (1, 2)
                           for fill in (False, True) for tail in (0, 1)))


def _cells(mask, tip, heading):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_turn(p, heading - 1) for p in mask)}


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _components(cells):
    unseen, result = set(cells), []
    while unseen:
        start = unseen.pop()
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in unseen:
                        unseen.remove(p)
                        component.add(p)
                        stack.append(p)
        result.append(component)
    return result


def _identify_dart(cells):
    """Infer the symmetry axis and projecting tip from the actual bitmap."""
    matches = []
    for tip in sorted(cells):
        for heading in range(4):
            offsets = {_turn((r - tip[0], c - tip[1]), 1 - heading)
                       for r, c in cells}
            if (all(c <= 0 for r, c in offsets)
                    and {p for p in offsets if p[1] == 0} == {(0, 0)}
                    and any(r != 0 for r, c in offsets)
                    and {(-r, c) for r, c in offsets} == offsets):
                matches.append((tuple(sorted(offsets)), tip, heading))
    assert len(matches) == 1, "dart must have one symmetry axis and projecting tip"
    return matches[0]


def _spans(centers):
    edges, degree = [], Counter()
    for i, a in enumerate(centers):
        for b in centers[i + 1:]:
            if abs(a[0] - b[0]) == abs(a[1] - b[1]):
                assert abs(a[0] - b[0]) >= 4
                slope = 1 if (b[0] - a[0]) * (b[1] - a[1]) > 0 else -1
                edges.append((a, b, slope))
                degree[a] += 1
                degree[b] += 1
    assert edges and max(degree.values()) == 1, "endpoint partners must be unique"
    return edges


def _on_span(p, edge):
    a, b, s = edge
    return (p[1] - a[1] == s * (p[0] - a[0])
            and min(a[0], b[0]) < p[0] < max(a[0], b[0]))


def _mirror(p, primary_slope):
    slope = -primary_slope
    return slope, p[1] - slope * p[0]


def _decode(grid):
    grid = np.asarray(grid)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 4 and counts.max() > grid.size // 2
    bg = int(colors[np.argmax(counts)])
    components = {int(color): _components(map(tuple, np.argwhere(grid == color)))
                  for color in colors if color != bg}
    ring_options = []
    for color, groups in components.items():
        centers = []
        for g in groups:
            r0, c0 = min(r for r, c in g), min(c for r, c in g)
            center = r0 + 1, c0 + 1
            if g != {(center[0] + r, center[1] + c) for r, c in RING}:
                break
            assert grid[center] == bg, "endpoint centers must be blank"
            centers.append(center)
        else:
            if len(centers) >= 2:
                ring_options.append((color, sorted(centers)))
    assert len(ring_options) == 1
    primary, centers = ring_options[0]
    dot_options = [c for c, groups in components.items()
                   if c != primary and all(len(g) == 1 for g in groups)]
    assert len(dot_options) == 1, "switches must be isolated singletons"
    secondary = dot_options[0]
    ink = next(c for c in components if c not in (primary, secondary))
    assert len(components[ink]) == 1
    mask, tip, heading = _identify_dart(components[ink][0])
    assert 0 < tip[0] < grid.shape[0] - 1 and 0 < tip[1] < grid.shape[1] - 1
    edges = _spans(centers)
    dots = sorted(next(iter(g)) for g in components[secondary])
    mirrors, active, inactive, beyond = [], [], [], []
    for p in dots:
        containing = [e for e in edges if _on_span(p, e)]
        assert len(containing) <= 1, "a switch cannot belong to two spans"
        if containing:
            mirrors.append(_mirror(p, containing[0][2]))
            active.append(p)
        else:
            inactive.append(p)
            aligned = [e for e in edges
                       if p[1] - e[0][1] == e[2] * (p[0] - e[0][0])]
            if aligned:
                assert len(aligned) == 1
                beyond.append((p, _mirror(p, aligned[0][2])))
    assert mirrors and len(mirrors) == len(set(mirrors))
    assert all(tip[1] - s * tip[0] != k for s, k in mirrors)
    return dict(shape=grid.shape, bg=bg, ink=ink, primary=primary,
                secondary=secondary, centers=centers, edges=edges, dots=dots,
                active=active, inactive=inactive, beyond=beyond, mirrors=mirrors,
                mask=mask, tip=tip, heading=heading)


def _boundary_distance(tip, vector, shape):
    r, c = tip
    dr, dc = vector
    return (shape[0] - 1 - r if dr > 0 else r) if dr else (
        shape[1] - 1 - c if dc > 0 else c)


def _trace(shape, tip, heading, mirrors, consume=True, farthest=False, sites=None):
    """Analytic ray intersections, no raster drawing or generation metadata.

    Boundary arrival wins over a deflector at that boundary. Persistent mode is
    used only for a teaching counterfactual; None denotes a repeated state.
    """
    remaining = list(range(len(mirrors)))
    hits, states = [], set()
    while True:
        state = (tip, heading, tuple(remaining))
        if state in states:
            return None
        states.add(state)
        dr, dc = DIRS[heading]
        exit_distance = _boundary_distance(tip, (dr, dc), shape)
        choices = []
        for i in remaining:
            s, k = mirrors[i]
            distance = (k - tip[1] + s * tip[0]) // (dc - s * dr)
            if 0 < distance < exit_distance:
                if sites is not None and (tip[0] + dr * distance, tip[1] + dc * distance) != sites[i]:
                    continue
                choices.append((distance, i))
        if not choices:
            end = tip[0] + dr * exit_distance, tip[1] + dc * exit_distance
            return end, heading, hits
        choices.sort()
        distance, i = choices[-1] if farthest else choices[0]
        assert sum(d == distance for d, _ in choices) == 1, "simultaneous mirror hit"
        position = tip[0] + dr * distance, tip[1] + dc * distance
        s, _ = mirrors[i]
        new_heading = DIRS.index((s * dc, s * dr))
        hits.append((i, position, heading, new_heading, distance))
        tip, heading = position, new_heading
        if consume:
            remaining.remove(i)


def _render(a, endpoint, heading):
    cells = _cells(a['mask'], endpoint, heading)
    assert _inside(cells, a['shape']), "the whole terminal mask must fit"
    result = np.full(a['shape'], a['bg'], dtype=np.int8)
    for p in cells:
        result[p] = a['ink']
    return result


def _analyze(grid):
    a = _decode(grid)
    end, heading, hits = _trace(a['shape'], a['tip'], a['heading'], a['mirrors'])
    for _, p, before, after, _ in hits:
        assert _inside(_cells(a['mask'], p, before), a['shape'])
        assert _inside(_cells(a['mask'], p, after), a['shape'])
    a.update(end=end, final=heading, hits=hits)
    return _render(a, end, heading), a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return only the rigid dart after its tip first reaches the boundary."""
    return _analyze(np.asarray(input_grid))[0]


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _different_trace(a, mirrors, consume=True, **kwargs):
    try:
        alternative = _trace(a['shape'], a['tip'], a['heading'], mirrors, consume, **kwargs)
    except AssertionError:
        # An ambiguous alternative is not a useful distinguishing witness.
        return False
    return alternative is None or alternative[:2] != (a['end'], a['final'])


def _features(a):
    persistent = _different_trace(a, a['mirrors'], False)
    beyond = any(_different_trace(a, a['mirrors'] + [m]) for p, m in a['beyond'])
    remote = max((max(abs(p[0] - a['active'][i][0]), abs(p[1] - a['active'][i][1]))
                  for i, p, _, _, _ in a['hits']), default=0)
    same_slope = list(dict.fromkeys((-s, p[1] + s * p[0])
                                   for p, (s, k) in zip(a['active'], a['mirrors'])))
    return dict(consumption=persistent, finite=beyond, remote=remote,
                turns=len(a['hits']), slopes={s for s, k in a['mirrors']},
                no_b=_different_trace(a, []), rotated=a['heading'] != a['final'],
                nearest=_different_trace(a, a['mirrors'], farthest=True),
                perpendicular=_different_trace(a, same_slope),
                full_line=_different_trace(a, a['mirrors'], sites=a['active']))


def _sample_geometry(rng, shape, target, parallel=False, need_beyond=False):
    """Place freely located ring centers; no path/layout bank is used."""
    h, w = shape
    if parallel:
        specs = [(int(rng.choice((-1, 1))), target)]
    elif need_beyond:
        s = int(rng.choice((-1, 1)))
        specs = [(s, target - 1), (-s, 1)]
    else:
        number = int(rng.integers(1, min(3, target) + 1))
        counts = [1] * number
        for _ in range(target - number):
            counts[int(rng.integers(number))] += 1
        shared_slope = int(rng.choice((-1, 1))) if rng.random() < .35 else None
        specs = [(shared_slope or int(rng.choice((-1, 1))), n) for n in counts]
    if len(specs) < 3 and rng.random() < .30:
        specs.append((int(rng.choice((-1, 1))), 0))
    centers, edges, active, mirrors, ring_cells = [], [], [], [], set()
    for slope, count in specs:
        for attempt in range(90):
            steps = int(rng.integers(max(5, 2 * count + 2), min(h, w) - 2))
            r = int(rng.integers(1, h - steps - 1))
            c = int(rng.integers(1, w - steps - 1)) + (steps if slope == -1 else 0)
            a, b = (r, c), (r + steps, c + slope * steps)
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) < 4
                   or abs(p[0] - q[0]) == abs(p[1] - q[1])
                   for p in (a, b) for q in centers):
                continue
            new_rings = {(p[0] + dr, p[1] + dc) for p in (a, b) for dr, dc in RING}
            if new_rings & set(active):
                continue
            edge = a, b, slope
            if any(_on_span(p, edge) for p in active):
                continue
            eligible = [(r + j, c + slope * j) for j in range(2, steps - 1)]
            chosen = []
            for ix in rng.permutation(len(eligible)):
                p = eligible[int(ix)]
                if p in ring_cells or any(_on_span(p, e) for e in edges):
                    continue
                if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in active + chosen):
                    continue
                m = _mirror(p, slope)
                if m in mirrors:
                    continue
                chosen.append(p)
                if len(chosen) == count:
                    break
            if count and len(chosen) != count:
                continue
            if not count:
                chosen = []
            centers.extend((a, b))
            edges.append(edge)
            ring_cells |= new_rings
            active.extend(chosen)
            mirrors.extend(_mirror(p, slope) for p in chosen)
            break
        else:
            return None
    # Some unpaired endpoint glyphs are inert too.
    if rng.random() < .20:
        for _ in range(30):
            p = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
            cells = {(p[0] + r, p[1] + c) for r, c in RING}
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) < 4
                   or abs(p[0] - q[0]) == abs(p[1] - q[1]) for q in centers):
                continue
            if cells & set(active):
                continue
            centers.append(p)
            ring_cells |= cells
            break
    inactive, beyond = [], []
    for a, b, s in edges:
        for j in range(-max(shape), max(shape) + 1):
            p = a[0] + j, a[1] + s * j
            if not _inside([p], shape) or p in ring_cells or any(_on_span(p, e) for e in edges):
                continue
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in active):
                continue
            if p in centers:
                continue
            aligned = [e for e in edges if p[1] - e[0][1] == e[2] * (p[0] - e[0][0])]
            if len(aligned) == 1 and _mirror(p, s) not in mirrors:
                beyond.append((p, _mirror(p, s)))
    if need_beyond and not beyond:
        return None
    if beyond and (need_beyond or rng.random() < .60):
        p, m = beyond[int(rng.integers(len(beyond)))]
        inactive.append(p)
        beyond = [(p, m)]
    else:
        beyond = []
    for _ in range(int(rng.integers(3))):
        for attempt in range(35):
            p = int(rng.integers(h)), int(rng.integers(w))
            if p in ring_cells or p in centers or any(_on_span(p, e) for e in edges):
                continue
            if any(p[1] - e[0][1] == e[2] * (p[0] - e[0][0]) for e in edges):
                continue
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in active + inactive):
                continue
            inactive.append(p)
            break
    return dict(centers=centers, edges=edges, active=active, inactive=inactive,
                mirrors=mirrors, ring_cells=ring_cells, beyond=beyond)


def _make_pair(rng, palette, mask, bounds, target, witness, used):
    low, high = bounds
    for attempt in range(220):
        shape = int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))
        geometry = _sample_geometry(rng, shape, target, witness == 'parallel', witness == 'mixed')
        if geometry is None:
            continue
        occupied = geometry['ring_cells'] | set(geometry['centers']) | set(geometry['active']) | set(geometry['inactive'])
        for _ in range(200):
            tip = int(rng.integers(1, shape[0] - 1)), int(rng.integers(1, shape[1] - 1))
            heading = int(rng.integers(4))
            start = _cells(mask, tip, heading)
            if not _inside(start, shape) or start & occupied:
                continue
            if any(tip[1] - s * tip[0] == k for s, k in geometry['mirrors']):
                continue
            try:
                end, final, hits = _trace(shape, tip, heading, geometry['mirrors'])
            except AssertionError:
                continue
            if not hits or not _inside(_cells(mask, end, final), shape):
                continue
            if any(not _inside(_cells(mask, p, d), shape)
                   for i, p, before, after, dist in hits for d in (before, after)):
                continue
            a = dict(geometry, shape=shape, mask=mask, tip=tip, heading=heading,
                     end=end, final=final, hits=hits)
            f = _features(a)
            if not f['no_b'] or f['remote'] < 2 or not f['full_line']:
                continue
            if witness == 'parallel' and (len(hits) < 2 or not f['consumption']):
                continue
            if witness == 'mixed' and (len(hits) < 3 or not f['rotated'] or not f['finite']
                                       or not f['nearest'] or not f['perpendicular']):
                continue
            bg, primary, secondary, ink = palette
            grid = np.full(shape, bg, dtype=np.int8)
            for p in geometry['ring_cells']:
                grid[p] = primary
            for p in geometry['active'] + geometry['inactive']:
                grid[p] = secondary
            for p in start:
                grid[p] = ink
            if _key(grid) in used or np.count_nonzero(grid == bg) <= grid.size // 2:
                continue
            a.update(bg=bg, primary=primary, secondary=secondary, ink=ink)
            output = _render(a, end, final)
            return {'input': grid, 'output': output}
    raise RuntimeError(f'bounded geometric search exhausted: {witness}, target={target}, bounds={bounds}')


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # active-count, board-scale, or new-mask shift
    shared_palette = bool(rng.random() < .25)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    mask_order = list(map(int, rng.permutation(len(MASKS))))
    novel = MASKS[mask_order.pop()]
    train_pool = [MASKS[i] for i in mask_order] if regime == 2 else list(MASKS)
    first = train_pool[int(rng.integers(len(train_pool)))]
    second_pool = [m for m in train_pool if m != first]
    same_area = [m for m in second_pool if len(m) == len(first)]
    if same_area and rng.random() < .5:
        second_pool = same_area
    second = second_pool[int(rng.integers(len(second_pool)))]
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            witness = ('parallel' if i == 0 else 'mixed' if i == 1 else '') if split == 'train' else ''
            target = 2 if witness == 'parallel' else 3 if witness == 'mixed' else int(rng.integers(1, 4))
            if split == 'test' and regime == 0:
                target = int(rng.integers(4, 6))
            bounds = ((15, 22) if split == 'train' else (24, 30)) if regime == 1 else (
                (15, 28) if split == 'train' else (17, 30))
            mask = first if split == 'train' and i == 0 else second if split == 'train' and i == 1 else (
                novel if split == 'test' and regime == 2 else train_pool[int(rng.integers(len(train_pool)))])
            if not shared_palette:
                palette = tuple(map(int, rng.choice(10, 4, replace=False)))
            pair = _make_pair(rng, palette, mask, bounds, target, witness, used)
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for value in pair.values():
                g = np.asarray(value)
                assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
                assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))
            assert _key(pair['input']) not in seen, 'duplicate input'
            seen.add(_key(pair['input']))
            output, a = _analyze(pair['input'])
            assert np.array_equal(output, pair['output']), 'incorrect terminal dart'
            assert output.shape == np.asarray(pair['input']).shape
            assert set(map(int, np.unique(output))) == {a['bg'], a['ink']}
            assert np.count_nonzero(output == a['ink']) == len(a['mask'])
            assert len(a['hits']) == len({hit[0] for hit in a['hits']})
            assert a['end'][0] in (0, output.shape[0] - 1) or a['end'][1] in (0, output.shape[1] - 1)
            f = _features(a)
            assert f['no_b'] and f['remote'] >= 2 and f['full_line'], 'every pair must need remote deflection'
            analyses[split].append((a, f))
    train = analyses['train']
    assert len({a['mask'] for a, f in train}) >= 2, 'demonstrate actual mask preservation'
    assert any(f['consumption'] and f['turns'] >= 2 for a, f in train)
    assert any(f['finite'] and f['rotated'] and f['turns'] >= 3
               and f['nearest'] and f['perpendicular'] for a, f in train)
    assert set().union(*(f['slopes'] for a, f in train)) == {-1, 1}
    # At least one intended extrapolation is visible without knowing the seed.
    tests = [a for a, f in analyses['test']]
    assert (all(len(a['mirrors']) > max(len(t['mirrors']) for t, f in train) for a in tests)
            or all(min(a['shape']) > max(max(t['shape']) for t, f in train) for a in tests)
            or all(a['mask'] not in {t['mask'] for t, f in train} for a in tests))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                out, a = _analyze(pair['input'])
                assert np.array_equal(out, pair['output'])
                records.append((split, a, _features(a)))
    assert len({_key(p['test'][0]['input']) for p in puzzles}) == 42
    assert len({a['shape'] for s, a, f in records}) >= 70
    assert len({a['mask'] for s, a, f in records}) >= 12
    assert {a['heading'] for s, a, f in records} == set(range(4))
    assert {a['final'] for s, a, f in records} == set(range(4))
    assert {len(a['mirrors']) for s, a, f in records} >= {1, 2, 3, 4, 5}
    assert len({len(a['centers']) for s, a, f in records}) >= 4
    assert len({tuple(a['hits'][i][1] for i in range(len(a['hits']))) for s, a, f in records}) >= 150
    assert all({a[role] for s, a, f in records} == set(range(10))
               for role in ('bg', 'ink', 'primary', 'secondary'))
    assert any(len(a['hits']) < len(a['mirrors']) for s, a, f in records)
    assert any(not a['inactive'] for s, a, f in records)
    assert any(a['heading'] == a['final'] for s, a, f in records)
    # Repeated seeds and all legal counts. This is bounded, with no external IO.
    def signature(p):
        return tuple((_key(pair['input']), _key(pair['output']))
                     for s in ('train', 'test') for pair in p[s])
    assert signature(generate_puzzle(0)) == signature(puzzles[0])
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + train * 10 + test, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = copy.deepcopy(puzzles[0])
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
