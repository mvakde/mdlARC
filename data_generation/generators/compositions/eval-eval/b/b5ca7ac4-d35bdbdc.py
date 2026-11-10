"""Forwarded-address packing. No file access, hidden puzzle state, or parent code.

All sampling uses one default_rng. Validation reconstructs visible relations and
also packs survivors by separate one-cell movement. self_test streams its corpus.
"""
from collections import Counter
import hashlib

import numpy as np


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _band(a, b):
    return max(a['r'], b['r']) < min(a['r'] + a['h'], b['r'] + b['h'])


def _overlap(a, b):
    return (_band(a, b)
            and max(a['c'], b['c']) < min(a['c'] + a['w'], b['c'] + b['w']))


def _separate(a, b):
    # A blank row or column separates the entire input footprints.
    return (a['r'] + a['h'] < b['r'] or b['r'] + b['h'] < a['r']
            or a['c'] + a['w'] < b['c'] or b['c'] + b['w'] < a['c'])


def _patch(o, forwarded=False):
    a = np.full((o['h'], o['w']), o['body'], dtype=np.int8)
    a[o['h'] // 2, o['w'] // 2] = o['resolved'] if forwarded else o['value']
    return a


def _render(shape, background, rails, objects, forwarded=False):
    h, w = shape
    result = np.full((h, w + 2), background, dtype=np.int8)
    result[:, 0], result[:, -1] = rails
    occupied = np.zeros(shape, dtype=bool)
    for o in objects:
        r, c, oh, ow = (o[k] for k in ('r', 'c', 'h', 'w'))
        assert 0 <= r <= h - oh and 0 <= c <= w - ow
        assert not occupied[r:r + oh, c:c + ow].any(), 'overlapping render'
        occupied[r:r + oh, c:c + ow] = True
        result[r:r + oh, c + 1:c + 1 + ow] = _patch(o, forwarded)
    return result


def _decode(input_grid):
    a = _grid(input_grid)
    candidates = []
    for transpose in (False, True):
        g = a.T if transpose else a
        if (g.shape[1] >= 5 and np.all(g[:, 0] == g[0, 0])
                and np.all(g[:, -1] == g[0, -1]) and g[0, 0] != g[0, -1]):
            candidates.append((transpose, g))
    assert len(candidates) == 1, 'need one pair of opposite full rails'
    transpose, g = candidates[0]
    rails = int(g[0, 0]), int(g[0, -1])
    strip = g[:, 1:-1]
    counts = np.bincount(strip.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] * 2 > strip.size and bg not in rails, 'majority background'
    todo = set(zip(*np.where(strip != bg)))
    objects = []
    while todo:
        first = min(todo)
        todo.remove(first)
        cells, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    stack.append(q)
        r = min(p[0] for p in cells)
        c = min(p[1] for p in cells)
        h = int(max(p[0] for p in cells) - r + 1)
        w = int(max(p[1] for p in cells) - c + 1)
        assert h in (3, 5) and w in (3, 5) and len(cells) == h * w, 'complete odd tile'
        p = strip[r:r + h, c:c + w]
        body, value = int(p[0, 0]), int(p[h // 2, w // 2])
        expected = np.full_like(p, body)
        expected[h // 2, w // 2] = value
        assert np.array_equal(p, expected), 'uniform body except center'
        assert body != bg and value != bg
        objects.append(dict(r=int(r), c=int(c), h=h, w=w, body=body, value=value,
                            index=len(objects)))
    assert objects
    assert all(_separate(a, b) for i, a in enumerate(objects) for b in objects[i + 1:])
    mapping = {}
    for o in objects:
        assert o['body'] not in mapping or mapping[o['body']] == o['value'], 'conflicting donors'
        mapping[o['body']] = o['value']
    return transpose, strip.shape, bg, rails, objects


def _resolve(objects, rails):
    result = []
    for i, o in enumerate(objects):
        donors = [j for j, p in enumerate(objects) if j != i and p['body'] == o['value']]
        values = {objects[j]['value'] for j in donors}
        assert len(values) <= 1
        z = next(iter(values)) if values else None
        d = rails.index(z) if z in rails else -1
        result.append(dict(o, donors=donors, resolved=z, d=d))
    return result


def _settle(objects, width):
    result = [dict(o) for o in objects if o['d'] >= 0]
    for d in (0, 1):
        group = [o for o in result if o['d'] == d]
        group.sort(key=lambda o: (o['c'] if d == 0 else -o['c'] - o['w'], o['r']))
        previous = []
        for o in group:
            blockers = [p for p in previous if _band(o, p)]
            o['c'] = (max((p['c'] + p['w'] for p in blockers), default=0) if d == 0
                      else min((p['c'] for p in blockers), default=width) - o['w'])
            o['depth'] = max((p['depth'] for p in blockers), default=0) + 1
            assert 0 <= o['c'] <= width - o['w'], 'packing chain exceeds width'
            previous.append(o)
    assert all(not _overlap(a, b) for i, a in enumerate(result) for b in result[i + 1:]), 'final collision'
    return result


def _unit_steps(objects, shape):
    """Separate collision check: unit moves cannot jump an earlier blocker."""
    result = [dict(o) for o in objects if o['d'] >= 0]
    for d, delta in ((0, -1), (1, 1)):
        group = [o for o in result if o['d'] == d]
        occupied = np.zeros(shape, dtype=bool)
        for o in group:
            occupied[o['r']:o['r'] + o['h'], o['c']:o['c'] + o['w']] = True
        for _ in range(shape[1] * max(1, len(group)) + 1):
            moved = False
            for o in group:
                r, c, h, w = (o[k] for k in ('r', 'c', 'h', 'w'))
                occupied[r:r + h, c:c + w] = False
                target = c + delta
                if 0 <= target <= shape[1] - w and not occupied[r:r + h, target:target + w].any():
                    o['c'] = target
                    moved = True
                occupied[r:r + h, o['c']:o['c'] + w] = True
            if not moved:
                break
        else:
            raise AssertionError('unit-step movement failed to converge')
    return result


def _gap_fill(objects, width):
    """Named competing shortcut, used only to check the training contrast."""
    result = []
    for d in (0, 1):
        group = [dict(o) for o in objects if o['d'] == d]
        group.sort(key=lambda o: (o['c'] if d == 0 else -o['c'] - o['w'], o['r']))
        previous = []
        for o in group:
            positions = range(width - o['w'] + 1)
            if d:
                positions = reversed(positions)
            for c in positions:
                candidate = dict(o, c=c)
                if all(not _overlap(candidate, p) for p in previous):
                    previous.append(candidate)
                    result.append(candidate)
                    break
            else:
                raise AssertionError('gap fill cannot fit')
    return {o['index']: o['c'] for o in result}


def _analyze(input_grid):
    transpose, shape, bg, rails, objects = _decode(input_grid)
    resolved = _resolve(objects, rails)
    settled = _settle(resolved, shape[1])
    return transpose, shape, bg, rails, resolved, settled


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from visible input pixels only."""
    transpose, shape, bg, rails, _, settled = _analyze(input_grid)
    result = _render(shape, bg, rails, settled, forwarded=True)
    return result.T.copy() if transpose else result


def _random_relations(rng, rails, background):
    pool = [c for c in range(10) if c != background]
    for _ in range(96):
        k = int(rng.integers(3, 8))
        bodies = list(map(int, rng.choice(pool, k, replace=False)))
        mapping = {b: int(_pick(rng, pool)) for b in bodies}
        active = [b for b in bodies if mapping[b] != b and mapping[b] in mapping
                  and mapping[mapping[b]] in rails]
        if active:
            return mapping, bodies, int(_pick(rng, active))
    # Finite semantic fallback, still a visible partial functional map.
    a, b, t = [c for c in pool if c not in rails][:3]
    mapping = {a: b, b: rails[0], rails[0]: rails[1], rails[1]: t}
    return mapping, list(mapping), a


def _geometry(rng, bodies, mapping, rails, chain_length, direction, max_depth, force_witness):
    """Random geometry first; spacious constructive geometry after bounded tries."""
    for _ in range(28):
        ch = int(_pick(rng, (3, 3, 5)))
        widths = [int(_pick(rng, (3, 3, 5))) for _ in range(chain_length)]
        # Long chains need room for input gaps as well as final footprints.
        if sum(widths) + chain_length + 1 > 28:
            continue
        minimum_w = max(15, sum(widths) + chain_length + 1)
        w = int(rng.integers(minimum_w, 29))
        step = (ch + 1) // 2
        span = ch + (chain_length - 1) * step
        h = int(rng.integers(max(15, span + 3), 31))
        if rng.random() < .16:
            h = 30
            w = 28
        row = int(rng.integers(h - span + 1))
        reverse = bool(rng.integers(2))
        spare = w - sum(widths) - (chain_length - 1)
        gaps = rng.multinomial(spare, np.full(chain_length + 1, 1 / (chain_length + 1)))
        cursor = int(gaps[0])
        objects = []
        for i, ow in enumerate(widths):
            c = cursor if direction == 0 else w - cursor - ow
            r = row + ((chain_length - 1 - i) if reverse else i) * step
            body = bodies[i]
            objects.append(dict(r=r, c=c, h=ch, w=ow, body=body, value=mapping[body], index=i))
            cursor += ow + 1 + int(gaps[i + 1])
        good = True
        for i in range(chain_length, len(bodies)):
            body = bodies[i]
            found = False
            for _ in range(56):
                oh, ow = int(_pick(rng, (3, 3, 5))), int(_pick(rng, (3, 3, 5)))
                o = dict(r=int(rng.integers(h - oh + 1)), c=int(rng.integers(w - ow + 1)),
                         h=oh, w=ow, body=body, value=mapping[body], index=i)
                if all(_separate(o, p) for p in objects):
                    objects.append(o)
                    found = True
                    break
            if not found:
                good = False
                break
        if not good or sum(o['h'] * o['w'] for o in objects) * 2 >= h * w:
            continue
        try:
            resolved = _resolve(objects, rails)
            settled = _settle(resolved, w)
            if max((o['depth'] for o in settled), default=0) > max_depth:
                continue
            if force_witness and _gap_fill(resolved, w) == {o['index']: o['c'] for o in settled}:
                continue
        except AssertionError:
            continue
        return (h, w), objects
    # Up to five diagonal chain tiles, plus at most eight other tiles in three
    # separate rows of at most three. Final row widths are at most 15 and 9.
    # This guarantees capacity, separation and the required training depth cap.
    objects = []
    for i, body in enumerate(bodies):
        if i < chain_length:
            c = 2 + 4 * i
            if direction:
                c = 28 - c - 3
            r = 1 + 2 * i
        else:
            j = i - chain_length
            r, c = 17 + 4 * (j // 3), 2 + 9 * (j % 3)
        objects.append(dict(r=r, c=c, h=3, w=3, body=body, value=mapping[body], index=i))
    assert len(bodies) - chain_length <= 9
    return (30, 28), objects


def _make_pair(rng, parameters, mode, chain_length, background):
    rails = parameters['rails']
    if mode in (0, 1):
        a, b, c, d, t = parameters['identities']
        near, far = rails[mode], rails[1 - mode]
        mapping = {a: b, b: near, near: far, far: t, c: b, d: far}
        bodies = [a if i % 2 == 0 else c for i in range(chain_length)] + [b, near, far, d]
        direction = mode
    else:
        mapping, keys, address = _random_relations(rng, rails, background)
        direction = rails.index(mapping[mapping[address]])
        bodies = [address] * chain_length + [b for b in keys if b != address]
        # Extra copies vary multiplicities, including consistent donor copies.
        extras = int(rng.integers(3))
        bodies += [int(_pick(rng, keys)) for _ in range(extras)]
    shape, objects = _geometry(rng, bodies, mapping, rails, chain_length, direction,
                               3 if mode != 'test' else 20, mode == 1)
    resolved = _resolve(objects, rails)
    settled = _settle(resolved, shape[1])
    inp = _render(shape, background, rails, objects)
    out = _render(shape, background, rails, settled, forwarded=True)
    if parameters['transpose']:
        inp, out = inp.T.copy(), out.T.copy()
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    parameters = dict(rails=tuple(palette[:2]), identities=palette[2:7],
                      backgrounds=palette[7:], transpose=bool(rng.integers(2)),
                      varied_background=bool(rng.integers(2)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            mode = i if split == 'train' and i < 2 else ('test' if split == 'test' else 'random')
            length = (2 + i if isinstance(mode, int) else
                      int(rng.integers(4, 6)) if split == 'test' else int(rng.integers(1, 4)))
            bg = (parameters['backgrounds'][i] if isinstance(mode, int)
                  else int(_pick(rng, parameters['backgrounds'])))
            if not parameters['varied_background']:
                bg = parameters['backgrounds'][0]
            for _ in range(40):
                pair = _make_pair(rng, parameters, mode, length, bg)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('bounded distinct-input sampling exhausted')
    # Randomize demonstration order without changing the two required witnesses.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp):
    transpose, shape, bg, rails, objects, settled = _analyze(inp)
    kept = {o['index'] for o in settled}
    by_id = {o['index']: o for o in objects}
    moving = sum(o['c'] != by_id[o['index']]['c'] for o in settled)
    return dict(axis=int(transpose), shape=tuple(inp.shape), background=bg,
                rails=rails, tiles=len(objects), survivors=len(settled),
                depths=tuple(max((o['depth'] for o in settled if o['d'] == d), default=0) for d in (0, 1)),
                footprints=tuple(sorted({(o['h'], o['w']) for o in objects})), moving=moving,
                erased_donor=any(j not in kept for o in settled for j in o['donors']),
                surviving_donor=any(j in kept and by_id[j]['resolved'] != by_id[j]['value']
                                    for o in settled for j in o['donors']),
                nonterminal=any(o['donors'] and o['d'] < 0 for o in objects),
                missing=any(not o['donors'] for o in objects),
                monochrome=any(o['body'] == o['value'] for o in objects),
                duplicate=len(objects) > len({o['body'] for o in objects}),
                stationary=any(o['c'] == by_id[o['index']]['c'] for o in settled),
                stable_witness=_gap_fill(objects, shape[1]) != {o['index']: o['c'] for o in settled},
                layout=tuple((o['r'], o['c'], o['h'], o['w']) for o in objects))


def measure_puzzle(puzzle):
    return [dict(split=split, index=i, **_features(np.asarray(pair['input'])))
            for split in ('train', 'test') for i, pair in enumerate(puzzle[split])]


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, convention, relations = set(), None, {}
    witnesses = Counter()
    max_train_depth = 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            transpose, shape, bg, rails, objects, settled = _analyze(inp)
            current = transpose, rails
            if convention is None:
                convention = current
            assert current == convention, 'puzzle axis and rails must be consistent'
            expected = _render(shape, bg, rails, settled, forwarded=True)
            actual = out.T if transpose else out
            assert np.array_equal(actual, expected), 'incorrect dispatch, patch or packing'
            stepped = _unit_steps(objects, shape)
            assert np.array_equal(_render(shape, bg, rails, stepped, True), actual), 'unit-step mismatch'
            assert np.all(actual[:, 0] == rails[0]) and np.all(actual[:, -1] == rails[1])
            # Every surviving footprint has its exact original body and only its
            # center changed. Equality above also forbids any unaccounted pixels.
            for o in settled:
                original = objects[o['index']]
                assert (o['r'], o['h'], o['w'], o['body']) == tuple(original[k] for k in ('r', 'h', 'w', 'body'))
            depth = max((o['depth'] for o in settled), default=0)
            f = _features(inp)
            if split == 'test':
                assert depth >= 4, 'test must extend the demonstrated packing chain'
                continue
            assert depth <= 3
            max_train_depth = max(max_train_depth, depth)
            for name in ('erased_donor', 'surviving_donor', 'nonterminal', 'missing', 'stable_witness'):
                witnesses[name] += bool(f[name])
            for o in objects:
                if o['d'] >= 0:
                    relations.setdefault((o['body'], o['value']), set()).add(o['d'])
            inplace = _render(shape, bg, rails, [o for o in objects if o['d'] >= 0], True)
            witnesses['without_A'] += not np.array_equal(inplace, actual)
            direct = [dict(o, d=rails.index(o['value']), resolved=o['value'])
                      for o in objects if o['value'] in rails]
            try:
                wrong = _render(shape, bg, rails, _settle(direct, shape[1]), True)
                witnesses['without_B'] += not np.array_equal(wrong, actual)
            except AssertionError:
                witnesses['without_B'] += 1
    assert max_train_depth == 3, 'training must teach a three-tile chain'
    assert any(v == {0, 1} for v in relations.values()), 'same relation must dispatch both ways'
    assert all(witnesses[k] for k in ('erased_donor', 'surviving_donor', 'nonterminal', 'missing',
                                    'stable_witness', 'without_A', 'without_B')), dict(witnesses)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for s in ('train', 'test'):
        for p in puzzle[s]:
            for k in ('input', 'output'):
                a = np.asarray(p[k], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed streaming audit; no files or saved execution traces."""
    fingerprints, shapes, layouts, axes, palettes, counts, footprints, depths = (set() for _ in range(8))
    features = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for f in measure_puzzle(puzzle):
            shapes.add(f['shape'])
            layouts.add(f['layout'])
            axes.add(f['axis'])
            palettes.add(f['rails'])
            counts.add(f['tiles'])
            footprints.update(f['footprints'])
            depths.update(f['depths'])
            for name in ('erased_donor', 'surviving_donor', 'nonterminal', 'missing', 'monochrome',
                         'duplicate', 'stationary', 'stable_witness'):
                features[name] += bool(f[name])
    assert len(fingerprints) == 42 and axes == {0, 1}
    assert len(shapes) >= 50 and len(layouts) >= 150 and len(palettes) >= 25
    assert len(counts) >= 6 and footprints == {(3, 3), (3, 5), (5, 3), (5, 5)}
    assert {1, 2, 3, 4, 5} <= depths
    assert all(features[k] > 0 for k in features)
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    # Explicit expected locations/centers, independent of the forwarding and
    # packing routines. The last chain tile cannot jump back to column zero.
    relations = [(0, 3, 3, 4), (2, 9, 6, 4), (4, 15, 3, 4),
                 (10, 8, 4, 1), (14, 0, 1, 2), (14, 8, 2, 5), (14, 16, 7, 2)]
    tiles = [dict(r=r, c=c, h=3, w=3, body=b, value=v) for r, c, b, v in relations]
    fixture = _render((19, 24), 0, (1, 2), tiles)
    expected = [dict(tiles[i], c=c, resolved=z) for i, c, z in
                ((0, 0, 1), (1, 3, 1), (2, 6, 1), (3, 21, 2))]
    assert np.array_equal(solve(fixture), _render((19, 24), 0, (1, 2), expected, True))
    # A monochrome tile is useful to another tile but cannot supply itself.
    mono = [dict(r=7, c=8, h=3, w=3, body=1, value=1),
            dict(r=1, c=10, h=3, w=3, body=3, value=1)]
    expected = [dict(mono[1], c=0, resolved=1)]
    assert np.array_equal(solve(_render((12, 18), 0, (1, 2), mono)),
                          _render((12, 18), 0, (1, 2), expected, True))
    mono.append(dict(r=7, c=2, h=3, w=3, body=1, value=1))
    expected = [dict(mono[0], c=3, resolved=1), dict(mono[1], c=0, resolved=1),
                dict(mono[2], c=0, resolved=1)]
    assert np.array_equal(solve(_render((12, 18), 0, (1, 2), mono)),
                          _render((12, 18), 0, (1, 2), expected, True))
    # A two-cycle at the terminals produces two solid tiles, simultaneously.
    cycle = [dict(r=2, c=8, h=3, w=3, body=1, value=2),
             dict(r=4, c=2, h=3, w=3, body=2, value=1)]
    expected = [dict(cycle[0], c=0, resolved=1), dict(cycle[1], c=15, resolved=2)]
    assert np.array_equal(solve(_render((9, 18), 0, (1, 2), cycle)),
                          _render((9, 18), 0, (1, 2), expected, True))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(3, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), shapes=len(shapes),
                layouts=len(layouts), rail_palettes=len(palettes), tile_counts=sorted(counts),
                footprints=sorted(footprints), depths=sorted(depths), features=dict(features))
