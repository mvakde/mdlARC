"""Rotational stencil relay: shape-scoped, simultaneous marked-point transfer.

Only standard library and NumPy. No file I/O, stored examples, or parent imports.
"""
from collections import Counter
from hashlib import sha256

import numpy as np


_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _crop(mask):
    ys, xs = np.where(mask)
    return mask[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()


def _key(mask):
    return mask.shape, np.asarray(mask, dtype=np.uint8).tobytes()


def _canonical(mask):
    variants = [np.rot90(mask, k) for k in range(4)]
    keys = [_key(m) for m in variants]
    k = min(range(4), key=lambda i: keys[i])
    return variants[k].copy(), k, keys[k]


def _parts(cells, directions=_D8):
    remaining = set(cells)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r+dr, c+dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _parse(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert int(counts.max()) * 2 > a.size, 'background must occupy over half the grid'
    objects, bodies = [], set()
    for cells in _parts(map(tuple, np.argwhere(a != bg))):
        assert 5 <= len(cells) <= 30
        top = min(r for r, c in cells)
        left = min(c for r, c in cells)
        h = max(r for r, c in cells)-top+1
        w = max(c for r, c in cells)-left+1
        assert 2 <= h <= 6 and 2 <= w <= 6
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r-top, c-left] = True
        colors_in_piece = Counter(int(a[p]) for p in cells)
        assert len(colors_in_piece) == 2
        body = max(colors_in_piece, key=colors_in_piece.get)
        value = next(c for c in colors_in_piece if c != body)
        assert colors_in_piece[body] == len(cells)-1 and colors_in_piece[value] == 1
        dot = next(p for p in sorted(cells) if int(a[p]) == value)
        canonical, turn, shape_key = _canonical(mask)
        assert len({_key(np.rot90(mask, k)) for k in range(4)}) == 4, 'ambiguous rotation'
        identity = shape_key, body
        assert identity not in bodies, 'duplicate body identity within a shape class'
        bodies.add(identity)
        dot_mask = np.zeros(mask.shape, dtype=bool)
        dot_mask[dot[0]-top, dot[1]-left] = True
        registered = tuple(map(int, np.argwhere(np.rot90(dot_mask, turn))[0]))
        objects.append(dict(cells=cells, top=top, left=left, mask=mask,
                            canonical=canonical, turn=turn, key=shape_key,
                            body=body, value=value, dot=dot, registered=registered))
    assert 2 <= len(objects) <= 9
    return a, bg, objects


def _lookup(objects):
    address = {(o['key'], o['body']): i for i, o in enumerate(objects)}
    return [address.get((o['key'], o['value'])) for o in objects]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all identities, rotation registrations and messages from the input."""
    a, bg, objects = _parse(input_grid)
    donors = _lookup(objects)
    out = np.full(a.shape, bg, dtype=np.int8)
    for obj, j in zip(objects, donors):
        if j is None:
            continue
        donor = objects[j]
        # Build in the common shape coordinates, then restore this receiver's pose.
        patch = np.full(obj['canonical'].shape, bg, dtype=np.int8)
        patch[obj['canonical']] = obj['body']
        patch[donor['registered']] = donor['value']
        patch = np.rot90(patch, -obj['turn'])
        ys, xs = np.where(obj['mask'])
        out[obj['top']+ys, obj['left']+xs] = patch[ys, xs]
    return out


def _shape(rng, forbidden, extent, chiral=False):
    def acceptable(mask):
        if min(mask.shape) < 2 or not 5 <= int(mask.sum()) <= 30:
            return False
        keys = {_key(np.rot90(mask, k)) for k in range(4)}
        key = min(keys)
        return (len(keys) == 4 and key not in forbidden and
                (not chiral or _canonical(np.fliplr(mask))[2] != key))

    for _ in range(160):
        h, w = map(int, rng.integers(2, extent+1, size=2))
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        target = int(rng.integers(5, min(24, h*w)+1)) if h*w >= 5 else 5
        directions = _D4 if rng.random() < .65 else _D8
        while len(cells) < target:
            candidates = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in directions
                                 if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            if not candidates:
                break
            cells.add(candidates[int(rng.integers(len(candidates)))])
        mask = np.zeros((h, w), dtype=bool)
        for p in cells:
            mask[p] = True
        mask = _crop(mask)
        if acceptable(mask):
            return _canonical(mask)[0]
    # A finite fallback preserves all semantic requirements, including chirality.
    for bits in range(1, 512):
        mask = _crop(np.array([(bits >> k) & 1 for k in range(9)], dtype=bool).reshape(3, 3))
        if acceptable(mask) and len(_parts(map(tuple, np.argwhere(mask)))) == 1:
            return _canonical(mask)[0]
    raise AssertionError('shape vocabulary exhausted')


def _objects(rng, bases, graphs, palette, distinct_dots=False):
    result = []
    for class_id, (base, relations) in enumerate(zip(bases, graphs)):
        points = list(map(tuple, np.argwhere(base)))
        order = rng.permutation(len(points))
        phase = int(rng.integers(4))
        for i, (body, value) in enumerate(relations):
            dot = points[int(order[i % len(points)])] if distinct_dots else points[int(rng.integers(len(points)))]
            result.append(dict(base=base, class_id=class_id, turn=(phase+i) % 4,
                               body=int(palette[body]), value=int(palette[value]), dot=dot))
    return result


def _random_graph(rng, count, first_body=None):
    bodies = list(map(int, rng.choice(9, count, replace=False)))
    if first_body is not None and first_body not in bodies:
        bodies[0] = first_body
    rel = []
    for body in bodies:
        choices = [b for b in bodies if b != body]
        if choices and rng.random() < .78:
            value = int(rng.choice(choices))
        else:
            value = int(rng.choice([v for v in range(9) if v != body]))
        rel.append((body, value))
    if count >= 2 and not any(v in bodies for b, v in rel):
        rel[0] = bodies[0], bodies[1]
    return rel


def _layout(rng, objects):
    masks = [np.rot90(o['base'], o['turn']) for o in objects]
    area = sum(int(m.sum()) for m in masks)
    padded_area = sum((m.shape[0]+1)*(m.shape[1]+1) for m in masks)
    options = [(h, w) for h in range(9, 31) for w in range(9, 31)
               if h*w > max(2*area, 1.2*padded_area)
               and h >= max(m.shape[0] for m in masks)+2
               and w >= max(m.shape[1] for m in masks)+2]
    areas = np.array([h*w for h, w in options], dtype=float)
    temperature = float(rng.choice([65, 160, 500]))
    weights = np.exp(-(areas-areas.min())/temperature)
    weights /= weights.sum()
    for attempt in range(24):
        shape = options[int(rng.choice(len(options), p=weights))] if attempt < 23 else (30, 30)
        blocked = np.zeros(shape, dtype=bool)
        positions = [None]*len(objects)
        order = sorted(range(len(objects)), key=lambda i: -int(masks[i].sum()))
        for i in order:
            mask = masks[i]
            ys, xs = np.where(mask)
            h, w = mask.shape
            possible = (shape[0]-h+1)*(shape[1]-w+1)
            for code in rng.permutation(possible):
                top, left = divmod(int(code), shape[1]-w+1)
                if np.any(blocked[top+ys, left+xs]):
                    continue
                positions[i] = top, left
                for r, c in zip(top+ys, left+xs):
                    blocked[max(0, r-1):min(shape[0], r+2), max(0, c-1):min(shape[1], c+2)] = True
                break
            if positions[i] is None:
                break
        if all(p is not None for p in positions):
            return shape, positions
    # At most nine <=6x6 pieces fit in separated 9x9 cells of a 30x30 canvas.
    slots = list(map(int, rng.permutation(9)))
    positions = []
    for i, mask in enumerate(masks):
        row, col = divmod(slots[i], 3)
        positions.append((row*10+int(rng.integers(10-mask.shape[0])),
                          col*10+int(rng.integers(10-mask.shape[1]))))
    return (30, 30), positions


def _render(rng, objects, bg):
    shape, positions = _layout(rng, objects)
    inp = np.full(shape, bg, dtype=np.int8)
    out = inp.copy()
    # Ground truth uses the constructed graph, independently of input parsing.
    address = {(o['class_id'], o['body']): o for o in objects}
    for obj, (top, left) in zip(objects, positions):
        patch = np.full(obj['base'].shape, bg, dtype=np.int8)
        patch[obj['base']] = obj['body']
        patch[obj['dot']] = obj['value']
        patch = np.rot90(patch, obj['turn'])
        mask = np.rot90(obj['base'], obj['turn'])
        ys, xs = np.where(mask)
        inp[top+ys, left+xs] = patch[ys, xs]
        donor = address.get((obj['class_id'], obj['value']))
        if donor is not None:
            answer = np.full(obj['base'].shape, bg, dtype=np.int8)
            answer[obj['base']] = obj['body']
            answer[donor['dot']] = donor['value']
            answer = np.rot90(answer, obj['turn'])
            out[top+ys, left+xs] = answer[ys, xs]
    return {'input': inp, 'output': out}


def _make_pair(rng, mode, extent, fixed_bg):
    bg = int(rng.integers(10)) if fixed_bg is None else fixed_bg
    palette = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    forbidden = set()

    def fresh(chiral=False):
        mask = _shape(rng, forbidden, extent, chiral)
        forbidden.add(_key(mask))
        return mask

    if mode == 'open':
        base = fresh(chiral=True)
        mirror = _canonical(np.fliplr(base))[0]
        bases = [base, mirror]
        graphs = [[(0, 1), (1, 2), (2, 3)], [(3, 0)]]
    elif mode == 'scope':
        bases = [fresh(), fresh()]
        graphs = [[(0, 1), (1, 2), (2, 1)], [(0, 1), (1, 0)]]
    elif mode == 'test':
        extensions = [[1], [2], [1, 1], [3], [2, 1]]
        sizes = [4] + extensions[int(rng.integers(len(extensions)))]
        bases = [fresh() for _ in sizes]
        chain = int(rng.integers(3))
        graphs = [[(0, 1), (1, 2), (2, 3), (3, 4)] if chain == 0 else
                  [(0, 1), (1, 2), (2, 3), (3, 0)] if chain == 1 else
                  [(0, 1), (1, 2), (2, 1), (3, 2)]]
        graphs += [_random_graph(rng, n, first_body=0) for n in sizes[1:]]
    else:
        profiles = [[2], [2, 1], [3, 1], [3, 2], [2, 2, 1], [3, 3], [2, 2, 2, 1], [3, 2, 2, 1]]
        sizes = profiles[int(rng.integers(len(profiles)))]
        bases = [fresh() for _ in sizes]
        graphs = [_random_graph(rng, n) for n in sizes]
    objects = _objects(rng, bases, graphs, palette, distinct_dots=mode in ('open', 'scope', 'test'))
    if mode == 'random':
        # Identical poses as well as relative quarter/half-turns are legitimate.
        for obj in objects:
            obj['turn'] = int(rng.integers(4))
    return _render(rng, objects, bg)


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name+' is outside its legal range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    ne = _count(num_test, 1, 4, 'num_test')
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    fixed_bg = int(rng.integers(10)) if rng.random() < .25 else None
    extent = int(rng.choice([3, 4, 5, 6], p=[.15, .35, .35, .15]))
    modes = ['open', 'scope'] + ['random']*(nt-2)
    rng.shuffle(modes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, sequence in (('train', modes), ('test', ['test']*ne)):
        for mode in sequence:
            for _ in range(16):
                pair = _make_pair(rng, mode, extent, fixed_bg)
                key = _fingerprint(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('distinct input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return sha256(bytes(a.shape)+a.tobytes()).hexdigest()


def _features(input_grid):
    a, bg, objects = _parse(input_grid)
    donors = _lookup(objects)
    groups = Counter(o['key'] for o in objects)
    use = Counter(j for j in donors if j is not None)
    f = dict(shape=a.shape, background=bg, pieces=len(objects), classes=len(groups),
             multiplicities=tuple(sorted(groups.values())), survivors=sum(j is not None for j in donors),
             areas=tuple(sorted(len(o['cells']) for o in objects)),
             extents=tuple(sorted(o['mask'].shape for o in objects)),
             diagonal=any(len(_parts(o['cells'], _D4)) > 1 for o in objects),
             no_A=False, no_B=False, reflection=False, rotation=False, point_transfer=False,
             simultaneous=False, deleted_donor=False, shared_donor=max(use.values(), default=0) > 1,
             solid=False, conflicting_bodies=False, same_request_different_answer=False)
    for i, (obj, j) in enumerate(zip(objects, donors)):
        if j is None:
            f['no_A'] |= any(other['body'] == obj['value'] for other in objects)
            f['no_B'] |= groups[obj['key']] > 1
            f['reflection'] |= any(other['body'] == obj['value'] and other['key'] != obj['key']
                                   and _canonical(np.fliplr(other['mask']))[2] == obj['key']
                                   for other in objects)
            continue
        donor = objects[j]
        f['rotation'] |= _key(obj['mask']) != _key(donor['mask'])
        f['point_transfer'] |= (obj['registered'] != donor['registered'] and donor['value'] != obj['body'])
        f['solid'] |= donor['value'] == obj['body']
        f['deleted_donor'] |= donors[j] is None
        f['simultaneous'] |= donors[j] is not None and objects[donors[j]]['value'] != donor['value']
        for k, other in enumerate(objects):
            if other['key'] != obj['key'] and other['body'] == obj['body']:
                f['conflicting_bodies'] |= other['value'] != obj['value']
                f['same_request_different_answer'] |= (other['value'] == obj['value'] and
                                                       donors[k] is not None and
                                                       objects[donors[k]]['value'] != donor['value'])
    return f


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {}
    for split in ('train', 'test'):
        features[split] = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen
            seen.add(key)
            assert np.array_equal(solve(inp), out), 'registered relay output mismatch'
            _, bg, objects = _parse(inp)
            donors = _lookup(objects)
            assert np.all(out[inp == bg] == bg), 'relay painted outside input silhouettes'
            for obj, j in zip(objects, donors):
                actual = [int(out[p]) for p in sorted(obj['cells'])]
                if j is None:
                    assert set(actual) == {bg}, 'unanswered stencil survived'
                else:
                    assert bg not in actual, 'successful stencil lost support'
                    donor = objects[j]
                    # One registered colored point, possibly invisible in a solid result.
                    assert Counter(actual) == Counter([obj['body']]*(len(actual)-1)+[donor['value']])
            f = _features(inp)
            assert 1 <= f['survivors'] <= f['pieces']
            assert not np.array_equal(inp, out)
            features[split].append(f)
    for name in ('no_A', 'no_B', 'reflection', 'rotation', 'point_transfer', 'simultaneous',
                 'deleted_donor', 'shared_donor', 'solid', 'conflicting_bodies', 'same_request_different_answer'):
        assert any(f[name] for f in features['train']), 'missing training witness: '+name
    assert all(max(f['multiplicities']) <= 3 for f in features['train'])
    assert all(4 in f['multiplicities'] and f['classes'] >= 2 for f in features['test'])


def self_test():
    """42 seeds, streaming measurements, count boundaries and negative checks."""
    digests, shapes, structures, areas, multiplicities, counts, backgrounds = (set() for _ in range(7))
    witnesses = Counter()
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_fingerprint(p[r]) for s in ('train', 'test') for p in puzzle[s]
                          for r in ('input', 'output'))
        digests.add(signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = np.asarray(pair['input'])
                f = _features(a)
                total += 1
                shapes.add(a.shape)
                areas.update(f['areas'])
                multiplicities.update(f['multiplicities'])
                counts.add(f['pieces'])
                backgrounds.add(f['background'])
                roles = {}
                normalized = np.array([roles.setdefault(int(c), len(roles)) for c in a.ravel()],
                                      dtype=np.uint8).reshape(a.shape)
                structures.add(_fingerprint(normalized))
                for name, value in f.items():
                    if isinstance(value, (bool, np.bool_)) and value:
                        witnesses[name] += 1
        if seed == 0:
            again = generate_puzzle(0)
            assert signature == tuple(_fingerprint(p[r]) for s in ('train', 'test') for p in again[s]
                                      for r in ('input', 'output'))
            sample = puzzle['train'][0]
            original = sample['input'].copy()
            # Global pose and literal-color changes must commute with the relay.
            for k in range(4):
                transformed = np.rot90(original, k)
                assert np.array_equal(solve(transformed), np.rot90(sample['output'], k))
            assert np.array_equal(solve(np.fliplr(original)), np.fliplr(sample['output']))
            permutation = np.random.default_rng(187).permutation(10)
            assert np.array_equal(solve(permutation[original]), permutation[sample['output']])
            assert np.array_equal(original, sample['input']), 'solver mutated its input'
            assert np.array_equal(solve(original.tolist()), sample['output'])
            bad = {s: [dict(input=p['input'].copy(), output=p['output'].copy()) for p in puzzle[s]]
                   for s in ('train', 'test')}
            bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('corrupted output accepted')
    # A larger, unseen class tests that solve does not depend on sampler profiles.
    extra_rng = np.random.default_rng(781)
    base = _shape(extra_rng, set(), 5)
    objects = _objects(extra_rng, [base], [[(i, (i+1) % 9) for i in range(9)]],
                       list(range(9)), distinct_dots=True)
    extended = _render(extra_rng, objects, 9)
    assert np.array_equal(solve(extended['input']), extended['output'])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(9000+10*nt+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    assert len(digests) == 42 and len(structures) >= 180
    assert len(shapes) >= 45 and len(areas) >= 10 and len(counts) >= 6
    assert {1, 2, 3, 4} <= multiplicities and backgrounds == set(range(10))
    assert all(witnesses[name] >= 42 for name in
               ('no_A', 'no_B', 'reflection', 'rotation', 'point_transfer', 'simultaneous',
                'deleted_donor', 'shared_donor', 'solid', 'conflicting_bodies', 'same_request_different_answer'))
    assert witnesses['diagonal'] > 0
    return dict(seeds=42, pairs=total, distinct_puzzles=len(digests), grid_shapes=len(shapes),
                normalized_structures=len(structures), areas=sorted(areas), piece_counts=sorted(counts),
                multiplicities=sorted(multiplicities), backgrounds=sorted(backgrounds), witnesses=dict(witnesses))
