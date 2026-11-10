"""Folded broadcasts: template-relative ray addresses become square depths.

Only NumPy and the standard library are used. solve reads pixels only. The
constructor builds oriented blocks; the decoder works with object coordinates.
"""
from __future__ import annotations

import numpy as np


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _components(a, bg):
    seen, result = set(), []
    h, w = a.shape
    for r, c in np.argwhere(a != bg):
        p = int(r), int(c)
        if p in seen:
            continue
        seen.add(p)
        pending, group = [p], []
        while pending:
            r, c = pending.pop()
            group.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= q[0] < h and 0 <= q[1] < w and q not in seen
                        and a[q] != bg):
                    seen.add(q)
                    pending.append(q)
        result.append(group)
    return result


def _ray(e, side, slot, length, shape):
    p = e['coords'][side * length + slot]
    q = e['coords'][(1-side) * length + slot]
    normal = p[0]-q[0], p[1]-q[1]
    r, c = p[0]+normal[0], p[1]+normal[1]
    cells = []
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        cells.append((r, c))
        r, c = r+normal[0], c+normal[1]
    return cells


def _square(e, side, length):
    p = e['coords'][side * length]
    q = e['coords'][(1-side) * length]
    next_p = e['coords'][side * length + 1]
    normal = p[0]-q[0], p[1]-q[1]
    tangent = next_p[0]-p[0], next_p[1]-p[1]
    n = 2 * length
    return [((p[0]+(i+1)*normal[0]+j*tangent[0],
              p[1]+(i+1)*normal[1]+j*tangent[1]),
             min(i, j, n-1-i, n-1-j))
            for i in range(n) for j in range(n)]


def _decode(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    if np.sum(counts == counts.max()) != 1:
        raise ValueError("Background must be the unique most frequent input color")
    bg = int(values[np.argmax(counts)])
    h, w = a.shape
    emitters, signals = [], []
    for group in _components(a, bg):
        rs, cs = zip(*group)
        r, c = min(rs), min(cs)
        block = a[r:max(rs)+1, c:max(cs)+1]
        if (min(block.shape) == 2 and 3 <= max(block.shape) <= 6
                and block.size == len(group)):
            emitters.append({'r': r, 'c': c, 'arr': block.copy(),
                             'cells': set(group)})
        else:
            for p in group:
                if not (p[0] in (0, h-1) or p[1] in (0, w-1)):
                    raise ValueError("Non-emitter content must be boundary observations")
                if p[0] in (0, h-1) and p[1] in (0, w-1):
                    raise ValueError("Corner clues are not permitted")
                signals.append((p, int(a[p])))
    if not 2 <= len(emitters) <= 6 or not signals:
        raise ValueError("Expected 2..6 copies and at least one boundary observation")
    template = emitters[0]['arr'].copy()
    if template.shape[0] != 2:
        template = np.rot90(template)
    length = template.shape[1]
    equal = np.flatnonzero(template[0] == template[1])
    if len(equal) != 1 or equal[0] not in (0, length-1):
        raise ValueError("A unique monochrome end cap must indicate the depth origin")
    if equal[0] == length-1:
        template = np.fliplr(template)
    cap = int(template[0, 0])
    if np.count_nonzero(template == cap) != 2:
        raise ValueError("End-cap color must occur only in the cap")
    structural = set(map(int, np.unique(template)))
    if not 3 <= len(structural) <= 4 or bg in structural:
        raise ValueError("Expected 3..4 structural colors distinct from background")
    if any(color in structural for _, color in signals):
        raise ValueError("Signal and structural palettes must be disjoint")
    forms = [_orient(template, t) for t in range(8)]
    if len({_key(b) for b in forms}) != 8:
        raise ValueError("Template must establish unique correspondence")
    labels = np.arange(2*length).reshape(2, length)
    for e in emitters:
        matches = [t for t in range(8) if np.array_equal(e['arr'], forms[t])]
        if len(matches) != 1:
            raise ValueError("All emitters must be exact transformed copies")
        e['t'] = matches[0]
        loc = _orient(labels, matches[0])
        e['coords'] = {int(loc[r, c]): (e['r']+r, e['c']+c)
                       for r, c in np.ndindex(loc.shape)}
    occupied = set().union(*(e['cells'] for e in emitters))
    corridors = {}
    for index, e in enumerate(emitters):
        for side in (0, 1):
            for slot in range(length):
                ray = _ray(e, side, slot, length, a.shape)
                if (ray and not occupied.intersection(ray)
                        and all(a[p] == bg for p in ray[:-1])):
                    corridors.setdefault(ray[-1], []).append((index, side, slot))
    code, sides = {}, set()
    allocations = [set() for _ in emitters]
    observations = []
    for p, color in signals:
        matches = corridors.get(p, [])
        if len(matches) != 1:
            raise ValueError("Every clue must have a unique clear normal projection")
        index, side, slot = matches[0]
        if slot in code and code[slot] != color:
            raise ValueError("Redundant observations disagree")
        code[slot] = color
        sides.add(side)
        allocations[index].add(slot)
        observations.append((index, slot, p, color))
    if len(sides) != 1:
        raise ValueError("Observations must specify one shared template-relative side")
    side = sides.pop()
    squares, used = [], set()
    signal_cells = {p for p, _ in signals}
    for e in emitters:
        square = _square(e, side, length)
        cells = {p for p, _ in square}
        if (not all(0 <= r < h and 0 <= c < w for r, c in cells)
                or cells.intersection(occupied | signal_cells | used)):
            raise ValueError("Projected squares must fit and avoid objects, clues and each other")
        if any(a[p] != bg for p in cells):
            raise ValueError("Projected squares must initially be empty")
        used.update(cells)
        squares.append(square)
    return {'array': a, 'background': bg, 'template': template, 'length': length,
            'emitters': emitters, 'side': side, 'code': code, 'squares': squares,
            'allocations': allocations, 'observations': observations}


def _paint(info, codes=None):
    out = info['array'].copy()
    if codes is None:
        codes = [info['code']] * len(info['emitters'])
    for square, code in zip(info['squares'], codes):
        for p, depth in square:
            if depth in code:
                out[p] = code[depth]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer correspondences, shared side and sparse code, then fold into frames."""
    return _paint(_decode(input_grid))


def _template(rng, length, colors):
    # Cap is the only uniform transverse pair. Its color never occurs elsewhere.
    body = colors[1:]
    a = np.empty((2, length), dtype=np.int8)
    a[:, 0] = colors[0]
    style = int(rng.integers(3))
    for slot in range(1, length):
        if style == 0:
            x, y = rng.choice(body, 2, replace=False)
        elif style == 1:
            x, y = body[0], body[1+(slot % (len(body)-1))]
        else:
            x, y = body[slot % len(body)], body[(slot+1) % len(body)]
        a[:, slot] = x, y
    # Ensure a four-color palette really has four colors, without creating a cap.
    if len(colors) == 4 and colors[-1] not in a:
        a[1, -1] = colors[-1]
        if a[0, -1] == a[1, -1]:
            a[0, -1] = body[0]
    return a


def _candidate(template, t, r, c):
    length = template.shape[1]
    labels = np.full((2+2*length, 2*length), -1, dtype=int)
    labels[:2, :length] = np.arange(2*length).reshape(2, length)
    square_mask = np.zeros(labels.shape, dtype=bool)
    square_mask[2:] = True
    labels, square_mask = _orient(labels, t), _orient(square_mask, t)
    coords = {int(labels[i, j]): (r+i, c+j)
              for i, j in np.argwhere(labels >= 0)}
    return {'coords': coords, 'cells': set(coords.values()),
            'square_cells': {(r+i, c+j) for i, j in np.argwhere(square_mask)},
            't': t, 'r': r, 'c': c, 'shape': labels.shape}


def _layout(rng, template, ts, shape):
    placed = []
    for t in ts:
        bh, bw = _orient(np.empty((2+2*template.shape[1], 2*template.shape[1])), t).shape
        options = [(r, c) for r in range(1, shape[0]-bh)
                   for c in range(1, shape[1]-bw)]
        if not options:
            return None
        rng.shuffle(options)
        for r, c in options[:350]:
            # One empty row or column separates full virtual block rectangles.
            if all(r+bh < e['r'] or e['r']+e['shape'][0] < r
                   or c+bw < e['c'] or e['c']+e['shape'][1] < c for e in placed):
                placed.append(_candidate(template, t, r, c))
                break
        else:
            return None
    return placed


def _clues(rng, placed, length, slots, shape, mode):
    occupied = set().union(*(e['cells'] for e in placed))
    squares = set().union(*(e['square_cells'] for e in placed))
    alignments = {}
    for index, e in enumerate(placed):
        for side in (0, 1):
            for slot in range(length):
                ray = _ray(e, side, slot, length, shape)
                if ray and not occupied.intersection(ray):
                    alignments.setdefault(ray[-1], []).append((index, side, slot))
    options = {}
    for slot in slots:
        options[slot] = []
        for index, e in enumerate(placed):
            ray = _ray(e, 1, slot, length, shape)
            p = ray[-1]
            if alignments.get(p) != [(index, 1, slot)] or p in squares:
                continue
            if p[0] in (0, shape[0]-1) and p[1] in (0, shape[1]-1):
                continue
            if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1 for q in occupied):
                continue
            options[slot].append((index, slot, p))
        if not options[slot]:
            return None
    if mode == 'full':
        common = set.intersection(*({x[0] for x in options[s]} for s in slots))
        if not common:
            return None
        index = int(rng.choice(sorted(common)))
        result = [next(x for x in options[s] if x[0] == index) for s in slots]
    else:
        result = [choices[int(rng.integers(len(choices)))] for choices in options.values()]
        if mode == 'split' and len({x[0] for x in result}) == 1:
            old = result[0][0]
            alternatives = [(i, x) for i, s in enumerate(slots)
                            for x in options[s] if x[0] != old]
            if not alternatives:
                return None
            i, replacement = alternatives[int(rng.integers(len(alternatives)))]
            result[i] = replacement
    if mode == 'redundant':
        extra = [x for choices in options.values() for x in choices if x not in result]
        rng.shuffle(extra)
        result += extra[:int(rng.integers(1, min(3, len(extra))+1))] if extra else []
    return result


def _wrapping(code, length, bg):
    # Constructor uses center-out padding, distinct from solver's coordinate map.
    a = np.full((2, 2), code.get(length-1, bg), dtype=np.int8)
    for depth in range(length-2, -1, -1):
        a = np.pad(a, 1, constant_values=code.get(depth, bg))
    return a


def _make_pair(rng, length, count, demonstration=None, sparse=False):
    palette = list(map(int, rng.permutation(10)))
    bg = palette[0]
    k = int(rng.integers(3, 5))
    template = _template(rng, length, palette[1:1+k])
    if demonstration == 'pool':
        slot_count = int(rng.integers(2, length))
        inner = sorted(map(int, rng.choice(range(1, length-1), slot_count-2, replace=False)))
        slots = [0] + inner + [length-1]
    else:
        slot_count = int(rng.integers(1 if not demonstration else 2,
                                      length if sparse else length+1))
        slots = sorted(map(int, rng.choice(length, slot_count, replace=False)))
    available = palette[1+k:]
    richness = int(rng.integers(1 if not demonstration else 2,
                               min(len(slots), len(available))+1))
    colors = list(map(int, rng.choice(available, richness, replace=False)))
    values = colors + list(map(int, rng.choice(colors, len(slots)-richness)))
    rng.shuffle(values)
    if demonstration == 'pool' and values[0] == values[-1]:
        pos = next(i for i, v in enumerate(values) if v != values[0])
        values[pos], values[-1] = values[-1], values[pos]
    code = dict(zip(slots, values))
    base = int(rng.integers(8))
    if demonstration == 'pool':
        ts = [base, base ^ 4]
        mode = 'split'
    elif demonstration == 'rotate':
        ts = [base, 4*(base//4)+(base+1) % 4]
        mode = 'full'
    else:
        orientation_mode = int(rng.integers(3))
        if orientation_mode == 0:
            ts = [base] * count
        elif orientation_mode == 1:
            ts = [base if i % 2 == 0 else base ^ 4 for i in range(count)]
        else:
            ts = list(map(int, rng.integers(8, size=count)))
        mode = str(rng.choice(['full', 'split', 'spread', 'redundant']))
        if len(slots) == 1 and mode == 'split':
            mode = 'spread'
    # Random packing of invisible footprints; no fixed rows of displayed panels.
    placed = clues = None
    for attempt in range(160):
        low = 2*length+4
        if attempt < 90:
            shape = tuple(map(int, rng.integers(low, 31, size=2)))
            if shape[0]*shape[1] < count*(2*length+2)*(2*length)*1.25:
                continue
        else:
            shape = (30, 30)
        rng.shuffle(ts)
        placed = _layout(rng, template, ts, shape)
        if placed is not None:
            clues = _clues(rng, placed, length, slots, shape, mode)
            if clues is not None:
                break
    else:
        return None
    inp = np.full(shape, bg, dtype=np.int8)
    out = inp.copy()
    frames = _wrapping(code, length, bg)
    tile = np.full((2+2*length, 2*length), bg, dtype=np.int8)
    tile[:2, :length] = template
    filled = tile.copy()
    filled[2:] = frames
    for e in placed:
        r, c = e['r'], e['c']
        bh, bw = e['shape']
        inp[r:r+bh, c:c+bw] = _orient(tile, e['t'])
        out[r:r+bh, c:c+bw] = _orient(filled, e['t'])
    for index, slot, p in clues:
        inp[p] = out[p] = code[slot]
    return {'input': inp, 'output': out}


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10),
                                ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # Shift affects the visible structures only, never the transformation.
    regime = int(rng.integers(4))  # longer, shorter, more copies, denser code
    train_lengths = list(range(3, 6)) if regime == 0 else list(range(4, 7)) if regime == 1 else list(range(3, 7))
    first_lengths = list(map(int, rng.choice(train_lengths, 2, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            demo = ('pool', 'rotate')[index] if split == 'train' and index < 2 else None
            for attempt in range(100):
                if split == 'train':
                    length = first_lengths[index] if index < 2 else int(rng.choice(train_lengths))
                    copies = 2 if demo or regime == 2 else int(rng.integers(2, min(6, 18//length)+1))
                else:
                    length = 6 if regime == 0 else 3 if regime == 1 else int(rng.integers(3, 5)) if regime == 2 else int(rng.integers(3, 7))
                    copies = int(rng.integers(4, min(6, 18//length)+1)) if regime == 2 else int(rng.integers(2, min(6, 18//length)+1))
                # Avoid dense packing combinations unsupported by 30x30.
                copies = min(copies, {3: 6, 4: 5, 5: 4, 6: 2}[length])
                pair = _make_pair(rng, length, copies, demo,
                                  sparse=split == 'train' and regime == 3)
                if pair is None:
                    continue
                if split == 'test' and regime == 3:
                    # Generate a complete slot code with the same ordinary constructor.
                    info = _decode(pair['input'])
                    if len(info['code']) != length:
                        continue
                key = _key(pair['input'])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("Bounded packing exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternatives(info):
    local = [dict() for _ in info['emitters']]
    for index, slot, _, color in info['observations']:
        local[index][slot] = color
    local_answer = _paint(info, local)
    rays = info['array'].copy()
    occupied = set().union(*(x['cells'] for x in info['emitters']))
    for e in info['emitters']:
        for slot, color in info['code'].items():
            for p in _ray(e, info['side'], slot, info['length'], rays.shape):
                if p not in occupied:
                    rays[p] = color
    reversed_code = {info['length']-1-s: c for s, c in info['code'].items()}
    compressed_code = dict(enumerate(info['code'][s] for s in sorted(info['code'])))
    return local_answer, rays, _paint(info, [reversed_code]*len(local)), _paint(info, [compressed_code]*len(local))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            info = _decode(inp)
            assert np.array_equal(_paint(info), out), "Incorrect folded broadcast"
            # Every copied frame has the same literal color at the same depth;
            # transparent layers, remote clues and all bar cells stay unchanged.
            region = set()
            for square in info['squares']:
                for p, depth in square:
                    assert int(out[p]) == info['code'].get(depth, info['background'])
                    region.add(p)
            for p in np.ndindex(inp.shape):
                if p not in region:
                    assert out[p] == inp[p]
            assert not np.array_equal(inp, out)
            if split == 'train':
                training.append(info)
    assert len({i['length'] for i in training}) >= 2
    assert any(all(len(a) < len(i['code']) for a in i['allocations'])
               and 0 in i['code'] and i['length']-1 in i['code']
               and len(i['code']) < i['length'] for i in training), "Missing sparse pooled demonstration"
    assert any(any(len(a) == len(i['code']) for a in i['allocations'])
               and any(not a for a in i['allocations']) for i in training), "Missing complete-code transfer"
    assert any(len({e['arr'].shape[0] == 2 for e in i['emitters']}) == 2
               for i in training), "Missing quarter-turn transfer"
    assert any(len({e['t']//4 for e in i['emitters']}) == 2
               for i in training), "Missing reflection transfer"
    # Concrete countermodels must be ruled out by demonstrations, even with 2/1.
    alternatives = [_alternatives(i) for i in training]
    for alternative in range(4):
        assert any(not np.array_equal(_paint(i), predictions[alternative])
                   for i, predictions in zip(training, alternatives)), "Training admits a tested shortcut"


def self_test():
    """Bounded pixel-derived coverage, count boundaries and mutation checks."""
    signatures, shapes, lengths, counts, slot_counts, templates = set(), set(), set(), set(), set(), set()
    colors, transforms, code_patterns = set(), set(), set()
    full = sparse = repeated = redundant = parallel = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = tuple(_key(x['input']) for x in p['train']+p['test'])
        signatures.add(signature)
        for pair in p['train']+p['test']:
            info = _decode(pair['input'])
            assert np.array_equal(solve(pair['input']), pair['output'])
            shapes.add(pair['input'].shape)
            lengths.add(info['length'])
            counts.add(len(info['emitters']))
            slot_counts.add(len(info['code']))
            roles = {}
            templates.add(tuple(roles.setdefault(int(v), len(roles)) for v in info['template'].flat))
            code_patterns.add((info['length'], tuple(sorted(info['code']))))
            colors.update(map(int, np.unique(pair['input'])))
            transforms.update(e['t'] for e in info['emitters'])
            full += len(info['code']) == info['length']
            sparse += len(info['code']) < info['length']
            repeated += len(set(info['code'].values())) < len(info['code'])
            redundant += len(info['observations']) > len(info['code'])
            parallel += len({e['t'] for e in info['emitters']}) < len(info['emitters'])
    assert len(signatures) == 42 and len(shapes) >= 35
    assert lengths == {3, 4, 5, 6} and counts == set(range(2, 7))
    assert slot_counts == set(range(1, 7)) and len(code_patterns) >= 25
    assert len(templates) >= 12 and colors == set(range(10)) and transforms == set(range(8))
    assert min(full, sparse, repeated, redundant, parallel) >= 4
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(10000+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p, q = generate_puzzle(12345, 3, 2), generate_puzzle(12345, 3, 2)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    # Input-only inference is equivariant under every whole-grid isometry and
    # arbitrary role recoloring. This also exercises caps facing every edge.
    palette = np.array([6, 2, 9, 1, 5, 0, 8, 4, 7, 3], dtype=np.int8)
    for t in range(8):
        pair = p['test'][0]
        inp = _orient(palette[pair['input']], t)
        out = _orient(palette[pair['output']], t)
        assert np.array_equal(solve(inp), out)
    # Handwritten pixels exercise a square touching the bottom boundary, an
    # unhinted copy and adjacent same-color clues. No construction helper is used.
    fixture = np.zeros((12, 22), dtype=np.int8)
    bar = np.array([[1, 2, 2], [1, 3, 3]], dtype=np.int8)
    fixture[1:3, 1:4] = bar
    fixture[4:6, 13:16] = bar
    for middle in (False, True):
        inp = fixture.copy()
        inp[11, 1] = 4
        inp[11, 3] = 5
        if middle:
            inp[11, 2] = 4
        expected = inp.copy()
        block = np.zeros((6, 6), dtype=np.int8)
        block[[0, -1], :] = 4
        block[:, [0, -1]] = 4
        if middle:
            block[1:5, 1:5] = 4
        block[2:4, 2:4] = 5
        expected[3:9, 1:7] = block
        expected[6:12, 13:19] = block
        assert np.array_equal(solve(inp), expected)
    inconsistent = inp.copy()
    inconsistent[0, 1] = 4  # Same address, opposite template-relative side.
    try:
        solve(inconsistent)
    except ValueError:
        pass
    else:
        raise AssertionError("Inconsistent active sides accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    info = _decode(bad['test'][0]['input'])
    pos = next(pos for pos, depth in info['squares'][0] if depth in info['code'])
    bad['test'][0]['output'][pos] = info['background']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("A damaged ring was accepted")
    return {'seeds': 42, 'puzzles': len(signatures), 'shapes': len(shapes),
            'lengths': sorted(lengths), 'copy_counts': sorted(counts),
            'slot_counts': sorted(slot_counts), 'templates': len(templates),
            'sparse': sparse, 'full': full, 'repeated_colors': repeated,
            'redundant_observations': redundant, 'parallel': parallel}
