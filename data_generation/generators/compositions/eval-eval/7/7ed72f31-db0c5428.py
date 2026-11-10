"""Self-programming compass: inward samples decode a geometric routing operator.

No parent imports, file access, training state, or RNG are used by solve().
Generation and scalar validation use separate coordinate constructions.
"""
from __future__ import annotations

import hashlib
import numpy as np

DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
SIGNS = {"H": (-1, 1), "V": (1, -1), "P": (-1, -1)}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _decode(code, carrier):
    cells = set(map(tuple, np.argwhere(code == carrier).tolist()))
    _require((1, 1) in cells, "missing center of instruction")
    if cells == {(1, 1)}:
        return "P"
    if cells <= {(1, 0), (1, 1), (1, 2)}:
        return "H"
    if cells <= {(0, 1), (1, 1), (2, 1)}:
        return "V"
    raise AssertionError("inward samples do not encode a line or singleton")


def _parse(input_grid):
    """Segment full rectangles and one isolated dot, then decode the visible samples."""
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    _require(int(counts.max()) > a.size // 2, "background must be a strict majority")
    bg = int(values[counts.argmax()])
    pieces = _components(a != bg)
    dots = [p for p in pieces if len(p) == 1]
    _require(len(dots) == 1, "expected one isolated central dot")
    hub = dots[0][0]
    carrier = int(a[hub])
    blocks = [p for p in pieces if len(p) != 1]
    _require(3 <= len(blocks) <= 8, "expected three to eight tiles")
    _require(1 <= hub[0] < a.shape[0] - 1 and 1 <= hub[1] < a.shape[1] - 1,
             "compass code must fit")
    code = np.full((3, 3), bg, dtype=a.dtype)
    code[1, 1] = carrier
    _require(np.count_nonzero(a[hub[0]-1:hub[0]+2, hub[1]-1:hub[1]+2] != bg) == 1,
             "compass initially contains only its central dot")
    tiles, seen, sizes = [], set(), set()
    carrier_count = total = 0
    for cells in blocks:
        coords = np.array(cells)
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        h, w = r1 - r0 + 1, c1 - c0 + 1
        _require(h in (3, 5) and w in (3, 5) and len(cells) == h * w,
                 "source tiles must be full odd rectangles of side three or five")
        sizes.add((h, w))
        center = (r0 + h // 2, c0 + w // 2)
        dy, dx = center[0] - hub[0], center[1] - hub[1]
        d = (int(np.sign(dy)), int(np.sign(dx)))
        _require(d in DIRECTIONS and d not in seen, "one source per compass direction")
        _require((not dy or abs(dy) >= h + 1) and (not dx or abs(dx) >= w + 1),
                 "source is too close to its hub")
        seen.add(d)
        t = a[r0:r1+1, c0:c1+1].copy()
        n = int(np.count_nonzero(t == carrier))
        _require(n > 0, "carrier must occur in each tile")
        carrier_count += n
        total += t.size
        sample = int(t[h // 2 - d[0] * (h // 2), w // 2 - d[1] * (w // 2)])
        code[1 + d[0], 1 + d[1]] = sample
        tiles.append(dict(direction=d, center=center, top=r0, left=c0, tile=t, sample=sample))
    _require(len(sizes) == 1, "tiles must share their dimensions within an example")
    _require(carrier_count > total // 2, "dot color must be the global tile carrier")
    _require(sum(t['sample'] != carrier for t in tiles) >= 2, "need two data directions")
    kind = _decode(code, carrier)
    sy, sx = SIGNS[kind]
    reserved = set()
    for item in tiles:
        cy, cx = item['center']
        h, w = item['tile'].shape
        ty, tx = hub[0] + 2 * sy * (cy - hub[0]), hub[1] + 2 * sx * (cx - hub[1])
        for r in range(ty - h // 2, ty + h // 2 + 1):
            for c in range(tx - w // 2, tx + w // 2 + 1):
                _require(0 <= r < a.shape[0] and 0 <= c < a.shape[1], "copy leaves the canvas")
                _require(a[r, c] == bg and (r, c) not in reserved, "copy collision")
                _require(max(abs(r-hub[0]), abs(c-hub[1])) > 1, "copy collides with compass")
                reserved.add((r, c))
    return dict(background=bg, carrier=carrier, hub=hub, tiles=tiles,
                code=code, kind=kind, size=next(iter(sizes)), destinations=reserved)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read, decode, and execute the compass using only the supplied input."""
    a = _grid(input_grid)
    info = _parse(a)
    out = a.copy()
    r, c = info['hub']
    out[r-1:r+2, c-1:c+2] = info['code']
    sy, sx = SIGNS[info['kind']]
    for item in info['tiles']:
        cy, cx = item['center']
        tile = item['tile'][::sy, ::sx]
        h, w = tile.shape
        y, x = r + 2 * sy * (cy - r), c + 2 * sx * (cx - c)
        out[y-h//2:y+h//2+1, x-w//2:x+w//2+1] = tile
    return out


def _four_views(tile):
    return (tile, tile[::-1, :], tile[:, ::-1], tile[::-1, ::-1])


def _data_direction(kind, direction):
    return not ((kind == 'H' and direction[0] == 0) or
                (kind == 'V' and direction[1] == 0))


def _make_tile(rng, size, carrier, sample, details, direction, unique=False, repeated=False):
    h, w = size
    inward = (h // 2 - direction[0] * (h // 2), w // 2 - direction[1] * (w // 2))
    tile = np.full(size, carrier, dtype=np.int8)
    tile[inward] = sample
    positions = [q for q in np.ndindex(size) if q != inward]
    rng.shuffle(positions)
    # Marks, stripes and connected patches provide different internal geometries.
    budget = int(rng.integers(0, max(1, h*w//2-1)))
    style = int(rng.integers(3))
    if style == 1:
        axis = int(rng.integers(2))
        fixed = int(rng.integers(size[axis]))
        positions.sort(key=lambda q: q[axis] != fixed)
    elif style == 2:
        anchor = positions[0]
        positions.sort(key=lambda q: abs(q[0]-anchor[0])+abs(q[1]-anchor[1]))
    palette = [x for x in details if not unique or x != sample]
    for q in positions[:budget]:
        tile[q] = int(rng.choice(palette))
    if unique:
        # A unique inward value identifies the spatial sample, while an off-axis
        # auxiliary mark distinguishes all four axis-preserving isometries.
        candidates = positions.copy()
        rng.shuffle(candidates)
        auxiliary = next(x for x in details if x != sample)
        for q in candidates:
            trial = tile.copy()
            trial[q] = auxiliary
            if np.count_nonzero(trial == carrier) > trial.size // 2 and len(
                    {v.tobytes() for v in _four_views(trial)}) == 4:
                tile = trial
                break
        else:
            for q in candidates:
                trial = np.full(size, carrier, dtype=np.int8)
                trial[inward] = sample
                trial[q] = auxiliary
                if len({v.tobytes() for v in _four_views(trial)}) == 4:
                    tile = trial
                    break
            else:
                raise AssertionError('no orientation witness')
    if repeated:
        _require(not unique and sample != carrier, 'incompatible tile witnesses')
        tile[positions[0]] = sample
    _require(np.count_nonzero(tile == carrier) > tile.size // 2, 'tile lost carrier clarity')
    return tile


def _arms(rng, kind):
    if kind == 'P':
        return set()
    options = [(0, -1), (0, 1)] if kind == 'H' else [(-1, 0), (1, 0)]
    return set(options) if rng.random() < .5 else {options[int(rng.integers(2))]}


def _pair(rng, kind, required, unique_dirs, repeat_dir, count, size, reach,
          palette_preference):
    controls = _arms(rng, kind)
    directions = set(required) | controls
    count = max(count, len(directions), len(controls) + 2)
    candidates = [d for d in DIRECTIONS if d not in directions]
    rng.shuffle(candidates)
    directions.update(candidates[:count-len(directions)])
    directions = sorted(directions)
    data_dirs = [d for d in directions if d not in controls]
    if repeat_dir is None:
        choices = [d for d in data_dirs if d not in unique_dirs]
        if choices and rng.random() < .65:
            repeat_dir = choices[int(rng.integers(len(choices)))]
    # Each example has an orientation witness outside any repeated-sample tile.
    orientation = next(d for d in data_dirs if d != repeat_dir)
    unique_dirs = set(unique_dirs) | {orientation}
    if rng.random() < .45:
        bg, carrier = palette_preference
    else:
        bg, carrier = map(int, rng.choice(10, size=2, replace=False))
    details = [int(x) for x in rng.permutation([v for v in range(10) if v not in (bg, carrier)])]
    details = details[:int(rng.integers(2, 9))]
    samples = {d: (carrier if d in controls else int(rng.choice(details))) for d in directions}
    samples[data_dirs[0]], samples[data_dirs[1]] = details[0], details[1]
    h, w = size
    bases = []
    for extent in size:
        if extent == 5:
            bases.append(6)
        elif reach == 'far':
            bases.append(5)
        elif reach == 'near':
            bases.append(4)
        else:
            bases.append(int(rng.integers(4, 6)))
    objects = []
    for d in directions:
        offsets = [0 if not sign else sign * (base + (int(rng.integers(2)) if extent == 3 else 0))
                   for sign, base, extent in zip(d, bases, size)]
        tile = _make_tile(rng, size, carrier, samples[d], details, d,
                          unique=d in unique_dirs, repeated=d == repeat_dir)
        objects.append(dict(direction=d, offset=offsets, tile=tile))
    if reach == 'far':
        obj = objects[int(rng.integers(len(objects)))]
        axes = [j for j in (0, 1) if obj['offset'][j] and size[j] == 3]
        j = axes[int(rng.integers(len(axes)))]
        obj['offset'][j] = int(np.sign(obj['offset'][j])) * 6
    sy, sx = SIGNS[kind]
    lo, hi = [-1, -1], [1, 1]
    for obj in objects:
        for center in (obj['offset'], (2*sy*obj['offset'][0], 2*sx*obj['offset'][1])):
            for j, extent in enumerate(size):
                lo[j] = min(lo[j], center[j] - extent//2)
                hi[j] = max(hi[j], center[j] + extent//2)
    spans = [hi[j] - lo[j] + 1 for j in (0, 1)]
    _require(max(spans) <= 30, 'construction exceeds bounds')
    padding = [int(rng.integers(31 - span)) for span in spans]
    before = [int(rng.integers(p + 1)) for p in padding]
    hub = tuple(before[j] - lo[j] for j in (0, 1))
    a = np.full(tuple(spans[j]+padding[j] for j in (0, 1)), bg, dtype=np.int8)
    a[hub] = carrier
    for obj in objects:
        y, x = (hub[j] + obj['offset'][j] for j in (0, 1))
        a[y-h//2:y+h//2+1, x-w//2:x+w//2+1] = obj['tile']
    # Generation uses its explicit construction plan, not solve().
    b = a.copy()
    for obj in objects:
        dr, dc = obj['direction']
        b[hub[0]+dr, hub[1]+dc] = samples[(dr, dc)]
        y, x = hub[0] + 2*sy*obj['offset'][0], hub[1] + 2*sx*obj['offset'][1]
        b[y-h//2:y+h//2+1, x-w//2:x+w//2+1] = obj['tile'][::sy, ::sx]
    _require(_parse(a)['kind'] == kind, 'sampling did not encode planned operator')
    return dict(input=a, output=b)


def _legal_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _legal_count(num_train, 'num_train', 2, 10)
    m = _legal_count(num_test, 'num_test', 1, 4)
    if n <= m:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    pools = [('H', 'V'), ('H', 'P'), ('V', 'P')]
    pool = pools[int(rng.integers(len(pools)))]
    if n >= 3 and rng.random() < .4:
        pool = ('H', 'V', 'P')
    kinds = list(pool) + [str(rng.choice(pool)) for _ in range(n-len(pool))]
    rng.shuffle(kinds)
    required, witnesses = [set() for _ in range(n)], [set() for _ in range(n)]
    # Every direction has a non-control, unique-position training witness.
    order = list(DIRECTIONS)
    rng.shuffle(order)
    for d in order:
        choices = [i for i, k in enumerate(kinds) if _data_direction(k, d)]
        smallest = min(len(required[i]) for i in choices)
        choices = [i for i in choices if len(required[i]) == smallest]
        i = choices[int(rng.integers(len(choices)))]
        required[i].add(d)
        witnesses[i].add(d)
    # A repeated detail sample in training refutes unique-color selection.
    choices = [(i, d) for i, k in enumerate(kinds) for d in DIRECTIONS
               if _data_direction(k, d) and d not in witnesses[i]]
    i, d = choices[int(rng.integers(len(choices)))]
    required[i].add(d)
    repeated = [None] * n
    repeated[i] = d
    regime = str(rng.choice(['reach', 'sparse', 'factor']))
    preference = tuple(map(int, rng.choice(10, size=2, replace=False)))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', n), ('test', m)):
        for index in range(count):
            test = split == 'test'
            kind = str(rng.choice(pool)) if test else kinds[index]
            req = set() if test else required[index]
            uni = set() if test else witnesses[index]
            rep = None if test else repeated[index]
            if regime == 'reach':
                size = (3, 3)
                reach = 'far' if test else 'near'
                desired = int(rng.integers(3, 9))
            elif regime == 'sparse':
                size = (int(rng.choice([3, 5])), int(rng.choice([3, 5])))
                reach = 'any'
                desired = int(rng.integers(3, 6) if test else rng.integers(6, 9))
            else:
                size = (5, 5) if test else ((3, 5) if index % 2 == 0 else (5, 3))
                reach = 'any'
                desired = int(rng.integers(3, 9))
            for _ in range(16):
                pair = _pair(rng, kind, req, uni, rep, desired, size, reach, preference)
                fingerprint = _key(pair['input'])
                if fingerprint not in seen:
                    break
            else:
                raise AssertionError('bounded distinct-input attempts exhausted')
            seen.add(fingerprint)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
    a, b = _grid(pair['input']), _grid(pair['output'])
    _require(a.shape == b.shape, 'output dimensions changed')
    info = _parse(a)
    r, c = info['hub']
    sy, sx = SIGNS[info['kind']]
    expected = a.copy()
    assigned = set()
    for item in info['tiles']:
        dr, dc = item['direction']
        # Check source-facing coordinate directly in the full input array.
        h, w = item['tile'].shape
        cy, cx = item['center']
        sr, sc = cy - dr*(h//2), cx - dc*(w//2)
        expected[r+dr, c+dc] = a[sr, sc]
        assigned.add((r+dr, c+dc))
        # Scalar cell mapping differs from the slice-based generation and solve.
        for y in range(item['top'], item['top'] + h):
            for x in range(item['left'], item['left'] + w):
                q = (r + 2*sy*(cy-r) + sy*(y-cy), c + 2*sx*(cx-c) + sx*(x-cx))
                _require(q not in assigned and a[q] == info['background'], 'invalid destination')
                assigned.add(q)
                expected[q] = a[y, x]
    _require(np.array_equal(expected, b), 'incorrect compass or routed tile')
    _require(np.array_equal(a[a != info['background']], b[a != info['background']]),
             'original foreground changed')
    _require(np.count_nonzero(a != b) == len(info['tiles']) * (np.prod(info['size']) + 1),
             'incorrect number of additions')
    _require(set(map(int, np.unique(a))) == set(map(int, np.unique(b))), 'palette changed')
    return info


def _training_evidence(infos):
    # Candidate positions are the nine corners/edge midpoints/center of a tile.
    # Keep a separate candidate set per direction; this is stronger than one
    # global inward-vs-outward comparison and works across 3x5/5x3 rectangles.
    candidates = {d: set(np.ndindex(3, 3)) for d in DIRECTIONS}
    seen_dirs, orientation, repeat = set(), set(), False
    for info in infos:
        carrier = info['carrier']
        for item in info['tiles']:
            d, tile, sample = item['direction'], item['tile'], item['sample']
            h, w = tile.shape
            seen_dirs.add(d)
            candidates[d] &= {(y, x) for y, x in np.ndindex(3, 3)
                              if tile[y*(h//2), x*(w//2)] == sample}
            if sample != carrier and np.count_nonzero(tile == sample) > 1:
                repeat = True
            if len({v.tobytes() for v in _four_views(tile)}) == 4:
                orientation.add(info['kind'])
    return candidates, seen_dirs, orientation, repeat


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, 'invalid pair counts')
    infos, seen = {}, set()
    for split in ('train', 'test'):
        infos[split] = []
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(np.asarray(pair['input']))
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            infos[split].append(info)
    candidates, directions, orientation, repeat = _training_evidence(infos['train'])
    _require(directions == set(DIRECTIONS), 'training lacks compass-direction coverage')
    for d in DIRECTIONS:
        _require(candidates[d] == {(1-d[0], 1-d[1])}, 'training conceals a sampling-coordinate ambiguity')
    kinds = {i['kind'] for i in infos['train']}
    _require(len(kinds) >= 2 and bool(kinds & {'H', 'V'}), 'training must show content-dependent control')
    _require(kinds <= orientation, 'training hides tile orientation behind symmetry')
    _require(repeat, 'training must show positional sampling despite repeated colors')
    for info in infos['test']:
        _require(info['kind'] in kinds, 'query requests an undemonstrated operation')
    return None


def measure_puzzle(puzzle):
    """Small observable summary; no scene payloads or execution traces retained."""
    result = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = np.asarray(pair['input'])
            info = _parse(a)
            r, c = info['hub']
            labels = {}
            normalized = bytes(labels.setdefault(int(v), len(labels)) for v in a.flat)
            result.append(dict(split=split, shape=list(a.shape), size=list(info['size']),
                               count=len(info['tiles']), kind=info['kind'], hub=[r, c],
                               carrier=info['carrier'], background=info['background'],
                               arms=int(np.count_nonzero(info['code'] == info['carrier']))-1,
                               reach=max(max(abs(t['center'][0]-r), abs(t['center'][1]-c))
                                         for t in info['tiles']),
                               directions=tuple(t['direction'] for t in info['tiles']),
                               repeated=sum(t['sample'] != info['carrier'] and
                                            np.count_nonzero(t['tile'] == t['sample']) > 1
                                            for t in info['tiles']),
                               structure=hashlib.sha256(bytes(a.shape)+normalized).hexdigest()))
    return result


def self_test():
    # At most one puzzle plus one bounded corruption copy is live at a time.
    summaries, signatures, shapes, normalized = [], set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        rows = measure_puzzle(p)
        summaries.extend(rows)
        signatures.add(tuple(_key(np.asarray(x['input'])) for s in ('train', 'test') for x in p[s]))
        shapes.update(tuple(x['shape']) for x in rows)
        normalized.update(x['structure'] for x in rows)
        for split in ('train', 'test'):
            for pair in p[split]:
                a, b = pair['input'], pair['output']
                saved = a.copy()
                _require(np.array_equal(solve(a), b), 'solve disagrees')
                _require(np.array_equal(saved, a), 'solve mutated its argument')
                # Semantic color equivariance, including black as an active color.
                permutation = np.random.default_rng(seed + 700).permutation(10)
                _require(np.array_equal(solve(permutation[a]), permutation[b]), 'literal-color dependence')
        train, tests = [r for r in rows if r['split'] == 'train'], [r for r in rows if r['split'] == 'test']
        shift = (all(r['reach'] <= 5 and r['size'] == [3, 3] for r in train)
                 and all(r['reach'] == 6 for r in tests)) or (
                     min(r['count'] for r in train) >= 6 and max(r['count'] for r in tests) <= 5) or (
                     {tuple(r['size']) for r in train} == {(3, 5), (5, 3)}
                     and all(r['size'] == [5, 5] for r in tests))
        _require(shift, 'meaningful query shift absent')
    _require(len(signatures) == 42, 'seeds collapsed')
    _require(len(shapes) >= 35 and len(normalized) >= 180, 'insufficient structural variation')
    _require({tuple(r['size']) for r in summaries} == {(3, 3), (3, 5), (5, 3), (5, 5)}, 'tile sizes missing')
    _require({r['count'] for r in summaries} == set(range(3, 9)), 'tile counts missing')
    _require({r['kind'] for r in summaries} == {'H', 'V', 'P'}, 'operators missing')
    _require({r['arms'] for r in summaries} == {0, 1, 2}, 'marker extents missing')
    _require(len({r['directions'] for r in summaries}) >= 35, 'constellation subsets collapsed')
    _require({r['carrier'] for r in summaries} == set(range(10)), 'carrier palette gap')
    _require({r['background'] for r in summaries} == set(range(10)), 'background palette gap')
    # Test all legal counts, strict type handling, and deterministic generation.
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(917+n*11+m, n, m)
            _require(len(p['train']) == n and len(p['test']) == m, 'counts ignored')
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ('input', 'output')), 'not reproducible')
    # Corrupt the compass, a routed copy, a source, and untouched background.
    for location in ('code', 'copy', 'source', 'unused'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in z.items()} for z in p[s]] for s in p}
        pair = bad['test'][0]
        info = _parse(pair['input'])
        r, c = info['hub']
        if location == 'code':
            point = (r, c)
        elif location == 'copy':
            point = min(info['destinations'])
        elif location == 'source':
            point = info['tiles'][0]['center']
        else:
            assigned = info['destinations'] | {(r+dr, c+dc) for dr, dc in DIRECTIONS} | {(r, c)}
            point = next(q for q in np.ndindex(pair['input'].shape)
                         if pair['input'][q] == info['background'] and q not in assigned)
        pair['output'][point] = (int(pair['output'][point])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted label accepted')
    return dict(seeds=42, pairs=len(summaries), distinct_puzzles=len(signatures),
                grid_shapes=len(shapes), normalized_structures=len(normalized),
                kinds=sorted({r['kind'] for r in summaries}),
                counts=sorted({r['count'] for r in summaries}))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
