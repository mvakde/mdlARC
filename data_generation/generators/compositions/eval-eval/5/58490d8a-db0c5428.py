"""Finite colored stock shared clockwise by eight inward-keyed radial ports.

Only standard-library modules and NumPy are used. The solver reads every role,
count, cursor and motif from the input; it does not access generator state.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np


# Clockwise, starting north. These indices describe space, not literal colors.
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1))
DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
SINGLE = ((0, 0),)
GLYPHS = (DIAMOND, SQUARE, SINGLE,
          ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
          ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
          ((0, 0), (0, 1), (0, 2)),
          ((0, 0), (1, 1), (2, 2)),
          ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "Grid must be two-dimensional with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "Colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _components(mask, diagonal=True):
    todo = set(map(tuple, np.argwhere(mask)))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r)+abs(c) == 1)]
    result = []
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, component = [p], []
        while stack:
            r, c = stack.pop()
            component.append((int(r), int(c)))
            for dr, dc in steps:
                q = (r+dr, c+dc)
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(component)
    return result


def _allocation(samples, counts, cursor, clockwise=True):
    """Round-robin independently for each requested color, conserving stock."""
    allocation = [0]*8
    step = 1 if clockwise else -1
    traversal = [(cursor+step*j) % 8 for j in range(8)]
    for color in set(samples):
        ports = [i for i in traversal if samples[i] == color]
        q, remainder = divmod(counts.get(color, 0), len(ports))
        for j, i in enumerate(ports):
            allocation[i] = q + int(j < remainder)
    return tuple(allocation)


def _analyze(input_grid):
    a = _grid(input_grid)
    frequencies = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    _require(frequencies[bg] > a.size/2, "Background must be a strict majority")
    candidates = []
    for carrier in range(10):
        if carrier == bg:
            continue
        points = np.argwhere(a == carrier)
        if len(points) <= 36:
            continue
        lo, hi = points.min(axis=0), points.max(axis=0)
        if tuple(hi-lo) != (8, 8):
            continue
        r, c = map(int, lo+4)
        if not (5 <= r <= a.shape[0]-6 and 5 <= c <= a.shape[1]-6):
            continue
        hub = a[r-1:r+2, c-1:c+2]
        marks = np.argwhere(hub != bg)
        if len(marks) != 1 or tuple(marks[0]) == (1, 1):
            continue
        offset = tuple(map(int, marks[0]-1))
        cursor = DIRECTIONS.index(offset)
        marker = int(hub[tuple(marks[0])])
        if marker == carrier or np.count_nonzero(a == marker) != 1:
            continue
        tiles, samples = [], []
        valid = True
        for dr, dc in DIRECTIONS:
            t = a[r+3*dr-1:r+3*dr+2, c+3*dc-1:c+3*dc+2]
            if t.shape != (3, 3) or np.any(t == bg) or np.any(t == marker):
                valid = False
                break
            sample = int(t[1-dr, 1-dc])
            if sample == carrier or not np.any(t == carrier):
                valid = False
                break
            tiles.append(t.copy())
            samples.append(sample)
        if not valid:
            continue
        halo = np.ones((11, 11), dtype=bool)
        halo[1:10, 1:10] = False
        if not np.all(a[r-5:r+6, c-5:c+6][halo] == bg):
            continue
        candidates.append(dict(background=bg, carrier=carrier, marker=marker,
                               center=(r, c), cursor=cursor,
                               tiles=tuple(tiles), samples=tuple(samples)))
    _require(len(candidates) == 1, "There must be one visibly identifiable source ring")
    m = candidates[0]
    r, c = m['center']
    outside = np.ones(a.shape, dtype=bool)
    outside[r-4:r+5, c-4:c+5] = False
    objects, counts = {}, {}
    owner = np.full(a.shape, -1, dtype=int)
    serial = 0
    for color in set(map(int, a[outside])) - {bg}:
        _require(color not in (m['carrier'], m['marker']), "Reserved role in inventory")
        components = _components((a == color) & outside)
        objects[color] = tuple(_normalize(part) for part in components)
        counts[color] = len(components)
        for part in components:
            for rr, cc in part:
                owner[rr, cc] = serial
            serial += 1
    for rr, cc in np.argwhere(owner >= 0):
        neighbors = owner[max(0, rr-1):rr+2, max(0, cc-1):cc+2]
        _require(bool(np.all((neighbors < 0) | (neighbors == owner[rr, cc]))),
                 "Different inventory objects must have a diagonal-inclusive gap")
    _require(set(m['samples']) <= set(counts), "Each requested color needs positive stock")
    _require(1 <= len(set(m['samples'])) <= 6, "One through six requested colors")
    _require(sum(counts[k] for k in set(m['samples'])) <= 16, "Too much requested stock")
    for k in set(m['samples']):
        _require(counts[k] <= 2*m['samples'].count(k), "A port may receive at most two packets")
    allocation = _allocation(m['samples'], counts, m['cursor'])
    for i, n in enumerate(allocation):
        dr, dc = DIRECTIONS[i]
        for j in range(n):
            distance = 6+4*j
            rr, cc = r+distance*dr, c+distance*dc
            _require(1 <= rr < a.shape[0]-1 and 1 <= cc < a.shape[1]-1,
                     "An emitted tile would be clipped")
    m.update(outside=outside, objects=objects, counts=counts, allocation=allocation,
             ignored=tuple(sorted(set(counts)-set(m['samples']))))
    return m


def _build_output(a, center, bg, carrier, tiles, samples, allocation):
    # Inventory is consumed/ignored; preserve only the original ring first.
    r, c = center
    b = np.full(a.shape, bg, dtype=np.int8)
    b[r-4:r+5, c-4:c+5] = a[r-4:r+5, c-4:c+5]
    b[r-1:r+2, c-1:c+2] = bg
    b[r, c] = carrier
    for i, (dr, dc) in enumerate(DIRECTIONS):
        if allocation[i]:
            b[r+dr, c+dc] = samples[i]
        for j in range(allocation[i]):
            distance = 6+4*j
            rr, cc = r+distance*dr, c+distance*dc
            b[rr-1:rr+2, cc-1:cc+2] = tiles[i][::-1, ::-1]
    return b


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from one visible input, without training or a seed."""
    a = _grid(input_grid)
    m = _analyze(a)
    return _build_output(a, m['center'], m['background'], m['carrier'],
                         m['tiles'], m['samples'], m['allocation'])


def _d4(tile):
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _tile(rng, carrier, sample, inward, details, strong=False):
    t = np.full((3, 3), carrier, dtype=np.int8)
    t[inward] = sample
    positions = [p for p in np.ndindex(3, 3) if p != inward]
    rng.shuffle(positions)
    if strong:
        other = int(rng.choice([v for v in details if v != sample]))
        for p in positions:
            trial = t.copy()
            trial[p] = other
            if len({view.tobytes() for view in _d4(trial)}) == 8:
                return trial
        raise AssertionError("No asymmetric tile witness")
    for p in positions[:int(rng.integers(4))]:
        t[p] = int(rng.choice(details))
        # Keep the carrier's visible bounding box stable, even in rare draws
        # where three decorations would cover an entire outside tile edge.
        if any(np.all(edge != carrier) for edge in (t[0], t[-1], t[:, 0], t[:, -1])):
            t[p] = carrier
    return t


def _shape(rng):
    if rng.random() < .55:
        cells = GLYPHS[int(rng.integers(len(GLYPHS)))]
    else:
        # Procedural, eight-connected masks; no color-specific shape vocabulary.
        cells = {(1, 1)}
        target = int(rng.integers(1, 10))
        while len(cells) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < 3 and 0 <= c+dc < 3} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.random() < .5:
        cells = [(r, -c) for r, c in cells]
    for _ in range(int(rng.integers(4))):
        cells = [(-c, r) for r, c in cells]
    return _normalize(cells)


def _place(a, blocked, color, cells, rng):
    sh = 1+max(r for r, c in cells)
    sw = 1+max(c for r, c in cells)
    nh, nw = a.shape[0]-sh+1, a.shape[1]-sw+1
    permitted = np.ones((nh, nw), dtype=bool)
    for r, c in cells:
        permitted &= ~blocked[r:r+nh, c:c+nw]
    places = np.argwhere(permitted)
    if not len(places):
        return False
    r0, c0 = places[int(rng.integers(len(places)))]
    for dr, dc in cells:
        r, c = int(r0+dr), int(c0+dc)
        a[r, c] = color
        blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return True


def _partition(rng, k):
    labels = list(range(k)) + list(map(int, rng.integers(k, size=8-k)))
    rng.shuffle(labels)
    return labels


def _make_pair(rng, cursor, mode, force_key=None):
    order = list(map(int, rng.permutation(10)))
    if force_key is not None:
        order.remove(force_key)
        order.insert(3, force_key)
    bg, carrier, marker = order[:3]
    k = 3 if mode == 'count_witness' else int(rng.integers(2, 6)) if mode == 'geometry_witness' else int(rng.integers(1, 7))
    keys = order[3:3+k]
    aux = order[3+k:]
    if mode == 'count_witness':
        # A two-port pool of three objects exposes the remainder, rotation of
        # the cursor, and clockwise versus counterclockwise arbitration.
        for _ in range(100):
            positions = list(map(int, rng.permutation([i for i in range(8) if i != cursor])))
            labels = [2]*8
            for i in positions[:2]:
                labels[i] = 0
            for i in [cursor]+positions[2:4]:
                labels[i] = 1
            samples = tuple(keys[j] for j in labels)
            counts = dict(zip(keys, (3, 1, 4)))
            right = _allocation(samples, counts, cursor)
            if right != _allocation(samples, counts, cursor, False) and right != _allocation(samples, counts, 0):
                break
        else:
            raise AssertionError("Allocation witness construction failed")
    else:
        labels = _partition(rng, k)
        samples = tuple(keys[j] for j in labels)
        counts = {color: int(rng.integers(samples.count(color) if mode == 'geometry_witness' else 1,
                                        2*samples.count(color)+1)) for color in keys}
        if mode == 'geometry_witness':
            # A genuinely asymmetric second packet demonstrates that every
            # packet uses the same half-turn, rather than alternating turns.
            color = int(rng.choice(keys))
            counts[color] = 2*samples.count(color)
        if mode == 'query' and all(counts[color] % samples.count(color) == 0 for color in keys):
            # With at most six colors over eight ports a shared pool exists.
            color = int(rng.choice([v for v in keys if samples.count(v) > 1]))
            m = samples.count(color)
            counts[color] = int(rng.choice([n for n in range(1, 2*m+1) if n % m]))
    allocation = _allocation(samples, counts, cursor)
    details = keys + aux[:int(rng.integers(1, min(3, len(aux))+1))]
    tiles = tuple(_tile(rng, carrier, samples[i], (1-dr, 1-dc), details,
                        mode == 'geometry_witness') for i, (dr, dc) in enumerate(DIRECTIONS))
    if mode == 'count_witness':
        # A stock color occurring as decoration still does not request stock.
        # The center of a source tile is never its inward-facing sample.
        tiles[0][1, 1] = aux[0]
    # Counts determine labels before object packing; parsing cannot silently
    # relabel a merged, missing or split object to hide a placement error.
    object_list = []
    for color in keys:
        prototype = _shape(rng)
        for j in range(counts[color]):
            cells = prototype if rng.random() < .25 else _shape(rng)
            if mode == 'count_witness' and color == keys[0]:
                cells = (DIAMOND, SQUARE, SINGLE)[j]
            object_list.append((color, cells))
    distractor_colors = aux[:int(rng.integers(0, min(2, len(aux))+1))]
    if mode == 'count_witness':
        distractor_colors = aux[:1]
    for color in distractor_colors:
        number = 3 if mode == 'count_witness' else int(rng.integers(1, 4))
        for j in range(number):
            object_list.append((color, _shape(rng)))
    rng.shuffle(object_list)
    object_list.sort(key=lambda item: -len(item[1]))
    # Actual allocated rays determine each required extent independently.
    north = south = west = east = 5  # one background-cell halo around source
    for i, (dr, dc) in enumerate(DIRECTIONS):
        extent = 3+4*allocation[i] if allocation[i] else 5
        if dr < 0:
            north = max(north, extent)
        if dr > 0:
            south = max(south, extent)
        if dc < 0:
            west = max(west, extent)
        if dc > 0:
            east = max(east, extent)
    for attempt in range(7):
        h = 30 if attempt >= 4 else int(rng.integers(max(15, north+south+1), 31))
        w = 30 if attempt >= 4 else int(rng.integers(max(15, west+east+1), 31))
        r = int(rng.integers(north, h-south))
        c = int(rng.integers(west, w-east))
        a = np.full((h, w), bg, dtype=np.int8)
        for i, (dr, dc) in enumerate(DIRECTIONS):
            rr, cc = r+3*dr, c+3*dc
            a[rr-1:rr+2, cc-1:cc+2] = tiles[i]
        dr, dc = DIRECTIONS[cursor]
        a[r+dr, c+dc] = marker
        blocked = np.zeros(a.shape, dtype=bool)
        blocked[r-5:r+6, c-5:c+6] = True
        if attempt == 6:
            # At least 33 separated slots survive an 11x11 source halo on
            # 30x30; there are at most 22 inventory objects. No motif is changed.
            slots = [(rr, cc) for rr in range(0, 28, 4) for cc in range(0, 28, 4)
                     if not blocked[rr:rr+3, cc:cc+3].any()]
            _require(len(slots) >= len(object_list), "Fallback packing capacity")
            rng.shuffle(slots)
            for (color, cells), (rr, cc) in zip(object_list, slots):
                for y, x in cells:
                    a[rr+y, cc+x] = color
        elif not all(_place(a, blocked, color, cells, rng) for color, cells in object_list):
            continue
        if np.count_nonzero(a == bg) <= a.size/2:
            continue
        b = _build_output(a, (r, c), bg, carrier, tiles, samples, allocation)
        m = _analyze(a)
        _require(all(m['counts'][color] == counts[color] for color in keys), "Packing changed counts")
        return dict(input=a, output=b), aux[0]
    raise AssertionError("Bounded packing failed")


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10),
                                  (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    anchor = int(rng.integers(8))
    train_cursors = [(anchor+j) % 8 for j in range(3)]
    test_cursors = [i for i in range(8) if i not in train_cursors]
    first_cursor = int(rng.choice([i for i in train_cursors if i != 0]))
    puzzle, seen = dict(train=[], test=[]), set()
    role_change = None
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            mode = ('count_witness' if index == 0 else 'geometry_witness' if index == 1 else 'ordinary') if split == 'train' else 'query'
            cursor = first_cursor if mode == 'count_witness' else int(rng.choice(train_cursors if split == 'train' else test_cursors))
            for _ in range(20):
                pair, ignored = _make_pair(rng, cursor, mode,
                                           role_change if mode == 'geometry_witness' else None)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise AssertionError("Unable to obtain a distinct input")
            if mode == 'count_witness':
                role_change = ignored
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _scalar_check(a, b, m):
    """Coordinate oracle for the validator, separate from slice output assembly."""
    _require(a.shape == b.shape, "Output must keep the input canvas")
    r, c = m['center']
    bg, carrier = m['background'], m['carrier']
    target = np.full(a.shape, bg, dtype=np.int8)
    assigned = np.zeros(a.shape, dtype=bool)
    for i, (dr, dc) in enumerate(DIRECTIONS):
        for y in (-1, 0, 1):
            for x in (-1, 0, 1):
                rr, cc = r+3*dr+y, c+3*dc+x
                target[rr, cc] = a[rr, cc]
                assigned[rr, cc] = True
        for j in range(m['allocation'][i]):
            distance = 6+4*j
            for y in (-1, 0, 1):
                for x in (-1, 0, 1):
                    rr, cc = r+distance*dr-y, c+distance*dc-x
                    _require(not assigned[rr, cc], "Emission footprints must not overlap")
                    target[rr, cc] = a[r+3*dr+y, c+3*dc+x]
                    assigned[rr, cc] = True
        if m['allocation'][i]:
            target[r+dr, c+dc] = a[r+2*dr, c+2*dc]
    target[r, c] = carrier
    _require(np.array_equal(b, target), "Wrong stock allocation, hub, copy, preservation or erasure")
    _require(np.count_nonzero(b != bg) == 73 + sum(n > 0 for n in m['allocation']) + 9*sum(m['allocation']),
             "Packet area or hub occupancy is wrong")
    _require(not np.any(b == m['marker']), "Cursor must be consumed")
    for color in set(m['samples']):
        _require(sum(m['allocation'][i] for i in range(8) if m['samples'][i] == color) == m['counts'][color],
                 "Requested inventory must be conserved")


def _geometry_candidates(train):
    positions = [set(range(9)) for _ in range(8)]
    transforms = [set(range(8)) for _ in range(8)]
    for pair, m in train:
        r, c = m['center']
        b = np.asarray(pair['output'])
        for i, (dr, dc) in enumerate(DIRECTIONS):
            if not m['allocation'][i]:
                continue
            value = int(b[r+dr, c+dc])
            positions[i] &= {j for j in range(9) if m['tiles'][i].flat[j] == value}
            tile = b[r+6*dr-1:r+6*dr+2, c+6*dc-1:c+6*dc+2]
            transforms[i] &= {j for j, t in enumerate(_d4(m['tiles'][i])) if np.array_equal(t, tile)}
    return positions, transforms


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Wrong puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "Splits must be lists")
    try:
        _counts(len(puzzle['train']), len(puzzle['test']))
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    seen, meta = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Wrong pair schema")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(_key(a) not in seen, "Duplicate input")
            seen.add(_key(a))
            m = _analyze(a)
            _scalar_check(a, b, m)
            meta[split].append((pair, m))
    training = [m for pair, m in meta['train']]
    # Exact behavioral witnesses, measured from returned cells, not sampler tags.
    _require(any(DIAMOND in m['objects'][color] and SQUARE in m['objects'][color] and SINGLE in m['objects'][color]
                 for m in training for color in set(m['samples'])), "Missing connectivity/area counting witness")
    _require(any(m['ignored'] for m in training), "Missing unkeyed inventory demonstration")
    _require(any(set(a['ignored']) & set(b['samples']) for a in training for b in training),
             "Key membership must change locally between demonstrations")
    _require(any(0 in m['allocation'] and 2 in m['allocation'] for m in training),
             "Training must show exhausted ports and repeated packets")
    _require(any(n == 2 and not np.array_equal(t, t[::-1, ::-1])
                 for m in training for n, t in zip(m['allocation'], m['tiles'])),
             "Repeating the same half-turn needs an asymmetric second packet")
    _require(any(m['allocation'] != _allocation(m['samples'], m['counts'], m['cursor'], False) for m in training),
             "Clockwise versus counterclockwise allocation is ambiguous")
    _require(any(m['allocation'] != _allocation(m['samples'], m['counts'], 0) for m in training),
             "Ignoring the cursor is not contradicted")
    _require(any(any(m['allocation'][i] != m['counts'][color] for i, color in enumerate(m['samples'])) for m in training),
             "Independent per-port counting is not contradicted")
    positions, transforms = _geometry_candidates(meta['train'])
    _require(all(positions) and all(transforms), "Contradictory tile sampling/rotation demonstrations")
    train_cursors = {m['cursor'] for m in training}
    for pair, m in meta['test']:
        _require(m['cursor'] not in train_cursors, "Missing held-out cursor shift")
        _require(any(m['counts'][v] % m['samples'].count(v) for v in set(m['samples'])),
                 "A query must use cursor-sensitive remainder allocation")
        for i, tile in enumerate(m['tiles']):
            if m['allocation'][i]:
                _require(len({int(tile.flat[j]) for j in positions[i]}) == 1,
                         "Query sampling position lacks evidence")
                _require(len({_d4(tile)[j].tobytes() for j in transforms[i]}) == 1,
                         "Query rotation lacks evidence")
        _require(np.array_equal(solve(np.asarray(pair['input']).copy()), pair['output']), "Input-only solve disagrees")


def _digest(puzzle):
    raw = json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
                      for s in ('train', 'test')}, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def self_test():
    """Bounded 42-seed checks, observables, counterfactuals, and corruption tests."""
    signatures, shapes, layouts, patterns, object_shapes, motifs = set(), set(), set(), set(), set(), set()
    backgrounds, carriers, markers, cursors, key_counts, stocks, allocations = (set() for _ in range(7))
    ignored_sizes, areas = set(), set()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed == 0:
            first = puzzle
        signatures.add(_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                m = _analyze(a)
                shapes.add(a.shape)
                layouts.add((a.shape, m['center']))
                backgrounds.add(m['background'])
                carriers.add(m['carrier'])
                markers.add(m['marker'])
                cursors.add(m['cursor'])
                key_counts.add(len(set(m['samples'])))
                stocks.update(m['counts'][v] for v in set(m['samples']))
                allocations.add(m['allocation'])
                ignored_sizes.add(len(m['ignored']))
                labels = {}
                patterns.add(tuple(labels.setdefault(v, len(labels)) for v in m['samples']))
                for values in m['objects'].values():
                    object_shapes.update(values)
                    areas.update(map(len, values))
                for tile in m['tiles']:
                    labels = {m['carrier']: 0}
                    motifs.add(tuple(labels.setdefault(int(v), len(labels)) for v in tile.flat))
                preserved = a.copy()
                result = solve(a)
                _require(np.array_equal(a, preserved), "Solver mutated its input")
                _require(np.array_equal(result, pair['output']), "Self-test prediction mismatch")
                # Changing only the visible cursor must use the same inventories.
                r, c = m['center']
                for new_cursor in ((m['cursor']+1) % 8, (m['cursor']+4) % 8):
                    dr, dc = DIRECTIONS[new_cursor]
                    # A changed allocation can reach a previously unneeded edge;
                    # pad to 30x30 with a centered ring to avoid artificial clipping.
                    padded = np.full((30, 30), m['background'], dtype=np.int8)
                    for i, (dy, dx) in enumerate(DIRECTIONS):
                        padded[14+3*dy-1:14+3*dy+2, 14+3*dx-1:14+3*dx+2] = m['tiles'][i]
                    padded[14+dr, 14+dc] = m['marker']
                    # Reuse stock as separated singleton tokens in deterministic slots.
                    slots = [(rr, cc) for rr in range(0, 30, 2) for cc in range(0, 30, 2)
                             if not (8 <= rr <= 20 and 8 <= cc <= 20)]
                    n = 0
                    for color, count in sorted(m['counts'].items()):
                        for _ in range(count):
                            rr, cc = slots[n]
                            padded[rr, cc] = color
                            n += 1
                    pm = _analyze(padded)
                    _require(pm['allocation'] == _allocation(m['samples'], m['counts'], new_cursor),
                             "Cursor counterfactual broke conservation")
                    _scalar_check(padded, solve(padded), pm)
    _require(len(signatures) == 42, "Seeds collapsed to identical puzzles")
    _require(len(shapes) >= 30 and len(layouts) >= 100, "Canvas/layout diversity is too small")
    _require(len(patterns) >= 50 and len(allocations) >= 70, "Pool partitions or allocations collapsed")
    _require(len(object_shapes) >= 50 and len(motifs) >= 50, "Structural variation is too small")
    _require(backgrounds == carriers == markers == set(range(10)), "Palette role omitted")
    _require(cursors == set(range(8)) and key_counts == set(range(1, 7)), "Direction or key-count gap")
    _require({1, 2, 3, 4, 5, 6} <= stocks and ignored_sizes == {0, 1, 2}, "Stock/distractor coverage gap")
    _require(areas == set(range(1, 10)), "Object-size coverage gap")
    _require(_digest(first) == _digest(generate_puzzle(0)), "Nondeterministic puzzle")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(91, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "Explicit count mismatch")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("Illegal pair counts accepted")
    for site in ('hub', 'ring', 'packet', 'unused'):
        bad = copy.deepcopy(first)
        pair = bad['test'][0]
        m = _analyze(pair['input'])
        r, c = m['center']
        i = next(i for i, n in enumerate(m['allocation']) if n)
        dr, dc = DIRECTIONS[i]
        where = {'hub': (r, c), 'ring': (r-4, c-4),
                 'packet': (r+6*dr, c+6*dc), 'unused': (0, 0)}[site]
        pair['output'][where] = (int(pair['output'][where])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("Corrupted output accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), canvas_shapes=len(shapes),
                layouts=len(layouts), port_partitions=len(patterns), allocations=len(allocations),
                object_shapes=len(object_shapes), normalized_tile_motifs=len(motifs),
                semantic_verification='generator-owned checks; no independent solver or blind prediction')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
