"""Turning tile relay: input-only gating, vacancy transport, and half-turns.

Standard library + NumPy. No filesystem, parent imports, caches, or saved traces.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _rotate(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return int(r), int(c)


def _direction(p, q):
    r, c = q[0] - p[0], q[1] - p[1]
    if r == 0 and c:
        return 1 if c > 0 else 3
    if c == 0 and r:
        return 2 if r > 0 else 0
    return None


def _distance(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def _square(p, half):
    return np.s_[p[0]-half:p[0]+half+1, p[1]-half:p[1]+half+1]


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _components(cells):
    cells, result = set(cells), []
    while cells:
        group = {cells.pop()}
        todo = list(group)
        while todo:
            r, c = todo.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r+dr, c+dc)
                    if q in cells:
                        cells.remove(q)
                        group.add(q)
                        todo.append(q)
        result.append(group)
    return result


def _identify_arrow(cells):
    matches = []
    for tip in sorted(cells):
        for d in range(4):
            mask = {_rotate((r-tip[0], c-tip[1]), 1-d) for r, c in cells}
            if any(c > 0 for r, c in mask):
                continue
            if {p for p in mask if p[1] == 0} != {(0, 0)}:
                continue
            if not any(r for r, c in mask):
                continue
            if {(-r, c) for r, c in mask} == mask:
                matches.append((tip, d, tuple(sorted(mask))))
    _require(len(matches) == 1, "arrow must have one geometric tip and facing")
    return matches[0]


def _arrow_cells(mask, tip, facing):
    return {(tip[0]+r, tip[1]+c) for r, c in
            (_rotate(p, facing-1) for p in mask)}


def _analyze(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) * 2 > a.size, "background must be a strict majority")
    components = _components((int(r), int(c)) for r, c in zip(*np.where(a != bg)))
    tiles, arrows, sizes = {}, [], set()
    for cells in components:
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        r1, c1 = max(r for r, c in cells), max(c for r, c in cells)
        s = r1-r0+1
        if s in (3, 5) and c1-c0+1 == s and len(cells) == s*s:
            p = (int((r0+r1)//2), int((c0+c1)//2))
            tiles[p] = a[r0:r1+1, c0:c1+1].copy()
            sizes.add(s)
        else:
            arrows.append(cells)
    _require(len(arrows) == 1 and tiles and len(sizes) == 1, "one arrow and equal square tiles required")
    s = sizes.pop()
    h = s//2
    cells = arrows[0]
    arrow_colors = {int(a[r, c]) for r, c in cells}
    _require(len(arrow_colors) == 1, "arrow must be monochrome")
    ink = arrow_colors.pop()
    tip, facing, mask = _identify_arrow(cells)
    _require(all(abs(r) <= h and abs(c) <= h for r, c in mask), "arrow exceeds socket")
    carriers = []
    for t in tiles.values():
        cs, ns = np.unique(t, return_counts=True)
        _require(int(ns.max()) * 2 > s*s, "tile carrier must be a strict majority")
        carriers.append(int(cs[np.argmax(ns)]))
    _require(len(set(carriers)) == 1 and carriers[0] not in (bg, ink), "common distinct carrier required")
    centers = [tip] + list(tiles)
    for i, p in enumerate(centers):
        _require(h <= p[0] < a.shape[0]-h and h <= p[1] < a.shape[1]-h, "socket outside canvas")
        for q in centers[i+1:]:
            _require(max(abs(p[0]-q[0]), abs(p[1]-q[1])) > s, "square sockets must be separated")
    initial = np.full((s, s), bg, dtype=a.dtype)
    for r, c in _arrow_cells(mask, (h, h), facing):
        initial[r, c] = ink
    _require(np.array_equal(a[_square(tip, h)], initial), "arrow socket has extra content")
    return dict(grid=a, bg=bg, ink=ink, carrier=carriers[0], size=s,
                half=h, tip=tip, facing=facing, mask=mask, tiles=tiles)


def _trace(info, sample=None, ungated=False, limit=None):
    """Read only original, unconsumed tiles; transported tiles are consumed."""
    if sample is None:
        sample = (0, -info['half'])
    remaining = set(info['tiles'])
    points, directions = [info['tip']], []
    while remaining and (limit is None or len(directions) < limit):
        p = points[-1]
        eligible = {}
        for q in sorted(remaining):
            d = _direction(p, q)
            if d is None:
                continue
            r, c = _rotate(sample, d-1)
            t = info['tiles'][q]
            if ungated or int(t[info['half']+r, info['half']+c]) == info['ink']:
                eligible.setdefault(d, []).append(q)
        if not eligible:
            break
        if ungated:
            # An explicit simple alternative for ablation, with no hidden tie.
            options = [(_distance(p, q), d, q) for d, qs in eligible.items() for q in qs]
            _, d, q = min(options)
        else:
            _require(len(eligible) == 1, "more than one qualifying ray")
            d, qs = next(iter(eligible.items()))
            q = min(qs, key=lambda x: _distance(p, x))
        if not directions:
            _require(d == info['facing'], "first hop must follow the arrow")
        remaining.remove(q)
        points.append(q)
        directions.append(d)
    return points, directions


def _view(tile, operation):
    # 0..3: rotations; 4..7: reflected rotations. Correct operation is 2.
    return np.rot90(tile if operation < 4 else tile[:, ::-1], operation % 4)


def _render(info, points, directions, operation=2):
    a = info['grid']
    b = a.copy()
    h = info['half']
    for p in points:
        b[_square(p, h)] = info['bg']
    for p, q in zip(points, points[1:]):
        b[_square(p, h)] = _view(info['tiles'][q], operation)
    facing = directions[-1] if directions else info['facing']
    for r, c in _arrow_cells(info['mask'], points[-1], facing):
        b[r, c] = info['ink']
    return b


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from input cells. Does not mutate the input."""
    info = _analyze(input_grid)
    points, directions = _trace(info)
    return _render(info, points, directions)


def _mask_bank(size):
    h = size//2
    masks = set()
    for span in range(1, h+1):
        for extension in range(1, h+1):
            for solid in (False, True):
                for tail in range(h+1):
                    depth = extension-1+span+tail
                    if depth > h:
                        continue
                    cells = {(0, -i) for i in range(extension)}
                    for k in range(1, span+1):
                        rows = range(-k, k+1) if solid else (-k, k)
                        cells.update((r, -extension+1-k) for r in rows)
                    for k in range(1, tail+1):
                        cells.update((r, -extension+1-span-k) for r in (-span, span))
                    masks.add(tuple(sorted(cells)))
    return sorted(masks)


def _levels(rng, length, size):
    h = size//2
    low, high = h, length-h-1
    max_count = (high-low)//(size+1)+1
    count = int(rng.integers(3, max_count+1))
    spare = high-low-(count-1)*(size+1)
    extras = rng.multinomial(spare, np.full(count+1, 1/(count+1)))
    result = [low+int(extras[0])]
    for i in range(1, count):
        result.append(result[-1]+size+1+int(extras[i]))
    return result


def _compatible_append(points, q):
    """The new tile's entry face must not compete at any earlier tip."""
    entry = _direction(points[-1], q)
    for p, nxt in zip(points, points[1:]):
        if _direction(p, q) == entry:
            if _direction(p, nxt) != entry or _distance(p, q) <= _distance(p, nxt):
                return False
    return True


def _layout(rng, size, hops, decoys, bypass, need_turn):
    low = 16 if size == 3 else 22
    for _ in range(160):
        shape = tuple(map(int, rng.integers(low, 31, size=2)))
        rows, cols = _levels(rng, shape[0], size), _levels(rng, shape[1], size)
        sockets = [(r, c) for r in rows for c in cols]
        if len(sockets) < hops+decoys+1:
            continue
        # At most 9 arrow cells; maintain a strict background majority.
        if (hops+decoys)*size*size + 9 >= shape[0]*shape[1]/2:
            continue
        for _ in range(12):
            points = [sockets[int(rng.integers(len(sockets)))]]
            reserved = None
            while len(points) <= hops:
                options = []
                for q in sockets:
                    if q in points or q == reserved or _direction(points[-1], q) is None:
                        continue
                    if not _compatible_append(points, q):
                        continue
                    between = [x for x in sockets if x not in (points[-1], q)
                               and _direction(points[-1], x) == _direction(points[-1], q)
                               and _distance(points[-1], x) < _distance(points[-1], q)]
                    if len(points) == 1 and bypass and not between:
                        continue
                    options.append((q, between))
                if not options:
                    break
                q, between = options[int(rng.integers(len(options)))]
                if len(points) == 1 and bypass:
                    reserved = between[int(rng.integers(len(between)))]
                points.append(q)
            if len(points) != hops+1:
                continue
            ds = [_direction(p, q) for p, q in zip(points, points[1:])]
            if need_turn and len(set(ds)) == 1:
                continue
            available = [q for q in sockets if q not in points and q != reserved]
            others = [reserved] if reserved is not None else []
            count = decoys-len(others)
            if count < 0 or len(available) < count:
                continue
            if count:
                others.extend(available[int(i)] for i in rng.choice(len(available), count, replace=False))
            if need_turn:
                initial_ray = [q for q in points[1:]+others if _direction(points[0], q) == ds[0]]
                if min(initial_ray, key=lambda q: _distance(points[0], q)) != points[1]:
                    continue
            return shape, points, ds, others
    raise RuntimeError('bounded layout search exhausted')


def _tile(rng, size, carrier, ink, details, entry):
    """Only the entry midpoint is an ink port; internal ink may repeat."""
    h = size//2
    ports = {_rotate((0, -h), d-1) for d in range(4)}
    extras = [(r, c) for r in range(-h, h+1) for c in range(-h, h+1)
              if (r, c) not in ports]
    for _ in range(64):
        t = np.full((size, size), carrier, dtype=np.int8)
        for r, c in ports:
            if rng.random() < .35:
                t[h+r, h+c] = int(rng.choice(details))
        if entry is not None:
            r, c = _rotate((0, -h), entry-1)
            t[h+r, h+c] = ink
        # Leave enough carrier even if every side port was decorated.
        budget = (size*size-1)//2-int(np.count_nonzero(t != carrier))
        if budget < 1:
            continue
        n = int(rng.integers(1, min(len(extras), budget, 8)+1))
        indices = rng.choice(len(extras), n, replace=False)
        for j in indices:
            r, c = extras[int(j)]
            t[h+r, h+c] = int(rng.choice([ink]+details))
        if entry is None and not np.any(t == ink):
            r, c = extras[int(indices[0])]
            t[h+r, h+c] = ink
        # Every selected payload is a transformation witness, also in tiny puzzles.
        if len({_view(t, op).tobytes() for op in range(8)}) == 8:
            return t
    # Guaranteed chiral fallback: a marked side and a distinct adjacent corner.
    t = np.full((size, size), carrier, dtype=np.int8)
    d = entry if entry is not None else 1
    side = _rotate((0, -h), d-1)
    corner = _rotate((-h, -h), d-1)
    t[h+side[0], h+side[1]] = ink if entry is not None else details[0]
    t[h+corner[0], h+corner[1]] = details[0] if entry is not None else ink
    return t


def _make_pair(rng, size, mask, hops, decoys, bypass, need_turn, palette):
    shape, points, ds, others = _layout(rng, size, hops, decoys, bypass, need_turn)
    if palette is None:
        palette = tuple(map(int, rng.permutation(10)))
    bg, carrier, ink = palette[:3]
    details = list(palette[3:3+int(rng.integers(1, 8))])
    a = np.full(shape, bg, dtype=np.int8)
    for q, d in zip(points[1:], ds):
        a[_square(q, size//2)] = _tile(rng, size, carrier, ink, details, d)
    for q in others:
        a[_square(q, size//2)] = _tile(rng, size, carrier, ink, details, None)
    for r, c in _arrow_cells(mask, points[0], ds[0]):
        a[r, c] = ink
    info = _analyze(a)
    observed, directions = _trace(info)
    _require(observed == points and directions == ds, "rendered route disagrees with construction")
    # Construction uses its explicit itinerary; the public solve rereads input.
    return {'input': a, 'output': _render(info, points, ds)}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _candidate_rules(train):
    """Finite, author-owned audit of sample-coordinate and D4 alternatives."""
    infos = [(_analyze(p['input']), np.asarray(p['output'])) for p in train]
    h = infos[0][0]['half']
    survivors = []
    for r in range(-h, h+1):
        for c in range(-h, h+1):
            sample = (r, c)
            candidates = set(range(8))
            for info, expected in infos:
                try:
                    points, ds = _trace(info, sample)
                except AssertionError:
                    candidates.clear()
                    break
                candidates = {op for op in candidates if np.array_equal(
                    _render(info, points, ds, op), expected)}
                if not candidates:
                    break
            survivors.extend((sample, op) for op in sorted(candidates))
    _require(((0, -h), 2) in survivors, "intended rule absent from training")
    return survivors


def _coverage(puzzle):
    candidates = _candidate_rules(puzzle['train'])
    for pair in puzzle['test']:
        info = _analyze(pair['input'])
        expected = np.asarray(pair['output'])
        for sample, op in candidates:
            try:
                points, ds = _trace(info, sample)
            except AssertionError:
                return False
            if not np.array_equal(_render(info, points, ds, op), expected):
                return False
    return True


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 'num_train', 2, 10)
    m = _count(num_test, 'num_test', 1, 4)
    if n <= m:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    size = int(rng.choice([3, 5]))
    bank = _mask_bank(size)
    mask_order = list(map(int, rng.permutation(len(bank))))
    longer = bool(rng.integers(2))
    shared_palette = tuple(map(int, rng.permutation(10))) if rng.random() < .25 else None
    for _ in range(12):
        puzzle, seen = {'train': [], 'test': []}, set()
        for split, count in (('train', n), ('test', m)):
            for i in range(count):
                if split == 'train':
                    hops = 2 if i == 0 else 3 if i == 1 else int(rng.integers(2, 4 if size == 5 else 5))
                    bypass = i == 0
                    decoys = 1 if bypass else 0
                    if not longer and i > 1:
                        decoys = int(rng.integers(2))
                else:
                    hops = int(rng.integers(4, 7) if size == 5 else rng.integers(5, 9)) if longer else int(rng.integers(2, 5))
                    bypass = bool(rng.integers(2))
                    decoys = int(rng.integers(2, 4)) if not longer else int(rng.integers(int(bypass), 3))
                mask = bank[mask_order[i % len(bank)]] if split == 'train' and i < 2 else bank[int(rng.integers(len(bank)))]
                for _ in range(16):
                    pair = _make_pair(rng, size, mask, hops, decoys, bypass,
                                      need_turn=(split == 'train' and i == 1), palette=shared_palette)
                    key = _key(pair['input'])
                    if key not in seen:
                        seen.add(key)
                        puzzle[split].append(pair)
                        break
                else:
                    raise RuntimeError('distinct input search exhausted')
        if _coverage(puzzle):
            rng.shuffle(puzzle['train'])
            validate_generated_puzzle(puzzle)
            return puzzle
    raise RuntimeError('bounded demonstration coverage search exhausted')


def _features(pair):
    info = _analyze(pair['input'])
    points, ds = _trace(info)
    aligned = [q for q in info['tiles'] if _direction(points[0], q) == ds[0]]
    nearest = min(aligned, key=lambda q: _distance(points[0], q))
    turns = [(b-a) % 4 for a, b in zip(ds, ds[1:]) if a != b]
    distances = tuple(_distance(p, q) for p, q in zip(points, points[1:]))
    # Translation-normalized route, retaining actual distances and turn senses.
    route = tuple((r-points[0][0], c-points[0][1]) for r, c in points)
    return dict(size=info['size'], shape=info['grid'].shape, mask=info['mask'],
                hops=len(ds), decoys=len(info['tiles'])-len(ds),
                bypass=nearest != points[1], turn=bool(turns), mixed_turns=1 in turns and 3 in turns,
                distances=distances, route=route, initial=info['facing'], final=ds[-1],
                bg=info['bg'], carrier=info['carrier'], ink=info['ink'])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "split must be a list")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid counts")
    seen, features, queries, sizes = set(), [], [], set()
    ungated_failure = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "canvas changed")
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(a)
            points, ds = _trace(info)
            _require(len(ds) >= 2, "need a multi-hop relay")
            sizes.add(info['size'])
            expected = _render(info, points, ds)
            _require(np.array_equal(b, expected), "incorrect relay output")
            _require(np.array_equal(np.bincount(a.ravel(), minlength=10),
                                   np.bincount(b.ravel(), minlength=10)), "color counts not conserved")
            # Direct scalar payload map, independent of NumPy tile views/slices.
            h = info['half']
            changed_sockets = np.zeros(a.shape, dtype=bool)
            for p in points:
                changed_sockets[_square(p, h)] = True
            for p, q in zip(points, points[1:]):
                for dr in range(-h, h+1):
                    for dc in range(-h, h+1):
                        _require(b[p[0]-dr, p[1]-dc] == a[q[0]+dr, q[1]+dc], "payload scalar map failed")
                d = _direction(p, q)
                dy, dx = DIRS[d]
                _require(a[q[0]-h*dy, q[1]-h*dx] == info['ink'], "wrong gate")
            _require(np.array_equal(a[~changed_sockets], b[~changed_sockets]), "unused region changed")
            expected_arrow = _arrow_cells(info['mask'], points[-1], ds[-1])
            actual_arrow = {(points[-1][0]+r-h, points[-1][1]+c-h)
                            for r, c in zip(*np.where(b[_square(points[-1], h)] == info['ink']))}
            _require(actual_arrow == expected_arrow, "terminal arrow not rigid")
            _require(not np.array_equal(_render(info, *_trace(info, limit=1)), b), "one-hop shortcut survives")
            _require(not np.array_equal(_render(info, points, ds, 0), b), "unrotated payload shortcut survives")
            if split == 'train':
                features.append(_features(pair))
                try:
                    up, ud = _trace(info, ungated=True)
                    ungated_failure |= not np.array_equal(_render(info, up, ud), b)
                except AssertionError:
                    ungated_failure = True
            else:
                queries.append(_features(pair))
    _require(len(sizes) == 1, "tile size must be fixed within puzzle")
    _require(len({f['mask'] for f in features}) >= 2, "two arrow masks must be demonstrated")
    _require(len({f['hops'] for f in features}) >= 2, "training needs distinct hop counts")
    _require(any(f['turn'] for f in features), "training must show a turn")
    _require(any(f['bypass'] for f in features) and any(not f['bypass'] for f in features), "gating witnesses missing")
    _require(ungated_failure, "ungated shortcut survives training")
    longer_queries = all(f['hops'] > max(t['hops'] for t in features) for f in queries)
    more_distraction = all(f['decoys'] > max(t['decoys'] for t in features) for f in queries)
    _require(longer_queries or more_distraction, "train/test shift missing")
    _require(_coverage(puzzle), "training leaves query-relevant finite-rule ambiguity")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for name in ('input', 'output'):
                a = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streaming seeds, small feature sets, and bounded negative tests."""
    hashes, features, candidate_counts = set(), [], Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        hashes.add(_digest(p))
        candidate_counts[len(_candidate_rules(p['train']))] += 1
        for split in ('train', 'test'):
            for pair in p[split]:
                before = pair['input'].copy()
                _require(np.array_equal(solve(pair['input']), pair['output']), "solve disagrees")
                _require(np.array_equal(pair['input'], before), "solve mutated input")
                features.append(_features(pair))
    _require(len(hashes) == 42, "seeds collapsed")
    _require({f['size'] for f in features} == {3, 5}, "tile scales missing")
    _require(len({f['mask'] for f in features}) >= 6, "mask geometry lacks variation")
    _require(len({f['shape'] for f in features}) >= 50, "canvas shapes lack variation")
    _require(len({f['route'] for f in features}) >= 100, "route geometry lacks variation")
    _require({f['initial'] for f in features} == set(range(4)), "initial directions missing")
    _require({f['final'] for f in features} == set(range(4)), "terminal directions missing")
    _require({f['bg'] for f in features} == set(range(10)), "background roles missing")
    _require({f['carrier'] for f in features} == set(range(10)), "carrier roles missing")
    _require({f['ink'] for f in features} == set(range(10)), "ink roles missing")
    _require(sum(f['mixed_turns'] for f in features) >= 15, "mixed turns missing")
    _require(any(len(set(f['distances'])) > 1 for f in features), "unequal distances missing")
    _require(any(f['decoys'] >= 3 for f in features), "distractor counts missing")
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(1000+n*10+m, n, m)
            _require(len(p['train']) == n and len(p['test']) == m, "counts ignored")
    _require(_digest(generate_puzzle(123, 5, 2)) == _digest(generate_puzzle(123, 5, 2)), "nondeterminism")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(7, 2, 1)
    q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    q['test'][0]['output'][0, 0] = (int(q['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    # Also reject a missing transport turn even though global color counts survive.
    q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    info = _analyze(q['test'][0]['input'])
    points, ds = _trace(info)
    q['test'][0]['output'] = _render(info, points, ds, 0)
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError('identity payload transport accepted')
    return {'seeds': 42, 'pairs': len(features), 'distinct_puzzles': len(hashes),
            'route_geometries': len({f['route'] for f in features}),
            'canvas_shapes': len({f['shape'] for f in features}),
            'arrow_masks': len({f['mask'] for f in features}),
            'tile_sizes': sorted({f['size'] for f in features}),
            'hop_counts': sorted({f['hops'] for f in features}),
            'candidate_rule_counts': dict(sorted(candidate_counts.items())),
            'semantic_verification': 'generator-owned checks only'}
