"""Count-addressed ribbon cutting. Standard library + NumPy; no file I/O.

All answer conventions are fixed and visible. The builder labels from its own
ordered ribbon; solve reconstructs the order and object counts from input pixels.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import numpy as np

_D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors outside 0..9")
    return a


def _components(a, bg):
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        cells, stack = {root}, [root]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _analyze(value):
    """Only visible input pixels; no seed, examples, or generator parameters."""
    a = _grid(value)
    histogram = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(histogram.argmax())
    _require(int(histogram[bg]) * 2 > a.size, "background needs a strict majority")
    parts = _components(a, bg)
    ribbons = [p for p in parts if len({int(a[q]) for q in p}) > 1]
    _require(len(ribbons) == 1, "expected one multicolored component")
    ribbon = ribbons[0]
    _require(6 <= len(ribbon) <= 30, "ribbon length")
    neighbors = {p: [(p[0]+dr, p[1]+dc) for dr, dc in _D4
                     if (p[0]+dr, p[1]+dc) in ribbon] for p in ribbon}
    ends = [p for p in ribbon if len(neighbors[p]) == 1]
    _require(len(ends) == 2 and all(len(n) in (1, 2) for n in neighbors.values()),
             "ribbon must be an induced chain")
    _require(ends[0][0] != ends[1][0], "endpoint rows must differ")
    current, previous = min(ends, key=lambda p: p[0]), None
    path = []
    while current is not None:
        _require(current not in path, "cycle")
        path.append(current)
        choices = [q for q in neighbors[current] if q != previous]
        previous, current = current, choices[0] if choices else None
    _require(len(path) == len(ribbon), "incomplete ribbon")
    seq = [int(a[p]) for p in path]
    palette = set(seq)
    _require(len(palette) >= 3, "ribbon needs at least three colors")
    objects = []
    labels = {p: i for i, part in enumerate(parts) for p in part}
    for p, i in labels.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = p[0]+dr, p[1]+dc
                _require(q not in labels or labels[q] == i, "components need a diagonal moat")
    for cells in parts:
        if cells is ribbon:
            continue
        colors = {int(a[p]) for p in cells}
        _require(len(colors) == 1 and 1 <= len(cells) <= 9, "loose object shape")
        _require(max(r for r, c in cells)-min(r for r, c in cells) < 4
                 and max(c for r, c in cells)-min(c for r, c in cells) < 4,
                 "loose object must fit 4x4")
        objects.append((colors.pop(), cells))
    _require(len(objects) <= 13, "too many loose objects")
    all_counts = Counter(c for c, cells in objects)
    counts = {c: n for c, n in all_counts.items() if c in palette}
    _require(len(counts) == 2, "exactly two queried ribbon colors required")
    _require(all(1 <= n <= min(5, seq.count(c)) for c, n in counts.items()),
             "ordinal outside visible occurrence capacity")
    indices = sorted([i for i, v in enumerate(seq) if v == c][n-1]
                     for c, n in counts.items())
    areas = {c: sum(len(cells) for color, cells in objects if color == c) for c in counts}
    return dict(background=bg, path=path, sequence=seq, objects=objects,
                counts=counts, indices=indices, areas=areas)


def _cut(sequence, counts):
    indices = sorted([i for i, value in enumerate(sequence) if value == c][n-1]
                     for c, n in counts.items())
    return tuple(sequence[indices[0]:indices[1]+1])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the inclusive count-addressed subpath, in upper-endpoint order."""
    info = _analyze(input_grid)
    lo, hi = info['indices']
    return np.asarray(info['sequence'][lo:hi+1], dtype=np.int8).reshape(-1, 1)


def _shortcuts_fail(path, sequence, counts):
    answer = _cut(sequence, counts)
    color = dict(zip(path, sequence))
    orders = (sorted(path), sorted(path, key=lambda p: (p[1], p[0])))
    return (_cut(sequence[::-1], counts) != answer and
            all(_cut([color[p] for p in order], counts) != answer for order in orders))


def _normalize(path):
    r0, c0 = min(r for r, c in path), min(c for r, c in path)
    return [(r-r0, c-c0) for r, c in path]


def _fallback_path(length=24):
    path = [(0, 0)]
    for (dr, dc), n in (((0, 1), 4), ((1, 0), 3), ((0, -1), 5),
                        ((1, 0), 2), ((0, 1), length-15)):
        for _ in range(n):
            path.append((path[-1][0]+dr, path[-1][1]+dc))
    return _normalize(path)


def _make_path(rng, length):
    """Bounded induced walks; every new cell touches only the old tip."""
    for _ in range(100):
        path, occupied = [(0, 0)], {(0, 0)}
        previous = None
        persistence = float(rng.uniform(.7, 5.0))
        for _ in range(length-1):
            choices, weights = [], []
            for dr, dc in _D4:
                q = path[-1][0]+dr, path[-1][1]+dc
                if q in occupied or sum((q[0]+a, q[1]+b) in occupied for a, b in _D4) != 1:
                    continue
                rs = [p[0] for p in path] + [q[0]]
                cs = [p[1] for p in path] + [q[1]]
                if max(rs)-min(rs) >= 17 or max(cs)-min(cs) >= 17:
                    continue
                choices.append((q, (dr, dc)))
                weights.append(persistence if previous == (dr, dc) else 1.0)
            if not choices:
                break
            probs = np.asarray(weights) / sum(weights)
            q, previous = choices[int(rng.choice(len(choices), p=probs))]
            path.append(q)
            occupied.add(q)
        if len(path) == length and path[0][0] != path[-1][0]:
            if path[0][0] > path[-1][0]:
                path.reverse()
            return _normalize(path)
    # Used only at lengths >=18; shorter ordinary requests are promoted here.
    return _fallback_path(max(18, length))


def _sequence(rng, length, palette):
    seq = []
    while len(seq) < length:
        choices = [c for c in palette if not seq or c != seq[-1]]
        color = int(rng.choice(choices))
        seq.extend([color] * int(rng.integers(1, 5)))
    seq = seq[:length]
    # Keep each chosen semantic category visibly present; boundaries remain
    # independent of bends and most runs remain freely sampled.
    for c, i in zip(palette, rng.choice(length, len(palette), replace=False)):
        seq[int(i)] = c
    return seq


def _witness_sequence(rng, length, palette):
    x, y, z = palette[:3]
    seq = _sequence(rng, length, palette)
    a = int(rng.integers(1, 5))
    b = int(rng.integers(max(a+5, length//2), length-5))
    seq = [int(rng.choice([c for c in palette if c != y])) if c == y else c for c in seq]
    for i in range(a):
        if seq[i] == x:
            seq[i] = z
    seq[a:a+3] = [x, x, z]
    seq[b-2], seq[b-1], seq[b] = y, z, y
    seq[-3], seq[-2], seq[-1] = y, x, z
    return seq


def _teaching_ribbon(rng, palette):
    counts = ({palette[0]: 1, palette[1]: 2}, {palette[0]: 2, palette[1]: 2})
    for _ in range(32):
        path = _make_path(rng, int(rng.integers(20, 31)))
        seq = _witness_sequence(rng, len(path), palette)
        if all(_shortcuts_fail(path, seq, c) for c in counts):
            return path, seq, counts
    # A fully specified non-palindromic fallback retains both teaching witnesses.
    path = _fallback_path(24)
    x, y, z = palette[:3]
    seq = [z]*24
    for i in (1, 2, 8, 10, 22):
        seq[i] = x
    for i in (13, 15, 21):
        seq[i] = y
    assert all(_shortcuts_fail(path, seq, c) for c in counts)
    return path, seq, counts


def _shape(rng, area=None):
    """Shape is independent of count; bounded 4x4 polyomino growth."""
    area = int(rng.integers(1, 10)) if area is None else int(area)
    cells = {(int(rng.integers(4)), int(rng.integers(4)))}
    while len(cells) < area:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4
                           if 0 <= r+dr < 4 and 0 <= c+dc < 4} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    return tuple(sorted(_normalize(list(cells))))


def _specifications(rng, counts, palette, bg, forced=None):
    forced = {} if forced is None else forced
    specs = [(c, _shape(rng, forced[c][i] if c in forced else None))
             for c, n in counts.items() for i in range(n)]
    distractors = [c for c in range(10) if c != bg and c not in palette]
    nd = int(rng.integers(0, 4))
    specs.extend((int(rng.choice(distractors)), _shape(rng)) for _ in range(nd))
    return specs


def _try_pack(rng, path, seq, bg, specs, h, w, top, left):
    grid = np.full((h, w), bg, dtype=np.int8)
    blocked = np.zeros((h, w), dtype=bool)

    def put(cells, color):
        for r, c in cells:
            grid[r, c] = color
            blocked[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True

    for (r, c), color in zip(path, seq):
        put([(r+top, c+left)], color)
    ordered = sorted(specs, key=lambda s: (-len(s[1]), float(rng.random())))
    for color, shape in ordered:
        sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
        candidates = np.ones((h-sh+1, w-sw+1), dtype=bool)
        for dr, dc in shape:
            candidates &= ~blocked[dr:dr+h-sh+1, dc:dc+w-sw+1]
        rr, cc = np.nonzero(candidates)
        if not len(rr):
            return None
        j = int(rng.integers(len(rr)))
        put([(int(rr[j])+dr, int(cc[j])+dc) for dr, dc in shape], color)
    if np.count_nonzero(grid == bg)*2 <= grid.size:
        return None
    return grid


def _reserved_pack(path, seq, bg, specs):
    """At most 13 objects, each <=4x4; 15 disjoint reserved boxes suffice."""
    grid = np.full((30, 30), bg, dtype=np.int8)
    for p, c in zip(path, seq):
        grid[p] = c
    slots = [(r, c) for r in range(0, 30, 5) for c in (19, 24)]
    slots += [(20, c) for c in (0, 5, 10)]
    assert len(specs) <= len(slots)
    for (color, shape), (r, c) in zip(specs, slots):
        for dr, dc in shape:
            grid[r+dr, c+dc] = color
    return grid


def _scenes(rng, path, seq, bg, all_specs):
    """One or two scenes sharing exactly the same ribbon placement."""
    bh, bw = max(r for r, c in path)+1, max(c for r, c in path)+1
    for attempt in range(8):
        if attempt == 0:
            h = int(rng.integers(max(12, bh), max(16, bh+7)+1))
            w = int(rng.integers(max(12, bw), max(16, bw+7)+1))
        else:
            low = 17 if attempt < 4 else 27
            h = int(rng.integers(max(low, bh), 31))
            w = int(rng.integers(max(low, bw), 31))
        top, left = int(rng.integers(h-bh+1)), int(rng.integers(w-bw+1))
        grids = [_try_pack(rng, path, seq, bg, specs, h, w, top, left) for specs in all_specs]
        if all(g is not None for g in grids):
            return grids
    return [_reserved_pack(path, seq, bg, specs) for specs in all_specs]


def _ordinary(rng, bg, palette=None, query_pair=None, large_rank=False,
              allow_above=True):
    for _ in range(48):
        if palette is None:
            local_palette = [int(c) for c in rng.choice([c for c in range(10) if c != bg],
                                                        int(rng.integers(3, 7)), replace=False)]
        else:
            local_palette = palette
        path = _make_path(rng, int(rng.integers(12, 31)))
        if not allow_above and min(r for r, c in path) < path[0][0]:
            continue
        seq = _sequence(rng, len(path), local_palette)
        options = []
        pairs = [query_pair] if query_pair is not None else combinations(local_palette, 2)
        for x, y in pairs:
            for nx in range(1, min(5, seq.count(x))+1):
                for ny in range(1, min(5, seq.count(y))+1):
                    if large_rank and max(nx, ny) < 3:
                        continue
                    counts = {x: nx, y: ny}
                    crop = _cut(seq, counts)
                    if (4 <= len(crop) <= len(seq)-2 and set(crop)-{x, y}
                            and _shortcuts_fail(path, seq, counts)):
                        options.append(counts)
        if options:
            counts = options[int(rng.integers(len(options)))]
            specs = _specifications(rng, counts, local_palette, bg)
            grid = _scenes(rng, path, seq, bg, [specs])[0]
            return {'input': grid, 'output': np.array(_cut(seq, counts), dtype=np.int8).reshape(-1, 1)}
    # Guaranteed bounded semantic fallback: this sequence supports rank 3 as
    # well as category changes. Search only its finite, visible address space.
    local_palette = local_palette[:3]
    if query_pair is not None:
        local_palette = list(dict.fromkeys(list(query_pair)+local_palette))[:3]
    x, y, z = local_palette
    path = _fallback_path(30)
    seq = [x, x, z, y, z, x, z, y, x, z, y, z, x, y, z,
           z, x, z, y, z, x, y, z, x, z, y, z, x, y, z]
    for nx in range(3 if large_rank else 1, 6):
        for ny in range(1, 6):
            counts = {x: nx, y: ny}
            crop = _cut(seq, counts)
            if 4 <= len(crop) <= 28 and set(crop)-{x, y} and _shortcuts_fail(path, seq, counts):
                specs = _specifications(rng, counts, local_palette, bg)
                grid = _scenes(rng, path, seq, bg, [specs])[0]
                return {'input': grid, 'output': np.array(crop, dtype=np.int8).reshape(-1, 1)}
    raise RuntimeError('finite address fallback failed')


def _count_arguments(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name+' must be an integer')
        if not lo <= int(value) <= hi:
            raise ValueError(name+' outside legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count_arguments(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = [int(c) for c in rng.permutation(10)]
    bg = colors[0]
    palette = colors[1:1+int(rng.integers(3, 6))]
    ordinal_regime = bool(rng.integers(2))
    path, seq, count_rows = _teaching_ribbon(rng, palette)
    x, y = palette[:2]
    area = int(rng.integers(4, 10))
    first = int(rng.integers(1, area))
    forces = ({x: [area], y: [1, 2]}, {x: [first, area-first], y: [3, 4]})
    specs = [_specifications(rng, c, palette, bg, f) for c, f in zip(count_rows, forces)]
    scenes = _scenes(rng, path, seq, bg, specs)
    puzzle = {'train': [{'input': g, 'output': np.array(_cut(seq, c), dtype=np.int8).reshape(-1, 1)}
                        for g, c in zip(scenes, count_rows)], 'test': []}
    seen = {_key(g) for g in scenes}
    for split, count in (('train', nt-2), ('test', nq)):
        for _ in range(count):
            for attempt in range(12):
                if split == 'test' and not ordinal_regime:
                    # A formerly unqueried ribbon color becomes an address.
                    local_bg, local_palette = bg, palette
                    query = (palette[2], palette[int(rng.integers(2))])
                else:
                    local_bg, local_palette, query = int(rng.integers(10)), None, None
                taught_above = any(
                    min(r for r, c in d['path']) < d['path'][0][0]
                    for d in (_analyze(p['input']) for p in puzzle['train']))
                pair = _ordinary(rng, local_bg, local_palette, query,
                                 large_rank=split == 'test' and ordinal_regime,
                                 allow_above=split == 'train' or taught_above)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct-input generation exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema'
    assert all(isinstance(puzzle[s], list) for s in puzzle), 'split schema'
    _count_arguments(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema'
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen, 'duplicate input'
            seen.add(_key(a))
            info = _analyze(a)
            expected = np.array(_cut(info['sequence'], info['counts']), dtype=np.int8).reshape(-1, 1)
            assert np.array_equal(b, expected), 'wrong addressed interval'
            assert len(b) < len(info['path']), 'generated examples must crop'
            assert set(b.ravel())-set(info['counts']), 'missing unqueried interior color'
            assert _shortcuts_fail(info['path'], info['sequence'], info['counts']), 'traversal shortcut'
            infos[split].append(info)
    a, b = infos['train'][:2]
    assert (a['path'], a['sequence'], a['background']) == (b['path'], b['sequence'], b['background']), 'teaching ribbons differ'
    changed = [c for c in a['counts'] if a['counts'][c] != b['counts'].get(c)]
    assert len(changed) == 1 and set(a['counts']) == set(b['counts']), 'count intervention'
    x = changed[0]
    y = next(c for c in a['counts'] if c != x)
    assert (a['counts'][x], b['counts'][x]) == (1, 2), 'one/two progression'
    assert a['areas'][x] == b['areas'][x], 'area collision missing'
    assert a['counts'][y] == b['counts'][y] == 2 and a['areas'][y] != b['areas'][y], 'same count/different area'
    positions = [i for i, c in enumerate(a['sequence']) if c == x]
    assert len(positions) >= 3 and positions[1] == positions[0]+1, 'cell versus run witness'
    py = [i for i, c in enumerate(a['sequence']) if c == y]
    assert len(py) >= 3 and py[1] > positions[1], 'internal second ordinal witness'
    assert _cut(a['sequence'], a['counts']) != _cut(b['sequence'], b['counts']), 'B ablation unchanged'
    assert all(0 < d['indices'][0] < d['indices'][1] < len(d['path'])-1 for d in (a, b)), 'teaching crop must be internal'
    if any(min(r for r, c in d['path']) < d['path'][0][0] for d in infos['test']):
        assert any(min(r for r, c in d['path']) < d['path'][0][0]
                   for d in infos['train']), 'above-endpoint test needs a demonstration'


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """42 seeds streamed in memory; no files, logs, or execution traces."""
    signatures, geometries, shapes, backgrounds, masks = set(), set(), set(), set(), set()
    lengths, turns, ranks, palette_sizes, crop_lengths = set(), set(), set(), set(), set()
    above = distractors = border = category_shift = ordinal_shift = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(hashlib.sha256(repr(_signature(p)).encode()).digest())
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                info = _analyze(a)
                path, seq = info['path'], info['sequence']
                shapes.add(a.shape)
                backgrounds.add(info['background'])
                lengths.add(len(path))
                palette_sizes.add(len(set(seq)))
                ranks.update(info['counts'].values())
                crop_lengths.add(len(pair['output']))
                geometries.add(tuple((r-path[0][0], c-path[0][1]) for r, c in path))
                steps = [(q[0]-v[0], q[1]-v[1]) for v, q in zip(path, path[1:])]
                turns.add(sum(v != q for v, q in zip(steps, steps[1:])))
                above += min(r for r, c in path) < path[0][0]
                border += any(r in (0, a.shape[0]-1) or c in (0, a.shape[1]-1) for r, c in path)
                distractors += any(c not in set(seq) for c, cells in info['objects'])
                for c, cells in info['objects']:
                    masks.add(tuple(sorted(_normalize(list(cells)))))
                before = a.copy()
                assert np.array_equal(solve(a), pair['output']) and np.array_equal(a, before)
                # Same objects, radically smaller area: answers must stay fixed.
                small = a.copy()
                for c, cells in info['objects']:
                    for q in cells:
                        small[q] = info['background']
                    small[min(cells)] = c
                assert np.array_equal(solve(small), pair['output']), 'area dependence'
                # Fixed invertible relabeling, including the background role.
                perm = np.array([7, 3, 8, 1, 9, 0, 6, 4, 2, 5], dtype=np.int8)
                assert np.array_equal(solve(perm[a]), perm[pair['output']]), 'literal color dependence'
        a, b = [_analyze(pair['input']) for pair in p['train'][:2]]
        x = next(c for c in a['counts'] if a['counts'][c] == 1)
        changed = p['train'][1]['input'].copy()
        cells = next(cells for c, cells in b['objects'] if c == x)
        for q in cells:
            changed[q] = b['background']
        assert np.array_equal(solve(changed), p['train'][0]['output']), 'count intervention'
        train_queries = set(a['counts'])
        for pair in p['test']:
            d = _analyze(pair['input'])
            category_shift += bool(set(d['counts'])-train_queries)
            ordinal_shift += max(d['counts'].values()) >= 3
    assert len(signatures) == 42 and len(geometries) >= 120
    assert len(shapes) >= 35 and backgrounds == set(range(10))
    assert len(lengths) >= 15 and len(turns) >= 8
    assert ranks == set(range(1, 6)) and len(palette_sizes) >= 4 and len(crop_lengths) >= 16
    assert len(masks) >= 80 and min(above, distractors, border, category_shift, ordinal_shift) >= 5
    # Valid family boundaries deliberately outside the ordinary sampling band:
    # a six-cell ribbon, both endpoint cuts, and a two-cell interval.
    fixture = np.zeros((12, 12), dtype=np.int8)
    fixture[2:8, 2] = [1, 2, 3, 3, 2, 4]
    fixture[1, 8], fixture[9, 8] = 1, 4
    assert np.array_equal(solve(fixture).ravel(), [1, 2, 3, 3, 2, 4])
    fixture[9, 8] = 2
    assert np.array_equal(solve(fixture).ravel(), [1, 2])
    # Highest supported ordinal, with the far cut at the last ribbon cell.
    fixture = np.zeros((22, 14), dtype=np.int8)
    fixture[2:12, 2] = [1, 1, 1, 1, 1, 2, 3, 3, 3, 3]
    for i, color in enumerate([1]*5+[3]*4):
        fixture[1+2*(i//2), 8+2*(i%2)] = color
    assert np.array_equal(solve(fixture).ravel(), [1, 2, 3, 3, 3, 3])
    # The constructive geometry safety net remains a single induced chain.
    for length in range(18, 31):
        path = _fallback_path(length)
        cells = set(path)
        degrees = [sum((r+dr, c+dc) in cells for dr, dc in _D4) for r, c in path]
        assert len(path) == length and degrees.count(1) == 2 and set(degrees) <= {1, 2}
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    assert _signature(generate_puzzle(17, 2, 1)) == _signature(generate_puzzle(17, 2, 1))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(0, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
