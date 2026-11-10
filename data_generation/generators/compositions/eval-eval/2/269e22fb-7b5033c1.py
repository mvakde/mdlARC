"""Necklace bookmark: register a crop, then unroll a cycle from its directed end.

Only the standard library and NumPy are used. The solver reads both objects
from the input; it has no reference bank, puzzle state, or generation metadata.
"""
from collections import defaultdict

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _neighbors(cells):
    return {p: sorted((p[0] + dr, p[1] + dc) for dr, dc in _DIRS
                      if (p[0] + dr, p[1] + dc) in cells) for p in cells}


def _trace(neighbors, start, first=None):
    """A deterministic cycle order, or the order of an open chain."""
    ordered, previous, current = [], None, start
    while current not in ordered:
        ordered.append(current)
        choices = [p for p in neighbors[current] if p != previous]
        if not choices:
            break
        nxt = first if previous is None and first is not None else choices[0]
        previous, current = current, nxt
    if len(ordered) != len(neighbors):
        raise ValueError("the object is not one simple chain or cycle")
    return ordered


def _read_objects(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[int(counts.argmax())])
    if int(counts.max()) * 2 <= a.size:
        raise ValueError("background must occupy a strict majority")
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        components.append(component)
    if len(components) != 2:
        raise ValueError("expected exactly one loop and one open bookmark")
    loop = query = None
    for cells in components:
        graph = _neighbors(cells)
        endpoints = [p for p in cells if len(graph[p]) == 1]
        if all(len(graph[p]) == 2 for p in cells) and loop is None:
            start = min(cells)
            loop = _trace(graph, start, graph[start][0])
        elif (len(endpoints) == 2 and all(len(graph[p]) in (1, 2) for p in cells)
              and query is None):
            if endpoints[0][0] == endpoints[1][0]:
                raise ValueError("bookmark endpoints must have different rows")
            query = _trace(graph, min(endpoints, key=lambda p: p[0]))
        else:
            raise ValueError("branches, contacts, or extra objects are invalid")
    if loop is None or query is None or not 8 <= len(loop) <= 30:
        raise ValueError("a loop of 8..30 cells is required")
    if not 3 <= len(query) <= len(loop) - 2:
        raise ValueError("bookmark must leave at least two loop cells unseen")
    lc, qc = {int(a[p]) for p in loop}, {int(a[p]) for p in query}
    if len(lc) != len(qc) or not 2 <= len(lc) <= 4:
        raise ValueError("both objects must expose all two to four color roles")
    # A tight crop is read from the bookmark's bounding rectangle. No other
    # foreground object may occupy that rectangle.
    rr, cc = zip(*query)
    r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
    if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in loop):
        raise ValueError("the bookmark's rectangle contains another object")
    return a, bg, loop, query


def _transforms(points):
    """All eight lattice isometries; translation is handled separately."""
    p = np.asarray(points, dtype=int)
    r, c = p[:, 0], p[:, 1]
    return [np.column_stack((x, y)) for x, y in
            ((r, c), (c, -r), (-r, -c), (-c, r),
             (r, -c), (-r, c), (c, r), (-c, -r))]


def _shape(points):
    p = np.asarray(points, dtype=int)
    p = p - p[0]
    return min(tuple(map(tuple, t.tolist())) for t in _transforms(p))


def _object_shape(points):
    """Geometry only: remove translation, D4, cyclic phase, and trace direction."""
    return min(tuple(sorted(map(tuple, _tight(t).tolist()))) for t in _transforms(points))


def _word(colors):
    roles = {}
    return tuple(roles.setdefault(int(c), len(roles)) for c in colors)


def _bijection(source, target):
    forward, backward = {}, {}
    for s, t in zip(source, target):
        s, t = int(s), int(t)
        if (s in forward and forward[s] != t) or (t in backward and backward[t] != s):
            return None
        forward[s], backward[t] = t, s
    return forward


def _is_crop(loop, ids):
    arc = [loop[i] for i in ids]
    rr, cc = zip(*arc)
    return sum(min(rr) <= r <= max(rr) and min(cc) <= c <= max(cc)
               for r, c in loop) == len(ids)


def _registrations(a, loop, query, geometry=True):
    """Match the directed chain under D4 and a complete color bijection.

    Ordered relative coordinates suffice for exact rectangular matching because
    _is_crop also excludes every other loop cell from the implied rectangle.
    This checks foreground AND background, rather than only a subpath shape.
    """
    L, q = len(loop), len(query)
    query_shape = _shape(query) if geometry else None
    query_colors = [int(a[p]) for p in query]
    loop_colors = [int(a[p]) for p in loop]
    palette = set(loop_colors)
    found = []
    for start in range(L):
        for direction in (-1, 1):
            ids = [(start + direction * i) % L for i in range(q)]
            mapping = _bijection([loop_colors[i] for i in ids], query_colors)
            if mapping is None or set(mapping) != palette:
                continue
            if geometry and (not _is_crop(loop, ids)
                             or _shape([loop[i] for i in ids]) != query_shape):
                continue
            sequence = tuple(mapping[loop_colors[(start + direction * i) % L]]
                             for i in range(L))
            found.append((start, direction, mapping, sequence))
    return found


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the complete loop in the bookmark's colors and directed phase."""
    a, _, loop, query = _read_objects(input_grid)
    registrations = _registrations(a, loop, query)
    if len(registrations) != 1:
        raise ValueError("bookmark must identify one directed registration")
    return np.asarray(registrations[0][3], dtype=np.int8).reshape(-1, 1)


def _rectangle(h, w):
    return ([(0, c) for c in range(w)] + [(r, w - 1) for r in range(1, h)]
            + [(h - 1, c) for c in range(w - 2, -1, -1)]
            + [(r, 0) for r in range(h - 2, 0, -1)])


def _valid_cycle(points):
    cells = set(points)
    return (len(cells) == len(points)
            and all(sum((r + dr, c + dc) in cells for dr, dc in _DIRS) == 2
                    for r, c in points))


def _make_cycle(rng, length):
    """Grow a rectangle by rerouting straight triples through rectangular ears.

    The removed middle cell prevents an old edge from becoming a chord. Both
    inward and outward moves are considered, and actual adjacency is checked.
    """
    bases = [(h, w) for h in range(3, 7) for w in range(3, 8)
             if 2 * (h + w) - 4 <= length - 4]
    h, w = bases[int(rng.integers(len(bases)))]
    points = _rectangle(h, w)
    while len(points) < length:  # each iteration adds at least two cells
        options = []
        for i, middle in enumerate(points):
            a, b = points[i - 1], points[(i + 1) % len(points)]
            v = middle[0] - a[0], middle[1] - a[1]
            if v != (b[0] - middle[0], b[1] - middle[1]):
                continue
            for sign in (-1, 1):
                n = -v[1] * sign, v[0] * sign
                for depth in range(1, min(3, (length - len(points)) // 2) + 1):
                    new = [(a[0] + j * n[0], a[1] + j * n[1]) for j in range(1, depth + 1)]
                    new += [(middle[0] + depth * n[0], middle[1] + depth * n[1])]
                    new += [(b[0] + j * n[0], b[1] + j * n[1]) for j in range(depth, 0, -1)]
                    candidate = points[:i] + new + points[i + 1:]
                    if _valid_cycle(candidate):
                        options.append(candidate)
        if not options:
            return None
        points = options[int(rng.integers(len(options)))]
    return points


def _coloring(rng, length, k, witness):
    if witness or rng.random() < .6:
        motif = list(map(int, rng.permutation(k)))
        motif += list(map(int, rng.integers(k, size=int(rng.integers(3)))))
        rng.shuffle(motif)
        phase = int(rng.integers(len(motif)))
        colors = [motif[(i + phase) % len(motif)] for i in range(length)]
        # Defects make equal local words have different complete continuations.
        for i in rng.choice(length, int(rng.integers(1, 4)), replace=False):
            colors[int(i)] = int(rng.integers(k))
    else:
        colors = [int(rng.integers(k))]
        persistence = float(rng.uniform(.15, .75))
        for _ in range(length - 1):
            colors.append(colors[-1] if rng.random() < persistence else int(rng.integers(k)))
    return colors


def _catalog(loop, colors, qlo, qhi, witness):
    """Enumerate geometry-identifying crops; no seed-specific rules are stored."""
    L, k = len(loop), len(set(colors))
    geometry_groups, word_answers = defaultdict(list), defaultdict(set)
    entries = []
    for q in range(qlo, min(qhi, L - 2) + 1):
        for start in range(L):
            for direction in (-1, 1):
                ids = [(start + direction * i) % L for i in range(q)]
                arc_colors = [colors[i] for i in ids]
                if len(set(arc_colors)) != k:
                    continue
                word = _word(arc_colors)
                role_map = {v: i for i, v in enumerate(dict.fromkeys(arc_colors))}
                full = tuple(role_map[colors[(start + direction * i) % L]] for i in range(L))
                word_answers[q, word].add(full)
                if not _is_crop(loop, ids):
                    continue
                points = [loop[i] for i in ids]
                key = q, word, _shape(points)
                entry = (start, direction, points, arc_colors, full, key)
                geometry_groups[key].append(entry)
                entries.append(entry)
    return [e for e in entries if len(geometry_groups[e[5]]) == 1
            and (not witness or (len(word_answers[e[5][0], e[5][1]]) > 1
                                 and e[3] != e[3][::-1]))]


def _tight(points):
    p = np.asarray(points, dtype=int)
    return p - p.min(axis=0)


def _pack(rng, loop, loop_colors, query, query_colors, bg):
    loop, query = _tight(loop), _tight(query)
    shapes = [loop.max(axis=0) + 1, query.max(axis=0) + 1]
    axis = int(rng.integers(2))
    order = list(map(int, rng.permutation(2)))
    gap_max = min(4, 30 - int(shapes[0][axis] + shapes[1][axis]))
    gap = int(rng.integers(1, gap_max + 1))
    base = np.maximum(shapes[0], shapes[1])
    base[axis] = shapes[0][axis] + shapes[1][axis] + gap
    canvas = [min(30, int(s) + int(rng.integers(0, min(6, 31 - int(s))))) for s in base]
    while canvas[0] * canvas[1] <= 2 * (len(loop) + len(query)):
        grow = 0 if canvas[0] <= canvas[1] else 1
        if canvas[grow] == 30:
            grow = 1 - grow
        canvas[grow] += 1
    offsets = [np.zeros(2, dtype=int), np.zeros(2, dtype=int)]
    leading = int(rng.integers(canvas[axis] - int(base[axis]) + 1))
    offsets[order[0]][axis] = leading
    offsets[order[1]][axis] = leading + shapes[order[0]][axis] + gap
    for i in range(2):
        offsets[i][1 - axis] = int(rng.integers(canvas[1 - axis] - int(shapes[i][1 - axis]) + 1))
    grid = np.full(canvas, bg, dtype=np.int8)
    for points, colors, offset in zip((loop, query), (loop_colors, query_colors), offsets):
        for point, color in zip(points + offset, colors):
            grid[tuple(point)] = color
    return grid


def _distinct_canvas(grid, bg, seen):
    if _key(grid) not in seen:
        return grid
    # At most thirteen previous inputs exist. Changing only outer background
    # and common translation keeps both objects and the exact crop unchanged.
    rr, cc = np.nonzero(grid != bg)
    tight = grid[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    bh, bw = tight.shape
    ink = int(np.sum(tight != bg))
    for h in range(bh, 31):
        for w in range(bw, 31):
            if h * w <= 2 * ink:
                continue
            for r in range(h - bh + 1):
                for c in range(w - bw + 1):
                    candidate = np.full((h, w), bg, dtype=np.int8)
                    candidate[r:r + bh, c:c + bw] = tight
                    if _key(candidate) not in seen:
                        return candidate
    raise RuntimeError("no distinct legal canvas")


def _make_pair(rng, length_range, qrange, witness, need_above, plan, seen):
    # Geometry and coloring are resampled together under bounded rejection.
    # A fixed RNG stream is a tested procedural safety net, not a bitmap bank;
    # it follows exactly the same rule and acceptance checks as normal sampling.
    for attempt in range(160):
        if attempt == 96:
            rng = np.random.default_rng(20261015)
        length = 2 * int(rng.integers(length_range[0] // 2, length_range[1] // 2 + 1))
        loop = _make_cycle(rng, length)
        if loop is None:
            continue
        k = int(rng.integers(2, 5))
        colors = _coloring(rng, length, k, witness)
        if len(set(colors)) != k:
            continue
        entries = _catalog(loop, colors, *qrange, witness)
        if not entries:
            continue
        for j in rng.permutation(len(entries)):
            start, direction, arc, arc_colors, _, _ = entries[int(j)]
            orientations = []
            for transformed in _transforms(arc):
                if transformed[0, 0] >= transformed[-1, 0]:
                    continue
                above = int(transformed[:, 0].min()) < int(transformed[0, 0])
                if need_above is None or above == need_above:
                    orientations.append(transformed)
            if not orientations:
                continue
            query = orientations[int(rng.integers(len(orientations)))]
            transformed_loop = _transforms(loop)[int(rng.integers(8))]
            bg = plan['background'] if plan['stable'] else int(rng.integers(10))
            available = [c for c in range(10) if c != bg]
            if plan['stable']:
                reference_palette = plan['reference'][:k]
                query_palette = plan['query'][:k]
            else:
                reference_palette = list(map(int, rng.choice(available, k, replace=False)))
                query_palette = list(map(int, rng.choice(available, k, replace=False)))
            if witness and reference_palette == query_palette:
                query_palette = query_palette[1:] + query_palette[:1]
            lc = [reference_palette[c] for c in colors]
            qc = [query_palette[c] for c in arc_colors]
            expected = tuple(query_palette[colors[(start + direction * i) % length]] for i in range(length))
            # The crop selects the cyclic phase, not the reference's topmost
            # pixel, its row order, or its lower bookmark endpoint.
            if witness:
                root = min(range(length), key=lambda i: tuple(transformed_loop[i]))
                rooted = [tuple(query_palette[colors[(root + d * i) % length]] for i in range(length))
                          for d in (-1, 1)]
                reverse_start = (start + direction * (len(arc) - 1)) % length
                wrong_end = tuple(query_palette[colors[(reverse_start - direction * i) % length]]
                                  for i in range(length))
                row_order = sorted(range(length), key=lambda i: tuple(transformed_loop[i]))
                row_word = tuple(query_palette[colors[i]] for i in row_order)
                if expected in rooted or expected == wrong_end or expected == row_word:
                    continue
            inp = _distinct_canvas(_pack(rng, transformed_loop, lc, query, qc, bg), bg, seen)
            seen.add(_key(inp))
            return {'input': inp, 'output': np.asarray(expected, dtype=np.int8).reshape(-1, 1)}
    raise RuntimeError("bounded necklace sampling exhausted; no ambiguous crop is accepted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    plan = {'stable': bool(rng.integers(2)), 'background': palette[0],
            'reference': palette[1:5], 'query': palette[5:9]}
    shift = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            if shift == 0:  # more unseen necklace cells, at familiar crop scale
                lengths = (16, 24) if split == 'train' else (26, 30)
                crops = (6, 14)
            else:  # less visible evidence, at the same necklace scale
                lengths = (24, 30)
                crops = (10, 17) if split == 'train' else (5, 9)
            witness = index == 0 or (split == 'train' and index == 1)
            # Explicitly distinguish the higher endpoint from the uppermost
            # interior cell even in a two-demonstration puzzle.
            need_above = True if split == 'train' and index == 0 else None
            pair = _make_pair(rng, lengths, crops, witness, need_above, plan, seen)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a, bg, loop, query = _read_objects(pair['input'])
    matches = _registrations(a, loop, query)
    if len(matches) != 1:
        raise ValueError("nonunique directed crop registration")
    start, direction, mapping, sequence = matches[0]
    qcolors = tuple(int(a[p]) for p in query)
    row_word = tuple(mapping[int(a[p])] for p in sorted(loop))
    root = 0  # _read_objects roots its cycle at the lexicographic minimum
    roots = [tuple(mapping[int(a[loop[(root + d * i) % len(loop)]])]
                   for i in range(len(loop))) for d in (-1, 1)]
    wrong_start = (start + direction * (len(query) - 1)) % len(loop)
    wrong_end = tuple(mapping[int(a[loop[(wrong_start - direction * i) % len(loop)]])]
                      for i in range(len(loop)))
    return dict(background=bg, length=len(loop), clue=len(query), colors=len(mapping),
                loop_shape=_object_shape(loop), query_shape=_object_shape(query),
                above=min(r for r, c in query) < query[0][0],
                geometry_needed=len({m[3] for m in _registrations(a, loop, query, False)}) > 1,
                direction_needed=sequence != wrong_end, phase_needed=sequence not in roots,
                row_failure=sequence != row_word, nonpalindrome=qcolors != qcolors[::-1],
                renamed=any(s != t for s, t in mapping.items()),
                recurrence=len(set(sequence)) < len(_runs(sequence)),
                input_shape=a.shape, output=sequence)


def _runs(sequence):
    return [v for i, v in enumerate(sequence) if i == 0 or v != sequence[i - 1]]


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _count(nt, 'num_train', 2, 10)
    _count(nq, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("training count must exceed test count")
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError("pairs must have exactly input and output")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            if _key(inp) in seen:
                raise ValueError("duplicate input")
            seen.add(_key(inp))
            f = _features(pair)
            expected = np.asarray(f['output'], dtype=int).reshape(-1, 1)
            if not np.array_equal(out, expected):
                raise ValueError("output does not follow the registered cycle")
            if not np.array_equal(solve(inp.copy()), out):
                raise ValueError("input-only solve disagrees")
            features[split].append(f)
    for split in ('train', 'test'):
        if not any(all(f[k] for k in ('geometry_needed', 'direction_needed', 'phase_needed',
                                     'row_failure', 'nonpalindrome')) for f in features[split]):
            raise ValueError("missing a joint crop/traversal witness")
    if not any(f['above'] and f['nonpalindrome'] for f in features['train']):
        raise ValueError("training must teach upper endpoint versus uppermost cell")
    if not any(f['renamed'] for f in features['train']):
        raise ValueError("training must expose color renaming")


def _puzzle_key(puzzle):
    return tuple((s, _key(p['input']), _key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded actual-pixel checks on 42 seeds, counts, symmetry, and corruption."""
    puzzles, records = [], []
    for seed in range(42):
        nt = 2 + seed % 9
        nq = min(1 + (seed // 9) % 4, nt - 1)
        puzzle = generate_puzzle(seed, nt, nq)
        puzzles.append(puzzle)
        records.extend(_features(pair) for s in puzzle for pair in puzzle[s])
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0, 2, 1))
    assert {len(p['train']) for p in puzzles} == set(range(2, 11))
    assert {len(p['test']) for p in puzzles} == set(range(1, 5))
    assert len({f['loop_shape'] for f in records}) >= 80
    assert len({f['query_shape'] for f in records}) >= 50
    assert len({f['input_shape'] for f in records}) >= 60
    assert {f['length'] for f in records} == set(range(16, 31, 2))
    assert len({f['clue'] for f in records}) >= 10
    assert {f['colors'] for f in records} == {2, 3, 4}
    assert {f['background'] for f in records} == set(range(10))
    assert sum(f['geometry_needed'] for f in records) >= 100
    assert sum(f['above'] for f in records) >= 42
    assert sum(f['renamed'] for f in records) >= 100
    # A global D4 operation can exchange the bookmark's higher endpoint.
    # Re-rooting at its old end reverses the clue and the complementary arc
    # separately: reversing the entire output would put the seam elsewhere.
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        a, _, loop, query = _read_objects(pair['input'])
        assert pair['output'][:len(query), 0].tolist() == [int(a[p]) for p in query]
        markers = np.zeros_like(a)
        markers[query[0]], markers[query[-1]] = 1, 2
        for rotation in range(4):
            for reflection in (False, True):
                transformed = np.rot90(a, rotation)
                marked = np.rot90(markers, rotation)
                if reflection:
                    transformed, marked = np.fliplr(transformed), np.fliplr(marked)
                r0 = int(np.argwhere(marked == 1)[0, 0])
                r1 = int(np.argwhere(marked == 2)[0, 0])
                if r0 == r1:
                    try:
                        solve(transformed)
                    except ValueError:
                        continue
                    raise AssertionError("equal-row bookmark ends accepted")
                out = pair['output']
                expected = out if r0 < r1 else np.concatenate((out[:len(query)][::-1],
                                                               out[len(query):][::-1]))
                assert np.array_equal(solve(transformed), expected)
    # A valid smaller family member outside the favored sampling bands.
    square = _rectangle(3, 3)
    small_colors = [1, 1, 1, 1, 1, 1, 2, 2]
    small_arc = [square[i] for i in (1, 0, 7, 6, 5)]
    small = _pack(np.random.default_rng(7), square, small_colors, small_arc,
                  [3, 3, 4, 4, 3], 0)
    assert solve(small)[:, 0].tolist() == [3, 3, 4, 4, 3, 3, 3, 3]
    # A congruent subpath need not be an exact rectangular crop: this six-bead
    # arc's rectangle includes the other two beads of the eight-bead reference.
    assert not _is_crop(square, list(range(6)))
    bad_crop = _pack(np.random.default_rng(8), square, [1, 2] * 4, square[:6],
                     [3, 4] * 3, 0)
    try:
        solve(bad_crop)
    except ValueError:
        pass
    else:
        raise AssertionError("nonrectangular fragment accepted as an exact crop")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    fallback_plan = {'stable': True, 'background': 0,
                     'reference': [1, 2, 3, 4], 'query': [5, 6, 7, 8]}
    for stable in (False, True):
        fallback_plan['stable'] = stable
        for lengths, crops, training in (((16, 24), (6, 14), True),
                                          ((26, 30), (6, 14), False),
                                          ((24, 30), (10, 17), True),
                                          ((24, 30), (5, 9), False)):
            for witness, above in ((False, None), (True, None), *([(True, True)] if training else [])):
                pair = _make_pair(np.random.default_rng(20261015), lengths, crops,
                                  witness, above, fallback_plan, set())
                assert np.array_equal(solve(pair['input']), pair['output'])
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]]
           for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=len(records),
                loop_shapes=len({f['loop_shape'] for f in records}),
                query_shapes=len({f['query_shape'] for f in records}),
                input_shapes=len({f['input_shape'] for f in records}),
                geometry_witnesses=sum(f['geometry_needed'] for f in records),
                above_endpoint=sum(f['above'] for f in records))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
