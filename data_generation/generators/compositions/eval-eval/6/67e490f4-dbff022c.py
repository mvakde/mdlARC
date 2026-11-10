"""Cavity coalitions: map each closed cavity's owner, then vote by cavity shape.

The key and cavities are read from the original input. There are no seed-dependent
solving conventions. Construction labels are used only to build/check sampled
answers, and are never available to solve().
"""
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np

_STEPS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_STEPS8 = _STEPS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid dimensions'
    assert np.issubdtype(a.dtype, np.integer), 'noninteger cells'
    assert np.all((0 <= a) & (a <= 9)), 'invalid colors'
    return a.astype(np.int8, copy=True)


def _normalize(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-y, c-x) for r, c in cells))


@lru_cache(maxsize=8192)
def _rotations(shape):
    s, results = _normalize(shape), set()
    for _ in range(4):
        results.add(s)
        s = _normalize((c, -r) for r, c in s)
    return tuple(sorted(results))


@lru_cache(maxsize=8192)
def _variants(shape):
    s = _normalize(shape)
    return tuple(sorted(set(_rotations(s)) | set(_rotations(_normalize((r, -c) for r, c in s)))))


def _canonical(cells):
    return min(_variants(_normalize(cells)))


def _extent(shape):
    return max(r for r, c in shape)+1, max(c for r, c in shape)+1


def _components(mask, diagonal=False):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _STEPS8 if diagonal else _STEPS4:
                yy, xx = y+dy, x+dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(tuple(cells))
    return result


def _regions(background_mask):
    h, w = background_mask.shape
    return [cells for cells in _components(background_mask)
            if not any(r in (0, h-1) or c in (0, w-1) for r, c in cells)]


def _read(input_grid):
    """Recover key, contour owners and exact cavity geometry from input alone."""
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, 'ambiguous background'
    bg = int(colors[np.argmax(counts)])
    h, w = g.shape
    clusters = _components(g != bg, diagonal=True)
    candidates = []
    for cells in clusters:
        rs, cs = zip(*cells)
        y, x, b, e = min(rs), min(cs), max(rs), max(cs)
        sh, sw = b-y+1, e-x+1
        if len(cells) != sh*sw or len({int(g[q]) for q in cells}) < 2:
            continue
        lanes = []
        if sh == 2 and 3 <= sw <= 7:
            if y == 0:
                lanes.append(('top', g[y, x:e+1], g[b, x:e+1]))
            if b == h-1:
                lanes.append(('bottom', g[b, x:e+1], g[y, x:e+1]))
        if sw == 2 and 3 <= sh <= 7:
            if x == 0:
                lanes.append(('left', g[y:b+1, x], g[y:b+1, e]))
            if e == w-1:
                lanes.append(('right', g[y:b+1, e], g[y:b+1, x]))
        for side, sources, fills in lanes:
            sources, fills = tuple(map(int, sources)), tuple(map(int, fills))
            if len(set(sources)) == len(sources):
                candidates.append((side, dict(zip(sources, fills)), frozenset(cells)))
    assert len(candidates) == 1, 'expected one unambiguous local edge key'
    side, mapping, key_cells = candidates[0]
    ids = np.full(g.shape, -1, dtype=int)
    objects = []
    for cells in clusters:
        if frozenset(cells) == key_cells:
            continue
        ink = {int(g[q]) for q in cells}
        assert len(ink) == 1, 'non-key ink must be monochromatic'
        index = len(objects)
        for q in cells:
            ids[q] = index
        objects.append({'color': ink.pop(), 'cells': cells, 'holes': []})
    holes = []
    for cells in _regions(g == bg):
        boundary = set()
        for r, c in cells:
            for dr, dc in _STEPS4:
                q = r+dr, c+dc
                if g[q] != bg:
                    boundary.add(q)
        owners = {int(ids[q]) for q in boundary}
        assert len(owners) == 1 and -1 not in owners, 'ambiguous cavity owner'
        owner = owners.pop()
        color = objects[owner]['color']
        assert {int(g[q]) for q in boundary} == {color}
        hole = {'cells': cells, 'shape': _canonical(cells), 'raw': _normalize(cells),
                'owner': owner, 'color': color, 'active': color in mapping}
        objects[owner]['holes'].append(len(holes))
        holes.append(hole)
    assert objects and any(hole['active'] for hole in holes), 'no eligible cavities'
    return {'grid': g, 'background': bg, 'side': side, 'mapping': mapping,
            'key_cells': key_cells, 'objects': objects, 'holes': holes}


def _signature(hole, equivalence):
    if equivalence == 'area':
        return len(hole['cells'])
    if equivalence == 'bbox':
        return tuple(sorted(_extent(hole['raw'])))
    if equivalence == 'rotation':
        return min(_rotations(hole['raw']))
    if equivalence == 'translation':
        return hole['raw']
    return hole['shape']


def _prediction(data, policy='rule'):
    """The rule, plus explicit competing policies for demonstration checks."""
    g, mapping = data['grid'], data['mapping']
    out = g.copy()
    eligible = [h for h in data['holes'] if h['active'] or policy == 'include_unlisted']
    if policy == 'own_lookup':
        for hole in eligible:
            for q in hole['cells']:
                out[q] = mapping[hole['color']]
        return out, {}
    equivalence = policy if policy in ('area', 'bbox', 'rotation', 'translation') else 'free'
    ballots, visited = defaultdict(Counter), set()
    for hole in eligible:
        shape = _signature(hole, equivalence)
        if policy == 'contour_votes':
            token = shape, hole['owner']
            if token in visited:
                continue
            visited.add(token)
        color = hole['color']
        if policy not in ('raw_then_map', 'raw_no_key'):
            color = mapping.get(color, color)
        if policy == 'recursive':
            walked = set()
            while color in mapping and color not in walked:
                walked.add(color)
                color = mapping[color]
        ballots[shape][color] += 1
    winners = {}
    for shape, counts in ballots.items():
        high = max(counts.values())
        best = [c for c, n in counts.items() if n == high]
        assert len(best) == 1, 'tied shape ballot'
        winner = best[0]
        winners[shape] = mapping[winner] if policy == 'raw_then_map' else winner
    for hole in eligible:
        for q in hole['cells']:
            out[q] = winners[_signature(hole, equivalence)]
    return out, dict(ballots)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply one key lookup to each ballot, pool by D4 hole shape, then fill."""
    return _prediction(_read(input_grid))[0]


def _fails(data, output, policy):
    try:
        alternative = _prediction(data, policy)[0]
    except AssertionError:
        return True
    return not np.array_equal(alternative, output)


def _diagonal_leak_witness(data):
    """A filled cavity would escape if background used eight-neighbor flood."""
    g = data['grid']
    leaked = set()
    for cells in _components(g == data['background'], diagonal=True):
        if any(r in (0, g.shape[0]-1) or c in (0, g.shape[1]-1) for r, c in cells):
            leaked.update(cells)
    return any(h['active'] and any(q in leaked for q in h['cells']) for h in data['holes'])


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _hole_free(shape):
    h, w = _extent(shape)
    mask = np.ones((h+2, w+2), dtype=bool)
    for r, c in shape:
        mask[r+1, c+1] = False
    return len(_components(mask)) == 1


@lru_cache(maxsize=1)
def _small_shapes():
    # Enumerate small free polyominoes instead of fixing an anchor vocabulary.
    layer = {((0, 0),)}
    pool = set(layer)
    for _ in range(5):
        nxt = set()
        for shape in sorted(layer):
            cells = set(shape)
            frontier = {(r+dr, c+dc) for r, c in cells for dr, dc in _STEPS4} - cells
            for q in sorted(frontier):
                nxt.add(_canonical(cells | {q}))
        layer = nxt
        pool.update(s for s in layer if _hole_free(s))
    return tuple(sorted(pool))


def _shape_pool(rng):
    pool = set(_small_shapes())
    for n in range(7, 10):
        pool.add(_canonical(tuple((0, i) for i in range(n))))
    for _ in range(35):
        cells = {(0, 0)}
        for _ in range(int(rng.integers(6, 10))-1):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _STEPS4} - cells)
            cells.add(_choice(rng, frontier))
        shape = _canonical(cells)
        if max(_extent(shape)) <= 6 and _hole_free(shape):
            pool.add(shape)
    return tuple(sorted(pool))


def _dilate(mask, diagonal=False):
    result = mask.copy()
    for dr, dc in _STEPS8 if diagonal else _STEPS4:
        src_y = slice(max(0, -dr), min(mask.shape[0], mask.shape[0]-dr))
        src_x = slice(max(0, -dc), min(mask.shape[1], mask.shape[1]-dc))
        dst_y = slice(max(0, dr), min(mask.shape[0], mask.shape[0]+dr))
        dst_x = slice(max(0, dc), min(mask.shape[1], mask.shape[1]+dc))
        result[dst_y, dst_x] |= mask[src_y, src_x]
    return result


def _frame(rng, shapes, color, winners, bg, *, open_frame=False, thin=False):
    """Build a single contour with exact labeled holes; preserve their geometry."""
    cols = len(shapes) if len(shapes) <= 2 or rng.random() < .55 else 2
    rows = [shapes[i:i+cols] for i in range(0, len(shapes), cols)]
    thickness = 1 if thin or open_frame else int(rng.choice([1, 1, 1, 2]))
    gap = 1 if thin else int(rng.choice([1, 1, 2]))
    row_heights = [max(_extent(s)[0] for s in row) for row in rows]
    row_widths = [sum(_extent(s)[1] for s in row)+gap*(len(row)-1) for row in rows]
    h = 2*thickness+sum(row_heights)+gap*(len(rows)-1)
    w = 2*thickness+max(row_widths)
    labels = np.zeros((h, w), dtype=np.int16)
    fills = {}
    y, label = thickness, 1
    for row, rh, rw in zip(rows, row_heights, row_widths):
        x = thickness+int(rng.integers(max(row_widths)-rw+1))
        for shape in row:
            sh, sw = _extent(shape)
            yy = y+int(rng.integers(rh-sh+1))
            for r, c in shape:
                labels[yy+r, x+c] = label
            fills[label] = winners.get(_canonical(shape), bg)
            label += 1
            x += sw+gap
        y += rh+gap
    holes = labels != 0
    ink = holes.copy()
    for _ in range(thickness):
        ink = _dilate(ink, diagonal=not thin and rng.random() < .6)
    ink &= ~holes
    intended = {frozenset(zip(*np.nonzero(labels == i))) for i in fills}
    actual = {frozenset((r-1, c-1) for r, c in cells)
              for cells in _regions(~np.pad(ink, 1))}
    if actual != intended or len(_components(ink, diagonal=True)) != 1:
        # A solid shared wall removes unintended pockets in deep concavities.
        ink = ~holes
    if open_frame:
        assert len(shapes) == 1
        r, c = min(zip(*np.nonzero(holes)))
        ink[:r+1, c] = False
        assert not _regions(~np.pad(ink, 1)), 'opening failed to leak'
        labels[:] = 0
        fills = {}
    # Rotation preserves handedness; reflection of holes was sampled separately.
    turns = int(rng.integers(4))
    ink, labels = np.rot90(ink, turns).copy(), np.rot90(labels, turns).copy()
    assert len(_components(ink, diagonal=True)) == 1
    return {'ink': ink, 'labels': labels, 'color': color, 'fills': fills}


def _mapping(rng, bg, lesson=None, forbidden_winner=None):
    colors = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    if lesson == 1 and colors[0] == forbidden_winner:
        colors[0], colors[-1] = colors[-1], colors[0]
    a, b, c, d, e = colors[:5]
    n = int(rng.integers(3, 8))
    if lesson == 0:
        sources = [a, b, c] + [x for x in colors[3:] if x != d][:n-3]
        mapping = {a: b, b: d, c: d}
    elif lesson == 1:
        sources = [a, b, c] + [x for x in colors[3:] if x not in (d, e)][:n-3]
        mapping = {a: a, b: d, c: e}
    else:
        sources = colors[:n]
        destinations = _choice(rng, (colors[:3], colors[-3:], [colors[1], colors[3], colors[-1]], colors))
        mapping = {s: int(_choice(rng, destinations)) for s in sources}
        # Guarantee two effective ballot colors, while allowing shared fills/chains.
        mapping[sources[0]] = colors[-1]
        mapping[sources[1]] = colors[0]
    for s in sources:
        if s not in mapping:
            mapping[s] = int(_choice(rng, colors))
    inert = _choice(rng, [x for x in colors if x not in mapping])
    return mapping, (a, b, c, d, e), inert


def _vote_specs(rng, shapes, mapping, lesson=None):
    sources = list(mapping)
    if lesson is not None:
        a, b, c = sources[:3]
        raw = [{a: 3, b: 2, c: 2}, {a: 3, b: 1} if lesson == 0 else {b: 3, c: 1}]
    else:
        raw = []
        for i, shape in enumerate(shapes):
            large = max(_extent(shape)) >= 5 or len(shape) >= 8
            patterns = ((2, 1), (3, 2), (3, 2, 2)) if i == 0 else ((1,), (2,), (2, 1), (3, 1), (3, 2, 1))
            if large or len(shapes) >= 3:
                patterns = ((2, 1),) if i == 0 else ((1,), (2, 1))
            for _ in range(50):
                pattern = _choice(rng, patterns)
                selected = list(map(int, rng.choice(sources, len(pattern), replace=False)))
                counts = dict(zip(selected, pattern))
                effective = Counter()
                for s, n in counts.items():
                    effective[mapping[s]] += n
                high = max(effective.values())
                if sum(v == high for v in effective.values()) == 1 and (i != 0 or len(effective) >= 2):
                    raw.append(counts)
                    break
            else:
                # Always feasible: two sources with different destinations, 2 vs 1.
                s = sources[0]
                t = next(t for t in sources if mapping[t] != mapping[s])
                raw.append({s: 2, t: 1})
    winners = {}
    tokens = defaultdict(list)
    for shape, counts in zip(shapes, raw):
        effective = Counter()
        for color, n in counts.items():
            effective[mapping[color]] += n
            for _ in range(n):
                if lesson == 0 and shape == shapes[0]:
                    mirrored = _normalize((r, -c) for r, c in shape)
                    variants = _rotations(shape if color == list(mapping)[0] else mirrored)
                else:
                    variants = _variants(shape)
                tokens[color].append(_choice(rng, variants))
        winner = max(effective, key=effective.get)
        assert list(effective.values()).count(effective[winner]) == 1
        winners[shape] = winner
    return tokens, winners


def _make_frames(rng, shapes, mapping, inert, bg, lesson=None):
    tokens, winners = _vote_specs(rng, shapes, mapping, lesson)
    frames = []
    a = list(mapping)[0]
    for color, holes in tokens.items():
        holes = list(holes)
        if lesson == 0 and color == a:
            # Three solo anchor contours vs two coalition contours: counting
            # contours instead of their cavities reverses the anchor election.
            anchors = [s for s in holes if _canonical(s) == shapes[0]]
            companions = [s for s in holes if _canonical(s) != shapes[0]]
            groups = [[s] for s in anchors] + [companions]
        elif lesson == 0:
            groups = [holes]  # b has two anchor holes and one companion hole.
        else:
            rng.shuffle(holes)
            groups = []
            while holes:
                size = int(rng.choice([1, 1, 2, 3]))
                groups.append(holes[:size])
                holes = holes[size:]
        for group in groups:
            if group:
                frames.append(_frame(rng, group, color, winners, bg,
                                     thin=lesson is not None and len(group) == 1))
    # Unlisted holes can exactly match active holes but are neither votes nor recipients.
    if lesson is not None or rng.random() < .65:
        extras = [_choice(rng, _variants(_choice(rng, shapes))) for _ in range(int(rng.integers(1, 3)))]
        frames.append(_frame(rng, extras, inert, {}, bg, thin=lesson is not None))
    # The opening is an enclosure witness, not irrelevant added noise.
    if lesson is not None or rng.random() < .45:
        frame_shape = _choice(rng, _variants(shapes[0]))
        frames.append(_frame(rng, [frame_shape], a, {}, bg, open_frame=True))
    return frames, winners


def _put_key(grid, side, mapping, start):
    for i, (source, fill) in enumerate(mapping.items()):
        p = start+i
        if side == 'top':
            grid[0, p], grid[1, p] = source, fill
        elif side == 'bottom':
            grid[-1, p], grid[-2, p] = source, fill
        elif side == 'left':
            grid[p, 0], grid[p, 1] = source, fill
        else:
            grid[p, -1], grid[p, -2] = source, fill


def _rectangle_positions(available, h, w):
    rh, rw = available.shape[0]-h+1, available.shape[1]-w+1
    if rh <= 0 or rw <= 0:
        return np.empty((0, 2), dtype=int)
    valid = np.ones((rh, rw), dtype=bool)
    # Summed-area table: test a whole bounding box without a slot lattice.
    sums = np.pad((~available).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    valid &= sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w] == 0
    return np.argwhere(valid)


def _pack(rng, frames, mapping, side, bg):
    area = sum((f['ink'].shape[0]+1)*(f['ink'].shape[1]+1) for f in frames)
    maxdim = max(max(f['ink'].shape) for f in frames)
    for attempt in range(60):
        if attempt >= 20:
            h = w = 30
        else:
            scale = int(np.clip(np.sqrt(area*1.55)+3, 10, 30))
            h = int(np.clip(scale+rng.integers(-5, 4), max(8, maxdim), 30))
            w = int(np.clip(scale+rng.integers(-5, 4), max(8, maxdim), 30))
            if rng.random() < .2:
                h = w = max(h, w)
        if h*w < area+3*(len(mapping)+1):
            continue
        grid = np.full((h, w), bg, dtype=np.int8)
        axis = w if side in ('top', 'bottom') else h
        start = int(_choice(rng, [0, axis-len(mapping)])) if rng.random() < .35 else int(rng.integers(axis-len(mapping)+1))
        _put_key(grid, side, mapping, start)
        available = ~_dilate(grid != bg, diagonal=True)
        placed = []
        order = list(map(int, rng.permutation(len(frames))))
        order.sort(key=lambda j: (-max(frames[j]['ink'].shape), -frames[j]['ink'].size))
        for index in order:
            frame = frames[index]
            orientation = int(rng.integers(2))
            for turn in (orientation, 1-orientation):
                ink, labels = np.rot90(frame['ink'], turn), np.rot90(frame['labels'], turn)
                positions = _rectangle_positions(available, *ink.shape)
                if len(positions):
                    break
            else:
                break
            fh, fw = ink.shape
            edge_positions = [(r, c) for r, c in positions if r == 0 or c == 0 or r+fh == h or c+fw == w]
            choices = edge_positions if edge_positions and rng.random() < .35 else positions
            y, x = map(int, _choice(rng, choices))
            grid[y:y+fh, x:x+fw][ink] = frame['color']
            available[max(0, y-1):min(h, y+fh+1), max(0, x-1):min(w, x+fw+1)] = False
            placed.append((y, x, labels.copy(), frame['fills']))
        else:
            colors, counts = np.unique(grid, return_counts=True)
            if int(colors[np.argmax(counts)]) != bg or np.count_nonzero(counts == counts.max()) != 1:
                continue
            out = grid.copy()
            for y, x, labels, fills in placed:
                patch = out[y:y+labels.shape[0], x:x+labels.shape[1]]
                for label, color in fills.items():
                    patch[labels == label] = color
            return {'input': grid, 'output': out}
    return None


def _make_pair(rng, shapes, side, bg, lesson=None, forbidden_winner=None):
    # Retry geometry, not the semantic rule or the two mandatory witnesses.
    for _ in range(24):
        mapping, _, inert = _mapping(rng, bg, lesson, forbidden_winner)
        frames, winners = _make_frames(rng, shapes, mapping, inert, bg, lesson)
        if any(max(f['ink'].shape) > 28 for f in frames):
            continue
        items = list(mapping.items())
        rng.shuffle(items)
        pair = _pack(rng, frames, dict(items), side, bg)
        if pair is None:
            continue
        data = _read(pair['input'])
        if not np.array_equal(_prediction(data)[0], pair['output']):
            raise AssertionError('planned holes differ from recovered scene')
        if lesson == 0 and not all(_fails(data, pair['output'], p) for p in
                                  ('raw_then_map', 'contour_votes', 'rotation', 'area', 'bbox', 'recursive')):
            continue
        return pair, winners
    raise RuntimeError('bounded scene packing exhausted')


def _counts(num_train, num_test):
    for name, n, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= n <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _grid_key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    pool = _shape_pool(rng)
    anchors = [s for s in pool if 4 <= len(s) <= 6 and max(_extent(s)) <= 4
               and len(_variants(s)) > len(_rotations(s))
               and any(t != s and len(t) == len(s) and sorted(_extent(t)) == sorted(_extent(s)) for t in pool)]
    anchor = _choice(rng, anchors)
    companion = _choice(rng, [s for s in pool if s != anchor and len(s) == len(anchor)
                              and sorted(_extent(s)) == sorted(_extent(anchor))])
    sides = list(map(str, rng.permutation(['top', 'left', 'bottom', 'right'])))
    train_side_count = int(rng.integers(2, 4))
    novel_side = rng.random() < .55
    backgrounds = list(map(int, rng.permutation(10)))
    puzzle, seen, known = {'train': [], 'test': []}, set(), set()
    first_winner = None
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            lesson = i if split == 'train' and i < 2 else None
            if lesson is not None:
                shapes = [anchor, companion]
            else:
                k = int(rng.choice([1, 2, 3, 4], p=[.22, .37, .28, .13]))
                # Larger repertoires use smaller geometry to leave room for ballots.
                available = [s for s in pool if len(s) <= (6 if k >= 3 else 9)]
                shapes = [_choice(rng, available)]
                if split == 'test':
                    shapes = [_choice(rng, [s for s in available if s not in known])]
                rest = [s for s in available if s != shapes[0]]
                shapes += [rest[int(j)] for j in rng.choice(len(rest), k-1, replace=False)]
            side = sides[i % train_side_count] if split == 'train' else (
                sides[-1] if novel_side else _choice(rng, sides[:min(num_train, train_side_count)]))
            bg = backgrounds[(i if split == 'train' else i+3) % 10]
            for _ in range(8):
                pair, winners = _make_pair(rng, shapes, side, bg, lesson, first_winner)
                key = _grid_key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train':
                known.update(shapes)
                if i == 0:
                    first_winner = winners[anchor]
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
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape, 'canvas size changed'
            key = _grid_key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            data = _read(inp)
            expected, ballots = _prediction(data)
            assert np.array_equal(expected, out), 'wrong cavity vote/fill'
            assert np.array_equal(inp[inp != data['background']], out[inp != data['background']]), 'ink/key changed'
            assert np.any(inp != out), 'no visible transformation'
            active = [h for h in data['holes'] if h['active']]
            assert sum(map(lambda v: sum(v.values()), ballots.values())) == len(active), 'not one vote per cavity'
            assert all(out[q] == data['background'] for h in data['holes'] if not h['active'] for q in h['cells'])
            data['ballots'], data['output'] = ballots, out
            analyses[split].append(data)
    train = analyses['train']
    # These witnesses must be recoverable from returned arrays, not sampler flags.
    for policy in ('own_lookup', 'raw_no_key', 'raw_then_map', 'contour_votes', 'area', 'bbox',
                   'rotation', 'translation', 'recursive', 'include_unlisted'):
        assert any(_fails(d, d['output'], policy) for d in train), 'missing lesson: '+policy
    assert any(any(not o['holes'] and o['color'] in d['mapping'] for o in d['objects']) for d in train), 'no open mapped contour lesson'
    assert any(_diagonal_leak_witness(d) for d in train), 'no four-neighbor leakage lesson'
    assert any(any(not h['active'] for h in d['holes']) for d in train), 'no inert cavity lesson'
    assert any(any(d['mapping'][h['color']] == h['color'] for h in d['holes'] if h['active']) for d in train), 'no self-map lesson'
    assert any(any(len(v) >= 3 and 2*max(v.values()) <= sum(v.values()) for v in d['ballots'].values()) for d in train), 'no nonmajority plurality lesson'
    assert any(any(len(o['holes']) > 1 and len({int(d['output'][d['holes'][j]['cells'][0]]) for j in o['holes']}) > 1 for o in d['objects']) for d in train), 'no mixed-shape contour lesson'
    history = defaultdict(set)
    for d in train:
        for h in d['holes']:
            if h['active']:
                history[h['shape']].add(int(d['output'][h['cells'][0]]))
    assert any(len(values) >= 2 for values in history.values()), 'shape-color dictionary not refuted'
    known = set(history)
    assert len({d['background'] for d in train}) >= 2
    assert len({d['side'] for d in train}) >= 2
    for d in analyses['test']:
        assert any(h['active'] and h['shape'] not in known for h in d['holes']), 'missing novel-shape shift'


def _features(pair):
    d = _read(pair['input'])
    out, ballots = _prediction(d)
    holes, objects = d['holes'], d['objects']
    return {'dimensions': out.shape, 'background': d['background'], 'side': d['side'],
            'key_length': len(d['mapping']), 'objects': len(objects), 'holes': len(holes),
            'classes': len(ballots), 'shapes': frozenset(h['shape'] for h in holes),
            'geometry': tuple(sorted((_extent(_normalize(o['cells'])), len(o['holes'])) for o in objects)),
            'ballots': tuple(sorted(tuple(sorted(v.values(), reverse=True)) for v in ballots.values())),
            'multi': any(len(o['holes']) > 1 for o in objects),
            'mixed': any(len({holes[j]['shape'] for j in o['holes']}) > 1 for o in objects),
            'diagonal': any(o['holes'] and len(_components(_object_mask(d['grid'].shape, o['cells']))) > 1 for o in objects),
            'border': any(any(r in (0, out.shape[0]-1) or c in (0, out.shape[1]-1) for r, c in o['cells']) for o in objects),
            'open': any(not o['holes'] and o['color'] in d['mapping'] for o in objects),
            'pooling_reversal': _fails(d, out, 'raw_then_map')}


def _object_mask(shape, cells):
    mask = np.zeros(shape, dtype=bool)
    for q in cells:
        mask[q] = True
    return mask


def _puzzle_key(p):
    return tuple((_grid_key(np.asarray(pair['input'])), _grid_key(np.asarray(pair['output'])))
                 for split in ('train', 'test') for pair in p[split])


def self_test():
    """Bounded structural, semantic-witness and diversity checks on 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    features = []
    vocabulary = set()
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                original = np.asarray(pair['input']).copy()
                assert np.array_equal(solve(original), pair['output'])
                assert np.array_equal(original, pair['input']), 'solve mutated its input'
                f = _features(pair)
                features.append(f)
                vocabulary.update(f['shapes'])
    assert len({f['dimensions'] for f in features}) >= 35
    assert {f['background'] for f in features} == set(range(10))
    assert {f['side'] for f in features} == {'top', 'left', 'bottom', 'right'}
    assert {f['key_length'] for f in features} == set(range(3, 8))
    assert {f['classes'] for f in features} == {1, 2, 3, 4}
    assert len(vocabulary) >= 60
    assert len({f['objects'] for f in features}) >= 8
    assert len({f['holes'] for f in features}) >= 10
    assert len({f['geometry'] for f in features}) >= 150
    assert len({f['ballots'] for f in features}) >= 20
    assert max(len(s) for s in vocabulary) >= 9
    for flag in ('multi', 'mixed', 'diagonal', 'border', 'open', 'pooling_reversal'):
        assert sum(f[flag] for f in features) >= 20, 'weak coverage: '+flag
    # Geometry/color equivariance uses mathematical transformations, no examples
    # or hidden generation data in the input-only solver.
    pair = puzzles[0]['train'][0]
    palette = np.array([8, 5, 3, 6, 0, 9, 2, 7, 1, 4], dtype=np.int8)
    for turn in range(4):
        for mirror in (False, True):
            inp, out = [np.rot90(palette[np.asarray(pair[k])], turn) for k in ('input', 'output')]
            if mirror:
                inp, out = np.fliplr(inp), np.fliplr(out)
            assert np.array_equal(solve(inp), out)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(9173, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    d = _read(broken['test'][0]['input'])
    q = min(d['key_cells'])
    broken['test'][0]['output'][q] = d['background']
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted changed key')
    # Return descriptive data for local measurement; host only requires no error.
    return {'seeds': 42, 'pairs': len(features), 'dimensions': len({f['dimensions'] for f in features}),
            'cavity_shapes': len(vocabulary), 'object_counts': sorted({f['objects'] for f in features}),
            'cavity_counts': sorted({f['holes'] for f in features}),
            'ballot_patterns': len({f['ballots'] for f in features})}
