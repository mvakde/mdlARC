"""Rank-loaded lanterns: independent queues control chromatic light transport.

Only standard-library modules and NumPy are used. Parsing and solving require
only the input. Sampling profiles and teaching checks never enter solve().
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=None)
def _glyph(size, period):
    """Canonical right-facing, thin-boundary glyph; returns an immutable mask."""
    mask = np.zeros((size + 4, size + 2 * period), dtype=bool)
    mask[2, :size] = mask[size + 1, :size] = True
    mask[2: size + 2, 0] = mask[2: size + 2, size - 1] = True
    for k in range(1, 2 * period + 1):
        step = (k + period - 1) // period
        mask[2 - step, size - 1 + k] = True
        mask[size + 1 + step, size - 1 + k] = True
    mask.flags.writeable = False
    return mask


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        points, pending = [start], [start]
        while pending:
            r, c = pending.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        points.append(q)
                        pending.append(q)
        result.append(points)
    return result


def _body_offset(size, period, direction):
    body = np.zeros(_glyph(size, period).shape, dtype=bool)
    body[2:size + 2, :size] = True
    coords = np.argwhere(np.rot90(body, direction))
    return tuple(map(int, coords.min(axis=0)))


def _parse(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= grid.size // 2:
        raise ValueError("background must occupy more than half the input")
    objects = []
    rectangles = []
    for points in _components(grid != bg):
        coords = np.asarray(points, dtype=int)
        r, c = map(int, coords.min(axis=0))
        bottom, right = map(int, coords.max(axis=0) + 1)
        color_values = {int(grid[y, x]) for y, x in points}
        if len(color_values) != 1:
            raise ValueError("foreground instances must be monochrome")
        color = color_values.pop()
        crop = np.zeros((bottom-r, right-c), dtype=bool)
        crop[coords[:, 0]-r, coords[:, 1]-c] = True
        candidates = []
        h, w = crop.shape
        if h == w and 3 <= h <= 6 and np.all(crop):
            candidates.append(dict(kind="S", top=r, left=c, size=h))
        for direction in range(4):
            canonical = np.rot90(crop, -direction)
            size = canonical.shape[0] - 4
            difference = canonical.shape[1] - size
            if 3 <= size <= 6 and difference in (2, 4, 6):
                period = difference // 2
                if np.array_equal(canonical, _glyph(size, period)):
                    dr, dc = _body_offset(size, period, direction)
                    candidates.append(dict(kind="H", top=r+dr, left=c+dc,
                                           size=size, period=period,
                                           direction=direction))
        if len(candidates) != 1:
            raise ValueError("each component must identify one complete square or lantern")
        obj = candidates[0]
        obj.update(color=color, points=tuple(points))
        # Require empty square interiors in the actual grid, not just this mask.
        if obj['kind'] == 'H':
            y, x, s = obj['top'], obj['left'], obj['size']
            if not np.all(grid[y+1:y+s-1, x+1:x+s-1] == bg):
                raise ValueError("lantern reservoir must be initially empty")
        rectangles.append((r, c, bottom, right))
        objects.append(obj)
    if not objects or len({o['size'] for o in objects}) != 1:
        raise ValueError("one common body scale is required")
    # Bounding-box separation is the explicit source grammar, including rails.
    for i, (r, c, b, e) in enumerate(rectangles):
        for rr, cc, bb, ee in rectangles[:i]:
            if not (b < rr or bb < r or e < cc or ee < c):
                raise ValueError("object boxes require a background gap")
    queues = {}
    for kind in ('H', 'S'):
        queue = sorted((o for o in objects if o['kind'] == kind),
                       key=lambda o: (o['top'], o['left']))
        if not queue or len({o['top'] for o in queue}) != len(queue):
            raise ValueError("both classes and distinct same-class top rows are required")
        queues[kind] = queue
    return dict(grid=grid, background=bg, objects=objects, **queues)


def _canonical_body(obj, shape):
    r, c, s, d = obj['top'], obj['left'], obj['size'], obj['direction']
    h, w = shape
    return ((r, c), (c, h-r-s), (h-r-s, w-c-s), (w-c-s, r))[d]


def _load(analysis, assignment='rank'):
    hollows, solids = analysis['H'], analysis['S']
    pairs = []
    if assignment == 'greedy':
        slots = [None, None]
        rows = []
        for obj in sorted(analysis['objects'], key=lambda o: (o['top'], o['left'])):
            col = int(obj['kind'] == 'S')
            if slots[col] is not None:
                rows.append(slots)
                slots = [None, None]
            slots[col] = obj
        rows.append(slots)
        return [(h, s['color']) for h, s in rows if h is not None and s is not None]
    if assignment == 'glyph_top':
        hollows = sorted(hollows, key=lambda o: (min(y for y, x in o['points']),
                                                o['left']))
    elif assignment in ('horizontal', 'horizontal_reverse'):
        reverse = assignment == 'horizontal_reverse'
        hollows = sorted(hollows, key=lambda o: (o['left'], o['top']), reverse=reverse)
        solids = sorted(solids, key=lambda o: (o['left'], o['top']), reverse=reverse)
    elif assignment == 'reverse_hollows':
        hollows = list(reversed(hollows))
    elif assignment == 'reverse_solids':
        solids = list(reversed(solids))
    for i, obj in enumerate(hollows):
        if i >= len(solids):
            break
        color = solids[i]['color']
        if assignment == 'own':
            color = obj['color']
        elif assignment == 'nearest':
            donor = min(solids, key=lambda o: (abs(obj['top']-o['top']) +
                        abs(obj['left']-o['left']), o['top'], o['left']))
            color = donor['color']
        pairs.append((obj, color))
    if assignment == 'activate_unmatched':
        pairs.extend((obj, solids[-1]['color']) for obj in hollows[len(solids):])
    return pairs


def _beam(analysis, obj, color, opacity='chromatic', geometry='cone'):
    original = analysis['grid']
    direction = obj['direction']
    grid = np.rot90(original, -direction)
    r, c = _canonical_body(obj, original.shape)
    s, p = obj['size'], obj['period']
    footprint = np.zeros(grid.shape, dtype=bool)
    footprint[r+1:r+s-1, c+1:c+s-1] = True
    if geometry != 'interior':
        for x in range(c+s, grid.shape[1]):
            step = (x-(c+s-1)+p-1)//p if geometry == 'cone' else 0
            lo, hi = r-step, r+s-1+step
            footprint[max(0, lo+1):min(grid.shape[0], hi), x] = True
    own = np.zeros(original.shape, dtype=bool)
    for y, x in obj['points']:
        own[y, x] = True
    own = np.rot90(own, -direction)
    opaque = (grid != analysis['background']) & ~own
    if opacity == 'chromatic':
        opaque &= grid != color
    elif opacity == 'none':
        opaque[:] = False
    blocked = np.maximum.accumulate(opaque & footprint, axis=1)
    return np.rot90(footprint & ~blocked, direction).copy()


def _render(analysis, assignment='rank', opacity='chromatic',
            overlap='agreement', geometry='cone', details=False):
    grid, bg = analysis['grid'], analysis['background']
    colors = np.zeros(grid.shape, dtype=np.uint16)
    arrivals = np.zeros(grid.shape, dtype=np.int8)
    first = np.full(grid.shape, bg, dtype=np.int8)
    for obj, color in _load(analysis, assignment):
        lit = _beam(analysis, obj, color, opacity, geometry) & (grid == bg)
        first[lit & (arrivals == 0)] = color
        colors[lit] |= np.uint16(1 << color)
        arrivals[lit] += 1
    out = grid.copy()
    if overlap == 'priority':
        out[arrivals > 0] = first[arrivals > 0]
    else:
        for color in range(10):
            out[colors == (1 << color)] = color
    if details:
        single = (colors != 0) & ((colors & (colors-1)) == 0)
        return out, dict(agreement=int(np.sum((arrivals >= 2) & single)),
                         conflict=int(np.sum((arrivals >= 2) & ~single)),
                         changed=int(np.sum(out != grid)))
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve every member of the stated grammar using only visible input."""
    return _render(_parse(input_grid))


def _features(analysis):
    out, result = _render(analysis, details=True)
    for mode in ('own', 'nearest', 'horizontal', 'horizontal_reverse', 'glyph_top',
                 'reverse_hollows', 'reverse_solids', 'greedy', 'activate_unmatched'):
        result[mode] = int(np.sum(out != _render(analysis, assignment=mode)))
    for name, kwargs in (('opaque', dict(opacity='all')),
                         ('shadows', dict(opacity='none')),
                         ('priority', dict(overlap='priority')),
                         ('straight', dict(geometry='straight')),
                         ('interior', dict(geometry='interior'))):
        result[name] = int(np.sum(out != _render(analysis, **kwargs)))
    result.update(hollows=len(analysis['H']), solids=len(analysis['S']),
                  size=analysis['H'][0]['size'],
                  directions=tuple(o['direction'] for o in analysis['H']),
                  periods=tuple(o['period'] for o in analysis['H']))
    return result


def _put(grid, kind, size, period, direction, top, left, color):
    """Coordinates here describe the entire glyph's bounding box."""
    mask = np.ones((size, size), dtype=bool) if kind == 'S' else np.rot90(
        _glyph(size, period), direction)
    h, w = mask.shape
    grid[top:top+h, left:left+w][mask] = color


def _teaching_grid(rng, style, bg, palette):
    """Two coverage-rich scenes; positions, size, cadence and reflection vary."""
    s = int(rng.integers(3, 5))
    width = int(rng.integers(26, 31))
    wall1, wall2, wall3, dye1, dye2 = palette[:5]
    if style == 'distinct':
        # An upward lantern's rails extend above the first lantern, but its
        # square body starts lower. This explicitly teaches body-edge ranking.
        first_r = 9
        body_rows = [9, 12, int(rng.integers(21, 23))]
        lefts = [int(rng.integers(0, 2)), 14, int(rng.integers(1, 4))]
    else:
        first_r = int(rng.integers(4, 6)) if s == 3 else 4
        body_rows = [first_r, first_r+s+6]
        lefts = [int(rng.integers(0, 2)), int(rng.integers(3, 6))]
    height = int(rng.integers(max(body_rows)+s+2, 31))
    grid = np.full((height, width), bg, dtype=np.int8)
    for i, (r, c) in enumerate(zip(body_rows, lefts)):
        p = 3 if style == 'distinct' and i == 1 else int(rng.integers(1, 3))
        d = 1 if style == 'distinct' and i == 1 else 0
        dr, dc = _body_offset(s, p, d)
        _put(grid, 'H', s, p, d, r-dr, c-dc, [wall1, wall2, wall3][i])
    # Nonalternating S,S,H,... order; horizontal donor order opposes rank.
    _put(grid, 'S', s, 1, 0, 0, width-s-3, dye1)
    near = int(rng.integers(12, min(16, width-2*s-3)))
    _put(grid, 'S', s, 1, 0, 1, near, dye2 if style == 'distinct' else dye1)
    if style == 'agree':
        # An excess donor is still an opaque spatial object.
        _put(grid, 'S', s, 1, 0, first_r+1,
             int(rng.integers(15, width-s-2)), dye2)
    if rng.integers(2):
        grid = grid[:, ::-1].copy()
    return grid


def _free_grid(rng, bg, palette):
    s = int(rng.choice([3, 4, 5, 6], p=[.36, .30, .22, .12]))
    nh = int(rng.integers(1, 5 if s <= 4 else 4))
    ns = int(rng.integers(1, 6))
    minimum = 18 if nh+ns <= 4 else 24
    height, width = map(int, rng.integers(minimum, 31, size=2))
    if rng.random() < .15:
        height = width = 30
    grid = np.full((height, width), bg, dtype=np.int8)
    boxes, tops = [], {'H': set(), 'S': set()}
    # Large shapes first improves random packing without a fixed layout lattice.
    kinds = ['H']*nh + ['S']*ns
    for kind in kinds:
        p = int(rng.integers(1, 4))
        d = int(rng.integers(4))
        mask = np.ones((s, s), dtype=bool) if kind == 'S' else np.rot90(_glyph(s, p), d)
        h, w = mask.shape
        dr = 0 if kind == 'S' else _body_offset(s, p, d)[0]
        sites = []
        for r in range(height-h+1):
            if r+dr in tops[kind]:
                continue
            for c in range(width-w+1):
                if all(r+h < rr or bb < r or c+w < cc or ee < c
                       for rr, cc, bb, ee in boxes):
                    sites.append((r, c))
        if not sites:
            return None
        r, c = sites[int(rng.integers(len(sites)))]
        # Repeated colors are instances, not an alternative color dictionary.
        pool = palette[:int(rng.integers(2, len(palette)+1))]
        color = int(rng.choice(pool))
        _put(grid, kind, s, p, d, r, c, color)
        tops[kind].add(r+dr)
        boxes.append((r, c, r+h, c+w))
    return grid


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _integer(value, name, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(name + ' outside legal range')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    shared = bool(rng.random() < .22)
    base = rng.permutation(10)
    puzzle, seen = {'train': [], 'test': []}, set()
    teaching_roles = None
    for i in range(num_train+num_test):
        style = 'distinct' if i == 0 else ('agree' if i == 1 else 'free')
        colors = base if shared else rng.permutation(10)
        bg, palette = int(colors[0]), list(map(int, colors[1:]))
        if style == 'distinct':
            teaching_roles = palette[0], palette[3]
        elif style == 'agree':
            # The first frame color becomes a donor color; the first donor
            # color becomes a frame color, even for a two-demo puzzle.
            wall_color, dye_color = teaching_roles
            if bg in teaching_roles:
                choices = [v for v in range(10) if v not in teaching_roles]
                bg = int(rng.choice(choices))
                palette = list(map(int, rng.permutation([v for v in range(10) if v != bg])))
            for position, value in ((0, dye_color), (3, wall_color)):
                other = palette.index(value)
                palette[position], palette[other] = palette[other], palette[position]
        accepted = False
        for attempt in range(400):
            grid = (_teaching_grid(rng, style, bg, palette) if style != 'free'
                    else _free_grid(rng, bg, palette))
            if grid is None:
                continue
            try:
                a = _parse(grid)
            except ValueError:
                continue
            if _key(grid) in seen:
                continue
            if style == 'distinct':
                f = _features(a)
                required = ('own', 'nearest', 'horizontal', 'horizontal_reverse',
                            'glyph_top',
                            'reverse_hollows', 'reverse_solids', 'greedy',
                            'activate_unmatched', 'priority', 'straight', 'interior')
                if not all(f[k] > 0 for k in required) or not f['conflict']:
                    continue
            elif style == 'agree':
                f = _features(a)
                if not all(f[k] > 0 for k in ('agreement', 'opaque', 'shadows')):
                    continue
            output = _render(a)
            if np.array_equal(grid, output):
                continue
            puzzle['train' if i < num_train else 'test'].append(
                {'input': grid, 'output': output})
            seen.add(_key(grid))
            accepted = True
            break
        if not accepted:
            raise RuntimeError('bounded layout construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate returned arrays and actual teaching counterexamples."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, teaching, roles = set(), [], {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            a = _parse(inp)
            assert np.array_equal(_render(a), out), 'input-derived answer mismatch'
            foreground = inp != a['background']
            assert np.array_equal(out[foreground], inp[foreground]), 'foreground changed'
            assert np.any(inp != out)
            assert set(map(int, out[inp != out])) <= {o['color'] for o in a['S']}
            if split == 'train':
                teaching.append(_features(a))
                for obj in a['objects']:
                    roles.setdefault(obj['color'], set()).add(obj['kind'])
    for k in ('own', 'nearest', 'horizontal', 'horizontal_reverse',
              'glyph_top',
              'reverse_hollows', 'reverse_solids', 'greedy', 'activate_unmatched',
              'priority', 'straight', 'interior', 'opaque', 'shadows',
              'agreement', 'conflict'):
        assert any(f[k] > 0 for f in teaching), 'missing teaching witness: ' + k
    assert any(f['hollows'] > f['solids'] for f in teaching)
    assert any(f['hollows'] < f['solids'] for f in teaching)
    assert any(len(kinds) == 2 for kinds in roles.values()), 'missing geometric color-role switch'


def _fixture_tests():
    # Explicit cells and phase, independently of a saved generator answer.
    for p in (1, 2, 3):
        g = np.zeros((15, 23), dtype=np.int8)
        _put(g, 'H', 3, p, 0, 5, 1, 1)  # body r=7, right side c=3
        _put(g, 'S', 3, 1, 0, 0, 12, 2)
        out = solve(g)
        x = 4+2*p
        boundary = 7-((x-3+p-1)//p)
        assert out[boundary, x] == 0
        assert out[boundary+1, x] == 2
        assert out[8, 2] == 2  # square reservoir
        # With one object per class the rank is invariant under rotation.
        for direction in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(g, direction)), np.rot90(out, direction))
    # Matching donor is transparent; excess differently colored donor is opaque.
    g = np.zeros((18, 30), dtype=np.int8)
    _put(g, 'H', 3, 1, 0, 5, 1, 1)
    _put(g, 'S', 3, 1, 0, 7, 10, 2)
    _put(g, 'S', 3, 1, 0, 11, 16, 3)
    _put(g, 'S', 3, 1, 0, 12, 22, 4)
    out = solve(g)
    assert out[8, 15] == 2 and out[12, 21] == 0 and out[10, 21] == 2
    assert out[7, 10] == 2 and out[11, 16] == 3
    assert out[12, 28] == 0 and out[10, 28] == 2
    assert out[14, 20] == 2 and out[14, 28] == 0
    # Same lane occlusion persists; conflict darkness is simultaneous, not paint order.
    g = np.zeros((28, 28), dtype=np.int8)
    _put(g, 'H', 3, 1, 0, 2, 1, 1)
    _put(g, 'H', 3, 1, 0, 12, 1, 1)
    _put(g, 'S', 3, 1, 0, 0, 18, 2)
    _put(g, 'S', 3, 1, 0, 1, 23, 3)
    assert solve(g)[10, 16] == 0  # two differently colored beams
    g[g == 3] = 2
    assert solve(g)[10, 16] == 2  # agreement from two beams
    # Extra lanterns do not emit; they are still preserved optical material.
    g = np.zeros((23, 25), dtype=np.int8)
    _put(g, 'H', 3, 1, 0, 2, 1, 1)
    _put(g, 'H', 3, 1, 3, 13, 0, 3)
    _put(g, 'S', 3, 1, 0, 0, 17, 2)
    out = solve(g)
    assert out[5, 2] == 2 and out[14, 3] == 0


def self_test():
    """Bounded, meaningful checks over 42 seeds, counts and semantic fixtures."""
    _fixture_tests()
    shapes, layouts, scales, periods, directions, counts, backgrounds = (
        set() for _ in range(7))
    signatures, varying_sizes, varying_bg = set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p['input']) for split in ('train', 'test') for p in puzzle[split])
        signatures.add(signature)
        per_sizes, per_bg = set(), set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                before = inp.copy()
                assert np.array_equal(solve(inp), pair['output'])
                assert np.array_equal(inp, before), 'solve mutated input'
                a = _parse(inp)
                shapes.add(inp.shape)
                s = a['H'][0]['size']
                scales.add(s)
                per_sizes.add(s)
                per_bg.add(a['background'])
                backgrounds.add(a['background'])
                periods.update(o['period'] for o in a['H'])
                directions.update(o['direction'] for o in a['H'])
                counts.add((len(a['H']), len(a['S'])))
                layouts.add((inp.shape, s, tuple((o['kind'], o['top'], o['left'],
                            o.get('period', 0), o.get('direction', -1))
                            for o in a['objects'])))
        varying_sizes += len(per_sizes) > 1
        varying_bg += len(per_bg) > 1
        if seed in (0, 13, 41):
            again = generate_puzzle(seed)
            assert signature == tuple(_key(p['input']) for split in ('train', 'test') for p in again[split])
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(layouts) >= 170
    assert scales == {3, 4, 5, 6} and periods == {1, 2, 3}
    assert directions == {0, 1, 2, 3} and len(counts) >= 12
    assert backgrounds == set(range(10))
    assert varying_sizes >= 25 and varying_bg >= 20
    # Valid labels alone are insufficient: remove all rank/interaction evidence.
    simple = {'train': [], 'test': []}
    for p in (1, 2, 3):
        g = np.zeros((15, 23), dtype=np.int8)
        _put(g, 'H', 3, p, 0, 5, 1, 1)
        _put(g, 'S', 3, 1, 0, 0, 12, 2)
        simple['train' if p < 3 else 'test'].append({'input': g, 'output': solve(g)})
    try:
        validate_generated_puzzle(simple)
    except AssertionError as error:
        assert 'missing teaching witness' in str(error)
    else:
        raise AssertionError('evidence-free demonstrations accepted')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12345, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(0, 2, 1)
    pair = damaged['test'][0]
    y, x = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][y, x] = pair['input'][y, x]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt label accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, legal counts, semantic fixtures)')
