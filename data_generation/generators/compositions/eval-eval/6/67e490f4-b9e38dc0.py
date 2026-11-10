"""Shadow election: opaque polyomino voters compete only when wholly exposed.

No reference files, cached examples, output labels, or random state are used by
solve. Generation and the input decoder use different visibility calculations.
The module does not write files. Optional diagnostics belong in scratch/.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _normal(cells):
    cells = tuple((int(y), int(x)) for y, x in cells)
    y0, x0 = min(y for y, x in cells), min(x for y, x in cells)
    return tuple(sorted((y-y0, x-x0) for y, x in cells))


@lru_cache(maxsize=2048)
def _variants(shape, mirrors=True):
    result = set()
    for sign in ((1, -1) if mirrors else (1,)):
        a = _normal((y, sign*x) for y, x in shape)
        for _ in range(4):
            result.add(a)
            a = _normal((x, -y) for y, x in a)
    return tuple(sorted(result))


def _signature(cells):
    return min(_variants(_normal(cells)))


def _size(cells):
    return max(y for y, x in cells)+1, max(x for y, x in cells)+1


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        group, stack = [first], [first]
        while stack:
            y, x = stack.pop()
            for dy, dx in _N4:
                q = (y+dy, x+dx)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    group.append(q)
        result.append(tuple(sorted(group)))
    return result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= v <= 30 for v in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected an integer ARC grid with dimensions 1..30')
    return a.astype(np.int8, copy=False)


def _panel(a):
    candidates = []
    for col in np.unique(a):
        for cells in _components(a == col):
            if len(cells) < 16:
                continue
            ys, xs = zip(*cells)
            y, x, b, r = min(ys), min(xs), max(ys), max(xs)
            box = a[y:b+1, x:r+1]
            other = set(map(int, np.unique(box))) - {int(col)}
            if (min(box.shape) >= 4 and len(other) == 1
                    and np.all(box[0] == col) and np.all(box[-1] == col)
                    and np.all(box[:, 0] == col) and np.all(box[:, -1] == col)
                    and len(cells) == np.count_nonzero(a == col)):
                candidates.append((int(col), other.pop(), (y, x, b, r)))
    if len(candidates) != 1:
        raise ValueError('expected one rectangular cavity panel')
    return candidates[0]


def _decode(value):
    """Input-only parser and per-lane first-object visibility calculation."""
    original = _grid(value)
    carrier, bg, original_box = _panel(original)
    matches = []
    for rot in range(4):
        a = np.rot90(original, -rot)
        for color in np.unique(a):
            color = int(color)
            if color in (carrier, bg):
                continue
            points = np.argwhere(a == color)
            if len(points) < 11:
                continue
            rear, mouth = int(points[:, 1].min()), int(points[:, 1].max())
            cap = np.flatnonzero(a[:, rear] == color)
            if (len(cap) < 3 or np.any(np.diff(cap) != 1) or mouth-rear < 3):
                continue
            rails = [np.flatnonzero(a[:, x] == color) for x in range(rear+1, mouth+1)]
            if any(len(r) != 2 for r in rails):
                continue
            top, bottom = np.array(rails).T
            if top[0] != cap[0] or bottom[0] != cap[-1]:
                continue
            j = np.arange(len(rails))
            top_fits = [p for p in (1, 2, 3) if np.array_equal(top, top[0]-j//p)
                        and top[0]-top[-1] >= 2]
            bottom_fits = [p for p in (1, 2, 3) if np.array_equal(bottom, bottom[0]+j//p)
                           and bottom[-1]-bottom[0] >= 2]
            if len(top_fits) != 1 or len(bottom_fits) != 1:
                continue
            period, lower_period = top_fits[0], bottom_fits[0]
            ys, xs = np.indices(a.shape)
            steps = (xs-rear-1)//period
            lower_steps = (xs-rear-1)//lower_period
            cone = (xs > rear) & (ys > int(top[0])-steps) & (ys < int(bottom[0])+lower_steps)
            if np.any((a == carrier) & cone):
                continue
            _, _, box = _panel(a)
            y, x, b, r = box
            panel_mask = np.zeros(a.shape, dtype=bool)
            panel_mask[y:b+1, x:r+1] = True
            if np.any(panel_mask & cone):
                continue
            objects = []
            for c in np.unique(a):
                if int(c) in (bg, carrier, color):
                    continue
                objects.extend((int(c), cells) for cells in _components(a == c))
            if not objects or any(min(x for y, x in cells) <= mouth for c, cells in objects):
                continue
            labels = np.full(a.shape, -1, dtype=int)
            for i, (c, cells) in enumerate(objects):
                for yy, xx in cells:
                    labels[yy, xx] = i
            # Only an in-cone cell starts a straight opaque lane shadow.
            first = np.full(a.shape[0], -1, dtype=int)
            for yy in range(a.shape[0]):
                for xx in range(mouth+1, a.shape[1]):
                    if cone[yy, xx] and labels[yy, xx] >= 0:
                        first[yy] = labels[yy, xx]
                        break
            records = []
            for i, (c, cells) in enumerate(objects):
                fronts = {}
                for yy, xx in cells:
                    fronts[yy] = min(xx, fronts.get(yy, xx))
                contained = all(cone[yy, xx] for yy, xx in cells)
                exposed = [bool(cone[yy, xx] and first[yy] == i) for yy, xx in fronts.items()]
                records.append(dict(color=c, cells=cells, shape=_signature(cells),
                                    contained=contained, eligible=contained and all(exposed),
                                    any_visible=any(exposed), exposed_lanes=sum(exposed),
                                    lanes=len(fronts)))
            cavities = _components(a[y:b+1, x:r+1] == bg)
            keys = {_signature(c) for c in cavities}
            if keys != {o['shape'] for o in records}:
                continue
            matches.append(dict(grid=a, rotation=rot, background=bg, carrier=carrier,
                                structure=color, bbox=box, original_box=original_box,
                                rear=rear, mouth=mouth, period=period, cone=cone,
                                periods=(period, lower_period),
                                objects=records, cavities=cavities, shapes=keys))
    if len(matches) != 1:
        raise ValueError('input must identify one emitter and voting field')
    return matches[0]


def _answer(a, mode='full'):
    votes = defaultdict(Counter)
    for o in a['objects']:
        use = (True if mode == 'all' else o['any_visible'] if mode == 'any'
               else o['contained'] if mode == 'cone' else o['eligible'])
        if use:
            key = (len(o['shape']) if mode == 'area' else
                   min(_variants(_normal(o['cells']), False)) if mode == 'rotation' else o['shape'])
            votes[key][o['color']] += 1
    winners = {}
    for shape in a['shapes']:
        if mode == 'rotation':
            continue  # Each cavity orientation has its own rotation-only ballot.
        key = len(shape) if mode == 'area' else shape
        counts = votes[key]
        if not counts:
            raise ValueError('every cavity shape needs an exposed voter')
        high = max(counts.values())
        best = [c for c, n in counts.items() if n == high]
        if len(best) != 1:
            raise ValueError('tied exposed-object ballot')
        winners[shape] = best[0]
    y, x, b, r = a['bbox']
    out = a['grid'][y:b+1, x:r+1].copy()
    for cells in a['cavities']:
        if mode == 'rotation':
            counts = votes[min(_variants(_normal(cells), False))]
            high = max(counts.values(), default=0)
            best = [c for c, n in counts.items() if n == high]
            if len(best) != 1:
                raise ValueError('rotation-only cavity has no unique supported color')
            color = best[0]
        else:
            color = winners[_signature(cells)]
        for yy, xx in cells:
            out[yy, xx] = color
    return np.rot90(out, a['rotation']).copy(), winners, votes


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _answer(_decode(input_grid))[0]


def _wrong(a, out, mode):
    try:
        return not np.array_equal(_answer(a, mode)[0], out)
    except ValueError:
        return True


@lru_cache(maxsize=1)
def _pool():
    # Enumerate connected two-row polyominoes, rather than a fixed tiny catalog.
    shapes = set()
    for w in (3, 4):
        for bits in range(1, 1 << (2*w)):
            cells = [(i//w, i % w) for i in range(2*w) if bits & (1 << i)]
            if not 3 <= len(cells) <= 7:
                continue
            mask = np.zeros((2, w), dtype=bool)
            for p in cells:
                mask[p] = True
            if len(_components(mask)) == 1:
                shapes.add(_signature(cells))
    for length in range(2, 7):
        shapes.add(_signature([(0, i) for i in range(length)]))
    shapes.add(_signature(((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))))
    return tuple(sorted(shapes))


def _choice(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _block(free, cells):
    h, w = free.shape
    for y, x in cells:
        free[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)] = False


def _positions(free, shape):
    h, w = _size(shape)
    ny, nx = free.shape[0]-h+1, free.shape[1]-w+1
    if min(ny, nx) <= 0:
        return []
    good = np.ones((ny, nx), dtype=bool)
    for y, x in shape:
        good &= free[y:y+ny, x:x+nx]
    return [tuple(map(int, p)) for p in np.argwhere(good)]


def _place_panel(rng, grid, cone, shapes, bg, carrier):
    h, w = grid.shape
    options = []
    # All positions are considered, including every edge. No slots are used.
    free = (grid == bg) & ~cone
    summed = np.pad(free.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for ph in range(5, 12):
        for pw in range(5, 15):
            # A vectorized rectangular availability test keeps retries cheap.
            sums = summed[ph:, pw:]-summed[:-ph, pw:]-summed[ph:, :-pw]+summed[:-ph, :-pw]
            options.extend((ph, pw, int(y), int(x)) for y, x in np.argwhere(sums == ph*pw))
    rng.shuffle(options)
    for ph, pw, y, x in options[:120]:
        available = np.zeros((ph, pw), dtype=bool)
        available[1:-1, 1:-1] = True
        placed = []
        for shape in sorted(shapes, key=lambda s: (len(s), max(_size(s))), reverse=True):
            variants = list(_variants(shape))
            rng.shuffle(variants)
            for variant in variants:
                positions = _positions(available, variant)
                if positions:
                    yy, xx = _choice(rng, positions)
                    cells = tuple((yy+dy, xx+dx) for dy, dx in variant)
                    _block(available, cells)
                    placed.append((shape, cells))
                    break
            else:
                break
        if len(placed) != len(shapes):
            continue
        for _ in range(int(rng.integers(3))):
            shape = _choice(rng, shapes)
            variant = _choice(rng, _variants(shape))
            positions = _positions(available, variant)
            if positions:
                yy, xx = _choice(rng, positions)
                cells = tuple((yy+dy, xx+dx) for dy, dx in variant)
                _block(available, cells)
                placed.append((shape, cells))
        box = np.full((ph, pw), carrier, dtype=np.int8)
        for shape, cells in placed:
            for q in cells:
                box[q] = bg
        if len(_components(box == carrier)) != 1:
            continue
        grid[y:y+ph, x:x+pw] = box
        return (y, x, ph, pw), placed
    return None


def _construct(rng, shapes, teaching, test, forbidden_winner=None):
    colors = list(map(int, rng.permutation(10)))
    bg, carrier, wall = colors[:3]
    if forbidden_winner == colors[3]:
        colors[3], colors[4] = colors[4], colors[3]
    winners = {s: colors[3+i] for i, s in enumerate(shapes)}
    height = int(rng.integers(26, 31))
    width = int(rng.integers(29, 31) if test else rng.integers(25, 29))
    center = int(rng.integers(height//2, min(height-5, height//2+4)))
    period = int(rng.integers(1, 3))
    lower_period = (3-period if teaching == 2 or (not teaching and rng.random() < .35)
                    else period)
    upper_half, lower_half = map(int, rng.integers(2, 5, size=2))
    rear = int(rng.integers(0, 3))
    length = 2*max(period, lower_period)+1+int(rng.integers(3))
    mouth = rear+length
    if (center-upper_half-(length-1)//period < 0
            or center+lower_half+(length-1)//lower_period >= height):
        return None
    grid = np.full((height, width), bg, dtype=np.int8)
    grid[center-upper_half:center+lower_half+1, rear] = wall
    for j in range(length):
        grid[center-upper_half-j//period, rear+1+j] = wall
        grid[center+lower_half+j//lower_period, rear+1+j] = wall
    yy, xx = np.indices(grid.shape)
    delta = (xx-rear-1)//period
    lower_delta = (xx-rear-1)//lower_period
    cone = (xx > rear) & (yy > center-upper_half-delta) & (yy < center+lower_half+lower_delta)
    panel = _place_panel(rng, grid, cone, shapes, bg, carrier)
    if panel is None:
        return None
    bbox, cavities = panel
    # Select exposed voters first in disjoint transverse bands. These bands
    # are merely construction scaffolding and are neither marked nor decoded.
    specs = []
    for i, s in enumerate(shapes):
        if teaching or i == 0:
            nwin = (3 if teaching == 2 and i == 0 else 2) if teaching else int(rng.integers(2, 4))
            vote_colors = [winners[s]] * nwin + [winners[shapes[(i+1) % len(shapes)]]
                                               if len(shapes) > 1 else colors[6]]
            if i == 1:
                vote_colors[-1] = colors[6]
            if teaching == 1 and i == 0:
                # A unique plurality need not hold more than half the ballots.
                vote_colors.append(colors[7])
        else:
            vote_colors = [winners[s]] * int(rng.integers(1, 3))
        # In teaching cases the winning handedness opposes the first cavity.
        raw = next(_normal(c) for shape, c in cavities if shape == s)
        mirrored = _normal((y, -x) for y, x in raw)
        for c in vote_colors:
            variants = _variants(mirrored if c == winners[s] else raw, mirrors=False) if teaching else _variants(s)
            short = [v for v in variants if _size(v)[0] <= 2]
            variants = short or list(variants)
            variant = _choice(rng, variants)
            specs.append((s, c, variant))
    rng.shuffle(specs)
    band_height = sum(_size(v)[0] for s, c, v in specs)+len(specs)-1
    if band_height > height:
        return None
    start_y = int(rng.integers(height-band_height+1))
    free = grid == bg
    free[:, :mouth+1] = False
    free[:, width-7:] = False
    objects = []
    for s, c, v in specs:
        sh, sw = _size(v)
        available = free & cone
        available[:start_y] = False
        available[start_y+sh:] = False
        positions = _positions(available, v)
        if not positions:
            return None
        y, x = _choice(rng, positions)
        cells = tuple((y+dy, x+dx) for dy, dx in v)
        for q in cells:
            grid[q] = c
        objects.append((s, c, cells, True))
        _block(free, cells)
        start_y += sh+1
    # Generate partial exposure using a union of explicit foreign-object
    # shadow masks, independently of the decoder's first-hit lane table.
    shadow = np.zeros(grid.shape, dtype=bool)
    def status(cells):
        fronts = {y: min(x for yy, x in cells if yy == y) for y, x in cells}
        exposed = []
        for y, x in fronts.items():
            exposed.append(bool(cone[y, x] and not shadow[y, x]))
        return all(cone[q] for q in cells) and all(exposed), any(exposed)
    # Three partial losing voters make both all-object and any-visible voting
    # select the wrong unique winner for the anchor's 2:1 exposed ballot.
    anchor = shapes[0]
    losing = winners[shapes[1]] if len(shapes) > 1 else colors[6]
    for index in range(int(rng.integers(3, 5))):
        shadow.fill(False)
        for s, c, old, eligible in objects:
            for y, x in old:
                if cone[y, x]:
                    shadow[y, x+1:] = True
        candidates = []
        free = grid == bg
        for s, c, old, eligible in objects:
            _block(free, old)
        free[:, :width-7] = False
        for v in _variants(anchor):
            for y, x in _positions(free & cone, v):
                cells = tuple((y+dy, x+dx) for dy, dx in v)
                full, some = status(cells)
                if some and not full:
                    candidates.append(cells)
        if not candidates:
            return None
        cells = _choice(rng, candidates)
        for q in cells:
            grid[q] = losing
        objects.append((anchor, losing, cells, False))
    # Optional inert or boundary-straddling voters; all still match a cavity.
    for _ in range(int(rng.integers(0, 3))):
        s = _choice(rng, shapes)
        v = _choice(rng, _variants(s))
        free = grid == bg
        for ss, c, old, eligible in objects:
            _block(free, old)
        free[:, :mouth+1] = False
        choices = []
        for y, x in _positions(free, v):
            cells = tuple((y+dy, x+dx) for dy, dx in v)
            if not any(cone[q] for q in cells):
                choices.append(cells)
        if choices:
            cells = _choice(rng, choices)
            c = _choice(rng, colors[3:])
            for q in cells:
                grid[q] = c
            objects.append((s, c, cells, False))
    # Labels are constructed from the deliberately exposed ballots, not solve.
    y, x, ph, pw = bbox
    out = grid[y:y+ph, x:x+pw].copy()
    for s, cells in cavities:
        for q in cells:
            out[q] = winners[s]
    rotation = int(rng.integers(4))
    return {'input': np.rot90(grid, rotation).copy(), 'output': np.rot90(out, rotation).copy()}


def _counts(num_train, num_test):
    for v, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                or not low <= v <= high):
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    pool = _pool()
    anchors = [s for s in pool if 4 <= len(s) <= 6 and sorted(_size(s)) == [2, 3]
               and _normal((y, -x) for y, x in s) not in _variants(s, False)]
    anchor = _choice(rng, anchors)
    companion = _choice(rng, [s for s in pool if s != anchor and len(s) == len(anchor)
                             and sorted(_size(s)) == sorted(_size(anchor))])
    puzzle, seen, known = {'train': [], 'test': []}, set(), set()
    first_winner = None
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            teaching = index+1 if split == 'train' and index < 2 else 0
            if teaching:
                shapes = [anchor, companion]
            elif split == 'test':
                novel = [s for s in pool if s not in known and len(s) <= 6]
                shapes = [anchor, _choice(rng, novel)]
                if rng.random() < .55:
                    shapes.append(companion if companion not in shapes else _choice(
                        rng, [s for s in pool if s not in shapes and len(s) <= 5]))
            else:
                shapes = [anchor, _choice(rng, [s for s in pool if s != anchor and len(s) <= 6])]
                if rng.random() < .25:
                    shapes.append(_choice(rng, [s for s in pool if s not in shapes and len(s) <= 5]))
            for attempt in range(180):
                pair = _construct(rng, shapes, teaching, split == 'test',
                                  first_winner if teaching and index == 1 else None)
                if pair is None or _key(pair['input']) in seen:
                    continue
                try:
                    a = _decode(pair['input'])
                    prediction, winners, votes = _answer(a)
                    if not np.array_equal(prediction, pair['output']):
                        continue
                    if not all(_wrong(a, prediction, mode) for mode in ('all', 'any', 'cone')):
                        continue
                    if teaching and not _wrong(a, prediction, 'area'):
                        continue
                    if teaching and not _wrong(a, prediction, 'rotation'):
                        continue
                except ValueError:
                    continue
                puzzle[split].append(pair)
                seen.add(_key(pair['input']))
                if split == 'train':
                    known.update(a['shapes'])
                    if index == 0:
                        first_winner = winners[anchor]
                break
            else:
                raise RuntimeError('bounded scene construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train, test = set(), [], []
    history = defaultdict(set)
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _decode(inp)
            predicted, winners, votes = _answer(a)
            assert np.array_equal(predicted, out), 'wrong shape winner or panel crop'
            y, x, b, r = a['original_box']
            box = inp[y:b+1, x:r+1]
            assert box.shape == out.shape
            assert np.array_equal(box == a['carrier'], out == a['carrier'])
            assert np.all(out[box == a['carrier']] == a['carrier'])
            assert np.all(out[box == a['background']] != a['background'])
            # Check isolated objects independent of the monochrome segmentation.
            agrid = a['grid'].copy()
            yy, xx, bb, rr = a['bbox']
            agrid[yy:bb+1, xx:rr+1] = a['background']
            mask = (agrid != a['background']) & (agrid != a['structure'])
            assert len(_components(mask)) == len(a['objects'])
            assert all(_wrong(a, out, m) for m in ('all', 'any', 'cone'))
            assert any(o['contained'] and o['any_visible'] and not o['eligible'] for o in a['objects'])
            for shape in a['shapes']:
                assert any(o['eligible'] and o['shape'] == shape for o in a['objects'])
            (train if split == 'train' else test).append((a, out))
            if split == 'train':
                for shape, c in winners.items():
                    history[shape].add(c)
    assert any(_wrong(a, out, 'area') for a, out in train)
    assert any(_wrong(a, out, 'rotation') for a, out in train)
    assert any(a['periods'][0] != a['periods'][1] for a, out in train)
    assert any(len(v) >= 3 and 2*max(v.values()) <= sum(v.values())
               for a, out in train for v in _answer(a)[2].values())
    assert any(len(cs) > 1 for cs in history.values())
    known = set().union(*(a['shapes'] for a, out in train))
    assert all(a['shapes']-known for a, out in test), 'test must introduce locally supported geometry'


def self_test():
    """42 bounded seeds, metamorphic checks and deliberate label corruption.

    Retains hashes and small sets only; no grids/traces/logs are stored on disk.
    """
    signatures, dimensions, panels, palettes, shapes, outlines = set(), set(), set(), set(), set(), set()
    directions, periods, totals, partials = set(), set(), set(), set()
    repeats = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                a = _decode(inp)
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                dimensions.add(inp.shape)
                panels.add(out.shape)
                palettes.add((a['background'], a['carrier'], a['structure']))
                shapes.update(a['shapes'])
                directions.add(a['rotation'])
                periods.update(a['periods'])
                totals.add(len(a['objects']))
                partials.add(sum(o['any_visible'] and not o['eligible'] for o in a['objects']))
                repeats += len(a['cavities']) > len(a['shapes'])
                walls = np.argwhere(a['grid'] == a['structure'])
                outlines.add(_normal(walls))
        signatures.add(digest.digest())
        # Equivariance and input immutability, independent of construction labels.
        pair = p['test'][0]
        before = pair['input'].copy()
        for k in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(before, k)), np.rot90(pair['output'], k))
        permutation = np.random.default_rng(seed+10000).permutation(10)
        assert np.array_equal(solve(permutation[before]), permutation[pair['output']])
        assert np.array_equal(before, pair['input'])
    assert len(signatures) == 42
    assert len(dimensions) >= 15 and len(panels) >= 15
    assert len(shapes) >= 20 and len(palettes) >= 70 and len(outlines) >= 10
    assert directions == {0, 1, 2, 3} and periods == {1, 2}
    assert len(totals) >= 5 and len(partials) >= 2 and repeats >= 10
    # Fixed geometric examples distinguish exact shapes and rotations/reflections.
    assert _signature(((0, 0), (0, 1), (0, 2), (1, 0))) != _signature(((0, 0), (0, 1), (1, 1), (1, 2)))
    first = generate_puzzle(91, 2, 1)
    second = generate_puzzle(91, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], second[s]) for k in a)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            _counts(nt, ne)
    # The host generates every legal combination; keep the self-test's grid
    # construction to the three count boundaries instead of duplicating it.
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(100+nt*10+ne, nt, ne)
        assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, legal count validation and count boundaries)')
