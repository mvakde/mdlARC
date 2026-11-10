"""Optical elections: a reflected beam supplies shape-indexed color ballots.

Only NumPy and the standard library are used. No parent generator is imported.
Generation plans are discarded; solve() reconstructs everything from the input.
"""
from collections import Counter, defaultdict
import hashlib
import json

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))
S4 = ((0, 0), (0, 1), (1, 1), (1, 2))
L4 = ((0, 0), (0, 1), (0, 2), (1, 0))


def _norm(cells):
    cells = tuple(cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r-y), int(c-x)) for r, c in cells))


def _variants(cells, reflect=True):
    out = set()
    for flip in range(2 if reflect else 1):
        s = _norm((r, -c if flip else c) for r, c in cells)
        for _ in range(4):
            out.add(s)
            s = _norm((c, -r) for r, c in s)
    return tuple(sorted(out))


def _key(cells):
    return min(_variants(cells))


def _size(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    out = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for dr, dc in DIR4:
                p = (r+dr, c+dc)
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        out.append(tuple(sorted(cells)))
    return out


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError('expected an integer ARC grid of size 1..30')
    return a.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _source(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return ((r, c), (r-dr, c), (r, c-dc))


def _parse_hardware(grid, bg, carrier):
    sources, mirrors = [], []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, carrier}):
        for cells in _components(grid == color):
            h, w = _size(_norm(cells))
            if len(cells) == 3 and (h, w) == (2, 2):
                y, x = min(r for r, c in cells), min(c for r, c in cells)
                missing = next((r, c) for r in (y, y+1) for c in (x, x+1)
                               if (r, c) not in cells)
                elbow = (2*y+1-missing[0], 2*x+1-missing[1])
                sources.append(dict(cells=cells, elbow=elbow, color=color,
                                    direction=(elbow[0]-missing[0], elbow[1]-missing[1])))
            elif len(cells) % 2 == 1 and (h == 1 or w == 1) and len(cells) == h*w:
                mirrors.append(dict(cells=cells, color=color,
                                    mid=(sum((cells[0][0], cells[-1][0]))//2,
                                         sum((cells[0][1], cells[-1][1]))//2),
                                    axes=(0, 1) if len(cells) == 1 else ((0,) if h == 1 else (1,))))
            else:
                raise ValueError('colored hardware must be an L or an odd straight mirror')
    if len(sources) != 1:
        raise ValueError('exactly one colored L source is required')
    return sources[0], mirrors


def _trace(shape, source, mirrors, straight=False, monochrome=False):
    """Cell tracer. Neutral tokens are transparent, so do not enter this API."""
    p, d, color = source['elbow'], source['direction'], source['color']
    occupied = {p for obj in [source] + mirrors for p in obj['cells']}
    seen, used, path, contacts = set(), set(), [], []
    while True:
        p = (p[0]+d[0], p[1]+d[1])
        if not _inside(p, shape):
            return path, contacts
        if (p, d) in seen:
            raise ValueError('beam loop')
        seen.add((p, d))
        if p in occupied:
            if straight:
                return path, contacts
            raise ValueError('beam hits hardware rather than a midpoint face')
        hits = []
        if not straight:
            for i, mirror in enumerate(mirrors):
                for axis in mirror['axes']:
                    q = mirror['mid']
                    if p[1-axis] == q[1-axis] and q[axis]-p[axis] == d[axis]:
                        hits.append((i, axis))
        if len(hits) > 1:
            raise ValueError('ambiguous mirror contact')
        if hits:
            i, axis = hits[0]
            used.add(i)
            if not monochrome:
                color = mirrors[i]['color']
            d = tuple(-v if j == axis else v for j, v in enumerate(d))
            contacts.append((i, axis, p, color))
        path.append((p, color))
        if len(path) > 4*shape[0]*shape[1]:
            raise ValueError('unbounded beam')


def _decode(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= grid.size/2:
        raise ValueError('background must occupy more than half the input')
    rectangles = []
    for carrier in sorted(set(map(int, np.unique(grid))) - {bg}):
        for cells in _components(grid == carrier):
            if len(cells) < 16:
                continue
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            r1, c1 = max(r for r, c in cells)+1, max(c for r, c in cells)+1
            box = grid[r0:r1, c0:c1]
            if min(box.shape) < 4 or set(map(int, np.unique(box))) != {bg, carrier}:
                continue
            if (np.all(box[0] == carrier) and np.all(box[-1] == carrier)
                    and np.all(box[:, 0] == carrier) and np.all(box[:, -1] == carrier)):
                rectangles.append((carrier, (r0, c0, r1, c1)))
    if len(rectangles) != 1:
        raise ValueError('one connected rectangular stencil with cavities is required')
    carrier, bbox = rectangles[0]
    r0, c0, r1, c1 = bbox
    box = grid[r0:r1, c0:c1]
    cavities = _components(box == bg)
    exterior = grid == carrier
    exterior[r0:r1, c0:c1] = False
    tokens = _components(exterior)
    if not tokens:
        raise ValueError('no neutral voting objects')
    source, mirrors = _parse_hardware(grid, bg, carrier)
    path, contacts = _trace(grid.shape, source, mirrors)
    if {i for i, axis, p, color in contacts} != set(range(len(mirrors))):
        raise ValueError('every mirror must participate')
    if any(r0 <= r < r1 and c0 <= c < c1 for (r, c), color in path):
        raise ValueError('stencil must lie outside the beam')
    classes = {_key(cells) for cells in cavities}
    if {_key(cells) for cells in tokens} != classes:
        raise ValueError('all neutral objects must match a cavity class')
    return dict(grid=grid, bg=bg, carrier=carrier, bbox=bbox, box=box,
                cavities=cavities, tokens=tokens, source=source, mirrors=mirrors,
                path=path, contacts=contacts)


def _tally(data, path=None, grouping='shape', pixels=False, all_tokens=False):
    path = data['path'] if path is None else path
    light = defaultdict(set)
    for p, color in path:
        light[p].add(color)
    def signature(cells):
        if grouping == 'area':
            return len(cells)
        if grouping == 'box':
            return tuple(sorted(_size(_norm(cells))))
        if grouping == 'rotation':
            return min(_variants(cells, False))
        return _key(cells)
    votes, exposures = defaultdict(Counter), []
    for cells in data['tokens']:
        colors = set().union(*(light.get(p, set()) for p in cells))
        if len(colors) > 1:
            raise ValueError('an object cannot receive multiple beam colors')
        if colors:
            color = next(iter(colors))
            n = sum(bool(light.get(p)) for p in cells)
            votes[signature(cells)][color] += n if pixels else 1
            exposures.append((cells, color, n))
        elif all_tokens:
            # Explicit cheap alternative: let unlit shapes vote neutral ink.
            votes[signature(cells)][data['carrier']] += 1
    winners = {}
    for cells in data['cavities']:
        key = signature(cells)
        ballot = votes[key]
        if not ballot:
            raise ValueError('cavity lacks an illuminated supporting object')
        high = max(ballot.values())
        best = [c for c, n in ballot.items() if n == high]
        if len(best) != 1:
            raise ValueError('tied election')
        winners[key] = best[0]
    answer = data['box'].copy()
    for cells in data['cavities']:
        for p in cells:
            answer[p] = winners[signature(cells)]
    return answer, dict(votes=dict(votes), winners=winners, exposures=exposures)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, trace the beam, count one ballot per lit shape, and crop."""
    return _tally(_decode(input_grid))[0]


def _fails(data, expected, mode):
    try:
        if mode in ('straight', 'monochrome'):
            path, _ = _trace(data['grid'].shape, data['source'], data['mirrors'],
                             straight=mode == 'straight', monochrome=mode == 'monochrome')
            actual = _tally(data, path=path)[0]
        else:
            actual = _tally(data, grouping=mode if mode in ('area', 'box', 'rotation') else 'shape',
                            pixels=mode == 'pixels', all_tokens=mode == 'unlit')[0]
        return not np.array_equal(expected, actual)
    except ValueError:
        return True


def _choose(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _blocked(cells):
    return {(r+dr, c+dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _place_options(shape, shape_variants, forbidden, path_colors=None, shade=None, hits=None):
    """Enumerate bounded bitmap placements, optionally on one beam color."""
    h, w = shape
    options = []
    for variant in shape_variants:
        vh, vw = _size(variant)
        if path_colors is not None and shade is not None:
            origins = {(p[0]-r, p[1]-c) for p, cs in path_colors.items() if cs == {shade}
                       for r, c in variant}
        else:
            origins = ((r, c) for r in range(h-vh+1) for c in range(w-vw+1))
        for y, x in sorted(origins):
            if y < 0 or x < 0 or y+vh > h or x+vw > w:
                continue
            cells = tuple((y+r, x+c) for r, c in variant)
            if any(p in forbidden for p in cells):
                continue
            if path_colors is not None:
                colors = set().union(*(path_colors.get(p, set()) for p in cells))
                if (shade is None and colors) or (shade is not None and colors != {shade}):
                    continue
                n = sum(p in path_colors for p in cells)
                if hits is not None and n != hits:
                    continue
            options.append(cells)
    return options


def _hardware_scene(rng, teaching, axis):
    h, w = (int(rng.integers(25, 31)), int(rng.integers(25, 31))) if teaching else (
        int(rng.integers(20, 31)), int(rng.integers(20, 31)))
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, start = palette[:3]
    direction = _choose(rng, DIAG)
    # Launch location is free; long clear routes are selected by geometry.
    elbow = (int(rng.integers(1, h-1)), int(rng.integers(1, w-1)))
    source = dict(cells=_source(elbow, direction), elbow=elbow,
                  direction=direction, color=start)
    nmirror = 2 if teaching else int(rng.integers(1, 5))
    mirrors, anchor, d = [], elbow, direction
    for j in range(nmirror):
        maxdist = min(h-1-anchor[0] if d[0] > 0 else anchor[0],
                      w-1-anchor[1] if d[1] > 0 else anchor[1])
        low = 8 if teaching else 4
        if maxdist < low+1:
            return None
        distance = int(rng.integers(low, maxdist))
        contact = (anchor[0]+distance*d[0], anchor[1]+distance*d[1])
        a = axis if teaching else int(rng.integers(2))
        mid = tuple(contact[k]+(d[k] if k == a else 0) for k in (0, 1))
        length = int(rng.choice([1, 3, 5], p=[.15, .65, .20]))
        if teaching:
            length = int(rng.choice([3, 5]))
        cells = tuple((mid[0]+(t if a == 1 else 0), mid[1]+(t if a == 0 else 0))
                      for t in range(-length//2+1, length//2+1))
        if not all(_inside(p, (h, w)) for p in cells):
            return None
        if any(p in _blocked(q['cells']) for q in [source]+mirrors for p in cells):
            return None
        color = palette[3+j] if teaching else int(_choose(rng, palette[2:]))
        mirrors.append(dict(cells=cells, mid=mid, color=color,
                            axes=(0, 1) if length == 1 else (a,)))
        anchor = contact
        d = tuple(-v if k == a else v for k, v in enumerate(d))
    try:
        path, contacts = _trace((h, w), source, mirrors)
    except ValueError:
        return None
    if len(contacts) != nmirror or len({i for i, a, p, c in contacts}) != nmirror:
        return None
    if teaching and len(path) < 27:
        return None
    if not teaching and len(path) < 15:
        return None
    grid = np.full((h, w), bg, dtype=np.int8)
    for obj in [source]+mirrors:
        for p in obj['cells']:
            grid[p] = obj['color']
    return grid, bg, carrier, source, mirrors, path


def _pool(rng):
    shapes = {_key(S4), _key(L4), _key(((0, 0),)), _key(((0, 0), (0, 1)))}
    for n in range(3, 7):
        shapes.add(_key(tuple((0, c) for c in range(n))))
        shapes.add(_key(tuple([(r, 0) for r in range(n-1)] + [(n-2, 1)])))
    shapes.update({_key(((0, 0), (0, 1), (1, 0), (1, 1))),
                   _key(((0, 0), (0, 1), (0, 2), (1, 1))),
                   _key(((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)))})
    for _ in range(10):
        cells = {(0, 0)}
        for _ in range(int(rng.integers(3, 7))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in DIR4} - cells)
            cells.add(_choose(rng, frontier))
        if max(_size(_norm(cells))) <= 4:
            shapes.add(_key(cells))
    # Exclude internally holed pieces: the stencil carrier must stay connected.
    result = []
    for s in sorted(shapes):
        h, w = _size(s)
        mask = np.ones((h+2, w+2), dtype=bool)
        for r, c in s:
            mask[r+1, c+1] = False
        if len(_components(mask)) == 1:
            result.append(s)
    return result


def _construct(rng, pool, teaching=False, axis=0, swap=False, required_shape=None):
    hardware = _hardware_scene(rng, teaching, axis)
    if hardware is None:
        return None
    inp, bg, carrier, source, mirrors, path = hardware
    h, w = inp.shape
    light = defaultdict(set)
    for p, color in path:
        light[p].add(color)
    shades = list(dict.fromkeys(color for p, color in path))
    if len(shades) < (3 if teaching else 2):
        return None
    shapes = [_key(S4), _key(L4)] if teaching else [pool[int(i)] for i in rng.choice(
        len(pool), int(rng.integers(1, 4)), replace=False)]
    if required_shape is not None and required_shape not in shapes:
        shapes[0] = required_shape
    # Different winners for equally sized and equally boxed teaching shapes.
    order = list(shades)
    if swap:
        order[0], order[1] = order[1], order[0]
    if not teaching:
        rng.shuffle(order)
    ballots = {}
    for i, shape in enumerate(shapes):
        winner = order[i % len(order)]
        others = [s for s in order if s != winner]
        if teaching and i == 0:
            ballot = [(winner, 2), (others[0], 1), (others[1], 1)]
        elif teaching:
            ballot = [(winner, 2), (order[0], 1)]
        else:
            mode = int(rng.integers(5))
            if mode == 0:
                ballot = [(winner, int(rng.integers(1, 4)))]
            elif mode in (1, 2) or len(others) == 1:
                n = int(rng.integers(2, 4))
                ballot = [(winner, n), (others[0], n-1)]
            else:
                n = 2 if mode == 3 else 3
                ballot = [(winner, n), (others[0], n-1), (others[1], n-1)]
        ballots[shape] = ballot
    # Make the stencil first. Cavity locations are packed freely, not in slots.
    sh, sw = int(rng.integers(6, 12)), int(rng.integers(7, 14))
    if not teaching and rng.random() < .3:
        sh, sw = int(rng.integers(6, 10)), int(rng.integers(14, 19))
    box = np.full((sh, sw), carrier, dtype=np.int8)
    border = {(r, c) for r in range(sh) for c in range(sw)
              if r in (0, sh-1) or c in (0, sw-1)}
    forbidden = set(border)
    cavity_placements = []
    for shape in shapes:
        variants = (_norm(S4),) if teaching and shape == _key(S4) else _variants(shape)
        options = _place_options((sh, sw), variants, forbidden)
        if not options:
            return None
        cells = _choose(rng, options)
        cavity_placements.append((shape, cells))
        forbidden |= _blocked(cells)
    for _ in range(int(rng.integers(0, 4))):
        shape = _choose(rng, shapes)
        options = _place_options((sh, sw), _variants(shape), forbidden)
        if options:
            cells = _choose(rng, options)
            cavity_placements.append((shape, cells))
            forbidden |= _blocked(cells)
    for shape, cells in cavity_placements:
        for p in cells:
            box[p] = bg
    if len(_components(box == carrier)) != 1:
        return None
    hardware_cells = {p for obj in [source]+mirrors for p in obj['cells']}
    positions = [(y, x) for y in range(h-sh+1) for x in range(w-sw+1)
                 if not any(y <= p[0] < y+sh and x <= p[1] < x+sw
                            for p in set(light) | _blocked(hardware_cells))]
    if not positions:
        return None
    y, x = _choose(rng, positions)
    inp[y:y+sh, x:x+sw] = box
    rect = {(r, c) for r in range(y, y+sh) for c in range(x, x+sw)}
    forbidden = _blocked(hardware_cells | rect)
    specs = []
    for shape in shapes:
        for color, n in ballots[shape]:
            variants = _variants(shape)
            hits = None
            if teaching and shape == _key(S4):
                raw = _norm(S4)
                if color == ballots[shape][0][0]:
                    variants = tuple(s for s in _variants(shape) if s not in _variants(raw, False))
                    hits = 1
                else:
                    variants = _variants(raw, False)
                    hits = 2
            specs.extend((shape, color, variants, hits) for _ in range(n))
    # Constrained hit patterns are placed first; bounded restarts handle packing.
    specs.sort(key=lambda s: (s[3] is None, -len(s[0])))
    for shape, color, variants, hits in specs:
        options = _place_options(inp.shape, variants, forbidden, light, color, hits)
        if not options:
            return None
        cells = _choose(rng, options)
        for p in cells:
            inp[p] = carrier
        forbidden |= _blocked(cells)
    # Abstention is visible: unlit twins are legitimate objects, not random noise.
    abstainers = [shapes[0]]*3 if teaching else [_choose(rng, shapes) for _ in range(int(rng.integers(1, 5)))]
    for shape in abstainers:
        options = _place_options(inp.shape, _variants(shape), forbidden, light)
        if not options:
            return None
        cells = _choose(rng, options)
        for p in cells:
            inp[p] = carrier
        forbidden |= _blocked(cells)
    out = box.copy()
    for shape, cells in cavity_placements:
        for p in cells:
            out[p] = ballots[shape][0][0]
    try:
        data = _decode(inp)
        actual, election = _tally(data)
    except ValueError:
        return None
    if not np.array_equal(out, actual):
        return None
    if teaching and not all(_fails(data, out, mode) for mode in
                            ('area', 'box', 'rotation', 'pixels', 'straight', 'monochrome', 'unlit')):
        return None
    return dict(input=inp, output=out)


def _counts(num_train, num_test):
    for name, n, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= n <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _fingerprint(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    pool = _pool(rng)
    reserved = _choose(rng, [s for s in pool if s not in (_key(S4), _key(L4)) and len(s) >= 3])
    train_pool = [s for s in pool if s != reserved]
    axis = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', num_train), ('test', num_test)):
        for i in range(n):
            teaching = split == 'train' and i < 2
            for attempt in range(6000):
                pair = _construct(rng, train_pool if split == 'train' else pool, teaching=teaching,
                                  axis=axis if i == 0 else 1-axis, swap=i == 1,
                                  required_shape=reserved if split == 'test' else None)
                if pair is None or _fingerprint(pair['input']) in seen:
                    continue
                if teaching and i == 1:
                    a = _tally(_decode(puzzle['train'][0]['input']))[1]['winners']
                    b = _tally(_decode(pair['input']))[1]['winners']
                    if a[_key(S4)] == b[_key(S4)]:
                        continue
                break
            else:
                raise RuntimeError('bounded optical-scene packing exhausted')
            seen.add(_fingerprint(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, axes, history = set(), [], set(), defaultdict(set)
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _fingerprint(inp) not in seen, 'duplicate input'
            seen.add(_fingerprint(inp))
            data = _decode(inp)
            expected, election = _tally(data)
            assert np.array_equal(expected, out), 'wrong crop or election result'
            assert out.shape == data['box'].shape
            assert np.array_equal(out[data['box'] == data['carrier']],
                                  data['box'][data['box'] == data['carrier']])
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert len(election['exposures']) < len(data['tokens']), 'missing abstaining object'
            assert any(n >= 2 for cells, color, n in election['exposures']) or split == 'test' or len(training) >= 2
            # All objects are isolated even when their ink color agrees.
            groups = [data['source']['cells']] + [m['cells'] for m in data['mirrors']] + data['tokens']
            r0, c0, r1, c1 = data['bbox']
            groups.append(tuple((r, c) for r in range(r0, r1) for c in range(c0, c1)))
            for i, cells in enumerate(groups):
                near = _blocked(cells)
                assert not any(p in near for other in groups[i+1:] for p in other), 'objects touch'
            if split == 'train':
                training.append((data, out, election))
                axes.update(a for i, a, p, c in data['contacts'] if len(data['mirrors'][i]['cells']) > 1)
                for shape, color in election['winners'].items():
                    history[shape].add(color)
    assert axes == {0, 1}, 'training must expose both reflection axes'
    for mode in ('area', 'box', 'rotation', 'pixels', 'straight', 'monochrome', 'unlit'):
        assert any(_fails(data, out, mode) for data, out, e in training), f'untaught shortcut: {mode}'
    assert any(len(colors) > 1 for colors in history.values()), 'fixed shape-color shortcut'
    assert any(len(v) >= 3 and max(v.values())*2 <= sum(v.values())
               for d, o, e in training for v in e['votes'].values()), 'plurality evidence missing'
    known = {s for d, o, e in training for s in e['votes']}
    assert any(set(_tally(_decode(p['input']))[1]['votes']) - known for p in puzzle['test']), 'test vocabulary shift missing'


def self_test():
    """Stream 42 seeds; keep only small signature sets, never save puzzles."""
    shapes, vocab, signatures, axes, ballots, counts, palettes = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _decode(pair['input'])
                _, election = _tally(data)
                digest.update(pair['input'].tobytes())
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                vocab.update(election['votes'])
                axes.add(tuple(a for i, a, p, c in data['contacts']))
                ballots.update(tuple(sorted(v.values(), reverse=True)) for v in election['votes'].values())
                counts.add((len(data['tokens']), len(data['cavities']), len(data['mirrors'])))
                palettes.add((data['bg'], data['carrier']))
        signatures.add(digest.digest())
    assert len(signatures) == 42
    assert len(shapes) >= 100 and len(vocab) >= 20 and len(counts) >= 25
    assert len(axes) >= 8 and len(ballots) >= 5 and len(palettes) >= 30
    a, b = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            _counts(train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {'train': a['train'], 'test': [dict(a['test'][0])]}
    bad['test'][0]['output'] = bad['test'][0]['output'].copy()
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt crop accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), dimension_pairs=len(shapes),
                shape_classes=len(vocab), contact_axis_sequences=len(axes),
                ballot_patterns=len(ballots), count_combinations=len(counts), role_pairs=len(palettes))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
