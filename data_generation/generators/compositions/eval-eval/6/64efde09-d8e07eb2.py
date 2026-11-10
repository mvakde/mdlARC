"""Broadcast locks: a pooled, oriented ray code selects exact board glyphs.

Every convention used by solve is visible in its one input. Construction draws
labels directly; the decoder re-segments the returned pixels for validation.
"""

import itertools
import numpy as np


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr) + 1, max(cc) + 1


def _trimkey(a, bg):
    rr, cc = np.where(a != bg)
    assert len(rr)
    return _key(a[rr.min():rr.max()+1, cc.min():cc.max()+1])


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        todo, cells = [(r, c)], []
        seen[r, c] = True
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < len(mask) and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        result.append(cells)
    return result


def _frame(a, color):
    a[0, :] = a[-1, :] = color
    a[:, 0] = a[:, -1] = color


def _line(positions, n):
    positions = set(positions)
    if len(positions) != n:
        return None
    if len({p // n for p in positions}) == 1:
        return 'row'
    if len({p % n for p in positions}) == 1:
        return 'col'
    return None


def _ray(start, direction, shape):
    r, c = start
    dy, dx = direction
    result = []
    r, c = r + dy, c + dx
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        result.append((r, c))
        r, c = r + dy, c + dx
    return tuple(result)


def _grid(a):
    a = np.asarray(a)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _rectangles(grid, bg):
    """Find the one frame color, using complete hollow rectangle components."""
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        boards, strips = [], []
        for cells in _components(grid == color):
            r, c, y, x = _bbox(cells)
            h, w = y-r, x-c
            if min(h, w) < 5:
                continue
            if len(cells) != 2*h + 2*w - 4:
                continue
            a = grid[r:y, c:x]
            if not (np.all(a[[0, -1]] == color)
                    and np.all(a[:, [0, -1]] == color)
                    and np.all(a[1:-1, 1:-1] != color)):
                continue
            if h == w and h in (8, 11):
                boards.append((r, c, y, x))
            if min(h, w) == 5 and max(h, w) in (8, 11, 14):
                strips.append((r, c, y, x))
        if 2 <= len(boards) == len(strips) <= 3:
            candidates.append((color, boards, strips))
    assert len(candidates) == 1, 'ambiguous frame color'
    return candidates[0]


def _module_options(grid, board, strip):
    """Two local longitudinal directions; the common colored strip resolves them."""
    r = min(board[0], strip[0])
    c = min(board[1], strip[1])
    y = max(board[2], strip[2])
    x = max(board[3], strip[3])
    crop = grid[r:y, c:x]
    roles = np.zeros(crop.shape, np.int8)
    for value, rect in ((1, board), (2, strip)):
        a, b, d, e = rect
        roles[a-r:d-r, b-c:e-c] = value
    options = []
    bsize = board[2] - board[0]
    for t in range(8):
        labels = _orient(roles, t)
        h, w = labels.shape
        if h != bsize + 6 or w not in (8, 11, 14):
            continue
        if not np.all(labels[-5:] == 2):
            continue
        rr, cc = np.where(labels == 1)
        if (rr.min() != 0 or rr.max() != bsize-1 or
                cc.max()-cc.min()+1 != bsize):
            continue
        offset = int(cc.min())
        if offset not in ((w-bsize)//2, (w-bsize+1)//2):
            continue
        canonical = _orient(crop, t)
        options.append({'r': r, 'c': c, 'crop': crop, 't': t,
                        'canonical': canonical, 'strip': canonical[-5:],
                        'offset': offset, 'b': bsize,
                        'rect': (r, c, y, x)})
    return options


def _decode(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    assert sum(counts == counts.max()) == 1
    bg = int(values[np.argmax(counts)])
    frame, boards, strips = _rectangles(grid, bg)
    option_groups, used = [], set()
    for strip in strips:
        matches = [(i, _module_options(grid, board, strip))
                   for i, board in enumerate(boards)]
        matches = [(i, opts) for i, opts in matches if opts]
        assert len(matches) == 1, 'board/strip pairing is ambiguous'
        index, options = matches[0]
        assert index not in used
        used.add(index)
        option_groups.append(options)
    # The reference coordinate system is chosen from visible strip pixels.
    reference = min(option_groups[0], key=lambda m: _key(m['strip']))['strip']
    assert len({_key(_orient(reference, t)) for t in range(8)}) == 8
    modules = []
    occupied = np.zeros(grid.shape, bool)
    inks = set()
    for options in option_groups:
        matches = [m for m in options if np.array_equal(m['strip'], reference)]
        assert len(matches) == 1, 'strips are not uniquely corresponding copies'
        m = matches[0]
        a, bsize, offset = m['canonical'], m['b'], m['offset']
        n, q = (bsize-2)//3, (a.shape[1]-2)//3
        assert n in (2, 3) and q in (n, n+1)
        expected = np.full(a.shape, bg, np.int8)
        _frame(expected[:bsize, offset:offset+bsize], frame)
        _frame(expected[-5:], frame)
        keys, tokens = [], []
        for j in range(q):
            token = a[-4:-1, 1+3*j:4+3*j].copy()
            keys.append(token)
            expected[-4:-1, 1+3*j:4+3*j] = token
        for y, x in itertools.product(range(n), repeat=2):
            token = a[1+3*y:4+3*y, offset+1+3*x:offset+4+3*x].copy()
            tokens.append(token)
            expected[1+3*y:4+3*y, offset+1+3*x:offset+4+3*x] = token
        assert np.array_equal(a, expected), 'ink outside framed glyph lattices'
        for token in keys + tokens:
            colors = set(map(int, np.unique(token[token != bg])))
            assert len(colors) == 1 and frame not in colors
            inks.update(colors)
        lookup = {_key(token): j for j, token in enumerate(tokens)}
        assert len(lookup) == n*n
        assert len({_trimkey(token, bg) for token in tokens}) == n*n
        assert len({_key(token) for token in keys}) == q
        assert all(_key(token) in lookup for token in keys)
        m.update(n=n, q=q, keys=keys, tokens=tokens,
                 positions=[lookup[_key(token)] for token in keys])
        # Canonical cells mapped back to the original displayed crop.
        coords = _orient(np.arange(m['crop'].size).reshape(m['crop'].shape), m['t'])
        m['coords'] = coords
        rays = []
        for j in range(q):
            p, inner = int(coords[-1, 2+3*j]), int(coords[-2, 2+3*j])
            pr, pc = divmod(p, m['crop'].shape[1])
            ir, ic = divmod(inner, m['crop'].shape[1])
            rays.append(_ray((m['r']+pr, m['c']+pc), (pr-ir, pc-ic), grid.shape))
        m['rays'] = rays
        r, c, y, x = m['rect']
        assert not np.any(occupied[r:y, c:x])
        occupied[r:y, c:x] = True
        modules.append(m)
    assert len({(m['n'], m['q']) for m in modules}) == 1
    signals = [(int(r), int(c), int(grid[r, c]))
               for r, c in np.argwhere((grid != bg) & ~occupied)]
    assert signals
    code, allocations = {}, [dict() for _ in modules]
    for r, c, color in signals:
        assert r in (0, len(grid)-1) or c in (0, grid.shape[1]-1)
        assert color not in inks | {bg, frame}
        matches = []
        for i, m in enumerate(modules):
            for j, ray in enumerate(m['rays']):
                if (len(ray) >= 2 and ray[-1] == (r, c)
                        and all(not occupied[p] for p in ray)):
                    matches.append((i, j))
        assert len(matches) == 1, 'ambiguous boundary observation'
        i, j = matches[0]
        assert j not in code or code[j] == color
        code[j] = color
        allocations[i][j] = color
    assert 1 <= len(code) <= modules[0]['n']
    painted = set()
    for m in modules:
        for j, color in code.items():
            ray = m['rays'][j]
            assert len(ray) >= 2 and all(not occupied[p] for p in ray)
            assert not painted.intersection(ray), 'intersecting broadcast lanes'
            painted.update(ray)
            assert all(int(grid[p]) in (bg, color) for p in ray)
        m['selected'] = [m['positions'][j] for j in code]
        m['line'] = _line(m['selected'], m['n'])
    return {'grid': grid, 'bg': bg, 'frame': frame, 'modules': modules,
            'code': code, 'allocations': allocations, 'inks': inks,
            'signals': signals}


def _paint(info, local_only=False, all_open=False):
    """The two flags are explicit concept-omission counterfactuals for QA."""
    grid, bg = info['grid'], info['bg']
    out = grid.copy()
    for i, m in enumerate(info['modules']):
        code = info['allocations'][i] if local_only else info['code']
        coords = m['coords']
        for j, color in code.items():
            y, x = divmod(m['positions'][j], m['n'])
            cells = coords[1+3*y:4+3*y, m['offset']+1+3*x:m['offset']+4+3*x]
            for label in cells.flat:
                r, c = divmod(int(label), m['crop'].shape[1])
                p = m['r']+r, m['c']+c
                if grid[p] == bg:
                    out[p] = color
        selected = [m['positions'][j] for j in code]
        if all_open or _line(selected, m['n']):
            for j, color in code.items():
                for p in m['rays'][j]:
                    out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer geometry, shared code, exact lookup and each gate from input only."""
    return _paint(_decode(input_grid))


def _mask_bank():
    bank = {}
    for bits in range(1, 512):
        mask = np.array([(bits >> j) & 1 for j in range(9)], bool).reshape(3, 3)
        rr, cc = np.where(mask)
        crop = mask[rr.min():rr.max()+1, cc.min():cc.max()+1]
        a = np.zeros((3, 3), bool)
        y, x = (3-len(crop))//2, (3-crop.shape[1])//2
        a[y:y+len(crop), x:x+crop.shape[1]] = crop
        bank[_key(a)] = a
    return tuple(bank.values())


_MASKS = _mask_bank()


def _vocabulary(rng, q, bg, frame, inks, teach):
    for _ in range(100):
        masks = [_MASKS[int(i)].copy()
                 for i in rng.choice(len(_MASKS), q, replace=False)]
        # Make shape and ink separately insufficient in the first lesson.
        colors = list(map(int, rng.choice(inks, q)))
        if teach:
            if np.all(masks[0]):
                continue
            masks[1] = masks[0].copy()
            colors[0], colors[1], colors[2] = inks[0], inks[1], inks[0]
        keys = []
        for mask, color in zip(masks, colors):
            a = np.full((3, 3), bg, np.int8)
            a[mask] = color
            keys.append(a)
        if len({_key(a) for a in keys}) != q:
            continue
        strip = np.full((5, 3*q+2), bg, np.int8)
        _frame(strip, frame)
        for j, token in enumerate(keys):
            strip[1:4, 1+3*j:4+3*j] = token
        if len({_key(_orient(strip, t)) for t in range(8)}) == 8:
            return keys, strip
    raise AssertionError('asymmetric vocabulary sampling exhausted')


def _make_module(rng, n, keys, strip, active, kind, bg, frame, inks, code):
    q, b = len(keys), 3*n+2
    w, h = strip.shape[1], b+6
    offset = (w-b)//2
    inp = np.full((h, w), bg, np.int8)
    _frame(inp[:b, offset:offset+b], frame)
    inp[-5:] = strip
    if kind == 'row':
        r = int(rng.integers(n))
        positions = [r*n+j for j in range(n)]
    elif kind == 'col':
        c = int(rng.integers(n))
        positions = [j*n+c for j in range(n)]
    else:
        # All scatter sets, not just a finite bank of failure templates.
        choices = [p for p in itertools.combinations(range(n*n), len(active))
                   if _line(p, n) is None]
        positions = list(choices[int(rng.integers(len(choices)))])
    if len(active) < n:
        positions = list(map(int, rng.choice(n*n, len(active), replace=False)))
    rng.shuffle(positions)
    tokens = [None]*(n*n)
    for j, p in zip(active, positions):
        tokens[p] = keys[j]
    remaining = [a for j, a in enumerate(keys) if j not in active]
    seen = {_trimkey(a, bg) for a in keys}
    candidates = list(itertools.product(range(len(_MASKS)), inks))
    rng.shuffle(candidates)
    for index, ink in candidates:
        if len(remaining) == n*n-len(active):
            break
        a = np.full((3, 3), bg, np.int8)
        a[_MASKS[index]] = ink
        if _trimkey(a, bg) not in seen:
            remaining.append(a)
            seen.add(_trimkey(a, bg))
    assert len(remaining) == n*n-len(active)
    rng.shuffle(remaining)
    for p in range(n*n):
        if tokens[p] is None:
            tokens[p] = remaining.pop()
        r, c = divmod(p, n)
        inp[1+3*r:4+3*r, offset+1+3*c:offset+4+3*c] = tokens[p]
    out = inp.copy()
    for j, p in zip(active, positions):
        r, c = divmod(p, n)
        block = out[1+3*r:4+3*r, offset+1+3*c:offset+4+3*c]
        block[block == bg] = code[j]
    labels = np.arange(inp.size).reshape(inp.shape)
    return {'input': inp, 'output': out, 'labels': labels,
            'open': _line(positions, n) is not None, 'n': n, 'q': q}


def _place(rng, modules, transforms, active):
    shapes = [_orient(m['input'], t).shape for m, t in zip(modules, transforms)]
    if len(shapes) == 2:
        (h1, w1), (h2, w2) = shapes
        if h1+h2+1 > 30 and w1+w2+1 > 30:
            return None
    for attempt in range(100):
        # Rectangles and squares of many sizes; only the physical packing matters.
        h, w = map(int, rng.integers(20, 31, 2))
        if attempt > 65:
            h, w = 30, 30
        placed = []
        for m, t, (mh, mw) in zip(modules, transforms, shapes):
            candidates = [(r, c) for r in range(h-mh+1) for c in range(w-mw+1)]
            rng.shuffle(candidates)
            for r, c in candidates:
                if any(not (r+mh < p['r'] or p['r']+p['h'] < r or
                            c+mw < p['c'] or p['c']+p['w'] < c) for p in placed):
                    continue
                labels = _orient(m['labels'], t)
                rays = []
                for j in range(m['q']):
                    a = int(m['labels'][-1, 2+3*j])
                    b = int(m['labels'][-2, 2+3*j])
                    ar, ac = map(int, np.argwhere(labels == a)[0])
                    br, bc = map(int, np.argwhere(labels == b)[0])
                    rays.append(_ray((r+ar, c+ac), (ar-br, ac-bc), (h, w)))
                if any(len(rays[j]) < 2 for j in active):
                    continue
                placed.append({'r': r, 'c': c, 'h': mh, 'w': mw,
                               't': t, 'module': m, 'rays': rays})
                break
            else:
                break
        if len(placed) != len(modules):
            continue
        occupied = set().union(*(set(itertools.product(range(p['r'], p['r']+p['h']),
                                                        range(p['c'], p['c']+p['w']))) for p in placed))
        used = set()
        for p in placed:
            good = True
            for j in active:
                ray = p['rays'][j]
                if occupied.intersection(ray) or used.intersection(ray):
                    good = False
                    break
                used.update(ray)
            if not good:
                break
        else:
            # Every selected port must have at least one unambiguous endpoint.
            endpoints = {}
            for i, p in enumerate(placed):
                for j, ray in enumerate(p['rays']):
                    if len(ray) >= 2 and not occupied.intersection(ray):
                        endpoints.setdefault(ray[-1], []).append((i, j))
            options = {j: [i for i, p in enumerate(placed)
                           if endpoints.get(p['rays'][j][-1]) == [(i, j)]] for j in active}
            if all(options.values()):
                return (h, w), placed, options
    return None


def _pair(rng, n, lesson=None, testing=False, repeat_beams=False):
    for attempt in range(180):
        palette = list(map(int, rng.permutation(10)))
        bg, frame = palette[:2]
        ink_count = int(rng.integers(2, 5))
        inks = palette[2:2+ink_count]
        beam_colors = palette[2+ink_count:]
        if lesson == 'pool' or (testing and n == 2):
            q = n+1
        elif lesson in ('closed', 'scale', 'partial'):
            q = n
        else:
            q = int(rng.choice([n, n+1]))
        keys, strip = _vocabulary(rng, q, bg, frame, inks, lesson == 'pool')
        k = n
        # Partial codes are taught as an extra lesson before they occur in tests.
        if lesson == 'partial':
            k = int(rng.integers(1, n))
        if lesson == 'pool':
            active = [0, 2]
        elif testing and n == 2:
            start = int(rng.integers(2))
            active = [start, start+1]
        else:
            active = sorted(map(int, rng.choice(q, k, replace=False)))
        colors = list(map(int, rng.choice(beam_colors, k, replace=False)))
        if lesson == 'closed' and repeat_beams:
            colors = [beam_colors[0]] * k
        elif lesson is None and repeat_beams and rng.random() < .3:
            colors = list(map(int, rng.choice(beam_colors[:2], k)))
        code = dict(zip(active, colors))
        count = 3 if n == 2 and q == 2 and rng.random() < .3 else 2
        kinds = [('row' if lesson == 'pool' else 'col' if lesson == 'closed'
                  else str(rng.choice(['row', 'col']))), 'scatter']
        if count == 3:
            kinds.append(str(rng.choice(['row', 'col', 'scatter'])))
        modules = [_make_module(rng, n, keys, strip, active, kind, bg, frame, inks, code)
                   for kind in kinds]
        base = int(rng.integers(8))
        if lesson in ('pool', 'scale'):
            transforms = [base, base ^ 4]
        elif testing:
            transforms = [base, (base//4)*4+(base+2)%4]
            if count == 3:
                transforms.append(int(rng.integers(8)))
        elif lesson == 'closed':
            transforms = [base, (base//4)*4+(base+1)%4]
            if count == 3:
                transforms.append(int(rng.integers(8)))
        else:
            transforms = list(map(int, rng.integers(8, size=count)))
        placement = _place(rng, modules, transforms, active)
        if placement is None:
            continue
        shape, placed, options = placement
        # Pool lesson uses incomplete observations on different locks; second
        # lesson puts the entire code on a CLOSED lock, including a dark winner.
        allocations = []
        if lesson == 'pool':
            allocations = [(active[0], 0), (active[1], 1)]
        elif lesson == 'closed' or testing:
            allocations = [(j, 1) for j in active]
        else:
            allocations = [(j, int(rng.choice(options[j]))) for j in active]
        if any(i not in options[j] for j, i in allocations):
            continue
        if lesson is None and not testing and rng.random() < .3:
            for j in active:
                for i in options[j]:
                    if (j, i) not in allocations and rng.random() < .4:
                        allocations.append((j, i))
        endpoints = [placed[i]['rays'][j][-1] for j, i in allocations]
        if any(max(abs(a[0]-b[0]), abs(a[1]-b[1])) <= 1
               for a, b in itertools.combinations(endpoints, 2)):
            continue
        inp = np.full(shape, bg, np.int8)
        out = inp.copy()
        for p in placed:
            r, c, h, w = (p[name] for name in ('r', 'c', 'h', 'w'))
            inp[r:r+h, c:c+w] = _orient(p['module']['input'], p['t'])
            out[r:r+h, c:c+w] = _orient(p['module']['output'], p['t'])
        for j, i in allocations:
            endpoint = placed[i]['rays'][j][-1]
            inp[endpoint] = out[endpoint] = code[j]
        for p in placed:
            if p['module']['open']:
                for j, color in code.items():
                    for position in p['rays'][j]:
                        out[position] = color
        try:
            info = _decode(inp)
            assert np.array_equal(out, _paint(info))
            assert not np.array_equal(inp, out)
            if lesson == 'pool':
                assert not np.array_equal(out, _paint(info, local_only=True))
                # Ensure an actually visible shape-and-color distinction.
                assert any(np.any(token == bg) for token in keys[::2])
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    raise RuntimeError('bounded scene construction exhausted')


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(name + ' outside its integer range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ne = _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    repeat_beams = bool(rng.integers(2))
    result = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            lesson = None
            if split == 'train' and i < 2:
                n, lesson = 2, ('pool', 'closed')[i]
            elif split == 'train' and i == 3:
                n, lesson = 3, 'partial'
            elif split == 'train' and i == 2:
                n, lesson = 3, 'scale'
            else:
                n = (3 if nt >= 3 else 2) if split == 'test' else int(rng.choice([2, 3]))
            for _ in range(10):
                pair = _pair(rng, n, lesson, testing=split == 'test',
                             repeat_beams=repeat_beams)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError('could not produce a distinct example')
    for split in result:
        result[split] = [result[split][int(i)] for i in rng.permutation(len(result[split]))]
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(_paint(info), out), 'incorrect lock completion'
            assert np.array_equal(out[inp != info['bg']], inp[inp != info['bg']])
            assert not np.array_equal(inp, out)
            if split == 'train':
                training.append(info)
            else:
                testing.append(info)
    assert any(all(len(a) < len(i['code']) for a in i['allocations']) for i in training)
    assert any(not np.array_equal(_paint(i), _paint(i, local_only=True)) for i in training)
    assert any(not np.array_equal(_paint(i), _paint(i, all_open=True)) for i in training)
    assert {'row', 'col'} <= {m['line'] for i in training for m in i['modules']}
    assert any(m['line'] is None for i in training for m in i['modules'])
    # The confusable keys must actually participate in selection, rather than
    # merely exist somewhere in a sampled inventory.
    shape_needed = color_needed = dark_winner = False
    for info in training:
        active = set(info['code'])
        keys = info['modules'][0]['keys']
        for j in active:
            for k, other in enumerate(keys):
                same_shape = np.array_equal(keys[j] != info['bg'], other != info['bg'])
                shape_needed |= k not in active and same_shape and not np.array_equal(keys[j], other)
                same_ink = int(keys[j][keys[j] != info['bg']][0]) == int(other[other != info['bg']][0])
                color_needed |= k in active and same_ink and not same_shape
        dark_winner |= any(m['line'] and not a
                           for m, a in zip(info['modules'], info['allocations']))
    assert shape_needed and color_needed and dark_winner
    if any(len(set(i['code'].values())) < len(i['code']) for i in testing):
        assert any(len(set(i['code'].values())) < len(i['code']) for i in training)
    if any(i['modules'][0]['n'] == 3 for i in testing):
        assert any(i['modules'][0]['n'] == 3 for i in training)


def _features(info):
    ms = info['modules']
    return {'n': ms[0]['n'], 'q': ms[0]['q'], 'count': len(ms),
            'ports': tuple(sorted(info['code'])), 'colors': tuple(info['code'].values()),
            'shape': info['grid'].shape, 'background': info['bg'],
            'directions': tuple((ray[1][0]-ray[0][0], ray[1][1]-ray[0][1])
                                for m in ms for ray in [m['rays'][next(iter(info['code']))]]),
            'outcomes': tuple(m['line'] for m in ms),
            'allocations': tuple(len(a) for a in info['allocations']),
            'ray_lengths': tuple(len(m['rays'][j]) for m in ms for j in info['code'])}


def self_test():
    records, fingerprints = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        assert all(np.array_equal(a[k], b[k]) for s in puzzle
                   for a, b in zip(puzzle[s], again[s]) for k in a)
        fingerprints.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for s in puzzle:
            for pair in puzzle[s]:
                info = _decode(pair['input'])
                records.append(_features(info))
                assert np.array_equal(solve(pair['input']), pair['output'])
    assert len(fingerprints) == 42
    assert {r['n'] for r in records} == {2, 3}
    assert {r['q'] for r in records} == {2, 3, 4}
    assert {r['count'] for r in records} == {2, 3}
    assert len({r['shape'] for r in records}) >= 20
    assert len({(r['q'], r['ports']) for r in records}) >= 9
    assert len({r['background'] for r in records}) >= 8
    assert len({d for r in records for d in r['directions']}) == 4
    assert any(len(set(r['colors'])) < len(r['colors']) for r in records)
    lengths = {length for r in records for length in r['ray_lengths']}
    assert min(lengths) == 2 and len(lengths) >= 8
    assert max(lengths) >= 4*min(lengths)
    sample = generate_puzzle(7001, 3, 2)
    for t in range(8):
        palette = np.roll(np.arange(10, dtype=np.int8), t)
        transformed = {s: [{k: _orient(palette[v], t) for k, v in p.items()}
                           for p in sample[s]] for s in sample}
        validate_generated_puzzle(transformed)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    for nt, ne in ((2, 1), (5, 4), (10, 4)):
        generate_puzzle(2000+nt, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (3, False), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')


if __name__ == '__main__':
    self_test()
