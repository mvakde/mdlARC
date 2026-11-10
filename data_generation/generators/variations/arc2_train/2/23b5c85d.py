"""Generate scenes whose answer is the smallest rectangular footprint."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    'ed4e5be7b5c57254ebfb4b4ca172804b196fbc8669e22168e652ea5e25c13272',
    '2c2270bd0d4d7c1fc9e92e0991c6a6682d494702dc519b3ce24cdbbc37491f11',
    'b2fac621c6136462107f2b26943d224d764cc886cd23b84a643c66a1156394dc',
    '3e03a5e9a40acc6bf0d332bc74108084cc4a6ee81fb342acfba8b4e784ce894a',
    '1f8a3700202b3c4a752a7da5b8ee8c4d64c295fa1c2d276ba24aa0885337f976',
    'fbf07933f16e7c8d3a2698fe8bde4fafc8d77db10987ce196f25634bc2045adc',
}


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return int(seed)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer from {low} to {high}')
    return int(value)


def _counts(rng, num_train, num_test):
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _box(rect):
    r, c, h, w = rect[:4]
    return r, r + h - 1, c, c + w - 1


def _separate(a, b, gap=0):
    ar0, ar1, ac0, ac1 = _box(a)
    br0, br1, bc0, bc1 = _box(b)
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _relation(target, other):
    if _separate(target, other):
        return 'isolated'
    tr0, tr1, tc0, tc1 = _box(target)
    br0, br1, bc0, bc1 = _box(other)
    if br0 < tr0 and tr1 < br1 and bc0 < tc0 and tc1 < bc1:
        return 'contained'
    if br0 < tr0 and tr1 < br1:
        if tc0 < bc0 <= tc1 < bc1:
            return 'left'
        if bc0 < tc0 <= bc1 < tc1:
            return 'right'
    if bc0 < tc0 and tc1 < bc1:
        if tr0 < br0 <= tr1 < br1:
            return 'top'
        if br0 < tr0 <= br1 < tr1:
            return 'bottom'
    if not (tr0 <= br0 and tr1 >= br1) and not (br0 <= tr0 and br1 >= tr1):
        if not (tc0 <= bc0 and tc1 >= bc1) and not (bc0 <= tc0 and bc1 >= tc1):
            return 'corner'
    return 'complex'


def _components(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == background:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0 <= yy < height and 0 <= xx < width and not seen[yy, xx] and int(grid[yy, xx]) == color:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            r0 = min(y for y, _ in cells)
            r1 = max(y for y, _ in cells)
            c0 = min(x for _, x in cells)
            c1 = max(x for _, x in cells)
            result.append((r0, c0, r1-r0+1, c1-c0+1, color, len(cells)))
    return result


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    assert int(np.sum(counts == counts[index])) == 1
    return int(values[index])


def _digest(input_grid, output_grid):
    shape = (*input_grid.shape, *output_grid.shape)
    payload = str(shape).encode() + input_grid.astype(np.uint8).tobytes() + output_grid.astype(np.uint8).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _signature(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _analyze(input_grid, output_grid):
    bg = _background(input_grid)
    rects = _components(input_grid, bg)
    assert 2 <= len(rects) <= 7
    # Every visible color group determines the complete footprint from its
    # extrema; any occluded cells lie underneath another such footprint.
    order = {i: set() for i in range(len(rects))}
    for i, rect in enumerate(rects):
        r, c, h, w, color, visible = rect
        assert visible > 0
        patch = input_grid[r:r+h, c:c+w]
        assert not np.any(patch == bg)
        assert any(np.all(patch[y, :] == color) for y in range(h)) or any(np.all(patch[:, x] == color) for x in range(w))
        for y in range(r, r+h):
            for x in range(c, c+w):
                observed = int(input_grid[y, x])
                if observed != color:
                    candidates = [j for j, other in enumerate(rects) if j != i and other[4] == observed and other[0] <= y < other[0]+other[2] and other[1] <= x < other[1]+other[3]]
                    assert len(candidates) == 1
                    order[i].add(candidates[0])
    done = set()
    while len(done) < len(rects):
        available = [i for i in range(len(rects)) if i not in done and not (order[i] - done)]
        assert available
        done.add(available[0])
    areas = [rect[2] * rect[3] for rect in rects]
    minimum = min(areas)
    assert areas.count(minimum) == 1
    target = rects[areas.index(minimum)]
    assert all(rect[5] > minimum for rect in rects if rect != target)
    r, c, h, w, color, visible = target
    assert visible == minimum
    assert np.all(input_grid[r:r+h, c:c+w] == color)
    assert output_grid.shape == (h, w)
    assert np.all(output_grid == color)
    return {'background': bg, 'rectangles': rects, 'target': target,
            'relations': [_relation(target, rect) for rect in rects if rect != target]}


def _dims(rng, special):
    if special:
        return ((2, 9), (9, 2), (2, 8), (8, 2),
                (1, 11), (11, 1))[int(rng.integers(6))]
    if rng.random() < .13:
        length = int(rng.integers(4, 13))
        return (1, length) if rng.random() < .5 else (length, 1)
    menu = [(h, w) for h in range(2, 8) for w in range(2, 12)
            if 4 <= h*w <= 24]
    return menu[int(rng.integers(len(menu)))]


def _canvas(rng, th, tw, special):
    if special:
        return int(rng.integers(21, 31)), int(rng.integers(21, 31))
    if rng.random() < .22:
        short = int(rng.integers(10, 19))
        long = int(rng.integers(22, 31))
        height, width = (short, long) if rng.random() < .5 else (long, short)
        return (height, width) if th+1 <= height and tw+1 <= width else None
    band = int(rng.choice(3, p=[0.28, 0.34, 0.38]))
    lo, hi = ((10, 15), (16, 23), (24, 30))[band]
    lo_h, lo_w = max(lo, th+2), max(lo, tw+2)
    if lo_h > hi or lo_w > hi:
        return None
    return int(rng.integers(lo_h, hi+1)), int(rng.integers(lo_w, hi+1))


def _location(rng, limit):
    if limit == 0:
        return 0
    if rng.random() < 0.28:
        return 0 if rng.random() < 0.5 else limit
    return int(rng.integers(0, limit+1))


def _filler_dims(rng, minimum, height, width, special=False):
    if special:
        if minimum == 11:
            return (3, 4) if rng.integers(0, 2) else (4, 3)
        return (4, 5) if rng.integers(0, 2) else (5, 4)
    for _ in range(30):
        h = int(rng.integers(1, min(13, height+1)))
        w = int(rng.integers(1, min(13, width+1)))
        if minimum < h*w <= max(minimum+55, 65):
            return h, w
    return None


def _pick_rect(rng, height, width, h, w, target, existing, desired):
    positions = [(r, c) for r in range(height-h+1) for c in range(width-w+1)]
    for k in rng.permutation(len(positions)):
        r, c = positions[int(k)]
        candidate = (r, c, h, w)
        relation = _relation(target, candidate)
        if desired in ('isolated','contained','left','right','top','bottom','corner'):
            if relation != desired:
                continue
        elif desired == 'free':
            if not all(_separate(candidate, other) for other in existing):
                continue
        elif desired == 'group':
            if not any(not _separate(candidate, other) and _relation(candidate, other) in ('corner','left','right','top','bottom','contained') for other in existing):
                continue
        if any(candidate == other for other in existing):
            continue
        # Avoid identical-color component ambiguity from direct face adjacency
        # after recoloring; color assignment handles separated duplicates.
        return candidate
    return None


def _nested_filler(rng, existing, minimum):
    for base in rng.permutation(len(existing)):
        r, c, h, w = existing[int(base)]
        if h < 4 or w < 4:
            continue
        candidates = [(hh, ww) for hh in range(2, h-1)
                      for ww in range(2, w-1) if hh*ww > minimum]
        if not candidates:
            continue
        hh, ww = candidates[int(rng.integers(len(candidates)))]
        return (r + int(rng.integers(1, h-hh)),
                c + int(rng.integers(1, w-ww)), hh, ww)
    return None


def _render_scene(height, width, rects, colors, background):
    grid = np.full((height, width), background, dtype=np.int8)
    for (r, c, h, w), color in zip(rects[1:]+rects[:1], colors[1:]+colors[:1]):
        grid[r:r+h, c:c+w] = color
    return grid


def _pair(rng, special, regime):
    for _ in range(180):
        th, tw = _dims(rng, special)
        canvas = _canvas(rng, th, tw, special)
        if canvas is None:
            continue
        height, width = canvas
        target = (_location(rng, height-th), _location(rng, width-tw), th, tw)
        minimum = th*tw
        if special:
            count = 3
            mode = 'isolated'
        else:
            count = int(rng.choice([2,3,4,5,6], p=[.20,.22,.24,.20,.14] if regime == 'source_train' else [.12,.20,.23,.25,.20]))
            mode = str(rng.choice(['isolated','contained','left','right','top','bottom','corner'], p=[.26,.19,.11,.11,.11,.11,.11]))
        rects = [target]
        failed = False
        for index in range(1, count):
            desired = mode if index == 1 else ('free' if special else str(rng.choice(['free','group','inside','isolated'], p=[.37,.30,.22,.11])))
            found = None
            if desired == 'inside':
                found = _nested_filler(rng, rects[1:], minimum)
            for _ in range(28):
                if found is not None:
                    break
                dims = ((4, 4) if minimum == 11 else (5, 5)) if special and index == 2 else _filler_dims(rng, minimum, height, width, special and index == 1)
                if dims is None:
                    continue
                h, w = dims
                if h > height or w > width:
                    continue
                found = _pick_rect(rng, height, width, h, w, target,
                                   rects if special and index > 1 else rects[1:],
                                   'group' if desired == 'inside' else desired)
                if found is not None:
                    break
            if found is None:
                failed = True
                break
            rects.append(found)
        if failed:
            continue
        palette = [int(x) for x in rng.permutation(10)]
        background = palette[0]
        colors = palette[1:count+1]
        # Reuse a fill only when its objects are separated by background.
        reuse_pairs = [(i,j) for i in range(count) for j in range(i+1,count) if _separate(rects[i], rects[j], 1)]
        if special and (0, 1) not in reuse_pairs:
            continue
        if reuse_pairs and (special or rng.random() < .34):
            pair = (0, 1) if special else reuse_pairs[int(rng.integers(len(reuse_pairs)))]
            colors[pair[1]] = colors[pair[0]]
        grid = _render_scene(height, width, rects, colors, background)
        output = np.full((th, tw), colors[0], dtype=np.int8)
        try:
            analysis = _analyze(grid, output)
        except AssertionError:
            continue
        reconstructed = sorted((r[:5] for r in analysis['rectangles']))
        intended = sorted((*rect, color) for rect, color in zip(rects, colors))
        if reconstructed != intended:
            continue
        if special:
            # Keep the area/perimeter disagreement visible in training.
            competitor = rects[1]
            if competitor[2]*competitor[3] <= minimum or 2*(competitor[2]+competitor[3]) >= 2*(th+tw):
                continue
        return {'input': grid, 'output': output}
    raise ValueError('could not construct a rectangle scene')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    # The sole puzzle-level convention is the minimum-footprint rule.
    # Colors, scale, placement, and object count are visible pair properties.
    special_index = int(rng.integers(num_train))
    result = {'train': [], 'test': []}
    signatures = set()
    for split, count in (('train',num_train), ('test',num_test)):
        for index in range(count):
            for _ in range(10):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(pair_seed)
                pair = _pair(pair_rng, split == 'train' and index == special_index,
                             'source_train' if split == 'train' else 'target_test')
                signature = _signature(pair['input'])
                if signature not in signatures:
                    signatures.add(signature)
                    result[split].append(pair)
                    break
            else:
                raise ValueError('could not produce unique inputs')
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train','test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    input_signatures = set()
    pair_signatures = set()
    analyses = {'train': [], 'test': []}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input','output'}
            for grid in (pair['input'], pair['output']):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            inp, out = pair['input'], pair['output']
            assert not np.array_equal(inp, out)
            isig = _signature(inp)
            psig = (isig, _signature(out))
            assert isig not in input_signatures and psig not in pair_signatures
            input_signatures.add(isig)
            pair_signatures.add(psig)
            assert _digest(inp, out) not in _SOURCE_DIGESTS
            analyses[split].append(_analyze(inp, out))
    assert any(any(r[2]*r[3] > a['target'][2]*a['target'][3] and
                   r[2]+r[3] < a['target'][2]+a['target'][3]
                   for r in a['rectangles'] if r != a['target'])
               for a in analyses['train'])
    assert any(len({r[4] for r in a['rectangles']}) < len(a['rectangles'])
               for a in analyses['train'])
    assert any(sum(r[5] for r in a['rectangles'] if r[4] == a['target'][4]) >
               min(sum(r[5] for r in a['rectangles'] if r[4] == color)
                   for color in {r[4] for r in a['rectangles']})
               for a in analyses['train'])


def _same(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ('input','output'))
        for s in ('train','test'))


def self_test() -> None:
    assert _same(generate_puzzle(37), generate_puzzle(37))
    assert not _same(generate_puzzle(37), generate_puzzle(38))
    for t, e in ((2,1),(10,1),(10,4),(7,3)):
        p = generate_puzzle(1000+t*10+e, t, e)
        assert (len(p['train']),len(p['test'])) == (t,e)
    for t, e in ((None,1),(None,4),(2,None),(10,None)):
        p = generate_puzzle(2000+(t or 0)*10+(e or 0), t, e)
        assert t is None or len(p['train']) == t
        assert e is None or len(p['test']) == e
    for kw in ({'num_train':1},{'num_train':11},{'num_test':0},
               {'num_test':5},{'num_train':2,'num_test':2},
               {'num_train':3,'num_test':4},{'num_train':2.5,'num_test':1},
               {'num_train':True,'num_test':1}):
        try:
            generate_puzzle(123, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    puzzles = [generate_puzzle(i) for i in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    backgrounds, shapes, outputs = set(), set(), set()
    counts = {'train': set(), 'test': set()}
    relations = {'train': set(), 'test': set()}
    reused = compact = edge = grouped = nested = corner = elongated = near = one_cell = 0
    rarest_false = disagreement = combined = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_bg = set()
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                a = _analyze(inp, out)
                backgrounds.add(a['background'])
                local_bg.add(a['background'])
                shapes.add(inp.shape)
                outputs.add(out.shape)
                rects, target = a['rectangles'], a['target']
                counts[split].add(len(rects))
                relations[split].update(a['relations'])
                same_color = len({r[4] for r in rects}) < len(rects)
                reused += same_color
                compact += max(inp.shape) <= 15
                elongated += max(inp.shape) / min(inp.shape) >= 1.5
                one_cell += 1 in out.shape
                near += sorted(r[2]*r[3] for r in rects)[1] - out.size <= 3
                touches = any(r[0] == 0 or r[1] == 0 or r[0]+r[2] == inp.shape[0] or r[1]+r[3] == inp.shape[1] for r in rects)
                edge += touches
                corner += any((r[0] == 0 or r[0]+r[2] == inp.shape[0]) and
                              (r[1] == 0 or r[1]+r[3] == inp.shape[1]) for r in rects)
                groups = [(x,y) for i,x in enumerate(rects) for y in rects[i+1:]
                          if x != target and y != target and not _separate(x,y)]
                grouped += bool(groups)
                nested += any((x[0] <= y[0] and x[1] <= y[1] and
                               x[0]+x[2] >= y[0]+y[2] and x[1]+x[3] >= y[1]+y[3]) or
                              (y[0] <= x[0] and y[1] <= x[1] and
                               y[0]+y[2] >= x[0]+x[2] and y[1]+y[3] >= x[1]+x[3])
                              for x,y in groups)
                combined += bool(groups) and (same_color or any(r != 'isolated' for r in a['relations']))
                color_totals = {c: sum(r[5] for r in rects if r[4] == c)
                                for c in {r[4] for r in rects}}
                rarest_false += color_totals[target[4]] > min(color_totals.values())
                disagreement += any(r[2]*r[3] > out.size and
                                    r[2]+r[3] < sum(out.shape) for r in rects)
        assert len(local_bg) >= 2
    assert len(backgrounds) >= 7 and len(shapes) >= 35 and len(outputs) >= 20
    assert {2,3,4,5,6} <= counts['train'] and {2,3,4,5,6} <= counts['test']
    for split in ('train','test'):
        assert {'isolated','contained','left','right','top','bottom','corner'} <= relations[split]
    assert reused >= 50 and rarest_false >= 20 and disagreement >= 42
    assert compact >= 20 and elongated >= 20 and edge >= 40 and corner >= 10
    assert grouped >= 30 and nested >= 10 and combined >= 15
    assert near >= 20 and one_cell >= 20
