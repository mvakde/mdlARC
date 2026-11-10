"""Rotating room stencils: topology drives rotation, and coverage drives color.

Only the input decoder is used by solve. Construction retains clean geometry
for checking exact reconstruction, and computes labels with NumPy rotations.
Validation recomputes labels with explicit coordinate turns instead.
"""
from collections import deque
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be in 1..4 and less than num_train")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _rooms(walls):
    """Return actual bounded four-connected regions, never larger rectangles."""
    free = np.pad(~walls, 1, constant_values=True)
    seen = np.zeros(free.shape, dtype=bool)
    regions = []
    h, w = free.shape
    for r, c in np.argwhere(free):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        todo = deque([(r, c)])
        seen[r, c] = True
        cells, outside = [], False
        while todo:
            y, x = todo.popleft()
            cells.append((y - 1, x - 1))
            outside |= y in (0, h - 1) or x in (0, w - 1)
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and free[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        if not outside:
            region = np.zeros(walls.shape, dtype=bool)
            for y, x in cells:
                region[y, x] = True
            regions.append(region)
    return regions


def _wall_union(shape, boxes):
    a = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        a[t, l:r+1] = a[b, l:r+1] = True
        a[t:b+1, l] = a[t:b+1, r] = True
    return a


def _bbox(mask):
    p = np.argwhere(mask)
    assert len(p)
    t, l = map(int, p.min(0))
    b, r = map(int, p.max(0))
    return t, l, b, r


def _interiors(walls):
    regions = _rooms(walls)
    assert 1 <= len(regions) <= 8, "need 1..8 positive-area rooms"
    boxes = []
    for region in regions:
        t, l, b, r = _bbox(region)
        assert region.sum() == (b-t+1) * (r-l+1), "rooms must be rectangular"
        boxes.append((t-1, l-1, b+1, r+1))
    assert np.array_equal(_wall_union(walls.shape, boxes), walls), "unexplained wall pixels"
    # Independent planar cycle-rank check, correcting pixel-sized empty faces.
    edges = int((walls[:, :-1] & walls[:, 1:]).sum())
    edges += int((walls[:-1] & walls[1:]).sum())
    unit_faces = int((walls[:-1, :-1] & walls[1:, :-1]
                      & walls[:-1, 1:] & walls[1:, 1:]).sum())
    assert edges - int(walls.sum()) + 1 - unit_faces == len(regions)
    return np.logical_or.reduce(regions), regions


def _separated(a, b):
    t, l, d, r = a
    u, v, e, s = b
    return d + 1 < u or e + 1 < t or r + 1 < v or s + 1 < l


def _decode(input_grid):
    grid = _grid(input_grid)
    colors, frequencies = np.unique(grid, return_counts=True)
    assert np.sum(frequencies == frequencies.max()) == 1, "background must be unique mode"
    bg = int(colors[frequencies.argmax()])
    t, l, b, r = _bbox(grid != bg)
    assert b-t == r-l and b-t >= 6, "the outlined board must be square"
    frame = grid[t:b+1, l:r+1]
    neutral = int(frame[0, 0])
    assert neutral != bg
    assert np.all(frame[0] == neutral) and np.all(frame[-1] == neutral)
    assert np.all(frame[:, 0] == neutral) and np.all(frame[:, -1] == neutral)
    board = frame[1:-1, 1:-1].copy()
    palette = [int(c) for c in np.unique(board) if c not in (bg, neutral)]
    assert 1 <= len(palette) <= 4
    networks, repaired_union = [], np.zeros(board.shape, dtype=bool)
    for color in palette:
        visible = board == color
        repair = np.zeros(board.shape, dtype=bool)
        # Only single neutral punctures bracketed by visible matching walls.
        repair[:, 1:-1] |= visible[:, :-2] & visible[:, 2:]
        repair[1:-1, :] |= visible[:-2, :] & visible[2:, :]
        repair &= board == neutral
        punctures = list(map(tuple, np.argwhere(repair).tolist()))
        assert all(max(abs(y-a), abs(x-c)) > 1
                   for j, (y, x) in enumerate(punctures)
                   for a, c in punctures[j+1:]), 'punctures must be isolated'
        walls = visible | repair
        box = _bbox(walls)
        assert all(_separated(box, n['box']) for n in networks), "input frames need a blank gap"
        y, x, d, e = box
        assert walls[y, x:e+1].all() and walls[d, x:e+1].all()
        assert walls[y:d+1, x].all() and walls[y:d+1, e].all()
        filled, regions = _interiors(walls)
        networks.append(dict(color=color, walls=walls, filled=filled,
                             regions=regions, count=len(regions), box=box,
                             damage=int(repair.sum())))
        repaired_union |= repair
    assert np.array_equal(repaired_union, board == neutral), "neutral interior pixels must be supported punctures"
    return dict(background=bg, neutral=neutral, board=board, networks=networks,
                position=(t, l), size=len(board), input_shape=grid.shape)


def _turned(mask, turns, coordinate=False):
    if not coordinate:
        return np.rot90(mask, -int(turns) % 4)
    n = len(mask)
    out = np.zeros(mask.shape, dtype=bool)
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        for _ in range(int(turns) % 4):
            row, col = col, n-1-row
        out[row, col] = True
    return out


def _render(info, turns=None, coordinate=False, fill=True, collision=True):
    shape = info['board'].shape
    coverage = np.zeros(shape, dtype=np.int8)
    out = np.full(shape, info['background'], dtype=np.int8)
    for index, net in enumerate(info['networks']):
        k = net['count'] if turns is None else turns[index]
        mask = _turned(net['filled'] if fill else net['walls'], k, coordinate)
        out[mask] = net['color']
        coverage += mask
    if collision:
        out[coverage >= 2] = info['neutral']
    return out, coverage


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair, flood-count, fill, rotate about the board center, and combine."""
    return _render(_decode(input_grid), coordinate=True)[0]


def _capacity(box):
    t, l, b, r = box
    return ((b-t)//2) * ((r-l)//2)


def _partition(rng, h, w, count, style):
    outer = (0, 0, h-1, w-1)
    if _capacity(outer) < count:
        return None
    if style == 'strip' and max(h, w) < 2*count+1:
        # Keep the requested room count when a long strip cannot fit; a
        # recursive partition can use both axes inside the same rectangle.
        style = 'partial'
    if style == 'grid':
        factors = [(a, count//a) for a in range(1, count+1)
                   if count % a == 0 and 2*a < h and 2*(count//a) < w]
        if not factors:
            return None
        if count == 4:
            if h < 5 or w < 5:
                return None
            factors = [(2, 2)]
        nr, nc = factors[int(rng.integers(len(factors)))]
        def cuts(parts, span):
            lengths = np.full(parts, 2, dtype=int)
            for _ in range(span-2*parts):
                lengths[int(rng.integers(parts))] += 1
            return [0] + list(map(int, np.cumsum(lengths)))
        ys, xs = cuts(nr, h-1), cuts(nc, w-1)
        rooms = [(ys[i], xs[j], ys[i+1], xs[j+1]) for i in range(nr) for j in range(nc)]
    else:
        rooms, first_axis = [outer], None
        for step in range(count-1):
            options = []
            for i, (t, l, b, r) in enumerate(rooms):
                for axis, start, end in ((0, t, b), (1, l, r)):
                    if style == 'strip' and first_axis is not None and axis != first_axis:
                        continue
                    if style == 'partial' and count >= 3 and step == 1 and axis == first_axis:
                        continue
                    for cut in range(start+2, end-1):
                        halves = ((t,l,cut,r), (cut,l,b,r)) if axis == 0 else ((t,l,b,cut), (t,cut,b,r))
                        capacity = sum(_capacity(box) for j, box in enumerate(rooms) if j != i)
                        capacity += sum(_capacity(box) for box in halves)
                        if capacity >= count:
                            options.append((i, axis, halves))
            if not options:
                return None
            i, axis, halves = options[int(rng.integers(len(options)))]
            first_axis = axis if first_axis is None else first_axis
            rooms[i:i+1] = list(halves)
    walls = _wall_union((h, w), rooms)
    if style == 'partial' and count >= 3 and not _partial(walls):
        return None
    return walls


def _partial(walls):
    t, l, b, r = _bbox(walls)
    horizontal = np.flatnonzero(np.any(walls[:, :-1] & walls[:, 1:], axis=1))
    vertical = np.flatnonzero(np.any(walls[:-1] & walls[1:], axis=0))
    return any(not walls[y, l:r+1].all() for y in horizontal) or any(not walls[t:b+1, x].all() for x in vertical)


def _lattice_count(walls):
    hs = np.count_nonzero(np.any(walls[:, :-1] & walls[:, 1:], axis=1))
    vs = np.count_nonzero(np.any(walls[:-1] & walls[1:], axis=0))
    return int((hs-1) * (vs-1))


def _two_by_two(net):
    walls = net['walls']
    hs = np.count_nonzero(np.any(walls[:, :-1] & walls[:, 1:], axis=1))
    vs = np.count_nonzero(np.any(walls[:-1] & walls[1:], axis=0))
    return net['count'] == 4 and hs == vs == 3 and not _partial(walls)


def _place_networks(rng, n, counts, styles, colors):
    boxes, networks = [], []
    # Put more complex objects first, but randomize equal-count ordering.
    order = sorted(rng.permutation(len(counts)), key=lambda i: -counts[int(i)])
    for index in order:
        index, accepted = int(index), None
        count = counts[index]
        for _ in range(110):
            h = int(rng.integers(3, min(n-1, 14)+1))
            w = int(rng.integers(3, min(n-1, 14)+1))
            if count == 1 and rng.random() < .25:
                h = int(rng.integers(3, n-1))
                w = int(rng.integers(3, n-1))
            local = _partition(rng, h, w, count, styles[index])
            if local is None:
                continue
            t, l = int(rng.integers(n-h+1)), int(rng.integers(n-w+1))
            box = (t, l, t+h-1, l+w-1)
            if not all(_separated(box, old) for old in boxes):
                continue
            walls = np.zeros((n, n), dtype=bool)
            walls[t:t+h, l:l+w] = local
            filled, regions = _interiors(walls)
            assert len(regions) == count
            accepted = dict(color=colors[index], walls=walls, filled=filled,
                            regions=regions, count=count, box=box, damage=0)
            break
        if accepted is None:
            return None
        networks.append(accepted)
        boxes.append(accepted['box'])
    return networks


def _damage(rng, board, neutral, networks, required):
    total = 0
    for net in networks:
        if not required and rng.random() < .35:
            continue
        walls = net['walls']
        hmid = np.zeros(walls.shape, dtype=bool)
        vmid = np.zeros(walls.shape, dtype=bool)
        hmid[:, 1:-1] = walls[:, :-2] & walls[:, 2:]
        vmid[1:-1] = walls[:-2] & walls[2:]
        # A puncture lies on a straight segment, never a corner or junction.
        safe = walls & (hmid ^ vmid)
        degree = np.zeros(walls.shape, dtype=int)
        degree[1:] += walls[:-1]
        degree[:-1] += walls[1:]
        degree[:, 1:] += walls[:, :-1]
        degree[:, :-1] += walls[:, 1:]
        candidates = np.argwhere(safe & (degree == 2))
        rng.shuffle(candidates)
        wanted = int(rng.integers(1, min(7, max(2, len(candidates)//5))+1))
        chosen = []
        for y, x in candidates:
            if any(max(abs(int(y)-a), abs(int(x)-b)) <= 1 for a, b in chosen):
                continue
            board[y, x] = neutral
            chosen.append((int(y), int(x)))
            if len(chosen) >= wanted:
                break
        total += len(chosen)
    return total


def _assemble(rng, bg, neutral, networks, required_damage):
    n = len(networks[0]['walls'])
    board = np.full((n, n), bg, dtype=np.int8)
    for net in networks:
        board[net['walls']] = net['color']
    if required_damage and _damage(rng, board, neutral, networks, True) == 0:
        return None
    if not required_damage:
        _damage(rng, board, neutral, networks, False)
    s = n+2
    frame = np.full((s, s), neutral, dtype=np.int8)
    frame[1:-1, 1:-1] = board
    h, w = int(rng.integers(s, 31)), int(rng.integers(s, 31))
    y, x = int(rng.integers(h-s+1)), int(rng.integers(w-s+1))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[y:y+s, x:x+s] = frame
    try:
        info = _decode(grid)
    except AssertionError:
        return None
    original = {net['color']: net for net in networks}
    if not all(np.array_equal(net['walls'], original[net['color']]['walls']) for net in info['networks']):
        return None
    return grid, info


def _alternatives(info):
    ns = info['networks']
    counts = [n['count'] for n in ns]
    return {
        'omit_rotation': _render(info, [0]*len(ns))[0],
        'one_turn_each': _render(info, [1]*len(ns))[0],
        'counterclockwise': _render(info, [-k for k in counts])[0],
        'lattice_formula': _render(info, [_lattice_count(n['walls']) for n in ns])[0],
        'rotate_walls': _render(info, fill=False)[0],
        'overwrite_collisions': _render(info, collision=False)[0],
        'rank_counts': _render(info, [sorted(set(counts)).index(k)+1 for k in counts])[0],
        'wall_pixels': _render(info, [int(n['walls'].sum()) for n in ns])[0],
        'frame_area': _render(info, [(n['box'][2]-n['box'][0]+1) * (n['box'][3]-n['box'][1]+1)
                                    for n in ns])[0],
        'dividers_instead_of_rooms': _render(info, [k-1 for k in counts])[0],
    }


def _make_pair(rng, mode, split, witness=-1, previous_bg=None):
    if witness == 0:
        counts, styles = [1, 2, 3], ['strip', 'strip', 'partial']
    elif witness == 1:
        counts = [4, int(rng.choice([1, 3]))]
        styles = ['partial', 'partial']
    elif split == 'test' and mode == 1:
        counts = [4, int(rng.integers(1, 4))]
        styles = ['grid', str(rng.choice(['strip', 'partial']))]
    elif split == 'test' and mode == 2:
        repeated = int(rng.choice([1, 2, 3]))
        counts = [4, repeated, repeated, int(rng.integers(1, 5))]
        styles = [str(rng.choice(['strip', 'partial'])) for _ in counts]
    else:
        number = int(rng.integers(1 if split == 'train' else 2, 4))
        counts = list(map(int, rng.integers(1, 5, size=number)))
        if split == 'test':
            counts[int(rng.integers(number))] = int(rng.integers(5, 9))
        styles = [str(rng.choice(['strip', 'partial', 'partial'])) for _ in counts]
    bg = int(rng.choice([c for c in range(10) if c != previous_bg]))
    rest = list(map(int, rng.choice([c for c in range(10) if c != bg], len(counts)+1, replace=False)))
    neutral, colors = rest[0], rest[1:]
    for attempt in range(400):
        low = 17 if len(counts) >= 3 else 13
        n = int(rng.integers(low, 28))
        nets = _place_networks(rng, n, counts, styles, colors)
        if nets is None:
            continue
        built = _assemble(rng, bg, neutral, nets, witness == 0)
        if built is None:
            continue
        grid, info = built
        out, coverage = _render(info)
        if witness == 0:
            alternatives = _alternatives(info)
            if not np.any(coverage >= 2) or any(np.array_equal(out, alternate)
                    for name, alternate in alternatives.items() if name != 'rank_counts'):
                continue
        if witness == 1:
            # An asymmetric count-four stencil visibly establishes a full cycle.
            four = next(n for n in info['networks'] if n['count'] == 4)
            if any(np.array_equal(four['filled'], _turned(four['filled'], k)) for k in (1, 2, 3)):
                continue
        return {'input': grid, 'output': out}, bg
    raise RuntimeError('bounded geometric sampling exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt, ne = _check_counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(3))
    puzzle, seen, previous_bg = {'train': [], 'test': []}, set(), None
    for split, number in (('train', nt), ('test', ne)):
        for index in range(number):
            witness = index if split == 'train' and index < 2 else -1
            for _ in range(20):
                pair, bg = _make_pair(rng, mode, split, witness, previous_bg)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    previous_bg = bg
                    break
            else:
                raise RuntimeError('could not produce a distinct input')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _metrics(info):
    out, cover = _render(info)
    return dict(size=info['size'], shape=list(info['input_shape']),
                position=list(info['position']), background=info['background'], neutral=info['neutral'],
                networks=len(info['networks']), counts=[n['count'] for n in info['networks']],
                partial=sum(_partial(n['walls']) for n in info['networks']),
                damage=sum(n['damage'] for n in info['networks']),
                collisions=int((cover >= 2).sum()), triple=int((cover >= 3).sum()),
                room_sizes=[list(np.ptp(np.argwhere(region), axis=0)+1)
                            for n in info['networks'] for region in n['regions']])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training_infos, training_outputs, test_infos = set(), [], [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _decode(inp)
            expected, cover = _render(info, coordinate=True)
            assert out.shape == info['board'].shape and np.array_equal(out, expected), 'wrong stencils'
            assert np.array_equal(solve(inp), out)
            assert np.array_equal(out == info['neutral'], cover >= 2)
            assert np.count_nonzero(out != info['background']) == np.count_nonzero(cover)
            assert all(np.sum(_turned(n['filled'], n['count'])) == n['filled'].sum() for n in info['networks'])
            if split == 'train':
                training_infos.append(info)
                training_outputs.append(out)
            else:
                test_infos.append(info)
    assert {1, 2, 3, 4}.issubset({n['count'] for i in training_infos for n in i['networks']})
    assert any(len({n['count'] % 4 for n in i['networks']}) >= 2 for i in training_infos)
    assert any(_partial(n['walls']) for i in training_infos for n in i['networks'])
    assert any(n['damage'] for i in training_infos for n in i['networks'])
    assert any(np.any(_render(i)[1] >= 2) for i in training_infos)
    for name in _alternatives(training_infos[0]):
        assert any(not np.array_equal(out, _alternatives(info)[name])
                   for info, out in zip(training_infos, training_outputs)), f'undistinguished shortcut: {name}'
    # Skipping repair must change the result, rather than just a hidden wall.
    repair_matters = False
    for info, out in zip(training_infos, training_outputs):
        naive = dict(info, networks=[])
        for net in info['networks']:
            walls = info['board'] == net['color']
            regions = _rooms(walls)
            filled = np.logical_or.reduce(regions) if regions else np.zeros(walls.shape, bool)
            naive['networks'].append(dict(net, filled=filled, count=len(regions)))
        repair_matters |= not np.array_equal(out, _render(naive)[0])
    assert repair_matters
    # Recover the actual held-out factor from pixels, not the sampled mode.
    assert all(len(info['networks']) == 4
               or max(n['count'] for n in info['networks']) > 4
               or (any(_two_by_two(n) for n in info['networks'])
                   and not any(_two_by_two(n) for tr in training_infos for n in tr['networks']))
               for info in test_infos), 'test must exercise a demonstrated generalization'


def self_test():
    """Bounded pixel-derived checks and structural diversity over 42 seeds."""
    # The solver supports valid boards beyond the ordinary sampler's sizes.
    # A tiny board needs exterior background to retain a unique modal role.
    for n in (5, 28):
        board = np.zeros((n, n), dtype=np.int8)
        walls = _wall_union(board.shape, [(0, 0, n-1, n-1)])
        board[walls] = 2
        board[0, 1] = 7
        frame = np.full((n+2, n+2), 7, dtype=np.int8)
        frame[1:-1, 1:-1] = board
        if n == 5:
            inp = np.zeros((9, 9), dtype=np.int8)
            inp[1:8, 1:8] = frame
        else:
            inp = frame
        expected = np.zeros((n, n), dtype=np.int8)
        expected[1:-1, 1:-1] = 2
        assert np.array_equal(solve(inp), expected)
    records, fingerprints, topologies = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                record = _metrics(info)
                record['split'] = split
                records.append(record)
                for net in info['networks']:
                    walls = net['walls']
                    t, l, b, r = net['box']
                    local = walls[t:b+1, l:r+1]
                    # Remove spacing: keep ordered line coordinates and segments.
                    ys = np.flatnonzero(np.any(local[:, :-1] & local[:, 1:], axis=1))
                    xs = np.flatnonzero(np.any(local[:-1] & local[1:], axis=0))
                    horizontal = tuple(tuple(bool(local[y, a:z+1].all()) for a, z in zip(xs, xs[1:])) for y in ys)
                    vertical = tuple(tuple(bool(local[a:z+1, x].all()) for a, z in zip(ys, ys[1:])) for x in xs)
                    topologies.add((len(ys), len(xs), horizontal, vertical))
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], again[s]) for k in ('input', 'output'))
            pair = puzzle['train'][0]
            # Equivariance to quarter-turns and arbitrary literal color changes.
            assert np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output']))
            permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
            assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    assert len(fingerprints) == 42
    assert len({tuple(r['shape']) for r in records}) >= 30
    assert len({r['size'] for r in records}) >= 10
    assert {r['background'] for r in records} == set(range(10))
    assert {r['neutral'] for r in records} == set(range(10))
    assert {r['networks'] for r in records} == {1, 2, 3, 4}
    assert set(range(1, 9)).issubset({k for r in records for k in r['counts']})
    assert len(topologies) >= 20
    assert any(r['collisions'] == 0 for r in records) and any(r['collisions'] > 0 for r in records)
    assert any(r['triple'] > 0 for r in records)
    assert any(r['damage'] == 0 for r in records) and any(r['damage'] > 0 for r in records)
    assert any(min(d) == 1 for r in records for d in r['room_sizes'])
    assert any(max(d) >= 10 for r in records for d in r['room_sizes'])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (3,1.5), (None,1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(8, 2, 1)
    p['test'][0]['output'] = p['test'][0]['output'].copy()
    p['test'][0]['output'][0,0] = (int(p['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    return {'seeds': 42, 'pairs': len(records), 'topologies': len(topologies),
            'board_sizes': sorted({r['size'] for r in records}),
            'shapes': len({tuple(r['shape']) for r in records})}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
