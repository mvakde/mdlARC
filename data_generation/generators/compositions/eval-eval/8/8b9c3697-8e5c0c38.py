"""Mirror-locked docking: reflection-gated, stable particle compaction.

Only standard library and NumPy are used. No files, examples, or random state
are consulted by solve. Zero is the fixed trail color, absent from inputs.
"""
from collections import Counter
import hashlib
import numpy as np


_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # outward


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRECTIONS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _parse_fork(cells):
    """Recognize a base and three solid, positive, separated prongs."""
    choices = []
    for outward in _DIRECTIONS:
        perp = (-outward[1], outward[0])
        k0 = min(r*outward[0] + c*outward[1] for r, c in cells)
        q0 = min(r*perp[0] + c*perp[1] for r, c in cells)
        local = {(r*outward[0]+c*outward[1]-k0,
                  r*perp[0]+c*perp[1]-q0) for r, c in cells}
        width = max(q for _, q in local) + 1
        if width < 6 or {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [max((k for k, x in local if x == q), default=0) for q in range(width)]
        if local != {(k, q) for q, n in enumerate(lengths) for k in range(n+1)}:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, n = q, lengths[q]
            while q < width and lengths[q] == n:
                q += 1
            runs.append((start, q, n))
        if len(runs) != 3 or any(runs[i][1] >= runs[i+1][0] for i in (0, 1)):
            continue
        a, b, stop_length = runs[1]
        # Centered middle prong; guards mirror in width/position, not length.
        if a+b != width or runs[0][:2] != (width-runs[2][1], width-runs[2][0]):
            continue
        choices.append(dict(outward=outward, perp=perp, k0=k0, q0=q0,
                            width=width, a=a, b=b, stop=stop_length+1,
                            lengths=tuple(x[2] for x in runs), cells=set(cells)))
    return choices[0] if len(choices) == 1 else None


def _local(fork, p):
    r, c = p
    d, t = fork['outward'], fork['perp']
    return r*d[0]+c*d[1]-fork['k0'], r*t[0]+c*t[1]-fork['q0']


def _world(fork, k, q):
    d, t = fork['outward'], fork['perp']
    k, q = k+fork['k0'], q+fork['q0']
    return k*d[0]+q*t[0], k*d[1]+q*t[1]


def _analyze(grid):
    a = _grid(grid)
    assert not np.any(a == 0), 'trail is absent from input'
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > a.size/2, 'background must be a strict majority'
    forks, particles = [], {}
    for color in np.unique(a):
        color = int(color)
        if color == bg:
            continue
        cells = set(map(tuple, np.argwhere(a == color).tolist()))
        for component in _components(cells):
            f = _parse_fork(component)
            if f is None:
                particles.update((p, color) for p in component)
            else:
                f['color'] = color
                forks.append(f)
    assert forks, 'at least one fork is required'
    structural = set().union(*(f['cells'] for f in forks))
    assert not ({f['color'] for f in forks} & set(particles.values()))
    assigned = [dict() for _ in forks]
    for p, color in particles.items():
        owners = []
        for i, f in enumerate(forks):
            k, q = _local(f, p)
            if k >= f['stop'] and f['a'] <= q < f['b']:
                owners.append((i, k, q))
        assert len(owners) <= 1, 'overlapping particle catchments'
        if owners:
            i, k, q = owners[0]
            assigned[i][k, q] = color
    lanes, accepted, routes = [], set(), set()
    for fi, (f, local) in enumerate(zip(forks, assigned)):
        axis2 = f['a'] + f['b'] - 1
        for q in range(f['a'], (f['a']+f['b']+1)//2):
            mirror = axis2-q
            qs = (q,) if q == mirror else (q, mirror)
            entries = sorted((k, color) for (k, x), color in local.items()
                             if x == q and local.get((k, mirror)) == color)
            if not entries:
                continue
            final, route = {}, set()
            for rank, (start, color) in enumerate(entries):
                target = f['stop']+rank
                assert target <= start
                for x in qs:
                    source = _world(f, start, x)
                    accepted.add(source)
                    final[_world(f, target, x)] = color
                    route.update(_world(f, k, x) for k in range(target, start+1))
            assert not (route & structural), 'route hits another receiver'
            assert not (route & routes), 'overlapping swept routes'
            assert all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] for r, c in route)
            routes |= route
            lanes.append(dict(fi=fi, qs=qs, entries=entries, final=final, route=route))
    return dict(bg=bg, forks=forks, particles=particles, lanes=lanes,
                accepted=accepted, routes=routes)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read forks, reflect by their centerlines, and compact each mirror orbit."""
    a = _grid(input_grid)
    info = _analyze(a)
    out = a.copy()
    for p in info['particles']:
        out[p] = info['bg']
    for lane in info['lanes']:
        for p in lane['route']:
            out[p] = 0
        for p, color in lane['final'].items():
            out[p] = color
    return out


def _module(rng, span, fork_color, token_colors, anchor_count, crowded, lesson=None):
    """Construct one canonical receiver and independently paint its answer."""
    guard = 1 if crowded else int(rng.integers(1, 3))
    gap = 1 if crowded else int(rng.integers(1, 3))
    tail = 0 if crowded else int(rng.integers(2))
    width = span+2*(guard+gap+tail)
    a, b = tail+guard+gap, tail+guard+gap+span
    lengths = tuple(map(int, rng.integers(1, 6, 3)))
    if lesson == 0:
        lengths = (int(rng.integers(3, 6)), int(rng.integers(1, 3)), int(rng.integers(3, 6)))
    elif lesson == 1:
        lengths = (int(rng.integers(1, 4)), int(rng.integers(4, 6)), int(rng.integers(1, 4)))
    cells = {(0, q): fork_color for q in range(width)}
    for lo, hi, n in ((tail, tail+guard, lengths[0]), (a, b, lengths[1]),
                      (width-tail-guard, width-tail, lengths[2])):
        cells.update(((k, q), fork_color) for k in range(1, n+1) for q in range(lo, hi))
    output = dict(cells)
    start = max(lengths)+int(rng.integers(2, 4))
    extent = max(8, anchor_count+4)+int(rng.integers(3))
    orbit_lefts = list(range(a, (a+b+1)//2))
    for index, q in enumerate(orbit_lefts):
        mirror = a+b-1-q
        qs = (q,) if q == mirror else (q, mirror)
        if index == 0:
            n = anchor_count
            rest = [x for x in range(3, extent+1) if x != 4]
            offsets = sorted([1, 4]+list(map(int, rng.choice(rest, n-2, replace=False))))
        else:
            # The second orbit is short: demonstrate independent lane packing.
            n = 1 if index == 1 else int(rng.integers(anchor_count+1))
            offsets = sorted(map(int, rng.choice(extent+1, n, replace=False)))
        colors = list(map(int, rng.choice(token_colors, n)))
        if index == 0:
            colors[0], colors[1] = token_colors[:2]
            if lesson is not None:
                colors[:2] = sorted(colors[:2], reverse=lesson == 0)
            elif rng.integers(2):
                colors[0], colors[1] = colors[1], colors[0]
        for offset, color in zip(offsets, colors):
            for x in qs:
                cells[start+offset, x] = color
        for rank, color in enumerate(colors):
            for x in qs:
                output[lengths[1]+1+rank, x] = color
        if n:
            for k in range(lengths[1]+1+n, start+offsets[-1]+1):
                for x in qs:
                    output[k, x] = 0
    # Four guaranteed unsupported cells: a nearer singleton, a wrong-color
    # reflected pair, and a farther singleton that must not lengthen the trail.
    left, right = a, b-1
    cells[start, left] = token_colors[0]
    cells[start+2, left] = token_colors[0]
    cells[start+2, right] = token_colors[1]
    cells[start+extent+1, right] = token_colors[-1]
    for _ in range(int(rng.integers(5))):
        q = int(rng.integers(a, b))
        mirror = a+b-1-q
        k = start+int(rng.integers(extent+2))
        if q != mirror and (k, q) not in cells and (k, mirror) not in cells:
            cells[k, q] = int(rng.choice(token_colors))
    # An out-of-aperture distraction, even when it belongs to a connected blob.
    cells[start+int(rng.integers(extent+2)), 0] = int(rng.choice(token_colors))
    return cells, output, start+extent+2, width


def _pair(rng, count, parity, rotation, bg, deep, lesson=None):
    pool = list(map(int, rng.permutation([x for x in range(1, 10) if x != bg])))
    nf = int(rng.integers(1, count+1))
    fork_colors, token_colors = pool[:nf], pool[nf:nf+int(rng.integers(2, 5))]
    modules = []
    for i in range(count):
        if count == 1:
            spans = [3, 5, 7] if parity == 0 else [4, 6]
        elif count == 2:
            spans = [2, 3, 4, 5, 6]
        else:
            spans = [2, 3, 4]
        span = int(rng.choice(spans))
        n = int(rng.integers(4, 7) if deep else rng.integers(2, 4))
        modules.append(_module(rng, span, fork_colors[i % nf], token_colors, n, count > 1, lesson))
    gaps = list(map(int, rng.integers(1, 3, max(0, count-1))))
    used_width = sum(m[3] for m in modules)+sum(gaps)
    assert used_width <= 28
    extra = int(rng.integers(min(5, 30-used_width)+1))
    width = max(12, used_width+extra)
    height = max(14, max(m[2] for m in modules)+int(rng.integers(1, 6)))
    height = min(30, height)
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    col = int(rng.integers(width-used_width+1))
    for i, (cells, answer, h, w) in enumerate(modules):
        row = int(rng.integers(height-h+1))
        for (k, q), color in cells.items():
            inp[row+k, col+q] = color
        for (k, q), color in answer.items():
            out[row+k, col+q] = color
        col += w+(gaps[i] if i < count-1 else 0)
    return {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(out, rotation).copy()}


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    rotation = int(rng.integers(4))
    depth_shift = bool(rng.integers(2))
    backgrounds = list(map(int, rng.permutation(range(1, 10))))
    puzzle, seen = {'train': [], 'test': []}, set()
    ordinal = 0
    for split, number in (('train', num_train), ('test', num_test)):
        for index in range(number):
            bg = backgrounds[ordinal % 9]
            if split == 'train' and index < 2:
                count, parity, turn = 1, index, (rotation+index) % 4
            else:
                count = (int(rng.integers(1, 4)) if depth_shift else
                         3 if split == 'test' else int(rng.integers(1, 3)))
                parity, turn = int(rng.integers(2)), int(rng.integers(4))
            for _ in range(32):
                lesson = index if split == 'train' and index < 2 else None
                pair = _pair(rng, count, parity, turn, bg, depth_shift and split == 'test', lesson)
                key = _key(pair['input'])
                # Parser agreement rejects accidental fork-shaped particle blobs.
                try:
                    prediction = solve(pair['input'])
                except AssertionError:
                    continue
                if key not in seen and np.array_equal(prediction, pair['output']):
                    break
            else:
                raise RuntimeError('bounded construction failed')
            seen.add(key)
            puzzle[split].append(pair)
            ordinal += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit_pair(pair):
    """Check geometric conservation and stable sequences, beyond array equality."""
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    info = _analyze(inp)
    assert np.array_equal(solve(inp), out)
    for f in info['forks']:
        assert min(f['lengths']) >= 1
        assert all(out[p] == f['color'] for p in f['cells'])
    final = {}
    for lane in info['lanes']:
        f = info['forks'][lane['fi']]
        for q in lane['qs']:
            sequence = [int(out[_world(f, f['stop']+i, q)]) for i in range(len(lane['entries']))]
            assert sequence == [color for _, color in lane['entries']]
        assert not (set(final) & set(lane['final']))
        final.update(lane['final'])
        assert all(out[p] == 0 for p in lane['route']-set(lane['final']))
    assert Counter(final.values()) == Counter(inp[p] for p in info['accepted'])
    actual_particles = {tuple(p): int(out[tuple(p)]) for p in np.argwhere((out != info['bg']) & (out != 0))
                        if all(tuple(p) not in f['cells'] for f in info['forks'])}
    assert actual_particles == final
    assert info['lanes'] and info['particles'].keys()-info['accepted']
    assert np.any(out == 0)
    return info


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _audit_pair(pair)
            key = _key(np.asarray(pair['input']))
            assert key not in seen
            seen.add(key)
            if split == 'train':
                training.append(info)
            else:
                testing.append(info)
    assert len({i['bg'] for i in training}) >= 2
    assert {(f['b']-f['a']) % 2 for i in training[:2] for f in i['forks']} == {0, 1}
    assert len({f['outward'] for i in training[:2] for f in i['forks']}) == 2
    for info in training[:2]:
        # Both parent mechanisms are answer-changing in each teaching example.
        assert len(info['particles']) > len(info['accepted'])
        assert any(_world(info['forks'][l['fi']], k, l['qs'][0]) not in l['final']
                   for l in info['lanes'] for k, _ in l['entries'])
        assert any(len({color for _, color in l['entries']}) > 1 for l in info['lanes'])
        assert any(len(l['entries']) > 1 and any(b[0]-a[0] > 1 for a, b in zip(l['entries'], l['entries'][1:]))
                   for l in info['lanes'])
        assert len({len(l['entries']) for l in info['lanes']}) >= 2
    first, second = (x['forks'][0]['lengths'] for x in training[:2])
    assert first[1] < min(first[0], first[2]) and second[1] > max(second[0], second[2])
    assert training[0]['lanes'][0]['entries'][0][1] > training[0]['lanes'][0]['entries'][1][1]
    assert training[1]['lanes'][0]['entries'][0][1] < training[1]['lanes'][0]['entries'][1][1]
    max_stack = max(len(l['entries']) for i in training for l in i['lanes'])
    max_forks = max(len(i['forks']) for i in training)
    assert (all(max(len(l['entries']) for l in i['lanes']) > max_stack for i in testing)
            or all(len(i['forks']) > max_forks for i in testing)), 'missing train/test shift'


def self_test() -> dict:
    """42 seeds; bounded aggregate evidence, no saved grids or execution traces."""
    shapes, geometries, directions, backgrounds, counts, stacks, palettes = (set() for _ in range(7))
    hashes = set()
    pair_count = unmatched = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _audit_pair(pair)
                inp = np.asarray(pair['input'])
                digest.update(bytes(inp.shape)+inp.tobytes())
                shapes.add(inp.shape)
                backgrounds.add(info['bg'])
                palettes.add(tuple(map(int, np.unique(inp))))
                counts.add(len(info['forks']))
                unmatched += len(info['particles'])-len(info['accepted'])
                pair_count += 1
                for f in info['forks']:
                    directions.add(f['outward'])
                    geometries.add((f['width'], f['b']-f['a'], f['lengths']))
                stacks.update(len(l['entries']) for l in info['lanes'])
        hashes.add(digest.digest())
        if seed < 2:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], repeat[s])
                       for k in ('input', 'output'))
    assert len(hashes) == 42 and len(shapes) >= 40 and len(geometries) >= 100
    assert len(directions) == 4 and backgrounds == set(range(1, 10))
    assert counts == {1, 2, 3} and set(range(1, 7)) <= stacks and len(palettes) >= 30
    # Rotation covariance and input immutability, without any training context.
    p = generate_puzzle(83, 2, 1)['test'][0]
    original = p['input'].copy()
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(original, turn)), np.rot90(p['output'], turn))
    assert np.array_equal(original, p['input'])
    # A small hand-constructed extension checks unsampled but valid cases:
    # stationary particles, a self-supported axis, an idle fork, and mixed
    # directions. Expected arrays are painted directly, without solve helpers.
    mixed_in = np.ones((30, 30), dtype=np.int8)
    mixed_out = mixed_in.copy()
    for span, row, col, turn, idle in ((3, 1, 1, 0, False), (4, 18, 18, 1, False),
                                       (4, 1, 12, 0, True)):
        width = span+4
        source = np.ones((10, width), dtype=np.int8)
        answer = source.copy()
        source[0, :] = 9
        source[1, 0] = 9
        source[1:3, 2:2+span] = 9
        source[1:5, -1] = 9
        answer[:] = source
        if idle:
            source[6, 2] = 2
        else:
            for q in (2, span+1):
                source[3, q], source[7, q] = 2, 3
                answer[3, q], answer[4, q] = 2, 3
                answer[5:8, q] = 0
            if span % 2:
                q = 2+span//2
                source[6, q] = 3
                answer[3, q] = 3
                answer[4:7, q] = 0
            else:
                source[8, 3] = 2
        source, answer = np.rot90(source, turn), np.rot90(answer, turn)
        h, w = source.shape
        mixed_in[row:row+h, col:col+w] = source
        mixed_out[row:row+h, col:col+w] = answer
    assert np.array_equal(solve(mixed_in), mixed_out)
    # A structurally complete machine with no accepted units writes no trail.
    assert np.array_equal(solve(source), answer)
    bad = generate_puzzle(84, 2, 1)
    info = _analyze(bad['test'][0]['input'])
    pos = next(iter(info['lanes'][0]['final']))
    bad['test'][0]['output'][pos] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a destroyed particle')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=pair_count, grid_shapes=len(shapes), geometries=len(geometries),
                directions=len(directions), backgrounds=sorted(backgrounds), fork_counts=sorted(counts),
                stack_lengths=sorted(stacks), palettes=len(palettes), unsupported_particles=unmatched)
