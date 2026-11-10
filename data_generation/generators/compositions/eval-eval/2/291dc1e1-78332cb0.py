"""Centered comb assembly: a border-directed rotation makes port matching meaningful.

All conventions are fixed by geometry. Outputs are constructed from the sampled
combs; solve independently extracts the visible combs and searches their ports.
No parent generator, examples, filesystem access, or global random state is used.
"""

from collections import Counter
from itertools import combinations
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    candidates = []
    if min(a.shape) < 5:
        raise ValueError("No readable L-shaped key")
    for bottom in (False, True):
        for right in (False, True):
            b = a[::-1] if bottom else a
            b = b[:, ::-1] if right else b
            origin, horizontal, vertical, bg = map(
                int, (b[0, 0], b[0, 1], b[1, 0], b[-1, -1]))
            if len({origin, horizontal, vertical, bg}) != 4:
                continue
            if (np.all(b[0, 1:] == horizontal)
                    and np.all(b[1:, 0] == vertical)
                    and np.all(b[-1, 1:] == bg)
                    and np.all(b[1:, -1] == bg)):
                candidates.append((bottom, right, origin, horizontal, vertical, bg))
    if len(candidates) != 1:
        raise ValueError("The input must contain one unambiguous L-shaped key")
    return candidates[0]


def _components(mask):
    left = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while left:
        start = min(left)
        left.remove(start)
        points, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in left:
                    left.remove(q)
                    points.add(q)
                    stack.append(q)
        result.append(points)
    return result


def _extract(value):
    """Recover reserved three-row combs, including empty terminal tooth rows."""
    a = _grid(value)
    bottom, right, origin, horizontal, vertical, bg = _key(a)
    mask = a != bg
    mask[-1 if bottom else 0, :] = False
    mask[:, -1 if right else 0] = False
    components = _components(mask)
    if not 2 <= len(components) <= 10:
        raise ValueError("There must be 2..10 separated combs")
    records, axes = [], set()
    for points in components:
        r0, r1 = min(r for r, c in points), max(r for r, c in points)+1
        c0, c1 = min(c for r, c in points), max(c for r, c in points)+1
        h, w = r1-r0, c1-c0
        if w >= 4 and 2 <= h <= 3:
            axis = 0
            spines = [r for r in range(r0, r1) if np.all(mask[r, c0:c1])]
            if len(spines) != 1:
                raise ValueError("A horizontal comb needs one full spine")
            s = spines[0]
            box = (s-1, s+2, c0, c1)
        elif h >= 4 and 2 <= w <= 3:
            axis = 1
            spines = [c for c in range(c0, c1) if np.all(mask[r0:r1, c])]
            if len(spines) != 1:
                raise ValueError("A vertical comb needs one full spine")
            s = spines[0]
            box = (r0, r1, s-1, s+2)
        else:
            raise ValueError("A comb must have a long spine and one-cell teeth")
        r0, r1, c0, c1 = box
        if not (0 < r0 < r1 < a.shape[0] and 0 < c0 < c1 < a.shape[1]):
            raise ValueError("Comb tooth rows must be fully visible inside the key")
        cropped_points = {(r, c) for r in range(r0, r1) for c in range(c0, c1)
                          if mask[r, c]}
        if cropped_points != points:
            raise ValueError("Comb reservations must not contain another object")
        axes.add(axis)
        records.append((box, s))
    if len(axes) != 1:
        raise ValueError("All spines must lie on one common axis")
    axis = axes.pop()
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    blocks = []
    for (r0, r1, c0, c1), s in records:
        block = np.rot90(a[r0:r1, c0:c1], rotation).copy()
        if (block.shape[0] != 3 or not np.all(block[1] != bg)
                or np.any(block[[0, 2]][:, [0, -1]] != bg)
                or not np.any(block[[0, 2]] != bg)):
            raise ValueError("Only interior spine positions may carry teeth")
        blocks.append(block)
    # Input lanes are separated reserved bands; free offsets within a lane.
    lane_positions = sorted({s for box, s in records})
    if any(y-x < 4 for x, y in zip(lane_positions, lane_positions[1:])):
        raise ValueError("Lane reservations need a clear background separator")
    lengths = [b.shape[1] for b in blocks]
    if len({w % 2 for w in lengths}) != 1:
        raise ValueError("All spine lengths must have compatible parity")
    slow_reverse = right if axis else bottom
    fast_reverse = bottom if axis else right
    # This order is only an ablation baseline, never the assembly rule.
    raster = tuple(sorted(range(len(blocks)), key=lambda i: (
        (-1 if slow_reverse else 1)*records[i][1],
        (-1 if fast_reverse else 1)*records[i][0][0 if axis else 2])))
    return dict(blocks=blocks, background=bg, axis=axis, rotation=rotation,
                key=(bottom, right, origin, horizontal, vertical, bg),
                boxes=tuple(box for box, s in records), raster=raster,
                lanes=len(lane_positions), lengths=tuple(lengths))


def _signatures(blocks, bg, centered=True):
    width = max(b.shape[1] for b in blocks)
    return [(tuple(int(x)+((width-b.shape[1])//2 if centered else 0)
                   for x in np.flatnonzero(b[0] != bg)),
             tuple(int(x)+((width-b.shape[1])//2 if centered else 0)
                   for x in np.flatnonzero(b[2] != bg))) for b in blocks]


def _orders(signatures):
    """Return up to two directed all-piece orders; local matches may branch."""
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in signatures):
        return []
    answers = []

    def visit(path, unused):
        if len(answers) >= 2:
            return
        if not unused:
            if path[-1] == ends[0]:
                answers.append(tuple(path))
            return
        port = signatures[path[-1]][1]
        if not port:
            return
        for i in unused:
            if signatures[i][0] == port:
                visit(path+[i], tuple(j for j in unused if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _decode(value):
    d = _extract(value)
    signatures = _signatures(d['blocks'], d['background'])
    orders = _orders(signatures)
    if len(orders) != 1:
        raise ValueError("Centered ports must specify exactly one complete chain")
    d.update(signatures=signatures, order=orders[0])
    return d


def _stack(blocks, bg):
    width = max(b.shape[1] for b in blocks)
    result = np.full((3*len(blocks), width), bg, dtype=np.int8)
    for i, b in enumerate(blocks):
        start = (width-b.shape[1])//2
        result[3*i:3*i+3, start:start+b.shape[1]] = b
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use input geometry only; the key and spines expose every convention."""
    d = _decode(input_grid)
    return _stack([d['blocks'][i] for i in d['order']], d['background'])


def _kind(signature):
    if len(signature) == 1:
        return 'single'
    return 'thick' if signature[-1]-signature[0]+1 == len(signature) else 'separated'


def _pool(positions, capability):
    result = [(p,) for p in positions]
    if capability:
        for size in (2, 3):
            for sig in combinations(positions, size):
                if capability == 2 or _kind(sig) == 'thick':
                    result.append(sig)
    return result


def _choose(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _layout(rng, n, capability, witness):
    # Enough common signatures for a constructive unique path with self-loops.
    needed = (n-1+1)//2
    minimum = 4
    while len(_pool(tuple(range(1, minimum-1)), capability)) < needed:
        minimum += 1
    if witness and capability:
        minimum = max(minimum, 5 if capability == 2 else 4)
    parity = int(rng.integers(2))
    base = minimum + ((parity-minimum) % 2)
    capacity = 28//(base+1)
    low = max(2 if witness else 1, (n+capacity-1)//capacity)
    high = min(n, 7)
    lane_count = int(rng.integers(low, high+1))
    counts = [1]*lane_count
    balanced = rng.random() < .45
    for _ in range(n-lane_count):
        eligible = [i for i, c in enumerate(counts) if c < capacity]
        if balanced:
            least = min(counts[i] for i in eligible)
            eligible = [i for i in eligible if counts[i] == least]
        counts[_choose(rng, eligible)] += 1
    maximum = _choose(rng, list(range(base, 28, 2)))
    equal = not witness and rng.random() < .18
    common_upper = min((28-c)//c for c in counts)
    common = _choose(rng, list(range(base, min(maximum, common_upper)+1, 2)))
    groups = []
    for count in counts:
        group = ([common]*count if equal else
                 rng.choice(np.arange(base, maximum+1, 2), size=count).astype(int).tolist())
        while sum(group)+count > 28:
            eligible = [i for i, w in enumerate(group) if w > base]
            group[_choose(rng, eligible)] -= 2
        rng.shuffle(group)
        groups.append(group)
    if witness and len({w for group in groups for w in group}) == 1:
        # A +2 always fits some lane in these small witness layouts.
        eligible = [i for i, g in enumerate(groups) if sum(g)+len(g)+2 <= 28]
        lane = _choose(rng, eligible)
        groups[lane][int(rng.integers(len(groups[lane])))] += 2
    minimum_width = max(sum(g)+len(g)+1 for g in groups)
    width = int(rng.integers(minimum_width, 30))
    minimum_height = 4*lane_count+1
    height = int(rng.integers(minimum_height, 30))
    slow_gaps = 1+rng.multinomial(height-minimum_height, rng.dirichlet(np.ones(lane_count+1)))
    placements, lengths = [], []
    v = int(slow_gaps[0])
    for lane, group in enumerate(groups):
        spare = width-sum(group)-len(group)-1
        gaps = 1+rng.multinomial(spare, rng.dirichlet(np.ones(len(group)+1)))
        u = int(gaps[0])
        for j, length in enumerate(group):
            placements.append((v, u))
            lengths.append(length)
            u += length+int(gaps[j+1])
        v += 3+int(slow_gaps[lane+1])
    return lengths, placements, (height, width)


def _links(rng, lengths, capability, required):
    n, width = len(lengths), max(lengths)
    domains = []
    for a, b in zip(lengths, lengths[1:]):
        shorter = min(a, b)
        start = (width-shorter)//2
        domains.append(_pool(tuple(range(start+1, start+shorter-1)), capability))

    def acceptable(links):
        return (required <= {_kind(s) for s in links}
                and len(_orders(list(zip([()]+links, links+[()])))) == 1)

    for _ in range(60):
        links = [_choose(rng, pool) for pool in domains]
        if acceptable(links):
            return links
    # Guaranteed fallback: distinct states with at most one forced self-loop
    # per state. All these states fit the shortest comb, hence all neighbors.
    shorter = min(lengths)
    start = (width-shorter)//2
    pool = _pool(tuple(range(start+1, start+shorter-1)), capability)
    selected = []
    for kind in sorted(required):
        selected.append(_choose(rng, [s for s in pool if _kind(s) == kind]))
    count = max(len(selected), (n-1+1)//2)
    rest = [s for s in pool if s not in selected]
    rng.shuffle(rest)
    selected += rest[:count-len(selected)]
    rng.shuffle(selected)
    doubled = set(map(int, rng.choice(count, size=n-1-count, replace=False)))
    links = [s for i, s in enumerate(selected) for _ in range(2 if i in doubled else 1)]
    assert acceptable(links)
    return links


def _make_combs(rng, lengths, links, bg, witness_simple=False):
    colors = [c for c in range(10) if c != bg]
    size = 1 if witness_simple else int(rng.integers(1, 6))
    palette = rng.choice(colors, size=size, replace=False)
    width = max(lengths)
    blocks = []
    for i, length in enumerate(lengths):
        b = np.full((3, length), bg, dtype=np.int8)
        mode = int(rng.integers(4))
        same = [old for old in blocks if old.shape[1] == length]
        if same and rng.random() < .2:
            b[1] = _choose(rng, same)[1]
        elif mode == 0:
            b[1] = rng.choice(palette)
        elif mode == 1:
            period = int(rng.integers(1, min(4, length)+1))
            b[1] = np.resize(rng.choice(palette, size=period), length)
        else:
            b[1] = rng.choice(palette, size=length)
        offset = (width-length)//2
        for row, sig in ((0, () if i == 0 else links[i-1]),
                         (2, () if i == len(lengths)-1 else links[i])):
            for position in sig:
                b[row, position-offset] = rng.choice(palette)
        blocks.append(b)
    return blocks


def _paint(blocks, placements, shape, config, roles):
    rotation, origin_bottom = config
    bg, origin, fast, slow = roles
    interior = np.full(shape, bg, dtype=np.int8)
    for b, (v, u) in zip(blocks, placements):
        interior[v:v+3, u:u+b.shape[1]] = b
    grid = np.full((shape[0]+1, shape[1]+1), bg, dtype=np.int8)
    if origin_bottom:
        grid[:-1, 1:] = interior
        grid[-1, 0], grid[-1, 1:], grid[:-1, 0] = origin, fast, slow
    else:
        grid[1:, 1:] = interior
        grid[0, 0], grid[0, 1:], grid[1:, 0] = origin, fast, slow
    return np.rot90(grid, rotation).copy()


def _ablation_evidence(d):
    expected = _stack([d['blocks'][i] for i in d['order']], d['background'])
    raster = _stack([d['blocks'][i] for i in d['raster']], d['background'])
    local = _orders(_signatures(d['blocks'], d['background'], centered=False))
    local_different = len(local) != 1 or not np.array_equal(
        _stack([d['blocks'][i] for i in local[0]], d['background']), expected)
    count_signatures = [(tuple([len(a)]) if a else (), tuple([len(b)]) if b else ())
                        for a, b in d['signatures']]
    return (not np.array_equal(expected, raster), local_different,
            len(_orders(count_signatures)) != 1)


def _pair(rng, n, capability, config, witness=False, simple=False):
    required = ({'single', 'thick', 'separated'} if witness and capability == 2 else
                {'single', 'thick'} if witness and capability == 1 else {'single'})
    for _ in range(100):
        lengths, placements, shape = _layout(rng, n, capability, witness)
        chain_slots = rng.permutation(n).astype(int).tolist()
        chain_lengths = [lengths[i] for i in chain_slots]
        links = _links(rng, chain_lengths, capability, required)
        bg, origin, fast, slow = map(int, rng.choice(10, size=4, replace=False))
        chain = _make_combs(rng, chain_lengths, links, bg, simple)
        in_slots = [None]*n
        for b, slot in zip(chain, chain_slots):
            in_slots[slot] = b
        inp = _paint(in_slots, placements, shape, config, (bg, origin, fast, slow))
        pair = {'input': inp, 'output': _stack(chain, bg)}
        d = _decode(inp)
        assert np.array_equal(solve(inp), pair['output'])
        evidence = _ablation_evidence(d)
        asymmetric = (not np.array_equal(pair['output'], pair['output'][::-1])
                      and not np.array_equal(pair['output'], pair['output'][:, ::-1]))
        if not witness or (evidence[0] and evidence[1] and asymmetric
                           and (not simple or evidence[2])):
            return pair
    raise RuntimeError("Could not construct a distinguishing demonstration")


def _array_key(a):
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(np.asarray(p['input'])), _array_key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    capability = int(rng.integers(3))
    k, side = int(rng.integers(4)), bool(rng.integers(2))
    configs = [(k, side), (k ^ 1, not side)]
    held = [(j, b) for j in range(4) for b in (False, True) if (j, b) not in configs]
    puzzle, used = {'train': [], 'test': []}, set()
    max_train = 0
    for i in range(num_train):
        n = int(rng.integers(4, 7) if i < 2 else rng.integers(2, 8))
        cap = 0 if i == 0 else capability
        cfg = configs[i] if i < 2 else _choose(rng, configs)
        for _ in range(50):
            p = _pair(rng, n, cap, cfg, witness=i < 2, simple=i == 0)
            if _array_key(p['input']) not in used:
                break
        else:
            raise RuntimeError('Could not create a distinct input')
        puzzle['train'].append(p)
        used.add(_array_key(p['input']))
        max_train = max(max_train, n)
    for _ in range(num_test):
        n = int(rng.integers(max(6, max_train+1), 11))
        cfg = _choose(rng, held)
        for _ in range(50):
            p = _pair(rng, n, capability, cfg)
            if _array_key(p['input']) not in used:
                break
        else:
            raise RuntimeError('Could not create a distinct input')
        puzzle['test'].append(p)
        used.add(_array_key(p['input']))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate actual grids, conservation, connections and curriculum evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _array_key(inp)
            assert key not in seen, 'Repeated input'
            seen.add(key)
            d = _decode(inp)
            bg, blocks, order = d['background'], d['blocks'], d['order']
            assert sorted(order) == list(range(len(blocks)))
            assert out.shape == (3*len(blocks), max(d['lengths']))
            assert np.array_equal(out, _stack([blocks[i] for i in order], bg)), 'Incorrect intact assembly'
            assert Counter(int(c) for b in blocks for c in b.flat if c != bg) == Counter(
                int(c) for c in out.flat if c != bg), 'Payload colors must be conserved'
            assert np.all(out[0] == bg) and np.all(out[-1] == bg)
            for i in range(len(blocks)-1):
                a, b = out[3*i+2] != bg, out[3*i+3] != bg
                assert a.any() and np.array_equal(a, b), 'Complete tooth signatures must align'
            assert len(_components(out != bg)) == 1, 'Assembled combs must form one connected object'
            d['kinds'] = {_kind(d['signatures'][i][1]) for i in order[:-1]}
            records[split].append(d)
    train, test = records['train'], records['test']
    evidence = [_ablation_evidence(d) for d in train]
    assert any(e[0] for e in evidence), 'Teach assembly rather than raster stacking'
    assert any(e[1] for e in evidence), 'Teach center-relative rather than local ports'
    assert any(e[2] for e in evidence), 'Teach positions rather than port cardinality'
    assert {d['axis'] for d in train} == {0, 1}, 'Demonstrate both orientations'
    assert any(d['rotation'] in (1, 2, 3) for d in train), 'Demonstrate proper rotation'
    for axis in (0, 1):
        assert any(d['axis'] == axis and not np.array_equal(
            _stack([d['blocks'][i] for i in d['order']], d['background']),
            _stack([d['blocks'][i] for i in d['order']], d['background'])[::-1])
            and not np.array_equal(
                _stack([d['blocks'][i] for i in d['order']], d['background']),
                _stack([d['blocks'][i] for i in d['order']], d['background'])[:, ::-1])
            for d in train), 'Demonstrate directional teeth on both axes'
    assert len({w for d in train for w in d['lengths']}) >= 2
    demonstrated = set.union(*(d['kinds'] for d in train))
    assert set.union(*(d['kinds'] for d in test)) <= demonstrated
    assert all(len(d['blocks']) > max(len(t['blocks']) for t in train) for d in test)
    assert all((d['axis'], d['key'][0], d['key'][1]) not in
               {(t['axis'], t['key'][0], t['key'][1]) for t in train} for d in test)


def self_test():
    """Bounded 42-seed checks, explicit counts, invariance and negative witnesses."""
    puzzles, features, masks = set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                features.append((split, d, pair['input'].shape, pair['output'].shape))
                masks.update((b.shape, (b != d['background']).tobytes()) for b in d['blocks'])
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        # Rotating the entire input and arbitrarily permuting all color roles
        # preserve the normalized answer under the corresponding color map.
        pair = puzzle['test'][0]
        mapping = np.random.default_rng(seed+1000).permutation(10).astype(np.int8)
        for k in range(4):
            assert np.array_equal(solve(np.rot90(pair['input'], k)), pair['output'])
            assert np.array_equal(solve(mapping[np.rot90(pair['input'], k)]), mapping[pair['output']])
    assert len(puzzles) == 42 and len(masks) >= 100
    assert len({x[2] for x in features}) >= 35
    assert len({x[3] for x in features}) >= 25
    assert {d['axis'] for s, d, a, b in features} == {0, 1}
    assert {(d['axis'], d['key'][0], d['key'][1]) for s, d, a, b in features} == {
        (axis, bottom, right) for axis in (0, 1) for bottom in (False, True) for right in (False, True)}
    assert {d['background'] for s, d, a, b in features} == set(range(10))
    assert {len(d['blocks']) for s, d, a, b in features} >= set(range(2, 11))
    assert {max(d['lengths']) % 2 for s, d, a, b in features} == {0, 1}
    assert max(max(d['lengths']) for s, d, a, b in features) >= 20
    assert {k for s, d, a, b in features for k in
            {_kind(d['signatures'][i][1]) for i in d['order'][:-1]}} == {'single', 'thick', 'separated'}
    assert any(len({sig[1] for sig in d['signatures'] if sig[1]}) < len(d['blocks'])-1
               for s, d, a, b in features), 'Repeated joins should occur'
    for train_count in range(2, 11):
        for test_count in range(1, min(4, train_count-1)+1):
            p = generate_puzzle(100+train_count*4+test_count, train_count, test_count)
            assert len(p['train']) == train_count and len(p['test']) == test_count
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.2), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts were accepted')
    # Unique forced self-loop, competing self-loops, and two excursions.
    a, b, c = (1,), (2,), (3,)
    assert len(_orders([((), a), (a, a), (a, ())])) == 1
    assert len(_orders([((), a), (a, a), (a, a), (a, ())])) == 2
    assert len(_orders([((), a), (a, b), (b, a), (a, c), (c, a), (a, ())])) == 2
    # A manually specified centered join has different raw local coordinates.
    short = np.array([[0, 0, 0, 0], [4, 4, 4, 4], [0, 7, 0, 0]], dtype=np.int8)
    long = np.array([[0, 0, 0, 6, 0, 0, 0, 0], [2]*8, [0]*8], dtype=np.int8)
    assert _orders(_signatures([short, long], 0)) == [(0, 1)]
    assert _orders(_signatures([short, long], 0, False)) == []
    def expect_bad_input(inp):
        try:
            solve(inp)
        except ValueError:
            return
        raise AssertionError('An invalid comb input was accepted')

    # Direct invalid inputs, rather than assertions about generation metadata.
    roles = (0, 9, 1, 2)
    joined = _paint([short, long], [(1, 1), (5, 1)], (9, 10), (0, False), roles)
    assert np.array_equal(solve(joined), _stack([short, long], 0))
    hole = joined.copy()
    hole[3, 3] = 0
    expect_bad_input(hole)
    tip = joined.copy()
    tip[4, 2] = 7
    expect_bad_input(tip)
    odd = np.zeros((3, 5), dtype=np.int8)
    odd[1] = 3
    odd[0, 1] = 3
    expect_bad_input(_paint([short, odd], [(1, 1), (5, 1)], (9, 10), (0, False), roles))
    ambiguous = []
    for i, (incoming, outgoing) in enumerate((((), a), (a, a), (a, a), (a, ()))):
        comb = np.zeros((3, 6), dtype=np.int8)
        comb[1] = i+3
        for row, sig in ((0, incoming), (2, outgoing)):
            for col in sig:
                comb[row, col] = i+3
        ambiguous.append(comb)
    expect_bad_input(_paint(ambiguous, [(1+4*i, 1) for i in range(4)],
                            (17, 8), (0, False), roles))
    # A wrong label must fail; the solver never chooses a parse using that label.
    p = generate_puzzle(701, 2, 1)
    p['test'][0]['output'][1, 0] = (int(p['test'][0]['output'][1, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('Damaged output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, legal counts, geometry and color invariance')
