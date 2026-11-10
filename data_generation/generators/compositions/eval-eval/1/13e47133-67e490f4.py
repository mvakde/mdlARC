"""Peeling ballots: erosion changes the silhouette that selects each band.

Standard library + NumPy; no files, parent imports, persistent state, or I/O.
"""
from collections import Counter, defaultdict, deque
from functools import lru_cache
import hashlib

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
L4 = ((0, 0), (1, 0), (2, 0), (2, 1))
S4 = ((0, 1), (0, 2), (1, 0), (1, 1))
T4 = ((0, 0), (0, 1), (0, 2), (1, 1))


def _norm(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _reflect(cells):
    return _norm((r, -c) for r, c in cells)


@lru_cache(maxsize=512)
def _rotations(cells):
    current, options = _norm(cells), set()
    for _ in range(4):
        options.add(current)
        current = _norm((c, -r) for r, c in current)
    return tuple(sorted(options))


@lru_cache(maxsize=512)
def _variants(cells):
    return tuple(sorted(set(_rotations(cells)) | set(_rotations(_reflect(cells)))))


def _canon(cells):
    return min(_variants(_norm(cells)))


def _size(cells):
    raw = _norm(cells)
    return max(r for r, _ in raw)+1, max(c for _, c in raw)+1


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _split(cells, neighbors=DIR4):
    pending, result = set(cells), []
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in neighbors:
                point = r+dr, c+dc
                if point in pending:
                    pending.remove(point)
                    stack.append(point)
        result.append(tuple(sorted(component)))
    return result


def _erode(cells, neighbors=DIR8):
    occupied = set(cells)
    return {p for p in occupied
            if all((p[0]+dr, p[1]+dc) in occupied for dr, dc in neighbors)}


def _dilate(cells, radius):
    return _norm({(r+dr, c+dc) for r, c in cells
                  for dr in range(-radius, radius+1)
                  for dc in range(-radius, radius+1)})


def _peels(cells, neighbors=DIR8, rectangle=False):
    """Yield ephemeral (depth, current component, boundary) records."""
    active, depth = set(cells), 0
    box_depths = {}
    if rectangle:
        for component in _split(active):
            rs, cs = zip(*component)
            for r, c in component:
                box_depths[r, c] = min(r-min(rs), max(rs)-r, c-min(cs), max(cs)-c)
    while active:
        survivors = set()
        for component in _split(active):
            inside = ({p for p in component if box_depths[p] > depth}
                      if rectangle else _erode(component, neighbors))
            yield depth, component, set(component)-inside
            survivors.update(inside)
        active, depth = survivors, depth+1


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'bad grid size'
    assert np.issubdtype(a.dtype, np.integer), 'noninteger cells'
    assert np.all((0 <= a) & (a <= 9)), 'bad color'
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        h.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _parse(input_grid):
    """Recover roles, carrier, all ballots, and all recursive queries."""
    grid = _grid(input_grid)
    candidates = []
    for color in map(int, np.unique(grid)):
        color_cells = set(map(tuple, np.argwhere(grid == color)))
        for cells in _split(color_cells):
            if len(cells) < 8 or len(cells) != len(color_cells):
                continue
            rs, cs = zip(*cells)
            y, x, bottom, right = min(rs), min(cs), max(rs), max(cs)
            box = grid[y:bottom+1, x:right+1]
            if min(box.shape) < 3:
                continue
            if not (np.all(box[0] == color) and np.all(box[-1] == color)
                    and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)):
                continue
            other = set(map(int, np.unique(box)))-{color}
            if len(other) == 1:
                candidates.append((color, other.pop(), y, x, bottom, right))
    assert len(candidates) == 1, 'carrier rectangle is not unique'
    carrier, bg, y, x, bottom, right = candidates[0]
    box = grid[y:bottom+1, x:right+1]
    active = set(map(tuple, np.argwhere(box == bg)))
    assert active, 'no cavities'
    assert len(_split(map(tuple, np.argwhere(box == carrier)))) == 1
    exterior = grid.copy()
    exterior[y:bottom+1, x:right+1] = bg
    foreground = set(map(tuple, np.argwhere(exterior != bg)))
    objects = _split(foreground)
    assert objects and len(_split(foreground, DIR8)) == len(objects), 'touching voters'
    expanded = {(r, c) for r in range(max(0, y-1), min(grid.shape[0], bottom+2))
                for c in range(max(0, x-1), min(grid.shape[1], right+2))}
    assert not foreground & expanded, 'voter touches carrier'
    ballots, raws = defaultdict(Counter), []
    for cells in objects:
        colors = {int(grid[p]) for p in cells}
        assert len(colors) == 1, 'polychromatic vote'
        color = colors.pop()
        assert color not in (bg, carrier)
        raw = _norm(cells)
        ballots[_canon(raw)][color] += 1
        raws.append((color, raw))
    queries = list(_peels(active))
    needed = {_canon(component) for _, component, _ in queries}
    assert set(ballots) == needed, 'unsupported or irrelevant shape class'
    winners = {}
    for shape, counts in ballots.items():
        high = max(counts.values())
        best = [c for c, n in counts.items() if n == high]
        assert len(best) == 1, 'tied ballot'
        winners[shape] = best[0]
    return dict(grid=grid, box=box, background=bg, carrier=carrier,
                bbox=(y, x, bottom, right), active=active, queries=queries,
                ballots=dict(ballots), winners=winners, objects=raws)


def _paint(data):
    out = data['box'].copy()
    for _, component, boundary in data['queries']:
        color = data['winners'][_canon(component)]
        for p in boundary:
            out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only, stateless implementation of recursive contour voting."""
    return _paint(_parse(input_grid))


def _signature(cells, equivalence):
    raw = _norm(cells)
    if equivalence == 'area':
        return len(raw)
    if equivalence == 'bbox':
        return tuple(sorted(_size(raw)))
    if equivalence == 'rotation':
        return min(_rotations(raw))
    if equivalence == 'translation':
        return raw
    return _canon(raw)


def _alternative(data, equivalence='free', geometry='eight', initial_only=False):
    counts = defaultdict(Counter)
    for color, raw in data['objects']:
        counts[_signature(raw, equivalence)][color] += 1
    out = data['box'].copy()
    if initial_only:
        queries = ((0, c, c) for c in _split(data['active']))
    else:
        queries = _peels(data['active'], DIR4 if geometry == 'four' else DIR8,
                         rectangle=geometry == 'box')
    for _, component, boundary in queries:
        by_color = counts.get(_signature(component, equivalence))
        if not by_color:
            return None
        high = max(by_color.values())
        best = [color for color, n in by_color.items() if n == high]
        if len(best) != 1:
            return None
        for p in boundary:
            out[p] = best[0]
    return out


def _fails(data, output, **kwargs):
    result = _alternative(data, **kwargs)
    return result is None or not np.array_equal(result, output)


def _distance_output(mask, carrier, winners):
    """Generation oracle: BFS distance levels, not solve's iterative peeling."""
    active = set(map(tuple, np.argwhere(mask)))
    depths, queue = {}, deque()
    for r, c in sorted(active):
        if any((r+dr, c+dc) not in active for dr, dc in DIR8):
            depths[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            point = r+dr, c+dc
            if point in active and point not in depths:
                depths[point] = depths[r, c]+1
                queue.append(point)
    output = np.full(mask.shape, carrier, dtype=np.int8)
    for d in range(max(depths.values())+1):
        for component in _split(p for p in active if depths[p] >= d):
            color = winners[_canon(component)]
            for p in component:
                if depths[p] == d:
                    output[p] = color
    return output


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False


def _positions(available, shape):
    h, w = _size(shape)
    nr, nc = available.shape[0]-h+1, available.shape[1]-w+1
    if nr <= 0 or nc <= 0:
        return np.empty((0, 2), dtype=int)
    valid = np.ones((nr, nc), dtype=bool)
    for r, c in shape:
        valid &= available[r:r+nr, c:c+nc]
    return np.argwhere(valid)


def _put(available, options, rng):
    for i in rng.permutation(len(options)):
        shape = options[int(i)]
        positions = _positions(available, shape)
        if len(positions):
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            cells = tuple((r+dr, c+dc) for dr, dc in shape)
            _block(available, cells)
            return cells
    return None


def _target_mask(rng, shapes):
    """Pack cavities on actual legal cells, then trim with a variable margin."""
    footprint = sum((h+1)*(w+1) for h, w in map(_size, shapes))
    max_side = max(max(_size(s)) for s in shapes)
    # The orientation is free. Long necks and compact thick shapes share this code.
    choices = [(h, w) for h in range(5, 24) for w in range(5, 24)
               if max(h, w) >= max_side+2 and (h-2)*(w-2) >= footprint
               and h*w <= max(240, footprint*2)]
    choices.sort(key=lambda s: s[0]*s[1])
    # Bias toward economical carrier area while retaining several aspect ratios.
    h, w = _pick(rng, choices[:max(12, len(choices)//2)])
    available = np.ones((h, w), dtype=bool)
    available[[0, -1], :] = False
    available[:, [0, -1]] = False
    occupied = []
    for shape in sorted(shapes, key=lambda s: max(_size(s))*len(s), reverse=True):
        cells = _put(available, _variants(_norm(shape)), rng)
        if cells is None:
            return None
        occupied.extend(cells)
    rs, cs = zip(*occupied)
    y = max(0, min(rs)-1-int(rng.integers(3)))
    x = max(0, min(cs)-1-int(rng.integers(3)))
    bottom = min(h-1, max(rs)+1+int(rng.integers(3)))
    right = min(w-1, max(cs)+1+int(rng.integers(3)))
    mask = np.zeros((bottom-y+1, right-x+1), dtype=bool)
    for r, c in occupied:
        mask[r-y, c-x] = True
    if len(_split(map(tuple, np.argwhere(~mask)))) != 1:
        return None
    return mask


def _branch(rng, teaching=False):
    if teaching:
        h1, w1, h2, w2 = 5, 5, 5, 7
    else:
        h1, h2 = int(rng.choice((3, 5, 7))), int(rng.choice((3, 5)))
        w1, w2 = int(rng.integers(3, 7)), int(rng.integers(3, 8))
    gap = int(rng.integers(1, 4))
    offset = int(rng.integers(-1, 2))
    top1, top2 = max(0, -offset), max(0, offset)
    left = {(top1+r, c) for r in range(h1) for c in range(w1)}
    right = {(top2+r, w1+gap+c) for r in range(h2) for c in range(w2)}
    lo, hi = max(top1, top2), min(top1+h1-1, top2+h2-1)
    row = int(rng.integers(lo, hi+1))
    neck = {(row, c) for c in range(w1, w1+gap)}
    return _norm(left | neck | right)


def _grown(rng, area):
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in DIR4}-cells)
            cells.add(_pick(rng, frontier))
        shape = _norm(cells)
        h, w = _size(shape)
        if max(h, w) <= 4:
            # Exclude internal holes, including holes later closed by dilation.
            padded = {(r, c) for r in range(-1, h+1) for c in range(-1, w+1)}-set(shape)
            if len(_split(padded)) == 1:
                return shape
    # A connected staircase, still outside the four-cell teaching vocabulary.
    return _norm((i//2, (i+1)//2) for i in range(area))


def _shapes(rng, kind, anchor, companion):
    if kind == 'first':
        return [_dilate(anchor, 1), _dilate(companion, 1)]
    if kind == 'second':
        return [_branch(rng, teaching=True), anchor]
    if kind == 'test':
        if rng.random() < .3:
            result = [_branch(rng)]
        else:
            base = _grown(rng, int(rng.integers(5, 8)))
            radius = 3 if rng.random() < .3 else 2
            result = [_dilate(base, radius)]
    elif rng.random() < .45:
        result = [_branch(rng)]
    else:
        base = _grown(rng, int(rng.integers(1, 7)))
        result = [_dilate(base, int(rng.integers(1, 3)))]
    # Repeats may reuse an inner lookup already supplied by the main cavity.
    if rng.random() < .55:
        cores = [_norm(c) for _, c, _ in _peels(result[0]) if len(c) <= 7]
        extra = _pick(rng, cores) if cores else _norm(anchor)
        result.append(extra)
        if rng.random() < .25:
            result.append(extra)
    return result


def _counts(rng, shape):
    if len(shape) > 18 or max(_size(shape)) > 7:
        return (1,)
    if len(shape) > 8:
        return _pick(rng, ((1,), (1,), (2,), (2, 1)))
    return _pick(rng, ((1,), (2,), (2, 1), (3, 2), (3, 2, 2), (2, 1, 1), (4, 3)))


def _make_pair(rng, shapes, roles, kind, anchor, forbidden):
    bg, carrier = roles
    colors = [int(c) for c in rng.permutation(10) if c not in roles]
    for attempt in range(48):
        mask = _target_mask(rng, shapes)
        if mask is None:
            continue
        active = set(map(tuple, np.argwhere(mask)))
        queries = list(_peels(active))
        needed = sorted({_canon(c) for _, c, _ in queries}, key=lambda s: (len(s), s))
        if len(needed) > 10:
            continue
        winners = {s: colors[i % len(colors)] for i, s in enumerate(needed)}
        anchor_key = _canon(anchor)
        if kind == 'second' and winners[anchor_key] == forbidden:
            choices = [c for c in colors if c != forbidden]
            winners[anchor_key] = _pick(rng, choices)
        # The unequal remnants must teach shape-conditioned colors at equal depth.
        if kind == 'second':
            child_groups = [list(_split(_erode(c))) for c in _split(active)]
            children = next(g for g in child_groups if len(g) >= 2)
            left, right = map(_canon, children[:2])
            if winners[left] == winners[right]:
                winners[right] = _pick(rng, [c for c in colors if c != winners[left]])
        ballots = {s: _counts(rng, s) for s in needed}
        if kind == 'first':
            ballots[anchor_key] = (3, 2, 2)
        def cost(s):
            h, w = _size(s)
            return (h+1)*(w+1)*sum(ballots[s])
        for s in sorted(needed, key=cost, reverse=True):
            if sum(cost(t) for t in needed) <= 400:
                break
            if not (kind == 'first' and s == anchor_key):
                ballots[s] = (1,)
        specs = []
        raw_anchor = next((_norm(c) for _, c, _ in queries if _canon(c) == anchor_key), None)
        for shape in needed:
            losers = [c for c in colors if c != winners[shape]]
            rng.shuffle(losers)
            for j, (color, n) in enumerate(zip([winners[shape]]+losers, ballots[shape])):
                options = _variants(shape)
                if kind == 'first' and shape == anchor_key:
                    # Winner and second loser are reflected; the unreflected
                    # first loser then wins the rotation-only counterfactual.
                    options = (_reflect(raw_anchor),) if j != 1 else (raw_anchor,)
                specs.extend((shape, color, options) for _ in range(n))
        rng.shuffle(specs)
        specs.sort(key=lambda item: (max(_size(item[0])), len(item[0])), reverse=True)
        sh, sw = mask.shape
        exterior_cost = sum(cost(s) for s in needed)
        side = int(np.ceil(np.sqrt(sh*sw+exterior_cost*1.7)))+2
        height = min(30, max(sh+3, side+int(rng.integers(-2, 4))))
        width = min(30, max(sw+3, side+int(rng.integers(-2, 4))))
        if attempt >= 6 or rng.random() < .18:
            height = width = 30
        y, x = int(rng.integers(height-sh+1)), int(rng.integers(width-sw+1))
        edge = int(rng.integers(7))
        if edge == 0:
            y = 0
        elif edge == 1:
            y = height-sh
        elif edge == 2:
            x = 0
        elif edge == 3:
            x = width-sw
        available = np.ones((height, width), dtype=bool)
        available[max(0, y-1):min(height, y+sh+1), max(0, x-1):min(width, x+sw+1)] = False
        placed = []
        for _, color, options in specs:
            cells = _put(available, options, rng)
            if cells is None:
                break
            placed.append((color, cells))
        else:
            inp = np.full((height, width), bg, dtype=np.int8)
            box = np.full(mask.shape, carrier, dtype=np.int8)
            box[mask] = bg
            inp[y:y+sh, x:x+sw] = box
            for color, cells in placed:
                for p in cells:
                    inp[p] = color
            out = _distance_output(mask, carrier, winners)
            # Global symmetries move layouts and preserve the crop relation.
            turns = int(rng.integers(4))
            inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
            if rng.random() < .5:
                inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
            return {'input': inp, 'output': out}
    return None


def _legal_count(value, low, high):
    return (isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            and low <= int(value) <= high)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _legal_count(num_train, 2, 10) or not _legal_count(num_test, 1, 4) or num_train <= num_test:
        raise ValueError('require 2..10 training, 1..4 test, and num_train > num_test')
    if (not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    anchor = _canon(_pick(rng, (L4, S4)))
    companion = _canon(_pick(rng, [s for s in (L4, S4, T4) if _canon(s) != anchor]))
    backgrounds = list(map(int, rng.permutation(10)))
    carriers = backgrounds[1:]+backgrounds[:1]
    puzzle, seen, known = {'train': [], 'test': []}, set(), set()
    first_winner = None
    for i in range(num_train+num_test):
        kind = 'first' if i == 0 else 'second' if i == 1 else 'train' if i < num_train else 'test'
        roles = backgrounds[i % 10], carriers[i % 10]
        for _ in range(32):
            shapes = _shapes(rng, kind, anchor, companion)
            pair = _make_pair(rng, shapes, roles, kind, anchor, first_winner)
            if pair is None or _key(pair['input']) in seen:
                continue
            data = _parse(pair['input'])
            if i >= num_train and not set(data['ballots'])-known:
                continue
            break
        else:
            raise RuntimeError('bounded construction exhausted')
        split = 'train' if i < num_train else 'test'
        puzzle[split].append(pair)
        seen.add(_key(pair['input']))
        if split == 'train':
            known.update(data['ballots'])
        if i == 0:
            first_winner = data['winners'][anchor]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            data = _parse(inp)
            assert out.shape == data['box'].shape, 'wrong crop'
            assert np.array_equal(out, _paint(data)), 'wrong contour colors'
            carrier_mask = data['box'] == data['carrier']
            assert np.array_equal(out[carrier_mask], data['box'][carrier_mask])
            assert np.all(out[~carrier_mask] != data['background'])
            assert np.all(out[~carrier_mask] != data['carrier'])
            analyses[split].append((data, out))
    train = analyses['train']
    # Concrete witnesses: omitting either inherited operation changes an answer.
    assert any(_fails(a, o, initial_only=True) for a, o in train), 'A has no effect'
    depth_witness = False
    branch_witness = False
    history, known = defaultdict(set), set()
    for a, _ in train:
        by_depth = defaultdict(set)
        for depth, component, _ in a['queries']:
            shape = _canon(component)
            by_depth[depth].add(a['winners'][shape])
        depth_witness |= any(len(colors) > 1 for colors in by_depth.values())
        for component in _split(a['active']):
            children = _split(_erode(component))
            if len(children) > 1:
                branch_witness |= len({a['winners'][_canon(c)] for c in children}) > 1
        for shape, color in a['winners'].items():
            history[shape].add(color)
        known.update(a['ballots'])
    assert depth_witness, 'depth-only coloring is not refuted'
    assert branch_witness, 'no independently colored eroded branches'
    assert any(len(colors) > 1 for colors in history.values()), 'fixed color map not refuted'
    for mode in ('area', 'bbox', 'rotation', 'translation'):
        assert any(_fails(a, o, equivalence=mode) for a, o in train), mode
    for geometry in ('four', 'box'):
        assert any(_fails(a, o, geometry=geometry) for a, o in train), geometry
    assert any(len(counts) >= 3 and max(counts.values())*2 <= sum(counts.values())
               for a, _ in train for counts in a['ballots'].values()), 'no plurality witness'
    assert len({a['background'] for a, _ in train}) >= 2
    assert len({a['carrier'] for a, _ in train}) >= 2
    assert all(set(a['ballots'])-known for a, _ in analyses['test']), 'no new test silhouette'


def self_test():
    """Bounded 42-seed variation audit; no files and no saved execution traces."""
    signatures, dimensions, roles, silhouettes, geometries = (set() for _ in range(5))
    cavity_counts, class_counts, depths, patterns = (set() for _ in range(4))
    branches = repeats = nonmajorities = test_branches = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _parse(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                dimensions.add((pair['input'].shape, pair['output'].shape))
                roles.add((data['background'], data['carrier']))
                silhouettes.update(data['ballots'])
                geometries.add(_key((data['box'] == data['background']).astype(np.int8)))
                cavities = _split(data['active'])
                cavity_counts.add(len(cavities))
                class_counts.add(len(data['ballots']))
                depths.add(max(d for d, _, _ in data['queries']))
                repeats += len({_canon(c) for c in cavities}) < len(cavities)
                branching = any(len(_split(_erode(c))) > 1 for c in cavities)
                branches += branching
                test_branches += split == 'test' and branching
                for counts in data['ballots'].values():
                    pattern = tuple(sorted(counts.values(), reverse=True))
                    patterns.add(pattern)
                    nonmajorities += len(pattern) >= 3 and pattern[0]*2 <= sum(pattern)
    assert len(signatures) == 42
    assert len(dimensions) >= 70 and len(roles) >= 40
    assert len(silhouettes) >= 70 and len(geometries) >= 120
    assert {1, 2, 3} <= cavity_counts and {1, 2, 3} <= depths
    assert len(class_counts) >= 4 and len(patterns) >= 5
    assert branches >= 42 and repeats >= 10 and nonmajorities >= 42
    assert test_branches >= 5
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(9173+nt*10+ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    original = generate_puzzle(81, 2, 1)
    for corruption in ('carrier', 'contour', 'vote'):
        broken = {s: [{k: np.array(v, copy=True) for k, v in p.items()}
                      for p in original[s]] for s in original}
        pair = broken['test'][0]
        data = _parse(pair['input'])
        if corruption == 'carrier':
            pair['output'][0, 0] = data['background']
        elif corruption == 'contour':
            point = min(data['active'])
            pair['output'][point] = data['carrier']
        else:
            # Delete a whole shape class's exterior evidence, not metadata.
            y, x, bottom, right = data['bbox']
            grid = pair['input']
            external = grid.copy()
            external[y:bottom+1, x:right+1] = data['background']
            shape = next(iter(data['ballots']))
            for cells in _split(map(tuple, np.argwhere(external != data['background']))):
                if _canon(cells) == shape:
                    for point in cells:
                        grid[point] = data['background']
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted '+corruption+' accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'shape_pairs': len(dimensions),
            'role_pairs': len(roles), 'silhouette_classes': len(silhouettes),
            'carrier_geometries': len(geometries), 'cavity_counts': sorted(cavity_counts),
            'class_counts': sorted(class_counts), 'maximum_depths': sorted(depths),
            'branching_pairs': branches, 'repeated_cavity_pairs': repeats,
            'test_branching_pairs': test_branches,
            'nonmajority_ballots': nonmajorities, 'ballot_patterns': sorted(patterns)}
