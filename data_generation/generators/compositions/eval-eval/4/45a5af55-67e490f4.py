"""Peeling shape elections: exact-shape ballots color successive erosion shells."""
from collections import Counter
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _normal(cells):
    cells = tuple(cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r-y), int(c-x)) for r, c in cells))


def _rotations(cells):
    result, shape = set(), _normal(cells)
    for _ in range(4):
        result.add(shape)
        shape = _normal((c, -r) for r, c in shape)
    return tuple(sorted(result))


def _mirror(cells):
    return _normal((r, -c) for r, c in cells)


def _variants(cells):
    cells = tuple(cells)
    return tuple(sorted(set(_rotations(cells)) | set(_rotations(_mirror(cells)))))


def _canonical(cells):
    return min(_variants(cells))


def _extent(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, found = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _STEPS:
                point = (r+dr, c+dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
                    found.append(point)
        result.append(tuple(sorted(found)))
    return result


def _mask_components(mask):
    return _components(map(tuple, np.argwhere(mask)))


def _erosion(cells):
    cells = set(cells)
    return {p for p in cells if all((p[0]+dr, p[1]+dc) in cells for dr, dc in _STEPS)}


def _shell_records(cells):
    """Re-segment after each erosion; a split creates independent shape keys."""
    active, depth = set(cells), 0
    records = []
    while active:
        survivors = set()
        for component in _components(active):
            inside = _erosion(component)
            records.append((depth, _canonical(component), component,
                            tuple(sorted(set(component)-inside))))
            survivors.update(inside)
        active, depth = survivors, depth+1
    return records


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'grid size'
    assert np.issubdtype(a.dtype, np.integer), 'integer cells required'
    assert np.all((0 <= a) & (a <= 9)), 'color range'
    return a.astype(np.int8, copy=False)


def _read_input(value):
    grid = _grid(value)
    rectangles = []
    for color in np.unique(grid):
        for cells in _mask_components(grid == color):
            rs, cs = zip(*cells)
            y, x, end_y, end_x = min(rs), min(cs), max(rs)+1, max(cs)+1
            box = grid[y:end_y, x:end_x]
            if min(box.shape) < 3:
                continue
            if not (np.all(box[0] == color) and np.all(box[-1] == color)
                    and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)):
                continue
            other = set(map(int, np.unique(box))) - {int(color)}
            if len(other) == 1 and np.count_nonzero(grid == color) == len(cells):
                rectangles.append((int(color), other.pop(), (y, x, end_y, end_x)))
    assert len(rectangles) == 1, 'unique perforated rectangular carrier required'
    carrier, background, bbox = rectangles[0]
    y, x, end_y, end_x = bbox
    box = grid[y:end_y, x:end_x]
    cavities = _mask_components(box == background)
    assert cavities, 'no cavities'
    exterior = grid.copy()
    exterior[y:end_y, x:end_x] = background
    objects = []
    # Segment foreground irrespective of color, then require monochrome objects.
    for cells in _mask_components(exterior != background):
        colors = {int(exterior[r, c]) for r, c in cells}
        assert len(colors) == 1, 'touching objects with different colors'
        color = colors.pop()
        assert color not in (carrier, background)
        assert all(not (y-1 <= r <= end_y and x-1 <= c <= end_x) for r, c in cells), 'ballot touches carrier moat'
        objects.append((color, cells))
    assert objects, 'no ballots'
    # The generator uses a full one-cell moat, also making diagonal contacts absent.
    assert len(_mask_components(box == carrier)) == 1, 'disconnected carrier'
    records = [record for cavity in cavities for record in _shell_records(cavity)]
    return dict(grid=grid, box=box, carrier=carrier, background=background,
                bbox=bbox, cavities=cavities, objects=objects, records=records)


def _elections(data, equivalence='free'):
    def key(cells):
        if equivalence == 'area':
            return len(cells)
        if equivalence == 'box':
            return tuple(sorted(_extent(_normal(cells))))
        if equivalence == 'rotation':
            return min(_rotations(cells))
        if equivalence == 'translation':
            return _normal(cells)
        return _canonical(cells)
    ballots = {}
    for color, cells in data['objects']:
        ballots.setdefault(key(cells), Counter())[color] += 1
    required = {key(cells) for _, _, cells, _ in data['records']}
    if equivalence == 'free':
        assert set(ballots) == required, 'unsupported shell shape or irrelevant ballot'
    else:
        assert required <= set(ballots), 'unsupported shape under shortcut'
    winners = {}
    for shape in required:
        counts = ballots[shape]
        high = max(counts.values())
        best = [c for c, n in counts.items() if n == high]
        assert len(best) == 1, 'tied plurality'
        winners[shape] = best[0]
    return winners, ballots, key


def _answer(data, winners, key, uniform=False):
    result = data['box'].copy()
    if uniform:
        groups = [(cells, winners[key(cells)]) for cells in data['cavities']]
    else:
        groups = [(shell, winners[key(cells)]) for _, _, cells, shell in data['records']]
    for cells, color in groups:
        for r, c in cells:
            result[r, c] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    data = _read_input(input_grid)
    winners, _, key = _elections(data)
    return _answer(data, winners, key)


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _rectangle(h, w):
    return _canonical((r, c) for r in range(h) for c in range(w))


def _thick_l(h, w, thickness):
    return _canonical((r, c) for r in range(h) for c in range(w)
                      if r >= h-thickness or c < thickness)


def _dumbbell(gap, shifted=True):
    # Offset lobes make this chiral; a one-cell neck vanishes on first erosion.
    dy = int(shifted)
    cells = {(r, c) for r in range(3) for c in range(3)}
    cells |= {(r+dy, c+3+gap) for r in range(3) for c in range(3)}
    cells |= {(1, c) for c in range(3, 3+gap)}
    return _canonical(cells)


def _training_pool(rng):
    shapes = {_rectangle(h, w) for h in range(3, 7) for w in range(3, 7)}
    for h in range(4, 7):
        for w in range(4, 7):
            shapes.add(_thick_l(h, w, int(rng.integers(2, 4))))
    for n in range(5, 8):
        shapes.add(_canonical((r, c) for r in range(n) for c in range(n)
                              if abs(r-n//2) <= 1 or abs(c-n//2) <= 1))
    shapes.update(_dumbbell(gap, shifted) for gap in (1, 2, 3) for shifted in (True, False))
    # Grow different 4-connected, hole-free forms around a solid 3x3 nucleus.
    for _ in range(12):
        cells = {(r, c) for r in range(3) for c in range(3)}
        for _ in range(int(rng.integers(2, 10))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _STEPS}-cells)
            cells.add(_choice(rng, frontier))
        shape = _canonical(cells)
        h, w = _extent(shape)
        complement = {(r, c) for r in range(-1, h+1) for c in range(-1, w+1)}-set(shape)
        if max(h, w) <= 7 and len(_components(complement)) == 1:
            shapes.add(shape)
    # Reserve deeper iterative nesting for the test shift, based on actual
    # erosion depth rather than bounding-box dimensions of concave shapes.
    return tuple(sorted(s for s in shapes if max(r[0] for r in _shell_records(s)) <= 2))


def _positions(available, shape):
    h, w = _extent(shape)
    nr, nc = available.shape[0]-h+1, available.shape[1]-w+1
    if nr <= 0 or nc <= 0:
        return np.empty((0, 2), dtype=int)
    mask = np.ones((nr, nc), dtype=bool)
    for r, c in shape:
        mask &= available[r:r+nr, c:c+nc]
    return np.argwhere(mask)


def _reserve(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False


def _pack(available, variants, rng):
    for i in rng.permutation(len(variants)):
        shape = variants[int(i)]
        positions = _positions(available, shape)
        if len(positions):
            y, x = map(int, _choice(rng, positions))
            cells = tuple((r+y, c+x) for r, c in shape)
            _reserve(available, cells)
            return cells
    return None


def _ballot(rng, area):
    if area > 16:
        return (1,) if rng.random() < .85 else (2, 1)
    if area > 6:
        return _choice(rng, ((1,), (1,), (2, 1), (3, 2)))
    return _choice(rng, ((1,), (2,), (2, 1), (3, 2), (3, 2, 2), (2, 1, 1)))


def _make_pair(rng, targets, rich=None, reflected=None, forbidden=None, previous_roles=None):
    targets = tuple(targets)
    required = sorted({s for t in targets for _, s, _, _ in _shell_records(t)})
    palette = list(map(int, rng.permutation(10)))
    if previous_roles is not None:
        bg = _choice(rng, [c for c in palette if c != previous_roles[0]])
        carrier = _choice(rng, [c for c in palette if c not in (bg, previous_roles[1])])
        palette = [bg, carrier] + [c for c in palette if c not in (bg, carrier)]
    bg, carrier, colors = palette[0], palette[1], palette[2:]
    # Only teaching contrasts are compulsory; other shells may repeat colors.
    edges = set()
    first_records = _shell_records(targets[0])
    if len(first_records) > 1:
        edges.add(tuple(sorted((targets[0], first_records[1][1]))))
    if rich is not None and len(targets) > 1:
        edges.add(tuple(sorted((targets[0], targets[1]))))
        if len(targets) >= 3:
            chain = _shell_records(targets[2])
            for earlier, later in zip(chain, chain[1:]):
                edges.add(tuple(sorted((earlier[1], later[1]))))
    if reflected is not None:
        branches = sorted({s for depth, s, _, _ in _shell_records(reflected) if depth == 1})
        for a, b in zip(branches, branches[1:]):
            edges.add(tuple(sorted((a, b))))
    winners = {}
    for shape in required:
        excluded = {winners[t] for a, b in edges for t in (a, b)
                    if shape in (a, b) and t != shape and t in winners}
        if forbidden and shape in forbidden:
            excluded.add(forbidden[shape])
        winners[shape] = _choice(rng, [c for c in colors if c not in excluded])
    ballots = {s: _ballot(rng, len(s)) for s in required}
    if rich is not None:
        ballots[rich] = (3, 2, 2)
    if reflected is not None:
        ballots[reflected] = (2, 1)
    def footprint(s):
        h, w = _extent(s)
        return (h+1)*(w+1)*sum(ballots[s])
    for s in sorted(required, key=footprint, reverse=True):
        if sum(footprint(t) for t in required) <= 360:
            break
        if s not in (rich, reflected):
            ballots[s] = (1,)
    specs = []
    for shape in required:
        losers = [c for c in colors if c != winners[shape]]
        rng.shuffle(losers)
        for color, number in zip([winners[shape]]+losers, ballots[shape]):
            variants = _variants(shape)
            if shape == reflected:
                variants = _rotations(_mirror(shape)) if color == winners[shape] else _rotations(shape)
            specs.extend((shape, color, variants) for _ in range(number))
    shelf_variants = [_choice(rng, [v for v in _variants(t) if _extent(v)[0] == min(_extent(z)[0] for z in _variants(t))])
                      for t in targets]
    shelf_h = max(_extent(t)[0] for t in shelf_variants)+2
    shelf_w = sum(_extent(t)[1] for t in shelf_variants)+len(targets)+1
    assert shelf_w <= 28 and shelf_h <= 28, 'target footprint too large'
    for attempt in range(80):
        if attempt >= 24:
            sh, sw = shelf_h, shelf_w
            placements, x = [], 1
            for t, variant in zip(targets, shelf_variants):
                y = int(rng.integers(1, sh-_extent(variant)[0]))
                cells = tuple((r+y, c+x) for r, c in variant)
                placements.append((t, cells))
                x += _extent(variant)[1]+1
        else:
            sh = min(24, shelf_h+int(rng.integers(0, 5)))
            sw = min(26, shelf_w+int(rng.integers(0, 5)))
            if rng.random() < .3:
                side = min(24, max(sh, int(np.ceil(np.sqrt(sh*sw)))+1))
                sh = sw = side
            available = np.ones((sh, sw), dtype=bool)
            available[[0, -1], :] = False
            available[:, [0, -1]] = False
            placements = []
            for t in sorted(targets, key=lambda s: (len(s), max(_extent(s))), reverse=True):
                variants = _rotations(t) if t == reflected else _variants(t)
                cells = _pack(available, variants, rng)
                if cells is None:
                    break
                placements.append((t, cells))
            if len(placements) != len(targets):
                continue
        # The shelf fallback must preserve the chirality teaching contrast.
        if reflected is not None:
            placements = [(t, cells) for t, cells in placements
                          if t != reflected or _normal(cells) in _rotations(t)]
            if len(placements) != len(targets):
                # Flip the entire shelf cavity vertically if necessary.
                placements = []
                x = 1
                for t, variant in zip(targets, shelf_variants):
                    if t == reflected and variant not in _rotations(t):
                        variant = _normal((-r, c) for r, c in variant)
                    cells = tuple((r+1, c+x) for r, c in variant)
                    placements.append((t, cells))
                    x += _extent(variant)[1]+1
        cost = sum(footprint(s) for s in required)
        side = int(np.clip(np.sqrt(sh*sw+cost*1.8)+rng.integers(1, 4), 17, 30))
        h = min(30, max(sh+2, side+int(rng.integers(-1, 3))))
        w = min(30, max(sw+2, side+int(rng.integers(-1, 3))))
        if attempt >= 10:
            h = w = 30
        y, x = int(rng.integers(h-sh+1)), int(rng.integers(w-sw+1))
        edge = int(rng.integers(5))
        if edge == 0:
            y = 0
        elif edge == 1:
            y = h-sh
        elif edge == 2:
            x = 0
        elif edge == 3:
            x = w-sw
        available = np.ones((h, w), dtype=bool)
        available[max(0, y-1):min(h, y+sh+1), max(0, x-1):min(w, x+sw+1)] = False
        external = []
        order = [specs[int(i)] for i in rng.permutation(len(specs))]
        order.sort(key=lambda v: (max(_extent(v[0])), len(v[0])), reverse=True)
        for s, color, variants in order:
            cells = _pack(available, variants, rng)
            if cells is None:
                break
            external.append((color, cells))
        if len(external) != len(specs):
            continue
        box = np.full((sh, sw), carrier, dtype=np.int8)
        out = box.copy()
        for t, cells in placements:
            for r, c in cells:
                box[r, c] = bg
            for _, shape, _, shell in _shell_records(cells):
                for r, c in shell:
                    out[r, c] = winners[shape]
        if len(_mask_components(box == carrier)) != 1:
            continue
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[y:y+sh, x:x+sw] = box
        for color, cells in external:
            for r, c in cells:
                inp[r, c] = color
        # Rotate/reflect the complete scene and crop together. This varies
        # carrier aspect and edge without introducing a hidden reading direction.
        k = int(rng.integers(4))
        inp, out = np.rot90(inp, k), np.rot90(out, k)
        if rng.random() < .5:
            inp, out = np.fliplr(inp), np.fliplr(out)
        return {'input': inp.copy(), 'output': out.copy()}
    raise RuntimeError('bounded ballot packing exhausted')


def _check_counts(num_train, num_test):
    for name, count, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not low <= count <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    plus9 = _canonical({(2, c) for c in range(5)} | {(r, 2) for r in range(5)})
    tee16 = _canonical({(r, c) for r in range(2) for c in range(6)} |
                       {(r, c) for r in range(2, 4) for c in (2, 3)})
    witnesses = ((_rectangle(3, 3), plus9), (_rectangle(3, 4), _thick_l(4, 4, 2)),
                 (_rectangle(3, 5), _thick_l(4, 4, 3)), (_rectangle(4, 4), tee16),
                 (_rectangle(4, 5), _dumbbell(2)))
    anchor, companion = _choice(rng, witnesses)
    branch = _dumbbell(int(rng.integers(1, 4)))
    pool = _training_pool(rng)
    result, seen = {'train': [], 'test': []}, set()
    first_data = None
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            rich, reflected, forbidden, roles = None, None, None, None
            if split == 'train' and i == 0:
                # A third, three-shell cavity teaches iteration before tests
                # extrapolate to four/five shells, even with only two examples.
                targets = [anchor, companion, _rectangle(int(rng.integers(5, 7)), int(rng.integers(5, 7)))]
                rich = anchor
            elif split == 'train' and i == 1:
                targets, reflected = [anchor, branch], branch
                forbidden = {anchor: first_data['winners'][anchor]}
                roles = first_data['background'], first_data['carrier']
            elif split == 'train':
                n = int(rng.integers(1, 4))
                # Large pluses are used alone; multi-cavity layouts stay feasible.
                choices = [s for s in pool if n == 1 or max(_extent(s)) <= 6]
                targets = [_choice(rng, choices) for _ in range(n)]
            else:
                # A real iteration shift: all tests have a >=7-wide solid
                # cavity nucleus; demonstrations have at most three shells.
                side = int(rng.integers(7, 10))
                targets = [_rectangle(side, side+int(rng.integers(0, 2)))]
                if rng.random() < .55:
                    targets.append(_choice(rng, [s for s in pool if max(_extent(s)) <= 5]))
            for _ in range(12):
                pair = _make_pair(rng, targets, rich, reflected, forbidden, roles)
                key = _grid_key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('duplicate constructions')
            seen.add(key)
            result[split].append(pair)
            if split == 'train' and i == 0:
                first_data = _read_input(pair['input'])
                first_data['winners'] = _elections(first_data)[0]
    rng.shuffle(result['train'])
    validate_generated_puzzle(result)
    return result


def _shortcut_fails(pair, mode):
    data = _read_input(pair['input'])
    try:
        winners, _, key = _elections(data, 'free' if mode in ('uniform', 'two_pass') else mode)
        output = _answer(data, winners, key, uniform=mode == 'uniform')
        if mode == 'two_pass':
            output = data['box'].copy()
            for depth, shape, cells, shell in data['records']:
                if depth <= 1:
                    for r, c in (shell if depth == 0 else cells):
                        output[r, c] = winners[shape]
    except AssertionError:
        return True
    return not np.array_equal(output, pair['output'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _grid_key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            data = _read_input(inp)
            winners, ballots, classify = _elections(data)
            assert np.array_equal(out, _answer(data, winners, classify)), 'wrong shell fill or crop'
            assert np.all(out[data['box'] == data['carrier']] == data['carrier'])
            shell_cells = [p for _, _, _, shell in data['records'] for p in shell]
            cavity_cells = [p for cells in data['cavities'] for p in cells]
            assert len(shell_cells) == len(set(shell_cells)) and set(shell_cells) == set(cavity_cells)
            # Full moat around all ballot components, including at canvas edges.
            cell_object = {p: i for i, (_, cells) in enumerate(data['objects']) for p in cells}
            for (r, c), i in cell_object.items():
                assert all(cell_object.get((r+dr, c+dc), i) == i
                           for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            data['winners'], data['ballots'] = winners, ballots
            analyses[split].append(data)
    train = puzzle['train']
    assert any(_shortcut_fails(p, 'uniform') for p in train), 'no nested-depth evidence'
    assert any(_shortcut_fails(p, 'two_pass') for p in train), 'iteration not distinguished'
    assert any(_shortcut_fails(p, 'area') for p in train), 'no exact-shape evidence'
    assert any(_shortcut_fails(p, 'rotation') for p in train), 'no reflection evidence'
    assert any(len(v) >= 3 and max(v.values())*2 <= sum(v.values())
               for d in analyses['train'] for v in d['ballots'].values()), 'no plurality evidence'
    assert any(sum(rec[0] == 1 for rec in _shell_records(c)) >= 2
               for d in analyses['train'] for c in d['cavities']), 'no splitting evidence'
    assert any(len({d['winners'][shape] for depth, shape, _, _ in _shell_records(c) if depth == 1}) > 1
               for d in analyses['train'] for c in d['cavities']), 'branch elections not distinguished'
    history = {}
    for d in analyses['train']:
        for s, color in d['winners'].items():
            history.setdefault(s, set()).add(color)
    assert any(len(colors) > 1 for colors in history.values()), 'fixed shape-color dictionary not refuted'
    assert len({d['background'] for d in analyses['train']}) > 1
    assert len({d['carrier'] for d in analyses['train']}) > 1
    train_depth = max(r[0] for d in analyses['train'] for r in d['records'])
    assert train_depth >= 2, 'repeated erosion must be demonstrated'
    assert all(max(r[0] for r in d['records']) > train_depth for d in analyses['test']), 'missing depth shift'


def self_test():
    puzzles, vocabulary, dimensions, cavity_counts, ballots, colors, layouts = set(), set(), set(), set(), set(), set(), set()
    equal_shells, split_records, depths = 0, 0, set()
    for seed in range(42):
        nt = 2+seed % 9
        ne = 1+seed % min(4, nt-1)
        puzzle = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(puzzle)
        signature = tuple(_grid_key(np.asarray(p[r])) for s in ('train', 'test') for p in puzzle[s] for r in ('input', 'output'))
        puzzles.add(signature)
        for split in ('train', 'test'):
            for p in puzzle[split]:
                assert np.array_equal(solve(p['input']), p['output'])
                data = _read_input(p['input'])
                winners, counts, _ = _elections(data)
                vocabulary.update(counts)
                ballots.update(tuple(sorted(v.values(), reverse=True)) for v in counts.values())
                dimensions.add((p['input'].shape, p['output'].shape))
                cavity_counts.add(len(data['cavities']))
                colors.update(map(int, np.unique(p['input'])))
                layouts.add((data['bbox'], tuple(sorted(len(c) for c in data['cavities']))))
                depths.add(max(r[0] for r in data['records']))
                for c in data['cavities']:
                    recs = _shell_records(c)
                    split_records += sum(r[0] == 1 for r in recs) > 1
                    equal_shells += len(recs) > 1 and winners[recs[0][1]] == winners[recs[1][1]]
    assert len(puzzles) == 42
    assert len(vocabulary) >= 50 and len(dimensions) >= 80 and len(layouts) >= 100
    assert {1, 2, 3} <= cavity_counts and len(ballots) >= 6
    assert colors == set(range(10)) and {1, 2, 3, 4} <= depths
    assert equal_shells >= 5 and split_records >= 42
    # Explicit geometry fixtures: 4-neighbor erosion, not nearest box edge,
    # diagonal erosion, or erosion without component splitting.
    assert sorted(len(c) for c in _components(_erosion(_dumbbell(2)))) == [1, 2]
    square = _rectangle(7, 7)
    assert [len(r[2]) for r in _shell_records(square)] == [49, 25, 9, 1]
    l_shape = _thick_l(4, 4, 3)
    assert len(_erosion(l_shape)) == 4
    # A thin, immediate-terminal cavity outside the sampled teaching regime.
    thin = np.zeros((11, 12), dtype=np.int8)
    thin[:5, :8] = 8
    thin[2, 2:6] = 0
    thin[7, 2:6] = 4
    expected = thin[:5, :8].copy()
    expected[2, 2:6] = 4
    assert np.array_equal(solve(thin), expected)
    tied = thin.copy()
    tied[9, 7:11] = 5
    try:
        solve(tied)
    except AssertionError:
        pass
    else:
        raise AssertionError('tied ballot accepted')
    first = generate_puzzle(918273, 2, 1)
    second = generate_puzzle(918273, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s]) for k in a)
    # Non-array callers and output corruption are checked too.
    list_puzzle = {s: [{k: v.tolist() for k, v in p.items()} for p in first[s]] for s in first}
    validate_generated_puzzle(list_puzzle)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('invalid counts accepted')


if __name__ == '__main__':
    self_test()
