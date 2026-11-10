"""Ordered mosaic streams: a corner-keyed inventory meets global exact cover.

The double-marked arm is the fast axis. The stencil is always scanned in
physical top-to-bottom, left-to-right order, independent of the source key.
Only an increasing subsequence of source occurrences can be used. Among complete
coverings, the lexicographically earliest occurrence sequence wins.
"""
from functools import lru_cache
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    """Locate the unique L key by border geometry, with no output information."""
    candidates = []
    if min(a.shape) < 4:
        raise ValueError("Source panel too small")
    for bottom in (False, True):
        for right in (False, True):
            b = a[::-1] if bottom else a
            b = b[:, ::-1] if right else b
            origin, guide, bg = map(int, (b[0, 0], b[0, 1], b[-1, -1]))
            if len({origin, guide, bg}) != 3:
                continue
            if not (np.all(b[-1, 1:] == bg) and np.all(b[1:, -1] == bg)):
                continue
            for axis in (0, 1):
                fast, slow = (b[0, 1:], b[1:, 0]) if axis == 0 else (b[1:, 0], b[0, 1:])
                expected = np.full(fast.shape, guide, dtype=np.int8)
                expected[1] = origin  # the second cell away from the origin
                if np.array_equal(fast, expected) and np.all(slow == guide):
                    candidates.append((axis, bottom, right, origin, guide, bg))
    if len(candidates) != 1:
        raise ValueError("Source must have one double-marked L key")
    return candidates[0]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, points = [first], [first]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    points.append(q)
        groups.append(points)
    return groups


def _source(a):
    axis, bottom, right, origin, guide, bg = _key(a)
    mask = a != bg
    mask[-1 if bottom else 0, :] = False
    mask[:, -1 if right else 0] = False
    lanes = {}
    for points in _components(mask):
        r0, c0 = map(int, np.min(points, axis=0))
        r1, c1 = map(int, np.max(points, axis=0) + 1)
        if not (len(points) == (r1-r0)*(c1-c0)
                and (c1-c0 if axis else r1-r0) == 2
                and 1 <= (r1-r0 if axis else c1-c0) <= 9):
            raise ValueError("Every source object must be a filled two-thick strip")
        if not (r0 > 0 and c0 > 0 and r1 < a.shape[0] and c1 < a.shape[1]):
            raise ValueError("Source objects need clear outer margins")
        lanes.setdefault(c0 if axis else r0, []).append((r0, r1, c0, c1))
    if not 1 <= sum(map(len, lanes.values())) <= 14:
        raise ValueError("Source inventory must contain 1..14 occurrences")
    ordered_lanes = sorted(lanes)
    if any(b-a < 3 for a, b in zip(ordered_lanes, ordered_lanes[1:])):
        raise ValueError("Source lanes must be aligned and separated")
    slow_reverse, fast_reverse = (right, bottom) if axis else (bottom, right)
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    tiles, boxes, populations = [], [], []
    for v in sorted(lanes, reverse=slow_reverse):
        group = sorted(lanes[v], key=lambda x: x[0] if axis else x[2], reverse=fast_reverse)
        intervals = sorted((b[0], b[1]) if axis else (b[2], b[3]) for b in group)
        if any(b[0]-a[1] < 1 for a, b in zip(intervals, intervals[1:])):
            raise ValueError("Source strips must be separated")
        populations.append(len(group))
        for r0, r1, c0, c1 in group:
            tiles.append(np.rot90(a[r0:r1, c0:c1], rotation).copy())
            boxes.append((r0, r1, c0, c1))
    return dict(tiles=tiles, boxes=boxes, populations=populations,
                axis=axis, bottom=bottom, right=right, origin=origin,
                guide=guide, vacancy=bg, panel=a)


def _parse(input_grid):
    a = _grid(input_grid)
    candidates = []
    for axis in (0, 1):
        for cut in range(1, a.shape[axis]-1):
            line = a[cut, :] if axis == 0 else a[:, cut]
            if not np.all(line == line[0]):
                continue
            left, right = (a[:cut], a[cut+1:]) if axis == 0 else (a[:, :cut], a[:, cut+1:])
            for source, board in ((left, right), (right, left)):
                try:
                    d = _source(source)
                except ValueError:
                    continue
                colors = set(map(int, np.unique(board)))
                vacancy = d['vacancy']
                if int(line[0]) != d['origin'] or len(colors) != 2 or vacancy not in colors:
                    continue
                board_color = next(iter(colors - {vacancy}))
                if board_color in (d['origin'], d['guide']) or np.any(source == board_color):
                    continue
                d.update(board=board.copy(), board_color=board_color,
                         mask=board == vacancy, divider_axis=axis)
                candidates.append(d)
    if len(candidates) != 1:
        raise ValueError("Input must identify one source, divider and stencil")
    return candidates[0]


def _plan(mask, tiles):
    """First complete anchored covering in increasing source-index order.

    Search state is only the still-empty visible cells and the stream cursor.
    Skipping is permanent. Caching failures and successes bounds repeated work.
    """
    h, w = mask.shape
    full = sum(1 << (int(r)*w+int(c)) for r, c in np.argwhere(mask))
    lengths = tuple(int(t.shape[1]) for t in tiles)

    @lru_cache(None)
    def visit(empty, cursor):
        if empty == 0:
            return ()
        if sum(2*l for l in lengths[cursor:]) < empty.bit_count():
            return None
        first = (empty & -empty).bit_length()-1
        r, c = divmod(first, w)
        if r+2 > h:
            return None
        for i in range(cursor, len(tiles)):
            length = lengths[i]
            if c+length > w:
                continue
            row = ((1 << length)-1) << first
            footprint = row | (row << w)
            if empty & footprint != footprint:
                continue
            tail = visit(empty ^ footprint, i+1)
            if tail is not None:
                return ((i, r, c),) + tail
        return None

    plan = visit(full, 0)
    if plan is None:
        raise ValueError("No increasing inventory subsequence can cover the stencil")
    return plan


def _render(board, tiles, plan):
    out = board.copy()
    for i, r, c in plan:
        tile = tiles[i]
        out[r:r+2, c:c+tile.shape[1]] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _parse(input_grid)
    return _render(d['board'], d['tiles'], _plan(d['mask'], d['tiles']))


def _pattern(rng, length, palette, asymmetric=False):
    a = np.full((2, length), rng.choice(palette), dtype=np.int8)
    mode = int(rng.integers(5))
    if mode == 1:
        a[:] = rng.choice(palette, size=(2, 1))
    elif mode == 2:
        a[:] = rng.choice(palette, size=(1, length))
    elif mode == 3:
        yy, xx = np.indices(a.shape)
        a[:] = np.array(rng.choice(palette, size=2))[(yy+xx) % 2]
    elif mode == 4:
        a[:] = rng.choice(palette, size=a.shape)
    if asymmetric:
        a.fill(palette[0])
        a[0, 0] = palette[1]
        a[1, -1] = palette[2]
    return a


def _witness(rng, palette, which):
    """Construct labels from explicit rectangles, then verify input reconstruction."""
    small = int(rng.integers(1, 4))
    long = small + int(rng.integers(2, 7))
    if which == 0:
        other = long-small
        rects = [(0, 0, long), (2, int(rng.integers(long-other+1)), other)]
        whole = _pattern(rng, long, palette, True)
        later = whole.copy()
        later[0, 0] = palette[2]  # same footprint, distinguishable occurrence
        tiles = [_pattern(rng, small, palette), whole,
                 _pattern(rng, other, palette), later]
        selected = [1, 2]
    else:
        tail_x = int(rng.integers(long-small+1))
        rects = [(0, 0, small), (2, 0, long), (4, tail_x, small)]
        later = _pattern(rng, long, palette, True)
        early = later.copy()
        early[0, 0] = palette[2]
        short = _pattern(rng, small, palette)
        tiles = [early, short, later, short.copy()]
        selected = [1, 2, 3]
    return rects, tiles, selected


def _grow(rng):
    h, w = int(rng.integers(6, 13)), int(rng.integers(7, 19))
    mask = np.zeros((h, w), dtype=bool)
    rects = []
    target = int(rng.integers(2, 9))
    for i in range(target):
        length = int(rng.integers(1, min(9, w)+1))
        if rects and rng.random() < .28:
            length = rects[int(rng.integers(len(rects)))][2]
        options = set()
        for _ in range(90):
            if not rects or rng.random() < .16:
                r, c = int(rng.integers(h-1)), int(rng.integers(w-length+1))
            else:
                pr, pc, pl = rects[int(rng.integers(len(rects)))]
                side = int(rng.integers(4))
                if side < 2:
                    r = pr-2 if side == 0 else pr+2
                    c = pc + int(rng.integers(1-length, pl))
                else:
                    c = pc-length if side == 2 else pc+pl
                    r = pr + int(rng.integers(-1, 2))
            if 0 <= r <= h-2 and 0 <= c <= w-length and not np.any(mask[r:r+2, c:c+length]):
                options.add((r, c))
        if not options:
            break
        r, c = sorted(options)[int(rng.integers(len(options)))]
        rects.append((r, c, length))
        mask[r:r+2, c:c+length] = True
    if len(rects) < 2:
        return None
    rmin, cmin = min(r for r, c, l in rects), min(c for r, c, l in rects)
    return sorted((r-rmin, c-cmin, l) for r, c, l in rects)


def _ordinary(rng, palette):
    rects = _grow(rng)
    if rects is None:
        return None
    tiles, selected, old = [], [], []
    for r, c, length in rects:
        if rng.random() < .30 and len(tiles) < 11:
            tiles.append(_pattern(rng, int(rng.integers(1, 10)), palette))
        matches = [t for t in old if t.shape[1] == length]
        if matches and rng.random() < .35:
            tile = matches[int(rng.integers(len(matches)))].copy()
        else:
            tile = _pattern(rng, length, palette)
        selected.append(len(tiles))
        tiles.append(tile)
        old.append(tile)
    if rng.random() < .40 and len(tiles) < 14:
        tiles.append(_pattern(rng, int(rng.integers(1, 10)), palette))
    return rects, tiles, selected


def _layout(rng, tiles, test=False):
    """Distribute the ordered stream over aligned lanes with varied gaps."""
    n = len(tiles)
    lengths = [t.shape[1] for t in tiles]
    feasible = []
    for lanes in range(1, min(n, 5)+1):
        # Random consecutive lane populations; no hard per-lane count cap.
        for _ in range(8):
            cuts = sorted(map(int, rng.choice(np.arange(1, n), lanes-1, replace=False)))
            ends = [0] + cuts + [n]
            groups = [lengths[a:b] for a, b in zip(ends, ends[1:])]
            min_w = max(sum(g)+len(g)+1 for g in groups) + 1
            min_h = 3*lanes+2
            if min_w <= 28 and min_h <= 18:
                feasible.append((groups, min_h, min_w))
    if not feasible:
        return None
    if test:
        eligible = [f for f in feasible if len(f[0]) >= 3] or feasible
    else:
        eligible = [f for f in feasible if len(f[0]) >= 2] or feasible
    groups, min_h, min_w = eligible[int(rng.integers(len(eligible)))]
    positions, v = [], 1
    for group in groups:
        u = 1
        for length in group:
            positions.append((v, u, length))
            u += length + 1
        v += 3
    return positions, (min_h, min_w), [len(g) for g in groups]


def _paint_source(tiles, positions, shape, config, roles):
    axis, bottom, right = config
    vacancy, board, origin, guide = roles
    canonical_shape = shape[::-1] if axis else shape
    a = np.full(canonical_shape, vacancy, dtype=np.int8)
    a[0, 0], a[0, 1:], a[1:, 0] = origin, guide, guide
    a[0, 2] = origin
    reverse_thickness = (bottom == right) if axis else (bottom != right)
    for tile, (v, u, length) in zip(tiles, positions):
        a[1+v:3+v, 1+u:1+u+length] = tile[::-1] if reverse_thickness else tile
    if axis:
        a = a.T
    if bottom:
        a = a[::-1]
    if right:
        a = a[:, ::-1]
    return a.copy()


def _build(rng, config, witness=None, test=False):
    for attempt in range(180):
        roles = tuple(map(int, rng.choice(10, 4, replace=False)))
        vacancy, board_color, origin, guide = roles
        pool = [c for c in range(10) if c not in (vacancy, board_color)]
        palette = list(map(int, rng.choice(pool, int(rng.integers(3 if witness is not None else 1, 9)), replace=False)))
        data = _witness(rng, palette, witness) if witness is not None else _ordinary(rng, palette)
        if data is None:
            continue
        rects, tiles, selected = data
        margins = list(map(int, rng.integers(0, 3, 4)))
        top, bottom, left, right = margins
        rects = [(r+top, c+left, l) for r, c, l in rects]
        bh = max(r+2 for r, c, l in rects) + bottom
        bw = max(c+l for r, c, l in rects) + right
        layout = _layout(rng, tiles, test)
        if layout is None:
            continue
        positions, source_min, populations = layout
        sh, sw = source_min[::-1] if config[0] else source_min
        possible = [axis for axis in (0, 1)
                    if (sh+1+bh if axis == 0 else sw+1+bw) <= 30]
        if not possible:
            continue
        join_axis = int(rng.choice(possible))
        if join_axis == 0:
            common = max(sw, bw) + int(rng.integers(min(3, 31-max(sw, bw))))
            sw = bw = common
            slack = 30-sh-1-bh
            sh += int(rng.integers(min(3, slack+1)))
        else:
            common = max(sh, bh) + int(rng.integers(min(3, 31-max(sh, bh))))
            sh = bh = common
            slack = 30-sw-1-bw
            sw += int(rng.integers(min(3, slack+1)))
        # Distribute source slack into margins and gaps while preserving lanes.
        canonical = (sw, sh) if config[0] else (sh, sw)
        extra_v = canonical[0]-source_min[0]
        lane_shifts = sorted(map(int, rng.integers(extra_v+1, size=len(populations))))
        spread, index = [], 0
        for lane, count in enumerate(populations):
            group = positions[index:index+count]
            occupied = sum(p[2] for p in group)
            spare = canonical[1] - 1 - (occupied + count + 1)
            gaps = 1 + rng.multinomial(spare, rng.dirichlet(np.ones(count+1)))
            u = int(gaps[0])
            for j, (v, old_u, length) in enumerate(group):
                spread.append((v+lane_shifts[lane], u, length))
                u += length + int(gaps[j+1])
            index += count
        stencil = np.full((bh, bw), board_color, dtype=np.int8)
        answer = stencil.copy()
        for idx, (r, c, length) in zip(selected, rects):
            stencil[r:r+2, c:c+length] = vacancy
            answer[r:r+2, c:c+length] = tiles[idx]
        if not np.any(stencil == board_color):
            continue
        try:
            plan = _plan(stencil == vacancy, tiles)
        except ValueError:
            continue
        if not np.array_equal(_render(stencil, tiles, plan), answer):
            continue  # an earlier valid subsequence would give a different label
        source = _paint_source(tiles, spread, (sh, sw), config, roles)
        divider = np.full((1, sw) if join_axis == 0 else (sh, 1), origin, dtype=np.int8)
        panels = (source, divider, stencil) if rng.random() < .5 else (stencil, divider, source)
        inp = np.concatenate(panels, axis=join_axis)
        pair = {'input': inp, 'output': answer}
        d = _parse(inp)
        if not np.array_equal(_render(d['board'], d['tiles'], _plan(d['mask'], d['tiles'])), answer):
            continue
        if witness == 0 and not _fast_major_differs(d, answer):
            continue
        return pair
    raise RuntimeError("Bounded construction attempts exhausted")


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _array_key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    configs = [(a, b, c) for a in (0, 1) for b in (False, True) for c in (False, True)]
    first = configs[int(rng.integers(8))]
    second = (1-first[0], not first[1], not first[2])
    # Test with a new conjunction of already demonstrated axis and corner factors.
    held_options = [c for c in configs if c not in (first, second)]
    held = [held_options[int(j)] for j in rng.choice(len(held_options), 2, replace=False)]
    demonstrated = [c for c in configs if c not in held]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            config = ((first, second)[i] if split == 'train' and i < 2
                      else demonstrated[int(rng.integers(len(demonstrated)))] if split == 'train'
                      else held[int(rng.integers(len(held)))])
            witness = i if split == 'train' and i < 2 else None
            for _ in range(20):
                pair = _build(rng, config, witness, test=split == 'test')
                key = _array_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not produce distinct input")
    rng.shuffle(puzzle['train'])
    return puzzle


def _greedy_plan(mask, tiles):
    empty = mask.copy()
    plan, cursor = [], 0
    while np.any(empty):
        r, c = map(int, np.argwhere(empty)[0])
        for i in range(cursor, len(tiles)):
            length = tiles[i].shape[1]
            if r+2 <= mask.shape[0] and c+length <= mask.shape[1] and np.all(empty[r:r+2, c:c+length]):
                empty[r:r+2, c:c+length] = False
                plan.append((i, r, c))
                cursor = i+1
                break
        else:
            return None
    return tuple(plan)


def _order_witness(d, plan):
    """Swapping two same-size stream occurrences changes the ordered answer."""
    expected = _render(d['board'], d['tiles'], plan)
    for i in range(len(d['tiles'])):
        for j in range(i+1, len(d['tiles'])):
            if (d['tiles'][i].shape == d['tiles'][j].shape
                    and not np.array_equal(d['tiles'][i], d['tiles'][j])):
                swapped = list(d['tiles'])
                swapped[i], swapped[j] = swapped[j], swapped[i]
                other = _render(d['board'], swapped, _plan(d['mask'], swapped))
                if not np.array_equal(expected, other):
                    return True
    return False


def _fast_major_differs(d, expected):
    """A global fast-coordinate sort must not explain all demonstrations."""
    axis = d['axis']
    reverse = d['bottom'] if axis else d['right']
    coordinate = (0 if axis else 2) + int(reverse)
    order = sorted(range(len(d['tiles'])), key=lambda i: d['boxes'][i][coordinate], reverse=reverse)
    tiles = [d['tiles'][i] for i in order]
    try:
        alternative = _render(d['board'], tiles, _plan(d['mask'], tiles))
    except ValueError:
        return True
    return not np.array_equal(alternative, expected)


def _cells_only(d):
    """Ablation: keep A's stream but discard B's rectangular footprints."""
    out = d['board'].copy()
    pixels = np.concatenate([t.ravel() for t in d['tiles']])
    out[d['mask']] = pixels[:int(d['mask'].sum())]
    return out


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise AssertionError("Puzzle keys must be train/test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise AssertionError("Splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, teaching = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _array_key(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            d = _parse(inp)
            plan = _plan(d['mask'], d['tiles'])
            assert np.array_equal(_render(d['board'], d['tiles'], plan), out), "Wrong mosaic"
            assert np.all(out[~d['mask']] == d['board_color']), "Board margins changed"
            assert np.all((out[d['mask']] != d['vacancy']) & (out[d['mask']] != d['board_color']))
            used = [i for i, r, c in plan]
            assert used == sorted(set(used)), "Stream occurrences reused or reordered"
            assert sum(d['tiles'][i].size for i in used) == int(d['mask'].sum())
            assert all(np.all(d['mask'][r:r+2, c:c+d['tiles'][i].shape[1]]) for i, r, c in plan)
            if split == 'train':
                teaching.append((d, plan))
    assert {d['axis'] for d, p in teaching} == {0, 1}, "Teach both source axes"
    assert any(_order_witness(d, p) for d, p in teaching), "Order must alter the answer"
    assert any(_greedy_plan(d['mask'], d['tiles']) is None for d, p in teaching), "Teach global packing"
    assert any(any(not np.array_equal(t, t[::-1]) and not np.array_equal(t, t[:, ::-1])
                   for i, r, c in p for t in [d['tiles'][i]]) for d, p in teaching)
    assert any(len(d['tiles']) > len(p) for d, p in teaching), "Teach skipped inventory"
    assert any(len({_array_key(d['tiles'][i]) for i, r, c in p}) < len(p)
               for d, p in teaching), "Teach repeated occurrences"
    assert any(_fast_major_differs(d, _render(d['board'], d['tiles'], p))
               for d, p in teaching), "Teach lane-major traversal"
    assert any(not np.array_equal(_cells_only(d), _render(d['board'], d['tiles'], p))
               for d, p in teaching), "Discarding rectangle geometry must change pixels"
    training_configs = {(d['axis'], d['bottom'], d['right']) for d, p in teaching}
    for pair in puzzle['test']:
        d = _parse(pair['input'])
        assert (d['axis'], d['bottom'], d['right']) not in training_configs, "Test should recombine key factors"


def self_test():
    shapes, layouts, signatures, configs, lengths, inventory, lanes = set(), set(), set(), set(), set(), set(), set()
    joins, vacancies, boards, component_counts, selected_counts, selected_lengths = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_array_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _parse(pair['input'])
                plan = _plan(d['mask'], d['tiles'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                configs.add((d['axis'], d['bottom'], d['right']))
                shapes.add((pair['input'].shape, pair['output'].shape))
                layouts.add((d['mask'].shape, d['mask'].tobytes(), tuple(d['boxes'])))
                lengths.update(t.shape[1] for t in d['tiles'])
                inventory.add(len(d['tiles']))
                lanes.add(tuple(d['populations']))
                joins.add(d['divider_axis'])
                vacancies.add(d['vacancy'])
                boards.add(d['board_color'])
                component_counts.add(len(_components(d['mask'])))
                selected_counts.add(len(plan))
                selected_lengths.update(d['tiles'][i].shape[1] for i, r, c in plan)
    assert len(signatures) == 42 and len(layouts) >= 160 and len(shapes) >= 80
    assert len(configs) == 8 and len(lanes) >= 12
    assert min(lengths) == 1 and max(lengths) == 9
    assert selected_lengths == set(range(1, 10)), "All permitted widths can be selected"
    assert len(inventory) >= 5 and len(selected_counts) >= 5
    assert joins == {0, 1} and vacancies == boards == set(range(10))
    assert 1 in component_counts and max(component_counts) >= 2
    # A locally fitting tile must sometimes be rejected for global feasibility.
    mask = np.zeros((4, 5), dtype=bool)
    mask[:2] = True
    mask[2:, :2] = True
    tiles = [np.full((2, 3), 3), np.full((2, 5), 4), np.full((2, 2), 5), np.full((2, 5), 6)]
    assert [i for i, r, c in _plan(mask, tiles)] == [1, 2]
    assert _greedy_plan(mask, tiles) is None
    assert _plan(np.ones((2, 4), bool), [np.full((2, 2), 3)]*2) == ((0, 0, 0), (1, 0, 2))
    try:
        _plan(np.ones((2, 4), bool), [np.full((2, 2), 3)])
    except ValueError:
        pass
    else:
        raise AssertionError("Finite inventory ignored")
    # Input-only parsing at the 14-occurrence boundary, under all eight keys.
    # These pairs deliberately exceed the sampler's usual selected count.
    tiles = [np.array([[1, 8, 9][i % 3], [1, 8, 9][(i+1) % 3]], dtype=np.int8).reshape(2, 1)
             for i in range(14)]
    positions = [(1+3*(i//7), 1+2*(i % 7), 1) for i in range(14)]
    for axis in (0, 1):
        for bottom in (False, True):
            for right in (False, True):
                source = _paint_source(tiles, positions, (16, 8) if axis else (8, 16),
                                       (axis, bottom, right), (4, 0, 8, 9))
                stencil = np.zeros((16, 8) if axis else (6, 16), dtype=np.int8)
                answer = stencil.copy()
                for i, tile in enumerate(tiles):
                    r, c = 1+2*(i//7), 1+i % 7
                    stencil[r:r+2, c:c+1] = 4
                    answer[r:r+2, c:c+1] = tile
                divider = np.full((16, 1) if axis else (1, 16), 8, dtype=np.int8)
                inp = np.concatenate((source, divider, stencil), axis=1 if axis else 0)
                assert np.array_equal(solve(inp), answer)
    # All legal count combinations, and deterministic repetition.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(4100+nt*10+ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
            validate_generated_puzzle(p)
    p, q = generate_puzzle(72, 2, 1), generate_puzzle(72, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = generate_puzzle(70, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Bad label accepted")
    return {'seeds_checked': 42, 'distinct_puzzles': len(signatures),
            'layouts': len(layouts), 'shape_pairs': len(shapes),
            'key_configurations': len(configs), 'lane_populations': len(lanes),
            'supplied_counts': sorted(inventory), 'selected_counts': sorted(selected_counts),
            'strip_lengths': sorted(lengths), 'selected_strip_lengths': sorted(selected_lengths)}


if __name__ == '__main__':
    print(self_test())
