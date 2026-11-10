"""Resonant shape voting: an object's orientation supplies its clock period.

No files, parent generators, seeds, or cached puzzles are consulted by solve.
All searches, caches, retries, and self-test measurements are bounded.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import numpy as np


_STEP = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _normal(cells):
    cells = tuple(cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r-y), int(c-x)) for r, c in cells))


def _turn(cells):
    return _normal((c, -r) for r, c in cells)


def _mirror(cells):
    return _normal((r, -c) for r, c in cells)


@lru_cache(maxsize=2048)
def _rotations(cells):
    result, raw = set(), _normal(cells)
    for _ in range(4):
        result.add(raw)
        raw = _turn(raw)
    return tuple(sorted(result))


@lru_cache(maxsize=2048)
def _variants(cells):
    return tuple(sorted(set(_rotations(cells)) | set(_rotations(_mirror(cells)))))


def _key(cells):
    return min(_variants(_normal(cells)))


def _size(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _period(cells, axis=1):
    """Longest uninterrupted run parallel to the time axis, not box extent."""
    points = set(cells)
    dy, dx = (0, 1) if axis == 1 else (1, 0)
    longest = 0
    for r, c in points:
        if (r-dy, c-dx) in points:
            continue
        n = 1
        while (r+n*dy, c+n*dx) in points:
            n += 1
        longest = max(longest, n)
    return longest


def _components(mask, diagonal=False):
    h, w = mask.shape
    pending = set(zip(*np.nonzero(mask)))
    result = []
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if r or c) if diagonal else _STEP
    while pending:
        first = min(pending)
        pending.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                point = (r+dr, c+dc)
                if point in pending:
                    pending.remove(point)
                    stack.append(point)
        result.append(tuple(sorted(cells)))
    return result


def _hole_free(shape):
    h, w = _size(shape)
    mask = np.ones((h+2, w+2), dtype=bool)
    for r, c in shape:
        mask[r+1, c+1] = False
    return len(_components(mask)) == 1


@lru_cache(maxsize=1)
def _small_pool():
    """Enumerate connected free shapes of at most seven cells in a 3x3 box."""
    level, pool = {((0, 0),)}, set()
    for area in range(1, 8):
        pool.update(s for s in level if _hole_free(s))
        if area == 7:
            break
        following = set()
        for shape in sorted(level):
            occupied = set(shape)
            frontier = {(r+dr, c+dc) for r, c in shape for dr, dc in _STEP} - occupied
            for point in sorted(frontier):
                grown = _normal(occupied | {point})
                if max(_size(grown)) <= 3:
                    following.add(_key(grown))
        level = following
    return tuple(sorted(pool))


def _pool(rng):
    shapes = set(_small_pool())
    # Fresh shapes extend the finite small-shape repertoire without files.
    for _ in range(64):
        cells = {(0, 0)}
        for _ in range(int(rng.integers(3, 9))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _STEP} - cells)
            legal = [p for p in frontier if max(_size(_normal(cells | {p}))) <= 4]
            if not legal:
                break
            cells.add(legal[int(rng.integers(len(legal)))])
        raw = _key(cells)
        if _hole_free(raw):
            shapes.add(raw)
    return tuple(sorted(shapes))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _as_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _read(input_grid):
    """Recover the frame, emitting edge, cavities, and whole voting objects."""
    grid = _as_grid(input_grid)
    candidates = []
    for color in np.unique(grid):
        for cells in _components(grid == color):
            if len(cells) < 20:
                continue
            rs, cs = zip(*cells)
            y, x, bottom, right = min(rs), min(cs), max(rs), max(cs)
            box = grid[y:bottom+1, x:right+1]
            if min(box.shape) < 5:
                continue
            if not (np.all(box[0] == color) and np.all(box[-1] == color)
                    and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)):
                continue
            others = set(map(int, np.unique(box))) - {int(color)}
            if len(others) != 1 or np.count_nonzero(grid == color) != len(cells):
                continue
            candidates.append((int(color), others.pop(), (y, x, bottom, right)))
    if len(candidates) != 1:
        raise ValueError("one unambiguous perforated rectangular carrier is required")
    carrier, bg, bbox = candidates[0]
    y, x, bottom, right = bbox
    box = grid[y:bottom+1, x:right+1]
    exterior = grid.copy()
    exterior[y:bottom+1, x:right+1] = bg
    objects = _components(exterior != bg)
    if not objects:
        raise ValueError("missing exterior voters")
    rr, cc = zip(*(p for cells in objects for p in cells))
    sides = [name for name, ok in (("left", max(cc) < x), ("right", min(cc) > right),
             ("top", max(rr) < y), ("bottom", min(rr) > bottom)) if ok]
    if len(sides) != 1:
        raise ValueError("all voters must identify one unique emitting edge")
    side = sides[0]
    axis = 1 if side in ("left", "right") else 0
    votes = []
    for cells in objects:
        colors = {int(grid[r, c]) for r, c in cells}
        if len(colors) != 1 or next(iter(colors)) in (bg, carrier):
            raise ValueError("voters must be separate monochromatic objects")
        votes.append(dict(shape=_key(cells), color=colors.pop(), period=_period(cells, axis),
                          raw=_normal(cells), cells=cells))
    cavities = []
    for cells in _components(box == bg):
        rs, cs = zip(*cells)
        distance = {"left": min(cs), "right": box.shape[1]-1-max(cs),
                    "top": min(rs), "bottom": box.shape[0]-1-max(rs)}[side]
        if distance < 1:
            raise ValueError("cavities cannot touch the carrier border")
        cavities.append(dict(shape=_key(cells), cells=cells, raw=_normal(cells), t=distance-1))
    if not cavities or {v['shape'] for v in votes} != {c['shape'] for c in cavities}:
        raise ValueError("every shape class must have both cavities and voters")
    return dict(grid=grid, box=box, background=bg, carrier=carrier, bbox=bbox,
                side=side, axis=axis, votes=votes, cavities=cavities)


def _signature(shape, equivalence):
    if equivalence == 'area':
        return len(shape)
    if equivalence == 'box':
        return tuple(sorted(_size(shape)))
    if equivalence == 'translation':
        return _normal(shape)
    if equivalence == 'rotation':
        return min(_rotations(_normal(shape)))
    return _key(shape)


def _prediction(data, clock='run', equivalence='free', phase=0, priority='vote'):
    out = data['box'].copy()
    for cavity in data['cavities']:
        target = _signature(cavity['raw'], equivalence)
        counts, active = Counter(), []
        for obj in data['votes']:
            if _signature(obj['raw'], equivalence) != target:
                continue
            p = obj['period']
            if clock == 'box':
                p = _size(obj['raw'])[data['axis']]
            elif clock == 'area':
                p = len(obj['raw'])
            if clock != 'all' and (cavity['t']+phase) % p:
                continue
            counts[obj['color']] += 1
            active.append(obj)
        if not counts:
            continue
        if priority == 'nearest':
            y, x, bottom, right = data['bbox']
            def gap(obj):
                rs, cs = zip(*obj['cells'])
                return {'left': x-max(cs), 'right': min(cs)-right,
                        'top': y-max(rs), 'bottom': min(rs)-bottom}[data['side']]
            winner = min(active, key=gap)['color']
        else:
            high = max(counts.values())
            best = [c for c, n in counts.items() if n == high]
            if len(best) != 1:
                raise ValueError("nonempty active ballots must have a unique plurality")
            winner = best[0]
        for r, c in cavity['cells']:
            out[r, c] = winner
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill each cavity from its locally timed, exact-shape component ballot."""
    return _prediction(_read(input_grid))


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False


def _places(available, raw, fixed_x=None):
    h, w = _size(raw)
    rows, cols = available.shape[0]-h+1, available.shape[1]-w+1
    if rows <= 0 or cols <= 0:
        return []
    xs = range(cols) if fixed_x is None else [fixed_x]
    return [(y, x) for y in range(rows) for x in xs if 0 <= x < cols
            and all(available[y+r, x+c] for r, c in raw)]


def _pack_sources(rng, specs, h, w, forced_nearest=None):
    # Try irregular packing first. A 2-column shelf fallback has enough space
    # for ten objects in 4x4 envelopes, with a full background gap.
    for _ in range(12 if forced_nearest is None else 0):
        available = np.ones((h, w), dtype=bool)
        placed = []
        order = list(map(int, rng.permutation(len(specs))))
        order.sort(key=lambda i: len(specs[i]['raw']), reverse=True)
        for i in order:
            positions = _places(available, specs[i]['raw'])
            if not positions:
                break
            y, x = _pick(rng, positions)
            cells = tuple((y+r, x+c) for r, c in specs[i]['raw'])
            _block(available, cells)
            placed.append((i, cells))
        if len(placed) == len(specs):
            return placed
    placed = []
    order = list(map(int, rng.permutation(len(specs))))
    if forced_nearest is not None:
        order.remove(forced_nearest)
        order.insert(1, forced_nearest)
    offset_y = int(rng.integers(h-(5*((len(specs)+1)//2)-1)+1))
    offset_x = int(rng.integers(w-9+1))
    for slot, i in enumerate(order):
        sh, sw = _size(specs[i]['raw'])
        if forced_nearest is None:
            y = offset_y+5*(slot//2)+int(rng.integers(5-sh))
            x = offset_x+5*(slot % 2)+int(rng.integers(5-sw))
        else:
            # Teaching forms all have 3x3 boxes. Put one losing anchor vote
            # uniquely closest to the frame; other boxes end one cell earlier.
            assert (sh, sw) == (3, 3)
            y = offset_y+5*(slot//2)
            x = w-3 if i == forced_nearest else offset_x+5*(slot % 2)
        placed.append((i, tuple((y+r, x+c) for r, c in specs[i]['raw'])))
    return placed


def _pack_cavities(rng, goals, h, w, teaching):
    # Fixed leading coordinates are essential: moving a hole changes its time.
    # Optional goals may be skipped; every class has a mandatory first goal.
    for attempt in range(32):
        available = np.ones((h, w), dtype=bool)
        available[[0, -1], :] = False
        available[:, [0, -1]] = False
        placed, failed = [], False
        order = list(range(len(goals)))
        if attempt < 31:
            rng.shuffle(order)
            order.sort(key=lambda i: not goals[i]['mandatory'])
        else:
            order.sort(key=lambda i: (goals[i]['t'], i))
        for i in order:
            goal = goals[i]
            variants = (goal['raw'],) if goal.get('raw') is not None else _variants(goal['shape'])
            options = [(raw, y, x) for raw in variants
                       for y, x in _places(available, raw, goal['t']+1)]
            if not options:
                if goal['mandatory'] or teaching:
                    failed = True
                    break
                continue
            raw, y, x = _pick(rng, options) if attempt < 31 else options[0]
            cells = tuple((y+r, x+c) for r, c in raw)
            _block(available, cells)
            placed.append((goal, cells))
        if not failed:
            return placed
    return None


def _oriented(rng, shape, period):
    return _pick(rng, [v for v in _variants(shape) if _period(v) == period])


def _teaching_ballots(rng, anchor, companion, colors, dominant):
    specs = []
    other = 5-dominant  # The two periods are 2 and 3.
    raw = _oriented(rng, anchor, dominant)
    # Opposite-handed winner, same-handed loser: rotation-only matching fails
    # at t=dominant, where the other-period votes are inactive.
    for color, count, form in ((colors[dominant-2], 3, _mirror(raw)),
                               (colors[other-2], 2, _turn(raw)), (colors[2], 2, raw)):
        specs.extend(dict(shape=anchor, color=color, raw=form, period=_period(form))
                     for _ in range(count))
    cp = int(rng.integers(2, 4))
    for p, color, count in ((cp, colors[3], 2), (5-cp, colors[4], 1)):
        for _ in range(count):
            form = _oriented(rng, companion, p)
            specs.append(dict(shape=companion, color=color, raw=form, period=p))
    return specs, raw


def _ordinary_ballots(rng, shapes, colors):
    specs = []
    # At most ten votes. Each color has one period and distinct weights, so
    # every active subset has a unique winner; no hidden tie breaker exists.
    budget = 10
    for index, shape in enumerate(shapes):
        remaining_classes = len(shapes)-index-1
        room = budget-remaining_classes
        weights = _pick(rng, [(1,), (2, 1), (3, 1), (2,)])
        if sum(weights) > room:
            weights = (1,)
        periods = sorted({_period(v) for v in _variants(shape)})
        palette = list(map(int, rng.choice(colors, len(weights), replace=False)))
        for color, count in zip(palette, weights):
            p = int(_pick(rng, periods))
            for _ in range(count):
                form = _oriented(rng, shape, p)
                specs.append(dict(shape=shape, color=color, raw=form, period=p))
        budget -= sum(weights)
    return specs


def _construct(rng, shapes, specs, roles, rotation, teaching=False, anchor_raw=None,
               lesson=0, test=False):
    bg, carrier = roles
    n = len(specs)
    min_h = max(14, 5*((n+1)//2)+1, 26 if teaching else 14)
    for attempt in range(24):
        h = int(rng.integers(min_h, 31)) if attempt < 20 else 30
        source_w = int(rng.integers(10, 13)) if not test else 10
        frame_w = int(rng.integers(14, 31-source_w)) if not test else 20
        goals = []
        if teaching:
            times = [0, 1, 2, 3, 6, 9] if lesson == 0 else [0, 1, 2, 3, 4, 8]
            for t in times:
                goals.append(dict(shape=shapes[0], t=t, mandatory=True,
                                  raw=anchor_raw if t == _period(anchor_raw) else None))
            for t in (0, 3):
                goals.append(dict(shape=shapes[1], t=t, mandatory=True, raw=None))
        else:
            for index, shape in enumerate(shapes):
                active_periods = [s['period'] for s in specs if s['shape'] == shape]
                # Reserve one supported cavity per class, and a genuinely
                # later test station (all training stations are at most nine).
                upper = min(frame_w-2-max(_size(v)[1] for v in _variants(shape)),
                            9 if not test else 99)
                candidates = [t for t in range(upper+1) if any(t % p == 0 for p in active_periods)]
                if test and index == 0:
                    candidates = [t for t in candidates if t > 9]
                if not candidates:
                    break
                goals.append(dict(shape=shape, t=int(_pick(rng, candidates)), mandatory=True, raw=None))
            if len(goals) != len(shapes):
                continue
            extras = int(rng.integers(1, 7))
            for _ in range(extras):
                shape = _pick(rng, shapes)
                upper = min(frame_w-2-max(_size(v)[1] for v in _variants(shape)),
                            9 if not test else 99)
                goals.append(dict(shape=shape, t=int(rng.integers(upper+1)), mandatory=False, raw=None))
        placements = _pack_cavities(rng, goals, h, frame_w, teaching)
        if placements is None:
            continue
        inp = np.full((h, source_w+frame_w), bg, dtype=np.int8)
        frame = np.full((h, frame_w), carrier, dtype=np.int8)
        out = frame.copy()
        # Ground truth is built from sampled objects and hole entry times.
        # The public solver instead recovers components from the final grid.
        for goal, cells in placements:
            ballot = Counter(s['color'] for s in specs if s['shape'] == goal['shape']
                             and goal['t'] % s['period'] == 0)
            winner = bg if not ballot else ballot.most_common(1)[0][0]
            if ballot:
                values = sorted(ballot.values(), reverse=True)
                assert len(values) == 1 or values[0] > values[1]
            for r, c in cells:
                frame[r, c], out[r, c] = bg, winner
        inp[:, source_w:] = frame
        for i, cells in _pack_sources(rng, specs, h, source_w-1,
                                     5 if teaching and attempt >= 20 else None):
            for r, c in cells:
                inp[r, c] = specs[i]['color']
        pair = {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
        if teaching and not _alternative_fails(pair, priority='nearest'):
            # Counting must visibly differ from A-style proximity overwriting.
            # Resample placement, never relabel a ballot or change the rule.
            continue
        return pair
    raise RuntimeError("bounded cavity packing exhausted")


def _counts(num_train, num_test):
    for name, n, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= n <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _grid_key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    pool = _pool(rng)
    eligible = [s for s in _small_pool() if _size(s) == (3, 3)
                and {_period(v) for v in _rotations(s)} == {2, 3}
                and _mirror(s) not in _rotations(s)]
    eligible = [s for s in eligible if any(t != s and len(t) == len(s) and _size(t) == (3, 3)
                and {_period(v) for v in _rotations(t)} == {2, 3} for t in _small_pool())]
    anchor = _pick(rng, eligible)
    companion = _pick(rng, [t for t in _small_pool() if t != anchor and len(t) == len(anchor)
                          and _size(t) == (3, 3) and {_period(v) for v in _rotations(t)} == {2, 3}])
    colors = list(map(int, rng.choice(10, 5, replace=False)))
    first_p, base = int(rng.integers(2, 4)), int(rng.integers(4))
    puzzle, seen, known = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            teaching = split == 'train' and index < 2
            for retry in range(8):
                if teaching:
                    shapes = [anchor, companion]
                    specs, raw = _teaching_ballots(rng, anchor, companion, colors,
                                                  first_p if index == 0 else 5-first_p)
                    role_choices = [c for c in range(10) if c not in colors]
                    roles = list(map(int, rng.choice(role_choices, 2, replace=False)))
                    rotation = (base+index) % 4
                else:
                    k = int(rng.integers(1, 4))
                    available = [s for s in pool if (max(_size(s)) <= 3 if split == 'train' else True)]
                    shapes = [_pick(rng, available)]
                    if split == 'test':
                        shapes[0] = _pick(rng, [s for s in available if s not in known])
                    others = [s for s in available if s not in shapes]
                    shapes += [others[int(i)] for i in rng.choice(len(others), k-1, replace=False)]
                    palette = list(map(int, rng.permutation(10)))
                    roles = palette[:2]
                    specs = _ordinary_ballots(rng, shapes, palette[2:])
                    raw = None
                    rotation = int(rng.integers(4))
                pair = _construct(rng, shapes, specs, roles, rotation, teaching, raw,
                                  index, split == 'test')
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train':
                        known.update(shapes)
                    break
            else:
                raise RuntimeError("bounded distinct-input retries exhausted")
    return puzzle


def _alternative_fails(pair, **options):
    try:
        result = _prediction(_read(pair['input']), **options)
    except ValueError:
        return True
    return not np.array_equal(result, pair['output'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _as_grid(pair['input']), _as_grid(pair['output'])
            key = _grid_key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            data = _read(inp)
            expected = _prediction(data)
            assert np.array_equal(out, expected), 'wrong clock, vote, fill, or crop'
            assert np.any(out != data['box']), 'at least one supported cavity must fill'
            assert np.all(out[data['box'] == data['carrier']] == data['carrier'])
            assert len(_components(data['box'] == data['carrier'])) == 1
            # Separate components even when colors differ, including diagonals.
            exterior = inp.copy()
            y, x, bottom, right = data['bbox']
            exterior[y:bottom+1, x:right+1] = data['background']
            assert len(_components(exterior != data['background'], True)) == len(data['votes'])
            assert len(_components(data['box'] == data['background'], True)) == len(data['cavities'])
            for cavity in data['cavities']:
                values = {int(out[r, c]) for r, c in cavity['cells']}
                assert len(values) == 1, 'a cavity must be uniformly filled or uniformly silent'
            records[split].append(data)
    # These are returned-grid teaching checks, not assumptions about RNG plans.
    for options in ({'clock': 'all'}, {'clock': 'box'}, {'clock': 'area'}, {'phase': 1},
                    {'equivalence': 'area'}, {'equivalence': 'box'},
                    {'equivalence': 'translation'}, {'equivalence': 'rotation'},
                    {'priority': 'nearest'}):
        assert any(_alternative_fails(pair, **options) for pair in puzzle['train']), options
    history, changing, silence, nonmajority = defaultdict(set), False, False, False
    for data, pair in zip(records['train'], puzzle['train']):
        local = defaultdict(set)
        for cavity in data['cavities']:
            r, c = cavity['cells'][0]
            color = int(np.asarray(pair['output'])[r, c])
            silence |= color == data['background']
            if color != data['background']:
                local[cavity['shape']].add(color)
                if cavity['t'] == 0:
                    history[cavity['shape']].add(color)
                ballot = Counter(v['color'] for v in data['votes'] if v['shape'] == cavity['shape']
                                 and cavity['t'] % v['period'] == 0)
                nonmajority |= max(ballot.values())*2 <= sum(ballot.values())
        changing |= any(len(colors) >= 2 for colors in local.values())
    assert silence and changing and nonmajority
    assert any(len(colors) >= 2 for colors in history.values()), 'reject fixed shape-color memory'
    known = {v['shape'] for data in records['train'] for v in data['votes']}
    max_time = max(c['t'] for data in records['train'] for c in data['cavities'])
    for data in records['test']:
        assert any(c['t'] > max_time for c in data['cavities']), 'actual temporal extrapolation required'
        assert any(v['shape'] not in known for v in data['votes']), 'complete local ballots for new shapes'
    return None


def self_test():
    """Stream 42 seeds; retain hashes and feature sets, never puzzle archives."""
    # Hand-calculated regression, independent of generation and election code.
    # F has a 3x3 box but longest horizontal run 2; its quarter-turn has run 3.
    f = ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1))
    turned_f = ((0, 2), (1, 0), (1, 1), (1, 2), (2, 1))
    fixture = np.zeros((18, 21), dtype=np.int8)
    fixture[:, 9:] = 6
    for raw, y, x, color in ((f, 1, 0, 4), (f, 6, 4, 4), (turned_f, 12, 0, 8)):
        for r, c in raw:
            fixture[y+r, x+c] = color
    expected = np.full((18, 12), 6, dtype=np.int8)
    # Entry times 0,1,2,3,6 have active ballots 2:1, empty, 2:0, 0:1, 2:1.
    for y, x, color in ((1, 1, 4), (5, 2, 0), (9, 3, 4), (13, 4, 8), (1, 7, 4)):
        for r, c in f:
            fixture[y+r, 9+x+c] = 0
            expected[y+r, x+c] = color
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(fixture, turn)), np.rot90(expected, turn))
    assert np.array_equal(solve(np.fliplr(fixture)), np.fliplr(expected))
    assert np.array_equal(solve(np.pad(fixture, 1, constant_values=0)), expected)
    tied = fixture.copy()
    for r, c in f:
        tied[1+r, c] = 8
    try:
        solve(tied)  # At time 2, the two period-2 colors tie one against one.
    except ValueError:
        pass
    else:
        raise AssertionError('nonempty tied cavity ballot accepted')
    packed = _pack_sources(np.random.default_rng(123), [dict(raw=f) for _ in range(10)], 26, 9, 5)
    mask = np.zeros((26, 9), dtype=bool)
    for i, cells in packed:
        assert max(c for r, c in cells) == 8 if i == 5 else max(c for r, c in cells) <= 7
        for r, c in cells:
            mask[r, c] = True
    assert len(_components(mask, True)) == 10, 'fallback must keep objects separated'
    moved = fixture.copy()
    moved[:, :9] = 0
    for raw, y, x, color in ((f, 1, 4, 4), (f, 6, 0, 4), (turned_f, 12, 4, 8)):
        for r, c in raw:
            moved[y+r, x+c] = color
    assert np.array_equal(solve(moved), expected), 'source distance must not shift phase'
    shapes, outputs, vocabulary, periods, roles, counts, signatures = set(), set(), set(), set(), set(), set(), set()
    directions, silent_pairs, dynamic_pairs, farther, total = set(), 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _read(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                digest.update(np.asarray(pair['input']).tobytes())
                shapes.add(pair['input'].shape)
                outputs.add(pair['output'].shape)
                vocabulary.update(v['shape'] for v in data['votes'])
                periods.update(v['period'] for v in data['votes'])
                roles.add((data['background'], data['carrier']))
                counts.add((len(data['votes']), len(data['cavities'])))
                directions.add(data['side'])
                local = defaultdict(set)
                silent = False
                for cavity in data['cavities']:
                    r, c = cavity['cells'][0]
                    color = int(pair['output'][r, c])
                    silent |= color == data['background']
                    local[cavity['shape']].add(color)
                    farther += int(cavity['t'] > 9)
                silent_pairs += int(silent)
                dynamic_pairs += int(any(len(v) >= 2 for v in local.values()))
                total += 1
        signatures.add(digest.digest())
        if seed == 0:
            # Geometry and palette metamorphisms exercise the input-only rule.
            pair = puzzle['train'][0]
            for turn in range(4):
                assert np.array_equal(solve(np.rot90(pair['input'], turn)), np.rot90(pair['output'], turn))
            mapping = np.array([9, 0, 8, 1, 7, 2, 6, 3, 5, 4], dtype=np.int8)
            assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
            broken = {s: [{r: g.copy() for r, g in p.items()} for p in puzzle[s]] for s in puzzle}
            broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
            try:
                validate_generated_puzzle(broken)
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError('mutated carrier accepted')
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(p[r], q[r]) for s in puzzle
                       for p, q in zip(puzzle[s], repeat[s]) for r in p)
            del repeat, broken
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(outputs) >= 25
    assert len(vocabulary) >= 35 and {1, 2, 3, 4} <= periods
    assert len(roles) >= 25 and len(counts) >= 25
    assert directions == {'left', 'right', 'top', 'bottom'}
    assert silent_pairs >= 84 and dynamic_pairs >= 84 and farther >= 42
    del puzzle, pair, data
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1234, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=total, input_shapes=len(shapes), output_shapes=len(outputs),
                shape_classes=len(vocabulary), periods=sorted(periods), role_pairs=len(roles),
                count_combinations=len(counts), silent_pairs=silent_pairs,
                changing_class_pairs=dynamic_pairs, later_cavities=farther)
