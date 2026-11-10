"""Coupled shipments: whole rectangular stock units, distributed routed covers.

Only NumPy and the standard library are used. The solver parses visible pixels
and searches the finite stock inventory; it does not consult generation state.
"""
import json

import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grids must have two dimensions in 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid colors must be integers in 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(points, grid=None, neighbors=FOUR):
    """Connected sets, optionally restricted to a single color."""
    unseen, result = set(points), []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = {start}, [start]
        color = None if grid is None else int(grid[start])
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in unseen and (grid is None or int(grid[p]) == color):
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _stock(grid, first, background, routes):
    points = [tuple(map(int, p)) for p in np.argwhere(grid[:first] != background)]
    _require(points, "missing stock")
    result = []
    for cells in _components(points):
        r0 = min(r for r, c in cells)
        r1 = max(r for r, c in cells) + 1
        c0 = min(c for r, c in cells)
        c1 = max(c for r, c in cells) + 1
        _require(len(cells) == (r1-r0)*(c1-c0), "stock units must be filled rectangles")
        _require(0 < c0 and c1 < grid.shape[1], "stock must be laterally inset")
        _require(set(map(int, np.unique(grid[r0:r1, c0:c1]))) <= routes,
                 "every stock color needs a matching dock")
        result.append(dict(cells=cells, box=(r0, c0, r1-r0, c1-c0),
                           components=_components(cells, grid)))
    return result


def _route_tile(grid, tile, lookup, grouped=False):
    """Return destination cells and components' displacements, or reject fit."""
    parts = tile['components']
    if grouped:
        by_color = {}
        for part in parts:
            color = int(grid[min(part)])
            by_color.setdefault(color, set()).update(part)
        parts = list(by_color.values())
    target, shifts = {}, []
    for part in parts:
        color = int(grid[min(part)])
        band = lookup[color]
        bottom = max(r for r, c in part)
        shift = band['bottom'] - bottom
        shifts.append((color, bottom, shift))
        for r, c in part:
            p = r + shift, c
            _require(band['top'] <= p[0] <= band['bottom'] and
                     band['left'] < c < band['right'], "stock fragment does not fit dock")
            _require(p not in target, "a stock tile's routed fragments collide")
            target[p] = band['rail']
    return target, shifts


def _covers(masks, full):
    """Enumerate up to two exact whole-tile subsets. No geometric templates."""
    by_cell = [[] for _ in range(full.bit_length())]
    for i, mask in enumerate(masks):
        bits = mask
        while bits:
            bit = bits & -bits
            by_cell[bit.bit_length()-1].append(i)
            bits ^= bit
    answers, dead = set(), set()

    def visit(covered, chosen):
        if len(answers) >= 2:
            return
        if covered == full:
            answers.add(tuple(sorted(chosen)))
            return
        if covered in dead:
            return
        empty = full ^ covered
        best = None
        while empty:
            bit = empty & -empty
            choices = [i for i in by_cell[bit.bit_length()-1]
                       if not (masks[i] & covered)]
            if not choices:
                dead.add(covered)
                return
            if best is None or len(choices) < len(best):
                best = choices
            empty ^= bit
        before = len(answers)
        for i in best:
            visit(covered | masks[i], chosen + (i,))
            if len(answers) >= 2:
                return
        # The failure cache is geometric: masks are nonempty, so a previously
        # chosen tile cannot be available again without intersecting covered.
        if len(answers) == before and not answers:
            dead.add(covered)

    visit(0, ())
    return sorted(answers)


def _interpret(canonical, first, background, bands):
    lookup = {b['route']: b for b in bands}
    _require(len(lookup) == len(bands) and len(bands) >= 2, "dock routes must be unique")
    tiles = _stock(canonical, first, background, set(lookup))
    holes = set()
    for band in bands:
        missing = {(r, c) for r in range(band['top'], band['bottom']+1)
                   for c in range(band['left']+1, band['right'])
                   if int(canonical[r, c]) == background}
        _require(missing, "every dock must have a request")
        holes.update(missing)
    targets, shifts = [], []
    for tile in tiles:
        target, shift = _route_tile(canonical, tile, lookup)
        targets.append(target)
        shifts.append(shift)
    index = {p: i for i, p in enumerate(sorted(holes))}
    eligible = [i for i, target in enumerate(targets) if set(target) <= holes]
    masks = [sum(1 << index[p] for p in targets[i]) for i in eligible]
    full = (1 << len(holes)) - 1
    covers = _covers(masks, full)
    _require(len(covers) == 1, "requests need exactly one whole-stock cover")
    selected = tuple(eligible[i] for i in covers[0])
    out = canonical.copy()
    for i in selected:
        for p in tiles[i]['cells']:
            out[p] = background
        for p, rail in targets[i].items():
            out[p] = rail
    return dict(canonical=canonical, first=first, background=background,
                bands=bands, tiles=tiles, holes=holes, targets=targets,
                shifts=shifts, eligible=eligible, selected=selected, expected=out)


def _canonical_candidates(grid, background):
    rows = []
    for row in grid:
        nz = np.flatnonzero(row != background)
        if not len(nz):
            rows.append(None)
            continue
        left, right = int(nz[0]), int(nz[-1])
        route, rail = int(row[left+1]) if right > left else -1, int(row[left])
        if (right-left >= 4 and rail == int(row[right]) and
                route != background and route != rail and
                int(row[right-1]) == route and
                np.all((row[left+1:right] == route) | (row[left+1:right] == background))):
            rows.append((left, right, route, rail))
        else:
            rows.append(False)
    for first in range(2, grid.shape[0]):
        if not rows[first] or rows[first-1] is not None or any(x is False for x in rows[first:]):
            continue
        bands, r = [], first
        while r < grid.shape[0]:
            if rows[r] is None:
                r += 1
                continue
            left, right, route, rail = rows[r]
            top = r
            while r+1 < grid.shape[0] and rows[r+1] == rows[top]:
                r += 1
            bands.append(dict(top=top, bottom=r, left=left, right=right,
                              route=route, rail=rail))
            r += 1
        if any(b['top'] <= a['bottom']+1 for a, b in zip(bands, bands[1:])):
            continue
        try:
            yield _interpret(grid, first, background, bands)
        except ValueError:
            continue


def _analyze(input_grid):
    grid = _grid(input_grid)
    candidates = []
    backgrounds = sorted({int(grid[0, 0]), int(grid[0, -1]),
                          int(grid[-1, 0]), int(grid[-1, -1])})
    for turn in range(4):
        canonical = np.rot90(grid, turn)
        for background in backgrounds:
            for a in _canonical_candidates(canonical, background):
                a['turn'] = turn
                a['output'] = np.rot90(a['expected'], -turn).copy()
                candidates.append(a)
    _require(len(candidates) == 1, "input must visibly identify one stock/dock interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer orientation, roles, fragments, whole-stock selection and output."""
    return _analyze(input_grid)['output']


def _make_pattern(rng, routes, max_height, kind=None, min_height=1):
    kind = kind or str(rng.choice(['solid', 'cut', 'frame', 'islands', 'random'],
                                 p=[.30, .30, .16, .16, .08]))
    if len(routes) < 2:
        kind = 'solid'
    if kind in ('frame', 'islands'):
        h = int(rng.integers(max(3, min_height), max_height+1))
        w = int(rng.integers(4 if kind == 'islands' else 3, 8))
    else:
        h, w = int(rng.integers(min_height, max_height+1)), int(rng.integers(1, 7))
    colors = list(map(int, rng.permutation(routes)))
    a = np.full((h, w), colors[0], dtype=np.int8)
    if kind == 'cut':
        vertical = bool(rng.integers(2))
        n = min(len(colors), w if vertical else h, int(rng.integers(2, 4)))
        if n >= 2:
            length = w if vertical else h
            cuts = [0] + sorted(map(int, rng.choice(np.arange(1, length), n-1, replace=False))) + [length]
            for i in range(n):
                if vertical:
                    a[:, cuts[i]:cuts[i+1]] = colors[i]
                else:
                    a[cuts[i]:cuts[i+1], :] = colors[i]
    elif kind == 'frame':
        a[1:-1, 1:-1] = colors[1]
        if len(colors) >= 3 and w >= 5 and rng.random() < .5:
            a[1:-1, w//2+1:-1] = colors[2]
    elif kind == 'islands':
        a.fill(colors[1])
        hl = int(rng.integers(2, h+1))
        hr = int(rng.integers(1, hl))
        wl = int(rng.integers(1, min(2, w-3)+1))
        wr = int(rng.integers(1, min(2, w-wl-1)+1))
        a[:hl, :wl] = colors[0]
        a[:hr, -wr:] = colors[0]
    elif kind == 'random':
        a = rng.choice(colors, size=(h, w)).astype(np.int8)
    return a


def _layout(rng, count, max_height):
    width = int(rng.integers(18, 31))
    stage = int(rng.integers(max(7, max_height+3), 11))
    palette = list(map(int, rng.permutation(10)))
    background, routes = palette[0], palette[1:count+1]
    extras = palette[count+1:]
    heights = [max_height]*count
    gaps = [1]*count
    margin = int(rng.integers(0, 3))
    budget = 30-stage-sum(heights)-sum(gaps)-margin
    if budget < 0:
        return None
    for _ in range(int(rng.integers(budget+1))):
        if rng.random() < .7:
            heights[int(rng.integers(count))] += 1
        else:
            gaps[int(rng.integers(count))] += 1
    grid = np.full((stage+sum(heights)+sum(gaps)+margin, width), background, dtype=np.int8)
    bands, cursor = [], stage
    common = int(rng.choice(extras))
    for height, gap, route in zip(heights, gaps, routes):
        cursor += gap
        rail = common if rng.random() < .4 else int(rng.choice([c for c in palette[1:] if c != route]))
        left, right = int(rng.integers(0, 3)), width-1-int(rng.integers(0, 3))
        band = dict(top=cursor, bottom=cursor+height-1, left=left, right=right,
                    route=route, rail=rail)
        bands.append(band)
        grid[cursor:cursor+height, left:right+1] = route
        grid[cursor:cursor+height, left] = rail
        grid[cursor:cursor+height, right] = rail
        cursor += height
    return grid, bands, stage, background, routes


def _place(rng, grid, pattern, stage, background, lookup, occupied,
           allowed=None, fixed_x=None):
    h, w = pattern.shape
    lo = max(b['left']+2 for b in lookup.values())
    hi = min(b['right']-1 for b in lookup.values()) - w
    xs = [fixed_x] if fixed_x is not None else list(map(int, rng.permutation(np.arange(lo, hi+1))))
    ys = list(map(int, rng.permutation(stage-h+1)))
    for x in xs:
        if x is None or x < lo or x > hi:
            continue
        for y in ys:
            moat = grid[max(0, y-1):min(stage, y+h+1), x-1:x+w+1]
            if np.any(moat != background):
                continue
            # Candidate analysis is local and is transactional until committed.
            mini = grid.copy()
            mini[y:y+h, x:x+w] = pattern
            cells = {(y+r, x+c) for r in range(h) for c in range(w)}
            tile = dict(cells=cells, box=(y, x, h, w), components=_components(cells, mini))
            try:
                target, shifts = _route_tile(mini, tile, lookup)
            except ValueError:
                continue
            if set(target) & occupied or (allowed is not None and not set(target) <= allowed):
                continue
            grid[y:y+h, x:x+w] = pattern
            return tile, target, shifts
    return None


def _attempt(rng, count, max_height, lesson, turn, require_tall=False):
    layout = _layout(rng, count, max_height)
    if layout is None:
        return None
    grid, bands, stage, background, routes = layout
    lookup = {b['route']: b for b in bands}
    occupied, placed, used_routes = set(), [], set()
    if lesson == 'independent':
        kind = 'islands'
    else:
        kind = str(rng.choice(['frame', 'islands', 'cut']))
    first = _make_pattern(rng, routes, max_height, kind)
    if require_tall:
        # Tall test stock is a scale recombination of an already taught motif.
        first = _make_pattern(rng, routes, max_height, 'frame', min_height=4)
    initial = _place(rng, grid, first, stage, background, lookup, occupied)
    if initial is None:
        return None
    tile, target, shift = initial
    if len({color for color, bottom, delta in shift}) < 2:
        return None
    placed.append(initial)
    occupied.update(target)
    used_routes.update(int(grid[min(part)]) for part in tile['components'])
    # Missing destinations get a small block, but its x/y positions are sampled.
    for route in routes:
        if route in used_routes:
            continue
        pattern = _make_pattern(rng, [route], max_height)
        result = _place(rng, grid, pattern, stage, background, lookup, occupied)
        if result is None:
            return None
        placed.append(result)
        occupied.update(result[1])
        used_routes.add(route)
    desired = int(rng.integers(max(2, len(placed)), 6))
    for _ in range(30):
        if len(placed) >= desired:
            break
        pattern = _make_pattern(rng, routes, max_height)
        result = _place(rng, grid, pattern, stage, background, lookup, occupied)
        if result is not None:
            placed.append(result)
            occupied.update(result[1])
    if len(placed) < 2:
        return None
    # Freeze the distributed request silhouette, independent of any decoy stock.
    for p in occupied:
        grid[p] = background
    out = grid.copy()
    for tile, target, shift in placed:
        for p in tile['cells']:
            out[p] = background
        for p, color in target.items():
            out[p] = color
    # Proper crops can form multi-destination sub-footprints. Actual routed masks,
    # not crop dimensions, decide local legality. The first lesson requires one
    # such spare, distinguishing "select all multicolor rectangles" as well.
    want_local = lesson == 'split' or rng.random() < .65
    if want_local:
        y0, x0, th, tw = placed[0][0]['box']
        original = grid[y0:y0+th, x0:x0+tw].copy()
        options = []
        for y in range(th):
            for x in range(tw):
                for h in range(1, th-y+1):
                    for w in range(1, tw-x+1):
                        if h*w == th*tw:
                            continue
                        crop = original[y:y+h, x:x+w].copy()
                        if lesson == 'split' and len(np.unique(crop)) < 2:
                            continue
                        if lesson == 'split' and not any(
                                all(y0+y <= r < y0+y+h and x0+x <= c < x0+x+w
                                    for r, c in part)
                                for part in placed[0][0]['components']):
                            continue
                        options.append((crop, x0+x))
        rng.shuffle(options)
        spare = None
        for pattern, x in options:
            spare = _place(rng, grid, pattern, stage, background, lookup, set(),
                           allowed=occupied, fixed_x=x)
            if spare is not None:
                break
        if spare is None:
            return None
        for p in spare[0]['cells']:
            out[p] = grid[p]
    for _ in range(int(rng.integers(0, 3))):
        pattern = _make_pattern(rng, routes, max_height)
        result = _place(rng, grid, pattern, stage, background, lookup, set())
        if result is not None:
            for p in result[0]['cells']:
                out[p] = grid[p]
    inp, out = np.rot90(grid, turn).copy(), np.rot90(out, turn).copy()
    try:
        a = _analyze(inp)
    except ValueError:
        return None
    if not np.array_equal(out, a['output']):
        return None
    if not _features(a)['independent'] and lesson == 'independent':
        return None
    if not _features(a)['coupled']:
        return None
    if lesson == 'split' and not _features(a)['fragment_exchange']:
        return None
    # Source connectivity never needs an unstated four-versus-eight choice.
    for stock in a['tiles']:
        if len(_components(stock['cells'], a['canonical'], EIGHT)) != len(stock['components']):
            return None
    return {'input': inp, 'output': out}


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                 and low <= int(value) <= high, f'{name} is outside its legal integer range')
    _require(num_train > num_test, 'num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, 'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    turns = list(map(int, rng.permutation(4)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', num_train), ('test', num_test)):
        for index in range(number):
            training = split == 'train'
            lesson = ('split' if index == 0 else 'independent' if index == 1 else None) if training else None
            if regime == 0:
                turn = turns[index % 2] if training else turns[2+index % 2]
            else:
                turn = turns[0] if rng.random() < .7 else turns[index % 2]
            count = 2 + index % 2 if training else (4 if regime == 1 else int(rng.integers(2, 4)))
            max_height = 3 if training or regime == 1 else (5 if regime == 2 else 4)
            pair = None
            for _ in range(180):
                candidate = _attempt(rng, count, max_height, lesson, turn,
                                     require_tall=not training and regime == 2)
                if candidate is not None and _key(candidate['input']) not in seen:
                    pair = candidate
                    break
            _require(pair is not None, 'bounded shipment construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def _features(a):
    selected = set(a['selected'])
    coupled, independent, largest, multi_local_spare = False, False, 0, False
    signatures = []
    for i, tile in enumerate(a['tiles']):
        r, c, h, w = tile['box']
        pattern = a['canonical'][r:r+h, c:c+w]
        role_map = {}
        signatures.append((h, w, tuple(role_map.setdefault(int(v), len(role_map)) for v in pattern.flat)))
        if i not in selected:
            multi_local_spare |= i in a['eligible'] and len(np.unique(pattern)) >= 2
            continue
        largest = max(largest, h)
        coupled |= len({color for color, bottom, shift in a['shifts'][i]}) >= 2
        by_color = {}
        for color, bottom, shift in a['shifts'][i]:
            by_color.setdefault(color, set()).add(bottom)
        independent |= any(len(depths) >= 2 for depths in by_color.values())
    return dict(turn=a['turn'], bands=len(a['bands']), stock=len(a['tiles']),
                selected=len(selected), spare=len(a['tiles'])-len(selected),
                local_spare=len(set(a['eligible'])-selected), coupled=bool(coupled),
                multi_local_spare=bool(multi_local_spare),
                fragment_exchange=_fragment_exchange(a) is not None,
                independent=bool(independent), largest=largest, signatures=signatures,
                shared_rails=len({b['rail'] for b in a['bands']}) < len(a['bands']),
                reused_rails=any(b['rail'] in {x['route'] for x in a['bands']} for b in a['bands']),
                depths=tuple(b['bottom']-b['top']+1 for b in a['bands']),
                shape=tuple(a['canonical'].shape), holes=len(a['holes']),
                request_components=len(_components(a['holes'])),
                background=a['background'])


def _fragment_exchange(a):
    """Find a visible counterexample to treating components as free inventory.

    One fragment in an eligible multicolor spare can replace a selected fragment
    exactly, but whole-spare selection is forbidden by the unique bundle cover.
    Return source masks for that exchange. This is a witness, not a solver.
    """
    grid = a['canonical']
    lookup = {b['route']: b for b in a['bands']}

    def target(part):
        band = lookup[int(grid[min(part)])]
        delta = band['bottom'] - max(r for r, c in part)
        return frozenset((r+delta, c) for r, c in part)

    used = [(part, target(part)) for i in a['selected'] for part in a['tiles'][i]['components']]
    for i in a['eligible']:
        if i in a['selected'] or len({int(grid[min(p)]) for p in a['tiles'][i]['components']}) < 2:
            continue
        for spare_part in a['tiles'][i]['components']:
            dest = target(spare_part)
            for selected_part, selected_target in used:
                if dest == selected_target:
                    return selected_part, spare_part
    return None


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ne = _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), []
    for pair in puzzle['train'] + puzzle['test']:
        _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
        inp, out = _grid(pair['input']), _grid(pair['output'])
        _require(inp.shape == out.shape, 'shipment canvas dimensions must be preserved')
        _require(_key(inp) not in seen, 'duplicate input')
        seen.add(_key(inp))
        a = _analyze(inp)
        _require(np.array_equal(a['output'], out), 'incorrect routed exact-cover output')
        expected, canon = a['expected'], a['canonical']
        modified = set(a['holes']) | set().union(*(a['tiles'][i]['cells'] for i in a['selected']))
        actual = {tuple(map(int, p)) for p in np.argwhere(expected != canon)}
        _require(actual == modified, 'only selected stock and vacancies may change')
        total = sum(len(a['tiles'][i]['cells']) for i in a['selected'])
        _require(total == len(a['holes']), 'shipment cell count must be conserved')
        union = set()
        for i in a['selected']:
            _require(not union & set(a['targets'][i]), 'selected footprints must not overlap')
            union.update(a['targets'][i])
        _require(union == a['holes'], 'selected footprints must cover requests exactly')
        for tile in a['tiles']:
            _require(len(_components(tile['cells'], canon, EIGHT)) == len(tile['components']),
                     'diagonal-only monochromatic contact is excluded by the sampler')
        f = _features(a)
        _require(f['coupled'], 'missing multi-destination coupling')
        analyses.append(a)
    _require(any(_features(a)['fragment_exchange'] for a in analyses[:nt]),
             'training must reject a multicolor spare with an exchangeable fragment')
    _require(any(_features(a)['independent'] for a in analyses[:nt]),
             'training must establish independent same-color component alignment')
    # All blanks are accounted for; protected interior strips and rails survive.
    return True


def self_test():
    """Bounded 42-seed semantic and structural suite, plus small regressions."""
    features, fingerprints = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        features.extend(_features(_analyze(p['input'])) for s in ('train', 'test') for p in puzzle[s])
        a = _analyze(puzzle['train'][0]['input'])
        original, replacement = _fragment_exchange(a)
        # Dropping only whole-tile coupling permits identical dock filling with
        # different, partial source erasures. The physical shipment rule forbids it.
        altered = a['expected'].copy()
        for cell in original:
            altered[cell] = a['canonical'][cell]
        for cell in replacement:
            altered[cell] = a['background']
        assert not np.array_equal(altered, a['expected'])
        assert np.array_equal(altered[a['first']:], a['expected'][a['first']:])
    assert len(fingerprints) == 42
    assert {f['turn'] for f in features} == set(range(4))
    assert {f['bands'] for f in features} == {2, 3, 4}
    assert len({f['shape'] for f in features}) >= 25
    assert len({sig for f in features for sig in f['signatures']}) >= 50
    assert len({f['depths'] for f in features}) >= 20
    assert len({f['stock'] for f in features}) >= 3
    assert len({f['selected'] for f in features}) >= 3
    assert any(f['spare'] == 0 for f in features) and any(f['spare'] >= 2 for f in features)
    assert max(f['largest'] for f in features) >= 4
    assert any(f['shared_rails'] for f in features) and any(f['reused_rails'] for f in features)
    # A geometrically fitting singleton cannot replace a coupled larger shipment.
    assert _covers([0b111, 0b001], 0b111) == [(0,)]
    # The same inventory can demand multiple units, and alternatives are rejected.
    assert _covers([0b001, 0b110], 0b111) == [(0, 1)]
    assert len(_covers([0b111, 0b001, 0b110], 0b111)) == 2
    assert _covers([0b001], 0b111) == []
    # With known docks, the two disconnected red islands bottom-align separately.
    grid = np.zeros((12, 8), dtype=np.int8)
    grid[:3, 2:6] = np.array([[1, 2, 2, 1], [1, 2, 2, 2], [2, 2, 2, 2]])
    lookup = {1: dict(top=5, bottom=7, left=0, right=7, rail=3),
              2: dict(top=9, bottom=11, left=0, right=7, rail=4)}
    cells = {(r, c) for r in range(3) for c in range(2, 6)}
    tile = dict(cells=cells, components=_components(cells, grid))
    target, shifts = _route_tile(grid, tile, lookup)
    grouped, _ = _route_tile(grid, tile, lookup, grouped=True)
    assert {(6, 2), (7, 2), (7, 5)} <= set(target)
    assert target != grouped and len(target) == 12
    # Actual fragment collision is forbidden even if recoloring hides it.
    grid[:3, 2:6] = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1]])
    tile['components'] = _components(cells, grid)
    try:
        _route_tile(grid, tile, lookup)
    except ValueError:
        pass
    else:
        raise AssertionError('self-colliding routed tile was accepted')
    first, second = generate_puzzle(777, 2, 1), generate_puzzle(777, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s])
               for k in ('input', 'output'))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            _counts(a, b)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    wrong = generate_puzzle(888, 2, 1)
    wrong['test'][0]['output'][0, 0] = (int(wrong['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    return {'seeds': 42, 'pairs': len(features), 'shapes': len({f['shape'] for f in features}),
            'tile_mosaics': len({sig for f in features for sig in f['signatures']}),
            'dock_counts': sorted({f['bands'] for f in features}),
            'stock_counts': sorted({f['stock'] for f in features}),
            'selected_counts': sorted({f['selected'] for f in features})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
