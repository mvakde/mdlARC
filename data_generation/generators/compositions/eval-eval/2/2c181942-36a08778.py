"""Contour-driven recruitment: color-matched assembly changes ray reachability.

Only standard Python and NumPy are used.  The solver recognizes all conventions
from one input; generation metadata never enters prediction.
"""
from __future__ import annotations

import numpy as np


N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _ensure(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    _ensure(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _ensure(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _ensure(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    todo = [cells.pop()]
    while todo:
        r, c = todo.pop()
        for dr, dc in N8:
            p = (r + dr, c + dc)
            if p in cells:
                cells.remove(p)
                todo.append(p)
    return not cells


def _canonical(mask):
    """Neck is the centered pair on the UPSTREAM extreme (top in this frame)."""
    choices = {}
    for t in range(4):
        m = np.rot90(mask, t).copy()
        h, w = m.shape
        if h < 2 or w < 2 or w % 2:
            continue
        edge = np.zeros(w, dtype=bool)
        edge[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(m[0], edge):
            choices.setdefault(_key(m), (m, []))[1].append(t)
    _ensure(_connected(mask) and len(choices) == 1, "ambiguous or missing attachment")
    return next(iter(choices.values()))


def _components(a, bg, guide):
    remaining = set(map(tuple, np.argwhere((a != bg) & (a != guide))))
    objects = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], [start]
        color = int(a[start])
        while todo:
            r, c = todo.pop()
            for dr, dc in N8:
                p = (r + dr, c + dc)
                if p in remaining and int(a[p]) == color:
                    remaining.remove(p)
                    todo.append(p)
                    cells.append(p)
        coords = np.array(cells, dtype=int)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        mask = np.zeros(hi - lo + 1, dtype=bool)
        mask[tuple((coords - lo).T)] = True
        objects.append(dict(color=color, coords=coords, mask=mask,
                            row=int(lo[0]), col=int(lo[1])))
    return objects


def _recognize(input_grid):
    original = _grid(input_grid)
    counts = np.bincount(original.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _ensure(counts[bg] > original.size / 2, "background must be a strict majority")
    candidates = []
    for turn in range(4):
        a = np.rot90(original, turn).copy()
        h, w = a.shape
        boundary = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
        colors = set(map(int, boundary)) - {bg}
        if len(colors) != 1:
            continue
        guide = colors.pop()
        entries = list(map(int, np.flatnonzero(a[0] == guide)))
        if not entries or any(b - x < 2 for x, b in zip(entries, entries[1:])):
            continue
        expected = np.zeros(a.shape, dtype=bool)
        lengths = []
        for c in entries:
            n = 0
            while n < h and a[n, c] == guide:
                n += 1
            if not 2 <= n < h:
                break
            expected[:n, c] = True
            lengths.append(n)
        else:
            if np.array_equal(expected, a == guide):
                candidates.append((turn, a, guide, entries, lengths))
    _ensure(len(candidates) == 1, "entry direction must be unambiguous")
    turn, a, guide, entries, lengths = candidates[0]
    objects = _components(a, bg, guide)
    stocks, stations = {}, {}
    occupied = set(map(tuple, np.argwhere(a == guide)))
    for obj in objects:
        coords, color, mask = obj['coords'], obj['color'], obj['mask']
        _ensure(np.all(coords > 0) and np.all(coords < np.array(a.shape) - 1),
                "only guide stubs may touch the input boundary")
        cells = set(map(tuple, coords))
        _ensure(all((r + dr, c + dc) not in occupied
                    for r, c in cells for dr, dc in N8 + ((0, 0),)),
                "input objects must have an eight-neighbor gap")
        occupied |= cells
        if mask.shape == (1, 2) and mask.all():
            _ensure(color not in stations, "repeated socket label")
            stations[color] = obj
        else:
            _ensure(len(coords) >= 4 and color not in stocks, "invalid inventory")
            stocks[color] = obj
    _ensure(stations and set(stations) <= set(stocks), "every socket needs one piece")
    footprints = set()
    for color, site in stations.items():
        mask, turns = _canonical(stocks[color]['mask'])
        offset = np.array((site['row'], site['col'] - mask.shape[1] // 2 + 1))
        coords = np.argwhere(mask) + offset
        _ensure(np.all(coords >= 0) and np.all(coords < np.array(a.shape)),
                "docked piece would be clipped")
        cells = set(map(tuple, coords))
        _ensure(all((r + dr, c + dc) not in footprints
                    for r, c in cells for dr, dc in N8 + ((0, 0),)),
                "hypothetical assemblies must have an eight-neighbor gap")
        _ensure(not any(a[r, c] == guide for r, c in cells), "piece overlaps entry")
        footprints |= cells
        site.update(canonical=mask, target=coords, turns=turns)
    return dict(a=a, turn=turn, bg=bg, guide=guide, entries=entries,
                stub_lengths=lengths, stocks=stocks, stations=stations)


def _step(board, row, active, bg, guide):
    """All collisions at a layer are simultaneous; runs use material, not tracks."""
    h, w = board.shape
    runs, following = {}, set()
    for c in sorted(active):
        if int(board[row, c]) in (bg, guide):
            following.add(c)
            continue
        lo = hi = c
        while lo > 0 and int(board[row, lo - 1]) not in (bg, guide):
            lo -= 1
        while hi + 1 < w and int(board[row, hi + 1]) not in (bg, guide):
            hi += 1
        runs.setdefault((lo, hi), []).append(c)
    for lo, hi in runs:
        following.update(c for c in (lo - 1, hi + 1) if 0 <= c < w)
    return runs, following


def _simulate(info, contour=True, activate_all=False):
    """Dynamic assembly and routing, plus two explicit counterfactual modes."""
    a, bg, guide = info['a'], info['bg'], info['guide']
    board = np.full_like(a, bg)
    board[a == guide] = guide
    stations = info['stations']
    for site in stations.values():
        board[site['row'], site['col']:site['col'] + 2] = site['color']
    output = board.copy()
    activated, events, incoming = set(), [], []
    if activate_all:
        for color, site in stations.items():
            board[tuple(site['target'].T)] = color
            output[tuple(site['target'].T)] = color
            activated.add(color)
    active = set(info['entries'])
    for row in range(1, a.shape[0]):
        incoming.append((row, tuple(sorted(active))))
        for color, site in stations.items():
            if (color not in activated and site['row'] == row
                    and active & {site['col'], site['col'] + 1}):
                activated.add(color)
                if contour:
                    board[tuple(site['target'].T)] = color
                    output[tuple(site['target'].T)] = color
        runs, following = _step(board, row, active, bg, guide)
        for c in following:
            if board[row, c] == bg:
                output[row, c] = guide
        for (lo, hi), incident in runs.items():
            events.append((row, lo, hi, tuple(incident)))
            for c in range(max(0, lo - 1), min(a.shape[1], hi + 2)):
                if board[row - 1, c] == bg:
                    output[row - 1, c] = guide
        active = following
    if not contour:
        # Wrong model: trace on bare sockets first, then paste recruited pieces.
        for color in activated:
            output[tuple(stations[color]['target'].T)] = color
    stats = dict(activated=activated, events=events, incoming=incoming,
                 inactive=set(stations) - activated)
    return output, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the answer using only the supplied input."""
    info = _recognize(input_grid)
    output, _ = _simulate(info)
    return np.rot90(output, -info['turn']).copy()


def _mask(rng, force_wide=False, small=False):
    for _ in range(100):
        w = int(rng.choice((4, 6) if force_wide else (2, 4, 6, 8),
                           p=None if force_wide else (.15, .43, .32, .10)))
        d = int(rng.integers(2, 5 if small else 6))
        m = np.zeros((d, w), dtype=bool)
        center = w // 2
        m[:, center - 1:center + 1] = True
        stem = int(rng.integers(1, d))
        kind = int(rng.integers(4))
        if w > 2 and kind == 3 and d >= w // 2:
            # Diagonal wings are one eight-connected silhouette.
            stem = d - w // 2 + 1
            m[stem:] = False
            for r in range(stem, d):
                reach = r - stem + 1
                m[r, center - 1 - reach] = m[r, center + reach] = True
        elif w > 2:
            for r in range(stem, d):
                left = 0 if kind == 0 else int(rng.integers(center))
                right = w if kind == 0 else int(rng.integers(center + 1, w + 1))
                m[r, left:right] = True
                if kind == 2:
                    for c in range(w):
                        if c not in (center - 1, center) and rng.random() < .18:
                            m[r, c] = False
            m[int(rng.integers(stem, d))] = True
        # Require a tight bounding box; reject competing geometric attachments.
        if not m[:, 0].any() or not m[:, -1].any():
            continue
        try:
            canonical, turns = _canonical(m)
        except AssertionError:
            continue
        if not np.array_equal(canonical, m) or (force_wide and len(turns) != 1):
            continue
        return m
    return np.array([[0, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)


def _coords(mask, row, left):
    return np.argwhere(mask) + (row, left - mask.shape[1] // 2 + 1)


def _blocked_array(shape, occupied):
    blocked = np.zeros(shape, dtype=bool)
    for r, c in occupied:
        blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    return blocked


def _legal_lefts(mask, row, blocked):
    """Vectorized geometric packing; returns socket left endpoints."""
    h, w = blocked.shape
    if row < 0 or row + mask.shape[0] > h:
        return []
    center = mask.shape[1] // 2
    lefts = np.arange(max(1, center - 1), min(w - 3, w - center - 1) + 1)
    local = np.argwhere(mask)
    rr = row + local[:, 0]
    cc = lefts[:, None] + local[:, 1] - center + 1
    good = ~np.any(blocked[rr, cc], axis=1)
    return list(map(int, lefts[good]))


def _frontier(board, entries, row):
    active = set(entries)
    for r in range(1, row):
        _, active = _step(board, r, active, 0, 1)
    return active


def _place_inventory(rng, grid, pieces):
    bg = 0
    blocked = _blocked_array(grid.shape, set(map(tuple, np.argwhere(grid != bg))))
    order = list(rng.permutation(len(pieces)))
    order.sort(key=lambda i: -pieces[i][1].size)
    for i in order:
        color, mask, turn = pieces[i]
        mask = np.rot90(mask, turn)
        local = np.argwhere(mask)
        h, w = mask.shape
        tops, lefts = np.indices((grid.shape[0] - h - 1, grid.shape[1] - w - 1))
        rr = tops.ravel()[:, None] + 1 + local[:, 0]
        cc = lefts.ravel()[:, None] + 1 + local[:, 1]
        options = np.flatnonzero(~np.any(blocked[rr, cc], axis=1))
        if not len(options):
            return None
        index = int(rng.choice(options))
        coords = np.column_stack((rr[index], cc[index]))
        grid[tuple(coords.T)] = color
        for y, x in coords:
            blocked[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
    return grid


def _construct(rng, scenario, palette, rotation, compact=False):
    """Plan causal events; then discard the plan and solve the completed input."""
    if scenario == 'chain':
        groups, inert, unmatched = [1, 1], 1, 1
    elif scenario == 'parallel':
        groups, inert, unmatched = [2], 1, int(rng.integers(2))
    elif scenario == 'combine':
        groups, inert, unmatched = [2, 1], int(rng.integers(1, 3)), int(rng.integers(2))
    elif scenario == 'cascade':
        groups, inert, unmatched = [1, 1, 1], int(rng.integers(1, 3)), int(rng.integers(2))
    else:
        groups = [int(rng.integers(1, 3))]
        if rng.random() < .5 and groups[0] == 1:
            groups.append(1)
        inert, unmatched = int(rng.integers(3)), int(rng.integers(2))
    masks = [[_mask(rng, force_wide=j == 0, small=compact) for _ in range(n)]
             for j, n in enumerate(groups)]
    first = int(rng.integers(3, 6))
    starts, end = [], first
    for layer in masks:
        starts.append(end)
        end += max(m.shape[0] for m in layer) + int(rng.integers(2, 4))
    h = max(end, int(rng.integers(18, 29)))
    w = int(rng.integers(18, 31))
    if h > 30:
        return None
    board, bare = np.zeros((h, w), dtype=np.int8), np.zeros((h, w), dtype=np.int8)
    occupied, sites, entries = set(), [], []
    # Pick first sites before choosing entry columns, so impacts vary across necks.
    for j, (row, layer) in enumerate(zip(starts, masks)):
        for mask in layer:
            if j:
                full_active = _frontier(board, entries, row)
                bare_active = _frontier(bare, entries, row)
            options = []
            blocked = _blocked_array((h, w), occupied)
            for left in _legal_lefts(mask, row, blocked):
                coords = _coords(mask, row, left)
                if j and not (full_active & {left, left + 1}):
                    continue
                if j == 1 and scenario in ('chain', 'cascade', 'combine'):
                    if bare_active & {left, left + 1}:
                        continue
                options.append((left, coords))
            if not options:
                return None
            left, coords = options[int(rng.integers(len(options)))]
            occupied |= set(map(tuple, coords))
            board[tuple(coords.T)] = 2
            bare[row, left:left + 2] = 2
            sites.append((row, left, mask, True))
            if j == 0:
                entries.append(left + int(rng.integers(2)))
        if j == 0:
            if any(abs(a - b) < 2 for a in entries for b in entries if a != b):
                return None
            for c in entries:
                n = int(rng.integers(2, min(first, 4) + 1))
                board[:n, c] = bare[:n, c] = 1
    for _ in range(inert):
        mask = _mask(rng, small=compact)
        options = []
        blocked = _blocked_array((h, w), occupied | set(map(tuple, np.argwhere(board == 1))))
        for row in range(2, h - mask.shape[0] + 1):
            active = _frontier(board, entries, row)
            for left in _legal_lefts(mask, row, blocked):
                if active & {left, left + 1}:
                    continue
                coords = _coords(mask, row, left)
                options.append((row, left, coords))
        if not options:
            return None
        row, left, coords = options[int(rng.integers(len(options)))]
        occupied |= set(map(tuple, coords))
        board[row, left:left + 2] = bare[row, left:left + 2] = 2
        sites.append((row, left, mask, False))
    if len(sites) + unmatched > 7:
        return None
    grid = np.zeros((h, w), dtype=np.int8)
    grid[board == 1] = 1
    pieces = []
    for i, (row, left, mask, reached) in enumerate(sites):
        color = i + 2
        grid[row, left:left + 2] = color
        turn = int(rng.integers(4))
        if i == 0 and scenario in ('chain', 'parallel'):
            turn = 1 if scenario == 'chain' else 3
        pieces.append((color, mask, turn))
    for i in range(unmatched):
        mask = _mask(rng, small=compact)
        if rng.random() < .3 and scenario != 'chain':
            mask = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool)
        pieces.append((len(sites) + i + 2, mask, int(rng.integers(4))))
    grid = _place_inventory(rng, grid, pieces)
    if grid is None:
        return None
    actual = np.rot90(np.asarray(palette, dtype=np.int8)[grid], rotation).copy()
    try:
        info = _recognize(actual)
        output, stats = _simulate(info)
        _, neck_stats = _simulate(info, contour=False)
    except AssertionError:
        return None
    planned = {int(palette[i + 2]) for i, s in enumerate(sites) if s[3]}
    if stats['activated'] != planned:
        return None
    if scenario in ('chain', 'cascade', 'combine') and not (stats['activated'] - neck_stats['activated']):
        return None
    if scenario == 'parallel':
        reached_rows = [info['stations'][c]['row'] for c in stats['activated']]
        if len(reached_rows) == len(set(reached_rows)):
            return None
    return dict(input=actual, output=np.rot90(output, -info['turn']).copy())


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    stable_roles = bool(rng.integers(2))
    base = list(map(int, rng.permutation(10)))
    first_rotation = int(rng.integers(4))
    train_scenarios = ['chain', 'parallel'] + ['general'] * (nt - 2)
    rng.shuffle(train_scenarios)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            scenario = train_scenarios[i] if split == 'train' else str(rng.choice(('combine', 'cascade')))
            # A new combination of taught serial/parallel recruitment, rather
            # than a new local operation or mandatory larger test canvas.
            for attempt in range(100):
                if stable_roles:
                    palette = base[:2] + list(map(int, rng.permutation(base[2:])))
                else:
                    palette = list(map(int, rng.permutation(10)))
                rotation = (first_rotation + i + (split == 'test')) % 4
                pair = _construct(rng, scenario, palette, rotation, compact=attempt >= 60)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analysis(pair):
    info = _recognize(pair['input'])
    expected, stats = _simulate(info)
    _ensure(np.array_equal(pair['output'], np.rot90(expected, -info['turn'])), 'incorrect answer')
    _, bare = _simulate(info, contour=False)
    all_output, _ = _simulate(info, activate_all=True)
    sites, stocks = info['stations'], info['stocks']
    stats.update(info=info, contour_witness=bool(stats['activated'] - bare['activated']),
                 all_dock_diff=not np.array_equal(expected, all_output),
                 unmatched=set(stocks) - set(sites),
                 rotated=any(not np.array_equal(stocks[c]['mask'], sites[c]['canonical'])
                             for c in stats['activated']),
                 simultaneous=len([sites[c]['row'] for c in stats['activated']]) >
                              len({sites[c]['row'] for c in stats['activated']}))
    # Conservation is checked per label, not just total occupied-cell counts.
    for color, site in sites.items():
        actual = set(map(tuple, np.argwhere(expected == color)))
        desired = (set(map(tuple, site['target'])) if color in stats['activated']
                   else {(site['row'], site['col']), (site['row'], site['col'] + 1)})
        _ensure(actual == desired, 'piece lost, duplicated, or incorrectly moved')
    _ensure(not (set(map(int, np.unique(expected))) & stats['unmatched']), 'unmatched inventory survived')
    _ensure(np.all(expected[info['a'] == info['guide']] == info['guide']), 'entry overwritten')
    return stats


def validate_generated_puzzle(puzzle) -> None:
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _ensure(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _ensure(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal counts')
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _ensure(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair')
            a, out = _grid(pair['input']), _grid(pair['output'])
            _ensure(a.shape == out.shape and not np.array_equal(a, out), 'unchanged or resized grid')
            _ensure(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            analyses[split].append(_analysis(dict(input=a, output=out)))
    train = analyses['train']
    _ensure(any(s['contour_witness'] for s in train), 'no contour-dependent activation taught')
    _ensure(any(s['simultaneous'] for s in train), 'parallel activation not taught')
    _ensure(any(s['inactive'] and s['all_dock_diff'] for s in train), 'reachability filtering not taught')
    _ensure(any(s['unmatched'] for s in train), 'color filtering not taught')
    _ensure(any(s['rotated'] for s in train), 'rotation not taught')
    _ensure(all(len(s['activated']) <= 2 for s in train), 'training composition budget exceeded')
    for s in analyses['test']:
        _ensure(s['contour_witness'], 'test must combine contour and recruitment')
        _ensure(len(s['activated']) >= 3, 'test must combine taught causal motifs')


def self_test() -> dict:
    """Bounded semantic checks and measured variation over 42 reproducible seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = dict(seeds=42, pairs=0, shapes=set(), silhouettes=set(), widths=set(),
                   depths=set(), turns=set(), backgrounds=set(), entry_counts=set(),
                   station_counts=set(), active_counts=set(), inert_counts=set(),
                   clipped=0, simultaneous=0, diagonal=0, contour_witnesses=0)
    keys = set()
    for puzzle in puzzles:
        keys.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                s = _analysis(pair)
                info = s['info']
                metrics['pairs'] += 1
                metrics['shapes'].add(pair['input'].shape)
                metrics['turns'].add(info['turn'])
                metrics['backgrounds'].add(info['bg'])
                metrics['entry_counts'].add(len(info['entries']))
                metrics['station_counts'].add(len(info['stations']))
                metrics['active_counts'].add(len(s['activated']))
                metrics['inert_counts'].add(len(s['inactive']))
                metrics['simultaneous'] += s['simultaneous']
                metrics['contour_witnesses'] += s['contour_witness']
                metrics['clipped'] += any(lo == 0 or hi == info['a'].shape[1] - 1
                                          for _, lo, hi, _ in s['events'])
                for site in info['stations'].values():
                    m = site['canonical']
                    metrics['silhouettes'].add(_key(m))
                    metrics['depths'].add(m.shape[0])
                    metrics['widths'].add(m.shape[1])
                    # A four-neighbor flood distinguishes diagonal wings.
                    todo = [tuple(np.argwhere(m)[0])]
                    found = {todo[0]}
                    while todo:
                        r, c = todo.pop()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            p = r + dr, c + dc
                            if (0 <= p[0] < m.shape[0] and 0 <= p[1] < m.shape[1]
                                    and m[p] and p not in found):
                                found.add(p)
                                todo.append(p)
                    metrics['diagonal'] += len(found) != int(m.sum())
                _ensure(np.array_equal(solve(pair['input'].copy()), pair['output']), 'input-only solve failed')
                # The family is covariant under all rotations and reflection.
                for t in (1, 2, 3):
                    _ensure(np.array_equal(solve(np.rot90(pair['input'], t)),
                                           np.rot90(pair['output'], t)), 'rotation covariance')
                _ensure(np.array_equal(solve(np.fliplr(pair['input'])),
                                       np.fliplr(pair['output'])), 'reflection covariance')
    _ensure(len(keys) == 42, 'seed diversity')
    _ensure(len(metrics['shapes']) >= 45 and len(metrics['silhouettes']) >= 60, 'structural diversity')
    _ensure(metrics['widths'] == {2, 4, 6, 8} and metrics['depths'] == {2, 3, 4, 5}, 'shape range')
    _ensure(metrics['turns'] == set(range(4)) and metrics['backgrounds'] == set(range(10)), 'role diversity')
    _ensure(metrics['entry_counts'] == {1, 2}, 'entry diversity')
    _ensure(len(metrics['station_counts']) >= 4 and len(metrics['active_counts']) >= 3, 'count diversity')
    _ensure(metrics['clipped'] > 0 and metrics['diagonal'] > 0, 'boundary/diagonal coverage')
    a, b = generate_puzzle(991, 2, 1), generate_puzzle(991, 2, 1)
    _ensure(all(_key(x[k]) == _key(y[k]) for split in a for x, y in zip(a[split], b[split])
                for k in ('input', 'output')), 'nondeterminism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + nq, nt, nq)
            _ensure(len(p['train']) == nt and len(p['test']) == nq, 'count mismatch')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    damaged = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in a[s]] for s in a}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged answer accepted')
    return {k: (len(v) if isinstance(v, set) else v) for k, v in metrics.items()}
