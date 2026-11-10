"""Invisible cisterns: conserved liquid in a vessel inferred from diagonal rails.

Only NumPy and the standard library are used. No parent module is imported.
The solver recognizes all roles from one input, reconstructs the implicit
vessel, then pours the visible supply. Sampling metadata is never consulted.
"""
from __future__ import annotations

from collections import deque
import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with axes 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    """Four-connected masks; connections through a corner do not count."""
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        cells = [start]
        while queue:
            r, c = queue.popleft()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    queue.append(p)
                    cells.append(p)
        part = np.zeros((h, w), dtype=bool)
        rr, cc = zip(*cells)
        part[rr, cc] = True
        result.append(part)
    return result


def _read_input(grid):
    """Background majority; liquid has a horizontal pair; rails do not."""
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    if counts.max() <= grid.size // 2:
        raise ValueError("background must occupy more than half the input")
    adjacent = (grid[:, :-1] == grid[:, 1:]) & (grid[:, :-1] != background)
    liquids = np.unique(grid[:, :-1][adjacent])
    if len(liquids) != 1:
        raise ValueError("exactly one nonbackground color has horizontal neighbors")
    liquid = int(liquids[0])
    marker = (grid != background) & (grid != liquid)
    if not marker.any():
        raise ValueError("missing rails")
    # No marker pixels may touch horizontally, even across marker colors.
    if np.any(marker[:, :-1] & marker[:, 1:]):
        raise ValueError("rails cannot touch horizontally")
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in np.argwhere(marker):
            lines.setdefault(int(c-slope*r), []).append(int(r))
        strokes, valid = [], True
        for q, rows in sorted(lines.items()):
            runs = []
            for r in sorted(rows):
                if not runs or r != runs[-1][-1]+1:
                    runs.append([])
                runs[-1].append(r)
            for run in runs:
                colors = {int(grid[r, q+slope*r]) for r in run}
                if len(run) < 2 or len(colors) != 1:
                    valid = False
                    break
                strokes.append(dict(q=q, lo=2*run[0]+slope*q,
                                    hi=2*run[-1]+slope*q,
                                    length=len(run), color=colors.pop()))
        if valid and len(strokes) >= 2:
            candidates.append((slope, sorted(strokes, key=lambda s: (s['q'], s['lo']))))
    if len(candidates) != 1:
        raise ValueError("rails must have exactly one recognizable diagonal slope")
    slope, strokes = candidates[0]
    return dict(background=background, liquid=liquid, marker=marker,
                slope=slope, strokes=strokes)


def _vessel(grid, info):
    """Continuous endpoint intervals, including scanlines between pixels."""
    rr, cc = np.indices(grid.shape)
    q = cc-info['slope']*rr
    t = rr+info['slope']*cc
    mask = np.zeros(grid.shape, dtype=bool)
    for level in range(int(t.min()), int(t.max())+1):
        active = [s['q'] for s in info['strokes'] if s['lo'] <= level <= s['hi']]
        if len(active) >= 2:
            mask |= (t == level) & (q > min(active)) & (q < max(active))
    return mask & ~info['marker']


def _pairwise_vessel(grid, info, alternative=''):
    """A separate geometric reconstruction, plus explicit weaker alternatives."""
    strokes = info['strokes']
    pairs = [(i, j) for i in range(len(strokes)) for j in range(i+1, len(strokes))]
    if alternative == 'consecutive':
        # Adjacency is within interval-overlap components, as in parent B's
        # plausible competing interpretation; no arbitrary distance grouping.
        unseen = set(range(len(strokes)))
        pairs = []
        while unseen:
            group = {min(unseen)}
            unseen -= group
            queue = list(group)
            while queue:
                i = queue.pop()
                neighbors = {j for j in unseen if
                             max(strokes[i]['lo'], strokes[j]['lo']) <=
                             min(strokes[i]['hi'], strokes[j]['hi'])}
                unseen -= neighbors
                group |= neighbors
                queue.extend(sorted(neighbors))
            ordered = sorted(group, key=lambda i: strokes[i]['q'])
            pairs.extend(zip(ordered, ordered[1:]))
    rr, cc = np.indices(grid.shape)
    q, t = cc-info['slope']*rr, rr+info['slope']*cc
    mask = np.zeros(grid.shape, dtype=bool)
    for i, j in pairs:
        a, b = strokes[i], strokes[j]
        if alternative == 'same_color' and a['color'] != b['color']:
            continue
        span = ((t >= max(a['lo'], b['lo'])) & (t <= min(a['hi'], b['hi']))
                & (q > min(a['q'], b['q'])) & (q < max(a['q'], b['q'])))
        if alternative == 'pixel_centers':
            span &= ((t-a['q']) % 2 == 0) & ((t-b['q']) % 2 == 0)
        mask |= span
    return mask & ~info['marker']


def _pour(grid, info, vessel, reverse=False, top_first=False):
    """Fall, spread on support, find an outlet, fall again, or retain a cell.

    All cells outside vessel (including unpainted space) are impermeable.
    Complete, uniquely allocated levels make left/right and source-order
    choices irrelevant for admitted family members.
    """
    liquid, background = info['liquid'], info['background']
    source = grid == liquid
    if np.any(source & ~vessel):
        raise ValueError("all liquid must begin inside the inferred vessel")
    filled = np.zeros(grid.shape, dtype=bool)
    points = sorted(map(tuple, np.argwhere(source)),
                    key=lambda p: (p[0] if top_first else -p[0], -p[1] if reverse else p[1]))
    h, w = grid.shape
    # Cache horizontal spans. Expanding the same supported span once per
    # unit is sufficient; doing it again at every column adds no routes.
    spans = {}
    for row in range(h):
        col = 0
        while col < w:
            if not vessel[row, col]:
                col += 1
                continue
            end = col
            while end+1 < w and vessel[row, end+1]:
                end += 1
            for c in range(col, end+1):
                spans[row, c] = (col, end)
            col = end+1
    for source_row, source_col in points:
        # Search hydraulic routes for this unit. At a dry unsupported cell
        # gravity forces a fall. Supported rows communicate sideways. Water
        # transmits pressure downward and can displace its surface upward by
        # one cell; this avoids treating an already deposited unit as a wall.
        visited = set()
        stack = [(int(source_row), int(source_col))]
        available = set()
        while stack:
            row, col = stack.pop()
            if (row, col) in visited:
                continue
            if not vessel[row, col]:
                continue
            if row+1 < h and vessel[row+1, col] and not filled[row+1, col]:
                visited.add((row, col))
                if not filled[row, col]:
                    available.add((row, col))
                elif row > 0 and vessel[row-1, col]:
                    stack.append((row-1, col))
                stack.append((row+1, col))
                continue
            left, right = spans[row, col]
            for c in range(left, right+1):
                visited.add((row, c))
                if not filled[row, c]:
                    available.add((row, c))
                elif row > 0 and vessel[row-1, c]:
                    stack.append((row-1, c))
                if row+1 < h and vessel[row+1, c]:
                    stack.append((row+1, c))
        if not available:
            raise ValueError("supply exceeds its accessible closed pocket")
        deepest = max(r for r, _ in available)
        slots = [c for r, c in available if r == deepest]
        filled[deepest, max(slots) if reverse else min(slots)] = True
    out = grid.copy()
    out[source] = background
    out[filled] = liquid
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    info = _read_input(grid)
    return _pour(grid, info, _vessel(grid, info))


def _catchments(vessel, pools):
    """Union of ALL dry routes, separate from the wetted hydraulic search.

    Each bit denotes a destination. The extra bit denotes unassigned sinks.
    A singleton destination bit certifies a safe initial source cell.
    """
    h, w = vessel.shape
    labels = np.zeros((h, w), dtype=np.int64)
    absorbing = np.zeros((h, w), dtype=np.int64)
    for i, pool in enumerate(pools):
        absorbing[pool] = 1 << i
    unknown = 1 << len(pools)
    for row in range(h-1, -1, -1):
        col = 0
        while col < w:
            if not vessel[row, col]:
                col += 1
                continue
            end = col
            while end+1 < w and vessel[row, end+1]:
                end += 1
            exits = 0
            for c in range(col, end+1):
                exits |= int(absorbing[row, c])
                if row+1 < h and vessel[row+1, c]:
                    exits |= int(labels[row+1, c])
            for c in range(col, end+1):
                if absorbing[row, c]:
                    labels[row, c] = absorbing[row, c]
                elif row+1 < h and vessel[row+1, c]:
                    labels[row, c] = labels[row+1, c]
                else:
                    labels[row, c] = exits or unknown
            col = end+1
    return labels


def _pool_candidates(vessel):
    rr = np.arange(vessel.shape[0])[:, None]
    result = []
    for top in range(1, vessel.shape[0]):
        for pool in _components(vessel & (rr >= top)):
            n = int(pool.sum())
            if not 3 <= n <= 100 or n >= int(vessel.sum()):
                continue
            if not pool[top].any() or not np.any(pool[top] & vessel[top-1]):
                continue
            result.append(pool)
    return result


def _source(rng, eligible, pool, style):
    n = int(pool.sum())
    outside = np.argwhere(eligible & ~pool)
    adjacent = np.argwhere(eligible[:, :-1] & eligible[:, 1:])
    if not len(outside) or not len(adjacent) or int(eligible.sum()) < n:
        raise ValueError("no visibly identified mobile supply")
    # One horizontal pair identifies this color as liquid; remaining cells
    # may be droplets, bands, compact patches, or already settled liquid.
    pair = adjacent[int(rng.integers(len(adjacent)))]
    mandatory = {tuple(map(int, pair)), (int(pair[0]), int(pair[1])+1)}
    anchor = outside[int(rng.integers(len(outside)))]
    mandatory.add(tuple(map(int, anchor)))
    points = np.argwhere(eligible)
    jitter = rng.random(len(points))
    dr, dc = abs(points[:, 0]-anchor[0]), abs(points[:, 1]-anchor[1])
    if style == 0:
        scores = dr+dc+jitter
    elif style == 1:
        scores = jitter
    elif style == 2:
        scores = 3*dr+0.1*dc+jitter
    elif style == 3:
        scores = 3*dc+0.1*dr+jitter
    else:
        scores = (~pool[points[:, 0], points[:, 1]])*3+jitter
    chosen = set(mandatory)
    for k in np.argsort(scores):
        if len(chosen) == n:
            break
        chosen.add(tuple(map(int, points[k])))
    source = np.zeros(pool.shape, dtype=bool)
    for p in chosen:
        source[p] = True
    if int(source.sum()) != n:
        raise ValueError("supply count failed")
    return source


def _interval(q, center, length, rng):
    anchor = center-(center-q) % 2
    before = int(rng.integers(0, length-1))
    lo = anchor-2*before
    return lo, lo+2*(length-1)


def _specs(rng, mode):
    """Sample rail geometry without ever invoking a parent generator."""
    if mode == 'witness':
        count = int(rng.integers(3, 5))
        gaps = [int(rng.integers(3, 8)) for _ in range(count-1)]
        qs = [0]
        for gap in gaps:
            qs.append(qs[-1]+gap)
        extent = int(rng.integers(6, 12))
        specs = []
        short = int(rng.integers(1, count-1))
        for i, q in enumerate(qs):
            if i == short:
                lo = int(rng.integers(-5, 2))
                lo -= (lo-q) % 2
                hi = lo+2*int(rng.integers(1, 3))
            else:
                lo = -extent+(q % 2)
                hi = extent-(extent-q) % 2
            specs.append((q, lo, hi))
        return specs
    if mode == 'separate':
        groups = 2
    elif mode == 'test':
        groups = int(rng.choice([1, 2, 3], p=[.35, .55, .10]))
    else:
        groups = int(rng.choice([1, 2, 3], p=[.65, .30, .05]))
    result = []
    for group in range(groups):
        maximum = 6 if groups == 1 else (4 if groups == 2 else 3)
        minimum = 4 if mode == 'test' and groups == 1 else 2
        count = int(rng.integers(minimum, maximum+1))
        qs = [0]
        for _ in range(count-1):
            qs.append(qs[-1]+int(rng.integers(2, 7 if count >= 5 else 10)))
        max_length = 14 if groups == 1 else (7 if groups == 2 else 4)
        lengths = [int(rng.integers(3, max_length+1)) for _ in qs]
        layout = int(rng.integers(4))
        if layout == 0:
            lengths = [lengths[0]]*count
        intervals = []
        for q, length in zip(qs, lengths):
            if layout == 1:
                lo = q % 2
                hi = lo+2*(length-1)
            elif layout == 2 and intervals:
                lo = intervals[-1][1]-int(rng.integers(1, 5))
                lo -= (lo-q) % 2
                hi = lo+2*(length-1)
            else:
                lo, hi = _interval(q, 0, length, rng)
            intervals.append((lo, hi))
        q_shift = 2*int(rng.integers(-4, 5))
        t_shift = 0
        if result:
            t_shift = max(hi for _, _, hi in result)+int(rng.integers(3, 7))-min(lo for lo, _ in intervals)
            t_shift += t_shift % 2
        result.extend((q+q_shift, lo+t_shift, hi+t_shift)
                      for q, (lo, hi) in zip(qs, intervals))
    return result


def _render(rng, specs, slope, background, liquid, palette, multicolor):
    strokes = []
    for q, lo, hi in specs:
        strokes.append([((t-slope*q)//2, (q+slope*t)//2)
                        for t in range(lo, hi+1, 2)])
    cells = [p for stroke in strokes for p in stroke]
    if len(set(cells)) != len(cells):
        raise ValueError("overlapping rails")
    rmin, cmin = np.min(cells, axis=0)
    rmax, cmax = np.max(cells, axis=0)
    spans = (int(rmax-rmin+1), int(cmax-cmin+1))
    if max(spans) > 30:
        raise ValueError("geometry exceeds ARC canvas")
    dims, offsets = [], []
    for span in spans:
        extra = int(rng.integers(0, min(6, 30-span)+1))
        if rng.random() < .12:
            extra = 30-span
        dims.append(span+extra)
        offsets.append(int(rng.integers(0, extra+1)))
    grid = np.full(tuple(dims), background, dtype=np.int8)
    for i, stroke in enumerate(strokes):
        color = palette[i % len(palette)] if multicolor else palette[0]
        for r, c in stroke:
            grid[r-int(rmin)+offsets[0], c-int(cmin)+offsets[1]] = color
    # Read the intended geometry from translated visible endpoints. After
    # sources are painted, _read_input independently recognizes maximal runs.
    marker = grid != background
    if np.any(marker[:, :-1] & marker[:, 1:]):
        raise ValueError("horizontally touching rails")
    translated = []
    dr, dc = offsets[0]-int(rmin), offsets[1]-int(cmin)
    for (q, lo, hi), stroke in zip(specs, strokes):
        translated.append(dict(q=q+dc-slope*dr, lo=lo+dr+slope*dc,
                               hi=hi+dr+slope*dc, length=len(stroke),
                               color=int(grid[stroke[0][0]+dr, stroke[0][1]+dc])))
    info = dict(background=background, liquid=liquid, marker=marker,
                slope=slope, strokes=sorted(translated, key=lambda s: (s['q'], s['lo'])))
    return grid, info


def _witnesses(grid, out, info, vessel):
    water = out == info['liquid']
    return {name: bool(np.any(water & ~_pairwise_vessel(grid, info, name)))
            for name in ('consecutive', 'pixel_centers', 'same_color')}


def _make_pair(rng, background, liquid, palette, mode, slope, seen,
               forbidden_volume=None, forbidden_depth=None):
    for _ in range(180):
        try:
            base, info = _render(rng, _specs(rng, mode), slope, background,
                                 liquid, palette, multicolor=mode == 'witness' or rng.random() < .65)
            vessel = _vessel(base, info)
            regions = _components(vessel)
            candidates = _pool_candidates(vessel)
            if not candidates:
                continue
            rng.shuffle(candidates)
            wanted = int(rng.integers(1, min(4, len(regions))+1))
            if mode == 'separate':
                if len(regions) < 2:
                    continue
                wanted = min(3, len(regions))
            pools = []
            occupied = np.zeros(base.shape, dtype=bool)
            for pool in candidates:
                # Never select two pools in the same region for the separate
                # witness. Other examples may use distinct pockets in one.
                touching = occupied.copy()
                touching[1:] |= occupied[:-1]
                touching[:-1] |= occupied[1:]
                touching[:, 1:] |= occupied[:, :-1]
                touching[:, :-1] |= occupied[:, 1:]
                if np.any(pool & touching):
                    continue
                if mode == 'separate' and any(np.any(region & pool) and np.any(region & occupied)
                                              for region in regions):
                    continue
                pools.append(pool)
                occupied |= pool
                if len(pools) == wanted:
                    break
            if len(pools) != wanted:
                continue
            if forbidden_depth is not None and all(_depth(p) == forbidden_depth for p in pools):
                continue
            if mode == 'separate' and len({int(np.nonzero(p)[0].min()) for p in pools}) < 2:
                continue
            labels = _catchments(vessel, pools)
            source = np.zeros(base.shape, dtype=bool)
            for i, pool in enumerate(pools):
                source |= _source(rng, labels == 1 << i, pool, int(rng.integers(5)))
            inp, out = base.copy(), base.copy()
            inp[source] = liquid
            out[occupied] = liquid
            if forbidden_volume is not None and int(source.sum()) == forbidden_volume:
                continue
            if int((inp == background).sum()) <= inp.size // 2 or _key(inp) in seen:
                continue
            parsed = _read_input(inp)
            if not np.array_equal(_vessel(inp, parsed), vessel):
                continue
            if not all(np.array_equal(_pour(inp, parsed, vessel, rev, order), out)
                       for rev in (False, True) for order in (False, True)):
                continue
            if not np.any(inp != out):
                continue
            if np.argwhere(out == liquid)[:, 0].sum() <= np.argwhere(inp == liquid)[:, 0].sum():
                continue
            witnesses = _witnesses(inp, out, parsed, vessel)
            if mode == 'witness' and not all(witnesses.values()):
                continue
            if mode == 'separate' and np.array_equal(_global_bottom(inp, parsed, vessel), out):
                continue
            return dict(input=inp, output=out)
        except ValueError:
            continue
    raise RuntimeError("bounded cistern construction exhausted")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    colors = list(map(int, rng.permutation(10)))
    background, liquid = colors[:2]
    marker_colors = colors[2:]
    first_slope = int(rng.choice([-1, 1]))
    puzzle = dict(train=[], test=[])
    seen = set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            mode = ('witness' if i == 0 else 'separate' if i == 1 else 'free') if split == 'train' else 'test'
            slope = first_slope if split == 'train' and i == 0 else -first_slope if split == 'train' and i == 1 else int(rng.choice([-1, 1]))
            palette = list(map(int, rng.choice(marker_colors, size=int(rng.integers(2, 5)), replace=False)))
            forbidden = int((puzzle['train'][0]['input'] == liquid).sum()) if split == 'train' and i == 1 else None
            first_depths = {_depth(p) for p in _components(puzzle['train'][0]['output'] == liquid)} if split == 'train' and i == 1 else set()
            depth = next(iter(first_depths)) if len(first_depths) == 1 else None
            pair = _make_pair(rng, background, liquid, palette, mode, slope, seen, forbidden, depth)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    return grid.shape, grid.tobytes()


def _depth(mask):
    rows = np.nonzero(mask)[0]
    return int(rows.max()-rows.min()+1)


def _global_bottom(grid, info, vessel):
    """Deliberately wrong: ignore source catchments and all spill barriers."""
    points = sorted(map(tuple, np.argwhere(vessel)), key=lambda p: (-p[0], p[1]))
    n = int((grid == info['liquid']).sum())
    out = grid.copy()
    out[out == info['liquid']] = info['background']
    for p in points[:n]:
        out[p] = info['liquid']
    return out


def _inspect_pair(inp, out):
    info = _read_input(inp)
    vessel = _vessel(inp, info)
    assert np.array_equal(vessel, _pairwise_vessel(inp, info))
    water, before = out == info['liquid'], inp == info['liquid']
    assert inp.shape == out.shape and np.any(inp != out)
    assert np.array_equal(inp[info['marker']], out[info['marker']]), "rails changed"
    changed = inp != out
    assert np.all(((inp == info['background']) | before)[changed])
    assert np.all(((out == info['background']) | water)[changed])
    assert not np.any(before & ~vessel) and not np.any(water & ~vessel)
    assert int(before.sum()) == int(water.sum()) > 0, "volume changed"
    assert int(water.sum()) < int(vessel.sum()), "must leave some vessel unfilled"
    assert np.argwhere(water)[:, 0].sum() > np.argwhere(before)[:, 0].sum()
    pools = _components(water)
    rr = np.arange(inp.shape[0])[:, None]
    for pool in pools:
        top = int(np.nonzero(pool)[0].min())
        containing = [p for p in _components(vessel & (rr >= top)) if np.any(p & pool)]
        assert len(containing) == 1 and np.array_equal(containing[0], pool), "incomplete or leaking level"
    destinations = _catchments(vessel, pools)
    assigned = np.zeros(inp.shape, dtype=bool)
    for i, pool in enumerate(pools):
        supply = before & (destinations == 1 << i)
        assert int(supply.sum()) == int(pool.sum()), "ambiguous or wrong basin allocation"
        assigned |= supply
    assert np.array_equal(assigned, before), "some source has unresolved routing"
    assert np.array_equal(solve(inp), out), "solver disagrees"
    witnesses = _witnesses(inp, out, info, vessel)
    # Physical marker pixels alone cannot explain the stable water: treat
    # every nonmarker cell as free and it falls past the invisible vessel.
    without_b = _pour(inp, info, ~info['marker'])
    return dict(background=info['background'], liquid=info['liquid'], slope=info['slope'],
                shape=inp.shape, strokes=len(info['strokes']),
                lengths=tuple(s['length'] for s in info['strokes']),
                colors=len(set(s['color'] for s in info['strokes'])),
                vessel_regions=len(_components(vessel)), pools=len(pools),
                levels=tuple(sorted({int(np.nonzero(p)[0].min()) for p in pools})),
                depths=tuple(sorted({_depth(p) for p in pools})),
                volume=int(water.sum()), capacity=int(vessel.sum()),
                retained=int((water & before).sum()),
                lateral=not np.array_equal(water.sum(axis=0), before.sum(axis=0)),
                omission_b=not np.array_equal(without_b, out),
                omission_a=int(water.sum()) < int(vessel.sum()),
                global_witness=not np.array_equal(_global_bottom(inp, info, vessel), out),
                edge=bool(info['marker'][0].any() or info['marker'][-1].any()
                          or info['marker'][:, 0].any() or info['marker'][:, -1].any()),
                **witnesses)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, stats = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            stats[split].append(_inspect_pair(inp, out))
    records = stats['train']+stats['test']
    assert len({(r['background'], r['liquid']) for r in records}) == 1
    assert len({r['volume'] for r in stats['train']}) >= 2, "volume variation not taught"
    assert len({d for r in stats['train'] for d in r['depths']}) >= 2, "variable fill depth not taught"
    assert {r['slope'] for r in stats['train']} == {-1, 1}, "both slopes must be taught"
    for concept in ('consecutive', 'pixel_centers', 'same_color', 'global_witness', 'omission_a', 'omission_b'):
        assert any(r[concept] for r in stats['train']), f"missing {concept} demonstration"
    assert any(r['pools'] >= 2 and len(r['levels']) >= 2 for r in stats['train']), "separate levels not taught"


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(p['input'], dtype=np.int8)), _key(np.asarray(p['output'], dtype=np.int8)))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """Bounded checks over 42 seeds, fixtures, legal counts, and corruptions."""
    # Hand-calculated suspended diamond: exact continuous bracketing gives
    # seven cells. Five liquid cells settle as its two complete bottom rows.
    inp = np.zeros((5, 5), dtype=np.int8)
    inp[(0, 1, 2, 2, 3, 4), (2, 1, 0, 4, 3, 2)] = 3
    inp[1, 2:4] = 1
    inp[2, 1:4] = 1
    expected = inp.copy()
    expected[expected == 1] = 0
    expected[2, 1:4] = 1
    expected[3, 1:3] = 1
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(solve(inp[:, ::-1]), expected[:, ::-1])
    records, fingerprints, structures, shapes = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid, out = pair['input'], pair['output']
                record = _inspect_pair(grid, out)
                records.append(record)
                info = _read_input(grid)
                vessel = _vessel(grid, info)
                for reverse in (False, True):
                    for top_first in (False, True):
                        assert np.array_equal(_pour(grid, info, vessel, reverse, top_first), out)
                normalized = np.where(grid == info['background'], 0,
                                      np.where(grid == info['liquid'], 1, 2)).astype(np.int8)
                structures.add(_key(normalized))
                shapes.add(grid.shape)
    assert len(fingerprints) == 42 and len(structures) >= 150 and len(shapes) >= 50
    assert len({r['background'] for r in records}) >= 8
    assert len({r['liquid'] for r in records}) >= 8
    assert {r['slope'] for r in records} == {-1, 1}
    assert len({r['strokes'] for r in records}) >= 5
    assert len({n for r in records for n in r['lengths']}) >= 8
    assert len({r['volume'] for r in records}) >= 25
    for feature in ('consecutive', 'pixel_centers', 'same_color', 'global_witness',
                    'omission_a', 'omission_b', 'lateral', 'edge'):
        assert sum(bool(r[feature]) for r in records) >= 20, feature
    assert sum(r['pools'] > 1 for r in records) >= 40
    assert sum(r['retained'] > 0 for r in records) >= 40
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(2026, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject a wrong label, even when its schema and palette are legal.
    p = generate_puzzle(7, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {'seeds': 42, 'pairs': len(records), 'shapes': len(shapes),
            'structures': len(structures), 'volumes': len({r['volume'] for r in records}),
            'semantic_verification': 'self-authored invariants, not independent approval'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
