"""Rail jigsaw: minimum-station assembly among geometrically compatible tiles.

No parent generator is imported. All semantics, including the palette, are
recovered from each input. The only persistent constants are family rules.
"""
from itertools import combinations
import hashlib

import numpy as np


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        part, stack = {root}, [root]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _square(mask):
    cells = np.argwhere(mask)
    if not len(cells):
        return 0
    shape = cells.max(0) - cells.min(0) + 1
    if shape[0] == shape[1] and len(cells) == int(np.prod(shape)):
        return int(shape[0])
    return 0


def _partition(grid):
    candidates = []
    h, w = grid.shape
    for divider in np.unique(grid):
        rows = np.flatnonzero(np.all(grid == divider, axis=1)).tolist()
        cols = np.flatnonzero(np.all(grid == divider, axis=0)).tolist()
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [h], [-1] + cols + [w]
        hh = [b - a - 1 for a, b in zip(rr, rr[1:])]
        ww = [b - a - 1 for a, b in zip(cc, cc[1:])]
        if min(hh + ww) < 3 or len(set(hh)) != 1 or len(set(ww)) != 1:
            continue
        tiles = [grid[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                 for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if any(np.any(p == divider) for p in tiles):
            continue
        corners = {int(p[r, c]) for p in tiles
                   for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) == 1:
            candidates.append(dict(tiles=tiles, bg=corners.pop(), divider=int(divider),
                                   layout=(len(hh), len(ww))))
    assert len(candidates) == 1, "need one equal-tile partition"
    return candidates[0]


def _route_paths(signatures, costs, start, end, cardinality=False):
    """Enumerate candidate assemblies, not physical arrangements of the input."""
    signature = (lambda x: len(x)) if cardinality else (lambda x: x)
    paths = []
    stack = [((start,), costs[start])]
    while stack:
        order, cost = stack.pop()
        if order[-1] == end:
            paths.append((cost, order))
            continue
        port = signature(signatures[order[-1]][1])
        for j, (entry, _) in enumerate(signatures):
            if j not in order and entry and signature(entry) == port:
                stack.append((order + (j,), cost + costs[j]))
    return sorted(paths)


def _acyclic(signatures):
    adjacency = {}
    for a, b in signatures:
        if a and b:
            adjacency.setdefault(a, []).append(b)
    visiting, done = set(), set()

    def visit(a):
        assert a not in visiting, "cyclic port network outside this family"
        if a in done:
            return
        visiting.add(a)
        for b in adjacency.get(a, []):
            visit(b)
        visiting.remove(a)
        done.add(a)

    for a in adjacency:
        visit(a)


def _read(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= n <= 30 for n in grid.shape)
    assert np.all((0 <= grid) & (grid <= 9))
    data = _partition(grid)
    tiles, bg = data['tiles'], data['bg']
    axes = []
    for axis in ('H', 'V'):
        oriented = [p if axis == 'H' else p.T for p in tiles]
        if all(np.all(p[0] == bg) and np.all(p[-1] == bg) for p in oriented):
            sigs = [(tuple(np.flatnonzero(p[:, 0] != bg).tolist()),
                     tuple(np.flatnonzero(p[:, -1] != bg).tolist())) for p in oriented]
            starts = [i for i, (a, b) in enumerate(sigs) if not a and b]
            ends = [i for i, (a, b) in enumerate(sigs) if a and not b]
            if (len(starts) == len(ends) == 1 and
                    all(a or b for a, b in sigs) and
                    all((a and b) or i in starts + ends for i, (a, b) in enumerate(sigs))):
                axes.append((axis, oriented, sigs, starts[0], ends[0]))
    assert len(axes) == 1, "ports must determine an axis and two directed terminals"
    axis, oriented, sigs, start, end = axes[0]
    assert all(len(_components(p != bg)) == 1 for p in oriented)
    boundary_colors = {int(v) for p in oriented for edge in (p[:, 0], p[:, -1])
                       for v in edge if v != bg}
    interior_colors = set(map(int, np.unique(grid))) - boundary_colors - {bg, data['divider']}
    endpoint_candidates = [c for c in interior_colors if _square(oriented[start] == c) >= 2]
    assert len(endpoint_candidates) == 1, "terminal square identifies endpoint color"
    endpoint = endpoint_candidates[0]
    size = _square(oriented[start] == endpoint)
    assert _square(oriented[end] == endpoint) == size
    assert all(not np.any(p == endpoint) for i, p in enumerate(oriented) if i not in (start, end))
    ordinary_candidates = interior_colors - {endpoint}
    assert len(ordinary_candidates) == 1, "one common interior station color"
    ordinary = ordinary_candidates.pop()
    costs = []
    for i, p in enumerate(oriented):
        assert set(map(int, np.unique(p))) <= boundary_colors | {bg, endpoint, ordinary}
        station = p == ordinary
        if i in (start, end):
            assert not station.any()
            station = p == endpoint
            costs.append(0)
        else:
            assert not station.any() or _square(station) == size
            costs.append(int(station.any()))
        if station.any() and i not in (start, end):
            # A station genuinely lies in series: removing it cannot leave an
            # entry-to-exit bypass, even with several fingers in a port.
            for part in _components((p != bg) & ~station):
                assert not (any(c == 0 for r, c in part) and
                            any(c == p.shape[1] - 1 for r, c in part))
    assert sum(costs) >= 3, "show enough ordinary stations to identify their role"
    _acyclic(sigs)
    # All tiles belong to one weakly connected abstract network, including spurs.
    states = {s for ab in sigs for s in ab if s}
    reached, pending = {sigs[start][1]}, [sigs[start][1]]
    while pending:
        a = pending.pop()
        for b, c in sigs:
            if b and c and a in (b, c):
                other = c if a == b else b
                if other not in reached:
                    reached.add(other)
                    pending.append(other)
    assert reached == states
    paths = _route_paths(sigs, costs, start, end)
    assert len(paths) >= 2, "provide competing assemblies"
    assert paths[0][0] >= 1 and paths[0][0] < paths[1][0], "minimum station count must be unique"
    data.update(axis=axis, oriented=oriented, signatures=sigs, start=start, end=end,
                ordinary=ordinary, endpoint=endpoint, size=size, costs=costs,
                paths=paths, order=paths[0][1])
    return data


def _pack(tiles, order, layout, divider):
    nr, nc = layout
    h, w = tiles[0].shape
    result = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, index in enumerate(order):
        r, c = divmod(slot, nc)
        result[r * (h + 1):r * (h + 1) + h,
               c * (w + 1):c * (w + 1) + w] = tiles[index]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select and translate intact tiles using only the supplied input."""
    data = _read(input_grid)
    k = len(data['order'])
    layout = (1, k) if data['axis'] == 'H' else (k, 1)
    output = _pack(data['tiles'], data['order'], layout, data['divider'])
    assert max(output.shape) <= 30
    return output


def _graph_paths(edges, target):
    result, stack = [], [(0, (), 0)]
    while stack:
        node, order, cost = stack.pop()
        if node == target:
            result.append((cost, order))
            continue
        for i, (a, b, weight) in enumerate(edges):
            if a == node:
                stack.append((b, order + (i,), cost + weight))
    return sorted(result)


def _graph_good(edges, target, witness=False, required_cost=None):
    paths = _graph_paths(edges, target)
    if len(paths) < 2 or paths[0][0] < 1 or paths[0][0] == paths[1][0]:
        return False
    if not 1 <= len(paths[0][1]) <= 3 or sum(e[2] for e in edges) < 3:
        return False
    if required_cost is not None and paths[0][0] != required_cost:
        return False
    if witness and not any(len(p[1]) < len(paths[0][1]) and p[0] > paths[0][0] for p in paths):
        return False
    return True


def _graph(rng, n, witness, required_cost=None):
    ecount = n - 2
    if not witness:
        for _ in range(250):
            m = int(rng.integers(4, min(8, ecount + 1) + 1))
            if m * (m - 1) // 2 < ecount:
                continue
            # Begin with a route, attach remaining states in either direction,
            # then add edges. This admits shared prefixes, suffixes and spurs.
            inner = sorted(rng.choice(np.arange(1, m - 1),
                                      int(rng.integers(1, m - 1)), replace=False).tolist())
            spine = [0] + inner + [m - 1]
            pairs = set(zip(spine, spine[1:]))
            active = set(spine)
            missing = [i for i in range(m) if i not in active]
            rng.shuffle(missing)
            for v in missing:
                other = int(rng.choice(sorted(active)))
                pairs.add(tuple(sorted((v, other))))
                active.add(v)
            extra = [p for p in combinations(range(m), 2) if p not in pairs]
            rng.shuffle(extra)
            pairs.update(extra[:ecount - len(pairs)])
            edges = [(a, b, int(rng.random() < .62)) for a, b in sorted(pairs)]
            if _graph_good(edges, m - 1, required_cost=required_cost):
                return edges, m
    if required_cost == 2:
        # A three-station route versus its two-station shortcut, with extra
        # outgoing spurs that cannot create a second route to the target.
        base = [(0, 1, 1), (1, 2, 1), (2, 5, 1), (0, 2, 1)]
        extras = [(0, 3, 1), (0, 4, 0), (1, 3, 0), (1, 4, 1), (2, 3, 1), (2, 4, 0)]
        edges = base + extras[:ecount - 4]
        assert _graph_good(edges, 5, required_cost=2)
        return edges, 6
    if ecount == 4:
        # Six-panel fallback: two routes of equal piece count, unequal cost.
        return [(0, 1, 1), (1, 3, 0), (0, 2, 1), (2, 3, 1)], 4
    # Constructive count-versus-length conflict, with variable additional links.
    # Five internal edges make routes with (3 pieces,1 station) and (2,2).
    base = [(0, 1, 0), (1, 2, 1), (2, 5, 0), (0, 3, 1), (3, 5, 1)]
    extras = [p for p in combinations(range(6), 2) if p not in {(a, b) for a, b, _ in base}]
    for _ in range(100):
        rng.shuffle(extras)
        # Attach a dead-end state, then add edges without destroying the witness.
        edges = base + [(int(rng.choice([0, 1, 2, 3])), 4, 1)]
        used = {(a, b) for a, b, _ in edges}
        for a, b in extras:
            if (a, b) in used or len(edges) == ecount:
                continue
            trial = edges + [(a, b, int(rng.random() < .7))]
            if _graph_good(trial, 5, True):
                edges = trial
                used.add((a, b))
        if len(edges) == ecount and _graph_good(edges, 5, True):
            return edges, 6
    # Fixed finite fallback; these additions all preserve the one-station winner.
    additions = [(1, 4, 1), (0, 4, 1), (3, 4, 0), (2, 3, 1), (4, 5, 1)]
    edges = base + additions[:ecount - 5]
    assert _graph_good(edges, 5, True)
    return edges, 6


def _signatures(rng, h, m):
    singles = [(r,) for r in range(1, h - 1)]
    doubles = list(combinations(range(1, h - 1), 2))
    # Equal-cardinality terminal ports make counting fingers an explicit decoy.
    endpoint_pool = singles if rng.random() < .55 else doubles
    indices = rng.choice(len(endpoint_pool), 2, replace=False)
    start, end = [endpoint_pool[int(i)] for i in indices]
    selected = [start, end]
    for pool in (singles, [s for s in doubles if s[1] == s[0] + 1],
                 [s for s in doubles if s[1] > s[0] + 1]):
        if not any(s in pool for s in selected):
            selected.append(pool[int(rng.integers(len(pool)))])
    remainder = [s for s in singles + doubles if s not in selected]
    rng.shuffle(remainder)
    selected.extend(remainder[:m - len(selected)])
    interior = selected[2:]
    rng.shuffle(interior)
    return [start] + interior + [end]


def _tile(rng, h, w, incoming, outgoing, station, size, bg, connector):
    p = np.full((h, w), bg, dtype=np.int8)

    def arm(rows, col, anchor):
        if not rows:
            return
        p[min(rows + (anchor,)):max(rows + (anchor,)) + 1, col] = connector
        for r in rows:
            if col == 1:
                p[r, :2] = connector
            else:
                p[r, w - 2:] = connector

    if station is None:
        spine = int(rng.integers(1, w - 1))
        rows = incoming + outgoing
        p[min(rows):max(rows) + 1, spine] = connector
        for r in incoming:
            p[r, :spine + 1] = connector
        for r in outgoing:
            p[r, spine:] = connector
    else:
        r = int(rng.integers(1, h - size))
        c = int(rng.integers(1, w - size))
        anchor = r + int(rng.integers(size))
        if incoming:
            arm(incoming, 1, anchor)
            p[anchor, 1:c + 1] = connector
        if outgoing:
            arm(outgoing, w - 2, anchor)
            p[anchor, c + size - 1:w - 1] = connector
        p[r:r + size, c:c + size] = station
    return p


def _make_pair(rng, n, witness, axis, size, required_cost=None):
    edges, m = _graph(rng, n, witness, required_cost)
    paths = _graph_paths(edges, m - 1)
    chosen = (0,) + tuple(i + 1 for i in paths[0][1]) + (n - 1,)
    k = len(chosen)
    configs = [(h, w, (nr, n // nr)) for h in range(6, 10)
               for w in range(5, min(8, 31 // k - 1) + 1)
               for nr in range(1, n + 1) if n % nr == 0
               and nr * (h + 1) - 1 <= 30 and (n // nr) * (w + 1) - 1 <= 30]
    h, w, layout = configs[int(rng.integers(len(configs)))]
    signatures = _signatures(rng, h, m)
    palette = list(map(int, rng.permutation(10)))
    bg, divider, ordinary, endpoint = palette[:4]
    connectors = palette[4:4 + int(rng.integers(1, 4))]
    records = [((), signatures[0], endpoint)]
    records += [(signatures[a], signatures[b], ordinary if weight else None)
                for a, b, weight in edges]
    records += [(signatures[-1], (), endpoint)]
    tiles = [_tile(rng, h, w, a, b, station, size, bg,
                   connectors[int(rng.integers(len(connectors)))]) for a, b, station in records]
    if axis == 'V':
        tiles = [p.T.copy() for p in tiles]
        layout = layout[::-1]
    permutation = rng.permutation(n).tolist()
    inp = _pack(tiles, permutation, layout, divider)
    out = _pack(tiles, chosen, (1, k) if axis == 'H' else (k, 1), divider)
    assert np.array_equal(solve(inp), out), "construction and input decode disagree"
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed)
    if seed < 0:
        seed %= 1 << 64
    rng = np.random.default_rng(seed)
    witness_index = int(rng.integers(num_train))
    cost_two_index = (witness_index + int(rng.integers(1, num_train))) % num_train
    axes = ['H', 'V'] + [str(rng.choice(['H', 'V'])) for _ in range(num_train - 2)]
    sizes = [2, 3] + [int(rng.integers(2, 4)) for _ in range(num_train - 2)]
    rng.shuffle(axes)
    rng.shuffle(sizes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            witness = split == 'train' and i == witness_index
            n = (int(rng.choice([8, 9, 10])) if witness else
                 int(rng.choice([6, 8, 9, 10])) if split == 'train' else
                 12 if i == 0 else int(rng.choice([9, 10, 12])))
            axis = axes[i] if split == 'train' else str(rng.choice(['H', 'V']))
            size = sizes[i] if split == 'train' else int(rng.integers(2, 4))
            for _ in range(20):
                pair = _make_pair(rng, n, witness, axis, size,
                                  required_cost=2 if split == 'train' and i == cost_two_index else None)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input generation exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(data):
    order, paths = data['order'], data['paths']
    return dict(tiles=len(data['tiles']), selected=len(order), stations=paths[0][0],
                routes=len(paths), size=data['size'], axis=data['axis'], layout=data['layout'],
                panel_shape=data['tiles'][0].shape,
                zero_tiles=data['costs'].count(0) - 2,
                conflict=any(len(p[1]) < len(order) and p[0] > paths[0][0] for p in paths),
                shared=any(len(set(p[1]) & set(order)) > 2 for p in paths[1:]),
                spur=len(set(range(len(data['tiles']))) - {i for _, p in paths for i in p}),
                kinds={('single' if len(s) == 1 else 'thick' if s[-1] == s[0] + 1 else 'separated')
                       for ab in data['signatures'] for s in ab if s})


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, metadata = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for a in pair.values():
                a = np.asarray(a)
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9))
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            data = _read(inp)
            assert np.array_equal(solve(inp), out)
            # Independently extract the output strip and compare literal tile
            # matrices. A deletion or recoloring inside a chosen tile must fail.
            parsed = _partition(out)
            expected_layout = (1, len(data['order'])) if data['axis'] == 'H' else (len(data['order']), 1)
            assert parsed['layout'] == expected_layout and parsed['divider'] == data['divider']
            assert len(parsed['tiles']) == len(data['order']) < len(data['tiles'])
            for p, index in zip(parsed['tiles'], data['order']):
                assert np.array_equal(p, data['tiles'][index])
            assert len({data['signatures'][data['start']][1], data['signatures'][data['end']][0]}) == 2
            # Ignoring full geometry permits the zero-station terminal-only
            # answer, since terminal signatures have the same cardinality.
            assert len(data['signatures'][data['start']][1]) == len(data['signatures'][data['end']][0])
            metadata[split].append(_features(data))
    tr, te = metadata['train'], metadata['test']
    assert any(f['conflict'] for f in tr), 'demonstrate station count versus number of pieces'
    assert {1, 2} <= {f['stations'] for f in tr}, 'demonstrate varying optimal station counts'
    assert {f['axis'] for f in tr} == {'H', 'V'}
    assert {f['size'] for f in tr} == {2, 3}
    assert set.union(*(f['kinds'] for f in te)) <= set.union(*(f['kinds'] for f in tr))
    assert max(f['tiles'] for f in tr) <= 10 and te[0]['tiles'] == 12


def self_test():
    """Bounded 42-seed semantic/diversity checks; no files or retained galleries."""
    digests, shapes, geometry, stats, palettes = set(), set(), set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _read(pair['input'])
                f = _features(data)
                stats.append(f)
                shapes.add(pair['input'].shape)
                palettes.add((data['bg'], data['divider'], data['ordinary'], data['endpoint']))
                for p in data['tiles']:
                    geometry.add(_key((p != data['bg']).astype(np.int8)))
                for role in ('input', 'output'):
                    digest.update(str(pair[role].shape).encode())
                    digest.update(pair[role].tobytes())
                # Exhaustive input-derived assembly check, no generation graph.
                costs = [sum(data['costs'][i] for i in order) for _, order in data['paths']]
                assert costs[0] < min(costs[1:])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                if seed < 3:
                    # Neither literal colors, tray order nor the global axis
                    # changes the logical assembly.
                    mapping = np.random.default_rng(100 + seed).permutation(10)
                    assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
                    assert np.array_equal(solve(pair['input'].T), pair['output'].T)
                    shuffled = _pack(data['tiles'], list(range(len(data['tiles'])))[::-1],
                                     data['layout'], data['divider'])
                    assert np.array_equal(solve(shuffled), pair['output'])
        digests.add(digest.hexdigest())
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(_key(p[k]) == _key(q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in p)
    assert len(digests) == 42 and len(shapes) >= 20 and len(geometry) >= 100
    assert len(palettes) >= 50
    assert {f['tiles'] for f in stats} == {6, 8, 9, 10, 12}
    assert {f['selected'] for f in stats} == {3, 4, 5}
    assert {f['size'] for f in stats} == {2, 3}
    assert sum(f['conflict'] for f in stats) >= 42
    assert any(f['shared'] for f in stats) and any(f['spur'] for f in stats)
    assert max(f['routes'] for f in stats) >= 4
    # Check all legal explicit counts without storing their results.
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            p = generate_puzzle(9000 + 10 * a + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Changing just one losing-route station into track creates an actual tie.
    # This checks rejection from rendered pixels, rather than traversal order.
    p = generate_puzzle(123, 2, 1)
    witness = next(pair for pair in p['train'] if _features(_read(pair['input']))['conflict'])
    data = _read(witness['input'])
    alternative = next(order for cost, order in data['paths'] if cost == 2)
    index = next(i for i in alternative if i not in data['order'] and data['costs'][i])
    tiles = [tile.copy() for tile in data['tiles']]
    connector = next(int(c) for c in np.unique(tiles[index])
                     if c not in (data['bg'], data['ordinary'], data['endpoint']))
    tiles[index][tiles[index] == data['ordinary']] = connector
    tied = _pack(tiles, list(range(len(tiles))), data['layout'], data['divider'])
    try:
        solve(tied)
    except AssertionError as error:
        assert 'minimum station count must be unique' in str(error)
    else:
        raise AssertionError('tied assemblies accepted')
    # Negative semantics: reject a wrong pixel in a copied tile.
    p = generate_puzzle(123, 2, 1)
    p['test'][0]['output'][1, 1] = (int(p['test'][0]['output'][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
