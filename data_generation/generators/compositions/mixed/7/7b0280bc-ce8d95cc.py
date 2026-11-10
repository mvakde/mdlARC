"""Select a minimum-station arc, then locally contract equal vector runs.

Input roles are recovered from square stations and four-neighbor connectors.
Output roles are fixed: ordinary stations -> 3, connectors -> 5. Endpoints
stay at their original coordinates. The compacted patch is a symbolic drawing;
it is not stretched or reconnected to those endpoints after contraction.
No parent modules, filesystem data, or generation state are used by solve.
"""
from __future__ import annotations

import numpy as np


STATION_HIGHLIGHT = 3
CONNECTOR_HIGHLIGHT = 5
INPUT_COLORS = (0, 1, 2, 4, 6, 7, 8, 9)
NEIGHBORS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in NEIGHBORS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.asarray(cells, dtype=int))
    return result


def _square_size(cells):
    extent = cells.max(0) - cells.min(0) + 1
    if extent[0] == extent[1] and len(cells) == int(np.prod(extent)):
        return int(extent[0])
    return 0


def _box(mask):
    cells = np.argwhere(mask)
    low, high = cells.min(0), cells.max(0) + 1
    return tuple(int(x) for x in (low[0], high[0], low[1], high[1]))


def _walk(adjacency, edges, start, target, first):
    nodes, used = [start], []
    current, edge = start, first
    while True:
        used.append(edge)
        a, b = edges[edge]
        nxt = b if a == current else a
        assert nxt not in nodes, "cycle did not reach second endpoint"
        nodes.append(nxt)
        if nxt == target:
            return nodes, used
        choices = [e for e in adjacency[nxt] if e != edge]
        assert len(choices) == 1
        current, edge = nxt, choices[0]


def _read_input(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    assert len(colors) == 4, "four distinct input roles required"
    assert set(map(int, colors)).isdisjoint((3, 5))
    background = int(colors[np.argmax(counts)])
    assert np.count_nonzero(counts == counts.max()) == 1
    by_color = {int(c): _components(a == c) for c in colors if c != background}
    endpoint_choices = [c for c, blocks in by_color.items()
                        if len(blocks) == 2 and _square_size(blocks[0]) >= 2
                        and _square_size(blocks[0]) == _square_size(blocks[1])]
    assert len(endpoint_choices) == 1, "ambiguous endpoint squares"
    endpoint = endpoint_choices[0]
    size = _square_size(by_color[endpoint][0])
    ordinary_choices = [c for c, blocks in by_color.items() if c != endpoint
                        and len(blocks) >= 3
                        and all(_square_size(b) == size for b in blocks)]
    assert len(ordinary_choices) == 1, "ambiguous ordinary stations"
    ordinary = ordinary_choices[0]
    connector = next(c for c in by_color if c not in (endpoint, ordinary))
    stations = _components((a == ordinary) | (a == endpoint))
    assert all(_square_size(b) == size for b in stations)
    station_map = np.full(a.shape, -1, dtype=int)
    endpoints = []
    for i, cells in enumerate(stations):
        station_map[tuple(cells.T)] = i
        if a[tuple(cells[0])] == endpoint:
            endpoints.append(i)
    assert len(endpoints) == 2
    components = by_color[connector]
    edges = []
    h, w = a.shape
    for cells in components:
        touching = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and station_map[rr, cc] >= 0:
                    touching.add(int(station_map[rr, cc]))
        assert len(touching) == 2, "connector must join two stations"
        edges.append(tuple(sorted(touching)))
    assert len(set(edges)) == len(edges), "parallel connectors are excluded"
    adjacency = [[] for _ in stations]
    for i, (a0, b0) in enumerate(edges):
        adjacency[a0].append(i)
        adjacency[b0].append(i)
    assert all(len(es) == 2 for es in adjacency), "one cycle required"
    paths = [_walk(adjacency, edges, *endpoints, e) for e in adjacency[endpoints[0]]]
    assert set(paths[0][1]).isdisjoint(paths[1][1])
    assert set(paths[0][1]) | set(paths[1][1]) == set(range(len(edges)))
    assert len(paths[0][0]) != len(paths[1][0]), "tied arcs are excluded"
    paths.sort(key=lambda p: len(p[0]))
    return dict(roles=(background, connector, ordinary, endpoint), size=size,
                stations=stations, components=components, edges=edges,
                paths=paths, endpoints=endpoints)


def _extract(a, data, path):
    """Only the intermediate stations and selected edges belong to the patch."""
    mask = np.zeros(a.shape, dtype=bool)
    for i in path[0][1:-1]:
        mask[tuple(data['stations'][i].T)] = True
    for i in path[1]:
        mask[tuple(data['components'][i].T)] = True
    r0, r1, c0, c1 = _box(mask)
    # This restriction makes literal cropping, clearing and reinsertion safe.
    crop_mask = mask[r0:r1, c0:c1]
    patch = a[r0:r1, c0:c1].copy()
    assert np.all(crop_mask | (patch == data['roles'][0])), "arc box contains another object"
    return mask, (r0, r1, c0, c1), patch


def _keep(vectors, global_unique=False):
    if global_unique:
        seen, keep = set(), []
        for i, v in enumerate(vectors):
            key = tuple(map(int, v))
            if key not in seen:
                seen.add(key)
                keep.append(i)
        return keep
    return [0] + [i for i in range(1, len(vectors))
                  if not np.array_equal(vectors[i], vectors[i - 1])]


def _contract(patch, mode='both'):
    # Both equivalence tests refer to the same original symbolic rectangle.
    rows = _keep(patch, mode == 'global') if mode in ('both', 'rows', 'global') else list(range(len(patch)))
    cols = _keep(patch.T, mode == 'global') if mode in ('both', 'cols', 'global') else list(range(patch.shape[1]))
    return patch[np.ix_(rows, cols)].copy()


def _render(a, data, path=None, mode='both'):
    path = data['paths'][0] if path is None else path
    _, box, patch = _extract(a, data, path)
    r0, r1, c0, c1 = box
    reduced = _contract(patch, mode)
    recolored = reduced.copy()
    recolored[reduced == data['roles'][2]] = STATION_HIGHLIGHT
    recolored[reduced == data['roles'][1]] = CONNECTOR_HIGHLIGHT
    out = a.copy()
    out[r0:r1, c0:c1] = data['roles'][0]
    hh, ww = reduced.shape
    out[r0:r0 + hh, c0:c0 + ww] = recolored
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the family rule from input pixels only, without mutating them."""
    a = _grid(input_grid)
    return _render(a, _read_input(a))


def _route(heights, sign):
    """A monotone-column orthogonal route, including possible station sites."""
    result = [(0, 0)]

    def move(target):
        r, c = result[-1]
        rr, cc = target
        while (r, c) != (rr, cc):
            r += int(np.sign(rr - r))
            c += int(np.sign(cc - c))
            result.append((r, c))

    move((sign * int(heights[0]), 0))
    for col in range(len(heights)):
        move((result[-1][0], col + 1))
        if col + 1 < len(heights):
            move((sign * int(heights[col + 1]), col + 1))
    move((0, len(heights)))
    assert len(result) == len(set(result))
    return result


def _profile(rng, width, depth, folded):
    if not folded:
        return [int(rng.integers(1, depth + 1))] * width
    result = rng.integers(1, depth + 1, size=width).tolist()
    # Two distinct heights also guarantee sufficient sites for denser tests.
    if depth > 1:
        result[0] = depth
        result[-1] = 1
    return result


def _positions(rng, intervals, size):
    pitch = size + 2
    slack = 30 - (intervals * pitch + size)
    assert slack >= 0
    extra = int(rng.integers(0, min(slack, 6) + 1))
    gaps = pitch + rng.multinomial(extra, [1 / intervals] * intervals)
    return np.concatenate(([0], np.cumsum(gaps))).astype(int)


def _sample_sites(rng, route, count):
    indices = sorted(map(int, rng.choice(np.arange(1, len(route) - 1), count, replace=False)))
    return [route[i] for i in indices]


def _make_pair(rng, kind, requested_short=None):
    witness = kind == 'witness'
    test = kind == 'test'
    size = 2 if witness else 3 if kind == 'second' else int(rng.integers(2, 4))
    width = 4 if witness else int(rng.integers(4 if test else 3, 6))
    if witness:
        upper_depth, lower_depth = 3, 1
        upper_h, lower_h = [3] * width, [1] * width
    else:
        upper_depth = int(rng.integers(2 if test else 1, 4))
        lower_depth = int(rng.integers(2 if test else 1, 4))
        maximum = (30 - size) // (size + 2)
        if upper_depth + lower_depth > maximum:
            lower_depth = maximum - upper_depth
        upper_h = _profile(rng, width, upper_depth, test or rng.random() < .75)
        lower_h = _profile(rng, width, lower_depth, test or rng.random() < .75)
    upper, lower = _route(upper_h, -1), _route(lower_h, 1)
    short_index = 0 if witness else int(rng.integers(2))
    routes = [upper, lower]
    short_route, long_route = routes[short_index], routes[1 - short_index]
    desired = requested_short if requested_short is not None else int(rng.integers(4, 6) if test else rng.integers(1, 4))
    short_count = min(desired, len(short_route) - 2, len(long_route) - 3)
    assert short_count >= 1
    if test:
        assert short_count >= 4
    long_count = int(rng.integers(short_count + 1, min(len(long_route) - 2, short_count + 4) + 1))
    if witness:
        short_sites = [(-3, 1), (-1, width)]
        long_count = int(rng.integers(4, 6))
    else:
        short_sites = _sample_sites(rng, short_route, short_count)
    long_sites = _sample_sites(rng, long_route, long_count)
    rows = _positions(rng, upper_depth + lower_depth, size)
    cols = _positions(rng, width, size)
    offset = size // 2

    def anchor(site):
        r, c = site
        return int(rows[r + upper_depth]), int(cols[c])

    symbolic = np.zeros((int(rows[-1]) + size, int(cols[-1]) + size), dtype=np.int8)
    for route in routes:
        for a, b in zip(route, route[1:]):
            ar, ac = anchor(a)
            br, bc = anchor(b)
            ar, ac, br, bc = ar + offset, ac + offset, br + offset, bc + offset
            if ar == br:
                symbolic[ar, min(ac, bc):max(ac, bc) + 1] = 1
            else:
                assert ac == bc
                symbolic[min(ar, br):max(ar, br) + 1, ac] = 1
    for site in short_sites + long_sites:
        r, c = anchor(site)
        symbolic[r:r + size, c:c + size] = 2
    for site in ((0, 0), (0, width)):
        r, c = anchor(site)
        symbolic[r:r + size, c:c + size] = 4
    symbolic = np.rot90(symbolic, int(rng.integers(4)))
    if rng.random() < .5:
        symbolic = symbolic[:, ::-1]
    hh, ww = symbolic.shape
    add_h = int(rng.integers(0, min(6, 30 - hh) + 1))
    add_w = int(rng.integers(0, min(6, 30 - ww) + 1))
    top = int(rng.integers(add_h + 1))
    left = int(rng.integers(add_w + 1))
    placed = np.zeros((hh + add_h, ww + add_w), dtype=np.int8)
    placed[top:top + hh, left:left + ww] = symbolic
    palette = list(map(int, rng.choice(INPUT_COLORS, 4, replace=False)))
    lookup = np.zeros(5, dtype=np.int8)
    lookup[[0, 1, 2, 4]] = palette
    inp = lookup[placed]
    data = _read_input(inp)
    assert len(data['paths'][0][0]) - 2 == short_count
    assert len(data['paths'][1][0]) - 2 == long_count
    if witness:
        assert _physical_conflict(data)
        _, _, patch = _extract(inp, data, data['paths'][0])
        assert _separated(patch) and _separated(patch.T)
    return {'input': inp, 'output': _render(inp, data)}


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f'{name} outside permitted range')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    specs = [('train', 'witness', 2), ('train', 'second', 3)]
    specs += [('train', 'ordinary', None)] * (num_train - 2)
    specs += [('test', 'test', None)] * num_test
    for split, kind, count in specs:
        for _ in range(64):
            pair = _make_pair(rng, kind, count)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError('bounded distinct-input sampling exhausted')
    order = rng.permutation(num_train)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    return puzzle


def _physical_conflict(data):
    costs = [sum(len(data['components'][e]) for e in path[1]) for path in data['paths']]
    return costs[0] > costs[1]


def _separated(vectors):
    runs = [tuple(map(int, vectors[i])) for i in _keep(vectors)]
    return len(set(runs)) < len(runs)


def features(input_grid):
    a = _grid(input_grid)
    d = _read_input(a)
    _, box, patch = _extract(a, d, d['paths'][0])
    reduced = _contract(patch)
    return dict(shape=list(a.shape), patch_shape=list(patch.shape),
                compact_shape=list(reduced.shape), anchor=[box[0], box[2]],
                station_size=d['size'], counts=[len(p[0]) - 2 for p in d['paths']],
                connector_cells=[sum(len(d['components'][e]) for e in p[1]) for p in d['paths']],
                physical_conflict=_physical_conflict(d),
                separated_rows=_separated(patch), separated_columns=_separated(patch.T),
                row_runs=[sum(1 for _ in group) for group in _runs(patch)],
                column_runs=[sum(1 for _ in group) for group in _runs(patch.T)],
                roles=list(d['roles']))


def _runs(vectors):
    """Explicit runs for validation, separate from the vector-index contraction."""
    current = [vectors[0]]
    for v in vectors[1:]:
        if np.array_equal(v, current[-1]):
            current.append(v)
        else:
            yield current
            current = [v]
    yield current


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    seen, measures = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == out.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            d = _read_input(a)
            mask, (r0, r1, c0, c1), patch = _extract(a, d, d['paths'][0])
            # Independently build explicit runs, retaining literal representatives.
            row_groups = list(_runs(patch))
            row_reduced = np.asarray([g[0] for g in row_groups])
            col_groups = list(_runs(row_reduced.T))
            compact = np.asarray([g[0] for g in col_groups]).T.copy()
            assert np.array_equal(compact, _contract(patch))
            assert np.array_equal(_contract(_contract(patch.T).T), compact)
            expanded = np.repeat(np.repeat(compact, [len(g) for g in row_groups], axis=0),
                                 [len(g) for g in col_groups], axis=1)
            assert np.array_equal(expanded, patch), 'contraction must retain every vector run'
            recolored = compact.copy()
            recolored[compact == d['roles'][2]] = 3
            recolored[compact == d['roles'][1]] = 5
            expected = a.copy()
            expected[r0:r1, c0:c1] = d['roles'][0]
            h, w = compact.shape
            expected[r0:r0 + h, c0:c0 + w] = recolored
            assert np.array_equal(out, expected)
            assert np.array_equal(out, solve(a))
            outside = np.ones(a.shape, dtype=bool)
            outside[r0:r1, c0:c1] = False
            assert np.array_equal(a[outside], out[outside])
            for n in d['endpoints']:
                cells = tuple(d['stations'][n].T)
                assert np.array_equal(a[cells], out[cells])
            # Both ablations use the same reinsertion/color policy as the answer.
            assert not np.array_equal(out, _render(a, d, mode='none')), 'B is necessary'
            assert not np.array_equal(out, _render(a, d, d['paths'][1])), 'A is necessary'
            assert compact.shape[0] < patch.shape[0] and compact.shape[1] < patch.shape[1]
            assert not np.array_equal(out, _render(a, d, mode='rows'))
            assert not np.array_equal(out, _render(a, d, mode='cols'))
            assert np.count_nonzero(mask) > 0 and np.any(out == 3) and np.any(out == 5)
            f = features(a)
            if f['separated_rows'] or f['separated_columns']:
                assert not np.array_equal(out, _render(a, d, mode='global'))
            measures[split].append(f)
    assert any(m['physical_conflict'] for m in measures['train'])
    assert any(m['separated_rows'] and m['separated_columns'] for m in measures['train'])
    assert len({m['counts'][0] for m in measures['train']}) >= 2
    assert max(m['counts'][0] for m in measures['train']) <= 3
    assert min(m['counts'][0] for m in measures['test']) >= 4


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """42 streamed seeds; no files, traces, persistent caches or retained runs."""
    small = np.asarray([[1, 1, 2, 1], [1, 1, 2, 1], [4, 4, 6, 4], [1, 1, 2, 1]])
    assert np.array_equal(_contract(small), np.asarray([[1, 2, 1], [4, 6, 4], [1, 2, 1]]))
    assert _separated(small) and _separated(small.T)
    shapes, structures, counts, palettes, anchors, puzzles = set(), set(), set(), set(), set(), set()
    found_tie = False
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        if seed < 3:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                before = a.copy()
                assert np.array_equal(solve(a), pair['output']) and np.array_equal(a, before)
                d = _read_input(a)
                f = features(a)
                shapes.add(a.shape)
                counts.add(tuple(f['counts']))
                palettes.add(tuple(f['roles']))
                anchors.add(tuple(f['anchor']))
                semantic = np.zeros(a.shape, dtype=np.int8)
                for i, c in enumerate(d['roles']):
                    semantic[a == c] = i
                r0, r1, c0, c1 = _box(a != d['roles'][0])
                structures.add(_key(semantic[r0:r1, c0:c1]))
                if not found_tie and len(d['stations']) % 2 == 0:
                    cycle = d['paths'][0][0] + d['paths'][1][0][-2:0:-1]
                    tied = a.copy()
                    tied[tied == d['roles'][3]] = d['roles'][2]
                    for n in (cycle[0], cycle[len(cycle) // 2]):
                        tied[tuple(d['stations'][n].T)] = d['roles'][3]
                    try:
                        _read_input(tied)
                    except AssertionError:
                        found_tie = True
                    else:
                        raise AssertionError('tied station counts accepted')
    assert len(puzzles) == 42 and len(shapes) >= 40 and len(structures) >= 180
    assert len(counts) >= 15 and len(palettes) >= 100 and len(anchors) >= 30 and found_tie
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(20261023, train, test)
            assert len(p['train']) == train and len(p['test']) == test
            validate_generated_puzzle(p)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupted = generate_puzzle(0, 2, 1)
    a = corrupted['test'][0]['output']
    a[0, 0] = (int(a[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds and all legal explicit count combinations)')
