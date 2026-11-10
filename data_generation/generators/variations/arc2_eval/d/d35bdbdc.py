"""Endpoint-selected, simultaneous body-to-center lookup puzzles.

Color roles are recovered geometrically. Lines select their attached endpoints;
all tokens, including other survivors, can supply an original lookup value.
"""
import heapq
import numpy as np


_SOURCE_INPUT_SIGNATURES = {
    (
        "0000000333", "0055555343", "0500000333", "0544400000",
        "5042400666", "0544405616", "0555550666", "0011100000",
        "0013100000", "0011100000",
    ),
    (
        "2220003330", "2620003230", "2220503330", "0005055550",
        "8880500111", "8380050141", "8880500111", "0505444000",
        "0550484000", "0000444000",
    ),
    (
        "1110000444", "1210000434", "1110050444", "0500505000",
        "0505333500", "0050363050", "0000333050", "2220000777",
        "2920000747", "2220000777",
    ),
    (
        "0000000200", "0040002320", "0424000200", "0040000500",
        "0000005000", "0000105030", "0701915333", "7170105030",
        "0705050000", "0550500000",
    ),
    (
        "0004440222", "0054245232", "0054445222", "0005005555",
        "0051110005", "0501315550", "0051115333", "6665005363",
        "6965555333", "6660000000",
    ),
    (
        "0001110666", "0001310616", "0001110666", "5555500050",
        "5222500050", "5242533350", "5222539305", "0550533305",
        "0777055005", "0727000550",
    ),
}

def _rng_for_seed(seed: int) -> np.random.Generator:
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError("seed must be an integer")
    return np.random.default_rng(int(seed) % (1 << 64))


def _check_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _check_count("num_train", num_train, 2, 10)
    _check_count("num_test", num_test, 1, 4)

    if num_train is not None and num_test is not None:
        train_count = int(num_train)
        test_count = int(num_test)
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if num_train is None and num_test is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(test_count + 1, 11))
        return train_count, test_count

    if num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(max(2, test_count + 1), 11))
        return train_count, test_count

    train_count = int(num_train)
    largest_test_count = min(4, train_count - 1)
    if largest_test_count < 1:
        raise ValueError("no legal test count exists for num_train")
    test_count = int(rng.integers(1, largest_test_count + 1))
    return train_count, test_count


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _components(cells, directions=_D4):
    todo = set(cells)
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _support(center, kind, shape):
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (kind == 'square' or dr == 0 or dc == 0)
            and 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]}


def _line_graph(cells):
    # Suppress diagonal shortcuts around a right-angle pixel corner. This also
    # handles the diagonal portions of the original winding lines.
    return {(r, c): {(r + dr, c + dc) for dr, dc in _D8
                     if (r + dr, c + dc) in cells
                     and not (dr and dc and
                              ((r + dr, c) in cells or (r, c + dc) in cells))}
            for r, c in cells}


def _parse_roles(grid, background, line, strict=True):
    tokens = []
    for cells in _components(zip(*np.where((grid != background) & (grid != line)))):
        if not 4 <= len(cells) <= 9:
            raise AssertionError('invalid token area')
        candidates = []
        for center in sorted(cells):
            for kind in ('square', 'cross'):
                if _support(center, kind, grid.shape) != cells:
                    continue
                bodies = {int(grid[p]) for p in cells if p != center}
                if len(bodies) == 1:
                    candidates.append(dict(cells=cells, center=center, kind=kind,
                                           body=bodies.pop(), value=int(grid[center]),
                                           clipped=len(cells) < (9 if kind == 'square' else 5)))
        assert len(candidates) == 1, 'token center must be geometrically unique'
        tokens.extend(candidates)
    assert len(tokens) >= 3
    line_cells = set(zip(*np.where(grid == line)))
    assert len(line_cells) >= 3
    graph = _line_graph(line_cells)
    line_parts = _components(line_cells, _D8)
    ends = {p for p, adjacent in graph.items() if len(adjacent) == 1}
    if strict:
        assert all(len(adjacent) in (1, 2) for adjacent in graph.values()), 'branched line'
        assert all(len(part & ends) == 2 for part in line_parts), 'closed or broken line'
    for token in tokens:
        r, c = token['center']
        token['ends'] = {(r + 2*dr, c + 2*dc) for dr, dc in _D4} & ends
        token['eligible'] = bool(token['ends'])
        assert len(token['ends']) <= 1
    assert any(t['eligible'] for t in tokens)
    assert all(sum(p in t['ends'] for t in tokens) <= 1 for p in ends)
    # A repeated body has a single visible value; no spatial tie-breaking.
    mapping = {}
    for token in tokens:
        body, value = token['body'], token['value']
        assert body not in mapping or mapping[body] == value, 'conflicting donors'
        mapping[body] = value
    return tokens, graph, line_parts


def _interpret(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert sum(counts == counts.max()) == 1
    background = int(colors[np.argmax(counts)])
    candidates = []
    for line in colors:
        if line == background:
            continue
        try:
            parsed = _parse_roles(grid, background, int(line))
        except AssertionError:
            continue
        candidates.append((background, int(line), *parsed))
    assert len(candidates) == 1, 'background/line/token interpretation must be unique'
    return candidates[0]


def _predict_from_input(grid):
    background, line, tokens, graph, parts = _interpret(grid)
    result = grid.copy()
    for i, token in enumerate(tokens):
        donors = [j for j, other in enumerate(tokens)
                  if j != i and other['body'] == token['value']]
        token['donors'] = donors
        token['retained'] = token['eligible'] and bool(donors)
        if token['retained']:
            result[token['center']] = tokens[donors[0]]['value']
        else:
            for p in token['cells']:
                result[p] = background
    return result, (background, line, tokens, graph, parts)


def _features(grid):
    output, (bg, line, tokens, graph, parts) = _predict_from_input(grid)
    retained = [t for t in tokens if t['retained']]
    compatible = [t for t in tokens if not t['eligible'] and t['donors']]
    bend_cells = set()
    diagonal = 0
    for p, adjacent in graph.items():
        diagonal += sum(p[0] != q[0] and p[1] != q[1] for q in adjacent)
        if len(adjacent) == 2:
            a, b = sorted(adjacent)
            if (a[0] + b[0], a[1] + b[1]) != (2*p[0], 2*p[1]):
                bend_cells.add(p)
    near = lambda t, cells: any(abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1
                               for p in t['cells'] for q in cells)
    solid = [t for t in retained if t['body'] != t['value']
             and output[t['center']] == t['body']]
    return dict(
        background=bg, line=line, tokens=len(tokens), lines=len(parts),
        eligible=sum(t['eligible'] for t in tokens), survivors=len(retained),
        failed_ends=sum(t['eligible'] and not t['donors'] for t in tokens),
        bare_ends=sum(len(adj) == 1 for adj in graph.values()) - sum(t['eligible'] for t in tokens),
        identity_colors=len({v for t in tokens for v in (t['body'], t['value'])}),
        repeated_bodies=len(tokens)-len({t['body'] for t in tokens}),
        monochrome=sum(t['body'] == t['value'] for t in tokens),
        unchanged=sum(output[t['center']] == t['value'] for t in retained),
        newly_solid=len(solid), compatible_distractors=len(compatible),
        isolated_compatible=sum(not near(t, graph) for t in compatible),
        interior_compatible=sum(near(t, graph) for t in compatible),
        bend_compatible=sum(near(t, bend_cells) for t in compatible),
        remote_donors=sum(any(not tokens[j]['eligible'] for j in t['donors']) for t in retained),
        retained_donors=sum(any(tokens[j]['retained'] for j in t['donors']) for t in retained),
        shared_donors=sum(sum(j in t['donors'] for t in retained) > 1 for j in range(len(tokens))),
        nonrecursive=sum(any(tokens[j]['retained'] and
                             output[tokens[j]['center']] != tokens[j]['value']
                             for j in t['donors']) for t in retained),
        unrelated_ends=sum(len([t for t in retained if t['ends'] & part]) == 2 and
                           all(a['value'] != b['body'] for a in retained for b in retained
                               if a is not b and a['ends'] & part and b['ends'] & part)
                           for part in parts),
        kinds=tuple(sorted({t['kind'] for t in tokens})),
        clipped=sum(t['clipped'] for t in tokens),
        clipped_retained=sum(t['clipped'] for t in retained),
        clipped_donor=sum(t['clipped'] and any(j in a['donors'] for a in retained)
                          for j, t in enumerate(tokens)),
        clipped_unused=sum(t['clipped'] and not t['retained']
                           and not any(j in a['donors'] for a in retained)
                           for j, t in enumerate(tokens)),
        bends=len(bend_cells), diagonal_edges=diagonal//2,
        line_length=len(graph), shape=grid.shape,
        density=round(float(np.mean(grid != bg)), 3),
        layout=_layout_signature([t['center'] for t in tokens]),
    )


def _layout_signature(centers):
    variants = []
    for swap in (False, True):
        for sr in (-1, 1):
            for sc in (-1, 1):
                pts = [(sr*(c if swap else r), sc*(r if swap else c)) for r, c in centers]
                mr, mc = min(r for r, c in pts), min(c for r, c in pts)
                variants.append(tuple(sorted((int(r-mr), int(c-mc)) for r, c in pts)))
    return min(variants)


def _relations(rng, mode, line_limit, allow_solid):
    lines = 1 if mode == 'remote' else int(rng.integers(1, line_limit + 1))
    if mode == 'chain':
        lines = line_limit  # A training witness for every allowed line count.
    eligible = int(rng.integers(lines if mode == 'random' else max(2, lines), 2*lines + 1))
    if mode == 'remote':
        rel = [(0, 1), (2, 3), (1, 1), (3, 4), (4, 1), (1, 1)]
        mapping = {0: 1, 1: 1, 2: 3, 3: 4, 4: 1}
    elif mode in ('chain', 'mutual'):
        mapping = {0: 1, 1: 2, 2: 3, 4: 1}
        if mode == 'mutual':
            mapping[1] = 0
        rel = [(0, 1), (1, mapping[1])]
        rel += [(int(b), mapping[int(b)]) for b in rng.choice(list(mapping), eligible-2)]
        rel += [(2, 3), (4, 1)]
    else:
        n = int(rng.integers(max(3, eligible+1), 13))
        k = int(rng.integers(3, min(8, n+2)+1))
        for _ in range(64):
            bodies = [int(v) for v in rng.integers(0, k, size=n)]
            mapping = {b: int(rng.integers(k)) for b in sorted(set(bodies))}
            mapping[bodies[0]] = bodies[-1]
            rel = [(b, mapping[b]) for b in bodies]
            newly_solid = any(a != b and b in mapping and mapping[b] == a
                              for a, b in rel[:eligible])
            if allow_solid or not newly_solid:
                break
        else:
            rel = [(0, 1)]*eligible + [(1, 2)]*(n-eligible)
        return rel, lines, eligible
    extra = int(rng.integers(0, min(3, 12-len(rel))+1))
    rel += [(int(b), mapping[int(b)]) for b in rng.choice(list(mapping), extra)]
    return rel, lines, eligible


def _route(rng, shape, occupied, existing, start, end, obstacles):
    h, w = shape
    blocked = set(occupied) | set(obstacles)
    for r, c in existing:
        blocked.add((r, c))
        blocked.update((r+dr, c+dc) for dr, dc in _D8)
    if start in blocked or end in blocked or start == end:
        return None
    # All step costs are below the cost of two steps. A shortest route therefore
    # cannot contain nonconsecutive eight-neighbor contacts or genuine branches.
    costs = rng.uniform(1., 1.8, size=shape)
    queue = [(0., start)]
    distance, previous = {start: 0.}, {}
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance[p]:
            continue
        if p == end:
            path = [p]
            while p != start:
                p = previous[p]
                path.append(p)
            return path[::-1] if len(path) >= 3 else None
        for dr, dc in _D8:
            q = p[0]+dr, p[1]+dc
            if not (0 <= q[0] < h and 0 <= q[1] < w) or q in blocked:
                continue
            new = cost + float(costs[q])
            if new < distance.get(q, float('inf')):
                distance[q] = new
                previous[q] = p
                heapq.heappush(queue, (new, q))
    return None


def _groups(rng, lines, eligible):
    order = list(map(int, rng.permutation(eligible)))
    groups = [[order.pop()] for _ in range(lines)]
    for i in rng.permutation(lines):
        if order:
            groups[int(i)].append(order.pop())
    return groups


def _layout(rng, n, lines, eligible, kinds, clip_count, compact, witness_near=False):
    groups = _groups(rng, lines, eligible)
    for attempt in range(28):
        low = 9 if n <= 5 else (12 if n <= 8 else 16)
        if compact:
            high = min(21, low+5)
        else:
            high = 30
        h = int(rng.integers(low, high+1))
        w = int(rng.integers(low, high+1))
        if not compact and rng.random() < .06:
            h = w = 30
        shape = h, w
        clipped_indices = set(map(int, rng.choice(n, clip_count, replace=False)))
        centers = [None]*n
        order = sorted(range(n), key=lambda i: i not in clipped_indices)
        good = True
        for i in order:
            if i in clipped_indices:
                candidates = ([(0, c) for c in range(1, w-1)] + [(h-1, c) for c in range(1, w-1)]
                              + [(r, 0) for r in range(1, h-1)] + [(r, w-1) for r in range(1, h-1)])
            else:
                candidates = [(r, c) for r in range(1, h-1) for c in range(1, w-1)]
            candidates = [p for p in candidates if all(q is None or
                          max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 4 for q in centers)]
            if not candidates:
                good = False
                break
            centers[i] = candidates[int(rng.integers(len(candidates)))]
        if not good:
            continue
        supports = [_support(p, kind, shape) for p, kind in zip(centers, kinds)]
        occupied = set().union(*supports)
        gray = set()
        for group in groups:
            routed = False
            for _ in range(12):
                contacts = []
                for i in group:
                    r, c = centers[i]
                    stubs = [(r+2*dr, c+2*dc) for dr, dc in _D4]
                    stubs = [p for p in stubs if 0 <= p[0] < h and 0 <= p[1] < w and p not in occupied
                             and all(j == i or all(max(abs(p[0]-q[0]),abs(p[1]-q[1])) > 1
                                                   for q in cells)
                                     for j,cells in enumerate(supports))]
                    if not stubs:
                        break
                    contacts.append(stubs[int(rng.integers(len(stubs)))])
                if len(contacts) != len(group):
                    continue
                if len(contacts) == 1:
                    free = [(r, c) for r in range(h) for c in range(w)
                            if (r,c) not in occupied and all(max(abs(r-a), abs(c-b)) >= 4 for a,b in centers)]
                    if not free:
                        continue
                    contacts.append(free[int(rng.integers(len(free)))])
                obstacles = set()
                for _ in range(int(rng.integers(0, 5))):
                    r, c = int(rng.integers(h)), int(rng.integers(w))
                    obstacles.update((r+dr, c+dc) for dr in range(int(rng.integers(1, 5)))
                                     for dc in range(int(rng.integers(1, 5))))
                obstacles.difference_update(contacts)
                path = _route(rng, shape, occupied, gray, *contacts, obstacles)
                if path is None:
                    continue
                ends = {path[0], path[-1]}
                attached = {i for i, (r,c) in enumerate(centers)
                            if any((r+2*dr,c+2*dc) in ends for dr,dc in _D4)}
                if attached != set(group):
                    continue
                gray.update(path)
                routed = True
                break
            if not routed:
                good = False
                break
        if good and witness_near:
            # A compatible non-endpoint must touch an interior stretch in a
            # training witness. Otherwise 'update every touching token' remains
            # an accidental explanation when all distractors happen to be remote.
            ends = {p for p, adj in _line_graph(gray).items() if len(adj) == 1}
            candidates = sorted({(r+2*dr,c+2*dc) for r,c in gray-ends for dr,dc in _D4})
            candidates = [p for p in candidates if 1 <= p[0] < h-1 and 1 <= p[1] < w-1
                          and all(j == 4 or max(abs(p[0]-q[0]),abs(p[1]-q[1])) >= 4
                                  for j,q in enumerate(centers))
                          and not any((p[0]+2*dr,p[1]+2*dc) in ends for dr,dc in _D4)
                          and not (_support(p,kinds[4],shape) & gray)]
            if candidates:
                centers[4] = candidates[int(rng.integers(len(candidates)))]
                supports[4] = _support(centers[4],kinds[4],shape)
            else:
                good = False
        if good:
            return shape, centers, supports, gray
    # Guaranteed spacious fallback: separated straight lanes, used only after
    # bounded routing attempts. All semantic roles and token morphologies survive.
    shape = (30, 30)
    centers = [None]*n
    gray = set()
    for k, group in enumerate(groups):
        r = 3 + 9*k
        centers[group[0]] = (r, 2)
        if len(group) == 2:
            centers[group[1]] = (r, 27)
        gray.update((r,c) for c in range(4, 26))
    if witness_near:
        centers[4] = (5,13)
    unused = [(r,c) for r in (7,16,25) for c in (2,9,16,27)
              if all(q is None or max(abs(r-q[0]),abs(c-q[1])) >= 4 for q in centers)]
    for i in range(n):
        if centers[i] is None:
            centers[i] = unused.pop(0)
    return shape, centers, [_support(p,k,shape) for p,k in zip(centers,kinds)], gray


def _make_pair(rng, parameters, mode, regime):
    rel, lines, eligible = _relations(rng, mode, parameters['line_limit'], parameters['solid'])
    n = len(rel)
    morphology = parameters['morphology']
    if morphology == 'shift':
        kind = 'square' if regime == 'source_train' else str(rng.choice(['square','cross']))
        kinds = [kind]*n
    elif morphology in ('square', 'cross'):
        kinds = [morphology]*n
    else:
        if rng.random() < .5:
            kinds = [str(rng.choice(['square','cross']))]*n
        else:
            kinds = [str(k) for k in rng.choice(['square','cross'], n)]
    clip_probability = parameters['clip_probability']
    if morphology == 'shift' and regime == 'source_train':
        clip_probability = 0.
    clip_count = (int(rng.choice([1,1,1,2,3])) if rng.random() < clip_probability else 0)
    shape, centers, supports, gray = _layout(rng, n, lines, eligible, kinds,
                                            min(n,clip_count), rng.random() < .35,
                                            witness_near=mode == 'remote')
    if parameters['varied_roles']:
        palette = list(map(int, rng.permutation(10)))
    else:
        bg, line = parameters['roles']
        palette = [bg, line] + list(map(int, rng.permutation([c for c in range(10) if c not in (bg,line)])))
    bg, line = palette[:2]
    identities = palette[2:]
    grid = np.full(shape, bg, dtype=np.int8)
    for p in gray:
        grid[p] = line
    for i, (body, value) in enumerate(rel):
        for p in supports[i]:
            grid[p] = identities[body]
        grid[centers[i]] = identities[value]
    # Construction uses the sampled visible relations simultaneously. Validation
    # separately segments the returned grids and rediscovers their line endpoints.
    output = grid.copy()
    for i, (body, value) in enumerate(rel):
        donors = [v for j, (b,v) in enumerate(rel) if j != i and b == value]
        if i < eligible and donors:
            output[centers[i]] = identities[donors[0]]
        else:
            for p in supports[i]:
                output[p] = bg
    rotation = int(rng.integers(4))
    grid, output = np.rot90(grid,rotation), np.rot90(output,rotation)
    if rng.random() < .5:
        grid, output = np.fliplr(grid), np.fliplr(output)
    return dict(input=grid.copy(), output=output.copy())


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output'])) for p in puzzle[s])
                 for s in ('train','test'))


def _check_training(train, test):
    assert any(f['compatible_distractors'] for f in train), 'line selection is not demonstrated'
    assert any(f['interior_compatible'] for f in train), 'endpoint versus interior selection is not demonstrated'
    assert any(f['remote_donors'] >= 2 and f['unrelated_ends'] for f in train)
    assert any(f['retained_donors'] and f['nonrecursive'] for f in train), 'simultaneity is not demonstrated'
    assert any(f['monochrome'] and f['unchanged'] for f in train)
    assert any(f['repeated_bodies'] for f in train)
    assert max(f['lines'] for f in test) <= max(f['lines'] for f in train)
    if any(f['newly_solid'] for f in test):
        assert any(f['newly_solid'] for f in train), 'solid-result extension requires a witness'
    # These are pair-local visible roles, not a memorized identity dictionary.
    if len({(f['background'],f['line']) for f in train}) == 1:
        assert {(f['background'],f['line']) for f in test} <= {(train[0]['background'],train[0]['line'])}
    assert len({f['layout'] for f in train+test}) > 1


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s], list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs, statistics = set(), set(), {}
    for split in ('train','test'):
        statistics[split] = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input','output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            ik, pk = _grid_key(inp), (_grid_key(inp), _grid_key(out))
            assert ik not in seen_inputs and pk not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            if inp.shape == (10,10):
                assert tuple(''.join(map(str,row)) for row in inp.tolist()) not in _SOURCE_INPUT_SIGNATURES
            predicted, _ = _predict_from_input(inp)
            assert np.array_equal(predicted, out), 'endpoint selection or simultaneous lookup mismatch'
            f = _features(inp)
            assert 3 <= f['tokens'] <= 12 and 1 <= f['lines'] <= 3
            assert 1 <= f['survivors'] < f['tokens']
            statistics[split].append(f)
    _check_training(statistics['train'],statistics['test'])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = _rng_for_seed(seed)
    nt, ne = _resolve_counts(rng, num_train, num_test)
    parameters = dict(
        roles=tuple(map(int, rng.choice(10, 2, replace=False))),
        varied_roles=bool(rng.random() < .65),
        morphology=str(rng.choice(['square','cross','shift','mixed'], p=[.2,.12,.23,.45])),
        clip_probability=float(rng.choice([0.,.25,.55,.8], p=[.12,.25,.4,.23])),
        line_limit=int(rng.choice([1,2,3], p=[.5,.32,.18])),
        solid=bool(nt >= 3 and rng.random() < .4),
    )
    modes = ['remote','chain'] + (['mutual'] if parameters['solid'] else [])
    modes += ['random']*(nt-len(modes))
    rng.shuffle(modes)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            mode = modes[i] if split == 'train' else 'random'
            for _ in range(32):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _make_pair(pair_rng, parameters, mode,
                                  'source_train' if split == 'train' else 'target_test')
                try:
                    predicted, _ = _predict_from_input(pair['input'])
                    assert np.array_equal(predicted,pair['output'])
                    f = _features(pair['input'])
                    assert 0 < f['survivors'] < f['tokens']
                    if mode == 'chain':
                        assert f['retained_donors'] and f['nonrecursive']
                    if mode == 'remote':
                        assert f['remote_donors'] >= 2 and f['unrelated_ends']
                    if mode == 'mutual':
                        assert f['newly_solid']
                    key = _grid_key(pair['input'])
                    assert key not in seen
                    if parameters['varied_roles'] and split == 'train' and i == 1:
                        first = _features(puzzle['train'][0]['input'])
                        assert (f['background'],f['line']) != (first['background'],first['line'])
                except AssertionError:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError('could not produce an unambiguous pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    rows = [_features(pair['input']) for p in puzzles for s in ('train','test') for pair in p[s]]
    assert len({f['layout'] for f in rows}) > 150
    assert len({f['shape'] for f in rows}) > 80
    assert len({f['tokens'] for f in rows}) >= 8
    assert {1,2,3} <= {f['lines'] for f in rows}
    assert len({f['survivors'] for f in rows}) >= 5
    assert len({f['identity_colors'] for f in rows}) >= 5
    assert len({(f['background'],f['line']) for f in rows}) >= 40
    assert any(max(f['shape']) <= 12 for f in rows)
    assert any(30 in f['shape'] for f in rows)
    assert any(f['shape'] == (30,30) for f in rows)
    for feature in ('failed_ends','bare_ends','repeated_bodies','monochrome','unchanged',
                    'newly_solid','compatible_distractors','isolated_compatible',
                    'interior_compatible','bend_compatible','remote_donors','retained_donors',
                    'shared_donors','nonrecursive','unrelated_ends','clipped_retained',
                    'clipped_donor','clipped_unused','diagonal_edges'):
        assert sum(f[feature] > 0 for f in rows) >= 5, feature
    assert sum(f['clipped'] == 1 for f in rows) >= 20
    assert sum(len(f['kinds']) == 2 for f in rows) >= 20
    assert sum(f['bends'] >= 3 for f in rows) >= 50
    assert sum(len(f['kinds']) == 2 and f['clipped'] and f['retained_donors'] for f in rows) >= 3
    within = [[_features(pair['input']) for split in ('train','test') for pair in p[split]]
              for p in puzzles]
    for feature in ('tokens','shape','layout'):
        assert sum(len({f[feature] for f in group}) > 1 for group in within) >= 40
    for feature in ('kinds','clipped','background','line'):
        assert sum(len({f[feature] for f in group}) > 1 for group in within) >= 15
    assert sum(len({f['background'] for f in group}) == 1 for group in within) >= 5
    assert any(all(f['lines'] == 1 and len(f['kinds']) == 1 for f in group) for group in within)
    for split in ('train','test'):
        split_rows = [_features(pair['input']) for p in puzzles for pair in p[split]]
        assert any('cross' in f['kinds'] for f in split_rows)
        assert any(f['clipped'] for f in split_rows)
    for counts in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        p = generate_puzzle(1000+sum(c or 0 for c in counts), *counts)
        for split, count in zip(('train','test'),counts):
            assert count is None or len(p[split]) == count
    for counts in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4)):
        try:
            generate_puzzle(7,*counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Negative semantic checks: preserve the line and use original donor values.
    pair = puzzles[0]['train'][0]
    bad = {s: [dict(input=q['input'].copy(),output=q['output'].copy()) for q in puzzles[0][s]]
           for s in ('train','test')}
    _, line, _, graph, _ = _interpret(pair['input'])
    cell = min(graph)
    bad['train'][0]['output'][cell] = (line+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('line corruption accepted')
