"""Fragmented cables: port-defined objects, unary ranks, in-place propagation.

Only standard-library modules and NumPy are used. No parent code or fixture is
loaded. Generation records construct labels; solve freshly decodes visible pixels.
"""
from collections import defaultdict

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.tobytes()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _neighbors(p):
    r, c = p
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _layout(panels, order, shape, divider):
    nr, nc = shape
    h, w = panels[0].shape
    result = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, index in enumerate(order):
        r, c = divmod(slot, nc)
        result[r * (h + 1):r * (h + 1) + h,
               c * (w + 1):c * (w + 1) + w] = panels[index]
    return result


def _partition(grid):
    candidates = []
    h, w = grid.shape
    for divider in np.unique(grid):
        rows = list(map(int, np.flatnonzero(np.all(grid == divider, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(grid == divider, axis=0))))
        if not rows and not cols:
            continue
        rb, cb = [-1] + rows + [h], [-1] + cols + [w]
        hs = [b - a - 1 for a, b in zip(rb, rb[1:])]
        ws = [b - a - 1 for a, b in zip(cb, cb[1:])]
        if min(hs + ws) < 4 or len(set(hs)) != 1 or len(set(ws)) != 1:
            continue
        if not 4 <= len(hs) * len(ws) <= 12:
            continue
        panels, origins = [], []
        for r in rb[:-1]:
            for c in cb[:-1]:
                panels.append(grid[r + 1:r + 1 + hs[0], c + 1:c + 1 + ws[0]].copy())
                origins.append((r + 1, c + 1))
        if any(np.any(p == divider) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels
                   for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        bg = next(iter(corners))
        if any(not np.any(p != bg) for p in panels):
            continue
        candidates.append(dict(panels=panels, origins=origins, background=bg,
                               divider=int(divider), layout=(len(hs), len(ws)),
                               tile_shape=(hs[0], ws[0])))
    _require(len(candidates) == 1, "unique equal-tile partition required")
    return candidates[0]


def _trace(panel, bg):
    """Return the two endpoints and the entire induced path, in either order."""
    cells = {tuple(map(int, x)) for x in np.argwhere(panel != bg)}
    _require(len(cells) >= 2, "empty or singleton fragment")
    adjacency = {p: [q for q in _neighbors(p) if q in cells] for p in cells}
    ends = sorted(p for p in cells if len(adjacency[p]) == 1)
    _require(len(ends) == 2 and all(1 <= len(v) <= 2 for v in adjacency.values()),
             "fragment must be an unbranched induced path")
    path, previous, current = [], None, ends[0]
    while True:
        path.append(current)
        nxt = [p for p in adjacency[current] if p != previous]
        if not nxt:
            break
        previous, current = current, nxt[0]
        _require(len(path) <= len(cells), "cycle")
    _require(len(path) == len(cells), "disconnected fragment")
    return path


def _prefix(values):
    color = int(values[0])
    n = 1
    while n < len(values) and values[n] == color:
        n += 1
    _require(n < len(values), "terminal has no payload")
    payload = int(values[n])
    _require(np.all(values[n:] == payload), "terminal payload is not uniform")
    return color, n, payload


def _canonical_decode(grid):
    """Decode a candidate with sources facing right, entirely from pixels."""
    data = _partition(grid)
    panels, bg = data['panels'], data['background']
    fragments, incoming, outgoing = [], {}, {}
    for i, panel in enumerate(panels):
        h, w = panel.shape
        _require(np.all(panel[0] == bg) and np.all(panel[-1] == bg), "wrong common axis")
        path = _trace(panel, bg)
        left = list(map(int, np.flatnonzero(panel[:, 0] != bg)))
        right = list(map(int, np.flatnonzero(panel[:, -1] != bg)))
        _require(len(left) <= 1 and len(right) <= 1 and (left or right), "singleton ports required")
        # Each port is an endpoint, never a side touch of the path.
        ports = {(r, 0) for r in left} | {(r, w - 1) for r in right}
        _require(ports <= {path[0], path[-1]}, "port must be a path endpoint")
        if left:
            _require(left[0] not in incoming, "incoming port positions must be unique")
            incoming[left[0]] = i
        if right:
            _require(right[0] not in outgoing, "outgoing port positions must be unique")
            outgoing[right[0]] = i
        if left and right:
            if path[0][1] != 0:
                path.reverse()
            values = np.array([panel[p] for p in path])
            _require(np.all(values == values[0]), "intermediate must have uniform payload")
            info = dict(kind='middle', payload=int(values[0]))
        else:
            # Always trace a terminal from its interior endpoint to its port.
            if path[0] in ports:
                path.reverse()
            _require(0 < path[0][0] < h - 1 and 0 < path[0][1] < w - 1,
                     "terminal cap must be strictly interior")
            cap, n, payload = _prefix(np.array([panel[p] for p in path]))
            if right:
                _require(1 <= n <= 3, "source unary rank must be 1..3")
                info = dict(kind='source', cap=cap, rank=n, payload=payload)
            else:
                _require(n == 1, "far cap must contain exactly one cell")
                info = dict(kind='far', cap=cap, payload=payload)
        info.update(tile=i, path=path, length=len(path),
                    left=left[0] if left else None, right=right[0] if right else None)
        fragments.append(info)
    _require(set(incoming) == set(outgoing), "unmatched port")
    sources = [f['tile'] for f in fragments if f['kind'] == 'source']
    _require(2 <= len(sources) <= 6, "need two to six complete cables")
    cables, covered = [], set()
    for start in sources:
        ids, i = [], start
        while True:
            _require(i not in covered, "cycle or shared fragment")
            ids.append(i)
            covered.add(i)
            f = fragments[i]
            if f['right'] is None:
                break
            i = incoming[f['right']]
        _require(2 <= len(ids) <= 4, "cable must have two to four fragments")
        first, last = fragments[ids[0]], fragments[ids[-1]]
        _require(last['kind'] == 'far' and first['cap'] == last['cap'], "matching cable caps")
        _require(all(fragments[j]['kind'] == 'middle' for j in ids[1:-1]), "bad cable interior")
        cables.append(dict(tiles=ids, cap=first['cap'], rank=first['rank'], source=start,
                           lengths=[fragments[j]['length'] for j in ids],
                           length=sum(fragments[j]['length'] for j in ids)))
    _require(len(covered) == len(panels), "orphan fragments or cycles")
    groups = defaultdict(list)
    for cable in cables:
        groups[cable['cap']].append(cable)
    _require(1 <= len(groups) <= 3, "one to three control families")
    controls = set(groups)
    _require(all(f['payload'] not in controls for f in fragments), "control and payload palettes overlap")
    for family in groups.values():
        ranks = {c['rank'] for c in family}
        _require(2 <= len(family) <= 4 and len(ranks) == 1, "family population or inconsistent unary keys")
        _require(next(iter(ranks)) <= len(family), "rank does not exist")
        _require(len({c['length'] for c in family}) == len(family), "tied cable lengths")
    data.update(fragments=fragments, cables=cables, groups=dict(groups))
    return data


def _selected(data, rule='true'):
    selected = []
    for family in data['groups'].values():
        rank = family[0]['rank']
        if rule == 'longest':
            rank = 1
        elif rule == 'shortest':
            rank = len(family)
        spatial = lambda c: c.get('display_source', c['source'])
        if rule == 'spatial':
            ordered = sorted(family, key=spatial)
        elif rule == 'reverse_spatial':
            ordered = sorted(family, key=spatial, reverse=True)
        else:
            score = {
                'fragment': lambda c: max(c['lengths']),
                'terminals': lambda c: c['lengths'][0] + c['lengths'][-1],
                'tiles': lambda c: len(c['tiles']),
            }.get(rule, lambda c: c['length'])
            ordered = sorted(family, key=lambda c: ((score(c) if rule == 'ascending' else -score(c)),
                                                    spatial(c)))
        selected.append(ordered[rank - 1])
    return selected


def _paint(grid, data, chosen, terminals_only=False):
    out = grid.copy()
    for cable in chosen:
        tiles = cable['tiles']
        if terminals_only:
            tiles = [tiles[0], tiles[-1]]
        for i in tiles:
            r0, c0 = data['origins'][i]
            for r, c in data['fragments'][i]['path']:
                out[r0 + r, c0 + c] = cable['cap']
    return out


def _decode(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        canonical = np.ascontiguousarray(np.rot90(grid, -rotation))
        try:
            data = _canonical_decode(canonical)
        except AssertionError:
            continue
        # Spatial diagnostic rules refer to the returned input's reading order,
        # not to an incidental rotation chosen by the decoder.
        h, w = canonical.shape
        for cable in data['cables']:
            r, c = data['origins'][cable['source']]
            r += data['tile_shape'][0] // 2
            c += data['tile_shape'][1] // 2
            physical = ((r, c), (w - 1 - c, r), (h - 1 - r, w - 1 - c), (c, h - 1 - r))[rotation]
            cable['display_source'] = physical[0] * grid.shape[1] + physical[1]
        expected = np.ascontiguousarray(np.rot90(_paint(canonical, data, _selected(data)), rotation))
        candidates.append((rotation, canonical, data, expected))
    _require(candidates, "no valid cable decoding")
    _require(len({_key(item[3]) for item in candidates}) == 1, "ambiguous cable answer")
    # All-one instructions may admit a harmless reversal of every cable.
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover cables, read their unary caps, and fill selected pieces in place."""
    return _decode(input_grid)[3].copy()


def _random_path(rng, h, w, left, right, rank):
    """Bounded randomized induced-path search, with a constructive fallback."""
    if left is not None and right is not None:
        start, target = (left, 0), (right, w - 1)
    else:
        row = left if left is not None else right
        edge = 0 if left is not None else w - 1
        start = (row, edge)
        minimum = max(3, rank + 2)
        choices = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
                   if abs(r - row) + abs(c - edge) >= minimum - 1]
        target = _pick(rng, choices)
    allowed = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)} | {start, target}
    path, visited, budget = [start], {start}, [700]
    # A positive coefficient favors detours; a negative one favors short paths.
    bias = float(rng.uniform(-1.4, 0.9))

    def walk(current):
        budget[0] -= 1
        if current == target:
            return True
        if budget[0] <= 0:
            return False
        options = [q for q in _neighbors(current) if q in allowed and q not in visited
                   and sum(v in visited for v in _neighbors(q)) == 1]
        weights = {q: float(rng.uniform(0, 2.5)) + bias * (abs(q[0] - target[0]) + abs(q[1] - target[1]))
                   for q in options}
        for q in sorted(options, key=lambda q: weights[q], reverse=True):
            path.append(q)
            visited.add(q)
            if walk(q):
                return True
            visited.remove(q)
            path.pop()
            if budget[0] <= 0:
                break
        return False

    if walk(start):
        return path
    # Horizontal, vertical, horizontal: no accidental self-contacts.
    hub = int(rng.integers(1, w - 1)) if left is not None and right is not None else target[1]
    waypoints = [start, (start[0], hub), (target[0], hub), target]
    path = [start]
    for endpoint in waypoints[1:]:
        while path[-1] != endpoint:
            r, c = path[-1]
            dr = int(np.sign(endpoint[0] - r))
            dc = int(np.sign(endpoint[1] - c)) if not dr else 0
            path.append((r + dr, c + dc))
    return path


def _configurations(cables, max_parts, force_four):
    """Feasible factor layouts and globally unique singleton-port budgets."""
    options = []
    for n in range(2 * cables, min(12, cables * max_parts) + 1):
        joins = n - cables
        if force_four and (max_parts < 4 or n < 2 * cables + 2):
            continue
        for h in range(max(6, joins + 2), 10):
            for w in range(4, 10):
                layouts = [(nr, n // nr) for nr in range(1, n + 1) if n % nr == 0
                           and nr * (h + 1) - 1 <= 30 and (n // nr) * (w + 1) - 1 <= 30]
                if layouts:
                    options.append((n, h, w, layouts))
    return options


def _make_pair(rng, spec, anchor):
    populations, ranks = spec['populations'], spec['ranks']
    count = sum(populations)
    options = _configurations(count, spec['max_parts'], spec.get('force_four', False))
    _require(options, "infeasible scene plan")
    # Choose total count first so the rectangle catalog does not weight it.
    n = int(_pick(rng, sorted({o[0] for o in options})))
    n, h, w, layouts = _pick(rng, [o for o in options if o[0] == n])
    shape = _pick(rng, layouts)
    parts = [2] * count
    extra = n - 2 * count
    if spec.get('force_four', False):
        parts[0] = 4
        extra -= 2
    while extra:
        i = int(_pick(rng, [i for i in range(count) if parts[i] < spec['max_parts']]))
        parts[i] += 1
        extra -= 1
    rng.shuffle(parts)
    colors = [int(c) for c in rng.permutation(10) if c != anchor]
    controls = [anchor] + colors[:len(populations) - 1]
    colors = [c for c in colors if c not in controls]
    bg, divider = colors[:2]
    payloads = colors[2:2 + int(rng.integers(1, min(3, len(colors) - 2) + 1))]
    rows = list(map(int, rng.choice(np.arange(1, h - 1), n - count, replace=False)))
    panels, winner_tiles = [], []
    used_lengths = defaultdict(set)
    cursor, chain_index = 0, 0
    for family, population in enumerate(populations):
        family_records = []
        for _ in range(population):
            k = parts[chain_index]
            chain_index += 1
            links = rows[cursor:cursor + k - 1]
            cursor += k - 1
            for attempt in range(100):
                paths = [_random_path(rng, h, w, None if j == 0 else links[j - 1],
                                      None if j == k - 1 else links[j], ranks[family] if j == 0 else 1)
                         for j in range(k)]
                length = sum(map(len, paths))
                if length not in used_lengths[family]:
                    break
            else:
                raise RuntimeError("could not draw distinct cable lengths")
            used_lengths[family].add(length)
            ids = []
            for j, path in enumerate(paths):
                panel = np.full((h, w), bg, dtype=np.int8)
                payload = int(_pick(rng, payloads))
                for p in path:
                    panel[p] = payload
                if j == 0:
                    # Terminal path was grown from the edge inward.
                    for p in path[-ranks[family]:]:
                        panel[p] = controls[family]
                elif j == k - 1:
                    panel[path[-1]] = controls[family]
                ids.append(len(panels))
                panels.append(panel)
            family_records.append(dict(tiles=ids, length=length, cap=controls[family]))
        chosen = sorted(family_records, key=lambda c: c['length'], reverse=True)[ranks[family] - 1]
        winner_tiles.append((chosen['tiles'], controls[family]))
    output_panels = [p.copy() for p in panels]
    for ids, cap in winner_tiles:
        for i in ids:
            output_panels[i][panels[i] != bg] = cap
    order = list(map(int, rng.permutation(n)))
    inp = _layout(panels, order, shape, divider)
    out = _layout(output_panels, order, shape, divider)
    if bool(rng.integers(2)):
        inp, out = np.flipud(inp), np.flipud(out)
    rotation = spec['rotation']
    return {'input': np.ascontiguousarray(np.rot90(inp, rotation)),
            'output': np.ascontiguousarray(np.rot90(out, rotation))}


_SHORTCUTS = ('longest', 'shortest', 'ascending', 'spatial', 'reverse_spatial',
              'fragment', 'terminals', 'tiles')


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape, "dimensions must be preserved")
    rotation, canonical, data, expected = _decode(inp)
    _require(np.array_equal(out, expected), "incorrect complete-cable fill")
    chosen = _selected(data)
    selected_tiles = {i: c['cap'] for c in chosen for i in c['tiles']}
    canonical_out = np.rot90(out, -rotation)
    # Explicit conservation and change-mask check, beyond array equality.
    _require(np.array_equal(inp == data['background'], out == data['background']), "cable geometry changed")
    _require(np.array_equal(inp == data['divider'], out == data['divider']), "dividers changed")
    for i, panel in enumerate(data['panels']):
        r, c = data['origins'][i]
        after = canonical_out[r:r + panel.shape[0], c:c + panel.shape[1]]
        if i in selected_tiles:
            _require(np.all(after[panel != data['background']] == selected_tiles[i]), "incomplete propagation")
        else:
            _require(np.array_equal(after, panel), "unselected fragment changed")
    true_ids = {c['source'] for c in chosen}
    defeats = {rule for rule in _SHORTCUTS
               if {c['source'] for c in _selected(data, rule)} != true_ids}
    # A proxy ranking is refuted only when no resolution of its ties could
    # place the actual winner at the requested ordinal.
    strict_proxy_conflicts = set()
    scores = {'fragment': lambda c: max(c['lengths']),
              'terminals': lambda c: c['lengths'][0] + c['lengths'][-1],
              'tiles': lambda c: len(c['tiles'])}
    for rule, score in scores.items():
        for family in data['groups'].values():
            winner = next(c for c in family if c['source'] in true_ids)
            higher = sum(score(c) > score(winner) for c in family)
            equal = sum(score(c) == score(winner) for c in family)
            if not higher + 1 <= winner['rank'] <= higher + equal:
                strict_proxy_conflicts.add(rule)
    return dict(rotation=rotation, canonical=canonical, data=data, selected=chosen, defeats=defeats,
                strict_proxy_conflicts=strict_proxy_conflicts,
                intermediate=any(len(c['tiles']) > 2 for c in chosen),
                payload_join=any(data['fragments'][a]['payload'] != data['fragments'][b]['payload']
                                 for cable in data['cables'] for a, b in zip(cable['tiles'], cable['tiles'][1:])))


def _teaching_ok(analyses):
    ranks = {c['rank'] for a in analyses for c in a['data']['cables']}
    cap_ranks = defaultdict(set)
    for a in analyses:
        for c in a['data']['cables']:
            cap_ranks[c['cap']].add(c['rank'])
    return ({1, 2} <= ranks
            and any({1, 2} <= s for s in cap_ranks.values())
            and any(c['rank'] == 2 and len(fam) >= 3 for a in analyses
                    for fam in a['data']['groups'].values() for c in fam)
            and any(len(a['data']['groups']) >= 2 for a in analyses)
            and any(a['intermediate'] for a in analyses)
            and any(a['payload_join'] for a in analyses)
            and {'fragment', 'terminals', 'tiles'} <=
                set().union(*(a['strict_proxy_conflicts'] for a in analyses))
            and set(_SHORTCUTS) <= set().union(*(a['defeats'] for a in analyses)))


def _teaching_features(analysis):
    """Finite coverage signature for retaining complementary demonstrations."""
    result = {'defeat:' + name for name in analysis['defeats']}
    result.update('strict:' + name for name in analysis['strict_proxy_conflicts'])
    if analysis['intermediate']:
        result.add('intermediate')
    if analysis['payload_join']:
        result.add('payload_join')
    return frozenset(result)


def validate_generated_puzzle(puzzle):
    """Check semantics, conservation, and contrastive teaching from actual grids."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    analyses, seen = {}, set()
    for split in ('train', 'test'):
        analyses[split] = []
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            _require(np.any(inp != out), "identity example")
            analyses[split].append(_analyze(pair))
    _require(_teaching_ok(analyses['train']), "training leaves a documented shortcut or omits a concept")
    train_ranks = {c['rank'] for a in analyses['train'] for c in a['data']['cables']}
    for a in analyses['test']:
        for c in a['data']['cables']:
            _require(c['rank'] in train_ranks or set(range(1, c['rank'])) <= train_ranks,
                     "test ordinal lacks demonstrated progression")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be {low}..{high}')
    return int(value)


def _spec(rng, shift, split, orientation, witness=None):
    target = split == 'test'
    max_parts = 4 if shift != 'fragments' or target else 3
    if witness == 1:
        populations, ranks = [2, 2], [1, int(rng.integers(1, 3))]
    elif witness == 2:
        populations, ranks = [3], [2]
    elif target and shift == 'families':
        populations = [2, 2, 2]
        ranks = list(map(int, rng.integers(1, 3, 3)))
    else:
        max_families = 2 if not target or shift != 'families' else 3
        f = int(rng.integers(1, max_families + 1))
        populations = [int(rng.integers(2, 5)) for _ in range(f)]
        while sum(populations) > (4 if target and shift == 'fragments' else 6):
            choices = [i for i in range(f) if populations[i] > 2]
            populations[int(_pick(rng, choices))] -= 1
        limit = 2 if shift == 'rank' and not target else 3
        ranks = [int(rng.integers(1, min(p, limit) + 1)) for p in populations]
        if target and shift == 'rank':
            # A requested third rank may be last of three or intermediate of four.
            populations[0] = int(rng.integers(3, 5))
            if sum(populations) > 6:
                populations = [populations[0]]
                ranks = [3]
            else:
                ranks[0] = 3
    return dict(populations=populations, ranks=ranks, max_parts=max_parts,
                force_four=target and shift == 'fragments',
                rotation=int(rng.integers(4)) if orientation is None else orientation)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate a reproducible puzzle with two contrastive teaching anchors."""
    a = _count(num_train, 'num_train', 2, 10)
    b = _count(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shift = _pick(rng, ('rank', 'fragments', 'families'))
    anchor = int(rng.integers(10))
    orientation = None if bool(rng.integers(2)) else int(rng.integers(4))
    # Retain a Pareto frontier of actual demonstrations. Discarding a useful
    # counterexample because its newly sampled companion misses another feature
    # makes rare-but-compatible combinations needlessly unreliable.
    pools, teaching = [[], []], None
    for _ in range(300):
        for index, witness in enumerate((1, 2)):
            pair = _make_pair(rng, _spec(rng, shift, 'train', orientation, witness), anchor)
            analysis = _analyze(pair)
            features = _teaching_features(analysis)
            for _, other_pair, other_analysis in pools[1 - index]:
                if _teaching_ok([analysis, other_analysis]):
                    teaching = [pair, other_pair] if index == 0 else [other_pair, pair]
                    break
            if teaching is not None:
                break
            if not any(features <= old_features for old_features, _, _ in pools[index]):
                pools[index] = [(fs, p, a) for fs, p, a in pools[index] if not fs <= features]
                pools[index].append((features, pair, analysis))
        if teaching is not None:
            break
    if teaching is None:
        raise RuntimeError('could not construct contrastive demonstrations')
    puzzle, seen = {'train': teaching, 'test': []}, {_key(p['input']) for p in teaching}
    for split, count in (('train', a - 2), ('test', b)):
        for _ in range(count):
            for attempt in range(100):
                pair = _make_pair(rng, _spec(rng, shift, split, orientation), anchor)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('could not construct a distinct input')
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    permutation = list(map(int, rng.permutation(a)))
    puzzle['train'] = [puzzle['train'][i] for i in permutation]
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed semantic, variation, and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "42 distinct puzzles")
    shapes, tile_shapes, layouts, paths, colors, family_counts = set(), set(), set(), set(), set(), set()
    totals, parts, ranks, populations, directions = set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze(pair)
                d = a['data']
                shapes.add(pair['input'].shape)
                tile_shapes.add(d['tile_shape'])
                layouts.add(d['layout'])
                family_counts.add(len(d['groups']))
                colors.add((d['background'], d['divider']))
                directions.add(a['rotation'])
                for panel in d['panels']:
                    paths.add(_key((panel != d['background']).astype(np.int8)))
                for family in d['groups'].values():
                    populations.add(len(family))
                    for c in family:
                        totals.add(c['length'])
                        parts.add(len(c['tiles']))
                        ranks.add(c['rank'])
    _require(len(shapes) >= 20 and len(tile_shapes) >= 12 and len(layouts) >= 5, "tile/layout diversity")
    _require(len(paths) >= 250 and len(totals) >= 25, "path/aggregate-length diversity")
    _require(parts == {2, 3, 4} and ranks == {1, 2, 3}, "fragment and ordinal ranges")
    _require(family_counts == {1, 2, 3} and populations == {2, 3, 4}, "family/population variation")
    _require(len(colors) >= 35 and directions == {0, 1, 2, 3}, "palette/direction variation")
    _require(_puzzle_key(puzzles[17]) == _puzzle_key(generate_puzzle(17)), "determinism")
    # The same pixels must solve under palette permutations, dihedral transforms,
    # and a new factor layout / permutation of their whole tiles.
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        for rotation in range(4):
            for reflect in (False, True):
                inp = np.rot90(np.fliplr(pair['input']) if reflect else pair['input'], rotation)
                out = np.rot90(np.fliplr(pair['output']) if reflect else pair['output'], rotation)
                palette = np.array([8, 2, 7, 0, 5, 1, 9, 3, 6, 4], dtype=np.int8)
                _require(np.array_equal(solve(palette[inp]), palette[out]), "orientation/color covariance")
        rotation, canonical, d, _ = _decode(pair['input'])
        expected = np.rot90(pair['output'], -rotation)
        output_panels = [expected[r:r + d['tile_shape'][0], c:c + d['tile_shape'][1]].copy()
                         for r, c in d['origins']]
        n = len(d['panels'])
        order = list(range(n - 1, -1, -1))
        h, w = d['tile_shape']
        for nr in range(1, n + 1):
            if n % nr or nr * (h + 1) - 1 > 30 or (n // nr) * (w + 1) - 1 > 30:
                continue
            x = _layout(d['panels'], order, (nr, n // nr), d['divider'])
            y = _layout(output_panels, order, (nr, n // nr), d['divider'])
            _require(np.array_equal(solve(x), y), "tile permutation/layout covariance")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(4000 + a * 10 + b, a, b)
            _require(len(puzzle['train']) == a and len(puzzle['test']) == b, "legal counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    wrong = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    pair = wrong['test'][0]
    r, c = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][r, c] = pair['input'][r, c]
    try:
        validate_generated_puzzle(wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError('partial cable fill accepted')
    return {'seeds': 42, 'grid_shapes': len(shapes), 'tile_shapes': len(tile_shapes),
            'layouts': len(layouts), 'path_masks': len(paths), 'total_lengths': len(totals),
            'palette_roles': len(colors), 'family_counts': sorted(family_counts),
            'fragment_counts': sorted(parts), 'ranks': sorted(ranks)}
