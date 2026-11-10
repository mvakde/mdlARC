"""Cadenced dye channels: periodic growth, branching, and component dyeing.

Only input geometry supplies roles and cadence. No source data, seed, or
generation metadata is consulted by solve(). All imports are standard/NumPy.
"""

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _recognize(grid):
    """Try all boundary directions; require one complete visible explanation.

    k records the rotation from the downward canonical scene to the input.
    A transverse adjacent pair distinguishes walls from isolated dye beads.
    """
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) not in (3, 4) or counts.max() * 2 <= grid.size:
        raise ValueError("need three/four colors and a majority background")
    bg = int(colors[np.argmax(counts)])
    choices = []
    for k in range(4):
        a = np.rot90(grid, -k)
        h, w = a.shape
        for guide in map(int, colors):
            if guide == bg:
                continue
            rows, cols = np.nonzero(a == guide)
            rr = np.unique(rows)
            if len(rr) != 2 or rr[0] != 0:
                continue
            p = int(rr[1])
            if 3 * p >= h:
                continue
            entries = np.flatnonzero(a[0] == guide)
            if (not len(entries) or np.any(np.diff(entries) < 2)
                    or not np.array_equal(entries, np.flatnonzero(a[p] == guide))
                    or len(rows) != 2 * len(entries)):
                continue
            if np.any((a[:p+1] != bg) & (a[:p+1] != guide)):
                continue
            other = [int(c) for c in colors if c not in (bg, guide)]
            wall_options = [c for c in other if np.any((a[:, :-1] == c) & (a[:, 1:] == c))]
            if len(wall_options) != 1:
                continue
            wall = wall_options[0]
            dye = next((c for c in other if c != wall), None)
            choices.append((k, bg, guide, wall, dye, p))
    if len(choices) != 1:
        raise ValueError("launch direction, wall role, or cadence is ambiguous")
    return choices[0]


def _run(row, c, wall):
    lo = hi = c
    while lo and row[lo-1] == wall:
        lo -= 1
    while hi + 1 < len(row) and row[hi+1] == wall:
        hi += 1
    return lo, hi


def _trace(a, bg, guide, wall, dye, period, mutation=None):
    """Construct the logical channel graph, then paint its components.

    Vertices are visible channel sites. Forward edges span one beat; wall
    events join incident tips, the upstream crosspiece, and available flanks.
    The input is immutable during growth. Union-find makes mergers independent
    of event order and lets a dye discovered later recolor shared ancestry.
    Mutations are deliberately wrong rules used only to audit demonstrations.
    """
    h, w = a.shape
    hop = 1 if mutation == 'dense' else period
    nodes, parent, sizes = {}, [], []

    def node(r, c):
        point = (r, c)
        if point not in nodes:
            nodes[point] = len(parent)
            parent.append(len(parent))
            sizes.append(1)
        return nodes[point]

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(x, y):
        x, y = root(x), root(y)
        if x == y:
            return
        if sizes[x] < sizes[y]:
            x, y = y, x
        parent[y] = x
        sizes[x] += sizes[y]

    entries = list(map(int, np.flatnonzero(a[0] == guide)))
    active = {c: node(0, c) for c in entries}
    events, forward_beads, incoming = [], set(), {}
    for r in range(hop, h, hop):
        incoming[r] = set(active)
        hits, following = {}, {}
        for c, previous in sorted(active.items()):
            value = int(a[r, c])
            if value == wall and mutation != 'ignore_walls':
                hits.setdefault(_run(a[r], c, wall), []).append(previous)
                continue
            current = node(r, c)
            join(previous, current)
            if dye is not None and value == dye:
                forward_beads.add((r, c))
                if mutation != 'no_stop':
                    continue
            following[c] = current
        for (lo, hi), incident in sorted(hits.items()):
            pivot = incident[0]
            for previous in incident[1:]:
                join(pivot, previous)
            for c in range(max(0, lo-1), min(w, hi+2)):
                if a[r-hop, c] != wall:
                    join(pivot, node(r-hop, c))
            flanks = [c for c in (lo-1, hi+1) if 0 <= c < w]
            for c in flanks:
                current = node(r, c)
                join(pivot, current)
                if dye is not None and a[r, c] == dye:
                    forward_beads.add((r, c))
                    if mutation != 'no_stop':
                        continue
                following[c] = current
            events.append((r, lo, hi, len(incident), len(flanks)))
        active = following

    dye_depth = {}
    for (r, c), n in nodes.items():
        if dye is not None and a[r, c] == dye:
            component = root(n)
            dye_depth[component] = max(r, dye_depth.get(component, -1))
    dyed = set(dye_depth)
    source_components = {}
    for c in entries:
        component = root(nodes[(0, c)])
        source_components[component] = source_components.get(component, 0) + 1
    out = a.copy()
    for (r, c), n in nodes.items():
        if a[r, c] == wall or (dye is not None and a[r, c] == dye):
            continue
        paint_dye = root(n) in dyed
        if mutation == 'global_dye':
            paint_dye = dye is not None and bool(np.any(a == dye))
        elif mutation == 'no_recolor':
            paint_dye = False
        elif mutation == 'bounded_dye':
            paint_dye = paint_dye and r <= dye_depth[root(n)]
        out[r, c] = dye if paint_dye else guide
    reached = {(r, lo, hi) for r, lo, hi, _, _ in events}
    walls = set()
    for r in range(h):
        c = 0
        while c < w:
            if a[r, c] == wall:
                lo, hi = _run(a[r], c, wall)
                walls.add((r, lo, hi))
                c = hi + 1
            else:
                c += 1
    dye_points = set() if dye is None else set(zip(*np.nonzero(a == dye)))
    touched = dye_points & set(nodes)
    stats = {
        'period': period, 'entries': len(entries), 'shape': (h, w),
        'events': events, 'hits': len(events),
        'splits': sum(e[4] == 2 for e in events),
        'clipped': sum(e[4] < 2 for e in events),
        'multi_hit': sum(e[3] > 1 for e in events),
        'simultaneous': len(events) > len({e[0] for e in events}),
        'interrupted': sum(np.any(a[r-period, max(0, lo-1):min(w, hi+2)] == wall)
                           for r, lo, hi, _, _ in events),
        'inert_walls': len(walls - reached),
        'offbeat_walls': sum(r % period != 0 for r, _, _ in walls),
        'forward_beads': len(forward_beads), 'touched_beads': len(touched),
        'inert_beads': len(dye_points - touched),
        'offbeat_beads': sum(r % period != 0 for r, _ in dye_points),
        'dyed_sources': sum(n for component, n in source_components.items() if component in dyed),
        'undyed_sources': sum(n for component, n in source_components.items() if component not in dyed),
        'merged_dyed_sources': sum(n for component, n in source_components.items() if component in dyed and n > 1),
        'components': len(source_components), 'sites': len(nodes),
        'incoming': incoming,
    }
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    k, bg, guide, wall, dye, p = _recognize(grid)
    a = np.rot90(grid, -k)
    out, _ = _trace(a, bg, guide, wall, dye, p)
    return np.rot90(out, k).copy()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _witness(rng, kind, p, variant=0):
    """Readable causal witnesses, with free widths, hit offsets, spans and tails."""
    levels = int(rng.integers(6, min(10, 29 // p + 1) + 1))
    h = (levels-1)*p + 1 + int(rng.integers(min(p, 30-(levels-1)*p)))
    w = int(rng.integers(15, 31))
    a = np.zeros((h, w), dtype=np.int8)
    if kind == 'separate':
        lo = int(rng.integers(2, max(3, w//2-1)))
        hi = int(rng.integers(lo+1, min(w-6, lo+6)+1))
        entry = int(rng.integers(lo, hi+1))
        remote = int(rng.integers(hi+4, w))
        a[[0, p], entry] = a[[0, p], remote] = 1
        a[2*p, lo:hi+1] = 2
        a[3*p, lo:hi+1] = 2  # same-length run, not reached
        a[4*p, lo-1] = 3
        a[5*p, entry] = 3  # off-channel, despite lying on a beat
        if p > 1:
            a[2*p-1, max(0, entry-1):entry+2] = 2
            a[3*p-1, hi+1] = 3
    else:
        middle = int(rng.integers(3, w-5))
        right = int(rng.integers(middle+2, w-2))
        e1 = int(rng.integers(0, middle))
        e2 = int(rng.integers(middle+1, right+1))
        a[[0, p], e1] = a[[0, p], e2] = 1
        a[2*p, :middle] = a[2*p, middle+1:right+1] = 2
        # Both merged descendants hit this later run. Its upstream crosspiece
        # is visibly interrupted by the preceding wall, which stays intact.
        a[3*p, middle:right+2] = 2
        a[4*p, middle-1] = 3
        a[5*p, middle] = 3
        # An inert short run; leaves both active columns untouched.
        a[3*p, 0:2] = 2
        if p > 1:
            a[2*p-1, middle:middle+2] = 2
            a[3*p-1, right+2] = 3
    # Reflection preserves the causal witness while varying the clipped side.
    if variant % 2:
        a = np.fliplr(a).copy()
    return a


def _active_after(row, active):
    following = set()
    for c in active:
        if row[c] == 2:
            lo, hi = _run(row, c, 2)
            following.update(x for x in (lo-1, hi+1) if 0 <= x < len(row))
        else:
            following.add(c)
    return following


def _random_scene(rng, p, require_dye=False):
    levels = int(rng.integers(5, min(13, 29//p+1)+1))
    h = (levels-1)*p+1+int(rng.integers(min(p, 30-(levels-1)*p)))
    w = int(rng.integers(9, 31))
    a = np.zeros((h, w), dtype=np.int8)
    entries = []
    quota = int(rng.integers(1, min(5, (w+1)//3)+1))
    for c in map(int, rng.permutation(w)):
        if all(abs(c-x) >= 2 for x in entries):
            entries.append(c)
        if len(entries) == quota:
            break
    a[0, entries] = a[p, entries] = 1
    active, incoming = set(entries), {}
    for level in range(2, levels):
        r = level*p
        incoming[r] = active.copy()
        # Independent event budget; multiple disjoint runs may be struck.
        pivots = list(sorted(active))
        rng.shuffle(pivots)
        if level == 2 or rng.random() < .66:
            for c in pivots[:int(rng.integers(1, min(3, len(pivots))+1))]:
                if a[r, c] != 0:
                    continue
                if rng.random() < .18:
                    length = int(rng.integers(max(2, w//2), w))
                else:
                    length = int(rng.integers(1, min(7, w)))
                left = int(rng.integers(max(0, c-length+1), min(c, w-length)+1))
                # Keep maximal runs separated; do not silently fuse proposals.
                if np.any(a[r, max(0, left-1):min(w, left+length+1)] == 2):
                    continue
                a[r, left:left+length] = 2
        active = _active_after(a[r], active)
        # Optional inert geometry uses the same span range, limited by free space.
        if rng.random() < .5:
            length = int(rng.integers(1, min(8, w)))
            left = int(rng.integers(w-length+1))
            if (not any(left <= c < left+length for c in incoming[r])
                    and not np.any(a[r, max(0, left-1):min(w, left+length+1)] == 2)):
                a[r, left:left+length] = 2

    # Dye markers are inserted after wall scheduling, then reachability is
    # recomputed. The earliest chosen forward contact is guaranteed reachable.
    options = [(r, c) for r, cols in incoming.items() if r >= 3*p and r+p < h
               for c in sorted(cols) if a[r, c] == 0]
    if options and (require_dye or rng.random() < .8):
        rng.shuffle(options)
        chosen = options[:int(rng.integers(1, min(4, len(options))+1))]
        for r, c in chosen:
            if not np.any(a[r, max(0, c-1):min(w, c+2)] == 3):
                a[r, c] = 3
    for _ in range(int(rng.integers(0, 5))):
        r = int(rng.integers(p+1, h))
        c = int(rng.integers(w))
        if a[r, c] == 0 and not np.any(a[r, max(0, c-1):min(w, c+2)] == 3):
            if rng.random() < .55:
                a[r, c] = 3
            elif r % p:
                n = int(rng.integers(2, min(w-c, 7)+1)) if w-c >= 2 else 1
                a[r, c:c+n] = 2
    if p > 1 and rng.random() < .65:
        r = int(_pick(rng, [r for r in range(p+1, h) if r % p]))
        c = int(rng.integers(w-1))
        if np.all(a[r, c:c+2] == 0):
            a[r, c:c+2] = 2
    return a


def _key(a):
    return a.shape, a.tobytes()


def _make_pair(rng, kind, p, palette, turns, seen):
    for attempt in range(100):
        a = (_witness(rng, kind, p, attempt) if kind in ('separate', 'merge')
             else _random_scene(rng, p, require_dye=kind == 'query'))
        if rng.random() < .5:
            a = np.fliplr(a).copy()
        out, stats = _trace(a, 0, 1, 2, 3 if np.any(a == 3) else None, p)
        if stats['hits'] < 1 or np.array_equal(a, out):
            continue
        if kind == 'separate' and not (stats['dyed_sources'] and stats['undyed_sources'] and stats['inert_beads']):
            continue
        if kind == 'merge' and not (stats['merged_dyed_sources'] >= 2 and stats['clipped']):
            continue
        if kind == 'query' and not (stats['forward_beads'] and stats['hits'] >= 2):
            continue
        inp = np.rot90(np.asarray(palette, dtype=np.int8)[a], turns).copy()
        expected = np.rot90(np.asarray(palette, dtype=np.int8)[out], turns).copy()
        if _key(inp) in seen:
            continue
        try:
            roles = _recognize(inp)
        except ValueError:
            continue
        dye = palette[3] if np.any(a == 3) else None
        if roles != (turns, *palette[:3], dye, p):
            continue
        return {'input': inp, 'output': expected}
    raise RuntimeError('bounded scene construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    mixed_palette = bool(rng.random() < .35)
    axis = int(rng.integers(2))
    first_period = int(rng.integers(2, 5))
    second_period = int(_pick(rng, [p for p in range(1, 5) if p != first_period]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            local_palette = palette
            if mixed_palette and i + (split == 'test') > 0:
                local_palette = tuple(map(int, rng.choice(10, 4, replace=False)))
            if split == 'train':
                kind = ('separate', 'merge')[i] if i < 2 else 'general'
                p = (first_period, second_period)[i] if i < 2 else int(rng.integers(1, 5))
                turns = (axis + 2*(i % 2)) % 4
            else:
                kind, p = 'query', int(rng.integers(2, 6))
                turns = (axis + 1 + 2*(i % 2)) % 4
            pair = _make_pair(rng, kind, p, local_palette, turns, seen)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def _analysis(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    k, bg, guide, wall, dye, p = _recognize(inp)
    a, o = np.rot90(inp, -k), np.rot90(out, -k)
    expected, stats = _trace(a, bg, guide, wall, dye, p)
    _require(np.array_equal(o, expected), 'incorrect channel graph or component color')
    _require(a.shape == o.shape and not np.array_equal(a, o), 'unchanged or resized scene')
    _require(np.all(o[a == wall] == wall), 'wall overwritten')
    if dye is not None:
        _require(np.all(o[a == dye] == dye), 'bead overwritten')
    _require(np.all(o[a == guide] != bg), 'launch dot erased')
    changed = a != o
    _require(np.all(np.nonzero(changed)[0] % p == 0), 'ink appeared between beats')
    _require(np.all(np.isin(o[changed], [guide] if dye is None else [guide, dye])), 'invalid ink')
    stats.update(turns=k, palette=(bg, guide, wall, dye))
    return a, o, stats


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train, test = set(), [], []
    mutants = {name: False for name in ('dense', 'ignore_walls', 'global_dye',
                                      'no_recolor', 'no_stop', 'bounded_dye')}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, o, stats = _analysis(pair)
            key = _key(_grid(pair['input']))
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            (train if split == 'train' else test).append(stats)
            if split == 'train':
                for name in mutants:
                    wrong, _ = _trace(a, *stats['palette'], stats['period'], mutation=name)
                    mutants[name] |= not np.array_equal(wrong, o)
    _require(all(mutants.values()), 'training admits a tested parent-omitting shortcut')
    _require(any(s['splits'] and s['forward_beads'] and s['undyed_sources'] for s in train),
             'need dyed branching and separate undyed channels')
    _require(any(s['merged_dyed_sources'] >= 2 for s in train), 'need dye through a merger')
    _require(any(s['inert_beads'] for s in train), 'need unreachable bead evidence')
    _require(any(s['offbeat_walls'] and s['offbeat_beads'] for s in train), 'need missed-beat evidence')
    _require(any(s['clipped'] for s in train), 'need boundary clipping evidence')
    _require(any(s['multi_hit'] and s['interrupted'] for s in train),
             'need shared-run collision and wall precedence evidence')
    _require(len({s['period'] for s in train}) >= 2, 'training cadence variation')
    _require({s['turns'] % 2 for s in train}.isdisjoint({s['turns'] % 2 for s in test}),
             'test must transfer the propagation axis')


def _manual_regressions():
    # Hand-painted fixture: the left descendant stops, its right sibling dyes,
    # an offbeat wall is skipped, and an independent inlet stays undyed.
    a = np.zeros((13, 13), dtype=np.int8)
    a[[0, 2], 4] = a[[0, 2], 11] = 1
    a[3, 3:6] = a[4, 3:6] = 2
    a[8, 2] = a[6, 4] = 3
    expected = a.copy()
    expected[[0, 2], 4] = 3
    expected[2, 2:7] = 3
    expected[[4, 6, 8], 2] = 3
    expected[[4, 6, 8, 10, 12], 6] = 3
    expected[::2, 11] = 1
    _require(np.array_equal(solve(a), expected), 'manual branch/phase regression')
    # Two simultaneously struck runs share a flank and connect two launch pairs.
    b = np.zeros((8, 13), dtype=np.int8)
    b[:2, [1, 7]] = 1
    b[2, :4] = b[2, 5:9] = 2
    b[4, 4] = 3
    expected_b = b.copy()
    expected_b[0, [1, 7]] = 3
    expected_b[1, :10] = 3
    expected_b[2:5, 4] = 3
    expected_b[2:, 9] = 3
    _require(np.array_equal(solve(b), expected_b), 'manual merger/clipping regression')
    for x, y in ((a, expected), (b, expected_b)):
        for k in range(4):
            for mapping in (np.arange(10), np.array([8, 0, 5, 2, 1, 3, 4, 6, 7, 9])):
                xx, yy = np.rot90(mapping[x], k), np.rot90(mapping[y], k)
                original = xx.copy()
                _require(np.array_equal(solve(xx), yy), 'role/rotation equivariance')
                _require(np.array_equal(original, xx), 'solve mutated the input')


def self_test() -> dict:
    """Bounded 42-seed audit, meaningful witnesses, counts and manual semantics."""
    _manual_regressions()
    shapes, layouts, palettes, periods, entries, lengths, turns = (set() for _ in range(7))
    totals = {s: {n: 0 for n in ('pairs', 'splits', 'clipped', 'multi_hit', 'simultaneous',
                                'interrupted', 'forward_beads', 'inert_beads', 'offbeat_walls',
                                'offbeat_beads', 'merged_dyed_sources', 'undyed_sources')}
              for s in ('train', 'test')}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                a, o, stats = _analysis(pair)
                shapes.add(pair['input'].shape)
                bg, guide, wall, dye = stats['palette']
                norm = np.zeros(a.shape, dtype=np.int8)
                norm[a == guide] = 1
                norm[a == wall] = 2
                if dye is not None:
                    norm[a == dye] = 3
                layouts.add(_key(norm))
                palettes.add(stats['palette'])
                periods.add(stats['period'])
                entries.add(stats['entries'])
                turns.add(stats['turns'])
                lengths.update(hi-lo+1 for _, lo, hi, _, _ in stats['events'])
                totals[split]['pairs'] += 1
                for name in totals[split]:
                    if name != 'pairs':
                        totals[split][name] += int(bool(stats[name]))
    _require(len(signatures) == 42 and len(layouts) >= 150, 'structural diversity collapsed')
    _require(len(shapes) >= 60 and len(palettes) >= 40, 'shape/palette diversity collapsed')
    _require(periods == {1, 2, 3, 4, 5} and turns == {0, 1, 2, 3}, 'cadence/direction coverage')
    _require({1, 2, 3, 4} <= entries and len(lengths) >= 10, 'source/span diversity')
    for name in ('splits', 'clipped', 'forward_beads', 'inert_beads', 'offbeat_walls',
                 'offbeat_beads', 'merged_dyed_sources', 'simultaneous', 'interrupted'):
        _require(totals['train'][name] > 0 and totals['test'][name] > 0, f'coverage missing {name}')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(7109, nt, nq)
            _require((len(p['train']), len(p['test'])) == (nt, nq), 'requested counts lost')
    a, b = generate_puzzle(2027, 10, 4), generate_puzzle(2027, 10, 4)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ('input', 'output')), 'nondeterminism')
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4, False), (2.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(1, *counts)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'shapes': len(shapes),
            'canonical_structures': len(layouts), 'palettes': len(palettes),
            'periods': sorted(periods), 'entry_counts': sorted(entries),
            'reached_lengths': sorted(lengths), 'features_by_split': totals,
            'semantic_scope': 'author-owned invariants and hand-painted regressions'}
