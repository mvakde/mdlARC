"""Matched seams activate switches.

All inference is from the query pixels. No parent code, files, examples, random
state, or generation metadata are used by solve(). See task_spec.md for the rule.
"""
from itertools import combinations, permutations
import hashlib

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _groups(points):
    remaining = set(points)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _normalize(part):
    r0 = min(r for r, c in part)
    c0 = min(c for r, c in part)
    return {(r-r0, c-c0): int(v) for (r, c), v in part.items()}


def _shape(part):
    return max(r for r, c in part) + 1, max(c for r, c in part) + 1


def _render(part, bg):
    part = _normalize(part)
    out = np.full(_shape(part), bg, dtype=np.int8)
    for p, v in part.items():
        out[p] = v
    return out


def _read_fragments(inp):
    """Recover roles, pieces, and complete same-orientation registration bars."""
    colors, counts = np.unique(inp, return_counts=True)
    assert counts.max() * 2 > inp.size, 'background must be a strict majority'
    bg = int(colors[counts.argmax()])
    foreground = [(int(n), int(c)) for c, n in zip(colors, counts) if c != bg]
    foreground.sort(reverse=True)
    assert len(foreground) >= 5 and foreground[0][0] > foreground[1][0]
    track = foreground[0][1]
    pieces = _groups(map(tuple, np.argwhere(inp != bg)))
    assert 2 <= len(pieces) <= 6, 'need two to six separate fragments'
    owner = {p: i for i, part in enumerate(pieces) for p in part}
    origins = [(min(r for r, c in part), min(c for r, c in part)) for part in pieces]
    parts = [_normalize({p: inp[p] for p in part}) for part in pieces]
    small = _groups(map(tuple, np.argwhere((inp != bg) & (inp != track))))
    beads, keys, occurrences = set(), [], {}
    for group in small:
        if len(group) == 1:
            beads.add(int(inp[next(iter(group))]))
            continue
        cells = sorted(group)
        assert len(cells) == 3
        r, c = cells[0]
        assert cells == [(r, c), (r, c+1), (r, c+2)], 'keys are horizontal triples'
        pattern = tuple(int(inp[p]) for p in cells)
        assert len(set(pattern)) == 3, 'a key has three different colors'
        i = owner[cells[0]]
        assert all(owner[p] == i for p in cells)
        r0, c0 = origins[i]
        anchor = r-r0, c-c0
        occurrences.setdefault(pattern, []).append((i, anchor))
        keys.append((i, anchor, pattern))
    assert len(beads) == 1, 'one geometrically identifiable bead color is required'
    bead = next(iter(beads))
    assert all(bead not in key for key in occurrences)
    for pattern, copies in occurrences.items():
        assert len({i for i, p in copies}) == len(copies), 'one key identity per fragment'
    assert all(any(i == j for j, p, key in keys) for i in range(len(parts)))
    return bg, track, bead, parts, occurrences


def _register(parts, occurrences, bg):
    """Exact key equality gives equations for translations, not a packing score."""
    edges = [[] for _ in parts]
    for copies in occurrences.values():
        i, (r, c) = copies[0]
        for j, (s, t) in copies[1:]:
            delta = r-s, c-t
            edges[i].append((j, delta))
            edges[j].append((i, (-delta[0], -delta[1])))
    offsets = {0: (0, 0)}
    stack = [0]
    while stack:
        i = stack.pop()
        for j, (dr, dc) in edges[i]:
            candidate = offsets[i][0]+dr, offsets[i][1]+dc
            if j in offsets:
                assert offsets[j] == candidate, 'inconsistent registration keys'
            else:
                offsets[j] = candidate
                stack.append(j)
    assert len(offsets) == len(parts), 'registration graph must connect all fragments'
    union = {}
    coverage = {}
    for i, part in enumerate(parts):
        dr, dc = offsets[i]
        for (r, c), v in part.items():
            p = r+dr, c+dc
            assert p not in union or union[p] == v, 'conflicting overlap'
            union[p] = v
            coverage[p] = coverage.get(p, 0) + 1
    # Fragments are complete rectangular views: observed empty space agrees too.
    for i, part in enumerate(parts):
        dr, dc = offsets[i]
        h, w = _shape(part)
        assert all((r-dr, c-dc) in part for r, c in union
                   if 0 <= r-dr < h and 0 <= c-dc < w), 'empty-space conflict'
    r0 = min(r for r, c in union)
    c0 = min(c for r, c in union)
    gate_data = []
    for pattern, copies in occurrences.items():
        i, (r, c) = copies[0]
        dr, dc = offsets[i]
        gate_data.append((r+dr-r0, c+dc-c0+1, pattern, len(copies)))
    gate_data.sort()
    assert len({(r, c) for r, c, key, n in gate_data}) == len(gate_data)
    raw = _render(union, bg)
    coverage = {(r-r0, c-c0): n for (r, c), n in coverage.items()}
    return raw, gate_data, coverage, offsets


def _paths(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    adjacency = {p: [q for dr, dc in _DIRS if (q := (p[0]+dr, p[1]+dc)) in cells]
                 for p in cells}
    paths = []
    for component in _groups(cells):
        assert all(len(adjacency[p]) in (1, 2) for p in component), 'branch or isolated rail'
        ends = sorted(p for p in component if len(adjacency[p]) == 1)
        assert len(ends) == 2, 'closed loop'
        ordered, previous = [ends[0]], None
        while ordered[-1] != ends[1]:
            choices = [p for p in adjacency[ordered[-1]] if p != previous]
            assert len(choices) == 1
            previous = ordered[-1]
            ordered.append(choices[0])
        assert len(ordered) == len(component)
        assert len({b[0]-a[0] for a, b in zip(ordered, ordered[1:]) if a[0] != b[0]}) <= 1
        paths.append(ordered)
    return sorted(paths, key=lambda p: (p[0], p[-1]))


def _transform(raw, gates, bg, track, bead):
    """Simultaneous local rewiring and per-path conservation of DISTINCT beads."""
    h, w = raw.shape
    assert h >= 3 and h % 2 == 1
    assert np.all(np.isin(raw[::2], [bg, track, bead]))
    assert not np.any(np.isin(raw[1::2], [track, bead]))
    assert all(np.any(np.isin(row, [track, bead])) for row in raw[::2])
    before = set(map(tuple, np.argwhere(raw == bead)))
    structural = raw.copy()
    structural[structural == bead] = track
    writes = {}
    shared_writes = 0
    key_cells = set()
    for r, c, pattern, copies in gates:
        assert r % 2 == 1 and 1 <= r < h-1 and 1 <= c < w-1
        assert tuple(map(int, raw[r, c-1:c+2])) == pattern
        key_cells.update((r, x) for x in range(c-1, c+2))
        assert np.all(raw[r-1, c-1:c+2] == track)
        assert np.all(raw[r+1, c-1:c+2] == track)
        if copies < 2:
            continue
        for rr in range(r-1, r+2):
            for cc in range(c-1, c+2):
                v = pattern[rr-r+1] if cc == c else track
                if (rr, cc) in writes:
                    assert writes[rr, cc] == v, 'incompatible switch neighborhoods'
                    shared_writes += 1
                writes[rr, cc] = v
    assert key_cells == set(map(tuple, np.argwhere((raw != bg) & (raw != track) & (raw != bead))))
    for p, v in writes.items():
        structural[p] = v
    paths = _paths(structural == track)
    ends = {p for path in paths for p in (path[0], path[-1])}
    assert before <= ends, 'beads must begin at final-network terminals'
    out, after = structural.copy(), set()
    single = collision = level = empty = settled = 0
    for path in paths:
        initial = before & {path[0], path[-1]}
        count = len(initial)
        if not count:
            empty += 1
            continue
        if path[0][0] == path[-1][0]:
            target = initial
            level += 1
        else:
            inward = path if path[0][0] > path[-1][0] else path[::-1]
            target = set(inward[:count])
            single += count == 1 and target != initial
            collision += count == 2
            settled += target == initial
        after.update(target)
        for p in target:
            out[p] = bead
    assert len(before) == len(after) == int(np.count_nonzero(out == bead))
    unchanged = set(np.ndindex(raw.shape)) - set(writes) - before - after
    assert all(out[p] == raw[p] for p in unchanged)
    return out, {'raw': raw, 'structural': structural, 'gates': gates, 'paths': paths,
                 'before': before, 'after': after, 'single': single, 'collision': collision,
                 'level': level, 'empty': empty, 'settled': settled,
                 'compatible_overlap_cells': shared_writes}


def _analyze(input_grid):
    inp = _grid(input_grid)
    bg, track, bead, parts, occurrences = _read_fragments(inp)
    raw, gates, coverage, offsets = _register(parts, occurrences, bg)
    out, info = _transform(raw, gates, bg, track, bead)
    assert out.shape == _render({p: int(out[p]) for p in map(tuple, np.argwhere(out != bg))}, bg).shape
    info.update(bg=bg, track=track, bead=bead, parts=parts, offsets=offsets,
                coverage=coverage, input_bead_copies=int(np.count_nonzero(inp == bead)),
                active=sum(n >= 2 for r, c, key, n in gates),
                inactive=sum(n == 1 for r, c, key, n in gates))
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only this input; roles and registrations are visible in pixels."""
    return _analyze(input_grid)[0]


def _slabs(rng, lanes, count):
    options = []
    for cuts in combinations(range(1, lanes-2), count-1):
        starts = (0,) + cuts
        ends = tuple(c+1 for c in cuts) + (lanes-1,)
        intervals = list(zip(starts, ends))
        if all(3 <= b-a+1 <= 5 for a, b in intervals):
            options.append(intervals)
    assert options
    return options[int(rng.integers(len(options)))]


def _pack(parts, bg, rng):
    """Random shelves of arbitrary rectangles; part order never encodes adjacency.

    At most five rectangles, each <=9 by 13, admit a two-column fallback.
    The finite search below also samples compact one-column presentations.
    """
    parts = [_normalize(p) for p in parts]
    shapes = [_shape(p) for p in parts]
    n, material = len(parts), sum(map(len, parts))
    options = []
    for mask in range(1, 1 << n):
        groups = [[i for i in range(n) if bool(mask & (1 << i)) == side] for side in (True, False)]
        groups = [g for g in groups if g]
        widths = [max(shapes[i][1] for i in g) for g in groups]
        heights = [sum(shapes[i][0] for i in g)+len(g)-1 for g in groups]
        minw, minh = sum(widths)+len(groups)-1, max(heights)
        if minw <= 30 and minh <= 30:
            options.append((groups, widths, minw, minh))
    assert options, 'fragments cannot fit'
    groups, widths, minw, minh = options[int(rng.integers(len(options)))]
    canvases = [(h, w) for h in range(minh, 31) for w in range(minw, 31) if h*w > 2*material]
    assert canvases
    h, w = canvases[int(rng.integers(len(canvases)))]
    # Distribute unused horizontal space among margins and between columns.
    xspace = rng.multinomial(w-minw, np.full(len(groups)+1, 1/(len(groups)+1)))
    grid = np.full((h, w), bg, dtype=np.int8)
    left = int(xspace[0])
    for col, group in enumerate(groups):
        group = [int(i) for i in rng.permutation(group)]
        needed = sum(shapes[i][0] for i in group)+len(group)-1
        yspace = rng.multinomial(h-needed, np.full(len(group)+1, 1/(len(group)+1)))
        top = int(yspace[0])
        for k, i in enumerate(group):
            ph, pw = shapes[i]
            x = left + int(rng.integers(widths[col]-pw+1))
            for (r, c), v in parts[i].items():
                grid[top+r, x+c] = v
            top += ph+1+int(yspace[k+1])
        left += widths[col]+1+int(xspace[col+1])
    assert np.count_nonzero(grid == bg)*2 > grid.size
    assert len(_groups(map(tuple, np.argwhere(grid != bg)))) == n
    return grid


def _make_pair(rng, palette, lanes, clone, emphasis):
    bg, track, bead = palette[:3]
    key_colors = palette[3:]
    possible_counts = [k for k in range(2, min(4, lanes-2)+1)
                       if any(all(3 <= b-a+1 <= 5 for a, b in zip((0,)+cuts, tuple(c+1 for c in cuts)+(lanes-1,)))
                              for cuts in combinations(range(1, lanes-2), k-1))]
    count = int(rng.choice(possible_counts))
    spans = _slabs(rng, lanes, count)
    # All windows share a seven-column core but extend independently on either
    # side. Their assembled rail endpoints therefore form an irregular outline.
    bounds = [(3-int(rng.integers(4)), 9+int(rng.integers(4))) for _ in spans]
    if rng.random() < .3:
        bounds = [bounds[0]]*count
    windows = [(a, b, l, u) for (a, b), (l, u) in zip(spans, bounds)]
    if clone:
        windows.append(windows[0])  # A contained copy changes seam multiplicity.
    carriers, domains = [], []
    for i in range(lanes-1):
        who = [j for j, (a, b, l, u) in enumerate(windows) if a <= i and i+1 <= b]
        lo = max(windows[j][2] for j in who)+2
        hi = min(windows[j][3] for j in who)-2
        carriers.append(who)
        domains.append([int(c) for c in rng.permutation(np.arange(lo, hi+1))])
    centers = []

    def choose_centers(i):
        if i == len(domains):
            return True
        for c in domains[i]:
            if not centers or abs(c-centers[-1]) >= 2:
                centers.append(c)
                if choose_centers(i+1):
                    return True
                centers.pop()
        return False

    assert choose_centers(0)
    patterns = list(permutations(key_colors, 3))
    chosen = rng.choice(len(patterns), size=2*(lanes-1), replace=False)
    patterns = [patterns[int(i)] for i in chosen]
    raw = np.full((2*lanes-1, 13), bg, dtype=np.int8)
    for a, b, l, u in windows:
        raw[2*a:2*b+1:2, l:u+1] = track
    gates = []
    for i, (c, who) in enumerate(zip(centers, carriers)):
        key = patterns[i]
        raw[2*i+1, c-1:c+2] = key
        gates.append((2*i+1, c, key, len(who)))
        # Extra singleton keys may inhabit exclusive tails. They never silently
        # become active when the contained first window is duplicated.
        options = []
        for x in range(2, 11):
            if abs(x-c) < 4:
                continue
            inside = [j for j, (a, b, l, u) in enumerate(windows)
                      if a <= i and i+1 <= b and l+2 <= x <= u-2]
            if len(inside) != 1:
                continue
            if all(j == inside[0] or not (a <= i and i+1 <= b)
                   or x+1 < l or x-1 > u for j, (a, b, l, u) in enumerate(windows)):
                options.append(x)
        if options and rng.random() < .6:
            x = int(rng.choice(options))
            key = patterns[lanes-1+i]
            raw[2*i+1, x-1:x+2] = key
            gates.append((2*i+1, x, key, 1))
    # Build the graph before placing beads. Beads may only occupy true terminals.
    structural, bare = _transform(raw, gates, bg, track, bead)
    vertical = [p for p in bare['paths'] if p[0][0] != p[-1][0]]
    level = [p for p in bare['paths'] if p[0][0] == p[-1][0]]
    assert len(vertical) >= 2 and level
    # The uppermost path proves motion and, in the clone example, bead deduplication.
    vertical.sort(key=lambda p: (p[0][0], p[0][1]))
    single_path = vertical[0]
    collision_path = vertical[1]
    selected = {min((single_path[0], single_path[-1]), key=lambda p: p[0]),
                collision_path[0], collision_path[-1]}
    selected.add(level[int(rng.integers(len(level)))][int(rng.choice([0, -1]))])
    used = {tuple(single_path), tuple(collision_path)}
    probability = float(rng.uniform(.12, .85))
    for path in bare['paths']:
        if tuple(path) not in used:
            selected.update(p for p in (path[0], path[-1]) if rng.random() < probability)
    for p in selected:
        raw[p] = bead
    intended, _ = _transform(raw, gates, bg, track, bead)
    pieces = []
    for a, b, l, u in windows:
        part = {(r-2*a, c-l): int(raw[r, c]) for r in range(2*a, 2*b+1)
                for c in range(l, u+1) if raw[r, c] != bg}
        assert len(_groups(part)) == 1
        pieces.append(part)
    inp = _pack(pieces, bg, rng)
    out, info = _analyze(inp)
    # Remove the unobserved construction margin before comparing, never use it
    # to help registration. The pixel-derived solver must find the same union.
    raw_tight = _render({p: int(raw[p]) for p in map(tuple, np.argwhere(raw != bg))}, bg)
    intended = _render({p: int(intended[p]) for p in map(tuple, np.argwhere(intended != bg))}, bg)
    assert np.array_equal(info['raw'], raw_tight)
    assert np.array_equal(out, intended)
    assert info['active'] and info['inactive'] and info['single'] and info['collision'] and info['level']
    if emphasis == 'clone':
        assert max(n for r, c, key, n in info['gates']) >= 3
        assert info['input_bead_copies'] > len(info['before'])
    return {'input': inp, 'output': intended}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must lie in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count('num_train', num_train, 2, 10)
    nq = _count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)))
    key_count = int(rng.integers(4, 8))
    palette = palette[:3+key_count]
    change_palette = rng.random() < .5
    positions = list(map(int, rng.permutation(nt)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            # Train on 4/5 lanes, test on 6/7/8. More seams and longer routes are
            # the shift; exact key colors, spacing and absolute locations vary.
            lanes = int(rng.integers(4, 6) if split == 'train' else rng.integers(6, 9))
            emphasis = ('plain' if index == positions[0] else 'clone' if index == positions[1] else 'mixed') if split == 'train' else 'mixed'
            clone = emphasis == 'clone' or (emphasis == 'mixed' and rng.random() < .45)
            pair_palette = tuple(map(int, rng.permutation(10)))[:3+key_count] if change_palette else palette
            for attempt in range(12):
                pair = _make_pair(rng, pair_palette, lanes, clone, emphasis)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('failed to make distinct inputs')
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train_info, test_info = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            predicted, info = _analyze(inp)
            assert np.array_equal(predicted, out), 'incorrect routing, bead count, or crop'
            assert info['active'] and info['inactive'], 'show both repeated and singleton switches'
            assert info['single'] and info['collision'] and info['level']
            assert info['before'] != info['after'], 'A must affect bead positions'
            assert not np.array_equal(info['raw'], out), 'registration alone is insufficient'
            # Turning every bar differs even before bead motion: a singleton
            # remains horizontal in the correct result.
            for r, c, pattern, copies in info['gates']:
                if copies == 1:
                    assert tuple(map(int, out[r, c-1:c+2])) == pattern
            (train_info if split == 'train' else test_info).append(info)
    assert any(max(n for r, c, key, n in info['gates']) >= 3 for info in train_info)
    assert any(info['input_bead_copies'] > len(info['before']) for info in train_info)
    assert max(i['raw'].shape[0] for i in train_info) < min(i['raw'].shape[0] for i in test_info)


def self_test():
    """Bounded 42-seed measurements, semantic invariants and metamorphic checks.

    No sample payloads, execution traces, or logs are written. At most one
    puzzle and small aggregate hash sets are retained during the seed sweep.
    """
    # Hand-traced mechanics regression, not an output synthesized by _transform.
    # The only shared key is 145. Its left path moves one bead to (4,0);
    # its right path packs two at (4,8),(4,7). The outer lanes are level.
    raw_rows = ('988888888', '012300000', '988888889', '000145000',
                '888888889', '000001670', '988888888')
    expected_rows = ('988888888', '012300000', '888818888', '000848000',
                     '988858899', '000001670', '988888888')
    raw = np.array([[int(c) for c in row] for row in raw_rows], dtype=np.int8)
    expected = np.array([[int(c) for c in row] for row in expected_rows], dtype=np.int8)
    observed = np.zeros((14, 11), dtype=np.int8)
    observed[1:6, 1:10] = raw[:5]
    observed[8:13, 1:10] = raw[2:]
    predicted, golden = _analyze(observed)
    assert np.array_equal(predicted, expected)
    assert golden['input_bead_copies'] == 8 and len(golden['before']) == 5
    assert (golden['single'], golden['collision'], golden['level']) == (1, 1, 2)
    # Changing one copy alone destroys the sole registration edge.
    inconsistent = observed.copy()
    inconsistent[4, 4:7] = [2, 4, 5]
    try:
        solve(inconsistent)
    except AssertionError:
        pass
    else:
        raise AssertionError('disconnected keys accepted')
    hashes, in_shapes, out_shapes, layouts, roles = set(), set(), set(), set(), set()
    fragments, active, beads, gaps, multiplicities = set(), set(), set(), set(), set()
    overlap = duplicate_beads = extra_keys = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        hashes.add(hashlib.sha256(repr(_puzzle_key(puzzle)).encode()).digest())
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                out, info = _analyze(pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), out)
                in_shapes.add(pair['input'].shape)
                out_shapes.add(out.shape)
                roles.add((info['bg'], info['track'], info['bead']))
                fragments.add(len(info['parts']))
                active.add(info['active'])
                beads.add(len(info['before']))
                multiplicities.update(n for r, c, key, n in info['gates'])
                gaps.update(abs(a[1]-b[1]) for a, b in zip(info['gates'], info['gates'][1:]) if a[0] != b[0])
                labels = {info['bg']: 0, info['track']: 1, info['bead']: 2}
                normalized = bytes(labels.setdefault(int(c), len(labels)) for c in info['raw'].flat)
                layouts.add((info['raw'].shape, hashlib.sha256(normalized).digest()))
                overlap += info['compatible_overlap_cells'] > 0
                duplicate_beads += info['input_bead_copies'] > len(info['before'])
                extra_keys += len(info['gates']) > (info['raw'].shape[0]-1)//2
    assert len(hashes) == 42 and len(layouts) >= 160
    assert len(in_shapes) >= 60 and len(out_shapes) >= 25 and len(roles) >= 30
    assert {2, 3, 4, 5} <= fragments and {1, 2, 3} <= active
    assert {1, 2, 3} <= multiplicities and len(beads) >= 8 and len(gaps) >= 4
    assert overlap and duplicate_beads and extra_keys
    # Repacking changes all physical fragment positions, never the answer.
    p = generate_puzzle(0)
    for pair in p['train']:
        out, info = _analyze(pair['input'])
        repacked = _pack(info['parts'], info['bg'], np.random.default_rng(123))
        assert np.array_equal(solve(repacked), out)
    # Delete one contained duplicate. The registered picture is unchanged, but
    # its switch multiplicity, topology and answer must change.
    demonstrated = False
    for pair in p['train']:
        out, info = _analyze(pair['input'])
        parts = info['parts']
        for i in range(len(parts)):
            if any(i != j and parts[i] == parts[j] for j in range(len(parts))):
                reduced = _pack([part for j, part in enumerate(parts) if j != i], info['bg'], np.random.default_rng(45))
                changed, details = _analyze(reduced)
                assert np.array_equal(info['raw'], details['raw'])
                assert not np.array_equal(out, changed)
                assert info['before'] == details['before']
                assert info['after'] != details['after']
                demonstrated = True
                break
        if demonstrated:
            break
    assert demonstrated
    # Input-only solve is pure and respects a complete color permutation.
    query = p['test'][0]['input'].copy()
    untouched = query.copy()
    result = solve(query)
    assert np.array_equal(query, untouched)
    mapping = np.array([7, 4, 1, 9, 0, 8, 5, 2, 6, 3], dtype=np.int8)
    assert np.array_equal(solve(mapping[query]), mapping[result])
    assert _puzzle_key(generate_puzzle(77, 3, 2)) == _puzzle_key(generate_puzzle(77, 3, 2))
    for nt, nq in ((2, 1), (10, 4), (10, 1)):
        validate_generated_puzzle(generate_puzzle(87, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.2, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    return {'seeds': 42, 'pairs': 210, 'distinct_puzzles': len(hashes), 'input_shapes': len(in_shapes),
            'output_shapes': len(out_shapes), 'normalized_layouts': len(layouts),
            'role_palettes': len(roles), 'fragment_counts': sorted(fragments),
            'active_switch_counts': sorted(active), 'bead_counts': sorted(beads),
            'key_multiplicities': sorted(multiplicities), 'compatible_overlap_pairs': overlap,
            'duplicate_bead_pairs': duplicate_beads, 'extra_singleton_key_pairs': extra_keys}
