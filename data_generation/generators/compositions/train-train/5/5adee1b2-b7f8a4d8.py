"""Rectangle-fed chambers: keyed gutter inlets and selective chamber flooding.

Only standard library and NumPy are used. See task_spec.md for the input grammar.
"""
from itertools import combinations
import json
import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SIDES = ('N', 'S', 'W', 'E')


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid shape')
    _require(np.issubdtype(a.dtype, np.integer), 'noninteger grid')
    _require(np.all((0 <= a) & (a <= 9)), 'invalid grid colors')
    return a.astype(np.int8, copy=False)


def _parts(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        cells = [(r, c)]
        seen[r, c] = True
        for x, y in cells:
            for dr, dc in _DIRS:
                u, v = x + dr, y + dc
                if 0 <= u < mask.shape[0] and 0 <= v < mask.shape[1] and mask[u, v] and not seen[u, v]:
                    seen[u, v] = True
                    cells.append((u, v))
        result.append(cells)
    return result


def _runs(values):
    runs = []
    for i in np.flatnonzero(values):
        i = int(i)
        if not runs or i != runs[-1][-1] + 1:
            runs.append([i])
        else:
            runs[-1].append(i)
    return runs


def _edges(rect):
    """Meta-grid edges, with H joining columns and V joining rows."""
    (a, b), (c, d) = rect
    return ({('H', r, j) for r in (a, b) for j in range(c, d)} |
            {('V', i, s) for s in (c, d) for i in range(a, b)})


def _corners(rect):
    return {(r, c) for r in rect[0] for c in rect[1]}


def _tile_sides(rect):
    result = {}
    for axis, r, c in _edges(rect):
        first, second = ((r, c), (r, c + 1)) if axis == 'H' else ((r, c), (r + 1, c))
        sa, sb = ('E', 'W') if axis == 'H' else ('S', 'N')
        result.setdefault(first, set()).add(sa)
        result.setdefault(second, set()).add(sb)
    return result


def _decode(input_grid):
    """Read geometry, key, and corner groups using input cells only."""
    g = _grid(input_grid)
    h, w = g.shape
    _require(h >= 14 and w >= 13, 'grid too small for chamber grammar')
    bg = int(g[0, 0])
    _require(np.all(g[0] == bg) and np.all(g[-3] == bg), 'missing horizontal board margins')
    _require(np.array_equal(g[-2], g[-1]), 'key rows disagree')
    mappings = []
    consumed = np.zeros(w, bool)
    for c in (0, 3):
        if np.any(g[-2:, c:c + 2] != bg):
            a, b = int(g[-1, c]), int(g[-1, c + 1])
            _require(a != bg and b != bg and a != b, 'invalid ordered key')
            mappings.append((a, b))
            consumed[c:c + 2] = True
        elif c == 0:
            raise AssertionError('missing first key')
    _require(np.all(g[-2:, ~consumed] == bg), 'unexpected key cell')
    roles = [bg] + [x for pair in mappings for x in pair]
    _require(len(roles) == len(set(roles)), 'overlapping key roles')
    board = g[1:-3, 1:-1]
    _require(np.all(g[1:-3, 0] == bg) and np.all(g[1:-3, -1] == bg), 'missing side margins')
    rr = _runs(np.any(board != bg, axis=1))
    cc = _runs(np.any(board != bg, axis=0))
    _require(len(rr) >= 2 and len(cc) >= 2, 'not a tile lattice')
    th, tw = len(rr[0]), len(cc[0])
    _require(5 <= th <= 7 and 5 <= tw <= 7, 'invalid tile extent')
    _require(all(len(x) == th for x in rr) and all(len(x) == tw for x in cc), 'unequal tiles')
    _require(rr[0][0] == cc[0][0] == 0 and rr[-1][-1] == board.shape[0] - 1 and cc[-1][-1] == board.shape[1] - 1, 'extra board margin')
    gy = rr[1][0] - rr[0][-1] - 1
    gx = cc[1][0] - cc[0][-1] - 1
    _require(gy in (1, 2) and gx in (1, 2), 'invalid gutter width')
    _require(all(x[0] == i * (th + gy) for i, x in enumerate(rr)) and
             all(x[0] == i * (tw + gx) for i, x in enumerate(cc)), 'irregular lattice')
    starts_r = [1 + x[0] for x in rr]
    starts_c = [1 + x[0] for x in cc]
    wall = int(g[starts_r[0], starts_c[0] + tw - 1])
    _require(wall not in roles, 'wall overlaps key roles')
    studs = [a for a, _ in mappings]
    tiles, positions = {}, {a: set() for a in studs}
    occupied = np.zeros(g.shape, bool)
    for r, y in enumerate(starts_r):
        for c, x in enumerate(starts_c):
            tile = g[y:y + th, x:x + tw]
            _require(set(map(int, np.unique(tile))) <= {bg, wall, *studs}, 'foreign tile color')
            for u, v in ((0, tw - 1), (th - 1, 0), (th - 1, tw - 1)):
                _require(tile[u, v] == wall, 'missing structural corner')
            token = int(tile[0, 0])
            _require(token in (wall, *studs), 'missing upper-left corner')
            for stud in studs:
                actual = set(map(tuple, np.argwhere(tile == stud)))
                _require(actual == ({(0, 0)} if token == stud else set()), 'stud outside its corner')
            if token in positions:
                positions[token].add((r, c))
            tiles[r, c] = (y, x, tile)
            occupied[y:y + th, x:x + tw] = True
    occupied[-2:, :] = True
    _require(np.all(g[~occupied] == bg), 'nonempty gutter or margin')
    rects, used = {}, set()
    for stud, fill in mappings:
        pos = positions[stud]
        rows = tuple(sorted({r for r, _ in pos}))
        cols = tuple(sorted({c for _, c in pos}))
        _require(len(rows) == len(cols) == 2 and pos == {(r, c) for r in rows for c in cols}, 'studs are not four rectangle corners')
        rect = (rows, cols)
        edges = _edges(rect)
        _require(not edges & used, 'incompatible shared gutter')
        used |= edges
        rects[fill] = rect
    return dict(grid=g, bg=bg, wall=wall, mappings=mappings, tiles=tiles,
                starts_r=starts_r, starts_c=starts_c, th=th, tw=tw,
                gy=gy, gx=gx, rects=rects, nrows=len(rr), ncols=len(cc))


def _sources(d):
    """Activate only B's side patches, without opening frame walls."""
    sources = np.full(d['grid'].shape, -1, dtype=np.int8)
    for fill, rect in d['rects'].items():
        for axis, r, c in _edges(rect):
            y, x = d['starts_r'][r], d['starts_c'][c]
            if axis == 'H':
                sources[y + 1:y + d['th'] - 1, x + d['tw']:x + d['tw'] + d['gx']] = fill
            else:
                sources[y + d['th']:y + d['th'] + d['gy'], x + 1:x + d['tw'] - 1] = fill
    return sources


def _component_inks(cells, y, x, sources):
    inks = set()
    h, w = sources.shape
    for r, c in cells:
        for dr, dc in _DIRS:
            u, v = y + r + dr, x + c + dc
            if 0 <= u < h and 0 <= v < w and sources[u, v] >= 0:
                inks.add(int(sources[u, v]))
    return inks


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill exactly the tile components touching an inferred gutter source."""
    d = _decode(input_grid)
    out = d['grid'].copy()
    sources = _sources(d)
    out[sources >= 0] = sources[sources >= 0]
    for y, x, tile in d['tiles'].values():
        for cells in _parts(tile == d['bg']):
            inks = _component_inks(cells, y, x, sources)
            _require(len(inks) <= 1, 'different inks meet in one chamber component')
            if inks:
                color = next(iter(inks))
                for r, c in cells:
                    out[y + r, x + c] = color
    return out


def _frame(th, tw):
    walls = np.zeros((th, tw), bool)
    walls[0] = walls[-1] = True
    walls[:, 0] = walls[:, -1] = True
    return walls


def _port(walls, side, position):
    if side == 'N':
        walls[0, position] = False
    elif side == 'S':
        walls[-1, position] = False
    elif side == 'W':
        walls[position, 0] = False
    else:
        walls[position, -1] = False


def _chamber(rng, th, tw, style):
    walls = _frame(th, tw)
    if style == 'sealed':
        return walls
    if style in ('split_v', 'split_h'):
        vertical = style == 'split_v'
        if vertical:
            walls[1:-1, tw // 2] = True
            _port(walls, 'W', int(rng.integers(1, th - 1)))
            _port(walls, 'E', int(rng.integers(1, th - 1)))
        else:
            walls[th // 2, 1:-1] = True
            _port(walls, 'N', int(rng.integers(1, tw - 1)))
            _port(walls, 'S', int(rng.integers(1, tw - 1)))
        return walls
    if style == 'ring':
        # A 3x3 internal wall ring; an extra row or column leaves exterior space.
        walls[1:4, 1:4] = True
        walls[2, 2] = False
        eligible = []
        if th >= 6:
            eligible.append('S')
        if tw >= 6:
            eligible.append('E')
        _require(eligible, 'ring needs room outside it')
        for side in eligible:
            _port(walls, side, 2)
        # Reflect to avoid an orientation cue; protected corners remain intact.
        if rng.random() < .5:
            walls = walls[::-1, :]
        if rng.random() < .5:
            walls = walls[:, ::-1]
        return walls.copy()
    if style == 'island':
        walls[th // 2, tw // 2] = True
    elif style == 'elbow':
        walls[2:-1, tw // 2] = True
        walls[2, 2:tw // 2 + 1] = True
    elif style != 'plain':
        raise ValueError('unknown chamber style')
    for side in rng.choice(_SIDES, size=int(rng.integers(1, 5)), replace=False):
        extent = tw if side in ('N', 'S') else th
        _port(walls, side, int(rng.integers(1, extent - 1)))
    return walls


def _geometry(rng, count, sizes):
    options = []
    for th, tw in sizes:
        for gy in (1, 2):
            for gx in (1, 2):
                for nr in range(2, 5):
                    for nc in range(2, 5):
                        if nr * th + (nr - 1) * gy + 4 > 30 or nc * tw + (nc - 1) * gx + 2 > 30:
                            continue
                        if nr * nc < (6 if count == 1 else 10):
                            continue
                        if count == 2 and max(nr, nc) < 4:
                            continue
                        options.append((th, tw, gy, gx, nr, nc))
    return options[int(rng.integers(len(options)))]


def _choose_rects(rng, nr, nc, count):
    rects = [(a, b) for a in combinations(range(nr), 2) for b in combinations(range(nc), 2)]
    for i in rng.permutation(len(rects)):
        first = rects[int(i)]
        if max(first[0][1] - first[0][0], first[1][1] - first[1][0]) < 2:
            continue
        if count == 1:
            return [first]
        compatible = [r for r in rects if not _corners(first) & _corners(r) and not _edges(first) & _edges(r)]
        if compatible:
            return [first, compatible[int(rng.integers(len(compatible)))]]
    raise AssertionError('no compatible rectangles')


def _make_pair(rng, count, sizes, include_ring):
    for attempt in range(150):
        th, tw, gy, gx, nr, nc = _geometry(rng, count, sizes)
        rects = _choose_rects(rng, nr, nc, count)
        side_maps = [_tile_sides(r) for r in rects]
        users = {p: [k for k, m in enumerate(side_maps) if p in m]
                 for p in ((r, c) for r in range(nr) for c in range(nc))}
        corners = [p for k, rect in enumerate(rects) for p in sorted(_corners(rect)) if users[p] == [k]]
        if len(corners) < 2:
            continue
        split_pos, ring_pos = [corners[int(i)] for i in rng.choice(len(corners), 2, replace=False)]
        bh, bw = nr * th + (nr - 1) * gy, nc * tw + (nc - 1) * gx
        colors = [int(x) for x in rng.permutation(10)]
        bg, wall = colors[:2]
        mappings = [(colors[2 + 2 * k], colors[3 + 2 * k]) for k in range(count)]
        g = np.full((bh + 4, bw + 2), bg, np.int8)
        for r in range(nr):
            for c in range(nc):
                pos = (r, c)
                if len(users[pos]) > 1:
                    style = 'sealed'
                elif pos == split_pos:
                    style = str(rng.choice(('split_v', 'split_h')))
                elif pos == ring_pos and include_ring:
                    style = 'ring'
                else:
                    styles = ('plain', 'plain', 'sealed', 'split_v', 'split_h', 'island', 'elbow')
                    style = str(rng.choice(styles))
                    if include_ring and rng.random() < .12:
                        style = 'ring'
                walls = _chamber(rng, th, tw, style)
                tile = np.where(walls, wall, bg).astype(np.int8)
                for k, rect in enumerate(rects):
                    if pos in _corners(rect):
                        tile[0, 0] = mappings[k][0]
                y, x = 1 + r * (th + gy), 1 + c * (tw + gx)
                g[y:y + th, x:x + tw] = tile
        for k, (stud, fill) in enumerate(mappings):
            g[-2:, 3 * k] = stud
            g[-2:, 3 * k + 1] = fill
        try:
            d = _decode(g)
            out = solve(g)
            features = _features(d, out)
            if not all(features[k] for k in ('partial', 'wet_inside', 'dry_open', 'dry_closed', 'unmarked_wet')):
                continue
            if include_ring and not features['wet_and_hole']:
                continue
            return {'input': g, 'output': out}
        except AssertionError:
            continue
    raise RuntimeError('bounded chamber construction exhausted')


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nv = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    scale_shift = bool(rng.integers(2))
    small = [(5, 6), (6, 5), (6, 6)]
    large = [(7, 5), (5, 7), (7, 6), (6, 7)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nv)):
        for i in range(count):
            circuits = (1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3))) if split == 'train' else (int(rng.integers(1, 3)) if scale_shift else 2)
            ring = circuits == 1 if split == 'train' else True
            sizes = small if scale_shift and split == 'train' else large if scale_shift else small + large
            for _ in range(30):
                pair = _make_pair(rng, circuits, sizes, ring)
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not make distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(d, out):
    bg = d['bg']
    f = dict(partial=False, wet_inside=False, dry_open=False, dry_closed=False, unmarked_wet=False,
             wet_and_hole=False, wet_tiles=0, dry_tiles=0, shared_tiles=0)
    sides = [_tile_sides(r) for r in d['rects'].values()]
    for pos, (y, x, tile) in d['tiles'].items():
        result = out[y:y + d['th'], x:x + d['tw']]
        wet = (tile == bg) & (result != bg)
        dry = (tile == bg) & (result == bg)
        f['partial'] |= bool(wet[1:-1, 1:-1].any() and dry[1:-1, 1:-1].any())
        f['wet_inside'] |= bool(wet[1:-1, 1:-1].any())
        f['unmarked_wet'] |= bool(wet[1:-1, 1:-1].any() and tile[0, 0] == d['wall'])
        f['wet_tiles'] += bool(wet.any())
        f['dry_tiles'] += bool(dry.any())
        f['shared_tiles'] += sum(pos in m for m in sides) > 1
        local_hole = False
        for cells in _parts(tile == bg):
            touches = any(r in (0, d['th'] - 1) or c in (0, d['tw'] - 1) for r, c in cells)
            unchanged = all(result[r, c] == bg for r, c in cells)
            f['dry_open'] |= touches and unchanged
            f['dry_closed'] |= not touches and unchanged
            local_hole |= not touches and unchanged
        f['wet_and_hole'] |= local_hole and bool(wet.any())
    return f


def _reference_output(d):
    """A different check: flood from each ink's patches in its allowed domain."""
    g, bg = d['grid'], d['bg']
    sources = _sources(d)
    allowed = np.zeros(g.shape, bool)
    for y, x, tile in d['tiles'].values():
        allowed[y:y + d['th'], x:x + d['tw']] = tile == bg
    out = g.copy()
    assigned = np.full(g.shape, -1, np.int8)
    for color in d['rects']:
        seen = sources == color
        queue = [tuple(map(int, p)) for p in np.argwhere(seen)]
        domain = allowed | seen
        for r, c in queue:
            for dr, dc in _DIRS:
                u, v = r + dr, c + dc
                if 0 <= u < g.shape[0] and 0 <= v < g.shape[1] and domain[u, v] and not seen[u, v]:
                    seen[u, v] = True
                    queue.append((u, v))
        _require(np.all((assigned[seen] < 0) | (assigned[seen] == color)), 'reference ink collision')
        assigned[seen] = color
        out[seen] = color
    return out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training_counts = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            g, out = _grid(pair['input']), _grid(pair['output'])
            _require(g.shape == out.shape, 'shape changed')
            sig = _signature(g)
            _require(sig not in seen, 'duplicate input')
            seen.add(sig)
            d = _decode(g)
            _require(np.array_equal(out, _reference_output(d)), 'incorrect circuit flooding')
            _require(np.array_equal(out, solve(g)), 'input-only solve disagrees')
            src = _sources(d)
            _require(np.all(out[src >= 0] == src[src >= 0]), 'incomplete side patches')
            _require(np.all(out[g != d['bg']] == g[g != d['bg']]), 'wall, stud, or key changed')
            _require(np.array_equal(out[-3:], g[-3:]), 'key or separator changed')
            features = _features(d, out)
            _require(all(features[k] for k in ('partial', 'wet_inside', 'dry_open', 'dry_closed', 'unmarked_wet')), 'missing topology counterexample')
            # A-only shortcut: flood every tile boundary. A dry open component
            # witnesses its failure. B-only shortcut: paint side gutters alone.
            _require(np.any((g != out) & (src < 0)), 'B-only shortcut gives the answer')
            _require(features['dry_open'], 'unrestricted exterior fill gives the answer')
            if split == 'train':
                training_counts.add(len(d['rects']))
    _require(training_counts == {1, 2}, 'training must expose one and two circuits')


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed diversity, semantic, determinism, and mutation checks."""
    signatures, shapes, geometries, layouts, palettes, structures = (set() for _ in range(6))
    test_counts, shared, ring_tests, axis_pairs = set(), 0, 0, set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_signature(x['input']) for s in ('train', 'test') for x in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                g, out = pair['input'], pair['output']
                d = _decode(g)
                f = _features(d, out)
                shapes.add(g.shape)
                geometries.add((d['th'], d['tw'], d['gy'], d['gx'], d['nrows'], d['ncols']))
                layouts.add((d['nrows'], d['ncols'], tuple(d['rects'].values())))
                palettes.add((d['bg'], d['wall'], tuple(d['mappings'])))
                roles = {}
                structures.add(tuple(roles.setdefault(int(c), len(roles)) for c in g.flat))
                axis_pairs.add((d['gy'], d['gx']))
                shared += f['shared_tiles']
                if split == 'test':
                    test_counts.add(len(d['rects']))
                    ring_tests += f['wet_and_hole']
                # Alter a keyed ink color only: geometry stays fixed, ink follows.
                used = set(map(int, np.unique(g)))
                unused = next(c for c in range(10) if c not in used)
                old = d['mappings'][0][1]
                mutated = g.copy()
                mutated[-2:, 1] = unused
                expected = out.copy()
                expected[out == old] = unused
                _require(np.array_equal(solve(mutated), expected), 'key remapping not obeyed')
                if len(d['mappings']) == 2:
                    swapped = g.copy()
                    swapped[-2:, 0:2], swapped[-2:, 3:5] = g[-2:, 3:5], g[-2:, 0:2]
                    expected_swap = out.copy()
                    expected_swap[-2:] = swapped[-2:]
                    _require(np.array_equal(solve(swapped), expected_swap), 'key order changes chamber painting')
    _require(len(signatures) == 42, 'seed duplication')
    _require(len(shapes) >= 25 and len(geometries) >= 35, 'geometry diversity too narrow')
    _require(len(layouts) >= 35 and len(palettes) >= 150 and len(structures) >= 180, 'structural diversity too narrow')
    _require(test_counts == {1, 2} and axis_pairs == {(1, 1), (1, 2), (2, 1), (2, 2)}, 'count or gutter coverage missing')
    _require(shared > 0 and ring_tests == 42, 'combined topology coverage missing')
    _require(_equal(generate_puzzle(987), generate_puzzle(987)), 'nondeterminism')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(17, a, b)
            _require(len(p['train']) == a and len(p['test']) == b, 'legal count failure')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A corrupted dry room and a missing source patch must both be rejected.
    p = generate_puzzle(5, 2, 1)
    for mode in ('dry', 'patch'):
        broken = {s: [{'input': q['input'].copy(), 'output': q['output'].copy()} for q in p[s]] for s in p}
        q = broken['train'][0]
        d = _decode(q['input'])
        if mode == 'patch':
            r, c = np.argwhere(_sources(d) >= 0)[0]
            q['output'][r, c] = d['bg']
        else:
            options = np.argwhere((q['input'] == d['bg']) & (q['output'] == d['bg']))
            for r, c in options:
                if any(y < r < y + d['th'] - 1 and x < c < x + d['tw'] - 1 for y, x, _ in d['tiles'].values()):
                    q['output'][r, c] = d['mappings'][0][1]
                    break
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('semantic corruption accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'shapes': len(shapes),
            'geometries': len(geometries), 'layouts': len(layouts),
            'palettes': len(palettes), 'structures': len(structures), 'shared_tiles': shared}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
