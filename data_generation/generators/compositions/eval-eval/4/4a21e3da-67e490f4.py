"""Regional shape elections after border-axis dissection.

Only NumPy and the standard library are used. Sampling metadata constructs the
answers; solve independently reconstructs all elections from the input grid.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape),
             "grids must be two dimensional with dimensions 1..30")
    _require(np.issubdtype(g.dtype, np.integer) and np.all((g >= 0) & (g <= 9)),
             "grid colors must be integers in 0..9")
    return g.astype(np.int8, copy=False)


def _neighbors(p):
    r, c = p
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        found, stack = {start}, [start]
        while stack:
            for p in _neighbors(stack.pop()):
                if p in todo:
                    todo.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(tuple(sorted(found)))
    return result


def _normal(cells):
    cells = tuple(cells)
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _rotate(s):
    return _normal((c, -r) for r, c in s)


def _reflect(s):
    return _normal((r, -c) for r, c in s)


@lru_cache(maxsize=4096)
def _rotations(s):
    found = set()
    for _ in range(4):
        found.add(s)
        s = _rotate(s)
    return tuple(sorted(found))


@lru_cache(maxsize=4096)
def _variants(s):
    return tuple(sorted(set(_rotations(s)) | set(_rotations(_reflect(s)))))


def _canonical(cells):
    return min(_variants(_normal(cells)))


def _extent(s):
    return max(r for r, c in s) + 1, max(c for r, c in s) + 1


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _key(g):
    return g.shape, g.astype(np.int8, copy=False).tobytes()


def _choice(rng, items):
    return items[int(rng.integers(len(items)))]


def _region_for(r, c, ar, ac, sides):
    if r == ar or c == ac:
        return None
    v = int(r > ar) if ar is not None else int('bottom' in sides)
    h = int(c > ac) if ac is not None else int('right' in sides)
    return v, h


def _active(corner, sides):
    v, h = corner
    return ('bottom' if v else 'top') in sides or ('right' if h else 'left') in sides


def _read_input(value):
    """Parse the visible roles, axes, fragments, and regional component ballots."""
    g = _grid(value)
    h, w = g.shape
    _require(min(h, w) >= 5, "a scene needs room on both sides of its axes")
    bg = int(g[0, 0])
    _require(all(int(g[p]) == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1))),
             "all four corner cells must be background")
    rim = {(0, c) for c in range(w)} | {(h - 1, c) for c in range(w)}
    rim |= {(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)}
    marker_cells = sorted(p for p in rim if int(g[p]) != bg)
    _require(len(marker_cells) in (1, 2), "one or two border markers are required")
    mark = int(g[marker_cells[0]])
    _require(all(int(g[p]) == mark for p in marker_cells), "markers must share a color")
    _require(np.count_nonzero(g == mark) == len(marker_cells), "marker color is reserved")
    markers = {}
    for r, c in marker_cells:
        side = 'top' if r == 0 else 'bottom' if r == h - 1 else 'left' if c == 0 else 'right'
        _require(side not in markers, "at most one marker per side")
        markers[side] = (r, c)
    sides = set(markers)
    vertical = sides & {'top', 'bottom'}
    horizontal = sides & {'left', 'right'}
    _require(len(vertical) <= 1 and len(horizontal) <= 1, "two markers must be adjacent")
    ac = markers[next(iter(vertical))][1] if vertical else None
    ar = markers[next(iter(horizontal))][0] if horizontal else None
    cells = [tuple(map(int, p)) for p in np.argwhere(g != bg) if tuple(map(int, p)) not in rim]
    fixed = {p: int(g[p]) for p in cells if p[0] == ar or p[1] == ac}
    paths = {}
    for side, (r, c) in markers.items():
        if side in ('top', 'bottom'):
            hits = [rr for rr, cc in fixed if cc == c]
            _require(bool(hits), "each marker axis must contain occupied interior cells")
            rows = range(0, max(hits) + 1) if side == 'top' else range(min(hits), h)
            paths[side] = tuple((rr, c) for rr in rows)
        else:
            hits = [cc for rr, cc in fixed if rr == r]
            _require(bool(hits), "each marker axis must contain occupied interior cells")
            cols = range(0, max(hits) + 1) if side == 'left' else range(min(hits), w)
            paths[side] = tuple((r, cc) for cc in cols)
    regions = defaultdict(list)
    for p in cells:
        corner = _region_for(*p, ar, ac, sides)
        if corner is not None:
            regions[corner].append(p)
    fragments = {}
    votes = {}
    for corner, region in sorted(regions.items()):
        fragments[corner] = []
        votes[corner] = defaultdict(Counter)
        # Components are segmented after the axes are removed. A source object
        # can therefore supply more than one component, even in one region.
        for part in _components(region):
            colors = {int(g[p]) for p in part}
            _require(len(colors) == 1, "off-axis fragments must be monochromatic and separated")
            color = colors.pop()
            shape = _canonical(part)
            fragments[corner].append((color, part, shape))
            votes[corner][shape][color] += 1
    winners = {}
    for corner, ballot in votes.items():
        winners[corner] = {}
        for shape, counts in ballot.items():
            best = max(counts.values())
            colors = [c for c, n in counts.items() if n == best]
            _require(len(colors) == 1, "ties are outside the family; no tie-breaking rule")
            winners[corner][shape] = colors[0]
    return dict(grid=g, bg=bg, mark=mark, markers=markers, sides=sides,
                ar=ar, ac=ac, fixed=fixed, paths=paths, fragments=fragments,
                votes=votes, winners=winners)


def _place(out, colored_cells, corner, box_cells=None):
    if not colored_cells:
        return
    r0, r1, c0, c1 = _bounds(box_cells if box_cells is not None else [p for p, c in colored_cells])
    dr = out.shape[0] - 1 - r1 if corner[0] else -r0
    dc = out.shape[1] - 1 - c1 if corner[1] else -c0
    for (r, c), color in colored_cells:
        out[r + dr, c + dc] = color


def _answer(data, selected=None, *, separate=False, before=False, nearest=False, move=True):
    out = np.full_like(data['grid'], data['bg'])
    for p, color in data['fixed'].items():
        out[p] = color
    for side, path in data['paths'].items():
        if nearest:
            marker = data['markers'][side]
            path = sorted(path, key=lambda p: abs(p[0] - marker[0]) + abs(p[1] - marker[1]))
            stop = next(i for i, p in enumerate(path) if p in data['fixed'])
            path = path[:stop + 1]
        for p in path:
            if out[p] == data['bg']:
                out[p] = data['mark']
    for corner, fs in data['fragments'].items():
        if not _active(corner, data['sides']):
            continue
        if selected is None:
            keep = [f for f in fs if f[0] == data['winners'][corner][f[2]]]
        else:
            keep = [f for i, f in enumerate(fs) if i in selected.get(corner, set())]
        colored = [(p, color) for color, part, shape in keep for p in part]
        if not move:
            for p, color in colored:
                out[p] = color
        elif separate:
            for color, part, shape in keep:
                _place(out, [(p, color) for p in part], corner)
        else:
            all_cells = [p for color, part, shape in fs for p in part] if before else None
            _place(out, colored, corner, all_cells)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone, with fixed geometric family conventions."""
    return _answer(_read_input(input_grid))


def _alternative(data, mode):
    """Input-derived deliberate shortcuts for teaching/validation diagnostics.

    These are mutants of the intended rule, not independent semantic solvers.
    Tied shortcut elections retain every tied color, avoiding invented tie breaks.
    """
    if mode == 'in_place':
        return _answer(data, move=False)
    if mode == 'components':
        return _answer(data, separate=True)
    if mode == 'old_box':
        return _answer(data, before=True)
    if mode == 'nearest':
        return _answer(data, nearest=True)
    if mode == 'all':
        return _answer(data, {k: set(range(len(fs))) for k, fs in data['fragments'].items()})
    def signature(part):
        raw = _normal(part)
        if mode == 'area':
            return len(raw)
        if mode == 'box':
            return tuple(sorted(_extent(raw)))
        if mode == 'rotation':
            return min(_rotations(raw))
        if mode == 'translation':
            return raw
        return _canonical(part)
    parent_for = {}
    if mode == 'precut':
        g = data['grid']
        for color in sorted(set(map(int, np.unique(g))) - {data['bg'], data['mark']}):
            for part in _components(map(tuple, np.argwhere(g == color))):
                identity = part[0]
                for p in part:
                    parent_for[p] = (identity, _canonical(part))
    ballots = defaultdict(lambda: defaultdict(Counter))
    keys = {}
    for corner, fs in data['fragments'].items():
        seen = set()
        for i, (color, part, shape) in enumerate(fs):
            key = parent_for[part[0]][1] if mode == 'precut' else signature(part)
            keys[corner, i] = key
            group = None if mode == 'global' else corner
            identity = parent_for[part[0]][0] if mode == 'precut' else i
            if identity not in seen:
                ballots[group][key][color] += 1
                seen.add(identity)
    selected = defaultdict(set)
    for corner, fs in data['fragments'].items():
        group = None if mode == 'global' else corner
        for i, (color, part, shape) in enumerate(fs):
            counts = ballots[group][keys[corner, i]]
            if counts[color] == max(counts.values()):
                selected[corner].add(i)
    return _answer(data, selected)


def _shape_pool(rng):
    pool = {((0, 0),)}
    for n in range(2, 8):
        pool.add(_canonical((0, c) for c in range(n)))
    for h in range(2, 5):
        for w in range(2, 5):
            pool.add(_canonical({(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}))
            if h * w <= 8:
                pool.add(_canonical((r, c) for r in range(h) for c in range(w)))
            if w >= 3:
                pool.add(_canonical({(r, 0) for r in range(h)} |
                                    {(r, w - 1) for r in range(h)} |
                                    {(h - 1, c) for c in range(w)}))
    pool.add(_canonical(((0, 0), (0, 1), (0, 2), (1, 1))))
    pool.add(_canonical(((0, 0), (0, 1), (1, 1), (1, 2))))
    for _ in range(70):
        cells = {(0, 0)}
        for _ in range(int(rng.integers(3, 8))):
            frontier = sorted({q for p in cells for q in _neighbors(p)} - cells)
            cells.add(_choice(rng, frontier))
        shape = _canonical(cells)
        if max(_extent(shape)) <= 5:
            pool.add(shape)
    return tuple(sorted(pool))


def _specs(shape, colors, counts, allowed=None):
    variants = _variants(shape) if allowed is None else allowed
    return [(shape, int(color), variants) for color, n in zip(colors, counts) for _ in range(n)]


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = False


def _positions(available, shape):
    h, w = _extent(shape)
    nr, nc = available.shape[0] - h + 1, available.shape[1] - w + 1
    if min(nr, nc) <= 0:
        return []
    valid = np.ones((nr, nc), dtype=bool)
    for r, c in shape:
        valid &= available[r:r + nr, c:c + nc]
    return np.argwhere(valid)


def _pack(available, allowed, rng):
    for j in rng.permutation(len(allowed)):
        shape = allowed[int(j)]
        positions = _positions(available, shape)
        if len(positions):
            r0, c0 = map(int, positions[int(rng.integers(len(positions)))])
            cells = tuple((r0 + r, c0 + c) for r, c in shape)
            _block(available, cells)
            return cells
    return None


def _region_bounds(h, w, ar, ac):
    rows = [(1, h - 1)] if ar is None else [(1, ar), (ar + 1, h - 1)]
    cols = [(1, ac), (ac + 1, w - 1)]
    return {(v, s): (r0, r1, c0, c1) for v, (r0, r1) in enumerate(rows)
            for s, (c0, c1) in enumerate(cols)}


def _draw(rng, h, w, double, roles, specifications, winners, *, bridge=False):
    """Pack prescribed ballots; construct ground truth from their sampled winners."""
    bg, mark, colors = roles
    ar = h // 2 + int(rng.integers(-1, 2)) if double else None
    ac = w // 2 + int(rng.integers(-1, 2))
    sides = {'top', 'right'} if double else {'top'}
    bounds = _region_bounds(h, w, ar, ac)
    placements, fixed = {}, {}
    # A bridge makes two differently voting fragments part of one original
    # source object. Other fragment placements are freely packed, not slotted.
    bridge_row = int(rng.integers(7, h - 7)) if bridge else None
    for corner, specs in sorted(specifications.items()):
        r0, r1, c0, c1 = bounds[corner]
        available = np.zeros((h, w), dtype=bool)
        available[r0:r1, c0:c1] = True
        placed, remaining = [], list(specs)
        if bridge and corner[0] == 0:
            shape, color, allowed = remaining.pop(0)
            raw = _choice(rng, allowed)
            edge_col = max(c for r, c in raw) if corner[1] == 0 else 0
            edge_rows = [r for r, c in raw if c == edge_col]
            edge_row = _choice(rng, edge_rows)
            ph, pw = _extent(raw)
            origin_r = bridge_row - edge_row
            origin_c = ac - pw if corner[1] == 0 else ac + 1
            cells = tuple((origin_r + r, origin_c + c) for r, c in raw)
            if not all(available[p] for p in cells):
                return None
            placed.append((color, cells, shape))
            _block(available, cells)
            fixed[bridge_row, ac] = color
        # Random order among similarly sized objects avoids a color-position map.
        rng.shuffle(remaining)
        remaining.sort(key=lambda s: (max(_extent(s[0])), len(s[0])), reverse=True)
        for shape, color, allowed in remaining:
            cells = _pack(available, allowed, rng)
            if cells is None:
                return None
            placed.append((color, cells, shape))
        placements[corner] = placed
    if double:
        for p in ((ar - 3, ac), (ar + 3, ac), (ar, ac - 3), (ar, ac + 3)):
            fixed[p] = int(_choice(rng, colors))
        if rng.random() < .45:
            fixed[ar, ac] = int(_choice(rng, colors))
        # Optional additional runs keep guide geometry independent of ballots.
        if rng.random() < .4:
            fixed[ar - 4, ac] = fixed[ar - 3, ac]
    else:
        a = bridge_row if bridge else int(rng.integers(6, h - 6))
        fixed[a - 3, ac] = int(_choice(rng, colors))
        fixed[a + 3, ac] = int(_choice(rng, colors))
        if rng.random() < .5:
            fixed[a + 4, ac] = fixed[a + 3, ac]
    inp = np.full((h, w), bg, dtype=np.int8)
    for fs in placements.values():
        for color, cells, shape in fs:
            for p in cells:
                inp[p] = color
    for p, color in fixed.items():
        inp[p] = color
    inp[0, ac] = mark
    if double:
        inp[ar, w - 1] = mark
    # Answers come from the prescribed class elections, not a call to solve.
    out = np.full_like(inp, bg)
    for p, color in fixed.items():
        out[p] = color
    for r in range(max(r for r, c in fixed if c == ac) + 1):
        if out[r, ac] == bg:
            out[r, ac] = mark
    if double:
        for c in range(min(c for r, c in fixed if r == ar), w):
            if out[ar, c] == bg:
                out[ar, c] = mark
    for corner, fs in placements.items():
        if not _active(corner, sides):
            continue
        cells = [(p, color) for color, part, shape in fs
                 if color == winners[corner][shape] for p in part]
        _place(out, cells, corner)
    return {'input': inp, 'output': out}


def _teaching_specs(anchor, companion, colors, double):
    x, y, z, q = colors[:4]
    normal = _rotations(anchor)
    mirror = _rotations(_reflect(anchor))
    if not double:
        specs = {
            (0, 0): _specs(anchor, [x], [2], mirror) + _specs(anchor, [y, z], [1, 1], normal)
                    + _specs(companion, [q, y], [2, 1]),
            (0, 1): _specs(anchor, [x], [1], normal) + _specs(anchor, [y], [3], mirror)
                    + _specs(anchor, [z], [1], normal) + _specs(companion, [q, x], [2, 1]),
        }
        winners = {(0, 0): {anchor: x, companion: q}, (0, 1): {anchor: y, companion: q}}
    else:
        singleton, domino = ((0, 0),), ((0, 0), (0, 1))
        specs, winners = {}, {}
        for i, corner in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            win, lose = colors[(i + 1) % len(colors)], colors[(i + 2) % len(colors)]
            specs[corner] = _specs(anchor, [win, lose], [2, 1])
            winners[corner] = {anchor: win}
            small = singleton if i == 0 else domino if i % 2 else singleton
            if i == 0:
                specs[corner] += _specs(small, [q, x, z], [3, 2, 2])
                winners[corner][small] = q
            else:
                specs[corner] += _specs(small, [colors[(i + 3) % len(colors)]], [1 + i % 2])
                winners[corner][small] = colors[(i + 3) % len(colors)]
    return specs, winners


def _random_specs(rng, pool, colors, double, mandatory=None):
    specs, winners = {}, {}
    corners = ((0, 0), (0, 1), (1, 0), (1, 1)) if double else ((0, 0), (0, 1))
    for i, corner in enumerate(corners):
        k = int(rng.integers(1, 3 if double else 4))
        eligible = [s for s in pool if len(s) <= (6 if double else 8) and max(_extent(s)) <= 4]
        if not eligible:
            eligible = list(pool)
        shapes = [eligible[int(j)] for j in rng.choice(len(eligible), min(k, len(eligible)), replace=False)]
        if i == 0 and mandatory is not None:
            shapes = [mandatory] + [s for s in shapes if s != mandatory][:k - 1]
        specs[corner], winners[corner] = [], {}
        footprint = 0
        for j, shape in enumerate(shapes):
            ballots = [(1,), (2,), (2, 1), (2, 1, 1), (3, 2)]
            if len(shape) <= 2 or not double:
                ballots += [(3, 2, 2), (4, 3, 1, 1)]
            # Two bulky classes in a small quadrant are kept legible and feasible.
            counts = (1,) if double and j else _choice(rng, ballots)
            ph, pw = _extent(shape)
            cost = (ph + 1) * (pw + 1)
            budget = 85 if double else 180
            if footprint + cost * sum(counts) > budget:
                counts = (1,)
            footprint += cost * sum(counts)
            winning = int(colors[(i + j + int(rng.integers(len(colors)))) % len(colors)])
            losers = [c for c in colors if c != winning]
            rng.shuffle(losers)
            ballot_colors = [winning] + losers[:len(counts) - 1]
            specs[corner] += _specs(shape, ballot_colors, counts)
            winners[corner][shape] = winning
    return specs, winners


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Reproducible full puzzle with two compulsory discriminating demonstrations."""
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and low <= value <= high, f'{name} is outside its legal integer range')
    _require(num_train > num_test, 'num_train must exceed num_test')
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             'seed must be a non-negative integer')
    rng = np.random.default_rng(int(seed))
    pool = _shape_pool(rng)
    eligible = [s for s in pool if 4 <= len(s) <= 6 and max(_extent(s)) <= 3
                and _reflect(s) not in _rotations(s)]
    eligible = [s for s in eligible if any(t != s and len(t) == len(s)
                and sorted(_extent(t)) == sorted(_extent(s)) for t in pool)]
    anchor = _choice(rng, eligible)
    companion = _choice(rng, [s for s in pool if s != anchor and len(s) == len(anchor)
                              and sorted(_extent(s)) == sorted(_extent(anchor))])
    base_turn = int(rng.integers(4))
    rotate_test = bool(rng.integers(2))
    palette_varies = rng.random() < .7
    palette = list(map(int, rng.permutation(10)))
    roles = (palette[0], palette[1], palette[2:])
    puzzle, seen, known = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            teaching = split == 'train' and i < 2
            double = i == 1 if teaching else bool(rng.integers(2))
            if palette_varies and not (split == 'train' and i == 0):
                palette = list(map(int, rng.permutation(10)))
                roles = (palette[0], palette[1], palette[2:])
            mandatory = None
            if split == 'test':
                unseen = [s for s in pool if s not in known and 3 <= len(s) <= 6 and max(_extent(s)) <= 4]
                _require(bool(unseen), 'procedural repertoire must contain a new supported test shape')
                mandatory = _choice(rng, unseen)
            if teaching:
                specs, winners = _teaching_specs(anchor, companion, roles[2], double)
            else:
                specs, winners = _random_specs(rng, pool, roles[2], double, mandatory)
            turn = base_turn
            if split == 'train' and i >= 2 and rng.random() < .5:
                turn = (base_turn + 2) % 4
            if split == 'test' and rotate_test:
                turn = (base_turn + _choice(rng, (1, 3))) % 4
            for attempt in range(120):
                if teaching:
                    h, w = map(int, rng.integers(27, 31, size=2))
                elif attempt < 50:
                    h, w = map(int, rng.integers(22 if double else 20, 31, size=2))
                else:
                    h = w = 30
                pair = _draw(rng, h, w, double, roles, specs, winners,
                             bridge=teaching and i == 0)
                if pair is None:
                    continue
                pair = {k: np.rot90(g, -turn).copy() for k, g in pair.items()}
                data = _read_input(pair['input'])
                if teaching and i == 0:
                    modes = ('global', 'precut', 'area', 'box', 'rotation', 'translation',
                             'components', 'old_box', 'all')
                    if any(np.array_equal(_alternative(data, m), pair['output']) for m in modes):
                        continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError(f'bounded packing search exhausted at seed {seed}/{split}/{i}')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                known.update(s for ballot in data['votes'].values() for s in ballot)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(data):
    fs = data['fragments']
    retained = [f for corner, parts in fs.items() if _active(corner, data['sides'])
                for f in parts if f[0] == data['winners'][corner][f[2]]]
    ballots = [tuple(sorted(counts.values(), reverse=True))
               for ballot in data['votes'].values() for counts in ballot.values()]
    shapes = {s for ballot in data['votes'].values() for s in ballot}
    return dict(shape=data['grid'].shape, orientation='+'.join(sorted(data['sides'])),
                roles=(data['bg'], data['mark']), shapes=shapes,
                ballots=ballots, fragment_count=sum(map(len, fs.values())),
                retained_fragments=len(retained), max_area=max(map(len, shapes), default=0),
                max_extent=max((max(_extent(s)) for s in shapes), default=0),
                nonmajority=any(len(b) > 1 and b[0] * 2 <= sum(b) for b in ballots),
                repeated_losers=any(len(b) > 1 and b[1] > 1 for b in ballots),
                discarded_cells=sum(len(part) for corner, parts in fs.items()
                                    if not _active(corner, data['sides']) for color, part, s in parts))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal pair counts')
    analyses, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'output must keep the canvas dimensions')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            data = _read_input(inp)
            _require(np.array_equal(_answer(data), out), 'incorrect solved grid')
            _require(not np.array_equal(inp, out), 'identity scene is not generated')
            # Validate conservation independently of translation coordinates:
            # each fixed cell and each winning fragment cell survives once.
            expected = Counter(data['fixed'].values())
            for corner, fs in data['fragments'].items():
                if _active(corner, data['sides']):
                    for color, cells, shape in fs:
                        if color == data['winners'][corner][shape]:
                            expected[color] += len(cells)
            actual = Counter(map(int, out.ravel()))
            for color in set(map(int, np.unique(inp))) - {data['bg'], data['mark']}:
                _require(actual[color] == expected[color], 'fragment cells clipped, merged, or duplicated')
            guide_cells = set(p for path in data['paths'].values() for p in path) - set(data['fixed'])
            _require(actual[data['mark']] == len(guide_cells), 'wrong guide endpoint or gap filling')
            for p, color in data['fixed'].items():
                _require(int(out[p]) == color, 'axial payload must remain fixed')
            for corner, fs in data['fragments'].items():
                if not _active(corner, data['sides']):
                    continue
                keep = [p for color, cells, shape in fs
                        if color == data['winners'][corner][shape] for p in cells]
                if keep:
                    r0, r1, c0, c1 = _bounds(keep)
                    dr = out.shape[0] - 1 - r1 if corner[0] else -r0
                    dc = out.shape[1] - 1 - c1 if corner[1] else -c0
                    targets = [(r + dr, c + dc) for r, c in keep]
                    box = _bounds(targets)
                    _require(box[1 if corner[0] else 0] == (out.shape[0] - 1 if corner[0] else 0)
                             and box[3 if corner[1] else 2] == (out.shape[1] - 1 if corner[1] else 0),
                             'surviving aggregate is not flush with its corner edges')
            analyses[split].append(data)
    train, test = analyses['train'], analyses['test']
    modes = ('global', 'precut', 'area', 'box', 'rotation', 'translation',
             'components', 'old_box', 'all', 'in_place', 'nearest')
    for mode in modes:
        _require(any(not np.array_equal(_alternative(d, mode), p['output'])
                     for d, p in zip(train, puzzle['train'])), f'missing training witness against {mode}')
    _require(any(len(d['markers']) == 1 for d in train)
             and any(len(d['markers']) == 2 and _features(d)['discarded_cells'] > 0 for d in train),
             'training must teach both arities and opposite-region deletion')
    _require(any(_features(d)['nonmajority'] and _features(d)['repeated_losers'] for d in train),
             'training must show a plurality with repeated losing colors')
    # Generated tests shift the repertoire, with fully visible new-shape ballots.
    known = set().union(*(_features(d)['shapes'] for d in train))
    _require(all(_features(d)['shapes'] - known for d in test), 'test must contain a supported new shape')


def _dissection_regression():
    """Hand-specified small case: one source makes two votes in one region.

    Its same-colored branch on the other side loses an independent election.
    This is a direct expected-grid fixture, not an independent solver.
    """
    inp = np.zeros((11, 15), dtype=np.int8)
    inp[0, 7] = 9
    inp[1, 7] = 4
    inp[3:8, 7] = 2
    inp[3, 6] = inp[7, 6] = 2
    inp[4, 8:10] = 2
    inp[5, 2] = 3
    inp[1:3, 1] = 4
    inp[2, 10:12] = 3
    inp[7:9, 11] = 3
    out = np.zeros_like(inp)
    out[:8, 7] = 9
    out[1, 7] = 4
    out[3:8, 7] = 2
    out[0:2, 0] = 4
    out[2, 5] = out[6, 5] = 2
    out[0, 13:15] = 3
    out[5:7, 14] = 3
    return {'input': inp, 'output': out}


def self_test():
    """Bounded checks on 42 seeds, explicit count boundaries, and corruption."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    structures, dimensions, orientations, roles, vocabulary, ballots, fragments = set(), set(), set(), set(), set(), set(), set()
    teaching_failures = Counter()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _read_input(pair['input'])
                f = _features(data)
                mapping = {}
                normalized = tuple(mapping.setdefault(int(c), len(mapping)) for c in pair['input'].ravel())
                structures.add((pair['input'].shape, normalized))
                dimensions.add(f['shape'])
                orientations.add(f['orientation'])
                roles.add(f['roles'])
                vocabulary.update(f['shapes'])
                ballots.update(f['ballots'])
                fragments.add(f['fragment_count'])
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), 'solver disagreement')
                if split == 'train':
                    for mode in ('global', 'precut', 'area', 'box', 'rotation', 'components', 'old_box'):
                        teaching_failures[mode] += not np.array_equal(_alternative(data, mode), pair['output'])
    _require(len(structures) >= 190 and len(dimensions) >= 40, 'insufficient structural/size variation')
    _require(len(orientations) == 8 and len(roles) >= 40, 'insufficient marker/palette variation')
    _require(len(vocabulary) >= 35 and len(ballots) >= 7 and len(fragments) >= 15,
             'insufficient shape, election, or component-count variation')
    _require(all(n >= 42 for n in teaching_failures.values()), 'missing discriminating puzzles')
    fixture = _dissection_regression()
    for reflect in (False, True):
        for turn in range(4):
            inp = np.rot90(np.fliplr(fixture['input']) if reflect else fixture['input'], turn).copy()
            out = np.rot90(np.fliplr(fixture['output']) if reflect else fixture['output'], turn).copy()
            unchanged = inp.copy()
            _require(np.array_equal(solve(inp), out), 'cut multiplicity or rotated small-scene regression')
            _require(np.array_equal(inp, unchanged), 'solve modified its input')
    tied = fixture['input'].copy()
    tied[9, 2] = 3  # now two singleton votes of each color on the left
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError('tied regional election accepted')
    for a, b in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(271828, a, b)
        q = generate_puzzle(271828, a, b)
        _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                     for k in ('input', 'output')), 'seed does not reproduce full puzzle')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=210, color_normalized_structures=len(structures),
                dimensions=len(dimensions), marker_configurations=len(orientations),
                role_pairs=len(roles), shape_classes=len(vocabulary),
                ballot_patterns=len(ballots), fragment_counts=len(fragments),
                training_shortcut_failures=dict(teaching_failures),
                hand_fixture_orientations=8, independent_solver=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
