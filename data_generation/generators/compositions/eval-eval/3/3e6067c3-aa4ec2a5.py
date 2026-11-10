"""Chronological stitching: ordered colored seams cause topological flooding.

Only Python's standard library and NumPy are required. The decoder, topology,
and output conventions depend solely on the supplied input pixels.
"""
from collections import deque
import hashlib
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "illegal grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(((0 <= a) & (a <= 9)).all()), "illegal color")
    return a.astype(np.int8, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        cells, stack = [], [(r, c)]
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _D4:
                yy, xx = y + dy, x + dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _holes(wall):
    """Four-connected background components unreachable from the canvas edge."""
    outside = np.zeros(wall.shape, bool)
    queue = deque()
    h, w = wall.shape
    for r in range(h):
        for c in (0, w - 1):
            if not wall[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in _D4:
            rr, cc = r + dr, c + dc
            if (0 <= rr < h and 0 <= cc < w and not wall[rr, cc]
                    and not outside[rr, cc]):
                outside[rr, cc] = True
                queue.append((rr, cc))
    return ~wall & ~outside


def _dilate(mask):
    """Chebyshev distance at most one, clipped to the grid."""
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
                                 for dr in (-1, 0, 1) for dc in (-1, 0, 1)])


def _line(a, b):
    """Open straight segment: terminal cells themselves are never painted."""
    if a[0] == b[0] and abs(a[1] - b[1]) >= 2:
        return [(a[0], c) for c in range(min(a[1], b[1]) + 1, max(a[1], b[1]))]
    if a[1] == b[1] and abs(a[0] - b[0]) >= 2:
        return [(r, a[1]) for r in range(min(a[0], b[0]) + 1, max(a[0], b[0]))]
    return []


def _clear_graph(grid, nodes, background):
    graph = [[] for _ in nodes]
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            cells = _line(a, nodes[j])
            if cells and all(grid[r, c] == background for r, c in cells):
                graph[i].append(j)
                graph[j].append(i)
    return graph


def _routes(graph, labels, key, limit=2):
    answers = []
    candidates = [[i for i, color in enumerate(labels) if color == token] for token in key]

    def visit(path, used):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for j in candidates[len(path)]:
            if j not in used and (not path or j in graph[path[-1]]):
                visit(path + [j], used | {j})

    visit([], set())
    return answers


def _parse(grid):
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "the full outer frame must identify background")
    occupied_rows = np.flatnonzero(np.any(grid != bg, axis=1))
    _require(len(occupied_rows) >= 3, "missing scene or key")
    key_y = int(occupied_rows[-1])
    key_xs = np.flatnonzero(grid[key_y] != bg)
    _require(3 <= len(key_xs) <= 14 and bool((np.diff(key_xs) == 2).all()),
             "key must be isolated tokens read left to right")
    key = [int(grid[key_y, x]) for x in key_xs]
    _require(len(set(key)) >= 2, "key needs at least two colors")
    # Carrier is visibly the one scene color absent from the key.
    carriers = set(map(int, np.unique(grid))) - {bg} - set(key)
    _require(len(carriers) == 1, "carrier must be the unique non-key scene color")
    carrier = carriers.pop()
    scene = grid.copy()
    scene[key_y] = bg
    node_mask = (scene != bg) & (scene != carrier)
    nodes = [tuple(map(int, p)) for p in np.argwhere(node_mask)]
    _require(3 <= len(nodes) <= 30, "illegal terminal count")
    labels = [int(scene[p]) for p in nodes]
    _require(set(labels) == set(key), "all terminal colors must occur in the key")
    scene_pixels = np.argwhere(scene != bg)
    _require(len(scene_pixels) and key_y - int(scene_pixels[:, 0].max()) >= 3,
             "two empty rows must separate key and network")
    # Terminal patches are isolated single pixels, even when labels repeat.
    _require(all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2
                 for i, a in enumerate(nodes) for b in nodes[i + 1:]),
             "terminals may not touch, even diagonally")
    initial_wall = scene != bg
    _require(not _holes(initial_wall).any(), "input network must have no sealed cavities")
    graph = _clear_graph(scene, nodes, bg)
    paths = _routes(graph, labels, key)
    _require(len(paths) == 1, "key must determine exactly one distinct-terminal route")
    route = paths[0]
    seam_masks, used = [], np.zeros(grid.shape, bool)
    for i, j in zip(route, route[1:]):
        mask = np.zeros(grid.shape, bool)
        for r, c in _line(nodes[i], nodes[j]):
            mask[r, c] = True
        _require(not (used & mask).any(), "seams must not cross or overlap")
        used |= mask
        seam_masks.append(mask)
    return dict(background=bg, carrier=carrier, key_y=key_y, key=key,
                nodes=nodes, labels=labels, graph=graph, route=route,
                initial_wall=initial_wall, seam_masks=seam_masks)


def _replay(grid, info, target_colors=False):
    """Compute topology before outlining; filled cells never become walls."""
    wall = info['initial_wall'].copy()
    out = grid.copy()
    before = np.zeros(grid.shape, bool)
    birth = np.full(grid.shape, -1, dtype=np.int16)
    seam_ids = np.full(grid.shape, -1, dtype=np.int16)
    new_regions, dependencies, interior_seams = [], [], 0
    for step, mask in enumerate(info['seam_masks']):
        interior_seams += int(bool((mask & before).any()))
        wall |= mask
        seam_ids[mask] = step
        color = info['key'][step + int(target_colors)]
        out[mask] = color
        after = _holes(wall)
        new = after & ~before
        birth[new] = step
        out[new] = color
        adjacent = set(map(int, seam_ids[_dilate(new) & wall])) - {-1} if new.any() else set()
        dependencies.append(adjacent)
        new_regions.append(new)
        before = after
    exterior = ~wall & ~before
    outline = _dilate(wall) & exterior
    out[outline] = info['carrier']
    final_seams = seam_ids >= 0
    # A later internal seam may replace a previous fill; it retains seam color.
    for step, mask in enumerate(info['seam_masks']):
        out[mask] = info['key'][step + int(target_colors)]
    return out, dict(wall=wall, holes=before, exterior=exterior, outline=outline,
                     birth=birth, seam_ids=seam_ids, seams=final_seams,
                     new_regions=new_regions, dependencies=dependencies,
                     interior_seams=interior_seams)


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    return _replay(grid, _parse(grid))[0]


def _features(grid, info=None, zones=None):
    info = _parse(grid) if info is None else info
    if zones is None:
        _, zones = _replay(grid, info)
    steps = [i for i, mask in enumerate(zones['new_regions']) if mask.any()]
    fill_colors = {info['key'][i] for i in steps}
    delayed = any(any(j < i and info['key'][j] != info['key'][i]
                      for j in zones['dependencies'][i]) for i in steps)
    outline_trap = _holes(zones['wall'] | zones['outline']) & zones['exterior'] & ~zones['outline']
    route_edges = {tuple(sorted((i, j))) for i, j in zip(info['route'], info['route'][1:])}
    missing = {tuple(sorted((i, j))) for i, js in enumerate(info['graph']) for j in js}
    axes = ['h' if info['nodes'][i][0] == info['nodes'][j][0] else 'v'
            for i, j in zip(info['route'], info['route'][1:])]
    rows = {r for r, c in info['nodes']}
    cols = {c for r, c in info['nodes']}
    return dict(shape=tuple(grid.shape), terminals=len(info['nodes']), length=len(info['key']),
                unused=len(info['nodes']) - len(info['key']), closures=len(steps),
                nonclosures=len(info['seam_masks']) - len(steps), delayed=delayed,
                fill_colors=len(fill_colors), trap=bool(outline_trap.any()),
                trap_mask=outline_trap, interior_seams=zones['interior_seams'],
                repeated=len(set(info['key'])) < len(info['key']),
                axes=set(axes), turns=sum(a != b for a, b in zip(axes, axes[1:])),
                unselected_edges=len(missing - route_edges),
                sparse=len(info['nodes']) < len(rows) * len(cols),
                fragments=len(_components(info['initial_wall'])),
                final_cavities=len(_components(zones['holes'])),
                seam_lengths=tuple(int(mask.sum()) for mask in info['seam_masks']))


def _reverse_fill_changed(grid, info, zones, forward):
    """The same final wall union can have different first-sealing colors."""
    reversed_info = dict(info, key=info['key'][::-1], seam_masks=info['seam_masks'][::-1])
    reverse_out, _ = _replay(grid, reversed_info)
    return bool((forward[zones['holes']] != reverse_out[zones['holes']]).any())


def _path(rng, graph, length):
    budget = 3000

    def visit(path):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            return path
        for j in rng.permutation(graph[path[-1]]):
            if int(j) not in path:
                found = visit(path + [int(j)])
                if found is not None:
                    return found
        return None

    for i in rng.permutation(len(graph)):
        found = visit([int(i)])
        if found is not None:
            return found
        if budget < 0:
            break
    return None


def _layout(rng, kind):
    if kind == 'intro':
        rows, cols, length = 2, int(rng.integers(2, 4)), int(rng.integers(3, 5))
    elif kind == 'chronology':
        rows, cols, length = 3, 4, int(rng.integers(6, 9))
    elif kind == 'divider':
        rows, cols, length = 3, 4, 8
    elif kind == 'test':
        rows, cols = [(3, 4), (4, 3), (4, 4), (4, 5), (5, 4)][int(rng.integers(5))]
        length = int(rng.integers(9, 13))
    else:
        rows, cols = [(2, 3), (3, 3), (3, 4), (4, 3), (4, 4)][int(rng.integers(5))]
        length = int(rng.integers(3, min(8, rows * cols) + 1))
    lattice = [(r, c) for r in range(rows) for c in range(cols)]
    if kind in ('ordinary', 'test') and rng.random() < .55:
        remove = min(len(lattice) - length, int(rng.integers(1, 5)))
        removed = set(map(int, rng.choice(len(lattice), remove, replace=False)))
        lattice = [p for i, p in enumerate(lattice) if i not in removed]
    edges = [(i, j) for i, a in enumerate(lattice) for j, b in enumerate(lattice)
             if i < j and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    graph = [[] for _ in lattice]
    for i, j in edges:
        graph[i].append(j)
        graph[j].append(i)
    path = _path(rng, graph, length)
    if path is None:
        return None
    forbidden = {tuple(sorted((i, j))) for i, j in zip(path, path[1:])}
    parent = list(range(len(lattice)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    walls = []
    probability = float(rng.uniform(.60, 1.0))
    for index in rng.permutation(len(edges)):
        i, j = edges[int(index)]
        if (i, j) not in forbidden and root(i) != root(j) and rng.random() < probability:
            parent[root(i)] = root(j)
            walls.append((i, j))
    if not walls:
        return None
    # Independently uneven coordinate spacing, with guaranteed canvas fit.
    ys = [0] + list(np.cumsum(rng.integers(3, 6 if rows >= 5 else 7, rows - 1)).astype(int))
    xs = [0] + list(np.cumsum(rng.integers(3, 6 if cols >= 5 else 7, cols - 1)).astype(int))
    return [(ys[r], xs[c]) for r, c in lattice], walls, path


def _geometry(nodes, walls, path):
    h = max(r for r, c in nodes) + 3
    w = max(c for r, c in nodes) + 3
    shifted = [(r + 1, c + 1) for r, c in nodes]
    mask = np.zeros((h, w), bool)
    for p in shifted:
        mask[p] = True
    for i, j in walls:
        for p in _line(shifted[i], shifted[j]):
            mask[p] = True
    if _holes(mask).any():
        return None
    seam_masks = []
    for i, j in zip(path, path[1:]):
        seam = np.zeros_like(mask)
        for p in _line(shifted[i], shifted[j]):
            if mask[p]:
                return None
            seam[p] = True
        seam_masks.append(seam)
    # Geometry screen before spending time on repeated-label assignment.
    fake = dict(initial_wall=mask, seam_masks=seam_masks, key=list(range(len(path))), carrier=9)
    _, z = _replay(np.zeros(mask.shape, np.int8), fake)
    return z


def _wanted(z, kind, allow_dividers):
    steps = [i for i, m in enumerate(z['new_regions']) if m.any()]
    if not steps or (not allow_dividers and z['interior_seams']):
        return False
    if kind == 'intro':
        return len(steps) < len(z['new_regions'])
    if kind == 'chronology':
        delayed = any(any(j < i for j in z['dependencies'][i]) for i in steps)
        trap = (_holes(z['wall'] | z['outline']) & z['exterior'] & ~z['outline']).any()
        return len(steps) >= 2 and delayed and bool(trap)
    if kind == 'divider':
        return z['interior_seams'] > 0
    return True


def _label(rng, nodes, wall_mask, path, colors, repeat):
    fake_grid = wall_mask.astype(np.int8)
    graph = _clear_graph(fake_grid, nodes, 0)
    length = len(path)
    upper = min(8, length - int(repeat))
    k = upper if rng.random() < .7 else int(rng.integers(2, upper + 1))
    pool = list(map(int, rng.permutation(colors)[:k]))
    for _ in range(160):
        key = pool + list(map(int, rng.choice(pool, length - k)))
        key = list(map(int, rng.permutation(key)))
        labels = list(map(int, rng.choice(pool, len(nodes))))
        for i, color in zip(path, key):
            labels[i] = color
        answers = _routes(graph, labels, key)
        if len(answers) == 1 and answers[0] == tuple(path):
            return labels, key
    return None


def _render(rng, layout, roles, kind, allow_dividers):
    nodes, walls, path = layout
    z = _geometry(nodes, walls, path)
    if z is None or not _wanted(z, kind, allow_dividers):
        return None
    bg, carrier = roles
    # Draw a tightly bounded network first; reflect/rotate it independently
    # of the key, whose direction remains left to right in every example.
    bh, bw = max(r for r, c in nodes) + 1, max(c for r, c in nodes) + 1
    body = np.zeros((bh, bw), bool)
    for p in nodes:
        body[p] = True
    for i, j in walls:
        for p in _line(nodes[i], nodes[j]):
            body[p] = True
    labeled = _label(rng, nodes, body, path, [c for c in range(10) if c not in roles],
                     repeat=kind == 'chronology')
    if labeled is None:
        return None
    labels, key = labeled
    net = np.where(body, carrier, bg).astype(np.int8)
    for p, color in zip(nodes, labels):
        net[p] = color
    net = np.rot90(net, int(rng.integers(4)))
    if rng.random() < .5:
        net = np.fliplr(net)
    bh, bw = net.shape
    min_h, min_w = bh + 5, max(bw + 2, 2 * len(key) + 1)
    if min_h > 30 or min_w > 30:
        return None
    h = int(rng.integers(min_h, min(30, min_h + 5) + 1))
    w = int(rng.integers(min_w, min(30, min_w + 5) + 1))
    if rng.random() < .12:
        h = w = max(h, w)
    top = int(rng.integers(1, h - bh - 3))
    left = int(rng.integers(1, w - bw))
    key_y = int(rng.integers(top + bh + 2, h - 1))
    key_x = int(rng.integers(1, w - (2 * len(key) - 1)))
    inp = np.full((h, w), bg, np.int8)
    inp[top:top + bh, left:left + bw] = net
    inp[key_y, key_x:key_x + 2 * len(key):2] = key
    info = _parse(inp)
    out, zones = _replay(inp, info)
    f = _features(inp, info, zones)
    if kind == 'intro' and f['axes'] != {'h', 'v'}:
        return None
    if kind == 'chronology' and not (f['delayed'] and f['fill_colors'] >= 2
                                    and f['unselected_edges'] and f['unused']
                                    and f['turns'] >= 2
                                    and _reverse_fill_changed(inp, info, zones, out)):
        return None
    return dict(input=inp, output=out)


def _fallback_layout(kind):
    if kind == 'intro':
        return [(0, 0), (0, 6), (6, 0), (6, 6)], [(0, 2), (2, 3)], [0, 1, 3]
    if kind == 'chronology':
        nodes = [(r, c) for r in (0, 5, 10) for c in (0, 5, 8, 13)]
        walls = [(0, 4), (5, 6), (6, 10), (6, 7), (8, 9), (4, 8), (2, 6), (1, 5), (4, 5)]
        return nodes, walls, [0, 1, 2, 3, 7, 11, 10]
    if kind == 'divider':
        nodes = [(r, c) for r in (0, 5, 10) for c in (0, 4, 8, 12)]
        # The upper enclosure closes before the final internal horizontal cut.
        return nodes, [(0, 4), (4, 8), (8, 9), (9, 10), (10, 11), (7, 11)], [0, 1, 2, 3, 7, 6, 5, 4]
    cols = 5 if kind == 'test' else 3
    nodes = [(r, 4 * c) for r in (0, 6) for c in range(cols)]
    walls = [(i, i + cols) for i in range(cols - 1)]
    path = list(range(cols)) + list(range(2 * cols - 1, cols - 1, -1))
    return nodes, walls, path


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _signature(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    fixed_roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    variable_palette = bool(rng.random() < .65)
    allow_dividers = nt >= 3 and bool(rng.random() < .55)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            kind = ('test' if split == 'test' else 'intro' if index == 0 else
                    'chronology' if index == 1 else 'divider' if index == 2 and allow_dividers else 'ordinary')
            roles = tuple(map(int, rng.choice(10, 2, replace=False))) if variable_palette else fixed_roles
            pair = None
            for _ in range(240 if kind == 'chronology' else 80):
                layout = _layout(rng, kind)
                if layout is not None:
                    candidate = _render(rng, layout, roles, kind, allow_dividers)
                    if candidate is not None and _signature(candidate['input']) not in seen:
                        pair = candidate
                        break
            if pair is None:
                # Fixed geometric witnesses, then visible randomized labels,
                # margins and orientation. No hidden rule changes in fallback.
                for _ in range(160):
                    candidate = _render(rng, _fallback_layout(kind), roles, kind, allow_dividers)
                    if candidate is not None and _signature(candidate['input']) not in seen:
                        pair = candidate
                        break
            _require(pair is not None, 'bounded sampler failed to construct required evidence')
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'dimensions changed')
            sig = _signature(inp)
            _require(sig not in seen, 'duplicate input')
            seen.add(sig)
            info = _parse(inp)
            expected, zones = _replay(inp, info)
            _require(np.array_equal(out, expected), 'output disagrees with causal enclosure rule')
            _require(np.array_equal(out[info['initial_wall']], inp[info['initial_wall']]),
                     'original wall or terminal changed')
            _require(np.array_equal(out[info['key_y']], inp[info['key_y']]), 'key changed')
            _require(not (zones['outline'] & zones['holes']).any(), 'outline entered cavity')
            for step, mask in enumerate(info['seam_masks']):
                _require(bool((out[mask] == info['key'][step]).all()), 'incorrect source seam color')
            for step, new in enumerate(zones['new_regions']):
                fill = new & ~zones['seams']
                _require(bool((out[fill] == info['key'][step]).all()), 'wrong first-sealing fill color')
            far = zones['exterior'] & ~zones['outline']
            far[info['key_y']] = False
            _require(bool((out[far] == info['background']).all()), 'exterior background was filled')
            f = _features(inp, info, zones)
            f['reverse_changes_fill'] = (split == 'train' and f['delayed']
                and f['fill_colors'] >= 2 and _reverse_fill_changed(inp, info, zones, out))
            _require(f['closures'] > 0, 'pair never demonstrates sealing')
            features[split].append(f)
    train, test = features['train'], features['test']
    _require(any(f['nonclosures'] for f in train), 'training must distinguish joining from sealing')
    _require(any(f['delayed'] and f['fill_colors'] >= 2 for f in train),
             'training must reveal chronological, color-dependent sealing')
    _require(any(f['reverse_changes_fill'] for f in train),
             'reversing the order must change cavity fill despite the same final wall mask')
    _require(any(f['trap'] for f in train), 'training must distinguish fill from outline closure')
    _require(any(f['repeated'] and f['unused'] and f['unselected_edges'] for f in train),
             'training must expose key selection with repeated labels and unused terminals')
    _require(set.union(*(f['axes'] for f in train)) == {'h', 'v'}, 'both alignment axes need evidence')
    _require(not any(f['interior_seams'] for f in test) or any(f['interior_seams'] for f in train),
             'internal dividers require training evidence')
    _require(min(f['length'] for f in test) > max(f['length'] for f in train),
             'test key must be longer than every training key')


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed structure, causal semantics, count and corruption checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(_same(puzzles[0], generate_puzzle(0)), 'seed reproducibility failed')
    hashes, shapes, masks, records, palettes = set(), set(), set(), [], set()
    for puzzle in puzzles:
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info = _parse(inp)
                predicted, zones = _replay(inp, info)
                _require(np.array_equal(predicted, out), 'input-only prediction mismatch')
                _require(np.array_equal(solve(inp.tolist()), out), 'list input unsupported')
                digest.update(repr(inp.shape).encode() + inp.tobytes() + out.tobytes())
                shapes.add(tuple(inp.shape))
                masks.add(_signature(info['initial_wall']))
                palettes.add((info['background'], info['carrier']))
                records.append(_features(inp, info, zones))
                # Removing A: no seams, no causal fill; only original outlining.
                no_a = inp.copy()
                no_a[_dilate(info['initial_wall']) & ~info['initial_wall']] = info['carrier']
                no_a[info['key_y']] = inp[info['key_y']]
                _require(not np.array_equal(no_a, out), 'A is dispensable')
                # Removing B: source-colored seams alone.
                no_b = inp.copy()
                for i, seam in enumerate(info['seam_masks']):
                    no_b[seam] = info['key'][i]
                _require(not np.array_equal(no_b, out), 'B is dispensable')
                if split == 'train':
                    target = _replay(inp, info, target_colors=True)[0]
                    _require(not np.array_equal(target, out), 'source/target ambiguity')
                if _features(inp, info, zones)['trap']:
                    trap = _features(inp, info, zones)['trap_mask']
                    _require(bool((out[trap] == info['background']).all()), 'outline trap filled')
        hashes.add(digest.digest())
    _require(len(hashes) == 42, 'cross-seed puzzle repetition')
    _require(len(shapes) >= 40 and len(masks) >= 180, 'insufficient geometric diversity')
    _require(len(palettes) >= 25, 'insufficient palette-role variation')
    _require(len({f['terminals'] for f in records}) >= 7, 'terminal counts do not vary')
    _require(len({f['fragments'] for f in records}) >= 7, 'wall-fragment counts do not vary')
    _require(len({f['final_cavities'] for f in records}) >= 5, 'cavity counts do not vary')
    _require(any(f['sparse'] for f in records), 'sparse networks absent')
    _require(any(f['interior_seams'] for f in records), 'divider behavior absent')
    _require(any(len(set(f['seam_lengths'])) >= 3 for f in records), 'unequal seams absent')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(40000 + 10 * nt + nq, nt, nq)
            _require((len(p['train']), len(p['test'])) == (nt, nq), 'counts ignored')
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(1, *args)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Deliberately corrupt each semantic zone without changing schema.
    pair = puzzles[0]['train'][1]
    info = _parse(pair['input'])
    _, z = _replay(pair['input'], info)
    zones = [info['initial_wall'], z['seams'], z['holes'], z['outline'],
             z['exterior'] & ~z['outline']]
    for zone in zones:
        _require(zone.any(), 'regression zone absent')
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        r, c = map(int, np.argwhere(zone)[0])
        bad['train'][1]['output'][r, c] = (int(pair['output'][r, c]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted semantic zone accepted')
    return dict(seeds=42, distinct_puzzles=len(hashes), grid_shapes=len(shapes),
                wall_structures=len(masks), palettes=len(palettes),
                terminal_counts=sorted({f['terminals'] for f in records}),
                cavity_counts=sorted({f['final_cavities'] for f in records}))
