"""Route-paint a station field, then shear every row about the bottom key.

No files, parent generators, cached examples, or random state are used by solve.
Generation uses bounded searches on at most twelve stations. Diagnostics are
in-memory; this module writes neither puzzles nor execution traces to disk.
"""
from __future__ import annotations

from hashlib import sha256
import numpy as np


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return grid.astype(np.int8, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    parts = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        todo, part = [(y, x)], []
        seen[y, x] = True
        while todo:
            r, c = todo.pop()
            part.append((r, c))
            for yy, xx in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        parts.append(part)
    return parts


def _parse(grid):
    h, w = grid.shape
    if h + w - 1 > 30:
        raise ValueError("completed shear would exceed 30 columns")
    bg = int(grid[0, 0])
    if not all(np.all(edge == bg) for edge in (grid[0], grid[:, 0], grid[:, -1])):
        raise ValueError("top and side margins must identify the background")
    xs = np.flatnonzero(grid[-1] != bg)
    if not 3 <= len(xs) <= 8 or not np.all(np.diff(xs) == 2):
        raise ValueError("bottom row must be a separated left-to-right key")
    key = tuple(int(grid[-1, x]) for x in xs)
    if len(set(key)) < 2:
        raise ValueError("the key must contain at least two labels")
    stations, geometry = [], None
    for part in _components(grid[:-1] != bg):
        ys, xx = zip(*part)
        y, x = min(ys), min(xx)
        size = max(ys) - y + 1
        if (not 3 <= size <= 6 or max(xx)-x+1 != size
                or len(part) != size*size or y+size > h-2):
            raise ValueError("stations must be separated filled squares above the key")
        block = grid[y:y+size, x:x+size]
        carrier = int(block[0, 0])
        py, px = np.nonzero(block != carrier)
        if len(py) == 0:
            raise ValueError("station has no central label")
        offset = int(py.min())
        patch = int(py.max()) - offset + 1
        if (offset < 1 or 2*offset+patch != size or int(px.min()) != offset
                or int(px.max()) != offset+patch-1 or len(py) != patch*patch):
            raise ValueError("label patch must be a centered square")
        label = int(block[offset, offset])
        if label in (bg, carrier) or not np.all(block[offset:offset+patch, offset:offset+patch] == label):
            raise ValueError("invalid station label")
        current = (size, patch, carrier)
        if geometry is not None and geometry != current:
            raise ValueError("all stations in one scene must share geometry and carrier")
        geometry = current
        stations.append((y, x, label))
    if geometry is None or not 3 <= len(stations) <= 12 or len(key) > len(stations):
        raise ValueError("invalid station count")
    size, patch, carrier = geometry
    if any(k in (bg, carrier) for k in key):
        raise ValueError("key labels must differ from structural colors")
    return dict(background=bg, carrier=carrier, size=size, patch=patch,
                stations=tuple(sorted(stations)), key=key,
                key_x=int(xs[0]), key_y=h-1)


def _band(a, b, size, patch):
    """Half-open rectangle strictly outside facing station edges."""
    inset = (size-patch)//2
    if a[0] == b[0] and abs(a[1]-b[1]) > size:
        return (a[0]+inset, a[0]+inset+patch,
                min(a[1], b[1])+size, max(a[1], b[1]))
    if a[1] == b[1] and abs(a[0]-b[0]) > size:
        return (min(a[0], b[0])+size, max(a[0], b[0]),
                a[1]+inset, a[1]+inset+patch)
    return None


def _region(rect):
    y0, y1, x0, x1 = rect
    return np.s_[y0:y1, x0:x1]


def _routes(graph, labels, key, limit=2):
    """Stop after two solutions: uniqueness is required, never tie-broken."""
    choices = [[i for i, color in enumerate(labels) if color == token] for token in key]
    answers = []
    def visit(path, used):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for i in choices[len(path)]:
            if not (used >> i) & 1 and (not path or i in graph[path[-1]]):
                visit(path+[i], used | (1 << i))
    visit([], 0)
    return answers


def _complete(grid):
    info = _parse(grid)
    stations, size, patch = info['stations'], info['size'], info['patch']
    graph = [[] for _ in stations]
    bands = {}
    for i, a in enumerate(stations):
        for j, b in enumerate(stations):
            if i == j:
                continue
            rect = _band(a, b, size, patch)
            if rect is not None and np.all(grid[_region(rect)] == info['background']):
                graph[i].append(j)
                bands[i, j] = rect
    routes = _routes(graph, [s[2] for s in stations], info['key'])
    if len(routes) != 1:
        raise ValueError("key must identify exactly one distinct-station clear route")
    route = routes[0]
    completed, painted = grid.copy(), np.zeros(grid.shape, dtype=bool)
    directions, rectangles = [], []
    for i, j in zip(route, route[1:]):
        rect = bands[i, j]
        region = _region(rect)
        if np.any(painted[region]):
            raise ValueError("route bands must not cross or overlap")
        painted[region] = True
        completed[region] = stations[i][2]
        a, b = stations[i], stations[j]
        directions.append('R' if b[1] > a[1] else 'L' if b[1] < a[1]
                          else 'D' if b[0] > a[0] else 'U')
        rectangles.append(rect)
    info.update(route=route, graph=graph, painted=painted,
                rectangles=tuple(rectangles), directions=tuple(directions))
    return completed, info


def _shear(completed, background):
    h, w = completed.shape
    output = np.full((h, w+h-1), background, dtype=np.int8)
    for r in range(h):
        start = h-1-r
        output[r, start:start+w] = completed[r]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible route, paint its gaps, then translate every full row."""
    completed, info = _complete(_grid(input_grid))
    return _shear(completed, info['background'])


def _logical_graph(cells):
    graph = [[] for _ in cells]
    for i, a in enumerate(cells):
        for j, b in enumerate(cells):
            if i == j:
                continue
            if a[0] == b[0]:
                clear = not any(c[0] == a[0] and min(a[1], b[1]) < c[1] < max(a[1], b[1]) for c in cells)
            elif a[1] == b[1]:
                clear = not any(c[1] == a[1] and min(a[0], b[0]) < c[0] < max(a[0], b[0]) for c in cells)
            else:
                continue
            if clear:
                graph[i].append(j)
    return graph


def _crosses(a, b, c, d):
    if a[0] == b[0] and c[1] == d[1]:
        return (min(a[1], b[1]) < c[1] < max(a[1], b[1])
                and min(c[0], d[0]) < a[0] < max(c[0], d[0]))
    if a[1] == b[1] and c[0] == d[0]:
        return _crosses(c, d, a, b)
    return False


def _sample_path(rng, cells, length, kind='free'):
    graph = _logical_graph(cells)
    budget = 800
    def visit(path, axes):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            return path if kind != 'both' or len(set(axes)) == 2 else None
        for nxt in rng.permutation(graph[path[-1]]):
            nxt = int(nxt)
            if nxt in path:
                continue
            a, b = cells[path[-1]], cells[nxt]
            axis = int(a[0] == b[0])
            if kind == 'straight' and axes and axis != axes[0]:
                continue
            if any(_crosses(a, b, cells[i], cells[j]) for i, j in zip(path, path[1:])):
                continue
            result = visit(path+[nxt], axes+[axis])
            if result is not None:
                return result
        return None
    for start in rng.permutation(len(cells)):
        result = visit([int(start)], [])
        if result is not None:
            return result, graph
        if budget < 0:
            break
    return None, graph


def _labels(rng, cells, path, graph, colors, anchor):
    length = len(path)
    unused = [i for i in range(len(cells)) if i not in path]
    if anchor == 0:
        a, b = [int(c) for c in rng.permutation(colors)[:2]]
        key = [a, a, b]
        labels = [0]*len(cells)
        for i, color in zip(path, key):
            labels[i] = color
        return labels, key
    if anchor == 1:
        key = [int(c) for c in rng.permutation(colors)[:length]]
        labels = [0]*len(cells)
        for i, color in zip(path, key):
            labels[i] = color
        for i in unused:
            labels[i] = key[0]
        return labels, key
    mode = str(rng.choice(('distinct', 'repeat_start', 'adjacent', 'mixed', 'mixed')))
    for _ in range(60):
        pool = [int(c) for c in rng.permutation(colors)]
        if mode == 'distinct':
            key = pool[:length]
        else:
            count = int(rng.integers(2, min(length, 6)+1))
            key = [pool[int(rng.integers(count))] for _ in range(length)]
            if mode == 'repeat_start':
                key[int(rng.integers(1, length))] = key[0]
            elif mode == 'adjacent':
                at = int(rng.integers(1, length))
                key[at] = key[at-1]
        if len(set(key)) < 2:
            continue
        labels = [0]*len(cells)
        for i, color in zip(path, key):
            labels[i] = color
        absent = [c for c in colors if c not in key]
        for i in unused:
            labels[i] = int(rng.choice(absent if absent and rng.random() < .5 else key))
        answers = _routes(graph, labels, key)
        if answers == [tuple(path)]:
            return labels, key
    return None


def _draw(rng, cells, path, labels, key, size, patch, roles, different_height=None):
    cells = [(r-min(y for y, x in cells), c-min(x for y, x in cells)) for r, c in cells]
    nr, nc = max(r for r, c in cells)+1, max(c for r, c in cells)+1
    ygaps, xgaps = [1]*(nr-1), [1]*(nc-1)
    def dimensions():
        bh, bw = nr*size+sum(ygaps), nc*size+sum(xgaps)
        return bh, bw, bh+3, max(bw+2, 2*len(key)+1)
    # Grow actual gaps independently. Width underneath a wider key is free.
    compact = bool(rng.random() < .35)
    axes = [(0, i) for i in range(nr-1)] + [(1, i) for i in range(nc-1)]
    for index in rng.permutation(len(axes)):
        axis, i = axes[int(index)]
        bh, bw, mh, mw = dimensions()
        allowance = 31-mh-mw + (max(0, 2*len(key)+1-(bw+2)) if axis else 0)
        gaps = xgaps if axis else ygaps
        feasible = []
        for extra in range((0 if compact else min(4, allowance))+1):
            gaps[i] += extra
            _, _, candidate_h, candidate_w = dimensions()
            # Keep at least one feasible height different from the first demo.
            if not (candidate_h == 31-candidate_w == different_height):
                feasible.append(extra)
            gaps[i] -= extra
        if not feasible:
            return None
        gaps[i] += int(rng.choice(feasible))
    bh, bw, mh, mw = dimensions()
    remaining = 31-mh-mw
    # Enumerate at most fifteen slack choices, retaining a height contrast for
    # the minimum two-example puzzle. Do not truncate the source rectangle.
    shapes = [(mh+dy, mw+dx) for dy in range(remaining+1)
              for dx in range(remaining-dy+1)
              if mh+dy != different_height]
    if not shapes:
        return None
    weights = np.asarray([1/(1+h-mh+w-mw)**2 for h, w in shapes])
    h, w = shapes[int(rng.choice(len(shapes), p=weights/weights.sum()))]
    if not compact and rng.random() < .20:
        boundary = [shape for shape in shapes if sum(shape) == 31]
        if boundary:
            h, w = boundary[int(rng.integers(len(boundary)))]
    ys, xs = [0], [0]
    for gap in ygaps:
        ys.append(ys[-1]+size+gap)
    for gap in xgaps:
        xs.append(xs[-1]+size+gap)
    top, left = int(rng.integers(1, h-bh-1)), int(rng.integers(1, w-bw))
    positions = [(top+ys[r], left+xs[c]) for r, c in cells]
    background, carrier = roles
    inp = np.full((h, w), background, dtype=np.int8)
    inset = (size-patch)//2
    for (y, x), label in zip(positions, labels):
        inp[y:y+size, x:x+size] = carrier
        inp[y+inset:y+inset+patch, x+inset:x+inset+patch] = label
    key_x = int(rng.integers(1, w-(2*len(key)-1)))
    inp[-1, key_x:key_x+2*len(key):2] = key
    # Build labels directly from sampled endpoints, independently of parsing,
    # graph decoding and solve's row-slice assignment.
    completed = inp.copy()
    for i, j in zip(path, path[1:]):
        a, b = positions[i], positions[j]
        if a[0] == b[0]:
            for y in range(a[0]+inset, a[0]+inset+patch):
                for x in range(min(a[1], b[1])+size, max(a[1], b[1])):
                    completed[y, x] = labels[i]
        else:
            for y in range(min(a[0], b[0])+size, max(a[0], b[0])):
                for x in range(a[1]+inset, a[1]+inset+patch):
                    completed[y, x] = labels[i]
    out = np.asarray([[background]*(h-1-r)+list(completed[r])+[background]*r
                      for r in range(h)], dtype=np.int8)
    return {'input': inp, 'output': out}


def _sample_pair(rng, roles, anchor, patch, test, different_height=None, short_test=False):
    colors = [c for c in range(10) if c not in roles]
    for _ in range(64):
        if anchor in (0, 1):
            size, length = 5, 3
            universe = [(0, 0), (0, 1), (1, 0), (1, 1)]
            if anchor == 0:
                missing = int(rng.integers(4))
                cells = [cell for i, cell in enumerate(universe) if i != missing]
            else:
                cells = universe
            path, graph = _sample_path(rng, cells, length, 'both')
            if anchor == 1 and (path is None or min(i for i in range(4) if i not in path) > path[0]):
                continue  # Make the canvas-first duplicate start explicitly wrong.
        else:
            size = int(rng.choice((3, 3, 4, 4, 5, 6)))
            lower, upper = (4, 6 if short_test else 8) if test else (3, 6)
            # Condition counts on fitting the chosen scale, rather than rejecting
            # almost every large station because a long route was sampled first.
            options = [(length, unused, r, c)
                       for length in range(lower, upper+1) for unused in range(4)
                       for r in range(1, 6) for c in range(1, 7)
                       if length+unused <= r*c <= min(12, length+unused+4)
                       and (r*size+r-1+3) + max(c*size+c-1+2, 2*length+1) <= 31]
            length, unused, nr, nc = options[int(rng.integers(len(options)))]
            n = length+unused
            universe = [(r, c) for r in range(nr) for c in range(nc)]
            cells = sorted(universe[int(i)] for i in rng.choice(len(universe), n, replace=False))
            kind = str(rng.choice(('free', 'free', 'free', 'straight')))
            path, graph = _sample_path(rng, cells, length, kind)
            patch = int(rng.choice(list(range(1 if size % 2 else 2, size-1, 2))))
        if path is None:
            continue
        assignment = _labels(rng, cells, path, graph, colors, anchor)
        if assignment is None:
            continue
        labels, key = assignment
        if _routes(graph, labels, key) != [tuple(path)]:
            continue
        pair = _draw(rng, cells, path, labels, key, size, patch, roles, different_height)
        if pair is not None:
            return pair
    # Guaranteed constructions retain every training promise and a longer test.
    # The second anchor's unused top-left duplicate cannot reach key entry two.
    if anchor == 0:
        cells, path = [(0, 0), (0, 1), (1, 0)], [1, 0, 2]
        size, labels = 5, [colors[0], colors[0], colors[1]]
        key = [colors[0], colors[0], colors[1]]
    elif anchor == 1:
        cells, path = [(0, 0), (0, 1), (1, 0), (1, 1)], [1, 3, 2]
        size = 5
        labels = [colors[0], colors[0], colors[2], colors[1]]
        key = [colors[0], colors[1], colors[2]]
    else:
        cells, path = [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 3, 2]
        size, patch = 3, 1
        labels = colors[:4]
        key = [labels[i] for i in path]
    # These compact bounding boxes have four or more spare cells. The spacing
    # allocator reserves a feasible distinct height at each growth step.
    result = _draw(rng, cells, path, labels, key, size, patch, roles, different_height)
    assert result is not None
    return result


def _signature(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    roles = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    varied_roles = bool(rng.integers(2))
    widths = tuple(int(c) for c in rng.permutation((1, 3)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            visible_roles = tuple(int(c) for c in rng.choice(10, 2, replace=False)) if varied_roles else roles
            anchor = index if split == 'train' and index < 2 else -1
            patch = widths[index] if anchor >= 0 else None
            different_height = puzzle['train'][0]['input'].shape[0] if anchor == 1 else None
            pair = _sample_pair(rng, visible_roles, anchor, patch, split == 'test',
                                different_height, short_test=nt == 2)
            # An affine palette permutation supplies forty distinct variants of
            # any scene (at least four colors). At most thirteen inputs precede
            # it; structural requirements and all equalities are preserved.
            if _signature(pair['input']) in seen:
                found = None
                for multiplier in (1, 3, 7, 9):
                    for offset in range(10):
                        remap = (multiplier*np.arange(10)+offset) % 10
                        candidate = {k: remap[v].astype(np.int8) for k, v in pair.items()}
                        if _signature(candidate['input']) not in seen:
                            found = candidate
                            break
                    if found is not None:
                        break
                if found is None:
                    raise RuntimeError('distinct palette construction failed')
                pair = found
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    completed, info = _complete(inp)
    h, w = inp.shape
    assert out.shape == (h, w+h-1), 'wrong tight shear dimensions'
    restored = np.empty_like(inp)
    for r in range(h):
        start = h-1-r
        restored[r] = out[r, start:start+w]
        assert np.all(out[r, :start] == info['background'])
        assert np.all(out[r, start+w:] == info['background'])
    assert np.array_equal(restored, completed), 'incorrect routed/sheared row contents'
    assert np.any(info['painted']) and np.all(inp[info['painted']] == info['background'])
    assert np.array_equal(restored[~info['painted']], inp[~info['painted']])
    assert np.array_equal(out[-1, :w], inp[-1]), 'bottom key must stay fixed'
    assert np.array_equal(out, solve(inp.copy()))
    # Necessity, with both counterfactuals compared on the required output size.
    assert not np.array_equal(out, _shear(inp, info['background'])), 'B has no effect'
    aligned = np.full(out.shape, info['background'], dtype=np.int8)
    aligned[:, :w] = completed
    assert not np.array_equal(out, aligned), 'A has no effect'
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    infos, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            signature = _signature(_grid(pair['input']))
            assert signature not in seen, 'duplicate input'
            seen.add(signature)
            infos[split].append(info)
    train = infos['train']
    assert any(a['size'] == b['size'] and a['patch'] != b['patch'] for a in train for b in train)
    assert len({np.asarray(pair['input']).shape[0] for pair in puzzle['train']}) >= 2
    assert any(len(set(a['key'])) < len(a['key']) for a in train)
    assert any(len(a['stations']) > len(a['route']) and
               any(s[2] in a['key'] for i, s in enumerate(a['stations']) if i not in a['route']) for a in train)
    assert {'h', 'v'} <= {('h' if d in 'LR' else 'v') for a in train for d in a['directions']}
    assert all(any(x != y for x, y in zip(a['key'], a['key'][1:])) for a in train)
    assert any(a['route'][0] != next(i for i, s in enumerate(a['stations']) if s[2] == a['key'][0]) for a in train)
    assert all(len(a['key']) >= 4 for a in infos['test'])


def _digest(puzzle):
    h = sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                grid = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def _fixture_checks():
    """Hand-constructed nonconsecutive repeat, margins and ambiguity checks."""
    inp = np.zeros((12, 13), dtype=np.int8)
    # A--B then down to A--C. The two A occurrences cannot be interchanged.
    for y, x, label in ((1, 1, 2), (1, 5, 3), (7, 5, 2), (7, 9, 4)):
        inp[y:y+3, x:x+3] = 1
        inp[y+1, x+1] = label
    inp[-1, 1:8:2] = [2, 3, 2, 4]
    completed = inp.copy()
    completed[2, 4] = 2
    completed[4:7, 6] = 3
    completed[8, 8] = 2
    # Explicit coordinate transport, separate from _shear and _draw.
    expected = np.zeros((12, 24), dtype=np.int8)
    for y in range(12):
        for x in range(13):
            expected[y, x+11-y] = completed[y, x]
    original = inp.copy()
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(inp, original), 'solve mutated its input'
    remap = np.asarray([7, 0, 8, 4, 9, 1, 2, 3, 5, 6], dtype=np.int8)
    assert np.array_equal(solve(remap[inp]), remap[expected])
    # Inserting a blank separator above the key changes the shear of every
    # station row but retains the key's zero shift and full canvas extent.
    taller = np.insert(inp, 11, 0, axis=0)
    taller_expected = np.zeros((13, 25), dtype=np.int8)
    taller_expected[:11, 1:] = expected[:11]
    taller_expected[-1, :13] = inp[-1]
    assert np.array_equal(solve(taller), taller_expected)
    wider = np.pad(inp, ((0, 0), (1, 0)))
    assert np.array_equal(solve(wider), np.pad(expected, ((0, 0), (1, 0))))
    # A-B-A without the final C is ambiguous by reversal, hence rejected.
    ambiguous = inp.copy()
    ambiguous[-1] = 0
    ambiguous[-1, 1:6:2] = [2, 3, 2]
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError('ambiguous route accepted')


def self_test():
    """Bounded, streaming 42-seed sweep plus API and corruption checks."""
    shapes, layouts, palettes, geometries, lengths, unused_counts = set(), set(), set(), set(), set(), set()
    signatures, normalized = set(), set()
    flags = dict(sparse=0, straight=0, turning=0, adjacent_equal=0,
                 repeated_start=0, absent_unused=0, duplicate_unused=0,
                 unequal_gaps=0, boundary_output=0)
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                total += 1
                inp = pair['input']
                completed, a = _complete(inp)
                stations, path = a['stations'], a['route']
                ys, xs = {s[0] for s in stations}, {s[1] for s in stations}
                shapes.add(inp.shape)
                layouts.add((a['size'], tuple((s[0]-min(ys), s[1]-min(xs)) for s in stations)))
                palettes.add((a['background'], a['carrier']))
                geometries.add((a['size'], a['patch']))
                lengths.add(len(path))
                unused = [s[2] for i, s in enumerate(stations) if i not in path]
                unused_counts.add(len(unused))
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                normalized.add((inp.shape, sha256(norm).hexdigest()))
                flags['sparse'] += len(stations) < len(ys)*len(xs)
                axes = [d in 'LR' for d in a['directions']]
                flags['straight'] += len(set(axes)) == 1
                flags['turning'] += len(set(axes)) == 2
                flags['adjacent_equal'] += any(x == y for x, y in zip(a['key'], a['key'][1:]))
                flags['repeated_start'] += a['key'][0] in a['key'][1:]
                flags['absent_unused'] += any(c not in a['key'] for c in unused)
                flags['duplicate_unused'] += any(c in a['key'] for c in unused)
                gaps = {True: [], False: []}
                for i, j in zip(path, path[1:]):
                    u, v = stations[i], stations[j]
                    horizontal = u[0] == v[0]
                    gaps[horizontal].append(abs(u[int(horizontal)]-v[int(horizontal)])-a['size'])
                flags['unequal_gaps'] += any(len(set(values)) > 1 for values in gaps.values())
                flags['boundary_output'] += pair['output'].shape[1] == 30
                assert np.array_equal(solve(inp), pair['output'])
                assert np.array_equal(inp, pair['input'])
                assert np.all(completed[a['painted']] != a['background'])
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(layouts) >= 50 and len(normalized) >= 170
    assert len(palettes) >= 20 and {3, 4, 5, 6} <= {s for s, p in geometries}
    assert {(5, 1), (5, 3), (6, 2), (6, 4)} <= geometries
    assert {3, 4, 5, 6, 7, 8} <= lengths
    assert {0, 1, 2, 3} <= unused_counts
    assert all(value > 0 for value in flags.values()), flags
    _fixture_checks()
    assert _digest(generate_puzzle(17)) == _digest(generate_puzzle(17))
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        a = generate_puzzle(700+nt+nq, nt, nq)
        assert (len(a['train']), len(a['test'])) == (nt, nq)
        assert _digest(a) == _digest(generate_puzzle(700+nt+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    puzzle = generate_puzzle(0)
    # Detect missing B, wrong A, changed padding, and altered station pixels.
    for defect in ('no_route', 'no_shear', 'padding', 'station'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzle[s]] for s in puzzle}
        pair = bad['test'][0]
        inp, out = pair['input'], pair['output']
        completed, a = _complete(inp)
        if defect == 'no_route':
            pair['output'] = _shear(inp, a['background'])
        elif defect == 'no_shear':
            out[:] = a['background']
            out[:, :inp.shape[1]] = completed
        elif defect == 'padding':
            out[0, 0] = (a['background']+1) % 10
        else:
            y, x, _ = a['stations'][0]
            out[y, x+inp.shape[0]-1-y] = (a['carrier']+1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f'accepted output defect: {defect}')
    return dict(seeds=42, pairs=total, distinct_puzzles=len(signatures),
                input_shapes=len(shapes), layouts=len(layouts),
                normalized_inputs=len(normalized), structural_palettes=len(palettes),
                station_patch_sizes=sorted(geometries), route_lengths=sorted(lengths),
                unused_counts=sorted(unused_counts), features=flags)
