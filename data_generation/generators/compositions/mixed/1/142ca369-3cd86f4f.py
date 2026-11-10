"""Paint reflected L-rays in source coordinates, then shear every complete row.

Only NumPy and the standard library are needed. No parent module, example label,
seed-dependent convention, or generation plan is consulted by solve().
"""
from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must be rectangular, two-dimensional, and at most 30 by 30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid colors must be integers from 0 to 9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _background(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    color = int(counts.argmax())
    _require(counts[color] * 2 > grid.size, "input background must be a strict majority")
    return color


def _parse(grid):
    """Decode same-color four-connected components into Ls and odd bars."""
    bg = _background(grid)
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != bg)}
    sources, mirrors = [], []
    while remaining:
        start = min(remaining)
        shade = int(grid[start])
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in NEIGHBORS:
                q = r + dr, c + dc
                if q in remaining and grid[q] == shade:
                    remaining.remove(q)
                    todo.append(q)
        cells = tuple(sorted(cells))
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
        if len(cells) == 3 and r1 - r0 == c1 - c0 == 1:
            missing = next(p for p in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
                           if p not in cells)
            elbow = r0 + r1 - missing[0], c0 + c1 - missing[1]
            direction = elbow[0] - missing[0], elbow[1] - missing[1]
            sources.append(dict(cells=cells, color=shade, elbow=elbow, direction=direction))
        else:
            _require(len(cells) % 2 == 1 and (r0 == r1 or c0 == c1)
                     and len(cells) == max(r1-r0, c1-c0)+1,
                     "each foreground component must be an L or a contiguous odd bar")
            # axis 0 reverses rows; axis 1 reverses columns; points allow both.
            axes = (0, 1) if len(cells) == 1 else ((0,) if r0 == r1 else (1,))
            mirrors.append(dict(cells=cells, color=shade,
                                midpoint=((r0+r1)//2, (c0+c1)//2), axes=axes))
    _require(bool(sources), "at least one L source is required")
    return bg, sources, mirrors


def _paint(grid):
    """Cell tracer; all route records are temporary and are never written to disk."""
    bg, sources, mirrors = _parse(grid)
    occupied = {p for obj in sources + mirrors for p in obj['cells']}
    painted, used, routes = {}, set(), []
    result = grid.copy()
    for source in sources:
        anchor, direction, shade = source['elbow'], source['direction'], source['color']
        position = anchor[0]+direction[0], anchor[1]+direction[1]
        visited, path, contacts = set(), [], []
        while _inside(position, grid.shape):
            state = position, direction
            _require(state not in visited, "cyclic ray")
            visited.add(state)
            _require(position not in occupied, "ray hits an object instead of a midpoint face")
            hits = []
            for index, mirror in enumerate(mirrors):
                for axis in mirror['axes']:
                    other = 1-axis
                    if (position[other] == mirror['midpoint'][other]
                            and mirror['midpoint'][axis]-position[axis] == direction[axis]):
                        hits.append((index, axis))
            _require(len(hits) <= 1, "ambiguous simultaneous midpoint contacts")
            if hits:
                index, axis = hits[0]
                old_shade = shade
                shade = mirrors[index]['color']
                used.add(index)
                contacts.append((index, axis, position, old_shade, shade))
                direction = ((-direction[0], direction[1]) if axis == 0
                             else (direction[0], -direction[1]))
            _require(position not in painted or painted[position] == shade,
                     "different ray colors demand the same cell")
            painted[position] = shade
            result[position] = shade
            path.append((position, shade))
            position = position[0]+direction[0], position[1]+direction[1]
        routes.append(dict(path=path, contacts=contacts, exit=position, direction=direction))
    _require(used == set(range(len(mirrors))), "every mirror must be reached")
    return result, dict(background=bg, sources=sources, mirrors=mirrors, routes=routes)


def _shear(painted, bg):
    h, w = painted.shape
    result = np.full((h, h+w-1), bg, dtype=np.int8)
    for r in range(h):
        result[r, h-1-r:h-1-r+w] = painted[r]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    _require(sum(grid.shape)-1 <= 30, "sheared canvas would exceed 30 columns")
    painted, info = _paint(grid)
    return _shear(painted, info['background'])


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r-dr, c), (r, c-dc)}


def _mirror_cells(midpoint, axis, length):
    r, c = midpoint
    return {(r, c+offset) if axis == 0 else (r+offset, c)
            for offset in range(-(length//2), length//2+1)}


def _place(grid, cells, shade, bg):
    if not all(_inside(p, grid.shape) and grid[p] == bg for p in cells):
        return None
    # Differently colored objects may touch. Same-color decomposition stays unique.
    for r, c in cells:
        for dr, dc in NEIGHBORS:
            q = r+dr, c+dc
            if q not in cells and _inside(q, grid.shape) and grid[q] == shade:
                return None
    result = grid.copy()
    for p in cells:
        result[p] = shade
    if np.count_nonzero(result == bg)*2 <= result.size:
        return None
    return result


def _add_source(grid, rng, palette, bg, fixed_direction=None, fixed_color=None):
    h, w = grid.shape
    # Bounded candidates include corners, edges and interior placements.
    candidates = []
    for _ in range(90):
        direction = fixed_direction or DIRECTIONS[int(rng.integers(4))]
        dr, dc = direction
        mode = int(rng.integers(4))
        r, c = int(rng.integers(h)), int(rng.integers(w))
        if mode in (0, 1):
            r = 1 if dr > 0 else h-2
        if mode in (0, 2):
            c = 1 if dc > 0 else w-2
        candidates.append(((r, c), direction))
    for elbow, direction in candidates:
        if not _inside((elbow[0]+2*direction[0], elbow[1]+2*direction[1]), grid.shape):
            continue
        shade = int(rng.choice(palette)) if fixed_color is None else fixed_color
        trial = _place(grid, _source_cells(elbow, direction), shade, bg)
        if trial is None:
            continue
        try:
            _, info = _paint(trial)
        except ValueError:
            continue
        if any(s['elbow'] == elbow and len(route['path']) >= 2
               for s, route in zip(info['sources'], info['routes'])):
            return trial, elbow
    return None


def _append_mirror(grid, elbow, rng, palette, axis=None, length=None, shade=None):
    _, before = _paint(grid)
    source, route = next((s, route) for s, route in zip(before['sources'], before['routes'])
                         if s['elbow'] == elbow)
    direction = route['direction']
    anchor = route['contacts'][-1][2] if route['contacts'] else elbow
    tail, q = [], (anchor[0]+direction[0], anchor[1]+direction[1])
    while _inside(q, grid.shape):
        tail.append(q)
        q = q[0]+direction[0], q[1]+direction[1]
    options = [(p, a, n) for p in tail
               for a in ((axis,) if axis is not None else (0, 1))
               for n in ((length,) if length is not None else (1, 3, 5, 7, 9))]
    rng.shuffle(options)
    prior = [(before['mirrors'][i]['midpoint'], a, p) for i, a, p, _, _ in route['contacts']]
    incoming = route['contacts'][-1][4] if route['contacts'] else source['color']
    for contact, a, n in options[:90]:
        midpoint = list(contact)
        midpoint[a] += direction[a]
        midpoint = tuple(midpoint)
        outgoing = (-direction[0], direction[1]) if a == 0 else (direction[0], -direction[1])
        if not _inside((contact[0]+2*outgoing[0], contact[1]+2*outgoing[1]), grid.shape):
            continue
        color = (incoming if rng.random() < .25 else int(rng.choice(palette))) if shade is None else shade
        trial = _place(grid, _mirror_cells(midpoint, a, n), color, before['background'])
        if trial is None:
            continue
        try:
            _, after = _paint(trial)
        except ValueError:
            continue
        new_route = next(r for s, r in zip(after['sources'], after['routes']) if s['elbow'] == elbow)
        actual = [(after['mirrors'][i]['midpoint'], aa, p) for i, aa, p, _, _ in new_route['contacts']]
        if actual == prior + [(midpoint, a, contact)]:
            return trial
    return None


def _shape(rng, *, teaching=False, test=False):
    if test:
        h = int(rng.integers(4, 28))
        return h, 31-h  # Every test reaches the width-30 boundary.
    if teaching:
        h = int(rng.integers(10, 18))
        return h, int(rng.integers(9, min(18, 29-h)+1))
    h = int(rng.integers(4, 26))
    return h, int(rng.integers(4, min(25, 29-h)+1))


def _colors(rng, anchor=None):
    backgrounds = [c for c in range(10) if c != anchor]
    bg = int(rng.choice(backgrounds))
    colors = [c for c in range(10) if c != bg and c != anchor]
    rng.shuffle(colors)
    count = int(rng.choice((1, 2, 3, 4, 5, 6, 7), p=(.15, .25, .23, .16, .10, .07, .04)))
    palette = colors[:count] if anchor is None else [anchor]+colors[:max(1, count-1)]
    return bg, palette


def _teaching_fallback(rng, axis, direction, anchor, same):
    """Analytic, variable-size two-bounce zigzag; no rejection loop."""
    bg, palette = _colors(rng, anchor)
    other = palette[1]
    distance = int(rng.integers(3, 6))
    h, w = 2*distance+3, distance+5
    grid = np.full((h, w), bg, dtype=np.int8)
    for p in _source_cells((1, 1), (1, 1)):
        grid[p] = anchor
    for p in _mirror_cells((1+distance, 2+distance), 1, 3):
        grid[p] = anchor if same else other
    grid[2*distance, 1] = other if same else anchor
    if axis == 0:
        grid = grid.T.copy()
    if direction[0] < 0:
        grid = grid[::-1].copy()
    if direction[1] < 0:
        grid = grid[:, ::-1].copy()
    return grid


def _teaching_scene(rng, axis, direction, anchor, same, forbidden_height=None):
    for _ in range(24):
        shape = _shape(rng, teaching=True)
        if shape[0] == forbidden_height:
            continue
        bg, palette = _colors(rng, anchor)
        grid = np.full(shape, bg, dtype=np.int8)
        first = _add_source(grid, rng, palette, bg, direction, anchor)
        if first is None:
            continue
        grid, elbow = first
        grid = _append_mirror(grid, elbow, rng, palette, axis,
                              int(rng.choice((3, 5, 7))), anchor if same else palette[1])
        if grid is None:
            continue
        # A second contact teaches chaining and a point's contacted face.
        grid = _append_mirror(grid, elbow, rng, palette, length=1,
                              shade=palette[1] if same else anchor)
        if grid is None:
            continue
        if rng.random() < .5:
            extra = _add_source(grid, rng, palette, bg)
            if extra is not None:
                grid = extra[0]
        return grid
    for _ in range(6):
        grid = _teaching_fallback(rng, axis, direction, anchor, same)
        if grid.shape[0] != forbidden_height:
            return grid
    # At least two of the three distances give distinct heights after transpose.
    grid = _teaching_fallback(np.random.default_rng(0), axis, direction, anchor, same)
    if grid.shape[0] == forbidden_height:
        grid = np.pad(grid, ((0, 1), (0, 0)), constant_values=_background(grid))
    return grid


def _ordinary_scene(rng, test):
    shape = _shape(rng, test=test)
    bg, palette = _colors(rng)
    grid = np.full(shape, bg, dtype=np.int8)
    target = int(rng.choice((1, 2, 3, 4, 6, 8), p=(.24, .25, .22, .15, .09, .05)))
    for index in range(target):
        item = _add_source(grid, rng, palette, bg)
        if item is None:
            break
        grid, elbow = item
        bounces = int(rng.choice((0, 1, 2, 3, 4), p=(.28, .29, .24, .14, .05)))
        for _ in range(bounces):
            extended = _append_mirror(grid, elbow, rng, palette)
            if extended is None:
                break
            grid = extended
    if np.all(grid == bg):
        # Always feasible for dimensions >=4, with two visible ray cells.
        for p in _source_cells((1, 1), (1, 1)):
            grid[p] = palette[0]
    return grid


def _counts(num_train, num_test):
    _require(all(isinstance(x, Integral) and not isinstance(x, (bool, np.bool_))
                 for x in (num_train, num_test)), "counts must be integers")
    _require(2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test,
             "need 2..10 train, 1..4 test, and num_train > num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    _require(isinstance(seed, Integral) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    anchor = int(rng.integers(10))
    axis = int(rng.integers(2))
    order = rng.permutation(4)
    first = _teaching_scene(rng, axis, DIRECTIONS[int(order[0])], anchor, True)
    second = _teaching_scene(rng, 1-axis, DIRECTIONS[int(order[1])], anchor, False,
                             forbidden_height=first.shape[0])
    puzzle = {'train': [], 'test': []}
    seen = set()
    for grid in (first, second):
        seen.add(_key(grid))
        puzzle['train'].append({'input': grid, 'output': solve(grid)})
    for split, count in (('train', ntrain), ('test', ntest)):
        while len(puzzle[split]) < count:
            for _ in range(24):
                grid = _ordinary_scene(rng, test=split == 'test')
                if _key(grid) not in seen:
                    break
            else:
                raise RuntimeError("bounded scene sampler could not find a distinct input")
            seen.add(_key(grid))
            puzzle[split].append({'input': grid, 'output': solve(grid)})
    # Do not make demonstration order part of the semantics.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _coverage(infos):
    bar_axes, directions, transitions = set(), set(), []
    point = False
    for info in infos:
        directions.update(s['direction'] for s in info['sources'])
        for route in info['routes']:
            for index, axis, _, a, b in route['contacts']:
                transitions.append((a, b))
                if len(info['mirrors'][index]['cells']) == 1:
                    point = True
                else:
                    bar_axes.add(axis)
    successors = {a: {b for aa, b in transitions if aa == a} for a, _ in transitions}
    return (bar_axes == {0, 1} and len(directions) >= 2 and point
            and any(a == b for a, b in transitions) and any(a != b for a, b in transitions)
            and any(len(values) > 1 for values in successors.values()))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training_info, heights = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            h, w = inp.shape
            _require(h+w-1 <= 30 and out.shape == (h, h+w-1), "incorrect canvas dimensions")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            painted, info = _paint(inp)
            bg = info['background']
            _require(np.any(painted != inp), "every pair must expose ray completion")
            # Inspect each full row vector and both padding regions, rather than
            # just calling solve() again. This also checks blank rows/columns.
            for r in range(h):
                start = h-1-r
                _require(np.array_equal(out[r, start:start+w], painted[r]), "painted row not preserved")
                _require(np.all(out[r, :start] == bg) and np.all(out[r, start+w:] == bg),
                         "incorrect background padding")
                for c in np.flatnonzero(inp[r] != bg):
                    _require(out[r, start+c] == inp[r, c], "source or mirror changed")
            _require(set(np.unique(out)) <= set(np.unique(inp)), "invented color")
            # Necessity witnesses: copying/shearing input alone misses new rays;
            # leaving painted rows unshifted misplaces at least one foreground cell.
            _require(not np.array_equal(out, _shear(inp, bg)), "parent A has no effect")
            unshifted = np.full(out.shape, bg, dtype=np.int8)
            unshifted[:, :w] = painted
            _require(not np.array_equal(out, unshifted), "parent B has no visible effect")
            if split == 'train':
                training_info.append(info)
                heights.add(h)
    _require(len(heights) >= 2 and _coverage(training_info), "insufficient teaching evidence")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _normalized_structure(grid):
    roles = {_background(grid): 0}
    values = [roles.setdefault(int(v), len(roles)) for v in grid.flat]
    return grid.shape, bytes(values)


def _regressions():
    # Literal route expectations: no generator, seed or stored training examples.
    grid = np.zeros((9, 10), dtype=np.int8)
    grid[0, 1] = grid[1, 0] = grid[1, 1] = 2
    grid[3:6, 5] = 4
    grid[6, 1] = 7
    painted = grid.copy()
    for p in ((2, 2), (3, 3)):
        painted[p] = 2
    for p in ((4, 4), (5, 3)):
        painted[p] = 4
    for p in ((6, 2), (7, 3), (8, 4)):
        painted[p] = 7
    expected = np.zeros((9, 18), dtype=np.int8)
    for r in range(9):
        for c in range(10):
            expected[r, c+8-r] = painted[r, c]
    _require(np.array_equal(solve(grid), expected), "literal two-bounce regression")
    # Both reflection axes, all launch rotations, and arbitrary padding colors.
    for transposed in (False, True):
        for flip_r in (False, True):
            for flip_c in (False, True):
                a, b = grid.copy(), painted.copy()
                if transposed:
                    a, b = a.T, b.T
                if flip_r:
                    a, b = a[::-1], b[::-1]
                if flip_c:
                    a, b = a[:, ::-1], b[:, ::-1]
                for bg in (0, 9):
                    aa, bb = a.copy(), b.copy()
                    aa[a == 0], bb[b == 0] = bg, bg
                    _require(np.array_equal(solve(aa), _shear(bb, bg)), "symmetry/color regression")
    # Color-aware touching sources; equal-color shared ray cells are permitted.
    adjacent = np.zeros((8, 10), dtype=np.int8)
    for offset, color in ((0, 1), (2, 3)):
        for p in _source_cells((1, 1+offset), (1, 1)):
            adjacent[p] = color
    complete = adjacent.copy()
    for r in range(2, 8):
        complete[r, r] = 1
        if r+2 < 10:
            complete[r, r+2] = 3
    _require(np.array_equal(solve(adjacent), _shear(complete, 0)), "touching L regression")
    for size, second in ((8, 2), (9, 1), (9, 2)):
        a = np.zeros((size, size), dtype=np.int8)
        for elbow, direction, color in (((1, 1), (1, 1), 1),
                                         ((1, size-2), (1, -1), second)):
            for p in _source_cells(elbow, direction):
                a[p] = color
        if size == 9 and second == 2:
            try:
                solve(a)
            except ValueError:
                pass
            else:
                raise AssertionError("conflicting shared cell accepted")
        else:
            solve(a)
    # Invalid shape decomposition and competing contact faces have no tie-breaker.
    bad = np.zeros((8, 8), dtype=np.int8)
    for p in _source_cells((1, 1), (1, 1)):
        bad[p] = 1
    bad[4, 3], bad[3, 4] = 2, 3
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous contact accepted")


def self_test():
    """42 seeds, streamed one puzzle at a time; no files or retained grid corpus."""
    structures, digests, shapes, backgrounds, directions = set(), set(), set(), set(), set()
    source_counts, bounce_counts, lengths, axes = set(), set(), set(), set()
    max_width, repeated_axes, changed, unchanged, palettes = 0, 0, 0, 0, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _puzzle_digest(puzzle)
        digests.add(digest)
        if seed in (0, 17):
            _require(digest == _puzzle_digest(generate_puzzle(seed)), "nondeterministic seed")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid = pair['input']
                painted, info = _paint(grid)
                structures.add(hashlib.sha256(repr(_normalized_structure(grid)).encode()).hexdigest())
                shapes.add(grid.shape)
                backgrounds.add(info['background'])
                palettes.add(tuple(map(int, np.unique(grid))))
                directions.update(s['direction'] for s in info['sources'])
                source_counts.add(len(info['sources']))
                lengths.update(len(m['cells']) for m in info['mirrors'])
                max_width = max(max_width, pair['output'].shape[1])
                for route in info['routes']:
                    bounce_counts.add(len(route['contacts']))
                    local_axes = [a for _, a, _, _, _ in route['contacts']]
                    axes.update(local_axes)
                    repeated_axes += sum(a == b for a, b in zip(local_axes, local_axes[1:]))
                    for _, _, _, old, new in route['contacts']:
                        changed += old != new
                        unchanged += old == new
                _require(np.array_equal(solve(grid.copy()), pair['output']), "solver disagreement")
                if split == 'test':
                    _require(pair['output'].shape[1] == 30, "missing boundary shift")
                else:
                    _require(pair['output'].shape[1] <= 28, "train/test size shift lost")
        # A mutation must be rejected by the family validator.
        if seed == 0:
            bad = {'train': puzzle['train'], 'test': [dict(puzzle['test'][0])]}
            bad['test'][0]['output'] = bad['test'][0]['output'].copy()
            bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
            try:
                validate_generated_puzzle(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("modified padding accepted")
    _require(len(digests) == 42 and len(structures) >= 180, "insufficient structural variation")
    _require(len(shapes) >= 65 and len(backgrounds) == 10 and len(palettes) >= 30,
             "insufficient shape/palette variation")
    _require(directions == set(DIRECTIONS) and axes == {0, 1}, "missing directions or axes")
    _require({1, 2, 3} <= source_counts and {0, 1, 2, 3} <= bounce_counts,
             "missing source-count or route-count variation")
    _require({1, 3, 5, 7} <= lengths and repeated_axes > 0 and changed > 0 and unchanged > 0,
             "missing mirror/transition variation")
    _regressions()
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            generate_puzzle(7300+train*10+test, train, test)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=210, structures=len(structures), shapes=len(shapes),
                backgrounds=sorted(backgrounds), source_counts=sorted(source_counts),
                bounce_counts=sorted(bounce_counts), mirror_lengths=sorted(lengths),
                output_width_max=max_width)
