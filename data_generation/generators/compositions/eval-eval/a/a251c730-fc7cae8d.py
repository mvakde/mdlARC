"""Visible keyed stencils, local marker alignment, and simultaneous welds.

Only standard-library modules and NumPy are used. No parent code or saved
examples are imported. The solver decodes its input without generation state.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

_NEIGHBORS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_MATRICES = (
    ((1, 0), (0, 1)),       # identity
    ((0, -1), (1, 0)),      # counterclockwise
    ((-1, 0), (0, -1)),     # half turn
    ((0, 1), (-1, 0)),      # clockwise
    ((1, 0), (0, -1)),      # left/right reflection
    ((-1, 0), (0, 1)),      # up/down reflection
    ((0, 1), (1, 0)),       # transpose
    ((0, -1), (-1, 0)),     # anti-transpose
)


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _move(matrix, position):
    r, c = position
    return (matrix[0][0] * r + matrix[0][1] * c,
            matrix[1][0] * r + matrix[1][1] * c)


def _matrix(source, destination):
    """Two identity-matched unit normals define one signed permutation."""
    result = np.zeros((2, 2), dtype=int)
    for color, direction in source.items():
        result += np.outer(destination[color], direction)
    key = tuple(tuple(int(v) for v in row) for row in result)
    assert key in _MATRICES, 'arms must determine a square-grid symmetry'
    return key


def _grid(value):
    array = np.asarray(value)
    assert array.ndim == 2 and all(1 <= n <= 30 for n in array.shape)
    assert np.issubdtype(array.dtype, np.integer)
    assert np.all((array >= 0) & (array <= 9))
    return array.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _NEIGHBORS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _panels(grid):
    """Clean, isolated monochromatic borders with a uniform inner fill rim."""
    panels = []
    for color in np.unique(grid):
        for cells in _components(grid == color):
            rows, cols = zip(*cells)
            r, c, b, e = min(rows), min(cols), max(rows), max(cols)
            h, w = b-r+1, e-c+1
            if min(h, w) < 5 or len(cells) != 2*h+2*w-4:
                continue
            if any(y not in (r, b) and x not in (c, e) for y, x in cells):
                continue
            inner = grid[r+1:b, c+1:e]
            fill = int(inner[0, 0])
            if fill == int(color):
                continue
            if not (np.all(inner[0] == fill) and np.all(inner[-1] == fill)
                    and np.all(inner[:, 0] == fill) and np.all(inner[:, -1] == fill)):
                continue
            if np.count_nonzero(inner == fill) <= inner.size//2:
                continue
            comps = _components(inner != fill)
            if comps:
                panels.append(dict(box=(r, c, h, w), frame=int(color),
                                   fill=fill, inner=inner, components=comps))
    assert len(panels) == 2, 'exactly two unambiguous framed panels required'
    return panels


def _hubs(target):
    inner = target['inner']
    result = []
    for cells in target['components']:
        assert len(cells) == 3, 'target contains only three-cell elbows'
        centers = [p for p in cells if sum((p[0]+dr, p[1]+dc) in cells
                                           for dr, dc in _NEIGHBORS) == 2]
        assert len(centers) == 1
        r, c = centers[0]
        directions = {}
        for y, x in cells:
            if (y, x) != (r, c):
                color = int(inner[y, x])
                assert color not in directions
                directions[color] = (y-r, x-c)
        values = list(directions.values())
        assert sum(a*b for a, b in zip(*values)) == 0, 'arms must be perpendicular'
        key = int(inner[r, c])
        assert key not in directions
        result.append(dict(position=(r, c), key=key, arms=directions,
                           cells=set(cells)))
    assert len(result) >= 2
    markers = set(result[0]['arms'])
    assert all(set(hub['arms']) == markers for hub in result)
    keys = {hub['key'] for hub in result}
    assert not keys & markers and 1 <= len(keys) <= 3
    return result, keys, markers


def _decode(input_grid):
    grid = _grid(input_grid)
    panels = _panels(grid)
    targets = [p for p in panels if all(len(comp) == 3 for comp in p['components'])]
    assert len(targets) == 1, 'sparse elbows identify the target'
    target = targets[0]
    reference = next(p for p in panels if p is not target)
    assert reference['frame'] == target['frame']
    assert reference['fill'] != target['fill']
    a, b = reference['box'], target['box']
    assert a[0]+a[2] <= b[0] or b[0]+b[2] <= a[0] or a[1]+a[3] <= b[1] or b[1]+b[3] <= a[1]
    outside = np.ones(grid.shape, dtype=bool)
    for panel in panels:
        r, c, h, w = panel['box']
        assert r >= 1 and c >= 1 and r+h < grid.shape[0] and c+w < grid.shape[1]
        outside[r:r+h, c:c+w] = False
    exterior = np.unique(grid[outside])
    assert len(exterior) == 1 and int(exterior[0]) != target['frame']
    hubs, keys, markers = _hubs(target)
    dictionaries, repetitions = {}, Counter()
    ri = reference['inner']
    for cells in reference['components']:
        assert len(cells) >= 6
        pivots = [p for p in cells if int(ri[p]) in keys]
        assert len(pivots) == 1, 'one requested key in each reference occurrence'
        r, c = pivots[0]
        key = int(ri[r, c])
        arms = {}
        decoration = []
        for y, x in cells:
            color, offset = int(ri[y, x]), (y-r, x-c)
            if color in markers:
                assert color not in arms and offset in _NEIGHBORS
                arms[color] = offset
            elif offset != (0, 0):
                assert color not in keys
                assert color not in (reference['fill'], target['fill'], target['frame'], int(exterior[0]))
                assert offset not in _NEIGHBORS, 'unoccupied immediate directions stay blank'
                decoration.append((offset[0], offset[1], color))
        assert set(arms) == markers
        directions = list(arms.values())
        assert sum(a*b for a, b in zip(*directions)) == 0
        assert len(decoration) >= 3
        stencil = (tuple(sorted(arms.items())), tuple(sorted(decoration)))
        assert key not in dictionaries or dictionaries[key] == stencil, 'inconsistent reference repetition'
        dictionaries[key] = stencil
        repetitions[key] += 1
    assert set(dictionaries) == keys, 'all and only requested keys have reference entries'
    forbidden = {reference['fill'], target['fill'], target['frame'], int(exterior[0])}
    assert not (keys | markers) & forbidden
    for hub in hubs:
        arms, decoration = dictionaries[hub['key']]
        hub['matrix'] = _matrix(dict(arms), hub['arms'])
        hub['decoration'] = decoration
    info = dict(grid=grid, reference=reference, target=target, hubs=hubs,
                dictionaries=dictionaries, repetitions=repetitions,
                background=int(exterior[0]), markers=markers, keys=keys)
    # This check does not depend on an output or sampled construction metadata.
    _render(info)
    return info


def _render(info, *, ignore_orientation=False, generic_key=None, weld=True):
    target = info['target']
    r, c, h, w = target['box']
    output = info['grid'][r:r+h, c:c+w].copy()
    fill, frame = target['fill'], target['frame']
    anchor_positions = {hub['position'] for hub in info['hubs']}
    count = np.zeros(target['inner'].shape, dtype=int)
    ink = np.full(count.shape, fill, dtype=np.int8)
    by_key = {}
    for hub in info['hubs']:
        ar, ac = hub['position']
        for direction in hub['arms'].values():
            output[ar+direction[0]+1, ac+direction[1]+1] = fill
        decoration = hub['decoration']
        source_arms = None
        if generic_key is not None:
            source_arms, decoration = info['dictionaries'][generic_key]
        matrix = _MATRICES[0] if ignore_orientation else (
            _matrix(dict(source_arms), hub['arms']) if source_arms is not None else hub['matrix'])
        cells = set()
        for dr, dc, color in decoration:
            dy, dx = _move(matrix, (dr, dc))
            y, x = ar+dy, ac+dx
            assert 0 <= y < count.shape[0] and 0 <= x < count.shape[1], 'stamps never clip'
            assert (y, x) not in anchor_positions, 'decoration cannot cover any key'
            assert (y, x) not in cells
            cells.add((y, x))
            count[y, x] += 1
            ink[y, x] = color
        by_key.setdefault(hub['key'], set()).update(cells)
    inner = output[1:-1, 1:-1]
    inner[count == 1] = ink[count == 1]
    if weld:
        inner[count >= 2] = frame
    else:
        inner[count >= 2] = ink[count >= 2]
    return output, count, by_key


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete the target using only its visible dictionary and local arms."""
    return _render(_decode(input_grid))[0]


def _same_color_weld(info):
    colors = {}
    for hub in info['hubs']:
        ar, ac = hub['position']
        for dr, dc, color in hub['decoration']:
            dy, dx = _move(hub['matrix'], (dr, dc))
            colors.setdefault((ar+dy, ac+dx), []).append(color)
    return any(len(values) >= 2 and len(set(values)) == 1 for values in colors.values())


def _shape(rng, radius, markers, key, decoration_colors):
    index = int(rng.integers(4))
    arms = {markers[0]: _NEIGHBORS[index], markers[1]: _NEIGHBORS[(index+1) % 4]}
    if rng.random() < .5:
        arms = {markers[1]: _NEIGHBORS[index], markers[0]: _NEIGHBORS[(index+1) % 4]}
    cells = {(0, 0), *arms.values()}
    wanted = int(rng.integers(4, 9 if radius == 2 else 14))
    style = int(rng.integers(3))
    for _ in range(wanted):
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _NEIGHBORS
                           if abs(r+dr) <= radius and abs(c+dc) <= radius
                           and ((r+dr, c+dc) not in _NEIGHBORS or (r+dr, c+dc) in arms.values())} - cells)
        weights = []
        for r, c in frontier:
            neighbors = sum((r+dr, c+dc) in cells for dr, dc in _NEIGHBORS)
            weights.append(1.0 if style == 0 else (1.0/neighbors if style == 1 else neighbors**2))
        probabilities = np.asarray(weights, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=probabilities/probabilities.sum()))])
    decorations = sorted(cells - {(0, 0)} - set(arms.values()))
    assignments = [int(_pick(rng, decoration_colors)) for _ in decorations]
    for i, color in enumerate(decoration_colors):
        assignments[i] = int(color)
    full = [(0, 0, key)] + [(dr, dc, color) for color, (dr, dc) in arms.items()]
    full += [(dr, dc, color) for (dr, dc), color in zip(decorations, assignments)]
    return dict(key=key, arms=arms, decoration=tuple(full[3:]), full=tuple(full))


def _palette(rng, nkeys, stable=None):
    colors = list(stable) if stable is not None else [int(x) for x in rng.permutation(10)]
    reuse = bool(rng.random() < .35)
    bg = colors[0]
    ref_fill = bg if reuse else colors[1]
    target_fill, frame = colors[2:4]
    markers = colors[4:6]
    keys = colors[6:6+nkeys]
    dec = colors[6+nkeys:]
    if reuse:
        dec = dec + [colors[1]]
    dec = dec[:int(rng.integers(1, min(2, len(dec))+1))]
    return dict(background=bg, reference_fill=ref_fill, target_fill=target_fill,
                frame=frame, markers=markers, keys=keys, decoration=dec)


def _reference_options(stencils, copies):
    instances = [stencil for stencil, n in zip(stencils, copies) for _ in range(n)]
    result = []
    for width in range(7, 29):
        y, x, shelf_height = 2, 2, 0
        placements = []
        for stencil in instances:
            rows, cols = zip(*[(r, c) for r, c, _ in stencil['full']])
            lo_r, lo_c = min(rows), min(cols)
            h, w = max(rows)-lo_r+1, max(cols)-lo_c+1
            if w+4 > width:
                break
            if x+w > width-2:
                y += shelf_height+1
                x, shelf_height = 2, 0
            placements.append((stencil, y-lo_r, x-lo_c))
            x += w+1
            shelf_height = max(shelf_height, h)
        else:
            height = y+shelf_height+2
            ink_count = sum(len(stencil['full']) for stencil in instances)
            if height <= 28 and 2*ink_count < (height-2)*(width-2):
                result.append(((height, width), placements))
    return result


def _geometry(rng, stencils, copies, anchor_count, layout):
    options = _reference_options(stencils, copies)
    axis = 0 if layout == 'stacked' else 1
    feasible = []
    for ref_shape, places in options:
        for h in range(9, 25):
            for w in range(9, 27):
                shape = (h, w)
                if ref_shape[axis]+shape[axis]+3 > 30:
                    continue
                if h*w < 20*anchor_count or min(h, w) < 10:
                    continue
                feasible.append((ref_shape, places, shape))
    if not feasible:
        return None
    ref_shape, places, target_shape = _pick(rng, feasible)
    shapes = [ref_shape, target_shape]
    gap = int(rng.integers(1, min(3, 28-ref_shape[axis]-target_shape[axis])+1))
    minimum = [max(ref_shape[i], target_shape[i])+2 for i in range(2)]
    minimum[axis] = ref_shape[axis]+target_shape[axis]+gap+2
    canvas = [int(rng.integers(n, 31)) for n in minimum]
    if rng.random() < .1:
        canvas = [30, 30]
    positions = [[0, 0], [0, 0]]
    order = [0, 1] if rng.random() < .5 else [1, 0]
    start = int(rng.integers(1, canvas[axis]-sum(s[axis] for s in shapes)-gap))
    positions[order[0]][axis] = start
    positions[order[1]][axis] = start+shapes[order[0]][axis]+gap
    other = 1-axis
    for i in (0, 1):
        positions[i][other] = int(rng.integers(1, canvas[other]-shapes[i][other]))
    boxes = [tuple(positions[i])+shapes[i] for i in (0, 1)]
    return tuple(canvas), boxes, places


def _placement(rng, stencils, ids, shape, triple):
    """Bounded placement; overlap is geometry, not a saved answer pattern."""
    H, W = shape[0]-2, shape[1]-2
    # First two (or three) belong to the same key, the remaining keys are
    # requested too. At least one remote hub exposes non-welded decoration.
    for restart in range(45):
        hubs, used_hubs, anchors = [], set(), set()
        cover = Counter()
        failed = False
        for index, type_id in enumerate(ids):
            stencil = stencils[type_id]
            chosen = None
            wants_collision = index == 1 or (triple and index == 2)
            for attempt in range(360):
                if index == 0:
                    matrix = _pick(rng, _MATRICES[1:2]+_MATRICES[3:4])
                elif index == 1:
                    matrix = _pick(rng, _MATRICES[4:])
                else:
                    matrix = _pick(rng, _MATRICES)
                decoration = [(*_move(matrix, (r, c)), color) for r, c, color in stencil['decoration']]
                if wants_collision:
                    points = [p for p, n in cover.items() if n == (2 if index == 2 else 1)]
                    if not points:
                        break
                    p = _pick(rng, sorted(points))
                    dy, dx, _ = _pick(rng, decoration)
                    ar, ac = p[0]-dy, p[1]-dx
                else:
                    ar = int(rng.integers(1, H-1))
                    ac = int(rng.integers(1, W-1))
                arms = {color: _move(matrix, direction) for color, direction in stencil['arms'].items()}
                hub_cells = {(ar, ac), *((ar+dr, ac+dc) for dr, dc in arms.values())}
                if any(not (1 <= y < H-1 and 1 <= x < W-1) for y, x in hub_cells):
                    continue
                footprint = {(ar+dr, ac+dc) for dr, dc, _ in decoration}
                if any(not (0 <= y < H and 0 <= x < W) for y, x in footprint):
                    continue
                if footprint & anchors or (ar, ac) in cover:
                    continue
                if any((y+dr, x+dc) in used_hubs for y, x in hub_cells
                       for dr, dc in ((0, 0),)+_NEIGHBORS):
                    continue
                if not wants_collision and index >= (3 if triple else 2):
                    # A remote instance defeats 'intersect all stencils'.
                    if index == (3 if triple else 2) and footprint & set(cover):
                        continue
                if not triple and any(cover[p] >= 2 for p in footprint):
                    continue
                new_cover = cover.copy()
                new_cover.update(footprint)
                if not any(n == 1 for n in new_cover.values()):
                    continue
                chosen = dict(position=(ar, ac), key=stencil['key'], arms=arms,
                              matrix=matrix, decoration=stencil['decoration'], cells=hub_cells)
                cover = new_cover
                break
            if chosen is None:
                failed = True
                break
            hubs.append(chosen)
            used_hubs.update(chosen['cells'])
            anchors.add(chosen['position'])
        if not failed and max(cover.values()) >= (3 if triple else 2):
            return hubs
    return None


def _draw_panel(grid, box, frame, fill):
    r, c, h, w = box
    grid[r:r+h, c:c+w] = frame
    grid[r+1:r+h-1, c+1:c+w-1] = fill


def _make_pair(rng, nkeys, total, triple, layout, stable, repeated=False):
    for attempt in range(90):
        radius = 2 if repeated or attempt > 35 else int(rng.integers(2, 4))
        palette = _palette(rng, nkeys, stable)
        stencils = [_shape(rng, radius, palette['markers'], key, palette['decoration'])
                    for key in palette['keys']]
        silhouettes = [tuple(sorted((r, c) for r, c, _ in s['decoration'])) for s in stencils]
        if len(set(silhouettes)) != nkeys:
            continue
        copies = [2 if repeated else int(rng.integers(1, 3)) for _ in stencils]
        if attempt > 35 and not repeated:
            copies = [1]*nkeys
        geometry = _geometry(rng, stencils, copies, total, layout)
        if geometry is None:
            continue
        canvas, boxes, ref_places = geometry
        core_count = 3 if triple else 2
        ids = [0]*core_count + list(range(1, nkeys))
        ids += [int(rng.integers(nkeys)) for _ in range(total-len(ids))]
        hubs = _placement(rng, stencils, ids, boxes[1][2:], triple)
        if hubs is None:
            continue
        grid = np.full(canvas, palette['background'], dtype=np.int8)
        _draw_panel(grid, boxes[0], palette['frame'], palette['reference_fill'])
        _draw_panel(grid, boxes[1], palette['frame'], palette['target_fill'])
        r, c, _, _ = boxes[0]
        for stencil, ar, ac in ref_places:
            for dy, dx, color in stencil['full']:
                grid[r+ar+dy, c+ac+dx] = color
        r, c, _, _ = boxes[1]
        for hub in hubs:
            ar, ac = hub['position']
            grid[r+ar+1, c+ac+1] = hub['key']
            for color, (dy, dx) in hub['arms'].items():
                grid[r+ar+dy+1, c+ac+dx+1] = color
        try:
            info = _decode(grid)
            output, count, by_key = _render(info)
            assert np.any(count == 1) and np.max(count) >= (3 if triple else 2)
            # Every requested template has a visible single-coverage cell.
            assert all(any(count[p] == 1 for p in cells) for cells in by_key.values())
            try:
                wrong = _render(info, ignore_orientation=True)[0]
            except AssertionError:
                if repeated:
                    continue
                wrong = None
            assert wrong is None or not np.array_equal(wrong, output)
            if nkeys > 1:
                try:
                    generic = _render(info, generic_key=min(info['keys']))[0]
                except AssertionError:
                    if repeated:
                        continue
                    generic = None
                assert generic is None or not np.array_equal(generic, output)
            if repeated:
                assert _same_color_weld(info), 'training must distinguish overlap from color conflict'
            return {'input': grid, 'output': output}
        except AssertionError:
            continue
    raise RuntimeError('bounded construction could not fit a readable stencil assembly')


def _swap_contrast(pair):
    """Keep every occupied input cell and its role; swap one hub's arm colors."""
    info = _decode(pair['input'])
    for hub in info['hubs']:
        changed = pair['input'].copy()
        r, c, _, _ = info['target']['box']
        ar, ac = hub['position']
        directions = list(hub['arms'].values())
        colors = list(hub['arms'])
        for direction, color in zip(directions, colors[::-1]):
            changed[r+ar+direction[0]+1, c+ac+direction[1]+1] = color
        try:
            changed_info = _decode(changed)
            output, count, by_key = _render(changed_info)
            _, original_count, _ = _render(info)
            if (np.any(count == 2) and np.any(count == 3) and np.any(count == 1) and np.max(count) == 3
                    and all(any(count[p] == 1 for p in cells) for cells in by_key.values())
                    and not np.array_equal(output, pair['output'])
                    and not np.array_equal(count >= 2, original_count >= 2)):
                return {'input': changed, 'output': output}
        except AssertionError:
            continue
    return None


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    stable = tuple(int(x) for x in rng.permutation(10)) if rng.random() < .25 else None
    layout = _pick(rng, ('stacked', 'side_by_side'))
    regime = int(rng.integers(3))
    # The legal two-example minimum still includes an exact correspondence
    # contrast, saturation at two and three, and two different key templates.
    for _ in range(60):
        first = _make_pair(rng, 2, 4, True, layout, stable, repeated=True)
        first_info = _decode(first['input'])
        _, first_count, _ = _render(first_info)
        if not np.any(first_count == 2):
            continue
        contrast = _swap_contrast(first)
        if contrast is not None:
            break
    else:
        raise RuntimeError('could not construct a correspondence contrast')
    puzzle = {'train': [first, contrast], 'test': []}
    used = {_key(first['input']), _key(contrast['input'])}
    for split, count in (('train', nt-2), ('test', ne)):
        for _ in range(count):
            for retry in range(20):
                is_test = split == 'test'
                nkeys = 3 if is_test and regime == 0 else int(rng.integers(1, 3 if not is_test else 4))
                total = int(rng.integers(5, 8)) if is_test else (4 if nkeys == 2 else int(rng.integers(3, 5)))
                pair_layout = layout if regime == 1 and not is_test else (
                    ('side_by_side' if layout == 'stacked' else 'stacked') if regime == 1 else _pick(rng, ('stacked', 'side_by_side')))
                pair = _make_pair(rng, nkeys, total, is_test, pair_layout, stable)
                key = _key(pair['input'])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('duplicate input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _omission_differs(info, output, **kwargs):
    try:
        return not np.array_equal(_render(info, **kwargs)[0], output)
    except AssertionError:
        # An alternative that clips or overwrites a key is already incompatible.
        return True


def _features(pair):
    info = _decode(pair['input'])
    _, count, _ = _render(info)
    ref, target = info['reference'], info['target']
    r, c, h, w = ref['box']
    y, x, k, l = target['box']
    layout = 'stacked' if r+h <= y or y+k <= r else 'side_by_side'
    signatures = tuple(sorted(tuple((dr, dc) for dr, dc, _ in decoration)
                              for _, decoration in info['dictionaries'].values()))
    return dict(input_shape=pair['input'].shape, output_shape=pair['output'].shape,
                reference_shape=(h, w), layout=layout, keys=len(info['keys']),
                anchors=len(info['hubs']), max_overlap=int(count.max()),
                overlaps=int(np.count_nonzero(count >= 2)), singles=int(np.count_nonzero(count == 1)),
                geometries=signatures, sizes=tuple(len(s) for s in signatures),
                matrices=tuple(sorted({_MATRICES.index(hub['matrix']) for hub in info['hubs']})),
                repetitions=tuple(sorted(info['repetitions'].values())),
                decorations=tuple(sorted({color for _, d in info['dictionaries'].values() for _, _, color in d})),
                role_profile=(info['background'], ref['fill'], target['fill'], target['frame'],
                              tuple(sorted(info['markers'])), tuple(sorted(info['keys']))))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, output = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            predicted, count, by_key = _render(info)
            assert np.array_equal(predicted, output), 'incorrect local stamp, weld, erasure, or crop'
            assert count.max() >= 2 and np.any(count == 1)
            assert all(any(count[p] == 1 for p in cells) for cells in by_key.values())
            # Fixed points and control erasure are checked from input roles.
            for hub in info['hubs']:
                ar, ac = hub['position']
                assert int(output[ar+1, ac+1]) == hub['key']
                for dr, dc in hub['arms'].values():
                    if count[ar+dr, ac+dc] == 0:
                        assert output[ar+dr+1, ac+dc+1] == info['target']['fill']
            assert np.all(output[1:-1, 1:-1][count >= 2] == info['target']['frame'])
            assert _omission_differs(info, output, ignore_orientation=True), 'orientation must affect the answer'
            assert _omission_differs(info, output, weld=False), 'welding must affect the answer'
            if len(info['keys']) > 1:
                assert _omission_differs(info, output, generic_key=min(info['keys'])), 'key selection must matter'
            info['count'] = count
            infos[split].append(info)
    assert any(np.any(info['count'] == 2) and np.any(info['count'] == 3)
               for info in infos['train']), 'training must demonstrate overlap saturation'
    assert all(info['count'].max() >= 3 for info in infos['test']), 'tests combine three-way overlap'
    assert min(len(info['hubs']) for info in infos['test']) > max(len(info['hubs']) for info in infos['train'])
    # Exact training contrast: identical dictionary, centers, arm locations and
    # canvas, with only one pair of arm colors exchanged; weld geometry changes.
    a, b = infos['train'][:2]
    assert a['target']['box'] == b['target']['box'] and a['dictionaries'] == b['dictionaries']
    assert a['reference']['box'] == b['reference']['box'] and a['repetitions'] == b['repetitions']
    ga, gb = a['grid'], b['grid']
    assert ga.shape == gb.shape and np.count_nonzero(ga != gb) == 2
    assert [h['position'] for h in a['hubs']] == [h['position'] for h in b['hubs']]
    assert [h['cells'] for h in a['hubs']] == [h['cells'] for h in b['hubs']]
    assert not np.array_equal(a['count'] >= 2, b['count'] >= 2)
    assert len(a['keys']) == 2 and all(n >= 2 for n in a['repetitions'].values())
    assert _same_color_weld(a), 'same-colored decorations weld too'
    assert not np.array_equal(_render(a, ignore_orientation=True)[0], puzzle['train'][0]['output'])
    assert not np.array_equal(_render(a, generic_key=min(a['keys']))[0], puzzle['train'][0]['output'])
    assert any(len({hub['matrix'] for hub in a['hubs'] if hub['key'] == key}) >= 2
               for key in a['keys'])


def self_test() -> dict:
    """Bounded checks and semantic diversity measurements over 42 seeds."""
    puzzles, records = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert np.array_equal(solve(np.asarray(pair['input']).copy()), pair['output'])
                records.append(_features(pair))
        assert len({_features(p)['geometries'] for s in puzzle for p in puzzle[s]}) >= 2
    assert len({tuple(_key(p['input']) for s in puzzle for p in puzzle[s]) for puzzle in puzzles}) == 42
    geometries = {shape for f in records for shape in f['geometries']}
    matrices = {m for f in records for m in f['matrices']}
    assert matrices == set(range(8))
    assert {f['keys'] for f in records} == {1, 2, 3}
    assert {f['anchors'] for f in records} >= {3, 4, 5, 6, 7}
    assert {f['layout'] for f in records} == {'stacked', 'side_by_side'}
    assert len(geometries) >= 70
    assert len({f['input_shape'] for f in records}) >= 20
    assert len({f['output_shape'] for f in records}) >= 25
    assert len({f['role_profile'] for f in records}) >= 30
    assert {len(f['decorations']) for f in records} >= {1, 2}
    assert {n for f in records for n in f['repetitions']} >= {1, 2}
    assert any(f['max_overlap'] >= 3 for f in records)
    assert any(f['input_shape'] == (30, 30) for f in records)
    assert _key(puzzles[0]['test'][0]['input']) == _key(generate_puzzle(0)['test'][0]['input'])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(31415+31*nt+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Label corruption, lost crop extent, and a malformed target arm all fail.
    original = puzzles[0]
    for mutation in ('weld', 'crop', 'arm'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in original[s]] for s in original}
        pair = bad['test'][0]
        info = _decode(pair['input'])
        _, count, _ = _render(info)
        if mutation == 'weld':
            y, x = np.argwhere(count >= 2)[0]
            pair['output'][y+1, x+1] = info['target']['fill']
        elif mutation == 'crop':
            pair['output'] = pair['output'][1:].copy()
        else:
            r, c, _, _ = info['target']['box']
            hub = info['hubs'][0]
            ar, ac = hub['position']
            dr, dc = next(iter(hub['arms'].values()))
            pair['input'][r+ar+dr+1, c+ac+dc+1] = info['target']['fill']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'{mutation} mutation accepted')
    return dict(seeds=42, pairs=len(records), stencil_geometries=len(geometries),
                input_shapes=len({f['input_shape'] for f in records}),
                output_shapes=len({f['output_shape'] for f in records}),
                symmetries=sorted(matrices), key_counts=sorted({f['keys'] for f in records}),
                target_counts=sorted({f['anchors'] for f in records}),
                maximum_overlap=max(f['max_overlap'] for f in records),
                note='Self-authored invariant and diversity checks; no independent semantic approval.')
