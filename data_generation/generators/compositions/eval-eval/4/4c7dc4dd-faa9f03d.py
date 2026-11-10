"""Corner-transport docking: a locally demonstrated involution rides a rerouted cable.

Only Python/NumPy are needed. No parent code, files, labels, or generation state
are consulted by solve(). See task_spec.md for the complete public convention.
"""
from __future__ import annotations

import json
import numpy as np

BG, REPAIR, DIRECTION, FRAME = 0, 2, 4, 8
PAYLOAD = (1, 3, 5, 6, 7, 9)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
# Each geometry is its own inverse and commutes with exchanging two colors.
OPS = tuple((g, swap) for g in range(6) for swap in (False, True))


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            for q in _neighbors(stack.pop(), remaining):
                remaining.remove(q)
                found.add(q)
                stack.append(q)
        result.append(found)
    return result


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
    return [(a[0] + i*d[0], a[1] + i*d[1])
            for i in range(max(abs(b[0]-a[0]), abs(b[1]-a[1]))+1)]


def _poly(points):
    cells = [points[0]]
    for a, b in zip(points, points[1:]):
        cells.extend(_line(a, b)[1:])
    return cells


def _simple(walk):
    cells = set(walk)
    return (len(cells) == len(walk) and len(cells) >= 2
            and all(len(_neighbors(p, cells)) == (1 if p in (walk[0], walk[-1]) else 2)
                    for p in cells))


def _turns(walk):
    steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _apply(icon, op):
    g, swap = op
    if g == 0:
        out = icon.copy()
    elif g == 1:
        out = icon[::-1].copy()
    elif g == 2:
        out = icon[:, ::-1].copy()
    elif g == 3:
        out = icon[::-1, ::-1].copy()
    elif g == 4:
        out = icon.T.copy()
    else:
        out = icon.T[::-1, ::-1].copy()
    if swap:
        colors = sorted(set(map(int, np.unique(icon))) - {BG})
        assert len(colors) == 2
        a, b = colors
        out = np.where(out == a, b, np.where(out == b, a, out)).astype(int)
    return out


def _rect_cells(box):
    r, c, m = box
    return {(y, x) for y in range(r, r+m) for x in range(c, c+m)}


def _border(box):
    r, c, m = box
    return {(y, x) for y, x in _rect_cells(box)
            if y in (r, r+m-1) or x in (c, c+m-1)}


def _inner(box):
    r, c, m = box
    return np.s_[r+1:r+m-1, c+1:c+m-1]


def _stations(grid):
    boxes = []
    for cells in _components(map(tuple, np.argwhere(grid == FRAME))):
        rs, cs = zip(*cells)
        r, c = min(rs), min(cs)
        h, w = max(rs)-r+1, max(cs)-c+1
        assert h == w and 5 <= h <= 7, 'stations must be square frames'
        box = (int(r), int(c), int(h))
        assert cells == _border(box), 'frame must be an intact hollow outline'
        boxes.append(box)
    assert len(boxes) == 4 and len({b[2] for b in boxes}) == 1
    return sorted(boxes)


def _recover(grid, boxes):
    outside = np.ones(grid.shape, bool)
    for b in boxes:
        r, c, m = b
        outside[r:r+m, c:c+m] = False
    colors = set(map(int, np.unique(grid[outside]))) - {BG, REPAIR, DIRECTION, FRAME}
    assert len(colors) == 1, 'one exterior cable color must be identifiable'
    trace = colors.pop()
    assert trace in PAYLOAD
    work = grid.copy()
    work[work == REPAIR] = trace
    cells = set(map(tuple, np.argwhere((work == trace) & outside)))
    # Exactly one-cell straight omissions, inferred simultaneously. Marked
    # bends are explicit waypoints; no unmarked missing corner is invented.
    gaps = set()
    for r, c in map(tuple, np.argwhere((work == BG) & outside)):
        for d in ((1, 0), (0, 1)):
            a, b = (r-d[0], c-d[1]), (r+d[0], c+d[1])
            if a in cells and b in cells:
                gaps.add((r, c))
    cells |= gaps
    return trace, cells, gaps


def _docks(box, cells):
    r, c, m = box
    result = []
    # The corner cells are not ports. The cable must approach normally.
    for off in range(1, m-1):
        for frame, d in (((r, c+off), (-1, 0)), ((r+m-1, c+off), (1, 0)),
                         ((r+off, c), (0, -1)), ((r+off, c+m-1), (0, 1))):
            p = _add(frame, d)
            if p in cells:
                result.append((p, frame))
    return result


def _walk(cells, start):
    assert len(_neighbors(start, cells)) == 1
    order, prev, cur = [], None, start
    while True:
        order.append(cur)
        options = [q for q in _neighbors(cur, cells) if q != prev]
        if not options:
            break
        assert len(options) == 1 and options[0] not in order, 'cables cannot branch or loop'
        prev, cur = cur, options[0]
    assert set(order) == cells
    return order


def _read(grid):
    """All semantic facts reconstructed from the input pixels alone."""
    grid = np.asarray(grid)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
    boxes = _stations(grid)
    icons = [grid[_inner(b)].copy() for b in boxes]
    blank = [i for i, x in enumerate(icons) if np.all(x == BG)]
    assert len(blank) == 1
    trace, cells, gaps = _recover(grid, boxes)
    for i, x in enumerate(icons):
        if i != blank[0]:
            colors = set(map(int, np.unique(x))) - {BG}
            assert len(colors) == 2 and colors <= set(PAYLOAD)-{trace}
    qs = list(map(tuple, np.argwhere(grid == DIRECTION)))
    assert len(qs) == 1
    q = qs[0]
    ns = _neighbors(q, cells)
    assert len(ns) == 1, 'direction mark must have one adjacent pivot'
    pivot = ns[0]
    d = (q[0]-pivot[0], q[1]-pivot[1])
    comps = _components(cells)
    assert len(comps) == 2
    active = next(c for c in comps if pivot in c)
    reference = next(c for c in comps if pivot not in c)
    assert len(_neighbors(pivot, active)) == 2
    old_start = (pivot[0]-d[0], pivot[1]-d[1])
    assert old_start in active
    retained_start = next(p for p in _neighbors(pivot, active) if p != old_start)
    assert (retained_start[0]-pivot[0])*d[0] + (retained_start[1]-pivot[1])*d[1] == 0
    docks = [_docks(b, cells) for b in boxes]
    assert all(len(ds) <= 1 for ds in docks)
    ref_indices = [i for i, ds in enumerate(docks) if ds and ds[0][0] in reference]
    active_indices = [i for i, ds in enumerate(docks) if ds and ds[0][0] in active]
    assert len(ref_indices) == 2 and len(active_indices) == 1 and not docks[blank[0]]
    source = active_indices[0]
    assert source != blank[0] and blank[0] not in ref_indices
    ref_walk = _walk(reference, docks[ref_indices[0]][0][0])
    assert ref_walk[-1] == docks[ref_indices[1]][0][0]
    assert _turns([docks[ref_indices[0]][0][1]] + ref_walk + [docks[ref_indices[1]][0][1]]) == 1
    a, b = [icons[i] for i in ref_indices]
    assert set(np.unique(a)) == set(np.unique(b))
    candidates = [op for op in OPS if np.array_equal(_apply(a, op), b)]
    assert len(candidates) == 1 and candidates[0] != (0, False), 'reference must teach a unique nonidentity involution'
    op = candidates[0]
    assert not np.array_equal(_apply(icons[source], op), icons[source]), 'target must witness this operation'
    full_active = _walk(active, docks[source][0][0])
    assert pivot in full_active and full_active.index(pivot) > 0
    old = list(reversed(full_active[:full_active.index(pivot)]))
    assert old[0] == old_start
    # Include the approach to the station face: this also checks the final
    # docking direction, independently of the chosen polyline representation.
    old_walk = [pivot] + old + [docks[source][0][1]]
    bends = _turns(old_walk)
    ray, p = [], q
    while _inside(p, grid.shape) and grid[p] != FRAME:
        assert grid[p] in (BG, DIRECTION), 'new ray must be clear'
        ray.append(p)
        p = _add(p, d)
    assert _inside(p, grid.shape) and p in _border(boxes[blank[0]])
    r, c, m = boxes[blank[0]]
    assert p[0] not in (r, r+m-1) or p[1] not in (c, c+m-1), 'no corner docking'
    assert _add(p, d) in _rect_cells(boxes[blank[0]]) - _border(boxes[blank[0]])
    assert ray and all(x not in cells for x in ray)
    assert all(len(_neighbors(x, cells)) == (1 if x == q else 0) for x in ray)
    return dict(boxes=boxes, icons=icons, blank=blank[0], source=source,
                trace=trace, cells=cells, gaps=gaps, pivot=pivot, direction=d,
                old=old, ray=ray, bends=bends, op=op, ref_indices=ref_indices,
                active=active, retained_turns=_turns(full_active[full_active.index(pivot):]))


def _render_answer(inp, scene, *, op=None, bends=None):
    out = inp.copy()
    out[out == REPAIR] = scene['trace']
    for p in scene['gaps']:
        out[p] = scene['trace']
    for p in scene['old']:
        out[p] = BG
    icon = scene['icons'][scene['source']]
    k = scene['bends'] if bends is None else bends
    action = scene['op'] if op is None else op
    answer = _apply(icon, action) if k % 2 else icon.copy()
    out[_inner(scene['boxes'][scene['source']])] = BG
    out[_inner(scene['boxes'][scene['blank']])] = answer
    for p in scene['ray']:
        out[p] = scene['trace']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = np.asarray(input_grid)
    return _render_answer(inp, _read(inp))


def _box_at_port(endpoint, outward, m, offset):
    """Put a station one cell beyond a cable endpoint along outward."""
    r, c = _add(endpoint, outward)
    if outward == (0, 1):
        return (r-offset, c, m)
    if outward == (0, -1):
        return (r-offset, c-m+1, m)
    if outward == (1, 0):
        return (r, c-offset, m)
    return (r-m+1, c-offset, m)


def _boxes_ok(boxes, shape):
    for i, (r, c, m) in enumerate(boxes):
        if r < 0 or c < 0 or r+m > shape[0] or c+m > shape[1]:
            return False
        for rr, cc, mm in boxes[:i]:
            # A one-cell gap also separates diagonally adjacent frames.
            if not (r+m < rr or rr+mm < r or c+m < cc or cc+mm < c):
                return False
    return True


def _cable_ok(walk, boxes, expected):
    if not _simple(walk):
        return False
    cells = set(walk)
    for i, b in enumerate(boxes):
        area = _rect_cells(b)
        if cells & area:
            return False
        contacts = _docks(b, cells)
        if len(contacts) != (1 if i in expected else 0):
            return False
        # No accidental contact with a frame corner or running alongside it.
        adj = {(p, q) for p in cells for q in _neighbors(p, area)}
        if len(adj) != len(contacts):
            return False
        if contacts and contacts[0][0] not in (walk[0], walk[-1]):
            return False
    return True


def _extend(rng, points, shape, turns):
    """Random rectilinear route, with alternating axes and variable spacings."""
    axis = 0  # first supplied segment is horizontal
    for _ in range(turns):
        cur = points[-1]
        choices = [v for v in range(1, shape[axis]-1) if 3 <= abs(v-cur[axis]) <= 11]
        rng.shuffle(choices)
        found = False
        for v in choices:
            q = list(cur)
            q[axis] = int(v)
            trial = points + [tuple(q)]
            if _simple(_poly(trial)):
                points = trial
                found = True
                break
        if not found:
            return None
        axis = 1-axis
    return points


def _layout(rng, n, bends, retained_bends):
    m = n+2
    low = max(20, 4*m)
    for _ in range(1600):
        h, w = map(int, rng.integers(low, 31, size=2))
        shape = (h, w)
        pivot = (int(rng.integers(2, h-2)), int(rng.integers(m+3, w-4)))
        blank_end = (pivot[0], int(rng.integers(m, pivot[1])))
        blank = _box_at_port(blank_end, (0, -1), m, int(rng.integers(1, m-1)))
        end = (pivot[0], min(w-2, pivot[1]+int(rng.integers(3, 8))))
        points = _extend(rng, [pivot, end], shape, bends)
        if points is None:
            continue
        old = _poly(points)
        last = (int(np.sign(points[-1][0]-points[-2][0])), int(np.sign(points[-1][1]-points[-2][1])))
        source = _box_at_port(old[-1], last, m, int(rng.integers(1, m-1)))
        boxes = [blank, source]
        if not _boxes_ok(boxes, shape) or not _cable_ok(old, boxes, {1}):
            continue
        ray = _line(pivot, blank_end)[1:]
        if any(_neighbors(p, set(old)) != ([pivot] if p == ray[0] else []) for p in ray):
            continue
        sign = int(rng.choice([-1, 1]))
        anchor_end = (pivot[0]+sign*int(rng.integers(2, 7)), pivot[1])
        if not _inside(anchor_end, shape):
            continue
        anchor_points = [pivot, anchor_end]
        # Reuse the walk builder in transposed coordinates for a vertical start.
        transposed = [(c, r) for r, c in anchor_points]
        transposed = _extend(rng, transposed, shape[::-1], retained_bends)
        if transposed is None:
            continue
        anchor = _poly([(c, r) for r, c in transposed])
        active = list(reversed(anchor)) + old[1:]
        if not _cable_ok(active, boxes, {1}):
            continue
        if any(_neighbors(p, set(active)) != ([pivot] if p == ray[0] else []) for p in ray):
            continue
        # Place the one-corner worked example in whatever free space remains.
        for _ in range(100):
            corner = (int(rng.integers(2, h-2)), int(rng.integers(2, w-2)))
            sr, sc = map(int, rng.choice([-1, 1], size=2))
            a = (corner[0]+sr*int(rng.integers(2, 10)), corner[1])
            b = (corner[0], corner[1]+sc*int(rng.integers(2, 10)))
            if not _inside(a, shape) or not _inside(b, shape):
                continue
            refs = [_box_at_port(a, (sr, 0), m, int(rng.integers(1, m-1))),
                    _box_at_port(b, (0, sc), m, int(rng.integers(1, m-1)))]
            all_boxes = boxes+refs
            ref_walk = _poly([a, corner, b])
            if not _boxes_ok(all_boxes, shape):
                continue
            if not _cable_ok(active, all_boxes, {1}) or not _cable_ok(ref_walk, all_boxes, {2, 3}):
                continue
            ref_set, active_set = set(ref_walk), set(active)
            if ref_set & active_set or any(_neighbors(p, active_set) for p in ref_set):
                continue
            if any(p in ref_set or _neighbors(p, ref_set) for p in ray):
                continue
            # A new ray can approach only its assigned vacant station.
            if not _cable_ok([pivot]+ray, all_boxes, {0}):
                continue
            return shape, all_boxes, active, ref_walk, old, ray, pivot
    raise RuntimeError(f'could not pack a {bends}-corner docking scene of scale {n}')


def _motif(rng, n):
    """Mix connected little silhouettes and sparse/structured multicolor stamps."""
    a = np.zeros((n, n), int)
    mode = int(rng.integers(3))
    if mode == 0:
        p = (int(rng.integers(n)), int(rng.integers(n)))
        for _ in range(int(rng.integers(n+2, n*n+2))):
            a[p] = 1
            candidates = [_add(p, d) for d in DIRS if _inside(_add(p, d), a.shape)]
            p = candidates[int(rng.integers(len(candidates)))]
    elif mode == 1:
        a = rng.choice([0, 1], (n, n), p=[.45, .55])
    else:
        r, c = map(int, rng.integers(n, size=2))
        a[r, :] = a[:, c] = 1
        a[int(rng.integers(n)), int(rng.integers(n))] = 1
    filled = list(map(tuple, np.argwhere(a == 1)))
    if len(filled) < 3:
        a[0, :3] = 1
        filled = list(map(tuple, np.argwhere(a == 1)))
    rng.shuffle(filled)
    for p in filled[:int(rng.integers(1, min(len(filled)-1, n)+1))]:
        a[p] = 3
    return a


def _paint(icon, palette):
    return np.where(icon == 1, palette[0], np.where(icon == 3, palette[1], BG)).astype(int)


def _make_pair(rng, bends, n, retained_bends, forbidden_op=None):
    shape, boxes, active, ref_walk, old, ray, pivot = _layout(rng, n, bends, retained_bends)
    operations = [op for op in OPS if op != (0, False) and op != forbidden_op]
    op = operations[int(rng.integers(len(operations)))]
    for _ in range(300):
        ref = _motif(rng, n)
        partner = _apply(ref, op)
        if [x for x in OPS if np.array_equal(_apply(ref, x), partner)] == [op]:
            break
    else:
        raise RuntimeError('could not make an informative reference motif')
    for _ in range(300):
        target = _motif(rng, n)
        if not np.array_equal(target, _apply(target, op)):
            break
    else:
        raise RuntimeError('could not make an informative target motif')
    palette = list(map(int, rng.permutation(PAYLOAD)))
    trace, ref_colors = palette[0], palette[1:3]
    mode = int(rng.integers(3))
    target_colors = ref_colors if mode == 0 else [ref_colors[0], palette[3]] if mode == 1 else palette[3:5]
    icons = [np.zeros((n, n), int), _paint(target, target_colors),
             _paint(ref, ref_colors), _paint(partner, ref_colors)]
    inp = np.zeros(shape, int)
    for box, icon in zip(boxes, icons):
        for p in _border(box):
            inp[p] = FRAME
        inp[_inner(box)] = icon
    latent = set(active) | set(ref_walk)
    for p in latent:
        inp[p] = trace
    inp[ray[0]] = DIRECTION
    # Reject close parallel runs that would falsely imply a one-cell bridge.
    if _recover(inp, boxes)[1] != latent:
        return None
    marked = {pivot}
    old_corners = [old[i] for i in range(1, len(old)-1)
                   if (old[i][0]-old[i-1][0], old[i][1]-old[i-1][1]) !=
                      (old[i+1][0]-old[i][0], old[i+1][1]-old[i][1])]
    if old_corners:
        marked.add(old_corners[int(rng.integers(len(old_corners)))])
    candidates = sorted(latent-marked)
    rng.shuffle(candidates)
    marked.update(candidates[:int(rng.integers(1, 5))])
    for p in marked:
        inp[p] = REPAIR
    # Omissions are straight and isolated; keep evidence at endpoints and bends.
    candidates = []
    for p in sorted(latent-marked-{pivot}):
        ns = _neighbors(p, latent)
        if len(ns) == 2 and ns[0][0]+ns[1][0] == 2*p[0] and ns[0][1]+ns[1][1] == 2*p[1]:
            if all(inp[q] == trace for q in ns):
                candidates.append(p)
    rng.shuffle(candidates)
    holes = 0
    desired = int(rng.integers(1, 4))
    for p in candidates:
        ns = _neighbors(p, latent)
        if not all(inp[q] == trace for q in ns):
            continue
        inp[p] = BG
        if _recover(inp, boxes)[1] == latent:
            holes += 1
            if holes == desired:
                break
        else:
            inp[p] = trace
    if not holes:
        return None
    # Construct the label from the undamaged route and sampled local relation.
    out = np.zeros(shape, int)
    for box, icon in zip(boxes, icons):
        for p in _border(box):
            out[p] = FRAME
        out[_inner(box)] = icon
    for p in latent-set(old[1:]) | set(ray):
        out[p] = trace
    out[_inner(boxes[1])] = BG
    moved = _apply(icons[1], op) if bends % 2 else icons[1]
    out[_inner(boxes[0])] = moved
    # Random global symmetries remove a privileged direction or scene position.
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    inp = np.rot90(np.fliplr(inp) if flip else inp, k).copy()
    out = np.rot90(np.fliplr(out) if flip else out, k).copy()
    try:
        scene = _read(inp)
        assert scene['bends'] == bends and np.array_equal(solve(inp), out)
        assert forbidden_op is None or scene['op'] != forbidden_op
        if forbidden_op is not None and bends % 2:
            occupied = scene['icons'][scene['source']]
            assert not np.array_equal(_apply(occupied, forbidden_op), _apply(occupied, scene['op']))
    except AssertionError:
        return None
    return {'input': inp, 'output': out}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _count(num_train, 'num_train', 2, 10)
    b = _count(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    first_op = None
    for split, count in (('train', a), ('test', b)):
        for i in range(count):
            # Both parity outcomes are taught even when only two trains exist.
            bends = (1 if i in (0, 2) else 2 if i == 1 else int(rng.integers(0, 3))) if split == 'train' else 3+i%2
            n = int(rng.choice([3, 3, 4])) if split == 'train' else int(rng.choice([3, 4]))
            retained_bends = int(rng.integers(0, 3))
            for _ in range(80):
                pair = _make_pair(rng, bends, n, retained_bends, first_op if split == 'train' and i in (1, 2) else None)
                if pair is None:
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == 'train' and i == 0:
                    first_op = _read(pair['input'])['op']
                break
            else:
                raise RuntimeError('bounded construction attempts exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, parities, train_scenes = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = (np.asarray(pair[k]) for k in ('input', 'output'))
            for x in (inp, out):
                assert x.ndim == 2 and all(1 <= n <= 30 for n in x.shape)
                assert np.issubdtype(x.dtype, np.integer) and np.all((0 <= x) & (x <= 9))
            key = (inp.shape, inp.astype(np.uint8).tobytes())
            assert key not in seen
            seen.add(key)
            scene = _read(inp)
            assert inp.shape == out.shape and np.array_equal(_render_answer(inp, scene), out)
            assert scene['gaps'] and np.count_nonzero(inp == REPAIR) >= 2
            assert not np.any(np.isin(out, [REPAIR, DIRECTION]))
            assert np.all(out[inp == FRAME] == FRAME)
            assert np.all(out[_inner(scene['boxes'][scene['source']])] == BG)
            assert np.array_equal(out[_inner(scene['boxes'][scene['blank']])],
                                  _apply(scene['icons'][scene['source']], scene['op']) if scene['bends'] % 2
                                  else scene['icons'][scene['source']])
            # Deleting through bends must preserve the whole retained cable.
            assert all(out[p] == scene['trace'] for p in scene['cells']-set(scene['old']))
            assert all(out[p] == BG for p in scene['old'])
            assert all(out[p] == scene['trace'] for p in scene['ray'])
            if split == 'train':
                parities.add(scene['bends'] % 2)
                train_scenes.append(scene)
    assert parities == {0, 1}, 'training must distinguish transport from unconditional transformation'
    assert train_scenes[0]['op'] != train_scenes[1]['op']
    if len(train_scenes) >= 3:
        first_op = train_scenes[0]['op']
        assert any(z['bends'] % 2 and not np.array_equal(
            _apply(z['icons'][z['source']], first_op), _apply(z['icons'][z['source']], z['op']))
                   for z in train_scenes[1:]), 'a global reference shortcut still fits all demonstrations'


def self_test():
    shapes, operators, directions, bends, retained, palettes, motifs, signatures = (set() for _ in range(8))
    examples = []
    for seed in range(42):
        puzzle = generate_puzzle(seed, num_test=1+seed%2)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((p['input'].shape, p['input'].tobytes()) for s in puzzle for p in puzzle[s]))
        for s in ('train', 'test'):
            for p in puzzle[s]:
                z = _read(p['input'])
                shapes.add(p['input'].shape)
                operators.add(z['op'])
                directions.add(z['direction'])
                bends.add(z['bends'])
                retained.add(z['retained_turns'])
                palettes.add(tuple(sorted(np.unique(p['input']))))
                motifs.add(tuple(map(tuple, z['icons'][z['source']] != BG)))
                examples.append((p, z))
    assert len(signatures) == 42 and len(shapes) >= 20
    assert len(operators) == 11 and directions == set(DIRS)
    assert bends >= {0, 1, 2, 3, 4} and retained >= {0, 1, 2}
    assert len(palettes) >= 15 and len(motifs) >= 100
    # Reproducibility, all counts, and strict count validation are bounded.
    a, b = generate_puzzle(43), generate_puzzle(43)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    for nt, nq in ((2, 1), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(99+nt*5+nq, nt, nq)
        assert len(p['train']) == nt and len(p['test']) == nq
        validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Same target and routes, a different informative reference: answers change.
    pair, z = next((p, z) for p, z in examples if z['bends'] % 2)
    twin = pair['input'].copy()
    ref_a, ref_b = z['ref_indices']
    first = z['icons'][ref_a]
    alternatives = []
    for op in OPS:
        if op in ((0, False), z['op']):
            continue
        twin[_inner(z['boxes'][ref_b])] = _apply(first, op)
        try:
            prediction = solve(twin)
        except AssertionError:
            continue
        destination = _inner(z['boxes'][z['blank']])
        if not np.array_equal(prediction[destination], pair['output'][destination]):
            alternatives.append(prediction)
    assert alternatives, 'complete reference must affect the output'
    assert not np.array_equal(_render_answer(pair['input'], z, op=(0, False)), pair['output'])
    # Ignoring parity, counting all active bends, or keeping the old branch fails.
    assert any(not np.array_equal(_render_answer(p['input'], z, bends=1), p['output'])
               for p, z in examples if z['bends'] % 2 == 0)
    assert any(not np.array_equal(_render_answer(p['input'], z, bends=z['bends']+z['retained_turns']), p['output'])
               for p, z in examples if z['retained_turns'] % 2)
    assert any(not np.array_equal(_render_answer(p['input'], z, bends=z['bends']+1+z['retained_turns']), p['output'])
               for p, z in examples)
    for p, z in examples[:8]:
        for k in (1, 2, 3):
            # Local involutions conjugate under whole-scene rotations.
            assert np.array_equal(solve(np.rot90(p['input'], k)), np.rot90(p['output'], k))
    # Full-output mutations, including unchanged reference and background cells.
    puzzle = generate_puzzle(123)
    q = puzzle['test'][0]
    z = _read(q['input'])
    sites = [z['old'][0], z['ray'][-1], z['pivot'], z['boxes'][0][:2],
             tuple(np.argwhere(q['input'] == BG)[0])]
    for point in sites:
        bad = {s: [dict(p) for p in puzzle[s]] for s in puzzle}
        bad['test'][0]['output'] = q['output'].copy()
        bad['test'][0]['output'][point] = (int(q['output'][point])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'pairs': len(examples), 'shapes': len(shapes),
            'operators': len(operators), 'directions': len(directions),
            'source_masks': len(motifs), 'palettes': len(palettes)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
