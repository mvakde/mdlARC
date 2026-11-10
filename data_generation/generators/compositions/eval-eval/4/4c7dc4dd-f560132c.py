"""Corner-fed dye through a broken tile.

The framed before/after example teaches a periodic geodesic color wave.
Four loose shards supply its new domain; an embedded corner key supplies phases.
No parent imports, files, random state, or labels are consulted by solve().
"""
from collections import deque
from itertools import combinations
from math import isqrt

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    _require(_integer(a) and 2 <= a <= 10, "num_train must be an integer in 2..10")
    _require(_integer(b) and 1 <= b <= 4, "num_test must be an integer in 1..4")
    _require(a > b, "num_train must exceed num_test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _signature(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _normal(coords):
    pts = [(int(r), int(c)) for r, c in coords]
    r0, c0 = min(r for r, _ in pts), min(c for _, c in pts)
    return tuple(sorted((r-r0, c-c0) for r, c in pts))


def _turn(shape, k):
    for _ in range(k % 4):
        shape = _normal((c, -r) for r, c in shape)
    return shape


def _dimensions(shape):
    return max(r for r, _ in shape)+1, max(c for _, c in shape)+1


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _DIRS:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _distances(mask, root):
    """Unit-cost, orthogonal shortest paths confined to this one silhouette."""
    h, w = mask.shape
    _require(0 <= root[0] < h and 0 <= root[1] < w and mask[root], "root outside silhouette")
    out = np.full((h, w), -1, dtype=np.int16)
    out[root] = 0
    queue = deque([root])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            rr, cc = r+dr, c+dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and out[rr, cc] < 0:
                out[rr, cc] = out[r, c]+1
                queue.append((rr, cc))
    _require(bool(np.all(out[mask] >= 0)), "disconnected silhouette")
    return out


def _corners(n):
    return ((0, 0), (0, n-1), (n-1, 0), (n-1, n-1))


def _bitmask(cells, n):
    return sum(1 << (int(r)*n+int(c)) for r, c in cells)


def _placements(shape, n, corner, anchor=False):
    found = {}
    corners = _corners(n)
    for k in ((0,) if anchor else range(4)):
        rotated = _turn(shape, k)
        h, w = _dimensions(rotated)
        if h > n or w > n:
            continue
        r0, c0 = (0 if corner < 2 else n-h), (0 if corner % 2 == 0 else n-w)
        pts = tuple((r+r0, c+c0) for r, c in rotated)
        hits = [p in pts for p in corners]
        if sum(hits) == 1 and hits[corner]:
            found.setdefault(_bitmask(pts, n), (pts, k))
    return [(bits, pts, k) for bits, (pts, k) in found.items()]


def _pack(shapes, n):
    """Exhaust all corner assignments and turns; deduplicate equal domains."""
    if len(shapes) != 4 or sum(map(len, shapes)) != n*n:
        return []
    first = _placements(shapes[0], n, 0, anchor=True)
    if not first:
        return []
    table = {(i, c): _placements(shapes[i], n, c) for i in range(1, 4) for c in range(1, 4)}
    full, answers = (1 << (n*n))-1, {}

    def visit(corner, left, used, records):
        if len(answers) > 1:
            return
        if corner == 4:
            if used == full:
                key = tuple(x[0] for x in records)
                answers.setdefault(key, records)
            return
        for piece in left:
            for bits, pts, k in table[piece, corner]:
                if not used & bits:
                    visit(corner+1, tuple(i for i in left if i != piece), used | bits,
                          records+[(bits, pts, k, piece)])

    bits, pts, k = first[0]
    visit(1, (1, 2, 3), bits, [(bits, pts, k, 0)])
    return list(answers.values())


def _owner_from_solution(solution, n):
    owner = np.full((n, n), -1, dtype=np.int8)
    for corner, (_, pts, _, _) in enumerate(solution):
        for p in pts:
            owner[p] = corner
    _require(bool(np.all(owner >= 0)), "incomplete packing")
    return owner


def _wave(owner, key, cycle, width, manhattan=False):
    n = len(owner)
    rr, cc = np.indices(owner.shape)
    out = np.empty_like(owner)
    for corner, root in enumerate(_corners(n)):
        mask = owner == corner
        d = abs(rr-root[0])+abs(cc-root[1]) if manhattan else _distances(mask, root)
        phase = cycle.index(int(key.flat[corner]))
        indices = (phase+d[mask]//width) % 4
        out[mask] = np.asarray(cycle, dtype=np.int8)[indices]
    return out


def _outline_box(comp, grid):
    r0, c0 = min(r for r, _ in comp), min(c for _, c in comp)
    r1, c1 = max(r for r, _ in comp)+1, max(c for _, c in comp)+1
    h, w = r1-r0, c1-c0
    if min(h, w) < 6 or len(comp) != 2*h+2*w-4:
        return None
    values = {int(grid[p]) for p in comp}
    if len(values) != 1:
        return None
    expected = {(r, c) for r in range(r0, r1) for c in range(c0, c1)
                if r in (r0, r1-1) or c in (c0, c1-1)}
    if set(comp) != expected:
        return None
    return r0, c0, h, w, values.pop()


def _infer_reference(before, after, bg, key_colors):
    mask = before != bg
    _require(np.array_equal(mask, after != bg), "reference silhouettes must match")
    colors, counts = np.unique(before[mask], return_counts=True)
    _require(len(colors) == 2 and sorted(map(int, counts))[0] == 1, "reference before needs one seed")
    carrier = int(colors[np.argmax(counts)])
    seed = int(colors[np.argmin(counts)])
    _require(seed in key_colors and carrier not in key_colors, "reference color roles invalid")
    _require(set(map(int, np.unique(after[mask]))) == set(key_colors), "reference must exhibit all key colors")
    root = tuple(map(int, np.argwhere(before == seed)[0]))
    d = _distances(mask, root)
    sequence = []
    for distance in range(int(d.max())+1):
        values = np.unique(after[d == distance])
        _require(len(values) == 1, "reference is not constant on distance shells")
        sequence.append(int(values[0]))
    candidates = []
    for width in range(1, 4):
        if len(sequence) <= 4*width:
            continue
        cycle = tuple(sequence[i*width] for i in range(4))
        if (cycle[0] == seed and set(cycle) == set(key_colors)
                and all(color == cycle[(i//width) % 4] for i, color in enumerate(sequence))):
            candidates.append((cycle, width))
    _require(len(candidates) == 1, "reference must determine one cyclic rhythm")
    cycle, width = candidates[0]
    return dict(cycle=cycle, width=width, mask=mask, root=root, distances=d, carrier=carrier)


def _read(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "canvas must have a uniform background border")
    all_comps = _components(grid != bg)
    frames = [box for comp in all_comps if (box := _outline_box(comp, grid)) is not None]
    _require(len(frames) == 2, "expected two framed reference panels")
    _require(frames[0][2:] == frames[1][2:], "reference frames must have matching dimensions and color")
    outside = grid != bg
    interiors = []
    for r, c, h, w, color in frames:
        interiors.append(grid[r+1:r+h-1, c+1:c+w-1].copy())
        outside[r:r+h, c:c+w] = False
        interior = interiors[-1]
        _require(all(np.all(edge == bg) for edge in (interior[0], interior[-1], interior[:, 0], interior[:, -1])),
                 "reference motif needs a background moat")
    comps = _components(outside)
    _require(len(comps) == 4, "expected four loose shards")
    for x, y in combinations(comps, 2):
        _require(min(max(abs(r-u), abs(c-v)) for r, c in x for u, v in y) >= 2,
                 "loose shards need a one-cell background gap")
    anchors, plain = [], []
    for comp in comps:
        colors, counts = np.unique([grid[p] for p in comp], return_counts=True)
        if len(colors) == 1:
            plain.append(comp)
            continue
        _require(len(colors) == 5 and np.count_nonzero(counts == 1) == 4 and np.count_nonzero(counts > 1) == 1,
                 "anchor needs one carrier plus four singleton key colors")
        singleton_colors = set(map(int, colors[counts == 1]))
        patch = [(r, c) for r, c in comp if int(grid[r, c]) in singleton_colors]
        r, c = min(r for r, _ in patch), min(c for _, c in patch)
        _require(set(patch) == {(r, c), (r+1, c), (r, c+1), (r+1, c+1)}, "key must be 2 by 2")
        anchors.append((comp, grid[r:r+2, c:c+2].copy(), (r, c)))
    _require(len(anchors) == 1 and len(plain) == 3, "expected one anchor and three plain shards")
    anchor, key, key_location = anchors[0]
    key_colors = set(map(int, key.flat))
    ordered = [anchor]+plain
    _require(bg not in key_colors and frames[0][4] not in key_colors, "key overlaps background/frame")
    carriers = {int(grid[p]) for comp in ordered for p in comp} - key_colors
    _require(carriers.isdisjoint({bg, frames[0][4]}), "shard carriers overlap background/frame")
    _require(all(int(grid[comp[0]]) not in key_colors for comp in plain), "plain shards must use carrier colors")
    shapes = [_normal(comp) for comp in ordered]
    total = sum(map(len, shapes))
    n = isqrt(total)
    _require(n*n == total, "shard area must be square")
    solutions = _pack(shapes, n)
    _require(len(solutions) == 1, "expected a unique corner packing")
    owner = _owner_from_solution(solutions[0], n)
    before_index = [i for i, x in enumerate(interiors) if np.any((x != bg) & ~np.isin(x, list(key_colors)))]
    _require(len(before_index) == 1, "reference states must be distinguishable by content")
    index = before_index[0]
    ref = _infer_reference(interiors[index], interiors[1-index], bg, key_colors)
    _require(ref['carrier'] in carriers, "reference carrier must be a visible shard carrier")
    return dict(grid=grid, background=bg, frames=frames, before_index=index,
                before=interiors[index], after=interiors[1-index], reference=ref,
                key=key, key_location=key_location, components=ordered, shapes=shapes,
                solution=solutions[0], owner=owner, carriers=carriers, side=n)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the reference, uniquely assemble the shards, and transfer its wave."""
    scene = _read(input_grid)
    ref = scene['reference']
    return _wave(scene['owner'], scene['key'], ref['cycle'], ref['width'])


def _key_spots(owner):
    return [(r, c) for r in range(len(owner)-1) for c in range(len(owner)-1)
            if np.all(owner[r:r+2, c:c+2] == 0)]


def _eligible(owner, demonstrate):
    n = len(owner)
    if tuple(int(owner[p]) for p in _corners(n)) != (0, 1, 2, 3) or not _key_spots(owner):
        return False
    shapes = []
    for corner in range(4):
        comps = _components(owner == corner)
        if len(comps) != 1 or len(comps[0]) < (7 if corner == 0 else 4):
            return False
        shapes.append(_normal(comps[0]))
    if demonstrate:
        if any(len({_turn(shape, k) for k in range(4)}) != 4 for shape in shapes):
            return False
        if len({min(_turn(shape, k) for k in range(4)) for shape in shapes}) != 4:
            return False
    solutions = _pack(shapes, n)
    return len(solutions) == 1 and np.array_equal(_owner_from_solution(solutions[0], n), owner)


def _random_cut(rng, n, kind):
    """Connected growth or boundary exchanges; no completed parent generator."""
    if kind == 0:
        owner = np.full((n, n), -1, dtype=np.int8)
        for i, (r, c) in enumerate(_corners(n)):
            owner[r, c] = i
        speed = rng.uniform(.3, 2.0, 4)
        for _ in range(n*n-4):
            frontier = []
            for r, c in np.argwhere(owner < 0).tolist():
                labels = sorted({int(owner[r+dr, c+dc]) for dr, dc in _DIRS
                                 if 0 <= r+dr < n and 0 <= c+dc < n and owner[r+dr, c+dc] >= 0})
                frontier.extend((r, c, label) for label in labels)
            weights = np.asarray([speed[label] for _, _, label in frontier])
            r, c, label = frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))]
            owner[r, c] = label
        return owner
    y, x, z = (int(rng.integers(2, n-1)) for _ in range(3))
    owner = np.empty((n, n), dtype=np.int8)
    owner[:y, :x], owner[:y, x:] = 0, 1
    owner[y:, :z], owner[y:, z:] = 2, 3
    # Connected boundary transfers can make returning arms, necks, and notches.
    corners = set(_corners(n))
    for _ in range(n*n*2):
        r, c = int(rng.integers(n)), int(rng.integers(n))
        if (r, c) in corners:
            continue
        old = int(owner[r, c])
        options = sorted({int(owner[r+dr, c+dc]) for dr, dc in _DIRS
                          if 0 <= r+dr < n and 0 <= c+dc < n} - {old})
        if not options:
            continue
        new = int(rng.choice(options))
        owner[r, c] = new
        if (min(np.count_nonzero(owner == old), np.count_nonzero(owner == new)) < 4
                or len(_components(owner == old)) != 1 or len(_components(owner == new)) != 1
                or not _key_spots(owner)):
            owner[r, c] = old
    return owner


def _make_owner(rng, n, width, demonstrate):
    key, cycle = np.arange(4).reshape(2, 2), (0, 1, 2, 3)
    for attempt in range(500):
        owner = _random_cut(rng, n, int(rng.random() < .8))
        if not _eligible(owner, demonstrate):
            continue
        # Require an observable effect of paths constrained by the shard seams.
        a, b = _wave(owner, key, cycle, width), _wave(owner, key, cycle, width, manhattan=True)
        if not np.array_equal(a, b):
            return owner
    # An analytic notched cut supplies a same-rule fallback for every sampled
    # side and width. Its three asymmetric mates are all different up to turns.
    return _fallback_owner(n)


def _fallback_owner(n):
    _require(6 <= n <= 9, "fallback side outside sampler range")
    owner = np.empty((n, n), dtype=np.int8)
    owner[:3, :4], owner[:3, 4:] = 0, 1
    owner[3:, :3], owner[3:, 3:] = 2, 3
    owner[0, 2:4] = 1
    owner[1, 2] = 1
    owner[3, 3] = 2
    owner[2, 4] = 3
    return owner


def _reference_shape(rng):
    """A folded corridor, with optional spurs, long enough to show a repeat."""
    w = int(rng.integers(5, 9))
    cells = [(0, c) for c in range(w)]+[(1, w-1)]+[(2, c) for c in range(w-1, -1, -1)]
    cells += [(3, 0)]+[(4, c) for c in range(w)]
    trim_a, trim_b = int(rng.integers(3)), int(rng.integers(3))
    cells = cells[trim_a:len(cells)-trim_b if trim_b else None]
    root = cells[0] if rng.integers(2) else cells[-1]
    if rng.random() < .5:
        # Outward spurs create equal-distance branches without closing the folds.
        cells += [(-1, c) for c in rng.choice(np.arange(max(1, trim_a), w-1),
                                           size=1, replace=False)]
    points = np.asarray(cells, dtype=int)
    root = np.asarray(root, dtype=int)
    for _ in range(int(rng.integers(4))):
        points = np.column_stack((points[:, 1], -points[:, 0]))
        root = np.array([root[1], -root[0]])
    if rng.integers(2):
        points[:, 1] *= -1
        root[1] *= -1
    mins = points.min(axis=0)
    points -= mins
    root -= mins
    shape = tuple(points.max(axis=0)+1)
    mask = np.zeros(shape, dtype=bool)
    mask[points[:, 0], points[:, 1]] = True
    d = _distances(mask, tuple(map(int, root)))
    _require(int(d.max()) >= 12, "reference too short to demonstrate a four-color repeat")
    return mask, tuple(map(int, root)), d


def _reference_panels(mask, root, d, bg, carrier, frame, cycle, width):
    h, w = mask.shape
    before = np.full((h+4, w+4), bg, dtype=np.int8)
    after = before.copy()
    for panel in (before, after):
        panel[0, :] = panel[-1, :] = frame
        panel[:, 0] = panel[:, -1] = frame
    before[2:h+2, 2:w+2][mask] = carrier
    before[root[0]+2, root[1]+2] = cycle[0]
    after[2:h+2, 2:w+2][mask] = np.asarray(cycle, dtype=np.int8)[(d[mask]//width) % 4]
    return before, after


def _layout(rng, images, bg):
    """Random ordered shelf partitions of all six objects, on either axis."""
    dims = [x.shape for x in images]
    options = []
    # Enumerating 720*32 layouts is unnecessary; bounded randomized shelves
    # keep transient space small and do not constrain pieces to quadrants.
    for attempt in range(1200):
        transpose = bool(rng.integers(2))
        sizes = [s[::-1] if transpose else s for s in dims]
        order = list(map(int, rng.permutation(6)))
        cuts = int(rng.integers(1, 32))
        groups, group = [], []
        for j, i in enumerate(order):
            group.append(i)
            if j == 5 or (cuts >> j) & 1:
                groups.append(group)
                group = []
        heights = [max(sizes[i][0] for i in g) for g in groups]
        widths = [sum(sizes[i][1] for i in g)+len(g)-1 for g in groups]
        h, w = sum(heights)+len(groups)-1, max(widths)
        if h+2 <= 30 and w+2 <= 30:
            options.append((transpose, sizes, groups, heights, widths, h, w))
            if len(options) >= 12:
                break
    if not options:
        return None
    transpose, sizes, groups, heights, widths, h, w = options[int(rng.integers(len(options)))]
    pads_h, pads_w = int(rng.integers(2, min(6, 31-h))), int(rng.integers(2, min(6, 31-w)))
    top, left = int(rng.integers(1, pads_h)), int(rng.integers(1, pads_w))
    result = np.full((h+pads_h, w+pads_w), bg, dtype=np.int8)
    row = top
    for g, height, width in zip(groups, heights, widths):
        col = left+int(rng.integers(w-width+1))
        for i in g:
            image = images[i].T if transpose else images[i]
            r = row+int(rng.integers(height-image.shape[0]+1))
            result[r:r+image.shape[0], col:col+image.shape[1]] = image
            col += image.shape[1]+1
        row += height+1
    return result.T.copy() if transpose else result


def _new_pair(rng, n, width, demonstrate=False, force_fallback=False):
    for attempt in ((24,) if force_fallback else range(25)):
        fallback = attempt == 24
        owner = _fallback_owner(n) if fallback else _make_owner(rng, n, width, demonstrate)
        palette = list(map(int, rng.permutation(10)))
        key = np.asarray(palette[:4], dtype=np.int8).reshape(2, 2)
        bg, frame, carrier = palette[4:7]
        pool = palette[6:6+int(rng.integers(1, 5))]
        cycle = tuple(int(x) for x in rng.permutation(key.reshape(-1)))
        if fallback:
            mask = np.zeros((5, 5), dtype=bool)
            mask[::2, :] = True
            mask[1, 4] = mask[3, 0] = True
            root = (0, 0)
            d = _distances(mask, root)
        else:
            mask, root, d = _reference_shape(rng)
        panels = list(_reference_panels(mask, root, d, bg, carrier, frame, cycle, width))
        if rng.integers(2):
            panels.reverse()
        images = []
        spots = _key_spots(owner)
        kr, kc = spots[int(rng.integers(len(spots)))]
        turns = [0]+(list(map(int, rng.permutation([1, 2, 3]))) if demonstrate
                     else list(map(int, rng.integers(0, 4, 3))))
        for corner in range(4):
            shape = _normal(np.argwhere(owner == corner))
            h, w = _dimensions(shape)
            image = np.full((h, w), bg, dtype=np.int8)
            piece_color = carrier if corner == 0 else int(rng.choice(pool))
            for p in shape:
                image[p] = piece_color
            if corner == 0:
                image[kr:kr+2, kc:kc+2] = key
            images.append(np.rot90(image, -turns[corner]).copy())
        grid = _fallback_layout(panels+images, bg) if fallback else _layout(rng, panels+images, bg)
        if grid is None:
            continue
        output = _wave(owner, key, cycle, width)
        return {"input": grid, "output": output}
    raise RuntimeError("bounded scene layout search exhausted")


def _fallback_layout(images, bg):
    # Used only with the analytic cut and 9x9 reference frames. Each shard
    # bounding box is at most 7x7, so these three shelves fit in 27x21 or less.
    groups = (images[:2], images[2:4], images[4:])
    heights = [max(im.shape[0] for im in group) for group in groups]
    widths = [sum(im.shape[1] for im in group)+1 for group in groups]
    grid = np.full((sum(heights)+4, max(widths)+2), bg, dtype=np.int8)
    row = 1
    for group, height in zip(groups, heights):
        col = 1
        for image in group:
            h, w = image.shape
            grid[row:row+h, col:col+w] = image
            col += w+1
        row += height+1
    return grid


def _reference_twin(pair):
    """Only the filled reference changes; the complete target is pixel-identical."""
    scene = _read(pair['input'])
    ref = scene['reference']
    width = 3-ref['width']  # first member has width 1 or 2
    cycle = (ref['cycle'][0],)+tuple(reversed(ref['cycle'][1:]))
    changed = scene['after'].copy()
    changed[ref['mask']] = np.asarray(cycle, dtype=np.int8)[(ref['distances'][ref['mask']]//width) % 4]
    grid = pair['input'].copy()
    r, c, h, w, _ = scene['frames'][1-scene['before_index']]
    grid[r+1:r+h-1, c+1:c+w-1] = changed
    output = _wave(scene['owner'], scene['key'], cycle, width)
    _require(not np.array_equal(output, pair['output']), "reference twin must change answer")
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return one reproducible puzzle; no generated artifacts are saved to disk."""
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    if regime == 0:
        train_sides, test_sides = (6, 7, 8), (9,)
    elif regime == 1:
        train_sides, test_sides = (8, 9), (6, 7)
    elif regime == 2:
        n = int(rng.integers(6, 10))
        train_sides = test_sides = (n,)
    else:
        train_sides = test_sides = (6, 7, 8, 9)
    first = _new_pair(rng, int(rng.choice(train_sides)), int(rng.integers(1, 3)), demonstrate=True)
    twin = _reference_twin(first)
    puzzle = {'train': [first, twin], 'test': []}
    seen_inputs = {_signature(p['input']) for p in puzzle['train']}
    seen_cuts = {_signature(_read(first['input'])['owner'])}
    for split, count, sides in (('train', a-2, train_sides), ('test', b, test_sides)):
        for _ in range(count):
            for _attempt in range(30):
                width = 3 if split == 'test' and regime in (2, 3) else int(rng.integers(1, 4))
                pair = _new_pair(rng, int(rng.choice(sides)), width)
                scene = _read(pair['input'])
                signature, cut = _signature(pair['input']), _signature(scene['owner'])
                if signature not in seen_inputs and cut not in seen_cuts:
                    seen_inputs.add(signature)
                    seen_cuts.add(cut)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-cut search exhausted")
    return puzzle


def _features(scene):
    owner, ref, n = scene['owner'], scene['reference'], scene['side']
    rr, cc = np.indices(owner.shape)
    detour_cells, max_detour, min_fill = 0, 0, 1.0
    areas = []
    for i, root in enumerate(_corners(n)):
        mask = owner == i
        d = _distances(mask, root)
        excess = d[mask]-(abs(rr-root[0])+abs(cc-root[1]))[mask]
        detour_cells += int(np.count_nonzero(excess))
        max_detour = max(max_detour, int(excess.max()))
        shape = _normal(np.argwhere(mask))
        h, w = _dimensions(shape)
        min_fill = min(min_fill, len(shape)/(h*w))
        areas.append(len(shape))
    phase_cycle = tuple(ref['cycle'].index(int(c)) for c in scene['key'].flat)
    return dict(side=n, canvas=list(scene['grid'].shape), width=ref['width'], cycle=phase_cycle,
                areas=areas, min_fill=round(min_fill, 4), detour_cells=detour_cells,
                max_detour=max_detour, background=scene['background'], carriers=len(scene['carriers']),
                turns=[record[2] for record in scene['solution']],
                before_index=scene['before_index'], reference_shape=list(scene['before'].shape))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = _signature(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            scene = _read(inp)
            ref = scene['reference']
            expected = _wave(scene['owner'], scene['key'], ref['cycle'], ref['width'])
            _require(np.array_equal(out, expected), "output disagrees with visible reference and packing")
            _require(set(map(int, np.unique(out))) == set(map(int, scene['key'].flat)), "output palette")
            _require(all(int(out[p]) == int(scene['key'].flat[i]) for i, p in enumerate(_corners(scene['side']))),
                     "key must set the four corner phases")
            solid = scene['key'].reshape(-1)[scene['owner']]
            _require(not np.array_equal(out, solid), "analogy must change B's solid-piece answer")
            analyses[split].append(scene)
    # The first two are a controlled reference contrast, even at minimum counts.
    x, y = analyses['train'][:2]
    _require(x['grid'].shape == y['grid'].shape, "reference twins must share the target canvas")
    after_box = x['frames'][1-x['before_index']]
    r, c, h, w, _ = after_box
    outside = np.ones(x['grid'].shape, dtype=bool)
    outside[r+1:r+h-1, c+1:c+w-1] = False
    _require(np.array_equal(x['grid'][outside], y['grid'][outside]), "reference twins altered target evidence")
    _require(not np.array_equal(puzzle['train'][0]['output'], puzzle['train'][1]['output']), "reference unused")
    first_ref = x['reference']
    _require(not np.array_equal(puzzle['train'][0]['output'], _wave(x['owner'], x['key'],
             first_ref['cycle'], first_ref['width'], manhattan=True)), "training must expose seam detours")
    _require({record[2] for record in x['solution'][1:]} == {1, 2, 3}, "training must exhibit all nonzero turns")
    train_cuts = {_signature(s['owner']) for s in analyses['train']}
    _require(all(_signature(s['owner']) not in train_cuts for s in analyses['test']), "test cuts must be new")


def self_test():
    """42 seeds, visible invariants, ablations and corruption; no disk writes."""
    canvases, cuts, ref_shapes, backgrounds, cycles, widths, turns, sides, orders = (set() for _ in range(9))
    successor_orders = set()
    min_fill, max_detour, total, detour_pairs, carrier_counts = 1.0, 0, 0, 0, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene = _read(pair['input'])
                f = _features(scene)
                _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), "input-only list solve")
                canvases.add(tuple(f['canvas']))
                cuts.add(_signature(scene['owner']))
                ref_shapes.add((_signature(scene['reference']['mask']), scene['reference']['root']))
                backgrounds.add(f['background'])
                cycles.add(f['cycle'])
                cycle = scene['reference']['cycle']
                key_values = list(map(int, scene['key'].flat))
                start = cycle.index(key_values[0])
                successor_orders.add(tuple(key_values.index(cycle[(start+i) % 4]) for i in range(4)))
                widths.add(f['width'])
                turns.update(f['turns'])
                sides.add(f['side'])
                orders.add(f['before_index'])
                carrier_counts.add(f['carriers'])
                min_fill = min(min_fill, f['min_fill'])
                max_detour = max(max_detour, f['max_detour'])
                total += 1
                detour_pairs += f['detour_cells'] > 0
    _require(len(canvases) >= 25 and len(cuts) >= 100 and len(ref_shapes) >= 30, "structural diversity collapsed")
    _require(backgrounds == set(range(10)), "background role diversity")
    _require(len(cycles) >= 12 and len(successor_orders) == 6 and widths == {1, 2, 3}
             and turns == {0, 1, 2, 3}, "rhythm/turn variation")
    _require(sides == {6, 7, 8, 9} and orders == {0, 1} and len(carrier_counts) >= 3, "scale/presentation variation")
    _require(min_fill < .6 and max_detour >= 4 and detour_pairs >= total*.8, "nontrivial geodesic domains missing")
    # Validate fallback geometry, all rhythm widths, every motion choice and
    # its dimensional guarantee without relying on an unlucky random seed.
    for n in range(6, 10):
        owner = _fallback_owner(n)
        _require(_eligible(owner, True), "fallback must uniquely pack asymmetric shards")
        key = np.arange(4).reshape(2, 2)
        for width in (1, 2, 3):
            _require(not np.array_equal(_wave(owner, key, (0, 1, 2, 3), width),
                         _wave(owner, key, (0, 1, 2, 3), width, manhattan=True)), "fallback detour")
            fallback_pair = _new_pair(np.random.default_rng(100*n+width), n, width,
                                      demonstrate=True, force_fallback=True)
            _require(np.array_equal(solve(fallback_pair['input']), fallback_pair['output']), "fallback pixel reconstruction")
        for turns in ((0, 0, 0, 0), (0, 1, 2, 3), (0, 2, 2, 0), (0, 3, 1, 2)):
            shapes = [_turn(_normal(np.argwhere(owner == i)), turns[i]) for i in range(4)]
            images = [np.full((9, 9), 0, dtype=np.int8) for _ in range(2)]
            images += [np.full(_dimensions(shape), 1, dtype=np.int8) for shape in shapes]
            _require(max(_fallback_layout(images, 0).shape) <= 30, "fallback canvas overflow")
    p = generate_puzzle(314159, 2, 1)
    q = generate_puzzle(314159, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ('input', 'output')), "determinism")
    # A controls the answer with absolutely identical target pixels.
    _require(not np.array_equal(p['train'][0]['output'], p['train'][1]['output']), "A ablation")
    # B's corner legend is causally necessary: swap its entries, keep reference.
    pair = p['train'][0]
    scene = _read(pair['input'])
    changed = pair['input'].copy()
    r, c = scene['key_location']
    changed[r, c], changed[r, c+1] = changed[r, c+1], changed[r, c]
    _require(not np.array_equal(solve(changed), pair['output']), "B key ablation")
    # Permuting the reference positions cannot reverse the operation.
    changed = pair['input'].copy()
    panels = [pair['input'][r+1:r+h-1, c+1:c+w-1].copy() for r, c, h, w, _ in scene['frames']]
    for (r, c, h, w, _), panel in zip(scene['frames'], panels[::-1]):
        changed[r+1:r+h-1, c+1:c+w-1] = panel
    _require(np.array_equal(solve(changed), pair['output']), "reference member order leaked")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=total, distinct_canvases=len(canvases), distinct_cuts=len(cuts),
                distinct_reference_shapes=len(ref_shapes), widths=sorted(widths), sides=sorted(sides),
                reference_phase_orders=len(cycles), cyclic_successor_orders=len(successor_orders),
                backgrounds=len(backgrounds), carrier_counts=sorted(carrier_counts),
                minimum_bbox_fill=min_fill, maximum_detour=max_detour, detour_pairs=detour_pairs)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
