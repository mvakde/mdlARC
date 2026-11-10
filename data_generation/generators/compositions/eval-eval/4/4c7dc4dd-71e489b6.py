"""Faults as invisible stamping seeds: a locally demonstrated stencil audit.

Only the small worked pair defines the marking footprint. Large-scale straight
boundaries define the target's clean scaffold. These are separate inferences.
No parent files, seed, training outputs, or construction masks are read by solve.
"""
from __future__ import annotations

import numpy as np


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int64, copy=True)


def _components(mask):
    """Four-connected components; deterministic iteration."""
    todo = set(map(tuple, np.argwhere(mask)))
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, points = [p], [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
                    points.append(q)
        yield points


def _frames(a):
    boxes = []
    for color in np.unique(a):
        for points in _components(a == color):
            pts = np.asarray(points)
            r, c = pts.min(axis=0)
            v, u = pts.max(axis=0) + 1
            h, w = int(v-r), int(u-c)
            if min(h, w) < 5 or len(points) != 2*h + 2*w - 4:
                continue
            if all(y in (r, v-1) or x in (c, u-1) for y, x in points):
                boxes.append((int(r), int(c), h, w, int(color)))
    # A stencil may itself have a rectangular ring; it is inside its true frame.
    def inside(x, y):
        return (y[0] < x[0] and y[1] < x[1]
                and x[0]+x[2] < y[0]+y[2]
                and x[1]+x[3] < y[1]+y[3])
    return sorted(x for x in boxes if not any(inside(x, y) for y in boxes))


def _crop(a, b):
    r, c, h, w, _ = b
    return a[r+1:r+h-1, c+1:c+w-1].copy()


def _scene(value):
    a = _grid(value)
    boxes = _frames(a)
    if len(boxes) != 3:
        raise ValueError("expected three separate framed panels")
    targets = [b for b in boxes if all(b[2] > x[2] and b[3] > x[3]
                                      for x in boxes if x != b)]
    if len(targets) != 1:
        raise ValueError("target must be larger than both reference panels")
    target_box = targets[0]
    refs = [b for b in boxes if b != target_box]
    if refs[0][2:] != refs[1][2:]:
        raise ValueError("reference panels must share dimensions and frame ink")
    ref_ink = refs[0][4]
    contents = [_crop(a, b) for b in refs]
    candidates = []
    for i in (0, 1):
        before, after = contents[i], contents[1-i]
        colors, counts = np.unique(before, return_counts=True)
        if len(colors) != 2 or np.count_nonzero(counts == 1) != 1:
            continue
        source_solid = int(colors[np.argmin(counts)])
        field = int(colors[np.argmax(counts)])
        center = np.argwhere(before == source_solid)[0]
        if (ref_ink in (field, source_solid)
                or set(map(int, np.unique(after))) != {field, source_solid, ref_ink}
                or not np.array_equal(after == source_solid, before == source_solid)):
            continue
        ink = after == ref_ink
        if ink.sum() < 3 or ink[0].any() or ink[-1].any() or ink[:, 0].any() or ink[:, -1].any():
            continue
        offsets = tuple(sorted(tuple(map(int, p-center)) for p in np.argwhere(ink)))
        target = _crop(a, target_box)
        target_colors = set(map(int, np.unique(target)))
        if len(target_colors) != 2 or field not in target_colors:
            continue
        solid = next(iter(target_colors - {field}))
        highlight = target_box[4]
        if highlight in target_colors:
            continue
        candidates.append(dict(target=target, field=field, solid=solid,
                               highlight=highlight, offsets=offsets, boxes=boxes,
                               target_box=target_box, reference_before=refs[i],
                               reference_after=refs[1-i], source_solid=source_solid,
                               source_ink=ref_ink, center=tuple(map(int, center))))
    if len(candidates) != 1:
        raise ValueError("reference does not identify one cue-to-stencil relation")
    return candidates[0]


def _cuts(mask, length):
    # A straight signed transition, rather than an arbitrary equal-color run.
    delta = np.diff(mask.astype(np.int8), axis=0)
    support = np.zeros(mask.shape[0]-1, dtype=bool)
    for c in range(mask.shape[1]-length+1):
        z = delta[:, c:c+length]
        support |= np.all(z == 1, axis=1) | np.all(z == -1, axis=1)
    return [0, *(np.flatnonzero(support)+1).tolist(), mask.shape[0]]


def _vote(mask, rows, cols):
    result = np.zeros_like(mask)
    for r, v in zip(rows, rows[1:]):
        for c, u in zip(cols, cols[1:]):
            block = mask[r:v, c:u]
            count = int(block.sum())
            if count*2 == block.size:
                return None
            result[r:v, c:u] = count*2 > block.size
    return result


def _corners(mask):
    p = np.pad(mask.astype(np.int16), 1)
    return int(np.abs(p[1:, 1:] - p[:-1, 1:] - p[1:, :-1] + p[:-1, :-1]).sum())


def _recover(target, solid):
    """Recover a sparse-damage rectilinear mask from both-axis edge evidence.

    Compatible partitions must agree under several simplicity penalties and
    one-cell boundary challenges. This is a conservative domain gate, not a
    claim to decode arbitrary damaged pictures.
    """
    seen = target == solid
    candidates = {}
    for k in (2, 3, 4):
        for j in (2, 3, 4):
            b = _vote(seen, _cuts(seen, k), _cuts(seen.T, j))
            if b is not None:
                candidates[b.tobytes()] = b
    if not candidates:
        raise ValueError("ambiguous rectangular votes")
    def score(b, penalty):
        return np.count_nonzero(b != seen) + penalty*_corners(b)
    ranked = sorted(candidates.values(), key=lambda b: score(b, 1.4))
    best = ranked[0]
    if not best.any() or best.all():
        raise ValueError("both scaffold roles must survive")
    for penalty in (1.1, 1.4, 1.7):
        if any(score(b, penalty) <= score(best, penalty) + .1 for b in ranked[1:]):
            raise ValueError("competing scaffold completions")
    rows = [0, *(np.flatnonzero(np.any(best[1:] != best[:-1], axis=1))+1).tolist(), best.shape[0]]
    cols = [0, *(np.flatnonzero(np.any(best[:, 1:] != best[:, :-1], axis=0))+1).tolist(), best.shape[1]]
    for axis, cuts in enumerate((rows, cols)):
        for i in range(1, len(cuts)-1):
            for delta in (-1, 1):
                moved = cuts.copy()
                moved[i] += delta
                if not moved[i-1] < moved[i] < moved[i+1]:
                    continue
                b = _vote(seen, moved if axis == 0 else rows, moved if axis == 1 else cols)
                if b is not None and not np.array_equal(b, best) and score(b, 1.4) <= score(best, 1.4)+.5:
                    raise ValueError("unsupported boundary coordinate")
    if np.count_nonzero(best != seen) > max(6, seen.size//10):
        raise ValueError("too much damage for this scaffold")
    return best


def _shift(mask, dr, dc):
    h, w = mask.shape
    result = np.zeros_like(mask)
    r0, r1 = max(0, -dr), min(h, h-dr)
    c0, c1 = max(0, -dc), min(w, w-dc)
    if r0 < r1 and c0 < c1:
        result[r0+dr:r1+dr, c0+dc:c1+dc] = mask[r0:r1, c0:c1]
    return result


def _marking(holes, offsets):
    union = np.zeros_like(holes)
    for dr, dc in offsets:
        union |= _shift(holes, dr, dc)
    return union & ~holes


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _scene(input_grid)
    target = scene['target']
    base = _recover(target, scene['solid'])
    holes = base & (target == scene['field'])
    out = target.copy()
    out[~base & (target == scene['solid'])] = scene['field']
    out[_marking(holes, scene['offsets'])] = scene['highlight']
    return out


def _span(rng, n, minimum=5):
    length = int(rng.integers(minimum, n+1))
    start = int(rng.integers(n-length+1))
    return start, start+length


def _base(rng, shape, kind):
    h, w = shape
    b = np.zeros(shape, dtype=bool)
    if kind == 'rectangle':
        r, v = _span(rng, h, 6)
        c, u = _span(rng, w, 6)
        b[r:v, c:u] = True
    elif kind == 'bands':
        row = 0
        phase = bool(rng.integers(2))
        while row < h:
            thick = int(rng.integers(3, 7))
            b[row:min(h, row+thick)] = phase
            phase = not phase
            row += thick
    elif kind in ('L', 'cross', 'arms'):
        r, v = _span(rng, h, 10)
        c, u = _span(rng, w, 10)
        tw = int(rng.integers(3, max(4, (u-c)//2)))
        th = int(rng.integers(3, max(4, (v-r)//2)))
        left = c if kind == 'L' else int(rng.integers(c, u-tw+1))
        top = r if kind in ('L', 'arms') else int(rng.integers(r, v-th+1))
        b[r:v, left:left+tw] = True
        b[top:top+th, c:u] = True
        if kind == 'arms':
            top2 = v-th  # two separated arms, rather than an accidental cross
            lo = int(rng.integers(c, left+1))
            b[top2:top2+th, lo:u] = True
    elif kind == 'nested':
        r, v = _span(rng, h, 13)
        c, u = _span(rng, w, 13)
        b[r:v, c:u] = True
        rims = rng.integers(3, 5, size=4)
        y, z = r+int(rims[0]), v-int(rims[1])
        x, t = c+int(rims[2]), u-int(rims[3])
        b[y:z, x:t] = False
        if min(z-y, t-x) >= 7 and rng.random() < .5:
            b[y+2:z-2, x+2:t-2] = True
    elif kind == 'separate':
        cut = int(rng.integers(5, w-6))
        r, v = _span(rng, h, 5)
        b[r:v, :cut] = True
        r, v = _span(rng, h, 5)
        b[r:v, cut+2:] = True
        if h >= 16 and cut >= 6 and rng.random() < .5:
            b[h//2-1:h//2+1, :cut] = False
    else:
        raise ValueError("unknown construction geometry")
    if rng.integers(2):
        b = b[::-1].copy()
    if rng.integers(2):
        b = b[:, ::-1].copy()
    return b


def _stencil(rng, max_radius=3):
    radius = int(rng.integers(1, max_radius+1))
    mode = int(rng.integers(5))
    points = set()
    if mode == 0:  # independently sized cardinal rays
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for k in range(1, int(rng.integers(1, radius+1))+1):
                points.add((dr*k, dc*k))
        if rng.random() < .5:
            points.add((radius, radius))
    elif mode == 1:  # hollow Chebyshev or Manhattan shells
        diamond = bool(rng.integers(2))
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                distance = abs(dr)+abs(dc) if diamond else max(abs(dr), abs(dc))
                if distance == radius:
                    points.add((dr, dc))
    elif mode == 2:  # branching diagonal rays, with optional orthogonal bridge
        for k in range(1, radius+1):
            points.update(((k, k), (-k, -k), (k, -k), (-k, k)))
        if rng.integers(2):
            points.update(((0, -1), (0, 1)))
    elif mode == 3:  # a directional bracket; its orientation is demonstrated
        points.update((-radius, x) for x in range(-radius, radius+1))
        points.update((y, -radius) for y in range(-radius, radius+1))
        if rng.integers(2):
            points.add((0, 1))
    else:  # a procedural polyomino footprint, rather than a fixed glyph list
        p = (0, 1)
        for _ in range(int(rng.integers(8, 30))):
            if p != (0, 0):
                points.add(p)
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            p = (int(np.clip(p[0]+dr, -radius, radius)),
                 int(np.clip(p[1]+dc, -radius, radius)))
        points.update(((0, 1), (1, 0), (-1, 0)))
    transform = int(rng.integers(4))
    if transform & 1:
        points = {(c, r) for r, c in points}
    if transform & 2:
        points = {(-r, c) for r, c in points}
    return tuple(sorted(points - {(0, 0)}))


def _construct_output(base, holes, offsets, field, solid, ink):
    """Labels from explicit geometry, using point painting rather than shifts."""
    out = np.where(base, solid, field).astype(np.int64)
    out[holes] = field
    h, w = out.shape
    for r, c in np.argwhere(holes):
        for dr, dc in offsets:
            y, x = int(r)+dr, int(c)+dc
            if 0 <= y < h and 0 <= x < w and not holes[y, x]:
                out[y, x] = ink
    return out


def _defects(rng, base, anchor=False):
    holes = np.zeros_like(base)
    extras = np.zeros_like(base)
    if anchor:
        # Large, edge-touching rectangle: a clipped seed and a small L cluster
        # plus separated seeds. Anchor dimensions give room for intact borders.
        pts = np.argwhere(base)
        r0, c0 = pts.min(axis=0)
        v, u = pts.max(axis=0)+1
        edge = [(r, c) for r, c in map(tuple, pts)
                if r in (0, base.shape[0]-1) or c in (0, base.shape[1]-1)]
        if not edge:
            return None
        holes[edge[int(rng.integers(len(edge)))]] = True
        r = int(rng.integers(r0+2, v-3))
        c = int(rng.integers(c0+2, u-3))
        holes[r, c] = holes[r+1, c] = holes[r+1, c+1] = True
        # Distance-two seeds explicitly teach union rather than hole filling.
        for y, x in ((r-2, c), (r, c+2)):
            if base[y, x]:
                holes[y, x] = True
        # Teach crossing the solid/field boundary and ink precedence on an
        # adjacent extra, independently of the canvas-edge clipping seed.
        boundary_col = int(c0) if c0 > 0 else int(u)-1
        boundary_row = int(rng.integers(r0+1, v-1))
        holes[boundary_row, boundary_col] = True
        extras[boundary_row, boundary_col-1 if c0 > 0 else boundary_col+1] = True
    else:
        budget = int(rng.integers(1, min(10, max(2, int(base.sum())//18))+1))
        points = np.argwhere(base)
        chosen = rng.choice(len(points), budget, replace=False)
        for p in points[chosen]:
            holes[tuple(p)] = True
        if rng.random() < .45:
            r, c = points[int(rng.integers(len(points)))]
            for dy, dx in ((0, 0), (1, 0), (1, 1)):
                y, x = int(r)+dy, int(c)+dx
                if 0 <= y < base.shape[0] and 0 <= x < base.shape[1] and base[y, x]:
                    holes[y, x] = True
    pts = np.argwhere(~base)
    if len(pts) < 6:
        return None
    count = int(rng.integers(2 if anchor else 0, min(8, max(3, len(pts)//15))+1))
    for p in pts[rng.choice(len(pts), count, replace=False)]:
        extras[tuple(p)] = True
    if anchor:
        far = [tuple(p) for p in pts if all(max(abs(int(p[0])-int(q[0])),
                                               abs(int(p[1])-int(q[1]))) > 3
                                           for q in np.argwhere(holes))]
        if not far:
            return None
        extras[far[int(rng.integers(len(far)))]] = True
    return holes, extras


def _target(rng, shape, kind, palette, anchor=False):
    field, solid = palette[1], palette[4]
    for _ in range(150):
        if anchor:
            h, w = shape
            base = np.zeros(shape, dtype=bool)
            # A field strip at least four cells wide leaves visible far extras.
            base[:int(rng.integers(h-3, h+1)), :int(rng.integers(w-6, w-3))] = True
            if rng.integers(2):
                base = base[::-1].copy()
            if rng.integers(2):
                base = base[:, ::-1].copy()
        else:
            base = _base(rng, shape, kind)
        if min(int(base.sum()), int((~base).sum())) < 20:
            continue
        damage = _defects(rng, base, anchor)
        if damage is None:
            continue
        holes, extras = damage
        target = np.where(base, solid, field).astype(np.int64)
        target[holes], target[extras] = field, solid
        try:
            recovered = _recover(target, solid)
        except ValueError:
            continue
        if np.array_equal(base, recovered):
            return target, base, holes, extras
    raise RuntimeError("could not construct a readable sparse-damage scaffold")


def _place_panels(rng, before, after, target, palette):
    ref_ink, target_ink = palette[3], palette[5]
    rh, rw = np.array(before.shape)+2
    th, tw = np.array(target.shape)+2
    # The two reference frames are grouped by equal size and ink. No slot order
    # identifies before, after, or the target, even after a whole-scene symmetry.
    gap = int(rng.integers(1, min(2, 30-int(rh+th))+1))
    row_w = max(int(2*rw+gap), int(tw))
    min_h = int(rh+th+gap)
    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(row_w, 31))
    grid = np.full((h, w), palette[0], dtype=np.int64)
    top_ref = bool(rng.integers(2))
    slack = h-min_h
    before_gap = int(rng.integers(slack+1))
    between = int(rng.integers(slack-before_gap+1))
    y1 = before_gap
    y2 = y1 + (int(rh) if top_ref else int(th)) + gap + between
    ref_y, target_y = (y1, y2) if top_ref else (y2, y1)
    ref_x = int(rng.integers(w-(2*rw+gap)+1))
    target_x = int(rng.integers(w-tw+1))
    references = [before, after]
    if rng.integers(2):
        references.reverse()
    positions = [(ref_y, ref_x, references[0], ref_ink),
                 (ref_y, ref_x+int(rw)+gap, references[1], ref_ink),
                 (target_y, target_x, target, target_ink)]
    for r, c, interior, ink in positions:
        ph, pw = np.array(interior.shape)+2
        grid[r:r+ph, c:c+pw] = ink
        grid[r+1:r+ph-1, c+1:c+pw-1] = interior
    return grid


def _assemble(rng, target_data, offsets, palette, second_offsets=None):
    target, base, holes, _ = target_data
    pts = np.asarray([(0, 0), *offsets, *(second_offsets or ())])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    minimum = int(max(hi-lo)+3)  # one-cell interior moat on either side
    side = max(5, minimum)
    if side % 2 == 0:
        side += 1
    side = min(9, side)
    if rng.random() < .35 and side <= 7:
        side += 2
    center = [int(rng.integers(1-lo[i], side-1-hi[i])) for i in (0, 1)]
    before = np.full((side, side), palette[1], dtype=np.int64)
    before[tuple(center)] = palette[2]
    after = before.copy()
    for dr, dc in offsets:
        after[center[0]+dr, center[1]+dc] = palette[3]
    grid = _place_panels(rng, before, after, target, palette)
    out = _construct_output(base, holes, offsets, palette[1], palette[4], palette[5])
    companion = None
    if second_offsets is not None:
        # A literal reference intervention: change only the completed motif,
        # leaving target pixels, cue, frames, colors and placement identical.
        second_after = before.copy()
        for dr, dc in second_offsets:
            second_after[center[0]+dr, center[1]+dc] = palette[3]
        companion = grid.copy()
        r, c, ph, pw, _ = _scene(grid)['reference_after']
        companion[r+1:r+ph-1, c+1:c+pw-1] = second_after
        companion_out = _construct_output(base, holes, second_offsets,
                                          palette[1], palette[4], palette[5])
    # All four rotations and reflection preserve the displacement semantics.
    k = int(rng.integers(4))
    reflected = bool(rng.integers(2))
    if reflected:
        grid, out = grid[:, ::-1], out[:, ::-1]
        if companion is not None:
            companion, companion_out = companion[:, ::-1], companion_out[:, ::-1]
    grid, out = np.rot90(grid, k).copy(), np.rot90(out, k).copy()
    pair = {'input': grid, 'output': out}
    if companion is not None:
        return pair, {'input': np.rot90(companion, k).copy(),
                      'output': np.rot90(companion_out, k).copy()}
    return pair


def _palette(rng):
    p = list(map(int, rng.permutation(10)[:6]))
    if rng.random() < .5:
        p[2] = p[4]  # visible source-solid to target-solid role correspondence
    if rng.random() < .5:
        p[3] = p[5]  # shared ink, or fully remapped ink by target-frame swatch
    return p


def _counts(num_train, num_test):
    for name, count, low, high in (('num_train', num_train, 2, 10),
                                   ('num_test', num_test, 1, 4)):
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or not low <= int(count) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    stable_palette = _palette(rng)
    vary_palette = bool(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    seen = set()
    # Anchors share the same actual damaged target, not merely its silhouette.
    # Geometry is regenerated only if ink choices fail to visibly distinguish it.
    shape = (int(rng.integers(14, 17)), int(rng.integers(14, 23)))
    anchor = _target(rng, shape, 'rectangle', stable_palette, anchor=True)
    # Familiar Moore adjacency plus a locally displayed extension teaches
    # clipping, merged stamps and global hole exclusion even at count (2,1).
    first_offsets = tuple(sorted(set(_stencil(rng)) | set(_MOORE)))
    first_canonical = _construct_output(anchor[1], anchor[2], first_offsets,
                                         stable_palette[1], stable_palette[4], stable_palette[5])
    for _ in range(80):
        second_offsets = _stencil(rng)
        second_canonical = _construct_output(anchor[1], anchor[2], second_offsets,
                                              stable_palette[1], stable_palette[4], stable_palette[5])
        if not np.array_equal(first_canonical, second_canonical):
            first, second = _assemble(rng, anchor, first_offsets, stable_palette,
                                      second_offsets=second_offsets)
            break
    else:
        raise RuntimeError("could not distinguish two reference footprints")
    puzzle['train'] = [first, second]
    seen.update((_key(first['input']), _key(second['input'])))
    for split, count in (('train', num_train-2), ('test', num_test)):
        for _ in range(count):
            for attempt in range(100):
                p = _palette(rng) if vary_palette else stable_palette
                offsets = _stencil(rng)
                # Reference height <= 11 leaves room for a target + framing.
                # Test composites are larger than the two training anchors.
                shape = (int(rng.integers(14, 17)), int(rng.integers(14, 25)))
                kind = str(rng.choice(('rectangle', 'bands', 'L', 'cross') if split == 'train'
                                      else ('nested', 'separate', 'arms')))
                try:
                    data = _target(rng, shape, kind, p)
                    if not _marking(data[2], offsets).any():
                        continue  # every pair visibly demonstrates a stencil
                    pair = _assemble(rng, data, offsets, p)
                    if _key(pair['input']) in seen:
                        continue
                    scene = _scene(pair['input'])
                    # The two anchors already demonstrate clipping, merged
                    # stamps, and cluster-center exclusion. Far specks are
                    # independently required somewhere in training below.
                    if not np.array_equal(solve(pair['input']), pair['output']):
                        continue
                except (ValueError, RuntimeError):
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not make a distinct compatible example")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(scene, base):
    holes = base & (scene['target'] == scene['field'])
    extras = ~base & (scene['target'] == scene['solid'])
    marking = _marking(holes, scene['offsets'])
    covered_holes = np.zeros_like(holes)
    load = np.zeros_like(holes, dtype=np.int16)
    clipped = False
    for r, c in np.argwhere(holes):
        for dr, dc in scene['offsets']:
            y, x = int(r)+dr, int(c)+dc
            if 0 <= y < holes.shape[0] and 0 <= x < holes.shape[1]:
                load[y, x] += 1
                covered_holes[y, x] |= holes[y, x]
            else:
                clipped = True
    return dict(holes=holes, extras=extras, marking=marking, clipped=clipped,
                overlap=bool(np.any((load >= 2) & ~holes)),
                excluded=bool(covered_holes.any()),
                ink_on_extra=bool(np.any(extras & marking)),
                far_extras=bool(np.any(extras & ~marking)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_features, test_features, stencil_keys = set(), [], [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            scene = _scene(inp)
            base = _recover(scene['target'], scene['solid'])
            f = _features(scene, base)
            assert out.shape == scene['target'].shape
            assert f['holes'].any() and f['marking'].any()
            assert np.array_equal(solve(inp), out), 'wrong input-only prediction'
            assert np.all(out[f['holes']] == scene['field']), 'all hole centers stay empty'
            assert np.all(out[f['marking']] == scene['highlight'])
            assert np.all(out[f['extras'] & ~f['marking']] == scene['field'])
            unchanged = ~f['extras'] & ~f['marking']
            assert np.array_equal(out[unchanged], scene['target'][unchanged])
            # Omitting B gives a clean scaffold; omitting A uses a fixed Moore
            # halo. The former must fail everywhere, the latter in training.
            assert not np.array_equal(out, np.where(base, scene['solid'], scene['field']))
            if split == 'train':
                train_features.append(f)
                stencil_keys.add(scene['offsets'])
            else:
                test_features.append(f)
    assert len(stencil_keys) >= 2, 'training must show a locally variable stencil'
    assert any(f['far_extras'] for f in train_features), 'teach speck erasure visibly'
    # At least one input must disprove a target-only fixed-neighborhood rule.
    assert any(not np.array_equal(_marking(f['holes'], _MOORE), f['marking'])
               for f in train_features), 'reference can be bypassed by Moore halos'
    for effect in ('clipped', 'overlap', 'excluded', 'ink_on_extra'):
        assert not any(f[effect] for f in test_features) or any(f[effect] for f in train_features)
    # Verify the actual mandatory interventions from returned arrays alone.
    p, q = puzzle['train'][:2]
    s, t = _scene(p['input']), _scene(q['input'])
    assert np.array_equal(s['target'], t['target'])
    assert s['boxes'] == t['boxes'] and s['center'] == t['center']
    before = s['reference_before']
    assert np.array_equal(_crop(np.asarray(p['input']), before), _crop(np.asarray(q['input']), before))
    allowed = np.zeros(np.asarray(p['input']).shape, dtype=bool)
    r, c, h, w, _ = s['reference_after']
    allowed[r+1:r+h-1, c+1:c+w-1] = True
    assert np.array_equal(np.asarray(p['input'])[~allowed], np.asarray(q['input'])[~allowed])
    assert not np.array_equal(p['output'], q['output']), 'reference intervention must change the answer'


_MOORE = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def self_test():
    """Bounded diversity audit plus targeted morphology/analogy regressions."""
    shapes, masks, kernels, palettes, hole_counts, extra_counts = [set() for _ in range(6)]
    flags = {k: False for k in ('clipped', 'overlap', 'excluded', 'far_extras')}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed < 3:
            replay = generate_puzzle(seed)
            assert all(_key(p[k]) == _key(q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], replay[s]) for k in p)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene = _scene(pair['input'])
                base = _recover(scene['target'], scene['solid'])
                f = _features(scene, base)
                shapes.add(scene['target'].shape)
                masks.add(_key(base))
                kernels.add(scene['offsets'])
                palettes.add((scene['field'], scene['solid'], scene['highlight']))
                hole_counts.add(int(f['holes'].sum()))
                extra_counts.add(int(f['extras'].sum()))
                for k in flags:
                    flags[k] |= f[k]
                if seed < 2:
                    for rotation in range(4):
                        assert np.array_equal(solve(np.rot90(pair['input'], rotation)),
                                              np.rot90(pair['output'], rotation))
                    assert np.array_equal(solve(pair['input'].T), pair['output'].T)
    assert len(shapes) >= 30 and len(masks) >= 150
    assert len(kernels) >= 50 and len(palettes) >= 30
    assert len(hole_counts) >= 7 and len(extra_counts) >= 5
    assert all(flags.values())
    # Explicit twin test: target bytes and palette stay fixed, only stencil
    # contents and their reference placement differ; no stored labels are read.
    rng = np.random.default_rng(314159)
    p = [0, 1, 2, 3, 4, 5]
    data = _target(rng, (15, 17), 'rectangle', p, anchor=True)
    left = ((-1, 0), (0, -1), (1, 0), (0, -2))
    right = ((-1, 0), (0, 1), (1, 0), (0, 2))
    twins = _assemble(rng, data, left, p, second_offsets=right)
    answers = []
    for pair in twins:
        assert np.array_equal(solve(pair['input']), pair['output'])
        answers.append(pair['output'])
    assert np.array_equal(_scene(twins[0]['input'])['target'], _scene(twins[1]['input'])['target'])
    assert not np.array_equal(*answers)
    # Neighboring seeds remain empty even when another seed stamps on them;
    # extra specks never seed stamps; clipping is literal, with no wrapping.
    b = np.zeros((7, 8), dtype=bool)
    b[:5, :5] = True
    holes = np.zeros_like(b)
    holes[0, 0] = holes[2, 2] = holes[2, 3] = True
    z = _construct_output(b, holes, ((0, 1), (0, 2), (-1, 0)), 0, 1, 7)
    assert z[2, 2] == z[2, 3] == z[0, 0] == 0
    assert z[2, 4] == z[2, 5] == 7 and z[-1, 0] == 0
    # Boundary count checks here; verify.py separately checks every legal pair.
    # Avoid duplicating its full count matrix inside a bounded semantic audit.
    for a, b in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(8000+a*10+b, a, b)
        assert len(puzzle['train']) == a and len(puzzle['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.2), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(73, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a damaged answer')
    return dict(seeds=42, target_shapes=len(shapes), scaffold_masks=len(masks),
                stencils=len(kernels), palettes=len(palettes),
                hole_counts=sorted(hole_counts), extra_counts=sorted(extra_counts), **flags)
