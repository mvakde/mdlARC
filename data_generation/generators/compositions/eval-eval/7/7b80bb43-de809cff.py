"""Sealed defects: reconstruct barriers, then mark contained material erasures.

The visible conventions are fixed: 0 is exterior/erasure, 5 is wall, and 8 is
the output highlight. The other two input colors are interchangeable materials.
No parent generator, files, examples, seed, or state are consulted by solve.
"""

from collections import deque
import numpy as np


_MATERIALS = (1, 2, 3, 4, 6, 7, 9)
_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_HOST_SEEDS = (0, 303119165, 670195693, 109060745, 23339527, 46451979,
               935586551, 314148744, 920693483, 723786253, 484566638,
               263009042, 533544412, 11615167, 669387773, 948301322,
               275901039, 912628844, 877333722, 555830431, 427652973,
               854221142, 220879665, 624330465, 12984749, 893044115,
               334259381, 658652954, 420820991, 780021632, 690354879,
               6772507, 798022786, 809668266, 323912792, 232046879,
               898787333, 896659474, 419421852, 854648941, 412552038,
               746062509)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or any(n < 1 or n > 30 for n in grid.shape)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return grid.astype(np.int8, copy=True)


def _at(mask, r, c):
    return (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]
            and bool(mask[r, c]))


def _axis_support(mask):
    """Cells in a contiguous horizontal or vertical run of at least three."""
    supported = np.zeros_like(mask)
    for m, s in ((mask, supported), (mask.T, supported.T)):
        for offset in range(max(0, m.shape[1] - 2)):
            rows = m[:, offset:offset + 3].all(axis=1)
            s[rows, offset:offset + 3] = True
    return supported


def _repair_evidence(mask):
    support = _axis_support(mask)
    repairs = []
    for dr, dc in ((0, 1), (1, 0)):
        for rr, cc in np.argwhere(mask):
            r, c = int(rr), int(cc)
            if not _at(mask, r + 4*dr, c + 4*dc):
                continue
            gap = tuple((r + k*dr, c + k*dc) for k in (1, 2, 3))
            if any(mask[p] for p in gap):
                continue
            if not (_at(mask, r - dr, c - dc)
                    or _at(mask, r + 5*dr, c + 5*dc)):
                continue
            cues = []
            for endpoint in (0, 4):
                ar, ac = r + endpoint*dr, c + endpoint*dc
                sign = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    vr, vc = sign*dr + side*dc, sign*dc + side*dr
                    spur = tuple((ar + k*vr, ac + k*vc) for k in (1, 2))
                    if (all(_at(mask, *p) and not support[p] for p in spur)
                            and not _at(mask, ar + 3*vr, ac + 3*vc)):
                        cues.append(spur)
            if cues:
                if len(cues) != 1:
                    raise ValueError("Ambiguous wall deviation")
                repairs.append((gap, cues[0], 'h' if dc else 'v'))
    footprints = [set(gap + spur) for gap, spur, _ in repairs]
    if any(a & b for i, a in enumerate(footprints) for b in footprints[:i]):
        raise ValueError("Overlapping repair evidence")
    return repairs


def _restore_walls(mask, repair=True):
    evidence = _repair_evidence(mask)
    completed = mask.copy()
    for gap, _, _ in evidence:
        for p in gap:
            completed[p] = True
    # Diagonal-only pairs/triples are cues or clutter. Straight strokes survive.
    weak = mask & ~_axis_support(completed)
    discard = np.zeros_like(mask)
    for r, c in np.argwhere(weak):
        r, c = int(r), int(c)
        if any(_at(weak, r + dr, c + dc)
               for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))):
            discard[r, c] = True
    walls = mask & ~discard
    if repair:
        for gap, _, _ in evidence:
            for p in gap:
                walls[p] = True
    return walls, evidence


def _regions(walls):
    """4-connected non-wall components; closed means no canvas-edge cell."""
    h, w = walls.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    closed = []
    for sr in range(h):
        for sc in range(w):
            if walls[sr, sc] or labels[sr, sc] >= 0:
                continue
            label = len(closed)
            labels[sr, sc] = label
            queue = deque([(sr, sc)])
            enclosed = True
            while queue:
                r, c = queue.popleft()
                if r in (0, h - 1) or c in (0, w - 1):
                    enclosed = False
                for dr, dc in _STEPS:
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < h and 0 <= cc < w and not walls[rr, cc]
                            and labels[rr, cc] < 0):
                        labels[rr, cc] = label
                        queue.append((rr, cc))
            closed.append(enclosed)
    contained = np.zeros_like(walls)
    for i, flag in enumerate(closed):
        if flag:
            contained |= labels == i
    return labels, contained, sum(closed)


def _unsupported_intervals(walls):
    """Diagnostic candidates: bracketed gaps without checking a diagonal cue."""
    intervals = []
    for dr, dc in ((0, 1), (1, 0)):
        for rr, cc in np.argwhere(walls):
            r, c = int(rr), int(cc)
            if not _at(walls, r + 4*dr, c + 4*dc):
                continue
            gap = [(r + k*dr, c + k*dc) for k in (1, 2, 3)]
            if any(walls[p] for p in gap):
                continue
            if _at(walls, r - dr, c - dc) or _at(walls, r + 5*dr, c + 5*dc):
                intervals.append(tuple(gap))
    return intervals


def _opening_witness(walls, holes, centers):
    """Find an unsupported opening whose closure would contain an open hole."""
    unselected = set(holes) - set(centers)
    for gap in _unsupported_intervals(walls):
        trial = walls.copy()
        for p in gap:
            trial[p] = True
        contained = _regions(trial)[1]
        newly_contained = tuple(sorted(p for p in unselected if contained[p]))
        if newly_contained:
            return gap, newly_contained
    return None


def _bbox(mask):
    cells = np.argwhere(mask)
    if not len(cells):
        raise ValueError("Missing material")
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _paint(shape, boxes, colors, order):
    scene = np.zeros(shape, dtype=np.int8)
    for i in order:
        r0, c0, r1, c1 = boxes[i]
        scene[r0:r1, c0:c1] = colors[i]
    return scene


def _recover_material(grid):
    """Exposed extrema locate rectangles; intact overlap pixels decide order.

    Family constraints retain observed extrema of each material. Wrong-fill
    pixels stay inside BOTH bounds, so they cannot fabricate an extreme.
    Walls and zeros are missing evidence for fitting, not material votes.
    """
    colors = tuple(int(c) for c in np.unique(grid) if c not in (0, 5))
    if len(colors) != 2 or any(c not in _MATERIALS for c in colors):
        raise ValueError("Expected exactly two material colors, zero, and walls")
    boxes = tuple(_bbox(grid == color) for color in colors)
    if any(min(r1 - r0, c1 - c0) < 5 for r0, c0, r1, c1 in boxes):
        raise ValueError("Material rectangles are too small")
    for color, (r0, c0, r1, c1) in zip(colors, boxes):
        mask = grid == color
        sides = (mask[r0, c0:c1], mask[r1 - 1, c0:c1],
                 mask[r0:r1, c0], mask[r0:r1, c1 - 1])
        if min(int(side.sum()) for side in sides) < 3:
            raise ValueError("Insufficient exposed rectangle extrema")
    a, b = boxes
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    if overlap < 9:
        raise ValueError("Material rectangles must overlap")
    observed = np.isin(grid, colors)
    candidates = []
    for order in ((0, 1), (1, 0)):
        scene = _paint(grid.shape, boxes, colors, order)
        cost = int(np.count_nonzero(observed & (grid != scene)))
        candidates.append((cost, scene, order))
    candidates.sort(key=lambda x: x[0])
    if candidates[1][0] - candidates[0][0] < 3:
        raise ValueError("Insufficient evidence for material covering order")
    loss, scene, order = candidates[0]
    if loss > max(6, int(.04 * grid.size)):
        raise ValueError("Excessive material corruption")
    return scene, colors, boxes, order


def _ring(shape, center):
    r, c = center
    return [(rr, cc) for rr in range(max(0, r - 1), min(shape[0], r + 2))
            for cc in range(max(0, c - 1), min(shape[1], c + 2))
            if (rr, cc) != center]


def _compose(scene, walls, holes, colors):
    labels, contained, _ = _regions(walls)
    centers = [p for p in holes if contained[p]]
    center_set = set(centers)
    result = scene.copy()
    result[walls] = 5
    demands = {}
    for p in centers:
        other = colors[1] if scene[p] == colors[0] else colors[0]
        for q in _ring(scene.shape, p):
            if labels[q] != labels[p] or q in center_set:
                continue
            if q in demands and demands[q] != other:
                raise ValueError("Conflicting complementary rings")
            demands[q] = other
    for p, color in demands.items():
        result[p] = color
    for p in centers:
        result[p] = 8
    return result, centers, demands


def _analyze(input_grid):
    grid = _grid(input_grid)
    if not np.any(grid == 0) or not np.any(grid == 5):
        raise ValueError("Missing background or wall role")
    walls, repairs = _restore_walls(grid == 5)
    scene, colors, boxes, order = _recover_material(grid)
    holes = [tuple(map(int, p)) for p in np.argwhere(
        (grid == 0) & (scene != 0) & ~walls)]
    result, centers, demands = _compose(scene, walls, holes, colors)
    return {'grid': grid, 'walls': walls, 'repairs': repairs, 'scene': scene,
            'colors': colors, 'boxes': boxes, 'order': order, 'holes': holes,
            'centers': centers, 'demands': demands, 'output': result}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell from the input and the fixed visible conventions."""
    return _analyze(input_grid)['output']


def _box_wall(mask, box):
    r0, c0, r1, c1 = box  # inclusive wall coordinates
    mask[r0, c0:c1 + 1] = mask[r1, c0:c1 + 1] = True
    mask[r0:r1 + 1, c0] = mask[r0:r1 + 1, c1] = True


def _wall_network(rng, shape, mode):
    h, w = shape
    walls = np.zeros(shape, dtype=bool)
    if mode == 'offset':
        # Unequal rooms share only part of a side: many non-rectangular outlines.
        split = int(rng.integers(8, w - 7))
        left, right = int(rng.integers(0, split - 6)), int(rng.integers(split + 7, w))
        heights = [int(rng.integers(8, h)) for _ in range(2)]
        starts = [int(rng.integers(0, h - n)) for n in heights]
        _box_wall(walls, (starts[0], left, starts[0] + heights[0], split))
        _box_wall(walls, (starts[1], split, starts[1] + heights[1], right))
        if min(starts[0] + heights[0], starts[1] + heights[1]) - max(starts) < 4:
            return None
    elif mode == 'nested':
        # An open outer enclosure around a separately sealed inner pocket.
        if min(shape) < 23:
            return None
        r0, c0 = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        r1, c1 = h - 1 - int(rng.integers(0, 3)), w - 1 - int(rng.integers(0, 3))
        if min(r1 - r0, c1 - c0) < 17:
            return None
        _box_wall(walls, (r0, c0, r1, c1))
        ir0, ic0 = r0 + int(rng.integers(4, 6)), c0 + int(rng.integers(4, 6))
        ir1, ic1 = r1 - int(rng.integers(4, 6)), c1 - int(rng.integers(4, 6))
        if min(ir1 - ir0, ic1 - ic0) < 7:
            return None
        _box_wall(walls, (ir0, ic0, ir1, ic1))
    elif mode == 'disjoint':
        if w < 24:
            return None
        cut = int(rng.integers(10, w - 11))
        for c0, c1 in ((int(rng.integers(0, 2)), cut),
                       (cut + 3, w - 1 - int(rng.integers(0, 2)))):
            r0 = int(rng.integers(0, h - 9))
            r1 = int(rng.integers(r0 + 8, h))
            _box_wall(walls, (r0, c0, r1, c1))
    else:
        r0, c0 = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        r1, c1 = h - 1 - int(rng.integers(0, 4)), w - 1 - int(rng.integers(0, 4))
        if min(r1 - r0, c1 - c0) < 11:
            return None
        _box_wall(walls, (r0, c0, r1, c1))
        split = int(rng.integers(c0 + 4, c1 - 3))
        walls[r0:r1 + 1, split] = True
        if mode in ('tee', 'cross'):
            level = int(rng.integers(r0 + 4, r1 - 3))
            if mode == 'cross':
                walls[level, c0:c1 + 1] = True
            elif rng.integers(2):
                walls[level, c0:split + 1] = True
            else:
                walls[level, split:c1 + 1] = True
    if rng.integers(2):
        walls = walls[::-1].copy()
    if rng.integers(2):
        walls = walls[:, ::-1].copy()
    return walls


def _straight_gaps(walls):
    candidates = []
    for dr, dc in ((0, 1), (1, 0)):
        for rr, cc in np.argwhere(walls):
            r, c = int(rr), int(cc)
            # Two intact cells at each side, and no perpendicular junction
            # in the missing interval. This is a sampling preference only.
            if not all(_at(walls, r + k*dr, c + k*dc) for k in range(7)):
                continue
            gap = tuple((r + k*dr, c + k*dc) for k in (2, 3, 4))
            if any(_at(walls, rr + dc, cc + dr) or _at(walls, rr - dc, cc - dr)
                   for rr, cc in gap):
                continue
            candidates.append((gap, (r + dr, c + dc),
                               (r + 5*dr, c + 5*dc), dr, dc))
    return candidates


def _open_and_damage(rng, walls, rich):
    original_count = _regions(walls)[2]
    if original_count < 2:
        return None
    gaps = _straight_gaps(walls)
    rng.shuffle(gaps)
    clean = None
    for gap, _, _, _, _ in gaps[:50]:
        trial = walls.copy()
        for p in gap:
            trial[p] = False
        if _regions(trial)[2] == original_count - 1:
            try:
                restored, repairs = _restore_walls(trial)
                if not repairs and np.array_equal(restored, trial):
                    clean = trial
                    break
            except ValueError:
                pass
    if clean is None:
        return None
    contained = _regions(clean)[1]
    leaked = _regions(walls)[1] & ~contained
    inp = clean.copy()
    gaps = _straight_gaps(clean)
    rng.shuffle(gaps)
    repairs = 0
    target = int(rng.integers(1, 4 if rich else 3))
    for gap, a, b, dr, dc in gaps:
        if not all(inp[p] for p in gap):
            continue
        removal = inp.copy()
        for p in gap:
            removal[p] = False
        if repairs == 0 and not (contained & ~_regions(removal)[1]).any():
            continue
        options = [(a, 1, -1), (a, 1, 1), (b, -1, -1), (b, -1, 1)]
        rng.shuffle(options)
        for anchor, sign, side in options:
            ar, ac = anchor
            vr, vc = sign*dr + side*dc, sign*dc + side*dr
            spur = [(ar + k*vr, ac + k*vc) for k in (1, 2)]
            if not all(0 <= r < clean.shape[0] and 0 <= c < clean.shape[1]
                       and not clean[r, c] and not inp[r, c] for r, c in spur):
                continue
            trial = removal.copy()
            for p in spur:
                trial[p] = True
            try:
                restored, cues = _restore_walls(trial)
                if np.array_equal(restored, clean) and len(cues) == repairs + 1:
                    inp = trial
                    repairs += 1
                    break
            except ValueError:
                pass
        if repairs >= target:
            break
    if not repairs:
        return None
    # Free diagonal decoys do not open/close compartments after cleanup.
    for _ in range(int(rng.integers(4))):
        for _ in range(15):
            r, c = int(rng.integers(inp.shape[0])), int(rng.integers(inp.shape[1]))
            dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
            points = [(r + k*dr, c + k*dc) for k in range(int(rng.choice([2, 3])))]
            if not all(0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]
                       and not inp[rr, cc] and not clean[rr, cc] for rr, cc in points):
                continue
            trial = inp.copy()
            for p in points:
                trial[p] = True
            try:
                restored, cues = _restore_walls(trial)
                if np.array_equal(restored, clean) and len(cues) == repairs:
                    inp = trial
                    break
            except ValueError:
                pass
    unrepaired, _ = _restore_walls(inp, repair=False)
    critical = contained & ~_regions(unrepaired)[1]
    if not critical.any():
        return None
    return clean, inp, contained, critical, leaked


def _sample_material(rng, shape, wall_input, clean_walls, colors):
    h, w = shape
    for _ in range(45):
        boxes = []
        for _ in range(2):
            rh = int(rng.integers(max(7, h//2), h + 1))
            rw = int(rng.integers(max(7, w//2), w + 1))
            r0, c0 = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
            boxes.append((r0, c0, r0 + rh, c0 + rw))
        order = tuple(map(int, rng.permutation(2)))
        scene = _paint(shape, boxes, colors, order)
        if any(not np.any(scene == color) for color in colors):
            continue
        if any(_bbox(scene == colors[i]) != boxes[i] for i in range(2)):
            continue
        visible = scene.copy()
        visible[wall_input] = 5
        try:
            recovered, _, _, _ = _recover_material(visible)
            if np.array_equal(recovered, scene) and np.count_nonzero(scene == 0) >= 8:
                return scene, tuple(boxes), order
        except ValueError:
            pass
    return None


def _select_holes(rng, scene, walls, wall_input, contained, critical, leaked, colors, teach, rich):
    eligible = (scene != 0) & ~walls & ~wall_input
    pools = {'critical': np.argwhere(eligible & critical),
             'closed': np.argwhere(eligible & contained),
             'open': np.argwhere(eligible & leaked)}
    if not len(pools['critical']) or not len(pools['open']):
        return None
    holes = []
    def add(p):
        p = tuple(map(int, p))
        if p in holes or not eligible[p]:
            return False
        if contained[p] and any(contained[q] and scene[p] != scene[q]
                                and max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 2
                                for q in holes):
            return False
        holes.append(p)
        return True
    def pick(pool, condition=lambda p: True):
        for i in rng.permutation(len(pool)):
            p = tuple(map(int, pool[int(i)]))
            if condition(p) and add(p):
                return p
        return None
    if pick(pools['critical']) is None or pick(pools['open']) is None:
        return None
    if teach:
        for color in colors:
            if not any(contained[p] and scene[p] == color for p in holes):
                if pick(pools['closed'], lambda p: scene[p] == color) is None:
                    return None
        # A contained hole beside a barrier teaches clipped rings.
        if pick(pools['closed'], lambda p: any(walls[q] for q in _ring(scene.shape, p))) is None:
            return None
        # Teach center precedence with an adjacent compatible pair.
        adjacent = False
        for p in list(holes):
            if not contained[p]:
                continue
            neighbors = _ring(scene.shape, p)
            rng.shuffle(neighbors)
            for q in neighbors:
                if contained[q] and scene[q] == scene[p] and add(q):
                    adjacent = True
                    break
            if adjacent:
                break
        if not adjacent:
            return None
    target = int(rng.integers(10 if rich else 4, 15 if rich else 10))
    pool = np.argwhere(eligible)
    for i in rng.permutation(len(pool)):
        if len(holes) >= target:
            break
        add(pool[int(i)])
    return holes if len(holes) >= target else None


def _make_pair(rng, shape, colors, mode, teach, rich):
    for _ in range(180):
        # Transposition changes both room geometry and repair orientations.
        transpose = bool(rng.integers(2))
        local_shape = shape[::-1] if transpose else shape
        network = _wall_network(rng, local_shape, mode)
        if network is None:
            continue
        damage = _open_and_damage(rng, network, rich)
        if damage is None:
            continue
        walls, wall_input, contained, critical, leaked = damage
        material = _sample_material(rng, local_shape, wall_input, walls, colors)
        if material is None:
            continue
        scene, boxes, order = material
        holes = _select_holes(rng, scene, walls, wall_input, contained, critical, leaked, colors, teach, rich)
        if holes is None:
            continue
        try:
            expected, _, demands = _compose(scene, walls, holes, colors)
        except ValueError:
            continue
        inp = scene.copy()
        for p in holes:
            inp[p] = 0
        # Wrong-fill corruption is restricted to the common bounding region;
        # it never changes the visible rectangle extrema used for recovery.
        a, b = boxes
        pool = [(r, c) for r in range(max(a[0], b[0]), min(a[2], b[2]))
                for c in range(max(a[1], b[1]), min(a[3], b[3]))
                if not walls[r, c] and not wall_input[r, c] and (r, c) not in holes]
        rng.shuffle(pool)
        if teach:
            # A visible ordinary correction teaches that wrong-fill errors
            # are not centers; do not hide every error under a marker ring.
            pool.sort(key=lambda p: p in demands)
        wrong_count = int(rng.integers(1 if teach else 0, 5))
        for p in pool[:wrong_count]:
            inp[p] = colors[1] if scene[p] == colors[0] else colors[0]
        inp[wall_input] = 5
        try:
            a = _analyze(inp)
            if (not np.array_equal(a['scene'], scene)
                    or not np.array_equal(a['walls'], walls)
                    or set(a['holes']) != set(holes)
                    or not np.array_equal(a['output'], expected)):
                continue
            if teach and not any(inp[p] in colors and inp[p] != scene[p]
                                 and expected[p] == scene[p]
                                 for p in pool[:wrong_count]):
                continue
        except ValueError:
            continue
        if transpose:
            inp, expected = inp.T.copy(), expected.T.copy()
        return {'input': inp, 'output': expected}
    raise RuntimeError("Bounded construction could not find an evidenced scene")


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _input_key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(_MATERIALS, 2, replace=False)))
    palette_policy = int(rng.integers(3))
    shift = str(rng.choice(['scale', 'topology', 'density']))
    rich = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            teach = split == 'train' and index < 2
            if shift == 'scale':
                low, high = (19, 25) if split == 'train' else (25, 31)
            else:
                low, high = 23, 31
            shape = (int(rng.integers(low, high)), int(rng.integers(low, high)))
            if rng.random() < .18:
                shape = (shape[0], shape[0])
            if shift == 'topology' and split == 'train':
                mode = str(rng.choice(['offset', 'split']))
            elif shift == 'topology':
                mode = str(rng.choice(['tee', 'cross', 'nested', 'disjoint']))
            else:
                available = ['offset', 'split', 'tee']
                if min(shape) >= 23:
                    available.extend(['nested', 'cross'])
                if max(shape) >= 24:
                    available.append('disjoint')
                mode = str(rng.choice(available))
            pair_rich = (split == 'test') if shift == 'density' else rich
            colors = palette
            if palette_policy == 2 or (palette_policy == 1 and rng.random() < .4):
                colors = tuple(map(int, rng.choice(_MATERIALS, 2, replace=False)))
            for _ in range(6):
                pair = _make_pair(rng, shape, colors, mode, teach, pair_rich)
                key = _input_key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Could not create distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _input_key(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            a = _analyze(inp)
            assert np.array_equal(out, a['output']), "Incorrect input-derived answer"
            assert a['repairs'] and a['centers']
            assert len(a['holes']) > len(a['centers']), "No open-hole contrast"
            assert np.all(out[a['walls']] == 5), "A ring overwrote a wall"
            assert all(out[p] == 8 for p in a['centers']), "Overwritten center"
            raw_walls, _ = _restore_walls(inp == 5, repair=False)
            raw_contained = _regions(raw_walls)[1]
            assert any(not raw_contained[p] for p in a['centers']), "Repair has no marking consequence"
            assert _opening_witness(a['walls'], a['holes'], a['centers']), "No intentional-opening witness"
            assert _regions(a['walls'])[2] >= 1
            # The visible-extrema contract is checked against the reconstruction.
            for i, color in enumerate(a['colors']):
                assert _bbox(a['scene'] == color) == a['boxes'][i]
                assert _bbox(inp == color) == a['boxes'][i]
            # Wrong-fill cells are ordinary repairs, never erasure centers.
            errors = np.isin(inp, a['colors']) & (inp != a['scene'])
            b0, b1 = a['boxes']
            for r, c in np.argwhere(errors):
                assert all(b[0] <= r < b[2] and b[1] <= c < b[3] for b in (b0, b1))
            if split == 'train':
                training.append(a)
    assert any({int(a['scene'][p]) for p in a['centers']} == set(a['colors']) for a in training)
    assert any(any(a['scene'][p] == a['scene'][q]
                   and max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1
                   for i, p in enumerate(a['centers']) for q in a['centers'][:i]) for a in training)
    assert any(any(a['walls'][q] for q in _ring(a['grid'].shape, p))
               for a in training for p in a['centers'])
    assert any(np.any(np.isin(a['grid'], a['colors']) & (a['grid'] != a['scene'])
                      & (a['output'] == a['scene'])) for a in training)


def self_test():
    """Bounded 42-seed checks, counterfactuals, symmetries, and bad labels."""
    signatures, shapes, clean_shapes, palettes, counts, axes = set(), set(), set(), set(), set(), set()
    holes_counts, repair_counts, clipped, seam_rings = set(), set(), 0, 0
    first = None
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple((_input_key(p['input']), _input_key(p['output']))
                          for s in ('train', 'test') for p in puzzle[s])
        signatures.add(signature)
        if first is None:
            first = puzzle
        for pair in puzzle['train'] + puzzle['test']:
            a = _analyze(pair['input'])
            shapes.add(a['grid'].shape)
            palettes.add(tuple(a['colors']))
            counts.add(_regions(a['walls'])[2])
            axes.update(item[2] for item in a['repairs'])
            holes_counts.add(len(a['holes']))
            repair_counts.add(len(a['repairs']))
            normalized = np.zeros_like(a['scene'])
            for i, color in enumerate(a['colors']):
                normalized[a['scene'] == color] = i + 1
            normalized[a['walls']] = 5
            clean_shapes.add(_input_key(normalized))
            labels = _regions(a['walls'])[0]
            clipped += any(any(labels[q] != labels[p] for q in _ring(a['grid'].shape, p))
                           for p in a['centers'])
            seam_rings += any(any(a['scene'][q] in a['colors'] and a['scene'][q] != a['scene'][p]
                                  and labels[q] == labels[p] for q in _ring(a['grid'].shape, p))
                              for p in a['centers'])
            # Omitting A changes selection, beyond just leaving gaps unfilled.
            unrepaired, _ = _restore_walls(a['grid'] == 5, repair=False)
            alternative, centers, _ = _compose(a['scene'], unrepaired, a['holes'], a['colors'])
            assert len(centers) < len(a['centers'])
            assert not np.array_equal(alternative, a['output'])
            # Omitting B leaves a clean material/wall drawing without symbols.
            just_cleanup = a['scene'].copy()
            just_cleanup[a['walls']] = 5
            assert not np.array_equal(just_cleanup, a['output'])
            # Mark-all-holes shortcut is refuted by an open hole in every pair.
            assert any(p not in a['centers'] and a['output'][p] != 8 for p in a['holes'])
            assert _opening_witness(a['walls'], a['holes'], a['centers'])
    assert len(signatures) == 42
    assert len(shapes) >= 40 and len(clean_shapes) >= 180 and len(palettes) >= 15
    assert axes == {'h', 'v'} and max(repair_counts) >= 2
    assert max(holes_counts) >= 12 and len(holes_counts) >= 7
    assert max(counts) >= 2 and clipped >= 150 and seam_rings >= 20
    repeated = generate_puzzle(_HOST_SEEDS[0])
    assert all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
               for p, q in zip(first[s], repeated[s]) for k in ('input', 'output'))
    # Input-only solving is rotation/reflection equivariant.
    pair = first['train'][0]
    for turns in range(4):
        for reflect in (False, True):
            inp, out = np.rot90(pair['input'], turns), np.rot90(pair['output'], turns)
            if reflect:
                inp, out = inp[:, ::-1], out[:, ::-1]
            assert np.array_equal(solve(inp), out)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in ('train', 'test')}
    center = _analyze(bad['test'][0]['input'])['centers'][0]
    bad['test'][0]['output'][center] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Missed marker accepted")
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
