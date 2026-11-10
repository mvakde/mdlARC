"""Repair across a membrane: input-only reconstruction and regional ring arbitration.

Only standard library and NumPy. No file I/O, parent imports, cached examples,
execution traces, or generator metadata are used by the solver.
"""
from itertools import permutations
import hashlib
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grids must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r, c in np.argwhere(mask):
        p = (int(r), int(c))
        if seen[p]:
            continue
        seen[p] = True
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in (_D8 if diagonal else _D4):
                q = (r + dr, c + dc)
                if 0 <= q[0] < h and 0 <= q[1] < w and mask[q] and not seen[q]:
                    seen[q] = True
                    stack.append(q)
        result.append(cells)
    return result


def _regions(cut):
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    _require(len(cells) >= 4, "short cut")
    h, w = cut.shape
    degrees = {}
    for r, c in cells:
        neighbors = []
        for dr, dc in _D8:
            q = (r + dr, c + dc)
            if q in cells:
                # Do not count a diagonal shortcut around an orthogonal bend.
                if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                    continue
                neighbors.append(q)
        degrees[r, c] = neighbors
    ends = [p for p in cells if len(degrees[p]) == 1]
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(len(ends) == 2 and all(len(v) in (1, 2) for v in degrees.values()), "branched/closed cut")
    _require(all(border(p) == (p in ends) for p in cells), "bad boundary contacts")
    _require(len(_components(cut, True)) == 1, "disconnected cut")
    cc = _components(~cut)
    _require(len(cc) == 2, "cut must make exactly two regions")
    labels = np.full(cut.shape, -1, dtype=np.int8)
    for i, component in enumerate(cc):
        for p in component:
            labels[p] = i
    return labels


def _sites(cut, labels, side):
    near = cut.copy()
    for r, c in np.argwhere(cut):
        near[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    return [tuple(map(int, p)) for p in np.argwhere((labels == side) & ~near)]


def _unpack(value):
    a = _grid(value)
    frame = int(a[0, 0])
    _require(bool(np.all(a[[0, -1], :] == frame) and np.all(a[:, [0, -1]] == frame)), "broken frame")
    options = []
    for vertical in (False, True):
        view = a.T if vertical else a
        if view.shape[1] < 7 or (view.shape[1] - 3) % 2:
            continue
        w = (view.shape[1] - 3) // 2
        if not np.all(view[:, w + 1] == frame):
            continue
        panels = [view[1:-1, 1:w + 1], view[1:-1, w + 2:-1]]
        for ci in (0, 1):
            control, scene = panels[ci], panels[1 - ci]
            vals, counts = np.unique(control, return_counts=True)
            if len(vals) != 2 or len(np.unique(scene)) != 3:
                continue
            bg = int(vals[np.argmax(counts)])
            ink = int(vals[0] if vals[1] == bg else vals[1])
            fills = tuple(int(c) for c in np.unique(scene) if c != bg)
            if len(fills) != 2 or len({frame, bg, ink, *fills}) != 5:
                continue
            components = sorted(_components(control == ink, True), key=len)
            if len(components) != 2 or len(components[0]) != 1:
                continue
            marker = components[0][0]
            cut = (control == ink).copy()
            cut[marker] = False
            try:
                labels = _regions(cut)
                side = int(labels[marker])
                if marker not in _sites(cut, labels, side):
                    continue
            except ValueError:
                continue
            # Normalize labels so 0 is the marked side, 1 the other side.
            owner = labels.copy()
            owner[labels == side], owner[labels == 1 - side] = 0, 1
            options.append(dict(array=a, frame=frame, background=bg, ink=ink,
                                fills=fills, control=control, scene=scene, cut=cut,
                                owner=owner, marker=marker, vertical=vertical, ci=ci))
    _require(len(options) == 1, "ambiguous panel roles")
    return options[0]


def _replace(d, scene):
    out = d['array'].copy()
    view = out.T if d['vertical'] else out
    w = scene.shape[1]
    start = 1 if d['ci'] == 1 else w + 2
    view[1:-1, start:start + w] = scene
    return out


def _paint(shape, bg, boxes):
    clean = np.full(shape, bg, dtype=np.int8)
    for r0, c0, r1, c1, color in boxes:
        clean[r0:r1, c0:c1] = color
    return clean


def _recover(scene, bg, fills):
    """Dense components propose boxes; covering orders are scored on input only.

    Each material rectangle has one >=8-cell connected surviving component
    reaching all four of its true bounding extremes. Sparse noise components
    have <=2 cells and cannot attach to a main component of their own color.
    These are construction/readability constraints, not an output lookup.
    """
    boxes = []
    for color in fills:
        for cc in _components(scene == color):
            if len(cc) <= 2:
                continue
            _require(len(cc) >= 8, "ambiguous medium component")
            rr, cols = zip(*cc)
            box = (min(rr), min(cols), max(rr) + 1, max(cols) + 1, color)
            _require(min(box[2] - box[0], box[3] - box[1]) >= 4, "thin rectangle")
            boxes.append(box)
    _require(2 <= len(boxes) <= 4, "need two to four supported rectangles")
    _require({b[4] for b in boxes} == set(fills), "missing fill role")
    candidates = {}
    for order in permutations(boxes):
        clean = _paint(scene.shape, bg, order)
        # Every proposed object must still supply its full visible bounding box.
        supported = True
        for r0, c0, r1, c1, color in boxes:
            visible = clean[r0:r1, c0:c1] == color
            if (visible.sum() < 8 or min(visible[0].sum(), visible[-1].sum(),
                                        visible[:, 0].sum(), visible[:, -1].sum()) < 2):
                supported = False
                break
        if supported:
            key = clean.tobytes()
            candidates[key] = (int(np.count_nonzero(clean != scene)), clean, order)
    _require(bool(candidates), "no supported covering order")
    ranked = sorted(candidates.values(), key=lambda t: (t[0], t[1].tobytes()))
    loss, clean, order = ranked[0]
    _require(len(ranked) == 1 or ranked[1][0] >= loss + 3, "uncertain occlusion")
    _require(loss <= max(10, .19 * scene.size), "excessive damage")
    _require(np.count_nonzero(clean == bg) >= 8, "missing exterior")
    for r0, c0, r1, c1, color in boxes:
        visible = clean[r0:r1, c0:c1] == color
        observed = scene[r0:r1, c0:c1] == color
        _require(np.count_nonzero(visible & observed) >= .72 * visible.sum(), "insufficient intact material")
    return clean, order


def _ring(shape, p):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in _D8
            if 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]]


def _fields(clean, holes, owner, fills):
    fields = np.full((2, *clean.shape), -1, dtype=np.int8)
    centers = set(holes)
    for p in holes:
        side = int(owner[p])
        _require(side in (0, 1), "a hole cannot lie on the cut")
        other = fills[1] if int(clean[p]) == fills[0] else fills[0]
        for q in _ring(clean.shape, p):
            if q in centers:
                continue
            old = int(fields[side][q])
            _require(old in (-1, other), "incompatible same-side rings")
            fields[side][q] = other
    return fields


def _arbitrate(clean, holes, owner, fields, ink):
    result = clean.copy()
    for r in range(clean.shape[0]):
        for c in range(clean.shape[1]):
            x, y = int(fields[0, r, c]), int(fields[1, r, c])
            if x < 0 and y < 0:
                continue
            if x < 0:
                result[r, c] = y
            elif y < 0:
                result[r, c] = x
            else:
                result[r, c] = y if owner[r, c] == 1 else x
    for p in holes:
        result[p] = ink
    return result


def _analyze(value):
    d = _unpack(value)
    clean, boxes = _recover(d['scene'], d['background'], d['fills'])
    holes = [tuple(map(int, p)) for p in np.argwhere((clean != d['background']) &
                                                    (d['scene'] == d['background']))]
    _require(bool(holes), "no erased cells")
    fields = _fields(clean, holes, d['owner'], d['fills'])
    d.update(clean=clean, boxes=boxes, holes=holes, fields=fields)
    return d


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _analyze(input_grid)
    repaired = _arbitrate(d['clean'], d['holes'], d['owner'], d['fields'], d['ink'])
    return _replace(d, repaired)


def _pack(control, scene, frame, vertical, ci):
    h, w = control.shape
    grid = np.full((h + 2, 2 * w + 3), frame, dtype=np.int8)
    for i, p in enumerate((control, scene) if ci == 0 else (scene, control)):
        start = 1 + i * (w + 1)
        grid[1:-1, start:start + w] = p
    return grid.T.copy() if vertical else grid


def _cut(rng, h, w, family):
    for _ in range(40):
        transpose = bool(rng.integers(2))
        a, b = (w, h) if transpose else (h, w)
        cut = np.zeros((a, b), dtype=bool)
        if family == 'straight':
            cut[:, int(rng.integers(2, b - 2))] = True
        elif family == 'diagonal':
            c0 = int(rng.integers(1, max(2, b // 3)))
            c1 = int(rng.integers(max(c0 + 2, 2 * b // 3), b - 1))
            cols = np.rint(np.linspace(c0, c1, a)).astype(int)
            # Interpolate if a shallow line would leave gaps in eight-connectivity.
            for r in range(a - 1):
                steps = max(1, abs(int(cols[r + 1]) - int(cols[r])))
                for k in range(steps + 1):
                    rr = int(round(r + k / steps))
                    cc = int(round(cols[r] + (cols[r + 1] - cols[r]) * k / steps))
                    cut[rr, cc] = True
            cut[a - 1, cols[-1]] = True
        elif family == 'elbow':
            r, c = int(rng.integers(3, a - 3)), int(rng.integers(2, b - 3))
            cut[:r + 1, c] = True
            cut[r, c:] = True
        else:
            rows = list(range(3, a - 2, 3))
            count = int(rng.integers(1, min(3, len(rows)) + 1))
            rows = sorted(int(r) for r in rng.choice(rows, count, replace=False))
            last, col = 0, int(rng.integers(2, b - 2))
            for r in rows:
                cut[last:r + 1, col] = True
                choices = [v for v in range(2, b - 2) if v != col]
                nxt = int(rng.choice(choices))
                cut[r, min(col, nxt):max(col, nxt) + 1] = True
                last, col = r, nxt
            cut[last:, col] = True
        if transpose:
            cut = cut.T.copy()
        if rng.integers(2):
            cut = cut[::-1].copy()
        if rng.integers(2):
            cut = cut[:, ::-1].copy()
        try:
            labels = _regions(cut)
            if all(_sites(cut, labels, i) for i in (0, 1)):
                return cut, labels
        except ValueError:
            pass
    raise ValueError("could not sample cut")


def _scene(rng, h, w, count):
    """Sample free rectangle dimensions/positions, accepting visible evidence."""
    for _ in range(100):
        boxes = []
        for i in range(count):
            rh = int(rng.integers(4, max(5, min(h, int(h * .78)) + 1)))
            rw = int(rng.integers(4, max(5, min(w, int(w * .85)) + 1)))
            r, c = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
            boxes.append((r, c, r + rh, c + rw, 1 + i % 2))
        ordered = [boxes[int(i)] for i in rng.permutation(count)]
        clean = _paint((h, w), 0, ordered)
        try:
            recovered, inferred = _recover(clean, 0, (1, 2))
            if (len(inferred) == count and np.array_equal(clean, recovered)
                    and len({b[:4] for b in inferred}) == count):
                # Require genuine overlap, rather than unrelated separate blocks.
                area = sum((r1 - r0) * (c1 - c0) for r0, c0, r1, c1, _ in inferred)
                if area > np.count_nonzero(clean):
                    return clean
        except ValueError:
            pass
    return None


def _distance(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def _choose_holes(rng, clean, cut, labels, target, teach):
    pool = [tuple(map(int, p)) for p in np.argwhere((clean != 0) & ~cut)]
    rng.shuffle(pool)
    holes = []
    if teach:
        # A cross-seam conflict with recipients on and off the cut demonstrates
        # spatial arbitration; it cannot be explained by one globally prior ring.
        for p in pool:
            for q in pool:
                if labels[p] == labels[q] or clean[p] == clean[q] or _distance(p, q) > 2:
                    continue
                shared = (set(_ring(clean.shape, p)) & set(_ring(clean.shape, q))) - {p, q}
                if any(cut[t] for t in shared) and any(not cut[t] for t in shared):
                    holes = [p, q]
                    break
            if holes:
                break
        if not holes:
            return None
        # Teach an intact complementary ring in each fill away from arbitration.
        for color in (1, 2):
            choices = [p for p in pool if clean[p] == color and len(_ring(clean.shape, p)) == 8
                       and all(clean[q] == color for q in _ring(clean.shape, p))
                       and all(_distance(p, q) >= 4 for q in holes)]
            if not choices:
                return None
            choices.sort(key=lambda p: not any(labels[q] >= 0 and labels[q] != labels[p]
                                               for q in _ring(clean.shape, p)))
            holes.append(choices[0])
        # An adjacent same-side, same-fill pair teaches center precedence.
        for p in pool:
            if any(_distance(p, q) == 1 and labels[p] == labels[q] and clean[p] == clean[q]
                   for q in holes) and all(clean[p] == clean[q] or _distance(p, q) > 2
                                         or labels[p] != labels[q] for q in holes):
                if p not in holes:
                    holes.append(p)
                    break
        else:
            return None
    for p in pool:
        if len(holes) >= target:
            break
        if p in holes:
            continue
        if any(labels[p] == labels[q] and clean[p] != clean[q] and _distance(p, q) <= 2 for q in holes):
            continue
        # Keep holes sparse enough for object reconstruction, including clusters.
        if sum(_distance(p, q) <= 1 for q in holes) >= 3:
            continue
        holes.append(p)
    return holes if len(holes) >= 3 else None


def _damage(rng, clean, holes, teach):
    damaged = clean.copy()
    for p in holes:
        damaged[p] = 0
    # Wrong-fill defects and exterior speckles are not report centers. Keep them
    # isolated from same-colored main objects so their box evidence stays clear.
    candidates = [tuple(map(int, p)) for p in np.argwhere(clean != 0)
                  if tuple(p) not in holes]
    rng.shuffle(candidates)
    wrong = 0
    for p in candidates:
        if wrong >= (1 if teach else int(rng.integers(0, 3))):
            break
        if len(_ring(clean.shape, p)) != 8 or not all(clean[q] == clean[p] for q in _ring(clean.shape, p)):
            continue
        if any(_distance(p, q) <= 1 for q in holes):
            continue
        trial = damaged.copy()
        trial[p] = 3 - int(clean[p])
        try:
            fit, _ = _recover(trial, 0, (1, 2))
            if np.array_equal(fit, clean):
                damaged, wrong = trial, wrong + 1
        except ValueError:
            pass
    if teach and wrong == 0:
        return None
    exterior = [tuple(map(int, p)) for p in np.argwhere(clean == 0)]
    rng.shuffle(exterior)
    speckles = 0
    for p in exterior:
        if speckles >= (1 if teach else 2):
            break
        if teach and any(_distance(p, q) <= 1 for q in holes):
            continue
        trial = damaged.copy()
        trial[p] = int(rng.integers(1, 3))
        try:
            fit, _ = _recover(trial, 0, (1, 2))
            if np.array_equal(fit, clean):
                damaged, speckles = trial, speckles + 1
        except ValueError:
            pass
    if teach and speckles == 0:
        return None
    return damaged


def _new_pair(rng, family, height_range, count, teach, palette, target):
    for _ in range(260):
        h, w = int(rng.integers(*height_range)), int(rng.integers(9, 14))
        cut, labels = _cut(rng, h, w, family)
        clean = _scene(rng, h, w, count)
        if clean is None:
            continue
        holes = _choose_holes(rng, clean, cut, labels, target, teach)
        if holes is None:
            continue
        damaged = _damage(rng, clean, holes, teach)
        if damaged is None:
            continue
        try:
            fit, _ = _recover(damaged, 0, (1, 2))
            if not np.array_equal(fit, clean):
                continue
        except ValueError:
            continue
        bg, ca, cb, ink, frame = palette
        side = int(rng.integers(2))
        choices = _sites(cut, labels, side)
        marker = choices[int(rng.integers(len(choices)))]
        control = np.full(clean.shape, bg, dtype=np.int8)
        control[cut], control[marker] = ink, ink
        mapping = np.array([bg, ca, cb], dtype=np.int8)
        inp = _pack(control, mapping[damaged], frame, bool(rng.integers(2)), int(rng.integers(2)))
        try:
            d = _analyze(inp)
            # Construction truth versus input-only reconstruction, not solve
            # versus itself. Palette changes cannot change the inferred geometry.
            if not np.array_equal(d['clean'], mapping[clean]) or set(d['holes']) != set(holes):
                continue
            # Later centers can occupy an originally conflicting ring pixel.
            # Check the returned scene's actual evidence, not just the initial
            # candidate pair, before accepting the controlled demonstrations.
            if teach and not all(_evidence(d).values()):
                continue
            out = _replace(d, _arbitrate(d['clean'], holes, d['owner'], d['fields'], ink))
            return {'input': inp, 'output': out}
        except ValueError:
            continue
    raise RuntimeError("bounded sampler exhausted")


def _twin(pair, rng):
    d = _analyze(pair['input'])
    control = d['control'].copy()
    control[d['marker']] = d['background']
    sites = _sites(d['cut'], d['owner'], 1)
    control[sites[int(rng.integers(len(sites)))]] = d['ink']
    inp = _pack(control, d['scene'], d['frame'], d['vertical'], d['ci'])
    return {'input': inp, 'output': solve(inp)}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(int(x) for x in rng.choice(10, 5, replace=False))
    regime = int(rng.integers(3))
    palette_mode = int(rng.integers(3))
    puzzle = {'train': [], 'test': []}
    seen = set()
    first = _new_pair(rng, 'diagonal', (12, 19), 2, True, palette, 6)
    puzzle['train'] = [first, _twin(first, rng)]
    seen.update(_key(p['input']) for p in puzzle['train'])
    for split, total in (('train', nt), ('test', nq)):
        for index in range(2 if split == 'train' else 0, total):
            for _ in range(8):
                if palette_mode == 1 and rng.random() < .5:
                    colors = tuple(int(x) for x in rng.choice(10, 5, replace=False))
                elif palette_mode == 2:
                    rest = [c for c in range(10) if c not in (palette[0], palette[3], palette[4])]
                    colors = (palette[0], *map(int, rng.choice(rest, 2, replace=False)), palette[3], palette[4])
                else:
                    colors = palette
                if regime == 0:
                    families = ('diagonal', 'straight') if split == 'train' else ('elbow', 'stepped')
                    heights = (12, 25)
                elif regime == 1:
                    families = ('straight', 'diagonal', 'elbow', 'stepped')
                    heights = (12, 19) if split == 'train' else (21, 29)
                else:
                    families = ('straight', 'diagonal', 'elbow', 'stepped')
                    heights = (12, 29)
                count = int(rng.choice([2, 3, 4], p=[.48, .40, .12])) if heights[1] > 20 else int(rng.choice([2, 3]))
                target = int(rng.integers(9, 15)) if regime == 2 and split == 'test' else int(rng.integers(3, 10))
                pair = _new_pair(rng, str(rng.choice(families)), heights, count, False, colors, target)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not produce distinct input')
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(d):
    f = d['fields']
    conflict = (f[0] >= 0) & (f[1] >= 0) & (f[0] != f[1])
    crossing0 = (f[0] >= 0) & (f[1] < 0) & (d['owner'] == 1) & (f[0] != d['clean'])
    crossing1 = (f[1] >= 0) & (f[0] < 0) & (d['owner'] == 0) & (f[1] != d['clean'])
    return dict(transparent=bool(np.any(crossing0 | crossing1)),
                seam_conflict=bool(np.any(conflict & d['cut'])),
                side_conflict=bool(np.any(conflict & ~d['cut'])),
                both_fills={int(d['clean'][p]) for p in d['holes']} == set(d['fills']),
                adjacent=any(_distance(p, q) == 1 and d['owner'][p] == d['owner'][q]
                             for i, p in enumerate(d['holes']) for q in d['holes'][:i]),
                wrong=bool(np.any((d['scene'] != d['clean']) & (d['scene'] != d['background']) &
                                  (d['clean'] != d['background']))),
                speckle=bool(np.any((d['scene'] != d['background']) & (d['clean'] == d['background']))))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, coverage = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'shape changed')
            key = _key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            d = _analyze(inp)
            expected = solve(inp)
            _require(np.array_equal(out, expected), 'incorrect repair or changed annotation/frame')
            _require(not np.array_equal(inp, out), 'identity pair')
            if split == 'train':
                coverage.update(k for k, v in _evidence(d).items() if v)
    _require(coverage == {'transparent', 'seam_conflict', 'side_conflict', 'both_fills', 'adjacent', 'wrong', 'speckle'},
             'training does not expose the complete mechanism')
    # The first two demonstrations isolate the marker's effect, including the
    # fact that off-cut ownership stays spatial while the seam priority changes.
    a, b = (_analyze(p['input']) for p in puzzle['train'][:2])
    _require(np.array_equal(a['scene'], b['scene']) and np.array_equal(a['cut'], b['cut']), 'missing controlled comparison')
    _require(a['owner'][b['marker']] == 1, 'marker comparison must select opposite sides')
    ra = _arbitrate(a['clean'], a['holes'], a['owner'], a['fields'], a['ink'])
    rb = _arbitrate(b['clean'], b['holes'], b['owner'], b['fields'], b['ink'])
    _require(np.any(ra[a['cut']] != rb[a['cut']]), 'marker has no observable effect')
    _require(np.array_equal(ra[~a['cut']], rb[~a['cut']]), 'marker changed non-seam ownership')


def self_test():
    """42 puzzles, streamed; metamorphic checks and bounded structural metrics."""
    shapes, layouts, scenes, palettes, counts, signatures = set(), set(), set(), set(), set(), set()
    ablations = dict(ignore_partition=0, ignore_marker=0, no_complement=0,
                     strict_clipping=0, no_reconstruction=0)
    totals = {k: 0 for k in ('transparent', 'seam_conflict', 'side_conflict', 'both_fills', 'adjacent', 'wrong', 'speckle')}
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        local_ablations = {k: False for k in ablations}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _analyze(pair['input'])
                pair_count += 1
                shapes.add(d['array'].shape)
                layouts.add((d['cut'].shape, d['cut'].tobytes()))
                normalized = np.zeros_like(d['clean'])
                used = [int(c) for c in d['clean'].flat if c != d['background']]
                first_color = used[0]
                normalized[d['clean'] == first_color] = 1
                normalized[(d['clean'] != d['background']) & (d['clean'] != first_color)] = 2
                scenes.add((normalized.shape, normalized.tobytes()))
                palettes.add((d['background'], d['fills'], d['ink'], d['frame']))
                counts.add(len(d['boxes']))
                signature.update(pair['input'].tobytes())
                signature.update(pair['output'].tobytes())
                for k, present in _evidence(d).items():
                    totals[k] += int(present)
                # Independent vectorized arbitration using per-side fields.
                f = d['fields']
                r = d['clean'].copy()
                only0, only1 = (f[0] >= 0) & (f[1] < 0), (f[1] >= 0) & (f[0] < 0)
                both = (f[0] >= 0) & (f[1] >= 0)
                r[only0], r[only1] = f[0][only0], f[1][only1]
                choose0 = both & (d['owner'] != 1)
                choose1 = both & (d['owner'] == 1)
                r[choose0], r[choose1] = f[0][choose0], f[1][choose1]
                for p in d['holes']:
                    r[p] = d['ink']
                assert np.array_equal(_replace(d, r), pair['output'])
                # Concrete simpler-rule counterfactuals, tested on the entire
                # first controlled comparison in every puzzle.
                if split == 'train' and any(pair is p for p in puzzle['train'][:2]):
                    desired = pair['output']
                    global_owner = np.zeros_like(d['owner'])
                    alternative = _arbitrate(d['clean'], d['holes'], global_owner, f, d['ink'])
                    local_ablations['ignore_partition'] |= not np.array_equal(_replace(d, alternative), desired)
                    fixed_owner = d['owner'].copy()
                    fixed_owner[d['cut']] = 1
                    alternative = _arbitrate(d['clean'], d['holes'], fixed_owner, f, d['ink'])
                    local_ablations['ignore_marker'] |= not np.array_equal(_replace(d, alternative), desired)
                    same = f.copy()
                    for color in d['fills']:
                        same[f == color] = next(c for c in d['fills'] if c != color)
                    alternative = _arbitrate(d['clean'], d['holes'], d['owner'], same, d['ink'])
                    local_ablations['no_complement'] |= not np.array_equal(_replace(d, alternative), desired)
                    clipped = f.copy()
                    clipped[0][d['owner'] == 1] = -1
                    clipped[1][d['owner'] == 0] = -1
                    alternative = _arbitrate(d['clean'], d['holes'], d['owner'], clipped, d['ink'])
                    local_ablations['strict_clipping'] |= not np.array_equal(_replace(d, alternative), desired)
                    alternative = _arbitrate(d['scene'], d['holes'], d['owner'], f, d['ink'])
                    local_ablations['no_reconstruction'] |= not np.array_equal(_replace(d, alternative), desired)
                # Whole-grid rotations/reflections and arbitrary color renaming
                # must commute with the input-only rule.
                if split == 'test':
                    assert np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output']))
                    assert np.array_equal(solve(pair['input'][::-1]), pair['output'][::-1])
                    rename = np.array([7, 4, 9, 0, 1, 8, 3, 6, 2, 5], dtype=np.int8)
                    assert np.array_equal(solve(rename[pair['input']]), rename[pair['output']])
        signatures.add(signature.hexdigest())
        for k, present in local_ablations.items():
            ablations[k] += int(present)
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(layouts) >= 100 and len(scenes) >= 120
    assert counts == {2, 3, 4} and len(palettes) >= 40
    assert all(totals[k] >= 84 for k in totals)
    assert all(v == 42 for v in ablations.values()), 'a simpler rule survived demonstrations'
    p = generate_puzzle(0)
    q = generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            z = generate_puzzle(42 + 10 * nt + nq, nt, nq)
            assert len(z['train']) == nt and len(z['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (10, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    return dict(seeds=42, pairs=pair_count, grid_shapes=len(shapes), cut_layouts=len(layouts),
                color_normalized_clean_scenes=len(scenes), palettes=len(palettes),
                rectangle_counts=sorted(counts), evidence=totals, ablations_distinguished=ablations)
