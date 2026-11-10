"""Dock closed contours, then label their convex and concave bends.

Only Python/NumPy are required. All geometric conventions are input-visible;
there are no parent imports, stored puzzles, latent palettes, or solver caches.
"""
from collections import deque
import json
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N,E,S,W
NEAR = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1))
HOST_SEEDS = (0,594690685,211423409,212452790,958087415,374165759,376319608,679217780,283529596,991646897,901771383,428187680,546815223,306443060,998370544,477178503,47853136,659807627,328930970,83033156,576710019,443098280,532013701,61987205,164830744,13497317,541002248,723226833,385552862,266187035,273964727,992807461,835847069,960089697,680320536,549123091,110570382,175294559,942925067,129927086,515634155,958211924)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(a.dtype.kind in "iu" and bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(points, diagonal=False):
    left = set(points)
    result = []
    offsets = NEAR if diagonal else DIRS
    while left:
        start = min(left)
        left.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in offsets:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    stack.append(q)
                    group.add(q)
        result.append(group)
    return result


def _transform(cells, pose):
    result = {}
    for (r, c), v in cells.items():
        if pose >= 4:
            c = -c
        for _ in range(pose % 4):
            r, c = c, -r
        result[r, c] = v
    r0 = min(r for r, _ in result)
    c0 = min(c for _, c in result)
    return {(r-r0, c-c0): v for (r, c), v in result.items()}


def _shift(cells, row, col):
    return {(r+row, c+col): v for (r, c), v in cells.items()}


def _signature(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _halo(points):
    return {(r+dr, c+dc) for r, c in points for dr, dc in NEAR}


def _cycle(core):
    """Classify cell-center turns by signed area; no compass colors here."""
    core = set(core)
    _require(len(core) >= 8, "contour too small")
    neighbors = {p: sorted((p[0]+dr, p[1]+dc) for dr, dc in DIRS
                           if (p[0]+dr, p[1]+dc) in core) for p in core}
    _require(all(len(v) == 2 for v in neighbors.values()), "contour has a contact, end or branch")
    start = min(core)
    path, seen, previous, current = [start], {start}, start, neighbors[start][0]
    while current != start:
        _require(current not in seen, "invalid cycle")
        seen.add(current)
        path.append(current)
        following = next(q for q in neighbors[current] if q != previous)
        previous, current = current, following
    _require(len(path) == len(core), "several cycles in one piece")
    area = sum(c*path[(i+1) % len(path)][0] - r*path[(i+1) % len(path)][1]
               for i, (r, c) in enumerate(path))
    _require(area != 0, "zero enclosed area")
    classes = {}
    for i, (r, c) in enumerate(path):
        pr, pc = path[i-1]
        nr, nc = path[(i+1) % len(path)]
        cross = (c-pc)*(nr-r) - (r-pr)*(nc-c)
        classes[r, c] = 0 if cross == 0 else 1 if cross*area > 0 else 2
    _require(sum(v == 1 for v in classes.values()) - sum(v == 2 for v in classes.values()) == 4,
             "invalid orthogonal winding")
    # Padded flood fill certifies a genuine, nonempty discrete interior.
    r0, r1 = min(r for r, _ in core)-1, max(r for r, _ in core)+1
    c0, c1 = min(c for _, c in core)-1, max(c for _, c in core)+1
    exterior, queue = {(r0, c0)}, deque([(r0, c0)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            q = r+dr, c+dc
            if r0 <= q[0] <= r1 and c0 <= q[1] <= c1 and q not in core and q not in exterior:
                exterior.add(q)
                queue.append(q)
    _require((r1-r0+1)*(c1-c0+1) > len(core)+len(exterior), "no enclosed background cell")
    return path, classes


def _structure(a, bg):
    rows, cols = set(), set()
    for _ in range(3):
        active_c = [c for c in range(a.shape[1]) if c not in cols]
        for r in range(a.shape[0]):
            v = a[r, active_c]
            if len(v) and v[0] != bg and np.all(v == v[0]):
                rows.add(r)
        active_r = [r for r in range(a.shape[0]) if r not in rows]
        for c in range(a.shape[1]):
            v = a[active_r, c]
            if len(v) and v[0] != bg and np.all(v == v[0]):
                cols.add(c)
    mask = np.zeros(a.shape, dtype=bool)
    mask[list(rows), :] = True
    mask[:, list(cols)] = True
    _require(rows and cols and not mask.all(), "missing colored frame")
    return mask


def _regions(a, mask):
    groups = _components(map(tuple, np.argwhere(~mask)))
    index = np.full(a.shape, -1, dtype=np.int8)
    rectangles, sides = [], []
    for i, group in enumerate(groups):
        r0, r1 = min(r for r, _ in group), max(r for r, _ in group)+1
        c0, c1 = min(c for _, c in group), max(c for _, c in group)+1
        _require(len(group) == (r1-r0)*(c1-c0), "nonrectangular field")
        side = []
        for d, (dr, dc) in enumerate(DIRS):
            colors = {int(a[r+dr, c+dc]) for r, c in group
                      if 0 <= r+dr < a.shape[0] and 0 <= c+dc < a.shape[1]
                      and mask[r+dr, c+dc]}
            _require(len(colors) == 1, "each field side needs one band identity")
            side.append(next(iter(colors)))
        _require(len(set(side)) == 4, "side identities must differ")
        for p in group:
            index[p] = i
        rectangles.append((r0, r1, c0, c1))
        sides.append(tuple(side))
    return rectangles, index, sides


def _piece_info(cells, reg, bands):
    core = {p for p, v in cells.items() if v not in bands}
    path, classes = _cycle(core)
    marks = {p for p, v in cells.items() if v == reg}
    _require(marks and all(classes[p] == 0 for p in marks), "registration must occupy straight contour cells")
    _require(len(_components(marks)) == 1, "registration strip disconnected")
    _require(len({r for r, _ in marks}) == 1 or len({c for _, c in marks}) == 1,
             "registration must be a straight strip")
    body_values = {cells[p] for p in core - marks}
    _require(len(body_values) == 1, "one contour color required")
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    tabs = {}
    tab_count = 0
    for v in sorted(bands):
        patch = {p for p, color in cells.items() if color == v}
        if not patch:
            continue
        tab_count += 1
        _require(len(patch) in (1, 2) and len(_components(patch)) == 1, "invalid compass tab")
        facing = []
        if all(r == r0-1 and c0 < c < c1 for r, c in patch):
            facing.append(0)
        if all(c == c1+1 and r0 < r < r1 for r, c in patch):
            facing.append(1)
        if all(r == r1+1 and c0 < c < c1 for r, c in patch):
            facing.append(2)
        if all(c == c0-1 and r0 < r < r1 for r, c in patch):
            facing.append(3)
        _require(len(facing) == 1, "tab must project from one contour bounding side")
        d = facing[0]
        dr, dc = DIRS[d]
        _require(all((r-dr, c-dc) in core for r, c in patch), "tab lacks contour support")
        tabs[len(patch)] = (d, v)
    _require(tab_count == 2 and set(tabs) == {1, 2}, "one single and one double compass tab required")
    _require((tabs[1][0]-tabs[2][0]) % 2 == 1, "tabs must face adjacent sides")
    return {"core": core, "path": path, "classes": classes, "marks": marks,
            "body": next(iter(body_values)), "tabs": tabs}


def _parse(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(np.sum(counts == counts[bg]) == 1, "ambiguous background")
    mask = _structure(a, bg)
    bands = set(map(int, np.unique(a[mask])))
    rectangles, region_index, sides = _regions(a, mask)
    sources, targets = [], []
    for group in _components(map(tuple, np.argwhere((a != bg) & ~mask)), diagonal=True):
        cells = {p: int(a[p]) for p in group}
        if len(set(cells.values())) == 1:
            targets.append(cells)
        else:
            sources.append(cells)
    _require(sources and len(sources) == len(targets), "piece/stencil count mismatch")
    colors = {v for target in targets for v in target.values()}
    _require(len(colors) == 1, "one registration color required")
    reg = next(iter(colors))
    _require(reg not in bands and reg != bg, "registration role collision")
    pieces = [_piece_info(s, reg, bands) for s in sources]
    _require(len({p['body'] for p in pieces}) == 1, "one body color per input required")
    _require(pieces[0]['body'] not in bands | {bg, reg}, "body role collision")
    return {"a": a, "bg": bg, "mask": mask, "bands": bands, "reg": reg,
            "rectangles": rectangles, "index": region_index, "sides": sides,
            "sources": sources, "targets": [set(t) for t in targets], "pieces": pieces}


def _candidates(source, info, compass=True):
    result, seen = [], set()
    for pose in range(8):
        transformed = _transform(source, pose)
        piece = _piece_info(transformed, info['reg'], info['bands'])
        marks = piece['marks']
        for ti, target in enumerate(info['targets']):
            if len(target) != len(marks):
                continue
            r, c = min(marks)
            tr, tc = min(target)
            dr, dc = tr-r, tc-c
            placed = _shift(transformed, dr, dc)
            if {(r+dr, c+dc) for r, c in marks} != target:
                continue
            if any(not (0 <= r < info['a'].shape[0] and 0 <= c < info['a'].shape[1])
                   for r, c in placed):
                continue
            regions = {int(info['index'][p]) for p in placed}
            if len(regions) != 1 or -1 in regions:
                continue
            region = next(iter(regions))
            sides = info['sides'][region]
            if compass and any(sides[d] != color for d, color in piece['tabs'].values()):
                continue
            signature = (ti, tuple(sorted(placed.items())))
            if signature in seen:
                continue
            seen.add(signature)
            result.append((ti, placed, piece, (dr, dc), sides, pose))
    return result


def _dock(info, compass=True, limit=2):
    choices = [_candidates(s, info, compass) for s in info['sources']]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    solutions = {}
    def search(depth, used, occupied, selected):
        if len(solutions) >= limit:
            return
        if depth == len(order):
            signature = tuple(sorted(occupied.items()))
            solutions[signature] = list(selected)
            return
        for candidate in choices[order[depth]]:
            ti, placed = candidate[:2]
            if ti in used or placed.keys() & occupied.keys():
                continue
            search(depth+1, used | {ti}, {**occupied, **placed}, selected+[candidate])
    search(0, set(), {}, [])
    return list(solutions.values())


def _render(info, docked, bends=True):
    result = np.full_like(info['a'], info['bg'])
    result[info['mask']] = info['a'][info['mask']]
    for _, placed, piece, (dr, dc), sides, _ in docked:
        for p, v in placed.items():
            result[p] = v
        if bends:
            labels = {k: sides[d] for k, (d, _) in piece['tabs'].items()}
            for (r, c), kind in piece['classes'].items():
                if kind:
                    result[r+dr, c+dc] = labels[kind]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive bands, pieces, exact rigid docking, then two interior-angle labels."""
    info = _parse(input_grid)
    answers = _dock(info)
    _require(len(answers) == 1, "docking does not determine exactly one placed map")
    return _render(info, answers[0])


def _polygon(vertices):
    cells = []
    for i, (r, c) in enumerate(vertices):
        nr, nc = vertices[(i+1) % len(vertices)]
        if (r == nr) == (c == nc):
            return None
        dr, dc = int(np.sign(nr-r)), int(np.sign(nc-c))
        cells.extend((r+t*dr, c+t*dc) for t in range(abs(nr-r)+abs(nc-c)))
    if len(set(cells)) != len(cells):
        return None
    try:
        _cycle(set(cells))
    except ValueError:
        return None
    return set(cells)


def _mutate(vertices, rng):
    i = int(rng.integers(len(vertices)))
    a, b, c = vertices[i-1], vertices[i], vertices[(i+1) % len(vertices)]
    if rng.random() < .6:
        # Replace one convex or concave vertex by a staircase shoulder.
        la, lc = abs(a[0]-b[0])+abs(a[1]-b[1]), abs(c[0]-b[0])+abs(c[1]-b[1])
        if min(la, lc) < 2:
            return None
        u, v = int(rng.integers(1, la)), int(rng.integers(1, lc))
        p = (b[0]+u*(a[0]-b[0])//la, b[1]+u*(a[1]-b[1])//la)
        q = (b[0]+v*(c[0]-b[0])//lc, b[1]+v*(c[1]-b[1])//lc)
        middle = p[0]+q[0]-b[0], p[1]+q[1]-b[1]
        return vertices[:i]+[p, middle, q]+vertices[i+1:]
    length = abs(c[0]-b[0])+abs(c[1]-b[1])
    if length < 4:
        return None
    u = int(rng.integers(1, length-2))
    v = int(rng.integers(u+2, length))
    depth = int(rng.choice((-2, -1, 1, 2)))
    dr, dc = (c[0]-b[0])//length, (c[1]-b[1])//length
    points = [(b[0]+u*dr, b[1]+u*dc),
              (b[0]+u*dr+depth*dc, b[1]+u*dc-depth*dr),
              (b[0]+v*dr+depth*dc, b[1]+v*dc-depth*dr),
              (b[0]+v*dr, b[1]+v*dc)]
    return vertices[:i+1]+points+vertices[i+1:]


def _tab_options(core, classes, direction, length):
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    supports = []
    for r, c in sorted(core):
        if classes[r, c] != 0:
            continue
        if (direction == 0 and r == r0 or direction == 1 and c == c1
                or direction == 2 and r == r1 or direction == 3 and c == c0):
            supports.append((r, c))
    options = []
    dr, dc = DIRS[direction]
    tangent = (0, 1) if direction % 2 == 0 else (1, 0)
    for r, c in supports:
        patch = {(r+t*tangent[0], c+t*tangent[1]) for t in range(length)}
        if patch <= set(supports):
            options.append({(rr+dr, cc+dc) for rr, cc in patch})
    return options


def _make_piece(rng, sides, body, reg, complexity, reg_length, cap):
    for _ in range(40):
        h, w = int(rng.integers(5, cap+1)), int(rng.integers(5, cap+1))
        vertices = [(0, 0), (0, w-1), (h-1, w-1), (h-1, 0)]
        core = _polygon(vertices)
        for _ in range(55):
            classes = _cycle(core)[1]
            n = sum(v == 2 for v in classes.values())
            if n >= complexity:
                break
            proposal = _mutate(vertices, rng)
            candidate = _polygon(proposal) if proposal else None
            if candidate:
                hh = max(r for r, _ in candidate)-min(r for r, _ in candidate)+1
                ww = max(c for _, c in candidate)-min(c for _, c in candidate)+1
                if max(hh, ww) <= cap and sum(v == 2 for v in _cycle(candidate)[1].values()) > n:
                    vertices, core = proposal, candidate
        classes = _cycle(core)[1]
        if sum(v == 2 for v in classes.values()) < complexity:
            continue
        singles = []
        for d in range(4):
            for e in ((d-1) % 4, (d+1) % 4):
                one, two = _tab_options(core, classes, d, 1), _tab_options(core, classes, e, 2)
                if one and two:
                    singles.append((d, e, one, two))
        if not singles:
            continue
        d, e, one, two = singles[int(rng.integers(len(singles)))]
        tabs = {p: sides[d] for p in one[int(rng.integers(len(one)))]}
        tabs.update({p: sides[e] for p in two[int(rng.integers(len(two)))]})
        # A registration strip can lie on any straight run, including a notch.
        strips = []
        for r, c in sorted(core):
            for dr, dc in ((0, 1), (1, 0)):
                marks = {(r+t*dr, c+t*dc) for t in range(reg_length)}
                if all(classes.get(p) == 0 for p in marks):
                    strips.append(marks)
        if not strips:
            continue
        marks = strips[int(rng.integers(len(strips)))]
        piece = {p: body for p in core}
        piece.update({p: reg for p in marks})
        piece.update(tabs)
        return _transform(piece, 0)
    return None


def _canvas(rng, layout, palette, count):
    bg = palette[0]
    if layout == 'frame':
        low = 19 if count == 1 else 25
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    else:
        h, w = int(rng.integers(26, 31)), int(rng.integers(28, 31))
        if layout == 'horizontal':
            h, w = w, h
    base = np.full((h, w), bg, dtype=np.int8)
    base[:2, :] = palette[3]
    base[-2:, :] = palette[5]
    base[2:-2, :2] = palette[6]
    base[2:-2, -2:] = palette[4]
    if layout == 'vertical':
        middle = int(rng.integers(13, w-12)) if w > 28 else 13
        base[2:-2, middle:middle+2] = palette[7]
    elif layout == 'horizontal':
        middle = int(rng.integers(13, h-12)) if h > 28 else 13
        base[middle:middle+2, 2:-2] = palette[7]
    mask = _structure(base, bg)
    rectangles, index, sides = _regions(base, mask)
    return base, mask, rectangles, index, sides


def _origins(piece, rectangles):
    h, w = max(r for r, _ in piece)+1, max(c for _, c in piece)+1
    return [(r, c) for r0, r1, c0, c1 in rectangles
            for r in range(r0, r1-h+1) for c in range(c0, c1-w+1)]


def _construct(rng, palette, layout, count, complexity, reg_lengths, cap, teaching):
    base, mask, rectangles, _, sides = _canvas(rng, layout, palette, count)
    finals, blocked = [], set()
    for i in range(count):
        found = False
        for _ in range(12):
            region = int(rng.integers(len(rectangles)))
            piece = _make_piece(rng, sides[region], palette[1], palette[2],
                                complexity if i == 0 else int(rng.integers(0, complexity+1)),
                                int(rng.choice(reg_lengths)), cap)
            if piece is None:
                continue
            origins = _origins(piece, [rectangles[region]])
            rng.shuffle(origins)
            for r, c in origins:
                placed = _shift(piece, r, c)
                if not (placed.keys() & blocked):
                    finals.append(placed)
                    blocked.update(_halo(placed))
                    found = True
                    break
            if found:
                break
        if not found:
            return None
    targets = {p: palette[2] for final in finals for p, v in final.items() if v == palette[2]}
    blocked = _halo(targets)
    sources = []
    for i, final in enumerate(finals):
        poses = list(map(int, rng.permutation(8)))
        if i == 0 and teaching is not None:
            poses = [p for p in poses if 0 < p < 4] if teaching == 'rotation' else [p for p in poses if p >= 4]
        found = False
        for pose in poses:
            piece = _transform(final, pose)
            if i == 0 and teaching == 'reflection' and any(
                    _signature(_transform(piece, t)) == _signature(final) for t in range(4)):
                continue
            origins = _origins(piece, rectangles)
            rng.shuffle(origins)
            for r, c in origins:
                placed = _shift(piece, r, c)
                if placed.keys() & blocked or placed == final:
                    continue
                sources.append(placed)
                blocked.update(_halo(placed))
                found = True
                break
            if found:
                break
        if not found:
            return None
    inp = base.copy()
    for p, v in targets.items():
        inp[p] = v
    for piece in sources:
        for p, v in piece.items():
            inp[p] = v
    try:
        info = _parse(inp)
        answers = _dock(info)
        if len(answers) != 1:
            return None
        raw = base.copy()
        for piece in finals:
            for p, v in piece.items():
                raw[p] = v
        if not np.array_equal(_render(info, answers[0], bends=False), raw):
            return None
        if teaching and len(_dock(info, compass=False)) < 2:
            return None
        out = _render(info, answers[0])
    except ValueError:
        return None
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                 and low <= int(value) <= high, name+' outside legal range')
    _require(num_train > num_test, 'num_train must exceed num_test')
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(('count', 'geometry', 'registration')))
    puzzle, used = {'train': [], 'test': []}, set()
    previous_labels = None
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            teaching = ('rotation' if i == 0 else 'reflection' if i == 1 else None) if split == 'train' else None
            pieces = (1 if split == 'train' else int(rng.integers(2, 4))) if regime == 'count' else int(rng.integers(1, 3))
            complexity = (3 if split == 'test' else 1) if regime == 'geometry' else int(rng.integers(1, 3))
            reg_lengths = ((1, 2) if split == 'train' else (3, 4)) if regime == 'registration' else (1, 2, 3)
            layout = str(rng.choice(('frame', 'vertical', 'horizontal')))
            # Richer loops need space; narrow fields still occur in other regimes.
            if regime == 'geometry' and split == 'test':
                layout = 'frame'
            cap = 9 if regime == 'geometry' and split == 'test' else 8
            palette = list(map(int, rng.permutation(10)))
            for _ in range(70):
                pair = _construct(rng, palette, layout, pieces, complexity, reg_lengths, cap, teaching)
                if pair is None or _key(pair['input']) in used:
                    continue
                info = _parse(pair['input'])
                labels = {(p['tabs'][1][1], p['tabs'][2][1]) for p in info['pieces']}
                if split == 'train' and i == 1 and labels == previous_labels:
                    palette = list(map(int, rng.permutation(10)))
                    continue
                previous_labels = labels
                used.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded construction exhausted; no silent rule-changing fallback')
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'invalid split counts')
    seen, train_labels, moves = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape and _key(inp) not in seen, 'shape mismatch or duplicate input')
            seen.add(_key(inp))
            info = _parse(inp)
            answers = _dock(info)
            _require(len(answers) == 1, 'nonunique docking')
            answer = answers[0]
            _require(np.array_equal(out, _render(info, answer)), 'incorrect docking/curvature output')
            _require(np.array_equal(inp[info['mask']], out[info['mask']]), 'band changed')
            raw = _render(info, answer, bends=False)
            _require(not np.array_equal(out, raw), 'parent A has no effect')
            _require(not np.array_equal(inp, out), 'identity example')
            all_occupied = set()
            for _, placed, piece, (dr, dc), sides, _ in answer:
                _require(not (all_occupied & placed.keys()), 'simultaneous overlap')
                all_occupied.update(placed)
                classes = piece['classes']
                _require(sum(v == 1 for v in classes.values()) - sum(v == 2 for v in classes.values()) == 4,
                         'incorrect bend balance')
                for (r, c), kind in classes.items():
                    if kind == 0:
                        _require(out[r+dr, c+dc] == placed[r+dr, c+dc], 'straight cell altered')
                for size, (d, color) in piece['tabs'].items():
                    _require(sides[d] == color, 'compass not matched')
                _require({(r+dr, c+dc) for r, c in piece['marks']} in info['targets'], 'inexact registration')
            allowed = all_occupied | set().union(*(set(s) for s in info['sources'])) | set().union(*info['targets'])
            unchanged = ~info['mask']
            for p in allowed:
                unchanged[p] = False
            _require(np.all(out[unchanged] == info['bg']), 'canvas corruption')
            if split == 'train':
                _require(any(2 in p['classes'].values() for p in info['pieces']), 'training lacks concavity')
                train_labels.append({(p['tabs'][1][1], p['tabs'][2][1]) for p in info['pieces']})
                if len(_dock(info, compass=False)) > 1:
                    moves.add('compass-needed')
                for source in info['sources']:
                    matches = [placed for _, placed, *_ in answer
                               if any(_signature(_transform(source, t)) == _signature(placed) for t in range(8))]
                    for placed in matches:
                        ts = [t for t in range(8) if _signature(_transform(source, t)) == _signature(placed)]
                        moves.add('translation' if 0 in ts else 'rotation' if any(t < 4 for t in ts) else 'reflection')
    _require(len({tuple(sorted(v)) for v in train_labels}) >= 2, 'training needs different label identities')
    _require({'rotation', 'reflection', 'compass-needed'} <= moves, 'missing docking demonstrations')
    return True


def _inside(path, row, col):
    # Independent geometric check of local interior quadrants, not another solver.
    return sum(1 for (r, c), (nr, nc) in zip(path, path[1:]+path[:1])
               if c == nc and min(r, nr) < row < max(r, nr) and c > col) % 2 == 1


def self_test():
    shapes, outlines, stencils, counts, concaves, palettes, layouts = set(), set(), set(), set(), set(), set(), set()
    overlap = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for pair in puzzle['train']+puzzle['test']:
            info = _parse(pair['input'])
            shapes.add(info['a'].shape)
            palettes.add((info['bg'], info['reg'], tuple(sorted(info['bands']))))
            counts.add(len(info['sources']))
            layouts.add(tuple((r1-r0, c1-c0) for r0, r1, c0, c1 in info['rectangles']))
            answer = _dock(info)[0]
            old = set().union(*(set(s) for s in info['sources']))
            overlap += bool(old & set().union(*(set(c[1]) for c in answer)))
            for piece in info['pieces']:
                path = piece['path']
                concaves.add(sum(v == 2 for v in piece['classes'].values()))
                core = {p: 1 for p in piece['core']}
                outlines.add(min(_signature(_transform(core, t)) for t in range(8)))
                stencils.add(len(piece['marks']))
                for (r, c), kind in piece['classes'].items():
                    if kind:
                        quadrants = sum(_inside(path, r+dr, c+dc)
                                        for dr in (-.25, .25) for dc in (-.25, .25))
                        assert quadrants == (1 if kind == 1 else 3)
            assert np.array_equal(solve(pair['input'].copy()), pair['output'])
    assert len(shapes) >= 30 and len(outlines) >= 75 and len(palettes) >= 100
    assert counts >= {1, 2, 3} and stencils >= {1, 2, 3, 4}
    assert concaves >= {0, 1, 2, 3} and len(layouts) >= 30 and overlap > 0
    first, repeat = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in first for x, y in zip(first[s], repeat[s]) for k in x)
    # The complete input/output relation is D4-equivariant and color-equivariant.
    # This checks the band lookup and interior-angle convention together.
    pair = first['train'][0]
    for pose in range(8):
        moved = {}
        for role in ('input', 'output'):
            grid = pair[role]
            cells = _transform({(r, c): int(grid[r, c]) for r, c in np.ndindex(grid.shape)}, pose)
            transformed = np.empty((max(r for r, _ in cells)+1, max(c for _, c in cells)+1), dtype=np.int8)
            for p, color in cells.items():
                transformed[p] = color
            moved[role] = transformed
        assert np.array_equal(solve(moved['input']), moved['output'])
    permutation = np.array([9, 6, 3, 0, 7, 4, 1, 8, 5, 2], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(4,None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': len(HOST_SEEDS), 'pairs': 42*5, 'shapes': len(shapes),
            'canonical_outlines': len(outlines), 'registration_lengths': sorted(stencils),
            'piece_counts': sorted(counts), 'concave_counts': sorted(concaves),
            'palettes': len(palettes), 'region_geometries': len(layouts), 'overlap_pairs': overlap}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
