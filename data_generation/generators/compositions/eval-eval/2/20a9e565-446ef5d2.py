"""A damaged growing picture whose fragments and continuation constrain each other.

Only NumPy and the standard library are used. Labels are drawn before damage and
cutting. Prediction enumerates complete fields and exact occupied-mask tilings.
"""
from functools import lru_cache
from itertools import permutations
import numpy as np


BACKGROUND, ERASED, MARKER = 0, 8, 9


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(name + " out of range")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("invalid ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack = [start]
        for r, c in stack:
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(stack)
    return result


def _external(point, corner):
    r, c = point
    dy = -1 if corner[0] == 'T' else 1
    dx = -1 if corner[1] == 'L' else 1
    return {(r+dy, c), (r, c+dx), (r+dy, c+dx)}


def _extract(g):
    marks = _components(g == MARKER)
    if len(marks) != 2 or any(len(q) != 3 for q in marks):
        raise ValueError("two external Ls required")
    corners = []
    for cells in marks:
        q = np.array(cells)
        lo, hi = q.min(0), q.max(0)
        if tuple(hi-lo) != (1, 1):
            raise ValueError("malformed L")
        square = {(int(r), int(c)) for r in range(lo[0], hi[0]+1)
                  for c in range(lo[1], hi[1]+1)}
        point = (square-set(cells)).pop()
        corner = {(1, 1): 'TL', (1, 0): 'TR', (0, 1): 'BL', (0, 0): 'BR'}[
            (point[0]-int(lo[0]), point[1]-int(lo[1]))]
        corners.append((point, corner))
    if {c for _, c in corners} not in ({'TL', 'BR'}, {'TR', 'BL'}):
        raise ValueError("brackets must oppose")
    r0 = min(p[0] for p, _ in corners)
    r1 = max(p[0] for p, _ in corners)
    c0 = min(p[1] for p, _ in corners)
    c1 = max(p[1] for p, _ in corners)
    for (r, c), corner in corners:
        if (r, c) != (r0 if corner[0]=='T' else r1, c0 if corner[1]=='L' else c1):
            raise ValueError("bracket orientation disagrees with target")
    occupied = [(p, c) for p, c in corners if int(g[p]) not in (BACKGROUND, MARKER)]
    if len(occupied) != 1:
        raise ValueError("exactly one bracket must pin a fragment")
    point, corner = occupied[0]
    pieces, anchor = [], None
    for cells in _components((g != BACKGROUND) & (g != MARKER)):
        q = np.array(cells)
        r, c = map(int, q.min(0))
        end = q.max(0)+1
        m = np.full((int(end[0])-r, int(end[1])-c), -1, dtype=np.int8)
        for y, x in cells:
            m[y-r, x-c] = g[y, x]
        if point in cells:
            anchor = len(pieces)
        pieces.append((m, (r, c)))
    if anchor is None or len(pieces) < 3:
        raise ValueError("missing anchor or fragments")
    h, w = r1-r0+1, c1-c0+1
    if sum(int(np.count_nonzero(m >= 0)) for m, _ in pieces) != h*w:
        raise ValueError("sheet area does not equal target area")
    m, (r, c) = pieces[anchor]
    expected = (r + (len(m)-1 if corner[0]=='B' else 0),
                c + (m.shape[1]-1 if corner[1]=='R' else 0))
    if point != expected or np.any(m < 0) or np.any(m == ERASED):
        raise ValueError("the corner cap must be intact")
    return dict(bounds=(r0, r1, c0, c1), corner=corner, pieces=pieces,
                anchor=anchor, point=point)


def _orient(a, fr, fc, axis):
    b = a[::-1] if fr else a
    b = b[:, ::-1] if fc else b
    return (b.T if axis else b).copy()


def _canonical(info, canvas, axis):
    fr, fc = info['corner'][0]=='B', info['corner'][1]=='R'
    r0, r1, c0, c1 = info['bounds']
    origin = (canvas[0]-1-r1 if fr else r0, canvas[1]-1-c1 if fc else c0)
    shape = (r1-r0+1, c1-c0+1)
    if axis:
        origin, shape = origin[::-1], shape[::-1]
    pieces = []
    for m, (r, c) in info['pieces']:
        pos = (canvas[0]-r-len(m) if fr else r,
               canvas[1]-c-m.shape[1] if fc else c)
        if axis:
            pos = pos[::-1]
        pieces.append((_orient(m, fr, fc, axis),
                       (pos[0]-origin[0], pos[1]-origin[1])))
    return shape, pieces


def _field(h, w, carrier, palette, b, step, initial, delta, style, spacing):
    """Input-side interpretation by cell predicates, not the label painter."""
    gap = step-b
    if gap < 1 or (w+gap) % step:
        return None
    n = (w+gap)//step
    heights = initial + delta*np.arange(n)
    if n < 3 or heights.min() < 3 or heights.max() > h:
        return None
    if style != 'solid' and b < 3:
        return None
    y, x = np.indices((h, w))
    k, within = x//step, x % step
    height = heights[k]
    mask = (within < b) & (y < height)
    if style == 'outline':
        mask &= (y == 0) | (y == height-1) | (within == 0) | (within == b-1)
    elif style == 'ladder':
        mask &= (y % spacing == 0) | (y == height-1) | (within == 0) | (within == b-1)
    a = np.full((h, w), carrier, dtype=np.int8)
    colors = np.asarray(palette, dtype=np.int8)
    a[mask] = colors[(k % len(palette))[mask]]
    return a


def _models(shape, anchor, colors):
    h, w = shape
    for carrier in colors:
        motif_colors = set(colors)-{carrier}
        if not 2 <= len(motif_colors) <= 4:
            continue
        row = anchor[0]
        active = np.flatnonzero(row != carrier)
        if len(active) == 0 or active[0] != 0:
            continue
        runs = np.split(active, np.flatnonzero(np.diff(active)>1)+1)
        if len(runs) < 2:
            continue
        b, step = len(runs[0]), int(runs[1][0])
        if b < 2 or step <= b:
            continue
        if any(len(run) != b or np.unique(row[run]).size != 1 for run in runs):
            continue
        first = [int(row[run[0]]) for run in runs]
        if first[0] == first[1] or carrier in first:
            continue
        gap = step-b
        if (w+gap) % step:
            continue
        n = (w+gap)//step
        if n < len(motif_colors)+1:
            continue
        remaining = sorted(motif_colors-set(first[:2]))
        for suffix in permutations(remaining):
            palette = tuple(first[:2])+suffix
            if any(v != palette[i % len(palette)] for i, v in enumerate(first)):
                continue
            for initial in range(3, h+1):
                low = -((initial-3)//(n-1))
                high = (h-initial)//(n-1)
                for delta in range(low, high+1):
                    for style, spacing in (('solid', 1), ('outline', 1), ('ladder', 2), ('ladder', 3)):
                        a = _field(h, w, carrier, palette, b, step, initial, delta, style, spacing)
                        if a is not None and anchor.shape[0] <= h and anchor.shape[1] <= w:
                            if np.array_equal(a[:len(anchor), :anchor.shape[1]], anchor):
                                yield a, dict(carrier=carrier, palette=palette, b=b, step=step,
                                              initial=initial, delta=delta, style=style,
                                              spacing=spacing, slots=n)


class _SearchLimit(ValueError):
    pass


@lru_cache(maxsize=384)
def _geometry_counterexample(shape, piece_keys, anchor, field_bytes):
    """A shape-valid packing that the known motif pixels rule out.

    This is a concrete interaction witness, not a count of generation flags.
    Search stops at the first witness, so it need not enumerate all packings.
    """
    h, w = shape
    field = np.frombuffer(field_bytes, dtype=np.int8).reshape(shape)
    matrices = [np.frombuffer(data, dtype=np.int8).reshape(s) for s, data in piece_keys]
    configs = []
    for m in matrices:
        cells = list(map(tuple, np.argwhere(m >= 0).tolist()))
        first = cells[0][1]
        bit = sum(1 << (r*w+c) for r, c in cells)
        configs.append((m.shape, first, bit))
    full = (1 << (h*w))-1
    positions = [None]*len(matrices)
    positions[anchor] = (0, 0)
    nodes = 0

    def visit(occ, remaining):
        nonlocal nodes
        nodes += 1
        if nodes > 40000:
            raise _SearchLimit("incomplete geometry witness search")
        if not remaining:
            damaged = np.full(shape, ERASED, dtype=np.int8)
            for m, (r, c) in zip(matrices, positions):
                mask = m >= 0
                damaged[r:r+len(m), c:c+m.shape[1]][mask] = m[mask]
            known = damaged != ERASED
            return np.any(damaged[known] != field[known])
        free = full ^ occ
        first_free = (free & -free).bit_length()-1
        r, c = divmod(first_free, w)
        for i in remaining:
            (ph, pw), first, bit = configs[i]
            x = c-first
            if x < 0 or r+ph > h or x+pw > w:
                continue
            placed = bit << (r*w+x)
            if not (placed & occ):
                positions[i] = (r, x)
                if visit(occ | placed, tuple(j for j in remaining if j != i)):
                    return True
        return False

    return bool(visit(configs[anchor][2], tuple(i for i in range(len(matrices)) if i != anchor)))


def _tile(field, pieces, anchor):
    """Find an exact occupied-mask tiling for one fully specified picture."""
    h, w = field.shape
    domains = []
    for i, (m, _) in enumerate(pieces):
        ph, pw = m.shape
        if ph > h or pw > w:
            return None
        known = (m >= 0) & (m != ERASED)
        rr, cc = np.where(known)
        windows = np.lib.stride_tricks.sliding_window_view(field, (ph, pw))
        matches = np.all(windows[..., rr, cc] == m[rr, cc], axis=-1)
        if i == anchor:
            positions = [(0, 0)] if matches[0, 0] else []
        else:
            positions = list(map(tuple, np.argwhere(matches).tolist()))
        cells = list(map(tuple, np.argwhere(m >= 0).tolist()))
        base = sum(1 << (r*w+c) for r, c in cells)
        ds = [(r, c, base << (r*w+c)) for r, c in positions]
        if not ds:
            return None
        domains.append(ds)
    initial = domains[anchor][0]
    placed = [None]*len(pieces)
    placed[anchor] = initial[:2]
    remaining = ((1 << len(pieces))-1) ^ (1 << anchor)
    full = (1 << (h*w))-1
    failed = set()
    nodes = 0

    def visit(occ, todo):
        nonlocal nodes
        nodes += 1
        if nodes > 40000:
            raise _SearchLimit("incomplete packing search")
        if not todo:
            return occ == full
        key = (occ, todo)
        if key in failed:
            return False
        best, options, union = None, None, occ
        for i, ds in enumerate(domains):
            if not (todo >> i) & 1:
                continue
            valid = [p for p in ds if not p[2] & occ]
            if not valid:
                failed.add(key)
                return False
            for p in valid:
                union |= p[2]
            if options is None or len(valid) < len(options):
                best, options = i, valid
        if union != full:
            failed.add(key)
            return False
        for r, c, bit in options:
            placed[best] = (r, c)
            if visit(occ | bit, todo ^ (1 << best)):
                return True
        failed.add(key)
        return False

    return tuple(placed) if visit(initial[2], remaining) else None


@lru_cache(maxsize=256)
def _analyse_cached(shape, data):
    # This bounded memoization contains only computations keyed by the complete
    # input. Generation never registers seeds, labels, or example dictionaries.
    g = np.frombuffer(data, dtype=np.int8).reshape(shape)
    info = _extract(g)
    colors = tuple(int(c) for c in np.unique(g) if 1 <= c <= 7)
    winner = None
    keys = set()
    cap_alternatives = set()
    for axis in (0, 1):
        target, pieces = _canonical(info, shape, axis)
        if pieces[info['anchor']][1] != (0, 0):
            raise ValueError("anchor is not at the specified target corner")
        anchor = pieces[info['anchor']][0]
        tried = set()
        for field, model in _models(target, anchor, colors):
            key = field.tobytes()
            hypothetical = field.T if axis else field
            hypothetical = _orient(hypothetical, info['corner'][0]=='B', info['corner'][1]=='R', 0)
            cap_alternatives.add(hypothetical.tobytes())
            if key in tried:
                continue
            tried.add(key)
            placements = _tile(field, pieces, info['anchor'])
            if placements is None:
                continue
            # Undo transposition first, then the two reflections.
            answer = field.T if axis else field
            answer = _orient(answer, info['corner'][0]=='B', info['corner'][1]=='R', 0)
            output_key = answer.tobytes()
            keys.add(output_key)
            if len(keys) > 1:
                raise ValueError("distinct growing pictures fit the fragments")
            if winner is None:
                winner = (answer, info, axis, field, model, pieces, placements)
    if winner is None:
        raise ValueError("no growth-and-tiling explanation")
    return winner + (len(cap_alternatives),)


def _analyse(g):
    a = _grid(g)
    return _analyse_cached(a.shape, a.tobytes())


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique completed crop solely from the given input."""
    return _analyse(input_grid)[0].copy()


def _paint(h, w, carrier, palette, b, step, initial, delta, style, spacing):
    """Generation-side stroke drawing, before erasure, partition, or scatter."""
    picture = np.full((h, w), carrier, dtype=np.int8)
    for k, x in enumerate(range(0, w, step)):
        length, color = initial+k*delta, palette[k % len(palette)]
        if style == 'solid':
            picture[:length, x:x+b] = color
        else:
            picture[:length, x] = color
            picture[:length, x+b-1] = color
            picture[0, x:x+b] = color
            picture[length-1, x:x+b] = color
            if style == 'ladder':
                for y in range(0, length, spacing):
                    picture[y, x:x+b] = color
    return picture


def _partition(rng, h, w, ah, aw, count, stepped):
    rects = [(0, 0, ah, aw)]
    if rng.integers(2):
        rects += [(0, aw, h, w-aw), (ah, 0, h-ah, aw)]
    else:
        rects += [(0, aw, ah, w-aw), (ah, 0, h-ah, w)]
    while len(rects) < count:
        options = [(i, axis) for i, (_, _, ph, pw) in enumerate(rects) if i
                   for axis, size in enumerate((ph, pw)) if size >= 4]
        weights = np.array([rects[i][2]*rects[i][3] for i, _ in options], dtype=float)
        i, axis = options[int(rng.choice(len(options), p=weights/weights.sum()))]
        r, c, ph, pw = rects.pop(i)
        cut = int(rng.integers(2, (ph if axis==0 else pw)-1))
        if axis == 0:
            rects += [(r, c, cut, pw), (r+cut, c, ph-cut, pw)]
        else:
            rects += [(r, c, ph, cut), (r, c+cut, ph, pw-cut)]
    labels = np.full((h, w), -1, dtype=np.int8)
    for i, (r, c, ph, pw) in enumerate(rects):
        labels[r:r+ph, c:c+pw] = i
    if stepped:
        options = []
        for r in range(h):
            for c in range(w):
                donor = int(labels[r, c])
                if donor == 0:
                    continue
                for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if 0 <= y < h and 0 <= x < w:
                        recipient = int(labels[y, x])
                        if recipient and recipient != donor:
                            options.append((r, c, donor, recipient))
        for ix in rng.permutation(len(options)):
            r, c, donor, recipient = options[int(ix)]
            if labels[r, c] != donor:
                continue
            before = labels.copy()
            labels[r, c] = recipient
            if any(len(_components(labels==i)) != 1 or np.count_nonzero(labels==i) < 4
                   for i in (donor, recipient)):
                labels = before
                continue
            if any(_notch_count(labels==i) for i in (donor, recipient)):
                break
            labels = before
    return labels


def _notch_count(mask):
    q = np.argwhere(mask)
    if len(q) == 0:
        return 0
    return int(np.prod(q.max(0)-q.min(0)+1)-len(q))


def _cut(labels, damaged):
    pieces = []
    for i in range(int(labels.max())+1):
        q = np.argwhere(labels == i)
        r, c = map(int, q.min(0))
        y, x = map(int, q.max(0)+1)
        m = np.where(labels[r:y, c:x] == i, damaged[r:y, c:x], -1).astype(np.int8)
        pieces.append((m, (r, c)))
    return pieces


def _gap(a, b):
    r, c, h, w = a
    y, x, ph, pw = b
    return r+h < y or y+ph < r or c+w < x or x+pw < c


def _scatter(rng, pieces, origin, corner, target, canvas):
    h, w = target
    point = (origin[0]+(h-1 if corner[0]=='B' else 0),
             origin[1]+(w-1 if corner[1]=='R' else 0))
    opposite = ('B' if corner[0]=='T' else 'T')+('R' if corner[1]=='L' else 'L')
    other = (origin[0]+(h-1 if opposite[0]=='B' else 0),
             origin[1]+(w-1 if opposite[1]=='R' else 0))
    marks = _external(point, corner) | _external(other, opposite)
    anchor_pos = (origin[0]+pieces[0][1][0], origin[1]+pieces[0][1][1])
    order = sorted(range(1, len(pieces)), key=lambda i: -pieces[i][0].size)
    for _ in range(8):
        positions = [None]*len(pieces)
        positions[0] = anchor_pos
        rects = [(*anchor_pos, *pieces[0][0].shape)]
        for i in order:
            ph, pw = pieces[i][0].shape
            options = [(r, c) for r in range(canvas[0]-ph+1)
                       for c in range(canvas[1]-pw+1)]
            for k in rng.permutation(len(options)):
                r, c = options[int(k)]
                if any(r-1 <= y <= r+ph and c-1 <= x <= c+pw for y, x in marks):
                    continue
                if r <= other[0] < r+ph and c <= other[1] < c+pw:
                    continue
                if all(_gap((r, c, ph, pw), old) for old in rects):
                    positions[i] = (r, c)
                    rects.append((r, c, ph, pw))
                    break
            if positions[i] is None:
                break
        if all(p is not None for p in positions):
            grid = np.zeros(canvas, dtype=np.int8)
            for (m, _), (r, c) in zip(pieces, positions):
                sub = grid[r:r+len(m), c:c+m.shape[1]]
                sub[m >= 0] = m[m >= 0]
            for p in marks:
                grid[p] = MARKER
            return grid
    return None


def _make_pair(rng, policy, sign, axis, erased_count, stepped):
    style = policy['style']
    p = int(rng.choice([2, 3, 4], p=[.45, .45, .10]))
    observed = p+1 + (int(rng.integers(2)) if p==2 else 0)
    n = observed+erased_count
    b = int(rng.integers(2 if style=='solid' else 3, 5))
    gap = int(rng.integers(1, 3))
    if n*(b+gap)-gap > 28:
        b, gap = (2 if style=='solid' else 3), 1
    if n*(b+gap)-gap > 28:
        return None
    step, w = b+gap, n*(b+gap)-gap
    magnitude = int(rng.choice([1, 1, 1, 2]))
    minimum = int(rng.integers(3 if style=='solid' else 4, 7))
    heights = [minimum+k*magnitude for k in range(n)]
    if sign < 0:
        heights.reverse()
    h = max(heights)+int(rng.integers(1, 4))
    if h > 23 or h*w > 400:
        return None
    colors = list(policy['roles']) if policy['reuse'] else list(map(int, rng.permutation(np.arange(1, 8))))
    carrier = colors[0]
    palette = tuple(colors[1:1+p])
    phase = int(rng.integers(p))
    palette = palette[phase:]+palette[:phase]
    spacing = int(rng.integers(2, 4))
    truth = _paint(h, w, carrier, palette, b, step, heights[0], sign*magnitude, style, spacing)
    damaged = truth.copy()
    damaged[:, observed*step:] = ERASED
    ah, aw = 2, step+b
    if rng.random() < .45:
        for _ in range(int(rng.integers(1, 4))):
            r = int(rng.integers(ah, h))
            c = int(rng.integers(observed*step))
            damaged[r:min(h, r+int(rng.integers(1, 3))),
                    c:min(observed*step, c+int(rng.integers(1, 3)))] = ERASED
    count = int(rng.integers(4, 9))
    labels = _partition(rng, h, w, ah, aw, count, stepped)
    if stepped and not any(_notch_count(labels==i) for i in range(count)):
        return None
    fr, fc = bool(rng.integers(2)), bool(rng.integers(2))
    truth, damaged, labels = [_orient(a, fr, fc, axis) for a in (truth, damaged, labels)]
    h, w = truth.shape
    corners = {'TL': (0, 0), 'TR': (0, w-1), 'BL': (h-1, 0), 'BR': (h-1, w-1)}
    corner = next(c for c, point in corners.items() if labels[point] == 0)
    pieces = _cut(labels, damaged)
    area = h*w
    ch = min(30, max(h+2, int(np.ceil(np.sqrt(2.1*area)))+int(rng.integers(2, 5))))
    cw = min(30, max(w+2, int(np.ceil(2.1*area/ch))+int(rng.integers(2, 5))))
    for canvas in ((ch, cw), (30, 30)):
        for _ in range(3):
            origin = (int(rng.integers(1, canvas[0]-h)), int(rng.integers(1, canvas[1]-w)))
            grid = _scatter(rng, pieces, origin, corner, (h, w), canvas)
            if grid is None:
                continue
            try:
                result = _analyse(grid)
                features = _check_semantics(grid, truth, result)
            except ValueError:
                continue
            if features['hidden'] != erased_count:
                continue
            if np.array_equal(result[0], truth):
                return {'input': grid, 'output': truth.copy()}
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ne = _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = int(rng.integers(3))
    train_axis = int(rng.integers(2))
    policy = dict(style=str(rng.choice(['solid', 'outline', 'ladder'])),
                  roles=tuple(map(int, rng.permutation(np.arange(1, 8)))),
                  reuse=bool(rng.integers(2)))
    broad = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            sign = (1 if i==0 else -1 if i==1 else int(rng.choice([-1, 1]))) if split=='train' else int(rng.choice([-1, 1]))
            axis = (1-train_axis if split=='test' else train_axis) if shift in (1, 2) else int(rng.integers(2))
            erased = int(rng.integers(2, 4)) if split=='test' and shift in (0, 2) else 1
            stepped = broad and (i==1 or (split=='test') or rng.random()<.35)
            for attempt in range(100):
                pair = _make_pair(rng, policy, sign, axis, erased, stepped)
                if pair is None:
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
    order = rng.permutation(nt)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    return puzzle


def _check_semantics(g, output, result=None):
    answer, info, axis, field, model, pieces, placements, cap_count = result or _analyse(g)
    if not np.array_equal(output, answer):
        raise ValueError("incorrect completed picture")
    if cap_count < 2:
        raise ValueError("anchor alone determines the picture")
    piece_keys = tuple((m.shape, m.tobytes()) for m, _ in pieces)
    if not _geometry_counterexample(field.shape, piece_keys, info['anchor'], field.tobytes()):
        raise ValueError("pattern does not distinguish a shape-valid alternative")
    h, w = field.shape
    cover = np.zeros((h, w), dtype=np.int8)
    known = np.zeros((h, w), dtype=bool)
    erased = np.zeros((h, w), dtype=bool)
    moved, notches = 0, 0
    for i, ((m, old), (r, c)) in enumerate(zip(pieces, placements)):
        ph, pw = m.shape
        mask = m >= 0
        cover[r:r+ph, c:c+pw][mask] += 1
        k = mask & (m != ERASED)
        if not np.array_equal(field[r:r+ph, c:c+pw][k], m[k]):
            raise ValueError("known pixels changed")
        known[r:r+ph, c:c+pw][k] = True
        erased[r:r+ph, c:c+pw][m == ERASED] = True
        moved += int(i != info['anchor'] and old != (r, c))
        notches += int(np.count_nonzero(mask) < mask.size)
    if not np.all(cover == 1) or placements[info['anchor']] != (0, 0) or not moved:
        raise ValueError("invalid occupied-mask assembly")
    if np.any((answer < 1) | (answer > 7)) or not np.any(erased):
        raise ValueError("uncompleted material")
    carrier = model['carrier']
    if not np.any(erased & (field==carrier)) or not np.any(erased & (field!=carrier)):
        raise ValueError("erasures must require both carrier and motif completion")
    slots_seen, hidden_slots = 0, 0
    for x in range(0, w, model['step']):
        slot_ink = field[:, x:x+model['b']] != carrier
        visible = known[:, x:x+model['b']] & slot_ink
        slots_seen += int(visible.any())
        hidden_slots += int(not visible.any())
    if slots_seen < len(model['palette'])+1 or hidden_slots < 1 or model['delta'] == 0:
        raise ValueError("insufficient growth/phase/forecast evidence")
    const = _field(h, w, carrier, model['palette'], model['b'], model['step'],
                   model['initial'], 0, model['style'], model['spacing'])
    if const is not None and np.array_equal(const[erased], field[erased]):
        raise ValueError("constant-height shortcut survives")
    r0, r1, c0, c1 = info['bounds']
    if np.array_equal(g[r0:r1+1, c0:c1+1], answer):
        raise ValueError("scattered crop already equals answer")
    return dict(axis=axis, corner=info['corner'], style=model['style'], delta=model['delta'],
                slots=model['slots'], seen=slots_seen, hidden=hidden_slots,
                cycle=len(model['palette']), pieces=len(pieces), notches=notches,
                shape=answer.shape, canvas=g.shape, cap_alternatives=cap_count,
                geometric_counterexample=True)


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("invalid puzzle keys")
    if any(not isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _count(nt, 'num_train', 2, 10)
    _count(ne, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError("invalid split sizes")
    seen, metrics = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError("invalid pair keys")
            g, out = _grid(pair['input']), _grid(pair['output'])
            key = (g.shape, g.tobytes())
            if key in seen:
                raise ValueError("duplicate inputs")
            seen.add(key)
            metrics[split].append(_check_semantics(g, out))
    if not any(m['delta'] > 0 for m in metrics['train']) or not any(m['delta'] < 0 for m in metrics['train']):
        raise ValueError("training must distinguish both signed growth directions")
    if any(m['notches'] for m in metrics['test']) and not any(m['notches'] for m in metrics['train']):
        raise ValueError("steps in test require a stepped demonstration")


def _key(puzzle):
    return tuple((s, tuple((_grid(p['input']).shape, _grid(p['input']).tobytes(),
                           _grid(p['output']).shape, _grid(p['output']).tobytes())
                          for p in puzzle[s])) for s in ('train', 'test'))


def self_test():
    metrics, fingerprints, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                g = pair['input']
                metrics.append(_check_semantics(g, pair['output']))
                rename = {}
                structures.add((g.shape, tuple(rename.setdefault(int(v), len(rename)) for v in g.flat)))
    assert len(fingerprints) == 42 and len(structures) >= 150
    assert {m['corner'] for m in metrics} == {'TL', 'TR', 'BL', 'BR'}
    assert {m['style'] for m in metrics} >= {'solid', 'outline', 'ladder'}
    assert {m['axis'] for m in metrics} == {0, 1}
    assert {m['cycle'] for m in metrics} >= {2, 3, 4}
    assert len({m['pieces'] for m in metrics}) >= 4
    assert len({m['shape'] for m in metrics}) >= 40
    assert sum(m['notches'] > 0 for m in metrics) >= 20
    assert sum(m['hidden'] >= 2 for m in metrics) >= 10
    assert all(m['cap_alternatives'] >= 2 for m in metrics)
    assert all(m['geometric_counterexample'] for m in metrics)
    # Numeric color order must not choose the carrier or cyclic phase.
    renaming = np.array([0, 7, 6, 5, 4, 3, 2, 1, 8, 9], dtype=np.int8)
    pair = puzzle['train'][0]
    assert np.array_equal(solve(renaming[pair['input']]), renaming[pair['output']])
    assert _key(generate_puzzle(93, 2, 1)) == _key(generate_puzzle(93, 2, 1))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.2), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(42, 2, 1)
    damaged['test'][0]['output'][0, 0] = ERASED
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
