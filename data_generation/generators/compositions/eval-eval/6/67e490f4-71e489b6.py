"""Consensus halos: negative-space footprints vote with their host colors.

The solver reads only pixels. Construction uses explicit clean masks; the solver
instead reconstructs a rectilinear color scaffold from long edges on both axes.
No parent generator, source pair, seed, or stored output is used for prediction.
"""
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np


CONFLICT = 9
_ORTHO = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MOORE = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
               if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _normal(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _rotate(cells):
    return _normal((c, -r) for r, c in cells)


def _reflect(cells):
    return _normal((r, -c) for r, c in cells)


def _rotations(cells):
    s, result = _normal(cells), set()
    for _ in range(4):
        result.add(s)
        s = _rotate(s)
    return tuple(sorted(result))


def _variants(cells):
    return tuple(sorted(set(_rotations(cells)) | set(_rotations(_reflect(cells)))))


def _canonical(cells):
    return min(_variants(cells))


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    offsets = _MOORE if diagonal else _ORTHO
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], []
        while stack:
            r, c = stack.pop()
            group.append((int(r), int(c)))
            for dr, dc in offsets:
                q = r+dr, c+dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        groups.append(tuple(sorted(group)))
    return groups


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[r:r+h, c:c+w]
                                 for r in range(3) for c in range(3)])


def _cuts(a, length):
    """Coordinates backed by a straight run of one ordered color transition."""
    old, new = a[:-1], a[1:]
    supported = np.zeros(a.shape[0]-1, dtype=bool)
    for c in range(a.shape[1]-length+1):
        before, after = old[:, c:c+length], new[:, c:c+length]
        supported |= ((before == before[:, :1]).all(axis=1)
                      & (after == after[:, :1]).all(axis=1)
                      & (before[:, 0] != after[:, 0]))
    return tuple([0] + (np.flatnonzero(supported)+1).tolist() + [a.shape[0]])


def _partition(a, yy, xx):
    clean = np.empty_like(a)
    for y, v in zip(yy, yy[1:]):
        for x, u in zip(xx, xx[1:]):
            counts = np.bincount(a[y:v, x:u].ravel(), minlength=10)
            best = np.flatnonzero(counts == counts.max())
            if len(best) != 1:
                return None
            clean[y:v, x:u] = best[0]
    return clean


def _corners(a):
    # Symmetric over colors; straight edges cost nothing, turns cost equally.
    count = 0
    for color in np.unique(a):
        p = np.pad((a == color).astype(np.int16), 1)
        count += int(np.abs(p[1:, 1:]-p[:-1, 1:]
                            -p[1:, :-1]+p[:-1, :-1]).sum())
    return count / 2


@lru_cache(maxsize=256)
def _recover_cached(shape, pixels):
    a = np.frombuffer(pixels, dtype=np.int8).reshape(shape)
    field = int(a[0, 0])
    work = a.copy()
    # Genuine enclosed field cavities have at least 25 cells. Small enclosed
    # gaps surrounded by one host are damage, including long thin footprints.
    # This visible scale separation avoids calling a six-cell bar a new cavity.
    for group in _components(a == field, False):
        if len(group) > 7 or any(r in (0, shape[0]-1) or c in (0, shape[1]-1)
                                for r, c in group):
            continue
        neighbors = {int(a[r+dr, c+dc]) for r, c in group for dr, dc in _ORTHO
                     if 0 <= r+dr < shape[0] and 0 <= c+dc < shape[1]
                     and int(a[r+dr, c+dc]) != field}
        if len(neighbors) == 1:
            color = neighbors.pop()
            for q in group:
                work[q] = color
    rows = sorted({_cuts(work, k) for k in (2, 3, 4, 5)})
    cols = sorted({_cuts(work.T, k) for k in (2, 3, 4, 5)})
    candidates = {}
    for yy in rows:
        for xx in cols:
            b = _partition(work, yy, xx)
            if b is not None:
                candidates[b.tobytes()] = b
    if not candidates:
        raise ValueError("no untied rectilinear reconstruction")
    records = [(int(np.count_nonzero(b != a)), _corners(b), raw)
               for raw, b in candidates.items()]
    records.sort(key=lambda t: (t[0]+1.6*t[1], t[2]))
    error, turns, chosen = records[0]
    for penalty in (1.35, 1.6, 1.85):
        if any(e+penalty*t <= error+penalty*turns+.15
               for e, t, raw in records[1:] if raw != chosen):
            raise ValueError("competing scaffold interpretations")
    if error > max(12, a.size//5):
        raise ValueError("too little intact scaffold")
    return chosen


def _recover(a):
    return np.frombuffer(_recover_cached(a.shape, a.tobytes()),
                         dtype=np.int8).reshape(a.shape).copy()


def _shape_key(cells, equivalence):
    raw = _normal(cells)
    if equivalence == "area":
        return len(raw)
    if equivalence == "box":
        return tuple(sorted((max(r for r, _ in raw)+1,
                             max(c for _, c in raw)+1)))
    if equivalence == "rotation":
        return min(_rotations(raw))
    if equivalence == "translation":
        return raw
    return _canonical(raw)


def _analyze(input_grid, equivalence="free"):
    a = _grid(input_grid)
    field = int(a[0, 0])
    if field == CONFLICT or CONFLICT in a:
        raise ValueError("9 is reserved for output conflicts")
    if any(int(a[r, c]) != field for r, c in
           ((0, 0), (0, -1), (-1, 0), (-1, -1))):
        raise ValueError("four corners must identify the field")
    base = _recover(a)
    if any(int(base[r, c]) != field for r, c in
           ((0, 0), (0, -1), (-1, 0), (-1, -1))):
        raise ValueError("scaffold does not preserve the field corners")
    colors = set(map(int, np.unique(base))) - {field}
    if len(colors) < 2:
        raise ValueError("at least two host colors are required")
    for color in colors:
        if any(len(g) < 25 for g in _components(base == color, False)):
            raise ValueError("intended colored regions must be substantial")
    for cells in _components(base == field, False):
        if not any(r in (0, a.shape[0]-1) or c in (0, a.shape[1]-1) for r, c in cells):
            extent = (max(r for r, _ in cells)-min(r for r, _ in cells)+1,
                      max(c for _, c in cells)-min(c for _, c in cells)+1)
            if len(cells) < 25 or min(extent) < 5:
                raise ValueError("genuine enclosed cavities must satisfy the larger scale")
    wrong = (base != field) & (a != field) & (a != base)
    if wrong.any():
        raise ValueError("corruption may only flip between host and field")
    missing = (base != field) & (a == field)
    extras = (base == field) & (a != field)
    groups = _components(missing, diagonal=equivalence != "orthogonal")
    if not groups:
        raise ValueError("a consensus scene needs missing footprints")
    ballots = defaultdict(Counter)
    host_colors, signatures, halos = [], [], []
    for cells in groups:
        if len(cells) > 7:
            raise ValueError("a missing footprint must contain at most seven cells")
        host = {int(base[q]) for q in cells}
        if len(host) != 1:
            raise ValueError("a footprint must interrupt exactly one host color")
        color = host.pop()
        signature = _shape_key(cells, equivalence)
        ballots[signature][color] += 1
        host_colors.append(color)
        signatures.append(signature)
        mask = np.zeros(a.shape, dtype=bool)
        for q in cells:
            mask[q] = True
        halos.append(_dilate(mask) & ~missing)
    winners = {}
    for signature, counts in ballots.items():
        high = max(counts.values())
        best = [c for c, n in counts.items() if n == high]
        if len(best) != 1:
            raise ValueError("each footprint class must have a unique plurality")
        winners[signature] = best[0]
    return dict(input=a, field=field, base=base, missing=missing, extras=extras,
                groups=groups, hosts=host_colors, signatures=signatures,
                ballots=dict(ballots), winners=winners, halos=halos)


def _prediction(data, mode="consensus"):
    result = data["input"].copy()
    result[data["extras"]] = data["field"]
    inks = np.zeros((10,) + result.shape, dtype=bool)
    for i, (signature, halo) in enumerate(zip(data["signatures"], data["halos"])):
        color = (data["hosts"][i] if mode == "local" else CONFLICT
                 if mode == "constant" else data["winners"][signature])
        inks[color] |= halo
    owners = inks.sum(axis=0)
    if mode == "fill":
        for signature, cells in zip(data["signatures"], data["groups"]):
            for q in cells:
                result[q] = data["winners"][signature]
        return result
    for color in range(10):
        result[(owners == 1) & inks[color]] = color
    result[owners > 1] = CONFLICT
    result[data["missing"]] = data["field"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover, count negative-space shape votes, erase specks, paint halos."""
    return _prediction(_analyze(input_grid))


def _choice(rng, items):
    return items[int(rng.integers(len(items)))]


def _pool(rng):
    shapes = set(_small_catalog()) | {((0, 0), (1, 1)),
                                      ((0, 0), (1, 1), (2, 2))}
    for length in range(3, 7):
        shapes.add(_canonical(tuple((0, c) for c in range(length))))
    for height in range(2, 5):
        for width in range(2, 5):
            cells = {(r, 0) for r in range(height)} | {(height-1, c) for c in range(width)}
            if len(cells) <= 7:
                shapes.add(_canonical(cells))
    for _ in range(60):
        cells = {(0, 0)}
        for _ in range(int(rng.integers(3, 7))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _ORTHO} - cells)
            cells.add(_choice(rng, frontier))
        s = _canonical(cells)
        if max(max(r for r, _ in s), max(c for _, c in s)) <= 3:
            shapes.add(s)
    return tuple(sorted(shapes))


@lru_cache(maxsize=1)
def _small_catalog():
    """A guaranteed vocabulary larger than any legal training schedule."""
    level, all_shapes = {((0, 0),)}, {((0, 0),)}
    for _ in range(5):
        following = set()
        for shape in sorted(level):
            cells = set(shape)
            frontier = {(r+dr, c+dc) for r, c in cells for dr, dc in _ORTHO} - cells
            for point in sorted(frontier):
                s = _canonical(cells | {point})
                if max(max(r for r, _ in s), max(c for _, c in s)) <= 3:
                    following.add(s)
        all_shapes.update(following)
        level = following
    return tuple(sorted(all_shapes))


# Same area and bounding box, but different free shapes. First member is chiral.
# Both pairs use diagonal connections whose four-connected pieces would mix
# differently colored classes. Thus four-neighbor voting is visibly refuted.
_TEACHING_PAIRS = (
    (((0, 0), (0, 1), (1, 2), (2, 1)),
     ((0, 0), (0, 1), (1, 1), (2, 2))),
    (((0, 0), (0, 1), (1, 1), (2, 0), (2, 2)),
     ((0, 0), (0, 1), (0, 2), (1, 0), (2, 1))),
)


def _positions(available, shape):
    hh = max(r for r, _ in shape)+1
    ww = max(c for _, c in shape)+1
    h, w = available.shape
    if hh > h or ww > w:
        return []
    valid = np.ones((h-hh+1, w-ww+1), dtype=bool)
    for r, c in shape:
        valid &= available[r:r+valid.shape[0], c:c+valid.shape[1]]
    return np.argwhere(valid)


def _place(available, variants, rng, edge=False):
    h, w = available.shape
    for index in rng.permutation(len(variants)):
        s = variants[int(index)]
        starts = _positions(available, s)
        if edge and len(starts):
            starts = np.asarray([p for p in starts
                                 if any(p[0]+r in (0, h-1) or p[1]+c in (0, w-1)
                                        for r, c in s)])
        if len(starts):
            y, x = map(int, starts[int(rng.integers(len(starts)))])
            return tuple((y+r, x+c) for r, c in s)
    return None


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False


def _joint(available, a, b, rng, edge=False):
    # A blank column separates the footprints while their Moore halos meet.
    for va in rng.permutation(len(_variants(a))):
        sa = _variants(a)[int(va)]
        for vb in rng.permutation(len(_variants(b))):
            sb = _variants(b)[int(vb)]
            width = max(c for _, c in sa)+1
            shifted = tuple((r, c+width+1) for r, c in sb)
            union = sa+shifted
            starts = _positions(available, union)
            if edge and len(starts):
                h, w = available.shape
                starts = np.asarray([p for p in starts
                                     if any(p[0]+r in (0, h-1) or p[1]+c in (0, w-1)
                                            for r, c in union)])
            if not len(starts):
                continue
            y, x = map(int, starts[int(rng.integers(len(starts)))])
            ga = tuple((y+r, x+c) for r, c in sa)
            gb = tuple((y+r, x+c) for r, c in shifted)
            if min(max(abs(r-s), abs(c-t)) for r, c in ga for s, t in gb) == 2:
                return ga, gb
    return None


def _boxes(rng, h, w, wanted):
    boxes = [(1, h-1, 1, w-1)]
    for _ in range(wanted-1):
        options = [(i, axis) for i, (y, v, x, u) in enumerate(boxes)
                   for axis, n in enumerate((v-y, u-x)) if n >= 17]
        if not options:
            break
        i, axis = _choice(rng, options)
        y, v, x, u = boxes.pop(i)
        n = (v-y, u-x)[axis]
        gap = int(rng.integers(1, min(4, n-14)+1))
        cut = int(rng.integers(7, n-gap-7+1))
        if axis == 0:
            boxes.extend(((y, y+cut, x, u), (y+cut+gap, v, x, u)))
        else:
            boxes.extend(((y, v, x, x+cut), (y, v, x+cut+gap, u)))
    return boxes


def _make_base(rng, field, colors, teaching, rich):
    h = int(rng.integers(27, 31)) if teaching is not None or rich else int(rng.integers(21, 31))
    w = int(rng.integers(27, 31)) if teaching is not None or rich else int(rng.integers(21, 31))
    boxes = _boxes(rng, h, w, len(colors)+int(rng.integers(2)))
    if len(boxes) < len(colors):
        return None
    assignments = list(colors) + [_choice(rng, colors) for _ in range(len(boxes)-len(colors))]
    rng.shuffle(assignments)
    base = np.full((h, w), field, dtype=np.int8)
    cavity_done = False
    for (y, v, x, u), color in zip(boxes, assignments):
        # Unequal insets and recursively split footprints remove a common pitch.
        top, bottom, left, right = map(int, rng.integers(0, 2, size=4))
        y, v, x, u = y+top, v-bottom, x+left, u-right
        base[y:v, x:u] = color
        hh, ww = v-y, u-x
        if teaching == 1 and not cavity_done and hh >= 13 and ww >= 13:
            cy, cx = y+(hh-5)//2, x+(ww-5)//2
            base[cy:cy+5, cx:cx+5] = field
            cavity_done = True
        elif teaching != 0 and min(hh, ww) >= 11 and rng.random() < .55:
            kind = int(rng.integers(3))
            th, tw = int(rng.integers(4, hh-3)), int(rng.integers(4, ww-3))
            if kind == 0:
                base[y:v-th, x+tw:u] = field  # L: retain a thick left/bottom arm.
            elif kind == 1:
                base[y:v, x:u] = field
                base[y:v, x+(ww-tw)//2:x+(ww+tw)//2] = color
                base[y+(hh-th)//2:y+(hh+th)//2, x:u] = color
            else:
                base[y:v, x:u] = field
                base[y:v, x:x+tw] = color
                base[y:y+th, x:u] = color
                if hh >= 15:
                    base[v-4:v, x:u] = color
    if teaching == 1 and not cavity_done:
        return None
    # A host can meet an edge without covering any of the four field corners.
    edge_color = colors[-1] if teaching == 1 else _choice(rng, colors)
    if teaching == 1 or rng.random() < .6:
        choices = []
        for axis in (0, 1):
            a = base if axis == 0 else base.T
            for reverse in (False, True):
                b = a[::-1] if reverse else a
                for depth in (1, 2):
                    cols = np.flatnonzero(b[depth] == edge_color)
                    if len(cols) >= 5:
                        choices.append((axis, reverse, depth, cols))
        if not choices:
            if teaching == 1:
                return None
        else:
            axis, reverse, depth, cols = _choice(rng, choices)
            a = base if axis == 0 else base.T
            b = a[::-1] if reverse else a
            b[:depth, cols] = edge_color
    return base


def _render(base, field, placements, extra_groups):
    """Labels from construction masks and direct neighbor ownership sets."""
    inp = base.copy()
    ballots = defaultdict(Counter)
    for s, color, cells in placements:
        ballots[s][color] += 1
        for q in cells:
            inp[q] = field
    for color, cells in extra_groups:
        for q in cells:
            inp[q] = color
    winners = {s: max(counts, key=counts.get) for s, counts in ballots.items()}
    out = base.copy()
    missing = {q for _, _, cells in placements for q in cells}
    owners = defaultdict(set)
    h, w = base.shape
    for s, _, cells in placements:
        for r, c in cells:
            for dr, dc in _MOORE:
                q = r+dr, c+dc
                if 0 <= q[0] < h and 0 <= q[1] < w and q not in missing:
                    owners[q].add(winners[s])
    for q, inks in owners.items():
        out[q] = next(iter(inks)) if len(inks) == 1 else CONFLICT
    for q in missing:
        out[q] = field
    return {"input": inp, "output": out}


def _make_pair(rng, shapes, field, colors, teaching=None, ballots=None):
    a, b = shapes[:2] if len(shapes) >= 2 else (shapes[0], shapes[0])
    if ballots is None:
        ballots = {}
        for s in shapes:
            winner = _choice(rng, colors)
            others = [c for c in colors if c != winner]
            rng.shuffle(others)
            counts = _choice(rng, ((2, 1), (3, 2), (3, 1, 1), (3, 2, 1), (1,)))
            ballots[s] = dict(zip([winner]+others, counts))
    if teaching == 0:
        ballots = {a: {colors[0]: 3, colors[1]: 2, colors[2]: 2},
                   b: {colors[1]: 2, colors[2]: 1}}
    elif teaching == 1:
        ballots = {a: {colors[0]: 3, colors[2]: 2},
                   b: {colors[0]: 2, colors[2]: 1}}
    rich = sum(sum(v.values()) for v in ballots.values()) >= 10
    for attempt in range(180):
        base = _make_base(rng, field, colors, teaching, rich)
        if base is None:
            continue
        available = {c: base == c for c in colors}
        placements, remaining = [], {s: dict(v) for s, v in ballots.items()}
        if teaching is not None:
            c = colors[2]
            # Distinct classes conflict in anchor one and agree in anchor two.
            pair = _joint(available[c], a, b, rng, edge=teaching == 1)
            if pair is None:
                continue
            for s, cells in zip((a, b), pair):
                placements.append((s, c, cells))
                remaining[s][c] -= 1
                for mask in available.values():
                    _block(mask, cells)
        specs = [(s, c) for s, counts in remaining.items() for c, n in counts.items() for _ in range(n)]
        rng.shuffle(specs)
        specs.sort(key=lambda item: len(item[0]), reverse=True)
        failed = False
        for s, c in specs:
            variants = _variants(s)
            if teaching == 0 and s == a:
                # Winning hosts use the opposite handedness to the joint query.
                joint_raw = _normal(placements[0][2])
                variants = (_rotations(_reflect(joint_raw)) if c == colors[0]
                            else _rotations(joint_raw))
            cells = _place(available[c], variants, rng)
            if cells is None:
                failed = True
                break
            placements.append((s, c, cells))
            for mask in available.values():
                _block(mask, cells)
        if failed:
            continue
        extra_available = base == field
        extra_available[[0, -1], :] = False
        extra_available[:, [0, -1]] = False
        extra_groups = []
        # Two positive copies would overturn the negative-space ballot if counted.
        extra_specs = [(colors[2], a), (colors[2], a)] if teaching == 0 else []
        extra_specs += [(_choice(rng, colors), _choice(rng, (((0, 0),), ((0, 0), (0, 1)))))
                        for _ in range(int(rng.integers(1, 5)))]
        for color, s in extra_specs:
            cells = _place(extra_available, _variants(s), rng)
            if cells is None:
                failed = True
                break
            extra_groups.append((color, cells))
            _block(extra_available, cells)
        if failed:
            continue
        if teaching == 1:
            missing = np.zeros(base.shape, dtype=bool)
            for _, _, cells in placements:
                for q in cells:
                    missing[q] = True
            near = extra_available & _dilate(missing)
            cells = _place(near, (((0, 0),),), rng)
            if cells is None:
                continue
            extra_groups.append((_choice(rng, colors), cells))
        pair = _render(base, field, placements, extra_groups)
        try:
            data = _analyze(pair["input"])
        except ValueError:
            continue
        if not np.array_equal(data["base"], base):
            continue
        if not np.array_equal(_prediction(data), pair["output"]):
            continue
        f = _features(data, pair["output"])
        if teaching == 0 and (not f["conflict"] or not _alternative_fails(pair, "rotation")):
            continue
        if teaching == 1 and not (f["cross_class_same_ink_overlap"] and f["edge"]
                                  and f["boundary"] and f["near_extra"] and f["cavities"]):
            continue
        if teaching is not None and not f["far_extra"]:
            continue
        k = int(rng.integers(4))
        reflect = rng.random() < .5
        return {role: np.rot90(np.fliplr(grid) if reflect else grid, k).copy()
                for role, grid in pair.items()}
    return None


def _alternative_fails(pair, mode):
    try:
        data = _analyze(pair["input"], mode if mode in
                        ("area", "box", "rotation", "translation", "orthogonal") else "free")
        prediction = _prediction(data, mode)
    except ValueError:
        return True
    return not np.array_equal(prediction, pair["output"])


def _features(data, output):
    base, missing = data["base"], data["missing"]
    h, w = base.shape
    boundary = (base != data["field"]) & _dilate(base == data["field"])
    count_by_ink = defaultdict(lambda: np.zeros(base.shape, dtype=np.int16))
    for s, halo in zip(data["signatures"], data["halos"]):
        count_by_ink[data["winners"][s]] += halo
    halo = np.logical_or.reduce(data["halos"])
    field_components = _components(base == data["field"], False)
    cavities = sum(not any(r in (0, h-1) or c in (0, w-1) for r, c in g)
                   for g in field_components)
    cross_class_agreement = any(
        si != sj and data["winners"][si] == data["winners"][sj]
        and bool((hi & hj & (output == data["winners"][si])).any())
        for i, (si, hi) in enumerate(zip(data["signatures"], data["halos"]))
        for sj, hj in zip(data["signatures"][i+1:], data["halos"][i+1:]))
    return dict(shape_count=len(data["ballots"]), groups=len(data["groups"]),
                holes=int(missing.sum()), extras=int(data["extras"].sum()),
                dimensions=base.shape, geometry=_key(base),
                shapes=frozenset(data["ballots"]),
                roles=(data["field"], tuple(sorted(data["hosts"]))),
                ballots=tuple(sorted(tuple(sorted(v.values(), reverse=True))
                                     for v in data["ballots"].values())),
                conflict=bool((output == CONFLICT).any()),
                same_ink_overlap=any((n > 1).any() for n in count_by_ink.values()),
                cross_class_same_ink_overlap=cross_class_agreement,
                edge=bool(missing[0].any() or missing[-1].any()
                          or missing[:, 0].any() or missing[:, -1].any()),
                boundary=bool((missing & boundary).any()),
                near_extra=bool((data["extras"] & halo).any()),
                far_extra=bool((data["extras"] & ~halo).any()),
                cavities=cavities,
                color_regions=sum(len(_components(base == c, False))
                                  for c in set(map(int, np.unique(base))) - {data["field"]}))


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    anchors = tuple(_canonical(s) for s in _choice(rng, _TEACHING_PAIRS))
    pool = sorted(set(_pool(rng)) | set(anchors))
    puzzle, seen, known = {"train": [], "test": []}, set(), set()
    previous_winner = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            teaching = i if split == "train" and i < 2 else None
            if teaching is not None:
                shapes = anchors
            else:
                k = int(rng.integers(1, 4)) if split == "train" else int(rng.integers(2, 4))
                candidates = [s for s in pool if len(s) <= (7 if split == "test" else 5)]
                shapes = tuple(candidates[int(j)] for j in rng.choice(len(candidates), k, replace=False))
                if split == "test":
                    novel = [s for s in candidates if s not in known]
                    shapes = tuple(dict.fromkeys((_choice(rng, novel),) + shapes[1:]))
            pair = None
            for _ in range(12):
                palette = list(map(int, rng.permutation(9)))
                field, colors = palette[0], palette[1:4]
                if teaching == 1 and colors[0] == previous_winner:
                    colors[0], colors[1] = colors[1], colors[0]
                pair = _make_pair(rng, shapes, field, colors, teaching)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            data = _analyze(pair["input"])
            if split == "train":
                known.update(data["ballots"])
                if i == 0:
                    previous_winner = data["winners"][anchors[0]]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses, features = set(), {"train": [], "test": []}, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            data = _analyze(inp)
            expected = _prediction(data)
            assert np.array_equal(out, expected), "incorrect consensus halo or cleanup"
            assert np.all(out[data["missing"]] == data["field"]), "missing centers must stay empty"
            halo = np.logical_or.reduce(data["halos"])
            unchanged = ~halo & ~data["extras"]
            assert np.array_equal(out[unchanged], inp[unchanged])
            assert np.all(out[data["extras"] & ~halo] == data["field"])
            analyses[split].append(data)
            features[split].append(_features(data, out))
    train = puzzle["train"]
    assert any(d["hosts"][j] != d["winners"][s]
               for d in analyses["train"] for j, s in enumerate(d["signatures"])), "show consensus transfer"
    for alternative in ("area", "box", "translation", "rotation", "orthogonal", "local", "constant", "fill"):
        assert any(_alternative_fails(p, alternative) for p in train), alternative
    for effect in ("conflict", "same_ink_overlap", "cross_class_same_ink_overlap", "edge", "boundary", "near_extra", "far_extra", "cavities"):
        assert any(f[effect] for f in features["train"]), f"teach {effect}"
    assert any(len(v) == 3 and 2*max(v.values()) <= sum(v.values())
               for d in analyses["train"] for v in d["ballots"].values())
    history = defaultdict(set)
    for data in analyses["train"]:
        for s, winner in data["winners"].items():
            history[s].add(winner)
    assert any(len(v) >= 2 for v in history.values()), "reject a stored shape-color dictionary"
    known = set().union(*(set(d["ballots"]) for d in analyses["train"]))
    assert all(set(d["ballots"]) - known for d in analyses["test"]), "test introduces a new footprint"


def self_test():
    """Bounded checks over 42 seeds, ablations, symmetries and count limits."""
    records, keys = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        key = tuple(_key(p["input"]) for split in ("train", "test") for p in puzzle[split])
        keys.add(key)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _analyze(pair["input"])
                records.append(_features(data, pair["output"]))
                assert np.array_equal(solve(pair["input"]), pair["output"])
        if seed < 3:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(p[r], q[r]) for s in puzzle
                       for p, q in zip(puzzle[s], repeat[s]) for r in ("input", "output"))
            for pair in puzzle["train"][:2] + puzzle["test"]:
                for k in range(4):
                    inp = np.rot90(np.fliplr(pair["input"]), k).copy()
                    out = np.rot90(np.fliplr(pair["output"]), k).copy()
                    assert np.array_equal(solve(inp), out)
    assert len(keys) == 42
    assert len({f["dimensions"] for f in records}) >= 25
    assert len({f["geometry"] for f in records}) >= 150
    assert len(set().union(*(f["shapes"] for f in records))) >= 25
    assert len({f["holes"] for f in records}) >= 15
    assert len({f["groups"] for f in records}) >= 8
    assert len({f["extras"] for f in records}) >= 8
    assert len({f["ballots"] for f in records}) >= 15
    assert len({f["roles"] for f in records}) >= 100
    for nt, ne in ((2, 1), (10, 4), (7, 3)):
        p = generate_puzzle(9173, nt, ne)
        assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal example counts accepted")
    broken = {s: [{r: _grid(g) for r, g in p.items()} for p in puzzle[s]] for s in puzzle}
    broken["test"][0]["output"][0, 0] = CONFLICT
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a spurious conflict")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
