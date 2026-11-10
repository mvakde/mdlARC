"""Repairable circuit jigsaws: jointly infer tiling, gap evidence, and tree topology.

Only the standard library and NumPy are used. No parent code or stored examples
are imported. solve() reads a single input; generation metadata never selects a
packing. See task_spec.md for the exact supported domain.
"""
from functools import lru_cache
import numpy as np


CORNERS = ('TL', 'TR', 'BL', 'BR')
HOST_SEEDS = (0, 281046569, 522373922, 667279653, 655834206, 708530515,
              25450476, 539126169, 340072496, 837522017, 226308685,
              674379480, 121095013, 192647986, 734081928, 302077046,
              440536765, 809973293, 981073152, 728238457, 501048228,
              160989424, 710509747, 502417633, 958523627, 523132445,
              581226212, 685351000, 304434063, 821828469, 734479229,
              326078933, 668988895, 257093978, 456092563, 722147456,
              516345099, 895048958, 899929887, 145397782, 465758585,
              384013880)


def _integer(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(name + ' is outside its allowed range')
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError('invalid ARC grid')
    return a.astype(np.int8)


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo = [start]
        for r, c in todo:
            neighbors = ((r+dy, c+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                         if (dy or dx) and (diagonal or abs(dy)+abs(dx) == 1))
            for q in neighbors:
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
        groups.append(todo)
    return groups


def _short_diagonals(mask):
    """Erased fragments are separate straight 45-degree pairs or triples."""
    for cells in _components(mask, diagonal=True):
        cells = sorted(cells)
        if len(cells) not in (2, 3):
            return False
        steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(cells, cells[1:])]
        if len(set(steps)) != 1 or steps[0][0] != 1 or abs(steps[0][1]) != 1:
            return False
    return True


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _strong(mask):
    """Mark cells on an intact horizontal or vertical run of length >= 3."""
    result = np.zeros_like(mask)
    for m, s in ((mask, result), (mask.T, result.T)):
        if m.shape[1] < 3:
            continue
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for k in range(3):
            s[:, k:k+triples.shape[1]] |= triples
    return result


def _gaps(mask):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        if m.shape[1] < 5:
            continue
        brackets = m[:, :-4] & m[:, 4:]
        brackets &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(brackets):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _cleanup(mask):
    """Simultaneous, input-only repair and weak-diagonal deletion.

    Evidence is measured before any repair. A second repair never supplies the
    evidence for a first one. Multiple cues for a gap or overlapping repair
    footprints are outside the family and rejected, rather than tie-broken.
    """
    strong = _strong(mask)
    repairs = []
    added = np.zeros_like(mask)
    anchors = np.zeros_like(mask)
    used = set()
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r-dr, c-dc) or _at(mask, r+5*dr, c+5*dc)):
            continue
        cues = []
        for endpoint in (0, 4):
            ar, ac = r+endpoint*dr, c+endpoint*dc
            forward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                dy, dx = forward*dr+side*dc, forward*dc+side*dr
                spur = tuple((ar+k*dy, ac+k*dx) for k in (1, 2))
                if (all(_at(mask, y, x) and not strong[y, x] for y, x in spur)
                        and not _at(mask, ar+3*dy, ac+3*dx)):
                    cues.append((endpoint, side, spur))
        if not cues:
            continue
        if len(cues) != 1:
            raise ValueError('multiple deviations support one gap')
        gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
        footprint = set(gap) | set(cues[0][2])
        if used & footprint:
            raise ValueError('overlapping repair evidence')
        used |= footprint
        for q in gap:
            added[q] = True
        anchors[r, c] = anchors[r+4*dr, c+4*dc] = True
        repairs.append(((r, c, dr, dc), cues[0]))
    weak = mask & ~_strong(mask | added) & ~anchors
    erased = np.zeros_like(mask)
    for dc in (-1, 1):
        if dc == 1:
            links = weak[:-1, :-1] & weak[1:, 1:]
            erased[:-1, :-1] |= links
            erased[1:, 1:] |= links
        else:
            links = weak[:-1, 1:] & weak[1:, :-1]
            erased[:-1, 1:] |= links
            erased[1:, :-1] |= links
    # Every genuine deviation must disappear, not accidentally become a branch.
    if any(not all(erased[q] for q in cue[2]) for _, cue in repairs):
        raise ValueError('deviation intersects supported linework')
    return (mask & ~erased) | added, repairs, added, erased


def _tree(mask):
    cells = int(mask.sum())
    if cells < 3 or len(_components(mask)) != 1:
        return False
    edges = int((mask[:-1] & mask[1:]).sum() + (mask[:, :-1] & mask[:, 1:]).sum())
    # On a connected graph, E=V-1 is equivalent to absence of cycles.
    return edges == cells-1 and bool((mask[:-1] & mask[1:]).any()) and bool((mask[:, :-1] & mask[:, 1:]).any())


def _marker_cells(point, corner):
    r, c = point
    dy = -1 if corner[0] == 'T' else 1
    dx = -1 if corner[1] == 'L' else 1
    return ((r+dy, c), (r, c+dx), (r+dy, c+dx))


def _extract(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 4 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError('expect four locally distinct color roles')
    bg = int(colors[np.argmax(counts)])
    markers = []
    for color, count in zip(colors, counts):
        if color == bg or count != 3:
            continue
        points = set(map(tuple, np.argwhere(grid == color).tolist()))
        r, c = np.min(list(points), axis=0)
        if max(y for y, _ in points) != r+1 or max(x for _, x in points) != c+1:
            continue
        missing = {(r, c), (r+1, c), (r, c+1), (r+1, c+1)} - points
        if len(missing) != 1:
            continue
        p = missing.pop()
        if grid[p] in (bg, color):
            continue
        corner = {(1, 1): 'TL', (1, 0): 'TR', (0, 1): 'BL', (0, 0): 'BR'}[(p[0]-r, p[1]-c)]
        markers.append((int(color), tuple(map(int, p)), corner))
    if len(markers) != 1:
        raise ValueError('external L marker is not unique')
    marker, point, corner = markers[0]
    remaining = [(int(c), int(n)) for c, n in zip(colors, counts) if c not in (bg, marker)]
    if remaining[0][1] == remaining[1][1]:
        raise ValueError('carrier must be the more frequent payload color')
    carrier, line = [c for c, _ in sorted(remaining, key=lambda t: -t[1])]
    pieces = []
    anchor = None
    for cells in _components((grid != bg) & (grid != marker)):
        q = np.asarray(cells)
        r, c = map(int, q.min(0))
        y, x = map(int, q.max(0)+1)
        matrix = np.full((y-r, x-c), -1, dtype=np.int8)
        for yy, xx in cells:
            matrix[yy-r, xx-c] = grid[yy, xx]
        if point in cells:
            anchor = len(pieces)
        pieces.append((matrix, (r, c)))
    if not 3 <= len(pieces) <= 6 or anchor is None:
        raise ValueError('expect 3..6 pieces including an anchor')
    matrix, old = pieces[anchor]
    ph, pw = matrix.shape
    expected = (old[0] + (ph-1 if corner[0] == 'B' else 0),
                old[1] + (pw-1 if corner[1] == 'R' else 0))
    if point != expected or matrix[point[0]-old[0], point[1]-old[1]] < 0:
        raise ValueError('marker does not pin a bounding-box corner')
    return bg, carrier, line, marker, point, corner, pieces, anchor


def _packings(pieces, anchor, point, corner, canvas):
    """Exhaust all translation-only occupied-mask tilings at the pinned corner.

    At most five unanchored pieces branch in a permutation tree: <=326 visits
    per board shape, and <=30 possible heights. Thus the supported family is
    below 9,780 visits; the larger defensive guard cannot cut off a legal search.
    """
    masks = [m >= 0 for m, _ in pieces]
    area = sum(int(m.sum()) for m in masks)
    nodes = 0
    for h in range(1, canvas[0]+1):
        if area % h:
            continue
        w = area // h
        oy = point[0] - (h-1 if corner[0] == 'B' else 0)
        ox = point[1] - (w-1 if corner[1] == 'R' else 0)
        if not (1 <= w <= canvas[1] and 0 <= oy <= canvas[0]-h and 0 <= ox <= canvas[1]-w):
            continue
        if any(m.shape[0] > h or m.shape[1] > w for m in masks):
            continue
        configs = []
        for m in masks:
            coords = [tuple(map(int, q)) for q in np.argwhere(m)]
            bits = sum(1 << (r*w+c) for r, c in coords)
            configs.append((m.shape, coords[0], bits))
        ar = pieces[anchor][1][0]-oy
        ac = pieces[anchor][1][1]-ox
        ah, aw = masks[anchor].shape
        if ar < 0 or ac < 0 or ar+ah > h or ac+aw > w:
            continue
        positions = [None]*len(pieces)
        positions[anchor] = (ar, ac)
        full = (1 << area)-1
        initial = configs[anchor][2] << (ar*w+ac)

        def visit(occupied, todo):
            nonlocal nodes
            nodes += 1
            if nodes > 120000:
                raise ValueError('complete packing search exceeds family budget')
            if not todo:
                yield (h, w), (oy, ox), tuple(positions)
                return
            free = full ^ occupied
            first = (free & -free).bit_length()-1
            r, c = divmod(first, w)
            # A piece covering the first empty cell must place its first occupied
            # cell there: any earlier occupied cell would collide. This ordering
            # is complete even when a bounding box contains notches.
            for i in todo:
                (ph, pw), (fr, fc), bits = configs[i]
                y, x = r-fr, c-fc
                if y < 0 or x < 0 or y+ph > h or x+pw > w:
                    continue
                placed = bits << (y*w+x)
                if placed & occupied:
                    continue
                positions[i] = (y, x)
                yield from visit(occupied | placed, tuple(j for j in todo if j != i))
        yield from visit(initial, tuple(i for i in range(len(pieces)) if i != anchor))


def _assemble(pieces, shape, positions):
    picture = np.full(shape, -1, dtype=np.int8)
    labels = np.full(shape, -1, dtype=np.int8)
    for i, ((m, _), (r, c)) in enumerate(zip(pieces, positions)):
        sub = picture[r:r+m.shape[0], c:c+m.shape[1]]
        lab = labels[r:r+m.shape[0], c:c+m.shape[1]]
        sub[m >= 0] = m[m >= 0]
        lab[m >= 0] = i
    if (picture < 0).any():
        raise ValueError('incomplete rectangular coverage')
    return picture, labels


def _cross_repairs(repairs, labels):
    total = 0
    for (r, c, dr, dc), cue in repairs:
        evidence = [(r+k*dr, c+k*dc) for k in range(5)] + list(cue[2])
        total += len({int(labels[q]) for q in evidence}) > 1
    return total


@lru_cache(maxsize=256)
def _infer(shape, data):
    grid = np.frombuffer(data, dtype=np.int8).reshape(shape)
    bg, carrier, line, marker, point, corner, pieces, anchor = _extract(grid)
    winner = None
    answer_key = None
    packing_count = valid_count = 0
    for target, origin, positions in _packings(pieces, anchor, point, corner, shape):
        packing_count += 1
        picture, labels = _assemble(pieces, target, positions)
        try:
            clean, repairs, added, erased = _cleanup(picture == line)
        except ValueError:
            continue
        if not repairs or not _tree(clean) or not _short_diagonals(erased):
            continue
        output = np.full(shape, bg, dtype=np.int8)
        oy, ox = origin
        output[oy:oy+target[0], ox:ox+target[1]] = np.where(clean, line, carrier)
        key = output.tobytes()
        if answer_key is not None and key != answer_key:
            raise ValueError('distinct repaired boards satisfy all constraints')
        valid_count += 1
        answer_key = key
        if winner is None:
            winner = (output, {'background': bg, 'carrier': carrier, 'line': line,
                      'marker': marker, 'point': point, 'corner': corner,
                      'pieces': pieces, 'anchor': anchor, 'positions': positions,
                      'origin': origin, 'target': target, 'labels': labels,
                      'raw': picture, 'clean': clean, 'repairs': repairs,
                      'added': added, 'erased': erased,
                      'cross_repairs': _cross_repairs(repairs, labels)})
    if winner is None:
        raise ValueError('no evidenced tree assembly')
    winner[1]['packing_count'] = packing_count
    winner[1]['valid_count'] = valid_count
    return winner


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique repaired assembly inferred entirely from this input."""
    grid = _grid(input_grid)
    return _infer(grid.shape, grid.tobytes())[0].copy()


def _skeleton(rng):
    """An embedded random lattice tree, plus one unfinished extra connection."""
    rows, cols = ((2, 2), (2, 3), (3, 2))[int(rng.integers(3))]
    ys = [int(rng.integers(2, 4))]
    xs = [int(rng.integers(2, 4))]
    for _ in range(rows-1):
        ys.append(ys[-1] + int(rng.integers(7, 10) if rows == 2 else rng.integers(4, 7)))
    for _ in range(cols-1):
        xs.append(xs[-1] + int(rng.integers(7, 10) if cols == 2 else rng.integers(4, 7)))
    h, w = ys[-1]+int(rng.integers(3, 5)), xs[-1]+int(rng.integers(3, 5))
    if h*w > 235:
        return None
    edges = []
    for r in range(rows):
        for c in range(cols):
            for rr, cc in ((r+1, c), (r, c+1)):
                if rr < rows and cc < cols:
                    edges.append(((r, c), (rr, cc)))
    long = [i for i, (a, b) in enumerate(edges) if abs(ys[a[0]]-ys[b[0]]) + abs(xs[a[1]]-xs[b[1]]) >= 7]
    unfinished = int(rng.choice(long))
    parent = list(range(rows*cols))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    chosen = []
    for i in rng.permutation(len(edges)):
        i = int(i)
        if i == unfinished:
            continue
        a, b = edges[i]
        u, v = root(a[0]*cols+a[1]), root(b[0]*cols+b[1])
        if u != v:
            parent[u] = v
            chosen.append(i)
    if len(chosen) != rows*cols-1:
        return None
    out = np.zeros((h, w), dtype=bool)
    for i in chosen:
        a, b = edges[i]
        out[ys[a[0]]:ys[b[0]]+1, xs[a[1]]:xs[b[1]]+1] = True
    a, b = edges[unfinished]
    r, c = ys[a[0]], xs[a[1]]
    dr, dc = int(a[0] != b[0]), int(a[1] != b[1])
    distance = ys[b[0]]-r + xs[b[1]]-c
    start = int(rng.integers(2, distance-3))
    for k in range(distance+1):
        if not start <= k < start+3:
            out[r+k*dr, c+k*dc] = True
    if not _tree(out):
        return None
    try:
        if not np.array_equal(_cleanup(out)[0], out):
            return None
    except ValueError:
        return None
    return out


def _partition(rng, shape, count, stepped):
    rects = [(0, 0, *shape)]
    while len(rects) < count:
        choices = [(i, axis) for i, (_, _, h, w) in enumerate(rects)
                   for axis, n in enumerate((h, w)) if n >= 6]
        if not choices:
            return None
        weights = np.asarray([rects[i][2]*rects[i][3] for i, _ in choices], dtype=float)
        i, axis = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
        r, c, h, w = rects.pop(i)
        cut = int(rng.integers(3, (h if axis == 0 else w)-2))
        if axis == 0:
            rects.extend(((r, c, cut, w), (r+cut, c, h-cut, w)))
        else:
            rects.extend(((r, c, h, cut), (r, c+cut, h, w-cut)))
    labels = np.full(shape, -1, dtype=np.int8)
    for i, (r, c, h, w) in enumerate(rects):
        labels[r:r+h, c:c+w] = i
    if stepped:
        options = []
        h, w = shape
        for r in range(1, h-1):
            for c in range(1, w-1):
                donor = int(labels[r, c])
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    recipient = int(labels[r+dy, c+dx])
                    # A tab in the middle of a straight seam, not at its end.
                    if (donor != recipient and labels[r+dx, c+dy] == donor
                            and labels[r-dx, c-dy] == donor
                            and labels[r+dy+dx, c+dx+dy] == recipient
                            and labels[r+dy-dx, c+dx-dy] == recipient):
                        options.append((r, c, donor, recipient))
        if not options:
            return None
        r, c, donor, recipient = options[int(rng.integers(len(options)))]
        labels[r, c] = recipient
        if any(len(_components(labels == i)) != 1 for i in (donor, recipient)):
            return None
    return labels


def _damage_options(out, labels, axis):
    options = []
    for dr, dc in ((0, 1), (1, 0)):
        if axis != 'hv' and axis != ('h' if dc else 'v'):
            continue
        for r, c in np.argwhere(out):
            r, c = int(r), int(c)
            if not all(_at(out, r+k*dr, c+k*dc) for k in range(5)):
                continue
            if not (_at(out, r-dr, c-dc) or _at(out, r+5*dr, c+5*dc)):
                continue
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            if any(_at(out, y+dc, x+dr) or _at(out, y-dc, x-dr) for y, x in gap):
                continue
            for endpoint in (0, 4):
                ar, ac = r+endpoint*dr, c+endpoint*dc
                forward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    dy, dx = forward*dr+side*dc, forward*dc+side*dr
                    spur = tuple((ar+k*dy, ac+k*dx) for k in (1, 2))
                    if all(0 <= y < out.shape[0] and 0 <= x < out.shape[1] and not out[y, x] for y, x in spur):
                        cross = len({int(labels[r+k*dr, c+k*dc]) for k in (0, 4)}) > 1
                        options.append((gap, spur, cross, 'h' if dc else 'v'))
    return options


def _damage(rng, out, labels, axis):
    inp = out.copy()
    options = _damage_options(out, labels, axis)
    required = [axis] if axis != 'hv' else ['h', 'v']
    rng.shuffle(required)
    edits = []
    for j, want in enumerate(required):
        candidates = [o for o in options if o[3] == want and (j != 0 or o[2])]
        success = False
        for index in rng.permutation(len(candidates)):
            gap, spur, _, _ = candidates[int(index)]
            footprint = set(gap) | set(spur)
            if any(max(abs(y-yy), abs(x-xx)) <= 1 for y, x in footprint for yy, xx in edits):
                continue
            trial = inp.copy()
            for q in gap:
                trial[q] = False
            for q in spur:
                trial[q] = True
            try:
                cleaned, evidence, _, _ = _cleanup(trial)
            except ValueError:
                continue
            if np.array_equal(cleaned, out) and len(evidence) == j+1:
                inp = trial
                edits.extend(footprint)
                success = True
                break
        if not success:
            return None
    # Occasionally add another independently evidenced break on an allowed axis.
    if axis == 'hv' or rng.random() < .25:
        for index in rng.permutation(len(options)):
            gap, spur, _, _ = options[int(index)]
            footprint = set(gap) | set(spur)
            if any(max(abs(y-yy), abs(x-xx)) <= 1 for y, x in footprint for yy, xx in edits):
                continue
            trial = inp.copy()
            for q in gap:
                trial[q] = False
            for q in spur:
                trial[q] = True
            try:
                cleaned, evidence, _, _ = _cleanup(trial)
            except ValueError:
                continue
            if np.array_equal(cleaned, out) and len(evidence) == len(required)+1:
                inp = trial
                break
    # Detached clutter is a presentation choice, never a separate target object.
    for _ in range(int(rng.integers(3))):
        for _ in range(20):
            r, c = int(rng.integers(out.shape[0])), int(rng.integers(out.shape[1]))
            dy, dx = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
            cells = [(r+k*dy, c+k*dx) for k in range(int(rng.integers(2, 4)))]
            if not all(0 <= y < out.shape[0] and 0 <= x < out.shape[1] and not inp[y, x] and not out[y, x] for y, x in cells):
                continue
            trial = inp.copy()
            for q in cells:
                trial[q] = True
            try:
                cleaned, _, _, erased = _cleanup(trial)
                if np.array_equal(cleaned, out) and _short_diagonals(erased):
                    inp = trial
                    break
            except ValueError:
                pass
    return inp


def _separated(a, b):
    r, c, h, w = a
    y, x, ph, pw = b
    return r+h < y or y+ph < r or c+w < x or x+pw < c


def _scatter(rng, matrices, anchor, final, marker, canvas):
    order = sorted((i for i in range(len(matrices)) if i != anchor), key=lambda i: -matrices[i].size)
    for _ in range(8):
        positions = [None]*len(matrices)
        positions[anchor] = final[anchor]
        occupied = [(*final[anchor], *matrices[anchor].shape)]
        for i in order:
            h, w = matrices[i].shape
            options = [(r, c) for r in range(canvas[0]-h+1) for c in range(canvas[1]-w+1)]
            for k in rng.permutation(len(options)):
                r, c = options[int(k)]
                box = (r, c, h, w)
                if (all(_separated(box, old) for old in occupied)
                        and all(not (r-1 <= y <= r+h and c-1 <= x <= c+w) for y, x in marker)):
                    positions[i] = (r, c)
                    occupied.append(box)
                    break
            if positions[i] is None:
                break
        if all(p is not None for p in positions):
            return positions
    return None


def _local_first(info):
    """Assemble after repairing each scattered piece in isolation (a shortcut)."""
    matrices = []
    for matrix, old in info['pieces']:
        try:
            clean = _cleanup(matrix == info['line'])[0]
        except ValueError:
            return None
        m = matrix.copy()
        m[m >= 0] = np.where(clean, info['line'], info['carrier'])[m >= 0]
        matrices.append((m, old))
    return _assemble(matrices, info['target'], info['positions'])[0]


def _make_pair(rng, count, axis, corner, stepped, palette):
    bg, carrier, line, marker_color = palette
    for _ in range(350):
        clean = _skeleton(rng)
        if clean is None:
            continue
        labels = _partition(rng, clean.shape, count, stepped)
        if labels is None:
            continue
        damaged = _damage(rng, clean, labels, axis)
        if damaged is None:
            continue
        matrices, local_origins = [], []
        for i in range(count):
            q = np.argwhere(labels == i)
            r, c = map(int, q.min(0))
            y, x = map(int, q.max(0)+1)
            m = np.where(labels[r:y, c:x] == i,
                         np.where(damaged[r:y, c:x], line, carrier), -1).astype(np.int8)
            matrices.append(m)
            local_origins.append((r, c))
        h, w = clean.shape
        canvas = (min(30, max(h+3, int(np.ceil(np.sqrt(h*w*2.6))))+int(rng.integers(0, 4))),
                  min(30, max(w+3, int(np.ceil(np.sqrt(h*w*2.6))))+int(rng.integers(0, 4))))
        # The marker alone reserves one cell outside the designated corner.
        oy = int(rng.integers(int(corner[0] == 'T'), canvas[0]-h-int(corner[0] == 'B')+1))
        ox = int(rng.integers(int(corner[1] == 'L'), canvas[1]-w-int(corner[1] == 'R')+1))
        local_point = (h-1 if corner[0] == 'B' else 0, w-1 if corner[1] == 'R' else 0)
        anchor = int(labels[local_point])
        point = (oy+local_point[0], ox+local_point[1])
        marker = _marker_cells(point, corner)
        final = [(oy+r, ox+c) for r, c in local_origins]
        positions = _scatter(rng, matrices, anchor, final, marker, canvas)
        if positions is None:
            continue
        grid = np.full(canvas, bg, dtype=np.int8)
        for m, (r, c) in zip(matrices, positions):
            sub = grid[r:r+m.shape[0], c:c+m.shape[1]]
            sub[m >= 0] = m[m >= 0]
        for q in marker:
            grid[q] = marker_color
        expected = np.full(canvas, bg, dtype=np.int8)
        expected[oy:oy+h, ox:ox+w] = np.where(clean, line, carrier)
        try:
            inferred, info = _infer(grid.shape, grid.tobytes())
        except ValueError:
            continue
        if (not np.array_equal(inferred, expected) or not info['cross_repairs']
                or info['packing_count'] < 2):
            continue
        local = _local_first(info)
        if local is not None and np.array_equal(local, np.where(clean, line, carrier)):
            continue
        return {'input': grid, 'output': expected}
    raise RuntimeError('could not construct an unambiguous circuit jigsaw')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _integer(num_train, 'num_train', 2, 10)
    t = _integer(num_test, 'num_test', 1, 4)
    if n <= t:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    # NumPy accepts arbitrarily large nonnegative integers. Normalize negatives
    # for a defined deterministic extension of the seed API.
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    shift = 'count' if rng.integers(2) else 'axes'
    steps = bool(rng.random() < .7)
    base = tuple(map(int, rng.choice(10, 4, replace=False)))
    palette_mode = int(rng.integers(3))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, total in (('train', n), ('test', t)):
        plans = []
        for index in range(total):
            if split == 'train':
                count = (3, 4)[index] if index < 2 else int(rng.integers(3, 5 if shift == 'count' else 6))
                axis = ('h', 'v')[index] if index < 2 else str(rng.choice(('h', 'v')))
            else:
                count = int(rng.integers(5, 7)) if shift == 'count' else int(rng.integers(3, 6))
                axis = str(rng.choice(('h', 'v', 'hv'))) if shift == 'count' else 'hv'
            stepped = count >= 4 and steps and ((split == 'train' and index == 1) or (index > 1 and rng.random() < .45) or (split == 'test' and rng.random() < .65))
            palette = base
            if palette_mode == 1 and rng.integers(2):
                palette = (base[1], base[0], base[2], base[3])
            elif palette_mode == 2 and rng.random() < .7:
                palette = tuple(map(int, rng.choice(10, 4, replace=False)))
            plans.append((count, axis, str(rng.choice(CORNERS)), stepped, palette))
        for plan in plans:
            for _ in range(8):
                pair = _make_pair(rng, *plan)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input generation exhausted')
        if split == 'train':
            rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    predicted, info = _infer(inp.shape, inp.tobytes())
    assert np.array_equal(predicted, out), 'incorrect or incomplete joint reconstruction'
    bg, carrier, line, marker = [info[k] for k in ('background', 'carrier', 'line', 'marker')]
    assert not np.any(out == marker)
    assert all(out[q] == bg for q in _marker_cells(info['point'], info['corner']))
    assert _tree(info['clean'])
    assert info['cross_repairs'] >= 1 and info['packing_count'] >= 2
    assert info['positions'][info['anchor']] == (
        info['pieces'][info['anchor']][1][0]-info['origin'][0],
        info['pieces'][info['anchor']][1][1]-info['origin'][1])
    assert np.count_nonzero(out != bg) == sum(int((m >= 0).sum()) for m, _ in info['pieces'])
    assert set(map(int, np.unique(out))) == {bg, carrier, line}
    assert int(np.bincount(out.ravel(), minlength=10).argmax()) == bg
    assert int((info['raw'] == line).sum()) + int(info['added'].sum()) - int(info['erased'].sum()) == int(info['clean'].sum())
    assert int(info['added'].sum()) == 3*len(info['repairs'])
    assert info['erased'].sum() >= 2*len(info['repairs'])
    assert _short_diagonals(info['erased'])
    assert len(_gaps(info['clean'])) >= 1, 'missing intentional-gap contrast'
    # The deliberate unfilled connection would create a cycle if closed.
    assert any(not _tree(np.logical_or(info['clean'], _gap_mask(info['clean'].shape, gap))) for gap in _gaps(info['clean']))
    local = _local_first(info)
    correct = np.where(info['clean'], line, carrier)
    assert local is None or not np.array_equal(local, correct), 'repair-first shortcut succeeded'
    # Assembly with unchanged noisy piece contents also differs from the answer.
    assert not np.array_equal(info['raw'], correct), 'B contribution is vacuous'
    # B alone on the scattered canvas cannot supply A's rectangular board.
    try:
        scattered_clean = _cleanup(inp == line)[0]
        b_only = inp.copy()
        payload = (inp != bg) & (inp != marker)
        b_only[payload] = np.where(scattered_clean, line, carrier)[payload]
        b_only[scattered_clean] = line
        b_only[inp == marker] = bg
        assert not np.array_equal(b_only, out), 'A contribution is vacuous'
    except ValueError:
        pass
    return info


def _gap_mask(shape, gap):
    mask = np.zeros(shape, dtype=bool)
    r, c, dr, dc = gap
    for k in (1, 2, 3):
        mask[r+k*dr, c+k*dc] = True
    return mask


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen = set()
    infos = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp = _grid(pair['input'])
            key = (inp.shape, inp.tobytes())
            assert key not in seen, 'duplicate input'
            seen.add(key)
            infos[split].append(_pair_info(pair))
    train_axes = {'h' if gap[2] == 0 else 'v' for info in infos['train'] for gap, _ in info['repairs']}
    assert train_axes == {'h', 'v'}, 'training must demonstrate both repair axes'
    assert len({len(info['pieces']) for info in infos['train']}) >= 2
    if any(any((m < 0).any() for m, _ in info['pieces']) for info in infos['test']):
        assert any(any((m < 0).any() for m, _ in info['pieces']) for info in infos['train'])


def _fingerprint(puzzle):
    return tuple((split, tuple((_grid(p['input']).shape, _grid(p['input']).tobytes(),
                              _grid(p['output']).tobytes()) for p in puzzle[split])) for split in ('train', 'test'))


def self_test():
    """Bounded semantic, mutation, equivariance, and 42-seed variation checks."""
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    infos = [_pair_info(pair) for p in puzzles for split in ('train', 'test') for pair in p[split]]
    assert {i['corner'] for i in infos} == set(CORNERS)
    assert {len(i['pieces']) for i in infos} == {3, 4, 5, 6}
    assert len({i['target'] for i in infos}) >= 20
    assert len({pair['input'].shape for p in puzzles for s in ('train', 'test') for pair in p[s]}) >= 20
    assert len({(i['target'], i['labels'].tobytes()) for i in infos}) >= 100
    assert len({(i['target'], i['clean'].tobytes()) for i in infos}) >= 100
    assert len({(i['background'], i['carrier'], i['line'], i['marker']) for i in infos}) >= 30
    assert {''.join(sorted({'h' if g[2] == 0 else 'v' for g, _ in i['repairs']})) for i in infos} == {'h', 'v', 'hv'}
    assert any(any((m < 0).any() for m, _ in i['pieces']) for i in infos)
    assert {(cue[0], cue[1]) for i in infos for _, cue in i['repairs']} == {(0, -1), (0, 1), (4, -1), (4, 1)}
    sample = puzzles[0]
    assert _fingerprint(sample) == _fingerprint(generate_puzzle(HOST_SEEDS[0]))
    # Every dihedral transform and every palette permutation preserves semantics.
    pair = sample['train'][0]
    for turns in range(4):
        for flip in (False, True):
            inp, out = np.rot90(pair['input'], turns), np.rot90(pair['output'], turns)
            if flip:
                inp, out = inp[:, ::-1], out[:, ::-1]
            assert np.array_equal(solve(inp), out)
    permutation = np.asarray([9, 2, 5, 1, 7, 0, 8, 4, 6, 3], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    for corruption in ('omit_repair', 'keep_spur', 'fill_intentional', 'marker', 'anchor', 'duplicate'):
        damaged = {s: [{k: _grid(v).copy() for k, v in pair.items()} for pair in sample[s]] for s in ('train', 'test')}
        pair = damaged['test'][0]
        info = _pair_info(pair)
        out = pair['output']
        oy, ox = info['origin']
        gap, cue = info['repairs'][0]
        if corruption == 'omit_repair':
            r, c, dr, dc = gap
            for k in (1, 2, 3):
                out[oy+r+k*dr, ox+c+k*dc] = info['carrier']
            for r, c in cue[2]:
                out[oy+r, ox+c] = info['line']
        elif corruption == 'keep_spur':
            for r, c in cue[2]:
                out[oy+r, ox+c] = info['line']
        elif corruption == 'fill_intentional':
            r, c, dr, dc = _gaps(info['clean'])[0]
            for k in (1, 2, 3):
                out[oy+r+k*dr, ox+c+k*dc] = info['line']
        elif corruption == 'marker':
            out[_marker_cells(info['point'], info['corner'])[0]] = info['marker']
        elif corruption == 'anchor':
            pair['output'] = np.roll(out, 1, axis=0)
        else:
            damaged['train'][1] = {k: v.copy() for k, v in damaged['train'][0].items()}
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('accepted mutation: ' + corruption)
    for n, t in ((2, 1), (10, 4)):
        p = generate_puzzle(71+n, n, t)
        assert len(p['train']) == n and len(p['test']) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
