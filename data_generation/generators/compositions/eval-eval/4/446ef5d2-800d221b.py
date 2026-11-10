"""Sealed seam reservoirs: anchored jigsaws whose reconstructed walls vote.

Only standard-library modules and NumPy are used. No parent code or data are
loaded. All role recovery, assembly, enclosure and voting use the given input.
"""
from functools import lru_cache
import hashlib
import heapq
import json
from numbers import Integral

import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNERS = ("TL", "TR", "BR", "BL")


def _checked_int(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + " is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    h, w = mask.shape
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        cells = [start]
        for r, c in cells:
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in pending:
                    pending.remove(q)
                    cells.append(q)
        result.append(cells)
    return result


def _marker(point, corner):
    r, c = point
    dr = -1 if corner[0] == "T" else 1
    dc = -1 if corner[1] == "L" else 1
    return ((r + dr, c), (r, c + dc), (r + dr, c + dc))


def _parse(g):
    # A blank outer margin makes background recognition independent of counts.
    bg = int(g[0, 0])
    assert all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1]))
    candidates = []
    for color in map(int, np.unique(g)):
        q = np.argwhere(g == color)
        if color == bg or len(q) != 3 or tuple(np.ptp(q, axis=0)) != (1, 1):
            continue
        r, c = map(int, q.min(axis=0))
        missing = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} - set(map(tuple, q.tolist()))
        point = missing.pop()
        if g[point] in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 0): "BR", (0, 1): "BL"}[(point[0] - r, point[1] - c)]
        candidates.append((color, point, corner))
    assert len(candidates) == 1, "need one external corner marker"
    marker, point, corner = candidates[0]
    pieces, anchor = [], None
    for cells in _components((g != bg) & (g != marker)):
        q = np.array(cells)
        r, c = map(int, q.min(axis=0))
        h, w = map(int, q.max(axis=0) - q.min(axis=0) + 1)
        assert len(cells) == h * w and min(h, w) >= 5
        block = g[r:r + h, c:c + w].copy()
        assert not np.any((block == bg) | (block == marker))
        if point in cells:
            anchor = len(pieces)
            assert point == (r + (h - 1 if corner[0] == "B" else 0),
                             c + (w - 1 if corner[1] == "R" else 0))
        pieces.append((block, (r, c)))
    assert 3 <= len(pieces) <= 6 and anchor is not None
    carrier = int(pieces[0][0][0, 0])
    for m, _ in pieces:
        assert all(m[p] == carrier for p in ((0, 0), (-1, 0), (0, -1), (-1, -1)))
    colors = set(map(int, np.unique(g)))
    assert len(colors) == 7
    remaining = colors - {bg, marker, carrier}
    # The key ink is the only remaining role confined to piece perimeters.
    inks = [color for color in remaining if all(not np.any(m[1:-1, 1:-1] == color) for m, _ in pieces)]
    assert len(inks) == 1
    ink = inks[0]
    rest = sorted(remaining - {ink})
    interpretations = []
    for well in rest:
        signals = tuple(x for x in rest if x != well)
        labels, boxes = [], []
        valid = True
        for m, _ in pieces:
            mask = (m == signals[0]) | (m == signals[1])
            q = np.argwhere(mask)
            if not len(q):
                valid = False
                break
            r, c = map(int, q.min(axis=0))
            y, x = map(int, q.max(axis=0) + 1)
            a, b = [int(np.sum(m == s)) for s in signals]
            if (r < 1 or c < 1 or y >= m.shape[0] or x >= m.shape[1]
                    or len(q) != (y - r) * (x - c) or min(a, b) == 0 or a == b):
                valid = False
                break
            labels.append(signals[int(b > a)])
            boxes.append((r, c, y - r, x - c))
        if valid and any(np.any(m[1:-1, 1:-1] == well) for m, _ in pieces):
            interpretations.append((well, signals, labels, boxes))
    assert len(interpretations) == 1, "ambiguous panel / reservoir roles"
    well, signals, labels, boxes = interpretations[0]
    return dict(bg=bg, marker=marker, point=point, corner=corner, pieces=pieces,
                anchor=anchor, carrier=carrier, ink=ink, well=well,
                signals=signals, labels=labels, boxes=boxes)


def _contact_ok(a, pa, b, pb):
    r, c = pa
    y, x = pb
    h, w = a.shape
    bh, bw = b.shape
    lo, hi = max(r, y), min(r + h, y + bh)
    if lo < hi:
        if c + w == x and not np.array_equal(a[lo-r:hi-r, -1], b[lo-y:hi-y, 0]):
            return False
        if x + bw == c and not np.array_equal(a[lo-r:hi-r, 0], b[lo-y:hi-y, -1]):
            return False
    lo, hi = max(c, x), min(c + w, x + bw)
    if lo < hi:
        if r + h == y and not np.array_equal(a[-1, lo-c:hi-c], b[0, lo-x:hi-x]):
            return False
        if y + bh == r and not np.array_equal(a[0, lo-c:hi-c], b[-1, lo-x:hi-x]):
            return False
    return True


def _assemble(info, canvas):
    pieces = [m for m, _ in info["pieces"]]
    count = len(pieces)
    area = sum(m.size for m in pieces)
    anchor = info["anchor"]
    point, corner = info["point"], info["corner"]
    first = None
    first_key = None
    # All area-factor rectangles, all translation-only rectangular tilings.
    # With at most five unanchored pieces the exhaustive search is bounded by
    # small permutations; no incomplete search is used as a uniqueness proof.
    for h in range(max(m.shape[0] for m in pieces), canvas[0] + 1):
        if area % h:
            continue
        w = area // h
        oy = point[0] - (h - 1 if corner[0] == "B" else 0)
        ox = point[1] - (w - 1 if corner[1] == "R" else 0)
        if not (1 <= oy and 1 <= ox and oy + h < canvas[0] and ox + w < canvas[1]):
            continue
        if any(m.shape[1] > w for m in pieces):
            continue
        ar, ac = info["pieces"][anchor][1]
        ar, ac = ar - oy, ac - ox
        ah, aw = pieces[anchor].shape
        if ar < 0 or ac < 0 or ar + ah > h or ac + aw > w:
            continue
        occupied = np.zeros((h, w), dtype=bool)
        occupied[ar:ar + ah, ac:ac + aw] = True
        positions = {anchor: (ar, ac)}

        def visit(remaining):
            nonlocal first, first_key
            if not remaining:
                image = np.empty((h, w), dtype=np.int8)
                ids = np.empty((h, w), dtype=np.int8)
                labels = np.empty((h, w), dtype=np.int8)
                for i, (r, c) in positions.items():
                    ph, pw = pieces[i].shape
                    image[r:r + ph, c:c + pw] = pieces[i]
                    ids[r:r + ph, c:c + pw] = i
                    labels[r:r + ph, c:c + pw] = info["labels"][i]
                key = (h, w, oy, ox, image.tobytes(), labels.tobytes())
                assert first_key is None or first_key == key, "multiple distinct edge-matched assemblies"
                if first_key is None:
                    first_key = key
                    first = (image, ids, labels, (oy, ox), positions.copy())
                return
            free = np.flatnonzero(~occupied)
            r, c = divmod(int(free[0]), w)
            for i in remaining:
                m = pieces[i]
                ph, pw = m.shape
                if r + ph > h or c + pw > w or occupied[r:r + ph, c:c + pw].any():
                    continue
                if not all(_contact_ok(m, (r, c), pieces[j], p) for j, p in positions.items()):
                    continue
                positions[i] = (r, c)
                occupied[r:r + ph, c:c + pw] = True
                visit(tuple(j for j in remaining if j != i))
                occupied[r:r + ph, c:c + pw] = False
                del positions[i]

        visit(tuple(i for i in range(count) if i != anchor))
    assert first is not None, "no edge-matched rectangular assembly"
    return first


def _regions(image, ids, well, count):
    h, w = image.shape
    result = []
    for cells in _components(image == well):
        weights = np.zeros(count, dtype=int)
        wall_cells = set()
        opened = any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
        for r, c in cells:
            for dr, dc in DIRS:
                y, x = r + dr, c + dc
                if 0 <= y < h and 0 <= x < w and image[y, x] != well:
                    weights[int(ids[y, x])] += 1
                    wall_cells.add((y, x))
        cell_weights = np.zeros(count, dtype=int)
        for p in wall_cells:
            cell_weights[int(ids[p])] += 1
        result.append(dict(cells=cells, weights=weights, cell_weights=cell_weights, open=opened,
                           span=len({int(ids[p]) for p in cells})))
    return result


def _fill(image, ids, well, majorities, signals):
    output = image.copy()
    records = _regions(image, ids, well, len(majorities))
    for rec in records:
        totals = [sum(int(weight) for weight, label in zip(rec["weights"], majorities) if label == s) for s in signals]
        rec["totals"] = totals
        if rec["open"]:
            rec["winner"] = None
            continue
        assert totals[0] != totals[1], "an enclosed reservoir has a tied wall vote"
        winner = signals[int(totals[1] > totals[0])]
        rec["winner"] = winner
        for p in rec["cells"]:
            output[p] = winner
    return output, records


@lru_cache(maxsize=96)
def _infer(shape, data):
    g = np.frombuffer(data, dtype=np.int8).reshape(shape)
    info = _parse(g)
    image, ids, labels, origin, positions = _assemble(info, shape)
    filled, records = _fill(image, ids, info["well"], info["labels"], info["signals"])
    out = np.full(shape, info["bg"], dtype=np.int8)
    r, c = origin
    out[r:r + image.shape[0], c:c + image.shape[1]] = filled
    info.update(image=image, ids=ids, labelmap=labels, origin=origin,
                positions=positions, regions=records)
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique reconstruction and its sealed-reservoir elections."""
    g = _grid(input_grid)
    return _infer(g.shape, g.tobytes())[0].copy()


def _partition(rng, count):
    # Random recursive cuts include unequal strips, T-junctions and grids.
    h = int(rng.integers(10, 19))
    w = int(rng.integers(12, 21))
    if h * w > 315:
        return None
    rects = [(0, 0, h, w)]
    for _ in range(count - 1):
        choices = [(i, axis) for i, (_, _, ph, pw) in enumerate(rects)
                   for axis, length in enumerate((ph, pw)) if length >= 10]
        if not choices:
            return None
        i, axis = choices[int(rng.integers(len(choices)))]
        r, c, ph, pw = rects.pop(i)
        cut = int(rng.integers(5, (ph if axis == 0 else pw) - 4))
        if axis == 0:
            rects.extend([(r, c, cut, pw), (r + cut, c, ph - cut, pw)])
        else:
            rects.extend([(r, c, ph, cut), (r, c + cut, ph, pw - cut)])
    if rng.integers(2):
        rects = [(c, r, pw, ph) for r, c, ph, pw in rects]
        h, w = w, h
    ids = np.empty((h, w), dtype=np.int8)
    for i, (r, c, ph, pw) in enumerate(rects):
        ids[r:r + ph, c:c + pw] = i
    return rects, ids


def _units(ids, forbidden):
    h, w = ids.shape
    parents = list(range(h * w))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for r in range(h):
        for c in range(w):
            for y, x in ((r + 1, c), (r, c + 1)):
                if y < h and x < w and ids[r, c] != ids[y, x]:
                    parents[root(y * w + x)] = root(r * w + c)
    groups = {}
    for r in range(h):
        for c in range(w):
            groups.setdefault(root(r * w + c), []).append((r, c))
    cells = list(groups.values())
    unit_ids = np.empty((h, w), dtype=int)
    for i, group in enumerate(cells):
        for p in group:
            unit_ids[p] = i
    allowed = {i for i, group in enumerate(cells) if not any(forbidden[p] for p in group)}
    interior = {i for i in allowed if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells[i])}
    graph = [set() for _ in cells]
    for r in range(h):
        for c in range(w):
            i = int(unit_ids[r, c])
            for dr, dc in DIRS:
                y, x = r + dr, c + dc
                if 0 <= y < h and 0 <= x < w and unit_ids[y, x] != i:
                    graph[i].add(int(unit_ids[y, x]))
    return cells, unit_ids, graph, allowed, interior


def _path(rng, start, target, graph, allowed):
    costs = 1 + rng.random(len(graph)) * 2
    queue, dist, prev = [(0., start)], {start: 0.}, {}
    while queue:
        d, i = heapq.heappop(queue)
        if d != dist[i]:
            continue
        if i == target:
            path = {i}
            while i != start:
                i = prev[i]
                path.add(i)
            return path
        for j in sorted(graph[i] & allowed):
            nd = d + costs[j]
            if nd < dist.get(j, float("inf")):
                dist[j], prev[j] = nd, i
                heapq.heappush(queue, (nd, j))
    return None


def _grow(rng, selected, graph, allowed, steps):
    selected = set(selected)
    for _ in range(steps):
        options = sorted(set().union(*(graph[i] for i in selected)) & allowed - selected)
        if not options:
            break
        selected.add(options[int(rng.integers(len(options)))])
    return selected


def _assignment(rng, weights, kind, cell_weights):
    count = len(weights)
    owners = weights > 0
    choices = []
    for bits in range(1, (1 << count) - 1):
        labels = np.array([(bits >> i) & 1 for i in range(count)])
        totals = [int(weights[labels == s].sum()) for s in (0, 1)]
        if min(totals) == 0 or totals[0] == totals[1]:
            continue
        winner = int(totals[1] > totals[0])
        votes = int(np.sum(labels[owners] == winner))
        if kind == 0 and (2 * votes >= int(owners.sum()) or 2 * int(np.sum(labels == winner)) >= count):
            continue
        if kind == 1:
            if np.sum(weights == weights.max()) != 1 or labels[int(weights.argmax())] == winner:
                continue
            cells = [int(cell_weights[labels == s].sum()) for s in (0, 1)]
            if cells[0] == cells[1] or int(cells[1] > cells[0]) == winner:
                continue
        choices.append((labels, winner))
    return choices[int(rng.integers(len(choices)))] if choices else None


def _design(rng, rects, ids, kind):
    h, w = ids.shape
    forbidden = np.zeros((h, w), dtype=bool)
    boundary = np.zeros((h, w), dtype=bool)
    corners = np.zeros((h, w), dtype=bool)
    ballots = []
    for r, c, ph, pw in rects:
        bh = int(rng.integers(2, min(3, ph - 2) + 1))
        bw = int(rng.integers(2, min(4, pw - 2) + 1))
        y = r + int(rng.integers(1, ph - bh))
        x = c + int(rng.integers(1, pw - bw))
        forbidden[y:y + bh, x:x + bw] = True
        ballots.append((y, x, bh, bw))
        boundary[r, c:c + pw] = boundary[r + ph - 1, c:c + pw] = True
        boundary[r:r + ph, c] = boundary[r:r + ph, c + pw - 1] = True
        for p in ((r, c), (r + ph - 1, c), (r, c + pw - 1), (r + ph - 1, c + pw - 1)):
            corners[p] = True
            forbidden[p] = True
    cells, unit_ids, graph, allowed, interior = _units(ids, forbidden)
    seam = [i for i in sorted(interior) if len({int(ids[p]) for p in cells[i]}) > 1]
    if len(seam) < 2:
        return None
    for _ in range(28):
        a, b = map(int, rng.choice(seam, 2, replace=False))
        path = _path(rng, a, b, graph, interior)
        if path is None:
            continue
        main = _grow(rng, path, graph, interior, int(rng.integers(0, 13)))
        mask = np.zeros(ids.shape, dtype=np.int8)
        for i in main:
            for p in cells[i]:
                mask[p] = 1
        rec = _regions(mask, ids, 1, len(rects))[0]
        if rec["span"] < 3 or np.count_nonzero(rec["weights"]) < 3:
            continue
        assignment = _assignment(rng, rec["weights"], kind, rec["cell_weights"])
        if assignment is not None:
            labels, winner = assignment
            break
    else:
        return None
    chosen = set(main)

    def separated(pool):
        touching = chosen | set().union(*(graph[i] for i in chosen))
        return pool - touching

    # A second, purely internal well in an opposing piece shows both outcomes
    # without requiring more training examples. Its position and shape vary.
    pool = {i for i in interior if len(cells[i]) == 1
            and not boundary[cells[i][0]] and labels[int(ids[cells[i][0]])] != winner}
    options = sorted(separated(pool))
    if not options:
        return None
    start = options[int(rng.integers(len(options)))]
    owner = int(ids[cells[start][0]])
    pool = {i for i in pool if int(ids[cells[i][0]]) == owner}
    chosen |= _grow(rng, {start}, graph, separated(pool), int(rng.integers(0, 5)))
    # Open mouths are live counterexamples to filling every blank patch.
    options = sorted(separated(allowed - interior))
    if not options:
        return None
    start = options[int(rng.integers(len(options)))]
    chosen |= _grow(rng, {start}, graph, separated(allowed), int(rng.integers(1, 7)))
    # Optional additional independent wells, never hidden distractors.
    for _ in range(int(rng.integers(0, 3))):
        options = sorted(separated(interior))
        if not options:
            break
        start = options[int(rng.integers(len(options)))]
        chosen |= _grow(rng, {start}, graph, separated(interior), int(rng.integers(0, 4)))
    well_mask = np.zeros(ids.shape, dtype=bool)
    for i in chosen:
        for p in cells[i]:
            well_mask[p] = True
    return ballots, boundary, corners, cells, chosen, well_mask, labels, winner


def _pattern(rng, h, w, major, minor, strength):
    area = h * w
    minority = ((area - 1) // 2 if strength == "weak" else 1 if strength == "strong"
                else int(rng.integers(1, (area - 1) // 2 + 1)))
    rr, cc = np.indices((h, w))
    style = int(rng.integers(4))
    if style == 0:
        order = rng.permutation(area)
    elif style == 1:
        y, x = int(rng.integers(h)), int(rng.integers(w))
        order = np.argsort((abs(rr-y) + abs(cc-x) + rng.random((h, w)) * .1).ravel())
    elif style == 2:
        order = np.argsort(((rr + cc) % 2 * area + rr * w + cc).ravel())
    else:
        order = np.arange(area) if rng.integers(2) else np.arange(area)[::-1]
    result = np.full(area, major, dtype=np.int8)
    result[order[:minority]] = minor
    return result.reshape(h, w)


def _scatter(rng, matrices, anchor, final, markers, canvas):
    h, w = canvas
    order = sorted((i for i in range(len(matrices)) if i != anchor), key=lambda i: -matrices[i].size)
    for _ in range(12):
        used = np.zeros(canvas, dtype=bool)
        for r, c in markers:
            used[max(0, r-1):r+2, max(0, c-1):c+2] = True
        r, c = final[anchor]
        ah, aw = matrices[anchor].shape
        used[r-1:r+ah+1, c-1:c+aw+1] = True
        positions = {anchor: (r, c)}
        for i in order:
            ph, pw = matrices[i].shape
            options = [(r, c) for r in range(1, h-ph) for c in range(1, w-pw)
                       if not used[r:r+ph, c:c+pw].any()]
            if not options:
                break
            r, c = options[int(rng.integers(len(options)))]
            positions[i] = (r, c)
            used[r-1:r+ph+1, c-1:c+pw+1] = True
        if len(positions) == len(matrices):
            return positions
    return None


def _make_pair(rng, count, corner, roles, kind):
    bg, marker, carrier, ink, well, s0, s1 = roles
    for _ in range(320):
        part = _partition(rng, count)
        if part is None:
            continue
        rects, ids = part
        design = _design(rng, rects, ids, kind)
        if design is None:
            continue
        ballots, boundary, corners, units, selected, well_mask, bits, winner = design
        h, w = ids.shape
        image = np.full((h, w), carrier, dtype=np.int8)
        majorities = [roles[5 + int(bit)] for bit in bits]
        for i, (r, c, bh, bw) in enumerate(ballots):
            strength = ("weak" if bits[i] == winner else "strong") if kind == 0 else "random"
            image[r:r+bh, c:c+bw] = _pattern(rng, bh, bw, majorities[i], roles[6-int(bits[i])], strength)
        if kind == 0 and np.sum(image == roles[5+winner]) >= np.sum(image == roles[6-winner]):
            continue
        for i, group in enumerate(units):
            if i in selected:
                for p in group:
                    image[p] = well
            elif any(boundary[p] for p in group) and not any(corners[p] for p in group):
                color = ink if rng.random() < .52 else carrier
                for p in group:
                    image[p] = color
        if np.sum(image == ink) < 4:
            continue
        try:
            filled, _ = _fill(image, ids, well, majorities, (s0, s1))
        except AssertionError:
            continue
        matrices = [image[r:r+ph, c:c+pw].copy() for r, c, ph, pw in rects]
        area = h * w
        base = int(np.ceil(np.sqrt(area * 2.25)))
        ch = min(30, max(h + 5, base) + int(rng.integers(0, 5)))
        cw = min(30, max(w + 5, base) + int(rng.integers(0, 5)))
        if ch-h < 4 or cw-w < 4:
            continue
        oy = int(rng.integers(2, ch-h-1))
        ox = int(rng.integers(2, cw-w-1))
        py = h-1 if corner[0] == "B" else 0
        px = w-1 if corner[1] == "R" else 0
        point = oy+py, ox+px
        anchor = int(ids[py, px])
        markers = _marker(point, corner)
        final = [(oy+r, ox+c) for r, c, _, _ in rects]
        scattered = _scatter(rng, matrices, anchor, final, markers, (ch, cw))
        if scattered is None:
            continue
        inp = np.full((ch, cw), bg, dtype=np.int8)
        for i, (r, c) in scattered.items():
            ph, pw = matrices[i].shape
            inp[r:r+ph, c:c+pw] = matrices[i]
        for p in markers:
            inp[p] = marker
        out = np.full_like(inp, bg)
        out[oy:oy+h, ox:ox+w] = filled
        try:
            predicted, info = _infer(inp.shape, inp.tobytes())
            if not np.array_equal(predicted, out):
                continue
            m = _measure(info, inp)
            if not (m["minority_witness"] if kind == 0 else m["largest_and_cells_wrong"] if kind == 1 else True):
                continue
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise RuntimeError("bounded reservoir construction exhausted")


def _measure(info, inp):
    closed = [r for r in info["regions"] if not r["open"]]
    minority, largest, fragment, cell_wrong, largest_and_cells = False, False, False, False, False
    for rec in closed:
        weights, winner = rec["weights"], rec["winner"]
        labels = np.array(info["labels"])
        owners = weights > 0
        minority |= 2 * np.sum(labels[owners] == winner) < np.sum(owners)
        biggest_wrong = np.sum(weights == weights.max()) == 1 and labels[int(weights.argmax())] != winner
        largest |= biggest_wrong
        cell_totals = [int(rec["cell_weights"][labels == s].sum()) for s in info["signals"]]
        wrong_cells = (cell_totals[0] != cell_totals[1]
                       and info["signals"][int(cell_totals[1] > cell_totals[0])] != winner)
        cell_wrong |= wrong_cells
        largest_and_cells |= biggest_wrong and wrong_cells
        fragment |= any(info["labelmap"][p] != winner for p in rec["cells"])
    raw = {s: int(np.sum(inp == s)) for s in info["signals"]}
    return dict(piece_count=len(info["pieces"]), target=info["image"].shape,
                corner=info["corner"], closed=len(closed),
                open=sum(r["open"] for r in info["regions"]),
                max_span=max(r["span"] for r in closed),
                minority_witness=bool(minority), largest_wrong=bool(largest),
                wall_cell_wrong=bool(cell_wrong), largest_and_cells_wrong=bool(largest_and_cells),
                fragment_wrong=bool(fragment), raw=raw,
                raw_wrong=any(raw[r["winner"]] < raw[next(s for s in info["signals"] if s != r["winner"])]
                              and 2 * sum(info["labels"][i] == r["winner"] for i in np.flatnonzero(r["weights"]))
                              < np.count_nonzero(r["weights"]) for r in closed),
                winners={r["winner"] for r in closed})


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one reproducible puzzle; plans never alter the solving rule."""
    nt = _checked_int(num_train, "num_train", 2, 10)
    ne = _checked_int(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    palette = tuple(map(int, rng.choice(10, 7, replace=False)))
    varying_palette = bool(rng.integers(2))
    count_shift = bool(rng.integers(2))
    first_corner = int(rng.integers(4))
    training_corners = (CORNERS[first_corner], CORNERS[(first_corner+2) % 4])
    testing_corners = (CORNERS[(first_corner+1) % 4], CORNERS[(first_corner+3) % 4])
    kinds = [0, 1] + [2] * (nt-2)
    rng.shuffle(kinds)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", nt), ("test", ne)):
        for i in range(total):
            if count_shift:
                count = int(rng.choice((3, 4) if split == "train" else (5, 6)))
                corner = training_corners[i % 2] if split == "train" else str(rng.choice(training_corners))
            else:
                count = int(rng.integers(3, 7))
                corner = (training_corners if split == "train" else testing_corners)[i % 2]
            roles = tuple(map(int, rng.choice(10, 7, replace=False))) if varying_palette else palette
            kind = kinds[i] if split == "train" else 2
            for _ in range(8):
                pair = _make_pair(rng, count, corner, roles, kind)
                key = pair["input"].shape, pair["input"].tobytes()
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not produce a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate returned cells, elections and training witnesses, not RNG tags."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (_grid(pair[k]) for k in ("input", "output"))
            assert inp.shape == out.shape
            key = inp.shape, inp.tobytes()
            assert key not in seen
            seen.add(key)
            expected, info = _infer(*key)
            assert np.array_equal(out, expected)
            assert not np.any(out == info["marker"])
            assert all(out[p] == info["bg"] for p in _marker(info["point"], info["corner"]))
            assert np.sum(out == info["bg"]) == np.sum(inp == info["bg"]) + 3
            for color in (info["carrier"], info["ink"]):
                assert np.sum(out == color) == np.sum(inp == color)
            oy, ox = info["origin"]
            for i, (m, old) in enumerate(info["pieces"]):
                r, c = info["positions"][i]
                block = out[oy+r:oy+r+m.shape[0], ox+c:ox+c+m.shape[1]]
                assert np.array_equal(block[m != info["well"]], m[m != info["well"]])
                if i == info["anchor"]:
                    assert (oy+r, ox+c) == old
            m = _measure(info, inp)
            assert m["closed"] >= 2 and m["open"] >= 1
            assert m["max_span"] >= 3 and m["fragment_wrong"]
            assert m["winners"] == set(info["signals"])
            if split == "train":
                train.append(m)
    assert any(m["minority_witness"] and m["raw_wrong"] for m in train)
    assert any(m["largest_and_cells_wrong"] for m in train)


def self_test():
    """42 seeds, all legal counts, determinism, diversity and semantic mutations.

    No files, logs or traces are retained. The 42 puzzles are processed one at a
    time, followed by a few determinism/mutation fixtures and the bounded cache.
    """
    # Arithmetic fixture: a sealed rectangular ring surrounds a wall island.
    # Middle-strip walls supply 14 unit sides but only nine distinct cells;
    # the two outer strips supply 5+5 sides/cells. Correct result is signal 7.
    ring = np.full((5, 7), 5, dtype=np.int8)
    ring[1, 1:6] = ring[3, 1:6] = 2
    ring[2, 1] = ring[2, 5] = 2
    ids = np.zeros(ring.shape, dtype=np.int8)
    ids[:, 2:5], ids[:, 5:] = 1, 2
    filled, recs = _fill(ring, ids, 2, [8, 7, 8], (7, 8))
    assert len(recs) == 1 and recs[0]["weights"].tolist() == [5, 14, 5]
    assert recs[0]["cell_weights"].tolist() == [5, 9, 5]
    assert np.all(filled[ring == 2] == 7) and np.all(filled[ring == 5] == 5)
    leaked = ring.copy()
    leaked[0, 1] = 2
    assert np.array_equal(_fill(leaked, ids, 2, [8, 7, 8], (7, 8))[0], leaked)

    shapes, targets, layouts, palettes, corners, counts, weights = (set() for _ in range(7))
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                predicted, info = _infer(inp.shape, inp.tobytes())
                assert np.array_equal(solve(inp.tolist()), pair["output"])
                digest.update(inp.tobytes())
                digest.update(predicted.tobytes())
                shapes.add(inp.shape)
                targets.add(info["image"].shape)
                palettes.add(tuple(sorted(map(int, np.unique(inp)))))
                corners.add(info["corner"])
                counts.add(len(info["pieces"]))
                normalized = {}
                signature = bytes(normalized.setdefault(int(x), len(normalized)) for x in info["image"].flat)
                layouts.add((info["image"].shape, signature))
                for rec in info["regions"]:
                    if not rec["open"]:
                        weights.add(tuple(sorted(map(int, rec["weights"][rec["weights"] > 0]))))
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42 and len(layouts) >= 180
    assert len(shapes) >= 20 and len(targets) >= 30 and len(palettes) >= 12
    assert corners == set(CORNERS) and counts == {3, 4, 5, 6} and len(weights) >= 25
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1) + 1):
            validate_generated_puzzle(generate_puzzle(104729 + nt*10 + ne, nt, ne))
    a, b = generate_puzzle(177, 2, 1), generate_puzzle(177, 2, 1)
    for split in a:
        for pa, pb in zip(a[split], b[split]):
            assert all(np.array_equal(pa[k], pb[k]) for k in pa)
    # Whole-scene symmetry and arbitrary palette permutations preserve the rule.
    pair = a["test"][0]
    for turns in (1, 2, 3):
        assert np.array_equal(solve(np.rot90(pair["input"], turns)), np.rot90(pair["output"], turns))
    assert np.array_equal(solve(pair["input"][:, ::-1]), pair["output"][:, ::-1])
    permutation = np.array([9, 2, 6, 0, 8, 1, 4, 7, 3, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    # Piece-list order cannot pick an assembly or change a wall's ownership.
    _, info = _infer(pair["input"].shape, pair["input"].tobytes())
    order = list(reversed(range(len(info["pieces"]))))
    reordered = dict(info, pieces=[info["pieces"][i] for i in order],
                     labels=[info["labels"][i] for i in order], anchor=order.index(info["anchor"]))
    reconstructed = _assemble(reordered, pair["input"].shape)
    assert np.array_equal(reconstructed[0], info["image"])
    assert np.array_equal(reconstructed[2], info["labelmap"])
    # Uniform same-size pieces admit several aspect ratios: reject ambiguity.
    ambiguous = dict(pieces=[(np.full((5, 5), 5, dtype=np.int8), (2, 2)) for _ in range(3)],
                     labels=[7, 7, 7], anchor=0, point=(2, 2), corner="TL")
    try:
        _assemble(ambiguous, (22, 22))
    except AssertionError as error:
        assert "multiple" in str(error)
    else:
        raise AssertionError("ambiguous assemblies accepted")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(19, 2, 1)
    for corruption in ("vote", "open", "marker", "duplicate"):
        q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = q["test"][0]
        inp = pair["input"]
        _, info = _infer(inp.shape, inp.tobytes())
        oy, ox = info["origin"]
        if corruption in ("vote", "open"):
            rec = next(r for r in info["regions"] if r["open"] == (corruption == "open"))
            y, x = rec["cells"][0]
            pair["output"][oy+y, ox+x] = (info["signals"][0] if rec["open"]
                                         else next(s for s in info["signals"] if s != rec["winner"]))
        elif corruption == "marker":
            pair["output"][_marker(info["point"], info["corner"])[0]] = info["marker"]
        else:
            q["train"][1] = {k: v.copy() for k, v in q["train"][0].items()}
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError("accepted corruption: " + corruption)


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
