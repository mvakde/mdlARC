"""Shape-keyed pigment reservoirs. Only the standard library and NumPy.

The solver recovers all conventions and data from one input. Generation has
no persisted state, and the parser has no dependency on the sampled vocabulary.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "Grids must have dimensions 1..30.")
    _require(np.issubdtype(a.dtype, np.integer) and
             np.all((0 <= a) & (a <= 9)), "Grids need integer colors 0..9.")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        queue = [(r, c)]
        seen[r, c] = True
        for y, x in queue:
            for dy, dx in STEPS:
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and mask[ny, nx]
                        and not seen[ny, nx]):
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        result.append(tuple(sorted(queue)))
    return result


def _normal(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - y, c - x) for r, c in cells))


def _rotate(cells):
    return _normal((c, -r) for r, c in cells)


def _reflect(cells):
    return _normal((r, -c) for r, c in cells)


@lru_cache(maxsize=4096)
def _rotations(shape):
    shapes = set()
    for _ in range(4):
        shapes.add(shape)
        shape = _rotate(shape)
    return tuple(sorted(shapes))


@lru_cache(maxsize=4096)
def _variants(shape):
    return tuple(sorted(set(_rotations(shape)) |
                        set(_rotations(_reflect(shape)))))


def _canonical(cells):
    return min(_variants(_normal(cells)))


def _size(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _hole_free(shape):
    h, w = _size(shape)
    complement = np.ones((h + 2, w + 2), bool)
    for r, c in shape:
        complement[r + 1, c + 1] = False
    return len(_components(complement)) == 1


def _distance(wall, bbox):
    """All warehouse pixels are simultaneous distance-zero sources."""
    y, x, h, w = bbox
    distance = np.full(wall.shape, -1, dtype=np.int16)
    distance[y:y + h, x:x + w] = 0
    queue = [(r, c) for r in range(y, y + h) for c in range(x, x + w)]
    for r, c in queue:
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < wall.shape[0] and 0 <= nc < wall.shape[1]
                    and not wall[nr, nc] and distance[nr, nc] < 0):
                distance[nr, nc] = distance[r, c] + 1
                queue.append((nr, nc))
    return distance


def _direct_distance(shape, bbox):
    y, x, h, w = bbox
    rr, cc = np.indices(shape)
    return (np.maximum(y - rr, 0) + np.maximum(rr - y - h + 1, 0) +
            np.maximum(x - cc, 0) + np.maximum(cc - x - w + 1, 0))


def _parse(value):
    a = _grid(value)
    wall_color = int(a[0, 0])
    _require(np.all(a[0] == wall_color) and np.all(a[-1] == wall_color)
             and np.all(a[:, 0] == wall_color) and
             np.all(a[:, -1] == wall_color), "A complete wall frame is required.")
    candidates = []
    for color in map(int, np.unique(a)):
        if color == wall_color:
            continue
        total = int(np.count_nonzero(a == color))
        for cells in _components(a == color):
            if len(cells) != total or len(cells) < 8:
                continue
            rs, cs = zip(*cells)
            y, x = min(rs), min(cs)
            h, w = max(rs) - y + 1, max(cs) - x + 1
            if min(h, w) < 3:
                continue
            box = a[y:y + h, x:x + w]
            other = set(map(int, np.unique(box))) - {color}
            if (len(other) == 1 and wall_color not in other and
                    np.all(box[0] == color) and np.all(box[-1] == color)
                    and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)):
                candidates.append((color, other.pop(), (y, x, h, w)))
    _require(len(candidates) == 1, "The cavity-bearing carrier must be unique.")
    carrier, bg, bbox = candidates[0]
    y, x, h, w = bbox
    box = a[y:y + h, x:x + w]
    cavities = [tuple((r + y, c + x) for r, c in cells)
                for cells in _components(box == bg)]
    _require(cavities, "A warehouse needs cavities.")
    for cells in cavities:
        _require(_hole_free(_normal(cells)), "Cavities must be hole-free.")
    structural = {wall_color, carrier, bg}
    mask = ~np.isin(a, tuple(structural))
    _require(not mask[y:y + h, x:x + w].any(), "Packets must be exterior.")
    objects = []
    for color in map(int, np.unique(a[mask])):
        for cells in _components(a == color):
            _require(_hole_free(_normal(cells)), "Packets must be hole-free.")
            objects.append((color, cells))
    _require(len(_components(mask)) == len(objects),
             "Different packets may not touch orthogonally.")
    target_shapes = {_canonical(c) for c in cavities}
    _require({_canonical(c) for color, c in objects} == target_shapes,
             "Every packet and cavity class needs a matching counterpart.")
    wall = a == wall_color
    distance = _distance(wall, bbox)
    for color, cells in objects:
        statuses = {bool(distance[cell] >= 0) for cell in cells}
        _require(len(statuses) == 1, "A connected packet has one reachability status.")
    return dict(grid=a, wall_color=wall_color, carrier=carrier, background=bg,
                bbox=bbox, cavities=cavities, objects=objects, wall=wall,
                distance=distance)


def _signature(cells, mode):
    raw = _normal(cells)
    if mode == "area":
        return len(raw)
    if mode == "bbox":
        return tuple(sorted(_size(raw)))
    if mode == "rotation":
        return min(_rotations(raw))
    return _canonical(raw)


def _transform(data, grouping="free", all_votes=False, metric="path",
               unlimited=False, serialization="global"):
    """Also supplies concrete counterfactuals for family-specific validation."""
    a, distance = data["grid"], data["distance"]
    slots, cavity_groups = defaultdict(list), defaultdict(list)
    for cells in data["cavities"]:
        key = _signature(cells, grouping)
        slots[key].extend(cells)
        cavity_groups[key].append(cells)
    packets, votes = defaultdict(list), defaultdict(Counter)
    for color, cells in data["objects"]:
        key = _signature(cells, grouping)
        if key not in slots:
            continue
        if all_votes or distance[cells[0]] >= 0:
            votes[key][color] += 1
            packets[key].append((color, cells))
    rank = _direct_distance(a.shape, data["bbox"]) if metric == "manhattan" else distance
    out, details = a.copy(), []
    for key, target in slots.items():
        counts = votes[key]
        _require(counts, "Each class needs a reachable ballot.")
        high = max(counts.values())
        winners = [c for c, n in counts.items() if n == high]
        _require(len(winners) == 1, "Reachable plurality must be unique.")
        winner = winners[0]
        # With all_votes the counterfactual also ignores wall membership during
        # transfer, giving it a concrete output rather than a hidden convention.
        supply = [cell for color, cells in packets[key] if color == winner for cell in cells]
        supply.sort(key=lambda cell: (int(rank[cell]), cell))
        cap = len(target)
        n = min(cap, len(supply))
        if metric == "path" and not all_votes and len(supply) > cap:
            _require(rank[supply[cap - 1]] < rank[supply[cap]],
                     "Capacity frontier must have a strict path-distance gap.")
        selected = supply if unlimited else supply[:n]
        for cell in selected:
            out[cell] = data["background"]
        if serialization == "component":
            ordered = [cell for group in sorted(cavity_groups[key], key=lambda g: min(g))
                       for cell in sorted(group)]
        else:
            ordered = sorted(target)
        for cell in ordered[:n]:
            out[cell] = winner
        details.append(dict(shape=key, winner=winner, votes=dict(counts),
                            capacity=cap, supply=tuple(supply), selected=tuple(supply[:n]),
                            target=tuple(sorted(target)), filled=n))
    return out, details


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return an answer using only this input's geometry, colors and packets."""
    return _transform(_parse(input_grid))[0]


def _alternative(data, **kwargs):
    try:
        return _transform(data, **kwargs)[0]
    except AssertionError:
        return None


def _different(data, expected, **kwargs):
    alternative = _alternative(data, **kwargs)
    return alternative is not None and not np.array_equal(alternative, expected)


def _choice(rng, items):
    return items[int(rng.integers(len(items)))]


def _shape_pool(rng):
    seeds = [((0, 0),), ((0, 0), (0, 1)),
             ((0, 1), (0, 2), (1, 0), (1, 1)),  # chiral S tetromino
             ((0, 0), (0, 1), (0, 2), (1, 1)),
             ((0, 0), (1, 0), (1, 1), (1, 2)),
             ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))]
    pool = {_canonical(s) for s in seeds}
    for length in range(3, 7):
        pool.add(_canonical(tuple((0, c) for c in range(length))))
    for h in range(2, 5):
        for w in range(2, 5):
            pool.add(_canonical(tuple({(r, 0) for r in range(h)} |
                                      {(h - 1, c) for c in range(w)})))
            if h * w <= 8:
                pool.add(_canonical(tuple((r, c) for r in range(h) for c in range(w))))
            if w >= 3 and 2 * h + w - 2 <= 9:
                pool.add(_canonical(tuple({(r, c) for r in range(h) for c in (0, w - 1)} |
                                          {(h - 1, c) for c in range(w)})))
    for _ in range(90):
        cells = {(0, 0)}
        for _ in range(int(rng.integers(3, 9))):
            frontier = sorted({(r + dr, c + dc) for r, c in cells
                               for dr, dc in STEPS} - cells)
            cells.add(_choice(rng, frontier))
        shape = _canonical(cells)
        if max(_size(shape)) <= 5 and _hole_free(shape):
            pool.add(shape)
    return tuple(sorted(pool))


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r - 1):min(h, r + 2),
                  max(0, c - 1):min(w, c + 2)] = False


def _pack(rng, available, variants, allowed=None):
    permitted = available if allowed is None else available & allowed
    for index in rng.permutation(len(variants)):
        shape = variants[int(index)]
        sh, sw = _size(shape)
        nr, nc = available.shape[0] - sh + 1, available.shape[1] - sw + 1
        if nr <= 0 or nc <= 0:
            continue
        valid = np.ones((nr, nc), bool)
        for r, c in shape:
            valid &= permitted[r:r + nr, c:c + nc]
        choices = np.argwhere(valid)
        if len(choices):
            y, x = map(int, _choice(rng, choices))
            cells = tuple((r + y, c + x) for r, c in shape)
            _block(available, cells)
            return cells
    return None


def _pack_frontier_part(rng, available, variants, distance, threshold, count):
    """Place a packet with exactly `count` pixels inside a path-distance wave."""
    permitted = available & (distance >= 0)
    for index in rng.permutation(len(variants)):
        shape = variants[int(index)]
        sh, sw = _size(shape)
        nr, nc = available.shape[0] - sh + 1, available.shape[1] - sw + 1
        if nr <= 0 or nc <= 0:
            continue
        valid = np.ones((nr, nc), bool)
        inside = np.zeros((nr, nc), dtype=np.int8)
        for r, c in shape:
            valid &= permitted[r:r + nr, c:c + nc]
            inside += distance[r:r + nr, c:c + nc] <= threshold
        choices = np.argwhere(valid & (inside == count))
        if len(choices):
            y, x = map(int, _choice(rng, choices))
            cells = tuple((r + y, c + x) for r, c in shape)
            _block(available, cells)
            return cells
    return None


def _split_supply(rng, available, shape, distance):
    """Three packets contribute s + k + (s-k) cells to a 2s-slot reservoir.

    The latter two straddle the same wavefront. All selected pixels have
    distance <= threshold; every retained pixel is strictly farther.
    """
    area = len(shape)
    thresholds = list(range(3, min(30, int(distance.max()))))
    for j in rng.permutation(len(thresholds)):
        threshold = thresholds[int(j)]
        for count in rng.permutation(np.arange(1, area)):
            trial = available.copy()
            first = _pack_frontier_part(rng, trial, _variants(shape), distance,
                                        threshold, int(count))
            if first is None:
                continue
            second = _pack_frontier_part(rng, trial, _variants(shape), distance,
                                         threshold, area - int(count))
            if second is None:
                continue
            whole = _pack(rng, trial, _variants(shape),
                          (distance >= 0) & (distance <= threshold))
            if whole is not None:
                available[:] = trial
                return (first, second, whole)
    return None


def _rectangle(wall, y, x, h, w):
    wall[y, x:x + w] = True
    wall[y + h - 1, x:x + w] = True
    wall[y:y + h, x] = True
    wall[y:y + h, x + w - 1] = True


def _walls(rng, shape, bbox, mode):
    h, w = shape
    y, x, sh, sw = bbox
    wall = np.zeros(shape, bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    cut = min(w - 5, x + sw + 2)
    if mode in ("closed", "detour", "rooms"):
        wall[:, cut] = True
        if mode == "detour":
            gap = h - int(rng.integers(3, 5))
            wall[gap:gap + int(rng.integers(1, 3)), cut] = False
        elif mode == "rooms":
            for _ in range(int(rng.integers(1, 4))):
                gap = int(rng.integers(2, h - 2))
                wall[gap:gap + 2, cut] = False
            row = int(rng.integers(2, h - 2))
            wall[row, :] = True
            for _ in range(int(rng.integers(0, 4))):
                col = int(rng.integers(2, w - 3))
                wall[row, col:col + 2] = False
    if mode in ("segments", "pockets", "rooms"):
        for _ in range(int(rng.integers(1, 5))):
            yy, xx = int(rng.integers(2, h - 5)), int(rng.integers(2, w - 5))
            hh, ww = int(rng.integers(3, min(8, h - yy))), int(rng.integers(3, min(8, w - xx)))
            if not (yy + hh < y or yy > y + sh or xx + ww < x or xx > x + sw):
                continue
            if mode == "pockets" or rng.random() < .4:
                _rectangle(wall, yy, xx, hh, ww)
                if rng.random() < .4:
                    wall[yy, xx + 1:xx + min(ww - 1, 3)] = False
            else:
                wall[yy, xx:xx + ww] = True
                if rng.random() < .5:
                    wall[yy:yy + hh, xx + ww - 1] = True
    wall[y:y + sh, x:x + sw] = False
    return wall, cut


def _ballot(rng):
    return _choice(rng, ((1,), (2,), (3,), (4,), (2, 1), (3, 1),
                         (3, 2), (3, 2, 2), (2, 1, 1), (4, 2, 1)))


def _cavity_layout(rng, shapes, multiplicities, teaching, fallback):
    if teaching == "partial":
        ah, aw = _size(shapes[0])
        ch, cw = _size(shapes[1])
        sh = ah + max(ch, cw) + 3 + int(rng.integers(3))
        sw = 3 * (aw + 1) + 1 + int(rng.integers(3))
        available = np.ones((sh, sw), bool)
        available[[0, -1]] = False
        available[:, [0, -1]] = False
        cavities = []
        for i in range(3):
            cells = tuple((r + 1, c + 1 + (aw + 1) * i) for r, c in shapes[0])
            _block(available, cells)
            cavities.append((shapes[0], cells))
        group = _pack(rng, available, _variants(shapes[1]))
        if group is None:
            return None
        cavities.append((shapes[1], group))
        return (sh, sw), cavities
    footprint = sum((_size(s)[0] + 1) * (_size(s)[1] + 1) * multiplicities[s]
                    for s in shapes)
    if teaching == "transport":
        sh, sw = int(rng.integers(8, 12)), int(rng.integers(9, 14))
    elif fallback:
        sh, sw = 16, 18
    else:
        category = int(rng.integers(3))
        small = int(np.clip(np.sqrt(footprint) + rng.integers(1, 5), 6, 14))
        large = int(np.clip(np.ceil(footprint / max(3, small - 2)) + 3, 8, 18))
        sh, sw = (small, large) if category == 0 else (large, small) if category == 1 else (small, small)
    available = np.ones((sh, sw), bool)
    available[[0, -1]] = False
    available[:, [0, -1]] = False
    cavities = []
    requests = [s for s in shapes for _ in range(multiplicities[s])]
    requests = [requests[int(i)] for i in rng.permutation(len(requests))]
    requests.sort(key=lambda s: (max(_size(s)), len(s)), reverse=True)
    for s in requests:
        cells = _pack(rng, available, _variants(s))
        if cells is None:
            return None
        cavities.append((s, cells))
    carrier = available.copy()
    carrier[:] = True
    for s, cells in cavities:
        for cell in cells:
            carrier[cell] = False
    if len(_components(carrier)) != 1:
        return None
    return (sh, sw), cavities


def _make_pair(rng, shapes, teaching=None, forbidden_winner=None):
    """Bounded packing of one integrated scene, with visible evidence targets."""
    palette = list(map(int, rng.permutation(10)))
    bg, wall_color, carrier = palette[:3]
    pigments = palette[3:]
    winners, ballots, multiplicities = {}, {}, {}
    for i, shape in enumerate(shapes):
        choices = [c for c in pigments if not (i == 0 and c == forbidden_winner)]
        if teaching and i == 1:
            choices = [c for c in choices if c != winners[shapes[0]]]
        winners[shape] = _choice(rng, choices)
        ballots[shape] = _ballot(rng)
        multiplicities[shape] = int(rng.integers(1, 4))
    if teaching == "partial":
        multiplicities[shapes[0]], multiplicities[shapes[1]] = 3, 1
        ballots[shapes[0]], ballots[shapes[1]] = (2, 1), (1,)
    elif teaching == "transport":
        multiplicities[shapes[0]], multiplicities[shapes[1]] = int(rng.integers(1, 3)), 2
        ballots[shapes[0]], ballots[shapes[1]] = (3, 2, 2), (3, 1)
    else:
        # Joint physical budgets are sampling constraints, never solver rules.
        # Large repertoires retain fewer repeats or cheaper ballots so that
        # separated objects and cavities can coexist on one 30x30 canvas.
        def cost(s):
            h, w = _size(s)
            return (h + 1) * (w + 1)
        while sum(cost(s) * multiplicities[s] for s in shapes) > 160:
            candidates = [s for s in shapes if multiplicities[s] > 1]
            if not candidates:
                break
            s = max(candidates, key=lambda s: cost(s) * multiplicities[s])
            multiplicities[s] -= 1
        while sum(cost(s) * sum(ballots[s]) for s in shapes) > 260:
            candidates = [s for s in shapes if sum(ballots[s]) > 1]
            if not candidates:
                break
            s = max(candidates, key=lambda s: cost(s) * sum(ballots[s]))
            ballots[s] = (1,) if sum(ballots[s]) <= 3 or rng.random() < .6 else (2, 1)
    color_ballots = {}
    for shape in shapes:
        losers = [c for c in pigments if c != winners[shape]]
        if teaching == "partial" and shape == shapes[0]:
            # Otherwise the companion's vote can tie the combined area ballot,
            # obscuring the concrete wrong-color shortcut we intend to teach.
            losers = [c for c in losers if c != winners[shapes[1]]]
        rng.shuffle(losers)
        color_ballots[shape] = tuple(zip([winners[shape]] + losers, ballots[shape]))
    for attempt in range(144):
        fallback = not teaching and attempt >= 112
        layout = _cavity_layout(rng, shapes, multiplicities, teaching, fallback)
        if layout is None:
            continue
        (sh, sw), local_cavities = layout
        h = int(rng.integers(24 if teaching else 22, 31))
        w = int(rng.integers(26 if teaching else 22, 31))
        if fallback:
            h = w = 30
        if teaching:
            y, x = int(rng.integers(1, 4)), int(rng.integers(1, 3))
        else:
            y, x = int(rng.integers(1, h - sh)), int(rng.integers(1, w - sw))
        bbox = y, x, sh, sw
        # Teaching requires an exterior region wide enough for all its votes.
        if teaching and x + sw + 2 > w - 6:
            continue
        mode = ("closed" if teaching == "partial" else "detour" if teaching
                else "open" if fallback else
                _choice(rng, ("open", "closed", "detour", "segments", "pockets", "rooms")))
        wall, cut = _walls(rng, (h, w), bbox, mode)
        distance = _distance(wall, bbox)
        direct = _direct_distance((h, w), bbox)
        available = ~wall
        available[max(0, y - 1):y + sh + 1, max(0, x - 1):x + sw + 1] = False
        reachable = distance >= 0
        unreachable = (distance < 0) & ~wall
        rr, cc = np.indices((h, w))
        requests = []
        for shape in shapes:
            for color, count in color_ballots[shape]:
                if (teaching == "transport" and shape == shapes[1]
                        and color == winners[shape]):
                    # This pigment is placed against a strict cell wavefront,
                    # explicitly separating cell transport from whole packets.
                    continue
                for i in range(count):
                    allowed = reachable
                    variants = _variants(shape)
                    if teaching == "partial" and shape == shapes[0]:
                        variants = (_reflect(shape),) if color == winners[shape] else (shape,)
                    if teaching == "transport" and shape == shapes[0] and color == winners[shape]:
                        if i < multiplicities[shape]:
                            allowed = reachable & (cc < cut) & (distance >= 7) & (distance <= 14)
                        else:
                            allowed = reachable & (cc > cut) & (direct <= 7) & (rr <= y + sh + 1)
                    requests.append((shape, color, variants, allowed))
            if teaching == "partial" and shape == shapes[0]:
                loser = color_ballots[shape][1][0]
                requests.extend((shape, loser, _variants(shape), unreachable) for _ in range(3))
            elif not teaching and unreachable.any():
                requests.extend((shape, _choice(rng, pigments), _variants(shape), unreachable)
                                for _ in range(int(rng.integers(4))))
        # Constrained transport anchors go first; large remaining packets next.
        order = list(rng.permutation(len(requests)))
        order.sort(key=lambda i: (int(requests[int(i)][3].sum()),
                                  -max(_size(requests[int(i)][0]))))
        objects, failed = [], False
        for i in order:
            shape, color, variants, allowed = requests[int(i)]
            cells = _pack(rng, available, variants, allowed)
            if cells is None:
                failed = True
                break
            objects.append((color, cells))
        if failed:
            continue
        if teaching == "transport":
            pieces = _split_supply(rng, available, shapes[1], distance)
            if pieces is None:
                continue
            objects.extend((winners[shapes[1]], cells) for cells in pieces)
        a = np.full((h, w), bg, dtype=np.int8)
        a[wall] = wall_color
        a[y:y + sh, x:x + sw] = carrier
        for shape, cells in local_cavities:
            for r, c in cells:
                a[y + r, x + c] = bg
        for color, cells in objects:
            for cell in cells:
                a[cell] = color
        # Orient the scene before solving: the global prefix is always displayed.
        a = np.rot90(a, int(rng.integers(4)))
        if rng.random() < .5:
            a = a[:, ::-1]
        a = a.copy()
        try:
            data = _parse(a)
            out, details = _transform(data)
        except AssertionError:
            continue
        if teaching == "partial":
            if not all(_different(data, out, **alt) for alt in
                       (dict(all_votes=True), dict(grouping="area"),
                        dict(grouping="bbox"), dict(grouping="rotation"),
                        dict(serialization="component"))):
                continue
        elif teaching == "transport":
            if not (_different(data, out, metric="manhattan") and
                    _different(data, out, unlimited=True)):
                continue
        return dict(input=a, output=out), winners[shapes[0]]
    raise AssertionError("Bounded scene packing failed; shapes/ballots were not silently dropped.")


def _counts(num_train, num_test):
    for value in (num_train, num_test):
        if (not isinstance(value, (int, np.integer)) or
                isinstance(value, (bool, np.bool_))):
            raise ValueError("Example counts must be integers.")
    num_train, num_test = int(num_train), int(num_test)
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require 2..10 training, 1..4 test, and num_train > num_test.")
    return num_train, num_test


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer.")
    rng = np.random.default_rng(int(seed))
    pool = _shape_pool(rng)
    anchors = [s for s in pool if 4 <= len(s) <= 6 and max(_size(s)) <= 3
               and _reflect(s) not in _rotations(s)
               and any(t != s and len(t) == len(s) and
                       sorted(_size(t)) == sorted(_size(s)) for t in pool)]
    anchor = _choice(rng, anchors)
    companion = _choice(rng, [t for t in pool if t != anchor and len(t) == len(anchor)
                             and sorted(_size(t)) == sorted(_size(anchor))])
    puzzle, seen, known = {"train": [], "test": []}, set(), set()
    first_winner = None
    for i in range(num_train + num_test):
        split = "train" if i < num_train else "test"
        teaching = "partial" if i == 0 else "transport" if i == 1 else None
        if teaching:
            shapes = (anchor, companion)
        else:
            k = int(rng.integers(1, 5))
            # Packing budgets favor compact objects when four classes coexist.
            available = [s for s in pool if len(s) <= (5 if k == 4 else 8)]
            chosen = [available[int(j)] for j in rng.choice(len(available), k, replace=False)]
            if split == "test":
                fresh = [s for s in available if s not in known and max(_size(s)) >= 4]
                if not fresh:
                    fresh = [s for s in available if s not in known]
                if not fresh:
                    fresh = [s for s in pool if s not in known]
                _require(fresh, "The procedural pool must retain fresh test support.")
                chosen[0] = _choice(rng, fresh)
            shapes = tuple(dict.fromkeys(chosen))
        for _ in range(12):
            pair, winner = _make_pair(rng, shapes, teaching,
                                      first_winner if i == 1 else None)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise AssertionError("Duplicate scenes after bounded retries.")
        seen.add(_key(pair["input"]))
        puzzle[split].append(pair)
        if split == "train":
            known.update(shapes)
        if i == 0:
            first_winner = winner
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must have exactly train/test keys.")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists.")
    try:
        _counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, records = set(), {"train": [], "test": []}
    for split in records:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "Pairs must have exactly input/output keys.")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "Scene dimensions are preserved.")
            _require(_key(a) not in seen, "Inputs must be distinct within a puzzle.")
            seen.add(_key(a))
            data = _parse(a)
            expected, details = _transform(data)
            _require(np.array_equal(expected, b), "Wrong election, selection or storage.")
            _require(np.array_equal(np.bincount(a.ravel(), minlength=10),
                                   np.bincount(b.ravel(), minlength=10)),
                     "Transfer must conserve every color's pixel count.")
            structural = (a == data["wall_color"]) | (a == data["carrier"])
            _require(np.array_equal(a[structural], b[structural]), "Walls and carrier are preserved.")
            _require(not np.array_equal(a, b), "Positive support must transfer material.")
            for d in details:
                _require(0 < d["filled"] <= d["capacity"], "Visible capacity is respected.")
                target = d["target"]
                _require(all(b[cell] == d["winner"] for cell in target[:d["filled"]])
                         and all(b[cell] == data["background"] for cell in target[d["filled"]:]),
                         "A shape reservoir must contain a global row-major prefix.")
                _require(len(d["target"]) % len(d["shape"]) == 0,
                         "Pooled capacity comes from complete matching cavities.")
            records[split].append((data, expected, details))
    train = records["train"]
    for alt in (dict(all_votes=True), dict(grouping="area"), dict(grouping="bbox"),
                dict(grouping="rotation"), dict(serialization="component"),
                dict(metric="manhattan"), dict(unlimited=True)):
        _require(any(_different(data, out, **alt) for data, out, details in train),
                 "Training lacks a concrete counterfactual witness: " + str(alt))
    _require(any(d["filled"] < d["capacity"] for data, out, ds in train for d in ds),
             "Training must demonstrate finite material, not unlimited paint.")
    _require(any(any(bool(set(cells) & {p for d in ds for p in d["selected"]}) and
                         not set(cells) <= {p for d in ds for p in d["selected"]}
                         for color, cells in data["objects"])
                 for data, out, ds in train),
             "Training must split a packet at the pixel-capacity frontier.")
    _require(any(len(d["votes"]) >= 3 and max(d["votes"].values()) * 2 <=
                 sum(d["votes"].values()) for data, out, ds in train for d in ds),
             "Training must demonstrate a non-majority plurality.")
    history = defaultdict(set)
    for data, out, ds in train:
        for d in ds:
            history[d["shape"]].add(d["winner"])
    _require(any(len(colors) >= 2 for colors in history.values()),
             "A recurring training shape must change elected pigment.")
    known = set(history)
    _require(all(any(d["shape"] not in known for d in ds)
                 for data, out, ds in records["test"]), "Each test must introduce a supported new shape.")


def _features(pair):
    data = _parse(pair["input"])
    out, details = _transform(data)
    selected = {cell for d in details for cell in d["selected"]}
    split_packets = sum(bool(set(cells) & selected) and not set(cells) <= selected
                        for color, cells in data["objects"])
    return dict(shape=data["grid"].shape, bbox_shape=data["bbox"][2:],
                classes=len(details), cavities=len(data["cavities"]),
                packets=len(data["objects"]), split_packets=split_packets,
                inaccessible=sum(data["distance"][cells[0]] < 0 for color, cells in data["objects"]),
                capacities=tuple(sorted(d["capacity"] for d in details)),
                multiplicities=tuple(sorted(d["capacity"] // len(d["shape"]) for d in details)),
                ballots=tuple(sorted(tuple(sorted(d["votes"].values(), reverse=True)) for d in details)),
                vocabulary=frozenset(d["shape"] for d in details),
                partial=sum(d["filled"] < d["capacity"] for d in details),
                excess=sum(len(d["supply"]) > d["capacity"] for d in details),
                maximum_distance=int(data["distance"].max()),
                roles=(data["background"], data["wall_color"], data["carrier"]))


def self_test():
    """Bounded diversity and semantics checks over 42 reproducible seeds."""
    # A minimal 3x3 warehouse verifies that solve has no sampler-size floor.
    tiny = np.zeros((7, 12), dtype=np.int8)
    tiny[[0, -1]] = 8
    tiny[:, [0, -1]] = 8
    tiny[2:5, 2:5] = 5
    tiny[3, 3] = 0
    tiny[3, 7] = 1
    expected = tiny.copy()
    expected[3, 7], expected[3, 3] = 0, 1
    _require(np.array_equal(solve(tiny), expected), "Input-only small-family support failed.")
    # Ambiguities are excluded, rather than silently resolved by queue order.
    tied_capacity = tiny.copy()
    tied_capacity[3, 7] = 0
    tied_capacity[2, 5], tied_capacity[4, 5] = 1, 1
    tied_election = tiny.copy()
    tied_election[5, 7] = 2
    for scene, reason in ((tied_capacity, "Capacity frontier"),
                          (tied_election, "Reachable plurality")):
        try:
            solve(scene)
        except AssertionError as error:
            _require(reason in str(error), "Tie fixture failed before the intended ambiguity check.")
        else:
            raise AssertionError("An ambiguous ballot or frontier was accepted.")
    puzzles, features, signatures = [], [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        signatures.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]))
        features.extend(_features(pair) for s in ("train", "test") for pair in p[s])
    _require(len(signatures) == 42, "Seed diversity collapsed.")
    for field, minimum in (("shape", 25), ("bbox_shape", 25), ("packets", 12),
                           ("cavities", 6), ("capacities", 15), ("ballots", 15),
                           ("roles", 30), ("maximum_distance", 15)):
        _require(len({f[field] for f in features}) >= minimum, "Missing variation: " + field)
    vocabulary = set().union(*(f["vocabulary"] for f in features))
    _require(len(vocabulary) >= 35, "Vocabulary diversity collapsed.")
    _require({1, 2, 3, 4} <= {f["classes"] for f in features}, "Class count diversity collapsed.")
    _require(any(f["split_packets"] for f in features), "Pixel frontiers never split a packet.")
    _require(any(f["inaccessible"] == 0 for f in features) and
             any(f["inaccessible"] >= 3 for f in features), "Reachability variety collapsed.")
    _require(any(f["partial"] and f["excess"] for f in features), "Coupled reservoir regimes absent.")
    _require({1, 2, 3} <= {m for f in features for m in f["multiplicities"]},
             "Visible-capacity variation collapsed.")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(9073, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "Requested counts ignored.")
    p, q = generate_puzzle(19001, 2, 1), generate_puzzle(19001, 2, 1)
    _require(all(np.array_equal(p[s][i][r], q[s][i][r]) for s in p
                 for i in range(len(p[s])) for r in ("input", "output")), "Determinism failed.")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted.")
    broken = {s: [{r: np.asarray(pair[r]).copy() for r in pair} for pair in puzzles[0][s]]
              for s in puzzles[0]}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted labels were accepted.")


if __name__ == "__main__":
    self_test()
    print("42-seed self_test passed")
