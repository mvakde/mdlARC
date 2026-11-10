"""A rigid square collector: swept catchment, first-contact priority, home counter.

Only NumPy and the standard library are used. No parent modules, files, latent
generation metadata, or training examples are consulted by solve().
"""
from __future__ import annotations

from collections import Counter
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_INF = 32767


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError("Example counts must be integers.")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require 2..10 train, 1..4 test, and num_train > num_test.")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("A grid must be a 1..30 by 1..30 integer array of colors 0..9.")
    return a.astype(np.int16, copy=False)


def _decode(a):
    """Geometry and the center dot encode every role locally."""
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    if not np.all(border == border[0]):
        raise ValueError("The complete border must have the wall color.")
    wall = int(border[0])
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 4:
        raise ValueError("Exactly four input colors are required.")
    candidates = []
    for color, count in zip(colors, counts):
        if color == wall or count not in (8, 24):
            continue
        k = 3 if count == 8 else 5
        cells = np.argwhere(a == color)
        r, c = map(int, cells.min(axis=0))
        bottom, right = map(int, cells.max(axis=0))
        if (bottom, right) != (r+k-1, c+k-1):
            continue
        patch = a[r:r+k, c:c+k]
        pattern = np.ones((k, k), dtype=bool)
        pattern[k//2, k//2] = False
        payload = int(patch[k//2, k//2])
        if payload != wall and np.array_equal(patch == color, pattern):
            candidates.append((r, c, k, int(color), payload))
    if len(candidates) != 1:
        raise ValueError("One unique square with a contrasting center dot is required.")
    r, c, k, casing, payload = candidates[0]
    floor_candidates = set(map(int, colors)) - {wall, casing, payload}
    if len(floor_candidates) != 1:
        raise ValueError("Floor role is ambiguous.")
    floor = floor_candidates.pop()
    home = np.zeros(a.shape, dtype=bool)
    home[r:r+k, c:c+k] = True
    return dict(wall=wall, floor=floor, casing=casing, payload=payload,
                pos=(r, c), k=k, home=home)


def _bfs(allowed, starts):
    distance = np.full(allowed.shape, -1, dtype=np.int16)
    queue = [tuple(map(int, p)) for p in starts]
    for p in queue:
        if not allowed[p]:
            raise ValueError("A search source is not traversable.")
        distance[p] = 0
    h, w = allowed.shape
    for r, c in queue:
        for dr, dc in _STEPS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = distance[r, c] + 1
                queue.append((nr, nc))
    return distance


def _fields(allowed, pos, k):
    """Erode into legal placements, search, then project distances onto pixels."""
    valid = np.lib.stride_tricks.sliding_window_view(allowed, (k, k)).all(axis=(-1, -2))
    placement = _bfs(valid, [pos])
    contact = np.full(allowed.shape, _INF, dtype=np.int16)
    centers = np.zeros(allowed.shape, dtype=bool)
    for r, c in np.argwhere(placement >= 0):
        patch = contact[r:r+k, c:c+k]
        np.minimum(patch, placement[r, c], out=patch)
        centers[r+k//2, c+k//2] = True
    return valid, placement, contact, centers


def _selected(supply, contact, capacity):
    eligible = supply & (contact < _INF)
    costs = np.sort(contact[eligible])
    n = min(capacity, len(costs))
    if len(costs) > capacity and costs[capacity-1] == costs[capacity]:
        raise ValueError("A first-contact tie crosses the counter capacity.")
    selected = eligible & (contact <= costs[n-1]) if n else np.zeros(supply.shape, dtype=bool)
    return selected, eligible


def _analyze(a):
    roles = _decode(a)
    k, pos, home = roles['k'], roles['pos'], roles['home']
    allowed = a != roles['wall']
    valid, placement, contact, centers = _fields(allowed, pos, k)
    supply = (a == roles['payload']) & ~home
    selected, eligible = _selected(supply, contact, k*k)
    point = _bfs(allowed, np.argwhere(home))
    r, c = pos
    rr, cc = np.indices(a.shape)
    direct = np.maximum(r-rr, 0) + np.maximum(rr-r-k+1, 0)
    direct += np.maximum(c-cc, 0) + np.maximum(cc-c-k+1, 0)
    retained = eligible & ~selected
    connected_blocked = supply & (point >= 0) & (contact == _INF)
    disconnected = supply & (point < 0)
    def reversal(metric):
        return bool(selected.any() and retained.any()
                    and int(metric[selected].max()) > int(metric[retained].min()))
    features = dict(k=k, capacity=k*k, height=a.shape[0], width=a.shape[1],
                    pos=pos, palette=tuple(roles[t] for t in ('wall', 'floor', 'casing', 'payload')),
                    selected=int(selected.sum()), eligible=int(eligible.sum()),
                    excess=int(retained.sum()), connected_blocked=int(connected_blocked.sum()),
                    disconnected=int(disconnected.sum()),
                    edge_pickups=int((selected & ~centers).sum()),
                    point_reversal=reversal(point), direct_reversal=reversal(direct),
                    max_contact=int(contact[contact < _INF].max()),
                    placements=int((placement >= 0).sum()),
                    catchment=int((contact < _INF).sum()),
                    floor_cells=int(allowed.sum()),
                    capacity_gap=int(contact[retained].min()-contact[selected].max()) if retained.any() else 0)
    return roles, selected, features, (valid, placement, contact, centers, point)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the answer using only this input; no puzzle/seed context is needed."""
    a = _grid(input_grid)
    roles, selected, _, _ = _analyze(a)
    output = a.copy()
    output[selected] = roles['floor']
    r, c = roles['pos']
    k = roles['k']
    for i in range(int(selected.sum())):
        output[r+i//k, c+i%k] = roles['payload']
    return output.astype(np.int8)


def _loop_geometry(rng, k):
    """A thin chord and a wide U detour; parameters are not a stored grid."""
    q = k + int(rng.integers(2))
    h = int(rng.integers(max(22, 4*q+2), 31))
    w = int(rng.integers(2*q+6, min(25, h-3)+1))
    mask = np.zeros((h, w), dtype=bool)
    mask[1:h-1, 1:1+q] = True
    mask[1:h-1, w-q-1:w-1] = True
    mask[h-q-1:h-1, 1:w-1] = True
    thin = int(rng.integers(1, k))
    offset = int(rng.integers(q-thin+1))
    mask[1+offset:1+offset+thin, 1:w-1] = True
    # Narrow side recesses and independently widened stretches diversify the U.
    for _ in range(int(rng.integers(4))):
        row = int(rng.integers(q+2, h-q-2))
        thickness = int(rng.integers(1, k))
        length = int(rng.integers(1, min(5, w-2*q-2)+1))
        if rng.integers(2):
            mask[row:row+thickness, 1+q:1+q+length] = True
        else:
            mask[row:row+thickness, w-q-1-length:w-q-1] = True
    pos = (1+int(rng.integers(3)), 1+int(rng.integers(q-k+1)))
    return mask, pos


def _lattice_geometry(rng, k, complex_scene):
    """A sampled room graph, with varying edge widths, offsets, and dead ends."""
    q = k + int(rng.integers(3))
    limit = min(4, (28+3)//(q+3))
    nr = int(rng.integers(2 if complex_scene else 1, limit+1))
    nc = int(rng.integers(2, limit+1))
    def axis(n):
        gaps = np.full(n-1, q+3, dtype=int)
        budget = 28-q-int(gaps.sum())
        for i in rng.permutation(n-1):
            extra = int(rng.integers(min(4, budget)+1))
            gaps[i] += extra
            budget -= extra
        return 1+np.concatenate(([0], np.cumsum(gaps)))
    rows, cols = axis(nr), axis(nc)
    h, w = int(rows[-1]+q+1), int(cols[-1]+q+1)
    mask = np.zeros((h, w), dtype=bool)
    nodes = [(r, c) for r in range(nr) for c in range(nc)]
    for r, c in nodes:
        mask[rows[r]:rows[r]+q, cols[c]:cols[c]+q] = True
    # Random spanning tree: arbitrary junctions, not a fixed route template.
    root = nodes[int(rng.integers(len(nodes)))]
    seen, stack, edges = {root}, [root], set()
    def neighbors(node):
        r, c = node
        return [(r+dr, c+dc) for dr, dc in _STEPS
                if 0 <= r+dr < nr and 0 <= c+dc < nc]
    def edge(a, b):
        return tuple(sorted((a, b)))
    while stack:
        options = [b for b in neighbors(stack[-1]) if b not in seen]
        if not options:
            stack.pop()
        else:
            b = options[int(rng.integers(len(options)))]
            edges.add(edge(stack[-1], b))
            seen.add(b)
            stack.append(b)
    for a in nodes:
        for b in neighbors(a):
            if rng.random() < (0.35 if complex_scene else 0.12):
                edges.add(edge(a, b))
    for a, b in sorted(edges):
        # Occasional absent tree edges give genuinely disconnected rooms.
        if rng.random() < .08:
            continue
        width = int(rng.integers(1, q+1))
        offset = int(rng.integers(q-width+1))
        ar, ac = int(rows[a[0]]), int(cols[a[1]])
        br, bc = int(rows[b[0]]), int(cols[b[1]])
        if ar == br:
            mask[ar+offset:ar+offset+width, ac+q:bc] = True
        else:
            mask[ar+q:br, ac+offset:ac+offset+width] = True
    # Start in any chamber and at any legal offset inside it.
    node = nodes[int(rng.integers(len(nodes)))]
    pos = (int(rows[node[0]])+int(rng.integers(q-k+1)),
           int(cols[node[1]])+int(rng.integers(q-k+1)))
    return mask, pos


def _partition_geometry(rng, k, complex_scene):
    h, w = map(int, rng.integers(k+7, 31, 2))
    mask = np.ones((h, w), dtype=bool)
    mask[0] = mask[-1] = False
    mask[:, 0] = mask[:, -1] = False
    for _ in range(int(rng.integers(2, 9 if complex_scene else 5))):
        horizontal = bool(rng.integers(2))
        if horizontal:
            row = int(rng.integers(2, h-2))
            mask[row, 1:w-1] = False
            for _ in range(int(rng.integers(1, 4))):
                col = int(rng.integers(1, w-1))
                gap = int(rng.integers(1, k+3))
                mask[row, col:min(w-1, col+gap)] = True
        else:
            col = int(rng.integers(2, w-2))
            mask[1:h-1, col] = False
            for _ in range(int(rng.integers(1, 4))):
                row = int(rng.integers(1, h-1))
                gap = int(rng.integers(1, k+3))
                mask[row:min(h-1, row+gap), col] = True
    pos = (int(rng.integers(1, h-k)), int(rng.integers(1, w-k)))
    r, c = pos
    # The initial footprint is a real room; recompute all paths after clearing it.
    mask[r:r+k, c:c+k] = True
    return mask, pos


def _open_geometry(rng, k):
    h, w = map(int, rng.integers(k+5, 31, 2))
    mask = np.ones((h, w), dtype=bool)
    mask[0] = mask[-1] = False
    mask[:, 0] = mask[:, -1] = False
    pos = (int(rng.integers(1, h-k)), int(rng.integers(1, w-k)))
    return mask, pos


def _pick(rng, mask, count, style, forced=()):
    available = set(map(int, np.flatnonzero(mask)))
    chosen = list(dict.fromkeys(map(int, forced)))
    if not set(chosen) <= available or len(chosen) > count or count > len(available):
        raise ValueError("Impossible payload placement.")
    available.difference_update(chosen)
    if count == 0:
        return []
    anchor = chosen[0] if chosen else int(rng.choice(sorted(available)))
    ar, ac = divmod(anchor, mask.shape[1])
    horizontal = bool(rng.integers(2))
    while len(chosen) < count:
        cells = np.array(sorted(available), dtype=int)
        rr, cc = cells//mask.shape[1], cells % mask.shape[1]
        if style == 0:
            cell = int(rng.choice(cells))
        else:
            score = (rr-ar)**2+(cc-ac)**2 if style == 1 else (
                abs(rr-ar)*30+abs(cc-ac) if horizontal else abs(cc-ac)*30+abs(rr-ar))
            cell = int(rng.choice(cells[score == score.min()]))
        chosen.append(cell)
        available.remove(cell)
    return chosen


def _excess_supply(rng, eligible, contact, point, direct, capacity, excess, style, witness):
    for a in rng.permutation(np.flatnonzero(eligible)):
        a = int(a)
        threshold = int(contact.flat[a])
        near = eligible & (contact <= threshold)
        far = eligible & (contact > threshold)
        if near.sum() < capacity or far.sum() < excess:
            continue
        choice = far.copy()
        if witness:
            choice &= (point < point.flat[a]) & (direct < direct.flat[a])
        if not choice.any():
            continue
        b = int(rng.choice(np.flatnonzero(choice)))
        return (_pick(rng, near, capacity, style, [a])
                + _pick(rng, far, excess, style, [b]))
    return None


def _make_pair(rng, k, request, colors, used, complex_scene=False):
    """Bounded, conditioned sampling; witness requirements concern visible input."""
    for attempt in range(80):
        if request == 'rank' or attempt >= 55:
            allowed, pos = _loop_geometry(rng, k)
        else:
            mode = int(rng.choice(4, p=(.42, .32, .14, .12)))
            if mode == 0:
                allowed, pos = _lattice_geometry(rng, k, complex_scene)
            elif mode == 1:
                allowed, pos = _partition_geometry(rng, k, complex_scene)
            elif mode == 2:
                allowed, pos = _loop_geometry(rng, k)
            else:
                allowed, pos = _open_geometry(rng, k)
        # Transform the input geometry before applying the displayed prefix rule.
        home = np.zeros(allowed.shape, dtype=bool)
        r, c = pos
        home[r:r+k, c:c+k] = True
        turn, reflect = int(rng.integers(4)), bool(rng.integers(2))
        allowed = np.rot90(allowed, turn)
        home = np.rot90(home, turn)
        if reflect:
            allowed, home = np.fliplr(allowed), np.fliplr(home)
        allowed, home = allowed.copy(), home.copy()
        pos = tuple(map(int, np.argwhere(home).min(axis=0)))
        _, _, contact, centers = _fields(allowed, pos, k)
        point = _bfs(allowed, np.argwhere(home))
        outside = ~home
        eligible = outside & (contact < _INF)
        blocked = outside & allowed & (point >= 0) & (contact == _INF)
        disconnected = outside & allowed & (point < 0)
        capacity = k*k
        style = int(rng.integers(3))
        r, c = pos
        rr, cc = np.indices(allowed.shape)
        direct = np.maximum(r-rr, 0)+np.maximum(rr-r-k+1, 0)
        direct += np.maximum(c-cc, 0)+np.maximum(cc-c-k+1, 0)
        mode = request if request in ('partial', 'rank') else str(rng.choice(
            ['small', 'full', 'excess', 'zero'], p=(.49, .15, .31, .05)))
        if mode == 'partial':
            n = int(rng.choice([2, 4]))
            edge = eligible & ~centers
            if not edge.any() or not blocked.any() or eligible.sum() < n:
                continue
            supply = _pick(rng, eligible, n, style, [rng.choice(np.flatnonzero(edge))])
            # Point motion would gather ALL these cells, since their total is < capacity.
            distract = _pick(rng, blocked, int(rng.integers(1, min(3, capacity-n, int(blocked.sum()))+1)), style)
        elif mode in ('rank', 'excess'):
            excess = int(rng.integers(1, min(13, capacity)+1))
            supply = _excess_supply(rng, eligible, contact, point, direct, capacity, excess, style, mode == 'rank')
            if supply is None:
                continue
            distract = []
        elif mode == 'zero':
            inactive = blocked | disconnected
            if not inactive.any():
                continue
            supply = []
            distract = _pick(rng, inactive, int(rng.integers(1, min(10, int(inactive.sum()))+1)), style)
        else:
            n = capacity if mode == 'full' else int(rng.integers(1, capacity))
            if eligible.sum() < n:
                continue
            supply = _pick(rng, eligible, n, style)
            distract = []
        # Other examples vary disconnected and point-connected distraction independently.
        if mode not in ('partial', 'zero'):
            for pool in (blocked, disconnected):
                amount = int(rng.integers(min(9, int(pool.sum()))+1))
                distract += _pick(rng, pool, amount, int(rng.integers(3)))
        wall, floor, casing, payload = colors
        a = np.full(allowed.shape, wall, dtype=np.int8)
        a[allowed] = floor
        a.flat[supply+distract] = payload
        a[home] = casing
        a[r+k//2, c+k//2] = payload
        sig = _signature(a)
        if sig in used or not np.any(a == floor):
            continue
        try:
            _, _, f, _ = _analyze(a)
        except ValueError:
            continue
        if mode == 'rank' and not (f['point_reversal'] and f['direct_reversal']):
            continue
        used.add(sig)
        return {'input': a, 'output': solve(a)}
    raise RuntimeError("Bounded construction exhausted for "+request)


def _signature(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.ravel()))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    base = tuple(map(int, rng.choice(10, 4, replace=False)))
    vary_palette = bool(rng.random() < .65)
    upscale = bool(rng.random() < .55)
    # Minimal evidence survives EVERY legal count request; obligations are shuffled.
    train_k, test_k = (3, 5) if upscale else (5, 3)
    specs = [(train_k, 'partial'), (train_k, 'rank')]
    specs += [(train_k, 'random') for _ in range(num_train-2)]
    specs = [specs[int(i)] for i in rng.permutation(len(specs))]
    puzzle, used = {'train': [], 'test': []}, set()
    for split, requests in (('train', specs), ('test', [
            (test_k, 'random') for _ in range(num_test)])):
        for k, request in requests:
            colors = tuple(map(int, rng.choice(10, 4, replace=False))) if vary_palette else base
            puzzle[split].append(_make_pair(rng, k, request, colors, used,
                                           complex_scene=(split == 'test' and not upscale)))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Audit actual supplied grids, labels, and the two semantic witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle schema.')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists.')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair schema.')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'Shape changed.')
            sig = _signature(a)
            _require(sig not in seen, 'Duplicate input.')
            seen.add(sig)
            roles, selected, f, fields = _analyze(a)
            _require(np.array_equal(b, solve(a)), 'Wrong collection, priority, or counter label.')
            outside = ~roles['home']
            erased = outside & (a != b)
            _require(np.array_equal(erased, selected), 'Wrong exterior cells changed.')
            _require(np.all(b[selected] == roles['floor']), 'Selected cells must become floor.')
            _require(np.array_equal(a[a == roles['wall']], b[a == roles['wall']]), 'Walls changed.')
            _require(f['selected'] == min(f['capacity'], f['eligible']), 'Capacity invariant failed.')
            _require(f['excess'] == 0 or f['capacity_gap'] > 0, 'Ambiguous capacity frontier.')
            if f['selected'] == 0:
                _require(f['connected_blocked']+f['disconnected'] > 0, 'Zero case needs exterior payload.')
                _require(np.array_equal(a, b), 'Zero collection must be unchanged.')
            # Every swept pixel must also be point-reachable; the converse is false.
            contact, point = fields[2], fields[4]
            _require(np.all(point[contact < _INF] >= 0), 'A footprint jumped a wall.')
            features[split].append(f)
    train = features['train']
    _require(any(f['selected'] in (2, 4) and f['edge_pickups'] > 0
                 and f['connected_blocked'] > 0
                 and f['eligible']+f['connected_blocked'] < f['capacity'] for f in train),
             'Training must distinguish point reach and center-only pickup and expose prefix order.')
    _require(any(f['excess'] > 0 and f['point_reversal'] and f['direct_reversal'] for f in train),
             'Training needs excess with square/point/direct distance reversals.')
    train_sizes = {f['k'] for f in train}
    test_sizes = {f['k'] for f in features['test']}
    _require(len(train_sizes) == len(test_sizes) == 1 and train_sizes != test_sizes,
             'Generated puzzles must shift visible footprint size and counter capacity.')
    return features


def _analytic_tests():
    # Straight three-wide strip with a two-wide continuation into another room.
    # Its first nine payload cells are eligible; a point also reaches the other two.
    a = np.zeros((7, 23), dtype=np.int8)
    a[2:5, 1:13] = 1
    a[2:4, 13:18] = 1
    a[1:6, 18:22] = 1
    a[2:5, 1:4] = 2
    a[3, 2] = 9
    a[2, 4:14] = 9
    a[2, 19] = 9
    expected = a.copy()
    expected[2, 4:13] = 1  # contact costs 1..9; the tenth is unsweepable
    expected[2:5, 1:4] = 9
    _require(np.array_equal(solve(a), expected), 'Analytic footprint/capacity regression.')
    roles, selected, f, _ = _analyze(a)
    _require(f['connected_blocked'] == 2 and f['selected'] == 9, 'Thin continuation ignored.')
    # Capacity cuts a connected fourteen-cell line even though all of it is sweepable.
    line = np.zeros((7, 24), dtype=np.int8)
    line[2:5, 1:23] = 1
    line[2:5, 1:4] = 2
    line[3, 2] = 9
    line[2, 4:18] = 9
    line_expected = line.copy()
    line_expected[2, 4:13] = 1
    line_expected[2:5, 1:4] = 9
    _require(np.array_equal(solve(line), line_expected), 'Capacity did not split a reachable cluster.')
    # Exact numeric cost reversal: a thin top chord versus the square's wide U.
    detour = np.zeros((21, 13), dtype=np.int8)
    detour[1:20, 1:4] = detour[1:20, 9:12] = 1
    detour[17:20, 1:12] = 1
    detour[1, 1:12] = 1
    detour[1:4, 1:4] = 2
    detour[2, 2] = 9
    detour[4:12, 1] = detour[15, 1] = detour[1, 9] = 9
    detour_expected = detour.copy()
    detour_expected[4:12, 1] = detour_expected[15, 1] = 1
    detour_expected[1:4, 1:4] = 9
    _require(np.array_equal(solve(detour), detour_expected), 'Thin-chord priority regression.')
    _, _, _, fields = _analyze(detour)
    _require((int(fields[2][15, 1]), int(fields[2][1, 9])) == (12, 40), 'Wrong square costs.')
    _require((int(fields[4][15, 1]), int(fields[4][1, 9])) == (12, 6), 'Wrong point costs.')
    # Whole-footprint contact, even when the center cannot occupy a wall-adjacent cell.
    b = np.zeros((7, 11), dtype=np.int8)
    b[1:6, 1:10] = 1
    b[2:5, 1:4] = 2
    b[3, 2] = b[1, 5] = b[4, 8] = 9
    predicted = solve(b)
    _require(predicted[1, 5] == predicted[4, 8] == 1, 'Edge cells were not swept.')
    _require(np.array_equal(predicted[2, 1:4], np.array([9, 9, 2])), 'Prefix is not displayed row-major.')
    _require(predicted[3, 2] == 9, 'Unfilled center cue must remain.')
    # At distance one, a free three-square exposes twelve new cells: a capacity tie.
    tied = np.zeros((9, 9), dtype=np.int8)
    tied[1:8, 1:8] = 1
    tied[3:6, 3:6] = 2
    tied[4, 4] = 9
    _, _, contact, _ = _fields(tied != 0, (3, 3), 3)
    tied[contact == 1] = 9
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError('Capacity tie silently resolved.')
    # Input-only solver supports a fully trapped home and zero collection.
    trapped = np.zeros((9, 10), dtype=np.int8)
    trapped[2:5, 2:5] = 2
    trapped[3, 3] = 9
    trapped[6, 7:9] = 1
    trapped[6, 8] = 9
    _require(np.array_equal(solve(trapped), trapped), 'Trapped zero case changed.')


def self_test():
    """Bounded semantic, mutation, count, determinism, and 42-seed variation checks."""
    _analytic_tests()
    puzzles, rows, structures = [], [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        features = validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair, f in zip(p[split], features[split]):
                rows.append(dict(f, split=split))
                a = pair['input']
                normalized = np.empty(a.shape, dtype=np.int8)
                for code, color in enumerate(f['palette']):
                    normalized[a == color] = code
                structures.add(_signature(normalized))
    fingerprint = lambda p: tuple(_signature(pair[key]) for s in ('train', 'test')
                                  for pair in p[s] for key in ('input', 'output'))
    _require(len({fingerprint(p) for p in puzzles}) == 42, 'Cross-seed repetition.')
    _require(len(structures) >= 150, 'Structural variation collapsed.')
    _require(len({(f['height'], f['width']) for f in rows}) >= 45, 'Canvas variation collapsed.')
    _require({f['k'] for f in rows} == {3, 5}, 'Footprint sizes missing.')
    for role in range(4):
        _require({f['palette'][role] for f in rows} == set(range(10)), 'Palette role coverage collapsed.')
    _require(len({f['selected'] for f in rows}) >= 15, 'Loads lack variation.')
    _require(len({f['max_contact'] for f in rows}) >= 25, 'Travel distances lack variation.')
    _require(any(f['disconnected'] for f in rows), 'Disconnected supply absent.')
    _require(any(f['selected'] == 0 for f in rows), 'Zero supply outcome absent.')
    _require(any(f['catchment'] == f['floor_cells'] for f in rows), 'Fully sweepable fields absent.')
    _require(any(f['max_contact'] >= 40 for f in rows), 'Long wide detours absent.')
    # Every legal explicit count is exercised in the host; cover boundaries here.
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(171, nt, ne)
        _require(len(p['train']) == nt and len(p['test']) == ne, 'Explicit counts ignored.')
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted.')
    _require(fingerprint(puzzles[0]) == fingerprint(generate_puzzle(0)), 'Seed is not deterministic.')
    for mutation in ('wall', 'prefix', 'blocked'):
        bad = {s: [{key: value.copy() for key, value in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
        pair = next(pair for pair in bad['train']
                    if _analyze(pair['input'])[2]['selected'] in (2, 4))
        a, b = pair['input'], pair['output']
        roles, _, _, fields = _analyze(a)
        if mutation == 'wall':
            b[0, 0] = roles['floor']
        elif mutation == 'prefix':
            b[roles['pos']] = roles['casing']
        else:
            candidates = (a == roles['payload']) & ~roles['home'] & (fields[2] == _INF)
            b[tuple(np.argwhere(candidates)[0])] = roles['floor']
        try:
            validate_generated_puzzle(bad)
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError('Validator accepted '+mutation+' corruption.')
    return dict(seeds=42, pairs=len(rows), normalized_structures=len(structures),
                shapes=len({(f['height'], f['width']) for f in rows}),
                sizes=sorted({f['k'] for f in rows}),
                selected_counts=sorted({f['selected'] for f in rows}),
                max_contact=max(f['max_contact'] for f in rows),
                point_reversals=sum(f['point_reversal'] for f in rows),
                connected_blocked_pairs=sum(f['connected_blocked'] > 0 for f in rows),
                disconnected_pairs=sum(f['disconnected'] > 0 for f in rows),
                zero_pairs=sum(f['selected'] == 0 for f in rows),
                selected_distribution=dict(sorted(Counter(f['selected'] for f in rows).items())),
                verification='Generator-owned checks and analytic fixtures; no independent solver.')
