"""Decode a stretched circuit, then propagate panel votes and elect its center.

The construction records write labels; the input-only parser checks them.
No parent module, file, stored example, or random state is used by solve().
"""

import hashlib
import heapq
import itertools

import numpy as np


DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))
HOST_SEEDS = (0, 105664788, 379998616, 54361924, 719612698, 796906735,
              426132965, 310267803, 593509221, 23309148, 230154082,
              550740406, 394300139, 432783737, 887861415, 351511893,
              58881375, 275159234, 614171908, 564496593, 640160365,
              93756296, 924076403, 679081235, 233375459, 343164559,
              673552441, 495102167, 721215379, 679008515, 982985330,
              785909930, 673834169, 78632045, 561073492, 938163705,
              559329239, 177601956, 453101580, 644683062, 702776922,
              590134351)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _starts(a):
    rows = np.r_[0, np.flatnonzero(np.any(a[1:] != a[:-1], axis=1)) + 1]
    cols = np.r_[0, np.flatnonzero(np.any(a[:, 1:] != a[:, :-1], axis=0)) + 1]
    return rows, cols


def _decode(a):
    """Crop only the exterior mat, preserving the internal circuit background."""
    mat = int(a[0, 0])
    assert a[0, -1] == a[-1, 0] == a[-1, -1] == mat
    rr, cc = np.nonzero(a != mat)
    assert len(rr)
    top, bottom, left, right = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
    crop = a[top:bottom, left:right]
    assert np.all(crop != mat) and np.count_nonzero(a != mat) == crop.size
    rows, cols = _starts(crop)
    logical = crop[np.ix_(rows, cols)].copy()
    rh = np.diff(np.r_[rows, crop.shape[0]])
    cw = np.diff(np.r_[cols, crop.shape[1]])
    assert np.array_equal(np.repeat(np.repeat(logical, rh, axis=0), cw, axis=1), crop)
    assert len(np.unique(logical)) == 4 and mat not in logical
    return logical, rh, cw, (top, a.shape[0] - bottom, left, a.shape[1] - right)


def _components(mask):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        idx = len(result)
        labels[r, c] = idx
        stack, cells = [(r, c)], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in DIRS:
                q = p[0] + dr, p[1] + dc
                if 0 <= q[0] < h and 0 <= q[1] < w and mask[q] and labels[q] < 0:
                    labels[q] = idx
                    stack.append(q)
        result.append(cells)
    return result, labels


def _windows(mask):
    h, w = mask.shape
    if min(h, w) < 3:
        return []
    found = np.ones((h - 2, w - 2), dtype=bool)
    for dr, dc in itertools.product(range(3), repeat=2):
        found &= mask[dr:dr + h - 2, dc:dc + w - 2]
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(found))]


def _ports(hub):
    r, c = hub
    return ([(r - 1, c + j) for j in range(3)]
            + [(r + j, c + 3) for j in range(3)]
            + [(r + 3, c + j) for j in range(3)]
            + [(r + j, c - 1) for j in range(3)])


def _interpret(a, background, wire, signals, hub):
    h, w = a.shape
    r, c = hub
    assert 0 < r < h - 3 and 0 < c < w - 3
    panels, panel_ids = _components(np.isin(a, signals))
    assert 2 <= len(panels) <= 8
    majorities, boxes, areas = [], [], []
    for cells in panels:
        rr, cc = zip(*cells)
        box = min(rr), max(rr) + 1, min(cc), max(cc) + 1
        assert len(cells) == (box[1] - box[0]) * (box[3] - box[2]) >= 3
        counts = [sum(a[p] == s for p in cells) for s in signals]
        assert min(counts) > 0 and counts[0] != counts[1]
        majorities.append(signals[int(counts[1] > counts[0])])
        boxes.append(box)
        areas.append(len(cells))
    mask = a == wire
    mask[r:r + 3, c:c + 3] = False
    routes, route_ids = _components(mask)
    assert routes
    port_set = set(_ports(hub))
    owners, route_labels = [], []
    for cells in routes:
        adjacent = set()
        for x, y in cells:
            for dr, dc in DIRS:
                q = x + dr, y + dc
                if 0 <= q[0] < h and 0 <= q[1] < w and panel_ids[q] >= 0:
                    adjacent.add(int(panel_ids[q]))
        assert len(adjacent) == 1 and port_set.intersection(cells)
        owner = adjacent.pop()
        owners.append(owner)
        route_labels.append(majorities[owner])
    weights = [0] * len(panels)
    port_labels = []
    for p in _ports(hub):
        if a[p] == wire:
            rid = int(route_ids[p])
            assert rid >= 0
            owner = owners[rid]
            weights[owner] += 1
            port_labels.append((p, majorities[owner]))
        else:
            assert a[p] == background
    assert all(weights)
    counts = [sum(label == s for _, label in port_labels) for s in signals]
    assert min(counts) > 0 and counts[0] != counts[1]
    winner = signals[int(counts[1] > counts[0])]
    return dict(background=background, wire=wire, signals=signals, hub=hub,
                panels=panels, boxes=boxes, areas=areas, majorities=majorities,
                routes=routes, owners=owners, route_labels=route_labels,
                weights=weights, port_labels=port_labels, winner=winner,
                margin=abs(counts[0] - counts[1]))


def _inspect(a):
    """Jointly infer roles; solid signal-colored patches are allowed in panels."""
    values = tuple(map(int, np.unique(a)))
    assert len(values) == 4
    answers = []
    for wire in values:
        hubs = _windows(a == wire)
        if len(hubs) != 1:
            continue
        for background in values:
            if background == wire:
                continue
            signals = tuple(s for s in values if s not in (wire, background))
            try:
                answers.append(_interpret(a, background, wire, signals, hubs[0]))
            except AssertionError:
                pass
    assert len(answers) == 1, "circuit must have one geometric interpretation"
    return answers[0]


def _vote(a, info):
    result = a.copy()
    for cells, label in zip(info['routes'], info['route_labels']):
        for p in cells:
            result[p] = label
    r, c = info['hub']
    result[r + 1, c + 1] = info['winner']
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All conventions are visible geometry; this function has no latent state."""
    a = _grid(input_grid)
    try:
        logical, _, _, _ = _decode(a)
        return _vote(logical, _inspect(logical))
    except AssertionError as error:
        raise ValueError("input is not an unambiguous stretched voting circuit") from error


def _rect(r, c, h, w):
    return {(x, y) for x in range(r, r + h) for y in range(c, c + w)}


def _neighbors(cells, h, w):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in DIRS
            if 0 <= r + dr < h and 0 <= c + dc < w}


def _pattern(rng, h, w, majority, minority, close):
    n = h * w
    m = (n - 1) // 2 if close else int(rng.integers(1, (n - 1) // 2 + 1))
    kind = int(rng.integers(3))
    rr, cc = np.indices((h, w))
    if kind == 0:
        order = rng.permutation(n)
    elif kind == 1:
        x, y = int(rng.integers(h)), int(rng.integers(w))
        order = np.argsort((abs(rr - x) + abs(cc - y) + rng.random((h, w)) * .4).ravel())
    else:
        order = np.argsort((((rr + cc) % 2) * n + rng.random((h, w))).ravel())
    a = np.full(n, majority, dtype=np.int8)
    a[order[:m]] = minority
    return a.reshape(h, w)


def _path(rng, start, targets, allowed):
    h, w = allowed.shape
    if not allowed[start]:
        return None
    costs = 1 + rng.random((h, w)) * float(rng.choice((.1, 1., 4., 9.)))
    distance, prev = {start: 0.}, {}
    heap = [(0., *start)]
    while heap:
        d, r, c = heapq.heappop(heap)
        p = r, c
        if d != distance[p]:
            continue
        if p in targets:
            path = {p}
            while p != start:
                p = prev[p]
                path.add(p)
            return path
        for dr, dc in DIRS:
            q = r + dr, c + dc
            if 0 <= q[0] < h and 0 <= q[1] < w and allowed[q]:
                nd = d + costs[q]
                if nd < distance.get(q, float('inf')):
                    distance[q], prev[q] = nd, p
                    heapq.heappush(heap, (nd, *q))
    return None


def _topology(rng, count, witness):
    if witness:
        choices = {3: ((3, 1, 1), (4, 1, 1), (4, 2, 1)),
                   4: ((4, 1, 1, 1),),
                   5: ((3, 1, 1, 1, 1), (2, 2, 1, 1, 1))}
        weights = list(choices[count][int(rng.integers(len(choices[count])))])
        rng.shuffle(weights)
    else:
        total = int(rng.integers(max(3, count), 9))
        choices = [x for x in itertools.product(range(1, 5), repeat=count) if sum(x) == total]
        weights = list(choices[int(rng.integers(len(choices)))])
    choices = [bits for bits in range(1, (1 << count) - 1)
               if sum(weights[i] for i in range(count) if bits >> i & 1) > sum(weights) / 2
               and (not witness or bits.bit_count() < count / 2)]
    if not choices:
        return None
    return weights, int(rng.choice(choices))


def _layout(rng, h, w, hub, count, weights):
    """Build separated panel-owned networks, allowing forks and shared ports."""
    r, c = hub
    reserved = _rect(r - 2, c - 2, 7, 7)
    panels = []
    for _ in range(count):
        for _ in range(50):
            ph, pw = int(rng.integers(1, 6)), int(rng.integers(1, 7))
            if ph * pw < 3:
                continue
            side = int(rng.integers(4))
            inset = int(rng.choice((0, 0, 0, 1, 2)))
            x = inset if side == 0 else h - ph - inset if side == 2 else int(rng.integers(h - ph + 1))
            y = inset if side == 3 else w - pw - inset if side == 1 else int(rng.integers(w - pw + 1))
            cells = _rect(x, y, ph, pw)
            if cells & reserved:
                continue
            panels.append((x, y, ph, pw, cells))
            reserved |= cells | _neighbors(cells, h, w)
            break
        else:
            return None
    angle = lambda p: float(np.arctan2(p[0] + (p[2] - 1) / 2 - r - 1,
                                       p[1] + (p[3] - 1) / 2 - c - 1))
    panels.sort(key=angle)
    ring = [(r - 1, c), (r - 1, c + 2), (r, c + 3), (r + 2, c + 3),
            (r + 3, c + 2), (r + 3, c), (r + 2, c - 1), (r, c - 1)]
    chosen = sorted(map(int, rng.choice(8, sum(weights), replace=False)))
    candidates = []
    for shift in range(len(chosen)):
        seq = chosen[shift:] + chosen[:shift]
        groups, start, cost = [], 0, 0.
        for panel, weight in zip(panels, weights):
            group = {ring[j] for j in seq[start:start + weight]}
            start += weight
            for x, y in group:
                delta = abs(angle(panel) - float(np.arctan2(x - r - 1, y - c - 1)))
                cost += min(delta, 2 * np.pi - delta) ** 2
            groups.append(group)
        candidates.append((cost + float(rng.random()), groups))
    groups = min(candidates, key=lambda t: t[0])[1]
    # Move a separated pair to a broad two-cell port, or a singleton to the middle.
    for side in range(4):
        ports = _ports(hub)[side * 3:side * 3 + 3]
        occupied = set.union(*groups).intersection(ports)
        if len(occupied) == 1 and rng.random() < .25:
            p = next(iter(occupied))
            group = next(g for g in groups if p in g)
            group.remove(p)
            group.add(ports[1])
        for group in groups:
            if ports[0] in group and ports[2] in group and rng.random() < .25:
                group.remove(ports[int(rng.choice((0, 2)))])
                group.add(ports[1])
    hub_cells = _rect(r, c, 3, 3)
    collar = _rect(r - 1, c - 1, 5, 5) - hub_cells
    all_panels = set.union(*(p[4] for p in panels))
    networks = [set(g) for g in groups]
    for owner in sorted(range(count), key=lambda i: -weights[i]):
        own = panels[owner][4]
        others = set.union(*(net for i, net in enumerate(networks) if i != owner))
        blocked = (hub_cells | collar | all_panels | others
                   | _neighbors((all_panels - own) | others, h, w)) - groups[owner]
        allowed = np.ones((h, w), dtype=bool)
        for p in blocked:
            allowed[p] = False
        border = {p for p in _neighbors(own, h, w) - own if allowed[p]}
        if not border:
            return None
        net = set()
        for port in sorted(groups[owner]):
            path = _path(rng, port, border | net, allowed)
            if path is None:
                return None
            net |= path
        if rng.random() < .4:
            border_mask = np.zeros((h, w), dtype=bool)
            for p in border:
                border_mask[p] = True
            for cells in _components(border_mask)[0]:
                if set(cells) & (net | _neighbors(net, h, w)) and rng.random() < .65:
                    net.update(cells)
        if net & collar != groups[owner]:
            return None
        networks[owner] = net
    return panels, networks


def _circuit(rng, roles, witness, close):
    """Construct logical geometry and an output directly from owned cell sets."""
    h, w = map(int, rng.integers(13, 23, size=2))
    hub = int(rng.integers(3, h - 5)), int(rng.integers(3, w - 5))
    count = int(rng.choice((3, 4, 5) if witness else (2, 3, 4, 5, 6)))
    topology = _topology(rng, count, witness)
    if topology is None:
        return None
    weights, bits = topology
    layout = _layout(rng, h, w, hub, count, weights)
    if layout is None:
        return None
    panels, networks = layout
    _, bg, wire, s0, s1 = roles
    winner, loser = (s0, s1) if rng.random() < .5 else (s1, s0)
    a = np.full((h, w), bg, dtype=np.int8)
    r, c = hub
    a[r:r + 3, c:c + 3] = wire
    labels = []
    for i, ((x, y, ph, pw, _), net) in enumerate(zip(panels, networks)):
        label = winner if bits >> i & 1 else loser
        labels.append(label)
        a[x:x + ph, y:y + pw] = _pattern(rng, ph, pw, label,
                                        loser if label == winner else winner, close)
        for p in net:
            a[p] = wire
    # This is construction normalization, before choosing the logical circuit.
    # Full vector duplicates outside the hub are dispensable geometric spacing.
    # The three true collector rows/columns must all survive, without exceptions.
    rows, cols = _starts(a)
    if not set(range(r, r + 3)) <= set(rows) or not set(range(c, c + 3)) <= set(cols):
        return None
    compact = a[np.ix_(rows, cols)].copy()
    if min(compact.shape) < 7 or max(compact.shape) > 22:
        return None
    expected = compact.copy()
    for (x, y, ph, pw, _), net, label in zip(panels, networks, labels):
        retained = [(i, j) for i, rr in enumerate(rows) for j, cc in enumerate(cols)
                    if x <= rr < x + ph and y <= cc < y + pw]
        values = [int(compact[p]) for p in retained]
        other = loser if label == winner else winner
        if not values or values.count(label) <= values.count(other) or other not in values:
            return None
        for i, rr in enumerate(rows):
            for j, cc in enumerate(cols):
                if (int(rr), int(cc)) in net:
                    expected[i, j] = label
    nr, nc = int(np.flatnonzero(rows == r)[0]), int(np.flatnonzero(cols == c)[0])
    expected[nr + 1, nc + 1] = winner
    try:
        info = _inspect(compact)
        assert info['winner'] == winner and np.array_equal(_vote(compact, info), expected)
        if witness:
            assert sum(s == winner for s in info['majorities']) < len(panels) / 2
            assert np.count_nonzero(compact == winner) < np.count_nonzero(compact == loser)
        return compact, expected, info
    except AssertionError:
        return None


def _profile(rng, length, test):
    bands = np.ones(length, dtype=int)
    if test:
        peak = int(rng.integers(length))
        bands[peak] = int(rng.integers(4, min(7, 29 - length) + 1))
        extra = int(rng.integers(0, min(5, 28 - int(bands.sum())) + 1))
    else:
        if length <= 13 and rng.random() < .08:
            return np.full(length, 2, dtype=int)
        extra = int(rng.integers(3, min(12, 28 - length) + 1))
    for _ in range(extra):
        choices = np.flatnonzero(bands < (7 if test else 3))
        if len(choices):
            bands[int(rng.choice(choices))] += 1
    return bands


def _distortions(a, info, rows, cols):
    flipped = 0
    weights = rows[:, None] * cols[None, :]
    for cells, majority in zip(info['panels'], info['majorities']):
        own = sum(int(weights[p]) for p in cells if a[p] == majority)
        other = sum(int(weights[p]) for p in cells if a[p] != majority)
        flipped += other > own
    r, c = info['hub']
    balance = 0
    for (x, y), label in info['port_labels']:
        width = int(cols[y] if x in (r - 1, r + 3) else rows[x])
        balance += width if label == info['winner'] else -width
    return flipped, balance < 0


def _expand(rng, a, expected, info, mat, test, witness):
    for _ in range(160 if witness else 1):
        rows = _profile(rng, a.shape[0], test)
        cols = _profile(rng, a.shape[1], test)
        if witness:
            if len(set(rows)) == 1 or len(set(cols)) == 1:
                continue
            if not _distortions(a, info, rows, cols)[0]:
                continue
        enlarged = np.repeat(np.repeat(a, rows, axis=0), cols, axis=1)
        eh = int(rng.integers(2, min(6, 30 - enlarged.shape[0]) + 1))
        ew = int(rng.integers(2, min(6, 30 - enlarged.shape[1]) + 1))
        top, left = int(rng.integers(1, eh)), int(rng.integers(1, ew))
        if rng.random() < .2:
            # One side may touch the boundary; all four corners still show mat.
            side = int(rng.integers(4))
            if side < 2:
                top = 0 if side == 0 else eh
            else:
                left = 0 if side == 2 else ew
        canvas = np.full((enlarged.shape[0] + eh, enlarged.shape[1] + ew), mat, dtype=np.int8)
        canvas[top:top + enlarged.shape[0], left:left + enlarged.shape[1]] = enlarged
        return {'input': canvas, 'output': expected.copy()}
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    shared = bool(rng.random() < .35)
    palette = list(map(int, rng.choice(10, 5, replace=False)))
    slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            witness_a = split == 'train' and index == slots[0]
            witness_b = split == 'train' and index == slots[1]
            for _ in range(1600):
                roles = palette if shared else list(map(int, rng.choice(10, 5, replace=False)))
                circuit = _circuit(rng, roles, witness_a, witness_b)
                if circuit is None:
                    continue
                pair = _expand(rng, *circuit, roles[0], split == 'test', witness_b)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded circuit construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _metrics(a, info, rows, cols):
    winner = info['winner']
    loser = next(s for s in info['signals'] if s != winner)
    flip, port_flip = _distortions(a, info, rows, cols)
    return dict(shape=list(a.shape), panel_count=len(info['panels']),
                port_count=len(info['port_labels']), weights=sorted(info['weights']),
                areas=info['areas'], margin=info['margin'],
                panel_count_wrong=2 * info['majorities'].count(winner) < len(info['panels']),
                raw_cells_wrong=int(np.count_nonzero(a == winner)) < int(np.count_nonzero(a == loser)),
                physical_panel_flips=int(flip), physical_port_flip=bool(port_flip),
                row_bands=list(map(int, rows)), column_bands=list(map(int, cols)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    tr, te = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, evidence = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a, rows, cols, margins = _decode(inp)
            info = _inspect(a)
            assert np.array_equal(out, _vote(a, info))
            assert inp.shape != out.shape
            assert len(rows) == out.shape[0] and len(cols) == out.shape[1]
            assert len(_starts(a)[0]) == a.shape[0] and len(_starts(a)[1]) == a.shape[1]
            assert sum(m == 0 for m in margins) <= 1
            panel_mask = np.isin(a, info['signals'])
            assert np.array_equal(out[panel_mask], a[panel_mask])
            assert np.array_equal(out[a == info['background']], a[a == info['background']])
            r, c = info['hub']
            border = out[r:r + 3, c:c + 3].copy()
            border[1, 1] = info['wire']
            assert np.all(border == info['wire'])
            assert all(out[p] == label for p, label in info['port_labels'])
            assert len(info['port_labels']) >= 3 and info['margin'] > 0
            if split == 'train':
                evidence.append(_metrics(a, info, rows, cols))
    assert any(m['panel_count_wrong'] and m['raw_cells_wrong'] for m in evidence)
    assert any(m['physical_panel_flips'] and len(set(m['row_bands'])) > 1
               and len(set(m['column_bands'])) > 1 for m in evidence)


def self_test():
    """42 streamed seeds, input-only invariants and metamorphic/mutation checks."""
    shapes, geometries, topologies, palettes, profiles, signatures = (set() for _ in range(6))
    flips, port_flips = 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(hashlib.sha256(b''.join(p['input'].tobytes()
                                              for s in puzzle for p in puzzle[s])).digest())
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                original = pair['input'].copy()
                assert np.array_equal(solve(original), pair['output'])
                assert np.array_equal(original, pair['input'])
                a, rows, cols, _ = _decode(original)
                info = _inspect(a)
                shapes.add(original.shape)
                geometries.add((a.shape, info['hub'], tuple(info['boxes'])))
                topologies.add(tuple(sorted(info['weights'])))
                palettes.add(tuple(sorted(map(int, np.unique(original)))))
                profiles.add((tuple(rows), tuple(cols)))
                local, ports = _distortions(a, info, rows, cols)
                flips += local > 0
                port_flips += ports
                if split == 'test':
                    assert max(rows) >= 4 and max(cols) >= 4
                else:
                    assert max(rows) <= 3 and max(cols) <= 3
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(geometries) >= 150
    assert len(topologies) >= 8 and len(palettes) >= 20 and len(profiles) >= 150
    assert flips >= 42 and port_flips > 0
    first, again = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    for s in first:
        for x, y in zip(first[s], again[s]):
            assert all(np.array_equal(x[k], y[k]) for k in x)
    pair = first['test'][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k))
    assert np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1])
    permutation = np.array([7, 4, 0, 8, 2, 9, 1, 6, 5, 3], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    a, _, _, _ = _decode(pair['input'])
    info = _inspect(a)
    # Re-encode with a completely different profile. Logical votes must persist.
    rows = 1 + (np.arange(a.shape[0]) % 2)
    cols = np.ones(a.shape[1], dtype=int)
    if rows.sum() <= 28:
        encoded = np.pad(np.repeat(np.repeat(a, rows, axis=0), cols, axis=1),
                         1, constant_values=int(pair['input'][0, 0]))
        assert np.array_equal(solve(encoded), pair['output'])
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    r, c = info['hub']
    broken['test'][0]['output'][r + 1, c + 1] = info['wire']
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a wrong collector center")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4., 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    for tr, te in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(314159, tr, te)
        assert len(p['train']) == tr and len(p['test']) == te


if __name__ == '__main__':
    self_test()
