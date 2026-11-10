"""Sliding catchment counters: rigid registration changes a shared flow graph.

Only NumPy and the standard library are used. Generation and self-test do not
write files. At most one puzzle and bounded sets of diagnostic hashes are kept.
"""
from collections import deque
import hashlib
import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORE = frozenset((r, c) for r in range(-1, 2) for c in range(-1, 2))
HOST_SEEDS = (0, 452741331, 943495059, 199057198, 173344075, 871407933,
              109728341, 323555320, 954923439, 244907746, 502334435,
              351908265, 276409336, 259920042, 169825498, 854831173,
              209583641, 701547719, 88847527, 708263232, 358652595,
              653165214, 503872439, 572742223, 886064313, 994350969,
              647954854, 60853782, 592318986, 343651015, 329799594,
              296532961, 14334362, 516032185, 748265220, 335684099,
              694489817, 686759899, 328738880, 747291174, 895643011, 62599595)


def _require(value, message):
    if not bool(value):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             'grid must contain integer colors 0..9')
    return a.astype(np.int8, copy=False)


def _components(mask):
    h, w = mask.shape
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, queue = {start}, [start]
        for r, c in queue:
            for dr, dc in DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    queue.append(p)
        result.append(cells)
    return result


def _dilate(points, h, w):
    return {(r + dr, c + dc) for r, c in points for dr, dc in ((0, 0),) + DIRS
            if 0 <= r + dr < h and 0 <= c + dc < w}


def _decode(value):
    """Recover palette, complete carriers and projections using only the input."""
    a = _grid(value)
    h, w = a.shape
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(counts.max() > a.size / 2, 'background must occupy a strict majority')
    # Payload never occupies the boundary. Only isolated edge cells that match
    # a distinct center in an eight-cell uniform 3x3 rim become anchors.
    singles = {}
    for cells in _components(a != bg):
        if len(cells) == 1:
            r, c = next(iter(cells))
            if (r in (0, h - 1)) != (c in (0, w - 1)):
                singles.setdefault(int(a[r, c]), []).append((int(r), int(c)))
    motifs = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            marker = int(a[r, c])
            if marker not in singles:
                continue
            rim = [int(a[r + dr, c + dc]) for dr, dc in CORE if (dr, dc) != (0, 0)]
            if len(set(rim)) == 1 and rim[0] not in (bg, marker):
                motifs.append((r, c, marker, rim[0]))
    _require(1 <= len(motifs) <= 3, 'need one to three marked counters')
    markers = {m[2] for m in motifs}
    bodies = {m[3] for m in motifs}
    _require(len(markers) == len(motifs) and len(bodies) == 1, 'nonunique counter identities')
    body = next(iter(bodies))
    _require(body not in markers and bg not in markers, 'overlapping color roles')
    payloads = set(map(int, colors)) - {bg, body} - markers
    _require(len(payloads) == 1, 'one geometrically recoverable payload color required')
    payload = next(iter(payloads))
    _require(not np.any(a[[0, h - 1], :] == payload)
             and not np.any(a[:, [0, w - 1]] == payload), 'payload must be off the boundary')
    objects = _components((a == body) | np.isin(a, sorted(markers)))
    movers, all_anchors = [], set()
    for r, c, marker, _ in sorted(motifs, key=lambda m: m[2]):
        anchors = set(singles[marker])
        rows, cols = {x for x, _ in anchors}, {y for _, y in anchors}
        horizontal = len(rows) == 1 and all(y in (0, w - 1) for _, y in anchors)
        vertical = len(cols) == 1 and all(x in (0, h - 1) for x, _ in anchors)
        _require(horizontal != vertical, 'anchors do not define one guide')
        axis = 0 if horizontal else 1
        guide = next(iter(rows if horizontal else cols))
        component = next(x for x in objects if (r, c) in x)
        _require(not component & anchors, 'carrier must be separate from anchors')
        _require(sum(int(a[p]) in markers for p in component) == 1,
                 'carrier contains multiple registrations')
        _require(np.count_nonzero(a == marker) == 1 + len(anchors),
                 'unexplained marker occurrence')
        core = {(r + dr, c + dc) for dr, dc in CORE}
        _require(core <= component and len(component - core) >= 2, 'counter needs wall arms')
        delta = (guide - r, 0) if axis == 0 else (0, guide - c)
        target = {(int(x + delta[0]), int(y + delta[1])) for x, y in component}
        _require(all(0 < x < h - 1 and 0 < y < w - 1 for x, y in target),
                 'complete counter and arms must remain inside boundary')
        movers.append(dict(marker=marker, center=(r, c), core=core, source=set(component),
                           target=target, delta=delta, axis=axis, guide=int(guide), anchors=anchors,
                           destination=(r + delta[0], c + delta[1])))
        all_anchors |= anchors
    source = set().union(*(m['source'] for m in movers))
    static = set(map(tuple, np.argwhere(a == body))) - source
    occupied = static | all_anchors
    for m in movers:
        _require(not m['target'] & occupied, 'projected carriers overlap')
        _require(all(a[p] != payload for p in m['target']), 'projection would cover payload')
        occupied |= m['target']
    _require(not any(a[p] == body for p in all_anchors), 'invalid anchor')
    return dict(grid=a, bg=bg, body=body, payload=payload, movers=movers,
                static=static, anchors=all_anchors)


def _settled(scene, mode='final'):
    """Build a simultaneous settled grid, optionally a diagnostic counterfactual."""
    a = scene['grid']
    result = a.copy()
    if mode == 'original':
        return result, [m['center'] for m in scene['movers']]
    for m in scene['movers']:
        for p in m['source']:
            result[p] = scene['bg']
    for m in scene['movers']:
        for r, c in m['source']:
            p = (r + m['delta'][0], c + m['delta'][1])
            if mode != 'no_arms' or (r, c) in m['core']:
                result[p] = a[r, c]
    return result, [m['destination'] for m in scene['movers']]


def _distances(blocked, center):
    distance = np.full(blocked.shape, -1, dtype=np.int16)
    queue = deque()
    for dr, dc in CORE:
        p = (center[0] + dr, center[1] + dc)
        distance[p] = 0
        queue.append(p)
    h, w = blocked.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = distance[r, c] + 1
                queue.append((nr, nc))
    return distance


def _field(scene, mode='final'):
    settled, centers = _settled(scene, mode)
    blocked = settled == scene['body']
    for r, c in centers:
        blocked[r - 1:r + 2, c - 1:c + 2] = False
    for p in scene['anchors']:
        blocked[p] = True
    distances = np.stack([_distances(blocked, p) for p in centers])
    positive = np.where(distances < 0, 30000, distances)
    minimum = positive.min(axis=0)
    owner = positive.argmin(axis=0).astype(np.int8)
    owner[minimum == 30000] = -1
    owner[(minimum < 30000) & ((positive == minimum).sum(axis=0) > 1)] = -2
    return settled, centers, distances, owner


def _allocation(scene, mode='final', strict=True):
    settled, centers, distances, owner = _field(scene, mode)
    payload = scene['grid'] == scene['payload']
    if strict:
        _require(not np.any(payload & (owner == -2)), 'payload has tied nearest counters')
    selected = []
    for i in range(len(centers)):
        candidates = np.flatnonzero(payload & (owner == i))
        order = sorted(map(int, candidates), key=lambda p: (int(distances[i].flat[p]), p))
        if strict and len(order) > 9:
            _require(distances[i].flat[order[8]] < distances[i].flat[order[9]],
                     'capacity frontier has a distance tie')
        selected.append(set(order[:9]))
    return settled, centers, distances, owner, selected


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _decode(input_grid)
    result, centers, _, _, selected = _allocation(scene)
    for (r, c), cells in zip(centers, selected):
        for cell in cells:
            result.flat[cell] = scene['bg']
        for j in range(len(cells)):
            result[r - 1 + j // 3, c - 1 + j % 3] = scene['payload']
    return result


def _arm_shape(rng, long=False):
    cells = set(CORE)
    sides = list(map(int, rng.choice(4, size=int(rng.integers(1, 4)), replace=False)))
    for side in sides:
        dr, dc = DIRS[side]
        lateral = int(rng.integers(-1, 2))
        r, c = dr + dc * lateral, dc + dr * lateral
        length = int(rng.integers(2, 7 if long else 5))
        for _ in range(length):
            r, c = r + dr, c + dc
            cells.add((r, c))
        if rng.random() < .75:
            turn = int(rng.choice([-1, 1]))
            for _ in range(int(rng.integers(1, 5 if long else 4))):
                r, c = r + dc * turn, c + dr * turn
                cells.add((r, c))
    return cells


def _layout(rng, k, mixed, test, palette):
    bg, body, payload = palette[:3]
    low = 15 if k == 1 else 20 if k == 2 else 24
    h, w = map(int, rng.integers(low, 31 if test else 29, size=2))
    inp = np.full((h, w), bg, dtype=np.int8)
    source, target, anchors, records = set(), set(), set(), []
    axes = [0, 1] if mixed else list(map(int, rng.integers(2, size=k)))
    if mixed and k > 2:
        axes += list(map(int, rng.integers(2, size=k - 2)))
    rng.shuffle(axes)
    for i in range(k):
        marker = palette[3 + i]
        placed = False
        for _ in range(90):
            offsets = _arm_shape(rng, long=test or rng.random() < .2)
            minr, maxr = min(r for r, _ in offsets), max(r for r, _ in offsets)
            minc, maxc = min(c for _, c in offsets), max(c for _, c in offsets)
            if h - maxr - 1 <= 1 - minr or w - maxc - 1 <= 1 - minc:
                continue
            r = int(rng.integers(1 - minr, h - maxr - 1))
            c = int(rng.integers(1 - minc, w - maxc - 1))
            axis = axes[i]
            guide = int(rng.integers(1 - (minr if axis == 0 else minc),
                                     (h - maxr - 1) if axis == 0 else (w - maxc - 1)))
            if abs(guide - (r if axis == 0 else c)) < 3:
                continue
            dr, dc = (guide - r, 0) if axis == 0 else (0, guide - c)
            src = {(r + x, c + y) for x, y in offsets}
            dst = {(r + dr + x, c + dc + y) for x, y in offsets}
            ends = (0, 1) if rng.random() < .3 else (int(rng.integers(2)),)
            anc = ({(guide, 0 if e == 0 else w - 1) for e in ends} if axis == 0 else
                   {(0 if e == 0 else h - 1, guide) for e in ends})
            if src & _dilate(source | anchors | anc, h, w) or dst & (target | anchors | anc):
                continue
            if anc & _dilate(source | anchors, h, w):
                continue
            records.append((r, c, marker, src, dst, anc))
            source |= src
            target |= dst
            anchors |= anc
            placed = True
            break
        if not placed:
            return None
    # Static barriers combine long cuts with openings, elbows, and closed pockets.
    forbidden = _dilate(source | anchors, h, w) | target
    wall = np.zeros((h, w), dtype=bool)
    for _ in range(int(rng.integers(1, 6))):
        horizontal = bool(rng.integers(2))
        line = int(rng.integers(2, (h if horizontal else w) - 2))
        start = int(rng.integers(0, (w if horizontal else h) // 2))
        end = int(rng.integers(max(start + 3, (w if horizontal else h) // 2), w if horizontal else h))
        if horizontal:
            wall[line, start:end + 1] = True
        else:
            wall[start:end + 1, line] = True
        if rng.random() < .6:
            gap = int(rng.integers(start, end + 1))
            if horizontal:
                wall[line, gap:gap + int(rng.integers(1, 4))] = False
            else:
                wall[gap:gap + int(rng.integers(1, 4)), line] = False
        if rng.random() < .4:
            if horizontal:
                wall[max(0, line - int(rng.integers(2, 7))):line + 1, end] = True
            else:
                wall[end, max(0, line - int(rng.integers(2, 7))):line + 1] = True
    for p in forbidden:
        wall[p] = False
    # One closed pocket makes unreachable supply possible without a fixed location.
    pockets = []
    for _ in range(100):
        rh, cw = int(rng.integers(4, 8)), int(rng.integers(4, 8))
        r, c = int(rng.integers(1, h - rh)), int(rng.integers(1, w - cw))
        box = {(x, y) for x in range(r, r + rh) for y in range(c, c + cw)}
        if box & forbidden:
            continue
        ring = {p for p in box if p[0] in (r, r + rh - 1) or p[1] in (c, c + cw - 1)}
        for p in box:
            wall[p] = p in ring
        pockets.append(box - ring)
        if len(pockets) >= int(rng.integers(1, 3)):
            break
    if not pockets:
        return None
    inp[wall] = body
    for r, c, marker, src, _, anc in records:
        for p in src:
            inp[p] = body
        inp[r, c] = marker
        for p in anc:
            inp[p] = marker
    return inp, records, pockets


def _sample(rng, cells, n, forced=()):
    chosen = set(map(int, forced))
    pool = np.array(sorted(set(map(int, cells)) - chosen), dtype=int)
    if len(chosen) > n or len(pool) < n - len(chosen):
        return None
    chosen.update(map(int, rng.choice(pool, size=n - len(chosen), replace=False)))
    return chosen


def _supplies(rng, scene, loads, demanding):
    a = scene['grid']
    _, centers, distances, owner = _field(scene)
    allowed = a == scene['bg']
    allowed[[0, -1], :] = False
    allowed[:, [0, -1]] = False
    for p in _dilate(scene['anchors'], *a.shape):
        allowed[p] = False
    for m in scene['movers']:
        for p in m['target']:
            allowed[p] = False
    _, _, _, free_owner = _field(scene, 'no_arms')
    rr, cc = np.indices(a.shape)
    chosen = set()
    for i, load in enumerate(loads):
        pool = allowed & (owner == i)
        if int(pool.sum()) < load:
            return None
        wing_witness = np.flatnonzero(pool & (free_owner != owner) & (free_owner >= 0))
        if load <= 9:
            force = [int(rng.choice(wing_witness))] if demanding and len(wing_witness) else []
            picked = _sample(rng, np.flatnonzero(pool), load, force)
        else:
            r, c = centers[i]
            geometric = np.maximum(np.abs(rr - r) - 1, 0) + np.maximum(np.abs(cc - c) - 1, 0)
            picked = None
            levels = list(map(int, np.unique(distances[i][pool])))
            rng.shuffle(levels)
            for threshold in levels:
                near = np.flatnonzero(pool & (distances[i] <= threshold))
                far = np.flatnonzero(pool & (distances[i] > threshold))
                if len(near) < 9 or len(far) < load - 9:
                    continue
                force_near, force_far = [], []
                if demanding:
                    # The near/far reversal directly distinguishes route distance.
                    p = int(near[np.argmax(geometric.flat[near])])
                    q = int(far[np.argmin(geometric.flat[far])])
                    if geometric.flat[p] <= geometric.flat[q]:
                        continue
                    force_near.append(p)
                    force_far.append(q)
                    # Adjacent selected cells expose counting cells rather than
                    # counting monochromatic connected objects.
                    near_set = set(map(int, near))
                    adjacent = next(((p, p + step) for p in sorted(near_set)
                                     for step in (1, a.shape[1])
                                     if p + step in near_set
                                     and (step != 1 or p // a.shape[1] == (p + 1) // a.shape[1])), None)
                    if adjacent is None:
                        continue
                    force_near.extend(adjacent)
                witnesses = [int(p) for p in wing_witness if distances[i].flat[p] <= threshold]
                if demanding and witnesses:
                    force_near.append(int(rng.choice(witnesses)))
                left = _sample(rng, near, 9, force_near)
                right = _sample(rng, far, load - 9, force_far)
                if left is not None and right is not None:
                    picked = left | right
                    break
        if picked is None:
            return None
        chosen |= picked
    inactive = np.flatnonzero(allowed & (owner == -1))
    if not len(inactive):
        return None
    chosen |= _sample(rng, inactive, int(rng.integers(1, min(7, len(inactive)) + 1)))
    return chosen


def _signature(a):
    return a.shape, a.tobytes()


def pair_features(pair):
    """All measurements are reconstructed from returned grids, never plans."""
    s = _decode(pair['input'])
    _, centers, d, owner, selected = _allocation(s)
    _, _, _, original_owner = _field(s, 'original')
    _, _, _, free_owner, free_selection = _allocation(s, 'no_arms', strict=False)
    supply = s['grid'] == s['payload']
    rr, cc = np.indices(s['grid'].shape)
    reversals, loads, overflow_to_free = [], [], False
    for i, cells in enumerate(selected):
        candidates = supply & (owner == i)
        loads.append(int(candidates.sum()))
        retained = set(map(int, np.flatnonzero(candidates))) - cells
        r, c = centers[i]
        geo = np.maximum(np.abs(rr - r) - 1, 0) + np.maximum(np.abs(cc - c) - 1, 0)
        reversals.append(bool(cells and retained and max(geo.flat[p] for p in cells)
                              > min(geo.flat[p] for p in retained)))
        for j in range(len(centers)):
            if j != i and len(selected[j]) < 9 and any(d[j].flat[p] >= 0 for p in retained):
                overflow_to_free = True
    active_payload = supply & (owner >= 0)
    selected_adjacent = any(any((p + 1 in cells and p // s['grid'].shape[1]
                                == (p + 1) // s['grid'].shape[1])
                               or p + s['grid'].shape[1] in cells for p in cells)
                            for cells in selected)
    return dict(shape=list(s['grid'].shape), counters=len(centers),
                axes=sorted({m['axis'] for m in s['movers']}),
                sides=sorted({('top' if r == 0 else 'bottom' if r == s['grid'].shape[0] - 1
                               else 'left' if c == 0 else 'right') for r, c in s['anchors']}),
                loads=loads, selected=[len(x) for x in selected],
                selected_adjacent=selected_adjacent,
                unreachable=int((supply & (owner == -1)).sum()),
                path_reversal=any(reversals), no_rerouting_witness=overflow_to_free,
                shared_supply=int((supply & ((d >= 0).sum(axis=0) >= 2)).sum()),
                alignment_changes_ownership=bool(np.any(active_payload & (owner != original_owner)
                                                       & (original_owner != -2))),
                arms_change_collection=selected != free_selection,
                arms_change_ownership=bool(np.any(active_payload & (owner != free_owner)
                                                  & (free_owner >= 0))),
                arm_sizes=[len(m['source']) - 9 for m in s['movers']],
                shifts=[list(m['delta']) for m in s['movers']],
                max_distance=int(d.max()), static_walls=len(s['static']),
                palette=[s['bg'], s['body'], s['payload']] + [m['marker'] for m in s['movers']])


def _make_pair(rng, palette, kind, test):
    # The first two requests teach partial serialization and competitive overflow.
    demanding = kind == 'competitive'
    k = 2 if demanding else 1 if kind == 'partial' else 3 if test else int(rng.integers(1, 3))
    for _ in range(180):
        layout = _layout(rng, k, demanding or (k > 1 and rng.random() < .65), test, palette)
        if layout is None:
            continue
        a, records, pockets = layout
        # Temporarily place one known payload in a closed pocket so role recovery
        # is the same input-only decoder used by solve, even before supply sampling.
        a[next(iter(sorted(pockets[0])))] = palette[2]
        if np.mean(a == palette[0]) <= .5:
            continue
        scene = _decode(a)
        scene['grid'] = a.copy()
        scene['grid'][scene['grid'] == palette[2]] = palette[0]
        if demanding:
            loads = [int(rng.integers(10, 18)), int(rng.integers(5, 9))]
            rng.shuffle(loads)
        elif kind == 'partial':
            loads = [int(rng.choice([2, 4]))]
        else:
            loads = [int(rng.integers(0, 18)) for _ in range(k)]
            if not any(loads):
                loads[int(rng.integers(k))] = 1
        supply = _supplies(rng, scene, loads, demanding)
        if supply is None:
            continue
        a = scene['grid'].copy()
        a.flat[list(supply)] = palette[2]
        if np.mean(a == palette[0]) <= .5:
            continue
        pair = dict(input=a, output=solve(a))
        f = pair_features(pair)
        if demanding and not (f['path_reversal'] and f['no_rerouting_witness']
                              and f['arms_change_collection'] and f['arms_change_ownership']
                              and f['alignment_changes_ownership'] and f['shared_supply']
                              and f['selected_adjacent']):
            continue
        return pair
    raise RuntimeError('bounded construction could not satisfy catchment evidence')


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if not _integer(value) or not low <= value <= high:
            raise ValueError('illegal example count')
    rng = np.random.default_rng(seed)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    palette = list(map(int, rng.permutation(10)))
    recolor = bool(rng.random() < .65)
    plans = ['partial', 'competitive'] + ['random'] * (num_train - 2)
    rng.shuffle(plans)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, kinds in (('train', plans), ('test', ['random'] * num_test)):
        for kind in kinds:
            for _ in range(12):
                local = list(map(int, rng.permutation(10))) if recolor else palette
                pair = _make_pair(rng, local, kind, split == 'test')
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded uniqueness construction failed')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'canvas dimensions changed')
            sig = _signature(inp)
            _require(sig not in seen, 'duplicate input')
            seen.add(sig)
            scene = _decode(inp)
            expected = solve(inp)
            _require(np.array_equal(out, expected), 'incorrect movement, allocation or prefix')
            # Explicit conservation, stationary evidence and rigid arm checks.
            for p in scene['anchors'] | scene['static']:
                _require(out[p] == inp[p], 'static wall or anchor changed')
            _require(np.count_nonzero(inp == scene['payload']) == np.count_nonzero(out == scene['payload']),
                     'payload cell count not conserved')
            for m in scene['movers']:
                for r, c in m['source'] - m['core']:
                    _require(out[r + m['delta'][0], c + m['delta'][1]] == scene['body'],
                             'wall arm did not move rigidly')
            f = pair_features(pair)
            _require(f['unreachable'] > 0, 'missing preserved unreachable supply')
            _require(all(x != [0, 0] for x in f['shifts']), 'generated counter must move')
            features[split].append(f)
    train = features['train']
    _require(any(2 in f['selected'] or 4 in f['selected'] for f in train), 'no partial prefix witness')
    _require(any(f['counters'] >= 2 and f['path_reversal'] and f['no_rerouting_witness']
                 and f['arms_change_collection'] and f['arms_change_ownership']
                 and f['alignment_changes_ownership'] and f['shared_supply']
                 and f['selected_adjacent']
                 and max(f['loads']) > 9 and any(5 <= n < 9 for n in f['selected']) for f in train),
             'missing integrated competitive overflow demonstration')


def self_test():
    fingerprints, structures, shapes = set(), set(), set()
    axes, sides, counts, selected, arm_sizes, distances = set(), set(), set(), set(), set(), set()
    varying, competition, topology = 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        local_shapes = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                f = pair_features(pair)
                shapes.add(tuple(a.shape))
                local_shapes.add(tuple(a.shape))
                axes.update(f['axes'])
                sides.update(f['sides'])
                counts.add(f['counters'])
                selected.update(f['selected'])
                arm_sizes.update(f['arm_sizes'])
                distances.add(f['max_distance'])
                competition += f['shared_supply'] > 0
                topology += f['arms_change_collection']
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in a.flat)
                structures.add((a.shape, hashlib.sha256(normalized).digest()))
                digest.update(str(a.shape).encode())
                for value in pair.values():
                    digest.update(value.tobytes())
        varying += len(local_shapes) > 1
        fingerprints.add(digest.digest())
    _require(len(fingerprints) == 42 and len(structures) >= 190, '42-seed structural diversity')
    _require(len(shapes) >= 50 and varying == 42, 'canvas variation')
    _require(axes == {0, 1} and sides == {'top', 'bottom', 'left', 'right'}, 'guide coverage')
    _require(counts == {1, 2, 3} and set(range(10)) <= selected, 'counter/load coverage')
    _require(len(arm_sizes) >= 15 and len(distances) >= 20, 'wall and route diversity')
    _require(competition >= 80 and topology >= 42, 'interaction missing')
    p = generate_puzzle(713, 2, 1)
    q = generate_puzzle(713, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x),
             'nondeterministic generation')
    # Recoloring is a family symmetry; rotation preserves movement and allocation,
    # while the output prefix must be rebuilt in displayed row-major order.
    permutation = np.random.default_rng(18).permutation(10)
    for pair in p['train'] + p['test']:
        original = _decode(pair['input'])
        settled, _, _, _, original_cells = _allocation(original)
        removed = np.zeros(pair['input'].shape, dtype=bool)
        for chosen in original_cells:
            removed.flat[list(chosen)] = True
        _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
                 'palette dependence')
        for turns in (1, 2, 3):
            inp = np.rot90(pair['input'], turns)
            sc = _decode(inp)
            centers = [m['destination'] for m in sc['movers']]
            expected = np.rot90(settled, turns).copy()
            expected[np.rot90(removed, turns)] = sc['bg']
            for (r, c), chosen in zip(centers, original_cells):
                for index in range(len(chosen)):
                    expected[r - 1 + index // 3, c - 1 + index % 3] = sc['payload']
            result = solve(inp)
            _require(np.array_equal(result, expected), 'rotated geometry/allocation/prefix')
    # Analytical case: a 12-cell connected line belongs to the left counter.
    # Exactly its nearest nine disappear. Three reachable excess cells stay,
    # despite spare capacity on the right; a sealed pocket also retains a cell.
    line = np.zeros((20, 26), dtype=np.int8)
    for r, c, marker, anchor in ((5, 5, 1, (5, 0)), (5, 19, 2, (5, 25))):
        line[r - 1:r + 2, c - 1:c + 2] = 5
        line[r, c] = marker
        line[anchor] = marker
    line[2:4, 4] = 5
    line[2:4, 20] = 5
    line[11, 18:23] = line[15, 18:23] = 5
    line[11:16, 18] = line[11:16, 22] = 5
    line[7:19, 5] = 9
    line[5:7, 17] = 9
    line[13, 20] = 9
    expected = line.copy()
    expected[7:16, 5] = 0
    expected[5:7, 17] = 0
    expected[4:7, 4:7] = 9
    expected[4, 18:20] = 9
    _require(np.array_equal(solve(line), expected), 'cellwise capacity/cluster/overflow regression')
    # Nearest-source ties are invalid, including at an otherwise empty counter.
    tied = line.copy()
    tied[tied == 9] = 0
    tied[5, 12] = 9
    try:
        solve(tied)
    except AssertionError as error:
        _require('tied nearest' in str(error), 'wrong nearest-tie rejection')
    else:
        raise AssertionError('nearest-counter tie accepted')
    # Ten equal-distance cells cannot be resolved by row order at capacity.
    capacity = line.copy()
    capacity[capacity == 9] = 0
    scene = _decode(line)
    right = next(m for m in scene['movers'] if m['marker'] == 2)
    for cell in right['source'] | right['anchors']:
        capacity[cell] = 0
    positions = [(3, c) for c in (5, 6)] + [(7, c) for c in (4, 5, 6)]
    positions += [(r, c) for r in (4, 5, 6) for c in (3, 7)]
    for cell in positions[:10]:
        capacity[cell] = 9
    try:
        solve(capacity)
    except AssertionError as error:
        _require('frontier' in str(error), 'wrong capacity-tie rejection')
    else:
        raise AssertionError('capacity tie accepted')
    for nt, ne in ((10, 4), (2, 1), (5, 4)):
        extra = generate_puzzle(148, nt, ne)
        _require(len(extra['train']) == nt, 'train count ignored')
        _require(len(extra['test']) == ne, 'test count ignored')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(4, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, structures=len(structures), shapes=len(shapes),
                counter_counts=sorted(counts), selected_counts=sorted(selected),
                arm_sizes=len(arm_sizes), route_lengths=len(distances),
                competitive_pairs=competition, arm_sensitive_pairs=topology)
