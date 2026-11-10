"""Blueprint-oriented collectors competing for payload in a shared maze.

Only NumPy and the standard library are used.  No parent module is imported.
The small blueprint transports a coordinate order, rather than an output crop.
"""
from collections import deque
from functools import lru_cache
import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 617003817, 447295166, 358496301, 821420266, 117380927,
              729076809, 575153443, 895483347, 161482177, 829146709,
              973383261, 721942480, 174972838, 388680808, 105363397,
              320608476, 583605355, 205132101, 811393768, 503780780,
              388241301, 274498933, 184335440, 574600140, 220154493,
              313167568, 497560129, 765198571, 88756401, 590983705,
              539035177, 867116627, 991518766, 564826861, 29319089,
              886316823, 324048839, 341840903, 530313642, 29963337,
              962467526)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid colors must be integers 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    h, w = mask.shape
    unseen = set(map(int, np.flatnonzero(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells = [start]
        for p in cells:
            r, c = divmod(p, w)
            for dr, dc in STEPS:
                rr, cc = r + dr, c + dc
                q = rr * w + cc
                if 0 <= rr < h and 0 <= cc < w and q in unseen:
                    unseen.remove(q)
                    cells.append(q)
        result.append(tuple(sorted(divmod(p, w) for p in cells)))
    return result


def _normal(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _size(cells):
    return max(r for r, c in cells) + 1, max(c for r, c in cells) + 1


def _bounds(cells):
    return min(r for r, c in cells), min(c for r, c in cells), \
        max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _mapped(cells, q, mirror):
    """Coordinates after D4, in the original cells' order (not re-sorted)."""
    transformed = [(r, -c if mirror else c) for r, c in cells]
    for _ in range(q):
        transformed = [(c, -r) for r, c in transformed]
    r0, c0 = min(r for r, c in transformed), min(c for r, c in transformed)
    return tuple((r-r0, c-c0) for r, c in transformed)


@lru_cache(maxsize=4096)
def _variants(shape):
    return tuple(sorted({tuple(sorted(_mapped(shape, q, m)))
                         for m in (False, True) for q in range(4)}))


def _canonical(shape):
    return _variants(_normal(shape))[0]


def _decode(cells, scale):
    r0, c0, r1, c1 = _bounds(cells)
    if (r1-r0) % scale or (c1-c0) % scale or len(cells) % (scale*scale):
        return None
    blocks = {}
    for r, c in cells:
        p = ((r-r0)//scale, (c-c0)//scale)
        blocks[p] = blocks.get(p, 0) + 1
    if any(n != scale*scale for n in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _interpret(a):
    """Recover roles, blueprint, scale, matching and transported slot orders."""
    colors = set(map(int, np.unique(a))) - {0}
    _require(len(colors) == 3, "need background 0 and three distinct local colors")
    components = {c: _components(a == c) for c in sorted(colors)}
    solutions = []
    for structure in sorted(colors):
        for comp in components[structure]:
            r0, c0, r1, c1 = _bounds(comp)
            if min(r1-r0, c1-c0) < 5:
                continue
            board = a[r0:r1, c0:c1]
            if not (np.all(board[0] == structure) and np.all(board[-1] == structure)
                    and np.all(board[:, 0] == structure) and np.all(board[:, -1] == structure)
                    and np.all((board == structure) | (board == 0))):
                continue
            holes = _components(board == 0)
            if not 2 <= len(holes) <= 4:
                continue
            shapes = [_normal(h) for h in holes]
            if not all(4 <= len(s) <= 9 and len(_variants(s)) == 8 for s in shapes):
                continue
            classes = {_canonical(s): s for s in shapes}
            if len(classes) != len(shapes):
                continue
            board_mask = np.zeros(a.shape, bool)
            board_mask[r0:r1, c0:c1] = True
            # The reference board is isolated, including from maze wall strokes.
            moat = np.zeros(a.shape, bool)
            moat[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
            if np.any(moat & ~board_mask & (a != 0)):
                continue
            for slot in sorted(colors - {structure}):
                pieces = components[slot]
                if len(pieces) != len(holes):
                    continue
                payload = next(iter(colors - {structure, slot}))
                if np.any((a == slot) & board_mask) or np.any((a == payload) & board_mask):
                    continue
                for k in (2, 3):
                    decoded = [_decode(p, k) for p in pieces]
                    if any(s is None for s in decoded):
                        continue
                    if {_canonical(s) for s in decoded} != set(classes):
                        continue
                    collectors = []
                    for cells, shape in zip(pieces, decoded):
                        reference = classes[_canonical(shape)]
                        matches = [(q, m, _mapped(reference, q, m))
                                   for m in (False, True) for q in range(4)
                                   if tuple(sorted(_mapped(reference, q, m))) == shape]
                        if len(matches) != 1:
                            break
                        q, mirror, order = matches[0]
                        pr, pc, _, _ = _bounds(cells)
                        mask = np.zeros(a.shape, bool)
                        for r, c in cells:
                            mask[r, c] = True
                        collectors.append(dict(cells=cells, shape=shape, reference=reference,
                                               order=order, origin=(pr, pc), mask=mask,
                                               capacity=len(shape), rotation=q, mirror=mirror))
                    else:
                        solutions.append(dict(structure=structure, slot=slot, payload=payload,
                                              scale=k, bounds=(r0, c0, r1, c1), board=board_mask,
                                              collectors=collectors, holes=shapes))
    _require(len(solutions) == 1, "expected one blueprint/scale/matching interpretation")
    return solutions[0]


def _distances(a, layout):
    """All footprints are endpoints; routes cannot pass through a collector."""
    ordinary = ((a == 0) | (a == layout['payload'])) & ~layout['board']
    h, w = a.shape
    result = []
    for collector in layout['collectors']:
        distance = np.full(a.shape, -1, dtype=np.int16)
        queue = deque(collector['cells'])
        distance[collector['mask']] = 0
        while queue:
            r, c = queue.popleft()
            for dr, dc in STEPS:
                rr, cc = r+dr, c+dc
                if (0 <= rr < h and 0 <= cc < w and ordinary[rr, cc]
                        and distance[rr, cc] < 0):
                    distance[rr, cc] = distance[r, c] + 1
                    queue.append((rr, cc))
        result.append(distance)
    return np.array(result)


def _ownership(distances):
    distances = np.where(distances < 0, 30000, distances)
    best = distances.min(axis=0)
    owner = distances.argmin(axis=0).astype(np.int8)
    owner[(best == 30000) | ((distances == best).sum(axis=0) != 1)] = -1
    return owner, best


def _analyze(a):
    layout = _interpret(a)
    distances = _distances(a, layout)
    owner, nearest = _ownership(distances)
    payload = a == layout['payload']
    tied = payload & (owner < 0) & (nearest < 30000)
    _require(not tied.any(), "a payload cell has tied nearest collectors")
    chosen = np.zeros(a.shape, bool)
    loads, supplies = [], []
    for i, collector in enumerate(layout['collectors']):
        candidates = payload & (owner == i)
        ds = np.sort(distances[i][candidates])
        n = min(collector['capacity'], len(ds))
        if len(ds) > n:
            _require(ds[n-1] < ds[n], "tied distance across capacity frontier")
        selected = candidates & (distances[i] <= ds[n-1]) if n else np.zeros(a.shape, bool)
        chosen |= selected
        supplies.append(int(candidates.sum()))
        loads.append(n)
    return layout, distances, owner, chosen, loads, supplies


def _paint(a, layout, selected, loads, *, native_order=False):
    out = a.copy()
    out[selected] = 0
    k = layout['scale']
    for collector, count in zip(layout['collectors'], loads):
        r0, c0 = collector['origin']
        order = sorted(collector['shape']) if native_order else collector['order']
        for r, c in order[:count]:
            out[r0+k*r:r0+k*(r+1), c0+k*c:c0+k*(c+1)] = layout['payload']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using the input alone; every convention is fixed or geometric."""
    a = _grid(input_grid)
    layout, _, _, selected, loads, _ = _analyze(a)
    return _paint(a, layout, selected, loads)


def _manhattan(shape, cells):
    rr, cc = np.indices(shape)
    result = np.full(shape, 30000, dtype=np.int16)
    for r, c in cells:
        result = np.minimum(result, abs(rr-r) + abs(cc-c))
    return result


def pair_features(input_grid):
    a = _grid(input_grid)
    layout, ds, owner, selected, loads, supplies = _analyze(a)
    payload = a == layout['payload']
    native = _paint(a, layout, selected, loads, native_order=True)
    correct = _paint(a, layout, selected, loads)
    reversal, competition, split_cluster, no_reroute = False, False, False, False
    reflected_order, rotated_order = False, False
    for i, collector in enumerate(layout['collectors']):
        picked = selected & (owner == i)
        retained = payload & (owner == i) & ~selected
        geometric = _manhattan(a.shape, collector['cells'])
        if picked.any() and retained.any():
            reversal |= int(geometric[picked].max()) > int(geometric[retained].min())
        for j, other in enumerate(layout['collectors']):
            if j != i and loads[j] < other['capacity']:
                no_reroute |= bool((retained & (ds[j] >= 0)).any())
        different_order = set(collector['order'][:loads[i]]) != set(sorted(collector['shape'])[:loads[i]])
        reflected_order |= collector['mirror'] and different_order
        rotated_order |= not collector['mirror'] and collector['rotation'] != 0 and different_order
        # A shared-reachability witness with at least one partial collector:
        # collecting independently would give it a larger prefix.
        reachable = payload & (ds[i] >= 0)
        competition |= loads[i] < collector['capacity'] and int(reachable.sum()) > supplies[i]
    for cells in _components(payload):
        split_cluster |= any(selected[p] for p in cells) and any(not selected[p] for p in cells)
    return dict(shape=tuple(a.shape), scale=layout['scale'], count=len(loads),
                capacities=tuple(c['capacity'] for c in layout['collectors']),
                loads=tuple(loads), supplies=tuple(supplies),
                excess=sum(max(0, s-c['capacity']) for s, c in zip(supplies, layout['collectors'])),
                unreachable=int((payload & np.all(ds < 0, axis=0)).sum()),
                selected=int(selected.sum()), wall_cells=int(((a == layout['structure']) & ~layout['board']).sum()),
                order_witness=not np.array_equal(native, correct),
                reflected_partial=any(c['mirror'] and 0 < n < c['capacity']
                                      for c, n in zip(layout['collectors'], loads)),
                rotated_partial=any(not c['mirror'] and c['rotation'] and 0 < n < c['capacity']
                                    for c, n in zip(layout['collectors'], loads)),
                path_reversal=bool(reversal), competition=bool(competition), split_cluster=bool(split_cluster),
                no_reroute_witness=bool(no_reroute), reflected_order_witness=bool(reflected_order),
                rotated_order_witness=bool(rotated_order),
                max_distance=int(ds.max()),
                classes=tuple(sorted(_canonical(s) for s in layout['holes'])),
                board_bounds=layout['bounds'],
                equal_area=len(set(len(s) for s in layout['holes'])) < len(layout['holes']),
                palette=(layout['structure'], layout['slot'], layout['payload']))


def _grow(rng, area, span):
    cells = {(0, 0)}
    while len(cells) < area:
        candidates = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in STEPS} - cells)
        candidates = [p for p in candidates if max(_size(_normal(cells | {p}))) <= span]
        if not candidates:
            return None
        cells.add(candidates[int(rng.integers(len(candidates)))])
    return _normal(cells)


def _shapes(rng, count, k):
    result, used = [], set()
    shared_area = int(rng.integers(5, 8))
    for i in range(count):
        area = shared_area if i < 2 else int(rng.integers(4, 10))
        for _ in range(250):
            s = _grow(rng, area, 3 if k == 3 else 4)
            if s is None or len(_variants(s)) != 8 or _canonical(s) in used:
                continue
            result.append(s)
            used.add(_canonical(s))
            break
        else:
            return None
    return result


def _blueprint(rng, shapes, structure):
    for _ in range(45):
        h, w = int(rng.integers(6, 13)), int(rng.integers(6, 14))
        board = np.full((h, w), structure, dtype=np.int8)
        occupied = set()
        for s in shapes:
            sh, sw = _size(s)
            origins = [(r, c) for r in range(1, h-sh) for c in range(1, w-sw)]
            for j in rng.permutation(len(origins)):
                r0, c0 = origins[int(j)]
                cells = {(r0+r, c0+c) for r, c in s}
                if cells & occupied or any((r+dr, c+dc) in occupied
                                           for r, c in cells for dr, dc in STEPS):
                    continue
                occupied |= cells
                for r, c in cells:
                    board[r, c] = 0
                break
            else:
                break
        else:
            return board
    return None


def _pack(rng, board, pieces, k, slot):
    for _ in range(45):
        h = int(rng.integers(24 if k == 3 else 22, 31))
        w = int(rng.integers(24 if k == 3 else 22, 31))
        a = np.zeros((h, w), dtype=np.int8)
        reserved = np.zeros(a.shape, bool)
        objects = [(board.shape, None)] + [((k*_size(s)[0], k*_size(s)[1]), s) for s in pieces]
        origins = [None] * len(objects)
        order = sorted(range(len(objects)), key=lambda i: -np.prod(objects[i][0]))
        for i in order:
            (oh, ow), s = objects[i]
            candidates = [(r, c) for r in range(h-oh+1) for c in range(w-ow+1)
                          if not reserved[r:r+oh, c:c+ow].any()]
            if not candidates:
                break
            r0, c0 = candidates[int(rng.integers(len(candidates)))]
            origins[i] = (r0, c0)
            if s is None:
                a[r0:r0+oh, c0:c0+ow] = board
            else:
                for r, c in s:
                    a[r0+k*r:r0+k*(r+1), c0+k*c:c0+k*(c+1)] = slot
            reserved[max(0, r0-1):r0+oh+1, max(0, c0-1):c0+ow+1] = True
        else:
            return a, reserved, origins
    return None


def _walls(rng, a, reserved, structure, kind):
    h, w = a.shape
    walls = np.zeros(a.shape, bool)
    if kind != 'open':
        for _ in range(int(rng.integers(4, 13))):
            if rng.integers(2):
                r, c0, c1 = int(rng.integers(h)), *sorted(map(int, rng.integers(0, w, 2)))
                walls[r, c0:c1+1] = True
            else:
                c, r0, r1 = int(rng.integers(w)), *sorted(map(int, rng.integers(0, h, 2)))
                walls[r0:r1+1, c] = True
        if kind == 'partitions':
            # Long partitions with one or two openings create route reversals.
            for _ in range(int(rng.integers(1, 4))):
                if rng.integers(2):
                    r = int(rng.integers(2, h-2))
                    walls[r, :] = True
                    c = int(rng.integers(w))
                    walls[r, c:min(w, c+int(rng.integers(1, 4)))] = False
                else:
                    c = int(rng.integers(2, w-2))
                    walls[:, c] = True
                    r = int(rng.integers(h))
                    walls[r:min(h, r+int(rng.integers(1, 4))), c] = False
        walls[reserved] = False
    # One independently placed sealed pocket, when requested; no decorated
    # fixed corner or compulsory number of unreachable tokens.
    if kind == 'pockets' or (kind == 'partitions' and rng.random() < .7):
        ph, pw = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        positions = [(r, c) for r in range(h-ph+1) for c in range(w-pw+1)
                     if not reserved[r:r+ph, c:c+pw].any()]
        if positions:
            r, c = positions[int(rng.integers(len(positions)))]
            walls[r:r+ph, c] = walls[r:r+ph, c+pw-1] = True
            walls[r, c:c+pw] = walls[r+ph-1, c:c+pw] = True
            walls[r+1:r+ph-1, c+1:c+pw-1] = False
    a = a.copy()
    a[walls] = structure
    return a


def _choose(rng, mask, n):
    pool = np.flatnonzero(mask)
    if len(pool) < n:
        return None
    return list(map(int, rng.choice(pool, n, replace=False)))


def _supply(rng, a, layout, mode):
    ds = _distances(a, layout)
    owner, _ = _ownership(ds)
    floor = (a == 0) & ~layout['board']
    chosen = []
    target = int(rng.integers(len(layout['collectors'])))
    for i, collector in enumerate(layout['collectors']):
        available = floor & (owner == i)
        cap = collector['capacity']
        saturated = mode == 'excess' and i == target
        if mode == 'random':
            saturated = rng.random() < .35
        if saturated:
            extra = int(rng.integers(1, 7))
            geometric = _manhattan(a.shape, collector['cells'])
            # Choose a strict distance shell, optionally forcing a reversal of
            # geometric proximity. No tie ordering is used by generation.
            frontiers = np.unique(ds[i][available])
            rng.shuffle(frontiers)
            for threshold in frontiers:
                near = available & (ds[i] <= threshold)
                far = available & (ds[i] > threshold)
                if int(near.sum()) < cap or int(far.sum()) < extra:
                    continue
                forced_near, forced_far = [], []
                if mode == 'excess':
                    best = np.flatnonzero(near & (geometric == geometric[near].max()))
                    anchor = int(rng.choice(best))
                    farther_path = far & (geometric < geometric.flat[anchor])
                    if not farther_path.any():
                        continue
                    forced_near = [anchor]
                    forced_far = [int(rng.choice(np.flatnonzero(farther_path)))]
                near.flat[forced_near] = False
                far.flat[forced_far] = False
                picked = _choose(rng, near, cap-len(forced_near))
                retained = _choose(rng, far, extra-len(forced_far))
                if picked is not None and retained is not None:
                    chosen += forced_near + picked + forced_far + retained
                    break
            else:
                return None
        else:
            n = int(rng.integers(2, cap)) if mode != 'random' else int(rng.integers(cap+1))
            picked = _choose(rng, available, n)
            if picked is None:
                return None
            chosen += picked
    unreachable = floor & np.all(ds < 0, axis=0)
    if mode == 'excess' and not unreachable.any():
        return None
    n = int(rng.integers(1 if mode == 'excess' else 0, min(8, int(unreachable.sum()))+1))
    chosen += _choose(rng, unreachable, n)
    if not chosen:
        return None  # Payload role must remain visible, even for all-zero loads.
    result = a.copy()
    result.flat[chosen] = layout['payload']
    return result


def _make_pair(rng, k, count, mode, structure, slot, payload, orientation):
    for _ in range(220):
        shapes = _shapes(rng, count, k)
        if shapes is None:
            continue
        board = _blueprint(rng, shapes, structure)
        if board is None:
            continue
        pieces = []
        for i, shape in enumerate(shapes):
            mirror = (orientation == 'reflection') if i == 0 else bool(rng.integers(2))
            q = int(rng.integers(1, 4)) if i == 0 and orientation == 'rotation' else int(rng.integers(4))
            pieces.append(tuple(sorted(_mapped(shape, q, mirror))))
        packed = _pack(rng, board, pieces, k, slot)
        if packed is None:
            continue
        base, reserved, origins = packed
        for _ in range(16):
            kind = 'partitions' if mode == 'excess' else str(rng.choice(['open', 'segments', 'pockets', 'partitions']))
            a = _walls(rng, base, reserved, structure, kind)
            # Geometry interpretation expects the payload color to be present;
            # construct its layout without a fake visible marker.
            collectors = []
            for piece, (r0, c0) in zip(pieces, origins[1:]):
                cells = tuple((r0+k*r+dr, c0+k*c+dc) for r, c in piece
                              for dr in range(k) for dc in range(k))
                mask = np.zeros(a.shape, bool)
                for p in cells:
                    mask[p] = True
                collectors.append(dict(cells=cells, capacity=len(piece), mask=mask))
            board_mask = np.zeros(a.shape, bool)
            r0, c0 = origins[0]
            board_mask[r0:r0+board.shape[0], c0:c0+board.shape[1]] = True
            layout = dict(collectors=collectors, payload=payload, board=board_mask)
            a = _supply(rng, a, layout, mode)
            if a is None:
                continue
            try:
                out = solve(a)
                f = pair_features(a)
            except ValueError:
                continue
            if mode != 'random':
                if not (f['competition'] and
                        f['reflected_order_witness' if orientation == 'reflection' else 'rotated_order_witness']):
                    continue
                if mode == 'excess' and not (f['path_reversal'] and f['unreachable'] and
                                            f['excess'] and f['no_reroute_witness']):
                    continue
            return dict(input=a, output=out)
    raise RuntimeError('bounded geometry construction exhausted')


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train, 2, 10, 'num_train')
    ne = _integer(num_test, 1, 4, 'num_test')
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    # A visible scale shift OR a new collector-count combination, never a
    # different rule or an invisible palette convention.
    shift = int(rng.integers(2))
    train_k = 2 if shift == 0 else int(rng.choice([2, 3]))
    specs = [(train_k, 2, 'partial', 'reflection'),
             (train_k, 2, 'excess', 'rotation')]
    for _ in range(nt-2):
        k = train_k
        count = int(rng.integers(2, 4)) if k == 2 and shift == 0 else 2
        specs.append((k, count, 'random', 'any'))
    specs = [specs[int(j)] for j in rng.permutation(len(specs))]
    tests = [(3 if shift == 0 else 2, 2 if shift == 0 else 3,
              str(rng.choice(['partial', 'excess', 'random'])),
              str(rng.choice(['reflection', 'rotation']))) for _ in range(ne)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, requests in (('train', specs), ('test', tests)):
        for k, count, mode, orientation in requests:
            for _ in range(8):
                structure, slot, payload = map(int, rng.choice(np.arange(1, 10), 3, replace=False))
                pair = _make_pair(rng, k, count, mode, structure, slot, payload, orientation)
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'invalid split counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape and _key(a) not in seen, 'dimensions or distinctness')
            seen.add(_key(a))
            layout, ds, owner, selected, loads, supplies = _analyze(a)
            expected = _paint(a, layout, selected, loads)
            _require(np.array_equal(expected, b), 'incorrect output')
            # Audit the actual change sets, not just a generated count label.
            footprints = np.logical_or.reduce([c['mask'] for c in layout['collectors']])
            _require(np.array_equal((a != b) & ~footprints, selected), 'only selected sources may be erased')
            _require(np.array_equal(a[layout['board']], b[layout['board']]), 'blueprint must remain intact')
            for c, n, supply in zip(layout['collectors'], loads, supplies):
                _require(n == min(c['capacity'], supply), 'wrong logical capacity')
                _require(int(((b == layout['payload']) & c['mask']).sum()) == n*layout['scale']**2,
                         'one collected cell must fill one whole macro slot')
            f = pair_features(a)
            _require(f['equal_area'], 'include equal-area noncongruent matching alternatives')
            features[split].append(f)
    train = features['train']
    _require(any(f['reflected_order_witness'] for f in train), 'reflection/order evidence missing')
    _require(any(f['rotated_order_witness'] for f in train), 'rotation/order evidence missing')
    _require(any(f['excess'] and f['path_reversal'] and f['unreachable'] and f['no_reroute_witness'] for f in train),
             'distance/capacity/reachability evidence missing')
    _require(any(f['competition'] for f in train), 'shared-field competition evidence missing')
    _require(any(0 < n < c for f in train for n, c in zip(f['loads'], f['capacities'])),
             'partial prefix evidence missing')


def self_test():
    """Bounded 42-seed variation audit plus targeted negative diagnostics."""
    all_features, signatures = [], set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        signatures.add(tuple((_key(pair['input']), _key(pair['output']))
                             for split in ('train', 'test') for pair in p[split]))
        for split in ('train', 'test'):
            for pair in p[split]:
                before = pair['input'].copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input'])
                f = pair_features(before)
                f['split'] = split
                all_features.append(f)
    assert len(signatures) == 42
    assert {f['scale'] for f in all_features} == {2, 3}
    assert {f['count'] for f in all_features} >= {2, 3}
    assert len({f['shape'] for f in all_features}) >= 25
    assert len({s for f in all_features for s in f['classes']}) >= 25
    assert len({f['board_bounds'] for f in all_features}) >= 50
    assert len({f['palette'] for f in all_features}) >= 30
    assert len({n for f in all_features for n in f['loads']}) >= 7
    assert max(f['max_distance'] for f in all_features) >= 25
    assert any(f['wall_cells'] == 0 for f in all_features)
    for split in ('train', 'test'):
        fs = [f for f in all_features if f['split'] == split]
        for name in ('order_witness', 'competition', 'excess', 'unreachable', 'path_reversal'):
            assert any(f[name] for f in fs), (split, name)
    a = generate_puzzle(123, 2, 1)
    b = generate_puzzle(123, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(100+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A coordinate transport diagnostic with an explicit, non-native prefix.
    shape = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2))
    assert _mapped(shape, 1, False) == ((0, 2), (0, 1), (0, 0), (1, 0), (2, 0))
    assert _mapped(shape, 0, True)[:2] == ((0, 2), (1, 2))
    # Tied ownership is excluded, rather than silently choosing collector 0.
    ds = np.array([[[1, 2, -1]], [[1, 3, -1]]])
    assert _ownership(ds)[0].tolist() == [[-1, 0, -1]]
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('incorrect label accepted')
    return {'seeds': 42, 'pairs': len(all_features),
            'shapes': len({s for f in all_features for s in f['classes']}),
            'canvas_sizes': len({f['shape'] for f in all_features})}
