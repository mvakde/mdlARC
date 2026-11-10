"""Puncture-created eight-slot collectors for loose wall-colored masonry.

Only NumPy and the standard library are used. solve() sees just one input.
Generation labels are painted from explicit construction masks; validation
reconstructs those masks from returned pixels before checking routing and slots.
"""
from collections import deque
import numpy as np

HIGHLIGHT = 9
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SLOTS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
         (1, -1), (1, 0), (1, 1))
HOST_SEEDS = (0, 652314504, 447640801, 989617770, 113781898,
              506979599, 722306966, 939874387, 22173081, 283809082,
              830721854, 239600467, 855671160, 151668972, 916604723,
              821204955, 754083105, 230941125, 703804556, 653878564,
              178473862, 918055094, 819226564, 22728837, 240841714,
              331506977, 901887979, 73060136, 73338691, 990993571,
              452009541, 567227752, 4652245, 729901701, 958579964,
              560310792, 353262773, 764074485, 985054087, 996040986,
              331467232, 963481064)


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)):
        raise ValueError('Counts must be integers.')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('Require 2..10 train, 1..4 test, and train > test.')
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError('Expected a 1..30 integer grid using colors 0..9.')
    return a.astype(np.int64, copy=True)


def _components(mask):
    """Four-connected sets, in displayed row-major order."""
    h, w = mask.shape
    remaining = set(map(int, np.flatnonzero(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group = [start]
        for cell in group:
            y, x = divmod(cell, w)
            for dy, dx in STEPS:
                yy, xx = y + dy, x + dx
                q = yy * w + xx
                if 0 <= yy < h and 0 <= xx < w and q in remaining:
                    remaining.remove(q)
                    group.append(q)
        groups.append(group)
    return groups


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = np.zeros_like(mask, dtype=bool)
    for y in range(3):
        for x in range(3):
            result |= p[y:y+h, x:x+w]
    return result


def _interior(mask):
    """Centers of intact 3x3 squares, without boundary wrapping."""
    result = np.zeros_like(mask, dtype=bool)
    if min(mask.shape) >= 3:
        result[1:-1, 1:-1] = np.logical_and.reduce(
            [mask[y:y+mask.shape[0]-2, x:x+mask.shape[1]-2]
             for y in range(3) for x in range(3)])
    return result


def _decode(value):
    a = _grid(value)
    field = int(a[0, 0])
    if not all(int(a[y, x]) == field for y, x in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError('All four corners must identify the field.')
    colors = set(map(int, np.unique(a)))
    if len(colors) != 2 or HIGHLIGHT in colors:
        raise ValueError('Input needs field and masonry colors, neither 9.')
    solid = next(iter(colors - {field}))
    wall = np.zeros(a.shape, bool)
    cargo = np.zeros(a.shape, bool)
    for group in _components(a == solid):
        if len(group) <= 4:
            cargo.flat[group] = True
        elif len(group) >= 16:
            wall.flat[group] = True
        else:
            raise ValueError('Ambiguous component between speck and wall sizes.')
    if not wall.any():
        raise ValueError('No supported wall scaffold.')
    # A puncture is a single field cell surrounded by eight intact wall cells.
    p = np.pad(wall, 1)
    surrounded = np.ones(a.shape, bool)
    for dy, dx in SLOTS:
        surrounded &= p[1+dy:1+dy+a.shape[0], 1+dx:1+dx+a.shape[1]]
    holes = surrounded & (a == field)
    centers = [tuple(map(int, q)) for q in np.argwhere(holes)]
    if not 1 <= len(centers) <= 8:
        raise ValueError('Require one to eight isolated punctures.')
    base = wall | holes
    if (cargo & _dilate(base)).any():
        raise ValueError('Loose masonry must be separated from the wall by field.')
    for group in _components(~base):
        if len(group) < 9:
            raise ValueError('A legitimate field region must be larger than damage.')
    return a, field, solid, base, holes, cargo, centers


def _distances(base, centers):
    """Path length from each footprint; all footprints are traversable ports."""
    h, w = base.shape
    ports = np.zeros_like(base)
    for y, x in centers:
        ports[y-1:y+2, x-1:x+2] = True
    allowed = ~base | ports
    results = []
    for y, x in centers:
        distance = np.full(base.shape, -1, dtype=np.int32)
        queue = deque()
        for yy in range(y-1, y+2):
            for xx in range(x-1, x+2):
                distance[yy, xx] = 0
                queue.append((yy, xx))
        while queue:
            yy, xx = queue.popleft()
            for dy, dx in STEPS:
                r, c = yy+dy, xx+dx
                if 0 <= r < h and 0 <= c < w and allowed[r, c] and distance[r, c] < 0:
                    distance[r, c] = distance[yy, xx] + 1
                    queue.append((r, c))
        results.append(distance)
    return np.stack(results)


def _direct_distances(shape, centers):
    yy, xx = np.indices(shape)
    return np.stack([np.maximum(np.abs(yy-y)-1, 0) + np.maximum(np.abs(xx-x)-1, 0)
                     for y, x in centers])


def _owners(distances):
    d = np.where(distances < 0, 10000, distances)
    best = d.min(axis=0)
    tied = (d == best).sum(axis=0) > 1
    owner = d.argmin(axis=0).astype(np.int32)
    owner[(best == 10000) | tied] = -1
    return owner, best, tied & (best < 10000)


def _collect(cargo, distances):
    owner, best, ties = _owners(distances)
    accepted = np.zeros_like(cargo)
    counts, assigned = [], []
    for i in range(len(distances)):
        mask = cargo & (owner == i)
        ordered = np.sort(best[mask])
        if len(ordered) > 8 and ordered[7] == ordered[8]:
            raise ValueError('A capacity frontier may not split equal distances.')
        n = min(8, len(ordered))
        if n:
            accepted |= mask & (best <= ordered[n-1])
        counts.append(n)
        assigned.append(len(ordered))
    return accepted, counts, assigned, owner, best, ties


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only reconstruction, geodesic ownership, collection and halos."""
    a, field, _, base, holes, cargo, centers = _decode(input_grid)
    accepted, counts, _, _, _, _ = _collect(cargo, _distances(base, centers))
    result = a.copy()
    result[accepted] = field
    for (y, x), n in zip(centers, counts):
        for dy, dx in SLOTS[:n]:
            result[y+dy, x+dx] = HIGHLIGHT
    result[holes] = field
    return result


def _paint(base, centers, cargo, field, solid):
    """Label construction uses the supplied scaffold, never calls solve()."""
    inp = np.where(base, solid, field).astype(np.int64)
    for p in centers:
        inp[p] = field
    inp[cargo] = solid
    accepted, counts, _, _, _, _ = _collect(cargo, _distances(base, centers))
    out = inp.copy()
    out[accepted] = field
    for (y, x), count in zip(centers, counts):
        # Deliberately explicit construction traversal of the displayed ring.
        slot = 0
        for yy in range(y-1, y+2):
            for xx in range(x-1, x+2):
                if (yy, xx) == (y, x):
                    continue
                if slot < count:
                    out[yy, xx] = HIGHLIGHT
                slot += 1
    for p in centers:
        out[p] = field
    return {'input': inp, 'output': out}


def _geometric_output(input_grid):
    """Audit-only alternative: ignore obstacles; break capacity ties by position.

    This is a deliberately simpler competing rule, not an independent solver.
    It retains the real receiver and cargo interpretation but uses Manhattan
    distance. The production rule never uses its capacity tie convention.
    """
    a, field, _, _, holes, cargo, centers = _decode(input_grid)
    owner, best, _ = _owners(_direct_distances(a.shape, centers))
    accepted = np.zeros(a.shape, bool)
    counts = []
    for i in range(len(centers)):
        cells = list(map(int, np.flatnonzero(cargo & (owner == i))))
        cells.sort(key=lambda p: (int(best.flat[p]), p))
        accepted.flat[cells[:8]] = True
        counts.append(min(8, len(cells)))
    out = a.copy()
    out[accepted] = field
    for (y, x), n in zip(centers, counts):
        for dy, dx in SLOTS[:n]:
            out[y+dy, x+dx] = HIGHLIGHT
    out[holes] = field
    return out


def _ring(base, y, x, hh, ww, thickness=3):
    base[y:y+hh, x:x+ww] = True
    base[y+thickness:y+hh-thickness, x+thickness:x+ww-thickness] = False


def _geometry(rng, kind):
    if kind == 'bars':
        h, w = int(rng.integers(23, 31)), int(rng.integers(16, 31))
        base = np.zeros((h, w), bool)
        first = int(rng.integers(5, 8))
        second = int(rng.integers(first+8, h-7))
        base[first:first+3] = True
        base[second:second+3] = True
        # The teaching case keeps every port in just one of the two barriers.
        row = (first if rng.random() < .5 else second) + 1
        available = np.zeros_like(base)
        available[row, 1:-1] = True
    elif kind == 'detour':
        h, w = int(rng.integers(24, 31)), int(rng.integers(23, 31))
        base = np.zeros((h, w), bool)
        mid = int(rng.integers(9, w-9))
        end = int(rng.integers(h-6, h-3))
        base[:end, mid:mid+3] = True
        lo = int(rng.integers(h-14, h-11))
        hi = int(rng.integers(2, 5))
        base[lo:h-2, 2:5] = True
        base[hi:hi+10, w-5:w-2] = True
        available = _interior(base)
        available[:, mid:mid+3] = False
    else:
        h, w = int(rng.integers(17, 31)), int(rng.integers(17, 31))
        base = np.zeros((h, w), bool)
        if kind == 'ring':
            y, x = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            hh, ww = int(rng.integers(11, h-y+1)), int(rng.integers(11, w-x+1))
            _ring(base, y, x, hh, ww)
        elif kind == 'islands':
            for _ in range(int(rng.integers(2, 5))):
                hh, ww = int(rng.integers(3, 8)), int(rng.integers(3, 9))
                y, x = int(rng.integers(1, h-hh)), int(rng.integers(1, w-ww))
                base[y:y+hh, x:x+ww] = True
                if rng.random() < .5:
                    base[y:y+3, x:min(w-1, x+ww+int(rng.integers(2, 7)))] = True
        elif kind in ('rooms', 'mixed'):
            midx = int(rng.integers(5, w-7))
            midy = int(rng.integers(5, h-7))
            base[:, midx:midx+3] = True
            base[midy:midy+3] = True
            # Optional wide doors and rail extensions compose the partition.
            if rng.random() < .65:
                y = int(rng.integers(1, h-6))
                base[y:y+5, midx:midx+3] = False
            if kind == 'mixed':
                for _ in range(int(rng.integers(1, 4))):
                    y, x = int(rng.integers(1, h-5)), int(rng.integers(1, w-5))
                    base[y:y+3, x:min(w-1, x+int(rng.integers(5, 12)))] = True
                    if rng.random() < .6:
                        base[y:min(h-1, y+int(rng.integers(5, 12))), x:x+3] = True
        else:
            raise ValueError('Unknown geometry kind.')
        available = _interior(base)
    # Rotate scaffolds before serialization; displayed slot order never rotates.
    k = int(rng.integers(4))
    base, available = np.rot90(base, k), np.rot90(available, k)
    if rng.random() < .5:
        base, available = np.fliplr(base), np.fliplr(available)
    return base.copy(), available.copy()


def _choose_centers(rng, available, wanted, overlap):
    pool = [tuple(map(int, q)) for q in np.argwhere(available)]
    if not pool:
        return None
    rng.shuffle(pool)
    chosen = []
    if overlap:
        candidates = [(a, b) for a in pool for b in pool
                      if a < b and max(abs(a[0]-b[0]), abs(a[1]-b[1])) == 2]
        if not candidates:
            return None
        chosen.extend(candidates[int(rng.integers(len(candidates)))])
    for p in pool:
        if len(chosen) >= wanted:
            break
        if all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 2 for q in chosen):
            chosen.append(p)
    return sorted(chosen) if len(chosen) == wanted else None


def _small_patches(mask):
    return all(len(g) <= 4 for g in _components(mask))


def _pick(rng, pool, number, forced=()):
    forced = list(dict.fromkeys(map(int, forced)))
    remaining = np.array([int(p) for p in np.flatnonzero(pool) if int(p) not in forced], int)
    if len(forced) > number or number > len(forced)+len(remaining):
        return None
    return forced + list(map(int, rng.choice(remaining, size=number-len(forced), replace=False)))


def _supply(rng, base, centers, profile):
    eligible = ~_dilate(base)
    eligible[0, 0] = eligible[0, -1] = eligible[-1, 0] = eligible[-1, -1] = False
    distances = _distances(base, centers)
    owner, best, ties = _owners(distances)
    direct_owner, _, _ = _owners(_direct_distances(base.shape, centers))
    witness = eligible & (owner >= 0) & (direct_owner >= 0) & (owner != direct_owner)
    unreachable = eligible & (best == 10000)
    tied = eligible & ties
    pools = [eligible & (owner == i) for i in range(len(centers))]
    if profile == 'partial' and (not unreachable.any() or not tied.any()
                                 or sum(p.sum() >= 4 for p in pools) < 2):
        return None
    if profile == 'saturated' and not witness.any():
        return None
    for _ in range(90):
        forced_witness = int(rng.choice(np.flatnonzero(witness))) if profile == 'saturated' else None
        amounts = [int(rng.integers(0, 9 if profile == 'free' else 6)) for _ in centers]
        active = [i for i, p in enumerate(pools) if p.sum() >= 4]
        if not active:
            return None
        saturated = None
        if profile == 'partial':
            a, b = map(int, rng.choice(active, 2, replace=False))
            amounts = [0]*len(centers)
            # Four slots exposes middle-left as slot four, distinguishing
            # row-major ring order from clockwise as well as column-major.
            amounts[a], amounts[b] = 4, int(rng.integers(1, 5))
        elif profile == 'saturated':
            options = [i for i, p in enumerate(pools) if p.sum() >= 12
                       and len(np.unique(best[p])) >= 3]
            if not options:
                return None
            saturated = int(rng.choice(options))
            amounts[saturated] = int(rng.integers(9, 14))
            wi = int(owner.flat[forced_witness])
            amounts[wi] = max(1, amounts[wi])
        elif rng.random() < .5:
            options = [i for i, p in enumerate(pools) if p.sum() >= 12]
            if options:
                saturated = int(rng.choice(options))
                amounts[saturated] = int(rng.integers(9, 14))
        if not any(amounts):
            amounts[int(rng.choice(active))] = int(rng.integers(1, 5))
        cargo = np.zeros_like(base)
        okay = True
        for i, number in enumerate(amounts):
            number = min(number, int(pools[i].sum()))
            forced = [forced_witness] if forced_witness is not None and owner.flat[forced_witness] == i else []
            if number <= 8:
                chosen = _pick(rng, pools[i], number, forced)
            else:
                thresholds = [int(t) for t in np.unique(best[pools[i]])
                              if ((best <= t) & pools[i]).sum() >= 8
                              and ((best > t) & pools[i]).sum() >= number-8]
                if not thresholds:
                    okay = False
                    break
                threshold = int(rng.choice(thresholds))
                near = pools[i] & (best <= threshold)
                far = pools[i] & (best > threshold)
                nf = [p for p in forced if near.flat[p]]
                ff = [p for p in forced if far.flat[p]]
                left, right = _pick(rng, near, 8, nf), _pick(rng, far, number-8, ff)
                chosen = None if left is None or right is None else left+right
            if chosen is None:
                okay = False
                break
            cargo.flat[chosen] = True
        if not okay:
            continue
        for mask, required in ((unreachable, profile == 'partial'), (tied, profile == 'partial')):
            if mask.any() and (required or rng.random() < .65):
                number = min(int(mask.sum()), int(rng.integers(1, 5)))
                cargo.flat[_pick(rng, mask, number)] = True
        if not _small_patches(cargo):
            continue
        try:
            accepted, _, _, _, _, _ = _collect(cargo, distances)
        except ValueError:
            continue
        if accepted.any():
            return cargo
    return None


def _make_pair(rng, colors, profile, wanted, kind=None):
    for attempt in range(180):
        mode = kind or str(rng.choice(('bars', 'detour', 'ring', 'islands', 'rooms', 'mixed')))
        base, available = _geometry(rng, mode)
        if any(len(g) < 20 for g in _components(base)):
            continue
        if any(len(g) < 9 for g in _components(~base)):
            continue
        centers = _choose_centers(rng, available, wanted, profile == 'partial' or rng.random() < .4)
        if centers is None:
            continue
        cargo = _supply(rng, base, centers, profile)
        if cargo is None:
            continue
        pair = _paint(base, centers, cargo, *colors)
        try:
            _, _, _, inferred, _, inferred_cargo, inferred_centers = _decode(pair['input'])
        except ValueError:
            continue
        if np.array_equal(inferred, base) and np.array_equal(inferred_cargo, cargo) and inferred_centers == centers:
            if profile == 'saturated' and np.array_equal(_geometric_output(pair['input']), pair['output']):
                continue
            return pair
    raise RuntimeError(f'Could not construct {profile}/{kind}/{wanted} within bounded attempts.')


def _signature(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    rng = np.random.default_rng(int(seed))
    fixed_colors = tuple(map(int, rng.choice(9, 2, replace=False)))
    recolor = bool(rng.random() < .7)
    anchors = [('partial', int(rng.integers(2, 5)), 'bars'),
               ('saturated', int(rng.integers(2, 5)), 'detour')]
    plans = anchors + [('free', int(rng.integers(2, 6)), None) for _ in range(nt-2)]
    plans = [plans[int(i)] for i in rng.permutation(nt)]
    # Test shift: more simultaneously competing ports on freshly composed walls.
    tests = [('free', int(rng.integers(6, 9)), str(rng.choice(('ring', 'rooms', 'mixed', 'islands'))))
             for _ in range(ne)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, requests in (('train', plans), ('test', tests)):
        for profile, wanted, kind in requests:
            colors = tuple(map(int, rng.choice(9, 2, replace=False))) if recolor else fixed_colors
            for _ in range(12):
                pair = _make_pair(rng, colors, profile, wanted, kind)
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not obtain a distinct input.')
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(input_grid):
    a, field, solid, base, holes, cargo, centers = _decode(input_grid)
    distances = _distances(base, centers)
    accepted, counts, assigned, owner, best, ties = _collect(cargo, distances)
    direct_owner, _, _ = _owners(_direct_distances(a.shape, centers))
    overlap_count = np.zeros(a.shape, int)
    for y, x in centers:
        overlap_count[y-1:y+2, x-1:x+2] += 1
    retained = cargo & (owner >= 0) & ~accepted
    gaps = [int(best[cargo & (owner == i) & ~accepted].min()
                - best[cargo & (owner == i) & accepted].max())
            for i, n in enumerate(assigned) if n > 8]
    geometric_wrong = cargo & (owner >= 0) & (direct_owner >= 0) & (owner != direct_owner)
    return dict(shape=list(a.shape), palette=[field, solid], receivers=len(centers),
                wall_components=len(_components(base)), field_components=len(_components(~base)),
                wall_cells=int(base.sum()), cargo=int(cargo.sum()),
                cargo_components=len(_components(cargo)),
                largest_cargo_patch=max(map(len, _components(cargo)), default=0),
                counts=counts, assigned=assigned, collected=int(accepted.sum()),
                excess=int(retained.sum()), unreachable=int((cargo & (best == 10000)).sum()),
                tied=int((cargo & ties).sum()), overlap=bool((overlap_count > 1).any()),
                path_owner_witnesses=int(geometric_wrong.sum()), capacity_gaps=gaps,
                max_path=int(distances.max()),
                max_detour=int((distances-_direct_distances(a.shape, centers))[distances >= 0].max()),
                hole_coordinates=[list(p) for p in centers])


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys.')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists.')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys.')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _check(a.shape == b.shape, 'Output must preserve dimensions.')
            sig = _signature(a)
            _check(sig not in seen, 'Duplicate input.')
            seen.add(sig)
            parsed, field, solid, base, holes, cargo, centers = _decode(a)
            distance = _distances(base, centers)
            accepted, counts, assigned, owner, _, ties = _collect(cargo, distance)
            _check(np.array_equal(solve(a), b), 'Output disagrees with input-only rule.')
            _check((b[holes] == field).all(), 'A missing center must remain empty.')
            _check((b[accepted] == field).all(), 'Accepted cargo must disappear.')
            _check((b[cargo & ~accepted] == solid).all(), 'Unaccepted cargo must remain.')
            _check(int(accepted.sum()) == sum(counts), 'Collection must conserve counted cells.')
            _check(all(n == min(8, q) for n, q in zip(counts, assigned)), 'Wrong local capacity.')
            _check(not (accepted & ties).any() and not (accepted & (owner < 0)).any(), 'Unowned cargo collected.')
            highlights = np.zeros(a.shape, bool)
            for (y, x), n in zip(centers, counts):
                for dy, dx in SLOTS[:n]:
                    highlights[y+dy, x+dx] = True
            _check(np.array_equal(b == HIGHLIGHT, highlights), 'Wrong ring prefix or overlap union.')
            change_allowed = highlights | accepted
            _check(np.array_equal(a[~change_allowed], b[~change_allowed]), 'Unrelated pixels changed.')
            _check(accepted.any(), 'Generated examples must show collection.')
            features[split].append(pair_features(a))
    train = features['train']
    _check(any(4 in f['counts'] and f['overlap'] and f['tied']
               and f['unreachable'] for f in train), 'Missing partial/overlap/tie/reachability demonstration.')
    _check(any(f['excess'] and f['path_owner_witnesses'] for f in train),
           'Need saturated and path-versus-geometric training evidence.')
    _check(any(not np.array_equal(_geometric_output(p['input']), p['output'])
               for p in puzzle['train']), 'Training must visibly distinguish geometric collection.')
    return None


def self_test():
    """Bounded 42-seed audit, interface checks, mutations and targeted geometry."""
    records, shapes, structures, signatures, palettes = [], set(), set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        # Parent ablations must change actual labels, not just hidden ownership.
        different_without_b = False
        different_clockwise = False
        for p in puzzle['train']:
            a, field, _, base, holes, cargo, centers = _decode(p['input'])
            a_only = a.copy()
            a_only[cargo] = field
            a_only[_dilate(holes) & ~holes] = HIGHLIGHT
            different_without_b |= not np.array_equal(a_only, p['output'])
            accepted, counts, _, _, _, _ = _collect(cargo, _distances(base, centers))
            clockwise = a.copy()
            clockwise[accepted] = field
            order = tuple(SLOTS[i] for i in (0, 1, 2, 4, 7, 6, 5, 3))
            for (y, x), n in zip(centers, counts):
                for dy, dx in order[:n]:
                    clockwise[y+dy, x+dx] = HIGHLIGHT
            different_clockwise |= not np.array_equal(clockwise, p['output'])
            # Without A's halo access ports, a center-only start is isolated:
            # all eight wall neighbors are intact, so no cargo is reachable.
            _check((holes & _interior(base)).sum() == len(centers), 'Ablation needs intact enclosing wall.')
            _check(not np.array_equal(a, p['output']), 'Omitting puncture access must change the answer.')
        _check(different_without_b, 'A-only cleanup/full-halo rule accidentally fits training.')
        _check(different_clockwise, 'Clockwise counter order accidentally fits training.')
        signatures.add(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                _, field, solid, base, _, _, _ = _decode(a)
                f = pair_features(a)
                f['split'] = split
                records.append(f)
                shapes.add(a.shape)
                structures.add(_signature(base.astype(int)))
                palettes.add((field, solid))
        if seed == 0:
            again = generate_puzzle(seed)
            _check(all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in p), 'Nondeterministic generation.')
    _check(len(signatures) == 42 and len(structures) >= 150, 'Structural variation collapsed.')
    _check(len(shapes) >= 45 and len(palettes) >= 25, 'Shape or visible role variation collapsed.')
    for feature, minimum in (('receivers', 5), ('cargo', 15), ('collected', 12),
                             ('excess', 4), ('unreachable', 4), ('tied', 4), ('wall_components', 3)):
        _check(len({f[feature] for f in records}) >= minimum, 'Insufficient variation: '+feature)
    _check(any(f['largest_cargo_patch'] > 1 for f in records), 'Adjacent cargo not exercised.')
    _check(any(0 in f['counts'] for f in records), 'Inactive punctures not exercised.')
    _check(any(f['max_detour'] >= 20 for f in records), 'Detouring paths not exercised.')
    _check({n for f in records for n in f['counts']} == set(range(9)), 'Missing local loads.')
    # Every explicit legal count, and representative illegal values/types.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(8125+nt*10+ne, nt, ne)
            _check(len(p['train']) == nt and len(p['test']) == ne, 'Wrong explicit counts.')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted.')
    # Targeted full halo, writable ring only, partial prefix, and excess.
    base = np.zeros((15, 22), bool)
    base[6:9, :] = True
    centers = [(7, 5)]
    cargo = np.zeros_like(base)
    cargo[2, 1:5] = True
    cargo[3, 7:11] = True
    cargo[14, 17:19] = True
    p = _paint(base, centers, cargo, 0, 3)
    _check(np.array_equal(solve(p['input']), p['output']), 'Targeted saturation failed.')
    _check((p['output'][6:9, 4:7] == 9).sum() == 8 and p['output'][7, 5] == 0,
           'Full Moore halo must exclude its center.')
    _check((p['output'] == 3).sum() == int(base.sum())-1+2-8, 'Excess not retained.')
    for k in range(4):
        # Compute new displayed prefixes on rotated inputs; never rotate labels.
        rotated = np.rot90(p['input'], k).copy()
        _check(np.count_nonzero(solve(rotated) == 9) == 8, 'Routing symmetry failed.')
    # An overlap-induced ownership tie remains, while an adjacent four-cell
    # patch contributes four slots. Recoloring may even swap field/masonry.
    tie_base = np.zeros((15, 24), bool)
    tie_base[6:9] = True
    tie_centers = [(7, 8), (7, 10)]
    tie_cargo = np.zeros_like(tie_base)
    tie_cargo[2, [3, 4, 5, 6, 9]] = True
    for field, solid in ((0, 4), (4, 0)):
        pair = _paint(tie_base, tie_centers, tie_cargo, field, solid)
        expected = pair['input'].copy()
        expected[2, [3, 4, 5, 6]] = field
        expected[6, [7, 8, 9]] = HIGHLIGHT
        expected[7, 7] = HIGHLIGHT
        _check(np.array_equal(solve(pair['input']), expected), 'Partial cell count/ownership tie failed.')
        _check(expected[2, 9] == solid and expected[7, 8] == field, 'Tie or missing center lost.')
        for k in range(4):
            inp = np.rot90(pair['input'], k).copy()
            _, _, _, rb, _, rc, ports = _decode(inp)
            displayed = _paint(rb, ports, rc, field, solid)['output']
            _check(np.array_equal(solve(inp), displayed), 'Rotated partial prefix order failed.')
    # A deliberate ninth/eighth distance tie must be rejected, not silently
    # resolved by BFS order or the location of a payload cell.
    tied_input = np.zeros((21, 25), int)
    tied_input[9:12] = 3
    tied_input[10, 12] = 0
    for y, x in ((3, 12), (4, 10), (4, 14), (5, 9), (5, 15),
                 (6, 8), (6, 16), (7, 7), (7, 17)):
        tied_input[y, x] = 3
    try:
        solve(tied_input)
    except ValueError as error:
        _check('capacity frontier' in str(error), 'Tie rejected for an unrelated reason.')
    else:
        raise AssertionError('Ambiguous capacity frontier accepted.')
    broken = generate_puzzle(0, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted labels accepted.')
    return {'seeds': 42, 'pairs': len(records), 'shapes': len(shapes),
            'scaffolds': len(structures), 'palettes': len(palettes)}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
