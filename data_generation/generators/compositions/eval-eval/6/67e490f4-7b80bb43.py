"""Mended districts: repaired fences delimit shape-conditioned ballot pools.

Only NumPy and the standard library are used. Generation builds clean geometry
and labels from ballots; solve recovers roles, repairs, districts and ballots
from the visible input. No parent module or stored example is used.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np


STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _normal(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    a, b = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r-a, c-b) for r, c in cells))


@lru_cache(maxsize=4096)
def _variants(shape):
    variants = set()
    for mirror in (1, -1):
        current = _normal((r, c*mirror) for r, c in shape)
        for _ in range(4):
            variants.add(current)
            current = _normal((c, -r) for r, c in current)
    return tuple(sorted(variants))


def _canon(cells):
    return min(_variants(_normal(cells)))


def _size(cells):
    return max(r for r, _ in cells)+1, max(c for _, c in cells)+1


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in STEPS:
                yy, xx = y+dy, x+dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(tuple(cells))
    return result


def _labels(walls):
    labels = np.full(walls.shape, -1, int)
    groups = _components(~walls)
    for i, cells in enumerate(groups):
        for p in cells:
            labels[p] = i
    return labels, groups


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _straight(mask):
    result = np.zeros_like(mask)
    for transpose in (False, True):
        m = mask.T if transpose else mask
        target = result.T if transpose else result
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for k in range(3):
            target[:, k:k+triples.shape[1]] |= triples
    return result


def _gaps(mask):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        hits = m[:, :-4] & m[:, 4:] & ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(hits):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _restore(mask):
    """Read all cues simultaneously, then delete unsupported oblique cells."""
    strong = _straight(mask)
    additions, anchors = np.zeros_like(mask), np.zeros_like(mask)
    repairs = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r-dr, c-dc) or _at(mask, r+5*dr, c+5*dc)):
            continue
        matches = []
        for end in (0, 4):
            ar, ac = r+end*dr, c+end*dc
            toward = 1 if end == 0 else -1
            for side in (-1, 1):
                dy, dx = toward*dr+side*dc, toward*dc+side*dr
                spur = tuple((ar+k*dy, ac+k*dx) for k in (1, 2))
                if (all(_at(mask, y, x) and not strong[y, x] for y, x in spur)
                        and not _at(mask, ar+3*dy, ac+3*dx)):
                    matches.append(spur)
        if matches:
            assert len(matches) == 1, 'ambiguous repair cue'
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            for p in gap:
                additions[p] = True
            anchors[r, c] = anchors[r+4*dr, c+4*dc] = True
            repairs.append((gap, matches[0]))
    weak = mask & ~_straight(mask | additions) & ~anchors
    oblique = np.zeros_like(mask)
    for dc in (-1, 1):
        a = weak[:-1, :-1] if dc == 1 else weak[:-1, 1:]
        b = weak[1:, 1:] if dc == 1 else weak[1:, :-1]
        links = a & b
        if dc == 1:
            oblique[:-1, :-1] |= links
            oblique[1:, 1:] |= links
        else:
            oblique[:-1, 1:] |= links
            oblique[1:, :-1] |= links
    return (mask & ~oblique) | additions, repairs


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and ((a >= 0) & (a <= 9)).all()
    return a.astype(int, copy=False)


def _read(inp):
    inp = _grid(inp)
    colors, counts = np.unique(inp, return_counts=True)
    assert (counts == max(counts)).sum() == 1
    bg = int(colors[np.argmax(counts)])
    wall = int(inp[0, 0])
    assert wall != bg
    assert (np.all(inp[0] == wall) and np.all(inp[-1] == wall)
            and np.all(inp[:, 0] == wall) and np.all(inp[:, -1] == wall))
    candidates = []
    for color in colors:
        color = int(color)
        if color in (bg, wall):
            continue
        plaques = []
        for cells in _components(inp == color):
            y, x = min(r for r, _ in cells), min(c for _, c in cells)
            b, e = max(r for r, _ in cells), max(c for _, c in cells)
            box = inp[y:b+1, x:e+1]
            if min(box.shape) < 3:
                break
            if not (np.all(box[0] == color) and np.all(box[-1] == color)
                    and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)
                    and set(map(int, np.unique(box))) == {bg, color}):
                break
            holes = [tuple((r+y, c+x) for r, c in hole) for hole in _components(box == bg)]
            plaques.append((y, x, b, e, holes))
        else:
            if plaques:
                candidates.append((color, plaques))
    assert len(candidates) == 1, 'one geometrically identifiable carrier role required'
    carrier, plaques = candidates[0]
    walls, repairs = _restore(inp == wall)
    labels, districts = _labels(walls)
    votes = defaultdict(Counter)
    objects = []
    for color in colors:
        color = int(color)
        if color in (bg, wall, carrier):
            continue
        for cells in _components(inp == color):
            ds = {int(labels[p]) for p in cells}
            assert len(ds) == 1 and -1 not in ds
            district = ds.pop()
            key = district, _canon(cells)
            votes[key][color] += 1
            objects.append((color, cells, key))
    holes = []
    for y, x, b, e, cavities in plaques:
        ds = set(map(int, labels[y:b+1, x:e+1].ravel()))
        assert len(ds) == 1 and -1 not in ds
        district = ds.pop()
        holes.extend((cells, (district, _canon(cells))) for cells in cavities)
    assert set(votes) == {key for _, key in holes}, 'every local class has holes and votes'
    assert {d for d, _ in votes} == set(range(len(districts))), 'each district participates'
    winners = {}
    for key, ballot in votes.items():
        high = max(ballot.values())
        best = [c for c, n in ballot.items() if n == high]
        assert len(best) == 1, 'local plurality must be unique'
        winners[key] = best[0]
    return dict(bg=bg, wall=wall, carrier=carrier, walls=walls, repairs=repairs,
                labels=labels, districts=districts, plaques=plaques, holes=holes,
                objects=objects, votes=dict(votes), winners=winners)


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    a = _read(inp)
    out = np.full(inp.shape, a['bg'], int)
    out[a['walls']] = a['wall']
    out[inp == a['carrier']] = a['carrier']
    for cells, key in a['holes']:
        for p in cells:
            out[p] = a['winners'][key]
    return out


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _halo(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([p[r:r+h, c:c+w] for r in range(3) for c in range(3)])


def _block(available, cells):
    for r, c in cells:
        available[max(0, r-1):r+2, max(0, c-1):c+2] = False


def _place(rng, available, variants, allowed=None):
    usable = available if allowed is None else available & allowed
    for index in rng.permutation(len(variants)):
        shape = variants[int(index)]
        h, w = _size(shape)
        nr, nc = usable.shape[0]-h+1, usable.shape[1]-w+1
        if min(nr, nc) <= 0:
            continue
        valid = np.ones((nr, nc), bool)
        for r, c in shape:
            valid &= usable[r:r+nr, c:c+nc]
        positions = np.argwhere(valid)
        if len(positions):
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            cells = tuple((r+y, c+x) for y, x in shape)
            _block(available, cells)
            return cells, shape, r, c
    return None


def _floorplan(rng, leaves_wanted, teaching=False):
    h, w = int(rng.integers(26 if teaching else 23, 31)), int(rng.integers(26 if teaching else 23, 31))
    wall = np.zeros((h, w), bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    leaves, nodes = [(1, h-2, 1, w-2)], []
    minimum = 9 if teaching else 8
    while len(leaves) < leaves_wanted:
        choices = []
        for i, (y, b, x, e) in enumerate(leaves):
            if b-y+1 >= 2*minimum+1:
                choices.append((i, 0))
            if e-x+1 >= 2*minimum+1:
                choices.append((i, 1))
        if not choices:
            return None
        i, axis = _pick(rng, choices)
        y, b, x, e = leaves.pop(i)
        if axis == 0:
            cut = int(rng.integers(y+minimum, b-minimum+1))
            wall[cut, x:e+1] = True
            cells = tuple((cut, c) for c in range(x, e+1))
            leaves.extend([(y, cut-1, x, e), (cut+1, b, x, e)])
        else:
            cut = int(rng.integers(x+minimum, e-minimum+1))
            wall[y:b+1, cut] = True
            cells = tuple((r, cut) for r in range(y, b+1))
            leaves.extend([(y, b, x, cut-1), (y, b, cut+1, e)])
        nodes.append(cells)
    # Doorways are deliberately missing sections, not corrupted fence segments.
    statuses = ['repair']
    for _ in nodes[1:]:
        statuses.append('door' if teaching else _pick(rng, ('door', 'door', 'repair', 'intact')))
    intended = wall.copy()
    for cells, status in zip(nodes, statuses):
        if status != 'door':
            continue
        starts = []
        for j in range(2, len(cells)-5):
            gap = cells[j+1:j+4]
            if all(not (wall[r-1, c] or wall[r+1, c]) if cells[0][0] == cells[-1][0]
                   else not (wall[r, c-1] or wall[r, c+1]) for r, c in gap):
                starts.append(j)
        if not starts:
            return None
        j = _pick(rng, starts)
        for p in cells[j+1:j+4]:
            intended[p] = False
    noisy = intended.copy()
    for cells, status in zip(nodes, statuses):
        if status != 'repair':
            continue
        options = []
        dr = int(cells[0][1] == cells[-1][1])
        dc = 1-dr
        for j in range(2, len(cells)-5):
            gap = cells[j+1:j+4]
            if not all(noisy[p] for p in gap):
                continue
            if any(_at(intended, r+dc, c+dr) or _at(intended, r-dc, c-dr) for r, c in gap):
                continue
            for end in (j, j+4):
                ar, ac = cells[end]
                toward = 1 if end == j else -1
                for side in (-1, 1):
                    spur = tuple((ar+k*(toward*dr+side*dc), ac+k*(toward*dc+side*dr)) for k in (1, 2))
                    if not all(0 < r < h-1 and 0 < c < w-1 and not noisy[r, c] for r, c in spur):
                        continue
                    trial = noisy.copy()
                    for p in gap:
                        trial[p] = False
                    for p in spur:
                        trial[p] = True
                    try:
                        restored, evidence = _restore(trial)
                    except AssertionError:
                        continue
                    if np.array_equal(restored, intended):
                        options.append(trial)
        if not options:
            return None
        noisy = _pick(rng, options)
    # Detached oblique clutter is optional; every accepted scene reconstructs.
    for _ in range(int(rng.integers(3))):
        free = ~_halo(noisy | intended)
        for _ in range(20):
            r, c = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
            dc = int(rng.choice((-1, 1)))
            cells = tuple((r+k, c+k*dc) for k in range(int(rng.integers(2, 4))))
            if all(0 < y < h-1 and 0 < x < w-1 and free[y, x] for y, x in cells):
                trial = noisy.copy()
                for p in cells:
                    trial[p] = True
                try:
                    if np.array_equal(_restore(trial)[0], intended):
                        noisy = trial
                        break
                except AssertionError:
                    pass
    if not np.array_equal(_restore(noisy)[0], intended):
        return None
    leaf_masks = []
    for y, b, x, e in leaves:
        m = np.zeros_like(wall)
        m[y:b+1, x:e+1] = True
        leaf_masks.append(m)
    return noisy, intended, leaf_masks


S4 = _canon(((0, 1), (0, 2), (1, 0), (1, 1)))
L4 = _canon(((0, 0), (1, 0), (2, 0), (2, 1)))
T4 = _canon(((0, 0), (0, 1), (0, 2), (1, 1)))


def _pool(rng):
    # Guaranteed small and large classes also bound rare procedural outcomes.
    shapes = {S4, L4, T4,
              _canon(tuple((r, c) for r in range(2) for c in range(3))),
              _canon(((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2))),
              _canon(((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3)))}
    for _ in range(100):
        cells = {(0, 0)}
        for _ in range(int(rng.integers(3, 7))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in STEPS} - cells)
            cells.add(_pick(rng, frontier))
        s = _canon(cells)
        h, w = _size(s)
        if h <= 4 and w <= 4:
            # An internal hole would make the carrier role ambiguous.
            m = np.ones((h+2, w+2), bool)
            for r, c in s:
                m[r+1, c+1] = False
            if len(_components(m)) == 1:
                shapes.add(s)
    return tuple(sorted(shapes))


def _ballot(rng, colors):
    pattern = _pick(rng, ((1,), (2,), (2, 1), (3, 1), (3, 2), (2, 1, 1), (3, 2, 1), (3, 2, 2)))
    palette = list(map(int, rng.choice(colors, len(pattern), replace=False)))
    return [(c, n) for c, n in zip(palette, pattern)]


def _make_pair(rng, pool, teaching, second=False, complex_test=False):
    for attempt in range(160):
        leaves_wanted = 3 if teaching else (2 if second else int(rng.integers(3 if complex_test else 2, 6)))
        floor = _floorplan(rng, leaves_wanted, teaching)
        if floor is None:
            continue
        noisy, intended, leaves = floor
        labels, regions = _labels(intended)
        if not 2 <= len(regions) <= 3:
            continue
        palette = list(map(int, rng.permutation(10)))
        bg, wall, carrier = palette[:3]
        X, Y, Z, U, V = palette[3:8]
        colors = palette[3:]
        region_leaves = {d: [m for m in leaves if np.any(m & (labels == d))] for d in range(len(regions))}
        if teaching:
            singles = [d for d, ls in region_leaves.items() if len(ls) == 1]
            merged = [d for d, ls in region_leaves.items() if len(ls) == 2]
            if len(singles) != 1 or len(merged) != 1:
                continue
            d0, d1 = singles[0], merged[0]
            s0, s1 = S4, L4
            specs = {(d0, s0): [(X, 3), (Y, 1)], (d1, s0): [(Y, 3), (X, 2)],
                     (d0, s1): [(Z, 2), (U, 1), (V, 1)], (d1, s1): [(U, 1)]}
        else:
            k = 2 if second else int(rng.integers(1, 3))
            shapes = list(_pick(rng, ((S4, L4), (S4, T4)))) if second else [pool[int(i)] for i in rng.choice(len(pool), k, replace=False)]
            specs = {(d, s): _ballot(rng, colors) for d in range(len(regions)) for s in shapes}
            # Physical load is bounded without silently dropping a requested class.
            if sum(sum(n for _, n in b) for b in specs.values()) > 20:
                continue
        # Geometry is fixed for a bounded content-packing search.
        for _ in range(18):
            available = ~_halo(noisy | intended)
            inp = np.full(intended.shape, bg, int)
            inp[noisy] = wall
            out = np.full(intended.shape, bg, int)
            out[intended] = wall
            failed = False
            plaque_jobs, vote_jobs = [], []
            special_leaf = None
            if teaching:
                special_leaf = max(region_leaves[d1], key=lambda m: int(m.sum()))
            for (d, shape), ballot in specs.items():
                winner = ballot[0][0]
                repeats = 1 + int(not teaching and rng.random() < .18 and len(specs) <= 3)
                for rep in range(repeats):
                    allowed = special_leaf if teaching and d == d1 and shape == s0 else labels == d
                    plaque_jobs.append((d, shape, winner, allowed))
                for color, n in ballot:
                    for index in range(n):
                        allowed = labels == d
                        if teaching and d == d1 and shape == s0:
                            # Plaque side: X2 Y1; far side: Y2. Door pooling reverses X to Y.
                            allowed = special_leaf if color == X or index == 0 else (labels == d) & ~special_leaf
                        vote_jobs.append((d, shape, color, allowed))
            # Rectangles first, with the tight teaching leaf first.
            plaque_jobs.sort(key=lambda job: int(job[3].sum()))
            for d, shape, winner, allowed in plaque_jobs:
                variants = _variants(shape)
                options = []
                for hole in variants:
                    hh, ww = _size(hole)
                    rect = tuple((r, c) for r in range(hh+2) for c in range(ww+2))
                    options.append((hole, rect))
                placed = None
                for j in rng.permutation(len(options)):
                    hole, rect = options[int(j)]
                    trial_available = available.copy()
                    result = _place(rng, trial_available, (rect,), allowed)
                    if result:
                        available = trial_available
                        placed = hole, result
                        break
                if placed is None:
                    failed = True
                    break
                hole, (cells, _, y, x) = placed
                for p in cells:
                    inp[p] = out[p] = carrier
                for r, c in hole:
                    inp[y+r+1, x+c+1] = bg
                    out[y+r+1, x+c+1] = winner
            if failed:
                continue
            # Most spatially constrained objects first, then large shapes.
            rng.shuffle(vote_jobs)
            vote_jobs.sort(key=lambda job: (int(job[3].sum()), -len(job[1])))
            for d, shape, color, allowed in vote_jobs:
                result = _place(rng, available, _variants(shape), allowed)
                if result is None:
                    failed = True
                    break
                for p in result[0]:
                    inp[p] = color
            if failed:
                continue
            colors_in, counts = np.unique(inp, return_counts=True)
            if int(colors_in[np.argmax(counts)]) != bg:
                continue
            pair = {'input': inp, 'output': out}
            try:
                a = _read(inp)
                if not np.array_equal(solve(inp), out):
                    continue
                if not all(_alternative_differs(pair, mode) for mode in ('global', 'unrepaired')):
                    continue
                if teaching and not all(_alternative_differs(pair, mode) for mode in ('global', 'area', 'unrepaired', 'close_all', 'nearest')):
                    continue
                # Reflection/rotation are taught when every test is free-shape.
                if teaching and not _alternative_differs(pair, 'translation'):
                    continue
                if teaching and not _alternative_differs(pair, 'rotation'):
                    continue
            except AssertionError:
                continue
            return pair
    raise RuntimeError('bounded packing exhausted')


def _alternative_differs(pair, mode):
    """Counterfactual checks on actual holes, not just unused object bookkeeping."""
    a = _read(pair['input'])
    if mode == 'nearest':
        for cells, key in a['holes']:
            distances = [(min(abs(r-y)+abs(c-x) for r, c in cells for y, x in obj), color)
                         for color, obj, obj_key in a['objects'] if obj_key == key]
            smallest = min(d for d, _ in distances)
            closest_colors = {color for d, color in distances if d == smallest}
            if len(closest_colors) == 1 and a['winners'][key] not in closest_colors:
                return True
        return False
    walls = a['walls'].copy()
    if mode == 'unrepaired':
        for gap, _ in a['repairs']:
            for p in gap:
                walls[p] = False
    elif mode == 'close_all':
        for r, c, dr, dc in _gaps(walls):
            for k in (1, 2, 3):
                walls[r+k*dr, c+k*dc] = True
    labels, _ = _labels(walls)
    def key(cells):
        d = 0 if mode == 'global' else int(labels[cells[0]])
        s = _canon(cells)
        if mode == 'area':
            s = len(cells)
        elif mode == 'translation':
            s = _normal(cells)
        elif mode == 'rotation':
            current, variants = _normal(cells), []
            for _ in range(4):
                variants.append(current)
                current = _normal((c, -r) for r, c in current)
            s = min(variants)
        return d, s
    counts = defaultdict(Counter)
    for color, cells, _ in a['objects']:
        counts[key(cells)][color] += 1
    for cells, original_key in a['holes']:
        ballot = counts[key(cells)]
        high = max(ballot.values(), default=0)
        best = [c for c, n in ballot.items() if n == high]
        if best != [a['winners'][original_key]]:
            return True
    return False


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= value <= high:
        raise ValueError(name + ' is out of range')
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 2, 10, 'num_train')
    t = _count(num_test, 1, 4, 'num_test')
    if n <= t:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    pool = _pool(rng)
    # Tests contain fresh classes with visible local ballots; no memorized dictionary.
    train_pool = [s for s in pool if len(s) <= 5]
    test_pool = [s for s in pool if len(s) >= 6]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for i in range(count):
            for _ in range(12):
                pair = _make_pair(rng, train_pool if split == 'train' else test_pool,
                                  teaching=split == 'train' and i == 0,
                                  second=split == 'train' and i == 1,
                                  complex_test=split == 'test')
                # Whole-scene rotations/reflections preserve fence and plaque semantics.
                turns, flip = int(rng.integers(4)), bool(rng.integers(2))
                pair = {k: np.rot90(v, turns)[:, ::-1].copy() if flip else np.rot90(v, turns).copy() for k, v in pair.items()}
                if split == 'train' and i == 1:
                    first_gap = _read(puzzle['train'][0]['input'])['repairs'][0][0]
                    this_gap = _read(pair['input'])['repairs'][0][0]
                    if (first_gap[0][0] == first_gap[1][0]) == (this_gap[0][0] == this_gap[1][0]):
                        pair = {k: np.rot90(v).copy() for k, v in pair.items()}
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('unique input construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, infos = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _read(inp)
            assert np.array_equal(solve(inp), out), 'incomplete repair, incorrect vote, or extraneous output'
            assert len(a['districts']) >= 2 and a['repairs']
            m = a['walls']
            assert not (m[:-1, :-1] & m[:-1, 1:] & m[1:, :-1] & m[1:, 1:]).any(), 'thick fence'
            # The separate objects are separated even under eight-neighbor contact.
            occupied = np.zeros(inp.shape, bool)
            for _, cells, _ in a['objects']:
                assert not any(occupied[p] for p in cells), 'touching ballot tokens'
                mask = np.zeros(inp.shape, bool)
                for p in cells:
                    mask[p] = True
                occupied |= _halo(mask)
            # Repairs have exactly three added fence cells and two erased cue cells.
            for gap, spur in a['repairs']:
                assert len(gap) == 3 and len(spur) == 2
                assert all(inp[p] == a['bg'] and out[p] == a['wall'] for p in gap)
                assert all(inp[p] == a['wall'] and out[p] == a['bg'] for p in spur)
            for color, cells, _ in a['objects']:
                assert all(out[p] == a['bg'] for p in cells)
            # Carrier cells and full frame survive; no token straddles a district.
            assert np.array_equal(inp == a['carrier'], out == a['carrier'])
            infos.append((split, a))
    train = puzzle['train']
    for mode in ('global', 'area', 'unrepaired', 'close_all', 'translation', 'rotation', 'nearest'):
        assert any(_alternative_differs(pair, mode) for pair in train), 'training lacks ' + mode + ' contrast'
    assert {gap[0][0] == gap[1][0] for split, a in infos if split == 'train'
            for gap, _ in a['repairs']} == {False, True}, 'training must show both repair axes'
    assert any(len(b) >= 3 and max(b.values())*2 <= sum(b.values())
               for split, a in infos if split == 'train' for b in a['votes'].values())
    trained = {s for split, a in infos if split == 'train' for _, s in a['votes']}
    assert all(any(s not in trained for _, s in a['votes']) for split, a in infos if split == 'test')


def self_test():
    shapes, geometries, structures, roles, ballots = set(), set(), set(), set(), set()
    districts, repairs, doors, signatures = set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _read(pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                geometries.add(pair['input'].shape)
                roles.add((a['bg'], a['wall'], a['carrier']))
                districts.add(len(a['districts']))
                repairs.add(len(a['repairs']))
                doors.add(len(_gaps(a['walls'])))
                shapes.update(s for _, s in a['votes'])
                ballots.update(tuple(sorted(v.values(), reverse=True)) for v in a['votes'].values())
                mapping = {}
                structures.add(tuple(mapping.setdefault(int(v), len(mapping)) for v in pair['input'].ravel()))
    assert len(signatures) == 42 and len(geometries) >= 25 and len(structures) >= 180
    assert len(roles) >= 80 and len(shapes) >= 20 and len(ballots) >= 7
    assert {2, 3} <= districts and max(repairs) >= 2 and max(doors) >= 2
    assert (3, 2, 2) in ballots and (1,) in ballots
    # The host exercises every legal count. Keep this additional sweep bounded.
    for n, t in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(321, n, t)
        assert len(p['train']) == n and len(p['test']) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0, 2, 1)
    q = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(p[s][i][k], q[s][i][k]) for s in p for i in range(len(p[s])) for k in ('input', 'output'))
    # Reject a partially completed answer, and then a wrong district winner.
    for kind in ('repair', 'vote'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad['train'][0]
        a = _read(pair['input'])
        if kind == 'repair':
            for pos in a['repairs'][0][0]:
                pair['output'][pos] = a['bg']
        else:
            cells, key = a['holes'][0]
            for pos in cells:
                pair['output'][pos] = next(c for c in range(10) if c not in (a['bg'], a['wall'], a['carrier'], a['winners'][key]))
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('bad label accepted')


if __name__ == '__main__':
    self_test()
