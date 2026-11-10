"""Damaged mosaic keys: input-only geometric restoration and annotated docking.

Only standard library and NumPy. No I/O or retained generation state.
"""
from itertools import combinations, permutations

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 121085091, 26757860, 695408811, 953626757, 140617248,
              120719864, 301479758, 968999938, 452256635, 801849811,
              498748630, 561992327, 731576747, 562512398, 974543617,
              287373137, 240850588, 557383507, 153611831, 195717949,
              717414845, 23713083, 174215630, 848102049, 901407209,
              710441764, 186304161, 482957161, 767410006, 302628987,
              416210477, 851511851, 13776289, 291158948, 584541455,
              237489943, 262990498, 927063867, 77615282, 924936168, 581427000)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError('Require integer counts 2..10 train, 1..4 test, train > test')


def _grid(value):
    a = np.asarray(value)
    if not (a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
            and np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9))):
        raise ValueError('Expected an integer ARC grid with dimensions 1..30 and colors 0..9')
    return a.astype(np.int8, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(mask):
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError('Empty shape')
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _cut(a, box):
    r0, c0, r1, c1 = box
    return a[r0:r1, c0:c1]


def _sym(a, op):
    """0..3 rotations; 4..7 left-right reflection followed by rotation."""
    return np.rot90(np.fliplr(a) if op >= 4 else a, op % 4).copy()


def _inverse(a, op):
    a = np.rot90(a, -(op % 4))
    return (np.fliplr(a) if op >= 4 else a).copy()


def _signature(a):
    return a.shape, a.tobytes()


def _maximal_rectangles(mask):
    """All inclusion-maximal solid rectangles, without smoothing or metadata."""
    h, w = mask.shape
    boxes = []
    for r0 in range(h):
        columns = np.ones(w, dtype=bool)
        for r1 in range(r0 + 1, h + 1):
            columns &= mask[r1 - 1]
            starts = np.flatnonzero(columns & ~np.r_[False, columns[:-1]])
            ends = np.flatnonzero(columns & ~np.r_[columns[1:], False]) + 1
            for c0, c1 in zip(starts, ends):
                if r1 - r0 < 3 or c1 - c0 < 3:
                    continue
                if r0 and mask[r0 - 1, c0:c1].all():
                    continue
                if r1 < h and mask[r1, c0:c1].all():
                    continue
                boxes.append((r0, int(c0), r1, int(c1)))
    return boxes


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _fit_mosaic(observed, mask, background, fills):
    """Enumerate supported rectangle covers and visible scenes, reject ties."""
    boxes = _maximal_rectangles(mask)
    scenes = {}
    for count in (2, 3):
        for indices in combinations(range(len(boxes)), count):
            selected = [boxes[i] for i in indices]
            links = [[j for j in range(count) if i != j and _overlap(a, selected[j])]
                     for i, a in enumerate(selected)]
            if sorted(map(len, links)) != [1, 1] + [2] * (count - 2):
                continue
            region_masks = []
            for box in selected:
                region = np.zeros_like(mask)
                _cut(region, box)[:] = True
                region_masks.append(region)
            if not np.array_equal(np.logical_or.reduce(region_masks), mask):
                continue
            start = next(i for i in range(count) if len(links[i]) == 1)
            path = [start]
            while len(path) < count:
                path.append(next(i for i in links[path[-1]] if i not in path))
            for first in (0, 1):
                colors = {i: fills[(first + k) % 2] for k, i in enumerate(path)}
                for order in permutations(range(count)):
                    clean = np.full(mask.shape, background, dtype=np.int8)
                    owners = np.full(mask.shape, -1, dtype=np.int8)
                    for i in order:
                        clean[region_masks[i]] = colors[i]
                        owners[region_masks[i]] = i
                    supported = True
                    for i, box in enumerate(selected):
                        visible = owners == i
                        if visible.sum() < 9 or _box(visible) != box:
                            supported = False
                            break
                        if np.count_nonzero(visible & (observed == colors[i])) < .70 * visible.sum():
                            supported = False
                            break
                    if not supported:
                        continue
                    loss = int(np.count_nonzero(clean != observed))
                    key = clean.tobytes()
                    record = (loss, clean, selected, tuple(order))
                    if key not in scenes or count < len(scenes[key][2]):
                        scenes[key] = record
    ranked = sorted(scenes.values(), key=lambda x: x[0])
    if not ranked:
        raise ValueError('No supported rectangular mosaic')
    best = ranked[0]
    if best[0] > max(10, int(.14 * mask.sum())):
        raise ValueError('Too much damage')
    if len(ranked) > 1 and ranked[1][0] <= best[0] + 2:
        raise ValueError('Competing reconstructed fill scenes')
    return best[1], best[2], best[3]


def _rings(clean, centers, fills, highlight):
    centers = set(centers)
    demands = {}
    for r, c in sorted(centers):
        color = fills[1] if clean[r, c] == fills[0] else fills[0]
        for y in range(max(0, r - 1), min(clean.shape[0], r + 2)):
            for x in range(max(0, c - 1), min(clean.shape[1], c + 2)):
                if (y, x) in centers:
                    continue
                if (y, x) in demands and demands[y, x] != color:
                    raise ValueError('Conflicting non-center rings')
                demands[y, x] = color
    out = clean.copy()
    for p, color in demands.items():
        out[p] = color
    for p in centers:
        out[p] = highlight
    return out


def _analyze(input_grid):
    inp = _grid(input_grid)
    values, frequencies = np.unique(inp, return_counts=True)
    if len(values) != 4 or sorted(frequencies)[-1] == sorted(frequencies)[-2]:
        raise ValueError('Require background, lock and two fill colors')
    bg = int(values[np.argmax(frequencies)])
    answers = {}
    for lock in map(int, values):
        if lock == bg or len(_components(inp == lock)) != 2:
            continue
        target_box = _box(inp == lock)
        target = _cut(inp, target_box)
        if not np.isin(target, [bg, lock]).all():
            continue
        slot_full = target == bg
        if not slot_full.any() or len(_components(slot_full)) != 1:
            continue
        slot_box = _box(slot_full)
        slot = _cut(slot_full, slot_box)
        fills = tuple(int(c) for c in values if c not in (bg, lock))
        key_box = _box(np.isin(inp, fills))
        key = _cut(inp, key_box)
        if np.any(key == lock):
            continue
        candidates = []
        for op in range(8):
            mask = _inverse(slot, op)
            if mask.shape == key.shape:
                mismatch = int(np.count_nonzero(mask != (key != bg)))
                candidates.append((mismatch, op, mask))
        candidates.sort(key=lambda x: x[0])
        if not candidates or candidates[0][0] > 10:
            continue
        # The complete silhouette must identify a unique orientation, despite
        # erasures and exterior speckles. Colors are not an orientation cue.
        if len(candidates) > 1 and candidates[1][0] <= candidates[0][0] + 2:
            continue
        _, op, mask = candidates[0]
        try:
            clean, rectangles, order = _fit_mosaic(key, mask, bg, fills)
            holes = (key == bg) & mask
            if not holes.any():
                continue
            moved = _sym(clean, op)
            if op >= 4:
                moved = np.where(moved == fills[0], fills[1],
                                 np.where(moved == fills[1], fills[0], moved)).astype(np.int8)
            moved_holes = _sym(holes, op)
            base = np.full(inp.shape, bg, dtype=np.int8)
            base[inp == lock] = lock
            dr = target_box[0] + slot_box[0]
            dc = target_box[1] + slot_box[1]
            destination = base[dr:dr + moved.shape[0], dc:dc + moved.shape[1]]
            destination[slot] = moved[slot]
            centers = [(int(r + dr), int(c + dc)) for r, c in np.argwhere(moved_holes)]
            out = _rings(base, centers, fills, lock)
        except ValueError:
            continue
        answers[_signature(out)] = {'output': out, 'base': base, 'clean': clean,
                                   'key': key, 'key_box': key_box, 'mask': mask,
                                   'holes': holes, 'op': op, 'fills': fills,
                                   'background': bg, 'lock': lock, 'centers': centers,
                                   'rectangles': rectangles, 'order': order,
                                   'target_box': target_box, 'destination': (dr, dc)}
    if len(answers) != 1:
        raise ValueError('Input has no unique supported docking answer')
    return next(iter(answers.values()))


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _analyze(input_grid)['output']


def _sample_clean(rng, count):
    for _ in range(250):
        boxes = []
        x, y = 0, 0
        for i in range(count):
            h, w = int(rng.integers(5, 8)), int(rng.integers(5, 7))
            if i:
                x += int(rng.integers(3, 5))
                y += int(rng.choice([-2, -1, 1, 2]))
            boxes.append((y, x, y + h, x + w))
        min_y = min(b[0] for b in boxes)
        boxes = [(a - min_y, b, c - min_y, d) for a, b, c, d in boxes]
        h, w = max(b[2] for b in boxes), max(b[3] for b in boxes)
        if h > 9 or w > 14:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for b in boxes:
            _cut(mask, b)[:] = True
        maximal = _maximal_rectangles(mask)
        if not all(b in maximal for b in boxes):
            continue
        if len({_signature(_sym(mask, op)) for op in range(8)}) != 8:
            continue
        # A column-convex crosscut must leave exactly two fixed lock fragments.
        framed = np.pad(mask, ((1, 1), (0, 0)))
        if len(_components(~framed)) != 2:
            continue
        order = tuple(map(int, rng.permutation(count)))
        clean = np.zeros(mask.shape, dtype=np.int8)
        owners = np.full(mask.shape, -1, dtype=np.int8)
        for i in order:
            _cut(clean, boxes[i])[:] = 1 + i % 2
            _cut(owners, boxes[i])[:] = i
        if any((owners == i).sum() < 12 or _box(owners == i) != b for i, b in enumerate(boxes)):
            continue
        # Randomize the incoming object's axis and chirality as well.
        incoming_op = int(rng.integers(8))
        return _sym(clean, incoming_op)
    raise RuntimeError('Bounded rectangle construction exhausted')


def _damage(rng, clean, teach):
    centers = []
    def add(p):
        if p in centers:
            return False
        if any(clean[p] != clean[q] and max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 2 for q in centers):
            return False
        centers.append(p)
        return True
    interior = {color: [(r, c) for r in range(1, clean.shape[0]-1)
                        for c in range(1, clean.shape[1]-1)
                        if np.all(clean[r-1:r+2, c-1:c+2] == color)] for color in (1, 2)}
    if not all(interior.values()):
        return None
    for color in (1, 2):
        rng.shuffle(interior[color])
        if not any(add(p) for p in interior[color]):
            return None
    if teach or rng.random() < .65:
        p = centers[0]
        neighbors = [(p[0]+dr, p[1]+dc) for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))]
        rng.shuffle(neighbors)
        if not any(add(q) for q in neighbors if clean[q] == clean[p]):
            return None
    candidates = list(map(tuple, np.argwhere(clean != 0)))
    rng.shuffle(candidates)
    target = int(rng.integers(len(centers), 7))
    for p in candidates:
        if len(centers) >= target:
            break
        add(p)
    observed = clean.copy()
    for p in centers:
        observed[p] = 0
    # Genuine exterior, not a hole: occasional foreground speckles inside the
    # clean bounding box. They cannot change its extents.
    exterior = list(map(tuple, np.argwhere(clean == 0)))
    rng.shuffle(exterior)
    for p in exterior[:int(rng.integers(0, 3))]:
        observed[p] = int(rng.integers(1, 3))
    wrong = [p for p in candidates if p not in centers]
    rng.shuffle(wrong)
    # Witnesses always include an ordinary color error outside every halo,
    # making the absence of a marker for wrong-fill noise observable.
    if teach:
        wrong.sort(key=lambda p: any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1 for q in centers))
    for p in wrong[:int(rng.integers(1 if teach else 0, 4))]:
        observed[p] = 3 - clean[p]
    if _box(observed != 0) != (0, 0, *clean.shape):
        return None
    return observed


def _pack(rng, a_shape, b_shape):
    # Place independently when possible; finite orthogonal fallback always fits
    # the bounded object sizes. A two-cell gap keeps rings off the old key.
    for _ in range(100):
        h, w = int(rng.integers(max(a_shape[0], b_shape[0]), 31)), int(rng.integers(max(a_shape[1], b_shape[1]), 31))
        ar, ac = int(rng.integers(h-a_shape[0]+1)), int(rng.integers(w-a_shape[1]+1))
        br, bc = int(rng.integers(h-b_shape[0]+1)), int(rng.integers(w-b_shape[1]+1))
        if (ar+a_shape[0]+2 <= br or br+b_shape[0]+2 <= ar
                or ac+a_shape[1]+2 <= bc or bc+b_shape[1]+2 <= ac):
            return (h, w), (ar, ac), (br, bc)
    vertical = a_shape[0] + b_shape[0] + 2 <= 30
    if vertical:
        h, w = a_shape[0]+b_shape[0]+2, max(a_shape[1], b_shape[1])
        return (h, w), (0, int(rng.integers(w-a_shape[1]+1))), (a_shape[0]+2, int(rng.integers(w-b_shape[1]+1)))
    h, w = max(a_shape[0], b_shape[0]), a_shape[1]+b_shape[1]+2
    if w > 30:
        raise RuntimeError('Objects do not fit')
    return (h, w), (int(rng.integers(h-a_shape[0]+1)), 0), (int(rng.integers(h-b_shape[0]+1)), a_shape[1]+2)


def _pair(rng, count, parity, palette, reused=None, teach=False):
    for _ in range(180):
        clean = reused[0].copy() if reused is not None else _sample_clean(rng, count)
        observed = reused[1].copy() if reused is not None else _damage(rng, clean, teach)
        if observed is None:
            continue
        op = int(rng.integers(4)) + 4 * parity
        moved = _sym(clean, op)
        # Add the two lock rails along either axis in destination coordinates.
        axis = int(rng.integers(2))
        padding = ((1, 1), (0, 0)) if axis == 0 else ((0, 0), (1, 1))
        slot = np.pad(moved != 0, padding)
        if len(_components(~slot)) != 2:
            continue
        shape, origin_key, origin_lock = _pack(rng, clean.shape, slot.shape)
        bg, lock, fill1, fill2 = palette
        mapping = np.array([bg, fill1, fill2], dtype=np.int8)
        inp = np.full(shape, bg, dtype=np.int8)
        kr, kc = origin_key
        lr, lc = origin_lock
        inp[kr:kr+clean.shape[0], kc:kc+clean.shape[1]] = mapping[observed]
        lock_view = inp[lr:lr+slot.shape[0], lc:lc+slot.shape[1]]
        lock_view[~slot] = lock
        # Construct expected output directly from the sampled clean object,
        # then require input-only inference to recover the same scene.
        moved_values = mapping[moved]
        if parity:
            moved_values = np.where(moved == 1, fill2, np.where(moved == 2, fill1, bg)).astype(np.int8)
        base = np.full(shape, bg, dtype=np.int8)
        base[inp == lock] = lock
        dr, dc = lr + (axis == 0), lc + (axis == 1)
        view = base[dr:dr+moved.shape[0], dc:dc+moved.shape[1]]
        view[moved != 0] = moved_values[moved != 0]
        holes = _sym((clean != 0) & (observed == 0), op)
        centers = [(int(r+dr), int(c+dc)) for r, c in np.argwhere(holes)]
        expected = _rings(base, centers, (fill1, fill2), lock)
        try:
            a = _analyze(inp)
        except ValueError:
            if reused is not None:
                # Placement, padding axis or destination symmetry can change;
                # the reused object and its actual defects stay fixed.
                continue
            continue
        if a['op'] != op or len(a['rectangles']) != count or not np.array_equal(a['clean'], mapping[clean]):
            continue
        if not np.array_equal(expected, a['output']):
            continue
        if teach:
            wrong = (observed != clean) & (observed != 0) & (clean != 0)
            clear_wrong = any(all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) > 1
                                 for q in np.argwhere((clean != 0) & (observed == 0)))
                              for p in np.argwhere(wrong))
            if not clear_wrong:
                continue
        return {'input': inp, 'output': expected}, (clean, observed)
    raise RuntimeError('Bounded unambiguous key construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    reused = None
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            n = 2 if split == 'train' else 3
            parity = i if split == 'train' and i < 2 else int(rng.integers(2))
            colors = palette if stable or (split == 'train' and i < 2) else tuple(map(int, rng.choice(10, 4, replace=False)))
            for _ in range(12):
                pair, material = _pair(rng, n, parity, colors,
                                       reused if split == 'train' and i == 1 else None,
                                       teach=split == 'train' and i < 2)
                key = _signature(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train' and i == 0:
                        reused = material
                    break
            else:
                raise RuntimeError('Could not construct distinct inputs')
    rng.shuffle(puzzle['train'])
    return puzzle


def _ablation_outputs(a):
    """Explicit wrong alternatives for checking that both inheritances matter."""
    no_b = a['base'].copy()  # repairs and docks, but omits erasure annotation
    no_a = np.full(a['base'].shape, a['background'], dtype=np.int8)
    no_a[a['base'] == a['lock']] = a['lock']
    kr, kc, kr1, kc1 = a['key_box']
    no_a[kr:kr1, kc:kc1] = a['clean']
    source_centers = [(int(r+kr), int(c+kc)) for r, c in np.argwhere(a['holes'])]
    no_a = _rings(no_a, source_centers, a['fills'], a['lock'])
    no_swap = a['base'].copy()
    for color, other in zip(a['fills'], a['fills'][::-1]):
        no_swap[a['base'] == color] = other
    no_swap = _rings(no_swap, a['centers'], a['fills'], a['lock'])
    return no_b, no_swap, no_a


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in seen
            seen.add(signature)
            a = _analyze(inp)
            assert np.array_equal(out, a['output']), 'Input-inferred answer differs'
            assert not np.array_equal(inp, out)
            kr0, kc0, kr1, kc1 = a['key_box']
            assert np.all(out[kr0:kr1, kc0:kc1] == a['background']), 'Old key not erased'
            assert all(out[p] == a['lock'] for p in a['centers'])
            no_b, no_swap, no_a = _ablation_outputs(a)
            assert not np.array_equal(out, no_b), 'B annotation has no effect'
            assert not np.array_equal(out, no_a), 'A docking has no effect'
            if a['op'] >= 4:
                assert not np.array_equal(out, no_swap), 'Reflection exchange has no effect'
            assert not np.array_equal(out, _rings(a['base'], [], a['fills'], a['lock']))
            (training if split == 'train' else testing).append(a)
    assert {a['op'] >= 4 for a in training} == {False, True}
    assert all(len(a['rectangles']) == 2 for a in training)
    assert all(len(a['rectangles']) == 3 for a in testing)
    for color_index in (0, 1):
        assert any(np.any(a['holes'] & (a['clean'] == a['fills'][color_index])) for a in training)
    assert any(any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1
                       and a['clean'][p] == a['clean'][q]
                       for i, p in enumerate(list(map(tuple, np.argwhere(a['holes']))))
                       for q in list(map(tuple, np.argwhere(a['holes'])))[:i]) for a in training)


def self_test():
    """Bounded: stream 42 puzzles, retain metrics and at most two puzzles."""
    shapes, structures, operations, backgrounds, locks, rect_shapes = set(), set(), set(), set(), set(), set()
    counts, signatures = set(), set()
    metrics = {'pairs': 0, 'adjacent': 0, 'boundary_holes': 0, 'speckled': 0,
               'wrong_fill': 0, 'seam_wrong': 0, 'edge_lock': 0, 'ring_crossing': 0}
    first = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if first is None:
            first = puzzle
        signatures.add(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for pair in puzzle['train'] + puzzle['test']:
            a = _analyze(pair['input'])
            shapes.add(pair['input'].shape)
            norm = {}
            structures.add(tuple(norm.setdefault(int(c), len(norm)) for c in pair['input'].flat))
            operations.add(a['op'])
            backgrounds.add(a['background'])
            locks.add(a['lock'])
            counts.add(len(a['rectangles']))
            rect_shapes.add(tuple(sorted((b[2]-b[0], b[3]-b[1]) for b in a['rectangles'])))
            holes = list(map(tuple, np.argwhere(a['holes'])))
            metrics['pairs'] += 1
            metrics['adjacent'] += any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1 for i, p in enumerate(holes) for q in holes[:i])
            metrics['boundary_holes'] += any(p[0] in (0, a['clean'].shape[0]-1) or p[1] in (0, a['clean'].shape[1]-1) for p in holes)
            metrics['speckled'] += bool(np.any((a['clean'] == a['background']) & (a['key'] != a['background'])))
            wrong = (a['clean'] != a['key']) & (a['clean'] != a['background']) & (a['key'] != a['background'])
            metrics['wrong_fill'] += bool(wrong.any())
            for r, c in np.argwhere(wrong):
                local = a['clean'][max(0,r-1):r+2, max(0,c-1):c+2]
                if all(color in local for color in a['fills']):
                    metrics['seam_wrong'] += 1
            box = a['target_box']
            metrics['edge_lock'] += box[0] == 0 or box[1] == 0 or box[2] == pair['input'].shape[0] or box[3] == pair['input'].shape[1]
            metrics['ring_crossing'] += bool(np.any((a['base'] == a['lock']) & np.isin(pair['output'], a['fills'])))
            assert np.array_equal(_rings(a['base'], a['centers'][::-1], a['fills'], a['lock']), pair['output'])
    assert len(signatures) == 42 and len(shapes) >= 35 and len(structures) >= 150
    assert operations == set(range(8)) and counts == {2, 3}
    assert len(backgrounds) >= 8 and len(locks) >= 8 and len(rect_shapes) >= 35
    assert all(v > 0 for v in metrics.values()), metrics
    repeated = generate_puzzle(HOST_SEEDS[0])
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(first[s], repeated[s]) for k in ('input', 'output'))
    # All legal count combinations, and representative illegal values/types.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(7100 + 10*a+b, a, b)
            validate_generated_puzzle(p)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('Mutated label accepted')
    fixture = np.ones((7, 8), dtype=np.int8)
    fixture[:, 4:] = 2
    try:
        _rings(fixture, [(3, 3), (3, 4)], (1, 2), 8)
    except ValueError:
        pass
    else:
        raise AssertionError('Conflicting rings accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'grid_shapes': len(shapes),
            'color_normalized_inputs': len(structures), 'rectangle_size_combinations': len(rect_shapes),
            'operations': sorted(operations), **metrics}
