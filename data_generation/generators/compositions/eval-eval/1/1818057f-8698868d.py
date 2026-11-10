"""Cross punches change stencil topology; a dot-count request selects the print.

Only standard library and NumPy. Prediction reads the supplied input alone.
No files, examples, seeds, or generation metadata enter solve().
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
VERIFICATION_SEEDS = (0, 538935795, 371537644, 253373609, 61059764, 429559012,
    577028018, 903921511, 692660511, 960476183, 91925742, 967722382, 876849350,
    966338448, 990646043, 32624841, 915157791, 479680283, 136716236, 175290747,
    823917978, 49662084, 399156218, 185894597, 318072339, 749555446, 371661352,
    965470059, 531429457, 826393838, 997707975, 995104183, 700691047, 758585740,
    471109172, 4208475, 766048378, 261208282, 623818164, 507391417, 129591896,
    325138567)


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    steps = STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else STEPS
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, group = [(r, c)], []
        while stack:
            a, b = stack.pop()
            group.append((a, b))
            for dr, dc in steps:
                x, y = a + dr, b + dc
                if 0 <= x < h and 0 <= y < w and mask[x, y] and not seen[x, y]:
                    seen[x, y] = True
                    stack.append((x, y))
        groups.append(group)
    return groups


def _count(mask, diagonal=False):
    return len(_components(mask, diagonal))


def _plus(mask):
    centers = np.zeros_like(mask, dtype=bool)
    if min(mask.shape) >= 3:
        centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
            & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:])
    union = centers.copy()
    union[:-1] |= centers[1:]
    union[1:] |= centers[:-1]
    union[:, :-1] |= centers[:, 1:]
    union[:, 1:] |= centers[:, :-1]
    return centers, union


def _sequential(ink):
    remaining = ink.copy()
    erased = np.zeros_like(ink)
    for r in range(1, ink.shape[0] - 1):
        for c in range(1, ink.shape[1] - 1):
            points = [(r, c)] + [(r + dr, c + dc) for dr, dc in STEPS]
            if all(remaining[p] for p in points):
                for p in points:
                    remaining[p] = False
                    erased[p] = True
    return erased


def _perimeter(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((grid >= 0) & (grid <= 9))):
        raise ValueError('expected a 1..30 integer grid with colors 0..9')
    return grid


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _decode(input_grid):
    """Segment, identify the visible request, punch, count and print."""
    inp = _grid(input_grid)
    bg = int(inp[0, 0])
    assert np.all(_perimeter(inp) == bg), 'a blank outer margin identifies air'
    boxes = [_box(g) for g in _components(inp != bg)]
    assert 3 <= len(boxes) <= 6
    queries = [i for i, (_, _, h, w) in enumerate(boxes)
               if all((b[2], b[3]) == (h - 2, w - 2)
                      for j, b in enumerate(boxes) if i != j)]
    assert len(queries) == 1, 'one request is two cells taller and wider than every stencil'
    qi = queries[0]
    qr, qc, qh, qw = boxes[qi]
    request = inp[qr:qr + qh, qc:qc + qw]
    punch = int(request[0, 0])
    assert punch != bg and np.all(_perimeter(request) == punch)
    interior = request[1:-1, 1:-1]
    carriers = np.unique(interior[interior != bg])
    assert len(carriers) == 1
    carrier = int(carriers[0])
    assert carrier not in (bg, punch)
    clue = interior == bg
    assert not (np.any(clue[1:] & clue[:-1]) or np.any(clue[:, 1:] & clue[:, :-1]))
    wanted = int(clue.sum())
    for i, (r, c, h, w) in enumerate(boxes):
        for a, b, hh, ww in boxes[i + 1:]:
            assert r + h < a or a + hh < r or c + w < b or b + ww < c
    records = []
    for i, (r, c, h, w) in enumerate(boxes):
        if i == qi:
            continue
        plate = inp[r:r + h, c:c + w].copy()
        material = int(plate[0, 0])
        assert material not in (bg, punch, carrier)
        assert np.all(_perimeter(plate) == material)
        assert set(map(int, np.unique(plate))) <= {material, bg, punch}
        air, ink = plate == bg, plate == punch
        centers, union = _plus(ink)
        opened = air | union
        printed = plate.copy()
        printed[opened] = carrier
        records.append(dict(box=(r, c, h, w), plate=plate, printed=printed,
            raw=_count(air), count=_count(opened), area=int(opened.sum()),
            cavity_sizes=tuple(sorted(map(len, _components(opened)))),
            eight=_count(opened, True), centers=int(centers.sum()),
            erased=int(union.sum()), surviving_ink=int((ink & ~union).sum()),
            sequential=not np.array_equal(union, _sequential(ink)),
            ink=ink, air=air, union=union))
    counts = [a['count'] for a in records]
    assert len(set(counts)) == len(counts), 'resulting cavity counts must be distinct'
    assert wanted in counts
    selected = counts.index(wanted)
    answer = np.full(request.shape, punch, dtype=int)
    answer[1:-1, 1:-1] = records[selected]['printed']
    distances = [abs(2 * r + h - (2 * qr + qh)) + abs(2 * c + w - (2 * qc + qw))
                 for r, c, h, w in [a['box'] for a in records]]
    info = dict(background=bg, punch=punch, carrier=carrier,
        shape=(qh - 2, qw - 2), wanted=wanted, records=records,
        selected=selected, query_box=boxes[qi], nearest=distances.index(min(distances)))
    return answer, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the requested punched stencil, using visible input conventions only."""
    return _decode(input_grid)[0]


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _cross(ink, r, c):
    ink[r, c] = True
    for dr, dc in STEPS:
        ink[r + dr, c + dc] = True


def _embed(rng, air, ink, shape):
    # Random orientation of a local construction, then an independent offset.
    k = int(rng.integers(4))
    air, ink = np.rot90(air, k), np.rot90(ink, k)
    if rng.random() < .5:
        air, ink = air[:, ::-1], ink[:, ::-1]
    h, w = shape
    dh, dw = h - air.shape[0], w - air.shape[1]
    assert min(dh, dw) >= 0
    r, c = int(rng.integers(dh + 1)), int(rng.integers(dw + 1))
    a, p = np.zeros(shape, bool), np.zeros(shape, bool)
    a[r:r + air.shape[0], c:c + air.shape[1]] = air
    p[r:r + ink.shape[0], c:c + ink.shape[1]] = ink
    return a, p


def _grow(rng, air, ink, steps, preserve_raw=False):
    union = _plus(ink)[1]
    k, raw = _count(air | union), _count(air)
    h, w = air.shape
    for _ in range(steps):
        candidates = []
        opened = air | union
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if air[r, c] or ink[r, c] or not any(opened[r + dr, c + dc] for dr, dc in STEPS):
                    continue
                candidates.append((r, c))
        for j in rng.permutation(len(candidates)):
            point = candidates[int(j)]
            air[point] = True
            if (_count(air | union) == k and _count(~air) == 1
                    and (not preserve_raw or _count(air) == raw)):
                break
            air[point] = False
        else:
            break
    return air


def _witness_plate(rng, shape, number, decoy=False):
    """Small topological counterexamples, expanded by connected cavity growth."""
    air, ink = np.zeros((7, 7), bool), np.zeros((7, 7), bool)
    if decoy:
        _cross(ink, 3, 3)
        air[1, 1] = True
        if number == 3:
            air[2, 2] = air[4, 4] = True
    elif number == 1:
        _cross(ink, 3, 3)
        ink[3, 5] = True  # attached ink beyond the five-cell cross survives
        air[2, 2] = air[4, 4] = True
    else:
        ink[2:5, 2:6] = True  # two adjacent centers; eight punched cells
        # Both old holes touch the new region only diagonally: 2 -> 3 holes,
        # despite two plus centers. This refutes count-of-pluses and raw +/- 1.
        air[5, 2] = air[1, 5] = True
    air, ink = _embed(rng, air, ink, shape)
    # Do not grow decoys; the first scene then guarantees an area inversion.
    if not decoy:
        _grow(rng, air, ink, int(rng.integers(0, 8)), preserve_raw=True)
    return air, ink


def _random_plate(rng, shape, k):
    h, w = shape
    for attempt in range(80):
        air, ink = np.zeros(shape, bool), np.zeros(shape, bool)
        if k:
            mode = int(rng.integers(5))
            r, c = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            _cross(ink, r, c)
            if mode == 1:
                # Add an adjacent center, possibly sharing more than one arm.
                dc = 1 if c + 2 < w - 1 else -1
                _cross(ink, r, c + dc)
            elif mode == 2 and k >= 2:
                r2, c2 = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
                _cross(ink, r2, c2)
            elif mode == 3:
                # A wide ink block has a variable run of adjacent centers.
                length = int(rng.integers(3, w - 1))
                start = int(rng.integers(1, w - length))
                ink[r - 1:r + 2, start:start + length] = True
            elif mode == 4:
                for _ in range(int(rng.integers(2, 5))):
                    r2, c2 = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
                    _cross(ink, r2, c2)
            # Variable ink tails are not punched unless they complete more pluses.
            if rng.random() < .6:
                dr, dc = _choice(rng, STEPS)
                for d in range(2, int(rng.integers(3, max(h, w)))):
                    a, b = r + d * dr, c + d * dc
                    if not (1 <= a < h - 1 and 1 <= b < w - 1):
                        break
                    ink[a, b] = True
        else:
            # A rotated T is a near miss, so zero remains a genuine option.
            r, c = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            ink[r, c - 1:c + 2] = True
            ink[r + 1, c] = True
            ink = np.rot90(ink, 2) if rng.random() < .5 else ink
        union = _plus(ink)[1]
        if _count(union) > k:
            continue
        # Add independent air seeds; they need not use a checkerboard parity.
        for j in rng.permutation((h - 2) * (w - 2)):
            if _count(air | union) >= k:
                break
            r, c = 1 + int(j) // (w - 2), 1 + int(j) % (w - 2)
            if ink[r, c] or air[r, c]:
                continue
            air[r, c] = True
            if _count(air | union) > k or _count(~air) != 1:
                air[r, c] = False
        if _count(air | union) != k:
            continue
        _grow(rng, air, ink, int(rng.integers(0, (h - 2) * (w - 2) // 2 + 1)))
        # Add detached or attached non-cross camouflage without changing the rule.
        for _ in range(int(rng.integers(0, 5))):
            r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
            if air[r, c]:
                continue
            old = bool(ink[r, c])
            ink[r, c] = True
            if _count(air | _plus(ink)[1]) != k:
                ink[r, c] = old
        assert _count(air | _plus(ink)[1]) == k and not np.any(air & ink)
        return air, ink
    # Guaranteed sparse fallback, retaining connected material and exact count.
    air, ink = np.zeros(shape, bool), np.zeros(shape, bool)
    for r, c in [(r, c) for r in range(1, h - 1, 2) for c in range(1, w - 1, 2)][:k]:
        air[r, c] = True
    assert _count(air) == k and _count(~air) == 1
    return air, ink


def _clues(rng, shape, k):
    h, w = shape
    for _ in range(20):
        mask = np.zeros(shape, bool)
        for j in rng.permutation(h * w):
            if int(mask.sum()) == k:
                return mask
            r, c = divmod(int(j), w)
            if not any(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                       for dr, dc in STEPS):
                mask[r, c] = True
    raise AssertionError('clue capacity exceeded')


def _free(boxes, shape):
    h, w = shape
    rr, cc = np.indices((29 - h, 29 - w))
    rr, cc = rr + 1, cc + 1
    free = np.ones(rr.shape, bool)
    for r, c, bh, bw in boxes:
        free &= (rr + h < r) | (r + bh < rr) | (cc + w < c) | (c + bw < cc)
    return [(int(r), int(c)) for r, c in zip(rr[free], cc[free])]


def _fallback_boxes(h, w, n):
    transpose = (n >= 4 and 3 * w + 4 > 28)
    if transpose:
        h, w = w, h
    first = 1 if n <= 3 else 2
    boxes = [(1, 1, h + 2, w + 2)]
    boxes += [(2 if j == 0 else 1, w + 4 + j * (w + 1), h, w) for j in range(first)]
    boxes += [(h + 4, 1 + j * (w + 1), h, w) for j in range(n - first)]
    if transpose:
        boxes = [(c, r, bw, bh) for r, c, bh, bw in boxes]
    assert max(r + bh for r, c, bh, bw in boxes) <= 29
    assert max(c + bw for r, c, bh, bw in boxes) <= 29
    return boxes


def _place(rng, bg, query, plates, target, refute_nearest=False):
    h, w = plates[0].shape
    n, boxes = len(plates), None
    for _ in range(24):
        r, c = _choice(rng, _free([], query.shape))
        trial = [(r, c, *query.shape)]
        for _ in plates:
            free = _free(trial, (h, w))
            if not free:
                break
            a, b = _choice(rng, free)
            trial.append((a, b, h, w))
        if len(trial) == n + 1:
            boxes = trial
            break
    if boxes is None:
        boxes = _fallback_boxes(h, w, n)
    # Reflect positions only. The visible stencil orientation is never remapped.
    if rng.random() < .5:
        boxes = [(30 - r - bh, c, bh, bw) for r, c, bh, bw in boxes]
    if rng.random() < .5:
        boxes = [(r, 30 - c - bw, bh, bw) for r, c, bh, bw in boxes]
    order = list(map(int, rng.permutation(n)))
    if refute_nearest:
        qr, qc, qh, qw = boxes[0]
        ds = [abs(2 * r + bh - 2 * qr - qh) + abs(2 * c + bw - 2 * qc - qw)
              for r, c, bh, bw in boxes[1:]]
        # Put a decoy at every minimum-distance position, including ties.
        eligible = [i for i, d in enumerate(ds) if d > min(ds)]
        if not eligible:
            boxes = _fallback_boxes(h, w, n)
            qr, qc, qh, qw = boxes[0]
            ds = [abs(2 * r + bh - 2 * qr - qh) + abs(2 * c + bw - 2 * qc - qw)
                  for r, c, bh, bw in boxes[1:]]
            eligible = [i for i, d in enumerate(ds) if d > min(ds)]
        if eligible:
            slot = _choice(rng, eligible)
            old = order.index(target)
            order[slot], order[old] = order[old], order[slot]
    canvas = np.full((30, 30), bg, dtype=int)
    for (r, c, bh, bw), obj in zip(boxes, [query] + [plates[i] for i in order]):
        canvas[r:r + bh, c:c + bw] = obj
    # Keep at least a one-cell blank margin; larger margins are independent.
    lo_r, lo_c = min(b[0] for b in boxes), min(b[1] for b in boxes)
    hi_r, hi_c = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    top, left = int(rng.integers(lo_r)), int(rng.integers(lo_c))
    bottom, right = int(rng.integers(hi_r + 1, 31)), int(rng.integers(hi_c + 1, 31))
    return canvas[top:bottom, left:right].copy()


def _make_pair(rng, n, witness=None):
    if n >= 4:
        h, w = int(rng.integers(7, 12)), int(rng.integers(7, 9))
        if rng.random() < .5:
            h, w = w, h
    else:
        h, w = map(int, rng.integers(7, 12, size=2))
    shape = (h, w)
    bg, punch, carrier = map(int, rng.choice(10, 3, replace=False))
    materials = [v for v in range(10) if v not in (bg, punch, carrier)]
    masks = []
    if witness:
        wanted = witness
        masks = [_witness_plate(rng, shape, witness), _witness_plate(rng, shape, witness, True)]
        counts = [witness, 2]
        additional = list(map(int, rng.choice([k for k in range(6) if k not in counts], n - 2, replace=False)))
    else:
        counts = list(map(int, rng.choice(6, n, replace=False)))
        wanted = _choice(rng, counts)
        additional = counts.copy()
    for k in additional:
        masks.append(_random_plate(rng, shape, k))
    actual = [_count(a | _plus(p)[1]) for a, p in masks]
    if witness:
        assert actual[:2] == counts
        wanted = witness
    target = actual.index(wanted)
    plates = []
    for air, ink in masks:
        plate = np.full(shape, _choice(rng, materials), dtype=int)
        plate[air], plate[ink] = bg, punch
        plates.append(plate)
    query = np.full((h + 2, w + 2), punch, dtype=int)
    query[1:-1, 1:-1] = carrier
    query[1:-1, 1:-1][_clues(rng, shape, wanted)] = bg
    inp = _place(rng, bg, query, plates, target, refute_nearest=(witness == 1))
    # Construct the label directly from the selected masks, not from solve().
    out = query.copy()
    opened = masks[target][0] | _plus(masks[target][1])[1]
    out[1:-1, 1:-1] = np.where(opened, carrier, plates[target])
    return {'input': inp, 'output': out}


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in (('train', nt), ('test', ne)):
        for i in range(total):
            witness = (1, 3)[i] if split == 'train' and i < 2 else None
            n = int(rng.integers(2, 4)) if split == 'train' else int(rng.integers(4, 6))
            for _ in range(40):
                pair = _make_pair(rng, n, witness)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded uniqueness attempts exhausted')
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _coverage(analyses):
    selected = [a['records'][a['selected']] for a in analyses]
    assert len({a['wanted'] for a in analyses}) >= 2, 'vary the requested number'
    assert any(a['raw'] != a['count'] and a['erased'] for a in selected)
    assert any(a['raw'] < a['count'] for a in selected), 'demonstrate creation of a new cavity'
    assert any(a['raw'] > a['count'] for a in selected), 'demonstrate merging of old cavities'
    assert any(a['centers'] != a['count'] for a in selected), 'refute plus-center counting'
    assert any(a['sequential'] for a in selected), 'demonstrate simultaneous overlaps'
    assert any(a['eight'] < a['count'] for a in selected), 'demonstrate four-connectivity'
    assert any(a['surviving_ink'] and a['erased'] for a in selected), 'preserve ink outside pluses'
    assert any(a['selected'] != a['nearest'] for a in analyses), 'refute nearest-stencil selection'
    assert any(any(b['count'] > t['count'] and b['area'] < t['area'] for b in a['records'])
               for a in analyses for t in [a['records'][a['selected']]]), 'refute hole-area ranking'
    assert any(any(b['raw'] == a['wanted'] and i != a['selected']
                       for i, b in enumerate(a['records'])) for a in analyses), 'refute old-hole matching'


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted, a = _decode(inp)
            assert np.array_equal(predicted, out), 'wrong selection, punch, orientation or substitution'
            # Direct cell checks state the local invariant separately from dilation.
            for record in a['records']:
                ink, direct = record['ink'], np.zeros_like(record['ink'])
                for r in range(1, ink.shape[0] - 1):
                    for c in range(1, ink.shape[1] - 1):
                        points = [(r, c)] + [(r + dr, c + dc) for dr, dc in STEPS]
                        if all(ink[p] for p in points):
                            for point in points:
                                direct[point] = True
                assert np.array_equal(direct, record['union'])
                assert _count(~record['air']) == 1, 'input stencil remains segmentable'
            if split == 'train':
                training.append(a)
    _coverage(training)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for p in puzzle[split]:
            for role in ('input', 'output'):
                g = np.asarray(p[role], dtype=np.uint8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """42 supplied seeds, bounded in memory; no generated artifacts or logs."""
    signatures, shapes, structures, plate_shapes = set(), set(), set(), set()
    raw_post, sizes, center_counts, backgrounds, palettes, numbers = set(), set(), set(), set(), set(), set()
    total, overlap, connectivity, merging, zero, multi_punch = 0, 0, 0, 0, 0, 0
    for seed in VERIFICATION_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        analyses = {s: [] for s in ('train', 'test')}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                predicted, a = _decode(pair['input'].copy())
                assert np.array_equal(predicted, pair['output'])
                analyses[split].append(a)
                g = pair['input']
                shapes.add(g.shape)
                roles = {}
                structures.add(bytes(roles.setdefault(int(v), len(roles)) for v in g.flat))
                plate_shapes.add(a['shape'])
                backgrounds.add(a['background'])
                palettes.add(len(np.unique(g)))
                numbers.add(len(a['records']))
                for r in a['records']:
                    total += 1
                    raw_post.add((r['raw'], r['count']))
                    sizes.update(r['cavity_sizes'])
                    center_counts.add(r['centers'])
                    overlap += r['sequential']
                    connectivity += r['eight'] < r['count']
                    merging += r['raw'] > r['count']
                    zero += r['count'] == 0
                    multi_punch += _count(r['union']) > 1
        assert max(len(a['records']) for a in analyses['train']) < min(len(a['records']) for a in analyses['test'])
        if seed in VERIFICATION_SEEDS[:3]:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
            for pair in puzzle['test']:
                permutation = np.random.default_rng(seed + 1).permutation(10)
                assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
                assert np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output']))
    assert len(signatures) == 42 and len(shapes) >= 25 and len(structures) >= 200
    assert len(plate_shapes) >= 15 and len(backgrounds) == 10 and len(palettes) >= 3
    assert numbers == {2, 3, 4, 5} and min(sizes) == 1 and max(sizes) >= 15
    assert len(raw_post) >= 12 and len(center_counts) >= 4
    assert min(overlap, connectivity, merging, zero, multi_punch) >= 3
    # Exhaustive local recognition and a destructive-scanning counterexample.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], bool).reshape(3, 3)
        assert bool(_plus(mask)[0][1, 1]) == all(mask[p] for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)))
    block = np.ones((3, 4), bool)
    assert _plus(block)[1].sum() == 8 and _sequential(block).sum() == 5
    for a, b in ((2, 1), (10, 1), (10, 4)):
        generate_puzzle(9901, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(0, 2, 1)
    bad['test'][0]['output'][1, 1] = (int(bad['test'][0]['output'][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a corrupted print')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), input_shapes=len(shapes),
        normalized_inputs=len(structures), stencil_shapes=len(plate_shapes), stencils=total,
        raw_to_final_counts=sorted(raw_post), cavity_area_range=[min(sizes), max(sizes)],
        center_counts=sorted(center_counts), palette_sizes=sorted(palettes),
        overlap_stencils=overlap, connectivity_witnesses=connectivity,
        merging_stencils=merging, zero_stencils=zero, multiple_punch_regions=multi_punch,
        independent_solver=False, blind_prediction=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
