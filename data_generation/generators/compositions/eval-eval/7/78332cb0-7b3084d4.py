"""Broken trail mosaic: jointly assemble rigid material and continuous ink.

Only the standard library and NumPy are used. The solver reads no examples,
random state, files, or construction metadata. Search is exhaustive; generation
uses bounded rejection to select uniquely solvable, instructive inputs.
"""

from collections import Counter
from math import isqrt
import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Integer grids required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "Colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, neighbors=EDGE):
    todo = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _tight(grid, cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    r1, c1 = max(r for r, c in cells), max(c for r, c in cells)
    p = np.full((r1-r0+1, c1-c0+1), -1, dtype=np.int8)
    for r, c in cells:
        p[r-r0, c-c0] = grid[r, c]
    return p


def _read(a):
    a = _grid(a)
    bg = int(a[0, 0])
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "The complete outer border identifies background")
    groups = _components(a != bg, EIGHT)
    _require(3 <= len(groups) <= 5, "Three to five separate material pieces required")
    pieces = [_tight(a, group) for group in groups]
    counts = Counter(map(int, a[a != bg]))
    single = [c for c, n in counts.items() if n == 1]
    _require(len(single) == 1, "One globally singleton anchor color required")
    marker = single[0]
    common = set.intersection(*(set(map(int, p[p >= 0])) for p in pieces))
    common.discard(marker)
    _require(len(common) == 1, "Ink is the unique color present in every piece")
    ink = common.pop()
    carriers = []
    for p in pieces:
        carrier = set(map(int, p[p >= 0])) - {ink, marker}
        _require(len(carrier) == 1, "One carrier identity per material piece")
        carrier = carrier.pop()
        _require(np.sum(p == carrier) >= 2, "Carriers cannot be singleton markers")
        _require(len(_components(p >= 0, EIGHT)) == 1, "Material must be connected")
        carriers.append(carrier)
    _require(len(set(carriers)) == len(pieces), "Carrier identities must be distinct")
    area = sum(len(g) for g in groups)
    side = isqrt(area)
    _require(side*side == area and 5 <= side <= 9, "Material area determines a square of side 5..9")
    return dict(grid=a, bg=bg, marker=marker, ink=ink, pieces=pieces,
                side=side, carriers=carriers)


def _bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length()-1
        mask ^= bit


def _neighbor_masks(side):
    return [sum(1 << (rr*side+cc) for dr, dc in EDGE
                if 0 <= (rr := r+dr) < side and 0 <= (cc := c+dc) < side)
            for r in range(side) for c in range(side)]


def _path_possible(signal, occupied, neighbors, complete=False):
    """Prune forks, cycles, and already sealed excess endpoints.

    Empty material positions may still supply future ink. Consequently partial
    disconnected fragments and partial endpoints are not rejected prematurely.
    """
    cells = list(_bits(signal))
    if not cells or not signal & 1:
        return False
    degrees = {}
    forced = 0
    for p in cells:
        degree = (neighbors[p] & signal).bit_count()
        if degree > (1 if p == 0 else 2):
            return False
        degrees[p] = degree
        free = bool(neighbors[p] & ~occupied)
        if degree == 0 and not free:
            return False
        if p != 0 and degree == 1 and not free:
            forced += 1
    if forced > 1:
        return False
    # Any cycle in a graph of maximum degree two is permanently sealed.
    remaining = set(cells)
    components = 0
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, vertices, total_degree = [start], 0, 0
        while stack:
            p = stack.pop()
            vertices += 1
            total_degree += degrees[p]
            for q in _bits(neighbors[p] & signal):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        if total_degree == 2*vertices:
            return False
        components += 1
    if complete:
        return (components == 1 and degrees[0] == 1 and
                sum(d == 1 for d in degrees.values()) == 2 and
                all(d in (1, 2) for d in degrees.values()))
    return True


def _candidates(piece, side, marker, ink):
    result, seen = [], set()
    marked = bool(np.any(piece == marker))
    for k in range(4):
        p = np.rot90(piece, k)
        h, w = p.shape
        if h > side or w > side:
            continue
        if marked and int(p[0, 0]) != marker:
            continue
        local = [(int(r)*side+int(c), int(p[r, c])) for r, c in np.argwhere(p >= 0)]
        material = sum(1 << pos for pos, color in local)
        signal = sum(1 << pos for pos, color in local if color in (ink, marker))
        offsets = [0] if marked else [r*side+c for r in range(side-h+1) for c in range(side-w+1)]
        for offset in offsets:
            key = material << offset, signal << offset
            if key in seen:
                continue
            seen.add(key)
            result.append(dict(mask=key[0], signal=key[1], k=k,
                               cells=tuple((pos+offset, color) for pos, color in local)))
    return result


def _assemblies(facts, respect_trail=True, limit=2, allow_rotations=True):
    """All rigid exact covers; stop after `limit` distinct colored answers.

    Occupancy alone is an explicit ablation, still preserving the printed ink.
    Candidate deduplication includes ink geometry, not just material silhouettes.
    """
    side, pieces = facts['side'], facts['pieces']
    choices = [_candidates(p, side, facts['marker'], facts['ink']) for p in pieces]
    if not allow_rotations:
        choices = [[p for p in options if p['k'] == 0] for options in choices]
    if any(not p for p in choices):
        return []
    anchor = next(i for i, p in enumerate(pieces) if np.any(p == facts['marker']))
    full = (1 << (side*side))-1
    neighbors = _neighbor_masks(side)
    answers, keys = [], set()
    placed = [None]*len(pieces)

    def search(occupied, signal, remaining):
        if len(answers) >= limit:
            return
        if respect_trail and not _path_possible(signal, occupied, neighbors, not remaining):
            return
        if not remaining:
            if occupied != full:
                return
            out = np.empty((side, side), dtype=np.int8)
            for p in placed:
                for pos, color in p['cells']:
                    out.flat[pos] = color
            key = out.tobytes()
            if key not in keys:
                keys.add(key)
                answers.append(dict(grid=out, placements=tuple(placed)))
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for p in choices[i]:
                if p['mask'] & first and not p['mask'] & occupied:
                    placed[i] = p
                    search(occupied | p['mask'], signal | p['signal'], rest)
                    if len(answers) >= limit:
                        return
            placed[i] = None

    rest = tuple(i for i in range(len(pieces)) if i != anchor)
    for p in choices[anchor]:
        placed[anchor] = p
        search(p['mask'], p['signal'], rest)
        if len(answers) >= limit:
            break
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _read(input_grid)
    answers = _assemblies(facts)
    _require(len(answers) == 1, "Exactly one colored square must satisfy material and trail")
    return answers[0]['grid'].copy()


def _material_witness(facts, answer):
    """Find a single-piece move that keeps the trail but breaks material tiling.

    This is a concrete ablation of B: retain the ink geometry and all rigid
    quarter-turn transformations, but disregard carrier overlap and holes.
    """
    placed = answer['placements']
    full = (1 << (facts['side']**2))-1
    trail = 0
    for p in placed:
        trail |= p['signal']
    neighbors = _neighbor_masks(facts['side'])
    for i, current in enumerate(placed):
        other_trail = trail ^ current['signal']
        other_material = full ^ current['mask']
        for p in _candidates(facts['pieces'][i], facts['side'], facts['marker'], facts['ink']):
            if p['signal'] & other_trail:
                continue
            overlaps = (p['mask'] & other_material).bit_count()
            holes = (full ^ (p['mask'] | other_material)).bit_count()
            if (overlaps or holes) and _path_possible(p['signal'] | other_trail, full, neighbors, True):
                return dict(piece=i, rotation=p['k'], overlaps=overlaps, holes=holes)
    return None


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _make_trail(rng, side):
    """Induced self-avoiding walk, rooted at the square's upper-left."""
    for _ in range(100):
        path = [(0, 0)]
        used = set(path)
        target = int(rng.integers(side+4, min(side*side, 4*side)+1))
        while len(path) < target:
            r, c = path[-1]
            options = []
            for dr, dc in EDGE:
                p = r+dr, c+dc
                if not (0 <= p[0] < side and 0 <= p[1] < side) or p in used:
                    continue
                contacts = sum((p[0]+er, p[1]+ec) in used for er, ec in EDGE)
                if contacts == 1:
                    options.append(p)
            if not options:
                break
            p = _choose(rng, options)
            path.append(p)
            used.add(p)
        if len(path) >= side+4:
            return path
    raise RuntimeError("Could not grow an induced trail")


def _grow_labels(rng, side, count, path):
    indices = [0] + sorted(map(int, rng.choice(np.arange(1, len(path)), count-1, replace=False)))
    labels = np.full((side, side), -1, dtype=np.int8)
    frontier, sizes = [set() for _ in range(count)], [1]*count
    weights = np.exp(rng.uniform(-1, 1, count))
    for i, index in enumerate(indices):
        labels[path[index]] = i
    for i, index in enumerate(indices):
        r, c = path[index]
        frontier[i].update((r+dr, c+dc) for dr, dc in EDGE
                           if 0 <= r+dr < side and 0 <= c+dc < side and labels[r+dr, c+dc] < 0)
    for _ in range(side*side-count):
        eligible = [i for i in range(count) if frontier[i]]
        i = min(eligible, key=lambda j: sizes[j]/weights[j]+float(rng.random()))
        p = _choose(rng, sorted(frontier[i]))
        labels[p] = i
        sizes[i] += 1
        for f in frontier:
            f.discard(p)
        r, c = p
        frontier[i].update((r+dr, c+dc) for dr, dc in EDGE
                           if 0 <= r+dr < side and 0 <= c+dc < side and labels[r+dr, c+dc] < 0)
    return labels


def _cut_labels(rng, side, count):
    rectangles = [(0, 0, side, side)]
    while len(rectangles) < count:
        options = []
        for i, (r, c, h, w) in enumerate(rectangles):
            options.extend((i, 0, k) for k in range(1, h) if min(k, h-k)*w >= 4)
            options.extend((i, 1, k) for k in range(1, w) if min(k, w-k)*h >= 4)
        if not options:
            return None
        i, axis, k = _choose(rng, options)
        r, c, h, w = rectangles.pop(i)
        rectangles.extend([(r, c, k, w), (r+k, c, h-k, w)] if axis == 0 else
                          [(r, c, h, k), (r, c+k, h, w-k)])
    labels = np.empty((side, side), dtype=np.int8)
    for i, (r, c, h, w) in enumerate(rectangles):
        labels[r:r+h, c:c+w] = i
    # Local transfers give arbitrary steps, concavities, and thin protrusions.
    for _ in range(int(rng.integers(side*2+1))):
        r, c = map(int, rng.integers(side, size=2))
        dr, dc = _choose(rng, EDGE)
        rr, cc = r+dr, c+dc
        if not (0 <= rr < side and 0 <= cc < side) or (r, c) == (0, 0):
            continue
        old, new = int(labels[r, c]), int(labels[rr, cc])
        if old == new or np.sum(labels == old) <= 4:
            continue
        labels[r, c] = new
        if len(_components(labels == old)) != 1:
            labels[r, c] = old
    return labels


def _scatter(rng, pieces, bg, anchor_rotation, marker):
    rotations = list(map(int, rng.integers(4, size=len(pieces))))
    anchor = next(i for i, p in enumerate(pieces) if np.any(p == marker))
    rotations[anchor] = anchor_rotation
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
    size = 28  # Reserve at least one background cell on every outer side.
    full = (1 << (size*size))-1
    firstcol = sum(1 << (r*size) for r in range(size))
    lastcol = firstcol << (size-1)
    for attempt in range(16):
        canvas = np.full((size, size), -1, dtype=np.int8)
        blocked, maxr, maxc = 0, 0, 0
        aspect = float(rng.uniform(.7, 1.4))
        order = (sorted(range(len(pieces)), key=lambda i: -np.sum(oriented[i] >= 0))
                 if attempt == 0 else list(map(int, rng.permutation(len(pieces)))))
        okay = True
        for i in order:
            p = oriented[i]
            h, w = p.shape
            base = sum(1 << (int(r)*size+int(c)) for r, c in np.argwhere(p >= 0))
            options = []
            for r in range(size-h+1):
                for c in range(size-w+1):
                    mask = base << (r*size+c)
                    if mask & blocked:
                        continue
                    hr, wc = max(maxr, r+h), max(maxc, c+w)
                    score = hr*wc + abs(hr-aspect*wc)*2 + float(rng.random())*35
                    options.append((score, r, c, mask))
            if not options:
                okay = False
                break
            _, r, c, mask = min(options)
            horizontal = mask | ((mask & ~firstcol) >> 1) | ((mask & ~lastcol) << 1)
            blocked |= (horizontal | (horizontal << size) | (horizontal >> size)) & full
            view = canvas[r:r+h, c:c+w]
            view[p >= 0] = p[p >= 0]
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if okay:
            points = np.argwhere(canvas >= 0)
            lo, hi = points.min(axis=0), points.max(axis=0)+1
            crop = canvas[lo[0]:hi[0], lo[1]:hi[1]]
            extras = [int(rng.integers(1, min(4, 29-d)+1)) for d in crop.shape]
            H, W = (d+e+1 for d, e in zip(crop.shape, extras))
            top, left = int(rng.integers(1, extras[0]+1)), int(rng.integers(1, extras[1]+1))
            result = np.full((H, W), bg, dtype=np.int8)
            result[top:top+crop.shape[0], left:left+crop.shape[1]] = np.where(crop >= 0, crop, bg)
            return result
    return None


def _pair(rng, side, count, rotation, require_multiple, require_material, previous_roles):
    # Palette is visible locally, with role changes in the demonstrations.
    palette = list(map(int, rng.permutation(10)))
    if previous_roles:
        palette.remove(previous_roles[0])
        palette.append(previous_roles[0])
    bg, ink, marker = palette[:3]
    carriers = palette[3:3+count]
    for attempt in range(400):
        path = _make_trail(rng, side)
        labels = (_grow_labels(rng, side, count, path) if rng.random() < .4
                  else _cut_labels(rng, side, count))
        if labels is None:
            continue
        signal = np.zeros((side, side), bool)
        for p in path:
            signal[p] = True
        if any(np.sum((labels == i) & signal) < 1 or
               np.sum((labels == i) & ~signal) < 2 for i in range(count)):
            continue
        if np.sum((labels == labels[0, 0]) & signal) < 2:
            # The anchor replaces one ink cell; the marked object must still
            # expose the common ink color, so all roles remain input-derived.
            continue
        out = np.empty((side, side), dtype=np.int8)
        for i in range(count):
            out[labels == i] = carriers[i]
        out[signal] = ink
        out[0, 0] = marker
        pieces = [_tight(out, set(map(tuple, np.argwhere(labels == i)))) for i in range(count)]
        fragment_counts = [len(_components((p == ink) | (p == marker))) for p in pieces]
        if require_multiple and max(fragment_counts) < 2:
            continue
        facts = dict(side=side, marker=marker, ink=ink, pieces=pieces)
        joint = _assemblies(facts)
        if len(joint) != 1 or not np.array_equal(joint[0]['grid'], out):
            continue
        if require_material and _material_witness(facts, joint[0]) is None:
            continue
        # Both the training and the test retain a real need for port continuity.
        if len(_assemblies(facts, respect_trail=False)) < 2:
            continue
        inp = _scatter(rng, pieces, bg, rotation, marker)
        if inp is None:
            continue
        return dict(input=inp, output=out), (bg, ink, marker)
    raise RuntimeError("Bounded search did not find an instructive mosaic")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen, roles = dict(train=[], test=[]), set(), None
    train_rotations = [0, _choose(rng, [1, 3])] + list(map(int, rng.choice([0, 1, 3], ntrain-2)))
    rng.shuffle(train_rotations)
    for split, n in (('train', ntrain), ('test', ntest)):
        for index in range(n):
            side = int(rng.integers(5, 7) if split == 'train' else rng.integers(7, 10))
            count = int(rng.integers(3, 5) if split == 'train' else
                        rng.integers(4, 5 if side == 9 else 6))
            if split == 'train' and index < 2:
                count = 3+index  # Demonstrate that all supplied pieces are used.
            rotation = train_rotations[index] if split == 'train' else 2
            require_multiple = split == 'train' and index == 1
            for _ in range(20):
                pair, new_roles = _pair(rng, side, count, rotation, require_multiple,
                                        split == 'train' and index == 0, roles)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("Could not make a distinct input")
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
            roles = new_roles
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    facts = _read(inp)
    side = facts['side']
    _require(out.shape == (side, side), "Output dimensions follow material area")
    _require(out[0, 0] == facts['marker'], "Anchor must be upper-left")
    _require(not np.any(out == facts['bg']), "Remove surrounding background only")
    for color in range(10):
        if color != facts['bg']:
            _require(np.sum(inp == color) == np.sum(out == color), "Conserve every printed material cell")
    joint = _assemblies(facts)
    _require(len(joint) == 1 and np.array_equal(joint[0]['grid'], out), "Unique joint rigid assembly required")
    geometric = _assemblies(facts, respect_trail=False)
    _require(len(geometric) == 2, "Ignoring trail continuity must admit a different colored answer")
    pieces = facts['pieces']
    ink, marker = facts['ink'], facts['marker']
    fragment_counts = [len(_components((p == ink) | (p == marker))) for p in pieces]
    signal = sum(1 << (int(r)*side+int(c)) for r, c in np.argwhere((out == ink) | (out == marker)))
    neighbors = _neighbor_masks(side)
    _require(_path_possible(signal, (1 << (side*side))-1, neighbors, True), "One complete induced trail")
    turns = sum((neighbors[p] & signal).bit_count() == 2 and
                len({q//side for q in _bits(neighbors[p] & signal)}) == 2 and
                len({q % side for q in _bits(neighbors[p] & signal)}) == 2 for p in _bits(signal))
    areas = [int(np.sum(p >= 0)) for p in pieces]
    concave = sum(bool(np.any((m := np.pad((p >= 0).astype(int), 1))[:-1, :-1] +
                             m[:-1, 1:] + m[1:, :-1] + m[1:, 1:] == 3)) for p in pieces)
    return dict(side=side, count=len(pieces), bg=facts['bg'], ink=ink, marker=marker,
                fragments=tuple(fragment_counts), rotation=any(p['k'] != 0 for p in joint[0]['placements']),
                material_witness=_material_witness(facts, joint[0]),
                area_ratio=max(areas)/min(areas), concave=concave, turns=turns,
                trail_length=signal.bit_count(), input_shape=inp.shape,
                masks=tuple(_key((p >= 0).astype(int)) for p in pieces))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Exact train/test schema required")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal pair counts")
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Exact pair schema required")
            key = _key(_grid(pair['input']))
            _require(key not in seen, "Inputs must be distinct")
            seen.add(key)
            records[split].append(_analyze(pair))
    train, test = records['train'], records['test']
    _require(any(max(r['fragments']) > 1 for r in train), "Teach disconnected ink on one rigid piece")
    _require(any(r['rotation'] for r in train), "Teach quarter-turn motion")
    _require(any(r['material_witness'] for r in train), "Teach exact material coverage, beyond joining ink")
    _require({3, 4} <= {r['count'] for r in train}, "Teach variable counts before larger tests")
    _require(len({r['bg'] for r in train}) > 1, "Teach locally identifiable palette roles")
    _require(min(r['side'] for r in test) > max(r['side'] for r in train), "Generated tests increase scale")
    return records


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed audit, input-only invariance checks, and semantic negatives."""
    records, puzzles, masks, shapes = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        facts = validate_generated_puzzle(puzzle)
        puzzles.append(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            records.extend(facts[split])
            for pair, r in zip(puzzle[split], facts[split]):
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), "Input-only solve mismatch")
                masks.update(r['masks'])
                shapes.add(r['input_shape'])
    _require(len(set(puzzles)) == 42, "Seeds must produce different puzzles")
    _require(_puzzle_key(generate_puzzle(17)) == puzzles[17], "Repeatability")
    _require({r['count'] for r in records} == {3, 4, 5}, "Vary material counts")
    _require({r['side'] for r in records} == {5, 6, 7, 8, 9}, "Vary square scale")
    _require(len(shapes) >= 30 and len(masks) >= 100, "Vary scattering and silhouettes")
    _require(len({r['trail_length'] for r in records}) >= 10, "Vary trail length")
    _require(sum(r['concave'] > 0 for r in records) >= 25, "Include concave material")
    _require(sum(max(r['fragments']) > 1 for r in records) >= 42, "Multiple ink fragments must be real")
    _require(any(r['area_ratio'] >= 4 for r in records), "Unequal piece sizes")
    # Move the scattered whole objects again: location cannot encode the answer.
    p = generate_puzzle(101, 2, 1)
    pair = p['train'][0]
    f = _read(pair['input'])
    moved = _scatter(np.random.default_rng(102), f['pieces'], f['bg'], 3, f['marker'])
    _require(moved is not None and np.array_equal(solve(moved), pair['output']), "Whole-piece rescattering invariance")
    mapping = np.array([7, 3, 9, 0, 2, 6, 1, 8, 5, 4], dtype=np.int8)
    _require(np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']]), "Visible role recoloring invariance")
    for counts in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(500+counts[0]+counts[1], *counts))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = int(bad['test'][0]['output'][0, 1])
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Damaged anchor accepted")
    neighbors = _neighbor_masks(5)
    full = (1 << 25)-1
    # Anchor plus a disconnected four-cell cycle, a fork, and a broken trail.
    for points in ((0, 1, 12, 13, 17, 18), (0, 1, 2, 6), (0, 1, 23, 24)):
        _require(not _path_possible(sum(1 << q for q in points), full, neighbors, True),
                 "Reject cycles, forks, and disconnected ink")
    return dict(seeds=42, pairs=len(records), input_shapes=len(shapes), material_masks=len(masks),
                sides=sorted({r['side'] for r in records}), counts=sorted({r['count'] for r in records}),
                multi_fragment_pairs=sum(max(r['fragments']) > 1 for r in records),
                concave_pairs=sum(r['concave'] > 0 for r in records))


if __name__ == '__main__':
    print(self_test())
