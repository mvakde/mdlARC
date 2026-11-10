"""Boundary stencil census: hub-directed rigid masks select a boundary mural.

Only standard-library Python and NumPy are used. No generation metadata is
needed by solve; gray rings visibly key the selector color in every input.
"""
from collections import Counter
import json
import numpy as np

RING = 5
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        start = int(r), int(c)
        if seen[start]:
            continue
        seen[start] = True
        stack, cells = [start], set()
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(cells)
    return result


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells) + 1, max(c for r, c in cells) + 1)


def _rectangle(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _adjacent(a, b):
    return any((r + dr, c + dc) in b for r, c in a for dr, dc in DIRS)


def _shift(cells, direction, shape):
    r0, c0, r1, c1 = _box(cells)
    dr, dc = ((-r0, 0), (shape[0] - r1, 0),
              (0, -c0), (0, shape[1] - c1))[direction]
    return {(r + dr, c + dc) for r, c in cells}


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError('Expected an integer ARC grid of size 1..30 and colors 0..9')
    return a.astype(np.int8, copy=False)


def _analyze(input_grid):
    """Parse all roles and compute scores using input pixels only."""
    a = _grid(input_grid)
    h, w = a.shape
    bg = int(a[0, 0])
    assert bg != RING and all(int(a[r, c]) == bg for r, c in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)))
    panels, panel_cells = [], set()
    for cells in _components(a != bg):
        if any(r in (0, h-1) or c in (0, w-1) for r, c in cells):
            box = _box(cells)
            assert cells == _rectangle(box), 'Boundary receivers must be filled rectangles'
            r0, c0, r1, c1 = box
            assert min(r1-r0, c1-c0) >= 2
            assert sum((r0 == 0, r1 == h, c0 == 0, c1 == w)) == 1
            assert not any(a[r, c] == RING for r, c in cells)
            panels.append({'cells': cells, 'box': box, 'panel': a[r0:r1, c0:c1].copy()})
            panel_cells |= cells
    assert 2 <= len(panels) <= 4
    sides = [next(i for i, v in enumerate((p['box'][0] == 0, p['box'][2] == h,
                                            p['box'][1] == 0, p['box'][3] == w)) if v) for p in panels]
    assert len(set(sides)) == len(sides), 'At most one receiver per side'
    hubs, hub_cells, selectors = [], set(), set()
    labels = {}
    for ring in _components(a == RING):
        r0, c0, r1, c1 = _box(ring)
        assert r1-r0 == c1-c0 == 3 and 0 < r0 and r1 < h and 0 < c0 and c1 < w
        center = (r0+1, c0+1)
        full = _rectangle((r0, c0, r1, c1))
        assert ring == full - {center}, 'A hub is an eight-cell gray ring'
        selector = int(a[center])
        assert selector not in (bg, RING)
        selectors.add(selector)
        hub_id = len(hubs)
        for cell in ring:
            labels[cell] = hub_id
        hubs.append(full)
        hub_cells |= full
    assert 2 <= len(hubs) <= 4 and len(selectors) == 1
    selector = next(iter(selectors))
    remaining = a != bg
    for r, c in panel_cells | hub_cells:
        remaining[r, c] = False
    probes = []
    for cells in _components(remaining):
        colors = {int(a[r, c]) for r, c in cells}
        assert len(colors) == 1 and not colors.intersection((bg, RING, selector))
        contacts = set()
        for r, c in cells:
            # The neighbor behind a probe determines its outward normal.
            for d, (dr, dc) in enumerate(DIRS):
                behind = (r-dr, c-dc)
                if behind in labels:
                    contacts.add((labels[behind], d))
        assert len(contacts) == 1, 'Each stencil has a unique hub and contact normal'
        hub, direction = next(iter(contacts))
        dest = _shift(cells, direction, a.shape)
        probes.append({'cells': cells, 'dest': dest, 'hub': hub, 'direction': direction})
    assert 3 <= len(probes) <= 8
    assert set(p['hub'] for p in probes) == set(range(len(hubs)))
    multiplicity = Counter(cell for p in probes for cell in p['dest'])
    union = set(multiplicity)
    bbox_union = set().union(*(_shift(_rectangle(_box(p['cells'])), p['direction'], a.shape) for p in probes))
    for p in panels:
        marked = {cell for cell in p['cells'] if a[cell] == selector}
        p['marked'] = marked
        p['score'] = len(marked & union)
        p['raw'] = len(marked)
        p['multiplicity'] = sum(multiplicity[cell] for cell in marked)
        p['bbox_score'] = len(marked & bbox_union)
        p['support'] = len(p['cells'] & union)
    scores = [p['score'] for p in panels]
    assert max(scores) > 0 and scores.count(max(scores)) == 1, 'Unique positive census maximum required'
    return {'background': bg, 'selector': selector, 'panels': panels, 'hubs': hubs,
            'probes': probes, 'union': union, 'multiplicity': multiplicity,
            'bbox_union': bbox_union, 'winner': scores.index(max(scores))}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the entire original receiver with the most distinct covered markers."""
    meta = _analyze(input_grid)
    return meta['panels'][meta['winner']]['panel'].copy()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _sample_cells(rng, cells, count):
    ordered = sorted(cells)
    if count == 0:
        return set()
    indices = rng.choice(len(ordered), size=count, replace=False)
    return {ordered[int(i)] for i in indices}


def _normal(cells):
    r0, c0, _, _ = _box(cells)
    return {(r-r0, c-c0) for r, c in cells}


def _shape(rng, depth=None, span=None):
    if depth is None:
        simple = float(rng.random())
        if simple < .16:
            return {(0, 0)}
        if simple < .34:
            length = int(rng.integers(2, 8))
            return {(0, c) for c in range(length)} if rng.random() < .5 else {(r, 0) for r in range(length)}
    depth = int(rng.integers(2, 4)) if depth is None else depth
    span = int(rng.integers(3, 8)) if span is None else span
    spine = int(rng.integers(span))
    cells = {(r, spine) for r in range(depth)}
    count = int(rng.integers(max(depth+1, 4), min(depth*span-1, 13)+1))
    while len(cells) < count:
        fringe = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in DIRS
                         if 0 <= r+dr < depth and 0 <= c+dc < span} - cells)
        if not fringe:
            break
        cells.add(_choose(rng, fringe))
    return _normal(cells)


def _attach(rng, shape, hub_box, direction, fixed_left=None):
    """A shape is oriented with its near-hub face downward before orientation."""
    cells = set(shape)
    if direction == 1:
        cells = _normal({(-r, c) for r, c in cells})
    elif direction == 2:
        cells = _normal({(c, r) for r, c in cells})
    elif direction == 3:
        cells = _normal({(c, -r) for r, c in cells})
    r0, c0, r1, c1 = hub_box
    sr0, sc0, sr1, sc1 = _box(cells)
    if direction in (0, 1):
        near = sr1-1 if direction == 0 else 0
        near_cols = sorted(c for r, c in cells if r == near)
        left = c0 + int(rng.integers(3)) - _choose(rng, near_cols) if fixed_left is None else fixed_left
        top = r0-sr1 if direction == 0 else r1
    else:
        near = sc1-1 if direction == 2 else 0
        near_rows = sorted(r for r, c in cells if c == near)
        top = r0 + int(rng.integers(3)) - _choose(rng, near_rows)
        left = c0-sc1 if direction == 2 else c1
    return {(r+top, c+left) for r, c in cells}


def _geometry(rng, panel_count, extra_hubs, extra_probes, witness):
    """Construct two overlapping launches, then add compatible independent geometry."""
    h, w = int(rng.integers(26, 31)), int(rng.integers(18, 31))
    top_depth = 2 if witness == 'union' else int(rng.integers(2, 4))
    bottom_depth = int(rng.integers(3, 5))
    col = int(rng.integers(6, w-8))
    first = (top_depth+4+int(rng.integers(2)), col)
    second_col = col+int(rng.integers(-1, 2)) if witness else int(rng.integers(5, w-6))
    second = (h-bottom_depth-6-int(rng.integers(2)), second_col)
    hub_boxes = [(r, c, r+3, c+3) for r, c in (first, second)]
    base = _shape(rng, depth=2, span=int(rng.integers(5, 8)))
    p0 = _attach(rng, base, hub_boxes[0], 0)
    left = _box(p0)[1] + int(rng.integers(-1, 2))
    # Most pairs use related but unequal shapes, preserving an overlapping core.
    other = set(base)
    if rng.random() < .75:
        fringe = sorted({(r+dr, c+dc) for r, c in other for dr, dc in DIRS
                         if 0 <= r+dr < 2 and -1 <= c+dc <= 7} - other)
        if fringe:
            other.add(_choose(rng, fringe))
            other = _normal(other)
    p1 = _attach(rng, other, hub_boxes[1], 0, fixed_left=left)
    if not witness:
        p1 = _attach(rng, _shape(rng), hub_boxes[1], 0)
    p2 = _attach(rng, _shape(rng, depth=int(rng.integers(2, 4))), hub_boxes[1], 1)
    probes = [(p0, 0, 0), (p1, 1, 0), (p2, 1, 1)]
    if not witness and rng.random() < .65:
        first_direction = int(rng.integers(panel_count))
        probes[0] = (_attach(rng, _shape(rng), hub_boxes[0], first_direction), 0, first_direction)
    # The source convention is local side contact, checked for every proposed probe.
    def valid_probe(cells, hid, direction):
        if any(r < 1 or r >= h-1 or c < 1 or c >= w-1 for r, c in cells):
            return False
        all_hubs = set().union(*(_rectangle(b) for b in hub_boxes))
        if cells & all_hubs:
            return False
        contacts = {(j, d) for r, c in cells for d, (dr, dc) in enumerate(DIRS)
                    for j, box in enumerate(hub_boxes) if (r-dr, c-dc) in _rectangle(box)}
        return contacts == {(hid, direction)} and all(not cells & old and not _adjacent(cells, old) for old, _, _ in probes if old is not cells)
    if any(not valid_probe(cells, hid, d) for cells, hid, d in probes):
        return None
    # Receiver lengths include the core footprints, with variable unused margins.
    boxes = []
    for d in (0, 1):
        destinations = set().union(*(_shift(c, di, (h, w)) for c, _, di in probes if di == d))
        _, a, _, b = _box(destinations)
        lo = max(1, a-int(rng.integers(1, 5)))
        hi = min(w-1, b+int(rng.integers(1, 6)))
        boxes.append((0, lo, top_depth, hi) if d == 0 else (h-bottom_depth, lo, h, hi))
    for d in range(2, panel_count):
        depth = int(rng.integers(2, 4))
        start, end = int(rng.integers(4, 8)), int(rng.integers(h-8, h-3))
        boxes.append((start, 0, end, depth) if d == 2 else (start, w-depth, end, w))
    all_panels = set().union(*(_rectangle(b) for b in boxes))
    if any(c & all_panels or _adjacent(c, all_panels) for c, _, _ in probes):
        return None
    if any(_rectangle(b) & all_panels or _adjacent(_rectangle(b), all_panels) for b in hub_boxes):
        return None
    # Add lateral launches to make every new receiver participate.
    for d in range(2, panel_count):
        placed = False
        for _ in range(35):
            hid = int(rng.integers(len(hub_boxes)))
            cells = _attach(rng, _shape(rng), hub_boxes[hid], d)
            if valid_probe(cells, hid, d) and not (cells & all_panels or _adjacent(cells, all_panels)):
                probes.append((cells, hid, d))
                placed = True
                break
        if not placed:
            return None
    for _ in range(extra_hubs):
        placed = False
        for _ in range(70):
            r, c = int(rng.integers(5, h-7)), int(rng.integers(5, w-7))
            box = r, c, r+3, c+3
            full = _rectangle(box)
            occupied = all_panels | set().union(*(_rectangle(b) for b in hub_boxes)) | set().union(*(p[0] for p in probes))
            if full & occupied or _adjacent(full, occupied):
                continue
            hub_boxes.append(box)
            hid = len(hub_boxes)-1
            d = int(rng.integers(panel_count))
            cells = _attach(rng, _shape(rng), box, d)
            if valid_probe(cells, hid, d) and not (cells & all_panels or _adjacent(cells, all_panels)):
                probes.append((cells, hid, d))
                placed = True
                break
            hub_boxes.pop()
        if not placed:
            return None
    for _ in range(extra_probes):
        for _ in range(25):
            hid, d = int(rng.integers(len(hub_boxes))), int(rng.integers(panel_count))
            cells = _attach(rng, _shape(rng), hub_boxes[hid], d)
            if valid_probe(cells, hid, d) and not (cells & all_panels or _adjacent(cells, all_panels)):
                probes.append((cells, hid, d))
                break
    if len(probes) > 8:
        return None
    return (h, w), hub_boxes, probes, boxes


def _paint(rng, geometry, palette, witness):
    shape, hubs, probes, boxes = geometry
    bg, selector, ink, comp1, comp2 = palette
    a = np.full(shape, bg, dtype=np.int8)
    mult = Counter(cell for cells, _, d in probes for cell in _shift(cells, d, shape))
    union = set(mult)
    bbox_union = set().union(*(_shift(_rectangle(_box(cells)), d, shape) for cells, _, d in probes))
    regions = [_rectangle(b) for b in boxes]
    supports = [p & union for p in regions]
    if witness:
        winner = 1 if witness == 'union' else 0
        loser = 1-winner
        if len(supports[winner]) < 4 or len(supports[loser]) < 3:
            return None
        if witness == 'union':
            eligible = {c for c in supports[0] if mult[c] >= 2}
            holes = (regions[0] & bbox_union) - union
            if (len(eligible) < 3 or len(holes) < 2
                    or len(regions[1])*3 <= len(regions[0])*4
                    or len(supports[0]) <= len(supports[1])):
                return None
            marks = [_sample_cells(rng, eligible, 3), _sample_cells(rng, supports[1], 4)]
            marks[0] |= _sample_cells(rng, holes, 2)
            # Keep winner's raw total below the loser's and bbox score at four.
            outside = regions[0] - union - marks[0]
            marks[0] |= _sample_cells(rng, outside, min(len(outside), int(rng.integers(0, 3))))
        else:
            marks = [_sample_cells(rng, supports[0], 4), _sample_cells(rng, supports[1], 3)]
            outside = regions[1] - union
            if len(outside) < 3:
                return None
            marks[1] |= _sample_cells(rng, outside, int(rng.integers(3, min(7, len(outside))+1)))
    else:
        eligible = [i for i, support in enumerate(supports) if len(support) >= 3]
        if not eligible:
            return None
        winner = _choose(rng, eligible)
        high = int(rng.integers(3, min(9, len(supports[winner]))+1))
        marks = []
        for i, support in enumerate(supports):
            score = high if i == winner else int(rng.integers(0, min(high-1, len(support))+1))
            m = _sample_cells(rng, support, score)
            outside = regions[i] - union
            m |= _sample_cells(rng, outside, int(rng.integers(0, min(12, len(outside))+1)))
            marks.append(m)
    for i, box in enumerate(boxes):
        # Color textures vary independently of score and winner identity.
        r0, c0, r1, c1 = box
        mode = int(rng.integers(4))
        for r, c in sorted(regions[i]):
            bit = int(rng.integers(2)) if mode == 0 else ((r if mode == 1 else c if mode == 2 else r+c) % 3 == 0)
            a[r, c] = comp1 if bit else comp2
        for cell in marks[i]:
            a[cell] = selector
    for box in hubs:
        r0, c0, r1, c1 = box
        a[r0:r1, c0:c1] = RING
        a[r0+1, c0+1] = selector
    for cells, _, _ in probes:
        for cell in cells:
            a[cell] = ink
    try:
        meta = _analyze(a)
    except AssertionError:
        return None
    # Match intended geometry, and reject accidental panel/probe merging.
    if len(meta['probes']) != len(probes) or len(meta['hubs']) != len(hubs):
        return None
    if witness == 'union':
        p = meta['panels']
        wi = meta['winner']
        if not (p[wi]['multiplicity'] < max(x['multiplicity'] for x in p)
                and p[wi]['bbox_score'] < max(x['bbox_score'] for x in p)
                and p[wi]['raw'] < max(x['raw'] for x in p)):
            return None
    return a


def _make_pair(rng, selector, background, panel_count, extra_hubs, extra_probes, witness, rotation):
    for _ in range(650):
        geometry = _geometry(rng, panel_count, extra_hubs, extra_probes, witness)
        if geometry is None:
            continue
        colors = [c for c in range(10) if c not in (RING, selector, background)]
        rng.shuffle(colors)
        a = _paint(rng, geometry, (background, selector, *colors[:3]), witness)
        if a is None:
            continue
        a = np.rot90(a, rotation).copy()
        if rng.random() < .5:
            a = a[:, ::-1].copy()
        return {'input': a, 'output': solve(a)}
    raise RuntimeError('Unable to construct boundary census geometry')


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(name + ' is outside its legal integer range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    roles = [c for c in range(10) if c != RING]
    rng.shuffle(roles)
    selector = roles[0]
    backgrounds = roles[1:3] if rng.random() < .5 else roles[1:2]
    orientation = int(rng.integers(4))
    specs = [('union', 2, 0, 0, orientation), ('geometry', 2, 0, 0, (orientation+1) % 4)]
    for _ in range(num_train-2):
        pc = int(rng.integers(2, 5))
        specs.append((None, pc, int(rng.integers(0, 2)), int(rng.integers(0, 2)), int(rng.integers(4))))
    rng.shuffle(specs)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            if split == 'train':
                witness, pc, eh, ep, rot = specs[i]
            else:
                witness, pc, eh, ep, rot = None, int(rng.integers(3, 5)), int(rng.integers(1, 3)), int(rng.integers(0, 2)), int(rng.integers(4))
            for _ in range(12):
                pair = _make_pair(rng, selector, _choose(rng, backgrounds), pc, eh, ep, witness, rot)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Unable to construct distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _strict_wrong(meta, field):
    values = [p[field] for p in meta['panels']]
    return values.count(max(values)) == 1 and values.index(max(values)) != meta['winner']


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, selectors, train, test = set(), set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            key = a.shape, a.tobytes()
            assert key not in seen
            seen.add(key)
            meta = _analyze(a)
            selectors.add(meta['selector'])
            assert np.array_equal(b, meta['panels'][meta['winner']]['panel'])
            assert a.shape != b.shape
            assert not any(RING in p['panel'] for p in meta['panels'])
            (train if split == 'train' else test).append(meta)
    assert len(selectors) == 1
    assert set(p['direction'] for m in train for p in m['probes']) == set(range(4))
    for field in ('raw', 'multiplicity', 'bbox_score', 'support'):
        assert any(_strict_wrong(m, field) for m in train), 'Training must defeat ' + field
    assert any(any(p['score'] * m['panels'][m['winner']]['panel'].size >
                       m['panels'][m['winner']]['score'] * p['panel'].size
                       for i, p in enumerate(m['panels']) if i != m['winner']) for m in train), 'Count, not density'
    assert any(len(m['panels']) == 2 for m in train)
    assert all(len(m['panels']) >= 3 and len(m['hubs']) >= 3 for m in test)


def _signature(puzzle):
    return tuple((pair[k].shape, pair[k].tobytes()) for split in ('train', 'test')
                 for pair in puzzle[split] for k in ('input', 'output'))


def self_test():
    signatures, layouts, dimensions, outputs, palettes = set(), set(), set(), set(), set()
    hub_counts, probe_counts, panel_counts, winner_sides = set(), set(), set(), set()
    overlaps = partial = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                m = _analyze(a)
                assert np.array_equal(solve(a), b)
                dimensions.add(a.shape)
                outputs.add(b.shape)
                palettes.add(tuple(int(x) for x in np.unique(a)))
                mapping = {}
                layouts.add((a.shape, bytes(mapping.setdefault(int(x), len(mapping)) for x in a.ravel())))
                hub_counts.add(len(m['hubs']))
                probe_counts.add(len(m['probes']))
                panel_counts.add(len(m['panels']))
                r0, c0, r1, c1 = m['panels'][m['winner']]['box']
                winner_sides.add(next(i for i, v in enumerate((r0 == 0, r1 == a.shape[0], c0 == 0, c1 == a.shape[1])) if v))
                overlaps += any(n > 1 for n in m['multiplicity'].values())
                panel_union = set().union(*(p['cells'] for p in m['panels']))
                partial += any(bool(p['dest'] & panel_union) and bool(p['dest'] - panel_union) for p in m['probes'])
    assert len(signatures) == 42 and len(layouts) >= 190
    assert len(dimensions) >= 30 and len(outputs) >= 20 and len(palettes) >= 20
    assert hub_counts == {2, 3, 4} and panel_counts == {2, 3, 4}
    assert len(probe_counts) >= 5 and winner_sides == set(range(4))
    assert 80 <= overlaps < 210 and partial > 0
    a = generate_puzzle(123, 5, 3)
    assert _signature(a) == _signature(generate_puzzle(123, 5, 3))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(900+nt*5+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'structures': len(layouts),
            'input_shapes': len(dimensions), 'output_shapes': len(outputs),
            'palettes': len(palettes), 'hub_counts': sorted(hub_counts),
            'probe_counts': sorted(probe_counts), 'panel_counts': sorted(panel_counts),
            'overlap_pairs': overlaps, 'partial_coverage_pairs': partial}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
