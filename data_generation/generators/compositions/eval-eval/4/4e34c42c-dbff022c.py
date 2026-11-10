"""Welded chambers: register stamps, cancel double walls, then fill once.

Only NumPy and the standard library are used. Nothing is read or written at
runtime. Construction coordinates are discarded before input-only solving.
"""
from collections import Counter, defaultdict
import hashlib
import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _groups(points, diagonal=False):
    remaining = set(points)
    groups = []
    steps = EIGHT if diagonal else STEPS
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _norm(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return {(r-r0, c-c0): int(v) for (r, c), v in cells.items()}, (r0, c0)


def _shape(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _render(cells, bg):
    cells, _ = _norm(cells)
    out = np.full(_shape(cells), bg, dtype=np.int8)
    for p, color in cells.items():
        out[p] = color
    return out


def _array(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), 'invalid shape'
    assert np.issubdtype(a.dtype, np.integer), 'noninteger grid'
    assert np.all((0 <= a) & (a <= 9)), 'invalid colors'
    return a


def _signature(cells):
    norm, origin = _norm(cells)
    return tuple((r, c, v) for (r, c), v in sorted(norm.items())), origin


def _parse(grid):
    grid = _array(grid)
    colors, counts = np.unique(grid, return_counts=True)
    assert counts.max()*2 > grid.size, 'background must be a strict majority'
    bg = int(colors[counts.argmax()])
    components = _groups(zip(*np.nonzero(grid != bg)), diagonal=True)
    candidates = []
    h, w = grid.shape
    for index, group in enumerate(components):
        cells = {p: int(grid[p]) for p in group}
        norm, (r0, c0) = _norm(cells)
        sh, sw = _shape(norm)
        if len(group) != sh*sw:
            continue
        lanes = []
        if sh == 2 and 3 <= sw <= 5:
            if r0 == 0:
                lanes.append(('top', grid[0, c0:c0+sw], grid[1, c0:c0+sw]))
            if r0+sh == h:
                lanes.append(('bottom', grid[-1, c0:c0+sw], grid[-2, c0:c0+sw]))
        if sw == 2 and 3 <= sh <= 5:
            if c0 == 0:
                lanes.append(('left', grid[r0:r0+sh, 0], grid[r0:r0+sh, 1]))
            if c0+sw == w:
                lanes.append(('right', grid[r0:r0+sh, -1], grid[r0:r0+sh, -2]))
        for side, outer, inner in lanes:
            source, fill = tuple(map(int, outer)), tuple(map(int, inner))
            if len(set(source)) == len(source):
                candidates.append((index, side, dict(zip(source, fill))))
    assert len(candidates) == 1, 'need one unambiguous local edge key'
    index, side, mapping = candidates[0]
    role_colors = set(mapping) | set(mapping.values())
    markers = set(map(int, colors)) - role_colors - {bg}
    assert len(markers) == 2, 'exactly two temporary stamp colors'
    parts, origins = [], []
    for j, group in enumerate(components):
        if j != index:
            part, origin = _norm({p: int(grid[p]) for p in group})
            parts.append(part)
            origins.append(origin)
    assert 2 <= len(parts) <= 6, 'need two to six fragments'
    return bg, side, mapping, markers, parts, origins


def _register(parts, markers):
    occurrences = defaultdict(list)
    for i, part in enumerate(parts):
        local = set()
        for group in _groups((p for p, v in part.items() if v in markers), diagonal=True):
            cells = {p: part[p] for p in group}
            sig, origin = _signature(cells)
            nh, nw = _shape(_norm(cells)[0])
            assert len(cells) == nh*nw and sorted((nh, nw)) in ([2, 2], [2, 3])
            assert set(cells.values()) == markers, 'stamp must use both temporary colors'
            assert sig not in local, 'a stamp cannot occur twice in one fragment'
            local.add(sig)
            occurrences[sig].append((i, origin))
        assert local, 'fragment lacks registration evidence'
    assert all(len(items) >= 2 for items in occurrences.values()), 'unmatched stamp'
    graph = [[] for _ in parts]
    for items in occurrences.values():
        i, a = items[0]
        for j, b in items[1:]:
            delta = a[0]-b[0], a[1]-b[1]
            graph[i].append((j, delta))
            graph[j].append((i, (-delta[0], -delta[1])))
    positions = {0: (0, 0)}
    stack = [0]
    while stack:
        i = stack.pop()
        for j, delta in graph[i]:
            proposed = positions[i][0]+delta[0], positions[i][1]+delta[1]
            if j in positions:
                assert positions[j] == proposed, 'inconsistent registration cycle'
            else:
                positions[j] = proposed
                stack.append(j)
    assert len(positions) == len(parts), 'disconnected stamp graph'
    wall_colors, multiplicity, stamp_points = {}, Counter(), set()
    for i, part in enumerate(parts):
        dr, dc = positions[i]
        for (r, c), color in part.items():
            p = r+dr, c+dc
            if color in markers:
                stamp_points.add(p)
            else:
                assert p not in wall_colors or wall_colors[p] == color, 'wall color conflict'
                wall_colors[p] = color
                multiplicity[p] += 1
    assert not stamp_points & wall_colors.keys(), 'stamp overlaps a wall'
    walls = {p: v for p, v in wall_colors.items() if multiplicity[p] % 2}
    assert walls and max(_shape(_norm(walls)[0])) <= 30
    return walls, wall_colors, multiplicity, positions, len(occurrences)


def _fill(walls, bg, mapping):
    """Pad before finding four-connected background: the crop is no wall."""
    raw = _render(walls, bg)
    a = np.pad(raw, 1, constant_values=bg)
    out = a.copy()
    regions = []
    for group in _groups(zip(*np.nonzero(a == bg))):
        if any(r in (0, a.shape[0]-1) or c in (0, a.shape[1]-1) for r, c in group):
            continue
        boundary = {int(a[r+dr, c+dc]) for r, c in group for dr, dc in STEPS
                    if int(a[r+dr, c+dc]) != bg}
        assert len(boundary) == 1, 'mixed-color cavity boundary'
        color = next(iter(boundary))
        if color in mapping:
            for p in group:
                out[p] = mapping[color]
        rr, cc = zip(*group)
        regions.append({'color': color, 'area': len(group), 'mapped': color in mapping,
                        'irregular': len(group) != (max(rr)-min(rr)+1)*(max(cc)-min(cc)+1)})
    return out[1:-1, 1:-1].copy(), regions


def _analyze(input_grid):
    bg, side, mapping, markers, parts, origins = _parse(input_grid)
    walls, union, counts, offsets, stamps = _register(parts, markers)
    out, regions = _fill(walls, bg, mapping)
    ordinary, _ = _fill(union, bg, mapping)
    _, union_regions = _fill(union, bg, {})
    unregistered = {(r+dr, c+dc): v for part, (dr, dc) in zip(parts, origins)
                    for (r, c), v in part.items() if v not in markers}
    without_A, _ = _fill(unregistered, bg, mapping)
    info = {'background': bg, 'side': side, 'mapping': tuple(mapping.items()),
            'key_length': len(mapping), 'fragments': len(parts), 'stamps': stamps,
            'erased_wall_cells': sum(n % 2 == 0 for n in counts.values()),
            'cavities': len(regions), 'mapped_cavities': sum(x['mapped'] for x in regions),
            'inert_cavities': sum(not x['mapped'] for x in regions),
            'irregular_cavities': sum(x['irregular'] for x in regions),
            'largest_cavity': max((x['area'] for x in regions if x['mapped']), default=0),
            'union_cavities': len(union_regions),
            'omitting_A_changes': not np.array_equal(out, without_A),
            'omitting_B_changes': not np.array_equal(out, _render(walls, bg)),
            'ordinary_union_changes': not np.array_equal(out, ordinary),
            'chain': any(x['mapped'] and mapping[x['color']] in mapping and
                         mapping[mapping[x['color']]] != mapping[x['color']] for x in regions),
            'self_map': any(x['mapped'] and mapping[x['color']] == x['color'] for x in regions),
            'both_axes': len({r for r, c in offsets.values()}) > 1 and
                         len({c for r, c in offsets.values()}) > 1}
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every offset, role and fill from the supplied pixels only."""
    bg, _, mapping, markers, parts, _ = _parse(input_grid)
    walls = _register(parts, markers)[0]
    return _fill(walls, bg, mapping)[0]


def _box_points(box):
    a, b, c, d = box
    return {(r, col) for r in range(a, c+1) for col in range(b, d+1)}


def _perimeter(box, rng):
    a, b, c, d = box
    cells = {p for p in _box_points(box) if p[0] in (a, c) or p[1] in (b, d)}
    # Vary negative-space profiles without changing seam-adjacent corner cells.
    if rng.random() < .7:
        for col in range(b+2, d-1):
            depth = int(rng.integers(0, min(3, c-a-1)))
            cells.update((r, col) for r in range(c-depth, c))
    return cells


def _rooms(rng, n):
    boxes = [(0, 0, int(rng.integers(3, 6)), int(rng.integers(3, 6)))]
    edges = []
    for i in range(1, n):
        for _ in range(70):
            parent = int(rng.integers(i))
            a, b, c, d = boxes[parent]
            hh, ww = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            direction = int(rng.integers(4))
            if direction in (0, 1):
                start = int(rng.integers(a-hh+2, c-1))
                box = (start, d, start+hh, d+ww) if direction == 0 else (start, b-ww, start+hh, b)
            else:
                start = int(rng.integers(b-ww+2, d-1))
                box = (c, start, c+hh, start+ww) if direction == 2 else (a-hh, start, a, start+ww)
            points = _box_points(box)
            if any(points & _box_points(other) for j, other in enumerate(boxes) if j != parent):
                continue
            shared = points & _box_points(boxes[parent])
            if len(shared) < 3:
                continue
            boxes.append(box)
            edges.append((parent, i, sorted(shared)))
            break
        else:
            raise AssertionError('room growth exhausted')
    return boxes, edges


def _stamp(rng, shared, all_walls, occupied, signatures, marker_colors):
    p0, p1 = shared[0], shared[-1]
    vertical = p0[1] == p1[1]
    candidates = []
    for endpoint, outward in ((p0, -1), (p1, 1)):
        for side in (-1, 1):
            width = int(rng.integers(2, 4))
            if vertical:
                coords = [(endpoint[0]+outward*r, endpoint[1]+side*c)
                          for r in (1, 2) for c in range(1, width+1)]
                door = endpoint[0], endpoint[1]+side
            else:
                coords = [(endpoint[0]+side*c, endpoint[1]+outward*r)
                          for r in (1, 2) for c in range(1, width+1)]
                door = endpoint[0]+side, endpoint[1]
            if set(coords) & all_walls:
                continue
            if any((r+dr, c+dc) in occupied for r, c in coords for dr, dc in ((0, 0),)+EIGHT):
                continue
            for _ in range(20):
                values = list(map(int, rng.choice(marker_colors, len(coords))))
                if len(set(values)) != 2:
                    continue
                cells = dict(zip(coords, values))
                sig, _ = _signature(cells)
                if sig not in signatures:
                    candidates.append((cells, sig, door))
                    break
    assert candidates, 'no clean exterior stamp site'
    return candidates[int(rng.integers(len(candidates)))]


def _auxiliary(rng, box, forbidden, other_boxes, color, force_closed=True):
    a, b, c, d = box
    candidates = []
    for col in range(b, d-1):
        candidates.extend([(a-3, col), (c+1, col)])
    for row in range(a, c-1):
        candidates.extend([(row, b-3), (row, d+1)])
    rng.shuffle(candidates)
    for r0, c0 in candidates:
        bbox = {(r0+r, c0+c) for r in range(3) for c in range(3)}
        if bbox & forbidden or any(bbox & _box_points(other) for other in other_boxes):
            continue
        # Distinct rooms can touch ink, but never acquire a mixed-color hole.
        kind = int(rng.integers(2))
        coords = {(r0+r, c0+c) for r in range(3) for c in range(3)
                  if (r in (0, 2) or c in (0, 2))} if kind == 0 else {
                      (r0, c0+1), (r0+1, c0), (r0+1, c0+2), (r0+2, c0+1)}
        if not force_closed and rng.random() < .5:
            coords.remove(min(coords))
        return {p: color for p in coords}
    raise AssertionError('no auxiliary room site')


def _place_key(grid, mapping, start):
    for j, (source, fill) in enumerate(mapping.items()):
        grid[0, start+j], grid[1, start+j] = source, fill


def _pack(rng, parts, mapping):
    parts = [_norm(p)[0] for p in parts]
    shapes = [_shape(p) for p in parts]
    assert all(h <= 26 and w <= 28 for h, w in shapes)
    order = sorted(range(len(parts)), key=lambda i: -len(parts[i]))
    area = sum(len(p) for p in parts)+2*len(mapping)
    for attempt in range(18):
        h = int(rng.integers(max(x[0] for x in shapes)+3, 31)) if attempt < 9 else 30
        w = int(rng.integers(max(max(x[1] for x in shapes), len(mapping)), 31)) if attempt < 9 else 30
        if h*w <= 2*area:
            continue
        grid = np.zeros((h, w), dtype=np.int8)
        start = int(rng.integers(w-len(mapping)+1))
        _place_key(grid, mapping, start)
        blocked = {(r+dr, c+dc) for r, c in zip(*np.nonzero(grid)) for dr, dc in ((0, 0),)+EIGHT}
        for i in order:
            ph, pw = shapes[i]
            positions = [(r0, c0) for r0 in range(h-ph+1) for c0 in range(w-pw+1)
                         if all((r+r0, c+c0) not in blocked for r, c in parts[i])]
            if not positions:
                break
            r0, c0 = positions[int(rng.integers(len(positions)))]
            for (r, c), v in parts[i].items():
                grid[r+r0, c+c0] = v
                blocked.update((r+r0+dr, c+c0+dc) for dr, dc in ((0, 0),)+EIGHT)
        else:
            return grid
    raise AssertionError('bounded packing exhausted')


def _make_pair(rng, n, leak, main_source=None):
    for _ in range(80):
        try:
            k = int(rng.integers(3, 6))
            markers = (k+3, k+4)
            mapping = {i: int(rng.integers(1, k+3)) for i in range(1, k+1)}
            mapping.update({1: 2, 2: k+1, 3: 3})
            order = rng.permutation(list(mapping))
            mapping = {int(i): mapping[int(i)] for i in order}
            source = int(rng.integers(1, k+1)) if main_source is None else main_source
            boxes, edges = _rooms(rng, n)
            parts = [{p: source for p in _perimeter(box, rng)} for box in boxes]
            signatures, occupied = set(), set()
            all_walls = set().union(*(set(p) for p in parts))
            door_options = []
            for i, j, shared in edges:
                # Shared partition endpoints have one copy; its interior has two.
                for p in (shared[0], shared[-1]):
                    parts[j].pop(p, None)
                stamp, sig, door = _stamp(rng, shared, all_walls, occupied, signatures, markers)
                signatures.add(sig)
                occupied.update(stamp)
                parts[i].update(stamp)
                parts[j].update(stamp)
                owners = [q for q in (i, j) if door in parts[q] and parts[q][door] == source]
                if len(owners) == 1:
                    other = j if owners[0] == i else i
                    door_options.append((other, door))
            if leak:
                assert door_options
                other, door = door_options[int(rng.integers(len(door_options)))]
                parts[other][door] = source
            # Paintable/self-map and fill-only inert cavities coexist in every pair.
            for i, color in ((0, 3), (n-1, k+1)):
                forbidden = set().union(*(set(p) for p in parts))
                aux = _auxiliary(rng, boxes[i], forbidden, [b for j, b in enumerate(boxes) if i != j], color)
                parts[i].update(aux)
            for i in range(1, n-1):
                if rng.random() < .35:
                    forbidden = set().union(*(set(p) for p in parts))
                    aux = _auxiliary(rng, boxes[i], forbidden, [b for j, b in enumerate(boxes) if i != j],
                                     int(rng.choice([2, k+1])), force_closed=False)
                    parts[i].update(aux)
            assert all(len(_groups(p, diagonal=True)) == 1 for p in parts)
            # Intended coordinates, not the pixel matcher, establish the target.
            counts, wall_colors = Counter(), {}
            for part in parts:
                for p, color in part.items():
                    if color not in markers:
                        assert p not in wall_colors or wall_colors[p] == color
                        counts[p] += 1
                        wall_colors[p] = color
            walls = {p: v for p, v in wall_colors.items() if counts[p] % 2}
            expected, regions = _fill(walls, 0, mapping)
            principal = [r for r in regions if r['color'] == source and r['area'] > 1]
            assert bool(principal) != leak, 'leak/closure construction disagrees'
            # Whole-example transformations preserve the pieces' orientations.
            inp = _pack(rng, parts, mapping)
            turn = int(rng.integers(4))
            inp, expected = np.rot90(inp, turn), np.rot90(expected, turn)
            if rng.random() < .5:
                inp, expected = np.fliplr(inp), np.fliplr(expected)
            palette = rng.permutation(10).astype(np.int8)
            inp, expected = palette[inp].copy(), palette[expected].copy()
            predicted, info = _analyze(inp)
            assert np.array_equal(predicted, expected), 'input reconstruction disagrees'
            assert info['omitting_A_changes'] and info['omitting_B_changes']
            assert info['ordinary_union_changes'] and info['erased_wall_cells'] > 0
            assert info['mapped_cavities'] and info['inert_cavities']
            return {'input': inp, 'output': expected}
        except AssertionError:
            continue
    raise RuntimeError('bounded chamber construction exhausted')


def _count(value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError('illegal example count')
    return int(value)


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            # Two minimal demonstrations already expose closed and leaking welds.
            n = 2 if split == 'train' and index < 2 else int(rng.integers(2, 4)) if split == 'train' else 4
            leak = index == 1 if split == 'train' and index < 2 else bool(rng.integers(2))
            main = 1 if split == 'train' and index == 0 else None
            for _ in range(8):
                pair = _make_pair(rng, n, leak, main)
                key = _fingerprint(pair['input'])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _array(pair['input']), _array(pair['output'])
            key = _fingerprint(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            predicted, info = _analyze(inp)
            assert np.array_equal(predicted, out), 'wrong output'
            assert info['erased_wall_cells'] > 0 and info['ordinary_union_changes']
            assert info['omitting_A_changes'] and info['omitting_B_changes']
            assert info['mapped_cavities'] and info['inert_cavities']
            records[split].append(info)
    train = records['train']
    assert any(x['chain'] for x in train), 'one-step lookup not demonstrated'
    assert any(x['self_map'] for x in train)
    assert any(x['largest_cavity'] > 1 for x in train), 'closed weld missing'
    assert any(x['largest_cavity'] == 1 for x in train), 'leaking weld missing'


def self_test():
    """42 streaming seeds; statistics/hashes only, no files or retained grids."""
    shapes, outputs, backgrounds, sides, normalized, signatures = set(), set(), set(), set(), set(), set()
    counts, key_lengths = set(), set()
    stats = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                predicted, info = _analyze(inp)
                assert np.array_equal(predicted, out)
                shapes.add(inp.shape)
                outputs.add(out.shape)
                backgrounds.add(info['background'])
                sides.add(info['side'])
                counts.add(info['fragments'])
                key_lengths.add(info['key_length'])
                roles = {}
                canonical = bytes(roles.setdefault(int(v), len(roles)) for v in inp.flat)
                normalized.add(hashlib.sha256(canonical).digest())
                stats['pairs'] += 1
                for name in ('chain', 'self_map', 'irregular_cavities', 'both_axes'):
                    stats[name] += bool(info[name])
                # Recoloring, repacking orientation and input immutability.
                if seed < 3:
                    original = inp.copy()
                    permutation = np.array([7, 2, 0, 9, 4, 1, 8, 6, 3, 5])
                    for turn in range(4):
                        assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn))
                    assert np.array_equal(solve(permutation[inp]), permutation[out])
                    assert np.array_equal(inp, original)
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(normalized) == 210
    assert len(shapes) >= 30 and len(outputs) >= 30
    assert backgrounds == set(range(10)) and len(sides) == 4
    assert counts == {2, 3, 4} and key_lengths == {3, 4, 5}
    assert stats['irregular_cavities'] >= 25 and stats['both_axes'] >= 100
    # Explicit parity, single-lookup and four-neighbor leakage witnesses.
    ring = {(r, c): 1 for r in range(5) for c in range(6) if r in (0, 4) or c in (0, 5)}
    painted, _ = _fill(ring, 0, {1: 2, 2: 4})
    assert np.all(painted[1:4, 1:5] == 2)
    opened = dict(ring)
    del opened[0, 2]
    assert np.array_equal(_fill(opened, 0, {1: 2})[0], _render(opened, 0))
    diamond = {(0, 1): 1, (1, 0): 1, (1, 2): 1, (2, 1): 1}
    assert _fill(diamond, 0, {1: 2})[0][1, 1] == 2
    stamp_a = {(-2, 1): 8, (-2, 2): 9, (-1, 1): 9, (-1, 2): 8}
    stamp_b = {(5, 1): 8, (5, 2): 8, (6, 1): 9, (6, 2): 8}
    triple = dict(ring, **{})
    triple.update(stamp_a)
    triple.update(stamp_b)
    # Repeated constraints form consistent cycles; all three wall copies survive.
    rebuilt = _register([triple.copy() for _ in range(3)], {8, 9})
    assert rebuilt[0] == ring and set(rebuilt[2].values()) == {3}
    assert np.array_equal(_fill(rebuilt[0], 0, {1: 2})[0], painted)
    first = dict(ring)
    first.update(stamp_a)
    door_copy = dict(stamp_a)
    door_copy[0, 2] = 1
    welded = _register([first, door_copy], {8, 9})[0]
    assert welded == opened, 'double-drawn exterior pixel must open a leak'
    bad_cycle = dict(triple)
    for p in stamp_b:
        del bad_cycle[p]
    bad_cycle.update({(r+1, c): v for (r, c), v in stamp_b.items()})
    for broken in ([triple, triple, bad_cycle], [first, {**door_copy, (0, 2): 2}]):
        try:
            _register(broken, {8, 9})
        except AssertionError:
            pass
        else:
            raise AssertionError('inconsistent cycle or wall conflict accepted')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(100+a*10+b, a, b)
            validate_generated_puzzle(puzzle)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(73, 2, 1), generate_puzzle(73, 2, 1)
    assert all(_fingerprint(a[k]) == _fingerprint(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in pairs] for s, pairs in p.items()}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'pairs': stats['pairs'], 'input_shapes': len(shapes),
            'output_shapes': len(outputs), 'normalized_inputs': len(normalized),
            'backgrounds': sorted(backgrounds), 'key_sides': sorted(sides),
            'fragment_counts': sorted(counts), 'key_lengths': sorted(key_lengths),
            'feature_pair_counts': dict(stats)}
