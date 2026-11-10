"""Braced skew frames: hidden rigid pieces govern a one-cell release.

Only the standard library and NumPy are used. Nothing is read from disk.
"""
from __future__ import annotations

import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             np.all((0 <= a) & (a <= 9)), "grid colors must be integers 0..9")
    return a


def _cells(mask):
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _piece(kind, r, c):
    h, w = {'S': (2, 2), 'H': (1, 3), 'V': (3, 1)}[kind]
    return frozenset((r + dr, c + dc) for dr in range(h) for dc in range(w))


def _exact_covers(cells, limit=2):
    """Enumerate up to two exact partitions, enough to reject ambiguity."""
    cells = frozenset(cells)
    if not cells:
        return []
    candidates = [(kind, part) for r, c in sorted(cells) for kind in 'SHV'
                  if (part := _piece(kind, r, c)) <= cells]
    lookup = {p: [] for p in cells}
    for i, (_, part) in enumerate(candidates):
        for p in part:
            lookup[p].append(i)
    answers = []

    def search(remaining, chosen):
        if len(answers) >= limit:
            return
        if not remaining:
            answers.append(tuple(candidates[i] for i in chosen))
            return
        options = min(([i for i in lookup[p]
                        if candidates[i][1] <= remaining]
                       for p in sorted(remaining)), key=len)
        for i in options:
            search(remaining - candidates[i][1], chosen + [i])
            if len(answers) >= limit:
                break

    search(cells, [])
    return answers


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in remaining:
                        remaining.remove(p)
                        stack.append(p)
                        component.add(p)
        result.append(frozenset(component))
    return result


def _box(cells):
    return (min(r for r, _ in cells), min(c for _, c in cells),
            max(r for r, _ in cells), max(c for _, c in cells))


def _separate(a, b):
    return (a[2] + 1 < b[0] or b[2] + 1 < a[0] or
            a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _frame_info(cells):
    cells = frozenset(cells)
    r0, c0, r1, c1 = _box(cells)
    h = r1 - r0 + 1
    _require(h >= 5, "frame too short")
    rows = [sorted(c for r, c in cells if r == y) for y in range(r0, r1 + 1)]
    top, bottom = rows[0], rows[-1]
    length = len(top)
    _require(length >= 5 and len(bottom) == length, "invalid caps")
    _require(all(b == a + 1 for a, b in zip(top, top[1:])) and
             all(b == a + 1 for a, b in zip(bottom, bottom[1:])),
             "caps must be solid")
    delta = bottom[0] - top[0]
    _require(1 <= abs(delta) <= h - 2, "invalid slant")
    direction = 1 if delta > 0 else -1
    starts = []
    for i, row in enumerate(rows):
        _require(bool(row), "missing frame row")
        xs = sorted(direction * c for c in row)
        starts.append(xs[0])
        if 0 < i < h - 1:
            _require(len(xs) == 2 and xs[1] - xs[0] == length,
                     "invalid staircase sides")
    steps = tuple(b - a for a, b in zip(starts, starts[1:]))
    _require(steps[0] == 0 and steps[-1] == 1 and
             set(steps) <= {0, 1}, "invalid staircase profile")
    far = c1 if direction == 1 else c0
    released = frozenset((r, c if r == r1 or c == far else c + direction)
                         for r, c in cells)
    _require(len(released) == len(cells), "self-colliding frame")
    # Strict interiors before and after release; clearance prevents crushing.
    interior = set()
    for i in range(1, h - 1):
        r = r0 + i
        old_left, old_right = rows[i]
        new_row = sorted(c for y, c in released if y == r)
        _require(len(new_row) == 2, "invalid released frame")
        interior.update((r, c) for c in range(max(old_left, new_row[0]) + 1,
                                             min(old_right, new_row[1])))
    return {'cells': cells, 'released': released, 'safe': frozenset(interior),
            'box': (r0, c0, r1, c1), 'h': h, 'length': length,
            'offset': abs(delta), 'direction': direction, 'steps': steps}


def _motion(cover, walls, direction):
    """Least fixed point of backward blockage on rigid bar dependencies."""
    squares = frozenset(p for k, part in cover if k == 'S' for p in part)
    bars = [part for k, part in cover if k != 'S']
    targets = [frozenset((r, c + direction) for r, c in part) for part in bars]
    wall_hit = {i for i, target in enumerate(targets) if target & walls}
    square_hit = {i for i, target in enumerate(targets) if target & squares}
    dependencies = [{j for j, part in enumerate(bars)
                     if i != j and target & part}
                    for i, target in enumerate(targets)]
    blocked = wall_hit | square_hit
    depth = {i: 1 for i in blocked}
    while True:
        additions = {i for i in range(len(bars)) if i not in blocked and
                     dependencies[i] & blocked}
        if not additions:
            break
        new_depths = {i: 1 + min(depth[j] for j in dependencies[i] & blocked)
                      for i in additions}
        blocked.update(additions)
        depth.update(new_depths)
    moving = set(range(len(bars))) - blocked
    occupied = frozenset(p for _, part in cover for p in part)
    real_squares = {part for k, part in cover if k == 'S'}
    false_squares = [_piece('S', r, c) for r, c in sorted(occupied)
                     if _piece('S', r, c) <= occupied and
                     _piece('S', r, c) not in real_squares and
                     not _piece('S', r, c) <= squares]
    return {'squares': squares, 'bars': bars, 'targets': targets,
            'blocked': blocked, 'moving': moving, 'wall_hit': wall_hit,
            'square_hit': square_hit, 'depth': max(depth.values(), default=0),
            'false_squares': len(false_squares),
            'moving_dependency': any(dependencies[i] & moving for i in moving)}


def _analyze_input(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    _require(len(values) == 3, "expected background, frame, and load colors")
    background = int(values[np.argmax(counts)])
    _require(int(counts.max()) * 2 > a.size, "background must be a strict majority")
    roles = [int(v) for v in values if v != background]
    solutions = []
    for frame_color, neutral in (roles, roles[::-1]):
        try:
            objects = [_frame_info(part) for part in
                       _components(_cells(a == frame_color))]
            _require(bool(objects), "no frames")
            _require(all(_separate(x['box'], y['box'])
                         for i, x in enumerate(objects) for y in objects[i + 1:]),
                     "frames not separated")
            loads = [set() for _ in objects]
            for p in sorted(_cells(a == neutral)):
                owners = [i for i, obj in enumerate(objects) if p in obj['safe']]
                _require(len(owners) == 1, "load must be in one common interior")
                loads[owners[0]].add(p)
            for obj, load in zip(objects, loads):
                covers = _exact_covers(frozenset(load))
                _require(len(covers) == 1, "load tiling must be unique")
                obj['cover'] = covers[0]
                obj['load'] = frozenset(load)
                obj['motion'] = _motion(obj['cover'], obj['released'], obj['direction'])
            solutions.append({'background': background, 'frame_color': frame_color,
                              'neutral': neutral, 'objects': objects})
        except ValueError:
            continue
    _require(len(solutions) == 1, "frame/load roles must be geometrically unique")
    return solutions[0]


def _render(shape, background, frame_color, neutral, objects):
    out = np.full(shape, background, dtype=np.int8)
    occupied = set()

    def paint(cells, color):
        for r, c in cells:
            _require(0 <= r < shape[0] and 0 <= c < shape[1], "out-of-bounds motion")
            _require((r, c) not in occupied, "output collision")
            occupied.add((r, c))
            out[r, c] = color

    for obj in objects:
        motion = _motion(obj['cover'], obj['released'], obj['direction'])
        paint(obj['released'], frame_color)
        paint(motion['squares'], frame_color)
        for i, bar in enumerate(motion['bars']):
            paint(bar if i in motion['blocked'] else motion['targets'][i], neutral)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all roles, frames, pieces, and collision constraints from input."""
    a = _grid(input_grid)
    scene = _analyze_input(a)
    return _render(a.shape, scene['background'], scene['frame_color'],
                   scene['neutral'], scene['objects'])


def _new_frame(rng, multiple, residual):
    h = int(rng.integers(8, 13 if multiple else 18))
    length = int(rng.integers(9, 16 if multiple else 20))
    offset = int(rng.integers(2 if residual else 1, min(h - 2, 4 if multiple else 6) + 1))
    steps = np.zeros(h - 1, dtype=int)
    steps[-1] = 1
    if offset > 1:
        steps[rng.choice(np.arange(1, h - 2), offset - 1, replace=False)] = 1
    starts = np.r_[0, np.cumsum(steps)]
    cells = {(0, c) for c in range(length)}
    cells.update((h - 1, c) for c in range(offset, offset + length))
    cells.update((r, int(starts[r]) + dc) for r in range(1, h - 1)
                 for dc in (0, length))
    return _frame_info(cells)


def _sample_load(rng, obj, count, force_wall, evidence):
    safe = obj['safe']
    occupied, placed = set(), []
    if force_wall:
        r1, far = obj['box'][2:]
        part = _piece('H', r1 - 1, far - 3)
        if not part <= safe:
            return None
        placed.append(('H', part))
        occupied.update(part)
        # Contrast the anchored extreme wall with a wall segment that releases.
        # A horizontal bar just behind the latter enters its vacated wall cell.
        options = []
        for r in range(1, r1):
            right = max(c for y, c in obj['cells'] if y == r)
            if right < far:
                candidate = _piece('H', r, right - 3)
                if candidate <= safe and not candidate & occupied:
                    options.append(candidate)
        if not options:
            return None
        part = options[int(rng.integers(len(options)))]
        placed.append(('H', part))
        occupied.update(part)
    # Both shape classes, then independent choices for remaining pieces.
    kinds = ['S'] + [str(rng.choice(['S', 'H', 'V'], p=[.20, .40, .40]))
                     for _ in range(count - len(placed) - 1)]
    if not force_wall and all(k == 'S' for k in kinds):
        kinds[-1] = str(rng.choice(['H', 'V']))
    rng.shuffle(kinds)
    bias = 12.0 if evidence else float(rng.choice([.5, 2, 12, 30]))
    for kind in kinds:
        options, weights = [], []
        for r, c in sorted(safe):
            part = _piece(kind, r, c)
            if not part <= safe or part & occupied:
                continue
            contacts = sum((y + dr, x + dc) in occupied for y, x in part
                           for dr, dc in _NEIGHBORS)
            options.append(part)
            weights.append(1 + bias * contacts)
        if not options:
            return None
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        part = options[int(rng.choice(len(options), p=weights))]
        placed.append((kind, part))
        occupied.update(part)
    covers = _exact_covers(frozenset(occupied))
    if len(covers) != 1:
        return None
    # Unique reconstruction must recover the actual planted rigid pieces.
    if set(covers[0]) != set(placed):
        return None
    return covers[0]


def _evidence_basic(obj):
    m = obj['motion']
    bar_kinds = [k for k, _ in obj['cover'] if k != 'S']
    return (5 <= len(obj['cover']) <= 6 and obj['offset'] >= 2 and
            bool(m['square_hit']) and
            any(bar_kinds[i] == 'H' for i in m['wall_hit']) and
            any(m['targets'][i] & obj['cells'] for i in m['moving']))


def _evidence_chain(obj):
    m = obj['motion']
    return (len(obj['cover']) == 6 and m['depth'] >= 2 and
            m['false_squares'] > 0 and m['moving_dependency'])


def _new_chamber(rng, multiple, count, direction, evidence=None):
    for _ in range(1800):
        obj = _new_frame(rng, multiple, evidence == 'basic')
        cover = _sample_load(rng, obj, count, evidence == 'basic', evidence)
        if cover is None:
            continue
        obj['cover'] = cover
        obj['motion'] = _motion(cover, obj['released'], 1)
        if evidence == 'basic' and not _evidence_basic(obj):
            continue
        if evidence == 'chain' and not _evidence_chain(obj):
            continue
        if direction == -1:
            width = obj['box'][3] + 1
            mirror = lambda part: frozenset((r, width - 1 - c) for r, c in part)
            mirrored = _frame_info(mirror(obj['cells']))
            mirrored['cover'] = tuple((k, mirror(part)) for k, part in cover)
            obj = mirrored
        obj['load'] = frozenset(p for _, part in obj['cover'] for p in part)
        obj['motion'] = _motion(obj['cover'], obj['released'], direction)
        return obj
    raise RuntimeError('could not sample an informative uniquely tiled chamber')


def _translate(obj, dr, dc):
    shift = lambda part: frozenset((r + dr, c + dc) for r, c in part)
    result = _frame_info(shift(obj['cells']))
    result['cover'] = tuple((k, shift(part)) for k, part in obj['cover'])
    return result


def _place(rng, objects, palette):
    if len(objects) == 1:
        positions = [(0, 0)]
    else:
        first, second = objects
        gap = int(rng.integers(1, 5))
        widths = [o['box'][3] + 1 for o in objects]
        heights = [o['h'] for o in objects]
        if sum(widths) + gap <= 30 and rng.random() < .45:
            positions = [(int(rng.integers(0, 5)), 0),
                         (int(rng.integers(0, 5)), widths[0] + gap)]
        else:
            positions = [(0, int(rng.integers(0, 7))),
                         (heights[0] + gap, int(rng.integers(0, 7)))]
    min_r = min(r for r, _ in positions)
    min_c = min(c for _, c in positions)
    positions = [(r - min_r, c - min_c) for r, c in positions]
    span_h = max(r + obj['h'] for obj, (r, _) in zip(objects, positions))
    span_w = max(c + obj['box'][3] + 1 for obj, (_, c) in zip(objects, positions))
    if span_h > 30 or span_w > 30:
        return None
    h = span_h + int(rng.integers(0, min(8, 30 - span_h) + 1))
    w = span_w + int(rng.integers(0, min(8, 30 - span_w) + 1))
    if rng.random() < .12:
        h = 30
    if rng.random() < .12:
        w = 30
    area = sum(len(obj['cells']) + sum(len(part) for _, part in obj['cover'])
               for obj in objects)
    while h * w <= 2 * area:
        if h < 30 and (h <= w or w == 30):
            h += 1
        elif w < 30:
            w += 1
        else:
            return None
    top = int(rng.integers(0, h - span_h + 1))
    left = int(rng.integers(0, w - span_w + 1))
    objects = [_translate(obj, r + top, c + left)
               for obj, (r, c) in zip(objects, positions)]
    background, frame_color, neutral = palette
    inp = np.full((h, w), background, dtype=np.int8)
    for obj in objects:
        for r, c in obj['cells']:
            inp[r, c] = frame_color
        for _, part in obj['cover']:
            for r, c in part:
                inp[r, c] = neutral
    # Ground truth uses planted pieces, before the input-only parser runs.
    out = _render(inp.shape, background, frame_color, neutral, objects)
    return {'input': inp, 'output': out}


def _array_key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _structure(a, scene):
    coords = np.argwhere(a != scene['background'])
    low, high = coords.min(axis=0), coords.max(axis=0) + 1
    crop = a[low[0]:high[0], low[1]:high[1]]
    encoded = np.zeros(crop.shape, dtype=np.int8)
    encoded[crop == scene['frame_color']] = 1
    encoded[crop == scene['neutral']] = 2
    return _array_key(encoded)


def _checked_count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be between {lo} and {hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _checked_count(num_train, 'num_train', 2, 10)
    num_test = _checked_count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(int(seed))
    multiple = bool(rng.random() < .48)
    fixed_palette = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    vary_palette = bool(rng.random() < .55)
    initial_direction = int(rng.choice([-1, 1]))
    multi_evidence = int(rng.integers(0, 2))
    puzzle, seen, structures = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            evidence = ('basic', 'chain')[index] if split == 'train' and index < 2 else None
            if multiple and ((split == 'train' and index == multi_evidence) or
                             (split == 'test' and index == 0)):
                object_count = 2
            else:
                object_count = 1 + int(multiple and rng.random() < .38)
            for _ in range(60):
                palette = (tuple(int(x) for x in rng.choice(10, 3, replace=False))
                           if vary_palette else fixed_palette)
                objects = []
                for obj_index in range(object_count):
                    role = evidence if obj_index == 0 else None
                    if split == 'test':
                        pieces = int(rng.integers(7, 10))
                    elif role == 'basic':
                        pieces = int(rng.integers(5, 7))
                    elif role == 'chain':
                        pieces = 6
                    else:
                        pieces = int(rng.integers(2, 7))
                    direction = (initial_direction * (1 if index == 0 else -1)
                                 if split == 'train' and index < 2 and obj_index == 0
                                 else int(rng.choice([-1, 1])))
                    objects.append(_new_chamber(rng, object_count == 2, pieces,
                                                direction, role))
                rng.shuffle(objects)
                pair = _place(rng, objects, palette)
                if pair is None:
                    continue
                scene = _analyze_input(pair['input'])
                key, crop = _array_key(pair['input']), _structure(pair['input'], scene)
                if key in seen or crop in structures:
                    continue
                seen.add(key)
                structures.add(crop)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('could not create a distinct scene')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _without_parent(a, scene, parent):
    """Concrete counterfactuals used to detect decorative parent contributions."""
    bg, fc, neutral = (scene[k] for k in ('background', 'frame_color', 'neutral'))
    out = np.full(a.shape, bg, dtype=np.int8)
    for obj in scene['objects']:
        for r, c in obj['cells'] if parent == 'A' else obj['released']:
            out[r, c] = fc
        if parent == 'A':
            for kind, part in obj['cover']:
                for r, c in part:
                    out[r, c] = fc if kind == 'S' else neutral
        else:
            for r, c in obj['load']:
                out[r, c + obj['direction']] = neutral
    return out


def _pair_analysis(pair):
    a, b = _grid(pair['input']), _grid(pair['output'])
    assert a.shape == b.shape
    scene = _analyze_input(a)
    bg, fc, neutral = (scene[k] for k in ('background', 'frame_color', 'neutral'))
    assert np.array_equal(b, _render(a.shape, bg, fc, neutral, scene['objects']))
    assert not np.array_equal(a, b)
    squares = sum(len(obj['motion']['squares']) for obj in scene['objects'])
    assert np.count_nonzero(a != bg) == np.count_nonzero(b != bg)
    assert np.count_nonzero(b == fc) == np.count_nonzero(a == fc) + squares
    assert np.count_nonzero(b == neutral) == np.count_nonzero(a == neutral) - squares
    for obj in scene['objects']:
        for r, c in obj['motion']['squares']:
            assert b[r, c] == fc
        for r, c in obj['cells']:
            if r == obj['box'][2] or c == (obj['box'][3] if obj['direction'] == 1 else obj['box'][1]):
                assert b[r, c] == fc
    return scene


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, structures = set(), set()
    scenes = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            scene = _pair_analysis(pair)
            a, b = np.asarray(pair['input']), np.asarray(pair['output'])
            key, crop = _array_key(a), _structure(a, scene)
            assert key not in seen and crop not in structures
            seen.add(key)
            structures.add(crop)
            assert 1 <= len(scene['objects']) <= 2
            for obj in scene['objects']:
                kinds = [k for k, _ in obj['cover']]
                assert 'S' in kinds and any(k != 'S' for k in kinds)
                assert (2 <= len(kinds) <= 6 if split == 'train' else 7 <= len(kinds) <= 9)
            if split == 'train':
                assert not np.array_equal(b, _without_parent(a, scene, 'A'))
                assert not np.array_equal(b, _without_parent(a, scene, 'B'))
            scenes[split].append(scene)
    objects = [obj for scene in scenes['train'] for obj in scene['objects']]
    assert any(_evidence_basic(obj) for obj in objects)
    assert any(_evidence_chain(obj) for obj in objects)
    assert {obj['direction'] for obj in objects} == {-1, 1}
    if any(len(scene['objects']) == 2 for scene in scenes['test']):
        assert any(len(scene['objects']) == 2 for scene in scenes['train'])


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(np.asarray(p['input'])), _array_key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed structural audit plus all legal explicit count requests."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    dimensions, structures, geometries, counts = set(), set(), set(), set()
    frame_counts, orientations, palettes, backgrounds = set(), set(), set(), set()
    edges = large = moving_dependencies = propagated = wall_stops = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                before = a.copy()
                prediction = solve(a)
                assert np.array_equal(a, before), 'solve mutated input'
                assert np.array_equal(prediction, pair['output'])
                scene = _analyze_input(a)
                dimensions.add(a.shape)
                structures.add(_structure(a, scene))
                frame_counts.add(len(scene['objects']))
                palettes.add((scene['background'], scene['frame_color'], scene['neutral']))
                backgrounds.add(scene['background'])
                edges += any(np.any(edge != scene['background'])
                             for edge in (a[0], a[-1], a[:, 0], a[:, -1]))
                large += 30 in a.shape
                for obj in scene['objects']:
                    kinds = [k for k, _ in obj['cover']]
                    geometries.add((obj['h'], obj['length'], obj['offset'], obj['steps'], obj['direction']))
                    counts.add((kinds.count('S'), kinds.count('H'), kinds.count('V')))
                    orientations.update(kinds)
                    m = obj['motion']
                    moving_dependencies += m['moving_dependency']
                    propagated += m['depth'] >= 2
                    wall_stops += bool(m['wall_hit'])
    assert len(dimensions) >= 30 and len(structures) >= 150
    assert len(geometries) >= 60 and len(counts) >= 25
    assert frame_counts == {1, 2} and orientations == {'S', 'H', 'V'}
    assert len(palettes) >= 30 and len(backgrounds) >= 8
    assert edges > 0 and large > 0
    assert moving_dependencies >= 42 and propagated >= 42 and wall_stops >= 42
    assert _puzzle_key(generate_puzzle(8675309, 7, 3)) == _puzzle_key(generate_puzzle(8675309, 7, 3))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            assert (len(puzzle['train']), len(puzzle['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'pairs': 210, 'dimensions': len(dimensions),
            'cropped_structures': len(structures), 'frame_geometries': len(geometries),
            'primitive_count_triples': len(counts), 'palettes': len(palettes),
            'backgrounds': len(backgrounds), 'edge_pairs': edges, 'size_30_pairs': large,
            'moving_dependencies': moving_dependencies, 'propagated_blockages': propagated,
            'wall_stop_chambers': wall_stops}
