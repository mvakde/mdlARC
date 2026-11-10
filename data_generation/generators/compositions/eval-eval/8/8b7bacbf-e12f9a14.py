"""Hollow relay frames: cavity filling enables further aperture emission.

Only standard-library modules and NumPy are used. The solver reconstructs the
optical circuit from its input, without construction metadata or examples.
"""
from collections import Counter, deque
import json
from pathlib import Path

import numpy as np


D8 = ((-1, 0), (0, 1), (1, 0), (0, -1),
      (-1, -1), (-1, 1), (1, 1), (1, -1))
CARDINAL = D8[:4]
DIAGONAL = D8[4:]


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_)) or not low <= value <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _perimeter(box):
    r0, c0, r1, c1 = box
    return ([(r0, c) for c in range(c0, c1 + 1)]
            + [(r1, c) for c in range(c0, c1 + 1)]
            + [(r, c0) for r in range(r0 + 1, r1)]
            + [(r, c1) for r in range(r0 + 1, r1)])


def _normal(point, box):
    r, c = point
    r0, c0, r1, c1 = box
    return (-1 if r == r0 else 1 if r == r1 else 0,
            -1 if c == c0 else 1 if c == c1 else 0)


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("background must be the unique modal input color")
    left = set(map(tuple, np.argwhere(a != bg)))
    frames = []
    while left:
        root = min(left)
        left.remove(root)
        cells = {root}
        queue = [root]
        for r, c in queue:
            for dr, dc in D8:
                p = (r + dr, c + dc)
                if p in left:
                    left.remove(p)
                    cells.add(p)
                    queue.append(p)
        rows, cols = zip(*cells)
        box = (min(rows), min(cols), max(rows), max(cols))
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            raise ValueError("foreground component is not a complete hollow/source frame")
        perimeter = set(_perimeter(box))
        if not perimeter <= cells:
            raise ValueError("frame perimeter is incomplete")
        interior = a[r0 + 1:r1, c0 + 1:c1]
        paint = None
        if not np.all(interior == bg):
            if (interior.shape != (2, 2) or not np.all(interior == interior[0, 0])
                    or int(interior[0, 0]) == bg):
                raise ValueError("only initial sources have colored 2x2 interiors")
            paint = int(interior[0, 0])
        expected = perimeter
        if paint is not None:
            expected |= {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
        if cells != expected:
            raise ValueError("unexpected frame contents")
        frames.append(dict(box=box, paint=paint, ports=set()))
    if not frames or not any(f['paint'] is not None for f in frames):
        raise ValueError("a source frame is required")
    perimeter_counts = Counter(int(a[p]) for f in frames for p in _perimeter(f['box']))
    if len(perimeter_counts) != 2:
        raise ValueError("the perimeters must expose wall and cap roles")
    wall, cap = sorted(perimeter_counts, key=lambda c: (-perimeter_counts[c], c))
    for f in frames:
        perimeter = _perimeter(f['box'])
        f['ports'] = {p for p in perimeter if a[p] == cap}
        if 2 * len(f['ports']) >= len(perimeter):
            raise ValueError("each frame needs a strict wall majority")
        if f['paint'] in (wall, cap):
            raise ValueError("source payload cannot be a perimeter role")
    # Bounding-box separation also excludes nesting, not just foreground contact.
    for i, f in enumerate(frames):
        for other in frames[:i]:
            r0, c0, r1, c1 = f['box']
            s0, t0, s1, t1 = other['box']
            if not (r1 + 1 < s0 or s1 + 1 < r0 or c1 + 1 < t0 or t1 + 1 < c0):
                raise ValueError("frame bounding rectangles must be separated")
    return bg, wall, cap, frames


def _optics(a, bg, frames):
    boundary_owner = {p: i for i, f in enumerate(frames) for p in _perimeter(f['box'])}
    graph = [set() for _ in frames]
    rays = [[] for _ in frames]
    for i, f in enumerate(frames):
        for port in sorted(f['ports']):
            direction = _normal(port, f['box'])
            r, c = port
            dr, dc = direction
            points, hit, hit_point, receiver = [], None, None, None
            r, c = r + dr, c + dc
            while 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
                p = (r, c)
                if a[p] != bg:
                    hit_point = p
                    hit = boundary_owner.get(p)
                    if hit is None:
                        raise ValueError("a ray reached an interior before a perimeter")
                    target = frames[hit]
                    if (p in target['ports']
                            and _normal(p, target['box']) == (-dr, -dc)):
                        receiver = hit
                        graph[i].add(hit)
                    break
                points.append(p)
                r, c = r + dr, c + dc
            rays[i].append(dict(port=port, direction=direction, points=points,
                                hit=hit, hit_point=hit_point, receiver=receiver))
    return graph, rays


def _propagate(frames, graph):
    colors = [f['paint'] for f in frames]
    distances = [0 if c is not None else None for c in colors]
    queue = deque(i for i, c in enumerate(colors) if c is not None)
    while queue:
        i = queue.popleft()
        for j in sorted(graph[i]):
            if colors[j] is None:
                colors[j] = colors[i]
                distances[j] = distances[i] + 1
                queue.append(j)
            elif colors[j] != colors[i]:
                raise ValueError("different source paints reach the same frame")
    return colors, distances


def _render(a, frames, rays, colors, fill=True):
    result = a.copy()
    painted = {}
    for f, paths, color in zip(frames, rays, colors):
        if color is None:
            continue
        r0, c0, r1, c1 = f['box']
        if fill:
            result[r0 + 1:r1, c0 + 1:c1] = color
        for p in f['ports']:
            result[p] = color
        for path in paths:
            for p in path['points']:
                if p in painted and painted[p] != color:
                    raise ValueError("differently colored beams share a pixel")
                painted[p] = color
                result[p] = color
    return result


def _analyze(input_grid):
    a = _grid(input_grid)
    bg, wall, cap, frames = _parse(a)
    graph, rays = _optics(a, bg, frames)
    colors, distances = _propagate(frames, graph)
    output = _render(a, frames, rays, colors)
    return output, dict(background=bg, wall=wall, cap=cap, frames=frames,
                        graph=graph, rays=rays, colors=colors, distances=distances)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible geometry and source cores alone; never mutate input."""
    return _analyze(input_grid)[0]


def _features(info):
    frames, graph, rays = info['frames'], info['graph'], info['rays']
    colors, distances = info['colors'], info['distances']
    active = {i for i, c in enumerate(colors) if c is not None}
    hollow = {i for i, f in enumerate(frames) if f['paint'] is None}
    received = set()
    blocked, walls_blocked, wrong_facing, escapes, turns = 0, 0, 0, 0, 0
    for i in active:
        for ray in rays[i]:
            j = ray['receiver']
            if j is not None:
                received.add(sum(map(abs, ray['direction'])))
                if distances[j] == distances[i] + 1:
                    for nxt in rays[j]:
                        k = nxt['receiver']
                        if (k is not None and distances[k] == distances[j] + 1
                                and nxt['direction'] != ray['direction']):
                            turns += 1
            if ray['hit'] in hollow - active and j is None:
                blocked += 1
                if ray['hit_point'] in frames[ray['hit']]['ports']:
                    wrong_facing += 1
                else:
                    walls_blocked += 1
            if ray['hit'] is None and ray['points']:
                escapes += 1
    return dict(sources=len(frames) - len(hollow), frames=len(frames),
                active_hollow=len(active & hollow), inactive=len(hollow - active),
                depth=max((distances[i] for i in active & hollow), default=0),
                turns=turns, blocked=blocked, walls_blocked=walls_blocked,
                wrong_facing=wrong_facing, escapes=escapes, received=received,
                degrees=tuple(sorted(len(g) for g in graph)),
                branches=sum(len(graph[i]) >= 3 for i in active),
                source_colors={f['paint'] for f in frames if f['paint'] is not None},
                cavity_shapes={(f['box'][2] - f['box'][0] - 1,
                                f['box'][3] - f['box'][1] - 1) for f in frames},
                port_counts=tuple(sorted(len(f['ports']) for f in frames)),
                directions={ray['direction'] for paths in rays for ray in paths})


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _clear(box, frames, shape):
    r0, c0, r1, c1 = box
    if not (0 <= r0 < r1 < shape[0] and 0 <= c0 < c1 < shape[1]):
        return False
    for f in frames:
        s0, t0, s1, t1 = f['box']
        if not (r1 + 1 < s0 or s1 + 1 < r0 or c1 + 1 < t0 or t1 + 1 < c0):
            return False
    return True


def _new_frame(box, paint=None, owner=None):
    return dict(box=box, paint=paint, owner=paint if paint is not None else owner, ports=set())


def _free_ports(frame, directions=D8):
    edge = _perimeter(frame['box'])
    if 2 * (len(frame['ports']) + 1) >= len(edge):
        return []
    return [p for p in edge if p not in frame['ports']
            and _normal(p, frame['box']) in directions]


def _place(rng, frames, shape, source=False, paint=None):
    for _ in range(100):
        h, w = (4, 4) if source else (int(rng.integers(3, 8)), int(rng.integers(3, 8)))
        r = int(rng.integers(shape[0] - h + 1))
        c = int(rng.integers(shape[1] - w + 1))
        box = (r, c, r + h - 1, c + w - 1)
        if _clear(box, frames, shape):
            frames.append(_new_frame(box, paint if source else None))
            return len(frames) - 1
    raise ValueError("no room for a separate frame")


def _attach(rng, frames, parent, shape, directions, productive=True):
    available = _free_ports(frames[parent], directions)
    if not available:
        raise ValueError("parent lacks a free aperture")
    for _ in range(150):
        p = _pick(rng, available)
        dr, dc = _normal(p, frames[parent]['box'])
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        distance = int(rng.integers(2, 7))
        q = (p[0] + distance * dr, p[1] + distance * dc)
        # q lies on the child's facing wall, at an arbitrary side offset.
        rr = h - 1 if dr < 0 else 0 if dr > 0 else int(rng.integers(1, h - 1))
        cc = w - 1 if dc < 0 else 0 if dc > 0 else int(rng.integers(1, w - 1))
        box = (q[0] - rr, q[1] - cc, q[0] - rr + h - 1, q[1] - cc + w - 1)
        if not _clear(box, frames, shape):
            continue
        # No existing frame may occlude the intended link.
        if any(f['box'][0] <= p[0] + k * dr <= f['box'][2]
               and f['box'][1] <= p[1] + k * dc <= f['box'][3]
               for k in range(1, distance) for f in frames):
            continue
        frames[parent]['ports'].add(p)
        child = _new_frame(box, owner=frames[parent]['owner'] if productive else None)
        if productive:
            child['ports'].add(q)
        frames.append(child)
        return len(frames) - 1, (dr, dc), q
    raise ValueError("attachment placement exhausted")


def _canvas(frames, shape):
    a = np.zeros(shape, dtype=np.int8)
    for f in frames:
        for p in _perimeter(f['box']):
            a[p] = 2 if p in f['ports'] else 1
        if f['paint'] is not None:
            r0, c0, r1, c1 = f['box']
            a[r0 + 1:r1, c0 + 1:c1] = f['paint']
    return a


def _facing_lesson_blocker(rng, frames, shape):
    """Two parallel beams: one meets a diagonal-facing cap, one a solid wall."""
    candidates = []
    for i, frame in enumerate(frames):
        if frame['owner'] is None or 2 * (len(frame['ports']) + 2) >= len(_perimeter(frame['box'])):
            continue
        free = set(_free_ports(frame, CARDINAL))
        for p in sorted(free):
            direction = _normal(p, frame['box'])
            adjacent = (p[0] + 1, p[1]) if direction[1] else (p[0], p[1] + 1)
            if adjacent in free and _normal(adjacent, frame['box']) == direction:
                candidates.append((i, p, adjacent, direction))
    if not candidates:
        raise ValueError("no paired side apertures for the facing lesson")
    for _ in range(150):
        i, p, adjacent, (dr, dc) = _pick(rng, candidates)
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        distance = int(rng.integers(2, 7))
        q = (p[0] + distance * dr, p[1] + distance * dc)
        rr, cc = (h - 1 if dr < 0 else 0), (w - 1 if dc < 0 else 0)
        box = (q[0] - rr, q[1] - cc, q[0] - rr + h - 1, q[1] - cc + w - 1)
        if not _clear(box, frames, shape):
            continue
        if any(f['box'][0] <= start[0] + k * dr <= f['box'][2]
               and f['box'][1] <= start[1] + k * dc <= f['box'][3]
               for start in (p, adjacent) for k in range(1, distance) for f in frames):
            continue
        frames[i]['ports'].update((p, adjacent))
        blocker = _new_frame(box)
        blocker['ports'].add(q)
        frames.append(blocker)
        return
    raise ValueError("could not place the facing lesson blocker")


def _construction_output(frames, a):
    """Draw the planned ownership directly, without discovering/propagating it.

    This metadata-based constructor deliberately differs from the solver's
    component parser and BFS. Agreement also checks accidental optical links.
    """
    answer = a.copy()
    occupied = {p: i for i, f in enumerate(frames) for p in _perimeter(f['box'])}
    beam_colors = {}
    for i, f in enumerate(frames):
        paint = f['owner']
        if paint is None:
            continue
        r0, c0, r1, c1 = f['box']
        answer[r0 + 1:r1, c0 + 1:c1] = paint
        for port in sorted(f['ports']):
            answer[port] = paint
            dr, dc = _normal(port, f['box'])
            for distance in range(1, max(a.shape) + 1):
                p = (port[0] + dr * distance, port[1] + dc * distance)
                if not (0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]):
                    break
                if a[p] != 0:
                    j = occupied[p]
                    target = frames[j]
                    if (p in target['ports'] and _normal(p, target['box']) == (-dr, -dc)
                            and target['owner'] != paint):
                        raise ValueError("extra gate creates unintended ownership")
                    break
                if p in beam_colors and beam_colors[p] != paint:
                    raise ValueError("planned beams have incompatible colors")
                beam_colors[p] = paint
                answer[p] = paint
    return answer


def _certify(frames, shape, depth, directions=None):
    a = _canvas(frames, shape)
    b = _construction_output(frames, a)
    prediction, info = _analyze(a)
    if not np.array_equal(prediction, b):
        raise ValueError("visible optics do not realize the planned circuit")
    f = _features(info)
    if f['depth'] != depth or f['turns'] < 1 or f['inactive'] < 1 or f['blocked'] < 1:
        raise ValueError("curriculum witness lost in final geometry")
    if directions is not None and not directions <= f['received']:
        raise ValueError("required side/corner reception missing")
    if directions == {1, 2} and not (f['wrong_facing'] and f['walls_blocked']):
        raise ValueError("facing versus solid-wall reception is not demonstrated")
    if not f['escapes']:
        raise ValueError("a visible escaping beam is needed")
    return a, b, info


def _random_scene(rng, depth, lesson, second_source, repeat_paint=False):
    shape = tuple(int(x) for x in rng.integers(24 if second_source or depth > 2 else 20, 31, 2))
    frames = []
    root = _place(rng, frames, shape, source=True, paint=3)
    parent, prior = root, None
    chain = [root]
    for step in range(depth):
        directions = (CARDINAL if lesson == 'side' else DIAGONAL) if step == 0 else D8
        if step == 1:
            directions = tuple(d for d in CARDINAL if d != prior and d != (-prior[0], -prior[1]))
        child, prior, _ = _attach(rng, frames, parent, shape, directions)
        chain.append(child)
        parent = child
    # A branch may share a relay rather than supplying a separate mini-puzzle.
    if rng.random() < .55:
        branch_parent = chain[int(rng.integers(0, len(chain) - 1))]
        _attach(rng, frames, branch_parent, shape, D8)
    if second_source:
        root2 = _place(rng, frames, shape, source=True, paint=3 if repeat_paint else 4)
        _attach(rng, frames, root2, shape, D8)
    if lesson == 'corner':
        _facing_lesson_blocker(rng, frames, shape)
    else:
        blocker_parent = _pick(rng, [i for i, f in enumerate(frames) if f['owner'] is not None])
        blocker, _, receiving_wall = _attach(rng, frames, blocker_parent, shape, D8, productive=False)
        # Give the unactivated blocker its own unused apertures.
        options = [p for p in _free_ports(frames[blocker]) if p != receiving_wall]
        if options:
            frames[blocker]['ports'].add(_pick(rng, options))
    if rng.random() < .45:
        _place(rng, frames, shape)
    # Add ports only when the complete visible circuit keeps the curriculum.
    # First seek an escape aperture, then optionally add other ports.
    for _ in range(50):
        active_ids = [i for i, f in enumerate(frames) if f['owner'] is not None]
        i = _pick(rng, active_ids)
        options = _free_ports(frames[i])
        if not options:
            continue
        port = _pick(rng, options)
        frames[i]['ports'].add(port)
        try:
            _certify(frames, shape, depth, {1, 2} if lesson == 'corner' else {1})
            break
        except ValueError:
            frames[i]['ports'].remove(port)
    else:
        raise ValueError("could not add an unambiguous escaping aperture")
    for _ in range(int(rng.integers(0, 6))):
        i = int(rng.integers(len(frames)))
        options = _free_ports(frames[i])
        if not options:
            continue
        port = _pick(rng, options)
        frames[i]['ports'].add(port)
        try:
            _certify(frames, shape, depth, {1, 2} if lesson == 'corner' else {1})
        except ValueError:
            frames[i]['ports'].remove(port)
    return _certify(frames, shape, depth, {1, 2} if lesson == 'corner' else {1})


def _fallback(rng, depth, lesson, second_source, repeat_paint=False):
    """Sparse safe coordinates, with a varying blocker height."""
    frames = [_new_frame((2, 2, 5, 5), paint=3)]
    if lesson == 'corner':
        boxes = [(8, 8, 11, 11), (8, 17, 11, 20), (17, 17, 20, 20), (17, 25, 20, 28)]
        links = [((5, 5), (8, 8)), ((9, 11), (9, 17)),
                 ((11, 18), (17, 18)), ((18, 20), (18, 25))]
    else:
        boxes = [(2, 10, 5, 13), (10, 10, 13, 13), (10, 20, 13, 23), (20, 20, 23, 23)]
        links = [((3, 5), (3, 10)), ((5, 11), (10, 11)),
                 ((11, 13), (11, 20)), ((13, 21), (20, 21))]
    for box in boxes[:depth]:
        frames.append(_new_frame(box, owner=3))
    for i, (p, q) in enumerate(links[:depth]):
        frames[i]['ports'].add(p)
        frames[i + 1]['ports'].add(q)
    # An escaping diagonal port is far from every other module.
    frames[0]['ports'].add((2, 2))
    frames[0]['ports'].add((5, 3))
    blocker = _new_frame((10, 3, 13, 6) if lesson == 'corner' else (10, 2, 13, 5))
    if lesson == 'corner':
        frames[0]['ports'].add((5, 4))
        blocker['ports'].add((10, 3))
    blocker['ports'].add((13, blocker['box'][3]))
    frames.append(blocker)
    if second_source:
        root = _new_frame((23, 2, 26, 5), paint=3 if repeat_paint else 4)
        target = _new_frame((23, 11, 26, 14), owner=root['paint'])
        root['ports'].add((24, 5))
        target['ports'].add((24, 11))
        frames.extend([root, target])
    # Change isolated/blocker extents without altering the blocked hit.
    blocker['box'] = (10, blocker['box'][1], 13 + int(rng.integers(3)), blocker['box'][3])
    blocker['ports'] = {(blocker['box'][2], blocker['box'][3])}
    if lesson == 'corner':
        blocker['ports'].add((10, 3))
    return _certify(frames, (30, 30), depth, {1, 2} if lesson == 'corner' else {1})


def _pair(rng, depth, lesson, second_source, palette, orientation, repeat_paint=False):
    for _ in range(65):
        try:
            a, b, _ = _random_scene(rng, depth, lesson, second_source, repeat_paint)
            fallback = False
            break
        except ValueError:
            continue
    else:
        a, b, _ = _fallback(rng, depth, lesson, second_source, repeat_paint)
        fallback = True
    lookup = np.array(palette, dtype=np.int8)
    a, b = lookup[a], lookup[b]
    rotation, reflected = orientation
    a, b = np.rot90(a, rotation), np.rot90(b, rotation)
    if reflected:
        a, b = np.fliplr(a), np.fliplr(b)
    a, b = a.copy(), b.copy()
    prediction, info = _analyze(a)
    if not np.array_equal(prediction, b) or _features(info)['depth'] != depth:
        raise ValueError("transform changed circuit semantics")
    return dict(input=a, output=b), fallback


def _key(a):
    return tuple(a.shape), a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base_palette = [int(c) for c in rng.permutation(10)]
    shared_orientation = bool(rng.integers(2))
    main_orientation = (int(rng.integers(4)), bool(rng.integers(2)))
    repeated_sources = rng.random() < .3
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            if split == 'train':
                depth = 2
                lesson = 'side' if index == 0 else 'corner' if index == 1 else _pick(rng, ('side', 'corner'))
                second = index == 1 or (index > 1 and rng.random() < .4)
            else:
                depth = 4 if rng.random() < .2 else 3
                lesson = _pick(rng, ('side', 'corner'))
                second = rng.random() < .4
            repeat = repeated_sources and second and not (split == 'train' and index == 1)
            palette = base_palette if rng.random() < .5 else [int(c) for c in rng.permutation(10)]
            if split == 'train' and index == 1:
                # Geometry reveals roles; teach that the literal palette is not the rule.
                palette = base_palette[1:] + base_palette[:1]
            orientation = main_orientation if shared_orientation else (int(rng.integers(4)), bool(rng.integers(2)))
            for _ in range(12):
                pair, _ = _pair(rng, depth, lesson, second, palette, orientation, repeat)
                key = _key(pair['input'])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise ValueError("could not generate a distinct scene")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _cardinal_prediction(a, info):
    graph = [{ray['receiver'] for ray in paths if ray['receiver'] is not None
              and sum(map(abs, ray['direction'])) == 1} for paths in info['rays']]
    colors, _ = _propagate(info['frames'], graph)
    return _render(a, info['frames'], info['rays'], colors)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    witnesses = Counter()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            predicted, info = _analyze(a)
            assert np.array_equal(predicted, b), "incorrect relay or beam output"
            f = _features(info)
            features[split].append(f)
            assert f['active_hollow'] >= 2 and f['inactive'] >= 1
            assert f['blocked'] >= 1 and f['escapes'] >= 1 and f['turns'] >= 1
            bg = info['background']
            assert np.any(a != b)
            changed = set(map(tuple, np.argwhere(a != b)))
            allowed = set()
            for frame, paths, color in zip(info['frames'], info['rays'], info['colors']):
                r0, c0, r1, c1 = frame['box']
                assert all(b[p] == a[p] for p in _perimeter(frame['box']) if a[p] == info['wall'])
                if color is None:
                    assert np.array_equal(a[r0:r1 + 1, c0:c1 + 1], b[r0:r1 + 1, c0:c1 + 1])
                    continue
                assert np.all(b[r0 + 1:r1, c0 + 1:c1] == color)
                if frame['paint'] is not None:
                    assert np.all(a[r0 + 1:r1, c0 + 1:c1] == b[r0 + 1:r1, c0 + 1:c1])
                allowed.update(frame['ports'])
                allowed.update((r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))
                allowed.update(p for path in paths for p in path['points'])
            assert changed <= allowed
            assert all(a[p] == bg or a[p] == info['cap'] for p in changed)
            if split == 'train':
                assert f['depth'] == 2
                frames, rays, colors = info['frames'], info['rays'], info['colors']
                sources_only = [frame['paint'] for frame in frames]
                witnesses['no_relay'] += not np.array_equal(_render(a, frames, rays, sources_only), b)
                witnesses['no_fill'] += not np.array_equal(_render(a, frames, rays, colors, fill=False), b)
                witnesses['all_chambers'] += f['inactive'] > 0
                witnesses['wall_hit'] += f['walls_blocked'] > 0
                witnesses['any_cap'] += f['wrong_facing'] > 0
                witnesses['cardinal_only'] += not np.array_equal(_cardinal_prediction(a, info), b)
            else:
                assert f['depth'] in (3, 4)
    assert all(witnesses[k] for k in ('no_relay', 'no_fill', 'all_chambers', 'wall_hit', 'any_cap', 'cardinal_only'))
    assert any(f['sources'] >= 2 and len(f['source_colors']) >= 2 for f in features['train'])
    assert {1, 2} <= set().union(*(f['received'] for f in features['train']))
    assert min(f['depth'] for f in features['test']) > max(f['depth'] for f in features['train'])


def _canonical_structure(a):
    """Ignore palette, margins, rotation and reflection for variation counts."""
    bg = int(np.bincount(a.ravel(), minlength=10).argmax())
    rr, cc = np.where(a != bg)
    cropped = a[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    keys = []
    for k in range(4):
        rotated = np.rot90(cropped, k)
        for transformed in (rotated, np.fliplr(rotated)):
            roles = {bg: 0}
            values = tuple(roles.setdefault(int(c), len(roles)) for c in transformed.ravel())
            keys.append((tuple(transformed.shape), values))
    return min(keys)


def _family_fixtures():
    """Known geometric edge cases outside the sampler's teaching requirements."""
    # A four-node cycle. Filling three relays completes every chamber; revisits
    # have no scheduling effect. Two equal-paint sources can share the cycle.
    cycle = [_new_frame((1, 1, 4, 4), paint=3),
             _new_frame((1, 10, 4, 13), owner=3),
             _new_frame((10, 10, 13, 13), owner=3),
             _new_frame((10, 1, 13, 4), owner=3)]
    for i, p, j, q in ((0, (2, 4), 1, (2, 10)), (1, (4, 11), 2, (10, 11)),
                        (2, (11, 10), 3, (11, 4)), (3, (10, 2), 0, (4, 2))):
        cycle[i]['ports'].add(p)
        cycle[j]['ports'].add(q)
    a = _canvas(cycle, (16, 16))
    assert np.array_equal(solve(a), _construction_output(cycle, a))
    cycle[2]['paint'] = 3
    a = _canvas(cycle, (16, 16))
    assert np.array_equal(solve(a), _construction_output(cycle, a))
    cycle[2]['paint'] = 4
    try:
        solve(_canvas(cycle, (16, 16)))
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting sources accepted on a cycle")

    # A ray hits an actual aperture, but the corner's normal does not face it.
    wrong = [_new_frame((4, 1, 7, 4), paint=3), _new_frame((5, 9, 8, 12))]
    wrong[0]['ports'].add((5, 4))
    wrong[1]['ports'].add((5, 9))
    a = _canvas(wrong, (13, 17))
    b = solve(a)
    assert np.array_equal(b, _construction_output(wrong, a))
    assert np.all(b[6:8, 10:12] == 0) and b[5, 9] == 2

    # Larger-than-sampled chamber, border frame, clipped outgoing beam.
    large = [_new_frame((0, 0, 3, 3), paint=3), _new_frame((0, 8, 8, 16), owner=3)]
    large[0]['ports'].update(((1, 3), (0, 0)))
    large[1]['ports'].update(((1, 8), (6, 16)))
    a = _canvas(large, (13, 21))
    b = solve(a)
    assert np.array_equal(b, _construction_output(large, a))
    assert np.all(b[1:8, 9:16] == 3) and b[0, 0] == 3

    # Sources need no reachable chamber; same-paint crossings remain straight.
    crossing = [_new_frame((1, 1, 4, 4), paint=3), _new_frame((7, 8, 10, 11), paint=3)]
    crossing[0]['ports'].add((2, 4))
    crossing[1]['ports'].add((7, 9))
    a = _canvas(crossing, (14, 17))
    assert np.array_equal(solve(a), _construction_output(crossing, a))
    crossing[1]['paint'] = crossing[1]['owner'] = 4
    try:
        solve(_canvas(crossing, (14, 17)))
    except ValueError:
        pass
    else:
        raise AssertionError("different paints accepted on a shared beam pixel")


def self_test():
    """Bounded family tests plus a measured 42-seed structural audit."""
    _family_fixtures()
    shapes, cavities, topologies, structures, directions = set(), set(), set(), set(), set()
    backgrounds, walls, caps, source_counts, depths = set(), set(), set(), set(), set()
    branch_pairs, repeated_pairs = 0, 0
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                _, info = _analyze(a)
                f = _features(info)
                shapes.add(tuple(a.shape))
                cavities.update(f['cavity_shapes'])
                topologies.add((f['sources'], f['active_hollow'], f['inactive'], f['degrees']))
                structures.add(_canonical_structure(a))
                directions.update(f['directions'])
                source_counts.add(f['sources'])
                depths.add(f['depth'])
                branch_pairs += f['branches'] > 0
                repeated_pairs += len(f['source_colors']) < f['sources']
                backgrounds.add(info['background'])
                walls.add(info['wall'])
                caps.add(info['cap'])
                assert np.array_equal(solve(a.tolist()), b)
        if seed < 3:
            original = puzzle['test'][0]
            a, b = original['input'], original['output']
            palette = np.random.default_rng(seed + 100).permutation(10)
            assert np.array_equal(solve(palette[a]), palette[b])
            assert np.array_equal(solve(np.rot90(a)), np.rot90(b))
            assert np.array_equal(solve(np.fliplr(a)), np.fliplr(b))
            assert np.array_equal(a, original['input'])
    assert len(shapes) >= 25 and len(cavities) >= 20
    assert len(topologies) >= 15 and len(structures) >= 80
    assert source_counts == {1, 2} and {2, 3, 4} <= depths
    assert directions == set(D8) and branch_pairs >= 10
    assert min(len(backgrounds), len(walls), len(caps)) >= 8
    assert repeated_pairs > 0
    def puzzle_key(p):
        return tuple((_key(q['input']), _key(q['output'])) for s in ('train', 'test') for q in p[s])
    assert puzzle_key(puzzles[0]) == puzzle_key(generate_puzzle(0))
    assert len({puzzle_key(p) for p in puzzles}) == 42
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(500 + nt * 10 + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in q.items()} for q in puzzles[0][s]] for s in puzzles[0]}
    q = bad['test'][0]
    r, c = np.argwhere(q['input'] != q['output'])[0]
    q['output'][r, c] = q['input'][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("partial answer accepted")
    report = dict(seeds=42, pairs=210, shapes=len(shapes), cavity_shapes=len(cavities),
                  optical_topologies=len(topologies), canonical_structures=len(structures),
                  branch_pairs=branch_pairs, repeated_source_pairs=repeated_pairs,
                  source_counts=sorted(source_counts), depths=sorted(depths),
                  background_colors=sorted(backgrounds), wall_colors=sorted(walls), cap_colors=sorted(caps),
                  independent_solver=False, blind_prediction=False)
    return report


if __name__ == '__main__':
    report = self_test()
    Path('scratch').mkdir(exist_ok=True)
    Path('scratch/self_test.json').write_text(json.dumps(report, separators=(',', ':')) + '\n')
    print(json.dumps(report, separators=(',', ':')))
