"""Beam-guided docking: a reflected pilot transports a rigid connector stencil.

No parent module, seed, training labels, or generation metadata enters solve().
The fixed objects remain, each mobile piece translates, and its travelled beam
is drawn up to the first complete, collision-free receiver match.
"""
from __future__ import annotations

from collections import Counter
import json
import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, found = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _normal(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _l_info(cells):
    if len(cells) != 3:
        return None
    t, l, b, r = _box(cells)
    if b - t != 1 or r - l != 1:
        return None
    missing = next(iter({(a, c) for a in (t, b) for c in (l, r)} - cells))
    elbow = (t + b - missing[0], l + r - missing[1])
    return elbow, (elbow[0] - missing[0], elbow[1] - missing[1])


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _decode(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= grid.size / 2:
        raise ValueError("background must be the unique majority color")
    objects = []
    for cells in _components(map(tuple, np.argwhere(grid != bg).tolist())):
        colors = {int(grid[q]) for q in cells}
        objects.append(dict(cells=cells, colors=colors))
    carriers = [p for p in objects if len(p['colors']) > 1]
    if len(carriers) < 3:
        raise ValueError("one mobile piece and at least two receivers are required")
    common = set.intersection(*(p['colors'] for p in carriers))
    if len(common) != 1:
        raise ValueError("the common connector hue must be visible uniquely")
    marker = common.pop()
    mobiles, receivers, mirrors = [], [], []
    for p in objects:
        cells, colors = p['cells'], p['colors']
        if len(colors) == 1:
            t, l, b, r = _box(cells)
            if (len(cells) not in (1, 3, 5, 7, 9)
                    or (t != b and l != r)
                    or len(cells) != max(b - t, r - l) + 1):
                raise ValueError("a free object must be an odd straight mirror")
            if marker in colors:
                raise ValueError("free connector pixels are not mirrors")
            mirrors.append(dict(cells=cells, color=next(iter(colors)),
                                mid=((t + b) // 2, (l + r) // 2),
                                axes=(0, 1) if len(cells) == 1 else ((0,) if t == b else (1,))))
            continue
        pins = {q for q in cells if grid[q] == marker}
        if not pins or len(colors) not in (2, 3):
            raise ValueError("receiver/mobile palettes must have two/three colors")
        if len(colors) == 2:
            body = cells - pins
            if len(_components(body)) != 1:
                raise ValueError("receiver body must be connected")
            receivers.append(dict(cells=cells, pins=pins, body=body,
                                  color=next(iter(colors - {marker}))))
        else:
            tags = []
            for shade in sorted(colors - {marker}):
                tag = {q for q in cells if grid[q] == shade}
                info = _l_info(tag)
                if info is not None:
                    tags.append((tag, shade, info))
            if len(tags) != 1:
                raise ValueError("mobile piece must have a unique monochromatic L pilot")
            tag, shade, (elbow, direction) = tags[0]
            body = cells - pins - tag
            if not body or len(_components(body)) != 1:
                raise ValueError("mobile payload must be connected")
            mobiles.append(dict(cells=cells, pins=pins, body=body, tag=tag,
                                color=shade, elbow=elbow, direction=direction,
                                mapping={q: int(grid[q]) for q in cells}))
    if not mobiles or len(receivers) < 2:
        raise ValueError("missing mobile piece or receivers")
    return grid, bg, marker, mobiles, receivers, mirrors


def _contact(mirror, position, direction):
    for axis in mirror['axes']:
        other = 1 - axis
        if (position[other] == mirror['mid'][other]
                and mirror['mid'][axis] - position[axis] == direction[axis]):
            return axis
    return None


def _landings(mobile, receivers, fixed, shape, geometry=True):
    result = {}
    for index, receiver in enumerate(receivers):
        if geometry and _normal(mobile['pins']) != _normal(receiver['pins']):
            continue
        if not geometry and len(mobile['pins']) != len(receiver['pins']):
            continue
        a, b = min(mobile['pins']), min(receiver['pins'])
        delta = (b[0] - a[0], b[1] - a[1])
        moved = _shift(mobile['cells'], delta)
        # Geometry-disabled ablation deliberately relaxes the shared-pin test.
        allowed = receiver['pins'] if geometry else _shift(mobile['pins'], delta)
        if not _inside(moved, shape) or moved & (fixed - allowed):
            continue
        point = (mobile['elbow'][0] + delta[0], mobile['elbow'][1] + delta[1])
        if point in result:
            raise ValueError("competing receiver matches at one pilot position")
        result[point] = (index, delta)
    return result


def _trace(mobile, receivers, mirrors, fixed, shape, *, geometry=True, reflect=True):
    landings = _landings(mobile, receivers, fixed, shape, geometry)
    direction = mobile['direction']
    position, color = mobile['elbow'], mobile['color']
    visited, path, contacts = set(), [], []
    while True:
        position = (position[0] + direction[0], position[1] + direction[1])
        if not _inside({position}, shape):
            raise ValueError("pilot left the canvas without docking")
        state = position, direction
        if state in visited:
            raise ValueError("pilot loop")
        visited.add(state)
        if position in fixed:
            raise ValueError("pilot struck an object instead of a midpoint face")
        hits = [(i, _contact(m, position, direction)) for i, m in enumerate(mirrors)] if reflect else []
        hits = [(i, a) for i, a in hits if a is not None]
        if len(hits) > 1:
            raise ValueError("ambiguous mirror contact")
        if hits:
            index, axis = hits[0]
            old = color
            color = mirrors[index]['color']
            direction = tuple(-v if j == axis else v for j, v in enumerate(direction))
            contacts.append((index, axis, position, old, color))
        path.append((position, color))
        if position in landings:
            receiver, delta = landings[position]
            return dict(path=path, contacts=contacts, receiver=receiver, delta=delta,
                        end=position, launch=mobile['direction'])


def _predict(value, *, geometry=True, reflect=True):
    grid, bg, marker, mobiles, receivers, mirrors = _decode(value)
    fixed = set().union(*(p['cells'] for p in receivers + mirrors))
    routes = [_trace(m, receivers, mirrors, fixed, grid.shape,
                     geometry=geometry, reflect=reflect) for m in mobiles]
    if len({r['receiver'] for r in routes}) != len(routes):
        raise ValueError("two mobile pieces claim the same receiver")
    answer = np.full(grid.shape, bg, dtype=np.int8)
    for q in fixed:
        answer[q] = grid[q]
    painted = {}
    for route in routes:
        for point, color in route['path']:
            if point in painted and painted[point] != color:
                raise ValueError("conflicting beam colors")
            painted[point] = color
            answer[point] = color
    placed = set()
    for mobile, route in zip(mobiles, routes):
        moved = _shift(mobile['cells'], route['delta'])
        if placed & moved:
            raise ValueError("docked pieces overlap")
        placed |= moved
        for q, color in mobile['mapping'].items():
            point = q[0] + route['delta'][0], q[1] + route['delta'][1]
            answer[point] = color
    return answer, dict(bg=bg, marker=marker, mobiles=mobiles, receivers=receivers,
                        mirrors=mirrors, routes=routes, fixed=fixed, painted=painted)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer local roles, trace pilots, match complete pins, and render docking."""
    return _predict(input_grid)[0]


def _sym(point, turns, flip):
    r, c = point
    if flip:
        c = -c
    for _ in range(turns):
        r, c = c, -r
    return r, c


def _rect(t, l, h, w):
    return {(r, c) for r in range(t, t + h) for c in range(l, l + w)}


def _skeleton(rng, t, l, h, w, targets):
    """Vary solid, framed, open, and organically grown connected payloads."""
    full = _rect(t, l, h, w)
    kind = int(rng.integers(5))
    if kind == 0 or min(h, w) == 1:
        body = set(full)
    elif kind == 1:
        body = {q for q in full if q[0] in (t, t + h - 1) or q[1] in (l, l + w - 1)}
    elif kind == 2:
        body = {q for q in full if q[0] == t or q[1] in (l, l + w - 1)}
    elif kind == 3:
        body = {q for q in full if q[0] == t + h - 1 or q[1] in (l, l + w - 1)}
    else:
        body = {min(full)}
        target = int(rng.integers(max(3, h * w // 3), h * w + 1))
        while len(body) < min(target, len(full)):
            frontier = sorted({(r + dr, c + dc) for r, c in body for dr, dc in N4} & full - body)
            body.add(frontier[int(rng.integers(len(frontier)))])
    for target in sorted(targets):
        a, c = min(body, key=lambda q: (abs(q[0] - target[0]) + abs(q[1] - target[1]), q))
        axes = (0, 1) if rng.integers(2) else (1, 0)
        for axis in axes:
            if axis == 0:
                while a != target[0]:
                    a += 1 if target[0] > a else -1
                    body.add((a, c))
            else:
                while c != target[1]:
                    c += 1 if target[1] > c else -1
                    body.add((a, c))
    return body


def _mobile_shape(rng):
    h, w = int(rng.integers(2, 6)), int(rng.integers(4, 8))
    # Two and three sparse pins: repeated area cannot identify a receiver.
    n = 2 if rng.random() < .75 else 3
    columns = sorted(int(v) for v in rng.choice(np.arange(-w + 2, 2), n, replace=False))
    if columns[-1] - columns[0] < 2:
        columns[0], columns[-1] = -w + 2, 1
    offsets = [int(rng.integers(-1, 1)) for _ in columns] if rng.random() < .55 else [0] * n
    pins = {(-h - 2 + offset, c) for c, offset in zip(columns, offsets)}
    targets = {(-h - 1, c) for c in columns} | {(-2, 0)}
    body = _skeleton(rng, -h - 1, -w + 2, h, w, targets)
    body |= {(r, c) for pr, c in pins for r in range(pr + 1, -h - 1)}
    tag = {(0, 0), (-1, 0), (0, -1)}
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    trans = lambda cells: {_sym(q, k, flip) for q in cells}
    return dict(body=trans(body), tag=trans(tag), pins=trans(pins),
                direction=_sym((1, 1), k, flip), h=h, columns=columns,
                offsets=offsets, k=k, flip=flip)


def _receiver_shape(rng, mobile, wrong=False):
    columns = list(mobile['columns'])
    if wrong:
        # Change spacing while retaining area, the same minimum pin, and face.
        columns[-1] += int(rng.choice((-1, 1, 2)))
        if len(set(columns)) != len(columns) or sorted(columns) == mobile['columns']:
            columns[-1] = mobile['columns'][-1] + 2
    row = -mobile['h'] - 2
    offsets = mobile.get('offsets', [0] * len(columns))
    pins = {(row + offset, c) for c, offset in zip(columns, offsets)}
    height = int(rng.integers(1, 4))
    left = min(columns) - int(rng.integers(2))
    width = max(columns) - left + 1 + int(rng.integers(2))
    bottom = min(r for r, c in pins) - 1
    body = _skeleton(rng, bottom - height + 1, left, height, width,
                     {(bottom, c) for c in columns})
    body |= {(r, c) for pr, c in pins for r in range(bottom + 1, pr)}
    trans = lambda cells: {_sym(q, mobile['k'], mobile['flip']) for q in cells}
    return dict(body=trans(body), pins=trans(pins))


def _mapping(body, pins, body_color, marker, tag=None, pilot=None):
    result = {q: body_color for q in body}
    result.update({q: marker for q in pins})
    if tag is not None:
        result.update({q: pilot for q in tag})
    return result


def _place_mapping(mapping, position):
    return {(r + position[0], c + position[1]): v for (r, c), v in mapping.items()}


def _add_object(objects, mapping, shape):
    cells = set(mapping)
    if not _inside(cells, shape):
        return False
    occupied = set().union(*(set(p) for p in objects)) if objects else set()
    buffer = occupied | {(r + dr, c + dc) for r, c in occupied for dr, dc in N4}
    if cells & buffer:
        return False
    objects.append(mapping)
    return True


def _mirror_cells(mid, axis, length):
    radius = length // 2
    return {(mid[0] + (d if axis == 1 else 0),
             mid[1] + (d if axis == 0 else 0)) for d in range(-radius, radius + 1)}


def _route_plan(rng, elbow, direction, shape, bounces, axes, lengths):
    position = elbow
    path, mirrors, contacts = [], [], []
    for bounce in range(bounces):
        distance = int(rng.integers(3, 9))
        segment = [(position[0] + d * direction[0], position[1] + d * direction[1])
                   for d in range(1, distance + 1)]
        if not _inside(segment, shape):
            return None
        point = segment[-1]
        axis = axes[bounce] if axes else int(rng.integers(2))
        length = lengths[bounce] if lengths else int(rng.choice((1, 3, 5, 7)))
        mid = tuple(v + (direction[j] if j == axis else 0) for j, v in enumerate(point))
        cells = _mirror_cells(mid, axis, length)
        if not _inside(cells, shape):
            return None
        mirrors.append(cells)
        contacts.append(point)
        path.extend(segment)
        position = point
        direction = tuple(-v if j == axis else v for j, v in enumerate(direction))
    distance = int(rng.integers(3, 10))
    tail = [(position[0] + d * direction[0], position[1] + d * direction[1])
            for d in range(1, distance + 1)]
    if not _inside(tail, shape):
        return None
    path.extend(tail)
    if len(set(path)) != len(path):
        return None
    return dict(path=path, mirrors=mirrors, contacts=contacts, end=tail[-1])


def _attempt_pair(rng, bg, marker, pilot, teaching=None, test=False):
    low = 20 if teaching is not None or test else 16
    shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
    foreground = [c for c in range(10) if c not in (bg, marker)]
    objects = []
    nmobile = 1
    for mobile_index in range(nmobile):
        stencil = _mobile_shape(rng)
        body_color = int(rng.choice([c for c in foreground if c != pilot]))
        mobile_map = _mapping(stencil['body'], stencil['pins'], body_color, marker, stencil['tag'], pilot)
        t, l, b, r = _box(mobile_map)
        start = (int(rng.integers(-t, shape[0] - b)), int(rng.integers(-l, shape[1] - r)))
        if not _add_object(objects, _place_mapping(mobile_map, start), shape):
            return None
        if teaching is not None:
            bounces = 2
            # Opposite first contacts teach both normal reversals in two demos;
            # repeated axes are allowed elsewhere.
            axes = (teaching, 1 - teaching)
            lengths = (int(rng.choice((3, 5))), 1)
        else:
            bounces = 3 if test else int(rng.choice((0, 1, 2, 3), p=(.08, .42, .38, .12)))
            axes, lengths = None, None
        plan = _route_plan(rng, start, stencil['direction'], shape, bounces, axes, lengths)
        if plan is None:
            return None
        for mirror_index, cells in enumerate(plan['mirrors']):
            shade = pilot if teaching == 0 and mirror_index == 0 else int(rng.choice(foreground))
            if teaching == 1 and mirror_index == 0:
                shade = int(rng.choice([c for c in foreground if c != pilot]))
            if not _add_object(objects, {q: shade for q in cells}, shape):
                return None
        # Actual matched receiver lies after the final reflection.
        receiver = _receiver_shape(rng, stencil)
        receiver_color = int(rng.choice(foreground))
        receiver_map = _mapping(receiver['body'], receiver['pins'], receiver_color, marker)
        if not _add_object(objects, _place_mapping(receiver_map, plan['end']), shape):
            return None
        # A same-area but different-spacing receiver lies earlier on the route.
        wrong = _receiver_shape(rng, stencil, wrong=True)
        wrong_color = int(rng.choice([c for c in foreground if c != receiver_color]))
        wrong_map = _mapping(wrong['body'], wrong['pins'], wrong_color, marker)
        candidates = plan['path'][1:-3]
        rng.shuffle(candidates)
        added = False
        for point in candidates:
            placed = _place_mapping(wrong_map, point)
            if set(placed) & set(plan['path']):
                continue
            if _add_object(objects, placed, shape):
                added = True
                break
        if not added:
            return None
        # An equally valid geometric key elsewhere cannot attract the pilot.
        extras = int(rng.integers(2, 4)) if test else int(rng.integers(1, 3))
        for extra in range(extras):
            other = _receiver_shape(rng, stencil, wrong=extra > 0 and rng.random() < .5)
            shade = int(rng.choice(foreground))
            mapping = _mapping(other['body'], other['pins'], shade, marker)
            t, l, b, r = _box(mapping)
            for _ in range(30):
                point = (int(rng.integers(-t, shape[0] - b)), int(rng.integers(-l, shape[1] - r)))
                placed = _place_mapping(mapping, point)
                if set(placed) & set(plan['path']) or point in plan['path']:
                    continue
                if _add_object(objects, placed, shape):
                    break
            else:
                return None
    grid = np.full(shape, bg, dtype=np.int8)
    for mapping in objects:
        for q, shade in mapping.items():
            grid[q] = shade
    try:
        output, info = _predict(grid)
        if len(info['mobiles']) != nmobile or not _pair_evidence(grid, output, info):
            return None
        # Plans are only proposals. The actual input trace must use every mirror.
        used = {i for route in info['routes'] for i, *_ in route['contacts']}
        if used != set(range(len(info['mirrors']))):
            return None
        if teaching is not None:
            transitions = [(a, b) for route in info['routes'] for _, _, _, a, b in route['contacts']]
            if not transitions or transitions[0][0] != pilot:
                return None
            if (teaching == 0 and transitions[0][1] != pilot
                    or teaching == 1 and transitions[0][1] == pilot):
                return None
            if not _features(grid, output, info)['passed_wrong_key']:
                return None
            first_contact = info['routes'][0]['contacts'][0]
            if output[first_contact[2]] != first_contact[4]:
                return None
            try:
                weak, _ = _predict(grid, geometry=False)
                if np.array_equal(weak, output):
                    return None
            except ValueError:
                # A rejected area-only placement also fails to explain this
                # demonstration; the independent fixture covers a wrong stop.
                pass
        return {'input': grid, 'output': output}
    except ValueError:
        return None


def _pair_evidence(inp, out, info):
    if np.array_equal(inp, out):
        return False
    final_cells = set()
    for mobile, route in zip(info['mobiles'], info['routes']):
        if route['delta'] == (0, 0):
            return False
        moved = _shift(mobile['cells'], route['delta'])
        final_cells |= moved
        if _shift(mobile['pins'], route['delta']) != info['receivers'][route['receiver']]['pins']:
            return False
    # Several exposed beam pixels make reflection/transport observable.
    for route in info['routes']:
        if sum(point not in final_cells for point, _ in route['path']) < 4:
            return False
    return True


def _features(inp, out=None, info=None):
    if info is None:
        out, info = _predict(inp)
    wrong = 0
    for mobile, route in zip(info['mobiles'], info['routes']):
        points = {p for p, _ in route['path'][:-1]}
        # Aligned leading pin, equal area, different complete geometry: the
        # tempting area/min-corner shortcut must have an observable early stop.
        for receiver in info['receivers']:
            if len(receiver['pins']) != len(mobile['pins']) or _normal(receiver['pins']) == _normal(mobile['pins']):
                continue
            a, b = min(mobile['pins']), min(receiver['pins'])
            point = (mobile['elbow'][0] + b[0] - a[0], mobile['elbow'][1] + b[1] - a[1])
            if point in points:
                wrong += 1
    transitions = [t[3:5] for route in info['routes'] for t in route['contacts']]
    axes = [t[1] for route in info['routes'] for t in route['contacts']]
    return dict(shape=tuple(inp.shape), bg=info['bg'], marker=info['marker'],
                mobiles=len(info['mobiles']), receivers=len(info['receivers']),
                body_shapes=[_normal(m['body']) for m in info['mobiles']],
                signatures=[_normal(m['pins']) for m in info['mobiles']],
                directions=[m['direction'] for m in info['mobiles']],
                axes=axes, transitions=transitions,
                bounces=[len(r['contacts']) for r in info['routes']],
                mirror_lengths=[len(m['cells']) for m in info['mirrors']],
                flights=[len(r['path']) for r in info['routes']],
                deltas=[r['delta'] for r in info['routes']], passed_wrong_key=wrong,
                unused_receivers=len(info['receivers']) - len(info['mobiles']))


def _key(grid):
    return grid.shape, grid.tobytes()


def _plain_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _plain_int(seed):
        raise ValueError("seed must be an integer")
    if (not _plain_int(num_train) or not _plain_int(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 training, 1..4 tests, and training > tests")
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    pilot = int(rng.choice([c for c in range(10) if c != marker]))
    puzzle, seen = {'train': [], 'test': []}, set()
    # Marker and the teaching pilot hue are puzzle-level sampling choices,
    # but every necessary semantic role is recoverable from each input.
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            teaching = index if split == 'train' and index < 2 else None
            bg = int(rng.choice([c for c in range(10) if c not in (marker, pilot)]))
            local_pilot = pilot if teaching is not None else int(rng.choice([c for c in range(10) if c not in (marker, bg)]))
            for attempt in range(2400):
                pair = _attempt_pair(rng, bg, marker, local_pilot, teaching, split == 'test')
                if pair is None or _key(pair['input']) in seen:
                    continue
                if teaching == 1:
                    first = _features(puzzle['train'][0]['input'])
                    second = _features(pair['input'])
                    if (second['shape'] == first['shape'] or second['directions'] == first['directions']
                            or set(first['axes'] + second['axes']) != {0, 1}):
                        continue
                break
            else:
                raise RuntimeError("bounded beam-docking construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _predict(inp)
            assert np.array_equal(out, expected), "incorrect full output"
            assert _pair_evidence(inp, out, info), "missing transport/visible beam evidence"
            assert all(out[q] == inp[q] for q in info['fixed']), "fixed objects changed"
            for mobile, route in zip(info['mobiles'], info['routes']):
                for q, shade in mobile['mapping'].items():
                    dest = q[0] + route['delta'][0], q[1] + route['delta'][1]
                    assert out[dest] == shade, "rigid piece not preserved"
            assert set(np.unique(out)) <= set(np.unique(inp))
            assert {i for r in info['routes'] for i, *_ in r['contacts']} == set(range(len(info['mirrors'])))
            # Vacated coordinates contain only the newly rendered scene.
            for mobile in info['mobiles']:
                for q in mobile['cells']:
                    if q not in info['fixed'] and q not in info['painted'] and not any(
                            q in _shift(m['cells'], r['delta']) for m, r in zip(info['mobiles'], info['routes'])):
                        assert out[q] == info['bg']
            f = _features(inp, out, info)
            f['split'] = split
            records.append(f)
    train = [f for f in records if f['split'] == 'train']
    assert {a for f in train for a in f['axes']} == {0, 1}
    assert len({d for f in train for d in f['directions']}) >= 2
    assert any(1 in f['mirror_lengths'] for f in train)
    transitions = [t for f in train for t in f['transitions']]
    assert any(a == b for a, b in transitions) and any(a != b for a, b in transitions)
    successors = {}
    for a, b in transitions:
        successors.setdefault(a, set()).add(b)
    assert any(len(v) >= 2 for v in successors.values()), "fixed successor-color shortcut"
    assert sum(bool(f['passed_wrong_key']) for f in train) >= 2
    assert all(f['mobiles'] == 1 and f['unused_receivers'] >= 2 for f in records)
    assert all(max(f['bounces']) >= 3 and f['receivers'] >= 4
               for f in records if f['split'] == 'test')


def self_test():
    """Bounded 42-seed variation survey plus direct rule and negative checks."""
    records, hashes, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        hashes.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                f = _features(inp)
                f['split'] = split
                records.append(f)
                roles = {}
                structures.add((inp.shape, bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)))
    assert len(hashes) == 42
    assert len(structures) >= 180
    assert len({f['shape'] for f in records}) >= 50
    assert len({s for f in records for s in f['signatures']}) >= 15
    assert len({s for f in records for s in f['body_shapes']}) >= 80
    assert {d for f in records for d in f['directions']} == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert {f['mobiles'] for f in records} == {1}
    assert {f['bg'] for f in records} == set(range(10))
    assert len({f['marker'] for f in records}) >= 8
    assert {n for f in records for n in f['bounces']} >= {0, 1, 2, 3}
    assert {n for f in records for n in f['mirror_lengths']} >= {1, 3, 5, 7}
    assert any(any(a == b for a, b in zip(f['axes'], f['axes'][1:])) for f in records)
    _regressions()
    first, second = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(first[s], second[s]) for k in ('input', 'output'))
    bad = {s: [dict(p) for p in first[s]] for s in ('train', 'test')}
    bad['test'][0]['output'] = first['test'][0]['output'].copy()
    _, info = _predict(first['test'][0]['input'])
    route, mobile = info['routes'][0], info['mobiles'][0]
    q = min(_shift(mobile['body'], route['delta']))
    bad['test'][0]['output'][q] = info['bg']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("deleted transported payload accepted")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=len(records), shapes=len({f['shape'] for f in records}),
                structures=len(structures), signatures=len({s for f in records for s in f['signatures']}),
                payload_shapes=len({s for f in records for s in f['body_shapes']}),
                mobile_counts=dict(Counter(f['mobiles'] for f in records)))


def _regressions():
    # Geometry, motion, and exact color propagation, specified directly rather
    # than obtained from the random construction or from solve's output.
    shape = (25, 28)
    body = _rect(-3, -2, 2, 4)
    tag, pins = {(0, 0), (-1, 0), (0, -1)}, {(-4, -2), (-4, 1)}
    # Launch SE at (7,5); vertical mirror at (14,13) turns it SW.
    inp = np.zeros(shape, dtype=np.int8)
    maps = [_place_mapping(_mapping(body, pins, 3, 2, tag, 1), (7, 5)),
            {(r, 13): 4 for r in range(13, 16)},
            _place_mapping(_mapping(_rect(-6, -2, 2, 4), pins, 5, 2), (20, 6)),
            # Same-area wrong key at the earlier position (12,10).
            _place_mapping(_mapping(_rect(-6, -2, 2, 5), {(-4, -2), (-4, 2)}, 6, 2), (12, 10)),
            # Identical correct key, off the reflected route.
            _place_mapping(_mapping(_rect(-6, -2, 2, 4), pins, 7, 2), (20, 22))]
    for mapping in maps:
        for q, shade in mapping.items():
            inp[q] = shade
    expected = np.zeros(shape, dtype=np.int8)
    for mapping in maps[1:]:
        for q, shade in mapping.items():
            expected[q] = shade
    for r in range(8, 14):
        expected[r, r - 2] = 1
    for r in range(14, 21):
        expected[r, 26 - r] = 4
    final = _place_mapping(_mapping(body, pins, 3, 2, tag, 1), (20, 6))
    for q, shade in final.items():
        expected[q] = shade
    out, info = _predict(inp)
    assert np.array_equal(out, expected), "hand-enumerated reflection/docking regression"
    assert info['routes'][0]['end'] == (20, 6)
    assert out[14, 12] == 4, "contact pixel must change color immediately"
    assert np.all(out[4:6, 3:7] == 0), "launch payload must be erased"
    weaker, weak_info = _predict(inp, geometry=False)
    assert weak_info['routes'][0]['end'] == (12, 10) and not np.array_equal(weaker, expected)
    try:
        _predict(inp, reflect=False)
    except ValueError:
        pass
    else:
        raise AssertionError("straight-line shortcut should strike the mirror")
    # Whole-scene symmetries change launch/face orientation, not the rule.
    for k in range(4):
        for flip in (False, True):
            a, b = inp.copy(), expected.copy()
            if flip:
                a, b = a[:, ::-1], b[:, ::-1]
            a, b = np.rot90(a, k), np.rot90(b, k)
            assert np.array_equal(solve(a), b)
    mirror = dict(mid=(14, 13), axes=(1,), color=4)
    assert _contact(mirror, (14, 12), (1, 1)) == 1
    assert _contact(mirror, (13, 12), (1, 1)) is None
    assert _normal({(0, 0), (0, 3)}) != _normal({(0, 0), (0, 4)})


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
