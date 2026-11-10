"""Hidden chromatic mirrors: reconstruct filters, then trace port-emitted light.

Only standard-library modules and NumPy are used. No parent generator is imported.
The scene renderer knows complete rectangles; solve reconstructs them from pixels.
"""
from __future__ import annotations

import copy
from itertools import permutations
import numpy as np


_CORNERS = ((-1, -1), (-1, 2), (2, -1), (2, 2))
_RING = tuple((y, x) for y in range(-1, 3) for x in range(-1, 3)
              if y in (-1, 2) or x in (-1, 2))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _check_counts(num_train, num_test):
    a, b = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError('require 2..10 training, 1..4 test, and num_train > num_test')
    return a, b


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)} | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)} | {(y, r) for y in range(t + 1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _inside(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _strict(point, box):
    t, b, l, r = box
    return t < point[0] < b and l < point[1] < r


def _bbox(points):
    yy, xx = zip(*points)
    return min(yy), max(yy), min(xx), max(xx)


def _halo(points):
    return {q for y, x in points for q in
            ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))}


def _components(points):
    todo, groups = set(points), []
    while todo:
        p = min(todo)
        todo.remove(p)
        group, stack = {p}, [p]
        while stack:
            y, x = stack.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _pattern(grid, box, visible, hidden):
    """All witnessed period <=3 diagonal-coordinate models must agree."""
    predictions, explanations = set(), []
    order = sorted(hidden)
    for sign in (1, -1):
        for period in (1, 2, 3):
            word, counts = {}, [0] * period
            for y, x in sorted(visible):
                k, color = (y + sign * x) % period, int(grid[y, x])
                if k in word and word[k] != color:
                    break
                word[k] = color
                counts[k] += 1
            else:
                if min(counts) >= 2:
                    predictions.add(tuple(word[(y + sign * x) % period] for y, x in order))
                    explanations.append((period, sign))
    _require(len(predictions) == 1, 'ambiguous hidden color phase')
    return dict(zip(order, next(iter(predictions)))), min(p for p, _ in explanations)


def _footprint(core):
    y, x = core
    return {(r, c) for r in range(y - 1, y + 3) for c in range(x - 1, x + 3)}


def _parse(input_grid):
    grid = np.asarray(input_grid)
    _require(grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer), 'integer 2D input required')
    _require(all(1 <= d <= 30 for d in grid.shape) and np.all((0 <= grid) & (grid <= 9)), 'invalid grid')
    bg = int(grid[0, 0])
    _require(all(int(grid[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))), 'background corners disagree')
    frames = []
    for color in sorted(set(map(int, grid.flat)) - {bg}):
        for group in _components(map(tuple, np.argwhere(grid == color))):
            a, z, c, d = _bbox(group)
            if z - a < 5 or d - c < 5 or len(group) != (z - a + 1) * (d - c + 1):
                continue
            box = (a - 1, z + 1, c - 1, d + 1)
            if not all(_inside(p, grid.shape) for p in _perimeter(box)):
                continue
            border = {int(grid[p]) for p in _perimeter(box)}
            if len(border) == 1 and not border & {bg, color}:
                frames.append((box, color, next(iter(border))))
    _require(len(frames) == 1, 'unique observation window required')
    main, fill, marker = frames[0]
    mask = _area(main)
    launchers, footprints = [], set()
    for y in range(1, grid.shape[0] - 2):
        for x in range(1, grid.shape[1] - 2):
            payload = int(grid[y, x])
            if payload in (bg, fill, marker) or not np.all(grid[y:y + 2, x:x + 2] == payload):
                continue
            core = (y, x)
            footprint = _footprint(core)
            if footprint & mask:
                continue
            ports, valid = [], True
            for dy, dx in _RING:
                value = int(grid[y + dy, x + dx])
                if (dy, dx) in _CORNERS and value == bg:
                    ports.append(((y + dy, x + dx), (-1 if dy < 0 else 1, -1 if dx < 0 else 1)))
                elif value != marker:
                    valid = False
                    break
            if valid and ports:
                _require(not footprint & footprints, 'overlapping launchers')
                footprints |= footprint
                launchers.append({'core': core, 'color': payload, 'ports': ports})
    _require(1 <= len(launchers) <= 3, 'require one to three launchers')
    points = {tuple(p) for p in np.argwhere(grid != bg)} - mask - footprints
    groups = _components(points)
    _require(1 <= len(groups) <= 6, 'unreadable outline fragments')
    candidates = []
    for i in range(len(groups)):
        for j in range(i, len(groups)):
            visible = groups[i] | groups[j]
            box = _bbox(visible)
            t, b, l, r = box
            full = _perimeter(box)
            if min(b - t, r - l) < 4 or full - mask != visible:
                continue
            hidden = full & mask
            if not any(_strict(p, main) for p in hidden):
                continue
            try:
                assignments, period = _pattern(grid, box, visible, hidden)
            except AssertionError:
                continue
            if {int(grid[p]) for p in visible} & {bg, fill, marker}:
                continue
            colors = {p: int(grid[p]) for p in visible}
            colors.update(assignments)
            candidates.append({'parts': {i, j}, 'box': box, 'colors': colors, 'period': period})
    covers = []

    def search(todo, selected, used):
        if len(covers) > 1:
            return
        if not todo:
            covers.append(selected)
            return
        first = min(todo)
        for candidate in candidates:
            parts, full = candidate['parts'], set(candidate['colors'])
            if first in parts and parts <= todo and not full & _halo(used):
                search(todo - parts, selected + [candidate], used | full)

    search(set(range(len(groups))), [], set())
    _require(len(covers) == 1 and 1 <= len(covers[0]) <= 3, 'ambiguous rectangle completion')
    outlines = sorted(covers[0], key=lambda o: o['box'])
    _require(not set().union(*(set(o['colors']) for o in outlines)) & footprints, 'mirror hits launcher')
    return {'shape': grid.shape, 'main': main, 'fill': fill, 'background': bg,
            'marker': marker, 'outlines': outlines, 'launchers': launchers}


def _walls(outlines):
    walls, corners = {}, set()
    for owner, outline in enumerate(outlines):
        t, b, l, r = outline['box']
        corners |= {(t, l), (t, r), (b, l), (b, r)}
        for (y, x), color in outline['colors'].items():
            _require((y, x) not in walls, 'overlapping mirrors')
            walls[y, x] = (int(color), 0 if y in (t, b) else 1, owner)
    return walls, corners


def _trace(shape, walls, corners, start, direction, color, mode='normal'):
    """A rejected step reflects at a side; the new step starts at the same cell.

    Alternative modes are only for curriculum checks. They return partial paths
    at cycles/corners so even incompatible shortcut laws have bounded execution.
    """
    p, d = start, direction
    path, events, seen = [], [], set()
    for _ in range(4 * shape[0] * shape[1]):
        if p in seen:
            if mode != 'normal':
                return path, events
            raise AssertionError('self-intersecting or trapped beam')
        seen.add(p)
        path.append(p)
        q = (p[0] + d[0], p[1] + d[1])
        if not _inside(q, shape):
            return path, events
        if q in walls:
            if q in corners:
                if mode != 'normal':
                    return path, events
                raise AssertionError('beam hits a mirror corner')
            gate, axis, owner = walls[q]
            if mode == 'opaque' or (mode == 'stop' and color != gate):
                return path, events
            reflect = color != gate if mode != 'reflect_all' else True
            if mode == 'once' and any(e['kind'] == 'reflect' for e in events):
                reflect = False
            events.append({'point': q, 'kind': 'reflect' if reflect else 'pass',
                           'owner': owner, 'axis': axis, 'gate': gate, 'incoming': d})
            if reflect:
                d = (-d[0], d[1]) if axis == 0 else (d[0], -d[1])
                q = (p[0] + d[0], p[1] + d[1])
                if not _inside(q, shape):
                    return path, events
                if q in walls:
                    if mode != 'normal':
                        return path, events
                    raise AssertionError('two mirror contacts in one step')
        p = q
    raise AssertionError('beam tracing exceeded its finite bound')


def _segments(path):
    # Twice a segment midpoint; opposite diagonals sharing it are an X crossing.
    return {(p[0] + q[0], p[1] + q[1]) for p, q in zip(path, path[1:])}


def _render(scene, mode='normal'):
    walls, corners = _walls(scene['outlines'])
    if mode == 'straight':
        walls = {}
    elif mode == 'visible_only':
        walls = {p: v for p, v in walls.items() if not p in _area(scene['main'])}
    t, b, l, r = scene['main']
    out = np.full((b - t - 1, r - l - 1), scene['fill'], dtype=np.int8)
    footprints = set().union(*(_footprint(a['core']) for a in scene['launchers']))
    used, segments, beams = set(), set(), []
    trace_mode = 'normal' if mode in ('straight', 'visible_only') else mode
    for launcher in scene['launchers']:
        for start, direction in launcher['ports']:
            path, events = _trace(scene['shape'], walls, corners, start, direction,
                                  launcher['color'], trace_mode)
            occupied = set(path)
            if mode == 'normal':
                _require(not (occupied - {start}) & footprints, 'beam strikes a launcher')
                _require(not occupied & used and not _segments(path) & segments, 'beams intersect')
                _require(len([p for p in path if _strict(p, scene['main'])]) >= 3, 'uninformative beam')
            used |= occupied
            segments |= _segments(path)
            for y, x in path:
                if _strict((y, x), scene['main']):
                    out[y - t - 1, x - l - 1] = launcher['color']
            beams.append({'path': path, 'events': events, 'color': launcher['color']})
    return out, beams


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every convention from visible pixels, then return the light crop."""
    return _render(_parse(input_grid))[0]


def _geometry(rng, count):
    h, w = map(int, rng.integers(24, 31, 2))
    oh = int(rng.integers(10, min(16, h - 10) + 1))
    ow = int(rng.integers(10, min(16, w - 10) + 1))
    t, l = int(rng.integers(4, h - oh - 4)), int(rng.integers(4, w - ow - 4))
    main = (t, t + oh + 1, l, l + ow + 1)
    mask, boxes, used = _area(main), [], set()
    for _ in range(count):
        for _ in range(500):
            if boxes and rng.random() < .55:
                a, z, c, d = boxes[0]
                if min(z - a, d - c) < 8:
                    continue
                a += int(rng.integers(2, 4)); z -= int(rng.integers(2, 4))
                c += int(rng.integers(2, 4)); d -= int(rng.integers(2, 4))
            elif not boxes and rng.random() < .22:
                # Broad shallow filters can be illuminated by two corners of
                # the same core. Other candidates still use unrestricted boxes.
                mt, mb, ml, mr = main
                if rng.random() < .5:
                    a = int(rng.integers(1, mt - 1))
                    z = int(rng.integers(mt + 2, min(mb, mt + 6)))
                    c = int(rng.integers(1, ml - 1))
                    d = int(rng.integers(mr + 2, w - 1))
                    if rng.random() < .5:
                        a, z = h - 1 - z, h - 1 - a
                else:
                    a = int(rng.integers(1, mt - 1))
                    z = int(rng.integers(mb + 2, h - 1))
                    c = int(rng.integers(1, ml - 1))
                    d = int(rng.integers(ml + 2, min(mr, ml + 6)))
                    if rng.random() < .5:
                        c, d = w - 1 - d, w - 1 - c
            else:
                a, z = sorted(map(int, rng.choice(np.arange(1, h - 1), 2, replace=False)))
                c, d = sorted(map(int, rng.choice(np.arange(1, w - 1), 2, replace=False)))
            box = (a, z, c, d)
            if min(z - a, d - c) < 4:
                continue
            full = _perimeter(box)
            visible = full - mask
            if (len(visible) < 8 or sum(_strict(p, main) for p in full) < 4
                    or _bbox(visible) != box or full & _halo(used)):
                continue
            boxes.append(box)
            used |= full
            break
        else:
            return None
    return (h, w), main, boxes


def _candidate_beams(rng, scene):
    walls, corners = _walls(scene['outlines'])
    blocked = _area(scene['main']) | set(walls)
    options = []
    for y in range(2, scene['shape'][0] - 3):
        for x in range(2, scene['shape'][1] - 3):
            core = (y, x)
            footprint = _footprint(core)
            if footprint & blocked:
                continue
            for dy, dx in _CORNERS:
                options.append((core, (y + dy, x + dx),
                                (-1 if dy < 0 else 1, -1 if dx < 0 else 1), footprint))
    rng.shuffle(options)
    palette = sorted({v[0] for v in walls.values()})
    for core, start, direction, footprint in options:
        for color in rng.permutation(palette):
            color = int(color)
            try:
                path, events = _trace(scene['shape'], walls, corners, start, direction, color)
            except AssertionError:
                continue
            if (set(path) - {start}) & footprint:
                continue
            hidden = [e for e in events if _strict(e['point'], scene['main'])]
            if not hidden or sum(_strict(p, scene['main']) for p in path) < 3:
                continue
            yield {'core': core, 'color': color, 'ports': [(start, direction)],
                   'path': path, 'events': hidden, 'footprint': footprint}


def _compatible(candidate, selected):
    for other in selected:
        if candidate['core'] == other['core']:
            if candidate['color'] != other['color'] or candidate['ports'] == other['ports']:
                return False
        elif candidate['footprint'] & other['footprint']:
            return False
        if (set(candidate['path']) & set(other['path'])
                or _segments(candidate['path']) & _segments(other['path'])
                or (set(candidate['path']) - {candidate['ports'][0][0]}) & other['footprint']
                or (set(other['path']) - {other['ports'][0][0]}) & candidate['footprint']):
            return False
    return True


def _other_ports(rng, scene, selected):
    """Try another visible corner of an existing core before adding a core."""
    walls, corners = _walls(scene['outlines'])
    bases = list(selected)
    rng.shuffle(bases)
    for base in bases:
        y, x = base['core']
        offsets = list(_CORNERS)
        rng.shuffle(offsets)
        for dy, dx in offsets:
            start = (y + dy, x + dx)
            direction = (-1 if dy < 0 else 1, -1 if dx < 0 else 1)
            try:
                path, events = _trace(scene['shape'], walls, corners, start, direction, base['color'])
            except AssertionError:
                continue
            hidden = [e for e in events if _strict(e['point'], scene['main'])]
            if (not hidden or sum(_strict(p, scene['main']) for p in path) < 3
                    or (set(path) - {start}) & base['footprint']):
                continue
            candidate = {'core': base['core'], 'color': base['color'], 'ports': [(start, direction)],
                         'path': path, 'events': hidden, 'footprint': base['footprint']}
            if _compatible(candidate, selected):
                yield candidate


def _draw_input(scene):
    grid = np.full(scene['shape'], scene['background'], dtype=np.int8)
    for outline in scene['outlines']:
        for p, color in outline['colors'].items():
            grid[p] = color
    t, b, l, r = scene['main']
    grid[t + 1:b, l + 1:r] = scene['fill']
    for p in _perimeter(scene['main']):
        grid[p] = scene['marker']
    for launcher in scene['launchers']:
        y, x = launcher['core']
        grid[y:y + 2, x:x + 2] = launcher['color']
        ports = {p for p, _ in launcher['ports']}
        for dy, dx in _RING:
            p = (y + dy, x + dx)
            grid[p] = scene['background'] if p in ports else scene['marker']
    return grid


def _construct_bounces(rng):
    """Construct a readable two-normal demonstration instead of hoping for one.

    The emitter sits inside a box, beside the window. Its southeast beam can
    visit the hidden bottom and right sides, then return through a matching gate.
    Side lengths, placement of the exposed sides, canvas, colors and orientation
    vary. Every candidate is traced and checked; no output pixels are stored.
    """
    geometry = [(a, b, l, r) for a in range(1, 5) for b in range(18, 22)
                for l in (1, 2) for r in range(12, 17)]
    rng.shuffle(geometry)
    # The final geometry guarantees a witnessed construction if random trials
    # are unhelpful. It is a coordinate formula, not a saved puzzle or answer.
    geometry = geometry[:50] + [(3, 18, 2, 12)]
    words = list(permutations((3, 4, 5)))
    for box in geometry:
        a, b, l, r = box
        main = (b - 11, b + 4, r - 4, r + 12)
        shape = (int(rng.integers(b + 6, 31)), int(rng.integers(r + 14, 31)))
        trials = [(sign, word) for sign in (-1, 1) for word in words]
        rng.shuffle(trials)
        for sign, word in trials:
            scene = {'shape': shape, 'main': main, 'background': 0, 'marker': 1, 'fill': 2,
                     'outlines': [{'box': box, 'colors': {p: word[(p[0] + sign * p[1]) % 3]
                                                        for p in _perimeter(box)}}],
                     'launchers': [{'core': (b - 5, r - 8), 'color': 3,
                                    'ports': [((b - 3, r - 6), (1, 1))]}]}
            try:
                out, beams = _render(scene)
                hidden = [e for e in beams[0]['events'] if _strict(e['point'], main)]
                _require(any(e['kind'] == 'pass' for e in hidden), 'construction lacks a transparent gate')
                _require({e['axis'] for e in hidden if e['kind'] == 'reflect'} == {0, 1},
                         'construction lacks both normals')
                grid = _draw_input(scene)
                parsed = _parse(grid)
                _require(np.array_equal(_render(parsed)[0], out), 'construction cannot be read')
                _require(all(not np.array_equal(_render(parsed, mode)[0], out) for mode in
                             ('straight', 'visible_only', 'opaque', 'stop', 'reflect_all', 'once')),
                         'construction does not separate shortcut laws')
            except AssertionError:
                continue
            lookup = rng.permutation(10).astype(np.int8)
            grid, out = lookup[grid], lookup[out]
            turns, flip = int(rng.integers(4)), bool(rng.integers(2))
            grid, out = np.rot90(grid, turns), np.rot90(out, turns)
            if flip:
                grid, out = np.fliplr(grid), np.fliplr(out)
            grid, out = np.ascontiguousarray(grid), np.ascontiguousarray(out)
            _require(np.array_equal(solve(grid), out), 'oriented construction changed the rule')
            return {'input': grid, 'output': out}
    raise RuntimeError('constructive repeated-bend witness failed')


def _make_pair(rng, count, demo):
    """demo='contrast': two colors; 'bounces': repeated reflection; 'network': multi-filter."""
    if demo == 'bounces':
        return _construct_bounces(rng)
    for _ in range(160):
        geometry = _geometry(rng, count)
        if geometry is None:
            continue
        shape, main, boxes = geometry
        roles = list(map(int, rng.permutation(10)))
        bg, marker, fill, *palette = roles
        palette = palette[:3]
        scene = {'shape': shape, 'main': main, 'background': bg, 'marker': marker,
                 'fill': fill, 'outlines': [], 'launchers': []}
        for index, box in enumerate(boxes):
            period = 2 if demo == 'contrast' else 3
            if demo == 'network' and index and rng.random() < .25:
                period = int(rng.integers(1, 4))
            sign, phase = int(rng.choice((-1, 1))), int(rng.integers(period))
            colors = list(map(int, rng.permutation(palette)))[:period]
            # AAB as well as ABC words, with a genuinely inferable period of 3.
            if period == 3 and rng.random() < .25:
                colors[2] = colors[0]
            assignments = {p: colors[(p[0] + sign * p[1] + phase) % period] for p in _perimeter(box)}
            scene['outlines'].append({'box': box, 'colors': assignments})
        # Certify phase and geometry before the more expensive beam search.
        mask = _area(main)
        try:
            for o in scene['outlines']:
                visible, hidden = set(o['colors']) - mask, set(o['colors']) & mask
                predicted, _ = _pattern(_draw_input(scene), o['box'], visible, hidden)
                _require(all(predicted[p] == o['colors'][p] for p in hidden), 'renderer phase ambiguity')
        except AssertionError:
            continue
        selected, candidates = [], _candidate_beams(rng, scene)
        for candidate in candidates:
            events = candidate['events']
            kinds = {e['kind'] for e in events}
            if demo == 'contrast':
                if not selected:
                    if 'reflect' not in kinds:
                        continue
                elif 'pass' not in kinds:
                    continue
            else:
                if not {'pass', 'reflect'} <= kinds:
                    continue
                if demo == 'network' and len({e['owner'] for e in events}) < count:
                    continue
                if demo == 'network' and len({e['owner'] for e in events if e['kind'] == 'reflect'}) < count:
                    continue
            if not _compatible(candidate, selected):
                continue
            selected.append(candidate)
            if demo == 'contrast' and len(selected) == 1:
                for other in _other_ports(rng, scene, selected):
                    if any(e['kind'] == 'pass' for e in other['events']):
                        selected.append(other)
                        break
            if demo != 'contrast' or len(selected) == 2:
                break
        if not selected or (demo == 'contrast' and len(selected) != 2):
            continue
        # Optional extra illuminating strands, including extra ports on a core.
        target = int(rng.integers(len(selected), min(3, len(selected) + 1) + 1))
        if len(selected) < target and rng.random() < .8:
            for candidate in _other_ports(rng, scene, selected):
                selected.append(candidate)
                if len(selected) == target:
                    break
        if len(selected) < target:
            for candidate in candidates:
                if _compatible(candidate, selected):
                    selected.append(candidate)
                    if len(selected) == target:
                        break
        launchers = {}
        for c in selected:
            key = (c['core'], c['color'])
            if key not in launchers:
                launchers[key] = {'core': c['core'], 'color': c['color'], 'ports': []}
            launchers[key]['ports'] += c['ports']
        scene['launchers'] = list(launchers.values())
        try:
            output, _ = _render(scene)
            grid = _draw_input(scene)
            recovered = _parse(grid)
            _require(np.array_equal(_render(recovered)[0], output), 'pixel reconstruction differs from complete scene')
            if any(np.array_equal(_render(recovered, alternative)[0], output)
                   for alternative in ('straight', 'visible_only', 'opaque', 'stop', 'reflect_all')):
                continue
        except AssertionError:
            continue
        return {'input': grid, 'output': output}
    raise RuntimeError(f'could not construct an informative {count}-mirror {demo} scene')


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(_integer(seed, 'seed'))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if split == 'train':
                n = 1 if index < 2 else (2 if index == 2 else int(rng.integers(1, 3)))
                demo = 'contrast' if index == 0 else 'bounces' if n == 1 else 'network'
            else:
                n = 2 if num_train == 2 else 3
                demo = 'network'
            for _ in range(8):
                pair = _make_pair(rng, n, demo)
                if split == 'train' and index == 1:
                    first_scene = _parse(puzzle['train'][0]['input'])
                    next_scene = _parse(pair['input'])
                    if ((first_scene['background'], first_scene['marker'])
                            == (next_scene['background'], next_scene['marker'])):
                        continue
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded uniqueness attempts exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_features(pair):
    scene = _parse(pair['input'])
    out, beams = _render(scene)
    hidden = [[e for e in b['events'] if _strict(e['point'], scene['main'])] for b in beams]
    return scene, out, {'mirrors': len(scene['outlines']), 'beams': len(beams),
                        'launchers': len(scene['launchers']),
                        'port_counts': tuple(len(a['ports']) for a in scene['launchers']),
                        'periods': tuple(o['period'] for o in scene['outlines']),
                        'passes': sum(e['kind'] == 'pass' for ee in hidden for e in ee),
                        'reflections': sum(e['kind'] == 'reflect' for ee in hidden for e in ee),
                        'repeat': max(sum(e['kind'] == 'reflect' for e in ee) for ee in hidden),
                        'chain': max(len({e['owner'] for e in ee}) for ee in hidden),
                        'reflect_chain': max(len({e['owner'] for e in ee if e['kind'] == 'reflect'}) for ee in hidden),
                        'contacts': {e['owner'] for ee in hidden for e in ee},
                        'axes': {e['axis'] for ee in hidden for e in ee if e['kind'] == 'reflect'},
                        'directions': {d for a in scene['launchers'] for _, d in a['ports']}}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            for value in pair.values():
                a = np.asarray(value)
                _require(a.ndim == 2 and np.issubdtype(a.dtype, np.integer), 'invalid array')
                _require(all(1 <= d <= 30 for d in a.shape) and np.all((0 <= a) & (a <= 9)), 'invalid cells or shape')
            key = _key(pair['input'])
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            scene, expected, f = _pair_features(pair)
            actual = np.asarray(pair['output'])
            _require(np.array_equal(expected, actual), 'incorrect beam crop')
            _require(f['passes'] > 0 and f['reflections'] > 0, 'both filter behaviors must be visible')
            _require(f['contacts'] == set(range(f['mirrors'])), 'unilluminated mirror')
            _require(np.any(actual != scene['fill']), 'blank light crop')
            t, b, l, r = scene['main']
            only_a = np.full(actual.shape, scene['fill'], dtype=np.int8)
            for o in scene['outlines']:
                for (y, x), color in o['colors'].items():
                    if t < y < b and l < x < r:
                        only_a[y - t - 1, x - l - 1] = color
            _require(not np.array_equal(only_a, actual), 'restoring filters alone gives the answer')
            _require(set(map(int, actual.flat)) <= {scene['fill']} | {a['color'] for a in scene['launchers']},
                     'mirror/frame decoration leaked into the answer')
            # Each ablation must actually change a returned answer, not just metadata.
            for alternative in ('straight', 'visible_only', 'opaque', 'stop', 'reflect_all'):
                _require(not np.array_equal(_render(scene, alternative)[0], actual),
                         f'example does not distinguish {alternative}')
            features[split].append(f)
    train = features['train']
    _require(train[0]['mirrors'] == 1 and 2 in train[0]['periods'], 'missing two-color contrast')
    _require(train[1]['mirrors'] == 1 and 3 in train[1]['periods'] and train[1]['repeat'] >= 2,
             'missing phase/repeated-bend demonstration')
    _require(train[1]['axes'] == {0, 1}, 'both mirror-side normals must be demonstrated')
    second_scene = _parse(puzzle['train'][1]['input'])
    _require(not np.array_equal(_render(second_scene, 'once')[0], puzzle['train'][1]['output']),
             'repeated reflection has no consequence')
    _require(len({(_parse(p['input'])['background'], _parse(p['input'])['marker'])
                  for p in puzzle['train']}) >= 2, 'training roles did not vary')
    for f in features['test']:
        _require(f['mirrors'] > max(t['mirrors'] for t in train), 'missing structural test shift')
        _require(f['chain'] == f['mirrors'], 'test must connect every filter along one ray')
        _require(f['reflect_chain'] == f['mirrors'], 'every test mirror must bend the same ray')


def self_test():
    """Bounded 42-seed variation audit, local physics fixtures, and API checks."""
    # Hand-specified local physics, not generated labels: (row, column).
    walls = {(4, 4): (2, 1, 0)}
    path, events = _trace((9, 9), walls, set(), (2, 2), (1, 1), 1)
    _require(path[:5] == [(2, 2), (3, 3), (4, 2), (5, 1), (6, 0)], 'vertical reflection fixture')
    _require(events[0]['kind'] == 'reflect' and (4, 4) not in path, 'mirror cell was painted')
    path, _ = _trace((9, 9), {(4, 4): (2, 0, 0)}, set(), (2, 2), (1, 1), 1)
    _require(path[:5] == [(2, 2), (3, 3), (2, 4), (1, 5), (0, 6)], 'horizontal reflection fixture')
    path, events = _trace((9, 9), walls, set(), (2, 2), (1, 1), 2)
    _require(path == [(i, i) for i in range(2, 9)] and events[0]['kind'] == 'pass', 'transmission fixture')
    box, mask = (1, 7, 1, 7), _area((4, 9, 4, 9))
    fixture = np.zeros((10, 10), dtype=np.int8)
    truth = {p: (2, 5, 7)[sum(p) % 3] for p in _perimeter(box)}
    for p, color in truth.items():
        if p not in mask:
            fixture[p] = color
    predicted, period = _pattern(fixture, box, set(truth) - mask, set(truth) & mask)
    _require(period == 3 and predicted == {p: truth[p] for p in set(truth) & mask}, 'hidden-phase fixture')
    features, shapes, layouts, backgrounds, directions, signatures = [], set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene, out, f = _pair_features(pair)
                features.append(f)
                shapes.add((tuple(np.asarray(pair['input']).shape), tuple(out.shape)))
                layouts.add((scene['shape'], scene['main'], tuple(o['box'] for o in scene['outlines']),
                             tuple((a['core'], tuple(a['ports'])) for a in scene['launchers'])))
                backgrounds.add(scene['background'])
                directions |= f['directions']
        if seed == 0:
            first = puzzle
    _require(len(signatures) == 42 and len(shapes) >= 50 and len(layouts) >= 200, 'variation collapsed')
    _require(backgrounds == set(range(10)) and len(directions) == 4, 'semantic palette/direction coverage collapsed')
    _require({f['mirrors'] for f in features} == {1, 2, 3}, 'mirror-count coverage collapsed')
    _require(len({f['launchers'] for f in features}) >= 2 and len({f['beams'] for f in features}) >= 2,
             'emitter variation collapsed')
    _require(set().union(*(f['axes'] for f in features)) == {0, 1}, 'both side normals required')
    repeat = generate_puzzle(0)
    _require(all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], repeat[s])
                 for k in ('input', 'output')), 'seed nondeterminism')
    # Host regression: random search previously exhausted the two-normal witness.
    validate_generated_puzzle(generate_puzzle(998636056))
    for a, b in ((2, 1), (3, 2), (5, 4), (10, 4)):
        validate_generated_puzzle(generate_puzzle(400 + a, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = copy.deepcopy(first)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt label accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'shape_pairs': len(shapes),
            'geometric_layouts': len(layouts), 'mirror_counts': sorted({f['mirrors'] for f in features}),
            'emitter_counts': sorted({f['launchers'] for f in features}),
            'beam_counts': sorted({f['beams'] for f in features}),
            'multiport_launchers': sum(n > 1 for f in features for n in f['port_counts'])}
