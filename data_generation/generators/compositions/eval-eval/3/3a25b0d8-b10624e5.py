"""Flagged addresses activate deformed chambers; no parent code is imported.

Generation paints labeled regions. The solver independently finds enclosures
and transports visible address squares using visible anchor geometry.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 984035382, 267698305, 15348418, 741175767, 22139316,
              866051335, 948128157, 701965635, 518462457, 622198125,
              651902484, 925352447, 187965707, 143715147, 773456492,
              382036393, 966857808, 271587096, 218666596, 627456634,
              885573740, 997719654, 401390465, 388662371, 474019240,
              90986410, 644503719, 874300173, 757455421, 70065806,
              402060647, 75819679, 699632039, 469510854, 752979920,
              395349453, 491329759, 110839730, 471694514, 721085507,
              281682892)
_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    steps = _STEPS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    part.append(p)
        parts.append(np.asarray(part, dtype=int))
    return parts


def _box(points):
    return (*map(int, points.min(0)), *map(int, points.max(0) + 1))


def _square(points):
    r, c, R, C = _box(points)
    return R - r if R - r == C - c and len(points) == (R - r) ** 2 else 0


def _enclosures(grid, wall):
    padded = np.pad(grid != wall, 1, constant_values=True)
    return [p - 1 for p in _components(padded)
            if np.all(p > 0) and np.all(p < np.array(padded.shape) - 1)]


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    assert np.all(grid[0] == bg) and np.all(grid[-1] == bg)
    assert np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)
    components = _components(grid != bg, diagonal=True)
    assert 2 <= len(components) <= 3, "one donor and one or two receivers"
    objects = []
    for points in components:
        r, c, R, C = _box(points)
        crop = np.full((R - r, C - c), bg, dtype=np.int8)
        crop[points[:, 0] - r, points[:, 1] - c] = grid[tuple(points.T)]
        objects.append({'grid': crop, 'offset': (r, c)})
    common = set.intersection(*(set(map(int, o['grid'].flat)) for o in objects)) - {bg}
    candidates = []
    for marker in sorted(common):
        all_parts = [_components(o['grid'] == marker) for o in objects]
        valid = True
        for parts in all_parts:
            sides = sorted(_square(p) for p in parts)
            if not sides or sides[-1] not in (2, 4) or sides.count(sides[-1]) != 1:
                valid = False
                break
            if any(s != sides[-1] // 2 for s in sides[:-1]) or len(parts) < 2:
                valid = False
                break
        if valid:
            candidates.append((marker, all_parts))
    assert len(candidates) == 1, "unique square-anchor marker color"
    marker, marker_parts = candidates[0]
    walls = common - {marker}
    assert len(walls) == 1
    wall = next(iter(walls))
    for obj, parts in zip(objects, marker_parts):
        crop = obj['grid']
        anchor = max(parts, key=len)
        ar, ac, _, _ = _box(anchor)
        unit = _square(anchor) // 2
        rooms = _enclosures(crop, wall)
        assert len(rooms) >= 3
        room_id = np.full(crop.shape, -1, dtype=int)
        for i, points in enumerate(rooms):
            room_id[tuple(points.T)] = i
        satellite = [p for p in parts if len(p) != len(anchor)
                     and np.all(room_id[tuple(p.T)] == -1)]
        assert len(satellite) == 1, "one exterior compass flag"
        sr, sc, _, _ = _box(satellite[0])
        assert sr - ar in (-2 * unit, 3 * unit)
        assert sc - ac in (-2 * unit, 3 * unit)
        flag = (int(np.sign(sr - ar)), int(np.sign(sc - ac)))
        fills, probes = [], []
        for i, points in enumerate(rooms):
            values = set(map(int, crop[tuple(points.T)]))
            colors = values - {bg, marker}
            assert len(colors) <= 1
            if colors:
                color = next(iter(colors))
                assert bg not in values and wall not in values
                pins = [p for p in parts if np.all(room_id[tuple(p.T)] == i)]
                assert len(pins) == 1 and _square(pins[0]) == unit
                assert len(points) > unit * unit
                pr, pc, _, _ = _box(pins[0])
                assert (pr - ar) % unit == (pc - ac) % unit == 0
                probes.append((pr, pc, color))
                fills.append(color)
            else:
                assert values == {bg}, "receiver rooms are completely blank"
        if fills:
            assert len(fills) == len(rooms) and len(set(fills)) >= 2
            assert len(parts) == len(rooms) + 2
        else:
            assert len(parts) == 2
        visible = (crop != bg) & (crop != wall) & (crop != marker)
        assert not np.any(visible & (room_id < 0))
        obj.update(anchor=(ar, ac), unit=unit, flag=flag, rooms=rooms,
                   room_id=room_id, probes=probes, colors=fills)
    donors = [i for i, obj in enumerate(objects) if obj['probes']]
    assert len(donors) == 1
    return {'grid': grid, 'background': bg, 'wall': wall, 'marker': marker,
            'objects': objects, 'donor': donors[0]}


def _transport(source, target, probe, *, scale=True, reflect=True):
    pr, pc, color = probe
    ar, ac = source['anchor']
    br, bc = target['anchor']
    su, tu = source['unit'], target['unit']
    offsets = [(pr - ar), (pc - ac)]
    size = tu if scale else su
    if scale:
        assert all(d % su == 0 for d in offsets)
        offsets = [d // su * tu for d in offsets]
    for axis in range(2):
        if reflect and source['flag'][axis] != target['flag'][axis]:
            offsets[axis] = 2 * tu - size - offsets[axis]
    return br + offsets[0], bc + offsets[1], size, color


def _render(data, *, flood=True, scale=True, reflect=True, strict=True):
    result = data['grid'].copy()
    donor = data['objects'][data['donor']]
    records = []
    for index, target in enumerate(data['objects']):
        if index == data['donor']:
            continue
        top, left = target['offset']
        panel = result[top:top + target['grid'].shape[0],
                       left:left + target['grid'].shape[1]]
        addressed = {}
        for probe in donor['probes']:
            r, c, size, color = _transport(donor, target, probe, scale=scale, reflect=reflect)
            inside = 0 <= r and 0 <= c and r + size <= panel.shape[0] and c + size <= panel.shape[1]
            if not inside:
                assert not strict, "projected address leaves its receiver"
                continue
            ids = np.unique(target['room_id'][r:r + size, c:c + size])
            if len(ids) != 1 or ids[0] < 0:
                assert not strict, "projected address straddles a wall or exterior"
                continue
            room = int(ids[0])
            if strict:
                assert room not in addressed, "addresses cannot collide"
            addressed[room] = color
            if flood:
                panel[tuple(target['rooms'][room].T)] = color
            else:
                panel[r:r + size, c:c + size] = color
        records.append({'object': index, 'addressed': addressed,
                        'blank': len(target['rooms']) - len(addressed)})
    return result, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from enclosure, visible marker sizes, and local flags."""
    return _render(_decode(input_grid))[0]


def _grow(rng, unit, height, width, sites, donor):
    """Random region growth with permanent one-cell walls between chambers."""
    h, w = height * unit, width * unit
    for _ in range(24):
        labels = np.full((h, w), -1, dtype=int)
        for i, (y, x) in enumerate(sites):
            # Donors retain a colored tail beyond each marker, even at unit 1.
            labels[y * unit:(y + (2 if donor else 1)) * unit,
                   x * unit:(x + 1) * unit] = i
        frontier = []
        for r, c in np.argwhere(labels >= 0):
            for dr, dc in _STEPS:
                frontier.append((int(r + dr), int(c + dc), int(labels[r, c])))
        while frontier:
            k = int(rng.integers(len(frontier)))
            r, c, label = frontier[k]
            frontier[k] = frontier[-1]
            frontier.pop()
            if not (0 < r < h - 1 and 0 < c < w - 1) or labels[r, c] >= 0:
                continue
            if any(labels[r + dr, c + dc] not in (-1, label) for dr, dc in _STEPS):
                continue
            labels[r, c] = label
            frontier.extend((r + dr, c + dc, label) for dr, dc in _STEPS)
        if len(_components(labels < 0, diagonal=True)) == 1:
            return labels
    # A guaranteed bounded fallback: disjoint rectangular rooms with wall gaps.
    labels = np.full((h, w), -1, dtype=int)
    for i, (y, x) in enumerate(sites):
        labels[y * unit:(y + 2) * unit, x * unit:(x + 1) * unit] = i
    return labels


def _object(rng, unit, height, width, sites, colors, roles, flips):
    bg, wall, marker = roles[:3]
    donor = colors is not None
    labels = _grow(rng, unit, height, width, sites, donor)
    inp = np.full(((height + 4) * unit, width * unit), bg, dtype=np.int8)
    floor = inp[4 * unit:]
    floor[labels < 0] = wall
    if donor:
        for i, color in enumerate(colors):
            floor[labels == i] = color
        for y, x in sites:
            floor[y * unit:(y + 1) * unit, x * unit:(x + 1) * unit] = marker
    inp[unit:2 * unit, unit:2 * unit] = wall
    inp[:unit, :unit] = marker
    inp[2 * unit:4 * unit, 2 * unit:4 * unit] = marker
    out = inp.copy()
    if not donor:
        for i, color in enumerate(roles[3]):
            out[4 * unit:][labels == i] = color
    for axis in range(2):
        if flips[axis]:
            inp = np.flip(inp, axis=axis)
            out = np.flip(out, axis=axis)
    return inp.copy(), out.copy()


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _make_pair(rng, index, test):
    # First two examples expose every relative flip and all three scale ratios.
    if not test and index == 0:
        su, dus, forced_rooms, decoys = 1, [1, 2], 3, 0
        relative = [(False, True), (True, False)]
    elif not test and index == 1:
        su, dus, forced_rooms, decoys = 2, [1, 1], 4, 1
        relative = [(False, False), (True, True)]
    else:
        target_count = int(rng.integers(1, 3))
        units = [int(x) for x in rng.integers(1, 3, size=target_count + 1)]
        while sum(units) > 4:
            units[int(rng.choice([i for i, u in enumerate(units) if u == 2]))] = 1
        su, dus = units[0], units[1:]
        forced_rooms, decoys = None, (2 if test else int(rng.integers(2)))
        relative = [tuple(bool(x) for x in rng.integers(2, size=2)) for _ in dus]
    n_targets = len(dus)
    max_width = min(7, (28 - n_targets) // (su + sum(dus)))
    width = int(rng.integers(5, max_width + 1))
    columns = list(range(1, width - 1, 2))
    min_rooms = forced_rooms or 3
    heights = [h for h in range(7, 11)
               if len(range(1, h - 2, 3)) * len(columns) >= min_rooms + decoys]
    height = int(rng.choice(heights))
    slots = [(y, x) for y in range(1, height - 2, 3) for x in columns]
    count = forced_rooms or int(rng.integers(3, min(6, len(slots) - decoys) + 1))
    order = rng.permutation(len(slots))
    sites = [slots[int(i)] for i in order[:count]]
    spare = [slots[int(i)] for i in order[count:]]
    palette = [int(x) for x in rng.permutation(10)]
    bg, wall, marker = palette[:3]
    richness = int(rng.integers(2, min(7, count) + 1))
    colors = palette[3:3 + richness]
    colors += [int(rng.choice(colors)) for _ in range(count - richness)]
    rng.shuffle(colors)
    base = tuple(bool(x) for x in rng.integers(2, size=2))
    objects = [_object(rng, su, height, width, sites, colors, (bg, wall, marker), base)]
    for unit, rel in zip(dus, relative):
        extra = [spare[int(i)] for i in rng.permutation(len(spare))[:decoys]]
        flips = tuple(a != b for a, b in zip(base, rel))
        objects.append(_object(rng, unit, height, width, sites + extra, None,
                               (bg, wall, marker, colors), flips))
    # Pack only the actual objects; there are no panel dividers or legend cells.
    if test:
        order = [1, 0, 2] if n_targets == 2 else [1, 0]
    elif index == 0:
        order = list(range(len(objects)))
    elif index == 1:
        order = list(range(1, len(objects))) + [0]
    else:
        order = [int(i) for i in rng.permutation(len(objects))]
    minimum_w = sum(o[0].shape[1] for o in objects) + n_targets + 2
    minimum_h = max(o[0].shape[0] for o in objects) + 2
    canvas_h = int(rng.integers(minimum_h, 31))
    canvas_w = int(rng.integers(minimum_w, 31))
    inp = np.full((canvas_h, canvas_w), bg, dtype=np.int8)
    out = inp.copy()
    slack = canvas_w - minimum_w
    allocations = rng.multinomial(slack, [1 / (n_targets + 2)] * (n_targets + 2))
    left = 1 + int(allocations[0])
    for pos, oi in enumerate(order):
        a, b = objects[oi]
        top = int(rng.integers(1, canvas_h - a.shape[0]))
        inp[top:top + a.shape[0], left:left + a.shape[1]] = a
        out[top:top + b.shape[0], left:left + b.shape[1]] = b
        left += a.shape[1] + 1 + int(allocations[pos + 1])
    # Common rotations preserve the input-visible, axis-relative rule.
    turns = int(rng.integers(4))
    return {'input': np.rot90(inp, turns).copy(), 'output': np.rot90(out, turns).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    num_train = _integer(num_train, 'num_train')
    num_test = _integer(num_test, 'num_test')
    if seed < 0 or not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('nonnegative seed; 2..10 train, 1..4 test, train > test required')
    rng = np.random.default_rng(seed)
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            pair = _make_pair(rng, index, split == 'test')
            # A finite recoloring fallback guarantees distinct inputs without
            # storing or retrying complete puzzles. There are only <=13 priors.
            if _key(pair['input']) in seen:
                original = pair
                for shift in range(1, 10):
                    for mult in (1, 3, 7, 9):
                        lookup = (np.arange(10) * mult + shift) % 10
                        candidate = {k: lookup[v].astype(np.int8) for k, v in original.items()}
                        if _key(candidate['input']) not in seen:
                            pair = candidate
                            break
                    if _key(pair['input']) not in seen:
                        break
                assert _key(pair['input']) not in seen
            seen.add(_key(pair['input']))
            result[split].append(pair)
    return result


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    data = _decode(inp)
    expected, records = _render(data)
    assert np.array_equal(out, expected), 'incorrect chamber activation'
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    changed = inp != out
    assert np.all(inp[changed] == data['background'])
    assert np.array_equal(inp == data['wall'], out == data['wall'])
    assert np.array_equal(inp == data['marker'], out == data['marker'])
    donor = data['objects'][data['donor']]
    r, c = donor['offset']
    assert np.array_equal(out[r:r + donor['grid'].shape[0], c:c + donor['grid'].shape[1]], donor['grid'])
    signs, ratios, blank, room_shapes, growth = [], [], [], [], []
    for rec in records:
        target = data['objects'][rec['object']]
        assert len(rec['addressed']) == len(donor['rooms'])
        assert rec['blank'] in (0, 1, 2)
        blank.append(rec['blank'])
        signs.append(tuple(a != b for a, b in zip(donor['flag'], target['flag'])))
        ratios.append((target['unit'], donor['unit']))
        for probe in donor['probes']:
            pr, pc, size, _ = _transport(donor, target, probe)
            ti = int(target['room_id'][pr, pc])
            si = int(donor['room_id'][probe[0], probe[1]])
            growth.append(int(np.sign(len(target['rooms'][ti]) - len(donor['rooms'][si]))))
        for points in target['rooms']:
            rr, cc, R, C = _box(points)
            room_shapes.append((R - rr, C - cc, len(points)))
    normalized = np.empty_like(inp)
    roles = {data['background']: 0, data['wall']: 1, data['marker']: 2}
    for color in inp.flat:
        roles.setdefault(int(color), len(roles))
    for color, role in roles.items():
        normalized[inp == color] = role
    return {'shape': inp.shape, 'rooms': len(donor['rooms']), 'receivers': len(records),
            'palette': (data['background'], data['wall'], data['marker']),
            'richness': len(set(donor['colors'])), 'blank': tuple(blank),
            'reflections': tuple(signs), 'ratios': tuple(ratios),
            'nonrectangular': sum(area < h * w for h, w, area in room_shapes),
            'area_signs': tuple(sorted(set(growth))),
            'without_A': not np.array_equal(_render(data, flood=False)[0], out),
            'without_B': not np.array_equal(_render(data, scale=False, reflect=False, strict=False)[0], out),
            'without_scale': not np.array_equal(_render(data, scale=False, strict=False)[0], out),
            'without_reflection': not np.array_equal(_render(data, reflect=False, strict=False)[0], out),
            'structure': hashlib.sha256(bytes(inp.shape) + normalized.tobytes()).hexdigest()}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, measurements = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            measurements[split].append(_analyze(pair))
    train = measurements['train']
    assert {flip for m in train for flip in m['reflections']} == {
        (False, False), (False, True), (True, False), (True, True)}
    assert {(a // np.gcd(a, b), b // np.gcd(a, b)) for m in train for a, b in m['ratios']} == {
        (1, 1), (1, 2), (2, 1)}
    assert {n for m in train for n in m['blank']} >= {0, 1}
    assert len({m['rooms'] for m in train}) >= 2
    assert all(m['without_A'] for m in train)
    for flag in ('without_B', 'without_scale', 'without_reflection'):
        assert any(m[flag] for m in train), f'training does not distinguish {flag}'
    assert all(set(m['blank']) == {2} for m in measurements['test'])


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for name in ('input', 'output'):
                h.update(bytes(pair[name].shape))
                h.update(pair[name].tobytes())
    return h.hexdigest()


def self_test():
    """42 fixed seeds; one puzzle live at a time; no disk writes or traces."""
    structures, shapes, palettes, rooms, reflections, ratios, puzzles = (set() for _ in range(7))
    nonrectangular = total = reused = 0
    area_signs, receiver_counts = set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _puzzle_digest(puzzle)
        assert digest == _puzzle_digest(generate_puzzle(seed))
        puzzles.add(digest)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _analyze(pair)
                total += 1
                structures.add(m['structure'])
                shapes.add(m['shape'])
                palettes.add(m['palette'])
                rooms.add(m['rooms'])
                reflections.update(m['reflections'])
                ratios.update((a // int(np.gcd(a, b)), b // int(np.gcd(a, b))) for a, b in m['ratios'])
                area_signs.update(m['area_signs'])
                receiver_counts.add(m['receivers'])
                nonrectangular += m['nonrectangular']
                reused += m['richness'] < m['rooms']
                # Input-only prediction must leave caller-owned arrays untouched.
                before = pair['input'].copy()
                assert np.array_equal(solve(pair['input']), pair['output'])
                assert np.array_equal(before, pair['input'])
    assert len(puzzles) == 42 and len(structures) >= 180
    assert len(shapes) >= 15 and len(palettes) >= 70
    assert rooms >= {3, 4, 5} and len(reflections) == 4 and len(ratios) == 3
    assert area_signs == {-1, 0, 1} and receiver_counts == {1, 2}
    assert nonrectangular >= 80 and reused >= 20
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(20261022, a, b)
            validate_generated_puzzle(p)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(0, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['output'].copy()
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': total, 'distinct_puzzles': len(puzzles),
            'color_normalized_structures': len(structures), 'canvas_shapes': len(shapes),
            'role_palettes': len(palettes), 'source_room_counts': sorted(rooms),
            'nonrectangular_receiver_rooms': nonrectangular, 'reused_color_pairs': reused,
            'all_legal_counts_checked': True, 'independent_solver': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
