"""One pulse through a field of scaffold-registered, two-state shutters.

No files, parent modules, random state, or demonstrations enter solve().
The construction renderer and the input reader use separate light algorithms.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


SEEDS = (0,257295271,329710964,73957879,32588652,936366682,544250249,
         318639001,429116173,655218033,812618831,493515170,852866320,
         311242175,169807559,45042671,48074372,61348537,825418089,
         369924060,296750638,799200096,330327799,815617209,726986770,
         768725285,436029080,16899749,287848319,16696410,527373959,
         972231013,618915421,396888751,747613798,545063212,443393640,
         296797234,487811710,952792985,644255569,612176149)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _sig(mapping):
    return tuple(sorted((y, x, int(c)) for (y, x), c in mapping.items()))


def _bounds(points):
    return (min(y for y, x in points), max(y for y, x in points),
            min(x for y, x in points), max(x for y, x in points))


def _groups(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = y + dy, x + dx
                    if q in remaining:
                        remaining.remove(q)
                        group.add(q)
                        stack.append(q)
        result.append(group)
    return result


def _fit(values, sign):
    """Fit the entire visible thin rail, including its observed stair phase."""
    t = np.arange(len(values))
    fits = []
    for run in (1, 2, 3):
        for phase in range(run):
            expected = values[0] + sign * ((t + phase) // run)
            if np.array_equal(values, expected) and abs(values[-1] - values[0]) >= 2:
                fits.append((run, phase))
    assert len(fits) == 1, "rail continuation must be unambiguous"
    return fits[0]


def _read_emitter(grid):
    bg = int(np.argmax(np.bincount(grid.ravel(), minlength=10)))
    assert np.count_nonzero(grid == bg) * 2 > grid.size
    matches = []
    for rotation in range(4):
        a = np.rot90(grid, -rotation)
        for wall in map(int, np.unique(a)):
            if wall == bg:
                continue
            pos = np.argwhere(a == wall)
            if len(pos) < 11:
                continue
            rear, mouth = int(pos[:, 1].min()), int(pos[:, 1].max())
            if mouth - rear < 4 or mouth >= a.shape[1] - 2:
                continue
            cap = np.flatnonzero(a[:, rear] == wall)
            if len(cap) < 3 or np.any(np.diff(cap) != 1):
                continue
            rails = [np.flatnonzero(a[:, x] == wall) for x in range(rear + 1, mouth + 1)]
            if not all(len(r) == 2 and r[1] - r[0] >= 2 for r in rails):
                continue
            low, high = np.array(rails).T
            if cap[0] != low[0] or cap[-1] != high[0]:
                continue
            try:
                fits = _fit(low, -1), _fit(high, 1)
            except AssertionError:
                continue
            footprint = np.zeros(a.shape, dtype=bool)
            for x in range(rear + 1, a.shape[1]):
                t = x - rear - 1
                lo = int(low[0]) - (t + fits[0][1]) // fits[0][0]
                hi = int(high[0]) + (t + fits[1][1]) // fits[1][0]
                footprint[:, x] = (np.arange(a.shape[0]) > lo) & (np.arange(a.shape[0]) < hi)
            chamber = footprint.copy()
            chamber[:, mouth + 1:] = False
            marker = chamber & (a != bg)
            colors = np.unique(a[marker])
            if len(colors) != 1 or not marker.any():
                continue
            light = int(colors[0])
            if light == wall or len(_groups(marker)) != 1:
                continue
            matches.append(dict(a=a, bg=bg, wall=wall, light=light, rear=rear,
                                mouth=mouth, footprint=footprint, chamber=chamber,
                                marker=marker, rotation=rotation, fits=fits))
    assert len(matches) == 1, "one visibly readable emitter is required"
    return matches[0]


def _rectangle(anchor, scale, union):
    y0, y1, x0, x1 = _bounds(union)
    return (anchor[0] + y0 * scale, anchor[1] + x0 * scale,
            (y1 - y0 + 1) * scale, (x1 - x0 + 1) * scale)


def _separate(a, b):
    y, x, h, w = a
    yy, xx, hh, ww = b
    return y + h < yy or yy + hh < y or x + w < xx or xx + ww < x


def _read_input(grid):
    info = _read_emitter(grid)
    a, bg, light = info['a'], info['bg'], info['light']
    foreground = (a != bg) & (a != info['wall']) & ~info['marker']
    objects = []
    for group in _groups(foreground):
        sensors = [p for p in group if a[p] == light]
        assert sensors, "every detached object has one receiving square"
        y0, y1, x0, x1 = _bounds(sensors)
        scale = y1 - y0 + 1
        assert 1 <= scale <= 3 and x1 - x0 + 1 == scale
        assert len(sensors) == scale * scale
        mapping = {}
        for y, x in group:
            logical = (y - y0) // scale, (x - x0) // scale
            mapping.setdefault(logical, set()).add(int(a[y, x]))
        raw = {}
        for (y, x), colors in mapping.items():
            assert len(colors) == 1
            c = next(iter(colors))
            block = a[y0 + y * scale:y0 + (y + 1) * scale,
                      x0 + x * scale:x0 + (x + 1) * scale]
            assert block.shape == (scale, scale) and np.all(block == c)
            raw[y, x] = c
        assert raw[0, 0] == light
        assert all(x >= 1 for (y, x), c in raw.items() if (y, x) != (0, 0))
        objects.append(dict(anchor=(y0, x0), scale=scale, raw=raw))
    signatures = sorted({_sig(o['raw']) for o in objects})
    assert len(signatures) == 2, "exactly two complete states must be visible"
    templates = [{(y, x): c for y, x, c in sig} for sig in signatures]
    assert { _sig(o['raw']) for o in objects if o['scale'] == 1 } == set(signatures)
    common = {p for p, c in templates[0].items() if templates[1].get(p) == c}
    assert (0, 0) in common and len(common) >= 4
    union = set(templates[0]) | set(templates[1])
    rects = []
    for obj in objects:
        obj['state'] = signatures.index(_sig(obj['raw']))
        rect = _rectangle(obj['anchor'], obj['scale'], union)
        y, x, h, w = rect
        assert y >= 0 and x > info['mouth'] + 1
        assert y + h <= a.shape[0] and x + w <= a.shape[1]
        assert all(_separate(rect, r) for r in rects), "reserve both state supports"
        rects.append(rect)
    info.update(objects=objects, templates=templates, common=common, union=union)
    return info


def _illumination(a, bg, wall, light, footprint, ignore_blocks=False):
    """Input-only lane scan. Sensors transmit; foreign material is opaque."""
    incident = np.zeros(a.shape, dtype=bool)
    for y in range(a.shape[0]):
        blocked = False
        for x in range(a.shape[1]):
            if not footprint[y, x]:
                continue
            incident[y, x] = not blocked
            if not ignore_blocks and a[y, x] not in (bg, wall, light):
                blocked = True
    return incident


def _draw(grid, mapping, anchor, scale):
    for (y, x), color in mapping.items():
        yy, xx = anchor[0] + y * scale, anchor[1] + x * scale
        grid[yy:yy + scale, xx:xx + scale] = color


def _face(incident, obj):
    y, x = obj['anchor']
    return incident[y:y + obj['scale'], x]


def _finish(a, bg, light, templates, objects, incident, mode='all'):
    out = a.copy()
    out[incident & (a == bg)] = light
    for obj in objects:
        face = _face(incident, obj)
        active = bool(face.all()) if mode == 'all' else bool(face.any())
        if active:
            current = templates[obj['state']]
            _draw(out, {p: bg for p in current}, obj['anchor'], obj['scale'])
            _draw(out, templates[1 - obj['state']], obj['anchor'], obj['scale'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, geometry, template registration and exposure from the input."""
    info = _read_input(_grid(input_grid))
    incident = _illumination(info['a'], info['bg'], info['wall'], info['light'], info['footprint'])
    out = _finish(info['a'], info['bg'], info['light'], info['templates'], info['objects'], incident)
    return np.rot90(out, info['rotation']).copy()


def _construction_light(a, bg, wall, light, footprint):
    """Union of strict downstream shadows, separate from solve's lane scan."""
    shadow = np.zeros(a.shape, dtype=bool)
    opaque = footprint & (a != bg) & (a != wall) & (a != light)
    for y, x in np.argwhere(opaque):
        shadow[y, x + 1:] = True
    return footprint & ~shadow


def _design(rng, palette, combined):
    _, _, light, carrier, ca, cb, stable = map(int, palette[:7])
    width = int(rng.integers(3, 5))
    depths = [int(rng.choice([2, 2, 2, 3])) for _ in range(2)]
    core = {(0, 0): light, (-1, 1): carrier, (0, 1): carrier, (1, 1): carrier}
    for x in range(2, width):
        core[0, x] = carrier
        for y in (-1, 1):
            if rng.random() < .5:
                core[y, x] = carrier
    if width == 4 and rng.random() < .5:
        core[0, 2] = stable
    templates = []
    for state, (sign, color) in enumerate(((-1, ca), (1, cb))):
        t = dict(core)
        for d in range(2, depths[state] + 1):
            t[sign * d, 1] = color
        for x in range(1, int(rng.integers(2, width + 1))):
            t[sign * depths[state], x] = color
        if combined:
            t[0, width - 1] = color
        templates.append(t)
    return templates, depths


def _scene(rng, style, test, palette, combined):
    bg, wall, light = map(int, palette[:3])
    templates, depths = _design(rng, palette, combined)
    union = set(templates[0]) | set(templates[1])
    uw = _bounds(union)[3] + 1
    height, width = int(rng.integers(25, 31)), int(rng.integers(26, 31))
    runs = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
    phases = [int(rng.integers(r)) for r in runs]
    minimum_visible = max(4, 2 * max(runs) + 1)
    n = int(rng.integers(minimum_visible, minimum_visible + 3))
    rear = int(rng.integers(0, 3))
    mouth = rear + n
    start = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
    extent = [start[j] + (n - 1 + phases[j]) // runs[j] for j in range(2)]
    center = int(rng.integers(max(extent[0], 7), min(height - extent[1], height - 6)))
    a = np.full((height, width), bg, dtype=np.int8)
    a[center - start[0]:center + start[1] + 1, rear] = wall
    footprint = np.zeros(a.shape, dtype=bool)
    for x in range(rear + 1, width):
        t = x - rear - 1
        lo = center - start[0] - (t + phases[0]) // runs[0]
        hi = center + start[1] + (t + phases[1]) // runs[1]
        footprint[max(0, lo + 1):min(height, hi), x] = True
        if x <= mouth:
            a[lo, x] = a[hi, x] = wall
    # Payload shape varies inside the closed-back chamber, independently of sensors.
    chamber_cells = [(y, x) for y, x in np.argwhere(footprint)
                     if rear < x <= min(rear + 2, mouth)]
    py, px = chamber_cells[int(rng.integers(len(chamber_cells)))]
    a[py, px] = light
    for qy, qx in rng.permutation(chamber_cells):
        if rng.random() < .35 and abs(int(qy) - int(py)) + abs(int(qx) - int(px)) == 1:
            a[qy, qx] = light
            break
    source_state = int(rng.integers(2))
    side = -1 if source_state == 0 else 1
    scale = (int(rng.choice([2, 3])) if test else 2) if style == 'partial' else 1
    depth = depths[source_state]
    # The enlarged face straddles the outermost occupied shadow lane.
    dy = -depth - (scale - 1) if side < 0 else depth
    ymin = min(-depths[0], dy - depths[0] * scale)
    ymax = max(depths[1], dy + (depths[1] + 1) * scale - 1)
    ychoices = [y for y in range(-ymin, height - ymax)
                if footprint[y, mouth + 2] and footprint[y + dy, min(width - 1, mouth + 2 + uw + 1)]]
    if not ychoices:
        return None
    sy = ychoices[int(rng.integers(len(ychoices)))]
    sx = mouth + int(rng.integers(2, 4))
    tx = sx + uw + int(rng.integers(1, 4))
    objects = [dict(anchor=(sy, sx), scale=1, state=source_state),
               dict(anchor=(sy + dy, tx), scale=scale, state=1 - source_state)]
    rects = [_rectangle(o['anchor'], o['scale'], union) for o in objects]
    if any(x + w > width or y < 0 or y + h > height for y, x, h, w in rects):
        return None
    for obj in objects:
        _draw(a, templates[obj['state']], obj['anchor'], obj['scale'])

    def add(desired_scale, desired_state, require=None, avoid_lane=None):
        incident = _construction_light(a, bg, wall, light, footprint)
        choices = []
        for yy in range(depths[0] * desired_scale, height - (depths[1] + 1) * desired_scale + 1):
            for xx in range(mouth + 2, width - uw * desired_scale + 1):
                obj = dict(anchor=(yy, xx), scale=desired_scale, state=desired_state)
                rect = _rectangle(obj['anchor'], desired_scale, union)
                if not all(_separate(rect, r) for r in rects):
                    continue
                if avoid_lane is not None and rect[0] <= avoid_lane < rect[0] + rect[2]:
                    continue
                face = _face(incident, obj)
                if require == 'active' and not face.all():
                    continue
                if require == 'outside' and _face(footprint, obj).any():
                    continue
                if isinstance(require, tuple) and tuple(map(bool, face)) != require:
                    continue
                choices.append((obj, rect))
        if not choices:
            return False
        obj, rect = choices[int(rng.integers(len(choices)))]
        _draw(a, templates[obj['state']], obj['anchor'], obj['scale'])
        objects.append(obj)
        rects.append(rect)
        return True

    # Both states are data and both exchange directions are demonstrated at unit scale.
    if not add(1, 1 - source_state, 'active', sy + side * depth):
        return None
    enlarged_scale = 3 if test else 2
    if style == 'basic' and not add(enlarged_scale, int(rng.integers(2)), 'active', sy + side * depth):
        return None
    if style == 'partial' and not test:
        # With only two demonstrations, show each possible one-lane failure.
        # No privileged sensor edge can then replace the all-lanes condition.
        first_pattern = tuple(map(bool, _face(_construction_light(a, bg, wall, light, footprint), objects[1])))
        if not add(2, int(rng.integers(2)), tuple(reversed(first_pattern))):
            return None
    # Optional unchanged exterior motif separates sensing from unconditional exchange.
    if rng.random() < .7:
        add(1, int(rng.integers(2)), 'outside')
    for _ in range(int(rng.integers(0, 3))):
        add(int(rng.choice([1, 1, 2])), int(rng.integers(2)))
    incident = _construction_light(a, bg, wall, light, footprint)
    if not _face(incident, objects[0]).all():
        return None
    target_hits = int(_face(incident, objects[1]).sum())
    if style == 'partial' and not 0 < target_hits < scale:
        return None
    if style == 'partial' and not test:
        patterns = {tuple(map(bool, _face(incident, o))) for o in objects if o['scale'] == 2}
        if not {(True, False), (False, True)} <= patterns:
            return None
    if style == 'basic' and target_hits != 0:
        return None
    if {o['state'] for o in objects if o['scale'] == 1 and _face(incident, o).all()} != {0, 1}:
        return None
    if style == 'basic' and not any(o['scale'] == enlarged_scale and _face(incident, o).all() for o in objects):
        return None
    out = _finish(a, bg, light, templates, objects, incident)
    rotation = int(rng.integers(4))
    return {'input': np.rot90(a, rotation).copy(), 'output': np.rot90(out, rotation).copy()}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt, ns = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    shared = rng.random() < .2
    shared_palette = rng.permutation(10)
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            style = 'basic' if split == 'train' and index == 0 else 'partial'
            if split == 'test':
                style = str(rng.choice(['partial', 'basic'], p=[.65, .35]))
            if split == 'train' and index >= 2:
                style = str(rng.choice(['basic', 'partial']))
            combined = not (split == 'train' and index == 0)
            for attempt in range(160):
                palette = shared_palette if shared else rng.permutation(10)
                pair = _scene(rng, style, split == 'test', palette, combined)
                if pair is None or _key(pair['input']) in seen:
                    continue
                try:
                    info = _analyze(pair)
                except AssertionError:
                    # Packing and palette proposals must also have a unique
                    # pixel-derived reading and the intended causal witness.
                    continue
                if not info['snapshot_witness']:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded scene construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    info = _read_input(inp)
    a, bg, light = info['a'], info['bg'], info['light']
    incident = _illumination(a, bg, info['wall'], light, info['footprint'])
    expected = _finish(a, bg, light, info['templates'], info['objects'], incident)
    assert np.array_equal(np.rot90(expected, info['rotation']), out), 'input-only prediction mismatch'
    objects = info['objects']
    active = [o for o in objects if _face(incident, o).all()]
    partial = [o for o in objects if 0 < _face(incident, o).sum() < o['scale']]
    shadowed = [o for o in objects if _face(info['footprint'], o).all() and not _face(incident, o).all()]
    snapshot_witness = False
    for source in active:
        counter = a.copy()
        _draw(counter, {p: bg for p in source['raw']}, source['anchor'], source['scale'])
        _draw(counter, info['templates'][1 - source['state']], source['anchor'], source['scale'])
        new_incident = _illumination(counter, bg, info['wall'], light, info['footprint'])
        if any(_face(new_incident, target).all() for target in shadowed):
            snapshot_witness = True
            break
    beam_only = a.copy()
    beam_only[incident & (a == bg)] = light
    unblocked = _illumination(a, bg, info['wall'], light, info['footprint'], True)
    ignore_a = _finish(a, bg, light, info['templates'], objects, unblocked)
    assert not np.array_equal(expected, beam_only), 'B must affect the answer'
    assert not np.array_equal(expected, ignore_a), 'A occlusion must affect the answer'
    assert active and shadowed and snapshot_witness
    assert {o['state'] for o in objects if o['scale'] == 1} == {0, 1}
    assert np.array_equal(a[a == info['wall']], expected[a == info['wall']])
    for obj in objects:
        for p in info['common']:
            yy = obj['anchor'][0] + p[0] * obj['scale']
            xx = obj['anchor'][1] + p[1] * obj['scale']
            s = obj['scale']
            assert np.array_equal(a[yy:yy+s, xx:xx+s], expected[yy:yy+s, xx:xx+s])
    info.update(active=active, partial=partial, shadowed=shadowed, incident=incident,
                snapshot_witness=snapshot_witness, expected=expected)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    infos, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp = _grid(pair['input'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            infos[split].append(_analyze(pair))
    training = infos['train']
    assert any(i['partial'] for i in training), 'teach full-face sensing'
    patterns = {tuple(map(bool, _face(i['incident'], o)))
                for i in training for o in i['objects'] if o['scale'] == 2}
    assert {(True, False), (False, True)} <= patterns, 'both receiving edges matter'
    assert any(o['scale'] > 1 for i in training for o in i['active']), 'teach scaled exchange'
    assert {o['state'] for i in training for o in i['active']} == {0, 1}
    # Every returned input itself provides both full local state templates.
    return None


def self_test():
    """42 seeds, bounded in memory; no sample/trace/log payloads are written."""
    # Literal geometry fixture: the next virtual stair pixel is excluded,
    # outside material is inert, and an in-cone blocker stops one straight lane.
    fixture = np.zeros((15, 14), dtype=np.int8)
    fixture[5:10, 1] = 2
    for x, lo in enumerate((5, 5, 4, 4, 3, 3), 2):
        fixture[lo, x] = fixture[14 - lo, x] = 2
    fixture[7, 2] = 3
    fixture[2, 3] = 5  # outside, although that lane enters the cone later
    fixture[7, 9] = 5
    e = _read_emitter(fixture)
    assert e['rotation'] == 0 and e['fits'] == ((2, 0), (2, 0))
    lit = _illumination(fixture, 0, 2, 3, e['footprint'])
    assert not lit[2, 8] and lit[3, 8] and lit[2, 10]
    assert lit[7, 9] and not lit[7, 10] and lit[6, 10] and lit[8, 10]
    assert np.array_equal(lit, _construction_light(fixture, 0, 2, 3, e['footprint']))
    shapes, outlines, designs, palettes, rotations, schedules, counts = (set() for _ in range(7))
    fingerprints = set()
    stats = dict(pairs=0, active=0, inactive=0, partial=0, snapshot=0, outside=0, active_scale_three=0)
    for seed in SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze(pair)
                assert np.array_equal(solve(pair['input']), pair['output'])
                digest.update(pair['input'].tobytes())
                shapes.add(pair['input'].shape)
                rotations.add(info['rotation'])
                wy, wx = np.nonzero(info['a'] == info['wall'])
                outlines.add(tuple(sorted(zip(map(int, wy - wy.min()), map(int, wx - wx.min())))))
                rolemap = {info['bg']: 0, info['light']: 1}
                design = tuple(tuple((y,x,rolemap.setdefault(c,len(rolemap))) for y,x,c in _sig(t))
                               for t in info['templates'])
                designs.add(design)
                palettes.add((info['bg'], info['wall'], info['light']))
                schedules.add(tuple(sorted(o['scale'] for o in info['objects'])))
                counts.add(len(info['objects']))
                stats['pairs'] += 1
                stats['active'] += len(info['active'])
                stats['active_scale_three'] += sum(int(o['scale'] == 3) for o in info['active'])
                stats['inactive'] += len(info['objects']) - len(info['active'])
                stats['partial'] += len(info['partial'])
                stats['snapshot'] += info['snapshot_witness']
                stats['outside'] += sum(not _face(info['footprint'], o).any() for o in info['objects'])
                if info['partial']:
                    rival = _finish(info['a'], info['bg'], info['light'], info['templates'],
                                    info['objects'], info['incident'], mode='any')
                    assert not np.array_equal(rival, info['expected'])
        fingerprints.add(digest.hexdigest())
    assert len(fingerprints) == 42 and len(shapes) >= 30 and len(outlines) >= 50
    assert len(designs) >= 70 and len(palettes) >= 60 and rotations == set(range(4))
    assert len(schedules) >= 10 and len(counts) >= 3
    assert stats['partial'] >= 42 and stats['outside'] >= 10
    assert stats['active_scale_three'] >= 4, 'scale three must sometimes exchange'
    # Counts are API parameters, rather than a hidden source of task semantics.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(91, nt, ns)
            assert len(p['train']) == nt and len(p['test']) == ns
    for nt, ns in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(p[k], q[k]) for s in a for p,q in zip(a[s],b[s]) for k in p)
    damaged = {s: [{k:v.copy() for k,v in p.items()} for p in a[s]] for s in a}
    damaged['test'][0]['output'][0,0] = (int(damaged['test'][0]['output'][0,0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    return dict(stats, seeds=42, shapes=len(shapes), outlines=len(outlines), designs=len(designs),
                palettes=len(palettes), scale_schedules=len(schedules), motif_counts=sorted(counts))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
