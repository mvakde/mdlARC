"""Prismatic stencil relays: input-local, orientation-addressed color masks.

Only the standard library and NumPy are used. Construction retains explicit
records; solve reconstructs the plates, code, membership and addresses from
pixels. No training data, random seed or construction records enter solve.
"""
from __future__ import annotations

import copy
import json
import numpy as np


HOST_SEEDS = (0, 908144162, 187174043, 238537529, 119549757, 748856250,
              614158671, 564073777, 156721941, 400617933, 37934169, 781484179,
              323115867, 386406356, 677930473, 138148396, 407370023, 141510089,
              306205447, 158962691, 372581709, 192956783, 629796058, 26244755,
              220328035, 566553704, 540867670, 490366553, 963774056, 526378461,
              991427860, 697516736, 129475272, 84961466, 266501190, 830917895,
              442137981, 349498560, 7563377, 245666710, 608100827, 88204351)


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _checked_grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _components(a, bg):
    remaining = set(map(tuple, np.argwhere(a != bg)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], set()
        while stack:
            r, c = stack.pop()
            cells.add((int(r), int(c)))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in remaining:
                        remaining.remove(p)
                        stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells) + 1, max(c for r, c in cells) + 1)


def _rect(shape, r, c, h, w):
    return {(y, x) for y in range(max(0, r), min(shape[0], r + h))
            for x in range(max(0, c), min(shape[1], c + w))}


def _halo(cells, shape):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]}


def _plate(template, t, r, c):
    a = _orient(template, t)
    labels = _orient(np.arange(template.size).reshape(template.shape), t)
    positions = {int(labels[y, x]): (int(y), int(x))
                 for y, x in np.ndindex(labels.shape)}
    return dict(arr=a, t=t, r=r, c=c, pos=positions,
                cells=_rect((30, 30), r, c, *a.shape))


def _ray(plate, side, slot, canonical_shape, shape):
    h, w = canonical_shape
    y, x = plate['pos'][side * w + slot]
    iy, ix = plate['pos'][(1 if side == 0 else h - 2) * w + slot]
    dr, dc = y - iy, x - ix
    r, c = plate['r'] + y + dr, plate['c'] + x + dc
    result = []
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        result.append((r, c))
        r, c = r + dr, c + dc
    return tuple(result), (dr, dc)


def _corridors(plates, canonical_shape, shape):
    occupied = set().union(*(p['cells'] for p in plates))
    endpoints = {}
    rays = {}
    for i, plate in enumerate(plates):
        for side in (0, canonical_shape[0] - 1):
            for slot in range(canonical_shape[1]):
                ray, direction = _ray(plate, side, slot, canonical_shape, shape)
                rays[i, side, slot] = ray
                if ray and not occupied.intersection(ray):
                    endpoints.setdefault(ray[-1], []).append((i, side, slot))
    return occupied, endpoints, rays


def _stamp(out, p, mask, color):
    r, c = p
    h = min(mask.shape[0], out.shape[0] - r)
    w = min(mask.shape[1], out.shape[1] - c)
    if h > 0 and w > 0:
        out[r:r + h, c:c + w][mask[:h, :w]] = color


def _answer(info, mode='rule'):
    out = np.full(info['shape'], info['bg'], dtype=np.int8)
    for p in info['preserved']:
        out[p] = info['input'][p]
    points = info['members'] if mode != 'all_markers' else info['points']
    for p in points:
        owners = info['addresses'].get(p, ())
        if len(owners) != 1:
            continue
        i, slot = owners[0]
        if mode == 'local_only' and (i, slot) not in info['observed']:
            continue
        color = info['code'][slot]
        if mode == 'unoriented':
            mask = info['canonical'] == color
        elif mode == 'unreflected':
            mask = _orient(info['canonical'] == color, info['plates'][i]['t'] % 4)
        elif mode == 'solid':
            mask = np.ones(info['plates'][i]['arr'].shape, dtype=bool)
        else:
            mask = info['plates'][i]['arr'] == color
        _stamp(out, p, mask, color)
    return out


def _decode(value):
    inp = _checked_grid(value)
    values, counts = np.unique(inp, return_counts=True)
    assert counts.max() > inp.size // 2, 'background must be a strict majority'
    bg, shape = int(values[np.argmax(counts)]), inp.shape
    components = _components(inp, bg)
    plates, payload = [], []
    for cells in components:
        r, c, rr, cc = _bbox(cells)
        a = inp[r:rr, c:cc]
        is_plate = (a.size == len(cells) and 3 <= min(a.shape) <= 4
                    and min(a.shape) < max(a.shape) <= 7
                    and 2 <= len(np.unique(a)) <= 3)
        if is_plate:
            plates.append(dict(arr=a.copy(), r=r, c=c, cells=cells))
        else:
            payload.append(cells)
    assert 2 <= len(plates) <= 4, 'two to four identifiable mosaic plates required'
    canonical = plates[0]['arr'].copy()
    if canonical.shape[0] > canonical.shape[1]:
        canonical = np.rot90(canonical)
    forms = [_orient(canonical, t) for t in range(8)]
    assert len({_key(a) for a in forms}) == 8, 'orientation must be unique'
    channels = set(map(int, np.unique(canonical)))
    assert all(np.count_nonzero(canonical == k) >= 3 for k in channels)
    assert all(len(_components((canonical == k).astype(np.int8), 0)) == 1
               for k in channels), 'each color layer is eight-connected'
    for plate in plates:
        ts = [t for t, a in enumerate(forms) if np.array_equal(a, plate['arr'])]
        assert len(ts) == 1, 'plates must be exact transformed copies'
        plate.update(_plate(canonical, ts[0], plate['r'], plate['c']))
    occupied, endpoints, rays = _corridors(plates, canonical.shape, shape)
    signals, materials = [], []
    for cells in payload:
        if len(cells) == 1 and int(inp[next(iter(cells))]) in channels:
            p = next(iter(cells))
            assert p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)
            signals.append(p)
        else:
            materials.append(cells)
    assert signals, 'boundary observations required'
    code, sides, observed = {}, set(), set()
    for p in signals:
        matches = endpoints.get(p, [])
        assert len(matches) == 1, 'a boundary observation needs one clear plate address'
        i, side, slot = matches[0]
        color = int(inp[p])
        assert slot not in code or code[slot] == color, 'inconsistent redundant clues'
        code[slot] = color
        sides.add(side)
        observed.add((i, slot))
    assert len(sides) == 1 and 1 <= len(code) <= 3
    side = sides.pop()
    addresses = {}
    directions = []
    for i, plate in enumerate(plates):
        for slot in code:
            ray = rays[i, side, slot]
            assert len(ray) >= 2 and not occupied.intersection(ray)
            for p in ray:
                addresses.setdefault(p, []).append((i, slot))
        directions.append(_ray(plate, side, next(iter(code)), canonical.shape, shape)[1])
    marker_values = {int(inp[p]) for cells in materials for p in cells} - channels
    assert len(marker_values) == 1, 'one disjoint wildcard receiver color required'
    marker = marker_values.pop()
    members, points, rejected, bare, painted = [], [], [], [], []
    for cells in materials:
        ps = [p for p in cells if int(inp[p]) == marker]
        points.extend(ps)
        r, c, rr, cc = _bbox(cells)
        if len(ps) == 1 and ps[0] == (r, c):
            members.append(ps[0])
            if len(cells) > 1:
                painted.append(ps[0])
        elif ps:
            rejected.extend(ps)
        else:
            bare.append(cells)
    preserved = occupied | set(signals)
    active, windows, clipping = [], set(), set()
    for p in points:
        assert len(addresses.get(p, ())) <= 1, 'receiver ownership tie'
    for p in members:
        owners = addresses.get(p, ())
        if not owners:
            continue
        i, slot = owners[0]
        mask = plates[i]['arr'] == code[slot]
        window = _rect(shape, *p, *mask.shape)
        assert not (window & (windows | preserved)), 'conflicting requested windows'
        windows.update(window)
        if p[0] + mask.shape[0] > shape[0]:
            clipping.add('bottom')
        if p[1] + mask.shape[1] > shape[1]:
            clipping.add('right')
        if p[0] + mask.shape[0] > shape[0] and p[1] + mask.shape[1] > shape[1]:
            clipping.add('corner')
        if p in painted:
            # A pre-rendered live component must be precisely its addressed mask
            # plus the marker at a blank origin, not an arbitrary attachment.
            component = next(cells for cells in materials if p in cells)
            expected = {(p[0] + int(r), p[1] + int(c))
                        for r, c in np.argwhere(mask)
                        if p[0] + r < shape[0] and p[1] + c < shape[1]}
            assert not mask[0, 0] and component == expected | {p}
            assert all(int(inp[q]) == code[slot] for q in expected)
        active.append((p, i, slot))
    assert active, 'at least one addressed receiver required'
    info = dict(input=inp, shape=shape, bg=bg, marker=marker, channels=channels,
                canonical=canonical, plates=plates, code=code, side=side,
                signals=signals, observed=observed, addresses=addresses,
                preserved=preserved, members=members, points=points,
                rejected=rejected, bare=bare, painted=painted, active=active,
                clipping=clipping, directions=tuple(directions), materials=materials)
    info['output'] = _answer(info)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode an individual input; do not consult any other example or state."""
    return _decode(input_grid)['output']


def _sample_mosaic(rng, h, w, colors):
    for attempt in range(220):
        a = rng.choice(colors, size=(h, w)).astype(np.int8)
        if attempt % 4 == 0:
            a[0] = colors[0]
            a[-1] = colors[1]
        if all(np.count_nonzero(a == c) >= 3 and
               len(_components((a == c).astype(np.int8), 0)) == 1 for c in colors):
            if len({_key(_orient(a, t)) for t in range(8)}) == 8:
                return a
    # Connected cap/arm partition with unique colored correspondence.
    a = np.full((h, w), colors[0], dtype=np.int8)
    a[-1] = colors[1]
    a[-2, -1] = colors[1]
    if len(colors) == 3:
        a[0, -2:] = colors[2]
        a[1, -1] = colors[2]
    assert len({_key(_orient(a, t)) for t in range(8)}) == 8
    return a


def _place_plates(rng, template, transforms, shape, fallback):
    if fallback:
        return [_plate(template, t, r, c)
                for t, r, c in ((0, 12, 2), (4, 12, 16), (3, 2, 22))]
    result, occupied = [], set()
    for t in transforms:
        h, w = _orient(template, t).shape
        for _ in range(60):
            r = int(rng.integers(1, shape[0] - h))
            c = int(rng.integers(1, shape[1] - w))
            plate = _plate(template, t, r, c)
            if not _halo(plate['cells'], shape).intersection(occupied):
                result.append(plate)
                occupied.update(plate['cells'])
                break
        else:
            return None
    return result


def _choose_clues(rng, plates, template, shape, side, count, mode):
    occupied, endpoints, rays = _corridors(plates, template.shape, shape)
    choices = {}
    for slot in range(template.shape[1]):
        if any(len(rays[i, side, slot]) < 2 or occupied.intersection(rays[i, side, slot])
               for i in range(len(plates))):
            continue
        options = []
        for i in range(len(plates)):
            p = rays[i, side, slot][-1]
            if endpoints.get(p) == [(i, side, slot)] and not _halo({p}, shape).intersection(occupied):
                options.append(i)
        if options:
            choices[slot] = options
    if len(choices) < count:
        return None
    for _ in range(50):
        slots = sorted(map(int, rng.choice(list(choices), count, replace=False)))
        if mode == 'full':
            common = set.intersection(*(set(choices[s]) for s in slots))
            if not common:
                continue
            owner = int(rng.choice(sorted(common)))
            owners = [owner] * count
        else:
            # At most two clue-bearing plates, leaving an unhinted third.
            group = list(map(int, rng.choice(len(plates), 2, replace=False)))
            options = [[i for i in choices[s] if i in group] for s in slots]
            if not all(options):
                continue
            owners = [int(rng.choice(o)) for o in options]
            if len(set(owners)) < 2:
                continue
        ps = [rays[i, side, s][-1] for i, s in zip(owners, slots)]
        if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1
               for j, p in enumerate(ps) for q in ps[j + 1:]):
            continue
        return slots, list(zip(owners, slots, ps)), occupied, rays
    return None


def _construct_pair(rng, kind, nplates, slot_count, live_goal, bare_required,
                    attached_required, painted_required, allowed_clipping=None):
    """Bounded random packing, followed by a compact spacious layout fallback."""
    pooled = kind in ('pool', 'test')
    multi = kind in ('full', 'test')
    clip = kind in ('full', 'test')
    for attempt in range(120):
        fallback = attempt >= 90
        h = 3 if fallback else int(rng.integers(3, 5))
        w = 4 if fallback else int(rng.integers(h + 1, 8))
        minimum = 26 if kind == 'test' else 18 if kind == 'ordinary' and nplates == 2 else 23
        shape = (30, 30) if fallback else tuple(map(int, rng.integers(minimum, 31, size=2)))
        palette = list(map(int, rng.permutation(10)))
        bg, marker = palette[:2]
        colors = palette[2:2 + int(rng.integers(2, 4))]
        template = _sample_mosaic(rng, h, w, colors)
        side = (h - 1 if allowed_clipping == {'bottom'} else 0) if fallback else int(rng.choice([0, h - 1]))
        if kind == 'pool':
            transforms = [0, 4, 1]
        elif kind == 'full':
            transforms = [0, 1, 2]
        else:
            transforms = [0, 4, 1][:nplates] + [int(rng.integers(8)) for _ in range(nplates - 3)]
        if kind == 'ordinary' and rng.random() < .5:
            transforms = list(map(int, rng.integers(8, size=nplates)))
        if not fallback:
            rng.shuffle(transforms)
        plates = _place_plates(rng, template, transforms, shape, fallback)
        if plates is None:
            continue
        count = min(slot_count, 2 if fallback else slot_count)
        clue_result = _choose_clues(rng, plates, template, shape, side, count,
                                     'pool' if pooled else 'full')
        if clue_result is None:
            continue
        slots, clues, occupied, rays = clue_result
        chosen_colors = list(map(int, rng.choice(colors, count, replace=True)))
        if count >= 2:
            chosen_colors[:2] = list(map(int, rng.choice(colors, 2, replace=False)))
        rng.shuffle(chosen_colors)
        code = dict(zip(slots, chosen_colors))
        inp = np.full(shape, bg, dtype=np.int8)
        for plate in plates:
            r, c = plate['r'], plate['c']
            hh, ww = plate['arr'].shape
            inp[r:r + hh, c:c + ww] = plate['arr']
        preserved = occupied | {p for i, s, p in clues}
        for i, s, p in clues:
            inp[p] = code[s]
        addresses = {}
        for i in range(len(plates)):
            for s in slots:
                for p in rays[i, side, s]:
                    addresses.setdefault(p, []).append((i, s))
        candidates = []
        forbidden = _halo(preserved, shape)
        for p, owners in addresses.items():
            if len(owners) != 1 or p in forbidden:
                continue
            i, s = owners[0]
            mask = plates[i]['arr'] == code[s]
            window = _rect(shape, *p, *mask.shape)
            if window & preserved or not mask[:shape[0] - p[0], :shape[1] - p[1]].any():
                continue
            clipped = p[0] + mask.shape[0] > shape[0] or p[1] + mask.shape[1] > shape[1]
            categories = set()
            if p[0] + mask.shape[0] > shape[0]:
                categories.add('bottom')
            if p[1] + mask.shape[1] > shape[1]:
                categories.add('right')
            if len(categories) == 2:
                categories.add('corner')
            if allowed_clipping is not None and not categories <= allowed_clipping:
                continue
            if kind == 'pool' and clipped:
                continue
            candidates.append(dict(p=p, i=i, s=s, mask=mask, window=window, clipped=clipped,
                                   paint_ok=not mask[0, 0] and mask[:2, :2].any() and not clipped))
        rng.shuffle(candidates)
        selected, used = [], set(preserved)
        # Reserve the rejected on-channel request before packing live targets.
        # Otherwise dense layouts can consume every place that demonstrates
        # why component membership is needed in addition to ray membership.
        attachment = None
        if attached_required:
            possible = list(addresses.items())
            rng.shuffle(possible)
            for p, owners in possible:
                if len(owners) != 1:
                    continue
                i, s = owners[0]
                mask = plates[i]['arr'] == code[s]
                last = max((int(r), int(c)) for r, c in np.argwhere(mask))
                origin = (p[0] - last[0] - 1, p[1] - last[1] - 1)
                if min(origin) < 0 or origin[0] + mask.shape[0] > shape[0] or origin[1] + mask.shape[1] > shape[1]:
                    continue
                hypothetical = _rect(shape, *p, *mask.shape)
                artifact = _rect(shape, *origin, *mask.shape) | {p}
                reserve = artifact | hypothetical
                if _halo(reserve, shape).intersection(used):
                    continue
                if not mask[:shape[0] - p[0], :shape[1] - p[1]].any():
                    continue
                attachment = (p, origin, mask, code[s], reserve)
                used |= reserve
                break
            if attachment is None:
                continue

        def add(entry):
            nonlocal used
            # Full windows have a halo, ensuring input components stay separate.
            if entry in selected or _halo(entry['window'], shape).intersection(used):
                return False
            if kind == 'pool' and any((e['i'], e['s']) == (entry['i'], entry['s']) for e in selected):
                return False
            selected.append(entry)
            used |= entry['window']
            return True

        success = True
        if multi:
            success = False
            for first in candidates:
                for second in candidates:
                    if first['p'] == second['p'] or (first['i'], first['s']) != (second['i'], second['s']):
                        continue
                    if _halo(first['window'], shape).intersection(second['window']):
                        continue
                    if any(_halo(e['window'], shape).intersection(used) for e in (first, second)):
                        continue
                    if add(first) and add(second):
                        success = True
                        break
                if success:
                    break
        if not success:
            continue
        # Every plate receives a live target: transformations cannot be ornamental.
        for i in range(len(plates)):
            if not any(e['i'] == i for e in selected):
                if not any(add(e) for e in candidates if e['i'] == i):
                    success = False
                    break
        if not success:
            continue
        if clip and not any(e['clipped'] for e in selected):
            if not any(add(e) for e in candidates if e['clipped']):
                continue
        paint_entry = next((e for e in selected if e['paint_ok']), None)
        if painted_required and paint_entry is None:
            paint_entry = next((e for e in candidates if e['paint_ok'] and add(e)), None)
            if paint_entry is None:
                continue
        for e in candidates:
            if len(selected) >= max(live_goal, len(plates)):
                break
            add(e)
        for e in selected:
            inp[e['p']] = marker
        if painted_required:
            _stamp(inp, paint_entry['p'], paint_entry['mask'], code[paint_entry['s']])
            inp[paint_entry['p']] = marker
        # A lower-right attached marker sits on an ACTIVE address. Filtering
        # components, rather than ray gating alone, is necessary to reject it.
        if attached_required:
            p, origin, mask, color, reserve = attachment
            _stamp(inp, origin, mask, color)
            inp[p] = marker
        # A fitting isolated off-address marker defeats any "stamp every point"
        # rule. It is not declared invalid by hidden generation metadata.
        free = list(np.ndindex(shape))
        rng.shuffle(free)
        off = None
        for p in free:
            window = _rect(shape, *p, h, w)
            if p not in addresses and p[0] + h <= shape[0] and p[1] + w <= shape[1]:
                if not _halo(window, shape).intersection(used):
                    off = p
                    inp[p] = marker
                    used |= window
                    break
        if off is None:
            continue
        if bare_required:
            material = None
            for p in free:
                i = int(rng.integers(len(plates)))
                color = int(rng.choice(colors))
                mask = plates[i]['arr'] == color
                window = _rect(shape, *p, *mask.shape)
                if p[0] + mask.shape[0] > shape[0] or p[1] + mask.shape[1] > shape[1]:
                    continue
                if not _halo(window, shape).intersection(used):
                    material = (p, mask, color)
                    break
            if material is None:
                continue
            _stamp(inp, *material)
        out = np.full(shape, bg, dtype=np.int8)
        for p in preserved:
            out[p] = inp[p]
        for e in selected:
            _stamp(out, e['p'], e['mask'], code[e['s']])
        try:
            info = _decode(inp)
            assert np.array_equal(info['output'], out)
            assert len(info['active']) == len(selected)
            if kind == 'pool':
                assert not np.array_equal(_answer(info, 'local_only'), out)
                assert not np.array_equal(_answer(info, 'unoriented'), out)
                assert not np.array_equal(_answer(info, 'unreflected'), out)
                assert any(info['canonical'][info['side'], s] != color
                           for s, color in info['code'].items())
            if attached_required:
                assert not np.array_equal(_answer(info, 'all_markers'), out)
            if painted_required:
                assert info['painted']
            if clip:
                assert info['clipping']
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    return _fallback_pair(rng, kind, bare_required, attached_required,
                          painted_required, allowed_clipping)


def _fallback_pair(rng, kind, bare, attached, painted, allowed_clipping=None):
    """A checked compact construction, used only after bounded random packing.

    Its asymmetric partition guarantees blank-origin pre-rendering, a visible
    reflected mask, and a nonempty clipped request on an unhinted plate. Public
    example counts are unaffected; requested internal density is a preference.
    """
    palette = list(map(int, rng.permutation(10)))
    bg, marker, a, b, c = palette[:5]
    template = np.array([[a, a, c, c], [a, a, b, c], [b, b, b, c]], dtype=np.int8)
    shape, slots, code = (30, 30), (0, 2), {0: a, 2: b}
    plates = _place_plates(rng, template, (0, 4, 1), shape, True)
    bottom = kind == 'test' and allowed_clipping == {'bottom'}
    side = 2 if bottom else 0
    if bottom:
        observations = [(0, 0), (2, 2)]
        records = [((17, 4), 0, 2), ((17, 19), 1, 0),
                   ((23, 19), 1, 0), ((28, 17), 1, 2), ((2, 9), 2, 0)]
        paint_record = records[-1]
        attach_record = ((25, 2), (23, 0), 0, 0)
    elif kind == 'pool':
        observations = [(0, 0), (1, 2)]
        records = [((7, 4), 0, 2), ((7, 19), 1, 0),
                   ((2, 17), 1, 2), ((2, 26), 2, 0)]
        paint_record = records[2]
        attach_record = ((8, 19), (6, 15), 1, 0)
    else:
        observations = [(0, 0), (1, 2)] if kind == 'test' else [(0, 0), (0, 2)]
        records = [((3, 2), 0, 0), ((8, 2), 0, 0),
                   ((2, 17), 1, 2), ((2, 28), 2, 0)]
        paint_record = records[2]
        attach_record = ((8, 19), (6, 15), 1, 0)
    inp = np.full(shape, bg, dtype=np.int8)
    preserved = set()
    for plate in plates:
        r, col = plate['r'], plate['c']
        h, w = plate['arr'].shape
        inp[r:r + h, col:col + w] = plate['arr']
        preserved |= plate['cells']
    for i, s in observations:
        endpoint = _ray(plates[i], side, s, template.shape, shape)[0][-1]
        inp[endpoint] = code[s]
        preserved.add(endpoint)
    for p, i, s in records:
        inp[p] = marker
    if painted:
        p, i, s = paint_record
        _stamp(inp, p, plates[i]['arr'] == code[s], code[s])
        inp[p] = marker
    if attached:
        p, origin, i, s = attach_record
        _stamp(inp, origin, plates[i]['arr'] == code[s], code[s])
        inp[p] = marker
    inp[22, 9] = marker  # fitting, isolated, off-address in either side regime
    if bare:
        _stamp(inp, (20, 12), template == c, c)
    out = np.full(shape, bg, dtype=np.int8)
    for p in preserved:
        out[p] = inp[p]
    for p, i, s in records:
        _stamp(out, p, plates[i]['arr'] == code[s], code[s])
    info = _decode(inp)
    assert np.array_equal(info['output'], out)
    _check_material(info)
    if allowed_clipping is not None:
        assert info['clipping'] <= allowed_clipping
    return {'input': inp, 'output': out}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    # Tests recombine split observations, multiple receivers and clipping.
    for split, count in (('train', nt), ('test', nq)):
        for j in range(count):
            kind = ('pool' if j == 0 else 'full' if j == 1 else 'ordinary') if split == 'train' else 'test'
            nplates = 3 if kind in ('pool', 'full') else int(rng.integers(3 if kind == 'test' else 2, 5))
            slots = 2 if kind in ('pool', 'full') else int(rng.integers(2 if kind == 'test' else 1, 4))
            live_goal = int(rng.integers(3, 6 if split == 'train' else 8))
            bare = kind == 'pool' or rng.random() < .3
            attached = kind == 'full' or (kind != 'pool' and rng.random() < .35)
            painted = kind == 'full' or (kind != 'pool' and rng.random() < .3)
            for _ in range(8):
                allowed = None if split == 'train' else set().union(
                    *(_decode(p['input'])['clipping'] for p in puzzle['train']))
                pair = _construct_pair(rng, kind, nplates, slots, live_goal, bare, attached, painted, allowed)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _crop(mask):
    r, c = np.nonzero(mask)
    return mask[r.min():r.max() + 1, c.min():c.max() + 1]


def _check_material(info):
    """Verify the generated bare/attached material against visible color layers."""
    inp, marker, channels = info['input'], info['marker'], info['channels']
    for cells in info['materials']:
        points = [p for p in cells if int(inp[p]) == marker]
        if len(cells) == 1:
            assert len(points) == 1
            continue
        assert len(points) <= 1
        inks = {int(inp[p]) for p in cells} & channels
        assert len(inks) == 1
        color = inks.pop()
        ink_cells = cells - set(points)
        r, c, rr, cc = _bbox(ink_cells)
        actual = np.zeros((rr - r, cc - c), dtype=bool)
        for y, x in ink_cells:
            actual[y - r, x - c] = True
        assert any(np.array_equal(actual, _crop(_orient(info['canonical'] == color, t)))
                   for t in range(8)), 'drawn material is not a visible stencil'
        if points and points[0] not in info['painted']:
            last = max(ink_cells)
            assert points[0] == (last[0] + 1, last[1] + 1), 'unsupported attachment'


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _checked_grid(pair['input']), _checked_grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(info['output'], out), 'incorrect input-derived output'
            assert not np.array_equal(inp, out)
            _check_material(info)
            infos[split].append(info)
    train = infos['train']
    assert any(not np.array_equal(_answer(i, 'local_only'), i['output']) for i in train), 'pooling untaught'
    assert any(not np.array_equal(_answer(i, 'unoriented'), i['output']) for i in train), 'orientation transfer untaught'
    assert any(not np.array_equal(_answer(i, 'unreflected'), i['output']) for i in train), 'reflection transfer untaught'
    assert any(not np.array_equal(_answer(i, 'all_markers'), i['output']) for i in train), 'component filtering untaught'
    assert any(not np.array_equal(_answer(i, 'solid'), i['output']) for i in train), 'stencil holes untaught'
    assert any(i['bare'] for i in train) and any(i['painted'] for i in train)
    assert any(any(p not in i['addresses'] for p in i['members']) for i in train), 'ray gating untaught'
    assert any(any(i['canonical'][i['side'], s] != color for s, color in i['code'].items())
               for i in train), 'boundary code must override a plate-edge-color shortcut'
    assert any(i['clipping'] for i in train), 'clipping untaught'
    assert any(len({(owner, s) for p, owner, s in i['active']}) < len(i['active']) for i in train), 'multiple receivers untaught'
    taught_clipping = set().union(*(i['clipping'] for i in train))
    assert all(i['clipping'] <= taught_clipping for i in infos['test']), 'untaught test boundary category'


def _features(info):
    masks = tuple(_key(info['canonical'] == c) for c in sorted(info['channels']))
    return dict(shape=info['shape'], plate_shape=info['canonical'].shape,
                plates=len(info['plates']), slots=len(info['code']), masks=masks,
                background=info['bg'], palette=tuple(sorted(info['channels'])),
                transforms=tuple(p['t'] for p in info['plates']),
                live=len(info['active']), off=len(info['members']) - len(info['active']),
                attached=len(info['rejected']), bare=len(info['bare']), painted=len(info['painted']),
                clipping=tuple(sorted(info['clipping'])),
                pooled=all(sum(j == i for j, s in info['observed']) < len(info['code'])
                           for i in range(len(info['plates']))),
                multiple=len({(i, s) for p, i, s in info['active']}) < len(info['active']))


def self_test() -> dict:
    """Bounded 42-seed suite, semantic ablations, mutation and recoloring checks."""
    signatures, features, normalized = set(), [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p['input']) for p in puzzle['train'] + puzzle['test']))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                features.append(_features(info))
                roles = {}
                normalized.add((pair['input'].shape, tuple(roles.setdefault(int(c), len(roles))
                                                         for c in pair['input'].flat)))
                assert np.array_equal(solve(pair['input']), pair['output'])
        # Both parents must affect at least one demonstration in every puzzle.
        validate_generated_puzzle(puzzle)
    assert len(signatures) == 42
    assert len(normalized) >= 160
    assert len({f['shape'] for f in features}) >= 25
    assert len({f['plate_shape'] for f in features}) >= 5
    assert len({f['masks'] for f in features}) >= 100
    assert {f['plates'] for f in features} == {2, 3, 4}
    assert {f['slots'] for f in features} == {1, 2, 3}
    assert len({f['live'] for f in features}) >= 4
    assert len({f['background'] for f in features}) == 10
    for name in ('attached', 'bare', 'painted', 'pooled', 'multiple', 'clipping'):
        assert sum(bool(f[name]) for f in features) >= 10
    original = generate_puzzle(91, 2, 1)
    regenerated = generate_puzzle(91, 2, 1)
    assert tuple(_key(p['input']) for p in original['train'] + original['test']) == tuple(
        _key(p['input']) for p in regenerated['train'] + regenerated['test'])
    # Exercise the final deterministic geometry directly, including each
    # optional input membership type and either taught clipping axis.
    fallback_rng = np.random.default_rng(4001)
    for allowed in ({'bottom'}, {'right'}, {'bottom', 'right', 'corner'}):
        for attach in (False, True):
            for paint in (False, True):
                pair = _fallback_pair(fallback_rng, 'test', True, attach, paint, allowed)
                assert np.array_equal(solve(pair['input']), pair['output'])
    backup = {'train': [_fallback_pair(fallback_rng, 'pool', True, False, False),
                        _fallback_pair(fallback_rng, 'full', False, True, True)],
              'test': [_fallback_pair(fallback_rng, 'test', False, True, True, {'right'})]}
    validate_generated_puzzle(backup)
    recolor = np.array([7, 2, 5, 0, 9, 3, 1, 8, 4, 6], dtype=np.int8)
    colored = {s: [{k: recolor[np.asarray(a)] for k, a in pair.items()} for pair in original[s]]
               for s in ('train', 'test')}
    validate_generated_puzzle(colored)
    bad = copy.deepcopy(original)
    query = bad['test'][0]
    info = _decode(query['input'])
    p, owner, slot = info['active'][0]
    h, w = info['plates'][owner]['arr'].shape
    query['output'][p[0]:p[0] + h, p[1]:p[1] + w] = info['bg']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('deleting an entire requested stencil was accepted')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(702 + nt * 10 + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (3, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(signatures),
                normalized_inputs=len(normalized), grid_shapes=len({f['shape'] for f in features}),
                plate_shapes=sorted({f['plate_shape'] for f in features}),
                stencil_dictionaries=len({f['masks'] for f in features}),
                live_counts=sorted({f['live'] for f in features}),
                clipping_pairs=sum(bool(f['clipping']) for f in features),
                attached_pairs=sum(bool(f['attached']) for f in features),
                semantic_verification='generator-owned checks only')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
