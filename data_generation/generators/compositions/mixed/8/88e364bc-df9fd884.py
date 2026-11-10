"""Key-directed portal couriers. Standard library and NumPy only.

solve reads visible keys, enclosed regions, U geometry, and marker components.
Generation constructs labels from sampled coordinates, separately from parsing.
No files, parent code, demonstrations, or random state are used by solve.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

CARDINAL = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 877466917, 88808394, 979515229, 59228269, 769229309,
              526471666, 524727930, 924213106, 875967903, 406972066,
              798431342, 365532402, 535313415, 984598397, 567607590,
              859972643, 877457287, 201664417, 525619830, 250727196,
              651431523, 984838962, 634809143, 685802592, 22171172,
              888787352, 204631702, 74225809, 393073197, 377132691,
              538355235, 88274385, 75510538, 1520303, 428603631,
              730266315, 820628911, 412971846, 5402182, 685617094,
              624986426)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must lie in {low}..{high}')
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), 'invalid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'noninteger grid')
    _require(np.all((0 <= a) & (a <= 9)), 'invalid color')
    return a.astype(np.int8, copy=True)


def _components(mask):
    h, w = mask.shape
    seen, result = ~mask.copy(), []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in CARDINAL:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _decode_keys(a, bg):
    keys = []
    for cells in _components(a != bg):
        r0, r1, c0, c1 = _bbox(cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if min(h, w) < 4 or len(cells) != h * w:
            continue
        sub = a[r0:r1 + 1, c0:c1 + 1]
        carrier = int(sub[0, 0])
        edge = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
        other = set(map(int, np.unique(sub))) - {carrier}
        if len(other) != 2 or np.any(edge != carrier):
            continue
        possible = []
        for tip in sorted(other):
            trail = next(iter(other - {tip}))
            ends = set(map(tuple, np.argwhere(sub == tip)))
            stems = set(map(tuple, np.argwhere(sub == trail)))
            if len(ends) < 2:
                continue
            for dr, dc in CARDINAL:
                used, lengths = set(), []
                for r, c in sorted(ends):
                    stroke, p = [], (r - dr, c - dc)
                    while p in stems:
                        stroke.append(p)
                        p = p[0] - dr, p[1] - dc
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(ends) and used == stems:
                    possible.append((tip, trail, (dr, dc), tuple(lengths)))
        _require(len(possible) == 1, 'ambiguous direction key')
        tip, trail, direction, lengths = possible[0]
        keys.append(dict(carrier=carrier, tip=tip, trail=trail, direction=direction,
                         bbox=(r0, r1, c0, c1), lengths=lengths))
    _require(keys and len({k['carrier'] for k in keys}) == len(keys), 'missing/conflicting keys')
    _require(len({(k['trail'], k['tip']) for k in keys}) == 1, 'inconsistent key roles')
    return keys


def _u_geometry(cells):
    """Exact rectangular U recognition; sign points from mouth to back."""
    cells = set(cells)
    r0, r1, c0, c1 = _bbox(cells)
    possible = []
    for axis in (0, 1):
        t0, t1 = (r0, r1) if axis == 0 else (c0, c1)
        x0, x1 = (c0, c1) if axis == 0 else (r0, r1)
        if t1 - t0 < 1 or x1 - x0 < 2:
            continue
        for sign in (-1, 1):
            back, mouth = (t1, t0) if sign == 1 else (t0, t1)
            expected = {(back, x) for x in range(x0, x1 + 1)}
            expected |= {(t, x) for t in range(t0, t1 + 1) for x in (x0, x1)}
            if axis == 1:
                expected = {(x, t) for t, x in expected}
            if cells == expected:
                possible.append(dict(axis=axis, sign=sign, back=back, mouth=mouth,
                                     aperture=(x0 + 1, x1 - 1), depth=t1 - t0, cells=cells))
    return possible[0] if len(possible) == 1 else None


def _read(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(np.count_nonzero(counts == counts[bg]) == 1, 'background not uniquely dominant')
    keys = _decode_keys(a, bg)
    kmask = np.zeros(a.shape, bool)
    for k in keys:
        r0, r1, c0, c1 = k['bbox']
        kmask[r0:r1 + 1, c0:c1 + 1] = True
    carriers = {k['carrier'] for k in keys}
    trail, tip = keys[0]['trail'], keys[0]['tip']
    _require(len({bg, trail, tip} | carriers) == 3 + len(carriers), 'role color collision')
    colors = set(map(int, np.unique(a[~kmask]))) - {bg} - carriers
    gates = {}
    for color in sorted(colors):
        parts = _components((a == color) & ~kmask)
        if len(parts) != 2:
            continue
        us = [_u_geometry(p) for p in parts]
        if all(u is not None for u in us):
            u, v = us
            if (u['axis'] == v['axis'] and u['sign'] == -v['sign']
                    and u['depth'] == v['depth']
                    and u['aperture'][1] - u['aperture'][0] == v['aperture'][1] - v['aperture'][0]):
                gates[color] = us
    _require(len(gates) == len(keys), 'need one opposed gate pair per system')
    movable = colors - set(gates)
    _require(len(movable) == 1, 'ambiguous marker color')
    marker = next(iter(movable))
    _require(marker not in {trail, tip} and not ({trail, tip} & set(gates)), 'key role collision')
    system, groups = {}, []
    # Enclosed free-space regions, rather than wall bounding boxes, bind markers.
    for region in _components((a == bg) | (a == marker)):
        marked = [p for p in region if a[p] == marker]
        if not marked:
            continue
        region_set = set(region)
        boundary, edge = set(), False
        for r, c in region:
            for dr, dc in CARDINAL:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < a.shape[0] and 0 <= nc < a.shape[1]):
                    edge = True
                elif (nr, nc) not in region_set:
                    boundary.add(int(a[nr, nc]))
        cs, gs = boundary & carriers, boundary & set(gates)
        _require(not edge and len(cs) == 1 and len(gs) <= 1
                 and boundary <= carriers | set(gates), 'marker not in unique enclosure')
        carrier = next(iter(cs))
        if gs:
            gate_color = next(iter(gs))
            pair = gates[gate_color]
            touching = [u for u in pair if any((r + dr, c + dc) in region_set
                        for r, c in u['cells'] for dr, dc in CARDINAL)]
            _require(len(touching) == 1, 'ambiguous source aperture')
            entry = touching[0]
            exit_gate = pair[1] if pair[0] is entry else pair[0]
            _require(carrier not in system or system[carrier]['entry'] is entry, 'multiple entry gates')
            system[carrier] = dict(entry=entry, exit=exit_gate, gate_color=gate_color)
        marked_mask = np.zeros(a.shape, bool)
        for p in marked:
            marked_mask[p] = True
        groups.extend(dict(cells=set(part), carrier=carrier) for part in _components(marked_mask))
    _require(set(system) == carriers and len({s['gate_color'] for s in system.values()}) == len(gates),
             'unassociated key or gateway')
    _require(sum(len(g['cells']) for g in groups) == int(np.count_nonzero(a == marker)), 'unassociated marker')
    fixed = (a != bg) & (a != marker) & ~kmask
    _require(set(map(int, np.unique(a[fixed]))) == carriers | set(gates), 'unexplained fixed cells')
    for s in system.values():
        u, v = s['entry'], s['exit']
        _require(v['back'] == (0 if u['sign'] == 1 else a.shape[u['axis']] - 1), 'exit not upstream')
        e0, e1 = u['aperture']
        x0, x1 = v['aperture']
        _require(e1 + 1 < x0 - 1 or x1 + 1 < e0 - 1, 'gateway spans overlap')
    return dict(background=bg, marker=marker, keys=keys,
                keymap={k['carrier']: k for k in keys}, systems=system, groups=groups)


def _predict(a, info, *, reflect=True, portals=True, reverse_axial=False,
             ignore_keys=False, local_reflection=False, swap_keys=False):
    """Execute geometry; the switches provide explicit counterfactual checks."""
    bg, marker = info['background'], info['marker']
    obstacles = (a != bg) & (a != marker)
    b = a.copy()
    b[a == marker] = bg
    records, claimed, swept_all = [], set(), set()
    for group in info['groups']:
        cells = group['cells']
        s = info['systems'][group['carrier']]
        u, v = s['entry'], s['exit']
        direction = info['keymap'][group['carrier']]['direction']
        if swap_keys and len(info['keys']) == 2:
            direction = next(k['direction'] for k in info['keys'] if k['carrier'] != group['carrier'])
        if ignore_keys:
            direction = (u['sign'], 0) if u['axis'] == 0 else (0, u['sign'])
        dr, dc = direction
        axis, sign = u['axis'], u['sign']
        along = direction[axis] == sign and direction[1 - axis] == 0
        transverse = [p[1 - axis] for p in cells]
        lo, hi = u['aperture']
        fits = min(transverse) >= lo and max(transverse) <= hi
        current, swept, transit = set(cells), set(cells), False
        for _ in range(31):
            shifted = {(r + dr, c + dc) for r, c in current}
            if any(not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) or obstacles[r, c]
                   for r, c in shifted):
                break
            lead = max(p[axis] for p in shifted) if sign == 1 else min(p[axis] for p in shifted)
            if portals and along and fits and lead == u['mouth']:
                transit = True
                ts = [p[axis] for p in cells]
                tmin, tmax = min(ts), max(ts)
                target = set()
                for p in cells:
                    t = tmin + tmax - p[axis] if reverse_axial else p[axis]
                    t += (a.shape[axis] - 1 - tmax) if sign == 1 else -tmin
                    offset = p[1 - axis] - lo
                    x = v['aperture'][1] - offset if reflect else v['aperture'][0] + offset
                    if local_reflection:
                        x = v['aperture'][0] + min(transverse) + max(transverse) - 2 * lo - offset
                    target.add((t, x) if axis == 0 else (x, t))
                # The complete downstream corridor must be clear.
                end = a.shape[axis] - 1 if sign == 1 else 0
                for x in {p[1 - axis] for p in target}:
                    for t in range(min(v['mouth'], end), max(v['mouth'], end) + 1):
                        p = (t, x) if axis == 0 else (x, t)
                        _require(not obstacles[p], 'blocked exit corridor')
                        swept.add(p)
                current = target
                break
            current = shifted
            swept.update(current)
        else:
            raise AssertionError('unbounded trajectory')
        _require(len(current) == len(cells) and claimed.isdisjoint(current), 'destination collision')
        _require(swept_all.isdisjoint(swept), 'interacting swept trajectories')
        _require(all(not obstacles[p] for p in current), 'destination overwrites structure')
        claimed.update(current)
        swept_all.update(swept)
        for p in current:
            b[p] = marker
        records.append(dict(carrier=group['carrier'], direction=direction, transit=transit,
                            starts=cells, targets=current, aperture=hi - lo + 1, axis=axis, fits=fits))
    return b, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer entirely from one input and the fixed family rule."""
    a = _grid(input_grid)
    return _predict(a, _read(a))[0]


def _key(rng, carrier, trail, tip, direction):
    lengths = [int(rng.integers(2, 4)), int(rng.integers(2, 4))]
    width, height = max(lengths) + 3, int(rng.integers(5, 7))
    glyph = np.full((height, width), carrier, np.int8)
    for row, length in zip((1, height - 2), lengths):
        start = int(rng.integers(1, width - length - 1))
        glyph[row, start:start + length] = trail
        glyph[row, start + length] = tip
    turns = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}[direction]
    return np.rot90(glyph, turns).copy()


def _motif(rng, width, length):
    for _ in range(40):
        mask = rng.random((width, length)) < float(rng.uniform(.5, .9))
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len(_components(mask)) != 1:
            continue
        if any(np.array_equal(mask, v) for v in (mask[::-1], mask[:, ::-1], mask[::-1, ::-1])):
            continue
        return mask
    mask = np.zeros((width, length), bool)
    mask[:, 0], mask[-1, :] = True, True
    return mask


def _draw_u(a, aperture_start, width, back, sign, depth, color):
    a[aperture_start - 1:aperture_start + width + 1, back] = color
    for t in range(back - sign * depth, back + sign, sign):
        a[aperture_start - 1, t] = color
        a[aperture_start + width, t] = color


def _make_scene(rng, marker, trail, tip, n, *, wide=False, partial=False, teaching=False):
    """Horizontal construction; rotations below provide all cardinal axes."""
    available = [x for x in range(10) if x not in (marker, trail, tip)]
    colors = list(map(int, rng.choice(available, 1 + 2 * n, replace=False)))
    bg, carriers, gate_colors = colors[0], colors[1:1 + n], colors[1 + n:]
    widths = list(map(int, rng.choice([2, 3, 4], n, replace=False)))
    if teaching:
        widths = [2, int(rng.choice([3, 4]))]
    source_heights = [w + 4 for w in widths]
    block_heights = [h + w + 3 for h, w in zip(source_heights, widths)]
    height = sum(block_heights) + n - 1
    extra = int(rng.integers(0, 31 - height))
    top = int(rng.integers(0, extra + 1))
    height += extra
    width = int(rng.integers(25, 31))
    a = np.full((height, width), bg, np.int8)
    systems, slots, row = [], [], top
    teaching_sign = int(rng.choice([-1, 1]))
    for i in range(n):
        aperture, h = widths[i], source_heights[i]
        exit_below = bool(rng.integers(2))
        sr = row if exit_below else row + aperture + 3
        er = row + h + 1 if exit_below else row
        aperture_high = bool(rng.integers(2))
        entry_start = sr + (2 if aperture_high else 1)
        offrow = sr + 1 if aperture_high else sr + h - 2
        exit_start = er + 1
        ew = int(rng.integers(14, 16))
        c1 = width - 1 - int(rng.integers(0, 3))
        c0 = c1 - ew + 1
        footprint = np.ones((h, ew), bool)
        for corner in (0, 1):
            if rng.random() < (.85 if wide else .6):
                dh, dw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
                if corner == 0:
                    footprint[:dh, :dw] = False
                else:
                    footprint[h - dh:, :dw] = False
        pad = np.pad(footprint, 1)
        inner = footprint.copy()
        for dr, dc in CARDINAL:
            inner &= pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + ew]
        a[sr:sr + h, c0:c1 + 1][footprint & ~inner] = carriers[i]
        depth = int(rng.integers(1, 3))
        _draw_u(a, entry_start, aperture, c1, 1, depth, gate_colors[i])
        _draw_u(a, exit_start, aperture, 0, -1, depth, gate_colors[i])
        length, motif_width, start_row = int(rng.integers(2, 4)), aperture, entry_start
        if partial and aperture >= 3:
            motif_width = int(rng.integers(1, aperture))
            offsets = [x for x in range(aperture - motif_width + 1) if 2 * x + motif_width != aperture]
            start_row += int(rng.choice(offsets))
        motif = np.ones((1, length), bool) if motif_width == 1 else _motif(rng, motif_width, length)
        start_col = c1 - depth - length - int(rng.integers(1, 3))
        group1 = {(start_row + int(r), start_col + int(c)) for r, c in np.argwhere(motif)}
        group2 = {(offrow, c0 + 3)}
        if rng.random() < .4:
            group2.add((offrow, c0 + 4))
        _require(max(c for _, c in group2) < min(c for _, c in group1) - 1, 'construction spacing')
        for p in group1 | group2:
            _require(a[p] == bg, 'marker hits wall')
            a[p] = marker
        sign = teaching_sign if teaching else (-1 if rng.integers(2) else 1)
        if sign == -1:
            a[row:row + block_heights[i]] = a[row:row + block_heights[i], ::-1]
            group1 = {(r, width - 1 - c) for r, c in group1}
            group2 = {(r, width - 1 - c) for r, c in group2}
            slot = (sr, width - c0 + 1, h, c0 - 2)
        else:
            slot = (sr, 1, h, c0 - 2)
        slots.append(slot)
        systems.append(dict(carrier=carriers[i], sign=sign, entry=entry_start,
                            exit=exit_start, width=aperture, groups=[group1, group2]))
        row += block_heights[i] + 1
    return dict(grid=a, background=bg, marker=marker, trail=trail, tip=tip, systems=systems, slots=slots)


def _label_from_construction(a, scene, directions):
    """Labels from source coordinates, separate from the input parser."""
    bg, marker = scene['background'], scene['marker']
    b = a.copy()
    b[a == marker] = bg
    wall, destinations = (a != bg) & (a != marker), set()
    for system, direction in zip(scene['systems'], directions):
        for index, starts in enumerate(system['groups']):
            if index == 0 and direction == (0, system['sign']):
                cmin, cmax = min(c for _, c in starts), max(c for _, c in starts)
                delta = a.shape[1] - 1 - cmax if system['sign'] == 1 else -cmin
                ends = {(system['exit'] + system['width'] - 1 - (r - system['entry']), c + delta)
                        for r, c in starts}
            else:
                ends, (dr, dc) = set(starts), direction
                for _ in range(30):
                    nxt = {(r + dr, c + dc) for r, c in ends}
                    if any(not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) or wall[r, c] for r, c in nxt):
                        break
                    ends = nxt
            _require(destinations.isdisjoint(ends), 'construction target collision')
            destinations.update(ends)
    for p in destinations:
        _require(b[p] == bg, 'target hits wall')
        b[p] = marker
    return b


def _finish_scene(rng, scene, directions, turns, mirror=False, key_order=None):
    a = scene['grid'].copy()
    order = rng.permutation(len(scene['slots'])) if key_order is None else key_order
    slots = [scene['slots'][int(i)] for i in order]
    for system, direction, slot in zip(scene['systems'], directions, slots):
        glyph = _key(rng, system['carrier'], scene['trail'], scene['tip'], direction)
        r, c, h, w = slot
        _require(glyph.shape[0] <= h and glyph.shape[1] <= w, 'key too large')
        r += int(rng.integers(0, h - glyph.shape[0] + 1))
        c += int(rng.integers(0, w - glyph.shape[1] + 1))
        _require(np.all(a[r:r + glyph.shape[0], c:c + glyph.shape[1]] == scene['background']), 'key overlaps')
        a[r:r + glyph.shape[0], c:c + glyph.shape[1]] = glyph
    b = _label_from_construction(a, scene, directions)
    if mirror:
        a, b = a[::-1].copy(), b[::-1].copy()
    return dict(input=np.rot90(a, turns).copy(), output=np.rot90(b, turns).copy())


def _fingerprint(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt, nq = _integer(num_train, 'num_train', 2, 10), _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    marker, trail, tip = map(int, rng.choice(10, 3, replace=False))
    puzzle, seen, base = {'train': [], 'test': []}, set(), None
    base_turn = int(rng.integers(4))
    teaching_perpendicular = int(rng.choice([-1, 1]))
    for index in range(nt + nq):
        test = index >= nt
        for _ in range(100):
            try:
                if index < 2:
                    if base is None:
                        base = _make_scene(rng, marker, trail, tip, 2, partial=True, teaching=True)
                    scene = base
                    active = (index == 0, index == 1)
                    dirs = [(0, s['sign'] if active[i] else -s['sign']) for i, s in enumerate(scene['systems'])]
                    if index == 0:
                        dirs[1] = (teaching_perpendicular, 0)
                    turns, mirror = base_turn, False
                else:
                    n = 2 if test else int(rng.choice([1, 2]))
                    scene = _make_scene(rng, marker, trail, tip, n, wide=test, partial=test and index % 2 == 0)
                    dirs = [(0, s['sign']) for s in scene['systems']]
                    if n == 2 and (test or rng.random() < .5):
                        j = int(rng.integers(n))
                        dirs[j] = (int(rng.choice([-1, 1])), 0) if test else (0, -scene['systems'][j]['sign'])
                    turns = (base_turn + (1 if test else int(rng.integers(4)))) % 4
                    mirror = bool(rng.integers(2))
                order = ([1, 0] if index == 0 else [0, 1]) if index < 2 else None
                pair = _finish_scene(rng, scene, dirs, turns, mirror, key_order=order)
                info = _read(pair['input'])
                expected, records = _predict(pair['input'], info)
                _require(np.array_equal(expected, pair['output']), 'construction/parser disagreement')
                _require(any(r['transit'] for r in records), 'scene needs a transit')
                fp = _fingerprint(pair['input'])
                if fp in seen:
                    continue
                seen.add(fp)
                break
            except AssertionError:
                if index == 0:
                    base = None
                continue
        else:
            raise RuntimeError(f'bounded generation failed at pair {index}')
        puzzle['test' if test else 'train'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'invalid pair counts')
    seen, roles, train_directions, features = set(), set(), set(), Counter()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'dimensions changed')
            fp = _fingerprint(a)
            _require(fp not in seen, 'duplicate input')
            seen.add(fp)
            info = _read(a)
            expected, records = _predict(a, info)
            _require(np.array_equal(expected, b), 'wrong answer')
            _require(not np.array_equal(a, b), 'no movement demonstrated')
            roles.add((info['marker'], info['keys'][0]['trail'], info['keys'][0]['tip']))
            marker, bg = info['marker'], info['background']
            _require(np.count_nonzero(a == marker) == np.count_nonzero(b == marker), 'lost marker cells')
            fixed = (a != bg) & (a != marker)
            _require(np.array_equal(a[fixed], b[fixed]), 'keys/walls/gates changed')
            for rec in records:
                if rec['transit']:
                    axis = rec['axis']
                    positive = rec['direction'][axis] == 1
                    ts = [p[axis] for p in rec['targets']]
                    _require((max(ts) if positive else min(ts)) == (a.shape[axis] - 1 if positive else 0), 'not flush')
                    ss = sorted(p[axis] for p in rec['starts'])
                    tt = sorted(ts)
                    _require([x - ss[0] for x in ss] == [x - tt[0] for x in tt], 'axial offsets changed')
            if split == 'train':
                train_directions.update(k['direction'] for k in info['keys'])
                features['portal'] += any(r['transit'] for r in records)
                features['wall_stop'] += any(not r['transit'] for r in records)
                features['wrong_direction'] += any(k['direction'][info['systems'][k['carrier']]['entry']['axis']]
                    != info['systems'][k['carrier']]['entry']['sign'] for k in info['keys'])
                for name, args in (('reflection', {'reflect': False}), ('key', {'ignore_keys': True}),
                                   ('axial', {'reverse_axial': True}), ('portal_needed', {'portals': False}),
                                   ('aperture_frame', {'local_reflection': True}), ('binding', {'swap_keys': True})):
                    alternative, _ = _predict(a, info, **args)
                    features[name] += not np.array_equal(expected, alternative)
    _require(len(roles) == 1, 'puzzle role conventions changed')
    _require(len(train_directions) >= 2, 'training lacks key polarity evidence')
    _require(all(features[x] for x in ('portal', 'wall_stop', 'wrong_direction', 'reflection',
                                     'key', 'axial', 'portal_needed', 'aperture_frame', 'binding')),
             'missing discriminating demonstrations')


def self_test():
    """42 sequential seeds; no files, execution traces, or retained puzzle bank."""
    puzzles, shapes, palettes, motifs, widths, directions, keysizes, systemcounts = (set() for _ in range(8))
    totals = Counter()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        sig = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                sig.update(bytes(a.shape))
                sig.update(a.tobytes())
                sig.update(pair['output'].tobytes())
                shapes.add(a.shape)
                info = _read(a)
                palettes.add((info['background'], info['marker'], tuple(k['carrier'] for k in info['keys'])))
                systemcounts.add(len(info['keys']))
                predicted, records = _predict(a, info)
                _require(np.array_equal(predicted, pair['output']), 'solver mismatch')
                for k in info['keys']:
                    r0, r1, c0, c1 = k['bbox']
                    keysizes.add((r1 - r0 + 1, c1 - c0 + 1))
                    directions.add(k['direction'])
                for rec in records:
                    widths.add(rec['aperture'])
                    r0, _, c0, _ = _bbox(rec['starts'])
                    motifs.add(tuple(sorted((r - r0, c - c0) for r, c in rec['starts'])))
                    totals['transits' if rec['transit'] else 'stops'] += 1
                totals['pairs'] += 1
        puzzles.add(sig.digest())
        a, b = p['test'][0]['input'], p['test'][0]['output']
        for k in range(4):
            _require(np.array_equal(solve(np.rot90(a, k)), np.rot90(b, k)), 'rotation failure')
        _require(np.array_equal(solve(a[:, ::-1]), b[:, ::-1]), 'reflection failure')
    _require(len(puzzles) == 42 and len(shapes) >= 20 and len(palettes) >= 35, 'scene diversity')
    _require(len(motifs) >= 30 and widths == {2, 3, 4} and directions == set(CARDINAL), 'geometry coverage')
    _require(len(keysizes) >= 4 and systemcounts == {1, 2}, 'key/count coverage')
    p, q = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x), 'nondeterminism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            q = generate_puzzle(1234, nt, nq)
            _require(len(q['train']) == nt and len(q['test']) == nq, 'explicit counts ignored')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p['test'][0]['output'] = p['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('identity corruption accepted')
    return dict(seeds=42, distinct_puzzles=len(puzzles), grid_shapes=len(shapes),
                palettes=len(palettes), normalized_marker_shapes=len(motifs),
                aperture_widths=sorted(widths), key_shapes=len(keysizes),
                directions=sorted(directions), system_counts=sorted(systemcounts), **dict(totals))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
