"""Diagonal harvest: continuous diagonal containment gates unary color transfers.

Only the standard library and NumPy are used. No parent code is imported.
"""
from __future__ import annotations

import copy
import itertools
import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        todo, cells = [p], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(sorted(cells))
    return result


def _strokes(cells):
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in cells:
            lines.setdefault(c-slope*r, []).append(r)
        runs = []
        valid = True
        for q, rows in sorted(lines.items()):
            rows.sort()
            pieces = []
            for r in rows:
                if not pieces or r != pieces[-1][-1]+1:
                    pieces.append([])
                pieces[-1].append(r)
            for piece in pieces:
                if len(piece) < 2:
                    valid = False
                runs.append((q, 2*piece[0]+slope*q, 2*piece[-1]+slope*q))
        if valid and len(runs) >= 2:
            candidates.append((slope, runs))
    if len(candidates) != 1:
        raise ValueError("boundary strokes need a unique common diagonal slope")
    return candidates[0]


def _envelope(shape, slope, strokes, alternative=""):
    rr, cc = np.indices(shape)
    q, t = cc-slope*rr, rr+slope*cc
    mask = np.zeros(shape, bool)
    if alternative in ("adjacent", "pixel_centers", "pair_union"):
        pairs = [(a, b) for i, a in enumerate(strokes) for b in strokes[i+1:]]
        if alternative == "adjacent":
            ordered = sorted(strokes)
            pairs = list(zip(ordered, ordered[1:]))
        for a, b in pairs:
            part = ((t >= max(a[1], b[1])) & (t <= min(a[2], b[2]))
                    & (q > min(a[0], b[0])) & (q < max(a[0], b[0])))
            if alternative == "pixel_centers":
                part &= ((t-a[0]) % 2 == 0) & ((t-b[0]) % 2 == 0)
            mask |= part
        return mask
    for level in range(int(t.min()), int(t.max())+1):
        active = [a[0] for a in strokes if a[1] <= level <= a[2]]
        if len(active) >= 2:
            mask |= (t == level) & (q > min(active)) & (q < max(active))
    return mask


def _parse(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if counts.max() <= g.size/2:
        raise ValueError("background must occupy a strict majority")
    singleton, tokens, bars = [], [], []
    for cells in _components(g != bg):
        values = {int(g[r, c]) for r, c in cells}
        if len(values) != 1:
            raise ValueError("objects must be monochrome and separated")
        color = values.pop()
        ys, xs = zip(*cells)
        y, x = min(ys), min(xs)
        h, w = max(ys)-y+1, max(xs)-x+1
        mask = np.zeros((h, w), bool)
        for r, c in cells:
            mask[r-y, c-x] = True
        obj = dict(y=y, x=x, h=h, w=w, color=color, mask=mask, cells=cells)
        if len(cells) == 1:
            singleton.append((cells[0], color))
        elif h == 1 and 2 <= w <= 4 and mask.all():
            bars.append(obj)
        elif 2 <= h <= 4 and 2 <= w <= 4:
            tokens.append(obj)
        else:
            raise ValueError("unsupported object geometry")
    if not bars or not singleton:
        raise ValueError("missing controls or boundaries")
    boundary_colors = {c for _, c in singleton}
    process_colors = {b['color'] for b in bars}
    if len(boundary_colors) != 1 or len(process_colors) != 1:
        raise ValueError("one boundary color and one control color per input")
    boundary = boundary_colors.pop()
    processed = process_colors.pop()
    slope, strokes = _strokes([p for p, _ in singleton])
    receivers, assigned = [], set()
    for bar in bars:
        lane = [i for i, a in enumerate(tokens)
                if a['x'] == bar['x'] and a['w'] == bar['w']]
        if len(lane) < 2:
            raise ValueError("each receiver must visibly reveal its pitch")
        lane.sort(key=lambda i: tokens[i]['y'])
        objs = [tokens[i] for i in lane]
        first = objs[0]
        if (any(a['color'] != first['color'] or
                not np.array_equal(a['mask'], first['mask']) for a in objs)
                or not 1 <= bar['y']-(objs[-1]['y']+first['h']) <= 3):
            raise ValueError("a control must lie under one regular receiver")
        pitches = {b['y']-a['y'] for a, b in zip(objs, objs[1:])}
        if len(pitches) != 1 or next(iter(pitches)) <= first['h']:
            raise ValueError("receiver tokens need one observable positive-gap pitch")
        for i, other in enumerate(tokens):
            if i not in lane and other['x'] < bar['x']+bar['w'] and other['x']+other['w'] > bar['x']:
                raise ValueError("receiver column bands cannot contain source objects")
        if assigned.intersection(lane):
            raise ValueError("ambiguous receiver controls")
        assigned.update(lane)
        receivers.append(dict(bar=bar, tokens=objs, color=first['color'],
                              mask=first['mask'], x=bar['x'], pitch=pitches.pop()))
    payloads = [a['color'] for a in receivers]
    if len(set(payloads)) != len(payloads) or set(payloads) & {bg, boundary, processed}:
        raise ValueError("receiver payloads must be distinct from each other and control roles")
    sources = [a for i, a in enumerate(tokens) if i not in assigned]
    if not sources or any(a['color'] not in payloads for a in sources):
        raise ValueError("each source must match exactly one receiver color")
    return dict(grid=g, background=bg, boundary=boundary, processed=processed,
                slope=slope, strokes=strokes, sources=sources, receivers=receivers,
                boundary_cells=[p for p, _ in singleton])


def _selected(obj, mask, mode="whole"):
    values = [bool(mask[r, c]) for r, c in obj['cells']]
    if mode == "intersection":
        return any(values)
    if mode == "center":
        # Explicit rival: sample the bottom/right middle pixel of the bbox.
        return bool(mask[obj['y']+obj['h']//2, obj['x']+obj['w']//2])
    return all(values)


def _answer(parsed, envelope, mode="whole"):
    g = parsed['grid']
    out = g.copy()
    amounts = {a['color']: 0 for a in parsed['receivers']}
    for obj in parsed['sources']:
        if mode == "all_sources" or _selected(obj, envelope, mode):
            amounts[obj['color']] += 1
            for r, c in obj['cells']:
                out[r, c] = parsed['processed']
    for rec in parsed['receivers']:
        for r, c in rec['bar']['cells']:
            out[r, c] = parsed['background']
        top = rec['tokens'][0]['y']
        for k in range(1, amounts[rec['color']]+1):
            y = top-k*rec['pitch']
            if y < 0:
                raise ValueError("complete receiver extension must fit")
            h, w = rec['mask'].shape
            block = out[y:y+h, rec['x']:rec['x']+w]
            if block.shape != (h, w) or np.any(block[rec['mask']] != parsed['background']):
                raise ValueError("receiver extension must not collide")
            block[rec['mask']] = rec['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from pixels alone, including local control color and receiver pitch."""
    p = _parse(input_grid)
    return _answer(p, _envelope(p['grid'].shape, p['slope'], p['strokes']))


def _shape(rng, receiver=False, width=None):
    if receiver:
        w = int(rng.integers(2, 4)) if width is None else width
        m = np.ones((2, w), bool)
        if rng.random() < .45:
            m[0, 1:] = False
        if rng.random() < .5:
            m = m[:, ::-1]
        return m.copy()
    kind = int(rng.choice(4, p=[.40, .35, .15, .10]))
    if kind == 0:
        return np.ones((2, 2), bool)
    if kind == 1:
        m = np.ones((2, 2), bool)
        m[int(rng.integers(2)), int(rng.integers(2))] = False
        return m
    if kind == 2:
        m = np.ones((2, 3), bool)
        m[0, int(rng.integers(3))] = False
        return m
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def _draw(g, y, x, color, mask):
    h, w = mask.shape
    g[y:y+h, x:x+w][mask] = color


def _dilate(mask, diagonal):
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not diagonal and abs(dr)+abs(dc) > 1:
                continue
            r0, r1 = max(0, dr), min(h, h+dr)
            c0, c1 = max(0, dc), min(w, w+dc)
            out[r0:r1, c0:c1] |= mask[r0-dr:r1-dr, c0-dc:c1-dc]
    return out


def _gate(rng, max_width, witness, staggered=False):
    slope = int(rng.choice([-1, 1]))
    span = int(rng.choice([12, 14, 16])) if staggered else int(rng.integers(12, 21))
    count = int(rng.integers(3 if witness or staggered else 2, min(5, span//3+1)+1))
    possible = [xs for xs in itertools.combinations(range(3, span-2), count-2)
                if all(b-a >= 3 for a, b in zip((0,)+xs, xs+(span,)))
                and (not staggered or span//2 in xs)]
    if not possible:
        raise ValueError("no separated boundary arrangement")
    qs = [0] + list(possible[int(rng.integers(len(possible)))]) + [span]
    specs = []
    for i, q in enumerate(qs):
        if staggered:
            # A long middle interval bridges two disjoint outer intervals.
            # Both lobes have enough area for whole glyphs, without requiring
            # a longitudinal position at which every stroke is active.
            if i == 0:
                lo, hi = -8, 0
            elif i == count-1:
                lo, hi = 2, 12
            elif q == span//2:
                lo, hi = -8, 12
            elif q < span//2:
                lo, hi = -4, 0
            else:
                lo, hi = 4, 8
        elif i in (0, count-1):
            lo = -int(rng.choice([4, 6, 8]))
            hi = int(rng.choice([6, 8, 10]))
        else:
            lo = int(rng.integers(-2, 3))
            hi = max(4+q % 2, lo+2*int(rng.integers(1, 4)))
        lo += (q-lo) % 2
        hi -= (hi-q) % 2
        if hi-lo < 2:
            hi = lo+2
        specs.append((q, lo, hi))
    if witness:
        i = int(rng.integers(1, count-1))
        q = specs[i][0]
        lo = 2+(q % 2)
        specs[i] = (q, lo, lo+2)
    cells = []
    for q, lo, hi in specs:
        for t in range(lo, hi+1, 2):
            cells.append(((t-slope*q)//2, (q+slope*t)//2))
    ys, xs = zip(*cells)
    h, w = max(ys)-min(ys)+5, max(xs)-min(xs)+5
    if w > max_width or h > 27:
        raise ValueError("gate does not fit")
    shifted = [(r-min(ys)+2, c-min(xs)+2) for r, c in cells]
    actual_slope, actual_specs = _strokes(shifted)
    if actual_slope != slope:
        raise ValueError("ambiguous gate")
    return shifted, h, w, slope, actual_specs


def _placement_pool(gate_block, region, adjacent, mask, category, skip, hole_witness=False):
    h, w = mask.shape
    offsets = np.argwhere(mask)
    pool = []
    for y in range(gate_block.shape[0]-h+1):
        for x in range(gate_block.shape[1]-w+1):
            rr, cc = offsets[:, 0]+y, offsets[:, 1]+x
            if gate_block[rr, cc].any():
                continue
            inside = region[rr, cc]
            kind = 'full' if inside.all() else ('partial' if inside.any() else 'outside')
            if kind != category or (skip and adjacent[rr, cc].all()):
                continue
            if hole_witness and region[y:y+h, x:x+w].all():
                continue
            pool.append((y, x))
    return pool


def _build_pair(rng, nrecv, anchor, force_staggered=False):
    palette = [int(c) for c in rng.permutation(10)]
    bg, boundary, processed = palette[:3]
    payloads = palette[3:3+nrecv]
    receiver_masks = [_shape(rng, True, 2 if nrecv == 3 else None) for _ in payloads]
    widths = [m.shape[1] for m in receiver_masks]
    pitch = [int(rng.integers(3, 5)) for _ in payloads]
    initial = [int(rng.integers(2, 4)) for _ in payloads]
    if anchor:
        captures = [1, int(rng.integers(2, 4))]
        rng.shuffle(captures)
        initial = [2, 3]
        rng.shuffle(initial)
    elif nrecv == 3:
        captures = [1, 1, 2]
        rng.shuffle(captures)
    else:
        captures = [int(rng.integers(2 if nrecv == 1 else 1, 5 if nrecv == 1 else 4))
                    for _ in payloads]
    bottom_margins = [int(rng.integers(0, 3)) for _ in payloads]
    control_gaps = [int(rng.integers(1, 4)) for _ in payloads]
    # Receivers can flank both sides of the catchment; they are not ranked by color.
    order = [int(i) for i in rng.permutation(nrecv)]
    nleft = int(rng.integers(nrecv+1))
    left_ids, right_ids = order[:nleft], order[nleft:]
    def band_width(ids):
        return sum(widths[i] for i in ids)+max(0, len(ids)-1)
    lw, rw = band_width(left_ids), band_width(right_ids)
    separation = (2 if left_ids else 0)+(2 if right_ids else 0)
    cells, gh, gw, slope, specs = _gate(rng, 30-lw-rw-separation, anchor, force_staggered)
    required_height = max(gh, max(1+bottom_margins[i]+control_gaps[i]+m.shape[0]
                                 +(initial[i]+captures[i]-1)*pitch[i]
                                 for i, m in enumerate(receiver_masks)))
    if required_height > 30:
        raise ValueError("not enough arithmetic headroom")
    upper_height = min(30, required_height+3) if rng.random() < .25 else 30
    height = int(rng.integers(required_height, upper_height+1))
    minimum_width = gw+lw+rw+separation
    width = int(rng.integers(minimum_width, min(30, minimum_width+3)+1))
    extra_left = int(rng.integers(width-minimum_width+1))
    zone_x = extra_left+lw+(2 if left_ids else 0)
    zone_y = int(rng.integers(height-gh+1))
    g = np.full((height, width), bg, np.int8)
    for r, c in cells:
        g[r+zone_y, c+zone_x] = boundary
    zone_gate = np.zeros((height, gw), bool)
    for r, c in cells:
        zone_gate[r+zone_y, c] = True
    local_cells = [(r+zone_y, c) for r, c in cells]
    _, local_specs = _strokes(local_cells)
    region = _envelope(zone_gate.shape, slope, local_specs)
    adjacent = _envelope(zone_gate.shape, slope, local_specs, 'adjacent')
    blocked = _dilate(zone_gate, False)
    receiver_records = []
    x_by_id = {}
    x = extra_left
    for i in left_ids:
        x_by_id[i] = x
        x += widths[i]+1
    x = zone_x+gw+(2 if right_ids else 0)
    for i in right_ids:
        x_by_id[i] = x
        x += widths[i]+1
    for i, m in enumerate(receiver_masks):
        bottom_margin = bottom_margins[i]
        control_gap = control_gaps[i]
        control_y = height-1-bottom_margin
        bottom_y = control_y-control_gap-m.shape[0]
        if bottom_y-(initial[i]+captures[i]-1)*pitch[i] < 0:
            raise ValueError("not enough arithmetic headroom")
        x = x_by_id[i]
        for k in range(initial[i]):
            _draw(g, bottom_y-k*pitch[i], x, payloads[i], m)
        g[control_y, x:x+m.shape[1]] = processed
        receiver_records.append((i, x, bottom_y, control_y, m))
    # Place a bypass witness first, then a partial glyph, then other contributors.
    jobs = []
    witness_channel = int(rng.integers(nrecv))
    remaining = captures.copy()
    if anchor:
        jobs.append((witness_channel, 'full', True))
        remaining[witness_channel] -= 1
    jobs.append((int(rng.integers(nrecv)), 'partial', False))
    for i, amount in enumerate(remaining):
        jobs += [(i, 'full', False)]*amount
    for i in range(nrecv):
        jobs.append((i, 'outside', False))
    jobs += [(int(rng.integers(nrecv)), 'outside', False)]*int(rng.integers(0, 3))
    source_records = []
    occupied = np.zeros_like(zone_gate)
    unit_witness_placed = False
    for i, category, skip in jobs:
        placed = False
        force_unit = anchor and category == 'full' and not skip and not unit_witness_placed
        for attempt in range(8):
            if force_unit:
                m = np.ones((2, 2), bool)
                corner = int(rng.integers(4))
                m[corner//2, corner % 2] = False
                if np.array_equal(m, receiver_masks[i]):
                    m = np.ones((2, 2), bool)
                    corner = (corner+1) % 4
                    m[corner//2, corner % 2] = False
            else:
                m = np.ones((2, 2), bool) if skip or attempt >= 5 else _shape(rng)
            pool = _placement_pool(blocked, region, adjacent, m, category, skip, force_unit)
            available = []
            offsets = np.argwhere(m)
            token_block = _dilate(occupied, True)
            for y, x in pool:
                rr, cc = offsets[:, 0]+y, offsets[:, 1]+x
                if not token_block[rr, cc].any():
                    available.append((y, x))
            if not available:
                continue
            y, x = available[int(rng.integers(len(available)))]
            _draw(g, y, x+zone_x, payloads[i], m)
            rr, cc = offsets[:, 0]+y, offsets[:, 1]+x
            occupied[rr, cc] = True
            source_records.append((i, y, x+zone_x, m, category))
            unit_witness_placed |= force_unit
            placed = True
            break
        if not placed:
            raise ValueError("source packing failed")
    # Labels use the construction's explicit object classes/counts. solve parses pixels.
    out = g.copy()
    for i, y, x, m, category in source_records:
        if category == 'full':
            _draw(out, y, x, processed, m)
    for i, x, bottom_y, control_y, m in receiver_records:
        out[control_y, x:x+m.shape[1]] = bg
        for k in range(initial[i], initial[i]+captures[i]):
            _draw(out, bottom_y-k*pitch[i], x, payloads[i], m)
    pair = dict(input=g, output=out)
    p = _parse(g)
    assert np.array_equal(solve(g), out)
    f = _features(p)
    if anchor and not (f['skip'] and f['bbox'] and f['center'] and f['partial']
                       and f['shape_copy'] and f['occupied_vs_box'] and len(f['eligible_areas']) >= 2):
        raise ValueError("anchor lacks a required geometric distinction")
    return pair


def _key(g):
    g = np.asarray(g)
    return (g.shape, g.astype(np.int8).tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = dict(train=[], test=[]), set()
    anchor_index = int(rng.integers(2))
    expansion = bool(rng.integers(2))
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            anchor = split == 'train' and index == anchor_index
            nrecv = 2 if anchor else int(rng.choice([1, 2]))
            if split == 'test' and index == 0:
                nrecv = 3 if expansion else 1
            staggered = (not anchor and nrecv == 1 and
                         (split == 'test' and index == 0 and not expansion or
                          (expansion or split == 'test') and rng.random() < .20))
            for _ in range(160):
                try:
                    pair = _build_pair(rng, nrecv, anchor, staggered)
                except ValueError:
                    continue
                key = _key(pair['input'])
                if key in seen:
                    continue
                if split == 'train' and index == 1:
                    if _parse(pair['input'])['processed'] == _parse(puzzle['train'][0]['input'])['processed']:
                        continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(p):
    shape = p['grid'].shape
    env = _envelope(shape, p['slope'], p['strokes'])
    adjacent = _envelope(shape, p['slope'], p['strokes'], 'adjacent')
    centers = _envelope(shape, p['slope'], p['strokes'], 'pixel_centers')
    amounts = {a['color']: 0 for a in p['receivers']}
    partial = skipped = bbox_witness = center_witness = False
    copy_witness, occupancy_witness, areas = False, False, set()
    receiver_masks = {a['color']: a['mask'] for a in p['receivers']}
    rows, cols = zip(*p['boundary_cells'])
    bbox = np.zeros(shape, bool)
    bbox[min(rows):max(rows)+1, min(cols):max(cols)+1] = True
    for a in p['sources']:
        selected = _selected(a, env)
        occupancy_witness |= selected != bool(env[a['y']:a['y']+a['h'], a['x']:a['x']+a['w']].all())
        amounts[a['color']] += int(selected)
        if selected:
            areas.add(len(a['cells']))
            copy_witness |= not np.array_equal(a['mask'], receiver_masks[a['color']])
        partial |= _selected(a, env, 'intersection') and not selected
        skipped |= selected and not _selected(a, adjacent)
        bbox_witness |= _selected(a, bbox) != selected
        center_witness |= _selected(a, env, 'center') != selected
    receivers = [(len(a['tokens']), amounts[a['color']]) for a in p['receivers']]
    return dict(shape=shape, slope=p['slope'], strokes=len(p['strokes']), processed=p['processed'],
                receivers=receivers, amounts=amounts, partial=bool(partial),
                skip=bool(skipped), bbox=bool(bbox_witness), center=bool(center_witness),
                shape_copy=bool(copy_witness), eligible_areas=sorted(areas),
                occupied_vs_box=bool(occupancy_witness),
                subpixel=any(_selected(a, env) != _selected(a, centers) for a in p['sources']),
                outside=sum(not _selected(a, env) for a in p['sources']),
                source_shapes={tuple(a['mask'].shape)+(a['mask'].tobytes(),) for a in p['sources']},
                pitches={a['pitch'] for a in p['receivers']},
                no_common=max(a[1] for a in p['strokes']) > min(a[2] for a in p['strokes']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            p = _parse(g)
            # Pairwise parallelogram union checks the sweep envelope separately.
            geom = _envelope(g.shape, p['slope'], p['strokes'], 'pair_union')
            assert np.array_equal(out, _answer(p, geom)), 'wrong containment/count/copy/erase answer'
            assert np.array_equal(out, solve(g)), 'input-only solver disagreement'
            f = _features(p)
            assert f['partial'] and f['outside'] > 0
            assert all(s >= 1 for d, s in f['receivers'])
            # A receiver is outside the aperture; no source can be double-counted.
            assert all(not geom[r, c] for a in p['receivers'] for b in a['tokens'] for r, c in b['cells'])
            for r, c in p['boundary_cells']:
                assert out[r, c] == g[r, c]
            assert np.sum(out == p['processed']) == sum(len(a['cells']) for a in p['sources'] if _selected(a, geom))
            for a in p['sources']:
                expected_color = p['processed'] if _selected(a, geom) else a['color']
                assert all(out[r, c] == expected_color for r, c in a['cells'])
            analyses[split].append(f)
    train = analyses['train']
    assert len({f['processed'] for f in train}) >= 2, 'missing processing-color demonstration'
    for feature in ('skip', 'bbox', 'center', 'subpixel', 'partial', 'occupied_vs_box'):
        assert any(f[feature] for f in train), f'missing {feature} demonstration'
    assert any(len(f['receivers']) >= 2 and len(set(f['amounts'].values())) >= 2 for f in train), 'missing unequal color-routing demonstration'
    assert any(f['shape_copy'] for f in train), 'missing receiver-shape copying demonstration'
    assert len({a for f in train for a in f['eligible_areas']}) >= 2, 'missing unequal-area unit counting demonstration'
    operands = [a for f in train for a in f['receivers']]
    assert len({d for d, s in operands}) >= 2
    assert len({s for d, s in operands}) >= 2
    for rival in (lambda d, s: 2*d, lambda d, s: d+1, lambda d, s: s,
                  lambda d, s: max(d, s), lambda d, s: d*s):
        assert any(rival(d, s) != d+s for d, s in operands), 'ambiguous arithmetic demonstrations'
    assert any(s1*d2 != s2*d1 for d1, s1 in operands for d2, s2 in operands), 'fixed destination multiplier remains possible'
    assert (any(len(f['receivers']) == 3 for f in analyses['test'])
            or (all(not f['no_common'] for f in train)
                and any(f['no_common'] for f in analyses['test']))), 'missing structural train/test shift'


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for split in ('train', 'test') for p in puzzle[split])


def _hand_fixture():
    # q=-6 and q=6; both continuous t intervals are [6,14]. Two 2x2
    # sources are wholly inside; a third has only its lower-right cell inside.
    g = np.zeros((20, 14), np.int8)
    for r, c in ((6, 0), (7, 1), (8, 2), (9, 3), (10, 4),
                 (0, 6), (1, 7), (2, 8), (3, 9), (4, 10)):
        g[r, c] = 5
    block = np.ones((2, 2), bool)
    for y, x in ((4, 4), (7, 5), (3, 1), (0, 0), (13, 12), (16, 12)):
        _draw(g, y, x, 1, block)
    g[19, 12:14] = 9
    expected = g.copy()
    for y, x in ((4, 4), (7, 5)):
        _draw(expected, y, x, 9, block)
    for y in (7, 10):
        _draw(expected, y, 12, 1, block)
    expected[19, 12:14] = 0
    return dict(input=g, output=expected)


def self_test():
    """Bounded structural/semantic/diversity checks on 42 reproducible seeds."""
    fixture = _hand_fixture()
    assert np.array_equal(solve(fixture['input']), fixture['output'])
    assert np.array_equal(solve(fixture['input'][:, ::-1]), fixture['output'][:, ::-1])
    lookup = np.array([7, 4, 0, 1, 3, 2, 5, 8, 9, 6], np.int8)
    assert np.array_equal(solve(lookup[fixture['input']]), lookup[fixture['output']])
    small = _envelope((5, 5), -1, [(2, -2, 2), (6, -2, 2)])
    expected_mask = np.zeros((5, 5), bool)
    expected_mask[(1, 1, 2, 2, 2, 3, 3), (2, 3, 1, 2, 3, 1, 2)] = True
    assert np.array_equal(small, expected_mask), 'continuous endpoints/parity/strict sides'
    fingerprints, shapes, layouts, slopes, stroke_counts, channels = set(), set(), set(), set(), set(), set()
    source_shapes, pitches, role_palettes = set(), set(), set()
    shift_count = staggered_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _parse(pair['input'])
                f = _features(p)
                shapes.add(f['shape'])
                slopes.add(f['slope'])
                stroke_counts.add(f['strokes'])
                channels.add(len(p['receivers']))
                source_shapes.update(f['source_shapes'])
                pitches.update(f['pitches'])
                role_palettes.add((p['background'], p['boundary'], p['processed']))
                q0 = min(a[0] for a in p['strokes'])
                t0 = min(a[1] for a in p['strokes'])
                layouts.add(tuple((q-q0, lo-t0, hi-t0) for q, lo, hi in p['strokes']))
                shift_count += split == 'test' and len(p['receivers']) == 3
                staggered_count += f['no_common']
    assert len(fingerprints) == 42 and len(shapes) >= 30 and len(layouts) >= 100
    assert slopes == {-1, 1} and {2, 3, 4, 5} <= stroke_counts
    assert channels == {1, 2, 3} and pitches == {3, 4}
    assert len(source_shapes) >= 7 and len(role_palettes) >= 100
    assert shift_count >= 8 and staggered_count >= 8
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(42, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Deliberate defects: lost contribution, recolored partial token, retained bar,
    # changed boundary, and missing demonstration. Validation must reject each.
    base = generate_puzzle(0, 2, 1)
    inadequate = dict(train=[fixture, {r: g[:, ::-1].copy() for r, g in fixture.items()}],
                      test=copy.deepcopy(base['test']))
    try:
        validate_generated_puzzle(inadequate)
    except AssertionError as exc:
        assert 'demonstration' in str(exc), 'correct labels must fail for teaching evidence'
    else:
        raise AssertionError('insufficient demonstrations accepted')
    for defect in ('sum', 'partial', 'bar', 'boundary'):
        bad = copy.deepcopy(base)
        pair = bad['test'][0]
        p = _parse(pair['input'])
        env = _envelope(p['grid'].shape, p['slope'], p['strokes'])
        if defect == 'sum':
            rec = p['receivers'][0]
            y = rec['tokens'][0]['y']-rec['pitch']
            _draw(pair['output'], y, rec['x'], p['background'], rec['mask'])
        elif defect == 'partial':
            a = next(a for a in p['sources'] if _selected(a, env, 'intersection') and not _selected(a, env))
            for r, c in a['cells']:
                pair['output'][r, c] = p['processed']
        elif defect == 'bar':
            for r, c in p['receivers'][0]['bar']['cells']:
                pair['output'][r, c] = p['processed']
        else:
            r, c = p['boundary_cells'][0]
            pair['output'][r, c] = p['background']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'corrupt {defect} accepted')
    return {'seeds': 42, 'distinct_puzzles': len(fingerprints), 'shapes': len(shapes),
            'diagonal_layouts': len(layouts), 'source_masks': len(source_shapes),
            'role_palettes': len(role_palettes), 'three_receiver_tests': shift_count,
            'staggered_pairs': staggered_count}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
