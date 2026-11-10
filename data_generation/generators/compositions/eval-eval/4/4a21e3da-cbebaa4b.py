"""Corner grafts: a guide cuts a carrier; exact keys attach donors to its cuts.

All conventions are visible in one input. No parent code, examples, files, or
generation state are used by solve. Generation and diagnostics stay in memory.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= x <= 30 for x in g.shape)
    assert np.issubdtype(g.dtype, np.integer) and np.all((g >= 0) & (g <= 9))
    return g.astype(np.int8)


def _components(cells):
    todo, result = set(cells), []
    while todo:
        start = min(todo)
        todo.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                p = (r + dr, c + dc)
                if p in todo:
                    todo.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(found)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _norm(cells):
    t, l, _, _ = _bbox(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _analyze(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    rim = np.concatenate((g[0], g[-1], g[1:-1, 0], g[1:-1, -1]))
    bg = int(np.bincount(rim, minlength=10).argmax())
    border = {(0, c) for c in range(w)} | {(h - 1, c) for c in range(w)}
    border |= {(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)}
    marks = sorted(p for p in border if g[p] != bg)
    assert len(marks) in (1, 2)
    marker = int(g[marks[0]])
    assert all(g[p] == marker for p in marks)
    ar = ac = None
    sides = set()
    for r, c in marks:
        assert (r in (0, h - 1)) != (c in (0, w - 1)), 'no corner markers'
        if r in (0, h - 1):
            assert ac is None
            ac = c
            sides.add('top' if r == 0 else 'bottom')
        else:
            assert ar is None
            ar = r
            sides.add('left' if c == 0 else 'right')
    if len(marks) == 2:
        assert ar is not None and ac is not None, 'two markers must be adjacent'

    def on_axis(p):
        return p[0] == ar or p[1] == ac

    foreground = set(map(tuple, np.argwhere(g != bg).tolist())) - set(marks)
    pieces = _components(foreground)
    carriers = [p for p in pieces if any(on_axis(q) for q in p)]
    assert len(carriers) == 1, 'one carrier intersects the guides; donors avoid them'
    carrier = carriers[0]
    assert len(pieces) == (5 if len(marks) == 2 else 3)
    colors = {int(g[p]) for p in carrier}
    assert len(colors) == 2 and marker in colors
    carrier_color = (colors - {marker}).pop()
    payload = {p for p in carrier if g[p] == carrier_color}
    assert len(_components(payload)) == 1
    assert all(not on_axis(p) for p in carrier - payload)
    for axis, coordinate in ((0, ac), (1, ar)):
        if coordinate is not None:
            assert any(p[1 - axis] == coordinate for p in payload)
    regions = {}
    for r, c in sorted(carrier):
        if on_axis((r, c)):
            continue
        vertical = ('top' if r < ar else 'bottom') if ar is not None else next(iter(sides))
        horizontal = ('left' if c < ac else 'right') if ac is not None else next(iter(sides))
        regions.setdefault((vertical, horizontal), set()).add((r, c))
    assert len(regions) == (4 if len(marks) == 2 else 2)
    donors = {}
    for piece in pieces:
        if piece is carrier:
            continue
        palette = {int(g[p]) for p in piece}
        assert len(palette) == 2 and marker in palette and carrier_color not in palette
        pins = {p for p in piece if g[p] == marker}
        body = piece - pins
        assert len(_components(body)) == 1 and 2 <= len(pins) <= 3
        signature = _norm(pins)
        assert signature not in donors, 'unique geometric keys, regardless of area'
        donors[signature] = (piece, pins)
    grafts, used = {}, set()
    for corner, region in regions.items():
        pins = {p for p in region if g[p] == marker}
        assert 2 <= len(pins) <= 3 and region - pins
        signature = _norm(pins)
        assert signature in donors and signature not in used
        used.add(signature)
        donor, donor_pins = donors[signature]
        rt, rl, _, _ = _bbox(pins)
        dt, dl, _, _ = _bbox(donor_pins)
        shift = (rt - dt, rl - dl)
        assert _shift(donor_pins, shift) == pins
        moved = {(r + shift[0], c + shift[1]): int(g[r, c]) for r, c in donor}
        assert region & set(moved) == pins, 'only matched keys overlap'
        cells = {p: int(g[p]) for p in region}
        cells.update(moved)
        grafts[corner] = dict(cells=cells, region=region, pins=pins, donor=donor,
                              donor_shift=shift, kept=bool(set(corner) & sides))
    assert used == set(donors)
    return dict(grid=g, bg=bg, marker=marker, marks=marks, sides=sides,
                ar=ar, ac=ac, carrier=carrier, payload=payload,
                carrier_color=carrier_color, grafts=grafts)


def _render(info, *, graft=True, union_box=True, nearest=False):
    """Ablations are used only to verify informative demonstrations."""
    g, bg, marker = info['grid'], info['bg'], info['marker']
    h, w = g.shape
    out = np.full(g.shape, bg, dtype=np.int8)
    for r, c in info['payload']:
        if r == info['ar'] or c == info['ac']:
            out[r, c] = info['carrier_color']
    for r, c in info['marks']:
        dr, dc = ((1, 0) if r == 0 else (-1, 0) if r == h - 1
                  else (0, 1) if c == 0 else (0, -1))
        ray = []
        while 0 <= r < h and 0 <= c < w:
            ray.append((r, c))
            r, c = r + dr, c + dc
        hits = [i for i, p in enumerate(ray) if p in info['payload']]
        stop = min(hits) if nearest else max(hits)
        for p in ray[:stop + 1]:
            if out[p] == bg:
                out[p] = marker
    for corner, part in info['grafts'].items():
        if not part['kept']:
            continue
        cells = part['cells'] if graft else {p: int(g[p]) for p in part['region']}
        t, l, b, r = _bbox(cells if union_box else part['region'])
        delta = (-t if corner[0] == 'top' else h - 1 - b,
                 -l if corner[1] == 'left' else w - 1 - r)
        for (r, c), color in cells.items():
            p = (r + delta[0], c + delta[1])
            if not (0 <= p[0] < h and 0 <= p[1] < w):
                # Invalid shortcut placements are themselves distinguishing.
                if not union_box:
                    continue
                raise AssertionError('graft does not fit')
            assert out[p] == bg, 'corner grafts and fixed guides must not collide'
            out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_analyze(input_grid))


def _without_a(info):
    """Give docking its correct matches, but omit cuts' motion/rays/deletion.

    A collision or out-of-bounds central assembly is also a failed ablation.
    This counterfactual is never used to construct a label or solve a query.
    """
    g, bg, marker = info['grid'], info['bg'], info['marker']
    out = np.full(g.shape, bg, dtype=np.int8)
    for p in info['carrier'] | set(info['marks']):
        out[p] = g[p]
    for part in info['grafts'].values():
        dr, dc = part['donor_shift']
        for r, c in part['donor']:
            p, color = (r + dr, c + dc), int(g[r, c])
            if not (0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]):
                return None
            if out[p] != bg and not (out[p] == marker and color == marker):
                return None
            out[p] = color
    return out


def _spatial_shortcuts(info):
    """Count actual failures of two plausible replacements for key matching."""
    wrong_side = wrong_nearest = 0
    parts = list(info['grafts'].values())
    centers = [np.mean(np.array(sorted(p['donor'])), axis=0) for p in parts]
    for i, (corner, part) in enumerate(info['grafts'].items()):
        if not part['kept']:
            continue
        r, c = centers[i]
        actual = (('top' if r < info['ar'] else 'bottom') if info['ar'] is not None else corner[0],
                  ('left' if c < info['ac'] else 'right') if info['ac'] is not None else corner[1])
        wrong_side += actual != corner
        target = np.mean(np.array(sorted(part['pins'])), axis=0)
        distances = [float(np.sum((center - target) ** 2)) for center in centers]
        wrong_nearest += distances[i] > min(distances) + 1e-8
    return wrong_side, wrong_nearest


def _short_endpoint(info):
    payload = info['payload']
    h, w = info['grid'].shape
    for r, c in info['marks']:
        if r in (0, h - 1):
            aligned = [rr for rr, cc in payload if cc == c]
            choose = max if r == 0 else min
            if choose(aligned) != choose(rr for rr, cc in payload):
                return True
        else:
            aligned = [cc for rr, cc in payload if rr == r]
            choose = max if c == 0 else min
            if choose(aligned) != choose(cc for rr, cc in payload):
                return True
    return False


def _path(rng, body, target, allowed):
    """Connect a target to a connected body by randomized shortest grid paths."""
    if target in body:
        return
    queue = [target]
    previous = {target: None}
    endpoint = None
    for p in queue:
        if p in body:
            endpoint = p
            break
        for i in rng.permutation(4):
            dr, dc = N4[int(i)]
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    assert endpoint is not None
    while endpoint is not None:
        body.add(endpoint)
        endpoint = previous[endpoint]


def _local_scene(rng, double, witness, key_size):
    # Canonical markers are top and, if present, left. Whole-scene symmetries
    # later cover every side; individual bodies never turn during solving.
    up, down, left, right = (int(v) for v in rng.integers(3, 6, size=4))
    bounds = (-up, -left, down, right)
    allowed = {(r, c) for r in range(-up, down + 1) for c in range(-left, right + 1)}
    labels = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if double else [(0, -1), (0, 1)]
    keys, targets, donor_data, signatures = {}, set(), {}, set()
    for label in labels:
        vr, hc = label
        rows = list(range(-up, 0)) if vr == -1 else list(range(1, down + 1)) if vr == 1 else list(range(-up, down + 1))
        cols = list(range(-left, 0)) if hc == -1 else list(range(1, right + 1))
        for _ in range(80):
            horizontal = vr == 0 or bool(rng.integers(2))
            tangent = rows if horizontal else cols
            ts = sorted(int(v) for v in rng.choice(tangent, key_size, replace=False))
            offsets = [int(v) for v in rng.integers(0, 2, size=key_size)]
            if horizontal:
                boundary = right if hc == 1 else -left
                pins = {(t, boundary + hc * (1 + o)) for t, o in zip(ts, offsets)}
                inward = {(t, boundary + hc * o) for t, o in zip(ts, offsets)}
                normal = (0, hc)
            else:
                boundary = down if vr == 1 else -up
                pins = {(boundary + vr * (1 + o), t) for t, o in zip(ts, offsets)}
                inward = {(boundary + vr * o, t) for t, o in zip(ts, offsets)}
                normal = (vr, 0)
            signature = _norm(pins)
            if signature not in signatures:
                break
        else:
            return None
        signatures.add(signature)
        keys[label] = pins
        # Attach each key tooth to the base rectangle; a one-cell stagger
        # changes key geometry without requiring connected marker blobs.
        for r, c in inward:
            targets.add((min(down, max(-up, r)), min(right, max(-left, c))))
        # Donors grow beyond the opposite face of the same key. Their depth,
        # tangential padding and silhouette change the fused bounding box.
        nr, nc = normal
        tangent_values = [r if horizontal else c for r, c in pins]
        lo = min(tangent_values) - int(rng.integers(0, 3))
        hi = max(tangent_values) + int(rng.integers(0, 3))
        depth = int(rng.integers(1, 5))
        if horizontal:
            edge = max(c * hc for r, c in pins) + 1
            donor_allowed = {(t, hc * n) for t in range(lo, hi + 1) for n in range(edge, edge + depth)}
            spine = {(t, hc * edge) for t in range(lo, hi + 1)}
            arms = {(r, hc * n) for r, c in pins for n in range(c * hc + 1, edge + 1)}
        else:
            edge = max(r * vr for r, c in pins) + 1
            donor_allowed = {(vr * n, t) for t in range(lo, hi + 1) for n in range(edge, edge + depth)}
            spine = {(vr * edge, t) for t in range(lo, hi + 1)}
            arms = {(vr * n, c) for r, c in pins for n in range(r * vr + 1, edge + 1)}
        # Solid, open comb, and irregular connected bodies share one operation.
        body = set(spine)
        style = int(rng.integers(3))
        if style == 0:
            body |= donor_allowed
        elif style == 1:
            for p in sorted(donor_allowed):
                if p[0] in (min(r for r, c in donor_allowed), max(r for r, c in donor_allowed)) or p[1] in (min(c for r, c in donor_allowed), max(c for r, c in donor_allowed)):
                    body.add(p)
        else:
            for _ in range(int(rng.integers(1, max(2, len(donor_allowed))))):
                frontier = sorted({(r + dr, c + dc) for r, c in body for dr, dc in N4} & donor_allowed - body)
                if not frontier:
                    break
                body.add(frontier[int(rng.integers(len(frontier)))])
        donor_data[label] = (body | arms, pins)
    body = {(int(rng.integers(-up, down + 1)), 0)}
    waypoints = sorted(targets)
    waypoints += [(int(rng.integers(-up, down + 1)), int(rng.integers(-left, right + 1)))
                  for _ in range(int(rng.integers(2, 8)))]
    # Endpoints on each axis are targets, but the path connecting them can detour.
    waypoints += [(-up + int(rng.integers(2)), 0), (down - int(rng.integers(2)), 0)]
    if double:
        waypoints += [(0, -left + int(rng.integers(2))), (0, right - int(rng.integers(2)))]
    rng.shuffle(waypoints)
    for p in waypoints:
        _path(rng, body, p, allowed)
    for label, pins in keys.items():
        vr, hc = label
        for r, c in pins:
            p = (min(down, max(-up, r)), min(right, max(-left, c)))
            # A pin's inward arm is just outside the base rectangle.
            if r < -up or r > down:
                body |= {(x, c) for x in range(min(r, p[0]) + 1, max(r, p[0]))}
            if c < -left or c > right:
                body |= {(r, x) for x in range(min(c, p[1]) + 1, max(c, p[1]))}
    pins = set().union(*keys.values())
    assert not body & pins and len(_components(body)) == 1
    if witness:
        values = sorted(r for r, c in body if c == 0)
        if not any(b > a + 1 for a, b in zip(values, values[1:])):
            return None
        groups = [{p for p in body if p[1] * hc > 0 and (not double or p[0] * vr > 0)}
                  for vr, hc in labels if not double or vr == -1 or hc == -1]
        if not any(len(_components(group)) > 1 for group in groups):
            return None
    return body, keys, donor_data


def _make_pair(rng, double, witness, palette, turn, reflect, key_size):
    for _ in range(160):
        scene = _local_scene(rng, double, witness, key_size)
        if scene is None:
            continue
        body, keys, donor_data = scene
        bg, marker, carrier_color, *donor_colors = palette
        carrier_map = {p: carrier_color for p in body}
        carrier_map.update({p: marker for ps in keys.values() for p in ps})
        # Compute complete graft extents before choosing a feasible canvas/axis.
        extents = {}
        for label, pins in keys.items():
            vr, hc = label
            region = {p for p in carrier_map if p[1] * hc > 0 and (not double or p[0] * vr > 0)}
            union = region | donor_data[label][0] | pins
            t, l, b, r = _bbox(union)
            extents[label] = (b - t + 1, r - l + 1)
        ct, cl, cb, cr = _bbox(carrier_map)
        minleft = max(-cl + 2, max(v[1] for k, v in extents.items() if k[1] == -1) + 2)
        minright = max(cr + 2, max(v[1] for k, v in extents.items() if k[1] == 1) + 2)
        if double:
            mintop = max(-ct + 2, max(v[0] for k, v in extents.items() if k[0] == -1) + 2)
            minbottom = max(cb + 2, max(v[0] for k, v in extents.items() if k[0] == 1) + 2)
        else:
            mintop = max(6, -_bbox(carrier_map)[0] + 2)
            minbottom = max(6, _bbox(carrier_map)[2] + 2)
        lowh = max(20, mintop + minbottom + 1)
        loww = max(20, minleft + minright + 1)
        if max(lowh, loww) > 30:
            continue
        h = int(rng.integers(lowh, 31))
        w = int(rng.integers(loww, 31))
        ar = int(rng.integers(mintop, h - minbottom))
        ac = int(rng.integers(minleft, w - minright))
        carrier = {(r + ar, c + ac): color for (r, c), color in carrier_map.items()}
        occupied = set(carrier)
        donors = list(donor_data.values())
        rng.shuffle(donors)
        maps = [carrier]
        for index, (donor_body, donor_pins) in enumerate(donors):
            cells = donor_body | donor_pins
            t, l, b, r = _bbox(cells)
            choices = [(rr - t, cc - l) for rr in range(1, h - (b - t) - 1)
                       for cc in range(1, w - (r - l) - 1)]
            rng.shuffle(choices)
            forbidden = {(r + dr, c + dc) for r, c in occupied for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            for delta in choices:
                moved = _shift(cells, delta)
                if any(c == ac or (double and r == ar) for r, c in moved) or moved & forbidden:
                    continue
                color = donor_colors[index % len(donor_colors)]
                placed = {p: color for p in _shift(donor_body, delta)}
                placed.update({p: marker for p in _shift(donor_pins, delta)})
                occupied |= moved
                maps.append(placed)
                break
            else:
                break
        if len(maps) != len(donors) + 1:
            continue
        g = np.full((h, w), bg, dtype=np.int8)
        for mapping in maps:
            for p, color in mapping.items():
                g[p] = color
        g[0, ac] = marker
        if double:
            g[ar, 0] = marker
        g = np.rot90(g, turn).copy()
        if reflect:
            g = g[:, ::-1].copy()
        info = _analyze(g)
        out = _render(info)
        if np.array_equal(_render(info, union_box=False), out):
            continue
        if witness and (min(_spatial_shortcuts(info)) == 0 or not _short_endpoint(info)):
            continue
        return {'input': g, 'output': out}
    raise RuntimeError('bounded carrier/donor construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _int(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _int(num_train) or not _int(num_test) or not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed)
    palette = [int(v) for v in rng.choice(10, 6, replace=False)]
    varied_palette = bool(rng.integers(2))
    base_turn = int(rng.integers(4))
    reflect = bool(rng.integers(2))
    # Both arities appear even with two demonstrations. A new whole-scene
    # orientation is the controlled query shift; key geometry already varies.
    train_turns = [base_turn, (base_turn + 1) % 4]
    held_out = {False: (base_turn + 2) % 4, True: (base_turn + 3) % 4}
    seen_key_sizes = set()
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            double = bool(i % 2) if split == 'train' else bool(rng.integers(2))
            if split == 'train':
                turn = train_turns[i] if i < 2 else int(rng.choice([v for v in range(4) if v != held_out[double]]))
                key_size = 2 if i < 2 else int(rng.choice([2, 3], p=[.75, .25]))
                seen_key_sizes.add(key_size)
            else:
                turn = held_out[double]
                key_size = int(rng.choice(sorted(seen_key_sizes)))
            if varied_palette:
                palette = [int(v) for v in rng.choice(10, 6, replace=False)]
            # Repeated donor hues are optional, never identifiers for pieces.
            actual_palette = palette[:3] + (palette[3:4] if rng.random() < .45 else palette[3:])
            for _ in range(12):
                pair = _make_pair(rng, double, split == 'train' and i < 2,
                                  actual_palette, turn, reflect, key_size)
                ik = _key(pair['input'])
                if ik not in seen:
                    seen.add(ik)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not construct distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(input_grid):
    info = _analyze(input_grid)
    g, payload = info['grid'], info['payload']
    gaps = []
    for axis, coordinate in ((0, info['ac']), (1, info['ar'])):
        if coordinate is not None:
            values = sorted(p[axis] for p in payload if p[1 - axis] == coordinate)
            gaps.extend(b - a - 1 for a, b in zip(values, values[1:]) if b > a + 1)
    parts = list(info['grafts'].values())
    body_masks = []
    for p in parts:
        donor_body = {q for q in p['donor'] if g[q] != info['marker']}
        body_masks.append(_norm(donor_body))
    wrong_side, wrong_nearest = _spatial_shortcuts(info)
    return dict(shape=g.shape, sides=tuple(sorted(info['sides'])), roles=(info['bg'], info['marker'], info['carrier_color']),
                regions=len(parts), gap_lengths=gaps,
                fragment_counts=[len(_components(p['region'] - p['pins'])) for p in parts if p['kept']],
                keys=[_norm(p['pins']) for p in parts], donor_masks=body_masks,
                donor_colors=[int(g[next(q for q in p['donor'] if g[q] != info['marker'])]) for p in parts],
                deleted=sum(len(p['cells']) for p in parts if not p['kept']),
                wrong_side=wrong_side, wrong_nearest=wrong_nearest,
                short_endpoint=_short_endpoint(info),
                carrier_area=len(payload), graft_boxes=[(_bbox(p['cells'])[2] - _bbox(p['cells'])[0] + 1,
                                                       _bbox(p['cells'])[3] - _bbox(p['cells'])[1] + 1) for p in parts])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, fs = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            expected = _render(info)
            assert np.array_equal(expected, out) and not np.array_equal(inp, out)
            no_a = _without_a(info)
            assert no_a is None or not np.array_equal(no_a, out), 'A must matter even with oracle docking'
            assert not np.array_equal(_render(info, graft=False), out), 'B must matter'
            assert not np.array_equal(_render(info, union_box=False), out), 'donor affects carrier translation'
            # Independently account for each colored cell, including deduped
            # keys, deleted donors, fixed carrier cells, and guide-gap paint.
            counts = Counter(int(inp[p]) for p in info['payload'] if p[0] == info['ar'] or p[1] == info['ac'])
            for p in info['grafts'].values():
                if p['kept']:
                    counts.update(p['cells'].values())
            for color, number in counts.items():
                if color != info['marker']:
                    assert np.count_nonzero(out == color) == number
            fixed = {p for p in info['payload'] if p[0] == info['ar'] or p[1] == info['ac']}
            guide_cells = set()
            for r, c in info['marks']:
                if r in (0, inp.shape[0] - 1):
                    aligned = [rr for rr, cc in info['payload'] if cc == c]
                    start, stop = (0, max(aligned)) if r == 0 else (min(aligned), inp.shape[0] - 1)
                    guide_cells |= {(rr, c) for rr in range(start, stop + 1)}
                else:
                    aligned = [cc for rr, cc in info['payload'] if rr == r]
                    start, stop = (0, max(aligned)) if c == 0 else (min(aligned), inp.shape[1] - 1)
                    guide_cells |= {(r, cc) for cc in range(start, stop + 1)}
            pins_in = sum(len(p['pins']) for p in info['grafts'].values())
            pins_out = sum(len(p['pins']) for p in info['grafts'].values() if p['kept'])
            assert np.count_nonzero(inp == info['marker']) == len(info['marks']) + 2 * pins_in
            assert np.count_nonzero(out == info['marker']) == pins_out + len(guide_cells - fixed)
            assert np.count_nonzero(out != info['bg']) == len(fixed | guide_cells) + sum(len(p['cells']) for p in info['grafts'].values() if p['kept'])
            for r, c in info['payload']:
                if r == info['ar'] or c == info['ac']:
                    assert out[r, c] == info['carrier_color']
            for corner, p in info['grafts'].items():
                if p['kept']:
                    t, l, b, r = _bbox(p['cells'])
                    assert b - t + 1 <= inp.shape[0] and r - l + 1 <= inp.shape[1]
            fs[split].append(pair_features(inp))
    assert {f['regions'] for f in fs['train']} == {2, 4}
    assert {f['sides'] for f in fs['test']}.isdisjoint({f['sides'] for f in fs['train']})
    assert {len(k) for f in fs['test'] for k in f['keys']} <= {len(k) for f in fs['train'] for k in f['keys']}
    assert any(f['gap_lengths'] for f in fs['train'])
    assert any(max(f['fragment_counts']) > 1 for f in fs['train'])
    assert any(f['deleted'] > 0 for f in fs['train'])
    assert all(any(f[name] > 0 for f in fs['train']) for name in ('wrong_side', 'wrong_nearest'))
    for arity in (2, 4):
        witnesses = [f for f in fs['train'] if f['regions'] == arity]
        assert any(f['gap_lengths'] and max(f['fragment_counts']) > 1 and f['short_endpoint'] for f in witnesses)
        assert any(len({len(k) for k in f['keys']}) < len(f['keys']) for f in witnesses)
    assert any(not np.array_equal(_render(_analyze(p['input']), nearest=True), p['output']) for p in puzzle['train'])


def self_test():
    """42 puzzles, bounded aggregate sets, no retained grids or disk writes."""
    shapes, keys, masks, roles, signatures = set(), set(), set(), set(), set()
    orientations = {'train': set(), 'test': set()}
    total = gaps = fragments = reuse = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = pair_features(pair['input'])
                shapes.add(f['shape'])
                keys.update(f['keys'])
                masks.update(f['donor_masks'])
                roles.add(f['roles'])
                orientations[split].add(f['sides'])
                total += 1
                gaps += bool(f['gap_lengths'])
                fragments += max(f['fragment_counts']) > 1
                reuse += len(set(f['donor_colors'])) < len(f['donor_colors'])
                for role in ('input', 'output'):
                    digest.update(bytes(pair[role].shape))
                    digest.update(pair[role].tobytes())
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 40 and len(keys) >= 25
    assert len(masks) >= 120 and len(roles) >= 30
    assert all(len(v) == 8 for v in orientations.values())
    assert gaps >= 84 and fragments >= 84 and reuse >= 50
    a, b = generate_puzzle(712926472, 2, 1), generate_puzzle(712926472, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for pair in a['train'] + a['test']:
        inp, out = pair['input'], pair['output']
        for turn in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn))
        assert np.array_equal(solve(inp[:, ::-1]), out[:, ::-1])
        permutation = np.array([9, 4, 7, 2, 0, 8, 3, 6, 1, 5], dtype=np.int8)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
    # A hand-built family member outside the sampler: 18x18 and mixed key
    # cardinalities. Expected corner translations are explicitly (-5,-3)
    # and (-5,+3), not obtained by calling the parser or renderer.
    small = np.zeros((18, 18), dtype=np.int8)
    carrier_body = ({(r, 8) for r in (5, 7, 8, 9)}
                    | {(r, c) for r in (5, 7) for c in range(5, 12)}
                    | {(6, 7), (6, 11), (6, 12)})
    carrier_keys = {(5, 4), (7, 4), (5, 12), (6, 13), (7, 12)}
    for p in carrier_body:
        small[p] = 3
    for p in carrier_keys | {(0, 8), (1, 3), (3, 3), (12, 13), (13, 14), (14, 13)}:
        small[p] = 2
    for r in (1, 2, 3):
        small[r, 2] = 1
    donor_right = {(12, 14), (12, 15), (13, 15), (14, 14), (14, 15)}
    for p in donor_right:
        small[p] = 4
    expected = np.zeros_like(small)
    expected[:10, 8] = 2
    for r in (5, 7, 8, 9):
        expected[r, 8] = 3
    for r, c in carrier_body | carrier_keys:
        if c < 8:
            expected[r - 5, c - 3] = small[r, c]
        elif c > 8:
            expected[r - 5, c + 3] = small[r, c]
    expected[:3, 0] = 1
    for r, c in donor_right:
        expected[r - 12, c + 2] = 4
    assert np.array_equal(solve(small), expected)
    # Negative semantic fixtures: deleting a donor cell or leaving an original
    # scattered donor in place must be detected by validation.
    for edit in ('erase', 'retain'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in pairs] for s, pairs in a.items()}
        p = bad['test'][0]
        info = _analyze(p['input'])
        if edit == 'erase':
            q = tuple(np.argwhere(p['output'] != info['bg'])[0])
            p['output'][q] = info['bg']
        else:
            donor = next(iter(info['grafts'].values()))['donor']
            for q in donor:
                p['output'][q] = p['input'][q]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('semantic corruption accepted')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=total, shapes=len(shapes), key_geometries=len(keys),
                donor_masks=len(masks), role_triples=len(roles),
                gapped_pairs=gaps, fragmented_pairs=fragments, repeated_donor_colors=reuse,
                orientations={s: len(v) for s, v in orientations.items()})
