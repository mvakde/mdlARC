"""Registered exposure: develop a detached stencil in a virtual spotlight.

Only Python's standard library and NumPy are used. Geometry, color roles,
registration and illumination are decoded from the supplied input pixels.
"""
from __future__ import annotations

import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(cells, diagonal=True):
    remaining = set(cells)
    parts = []
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if (dy or dx) and (diagonal or not (dy and dx))]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, pending = {start}, [start]
        while pending:
            y, x = pending.pop()
            for dy, dx in steps:
                q = (y + dy, x + dx)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    pending.append(q)
        parts.append(part)
    return parts


def _bbox(cells):
    return (min(y for y, x in cells), max(y for y, x in cells),
            min(x for y, x in cells), max(x for y, x in cells))


def _key(a):
    a = np.asarray(a)
    return (a.shape, a.tobytes())


def _fit_rail(values, sign):
    """Read the longest suffix showing two outward steps; least period wins.

    A thin rail has one pixel per axial column. The return value predicts
    displacement from its last visible pixel, including the observed phase.
    """
    candidates = []
    n = len(values)
    for period in (1, 2, 3):
        for phase in range(period):
            predicted = [values[-1] + sign * ((i - n + 1 + phase) // period)
                         for i in range(n)]
            length = 0
            for i in range(n - 1, -1, -1):
                if predicted[i] != values[i]:
                    break
                length += 1
            if length >= 3 and len(set(values[n - length:])) >= 3:
                candidates.append((length, -period, phase))
    if not candidates:
        raise ValueError("each terminal rail must show two readable outward steps")
    length, neg_period, phase = max(candidates)
    return (-neg_period, phase, int(values[-1]), length)


def _decode(value):
    original = _grid(value)
    counts = np.bincount(original.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= original.size / 2:
        raise ValueError("background must occupy more than half the input")
    matches = []
    for rotation in range(4):
        a = np.rot90(original, -rotation)
        all_cells = {tuple(map(int, q)) for q in np.argwhere(a != bg)}
        objects = _components(all_cells)
        if len(objects) != 2:
            continue
        colors = list(map(int, np.unique(a)))
        for marker in colors:
            if marker == bg:
                continue
            bars = _components({tuple(map(int, q)) for q in np.argwhere(a == marker)}, False)
            if (len(bars) != 2 or len(bars[0]) != len(bars[1])
                    or not 1 <= len(bars[0]) <= 4):
                continue
            length = len(bars[0])
            if any(len({x for y, x in b}) != 1
                   or max(y for y, x in b) - min(y for y, x in b) + 1 != length
                   for b in bars):
                continue
            own = [[b for b in bars if b <= obj] for obj in objects]
            if any(len(v) != 1 for v in own):
                continue
            for ei in range(2):
                emitter, plate = objects[ei], objects[1 - ei]
                eb, tb = own[ei][0], own[1 - ei][0]
                ey, _, ex, _ = _bbox(eb)
                ty, _, tx, _ = _bbox(tb)
                body = plate - tb
                if not body or any(x <= tx for y, x in body):
                    continue
                for material in colors:
                    if material in (bg, marker) or not any(a[q] == material for q in body):
                        continue
                    walls = {q for q in emitter if a[q] == material}
                    if len(walls) < 11:
                        continue
                    _, _, rear, mouth = _bbox(walls)
                    if mouth != ex or mouth - rear < 4:
                        continue
                    cap = sorted(y for y, x in walls if x == rear)
                    if len(cap) < 3 or cap != list(range(cap[0], cap[-1] + 1)):
                        continue
                    upper, lower = [], []
                    for x in range(rear + 1, mouth + 1):
                        rows = sorted(y for y, xx in walls if xx == x)
                        if len(rows) != 2 or rows[1] - rows[0] < 3:
                            break
                        upper.append(rows[0])
                        lower.append(rows[1])
                    if len(upper) != mouth - rear:
                        continue
                    if (abs(upper[0] - cap[0]) > 1 or abs(lower[0] - cap[-1]) > 1
                            or any(abs(v - u) > 1 for u, v in zip(upper, upper[1:]))
                            or any(abs(v - u) > 1 for u, v in zip(lower, lower[1:]))):
                        continue
                    if ey + length != upper[-1]:
                        continue
                    if len(_components(walls | eb)) != 1:
                        continue
                    other = emitter - walls - eb
                    payloads = {int(a[q]) for q in other}
                    if len(payloads) != 1 or not other:
                        continue
                    payload = payloads.pop()
                    if payload in (bg, marker, material) or any(a[q] == payload for q in body):
                        continue
                    if len(_components(other, False)) != 1:
                        continue
                    if not any(x == rear + 1 for y, x in other):
                        continue
                    if any(not (rear < x <= mouth and upper[x - rear - 1] < y
                                < lower[x - rear - 1]) for y, x in other):
                        continue
                    try:
                        fits = (_fit_rail(upper, -1), _fit_rail(lower, 1))
                    except ValueError:
                        continue
                    matches.append(dict(a=a, rotation=rotation, bg=bg, material=material,
                                        marker=marker, payload=payload, length=length,
                                        emitter=emitter, plate=plate, body=body,
                                        eb=eb, tb=tb, rear=rear, mouth=mouth,
                                        upper=upper, lower=lower, fits=fits,
                                        translation=(ey - ty, ex + 1 - tx)))
    if len(matches) != 1:
        raise ValueError(f"input has {len(matches)} admissible registered projectors")
    return matches[0]


def _inside(p, y, x, widening=True):
    if x <= p['rear']:
        return False
    if x <= p['mouth']:
        lo, hi = p['upper'][x - p['rear'] - 1], p['lower'][x - p['rear'] - 1]
    else:
        limits = []
        for fit, sign in zip(p['fits'], (-1, 1)):
            period, phase, end, _ = fit
            limits.append(end + (sign * ((x - p['mouth'] + phase) // period) if widening else 0))
        lo, hi = limits
    return lo < y < hi


def _exposure(p, registration=True, widening=True, shadows=True, all_material=False,
              global_blockers=False, clip=False):
    """Pixel-only lane scan. Flags are explicit diagnostic alternative rules."""
    a = p['a']
    dy, dx = p['translation'] if registration else (0, 0)
    lit, dark, effective, inert = set(), set(), set(), set()
    lanes = {}
    for q in p['body']:
        lanes.setdefault(q[0], []).append(q)
    for y in sorted(lanes):
        blocked = False
        for q in sorted(lanes[y], key=lambda v: v[1]):
            yy, xx = q[0] + dy, q[1] + dx
            inside = _inside(p, yy, xx, widening)
            if clip and not (0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]):
                inside = False
            if a[q] != p['material']:
                (effective if inside else inert).add(q)
                if inside or global_blockers:
                    blocked = True
            elif all_material or (inside and not (blocked and shadows)):
                lit.add(q)
            elif inside and blocked and shadows:
                dark.add(q)
    return lit, dark, effective, inert


def _exposable_gaps(p):
    """Empty support positions a mistaken background-painting rule would fill."""
    _, _, effective, _ = _exposure(p)
    dy, dx = p['translation']
    t, b, l, r = _bbox(p['body'])
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)
            if (y, x) not in p['body'] and _inside(p, y + dy, x + dx)
            and not any(q[0] == y and q[1] < x for q in effective)}


def _render(p, paint_gaps=False, **options):
    lit, _, _, _ = _exposure(p, **options)
    t, b, l, r = _bbox(p['body'])
    out = np.full((b - t + 1, r - l + 1), p['bg'], dtype=np.int8)
    for q in p['body']:
        out[q[0] - t, q[1] - l] = p['payload'] if q in lit else p['a'][q]
    if paint_gaps:
        for y, x in _exposable_gaps(p):
            out[y - t, x - l] = p['payload']
    return np.rot90(out, p['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Develop the cropped stencil using only this input, without latent state."""
    return _render(_decode(input_grid))


def _wrong_seam(p, adjustment):
    """Diagnostic only: overlap the bars or leave an extra column between them."""
    dy, dx = p['translation']
    wrong = dict(p, translation=(dy, dx + adjustment))
    return _render(wrong)


def _stencil(rng, length, width, ymin, ymax, style):
    rect = {(y, x) for y in range(ymin, ymax + 1) for x in range(1, width + 1)}
    if style in ('ring', 'ribbed'):
        thickness = int(rng.choice([1, 1, 2]))
        cells = {q for q in rect if q[0] < ymin + thickness or q[0] > ymax - thickness
                 or q[1] <= thickness or q[1] > width - thickness}
        if style == 'ribbed':
            divider = int(rng.integers(2, width))
            cells |= {(y, divider) for y in range(ymin, ymax + 1)}
    elif style == 'comb':
        cells = {(y, 1) for y in range(ymin, ymax + 1)}
        rows = {ymin, ymax, int(rng.integers(0, length + 1))}
        rows |= {y for y in range(ymin, ymax + 1) if rng.random() < .45}
        for y in sorted(rows):
            end = width if y in (ymin, ymax) else int(rng.integers(2, width + 1))
            cells |= {(y, x) for x in range(1, end + 1)}
    elif style == 'branch':
        cells = {(y, 1) for y in range(ymin, ymax + 1)}
        for y in sorted({ymin, ymax, *map(int, rng.integers(ymin, ymax + 1, size=3))}):
            end = int(rng.integers(3, width + 1))
            cells |= {(y, x) for x in range(1, end + 1)}
            limb = int(rng.integers(2, end + 1))
            stop = int(rng.integers(ymin, ymax + 1))
            cells |= {(r, limb) for r in range(min(y, stop), max(y, stop) + 1)}
        cells |= {(ymax, x) for x in range(1, width + 1)}
    elif style == 'stair':
        end = int(rng.integers(2, width + 1))
        cells = set()
        for y in range(ymin, ymax + 1):
            end = max(2, min(width, end + int(rng.integers(-1, 2))))
            cells |= {(y, x) for x in range(1, end + 1)}
        cells |= {(ymax, x) for x in range(1, width + 1)}
    else:
        cells = rect.copy()
        if style == 'notched':
            cut_y = int(rng.integers(ymin + 1, ymax))
            cut_x = int(rng.integers(2, width))
            cells -= {(y, x) for y, x in rect if y < cut_y and x >= cut_x}
        elif style == 'perforated':
            for y in range(ymin + 1, ymax, 2):
                for x in range(2, width, 2):
                    if rng.random() < .65:
                        cells.discard((y, x))
    # The bar attaches to the left rim. This rim is transmissive material,
    # not an opaque wall; internal gaps retain their background in the print.
    cells |= {(y, 1) for y in range(length)}
    return cells


def _stage(rng, emitter, target, bg):
    def normal(obj):
        t, b, l, r = _bbox(obj)
        return {(y - t, x - l): v for (y, x), v in obj.items()}, b - t + 1, r - l + 1
    e, eh, ew = normal(emitter)
    t, th, tw = normal(target)
    modes = list(rng.permutation(['horizontal', 'vertical', 'diagonal', 'interleaved']))
    offset = None
    forbidden = {(y + dy, x + dx) for y, x in e
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    for mode in modes:
        candidates = []
        if mode == 'interleaved':
            candidates = [(dy, dx) for dy in range(-th + 1, eh)
                          for dx in range(-tw + 1, ew)]
            rng.shuffle(candidates)
        else:
            for _ in range(12):
                gap = int(rng.integers(1, 6))
                if mode == 'horizontal':
                    dx = ew + gap if rng.integers(2) else -tw - gap
                    dy = int(rng.integers(-th + 1, eh))
                elif mode == 'vertical':
                    dy = eh + gap if rng.integers(2) else -th - gap
                    dx = int(rng.integers(-tw + 1, ew))
                else:
                    dy = eh + gap if rng.integers(2) else -th - gap
                    dx = ew + gap if rng.integers(2) else -tw - gap
                candidates.append((dy, dx))
        for dy, dx in candidates:
            moved = {(y + dy, x + dx) for y, x in t}
            if moved & forbidden:
                continue
            top, bottom, left, right = _bbox(set(e) | moved)
            if bottom - top < 30 and right - left < 30:
                offset = (dy, dx)
                break
        if offset is not None:
            break
    if offset is None:
        return None
    dy, dx = offset
    cells = dict(e)
    cells.update({(y + dy, x + dx): v for (y, x), v in t.items()})
    top, bottom, left, right = _bbox(cells)
    h, w = bottom - top + 1, right - left + 1
    pads = [int(rng.integers(0, 4)) for _ in range(4)]
    pads[0] = min(pads[0], 30 - h)
    pads[1] = min(pads[1], 30 - h - pads[0])
    pads[2] = min(pads[2], 30 - w)
    pads[3] = min(pads[3], 30 - w - pads[2])
    h += pads[0] + pads[1]
    w += pads[2] + pads[3]
    if rng.random() < .15:
        if rng.integers(2):
            h = 30
        else:
            w = 30
    if len(cells) >= h * w / 2:
        h = 30
        w = 30
    a = np.full((h, w), bg, dtype=np.int8)
    for (y, x), v in cells.items():
        a[y - top + pads[0], x - left + pads[2]] = v
    return a


_STYLES = ('ring', 'ribbed', 'comb', 'branch', 'stair', 'filled', 'notched', 'perforated')


def _build_pair(rng, palette, rotation, length, width, kind, support_witness=False):
    bg, fg, marker, payload, opaque1, opaque2 = map(int, palette[:6])
    # A separate rear reservoir permits bulges and necks. Only the terminal
    # flare is periodic; neither rail is extended backward through the chamber.
    neck = int(rng.integers(2, 5))
    u, v = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    upper, lower = [], []
    uu, vv = u, v
    for i in range(neck):
        if i:
            uu = max(2, min(4, uu + int(rng.integers(-1, 2))))
            vv = max(2, min(4, vv + int(rng.integers(-1, 2))))
        upper.append(-uu)
        lower.append(vv)
    periods = [int(rng.integers(1, 4)) for _ in range(2)]
    phases = [int(rng.integers(p)) for p in periods]
    flare = 2 * max(periods) + int(rng.integers(1, 3))
    for i in range(1, flare + 1):
        upper.append(-uu - (i + phases[0]) // periods[0])
        lower.append(vv + (i + phases[1]) // periods[1])
    mouth = len(upper)
    bar_y = upper[-1] - length
    emitter = {(y, 0): fg for y in range(upper[0], lower[0] + 1)}
    for x, (lo, hi) in enumerate(zip(upper, lower), 1):
        emitter[lo, x] = emitter[hi, x] = fg
    emitter.update({(bar_y + y, mouth): marker for y in range(length)})
    # Grow a connected, sometimes off-center reservoir patch touching the cap.
    chamber = {(y, x) for x in range(1, neck + 1)
               for y in range(upper[x - 1] + 1, lower[x - 1])}
    first_y = int(rng.integers(upper[0] + 1, lower[0]))
    patch = {(first_y, 1)}
    for _ in range(int(rng.integers(0, 6))):
        frontier = sorted({(y + dy, x + dx) for y, x in patch
                           for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                           if (y + dy, x + dx) in chamber and (y + dy, x + dx) not in patch})
        if not frontier:
            break
        patch.add(frontier[int(rng.integers(len(frontier)))])
    emitter.update({q: payload for q in patch})
    ymin = -int(rng.integers(1, 4))
    ymax = (length + lower[-1] - upper[-1] + int(rng.integers(1, 4))
            if rng.random() < .30 else length + int(rng.integers(3, 8)))
    style = (str(rng.choice(['ring', 'ribbed', 'perforated'])) if support_witness else
             str(rng.choice(['filled', 'notched', 'perforated'])) if kind == 'teach' else
             str(rng.choice(_STYLES)))
    material = _stencil(rng, length, width, ymin, ymax, style)
    target = {q: fg for q in material}
    target.update({(y, 0): marker for y in range(length)})

    # Construction's analytic ray formula is separate from input decoding's
    # suffix fit and its lane scan. No staged canvas bounds clip virtual rays.
    def inside(q):
        y, x = q
        distance = x + 1  # target bar is one column beyond the emitter bar
        lo = upper[-1] - (flare + distance + phases[0]) // periods[0] + (flare + phases[0]) // periods[0]
        hi = lower[-1] + (flare + distance + phases[1]) // periods[1] - (flare + phases[1]) // periods[1]
        return lo < y + bar_y < hi

    def shadow_candidates():
        return sorted(q for q in material if inside(q)
                      and any((q[0], x) in material and inside((q[0], x)) for x in range(1, q[1]))
                      and any((q[0], x) in material and inside((q[0], x)) for x in range(q[1] + 1, width + 1)))

    blockers = set()
    if kind == 'teach':
        choices = shadow_candidates()
        for color in (opaque1, opaque2):
            eligible = [q for q in choices if q[0] not in {r for r, c in blockers}]
            if not eligible:
                return None
            q = eligible[int(rng.integers(len(eligible)))]
            target[q] = color
            blockers.add(q)
        # A cell before its lane enters the cone is inert. Later material in
        # that same lane must still expose, which rejects global-row blocking.
        inert = sorted(q for q in material if not inside(q)
                       and q[0] not in {r for r, c in blockers}
                       and any((q[0], x) in material and inside((q[0], x))
                               for x in range(q[1] + 1, width + 1)))
        if not inert:
            return None
        q = inert[int(rng.integers(len(inert)))]
        target[q] = opaque1
        blockers.add(q)
    elif kind != 'clean':
        choices = shadow_candidates()
        if not choices:
            return None
        q = choices[int(rng.integers(len(choices)))]
        target[q] = opaque1
        blockers.add(q)
        if kind == 'repeat':
            later = sorted(v for v in material if v[0] == q[0] and v[1] > q[1] and inside(v))
            if not later:
                return None
            q2 = later[int(rng.integers(len(later)))]
            target[q2] = opaque2
            blockers.add(q2)
        for _ in range(int(rng.integers(0, 4))):
            available = sorted(material - blockers)
            q = available[int(rng.integers(len(available)))]
            target[q] = opaque1 if rng.integers(2) else opaque2
            blockers.add(q)
    body = material
    illuminated = {q for q in body - blockers if inside(q)
                   and not any(b[0] == q[0] and b[1] < q[1] and inside(b) for b in blockers)}
    out = np.full((ymax - ymin + 1, width), bg, dtype=np.int8)
    for y, x in body:
        out[y - ymin, x - 1] = payload if (y, x) in illuminated else target[y, x]
    grid = _stage(rng, emitter, target, bg)
    if grid is None:
        return None
    pair = {'input': np.rot90(grid, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
    try:
        p = _decode(pair['input'])
    except ValueError:
        return None
    if not np.array_equal(_render(p), pair['output']):
        raise AssertionError("construction disagrees with decoded exposure")
    lit, dark, effective, inert = _exposure(p)
    unlit = {q for q in p['body'] if p['a'][q] == p['material']} - lit
    if not lit or not unlit or np.array_equal(_render(p, registration=False), pair['output']):
        return None
    if kind != 'clean' and not dark:
        return None
    if kind == 'teach' and (len({int(p['a'][q]) for q in effective}) < 2
                           or np.array_equal(_render(p, global_blockers=True), pair['output'])
                           or np.array_equal(_wrong_seam(p, -1), pair['output'])
                           or np.array_equal(_wrong_seam(p, 1), pair['output'])):
        return None
    if support_witness and (not _exposable_gaps(p)
                            or np.array_equal(_render(p, clip=True), pair['output'])):
        return None
    return pair


def _integer(value, name, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"illegal {name}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.random() < .35)
    shared_direction = bool(rng.random() < .35)
    palette = rng.permutation(10)
    direction = int(rng.integers(4))
    shift = str(rng.choice(['depth', 'bar_interpolation', 'occlusion_recombination']))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            colors = palette if stable_palette else rng.permutation(10)
            rotation = direction if shared_direction else int(rng.integers(4))
            if shift == 'bar_interpolation':
                length = (1 if i % 2 == 0 else 3) if split == 'train' else 2
            else:
                length = int(rng.integers(1, 5))
            width = int(rng.integers(5, 8) if split == 'train' and shift == 'depth'
                        else rng.integers(8, 12) if split == 'test' and shift == 'depth'
                        else rng.integers(5, 12))
            kind = ('clean' if split == 'train' and i == 0 else
                    'teach' if split == 'train' and i == 1 else
                    'repeat' if split == 'test' and shift == 'occlusion_recombination' else
                    str(rng.choice(['clean', 'blocked', 'blocked', 'repeat'])))
            for _ in range(160):
                pair = _build_pair(rng, colors, rotation, length, width, kind,
                                   support_witness=(split == 'train' and i == 0))
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(p):
    lit, dark, effective, inert = _exposure(p)
    t, b, l, r = _bbox(p['body'])
    mask = np.zeros((b - t + 1, r - l + 1), dtype=np.int8)
    for y, x in p['body']:
        mask[y - t, x - l] = 1
    boxes = [_bbox(p['emitter']), _bbox(p['plate'])]
    a, z = boxes
    xs = a[3] < z[2] or z[3] < a[2]
    ys = a[1] < z[0] or z[1] < a[0]
    stage = 'diagonal' if xs and ys else 'horizontal' if xs else 'vertical' if ys else 'interleaved'
    return dict(direction=p['rotation'], length=p['length'], width=r - l + 1,
                height=b - t + 1, mask=_key(mask), stage=stage,
                periods=tuple(f[0] for f in p['fits']),
                phases=tuple(f[1] for f in p['fits']),
                palette=(p['bg'], p['material'], p['marker'], p['payload']),
                lit=len(lit), shadow=len(dark), effective=len(effective), inert=len(inert),
                blocker_colors=len({int(p['a'][q]) for q in effective}),
                repeated=len({q[0] for q in effective}) < len(effective),
                holes=int(mask.size - mask.sum()),
                exposable_gaps=len(_exposable_gaps(p)),
                widening_matters=not np.array_equal(_render(p), _render(p, widening=False)),
                global_blockers_wrong=not np.array_equal(_render(p), _render(p, global_blockers=True)),
                overlap_wrong=not np.array_equal(_render(p), _wrong_seam(p, -1)),
                gap_wrong=not np.array_equal(_render(p), _wrong_seam(p, 1)),
                virtual_outside=any(not (0 <= y + p['translation'][0] < p['a'].shape[0]
                                        and 0 <= x + p['translation'][1] < p['a'].shape[1])
                                    for y, x in lit))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, fs = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            p = _decode(inp)
            assert np.array_equal(_render(p), out), 'incorrect registered exposure'
            assert not np.any(out == p['marker']), 'registration ink must disappear'
            canonical_out = np.rot90(out, -p['rotation'])
            t, b, l, r = _bbox(p['body'])
            assert canonical_out.shape == (b - t + 1, r - l + 1)
            assert all(np.any(canonical_out[edge] != p['bg'])
                       for edge in ((0, slice(None)), (-1, slice(None)),
                                    (slice(None), 0), (slice(None), -1)))
            lit, dark, effective, inert = _exposure(p)
            assert lit
            assert any(p['a'][q] == p['material'] and q not in lit for q in p['body'])
            # Count and preserve the entire physical support, including opaque
            # inclusions and disconnected exposed islands; no beam paints gaps.
            assert np.count_nonzero(out != p['bg']) == len(p['body'])
            for y, x in p['body'] - lit:
                assert canonical_out[y - t, x - l] == p['a'][y, x]
            assert not np.array_equal(_render(p, registration=False), out), 'A must affect the answer'
            assert not np.array_equal(_render(p, all_material=True), out), 'B must affect the answer'
            fs[split].append(_features(p))
    assert any(f['effective'] == f['inert'] == 0 for f in fs['train'])
    assert any(f['shadow'] > 0 and f['blocker_colors'] >= 2
               and f['global_blockers_wrong'] and f['overlap_wrong'] and f['gap_wrong']
               for f in fs['train'])
    assert any(f['widening_matters'] for f in fs['train'])
    assert any(f['effective'] == f['inert'] == 0 and f['exposable_gaps'] > 0
               and f['virtual_outside'] for f in fs['train'])


def self_test() -> dict:
    """Bounded 42-seed survey, semantic alternatives, mutation and count checks."""
    # Hand-written pixel regression, with an explicit answer independent of
    # sampling, suffix-fitting and the analytic construction ray formula.
    manual = np.zeros((18, 25), dtype=np.int8)
    manual[5:10, 2] = 2
    for x, y in enumerate((5, 5, 5, 4, 3, 2), 3):
        manual[y, x] = manual[14 - y, x] = 2
    manual[7, 3] = 4
    manual[0:2, 8] = 8
    manual[3:17, 17:23] = 2
    manual[4:6, 16] = 8
    manual[3, 17] = 3       # outside now; later cells in this lane light
    manual[7, 19] = 3       # first effective blocker
    manual[7, 21] = 6       # repeated blocker, same shadow width
    manual[9, 18] = 6       # another opaque color and lane
    manual[11, 20] = 0      # support hole, never filled
    answer_rows = ('324444', '244444', '444444', '444444', '443262',
                   '444444', '462222', '444444', '444044', '444444',
                   '444444', '444444', '444444', '444444')
    answer = np.array([[int(c) for c in row] for row in answer_rows], dtype=np.int8)
    for k in range(4):
        assert np.array_equal(solve(np.rot90(manual, k)), np.rot90(answer, k))
    # Restage the same plate on the other side: relative registration,
    # rather than physical screen position, must determine its print.
    restaged = np.zeros((18, 25), dtype=np.int8)
    restaged[:, 12:19] = manual[:, 2:9]
    restaged[3:17, 1:8] = manual[3:17, 16:23]
    assert np.array_equal(solve(restaged), answer)
    all_features, signatures, shapes, masks, palettes = [], set(), set(), set(), set()
    alternatives = {name: False for name in ('registration', 'widening', 'shadows', 'all_material',
                                            'global_blockers', 'clip', 'paint_gaps',
                                            'seam_overlap', 'seam_gap')}
    for seed in range(42):
        nq = 1 + seed % 4
        nt = nq + 1 + (seed // 4) % (10 - nq)
        puzzle = generate_puzzle(seed, nt, nq)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(pair['input']) for s in ('train', 'test') for pair in puzzle[s])
        signatures.add(signature)
        if seed in (0, 13, 41):
            again = generate_puzzle(seed, nt, nq)
            assert signature == tuple(_key(pair['input']) for s in ('train', 'test') for pair in again[s])
            assert all(np.array_equal(x['output'], y['output']) for s in ('train', 'test')
                       for x, y in zip(puzzle[s], again[s]))
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                p = _decode(pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                f = _features(p)
                all_features.append(f)
                shapes.add(pair['input'].shape)
                masks.add(f['mask'])
                palettes.add(f['palette'])
                for name in alternatives:
                    if name in ('seam_overlap', 'seam_gap'):
                        alternative = _wrong_seam(p, -1 if name == 'seam_overlap' else 1)
                    else:
                        opts = {name: name in ('all_material', 'global_blockers', 'clip', 'paint_gaps')}
                        alternative = _render(p, **opts)
                    alternatives[name] |= not np.array_equal(alternative, pair['output'])
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(masks) >= 100 and len(palettes) >= 30
    assert {f['direction'] for f in all_features} == set(range(4))
    assert {f['length'] for f in all_features} == {1, 2, 3, 4}
    assert {n for f in all_features for n in f['periods']} == {1, 2, 3}
    assert {'horizontal', 'vertical', 'diagonal'} <= {f['stage'] for f in all_features}
    assert any(f['periods'][0] != f['periods'][1] for f in all_features)
    assert any(f['repeated'] for f in all_features)
    assert any(f['holes'] >= 10 for f in all_features)
    assert any(f['virtual_outside'] for f in all_features)
    assert all(alternatives.values()), alternatives
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(917, nt, nq)
            assert (len(p['train']), len(p['test'])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(151, 2, 1)
    query = broken['test'][0]
    p = _decode(query['input'])
    q = next(iter(_exposure(p)[0]))
    t, b, l, r = _bbox(p['body'])
    bad = np.rot90(query['output'], -p['rotation']).copy()
    bad[q[0] - t, q[1] - l] = p['material']
    query['output'] = np.rot90(bad, p['rotation']).copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('a missed exposed pixel was accepted')
    return {'seeds': 42, 'pairs': len(all_features), 'input_shapes': len(shapes),
            'stencil_masks': len(masks), 'palettes': len(palettes),
            'staging': sorted({f['stage'] for f in all_features}),
            'semantic_alternatives_distinguished': alternatives,
            'independent_semantic_approval': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
