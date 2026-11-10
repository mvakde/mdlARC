"""Compass-oriented registered stamps combine by odd/even coverage.

Only NumPy and the standard library are used. Decoding and solving depend on
the current input alone. Construction also computes answers from its sampled
records, and checks that the visible decoder recovers those records.
"""
from collections import Counter
import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
NEAR = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid grid color")
    return a.astype(int, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _choose(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _tf(p, t):
    """D4 about the registration origin, without renormalizing offsets."""
    r, c = p
    if t >= 4:
        c = -c
    for _ in range(t % 4):
        r, c = c, -r
    return int(r), int(c)


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in NEAR:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _bounds(points):
    return (min(r for r, c in points), max(r for r, c in points),
            min(c for r, c in points), max(c for r, c in points))


def _geometry(points):
    signatures = []
    for t in range(8):
        transformed = [_tf(p, t) for p in points]
        r0, _, c0, _ = _bounds(transformed)
        signatures.append(tuple(sorted((r - r0, c - c0) for r, c in transformed)))
    return min(signatures)


def _inside(p, frame):
    top, bottom, left, right = frame['box']
    return top < p[0] < bottom and left < p[1] < right


def _room(p, frames):
    found = [i for i, f in enumerate(frames) if _inside(p, f)]
    _require(len(found) == 1, "content must be in exactly one room")
    return found[0]


def _frames(a, bg):
    """Read long, four-colored rectangular borders; compass tips are short."""
    runs = []
    for r, row in enumerate(a):
        start = 0
        for end in range(1, a.shape[1] + 1):
            if end == a.shape[1] or row[end] != row[start]:
                if end - start >= 9 and row[start] != bg:
                    runs.append((r, start, end - 1, int(row[start])))
                start = end
    frames = []
    for top, left, right, north in runs:
        for bottom, l2, r2, south in runs:
            if (left, right) != (l2, r2) or bottom - top < 8:
                continue
            west, east = int(a[top + 1, left]), int(a[top + 1, right])
            colors = (north, east, south, west)
            if len(set(colors)) != 4 or bg in colors:
                continue
            if not (np.all(a[top + 1:bottom, left] == west)
                    and np.all(a[top + 1:bottom, right] == east)):
                continue
            frame = {'box': (top, bottom, left, right),
                     'sides': dict(zip(DIRS, colors))}
            if frame not in frames:
                frames.append(frame)
    _require(1 <= len(frames) <= 4, "missing or ambiguous room borders")
    wall = np.zeros(a.shape, dtype=bool)
    occupied_boxes = set()
    for f in frames:
        t, b, l, r = f['box']
        box = {(i, j) for i in range(t, b + 1) for j in range(l, r + 1)}
        _require(not (box & occupied_boxes), "room rectangles overlap")
        occupied_boxes |= box
        wall[t, l:r + 1] = wall[b, l:r + 1] = True
        wall[t + 1:b, l] = wall[t + 1:b, r] = True
    colors = set(frames[0]['sides'].values())
    _require(all(set(f['sides'].values()) == colors for f in frames), "room wall palettes differ")
    # All compasses must be dihedral versions of a common four-wall order.
    base = frames[0]['sides']
    _require(all(any(all(f['sides'].get(_tf(d, t)) == color for d, color in base.items())
                        for t in range(8)) for f in frames), "incompatible wall order")
    return frames, wall, colors


def _tips(a, component, band_points):
    r0, r1, c0, c1 = _bounds(component)
    found = {}
    for p in band_points:
        r, c = p
        direction = None
        if r == r0 - 1 and c0 <= c <= c1 and (r + 1, c) in component:
            direction = DIRS[0]
        elif c == c1 + 1 and r0 <= r <= r1 and (r, c - 1) in component:
            direction = DIRS[1]
        elif r == r1 + 1 and c0 <= c <= c1 and (r - 1, c) in component:
            direction = DIRS[2]
        elif c == c0 - 1 and r0 <= r <= r1 and (r, c + 1) in component:
            direction = DIRS[3]
        if direction is not None:
            found[p] = (direction, int(a[p]))
    _require(len(found) == 2, "a mold needs two compass tips")
    (d1, c1), (d2, c2) = found.values()
    _require(c1 != c2 and d1[0] * d2[0] + d1[1] * d2[1] == 0,
             "compass tips must be distinct and perpendicular")
    return found


def _parse_color(a, bg, neutral, frames, wall, bands, require_anchors=True):
    components = _components(map(tuple, np.argwhere((a == neutral) & ~wall)))
    _require(1 <= len(components) <= 2, "invalid mold count")
    band_points = set(map(tuple, np.argwhere(np.isin(a, list(bands)) & ~wall)))
    label_points = [tuple(p) for p in np.argwhere(~np.isin(a, [bg, neutral] + list(bands)))]
    templates, used_tips, used_labels = [], set(), set()
    for component in components:
        _require(3 <= len(component) <= 26, "invalid mask size")
        r0, r1, c0, c1 = _bounds(component)
        _require(max(r1 - r0, c1 - c0) <= 5, "mask is too large")
        tips = _tips(a, component, band_points)
        candidates = []
        near = {(r + dr, c + dc) for r, c in component
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        for s in label_points:
            for dr, dc in DIRS:
                target_p = s[0] + dr, s[1] + dc
                q = s[0] + 2 * dr, s[1] + 2 * dc
                if q not in near or not (0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]):
                    continue
                target = int(a[target_p])
                if target in {bg, neutral, int(a[s])} | bands or int(a[q]) not in (bg, neutral):
                    continue
                if wall[s] or wall[target_p] or wall[q]:
                    continue
                candidates.append((s, target_p, q, (dr, dc)))
        _require(len(candidates) == 1, "registration ray is missing or ambiguous")
        s, target_p, q, direction = candidates[0]
        cells = component | set(tips) | {s, target_p}
        room_ids = {_room(p, frames) for p in cells | {q}}
        _require(len(room_ids) == 1, "an instruction crosses a wall")
        _require(not (set(tips) & used_tips) and not ({s, target_p} & used_labels), "shared instruction labels")
        used_tips |= set(tips)
        used_labels |= {s, target_p}
        transforms = []
        for frame in frames:
            options = [t for t in range(8)
                       if all(frame['sides'][_tf(d, t)] == color for d, color in tips.values())]
            _require(len(options) == 1, "compass does not specify a unique handed pose")
            transforms.append(options[0])
        templates.append({'mask': component, 'q': q, 'direction': direction,
                          'source': int(a[s]), 'target': int(a[target_p]),
                          'labels': {s, target_p}, 'tips': tips, 'cells': cells,
                          'offsets': tuple(sorted((r - q[0], c - q[1]) for r, c in component)),
                          'transforms': transforms})
    _require(used_tips == band_points, "unassigned interior wall-colored pixels")
    sources = {rec['source'] for rec in templates}
    _require(len(sources) == len(templates), "a source has two local instructions")
    instruction_cells = set().union(*(t['cells'] for t in templates))
    for rec in templates:
        rec['anchors'] = sorted(tuple(p) for p in np.argwhere(a == rec['source'])
                                if tuple(p) not in instruction_cells)
        _require(bool(rec['anchors']) or not require_anchors, "instruction has no external triggers")
        for p in rec['anchors']:
            _room(p, frames)
    for p in map(tuple, np.argwhere((a != bg) & ~wall)):
        _room(p, frames)
    return templates


def _decode(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background is not a unique mode")
    bg = int(counts.argmax())
    frames, wall, bands = _frames(a, bg)
    candidates = []
    for neutral in sorted(set(map(int, np.unique(a))) - {bg} - bands):
        try:
            templates = _parse_color(a, bg, neutral, frames, wall, bands)
            info = {'background': bg, 'neutral': neutral, 'frames': frames,
                    'wall': wall, 'bands': bands, 'templates': templates}
            _render(a, info)  # also reject incompatible impression colors
            candidates.append(info)
        except ValueError:
            continue
    _require(len(candidates) == 1, "input does not identify one neutral mold role")
    return a, candidates[0]


def _render(a, info, *, compass=True, parity=True, one_copy=False, one_component=False,
            center=False, discard_multiple=False):
    """Correct rule by default; optional ablations are only for evidence checks."""
    out = a.copy()
    bg = info['background']
    erased = set()
    for rec in info['templates']:
        erased |= rec['cells'] | set(rec['anchors'])
    for p in erased:
        out[p] = bg
    hits, paint, complete, clipped, stamps = Counter(), {}, 0, 0, 0
    for rec in info['templates']:
        anchors = rec['anchors']
        if one_copy:
            anchors = anchors[:1]
        if one_component:
            anchors = [min(group) for group in _components(anchors)]
        offsets = rec['offsets']
        if center:
            r0, r1, c0, c1 = _bounds(rec['mask'])
            q = ((r0 + r1) // 2, (c0 + c1) // 2)
            offsets = tuple((r - q[0], c - q[1]) for r, c in rec['mask'])
        for p in anchors:
            room_index = _room(p, info['frames'])
            frame = info['frames'][room_index]
            t = rec['transforms'][room_index] if compass else 0
            footprint = {(p[0] + dr, p[1] + dc) for dr, dc in (_tf(o, t) for o in offsets)}
            visible = {q for q in footprint if _inside(q, frame)}
            complete += len(visible) == len(footprint)
            clipped += len(visible) != len(footprint)
            stamps += 1
            for q in visible:
                _require(q not in paint or paint[q] == rec['target'], "incompatible target-color overlap")
                paint[q] = rec['target']
                hits[q] += 1
    for p, count in hits.items():
        survives = (count == 1 if discard_multiple else count % 2 == 1 if parity else True)
        out[p] = paint[p] if survives else bg
    stats = {'hits': hits, 'paint': paint, 'erased': erased,
             'complete': int(complete), 'clipped': int(clipped), 'stamps': stamps,
             'cancelled': sum(n % 2 == 0 for n in hits.values()),
             'triple': sum(n >= 3 and n % 2 == 1 for n in hits.values())}
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, info = _decode(input_grid)
    return _render(a, info)[0]


def _one_more_pass(out, info):
    """Evidence counterexample: recursively stamp newly painted sources once."""
    result = out.copy()
    hits, paint = Counter(), {}
    for rec in info['templates']:
        # Every surviving active-source pixel was painted anew: all original
        # source pixels, including labels and covered anchors, were erased.
        new = [tuple(p) for p in np.argwhere(out == rec['source'])]
        for p in new:
            result[p] = info['background']
            i = _room(p, info['frames'])
            for dr, dc in (_tf(o, rec['transforms'][i]) for o in rec['offsets']):
                q = p[0] + dr, p[1] + dc
                if _inside(q, info['frames'][i]):
                    _require(q not in paint or paint[q] == rec['target'], 'conflicting recursive alternative')
                    hits[q] += 1
                    paint[q] = rec['target']
    for p, n in hits.items():
        result[p] = paint[p] if n % 2 else info['background']
    return result


def _make_frames(rng, count, large=False):
    minimum = 26 if large else 22 if count > 1 else 16
    h, w = int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31))
    outer = (int(rng.integers(2)), h - 1 - int(rng.integers(2)),
             int(rng.integers(2)), w - 1 - int(rng.integers(2)))

    def split(box, vertical):
        t, b, l, r = box
        lo, hi = (l, r) if vertical else (t, b)
        cut = int(rng.integers(lo + 9, hi - 8 + 1))
        return ((t, b, l, cut - 1), (t, b, cut + 1, r)) if vertical else (
            (t, cut - 1, l, r), (cut + 1, b, l, r))

    if count == 1:
        boxes = [outer]
    elif count == 2:
        boxes = list(split(outer, bool(rng.integers(2))))
    elif count == 3:
        vertical = bool(rng.integers(2))
        boxes = list(split(outer, vertical))
        i = int(rng.integers(2))
        boxes[i:i + 1] = split(boxes[i], not vertical)
    else:
        boxes = [sub for box in split(outer, True) for sub in split(box, False)]
    return (h, w), boxes


def _shape(rng, wanted=None, cap=6):
    if wanted == 'hole':
        width = int(rng.integers(3, min(5, cap) + 1))
        # An open outline with an interior registration hole and both long
        # runs and an orthogonal three-cell run. Three copies can meet.
        return {(r, c) for r in (-1, 1) for c in range(-1, width - 1)} | {
            (0, width - 2)}
    if wanted == 'occupied':
        length = int(rng.integers(3, min(5, cap) + 1))
        points = {(0, c) for c in range(length)} | {(1, length - 1)}
        if rng.random() < .5:
            points |= {(2, length - 1)}
        if rng.random() < .5:
            points |= {(1, length - 2)}
        return points
    family = _choose(rng, ('growth', 'growth', 'outline', 'line', 'branch', 'chevron'))
    h, w = int(rng.integers(2, cap + 1)), int(rng.integers(2, cap + 1))
    if family == 'outline':
        points = {(r, c) for r in range(h) for c in range(w)
                  if r in (0, h - 1) or c in (0, w - 1)}
        if rng.random() < .5:
            points.discard((0, int(rng.integers(w))))
    elif family == 'line':
        points = {(0, c) for c in range(int(rng.integers(3, cap + 1)))}
    elif family == 'branch':
        points = {(r, 0) for r in range(h)} | {(h // 2, c) for c in range(w)}
        if rng.random() < .5:
            points |= {(r, w - 1) for r in range(h // 2, h)}
    elif family == 'chevron':
        radius = int(rng.integers(1, 3))
        points = {(r, radius - abs(r)) for r in range(-radius, radius + 1)}
    else:
        points = {(0, 0)}
        desired = int(rng.integers(3, min(23, h * w) + 1))
        for _ in range(100):
            if len(points) >= desired:
                break
            r, c = _choose(rng, sorted(points))
            dr, dc = _choose(rng, DIRS if rng.random() < .8 else NEAR)
            if 0 <= r + dr < h and 0 <= c + dc < w:
                points.add((r + dr, c + dc))
    # Select a registration next to the actual mask, with a uniquely directed
    # source-target ray on its left. It need not be the center or occupied.
    near = {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    choices = []
    r0, r1, c0, c1 = _bounds(points)
    for qr, qc in sorted(near):
        if not (r0 <= qr <= r1 and c0 <= qc <= c1):
            continue
        if (qr, qc - 1) in points or (qr, qc - 2) in points:
            continue
        if (qr, qc - 3) in near:
            continue
        choices.append((qr, qc))
    if not choices:
        return _shape(rng, 'occupied', cap)
    qr, qc = _choose(rng, choices)
    return {(r - qr, c - qc) for r, c in points}


def _make_template(rng, neutral, source, target, base_sides, wanted=None, cap=6):
    offsets = _shape(rng, wanted, cap)
    r0, r1, c0, c1 = _bounds(offsets)
    selected = int(rng.integers(4))
    labels = {}
    for d in (DIRS[selected], DIRS[(selected + 1) % 4]):
        edge = sorted(p for p in offsets if (p[0] == r0 if d == DIRS[0] else
                       p[1] == c1 if d == DIRS[1] else p[0] == r1 if d == DIRS[2] else p[1] == c0))
        p = _choose(rng, edge)
        q = p[0] + d[0], p[1] + d[1]
        labels[q] = base_sides[d]
    # The labels and all mask cells undergo the same source-pose transform.
    raw_t = int(rng.integers(8))
    cells = {p: neutral for p in offsets}
    cells.update(labels)
    if (0, -1) in cells or (0, -2) in cells:
        return None
    cells[(0, -2)], cells[(0, -1)] = source, target
    return {'cells': {_tf(p, raw_t): v for p, v in cells.items()},
            'offsets': tuple(sorted(_tf(p, raw_t) for p in offsets)),
            'q': (0, 0), 'raw_t': raw_t}


def _place_templates(rng, a, frames, specs, base_sides, neutral, cap):
    blocked, records = set(), []
    for source, target, wanted in specs:
        found = False
        for _ in range(45):
            template = _make_template(rng, neutral, source, target, base_sides, wanted, cap)
            if template is None:
                continue
            points = set(template['cells']) | {(0, 0)}
            rlo, rhi, clo, chi = _bounds(points)
            options = []
            for frame in frames:
                t, b, l, r = frame['box']
                options += [(qr, qc) for qr in range(t + 1 - rlo, b - rhi)
                            for qc in range(l + 1 - clo, r - chi)]
            rng.shuffle(options)
            for q in options:
                placed = {(q[0] + r, q[1] + c): v for (r, c), v in template['cells'].items()}
                reserved = set(placed) | {q}
                if reserved & blocked:
                    continue
                for p, v in placed.items():
                    a[p] = v
                blocked |= {(r + dr, c + dc) for r, c in reserved
                            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                # Expected records are reconstructed below, after all molds
                # have been placed. There are not yet external triggers.
                records.append((q, source, target, template['offsets']))
                found = True
                break
            if found:
                break
        if not found:
            return None
    return records


def _construct(rng, palette, stage, policy, previous_geometries):
    bg, neutral = palette[:2]
    base_sides = dict(zip(DIRS, palette[2:6]))
    if stage in (0, 1):
        room_count, count = 2, 1 if stage == 0 else 2
    else:
        room_count = int(rng.integers(3, 5)) if stage == 'test' and policy == 'rooms' else int(rng.integers(1, 5))
        if policy == 'rooms' and stage != 'test':
            room_count = int(rng.integers(1, 3))
        count = int(rng.integers(1, 3))
    shape, boxes = _make_frames(rng, room_count, large=stage in (0, 1))
    a = np.full(shape, bg, dtype=int)
    frames = []
    poses = [int(rng.integers(8)) for _ in boxes]
    if stage in (0, 1):
        poses[1] = (poses[0] + (4 if stage == 0 else 1)) % 8
    for box, pose in zip(boxes, poses):
        sides = {_tf(d, pose): v for d, v in base_sides.items()}
        t, b, l, r = box
        a[t, l:r + 1], a[b, l:r + 1] = sides[DIRS[0]], sides[DIRS[2]]
        a[t + 1:b, l], a[t + 1:b, r] = sides[DIRS[3]], sides[DIRS[1]]
        frames.append({'box': box, 'sides': sides})
    s1, s2, t1, t2 = palette[6:10]
    if stage == 0:
        specs = [(s1, t1, 'occupied')]
    elif stage == 1:
        specs = [(s1, s2, 'hole'), (s2, t1, None)]
    elif count == 1:
        specs = [(s1, _choose(rng, (t1, t2, s2)), None)]
    else:
        mode = _choose(rng, ('independent', 'shared', 'chain', 'reciprocal'))
        targets = {'independent': (t1, t2), 'shared': (t1, t1),
                   'chain': (s2, t1), 'reciprocal': (s2, s1)}[mode]
        specs = [(s1, targets[0], None), (s2, targets[1], None)]
    cap = 5 if stage in (0, 1) else 6
    expected = _place_templates(rng, a, frames, specs, base_sides, neutral, cap)
    if expected is None:
        return None
    # Decode the visible instructions before adding triggers. Completed inputs
    # must pass the stricter public decoder with at least one trigger per mold.
    try:
        wall = np.zeros(shape, dtype=bool)
        for f in frames:
            t, b, l, r = f['box']
            wall[t, l:r + 1] = wall[b, l:r + 1] = True
            wall[t + 1:b, l] = wall[t + 1:b, r] = True
        templates = _parse_color(a, bg, neutral, frames, wall, set(base_sides.values()),
                                 require_anchors=False)
    except ValueError:
        return None
    for rec in templates:
        rec['anchors'] = []
    observed = {(rec['q'], rec['source'], rec['target'], rec['offsets']) for rec in templates}
    if observed != set(expected):
        return None
    if stage == 'test' and all(_geometry(rec['offsets']) in previous_geometries for rec in templates):
        return None
    info = {'background': bg, 'neutral': neutral, 'frames': frames,
            'wall': wall, 'bands': set(base_sides.values()), 'templates': templates}
    reserved = set().union(*(rec['cells'] | {rec['q']} for rec in templates))
    anchors, paint = set(), {}

    def footprint(rec, p):
        i = _room(p, frames)
        return {(p[0] + r, p[1] + c) for r, c in (_tf(o, rec['transforms'][i]) for o in rec['offsets'])
                if _inside((p[0] + r, p[1] + c), frames[i])}

    def can_place(rec, positions, complete=False):
        trial_paint = dict(paint)
        for p in positions:
            if p in reserved or p in anchors or not any(_inside(p, f) for f in frames) or a[p] != bg:
                return False
            fp = footprint(rec, p)
            if not fp or complete and len(fp) != len(rec['offsets']):
                return False
            if any(q in trial_paint and trial_paint[q] != rec['target'] for q in fp):
                return False
            trial_paint.update({q: rec['target'] for q in fp})
        return True

    def place(rec, positions):
        for p in positions:
            a[p] = rec['source']
            anchors.add(p)
            rec['anchors'].append(p)
            paint.update({q: rec['target'] for q in footprint(rec, p)})

    all_inside = [(r, c) for f in frames for r in range(f['box'][0] + 1, f['box'][1])
                  for c in range(f['box'][2] + 1, f['box'][3])]
    for i, rec in enumerate(templates):
        cells = [all_inside[int(j)] for j in rng.permutation(len(all_inside))]
        cluster_n = 2 if stage == 0 and i == 0 else 3 if stage == 1 and rec['source'] == s1 else 0
        if cluster_n:
            found = False
            steps = list(DIRS)
            rng.shuffle(steps)
            for p in cells:
                for dr, dc in steps:
                    positions = [(p[0] + k * dr, p[1] + k * dc) for k in range(cluster_n)]
                    if not can_place(rec, positions, complete=True):
                        continue
                    hits = Counter(q for q0 in positions for q in footprint(rec, q0))
                    if not any(n % 2 == 0 for n in hits.values()):
                        continue
                    if cluster_n == 3 and not any(n == 3 for n in hits.values()):
                        continue
                    place(rec, positions)
                    found = True
                    break
                if found:
                    break
            if not found:
                return None
        desired = (int(rng.integers(7, 9)) if stage == 'test' and policy == 'multiplicity'
                   else int(rng.integers(1, 7)))
        desired = max(desired, cluster_n + 1 if cluster_n else 1)
        # Both teaching examples cast their first mold in both compasses;
        # example 0 reverses handedness between the two rooms.
        if stage in (0, 1) and rec['source'] == s1:
            missing = [j for j in range(len(frames)) if not any(_inside(p, frames[j]) for p in rec['anchors'])]
            for j in missing:
                candidates = [p for p in cells if _inside(p, frames[j])]
                p = next((p for p in candidates if can_place(rec, [p], complete=True)), None)
                if p is None:
                    return None
                place(rec, [p])
        # The chain's second instruction teaches confinement explicitly:
        # at least one of its impressions crosses, and clips at, a room wall.
        if stage == 1 and rec['source'] == s2:
            p = next((p for p in cells if can_place(rec, [p])
                      and len(footprint(rec, p)) < len(rec['offsets'])), None)
            if p is None:
                return None
            place(rec, [p])
        # Independent scatter, adjacent growth, and edge/corner preferences.
        style = _choose(rng, ('scatter', 'cluster', 'edge', 'corner'))
        if style in ('edge', 'corner'):
            def distance(p):
                t, b, l, r = frames[_room(p, frames)]['box']
                ds = sorted((p[0] - t - 1, b - p[0] - 1, p[1] - l - 1, r - p[1] - 1))
                return ds[0] if style == 'edge' else ds[0] + ds[1]
            cells.sort(key=distance)
        for _ in range(desired - len(rec['anchors'])):
            adjacent = []
            if style == 'cluster':
                adjacent = [(p[0] + dr, p[1] + dc) for p in rec['anchors'] for dr, dc in DIRS]
                rng.shuffle(adjacent)
            p = next((p for p in adjacent + cells if can_place(rec, [p])), None)
            if p is None:
                break
            place(rec, [p])
        if not rec['anchors']:
            return None
        if stage == 'test' and policy == 'multiplicity' and len(rec['anchors']) < 7:
            return None
    # Inactive, including target-only, pixels are optional and stay clear of
    # all impressions. They are ordinary content, never a palette legend.
    inactive = [c for c in palette[6:] if c not in {rec['source'] for rec in templates}]
    for _ in range(int(rng.integers(4))):
        p = _choose(rng, all_inside)
        if p not in reserved and p not in paint and a[p] == bg:
            a[p] = _choose(rng, inactive)
    try:
        parsed_a, decoded = _decode(a)
        output, stats = _render(a, info)
        predicted, _ = _render(parsed_a, decoded)
        if not np.array_equal(output, predicted):
            return None
        if stage == 0:
            if not stats['cancelled'] or not stats['complete']:
                return None
            if np.array_equal(output, _render(a, decoded, compass=False)[0]):
                return None
            if np.array_equal(output, _render(a, decoded, one_component=True)[0]):
                return None
        if stage == 1:
            if not stats['triple'] or not stats['cancelled']:
                return None
            if np.array_equal(output, _render(a, decoded, center=True)[0]):
                return None
            first = next(t for t in decoded['templates'] if t['source'] == s1)
            if not any(output[p] == s2 and a[p] != s2 for p in stats['hits']
                       if stats['paint'][p] == first['target']):
                return None
            if np.array_equal(output, _one_more_pass(output, decoded)):
                return None
        return {'input': a, 'output': output}
    except ValueError:
        return None


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(name + ' out of range')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    policy = _choose(rng, ('rooms', 'multiplicity', 'geometry'))
    first_sources = [int(v) for v in rng.choice(10, 2, replace=False)]
    puzzle, seen, geometries = {'train': [], 'test': []}, set(), set()
    previous_bg = None
    for split, n in (('train', num_train), ('test', num_test)):
        for i in range(n):
            stage = i if split == 'train' else 'test'
            palette = [int(c) for c in rng.permutation(10)]
            if stage in (0, 1):
                pool = [c for c in palette if c not in first_sources]
                palette = pool[:6] + first_sources + pool[6:]
            if palette[0] == previous_bg:
                palette[0], palette[1] = palette[1], palette[0]
            previous_bg = palette[0]
            for _ in range(160):
                pair = _construct(rng, palette, stage, policy, geometries)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded construction exhausted for ' + str(stage))
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                _, info = _decode(pair['input'])
                geometries.update(_geometry(rec['offsets']) for rec in info['templates'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'invalid pair counts')
    seen, records = set(), []
    train_geometries = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, out = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == out.shape, 'grid shape changed')
            _require(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            _, info = _decode(a)
            predicted, stats = _render(a, info)
            _require(np.array_equal(out, predicted), 'output violates compass/parity stamping')
            _require(not np.array_equal(a, out), 'unchanged example')
            _require(np.array_equal(a[info['wall']], out[info['wall']]), 'wall changed')
            _require(stats['stamps'] == sum(len(rec['anchors']) for rec in info['templates']), 'trigger count changed')
            for p, n in stats['hits'].items():
                _require(out[p] == (stats['paint'][p] if n % 2 else info['background']), 'wrong interference')
            changed = stats['erased'] | set(stats['hits'])
            keep = np.ones(a.shape, dtype=bool)
            for p in changed:
                keep[p] = False
            _require(np.array_equal(a[keep], out[keep]), 'unrelated content changed')
            geometry = {_geometry(rec['offsets']) for rec in info['templates']}
            if split == 'train':
                train_geometries |= geometry
                records.append((a, out, info, stats))
            else:
                _require(bool(geometry - train_geometries), 'test lacks a new visible mask geometry')
    first_a, first_out, first, first_stats = records[0]
    second_a, second_out, second, second_stats = records[1]
    _require(len(first['frames']) >= 2 and first_stats['complete'] >= 3, 'missing local-compass evidence')
    first_mold = first['templates'][0]
    used_rooms = {_room(p, first['frames']) for p in first_mold['anchors']}
    _require(len(used_rooms) >= 2 and {first_mold['transforms'][i] >= 4 for i in used_rooms} == {False, True},
             'missing reflection-versus-rotation compass evidence')
    _require(first_stats['cancelled'] > 0, 'missing pair-cancellation evidence')
    for kw in ({'compass': False}, {'parity': False}, {'one_copy': True}, {'one_component': True}):
        try:
            alternative = _render(first_a, first, **kw)[0]
        except ValueError:
            continue  # a conflicting ablation is also inconsistent with the answer
        _require(not np.array_equal(alternative, first_out), 'training admits a parent-omitting shortcut')
    _require(second_stats['triple'] > 0, 'missing odd triple-overlap evidence')
    _require(second_stats['clipped'] > 0, 'missing room-clipping evidence')
    _require(not np.array_equal(_render(second_a, second, discard_multiple=True)[0], second_out),
             'discard-all-overlaps shortcut survives')
    _require(any((0, 0) not in rec['offsets'] for rec in second['templates']), 'missing hole registration')
    _require(not np.array_equal(_render(second_a, second, center=True)[0], second_out), 'centering shortcut survives')
    map1 = {rec['source']: rec['target'] for rec in first['templates']}
    map2 = {rec['source']: rec['target'] for rec in second['templates']}
    _require(any(source in map2 and map2[source] != target for source, target in map1.items()), 'mappings do not change')
    _require(any(rec['target'] in map2 for rec in second['templates']), 'missing simultaneous color chain')
    _require(not np.array_equal(_one_more_pass(second_out, second), second_out), 'recursion shortcut survives')


def self_test():
    """Bounded 42-seed semantic/diversity audit, plus interface regressions."""
    # Hand-specified exterior registration, wall clipping, adjacent triggers,
    # triple interference, and preserved target-only content. Expected paint
    # is explicitly listed rather than manufactured by the renderer.
    fixture = np.zeros((12, 14), dtype=int)
    fixture[0, :], fixture[-1, :] = 1, 3
    fixture[1:-1, 0], fixture[1:-1, -1] = 4, 2
    expected = fixture.copy()
    fixture[4:7, 4] = 8
    fixture[3, 4], fixture[5, 5] = 1, 2
    fixture[5, 1:3] = (7, 6)
    fixture[1:4, 9] = 7
    fixture[8, 8] = expected[8, 8] = 6
    expected[2, 10] = expected[4, 10] = 6
    assert np.array_equal(solve(fixture), expected)
    _, fixture_info = _decode(fixture)
    assert fixture_info['templates'][0]['q'] == (5, 3)
    assert fixture_info['templates'][0]['offsets'] == ((-1, 1), (0, 1), (1, 1))
    puzzles, shapes, geometries, palettes, rooms, mask_sizes = [], set(), set(), set(), set(), set()
    poses, registrations, clipped, cancelled, triple, anchor_counts = set(), set(), 0, 0, 0, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                a, info = _decode(p['input'])
                out, stats = _render(a, info)
                assert np.array_equal(out, p['output'])
                shapes.add(a.shape)
                palettes.add((info['background'], info['neutral'], tuple(sorted(info['bands']))))
                rooms.add(len(info['frames']))
                clipped += stats['clipped'] > 0
                cancelled += stats['cancelled'] > 0
                triple += stats['triple'] > 0
                for rec in info['templates']:
                    geometries.add(_geometry(rec['offsets']))
                    mask_sizes.add(len(rec['offsets']))
                    poses.update(rec['transforms'])
                    registrations.add('occupied' if (0, 0) in rec['offsets'] else 'hole')
                    anchor_counts.add(len(rec['anchors']))
    assert len(set(puzzles)) == 42
    assert len(shapes) >= 30 and len(palettes) >= 100 and len(geometries) >= 35
    assert rooms == {1, 2, 3, 4} and poses == set(range(8))
    assert registrations == {'occupied', 'hole'} and len(mask_sizes) >= 10
    assert min(anchor_counts) == 1 and max(anchor_counts) >= 7
    assert clipped >= 40 and cancelled >= 84 and triple >= 42
    def signature(puzzle):
        return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])
    assert signature(generate_puzzle(91, 2, 1)) == signature(generate_puzzle(91, 2, 1))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(123, train, test)
            assert len(p['train']) == train and len(p['test']) == test
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(7, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong answer accepted')
    return {'seeds': 42, 'distinct_shapes': len(shapes), 'mask_geometries': len(geometries),
            'local_palettes': len(palettes), 'room_counts': sorted(rooms),
            'mask_sizes': sorted(mask_sizes), 'anchor_counts': sorted(anchor_counts),
            'clipped_pairs': clipped, 'cancellation_pairs': cancelled, 'triple_pairs': triple}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
