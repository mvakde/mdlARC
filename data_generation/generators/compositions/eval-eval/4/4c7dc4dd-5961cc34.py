"""Arrival-oriented relay completion. Standard library and NumPy only.

The reference is decoded on every solve call. No generated metadata, stored
examples, parent imports, or seed-dependent solving conventions are used.
"""
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with axes 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _rim(side):
    return [(r, c) for r in range(side) for c in range(side)
            if r in (0, side - 1) or c in (0, side - 1)]


def _seed(n, kind, bg, body, accent):
    a = np.full((n, n), bg, dtype=np.int8)
    m = n // 2
    offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    if kind == 1:
        offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    elif kind == 2:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in offsets:
        a[m + dr, m + dc] = body
    a[m, m] = accent
    return a


def _seed_roles(a, bg):
    n = len(a)
    if a.shape != (n, n) or n not in (3, 5, 7):
        return None
    accent = int(a[n // 2, n // 2])
    colors = set(map(int, np.unique(a))) - {bg, accent}
    if accent == bg or len(colors) != 1:
        return None
    body = next(iter(colors))
    for kind in range(3):
        if np.array_equal(a, _seed(n, kind, bg, body, accent)):
            return body, accent, kind
    return None


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    todo = [min(cells)]
    seen = set(todo)
    while todo:
        r, c = todo.pop()
        for dr, dc in DIRS:
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _port(a, accent):
    n = len(a)
    positions = [(r, c) for r, c in _rim(n) if int(a[r, c]) == accent]
    sides = [d for d in range(4) if positions and all(
        (r == 0 if d == 0 else c == n - 1 if d == 1
         else r == n - 1 if d == 2 else c == 0) for r, c in positions)]
    if len(sides) != 1:
        raise ValueError("completed stencil needs one unambiguous boundary outlet")
    d = sides[0]
    span = sorted(c if d % 2 == 0 else r for r, c in positions)
    if span != list(range(span[0], span[0] + len(span))) or not (0 < span[0] and span[-1] < n - 1):
        raise ValueError("outlet must be contiguous and avoid interior corners")
    return d, positions


def _decode(inp):
    h, w = inp.shape
    bg = int(inp[0, 0])
    if any(int(inp[p]) != bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1))):
        raise ValueError("background corners disagree")
    border = [(r, c) for r, c in _rim_rect(h, w) if int(inp[r, c]) != bg]
    if len(border) != 1:
        raise ValueError("expected exactly one boundary launcher")
    start = border[0]
    direction = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    dr, dc = DIRS[direction]
    signal = int(inp[start])
    tail = []
    r, c = start
    while 0 <= r < h and 0 <= c < w and int(inp[r, c]) == signal:
        tail.append((r, c))
        r, c = r + dr, c + dc
    if not 2 <= len(tail) <= 4 or not (0 <= r < h and 0 <= c < w):
        raise ValueError("launcher tail length must be two through four")
    cap = r, c
    cap_color = int(inp[cap])
    if cap_color in (bg, signal):
        raise ValueError("missing launcher cap")
    candidates = []
    for side in (5, 7, 9):
        rim = _rim(side)
        middles = ((0, side // 2), (side // 2, side - 1),
                   (side - 1, side // 2), (side // 2, 0))
        for y in range(1, h - side):
            for x in range(1, w - side):
                f = int(inp[y, x])
                if f in (bg, signal, cap_color) or any(int(inp[y + rr, x + cc]) != f
                        for rr, cc in ((0, side - 1), (side - 1, 0), (side - 1, side - 1))):
                    continue
                exceptions = [(rr, cc) for rr, cc in rim if int(inp[y + rr, x + cc]) != f]
                marked = None
                if exceptions:
                    if len(exceptions) != 1 or exceptions[0] not in middles:
                        continue
                    rr, cc = exceptions[0]
                    if int(inp[y + rr, x + cc]) != cap_color:
                        continue
                    marked = middles.index((rr, cc))
                candidates.append(dict(r=y, c=x, side=side, frame=f, entry=marked,
                                       interior=inp[y + 1:y + side - 1, x + 1:x + side - 1].copy()))
    marked = [p for p in candidates if p['entry'] is not None]
    if len(marked) != 2 or len({(p['side'], p['frame'], p['entry']) for p in marked}) != 1:
        raise ValueError("reference must have two equal frames with matching inlet marks")
    source_choices = [(p, _seed_roles(p['interior'], bg)) for p in marked]
    source_choices = [(p, roles) for p, roles in source_choices if roles is not None]
    if len(source_choices) != 1:
        raise ValueError("cannot distinguish sparse reference from completed reference")
    sparse, (body, accent, kind) = source_choices[0]
    full = next(p for p in marked if p is not sparse)
    template = full['interior']
    if (set(map(int, np.unique(template))) - {bg} != {body, accent}
            or not np.all((sparse['interior'] == bg) | (sparse['interior'] == template))
            or not _connected(template != bg)):
        raise ValueError("completed stencil must preserve cues and form a connected footprint")
    out_dir, ports = _port(template, accent)
    if out_dir == sparse['entry']:
        raise ValueError("outlet may not reverse into the inlet")
    structural = {bg, signal, cap_color, sparse['frame'], body, accent}
    group_colors = set(map(int, np.unique(inp))) - structural
    if len(structural) != 6 or len(group_colors) != 1:
        raise ValueError("need seven distinct, visibly assigned palette roles")
    group_color = next(iter(group_colors))
    yy, xx = np.where(inp == group_color)
    box = (int(yy.min()), int(xx.min()), int(yy.max()) + 1, int(xx.max()) + 1)
    y0, x0, y1, x1 = box
    expected_group = {(r, c) for r in range(y0, y1) for c in range(x0, x1)
                      if r in (y0, y1 - 1) or c in (x0, x1 - 1)}
    if set(zip(yy.tolist(), xx.tolist())) != expected_group:
        raise ValueError("calibration enclosure is not one rectangular outline")
    if not (0 < y0 < y1 < h and 0 < x0 < x1 < w):
        raise ValueError("calibration enclosure must be interior")
    for p in marked:
        if not (y0 < p['r'] and x0 < p['c'] and p['r'] + p['side'] < y1
                and p['c'] + p['side'] < x1):
            raise ValueError("reference panels must be inside their enclosure")
    panels = [p for p in candidates if p['side'] == sparse['side'] and p['frame'] == sparse['frame']]
    queries = [p for p in panels if p['entry'] is None]
    if not queries or len(panels) != len(queries) + 2:
        raise ValueError("expected single incomplete panels outside the reference")
    for p in queries:
        if not np.array_equal(p['interior'], sparse['interior']):
            raise ValueError("single panels must carry the reference's symmetric cue")
    if any(_contains(box, cell) for cell in tail + [cap]):
        raise ValueError("launcher touches calibration enclosure")
    # Reconstruct the complete allowed input, catching stray or overlapping cells.
    reconstructed = np.full_like(inp, bg)
    for r, c in expected_group:
        reconstructed[r, c] = group_color
    occupied_boxes = []
    for p in panels:
        b = _panel_box(p)
        if any(_boxes_near(b, other, gap=1) for other in occupied_boxes):
            raise ValueError("panels need a background-cell gap")
        occupied_boxes.append(b)
        _draw_panel(reconstructed, p['r'], p['c'], p['interior'], p['frame'], p['entry'], cap_color)
    for p in queries:
        if _boxes_near(_panel_box(p), box, gap=1):
            raise ValueError("incomplete panel touches calibration enclosure")
    for cell in tail:
        reconstructed[cell] = signal
    reconstructed[cap] = cap_color
    if not np.array_equal(reconstructed, inp):
        raise ValueError("unrecognized cells or frame geometry")
    return dict(bg=bg, signal=signal, cap_color=cap_color, tail=tail, cap=cap,
                initial=direction, reference_entry=sparse['entry'], template=template,
                body=body, accent=accent, kind=kind, queries=queries, group=box,
                full=full, sparse=sparse, width=len(ports), turn=(out_dir - (sparse['entry'] + 2)) % 4)


def _rim_rect(h, w):
    return [(r, c) for r in range(h) for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)]


def _panel_box(p):
    return p['r'], p['c'], p['r'] + p['side'], p['c'] + p['side']


def _contains(box, cell):
    return box[0] <= cell[0] < box[2] and box[1] <= cell[1] < box[3]


def _boxes_near(a, b, gap=1):
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0]
                or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def _oriented(template, reference_entry, incoming, accent):
    k = ((incoming + 2) - reference_entry) % 4
    a = np.rot90(template, -k).copy()
    out_dir, positions = _port(a, accent)
    return a, out_dir, positions


def _frame_ports(p, local_ports, direction):
    result = []
    for r, c in local_ports:
        rr, cc = p['r'] + r + 1, p['c'] + c + 1
        dr, dc = DIRS[direction]
        result.append((rr + dr, cc + dc))
    return sorted(result)


def _trace(inp, scene):
    """Pixel-by-pixel traversal of the original frame envelopes."""
    h, w = inp.shape
    out = np.full_like(inp, scene['bg'])
    trace = set(scene['tail']) | {scene['cap']}
    front = [scene['cap']]
    incoming = scene['initial']
    reached, entries, gaps, segments = [], [], [], []
    boxes = [_panel_box(p) for p in scene['queries']]
    for _ in range(len(boxes) + 1):
        dr, dc = DIRS[incoming]
        segment = set()
        while True:
            front = [(r + dr, c + dc) for r, c in front]
            inside = [0 <= r < h and 0 <= c < w for r, c in front]
            if not any(inside):
                trace |= segment
                segments.append(segment)
                for cell in trace:
                    out[cell] = scene['signal']
                return out, dict(reached=reached, entries=entries, gaps=gaps, segments=segments, trace=trace)
            if not all(inside):
                raise ValueError("partial boundary exit")
            if any(_contains(scene['group'], cell) for cell in front):
                raise ValueError("signal intersects the calibration enclosure")
            lane_hits = [next((i for i, b in enumerate(boxes) if _contains(b, cell)), None) for cell in front]
            hit_set = set(lane_hits)
            if hit_set == {None}:
                segment.update(front)
                continue
            if len(hit_set) != 1 or None in hit_set:
                raise ValueError("partial or simultaneous multi-panel collision")
            index = lane_hits[0]
            if index in reached:
                raise ValueError("cycle or reactivation")
            reached.append(index)
            entries.append(incoming)
            gaps.append(len(segment) // len(front))
            trace |= segment
            segments.append(segment)
            p = scene['queries'][index]
            a, outgoing, local_ports = _oriented(scene['template'], scene['reference_entry'], incoming, scene['accent'])
            trace.update((p['r'] + r, p['c'] + c) for r, c in _rim(p['side']))
            trace.update((p['r'] + 1 + int(r), p['c'] + 1 + int(c)) for r, c in np.argwhere(a != scene['bg']))
            front = _frame_ports(p, local_ports, outgoing)
            incoming = outgoing
            break
    raise ValueError("unterminated route")


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    return _trace(inp, _decode(inp))[0]


def _draw_panel(grid, r, c, interior, frame, entry=None, cap_color=None):
    side = len(interior) + 2
    for rr, cc in _rim(side):
        grid[r + rr, c + cc] = frame
    grid[r + 1:r + side - 1, c + 1:c + side - 1] = interior
    if entry is not None:
        points = ((0, side // 2), (side // 2, side - 1), (side - 1, side // 2), (side // 2, 0))
        rr, cc = points[entry]
        grid[r + rr, c + cc] = cap_color


def _blueprint(rng, cue, bg, body, accent, width, turn, mask=None, start=None):
    """Canonical reference inlet is at the top; travel is down."""
    n = len(cue)
    m = n // 2
    if start is None:
        start = int(rng.integers(1, n - width))
    if mask is None:
        mask = cue != bg
        # Connect all sparse cues, then provide paths to both possible side ports.
        for r, c in np.argwhere(mask.copy()):
            mask[min(r, m):max(r, m) + 1, c] = True
            mask[m, min(c, m):max(c, m) + 1] = True
        for r in range(start, start + width):
            mask[r, :] = True
            mask[min(r, m):max(r, m) + 1, m] = True
        minimum = min(n * n, int(np.count_nonzero(cue != bg)) + 2)
        goal = int(rng.integers(max(minimum, int(mask.sum())), n * n + 1))
        while mask.sum() < goal:
            candidates = sorted({(r + dr, c + dc) for r, c in np.argwhere(mask)
                                 for dr, dc in DIRS if 0 <= r + dr < n and 0 <= c + dc < n
                                 and not mask[r + dr, c + dc]})
            mask[candidates[int(rng.integers(len(candidates)))]] = True
    else:
        mask = mask.copy()
    side = (2 + turn) % 4
    if side == 2:
        for c in range(start, start + width):
            mask[:, c] = True
    a = np.where(mask, body, bg).astype(np.int8)
    a[n // 2, n // 2] = accent
    if side == 1:
        a[start:start + width, -1] = accent
    elif side == 3:
        a[start:start + width, 0] = accent
    else:
        a[-1, start:start + width] = accent
    return a, mask, start


def _reference(rng, n, colors, cue, full, h, w, compact=False):
    bg, signal, cap, frame, group, body, accent = colors
    s = n + 2
    gap = int(rng.integers(1, 4)) if not compact else 1
    padding = int(rng.integers(1, 3)) if not compact else 1
    gh = s + 2 + (0 if compact else int(rng.integers(0, 2)))
    gw = 2 * s + gap + 2 * padding
    if gw > w - 2:
        gap, padding = 1, 1
        gw = 2 * s + 3
    r = 1
    c = int(rng.integers(1, w - gw))
    box = r, c, r + gh, c + gw
    entry = int(rng.integers(4))
    k = entry
    interiors = [np.rot90(cue, -k).copy(), np.rot90(full, -k).copy()]
    if rng.integers(2):
        interiors.reverse()
    locations = [(r + 1, c + padding), (r + gh - s - 1, c + padding + s + gap)]
    return box, entry, interiors, locations


def _assemble(h, w, colors, ref, cue, queries, tail, cap):
    bg, signal, cap_color, frame, group, body, accent = colors
    inp = np.full((h, w), bg, dtype=np.int8)
    box, entry, interiors, locations = ref
    y0, x0, y1, x1 = box
    inp[y0:y1, x0] = inp[y0:y1, x1 - 1] = group
    inp[y0, x0:x1] = inp[y1 - 1, x0:x1] = group
    for interior, (r, c) in zip(interiors, locations):
        _draw_panel(inp, r, c, interior, frame, entry, cap_color)
    for p in queries:
        _draw_panel(inp, p['r'], p['c'], cue, frame)
    for cell in tail:
        inp[cell] = signal
    inp[cap] = cap_color
    return inp


def _training_twins(rng, colors):
    n = int(rng.choice([3, 5]))
    s = n + 2
    bg, signal, cap_color, frame, group, body, accent = colors
    kind = 1 if n == 3 else int(rng.integers(3))
    cue = _seed(n, kind, bg, body, accent)
    width = 1 if n == 3 else int(rng.integers(2, n - 1))
    _, mask, start = _blueprint(rng, cue, bg, body, accent, width, 1)
    full1, _, _ = _blueprint(rng, cue, bg, body, accent, 1, 1, mask, start)
    full2, _, _ = _blueprint(rng, cue, bg, body, accent, width, -1, mask, start)
    h = int(rng.integers(27 if n == 5 else 23, 31))
    gap1, gap2 = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    min_w = 3 * s + gap1 + gap2 + 2
    w = int(rng.integers(min_w, 31))
    ref = _reference(rng, n, colors, cue, full1, h, w)
    low = ref[0][2] + 1
    high = h - s - 5
    r = int(rng.integers(low, max(low, high) + 1))
    first_c = int(rng.integers(1, w - (3 * s + gap1 + gap2)))
    cs = [first_c, first_c + s + gap1, first_c + 2 * s + gap1 + gap2]
    queries = [dict(r=r, c=c, side=s) for c in cs]
    center_c = cs[1] + int(rng.integers(1, s - 1))
    tail_length = int(rng.integers(2, min(5, h - (r + s) - 1)))
    tail = [(h - 1 - i, center_c) for i in range(tail_length)]
    cap = h - 1 - tail_length, center_c
    inp1 = _assemble(h, w, colors, ref, cue, queries, tail, cap)
    box, entry, interiors, locations = ref
    full_slot = next(i for i, a in enumerate(interiors) if _seed_roles(a, bg) is None)
    interiors2 = [a.copy() for a in interiors]
    interiors2[full_slot] = np.rot90(full2, -entry).copy()
    inp2 = _assemble(h, w, colors, (box, entry, interiors2, locations), cue, queries, tail, cap)
    rotation = int(rng.integers(4))
    pairs = []
    for inp in (inp1, inp2):
        inp = np.rot90(inp, rotation).copy()
        out = solve(inp)
        pairs.append({'input': inp, 'output': out})
    return pairs


def _launcher(rng, h, w):
    direction = int(rng.integers(4))
    length = int(rng.integers(2, 5))
    line = int(rng.integers(2, (w if direction % 2 == 0 else h) - 2))
    start = (h - 1, line) if direction == 0 else (line, 0) if direction == 1 else (0, line) if direction == 2 else (line, w - 1)
    dr, dc = DIRS[direction]
    tail = [(start[0] + i * dr, start[1] + i * dc) for i in range(length)]
    cap = start[0] + length * dr, start[1] + length * dc
    return tail, cap, direction


def _empty_segment(h, w, origins, direction, boxes, group):
    dr, dc = DIRS[direction]
    cells = set()
    front = origins
    while True:
        front = [(r + dr, c + dc) for r, c in front]
        if all(not (0 <= r < h and 0 <= c < w) for r, c in front):
            return cells
        if any(not (0 <= r < h and 0 <= c < w) or _contains(group, (r, c))
               or any(_contains(b, (r, c)) for b in boxes) for r, c in front):
            return None
        cells.update(front)


def _random_route(rng, h, w, ref, cue, full, colors, count):
    s = len(cue) + 2
    tail, cap, direction = _launcher(rng, h, w)
    if any(_contains(ref[0], p) for p in tail + [cap]):
        return None
    bg, signal, cap_color, frame, group, body, accent = colors
    budget = [240]

    def grow(queries, trace, origins, incoming):
        boxes = [_panel_box(p) for p in queries]
        if len(queries) == count:
            final = _empty_segment(h, w, origins, incoming, boxes, ref[0])
            return None if final is None else (queries, trace | final)
        width = len(origins)
        dr, dc = DIRS[incoming]
        options = []
        for distance in range(2, 25):
            for offset in range(1, s - width):
                r, c = min(origins)
                hit_r, hit_c = r + distance * dr, c + distance * dc
                if incoming == 0:
                    y, x = hit_r - s + 1, hit_c - offset
                elif incoming == 2:
                    y, x = hit_r, hit_c - offset
                elif incoming == 1:
                    y, x = hit_r - offset, hit_c
                else:
                    y, x = hit_r - offset, hit_c - s + 1
                b = y, x, y + s, x + s
                if not (1 <= y and 1 <= x and y + s < h and x + s < w):
                    continue
                if _boxes_near(b, ref[0]) or any(_boxes_near(b, old) for old in boxes):
                    continue
                if any(_contains(b, cell) for cell in trace):
                    continue
                segment = {(r + k * dr, c + k * dc) for r, c in origins for k in range(1, distance)}
                if any(_contains(ref[0], cell) or any(_contains(old, cell) for old in boxes)
                       or cell in tail + [cap] for cell in segment):
                    continue
                options.append((y, x, segment))
        rng.shuffle(options)
        for y, x, segment in options[:60]:
            if budget[0] <= 0:
                break
            budget[0] -= 1
            p = dict(r=y, c=x, side=s)
            # Full is canonical here, with its inlet at the top.
            a, outgoing, local_ports = _oriented(full, 0, incoming, accent)
            ports = _frame_ports(p, local_ports, outgoing)
            result = grow(queries + [p], trace | segment | set(ports), ports, outgoing)
            if result is not None:
                return result
        return None

    return grow([], set(tail) | {cap}, [cap], direction), tail, cap


def _fallback_route(n, count, turn, full, accent):
    """Deterministic finite packing; count is retained, never redrawn."""
    s = n + 2
    h = w = 30
    if count == 0:
        return [], [(29, 28), (28, 28)], (27, 28)
    if turn == 0:
        r = 20 if n == 7 else 19
        cs = [4 + i * (s + 1) for i in range(count)]
        queries = [dict(r=r, c=c, side=s) for c in cs]
        tail, cap = [(r + s // 2, 0), (r + s // 2, 1)], (r + s // 2, 2)
        return queries, tail, cap
    if count <= 3:
        first_row = 13 if n == 7 else 12
        positions = [(first_row, 3), (first_row, 20), (22, 20)][:count]
    else:
        positions = [(11, 3), (11, 20), (23, 20), (23, 9)]
    if turn == -1:
        positions = [(r, 30 - c - s) for r, c in positions]
    queries = [dict(r=r, c=c, side=s) for r, c in positions]
    column = queries[0]['c'] + s // 2
    return queries, [(29, column), (28, column)], (27, column)


def _add_decoys(rng, h, w, ref, cue, queries, trace, desired):
    s = len(cue) + 2
    if desired == 0:
        return list(queries)
    result = list(queries)
    options = [(r, c) for r in range(1, h - s) for c in range(1, w - s)]
    rng.shuffle(options)
    for r, c in options:
        b = r, c, r + s, c + s
        if _boxes_near(b, ref[0]) or any(_boxes_near(b, _panel_box(p)) for p in result):
            continue
        if any(_contains(b, cell) for cell in trace):
            continue
        result.append(dict(r=r, c=c, side=s))
        if len(result) - len(queries) == desired:
            return result
    return result if desired == 0 else None


def _make_pair(rng, colors, n, count, turn, decoys, *, width=None, fallback_only=False):
    bg, signal, cap_color, frame, group, body, accent = colors
    kind = 1 if n == 3 else int(rng.integers(3))
    cue = _seed(n, kind, bg, body, accent)
    width = int(rng.integers(1, min(3, n - 2) + 1)) if width is None else int(width)
    if not 1 <= width <= n - 2:
        raise ValueError('outlet width does not fit the stencil')
    full, _, _ = _blueprint(rng, cue, bg, body, accent, width, turn)
    for attempt in range(0 if fallback_only else 45):
        h = int(rng.integers(24 if count >= 2 else n + 17, 31))
        w = int(rng.integers(max(24, 2 * (n + 2) + 5), 31))
        ref = _reference(rng, n, colors, cue, full, h, w, compact=count >= 3)
        result = _random_route(rng, h, w, ref, cue, full, colors, count)
        if result is None:
            continue
        route, tail, cap = result
        if route is None:
            continue
        queries, trace = route
        if decoys:
            queries = _add_decoys(rng, h, w, ref, cue, queries, trace, decoys)
            if queries is None:
                continue
        if not queries:
            continue
        inp = _assemble(h, w, colors, ref, cue, queries, tail, cap)
        try:
            scene = _decode(inp)
            out, facts = _trace(inp, scene)
            if len(facts['reached']) != count:
                continue
        except ValueError:
            continue
        rotation = int(rng.integers(4))
        return {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
    # Fallback uses the same cue, template, turn, scale, and exact route count.
    h = w = 30
    ref = _reference(rng, n, colors, cue, full, h, w, compact=True)
    # Compact calibration is above the circuit. Its horizontal offset is
    # retried when the final ray or reserved decoy area needs more clearance.
    for _ in range(40):
        queries, tail, cap = _fallback_route(n, count, turn, full, accent)
        # For one hit, the quarter-turn ray exits sideways below the reference.
        inp = _assemble(h, w, colors, ref, cue, queries, tail, cap)
        try:
            scene = _decode(inp) if queries else None
            if scene is None:
                # A miss still needs a query: try free positions off this ray.
                direction = 0
                trace = _empty_segment(h, w, [cap], direction, [], ref[0])
                if trace is None:
                    ref = _reference(rng, n, colors, cue, full, h, w, compact=True)
                    continue
                trace |= set(tail) | {cap}
            else:
                out, facts = _trace(inp, scene)
                if len(facts['reached']) != count:
                    raise ValueError("fallback contact changed")
                trace = facts['trace']
            new_queries = _add_decoys(rng, h, w, ref, cue, queries, trace, max(decoys, 1 if count == 0 else 0)) if decoys or count == 0 else queries
            if new_queries is None:
                raise ValueError("fallback cannot pack decoys")
            inp = _assemble(h, w, colors, ref, cue, new_queries, tail, cap)
            scene = _decode(inp)
            out, facts = _trace(inp, scene)
            if len(facts['reached']) != count or len(scene['queries']) != count + max(decoys, 1 if count == 0 else 0):
                raise ValueError('fallback must retain activation and decoy counts')
            rotation = int(rng.integers(4))
            return {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
        except ValueError:
            ref = _reference(rng, n, colors, cue, full, h, w, compact=True)
    raise RuntimeError("bounded packing failed without reducing requested complexity")


def _counts(a, b):
    for value, name, lo, hi in ((a, 'num_train', 2, 10), (b, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    return int(a), int(b)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.permutation(10)[:7]))
    vary_palette = bool(rng.integers(2)) and num_train >= 3
    puzzle = {'train': _training_twins(rng, colors), 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, count in (('train', num_train - 2), ('test', num_test)):
        available = set()
        if split == 'test':
            for pair in puzzle['train']:
                scene = _decode(pair['input'])
                if _trace(pair['input'], scene)[1]['reached']:
                    available.add(scene['width'])
        for index in range(count):
            for _ in range(20):
                palette = tuple(map(int, rng.permutation(10)[:7])) if vary_palette else colors
                if split == 'test':
                    if rng.random() < .25:
                        n, hits = 7, int(rng.choice([1, 2]))
                    else:
                        n = int(rng.choice([3, 5], p=[.75, .25]))
                        hits = 3 if n == 5 else int(rng.choice([3, 4], p=[.8, .2]))
                    turn = int(rng.choice([-1, 1]))
                else:
                    n = int(rng.choice([3, 5]))
                    hits = int(rng.choice([0, 1, 2, 3]))
                    turn = int(rng.choice([-1, 0, 1]))
                decoys = int(rng.choice([0, 1, 2], p=[.3, .55, .15])) if n == 3 else int(rng.choice([0, 1]))
                if n == 7 and hits == 2:
                    decoys = 0
                if hits == 0:
                    decoys = max(1, decoys)
                test_width = int(rng.choice([v for v in sorted(available) if v <= n - 2])) if split == 'test' else None
                pair = _make_pair(rng, palette, n, hits, turn, decoys, width=test_width)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not generate distinct inputs')
    rng.shuffle(puzzle['train'])
    return puzzle


def _project_check(inp, scene):
    """Invariant checker using distances to frame sides, rather than ray stepping."""
    bg, signal = scene['bg'], scene['signal']
    expected = np.full_like(inp, bg)
    for cell in scene['tail'] + [scene['cap']]:
        expected[cell] = signal
    origins, incoming, reached = [scene['cap']], scene['initial'], []
    boxes = [_panel_box(p) for p in scene['queries']]
    for _ in range(len(boxes) + 1):
        dr, dc = DIRS[incoming]
        stops = []
        for r, c in origins:
            limit = r + 1 if incoming == 0 else inp.shape[1] - c if incoming == 1 else inp.shape[0] - r if incoming == 2 else c + 1
            hits = [(limit, -1)]
            for i, (r0, c0, r1, c1) in enumerate(boxes):
                if incoming == 0 and c0 <= c < c1 and r0 < r:
                    distance = r - (r1 - 1)
                elif incoming == 2 and c0 <= c < c1 and r1 - 1 > r:
                    distance = r0 - r
                elif incoming == 1 and r0 <= r < r1 and c1 - 1 > c:
                    distance = c0 - c
                elif incoming == 3 and r0 <= r < r1 and c0 < c:
                    distance = c - (c1 - 1)
                else:
                    continue
                if distance > 0:
                    hits.append((distance, i))
            stops.append(min(hits))
        assert len(set(stops)) == 1, 'partial or tied collision'
        distance, index = stops[0]
        for r, c in origins:
            for k in range(1, distance):
                cell = r + k * dr, c + k * dc
                assert not _contains(scene['group'], cell)
                expected[cell] = signal
        if index == -1:
            return expected, reached
        assert index not in reached, 'cyclic circuit'
        reached.append(index)
        p = scene['queries'][index]
        a, outgoing, local_ports = _oriented(scene['template'], scene['reference_entry'], incoming, scene['accent'])
        for r, c in _rim(p['side']):
            expected[p['r'] + r, p['c'] + c] = signal
        area = expected[p['r'] + 1:p['r'] + p['side'] - 1, p['c'] + 1:p['c'] + p['side'] - 1]
        area[a != bg] = signal
        origins = _frame_ports(p, local_ports, outgoing)
        incoming = outgoing
    raise AssertionError('unterminated circuit')


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, witnesses, training_facts = set(), {}, []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            scene = _decode(inp)
            predicted, facts = _trace(inp, scene)
            projected, reached = _project_check(inp, scene)
            assert np.array_equal(out, predicted) and np.array_equal(out, projected)
            assert reached == facts['reached']
            assert set(map(int, np.unique(out))) == {scene['bg'], scene['signal']}
            # Reachability applies to complete frames; every off-route cue disappears.
            for i, p in enumerate(scene['queries']):
                interior = out[p['r'] + 1:p['r'] + p['side'] - 1, p['c'] + 1:p['c'] + p['side'] - 1]
                if i not in reached:
                    assert np.all(interior == scene['bg'])
                else:
                    oriented, _, _ = _oriented(scene['template'], scene['reference_entry'], facts['entries'][reached.index(i)], scene['accent'])
                    assert np.array_equal(interior, np.where(oriented != scene['bg'], scene['signal'], scene['bg']))
            y0, x0, y1, x1 = scene['group']
            assert np.all(out[y0:y1, x0:x1] == scene['bg'])
            if split == 'train':
                training_facts.append((scene, facts))
                normalized = inp.copy()
                p = scene['full']
                normalized[p['r'] + 1:p['r'] + p['side'] - 1, p['c'] + 1:p['c'] + p['side'] - 1] = scene['bg']
                witnesses.setdefault(_key(normalized), []).append(_key(out))
    assert any(len(set(answers)) >= 2 for answers in witnesses.values()), 'training lacks reference-dependence twins'
    assert any(len(f['reached']) >= 2 and len(f['reached']) < len(s['queries']) for s, f in training_facts), 'training must show chaining and erasure'
    assert {1, 3} <= {s['turn'] for s, f in training_facts}, 'training must show both quarter-turn orientations'
    demonstrated_widths = {s['width'] for s, f in training_facts if f['reached']}
    assert 1 in demonstrated_widths
    assert all(_decode(_grid(p['input']))['width'] in demonstrated_widths for p in puzzle['test']), 'test width lacks training evidence'


def self_test():
    """Bounded 42-seed variation, invariants, counterfactuals and API checks."""
    signatures, shapes, masks, widths, counts, directions, keys, palettes = (set() for _ in range(8))
    crossings = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((_key(p['input']), _key(p['output'])) for split in ('train', 'test') for p in puzzle[split]))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                scene = _decode(p['input'])
                out, facts = _trace(p['input'], scene)
                shapes.add(p['input'].shape)
                masks.add(_key(scene['template'] != scene['bg']))
                widths.add(scene['width'])
                counts.add(len(facts['reached']))
                directions.update(facts['entries'])
                keys.add(scene['kind'])
                palettes.add((scene['bg'], scene['signal'], scene['cap_color']))
                crossings += any(a & b for i, a in enumerate(facts['segments']) for b in facts['segments'][i + 1:])
                # Inlet position/orientation is input evidence, not panel order.
                swapped = p['input'].copy()
                a, b = scene['sparse'], scene['full']
                for dest, source in ((a, b), (b, a)):
                    swapped[dest['r'] + 1:dest['r'] + dest['side'] - 1, dest['c'] + 1:dest['c'] + dest['side'] - 1] = source['interior']
                assert np.array_equal(solve(swapped), out)
                assert np.array_equal(solve(p['input'].tolist()), p['output'])
                # Global rotation must commute with the entire transformation.
                assert np.array_equal(solve(np.rot90(p['input'])), np.rot90(out))
    assert len(signatures) == 42 and len(shapes) >= 20 and len(masks) >= 50
    assert {1, 2, 3} <= widths and {0, 1, 2, 3, 4} <= counts
    assert directions == {0, 1, 2, 3} and keys == {0, 1, 2}
    assert len(palettes) >= 25
    # Exercise the guaranteed packing path and the wider legal outlet spans.
    check_rng = np.random.default_rng(15029)
    for n in (3, 5, 7):
        for width in range(1, n - 1):
            for turn in (-1, 0, 1):
                for hits in (0, 1, 2):
                    decoys = 1 if hits < 2 else 0
                    pair = _make_pair(check_rng, (0, 1, 2, 3, 4, 5, 6), n, hits, turn, decoys,
                                      width=width, fallback_only=True)
                    scene = _decode(pair['input'])
                    predicted, facts = _trace(pair['input'], scene)
                    assert len(facts['reached']) == hits and scene['width'] == width
                    assert len(scene['queries']) - hits == decoys
                    assert np.array_equal(_project_check(pair['input'], scene)[0], predicted)
                    assert np.array_equal(predicted, pair['output'])
    for turn in (-1, 1):
        for hits in (3, 4):
            pair = _make_pair(check_rng, (0, 1, 2, 3, 4, 5, 6), 3, hits, turn, 1,
                              fallback_only=True)
            scene = _decode(pair['input'])
            assert len(_trace(pair['input'], scene)[1]['reached']) == hits
    # Reject a partial strip contact, rather than making an arbitrary choice.
    partial = _make_pair(check_rng, (0, 1, 2, 3, 4, 5, 6), 5, 2, 1, 0,
                         width=3, fallback_only=True)
    scene = _decode(partial['input'])
    inp = partial['input'].copy()
    facts = _trace(inp, scene)[1]
    second = scene['queries'][facts['reached'][1]]
    # Shift the second panel perpendicular to its incoming beam until only
    # part of the width meets it. Reference pixels and the first hit stay put.
    incoming = facts['entries'][1]
    if incoming % 2:
        axis = 'r'
    else:
        axis = 'c'
    damaged_input = None
    for offset in range(-second['side'], second['side'] + 1):
        if offset == 0:
            continue
        trial = inp.copy()
        trial[second['r']:second['r'] + second['side'], second['c']:second['c'] + second['side']] = scene['bg']
        y = second['r'] + (offset if axis == 'r' else 0)
        x = second['c'] + (offset if axis == 'c' else 0)
        if not (0 < y and 0 < x and y + second['side'] < len(trial) and x + second['side'] < trial.shape[1]):
            continue
        _draw_panel(trial, y, x, scene['sparse']['interior'], second['frame'])
        try:
            moved = _decode(trial)
            _trace(trial, moved)
        except ValueError as error:
            if 'partial or simultaneous' in str(error):
                damaged_input = trial
                break
    assert damaged_input is not None, 'partial-contact diagnostic did not exercise collision rejection'
    # Determinism and all explicit count combinations, without more seed sweeps.
    a, b = generate_puzzle(117, 2, 1), generate_puzzle(117, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(800 + nt * 10 + ne, nt, ne)
            validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('damaged answer accepted')
    return {'seeds': 42, 'puzzles': len(signatures), 'canvas_shapes': len(shapes),
            'stencil_masks': len(masks), 'outlet_widths': sorted(widths),
            'active_counts': sorted(counts), 'arrival_directions': sorted(directions),
            'cue_kinds': sorted(keys), 'palettes': len(palettes), 'ray_crossings': crossings}
