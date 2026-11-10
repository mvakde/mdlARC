import hashlib

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SOURCE_INPUTS = {((23, 22), 'ce5fc75ef8dd3697723ef409f4eb8575e2d2fe751a01edca7bc989d1bf9f8a13'), ((23, 22), 'e407f75db2ea0f69bc873d82148d8e75c9a924c9144ca806269c814172f77102'), ((13, 15), '53650874095b58e64662694cef7b7219d3d0046c734659909bfabb760613ec7d'), ((23, 25), 'b831a819c32f81155f13bba871b415fcbff0e5ed3e645d627936ce1159d12c21'), ((27, 27), 'a7254ef6ce250d920b57efebc3dc10206284f8c8bc5adad324926f877c308043')}


def _step(cell: tuple[int, int], direction: int) -> tuple[int, int]:
    dr, dc = _DIRS[direction]
    return cell[0] + dr, cell[1] + dc


def _local_mask(
    height: int,
    width: int,
    corner_depth: int,
) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(height)
        for c in range(width)
        if min(r, height - 1 - r) + min(c, width - 1 - c)
        >= corner_depth
    }


def _edge_values(
    mask: set[tuple[int, int]],
    height: int,
    width: int,
    side: int,
    exclude_corners: bool = False,
) -> list[int]:
    if side == 0:
        return sorted(
            c
            for r, c in mask
            if r == 0 and (not exclude_corners or 0 < c < width - 1)
        )
    if side == 2:
        return sorted(
            c
            for r, c in mask
            if r == height - 1
            and (not exclude_corners or 0 < c < width - 1)
        )
    if side == 3:
        return sorted(
            r
            for r, c in mask
            if c == 0 and (not exclude_corners or 0 < r < height - 1)
        )
    return sorted(
        r
        for r, c in mask
        if c == width - 1
        and (not exclude_corners or 0 < r < height - 1)
    )


def _window_starts(values: list[int], length: int) -> list[int]:
    value_set = set(values)
    return [
        value
        for value in values
        if all(value + offset in value_set for offset in range(length))
    ]


def _choose_local_port(
    mask: set[tuple[int, int]],
    height: int,
    width: int,
    side: int,
    port_width: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    starts = _window_starts(
        _edge_values(mask, height, width, side, exclude_corners=True),
        port_width,
    )
    if not starts:
        return None

    start = int(starts[int(rng.integers(len(starts)))])

    if side == 0:
        return {(0, start + k) for k in range(port_width)}
    if side == 2:
        return {(height - 1, start + k) for k in range(port_width)}
    if side == 3:
        return {(start + k, 0) for k in range(port_width)}
    return {(start + k, width - 1) for k in range(port_width)}


def _translate(
    mask: set[tuple[int, int]],
    row_offset: int,
    column_offset: int,
) -> set[tuple[int, int]]:
    return {
        (r + row_offset, c + column_offset)
        for r, c in mask
    }


def _is_separated(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return False
    return True


def _simulate_geometry(
    height: int,
    width: int,
    launcher_tail: set[tuple[int, int]] | list[tuple[int, int]],
    launcher_cap: tuple[int, int],
    initial_direction: int,
    objects: list[dict],
) -> tuple[
    set[tuple[int, int]],
    list[int],
    list[tuple[int, int, int]],
] | None:
    owner: dict[tuple[int, int], int] = {}
    for object_index, obj in enumerate(objects):
        for cell in obj["cells"]:
            if cell in owner:
                return None
            owner[cell] = object_index

    trace = set(launcher_tail)
    trace.add(launcher_cap)

    activated: list[int] = []
    seen: set[int] = set()
    transitions: list[tuple[int, int, int]] = []

    incoming_direction = initial_direction
    front = [_step(launcher_cap, initial_direction)]
    step_limit = height * width * (len(objects) + 3)

    for _ in range(step_limit):
        inside = [
            0 <= r < height and 0 <= c < width
            for r, c in front
        ]

        if not any(inside):
            return trace, activated, transitions
        if not all(inside):
            return None

        hits = {owner[cell] for cell in front if cell in owner}

        if hits:
            if len(hits) != 1:
                return None

            object_index = next(iter(hits))

            # Every cell across the ray's width must meet the same object
            # simultaneously. This excludes ambiguous partial collisions.
            if not all(owner.get(cell) == object_index for cell in front):
                return None
            if object_index in seen:
                return None

            seen.add(object_index)
            activated.append(object_index)

            obj = objects[object_index]
            trace.update(obj["cells"])

            outgoing_direction = obj["out_dir"]
            port_width = len(obj["port"])
            transitions.append(
                (
                    incoming_direction,
                    outgoing_direction,
                    port_width,
                )
            )

            incoming_direction = outgoing_direction
            front = sorted(
                _step(cell, incoming_direction)
                for cell in obj["port"]
            )
        else:
            trace.update(front)
            front = [
                _step(cell, incoming_direction)
                for cell in front
            ]

    return None


def _connected_components(cells):
    remaining = set(cells)
    components = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                neighbor = (r + dr, c + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _footprint(rng, incoming, outgoing, in_width, out_width, size_limit,
               rich, large=False, small=False):
    """Connected rectangles, beveled polygons, steps, and boundary notches."""
    min_h = max(3, in_width if incoming % 2 else 3,
                out_width + 2 if outgoing % 2 else 3)
    min_w = max(3, in_width if not incoming % 2 else 3,
                out_width + 2 if not outgoing % 2 else 3)
    high = min(size_limit, 6) if small else size_limit
    h = int(rng.integers(min_h, max(min_h, high) + 1))
    w = int(rng.integers(min_w, max(min_w, high) + 1))
    if large:
        h, w = max(h, 8), max(w, 8)
    elif rich and not small and size_limit >= 7 and rng.random() < .20:
        # Long thin relays, with either elongation axis.
        if rng.random() < .5:
            h, w = min_h, int(rng.integers(max(7, min_w), 13))
        else:
            h, w = int(rng.integers(max(7, min_h), 13)), min_w
    kind = int(rng.choice(6 if rich else 3))
    mask = _local_mask(h, w, min(kind, 2))
    if kind >= 3:
        mask = _local_mask(h, w, int(rng.integers(2)))
        # Removing a corner block makes a step; an interior edge notch
        # preserves the bridge across the opposite side of the footprint.
        side = int(rng.integers(4))
        if kind == 3:
            a = int(rng.integers(1, max(2, h // 2)))
            b = int(rng.integers(1, max(2, w // 2)))
            cut = {(r, c) for r in range(a) for c in range(b)}
        else:
            depth = int(rng.integers(1, max(2, h // 2)))
            start = int(rng.integers(1, w - 1))
            length = int(rng.integers(1, w - start))
            cut = {(r, c) for r in range(depth)
                   for c in range(start, min(w - 1, start + length))}
        if side & 1:
            cut = {(h - 1 - r, c) for r, c in cut}
        if side & 2:
            cut = {(r, w - 1 - c) for r, c in cut}
        mask -= cut
    if len(mask) <= 2 * out_width or len(_connected_components(mask)) != 1:
        return None
    r0, c0 = min(r for r, c in mask), min(c for r, c in mask)
    mask = _translate(mask, -r0, -c0)
    h, w = max(r for r, c in mask) + 1, max(c for r, c in mask) + 1
    if min(h, w) < 3:
        return None
    port = _choose_local_port(mask, h, w, outgoing, out_width, rng)
    if port is None or len(_connected_components(mask - port)) != 1:
        return None
    if not all(_step(cell, (outgoing + 2) % 4) in mask - port for cell in port):
        return None
    return mask, port, h, w


def _make_launcher(height, width, direction, rng):
    length = int(rng.integers(2, min(5, (height if direction % 2 == 0 else width) - 3)))
    if direction % 2 == 0:
        line = int(rng.integers(2, width - 2))
        tail = [(k if direction == 2 else height - 1 - k, line) for k in range(length)]
    else:
        line = int(rng.integers(2, height - 2))
        tail = [(line, k if direction == 1 else width - 1 - k) for k in range(length)]
    return tail, _step(tail[-1], direction)


def _canvas(rng, n, decoys):
    lower = 9 if n <= 1 else (16 if n == 2 else (21 if n == 3 else 26))
    h, w = map(int, rng.integers(lower, 31, size=2))
    if n <= 2 and rng.random() < .3:
        h, w = int(rng.integers(9 if n <= 1 else 12, 17)), int(rng.integers(24, 31))
        if rng.random() < .5:
            h, w = w, h
    if decoys >= 3:
        h, w = max(18, h), max(18, w)
    return h, w


def _try_layout(rng, spec, params):
    n, decoys = spec['n'], spec['decoys']
    h, w = _canvas(rng, n, decoys)
    direction = spec['initial']
    tail, cap = _make_launcher(h, w, direction, rng)
    budget = [180]

    def grow(objects, occupied, rays, front, incoming):
        index = len(objects)
        if index == n:
            result = _simulate_geometry(h, w, tail, cap, direction, objects)
            if result is not None and result[1] == list(range(n)):
                turns = [(b - a) % 4 for a, b, _ in result[2]]
                if spec['turn'] and not any(t in (1, 3) for t in turns):
                    return None
                if spec['straight'] and 0 not in turns:
                    return None
                return objects, result[0]
            return None
        for _ in range(35):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            outgoing = (incoming + int(rng.choice([-1, 0, 1]))) % 4
            ow = spec['widths'][index]
            shape = _footprint(rng, incoming, outgoing, len(front), ow,
                               spec['size_limit'], params['rich'],
                               large=spec['large_index'] == index,
                               small=n >= 4)
            if shape is None:
                continue
            mask, port, oh, ow_object = shape
            starts = _window_starts(_edge_values(mask, oh, ow_object, (incoming + 2) % 4), len(front))
            if not starts:
                continue
            entry = int(rng.choice(starts))
            options = []
            for gap in range(1, 25):
                target = (front[0][0] + gap * _DIRS[incoming][0],
                          front[0][1] + gap * _DIRS[incoming][1])
                if incoming == 0:
                    r0, c0 = target[0] - oh + 1, target[1] - entry
                elif incoming == 2:
                    r0, c0 = target[0], target[1] - entry
                elif incoming == 1:
                    r0, c0 = target[0] - entry, target[1]
                else:
                    r0, c0 = target[0] - entry, target[1] - ow_object + 1
                if r0 < 1 or c0 < 1 or r0 + oh >= h or c0 + ow_object >= w:
                    continue
                cells = _translate(mask, r0, c0)
                if cells & rays or not _is_separated(cells, occupied):
                    continue
                approach = {(r + k * _DIRS[incoming][0], c + k * _DIRS[incoming][1])
                            for r, c in front for k in range(gap)}
                if approach & occupied:
                    continue
                options.append((gap, cells, _translate(port, r0, c0), approach))
            if not options:
                continue
            # Equal mass for available short/medium/long distance bands,
            # independent of relay count, port width, and scale.
            bands = [[x for x in options if lo <= x[0] <= hi]
                     for lo, hi in ((1, 3), (4, 8), (9, 24))]
            bands = [b for b in bands if b]
            band = bands[int(rng.integers(len(bands)))]
            _, cells, placed_port, approach = band[int(rng.integers(len(band)))]
            obj = {'cells': cells, 'port': placed_port, 'out_dir': outgoing}
            next_front = sorted(_step(cell, outgoing) for cell in placed_port)
            result = grow(objects + [obj], occupied | cells, rays | approach,
                          next_front, outgoing)
            if result is not None:
                return result
        return None

    route = grow([], set(tail) | {cap}, set(tail) | {cap},
                 [_step(cap, direction)], direction)
    if route is None:
        return None
    objects, trace = route
    occupied = set(tail) | {cap} | set().union(*(obj['cells'] for obj in objects))
    for index in range(decoys):
        for attempt in range(160):
            outgoing = int(rng.integers(4))
            width = int(rng.choice(spec['allowed_widths']))
            shape = _footprint(rng, outgoing, outgoing, 1, width,
                               spec['size_limit'], params['rich'],
                               large=spec['large_decoy'] and index == 0,
                               small=attempt > 70 and not (spec['large_decoy'] and index == 0))
            if shape is None:
                continue
            mask, port, oh, ow = shape
            if oh + 2 > h or ow + 2 > w:
                continue
            r0, c0 = int(rng.integers(1, h - oh)), int(rng.integers(1, w - ow))
            cells = _translate(mask, r0, c0)
            if cells & trace or not _is_separated(cells, occupied):
                continue
            objects.append({'cells': cells, 'port': _translate(port, r0, c0), 'out_dir': outgoing})
            occupied |= cells
            break
        else:
            return None
    if len(occupied) >= h * w // 2:
        return None
    return h, w, tail, cap, direction, objects


def _fallback_layout(spec, params, rng):
    """A bounded packing fallback retains counts and the width sequence."""
    n = spec['n']
    h = w = 30
    band_h = max(4, max(spec['widths'], default=1) + 2)
    if spec['large_index'] is not None:
        band_h = max(8, band_h)
    row, col = int(rng.integers(2, 7)), 4
    tail, cap = [(row + 1, 0), (row + 1, 1)], (row + 1, 2)
    objects = []
    for i in range(n):
        last = i == n - 1
        width = spec['widths'][i]
        oh = band_h
        ow = max(3, width + 2) if last else 3
        if spec['large_index'] == i:
            ow = max(8, ow)
        cells = {(r, c) for r in range(row, row + oh) for c in range(col, col + ow)}
        # Keep cuts outside the incoming and emitting strips. The final
        # relay has a concave notch, the others have one clipped corner.
        if params['rich']:
            cells.remove((row + oh - 1, col + (1 if last else 0)))
        if last:
            port = {(row, col + 1 + k) for k in range(width)}
            outgoing = 0
        else:
            port = {(row + 1 + k, col + ow - 1) for k in range(width)}
            outgoing = 1
        objects.append({'cells': cells, 'port': port, 'out_dir': outgoing})
        col += ow + 1
    for j in range(spec['decoys']):
        r0, c0 = 16 + (j // 2) * 7, 2 + (j % 2) * 14
        oh, ow = (8, 8) if spec['large_decoy'] and j == 0 else (3, 4)
        # Reserve a separate row for the small decoy below a large one.
        if j == 2 and spec['large_decoy']:
            r0 = 26
        cells = {(r, c) for r in range(r0, r0 + oh) for c in range(c0, c0 + ow)}
        port = {(r0, c0 + 1)}
        objects.append({'cells': cells, 'port': port, 'out_dir': 0})
    # Rotate the fallback to the requested launcher side.
    for _ in range((1 - spec['initial']) % 4):
        rot = lambda cell: (29 - cell[1], cell[0])
        tail, cap = [rot(c) for c in tail], rot(cap)
        objects = [{'cells': {rot(c) for c in o['cells']},
                    'port': {rot(c) for c in o['port']},
                    'out_dir': (o['out_dir'] - 1) % 4} for o in objects]
    return h, w, tail, cap, spec['initial'], objects


def _generate_pair(rng, spec, params, colors, fallback_only=False):
    layout = None
    for _ in range(0 if fallback_only else 18):
        layout = _try_layout(rng, spec, params)
        if layout is not None:
            break
    if layout is None:
        layout = _fallback_layout(spec, params, rng)
    h, w, tail, cap, direction, objects = layout
    simulation = _simulate_geometry(h, w, tail, cap, direction, objects)
    assert simulation is not None and simulation[1] == list(range(spec['n']))
    bg, signal, port, cap_color = colors[:4]
    carrier_pool = colors[4:]
    inp = np.full((h, w), bg, dtype=np.int8)
    for index, obj in enumerate(objects):
        carrier = carrier_pool[index % len(carrier_pool)]
        for cell in obj['cells']:
            inp[cell] = port if cell in obj['port'] else carrier
    for cell in tail:
        inp[cell] = signal
    inp[cap] = cap_color
    out = np.full_like(inp, bg)
    for cell in simulation[0]:
        out[cell] = signal
    return {'input': inp, 'output': out}


def _decode_input(inp):
    """Recover every role from visible geometry, without output or metadata."""
    h, w = inp.shape
    bg = int(inp[0, 0])
    assert all(int(inp[r, c]) == bg for r, c in ((0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    components = _connected_components({tuple(map(int, x)) for x in np.argwhere(inp != bg)})
    launchers = [c for c in components if any(r in (0, h - 1) or s in (0, w - 1) for r, s in c)]
    assert len(launchers) == 1
    launcher = launchers[0]
    boundary = [c for c in launcher if c[0] in (0, h - 1) or c[1] in (0, w - 1)]
    assert len(boundary) == 1
    start = boundary[0]
    initial = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    line = [(start[0] + k * _DIRS[initial][0], start[1] + k * _DIRS[initial][1]) for k in range(len(launcher))]
    assert set(line) == launcher and 3 <= len(line) <= 6
    signal, cap_color = int(inp[start]), int(inp[line[-1]])
    assert signal != cap_color and all(int(inp[c]) == signal for c in line[:-1])
    cap = line[-1]
    objects, carriers, port_colors = [], set(), set()
    for component in components:
        if component == launcher:
            continue
        palette, counts = np.unique([int(inp[c]) for c in component], return_counts=True)
        assert len(palette) == 2 and counts[0] != counts[1]
        pc = int(palette[int(np.argmin(counts))])
        carrier = int(palette[int(np.argmax(counts))])
        port = {c for c in component if int(inp[c]) == pc}
        body = component - port
        assert 1 <= len(port) <= 4 and len(body) > len(port)
        assert len(_connected_components(body)) == 1
        r0, r1 = min(r for r, c in component), max(r for r, c in component)
        c0, c1 = min(c for r, c in component), max(c for r, c in component)
        possible = [side for side, ok in enumerate((all(r == r0 for r, c in port),
                    all(c == c1 for r, c in port), all(r == r1 for r, c in port),
                    all(c == c0 for r, c in port))) if ok]
        assert len(possible) == 1
        side = possible[0]
        span = sorted(c if side % 2 == 0 else r for r, c in port)
        assert span == list(range(span[0], span[0] + len(span)))
        assert all(_step(cell, (side + 2) % 4) in body for cell in port)
        assert 0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1
        objects.append({'cells': component, 'port': port, 'out_dir': side,
                        'bbox': (r0, r1, c0, c1), 'h': r1 - r0 + 1, 'w': c1 - c0 + 1,
                        'carrier': carrier})
        carriers.add(carrier)
        port_colors.add(pc)
    assert objects and len(port_colors) == 1
    pc = next(iter(port_colors))
    assert len({bg, signal, cap_color, pc}) == 4
    assert not carriers & {bg, signal, cap_color, pc}
    assert np.count_nonzero(inp == bg) > inp.size // 2
    return bg, signal, pc, cap_color, set(line[:-1]), cap, initial, objects


def _predict_input(inp):
    """Separate solver: project each original object onto each beam lane.

    Unlike construction's advancing front, compute the nearest positive
    distance in each lane, then paint an entire segment in one operation.
    """
    bg, signal, pc, cap_color, tail, cap, direction, objects = _decode_input(inp)
    out = np.full_like(inp, bg)
    for cell in tail | {cap}:
        out[cell] = signal
    origins, reached, transitions, gaps, segments = [cap], [], [], [], []
    for _ in range(len(objects) + 1):
        dr, dc = _DIRS[direction]
        stops = []
        for r, c in origins:
            limit = r + 1 if direction == 0 else inp.shape[1] - c if direction == 1 else inp.shape[0] - r if direction == 2 else c + 1
            hits = [(limit, -1)]
            for index, obj in enumerate(objects):
                for rr, cc in obj['cells']:
                    distance = (rr - r) * dr + (cc - c) * dc
                    transverse = (rr - r) * dc - (cc - c) * dr
                    if transverse == 0 and distance > 0:
                        hits.append((distance, index))
            stops.append(min(hits))
        assert len(set(stops)) == 1, 'partial or tied contact'
        distance, index = stops[0]
        segment = {(r + k * dr, c + k * dc) for r, c in origins for k in range(1, distance)}
        for cell in segment:
            out[cell] = signal
        segments.append(segment)
        if index == -1:
            facts = _features(inp, objects, reached, transitions, gaps, segments)
            facts.update(initial=_decode_input_initial(cap, tail),
                         palette=(bg, signal, pc, cap_color),
                         carriers=sorted({obj['carrier'] for obj in objects}))
            return out, facts
        assert index not in reached, 'cycle or reactivation'
        reached.append(index)
        obj = objects[index]
        assert obj['out_dir'] != (direction + 2) % 4
        gaps.append(distance - 1)
        transitions.append((direction, obj['out_dir'], len(obj['port'])))
        for cell in obj['cells']:
            out[cell] = signal
        origins, direction = sorted(obj['port']), obj['out_dir']
    raise AssertionError('unterminated route')


def _decode_input_initial(cap, tail):
    return next(d for d in range(4) if _step(cap, (d + 2) % 4) in tail)


def _features(inp, objects, reached, transitions, gaps, segments):
    kinds, dims, asymmetric, notched = [], [], [], []
    for obj in objects:
        r0, r1, c0, c1 = obj['bbox']
        mask = _translate(obj['cells'], -r0, -c0)
        h, w = obj['h'], obj['w']
        dims.append((h, w))
        symmetric = (mask == {(h - 1 - r, c) for r, c in mask}
                     and mask == {(r, w - 1 - c) for r, c in mask})
        asymmetric.append(not symmetric)
        kind = next((depth for depth in range(3) if mask == _local_mask(h, w, depth)), -1)
        kinds.append(kind)
        notched.append(any((r, c) not in mask and
                           (any((r, j) in mask for j in range(c)) and any((r, j) in mask for j in range(c + 1, w))
                            or any((j, c) in mask for j in range(r)) and any((j, c) in mask for j in range(r + 1, h)))
                           for r in range(h) for c in range(w)))
    behind = 0
    # An unreached object lies farther along an incoming beam than its hit.
    for route_index, index in enumerate(reached):
        direction = transitions[route_index][0]
        hit = objects[index]['cells']
        dr, dc = _DIRS[direction]
        lanes = {r * dc - c * dr for r, c in segments[route_index]}
        far = max(r * dr + c * dc for r, c in hit)
        behind += any(j not in reached and any(r * dc - c * dr in lanes and r * dr + c * dc > far for r, c in obj['cells']) for j, obj in enumerate(objects))
    return {'shape': tuple(inp.shape), 'active_count': len(reached), 'decoys': len(objects) - len(reached),
            'activated': reached, 'transitions': transitions, 'gaps': gaps,
            'widths': [x[2] for x in transitions], 'object_dims': dims, 'kinds': kinds,
            'asymmetric': asymmetric, 'notched': notched,
            'active_asymmetric': any(asymmetric[i] for i in reached),
            'active_notched': any(notched[i] for i in reached),
            'large_first': bool(reached and min(dims[reached[0]]) >= 8),
            'large_later': any(min(dims[i]) >= 8 for i in reached[1:]),
            'large_decoy': any(min(dim) >= 8 for i, dim in enumerate(dims) if i not in reached),
            'crossing': any(a & b for i, a in enumerate(segments) for b in segments[i + 1:]),
            'behind_hit': bool(behind)}


def solve(train_pairs, test_input):
    """Prediction uses demonstrations and the test input, never test labels."""
    for pair in train_pairs:
        prediction, _ = _predict_input(pair['input'])
        assert np.array_equal(prediction, pair['output'])
    return _predict_input(test_input)[0]


def _check_training_coverage(train, test):
    assert any(f['active_count'] >= 2 for f in train)
    assert any(f['decoys'] >= 1 for f in train)
    transitions = [t for f in train for t in f['transitions']]
    assert any(a != b for a, b, _ in transitions)
    assert any(a == b for a, b, _ in transitions)
    widths = {t[2] for t in transitions}
    assert 1 in widths and len(widths) >= 2
    assert all(set(f['widths']) <= widths for f in test)
    if any(f['active_asymmetric'] for f in test):
        assert any(f['active_asymmetric'] for f in train)
    if any(f['active_notched'] for f in test):
        assert any(f['active_notched'] for f in train)
    if len({f['palette'] for f in train + test}) > 1:
        # Visible pair-local roles, including tail color, are demonstrated.
        assert all(len({f['palette'][i] for f in train}) >= 2 for i in range(4))
    if any(len(f['carriers']) > 1 for f in test):
        assert any(len(f['carriers']) > 1 and f['active_count'] >= 2 for f in train)


def _input_key(inp):
    return tuple(inp.shape), inp.astype(np.int8).tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= a <= 30 for a in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _input_key(inp)
            assert key not in seen
            assert (key[0], hashlib.sha256(key[1]).hexdigest()) not in _SOURCE_INPUTS
            seen.add(key)
            prediction, record = _predict_input(inp.copy())
            assert np.array_equal(prediction, out)
            assert 0 <= record['active_count'] <= 5 and 0 <= record['decoys'] <= 4
            facts[split].append(record)
    _check_training_coverage(facts['train'], facts['test'])


def _validate_count(value, name, minimum, maximum):
    if value is not None and (isinstance(value, (bool, np.bool_)) or
                             not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum):
        raise ValueError(f'{name} must be an integer from {minimum} through {maximum}')


def _specification(rng, params, regime, demo=None, allowed=None):
    n = int(rng.choice(6, p=[.12, .18, .24, .28, .12, .06]))
    if demo is not None:
        n = int(rng.choice([2, 3], p=[.4, .6]))
    decoys = int(rng.choice(5, p=[.27, .30, .23, .14, .06]))
    if n == 0 or demo == 0:
        decoys = max(1, decoys)
    allowed = list(range(1, params['max_width'] + 1)) if allowed is None else list(allowed)
    widths = list(map(int, rng.choice(allowed, size=n)))
    if demo is not None:
        widths[:2] = list(map(int, rng.permutation([1, params['max_width']])))
    source_train = params['scale_shift'] and regime == 'source_train'
    source_test = params['scale_shift'] and regime == 'target_test' and n > 0
    large_index = 0 if source_test else None
    if source_test:
        n = min(n, 3)
        widths = widths[:n]
    elif not source_train and n and n <= 3 and rng.random() < .22:
        large_index = int(rng.integers(n))
    return {'n': n, 'decoys': decoys, 'initial': params['side'] if params['fixed_side'] else int(rng.integers(4)),
            'widths': widths, 'allowed_widths': allowed,
            'size_limit': 6 if source_train or source_test else int(rng.choice([6, 8, 10], p=[.45, .30, .25])),
            'large_index': large_index, 'large_decoy': not source_train and decoys > 0 and rng.random() < .18,
            'turn': demo is not None, 'straight': demo == 0}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    _validate_count(num_train, 'num_train', 2, 10)
    _validate_count(num_test, 'num_test', 1, 4)
    rng = np.random.default_rng(seed)
    if num_test is None and num_train is None:
        num_test = 1 + int(seed) % 4
        lo = num_test + 1
        num_train = lo + (int(seed) // 4) % (11 - lo)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    params = {'rich': rng.random() < .70, 'vary_colors': rng.random() < .70,
              'fixed_side': rng.random() < .30, 'side': int(rng.integers(4)),
              'scale_shift': rng.random() < .20, 'max_width': int(rng.choice([2, 3, 4], p=[.5, .25, .25]))}
    if params['scale_shift']:
        params['max_width'] = 2
    multicarrier = params['rich'] and rng.random() < .6
    base = list(map(int, rng.permutation(10)))
    carrier_count = int(rng.integers(2, 5)) if multicarrier else 1
    puzzle, seen, train_facts = {'train': [], 'test': []}, set(), []
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            demo = index if split == 'train' and index < 2 else None
            allowed = sorted({v for f in train_facts for v in f['widths']}) if split == 'test' else None
            spec = _specification(rng, params, 'source_train' if split == 'train' else 'target_test', demo, allowed)
            if params['vary_colors']:
                # The first two demonstrations change every role, including
                # signal; later examples freely permute the complete palette.
                palette = base if index == 0 and split == 'train' else list(np.roll(base, 1)) if index == 1 and split == 'train' else list(map(int, rng.permutation(10)))
            else:
                palette = base
            colors = palette[:4 + carrier_count]
            for attempt in range(50):
                pair = _generate_pair(rng, spec, params, colors, fallback_only=attempt == 49)
                facts = _predict_input(pair['input'])[1]
                if demo is not None and params['rich'] and not facts['active_notched']:
                    continue
                if split == 'test' and facts['active_notched'] and not any(f['active_notched'] for f in train_facts):
                    continue
                if split == 'test' and facts['active_asymmetric'] and not any(f['active_asymmetric'] for f in train_facts):
                    continue
                key = _input_key(pair['input'])
                if key in seen or (key[0], hashlib.sha256(key[1]).hexdigest()) in _SOURCE_INPUTS:
                    continue
                break
            else:
                raise AssertionError('unable to provide distinct, informative demonstrations')
            puzzle[split].append(pair)
            seen.add(key)
            if split == 'train':
                train_facts.append(facts)
    # The two coverage examples are shuffled among all training examples.
    order = rng.permutation(len(puzzle['train']))
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((_input_key(p['input']), _input_key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test() -> None:
    puzzles, records = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _, facts = _predict_input(pair['input'].copy())
                records.append((seed, split, facts))
        training = [{k: v.copy() for k, v in p.items()} for p in puzzle['train']]
        for pair in puzzle['test']:
            assert np.array_equal(solve(training, pair['input'].copy()), pair['output'])
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    facts = [f for _, _, f in records]
    assert len({f['shape'] for f in facts}) >= 70
    assert {f['initial'] for f in facts} == set(range(4))
    for split in ('train', 'test'):
        rows = [f for _, s, f in records if s == split]
        assert set(range(4)) <= {f['active_count'] for f in rows}
        assert any(f['active_count'] >= 4 for f in rows)
        assert any(f['decoys'] == 0 for f in rows) and any(f['decoys'] >= 3 for f in rows)
        assert {1, 2, 3, 4} <= {v for f in rows for v in f['widths']}
        assert any(max(f['gaps'], default=0) >= 10 for f in rows)
        assert any(f['large_later'] for f in rows) and any(f['large_decoy'] for f in rows)
        assert any(min(f['shape']) <= 13 for f in rows)
        assert any(max(f['shape']) / min(f['shape']) >= 2 for f in rows)
        assert any(30 in f['shape'] for f in rows)
        assert any(f['active_notched'] for f in rows)
        assert {0, 1, 3} == {(b - a) % 4 for f in rows for a, b, _ in f['transitions']}
    assert sum(f['active_count'] == 3 for _, s, f in records if s == 'train') >= 25
    assert any(2 in f['kinds'] and any(min(d) <= 6 for d in f['object_dims']) for f in facts)
    assert any((3, 4) in f['object_dims'] or (4, 3) in f['object_dims'] for f in facts)
    assert any(f['crossing'] for f in facts) and any(f['behind_hit'] for f in facts)
    assert sum(len({f['palette'] for seed, _, f in records if seed == i}) > 1 for i in range(42)) >= 15
    assert sum(len({f['initial'] for seed, _, f in records if seed == i}) > 1 for i in range(42)) >= 15
    assert sum(len({f['active_count'] for seed, _, f in records if seed == i}) >= 3 for i in range(42)) >= 20
    assert any(len(f['carriers']) >= 2 and f['active_notched'] and max(f['gaps'], default=0) >= 9 for f in facts)
    assert all({f['palette'][role] for f in facts} == set(range(10)) for role in range(4))
    assert {c for f in facts for c in f['carriers']} == set(range(10))
    # Exercise the guaranteed packing path even when random placement
    # succeeds in all ordinary samples, including its widest ports.
    fallback_rng = np.random.default_rng(95137)
    for n in range(6):
        for direction in range(4):
            for rich in (False, True):
                spec = {'n': n, 'initial': direction, 'decoys': 4,
                        'widths': [1 + (i + direction) % 4 for i in range(n)],
                        'large_index': n - 1 if 0 < n <= 3 else None,
                        'large_decoy': True}
                pair = _generate_pair(fallback_rng, spec, {'rich': rich},
                                      [0, 2, 3, 4, 1], fallback_only=True)
                prediction, record = _predict_input(pair['input'])
                assert np.array_equal(prediction, pair['output'])
                assert record['active_count'] == n and record['decoys'] == 4
                assert not rich or not n or record['active_notched']
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
              for s in ('train', 'test')}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1073, *counts)
        validate_generated_puzzle(puzzle)
        for key, request in zip(('train', 'test'), counts):
            assert request is None or len(puzzle[key]) == request
    assert _puzzle_key(generate_puzzle(314159, 6, 3)) == _puzzle_key(generate_puzzle(314159, 6, 3))
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(7, *counts)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
