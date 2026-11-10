"""Snap relays: reachability triggers registered assembly and relocates emission.

Standard library + NumPy only. No files, global RNG state, or parent imports.
The constructor paints an intended route; solve reparses pixels and projects
nearest contacts. Checks are self-authored, not independent certification.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _shift(cells, delta):
    a, b = delta
    return {(r + a, c + b) for r, c in cells}


def _step(p, direction, distance=1):
    dr, dc = DIRS[direction]
    return p[0] + dr * distance, p[1] + dc * distance


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _components(cells):
    todo, parts = set(cells), []
    while todo:
        p = min(todo)
        todo.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in todo:
                        todo.remove(q)
                        part.add(q)
                        stack.append(q)
        parts.append(part)
    return parts


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _inside(cells, shape, margin=0):
    h, w = shape
    return all(margin <= r < h - margin and margin <= c < w - margin for r, c in cells)


def _grid(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _bar(cells):
    assert 2 <= len(cells) <= 6
    rows, cols = {r for r, c in cells}, {c for r, c in cells}
    assert (len(rows) == 1) != (len(cols) == 1)
    axis = 0 if len(rows) == 1 else 1
    span = sorted(c if axis == 0 else r for r, c in cells)
    assert span == list(range(span[0], span[0] + len(span)))
    return axis


def _backing(bar, material, footprint):
    axis = _bar(bar)
    directions = (0, 2) if axis == 0 else (1, 3)
    possible = [d for d in directions
                if any(_step(p, d) in material for p in bar)
                and all(_step(p, (d + 2) % 4) not in footprint for p in bar)]
    assert len(possible) == 1, 'bar must have exactly one backing side'
    return possible[0]


def _port_info(port, footprint):
    assert 1 <= len(port) <= 2
    top, bottom, left, right = _bbox(footprint)
    possible = [d for d, yes in enumerate((all(r == top for r, c in port),
                all(c == right for r, c in port), all(r == bottom for r, c in port),
                all(c == left for r, c in port))) if yes]
    assert len(possible) == 1, 'port must identify one extreme side'
    d = possible[0]
    spans = sorted(c if d % 2 == 0 else r for r, c in port)
    assert spans == list(range(spans[0], spans[0] + len(spans)))
    assert all(_step(p, (d + 2) % 4) in footprint - port for p in port)
    return d


def _decode(input_grid):
    """Infer launcher, palette, receiver/donor ownership and every translation."""
    a = _grid(input_grid)
    h, w = a.shape
    colors, counts = np.unique(a, return_counts=True)
    assert len(colors) == 5 and max(counts) > a.size // 2
    bg = int(colors[int(np.argmax(counts))])
    assert all(a[p] == bg for p in ((0, 0), (h - 1, 0), (0, w - 1), (h - 1, w - 1)))
    components = _components({tuple(map(int, p)) for p in np.argwhere(a != bg)})
    launch = [p for p in components if any(r in (0, h - 1) or c in (0, w - 1) for r, c in p)]
    assert len(launch) == 1
    launch = launch[0]
    edge = [p for p in launch if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    assert len(edge) == 1
    start = edge[0]
    direction = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    line = [_step(start, direction, k) for k in range(len(launch))]
    assert set(line) == launch and 3 <= len(line) <= 5
    signal, pc = int(a[start]), int(a[line[-1]])
    assert signal != pc and all(a[p] == signal for p in line[:-1])
    assert {tuple(map(int, p)) for p in np.argwhere(a == signal)} == set(line[:-1])
    choices = [int(c) for c in colors if c not in (bg, signal, pc)]
    candidates = []
    for marker, fg in (choices, choices[::-1]):
        try:
            by_length = {}
            for part in components:
                if part == launch:
                    continue
                guide = {p for p in part if a[p] == marker}
                body = {p for p in part if a[p] == fg}
                port = {p for p in part if a[p] == pc}
                assert guide and body and guide | body | port == part
                axis = _bar(guide)
                back = _backing(guide, body, part)
                record = dict(body=body, bar=guide, port=port, axis=axis, back=back)
                if port:
                    record['out_dir'] = _port_info(port, part)
                by_length.setdefault(len(guide), []).append(record)
            assert 1 <= len(by_length) <= 4
            objects = []
            for length, parts in sorted(by_length.items()):
                assert len(parts) == 2 and sum(bool(p['port']) for p in parts) == 1
                receiver = next(p for p in parts if not p['port'])
                donor = next(p for p in parts if p['port'])
                assert receiver['axis'] == donor['axis']
                assert receiver['back'] == (donor['back'] + 2) % 4
                target_bar = {_step(p, receiver['back']) for p in receiver['bar']}
                dest, source = min(target_bar), min(donor['bar'])
                delta = dest[0] - source[0], dest[1] - source[1]
                assert _shift(donor['bar'], delta) == target_bar
                moved = _shift(donor['body'] | donor['port'], delta)
                merged = receiver['body'] | moved
                port = _shift(donor['port'], delta)
                assert _inside(merged | target_bar, a.shape, 1)
                assert len(_components(merged)) == 1
                out_dir = donor['out_dir']
                assert not any(q in merged for p in port for k in range(1, 31)
                               for q in [_step(p, out_dir, k)])
                objects.append(dict(receiver=receiver['body'], rbar=receiver['bar'],
                                    donor=donor['body'] | donor['port'], dbar=donor['bar'],
                                    original_port=donor['port'], port=port, moved=moved,
                                    merged=merged, guides=receiver['bar'] | target_bar,
                                    out_dir=out_dir, delta=delta, length=length,
                                    axis=receiver['axis']))
            candidates.append(dict(bg=bg, signal=signal, pc=pc, fg=fg, marker=marker,
                                   tail=set(line[:-1]), cap=line[-1], initial=direction,
                                   objects=objects, shape=a.shape))
        except (AssertionError, ValueError):
            continue
    assert len(candidates) == 1, 'ambiguous or invalid input roles'
    return candidates[0]


def _run(decoded, variant='correct'):
    """Input-based lane projection, separate from route construction.

    Variants are finite, deliberately wrong rules for demonstration checks.
    No variant participates in solving or generation of the expected labels.
    """
    d = decoded
    objects, direction, origins = d['objects'], d['initial'], [d['cap']]
    trace, reached, transitions, gaps = d['tail'] | {d['cap']}, [], [], []
    ray_cells = set(trace)
    for _ in range(len(objects) + 1):
        dr, dc = DIRS[direction]
        stops = []
        for r, c in origins:
            limit = r + 1 if direction == 0 else d['shape'][1] - c if direction == 1 else d['shape'][0] - r if direction == 2 else c + 1
            hits = [(limit, -1)]
            for index, obj in enumerate(objects):
                if variant == 'ignore_ports' and index in reached:
                    continue
                for rr, cc in obj['receiver']:
                    along = (rr - r) * dr + (cc - c) * dc
                    across = (rr - r) * dc - (cc - c) * dr
                    if across == 0 and along > 0:
                        hits.append((along, index))
            stops.append(min(hits))
        assert len(set(stops)) == 1, 'partial-width or simultaneous different hits'
        distance, index = stops[0]
        segment = {_step(p, direction, k) for p in origins for k in range(1, distance)}
        trace.update(segment)
        ray_cells.update(segment)
        if index == -1:
            if variant == 'all_assemblies':
                trace.update(set().union(*(o['merged'] for o in objects)))
            if variant == 'paint_guides':
                trace.update(set().union(*(objects[i]['guides'] for i in reached)))
            if variant == 'erase_guides':
                trace -= set().union(*(objects[i]['guides'] for i in reached))
            out = np.full(d['shape'], d['bg'], dtype=np.int8)
            assert _inside(trace, d['shape'])
            for p in trace:
                out[p] = d['signal']
            return out, dict(active=len(reached), total=len(objects), reached=tuple(reached),
                             widths=tuple(t[2] for t in transitions),
                             turns=tuple((b - a) % 4 for a, b, _ in transitions),
                             gaps=tuple(gaps),
                             spare_passed=any(i not in reached and len(o['donor'] & ray_cells) >= 2
                                              and bool(o['donor'] - trace)
                                              for i, o in enumerate(objects)))
        assert index not in reached, 'cyclic route'
        obj = objects[index]
        assert obj['out_dir'] != (direction + 2) % 4, 'reverse emission'
        reached.append(index)
        gaps.append(distance - 1)
        if variant == 'no_registration':
            trace.update(obj['receiver'] | obj['donor'])
            new_origins = sorted(obj['original_port'])
        elif variant == 'xor_overlap':
            trace.update(obj['receiver'] ^ obj['moved'])
            new_origins = sorted(obj['port'])
        else:
            trace.update(obj['merged'])
            new_origins = sorted(obj['port'])
        if variant == 'opposite_seam':
            # The alternate adjacent order shifts the donor by two normals.
            rbar, dbar = obj['rbar'], _shift(obj['dbar'], obj['delta'])
            p, q = min(rbar), min(dbar)
            change = 2 * (p[0] - q[0]), 2 * (p[1] - q[1])
            trace -= obj['moved'] - obj['receiver']
            trace.update(_shift(obj['moved'], change))
            new_origins = sorted(_shift(obj['port'], change))
        out_dir = obj['out_dir']
        if variant == 'ignore_ports':
            new_origins = [_step(p, direction, distance) for p in origins]
            out_dir = direction
        if variant == 'unit_width':
            new_origins = new_origins[:1]
        transitions.append((direction, out_dir, len(new_origins)))
        direction, origins = out_dir, new_origins
    raise AssertionError('unterminated route')


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from the visible input and fixed family rules."""
    return _run(_decode(input_grid))[0]


def _rotate(cells, k):
    for _ in range(k):
        cells = {(-c, r) for r, c in cells}
    return cells


def _motif(rng, h, w):
    """Connected material with a complete registration-facing top row."""
    style = int(rng.integers(6))
    cells = {(r, c) for r in range(h) for c in range(w)}
    if style == 1:  # hollow ring
        cells = {(r, c) for r, c in cells if r in (0, h - 1) or c in (0, w - 1)}
    elif style == 2:  # lower corner steps / bevels
        depth = int(rng.integers(1, min(h - 1, w // 2) + 1))
        cells = {(r, c) for r, c in cells if r == 0 or
                 min(c, w - 1 - c) + (h - 1 - r) >= depth}
    elif style == 3:  # comb with a variable spine and tooth lengths
        spine = int(rng.integers(w))
        cells = {(0, c) for c in range(w)}
        for r in range(1, h):
            lo, hi = int(rng.integers(spine + 1)), int(rng.integers(spine, w))
            cells.update((r, c) for c in range(lo, hi + 1))
    elif style == 4:  # open angular hook
        side = int(rng.integers(2))
        cells = {(r, c) for r, c in cells if r == 0 or c == (w - 1 if side else 0) or r == h - 1}
    elif style == 5:  # boundary notch, preserving top and opposite side
        depth = int(rng.integers(1, max(2, h - 1)))
        cut = int(rng.integers(1, w - 1))
        cells -= {(r, c) for r in range(h - depth, h) for c in range(cut)}
    return cells


def _local_relay(rng, length, width, outgoing, collision=False, holes=False):
    for _ in range(100):
        rh, dh = int(rng.integers(2, 5)), int(rng.integers(3, 6))
        rw, dw = length + int(rng.integers(2, 5)), length + int(rng.integers(2, 5))
        ro, do = int(rng.integers(1, rw - length)), int(rng.integers(1, dw - length))
        receiver = {(-1 - r, c - ro) for r, c in _motif(rng, rh, rw)}
        donor = {(1 + r, c - do) for r, c in _motif(rng, dh, dw)}
        bar = {(0, c) for c in range(length)}
        if collision:
            # Cross the registration plane outside a bar endpoint. This creates
            # genuine material collisions after docking, without hiding guides.
            x = -1 if rng.integers(2) else length
            receiver.update((r, x) for r in (-1, 0, 1))
        if holes:
            donor.discard((1, int(rng.integers(length))))
            if rng.random() < .5:
                receiver.discard((-1, int(rng.integers(length))))
        footprint = donor | bar
        top, bottom, left, right = _bbox(footprint)
        possible = []
        for side in (1, 2, 3):
            for p in sorted(donor):
                port = {_step(p, 1 if side % 2 == 0 else 2, k) for k in range(width)}
                if not port <= donor:
                    continue
                try:
                    if _port_info(port, footprint) == side and all(q[0] >= 2 for q in port):
                        possible.append((side, port))
                except AssertionError:
                    pass
        if not possible:
            continue
        side, port = possible[int(rng.integers(len(possible)))]
        moved = _shift(donor, (-1, 0))
        moved_port = _shift(port, (-1, 0))
        merged = receiver | moved
        if collision and not receiver & moved:
            continue
        if not _shift(bar, (-1, 0)) - receiver and holes and not bar - moved:
            continue
        if len(_components(receiver | bar)) != 1 or len(_components(donor | bar)) != 1:
            continue
        if len(_components(merged)) != 1:
            continue
        if any(_step(p, side, k) in merged for p in moved_port for k in range(1, 20)):
            continue
        k = (side - outgoing) % 4
        return dict(receiver=_rotate(receiver, k), rbar=_rotate(bar, k),
                    donor=_rotate(moved, k), dbar=_rotate(_shift(bar, (-1, 0)), k),
                    port=_rotate(moved_port, k), out_dir=outgoing, length=length)
    raise AssertionError('local relay construction exhausted')


def _envelope(obj):
    return obj['receiver'] | obj['rbar'] | obj['donor'] | obj['dbar']


def _place(obj, delta):
    return {k: _shift(v, delta) if isinstance(v, set) else v for k, v in obj.items()}


def _entries(receiver, direction, width):
    """Every flat first-contact face that can receive a full-width strip."""
    lanes = {}
    for r, c in receiver:
        lane, along = (c, r) if direction % 2 == 0 else (r, c)
        lanes.setdefault(lane, []).append(along)
    edge = {lane: (max(values) if direction in (0, 3) else min(values))
            for lane, values in lanes.items()}
    result = []
    for lane in sorted(edge):
        if all(edge.get(lane + k) == edge[lane] for k in range(width)):
            result.append([(edge[lane], lane + k) if direction % 2 == 0
                           else (lane + k, edge[lane]) for k in range(width)])
    return result


def _boundary_ray(origins, direction, shape):
    cells = set()
    for p in origins:
        for k in range(1, 31):
            q = _step(p, direction, k)
            if not _inside({q}, shape):
                break
            cells.add(q)
    return cells


def _scene(rng, active, total, lengths, widths, deltas, palette, collisions, holes,
           spare_witness=False):
    # Counts/widths are chosen before retries and never simplified on failure.
    lower = 26 if total == 4 else 23 if active >= 2 else 18
    shape = tuple(map(int, rng.integers(lower, 31, size=2)))
    h, w = shape
    initial = int(rng.integers(4))
    axis_length = w if initial % 2 == 0 else h
    lane, tail_length = int(rng.integers(2, axis_length - 2)), int(rng.integers(2, 5))
    start = (h - 1, lane) if initial == 0 else (lane, 0) if initial == 1 else (0, lane) if initial == 2 else (lane, w - 1)
    tail = {_step(start, initial, k) for k in range(tail_length)}
    cap = _step(start, initial, tail_length)
    trace, rays, reserved, objects = tail | {cap}, tail | {cap}, set(), []
    origins, direction = [cap], initial
    for i in range(active):
        outgoing = (direction + deltas[i]) % 4
        placed = None
        blocked = _halo(reserved | tail | {cap})
        for _ in range(45):
            local = _local_relay(rng, lengths[i], widths[i], outgoing, collisions[i], holes[i])
            entries = _entries(local['receiver'], direction, len(origins))
            if not entries:
                continue
            for _ in range(22):
                entry = entries[int(rng.integers(len(entries)))]
                gap = int(rng.integers(1, 15))
                target = _step(origins[0], direction, gap + 1)
                delta = target[0] - entry[0][0], target[1] - entry[0][1]
                candidate = _place(local, delta)
                env = _envelope(candidate)
                approach = {_step(p, direction, k) for p in origins for k in range(1, gap + 1)}
                if not _inside(env, shape, 1) or env & blocked or env & rays:
                    continue
                if approach & reserved or approach & candidate['receiver']:
                    continue
                placed = candidate, env, approach
                break
            if placed is not None:
                break
        if placed is None:
            return None
        obj, env, approach = placed
        objects.append(obj)
        reserved |= env
        rays |= approach
        trace |= approach | obj['receiver'] | obj['donor']
        origins, direction = sorted(obj['port']), outgoing
    final = _boundary_ray(origins, direction, shape)
    if final & reserved:
        return None
    trace |= final
    rays |= final
    for i in range(active, total):
        placed = None
        blocked = _halo(reserved | tail | {cap})
        for _ in range(180):
            local = _local_relay(rng, lengths[i], int(rng.integers(1, 3)), int(rng.integers(4)),
                                 bool(rng.integers(2)), bool(rng.integers(2)))
            top, bottom, left, right = _bbox(_envelope(local))
            if bottom - top + 3 > h or right - left + 3 > w:
                continue
            delta = int(rng.integers(1 - top, h - bottom - 1)), int(rng.integers(1 - left, w - right - 1))
            obj = _place(local, delta)
            env = _envelope(obj)
            if env & blocked or env & rays:
                continue
            placed = obj, env
            break
        if placed is None:
            return None
        obj, env = placed
        objects.append(obj)
        reserved |= env
    # Spare parts occupy the same board, not a fixed inventory panel. Their
    # staging coordinates do not act as obstacles or as emission coordinates.
    occupied = tail | {cap} | set().union(*(o['receiver'] | o['rbar'] for o in objects))
    staged = {}
    order = sorted(range(total), key=lambda i: (not (spare_witness and i == active),
                                               -len(objects[i]['donor'])))
    for i in order:
        obj = objects[i]
        footprint = obj['donor'] | obj['dbar']
        top, bottom, left, right = _bbox(footprint)
        blocked = _halo(occupied)
        for _ in range(600):
            delta = int(rng.integers(1 - top, h - bottom - 1)), int(rng.integers(1 - left, w - right - 1))
            cells = _shift(footprint, delta)
            if cells & blocked or delta == (0, 0):
                continue
            if spare_witness and i == active:
                material = _shift(obj['donor'], delta)
                if len(material & rays) < 2 or not material - trace:
                    continue
            staged[i] = delta
            occupied |= cells
            break
        else:
            return None
    bg, signal, fg, marker, pc = palette
    if len(occupied) >= h * w // 2:
        return None
    inp, out = np.full(shape, bg, dtype=np.int8), np.full(shape, bg, dtype=np.int8)
    for i, obj in enumerate(objects):
        delta = staged[i]
        for cells, color in ((obj['receiver'], fg), (obj['rbar'], marker),
                             (_shift(obj['donor'], delta), fg),
                             (_shift(obj['dbar'], delta), marker),
                             (_shift(obj['port'], delta), pc)):
            for p in cells:
                inp[p] = color
    for p in tail:
        inp[p] = signal
    inp[cap] = pc
    for p in trace:
        out[p] = signal
    return dict(input=inp, output=out)


def _different(decoded, output, variant):
    try:
        return not np.array_equal(_run(decoded, variant)[0], output)
    except (AssertionError, IndexError, ValueError):
        # A competing rule that cannot finish also fails this demonstration.
        return True


def _remote_match_witness(decoded, reached):
    def center(cells):
        top, bottom, left, right = _bbox(cells)
        return (top + bottom) / 2, (left + right) / 2
    def distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    objects = decoded['objects']
    centers = [center(o['dbar']) for o in objects]
    return any(any(j != i and distance(center(objects[i]['rbar']), c) <
                   distance(center(objects[i]['rbar']), centers[i])
                   for j, c in enumerate(centers)) for i in reached)


def _make_pair(rng, profile, palette, seen):
    scene_profile = {k: v for k, v in profile.items() if k != 'address_witness'}
    for _ in range(100):
        pair = _scene(rng, palette=palette, **scene_profile)
        if pair is None or _key(pair['input']) in seen:
            continue
        try:
            d = _decode(pair['input'])
            answer, facts = _run(d)
            assert facts['active'] == profile['active']
            assert np.array_equal(answer, pair['output'])
            assert any(_different(d, pair['output'], v) for v in ('no_registration',))
            if any(profile['holes']):
                assert _different(d, pair['output'], 'paint_guides')
            if profile['spare_witness']:
                assert facts['spare_passed']
            if profile['address_witness']:
                assert _remote_match_witness(d, facts['reached'])
        except (AssertionError, ValueError):
            continue
        return pair
    raise AssertionError('bounded scene construction exhausted without simplifying the requested counts')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice(10, 5, replace=False)))
    vary_palette = bool(rng.integers(2))
    # The held-out axis is route depth, address length, or the combination of
    # all already demonstrated mechanisms. It does not change the rule.
    regime = int(rng.integers(3))
    turn_sign = int(rng.choice([-1, 1]))
    train_lengths = [2, 4, 5] if regime == 1 else [2, 3, 4, 5]
    seen, puzzle = set(), {'train': [], 'test': []}
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            demonstration = split == 'train' and index < 2
            active = index + 1 if demonstration else int(rng.integers(1, 4))
            if split == 'train' and regime == 0:
                active = min(active, 2)
            if split == 'test' and regime == 0:
                active = 3
            if split == 'test' and regime == 2:
                active = int(rng.integers(2, 4))
            total = min(4, active + int(rng.integers(0, 2)))
            if demonstration:
                total = active + 1
            available = train_lengths if split == 'train' else [2, 3, 4, 5]
            total = min(total, len(available))
            lengths = list(map(int, rng.choice(available, total, replace=False)))
            if split == 'test' and regime == 1:
                lengths[0] = 3
                remaining = [x for x in available if x != 3]
                lengths[1:] = list(map(int, rng.choice(remaining, total - 1, replace=False)))
            widths = list(map(int, rng.integers(1, 3, size=active)))
            if demonstration and index == 1:
                widths = list(map(int, rng.permutation([1, 2])))
            deltas = list(map(int, rng.choice([-1, 0, 1], active)))
            if demonstration and index == 0:
                deltas[0] = int(rng.choice([-1, 1]))
            if demonstration and index == 1:
                deltas[int(rng.integers(2))] = 0
            if regime == 2:
                if split == 'train':
                    sign = turn_sign if index % 2 == 0 else -turn_sign
                    deltas = list(map(int, rng.choice([0, sign], active)))
                    if demonstration and index == 0:
                        deltas[0] = sign
                    if demonstration and index == 1:
                        deltas = [0, sign]
                        rng.shuffle(deltas)
                else:
                    deltas[:2] = [turn_sign, -turn_sign]
                    rng.shuffle(deltas)
            collisions = [bool(rng.integers(2)) for _ in range(active)]
            holes = [bool(rng.integers(2)) for _ in range(active)]
            if demonstration:
                collisions[0] = holes[0] = index == 1
            colors = list(map(int, rng.permutation(palette))) if vary_palette else palette
            profile = dict(active=active, total=total, lengths=lengths, widths=widths,
                           deltas=deltas, collisions=collisions, holes=holes,
                           spare_witness=demonstration and index == 0,
                           address_witness=demonstration and index == 1)
            pair = _make_pair(rng, profile, colors, seen)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    order = rng.permutation(num_train)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _facts(pair):
    a, b = _grid(pair['input']), _grid(pair['output'])
    d = _decode(a)
    answer, facts = _run(d)
    assert a.shape == b.shape and np.array_equal(answer, b)
    assert set(map(int, np.unique(b))) == {d['bg'], d['signal']}
    objects = d['objects']
    # The footprint count equation checks union instead of XOR or overwriting.
    for obj in objects:
        assert len(obj['merged']) == len(obj['receiver']) + len(obj['moved']) - len(obj['receiver'] & obj['moved'])
        assert obj['delta'] != (0, 0)
    active = [objects[i] for i in facts['reached']]
    facts.update(shape=a.shape, initial=d['initial'], palette=(d['bg'], d['signal'], d['fg'], d['marker'], d['pc']),
                 lengths=tuple(o['length'] for o in objects), axes=tuple(o['axis'] for o in objects),
                 collisions=sum(len(o['receiver'] & o['moved']) for o in active),
                 uncovered=sum(len(o['guides'] - o['merged']) for o in active),
                 covered=sum(len(o['guides'] & o['merged']) for o in active),
                 donor_on_trace=any(any(b[p] == d['signal'] for p in o['donor']) for o in objects),
                 remote_match=_remote_match_witness(d, facts['reached']),
                 variants={v: _different(d, b, v) for v in ('no_registration', 'all_assemblies',
                           'ignore_ports', 'unit_width', 'opposite_seam', 'paint_guides', 'erase_guides',
                           'xor_overlap')})
    return facts


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen
            seen.add(key)
            facts[split].append(_facts(pair))
    train = facts['train']
    assert any(f['active'] >= 2 for f in train)
    assert any(f['active'] < f['total'] for f in train)
    assert any(f['spare_passed'] for f in train), 'must teach spare transparency'
    assert any(f['remote_match'] for f in train), 'must distinguish matching from proximity'
    assert {1, 2} <= {w for f in train for w in f['widths']}
    assert any(any(t in (1, 3) for t in f['turns']) for f in train)
    assert any(0 in f['turns'] for f in train)
    assert any(f['collisions'] for f in train)
    assert any(f['uncovered'] and f['covered'] for f in train)
    # Every named simplification must be falsified by the demonstrations.
    for variant in train[0]['variants']:
        assert any(f['variants'][variant] for f in train), f'unexposed alternative: {variant}'


def _boundary_case(miss=False):
    """A constructive four-machine regression, including an address of six.

    Four compact receivers run left to right; the spares lie below them.
    This regression is not used by generate_puzzle or solve.
    """
    a = np.zeros((30, 30), dtype=np.int8)
    b = np.zeros_like(a)
    material, guide, pc, signal = 1, 2, 3, 4
    row = 14 if miss else 8
    a[row, :2] = signal
    a[row, 2] = pc
    b[row, :3] = signal
    origins = {(row, 2)}
    for i, (length, width) in enumerate(zip((2, 3, 4, 6), (2, 1, 2, 1))):
        receiver = {(r, c) for r in (-2, -1) for c in range(-1, length + 1)}
        rbar = {(0, c) for c in range(length)}
        moved = {(r, c) for r in range(3) for c in range(-1, length + 1)}
        dbar = {(-1, c) for c in range(length)}
        port = {(2, c) for c in range(width)}
        at = (8, 6 + 6 * i)
        storage = (23, 2 + 6 * i)
        receiver, rbar = _shift(_rotate(receiver, 1), at), _shift(_rotate(rbar, 1), at)
        target = _shift(_rotate(moved, 1), at)
        target_port = _shift(_rotate(port, 1), at)
        for cells, color in ((receiver, material), (rbar, guide),
                             (_shift(_rotate(moved, 1), storage), material),
                             (_shift(_rotate(dbar, 1), storage), guide),
                             (_shift(_rotate(port, 1), storage), pc)):
            for p in cells:
                a[p] = color
        if not miss:
            contact_col = 4 + 6 * i
            for r, c in origins:
                b[r, c + 1:contact_col] = signal
            for p in receiver | target:
                b[p] = signal
            origins = target_port
    if miss:
        b[row, :] = signal
    else:
        for r, c in origins:
            b[r, c + 1:] = signal
    return {'input': a, 'output': b}


def self_test() -> None:
    """Stream 42 seeds, a maximum-count puzzle, and semantic negative checks."""
    fingerprints, structures, shapes, palettes = set(), set(), set(), set()
    records, counts = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                digest.update(str(a.shape).encode())
                digest.update(a.tobytes())
                facts = _facts(pair)
                records.append((split, facts))
                counts.add((facts['active'], facts['total']))
                shapes.add(a.shape)
                palettes.add(facts['palette'])
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, normalized))
                assert np.array_equal(solve(a.copy()), pair['output'])
        fingerprints.add(digest.hexdigest())
    assert len(fingerprints) == 42 and len(structures) >= 200
    assert len(shapes) >= 35 and len(palettes) >= 35
    assert {f['initial'] for _, f in records} == set(range(4))
    assert {a for _, f in records for a in f['axes']} == {0, 1}
    assert {n for _, f in records for n in f['lengths']} >= {2, 3, 4, 5}
    assert {1, 2, 3} <= {f['active'] for _, f in records}
    assert any(f['total'] == 4 for _, f in records)
    assert any(f['donor_on_trace'] for _, f in records)
    assert any(max(f['gaps'], default=0) >= 8 for _, f in records)
    for split in ('train', 'test'):
        rows = [f for s, f in records if s == split]
        assert any(f['active'] < f['total'] for f in rows)
        assert any(f['active'] == f['total'] for f in rows)
        assert any(f['collisions'] for f in rows) and any(f['uncovered'] for f in rows)
    for miss in (False, True):
        pair = _boundary_case(miss)
        assert np.array_equal(solve(pair['input']), pair['output'])
        facts = _facts(pair)
        assert facts['active'] == (0 if miss else 4)
        assert 6 in facts['lengths']
    def fingerprint(p):
        return tuple((_key(x['input']), _key(x['output'])) for s in ('train', 'test') for x in p[s])
    assert fingerprint(generate_puzzle(12345, 2, 1)) == fingerprint(generate_puzzle(12345, 2, 1))
    maximum = generate_puzzle(207, 10, 4)
    validate_generated_puzzle(maximum)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in maximum[s]] for s in maximum}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
