"""Plug-in pulse fans: registered silhouettes are executable run programs.

Only Python's standard library and NumPy are used. No parent files, examples,
seed-dependent solver conventions, or construction metadata are consulted.
"""
from __future__ import annotations

import math
import numpy as np


_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1)
                   for c in (-1, 0, 1) if r or c)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'], dtype=np.int8)),
                        _key(np.asarray(p['output'], dtype=np.int8)))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _hub_cells(r, c, n):
    k = n // 2 - 1
    return (((r, c + k), (r, c + k + 1)),
            ((r + k, c + n - 1), (r + k + 1, c + n - 1)),
            ((r + n - 1, c + k), (r + n - 1, c + k + 1)),
            ((r + k, c), (r + k + 1, c)))


def _draw_hub(a, r, c, n, labels):
    for side, cells in enumerate(_hub_cells(r, c, n)):
        for rr, cc in cells:
            a[rr, cc] = labels[side]


def _find_hubs(a, bg):
    """Find four centered two-cell bars on an otherwise empty even square."""
    candidates = []
    top_pairs = np.argwhere((a[:, :-1] != bg) & (a[:, :-1] == a[:, 1:]))
    for r0, first in top_pairs:
        r = int(r0)
        for n in range(4, min(a.shape[0] - r, a.shape[1]) + 1, 2):
            c = int(first) - (n // 2 - 1)
            if c < 0 or c + n > a.shape[1]:
                continue
            bars = _hub_cells(r, c, n)
            labels = [int(a[cells[0]]) for cells in bars]
            if bg in labels or len(set(labels)) != 4:
                continue
            if any(int(a[cells[1]]) != color for cells, color in zip(bars, labels)):
                continue
            if np.count_nonzero(a[r:r + n, c:c + n] != bg) == 8:
                candidates.append((r, c, n, tuple(labels)))
    return candidates


def _components(a, bg, hub):
    occupied = a != bg
    for cells in _hub_cells(*hub[:3]):
        for point in cells:
            occupied[point] = False
    visited = np.zeros(a.shape, dtype=bool)
    pieces = []
    for rr, cc in np.argwhere(occupied):
        start = (int(rr), int(cc))
        if visited[start]:
            continue
        stack, cells = [start], []
        visited[start] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                p = (r + dr, c + dc)
                if (0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]
                        and occupied[p] and not visited[p]):
                    visited[p] = True
                    stack.append(p)
        coords = np.array(cells, dtype=int)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        body = np.full(tuple(hi - lo + 1), bg, dtype=np.int8)
        body[coords[:, 0] - lo[0], coords[:, 1] - lo[1]] = a[coords[:, 0], coords[:, 1]]
        pieces.append(dict(coords=coords, body=body))
    return pieces


def _attachments(body, bg, hub_size):
    """Distinct colored programs, with the connector on the bottom edge.

    Equivalent quarter-turn choices collapse only if the complete colored
    matrix agrees, so a symmetric outline cannot conceal a different program.
    """
    found = {}
    for turn in range(4):
        canonical = np.rot90(body, turn).copy()
        h, w = canonical.shape
        if h < 2 or w < 2 or w % 2 or w > hub_size:
            continue
        center = w // 2 - 1
        mask = np.zeros(w, dtype=bool)
        mask[center:center + 2] = True
        if not np.array_equal(canonical[-1] != bg, mask):
            continue
        label = int(canonical[-1, center])
        if int(canonical[-1, center + 1]) != label:
            continue
        k = (label, _key(canonical))
        if k not in found:
            found[k] = dict(label=label, canonical=canonical, turns=[])
        found[k]['turns'].append(turn)
    return list(found.values())


def _runs(canonical, bg):
    """Read each radial lane from the connector outward; omit its two cells."""
    h, w = canonical.shape
    lanes = []
    for q in range(w):
        lane, t = [], 1
        while t < h:
            color = int(canonical[h - 1 - t, q])
            if color == bg:
                t += 1
                continue
            start = t
            while t < h and int(canonical[h - 1 - t, q]) == color:
                t += 1
            lane.append(dict(color=color, length=t - start, near=start))
        lanes.append(lane)
    return lanes


def _distance(shape, hub, side):
    r, c, n = hub[:3]
    return (r, shape[1] - c - n, shape[0] - r - n, c)[side]


def _fan_cell(hub, side, width, q, d):
    r, c, n = hub[:3]
    u = (n - width) // 2 + q
    return ((r - d, c + u),
            (r + u, c + n - 1 + d),
            (r + n - 1 + d, c + n - 1 - u),
            (r + n - 1 - u, c - d))[side]


def _inspect(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= a.size // 2:
        raise ValueError("input background must occupy more than half the grid")
    hubs = _find_hubs(a, bg)
    if len(hubs) != 1:
        raise ValueError("one unambiguous hub is required")
    hub = hubs[0]
    pieces = _components(a, bg, hub)
    if not pieces:
        raise ValueError("input must contain loose keys")
    r, c, n, labels = hub
    selected, used = [], set()
    for piece in pieces:
        coords = piece['coords']
        if any(r - 1 <= rr <= r + n and c - 1 <= cc <= c + n for rr, cc in coords):
            raise ValueError("keys must be clear of the hub's bounding square")
        candidates = _attachments(piece['body'], bg, n)
        matching = [p for p in candidates if p['label'] in labels]
        if not matching:
            piece['side'] = None
            continue
        if len(candidates) != 1:
            raise ValueError("matching key has competing centered connectors")
        item = matching[0]
        side = labels.index(item['label'])
        if side in used:
            raise ValueError("at most one matching key per port")
        used.add(side)
        piece.update(item)
        piece['side'] = side
        piece['lanes'] = _runs(item['canonical'], bg)
        piece['rotations'] = {(t - side) % 4 for t in item['turns']}
        selected.append(piece)
    return dict(background=bg, hub=hub, pieces=pieces, selected=selected)


def _predict(a, info, priority='near', period_delta=0, phase='hub', mirror=False):
    bg, hub = info['background'], info['hub']
    out = np.full_like(a, bg)
    _draw_hub(out, *hub[:3], hub[3])
    for p in info['selected']:
        width = p['canonical'].shape[1]
        for q, lane in enumerate(p['lanes']):
            for d in range(1, _distance(a.shape, hub, p['side']) + 1):
                hits = [b for b in lane if (d - 1 - (b['near'] - 1 if phase == 'gap' else 0))
                        % max(1, b['length'] + period_delta) == 0]
                if not hits:
                    continue
                if priority == 'short':
                    winner = min(hits, key=lambda b: (b['length'], b['near']))
                elif priority == 'long':
                    winner = min(hits, key=lambda b: (-b['length'], b['near']))
                elif priority == 'far':
                    winner = hits[-1]
                else:
                    winner = hits[0]
                out[_fan_cell(hub, p['side'], width, width - 1 - q if mirror else q, d)] = winner['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible hub, components, connector geometry, and run lengths."""
    a = _grid(input_grid)
    return _predict(a, _inspect(a))


def _paint_from_input(a, info):
    """Construction painter: superimpose far-to-near, instead of cell voting."""
    out = np.full_like(a, info['background'])
    hub = info['hub']
    _draw_hub(out, *hub[:3], hub[3])
    for p in info['selected']:
        width = p['canonical'].shape[1]
        for q, lane in enumerate(p['lanes']):
            for b in reversed(lane):
                for d in range(1, _distance(a.shape, hub, p['side']) + 1, b['length']):
                    out[_fan_cell(hub, p['side'], width, q, d)] = b['color']
    return out


def _dock_only(a, info):
    """Ablation of A: ordinary B-style shape-preserving docking, clipped."""
    out = np.full_like(a, info['background'])
    hub = info['hub']
    _draw_hub(out, *hub[:3], hub[3])
    for p in info['selected']:
        body = p['canonical']
        for rr, q in np.argwhere(body != info['background']):
            point = _fan_cell(hub, p['side'], body.shape[1], int(q), body.shape[0] - int(rr))
            if 0 <= point[0] < a.shape[0] and 0 <= point[1] < a.shape[1]:
                out[point] = body[rr, q]
    return out


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    stack = [points.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS:
            p = (r + dr, c + dc)
            if p in points:
                points.remove(p)
                stack.append(p)
    return not points


def _intervals(column):
    answer, t = [], 0
    while t < len(column):
        if not column[t]:
            t += 1
            continue
        start = t
        while t < len(column) and column[t]:
            t += 1
        answer.append((start, t))
    return answer


def _color_column(rng, mask, palette, max_blocks, max_period):
    intervals = _intervals(mask)
    minimum = sum(math.ceil((b - a) / max_period) for a, b in intervals)
    if minimum > max_blocks:
        return None
    out = np.full(len(mask), -1, dtype=int)
    available_extra = max_blocks - minimum
    last_color = None
    for a, b in intervals:
        count = math.ceil((b - a) / max_period)
        if available_extra and b - a > count and rng.random() < .48:
            count += 1
            available_extra -= 1
        lengths, remaining = [], b - a
        for i in range(count):
            slots = count - i - 1
            low = max(1, remaining - slots * max_period)
            high = min(max_period, remaining - slots)
            length = int(rng.integers(low, high + 1))
            lengths.append(length)
            remaining -= length
        cursor = a
        for length in lengths:
            options = [v for v in palette if v != last_color]
            color = int(rng.choice(options))
            out[cursor:cursor + length] = color
            cursor += length
            last_color = color
    return out


def _make_key(rng, bg, label, width, palette, forced=None, max_blocks=2):
    """Compose footprints and stripe partitions; no bank of complete shapes."""
    center = width // 2 - 1
    if forced:
        lengths, colors, gap = forced
        inner_gaps = [int(rng.integers(0, 2)) for _ in lengths[1:]]
        depth = sum(lengths) + gap + sum(inner_gaps)
    else:
        depth = int(rng.integers(2, 9 if max_blocks == 2 else 12))
    for attempt in range(32):
        footprint = np.zeros((depth, width), dtype=bool)  # near-first rows
        footprint[:, center:center + 2] = True
        # Independently stepped shoulders remain connected through the far row.
        for q in range(width):
            if q not in (center, center + 1):
                first = int(rng.integers(0, depth))
                if attempt == 31 or q in (0, width - 1) and rng.random() < .60:
                    first = depth - 1
                footprint[first:, q] = True
        if forced:
            footprint[:, center] = False
            cursor = gap
            for i, length in enumerate(lengths):
                if i:
                    cursor += inner_gaps[i - 1]
                footprint[cursor:cursor + length, center] = True
                cursor += length
        if attempt != 31:
            for q in range(width):
                if q in (center, center + 1):
                    continue
                for t in range(1, depth - 1):
                    if footprint[t, q] and rng.random() < .16:
                        changed = footprint.copy()
                        changed[t, q] = False
                        if (_connected(changed) and sum(math.ceil((b - a) / 4)
                                for a, b in _intervals(changed[:, q])) <= max_blocks):
                            footprint = changed
        data = np.full(footprint.shape, bg, dtype=np.int8)
        for q in range(width):
            colored = _color_column(rng, footprint[:, q], palette, max_blocks, 4)
            if colored is None:
                break
            data[colored >= 0, q] = colored[colored >= 0]
        else:
            if forced:
                data[:, center] = bg
                cursor = gap
                for i, (length, color) in enumerate(zip(lengths, colors)):
                    if i:
                        cursor += inner_gaps[i - 1]
                    data[cursor:cursor + length, center] = color
                    cursor += length
            body = np.full((depth + 1, width), bg, dtype=np.int8)
            body[:-1] = data[::-1]
            body[-1, center:center + 2] = label
            attachments = _attachments(body, bg, width)
            if (_connected(body != bg) and len(attachments) == 1
                    and attachments[0]['label'] == label
                    and np.array_equal(attachments[0]['canonical'], body)):
                return body
    raise RuntimeError("key construction exhausted bounded trials")


def _place(rng, shape, hub, bg, pieces):
    a = np.full(shape, bg, dtype=np.int8)
    _draw_hub(a, *hub[:3], hub[3])
    blocked = np.zeros(shape, dtype=bool)
    r, c, n = hub[:3]
    blocked[max(0, r - 1):r + n + 1, max(0, c - 1):c + n + 1] = True
    order = list(rng.permutation(len(pieces)))
    order.sort(key=lambda i: -pieces[i]['body'].size)
    for i in order:
        body = np.rot90(pieces[i]['body'], pieces[i]['input_turns'])
        h, w = body.shape
        if h > shape[0] or w > shape[1]:
            return None
        local = np.argwhere(body != bg)
        tops, lefts = np.indices((shape[0] - h + 1, shape[1] - w + 1))
        rr = tops.ravel()[:, None] + local[:, 0]
        cc = lefts.ravel()[:, None] + local[:, 1]
        choices = np.flatnonzero(~np.any(blocked[rr, cc], axis=1))
        if not len(choices):
            return None
        choice = int(rng.choice(choices))
        rows, cols = rr[choice], cc[choice]
        a[rows, cols] = body[local[:, 0], local[:, 1]]
        for row, col in zip(rows, cols):
            blocked[max(0, row - 1):row + 2, max(0, col - 1):col + 2] = True
    return a


def _plans(rng, ntrain, ntest, witness_colors, shift):
    plans = []
    for _ in range(ntrain):
        count = int(rng.integers(1, 5))
        plans.append(dict(sides=set(map(int, rng.choice(4, count, replace=False))),
                          forced={}, turns={}, distractors=int(rng.integers(0, 3)), shift='train'))
    slots = list(map(int, rng.choice(ntrain, 2, replace=False)))
    sides = list(map(int, rng.permutation(4)))
    rotations = list(map(int, rng.permutation(4)))
    witnesses = []
    for i, side in enumerate(sides):
        slot = slots[i // 2]
        plans[slot]['sides'].add(side)
        plans[slot]['turns'][side] = rotations[i]
        witnesses.append((slot, side))
    x, y = witness_colors
    for (slot, side), ls, cs in zip((witnesses[0], witnesses[2]),
                                   ((2, 3), (3, 2)), ((x, y), (y, x))):
        plans[slot]['forced'][side] = (ls, cs, 1)
    # Separate non-competing lanes establish numerical recurrence as well.
    for (slot, side), length, color in zip((witnesses[1], witnesses[3]), (1, 4), (x, y)):
        plans[slot]['forced'][side] = ((length,), (color,), 0)
    plans[slots[0]]['distractors'] = max(1, plans[slots[0]]['distractors'])
    for _ in range(ntest):
        count = int(rng.integers(1, 4 if shift == 'combine' else 5))
        chosen = set(map(int, rng.choice(4, count, replace=False)))
        plans.append(dict(sides=chosen, forced={}, turns={},
                          distractors=int(rng.integers(0, 3)), shift=shift,
                          special=int(rng.choice(sorted(chosen)))))
    return plans


def _build_pair(rng, plan, parameters):
    witness_colors = parameters['colors']
    bg = (parameters['background'] if parameters['stable_bg'] else
          int(rng.choice([c for c in range(10) if c not in witness_colors])))
    labels = tuple(map(int, rng.choice([c for c in range(10) if c != bg], 4, replace=False)))
    n = int(rng.choice(parameters['hub_sizes']))
    pool = [c for c in range(10) if c != bg]
    palette = list(map(int, rng.choice(pool, int(rng.integers(2, 7)), replace=False)))
    for color in witness_colors:
        if color not in palette:
            palette.append(color)
    pieces, requirements = [], [0, 0, 0, 0]
    for side in sorted(plan['sides']):
        width = int(rng.choice(list(range(4, n + 1, 2))))
        forced = plan['forced'].get(side)
        if plan['shift'] == 'combine' and side == plan['special']:
            lengths = ((4, 3, 2) if rng.random() < .5 else (3, 4, 2))
            if rng.random() < .30:
                lengths = (4, 3, 2, 1)
            colors = tuple(map(int, rng.choice(pool, len(lengths), replace=False)))
            forced = (lengths, colors, int(rng.integers(0, 2)))
            width = int(rng.choice((4, min(6, n))))
        body = _make_key(rng, bg, labels[side], width, palette, forced,
                         4 if plan['shift'] == 'combine' else 2)
        delta = plan['turns'].get(side, int(rng.integers(4)))
        pieces.append(dict(body=body, input_turns=(-side - delta) % 4))
        requirements[side] = (7 if forced and len(forced[0]) == 2 else 5) if plan['shift'] == 'train' else 0
        if plan['shift'] == 'combine' and side == plan['special']:
            requirements[side] = 5
    absent = [v for v in pool if v not in labels]
    for _ in range(plan['distractors']):
        label = int(rng.choice(absent))
        # A legitimate-looking key, using only absent colors, forces filtering.
        body = _make_key(rng, bg, label, int(rng.choice(list(range(4, n + 1, 2)))),
                         absent, max_blocks=2)
        pieces.append(dict(body=body, input_turns=int(rng.integers(4))))
    min_h, min_w = n + requirements[0] + requirements[2], n + requirements[1] + requirements[3]
    h = int(rng.integers(max(14, min_h), 31))
    w = int(rng.integers(max(14, min_w), 31))
    for attempt in range(22):
        r = int(rng.integers(requirements[0], h - n - requirements[2] + 1))
        c = int(rng.integers(requirements[3], w - n - requirements[1] + 1))
        if plan['shift'] == 'clip':
            distance = int(rng.integers(1, 4))
            side = plan['special']
            if side == 0:
                r = distance
            elif side == 1:
                c = w - n - distance
            elif side == 2:
                r = h - n - distance
            else:
                c = distance
        hub = (r, c, n, labels)
        a = _place(rng, (h, w), hub, bg, pieces)
        if a is not None:
            try:
                info = _inspect(a)
                if len(info['selected']) == len(plan['sides']):
                    return dict(input=a, output=_paint_from_input(a, info))
            except ValueError:
                pass
        if attempt % 2:
            h, w = min(30, h + 2), min(30, w + 2)
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    ntrain = _integer(num_train, 'num_train', 2, 10)
    ntest = _integer(num_test, 'num_test', 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.choice(10, 2, replace=False)))
    parameters = dict(colors=colors,
                      background=int(rng.choice([c for c in range(10) if c not in colors])),
                      stable_bg=bool(rng.random() < .30),
                      hub_sizes=(4, 6) if rng.random() < .25 else (4, 6, 8))
    shift = str(rng.choice(('combine', 'clip')))
    plans = _plans(rng, ntrain, ntest, colors, shift)
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, plan in enumerate(plans):
        split = 'train' if i < ntrain else 'test'
        for _ in range(24):
            pair = _build_pair(rng, plan, parameters)
            if pair is not None and _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("bounded distinct-scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(pairs, infos):
    periods, sides, rotations, precedence = set(), set(), set(), set()
    short, long, shifted, filtering = False, False, False, False
    for pair, info in zip(pairs, infos):
        bg, hub = info['background'], info['hub']
        out = np.asarray(pair['output'])
        filtering |= any(p['side'] is None for p in info['pieces'])
        for p in info['selected']:
            sides.add(p['side'])
            rotations.update(p['rotations'])
            width = p['canonical'].shape[1]
            distance = _distance(out.shape, hub, p['side'])
            for q, lane in enumerate(p['lanes']):
                if len(lane) == 1:
                    b = lane[0]
                    if distance > b['length']:
                        expected = [b['color']] + [bg] * (b['length'] - 1) + [b['color']]
                        cells = [int(out[_fan_cell(hub, p['side'], width, q, d)])
                                 for d in range(1, b['length'] + 2)]
                        if cells == expected:
                            periods.add(b['length'])
                for i, near in enumerate(lane):
                    for far in lane[i + 1:]:
                        if near['color'] == far['color']:
                            continue
                        collision, visible_far = False, False
                        for t in range(distance):
                            hits = [b for b in lane if t % b['length'] == 0]
                            if not hits:
                                continue
                            observed = int(out[_fan_cell(hub, p['side'], width, q, t + 1)])
                            collision |= (t > 0 and near in hits and far in hits and
                                          hits[0] is near and observed == near['color'])
                            visible_far |= hits[0] is far and observed == far['color']
                        if collision and visible_far:
                            short |= near['length'] < far['length']
                            long |= near['length'] > far['length']
                            shifted |= near['near'] > 1
                            precedence.add((near['color'], far['color']))
    return dict(periods=periods, sides=sides, rotations=rotations, short=short,
                long=long, shifted=shifted, filtering=filtering,
                cycle=any((b, a) in precedence for a, b in precedence))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _inspect(inp)
            assert info['selected'], 'each generated scene has an operating key'
            assert np.array_equal(out, _predict(inp, info)), 'incorrect fan arithmetic or registration'
            assert not np.array_equal(inp, out)
            bg, hub = info['background'], info['hub']
            # No cell outside the hub and the selected lane strips may be painted.
            allowed = np.zeros(inp.shape, dtype=bool)
            for cells in _hub_cells(*hub[:3]):
                for point in cells:
                    allowed[point] = True
                    assert out[point] == inp[point], 'hub changed'
            for p in info['selected']:
                width = p['canonical'].shape[1]
                for q, lane in enumerate(p['lanes']):
                    if lane:
                        for d in range(1, _distance(inp.shape, hub, p['side']) + 1):
                            allowed[_fan_cell(hub, p['side'], width, q, d)] = True
                    if split == 'train' and lane:
                        assert len(lane) <= 2, 'training program has too many runs'
                        assert _distance(inp.shape, hub, p['side']) > max(b['length'] for b in lane)
            assert np.all(out[~allowed] == bg), 'loose key or unrelated pixel retained'
            analyses[split].append(info)
    train = analyses['train']
    evidence = _evidence(puzzle['train'], train)
    assert {1, 4} <= evidence['periods'], 'uncontested recurrence witnesses missing'
    assert evidence['sides'] == set(range(4)) and evidence['rotations'] == set(range(4))
    assert evidence['short'] and evidence['long'] and evidence['cycle'], 'priority not identifiable'
    assert evidence['shifted'] and evidence['filtering'], 'phase/filtering witness missing'
    for options in ({'priority': 'short'}, {'priority': 'long'}, {'priority': 'far'},
                    {'period_delta': 1}, {'phase': 'gap'}, {'mirror': True}):
        assert any(not np.array_equal(p['output'], _predict(_grid(p['input']), info, **options))
                   for p, info in zip(puzzle['train'], train)), f'undistinguished alternative {options}'
    assert any(not np.array_equal(p['output'], _dock_only(_grid(p['input']), info))
               for p, info in zip(puzzle['train'], train)), 'A can be omitted'
    # Every held-out scene either increases run interaction or clips a taught fan.
    for info, pair in zip(analyses['test'], puzzle['test']):
        richer = any(len(lane) >= 3 for p in info['selected'] for lane in p['lanes'])
        clipped = any(1 <= _distance(np.asarray(pair['input']).shape, info['hub'], p['side']) <= 3
                      for p in info['selected'])
        assert richer or clipped, 'missing structural train/test shift'


def variation_metrics(puzzles):
    shapes, hubs, widths, depths, footprints, programs = set(), set(), set(), set(), set(), set()
    counts, blocks, gaps, colors, directions, turns = set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _grid(pair['input'])
                info = _inspect(a)
                shapes.add(a.shape)
                hubs.add(info['hub'][:3])
                colors.add(info['background'])
                counts.add((len(info['selected']), len(info['pieces']) - len(info['selected'])))
                for p in info['selected']:
                    body = p['canonical']
                    widths.add(body.shape[1])
                    depths.add(body.shape[0] - 1)
                    footprints.add(_key((body != info['background']).astype(np.int8)))
                    roles = {info['background']: 0}
                    normalized = np.array([roles.setdefault(int(v), len(roles)) for v in body.flat],
                                          dtype=np.int8).reshape(body.shape)
                    programs.add(_key(normalized))
                    directions.add(p['side'])
                    turns.update(p['rotations'])
                    for lane in p['lanes']:
                        blocks.add(len(lane))
                        gaps.update(b['near'] - (lane[i - 1]['near'] + lane[i - 1]['length'] if i else 1)
                                    for i, b in enumerate(lane))
    return dict(puzzles=len(puzzles), grid_shapes=len(shapes), hub_geometries=len(hubs),
                widths=sorted(widths), radial_depths=sorted(depths), footprints=len(footprints),
                normalized_programs=len(programs), count_configurations=sorted(counts),
                run_counts=sorted(blocks), source_gaps=sorted(gaps),
                backgrounds=sorted(colors), sides=sorted(directions), rotations=sorted(turns))


def self_test() -> dict:
    # Literal arithmetic fixture. The gap before the 1-run must not move phase.
    inp = np.zeros((18, 21), dtype=np.int8)
    _draw_hub(inp, 7, 8, 4, (4, 6, 7, 8))
    body = np.array([[5, 2, 3, 5], [0, 2, 3, 0], [0, 2, 3, 0],
                     [0, 1, 3, 0], [0, 1, 3, 0], [0, 0, 3, 0], [0, 4, 4, 0]], dtype=np.int8)
    inp[11:18, :4] = body
    expected = np.zeros_like(inp)
    _draw_hub(expected, 7, 8, 4, (4, 6, 7, 8))
    expected[:7, 8:12] = np.array([[5, 1, 3, 5], [5, 0, 0, 5], [5, 1, 0, 5],
                                  [5, 2, 0, 5], [5, 1, 0, 5], [5, 0, 0, 5], [5, 1, 3, 5]])
    assert np.array_equal(solve(inp), expected)
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(expected, turn))
        moved = np.zeros_like(inp)
        _draw_hub(moved, 7, 8, 4, (4, 6, 7, 8))
        rotated = np.rot90(body, turn)
        moved[11:11 + rotated.shape[0], :rotated.shape[1]] = rotated
        assert np.array_equal(solve(moved), expected), 'initial pose leaked into pulse program'
    permutation = np.array([9, 4, 6, 8, 0, 1, 2, 3, 5, 7], dtype=np.int8)
    assert np.array_equal(solve(permutation[inp]), permutation[expected])
    changed_run = inp.copy()
    changed_run[16, 1] = 1
    assert not np.array_equal(solve(changed_run), expected), 'run lengths have no effect'
    changed_label = inp.copy()
    _draw_hub(changed_label, 7, 8, 4, (6, 4, 7, 8))
    rerouted = solve(changed_label)
    outside_hub = np.ones(inp.shape, dtype=bool)
    outside_hub[7:11, 8:12] = False
    assert not np.array_equal(rerouted[outside_hub], expected[outside_hub]), 'color correspondence has no effect'
    assert np.all(rerouted[:7, 8:12] == 0), 'old port still emits after label exchange'
    assert rerouted[8, 12] == 1 and rerouted[9, 12] == 3, 'program did not rotate onto its new port'
    # Symmetry-equivalent attachment, a wider hub, and a zero-length fan.
    square_input = np.zeros_like(inp)
    _draw_hub(square_input, 7, 8, 6, (4, 6, 7, 8))
    square_input[14:16, :2] = 4
    square_answer = np.zeros_like(inp)
    _draw_hub(square_answer, 7, 8, 6, (4, 6, 7, 8))
    square_answer[:7, 10:12] = 4
    assert np.array_equal(solve(square_input), square_answer)
    zero_input = np.zeros_like(inp)
    _draw_hub(zero_input, 0, 8, 6, (4, 6, 7, 8))
    zero_input[14:16, :2] = 4
    zero_answer = np.zeros_like(inp)
    _draw_hub(zero_answer, 0, 8, 6, (4, 6, 7, 8))
    assert np.array_equal(solve(zero_input), zero_answer)
    # Diagonal wings are one key; empty data lanes stay empty despite the neck.
    wings = np.zeros_like(inp)
    _draw_hub(wings, 7, 8, 4, (4, 6, 7, 8))
    wings[14:16, :4] = [[5, 0, 0, 5], [0, 4, 4, 0]]
    wing_answer = np.zeros_like(inp)
    _draw_hub(wing_answer, 7, 8, 4, (4, 6, 7, 8))
    wing_answer[:7, 8] = 5
    wing_answer[:7, 11] = 5
    assert np.array_equal(solve(wings), wing_answer)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    metrics = variation_metrics(puzzles)
    assert metrics['grid_shapes'] >= 35 and metrics['hub_geometries'] >= 70
    assert metrics['widths'] == [4, 6, 8]
    assert metrics['footprints'] >= 60 and metrics['normalized_programs'] >= 150
    assert set((1, 2, 3, 4)) <= set(metrics['run_counts'])
    assert len(metrics['radial_depths']) >= 8 and len(metrics['count_configurations']) >= 8
    assert metrics['backgrounds'] == list(range(10))
    assert metrics['sides'] == metrics['rotations'] == list(range(4))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(991, nt, nq)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(9, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: _grid(v) for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
    hub = _inspect(damaged['test'][0]['input'])['hub']
    point = _hub_cells(*hub[:3])[0][0]
    damaged['test'][0]['output'][point] = (int(damaged['test'][0]['output'][point]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged hub accepted')
    return metrics


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
