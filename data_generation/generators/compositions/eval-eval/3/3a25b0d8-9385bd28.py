"""Permission blueprints: room correspondence gates ordered rectangle paint.

The public solver reads only the input. Generation constructs answers from
floorplan labels; validation reconstructs rooms, correspondence and layers.
No parent module, stored puzzle, file I/O or training-dependent convention is used.
"""
from __future__ import annotations

import numpy as np


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, points = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    points.append(q)
        parts.append(np.asarray(points, dtype=int))
    return parts


def _bbox(points):
    return tuple(map(int, points.min(0))) + tuple(map(int, points.max(0) + 1))


def _rooms(walls):
    mask = np.pad(~walls, 1, constant_values=True)
    return [p - 1 for p in _components(mask)
            if np.all(p > 0) and np.all(p < np.asarray(mask.shape) - 1)]


def _relations(rooms):
    boxes = [_bbox(p) for p in rooms]
    return [[tuple(-1 if a[k + 2] <= b[k] else
                   1 if b[k + 2] <= a[k] else 0 for k in (0, 1))
             for b in boxes] for a in boxes]


def _match(donor_rooms, target_rooms):
    """Exact orientation-preserving spatial-graph isomorphism, independent of area.

    The family preserves separated/overlapping projections on both axes.
    Ambiguous correspondences are excluded, even if their colors would agree.
    """
    a, b = _relations(donor_rooms), _relations(target_rooms)
    n = len(a)
    assert n == len(b), "room counts differ"
    signatures_a = [tuple(sorted(row)) for row in a]
    signatures_b = [tuple(sorted(row)) for row in b]
    domains = [[j for j in range(n) if signatures_a[i] == signatures_b[j]]
               for i in range(n)]
    answers = []

    def search(mapping, used):
        if len(answers) >= 2:
            return
        if len(mapping) == n:
            answers.append(dict(mapping))
            return
        choices = {i: [j for j in domains[i] if j not in used and all(
            a[i][k] == b[j][v] for k, v in mapping.items())]
            for i in range(n) if i not in mapping}
        i = min(choices, key=lambda k: (len(choices[k]), k))
        for j in choices[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({}, set())
    assert len(answers) == 1, "room correspondence is not unique"
    return answers[0]


def _markers(bounds, diagonal):
    r, c, R, C = bounds  # inclusive extrema
    if diagonal == 0:
        return ({(r, c), (r, c + 1), (r + 1, c)},
                {(R, C), (R, C - 1), (R - 1, C)})
    return ({(r, C), (r, C - 1), (r + 1, C)},
            {(R, c), (R, c + 1), (R - 1, c)})


def _recognize(points):
    assert len(points) == 6, "a layer needs six cue pixels"
    p = np.asarray(sorted(points))
    r, c, RR, CC = _bbox(p)
    bounds = r, c, RR - 1, CC - 1
    h, w = RR - r, CC - c
    assert min(h, w) >= 2 and (min(h, w) > 2 or max(h, w) >= 4)
    for diagonal in (0, 1):
        ends = _markers(bounds, diagonal)
        if not ends[0] & ends[1] and ends[0] | ends[1] == points:
            return bounds, diagonal, ends
    raise AssertionError("cue pixels are not opposite L corners")


def _checked_grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))
    return grid.astype(int, copy=False)


def _parse(input_grid):
    grid = _checked_grid(input_grid)
    bg = int(grid[0, 0])
    assert all(np.all(edge == bg) for edge in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "outer margin is background"
    candidates = []
    for wall in map(int, np.unique(grid)):
        if wall == bg or np.count_nonzero(grid == wall) < 24:
            continue
        parts = _components(grid == wall, diagonal=True)
        if len(parts) != 2:
            continue
        figures = []
        for part in parts:
            r, c, R, C = _bbox(part)
            walls = np.zeros((R - r, C - c), dtype=bool)
            walls[part[:, 0] - r, part[:, 1] - c] = True
            rooms = _rooms(walls)
            raw = grid[r:R, c:C]
            if not 4 <= len(rooms) <= 9:
                break
            values = [np.unique(raw[tuple(room.T)]) for room in rooms]
            donor = all(len(v) == 1 and int(v[0]) not in (bg, wall) for v in values)
            target = all(bg in v for v in values)
            if donor == target:
                break
            figures.append(dict(box=(r, c, R, C), walls=walls, rooms=rooms,
                                raw=raw, donor=donor, values=values))
        if len(figures) == 2 and sum(f['donor'] for f in figures) == 1:
            candidates.append((wall, figures))
    assert len(candidates) == 1, "framework color or roles are ambiguous"
    wall, figures = candidates[0]
    donor = next(f for f in figures if f['donor'])
    target = next(f for f in figures if not f['donor'])
    colors = [int(v[0]) for v in donor['values']]
    assert 2 <= len(set(colors)) <= 3
    correspondence = _match(donor['rooms'], target['rooms'])
    target_colors = [None] * len(colors)
    for i, j in correspondence.items():
        target_colors[j] = colors[i]
    outside = grid.copy()
    for f in figures:
        r, c, R, C = f['box']
        occupied = f['walls'].copy()
        for room in f['rooms']:
            occupied[tuple(room.T)] = True
        outside[r:R, c:C][occupied] = bg
    points = np.argwhere(outside != bg)
    assert len(points), "missing palette"
    r, c, R, C = _bbox(points)
    assert C - c == 2 and 2 <= R - r <= 4
    cues = list(map(int, grid[r:R, c]))
    fills = list(map(int, grid[r:R, c + 1]))
    assert len(set(cues)) == len(cues)
    assert not set(cues) & {bg, wall, *colors}
    assert wall not in fills
    assert all(outside[rr, cc] == grid[rr, cc]
               for rr in range(r, R) for cc in (c, c + 1))
    # A one-cell local moat makes the table identifiable without edge anchoring.
    for rr in range(r - 1, R + 1):
        for cc in range(c - 1, C + 1):
            if not (r <= rr < R and c <= cc < C):
                assert grid[rr, cc] == bg
    t = np.full(target['walls'].shape, bg, dtype=int)
    t[target['walls']] = wall
    room_ids = np.full(t.shape, -1, dtype=int)
    for i, room in enumerate(target['rooms']):
        room_ids[tuple(room.T)] = i
        t[tuple(room.T)] = target['raw'][tuple(room.T)]
        assert set(map(int, target['values'][i])) <= {bg, *cues}
    specs = []
    for cue, fill in zip(cues, fills):
        points = set(map(tuple, np.argwhere(t == cue)))
        bounds, diagonal, ends = _recognize(points)
        hosts = []
        for end in ends:
            ids = {int(room_ids[p]) for p in end}
            assert len(ids) == 1 and -1 not in ids, "each L must be inside one room"
            hosts.append(ids.pop())
        category = target_colors[hosts[0]]
        assert category == target_colors[hosts[1]], "endpoint rooms have different classes"
        specs.append(dict(cue=cue, fill=fill, bounds=bounds, diagonal=diagonal,
                          markers=points, category=category, hosts=hosts))
    return dict(background=bg, wall=wall, donor=donor, target=target,
                correspondence=correspondence, colors=colors,
                target_colors=target_colors, room_ids=room_ids, target_grid=t,
                specs=specs, table=(r, c, R, C))


def _inside(p, bounds):
    r, c = p
    return bounds[0] <= r <= bounds[2] and bounds[1] <= c <= bounds[3]


def _predict(info, *, ignore_gate=False, reverse=False, transparent=False,
             scan_match=False, gate_after_priority=False):
    output = info['target_grid'].copy()
    specs = info['specs'][::-1] if reverse else info['specs']
    categories = info['colors'] if scan_match else info['target_colors']
    if scan_match:
        specs = [dict(s, category=categories[s['hosts'][0]]) for s in specs]
    for p_array in np.argwhere(info['room_ids'] >= 0):
        p = tuple(p_array)
        category = categories[int(info['room_ids'][p])]
        for s in specs:
            if not _inside(p, s['bounds']):
                continue
            if not ignore_gate and category != s['category']:
                if gate_after_priority:
                    break
                continue
            if p in s['markers']:
                output[p] = s['cue']
                break
            if transparent and s['fill'] == info['background']:
                continue
            output[p] = s['fill']
            break
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the target crop from the query alone; no generation state is read."""
    return _predict(_parse(input_grid))


def _plan(rng, count):
    options = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]
    options = [shape for shape in options if shape[0] * shape[1] >= count]
    shape = options[int(rng.integers(len(options)))]
    labels = np.arange(shape[0] * shape[1]).reshape(shape)
    if labels.size > count and rng.random() < .55:
        r, c = [(0, 0), (0, shape[1] - 1), (shape[0] - 1, 0),
                (shape[0] - 1, shape[1] - 1)][int(rng.integers(4))]
        labels[r, c] = -1
    while len(set(labels.flat) - {-1}) > count:
        edges = [(int(labels[r, c]), int(labels[r + dr, c + dc]))
                 for r in range(shape[0]) for c in range(shape[1])
                 for dr, dc in ((1, 0), (0, 1))
                 if r + dr < shape[0] and c + dc < shape[1]
                 and labels[r, c] >= 0 and labels[r + dr, c + dc] >= 0
                 and labels[r, c] != labels[r + dr, c + dc]]
        a, b = edges[int(rng.integers(len(edges)))]
        labels[labels == b] = a
    result = labels.copy()
    for i, old in enumerate(sorted(set(labels.flat) - {-1})):
        result[labels == old] = i
    return result


def _geometry(rng, plan):
    heights = rng.integers(2, 5, size=plan.shape[0])
    widths = rng.integers(2, 5, size=plan.shape[1])
    ys, xs = np.r_[0, np.cumsum(heights + 1)], np.r_[0, np.cumsum(widths + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r in range(plan.shape[0]):
        for c in range(plan.shape[1]):
            k = int(plan[r, c])
            if k < 0:
                continue
            labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = k
            if r and plan[r - 1, c] == k:
                labels[ys[r], xs[c] + 1:xs[c + 1]] = k
            if c and plan[r, c - 1] == k:
                labels[ys[r] + 1:ys[r + 1], xs[c]] = k
            if r and c and np.all(plan[r - 1:r + 1, c - 1:c + 1] == k):
                labels[ys[r], xs[c]] = k
    # Local wall intrusions change starts, area and shape without encoding a
    # texture-stretching rule. Keeping a room connected prevents accidental splits.
    for k in sorted(set(plan.flat) - {-1}):
        points = np.argwhere(labels == k)
        r, c, R, C = _bbox(points)
        trial = labels == k
        draw = rng.random()
        if draw < .25:
            trial[r, :] = False  # may change scan order between neighboring rooms
        elif draw < .55:
            corner = [(r, c), (r, C - 1), (R - 1, c), (R - 1, C - 1)][int(rng.integers(4))]
            trial[corner] = False
        if trial.sum() >= 3 and len(_components(trial)) == 1:
            labels[(labels == k) & ~trial] = -2
    occupied = labels >= 0
    padded = np.pad(occupied, 1)
    near = np.zeros(labels.shape, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]]
    labels[(labels < 0) & near] = -2
    r, c, R, C = _bbox(np.argwhere(labels == -2))
    labels = labels[r:R, c:C]
    rooms = _rooms(labels == -2)
    assert len(rooms) == len(set(plan.flat) - {-1})
    assert all(len(set(map(int, labels[tuple(p.T)]))) == 1
               and int(labels[tuple(p[0])]) >= 0 for p in rooms)
    return labels


def _layer_candidates(labels, colors):
    # Enumerate visibly valid endpoint triplets, then join opposite corners.
    candidates = []
    h, w = labels.shape
    for diagonal in (0, 1):
        top, bottom = [], []
        for r in range(h):
            for c in range(w):
                offsets_top = ((0, 0), (0, 1), (1, 0)) if diagonal == 0 else ((0, 0), (0, -1), (1, 0))
                offsets_bottom = ((0, 0), (0, -1), (-1, 0)) if diagonal == 0 else ((0, 0), (0, 1), (-1, 0))
                for offsets, dest in ((offsets_top, top), (offsets_bottom, bottom)):
                    cells = {(r + dr, c + dc) for dr, dc in offsets}
                    if all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
                        ids = {int(labels[p]) for p in cells}
                        if len(ids) == 1 and min(ids) >= 0:
                            dest.append((r, c, colors[ids.pop()], cells))
        for r, c, category, first in top:
            for R, C, other, last in bottom:
                if category != other or R <= r or (C <= c if diagonal == 0 else C >= c):
                    continue
                bounds = (r, min(c, C), R, max(c, C))
                hh, ww = R - r + 1, abs(C - c) + 1
                if min(hh, ww) == 2 and max(hh, ww) < 4:
                    continue
                if first & last:
                    continue
                markers = first | last
                mask = np.zeros(labels.shape, dtype=bool)
                mask[bounds[0]:bounds[2] + 1, bounds[1]:bounds[3] + 1] = True
                class_mask = np.zeros(labels.shape, dtype=bool)
                for k, color in enumerate(colors):
                    if color == category:
                        class_mask |= labels == k
                mask &= class_mask
                candidates.append(dict(bounds=bounds, diagonal=diagonal, category=category,
                                       markers=markers, mask=mask))
    return candidates


def _choose_layers(rng, candidates, count, multiclass, require_overlap=True):
    assert len(candidates) >= 2
    if not require_overlap:
        disjoint = bool(rng.integers(2))
        for _ in range(60):
            chosen, used = [], set()
            for i in range(count):
                options = [s for s in candidates if not s['markers'] & used]
                if disjoint:
                    options = [s for s in options if all(
                        s['bounds'][2] < t['bounds'][0] or t['bounds'][2] < s['bounds'][0]
                        or s['bounds'][3] < t['bounds'][1] or t['bounds'][3] < s['bounds'][1]
                        for t in chosen)]
                if not options:
                    break
                s = options[int(rng.integers(len(options)))]
                chosen.append(s)
                used |= s['markers']
            if len(chosen) == count:
                return chosen
        # Free scenes still have a bounded readable construction if wholly
        # disjoint placement was infeasible for this particular floorplan.
    for _ in range(240):
        a, b = [candidates[int(i)] for i in rng.integers(len(candidates), size=2)]
        if a['category'] != b['category'] or a['markers'] & b['markers']:
            continue
        if not any(a['mask'][p] for p in b['markers']):
            continue
        common = a['mask'] & b['mask']
        for p in a['markers'] | b['markers']:
            common[p] = False
        if not common.any():
            continue
        chosen, used = [a, b], a['markers'] | b['markers']
        for i in range(2, count):
            options = [s for s in candidates if not s['markers'] & used
                       and (not multiclass or i != 2 or s['category'] != a['category'])]
            if not options:
                break
            s = options[int(rng.integers(len(options)))]
            chosen.append(s)
            used |= s['markers']
        if len(chosen) == count:
            return chosen
    raise AssertionError('no compatible layer selection')


def _embed(rng, donor, target, dlabels, tlabels, bg, rows):
    a, b = np.argwhere(dlabels != -1), np.argwhere(tlabels != -1)
    delta = a[:, None, :] - b[None, :, :]
    blocked = np.zeros((63, 63), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            q = delta.reshape(-1, 2) + (31 + dr, 31 + dc)
            blocked[q[:, 0], q[:, 1]] = True
    dh, dw = donor.shape
    th, tw = target.shape
    layouts = {'side': [], 'stack': [], 'diagonal': [], 'interleaved': []}
    for rr in range(-th - 2, dh + 3):
        for cc in range(-tw - 2, dw + 3):
            h, w = max(dh, rr + th) - min(0, rr), max(dw, cc + tw) - min(0, cc)
            if h > 28 or w > 28 or blocked[rr + 31, cc + 31]:
                continue
            ro = min(dh, rr + th) > max(0, rr)
            co = min(dw, cc + tw) > max(0, cc)
            mode = 'interleaved' if ro and co else 'side' if ro else 'stack' if co else 'diagonal'
            layouts[mode].append((rr, cc, h, w))
    available = [k for k, v in layouts.items() if v]
    assert available, 'figures do not fit'
    for _ in range(24):
        mode = available[int(rng.integers(len(available)))]
        rr, cc, h, w = layouts[mode][int(rng.integers(len(layouts[mode])))]
        H = int(rng.integers(h + 2, 31))
        W = int(rng.integers(w + 2, 31))
        top = int(rng.integers(1, H - h))
        left = int(rng.integers(1, W - w))
        grid = np.full((H, W), bg, dtype=int)
        occupied = np.zeros(grid.shape, dtype=bool)
        for picture, labels, r, c in ((donor, dlabels, top - min(0, rr), left - min(0, cc)),
                                     (target, tlabels, top + max(0, rr), left + max(0, cc))):
            ph, pw = picture.shape
            mask = labels != -1
            grid[r:r + ph, c:c + pw][mask] = picture[mask]
            occupied[r:r + ph, c:c + pw] |= mask
        positions = [(r, c) for r in range(1, H - len(rows)) for c in range(1, W - 2)
                     if not occupied[r - 1:r + len(rows) + 1, c - 1:c + 3].any()]
        if positions:
            r, c = positions[int(rng.integers(len(positions)))]
            grid[r:r + len(rows), c:c + 2] = rows
            return grid
    raise AssertionError('no free palette position')


def _compose(target, labels, colors, specs):
    output = target.copy()
    for s in reversed(specs):
        output[s['mask']] = s['fill']
        for p in s['markers']:
            output[p] = s['cue']
    return output


def _witnesses(info, output):
    found = set()
    if not np.array_equal(output, _predict(info, ignore_gate=True)):
        found.add('gate')
    if not np.array_equal(output, _predict(info, reverse=True)):
        found.add('priority')
    if not np.array_equal(output, _predict(info, transparent=True)):
        found.add('erase')
    if not np.array_equal(output, _predict(info, scan_match=True)):
        found.add('scan')
    if not np.array_equal(output, _predict(info, gate_after_priority=True)):
        found.add('bypass')
    if len({s['category'] for s in info['specs']}) >= 2:
        found.add('multiclass')
    if any(output[p] != s['cue'] for s in info['specs'] for p in s['markers']):
        found.add('occlusion')
    if any(output[p] == s['cue'] and s['fill'] != s['cue']
           for s in info['specs'] for p in s['markers']):
        found.add('retention')
    if any(s['fill'] != info['background'] and any(
            _inside(tuple(p), s['bounds']) and tuple(p) not in s['markers']
            and output[tuple(p)] == s['fill'] for p in np.argwhere(info['room_ids'] >= 0))
           for s in info['specs']):
        found.add('paint')
    for i, room in enumerate(info['target']['rooms']):
        category = info['target_colors'][i]
        untouched = any(info['target_grid'][tuple(p)] == info['background'] and not any(
            _inside(tuple(p), s['bounds']) and s['category'] == category
            for s in info['specs']) for p in room)
        painted = any(output[tuple(p)] != info['target_grid'][tuple(p)]
                      and output[tuple(p)] != info['background'] for p in room)
        if untouched and painted:
            found.add('partial')
    transferred = info['target_grid'].copy()
    for i, room in enumerate(info['target']['rooms']):
        for p in map(tuple, room):
            if transferred[p] == info['background']:
                transferred[p] = info['target_colors'][i]
    if not np.array_equal(output, transferred):
        found.add('rectangles')
    return found


def _make_pair(rng, room_count, layer_count, required):
    for attempt in range(180):
        try:
            palette = list(map(int, rng.permutation(10)))
            bg, wall = palette[:2]
            richness = int(rng.integers(2, min(3, room_count - 2, 8 - layer_count) + 1))
            classes = palette[2:2 + richness]
            # A repeated class supports windows spanning separate homologous rooms.
            colors = [classes[0]] * 3 + classes[1:]
            colors += [int(rng.choice(classes)) for _ in range(room_count - len(colors))]
            rng.shuffle(colors)
            plan = _plan(rng, room_count)
            dlabels, tlabels = _geometry(rng, plan), _geometry(rng, plan)
            dr, tr = _rooms(dlabels == -2), _rooms(tlabels == -2)
            mapping = _match(dr, tr)
            assert all(int(dlabels[tuple(dr[i][0])]) == int(tlabels[tuple(tr[j][0])])
                       for i, j in mapping.items())
            assert not np.array_equal(dlabels == -2, tlabels == -2)
            candidates = _layer_candidates(tlabels, colors)
            specs = _choose_layers(rng, candidates, layer_count, 'multiclass' in required,
                                   require_overlap=bool(required))
            cues = palette[2 + richness:2 + richness + layer_count]
            paints = [c for c in range(10) if c != wall]
            for s, cue in zip(specs, cues):
                s['cue'] = cue
                s['fill'] = int(rng.choice(paints))
            specs[0]['fill'] = bg if 'erase' in required else int(rng.choice(paints))
            specs[1]['fill'] = int(rng.choice([c for c in paints if c not in (bg, specs[0]['fill'], cues[1])]))
            # Randomly interleave extra layers; the selected upper layer stays
            # earlier than its lower partner but has no fixed absolute rank.
            order = list(map(int, rng.permutation(layer_count)))
            if order.index(0) > order.index(1):
                i, j = order.index(0), order.index(1)
                order[i], order[j] = order[j], order[i]
            specs = [specs[i] for i in order]
            donor = np.full(dlabels.shape, bg, dtype=int)
            target = np.full(tlabels.shape, bg, dtype=int)
            donor[dlabels == -2] = wall
            target[tlabels == -2] = wall
            for i, color in enumerate(colors):
                donor[dlabels == i] = color
            for s in specs:
                for p in s['markers']:
                    target[p] = s['cue']
            assert all(np.any(target[tlabels == i] == bg) for i in range(room_count))
            output = _compose(target, tlabels, colors, specs)
            assert np.any(output != target), 'no visible paint'
            rows = [[s['cue'], s['fill']] for s in specs]
            inp = _embed(rng, donor, target, dlabels, tlabels, bg, rows)
            info = _parse(inp)
            assert np.array_equal(output, _predict(info))
            assert required <= _witnesses(info, output)
            return {'input': inp, 'output': output}
        except AssertionError:
            continue
    return _fallback_pair(rng, room_count, layer_count, required)


def _fallback_pair(rng, room_count, layer_count, required):
    """Closed, fitting construction after bounded random spatial rejection.

    The first four rooms form a 2x2 motif; remaining rooms extend the blueprint.
    Shifted windows cross a forbidden upper-right room, clear lower paint,
    hide a lower cue, and leave some eligible cells outside both rectangles.
    """
    columns = (room_count + 1) // 2
    plan = np.arange(2 * columns).reshape(2, columns)
    plan[plan >= room_count] = -1

    def regular(cell_size):
        step = cell_size + 1
        labels = np.full((2 * step + 1, columns * step + 1), -1, dtype=int)
        for r in range(2):
            for c in range(columns):
                if plan[r, c] >= 0:
                    labels[r * step + 1:(r + 1) * step,
                           c * step + 1:(c + 1) * step] = plan[r, c]
        near = np.zeros(labels.shape, dtype=bool)
        padded = np.pad(labels >= 0, 1)
        for dr in range(3):
            for dc in range(3):
                near |= padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]]
        labels[(labels < 0) & near] = -2
        return labels

    dlabels, tlabels = regular(3), regular(4)
    dlabels[1, 1:4] = -2  # donor's first room is discovered after its right neighbor
    palette = list(map(int, rng.permutation(10)))
    bg, wall, a, b = palette[:4]
    colors = [a] * room_count
    colors[1] = b
    cues = palette[4:4 + layer_count]
    paints = palette[4 + layer_count:6 + layer_count]
    endpoints = [((1, 1, 8, 8), 0, a, bg), ((2, 2, 9, 9), 0, a, paints[0])]
    if layer_count >= 3:
        endpoints.append(((1, 6, 4, 9), 0, b, paints[1]))
    if layer_count == 4:
        endpoints.append(((6, 1, 9, 4), 1, a, paints[1]))
    donor, target = [np.full(labels.shape, bg, dtype=int) for labels in (dlabels, tlabels)]
    donor[dlabels == -2] = wall
    target[tlabels == -2] = wall
    for i, color in enumerate(colors):
        donor[dlabels == i] = color
    specs = []
    for cue, (bounds, diagonal, category, fill) in zip(cues, endpoints):
        ends = _markers(bounds, diagonal)
        markers = ends[0] | ends[1]
        mask = np.zeros(tlabels.shape, dtype=bool)
        mask[bounds[0]:bounds[2] + 1, bounds[1]:bounds[3] + 1] = True
        allowed = np.zeros(tlabels.shape, dtype=bool)
        for i, color in enumerate(colors):
            if color == category:
                allowed |= tlabels == i
        mask &= allowed
        specs.append(dict(cue=cue, fill=fill, category=category, bounds=bounds,
                          diagonal=diagonal, markers=markers, mask=mask))
        for p in markers:
            target[p] = cue
    output = _compose(target, tlabels, colors, specs)
    height = donor.shape[0] + target.shape[0] + 4
    width = max(target.shape[1] + 2, donor.shape[1] + 5)
    inp = np.full((height, width), bg, dtype=int)
    inp[1:1 + donor.shape[0], 1:1 + donor.shape[1]] = donor
    row = donor.shape[0] + 3
    inp[row:row + target.shape[0], 1:1 + target.shape[1]] = target
    col = donor.shape[1] + 2
    inp[2:2 + layer_count, col:col + 2] = [[s['cue'], s['fill']] for s in specs]
    info = _parse(inp)
    assert np.array_equal(output, _predict(info))
    assert required <= _witnesses(info, output), 'fallback does not teach its obligations'
    return {'input': inp, 'output': output}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Create one coherent puzzle, with a room-count or layer-count test shift."""
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['rooms', 'layers']))
    obligations = [set() for _ in range(num_train)]
    first, second = map(int, rng.choice(num_train, size=2, replace=False))
    obligations[first] = {'gate', 'priority', 'erase', 'occlusion', 'retention', 'paint', 'partial'}
    obligations[second] = {'multiclass', 'scan', 'gate', 'paint', 'bypass'}
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            required = obligations[index] if split == 'train' else {'gate', 'priority', 'paint'}
            if shift == 'rooms':
                n = int(rng.integers(4, 7) if split == 'train' else rng.integers(7, 10))
            else:
                n = int(rng.integers(4, 10))
            m = 4 if shift == 'layers' and split == 'test' else int(rng.integers(2, 4))
            if 'multiclass' in required:
                m = max(3, m)
            for _ in range(8):
                pair = _make_pair(rng, n, m, required)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('duplicate construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, witnessed, train_counts, test_counts = set(), set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, output = _checked_grid(pair['input']), _checked_grid(pair['output'])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _parse(inp)
            t = info['target_grid']
            assert output.shape == t.shape
            interior = info['room_ids'] >= 0
            assert np.array_equal(output[~interior], t[~interior]), 'wall or exterior changed'
            assert np.array_equal(output, _predict(info)), 'incorrect class-gated compositor'
            assert np.any((output != t) & interior), 'no visible paint'
            for p in np.argwhere(interior):
                p = tuple(p)
                if output[p] != t[p]:
                    category = info['target_colors'][int(info['room_ids'][p])]
                    assert any(_inside(p, s['bounds']) and s['category'] == category for s in info['specs'])
            found = _witnesses(info, output)
            if split == 'train':
                witnessed |= found
                train_counts.append((len(info['colors']), len(info['specs'])))
            else:
                test_counts.append((len(info['colors']), len(info['specs'])))
                assert {'gate', 'priority', 'paint'} <= found
    assert {'gate', 'priority', 'erase', 'occlusion', 'retention', 'paint', 'multiclass',
            'scan', 'bypass', 'partial', 'rectangles'} <= witnessed
    assert (min(n for n, m in test_counts) > max(n for n, m in train_counts)
            or min(m for n, m in test_counts) > max(m for n, m in train_counts)), 'missing structural shift'


def self_test():
    """Bounded 42-seed sweep, semantic counterfactuals and API/corruption checks."""
    structures, shapes, palettes, room_counts, layer_counts = set(), set(), set(), set(), set()
    layouts, signs, diagonals, changed_order, nonrectangular = set(), set(), set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                roles = {}
                norm = np.asarray([[roles.setdefault(int(c), len(roles)) for c in row]
                                   for row in pair['input']], dtype=np.int8)
                structures.add(_key(norm))
                shapes.add(pair['input'].shape)
                palettes.add((info['background'], info['wall']))
                room_counts.add(len(info['colors']))
                layer_counts.add(len(info['specs']))
                diagonals.update(s['diagonal'] for s in info['specs'])
                db, tb = info['donor']['box'], info['target']['box']
                ro = min(db[2], tb[2]) > max(db[0], tb[0])
                co = min(db[3], tb[3]) > max(db[1], tb[1])
                layouts.add('interleaved' if ro and co else 'side' if ro else 'stack' if co else 'diagonal')
                for i, j in info['correspondence'].items():
                    a, b = info['donor']['rooms'][i], info['target']['rooms'][j]
                    signs.add(int(np.sign(len(b) - len(a))))
                    changed_order += i != j
                    for room in (a, b):
                        r, c, R, C = _bbox(room)
                        nonrectangular += len(room) != (R - r) * (C - c)
    assert len(structures) >= 180 and len(shapes) >= 30 and len(palettes) >= 40
    assert room_counts == set(range(4, 10)) and layer_counts == {2, 3, 4}
    assert {'side', 'stack'} <= layouts and diagonals == {0, 1}
    assert signs == {-1, 0, 1} and changed_order >= 30 and nonrectangular >= 100
    for n in range(4, 10):
        for m in (2, 3, 4):
            required = {'gate', 'priority', 'erase', 'occlusion', 'retention', 'paint'}
            if m < 4:
                required.add('partial')
            if m >= 3:
                required.update({'multiclass', 'scan', 'bypass'})
            _fallback_pair(np.random.default_rng(2000 + 10 * n + m), n, m, required)
    a, b = generate_puzzle(991, 2, 1), generate_puzzle(991, 2, 1)
    assert all(_key(p[k]) == _key(q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    for counts in ((10, 4), (5, 3), (10, 1)):
        p = generate_puzzle(992, *counts)
        assert tuple(len(p[s]) for s in ('train', 'test')) == counts
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    corrupted = generate_puzzle(993, 2, 1)
    out = corrupted['test'][0]['output']
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        return {'seeds': 42, 'pairs': 210, 'structures': len(structures), 'shapes': len(shapes),
                'role_palettes': len(palettes), 'layouts': sorted(layouts),
                'scan_reordered_rooms': changed_order, 'nonrectangular_rooms': nonrectangular}
    raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    print(self_test())
