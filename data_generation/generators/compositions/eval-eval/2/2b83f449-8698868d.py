"""Cavity-resonant switches: input-visible keys condition track rewiring.

Only standard-library code and NumPy are used. No parent module is imported.
The solver reads keys, gates and initial lane occupancies from its argument.
"""
from __future__ import annotations

import json
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_HOST_SEEDS = (0, 303912197, 88474818, 15607604, 598502767, 173800904,
               747151881, 400903107, 39867968, 436692195, 420338273,
               348550295, 332499331, 913486995, 342712373, 744053458,
               374256585, 523378290, 261309430, 798209177, 74830101,
               401677890, 358818524, 705438102, 384590329, 330230457,
               374807988, 827302165, 454821474, 30333252, 482255466,
               532018015, 174440028, 275047867, 343053750, 16194661,
               484921698, 871442631, 439254718, 226316165, 91332746,
               758191684)


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _STEPS
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _border(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _paths(mask):
    """Order every simple path; reject branches, loops and gravity traps."""
    answer = []
    for cells in _components(mask):
        cell_set = set(cells)
        neighbors = {p: [q for dr, dc in _STEPS
                         if (q := (p[0] + dr, p[1] + dc)) in cell_set]
                     for p in cells}
        assert all(len(v) in (1, 2) for v in neighbors.values())
        ends = sorted(p for p in cells if len(neighbors[p]) == 1)
        assert len(ends) == 2
        order, prev, cur = [ends[0]], None, ends[0]
        while cur != ends[1]:
            nxt = [p for p in neighbors[cur] if p != prev]
            assert len(nxt) == 1
            prev, cur = cur, nxt[0]
            order.append(cur)
        assert len(order) == len(cells)
        vertical = {b[0] - a[0] for a, b in zip(order, order[1:]) if a[0] != b[0]}
        assert len(vertical) <= 1
        answer.append(order)
    return answer


def _rewire(core, background, track, bead, gates):
    structural = core.copy()
    structural[structural == bead] = track
    writes = {}
    for gate in gates:
        if not gate['active']:
            continue
        r, c, color = gate['row'], gate['col'], gate['color']
        for y in range(r - 1, r + 2):
            for x in range(c - 1, c + 2):
                value = color if x == c else track
                assert (y, x) not in writes or writes[y, x] == value
                if y != r:
                    assert core[y, x] in (track, bead), 'turn must not erase a break'
                writes[y, x] = value
    for p, value in writes.items():
        structural[p] = value
    paths = _paths(structural == track)
    ends = {p for path in paths for p in (path[0], path[-1])}
    beads = set(map(tuple, np.argwhere(core == bead).tolist()))
    assert beads <= ends
    for r, c in ends:
        assert r % 2 == 0
        assert c in (0, core.shape[1] - 1) or background in (core[r, c - 1], core[r, c + 1])
    return structural, paths, beads


def _settle(structural, paths, beads, bead):
    out = structural.copy()
    targets = set()
    single = collision = level = 0
    for path in paths:
        initial = beads.intersection(path)
        assert initial <= {path[0], path[-1]} and len(initial) <= 2
        if path[0][0] == path[-1][0]:
            final = initial
            level += bool(initial)
        else:
            inward = path[::-1] if path[-1][0] > path[0][0] else path
            final = set(inward[:len(initial)])
            single += len(initial) == 1 and initial != final
            collision += len(initial) == 2
        targets.update(final)
        for p in final:
            out[p] = bead
    assert len(targets) == len(beads)
    return out, {'single': single, 'collision': collision, 'level': level,
                 'targets': targets, 'moved': targets != beads}


def _read_input(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    assert np.all(_border(grid) == bg), 'one blank perimeter is required'
    keys, reserved = {}, np.zeros(grid.shape, dtype=bool)
    # Each key is a single-colored, connected square foreground with a full
    # perimeter. Lanes, beads and three-cell bars cannot have this geometry.
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for cells in _components(grid == color):
            r0, r1, c0, c1 = _box(cells)
            tile = grid[r0:r1, c0:c1]
            if not (5 <= r1 - r0 <= 8 and r1 - r0 == c1 - c0):
                continue
            if not (np.all(_border(tile) == color)
                    and set(map(int, np.unique(tile))) <= {bg, color}):
                continue
            assert color not in keys, 'one key per gate color'
            mask = tile == bg
            cavities = _components(mask)
            assert len(cavities) <= 4
            assert len(_components(~mask)) == 1, 'key foreground must stay connected'
            keys[color] = {'count': len(cavities), 'area': int(mask.sum()),
                           'eight': len(_components(mask, True)), 'mask': mask.copy(),
                           'sizes': tuple(sorted(map(len, cavities))),
                           'box': (r0, r1, c0, c1)}
            reserved[r0:r1, c0:c1] = True
    assert 2 <= len(keys) <= 5
    assert len({k['count'] for k in keys.values()}) == len(keys)
    remaining = (grid != bg) & ~reserved
    locations = np.argwhere(remaining)
    assert len(locations)
    r0, c0 = map(int, locations.min(axis=0))
    r1, c1 = map(int, locations.max(axis=0) + 1)
    assert not np.any(reserved[r0:r1, c0:c1])
    core = grid[r0:r1, c0:c1].copy()
    colors = set(map(int, np.unique(core))) - {bg} - set(keys)
    assert len(colors) == 2
    # Terminal beads are singletons; the track has runs of at least two cells.
    lengths = {c: max(map(len, _components(core == c))) for c in colors}
    track = max(colors, key=lambda c: lengths[c])
    bead = next(iter(colors - {track}))
    assert lengths[track] >= 2 and lengths[bead] == 1
    h, w = core.shape
    assert 5 <= h <= 15 and h % 2 == 1 and 9 <= w <= 28
    assert np.all(np.isin(core[::2], [bg, track, bead]))
    assert np.all(np.isin(core[1::2], [bg] + list(keys)))
    assert np.all(np.isin(core[::2, [0, w - 1]], [track, bead]))
    occupancy = [int(np.count_nonzero(row == bead)) for row in core[::2]]
    assert all(n <= 2 for n in occupancy)
    assert all(c in (0, w - 1) for r, c in np.argwhere(core == bead))
    for r in range(0, h, 2):
        breaks = np.flatnonzero(core[r] == bg).tolist()
        assert all(2 <= x <= w - 3 for x in breaks)
        assert all(b - a >= 3 for a, b in zip(breaks, breaks[1:]))
    gates = []
    for r in range(1, h, 2):
        x = 0
        while x < w:
            color = int(core[r, x])
            if color == bg:
                x += 1
                continue
            end = x + 1
            while end < w and core[r, end] == color:
                end += 1
            assert end - x == 3 and 1 <= x and end <= w - 1
            assert x == 0 or core[r, x - 1] == bg
            assert end == w or core[r, end] == bg
            total = occupancy[(r - 1) // 2] + occupancy[(r + 1) // 2]
            gates.append({'row': r, 'col': x + 1, 'color': color,
                          'total': total, 'threshold': keys[color]['count'],
                          'active': total == keys[color]['count']})
            x = end
    assert gates and set(g['color'] for g in gates) == set(keys)
    # All objects have disjoint boxes with an entire background row or column
    # between their boxes; spatial proximity never supplies a key association.
    boxes = [(r0, r1, c0, c1)] + [k['box'] for k in keys.values()]
    for i, (a, b, c, d) in enumerate(boxes):
        for e, f, g, j in boxes[i + 1:]:
            assert b < e or f < a or d < g or j < c
    structural, paths, beads = _rewire(core, bg, track, bead, gates)
    return {'grid': grid, 'core': core, 'box': (r0, r1, c0, c1),
            'bg': bg, 'track': track, 'bead': bead, 'keys': keys,
            'gates': gates, 'occupancy': occupancy, 'structural': structural,
            'paths': paths, 'beads': beads}


def _prediction(info):
    core_out, movement = _settle(info['structural'], info['paths'], info['beads'], info['bead'])
    out = info['grid'].copy()
    r0, r1, c0, c1 = info['box']
    out[r0:r1, c0:c1] = core_out
    return out, movement


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only this input; activity is decided before any bead is moved."""
    return _prediction(_read_input(input_grid))[0]


def _choice(rng, items):
    return items[int(rng.integers(len(items)))]


def _transforms(mask):
    return [np.rot90(mask, k) for k in range(4)] + [np.rot90(mask.T, k) for k in range(4)]


def _cavities(rng, size, count, witness=False):
    mask = np.zeros((size, size), dtype=bool)
    if not count:
        return mask
    if count == 1 and (witness or rng.random() < .35):
        mask[1:-1, 1:-1] = True
        return mask
    if count == 2 and witness:
        mask[1, 1] = mask[2, 2] = mask[2, 3] = True
    else:
        for _ in range(24):
            mask[:] = False
            for index in rng.permutation((size - 2) ** 2):
                r, c = divmod(int(index), size - 2)
                r, c = r + 1, c + 1
                if any(mask[r + dr, c + dc] for dr, dc in _STEPS):
                    continue
                mask[r, c] = True
                if len(_components(~mask)) != 1:
                    mask[r, c] = False
                if int(mask.sum()) == count:
                    break
            if int(mask.sum()) == count:
                break
        else:
            mask[:] = False
            corners = ((1, 1), (1, size - 2), (size - 2, 1), (size - 2, size - 2))
            for p in corners[:count]:
                mask[p] = True
    target = int(rng.integers(count, (size - 2) ** 2 + 1))
    if witness:
        target = int(rng.integers(3, 8))
    for _ in range((size - 2) ** 2):
        if int(mask.sum()) >= target:
            break
        candidates = []
        for r in range(1, size - 1):
            for c in range(1, size - 1):
                if mask[r, c] or not any(mask[r + dr, c + dc] for dr, dc in _STEPS):
                    continue
                trial = mask.copy()
                trial[r, c] = True
                if len(_components(trial)) != count or len(_components(~trial)) != 1:
                    continue
                if witness and len(_components(trial, True)) != 1:
                    continue
                candidates.append((r, c))
        if not candidates:
            break
        mask[_choice(rng, candidates)] = True
    return _choice(rng, _transforms(mask)).copy()


def _safe_breaks(rng, lanes, width, centers):
    cuts = []
    for lane in range(lanes):
        r = 2 * lane
        ports = sorted((c, y) for y, c in centers if abs(y - r) == 1)
        for (a, side_a), (b, side_b) in zip(ports, ports[1:]):
            if side_a == side_b:
                cuts.append((r, int(rng.integers(a + 2, b - 1))))
    for _ in range(int(rng.integers(4))):
        candidates = [(r, c) for r in range(0, 2 * lanes - 1, 2)
                      for c in range(2, width - 2)
                      if all(abs(c - x) >= 2 for y, x in centers if abs(y - r) == 1)
                      and all(abs(c - x) >= 3 for y, x in cuts if y == r)]
        if candidates:
            cuts.append(_choice(rng, candidates))
    return cuts


def _core_candidate(rng, counts, bg, track, bead, color_by_count, kind):
    witness = kind == 'witness'
    lanes = int(rng.integers(4, 7)) if witness or kind == 'both' else int(rng.integers(3, 7))
    if len(counts) >= 4:
        lanes = max(4, lanes)
    width = int(rng.integers(15 if witness else 12, 28))
    occupancy = list(map(int, rng.integers(0, 3, lanes)))
    if witness:
        j = int(rng.integers(lanes - 3))
        occupancy[j:j + 4] = [1, 0, 2, 1]
    elif kind == 'motion':
        lanes = max(lanes, 4)
        occupancy = list(map(int, rng.integers(0, 3, lanes)))
        j = int(rng.integers(lanes - 3))
        occupancy[j:j + 4] = [0, 1, 1, 0]
    elif kind in ('zero', 'four', 'both'):
        if kind == 'both':
            j = int(rng.integers(lanes - 3))
            block = [0, 0, 2, 2] if rng.random() < .5 else [2, 2, 0, 0]
            occupancy[j:j + 4] = block
        else:
            j = int(rng.integers(lanes - 1))
            occupancy[j:j + 2] = [0, 0] if kind == 'zero' else [2, 2]
            if kind == 'zero' and not sum(occupancy):
                occupancy[(j + 2) % lanes] = 1
    totals = [a + b for a, b in zip(occupancy, occupancy[1:])]
    active = []
    if witness:
        a = int(rng.integers(2, width - 5))
        b = int(rng.integers(a + 2, width - 2))
        if rng.random() < .5:
            a, b = width - 1 - a, width - 1 - b
        active = [(2 * j + 1, a), (2 * j + 3, b)]
    else:
        previous = []
        for i, total in enumerate(totals):
            chosen = []
            wanted = int(rng.integers(0, min(3, (width - 1) // 4) + 1)) if total in counts else 0
            if kind == 'motion' and i == j + 1:
                wanted = max(1, wanted)
            if kind in ('zero', 'four', 'both') and total in (0, 4) and total in counts:
                wanted = max(1, wanted)
            for c in rng.permutation(np.arange(2, width - 2)):
                c = int(c)
                if len(chosen) >= wanted:
                    break
                if all(abs(c - x) >= 2 for x in previous) and all(abs(c - x) >= 4 for x in chosen):
                    chosen.append(c)
            active.extend((2 * i + 1, c) for c in chosen)
            previous = chosen
        if not active:
            rows = [i for i, total in enumerate(totals) if total in counts]
            if not rows:
                return None
            active = [(2 * _choice(rng, rows) + 1, int(rng.integers(2, width - 2)))]
    cuts = _safe_breaks(rng, lanes, width, active)
    core = np.full((2 * lanes - 1, width), bg, dtype=np.int8)
    core[::2] = track
    for p in cuts:
        core[p] = bg
    records = []
    for r, c in active:
        k = totals[(r - 1) // 2]
        core[r, c - 1:c + 2] = color_by_count[k]
        records.append({'row': r, 'col': c, 'color': color_by_count[k], 'active': True})
    used = {totals[(r - 1) // 2] for r, c in active}
    needs = [k for k in counts if k not in used]
    if witness:
        needs = [1, 2] + needs  # The same references must stay inactive too.
    elif not needs:
        needs = [_choice(rng, counts)]
    needs += [_choice(rng, counts) for _ in range(int(rng.integers(3)))]
    for k in needs:
        target = {1: 3, 2: 1}.get(k) if witness else None
        places = [(2 * i + 1, c) for i, total in enumerate(totals)
                  if total != k and (target is None or total == target)
                  for c in range(2, width - 2)
                  if all(abs(c - g['col']) >= 4 for g in records if g['row'] == 2 * i + 1)]
        if not places:
            return None
        r, c = _choice(rng, places)
        core[r, c - 1:c + 2] = color_by_count[k]
        records.append({'row': r, 'col': c, 'color': color_by_count[k], 'active': False})
    # Endpoint choice preserves lane totals and hence all activity decisions.
    for i, n in enumerate(occupancy):
        sides = [0, width - 1] if n == 2 else ([_choice(rng, [0, width - 1])] if n else [])
        for c in sides:
            core[2 * i, c] = bead
    try:
        structural, paths, beads = _rewire(core, bg, track, bead, records)
    except AssertionError:
        return None
    if witness:
        # Put the lone bead on the path leading to an occupied double lane.
        r = 2 * j
        for c in (0, width - 1):
            path = next(p for p in paths if (r, c) in p)
            if any(p[0] == 2 * (j + 2) for p in (path[0], path[-1])):
                core[r, :] = np.where(core[r] == bead, track, core[r])
                core[r, c] = bead
                break
        structural, paths, beads = _rewire(core, bg, track, bead, records)
    out, movement = _settle(structural, paths, beads, bead)
    if witness and not (movement['collision'] and movement['level'] and movement['moved']):
        return None
    if kind == 'motion' and not (movement['single'] and movement['level']):
        return None
    return core, out


def _origins(boxes, height, width):
    rr, cc = np.indices((29 - height, 29 - width))
    free = np.ones(rr.shape, dtype=bool)
    for r, c, h, w in boxes:
        free &= (rr + height < r) | (r + h < rr) | (cc + width < c) | (c + w < cc)
    return np.argwhere(free)


def _place(rng, bg, core, core_out, motifs):
    h, w = core.shape
    objects = [core] + [motifs[int(i)] for i in rng.permutation(len(motifs))]
    boxes = None
    for _ in range(24):
        r, c = int(rng.integers(29 - h)), int(rng.integers(29 - w))
        trial = [(r, c, h, w)]
        for motif in objects[1:]:
            s = len(motif)
            free = _origins(trial, s, s)
            if not len(free):
                break
            rr, cc = map(int, free[int(rng.integers(len(free)))])
            trial.append((rr, cc, s, s))
        if len(trial) == len(objects):
            boxes = trial
            break
    if boxes is None:
        # Proven shelf fallback: at most five <=7-cell motifs use at most
        # two shelves (15 rows), under an <=11-row network: total <=27.
        boxes = [(0, 0, h, w)]
        row, col, shelf_height = h + 1, 0, 0
        for motif in objects[1:]:
            s = len(motif)
            if col + s > 28:
                row, col, shelf_height = row + shelf_height + 1, 0, 0
            boxes.append((row, col, s, s))
            col += s + 1
            shelf_height = max(shelf_height, s)
        assert max(r + bh for r, c, bh, bw in boxes) <= 28
    inp = np.full((30, 30), bg, dtype=np.int8)
    out = inp.copy()
    boxes = [(r + 1, c + 1, bh, bw) for r, c, bh, bw in boxes]
    for index, ((r, c, bh, bw), obj) in enumerate(zip(boxes, objects)):
        inp[r:r + bh, c:c + bw] = obj
        out[r:r + bh, c:c + bw] = core_out if index == 0 else obj
    lo_r, lo_c = min(b[0] for b in boxes), min(b[1] for b in boxes)
    hi_r, hi_c = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    a, c = int(rng.integers(lo_r)), int(rng.integers(lo_c))
    b, d = int(rng.integers(hi_r + 1, 31)), int(rng.integers(hi_c + 1, 31))
    inp, out = inp[a:b, c:d], out[a:b, c:d]
    if rng.random() < .5:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {'input': inp.copy(), 'output': out.copy()}


def _fallback_core(counts, bg, track, bead, color_by_count, kind):
    """Finite constructive fallback, still governed by the same visible rule."""
    width = 21
    if kind in ('witness', 'ordinary'):
        occupancy = [1, 0, 2, 1]
        bars = [(1, 3, 1), (3, 5, 2), (5, 11, 1), (1, 15, 2)]
        if 3 in counts:
            bars.append((1, 9, 3))
        beads = [(0, 20), (4, 0), (4, 20), (6, 20)]
    elif kind == 'motion':
        occupancy = [0, 1, 1, 0, 1]
        bars = [(1, 3, 1), (3, 5, 2), (3, 15, 1), (1, 15, 2)]
        if 3 in counts:
            bars.append((5, 11, 3))
        beads = [(2, 20), (4, 0), (8, 20)]
    else:
        occupancy = [0, 0, 2, 2]
        bars = []
        for k in counts:
            bars.extend({0: [(1, 3, 0), (5, 3, 0)],
                         4: [(5, 15, 4), (1, 15, 4)],
                         1: [(3, 3, 1)], 2: [(1, 9, 2)],
                         3: [(3, 15, 3)]}[k])
        beads = [(4, 0), (4, 20), (6, 0), (6, 20)]
    core = np.full((2 * len(occupancy) - 1, width), bg, dtype=np.int8)
    core[::2] = track
    records = []
    for r, c, k in bars:
        core[r, c - 1:c + 2] = color_by_count[k]
        total = occupancy[(r - 1) // 2] + occupancy[(r + 1) // 2]
        records.append({'row': r, 'col': c, 'color': color_by_count[k], 'active': total == k})
    for p in beads:
        core[p] = bead
    structural, paths, initial = _rewire(core, bg, track, bead, records)
    out, movement = _settle(structural, paths, initial, bead)
    if kind == 'witness':
        assert movement['collision'] and movement['level'] and movement['moved']
    if kind == 'motion':
        assert movement['single'] and movement['level']
    return core, out


def _make_pair(rng, kind):
    if kind in ('witness', 'motion', 'ordinary'):
        counts = [1, 2] + ([3] if rng.random() < .55 else [])
    else:
        mandatory = [0, 4] if kind == 'both' else ([0] if kind == 'zero' else [4])
        others = [k for k in range(5) if k not in mandatory]
        number = int(rng.integers(max(2, len(mandatory)), 6))
        counts = sorted(mandatory + list(map(int, rng.choice(others, number - len(mandatory), replace=False))))
    palette = list(map(int, rng.choice(10, 3 + len(counts), replace=False)))
    bg, track, bead = palette[:3]
    color_by_count = dict(zip(counts, palette[3:]))
    motifs = []
    for k in counts:
        size = int(rng.integers(5, 8 if len(counts) >= 4 else 9))
        mask = _cavities(rng, size, k, witness=kind == 'witness' and k in (1, 2))
        motif = np.full((size, size), color_by_count[k], dtype=np.int8)
        motif[mask] = bg
        motifs.append(motif)
    for _ in range(100):
        result = _core_candidate(rng, counts, bg, track, bead, color_by_count, kind)
        if result is not None:
            return _place(rng, bg, *result, motifs)
    return _place(rng, bg, *_fallback_core(counts, bg, track, bead, color_by_count, kind), motifs)


def _check_count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer from {lo} to {hi}')
    return int(value)


def _distinct_recolor(pair, seen):
    # Both selected colors occur. Their 90 distinct ordered images guarantee
    # more candidates than the at most 13 previously used puzzle inputs.
    first = list(map(int, np.unique(pair['input'])))[:2]
    source = first + [c for c in range(10) if c not in first]
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            target = [a, b] + [c for c in range(10) if c not in (a, b)]
            mapping = np.empty(10, dtype=np.int8)
            mapping[source] = target
            candidate = {k: mapping[v] for k, v in pair.items()}
            key = (candidate['input'].shape, candidate['input'].tobytes())
            if key not in seen:
                return candidate
    raise AssertionError('finite recoloring uniqueness bound violated')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _check_count(num_train, 'num_train', 2, 10)
    ne = _check_count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shift = _choice(rng, ['zero', 'four', 'both'])
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, kinds in (('train', ['witness', 'motion'] + ['ordinary'] * (nt - 2)),
                         ('test', [shift] * ne)):
        for kind in kinds:
            for _ in range(20):
                pair = _make_pair(rng, kind)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                pair = _distinct_recolor(pair, seen)
                seen.add((pair['input'].shape, pair['input'].tobytes()))
                puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, movement):
    active = [g for g in info['gates'] if g['active']]
    inactive = [g for g in info['gates'] if not g['active']]
    keys = info['keys']
    used_keys = [keys[c] for c in {g['color'] for g in active}]
    output, _ = _prediction(info)
    r0, r1, c0, c1 = info['box']
    final_counts = [int(np.count_nonzero(row == info['bead'])) for row in output[r0:r1:2, c0:c1]]
    return {'active': len(active), 'inactive': len(inactive),
            'thresholds': tuple(sorted(k['count'] for k in keys.values())),
            'area_inversion': any(a['count'] < b['count'] and a['area'] > b['area']
                                  for a in used_keys for b in used_keys),
            'connectivity_used': any(keys[g['color']]['eight'] != g['threshold'] for g in active),
            'large_used': any(keys[g['color']]['area'] > g['threshold'] for g in active),
            'conditional': bool({g['color'] for g in active} & {g['color'] for g in inactive}),
            'below_inactive': any(g['total'] < g['threshold'] for g in inactive),
            'above_inactive': any(g['total'] > g['threshold'] for g in inactive),
            'parity_witness': any(g['total'] % 2 == g['threshold'] % 2 for g in inactive),
            'sum_witness': any(0 < info['occupancy'][(g['row'] - 1) // 2] < g['total']
                               and 0 < info['occupancy'][(g['row'] + 1) // 2] < g['total'] for g in active),
            'snapshot_witness': any((final_counts[(g['row'] - 1) // 2] + final_counts[(g['row'] + 1) // 2]
                                     == g['threshold']) != g['active'] for g in info['gates']),
            **movement}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, test = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            signature = (inp.shape, inp.tobytes())
            assert signature not in seen
            seen.add(signature)
            info = _read_input(inp)
            expected, movement = _prediction(info)
            assert np.array_equal(out, expected), 'incorrect gate turn or bead settlement'
            r0, r1, c0, c1 = info['box']
            outside = np.ones(inp.shape, dtype=bool)
            outside[r0:r1, c0:c1] = False
            assert np.array_equal(inp[outside], out[outside])
            assert np.count_nonzero(inp == info['bead']) == np.count_nonzero(out == info['bead'])
            core_out = out[r0:r1, c0:c1]
            for path in info['paths']:
                assert sum(core_out[p] == info['bead'] for p in path) == len(info['beads'].intersection(path))
            lane_breaks = np.zeros(info['core'].shape, dtype=bool)
            lane_breaks[::2] = info['core'][::2] == info['bg']
            assert np.all(core_out[lane_breaks] == info['bg'])
            f = _features(info, movement)
            assert f['active'] and f['inactive'], 'each scene must contain a comparison'
            (training if split == 'train' else test).append(f)
    for name in ('single', 'collision', 'level', 'area_inversion', 'connectivity_used',
                 'large_used', 'conditional', 'sum_witness', 'snapshot_witness',
                 'below_inactive', 'above_inactive', 'parity_witness'):
        assert any(f[name] for f in training), f'missing training witness: {name}'
    # Matching is demonstrated for positive counts; the query extends it to
    # a visibly encoded new threshold, with an actually active such gate.
    train_counts = set(k for f in training for k in f['thresholds'])
    assert train_counts <= {1, 2, 3}
    for pair, f in zip(puzzle['test'], test):
        info = _read_input(pair['input'])
        assert any(g['active'] and g['threshold'] in (0, 4) and g['threshold'] not in train_counts
                   for g in info['gates'])


def _signature(puzzle):
    return tuple((s, _grid(p['input']).shape, _grid(p['input']).tobytes(), _grid(p['output']).tobytes())
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed checks, semantic counterexamples and diversity metrics."""
    metrics = {'puzzles': 42, 'pairs': 0}
    signatures, shapes, structures, key_shapes = set(), set(), set(), set()
    lane_counts, gate_counts, active_counts, break_counts, threshold_sets = set(), set(), set(), set(), set()
    palettes, shifts, overlaps = set(), set(), 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        tr = {k for p in puzzle['train'] for k in _features(_read_input(p['input']), _prediction(_read_input(p['input']))[1])['thresholds']}
        te = {k for p in puzzle['test'] for k in _features(_read_input(p['input']), _prediction(_read_input(p['input']))[1])['thresholds']}
        shifts.add(tuple(sorted((te - tr) & {0, 4})))
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                metrics['pairs'] += 1
                info = _read_input(pair['input'])
                predicted = solve(pair['input'].copy())
                assert np.array_equal(predicted, pair['output'])
                assert not np.shares_memory(predicted, pair['input'])
                f = _features(info, _prediction(info)[1])
                shapes.add(pair['input'].shape)
                roles = {info['bg']: 0, info['track']: 1, info['bead']: 2}
                roles.update({c: 3 + k['count'] for c, k in info['keys'].items()})
                normalized = np.array([[roles[int(c)] for c in row] for row in pair['input']], dtype=np.int8)
                structures.add((normalized.shape, normalized.tobytes()))
                palettes.add(tuple(sorted(map(int, np.unique(pair['input'])))))
                lane_counts.add(len(info['occupancy']))
                gate_counts.add(len(info['gates']))
                active_counts.add(f['active'])
                break_counts.add(int(np.count_nonzero(info['core'][::2] == info['bg'])))
                threshold_sets.add(f['thresholds'])
                active = [(g['row'], g['col']) for g in info['gates'] if g['active']]
                overlaps += any(abs(a - b) == 2 and abs(c - d) == 2
                                for a, c in active for b, d in active)
                for key in info['keys'].values():
                    mask = key['mask']
                    key_shapes.add((mask.shape, min(m.tobytes() for m in _transforms(mask))))
                # Removing resonance leaves the input lanes level. Removing
                # settlement preserves the turned graph but loses every move.
                if f['moved']:
                    no_settle = info['structural'].copy()
                    for p in info['beads']:
                        no_settle[p] = info['bead']
                    r0, r1, c0, c1 = info['box']
                    assert not np.array_equal(no_settle, predicted[r0:r1, c0:c1])
                    assert not np.array_equal(pair['input'], predicted)
    assert len(signatures) == 42 and len(structures) >= 190
    assert len(shapes) >= 25 and len(key_shapes) >= 60
    assert len(lane_counts) >= 4 and len(gate_counts) >= 8 and len(active_counts) >= 5
    assert len(break_counts) >= 5 and len(threshold_sets) >= 8 and len(shifts) == 3
    assert overlaps and len(palettes) >= 20
    metrics.update(distinct_puzzles=len(signatures), shapes=len(shapes),
                   color_normalized_structures=len(structures), cavity_shapes_D4=len(key_shapes),
                   lane_counts=sorted(lane_counts), gate_counts=sorted(gate_counts),
                   active_counts=sorted(active_counts), break_counts=sorted(break_counts),
                   threshold_sets=len(threshold_sets), shifts=[list(x) for x in sorted(shifts)],
                   overlapping_pairs=overlaps, palettes=len(palettes))
    assert _signature(generate_puzzle(123, 2, 1)) == _signature(generate_puzzle(123, 2, 1))
    for a, b in ((2, 1), (10, 4), (10, 1)):
        generate_puzzle(2523, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    sample = generate_puzzle(0, 2, 1)
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    # Explicit connectivity and hole-area fixtures, unrelated to any seed.
    diagonal = np.zeros((5, 5), dtype=bool)
    diagonal[1, 1] = diagonal[2, 2] = diagonal[2, 3] = True
    assert len(_components(diagonal)) == 2 and len(_components(diagonal, True)) == 1
    assert int(diagonal.sum()) == 3
    ring = np.zeros((5, 5), dtype=bool)
    ring[1:-1, 1:-1] = True
    assert len(_components(ring)) == 1 and ring.sum() > diagonal.sum()
    # Exercise every fallback threshold subset allowed by each curriculum,
    # rather than relying on the random sampler ever needing its fallback.
    from itertools import combinations
    for number in range(2, 6):
        for subset in combinations(range(5), number):
            if not set(subset).intersection((0, 4)):
                continue
            colors = dict(zip(subset, range(3, 3 + number)))
            core, core_out = _fallback_core(subset, 0, 1, 2, colors, 'both')
            assert np.count_nonzero(core == 2) == np.count_nonzero(core_out == 2)
    for kind in ('witness', 'motion', 'ordinary'):
        for subset in ((1, 2), (1, 2, 3)):
            _fallback_core(subset, 0, 1, 2, dict(zip(subset, range(3, 3 + len(subset)))), kind)
    repeated = generate_puzzle(0)
    last = repeated['train'][0]
    seen = {(last['input'].shape, last['input'].tobytes())}
    for _ in range(13):
        candidate = _distinct_recolor(last, seen)
        assert np.array_equal(solve(candidate['input']), candidate['output'])
        seen.add((candidate['input'].shape, candidate['input'].tobytes()))
    # Hand-worked semantic fixture. Original lane counts are 1,0,2,1.
    # Gates at (1,3) and (3,5) turn; the other two stay. The first
    # lane's right bead joins the third lane's left bead at (4,0),(4,1).
    core = np.zeros((7, 15), dtype=np.int8)
    core[::2] = 1
    for r, c, color in ((1, 3, 4), (3, 5, 6), (1, 11, 6), (3, 9, 4)):
        core[r, c - 1:c + 2] = color
    for p in ((0, 14), (4, 0), (4, 14), (6, 14)):
        core[p] = 2
    fixture = np.zeros((16, 17), dtype=np.int8)
    fixture[1:8, 1:16] = core
    fixture[10:15, 1:6] = np.where(ring, 0, 4)
    fixture[10:15, 10:15] = np.where(diagonal, 0, 6)
    expected = fixture.copy()
    changed = core.copy()
    changed[changed == 2] = 1
    for r, c, color in ((1, 3, 4), (3, 5, 6)):
        changed[r - 1:r + 2, c - 1:c + 2] = 1
        changed[r - 1:r + 2, c] = color
    for p in ((4, 0), (4, 1), (4, 14), (6, 14)):
        changed[p] = 2
    expected[1:8, 1:16] = changed
    assert np.array_equal(solve(fixture), expected)
    # Swap the visible cavity thresholds, preserving colors and every bead.
    # The active subset changes to (1,11),(3,9). This is still a valid graph,
    # but now the first lane's bead goes to the second lane's right endpoint.
    swapped = fixture.copy()
    swapped[10:15, 1:6] = np.where(diagonal, 0, 4)
    swapped[10:15, 10:15] = np.where(ring, 0, 6)
    new_output = solve(swapped)
    assert new_output[3, 15] == 2 and new_output[1, 15] == 1
    assert set(map(tuple, np.argwhere(new_output == 2))) != set(map(tuple, np.argwhere(expected == 2)))
    return metrics


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
