"""Rank-addressed, translation-only docking of socket-bearing attachments.

No parent code is imported. The decoder obtains every role and address from
input pixels; sampled records are used only to construct and cross-check labels.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid color")
    return a


def _groups(points):
    remaining, groups = set(points), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _normal(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return {(r - r0, c - c0): int(v) for (r, c), v in cells.items()}


def _shape(cells):
    return max(r for r, c in cells) + 1, max(c for r, c in cells) + 1


def _render(cells, bg):
    cells = _normal(cells)
    h, w = _shape(cells)
    _require(h <= 30 and w <= 30, "assembly exceeds ARC bounds")
    out = np.full((h, w), bg, dtype=np.int8)
    for p, color in cells.items():
        out[p] = color
    return out


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _socket_regions(parts, register):
    """A socket is a 3/5 by 2/3 plate with its west-center cell replaced.

    The registration ink occurs nowhere else. Its monochrome components
    therefore expose the complete plate geometry without a glyph dictionary.
    """
    result = []
    for part in parts:
        sockets = []
        for group in _groups(p for p, v in part.items() if v == register):
            top, left = min(r for r, c in group), min(c for r, c in group)
            h = max(r for r, c in group) - top + 1
            w = max(c for r, c in group) - left + 1
            _require(h in (3, 5) and w in (2, 3), "registration ink is not a socket")
            center = top + h // 2, left
            rectangle = {(r, c) for r in range(top, top + h)
                         for c in range(left, left + w)}
            _require(group == rectangle - {center}, "incomplete socket plate")
            _require(center in part and part[center] != register, "missing socket cap")
            sockets.append({'top': top, 'left': left, 'h': h, 'w': w,
                            'cap': part[center], 'center': center,
                            'footprint': rectangle})
        _require(sockets, "every fragment needs a socket")
        result.append(sockets)
    return result


def _parse(parts, register, bg):
    sockets = _socket_regions(parts, register)
    roots = [i for i, group in enumerate(sockets) if len(group) >= 2]
    _require(len(roots) == 1, "need one multi-socket scaffold")
    root_index = roots[0]
    _require(all(len(s) == 1 for i, s in enumerate(sockets) if i != root_index),
             "an attachment must have exactly one socket")
    root = parts[root_index]
    ports, payload = [], None
    _require(2 <= len(sockets[root_index]) <= 6, "scaffold needs two to six sockets")
    _require(len({s['center'][0] for s in sockets[root_index]}) == len(sockets[root_index]),
             "stem center rows must be distinct")
    cap_colors = {s['cap'] for s in sockets[root_index]}
    _require(1 <= len(cap_colors) <= 3 and bg not in cap_colors, "invalid cap families")
    for s in sockets[root_index]:
        r, c = s['center']
        pcolor = root.get((r, c - 1))
        _require(pcolor is not None and pcolor not in cap_colors | {register, bg},
                 "missing stem interior")
        if payload is None:
            payload = pcolor
        _require(payload == pcolor, "scaffold stems must share their payload ink")
        start = c - 1
        while root.get((r, start)) == payload:
            start -= 1
        _require(root.get((r, start)) == s['cap'], "stem caps do not match")
        _require(c - start + 1 >= 3, "stem needs an interior")
        ports.append(dict(s, start=start, length=c - start + 1))
    by_cap = {cap: [p for p in ports if p['cap'] == cap] for cap in cap_colors}
    for cap, family in by_cap.items():
        _require(2 <= len(family) <= 6, "need two to six competitors per family")
        _require(len({p['length'] for p in family}) == len(family), "tied stem lengths")
        _require(len({(p['h'], p['w']) for p in family}) == 1, "one socket shape per family")
    covered = set().union(*(p['footprint'] for p in ports))
    covered |= {(p['center'][0], p['start']) for p in ports}
    _require(all(v == payload or pos in covered for pos, v in root.items()),
             "unexplained scaffold ink")
    # Color P outside the measured stems is the connecting scaffold, not a
    # separate keyed object. All cap-colored cells have now been accounted for.
    union, selected, addresses, contributions = dict(root), [], set(), []
    modules = []
    for i, part in enumerate(parts):
        if i == root_index:
            continue
        s = sockets[i][0]
        cap = s['cap']
        _require(cap in by_cap, "attachment has no matching socket family")
        family = by_cap[cap]
        _require((s['h'], s['w']) == (family[0]['h'], family[0]['w']),
                 "attachment socket does not exactly match")
        key = {p for p, v in part.items() if v == cap and p not in s['footprint']}
        rank = len(key)
        _require(1 <= rank <= min(4, len(family)), "invalid unary address")
        r, c = s['center']
        expected_key = {(r - 1, c + s['w'] + j) for j in range(1, rank + 1)}
        _require(key == expected_key, "unary address must be above the attachment handle")
        other = {p: v for p, v in part.items() if p not in s['footprint'] and p not in key}
        _require(other and len(set(other.values())) == 1, "attachment handle/tail must be uniform")
        ink = next(iter(other.values()))
        _require(ink not in cap_colors | {bg, register, payload}, "attachment ink has another role")
        _require(all(part.get((r, c + s['w'] + j)) == ink for j in range(rank + 1)),
                 "key is not connected to a handle")
        _require((cap, rank) not in addresses, "duplicate address within a family")
        addresses.add((cap, rank))
        ordered = sorted(family, key=lambda p: p['length'], reverse=True)
        target = ordered[rank - 1]
        dr, dc = target['top'] - s['top'], target['left'] - s['left']
        moved = {(a + dr, b + dc): v for (a, b), v in part.items()}
        shared = set(moved) & set(union)
        _require(shared == target['footprint'], "attachment overlaps outside its socket")
        _require(all(union[p] == moved[p] for p in shared), "registration colors conflict")
        contributions.append(len(moved) - len(shared))
        union.update(moved)
        selected.append(target)
        modules.append({'cap': cap, 'rank': rank, 'source': s, 'target': target,
                        'offset': (dr, dc), 'ink': ink, 'cells': part})
    _require(1 <= len(modules) <= 4, "need one to four attachments")
    raw = dict(union)
    for p in selected:
        r, c = p['center']
        for x in range(p['start'] + 1, c):
            _require(union[r, x] == payload, "selected stem interior was obstructed")
            union[r, x] = p['cap']
    return _render(union, bg), {
        'background': bg, 'register': register, 'payload': payload,
        'parts': parts, 'root_index': root_index, 'root': root,
        'ports': ports, 'families': by_cap, 'modules': modules,
        'selected': selected, 'raw_union': raw, 'filled_union': union,
        'contributions': contributions,
    }


def _decode(input_grid):
    inp = _grid(input_grid)
    values, counts = np.unique(inp, return_counts=True)
    _require(counts.max() * 2 > inp.size, "background must occupy over half the input")
    bg = int(values[counts.argmax()])
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(inp, -rotation)
        cells = {(int(r), int(c)): int(canonical[r, c])
                 for r, c in zip(*np.nonzero(canonical != bg))}
        parts = [_normal({p: cells[p] for p in group}) for group in _groups(cells)]
        if not 2 <= len(parts) <= 5:
            continue
        for register in values:
            if int(register) == bg:
                continue
            try:
                expected, info = _parse(parts, int(register), bg)
            except AssertionError:
                continue
            candidates.append((rotation, expected, info))
    _require(len(candidates) == 1, "ambiguous or malformed docking scene")
    rotation, out, info = candidates[0]
    info['rotation'] = rotation
    return np.ascontiguousarray(np.rot90(out, rotation)), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode sockets, use each unary address, translate, merge, fill, crop."""
    return _decode(input_grid)[0]


def _unaddressed_assembly_count(info, limit=2):
    """Count up to two valid assemblies if the unary ordinals are ignored.

    Exact socket correspondence, translation, collision avoidance, distinct
    destinations and ARC dimensions still apply. No compactness score applies.
    This is a targeted parent-B ablation, not an independent reference solver.
    """
    count = 0

    def search(index, union, used):
        nonlocal count
        if count >= limit:
            return
        if index == len(info['modules']):
            norm = _normal(union)
            if max(_shape(norm)) <= 30:
                count += 1
            return
        module = info['modules'][index]
        source = module['source']
        for target in info['families'][module['cap']]:
            if target['center'] in used:
                continue
            dr, dc = target['top'] - source['top'], target['left'] - source['left']
            moved = {(r + dr, c + dc): v for (r, c), v in module['cells'].items()}
            shared = set(moved) & set(union)
            if shared != target['footprint'] or any(union[p] != moved[p] for p in shared):
                continue
            merged = dict(union)
            merged.update(moved)
            search(index + 1, merged, used | {target['center']})

    search(0, dict(info['root']), set())
    return count


def _plate(h, w, cap, register):
    return {(r, c): cap if (r, c) == (h // 2, 0) else register
            for r in range(h) for c in range(w)}


def _attachment(rng, h, w, rank, cap, register, ink):
    cells = _plate(h, w, cap, register)
    m = h // 2
    extra = int(rng.integers(1, 3))
    tip = w + rank + extra
    for c in range(w, tip + 1):
        cells[m, c] = ink
    for j in range(1, rank + 1):
        cells[m - 1, w + j] = cap
    kind = int(rng.integers(5))
    depth = int(rng.integers(1, 4 if h == 5 else 5))
    if kind == 0:  # straight handle
        pass
    elif kind == 1:  # descending hook
        for r in range(m + 1, m + depth + 1):
            cells[r, tip] = ink
        cells[m + depth, tip + 1] = ink
    elif kind == 2:  # lower fork
        for r in range(m + 1, m + depth + 1):
            cells[r, tip] = ink
        cells[m + depth, tip - 1] = ink
        cells[m + depth, tip + 1] = ink
    elif kind == 3:  # staggered elbow
        cells[m + 1, tip] = ink
        cells[m + 1, tip + 1] = ink
        for r in range(m + 2, m + depth + 2):
            cells[r, tip + 1] = ink
    else:  # two teeth of unequal depth
        for r in range(m + 1, m + depth + 1):
            cells[r, tip] = ink
        for r in range(m + 1, m + max(1, depth - 1) + 1):
            cells[r, tip - 2] = ink
    return cells


def _scaffold(rng, populations, caps, register, payload, trap=False):
    total = sum(populations)
    heights = [3 if total >= 6 else int(rng.choice([3, 5])) for _ in populations]
    widths = [int(rng.choice([2, 3])) for _ in populations]
    gaps = [int(rng.integers(1, 3)) for _ in range(total - 1)]
    if sum(n * h for n, h in zip(populations, heights)) + sum(gaps) > 27:
        heights = [3] * len(populations)
    sequence = [f for f, n in enumerate(populations) for _ in range(n)]
    if rng.random() < .7:
        rng.shuffle(sequence)
    else:
        order = [int(f) for f in rng.permutation(len(populations))]
        sequence = [f for f in order for _ in range(populations[f])]
    lengths = []
    for n in populations:
        if rng.random() < .4:
            start = int(rng.integers(3, 12 - n))
            ls = list(range(start, start + n))
        else:
            ls = [int(v) for v in rng.choice(np.arange(3, 11), n, replace=False)]
        rng.shuffle(ls)
        lengths.append(ls)
    if trap:
        # A shorter stem can end farther right because starts differ. Both
        # teaching anchors defeat ranking sockets by absolute terminal column.
        lengths[0] = [6, 5, 3] if populations[0] == 3 else [7, 6, 4, 3]
        rng.shuffle(lengths[0])
    cells, records, row, cursors = {}, [], 0, [0] * len(populations)
    centers = []
    for j, f in enumerate(sequence):
        h, w, cap = heights[f], widths[f], caps[f]
        length = lengths[f][cursors[f]]
        cursors[f] += 1
        start = int(rng.integers(2, 5))
        if trap and f == 0:
            ordered = sorted(lengths[0], reverse=True)
            if length == ordered[0]:
                start = 2
            elif length == ordered[1]:
                start = 4
        r, end = row + h // 2, start + length - 1
        rail = int(rng.integers(2))
        centers.append((r, rail))
        for c in range(rail, end):
            cells[r, c] = cap if c == start else payload
        for (a, b), color in _plate(h, w, cap, register).items():
            cells[row + a, end + b] = color
        records.append({'family': f, 'cap': cap, 'length': length,
                        'start': start, 'center': (r, end), 'top': row,
                        'left': end, 'h': h, 'w': w})
        row += h + (gaps[j] if j < total - 1 else 0)
    for (r0, x0), (r1, x1) in zip(centers, centers[1:]):
        bend = int(rng.integers(r0 + 1, r1))
        for r in range(r0, bend + 1):
            cells[r, x0] = payload
        for c in range(min(x0, x1), max(x0, x1) + 1):
            cells[bend, c] = payload
        for r in range(bend, r1 + 1):
            cells[r, x1] = payload
    return cells, records


def _pack(parts, bg, rng):
    """Try free rectangle placement, then a guaranteed separated side packing."""
    shapes = [_shape(p) for p in parts]
    area = sum(len(p) for p in parts)
    min_h, min_w = max(h for h, w in shapes), max(w for h, w in shapes)
    for _ in range(8):
        h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
        if h * w <= 2 * area:
            continue
        grid = np.full((h, w), bg, dtype=np.int8)
        occupied = np.zeros((h, w), dtype=bool)
        placed = True
        order = sorted(range(len(parts)), key=lambda i: -shapes[i][0] * shapes[i][1])
        for i in order:
            ph, pw = shapes[i]
            candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)
                          if not occupied[r:r + ph, c:c + pw].any()]
            if not candidates:
                placed = False
                break
            r0, c0 = candidates[int(rng.integers(len(candidates)))]
            for (r, c), color in parts[i].items():
                grid[r + r0, c + c0] = color
            occupied[max(0, r0 - 1):min(h, r0 + ph + 1),
                     max(0, c0 - 1):min(w, c0 + pw + 1)] = True
        if placed:
            return grid
    rh, rw = shapes[0]
    module_order = [int(i) for i in rng.permutation(np.arange(1, len(parts)))]
    stack_h = sum(shapes[i][0] for i in module_order) + len(module_order) - 1
    mw = max(shapes[i][1] for i in module_order)
    h, w = max(rh, stack_h), rw + 1 + mw
    _require(h <= 30 and w <= 30, "components exceed packing budget")
    # Enough extra room for an obvious background, then randomized margins.
    while h * w <= 2 * area:
        if h < 30:
            h += 1
        elif w < 30:
            w += 1
        else:
            raise AssertionError("background cannot dominate")
    h += int(rng.integers(min(3, 30 - h) + 1))
    w += int(rng.integers(min(3, 30 - w) + 1))
    grid = np.full((h, w), bg, dtype=np.int8)
    modules_left = bool(rng.integers(2))
    root_col = mw + 1 if modules_left else 0
    module_col = 0 if modules_left else rw + 1
    root_row = int(rng.integers(h - rh + 1))
    for (r, c), color in parts[0].items():
        grid[root_row + r, root_col + c] = color
    row = int(rng.integers(h - stack_h + 1))
    for i in module_order:
        ph, pw = shapes[i]
        col = module_col + int(rng.integers(mw - pw + 1))
        for (r, c), color in parts[i].items():
            grid[row + r, col + c] = color
        row += ph + 1
    return grid


def _make_pair(rng, spec, anchor, common_bg):
    for _ in range(120):
        try:
            f = len(spec['populations'])
            options = [c for c in range(10) if c != anchor]
            rng.shuffle(options)
            bg = options[0] if common_bg is None else common_bg
            options = [c for c in options if c != bg]
            register, payload = options[:2]
            caps = [anchor] + options[2:2 + f - 1]
            inks = options[2 + f - 1:]
            _require(inks, "no attachment ink")
            root, records = _scaffold(rng, spec['populations'], caps, register, payload,
                                      spec.get('trap', False))
            parts, union, chosen = [root], dict(root), []
            for family, rank in spec['requests']:
                members = sorted((p for p in records if p['family'] == family),
                                 key=lambda p: p['length'], reverse=True)
                target = members[rank - 1]
                module = _attachment(rng, target['h'], target['w'], rank,
                                     caps[family], register, int(rng.choice(inks)))
                moved = {(r + target['top'], c + target['left']): v for (r, c), v in module.items()}
                footprint = {(r + target['top'], c + target['left'])
                             for r, c in _plate(target['h'], target['w'], caps[family], register)}
                _require(set(moved) & set(union) == footprint, "module material collides")
                _require(all(union[p] == moved[p] for p in footprint), "socket colors disagree")
                union.update(moved)
                parts.append(module)
                chosen.append(target)
            for p in chosen:
                r, c = p['center']
                for x in range(p['start'] + 1, c):
                    union[r, x] = p['cap']
            expected = _render(union, bg)
            inp = _pack(parts, bg, rng)
            rotation = spec['rotation']
            inp = np.ascontiguousarray(np.rot90(inp, rotation))
            expected = np.ascontiguousarray(np.rot90(expected, rotation))
            actual, info = _decode(inp)
            _require(np.array_equal(actual, expected), "input decoder disagrees with construction")
            if spec.get('trap', False):
                _require(_unaddressed_assembly_count(info) >= 2,
                         "teaching scene lacks genuine registration ambiguity")
            return {'input': inp, 'output': expected}
        except AssertionError:
            continue
    raise RuntimeError("bounded docking construction exhausted")


def _check_count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _random_populations(rng, maximum=6):
    f = int(rng.integers(1, 4))
    counts = [2] * f
    budget = 6 - sum(counts)
    for _ in range(int(rng.integers(budget + 1))):
        available = [i for i, n in enumerate(counts) if n < maximum]
        if available:
            counts[int(rng.choice(available))] += 1
    return counts


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _check_count('num_train', num_train, 2, 10)
    nq = _check_count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    anchor = int(rng.integers(10))
    common_bg = int(rng.choice([c for c in range(10) if c != anchor])) if rng.random() < .4 else None
    shift = int(rng.integers(3))  # repeated-family addresses, population, held-out axis
    orientation = int(rng.integers(4))
    max_rank = min(4, nt)
    specs = {'train': [], 'test': []}
    for i in range(nt):
        if i < 2:
            populations = [3 if i == 0 else int(rng.integers(3, 5)), 2]
            requests = [(0, i + 1), (1, 2 - i)]
            if i == 1:
                requests.append((0, 1))
            trap = True
        elif i < max_rank:
            rank = i + 1
            maximum = 4 if shift == 1 else 6
            populations = [int(rng.integers(rank, maximum + 1))]
            requests, trap = [(0, rank)], False
        else:
            populations = _random_populations(rng, 4 if shift == 1 else 6)
            requests = [(f, int(rng.integers(1, min(max_rank, n) + 1)))
                        for f, n in enumerate(populations)]
            if rng.random() < .35:
                family = int(rng.integers(len(populations)))
                used = requests[family][1]
                rank = int(rng.choice([r for r in range(1, min(max_rank, populations[family]) + 1)
                                      if r != used]))
                requests.append((family, rank))
            trap = False
        rotation = orientation if shift == 2 else int(rng.integers(4))
        specs['train'].append({'populations': populations, 'requests': requests,
                               'trap': trap, 'rotation': rotation})
    for _ in range(nq):
        if shift == 0:
            populations = [int(rng.integers(3, 5))]
            if rng.random() < .5:
                populations.append(2)
            ranks = sorted(int(v) for v in rng.choice(np.arange(1, min(max_rank, populations[0]) + 1),
                                                       2, replace=False))
            requests = [(0, r) for r in ranks]
            if len(populations) > 1:
                requests.append((1, int(rng.integers(1, 3))))
        elif shift == 1:
            populations = [int(rng.integers(5, 7))]
            requests = [(0, int(rng.integers(1, max_rank + 1)))]
            if rng.random() < .5:
                other = int(rng.choice([r for r in range(1, max_rank + 1) if r != requests[0][1]]))
                requests.append((0, other))
        else:
            populations = _random_populations(rng)
            requests = [(f, int(rng.integers(1, min(max_rank, n) + 1)))
                        for f, n in enumerate(populations)]
        rotation = (orientation + int(rng.choice([1, 3]))) % 4 if shift == 2 else int(rng.integers(4))
        specs['test'].append({'populations': populations, 'requests': requests, 'rotation': rotation})
    # Anchor positions are not part of the visual rule.
    rng.shuffle(specs['train'])
    puzzle, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for spec in specs[split]:
            for _ in range(12):
                pair = _make_pair(rng, spec, anchor, common_bg)
                key = _signature(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    return puzzle


def _measure(pair):
    expected, info = _decode(pair['input'])
    _require(np.array_equal(expected, np.asarray(pair['output'])), "incorrect docking output")
    bg, register = info['background'], info['register']
    # A's merging consumes exactly the copied plates, not the unary address
    # or ornament. B's filling changes exactly the open stem interiors.
    raw = _render(info['raw_union'], bg)
    filled = _render(info['filled_union'], bg)
    interior_count = sum(p['length'] - 2 for p in info['selected'])
    _require(np.count_nonzero(raw != filled) == interior_count, "wrong recoloring support")
    _require(not np.array_equal(raw, filled), "parent B contributes no change")
    register_before = int(np.count_nonzero(np.asarray(pair['input']) == register))
    register_after = int(np.count_nonzero(np.asarray(pair['output']) == register))
    consumed = sum(m['source']['h'] * m['source']['w'] - 1 for m in info['modules'])
    _require(register_before - register_after == consumed, "copied sockets did not merge exactly")
    _require(all(n > 0 for n in info['contributions']), "attachment has no exclusive contribution")
    _require(sum(len(p) for p in info['parts']) - len(info['raw_union']) ==
             sum(m['source']['h'] * m['source']['w'] for m in info['modules']),
             "unexpected absorption outside registration plates")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid example counts")
    analyses, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            _grid(pair['input'])
            _grid(pair['output'])
            key = _signature(pair['input'])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(_measure(pair))
    training_ranks = {m['rank'] for a in analyses['train'] for m in a['modules']}
    _require({1, 2} <= training_ranks, "training must teach ranks one and two")
    cap_ranks = {}
    for a in analyses['train']:
        for m in a['modules']:
            cap_ranks.setdefault(m['cap'], set()).add(m['rank'])
    _require(any({1, 2} <= rs for rs in cap_ranks.values()), "missing same-cap rank contrast")
    _require(any(1 < m['rank'] < len(a['families'][m['cap']])
                 for a in analyses['train'] for m in a['modules']), "missing non-extreme selection")
    _require(any(len(a['families']) >= 2 for a in analyses['train']), "cap grouping is not demonstrated")
    _require(any(len(a['modules']) > len({m['cap'] for m in a['modules']}) for a in analyses['train']),
             "multiple addresses in one family are not demonstrated")
    _require(any(_unaddressed_assembly_count(a) >= 2 for a in analyses['train']),
             "training never needs ordinals to resolve registration")
    _require(all(m['rank'] in training_ranks for a in analyses['test'] for m in a['modules']),
             "test address was not demonstrated")
    # Reject three concrete simpler spatial/rank alternatives on the training
    # data. Socket destinations matter even before stem filling is considered.
    for alternative in ('longest', 'ascending', 'terminal_position'):
        differs = False
        for a in analyses['train']:
            for m in a['modules']:
                family = a['families'][m['cap']]
                if alternative == 'longest':
                    other = max(family, key=lambda p: p['length'])
                elif alternative == 'ascending':
                    other = sorted(family, key=lambda p: p['length'])[m['rank'] - 1]
                else:
                    other = sorted(family, key=lambda p: p['left'], reverse=True)[m['rank'] - 1]
                differs |= other['center'] != m['target']['center']
        _require(differs, f"training permits {alternative} shortcut")


def self_test():
    """Bounded pixel-derived semantic, diversity, and invariance checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    infos, input_shapes, output_shapes, structures = [], set(), set(), set()
    signatures = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                info = _measure(pair)
                infos.append(info)
                input_shapes.add(np.asarray(pair['input']).shape)
                output_shapes.add(np.asarray(pair['output']).shape)
                a = np.asarray(pair['output'])
                renaming = {}
                structures.add((a.shape, tuple(renaming.setdefault(int(v), len(renaming)) for v in a.flat)))
    _require(len(signatures) == 42, "cross-seed duplicates")
    _require(len(input_shapes) >= 40 and len(output_shapes) >= 35, "limited shape diversity")
    _require(len(structures) >= 150, "limited structural diversity")
    _require({1, 2, 3} <= {len(a['families']) for a in infos}, "family-count diversity")
    _require({2, 3, 4, 5, 6} <= {len(f) for a in infos for f in a['families'].values()}, "population diversity")
    _require({1, 2, 3, 4} <= {m['rank'] for a in infos for m in a['modules']}, "address diversity")
    _require({0, 1, 2, 3} == {a['rotation'] for a in infos}, "orientation diversity")
    _require(len({a['background'] for a in infos}) >= 9, "background diversity")
    _require({3, 5} <= {p['h'] for a in infos for p in a['ports']}, "socket-height diversity")
    _require({2, 3} <= {p['w'] for a in infos for p in a['ports']}, "socket-width diversity")
    _require(any(len(a['modules']) > len({m['cap'] for m in a['modules']}) for a in infos),
             "no repeated-family addressing")
    _require(any(m['rank'] == len(a['families'][m['cap']]) for a in infos for m in a['modules']),
             "no shortest selections")
    _require(any(len({p['start'] for p in a['ports']}) > 1 for a in infos), "starts never vary")
    # Full-example rotations and arbitrary palette permutations preserve the
    # result. Moving separated components is already exercised by the packer.
    palette = np.array([6, 1, 8, 0, 5, 9, 3, 7, 2, 4], dtype=np.int8)
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        for rot in range(4):
            x = np.ascontiguousarray(palette[np.rot90(pair['input'], rot)])
            y = np.ascontiguousarray(palette[np.rot90(pair['output'], rot)])
            _require(np.array_equal(solve(x), y), "rotation/recoloring invariant failed")
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        a, b = generate_puzzle(900 + nt + nq, nt, nq), generate_puzzle(900 + nt + nq, nt, nq)
        validate_generated_puzzle(a)
        _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                     for k in ('input', 'output')), "seed not deterministic")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    p = bad['test'][0]
    p['output'].flat[0] = (int(p['output'].flat[0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return {'seeds': 42, 'pairs': len(infos), 'input_shapes': len(input_shapes),
            'output_shapes': len(output_shapes), 'normalized_outputs': len(structures),
            'populations': sorted({len(f) for a in infos for f in a['families'].values()}),
            'ranks': sorted({m['rank'] for a in infos for m in a['modules']})}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
