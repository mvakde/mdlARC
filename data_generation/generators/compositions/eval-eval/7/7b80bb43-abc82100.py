"""Restored circuits reveal stencil sockets.

0 is background, 8 is stencil ink, 9 is a power node. Other roles are
read from each input's local source/target labels. No parent generator is used.
"""
from collections import deque
import numpy as np

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid shape")
    _require(np.issubdtype(g.dtype, np.integer) and np.all((g >= 0) & (g <= 9)),
             "grid must contain integer ARC colors")
    return g.astype(np.int8, copy=True)


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _at(mask, r, c):
    return _inside(mask.shape, (r, c)) and bool(mask[r, c])


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
             if (a or b) and (diagonal or abs(a) + abs(b) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _strong(mask):
    strong = np.zeros_like(mask)
    for transpose in (False, True):
        m = mask.T if transpose else mask
        s = strong.T if transpose else strong
        if m.shape[1] < 3:
            continue
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for offset in range(3):
            s[:, offset:offset + triples.shape[1]] |= triples
    return strong


def _gaps(mask):
    gaps = []
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        if m.shape[1] < 5:
            continue
        brackets = m[:, :-4] & m[:, 4:]
        brackets &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(brackets):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            gaps.append((r, c, dr, dc))
    return gaps


def _repair_evidence(mask):
    """Evaluate all cues on the original geometry, without iterative repairs."""
    strong = _strong(mask)
    evidence = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r - dr, c - dc) or _at(mask, r + 5 * dr, c + 5 * dc)):
            continue
        matches = []
        for endpoint in (0, 4):
            ar, ac = r + endpoint * dr, c + endpoint * dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward * dr + side * dc, toward * dc + side * dr
                spur = tuple((ar + k * sr, ac + k * sc) for k in (1, 2))
                if (all(_at(mask, rr, cc) and not strong[rr, cc] for rr, cc in spur)
                        and not _at(mask, ar + 3 * sr, ac + 3 * sc)):
                    matches.append(spur)
        if matches:
            _require(len(matches) == 1, "ambiguous diagonal repair")
            evidence.append(((r, c, dr, dc), matches[0]))
    return evidence


def _clean(mask, roots, repair=True, fill_all=False):
    evidence = _repair_evidence(mask)
    added, protect = np.zeros_like(mask), roots.copy()
    selected = [(gap, ()) for gap in _gaps(mask)] if fill_all else evidence if repair else []
    for (r, c, dr, dc), _ in selected:
        for k in (1, 2, 3):
            added[r + k * dr, c + k * dc] = True
        protect[r, c] = protect[r + 4 * dr, c + 4 * dc] = True
    weak = mask & ~_strong(mask | added) & ~protect
    erase = np.zeros_like(mask)
    for dc in (-1, 1):
        a = weak[:-1, :-1] if dc == 1 else weak[:-1, 1:]
        b = weak[1:, 1:] if dc == 1 else weak[1:, :-1]
        links = a & b
        if dc == 1:
            erase[:-1, :-1] |= links
            erase[1:, 1:] |= links
        else:
            erase[:-1, 1:] |= links
            erase[1:, :-1] |= links
    return (mask & ~erase) | added, evidence


def _terminals(mask, roots, powered=True):
    reached = np.zeros_like(mask)
    queue = deque(map(tuple, np.argwhere(mask & roots).tolist()))
    for p in queue:
        reached[p] = True
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            q = (r + dr, c + dc)
            if _inside(mask.shape, q) and mask[q] and not reached[q]:
                reached[q] = True
                queue.append(q)
    degree = np.zeros(mask.shape, dtype=np.int8)
    degree[:-1] += mask[1:]
    degree[1:] += mask[:-1]
    degree[:, :-1] += mask[:, 1:]
    degree[:, 1:] += mask[:, :-1]
    sockets = mask & (degree == 1) & ~roots
    if powered:
        sockets &= reached
    return set(map(tuple, np.argwhere(sockets).tolist())), reached


def _parse(grid):
    masks = _components(grid == 8, diagonal=True)
    _require(bool(masks), "expected at least one stencil instruction")
    records = []
    for cells in masks:
        near = {(r + a, c + b) for r, c in cells
                for a in (-1, 0, 1) for b in (-1, 0, 1)}
        candidates = []
        for qr, qc in sorted(near):
            if not _inside(grid.shape, (qr, qc)) or grid[qr, qc] not in (0, 8):
                continue
            for dr, dc in DIRS:
                sl, tl = (qr - 2 * dr, qc - 2 * dc), (qr - dr, qc - dc)
                if not _inside(grid.shape, sl) or not _inside(grid.shape, tl):
                    continue
                source, target = int(grid[sl]), int(grid[tl])
                if 1 <= source <= 7 and 1 <= target <= 7 and source != target:
                    candidates.append(dict(source=source, target=target, q=(qr, qc),
                        direction=(dr, dc), labels={sl, tl}, mask=cells,
                        offsets=tuple(sorted((r - qr, c - qc) for r, c in cells))))
        _require(len(candidates) == 1, "stencil has missing or ambiguous registration")
        records.append(candidates[0])
    _require(len({rec['source'] for rec in records}) == len(records), "repeated wire source")
    labels = set().union(*(rec['labels'] for rec in records))
    _require(len(labels) == 2 * len(records), "shared instruction labels")
    # Power nodes must be associated with one source color via visible neighbors.
    sources = {rec['source'] for rec in records}
    roots_by_source = {source: np.zeros(grid.shape, bool) for source in sources}
    for r, c in np.argwhere(grid == 9):
        r, c = int(r), int(c)
        nearby = {int(grid[r + dr, c + dc]) for dr, dc in DIRS
                  if _inside(grid.shape, (r + dr, c + dc))
                  and (r + dr, c + dc) not in labels
                  and int(grid[r + dr, c + dc]) in sources}
        _require(len(nearby) == 1, "power node needs one neighboring wire color")
        roots_by_source[next(iter(nearby))][r, c] = True
    for rec in records:
        rec['roots'] = roots_by_source[rec['source']]
        _require(rec['roots'].any(), "instruction has no power node")
        mask = grid == rec['source']
        for p in labels:
            mask[p] = False
        rec['raw'] = mask | rec['roots']
        rec['clean'], rec['evidence'] = _clean(rec['raw'], rec['roots'])
        rec['sockets'], rec['reached'] = _terminals(rec['clean'], rec['roots'])
    instruction_cells = labels | set().union(*(rec['mask'] | {rec['q']} for rec in records))
    all_wires = np.zeros(grid.shape, bool)
    for rec in records:
        _require(not (all_wires & rec['clean']).any(), 'reconstructed wire types intersect')
        all_wires |= rec['clean']
        _require(np.all(grid[rec['clean'] & ~rec['raw']] == 0), 'repair crosses non-background content')
        _require(not any(rec['clean'][p] for p in instruction_cells), 'instruction intersects reconstructed wire')
        for ar, ac in rec['sockets']:
            _require(not any((ar + dr, ac + dc) in instruction_cells for dr, dc in rec['offsets']),
                     'stamp intersects instruction')
    return records, labels


def _render(grid, records, labels, mode='normal'):
    output = grid.copy()
    output[grid == 8] = 0
    for p in labels:
        output[p] = 0
    paints, sockets_by_rec = {}, []
    for rec in records:
        clean, sockets = rec['clean'], rec['sockets']
        if mode in ('no_repair', 'fill_all'):
            clean, _ = _clean(rec['raw'], rec['roots'], repair=mode != 'no_repair',
                              fill_all=mode == 'fill_all')
            sockets, _ = _terminals(clean, rec['roots'])
        elif mode == 'all_ends':
            sockets, _ = _terminals(clean, rec['roots'], powered=False)
        elif mode == 'all_wire':
            sockets = set(map(tuple, np.argwhere(clean & ~rec['roots']).tolist()))
        output[rec['raw'] & ~rec['roots']] = 0
        output[clean & ~rec['roots']] = rec['source']
        sockets_by_rec.append(sockets)
    # Erase every selected terminal before combining any translated masks.
    for sockets in sockets_by_rec:
        for p in sockets:
            output[p] = 0
    for rec, sockets in zip(records, sockets_by_rec):
        offsets = rec['offsets'] if mode != 'no_stencil' else ((0, 0),)
        if mode == 'centered':
            rs, cs = zip(*rec['mask'])
            center = ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)
            offsets = tuple((r - center[0], c - center[1]) for r, c in rec['mask'])
        for ar, ac in sorted(sockets):
            socket_offsets = offsets
            if mode == 'wire_aligned':
                neighbors = [(dr, dc) for dr, dc in DIRS if _at(rec['clean'], ar + dr, ac + dc)]
                outward = (-neighbors[0][0], -neighbors[0][1])
                turns = next(k for k in range(4) if _rotate(rec['direction'], k) == outward)
                socket_offsets = tuple(_rotate(p, turns) for p in offsets)
            for dr, dc in socket_offsets:
                p = (ar + dr, ac + dc)
                if not _inside(grid.shape, p):
                    continue
                _require(p not in paints or paints[p] == rec['target'], "conflicting stencil paint")
                paints[p] = rec['target']
    for p, color in paints.items():
        output[p] = color
    return output, paints


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    records, labels = _parse(grid)
    return _render(grid, records, labels)[0]


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _rotate(p, turns):
    r, c = p
    for _ in range(turns):
        r, c = -c, r
    return r, c


def _stencil(rng, hole=False):
    if hole:
        radius = int(rng.integers(1, 4))
        width = int(rng.integers(1, 4))
        cells = {(r, 1) for r in range(-radius, radius + 1)}
        cells |= {(r, c) for r in (-radius, radius) for c in range(-1, width + 1)}
        if rng.random() < .5:
            cells |= {(radius + 1, width)}
    else:
        # Procedural connected growth in a half-plane keeps the label ray clear.
        cells = {(0, 0), (0, 1)}
        height, width = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        desired = int(rng.integers(3, min(16, height * width) + 1))
        for _ in range(desired * 15):
            if len(cells) >= desired:
                break
            r, c = _pick(rng, sorted(cells))
            dr, dc = _pick(rng, DIRS)
            if -height // 2 <= r + dr <= height // 2 and 0 <= c + dc < width:
                cells.add((r + dr, c + dc))
    if rng.integers(2):
        cells = {(-r, c) for r, c in cells}
    turns = int(rng.integers(4))
    return tuple(sorted(_rotate(p, turns) for p in cells)), _rotate((0, 1), turns)


def _damage_options(mask):
    candidates = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        for dr, dc in ((0, 1), (1, 0)):
            if not all(_at(mask, r + k * dr, c + k * dc) for k in range(5)):
                continue
            if not (_at(mask, r - dr, c - dc) or _at(mask, r + 5 * dr, c + 5 * dc)):
                continue
            gap = tuple((r + k * dr, c + k * dc) for k in (1, 2, 3))
            if any(_at(mask, rr + dc, cc + dr) or _at(mask, rr - dc, cc - dr)
                   for rr, cc in gap):
                continue
            for endpoint in (0, 4):
                ar, ac = r + endpoint * dr, c + endpoint * dc
                toward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    sr, sc = toward * dr + side * dc, toward * dc + side * dr
                    spur = tuple((ar + k * sr, ac + k * sc) for k in (1, 2))
                    if all(_inside(mask.shape, p) and not mask[p] for p in spur):
                        candidates.append((gap, spur))
    return candidates


def _circuit(rng, shape, repairs=1, contrast=False, loop=False):
    """Build clean linework first; recoverability is checked after corruption."""
    h, w = shape
    transpose = h > w
    lh, lw = (w, h) if transpose else (h, w)
    for _ in range(80):
        clean = np.zeros((lh, lw), bool)
        long_rows = [r for r in range(2, lh - 2) if max(r, lh - 1 - r) >= 7]
        if not long_rows:
            continue
        row = int(_pick(rng, long_rows))
        left, right = int(rng.integers(0, 3)), lw - 1 - int(rng.integers(0, 3))
        if right - left < 8:
            continue
        clean[row, left:right + 1] = True
        cols = [left, right]
        available = [c for c in range(left + 3, right - 2)]
        for _ in range(int(rng.integers(0, 4))):
            if not available:
                break
            col = int(_pick(rng, available))
            cols.append(col)
            available = [c for c in available if abs(c - col) >= 3]
        long_side = 1 if lh - 1 - row >= 7 else -1
        for col in cols:
            side = long_side if col == right else int(_pick(rng, (-1, 1)))
            limit = lh - 1 - row if side == 1 else row
            length = int(rng.integers(2, limit + 1))
            if col == right:
                length = int(rng.integers(7, limit + 1))
            for k in range(length + 1):
                clean[row + k * side, col] = True
            # Secondary perpendicular arms change branching beyond a comb.
            if col not in (left, right) and length >= 4 and rng.random() < .5:
                rr = row + length * side
                end = int(np.clip(col + _pick(rng, (-1, 1)) * int(rng.integers(2, 5)), 0, lw - 1))
                clean[rr, min(col, end):max(col, end) + 1] = True
        if loop:
            level = row + 3 * long_side
            clean[min(row, level):max(row, level) + 1, left] = True
            clean[level, left:right + 1] = True
        if contrast:
            # A genuine bracketed break without any attached deviation.
            candidates = []
            for k in range(1, lh - 3):
                cells = [(row + (k + j) * long_side, right) for j in range(5)]
                if all(_inside(clean.shape, p) and clean[p] for p in cells):
                    gap = cells[1:4]
                    if not any(_at(clean, r, c - 1) or _at(clean, r, c + 1) for r, c in gap):
                        candidates.append(gap)
            if not candidates:
                continue
            for p in _pick(rng, candidates):
                clean[p] = False
        if (clean[:-1, :-1] & clean[1:, :-1] & clean[:-1, 1:] & clean[1:, 1:]).any():
            continue
        # Choose power from the main component, independently of its location.
        # Leaf, junction, corner and interior placements all use the same rule.
        main = next(component for component in _components(clean) if (row, left) in component)
        root_candidates = sorted(main)
        wanted = int(_pick(rng, (1, 2, 2, 3, 4)))
        preferred = [p for p in root_candidates if sum(_at(clean, p[0] + dr, p[1] + dc)
                     for dr, dc in DIRS) == wanted]
        root = _pick(rng, preferred or root_candidates)
        roots = np.zeros_like(clean)
        roots[root] = True
        if not np.array_equal(_clean(clean, roots)[0], clean):
            continue
        raw = clean.copy()
        changes = np.zeros_like(clean)
        accepted = 0
        options = _damage_options(clean)
        for index in rng.permutation(len(options)):
            gap, spur = options[int(index)]
            if root in gap or root in spur:
                continue
            foot = set(gap + spur)
            if any(changes[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any() for r, c in foot):
                continue
            trial = raw.copy()
            for p in gap:
                trial[p] = False
            for p in spur:
                trial[p] = True
            try:
                recovered, evidence = _clean(trial, roots)
            except ValueError:
                continue
            if np.array_equal(recovered, clean) and len(evidence) == accepted + 1:
                raw = trial
                for p in foot:
                    changes[p] = True
                accepted += 1
                if accepted == repairs:
                    break
        if accepted != repairs:
            continue
        # Add detached oblique clutter only if it cannot alter reconstructed wires.
        for _ in range(int(rng.integers(0, 4))):
            for _ in range(25):
                r, c = int(rng.integers(lh)), int(rng.integers(lw))
                dc = int(_pick(rng, (-1, 1)))
                cells = [(r + k, c + k * dc) for k in range(int(rng.integers(2, 4)))]
                if not all(_inside(raw.shape, p) and not raw[p] and not clean[p] for p in cells):
                    continue
                trial = raw.copy()
                for p in cells:
                    trial[p] = True
                try:
                    recovered, evidence = _clean(trial, roots)
                except ValueError:
                    continue
                if np.array_equal(recovered, clean) and len(evidence) == accepted:
                    raw = trial
                    break
        sockets, reached = _terminals(clean, roots)
        no_repair, _ = _clean(raw, roots, repair=False)
        raw_sockets, _ = _terminals(no_repair, roots)
        if not sockets or sockets == raw_sockets:
            continue
        if contrast and not (_terminals(clean, roots, powered=False)[0] - sockets):
            continue
        if transpose:
            raw, clean, roots = raw.T, clean.T, roots.T
        if rng.integers(2):
            raw, clean, roots = raw[::-1], clean[::-1], roots[::-1]
        if rng.integers(2):
            raw, clean, roots = raw[:, ::-1], clean[:, ::-1], roots[:, ::-1]
        return raw.copy(), clean.copy(), roots.copy()
    raise ValueError("circuit construction exhausted")


def _key(grid):
    return tuple(grid.shape), grid.astype(np.int8).tobytes()


def _clipped_edges(records, shape):
    edges = set()
    for rec in records:
        for ar, ac in rec['sockets']:
            for dr, dc in rec['offsets']:
                r, c = ar + dr, ac + dc
                if r < 0: edges.add('top')
                if r >= shape[0]: edges.add('bottom')
                if c < 0: edges.add('left')
                if c >= shape[1]: edges.add('right')
    return edges


def _make_pair(rng, palette, index, testing, shift, training_masks):
    count = 1 if not testing and index == 0 else 2 if not testing and index == 1 else int(_pick(rng, (1, 1, 2)))
    for attempt in range(120):
        try:
            h, w = int(rng.integers(16, 31)), int(rng.integers(16, 31))
            if count == 2:
                if rng.integers(2):
                    h = int(rng.integers(25, 31))
                    cut = int(rng.integers(11, h - 11))
                    regions = [(0, 0, cut, w), (cut + 2, 0, h - cut - 2, w)]
                else:
                    w = int(rng.integers(25, 31))
                    cut = int(rng.integers(11, w - 11))
                    regions = [(0, 0, h, cut), (0, cut + 2, h, w - cut - 2)]
            else:
                regions = [(0, 0, h, w)]
            grid = np.zeros((h, w), np.int8)
            clean_grid = grid.copy()
            known = []
            for j, (r, c, rh, rw) in enumerate(regions):
                source = palette[j]
                if not testing and index == 0:
                    target = palette[1]
                elif not testing and index == 1:
                    target = palette[2] if j == 0 else palette[0]
                else:
                    target = int(_pick(rng, [x for x in palette if x != source]))
                hole = (not testing and index == 1 and j == 0) or ((testing or index != 0) and rng.random() < .5)
                offsets, direction = _stencil(rng, hole)
                if testing:
                    _require(offsets not in training_masks, 'test stencil repeats training geometry')
                repair_count = 1 if not testing and index == 0 else int(rng.integers(1, 3))
                if testing and shift == 'repairs':
                    repair_count = 3
                # Place circuits within their available area, allowing edge tips.
                inset_r = int(rng.integers(0, min(3, rh - 10)))
                inset_c = int(rng.integers(0, min(3, rw - 10)))
                ah, aw = rh - inset_r, rw - inset_c
                raw, clean, roots = _circuit(rng, (ah, aw), repair_count,
                    contrast=(not testing and index < 2) or rng.random() < .65,
                    loop=((testing or index > 0) and rng.random() < .3))
                dest = np.s_[r + inset_r:r + rh, c + inset_c:c + rw]
                grid[dest] = np.where(roots, 9, np.where(raw, source, 0))
                clean_grid[dest] = np.where(roots, 9, np.where(clean, source, 0))
                root_mask = np.zeros(grid.shape, bool)
                root_mask[dest] = roots
                wire_mask = np.zeros(grid.shape, bool)
                wire_mask[dest] = clean
                sockets, reached = _terminals(wire_mask, root_mask)
                known.append(dict(source=source, target=target, offsets=offsets,
                    direction=direction, roots=root_mask, clean=wire_mask,
                    sockets=sockets, reached=reached))
            # Plan footprint independently of instruction parsing.
            paints = {}
            for rec in known:
                for ar, ac in rec['sockets']:
                    for dr, dc in rec['offsets']:
                        p = (ar + dr, ac + dc)
                        if _inside(grid.shape, p):
                            _require(p not in paints or paints[p] == rec['target'], 'paint collision')
                            paints[p] = rec['target']
            # Instructions use free space throughout the canvas, not a fixed panel.
            occupied = set(map(tuple, np.argwhere((grid != 0) | (clean_grid != 0)).tolist())) | set(paints)
            for rec in known:
                dr, dc = rec['direction']
                rel = set(rec['offsets']) | {(0, 0), (-dr, -dc), (-2 * dr, -2 * dc)}
                options = [(r, c) for r in range(h) for c in range(w)
                           if all(_inside(grid.shape, (r + a, c + b))
                                  and (r + a, c + b) not in occupied for a, b in rel)]
                placed = False
                for idx in rng.permutation(len(options)):
                    qr, qc = options[int(idx)]
                    cells = {(qr + a, qc + b) for a, b in rec['offsets']}
                    if any(any(_at(grid == 8, r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1))
                           for r, c in cells):
                        continue
                    trial = grid.copy()
                    for p in cells:
                        trial[p] = 8
                    labels = {(qr - 2 * dr, qc - 2 * dc), (qr - dr, qc - dc)}
                    trial[qr - 2 * dr, qc - 2 * dc] = rec['source']
                    trial[qr - dr, qc - dc] = rec['target']
                    # Partial grids still have roots of not-yet-labeled sources;
                    # check unique label rays with the full parser only at end.
                    rec.update(q=(qr, qc), labels=labels, mask=cells)
                    grid = trial
                    occupied |= {(qr + a, qc + b) for a, b in rel}
                    placed = True
                    break
                _require(placed, 'no instruction space')
            parsed, labels = _parse(grid)
            for rec in known:
                actual = next(x for x in parsed if x['source'] == rec['source'])
                _require(actual['q'] == rec['q'] and actual['offsets'] == rec['offsets'], 'wrong label reading')
                _require(np.array_equal(actual['clean'], rec['clean']) and actual['sockets'] == rec['sockets'],
                         'constructed and recovered circuit differ')
            # Optional inactive-color content is preserved, even if it used to be
            # a source in another example. It stays clear of masks and paint.
            inactive = [x for x in palette if x not in {rec['source'] for rec in known}]
            for _ in range(int(rng.integers(0, 5))):
                p = (int(rng.integers(h)), int(rng.integers(w)))
                if grid[p] == 0 and p not in paints and p not in occupied:
                    trial = grid.copy()
                    trial[p] = int(_pick(rng, inactive))
                    try:
                        new_records, new_labels = _parse(trial)
                    except ValueError:
                        continue
                    if all(a['q'] == b['q'] and a['offsets'] == b['offsets'] for a, b in zip(parsed, new_records)):
                        grid = trial
                        clean_grid[p] = grid[p]
            expected = clean_grid.copy()
            for rec in known:
                for p in rec['sockets']:
                    expected[p] = 0
            for p, color in paints.items():
                expected[p] = color
            parsed, labels = _parse(grid)
            output, actual_paint = _render(grid, parsed, labels)
            _require(np.array_equal(output, expected), 'construction/output disagreement')
            _require(any(rec['evidence'] for rec in parsed), 'missing repairs')
            _require(not np.array_equal(_render(grid, parsed, labels, 'no_stencil')[0], output), 'irrelevant stencil')
            if testing and shift == 'clipping':
                _require(len(_clipped_edges(parsed, grid.shape)) >= 2, 'missing boundary recombination')
            if not testing and index < 2:
                _require(not np.array_equal(_render(grid, parsed, labels, 'no_repair')[0], output), 'irrelevant repair')
                _require(not np.array_equal(_render(grid, parsed, labels, 'all_ends')[0], output), 'missing power contrast')
                _require(not np.array_equal(_render(grid, parsed, labels, 'fill_all')[0], output), 'missing unsupported gap contrast')
                _require(any(len(rec['offsets']) == sum(_inside(grid.shape, (ar + dr, ac + dc))
                    for dr, dc in rec['offsets']) for rec in parsed for ar, ac in rec['sockets']), 'no complete teaching stamp')
                if index == 1:
                    _require(bool(_clipped_edges(parsed, grid.shape)), 'no clipping demonstration')
                else:
                    rec = parsed[0]
                    directions = {(dr, dc) for ar, ac in rec['sockets'] for dr, dc in DIRS
                                  if _at(rec['clean'], ar + dr, ac + dc)}
                    _require(len(rec['sockets']) >= 2 and len(directions) >= 2,
                             'teaching lacks multiple differently directed sockets')
                    _require(not np.array_equal(_render(grid, parsed, labels, 'wire_aligned')[0], output),
                             'wire-relative stencil rotation is not distinguished')
            return {'input': grid, 'output': expected}
        except ValueError:
            continue
    raise ValueError('bounded pair generation exhausted')


def _count(value, lo, hi, name):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)), name + ' must be an integer')
    value = int(value)
    _require(lo <= value <= hi, name + ' out of range')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)),
             'seed must be a nonnegative integer')
    seed = int(seed)
    _require(seed >= 0, 'seed must be a nonnegative integer')
    n, t = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    _require(n > t, 'num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(np.arange(1, 8))))
    shift = _pick(rng, ('mask', 'repairs', 'clipping'))
    puzzle, seen, training_masks = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', n), ('test', t)):
        for index in range(count):
            for _ in range(15):
                pair = _make_pair(rng, palette, index, split == 'test', shift, training_masks)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train':
                        training_masks.update(rec['offsets'] for rec in _parse(pair['input'])[0])
                    break
            else:
                raise ValueError('unable to generate distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    n, t = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= t <= 4 and n > t, 'invalid example counts')
    seen, teaching = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            grid, output = _grid(pair['input']), _grid(pair['output'])
            _require(grid.shape == output.shape and _key(grid) not in seen, 'shape mismatch or duplicate input')
            seen.add(_key(grid))
            records, labels = _parse(grid)
            expected, paint = _render(grid, records, labels)
            _require(np.array_equal(output, expected), 'incorrect full transformation')
            for p in map(tuple, np.argwhere(grid == 9).tolist()):
                if p not in paint:
                    _require(output[p] == 9, 'unpainted power node changed')
            _require(any(rec['evidence'] for rec in records), 'no repair evidence')
            _require(all(rec['sockets'] for rec in records), 'stencil has no powered socket')
            for rec in records:
                _require(not (rec['roots'] & ~rec['clean']).any(), 'lost power node')
                _require(not (rec['clean'][:-1, :-1] & rec['clean'][1:, :-1]
                    & rec['clean'][:-1, 1:] & rec['clean'][1:, 1:]).any(), 'thick wire block')
                for gap, spur in rec['evidence']:
                    r, c, dr, dc = gap
                    _require(all(rec['clean'][r + k * dr, c + k * dc] for k in (1, 2, 3)), 'unfilled repair')
                    _require(all(not rec['clean'][p] for p in spur), 'retained displacement')
            # Unpowered wire cells stay wire-colored unless an actual stamp lands there.
            for rec in records:
                for p in map(tuple, np.argwhere(rec['clean'] & ~rec['reached'] & ~rec['roots']).tolist()):
                    if p not in paint:
                        _require(output[p] == rec['source'], 'unpowered wire was stamped')
            if split == 'train':
                teaching.append((grid, records, labels, output))
    _require(any((0, 0) not in rec['offsets'] for _, records, _, _ in teaching for rec in records), 'training lacks hole registration')
    _require(any((0, 0) in rec['offsets'] and (sum(r for r, c in rec['offsets']) or sum(c for r, c in rec['offsets']))
        for _, records, _, _ in teaching for rec in records), 'training lacks off-center registration')
    _require(any(len(rec['sockets']) >= 2 for _, records, _, _ in teaching for rec in records),
             'training lacks multiple sockets in one powered component')
    for mode in ('no_repair', 'no_stencil', 'all_ends', 'fill_all', 'all_wire', 'centered', 'wire_aligned'):
        distinguishes = False
        for grid, records, labels, output in teaching:
            try:
                alternative = _render(grid, records, labels, mode)[0]
                distinguishes |= not np.array_equal(alternative, output)
            except ValueError:
                # Conflicting paint is itself incompatible with this shortcut.
                distinguishes = True
        _require(distinguishes, 'training fails to distinguish ' + mode)


def self_test():
    # Hand-authored bridge/intentional-break regression. A displaced horizontal
    # bridge powers the near end of an intentionally broken vertical branch;
    # its far fragment must remain unpowered. The hole version leaves the
    # selected socket empty rather than recoloring its registration pixel.
    for hole in (False, True):
        grid = np.zeros((15, 20), np.int8)
        grid[8, 2:18] = 1
        grid[8:15, 17] = 1
        grid[10:13, 17] = 0
        grid[8, 2] = 9
        grid[8, 8:11] = 0
        grid[9, 8] = grid[10, 9] = 1
        q = (2, 3)
        offsets = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0)) if hole else ((0, 0), (0, 1), (1, 1))
        grid[2, 1], grid[2, 2] = 1, 2
        for dr, dc in offsets:
            grid[q[0] + dr, q[1] + dc] = 8
        expected = np.zeros_like(grid)
        expected[8, 2:18] = 1
        expected[8:10, 17] = 1
        expected[13:15, 17] = 1
        expected[8, 2] = 9
        expected[9, 17] = 0
        for dr, dc in offsets:
            expected[9 + dr, 17 + dc] = 2
        _require(np.array_equal(solve(grid), expected), 'manual bridge/registration regression')
        records, _ = _parse(grid)
        _require(records[0]['sockets'] == {(9, 17)}, 'incorrect powered terminal')
        _require(_terminals(records[0]['clean'], records[0]['roots'], powered=False)[0]
                 - records[0]['sockets'] == {(13, 17), (14, 17)}, 'lost unpowered ends')
    # All selection precedes paint: this copy covers the original power node,
    # but both original endpoints still produce their complete registered masks.
    grid = np.zeros((11, 15), np.int8)
    grid[2, 6:9] = 8
    grid[2, 9], grid[2, 10] = 2, 1
    grid[8, 3:8] = 1
    grid[8, 5] = 9
    expected = np.zeros_like(grid)
    expected[8, 1:4] = expected[8, 5:8] = 2
    expected[8, 4] = 1
    _require(np.array_equal(solve(grid), expected), 'selection-after-paint regression')
    shapes, structures, masks, topologies, mappings, signatures = set(), set(), set(), set(), set(), set()
    axes, clipped, loops, socket_counts, label_directions, root_degrees = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        sig = tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])
        signatures.add(sig)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid = pair['input']
                _require(np.array_equal(solve(grid), pair['output']), 'solver disagreement')
                records, _ = _parse(grid)
                shapes.add(grid.shape)
                roles = {0: 0, 8: 1, 9: 2}
                structures.add((grid.shape, tuple(roles.setdefault(int(c), len(roles)) for c in grid.ravel())))
                for rec in records:
                    masks.add(rec['offsets'])
                    mappings.add((rec['source'], rec['target']))
                    socket_counts.add(len(rec['sockets']))
                    label_directions.add(rec['direction'])
                    for r, c in np.argwhere(rec['roots']):
                        root_degrees.add(sum(_at(rec['clean'], int(r) + dr, int(c) + dc) for dr, dc in DIRS))
                    for (r, c, dr, dc), spur in rec['evidence']:
                        axes.add((dr, dc))
                    nodes = int(rec['clean'].sum())
                    edges = int((rec['clean'][:-1] & rec['clean'][1:]).sum()
                                + (rec['clean'][:, :-1] & rec['clean'][:, 1:]).sum())
                    cycles = edges - nodes + len(_components(rec['clean']))
                    loops.add(bool(cycles))
                    topologies.add((len(rec['sockets']), len(_components(rec['clean'])), cycles, nodes))
                    for ar, ac in rec['sockets']:
                        for dr, dc in rec['offsets']:
                            r, c = ar + dr, ac + dc
                            if r < 0: clipped.add('top')
                            if r >= grid.shape[0]: clipped.add('bottom')
                            if c < 0: clipped.add('left')
                            if c >= grid.shape[1]: clipped.add('right')
    _require(len(signatures) == 42, 'seed diversity missing')
    _require(len(shapes) >= 35 and len(structures) >= 150, 'canvas/structural diversity missing')
    _require(len(masks) >= 50 and len(topologies) >= 50, 'geometry/topology diversity missing')
    _require(len(mappings) >= 20 and len(socket_counts) >= 4, 'role/count diversity missing')
    _require(axes == {(0, 1), (1, 0)} and loops == {False, True}, 'repair/topology variation missing')
    _require(label_directions == set(DIRS) and root_degrees == {1, 2, 3, 4}, 'instruction/power placement variation missing')
    _require(clipped == {'top', 'bottom', 'left', 'right'}, 'boundary variation missing')
    first, again = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    _require(all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], again[s])
                 for k in ('input', 'output')), 'nondeterministic generation')
    # Symmetry and input-only behavior: rotate and reflect full instructions,
    # networks and power nodes together, without changing semantic colors.
    for pair in first['train'] + first['test']:
        for turns in range(4):
            for reflection in (False, True):
                x, y = np.rot90(pair['input'], turns), np.rot90(pair['output'], turns)
                if reflection:
                    x, y = x[:, ::-1], y[:, ::-1]
                _require(np.array_equal(solve(x), y), 'symmetry regression')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    grid = bad['train'][0]['input']
    records, labels = _parse(grid)
    bad['train'][0]['output'] = _render(grid, records, labels, 'no_repair')[0]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('omitted repair accepted')
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, family invariants, ablations, and symmetries')
