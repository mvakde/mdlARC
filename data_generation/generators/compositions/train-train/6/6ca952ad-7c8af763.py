"""Landing electorates: size-gated rigid falls with contact-local rim votes.

Only the standard library and NumPy are used. No parent code or data is loaded.
"""
from collections import Counter

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid size")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(np.all((0 <= g) & (g <= 9)), "color outside 0..9")
    return g


def _neighbors(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        piece, stack = {start}, [start]
        while stack:
            for neighbor in _neighbors(stack.pop()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    piece.add(neighbor)
                    stack.append(neighbor)
        result.append(piece)
    return result


def _intervals(length, bars):
    return [(a + 1, b - 1) for a, b in zip(bars, bars[1:])]


def _decode(g):
    """Recover all roles and chambers from the input, never from an output."""
    g = _grid(g)
    rim = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    wall_colors = set(map(int, rim))
    _require(len(wall_colors) == 3, "outer rim must expose structure and both labels")
    structure = int(g[0, 0])
    labels = tuple(sorted(wall_colors - {structure}))
    wall = np.isin(g, list(wall_colors))
    hbars = np.flatnonzero(wall.all(axis=1)).tolist()
    vbars = np.flatnonzero(wall.all(axis=0)).tolist()
    _require(hbars[0] == vbars[0] == 0 and
             hbars[-1] == g.shape[0] - 1 and vbars[-1] == g.shape[1] - 1,
             "complete outer frame required")
    expected = np.zeros(g.shape, dtype=bool)
    expected[hbars, :] = True
    expected[:, vbars] = True
    _require(np.array_equal(expected, wall), "walls must be complete one-cell bars")
    rows, cols = _intervals(g.shape[0], hbars), _intervals(g.shape[1], vbars)
    _require(all(b - a + 1 >= 4 for a, b in rows + cols), "chamber too thin")
    _require(all(g[r, c] == structure for r in hbars for c in vbars),
             "corners and intersections must be structural")
    values, counts = np.unique(g[~wall], return_counts=True)
    _require(len(values) == 2 and counts[0] != counts[1], "two interior roles required")
    bg = int(values[np.argmax(counts)])
    cargo = int(values[np.argmin(counts)])
    _require(counts.max() > 2 * counts.min(), "background must dominate chamber interiors")
    panels = [(r0, r1, c0, c1) for r0, r1 in rows for c0, c1 in cols]
    return dict(bg=bg, cargo=cargo, structure=structure, labels=labels,
                wall=wall, panels=panels, rows=rows, cols=cols,
                hbars=hbars, vbars=vbars)


def _panel_of(piece, panels):
    matches = [p for p in panels if all(p[0] <= r <= p[1] and
                                      p[2] <= c <= p[3] for r, c in piece)]
    _require(len(matches) == 1, "object must belong to exactly one chamber")
    return matches[0]


def _land(piece, panel):
    fall = panel[1] - max(r for r, _ in piece)
    return {(r + fall, c) for r, c in piece}, fall


def _contact(piece, wall):
    h, w = wall.shape
    return {q for p in piece for q in _neighbors(p)
            if 0 <= q[0] < h and 0 <= q[1] < w and wall[q]}


def _boundary(panel):
    r0, r1, c0, c1 = panel
    return ({(r0 - 1, c) for c in range(c0, c1 + 1)} |
            {(r1 + 1, c) for c in range(c0, c1 + 1)} |
            {(r, c0 - 1) for r in range(r0, r1 + 1)} |
            {(r, c1 + 1) for r in range(r0, r1 + 1)})


def _votes(g, cells, labels):
    return tuple(int(sum(g[p] == color for p in cells)) for color in labels)


def _winner(g, cells, labels):
    a, b = _votes(g, cells, labels)
    _require(a != b, "every large object's landing contact must have a strict majority")
    return labels[int(b > a)]


def _records(g, d):
    records = []
    for piece in _components(g == d['cargo']):
        panel = _panel_of(piece, d['panels'])
        dest, fall = _land(piece, panel) if len(piece) >= 4 else (piece, 0)
        contact = _contact(dest, d['wall'])
        color = (_winner(g, contact, d['labels']) if len(piece) >= 4 else d['cargo'])
        records.append(dict(piece=piece, dest=dest, fall=fall, panel=panel,
                            contact=contact, color=color))
    return records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate and elect using only the given grid's visible geometry."""
    g = _grid(input_grid)
    d = _decode(g)
    records = _records(g, d)
    out = g.copy()
    out[g == d['cargo']] = d['bg']
    occupied = set()
    for rec in records:
        _require(not occupied & rec['dest'], "overlapping destinations")
        occupied.update(rec['dest'])
        for cell in rec['dest']:
            out[cell] = rec['color']
    return out


def _normalize(points):
    points = set(points)
    r0 = min(r for r, c in points)
    c0 = min(c for r, c in points)
    return tuple(sorted((r - r0, c - c0) for r, c in points))


def _orient(points, rng):
    points = list(points)
    for _ in range(int(rng.integers(4))):
        points = [(c, -r) for r, c in points]
    if rng.integers(2):
        points = [(r, -c) for r, c in points]
    return _normalize(points)


def _dims(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _shape(rng, maxh, maxw):
    for _ in range(100):
        mode = int(rng.integers(6))
        if mode == 0:
            h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            points = [(r, c) for r in range(h) for c in range(w)]
        elif mode == 1:
            points = [(0, c) for c in range(int(rng.integers(4, 8)))]
        elif mode == 2:
            h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            points = [(r, c) for r in range(h) for c in range(w)
                      if r in (0, h - 1) or c in (0, w - 1)]
        elif mode == 3:
            h, w = int(rng.integers(2, 6)), int(rng.integers(2, 5))
            points = [(r, 0) for r in range(h)] + [(h - 1, c) for c in range(w)]
        elif mode == 4:
            n = int(rng.integers(2, 5))
            points = [(r, c) for r in range(n) for c in (r, r + 1)]
        else:
            points = {(0, 0)}
            target = int(rng.integers(4, 13))
            while len(points) < target:
                p = sorted(points)[int(rng.integers(len(points)))]
                points.add(_neighbors(p)[int(rng.integers(4))])
        points = _orient(points, rng)
        h, w = _dims(points)
        if len(points) >= 4 and h <= maxh and w <= maxw:
            return points
    # A narrow lane can always hold this vertical tetromino.
    return tuple((r, 0) for r in range(4))


def _axis(sizes):
    bars, intervals, cursor = [0], [], 1
    for size in sizes:
        intervals.append((cursor, cursor + size - 1))
        cursor += size
        bars.append(cursor)
        cursor += 1
    return bars, intervals, cursor


def _base(row_sizes, col_sizes, roles):
    hb, rows, h = _axis(row_sizes)
    vb, cols, w = _axis(col_sizes)
    g = np.full((h, w), roles[0], dtype=np.int8)
    g[hb, :] = roles[1]
    g[:, vb] = roles[1]
    wall = np.zeros(g.shape, dtype=bool)
    wall[hb, :] = True
    wall[:, vb] = True
    panels = [(a, b, c, d) for a, b in rows for c, d in cols]
    return g, wall, panels, hb, vb


def _place(shape, panel, left, fall):
    h, w = _dims(shape)
    top = panel[1] - h + 1 - fall
    return {(top + r, left + c) for r, c in shape}


def _separate(a, b):
    return not any(p in b or any(q in b for q in _neighbors(p)) for p in a)


def _pack(rng, panel, count):
    r0, r1, c0, c1 = panel
    h, w = r1 - r0 + 1, c1 - c0 + 1
    # Reserve at least one blank column between moving components.
    for _ in range(100):
        shapes = [_shape(rng, h - 1, max(1, (w - count + 1) // count))
                  for _ in range(count)]
        widths = [_dims(s)[1] for s in shapes]
        spare = w - sum(widths) - count + 1
        if spare < 0:
            continue
        gaps = rng.multinomial(spare, [1 / (count + 1)] * (count + 1))
        left = c0 + int(gaps[0])
        pieces = []
        for i, shape in enumerate(shapes):
            maximum = h - _dims(shape)[0]
            fall = (0 if rng.random() < .15 else
                    int(rng.integers(1, maximum + 1)))
            pieces.append(_place(shape, panel, left, fall))
            left += widths[i] + 1 + int(gaps[i + 1])
        return pieces
    raise RuntimeError('packing failed')


def _notch(h, w):
    # Its bounding box touches many rim cells that the actual shape misses.
    return _normalize({(0, c) for c in range(w - 1)} |
                      {(r, w - 2) for r in range(h - 1)} |
                      {(h - 2, w - 1), (h - 1, w - 1)})


def _add_fixed(rng, g, pieces, panels, cargo, count, force_three=False):
    moving = [p for p in pieces if len(p) >= 4]
    swept = set()
    final = set()
    for p in moving:
        dest, fall = _land(p, _panel_of(p, panels))
        final.update(dest)
        swept.update((r + t, c) for r, c in p for t in range(fall + 1))
    fixed = set()
    for i in range(count):
        size = 3 if i == 0 and force_three else int(rng.integers(1, 4))
        for _ in range(200):
            p = panels[int(rng.integers(len(panels)))]
            s = ([(0, 0), (0, 1), (1, 0)] if size == 3 else
                 [(0, c) for c in range(size)])
            s = _orient(s, rng)
            sh, sw = _dims(s)
            top = int(rng.integers(p[0], p[1] - sh + 2))
            left = int(rng.integers(p[2], p[3] - sw + 2))
            candidate = {(top + r, left + c) for r, c in s}
            if i == 0 and force_three and max(r for r, c in candidate) == p[1]:
                continue
            occupied = set().union(*pieces) if pieces else set()
            if (not candidate & swept and _separate(candidate, occupied) and
                    _separate(candidate, final | fixed)):
                pieces.append(candidate)
                fixed.update(candidate)
                break
        else:
            raise RuntimeError('fixed pieces did not fit')
    for p in pieces:
        for cell in p:
            g[cell] = cargo


def _random_sizes(rng, count, low, high):
    sizes = list(map(int, rng.integers(low, high + 1, size=count)))
    while sum(sizes) + count + 1 > 30:
        sizes[int(np.argmax(sizes))] -= 1
    return sizes


def _make_pair(rng, roles, mode):
    bg, structure, cargo, l0, l1 = roles
    if mode == 'threshold':
        rs = [int(rng.integers(6, 10))]
        cs = [int(rng.integers(8, 13)), int(rng.integers(4, 8))]
    elif mode == 'shared':
        rs = ([int(rng.integers(6, 10))] if rng.integers(2) else
              list(map(int, rng.integers(6, 10, size=2))))
        cs = list(map(int, rng.integers(5, 9, size=3 if len(rs) == 1 else 2)))
    elif mode == 'transfer':
        rs = list(map(int, rng.integers(6, 12, size=int(rng.integers(1, 3)))))
        cs = [int(rng.integers(12, 19)), int(rng.integers(4, 8))]
    else:
        nr, nc = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        if nr * nc == 1:
            nc = 2
        rs, cs = _random_sizes(rng, nr, 5, 11), _random_sizes(rng, nc, 4, 12)
    g, wall, panels, hb, vb = _base(rs, cs, roles)
    pieces, forced = [], {}
    special = []
    if mode == 'threshold':
        p = panels[0]
        square = ((0, 0), (0, 1), (1, 0), (1, 1))
        bar = tuple((0, c) for c in range(4))
        pieces = [_place(square, p, p[2], int(rng.integers(1, rs[0] - 1))),
                  _place(bar, p, p[3] - 3, int(rng.integers(1, rs[0])))]
        special = [(pieces[0], l0), (pieces[1], l1)]
    elif mode == 'shared':
        h, w = int(rng.integers(4, 6)), int(rng.integers(3, 5))
        shape = _notch(h, w)
        mirror = _normalize((r, -c) for r, c in shape)
        p, q = panels[0], panels[1]
        pieces = [_place(shape, p, p[3] - w + 1, int(rng.integers(1, rs[0] - h + 1))),
                  _place(mirror, q, q[2], int(rng.integers(1, rs[0] - h + 1)))]
        # Both landed shapes touch the same two vertical-divider labels.
        for rec_piece, panel in zip(pieces, (p, q)):
            dest, _ = _land(rec_piece, panel)
            for cell in _contact(dest, wall):
                forced[cell] = l1 if cell[0] == panel[1] + 1 else l0
            r0, r1, c0, c1 = (min(r for r, c in dest), max(r for r, c in dest),
                              min(c for r, c in dest), max(c for r, c in dest))
            box = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
            for cell in _contact(box, wall) - _contact(dest, wall):
                forced[cell] = l1
        ring_panel = panels[2]
        rh, rw = int(rng.integers(3, 5)), int(rng.integers(3, 5))
        ring = [(r, c) for r in range(rh) for c in range(rw)
                if r in (0, rh - 1) or c in (0, rw - 1)]
        pieces.append(_place(ring, ring_panel, ring_panel[2],
                             int(rng.integers(1, ring_panel[1] - ring_panel[0] - rh + 2))))
    else:
        for i, panel in enumerate(panels):
            if i == 0 and mode == 'transfer':
                count = int(rng.integers(3, 5))
            elif i == 0 or rng.random() < .65:
                count = int(rng.integers(1, 3)) if panel[3] - panel[2] >= 7 else 1
            else:
                continue
            pieces.extend(_pack(rng, panel, count))
    _add_fixed(rng, g, pieces, panels, cargo, int(rng.integers(1, 5)),
               force_three=mode == 'threshold')
    _require(np.count_nonzero(g == bg) > 2 * np.count_nonzero(g == cargo),
             'interior background too sparse')
    contacts = [(_contact(_land(p, _panel_of(p, panels))[0], wall), p)
                for p in pieces if len(p) >= 4]
    # Sample genuine local elections, including single votes and mixed ballots.
    eligible = [(r, c) for r, c in map(tuple, np.argwhere(wall).tolist())
                if not (r in hb and c in vb)]
    exterior = [p for p in eligible if p[0] in (0, g.shape[0] - 1) or
                p[1] in (0, g.shape[1] - 1)]
    for _ in range(150):
        marked = g.copy()
        for cell in eligible:
            if rng.random() < .25:
                marked[cell] = (l0, l1)[int(rng.integers(2))]
        for cells, piece in contacts:
            if not any(cell in forced for cell in cells):
                ordered = sorted(cells)
                rng.shuffle(ordered)
                n = int(rng.integers(1, len(ordered) + 1))
                # An odd number avoids tied elections in an isolated footprint.
                if n % 2 == 0:
                    n -= 1
                for cell in cells:
                    marked[cell] = structure
                for cell in ordered[:n]:
                    marked[cell] = (l0, l1)[int(rng.integers(2))]
        for piece, color in special:
            cells = sorted(_contact(_land(piece, panels[0])[0], wall))
            rng.shuffle(cells)
            for cell in cells:
                marked[cell] = structure
            # Square: 2:1; bar: 3:1, both with genuine competition.
            n = 3 if len(cells) == 4 else min(4, len(cells))
            for j, cell in enumerate(cells[:n]):
                marked[cell] = (l1 if color == l0 else l0) if j == 0 else color
        for cell, color in forced.items():
            marked[cell] = color
        # These two visible colors make role decoding independent of training.
        for color in (l0, l1):
            if not any(marked[cell] == color for cell in exterior):
                choices = [p for p in exterior if p not in forced and
                           all(p not in cells for cells, _ in contacts)]
                if not choices:
                    break
                marked[choices[int(rng.integers(len(choices)))]] = color
        try:
            d = _decode(marked)
            records = _records(marked, d)
            if not any(r['fall'] > 0 for r in records):
                continue
            out = solve(marked)
            _analyze_pair(marked, out)
            return {'input': marked, 'output': out}
        except ValueError:
            continue
    raise RuntimeError('could not populate valid contact votes')


def _array_key(g):
    return tuple(g.shape), np.asarray(g, dtype=np.int8).tobytes()


def _analyze_pair(g, out):
    """Check geometry, sweeps, conservation and exact painted footprints."""
    g, out = _grid(g), _grid(out)
    _require(g.shape == out.shape, 'shape changed')
    d = _decode(g)
    recs = _records(g, d)
    small = [r for r in recs if len(r['piece']) < 4]
    large = [r for r in recs if len(r['piece']) >= 4]
    _require(small and large and any(r['fall'] for r in large), 'nontrivial size classes required')
    _require(np.array_equal(g[d['wall']], out[d['wall']]), 'rim changed')
    occupied = set()
    for rec in recs:
        _require(_separate(rec['dest'], occupied), 'destinations touch or overlap')
        occupied.update(rec['dest'])
        _require(all(out[p] == rec['color'] for p in rec['dest']), 'wrong destination paint')
        if len(rec['piece']) >= 4:
            sweep = {(r + t, c) for r, c in rec['piece'] for t in range(rec['fall'] + 1)}
            _require(not any(sweep & other['piece'] for other in recs if other is not rec),
                     'fall passes through another object')
    expected_cargo = sum(len(r['piece']) for r in recs)
    _require(np.count_nonzero((out != d['bg']) & ~d['wall']) == expected_cargo,
             'object area not conserved')
    remaining = ~d['wall'].copy()
    for p in occupied:
        remaining[p] = False
    _require(np.all(out[remaining] == d['bg']), 'stray interior paint')
    counts = Counter(r['panel'] for r in large)
    mixed = any(len({r['color'] for r in large if r['panel'] == p}) > 1 for p in counts)
    whole_trap = bottom_trap = box_trap = 0
    owners = Counter(p for r in large for p in r['contact'] if g[p] in d['labels'])
    for rec in large:
        a, b = _votes(g, _boundary(rec['panel']), d['labels'])
        whole_trap += a != b and d['labels'][int(b > a)] != rec['color']
        bottom = {p for p in rec['contact'] if p[0] == rec['panel'][1] + 1}
        a, b = _votes(g, bottom, d['labels'])
        bottom_trap += a != b and d['labels'][int(b > a)] != rec['color']
        dest = rec['dest']
        r0, r1 = min(r for r, c in dest), max(r for r, c in dest)
        c0, c1 = min(c for r, c in dest), max(c for r, c in dest)
        box = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        a, b = _votes(g, _contact(box, d['wall']), d['labels'])
        box_trap += a != b and d['labels'][int(b > a)] != rec['color']
    return dict(roles=(d['bg'], d['structure'], d['cargo'], *d['labels']),
                small_sizes=tuple(sorted(len(r['piece']) for r in small)),
                large_sizes=tuple(sorted(len(r['piece']) for r in large)),
                shapes=tuple(_normalize(r['piece']) for r in large),
                falls=tuple(r['fall'] for r in large),
                winners={r['color'] for r in large}, mixed=mixed,
                shared=sum(n > 1 for n in owners.values()),
                whole_trap=whole_trap, bottom_trap=bottom_trap, box_trap=box_trap,
                max_per_room=max(counts.values()), lattice=(len(d['rows']), len(d['cols'])),
                recs=recs)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'split type')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _check_counts(a, b)
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            g, out = _grid(pair['input']), _grid(pair['output'])
            key = _array_key(g)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            infos[split].append(_analyze_pair(g, out))
    train = infos['train']
    _require(len({i['roles'] for i in train + infos['test']}) == 1, 'palette roles changed')
    _require(any(3 in i['small_sizes'] for i in train) and
             any(4 in i['large_sizes'] for i in train), 'missing 3/4 boundary evidence')
    _require(any(len(r['piece']) == 3 and max(p[0] for p in r['piece']) < r['panel'][1]
                 for i in train for r in i['recs']) and
             any(len(r['piece']) == 4 and r['fall'] > 0
                 for i in train for r in i['recs']), '3/4 mobility not visibly demonstrated')
    _require(any(i['mixed'] for i in train), 'missing independent elections within one room')
    _require(any(i['whole_trap'] for i in train), 'whole-perimeter shortcut survives')
    _require(any(i['bottom_trap'] and i['box_trap'] and i['shared'] for i in train),
             'missing shape/contact/shared-boundary evidence')
    _require(set.union(*(i['winners'] for i in train)) == set(train[0]['roles'][3:]),
             'both winning labels must be demonstrated')
    _require(all(i['max_per_room'] <= 2 for i in train) and
             infos['test'][0]['max_per_room'] in (3, 4), 'missing within-room count shift')
    return None


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, size=5, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            mode = ('threshold' if split == 'train' and index == 0 else
                    'shared' if split == 'train' and index == 1 else
                    'transfer' if split == 'test' and index == 0 else 'ordinary')
            for _ in range(100):
                try:
                    pair = _make_pair(rng, roles, mode)
                    key = _array_key(pair['input'])
                    if key in seen:
                        continue
                    info = _analyze_pair(pair['input'], pair['output'])
                    if mode == 'threshold' and not (info['mixed'] and info['whole_trap']):
                        continue
                    if mode == 'shared' and not (info['shared'] and info['box_trap'] and info['bottom_trap']):
                        continue
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
                except (ValueError, RuntimeError):
                    continue
            else:
                raise RuntimeError('bounded construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((_array_key(p['input']), _array_key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed suite with measured, non-color structural variation."""
    dimensions, layouts, shapes, sizes, counts, palettes, puzzles = (set() for _ in range(7))
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                g, out = p['input'], p['output']
                info = _analyze_pair(g, out)
                assert np.array_equal(solve(g.tolist()), out)
                dimensions.add(g.shape)
                layouts.add(info['lattice'])
                shapes.update(info['shapes'])
                sizes.update(info['large_sizes'])
                counts.add((len(info['small_sizes']), len(info['large_sizes'])))
                palettes.add(info['roles'])
                for name in ('whole_trap', 'box_trap', 'bottom_trap', 'shared', 'mixed'):
                    totals[name] += info[name]
                totals['aligned'] += 0 in info['falls']
                totals['fall_one'] += 1 in info['falls']
                totals['pairs'] += 1
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
    assert len(puzzles) == 42 and len(palettes) >= 35
    assert len(dimensions) >= 45 and len(layouts) >= 8 and len(shapes) >= 70
    assert len(sizes) >= 10 and len(counts) >= 12
    assert totals['aligned'] >= 5 and totals['fall_one'] >= 15
    assert all(totals[n] >= 42 for n in ('whole_trap', 'box_trap', 'bottom_trap', 'shared', 'mixed'))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(10000 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=totals['pairs'], distinct_puzzles=len(puzzles),
                grid_shapes=len(dimensions), lattices=len(layouts), object_shapes=len(shapes),
                object_sizes=len(sizes), count_mixtures=len(counts), palettes=len(palettes),
                evidence=dict(totals))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
