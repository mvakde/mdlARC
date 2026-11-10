"""Registered aperture printing: enclosed leading voids are the stencils.

Only the input decoder is used by solve; it never reads a puzzle or RNG state.
Generation uses independently constructed plate/room records to render labels.
"""

from collections import Counter
import numpy as np


_DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _mode(values, excluded=()):
    counts = Counter(int(c) for c in values if int(c) not in excluded)
    _require(bool(counts), "missing color role")
    maximum = max(counts.values())
    winners = [c for c, n in counts.items() if n == maximum]
    _require(len(winners) == 1, "ambiguous color role")
    return winners[0]


def _decode(input_grid):
    """Read a rail, carriers, label rays, cavities and original receiver pixels."""
    original = _grid(input_grid)
    bg = _mode(original.ravel())
    scenes = []
    for turn in range(4):
        g = np.rot90(original, turn)
        h, w = g.shape
        for rail_col in range(1, w - 1):
            rail = int(g[0, rail_col])
            if rail == bg or not np.all(g[:, rail_col] == rail):
                continue
            if np.count_nonzero(g == rail) != h:
                continue
            body = _mode(g.ravel(), (bg, rail))
            if not np.any(g[:, :rail_col] == body) or np.any(g[:, rail_col + 1:] == body):
                continue
            scenes.append((g, turn, bg, rail, body, rail_col))
    _require(len(scenes) == 1, "rail, body and working side must be unique")
    g, turn, bg, rail, body, col = scenes[0]
    plates, instructions, sources = [], [], set()
    for cells in _components(g[:, :col] != bg):
        r0, r1, c0, c1 = _bbox(cells)
        _require(sum(g[p] == body for p in cells) >= 8, "carrier too small")
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        regions = _components(~mask)
        holes = [region for region in regions if not any(
            r in (0, mask.shape[0] - 1) or c in (0, mask.shape[1] - 1)
            for r, c in region)]
        label_mask = np.zeros_like(mask)
        for r, c in cells:
            if int(g[r, c]) != body:
                label_mask[r - r0, c - c0] = True
        plate = dict(cells=cells, bbox=(r0, r1, c0, c1), mask=mask,
                     shift=col - 1 - c1, holes=holes, records=[])
        _require(plate['shift'] >= 0, "carrier crosses rail")
        covered_holes = set()
        for labels in _components(label_mask):
            _require(len(labels) == 2, "labels must be separate two-cell pairs")
            a, b = sorted(labels)
            options = []
            for s, t in ((a, b), (b, a)):
                dr, dc = t[0] - s[0], t[1] - s[1]
                q = t[0] + dr, t[1] + dc
                if (dr, dc) not in _DIRS or not (0 <= q[0] < mask.shape[0] and
                                                0 <= q[1] < mask.shape[1]):
                    continue
                if not mask[q]:
                    options.append((s, t, q, (dr, dc)))
            _require(len(options) == 1, "registration must have a unique reading")
            s, t, q, direction = options[0]
            source, target = int(g[r0 + s[0], c0 + s[1]]), int(g[r0 + t[0], c0 + t[1]])
            _require(source != target and source not in sources, "ambiguous local mapping")
            sources.add(source)
            region = next(x for x in regions if q in x)
            closed = region in holes
            if closed:
                covered_holes.add(holes.index(region))
            contact = {r for r, c in region if c == mask.shape[1] - 2 and
                       mask[r, mask.shape[1] - 1]}
            active = closed and bool(contact)
            rec = dict(source=source, target=target, q=q, direction=direction,
                       closed=closed, active=active, contact=contact,
                       beam_rows={r0 + r for r, c in region},
                       region=region, offsets=tuple(sorted((r - q[0], c - q[1])
                                                           for r, c in region)), plate=plate)
            plate['records'].append(rec)
            instructions.append(rec)
        _require(len(covered_holes) == len(holes), "each enclosed cavity needs one label")
        _require(sum(rec['closed'] for rec in plate['records']) == len(holes),
                 "cavity has multiple registrations")
        plates.append(plate)
    _require(plates and instructions, "missing carriers or labels")
    # No hidden category can determine whether a cavity is active.
    receiver = g[:, col + 1:]
    _require(not np.any((receiver == rail) | (receiver == body)), "reserved color in receiver")
    translated = set()
    for plate in plates:
        dest = {(r, c + plate['shift']) for r, c in plate['cells']}
        _require(not translated & dest, "aligned carriers collide")
        translated |= dest
    return dict(grid=g, turn=turn, bg=bg, rail=rail, body=body, col=col,
                plates=plates, records=instructions)


def _paint_plan(scene, records=None, offsets_override=None):
    g, col, bg = scene['grid'], scene['col'], scene['bg']
    records = [x for x in scene['records'] if x['active']] if records is None else records
    paint, anchors, visits, clipped = {}, set(), Counter(), set()
    for rec in records:
        positions = [tuple(map(int, p)) for p in np.argwhere(g == rec['source'])
                     if p[1] > col and int(p[0]) in rec['beam_rows']]
        offsets = rec['offsets'] if offsets_override is None else offsets_override(rec)
        for ar, ac in positions:
            anchors.add((ar, ac))
            for dr, dc in offsets:
                r, c = ar + dr, ac + dc
                if not (0 <= r < g.shape[0] and col < c < g.shape[1]):
                    clipped.add('top' if r < 0 else 'bottom' if r >= g.shape[0]
                                else 'rail' if c <= col else 'outer')
                    continue
                _require((r, c) not in paint or paint[r, c] == rec['target'],
                         "incompatible print overlap")
                paint[r, c] = rec['target']
                visits[r, c] += 1
    return paint, anchors, visits, clipped


def _render(scene, records=None, offsets_override=None):
    g, col, bg = scene['grid'], scene['col'], scene['bg']
    out = g.copy()
    out[:, :col] = bg
    for plate in scene['plates']:
        for r, c in plate['cells']:
            out[r, c + plate['shift']] = g[r, c]
    paint, anchors, _, _ = _paint_plan(scene, records, offsets_override)
    for p in anchors:
        out[p] = bg
    for p, color in paint.items():
        out[p] = color
    return np.rot90(out, -scene['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_decode(input_grid))


def _shape_key(offsets):
    r0 = min(r for r, c in offsets)
    c0 = min(c for r, c in offsets)
    return tuple(sorted((r - r0, c - c0) for r, c in offsets))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _shortcut_witnesses(scene):
    """Compare specific simpler interpretations using only decoded pixels."""
    correct = _render(scene)
    def differs(offset_function):
        try:
            return not np.array_equal(correct, _render(scene, offsets_override=offset_function))
        except AssertionError:
            # An alternative that demands contradictory paint also fails.
            return True
    def rectangle(rec):
        r0, r1, c0, c1 = _bbox(rec['offsets'])
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    def centered(rec):
        r0, r1, c0, c1 = _bbox(rec['offsets'])
        # Replace the visible origin with the top/left middle of the box.
        return [(r - (r0 + r1) // 2, c - (c0 + c1) // 2) for r, c in rec['offsets']]
    def row_local(rec):
        return [(r, c) for r, c in rec['offsets'] if r + rec['q'][0] in rec['contact']]
    active = {x['source']: x['target'] for x in scene['records'] if x['active']}
    active_records = {x['source']: x for x in scene['records'] if x['active']}
    paint, anchors, visits, clipped = _paint_plan(scene)
    contact_records = [dict(rec, beam_rows={rec['plate']['bbox'][0] + r for r in rec['contact']})
                       for rec in active_records.values()]
    try:
        contact_projection_differs = not np.array_equal(correct, _render(scene, records=contact_records))
    except AssertionError:
        contact_projection_differs = True
    return dict(single_pixel=differs(lambda rec: [(0, 0)]),
                bounding_rectangle=differs(rectangle), center_origin=differs(centered),
                contact_rows_only=differs(row_local),
                contact_projection_only=contact_projection_differs,
                recursive=any(color in active and p[0] in active_records[color]['beam_rows'] and
                              scene['grid'][p] != color for p, color in paint.items()),
                ignore_beam=any(rec['active'] and any(
                    scene['grid'][r, c] == rec['source'] and r not in rec['beam_rows'] and
                    (r, c) not in paint for r in range(scene['grid'].shape[0])
                    for c in range(scene['col'] + 1, scene['grid'].shape[1]))
                    for rec in scene['records']),
                ignore_gate=any(not rec['active'] and any(
                    scene['grid'][r, c] == rec['source'] and (r, c) not in paint
                    for r in range(scene['grid'].shape[0])
                    for c in range(scene['col'] + 1, scene['grid'].shape[1]))
                    for rec in scene['records']))


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " out of range")
    return int(value)


def _carrier(rng, kinds, body, mappings, *, partial=False, geometry=None, tail=0, crossing_chain=False):
    """Carve connected row-convex apertures into a solid structural carrier."""
    width = int(rng.integers(6, 10)) if geometry is None else geometry[0]
    heights = [int(rng.integers(2 if partial else 1, 6 if partial else 4))
               for _ in kinds] if geometry is None else [len(geometry[1])]
    if crossing_chain:
        heights[0] = int(rng.integers(4, 6))
    h = 4 + sum(heights) + 2 * (len(kinds) - 1)
    a = np.full((h, width + tail), -1, dtype=int)
    a[:, :width] = body
    if tail:
        a[int(rng.choice([0, h - 1])), width:] = body
    records = []
    row = 2
    for i, (kind, hh, (source, target)) in enumerate(zip(kinds, heights, mappings)):
        right = width - 2 if kind != 'thick' else width - int(rng.integers(3, 5))
        if geometry is None:
            left = int(rng.integers(2, max(3, right)))
            ends = []
            for rr in range(hh):
                l = min(right, max(2, left + int(rng.integers(-1, 2))))
                end = right
                if partial and kind == 'active' and hh >= 2 and rr != 0:
                    end = max(l, right - int(rng.integers(1, 3)))
                ends.append((l, end))
            # Shared rightward spine keeps all rows in a single component.
            common = min(end for l, end in ends)
            ends = [(min(l, common), end) for l, end in ends]
        else:
            ends = list(geometry[1])
        hole = {(row + rr, c) for rr, (l, end) in enumerate(ends)
                for c in range(l, end + 1)}
        for p in hole:
            a[p] = -1
        direction = (0, 1)
        qr = row + int(rng.integers(hh))
        q = (qr, ends[qr - row][0])
        # Top/bottom wall labels provide visibly different off-center origins.
        if crossing_chain and i == 0 or geometry is None and rng.random() < .55:
            direction = (1, 0) if crossing_chain and i == 0 or rng.random() < .5 else (-1, 0)
            qr = row if direction == (1, 0) else row + hh - 1
            l, end = ends[qr - row]
            q = qr, int(rng.integers(l, end + 1))
        dr, dc = direction
        s, t = (q[0] - 2 * dr, q[1] - 2 * dc), (q[0] - dr, q[1] - dc)
        if a[s] != body or a[t] != body:
            raise ValueError("label placement collision")
        a[s], a[t] = source, target
        if kind == 'open':
            # Break the extreme wall on a different row if possible.
            door_row = row + int(rng.integers(hh))
            a[door_row, ends[door_row - row][1] + 1:width] = -1
        active = kind == 'active' and not tail
        records.append(dict(source=source, target=target, q=q, active=active,
                            beam_rows={r for r, c in hole},
                            offsets=tuple(sorted((r - q[0], c - q[1]) for r, c in hole))))
        row += hh + 2
    _require(len(_components(a != -1)) == 1, "constructed carrier split")
    return a, records


def _build_pair(rng, roles, teaching, turn):
    bg, rail, body, colors = roles
    a, b, c = colors[:3]
    carriers = []
    if teaching == 0:
        # Identical negative geometry: closed, open, and displaced extreme.
        width = int(rng.integers(6, 9))
        hh = int(rng.integers(2, 4))
        left = int(rng.integers(2, width - 3))
        ends = [(left + (rr % 2), width - 2) for rr in range(hh)]
        geometry = (width, ends)
        for kind, mapping, tail in [('active', (a, b), 0), ('open', (b, c), 0),
                                    ('active', (c, a), int(rng.integers(1, 3)))]:
            carriers.append(_carrier(rng, [kind], body, [mapping], geometry=geometry, tail=tail))
    elif teaching == 1:
        carriers.append(_carrier(rng, ['active', 'active'], body, [(a, c), (c, b)],
                                 partial=True, crossing_chain=True))
        if rng.random() < .6:
            carriers.append(_carrier(rng, ['thick'], body, [(b, a)]))
    else:
        n = int(rng.integers(1, 4))
        room_total = 0
        specs = []
        for i in range(n):
            room_count = int(rng.integers(1, 4)) if n == 1 else int(rng.integers(1, 3))
            room_count = min(room_count, 5 - room_total)
            kinds = [str(rng.choice(['active', 'active', 'active', 'thick', 'open']))
                     for _ in range(room_count)]
            if i == 0:
                kinds[0] = 'active'
            specs.append(kinds)
            room_total += room_count
        sources = list(map(int, rng.choice(colors, room_total, replace=False)))
        shared = int(rng.choice(colors)) if rng.random() < .25 else None
        index = 0
        for kinds in specs:
            mappings = []
            for kind in kinds:
                source = sources[index]
                target = shared if shared is not None and shared != source else int(
                    rng.choice([x for x in colors if x != source]))
                mappings.append((source, target))
                index += 1
            carriers.append(_carrier(rng, kinds, body, mappings, partial=rng.random() < .55))
    rng.shuffle(carriers)
    minimum_h = sum(x.shape[0] for x, _ in carriers) + len(carriers) - 1
    if minimum_h > 28:
        raise ValueError("carriers too tall")
    h = int(rng.integers(minimum_h + 1, min(30, minimum_h + 6) + 1))
    widest = max(x.shape[1] for x, _ in carriers)
    col = widest + int(rng.integers(1, 5))
    if col > 21:
        raise ValueError("working region too wide")
    band = int(rng.integers(6, min(14, 29 - col) + 1))
    w = col + 1 + band
    g = np.full((h, w), bg, dtype=np.int8)
    g[:, col] = rail
    gaps = rng.multinomial(h - minimum_h, np.full(len(carriers) + 1, 1 / (len(carriers) + 1)))
    row = int(gaps[0])
    records = []
    built_plates = []
    for i, (carrier, rr) in enumerate(carriers):
        shift = 0 if rng.random() < .25 else int(rng.integers(1, col - carrier.shape[1] + 1))
        if teaching == 0 and i == 0:
            shift = max(1, shift)
        start = col - carrier.shape[1] - shift
        region = g[row:row + carrier.shape[0], start:start + carrier.shape[1]]
        occupied = carrier != -1
        region[occupied] = carrier[occupied]
        cells = {(row + int(r), start + int(c)) for r, c in np.argwhere(occupied)}
        built_plates.append((cells, shift))
        records.extend(dict(rec, beam_rows={row + r for r in rec['beam_rows']}) for rec in rr)
        row += carrier.shape[0] + 1 + int(gaps[i + 1])
    # Keep two touching label pairs from masquerading as a larger label group.
    scene = _decode(g)
    decoded = [x for x in scene['records'] if x['active']]
    expected = sorted((x['source'], x['target'], x['offsets']) for x in records if x['active'])
    _require(sorted((x['source'], x['target'], x['offsets']) for x in decoded) == expected,
             "carving and input interpretation differ")
    paint, anchors, consumed = {}, set(), set()
    locations = [(r, cc) for r in range(h) for cc in range(col + 1, w)]

    def place(rec, p, complete=False, silent=False):
        if g[p] != bg or p in anchors:
            return False
        if not silent and p[0] not in rec['beam_rows']:
            return False
        footprint = {(p[0] + dr, p[1] + dc) for dr, dc in rec['offsets']}
        inside = {q for q in footprint if 0 <= q[0] < h and col < q[1] < w}
        if complete and inside != footprint:
            return False
        if silent:
            if p in paint:
                return False
        elif any(q in paint and paint[q] != rec['target'] for q in inside):
            return False
        # Silent trigger examples must remain visibly unchanged.
        silent_sources = {x['source'] for x in records if not x['active']}
        if not silent and any(int(g[q]) in silent_sources for q in inside):
            return False
        g[p] = rec['source']
        anchors.add(p)
        if not silent:
            for q in inside:
                paint[q] = rec['target']
            consumed.add(p)
        return True

    for rec in sorted(records, key=lambda x: not x['active']):
        shuffled = [locations[int(i)] for i in rng.permutation(len(locations))]
        if teaching == 1 and rec['source'] == a:
            # A lower-row copy reaches the next aperture's beam in color c.
            # It must stay c instead of recursively firing c->b.
            shuffled.sort(key=lambda p: p[0] != max(rec['beam_rows']))
        silent = not rec['active']
        desired = int(rng.integers(1, 6)) if not silent else int(rng.integers(1, 4))
        placed = []
        for p in shuffled:
            if place(rec, p, complete=not silent, silent=silent):
                placed.append(p)
                break
        if not placed:
            raise ValueError("no complete printing position")
        if teaching == 1:
            desired = max(3, desired)
        for _ in range(desired - 1):
            nearby = [(placed[-1][0] + dr, placed[-1][1] + dc) for dr, dc in _DIRS]
            candidates = nearby + shuffled if rng.random() < .6 else shuffled
            for p in candidates:
                if 0 <= p[0] < h and col < p[1] < w and place(rec, p, silent=silent):
                    placed.append(p)
                    break
    # Same color, wrong projected row: these original pixels stay inert.
    for rec in records:
        if not rec['active']:
            continue
        candidates = [p for p in locations if p[0] not in rec['beam_rows'] and
                      p not in paint and g[p] == bg]
        if not candidates:
            raise ValueError("no off-beam source contrast")
        p = candidates[int(rng.integers(len(candidates)))]
        g[p] = rec['source']
        anchors.add(p)
    # Add locally inactive content; paints can cover anchors but not these cells.
    active_sources = {x['source'] for x in records if x['active']}
    inert = [x for x in colors if x not in active_sources]
    for _ in range(int(rng.integers(0, 5))):
        if not inert:
            break
        p = locations[int(rng.integers(len(locations)))]
        if g[p] == bg and p not in paint:
            g[p] = int(rng.choice(inert))
    # Independent construction rendering: no solver-decoded cavity records.
    out = g.copy()
    out[:, :col] = bg
    for cells, shift in built_plates:
        for r, cc in cells:
            out[r, cc + shift] = g[r, cc]
    for p in consumed:
        out[p] = bg
    for p, color in paint.items():
        out[p] = color
    inp, out = np.rot90(g, turn).copy(), np.rot90(out, turn).copy()
    _require(np.array_equal(solve(inp), out), "constructed output disagrees with input")
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    roles = (palette[0], palette[1], palette[2], palette[3:])
    base = int(rng.integers(4))
    mode = int(rng.integers(3))  # fixed, mixed, or an orientation shift at test
    requirements = list(map(int, rng.permutation(num_train)))
    puzzle, seen, familiar = {'train': [], 'test': []}, set(), set()
    for split, count in [('train', num_train), ('test', num_test)]:
        for index in range(count):
            teaching = 0 if split == 'train' and index == requirements[0] else (
                1 if split == 'train' and index == requirements[1] else None)
            turn = int(rng.integers(4)) if mode == 1 else (
                (base + 1) % 4 if mode == 2 and split == 'test' else base)
            for attempt in range(120):
                try:
                    pair = _build_pair(rng, roles, teaching, turn)
                    scene = _decode(pair['input'])
                    if teaching is not None:
                        witnesses = _shortcut_witnesses(scene)
                        required = ('single_pixel', 'bounding_rectangle', 'center_origin',
                                    'ignore_gate', 'ignore_beam') if teaching == 0 else (
                                    'contact_rows_only', 'contact_projection_only',
                                    'recursive', 'ignore_beam')
                        if not all(witnesses[k] for k in required):
                            continue
                    shapes = {_shape_key(x['offsets']) for x in scene['records'] if x['active']}
                    if _key(pair['input']) in seen or split == 'test' and not shapes - familiar:
                        continue
                    # The teaching examples resolve full-aperture copying,
                    # rather than cropping away rows without extreme contact.
                    if teaching == 1 and not any(x['active'] and len(x['contact']) <
                        len({r for r, c in x['region']}) for x in scene['records']):
                        continue
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    if split == 'train':
                        familiar |= shapes
                    break
                except (ValueError, AssertionError):
                    continue
            else:
                raise RuntimeError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid pair counts")
    seen, roles, train_records, train_shapes = set(), set(), [], set()
    stencil_evidence = partial_evidence = moved = False
    witnesses = Counter()
    changing = {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            scene = _decode(inp)
            if split == 'train':
                witnesses.update(k for k, v in _shortcut_witnesses(scene).items() if v)
            roles.add((scene['bg'], scene['rail'], scene['body']))
            _require(np.array_equal(_render(scene), out), "wrong output")
            canonical_out = np.rot90(out, scene['turn'])
            col, g = scene['col'], scene['grid']
            _require(np.array_equal(canonical_out[:, col], g[:, col]), "rail changed")
            paint, anchors, visits, clipped = _paint_plan(scene)
            receiver_positions = [(r, c) for r in range(g.shape[0]) for c in range(col + 1, g.shape[1])]
            _require(all(canonical_out[p] == g[p] for p in receiver_positions
                         if p not in paint and p not in anchors), "unaffected content changed")
            for rec in scene['records']:
                all_positions = [tuple(map(int, p)) for p in np.argwhere(g == rec['source']) if p[1] > col]
                positions = [p for p in all_positions if not rec['active'] or p[0] in rec['beam_rows']]
                _require(positions, "labeled region has no remote source evidence")
                if not rec['active']:
                    _require(all(p not in paint for p in positions), "silent sources obscured")
                else:
                    _require(any(p[0] not in rec['beam_rows'] and p not in paint for p in all_positions),
                             "missing off-beam source contrast")
                if split == 'train':
                    train_records.append(rec)
                    if rec['active']:
                        train_shapes.add(_shape_key(rec['offsets']))
                        changing.setdefault(rec['source'], set()).add(rec['target'])
                        partial_evidence |= len(rec['contact']) < len({r for r, c in rec['region']})
                        for ar, ac in positions:
                            footprint = {(ar + dr, ac + dc) for dr, dc in rec['offsets']}
                            if all(0 <= r < g.shape[0] and col < c < g.shape[1] for r, c in footprint):
                                stencil_evidence |= len(footprint) >= 3 and bool(footprint - anchors)
                    moved |= rec['plate']['shift'] > 0
            if split == 'test':
                _require(any(_shape_key(rec['offsets']) not in train_shapes for rec in scene['records']
                             if rec['active']), "test lacks fresh aperture geometry")
    _require(len(roles) == 1, "structural palette changed within puzzle")
    _require(any(x['active'] for x in train_records), "no active training")
    _require(any(not x['closed'] for x in train_records), "no open contrast")
    _require(any(x['closed'] and not x['active'] for x in train_records), "no recessed contrast")
    _require(stencil_evidence and partial_evidence and moved, "incomplete demonstrations")
    _require(any(len(v) > 1 for v in changing.values()), "local color mapping not demonstrated")
    _require(all(witnesses[k] for k in ('single_pixel', 'bounding_rectangle', 'center_origin',
                                       'contact_rows_only', 'contact_projection_only', 'recursive',
                                       'ignore_gate', 'ignore_beam')),
             "training admits a designated shortcut")


def self_test():
    """Bounded 42-seed structural audit and hand-specified semantic regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, masks, orientations, counts, directions, palettes = set(), set(), set(), set(), set(), set(), set()
    stats = Counter()
    for puzzle in puzzles:
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene = _decode(pair['input'])
                shapes.add(pair['input'].shape)
                orientations.add(scene['turn'])
                counts.add(len(scene['plates']))
                palettes.add((scene['bg'], scene['rail'], scene['body']))
                paint, anchors, visits, clipped = _paint_plan(scene)
                stats['pairs'] += 1
                stats['overlapping_pairs'] += any(v > 1 for v in visits.values())
                stats['clipped_pairs'] += bool(clipped)
                stats['already_aligned_plates'] += sum(p['shift'] == 0 for p in scene['plates'])
                stats['multi_aperture_plates'] += sum(sum(x['active'] for x in p['records']) >= 2
                                                       for p in scene['plates'])
                for rec in scene['records']:
                    directions.add(rec['direction'])
                    if rec['active']:
                        masks.add(_shape_key(rec['offsets']))
                        stats['partial_contact_apertures'] += len(rec['contact']) < len({r for r, c in rec['region']})
                # Omitting A's gate changes a preserved silent source at its
                # registration. Extra prints need not have compatible overlap.
                if split == 'train':
                    stats.update('shortcut_' + k for k, v in _shortcut_witnesses(scene).items() if v)
                    stats['gate_witnesses'] += sum(not x['active'] for x in scene['records'])
                    for rec in scene['records']:
                        if rec['active'] and len(rec['offsets']) > 1:
                            stats['copy_witnesses'] += 1
    _require(len(signatures) == 42 and len(palettes) >= 30, "seed diversity collapsed")
    _require(len(shapes) >= 45 and len(masks) >= 35, "geometric diversity collapsed")
    _require(orientations == {0, 1, 2, 3} and {1, 2, 3} <= counts, "layout variation missing")
    _require({(0, 1), (1, 0), (-1, 0)} <= directions, "registration directions missing")
    _require(min(stats[k] for k in ('overlapping_pairs', 'clipped_pairs', 'already_aligned_plates',
                                    'multi_aperture_plates', 'partial_contact_apertures')) > 10,
             "compatible structural variation missing")
    _require(stats['gate_witnesses'] >= 84 and stats['copy_witnesses'] >= 84, "parent ablation evidence missing")
    _require(tuple(_key(p['input']) for p in generate_puzzle(812, 7, 3)['train']) ==
             tuple(_key(p['input']) for p in generate_puzzle(812, 7, 3)['train']), "nondeterminism")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(23, a, b)
    for a, b in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)]:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Small explicit scene: the staircase-shaped void copies with a left-edge
    # registration; a target-only cell remains, and outer clipping is exact.
    g = np.zeros((8, 16), dtype=np.int8)
    g[:, 8] = 8
    g[1:7, 1:7] = 7
    g[3, 3:6], g[4, 4:6] = 0, 0
    g[3, 1:3] = (1, 2)
    g[3, 10], g[4, 14], g[0, 15], g[7, 10] = 1, 1, 2, 1
    expected = np.zeros_like(g)
    expected[:, 8] = 8
    # Actual alignment shift is one: original extreme 6 -> rail-adjacent 7.
    expected[1:7, 2:8] = g[1:7, 1:7]
    expected[0, 15] = 2
    expected[7, 10] = 1
    for ar, ac in [(3, 10), (4, 14)]:
        for dr, dc in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]:
            if ar + dr < 8 and ac + dc < 16:
                expected[ar + dr, ac + dc] = 2
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(g, turn)), np.rot90(expected, turn)),
                 "registered negative-space regression failed")
    # An opening destroys printing; a one-cell protrusion changes the extreme.
    for mutation in ('open', 'tail'):
        changed = g.copy()
        if mutation == 'open':
            changed[4, 6] = 0
        else:
            changed[1, 7] = 7
        scene = _decode(changed)
        _require(not any(x['active'] for x in scene['records']), "gate mutation ineffective")
        predicted = solve(changed)
        _require(predicted[3, 10] == 1 and predicted[4, 14] == 1, "silent source consumed")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a bad answer")
    return dict(seeds=42, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                aperture_geometries=len(masks), structural_palettes=len(palettes),
                orientations=sorted(orientations), plate_counts=sorted(counts), **dict(stats))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
