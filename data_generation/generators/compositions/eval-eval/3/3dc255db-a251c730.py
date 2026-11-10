"""Chromatic probes: unary hollow-frame counters select reference regions.

All interpretation is input-only. Construction paints labels from its sampled
cell sets; the solver separately recovers frames, templates and components.
No parent generator, source fixture, or generation metadata is used by solve.
"""
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _components(cells, diagonal=False):
    remaining = set(cells)
    steps = ([(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c]
             if diagonal else D4)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _bounds(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _box_cells(box):
    r, c, h, w = box
    return {(r + y, c + x) for y in range(h) for x in range(w)}


def _shift(cells, origin):
    r, c = origin
    return {(y + r, x + c) for y, x in cells}


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return g


def _project(point, direction):
    r, c = point
    dr, dc = direction
    return -r * dr - c * dc, r * dc - c * dr


def _unproject(point, direction):
    u, v = point
    dr, dc = direction
    return -u * dr + v * dc, -u * dc - v * dr


def _reference(grid, bg):
    """Recognize the sole reference by a border and an empty inner rim."""
    panels = []
    for color in sorted(set(map(int, grid.flat)) - {bg}):
        for comp in _components(map(tuple, np.argwhere(grid == color)), True):
            r, c, h, w = _bounds(comp)
            if min(h, w) < 7:
                continue
            border = {(r + y, c + x) for y in range(h) for x in range(w)
                      if y in (0, h - 1) or x in (0, w - 1)}
            if comp != border:
                continue
            inner = grid[r + 1:r + h - 1, c + 1:c + w - 1]
            fill = int(inner[0, 0])
            if fill == color or not (np.all(inner[0] == fill)
                    and np.all(inner[-1] == fill) and np.all(inner[:, 0] == fill)
                    and np.all(inner[:, -1] == fill)):
                continue
            motifs = _components(map(tuple, np.argwhere(inner != fill)))
            if not motifs:
                continue
            templates, repetitions = {}, Counter()
            valid = True
            for motif in motifs:
                counts = Counter(int(inner[p]) for p in motif)
                keys = [k for k, n in counts.items() if n == 1]
                if len(keys) != 1 or len(motif) < 9:
                    valid = False
                    break
                key = keys[0]
                ar, ac = next(p for p in motif if inner[p] == key)
                template = {(y - ar, x - ac): int(inner[y, x]) for y, x in motif}
                if key in templates and templates[key] != template:
                    valid = False
                    break
                templates[key] = template
                repetitions[key] += 1
            if not valid:
                continue
            keys = set(templates)
            if any(bg in t.values() for t in templates.values()):
                continue
            if any(col in keys for t in templates.values() for p, col in t.items()
                   if p != (0, 0)):
                continue
            # Four separated cardinal cones make every selected component
            # outward of its query frame, without rotating the template.
            if any(not _valid_template(t, key) for key, t in templates.items()):
                continue
            panels.append(dict(box=(r, c, h, w), fill=fill, border=color,
                               templates=templates, repetitions=dict(repetitions)))
    if len(panels) != 1:
        raise ValueError("input must contain one unambiguous reference panel")
    return panels[0]


def _valid_template(t, key):
    if t.get((0, 0)) != key:
        return False
    counts = Counter(t.values())
    if any(n < 2 for col, n in counts.items() if col != key):
        return False
    for r, c in t:
        if (r, c) == (0, 0):
            continue
        # Exclude the diagonals: each cell has exactly one dominant axis.
        if abs(r) == abs(c) or max(abs(r), abs(c)) > 4:
            return False
        axial, lateral = max(abs(r), abs(c)), min(abs(r), abs(c))
        if lateral > 2 or lateral >= axial:
            return False
    for dr, dc in DIRS:
        axial = [s for s in range(1, 5) if (dr * s, dc * s) in t]
        if len(axial) < 2 or axial != list(range(1, max(axial) + 1)):
            return False
        sector = {p for p in t if p != (0, 0) and p[0] * dr + p[1] * dc > 0
                  and abs(p[0] * dc - p[1] * dr) < p[0] * dr + p[1] * dc}
        if len(_components(sector)) != 1:
            return False
    return True


def _selected(template, direction, quantity):
    probe = direction[0] * quantity, direction[1] * quantity
    if probe not in template or quantity < 1:
        raise ValueError("probe must land on a visible decoration cell")
    color = template[probe]
    same = {p for p, col in template.items() if col == color and p != (0, 0)}
    region = next(c for c in _components(same) if probe in c)
    return color, region


def _frame_geometry(frame, direction):
    """Read monotone thin sides and a possible tip stem in cross-section."""
    local = {_project(p, direction) for p in frame}
    u0, _, depth, _ = _bounds(local)
    rows = [sorted(v for u, v in local if u == level)
            for level in range(u0, u0 + depth)]
    if depth < 3 or len(rows[0]) != 1:
        return None
    tipv = rows[0][0]
    left = right = tipv
    cavity, split = set(), False
    for index, row in enumerate(rows[1:], 1):
        if len(row) == 1 and not split and row[0] == tipv:
            continue
        if len(row) != 2:
            return None
        lo, hi = row
        if not (lo < tipv < hi and 0 <= left - lo <= 1 and 0 <= hi - right <= 1):
            return None
        split, left, right = True, lo, hi
        cavity.update(_unproject((u0 + index, v), direction) for v in range(lo + 1, hi))
    if not cavity:
        return None
    return _unproject((u0, tipv), direction), cavity


def _analyze(value):
    grid = _grid(value)
    bg = int(grid[0, 0])
    if not (np.all(grid[0] == bg) and np.all(grid[-1] == bg)
            and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)):
        raise ValueError("input has a uniform background margin")
    reference = _reference(grid, bg)
    panel_cells = _box_cells(reference['box'])
    templates = reference['templates']
    outside = np.ones(grid.shape, dtype=bool)
    r, c, h, w = reference['box']
    outside[r:r + h, c:c + w] = False
    objects = []
    for color in sorted(set(map(int, grid[outside])) - {bg} - set(templates)):
        for frame in _components(map(tuple, np.argwhere(outside & (grid == color))), True):
            if len(frame) < 5:
                continue
            candidates = []
            for direction in DIRS:
                geometry = _frame_geometry(frame, direction)
                if geometry is None:
                    continue
                tip, cavity = geometry
                if any(not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1])
                       for y, x in cavity):
                    continue
                values = {int(grid[p]) for p in cavity} - {bg}
                if len(values) != 1 or not values <= set(templates):
                    continue
                key = next(iter(values))
                payload = {p for p in cavity if grid[p] == key}
                quantity = len(payload)
                if not 1 <= quantity < len(cavity):
                    continue
                try:
                    selected_color, region = _selected(templates[key], direction, quantity)
                except ValueError:
                    continue
                stamp = _shift(region, tip)
                if any(not (0 < y < grid.shape[0] - 1 and 0 < x < grid.shape[1] - 1)
                       or grid[y, x] != bg for y, x in stamp):
                    continue
                candidates.append(dict(frame=frame, cavity=cavity, tip=tip,
                    direction=direction, frame_color=color, key=key, quantity=quantity,
                    payload=payload, selected_color=selected_color, region=region,
                    stamp=stamp, box=_bounds(frame)))
            if len(candidates) != 1:
                raise ValueError("frame has no unique valid probe interpretation")
            objects.extend(candidates)
    if not objects:
        raise ValueError("no query frames")
    boxes, stamps, represented = set(panel_cells), set(), set()
    for obj in objects:
        box = _box_cells(obj['box'])
        if not boxes.isdisjoint(box) or not stamps.isdisjoint(obj['stamp']):
            raise ValueError("overlapping queries or region stamps")
        boxes.update(box)
        stamps.update(obj['stamp'])
        represented.update(obj['frame'] | obj['payload'])
    if not boxes.isdisjoint(stamps):
        raise ValueError("a selected region enters a frame or reference")
    visible_outside = set(map(tuple, np.argwhere(outside & (grid != bg))))
    if represented != visible_outside:
        raise ValueError("unassigned exterior marks")
    return grid, bg, reference, sorted(objects, key=lambda o: o['tip'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, bg, _, objects = _analyze(input_grid)
    output = grid.copy()
    for obj in objects:
        for cell in obj['payload']:
            output[cell] = bg
        for cell in obj['stamp']:
            output[cell] = obj['selected_color']
    return output


def _template(rng, key, decorations, lengths, witness=None):
    """Generate connected four-arm masks with colored branches, not a bank."""
    for _ in range(100):
        t = {(0, 0): key}
        for index, ((dr, dc), length) in enumerate(zip(DIRS, lengths)):
            word = [int(_pick(rng, decorations)) for _ in range(length)]
            if witness is not None:
                # The chosen color also occurs in disconnected other arms:
                # selecting a component is distinguishable from a color mask.
                word[0] = witness[1]
                if index != witness[0]:
                    # A perpendicular count-two probe must not equal a
                    # rotated copy of the controlled arm's selected piece.
                    word[1] = witness[2]
            if witness is not None and index == witness[0]:
                word[1], word[2] = witness[1], witness[2]
                word[0] = witness[1]
            for s, color in enumerate(word, 1):
                t[dr * s, dc * s] = color
                if s < 2:
                    continue
                for side in (-1, 1):
                    width = int(rng.integers(0, min(2, s - 1) + 1))
                    if witness is not None and index == witness[0] and s in (2, 3):
                        width = 1 if side == (1 if s == 2 else -1) else 0
                    for lateral in range(1, width + 1):
                        p = dr * s + dc * side * lateral, dc * s - dr * side * lateral
                        t[p] = color if rng.random() < .8 else int(_pick(rng, decorations))
                        if witness is not None and index == witness[0]:
                            t[p] = color
        if _valid_template(t, key):
            return t
    raise AssertionError("could not construct a well-identified template")


def _profile(rng, compact=False, witness=False):
    # A canonical upward frame. Each side expands separately by diagonal or
    # vertical steps; frames need only be eight-connected.
    if witness:
        reaches, tail, stem = (2, 2), 3, int(rng.integers(0, 2))
    elif compact:
        reaches = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
        tail, stem = int(rng.integers(2, 4)), 0
    else:
        reaches = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        tail, stem = int(rng.integers(1, 4)), int(rng.integers(0, 2))
    a, b = reaches
    frame = {(u, 0) for u in range(stem + 1)}
    cavity, left, right, u = set(), 0, 0, stem
    while left < a or right < b:
        # Both sides separate at once, then may advance independently.
        dl = left < a and (left == 0 or right == b or rng.random() < .8)
        dr = right < b and (right == 0 or left == a or rng.random() < .8)
        if not dl and not dr:
            dl = left < a
            dr = not dl
        left, right, u = left + int(dl), right + int(dr), u + 1
        frame.update(((u, -left), (u, right)))
        cavity.update((u, v) for v in range(-left + 1, right))
    for _ in range(tail):
        u += 1
        frame.update(((u, -a), (u, b)))
        cavity.update((u, v) for v in range(-a + 1, b))
    return frame, cavity


def _payload(rng, cells, quantity, arrangement):
    candidates = sorted(cells)
    if arrangement == 'scattered':
        # Required two-cell witness: different eight-connected components.
        pairs = [(a, b) for i, a in enumerate(candidates) for b in candidates[i + 1:]
                 if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1]
        if quantity == 2 and pairs:
            return set(_pick(rng, pairs))
    if arrangement == 'compact':
        chosen = {_pick(rng, candidates)}
        while len(chosen) < quantity:
            frontier = sorted({(r + dr, c + dc) for r, c in chosen for dr, dc in D4}
                              & (cells - chosen))
            if not frontier:
                break
            chosen.add(_pick(rng, frontier))
        if len(chosen) == quantity:
            return chosen
    return {candidates[int(i)] for i in rng.choice(len(candidates), quantity, replace=False)}


def _palette(rng, key_count, forced_key=None, distinct_fill=False):
    palette = [int(c) for c in rng.permutation(10)]
    if forced_key is not None:
        palette.remove(forced_key)
        palette.insert(0, forced_key)
    keys = palette[:key_count]
    bg = palette[key_count]
    fill = bg if rng.random() < .35 and not distinct_fill else palette[key_count + 1]
    rest = [c for c in palette if c not in keys and c not in (bg, fill)]
    decorations = rest[:int(rng.integers(2, min(3, len(rest) - 1) + 1))]
    # Border may reuse a decoration; the empty inner rim separates them.
    border = int(_pick(rng, rest))
    frame_pool = [c for c in rest + [fill] if c != bg]
    return dict(keys=keys, bg=bg, fill=fill, decorations=decorations,
                border=border, frame_pool=frame_pool)


def _reference_patch(rng, palette, templates, ids):
    # All entries fit a single row. Their actual bounds, anchor offsets and
    # spacing vary; vertical arrangements result from whole-scene rotations.
    entries = []
    for index in ids:
        key = palette['keys'][index]
        entries.append((key, templates[key], _bounds(templates[key])))
    gap = int(rng.integers(1, 3))
    height = max(b[2] for _, _, b in entries) + 4
    width = sum(b[3] for _, _, b in entries) + gap * (len(entries) - 1) + 4
    if width > 28:
        gap = 1
        width = sum(b[3] for _, _, b in entries) + len(entries) - 1 + 4
    patch = np.full((height, width), palette['border'], dtype=np.int8)
    patch[1:-1, 1:-1] = palette['fill']
    left = 2
    for key, template, (lr, lc, h, w) in entries:
        top = int(rng.integers(2, height - h - 1))
        for (r, c), color in template.items():
            patch[top + r - lr, left + c - lc] = color
        left += w + gap
    return patch


def _query(rng, palette, templates, key, q, direction, profile=None, arrangement='random', compact=False,
           frame_color=None):
    for _ in range(30):
        frame, cavity = profile if profile is not None else _profile(rng, compact=compact)
        if len(cavity) > q:
            break
    else:
        raise AssertionError("insufficient hollow capacity")
    # Canonical u points inward; the tip remains at (0,0).
    frame = {_unproject(p, direction) for p in frame}
    cavity = {_unproject(p, direction) for p in cavity}
    payload = _payload(rng, cavity, q, arrangement)
    color, region = _selected(templates[key], direction, q)
    lr, lc, h, w = _bounds(frame | cavity | region)
    frame_color = int(_pick(rng, palette['frame_pool'])) if frame_color is None else frame_color
    inp = np.full((h, w), palette['bg'], dtype=np.int8)
    out = inp.copy()
    for r, c in frame:
        inp[r - lr, c - lc] = out[r - lr, c - lc] = frame_color
    for r, c in payload:
        inp[r - lr, c - lc] = key
    for r, c in region:
        out[r - lr, c - lc] = color
    return inp, out


def _shelf(sizes, width):
    r = c = row_height = 0
    places = []
    for h, w in sizes:
        if w > width:
            return None
        if c + w > width:
            r, c, row_height = r + row_height + 1, 0, 0
        places.append((r, c))
        c += w + 1
        row_height = max(row_height, h)
    return r + row_height, places


def _arrange(rng, patches, bg):
    sizes = [p[0].shape for p in patches]
    options = []
    for width in range(max(w for h, w in sizes), 29):
        packed = _shelf(sizes, width)
        if packed is not None and packed[0] <= 28:
            options.append((packed[0], width, packed[1]))
    if not options:
        raise ValueError("scene cannot fit within ARC bounds")
    min_h, min_w, fallback = _pick(rng, options)
    height = int(rng.integers(min_h + 2, 31))
    width = int(rng.integers(min_w + 2, 31))
    positions = None
    # Try free placements so queries can lie beside as well as beyond the
    # reference. Bounded shelf recovery uses exactly the same semantic rule.
    for _ in range(12):
        guarded, proposed = set(), []
        for h, w in sizes:
            candidates = [(r, c) for r in range(1, height - h) for c in range(1, width - w)]
            rng.shuffle(candidates)
            for r, c in candidates:
                box = _box_cells((r, c, h, w))
                if box.isdisjoint(guarded):
                    proposed.append((r, c))
                    guarded.update((y + dr, x + dc) for y, x in box
                                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))
                    break
            else:
                break
        if len(proposed) == len(patches):
            positions = proposed
            break
    if positions is None:
        dy = int(rng.integers(1, height - min_h))
        dx = int(rng.integers(1, width - min_w))
        positions = [(r + dy, c + dx) for r, c in fallback]
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    for (a, b), (r, c) in zip(patches, positions):
        h, w = a.shape
        inp[r:r + h, c:c + w] = a
        out[r:r + h, c:c + w] = b
    turns = int(rng.integers(4))
    inp, out = np.rot90(inp, turns), np.rot90(out, turns)
    if rng.random() < .5:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {'input': inp.copy(), 'output': out.copy()}


def _make_pair(rng, palette, templates, plans, compact=False, reference_ids=None):
    if reference_ids is None:
        reference_ids = list(range(len(templates)))
        if len(templates) == 1:
            extent = max(max(abs(r), abs(c)) for r, c in next(iter(templates.values())))
            reference_ids *= int(rng.integers(1, 3 if extent == 4 else 4))
        rng.shuffle(reference_ids)
    reference = _reference_patch(rng, palette, templates, reference_ids)
    patches = [(reference, reference.copy())]
    for plan in plans:
        patches.append(_query(rng, palette, templates, **plan, compact=compact))
    pair = _arrange(rng, patches, palette['bg'])
    if not np.array_equal(solve(pair['input']), pair['output']):
        raise AssertionError("constructed label and input reader disagree")
    return pair


def _fresh_pair(rng, shift=None, forced_key=None):
    for attempt in range(40):
        types = 3 if shift == 'keys' else int(rng.integers(1, 3))
        max_length = 4 if shift == 'distance' else 3
        palette = _palette(rng, types, forced_key)
        templates = {key: _template(rng, key, palette['decorations'],
                        [int(rng.integers(2, max_length + 1)) for _ in DIRS])
                     for key in palette['keys']}
        count = int(rng.integers(types if shift == 'keys' else 1, 5))
        compact = count >= 3 or attempt >= 20
        plans = []
        for index in range(count):
            key = palette['keys'][index] if shift == 'keys' and index < types else int(_pick(rng, palette['keys']))
            direction = _pick(rng, DIRS)
            length = max(s for s in range(1, 5) if (direction[0] * s, direction[1] * s) in templates[key])
            q = int(rng.integers(1, length + 1))
            if shift == 'distance' and index == 0:
                # The changed dictionary visibly supplies the new axial cell.
                lengths = [int(rng.integers(2, 5)) for _ in DIRS]
                lengths[DIRS.index(direction)] = 4
                templates[key] = _template(rng, key, palette['decorations'], lengths)
                q = 4
            plans.append(dict(key=key, q=q, direction=direction,
                              arrangement=_pick(rng, ('compact', 'random', 'scattered'))))
        # Three nine-wide references cannot fit in one row. Three-key pairs
        # use length <=3; a one-key distance-shift pair has at most two copies.
        try:
            return _make_pair(rng, palette, templates, plans, compact)
        except ValueError:
            continue
    raise AssertionError("bounded scene construction failed")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = _palette(rng, 2, distinct_fill=True)
    direction_index = int(rng.integers(4))
    direction, other = DIRS[direction_index], DIRS[(direction_index + 1) % 4]
    a, b = palette['decorations'][:2]
    templates = {key: _template(rng, key, palette['decorations'], [3] * 4,
                  (direction_index, a, b) if index == 0 else (direction_index, b, a))
                 for index, key in enumerate(palette['keys'])}
    profile = _profile(rng, witness=True)
    witness_frame_color = int(_pick(rng, palette['frame_pool']))
    k0, k1 = palette['keys']
    plans = [
        [dict(key=k0, q=1, direction=direction, profile=profile, arrangement='compact'),
         dict(key=k0, q=2, direction=direction, profile=profile, arrangement='compact'),
         dict(key=k0, q=3, direction=direction, profile=profile, arrangement='compact')],
        [dict(key=k0, q=2, direction=direction, profile=profile, arrangement='scattered'),
         dict(key=k1, q=2, direction=direction, profile=profile, arrangement='compact'),
         dict(key=k0, q=2, direction=other, profile=profile, arrangement='compact')],
    ]
    for pair_plans in plans:
        for plan in pair_plans:
            plan['frame_color'] = witness_frame_color
    puzzle, seen = {'train': [], 'test': []}, set()
    # Share the visible dictionary for the two controlled demonstrations.
    # Whole-scene augmentation moves both dictionary and probes together.
    for witness_plans in plans:
        pair = _make_pair(rng, palette, templates, witness_plans, reference_ids=[0, 1])
        puzzle['train'].append(pair)
        seen.add(_key(pair['input']))
    shift = str(_pick(rng, ('keys', 'distance')))
    for split, count in (('train', num_train - 2), ('test', num_test)):
        for index in range(count):
            for _ in range(10):
                pair = _fresh_pair(rng, shift if split == 'test' else None,
                                   forced_key=k0 if split == 'test' and index == 0 else None)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError("could not make distinct inputs")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(pairs):
    records, counts, colors = [], set(), set()
    component_failure = whole_failure = all_color_failure = ray_failure = False
    count_witness = key_witness = direction_witness = transparency_witness = False
    for pair in pairs:
        grid, bg, ref, objects = _analyze(pair['input'])
        local_records = []
        for obj in objects:
            q, key, direction = obj['quantity'], obj['key'], obj['direction']
            color, region = obj['selected_color'], obj['region']
            template = ref['templates'][key]
            sig = tuple(sorted((r, c, col) for (r, c), col in template.items()))
            records.append((sig, q, len(_components(obj['payload'], True)),
                            frozenset(region), color, direction, key))
            counts.add(q)
            colors.add(color)
            cq = len(_components(obj['payload'], True))
            component_failure |= _selected(template, direction, cq) != (color, region)
            whole_failure |= set(template) - {(0, 0)} != region
            all_color_failure |= {p for p, col in template.items() if col == color} != region
            ray = {(direction[0] * s, direction[1] * s) for s in range(1, q + 1)}
            ray_failure |= region != ray or color != key
            relative_frame = frozenset(_project((r - obj['tip'][0], c - obj['tip'][1]), direction)
                                       for r, c in obj['frame'])
            # Reflection changes which side is longer but not the evidence.
            geometry = min(tuple(sorted(relative_frame)), tuple(sorted((u, -v) for u, v in relative_frame)))
            local_records.append((key, q, direction, geometry, obj['frame_color'], color))
            _, _, rh, rw = _bounds(region)
            transparency_witness |= (ref['fill'] != bg and rh * rw > len(region)
                                      and len(region) >= 3)
        for i, (key, q, d, geometry, frame_color, color) in enumerate(local_records):
            for k2, q2, d2, g2, f2, c2 in local_records[i + 1:]:
                count_witness |= (key == k2 and d == d2 and geometry == g2 and frame_color == f2
                                  and q != q2 and color != c2)
                key_witness |= (key != k2 and q == q2 and d == d2 and geometry == g2
                                and frame_color == f2 and color != c2)
                direction_witness |= (key == k2 and q == q2 and d != d2 and geometry == g2
                                      and frame_color == f2 and color != c2)
    assert {1, 2, 3} <= counts and len(colors) >= 2, "training must vary meaningful probe results"
    assert component_failure and whole_failure and all_color_failure and ray_failure
    assert count_witness and key_witness and direction_witness and transparency_witness
    # Compare in dictionary coordinates. Whole-scene rotations may move the
    # controlled witnesses, so their signatures are also checked up to D4.
    normalized, addressed_regions = {}, {}
    for sig, q, components, region, color, direction, key in records:
        projected = tuple(sorted((*_project((r, c), direction), col) for r, c, col in sig))
        region_projected = frozenset(_project(p, direction) for p in region)
        # Reflection may reverse the lateral axis; canonicalize jointly.
        reflected = tuple(sorted((u, -v, col) for u, v, col in projected))
        flip = reflected < projected
        signature = min(projected, reflected)
        region_projected = frozenset((u, -v if flip else v) for u, v in region_projected)
        normalized.setdefault((signature, q, color, region_projected), set()).add(components)
        addressed_regions.setdefault(signature, {})[q] = (color, region_projected)
    assert any(len(v) >= 2 for v in normalized.values()), "equal counts need different payload connectivity"
    assert any({1, 2, 3} <= set(v) and v[1] == v[2] and v[2] != v[3]
               for v in addressed_regions.values()), "training must distinguish addressing from parity"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, m = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, train_keys, train_q, test_keys, test_q = set(), set(), set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            grid, bg, ref, objects = _analyze(inp)
            assert 1 <= len(ref['templates']) <= 3 and 1 <= len(objects) <= 4
            assert np.array_equal(solve(inp), out), "incorrect payload clearing or selected-component copying"
            r, c, h, w = ref['box']
            assert np.array_equal(inp[r:r + h, c:c + w], out[r:r + h, c:c + w])
            removed = set().union(*(o['payload'] for o in objects))
            added = set().union(*(o['stamp'] for o in objects))
            changed = set(map(tuple, np.argwhere(inp != out)))
            assert changed == removed | added, "only payload and selected region may change"
            assert all(out[p] == bg for p in removed)
            assert all(out[p] == o['frame_color'] for o in objects for p in o['frame'])
            (train_keys if split == 'train' else test_keys).add(len(ref['templates']))
            (train_q if split == 'train' else test_q).update(o['quantity'] for o in objects)
    _training_evidence(puzzle['train'])
    assert max(test_keys) > max(train_keys) or max(test_q) > max(train_q), "missing compositional shift"


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    metrics = {name: set() for name in ('shapes', 'frames', 'templates', 'counts',
               'directions', 'keys', 'objects', 'regions', 'backgrounds', 'fills', 'repetitions')}
    fingerprints = set()
    shifts, shared_colors, unequal_sides, same_color_disconnected = set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        tkeys, tq, xkeys, xq = [], [], [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                g, bg, ref, objects = _analyze(pair['input'])
                metrics['shapes'].add(g.shape)
                metrics['backgrounds'].add(bg)
                metrics['fills'].add(ref['fill'])
                metrics['keys'].add(len(ref['templates']))
                metrics['objects'].add(len(objects))
                metrics['repetitions'].update(ref['repetitions'].values())
                for template in ref['templates'].values():
                    metrics['templates'].add(tuple(sorted(template)))
                (tkeys if split == 'train' else xkeys).append(len(ref['templates']))
                (tq if split == 'train' else xq).extend(o['quantity'] for o in objects)
                for obj in objects:
                    metrics['counts'].add(obj['quantity'])
                    metrics['directions'].add(obj['direction'])
                    local = [_project((r - obj['tip'][0], c - obj['tip'][1]), obj['direction'])
                             for r, c in obj['frame']]
                    metrics['frames'].add(tuple(sorted(local)))
                    metrics['regions'].add(frozenset(obj['region']))
                    left, right = min(v for u, v in local), max(v for u, v in local)
                    unequal_sides += abs(left) != abs(right)
                    shared_colors += obj['frame_color'] == ref['border']
                    template = ref['templates'][obj['key']]
                    same_color_disconnected += len(_components({p for p, col in template.items()
                                              if col == obj['selected_color']})) > 1
                # Equivariance checks operate on inputs alone and labels from
                # direct painting. Templates keep their grid orientation.
                if seed < 3:
                    for turns in range(4):
                        for reflect in (False, True):
                            a, b = np.rot90(g, turns), np.rot90(pair['output'], turns)
                            if reflect:
                                a, b = np.fliplr(a), np.fliplr(b)
                            assert np.array_equal(solve(a), b)
                            perm = np.random.default_rng(712 + turns).permutation(10)
                            assert np.array_equal(solve(perm[a]), perm[b])
        shifts.add('keys' if max(xkeys) > max(tkeys) else 'distance')
    assert len(fingerprints) == 42
    assert len(metrics['shapes']) >= 25 and len(metrics['templates']) >= 100
    assert len(metrics['frames']) >= 30 and len(metrics['regions']) >= 60
    assert metrics['counts'] == {1, 2, 3, 4}
    assert metrics['directions'] == set(DIRS) and metrics['keys'] == {1, 2, 3}
    assert metrics['objects'] == {1, 2, 3, 4} and metrics['repetitions'] >= {1, 2, 3}
    assert len(metrics['backgrounds']) == len(metrics['fills']) == 10
    assert shifts == {'keys', 'distance'}
    assert shared_colors and unequal_sides and same_color_disconnected
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(411, n, m)
            assert len(p['train']) == n and len(p['test']) == m
    for n, m in ((1, 1), (11, 1), (3, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _fingerprint(generate_puzzle(914, 2, 1)) == _fingerprint(generate_puzzle(914, 2, 1))
    # Negative label regression: unchanged exterior cells are substantive.
    p = generate_puzzle(13, 2, 1)
    broken = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {name: len(values) for name, values in metrics.items()} | {'seeds': 42, 'shifts': sorted(shifts)}
