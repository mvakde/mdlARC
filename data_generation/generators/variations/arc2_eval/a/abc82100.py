import numpy as np

_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))

# Rejecting an original input is sufficient to reject an exact original
# input/output pair.
_SOURCE_INPUTS = {
    (
        (5, 5),
        "12888"
        "00001"
        "00010"
        "00010"
        "00001",
    ),
    (
        (15, 15),
        "000000000000000"
        "060606060606060"
        "000000000000000"
        "020000000000040"
        "000000000000000"
        "020000000000040"
        "000080000000000"
        "020808000000040"
        "000070000000000"
        "080060080000000"
        "804204208000000"
        "080070080000020"
        "000060000000000"
        "070808070707000"
        "000080000000000",
    ),
    (
        (20, 20),
        "00000000000000000000"
        "00000000000000000000"
        "00040000000000040000"
        "00000000000000000000"
        "04000000080004000400"
        "00000000808000000000"
        "00040008000800040000"
        "00000080000080000000"
        "00000008000800000000"
        "00000000808000000000"
        "00000000080000000000"
        "00000000020000000000"
        "00000000040000000000"
        "00100000000000100000"
        "00000000000000700000"
        "10001000000080008000"
        "00000000000008080000"
        "00100000000000800000"
        "00000000000008080000"
        "00000000000080008000",
    ),
    (
        (8, 8),
        "11111200"
        "00000200"
        "00000200"
        "02180200"
        "00000200"
        "00800200"
        "00200200"
        "00100200",
    ),
    (
        (20, 20),
        "00000000000000000600"
        "20080006000000000006"
        "04180000001000000000"
        "00080000000005000000"
        "00000000000000000000"
        "00008000000000000400"
        "02480800000200000000"
        "00008000000000000000"
        "00000020000000000050"
        "00008000000004000000"
        "05788000050000000000"
        "00008000000000000000"
        "00000000000000002000"
        "00000000000000000000"
        "02000200000002000006"
        "00000000040000000000"
        "00500000000000000000"
        "00000000000004006000"
        "00000000000000000000"
        "50000000060000000000",
    ),
}


class _GenerationError(Exception):
    pass


class _ParseError(Exception):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rotate(
    offsets: tuple[tuple[int, int], ...],
    direction: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    if direction == (0, 1):
        return tuple(offsets)
    if direction == (1, 0):
        return tuple((c, -r) for r, c in offsets)
    if direction == (0, -1):
        return tuple((-r, -c) for r, c in offsets)
    if direction == (-1, 0):
        return tuple((-c, r) for r, c in offsets)
    raise ValueError("invalid direction")


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _input_fingerprint(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        "".join(str(int(x)) for x in grid.ravel()),
    )


def _components8(
    grid: np.ndarray,
    color: int,
) -> list[set[tuple[int, int]]]:
    remaining = {
        tuple(int(x) for x in position)
        for position in np.argwhere(grid == color)
    }
    components = []

    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = {start}

        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (r + dr, c + dc)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)

        components.append(component)

    components.sort(key=lambda component: min(component))
    return components


def _parse_instructions(grid, background, neutral):
    if background == neutral:
        raise _ParseError('neutral equals background')
    components = _components8(grid, neutral)
    if not components:
        raise _ParseError('no neutral mask')
    # Associate label rays with exact eight-connected components, including
    # diagonal outlines. A one-cell exterior registration hole is permitted.
    near = {}
    for index, component in enumerate(components):
        for r, c in component:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    near.setdefault((r + dr, c + dc), set()).add(index)
    candidates = [[] for _ in components]
    height, width = grid.shape
    label_cells = np.argwhere((grid != background) & (grid != neutral))
    for sr, sc in label_cells:
        sr, sc = int(sr), int(sc)
        source = int(grid[sr, sc])
        for dr, dc in _DIRECTIONS:
            tr, tc = sr + dr, sc + dc
            qr, qc = sr + 2 * dr, sc + 2 * dc
            if not (0 <= qr < height and 0 <= qc < width):
                continue
            target = int(grid[tr, tc])
            if target in (background, neutral, source) or int(grid[qr, qc]) not in (background, neutral):
                continue
            for index in sorted(near.get((qr, qc), ())):
                component = components[index]
                candidates[index].append({
                    'mask': frozenset(component), 'q': (qr, qc),
                    'direction': (dr, dc), 'source': source, 'target': target,
                    'source_label': (sr, sc), 'target_label': (tr, tc),
                    'offsets': tuple(sorted((r - qr, c - qc) for r, c in component))})
    if any(len(options) != 1 for options in candidates):
        raise _ParseError('mask has missing or ambiguous labels')
    records = [options[0] for options in candidates]
    labels = [p for rec in records for p in (rec['source_label'], rec['target_label'])]
    if len(labels) != len(set(labels)):
        raise _ParseError('instructions share labels')
    if len({rec['source'] for rec in records}) != len(records):
        raise _ParseError('source has multiple local instructions')
    return records


def _solve_grid(
    grid: np.ndarray,
    background: int,
    neutral: int,
) -> tuple[np.ndarray, list[dict], set[tuple[int, int]]]:
    instructions = _parse_instructions(grid, background, neutral)

    sources = [instruction["source"] for instruction in instructions]
    if len(sources) != len(set(sources)):
        raise _ParseError(
            "a source color has more than one local stencil"
        )

    labels = set()
    for instruction in instructions:
        labels.add(instruction["source_label"])
        labels.add(instruction["target_label"])

    anchors_by_instruction = []

    for instruction in instructions:
        anchors = [
            tuple(int(x) for x in position)
            for position in np.argwhere(
                grid == instruction["source"]
            )
            if tuple(int(x) for x in position) not in labels
        ]

        if not anchors:
            raise _ParseError(
                "an instruction has no external anchor"
            )

        anchors_by_instruction.append(anchors)

    output = grid.copy()

    # Erase all instruction masks and both cells of every label pair.
    output[output == neutral] = background
    for position in labels:
        output[position] = background

    # Erase every original source anchor before painting any stamp.
    for anchors in anchors_by_instruction:
        for position in anchors:
            output[position] = background

    height, width = grid.shape
    planned_paint = {}

    # All stamps are calculated from the original input. Generated target
    # cells are never reconsidered as source anchors.
    for instruction, anchors in zip(
        instructions,
        anchors_by_instruction,
    ):
        for anchor_row, anchor_column in anchors:
            for dr, dc in instruction["offsets"]:
                r = anchor_row + dr
                c = anchor_column + dc

                if not (0 <= r < height and 0 <= c < width):
                    continue

                previous = planned_paint.get((r, c))
                if (
                    previous is not None
                    and previous != instruction["target"]
                ):
                    raise _ParseError(
                        "incompatible stamp overlap"
                    )

                planned_paint[(r, c)] = instruction["target"]

    for position, color in planned_paint.items():
        output[position] = color

    return output, instructions, labels


def _background(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise _ParseError('background is not a unique mode')
    return int(counts.argmax())


def _signature(record):
    return (record['source'], record['target'], record['direction'], record['offsets'])


def _infer_neutral(train):
    # Only training pairs establish the shared neutral role. Neither test
    # labels nor hidden construction choices are consulted here.
    candidates = set(range(10))
    for pair in train:
        candidates &= set(int(c) for c in np.unique(pair['input']))
        candidates -= set(int(c) for c in np.unique(pair['output']))
    valid = []
    for color in sorted(candidates):
        try:
            if all(np.array_equal(_solve_grid(p['input'], _background(p['input']), color)[0],
                                  p['output']) for p in train):
                valid.append(color)
        except _ParseError:
            continue
    if len(valid) != 1:
        raise _ParseError('training does not identify a unique stencil color')
    return valid[0]


def predict(train_pairs, test_input):
    return _solve_grid(test_input, _background(test_input), _infer_neutral(train_pairs))[0]


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _shape(rng, family=None):
    family = family or _choose(rng, ('single', 'line', 'line', 'diamond', 'rectangle',
                                     'chevron', 'x', 'branch', 'irregular', 'irregular', 'c'))
    h, w = int(rng.integers(2, 8)), int(rng.integers(2, 8))
    if family == 'single':
        cells = {(0, 0)}
    elif family == 'line':
        cells = {(0, c) for c in range(int(rng.integers(2, 9)))}
    elif family == 'diamond':
        radius = int(rng.integers(1, 5))
        cells = {(r, c) for r in range(-radius, radius + 1)
                 for c in range(-radius, radius + 1) if abs(r) + abs(c) == radius}
    elif family == 'rectangle':
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
    elif family == 'chevron':
        radius = int(rng.integers(1, 5))
        cells = {(r, radius - abs(r)) for r in range(-radius, radius + 1)}
    elif family == 'x':
        radius = int(rng.integers(1, 5))
        cells = {(r, s * r) for r in range(-radius, radius + 1) for s in (-1, 1)}
    elif family == 'c':
        radius = int(rng.integers(2, 5))
        cells = {(r, w) for r in range(-radius, radius + 1)}
        cells |= {(r, c) for r in (-radius, radius) for c in range(-1, w + 1)}
    elif family == 'branch':
        cells = {(r, 0) for r in range(h)} | {(h // 2, c) for c in range(w)}
        cells |= {(r, w - 1) for r in range(h // 2, h)}
    else:
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        desired = int(rng.integers(3, max(4, h * w * 3 // 4)))
        for _ in range(desired * 12):
            if len(cells) >= desired:
                break
            r, c = _choose(rng, sorted(cells))
            dr, dc = _choose(rng, _DIRECTIONS if rng.random() < .65 else
                             ((1, 1), (1, -1), (-1, 1), (-1, -1)))
            if 0 <= r + dr < h and 0 <= c + dc < w:
                cells.add((r + dr, c + dc))
    return cells


def _registration_kind(offsets):
    cells = set(offsets)
    if (0, 0) in cells:
        return 'occupied'
    r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
    c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
    if not (r0 <= 0 <= r1 and c0 <= 0 <= c1):
        return 'adjacent'
    if r0 < 0 < r1 and c0 < 0 < c1:
        return 'interior_hole'
    return 'edge_hole'


def _registered_shape(rng, wanted=None, simple=False):
    for _ in range(60):
        family = ('c' if wanted == 'interior_hole' else 'chevron'
                  if wanted == 'edge_hole' else 'line' if wanted == 'offcenter' else None)
        if simple and wanted is None:
            family = _choose(rng, ('line', 'single', 'chevron'))
        cells = _shape(rng, family)
        near = {(r + dr, c + dc) for r, c in cells
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        choices = []
        for qr, qc in sorted(near):
            if (qr, qc - 1) in cells or (qr, qc - 2) in cells:
                continue
            # A reversed reading of the same two labels must not be
            # another valid registration next to this component.
            if (qr, qc - 3) in near and (qr, qc - 3) not in cells:
                continue
            offsets = tuple(sorted((r - qr, c - qc) for r, c in cells))
            kind = _registration_kind(offsets)
            offcenter = (sum(r for r, c in offsets) != 0 or sum(c for r, c in offsets) != 0)
            if wanted == 'offcenter' and (kind != 'occupied' or not offcenter or len(cells) < 3):
                continue
            if wanted not in (None, 'offcenter') and kind != wanted:
                continue
            choices.append(offsets)
        if choices:
            offsets = _choose(rng, choices)
            if rng.random() < .5:
                offsets = tuple((-r, c) for r, c in offsets)
            return tuple(sorted(offsets))
    # These registrations are unique even when the outside-box extension
    # is enabled. Fallbacks retain the requested teaching property.
    if wanted in ('interior_hole', 'edge_hole'):
        cells = {(-2, c) for c in range(-1, 2)} | {(2, c) for c in range(-1, 2)}
        cells |= {(r, 1) for r in range(-2, 3)}
        return tuple(sorted(cells))
    return ((0, 0), (0, 1), (0, 2))


def _mappings(rng, colors, count, forced=None):
    a = list(colors)
    if forced == 0:
        # The same source changes target in the next demonstration.
        pairs = [(a[0], a[1])]
        rest = a[2:]
    elif forced == 1:
        pairs = [(a[0], a[2]), (a[2], _choose(rng, (a[0], a[3]))) ]
        rest = [x for x in a if x not in (a[0], a[2])]
    else:
        a = [int(x) for x in rng.permutation(a)]
        mode = _choose(rng, ('independent', 'chain', 'cycle', 'shared', 'mixed'))
        if count == 1:
            return [(a[0], a[1])]
        if mode == 'cycle':
            return [(a[i], a[(i + 1) % count]) for i in range(count)]
        if mode == 'chain':
            return [(a[i], a[i + 1]) for i in range(count)]
        if mode == 'shared':
            return [(a[i], a[count]) for i in range(count)]
        if mode == 'independent':
            return [(a[i], a[count + i % (len(a) - count)]) for i in range(count)]
        return [(a[i], _choose(rng, [v for v in a if v != a[i]])) for i in range(count)]
    for source in rest[:max(0, count - len(pairs))]:
        pairs.append((source, _choose(rng, [v for v in a if v != source])))
    return pairs


def _place_instructions(rng, shape, bg, neutral, specifications, layout):
    grid = np.full(shape, bg, dtype=np.int8)
    records, occupied, neutral_halo = [], set(), set()
    n = len(specifications)
    margin = float(rng.uniform(.06, .32))
    middle = float(rng.uniform(.25, .75))
    for i, (source, target, direction, offsets) in enumerate(specifications):
        dr, dc = direction
        relative = set(offsets) | {(0, 0), (-dr, -dc), (-2 * dr, -2 * dc)}
        rlo, rhi = min(r for r, c in relative), max(r for r, c in relative)
        clo, chi = min(c for r, c in relative), max(c for r, c in relative)
        locations = [(r, c) for r in range(-rlo, shape[0] - rhi)
                     for c in range(-clo, shape[1] - chi)]
        if layout == 'row':
            aim = (middle * shape[0], (margin + (1 - 2 * margin) * i / max(1, n - 1)) * shape[1])
        elif layout == 'column':
            aim = ((margin + (1 - 2 * margin) * i / max(1, n - 1)) * shape[0], middle * shape[1])
        elif layout == 'group':
            aim = (middle * shape[0] + (i // 2 - .5) * 5,
                   middle * shape[1] + (i % 2 - .5) * 5)
        else:
            aim = (float(rng.uniform(0, shape[0])), float(rng.uniform(0, shape[1])))
        jitter = {q: float(rng.uniform(0, 3)) for q in locations}
        locations.sort(key=lambda q: abs(q[0] - aim[0]) + abs(q[1] - aim[1]) + jitter[q])
        for qr, qc in locations:
            mask = {(qr + r, qc + c) for r, c in offsets}
            sl, tl = (qr - 2 * dr, qc - 2 * dc), (qr - dr, qc - dc)
            cells = mask | {sl, tl, (qr, qc)}
            if cells & occupied or mask & neutral_halo:
                continue
            candidate = grid.copy()
            for p in mask:
                candidate[p] = neutral
            candidate[sl], candidate[tl] = source, target
            try:
                parsed = _parse_instructions(candidate, bg, neutral)
            except _ParseError:
                continue
            if len(parsed) != i + 1:
                continue
            record = next((x for x in parsed if x['q'] == (qr, qc) and x['source'] == source), None)
            if record is None or record['offsets'] != offsets:
                continue
            records.append(record)
            occupied |= cells
            neutral_halo |= {(r + a, c + b) for r, c in mask
                             for a in (-1, 0, 1) for b in (-1, 0, 1)}
            grid = candidate
            break
        else:
            raise _GenerationError('instructions do not fit with unique labels')
    return grid, records, occupied


def _footprint(record, anchor, shape):
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in record['offsets']
            if 0 <= ar + dr < shape[0] and 0 <= ac + dc < shape[1]}


def _construct_output(grid, bg, records, anchor_lists):
    # Construction uses the sampled placements; the separate input parser
    # must subsequently reconstruct and verify every one of these choices.
    output = grid.copy()
    for record, anchors in zip(records, anchor_lists):
        for p in record['mask'] | {record['source_label'], record['target_label']} | set(anchors):
            output[p] = bg
    paints = {}
    for record, anchors in zip(records, anchor_lists):
        for anchor in anchors:
            for p in _footprint(record, anchor, grid.shape):
                if p in paints and paints[p] != record['target']:
                    raise _GenerationError('conflicting simultaneous paint')
                paints[p] = record['target']
    for p, color in paints.items():
        output[p] = color
    return output


def _content(rng, grid, bg, neutral, records, occupied, colors, teaching=False, fallback=False):
    all_cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
    anchors, paint, lists = set(), {}, [[] for _ in records]
    policy = _choose(rng, ('interior', 'interior', 'one_edge', 'corner', 'free', 'free'))
    edge = int(rng.integers(4))
    signature = {_signature(x) + (x['q'],) for x in records}

    def unique():
        try:
            return {_signature(x) + (x['q'],) for x in _parse_instructions(grid, bg, neutral)} == signature
        except _ParseError:
            return False

    def place(i, p, complete=False):
        rec = records[i]
        if p in occupied or p in anchors or int(grid[p]) != bg:
            return False
        footprint = _footprint(rec, p, grid.shape)
        if not footprint or footprint & occupied:
            return False
        if (complete or policy == 'interior') and len(footprint) != len(rec['offsets']):
            return False
        if any(q in paint and paint[q] != rec['target'] for q in footprint):
            return False
        grid[p] = rec['source']
        if not unique():
            grid[p] = bg
            return False
        anchors.add(p)
        lists[i].append(p)
        for q in footprint:
            paint[q] = rec['target']
        return True

    for i, rec in enumerate(records):
        count = 2 if fallback else _choose(rng, (1, 2, 3, 4, 5, 6, 8, 10, 12))
        if teaching:
            count = max(2, count)
        style = _choose(rng, ('scattered', 'run', 'cluster', 'mixed'))
        cells = [all_cells[int(k)] for k in rng.permutation(len(all_cells))]
        # A complete copy in each teaching instruction makes geometry and
        # registration observable even if subsequent anchors clip.
        if teaching:
            for p in cells:
                if place(i, p, complete=True):
                    break
            else:
                raise _GenerationError('no complete teaching stamp')
        if policy in ('one_edge', 'corner'):
            def priority(p):
                r, c = p
                distances = (r, grid.shape[1] - 1 - c, grid.shape[0] - 1 - r, c)
                return distances[edge] + (distances[(edge + 1) % 4] if policy == 'corner' else 0)
            cells.sort(key=priority)
        step = _choose(rng, _DIRECTIONS)
        for _ in range(count):
            if len(lists[i]) >= count:
                break
            preferred = []
            if lists[i] and style != 'scattered' and (style != 'mixed' or len(lists[i]) % 2):
                if style == 'run':
                    r, c = lists[i][-1]
                    preferred = [(r + step[0], c + step[1])]
                else:
                    preferred = [(r + dr, c + dc) for r, c in lists[i]
                                 for dr, dc in _DIRECTIONS]
            # Shared-target sources can stamp through each other's anchors.
            if i and records[i - 1]['target'] == rec['target']:
                preferred += [(r + dr, c + dc) for r, c in lists[i - 1]
                              for dr, dc in rec['offsets'] if (dr, dc) != (0, 0)]
            candidates = preferred + cells
            found = False
            for p in candidates:
                if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1] and place(i, p):
                    found = True
                    break
            if not found:
                break
        if not lists[i] or teaching and len(lists[i]) < 2:
            raise _GenerationError('no usable anchors')

    sources = {x['source'] for x in records}
    inactive = [x for x in colors if x not in sources]
    rng.shuffle(inactive)
    n_inert = 0 if fallback else int(rng.integers(min(4, len(inactive)) + 1))
    for color in inactive[:n_inert]:
        count = int(rng.integers(1, 10))
        cells = [all_cells[int(k)] for k in rng.permutation(len(all_cells))]
        last = None
        for _ in range(count):
            nearby = [] if last is None else [(last[0] + dr, last[1] + dc) for dr, dc in _DIRECTIONS]
            for p in (nearby if rng.random() < .6 else []) + cells:
                if not (0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]):
                    continue
                if p in occupied or p in paint or int(grid[p]) != bg:
                    continue
                grid[p] = color
                if unique():
                    last = p
                    break
                grid[p] = bg
    return _construct_output(grid, bg, records, lists)


def _make_pair(rng, neutral, backgrounds, colors, regime, index, familiar):
    forced = index if regime == 'source_train' and index < 2 else None
    count = _choose(rng, (1, 1, 2, 2, 2, 3, 3, 4))
    if forced == 1:
        count = max(2, count)
    mapping = _mappings(rng, colors, count, forced)
    uniform = rng.random() < .4
    common_direction = _choose(rng, _DIRECTIONS)
    specs = []
    for i, (source, target) in enumerate(mapping):
        wanted = ('offcenter' if forced == 0 and i == 0 else
                  'interior_hole' if forced == 1 and i == 0 else
                  _choose(rng, ('occupied', 'occupied', 'occupied', 'edge_hole', 'interior_hole', 'adjacent')))
        direction = common_direction if uniform else _choose(rng, _DIRECTIONS)
        if forced is None and familiar and rng.random() < .3:
            old = _choose(rng, familiar)
            offsets = old['offsets']
            direction = old['direction']
        else:
            offsets = tuple(sorted(_rotate(_registered_shape(rng, wanted), direction)))
        specs.append((source, target, direction, offsets))
    layout = _choose(rng, ('row', 'column', 'group', 'scattered', 'scattered'))
    bg = _choose(rng, backgrounds)
    for attempt in range(45):
        fallback = attempt >= 25
        if fallback:
            specs = [(s, t, (0, 1), _registered_shape(rng,
                     'interior_hole' if forced == 1 and i == 0 else 'offcenter', simple=True))
                     for i, (s, t) in enumerate(mapping)]
        if attempt >= 8:
            shape = (30, 30)
        else:
            minimum = 5 if count == 1 else 10 if count == 2 else 14
            maximum = 13 if count == 1 and rng.random() < .4 else 30
            shape = (int(rng.integers(minimum, maximum + 1)), int(rng.integers(minimum, maximum + 1)))
        try:
            grid, records, occupied = _place_instructions(rng, shape, bg, neutral, specs, layout)
            output = _content(rng, grid, bg, neutral, records, occupied, colors,
                              teaching=forced is not None, fallback=fallback)
            if _background(grid) != bg or _background(output) != bg:
                continue
            solved, parsed, labels = _solve_grid(grid, bg, neutral)
            if not np.array_equal(solved, output):
                raise _GenerationError('construction and visible interpretation differ')
            if _input_fingerprint(grid) in _SOURCE_INPUTS:
                continue
            return {'input': grid, 'output': output}
        except (_GenerationError, _ParseError):
            continue
    raise ValueError('bounded placement exhausted')


def _analyze_puzzle(puzzle):
    try:
        neutral = _infer_neutral(puzzle['train'])
        groups = []
        for pair in puzzle['train'] + puzzle['test']:
            bg = _background(pair['input'])
            output, records, labels = _solve_grid(pair['input'], bg, neutral)
            _require(np.array_equal(output, pair['output']), 'incorrect registered-stencil output')
            groups.append((records, labels))
        training = [r for records, labels in groups[:len(puzzle['train'])] for r in records]
        testing = [r for records, labels in groups[len(puzzle['train']):] for r in records]
        _require(any((0, 0) in r['offsets'] and
                     (sum(x for x, y in r['offsets']) != 0 or sum(y for x, y in r['offsets']) != 0)
                     for r in training), 'training lacks off-center registration evidence')
        if any((0, 0) not in r['offsets'] for r in testing):
            _require(any((0, 0) not in r['offsets'] for r in training), 'training lacks hole registration')
        return {'neutral': neutral, 'parsed_groups': groups}
    except _ParseError as error:
        raise AssertionError(str(error)) from error


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'invalid counts')
    seen = set()
    for pair in puzzle['train'] + puzzle['test']:
        _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
        for grid in pair.values():
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'invalid array')
            _require(all(1 <= n <= 30 for n in grid.shape), 'invalid dimensions')
            _require(np.issubdtype(grid.dtype, np.integer) and bool(np.all((0 <= grid) & (grid <= 9))),
                     'invalid colors or dtype')
        _require(pair['input'].shape == pair['output'].shape, 'dimensions changed')
        key = _grid_key(pair['input'])
        _require(key not in seen, 'duplicate input')
        seen.add(key)
        _require(_input_fingerprint(pair['input']) not in _SOURCE_INPUTS, 'copied source input')
        _require(not np.array_equal(pair['input'], pair['output']), 'unchanged example')
    _analyze_puzzle(puzzle)


def _validated_count(value, minimum, maximum, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(name + ' out of range')
    return value


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train = _validated_count(num_train, 2, 10, 'num_train')
    num_test = _validated_count(num_test, 1, 4, 'num_test')
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed + seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    palette = [int(c) for c in rng.permutation(10)]
    neutral = palette[0]
    backgrounds = palette[1:3] if rng.random() < .55 else palette[1:2]
    # Reserve two potential backgrounds even in the constant-background
    # regime so that the visible mapping palette stays coherent.
    colors = palette[3:]
    puzzle = {'train': [], 'test': []}
    seen, familiar = set(), []
    for split, regime, count in (('train', 'source_train', num_train), ('test', 'target_test', num_test)):
        for index in range(count):
            for _ in range(10):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, neutral, backgrounds, colors, regime, index,
                                  familiar if split == 'test' else [])
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train':
                        familiar.extend(_parse_instructions(pair['input'], _background(pair['input']), neutral))
                    break
            else:
                raise ValueError('could not obtain distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


# Original training regressions: changing local mappings, hole-registered
# chevrons, large masks covering anchors, and adjacent single-cell runs.
_SOURCE_REGRESSIONS = (
    (('12888', '00001', '00010', '00010', '00001'),
     ('00000', '00002', '00022', '00022', '00002')),
    (('000000000000000', '060606060606060', '000000000000000', '020000000000040', '000000000000000', '020000000000040', '000080000000000', '020808000000040', '000070000000000', '080060080000000', '804204208000000', '080070080000020', '000060000000000', '070808070707000', '000080000000000'),
     ('070707070707070', '707070707070707', '040000000000020', '400000000000002', '040000000000020', '400000000000002', '040000000000020', '400000000000002', '040000000000020', '000000000000000', '000000000000040', '000000000000400', '000000000000040', '606000606060600', '060000060606000')),
    (('00000000000000000000', '00000000000000000000', '00040000000000040000', '00000000000000000000', '04000000080004000400', '00000000808000000000', '00040008000800040000', '00000080000080000000', '00000008000800000000', '00000000808000000000', '00000000080000000000', '00000000020000000000', '00000000040000000000', '00100000000000100000', '00000000000000700000', '10001000000080008000', '00000000000008080000', '00100000000000800000', '00000000000008080000', '00000000000080008000'),
     ('02020200000202020202', '00202000002000202000', '02020200000202020202', '20200020000020202020', '02000200000002000200', '00202000000000202000', '00020000000000020000', '00000000000000000000', '00000000000000000000', '00000000000000000000', '00000000000000000000', '00000000000000000000', '00000000000000000000', '70007000000000000000', '07070000000000000000', '00700070000000000000', '07070700000000000000', '70007000000000000000', '07070700000000000000', '00700070000000000000')),
    (('11111200', '00000200', '00000200', '02180200', '00000200', '00800200', '00200200', '00100200'),
     ('22222100', '00000100', '00000100', '00000100', '00000100', '00000100', '00000100', '00000100')),
)


def _geometry_key(offsets):
    variants = []
    for direction in _DIRECTIONS:
        for sign in (-1, 1):
            cells = _rotate(tuple((sign * r, c) for r, c in offsets), direction)
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            variants.append(tuple(sorted((r - r0, c - c0) for r, c in cells)))
    return min(variants)


def _graph_kind(records):
    mapping = {r['source']: r['target'] for r in records}
    sources, targets = set(mapping), set(mapping.values())
    if len(mapping) == 1:
        return 'single'
    if len(targets) == 1:
        return 'shared_target'
    if sources == targets:
        return 'reciprocal' if len(mapping) == 2 else 'cycle'
    if not sources & targets:
        return 'independent'
    if len(sources & targets) == len(sources) - 1 and len(targets) == len(sources):
        return 'open_chain'
    return 'mixed'


def pair_features(pair, neutral):
    grid = pair['input']
    bg = _background(grid)
    _, records, labels = _solve_grid(grid, bg, neutral)
    source_set = {rec['source'] for rec in records}
    targets = {rec['target'] for rec in records}
    anchor_lists = [[tuple(int(x) for x in p) for p in np.argwhere(grid == rec['source'])
                     if tuple(p) not in labels] for rec in records]
    anchors = set().union(*map(set, anchor_lists))
    visits = {}
    covered_anchors, clipped = set(), set()
    for i, (rec, positions) in enumerate(zip(records, anchor_lists)):
        for j, anchor in enumerate(positions):
            footprint = _footprint(rec, anchor, grid.shape)
            covered_anchors |= (footprint & anchors) - {anchor}
            for p in footprint:
                visits.setdefault(p, []).append((i, j))
            for dr, dc in rec['offsets']:
                r, c = anchor[0] + dr, anchor[1] + dc
                if r < 0: clipped.add('top')
                if r >= grid.shape[0]: clipped.add('bottom')
                if c < 0: clipped.add('left')
                if c >= grid.shape[1]: clipped.add('right')
    active_grid = np.zeros(grid.shape, dtype=np.int8)
    sizes, runs = [], []
    for i, positions in enumerate(anchor_lists):
        for p in positions:
            active_grid[p] = i + 1
        sizes.extend(len(x) for x in _components8(active_grid, i + 1))
        cells = set(positions)
        for r, c in positions:
            for dr, dc in ((1, 0), (0, 1)):
                length = 1
                while (r + length * dr, c + length * dc) in cells:
                    length += 1
                runs.append(length)
    inert = {tuple(int(x) for x in p) for p in np.argwhere((grid != bg) & (grid != neutral))
             if tuple(p) not in labels and int(grid[tuple(p)]) not in source_set}
    kinds = [_registration_kind(rec['offsets']) for rec in records]
    widths = [max(c for r, c in rec['offsets']) - min(c for r, c in rec['offsets']) + 1 for rec in records]
    heights = [max(r for r, c in rec['offsets']) - min(r for r, c in rec['offsets']) + 1 for rec in records]
    instruction_cells = [rec['mask'] | {rec['source_label'], rec['target_label']} for rec in records]
    separation = [max(abs(r - a), abs(c - b)) for i, cells in enumerate(instruction_cells)
                  for other in instruction_cells[i + 1:] for r, c in cells for a, b in other]
    masks = set().union(*(rec['mask'] for rec in records))
    return {
        'height': grid.shape[0], 'width': grid.shape[1], 'background': bg,
        'palette': len(np.unique(grid)), 'instruction_count': len(records),
        'graph': _graph_kind(records), 'directions': tuple(sorted(set(rec['direction'] for rec in records))),
        'mixed_directions': len(set(rec['direction'] for rec in records)) > 1,
        'registrations': tuple(kinds), 'hole_count': sum(k != 'occupied' for k in kinds),
        'mask_sizes': tuple(len(rec['offsets']) for rec in records),
        'mask_widths': tuple(widths), 'mask_heights': tuple(heights),
        'mask_density': tuple(round(len(rec['offsets']) / (w * h), 3)
                              for rec, w, h in zip(records, widths, heights)),
        'anchor_counts': tuple(map(len, anchor_lists)), 'anchor_group_max': max(sizes),
        'anchor_run_max': max(runs),
        'covered_anchors': len(covered_anchors),
        'overlap_cells': sum(len(v) > 1 for v in visits.values()),
        'cross_instruction_overlap': sum(len({i for i, j in v}) > 1 for v in visits.values()),
        'clipped_edges': tuple(sorted(clipped)),
        'inert_cells': len(inert), 'inert_colors': len({int(grid[p]) for p in inert}),
        'target_only_cells': sum(int(grid[p]) in targets - source_set for p in inert),
        'inert_preserved': all(int(pair['output'][p]) == int(grid[p]) for p in inert),
        'instruction_distance': min(separation) if separation else None,
        'anchor_mask_distance': min(max(abs(r - a), abs(c - b)) for r, c in anchors for a, b in masks),
        'registration_rows': tuple(rec['q'][0] for rec in records),
        'registration_columns': tuple(rec['q'][1] for rec in records),
    }


def _puzzle_key(puzzle):
    return tuple((_grid_key(pair['input']), _grid_key(pair['output']))
                 for split in ('train', 'test') for pair in puzzle[split])


def self_test() -> None:
    # All source training examples, including four opposing hole stencils,
    # overlapping large diamonds and runs with reciprocal single-cell rules.
    for x, y in _SOURCE_REGRESSIONS:
        grid = np.array([[int(c) for c in row] for row in x], dtype=np.int8)
        expected = np.array([[int(c) for c in row] for row in y], dtype=np.int8)
        assert np.array_equal(_solve_grid(grid, 0, 8)[0], expected)

    # Manually specified exterior-hole regression. Two adjacent anchors
    # produce overlapping vertical lines one column to their right;
    # the isolated target-colored pixel remains inert.
    grid = np.zeros((7, 10), dtype=np.int8)
    grid[2:5, 3] = 8
    grid[3, 0:2] = (1, 2)
    grid[1:3, 7] = 1
    grid[6, 0], grid[6, 2] = 2, 3
    expected = np.zeros_like(grid)
    expected[0:4, 8] = 2
    expected[6, 0], expected[6, 2] = 2, 3
    solved, records, _ = _solve_grid(grid, 0, 8)
    assert np.array_equal(solved, expected)
    assert records[0]['q'] == (3, 2)
    assert _registration_kind(records[0]['offsets']) == 'adjacent'

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    all_features, train_features, test_features = [], [], []
    geometries, signatures, layout_keys, roles = set(), set(), set(), set()
    changed, new_tests, changing_bg, preserved_past_source = 0, 0, 0, 0
    combined = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        neutral = _infer_neutral(puzzle['train'])
        roles.add(neutral)
        train_signatures, seen_mapping, bgs, past_sources = set(), {}, set(), set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                # Prediction API sees training and this input, never the
                # test output, seed, or generation records.
                assert np.array_equal(predict(puzzle['train'], pair['input']), pair['output'])
                feature = pair_features(pair, neutral)
                all_features.append(feature)
                (train_features if split == 'train' else test_features).append(feature)
                assert feature['inert_preserved']
                bgs.add(feature['background'])
                records = _parse_instructions(pair['input'], feature['background'], neutral)
                local = {r['source'] for r in records}
                labels = {p for r in records for p in (r['source_label'], r['target_label'])}
                for color in past_sources - local - {feature['background'], neutral}:
                    positions = [tuple(p) for p in np.argwhere(pair['input'] == color) if tuple(p) not in labels]
                    if positions and all(pair['output'][p] == color for p in positions):
                        preserved_past_source += 1
                past_sources |= local
                current = {_signature(r) for r in records}
                if split == 'train':
                    train_signatures |= current
                else:
                    new_tests += bool(current - train_signatures)
                for rec in records:
                    seen_mapping.setdefault(rec['source'], set()).add(rec['target'])
                    geometries.add(_geometry_key(rec['offsets']))
                    signatures.add(_signature(rec))
                layout_keys.add(tuple((r['q'], r['direction']) for r in records))
                combined += bool(feature['hole_count'] and feature['mixed_directions'] and
                                 feature['overlap_cells'] and feature['graph'] != 'single')
            if split == 'train':
                assert any(len(v) > 1 for v in seen_mapping.values())
        changed += any(len(v) > 1 for v in seen_mapping.values())
        changing_bg += len(bgs) > 1
    assert changed == 42 and changing_bg >= 10
    assert new_tests >= 30 and preserved_past_source >= 20
    assert len(geometries) >= 60 and len(signatures) >= 100 and len(layout_keys) >= 100
    assert len(roles) >= 8 and combined >= 10
    assert len({(f['height'], f['width']) for f in all_features}) >= 70
    assert len({f['palette'] for f in all_features}) >= 5
    assert len({v for f in all_features for v in f['mask_density']}) >= 20
    assert any(f['instruction_distance'] is not None and f['instruction_distance'] <= 2
               for f in all_features)
    assert min(min(f['height'], f['width']) for f in all_features) <= 8
    assert max(max(f['height'], f['width']) for f in all_features) == 30
    assert {'single', 'independent', 'reciprocal', 'open_chain', 'cycle', 'shared_target'} <= {f['graph'] for f in all_features}
    for features in (train_features, test_features):
        assert {1, 2, 3, 4} <= {f['instruction_count'] for f in features}
        assert {'occupied', 'edge_hole', 'interior_hole', 'adjacent'} <= {v for f in features for v in f['registrations']}
        assert set(_DIRECTIONS) <= {v for f in features for v in f['directions']}
        assert max(max(f['mask_sizes']) for f in features) >= 12
        assert max(max(f['mask_widths']) for f in features) >= 7
        assert any(f['covered_anchors'] > 0 for f in features)
        assert any(f['cross_instruction_overlap'] > 0 for f in features)
        assert max(f['anchor_group_max'] for f in features) >= 5
        assert max(f['anchor_run_max'] for f in features) >= 5
        assert min(min(f['anchor_counts']) for f in features) == 1
        assert max(max(f['anchor_counts']) for f in features) >= 10
        assert any(not f['clipped_edges'] for f in features)
        assert any(len(f['clipped_edges']) == 1 for f in features)
        assert any(len(f['clipped_edges']) >= 3 for f in features)
        assert min(f['inert_colors'] for f in features) == 0
        assert max(f['inert_colors'] for f in features) >= 3
        assert any(f['target_only_cells'] for f in features)
        assert min(f['anchor_mask_distance'] for f in features) == 1
        assert any(f['mixed_directions'] for f in features)
        assert any(f['instruction_count'] > 1 and not f['mixed_directions'] for f in features)
        assert any(f['instruction_count'] > 1 and len(set(f['registration_rows'])) == 1
                   for f in features)
        assert any(f['instruction_count'] > 1 and len(set(f['registration_columns'])) == 1
                   for f in features)
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (1011, None, 1), (1012, None, 4), (1013, 2, None), (1014, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None: assert len(puzzle['train']) == train
        if test is not None: assert len(puzzle['test']) == test
    assert _puzzle_key(generate_puzzle(777, 7, 3)) == _puzzle_key(generate_puzzle(777, 7, 3))
    assert _puzzle_key(generate_puzzle(777, 7, 3)) != _puzzle_key(generate_puzzle(778, 7, 3))
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
                   {'num_train': 2, 'num_test': 2}, {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(123, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    broken = generate_puzzle(901, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
