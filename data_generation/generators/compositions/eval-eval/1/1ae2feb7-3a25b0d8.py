"""Chamber echoes: transport pulse programs, then flood their target chambers.

No parent code is imported. Geometry and every program are recovered from the
input. A room's clock runs on shortest-path distance, never on donor pixel size.
"""
from collections import deque
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    steps = _STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _STEPS
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, points = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    points.append(p)
        parts.append(np.array(points, dtype=int))
    return parts


def _bbox(points):
    r, c = points.min(0)
    R, C = points.max(0) + 1
    return int(r), int(c), int(R), int(C)


def _enclosed(barrier):
    padded = np.pad(~barrier, 1, constant_values=True)
    return [p - 1 for p in _components(padded)
            if np.all(p > 0) and np.all(p < np.array(padded.shape) - 1)]


def _relations(rooms):
    boxes = [_bbox(room['points']) for room in rooms]
    return [[tuple(-1 if a[k + 2] <= b[k] else 1 if b[k + 2] <= a[k]
                   else 0 for k in (0, 1)) for b in boxes] for a in boxes]


def _match(donor, target):
    """Match the complete directional relation graph and doorway normals.

    Scan order, area, absolute coordinates and colors are not matching keys.
    """
    n = len(donor)
    assert n == len(target)
    a, b = _relations(donor), _relations(target)
    options = {i: [j for j in range(n) if donor[i]['normal'] == target[j]['normal']]
               for i in range(n)}
    mappings = []

    def search(mapping, used):
        if len(mapping) == n:
            mappings.append(dict(mapping))
            return
        choices = {i: [j for j in options[i] if j not in used and all(
            a[i][k] == b[j][v] for k, v in mapping.items())]
                   for i in range(n) if i not in mapping}
        i = min(choices, key=lambda k: len(choices[k]))
        for j in choices[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({}, set())
    assert mappings, "no matching compartment arrangement"
    return mappings


def _parse_with_port(inp, bg, wall, port):
    barrier = (inp == wall) | (inp == port)
    parts = _components(barrier, diagonal=True)
    assert len(parts) == 2
    objects, covered = [], barrier.copy()
    for part in parts:
        r, c, R, C = _bbox(part)
        crop = np.full((R - r, C - c), bg, dtype=np.int8)
        crop[part[:, 0] - r, part[:, 1] - c] = inp[tuple(part.T)]
        local_barrier = (crop == wall) | (crop == port)
        rooms = []
        ports = set(map(tuple, np.argwhere(crop == port)))
        used_ports = set()
        for points in _enclosed(local_barrier):
            cells = set(map(tuple, points))
            gates = [(g, (g[0] + dr, g[1] + dc), (dr, dc))
                     for g in ports for dr, dc in _STEPS
                     if (g[0] + dr, g[1] + dc) in cells]
            assert len(gates) == 1, "one doorway adjoining one interior cell per room"
            gate, origin, normal = gates[0]
            assert gate not in used_ports
            used_ports.add(gate)
            crop[tuple(points.T)] = inp[points[:, 0] + r, points[:, 1] + c]
            covered[points[:, 0] + r, points[:, 1] + c] = True
            ray, p = [], origin
            while p in cells:
                ray.append(p)
                p = p[0] + normal[0], p[1] + normal[1]
            runs, pos = [], 0
            while pos < len(ray):
                color = int(crop[ray[pos]])
                if color == bg:
                    pos += 1
                    continue
                start = pos
                while pos < len(ray) and crop[ray[pos]] == color:
                    pos += 1
                runs.append((color, pos - start, start + 1))
            colored = {p for p in cells if crop[p] != bg}
            assert colored <= set(ray), "instructions must lie on the doorway ray"
            assert all(crop[p] not in (wall, port) for p in cells)
            if runs:
                assert crop[origin] == bg, "at least one source-side gap"
                assert 1 <= len(runs) <= 3 and all(1 <= x[1] <= 5 for x in runs)
            rooms.append({'points': points, 'gate': gate, 'origin': origin,
                          'normal': normal, 'runs': runs})
        assert 2 <= len(rooms) <= 6 and used_ports == ports
        has_runs = [bool(room['runs']) for room in rooms]
        assert all(has_runs) or not any(has_runs), "a whole donor or a whole blank target"
        objects.append({'crop': crop, 'rooms': rooms, 'bbox': (r, c, R, C),
                        'donor': all(has_runs)})
    assert np.all(inp[~covered] == bg), "foreground outside the two frameworks"
    donors, targets = [o for o in objects if o['donor']], [o for o in objects if not o['donor']]
    assert len(donors) == len(targets) == 1
    donor, target = donors[0], targets[0]
    mappings = _match(donor['rooms'], target['rooms'])
    return {'bg': bg, 'wall': wall, 'port': port, 'donor': donor,
            'target': target, 'mappings': mappings}


def _analyze(inp):
    border = np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    assert np.all(border == border[0]), "uniform empty canvas border"
    bg = int(border[0])
    counts = np.bincount(inp.ravel(), minlength=10)
    counts[bg] = 0
    assert np.count_nonzero(counts == counts.max()) == 1 and counts.max() > 0
    wall = int(counts.argmax())  # Family constraint: framework is dominant foreground.
    candidates = []
    for port in np.unique(inp):
        port = int(port)
        if port in (bg, wall):
            continue
        try:
            candidates.append(_parse_with_port(inp, bg, wall, port))
        except AssertionError:
            pass
    assert len(candidates) == 1, "doorway role must have a unique geometric interpretation"
    return candidates[0]


def _distances(room):
    cells = set(map(tuple, room['points']))
    distances = {room['origin']: 1}
    queue = deque([room['origin']])
    while queue:
        p = queue.popleft()
        for dr, dc in _STEPS:
            q = p[0] + dr, p[1] + dc
            if q in cells and q not in distances:
                distances[q] = distances[p] + 1
                queue.append(q)
    assert len(distances) == len(cells)
    return distances


def _answer(info, variant='correct'):
    donor, target = info['donor']['rooms'], info['target']['rooms']
    mappings = info['mappings']
    if variant == 'scan':
        mappings = [dict(enumerate(range(len(donor))))]
    elif variant == 'area':
        mappings = [dict(zip(sorted(range(len(donor)), key=lambda i: len(donor[i]['points'])),
                             sorted(range(len(target)), key=lambda i: len(target[i]['points']))))]
    answers = []
    for mapping in mappings:
        out = info['target']['crop'].copy()
        for i, j in mapping.items():
            source, room = donor[i], target[j]
            runs = source['runs']
            for p, distance in _distances(room).items():
                if variant == 'manhattan':
                    distance = sum(abs(p[k] - room['gate'][k]) for k in (0, 1))
                elif variant == 'stripe':
                    distance = sum((p[k] - room['gate'][k]) * room['normal'][k] for k in (0, 1))
                if variant == 'uniform':
                    firing = [runs[0]]
                else:
                    firing = [run for run in runs if (
                        distance - (runs[0][2] if variant == 'gap_phase' else 1)) %
                        max(1, run[1] + (1 if variant == 'period_plus_one' else 0)) == 0]
                if firing:
                    if variant == 'short':
                        winner = min(firing, key=lambda x: (x[1], x[2]))
                    elif variant == 'long':
                        winner = min(firing, key=lambda x: (-x[1], x[2]))
                    elif variant == 'far':
                        winner = firing[-1]
                    else:
                        winner = firing[0]
                    out[p] = winner[0]
        if answers:
            assert np.array_equal(out, answers[0]), "ambiguous transported programs"
        else:
            answers.append(out)
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover both figures, match chambers, and evaluate local geodesic clocks."""
    return _answer(_analyze(_grid(input_grid)))


def _make_plan(rng, count, two_columns=False):
    cols = 2 if two_columns or count > 3 or rng.random() < .65 else 1
    rows = (count + cols - 1) // cols
    mask = np.ones((rows, cols), dtype=bool)
    if mask.sum() > count:
        corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
        mask[corners[int(rng.integers(len(corners)))]] = False
    return mask


def _geometry(plan, widths, heights, insets):
    ys = np.concatenate(([0], np.cumsum(np.array(heights) + 1)))
    xs = np.concatenate(([0], np.cumsum(np.array(widths) + 1)))
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    boxes = []
    for index, (r, c) in enumerate(np.argwhere(plan)):
        y, x, Y, X = int(ys[r]), int(xs[c]), int(ys[r + 1]), int(xs[c + 1])
        labels[y:Y + 1, x:X + 1] = -2
        top, bottom, left, right = insets[index]
        box = y + 1 + top, x + 1 + left, Y - bottom, X - right
        a, b, A, B = box
        labels[a:A, b:B] = index
        boxes.append(box)
    return labels, boxes


def _program(rng, payload, single=None):
    if single is not None:
        return [(int(rng.choice(payload)), single)]
    count = int(rng.choice([1, 2, 3], p=[.35, .45, .20]))
    lengths = list(map(int, rng.integers(1, 6, size=count)))
    while sum(lengths) > 8:
        i = next(i for i, v in enumerate(lengths) if v > 1)
        lengths[i] -= 1
    return [(int(rng.choice(payload)), length) for length in lengths]


def _paint(labels, bg, wall, port, gates, programs=None, gaps=None):
    out = np.full(labels.shape, bg, dtype=np.int8)
    out[labels == -2] = wall
    for index, (gate, normal) in enumerate(gates):
        out[gate] = port
        if programs is None:
            continue
        distance = 1
        for (color, length), gap in zip(programs[index], gaps[index]):
            distance += gap
            for d in range(distance, distance + length):
                p = gate[0] + normal[0] * d, gate[1] + normal[1] * d
                assert labels[p] == index
                out[p] = color
            distance += length
    return out


def _constructed_answer(labels, target, gates, programs):
    """Construction uses label masks and synchronous shell expansion.

    It does not extract objects, infer roles, match graphs, or call solve.
    """
    out = target.copy()
    for i, (gate, normal) in enumerate(gates):
        available = set(map(tuple, np.argwhere(labels == i)))
        shell = {(gate[0] + normal[0], gate[1] + normal[1])}
        distance = 1
        while shell:
            hits = [(color, length) for color, length in programs[i] if (distance - 1) % length == 0]
            if hits:
                for p in shell:
                    out[p] = hits[0][0]
            available -= shell
            shell = {(p[0] + dr, p[1] + dc) for p in shell for dr, dc in _STEPS
                     if (p[0] + dr, p[1] + dc) in available}
            distance += 1
        assert not available, "connected target chamber"
    return out


def _embed(rng, donor, target, bg):
    dh, dw = donor.shape
    th, tw = target.shape
    layouts = []
    if dh + th + 3 <= 30:
        layouts.append('stack')
    if dw + tw + 3 <= 30:
        layouts.append('side')
    assert layouts, "two frameworks do not fit"
    layout = str(rng.choice(layouts))
    first, second = (donor, target) if rng.random() < .5 else (target, donor)
    if layout == 'stack':
        gap = int(rng.integers(1, min(3, 28 - dh - th) + 1))
        h, w = dh + th + gap + 2, max(dw, tw) + 2
        positions = [(1, int(rng.integers(1, w - first.shape[1]))),
                     (first.shape[0] + gap + 1, int(rng.integers(1, w - second.shape[1])))]
    else:
        gap = int(rng.integers(1, min(3, 28 - dw - tw) + 1))
        h, w = max(dh, th) + 2, dw + tw + gap + 2
        positions = [(int(rng.integers(1, h - first.shape[0])), 1),
                     (int(rng.integers(1, h - second.shape[0])), first.shape[1] + gap + 1)]
    extra_h, extra_w = int(rng.integers(min(3, 30 - h) + 1)), int(rng.integers(min(3, 30 - w) + 1))
    shift = int(rng.integers(extra_h + 1)), int(rng.integers(extra_w + 1))
    canvas = np.full((h + extra_h, w + extra_w), bg, dtype=np.int8)
    for a, (r, c) in zip((first, second), positions):
        r, c = r + shift[0], c + shift[1]
        canvas[r:r + a.shape[0], c:c + a.shape[1]] = a
    return canvas


def _pair(rng, count, witness, colors, rotation, shift, fallback=False):
    # Each actual room has an exposed left/right edge; a connected coarse
    # polyomino supplies a single compartment framework, not separate cards.
    plan = _make_plan(rng, count, two_columns=witness == 0)
    cells = list(map(tuple, np.argwhere(plan)))
    a, b = colors
    roles = list(map(int, rng.choice([c for c in range(10) if c not in colors], 3, replace=False)))
    bg, wall, port = roles
    payload = [c for c in range(10) if c not in roles]
    programs = [_program(rng, payload) for _ in cells]
    if witness is not None:
        programs[0] = [(a, 2), (b, 3)] if witness == 0 else [(b, 3), (a, 2)]
        programs[1] = [(int(rng.choice(payload)), 3 if witness == 0 else 2)]
    gaps = [[int(rng.integers(1, 3))] + [int(rng.integers(0, 2)) for _ in p[1:]] for p in programs]
    for i, p in enumerate(programs):
        for j in range(1, len(p)):
            if p[j][0] == p[j - 1][0]:
                gaps[i][j] = 1
        # Cap the donor width without truncating any encoded run.
        if sum(length for _, length in p) + sum(gaps[i]) > 11:
            gaps[i][0] = 1
    face = (0, 3, 1, 2)[rotation]
    widths = [max(sum(length for _, length in programs[i]) + sum(gaps[i]) + (face >= 2)
                  for i, cell in enumerate(cells) if cell[1] == col)
              for col in range(plan.shape[1])]
    widths = [min(12, w + int(rng.integers(0, 2))) for w in widths]
    heights = [2] * plan.shape[0]
    insets = [[0, 0, 0, 0] for _ in cells]
    for inset in insets:
        inset[face] = int(rng.integers(2))
    target_widths = list(map(int, rng.integers(6, 9, size=plan.shape[1])))
    if shift == 'range':
        target_widths = list(map(int, rng.integers(10, 13, size=plan.shape[1])))
    target_heights = list(map(int, rng.integers(3, 5, size=plan.shape[0])))
    target_insets = [[0, 0, 0, 0] for _ in cells]
    for inset in target_insets:
        inset[face] = int(rng.integers(2))
    if witness == 0:
        # Change the visible room discovery order while preserving left/right
        # and overlap relationships. Matching by scan rank is then wrong.
        axis = 0 if rotation % 2 == 0 else 1
        first, second = next((i, j) for i in range(count) for j in range(i + 1, count)
                             if cells[i][axis] == cells[j][axis])
        insets[first][face], insets[second][face] = 1, 0
        target_insets[first][face], target_insets[second][face] = 0, 1
    if fallback:
        gaps = [[1] + [int(p[j][0] == p[j - 1][0]) for j in range(1, len(p))] for p in programs]
        widths = [max(8, max(sum(length for _, length in programs[i]) + sum(gaps[i]) + (face >= 2)
                            for i, cell in enumerate(cells) if cell[1] == col))
                  for col in range(plan.shape[1])]
        target_widths = [7 if shift != 'range' else 12] * plan.shape[1]
        target_heights = [3] * plan.shape[0]
    dl, db = _geometry(plan, widths, heights, insets)
    tl, tb = _geometry(plan, target_widths, target_heights, target_insets)
    donor_gates, target_gates = [], []
    for i, (coarse_r, coarse_c) in enumerate(cells):
        directions = []
        if coarse_c == 0 or not plan[coarse_r, coarse_c - 1]:
            directions.append(1)
        if coarse_c == plan.shape[1] - 1 or not plan[coarse_r, coarse_c + 1]:
            directions.append(-1)
        sign = int(rng.choice(directions))
        r, c, R, C = db[i]
        donor_row = int(rng.integers(r, R))
        donor_gates.append(((donor_row, c - 1 if sign == 1 else C), (0, sign)))
        r, c, R, C = tb[i]
        gate_row = int(rng.integers(r, R))
        notch_count = 2 if shift == 'maze' else int(rng.random() < .45)
        if witness == 0 and i == 0:
            notch_count = 1
        if notch_count:
            # Wall fingers leave a one-cell passage; two opposite fingers
            # create a winding route. They never divide or merge chambers.
            columns = [c + (C - c) // 2] if notch_count == 1 else [c + 1, C - 2]
            if sign == -1:
                columns = columns[::-1]
            gate_row = r
            for k, x in enumerate(columns):
                if k % 2 == 0:
                    tl[r:R - 1, x] = -2
                else:
                    tl[r + 1:R, x] = -2
        target_gates.append(((gate_row, c - 1 if sign == 1 else C), (0, sign)))
    donor = _paint(dl, bg, wall, port, donor_gates, programs, gaps)
    target = _paint(tl, bg, wall, port, target_gates)
    expected = _constructed_answer(tl, target, target_gates, programs)
    inp = _embed(rng, donor, target, bg)
    return {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(expected, rotation).copy()}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _witness_ok(pair, index):
    info = _analyze(pair['input'])
    assert np.array_equal(pair['output'], _answer(info))
    assert len(info['mappings']) == 1
    if index == 0:
        for variant in ('scan', 'area', 'manhattan', 'stripe', 'uniform', 'short', 'long', 'far',
                        'gap_phase', 'period_plus_one'):
            # The other priority order is supplied by the second demonstration.
            if variant == 'short':
                continue
            assert not np.array_equal(pair['output'], _answer(info, variant)), variant
    elif index == 1:
        assert not np.array_equal(pair['output'], _answer(info, 'short'))
    # Find a two-run room with both colors visible and another collision at 7.
    out = pair['output']
    good = False
    for i, room in enumerate(info['donor']['rooms']):
        if len(room['runs']) != 2 or {x[1] for x in room['runs']} != {2, 3}:
            continue
        target = info['target']['rooms'][info['mappings'][0][i]]
        ds = _distances(target)
        values = {int(out[p]) for p in ds}
        good |= (all(x[0] in values for x in room['runs']) and 7 in ds.values()
                 and all(out[p] == room['runs'][0][0] for p, d in ds.items() if d == 7))
    assert good, 'observable priority, recurrence, and noninitial collision'


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.choice(10, 2, replace=False)))
    base_rotation = int(rng.integers(4))
    curriculum = str(rng.choice(['turn', 'maze', 'range']))
    first_count = int(rng.integers(3, 7))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', num_train), ('test', num_test)):
        for index in range(number):
            witness = index if split == 'train' and index < 2 else None
            count = first_count if witness == 0 else int(rng.choice(
                [n for n in range(2, 7) if witness != 1 or n != first_count]))
            rotation = (base_rotation + (1 if split == 'test' and curriculum == 'turn'
                                         else 2 if witness == 1 else int(rng.choice([0, 2])))) % 4
            shift = curriculum if split == 'test' and curriculum in ('maze', 'range') else 'ordinary'
            for attempt in range(80):
                try:
                    pair = _pair(rng, count, witness, colors, rotation, shift, fallback=attempt >= 60)
                    info = _analyze(pair['input'])
                    assert np.array_equal(pair['output'], _answer(info))
                    assert len(info['mappings']) == 1
                    if witness is not None:
                        _witness_ok(pair, witness)
                    assert _key(pair['input']) not in seen
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError('bounded chamber construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def _teaching_evidence(pairs, infos):
    singles, orders, edges = set(), set(), set()
    for pair, info in zip(pairs, infos):
        out = np.asarray(pair['output'])
        for i, room in enumerate(info['donor']['rooms']):
            target = info['target']['rooms'][info['mappings'][0][i]]
            ds = _distances(target)
            runs = room['runs']
            if len(runs) == 1:
                color, length, _ = runs[0]
                if length + 1 in ds.values() and all(out[p] == color for p, d in ds.items()
                                                   if d in (1, length + 1)):
                    singles.add(length)
            if len(runs) == 2 and runs[0][0] != runs[1][0] and {x[1] for x in runs} == {2, 3}:
                values = {int(out[p]) for p in ds}
                if (7 in ds.values() and all(run[0] in values for run in runs)
                        and all(out[p] == runs[0][0] for p, d in ds.items() if d in (1, 7))):
                    orders.add(int(np.sign(runs[0][1] - runs[1][1])))
                    edges.add((runs[0][0], runs[1][0]))
    assert {2, 3} <= singles, 'two visible single-run numerical recurrences'
    assert orders == {-1, 1}, 'both period orders must visibly select the nearer run'
    assert any((b, a) in edges for a, b in edges), 'fixed color ranking must fail'


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    infos = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            expected = _answer(info)
            assert np.array_equal(out, expected), 'wrong correspondence, pulse, distance, or crop'
            target = info['target']['crop']
            assert out.shape == target.shape
            interiors = np.zeros(out.shape, dtype=bool)
            for room in info['target']['rooms']:
                interiors[tuple(room['points'].T)] = True
                distances = _distances(room)
                for distance in set(distances.values()):
                    assert len({int(out[p]) for p, d in distances.items() if d == distance}) == 1
            assert np.array_equal(out[~interiors], target[~interiors]), 'walls, gates, and exterior must survive'
            assert np.all(target[interiors] == info['bg'])
            assert np.array_equal(out == info['wall'], target == info['wall'])
            assert np.array_equal(out == info['port'], target == info['port'])
            assert np.any(out[interiors] != info['bg'])
            infos[split].append(info)
    _witness_ok(puzzle['train'][0], 0)
    _witness_ok(puzzle['train'][1], 1)
    _teaching_evidence(puzzle['train'], infos['train'])
    assert len({len(i['donor']['rooms']) for i in infos['train']}) >= 2
    for variant in ('uniform', 'scan', 'area', 'stripe', 'manhattan', 'short', 'long', 'far',
                    'gap_phase', 'period_plus_one'):
        assert any(not np.array_equal(np.asarray(p['output']), _answer(i, variant))
                   for p, i in zip(puzzle['train'], infos['train'])), variant


def _features(pair):
    info = _analyze(np.asarray(pair['input']))
    target, donor = info['target']['rooms'], info['donor']['rooms']
    mapping = info['mappings'][0]
    ds = [_distances(room) for room in target]
    deltas = [len(target[mapping[i]]['points']) - len(room['points']) for i, room in enumerate(donor)]
    return {'shape': tuple(pair['input'].shape), 'output_shape': tuple(pair['output'].shape),
            'rooms': len(target), 'wall': info['wall'], 'bg': info['bg'], 'port': info['port'],
            'normals': tuple(sorted(set(room['normal'] for room in target))),
            'max_depth': max(max(d.values()) for d in ds),
            'scan_changed': any(i != j for i, j in mapping.items()),
            'detour': any(d > sum(abs(p[k] - room['gate'][k]) for k in (0, 1))
                          for room, dist in zip(target, ds) for p, d in dist.items()),
            'area_signs': tuple(sorted(set(int(np.sign(d)) for d in deltas))),
            'periods': tuple(sorted(set(run[1] for room in donor for run in room['runs']))),
            'run_counts': tuple(sorted(set(len(room['runs']) for room in donor))),
            'topology': tuple(tuple(row) for row in _relations(target)),
            'structure': (info['target']['crop'].shape, (info['target']['crop'] == info['wall']).tobytes())}


def self_test():
    # Fixed arithmetic and geodesic expectations, not solve-generated labels.
    room = {'points': np.array([(r, c) for r in range(1, 4) for c in range(1, 8)
                               if not (c == 4 and r < 3)]),
            'origin': (1, 1), 'gate': (1, 0), 'normal': (0, 1)}
    distances = _distances(room)
    assert distances[(1, 3)] == 3 and distances[(1, 5)] == 9 and distances[(3, 7)] == 9
    runs = [(2, 2, 2), (3, 3, 4)]
    assert [next((c for c, length, _ in runs if (d - 1) % length == 0), 0)
            for d in range(1, 14)] == [2, 0, 2, 3, 2, 0, 2, 0, 2, 3, 2, 0, 2]
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                features.append(_features(pair))
    assert len({_key(p['train'][0]['input']) for p in puzzles}) == 42
    assert len({f['shape'] for f in features}) >= 35
    assert len({f['structure'] for f in features}) >= 100
    assert len({f['topology'] for f in features}) >= 15
    assert {f['rooms'] for f in features} == set(range(2, 7))
    assert {v for f in features for v in f['periods']} == set(range(1, 6))
    assert {v for f in features for v in f['run_counts']} == {1, 2, 3}
    assert {v for f in features for v in f['normals']} == set(_STEPS)
    assert len({(f['bg'], f['wall'], f['port']) for f in features}) >= 60
    assert sum(f['scan_changed'] for f in features) >= 42
    assert sum(f['detour'] for f in features) >= 42
    assert max(f['max_depth'] for f in features) >= 16
    # Rotation, literal recoloring, and exterior padding must commute with solve.
    pair = puzzles[0]['train'][0]
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(pair['input'], turn)), np.rot90(pair['output'], turn))
    recolor = np.array([4, 8, 0, 7, 2, 9, 1, 5, 6, 3], dtype=np.int8)
    assert np.array_equal(solve(recolor[pair['input']]), recolor[pair['output']])
    bg = _analyze(pair['input'])['bg']
    assert np.array_equal(solve(np.pad(pair['input'], 1, constant_values=bg)), pair['output'])
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            p = generate_puzzle(812, a, b)
            validate_generated_puzzle(p)
            assert (len(p['train']), len(p['test'])) == (a, b)
    assert all(_key(p[k]) == _key(q[k]) for split in ('train', 'test')
               for p, q in zip(puzzles[0][split], generate_puzzle(0)[split]) for k in ('input', 'output'))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (2.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    broken = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]]
              for s in ('train', 'test')}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted wall/crop accepted')


if __name__ == '__main__':
    self_test()
    print('chamber echoes self-test passed (42 seeds, counts, invariants, and metamorphisms)')
