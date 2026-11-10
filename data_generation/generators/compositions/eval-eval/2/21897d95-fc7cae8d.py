"""Compass-translated, simultaneous territory transfers.

Only standard Python and NumPy are used. No parent modules or stored examples
are imported. All randomness belongs to generate_puzzle's one local Generator.
"""

from collections import Counter
import json

import numpy as np


# Clockwise side normals: north, east, south, west.
D = ((-1, 0), (0, 1), (1, 0), (0, -1))
# Each tuple maps an old side index to its new side index. These act on
# DIRECTIONS only. The board's positions and dimensions never transform.
MAPS = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2),
        (0, 3, 2, 1), (2, 1, 0, 3), (3, 2, 1, 0), (1, 0, 3, 2))
NAMES = ('identity', 'clockwise', 'half_turn', 'counterclockwise',
         'left_right_mirror', 'up_down_mirror', 'transpose', 'anti_transpose')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    grid = np.asarray(value)
    _need(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), 'bad grid dimensions')
    _need(np.issubdtype(grid.dtype, np.integer), 'grid must contain integers')
    _need(bool(np.all((0 <= grid) & (grid <= 9))), 'bad grid colors')
    return grid.astype(np.int8, copy=True)


def _key(grid):
    return tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes()


def _add(point, direction):
    return point[0] + direction[0], point[1] + direction[1]


def _inside(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _components(grid, excluded=()):
    """Four-connected components; equal colors need not mean equal IDs."""
    labels = np.full(grid.shape, -1, dtype=np.int16)
    colors, cells = [], []
    for start in np.ndindex(grid.shape):
        color = int(grid[start])
        if labels[start] >= 0 or color in excluded:
            continue
        label = len(colors)
        colors.append(color)
        points, stack = [], [start]
        labels[start] = label
        while stack:
            point = stack.pop()
            points.append(point)
            for direction in D:
                q = _add(point, direction)
                if (_inside(q, grid.shape) and labels[q] < 0
                        and int(grid[q]) == color):
                    labels[q] = label
                    stack.append(q)
        cells.append(points)
    return labels, colors, cells


def _side(grid, side):
    return (grid[0, :], grid[:, -1], grid[-1, :], grid[:, 0])[side]


def _paint_side(grid, side, color):
    _side(grid, side)[:] = color


def _guide(grid, bounds, side):
    r, c, h, w = bounds
    if side == 0:
        return grid[r - 1, c:c + w]
    if side == 1:
        return grid[r:r + h, c + w]
    if side == 2:
        return grid[r + h, c:c + w]
    return grid[r:r + h, c - 1]


def _adjacent(sides):
    return len(sides) == 2 and (sides[0] - sides[1]) % 4 in (1, 3)


def _read_frame(grid):
    """Find a unique pair of interior bars matching adjacent full outer sides."""
    labels, colors, cells = _components(grid)
    outer = {}
    for s in range(4):
        values = np.unique(_side(grid, s)[1:-1])
        if len(values) == 1:
            outer[s] = int(values[0])
    answers = []
    for corner in range(4):
        sides = (corner, (corner + 1) % 4)
        if not all(s in outer for s in sides):
            continue
        controls = {s: outer[s] for s in sides}
        markers = set(controls.values())
        if len(markers) != 2:
            continue
        bars = []
        for color in sorted(markers):
            found = []
            for k, pts in enumerate(cells):
                if colors[k] != color or len(pts) < 5:
                    continue
                rr = [p[0] for p in pts]
                cc = [p[1] for p in pts]
                if min(rr) == 0 or min(cc) == 0 or max(rr) == grid.shape[0] - 1 or max(cc) == grid.shape[1] - 1:
                    continue
                if min(rr) == max(rr) and len(pts) == max(cc) - min(cc) + 1:
                    found.append(('H', min(rr), min(cc), max(cc), color))
                if min(cc) == max(cc) and len(pts) == max(rr) - min(rr) + 1:
                    found.append(('V', min(cc), min(rr), max(rr), color))
            if len(found) != 1:
                break
            bars.extend(found)
        if len(bars) != 2 or {b[0] for b in bars} != {'H', 'V'}:
            continue
        horizontal = next(b for b in bars if b[0] == 'H')
        vertical = next(b for b in bars if b[0] == 'V')
        _, hr, c0, c1, hc = horizontal
        _, vc, r0, r1, vcolor = vertical
        hs = 0 if hr == r0 - 1 else (2 if hr == r1 + 1 else -1)
        vs = 3 if vc == c0 - 1 else (1 if vc == c1 + 1 else -1)
        if hs < 0 or vs < 0:
            continue
        bounds = r0, c0, r1 - r0 + 1, c1 - c0 + 1
        r, c, h, w = bounds
        if r < 2 + (hs == 0) or c < 2 + (vs == 3):
            continue
        if r + h > grid.shape[0] - 2 - (hs == 2) or c + w > grid.shape[1] - 2 - (vs == 1):
            continue
        guides = {hs: hc, vs: vcolor}
        choices = [m for m in MAPS if all(controls.get(m[s]) == col for s, col in guides.items())]
        if len(choices) != 1:
            continue
        # No border clutter: outside the board and the four bars is one fill.
        outside = np.ones(grid.shape, dtype=bool)
        outside[r:r + h, c:c + w] = False
        for s in guides:
            _guide(outside, bounds, s)[:] = False
        for s in controls:
            _paint_side(outside, s, False)
        remaining = np.unique(grid[outside])
        if len(remaining) != 1 or int(remaining[0]) in markers:
            continue
        bg = int(remaining[0])
        board = grid[r:r + h, c:c + w].copy()
        if set(map(int, np.unique(board))) & markers:
            continue
        # Check controls including their single shared, arbitrary-color corner.
        if any(not np.all(_side(grid, s)[1:-1] == col) for s, col in controls.items()):
            continue
        answers.append(dict(bounds=bounds, board=board, background=bg,
                            guides=guides, controls=controls, mapping=choices[0]))
    _need(len(answers) == 1, 'frame/guide correspondence must be unique')
    return answers[0]


def _stencil(center, side):
    dr, dc = D[side]
    r, c = center
    return {(r, c), (r - dc, c + dr), (r + dc, c - dr), (r + dr, c + dc)}


def _read_arrows(board, bg):
    """Infer the reserved ink from complete isolated T components and halos."""
    _, colors, cells = _components(board)
    candidates = []
    for ink in sorted(set(colors) - {bg}):
        arrows, valid = [], True
        for color, pts in zip(colors, cells):
            if color != ink:
                continue
            points = set(pts)
            if len(points) != 4:
                valid = False
                break
            interpretations = []
            for center in pts:
                r, c = center
                if not (1 <= r < board.shape[0] - 1 and 1 <= c < board.shape[1] - 1):
                    continue
                for side in range(4):
                    if _stencil(center, side) != points:
                        continue
                    support = {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1)} - points
                    fills = {int(board[p]) for p in support}
                    if len(fills) == 1 and next(iter(fills)) not in (ink, bg):
                        interpretations.append(dict(center=center, side=side,
                                                    write=points, host=fills.pop()))
            if len(interpretations) != 1:
                valid = False
                break
            arrows.extend(interpretations)
        if not valid or not arrows:
            continue
        clean = board.copy()
        for arrow in arrows:
            for p in arrow['write']:
                clean[p] = arrow['host']
        ids, fills, territories = _components(clean, (bg,))
        if len(territories) < 2 or any(len(pts) < 9 for pts in territories):
            continue
        for arrow in arrows:
            r, c = arrow['center']
            source = int(ids[r, c])
            if source < 0 or not np.all(ids[r - 1:r + 2, c - 1:c + 2] == source):
                valid = False
                break
            arrow['source'] = source
        if valid:
            candidates.append(dict(ink=ink, clean=clean, ids=ids, colors=fills,
                                   territories=territories, arrows=arrows))
    _need(len(candidates) == 1, 'T ink and original hosts must be uniquely recoverable')
    return candidates[0]


def _hit(ids, center, source, side):
    """First other ORIGINAL component on the ray; background is transparent."""
    p, distance = center, 0
    while True:
        p = _add(p, D[side])
        distance += 1
        if not _inside(p, ids.shape):
            return -1, distance
        target = int(ids[p])
        if target >= 0 and target != source:
            return target, distance


def _paint(info, mapping=None, payload_mode='original'):
    mapping = info['mapping'] if mapping is None else mapping
    assignments, targets, distances = {}, [], []
    for a in info['arrows']:
        target, distance = _hit(info['ids'], a['center'], a['source'], mapping[a['side']])
        targets.append(target)
        distances.append(distance)
        if target < 0:
            continue
        payload = info['colors'][a['source']]
        _need(target not in assignments or assignments[target] == payload, 'conflicting incoming transfers')
        assignments[target] = payload
    if payload_mode == 'updated':
        # Deliberately wrong hypothesis for a diagnostic: read the final host
        # colors in a chain rather than its original colors.
        assignments = {target: assignments.get(a['source'], info['colors'][a['source']])
                       for a, target in zip(info['arrows'], targets) if target >= 0}
    painted = info['clean'].copy()
    if payload_mode == 'color_wide':
        for target, payload in assignments.items():
            painted[info['clean'] == info['colors'][target]] = payload
    else:
        for target, payload in assignments.items():
            painted[info['ids'] == target] = payload
    return painted, assignments, targets, distances


def _shielded(info):
    """Does a first hit protect a visibly unchanged island further along a ray?"""
    _, assignments, _, _ = _paint(info)
    for a in info['arrows']:
        p, first = a['center'], -1
        while True:
            p = _add(p, D[info['mapping'][a['side']]])
            if not _inside(p, info['ids'].shape):
                break
            k = int(info['ids'][p])
            if k < 0 or k == a['source']:
                continue
            if first < 0:
                first = k
            elif k != first and k not in assignments and info['colors'][k] != info['colors'][a['source']]:
                return True
    return False


def _decode(input_grid):
    frame = _read_frame(_grid(input_grid))
    frame.update(_read_arrows(frame['board'], frame['background']))
    predicted, assignments, targets, distances = _paint(frame)
    frame.update(assignments=assignments, targets=targets, distances=distances)
    return predicted, frame


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only this input; retain the board's position system and exact crop."""
    return _decode(input_grid)[0]


def _rooms(rng, h, w, desired):
    pads = [int(rng.integers(0, 2)) for _ in range(4)]
    rooms = [(pads[0], h - pads[2], pads[3], w - pads[1])]
    for _ in range(desired - 1):
        possibilities = []
        for i, (r0, r1, c0, c1) in enumerate(rooms):
            for axis, length in ((0, r1 - r0), (1, c1 - c0)):
                if length >= 9:
                    possibilities.append((i, axis))
        if not possibilities:
            break
        weights = np.array([(rooms[i][1] - rooms[i][0]) * (rooms[i][3] - rooms[i][2])
                            for i, _ in possibilities], dtype=float)
        i, axis = possibilities[int(rng.choice(len(possibilities), p=weights / weights.sum()))]
        r0, r1, c0, c1 = rooms.pop(i)
        low, high = (r0, r1) if axis == 0 else (c0, c1)
        gap = int(rng.integers(1, min(3, high - low - 8) + 1))
        cut = int(rng.integers(low + 4, high - gap - 3))
        rooms.extend(((r0, cut, c0, c1), (cut + gap, r1, c0, c1)) if axis == 0
                     else ((r0, r1, c0, cut), (r0, r1, cut + gap, c1)))
    labels = np.full((h, w), -1, dtype=np.int16)
    for k, (r0, r1, c0, c1) in enumerate(rooms):
        labels[r0:r1, c0:c1] = k
        # Independent small corner bites produce Ls, steps and concavities.
        # They cannot sever an island or remove its central 3x3 support.
        for corner in rng.permutation(4)[:int(rng.integers(0, 3))]:
            if min(r1 - r0, c1 - c0) < 6 or rng.random() < .25:
                continue
            depth = int(rng.integers(1, min(3, (r1 - r0) // 2)))
            breadth = int(rng.integers(1, min(4, (c1 - c0) // 2)))
            for step in range(depth):
                width = max(1, breadth - step)
                rr = r0 + step if corner in (0, 1) else r1 - 1 - step
                cc = slice(c0, c0 + width) if corner in (0, 3) else slice(c1 - width, c1)
                labels[rr, cc] = -1
    return labels


def _arrow_candidates(labels, mappings):
    candidates = []
    h, w = labels.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            source = int(labels[r, c])
            if source < 0 or not np.all(labels[r - 1:r + 2, c - 1:c + 2] == source):
                continue
            for side in range(4):
                targets = tuple(_hit(labels, (r, c), source, m[side])[0] for m in mappings)
                if all(t >= 0 for t in targets):
                    candidates.append(dict(center=(r, c), side=side, source=source,
                                           targets=targets, write=_stencil((r, c), side)))
    return candidates


def _compatible(candidate, selected):
    for old in selected:
        if any(x == y for x, y in zip(candidate['targets'], old['targets'])):
            return False
        # No shared 3x3 restoration support and no connected ink stencils.
        if max(abs(candidate['center'][a] - old['center'][a]) for a in (0, 1)) <= 2:
            return False
        if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 1
               for p in candidate['write'] for q in old['write']):
            return False
    return True


def _choose_plan(rng, candidates, count, teaching):
    if not candidates:
        return None
    # Order once; every bounded search thereafter is deterministic.
    ordered = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    if teaching:
        chain_pairs = []
        by_source = {}
        for a in ordered:
            by_source.setdefault(a['source'], []).append(a)
        for a in ordered:
            for b in by_source.get(a['targets'][0], ()):
                if b['targets'][0] != a['source'] and _compatible(b, [a]):
                    chain_pairs.append((a, b))
                    if len(chain_pairs) >= 80:
                        break
            if len(chain_pairs) >= 80:
                break
        if not chain_pairs:
            return None
        selected = list(chain_pairs[int(rng.integers(len(chain_pairs)))])
    else:
        selected = []
    for a in ordered:
        if len(selected) >= count:
            break
        if _compatible(a, selected):
            selected.append(a)
    if not selected or (teaching and (len(selected) < 3 or len({a['side'] % 2 for a in selected}) < 2)):
        return None
    return selected


def _fill_colors(rng, n, plan, mappings, palette, teaching):
    for _ in range(120):
        colors = [int(c) for c in rng.choice(palette, n)]
        if len(set(colors)) < 3:
            continue
        if any(colors[a['source']] == colors[t] for a in plan for t in a['targets']):
            continue
        if teaching:
            assigned = {a['targets'][0]: colors[a['source']] for a in plan}
            final = [assigned.get(k, col) for k, col in enumerate(colors)]
            if not any(colors[i] == colors[j] and final[i] != final[j]
                       for i in range(n) for j in range(i)):
                continue
        return colors
    return None


def _make_board(rng, mappings, teaching=False, test=False):
    """Construct graph labels first, then verify the visible input reconstruction."""
    for _ in range(140):
        low, high = (17, 25) if test else (15, 24)
        h, w = int(rng.integers(low, high)), int(rng.integers(low, high))
        if rng.random() < .2:
            w = h
        if teaching and h == w:
            continue
        labels = _rooms(rng, h, w, int(rng.integers(5 if teaching else 4, 10)))
        n = int(labels.max()) + 1
        if teaching and n < 5:
            continue
        candidates = _arrow_candidates(labels, mappings)
        count = int(rng.integers(3, min(6, n) + 1)) if teaching else int(rng.integers(1, min(7, n) + 1))
        plan = _choose_plan(rng, candidates, count, teaching)
        if plan is None:
            continue
        palette = [int(v) for v in rng.permutation(10)]
        bg, ink, m0, m1 = palette[:4]
        colors = _fill_colors(rng, n, plan, mappings, palette[4:], teaching)
        if colors is None:
            continue
        clean = np.full(labels.shape, bg, dtype=np.int8)
        for k, color in enumerate(colors):
            clean[labels == k] = color
        if teaching and not any(np.all(_side(clean, s) == bg) for s in range(4)):
            continue
        board = clean.copy()
        for a in plan:
            for p in a['write']:
                board[p] = ink
        info = dict(ids=labels, colors=colors, arrows=plan, clean=clean, mapping=mappings[0])
        outputs = [_paint(info, m)[0] for m in mappings]
        if teaching:
            if not _shielded(info):
                continue
            if np.array_equal(outputs[0], outputs[1]):
                continue
            try:
                if np.array_equal(outputs[0], _paint(info, MAPS[0])[0]):
                    continue
            except ValueError:
                continue
            if np.array_equal(outputs[0], _paint(info, payload_mode='updated')[0]):
                continue
            # Quarter-turn inverse is a distinct, visibly wrong reading.
            inverse = tuple(mappings[0].index(k) for k in range(4))
            try:
                if np.array_equal(outputs[0], _paint(info, inverse)[0]):
                    continue
            except ValueError:
                pass
        try:
            recovered = _read_arrows(board, bg)
        except ValueError:
            continue
        if not np.array_equal(recovered['clean'], clean) or len(recovered['arrows']) != len(plan):
            continue
        return board, outputs, (bg, ink, m0, m1)
    raise RuntimeError('bounded board construction exhausted')


def _configuration(inner_corner, outer_corner, reverse=False):
    inner = (inner_corner, (inner_corner + 1) % 4)
    outer = (outer_corner, (outer_corner + 1) % 4)
    if reverse:
        outer = outer[::-1]
    choices = [m for m in MAPS if all(m[s] == t for s, t in zip(inner, outer))]
    assert len(choices) == 1
    return inner, outer, choices[0]


def _wrap(rng, board, roles, inner, outer):
    bg, ink, m0, m1 = roles
    h, w = board.shape
    H, W = int(rng.integers(h + 5, 31)), int(rng.integers(w + 5, 31))
    r = int(rng.integers(2 + (0 in inner), H - h - 1 - (2 in inner)))
    c = int(rng.integers(2 + (3 in inner), W - w - 1 - (1 in inner)))
    grid = np.full((H, W), bg, dtype=np.int8)
    grid[r:r + h, c:c + w] = board
    bounds = r, c, h, w
    for s, color in zip(inner, (m0, m1)):
        _guide(grid, bounds, s)[:] = color
    for s, color in zip(outer, (m0, m1)):
        _paint_side(grid, s, color)
    return grid


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _need(_integer(seed) and seed >= 0, 'seed must be a nonnegative integer')
    _need(_integer(num_train) and 2 <= num_train <= 10, 'num_train must be an integer in 2..10')
    _need(_integer(num_test) and 1 <= num_test <= 4, 'num_test must be an integer in 1..4')
    _need(num_train > num_test, 'num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    # Two examples teach matching rather than an occupied-side lookup.
    ic = int(rng.integers(4))
    oc = (ic + int(rng.choice((1, 3)))) % 4
    inner, outer, first = _configuration(ic, oc)
    _, reversed_outer, second = _configuration(ic, oc, True)
    board, outs, roles = _make_board(rng, (first, second), teaching=True)
    inp = _wrap(rng, board, roles, inner, outer)
    changed = inp.copy()
    # Swapping ONLY border identities keeps all guide coordinates and contents.
    for s, color in zip(reversed_outer, roles[2:]):
        _paint_side(changed, s, color)
    puzzle = {'train': [dict(input=inp, output=outs[0]), dict(input=changed, output=outs[1])], 'test': []}
    held = [(i, o) for i in range(4) for o in range(4) if (i, o) != (ic, oc)]
    hi, ho = held[int(rng.integers(len(held)))]
    available = [(i, o, rev) for i in range(4) for o in range(4) for rev in (False, True)
                 if (i, o) != (hi, ho)]
    used = {_key(inp), _key(changed)}
    for split, count in (('train', num_train - 2), ('test', num_test)):
        for _ in range(count):
            for attempt in range(30):
                if split == 'test':
                    i, o, rev = hi, ho, bool(rng.integers(2))
                else:
                    i, o, rev = available[int(rng.integers(len(available)))]
                ins, os, mapping = _configuration(i, o, rev)
                if split == 'test' and mapping == MAPS[0]:
                    continue
                b, outputs, colors = _make_board(rng, (mapping,), test=split == 'test')
                g = _wrap(rng, b, colors, ins, os)
                if _key(g) in used:
                    continue
                # Keep both inherited concepts observable in every test.
                if split == 'test':
                    expected, info = _decode(g)
                    if np.array_equal(expected, info['clean']):
                        continue
                    try:
                        if mapping != MAPS[0] and np.array_equal(expected, _paint(info, MAPS[0])[0]):
                            continue
                    except ValueError:
                        pass
                puzzle[split].append(dict(input=g, output=outputs[0]))
                used.add(_key(g))
                break
            else:
                raise RuntimeError('bounded distinct-pair construction exhausted')
    # Pair order carries no instruction. Keep the controlled witnesses adjacent.
    if rng.random() < .5:
        puzzle['train'][0], puzzle['train'][1] = puzzle['train'][1], puzzle['train'][0]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, output):
    colors, assignments = info['colors'], info['assignments']
    final = [assignments.get(k, c) for k, c in enumerate(colors)]
    locality = any(colors[i] == colors[j] and final[i] != final[j]
                   for i in range(len(colors)) for j in range(i))
    chain = any(t == b['source'] and colors[a['source']] != colors[b['source']]
                and u >= 0 and u != a['source']
                for a, t in zip(info['arrows'], info['targets'])
                for b, u in zip(info['arrows'], info['targets']) if a is not b)
    nonrect = 0
    for pts in info['territories']:
        rr, cc = zip(*pts)
        nonrect += len(pts) != (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1)
    return dict(mapping=NAMES[MAPS.index(info['mapping'])], territories=len(colors),
                arrows=len(info['arrows']), directions=sorted({a['side'] for a in info['arrows']}),
                chain=chain, locality=locality, nonrectangular=nonrect,
                shielded=_shielded(info),
                repeated=len(colors) > len(set(colors)),
                blank_sides=[s for s in range(4) if np.all(_side(info['clean'], s) == info['background'])],
                shape=tuple(output.shape), ray_min=min(info['distances']), ray_max=max(info['distances']),
                changed=int(np.count_nonzero(output != info['clean'])),
                ink=info['ink'], background=info['background'])


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal counts')
    seen, decoded = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            predicted, info = _decode(inp)
            _need(np.array_equal(predicted, out), 'wrong routed transfer or board extent')
            _need(out.shape == info['board'].shape, 'board geometry must stay fixed')
            _need(all(t >= 0 for t in info['targets']), 'generated arrows must hit a territory')
            _need(all(info['colors'][a['source']] != info['colors'][t]
                      for a, t in zip(info['arrows'], info['targets'])), 'generated transfer must change its target')
            _need(bool(np.all(out[info['ids'] < 0] == info['background'])), 'a corridor was painted')
            for k, pts in enumerate(info['territories']):
                expected = info['assignments'].get(k, info['colors'][k])
                _need(all(out[p] == expected for p in pts), 'territory was not filled uniformly')
            _need(info['ink'] not in out and not set(info['guides'].values()) & set(map(int, np.unique(out))),
                  'annotation ink or key survived')
            decoded[split].append((inp, out, info, _features(info, out)))
    train = decoded['train']
    contrasts = [(a, b) for i, a in enumerate(train) for b in train[i + 1:]
                 if a[2]['bounds'] == b[2]['bounds'] and a[2]['guides'] == b[2]['guides']
                 and set(a[2]['controls']) == set(b[2]['controls'])
                 and a[2]['mapping'] != b[2]['mapping']
                 and np.array_equal(a[2]['board'], b[2]['board'])
                 and not np.array_equal(a[1], b[1])]
    _need(bool(contrasts), 'missing controlled correspondence contrast')
    _need(any(f['chain'] for _, _, _, f in train), 'missing simultaneous-chain demonstration')
    _need(any(f['locality'] for _, _, _, f in train), 'missing component-locality demonstration')
    _need(any(f['shielded'] for _, _, _, f in train), 'missing first-hit shielding demonstration')
    _need(any(f['blank_sides'] for _, _, _, f in train), 'missing guide-defined blank-margin demonstration')
    _need(any(info['mapping'] in (MAPS[1], MAPS[3]) for _, _, info, _ in train), 'missing quarter-turn direction mapping')
    _need(any(info['mapping'] in MAPS[4:] for _, _, info, _ in train), 'missing reflected direction mapping')
    # These are teaching checks, not hidden conventions used by solve.
    changed_by_A = any(not np.array_equal(out, info['clean']) for _, out, info, _ in train)
    changed_by_B = False
    snapshot = False
    for _, out, info, _ in train:
        try:
            changed_by_B |= not np.array_equal(out, _paint(info, MAPS[0])[0])
        except ValueError:
            changed_by_B = True
        snapshot |= not np.array_equal(out, _paint(info, payload_mode='updated')[0])
    _need(changed_by_A and changed_by_B and snapshot, 'parent-omission/snapshot teaching witness missing')
    train_config = {(tuple(sorted(info['guides'])), tuple(sorted(info['controls']))) for _, _, info, _ in train}
    for _, _, info, _ in decoded['test']:
        config = tuple(sorted(info['guides'])), tuple(sorted(info['controls']))
        _need(config not in train_config, 'test must recombine held-out guide/control side positions')
    return None


def self_test():
    """Bounded: 42 puzzles streamed, no sample/trace files or persistent cache."""
    metrics, signatures, shapes, palettes, partitions = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                out, info = _decode(pair['input'])
                _need(np.array_equal(out, pair['output']), 'solver regression')
                f = _features(info, out)
                f['split'] = split
                metrics.append(f)
                shapes.add(tuple(pair['input'].shape))
                palettes.add((info['background'], info['ink'], tuple(sorted(info['guides'].values()))))
                partitions.add(_key(info['ids']))
        signature = tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s])
        signatures.add(signature)
        if seed < 2:
            other = generate_puzzle(seed)
            _need(signature == tuple(_key(p['input']) for s in ('train', 'test') for p in other[s]), 'nondeterministic')
    _need(len(signatures) == 42, 'puzzle collapse')
    _need(len(partitions) >= 140 and len(shapes) >= 40 and len(palettes) >= 80, 'geometry/palette collapse')
    _need({f['mapping'] for f in metrics} == set(NAMES), 'missing compass symmetries')
    _need({d for f in metrics for d in f['directions']} == set(range(4)), 'missing arrow directions')
    _need(sum(f['nonrectangular'] > 0 for f in metrics) >= 25, 'missing irregular islands')
    _need(sum(f['chain'] for f in metrics) >= 42 and sum(f['locality'] for f in metrics) >= 42,
          'missing inherited concept coverage')
    _need({f['arrows'] for f in metrics} >= {1, 2, 3, 4}, 'arrow-count collapse')
    _need({s for f in metrics for s in f['blank_sides']} == set(range(4)), 'missing blank board margins')
    _need(max(f['ray_max'] for f in metrics) >= 10, 'missing long-distance transfers')
    # No output may depend on a seed, stored construction state or previous solve.
    first = generate_puzzle(123, 2, 1)
    query = first['test'][0]['input'].tolist()
    expected = first['test'][0]['output'].copy()
    generate_puzzle(456, 3, 2)
    _need(np.array_equal(solve(query), expected), 'solve depends on generation state')
    # Every legal explicit combination, and nonintegral/bool/None failures.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            _need(len(p['train']) == nt and len(p['test']) == ne, 'ignored counts')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    for mutation in ('output', 'crop', 'guide'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in first[s]] for s in first}
        p = bad['test'][0]
        if mutation == 'output':
            p['output'][0, 0] = (int(p['output'][0, 0]) + 1) % 10
        elif mutation == 'crop':
            p['output'] = p['output'][1:, :]
        else:
            info = _read_frame(p['input'])
            line = _guide(p['input'], info['bounds'], next(iter(info['guides'])))
            line[len(line) // 2] = info['background']
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError('mutation accepted: ' + mutation)
    return dict(seeds=42, pairs=len(metrics), distinct_partitions=len(partitions),
                input_shapes=len(shapes), role_palettes=len(palettes),
                mappings=dict(Counter(f['mapping'] for f in metrics)),
                arrow_counts=dict(sorted(Counter(f['arrows'] for f in metrics).items())),
                territory_counts=dict(sorted(Counter(f['territories'] for f in metrics).items())),
                chain_pairs=sum(f['chain'] for f in metrics), locality_pairs=sum(f['locality'] for f in metrics),
                irregular_pairs=sum(f['nonrectangular'] > 0 for f in metrics),
                shielding_pairs=sum(f['shielded'] for f in metrics),
                long_rays=max(f['ray_max'] for f in metrics),
                semantic_verification='generator-owned checks only')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
