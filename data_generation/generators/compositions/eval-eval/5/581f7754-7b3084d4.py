"""Registered jigsaws: boundary projection pins pivots; exact cover chooses turns.

Only the standard library and NumPy are used. No parent code or data is loaded.
"""
from __future__ import annotations

import json
from math import isqrt

import numpy as np

EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer cells required')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'colors outside 0..9')
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(points, directions=NEIGHBORS):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _halo(points):
    return {(r + dr, c + dc) for r, c in points for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _turn(cells, k):
    """Rotate colored offsets about (0,0), preserving the pivot cell."""
    result = []
    for r, c, color in cells:
        for _ in range(k % 4):
            r, c = -c, r
        result.append((r, c, color))
    return tuple(sorted(result))


def _decode(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) * 2 > a.size, 'background must occupy more than half the input')
    components = _components(map(tuple, np.argwhere(a != bg).tolist()))
    anchors, groups = {}, []
    for group in components:
        r, c = next(iter(group))
        if len(group) == 1 and (r in (0, h - 1) or c in (0, w - 1)):
            _require(r not in (0, h - 1) or c not in (0, w - 1), 'corner anchor')
            anchors.setdefault(int(a[r, c]), []).append((r, c))
        else:
            groups.append(group)
    _require(3 <= len(groups) <= 4 and len(anchors) == len(groups), 'need three or four registered pieces')
    guides = {}
    for color, points in anchors.items():
        rows, cols = {p[0] for p in points}, {p[1] for p in points}
        horizontal = len(rows) == 1 and all(c in (0, w - 1) for _, c in points)
        vertical = len(cols) == 1 and all(r in (0, h - 1) for r, _ in points)
        _require(horizontal != vertical, 'inconsistent boundary guide')
        axis = 0 if horizontal else 1
        guides[color] = (axis, points[0][axis])
    pieces, used, bodies = [], set(), set()
    for group in groups:
        _require(len(group) >= 3, 'piece smaller than three cells')
        marks = [p for p in group if int(a[p]) in guides]
        _require(len(marks) == 1, 'piece needs exactly one anchored-color pivot')
        pivot = marks[0]
        marker = int(a[pivot])
        _require(marker not in used, 'one piece per registration identity')
        used.add(marker)
        body = {int(a[p]) for p in group if p != pivot}
        _require(len(body) == 1, 'piece has more than one body color')
        body = next(iter(body))
        _require(body not in bodies, 'carrier colors must be distinct')
        bodies.add(body)
        axis, guide = guides[marker]
        target = list(pivot)
        target[axis] = guide
        cells = tuple(sorted((r - pivot[0], c - pivot[1], int(a[r, c])) for r, c in group))
        pieces.append(dict(marker=marker, body=body, pivot=pivot, target=tuple(target),
                           axis=axis, guide=guide, cells=cells, source=group))
    _require(used == set(guides), 'unused guide')
    area = sum(len(p['cells']) for p in pieces)
    side = isqrt(area)
    _require(side * side == area, 'piece area must be a square')
    return dict(grid=a, background=bg, anchors=anchors, pieces=pieces, side=side)


def _candidates(facts):
    h, w = facts['grid'].shape
    lists = []
    for piece in facts['pieces']:
        tr, tc = piece['target']
        seen, options = set(), []
        for k in range(4):
            cells = tuple(sorted((tr + r, tc + c, color) for r, c, color in _turn(piece['cells'], k)))
            if cells in seen or not all(0 < r < h - 1 and 0 < c < w - 1 for r, c, _ in cells):
                continue
            seen.add(cells)
            points = frozenset((r, c) for r, c, _ in cells)
            options.append(dict(cells=cells, points=points, turn=k))
        lists.append(options)
    return lists


def _assemblies(facts, limit=2):
    """Exhaustive at most 4**4 turns, with bbox and overlap pruning.

    Square position is not supplied. A complete union of s*s cells whose tight
    box is s by s is exactly a filled square. Colored answers, not turn labels,
    are deduplicated, so geometric symmetry introduces no false ambiguity.
    """
    candidates = _candidates(facts)
    order = sorted(range(len(candidates)), key=lambda i: (len(candidates[i]), -len(facts['pieces'][i]['cells'])))
    side = facts['side']
    placed, answers, seen = [None] * len(order), [], set()

    def search(depth, occupied):
        if len(answers) >= limit:
            return
        if depth == len(order):
            rows, cols = zip(*occupied)
            if max(rows) - min(rows) + 1 != side or max(cols) - min(cols) + 1 != side:
                return
            colored = tuple(sorted(cell for option in placed for cell in option['cells']))
            if colored not in seen:
                seen.add(colored)
                answers.append(tuple(placed))
            return
        i = order[depth]
        for option in candidates[i]:
            if occupied & option['points']:
                continue
            union = occupied | option['points']
            rows, cols = zip(*union)
            if max(rows) - min(rows) >= side or max(cols) - min(cols) >= side:
                continue
            placed[i] = option
            search(depth + 1, union)
        placed[i] = None

    search(0, frozenset())
    return answers


def _render(facts, assembly):
    out = np.full_like(facts['grid'], facts['background'])
    for color, points in facts['anchors'].items():
        for p in points:
            out[p] = color
    for option in assembly:
        for r, c, color in option['cells']:
            out[r, c] = color
    return out


def _turn_sets(facts, assembly):
    possible = []
    for piece, option in zip(facts['pieces'], assembly):
        tr, tc = piece['target']
        offsets = tuple(sorted((r - tr, c - tc, color) for r, c, color in option['cells']))
        possible.append({k for k in range(4) if _turn(piece['cells'], k) == offsets})
    return possible


def _in_square_counts(facts, assembly):
    points = set().union(*(option['points'] for option in assembly))
    rows, cols = zip(*points)
    top, left, side = min(rows), min(cols), facts['side']
    return tuple(sum(all(top <= r < top + side and left <= c < left + side
                         for r, c in option['points']) for option in options)
                 for options in _candidates(facts))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer pivots, guides, square side and unique assembly from the input only."""
    facts = _decode(input_grid)
    answers = _assemblies(facts)
    _require(len(answers) == 1, 'expected exactly one colored registered square')
    return _render(facts, answers[0])


def _connected(points, directions=NEIGHBORS):
    return len(_components(points, directions)) == 1


def _partition(rng, side, count, style, diagonal):
    """Procedural regions, including unequal growth, stepped cuts and pockets."""
    if style == 'pockets':
        labels = np.zeros((side, side), np.int8)
        for label in range(1, count):
            for _ in range(30):
                height = int(rng.integers(1, max(3, side // 2 + 1)))
                width = int(rng.integers(1, max(3, side // 2 + 1)))
                r = int(rng.integers(side - height + 1))
                c = int(rng.integers(side - width + 1))
                mask = {(rr, cc) for rr in range(r, r + height) for cc in range(c, c + width)}
                if height > 1 and width > 1 and rng.random() < .6:
                    mask.remove((r, c))
                if len(mask) < 3 or any(labels[p] != 0 for p in mask):
                    continue
                remainder = set(map(tuple, np.argwhere(labels == 0).tolist())) - mask
                if len(remainder) < 3 or not _connected(remainder):
                    continue
                for p in mask:
                    labels[p] = label
                break
            else:
                return None
        return labels
    if style == 'cuts':
        rectangles = [(0, 0, side, side)]
        while len(rectangles) < count:
            choices = []
            for i, (r, c, height, width) in enumerate(rectangles):
                choices += [(i, 0, k) for k in range(1, height) if min(k, height - k) * width >= 3]
                choices += [(i, 1, k) for k in range(1, width) if min(k, width - k) * height >= 3]
            if not choices:
                return None
            i, axis, k = choices[int(rng.integers(len(choices)))]
            r, c, height, width = rectangles.pop(i)
            rectangles.extend([(r, c, k, width), (r + k, c, height - k, width)] if axis == 0 else
                              [(r, c, height, k), (r, c + k, height, width - k)])
        labels = np.empty((side, side), np.int8)
        for i, (r, c, height, width) in enumerate(rectangles):
            labels[r:r + height, c:c + width] = i
        for _ in range(side * 2):
            r, c = map(int, rng.integers(side, size=2))
            dr, dc = EDGE[int(rng.integers(4))]
            q = r + dr, c + dc
            if not (0 <= q[0] < side and 0 <= q[1] < side):
                continue
            a, b = int(labels[r, c]), int(labels[q])
            if a == b or np.sum(labels == a) <= 3:
                continue
            labels[r, c] = b
            if not _connected(map(tuple, np.argwhere(labels == a).tolist())):
                labels[r, c] = a
        return labels
    directions = NEIGHBORS if diagonal else EDGE
    labels = np.full((side, side), -1, np.int8)
    seeds = list(map(int, rng.choice(side * side, size=count, replace=False)))
    weights = np.exp(rng.uniform(-1.1, 1.1, count))
    frontiers, sizes = [set() for _ in seeds], [1] * count
    for i, p in enumerate(seeds):
        labels.flat[p] = i
    def add_frontier(i, p):
        r, c = divmod(p, side)
        for dr, dc in directions:
            q = r + dr, c + dc
            if 0 <= q[0] < side and 0 <= q[1] < side and labels[q] < 0:
                frontiers[i].add(q[0] * side + q[1])
    for i, p in enumerate(seeds):
        add_frontier(i, p)
    for _ in range(side * side - count):
        eligible = [i for i in range(count) if frontiers[i]]
        i = min(eligible, key=lambda j: sizes[j] / weights[j] + float(rng.random()) * .5)
        options = sorted(frontiers[i])
        if diagonal and rng.random() < .75:
            edge_options = [p for p in options if any(0 <= (r := p // side + dr) < side and
                           0 <= (c := p % side + dc) < side and labels[r, c] == i for dr, dc in EDGE)]
            if edge_options:
                options = edge_options
        p = options[int(rng.integers(len(options)))]
        labels.flat[p] = i
        sizes[i] += 1
        for frontier in frontiers:
            frontier.discard(p)
        add_frontier(i, p)
    return labels


def _anchor_options(rng, axis, pivot, shape):
    h, w = shape
    ends = ((pivot[0], 0), (pivot[0], w - 1)) if axis == 0 else ((0, pivot[1]), (h - 1, pivot[1]))
    options = [(ends[0],), (ends[1],), ends]
    # Bilateral markings are optional, not correlated with turns or piece area.
    first = int(rng.choice(3, p=[.38, .38, .24]))
    return [options[first]] + [options[i] for i in rng.permutation(3) if i != first]


def _scatter(rng, targets, pivots, axes, anchors, shape, bg):
    """Scatter actual masks by sliding and turning around their rail pivots."""
    h, w = shape
    all_options = []
    for cells, pivot, axis in zip(targets, pivots, axes):
        options, seen = [], set()
        for k in rng.permutation(4):
            rotated = _turn(cells, int(k))
            if rotated in seen:
                continue
            seen.add(rotated)
            low = min(p[axis] for p in rotated)
            high = max(p[axis] for p in rotated)
            for t in range(-low, (h, w)[axis] - high):
                if t == pivot[axis]:
                    continue  # every generated registration actually changes position
                origin = list(pivot)
                origin[axis] = t
                colored = tuple((origin[0] + r, origin[1] + c, color) for r, c, color in rotated)
                if not all(0 <= r < h and 0 <= c < w for r, c, _ in colored):
                    continue
                points = frozenset((r, c) for r, c, _ in colored)
                options.append((points, colored, rotated != tuple(cells)))
        rng.shuffle(options)
        all_options.append(options)
    if any(not options for options in all_options):
        return None
    order = sorted(range(len(targets)), key=lambda i: (len(all_options[i]), -len(targets[i])))
    selected, budget = [None] * len(targets), [600]
    anchor_points = set(anchors)
    def search(depth, blocked, changed):
        budget[0] -= 1
        if budget[0] < 0:
            return False
        if depth == len(order):
            return changed >= 2
        i = order[depth]
        for option in all_options[i]:
            if option[0] & blocked:
                continue
            selected[i] = option
            if search(depth + 1, blocked | _halo(option[0]), changed + int(option[2])):
                return True
        selected[i] = None
        return False
    if not search(0, _halo(anchor_points), 0):
        return None
    inp = np.full(shape, bg, np.int8)
    for p, color in anchors.items():
        inp[p] = color
    for _, colored, _ in selected:
        for r, c, color in colored:
            inp[r, c] = color
    return inp


def _make_pair(rng, side, count, palette, axis_mode, style, require_diagonal=False, diagonal=False):
    bg = int(palette[0])
    for _ in range(160):
        labels = _partition(rng, side, count, 'growth' if require_diagonal else style, diagonal)
        if labels is None:
            continue
        groups = [set(map(tuple, np.argwhere(labels == i).tolist())) for i in range(count)]
        if min(map(len, groups)) < 3 or any(len(_components(g, EDGE)) > 3 for g in groups):
            continue
        has_diagonal = any(not _connected(g, EDGE) for g in groups)
        if require_diagonal and not has_diagonal:
            continue
        if has_diagonal and not diagonal:
            continue
        h = int(rng.integers(max(14, side + 7), min(30, 2 * side + 10) + 1))
        w = int(rng.integers(max(14, side + 7), min(30, 2 * side + 10) + 1))
        top = int(rng.integers(2, h - side - 1))
        left = int(rng.integers(2, w - side - 1))
        axes = [int(axis_mode)] * count if axis_mode in (0, 1) else [0, 1] + list(map(int, rng.integers(2, size=count - 2)))
        rng.shuffle(axes)
        local_pivots = []
        for group in groups:
            choices = sorted(group)
            # Both interior and eccentric registrations occur, independently of shape style.
            if rng.random() < .65:
                rr, cc = zip(*group)
                eccentric = [p for p in choices if 2 * p[0] != min(rr) + max(rr) or 2 * p[1] != min(cc) + max(cc)]
                choices = eccentric or choices
            local_pivots.append(choices[int(rng.integers(len(choices)))])
        pivots = [(r + top, c + left) for r, c in local_pivots]
        anchors = {}
        possible = True
        for i, (axis, pivot) in enumerate(zip(axes, pivots)):
            for points in _anchor_options(rng, axis, pivot, (h, w)):
                if not set(points) & _halo(anchors):
                    anchors.update({p: int(palette[1 + i]) for p in points})
                    break
            else:
                possible = False
                break
        if not possible:
            continue
        out = np.full((h, w), bg, np.int8)
        targets = []
        for i, (group, pivot) in enumerate(zip(groups, local_pivots)):
            body, marker = int(palette[5 + i]), int(palette[1 + i])
            cells = tuple(sorted((r - pivot[0], c - pivot[1], marker if (r, c) == pivot else body) for r, c in group))
            targets.append(cells)
            for r, c in group:
                out[top + r, left + c] = marker if (r, c) == pivot else body
        for p, color in anchors.items():
            out[p] = color
        inp = _scatter(rng, targets, pivots, axes, anchors, (h, w), bg)
        if inp is None:
            continue
        facts = _decode(inp)
        answers = _assemblies(facts)
        if len(answers) != 1 or not np.array_equal(_render(facts, answers[0]), out):
            continue
        if set.intersection(*_turn_sets(facts, answers[0])):
            continue  # independently chosen piece turns must be observable
        if max(_in_square_counts(facts, answers[0])) < 2:
            continue  # interlocking must matter even if the square box is known
        return {'input': inp, 'output': out}
    raise RuntimeError('bounded registered-jigsaw construction exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, 'num_train', 2, 10)
    ntest = _count(num_test, 'num_test', 1, 4)
    if ntrain <= ntest:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    stable_palette = bool(rng.random() < .25)
    scale_shift = bool(rng.random() < .65)
    diagonal_mode = bool(rng.random() < .45)
    base_axis = int(rng.integers(2))
    styles = list(rng.permutation(['growth', 'cuts', 'pockets']))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(ntrain + ntest):
        test = i >= ntrain
        count = 3 if i == 0 else 4 if i == 1 else int(rng.integers(3, 5))
        side = int(rng.integers(8, 11) if test and scale_shift else rng.integers(4, 8) if scale_shift else rng.integers(5, 9))
        axis_mode = 'mixed' if scale_shift or test else (base_axis if i % 2 == 0 else 1 - base_axis)
        colors = palette if stable_palette else rng.permutation(10)
        must_diagonal = diagonal_mode and (i == 0 or test and rng.random() < .45)
        for _ in range(12):
            pair = _make_pair(rng, side, count, colors, axis_mode, styles[i % 3], must_diagonal, diagonal_mode)
            signature = _key(pair['input'])
            if signature not in seen:
                break
        else:
            raise RuntimeError('could not produce a distinct input')
        seen.add(signature)
        puzzle['test' if test else 'train'].append(pair)
    return puzzle


def _pair_facts(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    facts = _decode(inp)
    _require(inp.shape == out.shape, 'canvas shape changed')
    answers = _assemblies(facts)
    _require(len(answers) == 1, 'nonunique or impossible registered square')
    answer = answers[0]
    _require(np.array_equal(_render(facts, answer), out), 'incorrect registered assembly')
    _require(not np.array_equal(inp, out), 'unchanged scene')
    bg, pieces = facts['background'], facts['pieces']
    # Every color, including every fixed anchor, is conserved.
    for color in range(10):
        _require(np.count_nonzero(inp == color) == np.count_nonzero(out == color), 'color area not preserved')
    occupied = set().union(*(option['points'] for option in answer))
    rr, cc = zip(*occupied)
    top, left = min(rr), min(cc)
    side = facts['side']
    _require(len(occupied) == side * side, 'not an exact cover')
    _require(all(0 < r < inp.shape[0] - 1 and 0 < c < inp.shape[1] - 1 for r, c in occupied), 'square touches outer boundary')
    changed, eccentric, displacements, degrees = 0, 0, [], []
    for piece, option in zip(pieces, answer):
        source_offsets = tuple(sorted(piece['cells']))
        target_offsets = tuple(sorted((r - piece['target'][0], c - piece['target'][1], v) for r, c, v in option['cells']))
        changed += source_offsets != target_offsets
        axis = piece['axis']
        _require(piece['target'][1 - axis] == piece['pivot'][1 - axis], 'along-guide coordinate changed')
        _require(piece['target'][axis] == piece['guide'], 'pivot missed guide')
        displacements.append(piece['target'][axis] - piece['pivot'][axis])
        degrees.append(sum((dr, dc) in {(r, c) for r, c, _ in piece['cells']} for dr, dc in EDGE))
        rvalues, cvalues = zip(*[(r, c) for r, c, _ in piece['cells']])
        eccentric += min(rvalues) + max(rvalues) != 0 or min(cvalues) + max(cvalues) != 0
        _require(int(out[piece['target']]) == piece['marker'], 'pivot not retained')
    # Family-specific ablations: omit jigsaw rotations, or omit rail registration.
    _require(changed >= 1, 'projection without jigsaw already suffices')
    # Rotating the entire completed square is a valid *free* rigid packing with
    # identical area and colors, but must violate at least one registered pin.
    square = out[top:top + side, left:left + side]
    alternate = np.rot90(square)
    violates_registration = any(int(alternate[p['target'][0] - top, p['target'][1] - left]) != p['marker'] for p in pieces)
    _require(violates_registration, 'registration does not distinguish free square rotation')
    diagonal = sum(not _connected(p['source'], EDGE) for p in pieces)
    holes = 0
    for p in pieces:
        coords = p['source']
        r0, r1 = min(r for r, _ in coords) - 1, max(r for r, _ in coords) + 1
        c0, c1 = min(c for _, c in coords) - 1, max(c for _, c in coords) + 1
        empty = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)} - coords
        holes += len(_components(empty, EDGE)) > 1
    return dict(side=side, count=len(pieces), axes=tuple(sorted(set(p['axis'] for p in pieces))),
                rotated=changed, diagonal=diagonal, holes=holes,
                areas=tuple(sorted(len(p['cells']) for p in pieces)),
                bilateral=sum(len(v) == 2 for v in facts['anchors'].values()),
                boundary_pieces=sum(any(r in (0, inp.shape[0] - 1) or c in (0, inp.shape[1] - 1) for r, c in p['source']) for p in pieces),
                displacements=tuple(displacements), degrees=tuple(degrees), background=bg,
                eccentric=eccentric, common_turn=bool(set.intersection(*_turn_sets(facts, answer))),
                shape=inp.shape, square_origin=(top, left),
                individually_possible=tuple(len(x) for x in _candidates(facts)),
                in_square_possible=_in_square_counts(facts, answer))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal pair counts')
    seen, stats = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp = _grid(pair['input'])
            signature = _key(inp)
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            stats[split].append(_pair_facts(pair))
    training_axes = {axis for f in stats['train'] for axis in f['axes']}
    _require(training_axes == {0, 1}, 'demonstrate both guide axes')
    _require({f['count'] for f in stats['train']} == {3, 4}, 'demonstrate use of different piece counts')
    _require(any(not f['common_turn'] for f in stats['train']), 'no independent piece-rotation demonstration')
    _require(any(f['eccentric'] for f in stats['train']), 'center alignment is not distinguished')
    _require(any(max(f['in_square_possible']) > 1 for f in stats['train']), 'no interlocking orientation choice')
    if any(f['diagonal'] for f in stats['test']):
        _require(any(f['diagonal'] for f in stats['train']), 'diagonal attachment lacks demonstration')
    if any(len(f['axes']) == 2 for f in stats['test']) and not any(len(f['axes']) == 2 for f in stats['train']):
        _require(all(len(f['axes']) == 1 for f in stats['train']) and training_axes == {0, 1}, 'mixed-axis shift lacks both axis demonstrations')


def _fingerprint(puzzle):
    return tuple((_key(_grid(p['input'])), _key(_grid(p['output']))) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed semantic, diversity, metamorphic and negative checks."""
    records, fingerprints = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        records.extend(_pair_facts(p) for split in puzzle.values() for p in split)
    _require(len(fingerprints) == 42, 'repeated puzzles across seeds')
    _require(len({f['shape'] for f in records}) >= 40, 'canvas diversity')
    _require({4, 5, 6, 7, 8, 9, 10} <= {f['side'] for f in records}, 'scale diversity')
    _require({f['count'] for f in records} == {3, 4}, 'piece count diversity')
    _require({f['axes'] for f in records} == {(0,), (1,), (0, 1)}, 'rail composition diversity')
    _require(sum(f['diagonal'] > 0 for f in records) >= 10, 'diagonal geometry missing')
    _require(sum(f['holes'] > 0 for f in records) >= 5, 'wrapping geometry missing')
    _require(sum(f['bilateral'] > 0 for f in records) >= 10, 'bilateral markings missing')
    _require(sum(f['boundary_pieces'] > 0 for f in records) >= 5, 'boundary-touching complete pieces missing')
    _require(max(max(f['areas']) / min(f['areas']) for f in records) >= 5, 'unequal areas missing')
    _require(len({f['areas'] for f in records}) >= 70, 'area structures too repetitive')
    _require({1, 2, 3, 4} <= {d for f in records for d in f['degrees']}, 'pivot position diversity')
    _require(any(d < 0 for f in records for d in f['displacements']) and any(d > 0 for f in records for d in f['displacements']), 'movement signs')
    _require({f['background'] for f in records} == set(range(10)), 'background role diversity')
    _require(all(max(f['in_square_possible']) > 1 for f in records), 'shape interlocking lacks evidence')
    base = generate_puzzle(2718, 3, 1)
    _require(_fingerprint(base) == _fingerprint(generate_puzzle(2718, 3, 1)), 'nondeterministic generation')
    # Palette permutations and whole-scene rotations preserve the rule exactly.
    colors = np.random.default_rng(34).permutation(10)
    for k in (1, 2, 3):
        q = {s: [{role: colors[np.rot90(value, k)].astype(np.int8) for role, value in p.items()} for p in base[s]] for s in base}
        validate_generated_puzzle(q)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(900 + 10 * a + b, a, b)
            _require(len(p['train']) == a and len(p['test']) == b, 'count ignored')
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{role: value.copy() for role, value in p.items()} for p in base[s]] for s in base}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted answer accepted')
    # Move a complete boundary registration to a different legal guide. The
    # original colored square must cease to be the answer, even though the
    # supplied pieces and the free jigsaw problem are unchanged.
    pair = base['train'][0]
    facts = _decode(pair['input'])
    first = facts['pieces'][0]
    _require(first['pivot'] != first['target'], 'A-ablation witness missing')
    moved = False
    for offset in (-2, 2, -3, 3):
        altered = pair['input'].copy()
        old = facts['anchors'][first['marker']]
        new = [tuple(p[j] + (offset if j == first['axis'] else 0) for j in (0, 1)) for p in old]
        if not all(0 <= p[0] < altered.shape[0] and 0 <= p[1] < altered.shape[1] for p in new):
            continue
        for p in old:
            altered[p] = facts['background']
        if any(altered[p] != facts['background'] for p in new):
            continue
        for p in new:
            altered[p] = first['marker']
        try:
            altered_facts = _decode(altered)
        except AssertionError:
            continue
        alternatives = _assemblies(altered_facts)
        _require(not any(np.array_equal(_render(altered_facts, a)[1:-1, 1:-1],
                                        pair['output'][1:-1, 1:-1]) for a in alternatives),
                 'changing a guide left the old interior assembly valid')
        moved = True
        break
    _require(moved, 'could not construct guide intervention')
    _require(any(f['rotated'] >= 2 for f in records), 'B-ablation witness missing')
    return {'seeds': 42, 'pairs': len(records), 'distinct_shapes': len({f['shape'] for f in records}),
            'distinct_area_vectors': len({f['areas'] for f in records})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
