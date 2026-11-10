"""Single-pulse blueprints: a matched enlarged clue activates miniature regions.

Only standard-library facilities and NumPy are used. The input-only reader does
not consult seeds, constructors, files, training labels, or cached examples.
"""

from collections import deque
import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    array = np.asarray(value)
    _need(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape), "invalid grid dimensions")
    _need(np.issubdtype(array.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((array >= 0) & (array <= 9))), "invalid grid colors")
    return array.astype(np.int8, copy=True)


def _add(point, direction, distance=1):
    return point[0] + direction[0] * distance, point[1] + direction[1] * distance


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue, cells = deque([start]), {start}
        while queue:
            point = queue.popleft()
            for direction in DIRECTIONS:
                other = _add(point, direction)
                if other in unseen:
                    unseen.remove(other)
                    cells.add(other)
                    queue.append(other)
        result.append(cells)
    return result


def _territories(plan, background):
    """Connectedness is measured on logical sites, across the visible gutters."""
    ids = np.full(plan.shape, -1, dtype=np.int16)
    colors, cells = [], []
    for color in sorted(int(v) for v in np.unique(plan) if v != background):
        for region in _components(plan == color):
            label = len(colors)
            colors.append(color)
            cells.append(region)
            for point in region:
                ids[point] = label
    return ids, colors, cells


def _t_cells(center, direction):
    perpendicular = (-direction[1], direction[0])
    return {center, _add(center, direction), _add(center, perpendicular),
            _add(center, perpendicular, -1)}


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _sequence_matches(plan, sequence, axis):
    length = len(sequence)
    result = []
    for r in range(plan.shape[0] - (length - 1 if axis == 0 else 0)):
        for c in range(plan.shape[1] - (length - 1 if axis == 1 else 0)):
            candidate = tuple(int(plan[r + (axis == 0) * k, c + (axis == 1) * k])
                              for k in range(length))
            if candidate == sequence:
                result.append((r, c))
    return result


def _pulse(plan, background, arrows, seed_sites):
    ids, colors, cells = _territories(plan, background)
    active = {int(ids[p]) for p in seed_sites}
    _need(-1 not in active, "the enlarged clue cannot contain blank sites")
    assignments = {}
    for arrow in arrows:
        source, target = int(ids[arrow['source_site']]), int(ids[arrow['target_site']])
        _need(source >= 0 and target >= 0 and source != target, "invalid territory edge")
        if source in active:
            payload = colors[source]
            _need(target not in assignments or assignments[target] == payload,
                  "conflicting active incoming transfers")
            assignments[target] = payload
    final = plan.copy()
    for label, payload in assignments.items():
        final[ids == label] = payload
    return final, dict(ids=ids, colors=colors, regions=cells, active=active, assignments=assignments)


def _render_final(inp, final, background, base, side, stride):
    output = inp.copy()
    for (r, c), color in np.ndenumerate(final):
        y, x = base[0] + r * stride, base[1] + c * stride
        output[y:y + side, x:x + side] = color
    return output


def _recover(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    background = int(np.argmax(counts))
    _need(counts[background] > np.partition(counts, -2)[-2], "background is not uniquely dominant")
    swatches, squares, arrow_shapes = [], [], []
    for cells in _components(inp != background):
        if len(cells) == 1:
            swatches.append(next(iter(cells)))
            continue
        monochrome = len({int(inp[p]) for p in cells}) == 1
        _need(monochrome, "objects must be monochromatic")
        center = next((p for p in sorted(cells)
                       if sum(_add(p, d) in cells for d in DIRECTIONS) == 3), None)
        if len(cells) == 4 and center is not None:
            missing = next(d for d in DIRECTIONS if _add(center, d) not in cells)
            direction = (-missing[0], -missing[1])
            _need(cells == _t_cells(center, direction), "malformed T arrow")
            arrow_shapes.append(dict(center=center, direction=direction, write=cells,
                                     ink=int(inp[center])))
            continue
        rows, cols = zip(*cells)
        r, c, bottom, right = min(rows), min(cols), max(rows) + 1, max(cols) + 1
        side = bottom - r
        _need(side >= 2 and right - c == side and len(cells) == side * side,
              "unrecognized object")
        squares.append((r, c, side, int(inp[r, c])))
    _need(len(swatches) >= 3 and len(squares) >= 2 and arrow_shapes, "missing blueprint, clue, or arrows")
    points = np.array(swatches, dtype=int)
    mini = tuple(int(v) for v in points.min(axis=0))
    end = tuple(int(v) + 1 for v in points.max(axis=0))
    row_gaps = np.diff(np.unique(points[:, 0]))
    col_gaps = np.diff(np.unique(points[:, 1]))
    _need(len(row_gaps) and len(col_gaps), "blueprint needs two dimensions")
    pitch = int(min(row_gaps.min(), col_gaps.min()))
    _need(pitch in (5, 6) and row_gaps.min() == col_gaps.min() == pitch,
          "miniature pitch needs adjacent witnesses in both axes")
    _need(bool(np.all((points - mini) % pitch == 0)), "irregular miniature lattice")
    plan = inp[mini[0]:end[0]:pitch, mini[1]:end[1]:pitch].copy()
    _need(min(plan.shape) >= 2, "blueprint is too small")
    _need(np.count_nonzero(plan != background) == len(swatches), "unexplained swatches")
    inks = {a['ink'] for a in arrow_shapes}
    _need(len(inks) == 1, "arrows must share their local ink")
    ink = next(iter(inks))
    _need(ink not in plan and ink != background, "arrow ink must differ from blueprint colors")
    arrows = []
    for arrow in arrow_shapes:
        source_pixel = _add(arrow['center'], arrow['direction'], -2)
        offsets = (source_pixel[0] - mini[0], source_pixel[1] - mini[1])
        _need(all(v % pitch == 0 for v in offsets), "arrow is not attached to a lattice edge")
        source = tuple(v // pitch for v in offsets)
        target = _add(source, arrow['direction'])
        _need(all(0 <= p[d] < plan.shape[d] for p in (source, target) for d in (0, 1)),
              "arrow points outside the blueprint")
        arrows.append(dict(arrow, source_site=source, target_site=target))
    if len({s[0] for s in squares}) == 1:
        axis = 1
    else:
        _need(len({s[1] for s in squares}) == 1, "enlarged clue is not a straight run")
        axis = 0
    squares.sort(key=lambda s: s[axis])
    side = squares[0][2]
    _need(all(s[2] == side for s in squares), "clue squares have unequal sizes")
    differences = np.diff([s[axis] for s in squares])
    stride = int(differences[0])
    _need(bool(np.all(differences == stride)) and stride > side, "clue spacing is not uniform")
    sequence = tuple(s[3] for s in squares)
    matches = _sequence_matches(plan, sequence, axis)
    _need(len(matches) == 1, "clue colors do not uniquely locate the original miniature")
    match, length = matches[0], len(sequence)
    seed_sites = {(match[0] + (axis == 0) * k, match[1] + (axis == 1) * k)
                  for k in range(length)}
    base = squares[0][0] - match[0] * stride, squares[0][1] - match[1] * stride
    big_end = tuple(base[d] + (plan.shape[d] - 1) * stride + side for d in (0, 1))
    _need(all(0 <= base[d] < big_end[d] <= inp.shape[d] for d in (0, 1)),
          "complete enlarged array does not fit")
    mini_box = (mini[0] - 1, mini[1] - 1, end[0] + 1, end[1] + 1)
    _need(all(0 <= mini_box[d] for d in (0, 1)) and
          mini_box[2] <= inp.shape[0] and mini_box[3] <= inp.shape[1], "miniature border does not fit")
    _need(not _overlap(mini_box, (*base, *big_end)), "representations overlap")
    # Reconstruct every input pixel. This rejects noise, extra clue squares,
    # off-lattice marks, and unexpected nonbackground pixels in blank sites.
    reconstructed = np.full(inp.shape, background, dtype=np.int8)
    for (r, c), color in np.ndenumerate(plan):
        reconstructed[mini[0] + r * pitch, mini[1] + c * pitch] = color
    for arrow in arrows:
        for point in arrow['write']:
            reconstructed[point] = ink
    for (r, c) in seed_sites:
        y, x = base[0] + r * stride, base[1] + c * stride
        reconstructed[y:y + side, x:x + side] = plan[r, c]
    _need(np.array_equal(reconstructed, inp), "unexplained input pixels")
    final, pulse = _pulse(plan, background, arrows, seed_sites)
    expected = _render_final(inp, final, background, base, side, stride)
    meta = dict(background=background, ink=ink, pitch=pitch, plan=plan, final=final,
                mini=mini, base=base, side=side, stride=stride, axis=axis, match=match,
                length=length, arrows=arrows, seed_sites=seed_sites, **pulse)
    return expected, meta


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Match original colors, activate original components, transfer one pulse."""
    return _recover(input_grid)[0]


def _features(meta):
    ids, colors, regions = meta['ids'], meta['colors'], meta['regions']
    active, assignments, arrows = meta['active'], meta['assignments'], meta['arrows']
    edges = [(int(ids[a['source_site']]), int(ids[a['target_site']])) for a in arrows]
    final_colors = [assignments.get(i, color) for i, color in enumerate(colors)]
    active_edges = [(s, t) for s, t in edges if s in active]
    dormant_edges = [(s, t) for s, t in edges if s not in active]
    site_only = meta['plan'].copy()
    single_target = meta['plan'].copy()
    for arrow in arrows:
        s, t = int(ids[arrow['source_site']]), int(ids[arrow['target_site']])
        if arrow['source_site'] in meta['seed_sites']:
            site_only[ids == t] = colors[s]
        if s in active:
            single_target[arrow['target_site']] = colors[s]
    return dict(
        gate=any(t == u and u not in active and colors[u] != final_colors[v]
                 for s, t in active_edges for u, v in dormant_edges),
        simultaneous=any(t == u and colors[s] != colors[u] and colors[s] != final_colors[v]
                         for s, t in active_edges for u, v in active_edges),
        dormant=any(colors[s] != final_colors[t] for s, t in dormant_edges),
        regional_activation=not np.array_equal(site_only, meta['final']),
        multi_target=not np.array_equal(single_target, meta['final']),
        locality=any(colors[a] == colors[b] and final_colors[a] != final_colors[b]
                     for a in range(len(colors)) for b in range(a)),
        seed_repaint=any(meta['plan'][p] != meta['final'][p] for p in meta['seed_sites']),
        changed=bool(np.any(meta['plan'] != meta['final'])),
        added=any(meta['plan'][p] != meta['background'] and p not in meta['seed_sites']
                  for p in np.ndindex(meta['plan'].shape)),
        arrows=len(arrows), territories=len(colors), active=len(active),
        holes=int(np.count_nonzero(meta['plan'] == meta['background'])),
        plan_shape=meta['plan'].shape, pitch=meta['pitch'], side=meta['side'],
        gutter=meta['stride'] - meta['side'], axis=meta['axis'], match=meta['match'],
        length=meta['length'], mini=meta['mini'], base=meta['base'],
        background=meta['background'], ink=meta['ink'],
        directions=tuple(sorted({DIRECTIONS.index(a['direction']) for a in arrows})),
    )


def _asymmetric(plan):
    alternatives = [np.flip(plan, 0), np.flip(plan, 1), np.flip(plan, (0, 1))]
    if plan.shape[0] == plan.shape[1]:
        alternatives.extend([plan.T, np.rot90(plan), np.rot90(plan, 3)])
    return all(not np.array_equal(plan, alternative) for alternative in alternatives)


def _geometry_options(profile, test):
    options = []
    for h in range(2, 6):
        for w in range(2, 6):
            if profile != 'ordinary' and (min(h, w) < 3 or h * w < 9):
                continue
            if test and min(h, w) < 3:
                continue
            for pitch in (5, 6):
                mini = ((h - 1) * pitch + 3, (w - 1) * pitch + 3)
                for side in range(2, 5):
                    for gutter in range(1, 4):
                        big = ((h - 1) * (side + gutter) + side,
                               (w - 1) * (side + gutter) + side)
                        for separation in (0, 1):
                            required = [max(mini[d], big[d]) for d in (0, 1)]
                            required[separation] = mini[separation] + big[separation] + 1
                            if max(required) <= 30:
                                options.append((h, w, pitch, side, gutter, separation))
    return options


def _place(rng, shape, pitch, side, gutter, separation):
    mini_size = tuple((n - 1) * pitch + 3 for n in shape)
    big_size = tuple((n - 1) * (side + gutter) + side for n in shape)
    other = 1 - separation
    gap = int(rng.integers(1, min(3, 30 - mini_size[separation] - big_size[separation]) + 1))
    required = [max(mini_size[d], big_size[d]) for d in (0, 1)]
    required[separation] = mini_size[separation] + big_size[separation] + gap
    mode = int(rng.integers(5))
    if mode == 0:
        canvas = (30, 30)
    elif mode == 1:
        n = int(rng.integers(max(required), 31))
        canvas = (n, n)
    else:
        canvas = tuple(int(rng.integers(n, min(30, n + 4) + 1)) for n in required)
    board, base = [0, 0], [0, 0]
    padding = canvas[separation] - required[separation]
    start = int(rng.integers(padding + 1))
    if rng.integers(2):
        board[separation] = start
        base[separation] = start + mini_size[separation] + gap
    else:
        base[separation] = start
        board[separation] = start + big_size[separation] + gap
    board[other] = int(rng.integers(canvas[other] - mini_size[other] + 1))
    base[other] = int(rng.integers(canvas[other] - big_size[other] + 1))
    return canvas, (board[0] + 1, board[1] + 1), tuple(base)


def _random_plan(rng, shape, background, palette):
    plan = rng.choice(palette, size=shape).astype(np.int8)
    for _ in range(int(rng.integers(0, shape[0] * shape[1] + 1))):
        p = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
        d = DIRECTIONS[int(rng.integers(4))]
        q = _add(p, d)
        if all(0 <= q[k] < shape[k] for k in (0, 1)):
            plan[p] = plan[q]
    hole_count = int(rng.choice([0, 0, 0, 1, 2, 3]))
    if hole_count:
        plan.flat[rng.choice(plan.size, min(hole_count, plan.size - 3), replace=False)] = background
    if any(np.all(edge == background) for edge in (plan[0], plan[-1], plan[:, 0], plan[:, -1])):
        return None
    occupied = np.argwhere(plan != background)
    if any(not np.any(np.diff(np.unique(occupied[:, d])) == 1) for d in (0, 1)):
        return None
    return plan


def _clues(plan, background, axis, test, profile):
    choices = []
    for length in range(2, plan.shape[axis] + 1):
        for r in range(plan.shape[0] - (length - 1 if axis == 0 else 0)):
            for c in range(plan.shape[1] - (length - 1 if axis == 1 else 0)):
                match = r, c
                if test and not (0 < match[1 - axis] < plan.shape[1 - axis] - 1 and
                                 length < plan.shape[axis]):
                    continue
                sites = {(r + (axis == 0) * k, c + (axis == 1) * k) for k in range(length)}
                sequence = tuple(int(plan[r + (axis == 0) * k, c + (axis == 1) * k])
                                 for k in range(length))
                if background in sequence or _sequence_matches(plan, sequence, axis) != [match]:
                    continue
                if np.count_nonzero(plan != background) <= length:
                    continue
                choices.append((match, length, sites))
    return choices


def _edge_candidates(plan, background, pitch):
    ids, colors, regions = _territories(plan, background)
    candidates = []
    for source in np.ndindex(plan.shape):
        if ids[source] < 0:
            continue
        for direction in DIRECTIONS:
            target = _add(source, direction)
            if not all(0 <= target[k] < plan.shape[k] for k in (0, 1)) or ids[target] < 0:
                continue
            if ids[source] == ids[target]:
                continue
            center = _add((source[0] * pitch, source[1] * pitch), direction, 2)
            candidates.append(dict(source_site=source, target_site=target, direction=direction,
                                   center=center, write=_t_cells(center, direction),
                                   source=int(ids[source]), target=int(ids[target])))
    return candidates, ids, colors, regions


def _select_arrows(rng, plan, background, pitch, sites, profile):
    candidates, ids, colors, regions = _edge_candidates(plan, background, pitch)
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    active = {int(ids[p]) for p in sites}
    selected, blocked, assignments = [], set(), {}

    def legal(a):
        return not (a['write'] & blocked) and (a['target'] not in assignments or
                                              assignments[a['target']] == colors[a['source']])

    def append(a):
        selected.append(a)
        assignments[a['target']] = colors[a['source']]
        blocked.update(a['write'])
        blocked.update(_add(p, d) for p in a['write'] for d in DIRECTIONS)

    if profile in ('gate', 'pulse'):
        found = False
        for first in candidates:
            if first['source'] not in active:
                continue
            if profile == 'gate' and (first['target'] in active or
                                      first['source_site'] in sites or len(regions[first['target']]) < 2):
                continue
            if profile == 'pulse' and first['target'] not in active:
                continue
            for second in candidates:
                if second['source'] != first['target'] or second['target'] == first['source']:
                    continue
                if colors[first['source']] == colors[second['target']]:
                    continue
                halo = first['write'] | {_add(p, d) for p in first['write'] for d in DIRECTIONS}
                if second['write'] & halo:
                    continue
                append(first)
                append(second)
                found = True
                break
            if found:
                break
        if not found:
            return None
    else:
        first = next((a for a in candidates if a['source'] in active), None)
        if first is None:
            return None
        append(first)
    # Optional arrows supply branching, consistent duplicate incoming transfers,
    # cycles on different contacts, and dormant instructions. No new activation.
    desired = int(rng.integers(max(1, len(selected)), min(8, max(3, len(candidates))) + 1))
    for candidate in candidates:
        if len(selected) >= desired:
            break
        if legal(candidate):
            append(candidate)
    final, pulse = _pulse(plan, background, selected, sites)
    meta = dict(plan=plan, final=final, seed_sites=sites, arrows=selected,
                pitch=pitch, side=2, stride=3, axis=1, match=(0, 0), length=2,
                mini=(0, 0), base=(0, 0), background=background, ink=0, **pulse)
    features = _features(meta)
    if not features['changed']:
        return None
    if profile == 'gate' and not all(features[k] for k in ('gate', 'regional_activation', 'multi_target', 'dormant')):
        return None
    if profile == 'pulse' and not all(features[k] for k in ('simultaneous', 'dormant', 'locality', 'seed_repaint')):
        return None
    return selected, final


def _make_pair(rng, axis, profile, test=False, avoid=None):
    options = _geometry_options(profile, test)
    if avoid is not None:
        options = [o for o in options if o[3] != avoid['side'] and (o[0], o[1]) != avoid['plan_shape']]
    for attempt in range(2400):
        h, w, pitch, side, gutter, separation = options[int(rng.integers(len(options)))]
        palette = rng.permutation(10)
        background, ink = int(palette[0]), int(palette[1])
        if avoid is not None and background == avoid['background']:
            continue
        plan = _random_plan(rng, (h, w), background, palette[2:2 + int(rng.integers(3, 7))])
        if plan is None or not _asymmetric(plan):
            continue
        clues = _clues(plan, background, axis, test, profile)
        if not clues:
            continue
        rng.shuffle(clues)
        for match, length, sites in clues[:12]:
            selection = _select_arrows(rng, plan, background, pitch, sites, profile)
            if selection is None:
                continue
            arrows, final = selection
            shape, mini, base = _place(rng, plan.shape, pitch, side, gutter, separation)
            inp = np.full(shape, background, dtype=np.int8)
            for (r, c), color in np.ndenumerate(plan):
                inp[mini[0] + r * pitch, mini[1] + c * pitch] = color
            for arrow in arrows:
                for r, c in arrow['write']:
                    inp[mini[0] + r, mini[1] + c] = ink
            stride = side + gutter
            for r, c in sites:
                y, x = base[0] + r * stride, base[1] + c * stride
                inp[y:y + side, x:x + side] = plan[r, c]
            out = _render_final(inp, final, background, base, side, stride)
            try:
                predicted, meta = _recover(inp)
            except ValueError:
                continue
            if np.array_equal(predicted, out):
                return dict(input=inp, output=out), _features(meta)
    raise RuntimeError("bounded blueprint construction exhausted")


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if not _integer(value) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(3))
    axes = (1,) if mode == 0 else (0,) if mode == 1 else (0, 1)
    train_axes = [axes[i % len(axes)] for i in range(nt)]
    rng.shuffle(train_axes)
    puzzle, seen, previous = dict(train=[], test=[]), set(), None
    for split, count in (('train', nt), ('test', ns)):
        for i in range(count):
            axis = train_axes[i] if split == 'train' else int(rng.choice(axes))
            profile = ('gate' if i == 0 else 'pulse' if i == 1 else 'ordinary') if split == 'train' else 'ordinary'
            for _ in range(24):
                pair, features = _make_pair(rng, axis, profile, test=split == 'test',
                                            avoid=previous if split == 'train' and i == 1 else None)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train' and i == 0:
                        previous = features
                    break
            else:
                raise RuntimeError("duplicate avoidance exhausted")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "wrong puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "wrong pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, "canvas shape changed")
            key = _key(inp)
            _need(key not in seen, "duplicate input")
            seen.add(key)
            expected, meta = _recover(inp)
            _need(np.array_equal(expected, out), "incorrect pulse, enlargement, or preservation")
            f = _features(meta)
            _need(f['changed'] and f['added'], "pair does not exercise transfer and completion")
            if split == 'test':
                cross = meta['match'][1 - meta['axis']]
                _need(0 < cross < meta['plan'].shape[1 - meta['axis']] - 1 and
                      meta['length'] < meta['plan'].shape[meta['axis']],
                      "test must combine an interior line with a partial run")
            # Every missing/blank enlarged site is checked in the complete
            # pixel comparison, including original clue sites that must repaint.
            features[split].append(f)
    training = features['train']
    for mechanism in ('gate', 'simultaneous', 'dormant', 'regional_activation',
                      'multi_target', 'locality', 'seed_repaint'):
        _need(any(f[mechanism] for f in training), "missing teaching witness: " + mechanism)
    _need({f['axis'] for f in features['test']} <= {f['axis'] for f in training}, "unshown clue axis")
    for field in ('plan_shape', 'side', 'background'):
        _need(len({f[field] for f in training}) >= 2, "training must vary " + field)
    return None


def self_test():
    """Bounded semantic, counterfactual, replay, and structural diversity checks."""
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple((_key(pair['input']), _key(pair['output']))
                          for split in ('train', 'test') for pair in puzzle[split])
        signatures.add(signature)
        if seed < 2:
            replay = generate_puzzle(seed)
            _need(signature == tuple((_key(p['input']), _key(p['output']))
                                      for s in ('train', 'test') for p in replay[s]), "replay failed")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                expected, meta = _recover(pair['input'])
                _need(np.array_equal(expected, pair['output']), "solver disagreement")
                f = _features(meta)
                if split == 'train' and f['dormant']:
                    all_sites = set(map(tuple, np.argwhere(meta['plan'] != meta['background'])))
                    ungated, _ = _pulse(meta['plan'], meta['background'], meta['arrows'], all_sites)
                    _need(not np.array_equal(ungated, meta['final']), "ignoring clue activation succeeded")
                if f['regional_activation']:
                    site_only = [a for a in meta['arrows'] if a['source_site'] in meta['seed_sites']]
                    reduced, _ = _pulse(meta['plan'], meta['background'], site_only, meta['seed_sites'])
                    # Another arrow can redundantly make the same assignment;
                    # at least one teaching case below must defeat this shortcut.
                    f['site_only_fails'] = not np.array_equal(reduced, meta['final'])
                if f['gate']:
                    reached = set(meta['active'])
                    while True:
                        extended = reached | {int(meta['ids'][a['target_site']]) for a in meta['arrows']
                                               if int(meta['ids'][a['source_site']]) in reached}
                        if extended == reached:
                            break
                        reached = extended
                    reached_sites = {next(iter(meta['regions'][label])) for label in reached}
                    propagated, _ = _pulse(meta['plan'], meta['background'], meta['arrows'], reached_sites)
                    _need(not np.array_equal(propagated, meta['final']), "activation cascading succeeded")
                # Ablating A still matches and enlarges but copies original fills.
                without_a = _render_final(pair['input'], meta['plan'], meta['background'],
                                           meta['base'], meta['side'], meta['stride'])
                _need(not np.array_equal(without_a, expected), "ignoring transfers succeeded")
                if seed == 0:
                    permutation = np.array([9, 3, 6, 0, 8, 2, 1, 5, 7, 4], dtype=np.int8)
                    _need(np.array_equal(solve(permutation[pair['input']]), permutation[expected]),
                          "literal colors secretly encode roles")
                    _need(np.array_equal(solve(pair['input'].T), expected.T), "transpose convention failed")
                    _need(np.array_equal(solve(np.rot90(pair['input'])), np.rot90(expected)),
                          "rotated geometry failed")
                records.append(dict(seed=seed, split=split, canvas=pair['input'].shape, **f))
    _need(len(signatures) == 42, "seed diversity collapsed")
    for field, minimum in (('canvas', 30), ('plan_shape', 8), ('match', 10),
                           ('mini', 30), ('base', 30), ('arrows', 5), ('territories', 5)):
        _need(len({r[field] for r in records}) >= minimum, "missing variation: " + field)
    for field, required in (('pitch', {5, 6}), ('side', {2, 3, 4}), ('gutter', {1, 2, 3}),
                            ('axis', {0, 1}), ('background', set(range(10))), ('ink', set(range(10)))):
        _need({r[field] for r in records} >= required, "missing roles/geometry: " + field)
    _need({d for r in records for d in r['directions']} == set(range(4)), "missing arrow directions")
    _need(any(r['holes'] == 0 for r in records) and any(r['holes'] >= 2 for r in records), "missing occupancy variation")
    _need(any(r['length'] >= 3 for r in records), "missing longer clue")
    # Concrete shortcut witnesses supplement general constructor/reader agreement.
    _need(any(r.get('site_only_fails', False) for r in records), "seed-site-only activation remains viable")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(81000 + 10 * a + b, a, b)
            _need(len(puzzle['train']) == a and len(puzzle['test']) == b, "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    damaged = generate_puzzle(5, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                canvas_shapes=len({r['canvas'] for r in records}),
                plan_shapes=sorted({r['plan_shape'] for r in records}),
                clue_matches=len({r['match'] for r in records}),
                arrow_counts=sorted({r['arrows'] for r in records}))
