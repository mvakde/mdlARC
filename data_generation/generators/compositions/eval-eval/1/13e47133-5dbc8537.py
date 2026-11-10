"""Hidden-seam contour mosaics: keyed rectangles and checkpoint-constrained packing.

Only visible input information is used by solve. No parent module is imported.
"""
from collections import deque
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        cells = []
        while q:
            y, x = q.popleft()
            cells.append((y, x))
            for dy, dx in DIR4:
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w and mask[yy, xx]
                        and not seen[yy, xx]):
                    seen[yy, xx] = True
                    q.append((yy, xx))
        result.append(cells)
    return result


def _bounds(cells):
    yy, xx = zip(*cells)
    return min(yy), min(xx), max(yy) + 1, max(xx) + 1


def _nonrectangular(mask):
    return any(len(cells) != (b - a) * (d - c)
               for cells in _components(mask)
               for a, c, b, d in [_bounds(cells)])


def _rectangle_depth(shape):
    y, x = np.indices(shape)
    h, w = shape
    return np.minimum.reduce((y, x, h - 1 - y, w - 1 - x))


def _erosion_depth(mask):
    """General eight-neighbor depth, also used for semantic witness checks."""
    depth = np.full(mask.shape, -1, dtype=int)
    current = mask.copy()
    h, w = mask.shape
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        inner = current.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                inner &= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        depth[current & ~inner] = level
        current = inner
        level += 1
    return depth


def _decode_tile(raw, neutral):
    """Decode all compatible inward corner diagonals; require one color period."""
    h, w = raw.shape
    depth = _rectangle_depth(raw.shape)
    cues = np.argwhere(raw != neutral)
    if not len(cues):
        period = (neutral,)
        directions = ()
    else:
        length = max(int(depth[y, x]) for y, x in cues) + 1
        candidates = []
        for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            y0, x0 = (0 if dy == 1 else h - 1), (0 if dx == 1 else w - 1)
            if length > (min(h, w) + 1) // 2:
                continue
            if not all((int(y), int(x)) ==
                       (y0 + int(depth[y, x]) * dy, x0 + int(depth[y, x]) * dx)
                       for y, x in cues):
                continue
            p = tuple(int(raw[y0 + i * dy, x0 + i * dx]) for i in range(length))
            candidates.append((p, dy, dx))
        periods = {p for p, _, _ in candidates}
        if len(periods) != 1:
            raise ValueError("Rectangle does not encode one inward-diagonal period")
        period = next(iter(periods))
        directions = tuple((dy, dx) for _, dy, dx in candidates)
    expanded = np.asarray(period, dtype=np.int8)[depth % len(period)]
    return dict(raw=raw.copy(), pattern=expanded, period=period,
                directions=directions, depth=depth)


def _parse(input_grid):
    grid = _grid(input_grid)
    separators = []
    for axis in (0, 1):
        for cut in range(1, grid.shape[axis] - 1):
            line = grid[cut, :] if axis == 0 else grid[:, cut]
            color = int(line[0])
            if np.all(line == color) and np.count_nonzero(grid == color) == len(line):
                separators.append((axis, cut, color))
    if len(separators) != 1:
        raise ValueError("Input needs one exclusive, full-span separator line")
    axis, cut, separator = separators[0]
    panels = ([grid[:cut], grid[cut + 1:]] if axis == 0
              else [grid[:, :cut], grid[:, cut + 1:]])
    background = int(grid[0, 0])
    if background == separator or any(not all(int(a[y, x]) == background
            for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1))) for a in panels):
        raise ValueError("Both panels must have the same background at their corners")
    counts = np.bincount(grid.ravel(), minlength=10)
    eligible = [c for c in range(10) if c not in (background, separator)]
    eligible.sort(key=lambda c: (-int(counts[c]), c))
    neutral = eligible[0]
    if counts[neutral] <= counts[eligible[1]]:
        raise ValueError("Neutral must be the unique most frequent non-background role")
    answers = []
    for tray_index in (0, 1):
        tray, board = panels[tray_index], panels[1 - tray_index]
        mask = board != background
        if not _nonrectangular(mask):
            continue
        tiles = []
        try:
            for cells in _components(tray != background):
                y, x, yy, xx = _bounds(cells)
                if len(cells) != (yy - y) * (xx - x):
                    raise ValueError("Tray pieces must be filled rectangles")
                tiles.append(_decode_tile(tray[y:yy, x:xx], neutral))
        except ValueError:
            continue
        if not tiles or any(not np.any(t['raw'] == neutral) for t in tiles):
            continue
        side = (('bottom', 'top') if axis == 0 else ('right', 'left'))[tray_index]
        answers.append(dict(grid=grid, board=board, tray=tray, mask=mask,
                            background=background, neutral=neutral, separator=separator,
                            side=side, tiles=tiles, axis=axis, cut=cut))
    if len(answers) != 1:
        raise ValueError("Input does not uniquely identify tray and destination")
    return answers[0]


def _pack(mask, board, tiles, neutral, *, respect_pegs=True, max_nodes=50000):
    """Exhaustive finite-inventory exact cover, stopping after two pixel answers.

    Distinct tilings and exchanged identical copies may have one rendered answer.
    A budget exhaustion raises, so it never masquerades as uniqueness.
    """
    coords = np.argwhere(mask)
    n = len(coords)
    ids = np.full(mask.shape, -1, dtype=int)
    ids[mask] = np.arange(n)
    groups, lookup = [], {}
    for tile in tiles:
        pattern = tile['pattern'] if isinstance(tile, dict) else tile
        k = _key(pattern)
        if k not in lookup:
            lookup[k] = len(groups)
            groups.append(dict(pattern=pattern, count=0))
        groups[lookup[k]]['count'] += 1
    placements, by_cell = [], [[] for _ in range(n)]
    for g, group in enumerate(groups):
        pattern = group['pattern']
        h, w = pattern.shape
        for y in range(mask.shape[0] - h + 1):
            for x in range(mask.shape[1] - w + 1):
                if not np.all(mask[y:y + h, x:x + w]):
                    continue
                patch = board[y:y + h, x:x + w]
                if respect_pegs and np.any((patch != neutral) & (patch != pattern)):
                    continue
                indices = tuple(int(v) for v in ids[y:y + h, x:x + w].flat)
                bits = sum(1 << i for i in indices)
                pi = len(placements)
                placements.append((g, bits, indices, tuple(int(v) for v in pattern.flat), y, x))
                for i in indices:
                    by_cell[i].append(pi)
    full = (1 << n) - 1
    inventory = [g['count'] for g in groups]
    start_inventory = tuple(inventory)
    rendered = [-1] * n
    outcomes, dead, path = {}, set(), []
    nodes = 0

    def search(covered):
        nonlocal nodes
        if len(outcomes) == 2:
            return True
        nodes += 1
        if nodes > max_nodes:
            raise ValueError("Exact-cover search budget exceeded")
        if covered == full:
            image = tuple(rendered)
            used = tuple(a - b for a, b in zip(start_inventory, inventory))
            if image not in outcomes:
                outcomes[image] = dict(placements=tuple(path), used=used,
                                      min_used=sum(used), max_used=sum(used))
            else:
                rec = outcomes[image]
                rec['min_used'] = min(rec['min_used'], sum(used))
                rec['max_used'] = max(rec['max_used'], sum(used))
            return True
        state = covered, tuple(inventory)
        if state in dead:
            return False
        if sum(g['pattern'].size * q for g, q in zip(groups, inventory)) < n - covered.bit_count():
            dead.add(state)
            return False
        best = None
        for i in range(n):
            if covered & (1 << i):
                continue
            options = [pi for pi in by_cell[i] if inventory[placements[pi][0]]
                       and not covered & placements[pi][1]]
            if not options:
                dead.add(state)
                return False
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        found = False
        for pi in best:
            g, bits, indices, colors, y, x = placements[pi]
            inventory[g] -= 1
            for i, color in zip(indices, colors):
                rendered[i] = color
            path.append((g, y, x))
            found |= search(covered | bits)
            path.pop()
            for i in indices:
                rendered[i] = -1
            inventory[g] += 1
            if len(outcomes) == 2:
                break
        if not found:
            dead.add(state)
        return found

    search(0)
    return outcomes, groups, nodes


def _analyze(input_grid):
    a = _parse(input_grid)
    outcomes, groups, nodes = _pack(a['mask'], a['board'], a['tiles'], a['neutral'])
    if len(outcomes) != 1:
        raise ValueError("Destination must have exactly one rendered completion")
    rendering, record = next(iter(outcomes.items()))
    out = a['board'].copy()
    out[a['mask']] = rendering
    a.update(output=out, groups=groups, record=record, nodes=nodes)
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _analyze(input_grid)['output'].copy()


def _make_tile(shape, period, neutral, direction):
    raw = np.full(shape, neutral, dtype=np.int8)
    dy, dx = direction
    y, x = (0 if dy == 1 else shape[0] - 1), (0 if dx == 1 else shape[1] - 1)
    for i, color in enumerate(period):
        raw[y + i * dy, x + i * dx] = color
    return _decode_tile(raw, neutral)


def _place_destination(rng, shape, tiles):
    """Grow arbitrary partial-edge joins; later pieces may instead form islands."""
    height, width = shape
    occupied = np.zeros(shape, dtype=bool)
    rects = []
    for i, tile in enumerate(tiles):
        h, w = tile['pattern'].shape
        candidates = set()
        if i == 0:
            candidates = {(y, x) for y in range(1, height - h)
                          for x in range(1, width - w)}
        else:
            for y, x, hh, ww in rects:
                for xx in range(x - w + 1, x + ww):
                    candidates.add((y - h, xx))
                    candidates.add((y + hh, xx))
                for yy in range(y - h + 1, y + hh):
                    candidates.add((yy, x - w))
                    candidates.add((yy, x + ww))
            if i >= 2 and rng.random() < .28:
                candidates = {(y, x) for y in range(1, height - h)
                              for x in range(1, width - w)
                              if not occupied[max(0, y - 1):y + h + 1,
                                              max(0, x - 1):x + w + 1].any()}
        legal = [(y, x) for y, x in sorted(candidates)
                 if 1 <= y and 1 <= x and y + h < height and x + w < width
                 and not occupied[y:y + h, x:x + w].any()]
        if not legal:
            return None
        y, x = legal[int(rng.integers(len(legal)))]
        occupied[y:y + h, x:x + w] = True
        rects.append((y, x, h, w))
    if not _nonrectangular(occupied):
        return None
    return occupied, rects


def _scatter(rng, shape, tiles, background):
    height, width = shape
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i]['raw'].size)
    for _ in range(12):
        occupied = np.zeros(shape, dtype=bool)
        positions = {}
        for i in order:
            h, w = tiles[i]['raw'].shape
            legal = [(y, x) for y in range(1, height - h) for x in range(1, width - w)
                     if not occupied[y - 1:y + h + 1, x - 1:x + w + 1].any()]
            if not legal:
                break
            y, x = legal[int(rng.integers(len(legal)))]
            positions[i] = (y, x)
            occupied[y:y + h, x:x + w] = True
        if len(positions) == len(tiles):
            tray = np.full(shape, background, dtype=np.int8)
            for i, (y, x) in positions.items():
                h, w = tiles[i]['raw'].shape
                tray[y:y + h, x:x + w] = tiles[i]['raw']
            return tray
    return None


def _whole_region_witness(a):
    """No contour palette on the merged region can reproduce two colors at a depth."""
    for cells in _components(a['mask']):
        component = np.zeros(a['mask'].shape, dtype=bool)
        for y, x in cells:
            component[y, x] = True
        depth = _erosion_depth(component)
        for d in range(int(depth.max()) + 1):
            if len(np.unique(a['output'][component & (depth == d)])) > 1:
                return True
    return False


def _choose_pegs(rng, mask, output, tiles, neutral, rects, force_ambiguity):
    board = np.full(mask.shape, neutral, dtype=np.int8)
    expected = tuple(int(v) for v in output[mask])
    if force_ambiguity:
        outcomes, _, _ = _pack(mask, board, tiles, neutral, respect_pegs=False)
        if len(outcomes) != 2:
            return None
    # Start with an interior sample, away from the copied diagonal when feasible.
    initial = []
    for tile, (y, x, h, w) in zip(tiles, rects):
        for dy, dx in np.argwhere((tile['depth'] > 0) &
                                  (tile['pattern'] != neutral) & (tile['raw'] == neutral)):
            initial.append((y + int(dy), x + int(dx)))
    if not initial:
        initial = [tuple(map(int, p)) for p in np.argwhere(mask & (output != neutral))]
    if not initial:
        return None
    y, x = initial[int(rng.integers(len(initial)))]
    board[y, x] = output[y, x]
    coords = np.argwhere(mask)
    for _ in range(11):
        outcomes, _, _ = _pack(mask, board, tiles, neutral)
        if len(outcomes) == 1 and expected in outcomes:
            return board
        alternatives = [v for v in outcomes if v != expected]
        if not alternatives:
            return None
        alt = alternatives[int(rng.integers(len(alternatives)))]
        choices = [i for i, (want, other) in enumerate(zip(expected, alt))
                   if want != other and want != neutral]
        if not choices:
            # Neutral cells are unobserved, not negative clues. Do not invent a marker.
            return None
        i = choices[int(rng.integers(len(choices)))]
        y, x = map(int, coords[i])
        board[y, x] = output[y, x]
    return None


def _build_pair(rng, profile, side, roles):
    background, neutral, separator = roles
    palette = [c for c in range(10) if c not in roles]
    for _ in range(180):
        height = int(rng.integers(21 if profile == 1 else 18, 31 if profile == 3 else 30))
        bw = int(rng.integers(11, 16))
        tw = int(rng.integers(11, 30 - bw))
        count = int(rng.integers(3, 7) if profile == 3 else rng.integers(2, 5))
        rng.shuffle(palette)
        a, b, c = palette[:3]
        selected = []
        shared_shape = (int(rng.integers(7, 10)), int(rng.integers(7, 10))) if profile == 1 else (
            int(rng.integers(5, 8)), int(rng.integers(5, 8)))
        for i in range(count):
            if profile in (0, 1) and i < 2:
                shape = shared_shape
                period = ((a, b) if i == 0 else (c,)) if profile == 0 else (a, neutral, b)
            else:
                shape = (int(rng.integers(3, 9)), int(rng.integers(3, 9)))
                cap = min(3, (min(shape) + 1) // 2)
                length = int(rng.integers(1, cap + 1))
                period = tuple(int(v) for v in rng.choice(palette, length, replace=False))
                if length >= 2 and rng.random() < .35:
                    values = list(period)
                    values[int(rng.integers(length - 1))] = neutral
                    period = tuple(values)
                if i >= 2 and rng.random() < .10:
                    period = (neutral,)
            direction = ((-1, int(rng.choice((-1, 1)))) if profile == 3 and i == 0
                         else (1 if profile in (0, 1) else int(rng.choice((-1, 1))),
                               int(rng.choice((-1, 1)))))
            selected.append(_make_tile(shape, period, neutral, direction))
        if sum(t['raw'].size for t in selected) > .67 * height * min(bw, tw):
            continue
        layout = _place_destination(rng, (height, bw), selected)
        if layout is None:
            continue
        mask, rects = layout
        output = np.full(mask.shape, background, dtype=np.int8)
        for tile, (y, x, h, w) in zip(selected, rects):
            output[y:y + h, x:x + w] = tile['pattern']
        supplied = list(selected)
        # A same-sized, locally fitting alternative is meaningful only because pegs
        # test its synthesized rings. It can be smaller OR larger than other tiles.
        surplus_count = 1 if profile == 0 else int(rng.choice((0, 0, 1, 2)))
        for j in range(surplus_count):
            donor = selected[j % len(selected)]
            length = max(2, len(donor['period']))
            length = min(length, (min(donor['raw'].shape) + 1) // 2)
            p = tuple(int(v) for v in rng.choice(palette, length, replace=False))
            if profile == 0:
                p = (a, c)
            supplied.append(_make_tile(donor['raw'].shape, p, neutral, (1, 1)))
        tray = _scatter(rng, (height, tw), supplied, background)
        if tray is None:
            continue
        try:
            board = _choose_pegs(rng, mask, output, supplied, neutral, rects, profile == 0)
        except ValueError:
            continue
        if board is None:
            continue
        board[~mask] = background
        wall = np.full((height, 1), separator, dtype=np.int8)
        if side in ('left', 'top'):
            inp = np.concatenate((board, wall, tray), axis=1)
        else:
            inp = np.concatenate((tray, wall, board), axis=1)
        if side in ('top', 'bottom'):
            inp, output = inp.T, output.T
        try:
            info = _analyze(inp)
        except ValueError:
            continue
        if not np.array_equal(info['output'], output):
            continue
        if profile == 0 and (not _whole_region_witness(info)
                             or info['record']['min_used'] >= len(supplied)):
            continue
        if profile == 1 and not any(q >= 2 for q in info['record']['used']):
            continue
        return dict(input=inp.copy(), output=output.copy())
    raise RuntimeError("Bounded search could not construct a contour mosaic")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    sides = [str(s) for s in rng.permutation(('left', 'right', 'top', 'bottom'))]
    regime = int(rng.integers(3))
    color_order = [int(c) for c in rng.permutation(10)]
    puzzle = {'train': [], 'test': []}
    seen = set()
    for i in range(num_train + num_test):
        training = i < num_train
        if regime == 0:
            side = sides[0]
        elif regime == 1:
            side = sides[i % 2]
        else:
            side = sides[i % 2] if training else sides[2 + (i - num_train) % 2]
        roles = tuple(color_order[(i + j * 3) % 10] for j in range(3))
        profile = (0 if i == 0 else 1 if i == 1 else 2) if training else 3
        for _ in range(10):
            pair = _build_pair(rng, profile, side, roles)
            k = _key(pair['input'])
            if k not in seen:
                seen.add(k)
                puzzle['train' if training else 'test'].append(pair)
                break
        else:
            raise RuntimeError("Could not construct distinct inputs")
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), []
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        inp, out = _grid(pair['input']), _grid(pair['output'])
        assert _key(inp) not in seen
        seen.add(_key(inp))
        a = _analyze(inp)
        assert np.array_equal(out, a['output']), "Incorrect contour packing output"
        assert np.all(out[~a['mask']] == a['background'])
        assert np.array_equal(out[a['mask'] & (a['board'] != a['neutral'])],
                              a['board'][a['mask'] & (a['board'] != a['neutral'])])
        assert a['separator'] not in out
        assert set(np.unique(out)).issubset(set(np.unique(inp)))
        assert 1 <= np.count_nonzero(a['mask'] & (a['board'] != a['neutral'])) <= 11
        assert _nonrectangular(a['mask'])
        for t in a['tiles']:
            assert np.array_equal(t['depth'], _erosion_depth(np.ones(t['raw'].shape, dtype=bool)))
        assert all(q <= g['count'] for q, g in zip(a['record']['used'], a['groups']))
        analyses.append(a)
    training = analyses[:nt]
    periods = [t['period'] for a in training for t in a['tiles']]
    assert {1, 2, 3}.issubset({len(p) for p in periods})
    assert any(a['neutral'] in t['period'][1:-1] for a in training for t in a['tiles'])
    assert any(len(t['period']) >= 2 and t['depth'].max() >= len(t['period'])
               for a in training for t in a['tiles'])
    assert any(_whole_region_witness(a) for a in training)
    assert any(any(q >= 2 for q in a['record']['used']) for a in training)
    assert any(a['record']['min_used'] < len(a['tiles']) for a in training)
    assert len({(a['background'], a['neutral'], a['separator']) for a in analyses}) >= 2
    # The first demonstration actually needs color compatibility, not just sizes.
    a = training[0]
    unobserved, _, _ = _pack(a['mask'], a['board'], a['tiles'], a['neutral'], respect_pegs=False)
    assert len(unobserved) == 2, "Geometry alone should admit distinct contour images"
    # A literal copy of sparse source keys cannot render the demonstrated image.
    sparse = [t['raw'] for t in a['tiles']]
    literal, _, _ = _pack(a['mask'], a['board'], sparse, a['neutral'])
    assert tuple(int(v) for v in a['output'][a['mask']]) not in literal


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    shapes, geometry, roles, dimensions, counts, directions = (set() for _ in range(6))
    signatures = set()
    repeats = unmarked = leading = surplus = 0
    sides = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                geometry.add(_key(a['mask']))
                roles.add((a['background'], a['neutral'], a['separator']))
                sides.add(a['side'])
                counts.add(a['record']['min_used'])
                repeats += any(q >= 2 for q in a['record']['used'])
                surplus += a['record']['min_used'] < len(a['tiles'])
                for tile in a['tiles']:
                    dimensions.add(tile['raw'].shape)
                    directions.update(tile['directions'])
                    unmarked += tile['period'] == (a['neutral'],)
                    leading += len(tile['period']) > 1 and tile['period'][0] == a['neutral']
    assert len(signatures) == 42 and len(shapes) >= 60 and len(geometry) >= 100
    assert len(roles) >= 60 and len(dimensions) >= 25 and len(counts) >= 4
    assert sides == {'left', 'right', 'top', 'bottom'}
    assert directions == {(1, 1), (1, -1), (-1, 1), (-1, -1)}
    assert repeats >= 42 and surplus >= 42 and unmarked > 0 and leading > 0
    assert _signature(generate_puzzle(123, 2, 1)) == _signature(generate_puzzle(123, 2, 1))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(901 + nt * 5 + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
            validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    mask = np.ones((3, 6), dtype=bool)
    board = np.full(mask.shape, 1, dtype=np.int8)
    tile = np.full((3, 3), 2, dtype=np.int8)
    assert not _pack(mask, board, [tile], 1)[0]
    assert len(_pack(mask, board, [tile, tile], 1)[0]) == 1
    assert len(_pack(mask, board, [tile, np.full((3, 3), 3)], 1)[0]) == 2
    # The returned-data validator must reject a changed answer background.
    original = generate_puzzle(7, 2, 1)
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in original[s]]
              for s in original}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (ValueError, AssertionError):
        return
    raise AssertionError("Changed output background accepted")
