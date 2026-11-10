"""Mirror relay printing: the reflected union of legend-selected ink seeds.

Only NumPy and the Python standard library are used. Nothing is read from disk.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'invalid color')
    return a.astype(np.int64, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple((split, _key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
        groups.append(cells)
    return groups


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _tick_positions(axis_sum):
    return (axis_sum // 2,) if axis_sum % 2 == 0 else (axis_sum // 2, axis_sum // 2 + 1)


def _border(shape):
    a = np.zeros(shape, dtype=bool)
    a[[0, -1], :] = True
    a[:, [0, -1]] = True
    return a


def _find_board(a):
    candidates = []
    for tick in np.unique(a):
        cells = np.argwhere(a == tick)
        if len(cells) not in (4, 6, 8):
            continue
        r, e, c, d = _box(cells)
        h, w = e - r, d - c
        if min(h, w) < 9:
            continue
        tile = a[r:e, c:d]
        edge = _border(tile.shape)
        if np.any((tile == tick) & ~edge):
            continue
        horizontal = tuple(map(int, np.flatnonzero(tile[:, 0] == tick)))
        vertical = tuple(map(int, np.flatnonzero(tile[0] == tick)))
        if (horizontal != tuple(map(int, np.flatnonzero(tile[:, -1] == tick)))
                or vertical != tuple(map(int, np.flatnonzero(tile[-1] == tick)))):
            continue
        if any(len(p) not in (1, 2) or min(p) <= 0 or max(p) >= length - 1
               or (len(p) == 2 and p[1] != p[0] + 1)
               for p, length in ((horizontal, h), (vertical, w))):
            continue
        others = np.unique(tile[edge & (tile != tick)])
        if len(others) != 1 or int(others[0]) == int(tick):
            continue
        # The ticks occur exclusively on this complete frame.
        if len(cells) != 2 * len(horizontal) + 2 * len(vertical):
            continue
        sums = tuple(2 * p[0] + len(p) - 1 for p in (horizontal, vertical))
        candidates.append(((r, e, c, d), int(others[0]), int(tick), sums))
    _require(len(candidates) == 1, 'need one tick-marked frame')
    return candidates[0]


def _dimension_partners(start, n, length, axis_sum):
    other = axis_sum - start - n + 1
    if 2 <= other <= length - n - 2:
        return tuple(sorted({start, other}))
    # A fringe tile's reflection is wholly outside the interior. Partial
    # footprints and tiles adjacent to the frame are excluded by the family.
    _require(other + n <= 1 or other >= length - 1, 'partially clipped reflected footprint')
    return (start,)


def _slot_orbit(r, c, n, shape, axes):
    rows = _dimension_partners(r, n, shape[0], axes[0])
    cols = _dimension_partners(c, n, shape[1], axes[1])
    return tuple((y, x) for y in rows for x in cols)


def _separated(positions, n):
    return all(abs(r - y) >= n + 1 or abs(c - x) >= n + 1
               for i, (r, c) in enumerate(positions) for y, x in positions[i + 1:])


def _parse(a):
    a = _grid(a)
    frame_box, frame, tick, axes = _find_board(a)
    r0, e0, c0, d0 = frame_box
    board = a[r0:e0, c0:d0]
    corners = [board[1, 1], board[1, -2], board[-2, 1], board[-2, -2]]
    _require(len(set(map(int, corners))) == 1, 'background corners must agree')
    bg = int(corners[0])
    _require(bg not in (frame, tick), 'structural colors must differ')
    slots, sizes = {}, set()
    for cells in _components(board[1:-1, 1:-1] != bg):
        r, e, c, d = _box(cells)
        tile = board[r + 1:e + 1, c + 1:d + 1]
        n = e - r
        _require(n == d - c and 3 <= n <= 5 and len(cells) == n * n,
                 'slots must be solid size-matched squares')
        _require(len(np.unique(tile)) == 1, 'slot must have one color')
        _require(r + 1 >= 2 and c + 1 >= 2 and e + 1 <= board.shape[0] - 2
                 and d + 1 <= board.shape[1] - 2, 'slots need a gap to the frame')
        color = int(tile[0, 0])
        _require(color != tick, 'tick color cannot be a slot')
        slots[r + 1, c + 1] = color
        sizes.add(n)
    _require(len(sizes) == 1 and slots, 'need one common slot size')
    n = sizes.pop()
    _require(_separated(list(slots), n), 'slots need a blank-cell gap')
    sources = set(slots.values()) - {frame}
    _require(1 <= len(sources) <= 3, 'need one to three source categories')
    exterior = a != bg
    exterior[r0:e0, c0:d0] = False
    keys, removed, key_boxes = {}, np.zeros(a.shape, dtype=bool), []
    for cells in _components(exterior):
        values = [int(a[r, c]) for r, c in cells]
        unique, counts = np.unique(values, return_counts=True)
        if len(unique) == 1:
            continue
        r, e, c, d = _box(cells)
        _require((e - r, d - c) == (2, 2) and len(cells) == 4
                 and set(map(int, counts)) == {1, 3}, 'invalid color key')
        source = int(unique[np.argmax(counts)])
        target = int(unique[np.argmin(counts)])
        _require(source in sources and source not in keys
                 and target not in (bg, frame, tick), 'invalid direct mapping')
        keys[source] = target
        removed[r:e, c:d] = True
        key_boxes.append((r, e, c, d))
    _require(set(keys) == sources, 'each source needs one key')
    stencil_cells = exterior & ~removed
    _require(set(map(int, np.unique(a[stencil_cells]))) == sources, 'unexpected exterior content')
    stencils, boxes = {}, []
    for source in sorted(sources):
        cells = np.argwhere(stencil_cells & (a == source))
        r, e, c, d = _box(cells)
        _require((e - r, d - c) == (n, n), 'stencil must expose its whole square bounds')
        _require(bool(np.all(np.isin(a[r:e, c:d], [bg, source]))), 'mixed stencil colors')
        _require(e <= r0 or r >= e0 or d <= c0 or c >= d0, 'stencil intersects board')
        stencils[source] = (a[r:e, c:d] == source).copy()
        boxes.append((r, e, c, d))
    clue_boxes = boxes + key_boxes
    for i, (r, e, c, d) in enumerate(clue_boxes):
        _require(e < r0 or r > e0 or d < c0 or c > d0,
                 'clues need a blank-cell gap to the frame')
        for rr, ee, cc, dd in clue_boxes[i + 1:]:
            _require(e < rr or ee < r or d < cc or dd < c,
                     'complete clue squares need a blank-cell gap')
    # Validate orbit membership and unanimous surviving identities, not nearest
    # source, legend order, or a common upper-left master position.
    orbits, seen = [], set()
    for p in sorted(slots):
        if p in seen:
            continue
        orbit = _slot_orbit(*p, n, board.shape, axes)
        _require(set(orbit) <= set(slots), 'missing reflected destination slot')
        _require(_separated(list(orbit), n), 'an orbit has overlapping footprints')
        labels = {slots[q] for q in orbit if slots[q] != frame}
        _require(len(labels) == 1, 'orbit needs unanimous surviving source labels')
        source = next(iter(labels))
        roots = tuple(q for q in orbit if slots[q] != frame)
        orbits.append((orbit, roots, source))
        seen.update(orbit)
    return dict(grid=a, board=board, frame_box=frame_box, frame=frame, tick=tick,
                bg=bg, axes=axes, n=n, slots=slots, stencils=stencils,
                mapping=keys, orbits=orbits, stencil_boxes=boxes)


def _paint_pixels(info, *, reflect=True, solid=False, same_orientation=False):
    """Prediction by pixel coordinates; no generation state is consulted."""
    result = info['board'].copy()
    bg, n = info['bg'], info['n']
    for r, c in info['slots']:
        result[r:r + n, c:c + n] = bg
    h, w = result.shape
    row_sum, col_sum = info['axes']
    for orbit, roots, source in info['orbits']:
        paint = info['mapping'][source]
        mask = np.ones((n, n), dtype=bool) if solid else info['stencils'][source]
        if same_orientation:
            for r, c in orbit:
                result[r:r + n, c:c + n][mask] = paint
            continue
        for r, c in roots:
            for dr, dc in np.argwhere(mask):
                y, x = r + int(dr), c + int(dc)
                rows = {y, row_sum - y} if reflect else {y}
                cols = {x, col_sum - x} if reflect else {x}
                for yy in rows:
                    for xx in cols:
                        if 1 <= yy < h - 1 and 1 <= xx < w - 1:
                            _require(result[yy, xx] in (bg, paint), 'conflicting ink colors')
                            result[yy, xx] = paint
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the marked folds, legend, and seeds, then take reflected ink union."""
    return _paint_pixels(_parse(input_grid))


def _render_tiles(board, slots, stencils, mapping, bg, frame, axes, n):
    """Construction/validation through tile transformations, separate from solve."""
    out = board.copy()
    for r, c in slots:
        out[r:r + n, c:c + n] = bg
    for (r, c), source in slots.items():
        if source == frame:
            continue
        for flip_r in (False, True):
            for flip_c in (False, True):
                y = axes[0] - r - n + 1 if flip_r else r
                x = axes[1] - c - n + 1 if flip_c else c
                if not (2 <= y <= board.shape[0] - n - 2 and 2 <= x <= board.shape[1] - n - 2):
                    continue
                mask = stencils[source]
                if flip_r:
                    mask = mask[::-1, :]
                if flip_c:
                    mask = mask[:, ::-1]
                out[y:y + n, x:x + n][mask] = mapping[source]
    return out


def _asymmetric(mask):
    return all(not np.array_equal(mask, other)
               for other in (mask[::-1], mask[:, ::-1], mask[::-1, ::-1]))


def _stencil(rng, n, informative=False):
    if informative:
        for _ in range(100):
            mask = rng.random((n, n)) < rng.uniform(.24, .60)
            if (all(edge.any() for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]))
                    and _asymmetric(mask) and not mask.all()):
                return mask
        # All sizes 3..5: touches four bounds and differs under both reflections.
        mask = np.zeros((n, n), dtype=bool)
        mask[0, :] = mask[:, 0] = True
        mask[-1, 1] = True
        return mask
    kind = int(rng.integers(8))
    y, x = np.indices((n, n))
    if kind == 0:
        mask = rng.random((n, n)) < rng.uniform(.15, .85)
    elif kind == 1:
        mask = (y == 0) | (x == 0) | (y == n - 1) | (x == n - 1)
    elif kind == 2:
        mask = (x == n // 2) | (y == n // 2)
    elif kind == 3:
        mask = x == y
    elif kind == 4:
        mask = (x + y) % 2 == 0
    elif kind == 5:
        mask = y % 2 == 0
    elif kind == 6:
        mask = rng.random((n, n)) < rng.uniform(.75, .96)
    else:
        mask = np.ones((n, n), dtype=bool)
    for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
        if not edge.any():
            edge[int(rng.integers(n))] = True
    return np.rot90(mask, int(rng.integers(4))).copy()


def _orbit_candidates(shape, axes, n):
    result, seen = [], set()
    for r in range(2, shape[0] - n - 1):
        for c in range(2, shape[1] - n - 1):
            try:
                orbit = _slot_orbit(r, c, n, shape, axes)
            except ValueError:
                continue
            if orbit in seen or not _separated(list(orbit), n):
                continue
            seen.add(orbit)
            fringe = any(not (1 <= axes[0] - y - n + 1 and axes[0] - y - n + 1 + n <= shape[0] - 1)
                         or not (1 <= axes[1] - x - n + 1 and axes[1] - x - n + 1 + n <= shape[1] - 1)
                         for y, x in orbit)
            result.append((orbit, fringe))
    return result


def _layout(rng, n, k, fringe_test, attempts=80):
    for _ in range(attempts):
        if fringe_test:
            h = 4 * n + 9  # n=3 or 4; room for a core quartet and a fringe.
            w = int(rng.integers(3 * n + 6 if k == 3 else 2 * n + 5, 23))
            a = h - 1 + int(rng.choice([-1, 1])) * (n + 3)
        else:
            h = int(rng.integers(3 * n + 6, min(26, 3 * n + 10)))
            w = int(rng.integers(3 * n + 6 if k == 3 else 2 * n + 5, min(25, 3 * n + 10)))
            fixed_row = int(rng.integers(n + 3, h - 2 * n - 2))
            a = 2 * fixed_row + n - 1
        # For k=3 this central column guarantees a third available orbit.
        if w >= 3 * n + 6:
            fixed_col = int(rng.integers(n + 3, w - 2 * n - 2))
            b = 2 * fixed_col + n - 1
        else:
            b = w - 1 + int(rng.integers(-1, 2))
        shape, axes = (h, w), (a, b)
        # A legend shelf must fit on at least one side.
        if min(h, w) + n + 4 > 30:
            continue
        available = _orbit_candidates(shape, axes, n)
        chosen, places = [], []
        requirements = [(4, False), (2, fringe_test)]
        for wanted, fringe in requirements:
            options = [o for o, f in available if len(o) == wanted and f == fringe
                       and _separated(places + list(o), n)]
            if not options:
                break
            o = options[int(rng.integers(len(options)))]
            chosen.append(o)
            places.extend(o)
        if len(chosen) != 2:
            continue
        desired = int(rng.integers(max(2, k), 6))
        while len(chosen) < desired:
            options = [o for o, f in available if o not in chosen and _separated(places + list(o), n)]
            if not options:
                break
            o = options[int(rng.integers(len(options)))]
            chosen.append(o)
            places.extend(o)
        if len(chosen) >= k:
            return shape, axes, chosen
    # Same rule and palette; a guaranteed 3×3 arrangement is a termination guard.
    # No fringe shift is silently dropped: its fallback has a separate fringe row.
    h = 4 * n + 9 if fringe_test else 3 * n + 6
    w = 3 * n + 6
    center_r = n + 3
    if fringe_test:
        center_r = (h - 1 + n + 3 - (n - 1)) // 2
    center_c = n + 3
    axes = (2 * center_r + n - 1, 2 * center_c + n - 1)
    quartet = tuple((r, c) for r in (center_r - n - 1, center_r + n + 1)
                    for c in (2, 2 * n + 4))
    second = tuple((2 if fringe_test else center_r, c) for c in (2, 2 * n + 4))
    third = ((center_r, center_c),)
    return (h, w), axes, [quartet, second, third][:max(2, k)]


def _mapping(rng, sources, external, kind):
    order = list(map(int, rng.permutation(sources)))
    if kind == 'cycle' and len(sources) > 1:
        return {s: order[(i + 1) % len(order)] for i, s in enumerate(order)}
    if kind == 'chain' and len(sources) > 1:
        return {s: order[i + 1] if i + 1 < len(order) else external[0]
                for i, s in enumerate(order)}
    if kind == 'many':
        target = int(rng.choice(external))
        return {s: target for s in sources}
    targets = list(map(int, rng.choice(external, len(sources), replace=False)))
    return dict(zip(sources, targets))


def _starts(rng, length, count, size):
    needed = count * size + count - 1
    _require(length >= needed, 'legend shelf too small')
    gaps = rng.multinomial(length - needed, np.full(count + 1, 1 / (count + 1)))
    p, result = int(gaps[0]), []
    for i in range(count):
        result.append(p)
        p += size + 1 + int(gaps[i + 1])
    return result


def _scene(rng, board, axes, bg, frame, tick, stencils, mapping):
    h, w = board.shape
    n = next(iter(stencils.values())).shape[0]
    k = len(stencils)
    sides = [s for s in ('above', 'below', 'left', 'right')
             if (h if s in ('above', 'below') else w) + n + 4 <= 30]
    side = str(rng.choice(sides))
    shelf = n + 4
    cross = max(w if side in ('above', 'below') else h, k * (n + 1) - 1)
    shape = (h + shelf, cross) if side in ('above', 'below') else (cross, w + shelf)
    grid = np.full(shape, bg, dtype=np.int64)
    offset = int(rng.integers(cross - (w if side in ('above', 'below') else h) + 1))
    if side == 'above':
        r0, c0, sr, kr = shelf, offset, 0, n + 1
    elif side == 'below':
        r0, c0, sr, kr = 0, offset, h + 1, h + n + 2
    elif side == 'left':
        r0, c0, sc, kc = offset, shelf, 0, n + 1
    else:
        r0, c0, sc, kc = offset, 0, w + 1, w + n + 2
    grid[r0:r0 + h, c0:c0 + w] = board
    for source, position in zip(rng.permutation(list(stencils)), _starts(rng, cross, k, n)):
        r, c = (sr, position) if side in ('above', 'below') else (position, sc)
        grid[r:r + n, c:c + n][stencils[int(source)]] = int(source)
    for source, position in zip(rng.permutation(list(stencils)), _starts(rng, cross, k, 2)):
        source = int(source)
        r, c = (kr, position) if side in ('above', 'below') else (position, kc)
        key = np.full((2, 2), source, dtype=np.int64)
        corner = int(rng.integers(4))
        key[corner // 2, corner % 2] = mapping[source]
        grid[r:r + 2, c:c + 2] = key
    # Optional surrounding space is genuine blank canvas and affects neither folds
    # nor stencil origins. It is only added when the 30×30 bound permits it.
    slack = (30 - grid.shape[0], 30 - grid.shape[1])
    padding = []
    for amount in slack:
        before = int(rng.integers(min(2, amount) + 1))
        after = int(rng.integers(min(2, amount - before) + 1))
        padding.append((before, after))
    return np.pad(grid, padding, constant_values=bg)


def _pair(rng, shift, training, index):
    if shift == 'scale':
        n = 3 if training else int(rng.choice([4, 5]))
    else:
        n = int(rng.choice([3, 4])) if index < 2 or shift == 'fringe' else int(rng.choice([3, 4, 5]))
    k = int(rng.integers(2 if training and index == 1 else 1, (2 if n == 5 else 3) + 1))
    palette = list(map(int, rng.permutation(10)))
    bg, frame, tick = palette[:3]
    sources, external = palette[3:3 + k], palette[3 + k:]
    shape, axes, orbits = _layout(rng, n, k, not training and shift == 'fringe')
    stencils = {}
    for j, source in enumerate(sources):
        if j and not (training and index < 2) and rng.random() < .18:
            stencils[source] = stencils[sources[int(rng.integers(j))]].copy()
        else:
            stencils[source] = _stencil(rng, n, informative=training and index < 2)
    kind = 'chain' if training and index == 1 else str(rng.choice(['disjoint', 'chain', 'cycle', 'many']))
    mapping = _mapping(rng, sources, external, kind)
    board = np.full(shape, bg, dtype=np.int64)
    board[_border(shape)] = frame
    for r in _tick_positions(axes[0]):
        board[r, 0] = board[r, -1] = tick
    for c in _tick_positions(axes[1]):
        board[0, c] = board[-1, c] = tick
    labels = sources + list(map(int, rng.choice(sources, max(0, len(orbits) - k))))
    slots = {}
    for j, (orbit, source) in enumerate(zip(orbits, labels)):
        if j == 0 and training and index == 0:
            count = 1
        elif j == 0 and training and index == 1:
            count = 2
        elif j == 0 and not training and shift == 'multiplicity':
            count = 3
        elif j == 1 and training and index == 0:
            count = 1  # teaches the fixed-fold self-union, without a second seed
        else:
            count = int(rng.integers(1, min(len(orbit), 2 if training else 3) + 1))
        roots = {orbit[int(q)] for q in rng.permutation(len(orbit))[:count]}
        for r, c in orbit:
            color = source if (r, c) in roots else frame
            slots[r, c] = color
            board[r:r + n, c:c + n] = color
    out = _render_tiles(board, slots, stencils, mapping, bg, frame, axes, n)
    inp = _scene(rng, board, axes, bg, frame, tick, stencils, mapping)
    turns = int(rng.integers(4))
    return {'input': np.rot90(inp, turns).copy(), 'output': np.rot90(out, turns).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return distinct examples with one fixed, input-visible family convention."""
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['scale', 'multiplicity', 'fringe']))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(int(count)):
            for _ in range(24):
                pair = _pair(rng, shift, split == 'train', i)
                signature = _key(pair['input'])
                if signature not in seen:
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
            seen.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, output):
    n, axes = info['n'], info['axes']
    evidence = dict(single_quartet=False, multiple_seed_union=False, fixed_fold_union=False,
                    diagonal_only=False, fringe=False, overlapping_map=False)
    for orbit, roots, source in info['orbits']:
        mask = info['stencils'][source]
        evidence['single_quartet'] |= len(orbit) == 4 and len(roots) == 1 and _asymmetric(mask)
        evidence['multiple_seed_union'] |= len(orbit) == 4 and len(roots) > 1 and any(
            np.count_nonzero(output[r:r + n, c:c + n] == info['mapping'][source]) > mask.sum()
            for r, c in orbit)
        if len(roots) == 1:
            r, c = roots[0]
            fixed = axes[0] - r - n + 1 == r or axes[1] - c - n + 1 == c
            evidence['fixed_fold_union'] |= fixed and np.count_nonzero(
                output[r:r + n, c:c + n] == info['mapping'][source]) > mask.sum()
            evidence['diagonal_only'] |= len(orbit) == 4
        for r, c in orbit:
            rr, cc = axes[0] - r - n + 1, axes[1] - c - n + 1
            evidence['fringe'] |= rr + n <= 1 or rr >= output.shape[0] - 1 or cc + n <= 1 or cc >= output.shape[1] - 1
    evidence['overlapping_map'] = any(t in info['mapping'] and info['mapping'][t] != t
                                      for t in info['mapping'].values())
    return {key: bool(value) for key, value in evidence.items()}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'invalid example counts')
    seen, evidence = set(), {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = _key(inp)
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            info = _parse(inp)
            _require(out.shape == info['board'].shape, 'wrong inclusive crop')
            expected = _render_tiles(info['board'], info['slots'], info['stencils'], info['mapping'],
                                     info['bg'], info['frame'], info['axes'], info['n'])
            _require(np.array_equal(out, expected), 'wrong reflected stencil union')
            _require(np.array_equal(solve(inp), out), 'pixel and tile reconstructions disagree')
            unchanged = np.ones(out.shape, dtype=bool)
            for r, c in info['slots']:
                unchanged[r:r + info['n'], c:c + info['n']] = False
            _require(np.array_equal(out[unchanged], info['board'][unchanged]), 'frame or background changed')
            _require(not np.any(out[1:-1, 1:-1] == info['frame']), 'unprinted vacancy remains')
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), 'invented paint color')
            if split == 'train':
                for key, value in _features(info, out).items():
                    evidence[key] = evidence.get(key, False) or bool(value)
    for key in ('single_quartet', 'multiple_seed_union', 'fixed_fold_union', 'diagonal_only', 'overlapping_map'):
        _require(evidence.get(key, False), f'training does not demonstrate {key}')
    # Counterfactuals test meaningful inheritance, not merely output nonidentity.
    _require(any(not np.array_equal(_paint_pixels(_parse(p['input']), reflect=False), p['output'])
                 for p in puzzle['train']), 'reflection contribution is dispensable')
    _require(any(not np.array_equal(_paint_pixels(_parse(p['input']), solid=True), p['output'])
                 for p in puzzle['train']), 'stencil contribution is dispensable')
    _require(any(not np.array_equal(_paint_pixels(_parse(p['input']), same_orientation=True), p['output'])
                 for p in puzzle['train']), 'unreflected stencil shortcut is not distinguished')


def self_test():
    """Bounded 42-seed semantic/diversity study and focused mutations.

    Returns compact descriptive metrics; does not write files or use the network.
    """
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    shapes, sizes, categories, orbit_sizes, multiplicities, offsets, tick_parities = (set() for _ in range(7))
    bg_colors, frame_colors, tick_colors, mask_signatures = (set() for _ in range(4))
    disconnected = fringe_tests = shared_geometry = unions = 0
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                sizes.add(info['n'])
                categories.add(len(info['stencils']))
                bg_colors.add(info['bg'])
                frame_colors.add(info['frame'])
                tick_colors.add(info['tick'])
                offsets.add(tuple(a - d + 1 for a, d in zip(info['axes'], info['board'].shape)))
                tick_parities.add(tuple(a % 2 for a in info['axes']))
                orbit_sizes.update(len(o) for o, roots, s in info['orbits'])
                multiplicities.update(len(roots) for o, roots, s in info['orbits'])
                masks = list(info['stencils'].values())
                mask_signatures.update(_key(m) for m in masks)
                disconnected += sum(len(_components(m)) > 1 for m in masks)
                shared_geometry += len({_key(m) for m in masks}) < len(masks)
                features = _features(info, pair['output'])
                fringe_tests += split == 'test' and features['fringe']
                unions += features['multiple_seed_union']
        # Whole-scene rotations and arbitrary palette permutations commute with
        # the rule, including the tick representation and direct key direction.
        pair = puzzle['test'][0]
        for turns in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair['input'], turns)), np.rot90(pair['output'], turns))
        perm = np.random.default_rng(1000 + len(puzzle['train']) + int(pair['input'].sum())).permutation(10)
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    assert sizes == {3, 4, 5} and categories == {1, 2, 3}
    assert orbit_sizes == {1, 2, 4} and {1, 2, 3} <= multiplicities
    assert len(shapes) >= 35 and len(offsets) >= 15 and len(tick_parities) >= 3
    assert min(map(len, (bg_colors, frame_colors, tick_colors))) >= 8
    assert len(mask_signatures) >= 90 and disconnected >= 15 and shared_geometry >= 3
    assert fringe_tests >= 3 and unions >= 42
    # Exercise both normal and fringe termination layouts explicitly, even if
    # random sampling never reaches them. Their semantics use the same solver.
    guard_rng = np.random.default_rng(765)
    for n, k, fringe in ((3, 1, False), (4, 3, False), (5, 2, False), (5, 3, False),
                          (3, 3, True), (4, 2, True)):
        shape, axes, orbits = _layout(guard_rng, n, k, fringe, attempts=0)
        stencils = {s: _stencil(guard_rng, n, informative=True) for s in range(3, 3 + k)}
        mapping = {s: s + 3 for s in stencils}
        board = np.zeros(shape, dtype=np.int64)
        board[_border(shape)] = 1
        for r in _tick_positions(axes[0]):
            board[r, 0] = board[r, -1] = 2
        for c in _tick_positions(axes[1]):
            board[0, c] = board[-1, c] = 2
        slots = {}
        for i, orbit in enumerate(orbits):
            source = 3 + i % k
            for j, (r, c) in enumerate(orbit):
                # Also exercise the legal all-four-seeds quartet, which the
                # normal sampler intentionally leaves outside its count range.
                is_seed = j == 0 or (n == 5 and k == 3 and i == 0)
                slots[r, c] = source if is_seed else 1
                board[r:r + n, c:c + n] = slots[r, c]
        expected = _render_tiles(board, slots, stencils, mapping, 0, 1, axes, n)
        inp = _scene(guard_rng, board, axes, 0, 1, 2, stencils, mapping)
        assert np.array_equal(solve(inp), expected)
        assert not fringe or _features(_parse(inp), expected)['fringe']
    # A seed's position controls orientation, and adding it combines support.
    pair = puzzles[0]['train'][0]
    info = _parse(pair['input'])
    orbit, roots, source = next((o, roots, s) for o, roots, s in info['orbits']
                               if len(o) == 4 and len(roots) == 1
                               and _asymmetric(info['stencils'][s]))
    old = roots[0]
    new = next(q for q in orbit if q[0] == old[0] and q != old)
    r0, _, c0, _ = info['frame_box']
    n = info['n']
    moved = pair['input'].copy()
    moved[r0 + old[0]:r0 + old[0] + n, c0 + old[1]:c0 + old[1] + n] = info['frame']
    moved[r0 + new[0]:r0 + new[0] + n, c0 + new[1]:c0 + new[1] + n] = source
    moved_out = solve(moved)
    assert not np.array_equal(moved_out, pair['output'])
    added = moved.copy()
    added[r0 + old[0]:r0 + old[0] + n, c0 + old[1]:c0 + old[1] + n] = source
    combined = solve(added)
    assert np.array_equal(combined, np.where(pair['output'] != info['bg'], pair['output'], moved_out))
    assert np.count_nonzero(combined == info['mapping'][source]) > np.count_nonzero(
        pair['output'] == info['mapping'][source])
    for train_count in range(2, 11):
        for test_count in range(1, min(4, train_count - 1) + 1):
            p = generate_puzzle(123, train_count, test_count)
            assert len(p['train']) == train_count and len(p['test']) == test_count
    for train_count, test_count in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                                  (True, 1), (4.0, 1), (4, False), (4, None)):
        try:
            generate_puzzle(0, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted illegal counts')
    # Incorrect labels and changed supposedly preserved cells must be rejected.
    for interior in (True, False):
        p = {s: [{k: a.copy() for k, a in q.items()} for q in puzzles[0][s]] for s in ('train', 'test')}
        pair = p['test'][0]
        r, c = (2, 2) if interior else (0, 0)
        pair['output'][r, c] = (int(pair['output'][r, c]) + 1) % 10
        try:
            validate_generated_puzzle(p)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted a corrupt output')
    return dict(seeds=42, distinct_puzzles=42, shape_pairs=len(shapes), stencil_sizes=sorted(sizes),
                categories=sorted(categories), orbit_sizes=sorted(orbit_sizes),
                seed_multiplicities=sorted(multiplicities), axis_offsets=len(offsets),
                tick_parities=len(tick_parities), distinct_stencils=len(mask_signatures),
                disconnected_stencils=disconnected, shared_geometry_pairs=int(shared_geometry),
                fringe_test_pairs=int(fringe_tests), multi_seed_union_pairs=int(unions))
