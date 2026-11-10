"""Quota mosaics: counted, color-keyed stamps jointly resolve rectangle packing.

Only NumPy and the standard library are used. No parent generator is imported.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = ([(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c]
             if diagonal else _STEPS)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.append(p)
                    stack.append(p)
        result.append(group)
    return result


def _tight(mask):
    ys, xs = np.where(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def _placements(mask, tiles):
    coords = list(map(tuple, np.argwhere(mask)))
    index = {p: i for i, p in enumerate(coords)}
    by_cell = [[] for _ in coords]
    placements = []
    for kind, tile in enumerate(tiles):
        h, w = tile.shape
        for r in range(mask.shape[0] - h + 1):
            for c in range(mask.shape[1] - w + 1):
                if not mask[r:r + h, c:c + w].all():
                    continue
                indices = tuple(index[(r + y, c + x)] for y in range(h) for x in range(w))
                bits = sum(1 << i for i in indices)
                colors = tuple(int(v) for v in tile.flat)
                p = (kind, bits, indices, colors, r, c)
                placements.append(p)
                for i in indices:
                    by_cell[i].append(p)
    return coords, by_cell, placements


def _cover(mask, tiles, quotas=None, budget=None):
    """Enumerate up to two distinct pixel renderings, with optional exact quotas.

    There is no search budget in public solve. Generation uses explicit budgets
    and rejects unfinished searches. Identical copies have no individual identity.
    """
    coords, by_cell, placements = _placements(mask, tiles)
    n = len(coords)
    if quotas is not None and sum(t.size * q for t, q in zip(tiles, quotas)) != n:
        return {}, {"nodes": 0, "dead_ends": 1, "exhaustive": True}
    full = (1 << n) - 1
    remaining = list(quotas) if quotas is not None else [n // t.size for t in tiles]
    rendered, path, outcomes, dead = [-1] * n, [], {}, set()
    nodes, failures, done = 0, 0, True

    def search(covered):
        nonlocal nodes, failures, done
        if not done or len(outcomes) >= 2:
            return
        nodes += 1
        if budget is not None and nodes > budget:
            done = False
            return
        if covered == full:
            if quotas is None or not any(remaining):
                outcomes.setdefault(tuple(rendered), tuple(path))
                return True
            return False
        state = covered, tuple(remaining)
        if state in dead:
            return False
        options = None
        for i in range(n):
            if covered & (1 << i):
                continue
            legal = [p for p in by_cell[i] if remaining[p[0]] and not (p[1] & covered)]
            if not legal:
                dead.add(state)
                failures += 1
                return False
            if options is None or len(legal) < len(options):
                options = legal
                if len(options) == 1:
                    break
        found = False
        for p in options:
            k, bits, indices, colors, r, c = p
            remaining[k] -= 1
            for i, color in zip(indices, colors):
                rendered[i] = color
            path.append((k, r, c))
            found = bool(search(covered | bits)) or found
            path.pop()
            remaining[k] += 1
            if not done or len(outcomes) >= 2:
                return found
        if not found:
            dead.add(state)
            failures += 1
        return found

    search(0)
    return outcomes, {"nodes": nodes, "dead_ends": failures, "exhaustive": done,
                      "legal_placements": len(placements)}


def _geometry_groups(mask, dimensions, budget=12000):
    """Construction screening: collect positive quota vectors with unique owners.

    This explores geometry, not puzzle labels. Distinct type labels deliberately
    make screening stricter than the eventual pixel uniqueness requirement.
    """
    tiles = [np.full(d, i + 10, dtype=int) for i, d in enumerate(dimensions)]
    coords, by_cell, _ = _placements(mask, tiles)
    n, full = len(coords), (1 << len(coords)) - 1
    counts, owners, path = [0] * len(tiles), [-1] * n, []
    outcomes = defaultdict(dict)
    nodes, done = 0, True

    def search(covered):
        nonlocal nodes, done
        nodes += 1
        if nodes > budget:
            done = False
            return
        if covered == full:
            q = tuple(counts)
            if min(q) > 0 and max(q) <= 12:
                if len(outcomes[q]) < 2:
                    outcomes[q].setdefault(tuple(owners), tuple(path))
            return
        options = None
        for i in range(n):
            if covered & (1 << i):
                continue
            legal = [p for p in by_cell[i] if not p[1] & covered]
            if not legal:
                return
            if options is None or len(legal) < len(options):
                options = legal
        for k, bits, indices, colors, r, c in options:
            counts[k] += 1
            for i in indices:
                owners[i] = k
            path.append((k, r, c))
            search(covered | bits)
            path.pop()
            counts[k] -= 1
            if not done:
                return

    search(0)
    return {q: next(iter(values.values())) for q, values in outcomes.items()
            if len(values) == 1}, done


def _grow_union(rng, dimensions, count):
    """Grow rectangles at partial edge contacts, not a fixed silhouette template."""
    side = min(19, max(9, count + 2))
    mask = np.zeros((side, side), dtype=bool)
    rectangles = []
    for _ in range(count):
        h, w = dimensions[int(rng.integers(len(dimensions)))]
        legal = []
        for attempt in range(65):
            if not rectangles:
                r, c = (side - h) // 2, (side - w) // 2
            else:
                y, x, th, tw = rectangles[int(rng.integers(len(rectangles)))]
                direction = int(rng.integers(4))
                if direction < 2:
                    r = y - h if direction == 0 else y + th
                    c = x + int(rng.integers(1 - w, tw))
                else:
                    c = x - w if direction == 2 else x + tw
                    r = y + int(rng.integers(1 - h, th))
            if (r >= 0 and c >= 0 and r + h <= side and c + w <= side
                    and not mask[r:r + h, c:c + w].any()):
                legal.append((r, c))
        if not legal:
            return None
        r, c = legal[int(rng.integers(len(legal)))]
        rectangles.append((r, c, h, w))
        mask[r:r + h, c:c + w] = True
    return _tight(mask)


def _base_geometry(rng, count, twins=False):
    # Bars with differing orientation admit competing decompositions. Some draws
    # use length three; the remaining degrees of freedom are not template indices.
    for attempt in range(180):
        length = 3 if attempt < 35 and rng.random() < .20 else 2
        dims = [(1, length), (length, 1)]
        mask = _grow_union(rng, dims, count)
        if mask is None or mask.shape[1] > 17 or mask.shape[0] > 25:
            continue
        groups, done = _geometry_groups(mask, dims)
        if done and len(groups) >= (2 if twins else 1):
            # Two distinct quota vectors always imply differing rendered owners.
            candidates = list(groups)
            rng.shuffle(candidates)
            selected = candidates[:2] if twins else candidates[:1]
            # A non-twin must still have a count-blind alternative.
            if not twins:
                markers = [np.full(d, i + 10) for i, d in enumerate(dims)]
                unlimited, stats = _cover(mask, markers, budget=12000)
                if not stats['exhaustive'] or len(unlimited) < 2:
                    continue
            return mask, dims, selected, [groups[q] for q in selected]
    # Guaranteed bounded construction: a switchable square plus independently
    # long forced horizontal and vertical tails. Tail lengths vary; both
    # inventories are strictly positive. The normal sampler grows random unions.
    a = max(1, (count - 2) // 2)
    b = max(1, count - 2 - a)
    mask = np.zeros((2 + 2 * b, 2 + 2 * a), dtype=bool)
    mask[:2, :2] = True
    mask[0, 2:] = True
    mask[2:, 0] = True
    dims = [(1, 2), (2, 1)]
    qs = [(a + 2, b), (a, b + 2)]
    paths = []
    for q in qs:
        outcomes, stats = _cover(mask, [np.full(d, i + 10) for i, d in enumerate(dims)], q)
        if len(outcomes) != 1:
            raise RuntimeError('Fallback did not have a unique quota cover')
        paths.append(next(iter(outcomes.values())))
    if not twins:
        index = int(rng.integers(2))
        qs, paths = [qs[index]], [paths[index]]
    return mask, dims, qs, paths


def _add_types(rng, mask, dims, quota, path, wanted):
    """Attach additional arbitrary rectangles or islands, checking global fit."""
    dims, quota, path = list(dims), list(quota), list(path)
    choices = [(1, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (1, 4), (4, 1), (3, 3)]
    while len(dims) < wanted:
        accepted = False
        for attempt in range(55):
            d = choices[int(rng.integers(len(choices)))]
            if d in dims:
                continue
            multiplicity = int(rng.integers(1, 3))
            padded = np.pad(mask, ((4, 4), (4, 4)))
            possible = True
            for _ in range(multiplicity):
                h, w = d
                options = []
                island = rng.random() < .25
                for r in range(padded.shape[0] - h + 1):
                    for c in range(padded.shape[1] - w + 1):
                        if padded[r:r + h, c:c + w].any():
                            continue
                        near = padded[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1]
                        contact = ((r > 0 and padded[r - 1, c:c + w].any())
                                   or (r + h < len(padded) and padded[r + h, c:c + w].any())
                                   or (c > 0 and padded[r:r + h, c - 1].any())
                                   or (c + w < padded.shape[1] and padded[r:r + h, c + w].any()))
                        suitable = not near.any() if island else contact
                        if suitable:
                            options.append((r, c))
                if not options:
                    possible = False
                    break
                r, c = options[int(rng.integers(len(options)))]
                padded[r:r + h, c:c + w] = True
            if not possible:
                continue
            new_mask = _tight(padded)
            if new_mask.shape[1] > 16 or new_mask.shape[0] > 24:
                continue
            new_dims, new_q = dims + [d], quota + [multiplicity]
            markers = [np.full(shape, i + 10) for i, shape in enumerate(new_dims)]
            solutions, stats = _cover(new_mask, markers, new_q, budget=8000)
            if not stats['exhaustive'] or len(solutions) != 1:
                continue
            unlimited, s = _cover(new_mask, markers, budget=8000)
            if not s['exhaustive'] or len(unlimited) < 2:
                continue
            mask, dims, quota = new_mask, new_dims, new_q
            path = list(next(iter(solutions.values())))
            accepted = True
            break
        if not accepted:
            break  # Type count is structural variation, never a solving convention.
    return mask, dims, tuple(quota), tuple(path)


def _pattern(shape, key, accent, rng, witness=False):
    h, w = shape
    tile = np.full(shape, key, dtype=np.int8)
    mode = int(rng.integers(6))
    yy, xx = np.indices(shape)
    if mode == 1:
        tile[(xx + yy) % 2 == 1] = accent
    elif mode == 2:
        tile[yy % 2 == 1] = accent
    elif mode == 3:
        tile[xx % 2 == 1] = accent
    elif mode == 4:
        tile[rng.random(shape) < .35] = accent
    elif mode == 5 or witness:
        tile[0, 0] = accent
    if witness and tile.size > 1:
        tile.fill(key)
        tile[0, 0] = accent
    if not (tile == key).any():
        tile[-1, -1] = key
    return tile


def _scatter_rectangles(tiles, height, width, bg, rng):
    for attempt in range(35):
        tray = np.full((height, width), bg, dtype=np.int8)
        used = np.zeros(tray.shape, dtype=bool)
        order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
        success = True
        for i in order:
            h, w = tiles[i].shape
            options = []
            for r in range(height - h + 1):
                for c in range(width - w + 1):
                    if not used[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1].any():
                        options.append((r, c))
            if not options:
                success = False
                break
            r, c = options[int(rng.integers(len(options)))]
            tray[r:r + h, c:c + w] = tiles[i]
            used[r:r + h, c:c + w] = True
        if success:
            return tray
    return None


def _counter_masks(rng, quotas, areas):
    objects = []
    for k, (count, area) in enumerate(zip(quotas, areas)):
        sizes = [1] * count
        for _ in range(area - count):
            sizes[int(rng.integers(count))] += 1
        for size in sizes:
            boxes = [(h, w) for h in range(1, 7) for w in range(1, 6) if h * w >= size]
            if not boxes:
                raise RuntimeError('Counter ink budget exceeds bounded construction capacity')
            bh, bw = boxes[int(rng.integers(len(boxes)))]
            obj = np.zeros((bh, bw), dtype=bool)
            start = (int(rng.integers(bh)), int(rng.integers(bw)))
            frontier = {start}
            for _ in range(size):
                choices = sorted(frontier)
                r, c = choices[int(rng.integers(len(choices)))]
                frontier.remove((r, c))
                obj[r, c] = True
                for dr, dc in _STEPS:
                    y, x = r + dr, c + dc
                    if 0 <= y < bh and 0 <= x < bw and not obj[y, x]:
                        frontier.add((y, x))
            objects.append((k, _tight(obj)))
    return objects


def _scatter_counters(rng, height, width, bg, keys, quotas, areas):
    objects = _counter_masks(rng, quotas, areas)
    for attempt in range(45):
        used = np.zeros((height, width), dtype=bool)
        counter = np.full((height, width), bg, dtype=np.int8)
        order = sorted(range(len(objects)), key=lambda i: -int(objects[i][1].sum()))
        success = True
        for i in order:
            k, obj = objects[i]
            h, w = obj.shape
            options = []
            for r in range(height - h + 1):
                for c in range(width - w + 1):
                    if not used[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1].any():
                        options.append((r, c))
            if not options:
                success = False
                break
            r, c = options[int(rng.integers(len(options)))]
            used[r:r + h, c:c + w] |= obj
            counter[r:r + h, c:c + w][obj] = keys[k]
        if success:
            return counter
    return None


def _decode(grid):
    """Unique three-panel semantic parse, using no output or generation state."""
    candidates = []
    for vertical in (False, True):
        g = grid.T if vertical else grid
        for divider in range(10):
            cuts = np.flatnonzero(np.all(g == divider, axis=0)).tolist()
            if len(cuts) != 2 or int((g == divider).sum()) != 2 * g.shape[0]:
                continue
            a, b = cuts
            if not (a > 0 and b > a + 1 and b < g.shape[1] - 1):
                continue
            board, tray, counter = g[:, :a], g[:, a + 1:b], g[:, b + 1:]
            colors = set(map(int, np.unique(board)))
            if len(colors) != 2:
                continue
            common = colors & set(map(int, np.unique(tray))) & set(map(int, np.unique(counter)))
            if len(common) != 1:
                continue
            bg = next(iter(common))
            board_bg = next(iter(colors - {bg}))
            if (tray == board_bg).any() or (counter == board_bg).any():
                continue
            keys = sorted(set(map(int, np.unique(counter))) - {bg})
            if not 2 <= len(keys) <= 4:
                continue
            tiles_by_key = {}
            valid = True
            for cells in _components(tray != bg):
                ys, xs = zip(*cells)
                tile = tray[min(ys):max(ys) + 1, min(xs):max(xs) + 1].copy()
                payload = set(map(int, np.unique(tile)))
                match = payload & set(keys)
                if tile.size != len(cells) or (tile == bg).any() or len(match) != 1:
                    valid = False
                    break
                key = next(iter(match))
                if key in tiles_by_key:
                    valid = False
                    break
                tiles_by_key[key] = tile
            if not valid or set(tiles_by_key) != set(keys):
                continue
            accent_colors = set(map(int, np.unique(tray))) - {bg} - set(keys)
            if len(accent_colors) > 1:
                continue
            quotas = [len(_components(counter == k)) for k in keys]
            if any(len(_components(counter == k, True)) != q for k, q in zip(keys, quotas)):
                continue
            # Counter objects of differing colors also have a visible blank moat.
            if len(_components(counter != bg, True)) != sum(quotas):
                continue
            tiles = [tiles_by_key[k] for k in keys]
            candidates.append(dict(vertical=vertical, board=board.copy(), mask=board == bg,
                                   tray=tray.copy(), counter=counter.copy(), keys=keys,
                                   tiles=tiles, quotas=quotas, vacancy=bg, board_bg=board_bg,
                                   divider=divider, accent=tuple(accent_colors)))
    if len(candidates) != 1:
        raise ValueError('Input must have one identifiable board/prototype/counter parse')
    return candidates[0]


def _analyze(input_grid):
    d = _decode(_grid(input_grid))
    outcomes, stats = _cover(d['mask'], d['tiles'], d['quotas'])
    if len(outcomes) != 1:
        raise ValueError('Counted inventory must give exactly one rendered completion')
    rendering, path = next(iter(outcomes.items()))
    out = d['board'].copy()
    out[d['mask']] = rendering
    d.update(output=out.T.copy() if d['vertical'] else out,
             canonical_output=out, path=path, stats=stats)
    return d


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the whole completed board solely from visible prototypes and counts."""
    return _analyze(input_grid)['output']


def _make_pairs(rng, mask, dims, quota_vectors, paths, vertical, scale_limit=2):
    """Build one example or the causal twin demonstrations from visible data."""
    colors = list(map(int, rng.choice(10, len(dims) + 4, replace=False)))
    board_bg, bg, divider, accent = colors[:4]
    keys = colors[4:]
    # Twins have the same ink area per key, despite differing component counts.
    maxima = [max(q[k] for q in quota_vectors) for k in range(len(dims))]
    areas = [m + int(rng.integers(2, 5)) for m in maxima]
    scales = [(h, w) for h in range(1, scale_limit + 1) for w in range(1, scale_limit + 1)]
    rng.shuffle(scales)
    for sh, sw in scales:
        scaled_mask = np.repeat(np.repeat(mask, sh, axis=0), sw, axis=1)
        scaled_dims = [(h * sh, w * sw) for h, w in dims]
        min_tray = max(w for h, w in scaled_dims)
        minimum_board_w = scaled_mask.shape[1]
        if minimum_board_w + min_tray + 7 > 30 or scaled_mask.shape[0] > 28:
            continue
        tiles = [_pattern(d, key, accent, rng, witness=i == 0 or len(quota_vectors) > 1)
                 for i, (d, key) in enumerate(zip(scaled_dims, keys))]
        # One random canvas and prototype layout is shared by the twin pair.
        for layout_attempt in range(40):
            max_pad = min(4, 30 - min_tray - 7 - minimum_board_w)
            padding = int(rng.integers(max_pad + 1))
            board_w = minimum_board_w + padding
            tray_w_max = min(9, 30 - board_w - 7)
            if min_tray > tray_w_max:
                continue
            tray_w = int(rng.integers(min_tray, tray_w_max + 1))
            counter_w = int(rng.integers(5, 30 - board_w - tray_w - 1))
            height_min = max(12, scaled_mask.shape[0], max(h for h, w in scaled_dims))
            if len(quota_vectors) > 1:
                # The demonstrations distinguish returning the entire board from
                # cropping to its vacancy silhouette, even for a (2, 1) request.
                height_min = max(height_min, scaled_mask.shape[0] + 1)
            height = int(rng.integers(height_min, 31))
            tray = _scatter_rectangles(tiles, height, tray_w, bg, rng)
            if tray is None:
                continue
            counters = [_scatter_counters(rng, height, counter_w, bg, keys, q, areas)
                        for q in quota_vectors]
            if any(c is None for c in counters):
                continue
            y = int(rng.integers(height - scaled_mask.shape[0] + 1))
            x = int(rng.integers(padding + 1))
            board = np.full((height, board_w), board_bg, dtype=np.int8)
            board[y:y + scaled_mask.shape[0], x:x + scaled_mask.shape[1]][scaled_mask] = bg
            line = np.full((height, 1), divider, dtype=np.int8)
            pairs = []
            valid = True
            for quota, path, counter in zip(quota_vectors, paths, counters):
                out = np.full(board.shape, board_bg, dtype=np.int8)
                for k, r, c in path:
                    tile = tiles[k]
                    h, w = tile.shape
                    out[y + r * sh:y + r * sh + h, x + c * sw:x + c * sw + w] = tile
                inp = np.concatenate((board, line, tray, line, counter), axis=1)
                pair = {'input': inp.T.copy() if vertical else inp,
                        'output': out.T.copy() if vertical else out}
                # Scaling may permit shifted tilings not present at unit scale;
                # actual pixel uniqueness is checked again, from the full input.
                try:
                    d = _analyze(pair['input'])
                except ValueError:
                    valid = False
                    break
                if not np.array_equal(d['output'], pair['output']):
                    valid = False
                    break
                unlimited, stats = _cover(d['mask'], d['tiles'], budget=20000)
                if not stats['exhaustive'] or len(unlimited) < 2:
                    valid = False
                    break
                pairs.append(pair)
            if valid:
                return pairs
    raise RuntimeError('Could not pack visible recipe panels within 30x30')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base_vertical = bool(rng.integers(2))
    transfer = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for attempt in range(12):
        mask, dims, qs, paths = _base_geometry(rng, int(rng.integers(5, 10)), twins=True)
        try:
            pairs = _make_pairs(rng, mask, dims, qs, paths, base_vertical, 3)
        except RuntimeError:
            continue
        if not np.array_equal(pairs[0]['output'], pairs[1]['output']):
            puzzle['train'].extend(pairs)
            seen.update(_key(p['input']) for p in pairs)
            break
    else:
        raise RuntimeError('Could not construct teaching twins')
    for split, amount in (('train', nt - 2), ('test', ne)):
        for index in range(amount):
            max_train_copies = max(sum(_decode(p['input'])['quotas']) for p in puzzle['train'])
            vertical = (not base_vertical if split == 'test' else base_vertical) if transfer else bool(rng.integers(2))
            for attempt in range(16):
                count = (int(rng.integers(max_train_copies + 1, max_train_copies + 4))
                         if split == 'test' and not transfer else int(rng.integers(5, 10)))
                mask, dims, qs, paths = _base_geometry(rng, count)
                wanted = int(rng.choice([2, 3, 4], p=[.25, .50, .25]))
                mask, dims, q, path = _add_types(rng, mask, dims, qs[0], paths[0], wanted)
                if split == 'test' and not transfer and sum(q) <= max_train_copies:
                    continue
                try:
                    pair = _make_pairs(rng, mask, dims, [q], [path], vertical,
                                       1 if split == 'test' and not transfer else 2)[0]
                except RuntimeError:
                    continue
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not generate a distinct, shift-compatible example')
    rng.shuffle(puzzle['train'])
    return puzzle


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    try:
        _counts(nt, ne)
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys')
            try:
                inp, out = _grid(pair['input']), _grid(pair['output'])
                d = _analyze(inp)
            except ValueError as exc:
                raise AssertionError(str(exc)) from exc
            _require(_key(inp) not in seen, 'Duplicate input')
            seen.add(_key(inp))
            _require(np.array_equal(out, d['output']), 'Incorrect quota mosaic output')
            _require(np.array_equal(d['canonical_output'][~d['mask']], d['board'][~d['mask']]),
                     'Board background or margins changed')
            _require(sum(t.size * q for t, q in zip(d['tiles'], d['quotas'])) == int(d['mask'].sum()),
                     'Inventory area is not conserved')
            used = [0] * len(d['keys'])
            for k, r, c in d['path']:
                used[k] += 1
            _require(used == d['quotas'], 'Not every counted stamp was used exactly once')
            unlimited, stats = _cover(d['mask'], d['tiles'], budget=50000)
            _require(stats['exhaustive'] and len(unlimited) >= 2,
                     'Counters do not distinguish competing rendered packings')
            d['ink_areas'] = tuple(int((d['counter'] == k).sum()) for k in d['keys'])
            (training if split == 'train' else testing).append(d)
    twins = [(a, b) for i, a in enumerate(training) for b in training[i + 1:]
             if a['vertical'] == b['vertical'] and np.array_equal(a['board'], b['board'])
             and np.array_equal(a['tray'], b['tray']) and a['keys'] == b['keys']
             and a['quotas'] != b['quotas'] and sum(a['quotas']) == sum(b['quotas'])
             and a['ink_areas'] == b['ink_areas']
             and not np.array_equal(a['output'], b['output'])]
    _require(bool(twins), 'Training needs same-geometry, equal-ink-area quota twins')
    _require(any(q > 1 for d in training for q in d['quotas']), 'No repetition witness')
    _require(any(t.size > 1 and not np.array_equal(t, t[::-1, ::-1])
                 for d in training for t in d['tiles']), 'No pattern orientation witness')
    for transform in range(1, 8):
        _require(any(not np.array_equal(t, np.rot90(t[:, ::-1] if transform >= 4 else t,
                                                   transform % 4))
                     for d in training for t in d['tiles']), 'Undemonstrated pattern orientation')
    _require(any(_tight(d['mask']).shape != d['board'].shape for d in training),
             'Missing full-board margin preservation witness')
    _require(any(any(len(c) > 1 for c in _components(d['counter'] == k))
                 for d in training for k in d['keys']), 'Pixel counting is not excluded')
    # Shared-mask twin outputs necessarily put different stamp types in some cells:
    # copying source order or matching entire vacancy components cannot explain them.
    _require(any(len(_components(d['mask'])) < sum(d['quotas']) for d in training),
             'No merged silhouette decomposition witness')
    train_axes = {d['vertical'] for d in training}
    _require(all(d['vertical'] not in train_axes or sum(d['quotas']) > max(sum(a['quotas']) for a in training)
                 for d in testing), 'Missing orientation or count train/test shift')


def self_test():
    """Bounded 42-seed semantic, variation, determinism and corruption checks."""
    signatures, input_shapes, output_shapes, silhouettes, tile_shapes = set(), set(), set(), set(), set()
    type_counts, counts, axes, components, palettes = set(), set(), set(), set(), set()
    pairs_checked = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple((_key(x['input']), _key(x['output'])) for s in ('train', 'test') for x in p[s]))
        if seed in (0, 17, 41):
            again = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], again[s])
                       for k in ('input', 'output'))
        for pair in p['train'] + p['test']:
            d = _decode(pair['input'])
            input_shapes.add(pair['input'].shape)
            output_shapes.add(pair['output'].shape)
            silhouettes.add(_key(_tight(d['mask'])))
            tile_shapes.update(t.shape for t in d['tiles'])
            type_counts.add(len(d['keys']))
            counts.add(sum(d['quotas']))
            axes.add(d['vertical'])
            components.add(len(_components(d['mask'])))
            palettes.add((d['board_bg'], d['vacancy'], d['divider'], tuple(d['keys'])))
            # Input-only solver is equivariant under every global palette permutation.
            mapping = (np.arange(10) + 3) % 10
            assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
            pairs_checked += 1
    assert len(signatures) == 42 and len(silhouettes) >= 70
    assert len(input_shapes) >= 30 and len(output_shapes) >= 25 and len(tile_shapes) >= 12
    assert type_counts == {2, 3, 4} and axes == {False, True}
    assert len(counts) >= 7 and max(components) > 1 and len(palettes) >= 100
    for nt in range(2, 11):
        for ne in range(1, min(nt - 1, 4) + 1):
            p = generate_puzzle(814, nt, ne)
            validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal example counts accepted')
    p = generate_puzzle(0)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'].flat[0] = (int(bad['test'][0]['output'].flat[0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Validator accepted a corrupted output')
    # Quotas resolve the same silhouette into two different mosaics. Insufficient
    # inventory is rejected, whereas a single identical rendering is accepted.
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=bool)
    tiles = [np.full((1, 2), 2), np.full((2, 1), 3)]
    a, _ = _cover(mask, tiles, [3, 1])
    b, _ = _cover(mask, tiles, [1, 3])
    assert len(a) == len(b) == 1 and next(iter(a)) != next(iter(b))
    assert not _cover(mask, tiles, [1, 1])[0]
    assert len(_cover(mask, tiles)[0]) == 2
    # A hand-built family member exercises counters outside the sampled shape
    # envelope and verifies the copied pattern phase against explicit pixels.
    board = np.full((12, 6), 1, dtype=np.int8)
    board[4:8, 1:5][mask] = 0
    catalog = np.zeros((12, 5), dtype=np.int8)
    catalog[1, :2] = [2, 7]
    catalog[4:6, 3] = [3, 7]
    counter = np.zeros((12, 11), dtype=np.int8)
    for c in (0, 3, 6):
        counter[:5, c:c + 2] = 2  # Three ten-cell objects.
    counter[:, 10] = 3  # One twelve-cell object, taller than generated counters.
    line = np.full((12, 1), 9, dtype=np.int8)
    fixture = np.concatenate((board, line, catalog, line, counter), axis=1)
    expected = np.full(board.shape, 1, dtype=np.int8)
    expected[4, 1:5] = [2, 7, 2, 7]
    expected[5, 1:3] = [2, 7]
    expected[6:8, 1] = [3, 7]
    assert np.array_equal(solve(fixture), expected)
    assert np.array_equal(solve(fixture.T), expected.T)
    insufficient = fixture.copy()
    insufficient[:5, -11:-9] = 0
    try:
        solve(insufficient)
    except ValueError:
        pass
    else:
        raise AssertionError('Solver accepted insufficient counted inventory')
    return {'seeds': 42, 'pairs_checked': pairs_checked, 'distinct_puzzles': len(signatures),
            'input_shapes': len(input_shapes), 'output_shapes': len(output_shapes),
            'silhouettes': len(silhouettes), 'prototype_shapes': len(tile_shapes),
            'type_counts': sorted(type_counts), 'stamp_totals': sorted(counts),
            'vacancy_component_counts': sorted(components),
            'semantic_checks': 'generator-owned exhaustive packing and counterfactual checks'}
