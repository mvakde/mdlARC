"""Register colored fragments, recognize crosses, highlight their source cells.

Only NumPy and the standard library are used. No generation state enters solve.
"""
from __future__ import annotations

import json
import numpy as np

_CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _turn(point, n, k):
    r, c = map(int, point)
    for _ in range(k % 4):
        r, c = c, n - 1 - r
    return r, c


def _cross_union(mask):
    centers = np.zeros_like(mask, dtype=bool)
    if min(mask.shape) >= 3:
        centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
            & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:])
    union = centers.copy()
    union[1:] |= centers[:-1]
    union[:-1] |= centers[1:]
    union[:, 1:] |= centers[:, :-1]
    union[:, :-1] |= centers[:, 1:]
    return centers, union


def _direct_crosses(points, n):
    """Coordinate membership checker, separate from mask shifts in production."""
    centers, union = set(), set()
    for r, c in sorted(points):
        if 0 < r < n - 1 and 0 < c < n - 1:
            cross = {(r, c)} | {(r + dr, c + dc) for dr, dc in _CARDINAL}
            if cross <= points:
                centers.add((r, c))
                union.update(cross)
    return centers, union


def _register(info):
    source, bg, counts = info['source'], info['background'], info['counts']
    occupied = np.zeros(source.shape, dtype=bool)
    masks = {}
    for color in info['colors']:
        mask = np.rot90(source == color, -counts.get(color, 0) % 4)
        if np.any(mask & occupied):
            raise ValueError("registered color masks collide")
        masks[color] = mask
        occupied |= mask
    return masks, occupied


def _candidate(g, bg, ink, corners):
    rows, cols = corners.T
    r, c, e, f = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    side = e - r + 1
    if side < 7 or f - c + 1 != side:
        return None
    if set(map(tuple, corners.tolist())) != {(r, c), (r, f), (e, c), (e, f)}:
        return None
    window = g[r:e + 1, c:f + 1]
    perimeter = np.ones(window.shape, dtype=bool)
    perimeter[1:-1, 1:-1] = False
    expected = np.full(window.shape, bg, dtype=np.int8)
    expected[0, 0] = expected[0, -1] = expected[-1, 0] = expected[-1, -1] = ink
    if not np.array_equal(window[perimeter], expected[perimeter]):
        return None
    source = window[1:-1, 1:-1].copy()
    colors = tuple(int(v) for v in np.unique(source) if v != bg)
    if not 2 <= len(colors) <= 5 or ink in colors:
        return None
    exterior = g != bg
    exterior = exterior.copy()
    exterior[r:e + 1, c:f + 1] = False
    # A clear one-cell halo surrounds the entire window, not just its pegs.
    if exterior[max(0, r - 1):min(len(g), e + 2),
                max(0, c - 1):min(g.shape[1], f + 2)].any():
        return None
    counts, bars = {}, {}
    for color in np.unique(g[exterior]):
        color = int(color)
        coords = np.argwhere(exterior & (g == color))
        yy, xx = coords.T
        horizontal = np.ptp(yy) == 0 and np.ptp(xx) + 1 == len(coords)
        vertical = np.ptp(xx) == 0 and np.ptp(yy) + 1 == len(coords)
        if color not in colors or not (horizontal or vertical):
            return None
        counts[color] = len(coords)
        bars[color] = {'coords': tuple(map(tuple, coords.tolist())),
                      'orientation': 'single' if len(coords) == 1 else ('horizontal' if horizontal else 'vertical')}
    if not counts:
        return None
    all_bar_cells = [(color, p) for color, bar in bars.items() for p in bar['coords']]
    for i, (color, p) in enumerate(all_bar_cells):
        for other, q in all_bar_cells[i + 1:]:
            if color != other and max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1:
                return None
    info = {'background': bg, 'ink': ink, 'row': r, 'col': c, 'size': side - 2,
            'source': source, 'colors': colors, 'counts': counts, 'bars': bars}
    try:
        _register(info)
    except ValueError:
        return None
    return info


def _locate(input_grid):
    g = _grid(input_grid)
    values, masses = np.unique(g, return_counts=True)
    bg = int(values[masses.argmax()])
    if int(masses.max()) <= g.size // 2:
        raise ValueError("background must occupy strictly more than half the input")
    candidates = []
    for color, mass in zip(values, masses):
        if color != bg and mass == 4:
            info = _candidate(g, bg, int(color), np.argwhere(g == color))
            if info is not None:
                candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("expected exactly one peg square with matching exterior bars")
    return candidates[0]


def _answer(info):
    masks, occupied = _register(info)
    _, chosen = _cross_union(occupied)
    source_members = np.zeros_like(occupied)
    for color, mask in masks.items():
        source_members |= np.rot90(chosen & mask, info['counts'].get(color, 0) % 4)
    out = info['source'].copy()
    out[source_members] = info['ink']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, counters, registration, and ink from this input alone."""
    return _answer(_locate(input_grid))


def _coordinate_answer(info, counts=None, kind='normal'):
    """Coordinate validation and concrete plausible alternative rules.

    Wrong registrations may collide; their occupied union still defines the
    alternative, with every colliding contributor marked if that cell qualifies.
    """
    counts = info['counts'] if counts is None else counts
    n, source, bg = info['size'], info['source'], info['background']
    contributors, per_color = {}, {}
    for r, c in np.argwhere(source != bg):
        p, color = (int(r), int(c)), int(source[r, c])
        q = _turn(p, n, counts.get(color, 0))
        contributors.setdefault(q, []).append((p, color))
        per_color.setdefault(color, set()).add(q)
    points = set(contributors)
    centers, chosen = _direct_crosses(points, n)
    if kind == 'all':
        chosen = points
    elif kind == 'monochrome':
        chosen = set()
        for cells in per_color.values():
            chosen.update(_direct_crosses(cells, n)[1])
    elif kind == 'centers':
        chosen = centers
    elif kind == 'sequential':
        remaining, chosen = points.copy(), set()
        for r, c in sorted(centers):
            cross = {(r, c)} | {(r + dr, c + dc) for dr, dc in _CARDINAL}
            if cross <= remaining:
                chosen.update(cross)
                remaining.difference_update(cross)
    elif kind == 'component':
        selected, stack = chosen.copy(), list(chosen)
        while stack:
            r, c = stack.pop()
            for dr, dc in _CARDINAL:
                q = r + dr, c + dc
                if q in points and q not in selected:
                    selected.add(q)
                    stack.append(q)
        chosen = selected
    if kind == 'registered':
        result = np.full(source.shape, bg, dtype=np.int8)
        for q, members in contributors.items():
            result[q] = info['ink'] if q in chosen else members[0][1]
        return result
    result = source.copy()
    for q in chosen:
        for p, _ in contributors[q]:
            result[p] = info['ink']
    return result


def _alternatives(info):
    counts = info['counts']
    ranked = {k: rank + 1 for rank, k in enumerate(sorted(set(counts.values())))}
    result = {
        'omit_B': _coordinate_answer(info, {}),
        'omit_A': _coordinate_answer(info, kind='all'),
        'monochrome': _coordinate_answer(info, kind='monochrome'),
        'registered_output': _coordinate_answer(info, kind='registered'),
        'counterclockwise': _coordinate_answer(info, {c: -k for c, k in counts.items()}),
        'shared_turn': _coordinate_answer(info, {c: 1 for c in info['colors']}),
        'shared_two': _coordinate_answer(info, {c: 2 for c in info['colors']}),
        'shared_three': _coordinate_answer(info, {c: 3 for c in info['colors']}),
        'ranked_lengths': _coordinate_answer(info, {c: ranked[k] for c, k in counts.items()}),
        'centers_only': _coordinate_answer(info, kind='centers'),
        'destructive': _coordinate_answer(info, kind='sequential'),
        'component': _coordinate_answer(info, kind='component'),
    }
    for k in (1, 2, 3):
        altered = {c: counts.get(c, k) for c in info['colors']}
        result[f'absent_{k}'] = _coordinate_answer(info, altered)
    return result


def _cross_points(r, c):
    return [(r, c)] + [(r + dr, c + dc) for dr, dc in _CARDINAL]


def _add(mask, points):
    for r, c in points:
        if 0 <= r < len(mask) and 0 <= c < len(mask):
            mask[r, c] = True


def _mixed(mask):
    centers, _ = _cross_union(mask)
    points = list(map(tuple, np.argwhere(centers).tolist()))
    # Crosses share cells exactly when their centers' Manhattan distance <=2.
    overlaps = [any(q != p and abs(p[0]-q[0]) + abs(p[1]-q[1]) <= 2
                    for q in points) for p in points]
    return any(overlaps) and not all(overlaps)


def _scene(rng, n, style):
    mask = np.zeros((n, n), dtype=bool)
    if style in ('simple', 'single'):
        r, c = map(int, rng.integers(1, n - 1, size=2))
        if style == 'simple':
            r, c = map(int, rng.integers(2, n - 2, size=2))
        _add(mask, _cross_points(r, c))
        if style == 'simple':
            dr, dc = _CARDINAL[int(rng.integers(4))]
            _add(mask, [(r + k*dr, c + k*dc) for k in range(2, int(rng.integers(3, 6)))])
            # A nearby or detached incomplete T provides negative evidence.
            a, b = map(int, rng.integers(1, n - 1, size=2))
            _add(mask, [(a, b), (a, b-1), (a, b+1), (a+1, b)])
    elif style in ('overlap', 'mixed'):
        r = int(rng.integers(0, n - 2))
        c = int(rng.integers(0, n - 3))
        mask[r:r + 3, c:c + 4] = True
        if style == 'mixed':
            options = []
            block = list(map(tuple, np.argwhere(mask).tolist()))
            for a in range(1, n - 1):
                for b in range(1, n - 1):
                    cross = _cross_points(a, b)
                    if all(max(abs(x-u), abs(y-v)) > 1 for x, y in cross for u, v in block):
                        options.append((a, b))
            if not options:
                return mask
            a, b = options[int(rng.integers(len(options)))]
            _add(mask, _cross_points(a, b))
            if rng.random() < .65:
                dr, dc = _CARDINAL[int(rng.integers(4))]
                _add(mask, [(a + k*dr, b + k*dc) for k in range(2, 5)])
    elif style == 'array':
        gap_r, gap_c = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        rows = range(int(rng.integers(1, 3)), n - 1, gap_r)
        cols = range(int(rng.integers(1, 3)), n - 1, gap_c)
        for r in rows:
            for c in cols:
                if rng.random() < .8:
                    _add(mask, _cross_points(r, c))
    else:  # scattering, branching strokes, blocks, clipped candidates
        number = int(rng.integers(1, max(2, n*n // 28)))
        for _ in range(number):
            r, c = map(int, rng.integers(1, n - 1, size=2))
            _add(mask, _cross_points(r, c))
    # A small grammar of varied surplus strokes and local near misses.
    if style not in ('simple', 'overlap', 'single'):
        for _ in range(int(rng.integers(1, max(3, n // 2)))):
            r, c = map(int, rng.integers(n, size=2))
            form = int(rng.integers(5))
            if form == 0:
                points = [(r, c), (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
            elif form == 1:
                points = [(r+a, c+b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
            elif form == 2:
                points = [(r+a, c+b) for a in range(int(rng.integers(2, 5)))
                          for b in range(int(rng.integers(2, 5)))]
            else:
                points = [(r, c)]
                direction = int(rng.integers(4))
                for _ in range(int(rng.integers(2, 9))):
                    if form == 4 and rng.random() < .4:
                        direction = (direction + int(rng.choice([-1, 1]))) % 4
                    dr, dc = _CARDINAL[direction]
                    r, c = r + dr, c + dc
                    points.append((r, c))
            if mask.sum() + len(points) <= .42 * n*n:
                _add(mask, points)
    mask = np.rot90(mask, int(rng.integers(4))).copy()
    if rng.random() < .5:
        mask = mask[:, ::-1].copy()
    if style == 'negative':
        mask[_cross_union(mask)[0]] = False
    return mask


def _paint_source(rng, mask, bg, colors, counts):
    n = len(mask)
    points = list(map(tuple, np.argwhere(mask).tolist()))
    if len(points) < len(colors):
        return None
    for _ in range(24):
        source = np.full(mask.shape, bg, dtype=np.int8)
        used, present = set(), set()
        for index in rng.permutation(len(points)):
            q = points[int(index)]
            choices = [(color, _turn(q, n, -counts.get(color, 0))) for color in colors]
            choices = [(color, p) for color, p in choices if p not in used]
            if not choices:
                break
            missing = [(color, p) for color, p in choices if color not in present]
            choices = missing or choices
            color, p = choices[int(rng.integers(len(choices)))]
            source[p] = color
            used.add(p)
            present.add(color)
        else:
            if present == set(colors):
                return source
    return None


def _window(source, bg, ink):
    n = len(source)
    result = np.full((n + 2, n + 2), bg, dtype=np.int8)
    result[1:-1, 1:-1] = source
    result[0, 0] = result[0, -1] = result[-1, 0] = result[-1, -1] = ink
    return result


def _bar_groups(blocked, row, col, side, length, forced=None):
    h, w = blocked.shape
    groups = []
    for orientation in ('horizontal', 'vertical'):
        if forced is not None and orientation != forced:
            continue
        bh, bw = (1, length) if orientation == 'horizontal' else (length, 1)
        if bh > h or bw > w:
            continue
        yy, xx = np.indices((h - bh + 1, w - bw + 1))
        clear = np.ones(yy.shape, dtype=bool)
        for i in range(length):
            clear &= ~blocked[yy + (i if bh > 1 else 0), xx + (i if bw > 1 else 0)]
        for side_id, predicate in enumerate((yy + bh <= row - 1, yy >= row + side + 1,
                                             xx + bw <= col - 1, xx >= col + side + 1)):
            starts = np.argwhere(clear & predicate)
            if len(starts):
                groups.append((orientation, side_id, starts))
    return groups


def _layout(rng, info, forced):
    window = _window(info['source'], info['background'], info['ink'])
    side = len(window)
    for _ in range(32):
        h = int(rng.integers(max(side, 12), 31))
        w = int(rng.integers(max(side, 12), 31))
        row, col = int(rng.integers(h - side + 1)), int(rng.integers(w - side + 1))
        inp = np.full((h, w), info['background'], dtype=np.int8)
        inp[row:row + side, col:col + side] = window
        blocked = np.zeros((h, w), dtype=bool)
        blocked[max(0, row-1):min(h, row+side+1), max(0, col-1):min(w, col+side+1)] = True
        ok = True
        order = sorted(info['counts'], key=lambda c: -info['counts'][c])
        for color in order:
            length = info['counts'][color]
            groups = _bar_groups(blocked, row, col, side, length, forced.get(color))
            if not groups:
                ok = False
                break
            orientation, _, starts = groups[int(rng.integers(len(groups)))]
            y, x = map(int, starts[int(rng.integers(len(starts)))])
            for k in range(length):
                a, b = y + (k if orientation == 'vertical' else 0), x + (k if orientation == 'horizontal' else 0)
                inp[a, b] = color
                blocked[max(0, a-1):min(h, a+2), max(0, b-1):min(w, b+2)] = True
        if ok:
            try:
                located = _locate(inp)
            except ValueError:
                continue
            if (located['row'], located['col'], located['size']) == (row, col, info['size']):
                return inp
    # Proven roomy fallback: five bars split between three bottom/right lanes.
    # Both compulsory repeated-count orientations can be honored.
    inp = np.full((30, 30), info['background'], dtype=np.int8)
    inp[:side, :side] = window
    orientations = dict(forced)
    for color in info['counts']:
        if color not in orientations:
            masses = {o: list(orientations.values()).count(o) for o in ('horizontal', 'vertical')}
            orientations[color] = min(masses, key=masses.get)
    lanes = {'horizontal': 0, 'vertical': 0}
    for color, length in info['counts'].items():
        orient = orientations[color]
        offset = side + 1 + 2 * lanes[orient]
        lanes[orient] += 1
        if offset >= 30:
            return None
        if orient == 'horizontal':
            inp[offset, :length] = color
        else:
            inp[:length, offset] = color
    try:
        located = _locate(inp)
    except ValueError:
        return None
    return inp if (located['row'], located['col'], located['size']) == (0, 0, info['size']) else None


def _fallback_pair(mode, split, witness, seen):
    """Finite constructive fallback, with more distinct inputs than pair slots.

    These are ordinary rule members, not solver special cases. Four sizes and
    ten cyclic palettes give forty distinct candidates; at most fourteen inputs
    can have been used. The coordinates avoid source and registered collisions.
    """
    for n in (13, 15, 20, 21):
        if witness == 1 or split == 'test':
            target = {(3,3):2, (3,4):2, (3,5):3, (3,6):3,
                      (4,3):2, (4,4):1, (4,5):1, (4,6):3,
                      (5,3):3, (5,4):2, (5,5):3, (5,6):2}
            counts = {1:3, 3:4} if witness == 1 else {1:5, 2:2, 3:4}
            if split == 'test':
                target.update({(8,8):1, (7,8):2, (9,8):1, (8,7):3, (8,9):2})
        else:
            four_colors = mode == 1 and witness == 0
            counts = {1:1, 2:5, 3:2, 4:2} if four_colors else {1:1, 2:2, 3:2}
            if witness < 0:
                counts = {1:1 if mode == 0 else 5, 2:2, 3:4}
            target = {(2,2):1, (1,2):2, (3,2):1, (2,1):3,
                      (2,3):4 if four_colors else 2, (4,2):1,
                      (8,8):1, (8,7):1, (8,9):1, (9,8):1}
        source = np.zeros((n,n), dtype=np.int8)
        for q, color in target.items():
            p = _turn(q,n,-counts.get(color,0))
            assert source[p] == 0, 'fallback must not collide'
            source[p] = color
        for shift in range(10):
            recolored = (source + shift) % 10
            bg, ink = shift, (9+shift) % 10
            transformed_counts = {(color+shift) % 10:k for color,k in counts.items()}
            inp = np.full((30,30), bg, dtype=np.int8)
            inp[:n+2,:n+2] = _window(recolored,bg,ink)
            lanes = {'horizontal':0, 'vertical':0}
            for j,(color,k) in enumerate(transformed_counts.items()):
                orientation = 'horizontal' if j % 2 == 0 else 'vertical'
                offset = n+3 + 2*lanes[orientation]
                lanes[orientation] += 1
                if orientation == 'horizontal':
                    inp[offset,:k] = color
                else:
                    inp[:k,offset] = color
            if _key(inp) not in seen:
                return {'input':inp, 'output':solve(inp)}
    raise AssertionError('fallback uniqueness bound exceeded')


def _make_pair(rng, mode, split, witness, seen):
    for _ in range(180):
        if witness == 0:
            n, style = int(rng.integers(7, 18)), 'simple'
            lengths = [1, 2, 2] if mode == 0 else ([1, 5, 2, 2] if rng.random() < .5 else [1, 2, 6, 6])
        elif witness == 1:
            n, style, lengths = int(rng.integers(7, 18)), 'overlap', [3, 0, 4]
        else:
            n = int(rng.integers(9, 22)) if split == 'test' else int(rng.integers(5, 22))
            choices = ['single', 'simple', 'overlap', 'scatter', 'array', 'negative']
            style = 'mixed' if split == 'test' and mode == 1 else str(rng.choice(choices))
            if split == 'test' and style == 'negative':
                style = 'scatter'
            count = int(rng.integers(2, 6))
            cap = 4 if mode == 0 and split == 'train' else 9
            lengths = [int(rng.integers(0, cap + 1)) for _ in range(count)]
            if not any(lengths):
                lengths[int(rng.integers(count))] = 1
            if split == 'test':
                lengths[int(rng.integers(count))] = int(rng.integers(5, 10))
        if n < 7 and style == 'simple':
            style = 'single'
        palette = list(map(int, rng.choice(10, len(lengths) + 2, replace=False)))
        bg, ink, colors = palette[0], palette[1], palette[2:]
        counts = {color: k for color, k in zip(colors, lengths) if k}
        forced = {}
        if witness == 0:
            a, b = colors[-2:]
            orientations = ('horizontal', 'vertical') if rng.random() < .5 else ('vertical', 'horizontal')
            forced = dict(zip((a, b), orientations))
        mask = _scene(rng, n, style)
        centers, union = _cross_union(mask)
        if witness == 0 and (not centers.any() or int(union.sum()) != 5 * int(centers.sum())):
            continue
        if witness == 1 and (int(union.sum()) == 5 * int(centers.sum()) or _mixed(mask)):
            continue
        if mode == 1 and split == 'train' and _mixed(mask):
            continue
        if mode == 1 and split == 'test' and not _mixed(mask):
            continue
        source = _paint_source(rng, mask, bg, colors, counts)
        if source is None:
            continue
        info = {'source': source, 'background': bg, 'ink': ink, 'colors': tuple(colors),
                'counts': counts, 'size': n}
        expected = _answer(info)
        if witness >= 0:
            # Every controlled/default layer contributes visible selected cells.
            selected = expected == ink
            if any(not np.any(selected & (source == color)) for color in colors):
                continue
            alternatives = _alternatives(info)
            required = ['omit_B', 'omit_A', 'monochrome', 'registered_output',
                        'counterclockwise', 'shared_turn', 'shared_two', 'shared_three', 'centers_only']
            required += ['component'] if witness == 0 else ['destructive', 'ranked_lengths', 'absent_1', 'absent_2', 'absent_3']
            if any(np.array_equal(expected, alternatives[name]) for name in required):
                continue
        if split == 'test':
            if not centers.any() or np.array_equal(expected, _coordinate_answer(info, {})):
                continue
        inp = _layout(rng, info, forced)
        if inp is not None and _key(inp) not in seen:
            # Label through the input-only path, not by exposing target metadata.
            return {'input': inp, 'output': solve(inp)}
    return _fallback_pair(mode,split,witness,seen)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = index if split == 'train' and index < 2 else -1
            pair = _make_pair(rng, mode, split, witness, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, informative_lengths, evidence = set(), set(), set()
    shift_records = {'train': [], 'test': []}
    orientations = {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _locate(inp)
            assert all(1 <= k <= 9 for k in info['counts'].values())
            assert out.shape == (info['size'], info['size']), 'wrong crop'
            expected = _coordinate_answer(info)
            assert np.array_equal(out, expected), 'wrong source-cell membership or repainting'
            assert np.array_equal(out, solve(inp)), 'production and coordinate checks disagree'
            source = info['source']
            changed = source != out
            assert np.all(out[changed] == info['ink'])
            assert np.array_equal(out[~changed], source[~changed])
            masks, occupancy = _register(info)
            centers, union = _cross_union(occupancy)
            shift_records[split].append((max(info['counts'].values()), _mixed(occupancy)))
            assert int(changed.sum()) == int(union.sum()), 'inverse mapping must conserve selected area'
            assert not np.any(changed & (source == info['background'])), 'background cannot contribute'
            if split == 'train':
                for color in info['colors']:
                    selected = changed & (source == color)
                    if selected.any():
                        if color not in info['counts']:
                            evidence.add('absent')
                        else:
                            k = info['counts'][color]
                            informative_lengths.add(k)
                            orientations.setdefault(k, set()).add(info['bars'][color]['orientation'])
                for name, alternative in _alternatives(info).items():
                    if not np.array_equal(out, alternative):
                        evidence.add(name)
                if len({k % 4 for k in info['counts'].values()}) > 1:
                    evidence.add('different_counts')
            else:
                assert centers.any(), 'generated tests must exercise positive recognition'
                assert not np.array_equal(out, _coordinate_answer(info, {})), 'test must need registration'
    assert {1, 2, 3, 4} <= informative_lengths, 'teach short counts and a full cycle'
    assert any({'horizontal', 'vertical'} <= directions for directions in orientations.values())
    required = {'absent', 'different_counts', 'omit_A', 'omit_B', 'monochrome',
                'registered_output', 'counterclockwise', 'shared_turn', 'shared_two', 'shared_three',
                'ranked_lengths', 'absent_1', 'absent_2', 'absent_3', 'centers_only', 'destructive', 'component'}
    assert required <= evidence, f'missing training evidence: {required - evidence}'
    assert all(k >= 5 for k, _ in shift_records['test']), 'tests need cyclic long counters'
    if max(k for k, _ in shift_records['train']) >= 5:
        assert not any(mixed for _, mixed in shift_records['train'])
        assert all(mixed for _, mixed in shift_records['test']), 'hold out mixed cross geometry'


def _features(pair):
    info = _locate(pair['input'])
    masks, occupancy = _register(info)
    centers, union = _cross_union(occupancy)
    points = list(map(tuple, np.argwhere(centers).tolist()))
    n, r, c = info['size'], info['row'], info['col']
    shape = np.asarray(pair['input']).shape
    return {'input_shape': list(shape), 'size': n, 'position': [r, c],
            'background': info['background'], 'ink': info['ink'], 'layers': len(info['colors']),
            'lengths': sorted(info['counts'].values()), 'unmarked': len(info['colors']) - len(info['counts']),
            'centers': int(centers.sum()), 'selected': int(union.sum()),
            'content': int(occupancy.sum()), 'overlap': int(union.sum()) < 5 * int(centers.sum()),
            'mixed': _mixed(occupancy), 'edge_cross': any(a in (1,n-2) or b in (1,n-2) for a,b in points),
            'edge_window': min(r,c,shape[0]-r-n-2,shape[1]-c-n-2) == 0,
            'orientations': sorted({v['orientation'] for v in info['bars'].values()}),
            'multicolor': any(len({color for color, mask in masks.items()
                            if any(mask[p] for p in _cross_points(a,b))}) > 1 for a,b in points),
            'all_selected': int(union.sum()) == int(occupancy.sum())}


def self_test():
    """Bounded semantic regressions and measurable diversity over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fingerprints = {tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]) for p in puzzles}
    assert len(fingerprints) == 42
    assert tuple(_key(x['input']) for s in puzzles[7] for x in puzzles[7][s]) == tuple(
        _key(x['input']) for s in puzzles[7] for x in generate_puzzle(7)[s])
    records = [_features(pair) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    assert len({tuple(f['input_shape']) for f in records}) >= 40
    assert len({f['size'] for f in records}) >= 12
    assert len({tuple(f['position']) for f in records}) >= 30
    assert {f['background'] for f in records} == set(range(10))
    assert {f['ink'] for f in records} == set(range(10))
    assert {f['layers'] for f in records} >= {2,3,4,5}
    assert {k for f in records for k in f['lengths']} == set(range(1,10))
    assert len({f['centers'] for f in records}) >= 7
    for feature in ('overlap', 'mixed', 'edge_cross', 'edge_window', 'multicolor'):
        assert sum(f[feature] for f in records) >= 5, feature
    assert any(f['centers'] == 0 for f in records)
    # Every binary 3x3 neighborhood: cardinal arms matter, diagonals do not.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3,3)
        points = set(map(tuple, np.argwhere(mask).tolist()))
        centers, union = _cross_union(mask)
        direct_centers, direct_union = _direct_crosses(points, 3)
        assert set(map(tuple, np.argwhere(centers).tolist())) == direct_centers
        assert set(map(tuple, np.argwhere(union).tolist())) == direct_union
    block = np.ones((3,4), dtype=bool)
    assert _cross_union(block)[0].sum() == 2 and _cross_union(block)[1].sum() == 8
    clipped = np.zeros((5,5), dtype=bool)
    clipped[0,1:4] = True
    clipped[1,2] = clipped[-1,2] = True
    assert not _cross_union(clipped)[1].any(), 'no toroidal boundary neighbors'
    # Cycle aliases and inverse conservation on all generated demonstrations.
    for p in puzzles:
        for pair in p['train']:
            info = _locate(pair['input'])
            cycle_counts = {color: k+4 for color,k in info['counts'].items()}
            assert np.array_equal(_coordinate_answer(info, cycle_counts), pair['output'])
    # Explicit causal witness: both inputs satisfy the grammar, including no
    # collisions. Moving color 2's registration destroys a mixed cross and
    # withdraws highlights from colors 1 and 3, whose own counters are unchanged.
    n, counts = 9, {1: 1, 2: 2}
    source = np.zeros((n, n), dtype=np.int8)
    for q, color in [((2,2),1), ((1,2),2), ((3,2),1), ((2,1),3), ((2,3),2)]:
        source[_turn(q, n, -counts.get(color,0))] = color
    causal = np.zeros((15,19), dtype=np.int8)
    causal[:11,:11] = _window(source,0,9)
    causal[1,13] = 1
    causal[4,13:15] = 2
    altered = causal.copy()
    altered[4,15] = 2
    before, after = solve(causal), solve(altered)
    assert np.sum(before == 9) == 5 and not np.any(after == 9)
    assert np.any((before != after) & (source == 1))
    assert np.any((before != after) & (source == 3))
    # Exercise every finite fallback size and curriculum, including uniqueness.
    for mode in (0,1):
        seen = set()
        fallback = {'train':[], 'test':[]}
        for witness in (0,1):
            pair = _fallback_pair(mode,'train',witness,seen)
            seen.add(_key(pair['input']))
            fallback['train'].append(pair)
        pair = _fallback_pair(mode,'test',-1,seen)
        seen.add(_key(pair['input']))
        fallback['test'].append(pair)
        validate_generated_puzzle(fallback)
        # Fill all ten palettes at each size and advance to the next size.
        for _ in range(37):
            pair = _fallback_pair(mode,'test',-1,seen)
            seen.add(_key(pair['input']))
            info = _locate(pair['input'])
            assert np.array_equal(pair['output'],_coordinate_answer(info))
    # All legal explicit count combinations. Counts do not alter the rule.
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p = generate_puzzle(101 + a*10 + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a,b in ((1,1),(11,1),(4,0),(4,5),(2,2),(3,4),(True,1),(4,False),(3.5,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for bad_seed in (-1,True,1.5):
        try:
            generate_puzzle(bad_seed)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal seed accepted')
    bad = {s: [{k: np.asarray(v).copy() for k,v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt label accepted')
    # A bar color absent from the source must never invent a control layer.
    pair = puzzles[0]['train'][0]
    info = _locate(pair['input'])
    broken = pair['input'].copy()
    coordinate = next(iter(info['bars'].values()))['coords'][0]
    absent = next(c for c in range(10) if c not in info['colors'] and c not in (info['background'],info['ink']))
    broken[coordinate] = absent
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('unmatched counter accepted')
    # Return descriptive measurements; the verifier ignores this return value.
    return {'seeds': 42, 'pairs': len(records), 'input_shapes': len({tuple(f['input_shape']) for f in records}),
            'interior_sizes': sorted({f['size'] for f in records}),
            'center_counts': sorted({f['centers'] for f in records}),
            'feature_counts': {name: sum(f[name] for f in records)
                               for name in ('overlap','mixed','edge_cross','edge_window','multicolor')}}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
