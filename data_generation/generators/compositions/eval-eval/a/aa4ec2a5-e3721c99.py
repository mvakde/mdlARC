"""Nested stencil casting: topology selects a chamber's dye, geometry casts it.

Only the returned pixels are used by solve and validation.  Sampling preferences
are not answer conventions.  Dependencies: Python standard library and NumPy.
"""
from __future__ import annotations

from itertools import permutations
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _components(mask, neighbors=N4):
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    seen = np.zeros_like(mask)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        cells, stack = [], [(r, c)]
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for dr, dc in neighbors:
                y, x = rr + dr, cc + dc
                if 0 <= y < h and 0 <= x < w and mask[y, x] and not seen[y, x]:
                    seen[y, x] = True
                    stack.append((y, x))
        groups.append(cells)
    return groups


def _mask(cells, shape):
    a = np.zeros(shape, bool)
    if cells:
        a[tuple(np.asarray(cells).T)] = True
    return a


def _enclosed(mask, neighbors=N4):
    """Return distinct bounded components of the complement, with true outside."""
    mask = np.asarray(mask, bool)
    rr, cc = np.nonzero(mask)
    if not len(rr):
        return []
    r0, c0 = int(rr.min()), int(cc.min())
    r1, c1 = int(rr.max()) + 1, int(cc.max()) + 1
    # No bounded region can lie outside the barrier's bounding box. Cropping
    # saves full-canvas flood fills for each small stamp without changing holes.
    p = np.pad(mask[r0:r1, c0:c1], 1)
    result = []
    for cells in _components(~p, neighbors):
        if any(r in (0, p.shape[0] - 1) or c in (0, p.shape[1] - 1) for r, c in cells):
            continue
        result.append(_mask([(r + r0 - 1, c + c0 - 1) for r, c in cells], mask.shape))
    return result


def _union(masks, shape):
    a = np.zeros(shape, bool)
    for b in masks:
        a |= b
    return a


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w] for dr, dc in N8])


def _orth_dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w] for dr, dc in N4])


def _trim(mask):
    rr, cc = np.nonzero(mask)
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _signature(grid):
    a = np.asarray(grid, np.uint8)
    return a.shape, a.tobytes()


def _separated_runs(mask):
    for rows in (mask, mask.T):
        for row in rows:
            occupied = np.flatnonzero(row)
            if len(occupied) > 1 and np.any(np.diff(occupied) > 1):
                return True
    return False


def _parse(input_grid):
    """Recognize a separated key, then wall-only chambers and their stamps.

    Try all orientations; exactly one must be a valid key-above-field reading.
    There is no dominant-color heuristic, palette memory, or generation metadata.
    """
    original = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        grid = np.rot90(original, rotation)
        for cut in range(2, grid.shape[0] - 2):
            wall = int(grid[cut, 0])
            if not np.all(grid[cut] == wall):
                continue
            panel, work = grid[:cut], grid[cut + 1:]
            pv, wv = set(map(int, np.unique(panel))), set(map(int, np.unique(work)))
            common = pv & wv
            if len(wv) != 3 or wall not in wv or wall in pv or len(common) != 1:
                continue
            bg = next(iter(common))
            query = next(iter(wv - {bg, wall}))
            colors = sorted(pv - {bg})
            if query in pv or not 2 <= len(colors) <= 4:
                continue
            mapping, entries = {}, []
            good = True
            for color in colors:
                glyph = panel == color
                parts = _components(glyph)
                if len(parts) != 1 or glyph[0].any() or glyph[-1].any() or glyph[:, 0].any() or glyph[:, -1].any():
                    good = False
                    break
                holes = _enclosed(glyph)
                if any(np.any(panel[hole] != bg) for hole in holes) or len(holes) in mapping:
                    good = False
                    break
                mapping[len(holes)] = color
                entries.append((len(holes), color, _trim(glyph)))
            if not good:
                continue
            chambers = _enclosed(work == wall)
            if not chambers:
                continue
            chamber_labels = np.full(work.shape, -1, int)
            for i, domain in enumerate(chambers):
                chamber_labels[domain] = i
            parts = _components(work == query)
            if not parts or len(parts) != len(_components(work == query, N8)):
                continue
            stamps = []
            for cells in parts:
                body = _mask(cells, work.shape)
                ids = np.unique(chamber_labels[body])
                if len(ids) != 1 or ids[0] < 0:
                    good = False
                    break
                holes = _enclosed(body)
                cavity = _union(holes, work.shape)
                if np.any(work[cavity] != bg):
                    good = False
                    break
                stamps.append(dict(body=body, holes=cavity, count=len(holes),
                                   chamber=int(ids[0]), local=_trim(body)))
            if not good:
                continue
            rooms = []
            all_holes = _union([s['holes'] for s in stamps], work.shape)
            for i, domain in enumerate(chambers):
                children = [s for s in stamps if s['chamber'] == i]
                body = _union([s['body'] for s in children], work.shape)
                holes = _union([s['holes'] for s in children], work.shape)
                # The moat is OUTSIDE each original stamp, never in its holes.
                halo = _dilate(body) & domain & (work == bg) & ~all_holes
                rooms.append(dict(domain=domain, stamps=children, body=body, holes=holes,
                                  halo=halo, total=sum(s['count'] for s in children)))
            candidates.append(dict(grid=grid.copy(), work=work.copy(), cut=cut, rotation=rotation,
                                   background=bg, wall=wall, query=query, mapping=mapping,
                                   entries=entries, rooms=rooms, stamps=stamps))
    _require(len(candidates) == 1, "input must have one unambiguous key/chamber interpretation")
    return candidates[0]


def _cast(parsed, omit_a=False, omit_b=False):
    output = parsed['grid'].copy()
    work = output[parsed['cut'] + 1:]
    bg, query = parsed['background'], parsed['query']
    for room in parsed['rooms']:
        dye = parsed['mapping'].get(room['total'])
        if omit_b:
            dye = next(iter(parsed['mapping'].values()))
        if dye is None:
            work[room['domain']] = bg
            continue
        work[room['domain']] = dye
        if omit_a:
            work[room['body']] = query
            continue
        for stamp in room['stamps']:
            # Whole hollow bodies take the dye; solids retain query color.
            work[stamp['body']] = dye if stamp['count'] else query
        work[room['holes']] = query
        work[room['halo']] = bg
    return np.rot90(output, -parsed['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any valid family input using only its visible topology and key."""
    return _cast(_parse(input_grid))


def _compact(k):
    if k == 0:
        return np.ones((3, 3), bool)
    rows = 1 if k <= 2 else 2
    cols = (k + rows - 1) // rows
    a = np.ones((2 * rows + 1, 2 * cols + 1), bool)
    for i in range(k):
        a[1 + 2 * (i // cols), 1 + 2 * (i % cols)] = False
    return a


def _orient(a, rng):
    a = np.rot90(a, int(rng.integers(4)))
    if rng.random() < .5:
        a = np.fliplr(a)
    return a.copy()


def _stamp(k, rng, compact=False, open_shape=False, force_irregular=False):
    a = _compact(k)
    if k and (force_irregular or (not compact and rng.random() < .28)):
        # Enlarge one pocket in two axes, then give it an L-shaped interior.
        # Other pockets can remain narrow: area is irrelevant to the class.
        a = np.insert(a, 1, a[1], axis=0)
        a = np.insert(a, 1, a[:, 1], axis=1)
        a[1, 1] = True
    irregular_base = a.copy()
    if k == 0:
        if open_shape or (not compact and rng.random() < .48):
            h, w = (3, 3) if compact else (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
            a = np.ones((h, w), bool)
            a[:-1, 1:-1] = False  # genuine open bay, not a sealed cavity
        elif not compact:
            style = int(rng.integers(3))
            if style == 0:
                a = np.ones((int(rng.integers(1, 3)), int(rng.integers(1, 6))), bool)
            elif style == 1:
                a = np.asarray([[1, 0, 0], [1, 1, 0], [0, 1, 1]], bool)
            else:
                a = np.ones((int(rng.integers(2, 5)), int(rng.integers(2, 5))), bool)
    elif not compact:
        # Duplicate an interior row/column to change cavity areas and scale.
        if rng.random() < .65:
            axis = int(rng.integers(2))
            at = int(rng.integers(1, a.shape[axis] - 1))
            a = np.insert(a, at, np.take(a, at, axis=axis), axis=axis)
        # Fill a corner of a sufficiently large cavity: a nonrectangular hole.
        for hole in _enclosed(a):
            rr, cc = np.nonzero(hole)
            if len(rr) >= 4 and rng.random() < .55:
                b = a.copy()
                b[rr[0], cc[0]] = True
                if len(_components(b)) == 1 and len(_enclosed(b)) == k:
                    a = b
        if rng.random() < .45:
            b = np.pad(a, 1)
            col = int(rng.integers(1, a.shape[1] + 1))
            b[0, col] = True
            a = _trim(b)  # attached body lobe, not a separate object
    if force_irregular and not any(not _trim(h).all() for h in _enclosed(a)):
        a = irregular_base
    if not compact:
        a = _orient(a, rng)
    _require(len(_components(a)) == 1 and len(_enclosed(a)) == k, "stamp construction changed topology")
    return a


def _unit(counts, rng, compact=False, open_shape=False, irregular_anchor=False, teach_moat=False):
    """A wall-bounded chamber, with independently shaped and arranged stamps."""
    stamps = [_stamp(k, rng, compact, open_shape and k == 0, irregular_anchor and k == 2)
              for k in counts]
    if len(stamps) > 1 and not compact:
        rng.shuffle(stamps)
    margin = 1 if compact else int(rng.integers(0, 3))
    if teach_moat:
        margin = max(1, margin)
    # An all-solid chamber needs visible floor beyond its moat; otherwise its
    # correctly cast output could be an uninformative identity example.
    xmargin = max(margin, 2) if counts and not sum(counts) else margin
    gap = 1 if compact else int(rng.integers(1, 4))
    if len(stamps) <= 2:
        columns = max(1, len(stamps))
    else:
        columns = 2
    if not compact and len(stamps) == 2 and rng.random() < .35:
        columns = 1
    groups = [stamps[i:i + columns] for i in range(0, len(stamps), columns)]
    if not groups:
        groups = [[]]
    heights = [max((s.shape[0] for s in g), default=int(rng.integers(2, 5))) for g in groups]
    widths = [sum(s.shape[1] for s in g) + gap * max(0, len(g) - 1) for g in groups]
    ih = max(3, sum(heights) + gap * (len(groups) - 1) + 2 * margin)
    iw = max(3, max(widths, default=0) + 2 * xmargin)
    body = np.zeros((ih, iw), bool)
    r = margin
    for group, height, width in zip(groups, heights, widths):
        slack = iw - 2 * xmargin - width
        c = xmargin + (0 if compact else int(rng.integers(slack + 1)))
        for stamp in group:
            dy = 0 if compact else int(rng.integers(height - stamp.shape[0] + 1))
            body[r + dy:r + dy + stamp.shape[0], c:c + stamp.shape[1]] |= stamp
            c += stamp.shape[1] + gap
        r += height + gap
    domain = np.ones((ih, iw), bool)
    visible_floor = domain & ~body & ~_dilate(body)
    if not compact:
        # Wall peninsulas sculpt the floor, keeping its stamps and connectivity.
        for _ in range(int(rng.integers(1, 6))):
            b = domain.copy()
            side = int(rng.integers(4))
            if side < 2:
                c = int(rng.integers(iw))
                depth = int(rng.integers(1, min(4, ih)))
                if side == 0:
                    b[:depth, c] = False
                else:
                    b[-depth:, c] = False
            else:
                y = int(rng.integers(ih))
                depth = int(rng.integers(1, min(4, iw)))
                if side == 2:
                    b[y, :depth] = False
                else:
                    b[y, -depth:] = False
            if (not np.any(body & ~b) and len(_components(b)) == 1
                    and (not teach_moat or not np.any(_dilate(body) & ~b))
                    and (sum(counts) or not counts or np.any(b & visible_floor))):
                # Do not insert wall material into original stamp cavities.
                if not np.any(_union(_enclosed(body), body.shape) & ~b):
                    domain = b
    thickness = 1 if compact else int(rng.choice([1, 1, 2]))
    domain = np.pad(domain, thickness)
    body = np.pad(body, thickness)
    wall = ~domain
    if not compact and rng.random() < .45:
        protected = _dilate(domain)
        for r, c in ((0, 0), (0, wall.shape[1] - 1), (wall.shape[0] - 1, 0), (wall.shape[0] - 1, wall.shape[1] - 1)):
            if not protected[r, c] and rng.random() < .6:
                wall[r, c] = False
    _require(len(_enclosed(wall)) == 1 and np.array_equal(_enclosed(wall)[0], domain), "chamber construction failed")
    unit = np.zeros(domain.shape, np.int8)
    unit[wall] = 1
    unit[body] = 2
    return unit


def _locations(free, shape):
    h, w = shape
    if h > free.shape[0] or w > free.shape[1]:
        return np.empty((0, 2), int)
    p = np.pad((~free).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = p[h:, w:] - p[:-h, w:] - p[h:, :-w] + p[:-h, :-w]
    return np.argwhere(sums == 0)


def _pack(units, shape, rng):
    for _ in range(6):
        field = np.zeros(shape, np.int8)
        free = np.ones(shape, bool)
        for i in sorted(range(len(units)), key=lambda j: -units[j].size):
            unit = units[i]
            spots = _locations(free, unit.shape)
            if not len(spots):
                break
            if rng.random() < .35:
                edge = spots[(spots[:, 0] == 0) | (spots[:, 1] == 0) |
                             (spots[:, 0] + unit.shape[0] == shape[0]) |
                             (spots[:, 1] + unit.shape[1] == shape[1])]
                if len(edge):
                    spots = edge
            r, c = map(int, spots[int(rng.integers(len(spots)))])
            h, w = unit.shape
            field[r:r + h, c:c + w] = unit
            free[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = False
        else:
            return field
    return None


def _fallback_field(cases, rng, testing, open_shape, irregular_anchor=False):
    units = [_unit(c, rng, compact=True, open_shape=open_shape,
                   irregular_anchor=irregular_anchor and i == 0) for i, c in enumerate(cases)]
    field = np.zeros((19, 30), np.int8)
    positions = [(0, 0), (0, 16), (8, 16), (12, 0)] if testing else [(0, 0), (0, 16), (9, 0), (9, 16)]
    for unit, (r, c) in zip(units, positions):
        h, w = unit.shape
        _require(r + h <= 19 and c + w <= 30 and not np.any(field[r:r + h, c:c + w]), "fallback capacity")
        field[r:r + h, c:c + w] = unit
    return field


def _join_walls(field, rng):
    """Optional wall bridges; accept only if they preserve every chamber."""
    old = _union(_enclosed(field == 1), field.shape)
    for _ in range(3):
        parts = _components(field == 1)
        if len(parts) <= 1:
            break
        i, j = map(int, rng.choice(len(parts), 2, replace=False))
        a = parts[i][int(rng.integers(len(parts[i]))) ]
        b = parts[j][int(rng.integers(len(parts[j]))) ]
        path = np.zeros_like(field, bool)
        path[min(a[0], b[0]):max(a[0], b[0]) + 1, a[1]] = True
        path[b[0], min(a[1], b[1]):max(a[1], b[1]) + 1] = True
        if np.any(path & (old | (field == 2))):
            continue
        candidate = field.copy()
        candidate[path] = 1
        if np.array_equal(_union(_enclosed(candidate == 1), field.shape), old):
            field = candidate
    return field


def _legend(keys, colors, width, rng):
    glyphs = []
    for k in keys:
        a = _compact(k)
        if rng.random() < .5:
            a = a.T.copy()
        if rng.random() < .4 and max(a.shape) < 7:
            axis = int(rng.integers(2))
            at = int(rng.integers(a.shape[axis]))
            b = np.insert(a, at, np.take(a, at, axis=axis), axis=axis)
            if len(_enclosed(b)) == k and max(b.shape) <= 7:
                a = b
        glyphs.append(a)
    gap = int(rng.integers(1, 3))
    total = sum(a.shape[1] for a in glyphs) + gap * (len(keys) - 1) + 2
    if total > 30:
        glyphs, gap = [_compact(k) for k in keys], 1
        total = sum(a.shape[1] for a in glyphs) + len(keys) + 1
    width = max(width, total)
    h = max(a.shape[0] for a in glyphs) + 2
    panel = np.zeros((h, width), np.int8)
    c = int(rng.integers(1, width - total + 2))
    for a, color in zip(glyphs, colors):
        r = int(rng.integers(1, h - a.shape[0]))
        panel[r:r + a.shape[0], c:c + a.shape[1]][a] = color
        c += a.shape[1] + gap
    return panel


def _recipe(rng, split, index):
    if split == 'train' and index == 0:
        return [[1, 1], [0], [1], []], [2, 0], True
    if split == 'train' and index == 1:
        return [[2, 0], [1], [1, 2]], [1, 2], False
    count = int(rng.integers(1, 5))
    if split == 'test':
        main = [[1, 1, 0], [2, 1, 0], [1, 1, 1], [2, 2, 0],
                [1, 1, 1, 1], [2, 1, 1, 0], [2, 2, 1, 0], [2, 2, 1, 1]]
        cases = [list(main[int(rng.integers(len(main)))])]
        rng.shuffle(cases[0])
    else:
        cases = [[int(rng.integers(3)) for _ in range(int(rng.integers(1, 3)))]]
    while len(cases) < count:
        cases.append([int(rng.integers(3)) for _ in range(int(rng.choice([0, 1, 1, 2])))])
    totals = [sum(c) for c in cases]
    size = int(rng.integers(2, 5))
    # Always display the main chamber's class; extra entries may be unused.
    keys = [totals[0]]
    pool = [k for k in range(7) if k not in keys]
    keys += list(map(int, rng.choice(pool, size=size - 1, replace=False)))
    rng.shuffle(keys)
    return cases, keys, bool(rng.random() < .35)


def _make_pair(rng, roles, split, index, rotation):
    bg, wall, query = roles
    cases, keys, opened = _recipe(rng, split, index)
    palette = list(map(int, rng.permutation([c for c in range(10) if c not in roles])))
    # A sentinel background distinguishes a zero-valued dye from empty space.
    glyph_colors = palette[:len(keys)]
    sentinel_colors = list(range(11, 11 + len(keys)))
    panel = _legend(keys, sentinel_colors, int(rng.integers(18, 31)), rng)
    actual_panel = np.full(panel.shape, bg, np.int8)
    for sentinel, color in zip(sentinel_colors, glyph_colors):
        actual_panel[panel == sentinel] = color
    panel = actual_panel
    max_h = 28 - panel.shape[0]
    field = None
    for attempt in range(18):
        compact = attempt >= 12
        units = [_unit(c, rng, compact, opened, split == 'train' and index == 1 and i == 0,
                       split == 'train' and index == 0 and i == 0)
                 for i, c in enumerate(cases)]
        h = max_h if attempt >= 6 else int(rng.integers(max(12, max_h - 6), max_h + 1))
        w = 30 if attempt >= 6 else panel.shape[1]
        field = _pack(units, (h, w), rng)
        if field is not None:
            break
    if field is None:
        field = _fallback_field(cases, rng, split == 'test', opened, split == 'train' and index == 1)
    if rng.random() < .45:
        field = _join_walls(field, rng)
    width = max(field.shape[1], panel.shape[1])
    grid = np.full((panel.shape[0] + 2 + field.shape[0], width), bg, np.int8)
    grid[:panel.shape[0], :panel.shape[1]] = panel
    grid[panel.shape[0]] = wall
    target = grid[panel.shape[0] + 2:, :field.shape[1]]
    target[field == 1], target[field == 2] = wall, query
    inp = np.rot90(grid, rotation).copy()
    return dict(input=inp, output=solve(inp))


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    stable = bool(rng.random() < .3)
    roles = tuple(map(int, rng.permutation(10)[:3]))
    preferred_side = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for attempt in range(32):
                local_roles = roles if stable else tuple(map(int, rng.permutation(10)[:3]))
                # Side changes are shown early, but there is no split-only side.
                rotation = (preferred_side + index) % 4 if split == 'train' and index < 2 else int(rng.integers(4))
                pair = _make_pair(rng, local_roles, split, index, rotation)
                sig = _signature(pair['input'])
                if sig not in seen and not np.array_equal(pair['input'], pair['output']):
                    break
            else:
                # At most 13 prior inputs; 7P2 >=42 visible dye assignments
                # guarantee uniqueness while preserving all geometric evidence.
                parsed = _parse(pair['input'])
                dye_colors = sorted(parsed['mapping'].values())
                available = [c for c in range(10) if c not in local_roles]
                for assignment in permutations(available, len(dye_colors)):
                    inp = pair['input'].copy()
                    for old, new in zip(dye_colors, assignment):
                        inp[pair['input'] == old] = new
                    if _signature(inp) not in seen:
                        pair = dict(input=inp, output=solve(inp))
                        sig = _signature(inp)
                        break
                else:
                    raise AssertionError("finite palette uniqueness exhausted")
            seen.add(sig)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid example counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "dimensions or identity output")
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            p = _parse(inp)
            canonical_output = np.rot90(out, p['rotation'])
            work_out = canonical_output[p['cut'] + 1:]
            domains = _union([r['domain'] for r in p['rooms']], p['work'].shape)
            _require(np.array_equal(canonical_output[:p['cut'] + 1], p['grid'][:p['cut'] + 1]), "key/separator changed")
            _require(np.array_equal(work_out[~domains], p['work'][~domains]), "wall or outside changed")
            for room in p['rooms']:
                dye = p['mapping'].get(room['total'])
                if dye is None:
                    _require(np.all(work_out[room['domain']] == p['background']), "unlisted chamber not emptied")
                    continue
                floor = room['domain'] & ~room['body'] & ~room['holes'] & ~room['halo']
                _require(np.all(work_out[floor] == dye), "wrong chamber floor dye")
                _require(np.all(work_out[room['halo']] == p['background']), "moat must be exterior background")
                _require(np.all(work_out[room['holes']] == p['query']), "original cavities not completely filled")
                for stamp in room['stamps']:
                    color = dye if stamp['count'] else p['query']
                    _require(np.all(work_out[stamp['body']] == color), "whole stamp body misclassified")
            _require(np.array_equal(solve(inp), out), "input-only solve mismatch")
            if split == 'train':
                training.append(p)
    # Pixel-derived teaching witnesses; no mandatory slot, size or palette.
    rooms = [(p, r) for p in training for r in p['rooms']]
    _require(any(r['total'] in p['mapping'] and sum(s['count'] > 0 for s in r['stamps']) >= 2
                 and any(s['count'] not in p['mapping'] for s in r['stamps']) for p, r in rooms), "missing group-only match")
    _require(any(r['total'] not in p['mapping'] and len(r['stamps']) >= 2
                 and all(s['count'] in p['mapping'] for s in r['stamps']) for p, r in rooms), "missing individually listed/group-unlisted contrast")
    _require(any(r['total'] in p['mapping'] and any(s['count'] == 0 and _separated_runs(s['local']) for s in r['stamps'])
                 for p, r in rooms), "missing active open solid")
    _require(any(not r['stamps'] and r['total'] in p['mapping'] for p, r in rooms), "missing active empty chamber")
    _require(any(r['total'] in p['mapping'] and np.any(r['halo'] & ~_orth_dilate(r['body']))
                 for p, r in rooms), "training does not distinguish diagonal from orthogonal moats")
    _require(any(r['total'] in p['mapping'] and any(s['count'] >= 2 for s in r['stamps']) for p, r in rooms), "missing multi-hole stencil")
    _require(any(r['total'] in p['mapping'] and any(not _trim(h).all() for s in r['stamps']
                 for h in _enclosed(s['local'])) for p, r in rooms), "missing active irregular cavity")
    _require(any(not np.array_equal(_cast(p, omit_a=True), _cast(p)) for p in training), "A ablation has no effect")
    _require(any(not np.array_equal(_cast(p, omit_b=True), _cast(p)) for p in training), "B ablation has no effect")


def diversity_features(pair):
    p = _parse(pair['input'])
    return dict(shape=list(np.asarray(pair['input']).shape), side=p['rotation'],
                roles=[p[k] for k in ('background', 'wall', 'query')],
                keys=sorted(p['mapping']), chambers=len(p['rooms']), stamps=len(p['stamps']),
                group_sizes=[len(r['stamps']) for r in p['rooms']], totals=[r['total'] for r in p['rooms']],
                erased=sum(r['total'] not in p['mapping'] for r in p['rooms']),
                irregular_chambers=sum(not _trim(r['domain']).all() for r in p['rooms']),
                wall_components=len(_components(p['work'] == p['wall'])),
                irregular_holes=sum(not _trim(h).all() for s in p['stamps'] for h in _enclosed(s['local'])),
                clipped_moats=sum(bool(np.any(_dilate(s['body']) & (p['work'] == p['wall']))) for s in p['stamps']),
                stamp_shapes=[list(s['local'].shape) for s in p['stamps']])


def _regressions():
    # Hand-built geometry and direct expected cells, not saved generated labels.
    g = np.zeros((18, 23), np.int8)
    g[1:4, 1:6] = 2; g[2, 2] = g[2, 4] = 0  # key: two holes -> red
    g[1:4, 9:12] = 3  # zero holes -> green
    g[5] = 8
    g[7:16, 1:22] = 8
    g[8:15, 2:21] = 0
    g[10:13, 4:7] = 5; g[11, 5] = 0
    g[10:13, 13:16] = 5; g[11, 14] = 0
    out = solve(g)
    _require(out[8, 10] == 2 and out[10, 4] == 2, "aggregate dye and whole hollow body")
    _require(out[11, 5] == 5 and out[11, 14] == 5, "cavity islands")
    _require(out[9, 3] == 0 and out[9, 4] == 0, "diagonal and orthogonal exterior moat")
    _require(out[7, 4] == 8 and out[6, 4] == 0, "wall and exterior preservation")
    # One missing wall cell of a stamp changes its topology before casting.
    opened = g.copy(); opened[10, 5] = 0
    _require(np.all(solve(opened)[8:15, 2:21] == 0), "one opened stamp makes group total unlisted")
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(g, k)), np.rot90(out, k)), "rotation equivariance")
    perm = np.asarray([7, 6, 9, 1, 4, 0, 2, 5, 3, 8])
    _require(np.array_equal(solve(perm[g]), perm[out]), "visible palette equivariance")
    # Four-connected topology is explicit: diagonal escape still enclosed.
    diagonal = np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 1]], bool)
    _require(len(_enclosed(diagonal)) == 1, "diagonal escape convention")
    diagonal[0, 1] = False
    _require(not _enclosed(diagonal), "orthogonal door convention")
    # Five stamps, including a three-hole body, exceed the usual sampler caps.
    # The visible aggregate key is six; solving still uses only these pixels.
    larger = np.zeros((30, 30), np.int8)
    prototype = _compact(6)
    larger[1:6, 1:8][prototype] = 2
    larger[1:4, 12:15] = 3
    larger[7] = 8
    larger[10:29, 1:29] = 8
    larger[11:28, 2:28] = 0
    for k, r, c in ((3, 13, 4), (1, 13, 15), (1, 13, 22), (1, 22, 4), (0, 22, 15)):
        stamp = np.ones((3, 2 * k + 1), bool) if k else np.ones((3, 3), bool)
        if k:
            stamp[1, 1:2 * k:2] = False
        larger[r:r + stamp.shape[0], c:c + stamp.shape[1]][stamp] = 5
    cast = solve(larger)
    _require(cast[12, 12] == 2 and cast[13, 4] == 2 and cast[14, 5] == 5,
             "generalized aggregate/hollow casting")
    _require(cast[22, 15] == 5 and cast[21, 14] == 0, "generalized solid/moat casting")


def self_test():
    _regressions()
    features, silhouettes, signatures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        repeated = generate_puzzle(seed)
        _require(all(np.array_equal(a[k], b[k]) for split in puzzle for a, b in zip(puzzle[split], repeated[split])
                     for k in ('input', 'output')), "nondeterminism")
        local = []
        for split in puzzle:
            for pair in puzzle[split]:
                f = diversity_features(pair)
                f['test'] = split == 'test'
                features.append(f); local.append(f)
                signatures.add(_signature(pair['input']))
                p = _parse(pair['input'])
                for stamp in p['stamps']:
                    silhouettes.add(_signature(stamp['local']))
        _require(max(max(f['group_sizes']) for f in local if f['test']) >= 3, "missing test aggregation shift")
        _require(max(max(f['group_sizes']) for f in local if not f['test']) <= 2, "train/test shift inverted")
        _require(len({str(f['totals']) for f in local}) >= 2, "unchanging chamber populations")
    _require(len(signatures) == 210, "cross-seed duplicate inputs")
    _require(len({tuple(f['shape']) for f in features}) >= 25, "insufficient size variation")
    _require(len(silhouettes) >= 35, "insufficient stamp geometry variation")
    _require(len({tuple(f['roles']) for f in features}) >= 40, "insufficient palette roles")
    _require(len({tuple(f['keys']) for f in features}) >= 15, "insufficient topology keys")
    _require({f['side'] for f in features} == {0, 1, 2, 3}, "missing key side")
    _require({f['chambers'] for f in features} == {1, 2, 3, 4}, "missing chamber counts")
    for name in ('irregular_chambers', 'irregular_holes', 'clipped_moats', 'erased'):
        _require(sum(bool(f[name]) for f in features) >= 10, "missing variation: " + name)
    _require(any(f['wall_components'] < f['chambers'] for f in features), "no multi-chamber wall assembly")
    rng = np.random.default_rng(92)
    for testing, cases in ((False, [[1, 1], [0], [1], [2, 0]]),
                           (True, [[2, 2, 1, 1], [1, 2], [2, 0], [1, 1]])):
        field = _fallback_field(cases, rng, testing, True)
        _require(len(_enclosed(field == 1)) == len(cases), "fallback chamber population")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(81 + nt * 7 + nq, nt, nq)
            _require(len(puzzle['train']) == nt and len(puzzle['test']) == nq, "count request not honored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject corruption of every material zone wherever it is visible.
    puzzle = generate_puzzle(31, 2, 1)
    for name in ('halo', 'holes', 'body', 'floor', 'erased', 'wall', 'key', 'solid'):
        chosen = None
        for split in puzzle:
            for index, pair in enumerate(puzzle[split]):
                p = _parse(pair['input'])
                active = [r for r in p['rooms'] if r['total'] in p['mapping']]
                z = np.zeros(p['work'].shape, bool)
                if name in ('halo', 'holes', 'body'):
                    z = _union([r[name] for r in active], z.shape)
                elif name == 'floor':
                    z = _union([r['domain'] & ~r['body'] & ~r['holes'] & ~r['halo'] for r in active], z.shape)
                elif name == 'erased':
                    z = _union([r['domain'] for r in p['rooms'] if r['total'] not in p['mapping']], z.shape)
                elif name == 'solid':
                    z = _union([s['body'] for r in active for s in r['stamps'] if not s['count']], z.shape)
                elif name == 'wall':
                    z = p['work'] == p['wall']
                zone = np.zeros(p['grid'].shape, bool)
                zone[p['cut'] + 1:] = z
                if name == 'key':
                    zone[:p['cut']] = p['grid'][:p['cut']] != p['background']
                if zone.any():
                    chosen = split, index, p, zone
                    break
            if chosen is not None:
                break
        _require(chosen is not None, "no corruption witness for " + name)
        split, index, p, zone = chosen
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzle[s]] for s in puzzle}
        canonical = np.rot90(bad[split][index]['output'], p['rotation'])
        r, c = np.argwhere(zone)[0]
        canonical[r, c] = (int(canonical[r, c]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted material zone accepted: " + name)
    return dict(seeds=42, pairs=len(features), shapes=len({tuple(f['shape']) for f in features}),
                stamp_silhouettes=len(silhouettes), key_sets=len({tuple(f['keys']) for f in features}))
