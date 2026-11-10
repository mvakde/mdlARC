"""Sealed chamber census: register transparent sheets, then count enclosed glyphs.

Only NumPy and the standard library are used. solve() reads one input only.
"""
from __future__ import annotations

from collections import deque
import numpy as np


STEPS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
STEPS8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIAMOND = {(0, 1), (1, 0), (1, 2), (2, 1)}
SQUARE = {(0, 0), (0, 1), (1, 0), (1, 1)}


def _components(cells, diagonal=True):
    remaining = set(cells)
    parts = []
    steps = STEPS8 if diagonal else STEPS4
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _shift(cells, dr, dc=0):
    return {(r + dr, c + dc) for r, c in cells}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('Expected a 1..30 integer grid with colors 0..9')
    return a.astype(np.int8, copy=False)


def _chart_info(panel, field):
    positions = [tuple(map(int, q)) for q in np.argwhere(panel != field)]
    colors = [int(panel[q]) for q in positions]
    if not 2 <= len(positions) <= 4 or len(set(colors)) != len(colors):
        return None
    if len({c for r, c in positions}) == 1:
        positions.sort()
        if any(b[0] - a[0] < 2 for a, b in zip(positions, positions[1:])):
            return None
        inset = positions[0][1]
        if 2 * inset == panel.shape[1] - 1:
            return None
        direction = (0, 1 if 2 * inset < panel.shape[1] - 1 else -1)
    elif len({r for r, c in positions}) == 1:
        positions.sort(key=lambda q: q[1])
        if any(b[1] - a[1] < 2 for a, b in zip(positions, positions[1:])):
            return None
        inset = positions[0][0]
        if 2 * inset == panel.shape[0] - 1:
            return None
        direction = (1 if 2 * inset < panel.shape[0] - 1 else -1, 0)
    else:
        return None
    return dict(panel=panel.copy(), positions=positions,
                keys=tuple(int(panel[q]) for q in positions), direction=direction)


def _bar(panel, marker, wall, field, axis):
    cells = [tuple(map(int, q)) for q in np.argwhere(panel == marker)]
    if not 2 <= len(cells) <= 5:
        return None
    oriented = {(q[axis], q[1-axis]) for q in cells}
    t, b, l, r = _bbox(oriented)
    if t != b or r-l+1 != len(cells):
        return None
    p = panel.T if axis else panel
    if not 0 < t < p.shape[0]-1:
        return None
    above, below = p[t-1, l:r+1], p[t+1, l:r+1]
    if np.all(above == wall) and np.all(below == field):
        sign = -1
    elif np.all(above == field) and np.all(below == wall):
        sign = 1
    else:
        return None
    ordinary = {}
    for y, x in np.argwhere((p != field) & (p != marker)):
        ordinary[(int(y)-t, int(x)-l)] = int(p[y, x])
    return dict(origin=(t, l), sign=sign, length=len(cells), cells=ordinary)


def _parse(input_grid):
    """Recover backgrounds, sheet ownership, chart, and registration from pixels."""
    grid = _grid(input_grid)
    candidates = []
    for background in map(int, np.unique(grid)):
        rects = _components(map(tuple, np.argwhere(grid != background)), False)
        if len(rects) != 3:
            continue
        cards, bounds, fields = [], [], []
        for cells in rects:
            t, b, l, r = _bbox(cells)
            if len(cells) != (b-t+1)*(r-l+1) or min(b-t+1, r-l+1) < 3:
                break
            p = grid[t:b+1, l:r+1]
            f = int(p[0, 0])
            if not all(np.all(e == f) for e in (p[0], p[-1], p[:, 0], p[:, -1])):
                break
            cards.append(p)
            bounds.append((t, b, l, r))
            fields.append(f)
        if len(cards) != 3 or len(set(fields)) != 1:
            continue
        field = fields[0]
        for chart_index in range(3):
            chart = _chart_info(cards[chart_index], field)
            if chart is None:
                continue
            sheets = [cards[i] for i in range(3) if i != chart_index]
            common = set(map(int, np.unique(sheets[0]))) & set(map(int, np.unique(sheets[1])))
            support = common - {background, field} - set(chart['keys'])
            for marker in sorted(support):
                for wall in sorted(support - {marker}):
                    for axis in (0, 1):
                        bars = [_bar(p, marker, wall, field, axis) for p in sheets]
                        if any(x is None for x in bars):
                            continue
                        if bars[0]['length'] != bars[1]['length'] or bars[0]['sign'] == bars[1]['sign']:
                            continue
                        bars.sort(key=lambda x: x['sign'])
                        try:
                            assembled = _assemble(bars, wall)
                        except ValueError:
                            continue
                        candidates.append(dict(background=background, field=field,
                                               marker=marker, wall=wall, axis=axis,
                                               bars=bars, chart=chart, bounds=bounds,
                                               chart_index=chart_index, assembled=assembled))
    if len(candidates) != 1:
        raise ValueError(f'Expected one visible interpretation, found {len(candidates)}')
    return candidates[0]


def _assemble(bars, wall, seam=-1, blend='union'):
    a = dict(bars[0]['cells'])
    b = {(r+seam, c): color for (r, c), color in bars[1]['cells'].items()}
    if any(a[p] != b[p] for p in a.keys() & b.keys()):
        raise ValueError('Different colors collide after registration')
    if blend == 'anchor':
        return a
    if blend == 'mover':
        return b
    result = dict(a)
    result.update(b)
    if blend == 'xor':
        for p in a.keys() & b.keys():
            del result[p]
    return result


def _enclosed(walls, occupied, diagonal=False):
    """Four-neighbor exterior flood fill; token pixels are traversable space."""
    if not walls:
        return set(), []
    t, b, l, r = _bbox(walls | set(occupied))
    t, b, l, r = t-1, b+1, l-1, r+1
    exterior, todo = {(t, l)}, deque([(t, l)])
    while todo:
        y, x = todo.popleft()
        for dy, dx in (STEPS8 if diagonal else STEPS4):
            q = (y+dy, x+dx)
            if t <= q[0] <= b and l <= q[1] <= r and q not in walls and q not in exterior:
                exterior.add(q)
                todo.append(q)
    holes = {(y, x) for y in range(t, b+1) for x in range(l, r+1)} - walls - exterior
    return holes, _components(holes, False)


def _census(assembled, wall, keys, diagonal=True, measure='objects', space_diagonal=False):
    walls = {q for q, c in assembled.items() if c == wall}
    enclosed, holes = _enclosed(walls, assembled, space_diagonal)
    counts = {}
    for color in keys:
        tokens = {q for q, c in assembled.items() if c == color}
        objects = _components(tokens, diagonal)
        eligible = [part for part in objects if part <= enclosed]
        if measure == 'area':
            counts[color] = sum(len(p) for p in eligible)
        elif measure == 'rooms':
            counts[color] = sum(bool(hole & tokens) for hole in holes)
        else:
            counts[color] = len(eligible)
    return counts, enclosed, holes


def _render_chart(meta, counts, stride=2, add_key=False):
    chart = meta['chart']
    result = np.full(chart['panel'].shape, meta['field'], np.int8)
    dr, dc = chart['direction']
    for (r, c), color in zip(chart['positions'], chart['keys']):
        n = counts.get(color, 0) + int(add_key)
        for i in range(n):
            rr, cc = r + stride*i*dr, c + stride*i*dc
            if not 0 < rr < result.shape[0]-1 or not 0 < cc < result.shape[1]-1:
                return None
            result[rr, cc] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate sheets and count wholly enclosed eight-connected color objects."""
    meta = _parse(input_grid)
    counts, _, _ = _census(meta['assembled'], meta['wall'], meta['chart']['keys'])
    output = _render_chart(meta, counts)
    if output is None:
        raise ValueError('The visible chart has insufficient tally capacity')
    return output


def _room_network(rng, witness, layout=None):
    """Rectangular mosaics with different partitions in their horizontal bands."""
    if witness:
        rows, columns = 2, [2, 2]
        first_widths = [int(rng.integers(5, 7)), int(rng.integers(5, 7))]
    else:
        rows = int(rng.choice([1, 2, 2]))
        first_n = int(rng.integers(2, 4))
        if layout == 'two_bands':
            rows, first_n = 2, 2
        elif layout == 'three_in_a_row':
            rows, first_n = 1, 3
        columns = [first_n]
        first_widths = [int(rng.integers(3, 7)) for _ in range(first_n)]
    width = 1 + sum(x+1 for x in first_widths)
    if not witness and rows == 2:
        columns.append(2 if layout == 'two_bands' else int(rng.integers(1, min(3, (width-1)//4)+1)))
    heights = [int(rng.integers(3, 5)) for _ in range(rows)]
    walls, rooms = set(), []
    top = 0
    for row in range(rows):
        n = columns[row]
        if row == 0:
            widths = first_widths
        elif witness:
            # Both lower rooms have room for two small glyphs.
            extra = width-1-2*6
            left_extra = int(rng.integers(extra+1))
            widths = [5+left_extra, 5+extra-left_extra]
        else:
            widths = [3]*n
            for _ in range(width-1-4*n):
                widths[int(rng.integers(n))] += 1
        left = 0
        bottom = top+heights[row]+1
        for inner_width in widths:
            right = left+inner_width+1
            interior = {(y, x) for y in range(top+1, bottom) for x in range(left+1, right)}
            border = {(y, x) for y in range(top, bottom+1) for x in range(left, right+1)
                      if y in (top, bottom) or x in (left, right)}
            walls |= border
            rooms.append(dict(bounds=(top, bottom, left, right), inside=interior, row=row))
            left = right
        top = bottom
    # Cut off exterior corners. The diagonal wall tips still seal space under
    # four-neighbor flooding, providing a visible digital-topology witness.
    corners = [(0, 0), (0, width-1), (top, 0), (top, width-1)]
    removed = {corners[0]} if witness == 1 else set()
    removed.update(corner for corner in corners if rng.random() < .30)
    walls.difference_update(removed)
    # Visible doorways produce a graph whose exterior reachability supplies
    # construction labels, separately from solve's pixel flood fill.
    owner = {q: i for i, room in enumerate(rooms) for q in room['inside']}
    doors = []
    for y, x in sorted(walls):
        for dy, dx in ((1, 0), (0, 1)):
            a, b = (y-dy, x-dx), (y+dy, x+dx)
            if a in walls or b in walls:
                continue
            ia, ib = owner.get(a, -1), owner.get(b, -1)
            if ia != ib and (ia >= 0 or ib >= 0):
                doors.append(((y, x), ia, ib))
    opened = []
    last = len(rooms)-1
    exterior_doors = [d for d in doors if set(d[1:]) == {-1, last}]
    d = exterior_doors[int(rng.integers(len(exterior_doors)))]
    opened.append(d)
    if witness == 2:
        # Connect upper-right to lower-right, then to the exterior: an indirect
        # leak changes the upper room despite its intact outer boundary.
        options = [d for d in doors if set(d[1:]) == {1, last}]
        opened.append(options[int(rng.integers(len(options)))])
    elif not witness:
        internal = [d for d in doors if min(d[1:]) >= 0]
        for index in rng.permutation(len(internal))[:int(rng.integers(0, min(3, len(internal))+1))]:
            opened.append(internal[int(index)])
        if rng.random() < .35:
            ext = [d for d in doors if -1 in d[1:]]
            opened.append(ext[int(rng.integers(len(ext)))])
    for q, _, _ in opened:
        walls.discard(q)
    graph = {i: set() for i in range(-1, len(rooms))}
    for _, a, b in opened:
        graph[a].add(b)
        graph[b].add(a)
    accessible, todo = {-1}, [-1]
    while todo:
        for node in sorted(graph[todo.pop()] - accessible):
            accessible.add(node)
            todo.append(node)
    return walls, rooms, accessible, opened


def _glyph(rng):
    mode = int(rng.integers(7))
    if mode == 0:
        cells = DIAMOND
    elif mode == 1:
        cells = SQUARE
    elif mode == 2:
        cells = {(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)}
    elif mode == 3:
        cells = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
    elif mode == 4:
        cells = {(0, x) for x in range(int(rng.integers(1, 4)))}
    else:
        cells = {(1, 1)}
        for _ in range(int(rng.integers(0, 7))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in STEPS8
                               if 0 <= r+dr <= 2 and 0 <= c+dc <= 2} - cells)
            if frontier:
                cells.add(frontier[int(rng.integers(len(frontier)))])
    cells = set(cells)
    for _ in range(int(rng.integers(4))):
        cells = {(-c, r) for r, c in cells}
    t, _, l, _ = _bbox(cells)
    return _shift(cells, -t, -l)


def _place_glyph(rng, room, shape, used):
    t, b, l, r = room['bounds']
    sh, sw = 1+max(y for y, x in shape), 1+max(x for y, x in shape)
    halo = {(y+dy, x+dx) for y, x in used for dy, dx in ((0, 0),) + STEPS8}
    options = [(y, x) for y in range(t+1, b-sh+1) for x in range(l+1, r-sw+1)
               if _shift(shape, y, x).isdisjoint(halo)]
    if not options:
        return None
    y, x = options[int(rng.integers(len(options)))]
    return _shift(shape, y, x)


def _populate(rng, rooms, accessible, keys, ignored, witness):
    objects, used = [], set()

    def put(room_id, color, shape, fixed=None):
        if fixed is None:
            cells = _place_glyph(rng, rooms[room_id], shape, used)
        else:
            cells = _shift(shape, *fixed)
            if not cells <= rooms[room_id]['inside']:
                return False
            halo = {(y+dy, x+dx) for y, x in used for dy, dx in ((0, 0),) + STEPS8}
            if not cells.isdisjoint(halo):
                return False
        if cells is None:
            return False
        used.update(cells)
        objects.append((color, cells, room_id))
        return True

    if witness == 1:
        t, _, l, _ = rooms[0]['bounds']
        assert put(0, keys[0], DIAMOND, (t+1, l+1))
        assert put(0, keys[0], {(0, 0)}, (t+1, l+5))
        assert put(2, keys[0], SQUARE)
        assert put(1, keys[1], _glyph(rng))
        assert put(3, keys[0], _glyph(rng))
        assert put(2, ignored, {(0, 0)})
    elif witness == 2:
        assert put(0, keys[0], _glyph(rng))
        assert put(1, keys[1], _glyph(rng))
        assert put(3, keys[1], {(0, 0)})
        assert put(2, ignored, _glyph(rng))
        if len(keys) == 3:
            assert put(2, keys[2], {(0, 0)})
        put(0, keys[0], {(0, 0)})
    else:
        # Every requested color is visible, but may have a zero sealed count.
        for color in keys:
            for _ in range(30):
                room_id = int(rng.integers(len(rooms)))
                if put(room_id, color, _glyph(rng)):
                    break
            else:
                return None
        for _ in range(int(rng.integers(2, 10))):
            color = int(rng.choice(list(keys) + ([ignored] if rng.random() < .5 else [])))
            room_id = int(rng.integers(len(rooms)))
            if not put(room_id, color, _glyph(rng)):
                put(room_id, color, {(0, 0)})
    counts = {c: sum(color == c and room not in accessible for color, _, room in objects) for c in keys}
    return objects, counts


def _sheet(rng, ordinary, marker_cells, palette):
    bg, field, wall, marker = palette[:4]
    t, b, l, r = _bbox(set(ordinary) | marker_cells)
    # The complete blank border identifies a support, even when its interior
    # wall or colored glyph pixels have multiple disconnected components.
    pads = [int(rng.integers(1, 3)) for _ in range(4)]
    grid = np.full((b-t+1+pads[0]+pads[1], r-l+1+pads[2]+pads[3]), field, np.int8)
    for (y, x), color in ordinary.items():
        grid[y-t+pads[0], x-l+pads[2]] = color
    for y, x in marker_cells:
        grid[y-t+pads[0], x-l+pads[2]] = marker
    return grid


def _pack(rng, cards, background, prefer=None):
    """Pack three rectangles in a row, column, or two shelves, then pad."""
    layouts = []
    # Candidate order and placement are independent of semantic card roles.
    import itertools
    gap = int(rng.integers(1, 4))
    for order in itertools.permutations(range(3)):
        a, b, c = order
        ha, wa = cards[a].shape
        hb, wb = cards[b].shape
        hc, wc = cards[c].shape
        options = [([(0, 0), (ha+gap, 0), (ha+hb+2*gap, 0)], 'stack'),
                   ([(0, 0), (0, wa+gap), (0, wa+wb+2*gap)], 'row'),
                   ([(0, 0), (ha+gap, 0), (0, max(wa, wb)+gap)], 'side'),
                   ([(0, 0), (0, wa+gap), (max(ha, hb)+gap, 0)], 'shelf')]
        for positions, mode in options:
            positions = dict(zip(order, positions))
            h = max(positions[i][0]+cards[i].shape[0] for i in range(3))
            w = max(positions[i][1]+cards[i].shape[1] for i in range(3))
            if max(h, w) <= 30:
                layouts.append((positions, h, w, mode))
    if not layouts:
        return None
    selected = [x for x in layouts if x[3] == prefer] if prefer else layouts
    if not selected:
        return None
    positions, h, w, mode = selected[int(rng.integers(len(selected)))]
    top = int(rng.integers(min(3, 31-h)))
    bottom = int(rng.integers(min(3, 31-h-top)))
    left = int(rng.integers(min(3, 31-w)))
    right = int(rng.integers(min(3, 31-w-left)))
    result = np.full((h+top+bottom, w+left+right), background, np.int8)
    for i, (y, x) in positions.items():
        card = cards[i]
        result[y+top:y+top+card.shape[0], x+left:x+left+card.shape[1]] = card
    return result


def _make_pair(rng, palette, witness, length, rotation, reflect, regime, is_test, used_inputs):
    bg, field, wall, marker = palette[:4]
    key_count = 2 if witness == 1 else (3 if witness == 2 else int(rng.integers(2, 5)))
    keys = tuple(palette[4:4+key_count])
    ignored = palette[6] if witness == 1 else int(rng.choice(palette[4+key_count:9]))
    for attempt in range(150):
        layout = ('three_in_a_row' if is_test else 'two_bands') if regime == 'geometry' else None
        walls, rooms, accessible, doors = _room_network(rng, witness, layout)
        try:
            populated = _populate(rng, rooms, accessible, keys, ignored, witness)
        except AssertionError:
            continue
        if populated is None:
            continue
        objects, counts = populated
        # Keep generated examples informative: at least one tally is nonempty.
        # The input-only rule still handles an all-zero census without changes.
        if not any(counts.values()):
            continue
        # Census labels come from room graph + construction object identities.
        # No token overlaps another token or a wall in the finished network.
        network_height = _bbox(walls)[1]+1
        if witness == 1:
            seam = 2  # splits the mandatory diamond between the two sheets
        else:
            seam = int(rng.integers(2, network_height-1))
        xoffset = length+2
        full = {(r, c+xoffset): wall for r, c in walls}
        for color, cells, room in objects:
            for r, c in cells:
                full[(r, c+xoffset)] = color
        a = {q: color for q, color in full.items() if q[0] < seam}
        b = {q: color for q, color in full.items() if q[0] >= seam}
        # Shared wall strokes test union instead of cancellation. Copy an
        # exterior side cell at the seam, where XOR would open a sealed room.
        if witness == 1 or rng.random() < .60:
            duplicate = (seam-1, xoffset)
            if duplicate in a and a[duplicate] == wall:
                b[duplicate] = wall
        if not witness and rng.random() < .40:
            duplicate = (seam, xoffset+_bbox(walls)[3])
            if duplicate in b and b[duplicate] == wall:
                a[duplicate] = wall
        for x in range(length):
            a[(seam-1, x)] = wall
            b[(seam, x)] = wall
        guide_a = {(seam, x) for x in range(length)}
        guide_b = {(seam-1, x) for x in range(length)}
        # Mover's raw marker row is one farther down. Its ordinary cells and
        # guide are shifted together before unrelated input staging.
        sheets = [_sheet(rng, a, guide_a, palette),
                  _sheet(rng, {(r+1, c): color for (r, c), color in b.items()},
                         _shift(guide_b, 1), palette)]
        rows = [int(rng.integers(1, 3))]
        for _ in range(key_count-1):
            rows.append(rows[-1]+int(rng.integers(2, 4)))
        ph = rows[-1]+int(rng.integers(2, 4))
        inset = int(rng.integers(1, 3))
        pw = max(2*inset+3, inset+2*(max(counts.values())-1)+int(rng.integers(2, 5)))
        chart = np.full((ph, pw), field, np.int8)
        output = chart.copy()
        for row, color in zip(rows, keys):
            chart[row, inset] = color
            for i in range(counts[color]):
                output[row, inset+2*i] = color
        cards = sheets+[chart]
        preferred = 'side' if regime == 'placement' and is_test else ('stack' if regime == 'placement' else None)
        grid = _pack(rng, cards, bg, preferred)
        if grid is None:
            continue
        if reflect:
            grid, output = np.fliplr(grid), np.fliplr(output)
        grid, output = np.rot90(grid, rotation).copy(), np.rot90(output, rotation).copy()
        if _key(grid) in used_inputs:
            continue
        try:
            meta = _parse(grid)
            assert np.array_equal(solve(grid), output)
        except (ValueError, AssertionError):
            continue
        if witness == 1:
            # These witnesses must survive the final rendered/input-only parse.
            actual = _census(meta['assembled'], wall, meta['chart']['keys'])[0]
            orth = _census(meta['assembled'], wall, meta['chart']['keys'], False)[0]
            chamber = _census(meta['assembled'], wall, meta['chart']['keys'], measure='rooms')[0]
            if actual == orth or actual == chamber:
                continue
        used_inputs.add(_key(grid))
        return {'input': grid, 'output': output}
    raise RuntimeError('Bounded sheet construction exhausted')


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)))
    stable_palette = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    mixed = bool(rng.integers(2))
    regime = str(rng.choice(['registration', 'geometry', 'placement']))
    puzzle, used = {'train': [], 'test': []}, set()
    ignored_witness = None
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            train = split == 'train'
            witness = index+1 if train and index < 2 else 0
            p = palette if stable_palette else tuple(map(int, rng.permutation(10)))
            if witness == 1:
                ignored_witness = p[6]
            elif witness == 2 and not stable_palette:
                p = list(p)
                old = p.index(ignored_witness)
                p[6], p[old] = p[old], p[6]
                p = tuple(p)
            if regime == 'registration':
                length = (2 if index % 2 == 0 else 4) if train else 3
            else:
                length = int(rng.integers(2, 6))
            k = (rotation+index % 2) % 4 if mixed else rotation
            pair = _make_pair(rng, p, witness, length, k, bool(rng.integers(2)), regime, not train, used)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternatives(meta):
    """Finite, explicit shortcut audit. None means an over-capacity wrong tally."""
    keys, wall, bars = meta['chart']['keys'], meta['wall'], meta['bars']
    assembled = meta['assembled']
    counts = _census(assembled, wall, keys)[0]
    raw = {c: sum(len(_components({q for q, color in bar['cells'].items() if color == c}))
                  for bar in bars) for c in keys}
    local = {c: 0 for c in keys}
    for bar in bars:
        for c, n in _census(bar['cells'], wall, keys)[0].items():
            local[c] += n
    yield 'raw_objects', _render_chart(meta, raw)
    yield 'separate_enclosures', _render_chart(meta, local)
    yield 'four_connected_glyphs', _render_chart(meta, _census(assembled, wall, keys, False)[0])
    yield 'eight_connected_space', _render_chart(meta, _census(assembled, wall, keys, space_diagonal=True)[0])
    yield 'colored_area', _render_chart(meta, _census(assembled, wall, keys, measure='area')[0])
    yield 'chamber_presence', _render_chart(meta, _census(assembled, wall, keys, measure='rooms')[0])
    yield 'include_key', _render_chart(meta, counts, add_key=True)
    yield 'solid_tally', _render_chart(meta, counts, stride=1)
    for seam in (0, 1):
        try:
            a = _assemble(bars, wall, seam)
            out = _render_chart(meta, _census(a, wall, keys)[0])
        except ValueError:
            out = None
        yield f'seam_{seam}', out
    for blend in ('anchor', 'mover', 'xor'):
        a = _assemble(bars, wall, blend=blend)
        yield blend, _render_chart(meta, _census(a, wall, keys)[0])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, eliminated, training, test_meta = set(), set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            meta = _parse(inp)
            counts, enclosed, holes = _census(meta['assembled'], meta['wall'], meta['chart']['keys'])
            assert np.array_equal(out, _render_chart(meta, counts))
            # Full union conserves every ordinary cell except duplicate pixels.
            a, b = meta['bars'][0]['cells'], {(r-1, c): color for (r, c), color in meta['bars'][1]['cells'].items()}
            assert len(meta['assembled']) == len(a)+len(b)-len(a.keys() & b.keys())
            assert meta['marker'] not in set(meta['assembled'].values())
            assert meta['wall'] not in meta['chart']['keys']
            if split == 'train':
                training.append((meta, counts))
                for name, wrong in _alternatives(meta):
                    if wrong is None or not np.array_equal(wrong, out):
                        eliminated.add(name)
            else:
                test_meta.append(meta)
    required = {'raw_objects', 'separate_enclosures', 'four_connected_glyphs', 'eight_connected_space', 'colored_area',
                'chamber_presence', 'include_key', 'solid_tally', 'seam_0', 'seam_1', 'anchor', 'mover', 'xor'}
    assert required <= eliminated, f'Training lacks witnesses against {required-eliminated}'
    assert any(0 in counts.values() for _, counts in training), 'Teach clearing a zero-count key'
    assert any(max(counts.values()) >= 3 for _, counts in training)
    assert any(any(c not in m['chart']['keys'] and c != m['wall']
                   for c in m['assembled'].values()) for m, _ in training)
    assert any((set(a['assembled'].values()) - set(a['chart']['keys']) - {a['wall']}) & set(b['chart']['keys'])
               for a, _ in training for b, _ in training), 'Teach local key membership'
    assert {m['axis'] for m in test_meta} <= {m['axis'] for m, _ in training}


def self_test():
    """Bounded 42-seed semantic/diversity test, plus interface and mutation checks."""
    signatures, shapes, palettes, lengths, axes = set(), set(), set(), set(), set()
    networks, counts_seen, dimensions, vocabularies = set(), set(), set(), set()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed == 0:
            first = puzzle
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                inp = pair['input'].copy()
                assert np.array_equal(solve(inp), pair['output'])
                assert np.array_equal(inp, pair['input']), 'solve must not mutate its input'
                m = _parse(inp)
                ns, _, holes = _census(m['assembled'], m['wall'], m['chart']['keys'])
                shapes.add((inp.shape, pair['output'].shape))
                dimensions.add(inp.shape)
                palettes.add((m['background'], m['field'], m['wall'], m['marker']))
                lengths.add(m['bars'][0]['length'])
                axes.add(m['axis'])
                counts_seen.update(ns.values())
                walls = {q for q, c in m['assembled'].items() if c == m['wall']}
                t, _, l, _ = _bbox(walls)
                networks.add(tuple(sorted(_shift(walls, -t, -l))))
                for color in m['chart']['keys']:
                    for part in _components({q for q, c in m['assembled'].items() if c == color}):
                        t, _, l, _ = _bbox(part)
                        vocabularies.add(tuple(sorted(_shift(part, -t, -l))))
    assert len(signatures) == 42
    assert len(shapes) >= 60 and len(dimensions) >= 40
    assert len(networks) >= 100 and len(vocabularies) >= 30
    assert len(palettes) >= 35 and lengths == {2, 3, 4, 5} and axes == {0, 1}
    assert {0, 1, 2, 3} <= counts_seen
    again = generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in ('train', 'test')
               for a, b in zip(first[s], again[s]) for k in ('input', 'output'))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(91+nt*5+nq, nt, nq)
            assert (len(p['train']), len(p['test'])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('A corrupted census was accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds and every legal explicit count combination')
