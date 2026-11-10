"""Template docking in shape-dependent configuration spaces.

Only NumPy and the standard library are used. The inventory is a library;
border coordinates are launch centers, not final stamp centers.
"""
from collections import deque
import copy
import json
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 783212213, 445602392, 610812360, 755165300, 226828893,
              978071456, 633327841, 815224175, 985265487, 617153431,
              832055800, 214761672, 566282312, 169796711, 274900760,
              672781685, 796029480, 436933559, 468032532, 932289388,
              385679985, 290663159, 486802007, 795954363, 926928187,
              869801853, 335235999, 511759766, 158626829, 642538280,
              327304607, 296538615, 544839249, 599215873, 241632252,
              462889016, 613733267, 208120557, 333768857, 732784417,
              764151018)


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("grid must be a two-dimensional integer array")
    if not all(1 <= n <= 30 for n in a.shape) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid dimensions/colors outside ARC bounds")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _flood(allowed, start):
    reached = np.zeros_like(allowed, dtype=bool)
    if not allowed[start]:
        return reached
    reached[start] = True
    queue = [start]
    h, w = allowed.shape
    while queue:
        r, c = queue.pop()
        for dr, dc in N4:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and allowed[y, x] and not reached[y, x]:
                reached[y, x] = True
                queue.append((y, x))
    return reached


def _components(mask):
    pending = mask.copy()
    result = []
    while pending.any():
        p = tuple(map(int, np.argwhere(pending)[0]))
        region = _flood(pending, p)
        result.append(region)
        pending[region] = False
    return result


def _look(mask, dr, dc):
    """At (r,c), return mask[r+dr,c+dc], with False beyond the array."""
    h, w = mask.shape
    answer = np.zeros_like(mask, dtype=bool)
    y0, y1 = max(0, -dr), min(h, h - dr)
    x0, x1 = max(0, -dc), min(w, w - dc)
    if y0 < y1 and x0 < x1:
        answer[y0:y1, x0:x1] = mask[y0+dr:y1+dr, x0+dc:x1+dc]
    return answer


def _rim(selected, holes=False, orthogonal=False):
    complement = np.pad(~selected, 1, constant_values=True)
    exterior = complement if holes else _flood(complement, (0, 0))
    h, w = selected.shape
    contact = np.zeros_like(selected)
    offsets = N4 if orthogonal else [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)]
    for dr, dc in offsets:
        contact |= exterior[1+dr:1+dr+h, 1+dc:1+dc+w]
    return selected & contact


def _offsets(mask):
    return [tuple(map(int, p - np.array(mask.shape) // 2))
            for p in np.argwhere(mask)]


def _space(selected, rim, offsets):
    feasible = np.ones_like(selected)
    contact = np.zeros_like(selected)
    for dr, dc in offsets:
        feasible &= _look(selected, dr, dc)
        contact |= _look(rim, dr, dc)
    return feasible, feasible & contact


def _nearest(feasible, docks, start, geometric=False):
    """All shortest-distance docks. None means an invalid or trapped request."""
    if not feasible[start]:
        return None, []
    if geometric:
        cells = np.argwhere(docks)
        if not len(cells):
            return None, []
        distance = np.abs(cells - np.array(start)).sum(axis=1)
        best = int(distance.min())
        return best, [tuple(map(int, p)) for p in cells[distance == best]]
    h, w = feasible.shape
    queue = deque([(start, 0)])
    seen = {start}
    best, answer = None, []
    while queue:
        (r, c), distance = queue.popleft()
        if best is not None and distance > best:
            break
        if docks[r, c]:
            best = distance
            answer.append((r, c))
            continue
        if best is not None:
            continue
        for dr, dc in N4:
            p = r + dr, c + dc
            if (0 <= p[0] < h and 0 <= p[1] < w
                    and feasible[p] and p not in seen):
                seen.add(p)
                queue.append((p, distance + 1))
    return best, sorted(answer)


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    if not np.all(grid[0] == bg):
        raise ValueError("the inventory's top gutter must be background")
    separators = [r for r in range(1, grid.shape[0]-3)
                  if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0])]
    if not separators:
        raise ValueError("missing inventory separator")
    sep = separators[0]
    wall = int(grid[sep, 0])
    tray = grid[:sep]
    if (not np.all(tray[-1] == bg) or not np.all(tray[:, [0, -1]] == bg)
            or np.any(tray == wall)):
        raise ValueError("inventory requires bottom/side gutters and no wall cells")
    arena = grid[sep+1:]
    if min(arena.shape) < 3 or not np.all(arena[[0, 0, -1, -1], [0, -1, 0, -1]] == wall):
        raise ValueError("missing arena corners")
    colors = [int(c) for c in np.unique(tray) if c != bg]
    if not colors:
        raise ValueError("empty inventory")
    core = arena[1:-1, 1:-1]
    marker_cells = np.argwhere(~np.isin(core, [bg, wall]))
    if len(marker_cells) != 1:
        raise ValueError("arena interior must contain one marker plus background/walls")
    marker_position = tuple(map(int, marker_cells[0]))
    marker = int(core[marker_position])
    if marker in colors:
        raise ValueError("marker color must be absent from inventory")
    templates = []
    for color in colors:
        cells = np.argwhere(tray == color)
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        shape = tuple(map(int, hi - lo + 1))
        if not all(n % 2 for n in shape):
            raise ValueError("templates require tight odd bounding boxes")
        mask = tray[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == color
        # Diagonal motifs are allowed, but each color must be one 8-connected object.
        reached = {tuple(map(int, np.argwhere(mask)[0]))}
        todo = list(reached)
        while todo:
            r, c = todo.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r+dr, c+dc
                    if (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]
                            and mask[p] and p not in reached):
                        reached.add(p)
                        todo.append(p)
        if len(reached) != int(mask.sum()):
            raise ValueError("one identifiable template per color is required")
        rows = np.flatnonzero((arena[1:-1, 0] == color)
                              & (arena[1:-1, -1] == color)).tolist()
        cols = np.flatnonzero((arena[0, 1:-1] == color)
                              & (arena[-1, 1:-1] == color)).tolist()
        templates.append(dict(color=color, mask=mask, offsets=_offsets(mask),
                              cells=cells, order=tuple(map(int, cells[0])),
                              rows=rows, cols=cols))
    # All edge labels are paired, including any label without a template.
    if (not np.array_equal(arena[0, 1:-1], arena[-1, 1:-1])
            or not np.array_equal(arena[1:-1, 0], arena[1:-1, -1])):
        raise ValueError("unpaired coordinates")
    if np.any(np.isin(np.concatenate((arena[0], arena[-1], arena[:, 0], arena[:, -1])), [bg, marker])):
        raise ValueError("border uses only wall or coordinate colors")
    templates.sort(key=lambda t: t['order'])
    selected = _flood(core != wall, marker_position)
    return dict(grid=grid, bg=bg, wall=wall, marker=marker,
                marker_position=marker_position, sep=sep, core=core,
                selected=selected, rim=_rim(selected), templates=templates)


def _evaluate(info, variant="correct"):
    selected = info['selected']
    if variant == "unseeded":
        selected = info['core'] != info['wall']
    elif variant == "largest":
        selected = max(_components(info['core'] != info['wall']), key=lambda m: int(m.sum()))
    rim = _rim(selected, holes=variant == "holes", orthogonal=variant == "orthogonal")
    output = info['grid'].copy()
    target = output[info['sep']+2:-1, 1:-1]
    occupied = np.zeros_like(selected)
    records = []
    for t in info['templates']:
        offsets = t['offsets']
        movement_offsets = offsets
        if variant == "point":
            movement_offsets = [(0, 0)]
        elif variant == "box":
            movement_offsets = _offsets(np.ones(t['mask'].shape, dtype=bool))
        feasible, docks = _space(selected, rim, movement_offsets)
        destinations = set()
        requests = []
        origins = list(zip(t['rows'], t['cols'])) if variant == 'zip' else [(r, c) for r in t['rows'] for c in t['cols']]
        for r, c in origins:
            start = (r, c)
            if variant == "direct":
                distance, ends = (0, [start]) if feasible[start] else (None, [])
            else:
                distance, ends = _nearest(feasible, docks, start, variant == "geometric")
            if variant == "one_tie" and ends:
                ends = ends[:1]
            destinations.update(ends)
            requests.append(dict(start=start, feasible=bool(feasible[start]),
                                 distance=distance, ends=ends))
        stamp = np.zeros_like(selected)
        for r, c in sorted(destinations):
            for dr, dc in offsets:
                y, x = r+dr, c+dc
                if 0 <= y < stamp.shape[0] and 0 <= x < stamp.shape[1] and selected[y, x]:
                    stamp[y, x] = True
        records.append(dict(template=t, feasible=feasible, docks=docks,
                            requests=requests, stamp=stamp, destinations=destinations))
    render_order = records
    if variant == 'reverse':
        render_order = records[::-1]
    elif variant == 'size':
        render_order = sorted(records, key=lambda r: r['template']['mask'].size)
    elif variant == 'destination':
        render_order = sorted(records, key=lambda r: min(r['destinations'], default=(999, 999)))
    for record in render_order:
        stamp = record['stamp']
        target[stamp & ~occupied] = record['template']['color']
        occupied |= stamp
    target[info['marker_position']] = info['marker']
    return output, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the visible inventory, coordinates, walls, and marker."""
    return _evaluate(_parse(input_grid))[0]


VARIANTS = ("point", "box", "unseeded", "largest", "holes", "orthogonal", "geometric",
            "one_tie", "direct", "reverse", "size", "destination", "zip")


def _features(input_grid, differences=False):
    p = _parse(input_grid)
    correct, records = _evaluate(p)
    result = dict(shape=p['grid'].shape, core_shape=p['core'].shape,
                  palette=(p['bg'], p['wall'], p['marker']),
                  compartments=len(_components(p['core'] != p['wall'])),
                  selected_area=int(p['selected'].sum()),
                  masks=tuple((t['mask'].shape, t['mask'].tobytes()) for t in p['templates']),
                  products=tuple((len(t['rows']), len(t['cols'])) for t in p['templates']),
                  marker=p['marker_position'], ties=0, trapped=0, invalid=0,
                  corners=tuple(i for i, cell in enumerate(((0, 0), (0, -1), (-1, 0), (-1, -1))) if p['selected'][cell]),
                  copies=sum(len(r['destinations']) for r in records),
                  distances=[], conflicts=0)
    seen = np.zeros_like(p['selected'])
    for record in records:
        result['conflicts'] += int(np.count_nonzero(seen & record['stamp']))
        seen |= record['stamp']
        for q in record['requests']:
            result['ties'] += len(q['ends']) > 1
            result['invalid'] += not q['feasible']
            result['trapped'] += q['feasible'] and not q['ends']
            if q['distance'] is not None:
                result['distances'].append(q['distance'])
    if differences:
        result['differences'] = {v: int(np.count_nonzero(correct != _evaluate(p, v)[0]))
                                 for v in VARIANTS}
    return result


def _mask(rng, h=3, w=3):
    mode = int(rng.integers(6))
    mask = np.ones((h, w), dtype=bool)
    if min(h, w) == 1 or mode == 0:
        return mask
    if mode == 1:
        mask[1:-1, 1:-1] = False
    elif mode == 2:
        mask[:] = False
        for r in range(h):
            mask[r, round(r*(w-1)/(h-1))] = True
        for c in range(w):
            mask[round(c*(h-1)/(w-1)), c] = True
    elif mode == 3:
        mask[:] = False
        mask[h//2] = True
        mask[:, w//2] = True
    elif mode == 4 and h == w:
        yy, xx = np.indices((h, w))
        mask = abs(yy-h//2) + abs(xx-w//2) == h//2
    else:
        # A spine plus random adjoining cells gives tight, connected silhouettes.
        mask[:] = False
        mask[h//2] = True
        mask[:, w//2] = True
        for _ in range(h*w):
            additions = mask.copy()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    additions |= _look(mask, dr, dc)
            mask |= additions & (rng.random((h, w)) < .12)
    if rng.random() < .5:
        mask = np.fliplr(mask)
    if rng.random() < .5:
        mask = np.flipud(mask)
    return mask.copy()


def _rotate_scene(walls, marker, templates, rotation, reflect):
    h, w = walls.shape
    def point(p):
        r, c = p
        hh, ww = h, w
        for _ in range(rotation):
            r, c = ww-1-c, r
            hh, ww = ww, hh
        if reflect:
            c = ww-1-c
        return r, c
    rotated = np.rot90(walls, rotation).copy()
    if reflect:
        rotated = np.fliplr(rotated).copy()
    transformed = []
    for t in templates:
        # Transform the axes even when one is absent.
        rows, cols = t['rows'], t['cols']
        rs, cs = rows, cols
        hh, ww = h, w
        for _ in range(rotation):
            rs, cs = [ww-1-c for c in cs], list(rs)
            hh, ww = ww, hh
        if reflect:
            cs = [ww-1-c for c in cs]
        mask = np.rot90(t['mask'], rotation).copy()
        if reflect:
            mask = np.fliplr(mask).copy()
        transformed.append(dict(mask=mask, rows=sorted(rs), cols=sorted(cs)))
    return rotated, point(marker), transformed


def _assemble(rng, walls, marker, templates, color_map, rotate=True):
    if rotate:
        walls, marker, templates = _rotate_scene(walls, marker, templates,
                                                int(rng.integers(4)), bool(rng.integers(2)))
    h, w = walls.shape
    width = w+2
    if sum(t['mask'].shape[1] for t in templates) + len(templates)-1 > width-2:
        return None
    tray_h = max(t['mask'].shape[0] for t in templates)+2
    if tray_h+1+h+2 > 30:
        return None
    tray_h += int(rng.integers(min(2, 31-(tray_h+1+h+2))))
    grid = np.zeros((tray_h+1+h+2, width), dtype=np.int8)
    grid[tray_h] = 8
    arena = grid[tray_h+1:]
    arena[[0, -1]] = 8
    arena[:, [0, -1]] = 8
    arena[1:-1, 1:-1][walls] = 8
    arena[1+marker[0], 1+marker[1]] = 6
    order = rng.permutation(len(templates))
    slack = width-2-sum(t['mask'].shape[1] for t in templates)-(len(templates)-1)
    gaps = rng.multinomial(slack, np.full(len(templates)+1, 1/(len(templates)+1)))
    col = 1+int(gaps[0])
    used_rows, used_cols = set(), set()
    for i, index in enumerate(order):
        t = templates[int(index)]
        color = [1, 2, 3, 4, 5, 7][int(index)]
        mh, mw = t['mask'].shape
        top = int(rng.integers(1, tray_h-mh))
        grid[top:top+mh, col:col+mw][t['mask']] = color
        for r in t['rows']:
            if r in used_rows:
                return None
            arena[r+1, 0] = arena[r+1, -1] = color
            used_rows.add(r)
        for c in t['cols']:
            if c in used_cols:
                return None
            arena[0, c+1] = arena[-1, c+1] = color
            used_cols.add(c)
        col += mw+1+int(gaps[i+1])
    # A paired label with no inventory object is permitted and inert.
    if rng.random() < .25:
        rs = [r for r in range(h) if r not in used_rows]
        cs = [c for c in range(w) if c not in used_cols]
        if rs and cs:
            r, c = int(rng.choice(rs)), int(rng.choice(cs))
            arena[r+1, 0] = arena[r+1, -1] = 9
            arena[0, c+1] = arena[-1, c+1] = 9
    grid = color_map[grid]
    return dict(input=grid, output=solve(grid))


def _gate_scene(rng):
    """A thin brush can escape an island's door; a square cannot."""
    h, w = int(rng.integers(15, 21)), int(rng.integers(21, 27))
    walls = np.zeros((h, w), dtype=bool)
    door = h//2
    left = int(rng.integers(5, 8))
    top, bottom, right = door-3, door+3, left+8
    walls[:, w-4] = True
    walls[top:bottom+1, left] = walls[top:bottom+1, right] = True
    walls[top, left:right+1] = walls[bottom, left:right+1] = True
    walls[door, left] = False
    side = int(rng.choice([-1, 1]))
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    templates = [dict(mask=cross, rows=[door+side], cols=[2]),
                 dict(mask=np.ones((1, 3), bool), rows=[door], cols=[left+3]),
                 dict(mask=np.ones((3, 3), bool), rows=[door-side], cols=[left+5])]
    free = ~walls
    free[:, w-4:] = False
    candidates = np.argwhere(free)
    marker = tuple(map(int, candidates[int(rng.integers(len(candidates)))]))
    return walls, marker, templates


def _random_scene(rng, witness=False, test=False):
    h = int(rng.integers(11 if witness else 9, 21 if test else 20))
    w = int(rng.integers(18 if witness else 13, 29))
    walls = np.zeros((h, w), dtype=bool)
    cut = int(rng.integers(6, max(7, w//2-1)))
    if witness or test or rng.random() < .7:
        style = int(rng.integers(3))
        if style == 2:
            cuts = sorted(set([0, h] + list(map(int, rng.integers(2, h-1, size=int(rng.integers(1, 5)))))))
            previous = cut
            for start, stop in zip(cuts, cuts[1:]):
                col = cut+int(rng.integers(-1, 2))
                walls[start:stop, col] = True
                if start:
                    walls[start, min(col, previous):max(col, previous)+1] = True
                previous = col
        else:
            walls[:, cut] = True
            if style == 1:
                for _ in range(int(rng.integers(1, 5))):
                    r = int(rng.integers(1, h-2))
                    walls[r:r+int(rng.integers(1, 4)), cut-int(rng.integers(0, 3)):cut+int(rng.integers(1, 4))] = True
        if test:
            y = int(rng.integers(3, h-3))
            edge = int(np.flatnonzero(walls[y])[0])
            walls[y, edge:] = True
            if w-cut >= 11 and rng.random() < .4:
                x = int(rng.integers(cut+4, w-3))
                walls[:y, x] = True
    # Boundary-connected intrusions and their variable corner shapes.
    for _ in range(int(rng.integers(0, 3 if witness else 4))):
        r = int(rng.integers(1, h-2))
        depth = int(rng.integers(1, 3))
        walls[r:r+int(rng.integers(1, 3)), :depth] = True
    regions = _components(~walls)
    selected = min(regions, key=lambda m: int(m.sum())) if witness else regions[int(rng.integers(len(regions)))]
    # Independent interior islands: no contact with walls or the canvas edge.
    for _ in range(int(rng.integers(1 if witness else 0, 4))):
        sh, sw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        choices = []
        for r in range(2, h-sh-1):
            for c in range(2, w-sw-1):
                if selected[r-1:r+sh+1, c-1:c+sw+1].all() and not walls[r-1:r+sh+1, c-1:c+sw+1].any():
                    choices.append((r, c))
        if choices:
            r, c = choices[int(rng.integers(len(choices)))]
            shape = np.ones((sh, sw), dtype=bool)
            island_kind = int(rng.integers(3))
            if island_kind == 1:
                shape[1:, 1:] = False
            elif island_kind == 2:
                for row in range(sh):
                    shape[row, int(rng.integers(1, sw+1)):] = False
            shape = np.rot90(shape, int(rng.integers(2))*2)
            walls[r:r+sh, c:c+sw] |= shape
    selected = _flood(~walls, tuple(map(int, np.argwhere(selected & ~walls)[0])))
    points = np.argwhere(selected)
    marker = tuple(map(int, points[int(rng.integers(len(points)))]))
    count = 4 if witness else int(rng.integers(1, 5))
    templates, used_r, used_c = [], set(), set()
    room_rim = _rim(selected)
    for index in range(count):
        if witness and index == 0:
            mask = np.eye(3, dtype=bool)
            if rng.random() < .5:
                mask = np.fliplr(mask).copy()
        elif witness and index == 1:
            mask = _mask(rng)
        elif witness:
            mask = np.ones((1, 1), dtype=bool)
        else:
            sizes = [(1, 1), (1, 3), (3, 1), (3, 3), (1, 5), (5, 1), (3, 5), (5, 3), (5, 5)]
            mh, mw = sizes[int(rng.integers(len(sizes)))]
            mask = _mask(rng, mh, mw)
        # Keep inventory packing possible without constraining movement geometry.
        if sum(t['mask'].shape[1] for t in templates)+mask.shape[1]+index > w:
            break
        feasible, docks = _space(selected, room_rim, _offsets(mask))
        choices = [(int(r), int(c)) for r, c in np.argwhere(feasible & ~docks)
                   if int(r) not in used_r and int(c) not in used_c]
        if not choices:
            choices = [(int(r), int(c)) for r, c in np.argwhere(feasible)
                       if int(r) not in used_r and int(c) not in used_c]
        if choices:
            r, c = choices[int(rng.integers(len(choices)))]
        else:
            rs, cs = [r for r in range(h) if r not in used_r], [c for c in range(w) if c not in used_c]
            if not rs or not cs:
                break
            r, c = int(rng.choice(rs)), int(rng.choice(cs))
        rows, cols = [r], [c]
        nr, nc = (2, 2) if witness and index == 0 else (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        extra_r = [v for v in range(h) if v not in used_r and v != r]
        extra_c = [v for v in range(w) if v not in used_c and v != c]
        rng.shuffle(extra_r)
        rng.shuffle(extra_c)
        rows += extra_r[:nr-1]
        cols += extra_c[:nc-1]
        if witness and index == 2:
            cols = []
        elif witness and index == 3:
            rows = []
        elif not witness and rng.random() < .17:
            if rng.random() < .5:
                rows = []
            else:
                cols = []
        used_r.update(rows)
        used_c.update(cols)
        templates.append(dict(mask=mask, rows=sorted(rows), cols=sorted(cols)))
    return walls, marker, templates


def _make_pair(rng, color_map, kind, disjoint_corners=()):
    for _ in range(500):
        scene = _gate_scene(rng) if kind == 'gate' else _random_scene(rng, kind == 'witness', kind == 'test')
        pair = _assemble(rng, *scene, color_map)
        if pair is None:
            continue
        if kind == 'witness':
            f = _features(pair['input'], True)
            if not (f['differences']['box'] and f['differences']['one_tie']
                    and f['differences']['largest'] and f['differences']['unseeded']
                    and f['differences']['orthogonal'] and f['differences']['zip']
                    and f['invalid']):
                continue
        elif kind == 'gate':
            f = _features(pair['input'], True)
            if not (f['trapped'] and f['differences']['geometric']
                    and f['differences']['holes'] and f['differences']['reverse']
                    and f['differences']['point'] and f['differences']['direct']
                    and f['differences']['size'] and f['differences']['destination']):
                continue
            if set(f['corners']) & set(disjoint_corners):
                continue
            p = _parse(pair['input'])
            brush = next(t for t in p['templates'] if min(t['mask'].shape) == 1)
            if not np.any(pair['output'][p['sep']+2:-1, 1:-1] == brush['color']):
                continue
        else:
            if np.array_equal(pair['input'], pair['output']):
                continue
        return pair
    raise RuntimeError(f"bounded construction failed for {kind}")


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.integers(2))
    base_palette = rng.permutation(10).astype(np.int8)
    puzzle = {'train': [], 'test': []}
    seen, first_corners = set(), ()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            kind = ('witness' if i == 0 else 'gate' if i == 1 else 'random') if split == 'train' else 'test'
            palette = base_palette if stable_palette else rng.permutation(10).astype(np.int8)
            pair = _make_pair(rng, palette, kind, first_corners if kind == 'gate' else ())
            if kind == 'witness':
                first_corners = _features(pair['input'])['corners']
            # Twenty distinct recolorings exceed the thirteen possible earlier inputs.
            # They preserve the required witness differences and the selected region.
            if _key(pair['input']) in seen:
                info = _parse(pair['input'])
                original = pair
                for index in range(20):
                    mapping = np.arange(10, dtype=np.int8)
                    if index >= 10:
                        bg, wall = info['bg'], info['wall']
                        mapping[bg], mapping[wall] = wall, bg
                    mapping = (mapping+index % 10) % 10
                    trial = mapping[original['input']]
                    if _key(trial) not in seen:
                        pair = {role: mapping[original[role]] for role in ('input', 'output')}
                        break
                else:
                    raise RuntimeError('distinct recolorings exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    a, b = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, witnesses = set(), {v: False for v in VARIANTS}
    saw_visible_escape = False
    training_features = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            p = _parse(inp)
            expected, records = _evaluate(p)
            assert np.array_equal(out, expected), 'wrong docking or preservation'
            assert not np.array_equal(inp, out), 'generated scene must visibly dock a template'
            core_out = out[p['sep']+2:-1, 1:-1]
            # Inventory, separator, frame, all walls and marker must survive exactly.
            assert np.array_equal(inp[:p['sep']+2], out[:p['sep']+2])
            assert np.array_equal(inp[:, [0, -1]], out[:, [0, -1]])
            assert np.array_equal(inp[-1], out[-1])
            assert np.array_equal(core_out[p['core'] == p['wall']], p['core'][p['core'] == p['wall']])
            assert np.array_equal(core_out[~p['selected']], p['core'][~p['selected']])
            assert core_out[p['marker_position']] == p['marker']
            for record in records:
                assert not np.any(record['stamp'] & ~p['selected'])
                for center in record['destinations']:
                    assert record['feasible'][center] and record['docks'][center]
                for q in record['requests']:
                    if q['ends']:
                        assert q['distance'] >= 0
                        assert all(record['docks'][x] for x in q['ends'])
            if split == 'train':
                f = _features(inp, True)
                training_features.append(f)
                if f['trapped']:
                    saw_visible_escape |= any(
                        min(record['template']['mask'].shape) == 1
                        and np.any(core_out == record['template']['color'])
                        and any(q['distance'] is not None and q['distance'] > 1
                                for q in record['requests'])
                        for record in records)
                for v in VARIANTS:
                    witnesses[v] |= f['differences'][v] > 0
    assert all(witnesses.values()), f'missing distinguishing demonstrations: {witnesses}'
    assert any(any(r > 1 and c > 1 for r, c in f['products']) for f in training_features)
    assert any(f['trapped'] for f in training_features)
    assert saw_visible_escape, 'missing visible thin-footprint escape beside a trapped footprint'
    assert any(f['invalid'] for f in training_features)
    assert not set.intersection(*(set(f['corners']) for f in training_features))
    assert any(any(r and not c for r, c in f['products']) for f in training_features)
    assert any(any(c and not r for r, c in f['products']) for f in training_features)


def self_test():
    """Bounded 42-seed audit, exact small fixture, and semantic mutations."""
    # Hand-derived example: a horizontal brush at (4,4) in a 9x9 open field
    # reaches left/right docks after three steps; top/bottom take four.
    rng = np.random.default_rng(891)
    scene = (np.zeros((9, 9), bool), (2, 4),
             [dict(mask=np.ones((1, 3), bool), rows=[4], cols=[4])])
    pair = _assemble(rng, *scene, np.arange(10, dtype=np.int8), rotate=False)
    p = _parse(pair['input'])
    expected = pair['input'].copy()
    core = expected[p['sep']+2:-1, 1:-1]
    core[4, :3] = core[4, -3:] = 1
    assert np.array_equal(solve(pair['input']), expected)
    # A center can sit on an island when its transparent mask avoids that cell.
    walls = np.zeros((5, 5), bool)
    walls[2, 1] = True
    diamond = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    pair = _assemble(rng, walls, (2, 4),
                     [dict(mask=diamond, rows=[2], cols=[1])],
                     np.arange(10, dtype=np.int8), rotate=False)
    p = _parse(pair['input'])
    expected = pair['input'].copy()
    core = expected[p['sep']+2:-1, 1:-1]
    for r, c in ((1, 1), (2, 0), (2, 2), (3, 1)):
        core[r, c] = 1
    assert np.array_equal(solve(pair['input']), expected)
    assert np.array_equal(_evaluate(p, 'box')[0], pair['input'])
    # A 3x3 square in a 3x3 island cavity cannot exit its one-cell doorway.
    walls = np.zeros((9, 9), bool)
    walls[2:7, 2] = walls[2:7, 6] = True
    walls[2, 2:7] = walls[6, 2:7] = True
    walls[4, 2] = False
    pair = _assemble(rng, walls, (0, 0),
                     [dict(mask=np.ones((3, 3), bool), rows=[4], cols=[4])],
                     np.arange(10, dtype=np.int8), rotate=False)
    assert np.array_equal(solve(pair['input']), pair['input'])
    assert _features(pair['input'])['trapped'] == 1
    summaries, signatures, shapes, masks, products, palettes = [], set(), set(), set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[role]) for s in ('train', 'test') for p in puzzle[s] for role in ('input', 'output'))
        signatures.add(signature)
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair['input'])
                local.append(f)
                shapes.add(f['shape'])
                masks.update(f['masks'])
                products.update(f['products'])
                palettes.add(f['palette'])
                # Bijective recoloring and within-region seed moves are exact symmetries.
                perm = rng.permutation(10).astype(np.int8)
                before = pair['input'].copy()
                assert np.array_equal(solve(perm[before]), perm[pair['output']])
                assert np.array_equal(before, pair['input'])
                info = _parse(before)
                new = tuple(map(int, np.argwhere(info['selected'])[0]))
                moved = before.copy()
                mc = moved[info['sep']+2:-1, 1:-1]
                mc[info['marker_position']] = info['bg']
                mc[new] = info['marker']
                prediction = solve(moved)
                excluded = np.ones(info['core'].shape, bool)
                excluded[info['marker_position']] = excluded[new] = False
                assert np.array_equal(prediction[info['sep']+2:-1, 1:-1][excluded],
                                      pair['output'][info['sep']+2:-1, 1:-1][excluded])
        assert len({f['shape'] for f in local}) >= 2
        assert len({f['compartments'] for f in local}) >= 2
        assert len({f['masks'] for f in local}) >= 2
        summaries.extend(local)
    assert len(signatures) == 42 and len(shapes) >= 50 and len(masks) >= 30
    assert {(1, 1), (1, 2), (2, 1), (2, 2)} <= products
    assert len(palettes) >= 20
    assert max(f['compartments'] for f in summaries) >= 3
    assert max(f['copies'] for f in summaries) >= 4
    fixed = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in fixed for x, y in zip(fixed[s], generate_puzzle(0, 2, 1)[s]) for k in x)
    for v in VARIANTS:
        for i, pair in enumerate(fixed['train']):
            wrong = _evaluate(_parse(pair['input']), v)[0]
            if not np.array_equal(wrong, pair['output']):
                corrupted = copy.deepcopy(fixed)
                corrupted['train'][i]['output'] = wrong
                try:
                    validate_generated_puzzle(corrupted)
                except AssertionError:
                    break
                raise AssertionError(f'validator accepted {v} shortcut')
        else:
            raise AssertionError(f'no mutation witness for {v}')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=len(summaries), grid_shapes=len(shapes),
                template_masks=len(masks), coordinate_products=sorted(products),
                palettes=len(palettes), independent_solver=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
