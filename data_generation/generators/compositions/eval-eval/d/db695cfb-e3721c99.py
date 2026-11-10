"""Cavity relay fan-out: input-only topology, finite activation, remote emission.

The generator does not import or execute either parent. All role assignments are
recoverable from a separator, connected glyphs, enclosed ports and exterior dots.
"""
from __future__ import annotations

from itertools import permutations
import numpy as np


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _cell_mask(cells):
    coords = np.asarray(cells, dtype=int)
    low, high = coords.min(0), coords.max(0) + 1
    mask = np.zeros(tuple(high-low), dtype=bool)
    mask[tuple((coords-low).T)] = True
    return mask, tuple(map(int, low))


def _cavities(mask):
    """Four-connected holes with a genuine exterior, including at canvas edges."""
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    return [[(r-1, c-1) for r, c in cells]
            for cells in _components(~padded)
            if all(0 < r < padded.shape[0]-1 and 0 < c < padded.shape[1]-1
                   for r, c in cells)]


def _holes(mask):
    return len(_cavities(mask))


def _segment(a, b):
    dr, dc = b[0]-a[0], b[1]-a[1]
    assert dr and abs(dr) == abs(dc)
    return tuple((a[0]+i*(1 if dr > 0 else -1),
                  a[1]+i*(1 if dc > 0 else -1)) for i in range(abs(dr)+1))


def _edges(dots):
    dots = sorted(dots)
    edges, degrees = [], dict.fromkeys(dots, 0)
    for i, a in enumerate(dots):
        for b in dots[i+1:]:
            if a[0] != b[0] and abs(a[0]-b[0]) == abs(a[1]-b[1]):
                edges.append((a, b))
                degrees[a] += 1
                degrees[b] += 1
    assert edges and max(degrees.values()) <= 1, 'Endpoint partners must be unique'
    assert all(abs(a[0]-b[0]) >= 2 for a, b in edges)
    slopes = {1 if (b[0]-a[0])*(b[1]-a[1]) > 0 else -1 for a, b in edges}
    assert len(slopes) == 1, 'All finite segments in one input are parallel'
    return edges, slopes.pop()


def _diagonal(point, slope, work):
    r, c = point
    return {(rr, c+slope*(rr-r)) for rr in range(work.shape[0])
            if 0 <= c+slope*(rr-r) < work.shape[1]
            and work[rr, c+slope*(rr-r)]}


def _as_grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid.astype(int, copy=False)


def _interpret_work(grid, work, bg, sep, mapping, panel, barrier):
    colors = set(map(int, np.unique(grid[work]))) - {bg}
    assert len(colors) == 3
    candidates = []
    for query, port, primary in permutations(sorted(colors)):
        if {query, port, primary} & (set(mapping.values()) | {sep}):
            continue
        # Connected silhouettes, singleton ports and singleton exterior dots
        # encode their own roles. Nothing is learned from a seed or output.
        if any(len(x) != 1 for color in (port, primary)
               for x in _components(work & (grid == color))):
            continue
        try:
            objects, all_ports, interiors = [], set(), set()
            for cells in _components(work & (grid == query)):
                assert len(cells) >= 4
                mask, pos = _cell_mask(cells)
                holes = _cavities(mask)
                assert 1 <= len(holes) <= 5
                ports = []
                for hole in holes:
                    absolute = {(pos[0]+r, pos[1]+c) for r, c in hole}
                    assert all(work[p] and grid[p] in (bg, port) for p in absolute)
                    visible = [p for p in absolute if grid[p] == port]
                    assert len(visible) <= 1
                    ports.extend(visible)
                    interiors.update(absolute)
                assert ports, 'Every relay has at least one marked aperture'
                all_ports.update(ports)
                objects.append(dict(cells=set(cells), mask=mask, pos=pos,
                                    holes=holes, ports=tuple(sorted(ports)), count=len(holes)))
            assert objects
            assert all_ports == {tuple(map(int, x)) for x in np.argwhere(work & (grid == port))}
            dots = {tuple(map(int, x)) for x in np.argwhere(work & (grid == primary))}
            assert not dots & interiors
            edges, slope = _edges(dots)
            spans = tuple(_segment(a, b) for a, b in edges)
            span_union = set().union(*(set(s) for s in spans))
            for obj in objects:
                obj['hit_ports'] = set(obj['ports']) & span_union
                obj['active'] = bool(obj['hit_ports'])
                obj['color'] = mapping.get(obj['count']) if obj['active'] else None
            # Opposite-slope rays are parallel. Coincident rays can merge only
            # when their legend color agrees; there is no hidden color priority.
            ray_colors = {}
            for obj in objects:
                if obj['active'] and obj['color'] is not None:
                    for p in obj['ports']:
                        signature = p[1] + slope*p[0]
                        assert signature not in ray_colors or ray_colors[signature] == obj['color']
                        ray_colors[signature] = obj['color']
            candidates.append(dict(background=bg, query=query, port=port, primary=primary,
                                   separator=sep, mapping=mapping, objects=objects,
                                   dots=dots, spans=spans, span_union=span_union,
                                   slope=slope, panel=panel, barrier=barrier, work=work))
        except AssertionError:
            continue
    assert len(candidates) == 1, 'Work area roles must have one interpretation'
    return candidates[0]


def _parse(grid):
    grid = _as_grid(grid)
    values, counts = np.unique(grid, return_counts=True)
    assert np.sum(counts == counts.max()) == 1
    bg = int(values[counts.argmax()])
    barriers = []
    for axis in (0, 1):
        for i in range(grid.shape[axis]):
            line = grid[i, :] if axis == 0 else grid[:, i]
            if line[0] != bg and np.all(line == line[0]) and 0 < i < grid.shape[axis]-1:
                barriers.append((axis, i, int(line[0])))
    assert len(barriers) == 1, 'One complete straight separator is required'
    axis, index, sep = barriers[0]
    barrier = np.zeros(grid.shape, dtype=bool)
    if axis == 0:
        barrier[index, :] = True
    else:
        barrier[:, index] = True
    candidates = []
    for before in (True, False):
        panel = np.zeros(grid.shape, dtype=bool)
        if axis == 0:
            panel[:index, :] = before
            panel[index+1:, :] = not before
        else:
            panel[:, :index] = before
            panel[:, index+1:] = not before
        work = ~panel & ~barrier
        try:
            mapping = {}
            for color in sorted(set(map(int, np.unique(grid[panel]))) - {bg}):
                assert color != sep
                components = _components(panel & (grid == color))
                assert len(components) == 1
                mask, pos = _cell_mask(components[0])
                holes = _cavities(mask)
                k = len(holes)
                assert 1 <= k <= 5 and k not in mapping
                assert all(grid[pos[0]+r, pos[1]+c] == bg for h in holes for r, c in h)
                mapping[k] = color
            assert 1 <= len(mapping) <= 4
            candidates.append(_interpret_work(grid, work, bg, sep, mapping, panel, barrier))
        except AssertionError:
            continue
    assert len(candidates) == 1, 'Legend and field must be visibly distinguishable'
    return candidates[0]


def _apply(grid, parsed, *, wake_all=False, touched_only=False, constant_class=None,
           touch_walls=False, class_by_ports=False):
    """Paint after simultaneous input-based decisions; keyword modes are ablations."""
    bg, primary = parsed['background'], parsed['primary']
    out = grid.copy()
    erased, protected = set(), set(parsed['dots'])
    emitting = []
    for obj in parsed['objects']:
        active = obj['active'] or wake_all or (touch_walls and bool(obj['cells'] & parsed['span_union']))
        k = len(obj['ports']) if class_by_ports else obj['count']
        if constant_class is not None:
            k = constant_class
        color = parsed['mapping'].get(k) if active else None
        if active and color is None:
            erased.update(obj['cells'] | set(obj['ports']))
            for p in obj['cells'] | set(obj['ports']):
                out[p] = bg
        else:
            protected.update(obj['cells'] | set(obj['ports']))
            if active:
                for p in obj['cells']:
                    out[p] = color
                ports = obj['hit_ports'] if touched_only else obj['ports']
                emitting.extend((p, color) for p in ports)
    # The erased pixels become drawable blank space. Retained silhouettes and
    # ports are protected, while their originally blank interiors transmit lines.
    for p in parsed['span_union'] - protected:
        out[p] = primary
    for point, color in emitting:
        for p in _diagonal(point, -parsed['slope'], parsed['work']) - protected:
            out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _as_grid(input_grid)
    return _apply(grid, _parse(grid))


def _compact(k):
    rows = 1 if k <= 3 else 2
    cols = (k+rows-1)//rows
    mask = np.ones((2*rows+1, 2*cols+1), dtype=bool)
    for i in range(k):
        mask[1+2*(i//cols), 1+2*(i%cols)] = False
    return mask


def _shape(k, rng, rich=False):
    """Unequal cavity compartments with safe topology-preserving contour edits."""
    rows = int(rng.choice((1, 2))) if k >= 2 else 1
    cols = (k+rows-1)//rows
    heights = [int(rng.integers(1, 4 if rich else 3)) for _ in range(rows)]
    widths = [int(rng.integers(1, 4 if rich else 3)) for _ in range(cols)]
    wall = 2 if rng.random() < .12 else 1
    mask = np.ones((sum(heights)+wall*(rows+1), sum(widths)+wall*(cols+1)), dtype=bool)
    rs, cs = [], []
    p = wall
    for n in heights:
        rs.append(p)
        p += n+wall
    p = wall
    for n in widths:
        cs.append(p)
        p += n+wall
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    for i in rng.permutation(len(slots))[:k]:
        r, c = slots[int(i)]
        h, w = heights[r], widths[c]
        hole = np.ones((h, w), dtype=bool)
        if h > 1 and w > 1 and rng.random() < .65:
            hole[int(rng.choice((0, h-1))), int(rng.choice((0, w-1)))] = False
        mask[rs[r]:rs[r]+h, cs[c]:cs[c]+w][hole] = False
    # Cutting corners includes diagonal-only escapes, which remain enclosed.
    for _ in range(5 if rich else 3):
        candidate = mask.copy()
        r = int(rng.choice((0, mask.shape[0]-1)))
        c = int(rng.choice((0, mask.shape[1]-1)))
        candidate[r, c] = False
        if len(_components(candidate)) == 1 and _holes(candidate) == k:
            mask = candidate
    if rich and rng.random() < .55:
        for _ in range(4):
            candidate = mask.copy()
            side = int(rng.integers(4))
            if side < 2:
                c = int(rng.integers(mask.shape[1]))
                candidate[0 if side == 0 else -1, c] = False
            else:
                r = int(rng.integers(mask.shape[0]))
                candidate[r, 0 if side == 2 else -1] = False
            if candidate.any() and len(_components(candidate)) == 1 and _holes(candidate) == k:
                mask = candidate
    mask = np.rot90(mask, int(rng.integers(4))).copy()
    mask, _ = _cell_mask([tuple(map(int, p)) for p in np.argwhere(mask)])
    holes = _cavities(mask)
    assert len(holes) == k and len(_components(mask)) == 1
    ports = [hole[int(rng.integers(len(hole)))] for hole in holes]
    return mask, ports


def _panel(rng, keys, roles, *, fallback=False):
    bg, query, port, primary, sep = roles
    colors = [int(x) for x in rng.permutation([x for x in range(10) if x not in roles])[:len(keys)]]
    mapping = dict(zip(keys, colors))
    glyphs = [_compact(k) for k in keys]
    if not fallback:
        glyphs = [g.T.copy() if rng.random() < .25 else g for g in glyphs]
    panel_height = max(g.shape[0] for g in glyphs) + 2
    needed_width = sum(g.shape[1] for g in glyphs) + len(keys)+1
    if needed_width > 30:
        return None
    h = 30 if fallback else int(rng.integers(max(23, panel_height+18), 31))
    w = 30 if fallback else int(rng.integers(max(20, needed_width), 31))
    grid = np.full((h, w), bg, dtype=int)
    grid[panel_height, :] = sep
    c = 1 if fallback else int(rng.integers(w-needed_width+2))
    for k, glyph in zip(keys, glyphs):
        r = 1 if fallback else int(rng.integers(1, panel_height-glyph.shape[0]))
        grid[r:r+glyph.shape[0], c:c+glyph.shape[1]][glyph] = mapping[k]
        c += glyph.shape[1]+1
    return grid, mapping, panel_height+1


def _pack(grid, offset, specifications, roles, rng, rich, partial_class=None):
    bg, query, port, primary, sep = roles
    height, width = grid.shape[0]-offset, grid.shape[1]
    free = np.ones((height, width), dtype=bool)
    objects = []
    # Mandatory roles are placed first, optional objects may be dropped.
    for k, status, mandatory in specifications:
        mask, ports = _shape(k, rng, rich)
        if k == partial_class and status == 'active' and mandatory:
            marked_count = k-1
        else:
            marked_count = k if rng.random() < .4 else int(rng.integers(1,k+1))
        ports = [ports[int(i)] for i in rng.permutation(k)[:marked_count]]
        h, w = mask.shape
        if h > height or w > width:
            if mandatory:
                return None
            continue
        choices = [(r, c) for r in range(height-h+1) for c in range(width-w+1)
                   if free[r:r+h, c:c+w].all()
                   and (status != 'active' or (r >= 2 and c >= 2 and r+h <= height-2 and c+w <= width-2))]
        if not choices:
            if mandatory:
                return None
            continue
        r, c = choices[int(rng.integers(len(choices)))]
        grid[offset+r:offset+r+h, c:c+w][mask] = query
        absolute_ports = [(offset+r+rr, c+cc) for rr, cc in ports]
        for p in absolute_ports:
            grid[p] = port
        free[max(0,r-1):min(height,r+h+1), max(0,c-1):min(width,c+w+1)] = False
        objects.append(dict(count=k, status=status, mandatory=mandatory,
                            ports=absolute_ports, box=(offset+r, offset+r+h, c, c+w)))
    return objects


def _add_segments(grid, offset, objects, roles, rng, extra_empty=False):
    bg, query, port, primary, sep = roles
    # Positive slope is canonical; reflection/rotation supplies both slopes.
    occupied_boxes = np.zeros(grid.shape, dtype=bool)
    occupied_boxes[:offset] = True
    for obj in objects:
        r, b, c, e = obj['box']
        occupied_boxes[r:b, c:e] = True
    dots, spans = [], []
    for obj in objects:
        if obj['status'] != 'active':
            continue
        possible = []
        for anchor in obj['ports']:
            if any(anchor in span for span in spans):
                possible.append((None, None))
                continue
            r, c = anchor
            before = [(r-d, c-d) for d in range(1, min(r-offset, c)+1)
                      if not occupied_boxes[r-d, c-d] and grid[r-d, c-d] == bg]
            after = [(r+d, c+d) for d in range(1, min(grid.shape[0]-1-r, grid.shape[1]-1-c)+1)
                     if not occupied_boxes[r+d, c+d] and grid[r+d, c+d] == bg]
            for a in before:
                for b in after:
                    if any(abs(p[0]-q[0]) == abs(p[1]-q[1]) for p in (a, b) for q in dots):
                        continue
                    if abs(a[0]-b[0]) < 2:
                        continue
                    span = set(_segment(a, b))
                    if any(span & set(other['ports']) for other in objects if other['status'] == 'inactive'):
                        continue
                    possible.append((a, b))
        if not possible:
            return False
        a, b = possible[int(rng.integers(len(possible)))]
        if a is not None:
            dots.extend((a, b))
            spans.append(set(_segment(a, b)))
    if extra_empty:
        choices = []
        all_ports = set().union(*(set(o['ports']) for o in objects))
        for _ in range(100):
            length = int(rng.integers(3, min(9, grid.shape[0]-offset, grid.shape[1])+1))
            r = int(rng.integers(offset, grid.shape[0]-length+1))
            c = int(rng.integers(grid.shape[1]-length+1))
            a, b = (r, c), (r+length-1, c+length-1)
            if occupied_boxes[a] or occupied_boxes[b]:
                continue
            if any(abs(p[0]-q[0]) == abs(p[1]-q[1]) for p in (a,b) for q in dots):
                continue
            if set(_segment(a,b)) & all_ports:
                continue
            choices.append((a,b))
        if choices:
            a,b = choices[int(rng.integers(len(choices)))]
            dots.extend((a,b))
    for p in dots:
        grid[p] = primary
    for _ in range(int(rng.integers(3))):
        choices = [tuple(map(int, p)) for p in np.argwhere((grid == bg) & ~occupied_boxes)
                   if not any(abs(int(p[0])-q[0]) == abs(int(p[1])-q[1]) for q in dots)
                   and not any(abs(int(p[0])-q[0])+abs(int(p[1])-q[1]) == 1 for q in dots)]
        if choices:
            p = choices[int(rng.integers(len(choices)))]
            grid[p] = primary
            dots.append(p)
    return True


def _fallback(rng, roles, multi, witness, ordinal):
    """Guaranteed fitting construction, not a bank of saved input/output pairs."""
    keys = [multi, 1]
    grid, mapping, offset = _panel(rng, keys, roles, fallback=True)
    bg, query, port, primary, sep = roles
    count_active = multi if witness != 'single' else 1
    second = 5 if witness != 'single' else multi
    # A relay witness puts an inactive port on the primary continuation beyond
    # its endpoint. The other witness puts inactive ports on an emitted ray.
    inactive_col = 12 if witness == 'single' else 17
    items = [(count_active, 4, 4), (second, 11, 18), (multi, 17, inactive_col), (1,17,23)]
    for i,(k,r,c) in enumerate(items):
        if i == 2:
            # A tall compact relay keeps its first port on the continuation,
            # while leaving room for a separate wall-only beam contact.
            mask = np.ones((5,3 if k == 2 else 5),dtype=bool)
            locations = [(1,1),(3,1)] if k == 2 else [(1,1),(1,3),(3,1),(3,3)][:k]
            for p in locations:
                mask[p] = False
        else:
            mask = _compact(k)
        grid[offset+r:offset+r+mask.shape[0],c:c+mask.shape[1]][mask] = query
        cavities = _cavities(mask)
        # The relay teaching witness has an unmarked cavity: class cannot be
        # guessed by counting dots, while two marked ports still permit fan-out.
        if i == 0 and witness != 'single':
            cavities = cavities[:-1]
        for hole in cavities:
            rr,cc = hole[0]
            grid[offset+r+rr,c+cc] = port
    for r,c in ((1,1),(9,9),(9,16),(19,26)):
        grid[offset+r,c] = primary
    # Translating only the legend guarantees 14 distinct fallback inputs without
    # changing any field relationship. Its occupied width is at most 11 cells.
    grid[:offset-1] = np.roll(grid[:offset-1],ordinal,axis=1)
    return grid


def _features(grid, parsed=None):
    p = _parse(grid) if parsed is None else parsed
    active = [o for o in p['objects'] if o['active']]
    accepted = [o for o in active if o['count'] in p['mapping']]
    inert_ports = set().union(*(set(o['ports']) for o in p['objects'] if not o['active']))
    ray_signatures = {x[1]+p['slope']*x[0] for o in accepted for x in o['ports']}
    primary_signatures = {s[0][1]-p['slope']*s[0][0] for s in p['spans']}
    return dict(shape=grid.shape, roles=tuple(p[k] for k in ('background','query','port','primary','separator')),
                keys=tuple(sorted(p['mapping'])), objects=len(p['objects']),
                counts=tuple(sorted(o['count'] for o in p['objects'])),
                active=len(active), erased=sum(o['count'] not in p['mapping'] for o in active),
                inactive=len(p['objects'])-len(active),
                accepted_counts=tuple(sorted({o['count'] for o in accepted})),
                spans=len(p['spans']), lengths=tuple(sorted(map(len,p['spans']))), slope=p['slope'],
                remote=sum(len(set(o['ports'])-o['hit_ports']) for o in accepted),
                unmarked=sum(o['count']-len(o['ports']) for o in p['objects']),
                accepted_unmarked=any(o['count'] > len(o['ports']) for o in accepted),
                continuation=any(x[1]-p['slope']*x[0] in primary_signatures for x in inert_ports),
                inert_on_ray=any(x[1]+p['slope']*x[0] in ray_signatures for x in inert_ports),
                wall_only=any(not o['active'] and bool(o['cells'] & p['span_union']) for o in p['objects']),
                unused=sum(not any(o['active'] and o['count'] == k for o in p['objects']) for k in p['mapping']),
                hole_areas=tuple(sorted(len(h) for o in p['objects'] for h in o['holes'])),
                irregular=sum(not _cell_mask(h)[0].all() for o in p['objects'] for h in o['holes']))


def _rank_alternative(grid, parsed, origin):
    """Wrong lookup by increasing row/column rank, for teaching checks only."""
    axis = 1 if np.any(np.all(parsed['barrier'],axis=1)) else 0
    colors = sorted(parsed['mapping'].values(), key=lambda color:
                    int(np.argwhere(parsed['panel'] & (grid == color))[:,axis].min()))
    alternative = dict(parsed)
    alternative['mapping'] = {origin+i:color for i,color in enumerate(colors)}
    return _apply(grid,alternative)


def _witness_ok(grid, p, witness, multi):
    f = _features(grid,p)
    if not f['accepted_counts']:
        return False
    if witness == 'relay':
        signatures = {s[0][1]-p['slope']*s[0][0] for s in p['spans']}
        same_class_continuation = any(
            not o['active'] and o['count'] == multi
            and any(x[1]-p['slope']*x[0] in signatures for x in o['ports'])
            for o in p['objects'])
        return (multi in f['accepted_counts'] and f['erased'] > 0 and f['remote'] > 0
                and same_class_continuation
                and f['wall_only']
                and f['accepted_unmarked']
                and any(not o['active'] and o['count'] == multi for o in p['objects'])
                and not np.array_equal(_apply(grid,p), _apply(grid,p,touched_only=True))
                and not np.array_equal(_apply(grid,p), _apply(grid,p,touch_walls=True))
                and not np.array_equal(_apply(grid,p), _apply(grid,p,class_by_ports=True))
                and all(not np.array_equal(_apply(grid,p),_rank_alternative(grid,p,origin))
                        for origin in (0,1))
                and not np.array_equal(_apply(grid,p), _apply(grid,p,wake_all=True)))
    if witness == 'single':
        return 1 in f['accepted_counts'] and f['inert_on_ray']
    return True


def _make_pair(rng, roles, multi, witness, test, rotation, ordinal, seen):
    for attempt in range(48):
        keys = [multi, 1]
        if witness is None:
            keys = list(map(int,rng.choice(np.arange(1,6),size=int(rng.integers(1,4)),replace=False)))
        if attempt < 30 and rng.random() < .35 and len(keys) < 4:
            extra = [k for k in range(1,6) if k not in keys and (witness != 'relay' or k != 5)]
            if extra:
                keys.append(int(rng.choice(extra)))
        rng.shuffle(keys)
        panel = _panel(rng,keys,roles)
        if panel is None:
            continue
        grid,mapping,offset = panel
        absent = [k for k in range(1,6) if k not in keys]
        if witness == 'relay':
            specs = [(multi,'active',True),(int(rng.choice(absent)),'active',True),
                     (multi,'inactive',True),(int(rng.choice(keys)),'inactive',True)]
        elif witness == 'single':
            specs = [(1,'active',True),(multi,'active',True),(int(rng.choice(keys)),'inactive',True)]
        else:
            specs = [(int(rng.choice(keys)),'active',True)]
        target = int(rng.integers(max(3,len(specs)), (8 if test else 6)))
        while len(specs) < target:
            specs.append((int(rng.integers(1,6)),'free',False))
        objects = _pack(grid,offset,specs,roles,rng,rich=test or attempt % 3 == 0,
                        partial_class=multi if witness == 'relay' else None)
        if objects is None or not _add_segments(grid,offset,objects,roles,rng,extra_empty=rng.random() < .4):
            continue
        try:
            p = _parse(grid)
        except AssertionError:
            continue
        if not _witness_ok(grid,p,witness,multi):
            continue
        if witness == 'single':
            grid = np.fliplr(grid)
        candidate = np.rot90(grid,rotation).copy()
        if not _witness_ok(candidate,_parse(candidate),witness,multi):
            continue
        if _key(candidate) not in seen:
            return {'input':candidate,'output':solve(candidate)}
    grid = _fallback(rng,roles,multi,witness,ordinal)
    if witness == 'single':
        grid = np.fliplr(grid)
    candidate = np.rot90(grid,rotation).copy()
    assert _key(candidate) not in seen
    assert _witness_ok(candidate,_parse(candidate),witness,multi)
    return {'input':candidate,'output':solve(candidate)}


def _key(grid):
    return tuple(grid.shape), np.asarray(grid,dtype=np.int8).tobytes()


def _counts(a,b):
    for value,lo,hi in ((a,2,10),(b,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= int(value) <= hi:
            raise ValueError('Counts must be integers: train 2..10, test 1..4')
    if a <= b:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base_roles = tuple(map(int,rng.permutation(10)[:5]))
    stable_roles = bool(rng.integers(2))
    multi = int(rng.integers(3,5))
    orientation = int(rng.integers(4))
    # Shift one structural axis: tests use a different panel side and richer
    # cavity contours/populations; topology and input-based activation do not change.
    test_rotation = (orientation+int(rng.choice((1,2,3)))) % 4
    witness_slots = rng.permutation(int(num_train))[:2]
    witnesses = {int(witness_slots[0]):'relay',int(witness_slots[1]):'single'}
    puzzle,seen = {'train':[],'test':[]},set()
    ordinal = 0
    for split,count in (('train',num_train),('test',num_test)):
        for i in range(int(count)):
            roles = base_roles if stable_roles else tuple(map(int,rng.permutation(10)[:5]))
            pair = _make_pair(rng,roles,multi,witnesses.get(i) if split == 'train' else None,
                              split == 'test',orientation if split == 'train' else test_rotation,ordinal,seen)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
            ordinal += 1
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen, training = set(), []
    accepted_counts, slopes = set(), set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            grid,out = _as_grid(pair['input']),_as_grid(pair['output'])
            assert grid.shape == out.shape and _key(grid) not in seen
            seen.add(_key(grid))
            p = _parse(grid)
            assert np.array_equal(out,_apply(grid,p)), 'Incorrect relay transformation'
            assert not np.array_equal(out,grid)
            assert np.array_equal(grid[p['panel'] | p['barrier']],out[p['panel'] | p['barrier']])
            for dot in p['dots']:
                assert out[dot] == p['primary']
            for obj in p['objects']:
                if not obj['active']:
                    assert all(out[x] == p['query'] for x in obj['cells'])
                    assert all(out[x] == p['port'] for x in obj['ports'])
                elif obj['count'] in p['mapping']:
                    assert all(out[x] == p['mapping'][obj['count']] for x in obj['cells'])
                    assert all(out[x] == p['port'] for x in obj['ports'])
                else:
                    # Removed material may be replaced by a line, but never by
                    # residual neutral material or by the original port role.
                    assert all(out[x] not in (p['query'],p['port']) for x in obj['cells'] | set(obj['ports']))
            if split == 'train':
                training.append((grid,out,p))
                accepted_counts.update(o['count'] for o in p['objects'] if o['active'] and o['count'] in p['mapping'])
                slopes.add(p['slope'])
    assert slopes == {-1,1} and 1 in accepted_counts and any(k >= 2 for k in accepted_counts)
    assert any(any(o['active'] and o['count'] not in p['mapping'] for o in p['objects']) for _,_,p in training)
    assert any(not np.array_equal(out,_apply(g,p,wake_all=True)) for g,out,p in training), 'A omission has no witness'
    assert any(not np.array_equal(out,_apply(g,p,touched_only=True)) for g,out,p in training), 'Remote relay has no witness'
    assert any(_features(g,p)['continuation'] for g,_,p in training), 'Finite activation has no witness'
    assert any(_features(g,p)['inert_on_ray'] for g,_,p in training), 'Nonrecursive emission has no witness'
    assert any(not np.array_equal(out,_apply(g,p,touch_walls=True)) for g,out,p in training), 'Port activation has no witness'
    assert any(not np.array_equal(out,_apply(g,p,class_by_ports=True)) for g,out,p in training), 'Cavity count has no witness'
    assert all(any(not np.array_equal(out,_rank_alternative(g,p,origin)) for g,out,p in training)
               for origin in (0,1)), 'Legend-rank decoding has no witness'
    assert any(_witness_ok(g,p,'relay',k) for g,_,p in training for k in p['mapping'] if k >= 3)
    assert any(_witness_ok(g,p,'single',None) for g,_,p in training)
    # Omitting B by treating all objects as one fixed cavity class fails.
    assert all(any(not np.array_equal(out,_apply(g,p,constant_class=k)) for g,out,p in training) for k in range(1,6))


def self_test():
    """Bounded 42-seed semantic, teaching, variation and corruption checks."""
    diagonal = np.array([[0,1,1],[1,0,1],[1,1,1]],dtype=bool)
    assert _holes(diagonal) == 1
    diagonal[0,1] = False
    assert _holes(diagonal) == 0
    # Hand-specified small field: one active two-hole relay, an inert one-hole
    # relay on its emitted ray, and another beyond the finite endpoint at the
    # bottom canvas edge. Assertions use literal coordinates and colors.
    fixture = np.zeros((16,20),dtype=int)
    fixture[5,:] = 8
    for k,color,c in ((1,3,1),(2,4,6)):
        glyph = _compact(k)
        fixture[1:1+glyph.shape[0],c:c+glyph.shape[1]][glyph] = color
    for k,r,c in ((2,8,5),(1,12,1),(1,13,10)):
        glyph = _compact(k)
        fixture[r:r+glyph.shape[0],c:c+glyph.shape[1]][glyph] = 5
        for hole in _cavities(glyph):
            rr,cc = hole[0]
            fixture[r+rr,c+cc] = 2
    for x in ((6,3),(12,9),(14,18)):
        fixture[x] = 1
    result = solve(fixture)
    assert result[6,9] == result[6,11] == 4  # touched and remote rays
    assert result[7,4] == 1  # finite segment, away from projection
    assert result[8,5] == 4 and result[9,6] == result[9,8] == 2
    assert result[12,3] == 5 and result[13,2] == 2  # inert on emitted ray
    assert result[13,10] == 5 and result[14,11] == 2  # beyond finite endpoint
    assert result[15,12] == 5  # canvas contact does not change enclosure
    erased_fixture = fixture.copy()
    erased_fixture[1:4,6:11] = 0  # class two is now unlisted
    erased_result = solve(erased_fixture)
    assert erased_result[8,6] == erased_result[9,8] == 0
    assert erased_result[8,5] == erased_result[9,6] == 1  # line replaces erased material
    assert erased_result[6,11] == 0 and erased_result[12,3] == 5
    untriggered = fixture.copy()
    untriggered[untriggered == 1] = 0
    untriggered[6,15] = untriggered[9,18] = 1
    untriggered_result = solve(untriggered)
    assert untriggered_result[7,16] == untriggered_result[8,17] == 1
    assert untriggered_result[8,5] == 5 and untriggered_result[6,11] == 0
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(fixture,rotation)),np.rot90(result,rotation))
    permutation = np.array([7,4,9,0,2,8,1,6,5,3])
    assert np.array_equal(solve(permutation[fixture]),permutation[result])
    # Two accepted relays may emit the same color on exactly the same lines.
    coincident = fixture.copy()
    glyph = _compact(2)
    coincident[12:15,1:6] = 0
    coincident[12:15,1:6][glyph] = 5
    coincident[13,2] = coincident[13,4] = 2
    coincident[11,0] = coincident[15,4] = 1
    coincident_result = solve(coincident)
    assert coincident_result[12,1] == coincident_result[6,9] == 4
    # Changing the second relay to class one would demand two colors on the
    # same projected line, violating the visible compatibility constraint.
    conflicting = fixture.copy()
    conflicting[11,0] = conflicting[15,4] = 1
    try:
        solve(conflicting)
    except AssertionError:
        pass
    else:
        raise AssertionError('Different-color coincident rays accepted')
    missing_port = fixture.copy()
    missing_port[9,6] = missing_port[9,8] = 0
    try:
        solve(missing_port)
    except AssertionError:
        pass
    else:
        raise AssertionError('A relay with no marked aperture was accepted')
    partial_ports = fixture.copy()
    partial_ports[9,8] = 0
    partial_result = solve(partial_ports)
    assert partial_result[8,5] == 4  # still class two, although it has one port
    assert partial_result[6,9] == 4 and partial_result[6,11] == 0
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_features(pair['input']) for p in puzzles for s in p for pair in p[s]]
    assert len({tuple(_key(pair['input']) for s in p for pair in p[s]) for p in puzzles}) == 42
    assert len({r['shape'] for r in records}) >= 35
    assert len({r['roles'] for r in records}) >= 25
    assert len({r['keys'] for r in records}) >= 12
    assert len({r['counts'] for r in records}) >= 30
    assert len({r['objects'] for r in records}) >= 4
    assert len({r['lengths'] for r in records}) >= 20
    assert {1,2,3,4,5} <= {k for r in records for k in r['counts']}
    assert any(r['irregular'] for r in records)
    assert all(any(_features(pair['input'])['accepted_unmarked'] for pair in p['train']) for p in puzzles)
    assert any(r['unused'] for r in records)
    assert any(r['erased'] == 0 for r in records)
    assert all(len({_features(pair['input'])['shape'] for s in p for pair in p[s]}) >= 2 for p in puzzles)
    p = generate_puzzle(5145886,10,4)
    q = generate_puzzle(5145886,10,4)
    assert all(np.array_equal(a[k],b[k]) for s in p for a,b in zip(p[s],q[s]) for k in ('input','output'))
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            _counts(a,b)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Directly exercise the finite fallback for all 14 pair ordinals.
    rng = np.random.default_rng(123)
    for i in range(14):
        for witness in ('relay','single'):
            multi = 3+i%2
            g = _fallback(rng,(0,5,2,1,8),multi,witness,i)
            parsed = _parse(g)
            assert _witness_ok(g,parsed,witness,multi)
            assert np.array_equal(solve(g),_apply(g,parsed))
    bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    cell = tuple(np.argwhere(_parse(bad['test'][0]['input'])['panel'])[0])
    bad['test'][0]['output'][cell] = (int(bad['test'][0]['output'][cell])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Legend corruption accepted')


if __name__ == '__main__':
    self_test()
    print('42-seed cavity relay self-test passed')
