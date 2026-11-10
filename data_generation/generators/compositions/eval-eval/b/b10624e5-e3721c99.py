"""Topology-selected, corner-aligned stamps clipped by enclosed chambers.

Construction uses a known lattice motif. solve decodes only actual input pixels.
No parent generator, source example, file, seed, or puzzle state is used by solve.
"""
from itertools import permutations
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    points = np.asarray(cells, dtype=int)
    low, high = points.min(0), points.max(0) + 1
    return int(low[0]), int(high[0]), int(low[1]), int(high[1])


def _cavities(cells):
    """Complement of one wall, padded with true exterior, in global coordinates."""
    r0, r1, c0, c1 = _box(cells)
    mask = np.zeros((r1-r0+2, c1-c0+2), dtype=bool)
    for r, c in cells:
        mask[r-r0+1, c-c0+1] = True
    holes = []
    for region in _components(~mask):
        if any(r in (0, mask.shape[0]-1) or c in (0, mask.shape[1]-1)
               for r, c in region):
            continue
        holes.append([(r+r0-1, c+c0-1) for r, c in region])
    return holes


def _decode(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and all(1 <= x <= 30 for x in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
    bg = int(grid[0, 0])
    assert np.all(grid[0] == bg) and np.all(grid[-1] == bg)
    assert np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)
    interpretations = []
    for wall in map(int, np.unique(grid)):
        if wall == bg:
            continue
        bodies = [{'cells': cells, 'holes': _cavities(cells)}
                  for cells in _components(grid == wall)]
        holes = [h for body in bodies for h in body['holes']]
        if len(holes) < 2:
            continue
        covered = {p for h in holes for p in h}
        other = {tuple(map(int, p)) for p in np.argwhere((grid != bg) & (grid != wall))}
        if not other or not other <= covered:
            continue
        color_sets = [set(int(grid[p]) for p in h) - {bg, wall} for h in holes]
        color_sets = [colors for colors in color_sets if colors]
        if len(color_sets) < 2:
            continue
        common = set.intersection(*color_sets)
        if len(common) != 1:
            continue
        anchor = next(iter(common))
        chambers, sources, good = [], [], True
        for bi, body in enumerate(bodies):
            body['marker_count'] = 0
            for region in body['holes']:
                marker = [p for p in region if int(grid[p]) == anchor]
                if not marker:
                    if all(int(grid[p]) == bg for p in region):
                        continue  # Empty chambers still contribute to the class.
                    good = False
                    break
                body['marker_count'] += 1
                r, b, c, e = _box(marker)
                side = b-r
                if side != e-c or len(marker) != side*side:
                    good = False
                    break
                r0, r1, c0, c1 = _box(region)
                if r1-r0 <= side or c1-c0 <= side:
                    good = False
                    break
                if r not in (r0, r1-side) or c not in (c0, c1-side):
                    good = False
                    break
                corner = (2 if r == r1-side else 0) + (1 if c == c1-side else 0)
                chamber = dict(cells=region, body=bi, anchor=(r, c, side),
                               corner=corner, box=(r0, r1, c0, c1))
                extras = [p for p in region if int(grid[p]) not in (bg, anchor)]
                if extras:
                    chamber['motif'] = {(y-r, x-c): int(grid[y, x])
                                        for y, x in region if int(grid[y, x]) != bg}
                    sources.append(chamber)
                chambers.append(chamber)
            if not good:
                break
        if not good or len(sources) != 1:
            continue
        source = sources[0]
        motif = source['motif']
        occupied = list(motif)
        br, er, bc, ec = _box(occupied)
        connected_mask = np.zeros((er-br, ec-bc), dtype=bool)
        for y, x in occupied:
            connected_mask[y-br, x-bc] = True
        if len(_components(connected_mask)) != 1 or len(set(motif.values()) - {anchor}) < 2:
            continue
        # Objects may not nest or contaminate one another's interiors.
        if any(int(grid[p]) == wall for h in holes for p in h):
            continue
        interpretations.append(dict(grid=grid, bg=bg, wall=wall, anchor=anchor,
                                    bodies=bodies, chambers=chambers, source=source,
                                    key=len(bodies[source['body']]['holes'])))
    assert len(interpretations) == 1, 'expected one observable wall/marker/reference interpretation'
    return interpretations[0]


def _scaled_stamp(data, side, corner, reflect=True):
    """Exact rational rasterization; equal-colored fine pixels may coalesce."""
    source = data['source']
    a = source['anchor'][2]
    motif = source['motif']
    r0, r1, c0, c1 = _box(list(motif))
    r0, c0 = r0//a*a, c0//a*a
    r1, c1 = ((r1+a-1)//a)*a, ((c1+a-1)//a)*a
    raw = np.full((r1-r0, c1-c0), data['bg'], dtype=np.int8)
    for (r, c), color in motif.items():
        raw[r-r0, c-c0] = color
    refined = np.repeat(np.repeat(raw, side, axis=0), side, axis=1)
    blocks = refined.reshape(refined.shape[0]//a, a, refined.shape[1]//a, a)
    minimum, maximum = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    assert np.array_equal(minimum, maximum), 'scale crosses a nonintegral color boundary'
    off_r, off_c = -r0*side//a, -c0*side//a
    delta = source['corner'] ^ corner if reflect else 0
    if delta & 2:
        minimum = minimum[::-1, :]
        off_r = minimum.shape[0]-off_r-side
    if delta & 1:
        minimum = minimum[:, ::-1]
        off_c = minimum.shape[1]-off_c-side
    return minimum, (off_r, off_c)


def _render(data, *, reflect=True, scale=True, mask=True, topology=True, complete=True,
            by_markers=False, drop_unmarked=False):
    output = data['grid'].copy()
    records = []
    source_markers = data['bodies'][data['source']['body']]['marker_count']
    def eligible(body):
        if drop_unmarked and body['marker_count'] < len(body['holes']):
            return False
        if not topology:
            return True
        return (body['marker_count'] == source_markers if by_markers else
                len(body['holes']) == data['key'])
    for bi, body in enumerate(data['bodies']):
        if not eligible(body):
            for p in body['cells']:
                output[p] = data['bg']
            for h in body['holes']:
                for p in h:
                    output[p] = data['bg']
    if not complete:
        return output, records
    for chamber in data['chambers']:
        if not eligible(data['bodies'][chamber['body']]):
            continue
        r, c, actual_side = chamber['anchor']
        side = actual_side if scale else data['source']['anchor'][2]
        stamp, (off_r, off_c) = _scaled_stamp(data, side, chamber['corner'], reflect)
        top, left = r-off_r, c-off_c
        r0, r1, c0, c1 = chamber['box']
        coords = chamber['cells'] if mask else [(y, x) for y in range(r0, r1) for x in range(c0, c1)]
        visible, intrusions = 0, 0
        chamber_set = set(chamber['cells'])
        for y in range(r0, r1):
            for x in range(c0, c1):
                sy, sx = y-top, x-left
                if 0 <= sy < stamp.shape[0] and 0 <= sx < stamp.shape[1] and stamp[sy, sx] != data['bg']:
                    intrusions += (y, x) not in chamber_set
        for y, x in coords:
            sy, sx = y-top, x-left
            if 0 <= sy < stamp.shape[0] and 0 <= sx < stamp.shape[1]:
                color = int(stamp[sy, sx])
                if color != data['bg']:
                    output[y, x] = color
                    visible += 1
        records.append(dict(body=chamber['body'], corner=chamber['corner'], side=actual_side,
                            delta=chamber['corner'] ^ data['source']['corner'],
                            total=int(np.count_nonzero(stamp != data['bg'])),
                            visible=visible, intrusions=intrusions,
                            source=chamber is data['source']))
    return output, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive class, contents, scales, orientations and stencil masks from input."""
    return _render(_decode(input_grid))[0]


def _motif(rng, bg, anchor, details, compact):
    a = int(rng.integers(1, 3))
    h, w = (3, 3) if compact else (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
    cells = {(r, c) for r in range(a) for c in range(a)}
    # A bent route guarantees real colored extent beyond both anchor edges.
    r, c = a-1, a-1
    route = [0]*(h-a) + [1]*(w-a)
    rng.shuffle(route)
    for axis in route:
        r += axis == 0
        c += axis == 1
        cells.add((r, c))
    target = int(rng.integers(len(cells), h*w+1))
    while len(cells) < target:
        frontier = sorted({q for r, c in cells for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                           if 0 <= q[0] < h and 0 <= q[1] < w and q not in cells})
        cells.add(frontier[int(rng.integers(len(frontier)))])
    raw = np.full((h, w), bg, dtype=np.int8)
    raw[:a, :a] = anchor
    others = sorted(cells - {(r, c) for r in range(a) for c in range(a)})
    rng.shuffle(others)
    for i, p in enumerate(others):
        raw[p] = details[i % len(details)] if i < len(details) else int(rng.choice(details))
    return raw, a


def _oriented(raw, corner):
    if corner & 2:
        raw = raw[::-1, :]
    if corner & 1:
        raw = raw[:, ::-1]
    return raw


def _known_stamp(raw, unit, corner, region, patch, bg):
    """Construction renderer: integer lattice tiles, not solver's rational pull."""
    image = _oriented(np.repeat(np.repeat(raw, unit, 0), unit, 1), corner)
    h, w = region.shape
    top = h-image.shape[0] if corner & 2 else 0
    left = w-image.shape[1] if corner & 1 else 0
    for y, x in np.argwhere(image != bg):
        yy, xx = int(y)+top, int(x)+left
        if 0 <= yy < h and 0 <= xx < w and region[yy, xx]:
            patch[yy, xx] = image[y, x]


def _hole(rng, h, w, side, corner, irregular):
    # Carve wall intrusions out of a chamber, retaining its marker corner.
    region = np.ones((h, w), dtype=bool)
    if irregular:
        cut_h = int(rng.integers(1, h-side+1))
        cut_w = int(rng.integers(1, w-side+1))
        region[-cut_h:, -cut_w:] = False
        if rng.random() < .4 and h > side+2:
            region[side:h-1, -1] = False
    return _oriented(region, corner).copy()


def _zero_body(rng, bg, wall):
    h, w = int(rng.integers(3, 8)), int(rng.integers(3, 9))
    mask = np.zeros((h, w), dtype=bool)
    style = int(rng.integers(4))
    mask[-1, :] = True
    mask[:, 0] = True
    if style == 0:
        mask[:, -1] = True  # open U, not a cavity
    elif style == 1:
        mask[h//2, :] = True
    elif style == 2:
        mask[:] = True
    inp = np.full(mask.shape, bg, dtype=np.int8)
    inp[mask] = wall
    inp = np.rot90(inp, int(rng.integers(4))).copy()
    return inp, np.full(inp.shape, bg, dtype=np.int8)


def _body(rng, count, raw, a, roles, source_corner, source_unit,
          source=False, retained=False, witness=False, clip_witness=False, test=False,
          marker_count=None):
    bg, wall, anchor = roles
    if count == 0:
        return _zero_body(rng, bg, wall)
    if count <= 2:
        rows, cols = (1, count) if rng.random() < .5 else (count, 1)
    else:
        options = [(2, (count+1)//2), ((count+1)//2, 2)]
        if not test and rng.random() < .15:
            options.append((1, count))
        rows, cols = options[int(rng.integers(len(options)))]
    regions, units, corners = [], [], []
    if marker_count is None:
        # A donor always has a decorated marker, and retained targets remain
        # useful. Optional empty chambers decouple topology from marker counts.
        marker_count = int(rng.integers(1, count+1))
    marked = ({0} | set(int(i) for i in rng.permutation(np.arange(1, count))[:marker_count-1])
              if marker_count else set())
    for i in range(count):
        is_ref = source and i == 0
        if witness:
            ci = i if source else i+2
            corner = source_corner ^ ci
            unit = source_unit if ci != 2 else (1 if source_unit == 2 else 2)
        elif is_ref:
            corner, unit = source_corner, source_unit
        elif clip_witness and i == 0:
            corner, unit = int(rng.integers(4)), 2
        else:
            corner, unit = int(rng.integers(4)), int(rng.choice([1, 1, 2]))
        side = a*unit
        if is_ref or witness:
            h, w = raw.shape[0]*unit, raw.shape[1]*unit
            h += int(rng.integers(0, 2))
            w += int(rng.integers(0, 2))
            irregular = False
        elif clip_witness and i == 0:
            h = w = 5
            irregular = True
        else:
            limit = 5 if test else 7
            h, w = int(rng.integers(side+1, max(side+2, limit+1))), int(rng.integers(side+1, max(side+2, limit+1)))
            irregular = bool(rng.random() < (.8 if test else .55))
        regions.append(_hole(rng, h, w, side, corner, irregular))
        corners.append(corner)
        units.append(unit)
    row_sizes = [max(regions[i].shape[0] for i in range(count) if i//cols == r) for r in range(rows)]
    col_sizes = [max(regions[i].shape[1] for i in range(count) if i%cols == c) for c in range(cols)]
    wall_width = 1 if witness or clip_witness or test else int(rng.choice([1, 1, 2]))
    h = sum(row_sizes)+(rows+1)*wall_width
    w = sum(col_sizes)+(cols+1)*wall_width
    if h > 27 or w > 27:
        return None
    inp = np.full((h, w), wall, dtype=np.int8)
    holes = []
    for i, region in enumerate(regions):
        ri, ci = divmod(i, cols)
        y = sum(row_sizes[:ri])+(ri+1)*wall_width + int(rng.integers(row_sizes[ri]-region.shape[0]+1))
        x = sum(col_sizes[:ci])+(ci+1)*wall_width + int(rng.integers(col_sizes[ci]-region.shape[1]+1))
        patch = inp[y:y+region.shape[0], x:x+region.shape[1]]
        patch[region] = bg
        side = a*units[i]
        ar = region.shape[0]-side if corners[i] & 2 else 0
        ac = region.shape[1]-side if corners[i] & 1 else 0
        assert region[ar:ar+side, ac:ac+side].all()
        if i in marked:
            patch[ar:ar+side, ac:ac+side] = anchor
        if source and i == 0:
            _known_stamp(raw, units[i], corners[i], region, patch, bg)
        holes.append((y, x, region))
    # Sculpt exterior only when it preserves both wall connectivity and every
    # original chamber. Corner trimming can expose a diagonal-only escape.
    for _ in range(int(rng.integers(1, 8))):
        candidate = inp.copy()
        corner = int(rng.integers(4))
        rr = 0 if not corner & 2 else h-1
        cc = 0 if not corner & 1 else w-1
        depth = int(rng.integers(1, wall_width+1))
        rstart, cstart = (0 if rr == 0 else h-depth), (0 if cc == 0 else w-depth)
        candidate[rstart:rstart+depth, cstart:cstart+depth] = bg
        components = _components(candidate == wall)
        if len(components) == 1 and len(_cavities(components[0])) == count:
            inp = candidate
    components = _components(inp == wall)
    if len(components) != 1 or len(_cavities(components[0])) != count:
        return None
    out = inp.copy() if retained else np.full(inp.shape, bg, dtype=np.int8)
    if retained:
        for i, (y, x, region) in enumerate(holes):
            if i in marked:
                _known_stamp(raw, units[i], corners[i], region,
                             out[y:y+region.shape[0], x:x+region.shape[1]], bg)
    return inp, out


def _pack(rng, items, bg):
    """Find fitting shelf partitions; no shelf or divider is drawn in the task."""
    n = len(items)
    candidates = []
    for order in permutations(range(n)):
        for breaks in range(1 << (n-1)):
            rows = [[order[0]]]
            for i in range(1, n):
                if breaks & (1 << (i-1)):
                    rows.append([])
                rows[-1].append(order[i])
            heights = [max(items[i][0].shape[0] for i in row) for row in rows]
            widths = [sum(items[i][0].shape[1] for i in row)+len(row)-1 for row in rows]
            h, w = sum(heights)+len(rows)+1, max(widths)+2
            if h <= 30 and w <= 30:
                candidates.append((rows, heights, widths, h, w))
    if not candidates:
        return None
    rows, heights, widths, mh, mw = candidates[int(rng.integers(len(candidates)))]
    h, w = int(rng.integers(mh, min(30, mh+6)+1)), int(rng.integers(mw, min(30, mw+6)+1))
    inp, out = np.full((h, w), bg, dtype=np.int8), np.full((h, w), bg, dtype=np.int8)
    # Distribute vertical slack among margins/gaps, and horizontal slack locally.
    gaps = np.ones(len(rows)+1, dtype=int)
    for _ in range(h-mh):
        gaps[int(rng.integers(len(gaps)))] += 1
    y = int(gaps[0])
    for ri, row in enumerate(rows):
        xgaps = np.ones(len(row)+1, dtype=int)
        for _ in range(w-widths[ri]-2):
            xgaps[int(rng.integers(len(xgaps)))] += 1
        x = int(xgaps[0])
        for ci, index in enumerate(row):
            a, b = items[index]
            dy = int(rng.integers(heights[ri]-a.shape[0]+1))
            inp[y+dy:y+dy+a.shape[0], x:x+a.shape[1]] = a
            out[y+dy:y+dy+a.shape[0], x:x+a.shape[1]] = b
            x += a.shape[1]+int(xgaps[ci+1])
        y += heights[ri]+int(gaps[ri+1])
    rotation = int(rng.integers(4))
    return {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(out, rotation).copy()}


def _make_pair(rng, roles, details, key, source_corner, mode):
    compact = mode in ('witness', 'clip')
    raw, a = _motif(rng, roles[0], roles[2], details, compact)
    unit = int(rng.integers(1, 3)) if mode == 'witness' else 1
    counts = [key, key]
    if mode == 'clip':
        counts.append(key)  # An entirely unmarked matching mold is retained.
    optional = int(rng.integers(2)) if mode not in ('witness', 'clip') else 0
    if optional and rng.random() < .5:
        counts.append(key)
    absent = [k for k in range(5) if k != key]
    counts.append(3 if mode == 'witness' else 2 if mode == 'clip' else int(rng.choice(absent)))
    if optional and len(counts) == 3:
        counts.append(int(rng.choice(absent)))
    items = []
    for i, k in enumerate(counts):
        marker_count = (k if mode == 'witness' and i < 2 else
                        key if mode == 'witness' and i == 2 else
                        0 if mode == 'clip' and i == 2 else
                        key if mode == 'clip' and i == 3 else None)
        item = _body(rng, k, raw, a, roles, source_corner, unit,
                     source=i == 0, retained=k == key,
                     witness=mode == 'witness' and i < 2,
                     clip_witness=mode == 'clip' and i == 1, test=mode == 'test',
                     marker_count=marker_count)
        if item is None:
            return None
        items.append(item)
    pair = _pack(rng, items, roles[0])
    if pair is None:
        return None
    data = _decode(pair['input'])
    predicted, records = _render(data)
    assert np.array_equal(predicted, pair['output']), 'construction and input decoder disagree'
    if not any(r['visible'] > r['side']**2 for r in records if not r['source']):
        return None
    if mode == 'clip' and not any(r['intrusions'] for r in records):
        return None
    return pair


def _grid_key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.uint8).tobytes()


def _fallback_pair(roles, details, key, corner, mode, seen):
    """Finite fitting construction, preserving the same teaching witnesses.

    At least 160 placements of a removed mold ensure freshness against <=13 pairs.
    This is a simpler member of the rule, not an alternate answer convention.
    """
    bg, wall, anchor = roles
    raw = np.array([[anchor, details[0], bg], [bg, details[0], bg],
                    [bg, details[0], details[1]]], dtype=np.int8)
    items = []
    for source in (True, False):
        nr, nc = (1, key) if key <= 2 else (2, 2)
        regions, units, corners = [], [], []
        for i in range(key):
            u = 1 if source or i else 2
            size = 3 if u == 1 else 6 if mode == 'witness' else 5 if mode == 'clip' else 3
            region = np.ones((size, size), dtype=bool)
            if u == 2 and mode != 'witness':
                region[-1, -1] = False
            co = corner ^ ((i if source else i+key) % 4)
            regions.append(_oriented(region, co).copy())
            units.append(u)
            corners.append(co)
        rh = [max(regions[i].shape[0] for i in range(key) if i//nc == r) for r in range(nr)]
        cw = [max(regions[i].shape[1] for i in range(key) if i%nc == c) for c in range(nc)]
        inp = np.full((sum(rh)+nr+1, sum(cw)+nc+1), wall, dtype=np.int8)
        targets = []
        for i, region in enumerate(regions):
            ri, ci = divmod(i, nc)
            r, c = sum(rh[:ri])+ri+1, sum(cw[:ci])+ci+1
            patch = inp[r:r+region.shape[0], c:c+region.shape[1]]
            patch[region] = bg
            s = units[i]
            ar = region.shape[0]-s if corners[i] & 2 else 0
            ac = region.shape[1]-s if corners[i] & 1 else 0
            patch[ar:ar+s, ac:ac+s] = anchor
            if source and i == 0:
                _known_stamp(raw, 1, corners[i], region, patch, bg)
            targets.append((r, c, region))
        out = inp.copy()
        for i, (r, c, region) in enumerate(targets):
            _known_stamp(raw, units[i], corners[i], region,
                         out[r:r+region.shape[0], c:c+region.shape[1]], bg)
        items.append((inp, out))
    if mode == 'clip':
        blank = np.full((5,5), wall, dtype=np.int8)
        blank[1:4,1:4] = bg
        items.append((blank, blank.copy()))
    base_in = np.full((30, 30), bg, dtype=np.int8)
    base_out = base_in.copy()
    x = 1
    for inp, out in items:
        base_in[1:1+inp.shape[0], x:x+inp.shape[1]] = inp
        base_out[1:1+out.shape[0], x:x+out.shape[1]] = out
        x += inp.shape[1]+1
    # A different-class mold has the same marker count as the donor, with one
    # extra unmarked chamber. This preserves the topology-versus-marker witness.
    dk = key+1
    dr, dc = (1, dk) if dk <= 2 else (2, (dk+1)//2)
    distractor = np.full((4*dr+1, 4*dc+1), wall, dtype=np.int8)
    for i in range(dk):
        y, x = 1+4*(i//dc), 1+4*(i%dc)
        distractor[y:y+3, x:x+3] = bg
        if i < key:
            distractor[y, x] = anchor
    for r in range(11, 30-distractor.shape[0]):
        for c in range(1, 30-distractor.shape[1]):
            inp = base_in.copy()
            inp[r:r+distractor.shape[0], c:c+distractor.shape[1]] = distractor
            if _grid_key(inp) not in seen:
                return {'input': inp, 'output': base_out.copy()}
    raise AssertionError('finite fallback placement space exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    num_train, num_test = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    stable_palette = bool(rng.random() < .35)
    base_corner = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            mode = 'test' if split == 'test' else 'witness' if i == 0 else 'clip' if i == 1 else 'mixed'
            key = int(rng.integers(3, 5)) if split == 'test' else 2 if i == 0 else 1 if i == 1 else int(rng.integers(1, 3))
            corner = base_corner if i == 0 else base_corner ^ int(rng.integers(1, 4))
            if not stable_palette:
                palette = [int(c) for c in rng.permutation(10)]
            roles, details = tuple(palette[:3]), palette[3:3+int(rng.integers(2, 4))]
            for attempt in range(192):
                pair = _make_pair(rng, roles, details, key, corner, mode)
                if pair is not None and split == 'train' and i == 1:
                    # Whole-scene rotation must not accidentally cancel the
                    # deliberately changed source corner in the second demo.
                    previous_corner = _decode(puzzle['train'][0]['input'])['source']['corner']
                    if _decode(pair['input'])['source']['corner'] == previous_corner:
                        continue
                if pair is not None and _grid_key(pair['input']) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_grid_key(pair['input']))
                    break
            else:
                if split == 'train' and i == 1:
                    corner = _decode(puzzle['train'][0]['input'])['source']['corner'] ^ 1
                pair = _fallback_pair(roles, details, key, corner, mode, seen)
                puzzle[split].append(pair)
                seen.add(_grid_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def family_features(pair):
    data = _decode(pair['input'])
    output, records = _render(data)
    assert np.array_equal(output, np.asarray(pair['output']))
    removed = [b for b in data['bodies'] if len(b['holes']) != data['key']]
    shapes = []
    irregular = 0
    for body in data['bodies']:
        for region in body['holes']:
            r0, r1, c0, c1 = _box(region)
            shapes.append((r1-r0, c1-c0))
            irregular += len(region) != (r1-r0)*(c1-c0)
    sr = data['source']['anchor'][2]
    motif = data['source']['motif']
    r0, r1, c0, c1 = _box(list(motif))
    return dict(shape=np.asarray(pair['input']).shape, roles=(data['bg'], data['wall'], data['anchor']),
                key=data['key'], bodies=len(data['bodies']), counts=tuple(len(b['holes']) for b in data['bodies']),
                marker_counts=tuple(b['marker_count'] for b in data['bodies']),
                empty_chambers=sum(len(b['holes'])-b['marker_count'] for b in data['bodies']),
                removed=len(removed), cavity_shapes=tuple(shapes), irregular=irregular,
                source_corner=data['source']['corner'], source_side=sr,
                corners=tuple(r['corner'] for r in records), deltas=tuple(r['delta'] for r in records),
                ratios=tuple(sorted({(r['side'], sr) for r in records})),
                full_scale=any(not r['source'] and r['side'] != sr and r['total'] == r['visible'] for r in records),
                clipped=sum(r['visible'] < r['total'] for r in records),
                intrusions=sum(r['intrusions'] for r in records),
                motif_span=(r1-r0, c1-c0), motif_cells=len(motif),
                completed=int(np.count_nonzero((data['grid'] == data['bg']) & (output != data['bg']))))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                a = np.asarray(grid)
                assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
                assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            data = _decode(inp)
            predicted, records = _render(data)
            assert np.array_equal(predicted, out), 'wrong cavity selection or anchored stencil'
            assert not np.array_equal(inp, out)
            assert any(len(b['holes']) != data['key'] for b in data['bodies'])
            assert sum(len(b['holes']) == data['key'] for b in data['bodies']) >= 2
            # All retained walls and the complete reference chamber are preserved.
            for body in data['bodies']:
                if len(body['holes']) == data['key']:
                    assert all(int(out[p]) == data['wall'] for p in body['cells'])
            assert all(out[p] == inp[p] for p in data['source']['cells'])
            feature = family_features(pair)
            assert feature['completed'] > 0
            if split == 'train':
                training.append((data, feature))
    features = [f for _, f in training]
    assert len({f['key'] for f in features}) >= 2, 'demonstrate reference-relative class equality'
    assert {1, 2, 3} <= {d for f in features for d in f['deltas']}, 'show every reflection'
    assert any(f['full_scale'] for f in features), 'show a fully visible unequal scale'
    assert any(f['intrusions'] > 0 for f in features), 'show actual cavity-mask clipping'
    for option in ('reflect', 'scale', 'mask', 'topology', 'complete'):
        assert any(not np.array_equal(_render(d, **{option: False})[0], _render(d)[0])
                   for d, _ in training), f'training fails to distinguish {option}'
    assert any(not np.array_equal(_render(d, by_markers=True)[0], _render(d)[0])
               for d, _ in training), 'training fails to distinguish hole and marker counts'
    assert any(not np.array_equal(_render(d, drop_unmarked=True)[0], _render(d)[0])
               for d, _ in training), 'training fails to distinguish erasing every empty chamber'


def _puzzle_key(puzzle):
    return tuple((_grid_key(p['input']), _grid_key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test() -> None:
    # Explicit topology fixtures: diagonal escape is enclosed, U opening is not.
    diagonal = np.array([[0,1,1], [1,0,1], [1,1,1]], dtype=bool)
    assert len(_cavities(_components(diagonal)[0])) == 1
    diagonal[0, 1] = False
    assert len(_cavities(_components(diagonal)[0])) == 0
    u = np.array([[1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1]], dtype=bool)
    assert len(_cavities(_components(u)[0])) == 0
    # A hand-built member outside the sampled key range and L-shaped hole masks.
    # Five chambers, with C-shaped target cavities and an erased open U.
    raw = np.array([[7,2,0], [0,2,0], [0,2,3]], dtype=np.int8)
    fixture_in = np.zeros((17,30), dtype=np.int8)
    fixture_out = fixture_in.copy()
    for is_source, left in ((True, 1), (False, 15)):
        body_in = np.full((9,13), 5, dtype=np.int8)
        pending = []
        for i in range(5):
            r, c = 1+4*(i//3), 1+4*(i%3)
            corner = i % 4
            region = np.ones((3,3), dtype=bool)
            if not is_source:
                region[1, 1:] = False
                region = _oriented(region, corner).copy()
            patch = body_in[r:r+3, c:c+3]
            patch[region] = 0
            ar, ac = (2 if corner & 2 else 0), (2 if corner & 1 else 0)
            marked = is_source or i != 4
            if marked:
                patch[ar, ac] = 7
            if is_source and i == 0:
                _known_stamp(raw, 1, corner, region, patch, 0)
            pending.append((r,c,region,corner,marked))
        body_out = body_in.copy()
        for r,c,region,corner,marked in pending:
            if marked:
                _known_stamp(raw, 1, corner, region, body_out[r:r+3,c:c+3], 0)
        fixture_in[1:10,left:left+13] = body_in
        fixture_out[1:10,left:left+13] = body_out
    fixture_in[12:15,1] = fixture_in[12:15,5] = 5
    fixture_in[14,1:6] = 5
    assert _decode(fixture_in)['key'] == 5
    assert np.array_equal(solve(fixture_in), fixture_out)
    assert not np.array_equal(_render(_decode(fixture_in), by_markers=True)[0], fixture_out)
    fine = dict(bg=0, source=dict(anchor=(0,0,2), corner=0,
                motif={(0,0):7, (0,1):7, (1,0):7, (1,1):7, (2,0):2, (2,1):3}))
    _scaled_stamp(fine, 4, 3)
    try:
        _scaled_stamp(fine, 1, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError('nonintegral fine-detail reduction accepted')
    features, signatures, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert family_features(puzzle['train'][0])['empty_chambers'] > 0
        signatures.add(_puzzle_key(puzzle))
        if seed < 3:
            assert _puzzle_key(generate_puzzle(seed)) == _puzzle_key(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                features.append(family_features(pair))
                a = pair['input']
                assert np.array_equal(solve(a.tolist()), pair['output'])
                colors = {}
                structures.add((a.shape, tuple(colors.setdefault(int(v), len(colors)) for v in a.flat)))
    assert len(signatures) == 42 and len(structures) >= 180
    assert len({f['shape'] for f in features}) >= 40
    assert {f['key'] for f in features} == {1, 2, 3, 4}
    assert {f['source_corner'] for f in features} == {0, 1, 2, 3}
    assert len({f['roles'] for f in features}) >= 35
    assert len({f['cavity_shapes'] for f in features}) >= 80
    assert len({f['motif_span'] for f in features}) >= 10
    assert {f['bodies'] for f in features} == {3, 4}
    assert any(0 in f['counts'] for f in features)
    assert any(f['source_side'] > 1 for f in features)
    assert any(f['clipped'] == 0 for f in features) and any(f['clipped'] >= 3 for f in features)
    assert sum(f['irregular'] > 0 for f in features) > 100
    assert sum(f['empty_chambers'] > 0 for f in features) > 120
    # Directly exercise finite packing recovery, including every reference count.
    fallback_seen = set()
    for key, mode in ((2, 'witness'), (1, 'clip'), (2, 'mixed'), (3, 'test'), (4, 'test')):
        pair = _fallback_pair((0, 5, 7), [2, 3], key, key % 4, mode, fallback_seen)
        fallback_seen.add(_grid_key(pair['input']))
        assert np.array_equal(solve(pair['input']), pair['output'])
        f = family_features(pair)
        assert f['completed'] > 0 and f['removed'] == 1
        if mode == 'witness':
            assert f['full_scale'] and set(f['deltas']) == {0, 1, 2, 3}
        if mode == 'clip':
            assert f['intrusions'] > 0
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(1000+train*10+test, train, test)
            assert (len(p['train']), len(p['test'])) == (train, test)
    for train, test in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (2.5,1), (None,1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Consistent recoloring and a rotation preserve the input-derived rule.
    p = generate_puzzle(712, 2, 1)
    perm = np.array([7,2,9,5,1,8,0,4,6,3], dtype=np.int8)
    for pair in p['train']+p['test']:
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
        assert np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output']))
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted answer accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, family invariants, explicit counts and counterfactuals)')
