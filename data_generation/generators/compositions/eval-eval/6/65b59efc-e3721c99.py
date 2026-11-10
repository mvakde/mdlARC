"""Cavity-addressed stamp expansion. All command data are visible in the input.

The upper glyphs classify objects by holes, the framed middle bitmaps are
matched to glyphs by color, and the lower framed scene supplies placement
masks through its enclosed background. No parent generator is imported.
"""

from collections import deque
import json
import numpy as np


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("Expected an integer ARC grid, with axes 1..30 and colors 0..9")
    return g.astype(np.int8, copy=False)


def _components(mask):
    """Four-neighbor components, in row-major order."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo = deque([(r, c)])
        cells = []
        while todo:
            a, b = todo.popleft()
            cells.append((a, b))
            for x, y in ((a-1, b), (a+1, b), (a, b-1), (a, b+1)):
                if 0 <= x < h and 0 <= y < w and mask[x, y] and not seen[x, y]:
                    seen[x, y] = True
                    todo.append((x, y))
        result.append(cells)
    return result


def _crop_cells(cells):
    rc = np.asarray(cells, dtype=int)
    lo, hi = rc.min(axis=0), rc.max(axis=0) + 1
    mask = np.zeros(tuple(hi-lo), dtype=bool)
    mask[tuple((rc-lo).T)] = True
    return mask, tuple(int(v) for v in lo)


def _holes(mask):
    """Cavity coordinates relative to mask; exterior padding is always open."""
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    h, w = padded.shape
    return [[(r-1, c-1) for r, c in cells]
            for cells in _components(~padded)
            if all(0 < r < h-1 and 0 < c < w-1 for r, c in cells)]


def _dihedral(mask, code):
    return np.rot90(np.fliplr(mask) if code >= 4 else mask, code % 4).copy()


def _rectangle(shape):
    m = np.ones(shape, dtype=bool)
    m[1:-1, 1:-1] = False
    return m


def _read_frames(region, bg, sep):
    frames = []
    covered = np.zeros(region.shape, dtype=bool)
    for cells in _components(region == sep):
        mask, (r, c) = _crop_cells(cells)
        h, w = mask.shape
        if h < 4 or w < 4 or not np.array_equal(mask, _rectangle((h, w))):
            raise ValueError("Malformed rectangular registration frame")
        covered[r:r+h, c:c+w] = True
        frames.append((r, c, region[r+1:r+h-1, c+1:c+w-1].copy()))
    if not frames or np.any(region[~covered] != bg):
        raise ValueError("Unregistered pixels outside frames")
    return sorted(frames, key=lambda x: (x[1], x[0]))


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if any(int(g[r, c]) != bg for r, c in
           ((0, 0), (0, g.shape[1]-1), (g.shape[0]-1, 0),
            (g.shape[0]-1, g.shape[1]-1))):
        raise ValueError("The four outer corners must be background")
    bars = [r for r in range(g.shape[0])
            if g[r, 0] != bg and np.all(g[r] == g[r, 0])]
    if len(bars) != 2:
        raise ValueError("Expected two full-width library dividers")
    a, b = bars
    sep = int(g[a, 0])
    if int(g[b, 0]) != sep or a < 3 or b <= a+3 or b >= g.shape[0]-4:
        raise ValueError("Invalid three-band presentation")
    glyph_strip = g[:a]
    if np.any(glyph_strip == sep):
        raise ValueError("Separator color is reserved for registration")
    keys = []
    by_color = {}
    by_count = {}
    for color in np.unique(glyph_strip):
        color = int(color)
        if color == bg:
            continue
        comps = _components(glyph_strip == color)
        if len(comps) != 1:
            raise ValueError("Each key color must have one connected prototype")
        mask, pos = _crop_cells(comps[0])
        count = len(_holes(mask))
        if not 1 <= count <= 5 or count in by_count:
            raise ValueError("Key classes must be distinct positive cavity counts 1..5")
        # This forbids a foreign glyph inside another glyph's cavity.
        if any(glyph_strip[pos[0]+r, pos[1]+c] != bg
               for hole in _holes(mask) for r, c in hole):
            raise ValueError("Prototype cavities must contain only background")
        keys.append((pos, count, color))
        by_color[color] = count
        by_count[count] = color
    if not 1 <= len(keys) <= 4:
        raise ValueError("Expected one to four key entries")
    keys.sort(key=lambda x: (x[0][1], x[0][0]))
    tile_frames = _read_frames(g[a+1:b], bg, sep)
    tiles = {}
    tile_order = []
    sizes = set()
    for _, _, tile in tile_frames:
        n, w = tile.shape
        if n != w or not 2 <= n <= 4:
            raise ValueError("Stamps must be square with common side 2..4")
        colors = np.unique(tile[tile != bg])
        if len(colors) != 1:
            raise ValueError("A stamp must have one visible identity color")
        color = int(colors[0])
        if color not in by_color or color in tiles:
            raise ValueError("Stamp identities must match key identities exactly")
        tiles[color] = tile == color
        tile_order.append(color)
        sizes.add(n)
    if set(tiles) != set(by_color) or len(sizes) != 1:
        raise ValueError("Missing stamp or inconsistent stamp scale")
    n = sizes.pop()
    frames = _read_frames(g[b+1:], bg, sep)
    if len(frames) != 1:
        raise ValueError("Expected exactly one framed work scene")
    field = frames[0][2]
    if max(field.shape)*n > 30:
        raise ValueError("Expanded work scene exceeds ARC dimensions")
    colors = np.unique(field[field != bg])
    if len(colors) != 1:
        raise ValueError("The work scene must have one neutral query color")
    query = int(colors[0])
    if query in (bg, sep) or query in by_color:
        raise ValueError("Query, background, separator, and key roles must be distinct")
    objects = []
    for cells in _components(field == query):
        mask, (r, c) = _crop_cells(cells)
        holes = [[(r+x, c+y) for x, y in hole] for hole in _holes(mask)]
        if any(field[x, y] != bg for hole in holes for x, y in hole):
            raise ValueError("Objects cannot be nested or contain foreign pixels")
        objects.append(dict(cells=cells, holes=holes, mask=mask, pos=(r, c)))
    return dict(bg=bg, sep=sep, query=query, n=n, keys=keys,
                mapping=by_count, tiles=tiles, tile_order=tile_order,
                field=field, objects=objects)


def _render(p, shortcut=None, transform=0):
    """Input-derived renderer and explicit wrong-rule counterfactuals."""
    n, bg = p['n'], p['bg']
    h, w = p['field'].shape
    out = np.full((h*n, w*n), bg, dtype=np.int8)
    mapping = p['mapping']
    if shortcut == 'rank':
        mapping = {i+1: color for i, (_, _, color) in enumerate(p['keys'])}
    for obj in p['objects']:
        count = len(obj['holes'])
        if shortcut == 'area':
            count = sum(map(len, obj['holes']))
        color = mapping.get(count)
        if shortcut == 'ignore_topology':
            color = p['keys'][0][2]
        if color is None:
            continue
        tile = p['tiles'][color]
        if shortcut == 'position':
            index = [e[2] for e in p['keys']].index(color)
            tile = p['tiles'][p['tile_order'][index]]
        if shortcut == 'solid':
            tile = np.ones_like(tile)
        tile = _dihedral(tile, transform)
        coords = [q for hole in obj['holes'] for q in hole]
        if shortcut == 'walls':
            coords = obj['cells']
        elif shortcut == 'one_per_hole':
            coords = [min(hole) for hole in obj['holes']]
        for r, c in coords:
            out[r*n:(r+1)*n, c*n:(c+1)*n] = np.where(tile, color, bg)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using this input alone; do not mutate the supplied grid."""
    return _render(_parse(input_grid))


def _compact(count):
    if count == 0:
        return np.ones((1, 2), dtype=bool)
    rows = 1 if count <= 2 else 2
    cols = (count+rows-1)//rows
    mask = np.ones((2*rows+1, 2*cols+1), dtype=bool)
    for i in range(count):
        mask[1+2*(i//cols), 1+2*(i % cols)] = False
    return mask


def _prototype(rng, count):
    mask = _compact(count)
    if count == 3 and rng.random() < .5:
        mask = np.ones((3, 7), dtype=bool)
        mask[1, 1::2] = False
    # Stretching keeps the topological class but changes hole area/proportions.
    if rng.random() < .45:
        axis = int(rng.integers(2))
        index = int(rng.integers(mask.shape[axis]))
        candidate = np.insert(mask, index, np.take(mask, index, axis=axis), axis=axis)
        if candidate.shape[0] <= 5 and candidate.shape[1] <= 9:
            mask = candidate
    candidate = _dihedral(mask, int(rng.integers(8)))
    return candidate if candidate.shape[0] <= 5 else mask


def _motifs(rng, n, count, witness=False):
    for _ in range(64):
        masks = []
        for _ in range(count):
            style = int(rng.integers(6))
            rr, cc = np.indices((n, n))
            if witness or style < 2:
                mask = rng.random((n, n)) < rng.uniform(.25, .8)
            elif style == 2:
                mask = (rr+cc) % 2 == int(rng.integers(2))
            elif style == 3:
                mask = _rectangle((n, n))
            elif style == 4:
                mask = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
            else:
                mask = rng.random((n, n)) < .95
            if not mask.any():
                mask.flat[int(rng.integers(n*n))] = True
            masks.append(mask)
        if not witness or (all(not m.all() for m in masks)
                           and len({m.tobytes() for m in masks}) == count
                           and all(any(not np.array_equal(m, _dihedral(m, t))
                                       for m in masks) for t in range(1, 8))):
            return masks
    # Two varied asymmetric anchors jointly determine every dihedral action.
    masks[0] = np.zeros((n, n), dtype=bool)
    masks[0][0, 0] = True
    masks[1] = np.zeros((n, n), dtype=bool)
    masks[1][0, :] = True
    return masks


def _object(rng, count, rich=False):
    if count == 0:
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = mask[-1, :] = True
        if rng.random() < .5:
            mask[:, -1] = True  # Open U, with no enclosed region.
        return _dihedral(mask, int(rng.integers(8)))
    if not rich and rng.random() < .6:
        return _dihedral(_compact(count), int(rng.integers(8)))
    # Carve separated, sometimes bent cavities into a connected solid contour.
    for _ in range(24):
        h = int(rng.integers(4, 8))
        w = int(rng.integers(4, 9))
        holes = []
        available = [(r, c) for r in range(1, h-1) for c in range(1, w-1)]
        rng.shuffle(available)
        for cell in available:
            if all(abs(cell[0]-r)+abs(cell[1]-c) > 1 for hole in holes for r, c in hole):
                holes.append({cell})
                if len(holes) == count:
                    break
        if len(holes) != count:
            continue
        for i, hole in enumerate(holes):
            for _ in range(int(rng.integers(1, 7))):
                frontier = sorted({(x, y) for r, c in hole
                                   for x, y in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                                   if 0 < x < h-1 and 0 < y < w-1 and (x,y) not in hole
                                   and all(abs(x-a)+abs(y-b) > 1 for j, other in enumerate(holes)
                                           if j != i for a,b in other)})
                if not frontier:
                    break
                hole.add(frontier[int(rng.integers(len(frontier)))])
        mask = np.ones((h, w), dtype=bool)
        for hole in holes:
            for r, c in hole:
                mask[r, c] = False
        if len(_components(mask)) != 1 or len(_holes(mask)) != count:
            continue
        # A corner or an open notch can meet the exterior only diagonally.
        for _ in range(5):
            candidate = mask.copy()
            side = int(rng.integers(4))
            r = (0 if side == 0 else h-1) if side < 2 else int(rng.integers(h))
            c = int(rng.integers(w)) if side < 2 else (0 if side == 2 else w-1)
            candidate[r, c] = False
            if len(_components(candidate)) == 1 and len(_holes(candidate)) == count:
                mask = candidate
        return _dihedral(mask, int(rng.integers(8)))
    return _dihedral(_compact(count), int(rng.integers(8)))


def _positions(free, shape):
    h, w = shape
    if h > free.shape[0] or w > free.shape[1]:
        return []
    return [(r, c) for r in range(free.shape[0]-h+1)
            for c in range(free.shape[1]-w+1) if free[r:r+h, c:c+w].all()]


def _put(field, free, mask, pos, query):
    r, c = pos
    h, w = mask.shape
    field[r:r+h, c:c+w][mask] = query
    free[max(0,r-1):r+h+1, max(0,c-1):c+w+1] = False
    return mask.copy(), (r, c)


def _pack_witness(rng, shape, bg, query, index):
    one = np.ones((4, 4), dtype=bool)
    one[1, 1] = one[1, 2] = one[2, 1] = False  # One L-shaped, three-cell cavity.
    two, three = _compact(2), _compact(3)
    if index == 1:
        # Equal bounding boxes with different topology defeat size lookup.
        two = np.ones((4,5), dtype=bool)
        two[1,1] = two[2,3] = False
        three = np.ones((4,5), dtype=bool)
        three[1,1] = three[1,3] = three[2,2] = False
    turn = int(rng.integers(8))
    masks = [_dihedral(one, int(rng.integers(8))), _dihedral(two, turn),
             _dihedral(three, turn if index == 1 else int(rng.integers(8)))]
    for _ in range(32):
        field = np.full(shape, bg, dtype=np.int8)
        free = np.ones(shape, dtype=bool)
        records = []
        for i in rng.permutation(3):
            mask = masks[int(i)]
            locations = _positions(free, mask.shape)
            if not locations:
                break
            records.append(_put(field, free, mask, locations[int(rng.integers(len(locations)))], query))
        if len(records) == 3:
            return field, free, records
    # Guaranteed fitting fallback: two shapes across the top, one below.
    field = np.full(shape, bg, dtype=np.int8)
    free = np.ones(shape, dtype=bool)
    masks = [one, two, three]
    positions = ((0,0), (0,5), (5,0)) if index == 0 else ((0,6), (0,0), (5,0))
    records = [_put(field, free, mask, pos, query)
               for mask, pos in zip(masks, positions)]
    if index % 2:
        field = np.fliplr(field).copy()
        free = np.fliplr(free).copy()
        records = [(np.fliplr(m).copy(), (r, shape[1]-c-m.shape[1]))
                   for m, (r,c) in records]
    return field, free, records


def _compose_input(rng, prototypes, colors, masks, glyph_order, tile_order, field, bg, sep):
    n = masks[0].shape[0]
    gh = max(m.shape[0] for m in prototypes)
    glyph_width = sum(m.shape[1] for m in prototypes) + len(colors)-1 + 2
    tile_width = len(colors)*(n+2) + len(colors)-1 + 2
    width = max(glyph_width, tile_width, field.shape[1]+4)
    width += int(rng.integers(min(3, 30-width)+1))
    a, b = gh+2, gh+2+1+n+4
    grid = np.full((b+1+field.shape[0]+4, width), bg, dtype=np.int8)
    grid[a, :] = grid[b, :] = sep
    c = 1+int(rng.integers(width-glyph_width+1))
    for i in glyph_order:
        proto = prototypes[i]
        r = 1+int(rng.integers(gh-proto.shape[0]+1))
        grid[r:r+proto.shape[0], c:c+proto.shape[1]][proto] = colors[i]
        c += proto.shape[1]+1
    c = 1+int(rng.integers(width-tile_width+1))
    for i in tile_order:
        r = a+2
        patch = grid[r:r+n+2, c:c+n+2]
        patch[_rectangle(patch.shape)] = sep
        patch[1:-1, 1:-1][masks[i]] = colors[i]
        c += n+3
    r = b+2
    c = int(rng.integers(1, width-field.shape[1]-2))
    patch = grid[r:r+field.shape[0]+2, c:c+field.shape[1]+2]
    patch[_rectangle(patch.shape)] = sep
    patch[1:-1, 1:-1] = field
    return grid


def _pair(rng, roles, split, index, regime):
    bg, sep, query = roles
    witness = split == 'train' and index < 2
    if witness:
        classes = [1, 2] if index == 0 else [1, 3]
    else:
        pool = np.arange(1, 4) if split == 'train' and regime == 2 else np.arange(1, 6)
        count = min(len(pool), int(rng.choice([1, 2, 2, 3, 4])))
        classes = [int(v) for v in rng.choice(pool, count, replace=False)]
    if split == 'test' and regime == 2:
        classes = [int(rng.choice([4, 5])), int(rng.choice([1, 2]))]
    if split == 'test' and regime == 3 and 1 not in classes:
        classes[0] = 1
    n = (2 if regime in (0, 1) else int(rng.choice([2, 3])))
    if split == 'test' and regime == 0:
        n = 3
    if split == 'test' and regime == 4:
        n, classes = 4, [int(rng.choice([1, 2]))]
    colors = [int(x) for x in rng.choice([c for c in range(10) if c not in roles], len(classes), replace=False)]
    prototypes = [_prototype(rng, k) for k in classes]
    if sum(p.shape[1] for p in prototypes)+len(classes)+1 > 30:
        prototypes = [_compact(k) for k in classes]
    max_h = min(30//n, 30-12-max(p.shape[0] for p in prototypes)-n)
    if witness:
        h = int(rng.integers(10, max_h+1))
        w = int(rng.integers(10, 30//n+1))
    else:
        h = int(rng.integers(min(7, max_h), max_h+1))
        w = int(rng.integers(7, 30//n+1))
    if split == 'test' and regime == 1:
        h, w = max_h, 15
    masks = _motifs(rng, n, len(classes), witness)
    glyph_order = [int(i) for i in rng.permutation(len(classes))]
    tile_order = glyph_order[::-1] if witness else [int(i) for i in rng.permutation(len(classes))]
    if witness:
        field, free, records = _pack_witness(rng, (h,w), bg, query, index)
        target = 3
    else:
        field = np.full((h,w), bg, dtype=np.int8)
        free = np.ones((h,w), dtype=bool)
        records = []
        # One represented positive class is mandatory. Optional objects may be
        # omitted when they do not fit; topology does not depend on population.
        first = 1 if split == 'test' and regime == 3 else int(rng.choice(classes))
        if regime == 1 or (split == 'train' and regime == 3):
            mask = _dihedral(_compact(first), int(rng.integers(8)))
        elif split == 'test' and regime == 3:
            mask = np.ones((5,6), dtype=bool)
            for r,c in ((1,1),(1,2),(1,3),(2,1),(2,2),(3,1),(3,2)):
                mask[r,c] = False
            mask = _dihedral(mask, int(rng.integers(8)))
        else:
            mask = _object(rng, first)
        locations = _positions(free, mask.shape)
        if not locations:
            mask = _compact(first)
            locations = _positions(free, mask.shape)
        records.append(_put(field, free, mask, locations[int(rng.integers(len(locations)))], query))
        target = int(rng.integers(2, 9)) if n == 2 else int(rng.integers(2, 5))
        if split == 'train' and regime == 1:
            target = int(rng.integers(2,4))
        if split == 'test' and regime == 1:
            target = 9
        for _ in range(target-1):
            k = int(rng.choice([0,1,1,2,2,3,4,5]))
            if regime == 1 or (split == 'train' and regime == 3):
                if regime == 1 and split == 'test':
                    k = int(rng.choice([0,1,1,2]))
                mask = _dihedral(_compact(k), int(rng.integers(8)))
            else:
                mask = _object(rng, k, rich=rng.random() < .45)
            locations = _positions(free, mask.shape)
            if not locations:
                mask = _compact(k)
                locations = _positions(free, mask.shape)
            if locations:
                records.append(_put(field, free, mask, locations[int(rng.integers(len(locations)))], query))
        if split == 'test' and regime == 1:
            while len(records) < 4:
                mask = _compact(0)
                locations = _positions(free, mask.shape)
                if not locations:
                    # Guaranteed four-object scene on an at-least-11 by 15 field.
                    field = np.full((h,w), bg, dtype=np.int8)
                    free = np.ones((h,w), dtype=bool)
                    forms = [_compact(first), _compact(1), _compact(0), _compact(0)]
                    records = [_put(field, free, m, q, query) for m,q in
                               zip(forms, ((0,0),(0,8),(4,8),(7,8)))]
                    break
                records.append(_put(field, free, mask, locations[int(rng.integers(len(locations)))], query))
    inp = _compose_input(rng, prototypes, colors, masks, glyph_order, tile_order, field, bg, sep)
    # Ground truth comes from construction records and Kronecker substitution,
    # not from the input parser or solve(). The validator reconstructs it anew.
    out = np.full((h*n,w*n), bg, dtype=np.int8)
    by_class = dict(zip(classes, zip(colors, masks)))
    for mask, (r,c) in records:
        holes = _holes(mask)
        if len(holes) not in by_class:
            continue
        color, tile = by_class[len(holes)]
        placement = np.zeros((h,w), dtype=bool)
        for hole in holes:
            for x,y in hole:
                placement[r+x,c+y] = True
        out[np.kron(placement, tile)] = color
    return {'input': inp, 'output': out}


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train',num_train,2,10), ('num_test',num_test,1,4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer between {lo} and {hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(5))
    # Distinct observable role triples guarantee distinct inputs without retries.
    triples = [(a,b,c) for a in range(10) for b in range(10) for c in range(10)
               if len({a,b,c}) == 3]
    choices = rng.permutation(len(triples))[:num_train+num_test]
    puzzle = {'train': [], 'test': []}
    ordinal = 0
    for split, count in (('train',num_train), ('test',num_test)):
        for index in range(count):
            puzzle[split].append(_pair(rng, triples[int(choices[ordinal])], split, index, regime))
            ordinal += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    """Schema, visible semantics, pixel conservation, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    training = []
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input','output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(inp)
            assert sig not in seen, "Duplicate input"
            seen.add(sig)
            p = _parse(inp)
            n, bg = p['n'], p['bg']
            assert out.shape == tuple(n*d for d in p['field'].shape)
            assert np.array_equal(out, _render(p)), "Wrong cavity/stamp transformation"
            assert set(map(int,np.unique(out))) <= {bg, *p['mapping'].values()}
            # Each accepted cavity cell creates exactly one complete stamp.
            amounts = {color: 0 for color in p['tiles']}
            for obj in p['objects']:
                color = p['mapping'].get(len(obj['holes']))
                if color is not None:
                    amounts[color] += sum(map(len,obj['holes'])) * int(p['tiles'][color].sum())
            for color, expected in amounts.items():
                assert int((out == color).sum()) == expected
            assert not np.array_equal(inp, out)
            if split == 'train':
                training.append((p,out))
    assert any(len({len(o['holes']) for o in p['objects']
                    if len(o['holes']) in p['mapping']}) >= 2 for p,_ in training)
    assert any(any(o['holes'] and len(o['holes']) not in p['mapping']
                   for o in p['objects']) for p,_ in training), "Missing positive unlisted class"
    assert any(any(len(hole) > 1 for o in p['objects'] for hole in o['holes'])
               for p,_ in training), "Missing cavity-area witness"
    assert any(any(a['mask'].shape == b['mask'].shape
                   and len(a['holes']) != len(b['holes'])
                   and p['mapping'].get(len(a['holes'])) != p['mapping'].get(len(b['holes']))
                   for i,a in enumerate(p['objects']) for b in p['objects'][i+1:])
               for p,_ in training), "Missing equal-size, different-topology witness"
    for wrong in ('ignore_topology','solid','rank','position','area','walls','one_per_hole'):
        assert any(not np.array_equal(out, _render(p, wrong)) for p,out in training), wrong
    for transform in range(1,8):
        assert any(not np.array_equal(out, _render(p, transform=transform))
                   for p,out in training), "Stamp orientation not demonstrated"


def diversity_features(pair):
    p = _parse(pair['input'])
    return dict(input_shape=tuple(np.asarray(pair['input']).shape),
                output_shape=tuple(np.asarray(pair['output']).shape), n=p['n'],
                classes=tuple(sorted(p['mapping'])),
                roles=(p['bg'],p['sep'],p['query']), objects=len(p['objects']),
                counts=tuple(len(o['holes']) for o in p['objects']),
                cavity_areas=tuple(len(h) for o in p['objects'] for h in o['holes']),
                irregular=sum(not _crop_cells(h)[0].all() for o in p['objects'] for h in o['holes']),
                unused=sum(not any(len(o['holes']) == k for o in p['objects']) for k in p['mapping']),
                removed=sum(bool(o['holes']) and len(o['holes']) not in p['mapping'] for o in p['objects']),
                edge=sum(any(r in (0,p['field'].shape[0]-1) or c in (0,p['field'].shape[1]-1)
                             for r,c in o['cells']) for o in p['objects']),
                motifs=tuple(m.tobytes().hex() for m in p['tiles'].values()),
                glyph_order=tuple(e[1] for e in p['keys']), tile_order=tuple(p['tile_order']))


def self_test():
    """Bounded 42-seed suite and hand-specified topological/expansion checks."""
    diagonal = np.array([[0,1,1],[1,0,1],[1,1,1]], dtype=bool)
    assert _holes(diagonal) == [[(1,1)]]
    diagonal[0,1] = False
    assert not _holes(diagonal)
    assert not _holes(np.array([[1,0,1],[1,0,1],[1,1,1]], dtype=bool))
    # A small manually specified command scene, independent of construction.
    p = dict(bg=0,n=2,field=np.array([[5,5,5,5],[5,0,0,5],[5,5,5,5]]),
             mapping={1:7},tiles={7:np.array([[1,0],[1,1]],dtype=bool)},
             objects=[dict(holes=[[(1,1),(1,2)]], cells=[])])
    expected = np.zeros((6,8),dtype=np.int8)
    expected[2,2:6] = [7,0,7,0]
    expected[3,2:6] = 7
    assert np.array_equal(_render(p),expected)
    rows = ('00000000','00777000','00707000','00777000','00000000',
            '66666666','00000000','06666000','06706000','06776000',
            '06666000','00000000','66666666','00000000','06666660',
            '06555560','06500560','06555560','06666660','00000000')
    hand_input = np.array([[int(c) for c in row] for row in rows],dtype=np.int8)
    assert np.array_equal(solve(hand_input),expected)
    # Opening this boundary-touching ring at the top makes an open U. Its
    # surrounding registration frame must not act as an additional wall.
    open_scene = hand_input.copy()
    open_scene[15,2:6] = [5,0,0,5]
    assert np.array_equal(solve(open_scene),np.zeros((6,8),dtype=np.int8))
    wrong_identity = hand_input.copy()
    wrong_identity[8:10,2:4][wrong_identity[8:10,2:4] == 7] = 3
    try:
        solve(wrong_identity)
    except ValueError:
        pass
    else:
        raise AssertionError("Unmatched stamp identity accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = []
    signatures = set()
    regimes = {i:0 for i in range(5)}
    for seed,puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(pair['input']) for split in puzzle for pair in puzzle[split]))
        for split in puzzle:
            for pair in puzzle[split]:
                before = pair['input'].copy()
                assert np.array_equal(solve(before),pair['output'])
                assert np.array_equal(before,pair['input']), "solve mutated its input"
                features.append(diversity_features(pair))
        # Audit sampling promises separately from the invariant solver rule.
        regime = int(np.random.default_rng(seed).integers(5))
        regimes[regime] += 1
        train = [diversity_features(pair) for pair in puzzle['train']]
        test = [diversity_features(pair) for pair in puzzle['test']]
        if regime == 0:
            assert {f['n'] for f in train} == {2} and {f['n'] for f in test} == {3}
        elif regime == 1:
            assert max(f['objects'] for f in train) <= 3 and min(f['objects'] for f in test) >= 4
        elif regime == 2:
            assert all(set(f['classes']) <= {1,2,3} for f in train)
            assert all(set(f['classes']) & {4,5} for f in test)
        elif regime == 3:
            assert max(a for f in train for a in f['cavity_areas']) <= 3
            assert all(1 in f['classes'] and 7 in f['cavity_areas'] for f in test)
        else:
            assert {f['n'] for f in train} <= {2,3} and {f['n'] for f in test} == {4}
    assert all(regimes.values()), "Missing shift regime in 42-seed suite"
    assert len(signatures) == 42
    assert {f['n'] for f in features} == {2,3,4}
    assert {len(f['classes']) for f in features} == {1,2,3,4}
    assert len({f['input_shape'] for f in features}) >= 25
    assert len({f['output_shape'] for f in features}) >= 25
    assert len({f['counts'] for f in features}) >= 45
    assert len({f['roles'] for f in features}) >= 100
    assert max(f['objects'] for f in features) >= 6
    assert max(a for f in features for a in f['cavity_areas']) >= 6
    assert sum(f['edge'] > 0 for f in features) >= 30
    assert sum(f['irregular'] > 0 for f in features) >= 80
    assert sum(f['unused'] > 0 for f in features) >= 10
    assert any(f['removed'] == 0 for f in features)
    assert len({m for f in features for m in f['motifs']}) >= 70
    # Determinism, every legal explicit count, and invalid argument types.
    assert all(np.array_equal(x[k],y[k]) for s in ('train','test')
               for x,y in zip(puzzles[0][s],generate_puzzle(0)[s]) for k in ('input','output'))
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            validate_generated_puzzle(generate_puzzle(42+a*10+b,a,b))
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(2.5,1),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Pixel corruption must not pass validation.
    bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError,ValueError):
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    # A finite descriptive report, without seed-specific solving behavior.
    return dict(seeds=42,pairs=len(features),regimes=regimes,scales=sorted({f['n'] for f in features}),
                input_shapes=len({f['input_shape'] for f in features}),
                output_shapes=len({f['output_shape'] for f in features}),
                class_sets=len({f['classes'] for f in features}),
                populations=sorted({f['objects'] for f in features}),
                max_cavity_area=max(a for f in features for a in f['cavity_areas']),
                motif_bitmaps=len({m for f in features for m in f['motifs']}))


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
