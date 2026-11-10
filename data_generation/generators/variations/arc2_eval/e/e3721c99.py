import numpy as np


def _as_optional_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    num_train = _as_optional_count(num_train, "num_train")
    num_test = _as_optional_count(num_test, "num_test")

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    code = int(seed)
    mix = (
        code * 6364136223846793005 + 1442695040888963407
    ) & ((1 << 64) - 1)

    if num_train is None and num_test is None:
        num_test = 1 + code % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + (
            (code // 4 + 3 * code + 1) % span
        )
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = minimum_train + mix % (11 - minimum_train)
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + mix % maximum_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    height, width = mask.shape

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue

            stack = [(row, col)]
            seen[row, col] = True
            cells = []

            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col))

                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and mask[next_row, next_col]
                        and not seen[next_row, next_col]
                    ):
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))

            result.append(cells)

    return result


def _trim_mask(mask):
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("object mask must not be empty")

    return np.asarray(
        mask[
            rows.min() : rows.max() + 1,
            cols.min() : cols.max() + 1,
        ],
        dtype=bool,
    )


def _count_holes(mask):
    mask = _trim_mask(np.asarray(mask, dtype=bool))

    # Padding supplies one guaranteed exterior background component.
    inverse = ~np.pad(
        mask,
        1,
        mode="constant",
        constant_values=False,
    )
    return len(_components(inverse)) - 1


def _is_connected(mask):
    return len(_components(mask)) == 1


def _cell_mask(cells):
    rc = np.asarray(cells, dtype=int)
    lo = rc.min(axis=0)
    hi = rc.max(axis=0) + 1
    mask = np.zeros(tuple(hi - lo), dtype=bool)
    mask[tuple((rc - lo).T)] = True
    return mask, tuple(lo)


def _holes(mask):
    padded = np.pad(mask, 1)
    return [cells for cells in _components(~padded)
            if all(0 < r < padded.shape[0]-1 and 0 < c < padded.shape[1]-1
                   for r, c in cells)]


def _grow_region(height, width, size, rng):
    cells = {(int(rng.integers(height)), int(rng.integers(width)))}
    frontier = set()
    while len(cells) < size:
        for r, c in cells:
            frontier.update((rr, cc) for rr, cc in
                            ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                            if 0 <= rr < height and 0 <= cc < width
                            and (rr, cc) not in cells)
        choices = sorted(frontier)
        chosen = choices[int(rng.integers(len(choices)))]
        frontier.remove(chosen)
        cells.add(chosen)
    result = np.zeros((height, width), dtype=bool)
    result[tuple(np.asarray(sorted(cells)).T)] = True
    # A growing cavity can curl around a foreground island. Absorb the island
    # into the cavity so that the surrounding query object remains connected.
    for island in _holes(result):
        result[tuple((np.asarray(island)-1).T)] = True
    return result


def _compact(k):
    if k == 0:
        return np.ones((2, 3), dtype=bool)
    rows = 1 if k <= 3 else 2
    cols = (k + rows - 1) // rows
    mask = np.ones((2*rows+1, 2*cols+1), dtype=bool)
    for i in range(k):
        mask[1+2*(i//cols), 1+2*(i%cols)] = False
    return mask


def _partition(height, width, k, rng):
    boxes = [(0, height, 0, width)]
    while len(boxes) < k:
        options = []
        for i, (r, b, c, e) in enumerate(boxes):
            for axis, length in ((0,b-r),(1,e-c)):
                if length >= 3:
                    options.append((i, axis, (b-r)*(e-c)))
        if not options:
            return None
        weights = np.asarray([x[2] for x in options], dtype=float)
        i, axis, _ = options[int(rng.choice(len(options), p=weights/weights.sum()))]
        r, b, c, e = boxes.pop(i)
        if axis == 0:
            mid = int(rng.integers(r+1, b-1))
            boxes.extend(((r,mid,c,e),(mid+1,b,c,e)))
        else:
            mid = int(rng.integers(c+1, e-1))
            boxes.extend(((r,b,c,mid),(r,b,mid+1,e)))
    return boxes


def _make_object(k, rng, large=False, simple=False):
    """Grow cavities in unequal compartments, then sculpt the exterior.

    Every accepted contour edit is checked against the actual topology. Padding
    means neither a canvas edge nor a diagonal gap changes the four-neighbor rule.
    """
    compact = _compact(k)
    if simple or rng.random() < .17:
        mask = compact.copy()
        if rng.random() < .4:
            mask = np.repeat(mask, 2, axis=int(rng.integers(2)))
        return np.rot90(mask, int(rng.integers(4))).copy()
    if k >= 2 and rng.random() < .25:
        # Independently sized lobes joined by a single crooked bridge. Unlike
        # a thin ring wall, this corridor contains genuine articulation cells.
        split = int(rng.integers(1,k))
        lobes = [_compact(split), _compact(k-split)]
        for i in range(2):
            if large or rng.random() < .35:
                lobes[i] = np.repeat(lobes[i],2,axis=int(rng.integers(2)))
        offsets = [int(rng.integers(4)),int(rng.integers(4))]
        gap = int(rng.integers(1,4))
        h = max(offsets[i]+lobes[i].shape[0] for i in range(2))
        left_width = lobes[0].shape[1]
        w = left_width+gap+lobes[1].shape[1]
        mask = np.zeros((h,w),dtype=bool)
        for lobe,r,c in ((lobes[0],offsets[0],0),
                         (lobes[1],offsets[1],left_width+gap)):
            mask[r:r+lobe.shape[0],c:c+lobe.shape[1]] = lobe
        r1,r2 = [offsets[i]+lobes[i].shape[0]//2 for i in range(2)]
        corridor = left_width+gap//2
        mask[r1,left_width-1:corridor+1] = True
        mask[min(r1,r2):max(r1,r2)+1,corridor] = True
        mask[r2,corridor:left_width+gap+1] = True
        assert _is_connected(mask) and _count_holes(mask) == k
        return _trim_mask(np.rot90(mask,int(rng.integers(4))))
    height = int(rng.integers(9, 15) if large else rng.integers(3, 9))
    width = int(rng.integers(11, 19) if large else rng.integers(3, 10))
    height = max(height, compact.shape[0])
    width = max(width, compact.shape[1])
    if k == 0 and rng.random() < .7:
        mask = np.zeros((height, width), dtype=bool)
        if rng.random() < .5:
            # An open, optionally thickened U; rotations include C shapes.
            thick = int(rng.integers(1, min(3, height//2, width//2)+1))
            mask[-thick:, :] = True
            mask[:, :thick] = True
            mask[:, -thick:] = True
        else:
            # Tree growth makes crooked strokes, branch points and thin bridges.
            mask[height//2, width//2] = True
            for _ in range(height*width):
                neighbors = np.zeros_like(mask, dtype=int)
                neighbors[1:] += mask[:-1]
                neighbors[:-1] += mask[1:]
                neighbors[:,1:] += mask[:,:-1]
                neighbors[:,:-1] += mask[:,1:]
                choices = np.argwhere((neighbors == 1) & ~mask)
                if not len(choices) or mask.sum() >= height*width*.35:
                    break
                mask[tuple(choices[int(rng.integers(len(choices)))])] = True
        for cavity in _holes(mask):
            mask[tuple((np.asarray(cavity)-1).T)] = True
        assert _is_connected(mask) and _count_holes(mask) == 0
        return _trim_mask(np.rot90(mask, int(rng.integers(4))))
    boxes = _partition(height-2, width-2, k, rng) if k else []
    if boxes is None:
        return compact
    mask = np.ones((height, width), dtype=bool)
    for r, b, c, e in boxes:
        h, w = b-r, e-c
        area = int(rng.integers(max(1, int(h*w*.35)), h*w+1))
        cavity = _grow_region(h, w, area, rng)
        mask[r+1:b+1,c+1:e+1][cavity] = False
    # Carve deep open bays and stepped/asymmetric contours in sizeable chunks.
    for _ in range(14 if large else 9):
        candidate = mask.copy()
        side = int(rng.integers(4))
        reach = int(rng.integers(1, max(2, min(height,width)//2)))
        breadth = int(rng.integers(1, 4))
        if side < 2:
            c = int(rng.integers(width))
            rows = slice(0, reach) if side == 0 else slice(height-reach, height)
            candidate[rows,c:min(width,c+breadth)] = False
        else:
            r = int(rng.integers(height))
            cols = slice(0, reach) if side == 2 else slice(width-reach,width)
            candidate[r:min(height,r+breadth),cols] = False
        if candidate.any() and _is_connected(candidate) and _count_holes(candidate) == k:
            mask = candidate
    assert _is_connected(mask) and _count_holes(mask) == k
    return _trim_mask(np.rot90(mask, int(rng.integers(4))))


def _glyph(k, rng):
    if rng.random() < .55:
        mask = np.ones((3, max(3, 2*k+1)), dtype=bool)
        mask[1,1:2*k:2] = False
    else:
        mask = _compact(k)
        # Miniature prototypes differ in wall thickness and cavity proportions.
        if rng.random() < .5:
            mask = np.insert(mask, int(rng.integers(mask.shape[0])),
                             mask[int(rng.integers(mask.shape[0]))], axis=0)
            if not _is_connected(mask) or _count_holes(mask) != k:
                mask = _compact(k)
    return np.rot90(mask, int(rng.integers(2))).copy()


def _panel(rng, keys, colors, bg, sep, small=False, corner=False):
    glyphs = [_glyph(k, rng) for k in keys]
    # Pack glyphs in one or several rows; each entry has its own color.
    rows = int(rng.integers(1, min(3, len(keys))+1)) if corner else 1
    groups = [list(x) for x in np.array_split(np.arange(len(keys)), rows)]
    gap = int(rng.integers(1, 3))
    margin = int(rng.integers(1, 3))
    widths = [sum(glyphs[i].shape[1] for i in group) + gap*(len(group)-1)
              for group in groups]
    heights = [max(glyphs[i].shape[0] for i in group) for group in groups]
    ph = sum(heights) + gap*(rows-1) + 2*margin
    pw = max(widths) + 2*margin
    if ph > 17 or pw > 30 or (corner and pw > 21):
        # Compact prototypes guarantee fitting even high, nonconsecutive keys.
        glyphs = [_compact(k) for k in keys]
        groups = [list(range(len(keys)))]
        if sum(g.shape[1]+1 for g in glyphs)+1 > (20 if corner else 30):
            middle = (len(keys)+1)//2
            groups = [list(range(middle)), list(range(middle,len(keys)))]
        groups = [x for x in groups if x]
        gap = margin = 1
        widths = [sum(glyphs[i].shape[1]+1 for i in g)-1 for g in groups]
        heights = [max(glyphs[i].shape[0] for i in g) for g in groups]
        ph, pw = sum(heights)+len(groups)+1, max(widths)+2
    h = int(rng.integers(max(ph+9, 13 if small else 22), 31))
    w = int(rng.integers(max(pw, 13 if small else 22), 31))
    if rng.random() < .15:
        h = w = 30
    if corner:
        w = max(w, pw+8)
        if w > 30:
            return None
    grid = np.full((h,w), bg, dtype=np.int8)
    allowed = np.ones((h,w), dtype=bool)
    panel_left = 0 if corner else int(rng.integers(w-pw+1))
    r = margin
    for group, row_height in zip(groups, heights):
        c = panel_left + margin + int(rng.integers(max(widths)-sum(glyphs[i].shape[1] for i in group)-gap*(len(group)-1)+1))
        for i in group:
            glyph = glyphs[i]
            rr = r + int(rng.integers(row_height-glyph.shape[0]+1))
            patch = grid[rr:rr+glyph.shape[0],c:c+glyph.shape[1]]
            patch[glyph] = colors[i]
            c += glyph.shape[1]+gap
        r += row_height+gap
    if corner:
        grid[:ph+1,pw] = sep
        grid[ph,:pw+1] = sep
        allowed[:ph+2,:pw+2] = False
    else:
        grid[ph,:] = sep
        allowed[:ph+2,:] = False
    return grid, allowed


def _locations(free, h, w):
    if h > free.shape[0] or w > free.shape[1]:
        return np.empty((0,2), dtype=int)
    summed = np.pad((~free).astype(int), ((1,0),(1,0))).cumsum(0).cumsum(1)
    totals = summed[h:,w:] - summed[:-h,w:] - summed[h:,:-w] + summed[:-h,:-w]
    return np.argwhere(totals == 0)


def _place(grid, out, free, mask, k, mapping, bg, query, rng):
    h,w = mask.shape
    positions = _locations(free,h,w)
    if not len(positions):
        return False
    if rng.random() < .42:
        edge = positions[(positions[:,0] == 0) | (positions[:,1] == 0)
                         | (positions[:,0]+h == grid.shape[0])
                         | (positions[:,1]+w == grid.shape[1])]
        if len(edge):
            positions = edge
    r,c = positions[int(rng.integers(len(positions)))]
    grid[r:r+h,c:c+w][mask] = query
    out[r:r+h,c:c+w][mask] = mapping.get(k,bg)
    gap = int(rng.choice([1,1,1,2,3]))
    free[max(0,r-gap):r+h+gap,max(0,c-gap):c+w+gap] = False
    return True


def _generate_pair(rng, preferences, regime, index, fallback=False):
    training = regime == 'source_train'
    mode, stable_palette, base_roles, orientation = preferences
    roles = base_roles if stable_palette else tuple(int(x) for x in rng.permutation(10)[:3])
    bg,query,sep = roles
    small = rng.random() < .25
    low = mode == 'simple' or (mode == 'source' and training)
    pool = list(range(3 if low else 7))
    size = int(rng.integers(1, min(5 if not small else 3,len(pool))+1))
    if training and index < 2:
        size = max(2,size)
    if mode == 'source' and training:
        keys = list(range(size))
    else:
        keys = [int(k) for k in rng.choice(pool, size=size, replace=False)]
        rng.shuffle(keys)
    if training and index == 0 and mode in ('rich','reverse','shared'):
        # A visible witness against interpreting entry rank as the class.
        keys = [2,0] + ([4] if not small else [])
    colors = [int(x) for x in rng.permutation([c for c in range(10) if c not in (bg,query)])[:len(keys)]]
    mapping = dict(zip(keys,colors))
    if mode in ('simple','shared'):
        side = orientation
    elif mode == 'source':
        side = int(rng.choice([0,4])) if training else int(rng.choice([1,3]))
    elif mode == 'reverse':
        side = int(rng.choice([1,3])) if training else int(rng.choice([0,2,4,5,6,7]))
    else:
        side = int(rng.integers(8))
    if fallback:
        side, small = 0, False
    result = _panel(rng,keys,colors,bg,sep,small,side >= 4)
    if result is None:
        return None
    grid, free = result
    out = grid.copy()
    absent = [k for k in range(7) if k not in mapping]
    # The first two demonstrations each contain erasure and two retained classes.
    mandatory = [keys[0]]
    if training and index < 2:
        mandatory = [keys[0],keys[1],int(rng.choice(absent))]
    elif rng.random() < .65:
        mandatory.append(int(rng.choice(keys)))
    target = int(rng.integers(3, 7 if small else 16))
    deletion_rate = float(rng.choice([0.,.15,.35,.65,.85]))
    favorite = int(rng.choice(keys))
    counts = mandatory[:]
    while len(counts) < target:
        if rng.random() < deletion_rate:
            counts.append(int(rng.choice(absent)))
        else:
            counts.append(favorite if rng.random() < .55 else int(rng.choice(keys)))
    placed = []
    for i,k in enumerate(counts):
        large = i == 0 and not small and rng.random() < (.15 if low else .55)
        mask = _make_object(k,rng,large=large,simple=mode == 'simple')
        ok = _place(grid,out,free,mask,k,mapping,bg,query,rng)
        if not ok:
            # Shrink a requested object instead of restarting an unbounded loop.
            compact = _compact(k)
            for mask in (compact,compact.T):
                if _place(grid,out,free,mask,k,mapping,bg,query,rng):
                    ok = True
                    break
        if ok:
            placed.append(k)
        elif i < len(mandatory):
            return None
    if len(placed) < 3:
        return None
    # Canonical top/TL construction is rotated only after joint packing.
    rotation = side if side < 4 else side-4
    return {'input':np.rot90(grid,rotation).copy(),
            'output':np.rot90(out,rotation).copy()}


def _separator_regions(grid):
    h,w = grid.shape
    for color in np.unique(grid):
        for cells in _components(grid == color):
            if len(cells) < 5 or len(cells) > h+w:
                continue
            mask, (r,c) = _cell_mask(cells)
            a,b = mask.shape
            straight = (a == 1 and b == w) or (b == 1 and a == h)
            corner = False
            if a > 1 and b > 1 and len(cells) == a+b-1:
                for rr in (0,a-1):
                    for cc in (0,b-1):
                        if np.all(mask[rr,:]) and np.all(mask[:,cc]):
                            corner = ((r == 0 or r+a == h) and (c == 0 or c+b == w))
                            if corner:
                                break
                    if corner:
                        break
            if not straight and not corner:
                continue
            barrier = np.zeros(grid.shape,dtype=bool)
            barrier[tuple(np.asarray(cells).T)] = True
            regions = _components(~barrier)
            if len(regions) == 2:
                yield int(color), barrier, regions, 'corner' if corner else 'straight'


def _parse_input(grid):
    candidates = []
    for sep,barrier,regions,kind in _separator_regions(grid):
        for pi in (0,1):
            panel = np.zeros(grid.shape,dtype=bool)
            panel[tuple(np.asarray(regions[pi]).T)] = True
            work = ~panel & ~barrier
            work_colors = [int(x) for x in np.unique(grid[work])]
            if len(work_colors) != 2:
                continue
            for bg,query in (work_colors,work_colors[::-1]):
                if sep in (bg,query) or np.any(grid[panel] == query):
                    continue
                mapping, entries = {}, []
                good = True
                for color in np.unique(grid[panel]):
                    color = int(color)
                    if color == bg:
                        continue
                    components = _components(panel & (grid == color))
                    if len(components) != 1:
                        good = False
                        break
                    mask, pos = _cell_mask(components[0])
                    k = _count_holes(mask)
                    if k in mapping:
                        good = False
                        break
                    mapping[k] = color
                    entries.append((pos,k,color,mask))
                if not good or not mapping:
                    continue
                objects = []
                for cells in _components(work & (grid == query)):
                    mask,pos = _cell_mask(cells)
                    cavities = _holes(mask)
                    # Every enclosed cell is actually background, including
                    # source objects with interlocking bounding boxes.
                    if any(grid[pos[0]+r-1,pos[1]+c-1] != bg
                           for cavity in cavities for r,c in cavity):
                        good = False
                        break
                    objects.append((cells,mask,pos,cavities))
                if not good or len(objects) < 2:
                    continue
                if kind == 'corner':
                    touches = ''.join(s for s,edge in [('top',panel[0]),('bottom',panel[-1]),
                                                      ('left',panel[:,0]),('right',panel[:,-1])]
                                      if edge.any())
                    side = 'corner_'+touches
                else:
                    side = ('top' if panel[0].all() else 'bottom' if panel[-1].all()
                            else 'left' if panel[:,0].all() else 'right')
                entries.sort(key=lambda x: x[0][::-1] if side in ('top','bottom') else x[0])
                candidates.append(dict(background=bg,query=query,separator=sep,
                                       mapping=mapping,entries=entries,objects=objects,
                                       legend_side=side,panel=panel,barrier=barrier))
    assert len(candidates) == 1, 'input must have one spatial/topological interpretation'
    return candidates[0]


def _predict(grid):
    parsed = _parse_input(grid)
    output = grid.copy()
    for cells,mask,pos,cavities in parsed['objects']:
        color = parsed['mapping'].get(len(cavities), parsed['background'])
        output[tuple(np.asarray(cells).T)] = color
    return output


def _analyze_pair(input_grid, output_grid, expected_regime=None):
    parsed = _parse_input(input_grid)
    assert np.array_equal(_predict(input_grid),output_grid), 'wrong topology transformation'
    return parsed


def _grid_signature(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _source_pairs():
    return [{'input':np.asarray([[int(c) for c in row] for row in a.split()],dtype=np.int8),
             'output':np.asarray([[int(c) for c in row] for row in b.split()],dtype=np.int8)}
            for a,b in _SOURCE_GRIDS]


def _fallback_pair(roles, ordinal):
    """A guaranteed finite fallback after unusually unsuccessful free packing."""
    bg,query,sep = roles
    colors = [c for c in range(10) if c not in roles][:2]
    grid = np.full((30,30),bg,dtype=np.int8)
    grid[5,:] = sep
    for k,c in enumerate((1,7)):
        glyph = _compact(k)
        grid[1:1+glyph.shape[0],c:c+glyph.shape[1]][glyph] = colors[k]
    out = grid.copy()
    r = 8+ordinal  # 0..13 makes all requested pair inputs distinct.
    for k,c in enumerate((1,11,21)):
        mask = _compact(k)
        grid[r:r+mask.shape[0],c:c+mask.shape[1]][mask] = query
        out[r:r+mask.shape[0],c:c+mask.shape[1]][mask] = colors[k] if k < 2 else bg
    return {'input':grid,'output':out}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    ntrain,ntest = _resolve_counts(seed,num_train,num_test)
    rng = np.random.default_rng(seed)
    preferences = (str(rng.choice(['source','simple','rich','reverse','shared'])),
                   bool(rng.random() < .30),tuple(int(x) for x in rng.permutation(10)[:3]),
                   int(rng.integers(8)))
    result = {'train':[],'test':[]}
    seen = set()
    for split,count in (('train',ntrain),('test',ntest)):
        regime = 'source_train' if split == 'train' else 'target_test'
        for i in range(count):
            for attempt in range(32):
                pair = _generate_pair(rng,preferences,regime,i,fallback=attempt >= 24)
                if pair is None or _grid_signature(pair['input']) in seen:
                    continue
                # Parsing uses no placement state, labels, or sampled preferences.
                try:
                    _analyze_pair(pair['input'],pair['output'])
                except AssertionError:
                    continue
                seen.add(_grid_signature(pair['input']))
                result[split].append(pair)
                break
            else:
                pair = _fallback_pair(preferences[2],len(seen))
                assert _grid_signature(pair['input']) not in seen
                seen.add(_grid_signature(pair['input']))
                result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    source = {(_grid_signature(p['input']),_grid_signature(p['output'])) for p in _source_pairs()}
    seen, witnesses = set(),set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            for grid in pair.values():
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype,np.integer) and grid.min() >= 0 and grid.max() <= 9
            inp,out = pair['input'],pair['output']
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            signature = _grid_signature(inp)
            assert signature not in seen
            seen.add(signature)
            assert (signature,_grid_signature(out)) not in source
            parsed = _analyze_pair(inp,out)
            if split == 'train':
                for cells,mask,pos,holes in parsed['objects']:
                    witnesses.add('erase' if len(holes) not in parsed['mapping'] else len(holes))
    # Palettes, sides and class subsets are visible locally; there is no hidden
    # ordering/color convention and no fixed train/test geometric restriction.
    assert 'erase' in witnesses and len(witnesses-{'erase'}) >= 2


def _puzzles_equal(a,b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k],y[k])
               for x,y in zip(a[s],b[s]) for k in ('input','output')) for s in a)


def _has_lobe_bridge(mask):
    """A cell whose removal separates two parts of at least three cells each."""
    vertices = {tuple(p) for p in np.argwhere(mask)}
    discovery, low, sizes = {},{},{}
    found = False

    def visit(v,parent):
        nonlocal found
        discovery[v] = low[v] = len(discovery)
        sizes[v] = 1
        pieces = []
        r,c = v
        for u in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if u not in vertices or u == parent:
                continue
            if u in discovery:
                low[v] = min(low[v],discovery[u])
            else:
                visit(u,v)
                sizes[v] += sizes[u]
                low[v] = min(low[v],low[u])
                if low[u] >= discovery[v]:
                    pieces.append(sizes[u])
        pieces.append(len(vertices)-1-sum(pieces))
        found = found or sum(n >= 3 for n in pieces) >= 2

    visit(min(vertices),None)
    return found


def diversity_features(pair):
    """Measurements reconstructed from returned pixels, not generation metadata."""
    grid = pair['input']
    p = _parse_input(grid)
    counts, areas, irregular, sizes = [],[],0,[]
    edge, unequal, open_bays, thin_necks, branched_holes = 0,0,0,0,0
    bridges, thick_walls, asymmetric, irregular_zero, diagonal_holes = 0,0,0,0,0
    uneven_cavities = 0
    for cells,mask,(r,c),cavities in p['objects']:
        counts.append(len(cavities))
        sizes.append(mask.shape)
        current = []
        for cavity in cavities:
            hole,_ = _cell_mask(cavity)
            current.append(len(cavity))
            irregular += int(not hole.all())
            neighbor = np.zeros(hole.shape,dtype=int)
            neighbor[1:] += hole[:-1]; neighbor[:-1] += hole[1:]
            neighbor[:,1:] += hole[:,:-1]; neighbor[:,:-1] += hole[:,1:]
            branched_holes += int(np.any(hole & (neighbor >= 3)) and not hole.all())
        areas.extend(current)
        unequal += int(len(set(current)) > 1)
        if len(cavities) >= 3:
            centers = np.asarray([np.mean(cavity,axis=0) for cavity in cavities])
            distances = np.abs(centers[:,None,:]-centers[None,:,:]).sum(axis=2)
            distances += np.eye(len(cavities))*10000
            uneven_cavities += int(np.ptp(distances.min(axis=1)) > .5)
        bridges += int(_has_lobe_bridge(mask))
        thick_walls += int(np.any(mask[:-1,:-1] & mask[1:,:-1] & mask[:-1,1:] & mask[1:,1:]))
        asymmetric += int(not np.array_equal(mask,mask[::-1]) and not np.array_equal(mask,mask[:,::-1]))
        irregular_zero += int(not cavities and not mask.all())
        edge += int(any(rr in (0,grid.shape[0]-1) or cc in (0,grid.shape[1]-1) for rr,cc in cells))
        # Exterior cells deep within the bounding box identify open notches.
        exterior = ~np.pad(mask,1)
        for cavity in cavities:
            exterior[tuple(np.asarray(cavity).T)] = False
        diagonal_holes += sum(any(exterior[r+dr,c+dc] for r,c in cavity
                                  for dr,dc in ((-1,-1),(-1,1),(1,-1),(1,1)))
                              for cavity in cavities)
        open_bays += int(mask.shape[0] >= 5 and mask.shape[1] >= 5 and exterior[3:-3,3:-3].any())
        neighbor = np.zeros(mask.shape,dtype=int)
        neighbor[1:] += mask[:-1]; neighbor[:-1] += mask[1:]
        neighbor[:,1:] += mask[:,:-1]; neighbor[:,:-1] += mask[:,1:]
        thin_necks += int(np.any(mask & (neighbor == 2)))
    mapping = p['mapping']
    order = tuple(e[1] for e in p['entries'])
    nearest = []
    for i,(cells,_,_,_) in enumerate(p['objects']):
        a = np.asarray(cells)
        nearest.append(min(int(np.abs(a[:,None,:]-np.asarray(other)[None,:,:]).sum(axis=2).min())
                           for j,(other,_,_,_) in enumerate(p['objects']) if j != i))
    return dict(shape=grid.shape,roles=(p['background'],p['query'],p['separator']),
                side=p['legend_side'],keys=tuple(sorted(mapping)),order=order,
                objects=len(counts),counts=tuple(counts),removed=sum(k not in mapping for k in counts),
                unused=sum(k not in counts for k in mapping),max_multiplicity=max(counts.count(k) for k in counts),
                cavity_areas=tuple(areas),irregular_cavities=irregular,branched_cavities=branched_holes,
                unequal_objects=unequal,edge_objects=edge,open_notches=open_bays,
                thin_wall_objects=thin_necks,object_sizes=tuple(sizes),
                lobe_bridges=bridges,thick_wall_objects=thick_walls,
                asymmetric_objects=asymmetric,irregular_zero_objects=irregular_zero,
                diagonal_only_cavities=diagonal_holes,
                uneven_cavity_spacing_objects=uneven_cavities,
                nearest_object_distances=tuple(nearest),
                separator_class=p['separator'] in mapping.values(),
                rank_witness=any(k in mapping and order.index(k) != k for k in counts),
                prototype_sizes=tuple(e[3].shape for e in p['entries']))


def self_test() -> None:
    source = _source_pairs()
    for pair in source:
        _analyze_pair(pair['input'],pair['output'])
    assert _parse_input(source[1]['input'])['mapping'] == {0:2,1:3}
    assert _parse_input(source[1]['input'])['legend_side'].startswith('corner')
    assert sorted(len(x[3]) for x in _parse_input(source[2]['input'])['objects'])[-1] == 5
    # Diagonal escape is closed under four-connectivity; opening an orthogonal
    # passage changes the class. These are independent hand-specified fixtures.
    diagonal = np.asarray([[0,1,1],[1,0,1],[1,1,1]],dtype=bool)
    assert _count_holes(diagonal) == 1
    diagonal[0,1] = False
    assert _count_holes(diagonal) == 0
    notch = np.asarray([[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]],dtype=bool)
    assert _count_holes(notch) == 0
    features, puzzles, feature_groups = [],[],[]
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        local_features = []
        for split in p:
            for pair in p[split]:
                local_features.append(diversity_features(pair))
                assert np.array_equal(_predict(pair['input'].copy()),pair['output'])
        features.extend(local_features)
        feature_groups.append(local_features)
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({f['shape'] for f in features}) >= 50
    assert any(f['shape'] == (30,30) for f in features)
    assert len({f['objects'] for f in features}) >= 8
    assert max(f['objects'] for f in features) >= 11
    assert len({f['keys'] for f in features}) >= 25
    assert len({f['side'] for f in features}) == 8
    assert max(a for f in features for a in f['cavity_areas']) >= 22
    for key in ('irregular_cavities','branched_cavities','unequal_objects','edge_objects',
                'open_notches','unused','separator_class','rank_witness','lobe_bridges',
                'thick_wall_objects','asymmetric_objects','irregular_zero_objects',
                'diagonal_only_cavities','uneven_cavity_spacing_objects'):
        assert sum(bool(f[key]) for f in features) >= 10, key
    assert any(f['removed'] == 0 for f in features)
    assert any(f['removed'] > f['objects']/2 for f in features)
    assert any(f['max_multiplicity'] >= 3 for f in features)
    assert any(6 in f['keys'] for f in features)
    assert any(len(f['keys']) == 1 for f in features)
    assert any(min(size) <= 2 for f in features for size in f['object_sizes'])
    assert any(max(size) >= 17 for f in features for size in f['object_sizes'])
    assert any(max(f['shape'])/min(f['shape']) >= 1.7 for f in features)
    assert any(f['side'].startswith('corner') and f['irregular_cavities'] and f['edge_objects']
               and f['rank_witness'] for f in features)
    assert sum(len(set(f['nearest_object_distances'])) > 1 for f in features) >= 100
    assert sum(len({f['roles'] for f in group}) > 1 for group in feature_groups) >= 15
    for key in ('shape','objects','keys','order','object_sizes','cavity_areas'):
        varied = sum(len({str(f[key]) for f in group}) > 1 for group in feature_groups)
        assert varied >= 38, ('within-puzzle diversity',key,varied)
    assert _puzzles_equal(generate_puzzle(731,7,3),generate_puzzle(731,7,3))
    assert not _puzzles_equal(generate_puzzle(731,7,3),generate_puzzle(732,7,3))
    fallback = {'train':[_fallback_pair((0,5,1),i) for i in range(10)],
                'test':[_fallback_pair((0,5,1),i) for i in range(10,14)]}
    validate_generated_puzzle(fallback)
    ambiguous = _fallback_pair((0,5,1),0)['input']
    ambiguous[1:4,1:4] = 0
    ambiguous[1:4,1:4][_compact(1)] = 2
    try:
        _parse_input(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate-topology legend accepted')
    for n,m in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        p = generate_puzzle(1001,n,m)
        validate_generated_puzzle(p)
        assert n is None or len(p['train']) == n
        assert m is None or len(p['test']) == m
    for n,m in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4),(True,1),(2.5,1)):
        try:
            generate_puzzle(0,n,m)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')
    # Validators must reject a wrong recolor and modifications outside queries.
    p = generate_puzzle(29,2,1)
    for query_cell in (True,False):
        bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
        pair = bad['test'][0]
        parsed = _parse_input(pair['input'])
        loc = np.argwhere((pair['input'] == parsed['query']) == query_cell)[0]
        pair['output'][tuple(loc)] = (int(pair['output'][tuple(loc)])+1)%10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted answer accepted')


# Exact source regressions; never used to construct generated answers.
_SOURCE_GRIDS = (('000000000000000000000000000000 044403330111110222222200000000 044403030101010202020200000000 044403330111110222222200000000 000000000000000000000000000000 111111111111111111111111111111 000000000000000000000000000000 000000000000000000000000000000 000000000055000000000000000000 000555000555000000000000000000 005505505555000000000000000000 005505505500000000005555550000 000555500000550000055000550000 000000000005550000555555500000 000000000555550005500055000000 000000000555500005555555000000 000005000000000000555000000000 000555500000000000000000000000 000500500000000000000000000000 000555500000005500000000555500 000555000000555550000055505500 000000000000505050000555500500 000000555500555555000550555500 000005550500505550000550055500 000555550500555000000055555000 005555055500000000000000000000 005505055500000005550000000000 005555555500000055555500000000 005505555000000055555500000000 000555550000000005550000000000', '000000000000000000000000000000 044403330111110222222200000000 044403030101010202020200000000 044403330111110222222200000000 000000000000000000000000000000 111111111111111111111111111111 000000000000000000000000000000 000000000000000000000000000000 000000000044000000000000000000 000333000444000000000000000000 003303304444000000000000000000 003303304400000000001111110000 000333300000440000011000110000 000000000004440000111111100000 000000000444440001100011000000 000000000444400001111111000000 000003000000000000111000000000 000333300000000000000000000000 000300300000000000000000000000 000333300000002200000000111100 000333000000222220000011101100 000000000000202020000111100100 000000000000222222000110111100 000000000000202220000110011100 000000000000222000000011111000 000000000000000000000000000000 000000000000000004440000000000 000000000000000044444400000000 000000000000000044444400000000 000000000000000004440000000000'), ('000000000100000000000000000000 022203330100000000550000000000 022203030100005555555555000000 022203330100055555555555000000 000000000100055555555555500000 000000000100005555555555500000 111111111100005555555555000000 000000000000000055555000000000 000000000000000000000000005500 000000000055500000005555555500 000000000005500000005550055500 000055000000000000055500055550 005555550000000000055500055550 005555555000000000005500055550 005555555000555500005555555500 055555555000555000000550000000 005555550000555500000000000000 000055000005555500000000000000 000000000005555500000000000000 000000000005555000005555500000 000000000000555000005505500000 000000000000555000005005500000 000055550000000000005005000000 000050055000500000005555500000 000550050005555555005555000000 000550550005050055000000000000 000055550005050055000000000000 000000500005555550000000000000 000000000000500000000000000000 000000000000000000000000000000', '000000000100000000000000000000 022203330100000000220000000000 022203030100002222222222000000 022203330100022222222222000000 000000000100022222222222200000 000000000100002222222222200000 111111111100002222222222000000 000000000000000022222000000000 000000000000000000000000003300 000000000022200000003333333300 000000000002200000003330033300 000022000000000000033300033330 002222220000000000033300033330 002222222000000000003300033330 002222222000222200003333333300 022222222000222000000330000000 002222220000222200000000000000 000022000002222200000000000000 000000000002222200000000000000 000000000002222000003333300000 000000000000222000003303300000 000000000000222000003003300000 000033330000000000003003000000 000030033000000000003333300000 000330030000000000003333000000 000330330000000000000000000000 000033330000000000000000000000 000000300000000000000000000000 000000000000000000000000000000 000000000000000000000000000000'), ('000000000550000000000000100000 005500000555000000000000108880 055555550555005555500000108080 050550050000055000500000108880 050050050000050000505550108080 055550050000055005505050108880 005550550000005555005550100000 000055500555000000000000106660 550000000555500000000000106060 550000000555500055555500106660 555000000055005550055550106060 555005550000055550550050106660 555055055550050055500050106060 550050005050050005550050106660 000055005550055005055550106060 000005555555005555000550106660 000000000055500055550500100000 005500000005500000055500104440 055555000000000000000000104040 555555500055555550000000104440 055555500555500055550000104040 000000000500500000555500104440 000000005500500005505500104040 055000055000500055005000104440 055500555000500550005000104040 055505505555555500005000104440 055005000555055000005000104040 000005000050005000055000104440 000005500050005000550000100000 000000555555555555500000100000', '000000000000000000000000100000 008800000000000000000000108880 088888880000000000000000108080 080880080000000000000000108880 080080080000000000000000108080 088880080000000000000000108880 008880880000000000000000100000 000088800000000000000000106660 000000000000000000000000106060 000000000000000066666600106660 000000000000006660066660106060 000008880000066660660060106660 000088088880060066600060106060 000080008080060006660060106660 000088008880066006066660106060 000008888888006666000660106660 000000000088800066660600100000 000000000008800000066600104440 000000000000000000000000104040 000000000044444440000000104440 000000000444400044440000104040 000000000400400000444400104440 000000004400400004404400104040 000000044000400044004000104440 000000444000400440004000104040 000004404444444400004000104440 000004000444044000004000104040 000004000040004000044000104440 000004400040004000440000100000 000000444444444444400000100000'), ('000001000000000000000000000000 044401000555555500000000000000 040401005555555500005555500000 044401005555500500555505500000 000001005555555500555555550000 077701000055550000505555555000 070701000000000000555555555000 077701000000000000555555500000 070701000000555500555555000000 077701000000500500000550005555 000001000005555550000000055555 022201000055505550000000055550 020201000055555550000000055550 022201000000555500000000000500 020201000000000000000000000000 022201000000000055000055550000 020201000000000555000555055500 022201000000000055005555055500 000001000000000000050555555500 011101000555550000050055555000 010101055005050000055005555000 011101055555555000005555555000 010101055555555000005500555000 011101055555555000005500550000 010101055555555000005555500000 011101055550055000000000000000 010101055055555000500000005550 011101005555550000555000005550 000001000005500005555500005550 000001000000000000555000000000', '000001000000000000000000000000 044401000444444400000000000000 040401004444444400007777700000 044401004444400400777707700000 000001004444444400777777770000 077701000044440000707777777000 070701000000000000777777777000 077701000000000000777777700000 070701000000777700777777000000 077701000000700700000770000000 000001000007777770000000000000 022201000077707770000000000000 020201000077777770000000000000 022201000000777700000000000000 020201000000000000000000000000 022201000000000000000022220000 020201000000000000000222022200 022201000000000000002222022200 000001000000000000020222222200 011101000111110000020022222000 010101011001010000022002222000 011101011111111000002222222000 010101011111111000002200222000 011101011111111000002200220000 010101011111111000002222200000 011101011110011000000000000000 010101011011111000000000000000 011101001111110000000000000000 000001000001100000000000000000 000001000000000000000000000000'))
