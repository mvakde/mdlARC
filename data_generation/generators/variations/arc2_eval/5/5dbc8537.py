"""Translation-only rectangle packing with input-derived uniqueness checks."""

import hashlib
import numpy as np

_SOURCE_DIGESTS = {'e32b24553544313ce532cb4103a025149483bd9b913f597db52a486530a5e8fb', '48612e78f5318d2e7e1db2ed6e15db30766031a831b858284f8b6031880bb146', '2afbb6eebe69fef07dc7945d6e5c6de07a96fae74bd2d8ec4067ed7d98733ec2', 'a60b66091981b768a59399597206e1a8730d21b3cffdbb5329a25a5bde31ab22'}


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    selector = int(seed)

    if num_train is None and num_test is None:
        num_test = 1 + selector % 4
        low_train = num_test + 1
        span = 10 - low_train + 1
        num_train = low_train + (selector // 4) % span
    elif num_train is None:
        low_train = num_test + 1
        span = 10 - low_train + 1
        num_train = low_train + selector % span
    elif num_test is None:
        high_test = min(4, num_train - 1)
        num_test = 1 + selector % high_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _array_key(array):
    return (
        tuple(int(x) for x in array.shape),
        tuple(int(x) for x in array.ravel()),
    )


def _components(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []

    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or seen[start_y, start_x]:
                continue

            seen[start_y, start_x] = True
            stack = [(start_y, start_x)]
            component = []

            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))

            result.append(component)

    return result


def _nonrectangular_component_count(mask):
    count = 0
    for component in _components(mask):
        ys = [cell[0] for cell in component]
        xs = [cell[1] for cell in component]
        bounding_area = (
            (max(ys) - min(ys) + 1)
            * (max(xs) - min(xs) + 1)
        )
        if bounding_area != len(component):
            count += 1
    return count


def _extract_rectangular_tiles(tray, background):
    foreground = tray != background
    tiles = []

    for component in _components(foreground):
        ys = [cell[0] for cell in component]
        xs = [cell[1] for cell in component]
        y0, y1 = min(ys), max(ys) + 1
        x0, x1 = min(xs), max(xs) + 1
        tile = tray[y0:y1, x0:x1]

        if not np.all(tile != background):
            raise ValueError("A tray component is not a filled rectangle")
        if tile.size != len(component):
            raise ValueError("A tray component has holes")

        tiles.append(tile.copy())

    if not tiles:
        raise ValueError("The tray contains no source tiles")
    return tiles


def _solve_renderings(vacancy_mask, tiles, max_nodes=250000):
    """
    Enumerate distinct rendered exact-cover completions.

    Identical tile matrices are grouped with a multiplicity, so exchanging
    visually identical copies does not create a second solution.
    """
    coords = np.argwhere(vacancy_mask)
    cell_count = len(coords)
    if cell_count == 0:
        return {}, [], True

    index_grid = np.full(vacancy_mask.shape, -1, dtype=np.int32)
    for index, (y, x) in enumerate(coords):
        index_grid[int(y), int(x)] = index

    groups = []
    group_lookup = {}
    for tile in tiles:
        key = (
            tuple(int(x) for x in tile.shape),
            tuple(int(x) for x in tile.ravel()),
        )
        if key in group_lookup:
            groups[group_lookup[key]]["count"] += 1
        else:
            group_lookup[key] = len(groups)
            groups.append({"pattern": tile.copy(), "count": 1})

    placements = []
    placements_by_cell = [[] for _ in range(cell_count)]
    mask_height, mask_width = vacancy_mask.shape

    for group_index, group in enumerate(groups):
        pattern = group["pattern"]
        tile_height, tile_width = pattern.shape

        for y in range(mask_height - tile_height + 1):
            for x in range(mask_width - tile_width + 1):
                region = vacancy_mask[
                    y:y + tile_height,
                    x:x + tile_width
                ]
                if not np.all(region):
                    continue

                bit_mask = 0
                assignments = []
                for dy in range(tile_height):
                    for dx in range(tile_width):
                        cell_index = int(index_grid[y + dy, x + dx])
                        bit_mask |= 1 << cell_index
                        assignments.append(
                            (cell_index, int(pattern[dy, dx]))
                        )

                placement_index = len(placements)
                placements.append(
                    (group_index, bit_mask, tuple(assignments))
                )
                for cell_index, _ in assignments:
                    placements_by_cell[cell_index].append(
                        placement_index
                    )

    full_mask = (1 << cell_count) - 1
    remaining = [group["count"] for group in groups]
    initial_counts = tuple(remaining)
    rendered = [-1] * cell_count
    outcomes = {}
    nodes = 0
    exhaustive = True

    path = []
    dead = set()
    failures = 0

    def search(covered):
        nonlocal nodes, exhaustive, failures

        if not exhaustive or len(outcomes) >= 2:
            return

        nodes += 1
        if nodes > max_nodes:
            exhaustive = False
            return

        if covered == full_mask:
            rendering = tuple(rendered)
            used_vector = tuple(
                initial_counts[i] - remaining[i]
                for i in range(len(groups))
            )
            used_total = sum(used_vector)

            if rendering not in outcomes:
                outcomes[rendering] = {
                    "min_used": used_total,
                    "max_used": used_total,
                    "used_vector": used_vector,
                    "placements": tuple(path),
                }
            else:
                record = outcomes[rendering]
                record["min_used"] = min(
                    record["min_used"], used_total
                )
                record["max_used"] = max(
                    record["max_used"], used_total
                )
            return

        state = (covered, tuple(remaining))
        if state in dead:
            return
        best_options = None

        for cell_index in range(cell_count):
            if covered & (1 << cell_index):
                continue

            options = []
            for placement_index in placements_by_cell[cell_index]:
                group_index, placement_mask, _ = placements[
                    placement_index
                ]
                if (
                    remaining[group_index] > 0
                    and not (covered & placement_mask)
                ):
                    options.append(placement_index)

            if not options:
                dead.add(state)
                failures += 1
                return
            if best_options is None or len(options) < len(best_options):
                best_options = options
                if len(best_options) == 1:
                    break

        for placement_index in best_options:
            group_index, placement_mask, assignments = placements[
                placement_index
            ]
            remaining[group_index] -= 1
            for cell_index, color in assignments:
                rendered[cell_index] = color

            path.append((group_index, tuple(assignments)))
            search(covered | placement_mask)
            path.pop()

            for cell_index, _ in assignments:
                rendered[cell_index] = -1
            remaining[group_index] += 1

            if not exhaustive or len(outcomes) >= 2:
                return

    search(0)
    for result in outcomes.values():
        result["nodes"] = nodes
        result["dead_ends"] = failures
    return outcomes, groups, exhaustive


def _panel_candidates(grid):
    """Find semantic panel cuts without an output shape or label."""
    for axis in (0, 1):
        for cut in range(1, grid.shape[axis]):
            a, b = (grid[:cut], grid[cut:]) if axis == 0 else (grid[:, :cut], grid[:, cut:])
            for stencil, tray, side in ((a, b, 'top' if axis == 0 else 'left'),
                                        (b, a, 'bottom' if axis == 0 else 'right')):
                colors = np.unique(stencil)
                if len(colors) != 2:
                    continue
                for board, vacancy in (colors, colors[::-1]):
                    if np.any(tray == board) or not np.any(tray == vacancy):
                        continue
                    try:
                        tiles = _extract_rectangular_tiles(tray, vacancy)
                    except ValueError:
                        continue
                    mask = stencil == vacancy
                    if sum(t.size for t in tiles) < int(mask.sum()):
                        continue
                    yield side, stencil, tray, int(board), int(vacancy), tiles


def _infer_input(grid):
    answers = []
    for side, stencil, tray, board, vacancy, tiles in _panel_candidates(grid):
        mask = stencil == vacancy
        outcomes, groups, exhaustive = _solve_renderings(mask, tiles, max_nodes=12000)
        # A timed-out interpretation is unresolved, never silently discarded.
        if not exhaustive:
            raise ValueError('Panel interpretation exceeded exact-cover budget')
        if not outcomes:
            continue
        if len(outcomes) != 1:
            raise ValueError('More than one rendered completion')
        rendering, record = next(iter(outcomes.items()))
        out = stencil.copy()
        out[mask] = rendering
        answers.append(dict(side=side, stencil=stencil, tray=tray, board=board,
                            vacancy=vacancy, tiles=tiles, mask=mask, output=out,
                            groups=groups, record=record))
    if len(answers) != 1:
        raise ValueError('Input must identify exactly one panel and completion')
    return answers[0]


def solve(train_pairs, test_input):
    """The same visible rectangle-transfer rule applies to every demonstration.

    This prediction API deliberately receives no test labels or generation state.
    Color roles are local, so training establishes the rule, not a color dictionary.
    """
    return _infer_input(np.array(test_input, copy=True))['output']


def _pattern(shape, palette, rng, mono=False, asymmetric=False):
    h, w = shape
    a = int(rng.choice(palette))
    tile = np.full(shape, a, dtype=np.int8)
    if mono or len(palette) == 1 or h*w == 1:
        return tile
    b = int(rng.choice([c for c in palette if c != a]))
    mode = int(rng.integers(0, 6))
    yy, xx = np.indices(shape)
    if mode == 1:
        tile[yy % 2 == 1] = b
    elif mode == 2:
        tile[xx % 2 == 1] = b
    elif mode == 3:
        tile[(yy + xx) % 2 == 1] = b
    elif mode == 4:
        tile[int(rng.integers(h)), int(rng.integers(w))] = b
    elif mode == 5:
        tile = rng.choice(palette, size=shape).astype(np.int8)
    if asymmetric:
        tile.fill(a)
        tile[0, 0] = b
        if h > 1 and w > 1 and len(palette) > 2:
            tile[-1, 0] = int(rng.choice([c for c in palette if c not in (a,b)]))
    return tile


def _grow_rectangles(rng, h, w, count, repeat):
    """Grow an irregular union by full/partial edge contacts, with optional islands."""
    mask = np.zeros((h, w), dtype=bool)
    rects = []
    max_h = int(rng.integers(3, min(8, h) + 1))
    max_w = int(rng.integers(3, min(8, w) + 1))
    connected = rng.random() < .72
    for i in range(count):
        if i == 1 and repeat:
            th, tw = rects[0][2:]
        elif rects and rng.random() < .22:
            th, tw = rects[int(rng.integers(len(rects)))][2:]
        elif rng.random() < .1:
            th, tw = 1, 1
        else:
            th = int(rng.integers(1, max_h + 1))
            tw = int(rng.integers(1, max_w + 1))
        candidates = []
        for _ in range(100):
            if not rects or (not connected and rng.random() < .25):
                y, x = int(rng.integers(h-th+1)), int(rng.integers(w-tw+1))
            else:
                ry, rx, rh, rw = rects[int(rng.integers(len(rects)))]
                side = int(rng.integers(4))
                if side < 2:
                    y = ry-th if side == 0 else ry+rh
                    x = rx + int(rng.integers(1-tw, rw))
                else:
                    x = rx-tw if side == 2 else rx+rw
                    y = ry + int(rng.integers(1-th, rh))
            if y < 0 or x < 0 or y+th > h or x+tw > w:
                continue
            if np.any(mask[y:y+th, x:x+tw]):
                continue
            candidates.append((y,x))
        if not candidates:
            return None
        y, x = candidates[int(rng.integers(len(candidates)))]
        rects.append((y,x,th,tw))
        mask[y:y+th, x:x+tw] = True
    if np.all(mask) or not np.any(mask):
        return None
    return mask, rects


def _scatter_tray(tiles, height, max_width, vacancy, rng):
    """Random individual positions with a one-cell separating moat; edges allowed."""
    margin = int(rng.choice([0, 0, 0, 1, 2]))
    minimum = max(t.shape[1] for t in tiles) + 2*margin
    if minimum > max_width or max(t.shape[0] for t in tiles)+2*margin > height:
        return None
    widths = list(range(minimum, max_width+1))
    # Large tiles first is only a packing aid; coordinates have no destination order.
    base_order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
    for _ in range(14):
        width = widths[int(rng.integers(len(widths)))]
        occupied = np.zeros((height, width), dtype=bool)
        positions = {}
        for i in base_order:
            th, tw = tiles[i].shape
            legal = []
            for y in range(margin,height-th+1-margin):
                for x in range(margin,width-tw+1-margin):
                    if not np.any(occupied[max(0,y-1):min(height,y+th+1),
                                           max(0,x-1):min(width,x+tw+1)]):
                        legal.append((y,x))
            if not legal:
                break
            y, x = legal[int(rng.integers(len(legal)))]
            positions[i] = (y,x)
            occupied[y:y+th,x:x+tw] = True
        if len(positions) == len(tiles):
            tray = np.full((height,width),vacancy,dtype=np.int8)
            for i,(y,x) in positions.items():
                th,tw = tiles[i].shape
                tray[y:y+th,x:x+tw] = tiles[i]
            return tray
    return None


def _build_pair(rng, side, roles, level, surplus, force_asym=False, force_repeat=False):
    board, vacancy = roles
    for attempt in range(160):
        # Both board dimensions and all tile aspect ratios are sampled anew.
        height = int(rng.integers(9,31))
        width = int(rng.integers(5,23))
        count = int(rng.integers(*((2,6),(5,12),(10,17))[level]))
        grown = _grow_rectangles(rng,height,width,count,force_repeat)
        if grown is None:
            continue
        mask, rects = grown
        # Geometry is tested with distinct dimension markers first. This inexpensive
        # stage rejects placement ambiguity before arranging and coloring the tray.
        dimensions = list(dict.fromkeys((h,w) for y,x,h,w in rects))
        markers = {d: np.full(d,i+10,dtype=int) for i,d in enumerate(dimensions)}
        geometric, _, done = _solve_renderings(mask,[markers[(h,w)] for y,x,h,w in rects],max_nodes=1800)
        if not done or len(geometric) != 1:
            continue
        colors = [c for c in range(10) if c not in roles]
        rng.shuffle(colors)
        palette_size = int(rng.integers(2 if force_asym else 1,9))
        palette = colors[:palette_size]
        mono = not force_asym and rng.random() < .27
        pats = {d:_pattern(d,palette,rng,mono) for d in dimensions}
        if force_asym:
            large = [d for d in dimensions if max(d) >= 3]
            if not large:
                continue
            d = large[int(rng.integers(len(large)))]
            pats[d] = _pattern(d,palette,rng,asymmetric=True)
        tiles = [pats[(h,w)] for y,x,h,w in rects]
        completed = np.full(mask.shape,board,dtype=np.int8)
        for (y,x,h,w), tile in zip(rects,tiles):
            completed[y:y+h,x:x+w] = tile
        # Vary surplus inventory independently of side and difficulty. Hard decoys
        # have legal local placements but are excluded by the complete exact cover.
        desired = int(rng.integers(1,4)) if surplus else 0
        dims = [(h,w) for h in range(1,9) for w in range(1,9)
                if (h,w) not in dimensions and h <= height and w <= 30-width]
        rng.shuffle(dims)
        want_local = rng.random() < .70
        added = 0
        for d in dims:
            if added == desired:
                break
            h,w = d
            legal = any(np.all(mask[y:y+h,x:x+w]) for y in range(height-h+1)
                        for x in range(width-w+1))
            if added == 0 and want_local and not legal:
                continue
            decoy_palette = colors if rng.random() < .5 else palette
            tile = _pattern(d,decoy_palette,rng,mono)
            outcomes, _, exhaustive = _solve_renderings(mask,tiles+[tile],max_nodes=2500)
            expected = tuple(int(v) for v in completed[mask])
            if exhaustive and len(outcomes) == 1 and expected in outcomes:
                if outcomes[expected]['max_used'] != count:
                    continue
                tiles.append(tile)
                added += 1
        if desired and added == 0:
            continue
        tray = _scatter_tray(tiles,height,30-width,vacancy,rng)
        if tray is None:
            continue
        stencil = np.full(mask.shape,board,dtype=np.int8)
        stencil[mask] = vacancy
        # Whole-pair isometries preserve translation of every tile's full matrix.
        if side == 'left':
            inp, out = np.concatenate((stencil,tray),axis=1), completed
        elif side == 'right':
            inp, out = np.concatenate((tray,stencil),axis=1), completed
        elif side == 'top':
            inp, out = np.concatenate((stencil,tray),axis=1).T, completed.T
        else:
            inp, out = np.concatenate((tray,stencil),axis=1).T, completed.T
        if rng.random() < .5:
            inp, out = (inp[::-1],out[::-1]) if side in ('left','right') else (inp[:,::-1],out[:,::-1])
        pair = {'input':inp.copy(), 'output':out.copy()}
        try:
            analysis = _infer_input(pair['input'])
        except ValueError:
            continue
        if np.array_equal(analysis['output'],out) and not _source_copy(pair):
            return pair
    # Bounded deterministic fallback: simple but still sampled geometry, guaranteed
    # inventories and palettes. No retry recursively calls the public generator.
    return _fallback_pair(rng,side,roles,surplus,force_asym,force_repeat)


def _fallback_pair(rng,side,roles,surplus,force_asym,force_repeat):
    board,vacancy = roles
    h,w = int(rng.integers(3,7)),int(rng.integers(3,7))
    palette = [c for c in range(10) if c not in roles]
    rng.shuffle(palette)
    p = _pattern((h,w),palette[:3],rng,asymmetric=force_asym)
    stencil = np.full((2*h+3,w+3),board,dtype=np.int8)
    out = stencil.copy()
    for y in (1,h+2):
        stencil[y:y+h,1:1+w] = vacancy
        out[y:y+h,1:1+w] = p
    tray = np.full((2*h+3,w+5),vacancy,dtype=np.int8)
    tray[0:h,0:w] = p
    tray[h+2:2*h+2,1:w+1] = p
    if surplus:
        tray[0, w+3] = palette[-1]
    inp = np.concatenate((stencil,tray) if side in ('left','top') else (tray,stencil),axis=1)
    if side in ('top','bottom'):
        inp,out = inp.T,out.T
    pair = {'input':inp.copy(),'output':out.copy()}
    assert np.array_equal(_infer_input(inp)['output'],out)
    return pair


def _source_copy(pair):
    key = (_array_key(pair['input']),_array_key(pair['output']))
    return hashlib.sha256(repr(key).encode()).hexdigest() in _SOURCE_DIGESTS


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    num_train,num_test = _resolve_counts(seed,num_train,num_test)
    rng = np.random.default_rng(seed)
    # Puzzle-level sampling controls the teaching regime, never hidden semantics.
    sides = list(rng.permutation(['left','right','top','bottom']))
    regime = int(rng.integers(4))  # consistent, familiar, held-out, opposite-side shift
    allow_surplus = rng.random() < .66
    difficulty = rng.choice([0,1,2],p=[.15,.55,.30])
    opposite = {'left':'right','right':'left','top':'bottom','bottom':'top'}
    puzzle = {'train':[],'test':[]}
    seen = set()
    previous_roles = None
    for i in range(num_train+num_test):
        training = i < num_train
        if regime == 0:
            side = sides[0]
        elif regime == 1:
            side = sides[i%min(3,num_train)]
        elif regime == 2:
            side = sides[i%2] if training else sides[2+int(rng.integers(2))]
        else:
            side = sides[0] if training else opposite[sides[0]]
        # Local roles vary; explicit reversals occur without any hidden color map.
        if i == 1 and rng.random() < .4:
            roles = previous_roles[::-1]
        else:
            choices = [(a,b) for a in range(10) for b in range(10)
                       if a != b and (a,b) != previous_roles]
            roles = choices[int(rng.integers(len(choices)))]
        previous_roles = roles
        level = int(rng.choice([0,1,2],p=[.16,.52,.32]))
        if rng.random() < .5:
            level = int(difficulty)
        if not training and regime in (0,1) and rng.random() < .6:
            level = 2
        surplus = allow_surplus and (i == 0 or rng.random() < .5)
        pair = _build_pair(rng,side,roles,level,surplus,
                           force_asym=i == 0,force_repeat=i == 1)
        # Recoloring cannot duplicate an earlier pair with different adjacent roles;
        # this extra loop covers possible nonadjacent repeats without unbounded retry.
        for retry in range(10):
            if _array_key(pair['input']) not in seen:
                break
            pair = _fallback_pair(rng,side,roles,surplus,i == 0,i == 1)
        assert _array_key(pair['input']) not in seen
        seen.add(_array_key(pair['input']))
        puzzle['train' if training else 'test'].append(pair)
    return puzzle


def _features(a):
    mask,tiles,groups,rec = a['mask'],a['tiles'],a['groups'],a['record']
    components = _components(mask)
    coords = [tuple(int(v) for v in p) for p in np.argwhere(mask)]
    labels = {p:i for i,comp in enumerate(components) for p in comp}
    component_tiles = [0]*len(components)
    used_shapes = set()
    used_patterns = []
    owner = np.full(mask.shape,-1,dtype=int)
    for index,(group,assignments) in enumerate(rec['placements']):
        component_tiles[labels[coords[assignments[0][0]]]] += 1
        used_shapes.add(groups[group]['pattern'].shape)
        used_patterns.append(groups[group]['pattern'])
        for cell,_ in assignments:
            owner[coords[cell]] = index
    contacts = {}
    for y,x in coords:
        for dy,dx in ((1,0),(0,1)):
            if y+dy < mask.shape[0] and x+dx < mask.shape[1]:
                i,j = int(owner[y,x]),int(owner[y+dy,x+dx])
                if j >= 0 and i != j:
                    key = (min(i,j),max(i,j),dy)
                    contacts[key] = contacts.get(key,0)+1
    degrees = [0]*len(used_patterns)
    partial_contact = False
    for (i,j,dy),length in contacts.items():
        degrees[i] += 1
        degrees[j] += 1
        partial_contact |= length < min(used_patterns[i].shape[dy],used_patterns[j].shape[dy])
    local_surplus = 0
    for g,n in zip(groups,rec['used_vector']):
        if g['count'] > n:
            h,w = g['pattern'].shape
            if any(np.all(mask[y:y+h,x:x+w]) for y in range(mask.shape[0]-h+1)
                   for x in range(mask.shape[1]-w+1)):
                local_surplus += g['count']-n
    payload = set(int(c) for t in tiles for c in t.flat)
    used_colors = set(int(c) for t in used_patterns for c in t.flat)
    edge = False
    tops, lefts = set(),set()
    for comp in _components(a['tray'] != a['vacancy']):
        ys,xs = zip(*comp)
        tops.add(min(ys)); lefts.add(min(xs))
        edge |= min(ys)==0 or min(xs)==0 or max(ys)==a['tray'].shape[0]-1 or max(xs)==a['tray'].shape[1]-1
    return dict(side=a['side'],roles=(a['board'],a['vacancy']),
                used=rec['min_used'],supplied=len(tiles),surplus=len(tiles)-rec['min_used'],
                local_surplus=local_surplus,coverage=float(mask.mean()),components=len(components),
                max_component_tiles=max(component_tiles),max_area=max(t.size for t in used_patterns),
                singleton=any(t.size==1 for t in used_patterns),
                repeats=any(n>1 for n in rec['used_vector']),
                max_multiplicity=max(rec['used_vector']),
                palette=len(payload),monochrome=all(len(np.unique(t))==1 for t in tiles),
                asymmetric=any(max(t.shape)>=3 and not np.array_equal(t,t[::-1,::-1]) for t in used_patterns),
                discard_only_colors=len(payload-used_colors),tray_edge=bool(edge),
                distinct_tray_rows=len(tops),distinct_tray_columns=len(lefts),
                board_edge=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
                nonrectangular=_nonrectangular_component_count(mask),
                dead_ends=rec['dead_ends'],nodes=rec['nodes'],
                max_degree=max(degrees),partial_contact=bool(partial_contact),
                broad_join=any(n>=3 for n in contacts.values()),
                narrow_join=any(n==1 for n in contacts.values()),
                board_larger_than_tray=mask.size>a['tray'].size,
                reverse_aspect=(a['side'] in ('left','right') and mask.shape[1]>mask.shape[0])
                    or (a['side'] in ('top','bottom') and mask.shape[0]>mask.shape[1]),
                shape=tuple(mask.shape),dimensions=tuple(sorted(used_shapes)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen = set()
    feats = []
    for p in puzzle['train']+puzzle['test']:
        assert isinstance(p,dict) and set(p)=={'input','output'}
        for g in p.values():
            assert isinstance(g,np.ndarray) and g.ndim==2
            assert all(1 <= n <= 30 for n in g.shape)
            assert np.issubdtype(g.dtype,np.integer) and np.all((0<=g)&(g<=9))
        key = _array_key(p['input'])
        assert key not in seen and not _source_copy(p)
        seen.add(key)
        try:
            a = _infer_input(p['input'])
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc
        assert np.array_equal(a['output'],p['output'])
        by_shape = {}
        used = {g['pattern'].shape for g,n in zip(a['groups'],a['record']['used_vector']) if n}
        for tile in a['tiles']:
            if tile.shape in used:
                k = _array_key(tile)
                assert tile.shape not in by_shape or by_shape[tile.shape]==k
                by_shape[tile.shape] = k
        feats.append(_features(a))
    training,testing = feats[:nt],feats[nt:]
    assert len({f['roles'] for f in feats}) >= 2
    assert any(f['asymmetric'] for f in training)
    assert any(f['repeats'] for f in training)
    assert not any(f['surplus'] for f in testing) or any(f['surplus'] for f in training)
    for pair in puzzle['test']:
        predicted = solve(puzzle['train'],pair['input'].copy())
        assert np.array_equal(predicted,pair['output'])


def _puzzle_signature(puzzle):
    return tuple((_array_key(p['input']),_array_key(p['output']))
                 for s in ('train','test') for p in puzzle[s])


def self_test() -> None:
    counts,tests,signatures = set(),set(),set()
    features=[]
    orientation_regimes=set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        counts.add(len(p['train'])); tests.add(len(p['test']))
        signatures.add(_puzzle_signature(p))
        ff=[_features(_infer_input(x['input'])) for x in p['train']+p['test']]
        features.extend(ff)
        tr={f['side'] for f in ff[:len(p['train'])]}
        te={f['side'] for f in ff[len(p['train']):]}
        orientation_regimes.add('consistent' if len(tr|te)==1 else 'held_out' if te-tr else 'familiar')
        assert len({f['roles'] for f in ff})>1
    assert len(counts)>=4 and len(tests)>=3 and len(signatures)==42
    assert {f['side'] for f in features}=={'left','right','top','bottom'}
    assert orientation_regimes=={'consistent','familiar','held_out'}
    assert sum(f['max_component_tiles']>=3 for f in features)>.55*len(features)
    assert max(f['max_component_tiles'] for f in features)>=12
    assert min(f['coverage'] for f in features)<.15 and max(f['coverage'] for f in features)>.45
    assert max(f['max_area'] for f in features)>=30
    assert max(f['max_multiplicity'] for f in features)>=4
    assert max(f['max_degree'] for f in features)>=4
    assert all(any(f[k] for f in features) for k in
               ('partial_contact','broad_join','narrow_join','reverse_aspect','dead_ends'))
    assert {f['board_larger_than_tray'] for f in features}=={False,True}
    assert any(f['components']>1 for f in features) and any(f['nonrectangular']==0 for f in features)
    assert max(f['distinct_tray_rows'] for f in features)>=8
    assert max(f['distinct_tray_columns'] for f in features)>=8
    assert all(any(f[k] for f in features) for k in ('singleton','repeats','local_surplus','monochrome','discard_only_colors','tray_edge','board_edge'))
    assert any(f['surplus']==0 for f in features) and max(f['surplus'] for f in features)>=2
    assert max(f['palette'] for f in features)==8 and min(f['palette'] for f in features)<=2
    assert any(f['side']=='right' and f['max_component_tiles']>=5 and f['repeats'] and f['local_surplus'] for f in features)
    assert len({f['shape'] for f in features})>60
    for i,(nt,ne) in enumerate([(2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)]):
        p=generate_puzzle(700+i,nt,ne)
        validate_generated_puzzle(p)
        assert nt is None or len(p['train'])==nt
        assert ne is None or len(p['test'])==ne
    assert _puzzle_signature(generate_puzzle(77,5,2))==_puzzle_signature(generate_puzzle(77,5,2))
    assert _puzzle_signature(generate_puzzle(77,5,2))!=_puzzle_signature(generate_puzzle(78,5,2))
    for nt,ne in [(1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4)]:
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            continue
        raise AssertionError('Invalid pair counts accepted')
    # Finite inventory and pixel-level ambiguity, including interchangeable copies.
    mask=np.ones((1,4),dtype=bool)
    tile=np.array([[2,3]],dtype=np.int8)
    assert not _solve_renderings(mask,[tile])[0]
    assert len(_solve_renderings(mask,[tile,tile])[0])==1
    assert len(_solve_renderings(mask,[tile,np.array([[4,5]])])[0])==2
    same_color = [np.full((1,4),2),np.full((1,2),2),np.full((1,2),2)]
    outcomes,_,exhaustive = _solve_renderings(mask,same_color)
    assert exhaustive and len(outcomes)==1
    record = next(iter(outcomes.values()))
    assert record['min_used']==1 and record['max_used']==2
    for side in ('left','right','top','bottom'):
        for surplus in (False,True):
            pair=_fallback_pair(np.random.default_rng(543),side,(0,1),surplus,True,True)
            assert np.array_equal(solve([],pair['input']),pair['output'])
