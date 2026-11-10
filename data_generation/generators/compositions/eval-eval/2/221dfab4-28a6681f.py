"""Sluice drilling: a boundary ruler perforates terrain before conserved flow.

Only standard-library modules and NumPy are used. No parent code or data is
loaded at runtime. All persistent experiments belong in scratch/; generation,
validation and self_test themselves do not write any files.
"""

import hashlib
import numpy as np


def _mask_bits(mask):
    return int.from_bytes(np.packbits(mask, bitorder='little').tobytes(), 'little')


def _bits_mask(bits, shape):
    size = shape[0] * shape[1]
    packed = np.frombuffer(bits.to_bytes((size + 7) // 8, 'little'), dtype=np.uint8)
    return np.unpackbits(packed, count=size, bitorder='little').reshape(shape).view(np.bool_)


def components(mask):
    """Four-connected components in the original row-major discovery order.

    Integer bits encode the temporary flood frontier; no trace or cache is saved.
    Boundary masks prevent horizontal shifts from wrapping to another row.
    """
    h, w = mask.shape
    remaining = _mask_bits(mask)
    left_edge = sum(1 << (r * w) for r in range(h))
    right_edge = left_edge << (w - 1)
    result = []
    while remaining:
        region = remaining & -remaining
        while True:
            expanded = (region | ((region & ~left_edge) >> 1)
                        | ((region & ~right_edge) << 1)
                        | (region >> w) | (region << w)) & remaining
            if expanded == region:
                break
            region = expanded
        result.append(_bits_mask(region, mask.shape))
        remaining ^= region
    return result


def pour(terrain, source, reverse=False, top_first=False):
    """The same unit-cell pour, using bounded row bits for occupancy queries."""
    h, w = terrain.shape
    row_mask = (1 << w) - 1
    packed = _mask_bits(terrain)
    solid = [(packed >> (r * w)) & row_mask for r in range(h)]
    water = [0] * h
    points = sorted(np.argwhere(source).tolist(),
                    key=lambda p: (p[0] if top_first else -p[0],
                                   -p[1] if reverse else p[1]))
    for r, c in points:
        bit = 1 << c
        while r > 0 and water[r] & bit:
            r -= 1
        if (solid[r] | water[r]) & bit:
            raise ValueError('blocked source')
        for _ in range(h + 1):
            if r + 1 < h and not ((solid[r + 1] | water[r + 1]) & bit):
                r += 1
                continue
            occupied = solid[r] | water[r]
            before = occupied & ((1 << c) - 1)
            after = occupied & ~((1 << (c + 1)) - 1)
            left = before.bit_length()
            right = (after & -after).bit_length() - 2 if after else w - 1
            interval = ((1 << (right + 1)) - 1) ^ ((1 << left) - 1)
            exits = 0 if r + 1 == h else interval & ~(solid[r + 1] | water[r + 1])
            if not exits:
                water[r] |= bit
                break
            lower = exits & ((1 << (c + 1)) - 1)
            upper = exits >> c
            a = lower.bit_length() - 1 if lower else None
            b = c + (upper & -upper).bit_length() - 1 if upper else None
            if a is None:
                c = b
            elif b is None:
                c = a
            elif c - a == b - c:
                c = b if reverse else a
            else:
                c = a if c - a < b - c else b
            bit = 1 << c
            r += 1
        else:
            raise ValueError('nontermination')
    return _bits_mask(sum(row << (r * w) for r, row in enumerate(water)), terrain.shape)


def capture(terrain, pools):
    """All dry-route destinations, with identical absorbing-pool semantics.

    Python integer lists avoid repeated NumPy scalar operations in this scan.
    """
    h, w = terrain.shape
    solid = terrain.tolist()
    fixed = [[0] * w for _ in range(h)]
    for i, pool in enumerate(pools):
        for r, c in np.argwhere(pool).tolist():
            fixed[r][c] = 1 << i
    labels = [[0] * w for _ in range(h)]
    unknown = 1 << len(pools)
    for r in range(h - 1, -1, -1):
        c = 0
        row_solid = solid[r]
        row_fixed = fixed[r]
        row_labels = labels[r]
        below_solid = solid[r + 1] if r + 1 < h else None
        below_labels = labels[r + 1] if r + 1 < h else None
        while c < w:
            if row_solid[c]:
                c += 1
                continue
            end = c
            while end + 1 < w and not row_solid[end + 1]:
                end += 1
            outlets = 0
            for x in range(c, end + 1):
                outlets |= row_fixed[x]
                if below_solid is not None and not below_solid[x]:
                    outlets |= below_labels[x]
            for x in range(c, end + 1):
                if row_fixed[x]:
                    row_labels[x] = row_fixed[x]
                elif below_solid is not None and not below_solid[x]:
                    row_labels[x] = below_labels[x]
                else:
                    row_labels[x] = outlets or unknown
            c = end + 1
    return np.asarray(labels, dtype=np.int64)


def drill(terrain,start,length,local=False,global_only=False):
    h,w=terrain.shape
    r=np.arange(h)[:,None]
    lane=np.zeros_like(terrain)
    lane[:,start:start+length]=True
    mask=((r%2==0)&lane&(not global_only)) | ((r%6==4)&(not local))
    return terrain & ~mask


def terrain_sample(rng,h,w):
    t=np.zeros((h,w),bool)
    # A varying bed and independent raised vessels/deflectors.
    if rng.random()<.7:
        bed=h-1
        for c in range(w):
            bed=int(np.clip(bed+rng.integers(-1,2),h-4,h))
            t[bed:,c]=True
    for _ in range(int(rng.integers(2,6))):
        width=int(rng.integers(3,min(12,w)+1))
        left=int(rng.integers(w-width+1)); right=left+width-1
        bottom=int(rng.integers(6,h))
        top=max(3,bottom-int(rng.integers(2,7)))
        floor=bottom
        for c in range(left,right+1):
            nxt=int(np.clip(floor+rng.integers(-1,2),max(top+1,bottom-2),min(h-1,bottom+1)))
            t[min(floor,nxt):max(floor,nxt)+1,c]=True
            floor=nxt
        t[top:bottom+1,left]=True
        t[max(3,top+int(rng.integers(-1,2))):floor+1,right]=True
    for _ in range(int(rng.integers(3))):
        r=int(rng.integers(3,h-2)); c=int(rng.integers(w-2))
        for x in range(c,min(w,c+int(rng.integers(2,9)))):
            rr=int(np.clip(r+rng.integers(-1,2),3,h-2))
            t[min(r,rr):max(r,rr)+1,x]=True
            r=rr
    return t


def pools_sample(rng, terrain):
    h,w=terrain.shape
    result=[]
    rows=np.arange(h)[:,None]
    for top in range(3,h):
        for p in components(~terrain & (rows>=top)):
            v=int(p.sum())
            if 3<=v<=min(125,h*w//3) and p[top].any() and np.any(p[top]&~terrain[top-1]):
                result.append(p)
    rng.shuffle(result)
    return result


def _construct(rng, require_local=False, minimum_pools=1, teaching=False, forbidden_volume=None):
    for attempt in range(180):
        h=int(rng.integers(14,31)); w=int(rng.integers(10,31))
        length=int(rng.integers(1,min(7,w-2)+1)); start=int(rng.integers(1,w-length))
        anchor=(1,start+(length-1)//2)
        before=terrain_sample(rng,h,w)
        after=drill(before,start,length)
        candidates=pools_sample(rng,after)
        if before.sum()>h*w*.38:
            continue
        for root in candidates[:18]:
            chosen=[root]
            if minimum_pools > 1:
                occupied=root.copy()
                for p in candidates:
                    touch=occupied.copy()
                    touch[1:] |= occupied[:-1]; touch[:-1] |= occupied[1:]
                    touch[:,1:] |= occupied[:,:-1]; touch[:,:-1] |= occupied[:,1:]
                    if not (p&touch).any():
                        chosen.append(p); occupied |= p
                        if len(chosen)>=3:
                            break
            if len(chosen) < minimum_pools:
                continue
            if teaching:
                evidence = _cut_evidence(before, start, length)
                if not all(evidence[k] for k in ('local_cut', 'remote_cut', 'odd_kept', 'remote_kept')):
                    continue
                if len(evidence['global_distances']) < 2:
                    continue
            labels=capture(after,chosen)
            if labels[anchor] not in [1<<i for i in range(len(chosen))]:
                continue
            source=np.zeros_like(before)
            valid=True
            for i,pool in enumerate(chosen):
                eligible=(labels==(1<<i)) & ~before
                eligible[0] = False  # The boundary bar is never overwritten by supply.
                pts=np.argwhere(eligible)
                volume=int(pool.sum())
                if len(pts)<volume:
                    valid=False; break
                score=rng.random(len(pts))
                if rng.random()<.5:
                    score += pts[:,0]*.15
                chosen_pts=pts[np.argsort(score)[:volume]]
                if eligible[anchor] and not np.any(np.all(chosen_pts==anchor,axis=1)):
                    chosen_pts[-1]=anchor
                source[chosen_pts[:,0],chosen_pts[:,1]]=True
            if not valid or (before.sum()+source.sum()+length)>h*w*.48:
                continue
            expected=np.logical_or.reduce(chosen)
            if int(source.sum()) == forbidden_volume:
                continue
            try:
                actual=pour(after,source)
                if not np.array_equal(actual,expected):
                    continue
                if not all(np.array_equal(pour(after,source,rev,top),expected) for rev in (False,True) for top in (False,True)):
                    continue
                nodrill=pour(before,source)
                if np.array_equal(nodrill,expected):
                    continue
                if np.array_equal(pour(drill(before,start,length,True),source),expected):
                    continue
                if require_local and np.array_equal(pour(drill(before,start,length,global_only=True),source),expected):
                    continue
                if np.array_equal(source.sum(axis=0), expected.sum(axis=0)):
                    continue
                if np.argwhere(expected)[:,0].sum()<=np.argwhere(source)[:,0].sum():
                    continue
                return before, after, source, expected, start, length
            except ValueError:
                continue
    return _fallback(rng, minimum_pools, forbidden_volume)


def _fallback(rng, minimum_pools, forbidden_volume=None):
    """A bounded hydraulic witness if natural proposals cannot teach the rule.

    Two independently released supplies meet in one receiving lake. The left
    supply needs the narrow pass; the right supply needs the remote pass.
    A separate retained cup demonstrates multiple catchments when requested.
    Heights, horizontal padding and placement vary; this is a rare fallback.
    """
    h = int(rng.choice([14, 20, 26]))
    w = int(rng.integers(23 if minimum_pools > 1 else 14, 31))
    offset = int(rng.integers(w - (23 if minimum_pools > 1 else 14) + 1))
    lake_width = 7 if forbidden_volume == (12 if minimum_pools > 1 else 8) else 8
    b = np.zeros((h, w), dtype=bool)
    b[3:7, offset + 2] = b[3:7, offset + 6] = True
    b[6, offset + 2:offset + 7] = True
    b[3, offset + 4:offset + 7] = True
    b[7:11, offset + 8] = b[7:11, offset + 12] = True
    b[10, offset + 8:offset + 13] = True
    b[h - 2, :offset + 3] = True
    b[h - 2, offset + 3 + lake_width:] = True
    b[h - 1] = True
    source = np.zeros_like(b)
    target = np.zeros_like(b)
    for r, c in ((1,3),(2,3),(3,3),(5,3),(7,9),(7,10),(8,9),(8,10)):
        source[r, offset + c] = True
    if lake_width == 7:
        source[5, offset + 3] = False
    target[h - 2, offset + 3:offset + 3 + lake_width] = True
    if minimum_pools > 1:
        b[7:10, offset + 15] = b[7:10, offset + 20] = True
        b[9, offset + 15:offset + 21] = True
        source[7, offset + 16:offset + 20] = True
        target[8, offset + 16:offset + 20] = True
    return b, drill(b, offset + 3, 2), source, target, offset + 3, 2


def _cut_evidence(terrain, start, length):
    rows = np.arange(terrain.shape[0])[:, None]
    lane = np.zeros_like(terrain)
    lane[:, start:start + length] = True
    special = np.broadcast_to(rows % 6 == 4, terrain.shape)
    ordinary = np.broadcast_to((rows % 6 == 0) | (rows % 6 == 2), terrain.shape)
    return {
        'local_cut': bool(np.any(terrain & lane & ordinary)),
        'remote_cut': bool(np.any(terrain & ~lane & special)),
        'odd_kept': bool(np.any(terrain & lane & (rows % 2 == 1))),
        'remote_kept': bool(np.any(terrain & ~lane & ordinary)),
        'global_distances': tuple(int(r) for r in np.flatnonzero(np.any(terrain & ~lane & special, axis=1))),
    }


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((array >= 0) & (array <= 9))):
        raise ValueError('expected an integer ARC grid with dimensions 1..30 and colors 0..9')
    return array.astype(np.int8, copy=False)


def _parse(input_grid):
    """Decode all roles from the input, including the bead below the edge bar."""
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    if counts.max() <= grid.size / 2:
        raise ValueError('background must occupy more than half of the input')
    options = []
    for rotation in range(4):
        g = np.rot90(grid, rotation)
        for color in values:
            if color == background:
                continue
            points = np.argwhere(g == color)
            if not len(points) or not np.all(points[:, 0] == 0):
                continue
            columns = np.sort(points[:, 1])
            start, length = int(columns[0]), len(columns)
            if start < 1 or start + length >= g.shape[1]:
                continue
            if not np.array_equal(columns, np.arange(start, start + length)):
                continue
            if g.shape[0] < 3:
                continue
            water = int(g[1, start + (length - 1) // 2])
            if water in (background, int(color)):
                continue
            terrain = (g != background) & (g != color) & (g != water)
            if not terrain.any():
                continue
            options.append((g.copy(), rotation, background, int(color), water, start, length))
    if len(options) != 1:
        raise ValueError('expected one edge bar with a liquid bead immediately inward of its midpoint')
    return options[0]


def _output_from_masks(g, remaining_terrain, water_mask, background, marker, water):
    output = np.full(g.shape, background, dtype=np.int8)
    output[remaining_terrain] = g[remaining_terrain]
    output[g == marker] = marker
    output[water_mask] = water
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Perforate the input's walls, then settle its visible conserved liquid.

    Rotation brings the bar to the top. Canonical rows measure inward distance;
    columns measure lateral position. No puzzle-level state or seed is used.
    """
    g, rotation, background, marker, water, start, length = _parse(input_grid)
    terrain = (g != background) & (g != marker) & (g != water)
    remaining = drill(terrain, start, length)
    # The fixed bar is a solid part of the boundary, never a water/terrain role.
    settled = pour(remaining | (g == marker), g == water)
    result = _output_from_masks(g, remaining, settled, background, marker, water)
    return np.rot90(result, -rotation).copy()


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError('require integer counts: 2..10 train, 1..4 test, train > test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    background, water = map(int, rng.choice(10, size=2, replace=False))
    base_rotation = int(rng.integers(4))
    palette = [c for c in range(10) if c not in (background, water)]
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            # Two demonstrated directions, then held-out reversed directions.
            rotation = (base_rotation + (index % 2 if split == 'train' else 2 + index % 2)) % 4
            accepted = None
            for _ in range(8):
                before, after, source, target, start, length = _construct(
                    rng, require_local=(split == 'train' and index == 0),
                    minimum_pools=(1 if split == 'train' and index == 0 else 2),
                    teaching=(split == 'train' and index == 0),
                    forbidden_volume=(int(np.count_nonzero(puzzle['train'][0]['input'] == water))
                                      if split == 'train' and index == 1 else None))
                colors = rng.permutation(palette)
                marker = int(colors[0])
                wall_colors = colors[1:1 + int(rng.integers(1, 4))]
                g = np.full(before.shape, background, dtype=np.int8)
                cells = np.argwhere(before)
                anchors = cells[rng.choice(len(cells), len(wall_colors), replace=False)]
                distances = ((cells[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
                assigned = wall_colors[np.argmin(distances, axis=1)]
                g[cells[:, 0], cells[:, 1]] = assigned
                g[source] = water
                g[0, start:start + length] = marker
                expected = _output_from_masks(g, after, target, background, marker, water)
                inp = np.rot90(g, -rotation).copy()
                out = np.rot90(expected, -rotation).copy()
                key = (inp.shape, inp.tobytes())
                if key in seen:
                    continue
                try:
                    _parse(inp)
                except ValueError:
                    # A tiny wall-color region must not mimic a second edge bar.
                    continue
                accepted = {'input': inp, 'output': out}
                seen.add(key)
                break
            if accepted is None:
                raise RuntimeError('bounded distinct-input construction exhausted')
            puzzle[split].append(accepted)
    return puzzle


def _stats(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    g, rotation, bg, marker, liquid, start, length = _parse(inp)
    o = np.rot90(out, rotation)
    terrain = (g != bg) & (g != marker) & (g != liquid)
    source, water = g == liquid, o == liquid
    remaining = np.zeros_like(terrain)
    # Direct coordinate statement of the intended drilling mask.
    for r, c in np.argwhere(terrain):
        cut = (r % 6 == 4) or (r % 6 in (0, 2) and start <= c < start + length)
        remaining[r, c] = not cut
    solids = remaining | (g == marker)
    assert np.array_equal(o[remaining], g[remaining]), 'remaining wall material changed'
    assert np.array_equal(o == marker, g == marker), 'edge bar moved'
    assert np.array_equal((o != bg) & (o != marker) & (o != liquid), remaining), 'wrong perforations'
    assert int(source.sum()) == int(water.sum()) > 0, 'liquid volume changed'
    assert not np.array_equal(source, water), 'liquid must move'
    assert np.argwhere(water)[:, 0].sum() > np.argwhere(source)[:, 0].sum(), 'liquid must move inward'
    assert np.any(source.sum(axis=0) != water.sum(axis=0)), 'liquid must spread laterally'
    pools = components(water)
    # Each output body fills the whole connected basin below its flat surface.
    rows = np.arange(len(g))[:, None]
    for pool in pools:
        top = int(np.argwhere(pool)[:, 0].min())
        closure = next(p for p in components(~solids & (rows >= top)) if np.any(p & pool))
        assert np.array_equal(closure, pool), 'partial level or leaking pool'
    labels = capture(solids, pools)
    accounted = np.zeros_like(source)
    for i, pool in enumerate(pools):
        supply = source & (labels == (1 << i))
        assert int(supply.sum()) == int(pool.sum()), 'ambiguous catchment or volume'
        accounted |= supply
    assert np.array_equal(source, accounted), 'source has competing destinations'
    for reverse in (False, True):
        for top_first in (False, True):
            assert np.array_equal(pour(solids, source, reverse, top_first), water), 'pour-order dependence'
    assert np.array_equal(solve(inp), out), 'solve mismatch'
    # These are hydrological ablations, not just comparisons of erased pixels.
    untouched = terrain | (g == marker)
    local_only = drill(terrain, start, length, local=True) | (g == marker)
    global_only = drill(terrain, start, length, global_only=True) | (g == marker)
    assert not np.array_equal(pour(untouched, source), water), 'A has no effect on the liquid'
    assert not np.array_equal(pour(local_only, source), water), 'remote pass has no effect on the liquid'
    local_matters = not np.array_equal(pour(global_only, source), water)
    evidence = _cut_evidence(terrain, start, length)
    water_top = tuple(sorted(int(np.argwhere(p)[:, 0].min()) for p in pools))
    return {'rotation': rotation, 'background': bg, 'water': liquid,
            'shape': inp.shape, 'span': length, 'start': start,
            'volume': int(source.sum()), 'pools': len(pools), 'levels': water_top,
            'source_components': len(components(source)), 'cuts': int((terrain & ~remaining).sum()),
            'terrain_components': len(components(terrain)), 'local_matters': local_matters,
            'evidence': evidence}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, roles, training, tests = set(), set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp = _grid(pair['input'])
            key = (inp.shape, inp.tobytes())
            assert key not in seen, 'duplicate input'
            seen.add(key)
            record = _stats(pair)
            roles.add((record['background'], record['water']))
            (training if split == 'train' else tests).append(record)
    assert len(roles) == 1, 'background and water roles are puzzle-wide'
    assert training[0]['pools'] == 1
    assert training[0]['local_matters'], 'training must teach the narrow cuts hydraulically'
    assert training[1]['pools'] >= 2
    assert training[0]['volume'] != training[1]['volume'], 'training must teach variable volume'
    assert all(t['pools'] >= 2 for t in tests)
    for field in ('local_cut', 'remote_cut', 'odd_kept', 'remote_kept'):
        assert any(t['evidence'][field] for t in training), 'missing drilling demonstration'
    assert len(set(d for t in training for d in t['evidence']['global_distances'])) >= 2
    train_directions = {t['rotation'] for t in training}
    assert len(train_directions) == 2
    assert all(t['rotation'] not in train_directions for t in tests), 'missing direction shift'


_HOST_SEEDS = (0,202869913,169591547,739689019,306445026,91635702,91024346,970031832,205803796,98528875,627792655,455518903,981442115,389268601,811979379,143172413,163259904,879609781,351826078,82831731,938610968,481363929,11169452,780578876,262209421,462442783,971247270,802783205,883401767,479235293,607429250,648388500,40556483,96057772,940052765,323352965,794954574,564948382,858901839,796015760,600057423,233229837)


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, count boundaries, hydrological invariants; no disk writes."""
    signatures, shapes, structures, widths, volumes = set(), set(), set(), set(), set()
    directions, backgrounds, waters, pool_counts, source_counts = set(), set(), set(), set(), set()
    cuts, levels = set(), set()
    # Directly exercise rare fallback layouts, including an excluded volume.
    for fallback_seed in range(42):
        b, a, s, target, start, length = _fallback(np.random.default_rng(fallback_seed),
                                                 1 if fallback_seed % 2 else 2,
                                                 12 if fallback_seed % 3 else None)
        grid = np.zeros(b.shape, dtype=np.int8)
        grid[b] = 3
        grid[s] = 1
        grid[0, start:start + length] = 2
        output = _output_from_masks(grid, a, target, 0, 2, 1)
        assert _stats({'input': grid, 'output': output})['local_matters']
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        if seed in _HOST_SEEDS[:3]:
            assert _digest(puzzle) == _digest(generate_puzzle(seed)), 'nondeterministic seed'
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                record = _stats(pair)
                shapes.add(record['shape'])
                directions.add(record['rotation'])
                backgrounds.add(record['background'])
                waters.add(record['water'])
                widths.add(record['span'])
                volumes.add(record['volume'])
                pool_counts.add(record['pools'])
                source_counts.add(record['source_components'])
                cuts.add(record['cuts'])
                levels.add(len(set(record['levels'])))
                # Normalize by semantic roles, keeping canonical geometry.
                g, _, bg, marker, water, _, _ = _parse(pair['input'])
                normalized = np.where(g == bg, 0, np.where(g == marker, 1, np.where(g == water, 2, 3))).astype(np.int8)
                structures.add((g.shape, normalized.tobytes()))
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(structures) >= 200
    assert directions == set(range(4))
    assert len(backgrounds) >= 8 and len(waters) >= 8
    assert len(widths) >= 5 and len(volumes) >= 25
    assert {1, 2}.issubset(pool_counts) and max(source_counts) >= 4
    assert len(cuts) >= 15 and 2 in levels
    # verify.py exhausts all 30 legal combinations; keep the owned self-test bounded.
    for a, b in ((2, 1), (3, 2), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1234 + a * 10 + b, a, b)
        validate_generated_puzzle(puzzle)
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.0,1),(4,None),(4,False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # An altered label must be rejected by actual pixel/invariant checks.
    broken = generate_puzzle(7, 2, 1)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'pairs': 210, 'distinct_puzzles': len(signatures),
            'grid_shapes': len(shapes), 'canonical_structures': len(structures),
            'bar_widths': sorted(widths), 'volumes': len(volumes),
            'directions': sorted(directions), 'backgrounds': sorted(backgrounds),
            'waters': sorted(waters), 'pool_counts': sorted(pool_counts),
            'source_component_counts': sorted(source_counts), 'erased_counts': len(cuts)}
