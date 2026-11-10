"""Rooted compass grafts: exact scaled copies, local compasses, and pruning.

Only standard library and NumPy. No parent imports, I/O, cached puzzles, or
generation state in solve. Experiments and reports are deliberately external.
"""
from collections import defaultdict, deque
import hashlib
import numpy as np


_STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SEEDS = (0,47320276,530248691,608672161,89043299,739961689,592181185,425547922,171682616,16615346,817498053,506032391,726913971,446334213,446385615,125319253,364086742,882184653,583620626,410843602,141771706,347682061,957574649,589020879,859972933,473199749,604927881,586487613,233763318,219819818,580149391,189978532,757428025,367385951,318880532,469319150,41725723,426516090,323677960,652842725,302714449,26055138)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert a.dtype.kind in "iu" and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _pose(a, t):
    return np.rot90(a[:, ::-1] if t >= 4 else a, t % 4).copy()


def _square(a, color):
    cells = np.argwhere(a == color)
    assert len(cells)
    r, c = cells.min(0).tolist()
    bottom, right = (cells.max(0) + 1).tolist()
    side = bottom - r
    assert side == right - c and len(cells) == side * side
    return int(r), int(c), int(side)


def _reach(mask, root):
    """Return the four-connected occupied component seeded by a root square."""
    r, c, side = root
    assert np.all(mask[r:r+side, c:c+side])
    seen = np.zeros(mask.shape, dtype=bool)
    seen[r:r+side, c:c+side] = True
    stack = deque((y, x) for y in range(r, r+side) for x in range(c, c+side))
    h, w = mask.shape
    while stack:
        y, x = stack.popleft()
        for dy, dx in _STEPS:
            yy, xx = y+dy, x+dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                seen[yy, xx] = True
                stack.append((yy, xx))
    return seen


def _connected(mask):
    p = np.argwhere(mask)
    return bool(len(p)) and np.array_equal(_reach(mask, (*map(int, p[0]), 1)), mask)


def _labels(a, bg, bands):
    core = np.argwhere((a != bg) & ~np.isin(a, list(bands)))
    assert len(core)
    lo, hi = core.min(0), core.max(0)
    labels = {}
    for color in sorted(bands):
        cells = np.argwhere(a == color)
        if not len(cells):
            continue
        facing = [d for d, yes in enumerate((np.all(cells[:, 0] < lo[0]),
                  np.all(cells[:, 1] > hi[1]), np.all(cells[:, 0] > hi[0]),
                  np.all(cells[:, 1] < lo[1]))) if yes]
        assert len(facing) == 1, "compass patch must project beyond one core side"
        labels[color] = facing[0]
    assert len(labels) == 2
    ds = list(labels.values())
    assert (ds[0] - ds[1]) % 4 in (1, 3), "labels must name adjacent sides"
    return labels


def _rooms(a, bg):
    """Find uniform four-colored rectangular outlines from run geometry."""
    runs = defaultdict(list)
    h, w = a.shape
    for r in range(h):
        start = 0
        for end in range(1, w+1):
            if end < w and a[r, end] == a[r, start]:
                continue
            if end-start >= 4 and start > 0 and end < w:
                corner, edge = int(a[r, start-1]), int(a[r, start])
                if corner != bg and edge not in (bg, corner) and a[r, end] == corner:
                    runs[start-1, end, corner].append((r, edge))
            start = end
    found = []
    for (left, right, stone), rows in sorted(runs.items()):
        for i, (top, north) in enumerate(rows):
            for bottom, south in rows[i+1:]:
                if bottom-top < 5 or north == south:
                    continue
                west, east = int(a[top+1, left]), int(a[top+1, right])
                colors = (north, east, south, west)
                if len(set(colors)) != 4 or bg in colors or stone in colors:
                    continue
                if (np.all(a[top+1:bottom, left] == west) and
                        np.all(a[top+1:bottom, right] == east)):
                    found.append((top, bottom+1, left, right+1, stone, colors))
    assert 1 <= len(found) <= 3, "expected one to three recognizable habitats"
    occupied = np.zeros(a.shape, bool)
    for r0, r1, c0, c1, _, _ in found:
        assert not occupied[r0:r1, c0:c1].any(), "habitats overlap"
        occupied[r0:r1, c0:c1] = True
    return found, occupied


def _decode(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert np.sum(counts == counts[bg]) == 1
    rooms, occupied = _rooms(a, bg)
    stones = {room[4] for room in rooms}
    palettes = {frozenset(room[5]) for room in rooms}
    assert len(stones) == len(palettes) == 1
    stone, bands = next(iter(stones)), next(iter(palettes))
    interiors = [a[r0+1:r1-1, c0+1:c1-1] for r0, r1, c0, c1, _, _ in rooms]
    roots = set.intersection(*(set(map(int, np.unique(p))) - {bg, stone} for p in interiors))
    assert len(roots) == 1
    root_color = next(iter(roots))
    assert root_color not in bands
    anchors = []
    for panel in interiors:
        assert set(map(int, np.unique(panel))) <= {bg, stone, root_color}
        anchors.append(_square(panel, root_color))
    exterior = (a != bg) & ~occupied
    assert _connected(exterior), "expected one outside donor"
    coords = np.argwhere(exterior)
    low, high = coords.min(0), coords.max(0)+1
    donor = a[low[0]:high[0], low[1]:high[1]].copy()
    # A donor's bounding box contains no habitat or unrelated component.
    assert not occupied[low[0]:high[0], low[1]:high[1]].any()
    assert stone not in donor
    root = _square(donor, root_color)
    labels = _labels(donor, bg, bands)
    payload = set(map(int, np.unique(donor))) - {bg, root_color} - bands
    assert 1 <= len(payload) <= 3 and _connected(donor != bg)
    return dict(bg=bg, stone=stone, bands=bands, root_color=root_color, rooms=rooms,
                interiors=interiors, anchors=anchors, donor=donor, root=root,
                labels=labels, exterior=exterior, occupied=occupied)


def _scaled(a, root, side, bg):
    """Exact rational raster, padded on the registered square's lattice."""
    r, c, old = root
    top, left = (-r)//old*old, (-c)//old*old
    bottom = (a.shape[0]-r+old-1)//old*old
    right = (a.shape[1]-c+old-1)//old*old
    raw = np.full((bottom-top, right-left), bg, dtype=np.int8)
    raw[-r-top:a.shape[0]-r-top, -c-left:a.shape[1]-c-left] = a
    fine = np.repeat(np.repeat(raw, side, 0), side, 1)
    blocks = fine.reshape(fine.shape[0]//old, old, fine.shape[1]//old, old)
    lo, hi = blocks.min((1, 3)), blocks.max((1, 3))
    assert np.array_equal(lo, hi), "requested scale straddles a color boundary"
    return lo, (-top*side//old, -left*side//old)


def _crop(raster, offset, shape, root, bg):
    r, c, _ = root
    top, left = r-offset[0], c-offset[1]
    panel = np.full(shape, bg, dtype=np.int8)
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(shape[0], top+raster.shape[0]), min(shape[1], left+raster.shape[1])
    if y0 < y1 and x0 < x1:
        panel[y0:y1, x0:x1] = raster[y0-top:y1-top, x0-left:x1-left]
    return panel


def _predict(a, compass=True, scaling=True, pruning=True):
    data = _decode(a)
    bg, donor, root_color = data['bg'], data['donor'], data['root_color']
    out, records = a.copy(), []
    for room, panel, anchor in zip(data['rooms'], data['interiors'], data['anchors']):
        r0, r1, c0, c1, stone, colors = room
        options = []
        for t in range(8) if compass else (0,):
            posed = _pose(donor, t)
            if compass and any(colors[d] != color for color, d in _labels(posed, bg, data['bands']).items()):
                continue
            source_root = _square(posed, root_color)
            side = anchor[2] if scaling else source_root[2]
            raster, offset = _scaled(posed, source_root, side, bg)
            raw = _crop(raster, offset, panel.shape, anchor, bg)
            mask = (raw != bg) & (panel != stone)
            # Counterfactual no-scale mode need not cover an enlarged seed;
            # seed cells remain intact as they do in the actual output.
            if scaling:
                reachable = _reach(mask, anchor)
            else:
                mask[anchor[0]:anchor[0]+anchor[2], anchor[1]:anchor[1]+anchor[2]] = True
                reachable = _reach(mask, anchor)
            kept = reachable if pruning else mask
            result = panel.copy()
            result[kept] = raw[kept]
            result[anchor[0]:anchor[0]+anchor[2], anchor[1]:anchor[1]+anchor[2]] = root_color
            options.append((result, dict(pose=t, side=anchor[2], source_side=source_root[2],
                clipped=int(np.count_nonzero(raster != bg)-np.count_nonzero(raw != bg)),
                cut=int(np.count_nonzero((raw != bg) & (panel == stone))),
                orphan=int(np.count_nonzero(mask & ~reachable)),
                orphan_payload=int(np.count_nonzero(mask & ~reachable & ~np.isin(raw, list(data['bands'])+[root_color]))),
                visible=int(np.count_nonzero(reachable)),
                free_connected=_connected(panel != stone),
                baffle_cells=int(np.count_nonzero(panel == stone)))))
        assert len(options) == 1, "compass must determine a unique pose"
        result, record = options[0]
        out[r0+1:r1-1, c0+1:c1-1] = result
        records.append(record)
    return out, data, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, exact transforms, scales, and surviving material from input."""
    return _predict(_grid(input_grid))[0]


def _path(start, end, rng):
    r, c = start
    result = {(r, c)}
    while (r, c) != end:
        if r != end[0] and (c == end[1] or rng.random() < .5):
            r += 1 if end[0] > r else -1
        else:
            c += 1 if end[1] > c else -1
        result.add((r, c))
    return result


def _prototype(rng, palette, compact=False):
    bg, root, body, detail, stone, *rest = palette
    bands = rest[:4]
    h, w = [int(rng.integers(3, 5 if compact else 8)) for _ in range(2)]
    side = int(rng.choice([1, 1, 2]))
    ar, ac = int(rng.integers(h-side+1)), int(rng.integers(w-side+1))
    roots = {(r, c) for r in range(ar, ar+side) for c in range(ac, ac+side)}
    core = set(roots)
    ends = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
            (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
    label_c, label_r = int(rng.integers(w)), int(rng.integers(h))
    ends += [(0, label_c), (label_r, w-1)]
    rng.shuffle(ends)
    for end in ends:
        choices = sorted(core)
        nearest = min(abs(r-end[0])+abs(c-end[1]) for r, c in choices)
        choices = [p for p in choices if abs(p[0]-end[0])+abs(p[1]-end[1]) == nearest]
        core.update(_path(choices[int(rng.integers(len(choices)))], end, rng))
    if rng.random() < .55:
        for _ in range(int(rng.integers(h*w//3+1))):
            frontier = sorted({(r+dr, c+dc) for r, c in core for dr, dc in _STEPS
                               if 0 <= r+dr < h and 0 <= c+dc < w} - core)
            if frontier:
                core.add(frontier[int(rng.integers(len(frontier)))])
    a = np.full((h+1, w+1), bg, dtype=np.int8)
    payload_colors = [body, detail] + ([rest[4]] if rng.random() < .35 else [])
    # Grow small repeated-color detail regions on a common body, rather than
    # using an unrelated random color at every pixel.
    colored = {p: body for p in core}
    for color in payload_colors[1:]:
        choices = sorted(core-roots)
        patch = {choices[int(rng.integers(len(choices)))]}
        for _ in range(int(rng.integers(1, max(2, len(core)//4)))):
            frontier = sorted({(r+dr, c+dc) for r, c in patch for dr, dc in _STEPS} & (core-roots) - patch)
            if frontier:
                patch.add(frontier[int(rng.integers(len(frontier)))])
        for p in patch:
            colored[p] = color
    for r, c in sorted(core):
        a[r+1, c] = root if (r, c) in roots else colored[r, c]
    a[0, label_c], a[label_r+1, w] = bands[0], bands[1]
    return _pose(a, int(rng.integers(8)))


def _cut_candidates(raw, anchor, bg, stone, bands, thickness=1, witness=False):
    """Short run cuts: material can disconnect while empty habitat stays connected."""
    occupied = raw != bg
    candidates = []
    h, w = raw.shape
    for transpose in (False, True):
        m = occupied.T if transpose else occupied
        for y in range(1, m.shape[0]-thickness):
            row = m[y:y+thickness].any(0)
            starts = np.flatnonzero(row & ~np.r_[False, row[:-1]])
            ends = np.flatnonzero(row & ~np.r_[row[1:], False])+1
            for x0, x1 in zip(starts.tolist(), ends.tolist()):
                if x0 < 1 or x1 >= m.shape[1] or x1-x0 > 8 or (thickness == 2 and x1-x0 < 3):
                    continue
                wall = np.zeros(raw.shape, bool)
                if transpose:
                    wall[x0:x1, y:y+thickness] = True
                else:
                    wall[y:y+thickness, x0:x1] = True
                r, c, side = anchor
                if wall[r:r+side, c:c+side].any():
                    continue
                kept = _reach(occupied & ~wall, anchor)
                orphan = occupied & ~wall & ~kept
                if kept.sum() <= side*side:
                    continue
                if witness and (orphan.sum() < 2 or not np.any(orphan & ~np.isin(raw, list(bands)))
                                or not _connected(~wall) or not _connected(occupied)):
                    continue
                candidates.append(wall)
    return candidates


def _habitat(rng, lattice, palette, unit, t, full, witness, thickness,
             obstacles=True, baffle_required=False):
    bg, root_color, _, _, stone = palette[:5]
    bands = palette[5:9]
    posed = _pose(lattice, t)
    raster = np.repeat(np.repeat(posed, unit, 0), unit, 1)
    rr, cc, side = _square(raster, root_color)
    if full:
        h, w = raster.shape[0]+int(rng.integers(1, 4)), raster.shape[1]+int(rng.integers(1, 4))
        if max(h, w) > 16:
            return None
        top, left = int(rng.integers(h-raster.shape[0]+1)), int(rng.integers(w-raster.shape[1]+1))
        anchor = (top+rr, left+cc, side)
    else:
        h, w = int(rng.integers(max(6, side+2), 15)), int(rng.integers(max(6, side+2), 15))
        anchor = (int(rng.integers(h-side+1)), int(rng.integers(w-side+1)), side)
    h, w = max(h, 4), max(w, 4)
    raw = _crop(raster, (rr, cc), (h, w), anchor, bg)
    wall = np.zeros((h, w), bool)
    if witness:
        cuts = _cut_candidates(raw, anchor, bg, stone, bands, thickness, True)
        if not cuts:
            return None
        wall |= cuts[int(rng.integers(len(cuts)))]
    elif obstacles and (baffle_required or rng.random() < .7):
        cuts = _cut_candidates(raw, anchor, bg, stone, bands, thickness)
        if cuts:
            wall |= cuts[int(rng.integers(len(cuts)))]
        elif baffle_required:
            return None
        if rng.random() < .35:
            y, x = int(rng.integers(h)), int(rng.integers(w))
            length = int(rng.integers(1, 5))
            extra = np.zeros((h, w), bool)
            if rng.random() < .5:
                extra[y:min(h, y+thickness), x:min(w, x+length)] = True
            else:
                extra[y:min(h, y+length), x:min(w, x+thickness)] = True
            r, c, s = anchor
            extra[r:r+s, c:c+s] = False
            wall |= extra
    keep = _reach((raw != bg) & ~wall, anchor)
    material = int(np.count_nonzero(raster != bg)) - side*side
    if keep.sum() - side*side < max(3, int(np.ceil(material * .18))):
        return None
    payload_mask = ~np.isin(raw, [bg, root_color] + bands)
    if np.count_nonzero(keep & payload_mask) < 2:
        return None
    labels = _labels(posed, bg, bands)
    colors = [None]*4
    for color, direction in labels.items():
        colors[direction] = color
    remaining = [color for color in bands if color not in labels]
    rng.shuffle(remaining)
    for d in range(4):
        if colors[d] is None:
            colors[d] = remaining.pop()
    inp = np.full((h+2, w+2), bg, dtype=np.int8)
    inp[0, 1:-1], inp[1:-1, -1], inp[-1, 1:-1], inp[1:-1, 0] = colors
    inp[0, 0] = inp[0, -1] = inp[-1, 0] = inp[-1, -1] = stone
    interior = inp[1:-1, 1:-1]
    interior[wall] = stone
    r, c, s = anchor
    interior[r:r+s, c:c+s] = root_color
    out = inp.copy()
    out[1:-1, 1:-1][keep] = raw[keep]
    return inp, out


def _pack(rng, boxes, h, w):
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i][0]*boxes[i][1])
    blocked = np.zeros((h, w), bool)
    origins = [None]*len(boxes)
    for i in order:
        bh, bw = boxes[i]
        if bh > h or bw > w:
            return None
        origins_i = [(r, c) for r in range(h-bh+1) for c in range(w-bw+1)]
        rng.shuffle(origins_i)
        for r, c in origins_i:
            if not blocked[r:r+bh, c:c+bw].any():
                origins[i] = (r, c)
                blocked[max(0, r-1):min(h, r+bh+1), max(0, c-1):min(w, c+bw+1)] = True
                break
        else:
            return None
    return origins


def _construct(rng, role, shift, training):
    palette = [int(x) for x in rng.permutation(10)]
    bg = palette[0]
    witness = role == 'pruning'
    lattice = _prototype(rng, palette, compact=(role == 'transfer'))
    source_unit = int(rng.integers(1, 3))
    donor = np.repeat(np.repeat(lattice, source_unit, 0), source_unit, 1)
    count = 2 if role == 'transfer' else (3 if not training and shift == 'count' else int(rng.integers(1, 3)))
    habitats = []
    for i in range(count):
        if role == 'transfer':
            unit, t, full = i+1, int(rng.integers(1, 4) if i == 0 else rng.integers(4, 8)), True
        else:
            unit = 1 if witness and i == 0 else (3 if not training and shift == 'scale' and i == 0 else int(rng.integers(1, 3)))
            t, full = int(rng.integers(8)), bool(witness and i == 0 or rng.random() < .25)
        thickness = 2 if not training and shift == 'baffle' else 1
        required = not training and shift == 'baffle' and i == 0
        habitat = _habitat(rng, lattice, palette, unit, t, full, witness and i == 0, thickness,
                           obstacles=(role != 'transfer'), baffle_required=required)
        if habitat is None:
            return None
        # The complete transfer witness has no baffles, even incidental ones.
        if role == 'transfer' and np.any(habitat[0][1:-1, 1:-1] == palette[4]):
            return None
        habitats.append(habitat)
    boxes = [donor.shape] + [p[0].shape for p in habitats]
    low = 28 if count == 3 else 22 if count == 2 else 18
    h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    origins = _pack(rng, boxes, h, w)
    if origins is None:
        return None
    inp = np.full((h, w), bg, dtype=np.int8)
    out = inp.copy()
    for (r, c), (a, b) in zip(origins, [(donor, donor)] + habitats):
        inp[r:r+a.shape[0], c:c+a.shape[1]] = a
        out[r:r+b.shape[0], c:c+b.shape[1]] = b
    try:
        predicted, _, records = _predict(inp)
        if not np.array_equal(predicted, out):
            return None
        if role == 'transfer' and any(x['clipped'] or x['orphan'] or x['baffle_cells'] for x in records):
            return None
        if witness and not any(x['orphan_payload'] and x['orphan'] >= 2 and x['free_connected'] for x in records):
            return None
    except AssertionError:
        return None
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    shift = ('scale', 'count', 'baffle')[int(rng.integers(3))]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            role = ('transfer' if index == 0 else 'pruning' if index == 1 else 'mixed') if split == 'train' else 'mixed'
            for _ in range(300):
                pair = _construct(rng, role, shift, split == 'train')
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded habitat construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    expected, data, records = _predict(inp)
    assert np.array_equal(expected, out), 'incorrect registered, scaled, compass-oriented rooted graft'
    assert not np.array_equal(inp, out)
    protected = ~data['occupied']
    assert np.array_equal(inp[protected], out[protected]), 'outside donor/background changed'
    for room, panel, anchor, record in zip(data['rooms'], data['interiors'], data['anchors'], records):
        r0, r1, c0, c1, stone, _ = room
        tile = out[r0:r1, c0:c1]
        original = inp[r0:r1, c0:c1]
        edge = np.ones(tile.shape, bool)
        edge[1:-1, 1:-1] = False
        assert np.array_equal(tile[edge], original[edge])
        assert np.array_equal(tile[1:-1, 1:-1][panel == stone], panel[panel == stone])
        assert _connected((tile[1:-1, 1:-1] != data['bg']) & (panel != stone))
        assert record['visible'] > anchor[2]**2
    return data, records


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    compass_needed = scale_needed = pruning_needed = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            data, records = _inspect(pair)
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            if split == 'train':
                training.append(records)
                compass_needed |= not np.array_equal(_predict(inp, compass=False)[0], out)
                scale_needed |= not np.array_equal(_predict(inp, scaling=False)[0], out)
                pruning_needed |= not np.array_equal(_predict(inp, pruning=False)[0], out)
    assert compass_needed and scale_needed and pruning_needed, 'training misses a mechanism'
    assert any(len(rs) == 2 and all(not r['clipped'] and not r['baffle_cells'] for r in rs)
               and any(r['side'] != r['source_side'] for r in rs)
               and any(0 < r['pose'] < 4 for r in rs) and any(r['pose'] >= 4 for r in rs)
               for rs in training), 'missing complete scaled rotation/reflection witness'
    assert any(r['orphan'] >= 2 and r['orphan_payload'] and r['free_connected']
               for rs in training for r in rs), 'missing non-spanning severed-branch witness'


def self_test():
    """Bounded 42-seed audit. Retains only scalar sets and hashes, never traces."""
    diagonal = np.array([[1,0,0],[0,1,1],[0,0,1]], dtype=bool)
    assert _reach(diagonal, (0, 0, 1)).sum() == 1
    fine_source = np.zeros((3, 4), dtype=np.int8)
    fine_source[1:3, 1:3] = 1
    fine_source[0, 1] = 2
    _scaled(fine_source, (1, 1, 2), 4, 0)
    try:
        _scaled(fine_source, (1, 1, 2), 3, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError('inexact fractional scale accepted')
    signatures, shapes, prototypes, backgrounds, poses, scales, counts = set(), set(), set(), set(), set(), set(), set()
    clipped = orphaned = fine = 0
    for seed in _SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data, records = _inspect(pair)
                inp = pair['input']
                digest.update(bytes(inp.shape) + inp.tobytes() + pair['output'].tobytes())
                shapes.add(inp.shape)
                backgrounds.add(data['bg'])
                counts.add(len(records))
                roles = {data['bg']: 0, data['root_color']: 1}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in data['donor'].ravel())
                prototypes.add((data['donor'].shape, normalized))
                fine += int(data['root'][2] > 1 and any(np.any(data['donor'][r] != data['donor'][r-1])
                             for r in range(1, data['donor'].shape[0]) if (r-data['root'][0]) % data['root'][2]))
                for record in records:
                    poses.add(record['pose'])
                    divisor = int(np.gcd(record['side'], record['source_side']))
                    scales.add((record['side']//divisor, record['source_side']//divisor))
                    clipped += record['clipped'] > 0
                    orphaned += record['orphan'] > 0
        signatures.add(digest.digest())
        if seed == 0:
            other = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], other[s]) for k in a)
            permutation = np.array([7,3,9,0,6,1,8,4,2,5], dtype=np.int8)
            for pair in puzzle['train'] + puzzle['test']:
                assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
                for t in range(8):
                    assert np.array_equal(solve(_pose(pair['input'], t)), _pose(pair['output'], t))
    assert len(signatures) == 42 and len(shapes) >= 35 and len(prototypes) >= 120
    assert backgrounds == set(range(10)) and poses == set(range(8)) and counts == {1, 2, 3}
    assert len(scales) >= 5 and clipped >= 30 and orphaned >= 42 and fine >= 15
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            generate_puzzle(9000+train*10+test, train, test)
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,True),(3.5,1),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(0, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), shapes=len(shapes), prototypes=len(prototypes),
                backgrounds=len(backgrounds), poses=sorted(poses), scales=sorted(scales),
                habitat_counts=sorted(counts), clipped_habitats=clipped, orphan_habitats=orphaned,
                fine_detail_pairs=fine)
