"""Ranked chamber filling: a topology signature indexes a unary length rank.

Only standard-library code and NumPy are used. The decoder receives pixels only;
construction records are never consulted by solve or validation.
"""
from collections import deque
import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a


def _components(mask):
    """Four-neighbor components, with deterministic row-major discovery."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        cells = []
        while queue:
            rr, cc = queue.popleft()
            cells.append((rr, cc))
            for y, x in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                if 0 <= y < h and 0 <= x < w and mask[y, x] and not seen[y, x]:
                    seen[y, x] = True
                    queue.append((y, x))
        result.append(np.asarray(cells, dtype=int))
    return result


def _mask_of(cells):
    lo = cells.min(axis=0)
    size = cells.max(axis=0) - lo + 1
    mask = np.zeros(tuple(size), dtype=bool)
    mask[tuple((cells-lo).T)] = True
    return mask, lo


def _holes(mask):
    """Hole coordinates relative to the unpadded object bounding box."""
    inverse = ~np.pad(mask, 1, constant_values=False)
    h, w = inverse.shape
    return [c-1 for c in _components(inverse)
            if not np.any((c[:, 0] == 0) | (c[:, 0] == h-1)
                          | (c[:, 1] == 0) | (c[:, 1] == w-1))]


def _span(cells):
    return int(np.max(cells.max(axis=0)-cells.min(axis=0)+1))


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "the canvas has a complete background border")
    keys, query_candidates = {}, []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        objects = []
        for cells in _components(grid == color):
            mask, lo = _mask_of(cells)
            holes = [h+lo for h in _holes(mask)]
            _require(all(np.all(grid[tuple(h.T)] == bg) for h in holes),
                     "cavities must contain only background")
            objects.append(dict(cells=cells, mask=mask, holes=holes, position=lo))
        solid = [o for o in objects if not o['holes']]
        hollow = [o for o in objects if o['holes']]
        if len(objects) == 2 and len(solid) == len(hollow) == 1:
            tally = solid[0]['mask']
            _require(tally.all() and min(tally.shape) == 1, "unary instruction must be straight")
            k, rank = len(hollow[0]['holes']), int(tally.sum())
            _require(1 <= k <= 5 and 1 <= rank <= k, "invalid topology/rank instruction")
            _require(k not in keys, "duplicate topology signature")
            keys[k] = dict(color=color, rank=rank, prototype=hollow[0], tally=solid[0])
        else:
            _require(not solid and len(hollow) >= 2, "query objects must all have cavities")
            query_candidates.append((color, objects))
    _require(keys and len(query_candidates) == 1, "one neutral query color and at least one instruction")
    query, objects = query_candidates[0]
    for o in objects:
        k = len(o['holes'])
        _require(1 <= k <= 5, "query cavity count outside family")
        spans = [_span(h) for h in o['holes']]
        _require(len(set(spans)) == k, "query cavity lengths must be distinct")
        o['spans'] = spans
        o['count'] = k
    _require(any(o['count'] in keys for o in objects), "at least one applicable instruction")
    return dict(grid=grid, background=bg, query=query, keys=keys, objects=objects)


def _answer(decoded, rule='length'):
    out = decoded['grid'].copy()
    bg, keys = decoded['background'], decoded['keys']
    for o in decoded['objects']:
        if o['count'] not in keys:
            out[tuple(o['cells'].T)] = bg
            continue
        key = keys[o['count']]
        holes = o['holes']
        if rule == 'all':
            selected = holes
        else:
            if rule == 'area':
                ordered = sorted(holes, key=len, reverse=True)
            elif rule in ('position', 'reverse_position'):
                ordered = sorted(holes, key=lambda h: tuple(h.min(axis=0)),
                                 reverse=rule == 'reverse_position')
            else:
                ordered = sorted(holes, key=_span, reverse=True)
            index = 0 if rule == 'longest' else len(holes)-1 if rule == 'shortest' else key['rank']-1
            if rule == 'second_shortest' and key['rank'] == 2:
                index = len(holes)-2
            selected = [ordered[index]]
        for hole in selected:
            out[tuple(hole.T)] = key['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill one ranked cavity in each listed object; erase unlisted shells."""
    return np.ascontiguousarray(_answer(_decode(input_grid)))


def _signature(grid):
    a = np.asarray(grid)
    return a.shape, tuple(int(c) for c in a.flat)


def _shortcut_survivors(decoded_train):
    return [rule for rule in ('all', 'longest', 'shortest', 'second_shortest',
                              'area', 'position', 'reverse_position')
            if all(np.array_equal(_answer(d, rule), _answer(d)) for d in decoded_train)]


def _topology_omission(d):
    """Ablation: apply the first instruction to every shell, without class filtering."""
    key = next(iter(d['keys'].values()))
    out = d['grid'].copy()
    for o in d['objects']:
        holes = sorted(o['holes'], key=_span, reverse=True)
        if len(holes) >= key['rank']:
            out[tuple(holes[key['rank']-1].T)] = key['color']
    return out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid counts")
    seen, train, test = set(), [], []
    for split, analyses in (('train', train), ('test', test)):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "dimensions must be preserved")
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            d = _decode(inp)
            expected = _answer(d)
            _require(np.array_equal(out, expected), "wrong ranked chamber transformation")
            _require(np.any(inp != out), "each example must change")
            # Check the task's unusual distinction between wall erasure and void filling.
            allowed = np.zeros(inp.shape, dtype=bool)
            for o in d['objects']:
                if o['count'] in d['keys']:
                    key = d['keys'][o['count']]
                    chosen = sorted(o['holes'], key=_span, reverse=True)[key['rank']-1]
                    allowed[tuple(chosen.T)] = True
                    _require(np.all(out[tuple(o['cells'].T)] == d['query']), "listed shell altered")
                    _require(np.all(out[tuple(chosen.T)] == key['color']), "selected void not filled")
                    filled = o['mask'].copy()
                    filled[tuple((chosen-o['position']).T)] = True
                    _require(len(_holes(filled)) == o['count']-1,
                             "filling must remove exactly one background cavity")
                else:
                    allowed[tuple(o['cells'].T)] = True
                    _require(np.all(out[tuple(o['cells'].T)] == d['background']), "unlisted shell not erased")
            _require(np.array_equal(inp != out, allowed), "unexpected change outside selected regions")
            analyses.append(d)
    represented = {d['keys'][o['count']]['rank'] for d in train for o in d['objects']
                   if o['count'] in d['keys']}
    _require({1, 2} <= represented, "training must demonstrate ranks one and two")
    _require(any(o['count'] >= 4 and d['keys'][o['count']]['rank'] == 2
                 for d in train for o in d['objects'] if o['count'] in d['keys']),
             "training must show an intermediate rank")
    _require(any(o['count'] not in d['keys'] for d in train for o in d['objects']),
             "training must demonstrate class-based erasure")
    contrasts = {}
    for d in train:
        for k, key in d['keys'].items():
            if any(o['count'] == k for o in d['objects']):
                contrasts.setdefault((k, key['color']), set()).add(key['rank'])
    _require(any({1, 2} <= ranks for ranks in contrasts.values()), "missing same-signature rank contrast")
    _require(not _shortcut_survivors(train), "a tested simpler rule fits all demonstrations")
    _require(any(not np.array_equal(_topology_omission(d), _answer(d)) for d in train),
             "topology contribution is not observable")
    for d in test:
        for o in d['objects']:
            if o['count'] in d['keys']:
                rank = d['keys'][o['count']]['rank']
                _require(rank in represented or set(range(1, rank)) <= represented,
                         "held-out ordinal lacks demonstrated predecessors")


def _trim(mask):
    return _mask_of(np.argwhere(mask))[0]


def _prototype(k, rng, compact=False):
    rows = 2 if k >= 4 else 1
    if not compact and k >= 3 and rng.random() < .35:
        rows = 2
    cols = (k+rows-1)//rows
    mask = np.ones((2*rows+1, 2*cols+1), dtype=bool)
    for i in range(k):
        mask[1+2*(i//cols), 1+2*(i%cols)] = False
    return np.rot90(mask, int(rng.integers(4))).copy()


def _key_tile(k, rank, color, rng, compact=False):
    glyph = _prototype(k, rng, compact)
    gh, gw = glyph.shape
    # Both reference and tally are monochrome but disconnected. No separator is needed.
    if compact:
        tile = np.full((max(gh, rank), gw+2), -1, dtype=int)
        tile[:gh, :gw][glyph] = color
        tile[:rank, -1] = color
    else:
        gap = int(rng.integers(1, 3))
        vertical = bool(rng.integers(2))
        tally_h, tally_w = (rank, 1) if vertical else (1, rank)
        h = max(gh, tally_h)
        tile = np.full((h, gw+gap+tally_w), -1, dtype=int)
        y = int(rng.integers(h-gh+1))
        tile[y:y+gh, :gw][glyph] = color
        y = int(rng.integers(h-tally_h+1))
        tile[y:y+tally_h, -tally_w:] = color
    return np.rot90(tile, int(rng.integers(4))).copy()


def _cavity(span, rng, irregular=False, minor=None):
    minor = int(rng.integers(1, min(4, span)+1)) if minor is None else minor
    hole = np.ones((span, minor), dtype=bool)
    if irregular and span >= 3 and minor >= 2:
        # A winding spine reaches both ends; optional bulbs vary area without
        # changing length. It cannot contain an island or a disconnected piece.
        hole[:] = False
        column = int(rng.integers(minor))
        for r in range(span):
            nxt = int(np.clip(column+int(rng.integers(-1, 2)), 0, minor-1))
            hole[r, min(column, nxt):max(column, nxt)+1] = True
            if rng.random() < .5:
                hole[r, max(0, nxt-1):min(minor, nxt+2)] = True
            column = nxt
        hole = _trim(hole)
    return np.rot90(hole, int(rng.integers(4))).copy()


def _query_mask(k, rng, rich=False, compact=False, witness=False, long=False):
    if witness:
        _require(k == 4, "the area contrast has four cavities")
        if witness == 'path':
            elbow = np.zeros((5, 3), dtype=bool)
            elbow[:, 0] = True
            elbow[-1, :] = True
            cavities = [np.rot90(h, int(rng.integers(4))).copy() for h in
                        (np.ones((6, 1), bool), elbow, np.ones((3, 1), bool), np.ones((1, 1), bool))]
        else:
            cavities = [_cavity(s, rng, minor=m) for s, m in ((1, 1), (2, 2), (3, 3), (5, 1))]
    elif compact:
        # Narrow vertical chambers fit even the five-hole class in 7x11.
        offset = 8-k if long else 0
        cavities = [np.ones((s+offset, 1), dtype=bool) for s in range(1, k+1)]
    else:
        upper = 9 if long else 6
        spans = [int(s) for s in rng.choice(np.arange(1, upper+1), k, replace=False)]
        if long and max(spans) <= 6:
            spans[0] = int(rng.integers(7, 10))
        cavities = [_cavity(s, rng, irregular=bool(rng.random() < .65)) for s in spans]
    rng.shuffle(cavities)
    groups = [cavities]
    if not compact and k >= 2 and (k >= 4 or rng.random() < .65):
        cut = (k+1)//2
        groups = [cavities[:cut], cavities[cut:]]
    heights = [max(h.shape[0]+2 for h in group) for group in groups]
    widths = [sum(h.shape[1]+1 for h in group)+1 for group in groups]
    height, width = sum(heights)-(len(groups)-1), max(widths)
    mask = np.ones((height, width), dtype=bool)
    r = 0
    for group, rh in zip(groups, heights):
        c = int(rng.integers(width-(sum(h.shape[1]+1 for h in group)+1)+1))
        for hole in group:
            hh, hw = hole.shape
            y = r+1+int(rng.integers(rh-hh-1))
            mask[y:y+hh, c+1:c+1+hw][hole] = False
            c += hw+1
        r += rh-1
    original_spans = sorted(_span(h) for h in _holes(mask))
    # Exterior sculpture produces bays, steps, and unequal walls. Every edit
    # is accepted only if the connected shell and ALL cavity spans survive.
    for _ in range(12 if rich else 4):
        candidate = mask.copy()
        side = int(rng.integers(4))
        reach = int(rng.integers(1, 3 if rich else 2))
        if side < 2:
            c = int(rng.integers(width))
            if side == 0:
                candidate[:reach, c] = False
            else:
                candidate[-reach:, c] = False
        else:
            r = int(rng.integers(height))
            if side == 2:
                candidate[r, :reach] = False
            else:
                candidate[r, -reach:] = False
        if (candidate.any() and len(_components(candidate)) == 1
                and sorted(_span(h) for h in _holes(candidate)) == original_spans):
            mask = candidate
    mask = _trim(mask)
    _require(len(_components(mask)) == 1 and len(_holes(mask)) == k, "construction changed topology")
    return np.rot90(mask, int(rng.integers(4))).copy()


def _positions(free, h, w):
    if h > free.shape[0] or w > free.shape[1]:
        return np.empty((0, 2), dtype=int)
    sums = np.pad((~free).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    count = sums[h:, w:]-sums[:-h, w:]-sums[h:, :-w]+sums[:-h, :-w]
    return np.argwhere(count == 0)


def _pack(tiles, rng, background, orientation, compact=False, arrangement=0):
    """Free placement with bounding-box clearance; no semantic slot order."""
    h = 30 if compact else int(rng.integers(18, 31))
    w = 30 if compact else int(rng.integers(18, 31))
    grid = np.full((h, w), background, dtype=np.int8)
    free = np.ones((h, w), dtype=bool)
    free[0] = free[-1] = False
    free[:, 0] = free[:, -1] = False
    # Largest-first packing, randomized among similarly sized tiles.
    ordered = sorted(tiles, key=lambda t: t[0].size*(1+rng.random()*.25), reverse=True)
    for tile, kind, mandatory in ordered:
        th, tw = tile.shape
        places = _positions(free, th, tw)
        if not len(places):
            if mandatory:
                return None
            continue
        # Some scenes cluster the instructions towards one edge; others mix
        # reference packets among queries. The decoder never uses this bias.
        if kind == 'key' and arrangement == 0:
            near = places[places[:, 0] <= places[:, 0].min()+2]
            places = near
        elif kind == 'key' and arrangement == 1:
            near = places[places[:, 1] <= places[:, 1].min()+2]
            places = near
        if compact or rng.random() < .5:
            # Boundary of currently free space, still inside background border.
            score = np.minimum.reduce([places[:, 0], places[:, 1],
                                       h-th-places[:, 0], w-tw-places[:, 1]])
            places = places[score <= score.min()+1]
        r, c = places[int(rng.integers(len(places)))]
        occupied = tile >= 0
        grid[r:r+th, c:c+tw][occupied] = tile[occupied]
        free[max(0, r-1):min(h, r+th+1), max(0, c-1):min(w, c+tw+1)] = False
    # Remove gratuitous empty margins but retain 1..3 background cells per side.
    rc = np.argwhere(grid != background)
    lo, hi = rc.min(axis=0), rc.max(axis=0)+1
    before = np.minimum(lo, rng.integers(1, 4, size=2))
    after = np.minimum(np.array(grid.shape)-hi, rng.integers(1, 4, size=2))
    grid = grid[lo[0]-before[0]:hi[0]+after[0], lo[1]-before[1]:hi[1]+after[1]]
    if rng.random() < .5:
        grid = np.fliplr(grid)
    return np.ascontiguousarray(np.rot90(grid, orientation))


def _make_pair(rng, preferences, spec, compact=False):
    anchor, base_roles, stable, shift, base_rotation = preferences
    if stable:
        bg, query = base_roles
    else:
        roles = [int(c) for c in rng.permutation([c for c in range(10) if c != anchor])]
        bg, query = roles[:2]
    primary = spec['class']
    classes = [primary]
    second_key = not compact and (rng.random() < .45 or
                                  (shift == 'recombine' and spec['split'] == 'test'))
    if second_key:
        pool = range(1, 5) if shift == 'signature' and spec['split'] == 'train' else range(1, 6)
        classes.append(int(rng.choice([k for k in pool if k != primary])))
    available = [int(c) for c in rng.permutation([c for c in range(10)
                                                if c not in (bg, query, anchor)])]
    colors = [anchor if k == 4 else available.pop() for k in classes]
    ranks = [spec['rank']]+[int(rng.integers(1, min(classes[1], 3)+1))] if len(classes) == 2 else [spec['rank']]
    if spec['split'] == 'train':
        ranks = [min(r, 2) for r in ranks]
    tiles = [(_key_tile(k, rank, color, rng, compact), 'key', True)
             for k, rank, color in zip(classes, ranks, colors)]
    # Two same-signature queries show that the operation repeats per object,
    # rather than selecting a single object from the whole scene.
    counts = [primary, primary]
    absent = [k for k in range(1, 6) if k not in classes]
    if spec['anchor'] or rng.random() < .7:
        counts.append(int(rng.choice(absent)))
    else:
        counts.append(int(rng.choice(classes)))
    required = len(counts)
    if not compact:
        while len(counts) < int(rng.integers(4, 7)):
            counts.append(int(rng.integers(1, 6)))
        if len(classes) == 2 and rng.random() < .75:
            counts.append(classes[1])
    rich = spec['split'] == 'test' and shift in ('contour', 'recombine')
    for i, k in enumerate(counts):
        witness = ('area' if i == 0 else 'path' if i == 1 else False) if spec['anchor'] else False
        mask = _query_mask(k, rng, rich=rich, compact=compact, witness=witness,
                           long=bool(shift == 'extent' and spec['split'] == 'test' and i < 2))
        tile = np.full(mask.shape, -1, dtype=int)
        tile[mask] = query
        tiles.append((tile, 'query', i < required))
    orientation = base_rotation
    if shift == 'recombine':
        orientation = int(rng.integers(4))
    arrangement = 2 if spec['split'] == 'test' and shift == 'recombine' else int(rng.integers(3))
    grid = _pack(tiles, rng, bg, orientation, compact, arrangement)
    if grid is None:
        return None
    try:
        d = _decode(grid)
    except AssertionError:
        return None
    return {'input': grid, 'output': _answer(d)}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be between {lo} and {hi}')
    return int(value)


def _fallback_pair(rng, preferences, spec, seen):
    """Guaranteed fit and uniqueness, retaining the teaching/ordinal witnesses.

    Three compact shells fit fixed, separated boxes. Nineteen different key
    translations exceed the maximum thirteen earlier inputs in a puzzle.
    This fallback constructs the SAME rule, with no new marker convention.
    """
    anchor, roles, _, shift, rotation = preferences
    if spec['anchor']:
        rotation = 0  # preserve the explicit row-major-order counterexample
    bg, query = roles
    k, rank = spec['class'], spec['rank']
    color = anchor if k == 4 else next(c for c in range(10) if c not in (bg, query, anchor))

    def chambers(shapes):
        cavities = [np.ones(shape, bool) if isinstance(shape, tuple) else shape for shape in shapes]
        mask = np.ones((max(h.shape[0] for h in cavities)+2,
                        sum(h.shape[1]+1 for h in cavities)+1), bool)
        c = 1
        for hole in cavities:
            h, w = hole.shape
            mask[1:1+h, c:c+w][hole] = False
            c += w+1
        return mask

    long = shift == 'extent' and spec['split'] == 'test'
    spans = list(range(1, k+1))
    if long:
        spans = [s+8-k for s in spans]
    first = chambers([(s, 1) for s in spans])
    second = first.copy()
    if spec['anchor']:
        # Deliberately non-spatial order, with a length/area inversion.
        first = chambers([(5, 1), (2, 2), (3, 3), (1, 1)])
        elbow = np.zeros((5, 3), dtype=bool)
        elbow[:, 0] = elbow[-1, :] = True
        second = chambers([(1, 1), (6, 1), elbow, (3, 1)])
    absent = next(c for c in range(1, 6) if c != k)
    third = chambers([(s, 1) for s in range(1, absent+1)])
    if spec['split'] == 'test' and shift in ('contour', 'recombine'):
        first[0, 0] = False  # diagonal opening leaves four-connected cavities enclosed
    rows = 2 if k >= 4 else 1
    cols = (k+rows-1)//rows
    glyph = np.ones((2*rows+1, 2*cols+1), bool)
    for i in range(k):
        glyph[1+2*(i//cols), 1+2*(i%cols)] = False
    tile = np.full((max(glyph.shape[0], rank), glyph.shape[1]+2), -1, int)
    tile[:glyph.shape[0], :glyph.shape[1]][glyph] = color
    tile[:rank, -1] = color
    for left in range(1, 20):
        grid = np.full((30, 30), bg, dtype=np.int8)
        for mask, r, c in ((first, 8, 1), (second, 8, 15), (third, 20, 1)):
            grid[r:r+mask.shape[0], c:c+mask.shape[1]][mask] = query
        th, tw = tile.shape
        occupied = tile >= 0
        grid[1:1+th, left:left+tw][occupied] = tile[occupied]
        grid = np.ascontiguousarray(np.rot90(grid, rotation))
        if _signature(grid) not in seen:
            return {'input': grid, 'output': solve(grid)}
    raise AssertionError('uniqueness proof violated')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    n = _count(num_train, 'num_train', 2, 10)
    m = _count(num_test, 'num_test', 1, 4)
    if n <= m:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    shift = str(rng.choice(['ordinal', 'extent', 'signature', 'contour', 'recombine']))
    preferences = (palette[2], tuple(palette[:2]), bool(rng.random() < .5),
                   shift, int(rng.integers(4)))
    anchors = [int(i) for i in rng.choice(n, 2, replace=False)]
    specs = {'train': [], 'test': []}
    for split, total in (('train', n), ('test', m)):
        for i in range(total):
            anchor = split == 'train' and i in anchors
            k = 4 if anchor else int(rng.integers(1, 6))
            if split == 'train' and shift == 'signature':
                k = 4 if anchor else int(rng.integers(1, 5))
            rank = anchors.index(i)+1 if anchor else int(rng.integers(1, min(k, 2)+1))
            if split == 'test' and shift in ('ordinal', 'recombine'):
                k, rank = int(rng.integers(3, 6)), 3
            elif split == 'test' and shift == 'signature':
                k = 5
            elif split == 'test' and rng.random() < .3:
                k, rank = int(rng.integers(3, 6)), 3
            specs[split].append(dict(split=split, **{'class': k}, rank=rank, anchor=anchor))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for spec in specs[split]:
            for attempt in range(36):
                pair = _make_pair(rng, preferences, spec, compact=attempt >= 32)
                if pair is not None and _signature(pair['input']) not in seen:
                    seen.add(_signature(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                pair = _fallback_pair(rng, preferences, spec, seen)
                seen.add(_signature(pair['input']))
                puzzle[split].append(pair)
    # With randomized chamber geometry, accidentally aligned spatial ranks are
    # possible. Repair a teaching pair, without changing the transformation.
    train = [_decode(p['input']) for p in puzzle['train']]
    if _shortcut_survivors(train):
        i = anchors[0]
        pair = _fallback_pair(rng, preferences, specs['train'][i], seen)
        seen.remove(_signature(puzzle['train'][i]['input']))
        seen.add(_signature(pair['input']))
        puzzle['train'][i] = pair
    validate_generated_puzzle(puzzle)
    return puzzle


def diversity_features(pair):
    d = _decode(pair['input'])
    objects = d['objects']
    holes = [h for o in objects for h in o['holes']]
    return dict(shape=tuple(d['grid'].shape), background=d['background'], query=d['query'],
                classes=tuple(sorted(d['keys'])), ranks=tuple(sorted(k['rank'] for k in d['keys'].values())),
                queries=len(objects), counts=tuple(sorted(o['count'] for o in objects)),
                spans=tuple(sorted(_span(h) for h in holes)),
                areas=tuple(sorted(len(h) for h in holes)),
                irregular=sum(not _mask_of(h)[0].all() for h in holes),
                removed=sum(o['count'] not in d['keys'] for o in objects),
                unused=sum(not any(o['count'] == k for o in objects) for k in d['keys']),
                area_disagreement=not np.array_equal(_answer(d, 'area'), _answer(d)))


def self_test():
    # Independent tiny topology fixtures: diagonal escape is still enclosed;
    # an orthogonal opening or a deep open notch is not a hole.
    diagonal = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    _require(len(_holes(diagonal)) == 1, 'four-neighbor diagonal convention')
    diagonal[0, 1] = False
    _require(not _holes(diagonal), 'orthogonal opening')
    _require(not _holes(np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], bool)), 'open notch')
    elbow = np.zeros((5, 3), dtype=bool)
    elbow[:, 0] = elbow[-1, :] = True
    # Both are simple paths: cell count equals path length in cells. The
    # seven-cell elbow has extent five; a six-cell straight path has extent six.
    _require(elbow.sum() == 7 and _span(np.argwhere(elbow)) == 5
             and _span(np.argwhere(np.ones((6, 1), bool))) == 6, 'extent versus path witness')
    features, puzzles = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        local = []
        for split in ('train', 'test'):
            for pair in p[split]:
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), 'input-only answer')
                local.append(diversity_features(pair))
        features.append(local)
    flat = [f for fs in features for f in fs]
    _require(len({_signature(p['train'][0]['input']) for p in puzzles}) == 42, 'seed diversity')
    _require(len({f['shape'] for f in flat}) >= 30, 'shape diversity')
    _require({1, 2, 3, 4, 5} <= {k for f in flat for k in f['counts']}, 'topology population coverage')
    _require({1, 2, 3} <= {r for f in flat for r in f['ranks']}, 'ordinal coverage')
    _require(len({f['queries'] for f in flat}) >= 3, 'query population variation')
    _require(len({f['classes'] for f in flat}) >= 10, 'class subset variation')
    _require(sum(f['irregular'] > 0 for f in flat) >= 40, 'irregular cavity coverage')
    _require(sum(f['area_disagreement'] for f in flat) >= 42, 'length versus area evidence')
    _require(any(f['removed'] == 0 for f in flat) and any(f['unused'] for f in flat), 'optional erasure and unused keys')
    _require(len({f['background'] for f in flat}) >= 8, 'semantic palettes')
    for field in ('shape', 'counts', 'spans', 'classes'):
        _require(sum(len({f[field] for f in fs}) > 1 for fs in features) >= 35,
                 f'within-puzzle {field} diversity')
    # Pixel-preserving rotations/reflections and an arbitrary color permutation.
    palette = np.array([7, 3, 8, 0, 6, 2, 9, 1, 5, 4], dtype=np.int8)
    for pair in puzzles[0]['train']+puzzles[0]['test']:
        for rot in range(4):
            for flip in (False, True):
                inp, out = pair['input'], pair['output']
                if flip:
                    inp, out = np.fliplr(inp), np.fliplr(out)
                inp, out = palette[np.rot90(inp, rot)], palette[np.rot90(out, rot)]
                _require(np.array_equal(solve(inp), out), 'geometry/palette invariance')
    for n, m in ((2, 1), (7, 3), (10, 1), (10, 4)):
        p = generate_puzzle(9200+n*10+m, n, m)
        _require(len(p['train']) == n and len(p['test']) == m, 'explicit counts')
    # Exercise the guaranteed fallback directly, including fourteen distinct
    # placements and the rank-four/five members supported by the rule/solver.
    rng = np.random.default_rng(999)
    preferences = (2, (0, 5), True, 'ordinal', 0)
    fallback, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', 10), ('test', 4)):
        for i in range(count):
            anchor = split == 'train' and i < 2
            k = 4 if anchor else 5
            rank = i+1 if anchor else 2 if split == 'train' else 3
            spec = dict(split=split, **{'class': k}, rank=rank, anchor=anchor)
            pair = _fallback_pair(rng, preferences, spec, seen)
            seen.add(_signature(pair['input']))
            fallback[split].append(pair)
    validate_generated_puzzle(fallback)
    for rank in (4, 5):
        spec = dict(split='test', **{'class': 5}, rank=rank, anchor=False)
        pair = _fallback_pair(rng, preferences, spec, set())
        d = _decode(pair['input'])
        out = solve(pair['input'])
        for o in d['objects']:
            if o['count'] == 5:
                filled = [h for h in o['holes'] if np.all(out[tuple(h.T)] == d['keys'][5]['color'])]
                _require(len(filled) == 1 and _span(filled[0]) == 6-rank, 'higher ordinal support')
    first, again = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    _require(all(_signature(a[k]) == _signature(b[k]) for s in first
                 for a, b in zip(first[s], again[s]) for k in a), 'determinism')
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # A cavity fill error and a preserved-wall error must both be rejected.
    for wall in (False, True):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
        pair = bad['test'][0]
        d = _decode(pair['input'])
        obj = next(o for o in d['objects'] if o['count'] in d['keys'])
        cell = obj['cells'][0] if wall else sorted(obj['holes'], key=_span, reverse=True)[d['keys'][obj['count']]['rank']-1][0]
        pair['output'][tuple(cell)] = (int(pair['output'][tuple(cell)])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('wrong output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, topology/ranking invariants, legal counts and metamorphic checks')
