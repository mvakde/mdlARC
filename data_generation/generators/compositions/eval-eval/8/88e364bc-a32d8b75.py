"""Porous convoys: tile substitution participates in rigid boundary collision.

All roles are reconstructed from visible fields. There is no parent import,
generation metadata, seed-dependent rule, or learned state in solve().
"""
from collections import deque
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= g) & (g <= 9)), "color range")
    return g.astype(int, copy=False)


def _components(mask):
    h, w = mask.shape
    unseen = mask.copy()
    result = []
    for rr, cc in np.argwhere(mask):
        start = int(rr), int(cc)
        if not unseen[start]:
            continue
        unseen[start] = False
        queue, cells = deque([start]), []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in CARDINAL:
                y, x = r + dr, c + dc
                if 0 <= y < h and 0 <= x < w and unseen[y, x]:
                    unseen[y, x] = False
                    queue.append((y, x))
        result.append(cells)
    return result


def _box(cells):
    p = np.asarray(cells)
    low, high = p.min(axis=0), p.max(axis=0) + 1
    return int(low[0]), int(high[0]), int(low[1]), int(high[1])


def _border_is(a, color, thickness=1):
    t = thickness
    return (min(a.shape) > 2 * t and np.all(a[:t] == color)
            and np.all(a[-t:] == color) and np.all(a[:, :t] == color)
            and np.all(a[:, -t:] == color))


def _arrow_read(a, wall):
    """Repeated strokes, with two or more trail cells before each tip."""
    if not _border_is(a, wall) or _border_is(a, wall, 2):
        return []
    colors = set(map(int, np.unique(a))) - {wall}
    if len(colors) != 2:
        return []
    readings = []
    for tip in sorted(colors):
        trail = next(iter(colors - {tip}))
        tips = set(map(tuple, np.argwhere(a == tip)))
        trails = set(map(tuple, np.argwhere(a == trail)))
        if len(tips) < 2:
            continue
        for dr, dc in DIRECTIONS:
            used, lengths = set(), []
            for r, c in sorted(tips):
                stroke = []
                r, c = r - dr, c - dc
                while (r, c) in trails:
                    stroke.append((r, c))
                    r, c = r - dr, c - dc
                if len(stroke) < 2 or used.intersection(stroke):
                    break
                used.update(stroke)
                lengths.append(len(stroke))
            if len(lengths) == len(tips) and used == trails:
                readings.append((trail, tip, (dr, dc)))
    return readings


def _decode(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(np.all(g[0] == bg) and np.all(g[-1] == bg)
             and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg), "blank outer margin")
    tiles, arrows = [], []
    for cells in _components(g != bg):
        r0, r1, c0, c1 = _box(cells)
        a = g[r0:r1, c0:c1]
        if len(cells) != a.size:
            continue
        wall = int(a[0, 0])
        if a.shape[0] == a.shape[1] and _border_is(a, wall, 2):
            tile = a[2:-2, 2:-2]
            if 2 <= len(tile) <= 5 and len(np.unique(tile)) == 2 and not np.any(tile == wall):
                tiles.append((tile.copy(), wall, (r0, r1, c0, c1)))
        for trail, tip, d in _arrow_read(a, wall):
            arrows.append((wall, trail, tip, d, (r0, r1, c0, c1)))
    _require(len(tiles) == len(arrows) == 1, "one explicit tile and one unique direction key")
    tile, wall, tile_box = tiles[0]
    awall, trail, tip, direction, arrow_box = arrows[0]
    _require(wall == awall and len({bg, wall, trail, tip}) == 4, "key/wall association")
    _require(not set(map(int, np.unique(tile))).intersection({bg, wall, tip}), "tile role conflict")
    legend = np.zeros(g.shape, bool)
    for r0, r1, c0, c1 in (tile_box, arrow_box):
        legend[r0:r1, c0:c1] = True
    stencils = []
    for cells in _components((g == wall) & ~legend):
        r0, r1, c0, c1 = _box(cells)
        a = g[r0:r1, c0:c1]
        if not _border_is(a, wall):
            continue
        inside = a[1:-1, 1:-1]
        if not set(map(int, np.unique(inside))) <= {bg, tip} or np.count_nonzero(inside == tip) < 2:
            continue
        mask = inside == tip
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        stencils.append((mask, (r0, r1, c0, c1)))
    _require(len(stencils) == 1, "one tightly bounded monochrome stencil")
    stencil, stencil_box = stencils[0]
    r0, r1, c0, c1 = stencil_box
    legend[r0:r1, c0:c1] = True
    _require(np.all(np.isin(g[~legend], (bg, wall, tip))), "unexplained arena cells")
    dots = np.argwhere((g == tip) & ~legend)
    _require(len(dots) == 1, "one launch dot")
    launch = tuple(map(int, dots[0]))
    # Reconstruct enclosed floor components, including chambers on both sides
    # of a baffle. The legend's holes cannot be mistaken for chamber floor.
    free = np.zeros(g.shape, bool)
    for cells in _components(((g == bg) | (g == tip)) & ~legend):
        if any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in cells):
            continue
        for p in cells:
            free[p] = True
    _require(free[launch], "launch must lie on enclosed floor")
    body = np.repeat(np.repeat(stencil, len(tile), axis=0), len(tile), axis=1)
    return dict(grid=g, bg=bg, wall=wall, trail=trail, tip=tip,
                tile=tile, stencil=stencil, direction=direction, launch=launch,
                body=body, free=free, legend=legend,
                boxes=(tile_box, arrow_box, stencil_box))


def _fits(free, body, anchor):
    r, c = anchor
    h, w = body.shape
    return (0 <= r and 0 <= c and r + h <= free.shape[0] and c + w <= free.shape[1]
            and bool(np.all(free[r:r + h, c:c + w][body])))


def _endpoint(free, body, start, direction, corner=True):
    """Solver path: direct translated-footprint collision, one pixel per step."""
    _require(_fits(free, body, start), "initial expanded stencil overlaps a wall")
    r, c = start
    dr, dc = direction
    distance = 0
    while _fits(free, body, (r + dr, c + dc)):
        if corner and dr and dc and (not _fits(free, body, (r + dr, c))
                                    or not _fits(free, body, (r, c + dc))):
            break
        r, c = r + dr, c + dc
        distance += 1
    return (r, c), distance


def _render(info, anchor, exchange=True, tile_turn=0, fill_holes=False):
    out = info['grid'].copy()
    out[info['launch']] = info['bg']
    tile = np.rot90(info['tile'], tile_turn)
    if exchange:
        colors = np.unique(tile)
        tile = colors[0] + colors[1] - tile
    n = len(tile)
    for rr, cc in np.argwhere(np.ones(info['stencil'].shape, bool) if fill_holes else info['stencil']):
        r, c = anchor[0] + int(rr) * n, anchor[1] + int(cc) * n
        out[r:r + n, c:c + n] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode and dock solely from input pixels; all conventions are fixed."""
    info = _decode(input_grid)
    end, _ = _endpoint(info['free'], info['body'], info['launch'], info['direction'])
    return _render(info, end)


def _legal_map(free, body):
    """Construction/validation path: intersect floor windows at body offsets."""
    h, w = body.shape
    ah, aw = free.shape[0] - h + 1, free.shape[1] - w + 1
    if ah <= 0 or aw <= 0:
        return np.zeros((0, 0), bool)
    legal = np.ones((ah, aw), bool)
    for r, c in np.argwhere(body):
        legal &= free[r:r + ah, c:c + aw]
    return legal


def _travel_map(legal, direction, corner=True):
    """Dynamic programming on anchors, independent of solver's ray loop."""
    dr, dc = direction
    h, w = legal.shape
    dist = np.full(legal.shape, -1, int)
    rows = range(h - 1, -1, -1) if dr > 0 else range(h)
    cols = range(w - 1, -1, -1) if dc > 0 else range(w)
    for r in rows:
        for c in cols:
            if not legal[r, c]:
                continue
            nr, nc = r + dr, c + dc
            step = 0 <= nr < h and 0 <= nc < w and legal[nr, nc]
            if step and corner and dr and dc:
                step = bool(legal[nr, c] and legal[r, nc])
            dist[r, c] = 1 + dist[nr, nc] if step else 0
    return dist


def _diagnostics(info):
    body, free, start, d = info['body'], info['free'], info['launch'], info['direction']
    legal = _legal_map(free, body)
    dist = _travel_map(legal, d)
    _require(0 <= start[0] < dist.shape[0] and 0 <= start[1] < dist.shape[1]
             and dist[start] >= 0, "illegal launch")
    steps = int(dist[start])
    end = start[0] + steps * d[0], start[1] + steps * d[1]
    rect = _travel_map(_legal_map(free, np.ones(body.shape, bool)), d)
    loose = _travel_map(legal, d, corner=False)
    n = len(info['tile'])
    tile_dist = _travel_map(_legal_map(free, np.ones((n, n), bool)), d)
    individual = [int(tile_dist[start[0] + int(r) * n, start[1] + int(c) * n])
                  for r, c in np.argwhere(info['stencil'])]
    point_dist = _travel_map(free, d)
    macro_dist = _travel_map(_legal_map(free, info['stencil']), d)
    return dict(distance=steps, end=end, rectangle_witness=int(rect[start]) != steps,
                corner_witness=int(loose[start]) != steps,
                rigid_witness=any(k != steps for k in individual),
                point_witness=int(point_dist[start]) != steps,
                scale_witness=int(macro_dist[start]) != steps,
                individual_distances=individual)


def _make_arrow(rng, direction, wall, trail, tip):
    dr, dc = direction
    count = int(rng.choice([2, 2, 3])) if 0 in direction else 2
    length = int(rng.choice([2, 2, 3]))
    points = []
    for i in range(count):
        stagger = int(rng.integers(2)) if dr and dc else 0
        base = (0, 2 * i) if dr else (2 * i, 0)
        for j in range(length + 1):
            points.append((base[0] + (j + stagger) * dr,
                           base[1] + (j + stagger) * dc, tip if j == length else trail))
    rr, cc = [p[0] for p in points], [p[1] for p in points]
    a = np.full((max(rr) - min(rr) + 3, max(cc) - min(cc) + 3), wall, int)
    for r, c, color in points:
        a[r - min(rr) + 1, c - min(cc) + 1] = color
    _require(_arrow_read(a, wall) == [(trail, tip, direction)], "ambiguous sampled key")
    return a


def _make_tile(rng, n, colors):
    for _ in range(30):
        bits = rng.random((n, n)) < rng.uniform(.25, .75)
        if bits.any() and not bits.all() and not np.array_equal(bits, np.rot90(bits, 2)):
            return np.where(bits, colors[0], colors[1]).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(int)


def _make_mask(rng, sh, sw, comb=False):
    for _ in range(30):
        mask = rng.random((sh, sw)) < rng.uniform(.35, .8)
        if comb:
            mask[:, int(rng.integers(1, sw - 1))] = False
            mask[0, 0] = mask[-1, -1] = True
        if mask.sum() < 2:
            continue
        rr, cc = np.nonzero(mask)
        mask = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        if comb and mask.all():
            continue
        return mask
    mask = np.ones((sh, sw), bool)
    if comb:
        mask[:, 1] = False
    return mask


def _arena(rng, h, w, comb=False):
    footprint = np.ones((h, w), bool)
    style = int(rng.integers(4))
    if style == 1:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, min(4, h // 3))), int(rng.integers(1, min(4, w // 3)))
            rows = slice(0, dh) if corner < 2 else slice(h - dh, h)
            cols = slice(0, dw) if corner % 2 == 0 else slice(w - dw, w)
            footprint[rows, cols] = False
    elif style == 2:
        y, x = np.indices((h, w))
        footprint = (np.minimum(y, h - 1 - y) + np.minimum(x, w - 1 - x) >= int(rng.integers(2, 5)))
    elif style == 3:
        left = right = 0
        for r in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, 2))
            right = int(np.clip(right + rng.integers(-1, 2), 0, 2))
            footprint[r, :left] = False
            if right:
                footprint[r, w - right:] = False
    # Eight-neighbor erosion makes a closed boundary with conservative corners.
    p = np.pad(footprint, 1)
    free = footprint.copy()
    for dr, dc in DIRECTIONS:
        free &= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    wall = footprint & ~free
    if comb:
        col = int(rng.integers(w // 3, 2 * w // 3 + 1))
        wall[:, col] |= footprint[:, col]
    else:
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < .5:
                c = int(rng.integers(3, w - 3))
                cut = int(rng.integers(3, h - 2))
                rows = slice(0, cut) if rng.random() < .5 else slice(cut, h)
                wall[rows, c] |= footprint[rows, c]
            else:
                r = int(rng.integers(3, h - 3))
                cut = int(rng.integers(3, w - 2))
                cols = slice(0, cut) if rng.random() < .5 else slice(cut, w)
                wall[r, cols] |= footprint[r, cols]
    return wall, footprint & ~wall


def _make_pair(rng, direction, n, witness=None):
    # Random bounded rejection selects *visible* geometry meeting a curriculum
    # witness. No witness flag or sampling choice is consulted by solve().
    for attempt in range(180):
        bg, wall, trail, tip, ca, cb = map(int, rng.choice(10, 6, replace=False))
        if rng.random() < .35:
            trail = int(rng.choice([ca, cb]))
        rotation = int(rng.integers(4))
        base_d = direction
        for _ in range(rotation):
            base_d = base_d[1], -base_d[0]
        # Comb witness is horizontal in the base map, then rotated to any
        # cardinal direction. This varies which geometric bottleneck is active.
        if witness == 'rectangle':
            rotation = next(k for k in range(4)
                            if _rotate_direction((0, 1), k) == direction)
            if rng.random() < .5:
                rotation = next(k for k in range(4)
                                if _rotate_direction((0, -1), k) == direction)
                base_d = (0, -1)
            else:
                base_d = (0, 1)
        arrow = _make_arrow(rng, base_d, wall, trail, tip)
        tile = _make_tile(rng, n, (ca, cb))
        sh = int(rng.integers(1 if witness != 'rectangle' else 2, min(4, 16 // n) + 1))
        sw = int(rng.integers(3 if witness == 'rectangle' else 2, min(5, 22 // n) + 1))
        mask = _make_mask(rng, sh, sw, comb=witness == 'rectangle')
        sh, sw = mask.shape
        tile_card = np.pad(tile, 2, constant_values=wall)
        stencil_card = np.pad(np.where(mask, tip, bg), 1, constant_values=wall)
        cards = [tile_card, stencil_card, arrow]
        band_h = max(a.shape[0] for a in cards) + int(rng.integers(2))
        total_w = sum(a.shape[1] for a in cards) + 2
        if total_w > 28:
            continue
        max_h = 30 - band_h - 3
        min_h = max(10, n * sh + 3)
        min_w = max(12, n * sw + 3)
        if min_h > max_h or min_w > 28:
            continue
        ah = int(rng.integers(min_h, max_h + 1))
        aw = int(rng.integers(min_w, 29))
        width = max(total_w, aw) + 2 + int(rng.integers(0, min(2, 30 - max(total_w, aw) - 1)))
        height = band_h + ah + 3
        a = np.full((height, width), bg, int)
        x = int(rng.integers(1, width - total_w))
        for ix in rng.permutation(3):
            card = cards[int(ix)]
            y = int(rng.integers(1, band_h - card.shape[0] + 2))
            a[y:y + card.shape[0], x:x + card.shape[1]] = card
            x += card.shape[1] + 1
        walls, floor = _arena(rng, ah, aw, comb=witness == 'rectangle')
        y, x = band_h + 2, int(rng.integers(1, width - aw))
        a[y:y + ah, x:x + aw][walls] = wall
        free = np.zeros(a.shape, bool)
        free[y:y + ah, x:x + aw] = floor
        a, free = np.rot90(a, rotation).copy(), np.rot90(free, rotation).copy()
        mask, tile = np.rot90(mask, rotation).copy(), np.rot90(tile, rotation).copy()
        body = np.repeat(np.repeat(mask, n, axis=0), n, axis=1)
        legal = _legal_map(free, body)
        dist = _travel_map(legal, direction)
        if not dist.size:
            continue
        choices = dist > 0
        choices &= free[:dist.shape[0], :dist.shape[1]]
        if witness == 'rectangle':
            rectangular = _travel_map(_legal_map(free, np.ones(body.shape, bool)), direction)
            choices &= rectangular != dist
        elif witness == 'corner':
            choices &= _travel_map(legal, direction, corner=False) != dist
        if not choices.any():
            continue
        candidates = np.argwhere(choices)
        launch = tuple(map(int, candidates[int(rng.integers(len(candidates)))]))
        a[launch] = tip
        info = _decode(a)
        _require(np.array_equal(info['free'], free), "floor recovery disagrees")
        diag = _diagnostics(info)
        if witness is not None and not diag['rigid_witness']:
            continue
        if witness == 'rectangle' and not (diag['point_witness'] and diag['scale_witness']):
            continue
        # Build labels with anchor-map arithmetic, rather than solve's ray loop.
        output = a.copy()
        output[launch] = bg
        swapped = ca + cb - tile
        end = diag['end']
        for rr, cc in np.argwhere(mask):
            r, c = end[0] + int(rr) * n, end[1] + int(cc) * n
            output[r:r + n, c:c + n] = swapped
        return {'input': a, 'output': output}
    raise RuntimeError(f"bounded construction exhausted for {direction}, tile {n}, witness {witness}")


def _rotate_direction(d, k):
    for _ in range(k):
        d = -d[1], d[0]
    return d


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    cardinal = CARDINAL[int(rng.integers(4))]
    diagonal = DIAGONAL[int(rng.integers(4))]
    sizes = list(map(int, rng.permutation([2, 3])))
    plans = [(cardinal, sizes[0], 'rectangle'), (diagonal, sizes[1], 'corner')]
    plans = [plans[int(i)] for i in rng.permutation(2)]
    pool = [cardinal, diagonal]
    held = [d for d in DIRECTIONS if d not in pool]
    plans += [(pool[int(rng.integers(2))], int(rng.integers(2, 6)), None) for _ in range(nt - 2)]
    plans += [(held[int(rng.integers(len(held)))], int(rng.integers(2, 6)), None) for _ in range(nq)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, (direction, n, witness) in enumerate(plans):
        for _ in range(8):
            pair = _make_pair(rng, direction, n, witness)
            key = _signature(pair['input'])
            if key not in seen:
                break
        else:
            raise RuntimeError('unique input construction exhausted')
        seen.add(key)
        puzzle['train' if i < nt else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(g):
    return g.shape, g.astype(np.uint8).tobytes()


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'convoy preserves canvas dimensions')
            sig = _signature(a)
            _require(sig not in seen, 'duplicate input')
            seen.add(sig)
            info = _decode(a)
            data = _diagnostics(info)
            _require(data['distance'] > 0, 'each example must show motion')
            expected = _render(info, data['end'])
            _require(np.array_equal(b, expected), 'wrong docking / exchange label')
            end, distance = _endpoint(info['free'], info['body'], info['launch'], info['direction'])
            _require(end == data['end'] and distance == data['distance'], 'two collision methods disagree')
            _require(np.array_equal(a[info['legend']], b[info['legend']]), 'legend changed')
            walls = (a == info['wall']) & ~info['legend']
            _require(np.array_equal(a[walls], b[walls]), 'wall painted through stencil hole')
            n = len(info['tile'])
            tile_colors = np.unique(info['tile'])
            # Tile roles are absent in the arena before docking. Thus every
            # output tile cell can be counted, including both binary roles.
            painted = (~info['legend']) & np.isin(b, tile_colors)
            _require(int(painted.sum()) == int(info['stencil'].sum()) * n * n, 'stamp area/count')
            for color in tile_colors:
                other = tile_colors[0] + tile_colors[1] - color
                _require(int(np.count_nonzero(b[~info['legend']] == color))
                         == int(info['stencil'].sum()) * int(np.count_nonzero(info['tile'] == other)),
                         'binary exchange histogram')
            (training if split == 'train' else testing).append((info, data))
    train_dirs = {i['direction'] for i, _ in training}
    _require(any(0 in d for d in train_dirs) and any(0 not in d for d in train_dirs), 'cardinal/diagonal evidence')
    for name in ('rectangle_witness', 'corner_witness', 'rigid_witness', 'point_witness', 'scale_witness'):
        _require(any(d[name] for _, d in training), f'missing training {name}')
    _require({len(i['tile']) for i, _ in training} >= {2, 3}, 'training must establish tile scale')
    _require(all(i['direction'] not in train_dirs for i, _ in testing), 'held-out direction shift')
    for alternative in ('static', 'reverse', 'no_exchange', 'rotate_tile', 'fill_holes'):
        different = False
        for info, data in training:
            if alternative == 'static':
                alt = _render(info, info['launch'])
            elif alternative == 'reverse':
                end, _ = _endpoint(info['free'], info['body'], info['launch'],
                                   tuple(-x for x in info['direction']))
                alt = _render(info, end)
            else:
                alt = _render(info, data['end'], exchange=alternative != 'no_exchange',
                              tile_turn=int(alternative == 'rotate_tile'),
                              fill_holes=alternative == 'fill_holes')
            different |= not np.array_equal(alt, _render(info, data['end']))
        _require(different, f'training admits {alternative}')


def self_test():
    """Bounded family tests and pixel-derived structural diversity, 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, masks, sizes, directions, moves, layouts = set(), set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        signatures.add(tuple(_signature(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                data = _diagnostics(info)
                _require(np.array_equal(solve(pair['input']), pair['output']), 'input-only solve')
                shapes.add(info['grid'].shape)
                masks.add(_signature(info['stencil']))
                sizes.add(len(info['tile']))
                directions.add(info['direction'])
                moves.add(data['distance'])
                # Which side the legends occupy relative to the floor.
                r0, r1, c0, c1 = _box(np.argwhere(info['free']))
                br0, br1, bc0, bc1 = info['boxes'][0]
                layouts.add('top' if br1 <= r0 else 'bottom' if br0 >= r1
                            else 'left' if bc1 <= c0 else 'right')
    _require(len(signatures) == 42, 'distinct seeded puzzles')
    _require(len(shapes) >= 25 and len(masks) >= 55, 'geometric variation')
    _require(sizes == {2, 3, 4, 5} and directions == set(DIRECTIONS), 'scale / compass coverage')
    _require(layouts == {'top', 'bottom', 'left', 'right'}, 'legend placement variety')
    _require(1 in moves and max(moves) >= 5, 'short and longer travel')
    _require(all(_signature(a[k]) == _signature(b[k])
                 for s in ('train', 'test') for a, b in zip(puzzles[0][s], generate_puzzle(0)[s])
                 for k in ('input', 'output')), 'determinism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(100 + nt * 4 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (damaged['test'][0]['output'][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'input_shapes': len(shapes),
            'stencil_structures': len(masks), 'tile_sizes': sorted(sizes),
            'directions': len(directions), 'travel_distances': sorted(moves)}


if __name__ == '__main__':
    print(self_test())
