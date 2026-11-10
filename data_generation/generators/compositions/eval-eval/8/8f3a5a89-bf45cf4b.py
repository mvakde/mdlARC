"""Socket-tile shoreline: a visible tile dictionary controls reachability and ink.

Only Python's standard library and NumPy are used. No parent code is imported.
"""
from collections import deque
import json
import numpy as np

BG, SEED, FRAME = 0, 1, 5
# Socket order is north, east, south, west.
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ROLE_POOL = (2, 3, 4, 6, 7, 8, 9)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(d < 1 or d > 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('grids must be integer 2D arrays, dimensions 1..30, colors 0..9')
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _decode(input_grid):
    """Read every convention from a frame, key squares, and visible prototypes."""
    grid = _grid(input_grid)
    frame_cells = list(map(tuple, np.argwhere(grid == FRAME)))
    if not frame_cells:
        raise ValueError('missing plan frame')
    r0, r1, c0, c1 = _bbox(frame_cells)
    fh, fw = r1-r0, c1-c0
    if fh < 6 or fw < 6 or fh % 2 or fw % 2:
        raise ValueError('plan frame must surround complete 2x2 rooms')
    framed = grid[r0:r1, c0:c1]
    boundary = np.zeros(framed.shape, bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    if not np.array_equal(framed == FRAME, boundary):
        raise ValueError('5 must be exactly the plan perimeter')
    plan = framed[1:-1, 1:-1]
    h, w = plan.shape[0]//2, plan.shape[1]//2
    rooms = np.zeros((h, w), np.int8)
    seed = None
    for r in range(h):
        for c in range(w):
            square = plan[2*r:2*r+2, 2*c:2*c+2]
            values = set(map(int, square.flat))
            if values == {BG}:
                continue
            keys = values - {BG, SEED}
            if len(keys) != 1 or BG in values or not keys <= set(ROLE_POOL):
                raise ValueError('room must be a key square with an optional seed dot')
            key = keys.pop()
            nseed = int(np.count_nonzero(square == SEED))
            if nseed not in (0, 1) or np.count_nonzero(square == key) != 4-nseed:
                raise ValueError('invalid room encoding')
            rooms[r, c] = key
            if nseed:
                if seed is not None:
                    raise ValueError('multiple seed dots')
                seed = r, c
    if seed is None or np.count_nonzero(grid == SEED) != 1:
        raise ValueError('exactly one plan seed is required')
    keys = set(map(int, np.unique(rooms))) - {BG}
    if not 2 <= len(keys) <= 3:
        raise ValueError('two or three room types are required')
    remainder = grid.copy()
    remainder[r0:r1, c0:c1] = BG
    tiles, sockets, boxes = {}, {}, {}
    while np.any(remainder != BG):
        # The solid outer border supplies the crop even when interior holes
        # isolate an ink cell. Consume the entire rectangle, not just its rim.
        cells = _components(remainder != BG)[0]
        tr0, tr1, tc0, tc1 = _bbox(cells)
        tile = remainder[tr0:tr1, tc0:tc1].copy()
        remainder[tr0:tr1, tc0:tc1] = BG
        th, tw = tile.shape
        if th < 3 or tw < 3 or not th % 2 or not tw % 2:
            raise ValueError('tiles must have odd dimensions at least three')
        if not (np.all(tile[0] != BG) and np.all(tile[-1] != BG)
                and np.all(tile[:, 0] != BG) and np.all(tile[:, -1] != BG)):
            raise ValueError('tile bounds must have a solid visible border')
        key = int(tile[th//2, tw//2])
        if key not in keys or key in tiles:
            raise ValueError('one prototype per room key is required')
        border = np.zeros(tile.shape, bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        port_positions = ((0, tw//2), (th//2, tw-1), (th-1, tw//2), (th//2, 0))
        for position in port_positions:
            border[position] = False
        if not np.all(tile[border] == key):
            raise ValueError('non-socket border cells must carry the key')
        ports = tuple(int(tile[q]) for q in port_positions)
        if any(p in keys or p in (BG, SEED, FRAME) for p in ports):
            raise ValueError('socket colors must be disjoint from key colors')
        if not set(map(int, tile.flat)) <= {BG, key, *ports}:
            raise ValueError('tile interiors use their key, socket colors, and background')
        tiles[key], sockets[key], boxes[key] = tile, ports, (tr0, tr1, tc0, tc1)
    if set(tiles) != keys or len({t.shape for t in tiles.values()}) != 1:
        raise ValueError('the complete dictionary must share one tile shape')
    th, tw = next(iter(tiles.values())).shape
    if h*th > 30 or w*tw > 30:
        raise ValueError('expanded plan exceeds grid bounds')
    return {'rooms': rooms, 'seed': seed, 'tiles': tiles, 'sockets': sockets,
            'tile_shape': (th, tw), 'frame_box': (r0, r1, c0, c1), 'tile_boxes': boxes}


def _reach(rooms, seed, sockets, ignore_ports=False):
    h, w = rooms.shape
    reached = np.zeros(rooms.shape, bool)
    reached[seed] = True
    todo = deque([seed])
    while todo:
        r, c = todo.popleft()
        key = int(rooms[r, c])
        for d, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r+dr, c+dc
            if not (0 <= nr < h and 0 <= nc < w) or not rooms[nr, nc] or reached[nr, nc]:
                continue
            other = int(rooms[nr, nc])
            if ignore_ports or sockets[key][d] == sockets[other][(d+2) % 4]:
                reached[nr, nc] = True
                todo.append((nr, nc))
    return reached


def _rim(selected, include_holes=False, diagonal=True):
    """Four-connected padded exterior; eight-neighbor contact inside the region."""
    complement = np.pad(~selected, 1, constant_values=True)
    if include_holes:
        exterior = complement
    else:
        exterior = np.zeros(complement.shape, bool)
        exterior[0, 0] = True
        todo = deque([(0, 0)])
        h, w = complement.shape
        while todo:
            r, c = todo.popleft()
            for dr, dc in DIRECTIONS:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and complement[nr, nc] and not exterior[nr, nc]:
                    exterior[nr, nc] = True
                    todo.append((nr, nc))
    h, w = selected.shape
    contact = np.zeros(selected.shape, bool)
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)] if diagonal else DIRECTIONS
    for dr, dc in offsets:
        contact |= exterior[1+dr:1+dr+h, 1+dc:1+dc+w]
    return selected & contact


def _analysis(info):
    selected = _reach(info['rooms'], info['seed'], info['sockets'])
    return selected, _rim(selected)


def _write_blocks(info, rim, solid=False):
    """Label construction uses explicit block writes, preserving all tile cells."""
    rooms, (th, tw) = info['rooms'], info['tile_shape']
    output = np.zeros((rooms.shape[0]*th, rooms.shape[1]*tw), np.int8)
    for r, c in np.argwhere(rim):
        key = int(rooms[r, c])
        output[r*th:(r+1)*th, c*tw:(c+1)*tw] = key if solid else info['tiles'][key]
    sr, sc = info['seed']
    output[sr*th+th//2, sc*tw+tw//2] = SEED
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only prediction: geometry, keys, sockets and material all are visible."""
    info = _decode(input_grid)
    _, rim = _analysis(info)
    rooms, (th, tw) = info['rooms'], info['tile_shape']
    rr, cc = np.indices((rooms.shape[0]*th, rooms.shape[1]*tw))
    output = np.zeros(rr.shape, np.int8)
    for key, tile in info['tiles'].items():
        active = rim[rr//th, cc//tw] & (rooms[rr//th, cc//tw] == key)
        output[active] = tile[rr[active] % th, cc[active] % tw]
    sr, sc = info['seed']
    output[sr*th+th//2, sc*tw+tw//2] = SEED
    return output


def _rotate_ports(ports, turns):
    # np.rot90 maps old east to new north.
    return tuple(ports[(d+turns) % 4] for d in range(4))


def _palette(rng, count):
    colors = [int(v) for v in rng.permutation(ROLE_POOL)]
    return colors[:count], colors[count:count+2]


def _tile(rng, shape, key, ports, asymmetric=False):
    th, tw = shape
    tile = np.full(shape, key, np.int8)
    r, c = np.indices(shape)
    interior = (r > 0) & (r < th-1) & (c > 0) & (c < tw-1)
    mode = int(rng.integers(4))
    if mode == 0:
        ink = (r+c) % 3 == 0
    elif mode == 1:
        ink = r % 2 == 1
    elif mode == 2:
        ink = (r-th//2)**2 + (c-tw//2)**2 <= min(th, tw)//2
    else:
        ink = rng.random(shape) < .35
    tile[interior & ink] = np.array(ports, np.int8)[(r[interior & ink]+2*c[interior & ink]) % 4]
    holes = interior & (rng.random(shape) < float(rng.uniform(0, .32)))
    tile[holes] = BG
    tile[th//2, tw//2] = key
    if asymmetric:
        # Both axes are at least five in the orientation/hole witness.
        tile[interior] = key
        tile[1, 1], tile[1, 2], tile[th-2, tw-2] = BG, ports[0], ports[1]
    positions = ((0, tw//2), (th//2, tw-1), (th-1, tw//2), (th//2, 0))
    for q, port in zip(positions, ports):
        tile[q] = port
    return tile


def _display(rng, rooms, seed, tiles):
    h, w = rooms.shape
    plan = np.full((2*h+2, 2*w+2), FRAME, np.int8)
    plan[1:-1, 1:-1] = np.repeat(np.repeat(rooms, 2, axis=0), 2, axis=1)
    sub_r, sub_c = map(int, rng.integers(2, size=2))
    plan[1+2*seed[0]+sub_r, 1+2*seed[1]+sub_c] = SEED
    dictionary = list(tiles.values())
    rng.shuffle(dictionary)
    th, tw = dictionary[0].shape
    options = []
    for horizontal in (False, True):
        for gap in (1, 2, 3):
            for dictionary_gap in (1, 2):
                dh = len(tiles)*th + (len(tiles)-1)*dictionary_gap if horizontal else th
                dw = tw if horizontal else len(tiles)*tw + (len(tiles)-1)*dictionary_gap
                bh = max(plan.shape[0], dh) if horizontal else plan.shape[0]+gap+dh
                bw = plan.shape[1]+gap+dw if horizontal else max(plan.shape[1], dw)
                if bh <= 30 and bw <= 30:
                    options.append((horizontal, gap, dictionary_gap, dh, dw, bh, bw))
    horizontal, gap, dg, dh, dw, bh, bw = options[int(rng.integers(len(options)))]
    ch = bh+int(rng.integers(min(3, 30-bh)+1))
    cw = bw+int(rng.integers(min(3, 30-bw)+1))
    reverse = bool(rng.integers(2))
    if horizontal:
        start = int(rng.integers(cw-bw+1))
        pc, dc = (start+dw+gap, start) if reverse else (start, start+plan.shape[1]+gap)
        pr, dr = int(rng.integers(ch-plan.shape[0]+1)), int(rng.integers(ch-dh+1))
    else:
        start = int(rng.integers(ch-bh+1))
        pr, dr = (start+dh+gap, start) if reverse else (start, start+plan.shape[0]+gap)
        pc, dc = int(rng.integers(cw-plan.shape[1]+1)), int(rng.integers(cw-dw+1))
    grid = np.zeros((ch, cw), np.int8)
    grid[pr:pr+plan.shape[0], pc:pc+plan.shape[1]] = plan
    for i, tile in enumerate(dictionary):
        tr, tc = (dr+i*(th+dg), dc) if horizontal else (dr, dc+i*(tw+dg))
        grid[tr:tr+th, tc:tc+tw] = tile
    return grid


def _witness(rng, index, keys, colors, rotation, tile_shape):
    a, b = keys[:2]
    p, q = colors
    sockets = {a: (p, p, p, p), b: (q, p, q, p)}
    if len(keys) == 3:
        sockets[keys[2]] = (p, q, p, q) if index == 0 else (q, p, q, p)
    if index == 0:
        h, w = 8, int(rng.integers(7, 9))
    else:
        th, tw = tile_shape
        # Bounds refer to the final orientation, independently of room layout.
        oh = int(rng.integers(4, min(6, 30//th)+1))
        ow = int(rng.integers(4, min(6, 30//tw)+1))
        h, w = (ow, oh) if rotation % 2 else (oh, ow)
    rooms = np.full((h, w), a, np.int8)
    rooms[-1] = b
    if index == 0:
        rooms[3, w//2] = BG             # an enclosed hole, not a new shoreline
        rooms[:2, 1] = BG              # concavity teaches diagonal rim contact
        rooms[2, -1] = b               # another type on the exterior shoreline
        rooms[2, -2] = b               # same type can also be an interior room
        rooms[4, 1] = keys[-1]
        rooms[0, w-3] = keys[-1]       # third type (if any) has visible output evidence
    else:
        rooms[0, 1] = BG
        rooms[1, -2] = b
        if len(keys) == 3:
            rooms[1, 1] = keys[2]
            rooms[-1, ::2] = keys[2]  # small selected region uses both dictionary entries
    rooms = np.rot90(rooms, rotation).copy()
    sockets = {key: _rotate_ports(ports, rotation) for key, ports in sockets.items()}
    # Locate the old bottom band after the shared rotation; witnesses choose
    # opposite sides, so no fixed corner selects both demonstration regions.
    old_band = np.zeros((h, w), bool)
    old_band[-1] = True
    band = np.rot90(old_band, rotation)
    candidates = np.argwhere(band if index else ((rooms != BG) & ~band))
    seed = tuple(map(int, candidates[int(rng.integers(len(candidates)))]))
    selected = _reach(rooms, seed, sockets)
    if index == 0:
        inner = np.argwhere(selected & ~_rim(selected))
        seed = tuple(map(int, inner[int(rng.integers(len(inner)))]))
    return rooms, seed, sockets


def _random_rooms(rng, shape, keys, sockets):
    h, w = shape
    r, c = np.indices(shape)
    mode = int(rng.integers(5))
    active = np.ones(shape, bool)
    if mode == 0:
        active = rng.random(shape) > float(rng.uniform(.08, .30))
    elif mode == 1:
        thickness = int(rng.integers(1, min(3, (min(shape)+1)//2)))
        active = (r < thickness) | (r >= h-thickness) | (c < thickness) | (c >= w-thickness)
    elif mode == 2:
        active[:, int(rng.integers(1, w-1))] = False
        active[int(rng.integers(1, h-1)), :int(rng.integers(1, w))] = False
    elif mode == 3:
        active = np.zeros(shape, bool)
        rr = cc = 0
        active[rr, cc] = True
        while rr < h-1 or cc < w-1:
            choices = ([(1, 0)] if rr < h-1 else []) + ([(0, 1)] if cc < w-1 else [])
            dr, dc = choices[int(rng.integers(len(choices)))]
            rr, cc = rr+dr, cc+dc
            active[rr, cc] = True
        active |= rng.random(shape) < .15
    else:
        for _ in range(int(rng.integers(1, 5))):
            rr, cc = int(rng.integers(h)), int(rng.integers(w))
            active[rr:min(h, rr+int(rng.integers(1, 4))), cc:min(w, cc+int(rng.integers(1, 4)))] = False
    if int(active.sum()) < len(keys)+1:
        active[0] = True
        active[:, 0] = True
    pattern = int(rng.integers(4))
    if pattern == 0:
        labels = r % len(keys)
    elif pattern == 1:
        labels = c % len(keys)
    elif pattern == 2:
        labels = (r+c) % len(keys)
    else:
        labels = rng.integers(len(keys), size=shape)
    rooms = np.where(active, np.asarray(keys, np.int8)[labels], BG).astype(np.int8)
    positions = np.argwhere(active)
    # Make dictionary bounds and correspondence visible even for a small path.
    for key, pos in zip(keys, positions[rng.permutation(len(positions))]):
        rooms[tuple(pos)] = key
    groups, left = [], rooms != BG
    while np.any(left):
        start = tuple(map(int, np.argwhere(left)[0]))
        region = _reach(rooms, start, sockets)
        groups.append(region)
        left &= ~region
    suitable = [m for m in groups if m.sum() >= 2] or groups
    region = suitable[int(rng.integers(len(suitable)))]
    candidates = np.argwhere(region)
    seed = tuple(map(int, candidates[int(rng.integers(len(candidates)))]))
    return rooms, seed


def _pair(rng, split, index, palette, rotation):
    keys, colors = palette
    if split == 'train' and index == 0:
        shape = (3, 3)
        rooms, seed, sockets = _witness(rng, 0, keys, colors, rotation, shape)
    elif split == 'train' and index == 1:
        shape = ((5, 5), (5, 7), (7, 5))[int(rng.integers(3))]
        rooms, seed, sockets = _witness(rng, 1, keys, colors, rotation, shape)
    else:
        dimensions = []
        for th in (3, 5, 7):
            for tw in (3, 5, 7):
                for h in range(3, min(10 if split == 'test' else 8, 30//th)+1):
                    for w in range(3, min(10 if split == 'test' else 8, 30//tw)+1):
                        if split == 'train' or max(h, w) >= 9:
                            dimensions.append((h, w, th, tw))
        h, w, th, tw = dimensions[int(rng.integers(len(dimensions)))]
        shape = th, tw
        p, q = colors
        profiles = ((p, p, p, p), (q, p, q, p), (p, q, p, q), (q, q, q, q))
        # Keys never have a hidden fixed socket profile; each prototype supplies it.
        sockets = {key: profiles[int(rng.integers(len(profiles)))] for key in keys}
        if len(set(sockets.values())) == 1:
            sockets[keys[-1]] = profiles[(profiles.index(sockets[keys[-1]])+1) % 4]
        rooms, seed = _random_rooms(rng, (h, w), keys, sockets)
    tiles = {key: _tile(rng, shape, key, sockets[key], asymmetric=(split == 'train' and index == 1)) for key in keys}
    inp = _display(rng, rooms, seed, tiles)
    info = _decode(inp)
    _, rim = _analysis(info)
    return {'input': inp, 'output': _write_blocks(info, rim)}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(array):
    a = np.asarray(array)
    return a.shape, tuple(map(int, a.flat))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(int(seed))
    nkeys = int(rng.integers(2, 4))
    shared_palette = _palette(rng, nkeys)
    stable = bool(rng.random() < .25)
    rotation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            palette = shared_palette if stable else _palette(rng, nkeys)
            for _ in range(64):
                pair = _pair(rng, split, index, palette, rotation)
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('unable to construct distinct input')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    rooms, seed = info['rooms'], info['seed']
    selected, rim = _analysis(info)
    ignored = _reach(rooms, seed, info['sockets'], ignore_ports=True)
    full = _rim(selected, include_holes=True)
    corners = {i for i, q in enumerate(((0, 0), (0, -1), (-1, 0), (-1, -1))) if selected[q]}
    left, areas = rooms != BG, []
    while np.any(left):
        start = tuple(map(int, np.argwhere(left)[0]))
        component = _reach(rooms, start, info['sockets'])
        areas.append(int(component.sum()))
        left &= ~component
    return {'map_shape': rooms.shape, 'tile_shape': info['tile_shape'],
            'selected': int(selected.sum()), 'rim': int(rim.sum()),
            'occupied': int(np.count_nonzero(rooms)), 'types': len(info['tiles']),
            'ports_matter': not np.array_equal(rim, _rim(ignored)),
            'hole_exclusion': bool(np.any(full & ~rim)),
            'diagonal': bool(np.any(rim & ~_rim(selected, diagonal=False))),
            'interior_seed': not bool(rim[seed]), 'corners': corners,
            'mixed_selected': len(set(map(int, rooms[selected]))) > 1,
            'components': len(areas), 'smaller_selection': int(selected.sum()) < max(areas)}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = _key(inp)
            assert signature not in seen, 'duplicate input'
            seen.add(signature)
            info = _decode(inp)
            selected, rim = _analysis(info)
            th, tw = info['tile_shape']
            rooms = info['rooms']
            assert out.shape == (rooms.shape[0]*th, rooms.shape[1]*tw)
            assert np.array_equal(out, solve(inp)), 'incorrect semantic output'
            assert np.count_nonzero(out == SEED) == 1
            assert FRAME not in out
            # Audit each complete block, including retained prototype holes and
            # the one explicitly specified seed overlay.
            for r in range(rooms.shape[0]):
                for c in range(rooms.shape[1]):
                    expected = info['tiles'][int(rooms[r, c])].copy() if rim[r, c] else np.zeros((th, tw), np.int8)
                    if (r, c) == info['seed']:
                        expected[th//2, tw//2] = SEED
                    assert np.array_equal(out[r*th:(r+1)*th, c*tw:(c+1)*tw], expected)
            assert selected[info['seed']] and np.any(rim)
            assert not np.array_equal(out, _write_blocks(info, rim, solid=True)), 'tile content must matter'
            infos[split].append(info)
    fs = [_features(i) for i in infos['train']]
    assert any(f['ports_matter'] for f in fs), 'teach socket-sensitive selection'
    assert any(f['hole_exclusion'] for f in fs), 'teach exterior rather than every boundary'
    assert any(f['diagonal'] for f in fs), 'teach eight-neighbor rim contact'
    assert any(f['interior_seed'] for f in fs) and any(not f['interior_seed'] for f in fs)
    assert any(f['mixed_selected'] for f in fs), 'teach cross-type socket compatibility'
    assert any(f['smaller_selection'] for f in fs), 'largest-component selection must fail'
    assert not set.intersection(*(f['corners'] for f in fs)), 'no fixed-corner selection'
    assert len({i['tile_shape'] for i in infos['train']}) >= 2
    assert len({i['rooms'].shape for i in infos['train']}) >= 2
    assert any(np.any(t == BG) for i in infos['train'] for t in i['tiles'].values())
    # An asymmetric displayed tile actually appears in a demonstration output.
    assert any(_asymmetric(t) for i in infos['train'] for key, t in i['tiles'].items()
               if np.any(_analysis(i)[1] & (i['rooms'] == key)))
    assert len({len(i['tiles']) for s in infos.values() for i in s}) == 1
    train_max = max(max(i['rooms'].shape) for i in infos['train'])
    assert all(max(i['rooms'].shape) > train_max for i in infos['test']), 'coarse-size extrapolation'


def _asymmetric(tile):
    return all(not np.array_equal(tile, t) for t in
               [np.rot90(tile, k) for k in (1, 2, 3)] + [np.fliplr(np.rot90(tile, k)) for k in range(4)])


def _regressions():
    # Literal macro geometry and literal tile roles; not generated examples.
    rooms = np.full((7, 7), 2, np.int8)
    rooms[3, 3] = BG
    rooms[-1] = 3
    sockets = {2: (4, 4, 4, 4), 3: (6, 4, 6, 4)}
    large = _reach(rooms, (2, 2), sockets)
    expected_selected = rooms == 2
    assert np.array_equal(large, expected_selected)
    rim = _rim(large)
    expected_rim = np.zeros(rooms.shape, bool)
    expected_rim[0] = True
    expected_rim[5] = True
    expected_rim[:6, 0] = expected_rim[:6, -1] = True
    assert np.array_equal(rim, expected_rim)
    assert _rim(large, include_holes=True)[2, 3] and not rim[2, 3]
    assert np.array_equal(_reach(rooms, (6, 1), sockets), rooms == 3)
    # A changed prototype socket opens the seam and changes other rooms' rim.
    opened = dict(sockets)
    opened[3] = sockets[2]
    assert np.array_equal(_reach(rooms, (6, 1), opened), rooms != BG)
    assert not np.array_equal(rim, _rim(_reach(rooms, (2, 2), opened)))
    # A concavity has a diagonal-only rim room.
    notch = np.ones((6, 6), bool)
    notch[:2, 2] = False
    assert _rim(notch)[2, 1] and not _rim(notch, diagonal=False)[2, 1]
    # Explicit non-reciprocal faces are legal. Geometric adjacency alone fails.
    line = np.array([[2, 3, 2]], np.int8)
    ports = {2: (4, 4, 4, 6), 3: (4, 6, 4, 4)}
    assert np.array_equal(_reach(line, (0, 0), ports), np.ones(line.shape, bool))
    ports[3] = (4, 4, 4, 4)
    assert np.array_equal(_reach(line, (0, 0), ports), np.array([[True, True, False]]))
    # A tile's isolated center ink still belongs to its solid-border crop.
    small = np.array([[2, 2, 3], [2, 2, 3], [2, 2, 3]], np.int8)
    tiles = {}
    for key, port in ((2, 4), (3, 6)):
        tile = np.full((5, 5), key, np.int8)
        tile[1:-1, 1:-1] = BG
        tile[2, 2] = key
        for position in ((0, 2), (2, 4), (4, 2), (2, 0)):
            tile[position] = port
        tiles[key] = tile
    inp = _display(np.random.default_rng(123), small, (1, 0), tiles)
    decoded = _decode(inp)
    assert all(np.array_equal(decoded['tiles'][k], t) for k, t in tiles.items())
    expected = np.concatenate((np.tile(tiles[2], (3, 2)), np.zeros((15, 5), np.int8)), axis=1)
    expected[7, 2] = SEED
    assert np.array_equal(solve(inp), expected)


def self_test():
    """Bounded semantic regressions, count coverage, and 42-seed diversity audit."""
    _regressions()
    fingerprints, map_shapes, tile_shapes, structures, role_palettes, selected_sizes = set(), set(), set(), set(), set(), set()
    socket_profiles, layouts, types, hole_totals = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        repeat = generate_puzzle(seed)
        assert all(_key(a[k]) == _key(b[k]) for s in puzzle for a, b in zip(puzzle[s], repeat[s]) for k in ('input', 'output'))
        fingerprints.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                f = _features(info)
                map_shapes.add(f['map_shape'])
                tile_shapes.add(f['tile_shape'])
                selected_sizes.add(f['selected'])
                types.add(f['types'])
                roles = {BG: 0}
                canonical = tuple(roles.setdefault(int(v), len(roles)) for v in info['rooms'].flat)
                structures.add((info['rooms'].shape, canonical))
                role_palettes.add(tuple(sorted(info['tiles'])))
                for key, ports in info['sockets'].items():
                    port_roles = {}
                    socket_profiles.add(tuple(port_roles.setdefault(v, len(port_roles)) for v in ports))
                hole_totals.update(int(np.count_nonzero(t == BG)) for t in info['tiles'].values())
                mb = info['frame_box']
                box = next(iter(info['tile_boxes'].values()))
                layouts.add('left' if box[3] <= mb[2] else 'right' if box[2] >= mb[3]
                            else 'above' if box[1] <= mb[0] else 'below')
                before = pair['input'].copy()
                assert np.array_equal(solve(before), pair['output']) and np.array_equal(before, pair['input'])
    assert len(fingerprints) == 42
    assert len(map_shapes) >= 25 and len(tile_shapes) >= 7
    assert len(structures) >= 140 and len(role_palettes) >= 15
    assert len(selected_sizes) >= 20 and types == {2, 3}
    assert len(socket_profiles) >= 2 and layouts == {'left', 'right', 'above', 'below'}
    assert 0 in hole_totals and max(hole_totals) >= 3
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(700+10*nt+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(0)
    out = damaged['test'][0]['output']
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    return {'seeds': 42, 'distinct_puzzles': len(fingerprints), 'map_shapes': len(map_shapes),
            'tile_shapes': len(tile_shapes), 'normalized_map_structures': len(structures),
            'key_palettes': len(role_palettes), 'selected_counts': len(selected_sizes),
            'dictionary_counts': sorted(types), 'layouts': sorted(layouts),
            'prototype_hole_counts': sorted(hole_totals)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
