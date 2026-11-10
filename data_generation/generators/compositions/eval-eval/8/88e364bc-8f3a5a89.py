"""Island shadows: topology chooses emitters and receivers of directed rays.

Only the standard library and NumPy are used. solve() reads its input alone.
Generation preferences and teaching witnesses are separate from the rule.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((a >= 0) & (a <= 9)), "color bounds")
    return a.astype(np.int8, copy=True)


def _components(mask):
    h, w = mask.shape
    seen = ~mask.copy()
    result = []
    for rr, cc in np.argwhere(mask):
        start = int(rr), int(cc)
        if seen[start]:
            continue
        seen[start] = True
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in N4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _mask(shape, cells):
    result = np.zeros(shape, dtype=bool)
    if cells:
        rr, cc = zip(*cells)
        result[rr, cc] = True
    return result


def _flood(allowed, start):
    h, w = allowed.shape
    found = np.zeros_like(allowed)
    if not allowed[start]:
        return found
    found[start] = True
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not found[nr, nc]:
                found[nr, nc] = True
                stack.append((nr, nc))
    return found


def _dilate(mask, diagonal=True):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = mask.copy()
    for dr, dc in N8 if diagonal else N4:
        result |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _decode(grid):
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) == 4, "four visible roles required")
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous background")
    background = int(values[counts.argmax()])
    keys = []
    for cells in _components(grid != background):
        r0, r1, c0, c1 = _bbox(cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 4 or w < 4 or len(cells) != h * w:
            continue
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        carrier = int(sub[0, 0])
        border = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
        others = set(map(int, np.unique(sub))) - {carrier}
        if len(others) != 2 or np.any(border != carrier):
            continue
        interpretations = []
        for tip in sorted(others):
            trail = next(iter(others - {tip}))
            tips = set(map(tuple, np.argwhere(sub == tip)))
            trails = set(map(tuple, np.argwhere(sub == trail)))
            if not 2 <= len(tips) <= 3:
                continue
            for dr, dc in N8:
                used, lengths = set(), []
                for tr, tc in sorted(tips):
                    r, c = tr - dr, tc - dc
                    stroke = []
                    while (r, c) in trails:
                        stroke.append((r, c))
                        r, c = r - dr, c - dc
                    if not 2 <= len(stroke) <= 4 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(tips) and used == trails:
                    interpretations.append((tip, trail, (dr, dc), tuple(lengths)))
        if len(interpretations) == 1:
            tip, trail, direction, lengths = interpretations[0]
            keys.append(dict(background=background, structure=carrier, marker=tip,
                             ink=trail, direction=direction, lengths=lengths,
                             key_bbox=(r0, r1, c0, c1)))
    _require(len(keys) == 1, "unique readable plaque required")
    info = keys[0]
    key_mask = np.zeros(grid.shape, dtype=bool)
    r0, r1, c0, c1 = info['key_bbox']
    key_mask[r0:r1 + 1, c0:c1 + 1] = True
    seeds = np.argwhere((grid == info['marker']) & ~key_mask)
    _require(len(seeds) == 1, "unique selecting marker required")
    _require(np.all(key_mask[grid == info['ink']]), "input ink only belongs in plaque")
    info.update(key_mask=key_mask, seed=tuple(map(int, seeds[0])))
    return info


def _topology(grid, info, seed=None):
    free = ((grid == info['background']) | (grid == info['marker'])) & ~info['key_mask']
    selected = _flood(free, info['seed'] if seed is None else seed)
    complement = np.pad(~selected, 1, constant_values=True)
    exterior = _flood(complement, (0, 0))
    filled = ~exterior[1:-1, 1:-1]
    sources = (grid == info['structure']) & filled & ~info['key_mask']
    rim = selected & _dilate(exterior, True)[1:-1, 1:-1]
    return dict(free=free, selected=selected, filled=filled, sources=sources,
                exterior=exterior, rim=rim)


def _trace(domain, start, direction, corner=True):
    """The origin may be outside domain for the false-emitter diagnostic."""
    h, w = domain.shape
    r, c = start
    dr, dc = direction
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or not domain[nr, nc]:
            return (r, c)
        if corner and dr and dc and (not domain[nr, c] or not domain[r, nc]):
            return (r, c)
        r, c = nr, nc


def _landings(domain, sources, direction, corner=True):
    return {_trace(domain, tuple(map(int, p)), direction, corner) for p in np.argwhere(sources)}


def _paint(grid, info, endpoints, selected=None):
    output = grid.copy()
    for cell in endpoints:
        if (selected is None or selected[cell]) and grid[cell] == info['background']:
            output[cell] = info['ink']
    return output


def _analyze(input_grid):
    grid = _grid(input_grid)
    info = _decode(grid)
    info.update(_topology(grid, info))
    endpoints = _landings(info['filled'], info['sources'], info['direction'])
    info.update(endpoints=endpoints, grid=grid)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the shadow marks using only roles and topology in this input."""
    info = _analyze(input_grid)
    return _paint(info['grid'], info, info['endpoints'])


def _alternative(info, kind, direction=None):
    """Explicit simpler rules used only to audit teaching evidence."""
    grid = info['grid']
    domain, sources = info['filled'], info['sources']
    selected = info['selected']
    corner = True
    if kind == 'all_rim':
        return _paint(grid, info, set(map(tuple, np.argwhere(info['rim']))))
    if kind == 'first_wall':
        domain = selected
    elif kind == 'no_corner':
        corner = False
    elif kind == 'box':
        r0, r1, c0, c1 = _bbox(list(map(tuple, np.argwhere(domain))))
        domain = np.zeros_like(domain)
        domain[r0:r1 + 1, c0:c1 + 1] = True
    elif kind == 'false_emitters':
        sources = (grid == info['structure']) & ~info['key_mask']
    elif kind == 'largest':
        rooms = []
        for cells in _components(info['free']):
            if not any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1) for r, c in cells):
                rooms.append(cells)
        largest = max(rooms, key=len)
        data = _topology(grid, info, largest[0])
        domain, sources, selected = data['filled'], data['sources'], data['selected']
    elif kind != 'direction':
        raise ValueError(kind)
    endpoints = _landings(domain, sources, direction or info['direction'], corner)
    return _paint(grid, info, endpoints, selected)


def _key(rng, direction, structure, ink, marker):
    count = int(rng.choice([2, 2, 2, 3]))
    length = int(rng.choice([2, 2, 3, 4]))
    dr, dc = direction
    points = []
    for i in range(count):
        stagger = int(rng.integers(0, 2))
        base = (0, 2 * i) if dr else (2 * i, 0)
        for j in range(length + 1):
            points.append((base[0] + (j + stagger) * dr,
                           base[1] + (j + stagger) * dc,
                           marker if j == length else ink))
    r0, r1, c0, c1 = _bbox([(r, c) for r, c, _ in points])
    padding = [int(rng.choice([1, 1, 2])) for _ in range(4)]
    key = np.full((r1 - r0 + 1 + padding[0] + padding[1],
                   c1 - c0 + 1 + padding[2] + padding[3]), structure, np.int8)
    for r, c, color in points:
        key[r - r0 + padding[0], c - c0 + padding[2]] = color
    return key


def _footprint(rng, h, w, kind):
    shape = np.ones((h, w), dtype=bool)
    if kind == 1:
        for corner in rng.choice(4, size=int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            rows = slice(0, dh) if corner < 2 else slice(h - dh, h)
            cols = slice(0, dw) if corner % 2 == 0 else slice(w - dw, w)
            shape[rows, cols] = False
    elif kind == 2:
        left, right = 0, 0
        for r in range(h):
            if r % int(rng.integers(2, 5)) == 0:
                left, right = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            shape[r, :left] = False
            if right:
                shape[r, w - right:] = False
    return shape


def _perimeter(footprint):
    h, w = footprint.shape
    p = np.pad(footprint, 1)
    inside = footprint.copy()
    for dr, dc in N4:
        inside &= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return footprint & ~inside


def _shape(rng, max_h=3, max_w=3):
    h, w = int(rng.integers(1, max_h + 1)), int(rng.integers(1, max_w + 1))
    shape = np.ones((h, w), dtype=bool)
    kind = int(rng.integers(5))
    if kind == 1:
        shape = np.ones((1, w), bool)
    elif kind == 2:
        shape = np.ones((h, 1), bool)
    elif kind == 3:
        shape[1:, 1:] = False
    elif kind == 4:
        for r in range(h):
            shape[r, int(rng.integers(1, w + 1)):] = False
    return np.rot90(shape, int(rng.integers(4))).copy()


def _insert_island(rng, wall, room, large=False):
    h, w = wall.shape
    for attempt in range(8):
        shape = _shape(rng, 4 if large else 3, 4 if large else 3) if attempt < 7 else np.ones((1, 1), bool)
        sh, sw = shape.shape
        halo = _dilate(np.pad(shape, 1))
        if halo.shape[0] > h or halo.shape[1] > w:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(room, halo.shape)
        candidates = np.argwhere(np.all(windows | ~halo, axis=(-1, -2)))
        if len(candidates):
            r, c = map(int, candidates[int(rng.integers(len(candidates)))])
            r, c = r + 1, c + 1
            wall[r:r + sh, c:c + sw] |= shape
            room[r:r + sh, c:c + sw] &= ~shape
            return True
    return False


def _teaching_terrain(rng, witness):
    h, w = int(rng.integers(15, 21)), int(rng.integers(22, 26))
    footprint = np.ones((h, w), bool)
    # Independently change the far silhouette, outside the witness rays.
    dh, dw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    footprint[h - dh:, w - dw:] = False
    wall = _perimeter(footprint)
    divider = int(rng.integers(9, 11)) if witness == 1 else int(rng.integers(6, 9))
    wall[:, divider] |= footprint[:, divider]
    if witness == 1:
        row = int(rng.integers(6, h - 5))
        wall[1:4, 2] = True  # Not an emitter: connected to the outer wall.
        wall[row:row + 2, divider - 1] = True  # Nonrectangular receiver.
        ha, wa = int(rng.integers(1, 4)), int(rng.integers(1, 3))
        hb = int(rng.integers(1, 4))
        wall[row:row + ha, 3:3 + wa] = True
        wall[row:row + hb, 6] = True  # An aligned downstream transparent island.
        wall[h - 5:h - 3, divider + 4:divider + 6] = True
        seed_hint, direction = (h - 3, 3), (0, 1)
    else:
        start = divider + 1
        wall[1:6, start + 6] = True  # A side-only block on the required beam.
        wall[3, start + 3] = True
        if rng.random() < .5:
            wall[4, start + 3] = True
        shape = _shape(rng, 3, 3)
        sh, sw = shape.shape
        wall[9:9 + sh, start + 3:start + 3 + sw] |= shape
        wall[h - 5:h - 3, 2:4] = True
        seed_hint, direction = (h - 3, w - 4), (1, 1)
    room = _flood(footprint & ~wall, seed_hint)
    return wall, footprint, room, direction


def _random_terrain(rng, test):
    h, w = int(rng.integers(12, 23)), int(rng.integers(13, 24))
    footprint = _footprint(rng, h, w, int(rng.integers(3)))
    wall = _perimeter(footprint)
    # Recursive full cuts subdivide actual free components, permitting branches.
    requested = int(rng.integers(1 if test else 0, 3))
    for _ in range(requested):
        rooms = [_mask(wall.shape, cells) for cells in _components(footprint & ~wall)]
        candidates = []
        for room in rooms:
            if int(room.sum()) < 70:
                continue
            r0, r1, c0, c1 = _bbox(list(map(tuple, np.argwhere(room))))
            for axis, low, high in ((0, r0, r1), (1, c0, c1)):
                if high - low >= 8:
                    for cut in range(low + 3, high - 2):
                        candidates.append((room, axis, cut))
        if not candidates:
            break
        room, axis, cut = candidates[int(rng.integers(len(candidates)))]
        line = np.zeros_like(wall)
        if axis == 0:
            line[cut, :] = room[cut, :]
        else:
            line[:, cut] = room[:, cut]
        wall |= line
    # Partial tongues extend from the nearest enclosing wall on a chosen shore.
    for _ in range(int(rng.integers(0, 4))):
        side = int(rng.integers(4))
        if side < 2:
            c = int(rng.integers(3, w - 3))
            rr = np.flatnonzero(footprint[:, c])
            if len(rr):
                length = int(rng.integers(1, 5))
                rows = slice(int(rr[0]), int(rr[0]) + length) if side == 0 else slice(int(rr[-1]) - length + 1, int(rr[-1]) + 1)
                wall[rows, c] |= footprint[rows, c]
        else:
            r = int(rng.integers(3, h - 3))
            cc = np.flatnonzero(footprint[r])
            if len(cc):
                length = int(rng.integers(1, 5))
                cols = slice(int(cc[0]), int(cc[0]) + length) if side == 2 else slice(int(cc[-1]) - length + 1, int(cc[-1]) + 1)
                wall[r, cols] |= footprint[r, cols]
    rooms = [_mask(wall.shape, cells) for cells in _components(footprint & ~wall)]
    candidates = [room for room in rooms if int(room.sum()) >= 35]
    if not candidates or (test and len(rooms) < 2):
        return None
    room = candidates[int(rng.integers(len(candidates)))].copy()
    n = int(rng.integers(2 if test else 1, 5 if test else 4))
    placed = sum(_insert_island(rng, wall, room, test) for _ in range(n))
    if placed < (2 if test else 1):
        return None
    others = [other.copy() for other in rooms if not np.any(other & room)]
    for _ in range(int(rng.integers(0, 4))):
        if others:
            _insert_island(rng, wall, others[int(rng.integers(len(others)))])
    direction = (0, 1) if rng.random() < .5 else (1, 1)
    return wall, footprint, room, direction


def _pack(rng, wall, footprint, room, direction, palette):
    background, structure, ink, marker = palette
    key = _key(rng, direction, structure, ink, marker)
    h, w = wall.shape
    kh, kw = key.shape
    possibilities = []
    if h + kh + 1 <= 30 and max(w, kw) <= 30:
        possibilities.append((h + kh + 1, max(w, kw)))
    if w + kw + 1 <= 30 and max(h, kh) <= 30:
        possibilities.append((max(h, kh), w + kw + 1))
    if not possibilities:
        return None
    base_h, base_w = possibilities[int(rng.integers(len(possibilities)))]
    height = base_h + int(rng.integers(0, min(3, 31 - base_h)))
    width = base_w + int(rng.integers(0, min(3, 31 - base_w)))
    r, c = int(rng.integers(height - h + 1)), int(rng.integers(width - w + 1))
    occupied = np.zeros((height, width), bool)
    occupied[r:r + h, c:c + w] = footprint
    windows = np.lib.stride_tricks.sliding_window_view(_dilate(occupied), key.shape)
    locations = np.argwhere(~windows.any(axis=(-1, -2)))
    if not len(locations):
        return None
    kr, kc = map(int, locations[int(rng.integers(len(locations)))])
    grid = np.full((height, width), background, np.int8)
    grid[r:r + h, c:c + w][wall] = structure
    grid[kr:kr + kh, kc:kc + kw] = key
    possible = np.argwhere(room)
    if not len(possible):
        return None
    sr, sc = map(int, possible[int(rng.integers(len(possible)))])
    grid[r + sr, c + sc] = marker
    return grid


def _transform(grid, rotation, reflect):
    result = np.rot90(grid, rotation)
    return np.fliplr(result).copy() if reflect else result.copy()


def _choose_transform(rng, direction, wanted_direction=None):
    transforms = []
    for rotation in range(4):
        dr, dc = direction
        for _ in range(rotation):
            dr, dc = -dc, dr
        for reflect in (False, True):
            transformed = (dr, -dc if reflect else dc)
            if wanted_direction is None or transformed == wanted_direction:
                transforms.append((rotation, reflect))
    return transforms[int(rng.integers(len(transforms)))]


def _guaranteed_pair(rng, palette, witness, wanted_direction, seen):
    """A fitting same-rule scene, with exhaustive fresh marker placement.

    This is the termination tail, not a special-case solving rule. There are
    more legal marker positions here than the maximum thirteen earlier inputs.
    """
    background, structure, ink, marker = palette
    h, w = 16, 24
    wall = _perimeter(np.ones((h, w), bool))
    divider = 9 if witness == 1 else 6
    wall[:, divider] = True
    if witness == 1:
        wall[1:4, 2] = True
        wall[7:9, 8] = True
        wall[7:9, 3:5] = True
        wall[7:9, 6] = True
        wall[11:13, 14:16] = True
        seed = (12, 3)
        direction = (0, 1)
    else:
        wall[1:6, 13] = True
        wall[3, 10] = True
        wall[9:11, 10:12] = True
        wall[11:13, 2:4] = True
        seed = (12, 19)
        direction = (1, 1)
    if wanted_direction is not None:
        direction = (0, 1) if 0 in wanted_direction else (1, 1)
    # Compact two-stroke key with two trails per stroke.
    dr, dc = direction
    points = [((0 if dr else 2 * i) + j * dr,
               (2 * i if dr else 0) + j * dc, marker if j == 2 else ink)
              for i in range(2) for j in range(3)]
    r0, r1, c0, c1 = _bbox([(r, c) for r, c, _ in points])
    key = np.full((r1 - r0 + 3, c1 - c0 + 3), structure, np.int8)
    for r, c, color in points:
        key[r - r0 + 1, c - c0 + 1] = color
    grid = np.full((22, 24), background, np.int8)
    grid[6:22][wall] = structure
    kc = int(rng.integers(24 - key.shape[1] + 1))
    grid[:key.shape[0], kc:kc + key.shape[1]] = key
    grid[seed[0] + 6, seed[1]] = marker
    info = _analyze(grid)
    positions = [tuple(map(int, p)) for p in np.argwhere(info['selected'])
                 if tuple(map(int, p)) not in info['endpoints']]
    _require(len(positions) > 13, 'fallback placement capacity')
    rotation, reflect = _choose_transform(rng, direction, wanted_direction)
    for i in rng.permutation(len(positions)):
        trial = grid.copy()
        trial[info['seed']] = background
        trial[positions[int(i)]] = marker
        trial = _transform(trial, rotation, reflect)
        if _fingerprint(trial) not in seen:
            return {'input': trial, 'output': solve(trial)}
    raise AssertionError('exhaustive fallback capacity unexpectedly exhausted')


def _make_pair(rng, palette, witness, test, wanted_direction=None, seen=frozenset()):
    for attempt in range(160):
        terrain = _teaching_terrain(rng, witness) if witness else _random_terrain(rng, test)
        if terrain is None:
            continue
        wall, footprint, room, direction = terrain
        if wanted_direction is not None:
            direction = (0, 1) if 0 in wanted_direction else (1, 1)
        grid = _pack(rng, wall, footprint, room, direction, palette)
        if grid is None:
            continue
        rotation, reflect = _choose_transform(rng, direction, wanted_direction)
        grid = _transform(grid, rotation, reflect)
        try:
            info = _analyze(grid)
        except AssertionError:
            continue
        if wanted_direction is not None and info['direction'] != wanted_direction:
            continue
        if not info['endpoints'] or any(not info['selected'][p] for p in info['endpoints']):
            continue
        if info['seed'] in info['endpoints']:
            continue
        output = _paint(grid, info, info['endpoints'])
        if np.array_equal(grid, output):
            continue
        # Require the rendered witnesses, not just the intended construction.
        if witness == 1 and any(np.array_equal(output, _alternative(info, kind))
                                for kind in ('first_wall', 'box', 'largest', 'false_emitters', 'all_rim')):
            continue
        if witness == 2 and any(np.array_equal(output, _alternative(info, kind))
                                for kind in ('no_corner', 'false_emitters', 'all_rim')):
            continue
        return {'input': grid, 'output': output}
    return _guaranteed_pair(rng, palette, witness, wanted_direction, seen)


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_fingerprint(puzzle):
    return tuple((_fingerprint(p['input']), _fingerprint(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    changing_palette = bool(rng.integers(2))
    heldout = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            local_palette = tuple(map(int, rng.choice(10, 4, replace=False))) if changing_palette else palette
            witness = index + 1 if split == 'train' and index < 2 else 0
            wanted = None
            if split == 'test' and heldout:
                known = {_decode(p['input'])['direction'] for p in puzzle['train']}
                choices = [d for d in N8 if d not in known]
                if choices:
                    wanted = choices[int(rng.integers(len(choices)))]
            for _ in range(12):
                pair = _make_pair(rng, local_palette, witness, split == 'test', wanted, seen)
                fp = _fingerprint(pair['input'])
                if fp not in seen:
                    break
            else:
                pair = _guaranteed_pair(rng, local_palette, witness, wanted, seen)
                fp = _fingerprint(pair['input'])
            seen.add(fp)
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    grid = info['grid']
    wall = (grid == info['structure']) & ~info['key_mask']
    islands = _components(info['sources'])
    rooms = [cells for cells in _components(info['free'])
             if not any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1) for r, c in cells)]
    dormant_mask = np.zeros_like(wall)
    for cells in rooms:
        if not info['selected'][cells[0]]:
            dormant_mask |= _topology(grid, info, cells[0])['sources']
    dormant = len(_components(dormant_mask))
    sizes = [len(cells) for cells in rooms]
    area = int(info['selected'].sum())
    rr, cc = np.where(wall)
    box = wall[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    irregular = not (box[0].all() and box[-1].all() and box[:, 0].all() and box[:, -1].all())
    r0, r1, c0, c1 = info['key_bbox']
    kr, kc = (r0 + r1) / 2, (c0 + c1) / 2
    wr, wc = (rr.min() + rr.max()) / 2, (cc.min() + cc.max()) / 2
    side = ('bottom' if kr > wr else 'top') if abs(kr - wr) >= abs(kc - wc) else ('right' if kc > wc else 'left')
    shapes = []
    for cells in islands:
        ir0, ir1, ic0, ic1 = _bbox(cells)
        normalized = tuple(sorted((r - ir0, c - ic0) for r, c in cells))
        shapes.append(normalized)
    return dict(shape=grid.shape, roles=tuple(info[k] for k in ('background', 'structure', 'ink', 'marker')),
                direction=info['direction'], rooms=len(rooms), islands=len(islands),
                island_areas=[len(cells) for cells in islands], island_shapes=shapes, dormant=dormant,
                selected='small' if area < max(sizes) else 'large', key_shape=(r1 - r0 + 1, c1 - c0 + 1),
                strokes=len(info['lengths']), lengths=info['lengths'], key_side=side,
                marks=len(info['endpoints']), merging=int(info['sources'].sum()) > len(info['endpoints']),
                irregular=irregular, marker_on_rim=bool(info['rim'][info['seed']]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'dimension preservation')
            fp = _fingerprint(a)
            _require(fp not in seen, 'duplicate input')
            seen.add(fp)
            info = _analyze(a)
            terrain = (a == info['structure']) & ~info['key_mask']
            _require(not np.any(_dilate(info['key_mask']) & terrain), 'plaque needs a background gap')
            _require(info['sources'].any(), 'some emitting islands required')
            _require(not info['selected'][0].any() and not info['selected'][-1].any()
                     and not info['selected'][:, 0].any() and not info['selected'][:, -1].any(), 'enclosed selection')
            _require(np.array_equal(info['filled'], info['selected'] | info['sources']), 'solid holes only')
            _require(not np.any(_dilate(info['sources']) & ~info['filled']), 'isolated island clearance')
            _require(all(info['selected'][p] and info['rim'][p] for p in info['endpoints']), 'external rim landings')
            expected = solve(a)
            _require(np.array_equal(expected, b), 'input-only rule mismatch')
            _require(not np.array_equal(a, b), 'observable shadow required')
            changed = a != b
            _require(np.all((a[changed] == info['background']) & (b[changed] == info['ink'])), 'ink additions only')
            _require(np.all(info['rim'][changed]), 'ink must lie on external rim')
            _require(b[info['seed']] == info['marker'], 'marker preserved')
            _require(np.array_equal(a[info['key_mask']], b[info['key_mask']]), 'plaque preserved')
            _require(np.array_equal(a[a == info['structure']], b[a == info['structure']]), 'all terrain preserved')
            if split == 'train':
                training.append((info, b))
    directions = {info['direction'] for info, _ in training}
    _require(any(0 in d for d in directions) and any(0 not in d for d in directions), 'cardinal and diagonal evidence')
    for kind in ('all_rim', 'first_wall', 'no_corner', 'box', 'false_emitters', 'largest'):
        _require(any(not np.array_equal(out, _alternative(info, kind)) for info, out in training),
                 f'missing teaching evidence against {kind}')
    compatible = set(N8)
    for info, out in training:
        compatible &= {d for d in N8 if np.array_equal(out, _alternative(info, 'direction', d))}
    _require(not compatible, 'constant-direction shortcut survives training')
    features = [_features(info) for info, _ in training]
    _require(any(f['selected'] == 'small' for f in features)
             and any(f['selected'] == 'large' for f in features), 'both relative room sizes taught')


def self_test():
    """Bounded authored audit with 42 seeds and family-specific counterfactuals."""
    puzzles, records = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        fs = [_features(_analyze(p['input'])) for s in ('train', 'test') for p in puzzle[s]]
        records.extend(fs)
        _require(len({f['direction'] for f in fs}) >= 2, 'within-puzzle directions')
        _require(len({f['shape'] for f in fs}) >= 2, 'within-puzzle dimensions')
        _require(len({f['island_shapes'].__repr__() for f in fs}) >= 2, 'within-puzzle island geometry')
    _require(len({_puzzle_fingerprint(p) for p in puzzles}) == 42, 'distinct seed puzzles')
    _require({f['direction'] for f in records} == set(N8), 'eight direction coverage')
    _require(len({f['shape'] for f in records}) >= 50, 'dimension diversity')
    _require(len({f['key_shape'] for f in records}) >= 15, 'plaque geometry diversity')
    _require({f['key_side'] for f in records} == {'top', 'bottom', 'left', 'right'}, 'plaque placement diversity')
    _require({f['strokes'] for f in records} == {2, 3}, 'stroke-count variation')
    _require({n for f in records for n in f['lengths']} == {2, 3, 4}, 'trail-length variation')
    _require({1, 2, 3}.issubset({f['islands'] for f in records}), 'emitter-count variation')
    _require({1, 2, 3}.issubset({f['rooms'] for f in records}), 'room topology variation')
    _require(len({shape for f in records for shape in f['island_shapes']}) >= 15, 'island shape variation')
    _require(any(f['merging'] for f in records), 'projection union evidence')
    _require(any(f['irregular'] for f in records) and any(not f['irregular'] for f in records), 'outer silhouette variation')
    _require(all(len({f['roles'][i] for f in records}) == 10 for i in range(4)), 'role-color diversity')
    # Exact small-scene answers, authored directly rather than from ray metadata.
    fixture = np.zeros((23, 25), dtype=np.int8)
    fixture[7, 1:24] = fixture[21, 1:24] = 1
    fixture[7:22, 1] = fixture[7:22, 23] = fixture[7:22, 10] = 1
    fixture[12:14, 4:6] = fixture[12:14, 7] = 1
    fixture[16:18, 15:17] = 1
    key = np.ones((5, 5), np.int8)
    key[1, 1:3] = key[3, 1:3] = 2
    key[1, 3] = key[3, 3] = 3
    fixture[:5, 18:23] = key
    fixture[18, 3] = 3
    expected = fixture.copy()
    expected[12:14, 9] = 2
    _require(np.array_equal(solve(fixture), expected), 'explicit transparent-island projection')
    reversed_key = fixture.copy()
    reversed_key[:5, 18:23] = np.fliplr(key)
    reversed_expected = reversed_key.copy()
    reversed_expected[12:14, 2] = 2
    _require(np.array_equal(solve(reversed_key), reversed_expected), 'key reversal changes the receiving side')
    extra_dormant = fixture.copy()
    extra_dormant[9:11, 16:18] = 1
    dormant_expected = expected.copy()
    dormant_expected[9:11, 16:18] = 1
    _require(np.array_equal(solve(extra_dormant), dormant_expected), 'dormant islands do not cast')
    moved = fixture.copy()
    moved[18, 3], moved[9, 3] = 0, 3
    moved_expected = expected.copy()
    moved_expected[18, 3], moved_expected[9, 3] = 0, 3
    _require(np.array_equal(solve(moved), moved_expected), 'seed movement within a room')
    across = fixture.copy()
    across[18, 3], across[18, 14] = 0, 3
    across_expected = across.copy()
    across_expected[16:18, 22] = 2
    _require(np.array_equal(solve(across), across_expected), 'seed movement across a divider')
    overlap = fixture.copy()
    overlap[18, 3], overlap[12, 9] = 0, 3
    overlap_expected = overlap.copy()
    overlap_expected[13, 9] = 2
    _require(np.array_equal(solve(overlap), overlap_expected), 'marker priority at a landing')
    # Remove the redundant downstream island, then attach the remaining emitter
    # to the exterior: it becomes a tongue and no longer emits.
    attached = fixture.copy()
    attached[12:14, 7] = 0
    attached[12, 2:4] = 1
    _require(np.array_equal(solve(attached), attached), 'attachment changes source topology')
    # The constructive tail supports every compass vector and enough distinct
    # marker positions even if all previous inputs also came from that tail.
    fallback_rng = np.random.default_rng(7788)
    fallback, tail_seen = {'train': [], 'test': []}, set()
    for witness in (1, 2, 0):
        pair = _guaranteed_pair(fallback_rng, (0, 1, 2, 3), witness, None, tail_seen)
        tail_seen.add(_fingerprint(pair['input']))
        fallback['train' if witness else 'test'].append(pair)
    validate_generated_puzzle(fallback)
    for direction in N8:
        pair = _guaranteed_pair(fallback_rng, (0, 1, 2, 3), 0, direction, tail_seen)
        _require(_decode(pair['input'])['direction'] == direction, 'fallback compass support')
        tail_seen.add(_fingerprint(pair['input']))
    for _ in range(14):
        pair = _guaranteed_pair(fallback_rng, (0, 1, 2, 3), 1, None, tail_seen)
        _require(_fingerprint(pair['input']) not in tail_seen, 'fallback freshness')
        tail_seen.add(_fingerprint(pair['input']))
    fixed = generate_puzzle(9001, 7, 3)
    _require(_puzzle_fingerprint(fixed) == _puzzle_fingerprint(generate_puzzle(9001, 7, 3)), 'determinism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345 + 10 * nt + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'legal count contract')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Equivariance and nonmutation checks work with inputs, not generation state.
    palette_permutation = np.array([3, 6, 8, 2, 0, 9, 1, 7, 5, 4], dtype=np.int8)
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        before = pair['input'].copy()
        for rot in range(4):
            for reflection in (False, True):
                a = _transform(before, rot, reflection)
                b = _transform(pair['output'], rot, reflection)
                _require(np.array_equal(solve(a), b), 'rotation/reflection equivariance')
        _require(np.array_equal(solve(palette_permutation[before]), palette_permutation[pair['output']]), 'palette equivariance')
        solve(before)
        _require(np.array_equal(before, pair['input']), 'solver mutated input')
    bad = copy.deepcopy(puzzles[0])
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'pairs': len(records), 'dimensions': len({f['shape'] for f in records}),
            'directions': sorted({f['direction'] for f in records}),
            'rooms': dict(sorted(Counter(f['rooms'] for f in records).items())),
            'islands': dict(sorted(Counter(f['islands'] for f in records).items())),
            'island_shapes': len({shape for f in records for shape in f['island_shapes']}),
            'plaque_shapes': len({f['key_shape'] for f in records}),
            'marker_moves_attachment_priority': 'passed', 'constructive_fallback': 'passed',
            'semantic_verification': 'authored invariants and counterfactuals only'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
