"""Cavity weaving: topology selects a texture and also supplies its placement mask.

All semantic information is visible. The sampler's curriculum is never used by
solve. No parent code, parent fixtures, files, or hidden parameters are imported.
"""
from functools import lru_cache
from itertools import combinations

import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = _N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in (_N8 if diagonal else _N4):
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(cells)
    return result


def _box(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))


def _cavities(wall, diagonal=False):
    """Background regions touching any crop side are exterior, not chambers."""
    h, w = wall.shape
    interior = np.zeros(wall.shape, dtype=bool)
    regions = []
    for cells in _components(~wall, diagonal):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        regions.append(cells)
        for point in cells:
            interior[point] = True
    return interior, regions


def _touches_sides(mask):
    return all(np.any(x) for x in (mask[0], mask[-1], mask[:, 0], mask[:, -1]))


def _transforms(a):
    return [np.rot90(a, k) for k in range(4)] + [np.rot90(a.T, k) for k in range(4)]


def _asymmetric(a):
    return all(not np.array_equal(a, other) for other in _transforms(a)[1:])


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('grids must be integer rectangles of colors 0..9, dimensions 1..30')
    return a.astype(np.int8, copy=True)


def _read(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not np.all(_perimeter(a) == bg):
        raise ValueError('the canvas must have a uniform background rim')
    shells, cards, boxes = [], {}, []
    for cells in _components(a != bg):
        r0, r1, c0, c1 = _box(cells)
        crop = a[r0:r1, c0:c1]
        colors = set(map(int, crop[crop != bg]))
        boxes.append((r0, r1, c0, c1))
        if len(colors) == 1:
            wall = crop != bg
            inside, regions = _cavities(wall)
            shells.append({'r': r0, 'c': c0, 'wall': wall,
                           'carrier': next(iter(colors)), 'inside': inside,
                           'regions': regions, 'key': len(regions)})
        else:
            if not np.all(_perimeter(crop) != bg):
                raise ValueError('stamp cards have a complete non-background perimeter')
            holes = crop == bg
            if np.any(holes[:-1] & holes[1:]) or np.any(holes[:, :-1] & holes[:, 1:]):
                raise ValueError('stamp key cells must be four-neighbor isolated')
            key = int(holes.sum())
            if key in cards:
                raise ValueError('stamp keys must be unique')
            cards[key] = {'stamp': crop.copy(), 'r': r0, 'c': c0}
    if not 1 <= len(shells) <= 3 or not 2 <= len(cards) <= 4:
        raise ValueError('expected one to three shells and two to four stamp cards')
    for i, (r0, r1, c0, c1) in enumerate(boxes):
        for s0, s1, d0, d1 in boxes[i + 1:]:
            if not (r1 < s0 or s1 < r0 or c1 < d0 or d1 < c0):
                raise ValueError('object boxes require at least one blank separating row or column')
    shapes = {v['stamp'].shape for v in cards.values()}
    if len(shapes) != 1 or any(s['key'] not in cards for s in shells):
        raise ValueError('cards must share dimensions and supply exactly one match per shell')
    r0 = min(s['r'] for s in shells)
    c0 = min(s['c'] for s in shells)
    r1 = max(s['r'] + s['wall'].shape[0] for s in shells)
    c1 = max(s['c'] + s['wall'].shape[1] for s in shells)
    sh, sw = next(iter(shapes))
    if (r1 - r0) * sh > 30 or (c1 - c0) * sw > 30:
        raise ValueError('expanded shell scene exceeds the ARC size bound')
    # Cards are separate sources, not furniture inside the output scene.
    for card in cards.values():
        r, c = card['r'], card['c']
        if not (r + sh < r0 or r1 < r or c + sw < c0 or c1 < c):
            raise ValueError('cards must be outside the collective shell bounding box')
    return {'bg': bg, 'shells': shells, 'cards': cards,
            'bounds': (r0, r1, c0, c1), 'scale': (sh, sw)}


def _render(info, selection='count', painting='stamp'):
    """Block-write renderer; optional ablations are used only in validation."""
    bg, cards = info['bg'], info['cards']
    r0, r1, c0, c1 = info['bounds']
    sh, sw = info['scale']
    out = np.full(((r1 - r0) * sh, (c1 - c0) * sw), bg, dtype=np.int8)
    first = min(cards, key=lambda k: (cards[k]['r'], cards[k]['c']))
    for shell in info['shells']:
        wall, inside = shell['wall'], shell['inside']
        key = shell['key']
        if selection == 'first':
            key = first
        elif selection == 'area':
            key = min(cards, key=lambda k: (abs(k - int(inside.sum())), k))
        elif selection == 'eight':
            key = len(_cavities(wall, diagonal=True)[1])
            if key not in cards:
                return None
        stamp = cards[key]['stamp']
        texture = np.where(stamp == bg, shell['carrier'], stamp)
        for r, c in np.argwhere(wall | inside):
            rr = (shell['r'] - r0 + int(r)) * sh
            cc = (shell['c'] - c0 + int(c)) * sw
            if wall[r, c]:
                out[rr:rr + sh, cc:cc + sw] = shell['carrier']
            elif painting == 'solid':
                out[rr:rr + sh, cc:cc + sw] = int(stamp[0, 0])
            else:
                out[rr:rr + sh, cc:cc + sw] = texture
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only the visible input and apply the fixed cavity-weaving rule."""
    return _render(_read(input_grid))


@lru_cache(maxsize=None)
def _fallback_holes(h, w, k):
    """A small deterministic feasibility search, never a bank of example pairs."""
    if k == 0:
        return np.zeros((h, w), dtype=bool)
    for parity in (0, 1):
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
                 if (r + c) % 2 == parity]
        # k <= 5; shapes used here have at most a 4x4 interior for cards.
        # For larger shells a spaced grid alone already supplies five cells.
        if len(cells) > 16:
            cells = [(r, c) for r in range(1, h - 1, 2) for c in range(1, w - 1, 2)]
        for chosen in combinations(cells, k):
            holes = np.zeros((h, w), dtype=bool)
            for point in chosen:
                holes[point] = True
            if len(_components(~holes)) == 1:
                return holes
    return None


def _isolated_holes(rng, shape, k):
    h, w = shape
    if k == 0:
        return np.zeros(shape, dtype=bool)
    for _ in range(12):
        holes = np.zeros(shape, dtype=bool)
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        for index in rng.permutation(len(cells)):
            point = cells[int(index)]
            r, c = point
            if any(holes[r + dr, c + dc] for dr, dc in _N4):
                continue
            holes[point] = True
            if len(_components(~holes)) != 1:
                holes[point] = False
            if int(holes.sum()) == k:
                return holes
    fallback = _fallback_holes(h, w, k)
    if fallback is None:
        raise ValueError('infeasible isolated-hole count')
    return fallback.copy()


def _wall(rng, shape, k, kind='mixed', bays=0):
    h, w = shape
    if kind == 'ring' and k == 1:
        holes = np.zeros(shape, dtype=bool)
        holes[1:-1, 1:-1] = True
    elif kind == 'diagonal' and k == 2:
        holes = np.zeros(shape, dtype=bool)
        holes[1, 1] = holes[2, 2] = holes[2, 3] = True
    else:
        holes = _isolated_holes(rng, shape, k)
        if k and rng.random() < .75:
            target = int(rng.integers(k, (h - 2) * (w - 2) + 1))
            for index in rng.permutation((h - 2) * (w - 2)):
                r, c = divmod(int(index), w - 2)
                r, c = r + 1, c + 1
                if holes[r, c] or not any(holes[r + dr, c + dc] for dr, dc in _N4):
                    continue
                trial = holes.copy()
                trial[r, c] = True
                if len(_components(trial)) == k and len(_components(~trial)) == 1:
                    holes = trial
                if int(holes.sum()) >= target:
                    break
    wall = ~holes
    if kind == 'diagonal':
        wall[0, 0] = False  # Diagonal contact with outside is not a leak.
    else:
        for _ in range(bays):
            exterior = ~wall & ~_cavities(wall)[0]
            candidates = [(r, c) for r in range(h) for c in range(w) if wall[r, c]
                          and (r in (0, h - 1) or c in (0, w - 1)
                               or any(0 <= r + dr < h and 0 <= c + dc < w
                                      and exterior[r + dr, c + dc] for dr, dc in _N4))]
            rng.shuffle(candidates)
            for point in candidates:
                trial = wall.copy()
                trial[point] = False
                if (_touches_sides(trial) and len(_components(trial)) == 1
                        and len(_cavities(trial)[1]) == k):
                    wall = trial
                    break
            else:
                break
    if rng.random() < .5:
        wall = np.flipud(wall)
    if rng.random() < .5:
        wall = np.fliplr(wall)
    return wall.copy()


def _card(rng, shape, k, bg, payload, directional=False):
    holes = _isolated_holes(rng, shape, k)
    rr, cc = np.indices(shape)
    for _ in range(48):
        mode = int(rng.integers(5))
        if directional or mode == 0:
            indices = rng.integers(len(payload), size=shape)
        elif mode == 1:
            indices = rr + int(rng.integers(1, 3)) * cc
        elif mode == 2:
            indices = rr if rng.random() < .5 else cc
        elif mode == 3:
            indices = np.minimum.reduce((rr, cc, shape[0] - 1 - rr, shape[1] - 1 - cc))
        else:
            indices = abs(2 * rr - shape[0] + 1) + abs(2 * cc - shape[1] + 1)
        stamp = np.asarray(payload, dtype=np.int8)[indices % len(payload)]
        stamp[holes] = bg
        if len(set(map(int, stamp[~holes]))) >= 2 and (not directional or _asymmetric(stamp)):
            return stamp
    # The top-left marked payload breaks every symmetry while the perimeter
    # remains complete. Key cells are never overwritten.
    stamp = np.full(shape, payload[0], dtype=np.int8)
    stamp[0, 1] = payload[1]
    stamp[-1, 0] = payload[-1]
    stamp[holes] = bg
    assert not directional or _asymmetric(stamp)
    return stamp


def _free_origins(height, width, h, w, boxes, rim=0):
    free = []
    for r in range(rim, height - h - rim + 1):
        for c in range(rim, width - w - rim + 1):
            if all(r + h < a or a + bh < r or c + w < b or b + bw < c
                   for a, b, bh, bw in boxes):
                free.append((r, c))
    return free


def _shell_record(wall, carrier, r, c):
    inside, regions = _cavities(wall)
    return {'r': r, 'c': c, 'wall': wall, 'carrier': int(carrier),
            'inside': inside, 'regions': regions, 'key': len(regions)}


def _place(rng, bg, objects, cards, scale, witness=False):
    low_r = min(x['r'] for x in objects)
    low_c = min(x['c'] for x in objects)
    gh = max(x['r'] + x['wall'].shape[0] for x in objects) - low_r
    gw = max(x['c'] + x['wall'].shape[1] for x in objects) - low_c
    # Pose the entire coarse scene before displaying it. Each posed wall is
    # subsequently copied as seen; the separately displayed cards are unchanged.
    flip_r, flip_c = bool(rng.integers(2)), bool(rng.integers(2))
    posed = []
    for obj in objects:
        wall = obj['wall']
        h, w = wall.shape
        r = low_r + gh - (obj['r'] - low_r) - h if flip_r else obj['r']
        c = low_c + gw - (obj['c'] - low_c) - w if flip_c else obj['c']
        if flip_r:
            wall = np.flipud(wall)
        if flip_c:
            wall = np.fliplr(wall)
        posed.append(_shell_record(wall.copy(), obj['carrier'], r, c))
    objects = posed
    # Group translation preserves the relative shell arrangement. Cards have
    # independent positions outside that entire reserved group rectangle.
    for attempt in range(32):
        gr = int(rng.integers(1, 30 - gh)) if attempt < 31 else 1
        gc = int(rng.integers(1, 30 - gw)) if attempt < 31 else 1
        boxes = [(gr, gc, gh, gw)]
        card_boxes = []
        sh, sw = scale
        for _ in cards:
            free = _free_origins(30, 30, sh, sw, boxes, rim=1)
            if not free:
                break
            r, c = free[int(rng.integers(len(free)))] if attempt < 31 else free[0]
            boxes.append((r, c, sh, sw))
            card_boxes.append((r, c))
        if len(card_boxes) == len(cards):
            break
    else:
        raise RuntimeError('bounded scene packing failed')
    order = list(map(int, rng.permutation(list(cards))))
    if witness:
        # Explicitly refute the spatial-first-card shortcut in a demonstration.
        first = min(range(len(card_boxes)), key=lambda i: card_boxes[i])
        match = order.index(objects[0]['key'])
        if match == first:
            other = (first + 1) % len(order)
            order[first], order[other] = order[other], order[first]
    canvas = np.full((30, 30), bg, dtype=np.int8)
    positioned = []
    for obj in objects:
        r, c = gr + obj['r'] - low_r, gc + obj['c'] - low_c
        wall = obj['wall']
        h, w = wall.shape
        canvas[r:r + h, c:c + w] = np.where(wall, obj['carrier'], bg)
        positioned.append(_shell_record(wall, obj['carrier'], r, c))
    positioned_cards = {}
    for key, (r, c) in zip(order, card_boxes):
        canvas[r:r + sh, c:c + sw] = cards[key]
        positioned_cards[key] = {'stamp': cards[key], 'r': r, 'c': c}
    info = {'bg': bg, 'shells': positioned, 'cards': positioned_cards,
            'bounds': (gr, gr + gh, gc, gc + gw), 'scale': scale}
    output = _render(info)
    # Crop only background margins; retain at least one full background rim.
    all_boxes = boxes
    a = min(x[0] for x in all_boxes)
    b = min(x[1] for x in all_boxes)
    d = max(x[0] + x[2] for x in all_boxes)
    e = max(x[1] + x[3] for x in all_boxes)
    top = int(rng.integers(a))
    left = int(rng.integers(b))
    bottom = int(rng.integers(d + 1, 31))
    right = int(rng.integers(e + 1, 31))
    return {'input': canvas[top:bottom, left:right].copy(), 'output': output}


_STAMPS = tuple((h, w) for h in range(3, 7) for w in range(3, 7))


def _sample_pair(rng, regime, split, index, palette, stable):
    witness = split == 'train' and index < 2
    colors = palette if stable else list(map(int, rng.permutation(10)))
    bg, carrier = colors[:2]
    payload = colors[2:2 + int(rng.integers(2, 6))]
    if rng.random() < .3:
        payload = [carrier] + payload  # Literal color reuse does not change roles.
    if witness:
        scale = ((4, 4) if rng.random() < .5 else (5, 5)) if index == 0 else (
            (3, 5), (5, 3), (4, 5), (5, 4))[int(rng.integers(4))]
        keys = [0, 1, 2]
        h = int(rng.integers(5, min(7, 30 // scale[0]) + 1))
        w = int(rng.integers(5, min(7, 30 // scale[1]) + 1))
        wall = _wall(rng, (h, w), index + 1, 'ring' if index == 0 else 'diagonal')
        objects = [_shell_record(wall, carrier, 0, 0)]
    elif split == 'test' and regime in ('reuse', 'grow', 'bays'):
        scale, keys = (3, 3), [0, 1]
        objects = []
        if regime == 'grow':
            shape = tuple(map(int, rng.integers(9, 11, size=2)))
            wall = _wall(rng, shape, 1, 'ring')
            objects = [_shell_record(wall, carrier, 0, 0)]
        elif regime == 'bays':
            shape = tuple(map(int, rng.integers(6, 11, size=2)))
            wall = np.ones(shape, dtype=bool)
            wall[2:-2, 2:-2] = False
            wall[:2, :2] = False
            if rng.random() < .5:
                wall = np.flipud(wall)
            if rng.random() < .5:
                wall = np.fliplr(wall)
            objects = [_shell_record(wall.copy(), carrier, 0, 0)]
        else:
            number = int(rng.integers(2, 4))
            # Three 3x3 shells fit in varied L arrangements inside a 10x10 map.
            origins = [(0, 0), (0, int(rng.integers(4, 8))), (int(rng.integers(4, 8)), 0)]
            if rng.random() < .5:
                origins = [(c, r) for r, c in origins]
            for j, (r, c) in enumerate(origins[:number]):
                key = 0 if j == number - 1 and rng.random() < .25 else 1
                wall = _wall(rng, (3, 3), key, bays=int(rng.integers(3)))
                local_carrier = carrier if rng.random() < .35 else colors[1 + j]
                objects.append(_shell_record(wall, local_carrier, r, c))
    else:
        novel = split == 'test' and regime == 'new_key'
        required = int(rng.integers(3, 6)) if novel else None
        eligible = [s for s in _STAMPS if required is None or _fallback_holes(*s, required) is not None]
        scale = eligible[int(rng.integers(len(eligible)))]
        possible = [k for k in range(6 if novel or regime != 'new_key' else 3)
                    if _fallback_holes(*scale, k) is not None]
        nonzero = [k for k in possible if k]
        target_key = required if required is not None else nonzero[int(rng.integers(len(nonzero)))]
        ncard = int(rng.integers(2, min(4, len(possible)) + 1))
        others = [k for k in possible if k != target_key]
        keys = [target_key] + list(map(int, rng.choice(others, ncard - 1, replace=False)))
        mh, mw = 30 // scale[0], 30 // scale[1]
        upper = 7 if split == 'train' and regime == 'grow' else 10
        shapes = [(h, w) for h in range(3, min(mh, upper) + 1)
                  for w in range(3, min(mw, upper) + 1)
                  if _fallback_holes(h, w, target_key) is not None]
        shape = shapes[int(rng.integers(len(shapes)))]
        bays = int(rng.integers(2 if split == 'train' and regime == 'bays' else 7))
        wall = _wall(rng, shape, target_key, bays=bays)
        objects = [_shell_record(wall, carrier, 0, 0)]
        # Additional shells are packed in free space when the visible scale
        # allows it. Equal keys are legitimate one-to-many uses of a card.
        max_objects = 1 if split == 'train' and regime == 'reuse' else int(rng.integers(1, 4))
        for j in range(1, max_objects):
            choices = []
            boxes = [(s['r'], s['c'], *s['wall'].shape) for s in objects]
            for h in range(3, min(mh, 6) + 1):
                for w in range(3, min(mw, 6) + 1):
                    free = _free_origins(mh, mw, h, w, boxes)
                    ks = [k for k in keys if _fallback_holes(h, w, k) is not None]
                    if free and ks:
                        choices.append(((h, w), free, ks))
            if not choices:
                break
            shape, free, ks = choices[int(rng.integers(len(choices)))]
            r, c = free[int(rng.integers(len(free)))]
            k = ks[int(rng.integers(len(ks)))]
            wall = _wall(rng, shape, k, bays=bays)
            local_carrier = carrier if rng.random() < .35 else colors[1 + j]
            objects.append(_shell_record(wall, local_carrier, r, c))
    cards = {k: _card(rng, scale, k, bg, payload, directional=witness) for k in keys}
    return _place(rng, bg, objects, cards, scale, witness=witness and index == 0)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f'{name} must be an integer from {lo} through {hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = ('new_key', 'reuse', 'grow', 'bays')[int(rng.integers(4))]
    stable = bool(rng.random() < .2)
    palette = list(map(int, rng.permutation(10)))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            for _ in range(24):
                pair = _sample_pair(rng, regime, split, index, palette, stable)
                key = _array_key(pair['input'])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction failed')
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _array_key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _check_cells(info, out):
    """Verify every fine cell by coarse indexing, separately from block writes."""
    r0, r1, c0, c1 = info['bounds']
    sh, sw = info['scale']
    assert out.shape == ((r1 - r0) * sh, (c1 - c0) * sw), 'wrong scene crop or scale'
    coarse_wall = np.full((r1 - r0, c1 - c0), -1, dtype=int)
    coarse_inside = np.full_like(coarse_wall, -1)
    for i, s in enumerate(info['shells']):
        r, c = s['r'] - r0, s['c'] - c0
        h, w = s['wall'].shape
        coarse_wall[r:r + h, c:c + w][s['wall']] = i
        coarse_inside[r:r + h, c:c + w][s['inside']] = i
    rr, cc = np.indices(out.shape)
    wi, ci = coarse_wall[rr // sh, cc // sw], coarse_inside[rr // sh, cc // sw]
    expected = np.full(out.shape, info['bg'], dtype=np.int8)
    for i, s in enumerate(info['shells']):
        expected[wi == i] = s['carrier']
        stamp = info['cards'][s['key']]['stamp']
        texture = np.where(stamp == info['bg'], s['carrier'], stamp)
        fine = texture[rr % sh, cc % sw]
        expected[ci == i] = fine[ci == i]
    assert np.array_equal(out, expected), 'incorrect wall, chamber texture, transparency, or exterior'


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _array_key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _read(inp)
            assert all(3 <= n <= 6 for n in info['scale']), 'card dimensions outside this family'
            assert all(0 <= k <= 5 for k in info['cards']), 'card key outside this family'
            assert all(all(3 <= n <= 10 for n in s['wall'].shape) for s in info['shells'])
            assert any(s['key'] > 0 for s in info['shells']), 'each pair must use a texture'
            _check_cells(info, out)
            assert np.array_equal(solve(inp.copy()), out)
            records[split].append(info)
    train = records['train']
    shells = [s for a in train for s in a['shells']]
    assert any(len(s['regions']) != len(_cavities(s['wall'], True)[1]) for s in shells), 'teach four-connectivity'
    assert any(np.any(~s['wall'] & ~s['inside']) for s in shells), 'teach exterior exclusion'
    assert any(any(len(cells) > 1 for cells in s['regions']) for s in shells), 'teach per-cell expansion'
    one = [int(s['inside'].sum()) for s in shells if s['key'] == 1]
    two = [int(s['inside'].sum()) for s in shells if s['key'] == 2]
    assert one and two and max(one) > min(two), 'cavity area must not substitute for count'
    assert any(_asymmetric(a['cards'][s['key']]['stamp']) for a in train for s in a['shells']), 'teach stamp orientation'
    assert any(_asymmetric(s['wall']) for s in shells), 'teach shell orientation'
    assert len({a['scale'] for a in train}) >= 2, 'teach visible independent scale dimensions'
    for selection in ('area', 'first', 'eight'):
        assert any((prediction := _render(a, selection=selection)) is None
                   or not np.array_equal(prediction, np.asarray(pair['output']))
                   for a, pair in zip(train, puzzle['train'])), 'a parent-A ablation survives training'
    assert any(not np.array_equal(_render(a, painting='solid'), np.asarray(pair['output']))
               for a, pair in zip(train, puzzle['train'])), 'a parent-B ablation survives training'


def _normalized(a):
    mapping = {}
    return a.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in a.ravel())


def self_test():
    """Bounded 42-seed semantic/diversity audit plus corruption/count tests."""
    fingerprints, in_shapes, out_shapes = set(), set(), set()
    wall_patterns, card_patterns, scales, keys, backgrounds, multiplicities = (set() for _ in range(6))
    exterior_cases = multi_cell_cases = repeated_cases = repeated_carriers = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = tuple(_array_key(p['input']) for split in puzzle for p in puzzle[split])
        fingerprints.add(fingerprint)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _read(pair['input'])
                in_shapes.add(pair['input'].shape)
                out_shapes.add(pair['output'].shape)
                scales.add(a['scale'])
                backgrounds.add(a['bg'])
                multiplicities.add(len(a['shells']))
                ks = [s['key'] for s in a['shells']]
                repeated_cases += len(set(ks)) < len(ks)
                repeated_carriers += len({s['carrier'] for s in a['shells']}) < len(a['shells'])
                for s in a['shells']:
                    keys.add(s['key'])
                    wall_patterns.add(_array_key(s['wall']))
                    exterior_cases += bool(np.any(~s['wall'] & ~s['inside']))
                    multi_cell_cases += any(len(cells) > 1 for cells in s['regions'])
                for card in a['cards'].values():
                    card_patterns.add(_normalized(card['stamp']))
    assert len(fingerprints) == 42
    assert len(in_shapes) >= 35 and len(out_shapes) >= 25
    assert len(wall_patterns) >= 90 and len(card_patterns) >= 150
    assert len(scales) >= 10 and len(backgrounds) >= 8
    assert {1, 2, 3} <= multiplicities and {0, 1, 2, 3, 4, 5} <= keys
    assert exterior_cases >= 42 and multi_cell_cases >= 42 and repeated_cases >= 8
    assert repeated_carriers >= 3
    a, b = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(123, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    # Opening one chamber changes the key of the entire shell: even an
    # untouched chamber must receive a different texture. Exercise solve on
    # this member, which is not a normally sampled closed-boundary template.
    wall = np.ones((5, 5), dtype=bool)
    wall[1, 1] = wall[2, 2] = wall[2, 3] = False
    rng = np.random.default_rng(819)
    cards = {k: _card(rng, (4, 4), k, 0, [3, 6, 7], True) for k in (0, 1, 2)}
    example = _place(rng, 0, [_shell_record(wall, 2, 0, 0)], cards, (4, 4))
    before = _read(example['input'])
    s = before['shells'][0]
    # _place may pose the coarse scene. Locate the singleton chamber and carve
    # toward its nearest edge, then inspect the untouched two-cell chamber.
    singleton = next(cells[0] for cells in s['regions'] if len(cells) == 1)
    sr, sc = singleton
    steps = [(sr, 0, (sr, 1)), (sr, 4, (sr, 3)),
             (0, sc, (1, sc)), (4, sc, (3, sc))]
    edge_r, edge_c, neighbor = min(steps, key=lambda t: abs(t[0] - sr) + abs(t[1] - sc))
    assert neighbor == singleton
    opened = example['input'].copy()
    opened[s['r'] + edge_r, s['c'] + edge_c] = 0
    after = _read(opened)
    assert s['key'] == 2 and after['shells'][0]['key'] == 1
    original, changed = solve(example['input']), solve(opened)
    cr, cc = next(cells[0] for cells in s['regions'] if len(cells) == 2)
    assert not np.array_equal(original[cr * 4:(cr + 1) * 4, cc * 4:(cc + 1) * 4],
                              changed[cr * 4:(cr + 1) * 4, cc * 4:(cc + 1) * 4])
    assert np.all(changed[sr * 4:(sr + 1) * 4, sc * 4:(sc + 1) * 4] == 0), 'opened chamber must join the exterior'
    return {'seeds': 42, 'distinct_puzzles': len(fingerprints), 'input_shapes': len(in_shapes),
            'output_shapes': len(out_shapes), 'shell_patterns': len(wall_patterns),
            'card_patterns': len(card_patterns), 'stamp_shapes': len(scales),
            'shell_counts': sorted(multiplicities), 'cavity_counts': sorted(keys),
            'backgrounds': len(backgrounds), 'exterior_shells': exterior_cases,
            'multi_cell_shells': multi_cell_cases, 'repeated_key_pairs': repeated_cases,
            'repeated_carrier_pairs': repeated_carriers}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
