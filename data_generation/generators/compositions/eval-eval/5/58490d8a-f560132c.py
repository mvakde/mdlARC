"""Seam census: assemble an inked jigsaw, count joined objects, tally its rim.

Only Python's standard library and NumPy are used. Sampling metadata never
enters solve(); labels are constructed before input-only reconstruction checks.
"""
import heapq
import itertools
import math

import numpy as np

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_X = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
_CROSS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer cells required')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'colors outside 0..9')
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = _EIGHT if diagonal else _ORTH
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.array(cells, dtype=np.int16))
    return result


def _tile(image, mask):
    coords = np.argwhere(mask)
    r, c = coords.min(axis=0)
    rr, cc = coords.max(axis=0) + 1
    return np.where(mask[r:rr, c:cc], image[r:rr, c:cc], -1).astype(np.int8)


def _bits(mask, side, row=0, col=0):
    result = 0
    for r, c in np.argwhere(mask):
        result |= 1 << ((int(r) + row) * side + int(c) + col)
    return result


def _placements(tile, side, corner, anchor=False):
    """An extreme corner fixes the translation of each rotated bounding box.

    Deduplicate only identical decorated placements, never just silhouettes:
    a symmetric silhouette can carry asymmetric ink.
    """
    found = {}
    vertices = ((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))
    for k in ((0,) if anchor else range(4)):
        t = np.rot90(tile, k).copy()
        h, w = t.shape
        if h > side or w > side:
            continue
        r = 0 if corner < 2 else side-h
        c = 0 if corner % 2 == 0 else side-w
        mask = t >= 0
        hits = [0 <= vr-r < h and 0 <= vc-c < w and mask[vr-r, vc-c]
                for vr, vc in vertices]
        if sum(hits) != 1 or not hits[corner]:
            continue
        canvas = np.full((side, side), -1, np.int8)
        canvas[r:r+h, c:c+w] = t
        sig = _signature(canvas)
        found.setdefault(sig, (_bits(mask, side, r, c), r, c, t, k))
    return list(found.values())


def _pack(tiles, side):
    """Enumerate all corner assignments; stop at two distinct inked squares."""
    anchor = _placements(tiles[0], side, 0, True)
    if len(anchor) != 1:
        return []
    candidates = {(i, j): _placements(tiles[i], side, j)
                  for i in range(1, 4) for j in range(1, 4)}
    full = (1 << (side*side)) - 1
    chosen, answers, seen = [anchor[0]], [], set()

    def search(corner, left, occupied):
        if len(answers) >= 2:
            return
        if corner == 4:
            if occupied != full:
                return
            image = np.full((side, side), -1, np.int8)
            owners = np.full((side, side), -1, np.int8)
            rotations = []
            for j, (_, r, c, t, k) in enumerate(chosen):
                part = image[r:r+t.shape[0], c:c+t.shape[1]]
                owner = owners[r:r+t.shape[0], c:c+t.shape[1]]
                part[t >= 0] = t[t >= 0]
                owner[t >= 0] = j
                rotations.append(k)
            sig = _signature(image)
            if sig not in seen:
                seen.add(sig)
                answers.append(dict(image=image, owners=owners, rotations=tuple(rotations)))
            return
        for i in left:
            for placement in candidates[i, corner]:
                if not placement[0] & occupied:
                    chosen.append(placement)
                    search(corner+1, tuple(x for x in left if x != i), occupied | placement[0])
                    chosen.pop()

    search(1, (1, 2, 3), anchor[0][0])
    return answers


def _tallies(side, field, key, counts):
    out = np.full((side, side), field, np.int8)
    # Key order is TL, TR, BL, BR; each corner takes its clockwise edge.
    origins = ((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))
    steps = ((0, 1), (1, 0), (-1, 0), (0, -1))
    for color, (r, c), (dr, dc) in zip(key, origins, steps):
        n = counts[int(color)]
        _require(1 <= n <= side//2, 'positive count must fit its outgoing edge')
        for j in range(n):
            out[r+2*j*dr, c+2*j*dc] = color
    return out


def _inspect(input_grid):
    a = _grid(input_grid)
    border = np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))
    bg = int(np.bincount(border, minlength=10).argmax())
    _require(np.count_nonzero(border == bg) > len(border)//2, 'readable background border')
    comps = _components(a != bg)
    _require(len(comps) == 4, 'exactly four piece footprints')
    for x, y in itertools.combinations(comps, 2):
        _require(np.max(np.abs(x[:, None] - y[None, :]), axis=2).min() >= 2,
                 'pieces must be separated even diagonally')
    total = sum(len(x) for x in comps)
    side = math.isqrt(total)
    _require(side*side == total, 'piece areas must sum to a square')
    values = a[a != bg]
    field = int(np.bincount(values, minlength=10).argmax())
    _require(np.count_nonzero(values == field) > total//2, 'common carrier must dominate footprints')
    patches = []
    for r in range(a.shape[0]-1):
        for c in range(a.shape[1]-1):
            patch = a[r:r+2, c:c+2]
            if len(set(map(int, patch.flat))) == 4 and not np.isin(patch, (bg, field)).any():
                patches.append((r, c))
    _require(len(patches) == 1, 'one recognizable four-color legend')
    kr, kc = patches[0]
    key = a[kr:kr+2, kc:kc+2].reshape(-1).copy()
    anchor = [i for i, comp in enumerate(comps) if any(r == kr and c == kc for r, c in comp)]
    _require(len(anchor) == 1, 'legend belongs to one anchor')
    ai = anchor[0]
    anchor_cells = set(map(tuple, comps[ai]))
    _require(all((kr+dr, kc+dc) in anchor_cells for dr in (0, 1) for dc in (0, 1)),
             'legend cannot cross a cut')
    # The legend is ink-free metadata. Its footprint remains part of the tile.
    cleared = a.copy()
    cleared[kr:kr+2, kc:kc+2] = field
    ring = cleared[max(0, kr-1):kr+3, max(0, kc-1):kc+3]
    _require(bool(np.all((ring == field) | (ring == bg))), 'legend needs an ink-free halo')
    comps = [comps[ai]] + [x for i, x in enumerate(comps) if i != ai]
    tiles = []
    for comp in comps:
        mask = np.zeros(a.shape, bool)
        mask[comp[:, 0], comp[:, 1]] = True
        tiles.append(_tile(cleared, mask))
    packed = _pack(tiles, side)
    _require(len(packed) == 1, 'decorated square must have a unique assembly')
    image, owners = packed[0]['image'], packed[0]['owners']
    ar, ac = comps[0].min(axis=0)
    local_kr, local_kc = int(kr-ar), int(kc-ac)
    _require(bool(np.all(image[max(0, local_kr-1):local_kr+3,
                               max(0, local_kc-1):local_kc+3] == field)),
             'reconstructed legend halo must be ink-free')
    ink_colors = sorted(set(map(int, image.flat)) - {field})
    counts = {color: len(_components(image == color, True)) for color in ink_colors}
    _require(set(map(int, key)) <= set(counts), 'every key needs positive ink objects')
    # Separate reconstructed objects retain a one-cell gap, even across colors.
    object_id = np.full(image.shape, -1, np.int16)
    serial = 0
    for color in ink_colors:
        for obj in _components(image == color, True):
            object_id[obj[:, 0], obj[:, 1]] = serial
            serial += 1
    for r, c in np.argwhere(object_id >= 0):
        nearby = object_id[max(0, r-1):r+2, max(0, c-1):c+2]
        _require(bool(np.all((nearby < 0) | (nearby == object_id[r, c]))), 'ink-object separation')
    expected = _tallies(side, field, key, counts)
    fragments = {int(color): sum(len(_components(t == color, True)) for t in tiles) for color in key}
    orth = {int(color): len(_components(image == color, False)) for color in key}
    crossings = [len(set(map(int, owners[obj[:, 0], obj[:, 1]])))
                 for color in key for obj in _components(image == color, True)]
    mixed = any(len({len(x) for x in _components(image == color, True)}) > 1 for color in key)
    return dict(side=side, background=bg, field=field, key=tuple(map(int, key)), counts=counts,
                image=image, owners=owners, tiles=tiles, fragments=fragments, orth=orth,
                crossings=tuple(crossings), mixed=mixed, expected=expected,
                rotations=packed[0]['rotations'], areas=tuple(len(x) for x in comps),
                ignored=tuple(c for c in ink_colors if c not in key))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover everything from the supplied grid; no training or seed state."""
    return _inspect(input_grid)['expected'].copy()


def _cut(rng, side, style):
    """Procedural corner partitions: wandering seams, growth, or corner carving."""
    if style == 0:
        split = int(rng.integers(3, side-3))
        widths = [int(rng.integers(3, side-2)), int(rng.integers(2, side-2))]
        owner = np.empty((side, side), np.int8)
        for r in range(side):
            band = int(r >= split)
            if r not in (0, split) and rng.random() < .65:
                widths[band] = int(np.clip(widths[band]+rng.choice((-1, 1)), 2, side-2))
            owner[r, :widths[band]] = 2*band
            owner[r, widths[band]:] = 2*band+1
        for _ in range(side*2):
            r, c = map(int, rng.integers(1, side-1, 2))
            before = int(owner[r, c])
            owner[r, c] = owner[rng.choice((r-1, r+1)), c] if rng.random() < .5 else owner[r, rng.choice((c-1, c+1))]
            if len(_components(owner == before)) != 1:
                owner[r, c] = before
        return owner
    if style == 1:
        owner = np.full((side, side), -1, np.int8)
        queue = []
        for i, (r, c) in enumerate(((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))):
            heapq.heappush(queue, (0., i, r, c))
        speeds = rng.uniform(.5, 1.8, 4)
        while queue:
            cost, i, r, c = heapq.heappop(queue)
            if owner[r, c] >= 0:
                continue
            owner[r, c] = i
            for dr, dc in _ORTH:
                rr, cc = r+dr, c+dc
                if 0 <= rr < side and 0 <= cc < side and owner[rr, cc] < 0:
                    heapq.heappush(queue, (cost+float(rng.uniform(.2, 2.0))*speeds[i], i, rr, cc))
        return owner
    large = int(rng.integers(4))
    owner = np.full((side, side), large, np.int8)
    for i in rng.permutation([j for j in range(4) if j != large]):
        i = int(i)
        h, w = map(int, rng.integers(3, side-2, 2))
        widths = [w, w]
        for depth in range(2, h):
            widths.append(int(rng.integers(1, w+1)))
        for depth, width in enumerate(widths):
            for x in range(width):
                r = depth if i < 2 else side-1-depth
                c = x if i % 2 == 0 else side-1-x
                if owner[r, c] != large:
                    return None
                owner[r, c] = i
    return owner


def _partition(rng, side, style):
    for attempt in range(300):
        owner = _cut(rng, side, style if attempt < 200 else int(rng.integers(3)))
        if owner is None or any(len(_components(owner == i)) != 1 for i in range(4)):
            continue
        if any(np.count_nonzero(owner == i) < 7 for i in range(4)):
            continue
        tiles = [_tile(np.full(owner.shape, i+1, np.int8), owner == i) for i in range(4)]
        # All four orientations must be geometrically distinguishable. This
        # guards against silently rotating ink on a symmetric silhouette.
        if any(len({_signature(np.rot90(t >= 0, k)) for k in range(4)}) != 4 for t in tiles):
            continue
        locations = [(r, c) for r in range(side-1) for c in range(side-1)
                     if np.all(owner[r:r+2, c:c+2] == 0)]
        if not locations:
            continue
        answers = _pack(tiles, side)
        if len(answers) == 1 and np.array_equal(answers[0]['image'], owner+1):
            return owner, locations
    raise RuntimeError('bounded partition search exhausted')


def _normalize(cells):
    r, c = min(x[0] for x in cells), min(x[1] for x in cells)
    return tuple(sorted((rr-r, cc-c) for rr, cc in cells))


def _glyph(rng):
    prototypes = (((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 1)),
                  ((0, 0), (1, 0), (1, 1)), _DIAMOND, _SQUARE, _X, _CROSS)
    if rng.random() < .30:
        h, w = 3, 3
        target = int(rng.integers(2, 6))
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _EIGHT
                               if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    else:
        cells = set(prototypes[int(rng.integers(len(prototypes)))])
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    for _ in range(int(rng.integers(4))):
        cells = {(-c, r) for r, c in cells}
    return _normalize(cells)


def _ink(rng, owner, field, key, counts, witness, min_cross, ignored):
    side = len(owner)
    positions = [(r, c) for r in range(side-1) for c in range(side-1)
                 if np.all(owner[r:r+2, c:c+2] == 0)]
    kr, kc = positions[int(rng.integers(len(positions)))]
    ink = np.full(owner.shape, field, np.int8)
    blocked = np.zeros(owner.shape, bool)
    blocked[max(0, kr-1):kr+3, max(0, kc-1):kc+3] = True
    target = int(np.argmax(counts))
    objects = []
    for i, (color, count) in enumerate(zip(key, counts)):
        for j in range(count):
            shape = (_DIAMOND, _SQUARE, ((0, 0),))[j] if witness and i == target and j < 3 else _glyph(rng)
            crossing = min_cross if i == target and j == 0 else 1
            objects.append((int(color), shape, crossing))
    # Joining witness is placed first; larger shapes precede flexible singletons.
    objects.sort(key=lambda x: (-x[2], -len(x[1])))
    if ignored is not None:
        for _ in range(int(rng.integers(1, 4))):
            objects.append((ignored, _glyph(rng), 1))
    for color, shape, crossing in objects:
        alternatives = [shape]
        if crossing >= 3:
            alternatives = [_X, _CROSS, _SQUARE, _DIAMOND]
            rng.shuffle(alternatives)
        placed = False
        for glyph in alternatives:
            h, w = 1+max(r for r, c in glyph), 1+max(c for r, c in glyph)
            options = []
            for r in range(side-h+1):
                for c in range(side-w+1):
                    cells = [(r+dr, c+dc) for dr, dc in glyph]
                    if any(blocked[p] for p in cells):
                        continue
                    if len({int(owner[p]) for p in cells}) < crossing:
                        continue
                    # Occupying different pieces necessarily creates at least
                    # one fragment per piece before scattering.
                    options.append(cells)
            if not options:
                continue
            cells = options[int(rng.integers(len(options)))]
            for r, c in cells:
                ink[r, c] = color
                blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
            placed = True
            break
        if not placed:
            return None
    if np.count_nonzero(ink == field) - 4 <= side*side//2:
        return None
    return ink, (kr, kc)


def _scatter(rng, tiles):
    """Free placement or arbitrary shelves; layouts are independent of corners."""
    if rng.random() < .45:
        h, w = map(int, rng.integers(23, 31, 2))
        grid = np.full((h, w), -1, np.int8)
        blocked = np.zeros((h, w), bool)
        offsets = [None]*4
        for i in rng.permutation(4):
            t = tiles[i]
            coords = np.argwhere(t >= 0)
            for _ in range(120):
                r = int(rng.integers(1, h-t.shape[0]))
                c = int(rng.integers(1, w-t.shape[1]))
                rr, cc = coords[:, 0]+r, coords[:, 1]+c
                if blocked[rr, cc].any():
                    continue
                grid[rr, cc] = t[coords[:, 0], coords[:, 1]]
                offsets[i] = (r, c)
                for dr, dc in _EIGHT + ((0, 0),):
                    blocked[rr+dr, cc+dc] = True
                break
            else:
                break
        if all(x is not None for x in offsets):
            return grid
    layouts = []
    gap = int(rng.integers(1, 4))
    for transposed in (False, True):
        dims = [t.shape[::-1] if transposed else t.shape for t in tiles]
        for order in itertools.permutations(range(4)):
            for cuts in range(8):
                groups, group = [], []
                for j, i in enumerate(order):
                    group.append(i)
                    if j == 3 or cuts & (1 << j):
                        groups.append(group)
                        group = []
                heights = [max(dims[i][0] for i in g) for g in groups]
                widths = [sum(dims[i][1] for i in g)+gap*(len(g)-1) for g in groups]
                h, w = sum(heights)+gap*(len(groups)-1), max(widths)
                if max(h, w) <= 28:
                    layouts.append((transposed, groups, dims, heights, widths, h, w))
    if not layouts and gap != 1:
        # A 2-by-2 arrangement of <=13-cell boxes fits with gap one.
        order = list(map(int, rng.permutation(4)))
        groups = [order[:2], order[2:]]
        dims = [t.shape for t in tiles]
        heights = [max(dims[i][0] for i in g) for g in groups]
        widths = [sum(dims[i][1] for i in g)+1 for g in groups]
        layouts = [(False, groups, dims, heights, widths, sum(heights)+1, max(widths))]
        gap = 1
    _require(bool(layouts), 'no legal scatter layout')
    transpose, groups, dims, heights, widths, h, w = layouts[int(rng.integers(len(layouts)))]
    if transpose:
        h, w = w, h
    top = int(rng.integers(1, min(4, 30-h)))
    left = int(rng.integers(1, min(4, 30-w)))
    bottom = int(rng.integers(1, min(4, 31-h-top)))
    right = int(rng.integers(1, min(4, 31-w-left)))
    grid = np.full((h+top+bottom, w+left+right), -1, np.int8)
    row = 0
    for g, height, width in zip(groups, heights, widths):
        totalwidth = w if not transpose else h
        col = int(rng.integers(totalwidth-width+1))
        for i in g:
            rr = row+int(rng.integers(height-dims[i][0]+1))
            r, c = (col+top, rr+left) if transpose else (rr+top, col+left)
            t = tiles[i]
            part = grid[r:r+t.shape[0], c:c+t.shape[1]]
            part[t >= 0] = t[t >= 0]
            col += dims[i][1]+gap
        row += height+gap
    return grid


def _make_pair(rng, side, style, counts, key, bg, field, witness, min_cross, rotations, ignored):
    for _ in range(80):
        owner, _ = _partition(rng, side, style)
        painted = None
        for _ in range(12):
            painted = _ink(rng, owner, field, key, counts, witness, min_cross, ignored)
            if painted is not None:
                break
        if painted is None:
            continue
        ink, (kr, kc) = painted
        # Labels depend on the complete-object construction, not on solve().
        expected = _tallies(side, field, key, dict(zip(map(int, key), map(int, counts))))
        cut_image = ink.copy()
        cut_image[kr:kr+2, kc:kc+2] = np.array(key).reshape(2, 2)
        tiles = [np.rot90(_tile(cut_image, owner == i), rotations[i]).copy() for i in range(4)]
        scattered = _scatter(rng, tiles)
        grid = np.where(scattered >= 0, scattered, bg).astype(np.int8)
        meta = _inspect(grid)
        _require(np.array_equal(meta['image'], ink), 'ink changed during reconstruction')
        _require(np.array_equal(meta['expected'], expected), 'construction/visible census mismatch')
        return {'input': grid, 'output': expected}
    raise RuntimeError('bounded ink placement search exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if not _integer(value) or not low <= value <= high:
            raise ValueError(name+' must be an integer in its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # count extrapolation, triple-seam transfer, scale reduction
    palette = list(map(int, rng.permutation(10)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', num_train), ('test', num_test)):
        for index in range(number):
            train = split == 'train'
            witness = train and index == 0
            side = int(rng.integers(12, 14) if train or regime != 2 else rng.integers(10, 12))
            style = int(rng.integers(3))
            if regime == 1:
                style = 0 if train else int(rng.integers(1, 3))
            counts = rng.integers(1, 4, 4).astype(int)
            if witness:
                counts[:] = rng.permutation([3, 2, 1, 1])
            if not train and regime == 0:
                counts[int(rng.integers(4))] = int(rng.integers(4, side//2+1))
            # Limit crowding while preserving a count witness and all keys.
            while sum(counts) > (10 if side < 12 else 12):
                candidates = [i for i in range(4) if counts[i] > 1 and i != int(np.argmax(counts))]
                counts[int(rng.choice(candidates))] -= 1
            key = tuple(palette[2:6]) if witness else tuple(map(int, rng.choice(10, 4, replace=False)))
            if train and index == 1 and key == tuple(palette[2:6]):
                key = key[1:]+key[:1]
            available = [c for c in range(10) if c not in key]
            bg, field = map(int, rng.choice(available, 2, replace=False))
            ignore = int(rng.choice([c for c in available if c not in (bg, field)])) if (index == 1 or rng.random() < .6) else None
            if witness:
                rotations = [0, 0, 0, 0]
            elif train and index == 1:
                rotations = [0, *map(int, rng.permutation([1, 2, 3]))]
            else:
                rotations = [0, *map(int, rng.integers(0, 4, 3))]
            crossing = 3 if not train and regime == 1 else 2
            for _ in range(8):
                pair = _make_pair(rng, side, style, counts, key, bg, field, witness, crossing, rotations, ignore)
                sig = _signature(pair['input'])
                if sig not in seen:
                    break
            else:
                raise RuntimeError('distinct input search exhausted')
            seen.add(sig)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(inp)
            _require(sig not in seen, 'duplicate input')
            seen.add(sig)
            m = _inspect(inp)
            _require(np.array_equal(m['expected'], out), 'wrong census or tally')
            _require(any(m['fragments'][c] > m['counts'][c] for c in m['key']), 'assembly must change a census')
            if split == 'train':
                training.append(m)
    _require(any(any(m['orth'][c] > m['counts'][c] for c in m['key']) and m['mixed']
                 and max(m['counts'][c] for c in m['key']) >= 3 for m in training),
             'training needs diagonal, mixed-area and extended-tally evidence')
    _require(any(m['ignored'] for m in training), 'training needs ignored ink evidence')
    _require(len({m['key'] for m in training}) >= 2, 'training legend colors must vary')
    _require({1, 2, 3} <= {k for m in training for k in m['rotations']}, 'training must expose quarter-turns')


def self_test():
    """Bounded, pixel-derived structural coverage over exactly 42 seeds."""
    puzzles, metas = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        local = [_inspect(pair['input']) for s in ('train', 'test') for pair in p[s]]
        _require(len({_signature(m['owners']) for m in local}) >= 2, 'cuts vary within each puzzle')
        metas.extend(local)
    sig = lambda p: tuple(_signature(pair[k]) for s in ('train', 'test') for pair in p[s] for k in ('input', 'output'))
    _require(len({sig(p) for p in puzzles}) == 42, 'seeds must change puzzles')
    _require(sig(generate_puzzle(0)) == sig(puzzles[0]), 'determinism')
    _require({m['field'] for m in metas} == set(range(10)), 'carrier palette variation')
    _require({m['background'] for m in metas} == set(range(10)), 'background palette variation')
    _require({m['side'] for m in metas} >= {10, 11, 12, 13}, 'assembly sizes')
    _require(any(max(m['crossings']) >= 3 for m in metas), 'multi-piece joins')
    _require(any(max(m['counts'][c] for c in m['key']) >= 5 for m in metas), 'larger tallies')
    _require(len({_signature(m['owners']) for m in metas}) >= 150, 'geometric cut variation')
    _require(len({m['areas'] for m in metas}) >= 80, 'piece-size variation')
    _require(max(max(m['areas'])/min(m['areas']) for m in metas) >= 4, 'unequal silhouettes')
    _require(all(any(m['fragments'][c] != m['counts'][c] for c in m['key']) for m in metas), 'B ablation')
    _require(all(any(np.count_nonzero(m['image'] == c) != m['counts'][c] for c in m['key']) for m in metas), 'A area ablation')
    # Routing changes when the visible key changes even though the census and
    # restored ink do not. This checks B's spatial-key role separately from joins.
    example = puzzles[0]['train'][0]['input'].copy()
    before = _inspect(example)
    locations = [(r, c) for r in range(example.shape[0]-1) for c in range(example.shape[1]-1)
                 if tuple(map(int, example[r:r+2, c:c+2].flat)) == before['key']]
    _require(len(locations) == 1, 'key location for routing check')
    r, c = locations[0]
    example[r, c], example[r, c+1] = example[r, c+1], example[r, c]
    after = _inspect(example)
    _require(np.array_equal(before['image'], after['image']) and before['counts'] == after['counts'],
             'legend cells must not enter the census')
    _require(not np.array_equal(before['expected'], after['expected']), 'visible key must route tallies')
    # Capacity includes the last legal mark but never the next corner.
    for side in (12, 13):
        rim = _tallies(side, 0, (1, 2, 3, 4), {c: side//2 for c in (1, 2, 3, 4)})
        _require(all(np.count_nonzero(rim == c) == side//2 for c in (1, 2, 3, 4)), 'boundary tally capacity')
        _require(np.all(rim[1:-1, 1:-1] == 0), 'tallies stay on perimeter')
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(100+nt, nt, nq)
        validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    corrupted = {s: [{k: v.copy() for k, v in x.items()} for x in puzzles[0][s]] for s in ('train', 'test')}
    corrupted['test'][0]['output'][1, 1] = (int(corrupted['test'][0]['output'][1, 1])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
