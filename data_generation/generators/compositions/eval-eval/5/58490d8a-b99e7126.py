"""Count exterior objects to warp a glyph's tile-scale stencil, then complete it.

No evidence files, example cache, random state, or training pairs are used by
solve. Labels are constructed from sampled object counts and checked by parsing
the finished inputs. Only Python's standard library and NumPy are required.
"""
from functools import lru_cache
from itertools import product
import numpy as np


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _grid(a):
    a = np.asarray(a)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, part = [start], []
        while todo:
            r, c = todo.pop()
            part.append((int(r), int(c)))
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(part)
    return result


def _normalize(points):
    r0, c0 = min(r for r, c in points), min(c for r, c in points)
    return tuple(sorted((r-r0, c-c0) for r, c in points))


def _asymmetric(mask):
    target = _key(mask)
    transforms = [np.rot90(mask, k) for k in range(4)]
    transforms += [np.fliplr(a) for a in transforms]
    return all(_key(a) != target for a in transforms[1:])


def _coordinates(left, top, counts):
    ys = tuple(map(int, np.cumsum([0] + [counts[c] for c in left])))
    xs = tuple(map(int, np.cumsum([0] + [counts[c] for c in top])))
    return ys, xs


def _offsets(mask, ys, xs):
    return tuple((ys[int(r)], xs[int(c)]) for r, c in np.argwhere(mask))


def _origins(offsets, window, lattice, visible):
    wh, ww = window
    nr, nc = lattice
    active = set(offsets)
    return [(r, c) for r in range(nr-wh+1) for c in range(nc-ww+1)
            if all((y-r, x-c) in active for y, x in visible)]


def _tile_start(r, c, h, w, t):
    return t + r*(h+t), t + c*(w+t)


def _fill(board, glyph, positions, h, w, t):
    result = board.copy()
    for r, c in positions:
        y, x = _tile_start(r, c, h, w, t)
        result[y:y+h, x:x+w] = glyph
    return result


def _board_candidates(a):
    """Mesh geometry is decoded before consulting any inventory or output."""
    found = []
    for divider in map(int, np.unique(a)):
        rr, cc = np.where(a == divider)
        r0, r1 = int(rr.min()), int(rr.max())+1
        c0, c1 = int(cc.min()), int(cc.max())+1
        board = a[r0:r1, c0:c1]
        bh, bw = board.shape
        if min(bh, bw) < 7 or max(bh, bw) > 26:
            continue
        if not all(board[r, c] == divider for r in (0, bh-1) for c in (0, bw-1)):
            continue
        for h in range(2, 5):
            for w in range(2, 5):
                for t in (1, 2):
                    if (bh-t) % (h+t) or (bw-t) % (w+t):
                        continue
                    nr, nc = (bh-t)//(h+t), (bw-t)//(w+t)
                    if min(nr, nc) < 2:
                        continue
                    yy, xx = np.indices(board.shape)
                    mesh = (yy % (h+t) < t) | (xx % (w+t) < t)
                    key_positions = [(0, t+j) for j in range(1, w)]
                    key_positions += [(t+i, 0) for i in range(1, h)]
                    holes = np.zeros(board.shape, bool)
                    for p in key_positions:
                        holes[p] = True
                    if not (np.all(board[mesh & ~holes] == divider)
                            and np.all(board[~mesh | holes] != divider)):
                        continue
                    top = tuple(int(board[0, t+j]) for j in range(1, w))
                    left = tuple(int(board[t+i, 0]) for i in range(1, h))
                    keys = set(left + top)
                    if not 2 <= len(keys) <= 4:
                        continue
                    records = {}
                    for r in range(nr):
                        for c in range(nc):
                            y, x = _tile_start(r, c, h, w, t)
                            tile = board[y:y+h, x:x+w]
                            records.setdefault(_key(tile), [tile.copy(), []])[1].append((r, c))
                    if len(records) != 2:
                        continue
                    ordered = sorted(records.values(), key=lambda v: len(v[1]), reverse=True)
                    ordinary, op = ordered[0]
                    glyph, visible = ordered[1]
                    if not len(op) > len(visible) >= 2:
                        continue
                    ordinary_colors = set(map(int, np.unique(ordinary)))
                    novel = set(map(int, np.unique(glyph))) - ordinary_colors
                    if len(ordinary_colors) != 2 or len(novel) != 1:
                        continue
                    ink = novel.pop()
                    mask = glyph == ink
                    if not (3 <= int(mask.sum()) <= 12 and not mask.all()
                            and np.all(mask.any(axis=0)) and np.all(mask.any(axis=1))):
                        continue
                    if len(np.unique(glyph[~mask])) != 1:
                        continue
                    if int(glyph[~mask][0]) not in ordinary_colors:
                        continue
                    if not np.any(ordinary[~mask] != glyph[~mask]):
                        continue
                    structural = ordinary_colors | {ink, divider}
                    if keys & structural:
                        continue
                    outside = np.ones(a.shape, bool)
                    outside[r0:r1, c0:c1] = False
                    if not outside.any():
                        continue
                    values = np.bincount(a[outside], minlength=10)
                    background = int(np.argmax(values))
                    if background in structural | keys or values[background] <= .55*outside.sum():
                        continue
                    if background in board:
                        continue
                    ring = np.zeros(a.shape, bool)
                    ring[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
                    ring[r0:r1, c0:c1] = False
                    if not np.all(a[ring] == background):
                        continue
                    exterior_colors = set(map(int, np.unique(a[outside]))) - {background}
                    if exterior_colors & structural or not keys <= exterior_colors:
                        continue
                    found.append(dict(board=board.copy(), bounds=(r0, r1, c0, c1),
                                      h=h, w=w, t=t, lattice=(nr, nc), divider=divider,
                                      background=background, outside=outside, top=top, left=left,
                                      keys=keys, ordinary=ordinary, glyph=glyph, ink=ink,
                                      mask=mask, visible=tuple(visible), exterior=exterior_colors))
    return found


def _decode(input_grid):
    a = _grid(input_grid)
    candidates = _board_candidates(a)
    assert len(candidates) == 1, "The framed tile board must be unique"
    m = candidates[0]
    owner = np.full(a.shape, -1, int)
    objects, serial = {}, 0
    for color in sorted(m['exterior']):
        parts = _components((a == color) & m['outside'])
        objects[color] = tuple(_normalize(p) for p in parts)
        for part in parts:
            for r, c in part:
                owner[r, c] = serial
            serial += 1
    for r, c in np.argwhere(owner >= 0):
        neighbors = owner[max(0, r-1):r+2, max(0, c-1):c+2]
        assert np.all((neighbors < 0) | (neighbors == owner[r, c])), "Objects touch"
    counts = {c: len(objects[c]) for c in m['keys']}
    assert all(1 <= n <= 3 for n in counts.values())
    for shapes in objects.values():
        for shape in shapes:
            assert max(r for r, c in shape) < 3 and max(c for r, c in shape) < 3
    ys, xs = _coordinates(m['left'], m['top'], counts)
    offsets = _offsets(m['mask'], ys, xs)
    window = ys[-1]+1, xs[-1]+1
    origins = _origins(offsets, window, m['lattice'], m['visible'])
    assert len(origins) == 1, "The count-warped stencil must have one fitting translation"
    origin = origins[0]
    active = {(origin[0]+r, origin[1]+c) for r, c in offsets}
    visible = set(m['visible'])
    assert visible < active, "The input must be a strict partial stencil"
    expected = _fill(m['board'], m['glyph'], active-visible, m['h'], m['w'], m['t'])
    m.update(counts=counts, objects=objects, ys=ys, xs=xs, offsets=offsets,
             window=window, origin=origin, active=active, missing=active-visible,
             expected=expected, distractors=set(objects)-m['keys'])
    return m


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the exact cropped completion solely from the visible input."""
    return _decode(input_grid)['expected'].copy()


@lru_cache(maxsize=12)
def _codes(n):
    codes = np.arange(1, 1 << n, dtype=np.uint32)
    sizes = np.zeros(len(codes), dtype=np.uint8)
    for bit in range(n):
        sizes += ((codes >> bit) & 1).astype(np.uint8)
    codes.flags.writeable = sizes.flags.writeable = False
    return codes, sizes


def _fragments(offsets, window, lattice, origin, partial_edge=False):
    """All strict clue subsets which exclude every competing translation."""
    codes, sizes = _codes(len(offsets))
    limit = min(len(offsets)-1, (lattice[0]*lattice[1]-1)//2)
    valid = codes[(sizes >= 2) & (sizes <= limit)].copy()
    active = set(offsets)
    wh, ww = window
    for r in range(lattice[0]-wh+1):
        for c in range(lattice[1]-ww+1):
            if (r, c) == origin:
                continue
            cover = sum(1 << i for i, (y, x) in enumerate(offsets)
                        if (origin[0]+y-r, origin[1]+x-c) in active)
            if cover:
                valid = valid[(valid & np.uint32(cover)) != valid]
    if partial_edge:
        keep = []
        for code in map(int, valid):
            selected = [p for i, p in enumerate(offsets) if code & (1 << i)]
            if (min(y for y, x in selected) > 0 or max(y for y, x in selected) < wh-1
                    or min(x for y, x in selected) > 0 or max(x for y, x in selected) < ww-1):
                keep.append(code)
        valid = np.asarray(keep, dtype=np.uint32)
    return valid


def _mask(rng, h, w, asymmetric=False):
    for _ in range(100):
        count = int(rng.integers(max(3, h, w), min(12, h*w-1)+1))
        mask = np.zeros((h, w), bool)
        mask.flat[rng.choice(h*w, count, replace=False)] = True
        if np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)):
            if not asymmetric or _asymmetric(mask):
                return mask
    # An asymmetric 3x3 design; this fallback is only used for the witnesses.
    if asymmetric and (h, w) == (3, 3):
        return np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]], bool)
    return None


_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SINGLE = ((0, 0),)
_SHAPES = (_DIAMOND, _SINGLE, ((0, 0), (0, 1), (1, 0), (1, 1)),
           ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
           ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)))


def _shape(rng):
    if rng.random() < .55:
        shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
    else:
        target = int(rng.integers(2, 8))
        cells = {(1, 1)}
        while len(cells) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < 3 and 0 <= c+dc < 3} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
        shape = tuple(cells)
    if rng.integers(2):
        shape = tuple((r, -c) for r, c in shape)
    for _ in range(int(rng.integers(4))):
        shape = tuple((-c, r) for r, c in shape)
    return _normalize(shape)


def _place(grid, blocked, color, shape, rng):
    sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
    nh, nw = grid.shape[0]-sh+1, grid.shape[1]-sw+1
    allowed = np.ones((nh, nw), bool)
    for dr, dc in shape:
        allowed &= ~blocked[dr:dr+nh, dc:dc+nw]
    positions = np.argwhere(allowed)
    if len(positions) == 0:
        return False
    r0, c0 = positions[int(rng.integers(len(positions)))]
    _paint_object(grid, blocked, color, shape, int(r0), int(c0))
    return True


def _paint_object(grid, blocked, color, shape, r0, c0):
    for dr, dc in shape:
        r, c = r0+dr, c0+dc
        grid[r, c] = color
        blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True


def _scene(rng, board, background, objects):
    bh, bw = board.shape
    assert len(objects) <= 13 and sum(len(s) for c, s in objects) <= 85
    for attempt in range(7):
        fallback = attempt == 6
        height = 30 if fallback else int(rng.integers(max(14, bh+1), 31))
        width = 30 if fallback else int(rng.integers(max(14, bw+1), 31))
        r0 = 0 if fallback else int(rng.integers(height-bh+1))
        c0 = 0 if fallback else int(rng.integers(width-bw+1))
        grid = np.full((height, width), background, np.int8)
        grid[r0:r0+bh, c0:c0+bw] = board
        blocked = np.zeros(grid.shape, bool)
        blocked[max(0, r0-1):r0+bh+1, max(0, c0-1):c0+bw+1] = True
        if fallback:
            # Thirteen non-touching 3x3 slots outside every <=26x26 board.
            slots = [(27, c) for c in range(0, 25, 4)]
            slots += [(r, 27) for r in range(0, 21, 4)]
            rng.shuffle(slots)
            for (color, shape), (r, c) in zip(objects, slots):
                assert not blocked[r:r+3, c:c+3].any()
                _paint_object(grid, blocked, color, shape, r, c)
        else:
            order = list(objects)
            rng.shuffle(order)
            order.sort(key=lambda v: -len(v[1]))
            if not all(_place(grid, blocked, c, s, rng) for c, s in order):
                continue
        outside = np.ones(grid.shape, bool)
        outside[r0:r0+bh, c0:c0+bw] = False
        if np.mean(grid[outside] == background) > .55:
            return grid
    raise AssertionError('The constructive packing fallback failed')


def _make_pair(rng, palette, *, witness=False, familiar=None, forbidden_metrics=(),
               force_three=False):
    """The first witness is followed by a separate identical-board contrast."""
    for attempt in range(100):
        if witness:
            h = w = 3
            t = 1
        elif familiar is not None:
            h, w = familiar.shape
            t = int(rng.choice((1, 2), p=(.8, .2)))
        else:
            h, w = map(int, rng.choice((2, 3, 4), size=2, p=(.25, .5, .25)))
            t = int(rng.choice((1, 2), p=(.8, .2)))
        maximum = (26-t)//(h+t), (26-t)//(w+t)
        background, divider, a, b, ink = map(int, palette[:5])
        if witness:
            colors = tuple(map(int, palette[5:7]))
            left, top = colors, colors[::-1]
            counts = dict(zip(colors, (2, 1)))
        else:
            k = int(rng.integers(2, min(4, h+w-2)+1))
            colors = tuple(map(int, palette[5:5+k]))
            labels = list(colors) + list(map(int, rng.choice(colors, h+w-2-k)))
            rng.shuffle(labels)
            left, top = tuple(labels[:h-1]), tuple(labels[h-1:])
            # Condition counts on geometry, rather than letting large tiles
            # vanish through repeated rejection of oversized random metrics.
            choices = []
            cap = 3 if force_three else 2
            for values in product(range(1, cap+1), repeat=k):
                if max(values) != cap:
                    continue
                trial = dict(zip(colors, values))
                ty, tx = _coordinates(left, top, trial)
                if ty[-1]+1 <= maximum[0] and tx[-1]+1 <= maximum[1]:
                    if (ty, tx) not in forbidden_metrics:
                        choices.append(trial)
            if not choices:
                continue
            counts = choices[int(rng.integers(len(choices)))]
        ys, xs = _coordinates(left, top, counts)
        window = ys[-1]+1, xs[-1]+1
        metric = (ys, xs)
        if (window[0] > maximum[0] or window[1] > maximum[1]
                or metric in forbidden_metrics):
            continue
        nr = int(rng.integers(window[0], maximum[0]+1))
        nc = int(rng.integers(window[1], maximum[1]+1))
        origin = int(rng.integers(nr-window[0]+1)), int(rng.integers(nc-window[1]+1))
        mask = familiar.copy() if familiar is not None else _mask(rng, h, w, bool(witness))
        if mask is None:
            continue
        offsets = _offsets(mask, ys, xs)
        if witness:
            # The two end points of the top row remain fixed when the two
            # interior gap lengths exchange. They locate the glyph uniquely
            # but do not disclose the unknown intermediate row/column metric.
            if not (mask[0, 0] and mask[0, -1]) or np.any(mask[1:, 0] & mask[1:, -1]):
                continue
            visible = {(origin[0], origin[1]), (origin[0], origin[1]+xs[-1])}
            if _origins(offsets, window, (nr, nc), visible) != [origin]:
                continue
        else:
            codes = _fragments(offsets, window, (nr, nc), origin)
            if len(codes) == 0:
                continue
            sizes = np.array([int(code).bit_count() for code in codes])
            size = int(rng.choice(np.unique(sizes)))
            code = int(rng.choice(codes[sizes == size]))
            visible = {(origin[0]+y, origin[1]+x) for i, (y, x) in enumerate(offsets)
                       if code & (1 << i)}
        active = {(origin[0]+y, origin[1]+x) for y, x in offsets}
        for _ in range(50):
            motif = rng.integers(0, 2, size=(h, w))
            if motif.min() != motif.max():
                break
        else:
            continue
        ordinary = np.where(motif, b, a).astype(np.int8)
        glyph_background = int(rng.choice([c for c in (a, b) if np.any(ordinary[~mask] != c)]))
        glyph = np.where(mask, ink, glyph_background).astype(np.int8)
        board = np.full((nr*(h+t)+t, nc*(w+t)+t), divider, np.int8)
        for r in range(nr):
            for c in range(nc):
                y, x = _tile_start(r, c, h, w, t)
                board[y:y+h, x:x+w] = glyph if (r, c) in visible else ordinary
        for j, color in enumerate(top, 1):
            board[0, t+j] = color
        for i, color in enumerate(left, 1):
            board[t+i, 0] = color
        output = _fill(board, glyph, active-visible, h, w, t)
        objects = []
        for color, count in counts.items():
            for j in range(count):
                shape = _shape(rng)
                if witness and count == 2:
                    shape = (_DIAMOND, _SINGLE)[j]
                objects.append((color, shape))
        unkeyed = list(map(int, palette[5+len(colors):]))
        nd = 1 if witness else int(rng.integers(0, min(2, len(unkeyed))+1))
        for color in unkeyed[:nd]:
            count = 3 if witness else int(rng.integers(1, 5))
            count = min(count, 13-len(objects))
            for j in range(count):
                shape = objects[int(rng.integers(len(objects)))][1] if rng.random() < .5 else _shape(rng)
                objects.append((color, shape))
        # Bound total ink without changing counts; this is chosen before packing.
        for i in range(len(objects)-1, -1, -1):
            if sum(len(s) for c, s in objects) <= 85:
                break
            objects[i] = objects[i][0], _SINGLE
        grid = _scene(rng, board, background, objects)
        pair = {'input': grid, 'output': output}
        decoded = _decode(grid)
        assert decoded['counts'] == counts
        assert decoded['origin'] == origin and decoded['ys'] == ys and decoded['xs'] == xs
        assert np.array_equal(decoded['expected'], output)
        return pair, decoded
    return None


def _swap_witness(rng, palette, reference):
    """Hold the whole cropped input board fixed; change only the inventory."""
    counts = {c: 3-n for c, n in reference['counts'].items()}
    ys, xs = _coordinates(reference['left'], reference['top'], counts)
    offsets = _offsets(reference['mask'], ys, xs)
    origin = reference['origin']
    assert _origins(offsets, (ys[-1]+1, xs[-1]+1), reference['lattice'],
                    reference['visible']) == [origin]
    active = {(origin[0]+y, origin[1]+x) for y, x in offsets}
    output = _fill(reference['board'], reference['glyph'], active-set(reference['visible']),
                   reference['h'], reference['w'], reference['t'])
    assert not np.array_equal(output, reference['expected'])
    objects = []
    for color, count in counts.items():
        for j in range(count):
            shape = (_DIAMOND, _SINGLE)[j] if count == 2 else _shape(rng)
            objects.append((color, shape))
    # Keep the ignored category present as another visible role, with fresh
    # shapes. Its inventory cannot change the board's requested metric.
    for _ in range(int(rng.integers(1, 5))):
        objects.append((int(palette[7]), _shape(rng)))
    grid = _scene(rng, reference['board'], reference['background'], objects)
    meta = _decode(grid)
    assert meta['counts'] == counts and np.array_equal(meta['expected'], output)
    return {'input': grid, 'output': output}, meta


def _legal_counts(num_train, num_test):
    for name, n, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                or not lo <= n <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _legal_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    stable_palette = bool(rng.random() < .3)
    extrapolate = bool(rng.integers(2))
    puzzle, seen, train_meta = {'train': [], 'test': []}, set(), []
    for split, number in (('train', nt), ('test', nq)):
        for index in range(number):
            for retry in range(40):
                local_palette = palette if stable_palette or (split == 'train' and index < 2) else rng.permutation(10)
                witness = index+1 if split == 'train' and index < 2 else 0
                familiar = train_meta[0]['mask'] if witness == 2 else None
                forbidden = ()
                if split == 'test' and not extrapolate:
                    chosen = train_meta[int(rng.integers(len(train_meta)))]
                    familiar = chosen['mask']
                    forbidden = {(m['ys'], m['xs']) for m in train_meta if _key(m['mask']) == _key(familiar)}
                if witness == 2:
                    result = _swap_witness(rng, palette, train_meta[0])
                else:
                    result = _make_pair(rng, local_palette, witness=witness, familiar=familiar,
                                        forbidden_metrics=forbidden,
                                        force_three=split == 'test' and extrapolate)
                if result is not None and _key(result[0]['input']) not in seen:
                    pair, meta = result
                    break
            else:
                raise RuntimeError('Bounded sampling could not produce a distinct valid pair')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                train_meta.append(meta)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _diagonal_and_area_witness(m):
    for color in m['keys']:
        shapes = m['objects'][color]
        if len({len(s) for s in shapes}) < 2:
            continue
        for s in shapes:
            mask = np.zeros((3, 3), bool)
            for p in s:
                mask[p] = True
            if len(_components(mask, False)) > 1:
                return True
    return False


def _partial_edge(m):
    oy, ox = m['origin']
    visible = [(r-oy, c-ox) for r, c in m['visible']]
    wh, ww = m['window']
    return (min(r for r, c in visible) > 0 or max(r for r, c in visible) < wh-1
            or min(c for r, c in visible) > 0 or max(c for r, c in visible) < ww-1)


def _without_counts_differs(m):
    active = {(m['origin'][0]+int(r), m['origin'][1]+int(c)) for r, c in np.argwhere(m['mask'])}
    altered = _fill(m['board'], m['glyph'], active-set(m['visible']), m['h'], m['w'], m['t'])
    return not np.array_equal(altered, m['expected'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _legal_counts(len(puzzle['train']), len(puzzle['test']))
    seen, metas = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            m = _decode(inp)
            assert np.array_equal(out, m['expected']), 'Wrong crop or stencil completion'
            assert not np.array_equal(out, m['board'])
            h, w, t = m['h'], m['w'], m['t']
            # Only missing tile interiors change, and every copy includes its background.
            changed = np.zeros(out.shape, bool)
            for r, c in m['missing']:
                y, x = _tile_start(r, c, h, w, t)
                changed[y:y+h, x:x+w] = True
                assert np.array_equal(out[y:y+h, x:x+w], m['glyph'])
                assert np.any((out[y:y+h, x:x+w] != m['ordinary']) & ~m['mask'])
            assert np.array_equal(out[~changed], m['board'][~changed])
            # Omitting B's ink stencil and filling its window must give a different result.
            oy, ox = m['origin']
            rectangle = {(oy+r, ox+c) for r in range(m['window'][0]) for c in range(m['window'][1])}
            rectangular = _fill(m['board'], m['glyph'], rectangle, h, w, t)
            assert not np.array_equal(rectangular, out)
            metas[split].append(m)
    training = metas['train']
    assert any(_diagonal_and_area_witness(m) for m in training)
    assert any(m['distractors'] and max(len(m['objects'][c]) for c in m['distractors']) >
               max(m['counts'].values()) for m in training)
    assert any(_partial_edge(m) for m in training)
    assert any(_asymmetric(m['mask']) for m in training)
    assert any(_without_counts_differs(m) for m in training)
    assert any(len(set(np.diff(m['ys']))) > 1 or len(set(np.diff(m['xs']))) > 1 for m in training)
    # An identical-board contrast prevents solving from B's geometry alone.
    assert any(_key(a['board']) == _key(b['board']) and a['counts'] != b['counts']
               and not np.array_equal(a['expected'], b['expected'])
               for a in training for b in training)
    assert max(n for m in training for n in m['counts'].values()) == 2
    for m in metas['test']:
        if max(m['counts'].values()) == 3:
            continue
        assert any(_key(m['mask']) == _key(old['mask']) for old in training)
        assert all((m['ys'], m['xs']) != (old['ys'], old['xs'])
                   for old in training if _key(m['mask']) == _key(old['mask']))


def _puzzle_key(p):
    return tuple(tuple((_key(v['input']), _key(v['output'])) for v in p[s]) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed semantic/diversity checks, plus mutation regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    records = [_decode(pair['input']) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    assert len({_key(m['mask']) for m in records}) >= 50
    assert len({m['board'].shape for m in records}) >= 30
    assert len({(m['h'], m['w']) for m in records}) == 9
    assert {m['t'] for m in records} == {1, 2}
    assert {n for m in records for n in m['counts'].values()} == {1, 2, 3}
    assert len({(m['ys'], m['xs']) for m in records}) >= 25
    assert len({m['origin'] for m in records}) >= 8
    assert len({s for m in records for ss in m['objects'].values() for s in ss}) >= 40
    assert {m['background'] for m in records} == set(range(10))
    assert {m['divider'] for m in records} == set(range(10))
    assert any(m['t'] == 2 and m['h'] != m['w'] for m in records)
    assert any(not m['distractors'] for m in records)
    assert any(len(m['missing']) == 1 for m in records)
    assert any(2*len(m['visible']) < int(m['mask'].sum()) for m in records)
    assert any(len(m['keys']) < len(m['top'])+len(m['left']) for m in records)
    tests = [_decode(p['test'][0]['input']) for p in puzzles]
    assert any(max(m['counts'].values()) == 3 for m in tests)
    assert any(max(m['counts'].values()) == 2 for m in tests)
    for nt, nq in ((2, 1), (10, 1), (10, 4), (3, 2)):
        p = generate_puzzle(2300+nt+nq, nt, nq)
        assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal example counts accepted')
    # The decoder's count is diagonal-inclusive, and object size is irrelevant.
    mask = np.zeros((3, 3), bool)
    for p in _DIAMOND:
        mask[p] = True
    assert len(_components(mask)) == 1 and len(_components(mask, False)) == 4
    sample = puzzles[0]
    for damage in ('divider', 'background_copy'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in sample[s]]
               for s in ('train', 'test')}
        pair = bad['test'][0]
        m = _decode(pair['input'])
        if damage == 'divider':
            pair['output'][-1, -1] = m['ink']
        else:
            r, c = min(m['missing'])
            y, x = _tile_start(r, c, m['h'], m['w'], m['t'])
            dr, dc = np.argwhere((m['ordinary'] != m['glyph']) & ~m['mask'])[0]
            pair['output'][y+dr, x+dc] = m['ordinary'][dr, dc]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('Damaged completion accepted')
    m = _decode(sample['train'][0]['input'])
    damaged = sample['train'][0]['input'].copy()
    damaged[m['bounds'][0], m['bounds'][2]] = m['ink']
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('Damaged input mesh accepted')
