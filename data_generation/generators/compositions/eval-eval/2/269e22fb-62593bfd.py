"""Companionship through peepholes: register partial shapes, then sort identities.

No parent code, source bitmap, seed-dependent solver state, or learned catalogue
is used. Whole shapes in the current input supply the reference vocabulary.
"""
from collections import Counter
import copy

import numpy as np


DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return tuple(a.shape), np.asarray(a, dtype=np.uint8).tobytes()


def _tight(mask):
    rr, cc = np.nonzero(mask)
    _require(len(rr) > 0, "empty shape")
    return np.asarray(mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1], dtype=bool).copy()


def _rotkey(mask):
    return min(_key(np.rot90(mask, k)) for k in range(4))


def _d4(mask):
    variants = {}
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            a = np.rot90(base, k).copy()
            variants[_key(a)] = a
    return list(variants.values())


def _components(grid, background, excluded=None):
    visited = np.zeros(grid.shape, dtype=bool) if excluded is None else excluded.copy()
    result = []
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if visited[r, c] or color == background:
                continue
            stack, cells = [(r, c)], []
            visited[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in DIR8:
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < H and 0 <= xx < W and not visited[yy, xx]
                            and int(grid[yy, xx]) == color):
                        visited[yy, xx] = True
                        stack.append((yy, xx))
            top = min(y for y, x in cells)
            left = min(x for y, x in cells)
            mask = np.zeros((max(y for y, x in cells) - top + 1,
                             max(x for y, x in cells) - left + 1), dtype=bool)
            for y, x in cells:
                mask[y - top, x - left] = True
            result.append(dict(mask=mask, color=color, top=top, left=left, crop=None))
    return result


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    i = int(np.argmax(counts))
    _require(counts[i] * 2 > grid.size, "background must occupy more than half the input")
    return int(colors[i])


def _perimeter(h, w):
    a = np.ones((h, w), dtype=bool)
    a[1:-1, 1:-1] = False
    return a


def _crop_index(references):
    """Index exact binary submatrices; deduplicate templates, never locations.

    Entries include full windows, so a patch equal to an entire smaller reference
    is not accidentally accepted as uniquely belonging to a larger one.
    """
    variants = {}
    for ref in references:
        for mask in _d4(ref):
            variants[_key(mask)] = mask
    index = {}
    for fullkey, mask in variants.items():
        H, W = mask.shape
        for h in range(1, H + 1):
            for w in range(1, W + 1):
                for r in range(H - h + 1):
                    for c in range(W - w + 1):
                        patch = mask[r:r + h, c:c + w]
                        if not patch.any() or patch.all():
                            continue
                        index.setdefault(_key(patch), []).append((fullkey, r, c))
    return variants, index


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = _background(grid)
    excluded = np.zeros(grid.shape, dtype=bool)
    windows = []
    for component in _components(grid, bg):
        mask = component['mask']
        h, w = mask.shape
        if min(h, w) < 3 or not np.array_equal(mask, _perimeter(h, w)):
            continue
        r, c = component['top'], component['left']
        patch = grid[r + 1:r + h - 1, c + 1:c + w - 1]
        colors = set(map(int, np.unique(patch)))
        if len(colors) != 2 or bg not in colors or component['color'] in colors:
            continue  # a genuine empty ring is an ordinary whole object
        _require(not excluded[r:r + h, c:c + w].any(), "overlapping frames")
        excluded[r:r + h, c:c + w] = True
        windows.append(dict(patch=patch != bg, color=next(iter(colors - {bg})),
                            inner_top=r + 1, inner_left=c + 1,
                            marker=component['color']))
    _require(bool(windows), "no peepholes")
    markers = {o['marker'] for o in windows}
    _require(len(markers) == 1, "frames must share one marker color")
    marker = next(iter(markers))
    _require(not np.any((grid == marker) & ~excluded), "marker used outside frames")
    whole = _components(grid, bg, excluded)
    _require(bool(whole), "no whole reference shapes")
    _require(all(o['color'] != marker for o in whole), "reference uses marker color")
    variants, index = _crop_index([o['mask'] for o in whole])
    objects = list(whole)
    for window in windows:
        matches = index.get(_key(window['patch']), [])
        _require(len(matches) == 1, "peephole needs one completion and crop location")
        fullkey, r, c = matches[0]
        mask = variants[fullkey].copy()
        h, w = window['patch'].shape
        _require(h * w < mask.size, "peephole must be a proper crop")
        obj = dict(mask=mask, color=window['color'],
                   top=window['inner_top'] - r, left=window['inner_left'] - c,
                   crop=(r, c, h, w))
        _require(0 <= obj['top'] <= grid.shape[0] - mask.shape[0]
                 and 0 <= obj['left'] <= grid.shape[1] - mask.shape[1],
                 "inferred source shape leaves canvas")
        objects.append(obj)
    return grid, bg, marker, objects, whole


def _paint(grid, obj, top, bg):
    mask = obj['mask']
    h, w = mask.shape
    left = obj['left']
    _require(0 <= top <= grid.shape[0] - h and 0 <= left <= grid.shape[1] - w,
             "destination out of bounds")
    region = grid[top:top + h, left:left + w]
    _require(bool(np.all(region[mask] == bg)), "destination collision")
    region[mask] = obj['color']


def _destinations(objects, H):
    counts = Counter(_rotkey(o['mask']) for o in objects)
    return [0 if counts[_rotkey(o['mask'])] >= 2 else H - o['mask'].shape[0]
            for o in objects]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover identities and anchors using only the current visible input."""
    grid, bg, marker, objects, whole = _decode(input_grid)
    output = np.full(grid.shape, bg, dtype=np.int8)
    for obj, top in zip(objects, _destinations(objects, grid.shape[0])):
        _paint(output, obj, top, bg)
    return output


def _connected(mask, directions=DIR8):
    cells = set(map(tuple, np.argwhere(mask)))
    seen = {min(cells)}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in directions:
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                stack.append(p)
    return seen == cells


def _sample_mask(rng, limit, area=None, chiral=False):
    for _ in range(250):
        kind = int(rng.integers(7))
        if kind < 3 or area is not None:
            target = int(rng.integers(5, min(18, limit * limit) + 1)) if area is None else area
            cells = {(0, 0)}
            dirs = DIR8 if kind == 0 else ((-1, 0), (1, 0), (0, -1), (0, 1))
            while len(cells) < target:
                possible = set()
                for r, c in sorted(cells):
                    for dr, dc in dirs:
                        p = r + dr, c + dc
                        trial = cells | {p}
                        if (p not in cells and max(x for x, y in trial) - min(x for x, y in trial) < limit
                                and max(y for x, y in trial) - min(y for x, y in trial) < limit):
                            possible.add(p)
                if not possible:
                    break
                options = sorted(possible)
                cells.add(options[int(rng.integers(len(options)))])
            if len(cells) != target:
                continue
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            mask = np.zeros((max(r for r, c in cells) - r0 + 1,
                             max(c for r, c in cells) - c0 + 1), dtype=bool)
            for r, c in cells:
                mask[r - r0, c - c0] = True
        else:
            h, w = int(rng.integers(3, limit + 1)), int(rng.integers(3, limit + 1))
            mask = np.zeros((h, w), dtype=bool)
            if kind == 3:  # bent branch/comb
                mask[:, 0] = True
                mask[-1, :] = True
                mask[int(rng.integers(1, h - 1)), :int(rng.integers(2, w + 1))] = True
            elif kind == 4:  # closed ring, or a ring with one diagonal corner gap
                mask[:] = _perimeter(h, w)
                if rng.random() < .5:
                    mask[0, 0] = False
            elif kind == 5:  # cross with independently placed arms
                mask[int(rng.integers(h)), :] = True
                mask[:, int(rng.integers(w))] = True
            else:  # bent body with a purely diagonal attachment
                mask[1:, 1] = True
                mask[-1, 1:] = True
                mask[0, 0] = True
        mask = _tight(mask)
        if min(mask.shape) < 2 or not _connected(mask):
            continue
        if chiral and len(_d4(mask)) != 8:
            continue
        return mask
    return None


def _references(rng, number, limit, same_area=False):
    for _ in range(80):
        refs = []
        for j in range(number):
            area = int(refs[0].sum()) if same_area and j == 1 else None
            a = _sample_mask(rng, limit, area=area, chiral=j == 0)
            if a is None or any(_key(a) in {_key(v) for v in _d4(b)} for b in refs):
                break
            refs.append(a)
        if len(refs) != number:
            continue
        variants, index = _crop_index(refs)
        banks = {}
        for patchkey, matches in index.items():
            if len(matches) != 1:
                continue
            fullkey, r, c = matches[0]
            h, w = patchkey[0]
            if h * w < variants[fullkey].size:
                banks.setdefault(fullkey, []).append((r, c, h, w))
        if all(len(banks.get(_key(a), [])) >= 2 for b in refs for a in _d4(b)):
            return refs, banks
    return None


def _make_objects(rng, refs, banks, mode):
    turns = [int(rng.integers(4)) for _ in refs]
    objects = [dict(mask=np.rot90(a, k).copy(), source=j, crop=None)
               for j, (a, k) in enumerate(zip(refs, turns))]
    additions = []
    if mode == 'pair':
        additions = [(0, False, (turns[0] + 1) % 4), (0, True, int(rng.integers(4)))]
    elif mode == 'triplet':
        k = int(rng.integers(4))
        additions = [(0, True, k), (0, True, (k + 1) % 4),
                     (0, True, int(rng.integers(4)))]
    elif mode == 'test':
        additions = [(j, bool(rng.integers(2)), int(rng.integers(4))) for j in range(len(refs))]
        for _ in range(int(rng.integers(0, 3))):
            additions.append((int(rng.integers(len(refs))), bool(rng.integers(2)), int(rng.integers(4))))
    else:
        for _ in range(int(rng.integers(1, 5))):
            additions.append((0, bool(rng.integers(2)), int(rng.integers(4))))
        if len(refs) > 1 and rng.random() < .5:
            objects.append(dict(mask=np.rot90(refs[1], int(rng.integers(4))).copy(), source=1, crop=None))
    if len(objects) + len(additions) > 8 or len(objects) + len(additions) < 3:
        return None
    for j, flip, k in additions:
        mask = np.rot90(np.fliplr(refs[j]) if flip else refs[j], k).copy()
        crops = banks[_key(mask)]
        # Evidence is easier in training; tests emphasize smaller distinctive
        # patches, without ever accepting an ambiguous one.
        candidates = [p for p in crops if p[2] * p[3] / mask.size >= .45] if mode != 'test' else []
        if mode == 'test':
            candidates = [p for p in crops if p[2] * p[3] / mask.size <= .7]
        candidates = candidates or crops
        crop = candidates[int(rng.integers(len(candidates)))]
        objects.append(dict(mask=mask, source=j, crop=crop))
    return objects


def _footprint(obj):
    h, w = obj['mask'].shape
    if obj['crop'] is None:
        return 0, 0, h, w
    r, c, a, b = obj['crop']
    return min(0, r - 1), min(0, c - 1), max(h, r + a + 1), max(w, c + b + 1)


def _separate(a, b):
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _pack(rng, objects):
    counts = Counter(_rotkey(o['mask']) for o in objects)
    sides = [counts[_rotkey(o['mask'])] > 1 for o in objects]
    feet = [_footprint(o) for o in objects]
    needed = []
    positions = {}
    for side in (True, False):
        cursor = 0
        indices = [i for i, s in enumerate(sides) if s == side]
        for i in rng.permutation(indices):
            i = int(i)
            f = feet[i]
            positions[i] = cursor - f[1]
            cursor += f[3] - f[1] + int(rng.choice([1, 1, 1, 2, 3]))
        needed.append(max(0, cursor - 1))
    minW = max(needed)
    if minW > 30:
        return False
    W = int(rng.integers(minW, min(30, minW + 7) + 1))
    maxheight = max(f[2] - f[0] for f in feet)
    minH = max(12, 2 * maxheight + 3)
    if minH > 30:
        return False
    H = int(rng.integers(minH, 31))
    for side, need in zip((True, False), needed):
        offset = int(rng.integers(W - need + 1))
        for i, s in enumerate(sides):
            if s == side:
                objects[i]['left'] = positions[i] + offset
    candidates = [list(map(int, rng.permutation(np.arange(-f[0], H - f[2] + 1)))) for f in feet]
    order = sorted(range(len(objects)), key=lambda i: -(feet[i][2] - feet[i][0]) * (feet[i][3] - feet[i][1]))
    boxes, rows, budget = [], {}, [2500]

    def place(step):
        if step == len(order):
            return True
        i = order[step]
        f, x = feet[i], objects[i]['left']
        for y in candidates[i]:
            budget[0] -= 1
            if budget[0] < 0:
                return False
            box = (y + f[0], x + f[1], y + f[2], x + f[3])
            if not all(_separate(box, b) for b in boxes):
                continue
            boxes.append(box)
            rows[i] = y
            if place(step + 1):
                return True
            boxes.pop()
        return False

    if not place(0):
        # Opposite source edges guarantee fit using the separated destination
        # groups. Frames can protrude one cell beyond the hidden bounding box.
        rows = {i: H - feet[i][2] if sides[i] else -feet[i][0] for i in range(len(objects))}
    for i, o in enumerate(objects):
        o['top'] = rows[i]
    return H, W


def _render(rng, objects, shape, bg, marker, mode):
    palette = [int(c) for c in rng.permutation(10) if c not in (bg, marker)]
    ncolors = int(rng.integers(1, min(8, len(objects)) + 1))
    for o in objects:
        o['color'] = int(rng.choice(palette[:ncolors]))
    if mode == 'pair':
        objects[0]['color'] = objects[-1]['color'] = palette[0]
        objects[-2]['color'] = palette[1]
    inp = np.full(shape, bg, dtype=np.int8)
    for o in objects:
        if o['crop'] is None:
            _paint(inp, o, o['top'], bg)
        else:
            r, c, h, w = o['crop']
            y, x = o['top'] + r - 1, o['left'] + c - 1
            rectangle = inp[y:y + h + 2, x:x + w + 2]
            rectangle[_perimeter(h + 2, w + 2)] = marker
            rectangle[1:-1, 1:-1] = np.where(o['mask'][r:r + h, c:c + w], o['color'], bg)
    if int(np.sum(inp == bg)) * 2 <= inp.size:
        return None
    out = np.full(shape, bg, dtype=np.int8)
    for o, top in zip(objects, _destinations(objects, shape[0])):
        _paint(out, o, top, bg)
    return dict(input=inp, output=out)


def _pair(rng, mode, marker, stable_bg):
    for attempt in range(120):
        number = 2 if mode in ('pair', 'triplet') else int(rng.integers(2, 4)) if mode == 'test' else int(rng.integers(1, 4))
        limit = 4 if mode == 'triplet' or number == 3 else int(rng.integers(3, 7))
        if attempt >= 25:
            limit = 3
        result = _references(rng, number, limit, same_area=mode == 'pair')
        if result is None:
            continue
        refs, banks = result
        objects = _make_objects(rng, refs, banks, mode)
        if objects is None:
            continue
        shape = _pack(rng, objects)
        if shape is False:
            continue
        bg = stable_bg if stable_bg is not None else int(rng.choice([c for c in range(10) if c != marker]))
        pair = _render(rng, objects, shape, bg, marker, mode)
        if pair is None:
            continue
        features = _validate_pair(pair)
        if mode == 'pair' and not (features['registration'] and features['raw_count_changes']):
            continue
        if mode in ('pair', 'triplet') and not (features['moving_up'] and features['moving_down']):
            continue
        return pair
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    stable_bg = int(rng.choice([c for c in range(10) if c != marker])) if rng.random() < .25 else None
    puzzle, seen = {'train': [], 'test': []}, set()
    modes = ['pair', 'triplet'] + ['free'] * (nt - 2)
    rng.shuffle(modes)
    for split, schedule in (('train', modes), ('test', ['test'] * nq)):
        for mode in schedule:
            for _ in range(20):
                pair = _pair(rng, mode, marker, stable_bg)
                key = _key(pair['input'])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("distinct input budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(obj, top):
    return obj['color'], int(top), obj['left'], _key(obj['mask'])


def _validate_pair(pair):
    inp, bg, marker, objects, whole = _decode(pair['input'])
    out = _grid(pair['output'])
    assert out.shape == inp.shape
    assert 3 <= len(objects) <= 8
    assert not np.any(out == marker)
    assert not np.array_equal(inp, out)
    assert np.array_equal(solve(inp), out)
    destinations = _destinations(objects, inp.shape[0])
    expected = sorted(_signature(o, d) for o, d in zip(objects, destinations))
    actual = sorted(_signature(o, o['top']) for o in _components(out, bg))
    assert expected == actual, "completion, color, orientation, column, or edge changed"
    # Exact occupied-cell clearance is also checked on implied full input shapes.
    for source in (True, False):
        occupied = set()
        for i, o in enumerate(objects):
            top = o['top'] if source else destinations[i]
            cells = {(top + int(r), o['left'] + int(c)) for r, c in np.argwhere(o['mask'])}
            halo = {(r + dr, c + dc) for r, c in occupied for dr, dc in ((0, 0),) + DIR8}
            assert not cells & halo, "distinct complete shapes touch"
            occupied |= cells
    counts = Counter(_rotkey(o['mask']) for o in objects)
    raw_masks = [o['mask'] if o['crop'] is None else
                 _tight(o['mask'][o['crop'][0]:o['crop'][0] + o['crop'][2],
                                  o['crop'][1]:o['crop'][1] + o['crop'][3]]) for o in objects]
    rawcounts = Counter(_rotkey(a) for a in raw_masks)
    up = [counts[_rotkey(o['mask'])] > 1 for o in objects]
    moves = [o['top'] != d for o, d in zip(objects, destinations)]
    result = dict(shape=tuple(inp.shape), marker=marker, background=bg, objects=len(objects),
                  whole=len(whole), windows=len(objects) - len(whole),
                  multiplicities=sorted(counts.values()),
                  registration=any(o['crop'] is not None and o['crop'][1] > 0 for o in objects),
                  raw_count_changes=any((rawcounts[_rotkey(a)] > 1) != u for a, u in zip(raw_masks, up)),
                  full_singleton=any(o['crop'] is None and not u for o, u in zip(objects, up)),
                  moving_up=any(m and u for m, u in zip(moves, up)),
                  moving_down=any(m and not u for m, u in zip(moves, up)),
                  quarter=False, mirror_negative=False, cross_color=False, different_color=False,
                  same_area=False, active_sources=0)
    active = set()
    for o in objects:
        if o['crop'] is not None:
            active.add(min(_key(a) for a in _d4(o['mask'])))
    result['active_sources'] = len(active)
    for i, a in enumerate(objects):
        for j, b in enumerate(objects[i + 1:], i + 1):
            same = _rotkey(a['mask']) == _rotkey(b['mask'])
            result['different_color'] |= same and a['color'] != b['color']
            result['cross_color'] |= up[i] != up[j] and a['color'] == b['color']
            result['quarter'] |= same and _key(a['mask']) != _key(b['mask']) and any(
                np.array_equal(np.rot90(a['mask'], k), b['mask']) for k in (1, 3))
            result['mirror_negative'] |= (not same and _rotkey(np.fliplr(a['mask'])) == _rotkey(b['mask'])
                                          and up[i] != up[j])
            result['same_area'] |= not same and a['mask'].sum() == b['mask'].sum() and up[i] != up[j]
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    features, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp = _grid(pair['input'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            features[split].append(_validate_pair(pair))
    allfeatures = features['train'] + features['test']
    assert len({f['marker'] for f in allfeatures}) == 1
    train = features['train']
    for witness in ('registration', 'raw_count_changes', 'full_singleton', 'quarter',
                    'mirror_negative', 'cross_color', 'different_color', 'same_area',
                    'moving_up', 'moving_down'):
        assert any(f[witness] for f in train), f"missing training evidence: {witness}"
    assert any(2 in f['multiplicities'] for f in train)
    assert any(3 in f['multiplicities'] for f in train)
    assert all(f['active_sources'] == 1 for f in train)
    assert all(f['active_sources'] >= 2 for f in features['test'])


def self_test():
    """Bounded actual-grid checks over 42 seeds, plus API and corruption checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    def signature(p):
        return tuple((_key(a['input']), _key(a['output'])) for s in ('train', 'test') for a in p[s])
    assert signature(puzzles[0]) == signature(generate_puzzle(0))
    assert len({signature(p) for p in puzzles}) == 42
    rows = [_validate_pair(a) for p in puzzles for s in ('train', 'test') for a in p[s]]
    assert len({f['shape'] for f in rows}) >= 35
    assert len({f['marker'] for f in rows}) == 10
    assert len({f['background'] for f in rows}) == 10
    assert len({f['objects'] for f in rows}) >= 5
    assert len({f['windows'] for f in rows}) >= 4
    references, crops, offsets, extents, areas = set(), set(), set(), set(), set()
    diagonal = False
    for p in puzzles:
        for split in ('train', 'test'):
            for pair in p[split]:
                grid, bg, marker, objects, whole = _decode(pair['input'])
                references.update(min(_key(a) for a in _d4(o['mask'])) for o in whole)
                for o in objects:
                    extents.add(o['mask'].shape)
                    areas.add(int(o['mask'].sum()))
                    diagonal |= not _connected(o['mask'], ((-1, 0), (1, 0), (0, -1), (0, 1)))
                    if o['crop']:
                        r, c, h, w = o['crop']
                        crops.add((h, w))
                        offsets.add((r, c))
    assert len(references) >= 100 and len(crops) >= 15 and len(offsets) >= 10
    assert len(areas) >= 10 and max(max(s) for s in extents) >= 6 and diagonal
    # The host exhausts all legal count combinations. Keep this self-test
    # bounded to the four count boundaries in addition to its 42-seed corpus.
    for nt, nq in ((2, 1), (5, 4), (10, 1), (10, 4)):
        generate_puzzle(500 + nt * 10 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = copy.deepcopy(puzzles[0])
    out = bad['test'][0]['output']
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=len(rows), dimensions=len({f['shape'] for f in rows}),
                reference_classes=len(references), crop_shapes=len(crops), crop_offsets=len(offsets),
                object_counts=sorted({f['objects'] for f in rows}))
