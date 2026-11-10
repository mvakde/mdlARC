"""Scaffold docks: plus recognition supplies obstacles for stable crate packing.

Only standard library and NumPy. All semantic roles are visible in the input.
Validation uses a generator-owned unit-step audit, not independent approval.
"""
from __future__ import annotations

import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SEEDS = (0, 921410210, 308799516, 449426960, 106467288, 819531890,
          213819472, 560860962, 749044294, 473916434, 502012063, 797051089,
          750338766, 282116239, 512727541, 157443146, 6464573, 955434839,
          53717339, 491883022, 633524806, 634057370, 484669391, 508971428,
          392639071, 416524910, 577234341, 305366880, 530259501, 599860018,
          426228335, 927785912, 876754124, 41285386, 382186289, 940057383,
          151232400, 211259106, 861785946, 524808676, 585880125, 979402343)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"invalid {name}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _canonical(grid):
    g = np.asarray(grid)
    if (g.ndim != 2 or not np.issubdtype(g.dtype, np.integer)
            or not all(5 <= n <= 30 for n in g.shape)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("invalid enclosed grid")
    s = int(g[0, 0])
    horizontal = np.all(g[0] == s) and np.all(g[-1] == s)
    vertical = np.all(g[:, 0] == s) and np.all(g[:, -1] == s)
    if horizontal == vertical:
        raise ValueError("enclosure must identify exactly one movement axis")
    axis = 1 if horizontal else 0
    a = g.copy() if horizontal else g.T.copy()
    low, high = int(a[1, 0]), int(a[1, -1])
    if (len({s, low, high}) != 3 or not np.all(a[1:-1, 0] == low)
            or not np.all(a[1:-1, -1] == high)):
        raise ValueError("invalid destination walls")
    values, counts = np.unique(a[1:-1, 1:-1], return_counts=True)
    bg = int(values[counts.argmax()])
    if counts.max() * 2 <= a[1:-1, 1:-1].size or bg in (s, low, high):
        raise ValueError("interior background must be a strict majority")
    return a, axis, (bg, s, low, high)


def _decode(g, roles):
    bg, s, low, high = roles
    h, w = g.shape
    covered = (g == bg) | (g == s)
    covered[[0, -1], :] = True
    covered[:, [0, -1]] = True
    objects = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if covered[r, c]:
                continue
            f = int(g[r, c])
            if f not in (low, high) or r + 2 >= h - 1 or c + 2 >= w - 1:
                raise ValueError("unframed payload or clipped crate")
            bottom, right = r + 2, c + 2
            while bottom < h - 1 and g[bottom, c + 1] != f:
                bottom += 1
            while right < w - 1 and g[r + 1, right] != f:
                right += 1
            if bottom >= h - 1 or right >= w - 1:
                raise ValueError("open crate frame")
            patch = g[r:bottom + 1, c:right + 1]
            inner = patch[1:-1, 1:-1]
            if (not np.all(patch[0] == f) or not np.all(patch[-1] == f)
                    or not np.all(patch[:, 0] == f) or not np.all(patch[:, -1] == f)
                    or np.any(np.isin(inner, (bg, s, f)))
                    or np.any(covered[r:bottom + 1, c:right + 1])):
                raise ValueError("ambiguous or overlapping crate")
            covered[r:bottom + 1, c:right + 1] = True
            objects.append(dict(r=r, c=c, h=bottom-r+1, w=right-c+1,
                                d=int(f == high), frame=f, patch=patch.copy()))
    if not objects or not covered.all():
        raise ValueError("scene must contain recognizable crates")
    return objects


def _scaffold(g, s):
    mask = g == s
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    return mask


def _union(mask):
    centers = np.zeros_like(mask)
    centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                             & mask[2:, 1:-1] & mask[1:-1, :-2]
                             & mask[1:-1, 2:])
    cells = centers.copy()
    cells[1:] |= centers[:-1]
    cells[:-1] |= centers[1:]
    cells[:, 1:] |= centers[:, :-1]
    cells[:, :-1] |= centers[:, 1:]
    return centers, cells


def _scan(mask, destructive=False):
    """Direct scanner for audits and the destructive alternative."""
    original = mask.copy()
    result = np.zeros_like(mask)
    count = 0
    for r in range(1, mask.shape[0]-1):
        for c in range(1, mask.shape[1]-1):
            pts = [(r, c)] + [(r+dr, c+dc) for dr, dc in _DIRS]
            if all(original[p] for p in pts):
                count += 1
                for p in pts:
                    result[p] = True
                    if destructive:
                        original[p] = False
    return count, result


def _band(a, b):
    return max(a['r'], b['r']) < min(a['r']+a['h'], b['r']+b['h'])


def _overlap(a, b):
    return _band(a, b) and max(a['c'], b['c']) < min(a['c']+a['w'], b['c']+b['w'])


def _pack(objects, docks, first_slot=False, check=True, global_layers=False):
    """Reachable anchor bound plus stable predecessor constraints."""
    width = docks.shape[1]
    result = [dict(o) for o in objects]
    for direction in (0, 1):
        indices = [i for i, o in enumerate(objects) if o['d'] == direction]
        indices.sort(key=lambda i: (objects[i]['c'] if direction == 0
                                    else -objects[i]['c']-objects[i]['w'],
                                    objects[i]['r']))
        settled = []
        for i in indices:
            src, o = objects[i], result[i]
            columns = np.flatnonzero(docks[o['r']:o['r']+o['h']].any(axis=0))
            if direction == 0:
                before = columns[columns < src['c']]
                bound = max(1, int(before.max())+1 if len(before) else 1)
                if first_slot:
                    options = range(bound, src['c']+1)
                else:
                    o['c'] = max([bound] + [p['c']+p['w'] for p in settled if global_layers or _band(o, p)])
            else:
                after = columns[columns >= src['c']+src['w']]
                bound = min(width-1, int(after.min()) if len(after) else width-1)-o['w']
                if first_slot:
                    options = range(bound, src['c']-1, -1)
                else:
                    o['c'] = min([bound] + [p['c']-o['w'] for p in settled if global_layers or _band(o, p)])
            if first_slot:
                for c in options:
                    o['c'] = c
                    if not any(_overlap(o, p) for p in settled):
                        break
            if (o['c'] > src['c'] if direction == 0 else o['c'] < src['c']):
                raise ValueError("packing moved away from destination")
            settled.append(o)
    if check:
        _check_footprints(result, docks)
    return result


def _check_footprints(objects, docks):
    h, w = docks.shape
    for i, o in enumerate(objects):
        r, c, a, b = (o[k] for k in ('r', 'c', 'h', 'w'))
        if (not (1 <= r and r+a <= h-1 and 1 <= c and c+b <= w-1)
                or docks[r:r+a, c:c+b].any()
                or any(_overlap(o, p) for p in objects[i+1:])):
            raise ValueError("final footprint collision")


def _render(shape, roles, scaffold, objects):
    bg, s, low, high = roles
    g = np.full(shape, bg, dtype=np.int64)
    g[scaffold] = s
    g[0] = g[-1] = s
    g[1:-1, 0] = low
    g[1:-1, -1] = high
    for o in objects:
        g[o['r']:o['r']+o['h'], o['c']:o['c']+o['w']] = o['patch']
    return g


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive axis, roles, motifs, footprints, and movement from this input."""
    g, axis, roles = _canonical(input_grid)
    objects = _decode(g, roles)
    _, docks = _union(_scaffold(g, roles[1]))
    result = _render(g.shape, roles, docks, _pack(objects, docks))
    return result if axis == 1 else result.T.copy()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _appearance(rng, objects, roles, shared=False):
    bg, s, low, high = roles
    same = None
    for i, o in enumerate(objects):
        frame = (low, high)[o['d']]
        colors = [c for c in range(10) if c not in (bg, s, frame)]
        if shared:
            colors = [c for c in colors if c not in (low, high)]
        shape = (o['h']-2, o['w']-2)
        payload = np.full(shape, _pick(rng, colors), dtype=np.int64)
        if payload.size > 1 and rng.random() < .7:
            selected = rng.choice(colors, size=int(rng.integers(2, min(4, len(colors))+1)), replace=False)
            rr, cc = np.indices(shape)
            mode = int(rng.integers(4))
            index = (rr+cc if mode == 0 else rr if mode == 1 else cc)
            if mode == 3:
                index = rng.integers(len(selected), size=shape)
            payload = selected[index % len(selected)].astype(np.int64)
        if shared and i < 2:
            if same is None:
                same = payload.copy()
            payload = same.copy()
        o['frame'] = frame
        o['patch'] = np.full((o['h'], o['w']), frame, dtype=np.int64)
        o['patch'][1:-1, 1:-1] = payload


def _points(mask, pts, forbidden=None):
    for r, c in pts:
        if 1 <= r < mask.shape[0]-1 and 1 <= c < mask.shape[1]-1:
            if forbidden is None or not forbidden[r, c]:
                mask[r, c] = True


def _plus(r, c):
    return [(r, c)] + [(r+dr, c+dc) for dr, dc in _DIRS]


def _occupied(shape, objects):
    mask = np.zeros(shape, dtype=bool)
    for o in objects:
        mask[o['r']:o['r']+o['h'], o['c']:o['c']+o['w']] = True
    return mask


def _witness(rng, roles, kind):
    if kind == 'dock':
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 7))
        height = int(rng.integers(max(11, h+7), 25))
        width = int(rng.integers(3*w+7, 31))
        c = int(rng.integers(w+4, min(width-2*w-1, width-w-5)))
        # Only the bottom arm intersects the left mover's row span. This
        # distinguishes real motif pixels from their 3x3 bounding rectangle.
        row = int(rng.integers(3, height-h-1))
        r = row-1
        objects = [dict(r=row, c=int(rng.integers(c+5, width-w)), h=h, w=w, d=0),
                   dict(r=row+1, c=int(rng.integers(1, c-w-1)), h=h, w=w, d=1)]
        mask = np.zeros((height, width), dtype=bool)
        _points(mask, _plus(r, c) + [(r-1, c-1)])
        _points(mask, [(row, c+2), (row, c+3), (row, c+4), (row-1, c+3)])
        _appearance(rng, objects, roles, shared=True)
    else:
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        step = int(rng.integers((h+1)//2, h))
        margin = int(rng.integers(1, 4))
        first_r = margin + 2*step
        independent_h = int(rng.integers(3, 6))
        independent_w = int(rng.integers(3, w+1))
        height = int(rng.integers(first_r+h+independent_h+2, 31))
        pin = int(rng.integers(2, 5))
        width = int(rng.integers(pin+4+3*w+1, 31))
        spare = width-1-(pin+4+3*w)
        gaps = rng.multinomial(spare, np.full(4, .25))
        col = pin+4+int(gaps[0])
        objects = []
        for i in range(3):
            objects.append(dict(r=first_r-i*step, c=col, h=h, w=w, d=0))
            col += w+int(gaps[i+1])
        # A separate row band resets to the wall despite being farthest from
        # it. It rules out treating every same-class crate as a global layer.
        objects.append(dict(r=int(rng.integers(first_r+h+1, height-independent_h)),
                            c=int(rng.integers(objects[-1]['c'], width-independent_w)),
                            h=independent_h, w=independent_w, d=0))
        mask = np.zeros((height, width), dtype=bool)
        center_r = first_r+h-1
        mask[center_r-1:center_r+2, pin-1:pin+3] = True
        if rng.random() < .5:
            mask[center_r, 1:pin-1] = True
        if rng.random() < .5:
            mask = mask[:, ::-1].copy()
            for o in objects:
                o['c'] = width-o['c']-o['w']
                o['d'] = 1
        _appearance(rng, objects, roles)
    if rng.random() < .5:
        mask = mask[::-1].copy()
        for o in objects:
            o['r'] = height-o['r']-o['h']
            o['patch'] = o['patch'][::-1].copy()
    return _render(mask.shape, roles, mask, objects)


def _free(rng, roles, shift, test, richness):
    if test and shift == 'sparse':
        height, width = int(rng.integers(8, 18)), int(rng.integers(9, 22))
        requested = 1
    elif test and shift == 'expansion':
        height, width = tuple(map(int, rng.integers(20, 31, size=2)))
        requested = int(rng.integers(4, 9))
    else:
        height, width = int(rng.integers(9, 31)), int(rng.integers(11, 31))
        requested = int(rng.integers(2 if test else 1, 7))
    objects = []
    area = 0
    interior_area = (height-2)*(width-2)
    prob_low = _pick(rng, (0., .15, .5, .5, .85, 1.))
    for _ in range(requested):
        for attempt in range(60):
            h, w = tuple(map(int, rng.integers(3, 8, size=2)))
            if h > height-2 or w > width-2 or area+h*w > .34*interior_area:
                continue
            r, c = int(rng.integers(1, height-h)), int(rng.integers(1, width-w))
            if objects and rng.random() < .45:
                other = _pick(rng, objects)
                r = _pick(rng, (other['r'], other['r']+other['h']-1,
                               other['r']-h+1, other['r']+other['h']))
                if not 1 <= r <= height-h-1:
                    continue
            if rng.random() < .2:
                c = _pick(rng, (1, width-w-1))
            o = dict(r=r, c=c, h=h, w=w, d=int(rng.random() >= prob_low))
            if not any(_overlap(o, p) for p in objects):
                objects.append(o)
                area += h*w
                break
    if not objects:
        objects = [dict(r=1, c=2, h=3, w=3, d=int(rng.integers(2)))]
        area = 9
    blocked = _occupied((height, width), objects)
    mask = np.zeros((height, width), dtype=bool)
    positive = rng.random() >= .12
    # Complete stamps, overlapping centers, and attached geometry all coexist.
    for _ in range(int(rng.integers(1, 8)) if positive else 0):
        for attempt in range(25):
            r, c = int(rng.integers(2, height-2)), int(rng.integers(2, width-2))
            pts = _plus(r, c)
            if not any(blocked[p] for p in pts):
                _points(mask, pts)
                if rng.random() < .35:
                    _points(mask, _plus(r, c+int(rng.integers(1, 3))), blocked)
                for dr, dc in _DIRS:
                    if rng.random() < .25:
                        _points(mask, [(r+dr*k, c+dc*k) for k in range(2, int(rng.integers(3, 7)))], blocked)
                if rng.random() < .5:
                    _points(mask, [(r-1, c-1), (r+1, c+1)], blocked)
                break
    for _ in range(int(rng.integers(0, 4+richness*5))):
        r, c = int(rng.integers(1, height-1)), int(rng.integers(1, width-1))
        kind = int(rng.integers(6))
        if kind == 0:
            pts = [(r, c+j) for j in range(int(rng.integers(2, 8)))]
        elif kind == 1:
            pts = [(r+j, c) for j in range(int(rng.integers(2, 8)))] + [(r, c+j) for j in range(4)]
        elif kind == 2:
            pts = [(r, c-1), (r, c), (r, c+1), (r+1, c)]
        elif kind == 3:
            pts = [(r+a, c+b) for a in (-1, 1) for b in (-1, 1)] + [(r, c)]
        elif kind == 4:
            pts = [(r+a, c+b) for a in range(3) for b in range(3) if (a, b) != (1, 1)]
        else:
            pts = [(r+a, c+b) for a in range(3) for b in range(int(rng.integers(3, 6)))]
        _points(mask, pts, blocked)
    # A finished negative has no complete plus, even if clutter made one.
    if not positive:
        centers, _ = _union(mask)
        mask[centers] = False
    available = max(0, int(.48*interior_area)-area)
    if int(mask.sum()) > available:
        pts = np.argwhere(mask)
        for i in rng.permutation(len(pts))[available:]:
            mask[tuple(pts[int(i)])] = False
    _appearance(rng, objects, roles)
    return _render(mask.shape, roles, mask, objects)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    a, b = _integer(num_train, 'num_train', 2, 10), _integer(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    low, high = palette[:2]
    initial = tuple(palette[2:4])
    axis = int(rng.integers(2))
    palette_mode = _pick(rng, ('stable', 'swap', 'free', 'free'))
    shift = _pick(rng, ('recombination', 'expansion', 'sparse'))
    richness = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', a), ('test', b)):
        for index in range(count):
            if palette_mode == 'stable':
                bg, s = initial
            elif palette_mode == 'swap':
                bg, s = initial if rng.random() < .5 else initial[::-1]
            else:
                bg, s = map(int, rng.choice(palette[2:], size=2, replace=False))
            roles = (bg, s, low, high)
            kind = 'dock' if split == 'train' and index == 0 else 'chain' if split == 'train' and index == 1 else 'free'
            for attempt in range(160):
                g = (_witness(rng, roles, kind) if kind != 'free'
                     else _free(rng, roles, shift, split == 'test', richness))
                inp = g if axis == 1 else g.T.copy()
                try:
                    out = solve(inp)
                except ValueError:
                    continue
                if _key(inp) not in seen:
                    seen.add(_key(inp))
                    puzzle[split].append({'input': inp, 'output': out})
                    break
            else:
                raise RuntimeError('bounded generation exhausted')
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(a)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _simulate(objects, docks):
    """Audit by one-cell translations, without the boundary recurrence."""
    result = [dict(o) for o in objects]
    width = docks.shape[1]
    for d in (0, 1):
        group = [o for o in result if o['d'] == d]
        for _ in range(width*max(1, len(group))+1):
            moved = False
            for o in group:
                trial = dict(o, c=o['c']+(-1 if d == 0 else 1))
                r, c, h, w = (trial[k] for k in ('r', 'c', 'h', 'w'))
                if (1 <= c <= width-w-1 and not docks[r:r+h, c:c+w].any()
                        and not any(_overlap(trial, p) for p in group if p is not o)):
                    o['c'] = c
                    moved = True
            if not moved:
                break
        else:
            raise AssertionError('sliding audit failed to converge')
    _check_footprints(result, docks)
    return result


def _features(pair):
    g, axis, roles = _canonical(pair['input'])
    objects = _decode(g, roles)
    mask = _scaffold(g, roles[1])
    centers, docks = _union(mask)
    settled = _pack(objects, docks)
    locations = [o['c'] for o in settled]
    _, seq = _scan(mask, destructive=True)
    boxes = np.zeros_like(mask)
    for r, c in np.argwhere(centers):
        boxes[r-1:r+2, c-1:c+2] = True
    try:
        global_witness = locations != [o['c'] for o in _pack(objects, docks, check=False, global_layers=True)]
    except ValueError:
        global_witness = True
    return dict(axis=axis, roles=roles, shape=g.shape, count=len(objects),
                footprints=sorted({(o['h'], o['w']) for o in objects}),
                centers=int(centers.sum()), retained=int(docks.sum()), erased=int((mask & ~docks).sum()),
                overlap=int(centers.sum())*5 > int(docks.sum()),
                simultaneous=not np.array_equal(docks, seq),
                stable=locations != [o['c'] for o in _pack(objects, docks, first_slot=True, check=False)],
                omit_A=locations != [o['c'] for o in _pack(objects, np.zeros_like(docks), check=False)],
                all_scaffold=locations != [o['c'] for o in _pack(objects, mask, check=False)],
                bounding_box=locations != [o['c'] for o in _pack(objects, boxes, check=False)],
                independent_bands=global_witness,
                crossing=any(a['d'] == 0 and b['d'] == 1 and _band(a, b)
                             and a['c'] > b['c'] and settled[i]['c'] < settled[j]['c']
                             for i, a in enumerate(objects) for j, b in enumerate(objects)),
                moved_frames={o['frame'] for o, p in zip(objects, settled) if o['c'] != p['c']},
                stationary=sum(o['c'] == p['c'] for o, p in zip(objects, settled)),
                travel=[abs(o['c']-p['c']) for o, p in zip(objects, settled)],
                patterned=sum(len(np.unique(o['patch'][1:-1, 1:-1])) > 1 for o in objects),
                shared=any(a['d'] != b['d'] and a['patch'].shape == b['patch'].shape
                           and np.array_equal(a['patch'][1:-1, 1:-1], b['patch'][1:-1, 1:-1])
                           for a in objects for b in objects),
                touching=any(_band(a, b) and (a['c']+a['w'] == b['c'] or b['c']+b['w'] == a['c'])
                             for i, a in enumerate(objects) for b in objects[i+1:]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, conventions = set(), set()
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            g, axis, roles = _canonical(inp)
            conventions.add((axis, roles[2:]))
            objects = _decode(g, roles)
            mask = _scaffold(g, roles[1])
            count, docks = _scan(mask)
            cc, uu = _union(mask)
            assert count == int(cc.sum()) and np.array_equal(docks, uu)
            settled = _simulate(objects, docks)
            actual = out if axis == 1 else out.T
            assert np.array_equal(actual, _render(g.shape, roles, docks, settled)), 'incorrect dock extraction or rigid packing'
            assert np.array_equal(actual[0], g[0]) and np.array_equal(actual[-1], g[-1])
            assert np.array_equal(actual[:, 0], g[:, 0]) and np.array_equal(actual[:, -1], g[:, -1])
            assert np.array_equal(_scaffold(actual, roles[1]), docks), 'nonmotif scaffold survived'
            for old, new in zip(objects, settled):
                assert old['r'] == new['r'] and old['h'] == new['h'] and old['w'] == new['w']
                assert np.array_equal(actual[new['r']:new['r']+new['h'], new['c']:new['c']+new['w']], old['patch'])
            features[split].append(_features(pair))
    assert len(conventions) == 1
    train = features['train']
    frames = next(iter(conventions))[1]
    assert set().union(*(f['moved_frames'] for f in train)) == set(frames)
    for feature in ('omit_A', 'all_scaffold', 'bounding_box', 'independent_bands',
                    'crossing', 'shared', 'simultaneous', 'stable'):
        assert any(f[feature] for f in train), f'missing training witness: {feature}'
    assert any(f['erased'] > 0 and f['centers'] > 0 for f in train)


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def _normalized(grid):
    roles = {}
    return tuple(roles.setdefault(int(c), len(roles)) for c in grid.flat)


def self_test():
    """Bounded deterministic 42-seed family audit; returns descriptive metrics."""
    puzzles = [generate_puzzle(seed) for seed in _SEEDS]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    rows = [_features(pair) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    shapes = {f['shape'] for f in rows}
    structures = {(_key(pair['input'])[0], _normalized(pair['input']))
                  for p in puzzles for s in ('train', 'test') for pair in p[s]}
    assert {f['axis'] for f in rows} == {0, 1}
    assert len(shapes) >= 45 and any(30 in shape for shape in shapes)
    assert len(structures) >= 180
    assert len({f['count'] for f in rows}) >= 5
    assert len({tuple(f['footprints']) for f in rows}) >= 35
    assert len({f['roles'] for f in rows}) >= 70
    assert any(f['count'] == 1 for f in rows)
    assert any(f['centers'] == 0 for f in rows)
    assert any(f['stationary'] for f in rows) and any(f['touching'] for f in rows)
    assert any(f['patterned'] for f in rows)
    assert any(len(f['footprints']) > 1 for f in rows)
    assert all(any(_features(pair)['omit_A'] for pair in p['train']) for p in puzzles)
    # Exhaust every binary 3x3 neighborhood and the shared-center regression.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        assert np.array_equal(_union(mask)[1], _scan(mask)[1])
    block = np.ones((3, 4), dtype=bool)
    assert _scan(block)[0] == 2 and _union(block)[1].sum() == 8
    assert _scan(block, destructive=True)[1].sum() == 5
    clipped = np.zeros((5, 6), dtype=bool)
    clipped[0, 1:4] = True
    clipped[1, 2] = clipped[-1, 2] = True
    assert not _union(clipped)[1].any()
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(17, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    # Reproduce an omitted cell caused by destructive overlap recognition.
    index = next(i for i, pair in enumerate(corrupt['train']) if _features(pair)['simultaneous'])
    source, axis, roles = _canonical(corrupt['train'][index]['input'])
    mask = _scaffold(source, roles[1])
    missing = _union(mask)[1] & ~_scan(mask, destructive=True)[1]
    r, c = map(int, np.argwhere(missing)[0])
    output = corrupt['train'][index]['output']
    if axis == 1:
        output[r, c] = roles[0]
    else:
        output[c, r] = roles[0]
    try:
        validate_generated_puzzle(corrupt)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupt label accepted')
    return dict(seeds=42, pairs=len(rows), shapes=len(shapes),
                normalized_input_structures=len(structures),
                counts=sorted({f['count'] for f in rows}),
                footprints=sorted({shape for f in rows for shape in f['footprints']}),
                axes=sorted({f['axis'] for f in rows}),
                conventions=len({(f['axis'], f['roles'][2:]) for f in rows}),
                role_assignments=len({f['roles'] for f in rows}),
                no_plus=sum(f['centers'] == 0 for f in rows),
                mixed_footprints=sum(len(f['footprints']) > 1 for f in rows),
                simultaneous=sum(f['simultaneous'] for f in rows),
                stable_witnesses=sum(f['stable'] for f in rows),
                dock_changes_answer=sum(f['omit_A'] for f in rows),
                bounding_box_witnesses=sum(f['bounding_box'] for f in rows),
                independent_band_witnesses=sum(f['independent_bands'] for f in rows),
                crossing_pairs=sum(f['crossing'] for f in rows),
                max_travel=max(max(f['travel']) for f in rows))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
