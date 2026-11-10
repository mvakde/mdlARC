"""Restore rectangular fragments against a hollow-cross pinwheel coordinate pattern.

Only the standard library and NumPy are used. No parent modules or data are read.
"""
from __future__ import annotations

from collections import Counter
import json
import numpy as np


def _key(a):
    a = np.ascontiguousarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('Expected a 1..30 by 1..30 integer color grid')
    return a.astype(np.int8, copy=True)


def _count(value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError('Illegal example count')
    return int(value)


def _cross(color_mask):
    n = int(color_mask.sum())
    if n % 4 or not 1 <= n // 4 <= 6:
        raise ValueError('Not a hollow cross')
    length = n // 4
    ys = np.flatnonzero(color_mask.sum(axis=1) == 2 * length)
    xs = np.flatnonzero(color_mask.sum(axis=0) == 2 * length)
    if len(ys) != 1 or len(xs) != 1:
        raise ValueError('Not a unique hollow cross')
    y, x = int(ys[0]), int(xs[0])
    if min(y, x, color_mask.shape[0]-1-y, color_mask.shape[1]-1-x) < length:
        raise ValueError('Clipped seed')
    expected = np.zeros_like(color_mask)
    for d in range(1, length+1):
        expected[y-d, x] = expected[y+d, x] = True
        expected[y, x-d] = expected[y, x+d] = True
    if not np.array_equal(expected, color_mask):
        raise ValueError('Extraneous seed pixels')
    return length, (y, x)


def _pinwheel(length, shape, center):
    """Input-only closed-form membership, including later in-bounds re-entry.

    Test line intervals in the four rotated north-arm coordinate systems.
    Each four-stage cycle moves the starts outward by eight cells.
    """
    radius = max(2, length)
    yy, xx = np.indices(shape)
    yy, xx = yy-center[0], xx-center[1]
    bound = max(int(np.abs(yy).max()), int(np.abs(xx).max()))
    result = np.zeros(shape, dtype=bool)
    for _ in range(4):
        result |= (xx == 0) & (-radius <= yy) & (yy <= -1)
        for q in range(bound//8+3):
            starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                      (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
            for s, ((y, x), (dy, dx)) in enumerate(zip(
                    starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along = (yy-y)*dy+(xx-x)*dx
                across = (yy-y)*dx-(xx-x)*dy
                run = 16*q+4*s+2
                result |= (across == 0) & (0 <= along) & (along <= run)
                result |= (along == run) & (0 <= across) & (across <= radius-2)
                result |= (across == radius-2) & (run <= along) & (along <= run+radius)
        yy, xx = xx, -yy
    return result


def _walk_pinwheel(length, shape, center):
    """Generation cross-check: explicitly walk steps, without interval formulas."""
    result = np.zeros(shape, dtype=bool)
    radius = max(2, length)
    bound = max(*shape)+max(abs(v) for v in center)

    def mark(y, x):
        y, x = y+center[0], x+center[1]
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            result[y, x] = True

    for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        y, x = 0, 0
        for _ in range(radius):
            y, x = y+dy, x+dx
            mark(y, x)
        dy, dx = dx, -dy
        for stage in range(4*(bound//8+3)):
            for vy, vx, steps in ((dy, dx, 4*stage+2),
                                   (dx, -dy, radius-2), (dy, dx, radius)):
                for _ in range(steps):
                    y, x = y+vy, x+vx
                    mark(y, x)
            dy, dx = dx, -dy
    return result


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _tiles(tray, vacancy):
    tiles = []
    for cells in _components(tray != vacancy):
        ys, xs = zip(*cells)
        tile = tray[min(ys):max(ys)+1, min(xs):max(xs)+1]
        if tile.size != len(cells) or np.any(tile == vacancy):
            raise ValueError('Source fragments must be filled rectangles')
        tiles.append(tile.copy())
    if not tiles:
        raise ValueError('Empty inventory')
    return tiles


def _parse(a):
    """Find the unique isolated-color separator and semantic board, from input."""
    candidates = []
    for axis in (0, 1):
        for cut in range(1, a.shape[axis]-1):
            line = a[cut, :] if axis == 0 else a[:, cut]
            separator = int(line[0])
            if not np.all(line == separator) or int((a == separator).sum()) != line.size:
                continue
            sides = (a[:cut], a[cut+1:]) if axis == 0 else (a[:, :cut], a[:, cut+1:])
            for order in (0, 1):
                board, tray = sides[order], sides[1-order]
                colors = list(map(int, np.unique(board)))
                if len(colors) != 3:
                    continue
                for ink in colors:
                    try:
                        length, center = _cross(board == ink)
                    except ValueError:
                        continue
                    for vacancy in [c for c in colors if c != ink]:
                        background = next(c for c in colors if c not in (ink, vacancy))
                        if np.any(tray == background) or not np.any(tray == vacancy):
                            continue
                        if int(board[center]) != background:
                            continue
                        try:
                            tiles = _tiles(tray, vacancy)
                        except ValueError:
                            continue
                        if any(np.any((t == background) | (t == separator)) for t in tiles):
                            continue
                        candidates.append(dict(board=board.copy(), tray=tray.copy(),
                            tiles=tiles, mask=board == vacancy, ink=ink, vacancy=vacancy,
                            background=background, separator=separator, length=length,
                            center=center, side=(('top', 'bottom') if axis == 0 else
                                                ('left', 'right'))[order]))
    if len(candidates) != 1:
        raise ValueError('Expected one semantic board and separator')
    return candidates[0]


def _assemble(info, use_ink=True, node_limit=60000):
    """Enumerate distinct rendered exact covers; physical copies are finite.

    A return record contains the successful placement path and *all* locally
    compatible placements, allowing validation of the global-packing witness.
    """
    mask, ink = info['mask'], info['ink']
    expected = _pinwheel(info['length'], mask.shape, info['center'])
    groups, lookup = [], {}
    for t in info['tiles']:
        key = _key(t)
        if key in lookup:
            groups[lookup[key]]['count'] += 1
        else:
            lookup[key] = len(groups)
            groups.append(dict(tile=t, count=1))
    coords = np.argwhere(mask)
    index = np.full(mask.shape, -1, dtype=int)
    index[mask] = np.arange(len(coords))
    placements, by_cell = [], [[] for _ in coords]
    for gi, group in enumerate(groups):
        tile = group['tile']
        h, w = tile.shape
        motif = tile == ink
        for y in range(mask.shape[0]-h+1):
            for x in range(mask.shape[1]-w+1):
                if not np.all(mask[y:y+h, x:x+w]):
                    continue
                if use_ink and not np.array_equal(motif, expected[y:y+h, x:x+w]):
                    continue
                cells = tuple(map(int, index[y:y+h, x:x+w].flat))
                bits = sum(1 << c for c in cells)
                pi = len(placements)
                placements.append((gi, y, x, bits, cells))
                for c in cells:
                    by_cell[c].append(pi)
    remaining = [g['count'] for g in groups]
    full = (1 << len(coords))-1
    rendered = [-1]*len(coords)
    outcomes, dead = {}, set()
    path, nodes = [], 0

    def search(covered):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            raise ValueError('Unresolved assembly search budget')
        if covered == full:
            outcomes.setdefault(tuple(rendered), tuple(path))
            return True
        state = (covered, tuple(remaining))
        if state in dead:
            return False
        best = None
        for c in range(len(coords)):
            if covered >> c & 1:
                continue
            options = [p for p in by_cell[c] if remaining[placements[p][0]]
                       and not (covered & placements[p][3])]
            if not options:
                dead.add(state)
                return False
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        found = False
        for pi in best:
            gi, y, x, bits, cells = placements[pi]
            remaining[gi] -= 1
            for c, value in zip(cells, groups[gi]['tile'].flat):
                rendered[c] = int(value)
            path.append(pi)
            found = search(covered | bits) or found
            path.pop()
            for c in cells:
                rendered[c] = -1
            remaining[gi] += 1
            if len(outcomes) >= 2:
                return True
        if not found:
            dead.add(state)
        return found

    search(0)
    return dict(outcomes=outcomes, groups=groups, placements=placements,
                nodes=nodes, coordinates=coords, expected=expected)


def _prediction(info, assembly):
    if len(assembly['outcomes']) != 1:
        raise ValueError('Assembly must have exactly one rendered completion')
    out = info['board'].copy()
    out[info['mask']] = next(iter(assembly['outcomes']))
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(_grid(input_grid))
    return _prediction(info, _assemble(info))


def _payload(shape, signature, ink, palette, rng):
    h, w = shape
    mode = int(rng.integers(5))
    yy, xx = np.indices(shape)
    if mode == 0:
        tile = np.full(shape, int(rng.choice(palette)), dtype=np.int8)
    elif mode <= 2:
        stripe = yy if mode == 1 else xx
        tile = np.asarray(palette, dtype=np.int8)[stripe % len(palette)]
    elif mode == 3:
        tile = np.asarray(palette, dtype=np.int8)[(yy+xx) % len(palette)]
    else:
        tile = rng.choice(palette, size=shape).astype(np.int8)
    tile[signature] = ink
    return tile


def _turn_windows(length, shape, center):
    """Small rectangles exposing the first complete turn/jog at each phase."""
    radius = max(2,length)
    lower = -radius+max(2,radius-1)-1
    corners = [(-radius,1),(-radius,radius+2),(lower,1),(lower,radius+2)]
    result = []
    for _ in range(4):
        ys,xs = zip(*corners)
        y,x = center[0]+min(ys),center[1]+min(xs)
        h,w = max(ys)-min(ys)+1,max(xs)-min(xs)+1
        if 0 <= y and 0 <= x and y+h <= shape[0] and x+w <= shape[1]:
            result.append((y,x,h,w))
        corners = [(x,-y) for y,x in corners]
    return result


def _rectangles(rng, shape, center, length, count, winding, teach=False):
    h, w = shape
    protected = np.zeros(shape, dtype=bool)
    cy, cx = center
    protected[cy,cx] = True
    for d in range(1,length+1):
        protected[cy-d,cx] = protected[cy+d,cx] = True
        protected[cy,cx-d] = protected[cy,cx+d] = True
    occupied = np.zeros(shape, dtype=bool)
    rects, first_signature = [], None
    if teach:
        windows = _turn_windows(length,shape,center)
        if not windows:
            return None
        y,x,th,tw = windows[int(rng.integers(len(windows)))]
        if protected[y:y+th,x:x+tw].any():
            return None
        rects.append((y,x,th,tw))
        occupied[y:y+th,x:x+tw] = True
    first_shape = tuple(map(int, rng.integers(2, 5, size=2)))
    for i in range(count-len(rects)):
        th, tw = first_shape if i < 2 else tuple(map(int, rng.integers(2, 6, size=2)))
        options, touching = [], []
        for y in range(h-th+1):
            for x in range(w-tw+1):
                if np.any((protected | occupied)[y:y+th, x:x+tw]):
                    continue
                signature = winding[y:y+th, x:x+tw]
                if not signature.any() or signature.all():
                    continue
                if i == 1 and np.array_equal(first_signature, signature):
                    continue
                options.append((y, x))
                # Genuine edge contact, not just diagonally touching corners.
                contact = ((y > 0 and occupied[y-1, x:x+tw].any())
                    or (y+th < h and occupied[y+th, x:x+tw].any())
                    or (x > 0 and occupied[y:y+th, x-1].any())
                    or (x+tw < w and occupied[y:y+th, x+tw].any()))
                if contact:
                    touching.append((y, x))
        choices = touching if touching and rng.random() < .85 else options
        if not choices:
            return None
        y, x = choices[int(rng.integers(len(choices)))]
        rects.append((y, x, th, tw))
        occupied[y:y+th, x:x+tw] = True
        if i == 0:
            first_signature = winding[y:y+th, x:x+tw].copy()
    return occupied, rects


def _scatter(tiles, height, width, vacancy, rng):
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
    for _ in range(20):
        used = np.zeros((height, width), dtype=bool)
        tray = np.full((height, width), vacancy, dtype=np.int8)
        for i in order:
            t = tiles[i]
            h, w = t.shape
            options = [(y,x) for y in range(height-h+1) for x in range(width-w+1)
                       if not used[max(0,y-1):min(height,y+h+1),
                                   max(0,x-1):min(width,x+w+1)].any()]
            if not options:
                break
            y, x = options[int(rng.integers(len(options)))]
            used[y:y+h, x:x+w] = True
            tray[y:y+h, x:x+w] = t
        else:
            return tray
    return None


def _build_pair(rng, length, count, rotation, surplus, teach=False):
    for _ in range(160):
        roles = list(map(int, rng.choice(10, size=4, replace=False)))
        background, vacancy, ink, separator = roles
        palette = [c for c in range(10) if c not in roles]
        rng.shuffle(palette)
        palette = palette[:int(rng.integers(1, 5))]
        h = int(rng.integers(max(18, 2*length+7), 29))
        w = int(rng.integers(max(11, 2*length+5), 18))
        center = (int(rng.integers(length+1, h-length-1)),
                  int(rng.integers(length+1, w-length-1)))
        winding = _walk_pinwheel(length, (h,w), center)
        grown = _rectangles(rng, (h,w), center, length, count, winding, teach)
        if grown is None:
            continue
        mask, rects = grown
        board = np.full((h,w), background, dtype=np.int8)
        cy, cx = center
        for d in range(1, length+1):
            board[cy-d, cx] = board[cy+d, cx] = ink
            board[cy, cx-d] = board[cy, cx+d] = ink
        board[mask] = vacancy
        out, tiles, cache = board.copy(), [], {}
        for y,x,th,tw in rects:
            signature = winding[y:y+th, x:x+tw]
            key = _key(signature)
            if key not in cache:
                cache[key] = _payload((th,tw), signature, ink, palette, rng)
            tile = cache[key].copy()
            tiles.append(tile)
            out[y:y+th, x:x+tw] = tile
        if surplus:
            # A strict subfragment fits the coordinate pattern locally, but its
            # area is less than every selected rectangle. Complete cover rejects
            # it: a single such fragment cannot replace any selected inventory.
            y,x,th,tw = rects[int(rng.integers(len(rects)))]
            dh,dw = (1, int(rng.integers(2, min(tw,3)+1))) if rng.random() < .5 else (
                    int(rng.integers(2, min(th,3)+1)), 1)
            if dh*dw >= min(t.size for t in tiles):
                dh,dw = 1,2
            sy = y+int(rng.integers(th-dh+1))
            sx = x+int(rng.integers(tw-dw+1))
            decoy = _payload((dh,dw), winding[sy:sy+dh,sx:sx+dw], ink, palette, rng)
            # Ensure it would change some payload if used in that visible fit.
            filler = ~winding[sy:sy+dh,sx:sx+dw]
            if not filler.any():
                continue
            old = int(out[sy:sy+dh,sx:sx+dw][filler][0])
            alternative = next(c for c in range(10) if c not in roles and c != old)
            decoy[tuple(np.argwhere(filler)[0])] = alternative
            tiles.append(decoy)
        max_tray_width = 29-w
        min_tray_width = max(max(t.shape[1] for t in tiles), 7, max_tray_width-4)
        tray_width = int(rng.integers(min_tray_width, max_tray_width+1))
        tray = _scatter(tiles, h, tray_width, vacancy, rng)
        if tray is None:
            continue
        inp = np.concatenate((board, np.full((h,1), separator, dtype=np.int8), tray), axis=1)
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
        try:
            info = _parse(inp)
            assembly = _assemble(info)
            prediction = _prediction(info, assembly)
        except ValueError:
            continue
        if np.array_equal(prediction, out):
            return {'input': inp, 'output': out}
    raise RuntimeError('Bounded fragment construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    # These parameters govern sampling and evidence, never the solving rule.
    regime = int(rng.integers(2))  # more fragments / unseen board side
    rotations = list(map(int, rng.permutation(4)))
    train_lengths = [2,3]+[int(rng.choice([1,2,3,4])) for _ in range(num_train-2)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(num_train+num_test):
        training = i < num_train
        length = train_lengths[i] if training else int(rng.choice(train_lengths))
        count = int(rng.integers(3,6) if training or regime else rng.integers(6,9))
        if regime == 1:
            rotation = rotations[i % min(2,num_train)] if training else rotations[2+i%2]
        else:
            rotation = int(rng.integers(4))
        surplus = i < 2 or rng.random() < .55
        for _ in range(12):
            teach = training and length not in train_lengths[:i]
            pair = _build_pair(rng, length, count, rotation, surplus, teach)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle['train' if training else 'test'].append(pair)
                break
        else:
            raise RuntimeError('Could not make distinct input')
    return puzzle


def _features(info, assembly):
    path = next(iter(assembly['outcomes'].values()))
    used = Counter(assembly['placements'][pi][0] for pi in path)
    groups = assembly['groups']
    selected = [groups[gi]['tile'] for gi,n in used.items() for _ in range(n)]
    shapes = Counter(t.shape for t in selected)
    different_same_shape = any(len({_key(t) for t in selected if t.shape == shape}) > 1
                               for shape in shapes)
    unused = sum(g['count']-used[i] for i,g in enumerate(groups))
    local_unused = any(g['count'] > used[i] and any(p[0] == i for p in assembly['placements'])
                       for i,g in enumerate(groups))
    completed = _prediction(info,assembly)
    local_conflict = any(groups[gi]['count'] > used[gi]
                         and not np.array_equal(groups[gi]['tile'],
                             completed[y:y+groups[gi]['tile'].shape[0],
                                       x:x+groups[gi]['tile'].shape[1]])
                         for gi,y,x,_,_ in assembly['placements'])
    owner = np.full(info['mask'].shape, -1, dtype=int)
    for j,pi in enumerate(path):
        gi,y,x,_,_ = assembly['placements'][pi]
        h,w = groups[gi]['tile'].shape
        owner[y:y+h, x:x+w] = j
    component_counts = [len({int(owner[p]) for p in cells}) for cells in _components(info['mask'])]
    turn_visible = any(info['mask'][y:y+h,x:x+w].all()
                       for y,x,h,w in _turn_windows(info['length'],info['mask'].shape,info['center']))
    negative_candidates = 0
    for group in groups:
        tile = group['tile']
        h,w = tile.shape
        motif = tile == info['ink']
        for y in range(info['mask'].shape[0]-h+1):
            for x in range(info['mask'].shape[1]-w+1):
                expected = assembly['expected'][y:y+h,x:x+w]
                if (info['mask'][y:y+h,x:x+w].all() and expected[motif].all()
                        and not np.array_equal(motif,expected)):
                    negative_candidates += 1
    return dict(side=info['side'], length=info['length'], shape=info['mask'].shape,
                center=info['center'], roles=(info['background'],info['vacancy'],info['ink']),
                used=len(path), supplied=len(info['tiles']), surplus=unused,
                local_surplus=local_unused, local_conflict=local_conflict,
                same_shape=different_same_shape,
                repeated=any(n > 1 for n in used.values()),
                max_component=max(component_counts), components=len(component_counts),
                tile_shapes=tuple(sorted(shapes)), marker_pixels=int(assembly['expected'][info['mask']].sum()),
                turn_visible=turn_visible, negative_candidates=negative_candidates,
                nodes=assembly['nodes'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), []
    for pair in puzzle['train']+puzzle['test']:
        assert isinstance(pair,dict) and set(pair) == {'input','output'}
        a,b = _grid(pair['input']),_grid(pair['output'])
        assert _key(a) not in seen
        seen.add(_key(a))
        info = _parse(a)
        assembly = _assemble(info)
        assert np.array_equal(b, _prediction(info,assembly))
        assert np.array_equal(_walk_pinwheel(info['length'],b.shape,info['center']), assembly['expected'])
        mask = info['mask']
        assert np.array_equal(b[~mask],info['board'][~mask])
        assert not np.any((b[mask] == info['vacancy']) | (b[mask] == info['background']))
        assert np.array_equal(b[mask] == info['ink'],assembly['expected'][mask])
        assert np.any(assembly['expected'] & (info['board'] == info['background'])), 'Occlusion must be visible'
        f = _features(info,assembly)
        assert f['same_shape'], 'Geometry alone must leave different rendered completions'
        features.append(f)
    train, test = features[:nt],features[nt:]
    assert {2,3} <= {f['length'] for f in train}
    assert all(any(t['length'] == length and t['turn_visible'] for t in train)
               for length in {t['length'] for t in train})
    assert all(f['length'] in {t['length'] for t in train} for f in test)
    assert all(f['local_surplus'] for f in train[:2])
    assert all(f['local_conflict'] for f in train[:2]), 'Local pattern matching must leave payload ambiguity'
    assert (min(f['used'] for f in test) > max(f['used'] for f in train)
            or not ({f['side'] for f in test} & {f['side'] for f in train}))


def self_test():
    """Bounded 42-seed semantic and variation audit; raises on any failure."""
    features, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for p in puzzle['train']+puzzle['test']))
        for pair in puzzle['train']+puzzle['test']:
            info = _parse(pair['input'])
            assembly = _assemble(info)
            features.append(_features(info,assembly))
        # Explicit ablation: two differently inked equal-sized selected tiles
        # exchange legally without B, producing a different rendered answer.
        info = _parse(puzzle['train'][0]['input'])
        assert len(_assemble(info, use_ink=False)['outcomes']) == 2
    assert len(signatures) == 42
    assert {f['side'] for f in features} == {'top','bottom','left','right'}
    assert {f['length'] for f in features} == {1,2,3,4}
    assert len({f['shape'] for f in features}) >= 50
    assert len({f['roles'] for f in features}) >= 50
    assert len({s for f in features for s in f['tile_shapes']}) >= 12
    assert max(f['max_component'] for f in features) >= 5
    assert any(f['components'] > 1 for f in features)
    assert max(f['used'] for f in features) >= 8
    assert {bool(f['surplus']) for f in features} == {False,True}
    assert any(f['repeated'] for f in features)
    assert any(f['negative_candidates'] for f in features)
    # Analytic and procedural definitions, short-spoke equivalence, rotations,
    # and crop/re-entry consistency are tested independently of puzzle labels.
    for length in range(1,7):
        shape, center = (23,29),(length,28-length)
        mask = _pinwheel(length,shape,center)
        assert np.array_equal(mask,_walk_pinwheel(length,shape,center))
        assert np.array_equal(np.rot90(mask),_pinwheel(length,shape[::-1],(shape[1]-1-center[1],center[0])))
        big = _pinwheel(length,(70,70),(35,35))
        y,x = 35-center[0],35-center[1]
        assert np.array_equal(mask,big[y:y+shape[0],x:x+shape[1]])
    assert np.array_equal(_pinwheel(1,(30,30),(15,15)),_pinwheel(2,(30,30),(15,15)))
    # Isolate the inventory invariant: one physical tile cannot fill two islands.
    # Disabling ink here exercises packing alone, not a second family rule.
    inventory = dict(mask=np.array([[True,True,False,True,True]]),
                     length=2,center=(0,0),ink=7,tiles=[np.array([[2,3]],dtype=np.int8)])
    assert not _assemble(inventory,use_ink=False)['outcomes']
    inventory['tiles'].append(inventory['tiles'][0].copy())
    assert len(_assemble(inventory,use_ink=False)['outcomes']) == 1
    a,b = generate_puzzle(123,2,1),generate_puzzle(123,2,1)
    assert all(np.array_equal(x[k],y[k]) for s in a for x,y in zip(a[s],b[s]) for k in x)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s:[{k:v.copy() for k,v in p.items()} for p in a[s]] for s in a}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt output accepted')
    return dict(seeds=42,pairs=len(features),distinct_shapes=len({f['shape'] for f in features}),
                max_joined_fragments=max(f['max_component'] for f in features))


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
