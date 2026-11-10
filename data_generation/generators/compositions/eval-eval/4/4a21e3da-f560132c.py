"""Surveyed jigsaws: source-coordinate cuts travel with reassembled tiles.

Only NumPy and the standard library are used. No parent module is imported.
Generation retains one small puzzle at a time and writes no files or traces.
"""
from __future__ import annotations

import itertools
import numpy as np


_SIDES = ("top", "right", "bottom", "left")
_D = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require integer counts 2..10 / 1..4 with num_train > num_test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=False)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        found, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    found.append(q)
                    stack.append(q)
        result.append(np.array(sorted(found), dtype=int))
    return result


def _normalize(coords):
    a = np.asarray(coords, dtype=int)
    return a - a.min(axis=0)


def _rotate(coords, turns):
    a = np.asarray(coords, dtype=int).copy()
    for _ in range(turns % 4):
        a = np.column_stack((a[:, 1].max() - a[:, 1], a[:, 0]))
    return _normalize(a)


def _mask(coords, n):
    result = 0
    for r, c in coords:
        result |= 1 << (int(r) * n + int(c))
    return result


def _packings(shapes, n):
    """Exhaust all corner-flush placements; preserve symmetric ink rotations."""
    corners = (0, n - 1, n * (n - 1), n * n - 1)
    anchor = shapes[0]
    if np.any(anchor.max(axis=0) >= n):
        return []
    anchor_mask = _mask(anchor, n)
    if [bool(anchor_mask & (1 << k)) for k in corners] != [True, False, False, False]:
        return []
    table = {}
    for i in range(1, 4):
        for corner in range(1, 4):
            options = []
            for turn in range(4):
                a = _rotate(shapes[i], turn)
                h, w = a.max(axis=0) + 1
                if h > n or w > n:
                    continue
                offset = np.array((0 if corner < 2 else n-h, 0 if corner % 2 == 0 else n-w))
                targets = a + offset
                m = _mask(targets, n)
                hits = [bool(m & (1 << k)) for k in corners]
                if sum(hits) == 1 and hits[corner]:
                    options.append((m, turn, targets))
            table[i, corner] = options
    full = (1 << (n * n)) - 1
    results = []
    for order in itertools.permutations((1, 2, 3)):
        for options in itertools.product(*(table[i, k] for k, i in enumerate(order, 1))):
            occupied = anchor_mask
            placed = [(anchor_mask, 0, anchor)] + [None] * 3
            for i, option in zip(order, options):
                if occupied & option[0]:
                    break
                occupied |= option[0]
                placed[i] = option
            else:
                if occupied == full:
                    results.append(placed)
    return results


def _parse(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    _require(min(h, w) >= 5, "canvas too small")
    rim = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    bg = int(np.bincount(rim, minlength=10).argmax())
    _require(np.count_nonzero(rim == bg) > len(rim) / 2, "background not identifiable")
    border = np.zeros_like(grid, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    points = np.argwhere(border & (grid != bg))
    _require(len(points) in (1, 2), "one or two rim markers required")
    marker = int(grid[tuple(points[0])])
    _require(np.all(grid[points[:, 0], points[:, 1]] == marker), "marker colors disagree")
    _require(np.count_nonzero(grid == marker) == len(points), "marker color appears inside")
    sides, ar, ac = set(), None, None
    for r, c in points:
        flags = [r == 0, c == w-1, r == h-1, c == 0]
        _require(sum(flags) == 1, "marker cannot be at a corner")
        side = _SIDES[flags.index(True)]
        _require(side not in sides, "duplicate marked side")
        sides.add(side)
        if side in ("top", "bottom"):
            ac = int(c)
        else:
            ar = int(r)
    _require(len(sides) == 1 or (ar is not None and ac is not None), "two markers must be adjacent")
    components = _components((grid != bg) & (grid != marker))
    _require(len(components) == 4, "four tile footprints required")
    anchors, plain = [], []
    for coords in components:
        values, counts = np.unique(grid[coords[:, 0], coords[:, 1]], return_counts=True)
        if len(values) == 1:
            plain.append((coords, int(values[0])))
        else:
            _require(len(values) == 5 and sum(counts == 1) == 4 and sum(counts > 1) == 1, "bad embedded key")
            carrier = int(values[counts > 1][0])
            p = coords[np.isin(grid[coords[:, 0], coords[:, 1]], values[counts == 1])]
            r, c = p.min(axis=0)
            _require(set(map(tuple, p)) == {(r,c),(r+1,c),(r,c+1),(r+1,c+1)}, "key must be 2 by 2")
            anchors.append((coords, carrier, grid[r:r+2, c:c+2].copy()))
    _require(len(anchors) == 1 and len(plain) == 3, "one keyed tile required")
    coords, carrier, key = anchors[0]
    _require(all(color == carrier for _, color in plain), "tiles must share a carrier")
    _require(len({bg, carrier, marker, *map(int, key.flat)}) == 7, "palette roles must be distinct")
    components = [coords] + [x[0] for x in plain]
    _require(ac is None or any(np.any(p[:,1] == ac) for p in components), "vertical guide misses all tiles")
    _require(ar is None or any(np.any(p[:,0] == ar) for p in components), "horizontal guide misses all tiles")
    for p, q in itertools.combinations(components, 2):
        _require(np.max(np.abs(p[:, None] - q[None, :]), axis=2).min() >= 2, "tiles need a blank-cell gap")
    total = sum(map(len, components))
    n = int(total ** .5)
    _require(n*n == total and n <= 30, "tile area must be square")
    return dict(grid=grid, bg=bg, carrier=carrier, marker=marker, key=key,
                components=components, shapes=[_normalize(p) for p in components],
                sides=sides, ar=ar, ac=ac, n=n)


def _survey(a):
    """Paint in SOURCE coordinates. Legend cells count as ordinary tile area."""
    inks, regions = [], []
    for coords in a['components']:
        local, labels = [], []
        for r, c in coords:
            if r == a['ar'] or c == a['ac']:
                local.append(a['carrier'])
                labels.append(-1)
                continue
            v = (0 if r < a['ar'] else 1) if a['ar'] is not None else int('bottom' in a['sides'])
            u = (0 if c < a['ac'] else 1) if a['ac'] is not None else int('right' in a['sides'])
            corner = 2*v + u
            retained = _SIDES[(0 if v == 0 else 2)] in a['sides'] or ('left' if u == 0 else 'right') in a['sides']
            local.append(int(a['key'][v, u]) if retained else a['bg'])
            labels.append(corner)
        inks.append(np.array(local, dtype=np.int8))
        regions.append(np.array(labels, dtype=int))
    return inks, regions


def _analysis(input_grid):
    a = _parse(input_grid)
    inks, labels = _survey(a)
    layouts = _packings(a['shapes'], a['n'])
    _require(layouts, "no legal square assembly")
    # Unique physical placement modulo harmless silhouette symmetries.
    _require(len({tuple(p[0] for p in layout) for layout in layouts}) == 1, "ambiguous footprint assembly")
    answers = {}
    for layout in layouts:
        out = np.full((a['n'], a['n']), a['bg'], dtype=np.int8)
        for (_, _, targets), ink in zip(layout, inks):
            out[targets[:, 0], targets[:, 1]] = ink
        answers.setdefault(out.tobytes(), out)
        _require(len(answers) == 1, "silhouette symmetry changes transported ink")
    a.update(inks=inks, labels=labels, layouts=layouts, output=next(iter(answers.values())))
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, survey cuts, and unique assembly from this input only."""
    return _analysis(input_grid)['output'].copy()


def _key_locations(owner):
    return [(r, c) for r in range(len(owner)-1) for c in range(len(owner)-1)
            if np.all(owner[r:r+2, c:c+2] == 0)]


def _partition(rng, n, asymmetric=False):
    """Connected random growth plus local seam exchanges, checked by packing."""
    for attempt in range(160):
        if attempt % 2 == 0:
            owner = np.full((n, n), -1, dtype=np.int8)
            owner[:2, :2] = 0
            owner[0, -1], owner[-1, 0], owner[-1, -1] = 1, 2, 3
            weights = rng.uniform(.3, 2.2, 4)
            while np.any(owner < 0):
                frontiers = []
                for i in range(4):
                    f = sorted({(r+dr, c+dc) for r,c in np.argwhere(owner == i) for dr,dc in _D
                                if 0 <= r+dr < n and 0 <= c+dc < n and owner[r+dr,c+dc] < 0})
                    frontiers.append(f)
                probabilities = np.array([weights[i] if f else 0 for i,f in enumerate(frontiers)])
                i = int(rng.choice(4, p=probabilities/probabilities.sum()))
                owner[frontiers[i][int(rng.integers(len(frontiers[i])))]] = i
        else:
            y = int(rng.integers(2, n-1))
            owner = np.empty((n, n), dtype=np.int8)
            for band, rows in enumerate((range(y), range(y, n))):
                cut = int(rng.integers(2, n-1))
                for r in rows:
                    if r >= 2 and rng.random() < .7:
                        cut = int(np.clip(cut + rng.integers(-1, 2), 1, n-1))
                    owner[r,:cut], owner[r,cut:] = 2*band, 2*band+1
            for _ in range(n*2):
                r,c = map(int, rng.integers(1, n-1, 2))
                old = int(owner[r,c])
                other = int(owner[r+int(rng.choice((-1,1))),c] if rng.random() < .5 else owner[r,c+int(rng.choice((-1,1)))])
                owner[r,c] = other
                if any(len(_components(owner == i)) != 1 for i in {old,other}):
                    owner[r,c] = old
        if not _key_locations(owner):
            continue
        shapes = [_normalize(np.argwhere(owner == i)) for i in range(4)]
        if min(map(len, shapes)) < 4 or len(shapes[0]) < 7:
            continue
        if asymmetric and any(len({_mask(_rotate(s,k), n) for k in range(4)}) != 4 for s in shapes):
            continue
        layouts = _packings(shapes, n)
        if len({tuple(p[0] for p in layout) for layout in layouts}) != 1:
            continue
        return owner, shapes
    raise RuntimeError("bounded partition search exhausted")


def _scatter(rng, shapes):
    """Random shelves: all orders/groupings, then transposition and padding."""
    dims = [s.max(axis=0)+1 for s in shapes]
    candidates = []
    order = list(map(int, rng.permutation(4)))
    for transpose in (False, True):
        ds = [d[::-1] if transpose else d for d in dims]
        for cutbits in range(8):
            groups, group = [], []
            for j,i in enumerate(order):
                group.append(i)
                if j == 3 or cutbits & (1 << j):
                    groups.append(group)
                    group = []
            gap = int(rng.integers(1, 4))
            hh = sum(max(ds[i][0] for i in g) for g in groups) + gap*(len(groups)-1)
            ww = max(sum(ds[i][1] for i in g) + gap*(len(g)-1) for g in groups)
            if hh > 28 or ww > 28:
                continue
            offsets, r = [None]*4, 0
            for g in groups:
                height = max(ds[i][0] for i in g)
                width = sum(ds[i][1] for i in g) + gap*(len(g)-1)
                c = int(rng.integers(ww-width+1))
                for i in g:
                    rr = r + int(rng.integers(height-ds[i][0]+1))
                    offsets[i] = (c,rr) if transpose else (rr,c)
                    c += ds[i][1] + gap
                r += height + gap
            candidates.append(offsets)
    _require(candidates, "no bounded scatter layout")
    offsets = np.array(candidates[int(rng.integers(len(candidates)))])
    extents = np.max(offsets + np.array(dims), axis=0)
    before = np.array([int(rng.integers(1, min(4, 30-int(e)))) for e in extents])
    after = np.array([int(rng.integers(1, min(5, 31-int(e)-int(b)))) for e,b in zip(extents,before)])
    return offsets + before, tuple(map(int, extents+before+after))


def _axis_candidates(rng, components, sides, shape):
    vertical = bool(sides & {'top','bottom'})
    horizontal = bool(sides & {'left','right'})
    rs = range(1,shape[0]-1) if horizontal else [None]
    cs = range(1,shape[1]-1) if vertical else [None]
    candidates = list(itertools.product(rs, cs))
    rng.shuffle(candidates)
    for ar,ac in candidates:
        # Each active axis actually crosses payload. Several fragments must be inked.
        if ac is not None and not any(np.any(p[:,1] == ac) for p in components):
            continue
        if ar is not None and not any(np.any(p[:,0] == ar) for p in components):
            continue
        groups, mixed, partial, alive = set(), 0, 0, True
        for p in components:
            labels = []
            for r,c in p:
                if r == ar or c == ac:
                    continue
                v = int(r > ar) if ar is not None else int('bottom' in sides)
                u = int(c > ac) if ac is not None else int('right' in sides)
                labels.append(2*v+u)
            groups.update(labels)
            kept = [k for k in labels if ('top' if k < 2 else 'bottom') in sides or ('left' if k%2 == 0 else 'right') in sides]
            mixed += len(set(kept)) > 1
            alive &= bool(kept)
            partial += bool(kept) and len(kept) < len(labels)
        if len(sides) == 1:
            if len(groups) == 2 and mixed >= 1:
                yield ar,ac
        elif len(groups) == 4 and mixed >= 1 and partial >= 1 and alive:
            yield ar,ac


def _make_pair(rng, n, sides, palette, demonstrate):
    owner, original = _partition(rng, n, asymmetric=demonstrate)
    turns = [0,1,2,3] if demonstrate else [0, *map(int,rng.integers(0,4,3))]
    oriented = [_rotate(s,k) for s,k in zip(original,turns)]
    # Keys use the unrotated anchor's local coordinates.
    anchor_min = np.argwhere(owner == 0).min(axis=0)
    keylocs = _key_locations(owner)
    kr,kc = keylocs[int(rng.integers(len(keylocs)))]
    kr,kc = int(kr-anchor_min[0]), int(kc-anchor_min[1])
    bg,carrier,marker = palette[:3]
    key = np.array(palette[3:],dtype=np.int8).reshape(2,2)
    for _ in range(20):
        offsets, canvas = _scatter(rng, oriented)
        comps = [s+offset for s,offset in zip(oriented,offsets)]
        for ar,ac in itertools.islice(_axis_candidates(rng,comps,sides,canvas),12):
            inp = np.full(canvas,bg,dtype=np.int8)
            for p in comps:
                inp[p[:,0],p[:,1]] = carrier
            r,c = offsets[0]
            inp[r+kr:r+kr+2,c+kc:c+kc+2] = key
            if 'top' in sides:
                inp[0,ac] = marker
            if 'bottom' in sides:
                inp[-1,ac] = marker
            if 'left' in sides:
                inp[ar,0] = marker
            if 'right' in sides:
                inp[ar,-1] = marker
            try:
                a = _analysis(inp)
            except AssertionError:
                continue
            # Construction-based transport: identify each source cell's intended
            # destination, independently of the packing search's placement order.
            expected = np.full((n,n),bg,dtype=np.int8)
            for i, shape in enumerate(original):
                source = _rotate(shape,turns[i]) + offsets[i]
                for target,p in zip(shape,source):
                    rr,cc = map(int,p)
                    if rr == ar or cc == ac:
                        color = carrier
                    else:
                        v = int(rr > ar) if ar is not None else int('bottom' in sides)
                        u = int(cc > ac) if ac is not None else int('right' in sides)
                        selected = ('top' if v == 0 else 'bottom') in sides or ('left' if u == 0 else 'right') in sides
                        color = int(key[v,u]) if selected else bg
                    # original is normalized: recover its square offset.
                    dest = target + np.argwhere(owner == i).min(axis=0)
                    expected[tuple(dest)] = color
            _require(np.array_equal(expected,a['output']), "constructed survey disagrees with input reconstruction")
            return {'input':inp,'output':expected}
    raise RuntimeError("bounded survey/scatter search exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train,num_test = _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    k = int(rng.integers(4))
    double = {_SIDES[k],_SIDES[(k+1)%4]}
    single = {_SIDES[(k+2+int(rng.integers(2)))%4]}
    regime = int(rng.integers(3))
    fixed = int(rng.integers(7,11))
    base = list(map(int,rng.choice(10,7,replace=False)))
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',num_train),('test',num_test)):
        for j in range(count):
            train = split == 'train'
            n = int(rng.integers(6,9) if train else rng.integers(9,12)) if regime == 0 else (
                int(rng.integers(9,12) if train else rng.integers(6,9)) if regime == 1 else fixed)
            if train and j == 0:
                sides = single
            elif j == 0 or (train and j == 1):
                sides = double
            else:
                t = int(rng.integers(4))
                sides = {_SIDES[t]} if rng.random() < .4 else {_SIDES[t],_SIDES[(t+1)%4]}
            palette = base.copy() if j == 0 and train else list(map(int,rng.choice(10,7,replace=False)))
            for attempt in range(12):
                try:
                    pair = _make_pair(rng,n,sides,palette,train and j == 0)
                except RuntimeError:
                    continue
                signature = (pair['input'].shape,pair['input'].tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct pair search exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    ink = np.concatenate(a['inks'])
    mixed = sum(len(set(map(int,x)) - {a['bg'],a['carrier']}) > 1 for x in a['inks'])
    partial = sum(np.any(x == a['bg']) and np.any(x != a['bg']) for x in a['inks'])
    return dict(n=a['n'], shape=a['grid'].shape, sides='+'.join(sorted(a['sides'])),
                arity=len(a['sides']), bg=a['bg'], area=tuple(sorted(map(len,a['shapes']))),
                mixed=mixed, partial=partial, erased=int(np.count_nonzero(ink == a['bg'])),
                axial=int(np.count_nonzero(ink == a['carrier'])),
                rotations=tuple(p[1] for p in a['layouts'][0]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "split schema")
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen,train,features = set(),[],{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'}, "pair schema")
            inp,out = _grid(pair['input']),_grid(pair['output'])
            signature = (inp.shape,inp.tobytes())
            _require(signature not in seen,"duplicate input")
            seen.add(signature)
            a = _analysis(inp)
            _require(np.array_equal(a['output'],out),"wrong transported survey")
            _require(out.shape == (a['n'],a['n']),"output square")
            f = _features(a)
            _require(f['axial'] > 0,"no axial witness")
            _require(f['mixed'] > 0,"no piece crosses a survey cut")
            if f['arity'] == 2:
                _require(f['erased'] > 0 and f['partial'] > 0,"no partial erasure witness")
                _require(set(np.concatenate(a['labels'])) - {-1} == set(range(4)),"missing source quadrant")
                _require(all(np.any((x != a['bg']) & (x != a['carrier'])) for x in a['inks']),"a whole tile lost its off-axis ink")
            _require(np.count_nonzero(out == a['bg']) == f['erased'],"erasure conservation")
            _require(np.count_nonzero(out == a['carrier']) == f['axial'],"axis conservation")
            # Parent-B whole-piece coloring ignores the source survey.
            b_only = np.empty_like(out)
            for placement in a['layouts'][0]:
                p = placement[2]
                hits = [bool(placement[0] & (1 << z)) for z in (0,a['n']-1,a['n']*(a['n']-1),a['n']**2-1)]
                b_only[p[:,0],p[:,1]] = a['key'].flat[hits.index(True)]
            _require(not np.array_equal(b_only,out),"survey can be omitted")
            # Keeping source geometry (even tightly cropped) is not assembly.
            source_ink = np.full_like(inp,a['bg'])
            for coords,inks in zip(a['components'],a['inks']):
                source_ink[coords[:,0],coords[:,1]] = inks
            p = np.vstack(a['components'])
            lo,hi = p.min(axis=0),p.max(axis=0)+1
            _require(not np.array_equal(source_ink[lo[0]:hi[0],lo[1]:hi[1]],out),"assembly can be omitted")
            features[split].append(f)
            if split == 'train':
                train.append(a)
    _require({1,2} <= {x['arity'] for x in features['train']},"train needs both marker arities")
    _require(any(x['partial'] for x in features['train']),"missing erasure demonstration")
    _require(any(all(p[1] != 0 for p in a['layouts'][0][1:]) for a in train),"missing rotation demonstration")
    used = set()
    for a in train:
        for inks in a['inks']:
            used.update(i for i,c in enumerate(a['key'].flat) if int(c) in inks)
    _require(used == set(range(4)),"training must expose every key position")


def self_test():
    """42 streaming seeds, diversity from pixels, metamorphisms, corruption."""
    shapes,areas,sides,backgrounds,sizes,normalized,shifts = set(),set(),set(),set(),set(),set(),set()
    counts = dict(puzzles=0,pairs=0,partial=0,multicolor_tiles=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        counts['puzzles'] += 1
        tr = [p['output'].shape[0] for p in puzzle['train']]
        for split in ('train','test'):
            for pair in puzzle[split]:
                a = _analysis(pair['input'])
                f = _features(a)
                shapes.add(f['shape']);areas.add(f['area']);sides.add(f['sides']);backgrounds.add(f['bg']);sizes.add(f['n'])
                counts['pairs'] += 1
                counts['partial'] += bool(f['partial'])
                counts['multicolor_tiles'] += f['mixed']
                roles = {a['bg']:0,a['carrier']:1,a['marker']:2,**{int(c):i+3 for i,c in enumerate(a['key'].flat)}}
                normalized.add((a['grid'].shape,bytes(roles[int(c)] for c in a['grid'].flat)))
                if split == 'test':
                    shifts.add('grow' if f['n'] > max(tr) else 'shrink' if f['n'] < min(tr) else 'equal')
        if seed < 3:
            pair = puzzle['test'][0]
            inp,out = pair['input'],pair['output']
            permutation = np.random.default_rng(seed+900).permutation(10)
            _require(np.array_equal(solve(permutation[inp]),permutation[out]),"color equivariance")
            a = _parse(inp)
            padded = np.pad(inp,1,constant_values=a['bg'])
            # Move markers to the enlarged outer rim; their sightlines stay fixed.
            for r,c in np.argwhere(inp == a['marker']):
                padded[r+1,c+1] = a['bg']
                rr = 0 if r == 0 else padded.shape[0]-1 if r == inp.shape[0]-1 else r+1
                cc = 0 if c == 0 else padded.shape[1]-1 if c == inp.shape[1]-1 else c+1
                padded[rr,cc] = a['marker']
            if max(padded.shape) <= 30:
                _require(np.array_equal(solve(padded),out),"margin invariance")
            # A-side counterfactual: unchanged pieces/key/axis, opposite marked
            # side. Every off-axis color must change; axial carrier must not.
            first = puzzle['train'][0]
            a = _parse(first['input'])
            swapped = first['input'].copy()
            r,c = np.argwhere(swapped == a['marker'])[0]
            swapped[r,c] = a['bg']
            rr = swapped.shape[0]-1-r if r in (0,swapped.shape[0]-1) else r
            cc = swapped.shape[1]-1-c if c in (0,swapped.shape[1]-1) else c
            swapped[rr,cc] = a['marker']
            changed = solve(swapped)
            _require(np.count_nonzero(changed != first['output']) == a['n']**2 - np.count_nonzero(first['output'] == a['carrier']),"marker-side counterfactual")
    _require(len(normalized) >= 200 and len(shapes) >= 35 and len(areas) >= 100,"structural variation collapsed")
    _require(len(sides) == 8 and backgrounds == set(range(10)) and sizes == set(range(6,12)),"role/orientation/scale coverage")
    _require(shifts == {'grow','shrink','equal'} and counts['partial'] >= 80,"shift or erasure coverage")
    _require(counts['multicolor_tiles'] >= 210,"mixed tiles missing")
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            _counts(a,b)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first = generate_puzzle(0,2,1)
    again = generate_puzzle(0,2,1)
    _require(all(np.array_equal(x[k],y[k]) for s in first for x,y in zip(first[s],again[s]) for k in x),"determinism")
    broken = {s:[{k:v.copy() for k,v in p.items()} for p in first[s]] for s in first}
    broken['test'][0]['output'][0,0] = (int(broken['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt answer accepted")
    return dict(counts,canvas_shapes=len(shapes),area_profiles=len(areas),normalized_inputs=len(normalized),
                marker_configurations=len(sides),output_sides=sorted(sizes),backgrounds=sorted(backgrounds),shifts=sorted(shifts))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(),separators=(",", ":")))
