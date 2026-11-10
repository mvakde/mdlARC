"""The saved rule: topological canonicalization, then adjacent vector collapse.

Only NumPy and the standard library are used. No parent module is imported.
The second stage is intentionally literal: minimal canonical diagrams happen to
be fixed points of it. See composition.md for this unresolved necessity limit.
Generation keeps at most one candidate at a time and writes no files or traces.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(g.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((g >= 0) & (g <= 9))), "grid colors must be 0..9")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, np.asarray(g, dtype=np.int8).tobytes()


def _hash(g):
    return hashlib.sha256(bytes(g.shape) + np.asarray(g, dtype=np.uint8).tobytes()).hexdigest()


# Exclusion hashes only; no puzzle or answer lookup is used by solve().
_PARENT_INPUT_HASHES = frozenset("""
02e0a84d14059147314273f0087294bb6e947f05146111ad4096a998150d8e49
080c2c26b703012bd672deefcbf326ab7af8a8a2c3deaaa522add973b24ea893
2157fa38649f3fc49b53160e9ab90cce43cf15975aac15da38437c84f9c49d67
3de6ae9237599a77e4450771f4853b37cd4bde7191280dea784fa78c01662e23
3e0dbba8e69a71acc3e04efc7404978d10b55538b434082c640f4f358e6b0502
44553ebeffd77fdb4d04a2284e96fc6cb350b1c750a8b696804fc16f923a6ea7
6d120cbb86ff9026f5b89c1381b0d2f7b6b4792b25b1fba0cd9dc280e7e54467
7774361869a82569e90757c6037b594f77b958c903de2b33fab9615fa5ed0529
8074e1d7bbd21273fa42000bfef9529ed4ec1ad14db6df566f408f9402a4a04d
816052668bcd4871e42a2210cfa70c95908bd4aafff2890ef8f8d2903d9dc4e3
ae407229ca89893c3304d1ea91e7bea9e7f506123602a6a82cc2e56b599a1825
b2af20eb4e325c4ce0e10b22a3f281a3391530bdb6f185ca8471ee8e4b417119
c6fcad7dfdf52f461457a5a79df561ba9b89b2b7758cd69a5cdf67f8fd5cada5
cb48274078ce7ffc5e5fdb2b81085a4cb02cd1a794f237876765be08f6221f86
d29a3290d22c52dd594cb30088c58fa0b30d790b9d69308afb0786ece3f2eb54
e6e7418e4915de31b38498cdc65683ee2f9efdc4a5d3c54521b7c5c65cad05e8
ebe5963682e37e60b0723b471ded315ef505325091e90656401e975e27a8b907
f30870ad9eb8baaa072fd8cd14cf9cbc19ab68a70de6e570abe2fa48251adad8
f89792bad9fbb15cb1bfc77a4cbdf8c921ab054022cbe69b51599e56a9c409f3
f9bc82eaf0276c1c3bd8cdab441bee36deaecc5fba6d256821598ed0fec4e87a
fbe1dbc2640f6a7f5f99a04a91c10ba54dc716fde07a0681832ea05b3ad01c28
fd60665b62655906cc5cc5a8e414303f4a44a4068f9ad5711cd8a749f2c234d2
""".split())


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells = [(int(y), int(x))], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _holes(mask):
    occupied = np.pad(mask, 1)
    outside = np.zeros_like(occupied)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = occupied.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
            if 0 <= rr < h and 0 <= cc < w and not occupied[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))
    return (~occupied & ~outside)[1:-1, 1:-1]


def _parse(g, background=None):
    colors, counts = np.unique(g, return_counts=True)
    _require(len(colors) == 2, "a diagram uses exactly two colors")
    if background is None:
        _require(counts[0] != counts[1], "background must have a strict majority")
        background = int(colors[np.argmax(counts)])
    _require(background in colors, "background absent")
    foreground = int(colors[colors != background][0])
    objects, labels = [], np.zeros(g.shape, dtype=int)
    for cells in _components(g == foreground):
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        mask = np.zeros(g.shape, dtype=bool)
        mask[cells[:, 0], cells[:, 1]] = True
        hole = _holes(mask)
        is_loop = bool(hole.any())
        if is_loop:
            _require(len(_components(hole)) == 1, "contour must enclose one region")
        else:
            _require(len(cells) >= .35 * int(np.prod(hi-lo+1)), "terminal blob is not compact")
        objects.append(dict(lo=lo, hi=hi, cells=cells, mask=mask, hole=hole,
                            loop=is_loop, children=[]))
        labels[mask] = len(objects)
    _require(any(o['loop'] for o in objects), "at least one contour is required")
    for first, second in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                          (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        _require(not np.any((first > 0) & (second > 0) & (first != second)), "distinct objects touch")
    roots, parents = [], []
    for i, obj in enumerate(objects):
        containers = []
        for j, other in enumerate(objects):
            if i == j or not other['loop']:
                continue
            inside = other['hole'][obj['cells'][:, 0], obj['cells'][:, 1]]
            _require(bool(inside.all()) or not inside.any(), "objects cross a contour")
            if inside.all():
                containers.append(j)
        parent = min(containers, key=lambda j: int(objects[j]['hole'].sum())) if containers else None
        parents.append(parent)
        (roots if parent is None else objects[parent]['children']).append(i)
    for obj in objects:
        _require(bool(obj['children']) == obj['loop'], "contours must have contents")
    return dict(objects=objects, roots=roots, parents=parents, bg=background, fg=foreground)


# A tile is a tight foreground mask plus positions of its descendant point leaves.
def _join(parts, axis, offsets, gaps):
    cross, low, position = 1-axis, min(offsets), 0
    starts, shape = [], [0, 0]
    for i, (mask, leaves) in enumerate(parts):
        start = [0, 0]
        start[axis], start[cross] = position, offsets[i]-low
        starts.append(start)
        position += mask.shape[axis] + (gaps[i] if i < len(gaps) else 0)
        shape[cross] = max(shape[cross], start[cross] + mask.shape[cross])
    shape[axis] = position
    result, anchors = np.zeros(shape, dtype=bool), []
    for (mask, leaves), (y, x) in zip(parts, starts):
        h, w = mask.shape
        result[y:y+h, x:x+w] |= mask
        anchors.extend((key, r+y, c+x) for key, r, c in leaves)
    return result, anchors


def _frame(tile):
    mask, leaves = tile
    out = np.pad(mask, 2)
    out[0, :] = out[-1, :] = True
    out[:, 0] = out[:, -1] = True
    return out, [(key, r+2, c+2) for key, r, c in leaves]


def _halo(tile, exterior):
    mask, leaves = tile
    h, w = mask.shape
    pads = [0, 0, 0, 0]
    for key, r, c in leaves:
        if key in exterior:
            pads = [max(pads[0], 1-r), max(pads[1], r+2-h),
                    max(pads[2], 1-c), max(pads[3], c+2-w)]
    return np.pad(mask, ((pads[0], pads[1]), (pads[2], pads[3])))


def _canonical(info):
    objects, cache = info['objects'], {}

    def aligned_offsets(parts, cross):
        constraints = []
        for i in range(len(parts)):
            for j in range(i+1, len(parts)):
                shifts = set()
                for ka, ra, ca in parts[i][1]:
                    for kb, rb, cb in parts[j][1]:
                        a, b = objects[ka], objects[kb]
                        if max(a['lo'][cross], b['lo'][cross]) <= min(a['hi'][cross], b['hi'][cross]):
                            shifts.add((ra, ca)[cross] - (rb, cb)[cross])
                _require(len(shifts) <= 1, "competing descendant anchors")
                if shifts:
                    constraints.append((i, j, shifts.pop()))
        offsets = {0: 0}
        for _ in parts:
            for i, j, delta in constraints:
                if i in offsets and j in offsets:
                    _require(offsets[j] == offsets[i]+delta, "inconsistent alignment")
                elif i in offsets:
                    offsets[j] = offsets[i]+delta
                elif j in offsets:
                    offsets[i] = offsets[j]-delta
        _require(len(offsets) == len(parts), "unanchored diagonal placement")
        return [offsets[i] for i in range(len(parts))]

    def region(ids):
        ids = tuple(sorted(ids))
        if ids in cache:
            return cache[ids]
        if len(ids) == 1:
            i = ids[0]
            result = (_frame(region(objects[i]['children'])) if objects[i]['loop']
                      else (np.ones((1, 1), dtype=bool), [(i, 0, 0)]))
        else:
            candidates = []
            for axis in (0, 1):
                groups, end = [], -2
                for i in sorted(ids, key=lambda k: int(objects[k]['lo'][axis])):
                    if objects[i]['lo'][axis] > end+1:
                        groups.append([])
                    groups[-1].append(i)
                    end = max(end, int(objects[i]['hi'][axis]))
                if len(groups) < 2:
                    continue
                try:
                    parts = [region(group) for group in groups]
                    offsets = aligned_offsets(parts, 1-axis)
                    candidates.append(_join(parts, axis, offsets, [1]*(len(parts)-1)))
                except ValueError:
                    continue
            _require(bool(candidates), "no consistent cardinal decomposition")
            result = candidates[0]
            for candidate in candidates[1:]:
                _require(np.array_equal(candidate[0], result[0]) and sorted(candidate[1]) == sorted(result[1]),
                         "ambiguous canonical arrangement")
        cache[ids] = result
        return result

    exterior = {i for i in info['roots'] if not objects[i]['loop']}
    return _halo(region(info['roots']), exterior)


def _collapse_vectors(g):
    """B: independent exact row/column runs, preserving separated recurrence."""
    rows = np.r_[True, np.any(g[1:] != g[:-1], axis=1)]
    cols = np.r_[True, np.any(g[:, 1:] != g[:, :-1], axis=0)]
    return g[np.ix_(rows, cols)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive A's diagram from input pixels, then apply B to that diagram."""
    info = _parse(_grid(input_grid))
    canonical = _canonical(info)
    _require(all(d <= 30 for d in canonical.shape), "canonical diagram exceeds 30 cells")
    return _collapse_vectors(np.where(canonical, info['fg'], info['bg']).astype(np.int8))


# New inputs are sampled as trees of points, actual contours, and spatial groups.
_P = ('P',)


def _leaf_count(node):
    if node[0] == 'P':
        return 1
    if node[0] == 'C':
        return _leaf_count(node[1])
    return sum(_leaf_count(child) for child in node[2])


def _group(rng, children, axis=None):
    if len(children) == 1:
        return children[0]
    axis = int(rng.integers(2)) if axis is None else axis
    return ('G', axis, tuple(children), tuple(int(rng.integers(_leaf_count(c))) for c in children))


def _random_loop(rng, depth, limit):
    count = int(rng.choice([1, 2, 3, 4], p=[.42, .35, .19, .04]))
    children = [(_random_loop(rng, depth+1, limit) if depth < limit and rng.random() < .27 else _P)
                for _ in range(count)]
    axis = int(rng.integers(2))
    if count >= 3 and rng.random() < .6:
        cut = int(rng.integers(1, count))
        content = _group(rng, [_group(rng, children[:cut], 1-axis),
                               _group(rng, children[cut:], 1-axis)], axis)
    else:
        content = _group(rng, children, axis)
    return ('C', content)


def _scene(rng, kind):
    if kind == 'simple':
        return ('C', _group(rng, [_P] * int(rng.integers(1, 4))))
    if kind == 'witness':
        inner = ('C', _P)
        side = _group(rng, [_P, inner] if rng.random() < .5 else [inner, _P], 1)
        content = _group(rng, [_P, side] if rng.random() < .5 else [side, _P], 0)
        outer = ('C', content)
        return _group(rng, [outer, _P] if rng.random() < .5 else [_P, outer])
    loop = _random_loop(rng, 1, 3 if kind == 'test' else 2)
    roots = [loop]
    if rng.random() < .55:
        roots.append(_random_loop(rng, 2, 2) if rng.random() < .2 else _P)
    rng.shuffle(roots)
    return _group(rng, roots)


def _blob(rng, compact):
    h, w = (int(rng.integers(1, 3 if compact else 5)) for _ in range(2))
    mask = np.ones((h, w), dtype=bool)
    if rng.random() < .7:
        for _ in range(min(6, h*w//2)):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            trial = mask.copy()
            trial[r, c] = False
            if trial.sum() >= .55*h*w and len(_components(trial)) == 1 and not _holes(trial).any():
                mask = trial
    y, x = np.nonzero(mask)
    return mask[y.min():y.max()+1, x.min():x.max()+1]


def _contour(rng, shape, padding):
    """A closed orthogonal path; independent dents stay outside child corridors."""
    h, w = shape
    top, bottom, left, right = padding

    def profile(length, first, last, cap):
        values = np.zeros(length, dtype=int)
        if cap > 0 and first < last and rng.random() < .9:
            steps = rng.integers(0, cap+1, size=last-first)
            if not steps.any():
                steps[len(steps)//2] = 1
            values[first:last] = steps
        return values

    t = profile(w, left, w-right, top-2)
    b = h-1-profile(w, left, w-right, bottom-2)
    l = profile(h, top, h-bottom, left-2)
    r = w-1-profile(h, top, h-bottom, right-2)
    path = [(int(t[x]), x) for x in range(w)]
    path += [(y, int(r[y])) for y in range(1, h)]
    path += [(int(b[x]), x) for x in range(w-2, -1, -1)]
    path += [(y, int(l[y])) for y in range(h-2, -1, -1)]
    path.append(path[0])
    mask = np.zeros(shape, dtype=bool)
    for (y, x), (yy, xx) in zip(path, path[1:]):
        mask[min(y, yy):max(y, yy)+1, x] = True
        mask[yy, min(x, xx):max(x, xx)+1] = True
    return mask


def _render(node, rng, compact):
    serial, exterior = [0], set()

    def draw(obj, depth):
        if obj[0] == 'P':
            key = serial[0]
            serial[0] += 1
            if depth == 0:
                exterior.add(key)
            mask = _blob(rng, compact)
            leaf = (key, (mask.shape[0]-1)//2, (mask.shape[1]-1)//2)
            return (mask, [leaf]), (np.ones((1, 1), dtype=bool), [(key, 0, 0)])
        if obj[0] == 'G':
            pairs = [draw(child, depth) for child in obj[2]]
            versions, axis = [], obj[1]
            for version in (0, 1):
                parts = [p[version] for p in pairs]
                offsets = [-p[1][anchor][2-axis] for p, anchor in zip(parts, obj[3])]
                gaps = ([int(rng.integers(1, 3 if compact else 6)) for _ in parts[:-1]]
                        if version == 0 else [1]*(len(parts)-1))
                versions.append(_join(parts, axis, offsets, gaps))
            return tuple(versions)
        metric, canonical = draw(obj[1], depth+1)
        padding = [int(rng.integers(2, 4 if compact else 6)) for _ in range(4)]
        top, bottom, left, right = padding
        h, w = metric[0].shape
        mask = _contour(rng, (h+top+bottom, w+left+right), padding)
        mask[top:top+h, left:left+w] |= metric[0]
        leaves = [(key, y+top, x+left) for key, y, x in metric[1]]
        return (mask, leaves), _frame(canonical)

    metric, canonical = draw(node, 0)
    return metric[0], _halo(canonical, exterior)


def _arranged_features(info):
    objects, mixed, depth = info['objects'], False, 0
    for i, obj in enumerate(objects):
        if obj['loop']:
            d, parent = 1, info['parents'][i]
            while parent is not None:
                d += 1
                parent = info['parents'][parent]
            depth = max(depth, d)
    for ids in [info['roots']] + [o['children'] for o in objects]:
        if len(ids) >= 3:
            centers = np.array([objects[i]['lo']+objects[i]['hi'] for i in ids])
            mixed |= bool(np.ptp(centers[:, 0]) and np.ptp(centers[:, 1]))
    return depth, mixed, any(not objects[i]['loop'] for i in info['roots'])


def _tree(info):
    def visit(ids):
        return tuple(sorted(('C', visit(info['objects'][i]['children'])) if info['objects'][i]['loop']
                            else ('P',) for i in ids))
    return visit(info['roots'])


def _pair_candidate(core, expected, palette, seen, held_out, witness):
    if max(*core.shape, *expected.shape) > 30:
        return None
    bg, fg = palette
    grid = np.where(core, fg, bg).astype(np.int8)
    if np.count_nonzero(grid == bg) <= grid.size//2:
        return None
    if _key(grid) in seen or _key(expected) in held_out or _hash(grid) in _PARENT_INPUT_HASHES:
        return None
    if not (grid.shape[0] > expected.shape[0] and grid.shape[1] > expected.shape[1]):
        return None
    try:
        info = _parse(grid)
        canonical = _canonical(info)
        if not np.array_equal(canonical, expected):
            return None
        normalized = _parse(expected.astype(np.int8), 0)
        if not np.array_equal(_canonical(normalized), expected) or _tree(info) != _tree(normalized):
            return None
        d, mixed, exterior = _arranged_features(normalized)
        if witness and not (d >= 2 and mixed and exterior):
            return None
        output = _collapse_vectors(np.where(expected, fg, bg).astype(np.int8))
        if not np.array_equal(output, solve(grid)):
            return None
    except ValueError:
        return None
    return {'input': grid, 'output': output}


def _make_pair(rng, palette, kind, seen, held_out):
    witness = kind == 'witness'
    # Failed draws occupy only RAM. The compact regime makes rich scenes fit.
    for attempt in range(96):
        core, canonical = _render(_scene(rng, kind), rng, compact=attempt >= 12 or witness)
        if max(*core.shape, *canonical.shape) > 30:
            continue
        turns = int(rng.integers(4))
        core, canonical = np.rot90(core, turns), np.rot90(canonical, turns)
        if rng.random() < .5:
            core, canonical = core[:, ::-1], canonical[:, ::-1]
        h, w = core.shape
        height = int(rng.integers(h, min(30, h+5)+1))
        width = int(rng.integers(w, min(30, w+5)+1))
        y, x = int(rng.integers(height-h+1)), int(rng.integers(width-w+1))
        canvas = np.zeros((height, width), dtype=bool)
        canvas[y:y+h, x:x+w] = core
        pair = _pair_candidate(canvas, canonical, palette, seen, held_out, witness)
        if pair is not None:
            return pair
    # Deterministic, finite fallback: procedural scenes, never source fixtures.
    # 128 injective stretch profiles per scene exceed the 13 seen-input bound.
    bases = []
    if witness:
        bases.extend(_render(_scene(rng, 'witness'), rng, True)[1] for _ in range(16))
    else:
        for depth in range(1, 4):
            for count in range(1, 6):
                mask = np.zeros((1, 2*count-1), dtype=bool)
                mask[0, ::2] = True
                for _ in range(depth):
                    mask = _frame((mask, []))[0]
                bases.extend((mask, mask.T))
    for canonical in bases:
        if _key(canonical) in held_out:
            continue
        for profile in range(128):
            rows, cols = np.ones(canonical.shape[0], dtype=int), np.ones(canonical.shape[1], dtype=int)
            rows[0] = cols[0] = 2
            # Distinct positions in the canonical sequence get distinct run
            # lengths; the last boundary is reserved for the seventh bit.
            for bit in range(3):
                rows[1+bit] = 1 + ((profile >> bit) & 1)
                cols[1+bit] = 1 + ((profile >> (bit+3)) & 1)
            rows[-1] = 1 + ((profile >> 6) & 1)
            core = np.pad(np.repeat(np.repeat(canonical, rows, 0), cols, 1), 2)
            pair = _pair_candidate(core, canonical, palette, seen, held_out, witness)
            if pair is not None:
                return pair
    raise RuntimeError('bounded candidate search exhausted')


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(name + ' is outside the allowed integer range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    _require(num_train > num_test, 'num_train must exceed num_test')
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             'seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    base_palette = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    palette_mode = int(rng.integers(3))
    puzzle, seen, train_diagrams = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if palette_mode == 0:
                palette = base_palette
            elif palette_mode == 1:
                palette = base_palette if index % 2 == 0 else base_palette[::-1]
            else:
                palette = tuple(int(c) for c in rng.choice(10, 2, replace=False))
            kind = ('simple' if index == 0 else 'witness' if index == 1 else 'train') if split == 'train' else 'test'
            pair = _make_pair(rng, palette, kind, seen, train_diagrams if split == 'test' else set())
            seen.add(_key(pair['input']))
            if split == 'train':
                train_diagrams.add(_key(pair['output'] == palette[1]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys must be train/test')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _count('training length', nt, 2, 10)
    _count('test length', ne, 1, 4)
    _require(nt > ne, 'training length must exceed test length')
    seen, train_keys, test_keys, witnesses = set(), set(), [], []
    a_changes = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys must be input/output')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            _require(_hash(inp) not in _PARENT_INPUT_HASHES, 'parent input copied')
            seen.add(_key(inp))
            info = _parse(inp)
            mask = _canonical(info)
            canonical = np.where(mask, info['fg'], info['bg']).astype(np.int8)
            _require(np.array_equal(out, _collapse_vectors(canonical)), 'incorrect composed output')
            _require(np.array_equal(out, solve(inp)), 'input-only solve disagrees')
            _require(not np.array_equal(inp, out), 'identity pair')
            _require(set(np.unique(inp)) == set(np.unique(out)), 'palette roles not retained')
            normalized = _parse(out, info['bg'])
            _require(_tree(info) == _tree(normalized), 'containment or leaf count changed')
            _require(np.array_equal(_canonical(normalized), out == info['fg']), 'noncanonical spacing or crop')
            for obj in normalized['objects']:
                if not obj['loop']:
                    _require(len(obj['cells']) == 1, 'blob was not reduced to one point')
                    continue
                h, w = obj['hi']-obj['lo']+1
                _require(len(obj['cells']) == 2*(h+w)-4, 'contour is not a one-cell rectangular frame')
                children = [normalized['objects'][i] for i in obj['children']]
                _require(np.array_equal(np.min([c['lo'] for c in children], axis=0), obj['lo']+2), 'wrong upper/left moat')
                _require(np.array_equal(np.max([c['hi'] for c in children], axis=0), obj['hi']-2), 'wrong lower/right moat')
            _require(np.array_equal(_collapse_vectors(out), out), 'output has redundant consecutive vectors')
            if split == 'train':
                train_keys.add(_key(mask))
                witnesses.append(_arranged_features(normalized))
                a_changes |= not np.array_equal(_collapse_vectors(inp), out)
            else:
                test_keys.append(_key(mask))
    _require(any(d >= 2 and mixed and exterior for d, mixed, exterior in witnesses),
             'training lacks nested mixed layout with an exterior anchor')
    _require(a_changes, 'training does not distinguish omission of A')
    _require(all(key not in train_keys for key in test_keys), 'test diagram already shown in training')
    # Deliberately no false claim that a B-changing canonical witness exists.


def self_test():
    """42 sequential puzzles, compact hash/feature sets, no saved corpus."""
    input_shapes, output_shapes, diagrams, trees, palettes, puzzle_hashes = (set() for _ in range(6))
    metrics = dict(seeds=42, pairs=0, b_changes=0, a_omission_changes=0, nonrectangular_contours=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info = _parse(inp)
                mask = _canonical(info)
                input_shapes.add(inp.shape)
                output_shapes.add(out.shape)
                diagrams.add(_key(mask))
                trees.add(_tree(info))
                palettes.add((info['bg'], info['fg']))
                digest.update(bytes.fromhex(_hash(inp)))
                digest.update(bytes.fromhex(_hash(out)))
                metrics['pairs'] += 1
                metrics['b_changes'] += int(not np.array_equal(_collapse_vectors(mask), mask))
                metrics['a_omission_changes'] += int(not np.array_equal(_collapse_vectors(inp), out))
                for obj in info['objects']:
                    if obj['loop']:
                        y, x = obj['lo']; yy, xx = obj['hi']
                        local = obj['mask'][y:yy+1, x:xx+1]
                        metrics['nonrectangular_contours'] += int(not (local[0].all() and local[-1].all()
                                                            and local[:, 0].all() and local[:, -1].all()))
        puzzle_hashes.add(digest.digest())
        # Rotation, palette permutation, and translation require no sampler state.
        inp, out = puzzle['test'][0]['input'], puzzle['test'][0]['output']
        permutation = np.random.default_rng(1000+seed).permutation(10)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        assert np.array_equal(solve(np.rot90(inp)), np.rot90(out))
        if max(inp.shape) <= 28:
            bg = _parse(inp)['bg']
            assert np.array_equal(solve(np.pad(inp, 1, constant_values=bg)), out)
    assert len(puzzle_hashes) == 42
    assert len(input_shapes) >= 50 and len(output_shapes) >= 20
    assert len(diagrams) >= 60 and len(trees) >= 15 and len(palettes) >= 30
    assert metrics['nonrectangular_contours'] >= 40
    assert metrics['b_changes'] == 0  # Explicitly measured limitation, not hidden.
    assert metrics['a_omission_changes'] >= 168
    first, second = generate_puzzle(209, 10, 4), generate_puzzle(209, 10, 4)
    assert all(_key(a[k]) == _key(b[k]) for s in ('train', 'test') for a, b in zip(first[s], second[s])
               for k in ('input', 'output'))
    validate_generated_puzzle(generate_puzzle(210, 2, 1))
    for a, b in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4,1.5), (None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Actual B work on arbitrary vector inputs: unequal run lengths, both axes,
    # and separated recurrences. This is an operation regression, not a claim
    # that such redundant rows exist after canonicalization.
    base = np.array([[1,0,1], [0,1,0], [1,0,1]], dtype=np.int8)
    expanded = np.repeat(np.repeat(base, [2,1,3], 0), [1,3,2], 1)
    assert np.array_equal(_collapse_vectors(expanded), base)
    rows_first = expanded[np.r_[True, np.any(expanded[1:] != expanded[:-1], axis=1)]]
    cols_first = expanded[:, np.r_[True, np.any(expanded[:, 1:] != expanded[:, :-1], axis=0)]]
    assert np.array_equal(_collapse_vectors(rows_first), _collapse_vectors(cols_first))
    # Wrong points and excess output margins must fail geometric validation.
    for mode in (0, 1):
        broken = {s: [{k:v.copy() for k,v in p.items()} for p in first[s]] for s in first}
        target = broken['test'][0]
        bg = _parse(target['input'])['bg']
        if mode == 0:
            parsed = _parse(target['output'], bg)
            leaf = next(o for o in parsed['objects'] if not o['loop'])
            target['output'][tuple(leaf['lo'])] = bg
        else:
            target['output'] = np.pad(target['output'], 1, constant_values=bg)
        try:
            validate_generated_puzzle(broken)
        except ValueError:
            pass
        else:
            raise AssertionError('corrupt output accepted')
    metrics.update(input_shapes=len(input_shapes), output_shapes=len(output_shapes),
                   canonical_diagrams=len(diagrams), containment_trees=len(trees), palettes=len(palettes))
    return metrics


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
