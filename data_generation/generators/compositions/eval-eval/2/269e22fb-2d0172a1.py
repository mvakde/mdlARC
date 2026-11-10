"""Recover the least common room seen in an exact, transformed crop.

No parent generator, reference bitmap, examples, or generation state is used by
solve. The detached rectangular card supplies the query within each input.
"""
from itertools import combinations
import numpy as np


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _d4(a, k):
    return np.rot90(a if k < 4 else np.fliplr(a), k % 4).copy()


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    h, w = mask.shape
    result = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack, cells = [(int(r), int(c))], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(np.array(cells, dtype=int))
    return result


def _hole(mask):
    padded = np.pad(mask, 1)
    reached = np.zeros(padded.shape, bool)
    reached[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < h and 0 <= xx < w and not padded[yy, xx] and not reached[yy, xx]:
                reached[yy, xx] = True
                stack.append((yy, xx))
    return (~padded & ~reached)[1:-1, 1:-1]


def _diagram(mask):
    objects = []
    labels = np.full(mask.shape, -1, int)
    for i, cells in enumerate(_components(mask)):
        own = np.zeros(mask.shape, bool)
        own[tuple(cells.T)] = True
        hole = _hole(own)
        if hole.any():
            assert len(_components(hole)) == 1, 'multiple holes in a contour'
        else:
            assert len(cells) >= .4 * np.prod(cells.max(0)-cells.min(0)+1), 'noncompact blob'
        objects.append(dict(cells=cells, lo=cells.min(0), hi=cells.max(0), hole=hole,
                            kind='C' if hole.any() else 'B'))
        labels[tuple(cells.T)] = i
    assert objects
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        shifted = np.roll(labels, (dy, dx), (0, 1))
        valid = np.ones(mask.shape, bool)
        if dy:
            valid[:dy] = False
        if dx == 1:
            valid[:, :1] = False
        elif dx == -1:
            valid[:, -1:] = False
        assert not np.any(valid & (labels >= 0) & (shifted >= 0) & (labels != shifted)), 'touching objects'
    parents = []
    for i, child in enumerate(objects):
        candidates = []
        for j, obj in enumerate(objects):
            if i == j or obj['kind'] != 'C':
                continue
            inside = obj['hole'][tuple(child['cells'].T)]
            assert inside.all() or not inside.any(), 'partial containment'
            if inside.all():
                candidates.append(j)
        parents.append(min(candidates, key=lambda j: objects[j]['hole'].sum()) if candidates else None)
    children = {i: [] for i in range(len(objects))}
    roots = []
    for i, p in enumerate(parents):
        (roots if p is None else children[p]).append(i)
    assert len(roots) == 1 and objects[roots[0]]['kind'] == 'C', 'one complete enclosing scene required'
    for i, obj in enumerate(objects):
        assert bool(children[i]) == (obj['kind'] == 'C'), 'empty contour'
    return dict(objects=objects, parents=parents, children=children, root=roots[0])


def _ancestors(info, leaf):
    result = []
    p = info['parents'][leaf]
    while p is not None:
        result.append(p)
        p = info['parents'][p]
    return result


def _scope(info, touched):
    assert len(touched) >= 2, 'the crop must touch at least two blobs'
    return next(i for i in _ancestors(info, touched[0])
                if all(i in _ancestors(info, j) for j in touched[1:]))


def _descendant_leaves(info, scope):
    return [i for i, o in enumerate(info['objects']) if o['kind'] == 'B' and scope in _ancestors(info, i)]


def _tile(mask, leaves):
    return mask, leaves


def _join(tiles, axis, offsets, gap):
    cross = 1-axis
    low = min(offsets)
    starts, cursor, shape = [], 0, [0, 0]
    for tile, offset in zip(tiles, offsets):
        start = [0, 0]
        start[axis], start[cross] = cursor, offset-low
        starts.append(start)
        cursor += tile[0].shape[axis] + gap
        shape[cross] = max(shape[cross], start[cross]+tile[0].shape[cross])
    shape[axis] = cursor-gap
    mask = np.zeros(shape, bool)
    leaves = []
    for (part, points), (y, x) in zip(tiles, starts):
        h, w = part.shape
        mask[y:y+h, x:x+w] |= part
        leaves.extend((i, r+y, c+x) for i, r, c in points)
    return _tile(mask, leaves)


def _normalizer(info):
    """Slicing whitespace orders siblings; overlapping leaf projections align them.

    All legal decompositions must agree, including the positions of the leaves.
    This is important for mixed arrangements and anchors across nested contours.
    """
    objects, cache = info['objects'], {}

    def group(ids):
        ids = tuple(sorted(ids))
        if ids in cache:
            return cache[ids]
        if len(ids) == 1:
            i = ids[0]
            if objects[i]['kind'] == 'B':
                answer = _tile(np.ones((1, 1), bool), [(i, 0, 0)])
            else:
                content, points = group(info['children'][i])
                mask = np.pad(content, 2)
                mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
                answer = _tile(mask, [(j, r+2, c+2) for j, r, c in points])
        else:
            answers = []
            for axis in (0, 1):
                parts, end = [], -1
                for i in sorted(ids, key=lambda j: objects[j]['lo'][axis]):
                    if objects[i]['lo'][axis] > end+1:
                        parts.append([])
                    parts[-1].append(i)
                    end = max(end, int(objects[i]['hi'][axis]))
                if len(parts) < 2:
                    continue
                try:
                    tiles = [group(p) for p in parts]
                    cross, edges = 1-axis, []
                    for a, b in combinations(range(len(tiles)), 2):
                        shifts = set()
                        for i, r, c in tiles[a][1]:
                            for j, rr, cc in tiles[b][1]:
                                if max(objects[i]['lo'][cross], objects[j]['lo'][cross]) <= min(objects[i]['hi'][cross], objects[j]['hi'][cross]):
                                    shifts.add((r, c)[cross]-(rr, cc)[cross])
                        assert len(shifts) <= 1, 'competing anchors'
                        if shifts:
                            edges.append((a, b, shifts.pop()))
                    offsets = {0: 0}
                    for _ in tiles:
                        for a, b, delta in edges:
                            if a in offsets and b in offsets:
                                assert offsets[b] == offsets[a]+delta
                            elif a in offsets:
                                offsets[b] = offsets[a]+delta
                            elif b in offsets:
                                offsets[a] = offsets[b]-delta
                    assert len(offsets) == len(tiles), 'unanchored diagonal'
                    answers.append(_join(tiles, axis, [offsets[j] for j in range(len(tiles))], 1))
                except AssertionError:
                    continue
            assert answers, 'no consistent cardinal layout'
            answer = answers[0]
            assert all(_key(a[0]) == _key(answer[0]) and sorted(a[1]) == sorted(answer[1]) for a in answers)
        cache[ids] = answer
        return answer

    return lambda scope: group([scope])


def _decode(grid):
    grid = np.asarray(grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= n <= 30 for n in grid.shape) and np.all((grid >= 0) & (grid <= 9))
    bg = int(grid[0, 0])
    assert np.all(grid[0] == bg) and np.all(grid[-1] == bg)
    assert np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)
    colors = [int(c) for c in np.unique(grid) if c != bg]
    assert len(colors) == 2, 'three color roles required'
    cards = []
    for color in colors:
        cells = np.argwhere(grid == color)
        lo, hi = cells.min(0), cells.max(0)
        h, w = hi-lo+1
        if min(h, w) < 4:
            continue
        expected = np.zeros(grid.shape, bool)
        r, c = lo; rr, cc = hi
        expected[r, c:cc+1] = expected[rr, c:cc+1] = True
        expected[r:rr+1, c] = expected[r:rr+1, cc] = True
        if np.array_equal(expected, grid == color):
            cards.append((color, lo, hi))
    assert len(cards) == 1, 'query marker must be uniquely identifiable'
    marker, lo, hi = cards[0]
    fg = next(c for c in colors if c != marker)
    r, c = lo; rr, cc = hi
    query = grid[r+1:rr, c+1:cc].copy()
    assert set(np.unique(query)) == {bg, fg}
    atlas = grid == fg
    atlas[r:rr+1, c:cc+1] = False
    positions = np.argwhere(atlas)
    low, high = positions.min(0), positions.max(0)
    # The card is detached from the source bounding box by a background gap.
    assert rr < low[0]-1 or r > high[0]+1 or cc < low[1]-1 or c > high[1]+1
    atlas = atlas[low[0]:high[0]+1, low[1]:high[1]+1]
    return atlas, query == fg, bg, fg, marker


def _matches(atlas, query):
    answers = []
    h, w = query.shape
    for k in range(8):
        full = _d4(atlas, k)
        if h > full.shape[0] or w > full.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
        for swap in (0, 1):
            patch = query if not swap else ~query
            rows, cols = np.nonzero(np.all(windows == patch, axis=(-2, -1)))
            answers.extend((k, swap, int(r), int(c)) for r, c in zip(rows, cols))
    return answers


def _analyze(grid):
    atlas, query, bg, fg, marker = _decode(grid)
    matches = _matches(atlas, query)
    assert len(matches) == 1, 'query orientation, color assignment and location must be unique'
    k, swap, r, c = matches[0]
    full = _d4(atlas, k)
    info = _diagram(full)
    support = np.zeros(full.shape, bool)
    h, w = query.shape
    support[r:r+h, c:c+w] = True
    touched = [i for i, o in enumerate(info['objects'])
               if o['kind'] == 'B' and support[tuple(o['cells'].T)].any()]
    scope = _scope(info, touched)
    hidden = set(_descendant_leaves(info, scope))-set(touched)
    assert hidden, 'at least one unobserved leaf in the chosen room is required'
    partial = any(not support[tuple(info['objects'][i]['cells'].T)].all() for i in touched)
    centers = [i for i in touched if support[tuple((info['objects'][i]['lo']+info['objects'][i]['hi'])//2)]]
    tile = _normalizer(info)(scope)
    return dict(atlas=atlas, query=query, bg=bg, fg=fg, marker=marker, k=k, swap=swap,
                info=info, scope=scope, touched=touched, hidden=hidden, partial=partial,
                centers=centers, support=support, tile=tile)


def solve(input_grid: np.ndarray) -> np.ndarray:
    details = _analyze(input_grid)
    bg, fg = details['bg'], details['fg']
    if details['swap']:
        bg, fg = fg, bg
    return np.where(details['tile'][0], fg, bg).astype(np.int8)


def _group(rng, children, axis=None):
    if len(children) == 1:
        return children[0]
    return ('G', int(rng.integers(2)) if axis is None else axis, tuple(children),
            tuple(int(rng.integers(_leaf_count(c))) for c in children))


def _leaf_count(node):
    if node[0] == 'B':
        return 1
    if node[0] == 'C':
        return _leaf_count(node[1])
    return sum(_leaf_count(c) for c in node[2])


def _scene(rng, rich):
    leaves = [('B',)] * int(rng.integers(3, 5))
    if rich and rng.random() < .4:
        leaves[int(rng.integers(len(leaves)))] = ('C', _group(rng, [('B',)] * int(rng.integers(1, 3))))
    axis = int(rng.integers(2))
    if rng.random() < .45:
        cut = int(rng.integers(1, len(leaves)))
        inner_content = _group(rng, [_group(rng, leaves[:cut], 1-axis),
                                    _group(rng, leaves[cut:], 1-axis)], axis)
    else:
        inner_content = _group(rng, leaves, axis)
    inner = ('C', inner_content)
    extras = [('B',)] * int(rng.integers(1, 4))
    if rich and rng.random() < .3:
        extras[int(rng.integers(len(extras)))] = ('C', _group(rng, [('B',)] * int(rng.integers(1, 3))))
    children = [inner] + extras
    rng.shuffle(children)
    axis = int(rng.integers(2))
    if len(children) >= 3 and rng.random() < .55:
        cut = int(rng.integers(1, len(children)))
        content = _group(rng, [_group(rng, children[:cut], 1-axis),
                               _group(rng, children[cut:], 1-axis)], axis)
    else:
        content = _group(rng, children, axis)
    return ('C', content), inner


def _outline(rng, h, w, pads):
    mask = np.zeros((h, w), bool)
    if rng.random() < .18:
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
        return mask
    t, b, l, r = [p-2 for p in pads]

    def profile(length, cap):
        n = int(rng.integers(2, min(6, length)+1))
        values = rng.integers(0, cap+1, n)
        return np.rint(np.interp(np.arange(length), np.linspace(0, length-1, n), values)).astype(int)

    top, bottom = profile(w, t), h-1-profile(w, b)
    left, right = profile(h, l), w-1-profile(h, r)
    x0, x1, y0, y1 = l, w-1-r, t, h-1-b
    path = [(int(top[x]), x) for x in range(x0, x1+1)]
    path += [(y0, x1), (y0, int(right[y0]))]
    path += [(y, int(right[y])) for y in range(y0, y1+1)]
    path += [(int(bottom[x1]), int(right[y1])), (int(bottom[x1]), x1)]
    path += [(int(bottom[x]), x) for x in range(x1, x0-1, -1)]
    path += [(y1, x0), (y1, int(left[y1]))]
    path += [(y, int(left[y])) for y in range(y1, y0-1, -1)]
    path += [(int(top[x0]), int(left[y0])), path[0]]
    for (y, x), (yy, xx) in zip(path, path[1:]):
        mask[min(y, yy):max(y, yy)+1, x] = True
        mask[yy, min(x, xx):max(x, xx)+1] = True
    return mask


def _draw(rng, scene, inner, compact=False):
    serial, target_keys = [0], []

    def draw(node):
        if node[0] == 'B':
            i = serial[0]; serial[0] += 1
            h, w = (int(rng.choice([1, 3], p=[.65, .35] if compact else [.45, .55])) for _ in range(2))
            mask = np.ones((h, w), bool)
            if h == w == 3 and rng.random() < .7:
                for y, x in ((0, 0), (0, 2), (2, 0), (2, 2)):
                    if rng.random() < .5:
                        mask[y, x] = False
            return _tile(mask, [(i, h//2, w//2)]), _tile(np.ones((1, 1), bool), [(i, 0, 0)])
        if node[0] == 'G':
            pairs = [draw(c) for c in node[2]]
            axis, cross = node[1], 1-node[1]
            versions = []
            for v in (0, 1):
                tiles = [p[v] for p in pairs]
                offsets = [-tile[1][anchor][1+cross] for tile, anchor in zip(tiles, node[3])]
                gap = int(rng.integers(1, 3 if compact else 4)) if v == 0 else 1
                versions.append(_join(tiles, axis, offsets, gap))
            return tuple(versions)
        metric, canonical = draw(node[1])
        pads = [int(rng.integers(2, 4 if compact else 5)) for _ in range(4)]
        t, b, l, r = pads
        h, w = metric[0].shape
        mask = _outline(rng, h+t+b, w+l+r, pads)
        mask[t:t+h, l:l+w] |= metric[0]
        points = [(i, y+t, x+l) for i, y, x in metric[1]]
        out = np.pad(canonical[0], 2)
        out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = True
        if node is inner:
            target_keys.extend(i for i, _, _ in points)
        return _tile(mask, points), _tile(out, [(i, y+2, x+2) for i, y, x in canonical[1]])

    metric, canonical = draw(scene)
    occupied = np.argwhere(metric[0])
    lo, hi = occupied.min(0), occupied.max(0)
    metric = _tile(metric[0][lo[0]:hi[0]+1, lo[1]:hi[1]+1],
                   [(i, y-int(lo[0]), x-int(lo[1])) for i, y, x in metric[1]])
    # Identify the intended inner room through a known generated leaf, then
    # discard the construction's IDs; all cropping checks use visible pixels.
    info = _diagram(metric[0])
    point = next((y, x) for i, y, x in metric[1] if i == target_keys[0])
    leaf = next(i for i, o in enumerate(info['objects']) if o['kind'] == 'B' and np.any(np.all(o['cells'] == point, axis=1)))
    # A wrapped leaf may be one level deeper; select the room with >=3 leaves.
    inner_scope = next(i for i in _ancestors(info, leaf) if len(_descendant_leaves(info, i)) >= 3)
    normal = _normalizer(info)
    assert np.array_equal(normal(info['root'])[0], canonical[0]), 'metric reconstruction disagrees'
    return metric[0], info, inner_scope, normal


def _crop(rng, atlas, info, target, partial):
    leaves = _descendant_leaves(info, target)
    pairs = [(a, b) for a, b in combinations(leaves, 2) if _scope(info, [a, b]) == target]
    for _ in range(140):
        a, b = pairs[int(rng.integers(len(pairs)))]
        objs = [info['objects'][i] for i in (a, b)]
        lo = np.min([o['lo'] for o in objs], axis=0)
        hi = np.max([o['hi'] for o in objs], axis=0)
        low = np.maximum(0, lo-rng.integers(0, 5, 2))
        high = np.minimum(np.array(atlas.shape)-1, hi+rng.integers(0, 5, 2))
        if partial:
            axis = int(rng.integers(2))
            if rng.random() < .5:
                low[axis] = lo[axis]+2
            else:
                high[axis] = hi[axis]-2
        if np.any(high-low < 1):
            continue
        r, c = low; rr, cc = high+1
        patch = atlas[r:rr, c:cc]
        support = np.zeros(atlas.shape, bool)
        support[r:rr, c:cc] = True
        touched = [i for i, o in enumerate(info['objects']) if o['kind'] == 'B' and support[tuple(o['cells'].T)].any()]
        if len(touched) < 2 or _scope(info, touched) != target or len(touched) == len(leaves):
            continue
        is_partial = any(not support[tuple(info['objects'][i]['cells'].T)].all() for i in touched)
        if is_partial != partial or not patch.any() or patch.all():
            continue
        centers = [i for i in touched if support[tuple((info['objects'][i]['lo']+info['objects'][i]['hi'])//2)]]
        if partial:
            if len(centers) == len(touched) or not centers:
                continue
            # For the outer-room clipping demonstration, ignoring edge slivers
            # must choose a smaller room, even if only one complete blob remains.
            if target == info['root']:
                center_scope = _scope(info, centers) if len(centers) >= 2 else _ancestors(info, centers[0])[0]
                if center_scope == target:
                    continue
        elif target != info['root'] and not support[tuple(info['objects'][target]['cells'].T)].any():
            continue  # The inner-room witness crosses its contour boundary.
        if len(_matches(atlas, patch)) == 1:
            return patch
    return None


def _pack(rng, atlas, query, palette):
    bg, fg, marker = palette
    card = np.full(np.array(query.shape)+2, marker, np.int8)
    card[1:-1, 1:-1] = np.where(query, fg, bg)
    source = np.where(atlas, fg, bg).astype(np.int8)
    arrangements = []
    for axis in (0, 1):
        shape = [max(source.shape[d], card.shape[d]) for d in (0, 1)]
        shape[axis] = source.shape[axis]+card.shape[axis]+1
        if max(shape) <= 28:
            arrangements.append((axis, shape))
    if not arrangements:
        return None
    axis, shape = arrangements[int(rng.integers(len(arrangements)))]
    cross = 1-axis
    shape = [min(30, n+2+int(rng.integers(0, min(3, 29-n)))) for n in shape]
    grid = np.full(shape, bg, np.int8)
    tiles = [source, card]
    if rng.random() < .5:
        tiles.reverse()
    cursor = 1
    for tile in tiles:
        start = [0, 0]
        start[axis] = cursor
        start[cross] = int(rng.integers(1, shape[cross]-tile.shape[cross]))
        y, x = start
        grid[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
        cursor += tile.shape[axis]+1
    return grid


def _pair(rng, palette, wanted, k, swap, partial, seen):
    for attempt in range(220):
        scene, inner = _scene(rng, rich=attempt < 100)
        try:
            atlas, info, inner_scope, normal = _draw(rng, scene, inner, compact=attempt >= 12)
        except AssertionError:
            continue
        if max(atlas.shape) > 24:
            continue
        target = inner_scope if wanted == 'inner' else info['root']
        canonical = normal(target)[0]
        try:
            ci = _diagram(canonical)
            assert np.array_equal(_normalizer(ci)(ci['root'])[0], canonical)
        except AssertionError:
            continue  # Compression must preserve an unambiguous diagram too.
        obj = info['objects'][target]
        y, x = obj['lo']; yy, xx = obj['hi']
        if _key(atlas[y:yy+1, x:xx+1]) == _key(canonical):
            continue  # Every pair needs a substantive metric-to-topology change.
        patch = _crop(rng, atlas, info, target, partial)
        if patch is None:
            continue
        query = _d4(patch, k)
        if swap:
            query = ~query
        grid = _pack(rng, atlas, query, palette)
        if grid is None or _key(grid) in seen:
            continue
        bg, fg, _ = palette
        if swap:
            bg, fg = fg, bg
        expected = np.where(_d4(canonical, k), fg, bg).astype(np.int8)
        assert np.array_equal(solve(grid), expected)
        seen.add(_key(grid))
        return dict(input=grid, output=expected)
    raise RuntimeError('bounded scene/crop search exhausted')


def _counts(num_train, num_test):
    for name, n, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
            raise ValueError(name+' must be an integer in its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    palette = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    vary_palette = bool(rng.integers(2))
    first_pose = int(rng.integers(8))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, n in (('train', num_train), ('test', num_test)):
        for j in range(n):
            colors = tuple(int(c) for c in rng.choice(10, 3, replace=False)) if vary_palette else palette
            wanted = ('inner' if j == 0 else 'root') if split == 'train' and j < 2 else 'inner' if split == 'test' and j == 0 else str(rng.choice(['inner', 'root']))
            k = first_pose if split == 'train' and j == 0 else (first_pose+1) % 8 if split == 'train' and j == 1 else int(rng.integers(8))
            swap = j % 2 if split == 'train' and j < 2 else int(rng.integers(2))
            partial = split == 'test' or (split == 'train' and j == 1)
            puzzle[split].append(_pair(rng, colors, wanted, k, swap, partial, seen))
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for v in pair.values():
                a = np.asarray(v)
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= n <= 30 for n in a.shape) and np.all((a >= 0) & (a <= 9))
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _analyze(inp)
            assert np.array_equal(solve(inp), out)
            output_bg, output_fg = (d['fg'], d['bg']) if d['swap'] else (d['bg'], d['fg'])
            assert set(np.unique(out)) == {output_bg, output_fg}
            oi = _diagram(out == output_fg)
            assert len(_descendant_leaves(oi, oi['root'])) == len(_descendant_leaves(d['info'], d['scope']))
            def tree(info, i):
                return ('B',) if info['objects'][i]['kind'] == 'B' else ('C', tuple(sorted(tree(info, j) for j in info['children'][i])))
            assert tree(oi, oi['root']) == tree(d['info'], d['scope'])
            obj = d['info']['objects'][d['scope']]
            y, x = obj['lo']; yy, xx = obj['hi']
            raw_room = _d4(d['atlas'], d['k'])[y:yy+1, x:xx+1]
            assert _key(raw_room) != _key(d['tile'][0]), 'omitting normalization must change the answer'
            if d['partial']:
                assert 0 < len(d['centers']) < len(d['touched'])
                if d['scope'] == d['info']['root']:
                    cs = _scope(d['info'], d['centers']) if len(d['centers']) >= 2 else _ancestors(d['info'], d['centers'][0])[0]
                    assert cs != d['scope'], 'clipped-blob evidence must affect enclosure'
            elif d['scope'] != d['info']['root']:
                assert d['support'][tuple(obj['cells'].T)].any(), 'inner-room crop crosses the boundary'
            for i, o in enumerate(oi['objects']):
                if o['kind'] == 'B':
                    assert len(o['cells']) == 1
                else:
                    lo, hi = o['lo'], o['hi']
                    assert len(o['cells']) == 2*int(np.sum(hi-lo))
                    kids = [oi['objects'][j] for j in oi['children'][i]]
                    assert np.array_equal(np.min([o['lo'] for o in kids], axis=0), lo+2)
                    assert np.array_equal(np.max([o['hi'] for o in kids], axis=0), hi-2)
            assert np.array_equal(_normalizer(oi)(oi['root'])[0], out == output_fg)
            if split == 'test':
                assert d['partial']
            records[split].append(d)
    assert {False, True} <= {d['scope'] == d['info']['root'] for d in records['train']}
    assert {0, 1} <= {d['swap'] for d in records['train']}
    assert any(d['partial'] and d['scope'] == d['info']['root'] for d in records['train'])
    assert any(not d['partial'] and d['scope'] != d['info']['root'] for d in records['train'])
    assert any(d['scope'] != d['info']['root'] for d in records['test'])
    for d in records['train']:
        if d['scope'] != d['info']['root']:
            assert _key(_normalizer(d['info'])(d['info']['root'])[0]) != _key(d['tile'][0]), 'omitting crop localization must change the answer'


def self_test():
    """Bounded 42-seed invariants, metamorphic checks and corruption regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, structures, topologies, query_shapes, colors = set(), set(), set(), set(), set()
    poses, swaps, leaf_counts, contour_counts, offsets = set(), set(), set(), set(), set()
    nonrect, mixed, inner = 0, 0, 0
    signatures = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                d = _analyze(pair['input'])
                shapes.add(pair['input'].shape)
                query_shapes.add(d['query'].shape)
                poses.add(d['k']); swaps.add(d['swap'])
                colors.update(np.unique(pair['input']).tolist())
                leaf_counts.add(len(_descendant_leaves(d['info'], d['scope'])))
                contour_counts.add(sum(o['kind'] == 'C' for o in d['info']['objects']))
                structures.add(min(_key(_d4(d['tile'][0], k)) for k in range(8)))
                info = d['info']
                def topology(i):
                    return ('B',) if info['objects'][i]['kind'] == 'B' else ('C', tuple(sorted(topology(j) for j in info['children'][i])))
                topologies.add(topology(info['root']))
                inner += d['scope'] != info['root']
                offsets.add(_matches(d['atlas'], d['query'])[0][2:])
                for o in info['objects']:
                    if o['kind'] == 'C':
                        h, w = o['hi']-o['lo']+1
                        nonrect += len(o['cells']) != 2*(h+w-2)
                oi = _diagram(d['tile'][0])
                for ids in oi['children'].values():
                    if len(ids) >= 3:
                        centers = np.array([oi['objects'][i]['lo']+oi['objects'][i]['hi'] for i in ids])
                        mixed += bool(np.ptp(centers[:, 0]) and np.ptp(centers[:, 1]))
        # Colors and card placement carry no hidden semantic state.
        p = puzzle['test'][0]
        permutation = np.random.default_rng(len(signatures)+200).permutation(10)
        assert np.array_equal(solve(permutation[p['input']]), permutation[p['output']])
        if max(p['input'].shape) <= 28:
            assert np.array_equal(solve(np.pad(p['input'], 1, constant_values=p['input'][0, 0])), p['output'])
        for k in (1, 4):
            assert np.array_equal(solve(_d4(p['input'], k)), _d4(p['output'], k))
    assert len(signatures) == 42 and len(shapes) >= 35 and len(query_shapes) >= 25
    assert len(structures) >= 15 and len(topologies) >= 10
    assert poses == set(range(8)) and swaps == {0, 1} and colors == set(range(10))
    assert len(leaf_counts) >= 4 and len(contour_counts) >= 2 and len(offsets) >= 25
    assert nonrect >= 40 and mixed >= 15 and inner >= 42
    a, b = generate_puzzle(42, 2, 1), generate_puzzle(42, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(100+nt*4+nq, nt, nq)
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = bad['test'][0]['output'][1, 1]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt contour accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                query_shapes=len(query_shapes), dihedral_normalized_answers=len(structures),
                containment_topologies=len(topologies), query_poses=sorted(poses),
                source_contour_counts=sorted(contour_counts), answer_leaf_counts=sorted(leaf_counts),
                match_offsets=len(offsets), nonrectangular_contours=int(nonrect),
                mixed_sibling_regions=mixed, inner_scope_pairs=inner, colors=sorted(colors))
