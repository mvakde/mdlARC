"""Overflow by containment: distorted reservoir diagrams to filled cup trees.

Only NumPy and the standard library are used. No parent module is imported.
Generation constructs a diagram and an abstract answer separately; solve parses
only visible cells. Validation additionally reads cup pixels and conservation.
"""
from __future__ import annotations

import hashlib
from collections import Counter
import numpy as np


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(np.asarray(cells, dtype=int))
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    free = ~padded
    exterior = np.zeros_like(free)
    stack = [(0, 0)]
    exterior[0, 0] = True
    h, w = free.shape
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < h and 0 <= xx < w and free[yy, xx] and not exterior[yy, xx]:
                exterior[yy, xx] = True
                stack.append((yy, xx))
    return (free & ~exterior)[1:-1, 1:-1]


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or a.dtype.kind not in 'iu' or not all(1 <= d <= 30 for d in a.shape):
        raise ValueError('expected an integer grid of dimensions 1..30')
    if np.any((a < 0) | (a > 9)):
        raise ValueError('colors must be 0..9')
    return a.astype(np.int8, copy=False)


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    if not np.all(border == bg) or len(np.unique(a)) != 4:
        raise ValueError('a blank border and four distinct color roles are required')
    records, loops, blobs = [], [], []
    labels = np.zeros(a.shape, dtype=int)
    for cells in _components(a != bg):
        mask = np.zeros(a.shape, dtype=bool)
        mask[cells[:, 0], cells[:, 1]] = True
        hole = _holes(mask)
        lo, hi = cells.min(0), cells.max(0)
        colors, counts = np.unique(a[mask], return_counts=True)
        obj = dict(cells=cells, mask=mask, hole=hole, lo=lo, hi=hi)
        i = len(records)
        if hole.any():
            if len(_components(hole)) != 1 or len(colors) != 2:
                raise ValueError('each reservoir must be one closed two-color contour')
            order = np.argsort(counts)
            cap = int(counts[order[0]])
            if not 1 <= cap <= 3 or counts[order[1]] <= cap:
                raise ValueError('capacity is 1..3 minority-colored wall cells')
            obj.update(kind='C', cap=cap, mark=int(colors[order[0]]),
                       wall=int(colors[order[1]]))
            loops.append(i)
        else:
            if len(colors) != 1 or len(cells) < .35 * np.prod(hi-lo+1):
                raise ValueError('a drop must be a compact solid monochrome blob')
            obj.update(kind='B', color=int(colors[0]))
            blobs.append(i)
        records.append(obj)
        labels[mask] = i+1
    for x, y in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        if np.any((x > 0) & (y > 0) & (x != y)):
            raise ValueError('distinct objects must have background clearance')
    if not loops or not blobs:
        raise ValueError('at least one reservoir and one drop are required')
    walls = {records[i]['wall'] for i in loops}
    marks = {records[i]['mark'] for i in loops}
    liquids = {records[i]['color'] for i in blobs}
    if len(walls) != 1 or len(marks) != 1 or len(liquids) != 1:
        raise ValueError('color roles must agree throughout the diagram')
    palette = (bg, next(iter(walls)), next(iter(marks)), next(iter(liquids)))
    if len(set(palette)) != 4:
        raise ValueError('color roles cannot overlap')
    parents = []
    for i, obj in enumerate(records):
        containers = []
        for j in loops:
            if i == j:
                continue
            inside = records[j]['hole'][obj['cells'][:, 0], obj['cells'][:, 1]]
            if inside.any() and not inside.all():
                raise ValueError('crossing contours')
            if inside.all():
                containers.append(j)
        parents.append(min(containers, key=lambda j: int(records[j]['hole'].sum()))
                       if containers else None)
    roots = [i for i in loops if parents[i] is None]
    if len(roots) != 1 or any(parents[i] is None for i in blobs):
        raise ValueError('one outer reservoir must contain every object')

    def build(i):
        kids = [j for j in loops if parents[j] == i]
        kids.sort(key=lambda j: int(records[j]['lo'][1]))
        for left, right in zip(kids, kids[1:]):
            if records[right]['lo'][1] <= records[left]['hi'][1]+1:
                raise ValueError('sibling reservoirs need separated horizontal projections')
        return dict(cap=records[i]['cap'], drops=sum(parents[j] == i for j in blobs),
                    kids=[build(j) for j in kids])
    return build(roots[0]), palette, records, parents


def _flow(node):
    """A postorder balance: children retain their share before passing surplus."""
    kids = [_flow(child) for child in node['kids']]
    available = node['drops'] + sum(k['spill'] for k in kids)
    return dict(cap=node['cap'], drops=node['drops'], kids=kids,
                fill=min(node['cap'], available), spill=max(0, available-node['cap']))


def _cup(cap, fill, palette):
    bg, wall, _, water = palette
    a = np.full((cap+1, 3), bg, dtype=np.int8)
    a[:, 0] = a[:, -1] = wall
    a[-1] = wall
    if fill:
        a[cap-fill:cap, 1] = water
    return a


def _draw_answer(node, palette, sink=True, placements=None):
    """Canonical tree. Children have bottom-aligned tiles and a one-column gap."""
    flow = _flow(node)

    def draw(f):
        cup = _cup(f['cap'], f['fill'], palette)
        parts = [draw(k) for k in f['kids']]
        if not parts:
            return cup, [(0, 0, f['cap'], f['fill'], False)]
        h = max(t.shape[0] for t, _ in parts)
        w = sum(t.shape[1] for t, _ in parts)+len(parts)-1
        out = np.full((h+2+cup.shape[0], w), palette[0], dtype=np.int8)
        locations, x = [], 0
        for tile, places in parts:
            y = h-tile.shape[0]
            out[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
            locations.extend((yy+y, xx+x, c, v, s) for yy, xx, c, v, s in places)
            x += tile.shape[1]+1
        y, x = h+2, (w-3)//2
        out[y:, x:x+3] = cup
        locations.append((y, x, f['cap'], f['fill'], False))
        return out, locations

    result, locations = draw(flow)
    if sink:
        if flow['spill'] > 4:
            raise ValueError('the four-unit final collector would overflow')
        h, w = result.shape
        out = np.full((h+7, w), palette[0], dtype=np.int8)
        out[:h] = result
        x = (w-3)//2
        out[h+2:, x:x+3] = _cup(4, flow['spill'], palette)
        locations.append((h+2, x, 4, flow['spill'], True))
        result = out
    if placements is not None:
        placements.extend(locations)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All roles, ownership, capacities, and order are reconstructed from input."""
    tree, palette, _, _ = _parse(input_grid)
    return _draw_answer(tree, palette)


def _pack(tiles, gap, bg, axis=1, offsets=None):
    if offsets is None:
        offsets = [0]*len(tiles)
    cross = 1-axis
    shape = [0, 0]
    shape[axis] = sum(t.shape[axis] for t in tiles)+gap*(len(tiles)-1)
    shape[cross] = max(t.shape[cross]+o for t, o in zip(tiles, offsets))
    a = np.full(shape, bg, dtype=np.int8)
    pos = 0
    for tile, off in zip(tiles, offsets):
        start = [off, off]
        start[axis] = pos
        y, x = start
        a[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
        pos += tile.shape[axis]+gap
    return a


def _outline(h, w, pads, rng, compact):
    top, bottom, left, right = pads
    tc, bc, lc, rc = [p-2 for p in pads]
    def profile(n, cap):
        if compact or cap == 0:
            return np.zeros(n, dtype=int)
        style = int(rng.integers(3))
        if style == 0:
            return rng.integers(0, cap+1, size=n)
        t = np.linspace(0, 1, n)
        p = np.abs(2*t-1) if style == 1 else t
        return np.rint(p*cap).astype(int)
    t, b = profile(w, tc), h-1-profile(w, bc)
    l, r = profile(h, lc), w-1-profile(h, rc)
    xl, xr, yt, yb = lc, w-1-rc, tc, h-1-bc
    path = [(int(t[x]), x) for x in range(xl, xr+1)]
    path += [(yt, xr), (yt, int(r[yt]))]
    path += [(y, int(r[y])) for y in range(yt, yb+1)]
    path += [(yb, xr), (int(b[xr]), xr)]
    path += [(int(b[x]), x) for x in range(xr, xl-1, -1)]
    path += [(yb, xl), (yb, int(l[yb]))]
    path += [(y, int(l[y])) for y in range(yb, yt-1, -1)]
    path += [(yt, xl), path[0]]
    mask = np.zeros((h, w), dtype=bool)
    for (y, x), (yy, xx) in zip(path, path[1:]):
        mask[min(y, yy):max(y, yy)+1, x] = True
        mask[yy, min(x, xx):max(x, xx)+1] = True
    return mask


def _draw_input(node, palette, rng, compact=False, varied_blobs=False, narrow=False):
    bg, wall, mark, water = palette
    kids = [_draw_input(k, palette, rng, compact, narrow=narrow) for k in node['kids']]
    drops = []
    for i in range(node['drops']):
        if varied_blobs and i < 2:
            h = w = 2 if i == 0 else 1
        elif compact:
            # Compact does not turn every test into a point-only diagram.
            h, w = (2, 1) if i == 0 and not kids else (1, 1)
        else:
            h, w = map(int, rng.integers(1, 4, size=2))
        blob = np.full((h, w), water, dtype=np.int8)
        if min(h, w) >= 2 and rng.random() < .45 and not (varied_blobs and i == 0):
            blob[0, 0] = bg
        drops.append(blob)
    content = None
    if kids:
        gap = 1 if compact else int(rng.integers(1, 4))
        offsets = [0 if compact else int(rng.integers(2)) for _ in kids]
        content = _pack(kids, gap, bg, offsets=offsets)
    if drops:
        # Near-square point arrangements avoid a count-to-width lookup.
        cols = min(len(drops), (3 if kids else 1 if narrow else int(rng.integers(1, 3)))
                   if compact else int(rng.integers(1, 4)))
        rows = [_pack(drops[i:i+cols], 1 if compact else int(rng.integers(1, 3)), bg)
                for i in range(0, len(drops), cols)]
        band = _pack(rows, 1, bg, axis=0)
        if content is None:
            content = band
        else:
            # Direct drops can be above, below, or beside child reservoirs.
            axis = 0 if compact else int(rng.integers(2))
            parts = [band, content] if rng.random() < .5 else [content, band]
            content = _pack(parts, 1 if compact else int(rng.integers(1, 3)), bg, axis)
    if content is None:
        content = np.full((1, 1), bg, dtype=np.int8)
    pads = [2]*4 if compact else list(map(int, rng.integers(2, 5, size=4)))
    t, b, l, r = pads
    h, w = content.shape
    mask = _outline(h+t+b, w+l+r, pads, rng, compact)
    a = np.full(mask.shape, bg, dtype=np.int8)
    a[mask] = wall
    a[t:t+h, l:l+w] = content
    coords = np.argwhere(mask)
    selected = coords[rng.choice(len(coords), size=node['cap'], replace=False)]
    a[selected[:, 0], selected[:, 1]] = mark
    return a


def _node(cap, drops, kids=()):
    return dict(cap=int(cap), drops=int(drops), kids=list(kids))


def _sample_tree(rng, mode):
    c = lambda: int(rng.integers(1, 4))
    if mode == 'single':
        cap = c()
        return _node(cap, cap+int(rng.integers(1, 4)))
    if mode == 'fork':
        a, b = c(), c()
        if rng.random() < .5:
            # Two unit surpluses combine into a partly filled parent.
            root, own = 3, 0
            kids = [_node(a, a+1), _node(b, b+1)]
        else:
            # Two spilling siblings, a retaining sibling, and direct supply.
            root = int(rng.integers(2, 4))
            own = root-1
            partial = int(rng.integers(2, 4))
            kids = [_node(a, a+int(rng.integers(1, 3))),
                    _node(b, b+int(rng.integers(1, 3))),
                    _node(partial, int(rng.integers(1, partial)))]
        rng.shuffle(kids)
        return _node(root, own, kids)
    if mode == 'deep':
        a, b, root = c(), c(), c()
        leaf = _node(a, a+int(rng.integers(2, 4)))
        middle = _node(b, b-1, [leaf])
        sidecap = c()
        side = _node(sidecap, int(rng.integers(0, sidecap+3)))
        if rng.random() < .5:
            middle['kids'].append(side)
            rng.shuffle(middle['kids'])
            kids = [middle]
        else:
            kids = [middle, side]
            rng.shuffle(kids)
        if rng.random() < .35:
            extra = c()
            kids.append(_node(extra, int(rng.integers(0, extra+1))))
            rng.shuffle(kids)
        return _node(root, int(rng.integers(0, root+1)), kids)
    kids = []
    for _ in range(int(rng.integers(0, 4))):
        cap = c()
        kids.append(_node(cap, int(rng.integers(0, cap+3))))
    cap = c()
    return _node(cap, int(rng.integers(1, cap+3)), kids)


def _counts(num_train, num_test):
    for v, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    base_palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable_palette = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            mode = ('single' if index == 0 else 'fork' if index == 1 else 'random') if split == 'train' else 'deep'
            palette = base_palette if stable_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
            # Select topology before drawing: larger forks must not lose to an
            # easier tree merely because the first metric realization was big.
            for _ in range(80):
                tree = _sample_tree(rng, mode)
                if _total_drops(tree) >= 1 and _flow(tree)['spill'] <= 4:
                    break
            else:
                raise RuntimeError('bounded abstract sampling exhausted')
            for attempt in range(80):
                core = _draw_input(tree, palette, rng, compact=attempt >= 16,
                                   varied_blobs=mode == 'single', narrow=attempt >= 64)
                if max(core.shape) > 28:
                    continue
                # One-cell border is semantic; all extra metric padding is irrelevant.
                pads = [(1+int(rng.integers(min(3, 29-d))), 1) for d in core.shape]
                inp = np.pad(core, pads, constant_values=palette[0])
                out = _draw_answer(tree, palette)
                if max(out.shape) > 30 or _key(inp) in seen:
                    continue
                try:
                    recovered, inferred, _, _ = _parse(inp)
                    if recovered != tree or inferred != palette:
                        continue
                    if not np.array_equal(solve(inp), out):
                        continue
                except ValueError:
                    continue
                seen.add(_key(inp))
                puzzle[split].append({'input': inp, 'output': out})
                break
            else:
                raise RuntimeError('bounded diagram construction exhausted')
        rng.shuffle(puzzle[split])
    return puzzle


def _total_drops(tree):
    return tree['drops']+sum(_total_drops(k) for k in tree['kids'])


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _walk(tree):
    yield tree
    for k in tree['kids']:
        yield from _walk(k)


def _depth(tree):
    return 1+max([0]+[_depth(k) for k in tree['kids']])


def _inspect(inp, out):
    tree, palette, records, parents = _parse(inp)
    positions = []
    expected = _draw_answer(tree, palette, placements=positions)
    assert np.array_equal(expected, out), 'incorrect canonical cup tree'
    bg, wall, mark, water = palette
    assert mark not in out, 'capacity marks are replaced by cup depths'
    assert int((out == water).sum()) == _total_drops(tree), 'abstract drop volume was not conserved'
    terrain = np.zeros(out.shape, dtype=bool)
    for y, x, cap, fill, sink in positions:
        cup = out[y:y+cap+1, x:x+3]
        assert np.all(cup[:, 0] == wall) and np.all(cup[:, -1] == wall)
        assert np.all(cup[-1] == wall)
        assert np.all(cup[:cap-fill, 1] == bg)
        assert np.all(cup[cap-fill:cap, 1] == water)
        terrain[y:y+cap+1, x] = True
        terrain[y:y+cap+1, x+2] = True
        terrain[y+cap, x:x+3] = True
        if not sink:
            assert 1 <= cap <= 3
    assert np.array_equal(terrain, out == wall), 'extra or missing cup wall'
    assert len(_components(out == wall)) == len(positions), 'cups touched or merged'
    assert np.any(out[0] != bg) and np.any(out[-1] != bg)
    assert np.any(out[:, 0] != bg) and np.any(out[:, -1] != bg)
    flow = _flow(tree)
    nodes = list(_walk(flow))
    # The alternative with no inter-reservoir overflow retains only direct drops.
    ignores_a = any(n['fill'] != min(n['cap'], n['drops']) for n in nodes) or flow['spill'] > 0
    # Literal liquid-pixel conservation conflicts with object-to-unit abstraction.
    ignores_b = int((inp == water).sum()) != _total_drops(tree)
    return dict(depth=_depth(tree), nodes=len(nodes), drops=_total_drops(tree),
                capacities=tuple(n['cap'] for n in nodes), spill=flow['spill'],
                fills=tuple(n['fill'] for n in nodes),
                internal_spills=sum(n['spill'] > 0 for n in nodes[1:]),
                partial=sum(0 < n['fill'] < n['cap'] for n in nodes),
                empty=sum(n['fill'] == 0 for n in nodes),
                fork=any(len(n['kids']) > 1 for n in nodes),
                fanout=max(len(n['kids']) for n in nodes),
                merging=sum(sum(k['spill'] > 0 for k in n['kids']) >= 2 for n in nodes),
                ignores_a=ignores_a, ignores_b=ignores_b,
                blob_areas=tuple(len(r['cells']) for r in records if r['kind'] == 'B'),
                input_shape=inp.shape, output_shape=out.shape, palette=palette)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, stats = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            stats[split].append(_inspect(inp, out))
    assert any(s['depth'] == 1 and s['spill'] > 0 and len(set(s['blob_areas'])) > 1
               for s in stats['train']), 'missing size-independent unit-volume demonstration'
    assert any(s['fork'] and s['merging'] > 0 and s['partial'] > 0
               for s in stats['train']), 'missing interacting overflow / retention demonstration'
    assert any(s['ignores_a'] and s['ignores_b'] for s in stats['train'])
    assert all(s['depth'] <= 2 for s in stats['train'])
    assert all(s['depth'] == 3 and s['internal_spills'] >= 2 for s in stats['test'])


def self_test():
    """42 streamed seeds, metamorphic checks, causal counterexamples, mutations.

    No grids or execution traces are saved. Only bounded signatures are retained.
    """
    signatures, inputs, outputs, shapes, palettes, trees = set(), set(), set(), set(), set(), set()
    hist = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                tree, palette, _, _ = _parse(inp)
                stats = _inspect(inp, out)
                normalized = np.zeros_like(inp)
                normout = np.zeros_like(out)
                for i, color in enumerate(palette):
                    normalized[inp == color] = i
                    normout[out == color] = i
                inputs.add(_key(normalized))
                outputs.add(_key(normout))
                shapes.add(inp.shape)
                palettes.add(palette)
                trees.add(repr(tree))
                signature.update(inp.tobytes())
                signature.update(out.tobytes())
                hist['pairs'] += 1
                for name in ('partial', 'empty', 'fork', 'merging', 'ignores_a', 'ignores_b'):
                    hist[name] += bool(stats[name])
                hist['three_children'] += stats['fanout'] == 3
                hist['five_reservoirs'] += stats['nodes'] == 5
        signatures.add(signature.digest())
        pair = puzzle['test'][0]
        inp, out = pair['input'], pair['output']
        tree, palette, _, _ = _parse(inp)
        rng = np.random.default_rng(1000+seed)
        perm = rng.permutation(10)
        assert np.array_equal(solve(perm[inp]), perm[out])
        # Horizontal reflection reverses each sibling ordering and output.
        assert np.array_equal(solve(inp[:, ::-1]), out[:, ::-1])
        # A new compact metric realization of the SAME hierarchy/capacities.
        for _ in range(64):
            core = _draw_input(tree, palette, rng, compact=True, narrow=_ >= 48)
            if max(core.shape) <= 28:
                redrawn = np.pad(core, 1, constant_values=palette[0])
                break
        else:
            raise AssertionError('bounded metric redraw exhausted')
        assert np.array_equal(solve(redrawn), out)
        # Both transformations are causal counterexamples, not accuracy claims.
        flow = _flow(tree)
        assert flow['kids'] and any(k['spill'] for k in flow['kids'])
    assert len(signatures) == 42 and len(inputs) >= 190
    assert len(outputs) >= 90 and len(trees) >= 100
    assert len(shapes) >= 60 and len(palettes) >= 40
    assert hist['ignores_a'] >= 100 and hist['ignores_b'] >= 100
    assert hist['partial'] >= 42 and hist['fork'] >= 84 and hist['empty'] >= 5
    assert hist['merging'] >= 42 and hist['three_children'] >= 10 and hist['five_reservoirs'] >= 3
    # Hand-computable balances independent of geometric construction.
    leaf = _node(1, 3)
    mid = _node(2, 1, [leaf])
    root = _node(1, 1, [mid])
    f = _flow(root)
    assert (f['kids'][0]['kids'][0]['fill'], f['kids'][0]['fill'], f['fill'], f['spill']) == (1, 2, 1, 1)
    # Same pixel quantity can mean different abstract volume: one 2x2 blob vs
    # four isolated points. Same tree and wall dimensions, different water.
    palette = (0, 5, 2, 8)
    one = np.zeros((12, 12), dtype=np.int8)
    one[1, 1:11] = one[10, 1:11] = 5
    one[1:11, 1] = one[1:11, 10] = 5
    one[1, 3:5] = 2
    one[4:6, 4:6] = 8
    four = one.copy()
    four[four == 8] = 0
    for y, x in ((4, 4), (4, 7), (7, 4), (7, 7)):
        four[y, x] = 8
    assert (one == 8).sum() == (four == 8).sum() == 4
    assert (solve(one) == 8).sum() == 1 and (solve(four) == 8).sum() == 4
    # Change ownership alone: taking a drop from an underfull child into its
    # parent changes retained water, although total count/capacities do not.
    t1 = _node(2, 0, [_node(2, 1)])
    t2 = _node(2, 1, [_node(2, 0)])
    assert not np.array_equal(_draw_answer(t1, palette), _draw_answer(t2, palette))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(907, a, b)
            validate_generated_puzzle(p)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    p = generate_puzzle(29)
    q = generate_puzzle(29)
    assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for role in ('water', 'wall', 'crop'):
        q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = q['test'][0]
        _, pal, _, _ = _parse(pair['input'])
        if role == 'crop':
            pair['output'] = np.pad(pair['output'], 1, constant_values=pal[0])
        else:
            y, x = np.argwhere(pair['output'] == pal[3 if role == 'water' else 1])[0]
            pair['output'][y, x] = pal[0]
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupt output accepted')
    return dict(seeds=42, pairs=hist['pairs'], distinct_inputs=len(inputs),
                distinct_outputs=len(outputs), distinct_trees=len(trees),
                input_shapes=len(shapes), palettes=len(palettes), invariants=dict(hist))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
