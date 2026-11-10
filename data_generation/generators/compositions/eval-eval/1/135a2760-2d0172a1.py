"""Repair-site containment maps. Standard library and NumPy only.

No parent generator, fixture, file, seed, or demonstration is used by solve().
The sampler constructs intended words and an expected pruned diagram; the visible
input parser must independently recover that diagram before a pair is returned.
"""
from __future__ import annotations

from collections import Counter
from itertools import product
import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


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
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.array(cells, dtype=int))
    return result


def _holes(mask):
    a = np.pad(mask, 1)
    outside = np.zeros_like(a, dtype=bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = a.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < h and 0 <= cc < w and not a[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))
    return (~a & ~outside)[1:-1, 1:-1]


def _records(a, bg):
    records = []
    for color in sorted(int(v) for v in np.unique(a) if v != bg):
        for cells in _components(a == color):
            lo, hi = cells.min(0), cells.max(0)
            local = np.zeros(hi-lo+1, dtype=bool)
            local[tuple((cells-lo).T)] = True
            hole = _holes(local)
            if hole.any() and len(_components(hole)) != 1:
                raise ValueError("a contour must have exactly one enclosed region")
            full_hole = np.zeros(a.shape, dtype=bool)
            full_hole[lo[0]:hi[0]+1, lo[1]:hi[1]+1] = hole
            records.append(dict(color=color, cells=cells, lo=lo, hi=hi,
                                local=local, hole=full_hole, loop=bool(hole.any())))
    # Objects of different colors must also remain separate, including diagonals.
    labels = np.zeros(a.shape, dtype=int)
    for i, obj in enumerate(records):
        labels[tuple(obj['cells'].T)] = i+1
    for u, v in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        if np.any((u > 0) & (v > 0) & (u != v)):
            raise ValueError("distinct objects touch")
    return records


def _hierarchy(records):
    loops = [i for i, obj in enumerate(records) if obj['loop']]
    parents, children = {}, {i: [] for i in loops}
    for i, obj in enumerate(records):
        containers = []
        for j in loops:
            if i == j:
                continue
            flags = records[j]['hole'][tuple(obj['cells'].T)]
            if flags.any() and not flags.all():
                raise ValueError("crossing objects")
            if flags.all():
                containers.append(j)
        parent = min(containers, key=lambda j: int(records[j]['hole'].sum())) if containers else None
        parents[i] = parent
        if parent is not None:
            children[parent].append(i)
    roots = [i for i in range(len(records)) if parents[i] is None]
    if len(roots) != 1 or roots[0] not in loops:
        raise ValueError("expected exactly one outer contour and no exterior blobs")
    return loops, parents, children, roots[0]


def _recover_word(word):
    """All supported periods must agree on the repaired *word*, not just a score."""
    word = np.asarray(word, dtype=np.int8)
    symbols = np.unique(word)
    if len(symbols) != 2:
        raise ValueError("each band must visibly use both signal colors")
    n = len(word)
    candidates = []
    for p in range(2, n//3+1):
        unit = np.empty(p, dtype=np.int8)
        for phase in range(p):
            counts = [int(np.count_nonzero(word[phase::p] == c)) for c in symbols]
            if counts[0] == counts[1]:
                break
            unit[phase] = symbols[int(counts[1] > counts[0])]
        else:
            clean = unit[np.arange(n) % p]
            if len(np.unique(unit)) == 2 and np.count_nonzero(clean != word) <= max(1, n//4):
                candidates.append((p, clean))
    if not candidates:
        raise ValueError("no supported binary rhythm")
    repaired = candidates[0][1]
    if any(not np.array_equal(repaired, q) for _, q in candidates[1:]):
        raise ValueError("competing rhythms prescribe different repairs")
    return repaired, candidates[0][0]


def _band_axis(objects):
    choices = []
    for axis in (0, 1):
        cross = 1-axis
        centers = [int(o['lo'][cross]+o['hi'][cross]) for o in objects]
        ordered = sorted(objects, key=lambda o: int(o['lo'][axis]))
        if (len(set(centers)) == 1 and
                all(b['lo'][axis]-a['hi'][axis] >= 2 for a, b in zip(ordered, ordered[1:]))):
            choices.append(axis)
    if len(choices) != 1:
        raise ValueError("band must be one clearly aligned row or column")
    return choices[0]


def _group_axis(boxes):
    if len(boxes) == 1:
        return 0  # Unused: composing a singleton is axis independent.
    choices = []
    for axis in (0, 1):
        cross = 1-axis
        ordered = sorted(boxes, key=lambda b: int(b[0][axis]))
        if (len({int(lo[cross]+hi[cross]) for lo, hi in boxes}) == 1
                and all(b[0][axis]-a[1][axis] >= 2 for a, b in zip(ordered, ordered[1:]))):
            choices.append(axis)
    if len(choices) != 1:
        raise ValueError("bands and immediate child loops need a unique centered cardinal order")
    return choices[0]


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("input needs a uniform background border")
    if len(np.unique(a)) != 4:
        raise ValueError("expected background, contour ink, and two signal colors")
    records = _records(a, bg)
    loops, parents, children, root = _hierarchy(records)
    inks = {records[i]['color'] for i in loops}
    if len(inks) != 1:
        raise ValueError("all contours share one ink")
    ink = inks.pop()
    if any((o['color'] == ink) != o['loop'] for o in records):
        raise ValueError("contour ink is reserved for closed contours")
    nodes = {}
    for i in loops:
        blobs = [j for j in children[i] if not records[j]['loop']]
        nested = [j for j in children[i] if records[j]['loop']]
        if len(blobs) < 6:
            raise ValueError("every contour needs its own repeated direct-blob band")
        for j in blobs:
            o = records[j]
            if len(o['cells']) < .4*o['local'].size:
                raise ValueError("terminal blobs must be compact and filled")
        axis = _band_axis([records[j] for j in blobs])
        blobs.sort(key=lambda j: int(records[j]['lo'][axis]))
        word = np.array([records[j]['color'] for j in blobs], dtype=np.int8)
        repaired, period = _recover_word(word)
        lo = np.min([records[j]['lo'] for j in blobs], axis=0)
        hi = np.max([records[j]['hi'] for j in blobs], axis=0)
        boxes = [(lo, hi)] + [(records[j]['lo'], records[j]['hi']) for j in nested]
        group_axis = _group_axis(boxes)
        order = [None] + nested
        order.sort(key=lambda j: int((lo if j is None else records[j]['lo'])[group_axis]))
        nodes[i] = dict(blobs=blobs, word=word, clean=repaired, period=period,
                        bad=np.flatnonzero(word != repaired).tolist(), axis=axis,
                        group_axis=group_axis, order=order, band_box=(lo, hi))
    info = dict(grid=a, bg=bg, ink=ink, records=records, loops=loops,
                parents=parents, children=children, root=root, nodes=nodes)
    if not any(n['bad'] for n in nodes.values()):
        raise ValueError("a diagnostic input must contain at least one fault")
    return info


def _points(colors, axis, bg):
    if not len(colors):
        return None
    row = np.full((1, 2*len(colors)-1), bg, dtype=np.int8)
    row[0, ::2] = colors
    return row.T.copy() if axis == 0 else row


def _compose(tiles, axis, bg, gaps=None):
    if not tiles:
        return None
    if gaps is None:
        gaps = [1]*(len(tiles)-1)
    cross = 1-axis
    shape = [0, 0]
    shape[axis] = sum(t.shape[axis] for t in tiles)+sum(gaps)
    shape[cross] = max(t.shape[cross] for t in tiles)
    result = np.full(shape, bg, dtype=np.int8)
    cursor = 0
    for k, tile in enumerate(tiles):
        start = [0, 0]
        start[axis] = cursor
        start[cross] = (shape[cross]-tile.shape[cross])//2
        r, c = start
        result[r:r+tile.shape[0], c:c+tile.shape[1]] = tile
        cursor += tile.shape[axis]+(gaps[k] if k < len(gaps) else 0)
    return result


def _frame(content, bg, ink):
    result = np.pad(content, 2, constant_values=bg)
    result[0, :] = result[-1, :] = ink
    result[:, 0] = result[:, -1] = ink
    return result


def _diagnostic(info, include_correct=False):
    def visit(i):
        node = info['nodes'][i]
        colors = node['clean'] if include_correct else node['clean'][node['bad']]
        band = _points(colors, node['axis'], info['bg'])
        parts = []
        for j in node['order']:
            tile = band if j is None else visit(j)
            if tile is not None:
                parts.append(tile)
        content = _compose(parts, node['group_axis'], info['bg'])
        return None if content is None else _frame(content, info['bg'], info['ink'])
    return visit(info['root'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer scoped rhythms, extract repair sites, and redraw their ancestor tree."""
    result = _diagnostic(_parse(input_grid))
    if result is None or max(result.shape) > 30:
        raise ValueError("diagnostic diagram does not fit the family bounds")
    return result


def _sample_word(rng, n, errors, period=None):
    available = list(range(max(2, errors), min(4, n//3)+1))
    for _ in range(256):
        p = int(rng.choice(available)) if period is None else period
        unit = rng.integers(0, 2, p, dtype=np.int8)
        if len(np.unique(unit)) != 2:
            continue
        clean = unit[np.arange(n) % p]
        noisy = clean.copy()
        if errors:
            phases = rng.choice(p, errors, replace=False)
            indices = [int(rng.choice(np.arange(int(s), n, p))) for s in phases]
            if errors == 2 and len(set(int(clean[k]) for k in indices)) != 2:
                continue
            noisy[indices] = 1-noisy[indices]
        try:
            actual, inferred_p = _recover_word(noisy)
        except ValueError:
            continue
        if inferred_p == p and np.array_equal(actual, clean):
            return noisy, clean
    # Finite exhaustive fallback: enumerate words and choices of error phases.
    for p in available if period is None else [period]:
        for mask in range(1, 2**p-1):
            unit = np.array([(mask >> j) & 1 for j in range(p)], dtype=np.int8)
            clean = unit[np.arange(n) % p]
            choices = [()] if errors == 0 else [(j,) for j in range(p)] if errors == 1 else [
                (j, k) for j in range(p) for k in range(j+1, p) if unit[j] != unit[k]]
            for positions in choices:
                noisy = clean.copy()
                noisy[list(positions)] = 1-noisy[list(positions)]
                try:
                    actual, inferred_p = _recover_word(noisy)
                except ValueError:
                    continue
                if inferred_p == p and np.array_equal(actual, clean):
                    return noisy, clean
    raise RuntimeError("no unambiguous word for requested parameters")


def _plan(rng, kind, small=False, count=None):
    def node(children=(), axis=1, group=0, errors=None, n=None, period=None):
        if n is None:
            n = int(rng.integers(6, 10 if small else 13))
        if errors is None:
            choices = [0, 1, 2, 3] if n == 12 else [0, 1, 2] if n >= 8 else [0, 1]
            errors = int(rng.choice(choices))
        noisy, clean = _sample_word(rng, n, errors, period)
        order = [-1]+list(range(len(children)))
        rng.shuffle(order)
        return dict(children=list(children), axis=axis, group=group,
                    noisy=noisy, clean=clean, order=order)
    if kind == 'pruning':
        # Independently sampled words; two opposite repairs plus a healthy branch.
        bad = node(errors=1, n=9, period=3)
        good = node(errors=0, n=int(rng.integers(6, 10)), period=2)
        return node([bad, good], errors=2, n=int(rng.integers(10, 13)), period=2)
    if kind == 'ancestry':
        leaf = node(errors=1, n=int(rng.integers(6, 9)), period=2)
        middle = node([leaf], errors=2, n=10, period=2)
        return node([middle], errors=0, n=int(rng.integers(6, 11)))
    if count is None:
        count = int(rng.integers(2 if kind == 'mixed' else 1, 5))
    skeleton = [[] for _ in range(count)]
    depths = [1]*count
    for i in range(1, count):
        choices = [j for j in range(i) if depths[j] < 3 and len(skeleton[j]) < 2]
        parent = int(rng.choice(choices))
        skeleton[parent].append(i)
        depths[i] = depths[parent]+1
    axes = rng.integers(0, 2, count).tolist()
    if kind == 'mixed' and len(set(axes)) == 1:
        axes[-1] = 1-axes[0]
    if small and kind != 'mixed':
        axes = [1]*count
    def build(i):
        kwargs = {}
        if kind == 'mixed' and small:
            kwargs = dict(n=int(rng.integers(8, 13)) if i == 0 else int(rng.integers(6, 8)),
                          errors=2 if i == 0 else 1 if i == 1 else int(rng.integers(2)))
        return node([build(j) for j in skeleton[i]], axis=axes[i],
                    group=int(rng.integers(2)) if not small else 1-axes[i], **kwargs)
    result = build(0)
    def has_error(t):
        return np.any(t['noisy'] != t['clean']) or any(has_error(c) for c in t['children'])
    if not has_error(result):
        result['noisy'][0] = 1-result['noisy'][0]
    return result


def _compact_layout(plan, rng, mixed):
    """Enumerate small axial compositions, never an output or topology catalog.

    Words, faults, contour tree and item order stay fixed. The at-most 4^4
    combinations only choose a band's direction and its region's group direction.
    """
    nodes = []
    def collect(t):
        nodes.append(t)
        for child in t['children']:
            collect(child)
    collect(plan)
    index = {id(t): k for k, t in enumerate(nodes)}
    choices = []
    for codes in product(range(4), repeat=len(nodes)):
        if mixed:
            axes = {codes[k] % 2 for k, t in enumerate(nodes) if np.any(t['noisy'] != t['clean'])}
            if axes != {0, 1}:
                continue
        def shape(t):
            code = codes[index[id(t)]]
            band = (2*len(t['noisy'])-1, 1) if code % 2 == 0 else (1, 2*len(t['noisy'])-1)
            parts = [band]+[shape(c) for c in t['children']]
            axis = code//2
            dims = [0, 0]
            dims[axis] = sum(p[axis] for p in parts)+len(parts)-1+4
            dims[1-axis] = max(p[1-axis] for p in parts)+4
            return tuple(dims)
        if max(shape(plan)) <= 28:
            choices.append(codes)
    if not choices:
        return False
    selected = choices[int(rng.integers(len(choices)))]
    for t, code in zip(nodes, selected):
        t['axis'], t['group'] = code % 2, code//2
    return True


def _draw_band(plan, rng, bg, signal, small):
    n = len(plan['noisy'])
    lengths = np.ones(n, dtype=int)
    widths = np.ones(n, dtype=int)
    gaps = [1]*(n-1)
    if not small:
        widths = rng.choice([1, 3], n, p=[.62, .38])
        if rng.random() < .35:
            lengths[int(rng.integers(n))] = 3
        if rng.random() < .3:
            gaps[int(rng.integers(n-1))] = 3
    band = np.full((int(max(widths)), int(sum(lengths)+sum(gaps))), bg, dtype=np.int8)
    cursor = 0
    for k, (length, width) in enumerate(zip(lengths, widths)):
        y = (band.shape[0]-width)//2
        mask = np.ones((int(width), int(length)), dtype=bool)
        if width == length == 3 and rng.random() < .65:
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            for index in rng.choice(4, int(rng.integers(1, 4)), replace=False):
                mask[corners[int(index)]] = False
        view = band[y:y+width, cursor:cursor+length]
        view[mask] = signal[int(plan['noisy'][k])]
        cursor += int(length)+(gaps[k] if k < n-1 else 0)
    return band.T.copy() if plan['axis'] == 0 else band


def _outline(shape, padding, rng, occupied):
    h, w = shape
    # Four independent orthogonal corner cutouts. Paths retain four-connectivity.
    top, bottom, left, right = padding
    bounds = [min(top, left)-1, min(top, right)-1,
              min(bottom, right)-1, min(bottom, left)-1]
    cuts = [int(rng.integers(k+1)) for k in bounds]
    if rng.random() < .15:
        cuts = [0]*4
    def trace(values):
        tl, tr, br, bl = values
        path = [(0, tl), (0, w-1-tr), (tr, w-1-tr), (tr, w-1),
                (h-1-br, w-1), (h-1-br, w-1-br), (h-1, w-1-br),
                (h-1, bl), (h-1-bl, bl), (h-1-bl, 0), (tl, 0),
                (tl, tl), (0, tl)]
        mask = np.zeros(shape, dtype=bool)
        for (y, x), (yy, xx) in zip(path, path[1:]):
            mask[min(y, yy):max(y, yy)+1, x] = True
            mask[yy, min(x, xx):max(x, xx)+1] = True
        return mask
    # A cut may approach a child's bounding box where that box is actually blank.
    # Test real pixels, preserving diagonal separation without forbidding all
    # stepped corners in compact scenes.
    expanded = np.pad(occupied, 1)
    near = np.zeros(shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            near |= expanded[dy:dy+h, dx:dx+w]
    for corner in range(4):
        while cuts[corner]:
            trial = [0]*4
            trial[corner] = cuts[corner]
            if not np.any(trace(trial) & near):
                break
            cuts[corner] -= 1
    return trace(cuts)


def _draw(plan, rng, palette, small=False):
    bg, ink, *signal = palette
    band = _draw_band(plan, rng, bg, signal, small)
    children = [_draw(child, rng, palette, small) for child in plan['children']]
    metric = [band if j == -1 else children[j][0] for j in plan['order']]
    gaps = [1 if small or rng.random() < .8 else 3 for _ in metric[:-1]]
    content = _compose(metric, plan['group'], bg, gaps)
    py, px = (2 if small or rng.random() < .83 else 3 for _ in range(2))
    padding = (py, py, px, px)
    inp = np.pad(content, ((py, py), (px, px)), constant_values=bg)
    inp[_outline(inp.shape, padding, rng, inp != bg)] = ink
    bad = np.flatnonzero(plan['noisy'] != plan['clean'])
    point_band = _points([signal[int(plan['clean'][k])] for k in bad], plan['axis'], bg)
    outputs = [point_band if j == -1 else children[j][1] for j in plan['order']]
    clean_content = _compose([p for p in outputs if p is not None], plan['group'], bg)
    out = None if clean_content is None else _frame(clean_content, bg, ink)
    return inp, out


def _make_pair(rng, palette, kind, rotation):
    # Hold the requested topology size fixed through retries, so packing rejection
    # cannot quietly replace richer scenes by mostly one-contour examples.
    count = int(rng.integers(2 if kind == 'mixed' else 1, 5))
    for attempt in range(256):
        small = attempt >= 32
        plan = _plan(rng, kind, small, count)
        if small and kind in ('mixed', 'random') and not _compact_layout(plan, rng, kind == 'mixed'):
            continue
        inp, out = _draw(plan, rng, palette, small)
        if out is None or max(*inp.shape, *out.shape) > 28:
            continue
        # Every side has a visible background halo. Extra placement is irrelevant.
        pad = []
        for d in inp.shape:
            budget = 30-d
            total = int(rng.integers(2, min(6, budget)+1))
            before = int(rng.integers(1, total))
            pad.append((before, total-before))
        inp = np.pad(inp, pad, constant_values=palette[0])
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
        if rng.random() < .5:
            inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        try:
            info = _parse(inp)
            inferred = _diagnostic(info)
            if not np.array_equal(inferred, out):
                continue
            if kind == 'mixed':
                faulty = [n for n in info['nodes'].values() if n['bad']]
                if len({n['axis'] for n in faulty}) < 2 or not any(len(n['bad']) >= 2 for n in faulty):
                    continue
        except ValueError:
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError(f"bounded scene construction exhausted: {kind}, {count} contours")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base = tuple(int(c) for c in rng.choice(10, 4, replace=False))
    stable = bool(rng.integers(3) == 0)
    rotation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            kind = ('pruning' if index == 0 else 'ancestry' if index == 1 else 'random') if split == 'train' else 'mixed'
            rot = (rotation+index) % 4 if split == 'train' and index < 2 else int(rng.integers(4))
            for _ in range(32):
                palette = base if stable else tuple(int(c) for c in rng.choice(10, 4, replace=False))
                pair = _make_pair(rng, palette, kind, rot)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    alive = {}
    depth = {}
    def visit(i, d):
        depth[i] = d
        node = info['nodes'][i]
        descendants = [visit(j, d+1) for j in node['order'] if j is not None]
        alive[i] = bool(node['bad']) or any(descendants)
        return alive[i]
    visit(info['root'], 1)
    return dict(alive=alive, depth=max(depth.values()), loops=len(alive),
                points=sum(len(n['bad']) for n in info['nodes'].values()),
                pruned=sum(not x for x in alive.values()),
                clean_ancestor=any(alive[i] and not n['bad'] for i, n in info['nodes'].items()),
                axes={n['axis'] for n in info['nodes'].values()},
                periods={n['period'] for n in info['nodes'].values()},
                double=any(len(n['bad']) == 2 for n in info['nodes'].values()))


def _check_output(info, output):
    a = _grid(output)
    assert np.array_equal(a, _diagnostic(info)), 'wrong diagnostic diagram'
    records = _records(a, info['bg'])
    loops, parents, children, root = _hierarchy(records)
    f = _features(info)
    assert len(loops) == sum(f['alive'].values()), 'wrong pruned frame count'
    points = [o for o in records if not o['loop']]
    assert len(points) == f['points'] and all(len(o['cells']) == 1 for o in points)
    assert np.array_equal(records[root]['lo'], [0, 0])
    assert np.array_equal(records[root]['hi'], np.array(a.shape)-1), 'extra crop margin'
    for i in loops:
        obj = records[i]
        h, w = obj['local'].shape
        assert obj['color'] == info['ink'] and len(obj['cells']) == 2*(h+w)-4
        assert obj['local'][0].all() and obj['local'][-1].all()
        assert obj['local'][:, 0].all() and obj['local'][:, -1].all()
        assert children[i], 'empty output frame'
        lo = np.min([records[j]['lo'] for j in children[i]], axis=0)
        hi = np.max([records[j]['hi'] for j in children[i]], axis=0)
        assert np.array_equal(lo, obj['lo']+2) and np.array_equal(hi, obj['hi']-2), 'wrong moat'
    return f


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _parse(inp)
            assert all(6 <= len(n['blobs']) <= 12 for n in info['nodes'].values())
            assert all(2 <= n['period'] <= 4 for n in info['nodes'].values())
            f = _check_output(info, out)
            assert 1 <= f['depth'] <= 3 and 1 <= f['loops'] <= 4
            # Removing periodic diagnosis keeps many correct blobs and healthy frames.
            assert not np.array_equal(_diagnostic(info, include_correct=True), out)
            if split == 'train':
                training.append(f)
            else:
                faulty = [n for n in info['nodes'].values() if n['bad']]
                assert {n['axis'] for n in faulty} == {0, 1}, 'test must recombine faulty band directions'
                assert f['double'] or any(len(n['bad']) == 3 for n in faulty)
    assert any(f['pruned'] for f in training), 'training needs a disappearing healthy branch'
    assert any(f['clean_ancestor'] for f in training), 'training needs an intact band above faults'
    assert any(f['depth'] == 3 for f in training), 'recursive retention must be demonstrated'
    assert any(f['double'] for f in training), 'training needs both directions of binary repair'
    assert set.union(*(f['axes'] for f in training)) == {0, 1}
    assert {2, 3}.issubset(set.union(*(f['periods'] for f in training)))


def _normalized(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.ravel())


def self_test():
    """Bounded 42-seed structural, semantic, negative, and metamorphic checks."""
    # Corrupt the first observed cycle, and require both substitution directions.
    word = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype=np.int8)
    repaired, period = _recover_word(word)
    assert period == 2 and np.array_equal(repaired, np.tile([0, 1], 4))
    assert np.flatnonzero(repaired != word).tolist() == [0, 5]
    try:
        _recover_word(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError('tied phase accepted')
    shapes, structures, signatures, periods, sizes = set(), set(), set(), set(), set()
    loop_counts, point_counts, band_counts, depths = set(), set(), set(), set()
    metrics = Counter()
    sample = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                f = _features(info)
                shapes.add(pair['input'].shape)
                structures.add(_normalized(pair['output']))
                periods.update(f['periods'])
                loop_counts.add(f['loops'])
                point_counts.add(f['points'])
                depths.add(f['depth'])
                band_counts.update(len(n['blobs']) for n in info['nodes'].values())
                sizes.update(len(o['cells']) for o in info['records'] if not o['loop'])
                metrics['pruning_pairs'] += bool(f['pruned'])
                metrics['clean_ancestor_pairs'] += f['clean_ancestor']
                metrics['mixed_pairs'] += len(f['axes']) == 2
                metrics['stepped_contours'] += sum(not (o['local'][0].all() and o['local'][-1].all()
                                                     and o['local'][:, 0].all() and o['local'][:, -1].all())
                                                  for o in info['records'] if o['loop'])
        pair = puzzle['test'][0]
        inp, answer = pair['input'], pair['output']
        rng = np.random.default_rng(84000+seed)
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[inp]), permutation[answer])
        assert np.array_equal(solve(np.rot90(inp)), np.rot90(answer))
        assert np.array_equal(solve(inp.T), answer.T)
        # Translation: add a blank row/column where the 30-cell bound permits it.
        padding = ((1 if inp.shape[0] < 30 else 0, 0), (0, 1 if inp.shape[1] < 30 else 0))
        shifted = np.pad(inp, padding, constant_values=int(inp[0, 0]))
        assert np.array_equal(solve(shifted), answer)
        if seed == 0:
            sample = puzzle
    assert len(signatures) == 42 and len(shapes) >= 40 and len(structures) >= 60
    assert periods == {2, 3, 4} and band_counts == set(range(6, 13))
    assert depths == {1, 2, 3} and len(loop_counts) >= 3 and len(point_counts) >= 5
    assert len(sizes) >= 4
    assert metrics['pruning_pairs'] >= 42 and metrics['clean_ancestor_pairs'] >= 42
    assert metrics['mixed_pairs'] >= 42 and metrics['stepped_contours'] >= 100
    # Independent metric realizations of one fixed logical scene have one answer.
    rng = np.random.default_rng(70123)
    def logical_node(n, errors, axis, children=()):
        noisy, clean = _sample_word(rng, n, errors, 2)
        return dict(children=list(children), axis=axis, group=1-axis,
                    noisy=noisy, clean=clean, order=[-1]+list(range(len(children))))
    plan = logical_node(8, 2, 1, [logical_node(6, 1, 0)])
    answers, drawings = set(), set()
    for _ in range(16):
        inp, expected = _draw(plan, rng, (0, 5, 2, 8), False)
        if max(inp.shape) <= 28:
            inp = np.pad(inp, 1)
            assert np.array_equal(solve(inp), expected)
            answers.add(_key(expected))
            drawings.add(_key(inp))
    assert len(answers) == 1 and len(drawings) >= 2
    # Three faults and a longer period are supported even when uncommon in draws.
    noisy, clean = _sample_word(rng, 12, 3, 3)
    three = dict(children=[], axis=1, group=0, noisy=noisy, clean=clean, order=[-1])
    inp, expected = _draw(three, rng, (0, 5, 2, 8), True)
    assert np.array_equal(solve(np.pad(inp, 1)), expected)
    assert len([o for o in _records(expected, 0) if not o['loop']]) == 3
    # A topology-free list of identical repair colors cannot encode these answers.
    trees = {}
    for pair in sample['train']:
        info = _parse(pair['input'])
        colors = tuple(sorted(int(n['clean'][k]) for n in info['nodes'].values() for k in n['bad']))
        trees.setdefault(len(colors), set()).add(_normalized(pair['output']))
    assert any(len(v) > 1 for v in trees.values())
    assert tuple(_key(p['input']) for p in generate_puzzle(0)['train']) == tuple(_key(p['input']) for p in sample['train'])
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1234+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Reject correct-shape color damage, missing frames, and excess crop margins.
    for mode in range(3):
        broken = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in sample[s]] for s in sample}
        pair = broken['test'][0]
        info = _parse(pair['input'])
        if mode == 0:
            o = next(o for o in _records(pair['output'], info['bg']) if not o['loop'])
            pair['output'][tuple(o['cells'][0])] = info['bg']
        elif mode == 1:
            pair['output'][0, 1] = info['bg']
        else:
            pair['output'] = np.pad(pair['output'], 1, constant_values=info['bg'])
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('damaged diagnostic accepted')
    return dict(seeds=42, input_shapes=len(shapes), normalized_outputs=len(structures),
                periods=sorted(periods), band_counts=sorted(band_counts),
                input_contour_counts=sorted(loop_counts), repair_point_counts=sorted(point_counts),
                blob_areas=sorted(sizes), **dict(metrics))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
