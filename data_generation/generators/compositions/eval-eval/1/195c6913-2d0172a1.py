"""Nested contour clocks: topology determines both geometry and cyclic phase.

No files, parent code, random state, or training examples are used by solve().
All experiments in self_test are bounded and remain in memory.
"""
from collections import Counter
import hashlib
import json

import numpy as np


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    h, w = mask.shape
    result = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        todo = [(int(y), int(x))]
        seen[y, x] = True
        cells = []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        result.append(cells)
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, bool)
    outside[0, 0] = True
    todo = [(0, 0)]
    h, w = padded.shape
    while todo:
        r, c = todo.pop()
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < h and 0 <= cc < w and not padded[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                todo.append((rr, cc))
    return (~padded & ~outside)[1:-1, 1:-1]


def _grid(grid):
    a = np.asarray(grid)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError('grid dimensions must be 1..30')
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError('grid must contain integer colors 0..9')
    return a


def _decode(grid):
    a = _grid(grid)
    h, w = a.shape
    assert h >= 10 and w >= 8
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1]))
    assert np.all(a[3:5] == bg)
    word, header = [], np.full((3, w), bg, dtype=a.dtype)
    cols = np.nonzero(a[1] != bg)[0].tolist()
    while cols:
        x = cols.pop(0)
        assert cols and cols.pop(0) == x+1
        color = int(a[1, x])
        assert np.all(a[1:3, x:x+2] == color)
        assert not cols or cols[0] >= x+3
        header[1:3, x:x+2] = color
        word.append(color)
    assert np.array_equal(header, a[:3])
    assert 2 <= len(word) <= 6 and len(set(word)) >= 2
    drawing = set(map(int, np.unique(a[5:]))) - {bg}
    assert len(drawing) == 1
    fg = drawing.pop()
    assert fg not in word
    objects = []
    for cells in _components(a == fg):
        coords = np.array(cells, dtype=int)
        lo, hi = coords.min(0), coords.max(0)
        mask = np.zeros(a.shape, bool)
        mask[coords[:, 0], coords[:, 1]] = True
        hole = _holes(mask)
        if hole.any():
            assert len(_components(hole)) == 1, 'one hole per contour'
            kind = 'C'
        else:
            assert len(cells) >= np.prod(hi-lo+1)*.5, 'compact filled blob required'
            kind = 'B'
        objects.append(dict(kind=kind, lo=lo, hi=hi, cells=coords, hole=hole, area=len(cells)))
    assert objects
    # Distinct objects may not touch even diagonally.
    labels = np.zeros(a.shape, int)
    for i, obj in enumerate(objects):
        labels[tuple(obj['cells'].T)] = i+1
    for left, right in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                        (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        assert not np.any((left > 0) & (right > 0) & (left != right))
    parents = []
    for i, child in enumerate(objects):
        candidates = []
        for j, obj in enumerate(objects):
            if j == i or obj['kind'] != 'C':
                continue
            inside = obj['hole'][tuple(child['cells'].T)]
            assert inside.all() or not inside.any(), 'crossing contours'
            if inside.all():
                candidates.append(j)
        parents.append(min(candidates, key=lambda j: int(objects[j]['hole'].sum())) if candidates else None)
    children = {i: [] for i in range(len(objects))}
    roots = []
    for i, parent in enumerate(parents):
        (roots if parent is None else children[parent]).append(i)
    assert all(objects[i]['kind'] == 'C' for i in roots), 'no exterior blobs'
    for i, obj in enumerate(objects):
        assert bool(children[i]) == (obj['kind'] == 'C'), 'contours must be nonempty'
    return dict(background=bg, foreground=fg, word=tuple(word), objects=objects,
                parents=parents, children=children, roots=roots)


def _tracks(entries, sizes, gaps):
    """Entries are (row-key, column-key, id), with top-left slot alignment."""
    ys = sorted({r for r, c, i in entries})
    xs = sorted({c for r, c, i in entries})
    assert len({(r, c) for r, c, i in entries}) == len(entries)
    row_heights = [max(sizes[i][0] for r, c, i in entries if r == y) for y in ys]
    col_widths = [max(sizes[i][1] for r, c, i in entries if c == x) for x in xs]
    starts = []
    for lengths, key, axis in ((row_heights, ys, 0), (col_widths, xs, 1)):
        cursor, offsets = 0, {}
        for j, (k, length) in enumerate(zip(key, lengths)):
            offsets[k] = cursor
            cursor += length
            if j+1 < len(key):
                cursor += gaps(axis, j)
        starts.append((cursor, offsets))
    positions = {i: (starts[0][1][r], starts[1][1][c]) for r, c, i in entries}
    return (starts[0][0], starts[1][0]), positions


def _normalized(info):
    objs = info['objects']
    sizes, relative = {}, {}

    def group(ids):
        for i in ids:
            node(i)
        entries = [(int(objs[i]['lo'][0]), int(objs[i]['lo'][1]), i) for i in ids]
        # Visible row/column tracks must have at least one blank separating line.
        for axis in (0, 1):
            keys = sorted({int(objs[i]['lo'][axis]) for i in ids})
            for a, b in zip(keys, keys[1:]):
                assert max(int(objs[i]['hi'][axis]) for i in ids if objs[i]['lo'][axis] == a) < b-1
        return _tracks(entries, sizes, lambda axis, j: 1)

    def node(i):
        if objs[i]['kind'] == 'B':
            sizes[i], relative[i] = (1, 1), {}
        else:
            (h, w), positions = group(info['children'][i])
            sizes[i] = (h+4, w+4)
            relative[i] = {j: (y+2, x+2) for j, (y, x) in positions.items()}

    shape, root_positions = group(info['roots'])
    boxes = {}

    def locate(i, y, x):
        boxes[i] = (y, x, *sizes[i])
        for j, (dy, dx) in relative[i].items():
            locate(j, y+dy, x+dx)

    for i, (y, x) in root_positions.items():
        locate(i, y, x)
    return shape, boxes


def _perimeter(h, w):
    return ([(0, x) for x in range(w)] + [(y, w-1) for y in range(1, h)]
            + [(h-1, x) for x in range(w-2, -1, -1)]
            + [(y, 0) for y in range(h-2, 0, -1)])


def _paint(info, shape, boxes, alternative=None):
    word = info['word']
    if alternative == 'deduplicate':
        word = tuple(dict.fromkeys(word))
    out = np.full(shape, info['background'], dtype=np.int8)
    objects = info['objects']
    global_phase = [0]

    def visit(i, phase):
        y, x, h, w = boxes[i]
        if alternative == 'global':
            phase = global_phase[0]
        if objects[i]['kind'] == 'B':
            out[y, x] = word[phase % len(word)]
            global_phase[0] += 1
            return
        path = _perimeter(h, w)
        for k, (dy, dx) in enumerate(path):
            tick = k
            if alternative == 'bend_reset':
                tick = k if k < w else k-w if k < w+h-1 else k-(w+h-1) if k < 2*w+h-2 else k-(2*w+h-2)
            out[y+dy, x+dx] = word[(phase+tick) % len(word)]
        global_phase[0] += len(path)
        for j in sorted(info['children'][i], key=lambda k: boxes[k][:2]):
            row = boxes[j][0]-y
            inherited = phase+len(path)-row
            if alternative == 'reset_children':
                inherited = 0
            elif alternative == 'top_port':
                inherited = phase+boxes[j][1]-x
            elif alternative == 'depth':
                inherited = phase+1
            elif alternative == 'metric':
                raw = objects[i]['hi']-objects[i]['lo']+1
                inherited = phase+2*int(raw.sum())-4-int(objects[j]['lo'][0]-objects[i]['lo'][0])
            visit(j, inherited)

    root_phase = 0
    for i in sorted(info['roots'], key=lambda k: boxes[k][:2]):
        visit(i, root_phase if alternative == 'continue_roots' else 0)
        root_phase += len(_perimeter(*boxes[i][2:]))
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode visible contours, normalize, and transmit clockwise clock phases."""
    info = _decode(input_grid)
    shape, boxes = _normalized(info)
    return _paint(info, shape, boxes)


def _blob():
    return ('B', ())


def _contour(children, slots):
    return ('C', tuple((r, c, child) for child, (r, c) in zip(children, slots)))


def _line(rng, children, axis=None):
    if axis is None:
        axis = int(rng.integers(2))
    rng.shuffle(children)
    return _contour(children, [(j, 0) if axis == 0 else (0, j) for j in range(len(children))])


def _small(rng):
    return _line(rng, [_blob() for _ in range(int(rng.integers(1, 4)))])


def _scene(rng, mode):
    if mode in ('vertical_witness', 'horizontal_witness'):
        axis = 0 if mode == 'vertical_witness' else 1
        children = [_small(rng), _blob()]
        if rng.random() < .4:
            children.append(_blob())
        outer = _line(rng, children, axis)
        roots = [outer, _small(rng)] if axis == 0 else [outer]
    elif mode == 'deep':
        inner = _small(rng)
        middle = _line(rng, [inner] + ([_blob()] if rng.random() < .5 else []))
        roots = [_line(rng, [middle] + ([_blob()] if rng.random() < .5 else []))]
    elif mode == 'matrix':
        slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
        if rng.random() < .5:
            slots.pop(int(rng.integers(4)))
        kids = [_blob() for _ in slots]
        kids[int(rng.integers(len(kids)))] = _small(rng)
        roots = [_contour(kids, slots)]
    else:
        roots = [_line(rng, [(_small(rng) if rng.random() < .35 else _blob())
                             for _ in range(int(rng.integers(1, 4)))])]
        if rng.random() < .3:
            roots.append(_small(rng))
    rng.shuffle(roots)
    axis = int(rng.integers(2))
    return tuple((j, 0, root) if axis == 0 else (0, j, root) for j, root in enumerate(roots))


def _outline(rng, h, w, padding):
    """Safe corner cuts and side bays; the interior content is never carved."""
    filled = np.ones((h, w), bool)
    top, bottom, left, right = padding
    for corner, cap in enumerate((min(top, left)-2, min(top, right)-2,
                                  min(bottom, left)-2, min(bottom, right)-2)):
        if cap and rng.random() < .6:
            k = int(rng.integers(1, cap+1))
            ys = slice(0, k) if corner < 2 else slice(h-k, h)
            xs = slice(0, k) if corner % 2 == 0 else slice(w-k, w)
            filled[ys, xs] = False
    if rng.random() < .7:
        side = int(rng.integers(4))
        cap = padding[side]-2
        if cap:
            if side < 2 and w >= 9:
                x = int(rng.integers(3, w-4))
                if side == 0:
                    filled[:cap, x:x+2] = False
                else:
                    filled[h-cap:, x:x+2] = False
            elif side >= 2 and h >= 9:
                y = int(rng.integers(3, h-4))
                if side == 2:
                    filled[y:y+2, :cap] = False
                else:
                    filled[y:y+2, w-cap:] = False
    # The 8-neighbor inner boundary includes stair-corner connectors.
    p = np.pad(filled, 1)
    interior = np.ones_like(filled)
    for dy in range(3):
        for dx in range(3):
            interior &= p[dy:dy+h, dx:dx+w]
    return filled & ~interior


def _draw_tree(entries, rng=None, compact=False):
    """Metric rendering, or canonical geometry when rng is None."""
    serial, records = [0], {}

    def group(parts, parent):
        sizes, masks, indexed = {}, {}, []
        for r, c, node in parts:
            i, mask = draw(node, parent)
            masks[i], sizes[i] = mask, mask.shape
            indexed.append((r, c, i))
        shape, positions = _tracks(indexed, sizes, lambda axis, j: 1 if rng is None else int(rng.integers(1, 4)))
        canvas = np.zeros(shape, bool)
        for i, (y, x) in positions.items():
            h, w = sizes[i]
            canvas[y:y+h, x:x+w] |= masks[i]
            # Descendants are already in their own tile coordinates.
            translate(i, y, x)
        return canvas, [i for r, c, i in indexed]

    def translate(i, dy, dx):
        rec = records[i]
        y, x, h, w = rec['box']
        rec['box'] = (y+dy, x+dx, h, w)
        for j in rec['children']:
            translate(j, dy, dx)

    def draw(node, parent):
        i = serial[0]
        serial[0] += 1
        records[i] = dict(kind=node[0], parent=parent, children=[], box=None)
        if node[0] == 'B':
            h, w = (1, 1) if rng is None else tuple(map(int, rng.integers(1, 3 if compact else 5, 2)))
            mask = np.ones((h, w), bool)
            if rng is not None and min(h, w) >= 2 and rng.random() < .5:
                mask[int(rng.choice([0, h-1])), int(rng.choice([0, w-1]))] = False
        else:
            content, kids = group(node[1], i)
            padding = [2]*4 if rng is None else list(map(int, rng.integers(2, 4 if compact else 5, 4)))
            t, b, l, r = padding
            h, w = content.shape[0]+t+b, content.shape[1]+l+r
            mask = np.zeros((h, w), bool) if rng is None else _outline(rng, h, w, padding)
            if rng is None:
                mask[0] = mask[-1] = True
                mask[:, 0] = mask[:, -1] = True
            mask[t:t+content.shape[0], l:l+content.shape[1]] |= content
            records[i]['children'] = kids
            for j in kids:
                translate(j, t, l)
        records[i]['box'] = (0, 0, h, w)
        return i, mask

    mask, roots = group(entries, None)
    return mask, records, roots


def _tree_answer(records, roots, shape, bg, word):
    """Construction-side recursive clock, without decoding input pixels."""
    result = np.full(shape, bg, np.int8)

    def ink(i, phase):
        y, x, h, w = records[i]['box']
        if records[i]['kind'] == 'B':
            result[y, x] = word[phase % len(word)]
            return
        # Four disjoint legs, corners counted exactly once.
        tick = phase
        for cells in ([(y, xx) for xx in range(x, x+w)],
                      [(yy, x+w-1) for yy in range(y+1, y+h)],
                      [(y+h-1, xx) for xx in range(x+w-2, x-1, -1)],
                      [(yy, x) for yy in range(y+h-2, y, -1)]):
            for p in cells:
                result[p] = word[tick % len(word)]
                tick += 1
        for j in records[i]['children']:
            child_top = records[j]['box'][0]
            ink(j, phase+2*(h+w)-4-(child_top-y))

    for i in roots:
        ink(i, 0)
    return result


_ALTERNATIVES = ('deduplicate', 'bend_reset', 'reset_children', 'top_port',
                 'depth', 'metric', 'global', 'continue_roots')


def _evidence(info, shape, boxes, output):
    return {name for name in _ALTERNATIVES
            if not np.array_equal(output, _paint(info, shape, boxes, name))}


def _make_pair(rng, mode):
    for attempt in range(120):
        scene = _scene(rng, mode)
        metric, _, _ = _draw_tree(scene, rng, compact=attempt >= 8)
        canonical, records, roots = _draw_tree(scene)
        mh, mw = metric.shape
        if mh > 24 or mw > 28 or max(canonical.shape) > 30:
            continue
        n = int(rng.choice([3, 5])) if mode == 'vertical_witness' else int(rng.integers(2, 7))
        distinct = int(rng.integers(2, n if mode == 'vertical_witness' else n+1))
        symbols = list(range(distinct)) + list(map(int, rng.integers(0, distinct, n-distinct)))
        rng.shuffle(symbols)
        palette = list(map(int, rng.permutation(10)))
        bg, fg = palette[:2]
        word = tuple(palette[2+k] for k in symbols)
        gaps = list(map(int, rng.integers(1, 3, n-1)))
        legend_width = 2*n+sum(gaps)
        width_low = max(mw+2, legend_width+2)
        height_low = mh+6
        if width_low > 30:
            continue
        height = int(rng.integers(height_low, min(30, height_low+3)+1))
        width = int(rng.integers(width_low, min(30, width_low+3)+1))
        a = np.full((height, width), bg, np.int8)
        y, x = int(rng.integers(5, height-mh)), int(rng.integers(1, width-mw))
        a[y:y+mh, x:x+mw][metric] = fg
        x = int(rng.integers(1, width-legend_width))
        for j, color in enumerate(word):
            a[1:3, x:x+2] = color
            x += 2 + (gaps[j] if j < n-1 else 0)
        expected = _tree_answer(records, roots, canonical.shape, bg, word)
        try:
            info = _decode(a)
            shape, boxes = _normalized(info)
            actual = _paint(info, shape, boxes)
            assert np.array_equal(actual, expected), 'metric rendering changed the intended topology/layout'
            evidence = _evidence(info, shape, boxes, actual)
            if mode == 'vertical_witness':
                assert set(_ALTERNATIVES) <= evidence
            if mode == 'horizontal_witness':
                assert {'bend_reset', 'reset_children', 'top_port', 'metric'} <= evidence
        except AssertionError:
            continue
        return dict(input=a, output=expected)
    raise RuntimeError('bounded scene construction exhausted')


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer in 2..10')
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError('num_test must be in 1..4 and smaller than num_train')
    rng = np.random.default_rng(seed)
    result, seen = {'train': [], 'test': []}, set()
    train_modes = ['vertical_witness', 'horizontal_witness'] + ['ordinary']*(num_train-2)
    rng.shuffle(train_modes)
    test_mode = str(rng.choice(['deep', 'matrix']))
    for split, modes in (('train', train_modes), ('test', [test_mode]*num_test)):
        for mode in modes:
            for retry in range(12):
                pair = _make_pair(rng, mode)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
            seen.add(_key(pair['input']))
            result[split].append(pair)
    return result


def _features(info, boxes):
    depths = []
    axes, matrix = set(), False
    signature = []
    for i, obj in enumerate(info['objects']):
        depth, p = 0, info['parents'][i]
        while p is not None:
            depth += 1
            p = info['parents'][p]
        if obj['kind'] == 'C':
            depths.append(depth+1)
        ids = info['children'][i]
        rows = {boxes[j][0] for j in ids}
        cols = {boxes[j][1] for j in ids}
        if len(rows) > 1:
            axes.add(0)
        if len(cols) > 1:
            axes.add(1)
        matrix |= len(rows) > 1 and len(cols) > 1
        signature.append((obj['kind'], depth, boxes[i]))
    return dict(depth=max(depths), axes=axes, matrix=matrix,
                contours=len(depths), blobs=len(info['objects'])-len(depths),
                signature=tuple(signature))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence, train_axes, train_features, test_features = set(), set(), set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = (_grid(pair[k]) for k in ('input', 'output'))
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _decode(a)
            shape, boxes = _normalized(info)
            assert np.array_equal(out, _paint(info, shape, boxes))
            # Independently check support, crop, one-cell leaves, and moats.
            support = np.zeros(shape, bool)
            for i, (y, x, h, w) in boxes.items():
                if info['objects'][i]['kind'] == 'B':
                    assert (h, w) == (1, 1)
                    support[y, x] = True
                else:
                    for dy, dx in _perimeter(h, w):
                        support[y+dy, x+dx] = True
                    kids = [boxes[j] for j in info['children'][i]]
                    assert min(k[0] for k in kids) == y+2
                    assert min(k[1] for k in kids) == x+2
                    assert max(k[0]+k[2] for k in kids) == y+h-2
                    assert max(k[1]+k[3] for k in kids) == x+w-2
                    if info['parents'][i] is None:
                        assert out[y, x] == info['word'][0]
                    for j in info['children'][i]:
                        yy, xx, hh, ww = boxes[j]
                        assert out[yy, xx] == out[yy, x], 'left-wall phase transmission'
            assert np.array_equal(out != info['background'], support)
            assert all(edge.any() for edge in (support[0], support[-1], support[:, 0], support[:, -1]))
            f = _features(info, boxes)
            if split == 'train':
                evidence |= _evidence(info, shape, boxes, out)
                train_axes |= f['axes']
                train_features.append(f)
            else:
                test_features.append(f)
    assert set(_ALTERNATIVES) <= evidence, set(_ALTERNATIVES)-evidence
    assert train_axes == {0, 1}
    assert max(f['depth'] for f in train_features) == 2
    assert all(not f['matrix'] for f in train_features)
    assert all(f['depth'] == 3 or f['matrix'] for f in test_features)


def self_test():
    """42 seeds, legal counts, semantic mutations, and geometric metamorphisms."""
    # A literal rule regression pins direction, corner counting, and the port.
    fixture = np.zeros((11, 11), np.int8)
    for x, color in ((1, 2), (4, 3), (7, 4)):
        fixture[1:3, x:x+2] = color
    fixture[5, 1:6] = fixture[9, 1:6] = 1
    fixture[5:10, 1] = fixture[5:10, 5] = 1
    fixture[7, 3] = 1
    expected = np.array([[int(c) for c in row] for row in
                         ('23423', '20004', '40402', '30003', '24324')], np.int8)
    assert np.array_equal(solve(fixture), expected)
    statistics = Counter()
    shapes, signatures, words, blob_shapes, palettes = set(), set(), set(), set(), set()
    hashes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, out = pair['input'], pair['output']
                digest.update(a.tobytes())
                info = _decode(a)
                shape, boxes = _normalized(info)
                f = _features(info, boxes)
                shapes.add(a.shape)
                signatures.add(f['signature'])
                mapping = {}
                words.add(tuple(mapping.setdefault(c, len(mapping)) for c in info['word']))
                palettes.add((info['background'], info['foreground']))
                statistics['pairs'] += 1
                statistics['depth_'+str(f['depth'])] += 1
                statistics['matrix'] += f['matrix']
                statistics['multiple_roots'] += len(info['roots']) > 1
                statistics['contours'] += f['contours']
                for obj in info['objects']:
                    dims = tuple(map(int, obj['hi']-obj['lo']+1))
                    if obj['kind'] == 'B':
                        blob_shapes.add(dims)
                    else:
                        # Missing bbox-edge pixels, not area, detect nonrectangles.
                        coords = set(map(tuple, obj['cells'].tolist()))
                        y, x = map(int, obj['lo'])
                        statistics['nonrectangular'] += any((y+dy, x+dx) not in coords for dy, dx in _perimeter(*dims))
        hashes.add(digest.hexdigest())
        # Recolor every role, preserving the visible program and semantics.
        pair = puzzle['test'][0]
        perm = np.random.default_rng(1000+seed).permutation(10)
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
        # Add diagram-only blank rows/columns inside the available canvas.
        a = pair['input']
        bg = int(a[0, 0])
        if a.shape[0] < 30:
            shifted = np.insert(a, 5, bg, axis=0)
            assert np.array_equal(solve(shifted), pair['output'])
            # Duplicate an occupied diagram row: change raw loop heights,
            # wall lengths, and often blob size without changing topology.
            occupied = np.nonzero(np.any(a[5:] != bg, axis=1))[0]+5
            row = int(occupied[len(occupied)//2])
            stretched = np.insert(a, row, a[row], axis=0)
            assert np.array_equal(solve(stretched), pair['output'])
        # Mutating a clock tick must be rejected by full-puzzle validation.
        if seed == 0:
            copied = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            out = copied['test'][0]['output']
            out[0, 0] = (int(out[0, 0])+1) % 10
            try:
                validate_generated_puzzle(copied)
            except AssertionError:
                pass
            else:
                raise AssertionError('mutated clock accepted')
    assert len(hashes) == 42 and len(shapes) >= 30
    assert len(signatures) >= 70 and len(words) >= 20
    assert len(blob_shapes) >= 10 and len(palettes) >= 50
    assert statistics['depth_3'] >= 8 and statistics['matrix'] >= 8
    assert statistics['nonrectangular'] >= 30
    # All 30 legal explicit count combinations, never retaining the suite.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(7000+10*nt+ne, nt, ne)
            validate_generated_puzzle(p)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(683082393, 6, 3), generate_puzzle(683082393, 6, 3)
    assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    return dict(statistics, distinct_input_shapes=len(shapes), structural_signatures=len(signatures),
                word_patterns=len(words), blob_shapes=len(blob_shapes), palettes=len(palettes),
                distinct_puzzles=len(hashes))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
