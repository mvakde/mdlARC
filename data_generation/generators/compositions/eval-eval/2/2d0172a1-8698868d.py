"""Recursive socket stencils: containment is a program, cavities are its arity.

Only standard-library facilities and NumPy are used. The solver reads pixels;
generation separately renders an ordered tree and its elastic stencil expansion.
"""
from __future__ import annotations

import copy
import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask)
    result = []
    steps = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _FOUR
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack = [(int(r), int(c))]
        seen[r, c] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in steps:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(np.array(cells, dtype=int))
    return result


def _holes(mask):
    """Enclosed complement of this component, including pixels of its children."""
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    outside = np.zeros_like(padded)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        y, x = stack.pop()
        for dy, dx in _FOUR:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not padded[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True
                stack.append((yy, xx))
    return (~padded & ~outside)[1:-1, 1:-1]


def _box(cells):
    low, high = cells.min(axis=0), cells.max(axis=0) + 1
    return tuple(map(int, (*low, *high)))  # y0, x0, y1, x1


def _record(cells, shape, color):
    mask = np.zeros(shape, dtype=bool)
    mask[cells[:, 0], cells[:, 1]] = True
    box = _box(cells)
    y0, x0, y1, x1 = box
    hole = np.zeros(shape, dtype=bool)
    hole[y0:y1, x0:x1] = _holes(mask[y0:y1, x0:x1])
    return dict(cells=cells, mask=mask, hole=hole, box=box, color=color)


def _compress(mask):
    """Delete consecutive duplicate rows and columns of a binary stencil."""
    rows = np.r_[True, np.any(mask[1:] != mask[:-1], axis=1)]
    result = mask[rows]
    cols = np.r_[True, np.any(result[:, 1:] != result[:, :-1], axis=0)]
    return result[:, cols].copy()


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _reading_order(ids, records):
    """Overlapping vertical projections define rows; read rows top to bottom."""
    bands = []
    bottom = -1
    for i in sorted(ids, key=lambda j: (records[j]['box'][0], records[j]['box'][1])):
        y0, x0, y1, x1 = records[i]['box']
        if y0 >= bottom:
            bands.append([])
        bands[-1].append(i)
        bottom = max(bottom, y1)
    result = []
    for band in bands:
        band.sort(key=lambda j: records[j]['box'][1])
        for a, b in zip(band, band[1:]):
            assert records[a]['box'][3] < records[b]['box'][1], 'ambiguous row of children'
        result.extend(band)
    return result


def _separate_boxes(boxes):
    for i, (a, b, c, d) in enumerate(boxes):
        for e, f, g, h in boxes[i + 1:]:
            assert c < e or g < a or d < f or h < b, 'objects need a blank gap'


def _tree_from_records(records, ink=None):
    """Recover actual containment, never bounding-box containment."""
    parents = []
    children = [[] for _ in records]
    for i, obj in enumerate(records):
        candidates = []
        for j, parent in enumerate(records):
            if i == j or not parent['hole'].any():
                continue
            inside = parent['hole'][obj['cells'][:, 0], obj['cells'][:, 1]]
            assert inside.all() or not inside.any(), 'crossing objects'
            if inside.all():
                candidates.append(j)
        parent = min(candidates, key=lambda j: int(records[j]['hole'].sum())) if candidates else None
        parents.append(parent)
        if parent is not None:
            children[parent].append(i)
    roots = [i for i, p in enumerate(parents) if p is None]
    assert len(roots) == 1 and records[roots[0]]['hole'].any(), 'one rooted enclosure is required'
    for i, obj in enumerate(records):
        if obj['hole'].any():
            assert children[i], 'empty enclosure'
            assert len(_components(obj['hole'])) >= 1
        else:
            assert not children[i]
            if ink is not None:
                assert obj['color'] == ink and len(obj['cells']) == 1, 'output leaves must be ink points'
    def build(i):
        if not children[i]:
            return None
        ordered = _reading_order(children[i], records)
        return tuple(build(j) for j in ordered)
    return build(roots[0]), children, parents


def _decode(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= d <= 30 for d in grid.shape) and np.all((0 <= grid) & (grid <= 9))
    edge = _perimeter(grid)
    assert np.all(edge == edge[0]), 'uniform exterior border required'
    bg = int(edge[0])
    by_color = {}
    for color in map(int, np.unique(grid)):
        if color != bg:
            by_color[color] = [_record(cells, grid.shape, color) for cells in _components(grid == color)]
    ink_candidates = []
    for color, objects in by_color.items():
        if any(i != j and parent['hole'][child['cells'][:, 0], child['cells'][:, 1]].all()
               for i, parent in enumerate(objects) for j, child in enumerate(objects)):
            ink_candidates.append(color)
    assert len(ink_candidates) == 1, 'the nested monochrome sketch must be unique'
    ink = ink_candidates[0]
    objects = by_color[ink]
    for obj in objects:
        if obj['hole'].any():
            assert len(_components(obj['hole'])) == 1, 'a sketch contour has one enclosed region'
        else:
            y0, x0, y1, x1 = obj['box']
            assert len(obj['cells']) >= .35 * (y1-y0) * (x1-x0), 'leaves must be compact'
    labels = np.zeros(grid.shape, dtype=int)
    for i, obj in enumerate(objects):
        labels[obj['mask']] = i + 1
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        assert not np.any((a > 0) & (b > 0) & (a != b)), 'touching sketch components'
    tree, children, parents = _tree_from_records(objects)
    assert all(len(c) <= 4 for c in children)
    all_cells = np.concatenate([obj['cells'] for obj in objects])
    sketch_box = _box(all_cells)
    stencils, panel_boxes, areas, cavity_sizes = {}, [], {}, {}
    connectivity = False
    for color, panels in by_color.items():
        if color == ink:
            continue
        for obj in panels:
            y0, x0, y1, x1 = obj['box']
            raw = grid[y0:y1, x0:x1]
            assert set(map(int, np.unique(raw))) <= {bg, color}
            mask = raw == color
            assert np.all(_perimeter(mask)) and len(_components(mask)) == 1
            cavities = _components(~mask)
            k = len(cavities)
            assert 1 <= k <= 4 and k not in stencils, 'unique nonzero cavity counts'
            for cavity in cavities:
                a, b, c, d = _box(cavity)
                assert len(cavity) == (c-a) * (d-b), 'sockets must be rectangles'
            canonical = _compress(mask)
            assert int((~canonical).sum()) == k and len(_components(~canonical)) == k
            assert len(_components(canonical)) == 1
            stencils[k] = (color, canonical)
            panel_boxes.append(obj['box'])
            areas[k] = int((~mask).sum())
            cavity_sizes[k] = tuple(sorted(map(len, cavities)))
            connectivity |= len(_components(~mask, True)) < k
    assert 2 <= len(stencils) <= 4
    _separate_boxes([sketch_box] + panel_boxes)
    used = set(_arities(tree))
    assert used <= set(stencils), 'a stencil for every used arity is required'
    return dict(background=bg, ink=ink, tree=tree, objects=objects, children=children,
                parents=parents, stencils=stencils, areas=areas, cavity_sizes=cavity_sizes,
                connectivity=bool(connectivity), sketch_box=sketch_box, panel_boxes=panel_boxes)


def _arities(tree):
    if tree is None:
        return []
    return [len(tree)] + [k for child in tree for k in _arities(child)]


def _depth(tree):
    return 0 if tree is None else 1 + max(map(_depth, tree))


def _leaf_count(tree):
    return 1 if tree is None else sum(map(_leaf_count, tree))


def _assemble(tree, stencils, background, ink):
    if tree is None:
        return np.array([[ink]], dtype=np.int8)
    color, mask = stencils[len(tree)]
    sockets = list(zip(*np.nonzero(~mask)))  # canonical row-major socket order
    parts = [_assemble(child, stencils, background, ink) for child in tree]
    row_sizes, col_sizes = np.ones(mask.shape[0], dtype=int), np.ones(mask.shape[1], dtype=int)
    for (r, c), part in zip(sockets, parts):
        row_sizes[r] = max(row_sizes[r], part.shape[0] + 2)
        col_sizes[c] = max(col_sizes[c], part.shape[1] + 2)
    expanded = np.repeat(np.repeat(mask, row_sizes, axis=0), col_sizes, axis=1)
    out = np.where(expanded, color, background).astype(np.int8)
    ys, xs = np.r_[0, np.cumsum(row_sizes)], np.r_[0, np.cumsum(col_sizes)]
    for (r, c), part in zip(sockets, parts):
        y = int(ys[r] + (row_sizes[r] - part.shape[0]) // 2)
        x = int(xs[c] + (col_sizes[c] - part.shape[1]) // 2)
        out[y:y+part.shape[0], x:x+part.shape[1]] = part
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from visible contours, cavity counts, and socket geometry."""
    info = _decode(input_grid)
    return _assemble(info['tree'], info['stencils'], info['background'], info['ink'])


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _stencil(rng, k, diagonal=False, shared=False):
    if diagonal:
        # Two corner-touching cavities distinguish four- from eight-connectivity.
        mask = np.ones((4, 4), dtype=bool)
        mask[1, 1] = mask[2, 2] = False
        return np.rot90(mask, int(rng.integers(4))).copy()
    for _ in range(100):
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        if (h - 2) * (w - 2) < k:
            continue
        mask = np.ones((h, w), dtype=bool)
        for index in rng.permutation((h - 2) * (w - 2)):
            r, c = divmod(int(index), w - 2)
            r, c = r + 1, c + 1
            if any(not mask[r+dy, c+dx] for dy, dx in _FOUR):
                continue
            trial = mask.copy()
            trial[r, c] = False
            if len(_components(trial)) != 1:
                continue
            mask = trial
            if int((~mask).sum()) == k:
                break
        if int((~mask).sum()) != k:
            continue
        mask = _compress(mask)
        sockets = np.argwhere(~mask)
        if shared and not any(a != b and (sockets[a, 0] == sockets[b, 0] or sockets[a, 1] == sockets[b, 1])
                              for a in range(k) for b in range(k)):
            continue
        return mask
    # A strip is always feasible. It is a constructive fallback, not an answer bank.
    mask = np.ones((3, 2*k+1), dtype=bool)
    mask[1, 1::2] = False
    return mask if rng.random() < .5 else mask.T.copy()


def _stretch_stencil(rng, mask, large_hole=False):
    if large_hole:
        return np.repeat(np.repeat(mask, [1, int(rng.integers(3, 5)), 1], 0),
                         [1, int(rng.integers(3, 5)), 1], 1)
    for _ in range(20):
        rows = rng.choice([1, 2, 3], len(mask), p=[.65, .3, .05])
        cols = rng.choice([1, 2, 3], mask.shape[1], p=[.65, .3, .05])
        if rows.sum() <= 9 and cols.sum() <= 9:
            return np.repeat(np.repeat(mask, rows, 0), cols, 1)
    return mask.copy()


def _random_tree(rng, max_depth=2, max_arity=4, budget=4):
    k = int(rng.integers(1, max_arity + 1))
    children = [None] * k
    if max_depth > 1 and budget > 1:
        available = list(map(int, rng.permutation(k)))
        n = int(rng.integers(0, min(k, budget-1, 2)+1))
        for i in available[:n]:
            children[i] = _random_tree(rng, max_depth-1, max_arity, max(1, (budget-1)//max(1, n)))
    return tuple(children)


def _test_tree(rng, max_arity):
    # A new third nesting level recombines arities already visibly available.
    bottom = tuple([None] * int(rng.integers(2, max_arity+1)))
    middle = [None] * int(rng.integers(1, min(3, max_arity)+1))
    middle[int(rng.integers(len(middle)))] = bottom
    root = [None] * int(rng.integers(1, min(3, max_arity)+1))
    root[int(rng.integers(len(root)))] = tuple(middle)
    if len(root) > 1 and rng.random() < .35:
        empty = [i for i, child in enumerate(root) if child is None]
        root[_pick(rng, empty)] = tuple([None] * int(rng.integers(1, max_arity+1)))
    return tuple(root)


def _blob(rng, compact):
    cap = 2 if compact else 3
    h, w = int(rng.integers(1, cap+1)), int(rng.integers(1, cap+1))
    mask = np.ones((h, w), dtype=bool)
    if min(h, w) >= 2 and rng.random() < .65:
        mask[_pick(rng, [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)])] = False
    return mask


def _join_parts(rng, parts, compact):
    # A row-band layout supports horizontal, vertical, and mixed sibling groups.
    rows, current = [], []
    for i, part in enumerate(parts):
        current.append(part)
        if i == len(parts)-1 or rng.random() < .42:
            rows.append(current)
            current = []
    row_masks = []
    cap = 2 if compact else 4
    for row in rows:
        gaps = [int(rng.integers(1, cap)) for _ in row[:-1]]
        h = max(p.shape[0] for p in row)
        w = sum(p.shape[1] for p in row) + sum(gaps)
        mask = np.zeros((h, w), dtype=bool)
        x = 0
        for i, part in enumerate(row):
            y = (h - part.shape[0]) // 2
            mask[y:y+part.shape[0], x:x+part.shape[1]] = part
            x += part.shape[1] + (gaps[i] if i < len(gaps) else 0)
        row_masks.append(mask)
    gaps = [int(rng.integers(1, cap)) for _ in row_masks[:-1]]
    h = sum(p.shape[0] for p in row_masks) + sum(gaps)
    w = max(p.shape[1] for p in row_masks)
    mask = np.zeros((h, w), dtype=bool)
    y = 0
    for i, part in enumerate(row_masks):
        x = _pick(rng, [0, (w-part.shape[1])//2, w-part.shape[1]])
        mask[y:y+part.shape[0], x:x+part.shape[1]] = part
        y += part.shape[0] + (gaps[i] if i < len(gaps) else 0)
    return mask


def _draw_tree(rng, tree, compact=False):
    if tree is None:
        return _blob(rng, compact)
    content = _join_parts(rng, [_draw_tree(rng, c, compact) for c in tree], compact)
    padding = [int(rng.integers(2, 4 if compact else 5)) for _ in range(4)]
    top, bottom, left, right = padding
    h, w = content.shape[0]+top+bottom, content.shape[1]+left+right
    filled = np.ones((h, w), dtype=bool)
    # Independent corner chamfers and occasional bays change the actual outline.
    for sy, sx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        cut = int(rng.integers(0, max(1, min(padding[sy], padding[2+sx])-1)))
        for y in range(cut):
            for x in range(cut-y):
                filled[h-1-y if sy else y, w-1-x if sx else x] = False
    if min(padding) >= 3 and h >= 9 and w >= 9 and rng.random() < .45:
        if rng.random() < .5:
            filled[h//2:h//2+2, 0] = False
        else:
            filled[0, w//2:w//2+2] = False
    padded = np.pad(filled, 1)
    eroded = np.ones((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy+h, dx:dx+w]
    outline = filled & ~eroded
    outline[top:top+content.shape[0], left:left+content.shape[1]] |= content
    return outline


def _pack(rng, shapes):
    """Pack whole object boxes in a 28x28 interior, with one-cell gaps."""
    order = sorted(range(len(shapes)), key=lambda i: -shapes[i][0]*shapes[i][1])
    if any(max(shape) > 28 for shape in shapes):
        return None
    for attempt in range(12):
        board_h = int(rng.integers(max(18, max(s[0] for s in shapes)), 29)) if attempt < 8 else 28
        board_w = int(rng.integers(max(18, max(s[1] for s in shapes)), 29)) if attempt < 8 else 28
        boxes = []
        placements = {}
        for i in order:
            h, w = shapes[i]
            if h > 28 or w > 28:
                return None
            yy, xx = np.indices((board_h-h+1, board_w-w+1))
            free = np.ones(yy.shape, dtype=bool)
            for a, b, c, d in boxes:
                free &= (yy+h < a) | (yy > c) | (xx+w < b) | (xx > d)
            origins = np.argwhere(free)
            if not len(origins):
                break
            # Boundary-biased positions improve packing without fixing a shelf layout.
            edges = origins[(origins[:, 0] == 0) | (origins[:, 1] == 0) |
                            (origins[:, 0] == board_h-h) | (origins[:, 1] == board_w-w)]
            options = edges if len(edges) and rng.random() < .8 else origins
            y, x = map(int, options[int(rng.integers(len(options)))])
            box = (y, x, y+h, x+w)
            boxes.append(box)
            placements[i] = box
        if len(placements) == len(shapes):
            return placements
    return None


def _make_pair(rng, kind, palette=None):
    for attempt in range(180):
        max_arity = int(rng.choice([3, 3, 4]))
        masks = {k: _stencil(rng, k, diagonal=(kind == 'unary' and k == 2),
                             shared=(kind == 'branch' and k == 3)) for k in range(1, max_arity+1)}
        if kind == 'branch':
            sockets = np.argwhere(~masks[3])
            eligible = [i for i in range(3) if any(i != j and np.any(sockets[i] == sockets[j]) for j in range(3))]
            nested = _pick(rng, eligible)
            children = [None, None, None]
            children[nested] = (None,)
            tree = tuple(children)
        elif kind == 'unary':
            tree = ((None, None),)
        elif kind == 'test':
            tree = _test_tree(rng, max_arity)
        else:
            tree = _random_tree(rng, max_depth=2, max_arity=max_arity)
        colors = list(map(int, rng.permutation(10))) if palette is None else list(palette)
        bg, ink = colors[:2]
        stencils = {k: (colors[k+1], mask) for k, mask in masks.items()}
        output = _assemble(tree, stencils, bg, ink)
        if max(output.shape) > 30:
            continue
        sketch = _draw_tree(rng, tree, compact=attempt >= 5 or rng.random() < .35)
        if max(sketch.shape) > 27:
            continue
        raw_panels = {}
        for k, mask in masks.items():
            raw_panels[k] = (_stretch_stencil(rng, mask, large_hole=True) if kind == 'unary' and k == 1
                             else mask.copy() if kind == 'unary' and k == 2
                             else _stretch_stencil(rng, mask))
        arrays = [np.where(sketch, ink, bg).astype(np.int8)]
        ks = list(map(int, rng.permutation(list(masks))))
        arrays += [np.where(raw_panels[k], stencils[k][0], bg).astype(np.int8) for k in ks]
        placements = _pack(rng, [a.shape for a in arrays])
        if placements is None:
            continue
        canvas = np.full((28, 28), bg, dtype=np.int8)
        for i, a in enumerate(arrays):
            y, x, yy, xx = placements[i]
            canvas[y:yy, x:xx] = a
        y0 = min(b[0] for b in placements.values())
        x0 = min(b[1] for b in placements.values())
        y1 = max(b[2] for b in placements.values())
        x1 = max(b[3] for b in placements.values())
        canvas = canvas[y0:y1, x0:x1]
        margins = [int(rng.integers(1, 4)) for _ in range(4)]
        margins[1] = min(margins[1], 30-canvas.shape[0]-1)
        margins[0] = min(margins[0], 30-canvas.shape[0]-margins[1])
        margins[3] = min(margins[3], 30-canvas.shape[1]-1)
        margins[2] = min(margins[2], 30-canvas.shape[1]-margins[3])
        grid = np.pad(canvas, ((margins[0], margins[1]), (margins[2], margins[3])), constant_values=bg)
        try:
            info = _decode(grid)
            assert info['tree'] == tree
            assert np.array_equal(_assemble(info['tree'], info['stencils'], bg, ink), output)
            if kind == 'branch':
                assert any(len(o['cells']) > 1 and not o['hole'].any() for o in info['objects'])
        except AssertionError:
            continue
        return dict(input=grid, output=output)
    raise RuntimeError('bounded scene construction exhausted')


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(name + ' must be an integer in its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    # Palette policy is the only puzzle-level sampling choice; it changes no rule.
    palette = rng.permutation(10) if rng.random() < .2 else None
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, kinds in (('train', ['branch', 'unary'] + ['random']*(nt-2)),
                         ('test', ['test']*ne)):
        for kind in kinds:
            for _ in range(20):
                pair = _make_pair(rng, kind, palette)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not produce a distinct input')
        rng.shuffle(puzzle[split])
    return puzzle


def _output_tree(grid, bg, ink):
    records = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        records += [_record(c, grid.shape, color) for c in _components(grid == color)]
    for obj in records:
        if obj['color'] != ink:
            y0, x0, y1, x1 = obj['box']
            local = obj['mask'][y0:y1, x0:x1]
            assert np.all(_perimeter(local)), 'every assembled enclosure has a rectangular boundary'
            for cavity in _components(obj['hole']):
                a, b, c, d = _box(cavity)
                assert len(cavity) == (c-a)*(d-b), 'assembled sockets are rectangles'
    return _tree_from_records(records, ink)[0]


def _training_coverage(infos):
    assert any(len(i['tree']) == 3 and any(c is not None for c in i['tree']) for i in infos)
    assert any(len(i['tree']) == 1 and i['tree'][0] is not None and len(i['tree'][0]) >= 2 for i in infos)
    assert any(i['connectivity'] for i in infos), 'demonstrate four-connected cavity counts'
    assert any(any(a < b and i['areas'][a] > i['areas'][b] for a in i['areas'] for b in i['areas'])
               for i in infos), 'neutral area must not determine the matching'
    assert any(any(len(o['cells']) > 1 and not o['hole'].any() for o in i['objects']) for i in infos)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for value in pair.values():
                a = np.asarray(value)
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= d <= 30 for d in a.shape) and np.all((0 <= a) & (a <= 9))
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(out, _assemble(info['tree'], info['stencils'], info['background'], info['ink'])), 'incorrect recursive socket assembly'
            assert _output_tree(out, info['background'], info['ink']) == info['tree'], 'containment or child order changed'
            assert int((out == info['ink']).sum()) == _leaf_count(info['tree'])
            assert len(_components(out == info['ink'])) == _leaf_count(info['tree'])
            assert np.all(_perimeter(out) == info['stencils'][len(info['tree'])][0]), 'exact root crop'
            assert not np.array_equal(inp, out)
            if split == 'train':
                training.append(info)
    _training_coverage(training)


def _puzzle_key(p):
    return tuple(tuple((_key(x['input']), _key(x['output'])) for x in p[s]) for s in ('train', 'test'))


def self_test():
    """Bounded fixed-seed, semantic, mutation, and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, answers, trees, sockets, palettes = set(), set(), set(), set(), set()
    shared, metric, mixed, areas, leaf_areas = 0, 0, 0, set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                shapes.add(pair['input'].shape)
                roles = {info['background']: 0, info['ink']: 1}
                for k, (color, mask) in sorted(info['stencils'].items()):
                    roles[color] = k+1
                    sockets.add(_key(mask))
                    ys, xs = np.nonzero(~mask)
                    shared += len(set(ys)) < k or len(set(xs)) < k
                    areas.update(info['cavity_sizes'][k])
                remap = np.array([roles.get(c, 9) for c in range(10)])
                answers.add(_key(remap[pair['output']]))
                palettes.add((info['background'], info['ink']))
                trees.add(repr(info['tree']))
                metric += any(sum(info['cavity_sizes'][k]) > k for k in info['stencils'])
                leaf_areas.update(len(o['cells']) for o in info['objects'] if not o['hole'].any())
                for ids in info['children']:
                    if len(ids) >= 3:
                        bands = {info['objects'][j]['box'][0] for j in ids}
                        mixed += len(bands) > 1
                if split == 'test':
                    assert _depth(info['tree']) == 3
        assert max(_depth(_decode(p['input'])['tree']) for p in puzzle['train']) <= 2
    assert len(shapes) >= 25 and len(answers) >= 65 and len(trees) >= 25
    assert len(sockets) >= 20 and len(palettes) >= 30
    assert shared >= 50 and metric >= 100 and mixed >= 20
    assert max(areas) >= 9 and len(leaf_areas) >= 5
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    for nt, ne in ((2, 1), (10, 4), (7, 3)):
        validate_generated_puzzle(generate_puzzle(314159, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    rng = np.random.default_rng(91287)
    for puzzle in puzzles[:8]:
        pair = puzzle['test'][0]
        info = _decode(pair['input'])
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
        # Arbitrary band stretching of a library stencil retains its meaning.
        for k, (color, mask) in info['stencils'].items():
            stretched = np.repeat(np.repeat(mask, 2, 0), 2, 1)
            assert np.array_equal(_compress(stretched), mask)
        # The broader rule also accepts repeated gallery colors.
        lookup = np.arange(10, dtype=np.int8)
        common = info['stencils'][min(info['stencils'])][0]
        for color, _ in info['stencils'].values():
            lookup[color] = common
        assert np.array_equal(solve(lookup[pair['input']]), lookup[pair['output']])
        # The unary-over-binary demonstration supplies explicit A/B ablations.
        witness = next(p for p in puzzle['train'] if _decode(p['input'])['tree'] == ((None, None),))
        wi = _decode(witness['input'])
        flat = (None, None)
        assert not np.array_equal(_assemble(flat, wi['stencils'], wi['background'], wi['ink']), witness['output'])
        rows = {}
        for k, (color, mask) in wi['stencils'].items():
            strip = np.ones((3, 2*k+1), dtype=bool)
            strip[1, 1::2] = False
            rows[k] = (color, strip)
        assert not np.array_equal(_assemble(wi['tree'], rows, wi['background'], wi['ink']), witness['output'])
        # Remove unused panels: count sets need not be consecutive or exhaustive.
        branch = next(p for p in puzzle['train']
                      if len(_decode(p['input'])['tree']) == 3
                      and sorted(map(repr, _decode(p['input'])['tree'])) == sorted(map(repr, [None, None, (None,)])))
        bi = _decode(branch['input'])
        sparse = branch['input'].copy()
        for k, (color, _) in bi['stencils'].items():
            if k not in (1, 3):
                sparse[sparse == color] = bi['background']
        assert np.array_equal(solve(sparse), branch['output'])
        # Order has a visible effect when children have unequal subtrees.
        moved = list(bi['tree'])
        old = next(i for i, child in enumerate(moved) if child is not None)
        new = (old + 1) % 3
        moved[old], moved[new] = moved[new], moved[old]
        assert not np.array_equal(_assemble(tuple(moved), bi['stencils'], bi['background'], bi['ink']), branch['output'])
    broken = copy.deepcopy(puzzles[0])
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, input_shapes=len(shapes), normalized_outputs=len(answers),
                ordered_trees=len(trees), canonical_stencils=len(sockets),
                background_ink_assignments=len(palettes), cavity_areas=sorted(areas),
                blob_areas=sorted(leaf_areas))


if __name__ == '__main__':
    print(self_test())
