"""Registered screens on a patchwork substrate. See task_spec.md for the rule.

No runtime files, parent imports, saved examples, or generation state are needed
by solve(). Generation constructs labels from its sampled objects; validation
recovers those objects from visible cells before comparing the labels.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
# The seven-cell region is always the source; the five-cell region is paint.
_ARROW = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in _D4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(cells)
    return result


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _view(grid, box):
    r, c, h, w = box
    return grid[r:r+h, c:c+w]


def _border(a, color):
    return (np.all(a[0] == color) and np.all(a[-1] == color)
            and np.all(a[:, 0] == color) and np.all(a[:, -1] == color))


def _coverage(shape, anchors, templates):
    hits = np.zeros(shape, dtype=np.int16)
    for r, c, key in anchors:
        for dr, dc in templates[key]:
            rr, cc = r + dr, c + dc
            if 1 <= rr < shape[0] - 1 and 1 <= cc < shape[1] - 1:
                hits[rr, cc] += 1
    return hits


def _paint(board, background, mapping, hits):
    """Set operation on ORIGINAL material identities, regardless of hit count."""
    result = board.copy()
    for source, target in mapping.items():
        mask = board == source
        result[mask] = background
        result[mask & (hits > 0)] = target
    return result


def _parse(input_grid):
    grid = _grid(input_grid)
    # Only the board can have a monochromatic hollow border this large.
    frames = []
    for color in np.unique(grid):
        for cells in _components(grid == color):
            box = _box(cells)
            _, _, h, w = box
            if min(h, w) >= 7 and len(cells) == 2*h + 2*w - 4 and _border(_view(grid, box), color):
                frames.append((box, int(color)))
    assert len(frames) == 1, 'one isolated, exclusive-color board frame required'
    box, frame = frames[0]
    assert np.count_nonzero(grid == frame) == 2*box[2] + 2*box[3] - 4
    board = _view(grid, box)
    background = int(board[1, 1])
    assert background != frame
    inner_rim = board[1:-1, 1:-1]
    assert _border(inner_rim, background), 'a clear inner rim identifies the background'
    exterior = grid != background
    r, c, h, w = box
    exterior[r:r+h, c:c+w] = False
    mapping, templates, copies, ink_colors = {}, {}, Counter(), set()
    clue_boxes = []
    for cells in _components(exterior):
        cb = _box(cells)
        tile = _view(grid, cb)
        colors, counts = np.unique([grid[a, b] for a, b in cells], return_counts=True)
        clue_boxes.append(cb)
        if len(cells) == 12 and sorted(tile.shape) == [3, 4] and sorted(counts.tolist()) == [5, 7]:
            source = int(colors[np.flatnonzero(counts == 7)[0]])
            target = int(colors[np.flatnonzero(counts == 5)[0]])
            mask = tile == source
            assert any(np.array_equal(mask, np.rot90(base, q))
                       for base in (_ARROW, np.fliplr(_ARROW)) for q in range(4)), 'unknown mapping glyph'
            assert source not in mapping and source != target
            mapping[source] = target
        else:
            assert 3 <= len(cells) <= 10 and max(tile.shape) <= 5
            assert len(colors) == 2 and sorted(counts.tolist())[0] == 1
            key = int(colors[np.flatnonzero(counts == 1)[0]])
            ink = int(colors[np.flatnonzero(counts > 1)[0]])
            ar, ac = next((a, b) for a, b in cells if grid[a, b] == key)
            offsets = tuple(sorted((a-ar, b-ac) for a, b in cells))
            assert key not in templates or templates[key] == offsets, 'conflicting screen copies'
            templates[key] = offsets
            copies[key] += 1
            ink_colors.add(ink)
    assert 1 <= len(templates) <= 3 and 1 <= len(mapping) <= 3 and len(ink_colors) == 1
    ink = next(iter(ink_colors))
    keys, materials, paints = set(templates), set(mapping), set(mapping.values())
    structural = {background, frame, ink}
    assert len(structural) == 3 and not keys & structural
    assert not (materials | paints) & (structural | keys)
    assert not keys & materials
    # Whole clue bounding rectangles have a blank-cell gap, including holes.
    for i, (a, b, hh, ww) in enumerate(clue_boxes):
        assert a+hh < r or r+h < a or b+ww < c or c+w < b
        for aa, bb, hhh, www in clue_boxes[i+1:]:
            assert a+hh < aa or aa+hhh < a or b+ww < bb or bb+www < b
    assert set(map(int, np.unique(board[1:-1, 1:-1]))) <= materials | keys | {background}
    assert all(np.any(board == s) for s in materials)
    anchors = [(int(a), int(b), int(board[a, b])) for a, b in np.argwhere(np.isin(board, sorted(keys)))]
    assert keys == {key for _, _, key in anchors} and 2 <= len(anchors) <= 8
    for i, (a, b, _) in enumerate(anchors):
        assert all(max(abs(a-aa), abs(b-bb)) >= 2 for aa, bb, _ in anchors[i+1:])
    hits = _coverage(board.shape, anchors, templates)
    return dict(grid=grid, box=box, board=board, background=background, frame=frame,
                ink=ink, templates=templates, copies=dict(copies), mapping=mapping,
                anchors=anchors, hits=hits)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input only; key direction is fixed by the glyph's areas."""
    a = _parse(input_grid)
    return _paint(a['board'], a['background'], a['mapping'], a['hits'])


def _features(a):
    board, hits, mapping = a['board'], a['hits'], a['mapping']
    material = np.isin(board, list(mapping))
    active = material & (hits > 0)
    expected = _paint(board, a['background'], mapping, hits)
    twice = expected.copy()
    # Counterfactual: remap a painted cell a second time if it was hit twice.
    for source, target in mapping.items():
        if target in mapping:
            twice[(board == source) & (hits > 1)] = mapping[target]
    first_key = min(a['templates'])
    wrong_templates = {key: a['templates'][first_key] for key in a['templates']}
    wrong_keys = _paint(board, a['background'], mapping,
                        _coverage(board.shape, a['anchors'], wrong_templates))
    rectangles = {}
    reflected = {}
    for key, offsets in a['templates'].items():
        r, c, h, w = _box(offsets)
        rectangles[key] = tuple((rr, cc) for rr in range(r, r+h) for cc in range(c, c+w))
        reflected[key] = tuple((dr, -dc) for dr, dc in offsets)
    wrong_rect = _paint(board, a['background'], mapping, _coverage(board.shape, a['anchors'], rectangles))
    wrong_flip = _paint(board, a['background'], mapping, _coverage(board.shape, a['anchors'], reflected))
    nearest_hits = np.zeros(board.shape, dtype=np.int16)
    for r, c in np.argwhere(material):
        ar, ac, key = min(a['anchors'], key=lambda p: (abs(r-p[0])+abs(c-p[1]), p))
        nearest_hits[r, c] = (int(r)-ar, int(c)-ac) in a['templates'][key]
    wrong_nearest = _paint(board, a['background'], mapping, nearest_hits)
    sizes = [_box(offsets)[2:] for offsets in a['templates'].values()]
    essential_keys = 0
    for key in a['templates']:
        other_hits = _coverage(board.shape, [anchor for anchor in a['anchors'] if anchor[2] != key], a['templates'])
        essential_keys += bool(np.any(active & (other_hits == 0)))
    straddling = 0
    for ar, ac, key in a['anchors']:
        touched = {int(board[ar+dr, ac+dc]) for dr, dc in a['templates'][key]
                   if 1 <= ar+dr < board.shape[0]-1 and 1 <= ac+dc < board.shape[1]-1}
        straddling += len(touched & set(mapping)) >= 2
    outside = 0
    for r, c, key in a['anchors']:
        outside += sum(not (1 <= r+dr < board.shape[0]-1 and 1 <= c+dc < board.shape[1]-1)
                       for dr, dc in a['templates'][key])
    return dict(keys=len(a['templates']), materials=len(mapping), anchors=len(a['anchors']),
                printed=int(active.sum()), erased=int((material & (hits == 0)).sum()),
                clipped=int(((board == a['background']) & (hits > 0))[2:-2, 2:-2].sum()),
                overlap=int((active & (hits > 1)).sum()), boundary_clipped=outside,
                even_overlap=int((active & (hits > 0) & (hits % 2 == 0)).sum()),
                nearest_anchor_diff=int(np.count_nonzero(expected != wrong_nearest)),
                repeated_map_diff=int(np.count_nonzero(expected != twice)),
                key_choice_diff=int(np.count_nonzero(expected != wrong_keys)),
                rectangle_diff=int(np.count_nonzero(expected != wrong_rect)),
                reflection_diff=int(np.count_nonzero(expected != wrong_flip)),
                max_extent=max(max(hw) for hw in sizes),
                footprint_sizes=tuple(sorted(len(s) for s in a['templates'].values())),
                geometry=tuple(sorted(a['templates'].values())),
                map_overlap=sum(t in mapping for t in mapping.values()),
                many_to_one=len(set(mapping.values())) < len(mapping),
                essential_keys=essential_keys, straddling_anchors=straddling,
                printed_sources=len(set(map(int, board[active]))),
                input_shape=a['grid'].shape, output_shape=board.shape,
                palette=(a['background'], a['frame'], a['ink']),
                disconnected_material=any(len(_components(board == s)) > 1 for s in mapping))


def _screen(rng, limit, minimum_extent=1, informative=False):
    for _ in range(200):
        h, w = int(rng.integers(2, limit+1)), int(rng.integers(2, limit+1))
        n = int(rng.integers(3, min(10, h*w)+1))
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        while len(cells) < n:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4
                               if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            cells.add(_pick(rng, frontier))
        ar, ac = _pick(rng, sorted(cells))
        offsets = tuple(sorted((r-ar, c-ac) for r, c in cells))
        _, _, bh, bw = _box(offsets)
        if max(bh, bw) < minimum_extent:
            continue
        if informative and (bh*bw == n or set(offsets) == {(r, -c) for r, c in offsets}):
            continue
        return offsets
    # This fixed recovery is itself a valid asymmetric, nonrectangular screen.
    n = max(3, minimum_extent)
    return tuple([(r, 0) for r in range(n)] + [(n-1, 1)])


def _available(occupied, h, w):
    table = np.pad(occupied.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = table[h:, w:] - table[:-h, w:] - table[h:, :-w] + table[:-h, :-w]
    return np.argwhere(sums == 0)


def _reserve(occupied, r, c, h, w):
    occupied[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True


def _pack(rng, rectangles):
    """Try dispersed clues; then use bounded, guaranteed two-bank packing."""
    for attempt in range(18):
        h = int(rng.integers(11, 20 if attempt < 10 else 15))
        w = int(rng.integers(12, 25))
        r = int(_pick(rng, [0, 30-h, int(rng.integers(31-h))]))
        c = int(_pick(rng, [0, 30-w, int(rng.integers(31-w))]))
        occupied = np.zeros((30, 30), dtype=bool)
        _reserve(occupied, r, c, h, w)
        places = {}
        order = list(range(len(rectangles)))
        rng.shuffle(order)
        order.sort(key=lambda i: -rectangles[i][0]*rectangles[i][1])
        for i in order:
            hh, ww, _, _ = rectangles[i]
            candidates = _available(occupied, hh, ww)
            if not len(candidates):
                break
            a, b = map(int, candidates[int(rng.integers(len(candidates)))])
            places[i] = (a, b)
            _reserve(occupied, a, b, hh, ww)
        if len(places) == len(rectangles):
            return (r, c, h, w), places
    places, top = {}, 0
    for kind in ('screen', 'map'):
        x, y, row_height = 0, top, 0
        order = [i for i, rect in enumerate(rectangles) if rect[2] == kind]
        rng.shuffle(order)
        for i in order:
            hh, ww, _, _ = rectangles[i]
            if x+ww > 30:
                y += row_height+1
                x, row_height = 0, 0
            places[i] = (y, x)
            x += ww+1
            row_height = max(row_height, hh)
        top = y+row_height+1
    assert top <= 18
    h, w = int(rng.integers(10, min(16, 30-top)+1)), int(rng.integers(12, 29))
    return (top, int(rng.integers(31-w)), h, w), places


def _anchor_positions(rng, shape, templates, overlap):
    h, w = shape
    keys = list(templates)
    positions = []
    if overlap:
        key = keys[0]
        offsets = templates[key]
        choices = [(u, v) for u in offsets for v in offsets
                   if max(abs(u[0]-v[0]), abs(u[1]-v[1])) >= 2
                   and u != (0, 0) and v != (0, 0)]
        if not choices:
            return None
        u, v = _pick(rng, choices)
        dr, dc = u[0]-v[0], u[1]-v[1]
        choices = [(r, c) for r in range(2, h-2) for c in range(2, w-2)
                   if 2 <= r+dr < h-2 and 2 <= c+dc < w-2]
        if not choices:
            return None
        r, c = _pick(rng, choices)
        positions = [(r, c, key), (r+dr, c+dc, key)]
    requests = [key for key in keys if not any(k == key for _, _, k in positions)]
    total = int(rng.integers(max(2, len(positions)+len(requests)), min(8, len(keys)+5)+1))
    requests += [_pick(rng, keys) for _ in range(total-len(positions)-len(requests))]
    for key in requests:
        candidates = [(r, c) for r in range(2, h-2) for c in range(2, w-2)
                      if all(max(abs(r-a), abs(c-b)) >= 2 for a, b, _ in positions)]
        if not candidates:
            return None
        r, c = _pick(rng, candidates)
        positions.append((r, c, key))
    return positions


def _substrate(rng, shape, background, sources):
    h, w = shape
    y, x = np.indices(shape)
    core = (y >= 2) & (y < h-2) & (x >= 2) & (x < w-2)
    style = int(rng.integers(4))
    if style == 0:
        # A filled sheet with a varying stepped contour.
        left = np.clip(np.cumsum(rng.integers(-1, 2, size=h)), 0, max(1, w//4))
        right = np.clip(np.cumsum(rng.integers(-1, 2, size=h)), 0, max(1, w//4))
        core &= (x >= 2+left[:, None]) & (x < w-2-right[:, None])
    elif style == 1:
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(2, h-3)), int(rng.integers(2, w-3))
            core[r:r+int(rng.integers(1, 4)), c:c+int(rng.integers(1, 4))] = False
    elif style == 2:
        islands = np.zeros(shape, dtype=bool)
        for _ in range(int(rng.integers(3, 8))):
            r, c = int(rng.integers(2, h-3)), int(rng.integers(2, w-3))
            islands[r:r+int(rng.integers(2, 7)), c:c+int(rng.integers(2, 7))] = True
        core &= islands
    else:
        period = int(rng.integers(5, 9))
        core &= (y + int(_pick(rng, [-2, -1, 1, 2]))*x + int(rng.integers(period))) % period < period-2
    cells = np.argwhere(core)
    if len(cells) < 3*len(sources):
        return None
    centers = cells[rng.choice(len(cells), len(sources), replace=False)]
    distance = np.stack([abs(y-r)+abs(x-c) for r, c in centers])
    territories = np.argmin(distance, axis=0)
    board = np.full(shape, background, dtype=np.int8)
    for i, source in enumerate(sources):
        board[core & (territories == i)] = source
    return board


def _mapping(rng, sources, extra, overlap_source=None):
    sources = list(sources)
    if overlap_source is not None:
        other = next(s for s in sources if s != overlap_source)
        return {overlap_source: other, other: extra[0]}
    kind = int(rng.integers(4))
    if kind == 0 and len(sources) > 1:
        order = list(map(int, rng.permutation(sources)))
        return {s: order[(i+1) % len(order)] for i, s in enumerate(order)}
    if kind == 1:
        target = int(_pick(rng, extra))
        return {s: target for s in sources}
    if kind == 2:
        order = list(map(int, rng.permutation(sources)))
        return {s: order[i+1] if i+1 < len(order) else extra[0] for i, s in enumerate(order)}
    return {s: int(_pick(rng, [t for t in sources+extra if t != s])) for s in sources}


def _generate_pair(rng, key_count, limit, test_scale, demonstrate, key_demo):
    # Bounded rejections enforce observable witnesses, not hidden output codes.
    for _ in range(160):
        palette = list(map(int, rng.permutation(10)))
        background, frame, ink = palette[:3]
        keys = palette[3:3+key_count]
        m = 2 if demonstrate else int(rng.integers(1, 4))
        sources = palette[3+key_count:3+key_count+m]
        extra = palette[3+key_count+m:]
        templates = {key: _screen(rng, limit, 4 if test_scale and i == 0 else 3 if i == 0 else 1,
                                   informative=i == 0) for i, key in enumerate(keys)}
        if key_demo and len(set(templates.values())) != len(templates):
            continue
        rectangles = []
        for i, key in enumerate(keys):
            _, _, h, w = _box(templates[key])
            repeats = 2 if (demonstrate and i == 0) or rng.random() < .23 else 1
            rectangles.extend((h, w, 'screen', key) for _ in range(repeats))
        # Glyph orientation varies independently of spatial order and mapping.
        glyphs = {}
        for source in sources:
            glyphs[source] = np.rot90(_ARROW, int(rng.integers(4))).copy()
            hh, ww = glyphs[source].shape
            rectangles.append((hh, ww, 'map', source))
        box, places = _pack(rng, rectangles)
        board = _substrate(rng, box[2:], background, sources)
        if board is None:
            continue
        anchors = _anchor_positions(rng, board.shape, templates, demonstrate)
        if anchors is None:
            continue
        for r, c, key in anchors:
            board[r, c] = key
        hits = _coverage(board.shape, anchors, templates)
        overlap_cells = np.argwhere(np.isin(board, sources) & (hits > 1))
        if demonstrate and not len(overlap_cells):
            continue
        overlap_source = int(board[tuple(overlap_cells[0])]) if demonstrate else None
        mapping = _mapping(rng, sources, extra, overlap_source)
        # Distinct generation path: construct the answer from sampled masks.
        output = board.copy()
        for r in range(1, board.shape[0]-1):
            for c in range(1, board.shape[1]-1):
                source = int(board[r, c])
                if source in mapping:
                    selected = any((r-ar, c-ac) in templates[key] for ar, ac, key in anchors)
                    output[r, c] = mapping[source] if selected else background
        board[0] = board[-1] = frame
        board[:, 0] = board[:, -1] = frame
        output[0] = output[-1] = frame
        output[:, 0] = output[:, -1] = frame
        grid = np.full((30, 30), background, dtype=np.int8)
        _view(grid, box)[:] = board
        for i, (hh, ww, kind, identity) in enumerate(rectangles):
            r, c = places[i]
            tile = grid[r:r+hh, c:c+ww]
            if kind == 'screen':
                lr, lc, _, _ = _box(templates[identity])
                for dr, dc in templates[identity]:
                    tile[dr-lr, dc-lc] = identity if (dr, dc) == (0, 0) else ink
            else:
                tile[:] = np.where(glyphs[identity], identity, mapping[identity])
        # Crop only unused canvas, with independent remaining margins.
        r, c, h, w = _box(np.argwhere(grid != background))
        r0, c0 = max(0, r-int(rng.integers(3))), max(0, c-int(rng.integers(3)))
        end_r, end_c = min(30, r+h+int(rng.integers(3))), min(30, c+w+int(rng.integers(3)))
        grid = grid[r0:end_r, c0:end_c].copy()
        turns, mirror = int(rng.integers(4)), bool(rng.integers(2))
        if mirror:
            grid, output = np.fliplr(grid), np.fliplr(output)
        grid, output = np.rot90(grid, turns).copy(), np.rot90(output, turns).copy()
        try:
            a = _parse(grid)
            f = _features(a)
            assert f['printed'] >= 3 and f['erased'] >= 3 and f['clipped'] >= 1
            assert f['essential_keys'] == key_count
            assert f['rectangle_diff'] and f['reflection_diff']
            assert not demonstrate or (f['repeated_map_diff'] and max(a['copies'].values()) >= 2)
            assert not demonstrate or (f['straddling_anchors'] and f['printed_sources'] == 2)
            assert not demonstrate or (f['even_overlap'] and f['nearest_anchor_diff'])
            assert not key_demo or f['key_choice_diff']
            assert np.array_equal(_paint(a['board'], a['background'], a['mapping'], a['hits']), output)
        except AssertionError:
            continue
        return {'input': grid, 'output': output}
    raise RuntimeError('bounded construction exhausted; no silent reduced-family fallback')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _array_key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(_pick(rng, ['keys', 'scale']))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if shift == 'keys':
                k = 3 if split == 'test' else 2 if index == 1 else int(rng.integers(1, 3))
                limit, test_scale = int(rng.integers(3, 6)), False
            else:
                k = max(2, int(rng.integers(1, 4))) if split == 'train' and index == 1 else int(rng.integers(1, 4))
                limit, test_scale = (3, False) if split == 'train' else (int(rng.integers(4, 6)), True)
            for _ in range(8):
                pair = _generate_pair(rng, k, limit, test_scale,
                                      split == 'train' and index == 0,
                                      split == 'train' and index == 1)
                key = _array_key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('could not produce distinct inputs')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    tr, te = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _array_key(inp)
            assert key not in seen
            seen.add(key)
            a = _parse(inp)
            predicted = _paint(a['board'], a['background'], a['mapping'], a['hits'])
            assert np.array_equal(predicted, out), 'incorrect material intersection, recoloring, preservation, or crop'
            assert _border(out, a['frame'])
            for r, c, identity in a['anchors']:
                assert out[r, c] == identity
            assert np.all(out[(a['board'] == a['background'])] == a['background'])
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            f = _features(a)
            assert f['printed'] >= 3 and f['erased'] >= 3 and f['clipped'] > 0
            assert f['essential_keys'] == f['keys'], 'every requested screen must affect material'
            assert f['rectangle_diff'] and f['reflection_diff']
            records[split].append(f)
    first, second = records['train'][:2]
    assert first['repeated_map_diff'] > 0, 'demonstrate union and one-step paint under overlapping screens'
    assert first['straddling_anchors'] > 0 and first['printed_sources'] == 2
    assert first['even_overlap'] > 0 and first['nearest_anchor_diff'] > 0
    assert max(_parse(puzzle['train'][0]['input'])['copies'].values()) >= 2
    assert second['keys'] >= 2 and second['key_choice_diff'] > 0, 'demonstrate key-specific geometry'
    assert (min(f['keys'] for f in records['test']) > max(f['keys'] for f in records['train'])
            or min(f['max_extent'] for f in records['test']) > max(f['max_extent'] for f in records['train'])), 'missing structural test shift'


def self_test():
    """42 puzzles in a streaming sweep; no disk output or retained grid corpus."""
    fingerprints, geometries, shapes, palettes, profiles = set(), set(), set(), set(), set()
    totals, key_counts, material_counts, extents, counts = Counter(), set(), set(), set(), Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        split_features = {}
        for split in ('train', 'test'):
            split_features[split] = []
            for pair in puzzle[split]:
                before = pair['input'].copy()
                assert np.array_equal(solve(pair['input']), pair['output'])
                assert np.array_equal(before, pair['input']), 'solve mutated its input'
                for role in ('input', 'output'):
                    digest.update(bytes(pair[role].shape))
                    digest.update(pair[role].tobytes())
                a = _parse(pair['input'])
                f = _features(a)
                split_features[split].append(f)
                geometries.update(f['geometry'])
                shapes.add((f['input_shape'], f['output_shape']))
                palettes.add(f['palette'])
                key_counts.add(f['keys'])
                material_counts.add(f['materials'])
                extents.add(f['max_extent'])
                profiles.add((f['keys'], f['materials'], f['anchors'], f['max_extent'], bool(f['map_overlap'])))
                for flag in ('overlap', 'boundary_clipped', 'map_overlap', 'many_to_one', 'disconnected_material', 'straddling_anchors'):
                    totals[flag] += bool(f[flag])
                counts['pairs'] += 1
                if seed < 4:
                    permutation = np.random.default_rng(7000+seed).permutation(10)
                    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
                    for turns in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(pair['input'], turns)), np.rot90(pair['output'], turns))
                    assert np.array_equal(solve(np.fliplr(pair['input'])), np.fliplr(pair['output']))
        fingerprints.add(digest.hexdigest())
        tr, te = split_features['train'], split_features['test']
        counts['key_shift'] += min(f['keys'] for f in te) > max(f['keys'] for f in tr)
        counts['scale_shift'] += min(f['max_extent'] for f in te) > max(f['max_extent'] for f in tr)
    assert len(fingerprints) == 42 and counts['pairs'] == 210
    assert key_counts == {1, 2, 3} and material_counts == {1, 2, 3} and {3, 4, 5} <= extents
    assert len(geometries) >= 120 and len(shapes) >= 80 and len(palettes) >= 100 and len(profiles) >= 60
    assert counts['key_shift'] >= 8 and counts['scale_shift'] >= 8
    assert all(totals[flag] >= 10 for flag in totals)
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(42000+train*10+test, train, test)
            assert (len(p['train']), len(p['test'])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    p = generate_puzzle(0)
    q = generate_puzzle(0)
    assert all(np.array_equal(a[role], b[role]) for s in p for a, b in zip(p[s], q[s]) for role in a)
    broken = {s: [{role: grid.copy() for role, grid in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('bad border label accepted')
    # Conflicting repetitions must not silently choose the last screen.
    a = _parse(p['train'][0]['input'])
    changed = p['train'][0]['input'].copy()
    exterior = changed != a['background']
    r, c, h, w = a['box']
    exterior[r:r+h, c:c+w] = False
    duplicated_key = next(key for key, count in a['copies'].items() if count >= 2)
    component = next(cells for cells in _components(exterior)
                     if any(changed[rr, cc] == duplicated_key for rr, cc in cells))
    ar, ac = next((rr, cc) for rr, cc in component if changed[rr, cc] == duplicated_key)
    # Keep a valid connected motif, but move its anchor within just one copy.
    rr, cc = next((ar+dr, ac+dc) for dr, dc in _D4 if (ar+dr, ac+dc) in component)
    changed[ar, ac], changed[rr, cc] = changed[rr, cc], changed[ar, ac]
    try:
        solve(changed)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged screen copy accepted')
    return dict(seeds=42, pairs=counts['pairs'], geometries=len(geometries), shape_pairs=len(shapes),
                structural_palettes=len(palettes), joint_profiles=len(profiles),
                key_shifts=counts['key_shift'], scale_shifts=counts['scale_shift'], witnesses=dict(totals))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
