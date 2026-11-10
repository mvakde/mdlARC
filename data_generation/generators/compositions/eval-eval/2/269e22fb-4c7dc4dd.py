"""Crop-addressed toggle stencils. All solving information is in each input.

Single rings enclose the source map and query crops. Double rings enclose
the demonstrated edited map and its calibration crop. The crop coordinate
system transports the demonstrated difference; overlapping toggles use XOR.
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np


def _transform(a, index):
    if index < 4:
        return np.rot90(a, index).copy()
    return (np.fliplr(a), np.flipud(a), a.T,
            np.flipud(np.fliplr(a.T)))[index - 4].copy()


_INVERSE = (0, 3, 2, 1, 4, 5, 6, 7)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _canonical(a):
    return min(_key(_transform(a, i) ^ s) for i in range(8) for s in (0, 1))


def _binary(a):
    colors = np.unique(a)
    if len(colors) != 2:
        raise ValueError("every map/crop needs exactly two interior colors")
    return (a == colors[1]).astype(np.uint8)


def _locations(full, patch):
    h, w = patch.shape
    if h > full.shape[0] or w > full.shape[1]:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    rows, cols = np.nonzero(np.all(windows == patch, axis=(-2, -1)))
    return list(zip(map(int, rows), map(int, cols)))


def _matches(orbit, patch):
    return [(i, swap, r, c) for i, full in enumerate(orbit)
            for swap in (0, 1)
            for r, c in _locations(full, patch ^ swap)]


def _unique_match(orbit, patch):
    matches = _matches(orbit, _binary(patch))
    if len(matches) != 1:
        raise ValueError("crop must have one orientation, role map and location")
    return matches[0]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("invalid ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = todo.pop()
        queue, cells = deque([start]), [start]
        while queue:
            r, c = queue.popleft()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in todo:
                    todo.remove(p)
                    queue.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _panels_for_color(grid, color):
    panels = []
    for cells in _components(grid == color):
        rs, cs = zip(*cells)
        r, c, h, w = min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        if min(h, w) < 4:
            return None
        sub = grid[r:r + h, c:c + w]
        thickness = None
        for t in (1, 2):
            if min(h, w) <= 2 * t:
                continue
            ring = np.ones((h, w), bool)
            ring[t:-t, t:-t] = False
            if np.array_equal(sub == color, ring) and len(cells) == int(ring.sum()):
                thickness = t
                break
        if thickness is None:
            return None
        t = thickness
        interior = sub[t:-t, t:-t].copy()
        if 0 in interior or len(np.unique(interior)) != 2:
            return None
        panels.append(dict(box=(r, c, h, w), thickness=t, grid=interior))
    return panels


def _extract(grid):
    scenes = []
    for color in np.unique(grid):
        if color == 0:
            continue
        panels = _panels_for_color(grid, int(color))
        if panels is None or not 4 <= len(panels) <= 6:
            continue
        large = [p for p in panels if min(p['grid'].shape) >= 7]
        small = [p for p in panels if 2 <= min(p['grid'].shape) and max(p['grid'].shape) <= 6]
        if len(large) != 2 or len(large) + len(small) != len(panels):
            continue
        before = [p for p in large if p['thickness'] == 1]
        after = [p for p in large if p['thickness'] == 2]
        keys = [p for p in small if p['thickness'] == 2]
        queries = [p for p in small if p['thickness'] == 1]
        if not (len(before) == len(after) == len(keys) == 1 and 1 <= len(queries) <= 3):
            continue
        before, after, calibration = before[0], after[0], keys[0]
        if (before['grid'].shape != after['grid'].shape
                or max(before['grid'].shape) > 11
                or not np.array_equal(np.unique(before['grid']), np.unique(after['grid']))):
            continue
        coverage = np.zeros(grid.shape, bool)
        disjoint = True
        for p in panels:
            r, c, h, w = p['box']
            if coverage[r:r + h, c:c + w].any():
                disjoint = False
            coverage[r:r + h, c:c + w] = True
        if not disjoint or np.any(grid[~coverage] != 0):
            continue
        scenes.append(dict(before=before, after=after, calibration=calibration,
                           queries=sorted(queries, key=lambda p: p['box']), frame=int(color)))
    if len(scenes) != 1:
        raise ValueError("expected one unambiguous collection of framed panels")
    return scenes[0]


def _analyze(grid):
    scene = _extract(grid)
    before, after = scene['before']['grid'], scene['after']['grid']
    orbit = [_transform(_binary(before), i) for i in range(8)]
    demo = _unique_match(orbit, scene['calibration']['grid'])
    i, _, r, c = demo
    difference = before != after
    if not difference.any():
        raise ValueError("the demonstrated stencil is empty")
    # Difference pixels are expressed in the calibration chip's local axes.
    offsets = np.argwhere(_transform(difference, i)) - np.array([r, c])
    masks, matches, losses = [], [], []
    for query in scene['queries']:
        j, swap, y, x = _unique_match(orbit, query['grid'])
        local = np.zeros(orbit[j].shape, bool)
        points = offsets + np.array([y, x])
        keep = ((points >= 0) & (points < np.array(local.shape))).all(axis=1)
        points = points[keep]
        local[points[:, 0], points[:, 1]] = True
        masks.append(_transform(local, _INVERSE[j]))
        matches.append((j, swap, y, x))
        losses.append(int((~keep).sum()))
    parity = np.zeros(before.shape, bool)
    union = np.zeros_like(parity)
    for mask in masks:
        parity ^= mask
        union |= mask
    return dict(scene=scene, demo=demo, offsets=offsets, difference=difference,
                matches=matches, masks=masks, parity=parity, union=union, losses=losses)


def _anchor_alternatives(analysis):
    """Two plausible wrong conventions, measured from input pixels only."""
    scene = analysis['scene']
    shape = scene['before']['grid'].shape
    demo_shape = np.array(scene['calibration']['grid'].shape)
    centered, bounded = np.zeros(shape, bool), np.zeros(shape, bool)
    for query, match in zip(scene['queries'], analysis['matches']):
        i, _, r, c = match
        qshape = np.array(query['grid'].shape)
        shifts = analysis['offsets'] + qshape // 2 - demo_shape // 2
        inside = ((analysis['offsets'] >= 0) & (analysis['offsets'] < qshape)).all(axis=1)
        for offsets, dest in ((shifts, centered), (analysis['offsets'][inside], bounded)):
            local = np.zeros(_transform(scene['before']['grid'], i).shape, bool)
            points = offsets + np.array([r, c])
            keep = ((points >= 0) & (points < np.array(local.shape))).all(axis=1)
            points = points[keep]
            local[points[:, 0], points[:, 1]] = True
            dest ^= _transform(local, _INVERSE[i])
    return centered, bounded


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only input pixels; no training labels, seed or saved reference."""
    analysis = _analyze(_grid(input_grid))
    before = analysis['scene']['before']['grid']
    colors = np.unique(before)
    result = before.copy()
    mask = analysis['parity']
    result[mask] = np.where(before[mask] == colors[0], colors[1], colors[0])
    return result


# The constructor uses point isometries, whereas solving transports full masks.
def _point(point, shape, transform):
    r, c = map(int, point)
    h, w = shape
    return ((r, c), (w - 1 - c, r), (h - 1 - r, w - 1 - c),
            (c, h - 1 - r), (r, w - 1 - c), (h - 1 - r, c),
            (c, r), (w - 1 - c, h - 1 - r))[transform]


def _origin(window, shape, transform):
    r, c, h, w = window
    corners = [_point(p, shape, transform) for p in
               ((r, c), (r + h - 1, c), (r, c + w - 1), (r + h - 1, c + w - 1))]
    return min(p[0] for p in corners), min(p[1] for p in corners)


def _project(offsets, window, shape, transform):
    origin = _origin(window, shape, transform)
    oriented_shape = shape if transform in (0, 2, 4, 5) else shape[::-1]
    cells = set()
    for dr, dc in offsets:
        p = (origin[0] + int(dr), origin[1] + int(dc))
        if 0 <= p[0] < oriented_shape[0] and 0 <= p[1] < oriented_shape[1]:
            cells.add(_point(p, oriented_shape, _INVERSE[transform]))
    return cells


def _line(a, p, q):
    n = max(abs(p[0] - q[0]), abs(p[1] - q[1])) + 1
    rr = np.rint(np.linspace(p[0], q[0], n)).astype(int)
    cc = np.rint(np.linspace(p[1], q[1], n)).astype(int)
    a[rr, cc] = 1


def _atlas_candidate(rng, shape):
    a = np.zeros(shape, np.uint8)
    h, w = shape
    for kind in rng.permutation(6):
        r0, r1 = sorted(map(int, rng.choice(h, 2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(w, 2, replace=False)))
        if kind == 0:  # open bend
            a[r0:r1 + 1, c0] = 1
            a[r1, c0:c1 + 1] = 1
        elif kind == 1:  # loop
            a[r0, c0:c1 + 1] = a[r1, c0:c1 + 1] = 1
            a[r0:r1 + 1, c0] = a[r0:r1 + 1, c1] = 1
        elif kind == 2:  # staircase
            _line(a, (r0, c0), (r1, c1))
        elif kind == 3:  # comb
            a[r0:r1 + 1, c0] = 1
            for r in range(r0, r1 + 1, 2):
                a[r, c0:c1 + 1] = 1
        elif kind == 4:  # short solid block
            a[r0:min(r0 + 2, h), c0:c1 + 1] = 1
        else:  # irregular orthogonal path
            p = (r0, c0)
            for k in range(4):
                q = (int(rng.integers(h)), p[1]) if k % 2 else (p[0], int(rng.integers(w)))
                _line(a, p, q)
                p = q
    if rng.integers(2):
        a ^= 1
    return a


_CROP_SHAPES = ((4, 4), (4, 5), (5, 4), (5, 5), (4, 6), (6, 4),
                (5, 6), (6, 5), (6, 6), (3, 5), (3, 6), (5, 3),
                (6, 3), (2, 6), (6, 2))


def _bank(a):
    orbit = [_transform(a, i) for i in range(8)]
    bank = []
    for h, w in _CROP_SHAPES:
        for r in range(a.shape[0] - h + 1):
            for c in range(a.shape[1] - w + 1):
                patch = a[r:r + h, c:c + w]
                if patch.min() != patch.max() and len(_matches(orbit, patch)) == 1:
                    bank.append((r, c, h, w))
    return bank


def _atlas(rng):
    shape = tuple(map(int, rng.integers(7, 12, 2)))
    for _ in range(100):
        a = _atlas_candidate(rng, shape)
        if not .20 <= a.mean() <= .80:
            continue
        if len({_key(_transform(a, i) ^ s) for i in range(8) for s in (0, 1)}) != 16:
            continue
        bank = _bank(a)
        broad = [b for b in bank if min(b[2:]) >= 4]
        narrow = [b for b in bank if min(b[2:]) <= 3]
        if len(broad) >= 18 and len(narrow) >= 10:
            return a, bank
    raise RuntimeError("could not build a structured atlas with distinctive strip crops")


def _stencil(rng):
    h, w = map(int, rng.integers(3, 7, 2))
    a = np.zeros((h, w), np.uint8)
    kind = int(rng.integers(4))
    if kind == 0:
        a[:, 0] = a[-1, :] = 1
        a[0, :int(rng.integers(1, w))] = 1
    elif kind == 1:
        a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 1
        a[int(rng.integers(1, h - 1)), 1] = 1
        a[0, int(rng.integers(1, w))] = 0
    elif kind == 2:
        c = int(rng.integers(1, w - 1))
        a[:, c] = 1
        a[0, :c + 1] = a[-1, c:] = 1
        a[int(rng.integers(1, h - 1)), :c + 1] = 1
    else:
        p = (int(rng.integers(h)), int(rng.integers(w)))
        for k in range(6):
            q = (int(rng.integers(h)), p[1]) if k % 2 else (p[0], int(rng.integers(w)))
            _line(a, p, q)
            p = q
    points = np.argwhere(a)
    points -= points.min(axis=0)
    points += rng.integers(-2, 3, 2)
    return [tuple(map(int, p)) for p in points]


def _chip(a, window, transform, palette):
    r, c, h, w = window
    return np.asarray(palette, np.int8)[_transform(a[r:r + h, c:c + w], transform)]


def _framed(a, thickness, color):
    t = thickness
    frame = np.full((a.shape[0] + 2 * t, a.shape[1] + 2 * t), color, np.int8)
    frame[t:-t, t:-t] = a
    return frame


def _layout(rng, maps, chips):
    """Two independently ordered shelves, with padding, offsets and transposition."""
    maps, chips = list(maps), list(chips)
    rng.shuffle(maps)
    rng.shuffle(chips)
    map_gap = int(rng.integers(1, min(4, 30 - sum(p.shape[1] for p in maps)) + 1))
    chip_slack = 30 - sum(p.shape[1] for p in chips)
    if chip_slack < len(chips) - 1:
        return None
    chip_gaps = [1] * (len(chips) - 1)
    for _ in range(int(rng.integers(chip_slack - sum(chip_gaps) + 1))):
        if chip_gaps:
            chip_gaps[int(rng.integers(len(chip_gaps)))] += 1
    widths = [sum(p.shape[1] for p in maps) + map_gap,
              sum(p.shape[1] for p in chips) + sum(chip_gaps)]
    heights = [max(p.shape[0] for p in maps), max(p.shape[0] for p in chips)]
    between = int(rng.integers(1, min(4, 30 - sum(heights)) + 1))
    width = int(rng.integers(max(widths), 31))
    height = int(rng.integers(sum(heights) + between, 31))
    top = int(rng.integers(height - sum(heights) - between + 1))
    grid = np.zeros((height, width), np.int8)
    order = int(rng.integers(2))
    shelves = [(maps, [map_gap], widths[0], heights[0]),
               (chips, chip_gaps, widths[1], heights[1])]
    if order:
        shelves.reverse()
    for panels, gaps, used_w, used_h in shelves:
        left = int(rng.integers(width - used_w + 1))
        for k, panel in enumerate(panels):
            y = top + int(rng.integers(used_h - panel.shape[0] + 1))
            grid[y:y + panel.shape[0], left:left + panel.shape[1]] = panel
            left += panel.shape[1] + (gaps[k] if k < len(gaps) else 0)
        top += used_h + between
    return grid


def _pair(rng, atlas, banks, base_palette, frame_color, variation, count, shifted, diagnostic):
    # Each attempted construction has the same supported rule. Rejection only
    # enforces readable crops and diagnostic properties, never changes labels.
    orientation = int(rng.integers(8))
    a = _transform(atlas, orientation)
    bank = banks[orientation]
    broad = [b for b in bank if min(b[2:]) >= 4]
    query_pool = [b for b in bank if min(b[2:]) <= 3] if shifted else broad
    available = [c for c in range(1, 10) if c != frame_color]
    palette = tuple(map(int, rng.choice(available, 2, replace=False))) if variation else base_palette
    before = np.asarray(palette, np.int8)[a]
    for _ in range(600):
        demo_window = broad[int(rng.integers(len(broad)))]
        demo_t = int(rng.integers(8))
        offsets = _stencil(rng)
        demo_cells = _project(offsets, demo_window, a.shape, demo_t)
        if len(demo_cells) != len(offsets) or len(offsets) < 5:
            continue  # calibration must reveal every stencil pixel
        queries, masks, query_specs = [], [], []
        for k in range(count):
            window = query_pool[int(rng.integers(len(query_pool)))]
            trans = int(rng.integers(8))
            if shifted:
                # Three strips fit beside the key by giving each a narrow width.
                h, w = _transform(np.zeros(window[2:], np.uint8), trans).shape
                if w > 3 or h < 5:
                    trans = (trans + 1) % 4 if trans < 4 else (6 if trans in (4, 5) else 4)
            cells = _project(offsets, window, a.shape, trans)
            if len(cells) < 3:
                break
            query_palette = palette if rng.random() < .25 else tuple(map(int, rng.choice(available, 2, replace=False)))
            queries.append(_chip(a, window, trans, query_palette))
            masks.append(cells)
            query_specs.append((window, trans))
        if len(queries) != count or len({_key(p) for p in queries}) != count:
            continue
        if count == 1 and len(masks[0]) != len(offsets):
            continue  # first demonstration shows an entire transported stencil
        parity, union = set(), set()
        for cells in masks:
            parity.symmetric_difference_update(cells)
            union.update(cells)
        if not parity or parity == demo_cells:
            continue
        rectangles = set()
        for window, _ in query_specs:
            r, c, h, w = window
            rectangles.symmetric_difference_update((y, x) for y in range(r, r + h) for x in range(c, c + w))
        if rectangles == parity:
            continue
        if count == 1:
            demo_shape = _transform(np.zeros(demo_window[2:], np.uint8), demo_t).shape
            query_shape = queries[0].shape
            centered_offsets = np.asarray(offsets) + np.array(query_shape) // 2 - np.array(demo_shape) // 2
            bounded_offsets = [p for p in offsets if 0 <= p[0] < query_shape[0] and 0 <= p[1] < query_shape[1]]
            window, trans = query_specs[0]
            if (_project(centered_offsets, window, a.shape, trans) == parity
                    or _project(bounded_offsets, window, a.shape, trans) == parity):
                continue  # teach upper-left anchoring and support beyond the chip
        if count > 1 and (union == parity or any(masks[i] == masks[j] for i in range(count) for j in range(i))):
            continue  # expose cancellation and distinct contributions
        # Ignore query orientation while retaining its true canonical window.
        wrong = set()
        for window, _ in query_specs:
            wrong.symmetric_difference_update(_project(offsets, window, a.shape, 0))
        if wrong == parity:
            continue
        if diagnostic and count > 1 and all(len(m) == len(offsets) for m in masks):
            continue  # second training example also teaches boundary clipping
        after_binary, out_binary = a.copy(), a.copy()
        for p in demo_cells:
            after_binary[p] ^= 1
        for p in parity:
            out_binary[p] ^= 1
        after = np.asarray(palette, np.int8)[after_binary]
        output = np.asarray(palette, np.int8)[out_binary]
        if len(np.unique(after)) != 2 or len(np.unique(output)) != 2:
            continue
        demo_palette = palette if rng.random() < .35 else tuple(map(int, rng.choice(available, 2, replace=False)))
        demo = _chip(a, demo_window, demo_t, demo_palette)
        if any(_key(demo) == _key(q) for q in queries):
            continue
        maps = [_framed(before, 1, frame_color), _framed(after, 2, frame_color)]
        chips = [_framed(demo, 2, frame_color)] + [_framed(q, 1, frame_color) for q in queries]
        grid = _layout(rng, maps, chips)
        if grid is None:
            continue
        if rng.integers(2):
            grid, output = grid.T.copy(), output.T.copy()
        # Pixel reading checks construction against the independently expressed
        # mask-transport algorithm. It does not supply the label.
        if not np.array_equal(solve(grid), output):
            raise AssertionError("point construction and input-only transport disagree")
        return {'input': grid, 'output': output}
    raise RuntimeError("could not construct the requested diagnostic stencil transfer")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError("counts must be integers in the supported ranges")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    atlas, base_bank = _atlas(rng)
    # These construction-local banks only save repeated crop-uniqueness work.
    # They are not visible to solve, which always re-reads its input pixels.
    banks = []
    for i in range(8):
        bank = []
        for window in base_bank:
            r, c = _origin(window, atlas.shape, i)
            h, w = window[2:]
            if i not in (0, 2, 4, 5):
                h, w = w, h
            bank.append((r, c, h, w))
        banks.append(bank)
    colors = rng.permutation(np.arange(1, 10))
    frame = int(colors[0])
    palette = tuple(map(int, colors[1:3]))
    varied_palette = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in (('train', nt), ('test', nq)):
        for index in range(total):
            count = (1 if index == 0 else 2) if split == 'train' else 3
            for _ in range(10):
                pair = _pair(rng, atlas, banks, palette, frame, varied_palette, count,
                             shifted=split == 'test', diagnostic=split == 'train' and index == 1)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, references, frame_colors = set(), set(), set()
    cancellation_train = clipped_train = orientation_train = False
    centered_train = bounded_train = False
    query_counts = {'train': set(), 'test': set()}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError("pair must have exactly input and output")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            if _key(inp) in seen:
                raise ValueError("duplicate input")
            seen.add(_key(inp))
            a = _analyze(inp)
            scene = a['scene']
            before = scene['before']['grid']
            if not np.array_equal(out, solve(inp)):
                raise ValueError("incorrect output")
            if not a['parity'].any() or np.array_equal(a['parity'], a['difference']):
                raise ValueError("transfer must change the map and differ from copying the demo")
            if any(mask.sum() < 3 for mask in a['masks']):
                raise ValueError("every query must make a substantial contribution")
            if len({_key(m) for m in a['masks']}) != len(a['masks']):
                raise ValueError("query effects must be distinct")
            if len(a['masks']) > 1 and np.array_equal(a['union'], a['parity']):
                raise ValueError("multi-query cases must expose overlap cancellation")
            # Omitting the demonstrated stencil and complementing each crop
            # rectangle instead must be wrong in the teaching examples.
            rectangle_parity = np.zeros(before.shape, bool)
            for query, match in zip(scene['queries'], a['matches']):
                i, _, r, c = match
                local = np.zeros(_transform(before, i).shape, bool)
                h, w = query['grid'].shape
                local[r:r + h, c:c + w] = True
                rectangle_parity ^= _transform(local, _INVERSE[i])
            if np.array_equal(rectangle_parity, a['parity']):
                raise ValueError("crop rectangle shortcut happens to solve this pair")
            references.add(_canonical(_binary(before)))
            frame_colors.add(scene['frame'])
            query_counts[split].add(len(scene['queries']))
            if split == 'train':
                cancellation_train |= not np.array_equal(a['union'], a['parity'])
                clipped_train |= any(a['losses'])
                orientation_train |= any(m[0] != a['demo'][0] for m in a['matches'])
                centered, bounded = _anchor_alternatives(a)
                centered_train |= not np.array_equal(centered, a['parity'])
                bounded_train |= not np.array_equal(bounded, a['parity'])
    if len(references) != 1 or len(frame_colors) != 1:
        raise ValueError("one puzzle must share its atlas and ring color")
    if query_counts != {'train': {1, 2}, 'test': {3}}:
        raise ValueError("missing demonstration/query-count shift")
    if not (cancellation_train and clipped_train and orientation_train and centered_train and bounded_train):
        raise ValueError("training must teach cancellation, clipping, local axes and crop anchoring")
    first = _analyze(_grid(puzzle['train'][0]['input']))
    if len(first['masks']) != 1 or any(first['losses']):
        raise ValueError("first training transfer must show a whole stencil")


def _signature(puzzle):
    return tuple((s, _key(p['input']), _key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed corpus plus semantic perturbations and count checks."""
    references, shapes, crop_shapes, layouts, palettes, stencil_shapes = (set() for _ in range(6))
    orientations, query_counts, stencil_sizes, losses = set(), set(), set(), set()
    signatures = set()
    examples = []
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        examples.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _analyze(pair['input'])
                scene = a['scene']
                references.add(_canonical(_binary(scene['before']['grid'])))
                shapes.add(scene['before']['grid'].shape)
                for panel in [scene['calibration']] + scene['queries']:
                    crop_shapes.add(panel['grid'].shape)
                    palettes.add(tuple(map(int, np.unique(panel['grid']))))
                layouts.add(tuple(sorted(p['box'] for p in [scene['before'], scene['after'], scene['calibration']] + scene['queries'])))
                offsets = a['offsets'] - a['offsets'].min(axis=0)
                stencil_shapes.add(tuple(map(tuple, offsets)))
                stencil_sizes.add(len(offsets))
                orientations.update(m[0] for m in [a['demo']] + a['matches'])
                query_counts.add(len(a['masks']))
                losses.update(a['losses'])
    assert len(signatures) == len(references) == 42
    assert len(shapes) >= 12 and len(crop_shapes) >= 12
    assert len(layouts) >= 180 and len(palettes) >= 25
    assert len(stencil_shapes) >= 70 and len(stencil_sizes) >= 12
    assert orientations == set(range(8)) and query_counts == {1, 2, 3}
    assert 0 in losses and max(losses) >= 5
    assert _signature(examples[0]) == _signature(generate_puzzle(0))
    for nt, nq in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(9000 + nt * 10 + nq, nt, nq)
        assert (len(p['train']), len(p['test'])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in examples[0][s]] for s in examples[0]}
    bad['test'][0]['output'][0, 0] = 0
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    _semantic_diagnostics(examples[0])
    return {'seeds': 42, 'references': len(references), 'map_shapes': len(shapes),
            'crop_shapes': len(crop_shapes), 'layouts': len(layouts),
            'stencil_shapes': len(stencil_shapes), 'stencil_sizes': len(stencil_sizes)}


def _semantic_diagnostics(puzzle):
    # Keep every source/query/calibration pixel fixed; change ONLY the worked
    # edited-map interior to demonstrate another legitimate stencil.
    pair = puzzle['train'][0]
    inp = pair['input']
    a = _analyze(inp)
    scene = a['scene']
    before = scene['before']['grid']
    colors = np.unique(before)
    r, c, h, w = scene['after']['box']
    changed = inp.copy()
    witnessed = False
    for cell in map(tuple, np.argwhere(~a['difference'])):
        edited = scene['after']['grid'].copy()
        edited[cell] = colors[0] if edited[cell] == colors[1] else colors[1]
        changed[r + 2:r + h - 2, c + 2:c + w - 2] = edited
        if len(np.unique(edited)) == 2 and not np.array_equal(solve(changed), solve(inp)):
            witnessed = True
            break
    assert witnessed, "worked pair must affect answer"
    # Identical colors and crop dimensions do not fix a spatial address.
    atlas = _binary(before)
    bank = _bank(atlas)
    query = scene['queries'][0]
    qr, qc, qh, qw = query['box']
    old = solve(inp)
    witnessed = False
    for window in bank:
        for t in range(8):
            chip = _chip(atlas, window, t, np.unique(query['grid']))
            if chip.shape != query['grid'].shape:
                continue
            variant = inp.copy()
            variant[qr + 1:qr + qh - 1, qc + 1:qc + qw - 1] = chip
            if not np.array_equal(old, solve(variant)):
                witnessed = True
                break
        if witnessed:
            break
    assert witnessed, "crop recognition must affect the answer"
    # Repeated patches give conflicting coordinate addresses; reject them.
    checkerboard = (np.indices((7, 9)).sum(axis=0) % 2).astype(np.uint8)
    orbit = [_transform(checkerboard, i) for i in range(8)]
    assert len(_matches(orbit, checkerboard[:2, :2])) > 1
    try:
        _unique_match(orbit, checkerboard[:2, :2] + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous crop was accepted")
    centered, bounded = _anchor_alternatives(a)
    assert not np.array_equal(centered, a['parity'])
    assert not np.array_equal(bounded, a['parity'])
    # Panel locations and literal nonzero color identities carry no semantics.
    maps = [_framed(scene['before']['grid'], 1, scene['frame']),
            _framed(scene['after']['grid'], 2, scene['frame'])]
    chips = [_framed(scene['calibration']['grid'], 2, scene['frame'])]
    chips += [_framed(q['grid'], 1, scene['frame']) for q in scene['queries']]
    for seed in range(3):
        rearranged = _layout(np.random.default_rng(seed), maps, chips)
        assert np.array_equal(solve(rearranged), solve(inp))
    lut = np.r_[0, np.random.default_rng(95).permutation(np.arange(1, 10))]
    assert np.array_equal(solve(lut[inp]), lut[solve(inp)])
    # Transposition is valid even when calibration/query sizes differ.
    assert np.array_equal(solve(inp.T), solve(inp).T)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
