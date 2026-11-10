"""A boundary-clocked stencil printer with a separate program in every lane.

The input alone supplies the dictionary, clock direction, marker color, and
special-tick color (the printing chamber's frame). No puzzle state is consulted.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grids must be two dimensional, with sides 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid cells must be integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    left = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while left:
        root = min(left)
        left.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _D4:
                p = (r + dr, c + dc)
                if p in left:
                    left.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _frames(grid):
    """Structural colors form isolated rectangular rings in this new family."""
    result = []
    for color in np.unique(grid):
        for cells in _components(grid == color):
            rr, cc = np.array(cells).T
            r, c, h, w = int(rr.min()), int(cc.min()), int(np.ptp(rr)) + 1, int(np.ptp(cc)) + 1
            if h < 5 or w < 5 or len(cells) != 2 * h + 2 * w - 4:
                continue
            if not all(y in (r, r + h - 1) or x in (c, c + w - 1) for y, x in cells):
                continue
            panel = grid[r:r + h, c:c + w].copy()
            values, counts = np.unique(panel[1:-1, 1:-1], return_counts=True)
            if counts.max() <= (h - 2) * (w - 2) / 2:
                continue
            result.append(dict(box=(r, c, h, w), panel=panel, fill=int(values[np.argmax(counts)]), frame=int(color)))
    return result


def _dictionary(panel, fill):
    inner = panel[1:-1, 1:-1]
    # A clear fill rim makes complete examples distinct from a clock/program.
    assert np.all(inner[0] == fill) and np.all(inner[-1] == fill)
    assert np.all(inner[:, 0] == fill) and np.all(inner[:, -1] == fill)
    templates, repetitions = {}, defaultdict(int)
    for cells in _components(inner != fill):
        assert 3 <= len(cells) <= 20, "a complete connected template has at least three cells"
        colors, counts = np.unique([inner[r, c] for r, c in cells], return_counts=True)
        singleton = colors[counts == 1]
        assert len(singleton) == 1, "the key is the only singleton color in an occurrence"
        key = int(singleton[0])
        ar, ac = next((r, c) for r, c in cells if inner[r, c] == key)
        sig = tuple(sorted((r - ar, c - ac, int(inner[r, c])) for r, c in cells))
        assert key not in templates or templates[key] == sig, "conflicting dictionary entries"
        templates[key] = sig
        repetitions[key] += 1
    assert 1 <= len(templates) <= 3
    assert all(color not in templates for sig in templates.values()
               for dr, dc, color in sig if (dr, dc) != (0, 0))
    return templates, dict(repetitions)


def _bar(inner, color):
    points = np.argwhere(inner == color)
    if not len(points):
        return None
    rr, cc = points.T
    candidates = []
    for turn, fixed, edge, lateral in ((0, rr, 0, cc), (1, cc, inner.shape[1] - 1, rr),
                                      (2, rr, inner.shape[0] - 1, cc), (3, cc, 0, rr)):
        if np.all(fixed == edge):
            start = int(lateral.min())
            if np.array_equal(np.sort(lateral), np.arange(start, start + len(points))):
                candidates.append(turn)
    return candidates[0] if len(candidates) == 1 else None


def _interpret(input_grid):
    a = _grid(input_grid)
    panels = _frames(a)
    assert len(panels) == 2, "exactly two isolated complete frames"
    interpretations = []
    for ref in panels:
        target = next(p for p in panels if p is not ref)
        try:
            raw_templates, _ = _dictionary(ref['panel'], ref['fill'])
            ti = target['panel'][1:-1, 1:-1]
            extras = set(map(int, np.unique(ti))) - {target['fill']} - set(raw_templates)
            assert len(extras) == 1
            marker = next(iter(extras))
            assert marker not in ref['panel'] and marker != target['frame']
            turn = _bar(ti, marker)
            assert turn is not None, "one edge bar, away from an ambiguous singleton corner"
            cp = np.rot90(target['panel'], turn).copy()
            cr = np.rot90(ref['panel'], turn).copy()
            templates, repetitions = _dictionary(cr, ref['fill'])
            inner = cp[1:-1, 1:-1]
            columns = np.flatnonzero(inner[0] == marker)
            start, length = int(columns[0]), len(columns)
            assert np.all(inner[1:] != marker)
            assert np.all(inner[0, :start] == target['fill']) and np.all(inner[0, start + length:] == target['fill'])
            commands = defaultdict(list)
            mask = (inner != target['fill']) & (inner != marker)
            components = _components(mask)
            assert components and all(len(c) == 1 for c in components), "isolated program keys"
            for cells in components:
                d, col = cells[0]
                key = int(inner[d, col])
                assert d > 0 and key in templates
                commands[col].append((d, key))
            assert 1 <= len(commands) <= 3
            commands = {col: sorted(cmds) for col, cmds in sorted(commands.items())}
            assert all(all(b[0] - x[0] >= 2 for x, b in zip(cmds, cmds[1:])) for cmds in commands.values())
            all_dr = [dr for sig in templates.values() for dr, _, _ in sig]
            assert max(all_dr) - min(all_dr) <= 4, "successive six-cell-spaced stamps must be separated"
            assert min(all_dr) >= -4 and max(all_dr) <= 4
            intervals = []
            for col, cmds in commands.items():
                offsets = [dc for _, key in cmds for _, dc, _ in templates[key]]
                lo, hi = col + min(offsets), col + max(offsets)
                assert 0 <= lo <= hi < inner.shape[1], "no lateral clipping"
                assert hi < start - 1 or lo > start + length, "stencils and clock have a fill gap"
                intervals.append((lo, hi))
            assert all(b[0] > x[1] + 1 for x, b in zip(sorted(intervals), sorted(intervals)[1:])), "lanes have separate stencil envelopes"
            # Structural roles cannot disappear into fill or masquerade as keys.
            motifs = {color for sig in templates.values() for _, _, color in sig}
            assert motifs.isdisjoint({ref['fill'], target['fill'], ref['frame'], target['frame'], marker})
            assert target['fill'] != target['frame'] and ref['fill'] != ref['frame']
            # Outside is the chamber fill, with a visible margin around both panels.
            exterior = np.ones(a.shape, dtype=bool)
            for p in panels:
                r, c, h, w = p['box']
                assert r >= 1 and c >= 1 and r + h < a.shape[0] and c + w < a.shape[1]
                exterior[r:r + h, c:c + w] = False
            assert np.all(a[exterior] == target['fill'])
            interpretations.append(dict(target=cp, reference=cr, fill=target['fill'], reference_fill=ref['fill'],
                                        highlight=target['frame'], marker=marker, start=start, length=length,
                                        turn=turn, templates=templates, repetitions=repetitions, commands=commands,
                                        target_box=target['box'], reference_box=ref['box']))
        except (AssertionError, StopIteration):
            continue
    assert len(interpretations) == 1, "input must have one dictionary/program interpretation"
    return interpretations[0]


def _render(info, choice='latest', ticks=True):
    """Event simulation. Commands update memory before an equal-depth tick."""
    panel = info['target']
    out = np.full(panel.shape, info['highlight'], dtype=np.int8)
    inner = out[1:-1, 1:-1]
    inner[:] = info['fill']
    depth, width = inner.shape
    start, length = info['start'], info['length']
    if ticks:
        for d in range(0, depth, 2):
            inner[d, start:start + length] = info['highlight'] if d % 6 == 4 else info['marker']
    for col, cmds in info['commands'].items():
        if choice == 'at_commands':
            events = list(cmds)
        else:
            events = []
            active, j = None, 0
            for d in range(4, depth, 6):
                while j < len(cmds) and cmds[j][0] <= d:
                    active = cmds[j][1]
                    j += 1
                if active is None:
                    continue
                if choice == 'first':
                    key = cmds[0][1]
                elif choice == 'nearest':
                    key = min(cmds, key=lambda p: (abs(p[0] - d), p[0]))[1]
                else:
                    key = active
                events.append((d, key))
        for d, key in events:
            for dr, dc, color in info['templates'][key]:
                r, c = d + dr, col + dc
                if 0 <= r < depth and 0 <= c < width:
                    inner[r, c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _interpret(input_grid)
    return np.rot90(_render(info), -info['turn']).copy()


def _bounds(sig):
    rr = [r for r, _, _ in sig]
    cc = [c for _, c, _ in sig]
    return min(rr), min(cc), max(rr) - min(rr) + 1, max(cc) - min(cc) + 1


def _make_templates(rng, count, keys, decorations, small, teach):
    height = int(rng.integers(2, 4 if small else 6))
    anchor_row = int(rng.integers(height))
    result = {}
    for key in keys[:count]:
        for _ in range(80):
            width = int(rng.integers(2, 4 if small else 6))
            ac = int(rng.integers(width))
            n = int(rng.integers(4 if teach else 3, min(14, height * width) + 1))
            occupied = {(anchor_row, ac)}
            while len(occupied) < n:
                frontier = sorted({(r + dr, c + dc) for r, c in occupied for dr, dc in _D4
                                   if 0 <= r + dr < height and 0 <= c + dc < width} - occupied)
                weights = np.array([1.0 / sum((r + dr, c + dc) in occupied for dr, dc in _D4) for r, c in frontier])
                idx = int(rng.choice(len(frontier), p=weights / weights.sum()))
                occupied.add(frontier[idx])
            offsets = sorted((r - anchor_row, c - ac) for r, c in occupied if (r, c) != (anchor_row, ac))
            shape = {(0, 0), *offsets}
            rr, cc = zip(*shape)
            nonsolid = len(shape) < (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1)
            asym = shape != {(r, -c) for r, c in shape} and shape != {(-c, r) for r, c in shape}
            if teach and not (nonsolid and asym):
                continue
            nd = 2 if len(offsets) >= 4 and len(decorations) > 1 and rng.random() < .45 else 1
            colors = [int(c) for c in rng.choice(decorations, nd, replace=False)]
            assigned = colors * 2 + [int(rng.choice(colors)) for _ in range(len(offsets) - 2 * nd)]
            if nd == 1:
                assigned = colors * len(offsets)
            rng.shuffle(assigned)
            result[int(key)] = tuple(sorted([(0, 0, int(key))] + [(r, c, color) for (r, c), color in zip(offsets, assigned)]))
            break
        else:
            return None
    return result


def _shelf(templates, ids, width):
    r, c, row_height = 2, 2, 0
    places = []
    for key in ids:
        lr, lc, h, w = _bounds(templates[key])
        if w + 4 > width:
            return None
        if c + w > width - 2:
            r += row_height + 1
            c, row_height = 2, 0
        places.append((key, r - lr, c - lc))
        c += w + 1
        row_height = max(row_height, h)
    return r + row_height + 2, places


def _program(rng, depth, keys, lane_count, teaching):
    result = []
    if teaching == 0:
        # The future switch is closer to tick 10 than the old command is.
        result.append([(int(rng.integers(1, 4)), keys[0]), (int(rng.integers(12, 15)), keys[1])])
    elif teaching == 1:
        result.append([(int(rng.integers(1, 4)), keys[0]), (10, keys[1])])
        result.append([(int(rng.integers(6, 10)), keys[1])])
    else:
        for _ in range(lane_count):
            candidates = np.arange(1, depth, 2)  # a random common parity per lane
            if rng.random() < .5:
                candidates = np.arange(2, depth, 2)
            n = int(rng.integers(1, min(4, len(candidates)) + 1))
            ds = sorted(map(int, rng.choice(candidates, n, replace=False)))
            result.append([(d, int(rng.choice(keys))) for d in ds])
    rng.shuffle(result)
    return result


def _target_geometry(rng, templates, programs, depth):
    lanes = []
    for cmds in programs:
        offsets = [dc for _, key in cmds for _, dc, _ in templates[key]]
        lo, hi = min(offsets), max(offsets)
        lanes.append((lo, hi))
    bar_length = int(rng.integers(1, 5))
    # Arrange the clock among the independent lanes, not always at one side.
    insertion = int(rng.integers(len(lanes) + 1))
    cursor = int(rng.integers(0, 3))
    commands, bar_start = {}, None
    for index in range(len(lanes) + 1):
        if index == insertion:
            bar_start = cursor
            cursor += bar_length + int(rng.integers(1, 3))
        if index < len(lanes):
            lo, hi = lanes[index]
            col = cursor - lo
            commands[col] = programs[index]
            cursor += hi - lo + 1 + int(rng.integers(1, 4))
    width = cursor - 1 + int(rng.integers(0, 3))
    if width > 28 or width < 5 or (bar_length == 1 and bar_start in (0, width - 1)):
        return None
    return depth + 2, width + 2, bar_start, bar_length, commands


def _scene_geometry(rng, target_shape, templates, ids):
    th, tw = target_shape
    candidates = []
    for rw in range(6, 29):
        packed = _shelf(templates, ids, rw)
        if packed is None:
            continue
        rh, places = packed
        if rh > 28:
            continue
        if (rh - 2) * (rw - 2) <= 2 * sum(len(templates[k]) for k in ids):
            continue
        for axis in (0, 1):
            if ((rh + th if axis == 0 else rw + tw) + 3 <= 30
                    and max(rw, tw) + 2 <= 30 and max(rh, th) + 2 <= 30):
                candidates.append((rh, rw, places, axis))
    if not candidates:
        return None
    rh, rw, places, axis = candidates[int(rng.integers(len(candidates)))]
    shapes = [(rh, rw), (th, tw)]
    gapmax = min(3, 28 - shapes[0][axis] - shapes[1][axis])
    gap = int(rng.integers(1, gapmax + 1))
    minimum = [max(rh, th) + 2, max(rw, tw) + 2]
    minimum[axis] = shapes[0][axis] + shapes[1][axis] + gap + 2
    canvas = [int(rng.integers(m, min(30, m + 3) + 1)) for m in minimum]
    positions = [[0, 0], [0, 0]]
    order = rng.permutation(2).tolist()
    initial = int(rng.integers(1, canvas[axis] - shapes[0][axis] - shapes[1][axis] - gap))
    positions[order[0]][axis] = initial
    positions[order[1]][axis] = initial + shapes[order[0]][axis] + gap
    for i in (0, 1):
        positions[i][1 - axis] = int(rng.integers(1, canvas[1 - axis] - shapes[i][1 - axis]))
    return canvas, [tuple(positions[i] + list(shapes[i])) for i in (0, 1)], places


def _pair(rng, rotation, teaching=None):
    for attempt in range(160):
        count = 2 if teaching is not None else int(rng.integers(1, 4))
        lanes = (1 if teaching == 0 else 2) if teaching is not None else int(rng.integers(1, 4))
        small = teaching is not None or lanes == 3 or attempt >= 100
        palette = list(map(int, rng.permutation(10)))
        fill, ref_fill, marker, frame = palette[:4]
        if teaching is None and rng.random() < .35:
            ref_fill = fill
        ref_frame = frame if rng.random() < .6 else palette[4]
        keys = palette[5:5 + count]
        decorations = palette[8:10] if count == 3 else palette[5 + count:10]
        templates = _make_templates(rng, count, keys, decorations, small, teaching is not None)
        if templates is None:
            continue
        depth = int(rng.integers(17, 21)) if teaching is not None else int(rng.integers(7, 26))
        programs = _program(rng, depth, keys, lanes, teaching)
        tg = _target_geometry(rng, templates, programs, depth)
        if tg is None:
            continue
        th, tw, start, length, commands = tg
        ids = [key for key in keys for _ in range(2 if teaching == 0 else int(rng.integers(1, 4)))]
        rng.shuffle(ids)
        geo = _scene_geometry(rng, (th, tw), templates, ids)
        if geo is None:
            continue
        canvas, boxes, places = geo
        inp = np.full(canvas, fill, dtype=np.int8)
        for box, border, interior_fill in zip(boxes, (ref_frame, frame), (ref_fill, fill)):
            r, c, h, w = box
            inp[r:r + h, c:c + w] = border
            inp[r + 1:r + h - 1, c + 1:c + w - 1] = interior_fill
        rr, rc, rh, rw = boxes[0]
        ref = inp[rr:rr + rh, rc:rc + rw]
        for key, ar, ac in places:
            for dr, dc, color in templates[key]:
                ref[ar + dr, ac + dc] = color
        tr, tc, _, _ = boxes[1]
        target = inp[tr:tr + th, tc:tc + tw]
        ti = target[1:-1, 1:-1]
        ti[0, start:start + length] = marker
        for col, cmds in commands.items():
            for d, key in cmds:
                ti[d, col] = key
        model = dict(target=target, fill=fill, marker=marker, highlight=frame,
                     start=start, length=length, templates=templates, commands=commands)
        out = _render(model)
        if not np.any(np.isin(out[1:-1, 1:-1], keys)):
            continue
        pair = dict(input=np.rot90(inp, -rotation).copy(), output=np.rot90(out, -rotation).copy())
        try:
            info = _interpret(pair['input'])
            if np.array_equal(pair['output'], np.rot90(_render(info), -info['turn'])):
                return pair
        except AssertionError:
            continue
    return _fallback_pair(rng, rotation, teaching)


def _fallback_pair(rng, rotation, teaching=None):
    """A guaranteed fitting construction after bounded natural proposals.

    Six four-cell stencils fit one reference shelf. Three lane envelopes fit
    the chamber; stacking both panels takes exactly 29 rows including margins.
    This preserves the causal witnesses rather than weakening the validator.
    """
    colors = list(map(int, rng.permutation(10)))
    fill, ref_fill, marker, frame, ref_frame = colors[:5]
    count = 2 if teaching is not None else int(rng.integers(1, 4))
    lanes = (1 if teaching == 0 else 2) if teaching is not None else int(rng.integers(1, 4))
    keys, decorations = colors[5:5 + count], colors[8:10]
    templates = {}
    for i, key in enumerate(keys):
        offsets = ((0, 1), (1, 1), (1, 2)) if i % 2 == 0 else ((1, 0), (1, 1), (1, 2))
        templates[key] = tuple(sorted([(0, 0, key)] + [(r, c, decorations[i % 2]) for r, c in offsets]))
    depth = 18
    if teaching is not None:
        programs = _program(rng, depth, keys, lanes, teaching)
    else:
        programs = [[(int(rng.integers(1, 4)), int(rng.choice(keys)))] for _ in range(lanes)]
    start, length = 1, 2
    commands = {4 + 4 * i: cmds for i, cmds in enumerate(programs)}
    inner_width = 4 * lanes + 4
    ids = keys * 2
    rng.shuffle(ids)
    ref_width = 4 * len(ids) + 3
    canvas_width = max(ref_width, inner_width + 2) + 2
    inp = np.full((29, canvas_width), fill, dtype=np.int8)
    target_first = bool(rng.integers(2))
    reference_top, target_top = (22, 1) if target_first else (1, 8)
    reference_left = int(rng.integers(1, canvas_width - ref_width))
    target_left = int(rng.integers(1, canvas_width - inner_width - 2))
    ref = inp[reference_top:reference_top + 6, reference_left:reference_left + ref_width]
    ref[:] = ref_frame
    ref[1:-1, 1:-1] = ref_fill
    for i, key in enumerate(ids):
        for dr, dc, color in templates[key]:
            ref[2 + dr, 2 + 4 * i + dc] = color
    target = inp[target_top:target_top + depth + 2, target_left:target_left + inner_width + 2]
    target[:] = frame
    ti = target[1:-1, 1:-1]
    ti[:] = fill
    ti[0, start:start + length] = marker
    for col, cmds in commands.items():
        for d, key in cmds:
            ti[d, col] = key
    model = dict(target=target, fill=fill, marker=marker, highlight=frame,
                 start=start, length=length, templates=templates, commands=commands)
    out = _render(model)
    pair = dict(input=np.rot90(inp, -rotation).copy(), output=np.rot90(out, -rotation).copy())
    info = _interpret(pair['input'])
    assert np.array_equal(np.rot90(_render(info), -info['turn']), pair['output'])
    return pair


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base = int(rng.integers(4))
    shift = int(rng.choice([1, 2, 3]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            rotation = base if split == 'train' else (base + shift) % 4
            teaching = index if split == 'train' and index < 2 else None
            for _ in range(20):
                pair = _pair(rng, rotation, teaching)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _cell_prediction(info):
    """Coordinate check of the intended rule, rather than replaying events."""
    panel = info['target']
    h, w = panel.shape[0] - 2, panel.shape[1] - 2
    paint = {}
    clipped = 0
    events = []
    for col, cmds in info['commands'].items():
        for tick in range(4, h, 6):
            previous = [(d, key) for d, key in cmds if d <= tick]
            if not previous:
                continue
            key = max(previous)[1]
            events.append((tick, col, key))
            for dr, dc, color in info['templates'][key]:
                r, c = tick + dr, col + dc
                if 0 <= r < h and 0 <= c < w:
                    assert (r, c) not in paint, "overlapping printed stencils"
                    paint[(r, c)] = color
                else:
                    clipped += 1
    out = np.empty_like(panel)
    for r in range(panel.shape[0]):
        for c in range(panel.shape[1]):
            if r in (0, panel.shape[0] - 1) or c in (0, panel.shape[1] - 1):
                color = info['highlight']
            else:
                d, col = r - 1, c - 1
                color = paint.get((d, col), info['fill'])
                if info['start'] <= col < info['start'] + info['length']:
                    color = (info['fill'] if d % 2 else info['highlight'] if d % 6 == 4 else info['marker'])
            out[r, c] = color
    return out, events, clipped


def _evidence(info):
    expected, events, clipped = _cell_prediction(info)
    printed = {key for _, _, key in events}
    hold = switch = late = inclusive = False
    for col, cmds in info['commands'].items():
        seq = [(d, k) for d, c, k in events if c == col]
        hold |= any(a[1] == b[1] and not any(a[0] < d <= b[0] for d, _ in cmds) for a, b in zip(seq, seq[1:]))
        switch |= len({k for _, k in seq}) > 1
        late |= bool(seq and cmds[0][0] > 4)
        inclusive |= any(d == t and k == key and i > 0 and cmds[i - 1][1] != key
                         for i, (d, key) in enumerate(cmds) for t, k in seq)
    signatures = list(info['templates'].values())
    nonsolid = any(len(sig) < _bounds(sig)[2] * _bounds(sig)[3] for sig in signatures)
    return dict(hold=hold, switch=switch, late=late, inclusive=inclusive, clipped=clipped,
                printed_keys=len(printed), transparent=info['fill'] != info['reference_fill'] and nonsolid,
                against_first=not np.array_equal(expected, _render(info, 'first')),
                against_nearest=not np.array_equal(expected, _render(info, 'nearest')),
                against_A_only=not np.array_equal(expected, _render(dict(info, commands={}))),
                against_B_only=not np.array_equal(expected, _render(info, 'at_commands', ticks=False)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, train_sides, test_sides = set(), [], set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, output = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _interpret(inp)
            predicted, events, _ = _cell_prediction(info)
            assert events, "at least one motif is printed"
            assert np.array_equal(output, np.rot90(predicted, -info['turn'])), "incorrect clock, memory, stencil, clipping, or crop"
            assert np.array_equal(solve(inp), output)
            if split == 'train':
                evidence.append(_evidence(info))
                train_sides.add(info['turn'])
            else:
                test_sides.add(info['turn'])
    for name in ('hold', 'switch', 'late', 'inclusive', 'transparent', 'against_first', 'against_nearest', 'against_A_only', 'against_B_only'):
        assert any(e[name] for e in evidence), f"training lacks {name} evidence"
    assert any(e['printed_keys'] >= 2 for e in evidence), "teach key-specific selection"
    assert train_sides.isdisjoint(test_sides), "the generator's test shift changes the sweep side"


def self_test():
    """Bounded 42-seed measurements plus direction/count/corruption regressions."""
    records, puzzles = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _interpret(pair['input'])
                _, events, clipped = _cell_prediction(info)
                records.append(dict(split=split, shape=pair['input'].shape, output_shape=pair['output'].shape,
                                    side=info['turn'], lanes=len(info['commands']), types=len(info['templates']),
                                    sizes=tuple(len(sig) for sig in info['templates'].values()),
                                    geometries=tuple(tuple((r, c) for r, c, _ in sig) for sig in info['templates'].values()),
                                    command_counts=tuple(len(cmds) for cmds in info['commands'].values()),
                                    repetitions=tuple(info['repetitions'].values()), clipped=clipped,
                                    bar_length=info['length'], events=len(events),
                                    colors=tuple(map(int, np.unique(pair['input']))),
                                    equal_fill=info['fill'] == info['reference_fill']))
    assert len(puzzles) == 42
    assert len({r['shape'] for r in records}) >= 30
    assert len({r['output_shape'] for r in records}) >= 30
    assert len({g for r in records for g in r['geometries']}) >= 80
    assert {r['lanes'] for r in records} == {1, 2, 3}
    assert {r['types'] for r in records} == {1, 2, 3}
    assert {n for r in records for n in r['command_counts']} >= {1, 2, 3, 4}
    assert {n for r in records for n in r['repetitions']} >= {1, 2, 3}
    assert {r['bar_length'] for r in records} >= {1, 2, 3, 4}
    assert {r['equal_fill'] for r in records} == {False, True}
    assert any(r['clipped'] for r in records)
    assert max(n for r in records for n in r['sizes']) >= 10
    for split in ('train', 'test'):
        assert {r['side'] for r in records if r['split'] == split} == {0, 1, 2, 3}
    for rotation in range(4):
        fallback = {'train': [_fallback_pair(np.random.default_rng(900 + rotation), rotation, i) for i in (0, 1)],
                    'test': [_fallback_pair(np.random.default_rng(950 + rotation), (rotation + 1) % 4)]}
        validate_generated_puzzle(fallback)
    original = generate_puzzle(0, 2, 1)
    duplicate = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in original for a, b in zip(original[s], duplicate[s]) for k in ('input', 'output'))
    # The solver must be equivariant to whole-scene turns and reflections.
    for pair in original['train'] + original['test']:
        for turn in range(4):
            for reflect in (False, True):
                inp, out = pair['input'], pair['output']
                if reflect:
                    inp, out = inp[:, ::-1], out[:, ::-1]
                assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(100 + a * 10 + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt frame accepted")
    # A bad dictionary repetition must be caught from the input, not its label.
    info = _interpret(original['train'][0]['input'])
    ref = info['reference'].copy()
    cells = _components(ref[1:-1, 1:-1] != info['reference_fill'])
    r, c = cells[0][0]
    ref[r + 1, c + 1] = info['reference_fill']
    try:
        _dictionary(ref, info['reference_fill'])
    except AssertionError:
        pass
    else:
        raise AssertionError("conflicting or incomplete reference accepted")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(puzzles),
                input_shapes=len({r['shape'] for r in records}), output_shapes=len({r['output_shape'] for r in records}),
                template_geometries=len({g for r in records for g in r['geometries']}),
                lane_counts=sorted({r['lanes'] for r in records}), key_counts=sorted({r['types'] for r in records}),
                clipped_pairs=sum(bool(r['clipped']) for r in records))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
