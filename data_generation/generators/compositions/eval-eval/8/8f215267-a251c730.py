"""Count-powered growth: component budgets reveal rooted template graph balls.

Only NumPy and the Python standard library are used. solve reads visible cells;
construction records are never passed to it. Run this file for bounded self-tests.
"""
from collections import deque
from itertools import permutations
import json
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_), 'integer cells required')
    _require(np.all((0 <= a) & (a <= 9)), 'colors outside 0..9')
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    answer = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in D4:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        answer.append(tuple(sorted(cells)))
    return answer


def _bounds(cells):
    rr, cc = zip(*[(p[0], p[1]) for p in cells])
    return min(rr), min(cc), max(rr) - min(rr) + 1, max(cc) - min(cc) + 1


def _crop(a, box):
    r, c, h, w = box
    return a[r:r+h, c:c+w]


def _separate(groups):
    labels = {p: i for i, cells in enumerate(groups) for p in cells}
    _require(len(labels) == sum(map(len, groups)), 'overlapping groups')
    _require(all(labels.get((r+dr, c+dc), i) == i
                 for (r, c), i in labels.items() for dr, dc in D8), 'groups touch')


def _frames(a, bg):
    frames = []
    for color in sorted(set(map(int, a.flat)) - {bg}):
        for cells in _components(a == color):
            r, c, h, w = _bounds(cells)
            if min(h, w) < 5 or len(cells) != 2*h + 2*w - 4:
                continue
            if any(y not in (r, r+h-1) and x not in (c, c+w-1) for y, x in cells):
                continue
            inner = a[r+1:r+h-1, c+1:c+w-1]
            fill = int(inner[0, 0])
            if fill == color or not (np.all(inner[0] == fill) and np.all(inner[-1] == fill)
                                     and np.all(inner[:, 0] == fill) and np.all(inner[:, -1] == fill)):
                continue
            if np.count_nonzero(inner == fill) <= inner.size / 2:
                continue
            frames.append(dict(box=(r, c, h, w), frame=color, fill=fill,
                               components=_components(inner != fill)))
    # A fill region can accidentally outline a smaller rectangle. The two
    # scene panels are identified by their shared border color, not such a
    # singleton nested candidate.
    frames = [p for p in frames if sum(q['frame'] == p['frame'] for q in frames) >= 2]
    _require(len(frames) == 2, 'expected exactly two shared-color rectangular panels')
    _require(frames[0]['frame'] == frames[1]['frame'], 'panel borders must share a color')
    x, y = frames[0]['box'], frames[1]['box']
    _require(x[0]+x[2] < y[0] or y[0]+y[2] < x[0]
             or x[1]+x[3] < y[1] or y[1]+y[3] < x[1], 'panels need a moat')
    return frames


def _distances(template):
    occupied = {(r, c) for r, c, _ in template}
    _require((0, 0) in occupied, 'missing root')
    distances = {(0, 0): 0}
    queue = deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in D4:
            p = (r+dr, c+dc)
            if p in occupied and p not in distances:
                distances[p] = distances[(r, c)] + 1
                queue.append(p)
    _require(len(distances) == len(occupied), 'disconnected template')
    return distances


def _dictionary(a, reference, keys):
    inner = _crop(a, reference['box'])[1:-1, 1:-1]
    templates, repetitions = {}, {}
    _separate(reference['components'])
    for cells in reference['components']:
        _require(len(cells) >= 3, 'incomplete reference motif')
        roots = [(r, c, int(inner[r, c])) for r, c in cells if int(inner[r, c]) in keys]
        _require(len(roots) == 1, 'reference motif needs exactly one target identity')
        ar, ac, key = roots[0]
        signature = tuple(sorted((r-ar, c-ac, int(inner[r, c])) for r, c in cells))
        _require(key not in templates or signature == templates[key], 'conflicting reference templates')
        templates[key] = signature
        repetitions[key] = repetitions.get(key, 0) + 1
    _require(set(templates) == keys, 'missing reference identity')
    return templates, repetitions


def _interpret(input_grid):
    """Parse all roles, budgets and colored graphs using the input only."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg)
             and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), 'uniform exterior margin required')
    panels = _frames(a, bg)
    exterior_mask = np.ones(a.shape, dtype=bool)
    for p in panels:
        r, c, h, w = p['box']
        _require(r > 0 and c > 0 and r+h < a.shape[0] and c+w < a.shape[1], 'outer margin missing')
        exterior_mask[r:r+h, c:c+w] = False
    objects = []
    for color in sorted(set(map(int, a[exterior_mask])) - {bg}):
        for cells in _components(exterior_mask & (a == color)):
            objects.append(dict(color=color, cells=cells, area=len(cells)))
    _separate([o['cells'] for o in objects])
    for o in objects:
        _require(o['color'] != panels[0]['frame'], 'border color outside a panel')
        _require(1 <= o['area'] <= 9, 'external object area outside 1..9')
        for r, c in o['cells']:
            _require(all(exterior_mask[r+dr, c+dc] for dr, dc in D8), 'object touches panel')
    interpretations = []
    for ti, target in enumerate(panels):
        try:
            _require(target['components'] and all(len(cells) == 1 for cells in target['components']), 'target must contain only roots')
            _separate(target['components'])
            inner = _crop(a, target['box'])[1:-1, 1:-1]
            anchors = [(cells[0][0], cells[0][1], int(inner[cells[0]])) for cells in target['components']]
            keys = {k for _, _, k in anchors}
            _require(1 <= len(keys) <= 3, 'root type count')
            reference = panels[1-ti]
            _require(not keys & {bg, target['fill'], reference['fill'], target['frame']}, 'root/structural color collision')
            templates, repetitions = _dictionary(a, reference, keys)
            decorations = {v for k, t in templates.items() for r, c, v in t if (r, c) != (0, 0)}
            _require(not decorations & (keys | {bg, target['fill'], reference['fill'], target['frame']}), 'invisible or ambiguous decoration')
            distances = {key: _distances(t) for key, t in templates.items()}
            counts = {key: sum(o['color'] == key for o in objects) for key in keys}
            areas = {key: sum(o['area'] for o in objects if o['color'] == key) for key in keys}
            _require(all(n <= 5 for n in counts.values()) and sum(counts.values()) <= 9, 'fuel count bounds')
            _require(sum(o['color'] not in keys for o in objects) <= 2, 'too many unmatched objects')
            full_groups = []
            boxes = []
            for ar, ac, key in anchors:
                full = [(ar+r, ac+c) for r, c, _ in templates[key]]
                _require(all(1 <= r < inner.shape[0]-1 and 1 <= c < inner.shape[1]-1 for r, c in full), 'full footprint clips inner rim')
                full_groups.append(full)
                r, c, h, w = _bounds(full)
                boxes.append([(y, x) for y in range(r, r+h) for x in range(c, c+w)])
                _require(all(int(inner[r, c]) == (key if (r, c) == (ar, ac) else target['fill']) for r, c in full), 'stamp overwrites a root')
            _separate(boxes)
            _separate(full_groups)
            interpretations.append(dict(input=a, background=bg, reference=reference, target=target,
                templates=templates, repetitions=repetitions, anchors=anchors, distances=distances,
                counts=counts, areas=areas, objects=objects))
        except ValueError:
            continue
    _require(len(interpretations) == 1, 'ambiguous or invalid reference/target roles')
    return interpretations[0]


def _render(info, budgets=None, metric='graph', unary=False, solid=False):
    result = _crop(info['input'], info['target']['box']).copy()
    budgets = info['counts'] if budgets is None else budgets
    for ar, ac, key in info['anchors']:
        template = info['templates'][key]
        if solid:
            r0, c0, h, w = _bounds(template)
            # Copying a reference rectangle wrongly replaces transparent holes.
            for r in range(r0, r0+h):
                for c in range(c0, c0+w):
                    result[ar+r+1, ac+c+1] = info['reference']['fill']
        if unary:
            # A deliberately simpler A-like code for ablation, clipped to crop.
            for i in range(1, budgets[key]+1):
                c = ac+i+1
                if c < result.shape[1]-1:
                    result[ar+1, c] = key
            continue
        for dr, dc, color in template:
            distance = abs(dr)+abs(dc) if metric == 'manhattan' else info['distances'][key][dr, dc]
            if distance <= budgets[key]:
                result[ar+dr+1, ac+dc+1] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the target crop with one graph-growth wave per matching object."""
    return _render(_interpret(input_grid))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _normalize(cells):
    r0, c0, _, _ = _bounds(cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _polyomino(rng, area=None, compact=False):
    n = int(rng.integers(1, 10)) if area is None else area
    side = 3 if compact else int(rng.integers(3, 5))
    cells = {(int(rng.integers(side)), int(rng.integers(side)))}
    style = int(rng.integers(3))
    while len(cells) < n:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in D4
                           if 0 <= r+dr < side and 0 <= c+dc < side} - cells)
        neighbors = np.array([sum((r+dr, c+dc) in cells for dr, dc in D4) for r, c in frontier])
        weights = np.ones(len(frontier)) if style == 0 else (neighbors.astype(float)**2 if style == 1 else 1/neighbors)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))])
    return _normalize(cells)


def _rooted_shape(rng, kind=None, compact=False):
    if kind == 'branch':
        # Distance two must exist: a saturated one-step star would not
        # distinguish radius one from an area-derived larger budget.
        cells = {(0, 0), (-1, 0), (1, 0), (0, 1), (1, 1)}
        if not compact:
            cells |= {(r, 0) for r in range(-int(rng.integers(1, 4)), 0)}
            cells |= {(0, c) for c in range(1, int(rng.integers(2, 4)))}
    elif kind == 'detour' or rng.random() < .28:
        h = 3 if compact else int(rng.integers(3, 6))
        w = 3 if compact else int(rng.integers(3, 6))
        cells = {(r, 0) for r in range(h)} | {(h-1, c) for c in range(w)} | {(r, w-1) for r in range(h)}
        if kind != 'detour':
            ar, ac = _pick(rng, sorted(cells))
            cells = {(r-ar, c-ac) for r, c in cells}
    else:
        side = 3 if compact else int(rng.integers(3, 6))
        n = int(rng.integers(4, min(14, side*side)+1))
        cells = {(int(rng.integers(side)), int(rng.integers(side)))}
        while len(cells) < n:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in D4
                               if 0 <= r+dr < side and 0 <= c+dc < side} - cells)
            cells.add(_pick(rng, frontier))
        ar, ac = _pick(rng, sorted(cells))
        cells = {(r-ar, c-ac) for r, c in cells}
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    return tuple(sorted(cells))


def _packing_options(rng, shapes, padding, gap, panel=False, occupied_area=0):
    """Shelf boxes with variable row alignment, width, slack and ordering."""
    if not shapes:
        return [dict(h=1, w=1, places=[])]
    dims = [_bounds(s)[2:] for s in shapes]
    order = list(map(int, rng.permutation(len(shapes))))
    options = []
    for width in range(max(5 if panel else 1, max(w for h, w in dims)+2*padding), 29):
        shelves, shelf, x, sh = [], [], padding, 0
        for index in order:
            h, w = dims[index]
            if shelf and x+w > width-padding:
                shelves.append((shelf, sh))
                shelf, x, sh = [], padding, 0
            shelf.append((index, x))
            x += w+gap
            sh = max(sh, h)
        shelves.append((shelf, sh))
        height = 2*padding + sum(h for row, h in shelves) + gap*(len(shelves)-1)
        if panel:
            height = max(5, height)
            while (height-2)*(width-2) <= 2*occupied_area:
                height += 1
        if height > 28:
            continue
        base_places, y = [], padding
        for row, sh in shelves:
            for index, x in row:
                h, w = dims[index]
                r0, c0, _, _ = _bounds(shapes[index])
                base_places.append((index, y+int(rng.integers(sh-h+1))-r0, x-c0))
            y += sh+gap
        options.append(dict(h=height, w=width, places=base_places))
        # Keep a minimum-size option as well as independently expanded panels.
        extra = int(rng.integers(1, 4))
        if height+extra <= 28:
            dy = int(rng.integers(extra+1))
            options.append(dict(h=height+extra, w=width,
                                places=[(i, r+dy, c) for i, r, c in base_places]))
    return options


def _arrange(options, order, kind, gap):
    dims = [(p['h'], p['w']) for p in options]
    positions = [(0, 0)]*3
    a, b, c = order
    if kind == 0:  # Three vertically stacked regions.
        y = 0
        for i in order:
            positions[i] = (y, 0)
            y += dims[i][0]+gap
        h, w = y-gap, max(x[1] for x in dims)
    elif kind == 1:  # Two above the third.
        top = max(dims[a][0], dims[b][0])
        positions[a], positions[b], positions[c] = (0, 0), (0, dims[a][1]+gap), (top+gap, 0)
        h, w = top+gap+dims[c][0], max(dims[a][1]+gap+dims[b][1], dims[c][1])
    elif kind == 2:  # Three side by side.
        x = 0
        for i in order:
            positions[i] = (0, x)
            x += dims[i][1]+gap
        h, w = max(x[0] for x in dims), x-gap
    else:  # Two to the left of the third.
        left = max(dims[a][1], dims[b][1])
        positions[a], positions[b], positions[c] = (0, 0), (dims[a][0]+gap, 0), (0, left+gap)
        h, w = max(dims[a][0]+gap+dims[b][0], dims[c][0]), left+gap+dims[c][1]
    return h, w, positions


def _find_layout(rng, option_lists):
    orders = list(permutations(range(3)))
    # Exhaustively test dimensions by broadcasting, avoiding a slow Python
    # Cartesian-product search. At most 48^3 triples per arrangement.
    hh, ww = [], []
    for axis, options in enumerate(option_lists):
        shape = [1, 1, 1]
        shape[axis] = len(options)
        hh.append(np.array([p['h'] for p in options]).reshape(shape))
        ww.append(np.array([p['w'] for p in options]).reshape(shape))
    arrangements = [(order, kind, gap) for order in orders for kind in range(4) for gap in (1, 2)]
    rng.shuffle(arrangements)
    found = None
    for order, kind, gap in arrangements:
        a, b, c = order
        if kind == 0:
            heights, widths = hh[0]+hh[1]+hh[2]+2*gap, np.maximum(np.maximum(ww[0], ww[1]), ww[2])
        elif kind == 2:
            heights, widths = np.maximum(np.maximum(hh[0], hh[1]), hh[2]), ww[0]+ww[1]+ww[2]+2*gap
        elif kind == 1:
            heights = np.maximum(hh[a], hh[b])+gap+hh[c]
            widths = np.maximum(ww[a]+ww[b]+gap, ww[c])
        else:
            heights = np.maximum(hh[a]+hh[b]+gap, hh[c])
            widths = np.maximum(ww[a], ww[b])+gap+ww[c]
        feasible = np.argwhere((heights <= 28) & (widths <= 28))
        if len(feasible):
            indices = feasible[int(rng.integers(len(feasible)))]
            chosen = [o[int(i)] for o, i in zip(option_lists, indices)]
            h, w, positions = _arrange(chosen, order, kind, gap)
            found = True
            break
    if found is None:
        return None
    # Align shorter regions freely in the unused cross-axis space.
    positions = list(positions)
    if kind == 0:
        positions = [(r, int(rng.integers(w-p['w']+1))) for (r, c), p in zip(positions, chosen)]
    elif kind == 2:
        positions = [(int(rng.integers(h-p['h']+1)), c) for (r, c), p in zip(positions, chosen)]
    elif kind == 1:
        i = order[2]
        positions[i] = (positions[i][0], int(rng.integers(w-chosen[i]['w']+1)))
    else:
        i = order[2]
        positions[i] = (int(rng.integers(h-chosen[i]['h']+1)), positions[i][1])
    extra_h, extra_w = int(rng.integers(min(3, 28-h)+1)), int(rng.integers(min(3, 28-w)+1))
    top, left = 1+int(rng.integers(extra_h+1)), 1+int(rng.integers(extra_w+1))
    boxes = [(r+top, c+left, p['h'], p['w']) for (r, c), p in zip(positions, chosen)]
    return (h+extra_h+2, w+extra_w+2), boxes, chosen


def _make_pair(rng, keys, counts, teaching=None):
    simple = rng.random() < .25
    for attempt in range(12):
        compact = simple or attempt >= 8
        available = [c for c in range(10) if c not in keys]
        rng.shuffle(available)
        bg, border, fill1, fill2 = available[:4]
        rf = bg if rng.random() < .55 else fill1
        tf = (fill2 if rf == bg else bg) if teaching == 0 else _pick(rng, [bg, fill1, fill2])
        decor = [c for c in available if c not in (bg, border, rf, tf)]
        decor = list(map(int, rng.choice(decor, min(len(decor), int(rng.integers(1, 3))), replace=False)))
        templates = []
        for i, key in enumerate(keys):
            kind = ('branch' if i == 0 else 'detour') if teaching == 0 else None
            if teaching == 1 and i == 0 and compact:
                kind = 'detour'
            shape = _rooted_shape(rng, kind, compact)
            templates.append(tuple((r, c, key if (r, c) == (0, 0) else _pick(rng, decor)) for r, c in shape))
        ref_ids = list(range(len(keys)))
        for i in range(len(keys)):
            if rng.random() < .20:
                ref_ids.append(i)
        if teaching == 0:
            target_ids = [0, 0, 1]
        elif teaching == 1:
            target_ids = [0, 1, 1]
        else:
            target_ids = list(range(len(keys)))
            limit = 5 if len(keys) == 3 else 4
            extra = int(rng.integers(1 if len(keys) == 3 else 0, limit-len(keys)+1))
            target_ids += [int(rng.integers(len(keys))) for _ in range(extra)]
        rng.shuffle(ref_ids)
        rng.shuffle(target_ids)
        objects = []
        for i, (key, count) in enumerate(zip(keys, counts)):
            forced = [6] if teaching == 0 and i == 0 else [2, 3] if teaching == 0 else [2, 4] if teaching == 1 and i == 0 else None
            for j in range(count):
                objects.append((key, _polyomino(rng, None if forced is None else forced[j], compact)))
        nd = 1 if teaching is not None else int(rng.integers(3))
        objects += [(_pick(rng, decor), _polyomino(rng, compact=compact)) for _ in range(nd)]
        ref_shapes = [templates[i] for i in ref_ids]
        target_shapes = [templates[i] for i in target_ids]
        object_shapes = [s for color, s in objects]
        gap = 1 if compact else int(rng.integers(1, 3))
        options = [_packing_options(rng, ref_shapes, 2, gap, True, sum(map(len, ref_shapes))),
                   _packing_options(rng, target_shapes, 2, gap, True),
                   _packing_options(rng, object_shapes, 0, 1)]
        if any(not o for o in options):
            continue
        layout = _find_layout(rng, options)
        if layout is None:
            continue
        shape, boxes, packed = layout
        inp = np.full(shape, bg, dtype=np.int8)
        for box, fill in zip(boxes[:2], (rf, tf)):
            panel = _crop(inp, box)
            panel[:] = border
            panel[1:-1, 1:-1] = fill
        reference, target = _crop(inp, boxes[0]), _crop(inp, boxes[1])
        for j, ar, ac in packed[0]['places']:
            for dr, dc, color in templates[ref_ids[j]]:
                reference[ar+dr, ac+dc] = color
        for j, ar, ac in packed[1]['places']:
            target[ar, ac] = keys[target_ids[j]]
        for j, ar, ac in packed[2]['places']:
            color, cells = objects[j]
            rr, cc, _, _ = boxes[2]
            for dr, dc in cells:
                inp[rr+ar+dr, cc+ac+dc] = color
        out = target.copy()
        # Construct from the sampled counts; parsing later recovers those counts.
        for j, ar, ac in packed[1]['places']:
            k = target_ids[j]
            distances = _distances(templates[k])
            for dr, dc, color in templates[k]:
                if distances[dr, dc] <= counts[k]:
                    out[ar+dr, ac+dc] = color
        rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
        if reflect:
            inp, out = np.fliplr(inp).copy(), np.fliplr(out).copy()
        try:
            info = _interpret(inp)
            _require(np.array_equal(out, _render(info)), 'construction/parser mismatch')
            if teaching == 0:
                _require(_features(info)['detour_witness'] and _features(info)['branch_witness'], 'teaching topology lost')
                _require(max(info['distances'][keys[0]].values()) >= 2, 'first area witness saturated')
            elif teaching == 1:
                _require(max(info['distances'][keys[0]].values()) >= 2, 'second area witness has no second shell')
            return {'input': inp, 'output': out}
        except ValueError:
            continue
    raise RuntimeError('bounded construction failed')


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    nt, nq = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed)
    teaching_keys = list(map(int, rng.choice(10, 2, replace=False)))
    puzzle, seen, observed = {'train': [], 'test': []}, set(), {0, 1, 2}
    for index in range(nt+nq):
        train = index < nt
        if index < 2:
            keys, counts = teaching_keys, [1, 2] if index == 0 else [2, 0]
        else:
            n = int(rng.integers(1, 3)) if train else 3
            keys = list(map(int, rng.choice(10, n, replace=False)))
            allowed = list(range(6)) if train else sorted(observed)
            counts = [int(_pick(rng, allowed)) for _ in keys]
            while sum(counts) > 9:
                counts[int(np.argmax(counts))] -= 1
            if not train:
                # Do not introduce a count by reducing an overflowing budget.
                counts = [int(_pick(rng, sorted(observed & {1, 2, 3}))) for _ in keys]
                counts[:2] = [1, 2]
                rng.shuffle(counts)
        for _ in range(5):
            pair = _make_pair(rng, keys, counts, index if index < 2 else None)
            key = _grid_key(pair['input'])
            if key not in seen:
                break
        else:
            raise RuntimeError('could not generate distinct input')
        seen.add(key)
        puzzle['train' if train else 'test'].append(pair)
        if train:
            observed.update(counts)
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(info):
    reached_branch = detour = partial = False
    geometries, depths = [], []
    for key, template in info['templates'].items():
        distances, budget = info['distances'][key], info['counts'][key]
        geometries.append(tuple((r, c) for r, c, color in template))
        depths.append(max(distances.values()))
        shell_sizes = [sum(d == j for d in distances.values()) for j in range(1, budget+1)]
        reached_branch |= any(n >= 2 for n in shell_sizes)
        detour |= any(abs(r)+abs(c) <= budget < d for (r, c), d in distances.items())
        partial |= 0 < budget < max(distances.values())
    return dict(geometries=geometries, depths=depths, branch_witness=reached_branch,
                detour_witness=detour, partial=partial,
                multiplicities={key: sum(k == key for r, c, k in info['anchors']) for key in info['templates']})


def _evidence(details):
    triples = [(key, d['counts'][key], d['areas'][key], max(d['distances'][key].values()))
               for d in details for key in d['templates']]
    # The rendered graph ball determines a single radius until saturation;
    # afterward it gives only a lower bound. Demand disjoint visible radius
    # intervals at equal color area, rather than just different latent counts.
    def interval(n, depth):
        return min(n, depth), n if n < depth else float('inf')
    collision = False
    for k, n, a, depth in triples:
        low, high = interval(n, depth)
        for kk, nn, aa, dd in triples:
            ll, hh = interval(nn, dd)
            collision |= k == kk and a == aa and (low > hh or ll > high)
    return collision, any(n == nn and a != aa for k, n, a, d in triples for kk, nn, aa, dd in triples)


def validate_generated_puzzle(puzzle):
    """Check grid-derived semantics and teaching evidence, with no latent data."""
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
        _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
        nt, nq = len(puzzle['train']), len(puzzle['test'])
        _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
        seen, details = set(), {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
                inp, out = _grid(pair['input']), _grid(pair['output'])
                key = _grid_key(inp)
                _require(key not in seen, 'duplicate input')
                seen.add(key)
                info = _interpret(inp)
                _require(np.array_equal(out, _render(info)), 'wrong growth, fill or crop')
                _require(not np.array_equal(inp, out), 'identity pair')
                _require(all(1 <= n <= 2 for n in info['repetitions'].values()), 'reference multiplicities')
                details[split].append(info)
        train, test = details['train'], details['test']
        _require(all(_evidence(train)), 'area/count teaching witnesses missing')
        _require(all(len(d['templates']) <= 2 for d in train) and all(len(d['templates']) == 3 for d in test), 'type-combination shift missing')
        observed = {n for d in train for n in d['counts'].values()}
        _require({0, 1, 2} <= observed, 'growth progression missing')
        _require(all(set(d['counts'].values()) <= observed for d in test), 'test budget not demonstrated')
        _require(any(_features(d)['branch_witness'] for d in train), 'branching not demonstrated')
        _require(any(_features(d)['detour_witness'] for d in train), 'geodesic detour not demonstrated')
        _require(any(d['reference']['fill'] != d['target']['fill'] and not np.array_equal(_render(d), _render(d, solid=True)) for d in train), 'transparency witness missing')
        _require(any(any(n > 1 and d['counts'][key] > 0 for key, n in _features(d)['multiplicities'].items()) for d in train), 'repeated fueled root missing')
        # Counterfactual ablations must actually change visible training outputs.
        alternatives = [False]*5
        for d in train:
            correct = _render(d)
            full = {k: max(ds.values()) for k, ds in d['distances'].items()}
            all_objects = {k: len(d['objects']) for k in d['templates']}
            alternatives[0] |= not np.array_equal(correct, _render(d, full))
            alternatives[1] |= not np.array_equal(correct, _render(d, d['areas']))
            alternatives[2] |= not np.array_equal(correct, _render(d, unary=True))
            alternatives[3] |= not np.array_equal(correct, _render(d, all_objects))
            with_reference = {k: n+d['repetitions'][k] for k, n in d['counts'].items()}
            alternatives[4] |= not np.array_equal(correct, _render(d, with_reference))
        _require(all(alternatives), 'a parent-omitting shortcut fits all training')
        return None
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise AssertionError(str(error)) from error


def self_test():
    """Bounded tests and actual feature coverage across exactly 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, structures, geometries, masks, shapes, depths, budgets, backgrounds = [], set(), set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _interpret(pair['input'])
                features = _features(info)
                records.append((info, features))
                backgrounds.add(info['background'])
                budgets.update(info['counts'].values())
                geometries.update(features['geometries'])
                depths.update(features['depths'])
                shapes.add(pair['input'].shape)
                # Canonicalize colors in row-major order, preserving geometry.
                roles = {}
                structures.add((pair['input'].shape, tuple(roles.setdefault(int(v), len(roles)) for v in pair['input'].flat)))
                for o in info['objects']:
                    masks.add(_normalize(o['cells']))
                _require(np.array_equal(solve(pair['input']), pair['output']), 'solve disagreement')
                # All global symmetries preserve both the rule and crop orientation.
                if split == 'test':
                    _require(np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output'])), 'rotation covariance')
    _require(len(geometries) >= 120 and len(masks) >= 60, 'insufficient geometry variation')
    _require(len(shapes) >= 30 and len(structures) == 210, 'insufficient structural variation')
    _require(backgrounds == set(range(10)), 'background roles missing')
    _require(set(range(6)) <= budgets and len(depths) >= 7, 'budget/depth range')
    _require({len(d['templates']) for d, f in records} == {1, 2, 3}, 'type count variation')
    _require(any(max(d['repetitions'].values()) == 2 for d, f in records), 'repeated reference coverage')
    _require(any(len({v for t in d['templates'].values() for r, c, v in t if (r, c) != (0, 0)}) > 1 for d, f in records), 'multicolor templates missing')
    a, b = generate_puzzle(73, 6, 3), generate_puzzle(73, 6, 3)
    _require(all(_grid_key(x[k]) == _grid_key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), 'nondeterministic')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(100+nt*5+nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Metamorphic area invariance: remove a non-cut vertex from a fuel object.
    info = _interpret(puzzles[0]['train'][0]['input'])
    original = _render(info)
    changed = False
    for obj in info['objects']:
        if obj['color'] not in info['templates'] or len(obj['cells']) < 2:
            continue
        for r, c in obj['cells']:
            inp = info['input'].copy()
            inp[r, c] = info['background']
            try:
                new = _interpret(inp)
            except ValueError:
                continue
            if new['counts'] == info['counts']:
                _require(np.array_equal(solve(inp), original), 'area invariance failed')
                changed = True
                break
        if changed:
            break
    _require(changed, 'area metamorphism not exercised')
    # Change an in-range reference decoration consistently across repetitions.
    key = next(k for k in info['templates'] if info['counts'][k] > 0)
    offset = next((r, c) for r, c, v in info['templates'][key] if 0 < info['distances'][key][r, c] <= info['counts'][key])
    old_color = next(v for r, c, v in info['templates'][key] if (r, c) == offset)
    free = sorted(set(range(10)) - (set(info['templates']) | {info['background'],
        info['reference']['fill'], info['target']['fill'], info['target']['frame'], old_color}))
    inp = info['input'].copy()
    ref = info['reference']
    interior = _crop(inp, ref['box'])[1:-1, 1:-1]
    for cells in ref['components']:
        roots = [(r, c) for r, c in cells if int(interior[r, c]) == key]
        if roots:
            r, c = roots[0]
            interior[r+offset[0], c+offset[1]] = free[0]
    _require(not np.array_equal(solve(inp), original), 'reference color had no effect')
    # A new disconnected fuel object adds exactly one graph shell at every root.
    key = next(k for k, ds in info['distances'].items()
               if info['counts'][k] < min(5, max(ds.values())))
    changed = False
    for r in range(2, info['input'].shape[0]-2):
        for c in range(2, info['input'].shape[1]-2):
            if not np.all(info['input'][r-1:r+2, c-1:c+2] == info['background']):
                continue
            if any(y <= r < y+h and x <= c < x+w
                   for y, x, h, w in (info['reference']['box'], info['target']['box'])):
                continue
            inp = info['input'].copy()
            inp[r, c] = key
            try:
                new = _interpret(inp)
            except ValueError:
                continue
            predicted = solve(inp)
            expected_changes = sum(d == info['counts'][key]+1 for d in info['distances'][key].values())
            expected_changes *= sum(k == key for ar, ac, k in info['anchors'])
            _require(np.count_nonzero(predicted != original) == expected_changes > 0, 'one-object shell metamorphism')
            changed = True
            break
        if changed:
            break
    if not changed:
        axis = 0 if info['input'].shape[0] <= 27 else 1 if info['input'].shape[1] <= 27 else None
        _require(axis is not None, 'metamorphic fixture needs room for extra exterior')
        padding = [(0, 0), (0, 0)]
        padding[axis] = (0, 3)
        inp = np.pad(info['input'], padding, constant_values=info['background'])
        position = [2, 2]
        position[axis] = info['input'].shape[axis]+1
        inp[tuple(position)] = key
        predicted = solve(inp)
        expected_changes = sum(d == info['counts'][key]+1 for d in info['distances'][key].values())
        expected_changes *= sum(k == key for ar, ac, k in info['anchors'])
        _require(np.count_nonzero(predicted != original) == expected_changes > 0, 'padded one-object shell metamorphism')
        changed = True
    _require(changed, 'fuel shell metamorphism not exercised')
    # Corrupting an output must fail semantic validation.
    broken = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt border accepted')
    return dict(seeds=42, pairs=210, rooted_geometries=len(geometries), object_masks=len(masks),
                input_shapes=len(shapes), normalized_structures=len(structures), graph_depths=sorted(depths),
                budgets=sorted(budgets), backgrounds=sorted(backgrounds),
                branch_pairs=sum(f['branch_witness'] for d, f in records),
                detour_pairs=sum(f['detour_witness'] for d, f in records))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
