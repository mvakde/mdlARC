"""Finite projector assembly, pooled local codes, and occluded projection.

No files, random state, or generation metadata are consulted by solve().
All searches and construction retries are bounded. No execution traces are saved.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


class _SearchLimit(ValueError):
    pass


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(a):
    a = np.asarray(a)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _border(shape):
    h, w = shape
    return ([(0, x) for x in range(1, w-1)]
            + [(h-1, x) for x in range(1, w-1)]
            + [(y, 0) for y in range(1, h-1)]
            + [(y, w-1) for y in range(1, h-1)])


def _project(piece, side, code, occupied):
    """Rays exclude their emitter and stop BEFORE the next occupied cell."""
    h, w = occupied.shape
    labels = piece['labels']
    length = labels.size // 2
    result = []
    for slot, color in sorted(code.items()):
        ay, ax = map(int, np.argwhere(labels == side*length+slot)[0])
        by, bx = map(int, np.argwhere(labels == (1-side)*length+slot)[0])
        dy, dx = ay-by, ax-bx
        y, x = piece['y']+ay+dy, piece['x']+ax+dx
        cells = []
        while 0 <= y < h and 0 <= x < w and not occupied[y, x]:
            cells.append((y, x))
            y, x = y+dy, x+dx
        endpoint = cells[-1] if cells and not (0 <= y < h and 0 <= x < w) else None
        result.append((slot, int(color), tuple(cells), endpoint, (dy, dx)))
    return result


def _piece(template, t, y, x):
    return {'a': _orient(template, t),
            'labels': _orient(np.arange(template.size).reshape(template.shape), t),
            'y': int(y), 'x': int(x), 't': int(t)}


def _occupied(shape, pieces):
    mask = np.zeros(shape, dtype=bool)
    for p in pieces:
        h, w = p['a'].shape
        mask[p['y']:p['y']+h, p['x']:p['x']+w] = True
    return mask


def _tray_decode(tray, bg):
    # Tiles are interior; boundary pixels are observations, including adjacent ones.
    if any(int(tray[p]) != bg for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))):
        raise ValueError("Tray corners must be background")
    interior = tray != bg
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    tiles = []
    for cells in _components(interior):
        ys, xs = zip(*cells)
        y, x = min(ys), min(xs)
        a = tray[y:max(ys)+1, x:max(xs)+1].copy()
        if (min(a.shape) != 2 or not 4 <= max(a.shape) <= 7
                or a.size != len(cells) or np.any(a == bg)):
            raise ValueError("Tray pieces must be complete 2 by L rectangles")
        if y < 2 or x < 2 or max(ys) >= tray.shape[0]-2 or max(xs) >= tray.shape[1]-2:
            raise ValueError("Tray tiles require a two-cell boundary clearance")
        tiles.append({'a': a, 'y': y, 'x': x})
    if not 3 <= len(tiles) <= 6:
        raise ValueError("Expected three through six physical tiles")
    template = tiles[0]['a']
    if template.shape[0] != 2:
        template = np.rot90(template).copy()
    forms = [_orient(template, t) for t in range(8)]
    if len({_key(f) for f in forms}) != 8:
        raise ValueError("Template correspondence must be asymmetric")
    if not 2 <= len(np.unique(template)) <= 3:
        raise ValueError("Expected two or three structural colors")
    for p in tiles:
        matches = [t for t, f in enumerate(forms) if np.array_equal(f, p['a'])]
        if len(matches) != 1:
            raise ValueError("Tiles must share one exact dihedral template")
        p.update(_piece(template, matches[0], p['y'], p['x']))
    occupied = _occupied(tray.shape, tiles)
    possible = {}
    for i, p in enumerate(tiles):
        for side in (0, 1):
            rays = _project(p, side, dict.fromkeys(range(template.shape[1]), 0), occupied)
            for slot, _, _, endpoint, _ in rays:
                if endpoint is not None:
                    possible.setdefault(endpoint, []).append((i, side, slot))
    observations = [(p, int(tray[p])) for p in _border(tray.shape) if tray[p] != bg]
    structural = set(map(int, template.flat))
    if not observations or structural.intersection(c for _, c in observations):
        raise ValueError("Signals must be distinct from structural colors")
    code, sides, allocations = {}, set(), [set() for _ in tiles]
    for point, color in observations:
        options = possible.get(point, [])
        if len(options) != 1:
            raise ValueError("Each calibration endpoint needs one unobstructed interpretation")
        i, side, slot = options[0]
        if slot in code and code[slot] != color:
            raise ValueError("Conflicting calibration observations")
        code[slot] = color
        sides.add(side)
        allocations[i].add(slot)
    if len(sides) != 1 or not 2 <= len(code) <= 3:
        raise ValueError("Expected a common side and two or three evidenced slots")
    groups = []
    for p in tiles:
        matching = [g for g in groups if np.array_equal(g['a'], p['a'])]
        if matching:
            matching[0]['count'] += 1
        else:
            groups.append(dict(p, count=1))
    return dict(template=template, tiles=tiles, groups=groups, code=code,
                side=sides.pop(), allocations=allocations, structural=structural,
                observations=observations)


def _parse(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("Background must be uniquely most frequent")
    bg = int(colors[np.argmax(counts)])
    dividers = []
    for axis in (0, 1):
        for cut in range(1, a.shape[axis]-1):
            line = a[cut, :] if axis == 0 else a[:, cut]
            if np.all(line == line[0]) and int(line[0]) != bg:
                dividers.append((axis, cut, int(line[0])))
    if len(dividers) != 1:
        raise ValueError("Expected one full-span colored separator")
    axis, cut, separator = dividers[0]
    panels = (a[:cut], a[cut+1:]) if axis == 0 else (a[:, :cut], a[:, cut+1:])
    interpretations = []
    for index in (0, 1):
        board, tray = panels[index], panels[1-index]
        if min(board.shape) < 5 or min(tray.shape) < 5:
            continue
        try:
            d = _tray_decode(tray, bg)
            interior_colors = set(map(int, board[1:-1, 1:-1].flat)) - {bg}
            if len(interior_colors) != 1:
                continue
            vacancy = interior_colors.pop()
            if vacancy in d['structural'] or vacancy in d['code'].values() or vacancy == separator:
                continue
            mask = board == vacancy
            if (mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any()
                    or mask.sum() != sum(t['a'].size for t in d['tiles'])):
                continue
            if any(board[p] != bg for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))):
                continue
            probes = [(p, int(board[p])) for p in _border(board.shape) if board[p] != bg]
            if not probes or any(c not in d['code'].values() for _, c in probes):
                continue
            if separator in set(map(int, tray.flat)) | set(map(int, board.flat)):
                continue
            d.update(board=board.copy(), tray=tray.copy(), mask=mask, probes=probes,
                     bg=bg, vacancy=vacancy, separator=separator,
                     panel_side=('top', 'bottom')[index] if axis == 0 else ('left', 'right')[index])
            interpretations.append(d)
        except ValueError:
            continue
    if len(interpretations) != 1:
        raise ValueError("Panels and local color roles must be uniquely readable")
    return interpretations[0]


def _anchors(d):
    """Trace each board endpoint inward to its first hidden rectangle cell."""
    h, w = d['mask'].shape
    result = []
    for (y, x), color in d['probes']:
        dy, dx = (1, 0) if y == 0 else (-1, 0) if y == h-1 else (0, 1) if x == 0 else (0, -1)
        yy, xx = y+dy, x+dx
        while 0 <= yy < h and 0 <= xx < w and not d['mask'][yy, xx]:
            yy, xx = yy+dy, xx+dx
        if not (0 <= yy < h and 0 <= xx < w):
            raise ValueError("A board endpoint must look at the stencil")
        result.append(((yy, xx), (y, x), color))
    return result


def _render(d, pieces, *, code=None, geometry_only=False):
    out = d['board'].copy()
    for p in pieces:
        h, w = p['a'].shape
        out[p['y']:p['y']+h, p['x']:p['x']+w] = p['a']
    if geometry_only:
        return out
    painted = {}
    for p in pieces:
        for _, color, cells, _, _ in _project(p, d['side'], d['code'] if code is None else code, d['mask']):
            for point in cells:
                if point in painted and painted[point] != color:
                    return None
                if int(d['board'][point]) not in (d['bg'], color):
                    return None
                painted[point] = color
                out[point] = color
    return out


def _search(d, *, use_inventory=True, geometry_only=False, limit=18000):
    """Enumerate rendered answers, stopping only after two different images."""
    mask = d['mask']
    coordinates = list(map(tuple, np.argwhere(mask).tolist()))
    indices = {p: i for i, p in enumerate(coordinates)}
    anchors = [] if geometry_only else _anchors(d)
    by_cell = [[] for _ in coordinates]
    candidates = []
    for group, g in enumerate(d['groups']):
        h, w = g['a'].shape
        for y in range(mask.shape[0]-h+1):
            for x in range(mask.shape[1]-w+1):
                if not mask[y:y+h, x:x+w].all():
                    continue
                p = dict(g, y=y, x=x)
                rays = _project(p, d['side'], d['code'], mask)
                endpoints = {end: c for _, c, _, end, _ in rays if end is not None}
                if any(y <= hit[0] < y+h and x <= hit[1] < x+w and endpoints.get(end) != c
                       for hit, end, c in anchors):
                    continue
                cells = [indices[(yy, xx)] for yy in range(y, y+h) for xx in range(x, x+w)]
                bits = sum(1 << i for i in cells)
                ci = len(candidates)
                candidates.append((group, bits, p))
                for cell in cells:
                    by_cell[cell].append(ci)
    remaining = [g['count'] for g in d['groups']]
    full = (1 << len(coordinates))-1
    outcomes, path, nodes = {}, [], 0

    def visit(covered):
        nonlocal nodes
        nodes += 1
        if nodes > limit:
            raise _SearchLimit("Assembly search exceeded its explicit node budget")
        if covered == full:
            if use_inventory and any(remaining):
                return
            pieces = [candidates[i][2] for i in path]
            out = _render(d, pieces, geometry_only=geometry_only)
            if out is not None:
                outcomes.setdefault(_key(out), (out, tuple(pieces)))
            return
        best = None
        for cell in range(len(coordinates)):
            if covered & (1 << cell):
                continue
            options = [ci for ci in by_cell[cell]
                       if not covered & candidates[ci][1]
                       and (not use_inventory or remaining[candidates[ci][0]] > 0)]
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        for ci in best:
            group, bits, _ = candidates[ci]
            remaining[group] -= 1
            path.append(ci)
            visit(covered | bits)
            path.pop()
            remaining[group] += 1
            if len(outcomes) >= 2:
                return

    visit(0)
    return list(outcomes.values()), nodes


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _parse(input_grid)
    answers, _ = _search(d)
    if len(answers) != 1:
        raise ValueError("The input does not determine one completed image")
    return answers[0][0].copy()


def _template(rng, length, colors):
    for _ in range(40):
        a = rng.choice(colors, size=(2, length)).astype(np.int8)
        style = int(rng.integers(4))
        if style == 1:
            a[0] = colors[0]
            a[1] = colors[1]
            a[1, :int(rng.integers(1, length))] = colors[-1]
        elif style == 2:
            a[:] = colors[0]
            a[1, 1:] = colors[1]
            a[0, -1] = colors[-1]
        elif style == 3:
            a[:] = colors[1]
            a[0, 0] = colors[0]
            a[1, -2:] = colors[-1]
        for i, c in enumerate(colors):
            a[1, i] = c
        if len({_key(_orient(a, t)) for t in range(8)}) == 8:
            return a
    a = np.full((2, length), colors[0], dtype=np.int8)
    a[1, 1:] = colors[1]
    if len(colors) == 3:
        a[1, 2] = colors[2]
    return a


def _grow(rng, template, transforms, shape):
    h, w = shape
    pieces = []
    occupied = np.zeros(shape, dtype=bool)
    for t in transforms:
        th, tw = _orient(template, t).shape
        choices = []
        for _ in range(150):
            if not pieces or (len(pieces) > 2 and rng.random() < .12):
                y, x = int(rng.integers(2, h-th-1)), int(rng.integers(2, w-tw-1))
            else:
                q = pieces[int(rng.integers(len(pieces)))]
                qh, qw = q['a'].shape
                side = int(rng.integers(4))
                if side < 2:
                    y = q['y']-th if side == 0 else q['y']+qh
                    x = q['x']+int(rng.integers(1-tw, qw))
                else:
                    x = q['x']-tw if side == 2 else q['x']+qw
                    y = q['y']+int(rng.integers(1-th, qh))
            if y < 2 or x < 2 or y+th > h-2 or x+tw > w-2:
                continue
            if not occupied[y:y+th, x:x+tw].any():
                choices.append((y, x))
        if not choices:
            return None
        y, x = choices[int(rng.integers(len(choices)))]
        p = _piece(template, t, y, x)
        pieces.append(p)
        occupied[y:y+th, x:x+tw] = True
    return pieces, occupied


def _source_layout(rng, template, transforms, shape, bg, side, code, mode):
    h, w = shape
    for _ in range(35):
        pieces = []
        occupied = np.zeros(shape, dtype=bool)
        for t in rng.permutation(transforms):
            th, tw = _orient(template, int(t)).shape
            choices = []
            for _ in range(100):
                y, x = int(rng.integers(2, h-th-1)), int(rng.integers(2, w-tw-1))
                if not occupied[y-1:y+th+1, x-1:x+tw+1].any():
                    choices.append((y, x))
            if not choices:
                break
            y, x = choices[int(rng.integers(len(choices)))]
            pieces.append(_piece(template, int(t), y, x))
            occupied[y:y+th, x:x+tw] = True
        if len(pieces) != len(transforms):
            continue
        options = {s: [] for s in code}
        for i, p in enumerate(pieces):
            for s, _, _, endpoint, _ in _project(p, side, code, occupied):
                if endpoint is not None and endpoint in _border(shape):
                    options[s].append((i, endpoint))
        if any(not choices for choices in options.values()):
            continue
        common = set(range(len(pieces)))
        for values in options.values():
            common &= {i for i, _ in values}
        if mode == 'full' and not common:
            continue
        selected, owner = [], int(rng.choice(sorted(common))) if mode == 'full' else None
        for s, choices in options.items():
            choices = [v for v in choices if owner is None or v[0] == owner]
            selected.append((s, choices[int(rng.integers(len(choices)))]))
        if mode == 'pool' and len({i for _, (i, _) in selected}) < 2:
            continue
        tray = np.full(shape, bg, dtype=np.int8)
        for p in pieces:
            ph, pw = p['a'].shape
            tray[p['y']:p['y']+ph, p['x']:p['x']+pw] = p['a']
        for s, (_, point) in selected:
            tray[point] = code[s]
        # A few redundant agreeing observations are legal in mixed examples.
        if mode == 'mixed' and rng.random() < .35:
            s = int(rng.choice(list(code)))
            _, point = options[s][int(rng.integers(len(options[s])))]
            tray[point] = code[s]
        try:
            _tray_decode(tray, bg)
        except ValueError:
            continue
        return tray
    return None


def _build_pair(rng, count, demonstration, transform):
    need_inventory = demonstration == 0
    for attempt in range(220):
        palette = list(map(int, rng.permutation(10)))
        bg, vacancy, separator = palette[:3]
        structural_count = int(rng.integers(2, 4))
        length = int(rng.integers(4, 8))
        template = _template(rng, length, palette[3:3+structural_count])
        slot_count = 2 if demonstration is not None else int(rng.integers(2, 4))
        slots = sorted(map(int, rng.choice(length, slot_count, replace=False)))
        signals = palette[3+structural_count:3+structural_count+slot_count]
        code = dict(zip(slots, signals))
        if demonstration is None and rng.random() < .20:
            code[slots[-1]] = code[slots[0]]
        side = int(rng.integers(2))
        transforms = [0, 4, 1] + list(map(int, rng.integers(0, 8, size=count-3)))
        rng.shuffle(transforms)
        height, width = int(rng.integers(19, 30)), int(rng.integers(max(10, length+4), 15))
        grown = _grow(rng, template, transforms, (height, width))
        if grown is None:
            continue
        pieces, mask = grown
        board = np.full(mask.shape, bg, dtype=np.int8)
        board[mask] = vacancy
        visible = {}
        owners = {}
        for i, p in enumerate(pieces):
            for s, color, cells, endpoint, _ in _project(p, side, code, mask):
                if endpoint is not None and endpoint in _border(mask.shape):
                    if endpoint in visible and visible[endpoint] != color:
                        break
                    visible[endpoint] = color
                    owners[endpoint] = i
        if len(visible) < 3:
            continue
        for point, color in visible.items():
            board[point] = color
        # Constructed labels are independent of the inference search.
        direct = dict(board=board, mask=mask, code=code, side=side, bg=bg)
        expected = _render(direct, pieces)
        if expected is None:
            continue
        if demonstration == 1 and not any(
                cells and end is None for p in pieces
                for _, _, cells, end, _ in _project(p, side, code, mask)):
            continue
        mode = 'pool' if demonstration == 0 else 'full' if demonstration == 1 else str(rng.choice(['pool', 'full', 'mixed']))
        total_width = int(rng.integers(max(24, width+length+5), 31))
        tray = _source_layout(rng, template, transforms, (height, total_width-width-1), bg, side, code, mode)
        if tray is None:
            continue

        def packed(current):
            stripe = np.full((height, 1), separator, dtype=np.int8)
            return np.concatenate((current, stripe, tray), axis=1)

        try:
            d = _parse(packed(board))
            answers, _ = _search(d)
            if len(answers) != 1 or not np.array_equal(answers[0][0], expected):
                continue
            # Remove observations while preserving uniqueness. Omit a whole tile's
            # endpoint set first, so finite inventory can infer an unhinted copy.
            removal_groups = [[p for p in visible if owners[p] == i] for i in rng.permutation(count)]
            removal_groups += [[p] for p in rng.permutation(list(visible))]
            for group in removal_groups:
                trial = board.copy()
                for point in group:
                    trial[tuple(point)] = bg
                if sum(trial[p] != bg for p in _border(trial.shape)) < 2:
                    continue
                td = _parse(packed(trial))
                trial_answers, _ = _search(td)
                if len(trial_answers) == 1 and np.array_equal(trial_answers[0][0], expected):
                    board, d = trial, td
            if not any(board[p] == bg for p in visible):
                continue
            if need_inventory and len(_search(d, use_inventory=False)[0]) < 2:
                continue
            if len(_search(d, geometry_only=True)[0]) < 2:
                continue
            if _features(d, answers[0][1])['visible_slots'] != len(code):
                continue
            # Preserve exact source matrices under whole-input isometries only.
            result = {'input': _orient(packed(board), transform),
                      'output': _orient(expected, transform)}
            if not np.array_equal(solve(result['input']), result['output']):
                continue
            return result
        except (ValueError, AssertionError):
            continue
    raise RuntimeError("Bounded construction exhausted; no ambiguous fallback was returned")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError("Illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            demonstration = i if split == 'train' and i < 2 else None
            n = (3 if i == 0 else 4) if demonstration is not None else int(rng.integers(3, 5) if split == 'train' else rng.integers(5, 7))
            for _ in range(8):
                pair = _build_pair(rng, n, demonstration, int(rng.integers(8)))
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct-input retry budget exhausted")
    return puzzle


def _features(d, pieces):
    rays = [r for p in pieces for r in _project(p, d['side'], d['code'], d['mask'])]
    projections = Counter(s for s, _, cells, _, _ in rays if cells)
    endpoints = {end for _, _, _, end, _ in rays if end is not None}
    observed = {point for point, _ in d['probes']}
    comps = _components(d['mask'])
    ownership = np.full(d['mask'].shape, -1, dtype=int)
    for i, p in enumerate(pieces):
        h, w = p['a'].shape
        ownership[p['y']:p['y']+h, p['x']:p['x']+w] = i
    component_tiles = [len({int(ownership[p]) for p in c}) for c in comps]
    return dict(tiles=len(pieces), length=d['template'].shape[1], slots=tuple(sorted(d['code'])),
                signals=len(set(d['code'].values())), side=d['panel_side'], shape=d['board'].shape,
                code_size=len(d['code']), source_pool=all(len(s) < len(d['code']) for s in d['allocations']),
                source_full=any(len(s) == len(d['code']) for s in d['allocations']),
                redundant=len(d['observations']) > len(d['code']),
                blocked=sum(end is None for _, _, _, end, _ in rays),
                positive_blocked=sum(bool(cells) and end is None for _, _, cells, end, _ in rays),
                zero_rays=sum(not cells for _, _, cells, _, _ in rays),
                directions=len({r[-1] for r in rays}), missing_endpoints=len(endpoints-observed),
                components=len(comps), max_component_tiles=max(component_tiles),
                repeated=max(g['count'] for g in d['groups']),
                visible_slots=len(projections), roles=(d['bg'], d['vacancy'], d['separator']),
                normalized=_key(np.where(d['mask'], 1, 0)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, testing = set(), [], []
    inventory_witness = False
    for split in ('train', 'test'):
        for i, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _parse(inp)
            answers, _ = _search(d)
            assert len(answers) == 1 and np.array_equal(answers[0][0], out)
            pieces = answers[0][1]
            assert len(pieces) == len(d['tiles'])
            assert Counter(_key(p['a']) for p in pieces) == Counter(_key(p['a']) for p in d['tiles'])
            assert not np.any(out == d['vacancy'])
            f = _features(d, pieces)
            assert f['missing_endpoints'] > 0
            assert f['visible_slots'] == f['code_size']
            # With only silhouettes and source inventory, template assignments
            # have different rendered answers. The ray code is consequential.
            assert len(_search(d, geometry_only=True)[0]) == 2
            if split == 'train':
                training.append(f)
                if i == 0:
                    assert f['source_pool']
                    assert len(_search(d, use_inventory=False)[0]) == 2
                    inventory_witness = True
                    first_code = {s: d['code'][s] for s in d['allocations'][0]}
                    partial = _render(d, pieces, code=first_code)
                    assert partial is None or not np.array_equal(partial, out)
                if i == 1:
                    assert f['source_full']
                    assert f['positive_blocked'] > 0
            else:
                testing.append(f)
    assert inventory_witness
    assert min(f['tiles'] for f in testing) > max(f['tiles'] for f in training)
    assert any(f['blocked'] for f in training)


def self_test():
    """42 streamed seeds, meaningful counterfactuals, no disk artifacts."""
    features, fingerprints = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _parse(pair['input'])
                answers, _ = _search(d)
                features.append(_features(d, answers[0][1]))
                digest.update(repr(pair['input'].shape).encode())
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
        fingerprints.add(digest.hexdigest())
    assert len(fingerprints) == 42
    assert {f['length'] for f in features} == {4, 5, 6, 7}
    assert {f['tiles'] for f in features} == {3, 4, 5, 6}
    assert {f['side'] for f in features} == {'left', 'right', 'top', 'bottom'}
    assert len({f['normalized'] for f in features}) > 150
    assert len({f['shape'] for f in features}) > 35
    assert any(f['code_size'] == 3 for f in features)
    assert any(f['signals'] < f['code_size'] for f in features)
    assert any(f['repeated'] >= 2 for f in features)
    assert any(f['max_component_tiles'] >= 4 for f in features)
    assert any(f['components'] > 1 for f in features)
    assert any(f['zero_rays'] > 0 for f in features)
    assert any(f['positive_blocked'] > 0 for f in features)
    # Input-only inference must commute with every whole-scene isometry and
    # arbitrary recoloring (these modify both source matrices and destination).
    pair = generate_puzzle(2718, 2, 1)['test'][0]
    recolor = np.array([7, 4, 9, 0, 2, 8, 1, 6, 3, 5], dtype=np.int8)
    for t in range(8):
        assert np.array_equal(solve(_orient(recolor[pair['input']], t)), _orient(recolor[pair['output']], t))
    assert _key(generate_puzzle(999, 2, 1)['test'][0]['input']) == _key(generate_puzzle(999, 2, 1)['test'][0]['input'])
    for a, b in ((1, 1), (11, 1), (2, 2), (4, 0), (6, 5), (3, 4), (True, 1), (3.0, 1), (3, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    corrupt = generate_puzzle(81, 2, 1)
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect output accepted")
