"""Phase brakes on rectangular tracks hidden by an opaque window.

The reader uses only pixels. Construction descriptors never reach solve().
Marker 1 is a courier; its adjacent marker 2 is its stationary direction tail.
All other color roles are inferred geometrically in each input.
"""

from __future__ import annotations

import copy
import json
import numpy as np


HEAD, TAIL = 1, 2
MAX_PERIOD = 6


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer grids required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a.astype(np.int8, copy=False)


def _walk(box):
    """Clockwise perimeter, without duplicated corners."""
    t, b, l, r = box
    return ([(t, x) for x in range(l, r + 1)]
            + [(y, r) for y in range(t + 1, b + 1)]
            + [(b, x) for x in range(r - 1, l - 1, -1)]
            + [(y, l) for y in range(b - 1, t, -1)])


def _inset(box, amount):
    t, b, l, r = box
    return t + amount, b - amount, l + amount, r - amount


def _contains(box, p, strict=False):
    t, b, l, r = box
    y, x = p
    return t < y < b and l < x < r if strict else t <= y <= b and l <= x <= r


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _components(points):
    remaining, result = set(points), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, pending = {p}, [p]
        while pending:
            y, x = pending.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    pending.append(q)
        result.append(group)
    return result


def _window(g):
    bg = int(g[0, 0])
    _require(bg not in (HEAD, TAIL), "background cannot be a marker")
    _require(all(int(g[p]) == bg for p in
                 ((0, 0), (0, g.shape[1] - 1), (g.shape[0] - 1, 0),
                  (g.shape[0] - 1, g.shape[1] - 1))), "background corners")
    candidates = []
    for fill in map(int, np.unique(g)):
        if fill in (bg, HEAD, TAIL):
            continue
        positions = {tuple(map(int, p)) for p in np.argwhere(g == fill)}
        # The fill role is used exclusively inside one window.
        t, b, l, r = _bbox(positions)
        if b - t < 1 or r - l < 1 or len(positions) != (b - t + 1) * (r - l + 1):
            continue
        box = t - 1, b + 1, l - 1, r + 1
        if box[0] < 1 or box[2] < 1 or box[1] >= g.shape[0] - 1 or box[3] >= g.shape[1] - 1:
            continue
        border = _walk(box)
        frame = int(g[border[0]])
        if frame in (bg, fill, HEAD, TAIL):
            continue
        if all(int(g[p]) == frame for p in border) and np.count_nonzero(g == frame) == len(border):
            candidates.append((box, frame, fill))
    _require(len(candidates) == 1, "one uniquely identifiable opaque window required")
    return bg, candidates[0]


def _complete_boundary(g, box, visible, window):
    """Continue a short periodic perimeter from any corner origin.

    A rotated perimeter can have its repetition seam at any of its four
    corners. Every residue needs two exposed witnesses. Different explanations
    are acceptable only if they agree on every hidden color.
    """
    cells = _walk(box)
    if not any(_contains(window, p) for p in cells):
        return {p: int(g[p]) for p in cells}, 0
    t, b, l, r = box
    origins = (0, r - l, r - l + b - t, 2 * (r - l) + b - t)
    observed = [(i, int(g[p])) for i, p in enumerate(cells) if p in visible]
    completions, fits = {}, []
    for origin in origins:
        for sign in (1, -1):
            for period in range(1, MAX_PERIOD + 1):
                word, counts = {}, [0] * period
                for index, color in observed:
                    residue = ((sign * (index - origin)) % len(cells)) % period
                    if residue in word and word[residue] != color:
                        break
                    word[residue] = color
                    counts[residue] += 1
                else:
                    if min(counts) < 2:
                        continue
                    values = tuple(word[((sign * (i - origin)) % len(cells)) % period]
                                   for i in range(len(cells)))
                    completions[values] = period
                    fits.append(period)
    if len(completions) != 1:
        return None
    values = next(iter(completions))
    return dict(zip(cells, values)), min(fits)


def _outlines(g, bg, frame, fill, window):
    guide = {tuple(map(int, p)) for p in np.argwhere(
        (g != bg) & (g != frame) & (g != fill) & (g != HEAD) & (g != TAIL))}
    parts = _components(guide)
    _require(parts, "no exposed rails")
    candidates = []
    for i, a in enumerate(parts):
        for j in range(i, len(parts)):
            visible = a if i == j else a | parts[j]
            box = _bbox(visible)
            t, b, l, r = box
            if b - t < 2 or r - l < 2:
                continue
            expected = {p for p in _walk(box) if not _contains(window, p)}
            if expected != visible:
                continue
            completion = _complete_boundary(g, box, visible, window)
            if completion is not None:
                colors, period = completion
                candidates.append(dict(parts=frozenset((i, j)), box=box,
                                       colors=colors, period=period,
                                       visible=visible))
    covers = []

    def cover(remaining, chosen, occupied):
        if len(covers) > 1:
            return
        if not remaining:
            covers.append(chosen)
            return
        first = min(remaining)
        for item in candidates:
            cells = set(item['colors'])
            halo = {q for y, x in cells for q in
                    ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))}
            if first in item['parts'] and item['parts'] <= remaining and not halo & occupied:
                cover(remaining - item['parts'], chosen + [item], occupied | cells)

    cover(set(range(len(parts))), [], set())
    _require(len(covers) == 1, "ambiguous rectangular outline completion")
    return covers[0]


def _flanks(outer, p):
    """Two radial rails at a straight corridor cell; corners are ungated."""
    t, b, l, r = _inset(outer, 1)
    y, x = p
    if y in (t, b) and x in (l, r):
        return None
    if y in (t, b):
        return (y - 1, x), (y + 1, x)
    _require(x in (l, r), "courier is off the corridor")
    return (y, x - 1), (y, x + 1)


def _trace(track, skip_initial=False, ignore_window=None):
    cells, colors = track['cells'], track['colors']
    start, step = track['start'], track['step']

    def passes(index):
        p = cells[index]
        if ignore_window is not None and _contains(ignore_window, p):
            return True
        pair = _flanks(track['outer'], p)
        return pair is None or colors[pair[0]] == colors[pair[1]]

    if not skip_initial and not passes(start):
        return dict(destination=cells[start], path=[cells[start]], reason='initial_mismatch', brake=start)
    path, index = [cells[start]], start
    for _ in range(len(cells)):
        nxt = (index + step) % len(cells)
        if not passes(nxt):
            return dict(destination=cells[index], path=path, reason='mismatch', brake=nxt)
        index = nxt
        path.append(cells[index])
        if index == start:
            return dict(destination=cells[start], path=path, reason='full_lap', brake=None)
    raise AssertionError('bounded loop tracing failed')


def _read(g):
    bg, (window, frame, fill) = _window(g)
    outlines = _outlines(g, bg, frame, fill, window)
    by_box = {o['box']: o for o in outlines}
    used, tracks = set(), []
    heads = {tuple(map(int, p)) for p in np.argwhere(g == HEAD)}
    tails = {tuple(map(int, p)) for p in np.argwhere(g == TAIL)}
    assigned_heads, assigned_tails = set(), set()
    for outer in sorted(by_box):
        inner = _inset(outer, 2)
        if inner not in by_box:
            continue
        _require(outer not in used and inner not in used, "overlapping track pairings")
        used.update((outer, inner))
        cells = _walk(_inset(outer, 1))
        on_track = heads & set(cells)
        _require(len(on_track) == 1, "each track needs exactly one courier")
        head = next(iter(on_track))
        index = cells.index(head)
        _require(_flanks(outer, head) is not None, "initial head must be on a straight side")
        adjacent = [k for k in (-1, 1) if cells[(index + k) % len(cells)] in tails]
        _require(len(adjacent) == 1, "courier needs one adjacent direction tail")
        tail = cells[(index + adjacent[0]) % len(cells)]
        _require(not _contains(window, head) and not _contains(window, tail), "markers must be exposed")
        assigned_heads.add(head)
        assigned_tails.add(tail)
        colors = dict(by_box[outer]['colors'])
        colors.update(by_box[inner]['colors'])
        track = dict(outer=outer, inner=inner, cells=cells, colors=colors,
                     start=index, step=-adjacent[0], head=head, tail=tail)
        track['route'] = _trace(track)
        tracks.append(track)
    _require(used == set(by_box), "all outlines must belong to track pairs")
    _require(assigned_heads == heads and assigned_tails == tails, "unassigned markers")
    _require(1 <= len(tracks) <= 3, "one to three tracks supported")
    for i, track in enumerate(tracks):
        t, b, l, r = track['outer']
        for other in tracks[i + 1:]:
            u, v, x, z = other['outer']
            _require(b < u or v < t or r < x or z < l, "track rectangles must be disjoint")
    return dict(background=bg, frame=frame, fill=fill, window=window,
                outlines=outlines, tracks=tracks)


def _answer(scene, *, couriers=True, skip_initial=False, ignore_hidden_gates=False):
    t, b, l, r = scene['window']
    out = np.full((b - t - 1, r - l - 1), scene['fill'], dtype=np.int8)
    for outline in scene['outlines']:
        for (y, x), color in outline['colors'].items():
            if _contains(scene['window'], (y, x), strict=True):
                out[y - t - 1, x - l - 1] = color
    if couriers:
        for track in scene['tracks']:
            route = _trace(track, skip_initial=skip_initial,
                           ignore_window=scene['window'] if ignore_hidden_gates else None)
            y, x = route['destination']
            if _contains(scene['window'], (y, x), strict=True):
                out[y - t - 1, x - l - 1] = HEAD
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover rails and relocate couriers using only the supplied input."""
    g = _grid(input_grid)
    return _answer(_read(g))


def _geometry(rng, n, corner, large, direction=1):
    if corner:
        h, w = map(int, rng.integers(21 if large else 18, 31, size=2))
        t, l = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        b = int(rng.integers(t + 10, h - 3))
        r = int(rng.integers(l + 10, w - 3))
        box = (t, b, l, r)
        mt = int(rng.integers(t + 4, b - 2))
        ml = int(rng.integers(l + 4, r - 2))
        window = (mt, int(rng.integers(b + 1, h - 1)),
                  ml, int(rng.integers(r + 1, w - 1)))
        return h, w, [box], window
    h = int(rng.integers(23 if large else 21, 31))
    min_w = 29 if n == 3 else 23 if n == 2 else 17
    w = int(rng.integers(min_w, 31))
    # No fixed-size route modules: distribute free width and vary each height.
    widths = [7] * n
    slack = w - 4 - 2 * (n - 1) - sum(widths)
    for _ in range(int(rng.integers(0, slack + 1))):
        widths[int(rng.integers(n))] += 1
    left = int(rng.integers(2, w - sum(widths) - 2 * (n - 1) - 1))
    boxes = []
    for width in widths:
        t = int(rng.integers(2, 5))
        b = int(rng.integers(max(t + 14, h - 7), h - 2))
        boxes.append((t, b, left, left + width - 1))
        left += width + 2
    lo = max(box[0] for box in boxes) + 3
    hi = min(box[1] for box in boxes) - 3
    mask_height = int(rng.integers(5, min(9, hi - lo + 1) + 1))
    mt = (lo if direction == 1 else hi - mask_height + 1) if large else int(rng.integers(lo, hi - mask_height + 2))
    window = (mt, mt + mask_height - 1, boxes[0][2] - 1, boxes[-1][3] + 1)
    if large:
        anchor = max(range(n), key=lambda i: boxes[i][0]) if direction == 1 else min(range(n), key=lambda i: boxes[i][1])
        boxes.insert(0, boxes.pop(anchor))
    return h, w, boxes, window


def _pattern(rng, box, palette, word=None, phase=None):
    if word is None:
        period = int(rng.integers(1, MAX_PERIOD + 1))
        count = int(rng.integers(1, min(len(palette), period) + 1))
        colors = list(map(int, rng.choice(palette, count, replace=False)))
        word = colors + list(map(int, rng.choice(colors, period - count)))
        rng.shuffle(word)
    if phase is None:
        phase = int(rng.integers(len(word)))
    return {p: int(word[(i + phase) % len(word)]) for i, p in enumerate(_walk(box))}, tuple(word), phase


def _rail_pair(rng, box, palette, kind, direction):
    inner = _inset(box, 2)
    if kind == 'short':
        a, b = map(int, rng.choice(palette, 2, replace=False))
        period = int(rng.integers(4, 7))
        outer_colors, _, _ = _pattern(rng, box, palette, (a,), 0)
        inner_colors, _, _ = _pattern(rng, inner, palette, [a] * (period - 1) + [b])
    elif kind in ('turn', 'turn_band'):
        period = int(rng.choice([5, 6] if kind == 'turn_band' else [3, 5, 6]))
        if kind == 'turn_band':
            a, b = map(int, rng.choice(palette, 2, replace=False))
            word = [a] * (period - 1) + [b]
        else:
            word = list(map(int, rng.choice(palette, period)))
        if len(set(word)) == 1:
            word[0] = next(c for c in palette if c != word[0])
        outer_colors, word, phase = _pattern(rng, box, palette, word)
        # Align a complete incoming side; four extra cells at a bend change
        # the phase difference of the inner and outer rectangle boundaries.
        t, b, l, r = _inset(box, 1)
        if direction == 1:
            p = (t, (l + r) // 2) if kind == 'turn_band' else ((t + b) // 2, r)
        else:
            p = (b, (l + r) // 2)
        flanks = _flanks(box, p)
        outer_cell = next(q for q in flanks if q in outer_colors)
        inner_cell = next(q for q in flanks if q not in outer_colors)
        oi = _walk(box).index(outer_cell)
        ii = _walk(inner).index(inner_cell)
        inner_colors, _, _ = _pattern(rng, inner, palette, word, (oi + phase - ii) % period)
    else:
        outer_colors, _, _ = _pattern(rng, box, palette)
        inner_colors, _, _ = _pattern(rng, inner, palette)
    colors = dict(outer_colors)
    colors.update(inner_colors)
    return colors


def _turns(path):
    directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _starts(box, colors, window, direction, need_turn=False, inert=False):
    cells, candidates = _walk(_inset(box, 1)), []
    for start, head in enumerate(cells):
        tail = cells[(start - direction) % len(cells)]
        if _contains(window, head) or _contains(window, tail) or _flanks(box, head) is None:
            continue
        track = dict(outer=box, inner=_inset(box, 2), cells=cells, colors=colors,
                     start=start, step=direction, head=head, tail=tail)
        route = _trace(track)
        if inert:
            if route['reason'] != 'initial_mismatch':
                continue
            counterfactual = _trace(track, skip_initial=True)
            if not _contains(window, counterfactual['destination'], strict=True):
                continue
        else:
            if route['reason'] != 'mismatch' or len(route['path']) < 3:
                continue
            if not _contains(window, route['destination'], strict=True):
                continue
            if not _contains(window, cells[route['brake']]):
                continue
            if need_turn and _turns(route['path']) == 0:
                continue
            if _contains(window, _trace(track, ignore_window=window)['destination'], strict=True):
                continue
        track['route'] = route
        candidates.append(track)
    return candidates


def _make_pair(rng, split, index, rotation):
    first = split == 'train' and index == 0
    turn_demo = split == 'train' and index == 1
    corner = turn_demo or (split == 'train' and not first and rng.random() < .35)
    n = 1 if corner else 2 if first else int(rng.integers(1 if split == 'train' else 2, 4))
    # The first two examples expose both directions before tests mix them.
    preferred_direction = 1 if first else -1 if turn_demo else int(rng.choice([-1, 1]))
    for _ in range(500):
        h, w, boxes, window = _geometry(rng, n, corner, split == 'test', preferred_direction)
        roles = list(map(int, rng.permutation([0, 3, 4, 5, 6, 7, 8, 9])))
        bg, frame, fill = roles[:3]
        palette = roles[3:3 + int(rng.integers(2, 6))]
        g = np.full((h, w), bg, dtype=np.int8)
        tracks, failed = [], False
        for number, box in enumerate(boxes):
            inert = first and number == 1
            direction = preferred_direction if number == 0 or first else int(rng.choice([-1, 1]))
            kind = 'turn' if corner else 'turn_band' if split == 'test' and number == 0 else 'short'
            colors = _rail_pair(rng, box, palette, kind, direction)
            starts = _starts(box, colors, window, direction,
                             need_turn=turn_demo or (split == 'test' and number == 0), inert=inert)
            if first and number == 0:
                starts = [track for track in starts if _turns(track['route']['path']) == 0]
            if not starts and not first and number > 0:
                # A secondary track may visibly fail its initial match.
                starts = _starts(box, colors, window, direction, inert=True)
            if not starts:
                failed = True
                break
            track = starts[int(rng.integers(len(starts)))]
            tracks.append(track)
            for p, color in colors.items():
                g[p] = color
        if failed:
            continue
        t, b, l, r = window
        g[t + 1:b, l + 1:r] = fill
        for p in _walk(window):
            g[p] = frame
        for track in tracks:
            g[track['head']] = HEAD
            g[track['tail']] = TAIL
        # Truth is drawn from the full pre-occlusion rails and their couriers.
        truth = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
        for track in tracks:
            for (y, x), color in track['colors'].items():
                if _contains(window, (y, x), strict=True):
                    truth[y - t - 1, x - l - 1] = color
            y, x = track['route']['destination']
            if _contains(window, (y, x), strict=True):
                truth[y - t - 1, x - l - 1] = HEAD
        inp, out = np.rot90(g, rotation).copy(), np.rot90(truth, rotation).copy()
        try:
            scene = _read(inp)
            if not np.array_equal(_answer(scene), out):
                continue
            if np.array_equal(_answer(scene, ignore_hidden_gates=True), out):
                continue
            if first and np.array_equal(_answer(scene, skip_initial=True), out):
                continue
        except AssertionError:
            continue
        return dict(input=inp, output=out)
    raise AssertionError('bounded construction exhausted; no ambiguous scene is accepted')


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def _signature(g):
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    base_rotation = int(rng.integers(4))
    mixed_orientations = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            rotation = (base_rotation + (index if mixed_orientations else 0)) % 4
            for _ in range(20):
                pair = _make_pair(rng, split, index, rotation)
                key = _signature(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('bounded unique-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'pair counts')
    seen, training_turns, training_directions = set(), set(), set()
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _signature(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            scene = _read(inp)
            expected = _answer(scene)
            _require(np.array_equal(out, expected), 'incorrect reconstructed traffic crop')
            _require(out.shape != inp.shape, 'strict window crop required')
            _require(np.count_nonzero(out == HEAD) >= 1, 'every pair must show an arrived courier')
            _require(np.count_nonzero(out == TAIL) == 0, 'stationary direction tails are outside the crop')
            _require(not np.array_equal(out, _answer(scene, couriers=False)), 'parent A must affect the answer')
            _require(not np.array_equal(out, _answer(scene, ignore_hidden_gates=True)),
                     'hidden rail phase must affect courier braking')
            for track in scene['tracks']:
                route = track['route']
                _require(route['destination'] in track['cells'], 'courier stays on its corridor')
                _require(len(route['path']) <= len(track['cells']) + 1, 'trace is bounded by one lap')
                if route['reason'] == 'mismatch':
                    pair_flanks = _flanks(track['outer'], track['cells'][route['brake']])
                    _require(pair_flanks is not None and track['colors'][pair_flanks[0]] != track['colors'][pair_flanks[1]],
                             'terminal must precede a real unequal-rail gate')
                if split == 'train':
                    if route['destination'] != track['head']:
                        training_turns.add(_turns(route['path']))
                    training_directions.add(track['step'])
            # Reconstructed rails and arriving heads cannot overlap. Other
            # interior cells retain fill; routes are never painted as trails.
            rail_cells = {p for o in scene['outlines'] for p in o['colors']}
            _require(all(t['route']['destination'] not in rail_cells for t in scene['tracks']), 'rail/token collision')
            for p in rail_cells:
                if _contains(scene['window'], p, strict=True):
                    y, x = p
                    t, _, l, _ = scene['window']
                    _require(out[y - t - 1, x - l - 1] != HEAD, 'rails must be preserved')
            if split == 'train' and index == 0:
                _require(any(t['route']['reason'] == 'initial_mismatch' for t in scene['tracks']), 'inactive entrance demonstration')
                _require(any(t['route']['destination'] != t['head'] and _turns(t['route']['path']) == 0
                             for t in scene['tracks']), 'active straight route demonstration')
                _require(not np.array_equal(out, _answer(scene, skip_initial=True)), 'initial color match must matter')
            if split == 'train' and index == 1:
                _require(any(_turns(t['route']['path']) > 0 for t in scene['tracks']), 'bend demonstration')
    _require(training_directions == {-1, 1}, 'both marker directions must be demonstrated')
    _require(0 in training_turns and max(training_turns) > 0, 'straight and turning routes must be demonstrated')


def self_test():
    """42 reproducible seeds, semantic ablations, metamorphic and mutation checks."""
    shapes, outputs, geometries, periods, counts, directions, palettes = set(), set(), set(), set(), set(), set(), set()
    firsts, turn_counts, inactive = set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        firsts.add(_signature(puzzle['train'][0]['input']))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                saved = inp.copy()
                _require(np.array_equal(solve(inp), out), 'input-only solver agreement')
                _require(np.array_equal(inp, saved), 'solve must not mutate input')
                scene = _read(inp)
                shapes.add(inp.shape)
                outputs.add(out.shape)
                geometries.add((scene['window'], tuple(t['outer'] for t in scene['tracks'])))
                periods.update(o['period'] for o in scene['outlines'] if o['period'])
                counts.add(len(scene['tracks']))
                directions.update(t['step'] for t in scene['tracks'])
                palettes.add(tuple(sorted(map(int, np.unique(inp)))))
                turn_counts.update(_turns(t['route']['path']) for t in scene['tracks'])
                inactive += sum(t['route']['reason'] == 'initial_mismatch' for t in scene['tracks'])
    _require(len(firsts) == 42, 'seed diversity')
    _require(len(shapes) >= 25 and len(outputs) >= 25 and len(geometries) >= 150, 'structural variation')
    _require(counts == {1, 2, 3} and directions == {-1, 1}, 'track/direction diversity')
    _require({1, 3, 4, 5, 6} <= periods and len(palettes) >= 15, 'pattern/palette diversity')
    _require({0, 1} <= turn_counts and inactive >= 42, 'motion/selection diversity')
    p = generate_puzzle(0, 2, 1)
    q = generate_puzzle(0, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ('input', 'output')), 'whole-puzzle determinism')
    # Rotation respects the marker's direction and the color sequence seam.
    for k in (1, 2, 3):
        pair = p['train'][1]
        _require(np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k)), 'rotation equivariance')
    # All non-marker roles can be renamed without changing the rule.
    permutation = np.arange(10, dtype=np.int8)
    other = [0, 3, 4, 5, 6, 7, 8, 9]
    permutation[other] = other[1:] + other[:1]
    _require(np.array_equal(solve(permutation[p['test'][0]['input']]), permutation[p['test'][0]['output']]), 'role renaming')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    bad = copy.deepcopy(p)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect label accepted')
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), output_shapes=len(outputs),
                geometries=len(geometries), periods=sorted(periods), tracks=sorted(counts),
                route_turns=sorted(turn_counts), initial_mismatches=inactive,
                palettes=len(palettes), independent_solver=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
