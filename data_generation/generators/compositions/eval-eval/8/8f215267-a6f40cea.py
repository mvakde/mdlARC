"""Contour census: count objects, then bead their hidden rectangular track.

No parent module is imported. The renderer uses sampled scene descriptors;
the solver recovers the screen, contours, gates and objects from pixels.
"""
from collections import Counter
import hashlib
import numpy as np


HOST_SEEDS = (0,132235620,408467888,556938690,14523149,441335573,255664425,
             743558562,355365282,421453601,47927325,43711906,539321841,
             395927640,378714359,517632681,81177101,187813379,968755442,
             828153625,526431840,769787557,83743672,741402692,321790783,
             61234470,902170765,174827341,995579300,506629353,622034646,
             366991837,135831580,773564234,806944429,684862604,770913546,
             276775514,199413815,572812823,703282436,941847003)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    _check(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), "grid dimensions")
    _check(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _check(bool(np.all((g >= 0) & (g <= 9))), "color range")
    return g


def _perimeter(box):
    t, b, l, r = box
    return ({(t, c) for c in range(l, r + 1)} |
            {(b, c) for c in range(l, r + 1)} |
            {(y, l) for y in range(t + 1, b)} |
            {(y, r) for y in range(t + 1, b)})


def _inside(p, box, strict=False):
    y, x = p
    t, b, l, r = box
    return t < y < b and l < x < r if strict else t <= y <= b and l <= x <= r


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _neighbors(p):
    y, x = p
    return ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))


def _groups(points):
    remaining = set(points)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, group = [p], {p}
        while stack:
            for q in _neighbors(stack.pop()):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _halo(points, diagonal=False):
    if diagonal:
        return {(y + dy, x + dx) for y, x in points
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    return set(points) | {q for p in points for q in _neighbors(p)}


def _hidden_path(box, screen, gate):
    """Directed strict-interior arc; its two border crossings are not included."""
    full = _perimeter(box)
    hidden = {p for p in full if _inside(p, screen, True)}
    crossings = full & _perimeter(screen)
    _check(len(crossings) == 2 and gate in crossings, "two contour crossings required")
    starts = [p for p in _neighbors(gate) if p in hidden]
    _check(len(starts) == 1, "entry gate has no unique inward continuation")
    prev, current = gate, starts[0]
    path = []
    while current in hidden:
        _check(current not in path, "cyclic hidden arc")
        path.append(current)
        following = [p for p in _neighbors(current) if p in full and p != prev]
        _check(len(following) == 1, "contour is not a thin simple path")
        prev, current = current, following[0]
    _check(current in crossings - {gate} and set(path) == hidden,
           "hidden contour is not one inferable arc")
    return path


def _turn_indices(path):
    return [i for i in range(1, len(path) - 1)
            if (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) !=
               (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])]


def _side(gate, screen):
    y, x = gate
    t, b, l, r = screen
    return "N" if y == t else "S" if y == b else "W" if x == l else "E"


def _read_input(value):
    g = _grid(value)
    bg = int(g[0, 0])
    _check(all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))),
           "background corner convention")
    # The fill has its own local color and is the only solid color rectangle
    # with both dimensions >=7. Objects have area <=8; contours are thin.
    candidates = []
    for color in sorted(int(c) for c in np.unique(g) if c != bg):
        points = {tuple(map(int, p)) for p in np.argwhere(g == color)}
        t, b, l, r = _bbox(points)
        if b - t + 1 < 7 or r - l + 1 < 7 or len(points) != (b - t + 1) * (r - l + 1):
            continue
        if t == 0 or l == 0 or b == g.shape[0] - 1 or r == g.shape[1] - 1:
            continue
        screen = (t - 1, b + 1, l - 1, r + 1)
        boundary = _perimeter(screen)
        histogram = Counter(int(g[p]) for p in boundary)
        frame, amount = histogram.most_common(1)[0]
        if frame not in (bg, color) and 1 <= len(boundary) - amount <= 3:
            candidates.append((screen, color, frame))
    _check(len(candidates) == 1, "screen is not uniquely identifiable")
    screen, fill, frame = candidates[0]
    gates = {}
    for p in sorted(_perimeter(screen)):
        c = int(g[p])
        if c != frame:
            _check(c not in (bg, fill) and c not in gates, "one gate per category")
            gates[c] = p
    _check(1 <= len(gates) <= 3, "category count")
    tracks, objects, full_used = [], [], set()
    for color in sorted(int(c) for c in np.unique(g) if c not in (bg, fill, frame)):
        exterior = {tuple(map(int, p)) for p in np.argwhere(g == color)
                    if not _inside(tuple(map(int, p)), screen)}
        groups = _groups(exterior)
        long = [p for p in groups if len(p) > 8]
        if color in gates:
            _check(len(long) == 1, "category must have one long exposed contour")
            visible = long[0]
            box = _bbox(visible)
            _check(box[1] - box[0] >= 3 and box[3] - box[2] >= 3, "contour dimensions")
            full = _perimeter(box)
            _check(visible == {p for p in full if not _inside(p, screen)},
                   "exposed contour does not determine a rectangle")
            _check(not (full & _halo(full_used)), "complete contours touch or intersect")
            full_used |= full
            path = _hidden_path(box, screen, gates[color])
            small = [p for p in groups if p is not visible]
            _check(all(1 <= len(p) <= 8 for p in small), "object size")
            count = len(small)
            _check(count <= (len(path) - 1) // 2, "tally exceeds hidden path capacity")
            tracks.append(dict(color=color, box=box, gate=gates[color], path=path,
                               visible=visible, count=count,
                               area=sum(map(len, small)), areas=tuple(map(len, small))))
        else:
            _check(not long, "unmatched objects cannot be long contours")
            small = groups
        objects.extend((color, p) for p in small)
    # Objects are wholly outside the closed screen and have a background moat,
    # including between differently colored objects and against every contour.
    occupied = set()
    for color, cells in objects:
        _check(1 <= len(cells) <= 8, "object area")
        _check(not (cells & _halo(full_used, True)), "object touches a contour")
        _check(not (cells & _halo(occupied, True)), "objects touch")
        _check(not any(_inside(q, screen) for q in _halo(cells, True)), "object touches screen")
        occupied |= cells
    t, b, l, r = screen
    out = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
    for track in tracks:
        for i in range(track['count']):
            y, x = track['path'][1 + 2 * i]
            out[y - t - 1, x - l - 1] = track['color']
    return out, dict(background=bg, fill=fill, frame=frame, screen=screen,
                     tracks=tracks, objects=objects)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the queried grid; neither training examples nor seed are needed."""
    return _read_input(input_grid)[0]


def _rand(rng, low, high):
    return int(rng.integers(low, high + 1))


def _random_box(rng, screen, shape, kind, corner=None):
    t, b, l, r = screen
    h, w = shape
    if kind == 'L':
        corner = _rand(rng, 0, 3) if corner is None else corner
        north, west = corner < 2, corner % 2 == 0
        iy, ix = _rand(rng, t + 2, b - 2), _rand(rng, l + 2, r - 2)
        oy = _rand(rng, 1, t - 2) if north else _rand(rng, b + 2, h - 2)
        ox = _rand(rng, 1, l - 2) if west else _rand(rng, r + 2, w - 2)
        return min(iy, oy), max(iy, oy), min(ix, ox), max(ix, ox)
    if rng.random() < .5:
        iy = _rand(rng, t + 2, b - 2)
        oy = _rand(rng, 1, t - 2) if rng.random() < .5 else _rand(rng, b + 2, h - 2)
        return min(iy, oy), max(iy, oy), _rand(rng, 1, l - 2), _rand(rng, r + 2, w - 2)
    ix = _rand(rng, l + 2, r - 2)
    ox = _rand(rng, 1, l - 2) if rng.random() < .5 else _rand(rng, r + 2, w - 2)
    return _rand(rng, 1, t - 2), _rand(rng, b + 2, h - 2), min(ix, ox), max(ix, ox)


def _gate_choices(box, screen, count, must_turn, must_show_phase=False):
    full = _perimeter(box)
    visible = {p for p in full if not _inside(p, screen)}
    if len(visible) <= 8 or _bbox(visible) != box or len(_groups(visible)) != 1:
        return []
    options = []
    for gate in sorted(full & _perimeter(screen)):
        try:
            path = _hidden_path(box, screen, gate)
        except AssertionError:
            continue
        turns = _turn_indices(path)
        if count > (len(path) - 1) // 2:
            continue
        if must_turn and not (turns and count >= 2 and 1 <= turns[0] < 2 * count - 1):
            continue
        if must_show_phase and (not turns or turns[0] % 2 != 0):
            continue
        options.append((gate, path))
    return options


def _geometry(rng, counts, demo, must_turn, fallback=False):
    n = len(counts)
    if fallback:
        shape, screen = (30, 30), (7, 22, 7, 22)
        if demo == 0:
            boxes = [(2, 10, 2, 14), (19, 27, 15, 27)]
        elif demo == 1:
            boxes = [(13, 27, 1, 28), (3, 10, 3, 14)]
        else:
            boxes = [(5, 15, 5, 15), (3, 18, 3, 18), (1, 21, 1, 21)][:n]
        tracks = []
        for i, box in enumerate(boxes):
            choices = _gate_choices(box, screen, counts[i], i in must_turn,
                                    demo == 1 and i in must_turn)
            if not choices:
                return None
            gate, path = choices[_rand(rng, 0, len(choices) - 1)]
            tracks.append(dict(box=box, gate=gate, path=path))
        return shape, screen, tracks
    compact = demo == 2 and n <= 2 and max(counts) <= 3 and rng.random() < .45
    low, inner_low = (19, 7) if compact else (23, 11)
    h, w = _rand(rng, low, 30), _rand(rng, low, 30)
    oh, ow = _rand(rng, inner_low, min(17, h - 10)), _rand(rng, inner_low, min(17, w - 10))
    t, l = _rand(rng, 4, h - oh - 6), _rand(rng, 4, w - ow - 6)
    screen = t, t + oh + 1, l, l + ow + 1
    kinds = ['L'] * n
    corners = [None] * n
    if demo == 0:
        corners = [0, 3]
    elif demo == 1:
        kinds[0] = 'straight'
    elif rng.random() < .45:
        kinds[_rand(rng, 0, n - 1)] = 'straight'
    # Sampling permits nested and nonconcentric loops, not just four corner slots.
    tracks, occupied = [None] * n, set()
    for i in sorted(range(n), key=lambda j: -counts[j]):
        for _ in range(220):
            box = _random_box(rng, screen, (h, w), kinds[i], corners[i])
            full = _perimeter(box)
            if full & _halo(occupied):
                continue
            choices = _gate_choices(box, screen, counts[i], i in must_turn,
                                    demo == 1 and i in must_turn)
            if not choices:
                continue
            gate, path = choices[_rand(rng, 0, len(choices) - 1)]
            tracks[i] = dict(box=box, gate=gate, path=path)
            occupied |= full
            break
        else:
            return None
    return (h, w), screen, tracks


def _shape(rng, area=None, compact=False):
    area = _rand(rng, 1, 4 if compact else 8) if area is None else area
    kind = _rand(rng, 0, 3)
    if kind == 0 and not compact:
        cells = {(0, x) for x in range(area)}
    elif kind == 1 or compact:
        width = min(area, _rand(rng, 2, 3))
        cells = {(i // width, i % width) for i in range(area)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            edge = sorted({q for p in cells for q in _neighbors(p)} - cells)
            cells.add(edge[_rand(rng, 0, len(edge) - 1)])
    for _ in range(_rand(rng, 0, 3)):
        cells = {(x, -y) for y, x in cells}
    if rng.random() < .5:
        cells = {(y, -x) for y, x in cells}
    y0, _, x0, _ = _bbox(cells)
    return tuple(sorted((y - y0, x - x0) for y, x in cells))


def _place_objects(rng, grid, screen, tracks, specs):
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[[0, -1], :] = True
    blocked[:, [0, -1]] = True
    t, b, l, r = screen
    blocked[max(0, t - 1):b + 2, max(0, l - 1):r + 2] = True
    for track in tracks:
        for y, x in _halo(_perimeter(track['box']), True):
            if 0 <= y < h and 0 <= x < w:
                blocked[y, x] = True
    order = sorted(range(len(specs)), key=lambda i: (-len(specs[i][1]), float(rng.random())))
    objects = []
    for i in order:
        color, cells = specs[i]
        sh, sw = max(y for y, x in cells) + 1, max(x for y, x in cells) + 1
        legal = np.ones((h - sh + 1, w - sw + 1), dtype=bool)
        for dy, dx in cells:
            legal &= ~blocked[dy:dy + h - sh + 1, dx:dx + w - sw + 1]
        ys, xs = np.nonzero(legal)
        if not len(ys):
            return False
        pick = _rand(rng, 0, len(ys) - 1)
        origin = int(ys[pick]), int(xs[pick])
        placed = {(origin[0] + y, origin[1] + x) for y, x in cells}
        objects.append((color, placed))
        for p in placed:
            grid[p] = color
        for y, x in _halo(placed, True):
            if 0 <= y < h and 0 <= x < w:
                blocked[y, x] = True
    return objects


def _make_pair(rng, bg, frame, fill, colors, distractors, counts, forced, demo, turns):
    for attempt in range(33):
        fallback = attempt >= 25
        geometry = _geometry(rng, counts, demo, turns, fallback)
        if geometry is None:
            continue
        shape, screen, tracks = geometry
        specs = [(color, _shape(rng, forced.get(i, [None] * count)[j], fallback))
                 for i, (color, count) in enumerate(zip(colors, counts)) for j in range(count)]
        specs += [(color, _shape(rng, compact=fallback)) for color in distractors]
        grid = np.full(shape, bg, dtype=np.int8)
        for color, track in zip(colors, tracks):
            for p in _perimeter(track['box']):
                grid[p] = color
        t, b, l, r = screen
        grid[t:b + 1, l:r + 1] = frame
        grid[t + 1:b, l + 1:r] = fill
        for color, track in zip(colors, tracks):
            grid[track['gate']] = color
        placed = _place_objects(rng, grid, screen, tracks, specs)
        if placed is False:
            continue
        if demo < 2 and not any(color == category and
                               all(not _inside(p, track['box']) for p in cells)
                               for color, cells in placed
                               for category, track in zip(colors, tracks)):
            continue
        # Render labels from requested object counts, not from solve().
        out = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
        for color, track, count in zip(colors, tracks, counts):
            for j in range(count):
                y, x = track['path'][1 + 2 * j]
                out[y - t - 1, x - l - 1] = color
        rotation, reflect = _rand(rng, 0, 3), bool(rng.integers(2))
        grid, out = np.rot90(grid, rotation), np.rot90(out, rotation)
        if reflect:
            grid, out = np.fliplr(grid), np.fliplr(out)
        return {'input': grid.copy(), 'output': out.copy()}
    raise AssertionError('bounded scene construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    num_train, num_test = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and num_train > num_test')
    rng = np.random.default_rng(seed)
    anchor = _rand(rng, 0, 9)
    novel = int(rng.choice([c for c in range(10) if c != anchor]))
    backgrounds = [int(c) for c in rng.permutation([c for c in range(10) if c != anchor])]
    puzzle, seen = {'train': [], 'test': []}, set()
    role_switch = None
    for index in range(num_train + num_test):
        train = index < num_train
        demo = index if index < 2 else 2
        n = 2 if index < 2 else _rand(rng, 1, 2) if train else 3
        counts = [_rand(rng, 0 if num_train >= 3 else 1, 4 if train else 5) for _ in range(n)]
        forced, turns = {}, set()
        if index == 0:
            counts, forced, turns = [1, 3], {0: [6], 1: [1, 1, 1]}, {1}
        elif index == 1:
            counts, forced, turns = [2, 3], {0: [2, 4], 1: [1, 2, 3]}, {1}
        elif train and index == 2:
            counts[0] = 0
        elif not train:
            counts[0] = 5
            turns = {0}
        bg = backgrounds[index % len(backgrounds)] if index < 2 else int(rng.choice(
            [c for c in range(10) if not (not train and index == num_train and c == novel)]))
        pool = [int(c) for c in rng.permutation([c for c in range(10) if c != bg])]
        if index < 2:
            pool.remove(anchor)
            if index == 1:
                pool.remove(role_switch)
                colors = [anchor, role_switch]
            else:
                second = next(c for c in reversed(pool) if c != novel)
                pool.remove(second)
                colors = [anchor, second]
        else:
            colors = []
            if not train and index == num_train:
                pool.remove(novel)
                colors.append(novel)
            while len(colors) < n:
                color = next(c for c in reversed(pool) if not train or c != novel)
                pool.remove(color)
                colors.append(color)
        frame, fill = pool.pop(), pool.pop()
        distractors = [int(rng.choice(pool)) for _ in range(_rand(rng, 1 if index < 2 else 0, 3))]
        if index == 0:
            role_switch = int(rng.choice([c for c in pool if c not in (backgrounds[1], novel)]))
            distractors[0] = role_switch
        for _ in range(5):
            pair = _make_pair(rng, bg, frame, fill, colors, distractors, counts, forced, demo, turns)
            key = (pair['input'].shape, pair['input'].tobytes())
            if key not in seen:
                break
        else:
            raise AssertionError('distinct scene retries exhausted')
        seen.add(key)
        puzzle['train' if train else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, details = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = (inp.shape, inp.astype(np.int8).tobytes())
            _check(key not in seen, 'duplicate input')
            seen.add(key)
            predicted, detail = _read_input(inp)
            _check(np.array_equal(predicted, out), 'wrong contour census output')
            _check(int(np.count_nonzero(out != detail['fill'])) == sum(t['count'] for t in detail['tracks']),
                   'dots lost or conflated')
            details[split].append(detail)
    training = details['train']
    witnesses = [(t['color'], t['count'], t['area']) for d in training for t in d['tracks']]
    _check(any(c == cc and a == aa and n != nn for c, n, a in witnesses for cc, nn, aa in witnesses),
           'missing same-color/area different-count witness')
    _check(any(n == nn and a != aa for c, n, a in witnesses for cc, nn, aa in witnesses),
           'missing same-count different-area witness')
    _check({1, 2, 3} <= {n for c, n, a in witnesses}, 'unary progression not taught')
    _check(len({d['background'] for d in training}) >= 2, 'local background roles not taught')
    _check(any(not _turn_indices(t['path']) for d in training for t in d['tracks']),
           'straight continuation not taught')
    _check(any(_turn_indices(t['path']) and _turn_indices(t['path'])[0] % 2 == 0 and
               _turn_indices(t['path'])[0] < 2 * t['count'] - 1
               for d in training for t in d['tracks']), 'corner pitch continuity not taught')
    _check(len({_side(t['gate'], d['screen']) for d in training for t in d['tracks']}) >= 2,
           'entry direction not varied')
    for d in training[:2]:
        _check(any(c == t['color'] and all(not _inside(p, t['box']) for p in cells)
                   for c, cells in d['objects'] for t in d['tracks']),
               'training does not distinguish color matching from enclosure')
        _check(any(c not in {t['color'] for t in d['tracks']} for c, cells in d['objects']),
               'unmatched distractor erasure not demonstrated')
    cats = {t['color'] for d in training for t in d['tracks']}
    discards = {c for d in training for c, cells in d['objects']
                if c not in {t['color'] for t in d['tracks']}}
    _check(bool(cats & discards), 'a local unmatched-to-category role switch is missing')
    _check(any(t['color'] not in cats for d in details['test'] for t in d['tracks']),
           'test must include a new local category')
    for d in training[:2] + details['test']:
        _check(any(_turn_indices(t['path']) and t['count'] >= 2 and
                   1 <= _turn_indices(t['path'])[0] < 2 * t['count'] - 1 for t in d['tracks']),
               'no counted dot continues around a hidden corner')
    for d in details['test']:
        _check(len(d['tracks']) == 3 and max(t['count'] for t in d['tracks']) == 5,
               'test composition/extrapolation shift missing')


def _shortcut_outputs(d):
    """Explicit concept omissions, used for checks rather than label generation."""
    t, b, l, r = d['screen']
    by_area = np.full((b - t - 1, r - l - 1), d['fill'], dtype=np.int8)
    straight = by_area.copy()
    reset = by_area.copy()
    for track in d['tracks']:
        path, color = track['path'], track['color']
        for i in range(min(track['area'], (len(path) - 1) // 2)):
            y, x = path[1 + 2 * i]
            by_area[y - t - 1, x - l - 1] = color
        gate = track['gate']
        dy, dx = path[0][0] - gate[0], path[0][1] - gate[1]
        for i in range(track['count']):
            y, x = gate[0] + (2 + 2 * i) * dy, gate[1] + (2 + 2 * i) * dx
            if _inside((y, x), d['screen'], True):
                straight[y - t - 1, x - l - 1] = color
        turns = _turn_indices(path)
        indices = list(range(1, len(path) - 1, 2))
        if turns:
            corner = turns[0]
            indices = list(range(1, corner + 1, 2)) + list(range(corner + 2, len(path) - 1, 2))
        for index in indices[:track['count']]:
            y, x = path[index]
            reset[y - t - 1, x - l - 1] = color
    return by_area, straight, reset


def _metamorphic_checks():
    """Two visible edits test area invariance and input-encoded gate direction."""
    p = generate_puzzle(0, 2, 1)
    pair = p['train'][1]
    out, d = _read_input(pair['input'])
    straight = next(t for t in d['tracks'] if not _turn_indices(t['path']))
    g = pair['input'].copy()
    old_gate = straight['gate']
    new_gate = next(iter((_perimeter(straight['box']) & _perimeter(d['screen'])) - {old_gate}))
    g[old_gate], g[new_gate] = d['frame'], straight['color']
    expected = out.copy()
    expected[expected == straight['color']] = d['fill']
    top, bottom, left, right = d['screen']
    for i in range(straight['count']):
        y, x = straight['path'][::-1][1 + 2 * i]
        expected[y - top - 1, x - left - 1] = straight['color']
    _check(not np.array_equal(out, expected) and np.array_equal(solve(g), expected),
           'moving the visible gate did not reverse the tally')
    # Enlarge one object while preserving connectivity, separation and color.
    # Its component count, hence the entire output, must remain unchanged.
    for color, cells in d['objects']:
        if color not in {t['color'] for t in d['tracks']} or len(cells) >= 8:
            continue
        for q in sorted({q for cell in cells for q in _neighbors(cell)} - cells):
            y, x = q
            if not (0 < y < g.shape[0] - 1 and 0 < x < g.shape[1] - 1):
                continue
            if int(pair['input'][q]) != d['background']:
                continue
            if any(_inside(v, d['screen']) for v in _halo({q}, True)):
                continue
            others = {tuple(map(int, v)) for v in np.argwhere(pair['input'] != d['background'])} - cells
            if _halo({q}, True) & others:
                continue
            enlarged = pair['input'].copy()
            enlarged[q] = color
            _check(np.array_equal(solve(enlarged), out), 'object area changed the count')
            return
    raise AssertionError('no area-invariance edit found')


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain scalar sets and hashes, never all puzzle payloads."""
    signatures, structures, shapes, backgrounds, sides, masks, counts = (set() for _ in range(7))
    kinds, nested, total, area_failures, straight_failures, reset_failures = set(), 0, 0, 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _digest(puzzle)
        _check(signature == _digest(generate_puzzle(seed)), 'nondeterministic puzzle')
        signatures.add(signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                g = pair['input']
                solved, d = _read_input(g)
                area_shortcut, straight_shortcut, reset_shortcut = _shortcut_outputs(d)
                area_failures += not np.array_equal(solved, area_shortcut)
                straight_failures += not np.array_equal(solved, straight_shortcut)
                reset_failures += not np.array_equal(solved, reset_shortcut)
                if split == 'train' and pair is puzzle['train'][1]:
                    _check(not np.array_equal(solved, area_shortcut), 'area omission survived teaching example')
                    _check(not np.array_equal(solved, reset_shortcut), 'corner reset survived teaching example')
                if split == 'test' or pair is puzzle['train'][0] or pair is puzzle['train'][1]:
                    _check(not np.array_equal(solved, straight_shortcut), 'geometry omission survived corner witness')
                total += 1
                shapes.add(g.shape)
                backgrounds.add(d['background'])
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in g.flat)
                structures.add(hashlib.sha256(bytes(g.shape) + norm).hexdigest())
                for t in d['tracks']:
                    sides.add(_side(t['gate'], d['screen']))
                    kinds.add('L' if _turn_indices(t['path']) else 'straight')
                    counts.add(t['count'])
                    a, b, c, e = t['box']
                    nested += sum(a < tt['box'][0] < tt['box'][1] < b and
                                  c < tt['box'][2] < tt['box'][3] < e for tt in d['tracks'])
                for color, group in d['objects']:
                    y0, _, x0, _ = _bbox(group)
                    masks.add(tuple(sorted((y - y0, x - x0) for y, x in group)))
    _check(len(signatures) == 42, 'repeated puzzles across seeds')
    _check(len(structures) >= 180 and len(shapes) >= 35, 'insufficient geometric variation')
    _check(backgrounds == set(range(10)) and sides == set('NSEW'), 'palette/direction coverage')
    _check(kinds == {'L', 'straight'} and nested >= 2, 'contour structure coverage')
    _check(counts >= set(range(6)) and len(masks) >= 60, 'object/count coverage')
    _metamorphic_checks()
    for nt in range(2, 11):
        for nq in range(1, min(nt - 1, 4) + 1):
            p = generate_puzzle(20261023 + nt * 10 + nq, nt, nq)
            _check(len(p['train']) == nt and len(p['test']) == nq, 'explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    bad = generate_puzzle(0, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a corrupted output')
    return dict(seeds=42, pairs=total, distinct_puzzles=len(signatures),
                color_normalized_inputs=len(structures), input_shapes=len(shapes),
                object_masks=len(masks), nested_pairs=nested,
                backgrounds=sorted(backgrounds), entry_sides=sorted(sides), counts=sorted(counts),
                area_shortcut_failures=area_failures, straight_shortcut_failures=straight_failures,
                corner_reset_failures=reset_failures)
