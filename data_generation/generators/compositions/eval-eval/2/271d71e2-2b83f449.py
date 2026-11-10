"""Fuel races on rewired tracks. Standard library and NumPy only.

No files, external examples, latent puzzle state, or RNG are used by solve.
Generation uses bounded rejection, keeps only one puzzle at a time, and writes
no diagnostics or execution traces. See task_spec.md for the complete rule.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 992643592, 797189813, 696193386, 339444278, 83099831,
              161461299, 521263189, 418248743, 241742982, 183408499,
              380642029, 284216724, 549527742, 769531477, 275790578,
              138607007, 795707694, 821773927, 308076931, 395703980,
              445898806, 829321515, 378391839, 391601944, 573805433,
              997465905, 905596052, 206844301, 644100516, 780446435,
              601992060, 984978359, 123372302, 2575308, 473287710,
              705550053, 869263455, 884456826, 831779277, 705258330, 84877620)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid dimensions')
    _need(np.issubdtype(a.dtype, np.integer), 'noninteger grid')
    _need(bool(np.all((0 <= a) & (a <= 9))), 'invalid colors')
    return a.astype(np.int8, copy=False)


def _neighbors(mask, p):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in _DIRS
            if 0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1]
            and mask[r + dr, c + dc]]


def _components(mask):
    seen = set()
    result = []
    for raw in np.argwhere(mask):
        p = tuple(map(int, raw))
        if p in seen:
            continue
        component, pending = [], [p]
        seen.add(p)
        while pending:
            q = pending.pop()
            component.append(q)
            for z in _neighbors(mask, q):
                if z not in seen:
                    seen.add(z)
                    pending.append(z)
        result.append(component)
    return result


def _paths(mask):
    paths = []
    for component in _components(mask):
        ends = [p for p in component if len(_neighbors(mask, p)) == 1]
        _need(len(ends) == 2 and all(len(_neighbors(mask, p)) in (1, 2)
                                   for p in component), 'not a simple open path')
        p, end = sorted(ends)
        path, previous = [p], None
        while p != end:
            nxt = [q for q in _neighbors(mask, p) if q != previous]
            _need(len(nxt) == 1, 'ambiguous path')
            previous, p = p, nxt[0]
            path.append(p)
        _need(len(path) == len(component), 'untraced path cells')
        paths.append(path)
    return paths


def _switch(core, bg, track, fuel):
    """Apply all writes simultaneously, including agreeing overlaps."""
    h, w = core.shape
    _need(h >= 5 and h % 2 == 1 and w >= 7, 'invalid lane geometry')
    _need(bool(np.all(np.isin(core[::2], [bg, track]))), 'invalid lanes')
    _need(bool(np.all(core[::2, [0, w - 1]] == track)), 'missing outer port')
    _need(bool(np.all(np.isin(core[1::2], [bg, fuel]))), 'invalid separators')
    writes, centers = {}, []
    for r in range(1, h, 2):
        cols = set(map(int, np.flatnonzero(core[r] == fuel)))
        for start in sorted(c for c in cols if c - 1 not in cols):
            _need({start, start + 1, start + 2} <= cols and start + 3 not in cols,
                  'switch must be exactly three cells')
            c = start + 1
            _need(2 <= c <= w - 3, 'switch too close to edge')
            _need(bool(np.all(core[[r - 1, r + 1], c - 1:c + 2] == track)),
                  'switch touches a break')
            centers.append((r, c))
            for y in range(r - 1, r + 2):
                for x in range(c - 1, c + 2):
                    value = fuel if x == c else track
                    _need((y, x) not in writes or writes[y, x] == value,
                          'conflicting local switch writes')
                    writes[y, x] = value
    _need(bool(centers), 'no switches')
    out = core.copy()
    for p, value in writes.items():
        out[p] = value
    paths = _paths(out == track)
    for path in paths:
        for r, c in (path[0], path[-1]):
            _need(r % 2 == 0 and (c in (0, w - 1) or
                  core[r, c - 1] == bg or core[r, c + 1] == bg), 'hidden terminal')
    return out, paths, centers, set(writes)


def _counter_cells(box, side):
    """A's travel-up raster order, rotated toward the track port."""
    t, b, l, r = box
    if side == 'left':  # reservoir left of network: travel right
        return [(y, x) for x in range(r - 1, l, -1) for y in range(t + 1, b)]
    return [(y, x) for x in range(l + 1, r) for y in range(b - 1, t, -1)]


def _rectangles(a, bg):
    boxes = []
    for color in sorted(set(map(int, np.unique(a))) - {bg}):
        cells = list(map(tuple, np.argwhere(a == color)))
        ys, xs = zip(*cells)
        t, b, l, r = map(int, (min(ys), max(ys), min(xs), max(xs)))
        if b - t < 2 or r - l < 2:
            continue
        border = {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)
                  if y in (t, b) or x in (l, r)}
        missing = border - set(cells)
        openings = {((t + b) // 2, l), ((t + b) // 2, r),
                    (t, (l + r) // 2), (b, (l + r) // 2)}
        if len(missing) == 1 and missing <= openings and set(cells) == border - missing:
            boxes.append(dict(box=(t, b, l, r), color=color, opening=next(iter(missing))))
    return boxes


def _parse_canonical(a):
    bg = int(a[0, 0])
    _need(bool(np.all(a[0] == bg) and np.all(a[-1] == bg) and
               np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)), 'missing background rim')
    stations = _rectangles(a, bg)
    _need(1 <= len(stations) <= 7, 'invalid reservoir count')
    stripped = a.copy()
    cover = np.zeros(a.shape, dtype=bool)
    for s in stations:
        t, b, l, r = s['box']
        _need(b - t in (2, 4) and 2 <= r - l <= 8, 'invalid counter shape')
        _need(not bool(np.any(cover[t:b + 1, l:r + 1])), 'overlapping reservoirs')
        cover[t:b + 1, l:r + 1] = True
        stripped[t:b + 1, l:r + 1] = bg
    loc = np.argwhere(stripped != bg)
    _need(len(loc) > 0, 'no network')
    t, l = map(int, loc.min(axis=0))
    b, r = map(int, loc.max(axis=0))
    core = stripped[t:b + 1, l:r + 1]
    _need(not bool(np.any(cover[t:b + 1, l:r + 1])), 'reservoir covers network')
    colors = set(map(int, np.unique(core))) - {bg}
    _need(len(colors) == 2, 'network needs track and switch colors')
    # Separator geometry identifies the ready/fuel role; it is not learned.
    possible = set(map(int, np.unique(core[1::2]))) - {bg}
    _need(len(possible) == 1, 'ambiguous switch color')
    fuel = next(iter(possible))
    track = next(iter(colors - {fuel}))
    structural, paths, centers, writes = _switch(core, bg, track, fuel)
    ends = {p for path in paths for p in (path[0], path[-1])}
    ports = set()
    for s in stations:
        y0, y1, x0, x1 = s['box']
        cy = (y0 + y1) // 2
        if x1 == l - 1:
            side, port = 'left', (cy - t, 0)
        elif x0 == r + 1:
            side, port = 'right', (cy - t, r - l)
        else:
            raise AssertionError('reservoir not attached to a side port')
        opening = (cy, x1 if side == 'left' else x0)
        _need(s['opening'] == opening and a[opening] == track, 'missing inward launch opening')
        _need(port in ends and port not in ports, 'missing or shared port')
        ports.add(port)
        coords = _counter_cells(s['box'], side)
        values = [int(a[p]) for p in coords]
        spent = values.count(bg)
        _need(values == [bg] * spent + [fuel] * (len(values) - spent),
              'counter not a direction-relative spent prefix')
        s.update(side=side, port=port, coords=coords, spent=spent,
                 budget=len(coords) - spent, capacity=len(coords))
    return dict(a=a, bg=bg, track=track, fuel=fuel, stations=stations,
                bounds=(t, b, l, r), core=core, structural=structural,
                paths=paths, centers=centers, writes=writes)


def _parse(inp):
    a = _grid(inp)
    found = []
    # Half-turn equivalents need not be tried: left and right ports are explicit.
    for k in (0, 1):
        try:
            info = _parse_canonical(np.rot90(a, k).copy())
            info['rotation'] = k
            found.append(info)
        except (AssertionError, ValueError, IndexError, StopIteration):
            pass
    _need(len(found) == 1, 'input needs one geometric interpretation')
    return found[0]


def _allocation(paths, stations, unlimited=False):
    """Arrival distances include the first step from a frame to its port."""
    by_port = {s['port']: s for s in stations}
    claims = {s['color']: [] for s in stations}
    neutral = []
    for path in paths:
        n = len(path)
        sa, sb = by_port.get(path[0]), by_port.get(path[-1])
        fa = (n if unlimited else sa['budget']) if sa else 0
        fb = (n if unlimited else sb['budget']) if sb else 0
        for i, p in enumerate(path):
            da, db = i + 1, n - i
            ra, rb = da <= fa, db <= fb
            if ra and (not rb or da < db):
                claims[sa['color']].append(p)
            elif rb and (not ra or db < da):
                claims[sb['color']].append(p)
            elif ra and rb:
                neutral.append(p)
    return claims, neutral


def _render(info, claims):
    out = info['a'].copy()
    t, b, l, r = info['bounds']
    core = info['structural'].copy()
    for color, cells in claims.items():
        for p in cells:
            core[p] = color
    out[t:b + 1, l:r + 1] = core
    for s in info['stations']:
        used = len(claims[s['color']])
        _need(used <= s['budget'], 'fuel overspent')
        for i, p in enumerate(s['coords']):
            out[p] = info['bg'] if i < s['spent'] + used else info['fuel']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    claims, _ = _allocation(info['paths'], info['stations'])
    return np.rot90(_render(info, claims), -info['rotation']).copy()


def _topology(rng, lanes, width):
    """Sample throughout the network, then open cycles at legal lane breaks."""
    core = np.zeros((2 * lanes - 1, width), dtype=np.int8)
    core[::2] = 1
    centers, previous = [], []
    for r in range(1, core.shape[0], 2):
        wanted = int(rng.choice((0, 1, 2), p=(.22, .57, .21)))
        chosen = []
        for x in rng.permutation(np.arange(2, width - 2)):
            c = int(x)
            if len(chosen) == wanted:
                break
            if all(abs(c - q) >= 4 for q in chosen) and all(abs(c - q) >= 2 for q in previous):
                chosen.append(c)
                core[r, c - 1:c + 2] = 2
                centers.append((r, c))
        previous = chosen
    if not centers:
        return None
    structural = core.copy()
    for r, c in centers:
        structural[r - 1:r + 2, c - 1:c + 2] = 1
        structural[r - 1:r + 2, c] = 2
    mask = structural == 1
    if any(len(_neighbors(mask, p)) not in (1, 2)
           for raw in np.argwhere(mask) for p in [tuple(map(int, raw))]):
        return None
    safe = [(r, c) for r in range(0, core.shape[0], 2) for c in range(2, width - 2)
            if all(abs(r - y) != 1 or abs(c - x) >= 2 for y, x in centers)]
    # A cycle needs exactly one break. No execution trace is retained.
    for component in _components(mask):
        if not any(len(_neighbors(mask, p)) == 1 for p in component):
            pool = sorted(set(safe) & set(component))
            if not pool:
                return None
            p = pool[int(rng.integers(len(pool)))]
            core[p] = 0
            mask[p] = False
    for _ in range(int(rng.choice((0, 0, 1, 2)))):
        pool = [p for p in safe if mask[p] and len(_neighbors(mask, p)) == 2
                and all(core[p[0], x] != 0 for x in range(p[1] - 1, p[1] + 2))]
        if pool:
            p = pool[int(rng.integers(len(pool)))]
            core[p] = 0
            mask[p] = False
    try:
        _, paths, _, _ = _switch(core, 0, 1, 2)
    except AssertionError:
        return None
    return core, paths


def _compatible_port(port, selected, width, halfheight=2):
    # Every sampled box fits even at maximal five-cell height.
    return port[1] in (0, width - 1) and all(
        port[1] != q[1] or abs(port[0] - q[0]) >= 2 * halfheight + 2
        for q in selected)


def _make_scene(rng, lesson, is_test, palette):
    lanes = int(rng.integers(8, 12) if is_test else rng.integers(5, 8))
    width = int(rng.choice((7, 9)) if lesson == 'collision' else
                rng.integers(7, 11) if lesson == 'takeover' else rng.integers(7, 15))
    result = _topology(rng, lanes, width)
    if result is None:
        return None
    core, paths = result
    paired = [p for p in paths if p[0][1] in (0, width - 1) and
              p[-1][1] in (0, width - 1) and p[0][0] != p[-1][0]
              and _compatible_port(p[-1], [p[0]], width)
              and (not lesson or 9 <= len(p) <= (25 if lesson == 'collision' else 18))]
    if not paired:
        return None
    path = paired[int(rng.integers(len(paired)))]
    if lesson == 'collision' and len(path) % 2 == 0:
        return None
    ports = [path[0], path[-1]]
    path_by_end = {q: p for p in paths for q in (p[0], p[-1])}
    # A third, unopposed weak source exposes an actual fuel-limited gap.
    remaining = [q for q, p in path_by_end.items()
                 if p is not path and len(p) >= 5 and _compatible_port(q, ports, width)]
    if lesson and not remaining:
        return None
    rng.shuffle(remaining)
    if remaining and (lesson or rng.random() < .8):
        ports.append(remaining[0])
    wanted = int(rng.integers(3, 8) if is_test else rng.integers(2, 6))
    if lesson:
        wanted = max(3, wanted)
    pool = list(path_by_end)
    rng.shuffle(pool)
    for p in pool:
        if len(ports) >= wanted:
            break
        if p not in ports and _compatible_port(p, ports, width):
            # Keep the weak teaching source unopposed.
            if lesson and path_by_end[p] is path_by_end[ports[2]]:
                continue
            ports.append(p)
    max_inner_width = min(7, (28 - width) // 2 - 2)
    if max_inner_width < 2:
        return None
    capmax = 3 * max_inner_width
    budgets = [None] * len(ports)
    n = len(path)
    if lesson == 'collision':
        for j in (0, 1):
            budgets[j] = n // 2 + int(rng.integers(2, 4))
    elif lesson == 'takeover':
        short = int(rng.integers(1, min(4, n // 3) + 1))
        if n - short + 1 > capmax:
            return None
        budgets[:2] = [short, n - short + 1]
        if rng.random() < .5:
            budgets[:2] = budgets[:2][::-1]
    if lesson:
        budgets[2] = int(rng.integers(1, 4))
    specs = []
    for j, port in enumerate(ports):
        budget = budgets[j]
        if budget is not None:
            if budget > capmax:
                return None
            shapes = [(ih, iw) for ih in (1, 3) for iw in range(1, max_inner_width + 1)
                      if ih * iw >= budget]
            ih, iw = shapes[int(rng.integers(len(shapes)))]
        else:
            ih = int(rng.choice((1, 3)))
            iw = int(rng.integers(1, max_inner_width + 1))
            budget = int(rng.integers(ih * iw + 1))
        specs.append((port, ih, iw, budget))
    left = max((iw + 2 for (r, c), ih, iw, f in specs if c == 0), default=0)
    right = max((iw + 2 for (r, c), ih, iw, f in specs if c != 0), default=0)
    # Independent asymmetric margins, staying inside 30 by 30.
    extra_y = int(rng.integers(0, min(3, 30 - core.shape[0] - 6) + 1))
    top = 3 + int(rng.integers(extra_y + 1))
    height = core.shape[0] + 6 + extra_y
    extra_x = int(rng.integers(0, min(2, 28 - width - left - right) + 1))
    offset = int(rng.integers(extra_x + 1))
    x0 = left + 1 + offset
    a = np.zeros((height, width + left + right + 2 + extra_x), dtype=np.int8)
    a[top:top + core.shape[0], x0:x0 + width] = core
    owners = list(map(int, rng.permutation(np.arange(3, 10))))
    for color, (port, ih, iw, budget) in zip(owners, specs):
        row, col = port
        cy = top + row
        y0, y1 = cy - (ih + 1) // 2, cy + (ih + 1) // 2
        if col == 0:
            side, x1 = 'left', x0 - 1
            xstart = x1 - iw - 1
        else:
            side, xstart = 'right', x0 + width
            x1 = xstart + iw + 1
        a[y0:y1 + 1, xstart:x1 + 1] = color
        coords = _counter_cells((y0, y1, xstart, x1), side)
        for i, p in enumerate(coords):
            a[p] = 0 if i < ih * iw - budget else 2
        a[cy, x1 if side == 'left' else xstart] = 1
    a = np.asarray(palette, dtype=np.int8)[a]
    a = np.rot90(a, int(rng.integers(4))).copy()
    try:
        info = _parse(a)
        claims, neutral = _allocation(info['paths'], info['stations'])
        features = _features(info, claims, neutral)
        if lesson == 'collision' and not features['ties']:
            return None
        if lesson == 'takeover' and not features['takeover']:
            return None
        if lesson and not (features['fuel_limited'] and features['blocked'] and
                           features['omit_a'] and features['omit_b']):
            return None
        return dict(input=a, output=np.rot90(_render(info, claims), -info['rotation']).copy())
    except AssertionError:
        return None


def _features(info, claims, neutral):
    used = {c: len(v) for c, v in claims.items()}
    stations = info['stations']
    unlimited, _ = _allocation(info['paths'], stations, unlimited=True)
    raw_paths = _paths(info['core'] == info['track'])
    unturned, _ = _allocation(raw_paths, stations)
    def label_map(groups):
        return {p: c for c, cells in groups.items() for p in cells}
    actual, alt_a, alt_b = map(label_map, (claims, unlimited, unturned))
    unchanged = set(map(tuple, np.argwhere(info['core'] == info['track']))) - info['writes']
    takeover = False
    for p in info['paths']:
        for s in stations:
            if s['port'] in (p[0], p[-1]) and used[s['color']] > (len(p) + 1) // 2:
                takeover = True
    return dict(lanes=(info['core'].shape[0] + 1) // 2, width=info['core'].shape[1],
                stations=len(stations), switches=len(info['centers']),
                breaks=int(np.count_nonzero(info['core'][::2] == info['bg'])),
                ties=len(neutral), takeover=takeover,
                fuel_limited=any(0 < s['budget'] == used[s['color']] for s in stations),
                blocked=any(0 < used[s['color']] < s['budget'] for s in stations),
                scan_example=any(s['box'][1] - s['box'][0] == 4 and
                                 s['box'][3] - s['box'][2] >= 3 and
                                 0 < used[s['color']] < s['budget'] for s in stations),
                gap=any(p not in actual for path in info['paths'] for p in path),
                omit_a=actual != alt_a,
                omit_b=(any(actual.get(p) != alt_b.get(p) for p in unchanged) or
                        any(used[c] != len(unturned[c]) for c in used)),
                capacities=tuple(s['capacity'] for s in stations),
                budgets=tuple(s['budget'] for s in stations),
                lengths=tuple(len(p) for p in info['paths']))


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    lessons = [None] * nt
    positions = list(map(int, rng.permutation(nt)))
    lessons[positions[0]], lessons[positions[1]] = 'collision', 'takeover'
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            lesson = lessons[i] if split == 'train' else None
            for _ in range(400):
                pair = _make_scene(rng, lesson, split == 'test', palette)
                if pair is None:
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'invalid counts')
    seen, roles, training, lane_counts = set(), set(), [], {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _need(a.shape == b.shape, 'dimensions changed')
            key = (a.shape, a.tobytes())
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            info = _parse(a)
            roles.add((info['bg'], info['track'], info['fuel']))
            claims, neutral = _allocation(info['paths'], info['stations'])
            expected = np.rot90(_render(info, claims), -info['rotation'])
            _need(np.array_equal(expected, b), 'incorrect output')
            # Conservation checked against the returned grid, not only claim lists.
            canonical = np.rot90(b, info['rotation'])
            t, bottom, l, right = info['bounds']
            network_out = canonical[t:bottom + 1, l:right + 1]
            for s in info['stations']:
                spent_after = sum(canonical[p] == info['bg'] for p in s['coords'])
                _need(spent_after - s['spent'] == np.count_nonzero(network_out == s['color']),
                      'painted distance does not equal counter consumption')
            f = _features(info, claims, neutral)
            lane_counts[split].append(f['lanes'])
            if split == 'train':
                training.append(f)
    _need(len(roles) == 1, 'puzzle-level structural palette changed')
    for name in ('ties', 'takeover', 'fuel_limited', 'blocked', 'gap', 'omit_a', 'omit_b', 'scan_example'):
        _need(any(f[name] for f in training), 'training lacks ' + name)
    _need(min(lane_counts['test']) > max(lane_counts['train']), 'missing network-size shift')


def _tick_allocation(paths, stations):
    """A bounded tick simulation cross-checks the closed arrival formula."""
    by_port = {s['port']: s for s in stations}
    claims = {s['color']: [] for s in stations}
    for path in paths:
        left, right = 0, len(path) - 1
        sa, sb = by_port.get(path[0]), by_port.get(path[-1])
        fa, fb = (sa['budget'] if sa else 0), (sb['budget'] if sb else 0)
        while left <= right and (fa or fb):
            if left == right and fa and fb:
                break
            if fa:
                claims[sa['color']].append(path[left])
                left, fa = left + 1, fa - 1
            if fb and left <= right:
                claims[sb['color']].append(path[right])
                right, fb = right - 1, fb - 1
    return claims


def self_test():
    """42 seeds, sequential; small aggregate sets only; no saved payloads."""
    shapes, structures, capacities, budgets, lengths, counts, switches, breaks = (set() for _ in range(8))
    colors_by_role = [set(), set(), set()]
    signatures = set()
    # Exhaustion, exact arrival, even/odd collision and the one-cell remainder.
    for n in range(2, 13):
        path = [(0, i) for i in range(n)]
        for fa in range(n + 2):
            for fb in range(n + 2):
                sources = [dict(port=path[0], color=3, budget=fa),
                           dict(port=path[-1], color=4, budget=fb)]
                fast, _ = _allocation([path], sources)
                tick = _tick_allocation([path], sources)
                _need(all(set(fast[c]) == set(tick[c]) for c in fast), 'boundary race mismatch')
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                info = _parse(a)
                claims, _ = _allocation(info['paths'], info['stations'])
                ticked = _tick_allocation(info['paths'], info['stations'])
                _need(all(set(claims[c]) == set(ticked[c]) for c in claims), 'tick/arrival mismatch')
                _need(np.array_equal(solve(a.copy()), pair['output']), 'input-only solve mismatch')
                f = _features(info, claims, [])
                shapes.add(a.shape)
                capacities.update(f['capacities'])
                budgets.update(f['budgets'])
                lengths.update(f['lengths'])
                counts.add(f['stations'])
                switches.add(f['switches'])
                breaks.add(f['breaks'])
                for values, name in zip(colors_by_role, ('bg', 'track', 'fuel')):
                    values.add(info[name])
                # Color-normalized structural digest, not a retained grid copy.
                mapping = {}
                normalized = bytes(mapping.setdefault(int(v), len(mapping)) for v in a.flat)
                structures.add(hashlib.sha256(bytes(a.shape) + normalized).digest())
                digest.update(bytes(a.shape) + a.tobytes() + pair['output'].tobytes())
        signatures.add(digest.digest())
    _need(len(signatures) == 42, 'seed diversity collapsed')
    _need(len(shapes) >= 35 and len(structures) >= 180, 'geometry diversity collapsed')
    _need(len(capacities) >= 10 and len(budgets) >= 12 and len(lengths) >= 20,
          'counter or route diversity collapsed')
    _need(len(counts) >= 4 and len(switches) >= 7 and len(breaks) >= 3,
          'source, switch or break diversity collapsed')
    _need(all(len(v) >= 8 for v in colors_by_role), 'palette roles insufficiently varied')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(101 + nt * 7 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    first, repeat = generate_puzzle(0), generate_puzzle(0)
    for split in ('train', 'test'):
        for p, q in zip(first[split], repeat[split]):
            _need(all(np.array_equal(p[k], q[k]) for k in p), 'nondeterminism')
    base = first['train'][0]['input']
    answer = solve(base)
    for k in range(4):
        _need(np.array_equal(solve(np.rot90(base, k)), np.rot90(answer, k)),
              'quarter-turn equivariance failed')
    # Solver coverage extends beyond the sampled two-or-more-source scenes.
    info = _parse(base)
    single = info['a'].copy()
    for s in info['stations'][1:]:
        t, b, l, r = s['box']
        single[t:b + 1, l:r + 1] = info['bg']
    for fuel_state in ('fresh', 'empty', 'partial'):
        variant = single.copy()
        source = info['stations'][0]
        for i, p in enumerate(source['coords']):
            spent = 0 if fuel_state == 'fresh' else len(source['coords']) if fuel_state == 'empty' else 1
            variant[p] = info['bg'] if i < spent else info['fuel']
        parsed = _parse(variant)
        _need(len(parsed['stations']) == 1, 'single-source parsing failed')
        ticked = _tick_allocation(parsed['paths'], parsed['stations'])
        expected = np.rot90(_render(parsed, ticked), -parsed['rotation'])
        _need(np.array_equal(solve(variant), expected), 'uniform/single counter regression')
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a changed output')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'input_shapes': len(shapes),
            'normalized_inputs': len(structures), 'capacities': sorted(capacities),
            'budgets': sorted(budgets), 'path_lengths': sorted(lengths),
            'reservoir_counts': sorted(counts), 'switch_counts': sorted(switches),
            'break_counts': sorted(breaks)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
