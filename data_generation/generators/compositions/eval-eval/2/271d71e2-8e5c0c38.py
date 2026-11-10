"""Balanced engines: reflection orbits are indivisible counter charges.

Only NumPy and the standard library are used. No files, caches, training pairs,
or RNG state are consulted by solve(). All diagnostic suites stay in memory.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


_DELTAS = ((-1, 0), (0, -1), (1, 0), (0, 1))  # quarter turns from travel-up


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "invalid grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _need(np.all((0 <= g) & (g <= 9)), "invalid color")
    return g


def _key(g):
    return g.shape, g.astype(np.uint8).tobytes()


def _counts(a, b):
    for value, lo, hi in ((a, 2, 10), (b, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("example counts must be integers")
        if not lo <= value <= hi:
            raise ValueError("example count outside legal range")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    while unseen:
        first = unseen.pop()
        component, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DELTAS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    stack.append(p)
        yield component


def _support(block, background):
    """Find a unique same-color reflection axis in a travel-up interior.

    axis2 is twice the axis coordinate. Background does not vote. Supported
    pairs of either state count as two votes; axis cells count as one vote.
    """
    width = block.shape[1]
    scores = np.zeros(2 * width - 1, dtype=int)
    for color in np.unique(block):
        if int(color) == background:
            continue
        mask = (block == color).astype(int)
        for row in mask:
            scores += np.convolve(row, row)
    winners = np.flatnonzero(scores == scores.max())
    _need(scores.max() > 0 and len(winners) == 1, "counter has a tied or empty symmetry explanation")
    axis = int(winners[0])
    units, supported = [], set()
    for r in range(block.shape[0]):
        for c in range(width):
            reflected = axis - c
            if (block[r, c] != background and c <= reflected < width
                    and block[r, c] == block[r, reflected]):
                unit = ((r, c),) if c == reflected else ((r, c), (r, reflected))
                units.append(unit)
                supported.update(unit)
    return axis, units, supported


def _bar(t, b, l, r, turn, offset):
    if turn == 0:
        return [(t - 1 - offset, c) for c in range(l, r + 1)]
    if turn == 1:
        return [(y, l - 1 - offset) for y in range(t, b + 1)]
    if turn == 2:
        return [(b + 1 + offset, c) for c in range(l, r + 1)]
    return [(y, r + 1 + offset) for y in range(t, b + 1)]


def _inside(points, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in points)


def _exact_bar(g, points, color):
    if not _inside(points, g.shape) or any(g[p] != color for p in points):
        return False
    if points[0][0] == points[-1][0]:
        ends = ((points[0][0], points[0][1] - 1), (points[-1][0], points[-1][1] + 1))
    else:
        ends = ((points[0][0] - 1, points[0][1]), (points[-1][0] + 1, points[-1][1]))
    return all(not _inside([p], g.shape) or g[p] != color for p in ends)


def _read_engine(g, background, frame, component):
    ys, xs = zip(*component)
    t, b, l, r = min(ys), max(ys), min(xs), max(xs)
    _need(b - t >= 2 and r - l >= 2, "frame too small")
    _need(len(component) == 2 * (b - t + r - l), "not a rectangular perimeter")
    _need(all(y in (t, b) or x in (l, r) for y, x in component), "not a hollow frame")
    inner = g[t + 1:b, l + 1:r]
    _need(not np.any(inner == frame), "frame color inside counter")
    candidates = []
    for turn in range(4):
        attached = _bar(t, b, l, r, turn, 0)
        if not _inside(attached, g.shape):
            continue
        marker = int(g[attached[0]])
        if marker in (background, frame) or not _exact_bar(g, attached, marker):
            continue
        distance, destination = None, None
        for d in range(1, max(g.shape)):
            pts = _bar(t, b, l, r, turn, d)
            if not _inside(pts, g.shape):
                break
            if _exact_bar(g, pts, marker):
                distance, destination = d, pts
                break
            if any(g[p] != background for p in pts):
                break
        if distance is not None:
            candidates.append((turn, marker, distance, attached, destination))
    _need(len(candidates) == 1, "leading/destination bars ambiguous")
    turn, marker, distance, attached, destination = candidates[0]
    block = np.rot90(g[t:b + 1, l:r + 1], -turn).copy()
    inner = block[1:-1, 1:-1]
    fuel_colors = set(map(int, np.unique(inner))) - {background, marker}
    _need(len(fuel_colors) <= 1, "multiple fuel colors in one engine")
    fuel = next(iter(fuel_colors)) if fuel_colors else None
    axis, units, supported = _support(inner, background)
    states = [int(inner[u[0]]) for u in units]
    spent = states.count(marker)
    _need(states == [marker] * spent + [fuel] * (len(states) - spent), "spent units must form a prefix")
    extras = set(map(tuple, np.argwhere(inner != background).tolist())) - supported
    _need(all(inner[p] == fuel for p in extras), "only fuel may be additive noise")
    # Include every future body location, even when the engine has no fuel.
    y0 = min(t, min(y for y, _ in destination))
    y1 = max(b, max(y for y, _ in destination))
    x0 = min(l, min(x for _, x in destination))
    x1 = max(r, max(x for _, x in destination))
    return dict(t=t, b=b, l=l, r=r, turn=turn, marker=marker, frame=frame,
                fuel=fuel, distance=distance, attached=attached, destination=destination,
                block=block, axis=axis, units=units, supported=supported,
                spent=spent, remaining=len(units) - spent, extras=extras,
                bounds=(y0, y1, x0, x1))


def _analyze(input_grid):
    g = _grid(input_grid)
    freqs = np.bincount(g.ravel().astype(int), minlength=10)
    background = int(freqs.argmax())
    _need(freqs[background] > g.size / 2, "background must occupy a strict majority")
    engines = []
    for color in map(int, np.unique(g)):
        if color == background:
            continue
        for component in _components(g == color):
            try:
                engines.append(_read_engine(g, background, color, component))
            except AssertionError:
                continue
    _need(bool(engines), "no engine")
    cover = np.zeros(g.shape, dtype=int)
    sweep = np.zeros(g.shape, dtype=int)
    for e in engines:
        cover[e['t']:e['b'] + 1, e['l']:e['r'] + 1] += (g[e['t']:e['b'] + 1, e['l']:e['r'] + 1] != background)
        for p in e['attached'] + e['destination']:
            cover[p] += 1
        y0, y1, x0, x1 = e['bounds']
        sweep[y0:y1 + 1, x0:x1 + 1] += 1
    _need(np.max(cover) == 1 and np.array_equal(cover > 0, g != background), "unexplained/overlapping content")
    _need(np.max(sweep) == 1, "overlapping travel corridors")
    return g, background, sorted(engines, key=lambda e: (e['t'], e['l']))


def _transform(g, background, engines, mode="normal"):
    out = g.copy()
    for e in engines:
        out[e['t']:e['b'] + 1, e['l']:e['r'] + 1] = background
        for p in e['attached']:
            out[p] = background
    for e in engines:
        movement = min(e['remaining'], e['distance'])
        if mode == "no_motion":
            movement = 0
        block = np.full_like(e['block'], background)
        block[0] = block[-1] = e['frame']
        block[:, 0] = block[:, -1] = e['frame']
        for i, unit in enumerate(e['units']):
            color = e['marker'] if i < e['spent'] + movement else e['fuel']
            for y, x in unit:
                block[y + 1, x + 1] = color
        rotated = np.rot90(block, e['turn'])
        dy, dx = _DELTAS[e['turn']]
        t, l = e['t'] + dy * movement, e['l'] + dx * movement
        h, w = rotated.shape
        out[t:t + h, l:l + w] = rotated
        for p in _bar(t, t + h - 1, l, l + w - 1, e['turn'], 0):
            out[p] = e['marker']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive bars, states, reflection units and motion entirely from the input."""
    g, background, engines = _analyze(input_grid)
    return _transform(g, background, engines)


def _shape(rng, height, width, parity, teaching):
    """A sparse mirrored occupancy pattern; no parent generator is called."""
    widths = [n for n in range(2, width + 1) if (n - 1) % 2 == parity]
    core_width = int(rng.choice(widths))
    left = int(rng.integers(width - core_width + 1))
    axis = 2 * left + core_width - 1
    radius = core_width // 2
    odd = core_width % 2
    family = int(rng.integers(5))
    units = []
    phase = float(rng.uniform(0, 2 * np.pi))
    for row in range(height):
        if family == 0:  # hollow contour
            selected = list(range(radius)) if row in (0, height - 1) else [0]
        elif family == 1:  # moving lobes
            count = 1 + int((radius - 1) * (.5 + .5 * np.sin(phase + row * 1.5)))
            selected = list(range(radius - count, radius))
        elif family == 2:  # neck with broader ends
            count = max(1, int(round(radius * abs(2 * row - height + 1) / max(1, height - 1))))
            selected = list(range(radius - count, radius))
        else:  # irregular mirrored islands, including empty rows
            selected = [j for j in range(radius) if rng.random() < (.45 if family == 3 else .8)]
        for j in selected:
            c = left + j
            units.append(((row, c), (row, axis - c)))
        if odd and (rng.random() < .6 or row in (0, height - 1)):
            units.append(((row, axis // 2),))
    units.sort(key=lambda u: u[0])
    if teaching and (len(units) < 7 or len({u[0][0] for u in units}) < 3):
        return None
    if not units or not any(len(u) == 2 for u in units):
        return None
    if teaching and parity == 0 and not any(len(u) == 1 for u in units):
        return None
    return axis, units


def _make_tile(rng, roles, turn, regime=None, parity=None, large=False, compact=False):
    background, frame, marker, fuel = roles
    teaching = regime in ("fuel_lesson", "destination_lesson")
    for _ in range(80):
        height = int(rng.integers(3 if teaching else 1, (6 if compact else 10 if large else 7)))
        width = int(rng.integers(4 if teaching else 3, 8 if compact else 11))
        kind = int(rng.integers(2)) if parity is None else parity
        shape = _shape(rng, height, width, kind, teaching)
        if shape is None:
            continue
        axis, units = shape
        n = len(units)
        if regime == "fuel_lesson":
            remaining = int(rng.integers(2, min(5, n)))
            spent = n - remaining
            if not any(len(u) == 2 for u in units[spent:]):
                continue
            distance = remaining + int(rng.integers(1, 5))
        elif regime == "destination_lesson":
            spent = int(rng.integers(1, n - 3))
            remaining = n - spent
            allowed = [d for d in range(2, min(6, remaining))
                       if units[spent][0][0] != units[spent + d - 1][0][0]
                       and (kind == 1 or any(len(u) == 1 for u in units[spent:spent + d]))]
            if not allowed:
                continue
            distance = int(rng.choice(allowed))
        else:
            spent = int(rng.integers(n + 1))
            remaining = n - spent
            if rng.random() < .18:
                spent, remaining = 0, n
            if rng.random() < .13:
                spent, remaining = n, 0
            distance = int(rng.integers(1, 15 if large else 10 if not compact else 7))
        clean = np.full((height, width), background, dtype=np.int8)
        for i, unit in enumerate(units):
            for p in unit:
                clean[p] = marker if i < spent else fuel
        noisy = clean.copy()
        occupied = {p for unit in units for p in unit}
        candidates = [(y, x) for y in range(height) for x in range(width)
                      if (y, x) not in occupied and 2 * x != axis
                      and (y, axis - x) not in occupied]
        rng.shuffle(candidates)
        desired = int(rng.integers(1 if teaching else 0, 6 if large else 4))
        added = 0
        for p in candidates:
            if added == desired:
                break
            trial = noisy.copy()
            trial[p] = fuel
            try:
                a, _, support = _support(trial, background)
            except AssertionError:
                continue
            if a == axis and support == occupied:
                noisy = trial
                added += 1
        if teaching and added == 0:
            continue
        movement = min(remaining, distance)
        tile_in = np.full((distance + height + 3, width + 2), background, dtype=np.int8)
        tile_out = tile_in.copy()
        if max(tile_in.shape) > 29:
            continue
        for tile, top, advanced, contents in ((tile_in, distance + 1, spent, noisy),
                                               (tile_out, distance + 1 - movement, spent + movement, None)):
            tile[0, :] = marker
            tile[top - 1, :] = marker
            tile[top:top + height + 2, :] = frame
            if contents is not None:
                tile[top + 1:top + height + 1, 1:-1] = contents
            else:
                tile[top + 1:top + height + 1, 1:-1] = background
                for i, unit in enumerate(units):
                    for y, x in unit:
                        tile[top + 1 + y, 1 + x] = marker if i < advanced else fuel
        return {"input": np.rot90(tile_in, turn).copy(), "output": np.rot90(tile_out, turn).copy()}
    return None


def _positions(occupied, shape):
    h, w = shape
    H, W = occupied.shape
    if h > H or w > W:
        return []
    padded = np.pad(occupied, 1)
    forbidden = np.zeros_like(occupied)
    for dy in range(3):
        for dx in range(3):
            forbidden |= padded[dy:dy + H, dx:dx + W]
    integral = np.pad(forbidden.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(sums == 0)


def _pair(rng, roles, lessons, count, large=False):
    background = roles[0]
    for attempt in range(24):
        H = 30 if attempt >= 12 else int(rng.integers(24 if lessons else 15, 31))
        W = 30 if attempt >= 12 else int(rng.integers(24 if lessons else 15, 31))
        inp = np.full((H, W), background, dtype=np.int8)
        out = inp.copy()
        occupied = np.zeros((H, W), dtype=bool)
        descriptions = list(lessons)
        while len(descriptions) < count:
            descriptions.append((int(rng.integers(4)), None, None))
        success = True
        for turn, regime, parity in descriptions:
            placed = False
            for _ in range(24):
                tile = _make_tile(rng, roles, turn, regime, parity, large,
                                  compact=(count >= 3 or attempt >= 12))
                if tile is None:
                    continue
                positions = _positions(occupied, tile['input'].shape)
                if len(positions) == 0:
                    continue
                h, w = tile['input'].shape
                if rng.random() < .35:
                    edge = positions[(positions[:, 0] == 0) | (positions[:, 1] == 0) |
                                     (positions[:, 0] + h == H) | (positions[:, 1] + w == W)]
                    if len(edge):
                        positions = edge
                y, x = map(int, positions[int(rng.integers(len(positions)))])
                inp[y:y + h, x:x + w] = tile['input']
                out[y:y + h, x:x + w] = tile['output']
                occupied[y:y + h, x:x + w] = True
                placed = True
                break
            if not placed:
                success = False
                break
        if not success or np.count_nonzero(inp != background) * 2 >= inp.size:
            continue
        # Crop with independently chosen margins, while retaining a majority
        # background. The full swept rectangles (not only colored cells) fit.
        if rng.random() < .7:
            ys, xs = np.nonzero(occupied)
            t = int(rng.integers(int(ys.min()) + 1))
            l = int(rng.integers(int(xs.min()) + 1))
            b = int(rng.integers(int(ys.max()) + 1, H + 1))
            r = int(rng.integers(int(xs.max()) + 1, W + 1))
            crop = inp[t:b, l:r]
            if np.count_nonzero(crop != background) * 2 < crop.size:
                inp, out = crop.copy(), out[t:b, l:r].copy()
        try:
            _analyze(inp)
        except AssertionError:
            continue
        return dict(input=inp, output=out)
    raise RuntimeError("bounded scene packing failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shared = bool(rng.integers(2))
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    directions = list(map(int, rng.permutation(4)))
    # Both parity types and stopping causes are taught even at two examples.
    lessons = [[], []]
    fuel_parity = int(rng.integers(2))
    lessons[0] = [(directions[0], "fuel_lesson", fuel_parity)]
    lessons[1] = [(directions[1], "destination_lesson", 1 - fuel_parity)]
    # A second engine in one lesson shows local axes and independent counters.
    lessons[int(rng.integers(2))].append((directions[2], None, None))
    rng.shuffle(lessons)
    # One concrete generalization per puzzle: deeper counters, longer travel,
    # or combining a greater number of engines. These affect no rule choices.
    shift = str(rng.choice(("depth", "journey", "count")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for i in range(count):
            teaching = lessons[i] if split == "train" and i < 2 else []
            roles = palette if shared else tuple(map(int, rng.choice(10, 4, replace=False)))
            n = max(len(teaching), int(rng.integers(1, 3 if shift == "count" else 4)))
            if split == "test" and shift == "count":
                n = int(rng.integers(3, 5))
            elif split == "test":
                n = int(rng.integers(1, 3))
            for _ in range(20):
                pair = _pair(rng, roles, teaching, n, large=(split == "test" and shift != "count"))
                if _key(pair['input']) in seen:
                    continue
                # Enforce a visible structural shift, instead of reporting a
                # sampled regime name as if it were measured evidence.
                if split == "test" and shift != "count":
                    previous = [_analyze(p['input'])[2] for p in puzzle['train']]
                    current = _analyze(pair['input'])[2]
                    if shift == "depth":
                        before = max(e['block'].shape[0] - 2 for es in previous for e in es)
                        after = max(e['block'].shape[0] - 2 for e in current)
                    else:
                        before = max(e['distance'] for es in previous for e in es)
                        after = max(e['distance'] for e in current)
                    if after <= before:
                        continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded uniqueness/shift sampling failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            g, background, engines = _analyze(pair['input'])
            output = _grid(pair['output'])
            _need(output.shape == g.shape, "size changed")
            _need(_key(g) not in seen, "duplicate input")
            seen.add(_key(g))
            _need(np.array_equal(_transform(g, background, engines), output), "incorrect output")
            for e in engines:
                # Check scalar steps and color-matched involution independently
                # of the vectorized symmetry scores used by the reader.
                interior = e['block'][1:-1, 1:-1]
                kept = {tuple(p) for p in np.argwhere(interior != background)
                        if 0 <= e['axis'] - p[1] < interior.shape[1]
                        and interior[tuple(p)] == interior[p[0], e['axis'] - p[1]]}
                _need(kept == e['supported'], "reflected support mismatch")
                fuel_units = [u for u in e['units'] if interior[u[0]] != e['marker']]
                steps = 0
                while fuel_units and steps < e['distance']:
                    fuel_units.pop(0)
                    steps += 1
                _need(steps == min(e['remaining'], e['distance']), "incorrect stopping")
                for p in e['destination']:
                    _need(output[p] == e['marker'], "destination changed")
                if split == 'train':
                    training.append(e)
    _need({e['axis'] % 2 for e in training if e['extras']} == {0, 1}, "training must teach both noisy axis kinds")
    _need(len({e['turn'] for e in training if e['remaining']}) >= 2, "training must demonstrate rotating order")
    _need(any(0 < e['remaining'] < e['distance'] and e['extras'] and
              any(len(u) == 2 for u in e['units'][e['spent']:]) for e in training),
          "missing symmetry-sensitive fuel stop")
    _need(any(min(int(np.count_nonzero(e['block'][1:-1, 1:-1] == e['fuel'])), e['distance'])
              != min(e['remaining'], e['distance']) for e in training if e['fuel'] is not None),
          "raw-pixel fuel shortcut not distinguished")
    _need(any(0 < e['distance'] < e['remaining'] and e['spent'] > 0 and
              e['units'][e['spent']][0][0] != e['units'][e['spent'] + e['distance'] - 1][0][0]
              for e in training), "missing partial, multirow destination stop")
    _need(any(any(len(u) == 1 for u in e['units']) for e in training), "missing singleton unit demonstration")
    _need(any(any(len(u) == 1 for u in e['units'][e['spent']:e['spent'] + min(e['remaining'], e['distance'])])
              for e in training), "training never burns a singleton")
    _need(any(not np.array_equal(_transform(g, bg, es, "no_motion"), pair['output'])
              for pair in puzzle['train'] for g, bg, es in [_analyze(pair['input'])]), "motion contribution absent")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """42 seeds, no sample files or execution traces, one puzzle retained at once."""
    shapes, directions, parity, regimes, counts, palettes, structures = set(), set(), set(), set(), set(), set(), set()
    records = Counter()
    # Four units on three rows: two pairs, a singleton, then a pair. Seven
    # supported fuel pixels plus one speck. At distance three, the singleton
    # burns and the last pair remains; at distance five, stop one cell short.
    for distance in (3, 5):
        fixture = np.zeros((17, 14), dtype=np.int8)
        top = distance + 2
        fixture[1, 3:10] = fixture[top - 1, 3:10] = 2
        fixture[top:top + 5, 3:10] = 8
        fixture[top + 1:top + 4, 4:9] = np.array([[5, 5, 0, 5, 5], [5, 0, 5, 0, 0], [5, 0, 0, 0, 5]])
        new_top = 2 if distance == 3 else 3
        expected = np.zeros_like(fixture)
        expected[1, 3:10] = expected[new_top - 1, 3:10] = 2
        expected[new_top:new_top + 5, 3:10] = 8
        last = 5 if distance == 3 else 2
        expected[new_top + 1:new_top + 4, 4:9] = np.array([[2, 2, 0, 2, 2], [0, 0, 2, 0, 0], [last, 0, 0, 0, last]])
        _need(np.array_equal(solve(fixture), expected), "paired/singleton tick fixture failed")
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        if seed < 3:
            _need(_equal(p, generate_puzzle(seed)), "nondeterminism")
            query = p['test'][0]
            for turn in range(4):
                _need(np.array_equal(solve(np.rot90(query['input'], turn)),
                                     np.rot90(query['output'], turn)), "rotation inconsistency")
            recolor = np.random.default_rng(500 + seed).permutation(10)
            _need(np.array_equal(solve(recolor[query['input']]), recolor[query['output']]),
                  "literal-color dependence")
        for split in ('train', 'test'):
            for pair in p[split]:
                g, bg, es = _analyze(pair['input'])
                before = g.copy()
                _need(np.array_equal(solve(g), pair['output']) and np.array_equal(g, before), "solve purity/output failure")
                shapes.add(g.shape)
                counts.add(len(es))
                palettes.add(tuple(map(int, np.unique(g))))
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in g.flat)
                structures.add(hashlib.sha256(bytes(g.shape) + norm).hexdigest())
                for e in es:
                    directions.add(e['turn'])
                    parity.add(e['axis'] % 2)
                    regime = ('zero' if not e['remaining'] else 'fuel' if e['remaining'] < e['distance']
                              else 'exact' if e['remaining'] == e['distance'] else 'destination')
                    regimes.add(regime)
                    records[regime] += 1
                    records['off_center'] += e['axis'] != e['block'].shape[1] - 3
                    records['noise'] += bool(e['extras'])
                    records['fresh'] += e['spent'] == 0
                    records['singletons'] += any(len(u) == 1 for u in e['units'])
                    records['edge'] += any(y in (0, g.shape[0] - 1) or x in (0, g.shape[1] - 1)
                                           for y, x in e['destination'])
                    # Omitting symmetry and treating fuel pixels as ticks must
                    # change the stop in every fuel lesson, not just its noise.
                    raw = np.count_nonzero(e['block'][1:-1, 1:-1] == e['fuel']) if e['fuel'] is not None else 0
                    records['pixel_count_disagrees'] += min(raw, e['distance']) != min(e['remaining'], e['distance'])
    _need(len(shapes) >= 35 and len(structures) == 210, "structural variation collapsed")
    _need(directions == set(range(4)) and parity == {0, 1}, "orientation/axis coverage")
    _need(regimes == {'zero', 'fuel', 'exact', 'destination'}, "stopping coverage")
    _need(counts == {1, 2, 3, 4} and len(palettes) >= 20, "count/palette coverage")
    _need(all(records[k] > 0 for k in ('off_center', 'noise', 'fresh', 'singletons', 'edge', 'pixel_count_disagrees')), "missing family feature")
    # All 30 legal explicit count combinations; no disk output.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(10000 + a * 10 + b, a, b)
            _need(len(p['train']) == a and len(p['test']) == b, "ignored explicit count")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    broken = generate_puzzle(0, 2, 1)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted corrupt output")
    try:
        _support(np.array([[1, 0, 1, 1]]), 0)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous axis accepted")
    return {'seeds': 42, 'pairs': 210, 'shapes': len(shapes), 'structures': len(structures),
            'counts': sorted(counts), 'features': {k: int(v) for k, v in records.items()}}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
