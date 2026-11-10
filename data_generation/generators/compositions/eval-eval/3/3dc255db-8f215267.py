"""Ordinal beacons: payload cells address color-matched components along a ray.

Only NumPy and the standard library are used. The reader has no access to
sampler records, seeds, examples, or parent files. See task_spec.md.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left
STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _parts(cells, diagonal=False):
    remaining = set(cells)
    result = []
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1))
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, component = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    component.add(p)
        result.append(component)
    return result


def _moat(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _box(cells):
    rows, cols = zip(*cells)
    return {(r, c) for r in range(min(rows), max(rows) + 1)
            for c in range(min(cols), max(cols) + 1)}


def _ray(shape, tip, direction):
    dr, dc = DIRS[direction]
    r, c = tip
    cells = []
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            return cells
        cells.append((r, c))


def _envelope(cells, direction):
    """Read closed thin pointed outlines by cross sections, not stencil lookup."""
    dr, dc = DIRS[direction]
    tr, tc = dc, -dr
    local = {(-r * dr - c * dc, r * tr + c * tc) for r, c in cells}
    y0, y1 = min(y for y, x in local), max(y for y, x in local)
    if y1 - y0 < 3:
        return None
    rows = [sorted(x for y, x in local if y == yy) for yy in range(y0, y1 + 1)]
    if any(not row for row in rows) or len(rows[0]) != 1:
        return None
    back = rows[-1]
    if len(back) < 3 or back != list(range(back[0], back[-1] + 1)):
        return None
    tip_x = rows[0][0]
    if not back[0] < tip_x < back[-1]:
        return None
    left = right = tip_x
    cavity = set()
    for yy, row in zip(range(y0 + 1, y1), rows[1:-1]):
        lo, hi = row[0], row[-1]
        if row != sorted({lo, hi}) or not (0 <= left - lo <= 1 and 0 <= hi - right <= 1):
            return None
        cavity.update((-yy * dr + x * tr, -yy * dc + x * tc)
                      for x in range(lo + 1, hi))
        left, right = lo, hi
    if (left, right) != (back[0], back[-1]) or not cavity:
        return None
    tip = -y0 * dr + tip_x * tr, -y0 * dc + tip_x * tc
    return tip, cavity


def _inspect(input_grid):
    a = _grid(input_grid)
    colors, sizes = np.unique(a, return_counts=True)
    bg = int(colors[int(np.argmax(sizes))])
    _require(np.count_nonzero(a == bg) > a.size / 2, "background must occupy over half the input")
    emitters = []
    for color in map(int, colors):
        if color == bg:
            continue
        for cells in _parts(map(tuple, np.argwhere(a == color)), diagonal=True):
            candidates = []
            for direction in range(4):
                geometry = _envelope(cells, direction)
                if geometry is None:
                    continue
                tip, cavity = geometry
                values = {int(a[p]) for p in cavity} - {bg}
                if len(values) != 1 or color in values:
                    continue
                fuel_color = next(iter(values))
                fuel = {p for p in cavity if a[p] == fuel_color}
                if len(fuel) == len(cavity):
                    continue
                box = _box(cells)
                if any(a[p] != bg for p in box - cells - cavity):
                    continue
                candidates.append(dict(frame=cells, cavity=cavity, box=box, tip=tip,
                                       direction=direction, category=color,
                                       fuel_color=fuel_color, fuel=fuel, q=len(fuel)))
            _require(len(candidates) <= 1, "ambiguous emitter direction")
            emitters.extend(candidates)
    _require(bool(emitters), "no pointed emitter")
    emitters.sort(key=lambda e: min(e['frame']))
    owned = set()
    for e in emitters:
        _require(owned.isdisjoint(e['box']), "overlapping emitter regions")
        owned.update(e['box'])
    removed = {p for e in emitters for p in e['frame'] | e['fuel']}
    targets = []
    for color in map(int, colors):
        if color == bg:
            continue
        remaining = set(map(tuple, np.argwhere(a == color))) - removed
        for cells in _parts(remaining):
            _require(cells.isdisjoint(owned), "external object inside emitter region")
            targets.append(dict(color=color, cells=cells))
    beam_colors = {}
    for e in emitters:
        ray = _ray(a.shape, e['tip'], e['direction'])
        _require(set(ray).isdisjoint(owned), "beam crosses an emitter")
        distances = {p: i + 1 for i, p in enumerate(ray)}
        hits = []
        for i, target in enumerate(targets):
            contacts = sorted(distances[p] for p in target['cells'] if p in distances)
            if contacts and target['color'] == e['category']:
                hits.append((contacts[0], i, tuple(contacts)))
        hits.sort()
        end = hits[e['q'] - 1][0] if len(hits) >= e['q'] else len(ray)
        e.update(ray=ray, hits=hits, end=end, beam=ray[:end])
        for p in e['beam']:
            _require(p not in beam_colors or beam_colors[p] == e['fuel_color'],
                     "differently colored beams overlap")
            beam_colors[p] = e['fuel_color']
    return dict(background=bg, emitters=emitters, targets=targets, beams=beam_colors)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover direction, fuel, categories and ordinal contacts from this input."""
    a = _grid(input_grid)
    info = _inspect(a)
    out = np.full_like(a, info['background'])
    for e in info['emitters']:
        for p in e['frame']:
            out[p] = e['category']
    for p, color in info['beams'].items():
        out[p] = color
    return out


def _profile(rng, anchor=False):
    a = int(rng.integers(1, 3 if anchor else 5))
    b = int(rng.integers(1, 3 if anchor else 5))
    if rng.random() < .45:
        b = a
    stem = int(rng.integers(0, 2 if anchor else 3))
    tail = int(rng.integers(4, 6)) if anchor else int(rng.integers(2, 6))
    left = right = 0
    rows = [(0, 0)] * (stem + 1)
    while left < a or right < b:
        dl = left < a and (right == b or rng.random() < .85)
        dd = right < b and (left == a or rng.random() < .85)
        if not dl and not dd:
            dl = left < a
            dd = not dl
        left += int(dl)
        right += int(dd)
        rows.append((-left, right))
    rows.extend([(-a, b)] * tail)
    frame, cavity = set(), set()
    for y, (lo, hi) in enumerate(rows):
        if y == len(rows) - 1:
            frame.update((y, x) for x in range(lo, hi + 1))
        else:
            frame.update({(y, lo), (y, hi)})
            cavity.update((y, x) for x in range(lo + 1, hi))
    return dict(frame=frame, cavity=cavity, tip=(0, 0))


def _fuel(rng, cavity, q, pattern):
    cells = sorted(cavity)
    if pattern == 'compact':
        selected = {cells[int(rng.integers(len(cells)))]}
        while len(selected) < q:
            candidates = sorted(p for p in cavity - selected
                                if any((p[0] + dr, p[1] + dc) in selected for dr, dc in STEPS))
            selected.add(candidates[int(rng.integers(len(candidates)))])
        return selected
    if pattern == 'spread':
        selected = {cells[int(rng.integers(len(cells)))]}
        while len(selected) < q:
            candidates = sorted(cavity - selected)
            distances = [min(abs(p[0]-s[0]) + abs(p[1]-s[1]) for s in selected) for p in candidates]
            best = max(distances)
            choices = [p for p, d in zip(candidates, distances) if d == best]
            selected.add(choices[int(rng.integers(len(choices)))])
        return selected
    return {cells[int(i)] for i in rng.choice(len(cells), q, replace=False)}


def _target_shape(rng, kind=None):
    """Local coordinates: forward distance and transverse offset; contains (0,0)."""
    kind = int(rng.integers(5)) if kind is None else kind
    if kind == 0:  # perpendicular bar
        left, right = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        return {(0, t) for t in range(-left, right + 1)}
    if kind == 1:  # two or three consecutive contacts, still ONE object
        return {(s, 0) for s in range(int(rng.integers(2, 4)))}
    if kind == 2:
        length, width = int(rng.integers(1, 3)), int(rng.integers(2, 5))
        return {(s, t) for s in range(length) for t in range(width)}
    if kind == 3:  # leave and re-enter the ray while connected off the ray
        length = int(rng.integers(3, 5))
        return {(0, 0), (length - 1, 0)} | {(s, 1) for s in range(length)}
    cells = {(0, 0)}
    area = int(rng.integers(2, 11))
    while len(cells) < area:
        candidates = sorted({(s + ds, t + dt) for s, t in cells for ds, dt in STEPS
                             if 0 <= s + ds < 3 and -2 <= t + dt <= 2} - cells)
        cells.add(candidates[int(rng.integers(len(candidates)))])
    return cells


def _oriented(profile, direction):
    dr, dc = DIRS[direction]
    tr, tc = dc, -dr
    def world(p):
        y, x = p
        return -y * dr + x * tr, -y * dc + x * tc
    points = {p: world(p) for p in profile['frame'] | profile['cavity']}
    r0 = min(r for r, c in points.values())
    c0 = min(c for r, c in points.values())
    def translate(p):
        r, c = world(p)
        return r - r0, c - c0
    frame = {translate(p) for p in profile['frame']}
    return dict(frame=frame, cavity={translate(p) for p in profile['cavity']},
                tip=translate(profile['tip']), direction=direction,
                height=max(r for r, c in frame) + 1, width=max(c for r, c in frame) + 1,
                transform=translate)


def _plan(rng, profile, direction, category, fuel_color, unmatched, q, pattern,
          count, teaching=None):
    spec = _oriented(profile, direction)
    spec.update(category=category, fuel_color=fuel_color, q=q,
                fuel={spec['transform'](p) for p in _fuel(rng, profile['cavity'], q, pattern)})
    del spec['transform']
    # An unmatched component precedes the ranked category in teaching scenes.
    lane = [(unmatched, _target_shape(rng, 0), int(rng.integers(1, 3)))]
    start = lane[0][2] + 2 + int(rng.integers(0, 2))
    for i in range(count):
        forced = 1 if teaching == 'long' and i == 0 else 3 if teaching == 'reenter' and i == 0 else None
        shape = _target_shape(rng, forced)
        lane.append((category, shape, start))
        start += max(s for s, t in shape) + 2 + int(rng.integers(0, 2))
    last = max(distance + max(s for s, t in shape) for color, shape, distance in lane)
    spec.update(lane=lane, room=max(last + 2, 9) + int(rng.integers(0, 3)))
    return spec


def _located(spec, r0, c0, shape):
    tip = spec['tip'][0] + r0, spec['tip'][1] + c0
    frame = {(r + r0, c + c0) for r, c in spec['frame']}
    fuel = {(r + r0, c + c0) for r, c in spec['fuel']}
    box = {(r, c) for r in range(r0, r0 + spec['height']) for c in range(c0, c0 + spec['width'])}
    dr, dc = DIRS[spec['direction']]
    tr, tc = dc, -dr
    targets = [dict(color=color, cells={(tip[0] + (distance+s)*dr + t*tr,
                                        tip[1] + (distance+s)*dc + t*tc) for s, t in mask})
               for color, mask, distance in spec['lane']]
    return dict(frame=frame, fuel=fuel, box=box, tip=tip, direction=spec['direction'],
                category=spec['category'], fuel_color=spec['fuel_color'], q=spec['q'],
                targets=targets, ray=set(_ray(shape, tip, spec['direction'])))


def _pack(rng, specs, h, w, strict):
    """Free coordinate packing, with no fixed anchor lattice or object bank."""
    placed, target_guard, body_guard, full_rays = [], set(), set(), set()
    for spec in specs:
        candidates = []
        for r0 in range(h - spec['height'] + 1):
            for c0 in range(w - spec['width'] + 1):
                tip = spec['tip'][0] + r0, spec['tip'][1] + c0
                room = (tip[0], w-1-tip[1], h-1-tip[0], tip[1])[spec['direction']]
                if room >= spec['room']:
                    candidates.append((r0, c0))
        accepted = None
        for index in rng.permutation(len(candidates)):
            r0, c0 = candidates[int(index)]
            e = _located(spec, r0, c0, (h, w))
            cells = {p for t in e['targets'] for p in t['cells']}
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                continue
            if not e['box'].isdisjoint(body_guard | target_guard | full_rays):
                continue
            if not e['ray'].isdisjoint(body_guard):
                continue
            if not cells.isdisjoint(target_guard | body_guard):
                continue
            if strict and (not cells.isdisjoint(full_rays)
                           or any(not t['cells'].isdisjoint(e['ray']) for p in placed for t in p['targets'])):
                continue
            # The independently planned lane objects must also remain distinct.
            seen, touching = set(), False
            for target in e['targets']:
                if not target['cells'].isdisjoint(seen):
                    touching = True
                    break
                seen.update(_moat(target['cells']))
            if touching:
                continue
            accepted = e
            break
        if accepted is None:
            return None
        placed.append(accepted)
        body_guard.update(_moat(accepted['box']))
        target_guard.update(_moat(p for t in accepted['targets'] for p in t['cells']))
        full_rays.update(accepted['ray'])
    return placed


def _construct_output(inp, bg, emitters, targets):
    """Builder labels use placed records and projections, not the input reader."""
    out = np.full_like(inp, bg)
    written = {}
    for e in emitters:
        for p in e['frame']:
            out[p] = e['category']
        dr, dc = DIRS[e['direction']]
        r, c = e['tip']
        room = (r, inp.shape[1]-1-c, inp.shape[0]-1-r, c)[e['direction']]
        entries = []
        for target in targets:
            if target['color'] != e['category']:
                continue
            distances = [(rr-r)*dr + (cc-c)*dc for rr, cc in target['cells']
                         if (rr-r)*dc - (cc-c)*dr == 0 and (rr-r)*dr + (cc-c)*dc > 0]
            if distances:
                entries.append(min(distances))
        entries.sort()
        end = entries[e['q']-1] if len(entries) >= e['q'] else room
        for distance in range(1, end + 1):
            p = r + distance*dr, c + distance*dc
            if p in written and written[p] != e['fuel_color']:
                return None
            written[p] = e['fuel_color']
            out[p] = e['fuel_color']
    return out


def _scene(rng, specs, bg, palette, strict=False, extras=0):
    mins = [10, 10]
    for spec in specs:
        mins[0] = max(mins[0], spec['height'] + (spec['room'] if spec['direction'] in (0, 2) else 0))
        mins[1] = max(mins[1], spec['width'] + (spec['room'] if spec['direction'] in (1, 3) else 0))
    if max(mins) > 30:
        return None
    placed = None
    for attempt in range(8):
        h = int(rng.integers(mins[0], 31)) if attempt < 5 else 30
        w = int(rng.integers(mins[1], 31)) if attempt < 5 else 30
        placed = _pack(rng, specs, h, w, strict)
        if placed is not None:
            break
    if placed is None:
        return None
    inp = np.full((h, w), bg, dtype=np.int8)
    targets = [t for e in placed for t in e['targets']]
    for e in placed:
        for p in e['frame']:
            inp[p] = e['category']
        for p in e['fuel']:
            inp[p] = e['fuel_color']
    blocked = _moat({p for e in placed for p in e['box']} | {p for t in targets for p in t['cells']})
    rays = {p for e in placed for p in e['ray']}
    # Additional objects form a shared field. Teaching extras stay off every
    # axis, guaranteeing an explicit same-category off-axis counterexample.
    for _ in range(extras):
        mask = _target_shape(rng)
        low_s, low_t = min(s for s, t in mask), min(t for s, t in mask)
        mask = {(s-low_s, t-low_t) for s, t in mask}
        mh, mw = max(s for s, t in mask)+1, max(t for s, t in mask)+1
        options = [(r, c) for r in range(h-mh+1) for c in range(w-mw+1)]
        for index in rng.permutation(len(options)):
            r, c = options[int(index)]
            cells = {(r+s, c+t) for s, t in mask}
            if cells.isdisjoint(blocked) and (not strict or cells.isdisjoint(rays)):
                color = placed[0]['category'] if strict else int(rng.choice(palette))
                targets.append(dict(color=color, cells=cells))
                blocked.update(_moat(cells))
                break
    for t in targets:
        for p in t['cells']:
            inp[p] = t['color']
    out = _construct_output(inp, bg, placed, targets)
    if out is None or np.count_nonzero(inp == bg) <= inp.size/2:
        return None
    try:
        detail = _inspect(inp)
    except ValueError:
        return None
    if len(detail['emitters']) != len(placed) or not np.array_equal(solve(inp), out):
        return None
    return dict(input=inp, output=out)


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    _require(low <= int(value) <= high, name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    _require(nt > nq, 'num_train must exceed num_test')
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    color_order = list(map(int, rng.permutation(10)))
    category, fuel, switch, novel = color_order[:4]
    background0, background1 = color_order[4:6]
    anchor = _profile(rng, anchor=True)
    base_direction = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for attempt in range(120):
                if split == 'train' and index < 2:
                    bg = (background0, background1)[index]
                    direction = (base_direction + 2*index) % 4
                    specs = [
                        _plan(rng, anchor, direction, category if index == 0 else switch,
                              fuel, switch if index == 0 else category, 2,
                              'compact' if index == 0 else 'spread', 3,
                              'long' if index == 0 else 'reenter'),
                        _plan(rng, anchor, direction, category,
                              fuel, switch, 3, 'random',
                              4 if index == 0 else int(rng.integers(0, 2))),
                    ]
                    pair = _scene(rng, specs, bg, [category, fuel, switch], strict=True, extras=1)
                else:
                    is_first_test = split == 'test' and index == 0
                    bg = int(rng.choice([c for c in range(10) if not (is_first_test and c == novel)]))
                    palette = [c for c in range(10) if c != bg]
                    cats = [c for c in palette if split == 'test' or c != novel]
                    n = 2 if is_first_test else int(rng.choice([1, 1, 2, 2, 3]))
                    direction = int(rng.integers(4))
                    directions = [direction] * n
                    if is_first_test:
                        directions[1] = (direction + int(rng.choice([1, 3]))) % 4
                    elif split == 'test' and rng.random() < .6:
                        directions = list(map(int, rng.integers(0, 4, n)))
                    shared_cat = novel if is_first_test else int(rng.choice(cats))
                    shared_fuel = int(rng.choice([c for c in palette if c != shared_cat]))
                    shared = is_first_test or rng.random() < .6
                    specs = []
                    for i in range(n):
                        cat = shared_cat if shared else int(rng.choice(cats))
                        fc = shared_fuel if shared and rng.random() < .8 or is_first_test else int(
                            rng.choice([c for c in palette if c != cat]))
                        profile = _profile(rng)
                        capacity = len(profile['cavity']) - 1
                        if capacity < 2:
                            break
                        q = 2 if is_first_test and i == 0 else int(rng.integers(1, min(5, capacity)+1))
                        matched = int(rng.integers(q, 6)) if is_first_test and i == 0 else int(rng.integers(0, 6))
                        unmatched = int(rng.choice([c for c in palette if c != cat]))
                        specs.append(_plan(rng, profile, directions[i], cat, fc, unmatched, q,
                                           str(rng.choice(['compact', 'spread', 'random'])), matched))
                    if len(specs) != n:
                        continue
                    pair = _scene(rng, specs, bg, palette, extras=int(rng.integers(0, 5)))
                    if pair is not None and is_first_test:
                        info = _inspect(pair['input'])
                        if not any(len(e['hits']) >= e['q'] and e['end'] < len(e['ray'])
                                   for e in info['emitters']):
                            continue
                if pair is None:
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError('bounded sampler could not place a scene')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _frame_signature(e):
    dr, dc = DIRS[e['direction']]
    tr, tc = dc, -dr
    r0, c0 = e['tip']
    return tuple(sorted((-(r-r0)*dr-(c-c0)*dc, (r-r0)*tr+(c-c0)*tc) for r, c in e['frame']))


def _alternative(info, shape, mode):
    """Concrete ablations used only to test teaching evidence."""
    out = np.full(shape, info['background'], dtype=np.int8)
    for e in info['emitters']:
        for p in e['frame']:
            out[p] = e['category']
        if mode == 'literal_length':  # A alone: payload count counts grid cells
            end = min(e['q'], len(e['ray']))
        elif mode == 'fixed_rank':  # omit A's changing unary payload
            end = e['hits'][1][0] if len(e['hits']) >= 2 else len(e['ray'])
        elif mode == 'fuel_components':
            q = len(_parts(e['fuel']))
            end = e['hits'][q-1][0] if len(e['hits']) >= q else len(e['ray'])
        elif mode == 'matched_pixels':  # omit B's component compression
            contacts = sorted(d for _, _, ds in e['hits'] for d in ds)
            end = contacts[e['q']-1] if len(contacts) >= e['q'] else len(e['ray'])
        elif mode == 'all_colors':
            lookup = {p: i+1 for i, p in enumerate(e['ray'])}
            contacts = sorted(min(lookup[p] for p in t['cells'] if p in lookup)
                              for t in info['targets'] if any(p in lookup for p in t['cells']))
            end = contacts[e['q']-1] if len(contacts) >= e['q'] else len(e['ray'])
        else:
            raise ValueError(mode)
        for p in e['ray'][:end]:
            out[p] = e['fuel_color']
    return out


def validate_generated_puzzle(puzzle):
    """Check returned pixels, rule semantics, minimal teaching witnesses and shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    details, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = inp.shape, inp.tobytes()
            assert key not in seen
            seen.add(key)
            d = _inspect(inp)
            assert 1 <= len(d['emitters']) <= 3
            assert np.array_equal(solve(inp), out) and not np.array_equal(inp, out)
            for e in d['emitters']:
                assert e['frame'].isdisjoint(e['fuel']) and 0 < e['q'] < len(e['cavity'])
                assert all(out[p] == e['category'] for p in e['frame'])
                assert all(out[p] == d['background'] for p in e['cavity'])
                assert all(out[p] == e['fuel_color'] for p in e['beam'])
                assert len({hit[1] for hit in e['hits']}) == len(e['hits'])
                if len(e['hits']) >= e['q']:
                    assert e['end'] == e['hits'][e['q']-1][0]
                else:
                    assert e['end'] == len(e['ray'])
            for t in d['targets']:
                assert len(_parts(t['cells'])) == 1
                assert all(out[p] == d['beams'].get(p, d['background']) for p in t['cells'])
            details[split].append(d)
    train = details['train']
    complete = [e for d in train for e in d['emitters']
                if len(e['hits']) >= e['q'] and e['end'] < len(e['ray'])]
    assert len({e['q'] for e in complete}) >= 2, 'constant rank ambiguity'
    matched = {}
    arrangements = {}
    for e in complete:
        matched.setdefault((_frame_signature(e), e['category'], e['fuel_color']), set()).add(e['q'])
        arrangements.setdefault(e['q'], set()).add(len(_parts(e['fuel'])))
    assert any(len(v) >= 2 for v in matched.values()), 'fuel rank confounded with outline or palette'
    assert any(len(v) >= 2 for v in arrangements.values()), 'cells versus fuel clusters not distinguished'
    assert any(len(e['hits']) < e['q'] for d in train for e in d['emitters']), 'no boundary fallback demonstration'
    assert any(len(ds) > 1 for e in complete for _, _, ds in e['hits'][:e['q']]), 'target area shortcut'
    assert any(any(t['color'] == e['category'] and t['cells'].isdisjoint(e['ray'])
                   for t in d['targets']) for d in train for e in d['emitters']), 'no off-axis category'
    for mode in ('literal_length', 'fixed_rank', 'fuel_components', 'matched_pixels', 'all_colors'):
        assert any(not np.array_equal(_alternative(d, np.asarray(p['input']).shape, mode), p['output'])
                   for d, p in zip(train, puzzle['train'])), 'shortcut survives: ' + mode
    training_categories = {e['category'] for d in train for e in d['emitters']}
    unmatched_colors = {t['color'] for d in train for t in d['targets']
                        if t['color'] not in {e['category'] for e in d['emitters']}}
    positive_categories = {e['category'] for e in complete}
    assert positive_categories & unmatched_colors, 'no effective category role switch'
    first_test = details['test'][0]
    assert len({e['direction'] for e in first_test['emitters']}) >= 2, 'missing mixed-direction shift'
    assert any(e['category'] not in training_categories for e in first_test['emitters']), 'no new category'
    assert any(len(e['hits']) >= e['q'] and e['end'] < len(e['ray']) for e in first_test['emitters'])


def _signature(puzzle):
    return tuple((np.asarray(p[k]).shape, np.asarray(p[k]).tobytes())
                 for split in ('train', 'test') for p in puzzle[split] for k in ('input', 'output'))


def _semantic_cases():
    # Hand-specified first-contact distances: 2, 8, 11. The first object
    # re-enters the axis at distance 4; the off-axis singleton touches the
    # second object diagonally and must remain a separate 4-component.
    a = np.zeros((19, 13), dtype=np.int8)
    frame = {(13, 6), (14, 5), (14, 7), (15, 4), (15, 8),
             (16, 4), (16, 8), (17, 4), (17, 8)} | {(18, c) for c in range(4, 9)}
    for p in frame:
        a[p] = 3
    a[15, 6] = a[16, 6] = 1
    first = {(11, 6), (11, 7), (10, 7), (9, 7), (9, 6)}
    for p in first | {(5, 6), (2, 6), (4, 7), (6, 1), (6, 2)}:
        a[p] = 3
    a[12, 6] = 8
    expected = np.zeros_like(a)
    for p in frame:
        expected[p] = 3
    expected[5:13, 6] = 1
    assert np.array_equal(solve(a), expected)
    assert [h[0] for h in _inspect(a)['emitters'][0]['hits']] == [2, 8, 11]
    larger = a.copy()
    larger[5, 4:6] = 3
    assert np.array_equal(solve(larger), expected), 'component area changed ordinal address'
    for q, first_row in ((1, 11), (3, 2), (4, 0)):
        changed = a.copy()
        changed[15, 6] = changed[16, 6] = 0
        for p in ((15, 6), (16, 6), (17, 6), (15, 5))[:q]:
            changed[p] = 1
        wanted = expected.copy()
        wanted[:13, 6] = 0
        wanted[first_row:13, 6] = 1
        assert np.array_equal(solve(changed), wanted)
    edge = np.zeros_like(a)
    wanted = np.zeros_like(a)
    for r, c in frame:
        edge[r-13, c] = wanted[r-13, c] = 3
    edge[2, 6] = edge[3, 6] = 1
    assert np.array_equal(solve(edge), wanted), 'edge tip should emit no visible cells'


def self_test():
    """Bounded 42-seed semantic, variation, count-interface and metamorphic tests."""
    _semantic_cases()
    shapes, profiles, backgrounds, dirs, ranks, object_masks = set(), set(), set(), set(), set(), set()
    emitter_counts, target_areas, contact_counts, endings = set(), set(), set(), set()
    signatures = set()
    reentries = crossings = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                d = _inspect(inp)
                shapes.add(inp.shape)
                backgrounds.add(d['background'])
                emitter_counts.add(len(d['emitters']))
                for t in d['targets']:
                    r0, c0 = min(r for r, c in t['cells']), min(c for r, c in t['cells'])
                    object_masks.add(tuple(sorted((r-r0, c-c0) for r, c in t['cells'])))
                    target_areas.add(len(t['cells']))
                used = Counter(p for e in d['emitters'] for p in e['beam'])
                crossings += any(n > 1 for n in used.values())
                for e in d['emitters']:
                    profiles.add(_frame_signature(e))
                    dirs.add(e['direction'])
                    ranks.add(e['q'])
                    contact_counts.add(len(e['hits']))
                    endings.add('ranked' if len(e['hits']) >= e['q'] else 'edge')
                    reentries += sum(any(b-a > 1 for a, b in zip(ds, ds[1:])) for _, _, ds in e['hits'])
                # Global rotations and color permutations preserve the rule.
                rotation = seed % 4
                permutation = np.random.default_rng(seed+10000).permutation(10)
                changed = permutation[np.rot90(inp, rotation)]
                assert np.array_equal(solve(changed), permutation[np.rot90(out, rotation)])
                # Payload rearrangement has no effect when its cell count is fixed.
                e = d['emitters'][0]
                altered = inp.copy()
                for p in e['fuel']:
                    altered[p] = d['background']
                for p in sorted(e['cavity'])[:e['q']]:
                    altered[p] = e['fuel_color']
                assert np.array_equal(solve(altered), out)
    assert len(signatures) == 42 and len(shapes) >= 35 and len(profiles) >= 35
    assert backgrounds == set(range(10)) and dirs == set(range(4))
    assert emitter_counts == {1, 2, 3} and {1, 2, 3, 4, 5} <= ranks
    assert len(object_masks) >= 35 and max(target_areas) >= 10
    assert {0, 1, 2, 3, 4, 5} <= contact_counts and endings == {'ranked', 'edge'}
    assert reentries >= 20 and crossings > 0
    assert _signature(generate_puzzle(123, 7, 3)) == _signature(generate_puzzle(123, 7, 3))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(0, 2, 1)
    damaged['test'][0]['output'] = damaged['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted unprocessed targets')


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
