"""Travelling switch-head: an itinerary controls repair, deletion and layering.

No imports of parent code, files, seed tables, or saved examples are used.
All generation/validation is in memory; self_test writes no artifacts.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


class _SamplingExhausted(RuntimeError):
    """A bounded candidate batch found no admissible scene."""


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _boundary(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _rotate(p, k):
    r, c = p
    for _ in range(k % 4):
        r, c = c, -r
    return r, c


def _pose(mask, tip, direction):
    return {_add(tip, _rotate(p, direction - 1)) for p in mask}


def _arrow(span, neck, solid, tails):
    cells = {(0, -i) for i in range(neck)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if solid else (-depth, depth)
        cells.update((r, 1 - neck - depth) for r in rows)
    for i in range(1, tails + 1):
        cells.update((r, 1 - neck - span - i) for r in (-span, span))
    return tuple(sorted(cells))


MASKS = tuple(_arrow(s, n, solid, t) for s in (1, 2)
              for n in (1, 2) for solid in (False, True) for t in (0, 1))


def _components(cells, diagonal=False):
    cells, result = set(cells), []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r, c) != (0, 0)] if diagonal else DIRS
    while cells:
        p = min(cells)
        cells.remove(p)
        component, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for d in steps:
                q = r + d[0], c + d[1]
                if q in cells:
                    cells.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _identify_head(cells):
    matches = []
    for p in sorted(cells):
        for direction in range(4):
            mask = {_rotate((r - p[0], c - p[1]), 1 - direction)
                    for r, c in cells}
            if (all(c <= 0 for r, c in mask)
                    and {v for v in mask if v[1] == 0} == {(0, 0)}
                    and max(abs(r) for r, c in mask) > 0
                    and {(-r, c) for r, c in mask} == mask):
                matches.append((tuple(sorted(mask)), p, direction))
    assert len(matches) == 1, "head must have one geometric pointing tip"
    return matches[0]


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert (dr == 0) != (dc == 0)
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _itinerary(tip, dots):
    remaining, points, directions = set(dots), [tip], []
    while remaining:
        rays = {}
        for q in sorted(remaining):
            if q[0] == tip[0] or q[1] == tip[1]:
                d = _direction(tip, q)
                rays.setdefault(d, []).append(q)
        assert len(rays) == 1, "there must be exactly one occupied ray"
        d, options = next(iter(rays.items()))
        tip = min(options, key=lambda q: abs(q[0] - tip[0]) + abs(q[1] - tip[1]))
        points.append(tip)
        directions.append(d)
        remaining.remove(tip)
    return points, directions


def _neighbors(p, cells):
    r, c = p
    return [q for q in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))
            if q in cells]


def _straight(p, cells):
    ns = _neighbors(p, cells)
    return (len(ns) == 2 and ns[0][0] + ns[1][0] == 2 * p[0]
            and ns[0][1] + ns[1][1] == 2 * p[1])


def _line(a, b):
    dr = int(b[0] > a[0]) - int(b[0] < a[0])
    dc = int(b[1] > a[1]) - int(b[1] < a[1])
    assert dr == 0 or dc == 0
    return [a] + [(a[0] + i * dr, a[1] + i * dc)
                  for i in range(1, max(abs(b[0] - a[0]), abs(b[1] - a[1])) + 1)]


def _ray(p, d, shape):
    cells = set()
    p = _add(p, d)
    while _inside(p, shape):
        cells.add(p)
        p = _add(p, d)
    return cells


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _wire_valid(cells, pivot):
    if not cells:
        return False
    adjacency = {p: _neighbors(p, cells) for p in cells}
    if (len(adjacency[pivot]) not in (2, 3)
            or any(not 1 <= len(ns) <= (3 if p == pivot else 2)
                   for p, ns in adjacency.items())
            or sum(map(len, adjacency.values())) != 2 * (len(cells) - 1)):
        return False
    seen, stack = {pivot}, [pivot]
    while stack:
        for q in adjacency[stack.pop()]:
            if q not in seen:
                seen.add(q)
                stack.append(q)
    return len(seen) == len(cells)


def _reconstruct(inp, bg, ink, points):
    """Repair marked corners, then bridge one/two-cell straight omissions.

    Endpoints of every bridge must already be visible in the original layer
    plus its repaired pivot. Bridges are simultaneous, never recursive guesses.
    """
    colors = sorted(set(map(int, np.unique(inp))) - {bg, ink})
    assert len(colors) == len(points) - 1
    owners, layers = [], {c: set(map(tuple, np.argwhere(inp == c).tolist())) for c in colors}
    for q in points[1:]:
        nearby = {int(inp[p]) for d in DIRS
                  if _inside(p := _add(q, d), inp.shape) and int(inp[p]) in colors}
        assert len(nearby) == 1, "each marked pivot needs a unique local wire color"
        c = next(iter(nearby))
        assert c not in owners, "one marked pivot per wire color"
        owners.append(c)
        layers[c].add(q)
    for c, cells in layers.items():
        observed = set(cells)
        for p in sorted(observed):
            for d in ((1, 0), (0, 1)):
                for length in (2, 3):
                    q = p[0] + length * d[0], p[1] + length * d[1]
                    if q not in observed:
                        continue
                    between = [(p[0] + i * d[0], p[1] + i * d[1])
                               for i in range(1, length)]
                    if all(x in observed for x in between):
                        continue  # A fully observed interval adds no pixels.
                    side = (-d[1], d[0])
                    def turns(x):
                        return _add(x, side) in observed or _add(x, (-side[0], -side[1])) in observed
                    if turns(p) and turns(q):
                        continue
                    if all(int(inp[x]) != ink for x in between):
                        cells.update(between)
    for c, q in zip(owners, points[1:]):
        assert _wire_valid(layers[c], q), "repaired wire must be a thin tree with its only junction at the pivot"
    membership = {}
    for c, cells in layers.items():
        for p in cells:
            membership.setdefault(p, []).append(c)
    for p, cs in membership.items():
        if len(cs) > 1:
            assert len(cs) == 2 and int(inp[p]) in cs, "original crossing owner must remain visible"
            a, b = cs
            assert _straight(p, layers[a]) and _straight(p, layers[b])
            assert not set(_neighbors(p, layers[a])) & set(_neighbors(p, layers[b]))
    return owners, layers


def _old_branch(cells, pivot, direction, shape):
    d = DIRS[direction]
    current = pivot[0] - d[0], pivot[1] - d[1]
    assert current in cells and _add(pivot, d) not in cells
    previous, walk = pivot, []
    while True:
        assert current not in walk
        walk.append(current)
        nxt = [p for p in _neighbors(current, cells) if p != previous]
        if not nxt:
            assert _boundary(current, shape), "opposite branch must end at the boundary"
            return tuple(walk)
        assert len(nxt) == 1
        previous, current = current, nxt[0]


def _read_with_ink(inp, bg, ink):
    pieces = _components(map(tuple, np.argwhere(inp == ink).tolist()), True)
    heads = [s for s in pieces if len(s) > 1]
    dots = {next(iter(s)) for s in pieces if len(s) == 1}
    assert len(heads) == 1 and 2 <= len(dots) <= 6
    mask, tip, facing = _identify_head(heads[0])
    points, directions = _itinerary(tip, dots)
    assert directions[0] == facing
    for i, p in enumerate(points):
        pose = _pose(mask, p, directions[max(0, i - 1)])
        assert all(_inside(x, inp.shape) for x in pose)
        assert all(max(abs(q[0] - x[0]), abs(q[1] - x[1])) > 1
                   for q in points[i + 1:] for x in pose)
    owners, layers = _reconstruct(inp, bg, ink, points)
    assert not heads[0] & set().union(*layers.values())
    edits = []
    for q, direction, c in zip(points[1:], directions, owners):
        old = _old_branch(layers[c], q, direction, inp.shape)
        edits.append((c, q, direction, old, _ray(q, DIRS[direction], inp.shape)))
    return dict(bg=bg, ink=ink, mask=mask, points=points, directions=directions,
                owners=owners, layers=layers, edits=edits, shape=inp.shape)


def _read_scene(inp):
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > inp.size // 2, "input background must occupy a strict majority"
    answers = []
    for ink in sorted(set(map(int, np.unique(inp))) - {bg}):
        try:
            answers.append(_read_with_ink(inp, bg, ink))
        except AssertionError:
            pass
    assert len(answers) == 1, "roles and itinerary must be uniquely recoverable"
    return answers[0]


def _render(inp, scene, rewire=True, reverse=False, park=True):
    layers = scene['layers']
    kept = {c: set(s) for c, s in layers.items()}
    if rewire:
        for c, q, d, old, ray in scene['edits']:
            kept[c].difference_update(old)
    memberships = {}
    for c, cells in kept.items():
        for p in cells:
            memberships.setdefault(p, []).append(c)
    out = np.full(inp.shape, scene['bg'], dtype=np.int8)
    for p, owners in memberships.items():
        if len(owners) == 1:
            out[p] = owners[0]
        else:
            assert int(inp[p]) in owners, "unresolved retained crossing"
            out[p] = inp[p]
    if rewire:
        edits = scene['edits'][::-1] if reverse else scene['edits']
        for c, q, d, old, ray in edits:
            for p in ray:
                out[p] = c
    if park:
        for p in _pose(scene['mask'], scene['points'][-1], scene['directions'][-1]):
            out[p] = scene['ink']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    return _render(inp, _read_scene(inp))


def _route(rng, shape, mask, count, crossing, need_rotation=False):
    """Incremental geometric walks, without a route bank or a lattice."""
    # These local views depend only on the current visible mask, not the seed.
    # Reuse them while considering candidates; do not rotate every past pose
    # again for every candidate distance.
    rotated = [tuple(_rotate(p, d - 1) for p in mask) for d in range(4)]
    bounds = [(min(r for r, c in cells), max(r for r, c in cells),
               min(c for r, c in cells), max(c for r, c in cells))
              for cells in rotated]
    clearance = [{(r + dr, c + dc) for r, c in cells
                  for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                 for cells in rotated]
    def fits(p, d):
        r0, r1, c0, c1 = bounds[d]
        return (0 <= p[0] + r0 and p[0] + r1 < shape[0]
                and 0 <= p[1] + c0 and p[1] + c1 < shape[1])
    for _ in range(70):
        start = (int(rng.integers(2, shape[0] - 2)), int(rng.integers(2, shape[1] - 2)))
        points, directions = [start], []
        initial_fits = [fits(start, d) for d in range(4)]
        initial_blocked = [{_add(start, offset) for offset in cells}
                           for cells in clearance]
        for index in range(count):
            candidates = []
            p = points[-1]
            blocked = set()
            if index:
                for j, old in enumerate(points):
                    blocked.update(_add(old, offset)
                                   for offset in clearance[directions[max(0, j - 1)]])
            if crossing and index == count - 1:
                prior_rays = [_ray(x, DIRS[d], shape) for x, d in zip(points[1:], directions)]
                prior_crossings = set()
                for i, a in enumerate(prior_rays):
                    for j in range(i + 1, len(prior_rays)):
                        if directions[i] % 2 != directions[j] % 2:
                            prior_crossings.update(a & prior_rays[j])
            for d, step in enumerate(DIRS):
                for distance in range(4, max(shape)):
                    q = p[0] + distance * step[0], p[1] + distance * step[1]
                    if not (2 <= q[0] < shape[0] - 2 and 2 <= q[1] < shape[1] - 2):
                        continue
                    if not fits(q, d):
                        continue
                    if index == 0 and not initial_fits[d]:
                        continue
                    if any(max(abs(q[0] - x[0]), abs(q[1] - x[1])) <= 2 for x in points):
                        continue
                    # A newly added future point may only share a past ray
                    # beyond its already chosen nearest successor.
                    if any((q[0] == a[0] or q[1] == a[1]) and
                           (_direction(a, q) != _direction(a, b) or
                            abs(q[0] - a[0]) + abs(q[1] - a[1]) <=
                            abs(b[0] - a[0]) + abs(b[1] - a[1]))
                           for a, b in zip(points, points[1:])):
                        continue
                    if q in (blocked if index else initial_blocked[d]):
                        continue
                    if crossing and index == count - 1:
                        new_ray = _ray(q, step, shape)
                        meetings = set(prior_crossings)
                        for old_direction, ray in zip(directions, prior_rays):
                            if old_direction % 2 != d % 2:
                                meetings.update(ray & new_ray)
                        if not meetings or meetings <= {_add(q, offset) for offset in rotated[d]}:
                            continue
                    candidates.append((q, d))
            if not candidates:
                break
            # Direction is sampled before distance, avoiding a long-ray bias.
            ds = sorted({d for q, d in candidates})
            d = int(rng.choice(ds))
            options = [q for q, dd in candidates if dd == d]
            points.append(options[int(rng.integers(len(options)))])
            directions.append(d)
        if len(directions) != count or (need_rotation and directions[-1] == directions[0]):
            continue
        if crossing:
            rays = [_ray(p, DIRS[d], shape) for p, d in zip(points[1:], directions)]
            head = _pose(mask, points[-1], directions[-1])
            if not any((a & rays[j]) - head for i, a in enumerate(rays)
                       for j in range(i + 1, len(rays))
                       if directions[i] % 2 != directions[j] % 2):
                continue
        return points, directions
    return None


def _extend_path(cells, segment, existing=None):
    if existing is None:
        existing = set(cells)
    for i, p in enumerate(segment):
        if p in existing:
            return None
        ns = _neighbors(p, existing)
        if ns != ([cells[-1]] if i == 0 else []):
            return None
    return cells + segment


def _poly_arm(rng, pivot, direction, shape, bends, boundary):
    cells, current, d = [pivot], pivot, direction
    for j in range(bends + (not boundary)):
        existing = set(cells)
        extension, choices = [], []
        minimum = 3 if boundary else 2
        # All candidate lengths share a straight prefix. A bad prefix makes
        # every longer candidate invalid, so inspect each added cell once.
        for length in range(1, 10):
            q = current[0] + length * d[0], current[1] + length * d[1]
            if not (1 <= q[0] < shape[0] - 1 and 1 <= q[1] < shape[1] - 1):
                break
            if q in existing or _neighbors(q, existing) != ([current] if length == 1 else []):
                break
            extension.append(q)
            if length >= minimum:
                choices.append(length)
        if not choices:
            return None
        length = choices[int(rng.integers(len(choices)))]
        cells = cells + extension[:length]
        current = cells[-1]
        d = (-d[1], d[0]) if rng.random() < .5 else (d[1], -d[0])
    if boundary:
        endpoint = ((0 if d[0] < 0 else shape[0] - 1, current[1]) if d[0]
                    else (current[0], 0 if d[1] < 0 else shape[1] - 1))
        if endpoint == current:
            return None
        cells = _extend_path(cells, _line(current, endpoint)[1:])
    return cells


def _compatible(cells, layers, forbidden):
    if cells & forbidden:
        return False
    for other in layers.values():
        for p in cells & other:
            if not _straight(p, cells) or not _straight(p, other):
                return False
            if set(_neighbors(p, cells)) & set(_neighbors(p, other)):
                return False
            if sum(p in s for s in layers.values()) > 1:
                return False
    return True


def _construct_wires(rng, shape, mask, points, directions, colors):
    layers, edits = {}, []
    head = _pose(mask, points[0], directions[0])
    for index, (q, direction, c) in enumerate(zip(points[1:], directions, colors)):
        forbidden = set(head)
        for other in points[1:]:
            if other != q:
                forbidden.update((other[0] + dr, other[1] + dc)
                                 for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        forward = DIRS[direction]
        backward = (-forward[0], -forward[1])
        for attempt in range(65):
            # A straight backward branch would pass through the preceding
            # itinerary point (or the initial head), concealing its evidence.
            bends = int(rng.choice([1, 2, 3], p=[.55, .33, .12]))
            old_walk = _poly_arm(rng, q, backward, shape, bends, True)
            if old_walk is None:
                continue
            side = (-forward[1], forward[0])
            if rng.random() < .5:
                side = (-side[0], -side[1])
            retained = _poly_arm(rng, q, side, shape, int(rng.integers(3)), False)
            if retained is None:
                continue
            cells = set(old_walk) | set(retained)
            if len(cells) != len(old_walk) + len(retained) - 1:
                continue
            if rng.random() < .25:
                extra = _poly_arm(rng, q, (-side[0], -side[1]), shape, 0, False)
                if extra is not None and not (set(extra) - {q}) & cells:
                    cells |= set(extra)
            if not _wire_valid(cells, q) or _add(q, forward) in cells:
                continue
            if not _compatible(cells, layers, forbidden):
                continue
            layers[c] = cells
            edits.append((c, q, direction, tuple(old_walk[1:]), _ray(q, forward, shape)))
            break
        else:
            return None
    return layers, edits


def _turns(walk):
    ds = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _metrics(inp, scene, out):
    edits = scene['edits']
    kept = {c: s - set(next(e[3] for e in edits if e[0] == c))
            for c, s in scene['layers'].items()}
    restored = sum(int(inp[p]) == c and int(out[p]) in kept and p in kept[int(out[p])]
                   and int(out[p]) != c for c, q, d, old, ray in edits for p in old)
    retained_gaps = sum(int(inp[p]) == scene['bg'] and int(out[p]) == c
                        for c, cells in kept.items() for p in cells)
    head = _pose(scene['mask'], scene['points'][-1], scene['directions'][-1])
    transverse = any((a[4] & b[4]) - head for i, a in enumerate(edits)
                     for b in edits[i + 1:] if a[2] % 2 != b[2] % 2)
    return dict(stations=len(edits), shape=inp.shape, mask=scene['mask'],
                initial=scene['directions'][0], final=scene['directions'][-1],
                directions=tuple(scene['directions']),
                old_bends=tuple(_turns((q,) + old) for c, q, d, old, ray in edits),
                junctions=sum(len(_neighbors(q, scene['layers'][c])) == 3 for c, q, d, old, ray in edits),
                restored_crossings=restored, retained_gaps=retained_gaps,
                gaps=sum(int(inp[p]) == scene['bg'] for s in scene['layers'].values() for p in s),
                order_witness=not np.array_equal(out, _render(inp, scene, reverse=True)),
                transverse_witness=transverse,
                no_rewire_diff=not np.array_equal(out, _render(inp, scene, rewire=False)),
                palette=(scene['bg'], scene['ink'], *scene['owners']))


def _make_pair(rng, mask, count, palette, witness=False, compact=False):
    bg, ink = palette
    for attempt in range(180):
        low = 16 if compact else 23 if count >= 4 else 20
        high = 26 if compact else 30
        shape = tuple(int(v) for v in rng.integers(low, high + 1, size=2))
        route = _route(rng, shape, mask, count, witness, need_rotation=compact)
        if route is None:
            continue
        points, directions = route
        colors = list(map(int, rng.choice([c for c in range(10) if c not in palette], count, replace=False)))
        wires = _construct_wires(rng, shape, mask, points, directions, colors)
        if wires is None:
            continue
        layers, edits = wires
        inp = np.full(shape, bg, dtype=np.int8)
        # Initial crossing ownership is visibly encoded, not inferred from color.
        for c in map(int, rng.permutation(colors)):
            for p in layers[c]:
                inp[p] = c
        for p in points[1:] + list(_pose(mask, points[0], directions[0])):
            inp[p] = ink
        scene = dict(bg=bg, ink=ink, mask=mask, points=points, directions=directions,
                     owners=colors, layers=layers, edits=edits, shape=shape)
        try:
            parsed = _read_scene(inp)
            assert parsed['layers'] == layers and parsed['points'] == points
            out = _render(inp, scene)
            info = _metrics(inp, parsed, out)
            assert any(info['old_bends']) and info['no_rewire_diff']
            if witness:
                assert info['order_witness'] and info['transverse_witness'] and info['restored_crossings'] > 0
        except AssertionError:
            continue
        # Damage only unambiguously bridgeable straight, non-crossing cells.
        candidates = [(c, p) for c, cells in layers.items() for p in sorted(cells)
                      if _straight(p, cells) and p not in points
                      and sum(p in s for s in layers.values()) == 1
                      and all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in points[1:])]
        rng.shuffle(candidates)
        desired, damaged = int(rng.integers(1, 7)), []
        for c, p in candidates:
            if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 2 for q in damaged):
                continue
            previous = int(inp[p])
            inp[p] = bg
            try:
                recovered = _read_scene(inp)
                assert recovered['layers'] == layers
                assert np.array_equal(_render(inp, recovered), out)
            except AssertionError:
                inp[p] = previous
            else:
                damaged.append(p)
                info = _metrics(inp, recovered, out)
                if len(damaged) >= desired and info['retained_gaps']:
                    break
        if not damaged or not info['retained_gaps']:
            continue
        return {'input': inp, 'output': out}
    raise _SamplingExhausted('bounded construction batch exhausted; no ambiguous fallback is labelled')


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.random() < .2)
    palette = tuple(map(int, rng.choice(10, 2, replace=False)))
    first = int(rng.integers(8))
    second = int(rng.choice([i for i in range(8) if i != first]))
    train_masks = [MASKS[first], MASKS[second]]
    train_masks += [MASKS[int(rng.integers(8))] for _ in range(nt - 2)]
    test_masks = [MASKS[int(rng.integers(8, len(MASKS)))] for _ in range(nq)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, masks in (('train', train_masks), ('test', test_masks)):
        for index, mask in enumerate(masks):
            count = (2 if index == 0 else 4 if index == 1 else int(rng.integers(2, 5))) if split == 'train' else int(rng.integers(5, 7))
            if not shared_palette:
                palette = tuple(map(int, rng.choice(10, 2, replace=False)))
            witness = (split == 'train' and index == 1) or split == 'test'
            for _ in range(8):
                try:
                    pair = _make_pair(rng, mask, count, palette, witness, split == 'train' and index == 0)
                except _SamplingExhausted:
                    continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded scene/distinct-input retries exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            scene = _read_scene(inp)
            assert np.array_equal(out, _render(inp, scene)), 'full-grid repair/rerouting mismatch'
            assert np.count_nonzero(out == scene['ink']) == len(scene['mask'])
            assert set(map(tuple, np.argwhere(out == scene['ink']).tolist())) == _pose(scene['mask'], scene['points'][-1], scene['directions'][-1])
            info = _metrics(inp, scene, out)
            assert info['no_rewire_diff'] and info['retained_gaps'] > 0
            assert any(info['old_bends'])
            records[split].append(info)
    train = records['train']
    assert len({m['mask'] for m in train}) >= 2
    assert len({m['stations'] for m in train}) >= 2
    assert any(m['initial'] != m['final'] for m in train)
    assert any(m['order_witness'] for m in train), 'temporal crossing ownership needs a demonstration'
    assert any(m['transverse_witness'] for m in train), 'perpendicular new-ray crossing needs a demonstration'
    assert any(m['restored_crossings'] for m in train), 'exposure of an underlying wire needs a demonstration'
    return None


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _semantic_fixture():
    """Hand-specified expected geometry, including a mask outside the sampler.

    The oracle below is a direct list of retained arms and four explicit slices;
    it does not call the reconstruction, branch follower, ray or pose renderer.
    """
    inp = np.zeros((20, 24), dtype=np.int8)
    vertices = {
        1: [(19, 7), (5, 7), (5, 10), (1, 10)],
        2: [(11, 0), (11, 10), (15, 10), (15, 14)],
        3: [(19, 17), (15, 17), (15, 20), (18, 20)],
        4: [(11, 23), (11, 20), (8, 20), (8, 4)],
    }
    # Color 1 visibly covers color 4 at (8,7); both old branches meet at
    # (11,7). Neither ownership fact is a global literal-color hierarchy.
    for c in (4, 2, 3, 1):
        for a, b in zip(vertices[c], vertices[c][1:]):
            for p in _line(a, b):
                inp[p] = c
    for p in ((5, 10), (15, 10), (15, 20), (8, 20), (5, 3),
              (4, 2), (6, 2), (4, 1), (6, 1), (4, 0), (6, 0)):
        inp[p] = 8
    inp[8, 12:14] = 0  # two consecutive omitted retained cells
    inp[3, 10] = 0
    expected = np.zeros_like(inp)
    expected[1:6, 10] = 1
    expected[15, 10:15] = 2
    expected[15:19, 20] = 3
    expected[8, 4:21] = 4
    expected[5, 11:24] = 1
    expected[16:20, 10] = 2
    expected[15, 21:24] = 3
    expected[0:8, 20] = 4
    for p in ((8, 20), (9, 19), (9, 21), (10, 19), (10, 21), (11, 19), (11, 21)):
        expected[p] = 8
    for k in range(4):
        for reflect in (False, True):
            a = np.fliplr(inp) if reflect else inp
            b = np.fliplr(expected) if reflect else expected
            assert np.array_equal(solve(np.rot90(a, k)), np.rot90(b, k)), 'hand-specified semantic fixture failed'
    assert expected[8, 7] == 4 and expected[5, 20] == 4 and expected[11, 7] == 0


def self_test():
    """42 sequential seeds; bounded memory, no saved puzzles or execution traces."""
    _semantic_fixture()
    digests, shapes, masks, routes, palettes = set(), set(), set(), set(), set()
    directions, bends, counts = set(), set(), set()
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _puzzle_digest(puzzle)
        digests.add(digest)
        if seed in (0, 17, 41):
            assert digest == _puzzle_digest(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                scene = _read_scene(inp)
                m = _metrics(inp, scene, out)
                shapes.add(inp.shape)
                masks.add(m['mask'])
                p0 = scene['points'][0]
                routes.add(tuple((r - p0[0], c - p0[1]) for r, c in scene['points']))
                palettes.add(m['palette'])
                directions.update(m['directions'])
                bends.update(m['old_bends'])
                counts.add(m['stations'])
                totals.update({k: int(m[k]) for k in ('junctions', 'restored_crossings', 'retained_gaps', 'gaps', 'order_witness')})
                # Rotation/reflection equivariance exercises the input-only API.
                before = inp.copy()
                assert np.array_equal(solve(inp), out)
                transformed = np.rot90(np.fliplr(inp)).copy()
                assert np.array_equal(solve(transformed), np.rot90(np.fliplr(out)))
                assert np.array_equal(inp, before), 'solve mutated its input'
        if seed < 2:
            pair = puzzle['test'][0]
            bad = {'train': puzzle['train'], 'test': [dict(pair)]}
            bad['test'][0]['output'] = pair['output'].copy()
            bad['test'][0]['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('changed output accepted')
    assert len(digests) == 42 and len(shapes) >= 45 and len(routes) >= 150, 'insufficient puzzle/shape/route variation'
    assert len(masks) >= 12 and len(palettes) >= 100, 'insufficient visible mask/palette variation'
    assert directions == set(range(4)) and counts == {2, 3, 4, 5, 6}, 'missing direction or station count'
    assert {1, 2} <= bends, 'one- and multi-bend old branches must occur'
    assert all(totals[k] > 0 for k in totals)
    for nt, nq in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(77, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    try:
        _itinerary((5, 5), {(5, 9), (9, 5)})
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous itinerary accepted')
    return {'seeds': 42, 'distinct_puzzles': len(digests), 'shapes': len(shapes),
            'masks': len(masks), 'routes': len(routes), 'palettes': len(palettes),
            'totals': dict(totals)}
