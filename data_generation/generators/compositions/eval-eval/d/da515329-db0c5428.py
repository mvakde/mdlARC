"""Keyed pinwheel relays: spatial keys choose half-turned tiles on winding arms.

Only NumPy and the Python standard library are used. No parent module, fixture,
seed, or generation metadata is consulted by solve().
"""
from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np


# Clockwise order. Each arm owns a cardinal tile and the next corner tile.
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1))
CARDINALS = DIRECTIONS[::2]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _box(shape, r, c):
    return (slice(max(0, r - 1), min(shape[0], r + 2)),
            slice(max(0, c - 1), min(shape[1], c + 2)))


def _base(shape, center, pitch, radius, bg, carrier, tiles):
    a = np.full(shape, bg, dtype=np.int8)
    cy, cx = center
    for dy, dx in CARDINALS:
        for d in range(2, pitch * radius + 1):
            a[cy + dy * d, cx + dx * d] = carrier
    for (dy, dx), tile in zip(DIRECTIONS, tiles):
        r, c = cy + pitch * dy, cx + pitch * dx
        a[r - 1:r + 2, c - 1:c + 2] = tile
    return a


def _parse(value):
    """Recover the unique ring, lattice, spoke radius, and singleton markers."""
    a = _grid(value)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if counts.max() * 2 <= a.size:
        raise ValueError("background must be a strict majority")
    occupied = a != bg
    if min(a.shape) < 3:
        raise ValueError("missing tile ring")
    full = np.lib.stride_tricks.sliding_window_view(occupied, (3, 3)).all(axis=(2, 3))
    starts = np.argwhere(full)
    answers = []
    for pitch in (3, 4):
        # Every valid ring has a full north tile; use its top-left as an anchor.
        for tr, tc in starts:
            cy, cx = int(tr + 1 + pitch), int(tc + 1)
            if not (pitch + 1 <= cy < a.shape[0] - pitch - 1
                    and pitch + 1 <= cx < a.shape[1] - pitch - 1):
                continue
            if np.any(a[cy - 1:cy + 2, cx - 1:cx + 2] != bg):
                continue
            tiles = []
            for dy, dx in DIRECTIONS:
                r, c = cy + pitch * dy, cx + pitch * dx
                if not full[r - 1, c - 1]:
                    break
                tiles.append(a[r - 1:r + 2, c - 1:c + 2].copy())
            if len(tiles) != 8:
                continue
            vv, cc = np.unique(np.stack(tiles), return_counts=True)
            carrier = int(vv[np.argmax(cc)])
            if cc.max() <= 36:
                continue
            samples = tuple(int(t[1 - dy, 1 - dx])
                            for t, (dy, dx) in zip(tiles, DIRECTIONS))
            if (any(s in (bg, carrier) for s in samples)
                    or any(samples[i] == samples[i + 1] for i in range(0, 8, 2))
                    or any(not np.any(t == carrier) for t in tiles)):
                continue
            lengths = []
            for dy, dx in CARDINALS:
                d = pitch + 2
                while (0 <= cy + d * dy < a.shape[0]
                       and 0 <= cx + d * dx < a.shape[1]
                       and a[cy + d * dy, cx + d * dx] == carrier):
                    d += 1
                lengths.append(d - 1)
            if len(set(lengths)) != 1 or lengths[0] % pitch:
                continue
            radius = lengths[0] // pitch
            if not 2 <= radius <= 4:
                continue
            if min(cy, cx, a.shape[0] - 1 - cy, a.shape[1] - 1 - cx) < pitch * radius:
                continue
            base = _base(a.shape, (cy, cx), pitch, radius, bg, carrier, tiles)
            if np.any(a[base != bg] != base[base != bg]):
                continue
            markers = []
            valid = True
            for r, c in np.argwhere((a != bg) & (base == bg)):
                r, c = int(r), int(c)
                color = int(a[r, c])
                # Tile destinations must not overwrite any original seed cell.
                if (color == carrier or (r - cy) % pitch or (c - cx) % pitch
                        or (abs(r - cy) <= 1 and abs(c - cx) <= 1)
                        or np.any(base[_box(a.shape, r, c)] != bg)):
                    valid = False
                    break
                markers.append((r, c, color))
            if valid and markers:
                answers.append(dict(grid=a, background=bg, carrier=carrier,
                                    center=(cy, cx), pitch=pitch, radius=radius,
                                    tiles=tuple(tiles), samples=samples,
                                    markers=tuple(markers), base=base))
    if len(answers) != 1:
        raise ValueError("input must encode exactly one valid keyed ring and cross")
    return answers[0]


def _walk_geometry(shape, center, pitch, radius):
    """Continue four infinite-plane paths; clip segments, never stop at exits."""
    wire = np.zeros(shape, dtype=bool)
    owner = np.full(shape, -1, dtype=np.int8)
    cy, cx = center

    def paint(y, x, ny, nx, phase):
        r0, c0 = cy + pitch * y, cx + pitch * x
        r1, c1 = cy + pitch * ny, cx + pitch * nx
        if r0 == r1 and 0 <= r0 < shape[0]:
            lo, hi = max(0, min(c0, c1)), min(shape[1] - 1, max(c0, c1))
            if lo <= hi:
                wire[r0, lo:hi + 1] = True
        elif c0 == c1 and 0 <= c0 < shape[1]:
            lo, hi = max(0, min(r0, r1)), min(shape[0] - 1, max(r0, r1))
            if lo <= hi:
                wire[lo:hi + 1, c0] = True
        dy = int(np.sign(ny - y))
        dx = int(np.sign(nx - x))
        for d in range(abs(ny - y) + abs(nx - x) + 1):
            yy, xx = y + d * dy, x + d * dx
            r, c = cy + pitch * yy, cx + pitch * xx
            if (yy, xx) != (0, 0) and 0 <= r < shape[0] and 0 <= c < shape[1]:
                if owner[r, c] not in (-1, phase):
                    raise AssertionError("winding phases intersect")
                owner[r, c] = phase

    distance = max(cy, cx, shape[0] - 1 - cy, shape[1] - 1 - cx) // pitch + 1
    for phase, (dy, dx) in enumerate(CARDINALS):
        y, x = radius * dy, radius * dx
        paint(0, 0, y, x, phase)
        dy, dx = dx, -dy
        for k in range(4 * (distance // 8 + 3)):
            for vy, vx, steps in ((dy, dx, 2 + 4 * k),
                                  (dx, -dy, radius - 2), (dy, dx, radius)):
                ny, nx = y + steps * vy, x + steps * vx
                paint(y, x, ny, nx, phase)
                y, x = ny, nx
            dy, dx = dx, -dy
    wire[cy, cx] = False
    return wire, owner


def _interval_geometry(shape, center, pitch, radius):
    """Closed-form segment membership, separately from the procedural walker."""
    yy, xx = np.indices(shape)
    yy, xx = yy - center[0], xx - center[1]
    lattice = (yy % pitch == 0) & (xx % pitch == 0)
    wire = np.zeros(shape, dtype=bool)
    owner = np.full(shape, -1, dtype=np.int8)
    bound = max(np.abs(yy).max(), np.abs(xx).max()) // pitch + 1
    for phase in range(4):
        mask = (xx == 0) & (yy >= -radius * pitch) & (yy <= -1)
        for q in range(int(bound) // 8 + 3):
            starts = ((-radius - 8*q, -8*q), (-2 - 8*q, radius + 8*q + 2),
                      (radius + 8*q + 4, 8*q + 4), (8*q + 6, -radius - 8*q - 6))
            for s, ((y, x), (dy, dx)) in enumerate(zip(
                    starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along = (yy - pitch*y)*dy + (xx - pitch*x)*dx
                across = (yy - pitch*y)*dx - (xx - pitch*x)*dy
                a = (16*q + 4*s + 2) * pitch
                jog, forward = (radius - 2) * pitch, radius * pitch
                mask |= ((across == 0) & (0 <= along) & (along <= a))
                mask |= ((along == a) & (0 <= across) & (across <= jog))
                mask |= ((across == jog) & (a <= along) & (along <= a + forward))
        wire |= mask
        nodes = mask & lattice
        if np.any(nodes & (owner >= 0)):
            raise AssertionError("closed-form phases intersect")
        owner[nodes] = phase
        yy, xx = -xx, yy
    return wire, owner


def _accepted(info, owner):
    result = []
    for r, c, color in info['markers']:
        phase = int(owner[r, c])
        if phase >= 0:
            for index in (2 * phase, 2 * phase + 1):
                if color == info['samples'][index]:
                    result.append((r, c, index))
    return result


def _render(info, geometry):
    a = info['grid']
    wire, owner = geometry
    b = a.copy()
    b[wire & (a == info['background'])] = info['carrier']
    cy, cx = info['center']
    b[cy - 1:cy + 2, cx - 1:cx + 2] = info['carrier']
    for (dy, dx), sample in zip(DIRECTIONS, info['samples']):
        b[cy + dy, cx + dx] = sample
    for r, c, index in _accepted(info, owner):
        target = _box(a.shape, r, c)
        tile = info['tiles'][index][::-1, ::-1]
        top, left = max(0, 1 - r), max(0, 1 - c)
        height, width = b[target].shape
        b[target] = tile[top:top + height, left:left + width]
    return b


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return _render(info, _walk_geometry(info['grid'].shape, info['center'],
                                       info['pitch'], info['radius']))


def _d4(tile):
    return [np.rot90(tile, k) for k in range(4)] + [np.rot90(tile[:, ::-1], k) for k in range(4)]


def _samples(rng, details, count):
    for _ in range(100):
        labels = list(range(count)) + list(map(int, rng.integers(count, size=8-count)))
        rng.shuffle(labels)
        if all(labels[i] != labels[i+1] for i in range(0, 8, 2)):
            return tuple(details[i] for i in labels)
    # Every count from 2 to 8 has a finite, valid cyclic fallback.
    return tuple(details[i % count] for i in range(8))


def _tiles(rng, carrier, samples, details, witness, repeated):
    tiles = []
    for index, ((dy, dx), sample) in enumerate(zip(DIRECTIONS, samples)):
        inward = (1 - dy, 1 - dx)
        tile = np.full((3, 3), carrier, dtype=np.int8)
        tile[inward] = sample
        positions = [i for i in range(9) if i != 3*inward[0] + inward[1]]
        rng.shuffle(positions)
        if witness:
            auxiliary = int(rng.choice([v for v in details if v != sample]))
            for flat in positions:
                trial = tile.copy()
                trial.flat[flat] = auxiliary
                if len({t.tobytes() for t in _d4(trial)}) == 8:
                    tile = trial
                    break
            else:
                raise AssertionError("no D4 witness")
        else:
            count = int(rng.integers(0, 4))
            for flat in positions[:count]:
                tile.flat[flat] = sample if rng.random() < .45 else int(rng.choice(details))
            if repeated and index == 0:
                tile.flat[positions[0]] = sample
        tiles.append(tile)
    return tuple(tiles)


def _layout(rng, pitch, radius, generous, edge):
    low = max(17, 2 * radius * pitch + 1, 6*pitch + 3 if generous else 1)
    if generous:
        low = max(low, 27)
    mode = int(rng.integers(5))
    if mode == 0:
        h = w = int(rng.integers(low, 31))
    elif mode == 1:
        h, w = (low, 30) if rng.random() < .5 else (30, low)
    else:
        h, w = map(int, rng.integers(low, 31, size=2))
    if generous and pitch == 4:
        # A first radius-two corner lies four lattice steps sideways. At pitch
        # four, one 30-cell axis is needed to expose it within ARC's size bound.
        if rng.random() < .5:
            h = 30
        else:
            w = 30
    margin = 3*pitch + 1 if generous else radius*pitch
    center = [int(rng.integers(margin, n-margin)) for n in (h, w)]
    if generous and pitch == 4:
        axis = 0 if h == 30 else 1
        center[axis] = int(rng.choice([margin, 29-margin]))
    if generous and radius == 3:
        # The ample opposite corner shows the first complete jog and the next
        # forward run (six), including its jog; a centered crop can hide them.
        center = [int(rng.choice([margin, n-1-margin])) for n in (h, w)]
    if edge:
        side = int(rng.integers(4))
        axis = side // 2
        center[axis] = margin if side % 2 == 0 else (h, w)[axis]-1-margin
    elif not generous and rng.random() < .4:
        center = [(h-1)//2, (w-1)//2]
    return (h, w), tuple(center)


def _new_pair(rng, radius, pitch, *, witness=False, repeated=False,
              generous=False, edge=False, palette=None):
    order = list(map(int, rng.permutation(10))) if palette is None else list(palette)
    bg, carrier, details = order[0], order[1], order[2:]
    count = int(rng.integers(2, 4)) if witness else int(rng.integers(2, 9))
    minimum = max(3, count)  # Two keys plus a wrong-key color must be possible.
    maximum = min(8, minimum + 2) if witness else 8
    details = details[:int(rng.integers(minimum, maximum + 1))]
    samples = _samples(rng, details, count)
    tiles = _tiles(rng, carrier, samples, details, witness, repeated)
    shape, center = _layout(rng, pitch, radius, generous, edge)
    base = _base(shape, center, pitch, radius, bg, carrier, tiles)
    wire, owner = _walk_geometry(shape, center, pitch, radius)
    cy, cx = center
    on = [[] for _ in range(4)]
    off = []
    for r in range(cy % pitch, shape[0], pitch):
        for c in range(cx % pitch, shape[1], pitch):
            if ((abs(r-cy) <= 1 and abs(c-cx) <= 1)
                    or np.any(base[_box(shape, r, c)] != bg)):
                continue
            if generous and not (1 <= r < shape[0]-1 and 1 <= c < shape[1]-1):
                continue
            phase = int(owner[r, c])
            (off if phase < 0 else on[phase]).append((r, c))
    for points in on + [off]:
        rng.shuffle(points)
    markers = {}

    def matched(point, phase, which=None):
        index = 2*phase + (int(rng.integers(2)) if which is None else which)
        markers[point] = samples[index]

    if generous:
        # Both templates and a wrong-key control in every arm, all fully visible.
        for phase in range(4):
            if len(on[phase]) < 3:
                raise AssertionError("teaching crop does not expose every arm")
            matched(on[phase][0], phase, 0)
            matched(on[phase][1], phase, 1)
            wrong = [v for v in details if v not in samples[2*phase:2*phase+2]]
            markers[on[phase][2]] = int(rng.choice(wrong))
    else:
        choices = [(point, phase) for phase, points in enumerate(on) for point in points]
        if edge:
            choices = [(point, phase) for point, phase in choices
                       if min(point[0], point[1], shape[0]-1-point[0], shape[1]-1-point[1]) == 0]
        if not choices:
            raise AssertionError("no matching destination for the requested crop")
        point, phase = choices[int(rng.integers(len(choices)))]
        matched(point, phase)
        others = [(pt, ph) for ph, points in enumerate(on) for pt in points if pt not in markers]
        rng.shuffle(others)
        point, phase = others.pop()
        matched(point, phase)
        point, phase = others.pop()
        markers[point] = int(rng.choice([v for v in details if v not in samples[2*phase:2*phase+2]]))
    if not off:
        raise AssertionError("missing off-path control")
    markers[off[0]] = int(rng.choice(samples))
    if len(off) > 1:
        markers[off[1]] = int(rng.choice(samples))
    all_points = [(pt, ph) for ph, points in enumerate(on) for pt in points]
    all_points += [(pt, -1) for pt in off]
    rng.shuffle(all_points)
    budget = int(rng.integers(max(14, len(markers)), 25))
    for point, phase in all_points:
        if len(markers) >= budget:
            break
        if point in markers:
            continue
        if phase >= 0 and rng.random() < .65:
            matched(point, phase)
        else:
            markers[point] = int(rng.choice(details))
    a = base.copy()
    for point, color in markers.items():
        a[point] = color
    info = dict(grid=a, background=bg, carrier=carrier, center=center, pitch=pitch,
                radius=radius, tiles=tiles, samples=samples,
                markers=tuple((r, c, color) for (r, c), color in markers.items()), base=base)
    return dict(input=a, output=_render(info, (wire, owner)))


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 'num_train', 2, 10)
    m = _count(num_test, 'num_test', 1, 4)
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    scale_shift = bool(rng.integers(2))
    palette = tuple(map(int, rng.permutation(10))) if rng.random() < .15 else None
    puzzle = dict(train=[], test=[])
    seen = set()
    plans = [(2, 3, True, False, True, False),
             (3, 3, False, True, True, False)]
    for index in range(n - 2):
        radius = 2 if index == 0 else int(rng.choice([2, 3] if scale_shift else [2, 3, 4]))
        pitch = 4 if index == 0 else (3 if radius == 4 else int(rng.integers(3, 5)))
        plans.append((radius, pitch, False, False, False, False))
    for i in range(m):
        radius = 4 if scale_shift else int(rng.integers(2, 4))
        pitch = 3 if radius == 4 or n == 2 else int(rng.integers(3, 5))
        plans.append((radius, pitch, False, False, False, not scale_shift or i == 0))
    for serial, (radius, pitch, witness, repeated, generous, edge) in enumerate(plans):
        for _ in range(64):
            pair = _new_pair(rng, radius, pitch, witness=witness, repeated=repeated,
                             generous=generous, edge=edge, palette=palette)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle['train' if serial < n else 'test'].append(pair)
                break
        else:
            raise AssertionError("distinct input retry bound exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    info = _parse(pair['input'])
    a, b = info['grid'], _grid(pair['output'])
    assert a.shape == b.shape
    wire, owner = _interval_geometry(a.shape, info['center'], info['pitch'], info['radius'])
    expected = a.copy()
    assigned = wire & (a == info['background'])
    expected[assigned] = info['carrier']
    cy, cx = info['center']
    # Scalar code and copy checks; do not use the slice-based answer renderer.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            expected[cy+dy, cx+dx] = (info['carrier'] if (dy, dx) == (0, 0)
                                     else info['samples'][DIRECTIONS.index((dy, dx))])
            assigned[cy+dy, cx+dx] = True
    accepted = _accepted(info, owner)
    for r, c, index in accepted:
        tile = info['tiles'][index]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if 0 <= r+dy < a.shape[0] and 0 <= c+dx < a.shape[1]:
                    expected[r+dy, c+dx] = tile[1-dy, 1-dx]
                    assigned[r+dy, c+dx] = True
    assert np.array_equal(b, expected), "wrong wires, center code, or tile transport"
    assert np.array_equal(a[~assigned], b[~assigned]), "unassigned cells changed"
    assert np.all(b[info['base'] != info['background']] == a[info['base'] != info['background']])
    assert set(map(int, np.unique(a))) == set(map(int, np.unique(b)))
    assert len(accepted) >= 2, "both mechanisms need nontrivial targets"
    assert any(owner[r, c] < 0 and color in info['samples'] for r, c, color in info['markers'])
    assert any(owner[r, c] >= 0 and color not in info['samples'][2*owner[r, c]:2*owner[r, c]+2]
               for r, c, color in info['markers']), "missing wrong-key control"
    info.update(owner=owner, accepted=accepted)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, m = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, data = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(info['grid'])
            assert key not in seen, "duplicate input"
            seen.add(key)
            data[split].append(info)
    train = data['train']
    assert {2, 3} <= {i['radius'] for i in train}, "jog dependence not demonstrated"
    assert {i['pitch'] for i in data['test']} <= {i['pitch'] for i in train}, "unseen lattice pitch"
    assert any(a['pitch'] == b['pitch'] and a['radius'] != b['radius']
               for a in train for b in train), "pitch and radius are confounded"
    def contains_points(info, points):
        cy, cx = info['center']
        pitch = info['pitch']
        h, w = info['grid'].shape
        for _ in range(4):
            if all(0 <= cy+pitch*y < h and 0 <= cx+pitch*x < w for y, x in points):
                return True
            points = [(x, -y) for y, x in points]
        return False
    assert any(i['radius'] == 2 and contains_points(i, [(-2, 0), (-2, 4), (-1, 4)])
               for i in train), "no visible first radius-two corner"
    assert any(i['radius'] == 3 and contains_points(i, [(-3, 0), (-3, 2), (-2, 2),
                                                        (-2, 5), (4, 5), (4, 4)])
               for i in train), "forward-run growth and jog not visible"
    # One full transport of each tile with unique inward key and eight D4 views
    # resolves every sample coordinate and tile operation, even with two examples.
    assert any(len(set(i['samples'])) < 4 and all(
        len({v.tobytes() for v in _d4(t)}) == 8
        and np.count_nonzero(t == i['samples'][j]) == 1
        and any(k == j and 1 <= r < i['grid'].shape[0]-1
                and 1 <= c < i['grid'].shape[1]-1 for r, c, k in i['accepted'])
        for j, t in enumerate(i['tiles'])) for i in train), "missing phase and rotation witnesses"
    assert any(any(np.count_nonzero(t == s) > 1 for t, s in zip(i['tiles'], i['samples']))
               for i in train), "repeated positional keys not demonstrated"
    assert any(any(min(r, c, i['grid'].shape[0]-1-r, i['grid'].shape[1]-1-c) == 0
                   for r, c, _ in i['accepted']) for i in data['test']), "missing boundary shift"
    # Radius-four queries must be backed by radius-two and radius-three training.
    if any(i['radius'] == 4 for i in data['test']):
        assert all(i['radius'] <= 3 for i in train)


def _signature(puzzle):
    return tuple(_key(np.asarray(pair[role], dtype=np.int8))
                 for split in ('train', 'test') for pair in puzzle[split]
                 for role in ('input', 'output'))


def self_test():
    """Bounded geometric, counterfactual, corruption and 42-seed variation checks."""
    shapes, layouts, palettes, sample_patterns, signatures = set(), set(), set(), set(), set()
    radii, pitches, palette_sizes, features = set(), set(), set(), Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                a = info['grid']
                args = (a.shape, info['center'], info['pitch'], info['radius'])
                walk, analytic = _walk_geometry(*args), _interval_geometry(*args)
                assert all(np.array_equal(x, y) for x, y in zip(walk, analytic))
                assert np.array_equal(solve(a), pair['output'])
                shapes.add(a.shape)
                layouts.add((*a.shape, *info['center'], info['pitch'], info['radius']))
                palettes.add((info['background'], info['carrier']))
                palette_sizes.add(len(np.unique(a)))
                radii.add(info['radius'])
                pitches.add(info['pitch'])
                labels = {}
                sample_patterns.add(tuple(labels.setdefault(s, len(labels)) for s in info['samples']))
                accepted = _accepted(info, walk[1])
                features['copies'] += len(accepted)
                features['clipped'] += sum(min(r, c, a.shape[0]-1-r, a.shape[1]-1-c) == 0
                                           for r, c, _ in accepted)
                features['repeated'] += sum(np.count_nonzero(t == s) > 1
                                            for t, s in zip(info['tiles'], info['samples']))
                features['radius_four_test'] += int(split == 'test' and info['radius'] == 4)
                # Omitting A: stamp any matching key without checking winding membership.
                off = next((r, c, color) for r, c, color in info['markers']
                           if walk[1][r, c] < 0 and color in info['samples'])
                r, c, _ = off
                assert np.count_nonzero(pair['output'][_box(a.shape, r, c)] != info['background']) == 1
                without_a = pair['output'].copy()
                index = info['samples'].index(off[2])
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if 0 <= r+dy < a.shape[0] and 0 <= c+dx < a.shape[1]:
                            without_a[r+dy, c+dx] = info['tiles'][index][1-dy, 1-dx]
                assert not np.array_equal(without_a, pair['output'])
                # Omitting B: use the cardinal tile on every marked path node.
                r, c, color = next((r, c, color) for r, c, color in info['markers']
                                  if walk[1][r,c] >= 0 and color not in
                                  info['samples'][2*walk[1][r,c]:2*walk[1][r,c]+2])
                without_b = pair['output'].copy()
                tile = info['tiles'][2*int(walk[1][r,c])]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if 0 <= r+dy < a.shape[0] and 0 <= c+dx < a.shape[1]:
                            without_b[r+dy, c+dx] = tile[1-dy, 1-dx]
                assert not np.array_equal(without_b, pair['output'])
                # Crop equivalence guards boundary exit and later re-entry.
                big = _walk_geometry((90, 90), (45, 45), info['pitch'], info['radius'])
                top, left = 45-info['center'][0], 45-info['center'][1]
                assert all(np.array_equal(x, y[top:top+a.shape[0], left:left+a.shape[1]])
                           for x, y in zip(walk, big))
                if split == 'test':
                    assert np.array_equal(solve(np.rot90(a)), np.rot90(pair['output']))
                    permutation = np.roll(np.arange(10, dtype=np.int8), 3)
                    assert np.array_equal(solve(permutation[a]), permutation[pair['output']])
    assert len(signatures) == 42 and len(shapes) >= 40 and len(layouts) >= 120
    assert len(palettes) >= 40 and len(sample_patterns) >= 25
    assert len(palette_sizes) >= 4, "palette richness collapsed"
    assert radii == {2, 3, 4} and pitches == {3, 4}
    assert all(features[k] > 0 for k in ('copies', 'clipped', 'repeated', 'radius_four_test'))
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(123, n, m)
            assert len(p['train']) == n and len(p['test']) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    good = generate_puzzle(0)
    for location in ('code', 'source', 'copy', 'unused'):
        bad = copy.deepcopy(good)
        pair = bad['test'][0]
        info = _parse(pair['input'])
        _, owner = _walk_geometry(info['grid'].shape, info['center'], info['pitch'], info['radius'])
        cy, cx = info['center']
        r, c, _ = _accepted(info, owner)[0]
        points = dict(code=(cy, cx), source=(cy-info['pitch'], cx), copy=(r, c),
                      unused=tuple(map(int, np.argwhere((pair['input'] == info['background'])
                                                        & (pair['output'] == info['background']))[0])))
        point = points[location]
        pair['output'][point] = (int(pair['output'][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corruption accepted")
    return dict(seeds=42, shapes=len(shapes), layouts=len(layouts), palettes=len(palettes),
                palette_sizes=sorted(palette_sizes), sample_patterns=len(sample_patterns),
                **{k: int(v) for k, v in features.items()})


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
