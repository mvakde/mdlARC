"""Mirror house: reflect chamber identities, not physical pixel coordinates.

One enclosed framework has ordered horizontal bands, with a palindromic but
nonconstant number of rooms per band. The two involutions reverse band order
and reverse room order within a band. Colors are constant on their orbits.
Geometry need not be symmetric. Blank rooms receive a surviving orbit color.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, points = [start], []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(np.array(sorted(points), dtype=int))
    return result


def _read_house(grid):
    """Recover all conventions and ordinal room coordinates from visible walls."""
    a = _grid(grid)
    bg = int(a[0, 0])
    if not all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])):
        raise ValueError("the exterior must have a blank border")
    occupied = np.argwhere(a != bg)
    if not len(occupied):
        raise ValueError("missing framework")
    r, c = occupied.min(0)
    R, C = occupied.max(0) + 1
    crop = a[r:R, c:C].copy()
    wall = int(crop[0, 0])
    if wall == bg or not all(np.all(edge == wall) for edge in
                            (crop[0], crop[-1], crop[:, 0], crop[:, -1])):
        raise ValueError("framework must have a closed rectangular perimeter")
    separators = list(map(int, np.flatnonzero(np.all(crop == wall, axis=1))))
    if any(b - x < 2 for x, b in zip(separators, separators[1:])):
        raise ValueError("empty or thick horizontal separator")
    bands = []
    for top, bottom in zip(separators, separators[1:]):
        rooms = [p + (top + 1, 0) for p in
                 _components(crop[top + 1:bottom] != wall)]
        # Every room intersects every row of its band. Thus left-to-right
        # order is visible without area, centroid, or generation labels.
        if not rooms or any(len(set(p[:, 0])) != bottom - top - 1 for p in rooms):
            raise ValueError("rooms must span their band")
        rooms.sort(key=lambda p: int(p[p[:, 0] == top + 1, 1].min()))
        for y in range(top + 1, bottom):
            spans = [(int(p[p[:, 0] == y, 1].min()),
                      int(p[p[:, 0] == y, 1].max())) for p in rooms]
            if any(right >= left for (_, right), (left, _) in zip(spans, spans[1:])):
                raise ValueError("room order crosses inside a band")
        for p in rooms:
            if len(np.unique(crop[tuple(p.T)])) != 1:
                raise ValueError("every room must be uniformly colored or blank")
        bands.append(rooms)
    counts = tuple(map(len, bands))
    if not counts or counts != counts[::-1]:
        raise ValueError("opposite bands must have the same room count")
    return bg, wall, crop, (int(r), int(c), int(R), int(C)), bands


def _orbits(counts):
    seen, result = set(), []
    for r, count in enumerate(counts):
        for c in range(count):
            if (r, c) not in seen:
                orbit = tuple(sorted({(r, c), (r, count - 1 - c),
                                      (len(counts) - 1 - r, c),
                                      (len(counts) - 1 - r, count - 1 - c)}))
                seen.update(orbit)
                result.append(orbit)
    return result


def _restore(info):
    bg, wall, crop, box, bands = info
    output = crop.copy()
    for orbit in _orbits(tuple(map(len, bands))):
        colors = {int(crop[tuple(bands[r][c][0])]) for r, c in orbit}
        colors.discard(bg)
        if len(colors) != 1:
            raise ValueError("empty or contradictory chamber orbit")
        color = next(iter(colors))
        for r, c in orbit:
            p = bands[r][c]
            if crop[tuple(p[0])] == bg:
                output[tuple(p.T)] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill blank chambers and return the tight framework crop; input only."""
    return _restore(_read_house(input_grid))


def _positive_partition(rng, total, count):
    parts = np.ones(count, dtype=int)
    for _ in range(total - count):
        parts[int(rng.integers(count))] += 1
    return parts


def _profile(rng, n_bands, max_count, large=False):
    for _ in range(100):
        half = [int(rng.integers(1, max_count + 1)) for _ in range((n_bands + 1) // 2)]
        if large:
            half[int(rng.integers(len(half)))] = int(rng.integers(5, 8))
        counts = tuple(half + half[:n_bands // 2][::-1])
        orbits = _orbits(counts)
        if (len(set(counts)) > 1 and len(orbits) >= 3
                and sum(len(o) == 4 for o in orbits) >= 2
                and sum(len(o) > 1 for o in orbits) >= 3):
            return counts
    # A bounded construction guard with the same rule, never a new convention.
    half = [4] * ((n_bands + 1) // 2)
    half[-1] = 3
    if large:
        half[0] = 6
    return tuple(half + half[:n_bands // 2][::-1])


def _floorplan(rng, counts):
    """Render an irregular ordered partition; labels are construction-only."""
    n = len(counts)
    width = int(rng.integers(2 * max(counts) + 3,
                             min(28, 4 * max(counts) + 9) + 1))
    heights = np.full(n, 2, dtype=int)
    extras = int(rng.integers(min(28 - (3 * n + 1), 3 * n) + 1))
    for _ in range(extras):
        eligible = np.flatnonzero(heights < 5)
        heights[int(rng.choice(eligible))] += 1
    labels = np.full((int(heights.sum()) + n + 1, width), -1, dtype=int)
    band_y, room_id = 1, 0
    for count, height in zip(counts, heights):
        widths = _positive_partition(rng, width - count - 1, count)
        cuts = np.concatenate(([0], np.cumsum(widths + 1)))
        patch = np.full((int(height), width), -1, dtype=int)
        for c in range(count):
            patch[:, cuts[c] + 1:cuts[c + 1]] = room_id + c
        # Move short portions of dividers by one pixel, preserving enclosures.
        # Verification uses labels only to ensure each trial retains the rooms.
        for _ in range(int(rng.integers(count + 1, 3 * count + 2))):
            if count < 2:
                break
            k = int(rng.integers(1, count))
            x, y = int(cuts[k]), int(rng.integers(height))
            sign = int(rng.choice((-1, 1)))
            if patch[y, x] != -1 or not 0 < x + sign < width - 1:
                continue
            trial = patch.copy()
            trial[y, x] = room_id + (k - 1 if sign > 0 else k)
            trial[y, x + sign] = -1
            # Wall diagonals separate rooms in four-connectivity. No leak or
            # room loss is allowed; every room still crosses the entire band.
            if all(len(_components(trial == room_id + c)) == 1 and
                   all(np.any(trial[y0] == room_id + c) for y0 in range(height))
                   for c in range(count)):
                # Also forbid joining different label regions across a leak.
                if len(_components(trial >= 0)) == count:
                    patch = trial
        # Small corner notches produce additional local shape changes.
        for c in range(count):
            color_mask = patch == room_id + c
            p = np.argwhere(color_mask)
            if len(p) < 4 or rng.random() < 0.35:
                continue
            y = 0 if rng.random() < 0.5 else int(height) - 1
            xs = np.flatnonzero(color_mask[y])
            x = int(xs[0] if rng.random() < 0.5 else xs[-1])
            trial = color_mask.copy()
            trial[y, x] = False
            if trial[y].any() and len(_components(trial)) == 1:
                patch[y, x] = -1
        labels[band_y:band_y + height] = patch
        room_id += count
        band_y += int(height) + 1
    return labels


def _logical_mask(rng, counts, orbits, hard):
    points = [(r, c) for r, n in enumerate(counts) for c in range(n)]
    mode = int(rng.integers(4))
    if mode == 0:
        blank = {p for p in points if rng.random() < 0.38}
    elif mode == 1:
        band = int(rng.integers(len(counts)))
        blank = {p for p in points if p[0] == band}
    elif mode == 2:
        right = bool(rng.integers(2))
        blank = {p for p in points if (p[1] >= counts[p[0]] // 2) == right}
    else:
        start = int(rng.integers(len(counts)))
        blank = {p for p in points if p[0] >= start and
                 p[1] <= (p[0] - start) % counts[p[0]]}
    four = [i for i, orbit in enumerate(orbits) if len(orbit) == 4]
    chosen = list(map(int, rng.choice(four, size=2, replace=False)))
    erased, visible = chosen
    second = int(rng.choice([i for i, o in enumerate(orbits)
                             if i not in chosen and len(o) > 1]))
    # A complete visible four-way orbit gives positive evidence for both folds.
    blank.difference_update(orbits[visible])
    for orbit in orbits:
        if all(p in blank for p in orbit):
            blank.remove(orbit[int(rng.integers(len(orbit)))])
    blank.difference_update(orbits[erased])
    survivor = orbits[erased][int(rng.integers(4))]
    if hard:
        blank.update(p for p in orbits[erased] if p != survivor)
    else:
        blank.add(orbits[erased][int(rng.integers(4))])
    blank.difference_update(orbits[second])
    blank.add(orbits[second][int(rng.integers(len(orbits[second])))])
    return blank, erased, second


def _paint(labels, counts, colors, bg, wall, blank):
    out = np.full(labels.shape, wall, dtype=np.int8)
    inp = out.copy()
    i = 0
    for r, count in enumerate(counts):
        for c in range(count):
            mask = labels == i
            out[mask] = colors[r, c]
            inp[mask] = bg if (r, c) in blank else colors[r, c]
            i += 1
    return inp, out


def _metrics(inp, out):
    info = _read_house(inp)
    bg, wall, crop, box, bands = info
    counts = tuple(map(len, bands))
    rooms = {(r, c): p for r, band in enumerate(bands) for c, p in enumerate(band)}
    values = {k: int(crop[tuple(p[0])]) for k, p in rooms.items()}
    blanks = {p for p in rooms if values[p] == bg}
    diagonal = changed_size = 0
    for orbit in _orbits(counts):
        for r, c in orbit:
            if (r, c) in blanks:
                direct = ((len(counts) - 1 - r, c), (r, counts[r] - 1 - c))
                diagonal += all(p in blanks for p in direct)
                witnesses = [p for p in orbit if p not in blanks]
                changed_size += any(len(rooms[p]) != len(rooms[r, c]) for p in witnesses)
    nonrect = sum(len(p) != int(np.prod(p.max(0) - p.min(0) + 1)) for p in rooms.values())
    # Concrete ablation of A: ordinary two-axis pixel repair, ignoring walls
    # as witnesses, cannot transfer a uniform color into a differently shaped room.
    pixel_wrong = 0
    for key in blanks:
        for r, c in rooms[key]:
            points = ((crop.shape[0] - 1 - r, c), (r, crop.shape[1] - 1 - c),
                      (crop.shape[0] - 1 - r, crop.shape[1] - 1 - c))
            v = {int(crop[p]) for p in points if crop[p] not in (bg, wall)}
            pixel_wrong += len(v) != 1 or next(iter(v)) != int(out[r, c])
    # Concrete ablation of B: nearest colored chamber, with all nearest ties.
    centers = {k: p.mean(0) for k, p in rooms.items()}
    nearest_wrong = 0
    for key in blanks:
        distances = {p: float(np.sum((centers[p] - centers[key]) ** 2))
                     for p in rooms if p not in blanks}
        best = min(distances.values())
        candidates = {values[p] for p, d in distances.items() if abs(d - best) < 1e-9}
        expected = int(out[tuple(rooms[key][0])])
        nearest_wrong += expected not in candidates
    return dict(profile=counts, bands=len(counts), rooms=len(rooms),
                blanks=len(blanks), nonrect=nonrect, diagonal_only=int(diagonal),
                changed_size=int(changed_size), pixel_wrong=int(pixel_wrong),
                nearest_wrong=int(nearest_wrong), bg=bg, wall=wall,
                fills=len(set(int(out[tuple(p[0])]) for p in rooms.values())),
                blank_colors=len({int(out[tuple(rooms[p][0])]) for p in blanks}),
                singleton_orbits=sum(len(o) == 1 for o in _orbits(counts)),
                frame_shape=crop.shape)


def _pair(rng, n_bands, max_count, large, hard):
    for _ in range(96):
        counts = _profile(rng, n_bands, max_count, large)
        orbits = _orbits(counts)
        labels = _floorplan(rng, counts)
        palette = list(map(int, rng.permutation(10)))
        bg, wall = palette[:2]
        k = int(rng.integers(2, min(8, len(orbits)) + 1))
        payload = palette[2:2 + k]
        blank, first, second = _logical_mask(rng, counts, orbits, hard)
        indices = rng.integers(k, size=len(orbits))
        indices[rng.permutation(len(orbits))[:k]] = np.arange(k)
        indices[first], indices[second] = 0, 1
        colors = {}
        for i, orbit in enumerate(orbits):
            for p in orbit:
                colors[p] = payload[int(indices[i])]
        crop, out = _paint(labels, counts, colors, bg, wall, blank)
        margins = []
        for size in crop.shape:
            extra = int(rng.integers(2, min(6, 30 - size) + 1))
            before = int(rng.integers(1, extra))
            margins.append((before, extra - before))
        inp = np.pad(crop, margins, constant_values=bg)
        try:
            predicted = solve(inp)
            if not np.array_equal(predicted, out):
                continue
            m = _metrics(inp, out)
            if m['nonrect'] and m['changed_size'] and m['pixel_wrong'] and m['nearest_wrong']:
                return {'input': inp, 'output': out}
        except ValueError:
            continue
    raise RuntimeError("bounded mirror-house construction exhausted")


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train, 'num_train', 2, 10)
    ns = _integer(num_test, 'num_test', 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # Only the curriculum varies at puzzle level; semantics are fixed.
    depth_shift = bool(rng.integers(2))
    hard_demo = int(rng.integers(nt))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            is_test = split == 'test'
            if depth_shift:
                n_bands = int(rng.integers(5, 8) if is_test else rng.integers(3, 5))
                max_count, large = 7, False
            else:
                n_bands = int(rng.integers(3, 8))
                max_count, large = (7, True) if is_test else (4, False)
            hard = (split == 'train' and index == hard_demo) or (is_test and rng.random() < 0.8)
            for _ in range(32):
                pair = _pair(rng, n_bands, max_count, large, hard)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, metrics = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _read_house(inp)
            bg, wall, crop, box, bands = info
            assert np.array_equal(solve(inp), out), "incorrect chamber reconstruction"
            assert out.shape == crop.shape
            assert np.array_equal(out == wall, crop == wall), "walls changed"
            assert np.array_equal(out[crop != bg], crop[crop != bg]), "visible cells changed"
            assert not np.any(out == bg), "unfilled room"
            assert np.array_equal(out != crop, crop == bg), "only blank rooms should change"
            counts = tuple(map(len, bands))
            assert len(set(counts)) >= 2, "a ragged profile is required"
            orbits = _orbits(counts)
            assert any(len(o) == 4 and all(crop[tuple(bands[r][c][0])] != bg
                                          for r, c in o) for o in orbits), \
                "both folds need a complete visible example"
            for orbit in orbits:
                surviving = [p for p in orbit if crop[tuple(bands[p[0]][p[1]][0])] != bg]
                assert surviving, "erased orbit has no witness"
                expected = {int(out[tuple(bands[r][c][0])]) for r, c in orbit}
                assert len(expected) == 1
                for r, c in orbit:
                    assert np.all(out[tuple(bands[r][c].T)] == next(iter(expected)))
            m = _metrics(inp, out)
            assert m['blank_colors'] >= 2 and m['fills'] >= 2
            assert m['nonrect'] > 0 and m['changed_size'] > 0
            assert m['pixel_wrong'] > 0 and m['nearest_wrong'] > 0
            metrics[split].append(m)
    assert any(m['diagonal_only'] for m in metrics['train']), "both folds must be demonstrated"
    tr, te = metrics['train'], metrics['test']
    depth = min(m['bands'] for m in te) > max(m['bands'] for m in tr)
    branch = min(max(m['profile']) for m in te) > max(max(m['profile']) for m in tr)
    assert depth or branch, "missing ordinal-complexity train/test shift"


def self_test():
    shapes, profiles, walls, backgrounds, fills = set(), set(), set(), set(), set()
    signatures, structures = set(), set()
    totals = dict(nonrect=0, diagonal_only=0, changed_size=0, pixel_wrong=0,
                  nearest_wrong=0, singleton_orbits=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s]
                          for k in ('input', 'output'))
        signatures.add(signature)
        for pair in puzzle['train'] + puzzle['test']:
            m = _metrics(pair['input'], pair['output'])
            shapes.add(m['frame_shape'])
            profiles.add(m['profile'])
            backgrounds.add(m['bg'])
            walls.add(m['wall'])
            fills.add(m['fills'])
            structures.add((pair['output'].shape, (pair['output'] == m['wall']).tobytes()))
            for key in totals:
                totals[key] += m[key]
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 35 and len(profiles) >= 50
    assert len(backgrounds) == len(walls) == 10 and len(fills) >= 4
    assert all(totals[k] > 0 for k in totals)
    assert totals['diagonal_only'] >= 42
    # Whole-puzzle reproducibility, explicit count boundaries, and illegal types.
    a, b = generate_puzzle(8128, 10, 4), generate_puzzle(8128, 10, 4)
    validate_generated_puzzle(a)
    assert all(_key(p[k]) == _key(q[k]) for s in a for p, q in zip(a[s], b[s])
               for k in ('input', 'output'))
    validate_generated_puzzle(generate_puzzle(8128, 2, 1))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Invariants: recoloring is equivariant; solve never mutates its input.
    pair = a['test'][0]
    original = pair['input'].copy()
    permutation = np.array([7, 2, 9, 0, 6, 1, 8, 4, 3, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[original]), permutation[pair['output']])
    solve(original)
    assert np.array_equal(original, pair['input'])
    # The rule cannot invent a missing orbit color or resolve disagreement.
    bg, wall, crop, box, bands = _read_house(original)
    orbits = _orbits(tuple(map(len, bands)))
    complete = next(o for o in orbits if len(o) == 4 and all(
        crop[tuple(bands[r][c][0])] != bg for r, c in o))
    for corruption in ('no_witness', 'conflict'):
        damaged = original.copy()
        selected = complete if corruption == 'no_witness' else complete[:1]
        current = int(crop[tuple(bands[selected[0][0]][selected[0][1]][0])])
        replacement = bg if corruption == 'no_witness' else next(
            c for c in range(10) if c not in (bg, wall, current))
        for r, c in selected:
            points = bands[r][c] + box[:2]
            damaged[tuple(points.T)] = replacement
        try:
            solve(damaged)
        except ValueError:
            pass
        else:
            raise AssertionError('unrecoverable or conflicting orbit accepted')
    # A wrong blank-room value and a changed wall must both be rejected.
    for kind in ('room', 'wall'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
        q = bad['test'][0]
        bg, wall, crop, _, _ = _read_house(q['input'])
        eligible = crop == (bg if kind == 'room' else wall)
        p = tuple(np.argwhere(eligible)[0])
        q['output'][p] = (int(q['output'][p]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupted answer accepted")
    return {'seeds': 42, 'pairs': 210, 'profiles': len(profiles),
            'frame_shapes': len(shapes), 'framework_structures': len(structures),
            'background_colors': len(backgrounds), 'wall_colors': len(walls),
            'payload_cardinalities': sorted(fills), **totals}


if __name__ == '__main__':
    print(self_test())
