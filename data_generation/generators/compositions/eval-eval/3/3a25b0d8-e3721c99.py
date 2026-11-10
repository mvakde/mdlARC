"""Island-keyed dye transfer: use the topology of enclosed negative space.

Standard library and NumPy only. No parent data, seed or construction state is
consulted by solve. Diagnostics are streamed/aggregated, never saved here.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    """Four-connected pixel components, in deterministic discovery order."""
    mask = np.asarray(mask, dtype=bool)
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, part = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in _STEPS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
                    part.append(p)
        parts.append(np.asarray(part, dtype=int))
    return parts


def _bbox(points):
    return tuple(map(int, points.min(0))) + tuple(map(int, points.max(0) + 1))


def _mask(points, shape):
    result = np.zeros(shape, dtype=bool)
    result[tuple(points.T)] = True
    return result


def _holes(mask):
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    return [p - 1 for p in _components(~padded)
            if np.all(p > 0) and np.all(p < np.array(padded.shape) - 1)]


def _hole_count(mask):
    return len(_holes(mask))


def _near(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = np.zeros_like(mask)
    for r in range(3):
        for c in range(3):
            result |= p[r:r + h, c:c + w]
    return result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _parse(input_grid):
    """Identify roles and actual rooms exclusively from input pixels.

    Holes of an OUTER framework component locate raw room enclosures. Detached
    framework islands lie inside those enclosures. Removing them gives the
    connected interior whose holes are the key class.
    """
    a = _grid(input_grid)
    bg = int(a[0, 0])
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    if not np.all(border == bg):
        raise ValueError("a uniform exterior border is required")
    candidates = []
    for wall in map(int, np.unique(a)):
        if wall == bg:
            continue
        parts = _components(a == wall)
        shells, small = [], []
        for points in parts:
            raw_rooms = _holes(_mask(points, a.shape))
            if len(raw_rooms) >= 2:
                shells.append((points, raw_rooms))
            else:
                small.append((points, raw_rooms))
        if len(shells) != 2 or any(holes for _, holes in small):
            continue
        figures, good = [], True
        all_enclosures = np.zeros(a.shape, dtype=bool)
        for outer, raw_rooms in shells:
            rooms, interior_union = [], np.zeros(a.shape, dtype=bool)
            for enclosure in raw_rooms:
                raw = _mask(enclosure, a.shape)
                all_enclosures |= raw
                interior = raw & (a != wall)
                values = np.unique(a[interior])
                if len(values) != 1 or len(_components(interior)) != 1:
                    good = False
                    break
                islands = _components(raw & (a == wall))
                if any(_hole_count(_mask(p, a.shape)) for p in islands):
                    good = False
                    break
                # Every hole of the room really is one solid framework island.
                holes = _holes(interior)
                if len(holes) != len(islands) or any(np.any(a[tuple(p.T)] != wall) for p in holes):
                    good = False
                    break
                interior_union |= interior
                rooms.append(dict(mask=interior, count=len(holes), color=int(values[0]),
                                  area=int(interior.sum()), islands=islands))
            if not good:
                break
            colors = [r["color"] for r in rooms]
            kind = ("target" if all(c == bg for c in colors) else
                    "donor" if all(c != bg for c in colors) and len(set(colors)) >= 2 else None)
            if kind is None:
                good = False
                break
            outer_mask = _mask(outer, a.shape)
            figures.append(dict(kind=kind, rooms=rooms, outer=outer_mask,
                                interior=interior_union, box=_bbox(outer)))
        if not good or {f["kind"] for f in figures} != {"donor", "target"}:
            continue
        if any(not np.all(all_enclosures[tuple(p.T)]) for p, _ in small):
            continue
        donor = next(f for f in figures if f["kind"] == "donor")
        target = next(f for f in figures if f["kind"] == "target")
        mapping = {r["count"]: r["color"] for r in donor["rooms"]}
        if len(mapping) != len(donor["rooms"]):
            continue
        occupied = (a == wall) | donor["interior"]
        if not np.array_equal(occupied, a != bg):
            continue
        candidates.append(dict(background=bg, wall=wall, donor=donor,
                               target=target, mapping=mapping))
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous atlas and empty framework")
    return candidates[0]


def _answer(a, parsed):
    bg, wall = parsed["background"], parsed["wall"]
    target = parsed["target"]
    r, c, R, C = target["box"]
    # Isolate the target. A donor could never leak into this reconstruction.
    full = np.full(a.shape, bg, dtype=np.int8)
    target_region = target["outer"] | target["interior"]
    for room in target["rooms"]:
        for island in room["islands"]:
            target_region[tuple(island.T)] = True
    full[target_region & (a == wall)] = wall
    for room in target["rooms"]:
        full[room["mask"]] = parsed["mapping"].get(room["count"], bg)
    return full[r:R, c:C].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill destination negative-space chambers using their island counts."""
    a = _grid(input_grid)
    return _answer(a, _parse(a))


def _footprint(rng, count):
    # Connected subsets have many local adjacency patterns, not a fixed panel.
    options = [(r, c) for r in range(1, 4) for c in range(1, 3)
               if r * c >= count and (r - 1) * c < count + 2]
    rows, cols = options[int(rng.integers(len(options)))]
    cells = {(int(rng.integers(rows)), int(rng.integers(cols)))}
    while len(cells) < count:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _STEPS
                           if 0 <= r + dr < rows and 0 <= c + dc < cols} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    points = np.asarray(sorted(cells), dtype=int)
    points -= points.min(0)
    return points


def _room(rng, h, w, k, richer):
    room = np.ones((h, w), dtype=bool)
    lattices = []
    for ro in (1, 2):
        for co in (1, 2):
            sites = [(r, c) for r in range(ro, h - 1, 2) for c in range(co, w - 1, 2)]
            if len(sites) >= k:
                lattices.append(sites)
    sites = lattices[int(rng.integers(len(lattices)))]
    chosen = rng.choice(len(sites), size=k, replace=False)
    islands = []
    for i in chosen:
        island = np.zeros_like(room)
        island[sites[int(i)]] = True
        islands.append(island)
    # Grow the islands without changing their identity, separation or holes.
    for i, island in enumerate(islands):
        wanted = int(rng.integers(1, 5 if richer else 3))
        for _ in range(8):
            if int(island.sum()) >= wanted:
                break
            other = np.zeros_like(room)
            for j, another in enumerate(islands):
                if j != i:
                    other |= another
            frontier = []
            for r, c in np.argwhere(island):
                for dr, dc in _STEPS:
                    rr, cc = int(r + dr), int(c + dc)
                    if (0 < rr < h - 1 and 0 < cc < w - 1 and not island[rr, cc]
                            and not _near(other)[rr, cc]):
                        frontier.append((rr, cc))
            if not frontier:
                break
            trial = island.copy()
            trial[frontier[int(rng.integers(len(frontier)))]] = True
            combined = trial | other
            if len(_components(~combined)) == 1 and _hole_count(trial) == 0:
                island[:] = trial
        room[island] = False
    # A room can become stepped or notched; the class stays exactly k.
    for _ in range(int(rng.integers(1, 7))):
        boundary = [(r, c) for r, c in np.argwhere(room)
                    if r in (0, h - 1) or c in (0, w - 1)]
        r, c = boundary[int(rng.integers(len(boundary)))]
        trial = room.copy()
        trial[r, c] = False
        if (trial[0].any() and trial[-1].any() and trial[:, 0].any() and trial[:, -1].any()
                and len(_components(trial)) == 1 and _hole_count(trial) == k):
            room = trial
    assert len(_components(room)) == 1 and _hole_count(room) == k
    return room


def _figure(rng, classes, bg, wall, pigments=None, richer=False):
    points = _footprint(rng, len(classes))
    rows, cols = map(int, points.max(0) + 1)
    widths = rng.integers(3, 6, size=cols)
    for (r, c), k in zip(points, classes):
        if k == 4:
            widths[c] = 5
    heights = np.full(rows, 3, dtype=int)
    for (r, c), k in zip(points, classes):
        capacity_columns = (int(widths[c]) - 1) // 2
        needed = (k + capacity_columns - 1) // capacity_columns
        heights[r] = max(heights[r], 2 * needed + 1)
    for r in range(rows):
        heights[r] = int(rng.integers(heights[r], 8))
    ys = np.concatenate(([0], np.cumsum(heights + 1)))
    xs = np.concatenate(([0], np.cumsum(widths + 1)))
    shape = (int(ys[-1]) + 1, int(xs[-1]) + 1)
    rectangles = np.zeros(shape, dtype=bool)
    rooms = []
    for (r, c), k in zip(points, classes):
        r0, c0 = int(ys[r]) + 1, int(xs[c]) + 1
        h, w = int(heights[r]), int(widths[c])
        rectangles[r0:r0 + h, c0:c0 + w] = True
        local = _room(rng, h, w, int(k), richer)
        mask = np.zeros(shape, dtype=bool)
        mask[r0:r0 + h, c0:c0 + w] = local
        rooms.append(mask)
    framework = _near(rectangles) & ~rectangles
    all_rooms = np.zeros(shape, dtype=bool)
    for mask in rooms:
        all_rooms |= mask
    framework |= rectangles & ~all_rooms
    grid = np.full(shape, bg, dtype=np.int8)
    grid[framework] = wall
    if pigments is not None:
        for k, mask in zip(classes, rooms):
            grid[mask] = pigments[k]
    # Extreme wall pixels can vanish at unused footprint margins: trim once.
    r, c, R, C = _bbox(np.argwhere(framework))
    return grid[r:R, c:C].copy(), [m[r:R, c:C].copy() for m in rooms]


def _embed(rng, donor, target, bg, rotations):
    gap = int(rng.integers(1, min(3, 28 - donor.shape[1] - target.shape[1]) + 1))
    base_w = donor.shape[1] + target.shape[1] + gap
    width = base_w + int(rng.integers(2, min(30 - base_w, 5) + 1))
    base_h = max(donor.shape[0], target.shape[0])
    height = base_h + int(rng.integers(2, min(30 - base_h, 5) + 1))
    left_margin = int(rng.integers(1, width - base_w))
    figures = [donor, target]
    if rng.random() < .5:
        figures.reverse()
    grid = np.full((height, width), bg, dtype=np.int8)
    col = left_margin
    for figure in figures:
        row = int(rng.integers(1, height - figure.shape[0]))
        grid[row:row + figure.shape[0], col:col + figure.shape[1]] = figure
        col += figure.shape[1] + gap
    rotation = int(rng.choice(rotations))
    return np.rot90(grid, rotation).copy(), rotation


def _inversion(keys, targets):
    ranks = {k: i for i, k in enumerate(keys)}
    seq = [ranks[k] for k in targets if k in ranks]
    return any(a > b for i, a in enumerate(seq) for b in seq[i + 1:])


def _pair(rng, split, index, preference):
    if split == "train" and index == 0:
        choices = [(a, b) for a in range(5) for b in range(a + 2, 5)]
        keys = list(choices[int(rng.integers(len(choices)))])
    elif split == "train" and index == 1:
        keys = [0] + list(map(int, rng.choice(np.arange(1, 5), size=2, replace=False)))
    else:
        count = int(rng.choice([2, 3] if split == "train" else [2, 3, preference]))
        keys = list(map(int, rng.choice(5, size=count, replace=False)))
    rng.shuffle(keys)
    palette = list(map(int, rng.permutation(10)))
    bg, wall = palette[:2]
    colors = palette[2:2 + len(keys)]
    if len(keys) > 2 and rng.random() < .35:
        colors[-1] = colors[0]
    mapping = dict(zip(keys, colors))
    absent = int(rng.choice([k for k in range(5) if k not in mapping]))
    # Every pair demonstrates two retained classes and selective withholding.
    favorite = int(rng.choice([k for k in keys if k > 0]))
    if split == "train":
        targets = keys + [absent, favorite]
    else:
        others = [k for k in keys if k != favorite]
        targets = [favorite] * 3 + [others[0], absent]
        if rng.random() < .7:
            targets.append(int(rng.choice(others + [absent])))
    rng.shuffle(targets)
    if not _inversion(keys, targets):
        targets.sort(key=lambda k: keys.index(k) if k in keys else -1, reverse=True)
    richer = split == "test" or index % 2 == 1
    donor, donor_masks = _figure(rng, keys, bg, wall, mapping, richer)
    target, masks = _figure(rng, targets, bg, wall, richer=richer)
    expected = target.copy()
    for k, mask in zip(targets, masks):
        expected[mask] = mapping.get(k, bg)
    # Rotate the constructed answer with the scene. solve receives neither
    # this orientation nor the room masks: it reconstructs both from pixels.
    rotations = list(range(4))
    if split == "train":
        # Demonstrations must defeat room-order copying in their VISIBLE
        # orientation. Room masks touch all four sides of their own original
        # boxes, so 0 and 180 degrees preserve the constructed inversion.
        def order(values, regions, quarter_turn):
            return [values[i] for _, i in sorted(
                (tuple(np.argwhere(np.rot90(mask, quarter_turn))[0]), i)
                for i, mask in enumerate(regions))]
        rotations = [q for q in rotations if _inversion(
            order(keys, donor_masks, q), order(targets, masks, q))]
        assert rotations
    inp, rotation = _embed(rng, donor, target, bg, rotations)
    return {"input": inp, "output": np.rot90(expected, rotation).copy()}


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    preference = int(rng.choice([3, 4]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(16):
                pair = _pair(rng, split, index, preference)
                if _key(pair["input"]) not in seen:
                    break
            else:
                # At most 14 existing examples versus 90 distinct role pairs.
                parsed = _parse(pair["input"])
                old = [parsed["background"], parsed["wall"]]
                old += [c for c in range(10) if c not in old]
                for bg in range(10):
                    for wall in range(10):
                        if bg == wall:
                            continue
                        new = [bg, wall] + [c for c in range(10) if c not in (bg, wall)]
                        lookup = np.empty(10, dtype=np.int8)
                        lookup[old] = new
                        replacement = {k: lookup[v] for k, v in pair.items()}
                        if _key(replacement["input"]) not in seen:
                            pair = replacement
                            break
                    else:
                        continue
                    break
            assert _key(pair["input"]) not in seen
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def measure_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    p = _parse(inp)
    assert np.array_equal(out, _answer(inp, p)), "incorrect chamber transfer"
    bg, wall = p["background"], p["wall"]
    target, donor = p["target"], p["donor"]
    r, c, R, C = target["box"]
    before = inp[r:R, c:C]
    assert np.array_equal(out == wall, before == wall), "framework/island recolored"
    should_change = np.zeros(inp.shape, dtype=bool)
    counts = [room["count"] for room in target["rooms"]]
    retained = set(counts) & p["mapping"].keys()
    for room in target["rooms"]:
        if room["count"] in p["mapping"]:
            should_change |= room["mask"]
    changed = out != before
    assert np.array_equal(changed, should_change[r:R, c:C])
    assert np.all(before[changed] == bg)
    assert len(retained) >= 2 and any(k not in p["mapping"] for k in counts)
    assert 2 <= len(donor["rooms"]) <= 4 and 2 <= len(target["rooms"]) <= 6
    assert all(0 <= room["count"] <= 4 for f in (donor, target) for room in f["rooms"])
    donor_order = [room["count"] for room in donor["rooms"]]
    # Rotation can reverse reading order. Inversion is used as a curriculum
    # witness only, never as a matching rule.
    inversion = _inversion(donor_order, counts)
    islands = [island for f in (donor, target) for room in f["rooms"] for island in room["islands"]]
    room_points = [np.argwhere(room["mask"]) for f in (donor, target) for room in f["rooms"]]
    return dict(shape=inp.shape, output_shape=out.shape, roles=(bg, wall),
                keys=tuple(sorted(p["mapping"])), donor_rooms=len(donor_order),
                target_rooms=len(counts), counts=tuple(counts), inversion=inversion,
                multiplicity=max(counts.count(k) for k in retained),
                unlisted=sum(k not in p["mapping"] for k in counts),
                unused=len(p["mapping"].keys() - set(counts)),
                pigments=len(set(p["mapping"].values())),
                island_areas=tuple(len(i) for i in islands),
                nonrectangular_rooms=sum(len(q) < (_bbox(q)[2] - _bbox(q)[0]) *
                                         (_bbox(q)[3] - _bbox(q)[1]) for q in room_points),
                structure=hashlib.sha256(bytes(inp.shape) + (inp == wall).tobytes()).hexdigest())


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for value in pair.values():
                _grid(value)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(measure_pair(pair))
    assert all(m["multiplicity"] == 2 and 4 <= m["target_rooms"] <= 5 for m in infos["train"])
    assert all(m["inversion"] for m in infos["train"]), "missing room-order counterexample"
    assert all(m["multiplicity"] == 3 and m["target_rooms"] >= 5 for m in infos["test"])
    assert len(infos["train"][0]["keys"]) == 2
    assert max(infos["train"][0]["keys"]) - min(infos["train"][0]["keys"]) >= 2
    assert 0 in infos["train"][1]["keys"] and len(infos["train"][1]["keys"]) == 3


def _hand_fixture():
    """Equal outer rectangles, reversed order, one unlisted class; no sampling."""
    donor = np.full((7, 13), 5, dtype=np.int8)
    donor[1:6, 1:6] = 2
    donor[1:6, 7:12] = 3
    donor[3, 9] = 5
    target = np.full((7, 19), 5, dtype=np.int8)
    for c in (1, 7, 13):
        target[1:6, c:c + 5] = 0
    target[3, 3] = 5
    target[2, 14] = target[4, 16] = 5
    expected = target.copy()
    expected[1:6, 1:6] = 3
    expected[3, 3] = 5
    expected[1:6, 7:12] = 2
    inp = np.zeros((17, 21), dtype=np.int8)
    inp[1:8, 1:14] = donor
    inp[9:16, 1:20] = target
    return inp, expected


def self_test():
    """Bounded 42-seed semantic and variation audit; returns a small report."""
    diagonal = np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert _hole_count(diagonal) == 1
    diagonal[0, 1] = False
    assert _hole_count(diagonal) == 0
    inp, expected = _hand_fixture()
    assert np.array_equal(solve(inp), expected)
    # A's payload is essential: exchanging pigments changes only matched fills.
    exchanged = inp.copy()
    exchanged[inp == 2], exchanged[inp == 3] = 3, 2
    swapped_expected = expected.copy()
    swapped_expected[expected == 2], swapped_expected[expected == 3] = 3, 2
    assert np.array_equal(solve(exchanged), swapped_expected)
    assert not np.array_equal(expected, swapped_expected)
    # B's topology is essential: a second separated island makes class 1 absent.
    two = inp.copy()
    two[11, 5] = 5
    unlisted = expected.copy()
    unlisted[1:6, 1:6] = 0
    unlisted[3, 3] = unlisted[2, 4] = 5
    assert np.array_equal(solve(two), unlisted)
    # Growing a single island keeps the count and hence the selected pigment.
    grown = inp.copy()
    grown[11, 4] = 5
    grown_expected = expected.copy()
    grown_expected[2, 3] = 5
    assert np.array_equal(solve(grown), grown_expected)
    ambiguous = inp.copy()
    ambiguous[4, 4] = 5  # donor's first chamber now also has one island
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate atlas class was accepted")
    features = []
    shapes, structures, roles, keysets, island_areas = set(), set(), set(), set(), set()
    room_counts, local_variation = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = measure_pair(pair)
                local.append(m)
                shapes.add(m["shape"])
                structures.add(m["structure"])
                roles.add(m["roles"])
                keysets.add(m["keys"])
                island_areas.update(m["island_areas"])
                room_counts.add((m["donor_rooms"], m["target_rooms"]))
        local_variation += int(len({m["structure"] for m in local}) == 5)
        features.extend(local)
    assert len(shapes) >= 30 and len(structures) >= 200
    assert len(roles) >= 40 and len(keysets) >= 15 and len(room_counts) >= 7
    assert len(island_areas) >= 3 and max(island_areas) >= 3
    assert local_variation == 42
    assert sum(m["nonrectangular_rooms"] > 0 for m in features) >= 150
    assert sum(m["inversion"] for m in features) >= 100
    assert any(m["unused"] for m in features)
    a, b = generate_puzzle(137, 10, 4), generate_puzzle(137, 10, 4)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s])
               for k in ("input", "output"))
    validate_generated_puzzle(a)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    bad = generate_puzzle(3, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong framework color accepted")
    return dict(seeds=42, pairs=len(features), input_shapes=len(shapes),
                framework_structures=len(structures), role_pairs=len(roles),
                key_subsets=len(keysets), room_count_pairs=sorted(room_counts),
                island_areas=sorted(island_areas),
                nonrectangular_pairs=sum(m["nonrectangular_rooms"] > 0 for m in features),
                order_inversions=sum(m["inversion"] for m in features),
                within_puzzle_structure_variation=local_variation)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
