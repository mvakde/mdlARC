"""Registered radial petals: input-only docking, inward sampling, and echoes.

No parent code, stored puzzles, files, or generation state are used by solve.
The hub's visible registration bars establish all coordinate conventions.
"""
from __future__ import annotations

import json
from itertools import product

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINALS = tuple(d for d in DIRECTIONS if 0 in d)
ALTERNATIVES = (
    "snap", "coincident", "reverse_seam", "nominal_sample", "center_sample",
    "far_sample", "carrier_sample", "identity", "radial_reflection",
    "no_echo", "no_collector", "paint_guides", "erase_guides",
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _components(cells):
    remaining, parts = set(cells), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, part = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _box(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _square(center):
    return {(center[0] + y, center[1] + x) for y in (-1, 0, 1) for x in (-1, 0, 1)}


def _straight(bar, axis):
    return (len({p[axis] for p in bar}) == 1 and
            max(p[1 - axis] for p in bar) - min(p[1 - axis] for p in bar) + 1 == len(bar))


def _direction(bar, center):
    # The hub bars have cardinal transverse midpoint within half a cell of zero;
    # diagonal docks have both midpoint coordinates at least two cells from zero.
    y, x = (sum(p[j] - center[j] for p in bar) / len(bar) for j in (0, 1))
    if abs(x) < 1:
        return (1 if y > 0 else -1, 0)
    if abs(y) < 1:
        return (0, 1 if x > 0 else -1)
    return (1 if y > 0 else -1, 1 if x > 0 else -1)


def _nearest(center):
    # Coordinates within the tile of the unique cell nearest the hub at (0,0).
    return tuple(int(np.clip(-p, -1, 1)) + 1 for p in center)


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(all(bool(np.all(edge == bg)) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "the input must have an empty border")
    occupied = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    parts = _components(occupied)
    _require(3 <= len(parts) <= 7, "expected one hub and two to six loose petals")
    centers = []
    h, w = a.shape
    for r, c in sorted(occupied):
        if r + 4 >= h or c + 4 >= w:
            continue
        patch = a[r:r + 5, c:c + 5]
        if (np.all(patch[1:4, 1:4] == bg) and np.all(patch[0] != bg) and
                np.all(patch[-1] != bg) and np.all(patch[:, 0] != bg) and
                np.all(patch[:, -1] != bg)):
            centers.append((r + 2, c + 2))
    _require(len(centers) == 1, "expected a unique framed 3x3 collector hole")
    center = centers[0]
    hub = next(part for part in parts if (center[0] - 2, center[1] - 2) in part)
    colors, counts = np.unique(a[a != bg], return_counts=True)
    carrier = int(colors[np.argmax(counts)])
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous common carrier")
    records = []
    for part in parts:
        if part == hub:
            continue
        t, b, l, right = _box(part)
        choices = []
        for r in range(t, b - 1):
            for c in range(l, right - 1):
                block = {(r + y, c + x) for y in range(3) for x in range(3)}
                if not block <= part:
                    continue
                tile = a[r:r + 3, c:c + 3].copy()
                if np.count_nonzero(tile == carrier) < 5:
                    continue
                bar = part - block
                if not 1 <= len(bar) <= 3:
                    continue
                key_colors = {int(a[p]) for p in bar}
                if len(key_colors) != 1 or carrier in key_colors:
                    continue
                key_color = next(iter(key_colors))
                anchor = {p for p in hub if int(a[p]) == key_color}
                if len(anchor) != len(bar):
                    continue
                tile_center = (r + 1, c + 1)
                for axis in (0, 1):
                    if not _straight(bar, axis) or not _straight(anchor, axis):
                        continue
                    guide_normal = next(iter(bar))[axis]
                    delta = tile_center[axis] - guide_normal
                    if abs(delta) != 2:
                        continue
                    if not all(abs(p[1 - axis] - tile_center[1 - axis]) <= 1 for p in bar):
                        continue
                    sign = delta // 2
                    direction = _direction(anchor, center)
                    if direction[axis] != sign:
                        continue
                    translation = [0, 0]
                    translation[axis] = next(iter(anchor))[axis] - sign - guide_normal
                    translation[1 - axis] = (min(p[1 - axis] for p in anchor) -
                                             min(p[1 - axis] for p in bar))
                    registered = tuple(tile_center[j] + translation[j] - center[j] for j in (0, 1))
                    depth = registered[axis] * sign
                    if not 3 <= depth <= 5:
                        continue
                    anchor_midpoint = sum(p[1 - axis] for p in anchor) / len(anchor) - center[1 - axis]
                    if abs(anchor_midpoint - depth * direction[1 - axis]) > 0.5:
                        continue
                    if abs(registered[1 - axis] - depth * direction[1 - axis]) > 1:
                        continue
                    choices.append(dict(tile=tile, key=key_color, axis=axis, sign=sign,
                                        direction=direction, depth=depth, registered=registered,
                                        source_center=tile_center, translation=tuple(translation),
                                        bar=bar, anchor=anchor))
        _require(len(choices) == 1, "petal/bar association is ambiguous")
        records.append(choices[0])
    keys = {p["key"] for p in records}
    _require(len(keys) == len(records), "each dock needs its own matching key color")
    _require(len({p["direction"] for p in records}) == len(records), "duplicate compass dock")
    _require(all(int(a[p]) in keys | {carrier} for p in hub), "unrecognized hub decoration")
    _require(all(not (set(map(int, np.unique(p["tile"]))) & (keys | {bg})) for p in records),
             "guide and background colors cannot be payload colors")
    footprint = set()
    for p in records:
        for v in (p["registered"], tuple(2 * x for x in p["registered"])):
            cells = _square(v)
            _require(not cells & footprint, "registered petals and echoes must not overlap")
            footprint |= cells
        _require(int(p["tile"][_nearest(p["registered"])]) != carrier,
                 "the inward sample must be a visible detail")
    hub_carrier = {(r - center[0], c - center[1]) for r, c in hub if int(a[r, c]) == carrier}
    _require(all(max(abs(r - center[0]), abs(c - center[1])) <= 6 for r, c in hub),
             "hub strokes must stay within six cells of their center")
    return dict(background=bg, carrier=carrier, center=center, hub=hub_carrier, petals=records)


def _crop_pixels(pixels, bg):
    t, b, l, r = _box(pixels)
    result = np.full((b - t + 1, r - l + 1), bg, dtype=np.int8)
    for (y, x), color in pixels.items():
        result[y - t, x - l] = color
    return result


def _render(info, alternative=None):
    """Coordinate dictionary construction; alternatives are bounded test probes."""
    bg, carrier = info["background"], info["carrier"]
    pixels = {p: carrier for p in info["hub"]}
    if alternative != "no_collector":
        pixels[(0, 0)] = carrier
    erased = set()
    for p in info["petals"]:
        axis, sign, direction = p["axis"], p["sign"], p["direction"]
        v = list(p["registered"])
        if alternative == "snap":
            v = [p["depth"] * x for x in direction]
        elif alternative == "coincident":
            v[axis] += sign
        elif alternative == "reverse_seam":
            v[axis] += 2 * sign
        tile = p["tile"]
        sample_at = _nearest(v)
        if alternative == "nominal_sample":
            sample_at = (1 - direction[0], 1 - direction[1])
        elif alternative == "center_sample":
            sample_at = (1, 1)
        elif alternative == "far_sample":
            sample_at = (2 - sample_at[0], 2 - sample_at[1])
        sample = carrier if alternative == "carrier_sample" else int(tile[sample_at])
        if alternative != "no_collector":
            pixels[direction] = sample
        for y, x in product(range(3), repeat=2):
            pixels[(v[0] + y - 1, v[1] + x - 1)] = int(tile[y, x])
        if alternative != "no_echo":
            echo = tile[::-1, ::-1]
            if alternative == "identity":
                echo = tile
            elif alternative == "radial_reflection":
                echo = tile[::-1, :] if direction[1] == 0 else tile[:, ::-1] if direction[0] == 0 else echo
            for y, x in product(range(3), repeat=2):
                pixels[(2 * v[0] + y - 1, 2 * v[1] + x - 1)] = int(echo[y, x])
        anchor = {(r - info["center"][0], c - info["center"][1]) for r, c in p["anchor"]}
        moved_bar = _shift(anchor, tuple(-sign if j == axis else 0 for j in (0, 1)))
        erased |= anchor | moved_bar
        if alternative == "paint_guides":
            for q in anchor | moved_bar:
                pixels[q] = p["key"]
    if alternative == "erase_guides":
        for q in erased:
            pixels.pop(q, None)
    return _crop_pixels(pixels, bg)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, translations, sample cells, and echo coordinates from pixels."""
    output = _render(_parse(input_grid))
    _require(all(n <= 30 for n in output.shape), "completion exceeds ARC dimensions")
    return output


def _d4(tile):
    return [np.rot90(tile, k) for k in range(4)] + [np.rot90(tile[:, ::-1], k) for k in range(4)]


def _tile(rng, carrier, details, registered, direction, witness):
    a = np.full((3, 3), carrier, dtype=np.int8)
    inward = _nearest(registered)
    nominal = (1 - direction[0], 1 - direction[1])
    sample = int(rng.choice(details))
    a[inward] = sample
    if witness:
        aux = next(c for c in details if c != sample)
        for index in rng.permutation(9):
            point = divmod(int(index), 3)
            if point == inward or (point == nominal and nominal != inward):
                continue
            candidate = a.copy()
            candidate[point] = aux
            if len({v.tobytes() for v in _d4(candidate)}) == 8:
                return candidate
        raise AssertionError("no asymmetric sample witness")
    count = int(rng.integers(0, 4))
    positions = [q for q in product(range(3), repeat=2) if q != inward]
    mode = int(rng.integers(3))
    rng.shuffle(positions)
    if mode == 1:
        positions.sort(key=lambda q: abs(q[0] - inward[0]) + abs(q[1] - inward[1]))
    elif mode == 2:
        positions.sort(key=lambda q: q[0] != inward[0])
    for q in positions[:count]:
        a[q] = int(rng.choice(details)) if mode == 0 else sample
    return a


def _dock_options(direction, depths, witness=False):
    choices = []
    axes = (0, 1) if all(direction) else (0,) if direction[0] else (1,)
    for axis in axes:
        sign, tangent_direction = direction[axis], direction[1 - axis]
        for depth, length in product(depths, (1, 2, 3)):
            anchor_start = tangent_direction * depth - length // 2
            for tile_start in range(-1, 3 - length):
                phase = -(length // 2) - tile_start
                if witness and not (length == 1 and phase != 0):
                    continue
                registered = [0, 0]
                registered[axis] = depth * sign
                registered[1 - axis] = tangent_direction * depth + phase
                anchor, bar = set(), set()
                for j in range(length):
                    p, q = [0, 0], [0, 0]
                    p[axis], p[1 - axis] = (depth - 1) * sign, anchor_start + j
                    q[axis], q[1 - axis] = -2 * sign, tile_start + j
                    anchor.add(tuple(p)); bar.add(tuple(q))
                choices.append(dict(direction=direction, axis=axis, sign=sign, depth=depth,
                                    phase=phase, length=length, registered=tuple(registered),
                                    anchor=anchor, bar=bar))
    return choices


def _geometry(rng, count, depths):
    # An off-axis cardinal petal makes bar alignment and closest-cell sampling
    # observable in every example. Remaining directions, lengths and depths vary.
    witness_direction = CARDINALS[int(rng.integers(4))]
    others = [d for d in DIRECTIONS if d != witness_direction]
    rng.shuffle(others)
    directions = [witness_direction] + others[:count - 1]
    options = [_dock_options(d, depths, i == 0) for i, d in enumerate(directions)]
    for values in options:
        rng.shuffle(values)

    def search(index, chosen, footprint, anchors):
        if index == count:
            return chosen
        for p in options[index]:
            occupied = _square(p["registered"]) | _square(tuple(2 * x for x in p["registered"]))
            if occupied & footprint or p["anchor"] & anchors:
                continue
            found = search(index + 1, chosen + [p], footprint | occupied, anchors | p["anchor"])
            if found is not None:
                return found
        return None

    result = search(0, [], set(), set())
    _require(result is not None, "compatible dock geometry exhausted")
    return result


def _hub_geometry(docks):
    cells = {(r, c) for r in range(-2, 3) for c in range(-2, 3) if max(abs(r), abs(c)) == 2}
    for p in docks:
        dr, dc = p["direction"]
        depth = p["depth"]
        cells.update((j * dr, j * dc) for j in range(2, depth - 1))
        if dr and dc:
            cells.add(((depth - 2) * dr, (depth - 1) * dc) if p["axis"] == 0 else
                      ((depth - 1) * dr, (depth - 2) * dc))
        cells |= p["anchor"]
    return cells


def _normalize_object(pixels):
    t, b, l, r = _box(pixels)
    return {(y - t, x - l): v for (y, x), v in pixels.items()}, (b - t + 1, r - l + 1)


def _stage(rng, objects, bg):
    normalized = [_normalize_object(obj) for obj in objects]
    # The hub is packed first; random placement uses masks rather than imposing
    # a row of storage compartments or a fixed source ordering.
    hub_h, hub_w = normalized[0][1]
    low_h, low_w = max(13, hub_h + 4), max(13, hub_w + 4)
    if len(objects) >= 5:
        low_h, low_w = max(low_h, 19), max(low_w, 19)
    for attempt in range(70):
        if attempt < 50:
            h = int(rng.integers(low_h, 31))
            w = int(rng.integers(low_w, 31))
        else:
            h = w = 30
        occupied, forbidden, pasted = set(), set(), {}
        order = [0] + list(map(int, rng.permutation(np.arange(1, len(objects)))))
        complete = True
        for index in order:
            pixels, (oh, ow) = normalized[index]
            placed = False
            positions = [(r, c) for r in range(1, h - oh) for c in range(1, w - ow)]
            rng.shuffle(positions)
            for r, c in positions:
                shifted = {(r + y, c + x): value for (y, x), value in pixels.items()}
                if not set(shifted) & forbidden:
                    pasted.update(shifted)
                    occupied |= set(shifted)
                    forbidden |= {(y + dy, x + dx) for y, x in shifted
                                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                    placed = True
                    break
            if not placed:
                complete = False
                break
        if complete:
            a = np.full((h, w), bg, dtype=np.int8)
            for point, color in pasted.items():
                a[point] = color
            return a
    raise AssertionError("bounded mask packing exhausted")


def _make_pair(rng, palette, count, depths, used):
    bg, carrier = palette[:2]
    keys, details = palette[2:2 + count], palette[2 + count:]
    docks = _geometry(rng, count, depths)
    hub = {p: carrier for p in _hub_geometry(docks)}
    objects, records = [hub], []
    for i, (p, key_color) in enumerate(zip(docks, keys)):
        for q in p["anchor"]:
            hub[q] = key_color
        tile = _tile(rng, carrier, details, p["registered"], p["direction"], i == 0)
        records.append(dict(p, tile=tile, key=key_color))
        pixels = {(y - 1, x - 1): int(tile[y, x]) for y, x in product(range(3), repeat=2)}
        pixels.update({q: key_color for q in p["bar"]})
        objects.append(pixels)
    # Labels come from the constructed coordinates, before source placement.
    # solve separately discovers the same coordinates from staged input pixels.
    output = _render(dict(background=bg, carrier=carrier, center=(0, 0),
                          hub={q for q, color in hub.items() if color == carrier}, petals=records))
    for _ in range(20):
        staged = _stage(rng, objects, bg)
        rotation = int(rng.integers(4))
        a = np.rot90(staged, rotation).copy()
        if _key(a) in used:
            continue
        return dict(input=a, output=np.rot90(output, rotation).copy())
    raise AssertionError("bounded distinct staging exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    n = _count(num_train, "num_train", 2, 10)
    m = _count(num_test, "num_test", 1, 4)
    if n <= m:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    stable_palette = bool(rng.integers(2))
    base_palette = list(map(int, rng.permutation(10)))
    regime = str(rng.choice(("depth_interpolation", "more_petals", "fewer_petals")))
    puzzle, used = {"train": [], "test": []}, set()
    for split, number in (("train", n), ("test", m)):
        for i in range(number):
            palette = base_palette if stable_palette else list(map(int, rng.permutation(10)))
            if regime == "depth_interpolation":
                count = int(rng.integers(2, 7))
                depths = [3 if i % 2 == 0 else 5] if split == "train" else [4]
            else:
                larger = (split == "test") == (regime == "more_petals")
                count = int(rng.integers(4, 7) if larger else rng.integers(2, 4))
                depths = [3, 4, 5]
            pair = _make_pair(rng, palette, count, depths, used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    a, b = _grid(pair["input"]), _grid(pair["output"])
    info = _parse(a)
    # Independent scalar addressing of every source payload, nearest sample,
    # doubled vector, original carrier, and unused output position.
    coordinates = set(info["hub"]) | {(0, 0)} | {p["direction"] for p in info["petals"]}
    for p in info["petals"]:
        coordinates |= _square(p["registered"]) | _square(tuple(2 * x for x in p["registered"]))
    t, bottom, left, right = _box(coordinates)
    _require(b.shape == (bottom - t + 1, right - left + 1), "incorrect tight crop")
    for y in range(t, bottom + 1):
        for x in range(left, right + 1):
            expected = info["carrier"] if (y, x) in info["hub"] or (y, x) == (0, 0) else info["background"]
            for p in info["petals"]:
                if (y, x) == p["direction"]:
                    rr = min(max(0, 1 - p["registered"][0]), 2)
                    cc = min(max(0, 1 - p["registered"][1]), 2)
                    expected = int(p["tile"][rr, cc])
                dr, dc = y - p["registered"][0], x - p["registered"][1]
                if abs(dr) <= 1 and abs(dc) <= 1:
                    expected = int(p["tile"][dr + 1, dc + 1])
                dr, dc = y - 2 * p["registered"][0], x - 2 * p["registered"][1]
                if abs(dr) <= 1 and abs(dc) <= 1:
                    expected = int(p["tile"][1 - dr, 1 - dc])
            _require(int(b[y - t, x - left]) == expected, "wrong docking, sample, echo, or retained scaffold cell")
    _require(not set(map(int, np.unique(b))) & {p["key"] for p in info["petals"]}, "guides survived")
    _require(not np.array_equal(a, b), "identity pair")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid counts")
    seen, analyses = set(), {"train": [], "test": []}
    surviving = set(ALTERNATIVES)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(info)
            if split == "train":
                surviving = {name for name in surviving if np.array_equal(_render(info, name), pair["output"])}
    _require(not surviving, "training leaves an implemented shortcut unresolved: " + repr(sorted(surviving)))
    # Both parent contributions must have a visible witness in demonstrations.
    _require(any(any(p["registered"] != tuple(p["depth"] * v for v in p["direction"])
                     for p in info["petals"]) for info in analyses["train"]), "registration phase not demonstrated")
    _require(any(any(0 in p["direction"] and len({v.tobytes() for v in _d4(p["tile"])}) == 8
                     for p in info["petals"]) for info in analyses["train"]), "half-turn lacks an asymmetric cardinal witness")


def self_test():
    # A legal input outside the sampler's teaching profile: all phases are zero,
    # both payloads are half-turn symmetric, and the hub has an extra branch.
    # These properties must not become hidden preconditions of solve.
    docks = [next(p for p in _dock_options((-1, 0), [4]) if p["phase"] == 0 and p["length"] == 2),
             next(p for p in _dock_options((1, 0), [3]) if p["phase"] == 0 and p["length"] == 3)]
    hub = {p: 1 for p in _hub_geometry(docks) | {(-2, -3), (-3, -3)}}
    objects, payloads = [hub], []
    for p, key_color, detail in zip(docks, (2, 3), (4, 5)):
        tile = np.array([[1, detail, 1], [1, 1, 1], [1, detail, 1]], dtype=np.int8)
        payloads.append(tile)
        for q in p["anchor"]:
            hub[q] = key_color
        pixels = {(y - 1, x - 1): int(tile[y, x]) for y, x in product(range(3), repeat=2)}
        pixels.update({q: key_color for q in p["bar"]})
        objects.append(pixels)
    fixture = _stage(np.random.default_rng(2027), objects, 0)
    expected = np.zeros((17, 6), dtype=np.int8)  # O has output index (9,3).
    for (r, c), color in hub.items():
        if color == 1:
            expected[r + 9, c + 3] = 1
    expected[8:11, 2:5] = [[0, 4, 0], [0, 1, 0], [0, 5, 0]]
    for r, tile in zip((-4, 3), payloads):
        expected[r + 8:r + 11, 2:5] = tile
        expected[2 * r + 8:2 * r + 11, 2:5] = tile[::-1, ::-1]
    _require(np.array_equal(solve(fixture), expected), "zero-phase/symmetric/branched input regression")
    fingerprints, shapes, out_shapes, counts, depths, lengths, directions = set(), set(), set(), set(), set(), set(), set()
    palettes, phases, structures, repeated = set(), set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        compact = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
        fingerprints.add(json.dumps(compact, separators=(",", ":")))
        for split in puzzle:
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "solve disagrees")
                shapes.add(pair["input"].shape); out_shapes.add(pair["output"].shape)
                counts.add(len(info["petals"])); palettes.add((info["background"], info["carrier"]))
                roles, normalized = {}, []
                for value in pair["input"].flat:
                    normalized.append(roles.setdefault(int(value), len(roles)))
                structures.add((pair["input"].shape, tuple(normalized)))
                for p in info["petals"]:
                    directions.add(p["direction"]); depths.add(p["depth"]); lengths.add(len(p["bar"]))
                    phases.add(p["registered"][1 - p["axis"]] - p["depth"] * p["direction"][1 - p["axis"]])
                    sample = p["tile"][_nearest(p["registered"])]
                    repeated += int(np.count_nonzero(p["tile"] == sample) > 1)
    _require(len(fingerprints) == 42 and len(structures) >= 200, "seed/structural diversity collapsed")
    _require(len(shapes) >= 60 and len(out_shapes) >= 25 and len(palettes) >= 40, "geometry/palette variation collapsed")
    _require(counts == set(range(2, 7)) and depths == {3, 4, 5} and lengths == {1, 2, 3}, "count/depth/bar range missing")
    _require(directions == set(DIRECTIONS) and phases >= {-1, 0, 1} and repeated >= 25, "radial/motif variation missing")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            puzzle = generate_puzzle(1701 + 13 * n + m, n, m)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (n, m), "explicit count ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(314159, 2, 1), generate_puzzle(314159, 2, 1)
    _require(all(_key(first[s][i][k]) == _key(second[s][i][k]) for s in first
                 for i in range(len(first[s])) for k in ("input", "output")), "nondeterministic puzzle")
    for name in ALTERNATIVES:
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        pair = bad["train"][0]
        info = _parse(pair["input"])
        wrong = _render(info, name)
        if np.array_equal(wrong, pair["output"]):
            continue
        pair["output"] = wrong
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted wrong " + name)


if __name__ == "__main__":
    self_test()
    print("self_test passed")
