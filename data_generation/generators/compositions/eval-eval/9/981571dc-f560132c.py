"""Mirror stencil jigsaw: joint exact cover and reflection-state inference.

Only NumPy and the standard library are used. No parent code is imported.
The input carries all placement, legend, and stencil evidence.
"""
from functools import lru_cache
from itertools import permutations
from math import isqrt

import numpy as np


BLANK, INK, GAP = 5, 8, 9
PIGMENTS = (0, 1, 2, 3, 4, 6, 7)
UNKNOWN, OUTSIDE = -1, -2
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in NEIGHBORS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(np.array(sorted(cells), dtype=int))
    return result


@lru_cache(maxsize=30)
def _orbits(side):
    groups, labels = [], np.empty(side * side, dtype=int)
    for r in range((side + 1) // 2):
        for c in range((side + 1) // 2):
            cells = tuple(sorted({(r, c), (side - 1 - r, c),
                                  (r, side - 1 - c),
                                  (side - 1 - r, side - 1 - c)}))
            index = len(groups)
            for y, x in cells:
                labels[y * side + x] = index
            groups.append(cells)
    labels.setflags(write=False)
    return tuple(groups), labels


def _complete(raw):
    """Reject disagreement or a witness-free orbit; otherwise copy its state."""
    groups, labels = _orbits(len(raw))
    flat = raw.ravel()
    zeros = np.bincount(labels[flat == 0], minlength=len(groups))
    ones = np.bincount(labels[flat == 1], minlength=len(groups))
    if np.any((zeros > 0) & (ones > 0)) or np.any(zeros + ones == 0):
        return None
    return (ones > 0).astype(np.int8)[labels].reshape(raw.shape)


def _parse(grid):
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0],
                             grid[1:-1, -1]))
    bg = int(np.bincount(border, minlength=10).argmax())
    _require(bg in PIGMENTS and np.count_nonzero(border == bg) > len(border) / 2,
             "background must occupy a strict majority of the border")
    components = _components(grid != bg)
    _require(len(components) == 4, "expected four connected sheets")
    # Empty Chebyshev halos make segmentation independent of marker colors.
    for i in range(4):
        for j in range(i):
            distances = np.max(np.abs(components[i][:, None] - components[j][None]), axis=2)
            _require(int(distances.min()) >= 2, "sheets must have a background-cell gap")
    total = sum(len(c) for c in components)
    side = isqrt(total)
    _require(side * side == total and 1 <= side <= 30,
             "sheet area must be a square")
    tiles, anchors = [], []
    for i, coords in enumerate(components):
        values = grid[coords[:, 0], coords[:, 1]]
        stamp = coords[~np.isin(values, (BLANK, INK, GAP))]
        if len(stamp):
            _require(len(stamp) == 4, "only the four stamp cells may use pigments")
            r, c = stamp.min(axis=0)
            _require(set(map(tuple, stamp)) == {(r, c), (r + 1, c),
                                               (r, c + 1), (r + 1, c + 1)},
                     "stamp must be a 2 by 2 patch")
            key = grid[r:r + 2, c:c + 2].copy()
            _require(len(set(key.flat)) == 4 and all(int(x) in PIGMENTS for x in key.flat),
                     "stamp pigments must be distinct")
            anchors.append((i, key, (int(r - coords[:, 0].min()),
                                      int(c - coords[:, 1].min()))))
        minimum = coords.min(axis=0)
        normalized = coords - minimum
        h, w = normalized.max(axis=0) + 1
        tile = np.full((h, w), OUTSIDE, dtype=np.int8)
        states = np.full(len(values), UNKNOWN, dtype=np.int8)
        states[values == BLANK], states[values == INK] = 0, 1
        tile[normalized[:, 0], normalized[:, 1]] = states
        tiles.append(tile)
    _require(len(anchors) == 1, "exactly one sheet must carry the stamp")
    anchor, key, stamp_location = anchors[0]
    order = [anchor] + [i for i in range(4) if i != anchor]
    return {"side": side, "background": bg, "key": key,
            "stamp_location": stamp_location,
            "tiles": [tiles[i] for i in order],
            "components": [components[i] for i in order]}


def _placements(tile, side, corner, anchor=False):
    """Corner contact fixes translation, but decorated rotations stay distinct."""
    result, seen = [], set()
    corners = ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1))
    for turns in ([0] if anchor else range(4)):
        rotated = np.rot90(tile, turns)
        h, w = rotated.shape
        if h > side or w > side:
            continue
        r0 = 0 if corner < 2 else side - h
        c0 = 0 if corner % 2 == 0 else side - w
        coords = np.argwhere(rotated != OUTSIDE)
        yy, xx = coords[:, 0] + r0, coords[:, 1] + c0
        points = set(zip(yy.tolist(), xx.tolist()))
        if [p for p in corners if p in points] != [corners[corner]]:
            continue
        indices = yy * side + xx
        values = rotated[coords[:, 0], coords[:, 1]].copy()
        mask = sum(1 << int(i) for i in indices)
        sig = (mask, values.tobytes())
        if sig not in seen:
            seen.add(sig)
            result.append({"indices": indices, "values": values,
                           "mask": mask, "turns": turns})
    return result


def _assemblies(parsed):
    n, tiles = parsed["side"], parsed["tiles"]
    table = {(i, c): _placements(tiles[i], n, c, i == 0)
             for i in range(4) for c in ([0] if i == 0 else (1, 2, 3))}
    full = (1 << (n * n)) - 1
    for anchor in table[0, 0]:
        for order in permutations((1, 2, 3)):
            def visit(corner, occupied, placed):
                if corner == 4:
                    if occupied == full:
                        yield (0,) + order, placed
                    return
                piece = order[corner - 1]
                for candidate in table[piece, corner]:
                    if candidate["mask"] & occupied == 0:
                        yield from visit(corner + 1, occupied | candidate["mask"],
                                         placed + [candidate])
            yield from visit(1, anchor["mask"], [anchor])


def _analyze(grid):
    parsed = _parse(grid)
    n = parsed["side"]
    outputs, geometries, rejected, accepted = {}, set(), set(), []
    for order, placed in _assemblies(parsed):
        raw = np.full(n * n, OUTSIDE, dtype=np.int8)
        owner = np.empty(n * n, dtype=np.int8)
        for corner, p in enumerate(placed):
            raw[p["indices"]] = p["values"]
            owner[p["indices"]] = corner
        raw, owner = raw.reshape(n, n), owner.reshape(n, n)
        sig = raw.tobytes(), owner.tobytes()
        geometries.add(sig)
        stencil = _complete(raw)
        if stencil is None:
            rejected.add(sig)
            continue
        output = np.where(stencil == 1, parsed["key"].ravel()[owner],
                          parsed["background"]).astype(np.int8)
        record = {"raw": raw, "owner": owner, "stencil": stencil,
                  "output": output, "piece_for_corner": order,
                  "turns": tuple(p["turns"] for p in placed)}
        accepted.append(record)
        outputs.setdefault(output.tobytes(), record)
    _require(len(outputs) == 1, "input must determine exactly one printed square")
    answer = next(iter(outputs.values()))
    # Prefer the least actual rotation among physically interchangeable sheets.
    answer = min(accepted, key=lambda a: (sum(t != 0 for t in a["turns"]),
                                         sum(a["turns"]), a["turns"]))
    parsed.update(answer)
    stamp = np.zeros((n, n), dtype=bool)
    r, c = parsed["stamp_location"]
    stamp[r:r + 2, c:c + 2] = True
    parsed["stamp_mask"] = stamp
    parsed["geometry_candidates"] = len(geometries)
    parsed["mirror_rejected"] = len(rejected)
    parsed["accepted_arrangements"] = len(accepted)
    parsed["translation_only"] = any(all(t == 0 for t in a["turns"]) for a in accepted)
    return parsed


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from pixels alone, including joint packing/repair ambiguity checks."""
    return _analyze(_grid(input_grid))["output"].copy()


def _key_locations(owner):
    n = len(owner)
    groups, _ = _orbits(n)
    locations = []
    for r in range(n - 1):
        for c in range(n - 1):
            points = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if (np.all(owner[r:r + 2, c:c + 2] == 0)
                    and all(not set(o) <= points for o in groups)):
                locations.append((r, c))
    return locations


def _valid_owner(owner):
    n = len(owner)
    corners = (owner[0, 0], owner[0, -1], owner[-1, 0], owner[-1, -1])
    return (tuple(corners) == (0, 1, 2, 3)
            and all(np.count_nonzero(owner == i) >= (6 if i == 0 else 3)
                    and len(_components(owner == i)) == 1 for i in range(4))
            and bool(_key_locations(owner)))


def _partition(rng, n, style):
    if style in ("rect", "hybrid", "seams"):
        y = int(rng.integers(2, n - 1))
        x = int(rng.integers(2, n - 1))
        z = int(rng.integers(1, n))
        if style == "rect" and n % 2 == 0 and rng.random() < 0.45:
            y = x = z = n // 2
        owner = np.empty((n, n), dtype=np.int8)
        top, bottom = x, z
        for r in range(n):
            if style == "seams" and r >= 2 and r < y:
                top = int(np.clip(top + rng.integers(-1, 2), 1, n - 1))
            if style != "rect" and r > y:
                bottom = int(np.clip(bottom + rng.integers(-1, 2), 1, n - 1))
            cut = top if r < y else bottom
            a, b = (0, 1) if r < y else (2, 3)
            owner[r, :cut], owner[r, cut:] = a, b
        if style != "rect":
            for _ in range(int(rng.integers(2, n + 2))):
                r, c = (int(v) for v in rng.integers(1, n - 1, size=2))
                # The hybrid keeps its top-right rectangle, which makes
                # geometry insufficient to orient its printed fragment.
                if style == "hybrid" and (owner[r, c] == 1 or any(
                        owner[r + dr, c + dc] == 1 for dr, dc in NEIGHBORS)):
                    continue
                old = int(owner[r, c])
                dr, dc = NEIGHBORS[int(rng.integers(4))]
                owner[r, c] = owner[r + dr, c + dc]
                if not _valid_owner(owner):
                    owner[r, c] = old
    elif style == "carved":
        large = int(rng.integers(4))
        owner = np.full((n, n), large, dtype=np.int8)
        for corner in rng.permutation([i for i in range(4) if i != large]):
            corner = int(corner)
            height, width = (int(v) for v in rng.integers(2, n - 1, size=2))
            profile = int(rng.integers(3))
            for d in range(height):
                if profile == 0:
                    reach = max(1, width - d * width // height)
                elif profile == 1:
                    reach = width if d in (0, height - 1) else 1
                else:
                    reach = int(rng.integers(1, width + 1))
                if corner == 0 and d < 2:
                    reach = max(2, reach)
                for e in range(reach):
                    r = d if corner < 2 else n - 1 - d
                    c = e if corner % 2 == 0 else n - 1 - e
                    if owner[r, c] != large:
                        return None
                    owner[r, c] = corner
    else:  # Connected frontier growth with independently biased region sizes.
        owner = np.full((n, n), -1, dtype=np.int8)
        owner[:2, :2] = 0
        owner[0, -1], owner[-1, 0], owner[-1, -1] = 1, 2, 3
        weights = rng.uniform(0.3, 2.0, 4)
        for _ in range(n * n - 7):
            options = []
            for r, c in np.argwhere(owner < 0):
                labels = {int(owner[r + dr, c + dc]) for dr, dc in NEIGHBORS
                          if 0 <= r + dr < n and 0 <= c + dc < n
                          and owner[r + dr, c + dc] >= 0}
                options.extend((int(r), int(c), label) for label in sorted(labels))
            probs = np.array([weights[label] for r, c, label in options])
            r, c, label = options[int(rng.choice(len(options), p=probs / probs.sum()))]
            owner[r, c] = label
    return owner if _valid_owner(owner) else None


def _stencil(rng, n):
    groups, _ = _orbits(n)
    style = int(rng.integers(4))
    band = int(rng.integers(1, 3))
    density = float(rng.uniform(0.3, 0.7))
    values = []
    coarse = rng.integers(0, 2, size=(n, n))
    for orbit in groups:
        r, c = orbit[0]
        if style == 0:
            v = int(rng.random() < density)
        elif style == 1:
            v = (max(r, c) // band) % 2
        elif style == 2:
            v = ((r + 2 * c) // band) % 2
        else:
            v = int(coarse[r // band, c // band])
        if style and rng.random() < 0.18:
            v = 1 - v
        values.append(v)
    values = np.array(values, dtype=np.int8)
    if len(set(values)) < 2:
        values[int(rng.integers(len(values)))] ^= 1
    result = np.empty((n, n), dtype=np.int8)
    for value, orbit in zip(values, groups):
        for p in orbit:
            result[p] = value
    return result


def _mask(rng, stencil, owner, stamp, hard, test):
    n = len(stencil)
    groups, _ = _orbits(n)
    mask = np.zeros((n, n), dtype=bool)
    # Training has local blocks / pixels; tests use new long or stepped damage.
    if test:
        vertical = bool(rng.integers(2))
        start = int(rng.integers(n))
        for t in range(int(rng.integers(n // 2, n + 1))):
            drift = (t // 2) % 2 if rng.random() < 0.5 else 0
            r, c = (t, min(n - 1, start + drift)) if vertical else (min(n - 1, start + drift), t)
            mask[r, c] = True
    elif rng.random() < 0.5:
        h, w = (int(v) for v in rng.integers(1, 3, size=2))
        r, c = int(rng.integers(n - h + 1)), int(rng.integers(n - w + 1))
        mask[r:r + h, c:c + w] = True
    else:
        mask = rng.random((n, n)) < rng.uniform(0.025, 0.10)
    mask |= stamp
    protected = stamp.copy()
    # Both possible missing states are demonstrated; hard pairs include two
    # triple-erased orbits, with ink and blank witnesses on other sheets.
    for state in (0, 1):
        options = [o for o in groups if len(o) == 4 and stencil[o[0]] == state
                   and len({int(owner[p]) for p in o}) >= 2
                   and sum(not stamp[p] for p in o) >= 2]
        if not options:
            return None
        orbit = options[int(rng.integers(len(options)))]
        available = [p for p in orbit if not stamp[p]]
        witness = available[int(rng.integers(len(available)))]
        if hard:
            for p in orbit:
                mask[p] = p != witness
                protected[p] = p != witness
        else:
            candidates = [p for p in orbit if p != witness and not stamp[p]]
            p = candidates[int(rng.integers(len(candidates)))]
            mask[p] = True
        mask[witness] = False
    for orbit in groups:
        if all(mask[p] for p in orbit):
            available = [p for p in orbit if not protected[p]]
            if not available:
                return None
            mask[available[int(rng.integers(len(available)))]] = False
    # The stamp is an actual occlusion, including at least one ink state.
    if not np.any(stencil[stamp] == 1):
        return None
    return mask


def _scatter(rng, tiles, key, stamp_location, bg):
    """Free halo placement or randomized shelves, with padding and a fallback."""
    count = len(tiles)
    free_budget = 12 if rng.random() < 0.55 else 0
    try_linear = bool(rng.random() < 0.30)
    for attempt in range(24):
        if attempt < free_budget:
            h = int(rng.integers(max(t.shape[0] for t in tiles) + 2, 31))
            w = int(rng.integers(max(t.shape[1] for t in tiles) + 2, 31))
            canvas = np.full((h, w), bg, dtype=np.int8)
            blocked = np.zeros((h, w), dtype=bool)
            offsets, failed = [None] * count, False
            for i in rng.permutation(count):
                i = int(i)
                tile = tiles[i]
                th, tw = tile.shape
                coords = np.argwhere(tile != OUTSIDE)
                for _ in range(80):
                    r, c = int(rng.integers(h - th + 1)), int(rng.integers(w - tw + 1))
                    yy, xx = coords[:, 0] + r, coords[:, 1] + c
                    if blocked[yy, xx].any():
                        continue
                    offsets[i] = (r, c)
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            blocked[np.clip(yy + dr, 0, h - 1), np.clip(xx + dc, 0, w - 1)] = True
                    break
                else:
                    failed = True
                    break
            if failed:
                continue
        else:
            # A shelf partition is a presentation choice, not an output layout.
            transposed = bool(rng.integers(2))
            dims = [t.shape[::-1] if transposed else t.shape for t in tiles]
            order = [int(i) for i in rng.permutation(count)]
            groups, group = [], []
            for j, i in enumerate(order):
                group.append(i)
                if j == 3 or (not (try_linear and attempt < 10)
                              and bool(rng.integers(2))):
                    groups.append(group)
                    group = []
            gap = int(rng.integers(1, 4))
            heights = [max(dims[i][0] for i in group) for group in groups]
            widths = [sum(dims[i][1] for i in group) + gap * (len(group) - 1) for group in groups]
            h, w = sum(heights) + gap * (len(groups) - 1), max(widths)
            if max(h, w) > 28:
                continue
            offsets, r = [None] * count, 0
            for group, height, width in zip(groups, heights, widths):
                c = int(rng.integers(w - width + 1))
                for i in group:
                    rr = r + int(rng.integers(height - dims[i][0] + 1))
                    offsets[i] = (c, rr) if transposed else (rr, c)
                    c += dims[i][1] + gap
                r += height + gap
            if transposed:
                h, w = w, h
            canvas = np.full((h, w), bg, dtype=np.int8)
        for i, tile in enumerate(tiles):
            coords = np.argwhere(tile != OUTSIDE)
            r, c = offsets[i]
            values = tile[coords[:, 0], coords[:, 1]]
            pixels = np.where(values == 0, BLANK, np.where(values == 1, INK, GAP))
            canvas[coords[:, 0] + r, coords[:, 1] + c] = pixels
        r, c = offsets[0]
        kr, kc = stamp_location
        canvas[r + kr:r + kr + 2, c + kc:c + kc + 2] = key
        coords = np.argwhere(canvas != bg)
        low, high = coords.min(axis=0), coords.max(axis=0) + 1
        canvas = canvas[low[0]:high[0], low[1]:high[1]]
        pads = []
        for d in canvas.shape:
            room = 30 - d
            before = int(rng.integers(min(3, room) + 1))
            after = int(rng.integers(min(3, room - before) + 1))
            pads.append((before, after))
        canvas = np.pad(canvas, pads, constant_values=bg)
        border = np.concatenate((canvas[0], canvas[-1], canvas[1:-1, 0], canvas[1:-1, -1]))
        if np.count_nonzero(border == bg) > len(border) / 2:
            return canvas
    # Always fits for n<=12, with a guaranteed blank border.
    dims = [t.shape for t in tiles]
    row = 2 + max(dims[0][0], dims[1][0])
    offsets = [(1, 1), (1, dims[0][1] + 2), (row, 1), (row, dims[2][1] + 2)]
    h = row + max(dims[2][0], dims[3][0]) + 1
    w = max(dims[0][1] + dims[1][1], dims[2][1] + dims[3][1]) + 3
    canvas = np.full((h, w), bg, dtype=np.int8)
    for tile, (r, c) in zip(tiles, offsets):
        coords = np.argwhere(tile != OUTSIDE)
        v = tile[coords[:, 0], coords[:, 1]]
        canvas[coords[:, 0] + r, coords[:, 1] + c] = np.where(v == 0, BLANK, np.where(v == 1, INK, GAP))
    kr, kc = stamp_location
    canvas[1 + kr:3 + kr, 1 + kc:3 + kc] = key
    return canvas


def _make_pair(rng, n, role, key, bg):
    test, hard = role == "test", role in ("ambiguous", "test")
    for attempt in range(96):
        if role in ("ambiguous", "test"):
            style = str(rng.choice(("rect", "hybrid")))
        else:
            style = str(rng.choice(("seams", "carved", "growth", "hybrid")))
        owner = _partition(rng, n, style)
        if owner is None:
            continue
        kr, kc = _key_locations(owner)[int(rng.integers(len(_key_locations(owner))))]
        stamp = np.zeros((n, n), dtype=bool)
        stamp[kr:kr + 2, kc:kc + 2] = True
        for _ in range(12):
            stencil = _stencil(rng, n)
            mask = _mask(rng, stencil, owner, stamp, hard, test)
            if mask is None:
                continue
            raw = stencil.copy()
            raw[mask] = UNKNOWN
            tiles = []
            turns = [0] + ([1, 2, 3] if role == "ambiguous" else [int(rng.integers(4)) for _ in range(3)])
            if role == "ambiguous":
                turns[1:] = [int(v) for v in rng.permutation((1, 2, 3))]
            for i in range(4):
                coords = np.argwhere(owner == i)
                low, high = coords.min(axis=0), coords.max(axis=0) + 1
                tile = np.full(tuple(high - low), OUTSIDE, dtype=np.int8)
                local = coords - low
                tile[local[:, 0], local[:, 1]] = raw[coords[:, 0], coords[:, 1]]
                tiles.append(np.rot90(tile, turns[i]))
            grid = _scatter(rng, tiles, key, (kr, kc), bg)
            output = np.where(stencil == 1, key.ravel()[owner], bg).astype(np.int8)
            if set(output.flat) != {bg, *key.flat}:
                continue
            try:
                analysis = _analyze(grid)
            except AssertionError:
                continue
            if not np.array_equal(analysis["output"], output):
                continue
            f = _features(analysis)
            if (not np.any(grid == GAP) or f["cross_sheet_witness"] == 0
                    or f["literal_pigment_conflicts"] == 0
                    or f["explicit_ink_gaps"] == 0 or f["explicit_blank_gaps"] == 0):
                continue
            if hard and (analysis["geometry_candidates"] < 2 or analysis["mirror_rejected"] < 1):
                continue
            if hard and f["diagonal_only"] == 0:
                continue
            if role == "demonstration" and f["quadrant_counterexample"] == 0:
                continue
            if role == "ambiguous" and analysis["translation_only"]:
                continue
            return {"input": grid, "output": output}
    raise RuntimeError("bounded joint-constraint sampling exhausted")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    # The minimal two-example training set already contrasts parity, local
    # evidence, ambiguity, and rotation. Positions are randomized.
    special = [int(i) for i in rng.choice(num_train, 2, replace=False)]
    puzzle, seen, train_keys = {"train": [], "test": []}, set(), []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            role = "test" if split == "test" else (
                "ambiguous" if index == special[1] else
                "demonstration" if index == special[0] else "local")
            if split == "train" and index == special[0]:
                n = int(rng.choice((9, 11) if regime == 2 else (7, 9)))
            elif role == "ambiguous":
                n = 10 if regime == 2 else int(rng.choice((6, 8)))
            elif split == "test":
                choices = ((11, 12) if regime == 0 else
                           (6, 7, 8) if regime == 2 else
                           tuple(q["output"].shape[0] for q in puzzle["train"])
                           if regime == 1 else tuple(range(6, 13)))
                n = int(rng.choice(choices))
            else:
                n = int(rng.integers(9, 12) if regime == 2 else rng.integers(6, 11))
            if split == "test" and train_keys and rng.random() < 0.30:
                key = train_keys[int(rng.integers(len(train_keys)))].copy()
            else:
                key = rng.choice(PIGMENTS, 4, replace=False).astype(np.int8).reshape(2, 2)
            if split == "train" and index == 1 and np.array_equal(key, train_keys[0]):
                key = np.roll(key.ravel(), 1).reshape(2, 2)
            bg = int(rng.choice([c for c in PIGMENTS if c not in key]))
            for _ in range(8):
                pair = _make_pair(rng, n, role, key, bg)
                sig = _signature(pair["input"])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input sampling exhausted")
            if split == "train":
                train_keys.append(key)
    return puzzle


def _features(a):
    n, raw, owner, stencil = a["side"], a["raw"], a["owner"], a["stencil"]
    unknown = raw == UNKNOWN
    explicit = unknown & ~a["stamp_mask"]
    rows, cols = np.indices((n, n))
    quadrants = 2 * (rows > (n - 1) / 2) + (cols > (n - 1) / 2)
    off_axis = (rows != (n - 1) / 2) & (cols != (n - 1) / 2)
    diagonal = cross = pigment_conflicts = 0
    for r, c in np.argwhere(unknown):
        rr, cc = n - 1 - r, n - 1 - c
        diagonal += int(raw[r, cc] == UNKNOWN and raw[rr, c] == UNKNOWN
                        and raw[rr, cc] >= 0)
        partners = ((r, cc), (rr, c), (rr, cc))
        cross += int(any(raw[p] >= 0 and owner[p] != owner[r, c] for p in partners))
    for orbit in _orbits(n)[0]:
        pigment_conflicts += int(len({int(owner[p]) for p in orbit if raw[p] == 1}) > 1)
    areas, fill, rectangles = [], [], 0
    for i in range(4):
        coords = np.argwhere(owner == i)
        low, high = coords.min(axis=0), coords.max(axis=0) + 1
        areas.append(len(coords))
        fill.append(len(coords) / int(np.prod(high - low)))
        rectangles += int(len(coords) == np.prod(high - low))
    return {"side": n, "background": a["background"],
            "areas": tuple(areas), "area_ratio": max(areas) / min(areas),
            "min_fill": min(fill), "rectangles": rectangles,
            "unknown": int(unknown.sum()), "masked_ink": int(np.count_nonzero(unknown & (stencil == 1))),
            "masked_blank": int(np.count_nonzero(unknown & (stencil == 0))),
            "explicit_ink_gaps": int(np.count_nonzero(explicit & (stencil == 1))),
            "explicit_blank_gaps": int(np.count_nonzero(explicit & (stencil == 0))),
            "stamp_ink": int(np.count_nonzero(a["stamp_mask"] & (stencil == 1))),
            "quadrant_counterexample": int(np.count_nonzero((stencil == 1) & off_axis & (owner != quadrants))),
            "diagonal_only": diagonal, "cross_sheet_witness": cross,
            "literal_pigment_conflicts": pigment_conflicts,
            "geometry_candidates": a["geometry_candidates"], "mirror_rejected": a["mirror_rejected"],
            "rotations": a["turns"], "translation_only": a["translation_only"],
            "density": float(stencil.mean()),
            "mask_components": len(_components(unknown))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    seen, records, keys = set(), {"train": [], "test": []}, []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            a = _analyze(inp)
            _require(np.array_equal(a["output"], out), "output disagrees with joint pixel reconstruction")
            _require(np.array_equal(out != a["background"], a["stencil"] == 1),
                     "printing must preserve symmetric ink occupancy")
            _require(np.array_equal(a["stencil"], a["stencil"][::-1])
                     and np.array_equal(a["stencil"], a["stencil"][:, ::-1]), "two stencil folds")
            _require(np.all(a["raw"][a["raw"] >= 0] == a["stencil"][a["raw"] >= 0]),
                     "visible stencil evidence changed")
            _require(set(out.flat) == {a["background"], *a["key"].flat},
                     "every key entry must print ink and some cells must remain blank")
            f = _features(a)
            _require(f["masked_ink"] > 0 and f["masked_blank"] > 0,
                     "both possible missing states must occur")
            _require(f["explicit_ink_gaps"] > 0 and f["explicit_blank_gaps"] > 0
                     and f["stamp_ink"] > 0,
                     "explicit erasures must include both states and stamp must hide ink")
            _require(f["cross_sheet_witness"] > 0 and f["literal_pigment_conflicts"] > 0,
                     "repair must cross sheets and differ from literal pigment mirroring")
            _require(np.any(inp == GAP), "explicit erasures required")
            for i in range(4):
                _require(len(_components(a["owner"] == i)) == 1, "assembled sheet disconnected")
            records[split].append(f)
            if split == "train":
                keys.append(tuple(a["key"].flat))
    _require(len(set(keys)) >= 2, "training must show a changing stamp")
    _require(any(f["diagonal_only"] > 0 and f["mirror_rejected"] > 0
                 and not f["translation_only"] for f in records["train"]),
             "training must expose joint ambiguity, both folds, and rotation")
    _require({f["side"] % 2 for f in records["train"]} == {0, 1},
             "training must demonstrate both axis parities")
    _require(any(f["quadrant_counterexample"] > 0 for f in records["train"]),
             "training must distinguish sheet coloring from quadrant coloring")
    for f in records["test"]:
        _require(f["diagonal_only"] > 0 and f["mirror_rejected"] > 0,
                 "test must require both folds to repair and reject geometric alternatives")


def self_test():
    """Bounded 42-seed feature audit, plus count and corruption checks."""
    shapes, layouts, backgrounds, keys, sides, rotations = set(), set(), set(), set(), set(), set()
    ratios, fills, densities, unknowns, ambiguous = [], [], [], set(), 0
    first_puzzle = None
    signatures = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        if first_puzzle is None:
            first_puzzle = p
        whole = tuple(_signature(q[k]) for s in ("train", "test") for q in p[s] for k in ("input", "output"))
        _require(whole not in signatures, "duplicate puzzle across seeds")
        signatures.add(whole)
        for split in ("train", "test"):
            for pair in p[split]:
                a = _analyze(pair["input"])
                f = _features(a)
                shapes.add(pair["input"].shape)
                layouts.add((a["side"], a["owner"].tobytes()))
                backgrounds.add(a["background"])
                keys.add(tuple(a["key"].flat))
                sides.add(a["side"])
                rotations.update(a["turns"])
                ratios.append(f["area_ratio"])
                fills.append(f["min_fill"])
                densities.append(f["density"])
                unknowns.add(f["unknown"])
                ambiguous += int(f["mirror_rejected"] > 0)
                # Independent expression of the local orbit invariant, not an
                # independent solver: no orbit may contradict or lack evidence.
                for orbit in _orbits(a["side"])[0]:
                    witnesses = {int(a["raw"][point]) for point in orbit if a["raw"][point] >= 0}
                    _require(witnesses == {int(a["stencil"][orbit[0]])}, "orbit evidence invariant")
    repeat = generate_puzzle(0)
    _require(all(np.array_equal(first_puzzle[s][i][k], repeat[s][i][k])
                 for s in ("train", "test") for i in range(len(repeat[s]))
                 for k in ("input", "output")), "seed determinism")
    before = first_puzzle["test"][0]["input"].copy()
    solve(first_puzzle["test"][0]["input"])
    _require(np.array_equal(before, first_puzzle["test"][0]["input"]), "solver mutated its input")
    a = _analyze(before)
    orbit = next(o for o in _orbits(a["side"])[0] if sum(a["raw"][p] >= 0 for p in o) >= 2)
    damaged_raw = a["raw"].copy()
    point = next(p for p in orbit if damaged_raw[p] >= 0)
    damaged_raw[point] ^= 1
    _require(_complete(damaged_raw) is None, "conflicting orbit was accepted")
    damaged_raw = a["raw"].copy()
    for point in orbit:
        damaged_raw[point] = UNKNOWN
    _require(_complete(damaged_raw) is None, "witness-free orbit was accepted")
    for nt, ns in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(121 + nt + ns, nt, ns)
        validate_generated_puzzle(p)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(len(shapes) >= 40 and len(layouts) >= 80, "geometry diversity")
    _require(sides == set(range(6, 13)) and rotations == {0, 1, 2, 3}, "scale/orientation diversity")
    _require(backgrounds == set(PIGMENTS) and len(keys) >= 80, "palette-role diversity")
    _require(max(ratios) >= 4 and min(fills) < 0.7, "unequal and concave sheets")
    _require(max(densities) - min(densities) >= 0.35 and len(unknowns) >= 12,
             "stencil and damage diversity")
    _require(ambiguous >= 84, "reflection must often constrain placement")
    corrupted = {s: [{k: v.copy() for k, v in q.items()} for q in first_puzzle[s]] for s in first_puzzle}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted printing was accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "owner_layouts": len(layouts), "backgrounds": sorted(backgrounds),
            "sides": sorted(sides), "rotations": sorted(rotations),
            "key_matrices": len(keys), "ambiguous_pairs": ambiguous,
            "max_area_ratio": max(ratios), "min_bbox_fill": min(fills),
            "stencil_density_range": [min(densities), max(densities)],
            "distinct_erasure_counts": len(unknowns)}
