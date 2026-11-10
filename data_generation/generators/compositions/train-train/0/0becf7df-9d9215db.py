"""Nested exchange membranes: keyed transpositions scoped by implied frames.

Only NumPy and the standard library are used. No parent code is imported.
"""

from collections import Counter

import numpy as np


_SIGNS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a


def _signature(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _geometry(n):
    rr, cc = np.indices((n, n))
    dr, dc = np.abs(rr - n // 2), np.abs(cc - n // 2)
    return np.maximum(dr, dc), dr == dc


def _key_candidates(a, bg):
    blocks = (a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:])
    good = np.ones(blocks[0].shape, dtype=bool)
    for i in range(4):
        good &= blocks[i] != bg
        for j in range(i):
            good &= blocks[i] != blocks[j]
    result = []
    for r, c in np.argwhere(good):
        r, c = int(r), int(c)
        halo = a[max(0, r - 1):r + 3, max(0, c - 1):c + 3].copy()
        hr, hc = r - max(0, r - 1), c - max(0, c - 1)
        halo[hr:hr + 2, hc:hc + 2] = bg
        if np.all(halo == bg):
            result.append((r, c))
    return result


def _key_masks(n, key):
    r, c = key
    mask = np.zeros((n, n), dtype=bool)
    mask[r:r + 2, c:c + 2] = True
    halo = np.zeros_like(mask)
    halo[max(0, r - 1):r + 3, max(0, c - 1):c + 3] = True
    return mask, halo


def _template(n, bg, orientation, rings):
    a = np.full((n, n), bg, dtype=np.int8)
    mask = np.zeros((n, n), dtype=bool)
    sr, sc = orientation
    center = n // 2
    for radius, corner, side in rings:
        r, c = center + sr * radius, center + sc * radius
        for cell, color in (((r, c), corner), ((r - 2 * sr, c), side), ((r, c - 2 * sc), side)):
            a[cell] = color
            mask[cell] = True
    return a, mask


def _decode(input_grid):
    a = _grid(input_grid)
    n = a.shape[0]
    _require(a.shape == (n, n) and 11 <= n <= 29 and n % 2, "expected an odd square")
    counts = np.bincount(a.ravel().astype(np.int64), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > a.size // 2, "background must be a strict majority")
    candidates = _key_candidates(a, bg)
    _require(len(candidates) == 1, "expected one isolated four-color key")
    key = candidates[0]
    kr, kc = key
    legend = a[kr:kr + 2, kc:kc + 2].copy()
    palette = set(map(int, legend.flat))
    _require(set(map(int, np.unique(a))) <= palette | {bg}, "unknown color")
    key_mask, halo = _key_masks(n, key)
    stripped = a.copy()
    stripped[key_mask] = bg
    center = n // 2
    source_candidates = []
    for orientation in _SIGNS:
        sr, sc = orientation
        radii = [r for r in range(2, center - 2) if stripped[center + sr * r, center + sc * r] != bg]
        if radii:
            source_candidates.append((orientation, radii))
    _require(len(source_candidates) == 1, "expected one common source corner")
    orientation, radii = source_candidates[0]
    _require(1 <= len(radii) <= 5 and all(b - a >= 2 for a, b in zip(radii, radii[1:])), "invalid radii")
    sr, sc = orientation
    rings = []
    for radius in radii:
        r, c = center + sr * radius, center + sc * radius
        corner = int(stripped[r, c])
        side = int(stripped[r - 2 * sr, c])
        _require(side == stripped[r, c - 2 * sc], "arm samples disagree")
        rings.append((radius, corner, side))
    template, template_mask = _template(n, bg, orientation, rings)
    _require(np.array_equal(a[template_mask], template[template_mask]), "invalid template")
    _require(not np.any(halo & template_mask), "key touches a template")
    payload = (a != bg) & ~key_mask & ~template_mask
    distance, diagonal = _geometry(n)
    boundaries = np.isin(distance, radii)
    _require(not np.any(payload & (diagonal | boundaries | halo)), "payload occupies a reserved location")
    # The entire key moat lies in one chamber, also clear of future frames.
    depth = sum((distance < radius).astype(np.int8) for radius in radii)
    _require(len(np.unique(depth[halo])) == 1 and not np.any(boundaries & halo), "key straddles a membrane")
    _require(np.any(payload), "empty payload")
    return a, bg, key, legend, orientation, rings, payload


def _render(a, bg, key, legend, rings, payload, mode="normal"):
    """Render from a decoded description; mode exposes specific false rules."""
    n = a.shape[0]
    center = n // 2
    distance, _ = _geometry(n)
    out = np.full_like(a, bg)
    kr, kc = key
    out[kr:kr + 2, kc:kc + 2] = legend
    for radius, corner, side in rings:
        lo, hi = center - radius, center + radius
        for position in range(lo + 2, hi, 2):
            out[lo, position] = out[hi, position] = side
            out[position, lo] = out[position, hi] = side
        for r in (lo, hi):
            for c in (lo, hi):
                out[r, c] = corner
    out[payload] = a[payload]
    row_for = {int(color): row for row in range(2) for color in legend[row]}
    for row in range(2):
        counts = np.zeros_like(a, dtype=np.int8)
        for radius, corner, side in rings:
            if mode == "visible_only" and side == bg:
                continue
            selector = row_for[side] if mode == "side_select" and side != bg else row_for[corner]
            if mode == "both_rows" or selector == row:
                counts += distance < radius
        toggle = counts % 2 == 1
        if mode == "no_pairing":
            toggle[:] = False
        elif mode == "global":
            toggle[:] = True
        elif mode == "any_enclosure":
            toggle = counts > 0
        first, second = map(int, legend[row])
        out[payload & toggle & (a == first)] = second
        out[payload & toggle & (a == second)] = first
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only visible key, corner templates, and token coordinates."""
    a, bg, key, legend, _, rings, payload = _decode(input_grid)
    return _render(a, bg, key, legend, rings, payload)


def _shape(rng):
    kind = int(rng.integers(7))
    h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    if kind == 0:
        shape = np.ones((1, 1), dtype=bool)
    elif kind == 1:
        shape = np.ones((1, int(rng.integers(2, 6))), dtype=bool)
    elif kind == 2:
        shape = np.ones((h, w), dtype=bool)
    elif kind == 3:
        shape = np.zeros((h, w), dtype=bool)
        shape[:, 0] = shape[-1, :] = True
    elif kind == 4:
        shape = np.zeros((max(3, h), max(3, w)), dtype=bool)
        shape[0, :] = shape[-1, :] = shape[:, 0] = shape[:, -1] = True
    elif kind == 5:
        shape = np.zeros((3, 3), dtype=bool)
        shape[1, :] = shape[:, 1] = True
    else:
        shape = np.array(((1, 1, 0), (0, 1, 1)), dtype=bool)
    return np.rot90(shape, int(rng.integers(4))).copy()


def _dilate4(mask):
    result = mask.copy()
    result[1:] |= mask[:-1]
    result[:-1] |= mask[1:]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result


def _sample_radii(rng, n, count):
    # Adding successive offsets to a compressed subset guarantees gaps >= 2.
    compressed = np.sort(rng.choice(np.arange(2, n // 2 - 2 - (count - 1)), size=count, replace=False))
    return [int(r) for r in compressed + np.arange(count)]


def _choose_key(rng, n, radii, outside):
    distance, _ = _geometry(n)
    depth = sum((distance < radius).astype(np.int8) for radius in radii)
    candidates = []
    for r in range(n - 1):
        for c in range(n - 1):
            d = distance[max(0, r - 1):r + 3, max(0, c - 1):c + 3]
            z = depth[max(0, r - 1):r + 3, max(0, c - 1):c + 3]
            if not np.isin(d, radii).any() and len(np.unique(z)) == 1 and (not outside or z.flat[0] == 0):
                candidates.append((r, c))
    edge = [p for p in candidates if p[0] in (0, n - 2) or p[1] in (0, n - 2)]
    interior = [p for p in candidates if p not in edge]
    pool = edge if rng.random() < 0.55 or not interior else interior
    return pool[int(rng.integers(len(pool)))]


def _paint_payload(rng, a, bg, key, legend, rings, full_coverage):
    n = len(a)
    radii = [r for r, _, _ in rings]
    distance, diagonal = _geometry(n)
    _, halo = _key_masks(n, key)
    free = ~diagonal & ~np.isin(distance, radii) & ~halo
    chamber = sum((distance < r).astype(np.int8) for r in radii)
    used = np.zeros_like(free)
    rr, cc = np.indices(a.shape)
    colors = list(map(int, legend.flat))
    # Every anchor chamber has all four colors, allowing like-for-like comparisons.
    for depth in range(len(rings) + 1):
        zone = free & (chamber == depth)
        phases = [p for p in (0, 1) if np.count_nonzero(zone & ((rr + cc) % 2 == p)) >= (4 if full_coverage else 1)]
        if not phases:
            return None
        p = int(rng.choice(phases))
        coordinates = np.argwhere(zone & ((rr + cc) % 2 == p))
        count = 4 if full_coverage else min(len(coordinates), int(rng.integers(1, 4)))
        chosen = coordinates[rng.choice(len(coordinates), size=count, replace=False)]
        local_colors = list(rng.permutation(colors)) if full_coverage else rng.choice(colors, size=count).tolist()
        if not full_coverage and depth == 1:
            # This chamber lies inside only the outer membrane. Guarantee a
            # visible exchange witness, even when other chambers cancel.
            active_row = int(np.argwhere(legend == rings[-1][1])[0, 0])
            if not any(color in legend[active_row] for color in local_colors):
                local_colors[0] = int(rng.choice(legend[active_row]))
        for (r, c), color in zip(chosen, local_colors):
            a[r, c] = color
            used[r, c] = True
    for _ in range(int(rng.integers(2, 10))):
        shape = _shape(rng)
        sh, sw = shape.shape
        for _ in range(45):
            r = int(rng.integers(n - sh + 1))
            c = int(rng.integers(n - sw + 1))
            patch = free[r:r + sh, c:c + sw] & ~_dilate4(used)[r:r + sh, c:c + sw]
            depths = chamber[r:r + sh, c:c + sw][shape]
            if not np.all(patch[shape]) or len(np.unique(depths)) != 1:
                continue
            palette = rng.choice(colors, size=int(rng.integers(1, 3)), replace=False)
            lr, lc = np.indices(shape.shape)
            labels = (lr if rng.random() < 0.5 else lc) % len(palette)
            a[r:r + sh, c:c + sw][shape] = palette[labels[shape]]
            used[r:r + sh, c:c + sw] |= shape
            break
    return used


def _make_pair(rng, count, legend=None, forbidden_bg=None, forbidden_size=None, forbidden_orientation=None, anchor=False, full_coverage=False):
    if legend is None:
        legend = rng.choice(10, size=4, replace=False).reshape(2, 2).astype(np.int8)
    colors = list(map(int, legend.flat))
    bg = int(rng.choice([c for c in range(10) if c not in colors and c != forbidden_bg]))
    sizes = [n for n in range(4 * count + 7, 30, 2) if n != forbidden_size]
    n = int(rng.choice(sizes))
    orientation = _SIGNS[int(rng.choice([i for i, o in enumerate(_SIGNS) if o != forbidden_orientation]))]
    for _ in range(60):
        radii = _sample_radii(rng, n, count)
        selectors = rng.integers(2, size=count).tolist()
        if anchor:
            repeated = int(rng.integers(2))
            selectors = [repeated, 1 - repeated, repeated]
        corners = [int(rng.choice(legend[row])) for row in selectors]
        sides = [bg if rng.random() < 0.35 else int(rng.choice(colors)) for _ in radii]
        if anchor:
            sides[0] = bg
            sides[1] = int(rng.choice(legend[1 - selectors[1]]))
            sides[2] = int(rng.choice(colors))
        rings = list(zip(radii, corners, sides))
        key = _choose_key(rng, n, radii, outside=full_coverage)
        a, _ = _template(n, bg, orientation, rings)
        kr, kc = key
        a[kr:kr + 2, kc:kc + 2] = legend
        payload = _paint_payload(rng, a, bg, key, legend, rings, full_coverage)
        if payload is None or _key_candidates(a, bg) != [key]:
            continue
        out = _render(a, bg, key, legend, rings, payload)
        if np.array_equal(a, out):
            continue
        return {"input": a, "output": out}
    raise RuntimeError("could not sample a valid membrane drawing")


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and low <= value <= high, "illegal example count")
    _require(num_train > num_test, "num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    more_layers = bool(rng.integers(2))
    train_counts = [3] + rng.integers(1 if more_layers else 3, 4 if more_layers else 6, size=num_train - 1).tolist()
    if not more_layers:
        train_counts[1] = int(rng.integers(4, 6))
    test_counts = rng.integers(4 if more_layers else 1, 6 if more_layers else 3, size=num_test).tolist()
    first_legend = rng.choice(10, size=4, replace=False).reshape(2, 2).astype(np.int8)
    flat = first_legend.ravel()
    second_legend = flat[[0, 2, 1, 3]].reshape(2, 2).copy()
    if rng.integers(2):
        second_legend = second_legend[::-1].copy()
    for row in second_legend:
        if rng.integers(2):
            row[:] = row[::-1]
    puzzle = {"train": [], "test": []}
    seen = set()
    first = None
    for split, counts in (("train", train_counts), ("test", test_counts)):
        for index, count in enumerate(counts):
            for _ in range(30):
                is_first = split == "train" and index == 0
                is_second = split == "train" and index == 1
                pair = _make_pair(
                    rng, count,
                    legend=first_legend if is_first else second_legend if is_second else None,
                    forbidden_bg=first[1] if is_second else None,
                    forbidden_size=len(first[0]) if is_second else None,
                    forbidden_orientation=first[4] if is_second else None,
                    anchor=is_first, full_coverage=is_first or is_second,
                )
                signature = _signature(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("duplicate input sampling exhausted")
            seen.add(signature)
            puzzle[split].append(pair)
            if is_first:
                first = _decode(pair["input"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible encoding, full semantics, and the teaching guarantees."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, decoded = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            data = _decode(pair["input"])
            a, bg, key, legend, orientation, rings, payload = data
            out = _grid(pair["output"])
            _require(a.shape == out.shape, "output shape changed")
            signature = _signature(a)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            _require(np.array_equal(out, _render(a, bg, key, legend, rings, payload)), "incorrect output")
            key_mask, _ = _key_masks(len(a), key)
            _, template_mask = _template(len(a), bg, orientation, rings)
            _require(np.array_equal(out[key_mask | template_mask], a[key_mask | template_mask]), "control marks changed")
            _require(np.array_equal(out[payload] == bg, a[payload] == bg), "token occupancy changed")
            _require(np.any(out[payload] != a[payload]), "pair has no active exchange witness")
            distance, _ = _geometry(len(a))
            exterior = payload & (distance > max(r for r, _, _ in rings))
            _require(np.any(exterior) and np.array_equal(out[exterior], a[exterior]), "missing unchanged exterior witness")
            # Independently check each token against a list of its enclosing codes.
            for r, c in np.argwhere(payload):
                color = int(a[r, c])
                row = int(np.argwhere(legend == color)[0, 0])
                hits = sum(distance[r, c] < radius and corner in legend[row] for radius, corner, _ in rings)
                expected = int(legend[row, 1 - int(np.argwhere(legend[row] == color)[0, 0])]) if hits % 2 else color
                _require(out[r, c] == expected, "incorrect enclosure parity")
            # Every dotted frame has exactly four coded corners; all other samples use its arm value.
            for radius, corner, side in rings:
                lo, hi = len(a) // 2 - radius, len(a) // 2 + radius
                for r in (lo, hi):
                    for c in (lo, hi):
                        _require(out[r, c] == corner, "wrong completed corner")
                expected_nonbg = 4 + (4 * (radius - 1) if side != bg else 0)
                _require(np.count_nonzero((out != bg) & (distance == radius)) == expected_nonbg, "wrong dotted perimeter")
            decoded[split].append(data)
    for data in decoded["train"][:2]:
        a, bg, key, legend, _, rings, payload = data
        distance, _ = _geometry(len(a))
        depth = sum((distance < radius).astype(int) for radius, _, _ in rings)
        for zone in range(len(rings) + 1):
            _require(set(map(int, a[payload & (depth == zone)])) == set(map(int, legend.flat)), "anchor lacks chamber color coverage")
    first, second = decoded["train"][:2]
    _require(set(map(int, first[3].flat)) == set(map(int, second[3].flat)), "first two palettes must be shared")
    matching = lambda key: {frozenset(map(int, row)) for row in key}
    _require(matching(first[3]) != matching(second[3]), "second key must change partners")
    a, bg, key, legend, _, rings, payload = first
    for false_rule in ("no_pairing", "global", "any_enclosure", "visible_only", "side_select", "both_rows"):
        _require(not np.array_equal(_render(a, bg, key, legend, rings, payload, false_rule), puzzle["train"][0]["output"]), "anchor admits shortcut: " + false_rule)
    tr = {len(d[5]) for d in decoded["train"]}
    te = {len(d[5]) for d in decoded["test"]}
    _require(max(tr) < min(te) or min(tr) > max(te), "missing ring-count generalization shift")


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _component_sizes(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    sizes = []
    while cells:
        stack = [cells.pop()]
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if cell in cells:
                    cells.remove(cell)
                    stack.append(cell)
        sizes.append(size)
    return sizes


def self_test():
    """Bounded diversity, counterfactual, count, determinism, and mutation checks."""
    stats = {name: set() for name in ("sizes", "backgrounds", "orientations", "ring_counts", "radius_gaps", "key_positions", "components", "visibility", "layouts")}
    pair_count, mixed_phase, black_payload, inner_key = 0, 0, 0, 0
    shifts = Counter()
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        train_counts = [len(_decode(p["input"])[5]) for p in puzzle["train"]]
        test_counts = [len(_decode(p["input"])[5]) for p in puzzle["test"]]
        shifts["more" if max(train_counts) < min(test_counts) else "fewer"] += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _decode(pair["input"])
                a, bg, key, legend, orientation, rings, payload = data
                original = a.copy()
                result = solve(a)
                assert np.array_equal(a, original) and np.array_equal(result, pair["output"])
                assert not np.shares_memory(result, a)
                assert np.array_equal(solve(a.tolist()), result)
                pair_count += 1
                stats["sizes"].add(len(a))
                stats["backgrounds"].add(bg)
                stats["orientations"].add(orientation)
                stats["ring_counts"].add(len(rings))
                radii = tuple(r for r, _, _ in rings)
                stats["radius_gaps"].update(b - a for a, b in zip(radii, radii[1:]))
                stats["key_positions"].add((key[0] in (0, len(a) - 2), key[1] in (0, len(a) - 2)))
                stats["components"].update(_component_sizes(payload))
                stats["visibility"].add(sum(side != bg for _, _, side in rings))
                stats["layouts"].add((len(a), key, orientation, radii, tuple(side != bg for _, _, side in rings), payload.tobytes()))
                mixed_phase += len({r % 2 for r in radii}) > 1
                black_payload += bg != 0 and np.any(a[payload] == 0)
                distance, _ = _geometry(len(a))
                inner_key += distance[key] < max(radii)
    assert _same(puzzles[0], generate_puzzle(0)) and not _same(puzzles[0], puzzles[1])
    assert len(stats["sizes"]) >= 8 and len(stats["backgrounds"]) == 10
    assert stats["orientations"] == set(_SIGNS) and stats["ring_counts"] == set(range(1, 6))
    assert len(stats["radius_gaps"]) >= 4 and len(stats["layouts"]) >= 200
    assert 1 in stats["components"] and max(stats["components"]) >= 8
    assert len(stats["key_positions"]) >= 3 and inner_key >= 1
    assert 0 in stats["visibility"] and max(stats["visibility"]) >= 4
    assert mixed_phase >= 30 and black_payload >= 15 and min(shifts.values()) >= 10
    # A hand-built member fixes expected chamber behavior without using _render.
    manual_rings = [(3, 1, 0), (6, 3, 2), (9, 2, 4)]
    manual, _ = _template(25, 0, (-1, 1), manual_rings)
    manual[:2, :2] = ((1, 2), (3, 4))
    expected_chambers = ((1, 2, 4, 3), (2, 1, 4, 3), (2, 1, 3, 4), (1, 2, 3, 4))
    for distance, expected_colors in zip((1, 4, 7, 10), expected_chambers):
        coordinates = ((12 - distance, 12), (12, 12 + distance), (12 + distance, 12), (12, 12 - distance))
        for color, cell in enumerate(coordinates, 1):
            manual[cell] = color
    predicted = solve(manual.astype(np.uint64))
    for distance, expected_colors in zip((1, 4, 7, 10), expected_chambers):
        coordinates = ((12 - distance, 12), (12, 12 + distance), (12 + distance, 12), (12, 12 - distance))
        assert tuple(int(predicted[cell]) for cell in coordinates) == expected_colors
    # Rotating the drawing preserves its rule after keeping the KEY ROW convention.
    # Merely rotating the key would turn row pairings into column pairings.
    for puzzle in puzzles[:3]:
        pair = puzzle["test"][0]
        a, bg, _, legend, _, _, _ = _decode(pair["input"])
        for turns in (1, 2, 3):
            rotated = np.rot90(a, turns).copy()
            expected = np.rot90(pair["output"], turns).copy()
            kr, kc = _key_candidates(rotated, bg)[0]
            rotated[kr:kr + 2, kc:kc + 2] = legend
            expected[kr:kr + 2, kc:kc + 2] = legend
            assert np.array_equal(solve(rotated), expected)
        permutation = np.random.default_rng(718).permutation(10)
        assert np.array_equal(solve(permutation[a]), permutation[pair["output"]])
    # An unequal arm pair is an invalid instruction, not token data.
    damaged = puzzles[0]["train"][0]["input"].copy()
    _, bg, _, _, (sr, sc), rings, _ = _decode(damaged)
    radius, _, _ = next(ring for ring in rings if ring[2] != bg)
    center = len(damaged) // 2
    damaged[center + sr * (radius - 2), center + sc * radius] = bg
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("solver accepted unequal arm samples")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(9000 + train * 10 + test, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject an incorrect output rather than merely measuring diversity.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("validator accepted a corrupt output")
    return {"seeds": 42, "pairs": pair_count, "sizes": sorted(stats["sizes"]), "backgrounds": sorted(stats["backgrounds"]), "ring_counts": sorted(stats["ring_counts"]), "radius_gaps": sorted(stats["radius_gaps"]), "payload_component_sizes": sorted(stats["components"]), "layouts": len(stats["layouts"]), "mixed_lattice_phases": int(mixed_phase), "black_payload_pairs": int(black_payload), "keys_inside_outer_ring": int(inner_key), "shifts": dict(shifts)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
