"""Bottom-anchored shear, then finite bridges and segment-gated diagonal beams.

No parent module, filesystem artifact, RNG state, or training pair is used by solve.
All geometric role inference is performed on the sheared input itself.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations

import numpy as np


def _array(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    return tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes()


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    i = int(np.argmax(counts))
    if int(counts[i]) * 2 <= grid.size:
        raise ValueError("The background must occupy more than half the input")
    return int(colors[i])


def _shear(grid, background):
    h, w = grid.shape
    if h + w - 1 > 30:
        raise ValueError("The widened canvas would exceed 30 columns")
    output = np.full((h, h + w - 1), background, dtype=np.int8)
    for r in range(h):
        offset = h - 1 - r
        output[r, offset:offset + w] = grid[r]
    return output


def _diagonal(a, b):
    return a != b and abs(a[0] - b[0]) == abs(a[1] - b[1])


def _span(a, b):
    a, b = sorted((a, b))
    slope = 1 if b[1] > a[1] else -1
    return tuple((r, a[1] + slope * (r - a[0]))
                 for r in range(a[0], b[0] + 1))


def _beam(point, slope, shape):
    r, c = point
    h, w = shape
    return frozenset((y, c + slope * (y - r)) for y in range(h)
                     if 0 <= c + slope * (y - r) < w)


def _construction(grid, primary, secondary):
    """Check a candidate role assignment and construct its finite/full lines."""
    dots = [tuple(map(int, p)) for p in np.argwhere(grid == primary)]
    seconds = (set() if secondary is None else
               {tuple(map(int, p)) for p in np.argwhere(grid == secondary)})
    edges = [(a, b) for a, b in combinations(dots, 2) if _diagonal(a, b)]
    degree = {p: 0 for p in dots}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    if not edges or any(d > 1 for d in degree.values()):
        raise ValueError("Primary endpoints must have unique diagonal partners")
    spans = tuple(_span(a, b) for a, b in edges)
    if any(len(s) < 3 for s in spans):
        raise ValueError("A primary segment needs at least three cells")
    union = set().union(*map(set, spans))
    slopes = tuple(1 if s[-1][1] > s[0][1] else -1 for s in spans)
    active = seconds & union
    lines = []
    for point in sorted(active):
        owners = [i for i, s in enumerate(spans) if point in s]
        if len(owners) != 1:
            raise ValueError("A trigger must belong to exactly one primary segment")
        line = _beam(point, -slopes[owners[0]], grid.shape)
        if line & union != {point} or line & set(dots):
            raise ValueError("A beam cannot overwrite another primary construction")
        lines.append(line)
    result = grid.copy()
    for span in spans:
        for point in span:
            result[point] = primary
    for line in lines:
        for point in line:
            result[point] = secondary
    endpoints = {p for e in edges for p in e}
    return result, {
        "primary": primary, "secondary": secondary, "edges": tuple(edges),
        "spans": spans, "slopes": slopes, "union": union,
        "active": active, "inactive": seconds - active,
        "unpaired": set(dots) - endpoints, "lines": tuple(lines),
        "triggers": tuple(sum(p in active for p in s) for s in spans),
    }


def _interpret(input_grid):
    grid = _array(input_grid)
    if grid.shape[0] < 3:
        raise ValueError("This marker family requires at least three rows")
    background = _background(grid)
    canvas = _shear(grid, background)
    foreground = sorted(set(map(int, np.unique(grid))) - {background})
    if not 1 <= len(foreground) <= 2:
        raise ValueError("Expected one or two foreground marker colors")
    candidates = []
    for primary in foreground:
        secondary = next((c for c in foreground if c != primary), None)
        try:
            result, analysis = _construction(canvas, primary, secondary)
        except ValueError:
            continue
        candidates.append((result, analysis))
    if len(candidates) != 1:
        raise ValueError("The input must determine one admissible marker-role assignment")
    result, analysis = candidates[0]
    analysis["background"] = background
    analysis["canvas"] = canvas
    return result, analysis


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Shear complete rows, infer the unique roles, and draw in new coordinates."""
    return _interpret(input_grid)[0]


def _shape(rng, band, specifications):
    # Direct source-coordinate geometry: -1 after shear means a source column;
    # +1 after shear means two source columns right per descending row.
    minimum_h = max(floor + 3 for _, floor, _ in specifications)
    minimum_w = max((2 * (floor + 2) + 1 if slope == 1 else 1)
                    for slope, floor, _ in specifications)
    pool = [(h, w) for h in range(max(3, minimum_h), 31)
            for w in range(minimum_w, 32 - h)]
    if band == "compact":
        preferred = [(h, w) for h, w in pool if h <= 12 and w <= 18]
    elif band == "tall":
        preferred = [(h, w) for h, w in pool if h >= 14 and w <= 10]
    elif band == "wide":
        preferred = [(h, w) for h, w in pool if h <= 10 and w >= 19]
    elif band == "boundary":
        preferred = [(h, w) for h, w in pool if h + w == 31]
    else:
        preferred = pool
    # Split profiles are sampling choices; solve accepts the full valid domain.
    choices = preferred if preferred else pool
    return choices[int(rng.integers(len(choices)))]


def _place(rng, shape, specifications):
    h, w = shape
    spans, endpoints, active = [], [], []
    primary_union, secondary_union = set(), set()
    for slope, floor, zero in specifications:
        maximum = h - 1 if slope == -1 else min(h - 1, (w - 1) // 2)
        minimum = max(2, floor + 2)
        if minimum > maximum:
            return None
        for _ in range(90):
            # Short, uniform, and nearly full-length spans all have support.
            roll = rng.random()
            if roll < .3:
                steps = int(rng.integers(minimum, min(maximum, minimum + 2) + 1))
            elif roll < .6:
                steps = int(rng.integers(max(minimum, maximum - 2), maximum + 1))
            else:
                steps = int(rng.integers(minimum, maximum + 1))
            r = int(rng.integers(h - steps))
            source_c = int(rng.integers(w if slope == -1 else w - 2 * steps))
            c = source_c + h - 1 - r
            span = _span((r, c), (r + steps, c + slope * steps))
            ends = (span[0], span[-1])
            if any(_diagonal(p, q) or p == q for p in ends for q in endpoints):
                continue
            cells = set(span)
            if cells & secondary_union:
                continue
            combined = cells | primary_union
            eligible = [(p, _beam(p, -slope, (h, h + w - 1)))
                        for p in span[1:-1] if p not in primary_union]
            eligible = [(p, line) for p, line in eligible if line & combined == {p}]
            cap = min(4, len(eligible), len(span) - 3)
            if cap < floor:
                continue
            if zero or cap == 0:
                count = 0
            elif floor:
                count = int(rng.integers(floor, cap + 1))
            else:
                count = int(rng.integers(1, cap + 1)) if rng.random() < .62 else 0
            chosen = [eligible[int(i)] for i in rng.permutation(len(eligible))[:count]]
            spans.append(span)
            endpoints.extend(ends)
            primary_union.update(cells)
            for p, line in chosen:
                active.append(p)
                secondary_union.update(line)
            break
        else:
            return None
    return spans, endpoints, active, primary_union, secondary_union


def _dots(rng, shape, geometry, need_primary, need_inert):
    h, w = shape
    spans, endpoints, active, primary_union, secondary_union = geometry
    domain = [(r, c + h - 1 - r) for r in range(h) for c in range(w)]
    primary = list(endpoints)
    extra_primary = max(int(rng.choice((0, 0, 0, 1, 1, 2, 3))), int(need_primary))
    for i in rng.permutation(len(domain)):
        if len(primary) == len(endpoints) + extra_primary:
            break
        p = domain[int(i)]
        if (p in primary_union or p in secondary_union
                or any(_diagonal(p, q) or p == q for q in primary)):
            continue
        primary.append(p)
    if len(primary) < len(endpoints) + extra_primary:
        return None
    extra_secondary = max(int(rng.choice((0, 0, 1, 1, 2, 3, 5))), int(need_inert))
    pool = [p for p in domain if p not in primary_union and p not in primary]
    if len(pool) < extra_secondary:
        return None
    inert = []
    if need_inert:
        outside = [p for p in pool if p not in secondary_union]
        if not outside:
            return None
        # Prefer a collinear dot beyond an endpoint: full-primary extension
        # would wrongly activate it, so finite segment membership matters.
        beyond = [p for p in outside if any(_diagonal(p, s[0]) and
                  _diagonal(p, s[-1]) for s in spans)]
        candidates = beyond if beyond and rng.random() < .7 else outside
        inert.append(candidates[int(rng.integers(len(candidates)))])
    for i in rng.permutation(len(pool)):
        if len(inert) == extra_secondary:
            break
        p = pool[int(i)]
        if p not in inert:
            inert.append(p)
    return primary, active + inert


def _make_pair(rng, colors, band, witness, extra_count, seen, excluded_shapes=()):
    background, primary, secondary = colors
    specifications = []
    if "slope" in witness:
        specifications.append((witness["slope"], 3 if witness.get("multi") else 1, False))
    if witness.get("zero"):
        specifications.append((int(rng.choice((-1, 1))), 0, True))
    count = max(len(specifications), extra_count)
    parallel = int(rng.choice((-1, 1))) if rng.random() < .35 else None
    while len(specifications) < count:
        specifications.append((parallel or int(rng.choice((-1, 1))), 0, False))
    if not witness and rng.random() < .2:
        specifications = [(slope, 0, True) for slope, _, _ in specifications]
    # Each loop discards its arrays/sets; no retry artifacts are retained.
    for _ in range(180):
        shape = _shape(rng, band, specifications)
        if shape in excluded_shapes:
            continue
        order = rng.permutation(len(specifications))
        geometry = _place(rng, shape, [specifications[int(i)] for i in order])
        if geometry is None:
            continue
        dots = _dots(rng, shape, geometry, witness.get("primary", False),
                     witness.get("inert", False))
        if dots is None or 2 * sum(map(len, dots)) >= np.prod(shape):
            continue
        h, w = shape
        grid = np.full(shape, background, dtype=np.int8)
        for color, points in ((primary, dots[0]), (secondary, dots[1])):
            for r, c in points:
                grid[r, c - (h - 1 - r)] = color
        if _key(grid) in seen:
            continue
        try:
            output, a = _interpret(grid)
        except ValueError:
            continue
        if a["primary"] != primary:
            continue
        if witness.get("multi") and not _evidence(grid, output, a)["padding_beam"]:
            continue
        if witness.get("zero") and not _evidence(grid, output, a)["finite_gap"]:
            continue
        # The intended placement and all visible, uniquely recovered edges agree.
        intended = {frozenset((s[0], s[-1])) for s in geometry[0]}
        if {frozenset(e) for e in a["edges"]} != intended:
            continue
        seen.add(_key(grid))
        return {"input": grid, "output": output}
    raise RuntimeError("Bounded free placement failed; no rule-changing fallback is used")


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    background, primary, secondary = map(int, rng.choice(10, 3, replace=False))
    # The test changes one geometry distribution; no new semantic convention.
    shift = str(rng.choice(("tall", "wide", "boundary")))
    witnesses = [{} for _ in range(num_train)]
    slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    for slot, slope in zip(slots, rng.permutation((-1, 1))):
        witnesses[slot]["slope"] = int(slope)
    # A triple of triggers rules out reversing the foreground colors: its
    # diagonal graph has degree two, whereas primary endpoints form a matching.
    minus_slot = next(slot for slot in slots if witnesses[slot]["slope"] == -1)
    witnesses[minus_slot]["multi"] = True
    for name in ("zero", "primary", "inert"):
        witnesses[int(rng.integers(num_train))][name] = True
    seen = set()
    puzzle = {"train": [], "test": []}
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = witnesses[i] if split == "train" else {}
            extra_count = int(rng.choice((1, 1, 1, 2, 2, 3, 4)))
            band = "compact"
            if split == "test":
                band = shift
            excluded_shapes = (puzzle["train"][0]["input"].shape,) if split == "train" and i == 1 else ()
            pair = _make_pair(rng, (background, primary, secondary), band,
                              witness, extra_count, seen, excluded_shapes)
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _wrong_order(grid, primary, secondary, background):
    """Explicit counterfactual: draw every raw diagonal pair before shearing."""
    result = grid.copy()
    points = [tuple(map(int, p)) for p in np.argwhere(grid == primary)]
    for a, b in combinations(points, 2):
        if not _diagonal(a, b):
            continue
        span = _span(a, b)
        slope = 1 if span[-1][1] > span[0][1] else -1
        for p in span:
            result[p] = primary
        if secondary is not None:
            for p in span:
                if grid[p] == secondary:
                    for q in _beam(p, -slope, grid.shape):
                        result[q] = secondary
    return _shear(result, background)


def _evidence(grid, output, a):
    canvas = a["canvas"]
    line_union = set().union(*a["lines"]) if a["lines"] else set()
    untriggered = [s for s, n in zip(a["spans"], a["triggers"]) if n == 0]
    return {
        "active_slopes": {slope for slope, n in zip(a["slopes"], a["triggers"]) if n},
        "multi": max(a["triggers"]) >= 2,
        "zero": bool(untriggered),
        "primary": bool(a["unpaired"]),
        "inert": bool(a["inactive"] - line_union),
        "new_primary": bool(np.any((output == a["primary"]) & (canvas != a["primary"]))),
        "new_secondary": (a["secondary"] is not None and
                          bool(np.any((output == a["secondary"]) & (canvas != a["secondary"])))),
        "wrong_order_differs": not np.array_equal(output, _wrong_order(
            grid, a["primary"], a["secondary"], a["background"])),
        "beyond": any(_diagonal(p, s[0]) and _diagonal(p, s[-1])
                      for p in a["inactive"] - line_union for s in a["spans"]),
        "finite_gap": any(output[p] == a["background"] for s, slope in
                          zip(a["spans"], a["slopes"]) for p in
                          _beam(s[0], slope, output.shape) - set(s)),
        "padding_beam": any(not (grid.shape[0] - 1 - r <= c <
                            grid.shape[0] - 1 - r + grid.shape[1]) for r, c in line_union),
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, teaching, train_shapes = set(), set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _array(pair["input"]), _array(pair["output"])
            expected, a = _interpret(grid)
            assert np.array_equal(expected, output), "Incorrect output"
            h, w = grid.shape
            assert output.shape == (h, h + w - 1)
            assert _key(grid) not in seen, "Duplicate input"
            seen.add(_key(grid))
            roles.add((a["background"], a["primary"]))
            # Every original foreground marker survives at its sheared address.
            for r, c in np.argwhere(grid != a["background"]):
                assert output[r, c + h - 1 - r] == grid[r, c]
            # No off-construction pixel changes, including the padding triangles.
            painted = a["union"] | (set().union(*a["lines"]) if a["lines"] else set())
            for r in range(h):
                for c in range(h + w - 1):
                    if (r, c) not in painted:
                        assert output[r, c] == a["canvas"][r, c]
            # The host verifier additionally compares complete pairs with both
            # original and selected parent puzzles to reject exact copies.
            if split == "train":
                teaching.append(_evidence(grid, output, a))
                train_shapes.add(grid.shape)
    assert len(roles) == 1, "Background and primary roles stay fixed within a puzzle"
    seconds = {int(c) for split in puzzle for pair in puzzle[split]
               for c in np.unique(pair["input"])} - set(next(iter(roles)))
    assert len(seconds) == 1, "One reserved secondary color per puzzle"
    assert set().union(*(e["active_slopes"] for e in teaching)) == {-1, 1}
    for field in ("multi", "zero", "primary", "inert", "new_primary", "new_secondary",
                  "wrong_order_differs", "padding_beam", "finite_gap"):
        assert any(e[field] for e in teaching), f"Missing teaching evidence: {field}"
    assert len(train_shapes) >= 2, "Demonstrate dimension-dependent registration"
    train_grids = [np.asarray(p["input"]) for p in puzzle["train"]]
    tests = [np.asarray(p["input"]) for p in puzzle["test"]]
    # All queries extrapolate at least one taught geometric measurement.
    max_h = max(g.shape[0] for g in train_grids)
    max_w = max(g.shape[1] for g in train_grids)
    max_out = max(sum(g.shape) - 1 for g in train_grids)
    assert all(g.shape[0] > max_h or g.shape[1] > max_w or
               sum(g.shape) - 1 > max_out for g in tests)


def measure_puzzle(puzzle):
    """Small, reconstructible metrics; no duplicate grid payloads."""
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            grid, output = np.asarray(pair["input"]), np.asarray(pair["output"])
            _, a = _interpret(grid)
            e = _evidence(grid, output, a)
            records.append({
                "split": split, "shape": list(map(int, grid.shape)),
                "roles": [a["background"], a["primary"], a["secondary"]],
                "lengths": list(map(len, a["spans"])), "slopes": list(a["slopes"]),
                "triggers": list(a["triggers"]), "primary_distractors": len(a["unpaired"]),
                "secondary_distractors": len(a["inactive"]),
                **{k: v for k, v in e.items() if k != "active_slopes"},
            })
    return records


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                grid = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds, count structural diversity, and test semantic mutations."""
    # Hand-derived digit grids pin coordinate registration and both beam slopes.
    vertical = np.zeros((8, 6), dtype=np.int8)
    vertical[0, 1] = vertical[5, 1] = 1
    vertical[[1, 3, 4, 7], 1] = 2
    expected_vertical = ("2020002010000", "0202000200000", "0020201020000",
                         "0002020002000", "0000202000200", "0001020200020",
                         "0000002020002", "0200000202000")
    oblique = np.zeros((7, 10), dtype=np.int8)
    oblique[1, 0] = oblique[5, 8] = 1
    oblique[3, 4] = 2
    expected_oblique = ("0000000000200000", "0000010002000000", "0000001020000000",
                        "0000000200000000", "0000002010000000", "0000020001000000",
                        "0000200000000000")
    for grid, rows in ((vertical, expected_vertical), (oblique, expected_oblique)):
        expected = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
        assert np.array_equal(solve(grid), expected)
        # Numeric color ordering and frequency do not determine marker roles.
        permutation = np.array([8, 7, 0, 1, 2, 3, 4, 5, 6, 9], dtype=np.int8)
        assert np.array_equal(solve(permutation[grid]), permutation[expected])
    # An inert dot already on a beam cannot launch any additional construction.
    through_inert = vertical.copy()
    through_inert[4, 3] = 2
    assert np.array_equal(solve(through_inert), solve(vertical))
    boundary = np.full((30, 1), 8, dtype=np.int8)
    boundary[[0, 29], 0] = 4
    boundary[[7, 11, 18], 0] = 7
    assert solve(boundary).shape == (30, 30)
    # A primary crossing has no conflict until it contains a secondary trigger.
    crossing = np.zeros((10, 17), dtype=np.int8)
    for point in ((1, 8), (8, 8), (1, 0), (7, 12)):
        crossing[point] = 1
    crossing_output = solve(crossing)
    assert np.count_nonzero(crossing_output == 1) == 14
    crossing[5, 8] = 2
    conflict = np.zeros((10, 17), dtype=np.int8)
    for point in ((1, 3), (8, 3), (1, 7), (8, 7)):
        conflict[point] = 1
    conflict[3, 3] = 2
    for invalid in (crossing, conflict):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Ambiguous or conflicting construction accepted")
    # Secondary absent: the one foreground color is the endpoint role.
    no_secondary = vertical.copy()
    no_secondary[no_secondary == 2] = 0
    empty_output, empty_a = _interpret(no_secondary)
    assert not empty_a["lines"] and len(empty_a["spans"]) == 1
    assert sum(empty_output.flat == 1) == 6
    # Reversing roles is genuinely ambiguous with just two collinear secondary
    # dots inside a primary span and no other witnesses; such inputs are excluded.
    ambiguous = np.zeros((8, 6), dtype=np.int8)
    ambiguous[0, 1] = ambiguous[7, 1] = 1
    ambiguous[2, 1] = ambiguous[5, 1] = 2
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous role assignment accepted")
    signatures, shapes, palettes, structures = set(), set(), set(), set()
    lengths, counts, trigger_counts, slopes = set(), set(), set(), set()
    flags = {name: 0 for name in ("beyond", "padding_beam", "multi", "zero")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for record in measure_puzzle(puzzle):
            shapes.add(tuple(record["shape"]))
            palettes.add(tuple(record["roles"]))
            lengths.update(record["lengths"])
            counts.add(len(record["lengths"]))
            trigger_counts.update(record["triggers"])
            slopes.update(record["slopes"])
            for name in flags:
                flags[name] += bool(record[name])
        for split in puzzle:
            for pair in puzzle[split]:
                grid = pair["input"]
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in grid.flat)
                structures.add((tuple(grid.shape), normalized))
                before = grid.copy()
                assert np.array_equal(solve(grid.copy()), pair["output"])
                assert np.array_equal(grid, before)
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(palettes) >= 25 and len(structures) >= 190
    assert {1, 2, 3, 4} <= counts and {0, 1, 2, 3} <= trigger_counts
    assert slopes == {-1, 1} and min(lengths) == 3 and max(lengths) >= 15
    assert all(flags.values())
    assert _digest(generate_puzzle(17)) == _digest(generate_puzzle(17))
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            generate_puzzle(1000 + n * 10 + m, n, m)
    for n, m in ((1, 1), (11, 1), (4, 0), (5, 5), (6, 5), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    puzzle = generate_puzzle(0)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "role_assignments": len(palettes), "normalized_inputs": len(structures),
            "segment_counts": sorted(counts), "segment_lengths": sorted(lengths),
            "trigger_counts": sorted(trigger_counts), "witness_pairs": flags}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
