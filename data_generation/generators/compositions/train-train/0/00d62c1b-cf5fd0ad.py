"""Seal the wallpaper: bounded voids in an infinitely repeated wall tile.

Only Python's standard library and NumPy are used. Zero is air, four is the
answer stain, and one other input color is wall. See task_spec.md for the rule.
"""
from collections import deque

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
WALL_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a


def _topology(free, neighbors=N4):
    """Label the quotient graph, remembering *unwrapped* cell coordinates.

    Two paths reaching the same residue with different lifts give a nonzero
    period translation. Repeating that walk proves the entire component is
    unbounded. Conversely consistent lifts provide a finite, explicit copy of
    that component in the plane. No finite wallpaper window is assumed.
    """
    h, w = free.shape
    seen = np.zeros(free.shape, dtype=bool)
    bounded = np.zeros(free.shape, dtype=bool)
    components = []
    for row, col in np.argwhere(free):
        start = (int(row), int(col))
        if seen[start]:
            continue
        lifts = {start: start}
        seen[start] = True
        stack = [start]
        cells, translations = [], set()
        while stack:
            p = stack.pop()
            cells.append(p)
            y, x = lifts[p]
            for dy, dx in neighbors:
                q = ((p[0] + dy) % h, (p[1] + dx) % w)
                if not free[q]:
                    continue
                expected = (y + dy, x + dx)
                if q in lifts:
                    previous = lifts[q]
                    delta = (expected[0] - previous[0], expected[1] - previous[1])
                    if delta != (0, 0):
                        translations.add((delta[0] // h, delta[1] // w))
                else:
                    lifts[q] = expected
                    seen[q] = True
                    stack.append(q)
        finite = not translations
        if finite:
            for p in cells:
                bounded[p] = True
        ys = [v[0] for v in lifts.values()]
        xs = [v[1] for v in lifts.values()]
        components.append({"cells": cells, "finite": finite,
                           "span": (max(ys) - min(ys) + 1, max(xs) - min(xs) + 1),
                           "translations": translations})
    return bounded, components


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill exactly the finite air components of the infinite periodic lift."""
    a = _grid(input_grid)
    colors = set(int(c) for c in np.unique(a))
    if 4 in colors or len(colors - {0}) > 1:
        raise ValueError("Input uses zero air and at most one non-four wall color")
    fill, _ = _topology(a == 0)
    answer = a.copy()
    answer[fill] = 4
    return answer


def _finite_canvas_fill(free):
    """Deliberately wrong parent-A-only boundary convention, for ablations."""
    h, w = free.shape
    reached = np.zeros(free.shape, dtype=bool)
    stack = []
    for r, c in np.argwhere(free):
        r, c = int(r), int(c)
        if r in (0, h - 1) or c in (0, w - 1):
            reached[r, c] = True
            stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and free[y, x] and not reached[y, x]:
                reached[y, x] = True
                stack.append((y, x))
    return free & ~reached


def _features(wall):
    h, w = wall.shape
    fill, comps = _topology(~wall)
    fill8, _ = _topology(~wall, N8)
    sx = sy = corner = inside = False
    finite_areas, spans = [], []
    for comp in comps:
        if not comp["finite"]:
            continue
        cells = set(comp["cells"])
        x = any(c == 0 and (r, w - 1) in cells for r, c in cells)
        y = any(r == 0 and (h - 1, c) in cells for r, c in cells)
        sx |= x
        sy |= y
        corner |= x and y
        inside |= all(r not in (0, h - 1) and c not in (0, w - 1) for r, c in cells)
        finite_areas.append(len(cells))
        spans.append(comp["span"])
    return {"fill": fill, "components": comps, "seam_x": sx, "seam_y": sy,
            "corner": corner, "inside": inside, "diagonal": bool(np.any(fill & ~fill8)),
            "finite_areas": tuple(sorted(finite_areas)), "spans": tuple(spans),
            "unbounded": int(np.count_nonzero(~wall & ~fill)),
            "finite_boundary_error": int(np.count_nonzero(fill != _finite_canvas_fill(~wall)))}


def _route(rng, h, w, kind):
    """Protected air with an explicit nonzero winding, never an output marker."""
    if kind == "horizontal":
        y = int(rng.integers(h))
        return {(y, c) for c in range(w)}
    if kind == "vertical":
        x = int(rng.integers(w))
        return {(r, x) for r in range(h)}
    # A closed monotone staircase from (y,x) to (y+h,x+w), modulo the tile.
    steps = [(1, 0)] * h + [(0, 1)] * w
    rng.shuffle(steps)
    y, x = int(rng.integers(h)), int(rng.integers(w))
    cells = {(y, x)}
    for dy, dx in steps:
        y, x = (y + dy) % h, (x + dx) % w
        cells.add((y, x))
    return cells


def _shape(rng, h, w, mode, large=False):
    max_h, max_w = min(15, h - 3), min(15, w - 3)
    if mode == "dot":
        return {(0, 0)}
    low = 3 if large else 1
    a = int(rng.integers(low, max(low, max_h) + 1))
    b = int(rng.integers(low, max(low, max_w) + 1))
    if mode == "line":
        a = 1
        b = max(2, b)
    cells = {(r, c) for r in range(a) for c in range(b)}
    if mode == "bend":
        thickness = int(rng.integers(1, min(3, a, b) + 1))
        cells = {(r, c) for r in range(a) for c in range(b)
                 if c < thickness or r >= a - thickness}
    elif mode == "notch" and a > 1 and b > 1:
        nr = int(rng.integers(1, a))
        nc = int(rng.integers(1, b))
        cells -= {(r, c) for r in range(nr) for c in range(nc)}
    elif mode == "island" and a >= 4 and b >= 4:
        ir, ic = int(rng.integers(1, a - 1)), int(rng.integers(1, b - 1))
        cells.remove((ir, ic))
    if rng.random() < 0.5:
        cells = {(a - 1 - r, c) for r, c in cells}
    if rng.random() < 0.5:
        cells = {(r, b - 1 - c) for r, c in cells}
    return cells


def _room(rng, wall, protected, interiors, mode, seam=None, large=False):
    h, w = wall.shape
    for _ in range(75):
        shape = _shape(rng, h, w, mode, large)
        rh, rw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
        if seam in ("x", "both") and rw < 2:
            continue
        if seam in ("y", "both") and rh < 2:
            continue
        y = (h - int(rng.integers(1, rh)) if seam in ("y", "both")
             else int(rng.integers(h)))
        x = (w - int(rng.integers(1, rw)) if seam in ("x", "both")
             else int(rng.integers(w)))
        inside = {((y + r) % h, (x + c) % w) for r, c in shape}
        if inside & (protected | interiors):
            continue
        neighbors = N8 if rng.random() < 0.45 else N4
        border = {((r + dy) % h, (c + dx) % w)
                  for r, c in inside for dy, dx in neighbors} - inside
        if border & (protected | interiors):
            continue
        # Wall reuse is allowed; distinct chamber interiors never overwrite it.
        # Dense solids can be carved; outline scenes must respect old walls.
        for p in inside:
            wall[p] = False
        for p in border:
            wall[p] = True
        interiors.update(inside)
        return True
    return False


def _diagonal_eyelet(rng, wall, protected, interiors):
    h, w = wall.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            p = (r, c)
            orth = {((r + dy) % h, (c + dx) % w) for dy, dx in N4}
            diag = {((r + dy) % h, (c + dx) % w) for dy, dx in N8[4:]}
            if p not in protected and not (orth & protected) and diag & protected:
                candidates.append(p)
    if not candidates:
        return False
    p = candidates[int(rng.integers(len(candidates)))]
    wall[p] = False
    for dy, dx in N4:
        wall[(p[0] + dy) % h, (p[1] + dx) % w] = True
    interiors.add(p)
    return True


def _ordinary(rng, h, w, seam, diagonal, dense, large=False):
    path = _route(rng, h, w, "diagonal" if diagonal else
                  str(rng.choice(("horizontal", "vertical", "diagonal"))))
    wall = np.full((h, w), dense, dtype=bool)
    for p in path:
        wall[p] = False
    interiors = set()
    if diagonal and not _diagonal_eyelet(rng, wall, path, interiors):
        return None
    first = str(rng.choice(("rect", "bend", "notch", "island")))
    if not _room(rng, wall, path, interiors, first, seam, large):
        return None
    for _ in range(int(rng.integers(2, 8 if large else 6))):
        mode = str(rng.choice(("rect", "line", "bend", "notch", "island", "dot")))
        _room(rng, wall, path, interiors, mode)
    if not dense:
        # Genuine turning branches, allowed to meet other walls or cross seams.
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            d = int(rng.integers(4))
            for _ in range(int(rng.integers(3, max(h, w) + 1))):
                if (r, c) in path or (r, c) in interiors:
                    break
                wall[r, c] = True
                if rng.random() < 0.25:
                    d = int(rng.integers(4))
                dr, dc = N4[d]
                r, c = (r + dr) % h, (c + dc) % w
    return wall


def _helix(rng, h, w, long=False):
    """A capped, chordless air path winds through several *different* copies.

    Parallel runs sit two rows apart. Each run stops before closing a horizontal
    circuit; a short vertical link starts the next run. Its projected graph is
    a path, so it is finite despite a lift wider than several input periods.
    A separate complete air row is a genuinely infinite corridor.
    """
    wall = np.ones((h, w), dtype=bool)
    wall[h - 1, :] = False
    max_runs = (h - 2) // 2
    minimum = min(max_runs, 4 if long else 3)
    runs = int(rng.integers(minimum, max_runs + 1))
    y, x = 1, int(rng.integers(w))
    wall[y, x] = False
    for i in range(runs):
        steps = int(rng.integers(max(2, w - 3), w - 1))
        for _ in range(steps):
            x = (x + 1) % w
            wall[y, x] = False
        if i + 1 < runs:
            for _ in range(2):
                y += 1
                wall[y, x] = False
    # Add independent eyelets only where they cannot touch the main air paths.
    for _ in range(int(rng.integers(0, 5))):
        candidates = []
        for r, c in np.argwhere(wall):
            r, c = int(r), int(c)
            if all(wall[(r + dr) % h, (c + dc) % w] for dr, dc in N4):
                candidates.append((r, c))
        if candidates:
            wall[candidates[int(rng.integers(len(candidates)))]] = False
    return wall


def _transform(rng, wall):
    if rng.random() < 0.5:
        wall = np.fliplr(wall)
    wall = np.rot90(wall, int(rng.integers(4)))
    return np.roll(wall, (int(rng.integers(wall.shape[0])),
                          int(rng.integers(wall.shape[1]))), axis=(0, 1)).copy()


def _sample(rng, mode, shape_exclusion=None):
    for _ in range(240):
        if mode == "anchor":
            h, w = int(rng.integers(9, 17)), int(rng.integers(9, 17))
            wall = _ordinary(rng, h, w, "both", True, True)
        elif mode == "helix":
            h, w = int(rng.integers(10, 20)), int(rng.integers(8, 18))
            wall = _helix(rng, h, w)
        elif mode == "test_helix":
            h, w = int(rng.integers(16, 31)), int(rng.integers(10, 27))
            wall = _helix(rng, h, w, True)
        else:
            is_test = mode == "test_rooms"
            h = int(rng.integers(13 if is_test else 7, 31 if is_test else 25))
            w = int(rng.integers(13 if is_test else 7, 31 if is_test else 25))
            wall = _ordinary(rng, h, w, "both" if is_test else
                             str(rng.choice(("x", "y", "both"))),
                             bool(rng.random() < 0.4), bool(rng.random() < 0.35), is_test)
        if wall is None:
            continue
        wall = _transform(rng, wall)
        if shape_exclusion == wall.shape:
            continue
        f = _features(wall)
        if not f["finite_areas"] or not f["unbounded"]:
            continue
        if mode == "anchor" and not (f["corner"] and f["diagonal"] and f["inside"]):
            continue
        if mode == "anchor" and (np.any(np.all(~wall, axis=0))
                                  or np.any(np.all(~wall, axis=1))):
            continue
        if mode in ("helix", "test_helix"):
            h2, w2 = wall.shape
            if not any(a > 2 * h2 or b > 2 * w2 for a, b in f["spans"]):
                continue
            if not (f["seam_x"] and f["seam_y"]):
                continue
        if mode == "test_rooms" and not f["corner"]:
            continue
        return wall
    raise RuntimeError("Could not sample the requested periodic topology")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    ns = _count(num_test, 1, 4, "num_test")
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    modes = ["anchor", "helix"] + ["rooms"] * (nt - 2)
    rng.shuffle(modes)
    test_modes = [str(rng.choice(("test_rooms", "test_helix"), p=(0.65, 0.35)))
                  for _ in range(ns)]
    puzzle = {"train": [], "test": []}
    seen = set()
    first_shape = None
    shared_wall = int(rng.choice(WALL_COLORS)) if rng.random() < 0.25 else None
    for index, mode in enumerate(modes + test_modes):
        split = "train" if index < nt else "test"
        for _ in range(30):
            wall = _sample(rng, mode, first_shape if index == 1 else None)
            color = shared_wall if shared_wall is not None else int(rng.choice(WALL_COLORS))
            inp = np.where(wall, color, 0).astype(np.int8)
            if _key(inp) in seen:
                continue
            seen.add(_key(inp))
            if first_shape is None:
                first_shape = inp.shape
            puzzle[split].append({"input": inp, "output": solve(inp)})
            break
        else:
            raise RuntimeError("Could not generate distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Schema plus semantic, evidence, seam, and winding invariants."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            palette = set(int(c) for c in np.unique(a))
            assert 0 in palette and len(palette) == 2 and next(iter(palette - {0})) in WALL_COLORS
            assert _key(a) not in seen
            seen.add(_key(a))
            f = _features(a != 0)
            assert f["finite_areas"] and f["unbounded"] > 0
            assert np.array_equal(b != a, f["fill"])
            assert np.all(b[f["fill"]] == 4)
            assert np.array_equal(b[~f["fill"]], a[~f["fill"]])
            assert np.array_equal(b, solve(a))
            # Every winding component remains open, even at an input edge.
            for comp in f["components"]:
                vals = [b[p] for p in comp["cells"]]
                assert all(v == (4 if comp["finite"] else 0) for v in vals)
            if split == "train":
                train_features.append(f)
            else:
                assert f["corner"] and f["finite_boundary_error"] > 0
    assert len({np.asarray(p["input"]).shape for p in train}) >= 2
    assert any(f["corner"] and f["diagonal"] and f["inside"]
               and not np.any(np.all(np.asarray(p["input"]) == 0, axis=0))
               and not np.any(np.all(np.asarray(p["input"]) == 0, axis=1))
               for p, f in zip(train, train_features))
    assert any(any(a > 2 * np.asarray(p["input"]).shape[0]
                       or b > 2 * np.asarray(p["input"]).shape[1] for a, b in f["spans"])
               for p, f in zip(train, train_features))


def _lift_oracle(free):
    """Separate test method: explicitly explore the infinite integer lattice.

    A bounded lift contains at most as many cells as its toroidal component.
    Seeing one more cell proves an infinite lift. No coordinate potentials or
    cycle-translation test is used here.
    """
    h, w = free.shape
    residues_seen = set()
    answer = np.zeros(free.shape, dtype=bool)
    for row, col in np.argwhere(free):
        start = (int(row), int(col))
        if start in residues_seen:
            continue
        component, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in N4:
                p = ((r + dr) % h, (c + dc) % w)
                if free[p] and p not in component:
                    component.add(p)
                    todo.append(p)
        residues_seen.update(component)
        lifted, queue = {start}, deque([start])
        while queue and len(lifted) <= len(component):
            r, c = queue.popleft()
            for dr, dc in N4:
                p = (r + dr, c + dc)
                if free[p[0] % h, p[1] % w] and p not in lifted:
                    lifted.add(p)
                    queue.append(p)
        if len(lifted) <= len(component):
            for p in component:
                answer[p] = True
    return answer


def self_test():
    """Bounded 42-seed audit, exact lift checks, ablations, and interface probes."""
    # A diamond is finite despite its diagonal routes into infinite air.
    diamond = np.zeros((7, 9), dtype=np.int8)
    for dr, dc in N4:
        diamond[3 + dr, 4 + dc] = 3
    expected = diamond.copy()
    expected[3, 4] = 4
    assert np.array_equal(solve(diamond), expected)
    seam = np.roll(diamond, (3, 4), axis=(0, 1))
    assert np.array_equal(solve(seam), np.roll(expected, (3, 4), axis=(0, 1)))
    # Full periodic rows and oblique periodic staircases never become chambers.
    stripes = np.full((5, 7), 6, dtype=np.int8)
    stripes[2, :] = 0
    assert np.array_equal(solve(stripes), stripes)
    staircase = np.full((7, 7), 2, dtype=np.int8)
    for r in range(7):
        staircase[r, r] = staircase[r, (r + 1) % 7] = 0
    assert not np.any(np.all(staircase == 0, axis=0))
    assert not np.any(np.all(staircase == 0, axis=1))
    assert np.array_equal(solve(staircase), staircase)
    # A finite single pixel has no relation to the outer crop's edge.
    eyelet = np.full((4, 5), 8, dtype=np.int8)
    eyelet[0, 0] = 0
    assert solve(eyelet)[0, 0] == 4
    assert solve(np.zeros((1, 1), dtype=int))[0, 0] == 0
    assert solve(np.ones((1, 1), dtype=int))[0, 0] == 1

    shapes, profiles, normalized, wall_colors, signatures = set(), set(), set(), set(), set()
    area_values, spans, finite_counts = set(), set(), set()
    diagonal_pairs = dense_pairs = shared_dividers = 0
    naive_errors = small_window_errors = oblique_components = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(q["input"]) for q in p["train"] + p["test"]))
        for q in p["train"] + p["test"]:
            a, b = q["input"], q["output"]
            wall = a != 0
            f = _features(wall)
            shapes.add(a.shape)
            normalized.add(_key(wall))
            wall_colors.update(int(c) for c in np.unique(a) if c)
            profiles.add((a.shape, f["finite_areas"], f["spans"], f["unbounded"]))
            area_values.update(f["finite_areas"])
            spans.update(f["spans"])
            finite_counts.add(len(f["finite_areas"]))
            diagonal_pairs += f["diagonal"]
            dense_pairs += np.mean(wall) > 0.6
            naive_errors += f["finite_boundary_error"] > 0
            assert np.array_equal(f["fill"], _lift_oracle(~wall))
            assert np.any(a != b)  # Removing A (just retain the tile) is wrong.
            assert np.any(b == 0)  # Filling every void is wrong as well.
            ids = {cell: i for i, comp in enumerate(f["components"])
                   if comp["finite"] for cell in comp["cells"]}
            shared_dividers += any(len({ids[((r + dr) % a.shape[0], (c + dc) % a.shape[1])]
                                       for dr, dc in N4
                                       if ((r + dr) % a.shape[0], (c + dc) % a.shape[1]) in ids}) > 1
                                   for r, c in np.argwhere(wall))
            for comp in f["components"]:
                if any(x and y for x, y in comp["translations"]):
                    oblique_components += 1
            # Wrong alternative: finite 3x3 wallpaper, crop its center answer.
            h, w = a.shape
            window = _finite_canvas_fill(np.tile(a == 0, (3, 3)))[h:2*h, w:2*w]
            small_window_errors += not np.array_equal(window, f["fill"])
            # Rule is independent of the tile's cut, orientation, and wall dye.
            shifted = np.roll(a, (2, -3), axis=(0, 1))
            assert np.array_equal(solve(shifted), np.roll(b, (2, -3), axis=(0, 1)))
            assert np.array_equal(solve(np.rot90(a)), np.rot90(b))
            recolored = np.where(a == 0, 0, WALL_COLORS[(seed + 1) % 8])
            assert np.array_equal(solve(recolored) == 4, b == 4)
    assert len(signatures) == 42 and len(normalized) >= 190
    assert len(shapes) >= 60 and len(profiles) >= 190 and wall_colors == set(WALL_COLORS)
    assert 1 in area_values and max(area_values) >= 70 and len(area_values) >= 25
    assert len(finite_counts) >= 4 and len(spans) >= 40
    assert diagonal_pairs >= 42 and dense_pairs >= 42 and shared_dividers >= 20
    assert naive_errors >= 150 and small_window_errors >= 42 and oblique_components >= 10
    # Exact determinism, including the outputs, at a larger explicit request.
    a, b = generate_puzzle(345678, 10, 4), generate_puzzle(345678, 10, 4)
    assert all(np.array_equal(x[k], y[k]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for k in ("input", "output"))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(17000 + 10 * nt + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    report = {"seeds": 42, "pairs": 210, "distinct_shapes": len(shapes),
            "distinct_wall_patterns": len(normalized), "topology_profiles": len(profiles),
            "bounded_areas": len(area_values), "max_bounded_area": max(area_values),
            "diagonal_pairs": diagonal_pairs, "dense_pairs": dense_pairs,
            "shared_divider_pairs": shared_dividers, "finite_boundary_failures": naive_errors,
            "finite_3x3_window_failures": small_window_errors,
            "oblique_winding_components": oblique_components}
    return {key: int(value) for key, value in report.items()}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
