"""Quarter-turn completion in the ordinal coordinates of an expanding lattice.

Only standard-library modules and NumPy are used. All interpretations in solve
are recovered from the input; generation parameters are never consulted.
"""

from itertools import permutations

import numpy as np


_SIDES = ("top", "right", "bottom", "left")


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    _need(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape),
          "grid dimensions must be 1..30")
    _need(np.issubdtype(grid.dtype, np.integer), "grid must be integral")
    _need(np.all((0 <= grid) & (grid <= 9)), "color outside 0..9")
    return grid.astype(np.int8, copy=False)


def _orbit(n, point):
    r, c = point
    result = set()
    for _ in range(4):
        result.add((r, c))
        r, c = c, n - 1 - r
    return tuple(sorted(result))


def _orbits(n):
    return sorted({_orbit(n, (r, c)) for r in range(n) for c in range(n)})


def _complete(labels, background):
    completed = labels.copy()
    records = []
    for orbit in _orbits(len(labels)):
        visible = [p for p in orbit if labels[p] != background]
        colors = {int(labels[p]) for p in visible}
        _need(len(colors) <= 1, "conflicting colors in a rotational orbit")
        color = next(iter(colors)) if colors else background
        for point in orbit:
            completed[point] = color
        records.append((orbit, tuple(visible), color))
    return completed, records


def _line_mask(shape, rows, cols):
    mask = np.zeros(shape, dtype=bool)
    for r in rows:
        mask[r, cols[0]:cols[-1] + 1] = True
    for c in cols:
        mask[rows[0]:rows[-1] + 1, c] = True
    return mask


def _infer_bounds(markers, old, shape):
    """Jointly assign dots to sides; no nearest-side heuristic is used."""
    top, bottom, left, right = old
    solutions = []
    for assignment in permutations(_SIDES, len(markers)):
        new = list(old)
        valid = True
        for (r, c), side in zip(markers, assignment):
            if side == "top" and r + 1 < top:
                new[0] = r + 1
            elif side == "bottom" and r - 1 > bottom:
                new[1] = r - 1
            elif side == "left" and c + 1 < left:
                new[2] = c + 1
            elif side == "right" and c - 1 > right:
                new[3] = c - 1
            else:
                valid = False
                break
        if not valid:
            continue
        t, b, l, rr = new
        if not (1 <= t <= top < bottom <= b < shape[0] - 1
                and 1 <= l <= left < right <= rr < shape[1] - 1):
            continue
        for (r, c), side in zip(markers, assignment):
            if side in ("top", "bottom"):
                valid &= l < c < rr and r == (t - 1 if side == "top" else b + 1)
            else:
                valid &= t < r < b and c == (l - 1 if side == "left" else rr + 1)
        if valid:
            solutions.append((tuple(new), dict(zip(assignment, markers))))
    _need(len(solutions) == 1, "control-to-side assignment is not unique")
    return solutions[0]


def _parse_structure(grid, background, structure):
    positions = np.argwhere(grid == structure)
    top, left = (int(x) for x in positions.min(axis=0))
    bottom, right = (int(x) for x in positions.max(axis=0))
    _need(0 < top < bottom < grid.shape[0] - 1
          and 0 < left < right < grid.shape[1] - 1,
          "old frame needs exterior background on every side")
    rows = [r for r in range(top, bottom + 1)
            if np.all(grid[r, left:right + 1] == structure)]
    cols = [c for c in range(left, right + 1)
            if np.all(grid[top:bottom + 1, c] == structure)]
    _need(4 <= len(rows) == len(cols) <= 7, "not a square logical lattice")
    _need(rows[0] == top and rows[-1] == bottom
          and cols[0] == left and cols[-1] == right, "incomplete perimeter")
    _need(all(b - a >= 2 for axis in (rows, cols)
              for a, b in zip(axis, axis[1:])), "zero-sized compartment")
    mask = _line_mask(grid.shape, rows, cols)
    _need(np.array_equal(mask, grid == structure), "extra or missing structural cells")
    outside = np.ones(grid.shape, dtype=bool)
    outside[top:bottom + 1, left:right + 1] = False
    locations = np.argwhere(outside & (grid != background))
    _need(1 <= len(locations) <= 4, "expected one to four controls")
    markers = [tuple(int(x) for x in p) for p in locations]
    colors = {int(grid[p]) for p in markers}
    _need(len(colors) == 1, "control dots must share a color")
    marker = colors.pop()
    _need(marker not in (background, structure), "control role collision")
    for r, c in markers:
        _need(not np.any(mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2]),
              "control touches initial structure")
    _need(all(max(abs(r - r2), abs(c - c2)) > 1
              for i, (r, c) in enumerate(markers) for r2, c2 in markers[i + 1:]),
          "controls touch each other")
    n = len(rows) - 1
    labels = np.full((n, n), background, dtype=np.int8)
    for r in range(n):
        for c in range(n):
            region = grid[rows[r] + 1:rows[r + 1], cols[c] + 1:cols[c + 1]]
            values = np.unique(region)
            _need(len(values) == 1 and int(values[0]) not in (structure, marker),
                  "compartment must be uniform background or payload")
            labels[r, c] = values[0]
    completed, records = _complete(labels, background)
    _need(any(0 < len(v) < len(o) for o, v, _ in records), "no damaged orbit")
    final, by_side = _infer_bounds(markers, (top, bottom, left, right), grid.shape)
    final_rows, final_cols = list(rows), list(cols)
    final_rows[0], final_rows[-1] = final[0], final[1]
    final_cols[0], final_cols[-1] = final[2], final[3]
    return dict(background=background, structure=structure, marker=marker,
                rows=rows, cols=cols, final_rows=final_rows, final_cols=final_cols,
                old=(top, bottom, left, right), final=final, by_side=by_side,
                markers=markers, labels=labels, completed=completed, records=records,
                order=n)


def _analyze_input(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    _need(all(int(grid[p]) == background for p in
              ((0, 0), (0, -1), (-1, 0), (-1, -1))), "corners must be background")
    interpretations = []
    for structure in sorted(int(c) for c in np.unique(grid) if c != background):
        try:
            interpretations.append(_parse_structure(grid, background, structure))
        except AssertionError:
            pass
    _need(len(interpretations) == 1, "frame and semantic roles are ambiguous")
    return grid, interpretations[0]


def _render(shape, rows, cols, labels, background, structure, markers, marker):
    result = np.full(shape, background, dtype=np.int8)
    for r in range(len(labels)):
        for c in range(len(labels)):
            result[rows[r] + 1:rows[r + 1], cols[c] + 1:cols[c + 1]] = labels[r, c]
    result[_line_mask(shape, rows, cols)] = structure
    for point in markers:
        result[point] = marker
    return result


def _render_info(shape, info, *, rotate=True, expand=True):
    return _render(shape,
                   info["final_rows"] if expand else info["rows"],
                   info["final_cols"] if expand else info["cols"],
                   info["completed"] if rotate else info["labels"],
                   info["background"], info["structure"],
                   info["markers"], info["marker"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, info = _analyze_input(input_grid)
    return _render_info(grid.shape, info)


def _area_witness(info):
    rows, cols = info["final_rows"], info["final_cols"]
    area = lambda p: (rows[p[0] + 1] - rows[p[0]] - 1) * (cols[p[1] + 1] - cols[p[1]] - 1)
    return any(area(a) != area(b) for orbit, visible, _ in info["records"]
               for a in visible for b in orbit if b not in visible)


def _reflection_witness(info):
    n, bg = info["order"], info["background"]
    return any(0 < len(visible) < len(orbit)
               and any(info["completed"][r, n - 1 - c] == bg for r, c in orbit)
               for orbit, visible, _ in info["records"])


def _edge_points(n, orbit, sides):
    return [p for p in orbit if any(
        p[0] == 0 if side == "top" else
        p[0] == n - 1 if side == "bottom" else
        p[1] == 0 if side == "left" else p[1] == n - 1
        for side in sides)]


def _interaction_witness(info):
    # These newly inferred regions grow into the band exposed by B. The same
    # painted cells therefore depend on both mechanisms, not just separate edits.
    return any(p not in visible
               for orbit, visible, _ in info["records"] if visible
               for p in _edge_points(info["order"], orbit, info["by_side"]))


def _delays(info):
    t, b, l, r = info["old"]
    return sum(not (t <= p[0] <= b if side in ("left", "right") else l <= p[1] <= r)
               for side, p in info["by_side"].items())


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _make_labels(rng, n, keep, allowed, palette, offaxis, sides):
    bg, _, _, *payload = palette
    orbits = [o for o in _orbits(n) if len(o) == 4]
    edge_orbits = [o for o in orbits if _edge_points(n, o, sides)]
    main = (_orbit(n, (0, 1)) if offaxis
            else edge_orbits[int(rng.integers(len(edge_orbits)))])
    forbidden = {_orbit(n, (0, n - 2))} if offaxis else set()
    available = [o for o in orbits if o != main and o not in forbidden]
    count = int(rng.integers(1, min(3, len(available) + 1, len(orbits) - 1) + 1))
    rng.shuffle(available)
    chosen = [main, *available[:count - 1]]
    complete = np.full((n, n), bg, dtype=np.int8)
    visible = complete.copy()
    for index, orbit in enumerate(chosen):
        color = int(payload[int(rng.integers(len(payload)))])
        survivors = keep if index == 0 else (4 if rng.random() < 0.35 else int(rng.choice(allowed)))
        if index == 0:
            edge = _edge_points(n, orbit, sides)
            missing = edge[int(rng.integers(len(edge)))]
            choices = [j for j, p in enumerate(orbit) if p != missing]
        else:
            choices = list(range(4))
        retained = set(int(x) for x in rng.choice(choices, size=survivors, replace=False))
        for j, point in enumerate(orbit):
            complete[point] = color
            if j in retained:
                visible[point] = color
    if n % 2 and rng.random() < 0.4:
        color = int(rng.choice(payload))
        visible[n // 2, n // 2] = complete[n // 2, n // 2] = color
    return visible, complete


def _make_pair(rng, n, keep, allowed, palette, sides, delay, offaxis, seen):
    bg, structure, marker, *_ = palette
    for _ in range(500):
        heights = [int(x) for x in rng.integers(1, 5 if n <= 4 else 4, size=n)]
        widths = [int(x) for x in rng.integers(1, 5 if n <= 4 else 4, size=n)]
        moves = {s: int(rng.integers(1, 5)) if s in sides else 0 for s in _SIDES}
        dependencies = {}
        adjacent = [(a, b) for a in sides for b in sides
                    if (a in ("top", "bottom")) != (b in ("top", "bottom"))]
        if delay:
            if not adjacent:
                continue
            a, b = adjacent[int(rng.integers(len(adjacent)))]
            dependencies[a] = b
            moves[b] = max(2, moves[b])
            if rng.random() < 0.4:
                dependencies[b] = a
                moves[a] = max(2, moves[a])
        margins = [int(x) for x in rng.integers(1, 4, size=4)]
        ft, fl = margins[0], margins[3]
        top, left = ft + moves["top"], fl + moves["left"]
        rows = [top]
        cols = [left]
        for size in heights:
            rows.append(rows[-1] + size + 1)
        for size in widths:
            cols.append(cols[-1] + size + 1)
        fb, fr = rows[-1] + moves["bottom"], cols[-1] + moves["right"]
        shape = (fb + margins[2] + 1, fr + margins[1] + 1)
        if max(shape) > 30:
            continue
        markers = []
        for side in sides:
            dep = dependencies.get(side)
            if side in ("top", "bottom"):
                low, high = left + 1, cols[-1]
                if dep == "left":
                    low, high = fl + 1, left
                elif dep == "right":
                    low, high = cols[-1] + 1, fr
                coordinate = int(rng.integers(low, high))
                point = (ft - 1 if side == "top" else fb + 1, coordinate)
            else:
                low, high = top + 1, rows[-1]
                if dep == "top":
                    low, high = ft + 1, top
                elif dep == "bottom":
                    low, high = rows[-1] + 1, fb
                coordinate = int(rng.integers(low, high))
                point = (coordinate, fl - 1 if side == "left" else fr + 1)
            markers.append(point)
        labels, complete = _make_labels(rng, n, keep, allowed, palette, offaxis, sides)
        inp = _render(shape, rows, cols, labels, bg, structure, markers, marker)
        final_rows, final_cols = rows.copy(), cols.copy()
        final_rows[0], final_rows[-1] = ft, fb
        final_cols[0], final_cols[-1] = fl, fr
        out = _render(shape, final_rows, final_cols, complete, bg, structure, markers, marker)
        try:
            _, info = _analyze_input(inp)
            if not _area_witness(info) or (offaxis and not _reflection_witness(info)):
                continue
            if delay and not _delays(info):
                continue
            _need(np.array_equal(out, _render_info(shape, info)), "construction/parse disagreement")
        except AssertionError:
            continue
        key = _key(inp)
        if key in seen:
            continue
        seen.add(key)
        return {"input": inp, "output": out}
    raise RuntimeError("bounded sampler could not build a valid lattice")


def _palette(rng):
    return tuple(int(x) for x in rng.choice(10, size=int(rng.integers(4, 7)), replace=False))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    held = int(rng.integers(1, 4))
    allowed = [x for x in (1, 2, 3) if x != held]
    rng.shuffle(allowed)
    switch_palette = bool(rng.integers(2))
    base_palette = _palette(rng)
    delayed = bool(rng.random() < 0.65)
    first_n = int(rng.integers(3, 5))
    second_n = int(rng.choice([n for n in (4, 5, 6) if n != first_n]))
    start = int(rng.integers(4))
    first_sides = [_SIDES[start]]
    if rng.random() < 0.65:
        first_sides.append(_SIDES[(start + 1) % 4])
    second_sides = [s for s in _SIDES if s not in first_sides]
    puzzle = {"train": [], "test": []}
    seen, trained_orders = set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index < 2:
                n = (first_n, second_n)[index]
                sides = (first_sides, second_sides)[index]
                keep = allowed[index]
                delay = delayed and index == 1
                offaxis = index == 1
            else:
                n = (int(rng.integers(3, 7)) if split == "train"
                     else int(rng.choice(sorted(trained_orders))))
                side_count = int(rng.integers(1, 5))
                sides = [str(x) for x in rng.choice(_SIDES, size=side_count, replace=False)]
                adjacent = any((a in ("top", "bottom")) != (b in ("top", "bottom"))
                               for a in sides for b in sides)
                delay = delayed and adjacent and rng.random() < 0.5
                keep = int(rng.choice(allowed)) if split == "train" else held
                offaxis = False
            palette = _palette(rng) if switch_palette else base_palette
            pair = _make_pair(rng, n, keep, allowed if split == "train" else [1, 2, 3],
                              palette, sides, delay, offaxis, seen)
            puzzle[split].append(pair)
            if split == "train":
                trained_orders.add(n)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, info = _analyze_input(pair["input"])
            out = _grid(pair["output"])
            _need(inp.shape == out.shape, "canvas changed")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            _need(np.array_equal(out, _render_info(inp.shape, info)), "incorrect completed lattice")
            _need(not np.array_equal(out, _render_info(inp.shape, info, rotate=False)),
                  "omitting parent A did not change the answer")
            _need(not np.array_equal(out, _render_info(inp.shape, info, expand=False)),
                  "omitting parent B did not change the answer")
            bg, sc, mc = (info[k] for k in ("background", "structure", "marker"))
            payload = (inp != bg) & (inp != sc) & (inp != mc)
            _need(np.array_equal(inp[payload], out[payload]), "original payload changed")
            _need(all(out[p] == mc for p in info["markers"]), "control changed")
            for row in info["rows"][1:-1]:
                _need(np.all(out[row, info["final"][2]:info["final"][3] + 1] == sc),
                      "horizontal divider moved or failed to extend")
            for col in info["cols"][1:-1]:
                _need(np.all(out[info["final"][0]:info["final"][1] + 1, col] == sc),
                      "vertical divider moved or failed to extend")
            _need(set(np.unique(inp)) == set(np.unique(out)), "color vocabulary changed")
            _need(any(len(o) == 4 and not v for o, v, _ in info["records"]),
                  "no empty orbit to distinguish filling everything")
            _need(_area_witness(info), "no ordinal-versus-metric witness")
            _need(_interaction_witness(info), "no newly completed compartment grows with the frame")
            infos[split].append(info)
    training = infos["train"]
    counts = {len(v) for i in training for o, v, _ in i["records"] if 0 < len(v) < len(o)}
    _need(len(counts) == 2, "training must demonstrate two survivor counts")
    held, = {1, 2, 3} - counts
    _need(_reflection_witness(training[1]), "training lacks off-axis reflection counterexample")
    _need(set().union(*(set(i["by_side"]) for i in training)) == set(_SIDES),
          "training does not demonstrate all expansion directions")
    orders = {i["order"] for i in training}
    _need(len(orders) >= 2, "training lacks logical scale variation")
    for info in infos["test"]:
        _need(any(len(v) == held for o, v, _ in info["records"] if len(o) == 4),
              "test lacks held-out survivor count")
        _need(info["order"] in orders, "test uses an undemonstrated lattice order")
        _need(not _delays(info) or any(_delays(i) for i in training),
              "delayed test control lacks a training precedent")


def self_test():
    """Bounded semantic, diversity, count, and metamorphic checks."""
    shapes, orders, backgrounds, side_sets, survivor_counts = set(), set(), set(), set(), set()
    fingerprints, layouts = set(), set()
    measures = dict(delayed=0, multi_orbit=0, color_sharing=0, nonzero_bg=0,
                    palette_switch=0, empty_center=0, complete_orbit=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, info = _analyze_input(pair["input"])
                shapes.add(inp.shape)
                orders.add(info["order"])
                backgrounds.add(info["background"])
                side_sets.add(tuple(sorted(info["by_side"])))
                palettes.add((info["background"], info["structure"], info["marker"]))
                layouts.add((tuple(np.diff(info["rows"])), tuple(np.diff(info["cols"])),
                             info["old"], info["final"], info["labels"].tobytes()))
                occupied = [(o, v, c) for o, v, c in info["records"] if v and len(o) == 4]
                survivor_counts.update(len(v) for o, v, _ in occupied if len(v) < 4)
                measures["delayed"] += _delays(info) > 0
                measures["multi_orbit"] += len(occupied) > 1
                measures["color_sharing"] += len({c for _, _, c in occupied}) < len(occupied)
                measures["complete_orbit"] += any(len(v) == 4 for _, v, _ in occupied)
                measures["nonzero_bg"] += info["background"] != 0
                measures["empty_center"] += (info["order"] % 2 == 1
                    and info["labels"][info["order"] // 2, info["order"] // 2] == info["background"])
        measures["palette_switch"] += len(palettes) > 1
        # Input-only solve respects all square isometries and arbitrary palette names.
        inp, out = (puzzle["test"][0][k] for k in ("input", "output"))
        transformed_in = np.rot90(np.fliplr(inp), seed % 4)
        transformed_out = np.rot90(np.fliplr(out), seed % 4)
        _need(np.array_equal(solve(transformed_in.tolist()), transformed_out), "isometry failure")
        permutation = np.random.default_rng(seed + 10000).permutation(10)
        _need(np.array_equal(solve(permutation[inp]), permutation[out]), "palette permutation failure")
    _need(len(fingerprints) == 42 and len(layouts) >= 180, "structural diversity collapsed")
    _need(len(shapes) >= 45 and orders == {3, 4, 5, 6} and len(backgrounds) == 10,
          "size or role diversity collapsed")
    _need(len(side_sets) >= 12 and survivor_counts == {1, 2, 3}, "controls or damage collapsed")
    _need(all(v >= 10 for v in measures.values()), "a compatible variation axis disappeared")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(8000 + 10 * a + b, a, b)
            _need((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(123456, 10, 4), generate_puzzle(123456, 10, 4)
    _need(all(np.array_equal(x[k], y[k]) for s in ("train", "test")
              for x, y in zip(a[s], b[s]) for k in ("input", "output")), "not deterministic")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), shapes=len(shapes),
                structural_layouts=len(layouts), orders=sorted(orders),
                backgrounds=sorted(backgrounds), side_sets=len(side_sets),
                **{k: int(v) for k, v in measures.items()})


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
