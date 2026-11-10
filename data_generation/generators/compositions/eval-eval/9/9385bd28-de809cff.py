"""Peeling repair windows: corner-defined depth meets selective defect marking.

The input contains its complete legend, including the highlight swatch. No
example, random seed, generation metadata, or mutable cache is used by solve.
Only the standard library and NumPy are required. No function writes files.
"""

from itertools import product

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_test >= num_train:
        raise ValueError("num_test must be an integer in 1..4 and less than num_train")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return grid.astype(np.int8, copy=True)


def _corners(box, diagonal):
    r, c, s, t = box  # Inclusive bounds.
    if diagonal == 0:
        return {(r, c), (r, c+1), (r+1, c), (s, t), (s, t-1), (s-1, t)}
    return {(r, t), (r, t-1), (r+1, t), (s, c), (s, c+1), (s-1, c)}


def _ring(shape, point):
    r, c = point
    return [(a, b) for a in range(max(0, r-1), min(shape[0], r+2))
            for b in range(max(0, c-1), min(shape[1], c+2)) if (a, b) != point]


def _intersects(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _parse(grid):
    """Locate a unique header/table and read all six-cell registration glyphs.

    The header's color occurs exactly twice in the input. Each cue occurs once
    in the table and six times in the scene. All semantic color roles are
    disjoint except that different layers may share an ordinary fill color.
    """
    h, w = grid.shape
    counts = np.bincount(grid.ravel(), minlength=10)
    candidates = []
    for highlight in np.flatnonzero(counts == 2):
        positions = np.argwhere(grid == highlight)
        r, c = map(int, positions[0])
        if not np.array_equal(positions[1], (r, c+1)):
            continue
        for n in range(2, 5):
            if r+n >= h or c+1 >= w:
                continue
            table = (r, c, r+n, c+1)
            border = [(a, b) for a in range(max(0, r-1), min(h, r+n+2))
                      for b in range(max(0, c-1), min(w, c+3))
                      if not (r <= a <= r+n and c <= b <= c+1)]
            backgrounds = {int(grid[p]) for p in border}
            if len(backgrounds) != 1:
                continue
            bg = backgrounds.pop()
            cues = list(map(int, grid[r+1:r+n+1, c]))
            fills = list(map(int, grid[r+1:r+n+1, c+1]))
            if (len(set(cues)) != n or len(set(fills)) < 2
                    or len(set(fills)) > 3 or bg == highlight
                    or set(cues) & (set(fills) | {bg, int(highlight)})
                    or set(fills) & {bg, int(highlight)}):
                continue
            specs, good = [], True
            for i, cue in enumerate(cues):
                if counts[cue] != 7:
                    good = False
                    break
                cells = {tuple(map(int, p)) for p in np.argwhere(grid == cue)}
                cells.remove((r+1+i, c))
                rows, cols = zip(*cells)
                box = (min(rows), min(cols), max(rows), max(cols))
                if min(box[2]-box[0]+1, box[3]-box[1]+1) < 4:
                    good = False
                    break
                diagonal = next((d for d in (0, 1) if cells == _corners(box, d)), None)
                moat = (max(0, r-1), max(0, c-1), min(h-1, r+n+1), min(w-1, c+2))
                if diagonal is None or _intersects(box, moat):
                    good = False
                    break
                specs.append(dict(box=box, cue=cue, fill=fills[i], cells=cells, diagonal=diagonal))
            if not good or set(map(int, np.unique(grid))) != {bg, int(highlight), *cues, *fills}:
                continue
            guides = set().union(*(s["cells"] for s in specs))
            if len(guides) != 6*n:
                continue
            candidates.append(dict(background=bg, highlight=int(highlight), table=table,
                                   specs=specs, guides=guides))
    if len(candidates) != 1:
        raise ValueError("input must have one readable legend and intact corner registrations")
    return candidates[0]


def _compose(shape, parsed, skip=None, reverse=False):
    """Opaque rectangles, preserving a layer's own corners until occluded."""
    clean = np.full(shape, parsed["background"], dtype=np.int8)
    owner = np.full(shape, -1, dtype=np.int8)
    order = list(range(len(parsed["specs"])))
    if not reverse:
        order.reverse()
    omitted = {int(skip)} if _integer(skip) else set(skip or ())
    for i in order:
        if i in omitted:
            continue
        spec = parsed["specs"][i]
        r, c, s, t = spec["box"]
        clean[r:s+1, c:t+1] = spec["fill"]
        owner[r:s+1, c:t+1] = i
        for point in spec["cells"]:
            clean[point] = spec["cue"]
    return clean, owner


def _analyze(grid):
    parsed = _parse(grid)
    clean, owner = _compose(grid.shape, parsed)
    guide_mask = np.zeros(grid.shape, dtype=bool)
    for p in parsed["guides"]:
        guide_mask[p] = True
    # Guides are registration ink, including those of obscured layers; they
    # are not defects. All other background-valued cells in a box are holes.
    holes = [tuple(map(int, p)) for p in np.argwhere(
        (owner >= 0) & (grid == parsed["background"]) & ~guide_mask)]
    mismatch = (grid != clean) & ~guide_mask
    wrong = [tuple(map(int, p)) for p in np.argwhere(
        mismatch & (owner >= 0) & (grid != parsed["background"]))]
    speckles = [tuple(map(int, p)) for p in np.argwhere(mismatch & (owner < 0))
                if not (parsed["table"][0] <= p[0] <= parsed["table"][2]
                        and parsed["table"][1] <= p[1] <= parsed["table"][3])]
    views = [_compose(grid.shape, parsed, skip=i)[0] for i in range(len(parsed["specs"]))]
    return dict(parsed=parsed, clean=clean, owner=owner, holes=holes,
                wrong=wrong, speckles=speckles, views=views)


def _stamp(grid, analysis, *, uniform=False, back_wins=False, all_errors=False, removal="owner"):
    parsed = analysis["parsed"]
    result = analysis["clean"].copy()
    centers = list(analysis["holes"])
    if all_errors:
        centers += analysis["wrong"]
    # Window ownership, not iteration order or literal color, resolves overlap.
    centers.sort(key=lambda p: int(analysis["owner"][p]), reverse=not back_wins)
    for p in centers:
        i = int(analysis["owner"][p])
        view = analysis["views"][i]
        if removal == "covering":
            omitted = [j for j, s in enumerate(parsed["specs"])
                       if s["box"][0] <= p[0] <= s["box"][2]
                       and s["box"][1] <= p[1] <= s["box"][3]]
            view = _compose(grid.shape, parsed, skip=omitted)[0]
        elif removal == "fill":
            omitted = [j for j, s in enumerate(parsed["specs"])
                       if s["fill"] == parsed["specs"][i]["fill"]]
            view = _compose(grid.shape, parsed, skip=omitted)[0]
        for q in _ring(grid.shape, p):
            result[q] = view[p] if uniform else view[q]
    for p in centers:
        result[p] = parsed["highlight"]
    r, c, s, t = parsed["table"]
    result[r:s+1, c:t+1] = grid[r:s+1, c:t+1]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair noise; expose each erased cell's counterfactual depth window."""
    grid = _grid(input_grid)
    return _stamp(grid, _analyze(grid))


def _features(grid, analysis=None):
    a = _analyze(grid) if analysis is None else analysis
    p, holes, owner = a["parsed"], a["holes"], a["owner"]
    output = _stamp(grid, a)
    hole_set = set(holes)
    windows = {}
    for center in holes:
        i = int(owner[center])
        for q in _ring(grid.shape, center):
            if q not in hole_set:
                windows.setdefault(q, []).append((i, int(a["views"][i][q])))
    collision = any(len({v for _, v in demands}) > 1 for demands in windows.values())
    multicolor = any(len({int(a["views"][int(owner[h])][q])
                         for q in _ring(grid.shape, h) if q not in hole_set}) > 1 for h in holes)
    depths = [sum(s["box"][0] <= h[0] <= s["box"][2]
                  and s["box"][1] <= h[1] <= s["box"][3] for s in p["specs"]) for h in holes]
    return dict(shape=list(grid.shape), layers=len(p["specs"]), background=p["background"],
                fills=len({s["fill"] for s in p["specs"]}),
                boxes=[list(s["box"]) for s in p["specs"]],
                diagonals=sorted({s["diagonal"] for s in p["specs"]}),
                table=list(p["table"][:2]), holes=len(holes), wrong=len(a["wrong"]),
                speckles=len(a["speckles"]), maximum_hole_depth=max(depths, default=0),
                clipped=any(len(_ring(grid.shape, h)) < 8 for h in holes),
                adjacent=any(max(abs(h[0]-q[0]), abs(h[1]-q[1])) == 1
                             for j, h in enumerate(holes) for q in holes[:j]),
                multicolor=multicolor, collision=collision,
                priority_witness=not np.array_equal(output, _stamp(grid, a, back_wins=True)),
                spatial_witness=not np.array_equal(output, _stamp(grid, a, uniform=True)),
                single_object_witness=not np.array_equal(output, _stamp(grid, a, removal="covering")),
                identity_witness=not np.array_equal(output, _stamp(grid, a, removal="fill")),
                noise_witness=not np.array_equal(output, _stamp(grid, a, all_errors=True)),
                peeling_witness=not np.array_equal(output, _restore_legend(grid, a["clean"], p)))


def _restore_legend(grid, scene, parsed):
    result = scene.copy()
    r, c, s, t = parsed["table"]
    result[r:s+1, c:t+1] = grid[r:s+1, c:t+1]
    return result


def _geometry(rng, shape, n):
    h, w = shape
    # A local legend may sit anywhere. Edge-biased locations leave room for
    # broader rectangles, while free locations create internal pockets.
    tr = int(rng.integers(h-n))
    tc = int(rng.integers(w-1))
    if rng.random() < .65:
        tc = int(rng.choice([0, w-2]))
    if rng.random() < .30:
        tr = int(rng.choice([0, h-n-1]))
    table = (tr, tc, tr+n, tc+1)
    moat = (max(0, tr-1), max(0, tc-1), min(h-1, tr+n+1), min(w-1, tc+2))
    specs, used = [], set()
    for i in range(n):
        for _ in range(100):
            rh = int(rng.integers(4, h+1))
            rw = int(rng.integers(4, w+1))
            r = int(rng.integers(h-rh+1))
            c = int(rng.integers(w-rw+1))
            if rng.random() < .22:
                r = int(rng.choice([0, h-rh]))
            if rng.random() < .22:
                c = int(rng.choice([0, w-rw]))
            box = (r, c, r+rh-1, c+rw-1)
            if _intersects(box, moat) or any(box == s["box"] for s in specs):
                continue
            d = int(rng.integers(2))
            cells = _corners(box, d)
            if cells & used:
                continue
            specs.append(dict(box=box, cells=cells, diagonal=d))
            used.update(cells)
            break
        else:
            return None
    specs = [specs[int(i)] for i in rng.permutation(n)]
    return table, specs


def _choose_holes(rng, clean, owner, parsed, views, teaching):
    available = [tuple(map(int, p)) for p in np.argwhere(owner >= 0)
                 if tuple(map(int, p)) not in parsed["guides"]]
    rng.shuffle(available)
    chosen = []
    if teaching == 0:
        # A ring crosses a lower rectangle's boundary while its center is
        # inside that lower rectangle: a uniform "other color" cannot work.
        for p in available:
            i = int(owner[p])
            view = views[i]
            ring = _ring(clean.shape, p)
            if (view[p] != parsed["background"] and view[p] != clean[p]
                    and len({int(view[q]) for q in ring}) >= 2
                    and any(view[q] == parsed["background"] for q in ring)):
                chosen = [p]
                break
        if not chosen:
            return None
    elif teaching == 1:
        # Adjacent holes across a depth seam create conflicting rings. A
        # separate depth-three hole teaches that only one object is lifted.
        pool = set(available)
        for p in available:
            for q in _ring(clean.shape, p):
                if q not in pool or owner[p] == owner[q]:
                    continue
                common = set(_ring(clean.shape, p)) & set(_ring(clean.shape, q)) - {p, q}
                if any(views[int(owner[p])][s] != views[int(owner[q])][s] for s in common):
                    chosen = [p, q]
                    break
            if chosen:
                break
        if not chosen:
            return None
        deep = [p for p in available if sum(s["box"][0] <= p[0] <= s["box"][2]
                    and s["box"][1] <= p[1] <= s["box"][3] for s in parsed["specs"]) >= 3]
        if not deep:
            return None
        if deep[0] not in chosen:
            chosen.append(deep[0])
    target = max(len(chosen), int(rng.integers(2, min(11, max(4, len(available)//18))+1)))
    # Mix independent points with local clusters. Centers may be adjacent and
    # different owners may make incompatible demands: priority handles both.
    if chosen and rng.random() < .60:
        near = [p for p in available if p not in chosen and
                any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 2 for q in chosen)]
        chosen.extend(near[:min(2, target-len(chosen))])
    for p in available:
        if len(chosen) >= target:
            break
        if p not in chosen:
            chosen.append(p)
    return chosen


def _candidate(rng, shape, n, teaching, repeated):
    geometry = _geometry(rng, shape, n)
    if geometry is None:
        return None
    table, specs = geometry
    colors = list(map(int, rng.permutation(10)))
    bg, highlight = colors[:2]
    cues = colors[2:2+n]
    fill_count = 2 if n == 2 or repeated or teaching == 1 else int(rng.integers(2, min(3, n)+1))
    fills = colors[2+n:2+n+fill_count]
    assignments = list(fills) + [int(rng.choice(fills)) for _ in range(n-fill_count)]
    rng.shuffle(assignments)
    for i, spec in enumerate(specs):
        spec.update(cue=cues[i], fill=assignments[i])
    parsed = dict(background=bg, highlight=highlight, table=table, specs=specs,
                  guides=set().union(*(s["cells"] for s in specs)))
    clean, owner = _compose(shape, parsed)
    if any(np.count_nonzero(owner == i) < 9 for i in range(n)):
        return None
    views = [_compose(shape, parsed, skip=i)[0] for i in range(n)]
    holes = _choose_holes(rng, clean, owner, parsed, views, teaching)
    if holes is None:
        return None
    grid = clean.copy()
    for p in holes:
        grid[p] = bg
    occupied = set(holes) | parsed["guides"]
    # Keep at least one ordinary error outside every window in a teaching
    # example, visibly repairing it without creating a highlighted center.
    window_cells = set().union(*(set(_ring(shape, p)) | {p} for p in holes))
    wrong_pool = [tuple(map(int, p)) for p in np.argwhere(owner >= 0)
                  if tuple(map(int, p)) not in occupied]
    rng.shuffle(wrong_pool)
    wrong_pool.sort(key=lambda p: p in window_cells)
    wrong_count = int(rng.integers(1 if teaching is not None else 0, 5))
    wrong_count = min(wrong_count, max(0, int(.23*np.count_nonzero(owner >= 0))-len(holes)))
    if teaching is not None and wrong_count == 0:
        return None
    if teaching is not None and not any(p not in window_cells for p in wrong_pool):
        return None
    for p in wrong_pool[:wrong_count]:
        grid[p] = int(rng.choice([f for f in fills if f != int(clean[p])]))
    r, c, s, t = table
    moat = (max(0, r-1), max(0, c-1), min(shape[0]-1, s+1), min(shape[1]-1, t+1))
    exterior = [tuple(map(int, p)) for p in np.argwhere(owner < 0)
                if not (moat[0] <= p[0] <= moat[2] and moat[1] <= p[1] <= moat[3])]
    rng.shuffle(exterior)
    speckle_count = int(rng.integers(1 if teaching is not None else 0, 5))
    if teaching is not None and not exterior:
        return None
    for p in exterior[:speckle_count]:
        grid[p] = int(rng.choice(fills))
    # All registration marks are visible in the input, even for hidden corners.
    for spec in specs:
        for p in spec["cells"]:
            grid[p] = spec["cue"]
    grid[r, c:c+2] = highlight
    for i, spec in enumerate(specs):
        grid[r+1+i, c:c+2] = spec["cue"], spec["fill"]
    a = _analyze(grid)
    if a["holes"] != sorted(holes):
        raise AssertionError("construction lost a hole")
    f = _features(grid, a)
    if teaching == 0 and not (f["spatial_witness"] and f["noise_witness"] and f["peeling_witness"]):
        return None
    if teaching == 1 and not (f["priority_witness"] and f["maximum_hole_depth"] >= 3
                              and f["adjacent"] and f["noise_witness"]
                              and f["single_object_witness"] and f["identity_witness"]):
        return None
    # Labels are constructed from the sampled geometry. Inference is checked
    # against it, rather than letting the parser silently redefine the sample.
    assert np.array_equal(a["clean"], clean)
    return {"input": grid, "output": _stamp(grid, a)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # Either transfer to an extra layer, or transfer to a larger canvas.
    depth_shift = bool(rng.integers(2))
    repeated = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    teaching_slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = teaching_slots.index(index) if split == "train" and index in teaching_slots else None
            n = 2 if teaching == 0 else (3 if teaching == 1 else int(rng.integers(2, 4)))
            if split == "test" and depth_shift:
                n = 4
            for attempt in range(320):
                if depth_shift:
                    low, high = 16, 31
                elif split == "train":
                    low, high = 14, 23
                else:
                    low, high = 24, 31
                shape = tuple(map(int, rng.integers(low, high, size=2)))
                candidate = _candidate(rng, shape, n, teaching, repeated)
                if candidate is None:
                    continue
                key = (candidate["input"].shape, candidate["input"].tobytes())
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(candidate)
                break
            else:
                raise RuntimeError("bounded construction could not meet the teaching constraints")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert grid.shape == output.shape
            key = (grid.shape, grid.tobytes())
            assert key not in seen, "duplicate inputs"
            seen.add(key)
            a = _analyze(grid)
            parsed = a["parsed"]
            fills = {s["fill"] for s in parsed["specs"]}
            table_mask = np.zeros(grid.shape, dtype=bool)
            r, c, s, t = parsed["table"]
            table_mask[r:s+1, c:t+1] = True
            guide_mask = np.zeros(grid.shape, dtype=bool)
            for p in parsed["guides"]:
                guide_mask[p] = True
            assert np.all(np.isin(grid[~table_mask & ~guide_mask], [parsed["background"], *fills]))
            assert a["holes"], "each example must demonstrate selective marking"
            assert all(np.count_nonzero(a["owner"] == i) >= 9 for i in range(len(parsed["specs"])))
            assert len(a["holes"])+len(a["wrong"]) <= .23*np.count_nonzero(a["owner"] >= 0)
            assert np.array_equal(output, _stamp(grid, a)), "incorrect input-derived output"
            assert np.array_equal(output, solve(grid.copy())), "solve disagrees"
            assert np.all(output[table_mask] == grid[table_mask]), "legend was modified"
            assert all(output[p] == parsed["highlight"] for p in a["holes"])
            assert not np.array_equal(grid, output)
            if split == "train":
                train_features.append(_features(grid, a))
    for witness in ("spatial_witness", "priority_witness", "noise_witness", "peeling_witness", "adjacent",
                    "single_object_witness", "identity_witness"):
        assert any(f[witness] for f in train_features), "training lacks " + witness
    assert any(f["maximum_hole_depth"] >= 3 for f in train_features)
    assert any(f["wrong"] and f["speckles"] for f in train_features)


def _reference(grid):
    """Slow pixel/coverage formulation used only by self_test.

    This shares the legend parser but not array compositing, hole detection,
    ownership arrays, or the ordered ring painter with the production solver.
    It is a consistency check, not an independent semantic judge.
    """
    p = _parse(grid)
    h, w = grid.shape
    coverage = {}
    for r, c in product(range(h), range(w)):
        coverage[r, c] = [i for i, s in enumerate(p["specs"])
                           if s["box"][0] <= r <= s["box"][2]
                           and s["box"][1] <= c <= s["box"][3]]
    def color(point, omitted=None):
        for i in coverage[point]:
            if i != omitted:
                s = p["specs"][i]
                return s["cue"] if point in s["cells"] else s["fill"]
        return p["background"]
    centers = {q: coverage[q][0] for q in coverage if coverage[q]
               and q not in p["guides"] and grid[q] == p["background"]}
    result = np.empty_like(grid)
    for q in coverage:
        if q in centers:
            result[q] = p["highlight"]
            continue
        demands = [i for center, i in centers.items()
                   if max(abs(q[0]-center[0]), abs(q[1]-center[1])) == 1]
        result[q] = color(q, min(demands)) if demands else color(q)
    return _restore_legend(grid, result, p)


def self_test():
    """Serial 42-seed variation/invariant sweep; no files or retained grids."""
    records, signatures, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                assert np.array_equal(pair["output"], _reference(grid))
                f = _features(grid)
                f["split"] = split
                records.append(f)
                signature.append((grid.shape, grid.tobytes(), pair["output"].tobytes()))
                roles = {}
                structures.add((grid.shape, bytes(roles.setdefault(int(c), len(roles)) for c in grid.flat)))
        signatures.add(tuple(signature))
    assert len(signatures) == 42
    assert len(structures) >= 180
    assert len({tuple(f["shape"]) for f in records}) >= 60
    assert {f["layers"] for f in records} == {2, 3, 4}
    assert {f["background"] for f in records} == set(range(10))
    assert {f["fills"] for f in records} == {2, 3}
    assert len({str(f["boxes"]) for f in records}) >= 180
    assert len({tuple(f["table"]) for f in records}) >= 35
    assert any(len(f["diagonals"]) == 2 for f in records)
    assert any(f["clipped"] for f in records)
    assert any(not f["clipped"] for f in records)
    assert any(f["maximum_hole_depth"] == 4 for f in records)
    assert any(f["wrong"] == 0 for f in records)
    assert any(f["speckles"] == 0 for f in records)
    for split in ("train", "test"):
        fs = [f for f in records if f["split"] == split]
        for feature in ("spatial_witness", "priority_witness", "adjacent", "clipped",
                        "single_object_witness", "identity_witness"):
            assert any(f[feature] for f in fs), (split, feature)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1729, a, b))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated output accepted")

