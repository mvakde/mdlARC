"""Predictive docks: geometrically query and capture an unseen sequence member.

Standard library and NumPy only. No runtime file I/O, caches, parent imports,
training-dependent conventions, or stored examples. Labels use explicit latent
rectangles; solve reconstructs geometry from the input components.
"""
from __future__ import annotations

import hashlib

import numpy as np


_OUTS = ((0, 1), (-1, 0), (0, -1), (1, 0))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "invalid color"
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _OUTS:
                point = r + dr, c + dc
                if point in unseen:
                    unseen.remove(point)
                    cells.add(point)
                    stack.append(point)
        result.append(cells)
    return result


def _bounds(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs), min(cs), max(cs)


def _is_rect(cells):
    r0, r1, c0, c1 = _bounds(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _dot(point, vector):
    return int(point[0] * vector[0] + point[1] * vector[1])


def _point(k, q, out, perp):
    return k * out[0] + q * perp[0], k * out[1] + q * perp[1]


def _local_rect(k, q, length, span, out, perp):
    return {_point(kk, qq, out, perp) for kk in range(k, k + length)
            for qq in range(q, q + span)}


def _paint(grid, cells, color):
    for point in cells:
        grid[point] = color


def _parse_fork(cells):
    possibilities = []
    for out in _OUTS:
        perp = out[1], -out[0]
        kbase = min(_dot(p, out) for p in cells)
        qbase = min(_dot(p, perp) for p in cells)
        local = {(_dot(p, out) - kbase, _dot(p, perp) - qbase) for p in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = []
        valid = True
        for q in range(width):
            depths = sorted(k for k, qq in local if qq == q and k > 0)
            if depths != list(range(1, len(depths) + 1)):
                valid = False
                break
            lengths.append(len(depths))
        if not valid:
            continue
        bands, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            lo, depth = q, lengths[q]
            while q < width and lengths[q] == depth:
                q += 1
            bands.append((lo, q, depth))
        if len(bands) != 3 or bands[0][0] != 0 or bands[2][1] != width:
            continue
        if not (bands[0][1] < bands[1][0] and bands[1][1] < bands[2][0]):
            continue
        a, b, depth = bands[1]
        if a + b != width:
            continue
        possibilities.append(dict(out=out, perp=perp, base=kbase, qbase=qbase,
                                  q=qbase + a, span=b - a, stop=kbase + depth + 1,
                                  bands=tuple(bands), width=width, cells=cells))
    assert len(possibilities) == 1, "fork requires a unique base and three positive prongs"
    return possibilities[0]


def _affine(values):
    values = list(map(int, values))
    delta = values[1] - values[0]
    assert all(v == values[0] + i * delta for i, v in enumerate(values)), "non-affine observations"
    return values[0], delta


def _geometry(model, index):
    h = model["h0"] + index * model["dh"]
    length = model["l0"] + index * model["dl"]
    k = model["k0"] + index * model["dk"]
    q = (model["q0"] + index * (model["h0"] + model["gap"]) +
         model["dh"] * index * (index - 1) // 2)
    return k, q, length, h


def _inside(geometry, limits):
    k, q, length, span = geometry
    ka, kb, qa, qb = limits
    return (length > 0 and span > 0 and ka <= k and k + length - 1 <= kb
            and qa <= q and q + span - 1 <= qb)


def _analyze(input_grid):
    grid = _grid(input_grid)
    vals, counts = np.unique(grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, "ambiguous background"
    background = int(vals[np.argmax(counts)])
    rectangles, forks = [], []
    for color in vals:
        if color == background:
            continue
        for cells in _components(grid == color):
            if _is_rect(cells):
                rectangles.append(dict(cells=cells, color=int(color)))
            else:
                fork = _parse_fork(cells)
                fork["color"] = int(color)
                forks.append(fork)
    assert 3 <= len(rectangles) <= 4 and 1 <= len(forks) <= 2, "invalid observation/receiver count"
    assert {r["color"] for r in rectangles}.isdisjoint(f["color"] for f in forks), "mixed color roles"
    out, perp = forks[0]["out"], forks[0]["perp"]
    assert all(f["out"] == out for f in forks), "receivers must share an outward direction"
    for rect in rectangles:
        ks = [_dot(p, out) for p in rect["cells"]]
        qs = [_dot(p, perp) for p in rect["cells"]]
        rect.update(k=min(ks), q=min(qs), length=max(ks) - min(ks) + 1,
                    span=max(qs) - min(qs) + 1)
    rectangles.sort(key=lambda r: r["q"])
    h0, dh = _affine(r["span"] for r in rectangles)
    l0, dl = _affine(r["length"] for r in rectangles)
    k0, dk = _affine(r["k"] for r in rectangles)
    gaps = [b["q"] - a["q"] - a["span"] for a, b in zip(rectangles, rectangles[1:])]
    assert len(set(gaps)) == 1 and gaps[0] > 0, "observations need one positive edge gap"
    colors = [r["color"] for r in rectangles]
    cycles = [p for p in range(1, len(colors)) if len(set(colors[:p])) == p
              and all(color == colors[i % p] for i, color in enumerate(colors))]
    assert cycles, "observe all color phases and a repeated phase"
    palette = tuple(colors[:min(cycles)])
    model = dict(h0=h0, dh=dh, l0=l0, dl=dl, k0=k0, dk=dk, q0=rectangles[0]["q"],
                 gap=gaps[0], palette=palette, observed=len(rectangles))
    corners = ((0, 0), (0, grid.shape[1] - 1), (grid.shape[0] - 1, 0),
               (grid.shape[0] - 1, grid.shape[1] - 1))
    limits = (min(_dot(p, out) for p in corners), max(_dot(p, out) for p in corners),
              min(_dot(p, perp) for p in corners), max(_dot(p, perp) for p in corners))
    # There are fewer than 30 positive, separated transverse slots in a grid.
    candidates = {}
    for i in range(-30, 31):
        if 0 <= i < len(rectangles):
            continue
        geom = _geometry(model, i)
        if not _inside(geom, limits):
            continue
        if not all(_inside(_geometry(model, j), limits)
                   for j in range(min(0, i), max(len(rectangles) - 1, i) + 1)):
            continue
        candidates[i] = geom
    matches, occupied = [], set()
    for fi, fork in enumerate(forks):
        valid = []
        for i, geom in candidates.items():
            k, q, length, span = geom
            if q != fork["q"] or span != fork["span"] or k <= fork["stop"]:
                continue
            final_geom = fork["stop"], q, length, span
            if not _inside(final_geom, limits):
                continue
            route = _local_rect(fork["stop"], q, k + length - fork["stop"], span, out, perp)
            if any(grid[p] != background for p in route):
                continue
            final = _local_rect(*final_geom, out, perp)
            source = _local_rect(*geom, out, perp)
            valid.append(dict(fi=fi, index=i, geometry=geom, final=final, source=source,
                              route=route, distance=k - fork["stop"],
                              color=palette[i % len(palette)]))
        assert len(valid) <= 1, "multiple hypothetical matches"
        if valid:
            match = valid[0]
            assert occupied.isdisjoint(match["route"]), "interacting captures"
            occupied |= match["route"]
            matches.append(match)
    assert matches, "at least one receiver must demonstrate capture"
    return dict(grid=grid, background=background, rectangles=rectangles, forks=forks,
                model=model, out=out, perp=perp, matches=matches,
                inactive=len(forks) - len(matches))


def _render(analysis):
    output = analysis["grid"].copy()
    for match in analysis["matches"]:
        fork = analysis["forks"][match["fi"]]
        _paint(output, match["route"], fork["color"])
        _paint(output, match["final"], match["color"])
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every predicted cell from this input alone."""
    return _render(_analyze(input_grid))


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(name + " out of range or not an integer")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _fork_cells(y, span, base, thickness, clearance, lengths):
    # Ground truth uses canonical row/column strokes, unlike input projection.
    top = y - thickness - clearance
    width = span + 2 * (thickness + clearance)
    cells = {(row, base) for row in range(top, top + width)}
    for start, stop, depth in ((top, top + thickness, lengths[0]),
                               (y, y + span, lengths[1]),
                               (y + span + clearance, top + width, lengths[2])):
        cells |= {(row, col) for row in range(start, stop)
                  for col in range(base + 1, base + depth + 1)}
    return cells


def _sample_geometry(rng, split, index, compact=False):
    initial = split == "train" and index < 2
    observed = 3 if initial or compact else int(rng.integers(3, 5))
    prefix = int(split == "train") if compact else 1 if initial else int(rng.integers(2))
    skipped = (int(split == "test") if compact else
               0 if split == "train" else int(rng.integers(1, 3)))
    total = prefix + observed + skipped + 1
    dh = 1 if initial or compact else int(rng.choice([0, 1, 1, 2]))
    hstart, gap = (1, 1) if compact else (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
    heights = [hstart + dh * j for j in range(total)]
    ys = [0]
    for h in heights[:-1]:
        ys.append(ys[-1] + h + gap)
    dl = 0 if compact else int(rng.choice([-1, 0, 1])) if dh else int(rng.choice([-1, 1]))
    min_length = 1 if compact else int(rng.integers(1, 4))
    lengths = [min_length + (j if dl >= 0 else total - j - 1) * abs(dl) for j in range(total)]
    dx = 0 if compact else int(rng.integers(-2, 3))
    queries = []
    if prefix:
        queries.append((0, not (initial and index == 0) and bool(initial or rng.random() < 0.7)))
    queries.append((total - 1, True))
    forks = []
    for qi, (member, active) in enumerate(queries):
        span = heights[member] + int(not active)
        thickness, clearance = ((1, 1) if compact else
                                (int(rng.integers(1, 3)), int(rng.integers(1, 3))))
        base = 0 if compact else int(rng.integers(4))
        if compact:
            arms = (1, 4, 1) if qi == 0 and prefix else (3, 1, 3)
        elif initial:
            # Both demonstrations contrast the central tip with guard tips.
            if qi == 0:
                arms = (int(rng.integers(1, 3)), int(rng.integers(4, 6)), int(rng.integers(1, 3)))
            else:
                arms = (int(rng.integers(3, 6)), int(rng.integers(1, 3)), int(rng.integers(3, 6)))
        else:
            arms = tuple(map(int, rng.integers(1, 6, size=3)))
        cells = _fork_cells(ys[member], span, base, thickness, clearance, arms)
        forks.append(dict(member=member, active=active, cells=cells,
                          stop=base + arms[1] + 1))
    farthest_tip = max(c for f in forks for _, c in f["cells"])
    source_min = farthest_tip + int(rng.integers(2, 5))
    xs = [source_min + j * dx - min(0, (total - 1) * dx) for j in range(total)]
    sources = [{(r, c) for r in range(ys[j], ys[j] + heights[j])
                for c in range(xs[j], xs[j] + lengths[j])} for j in range(total)]
    observed_indices = range(prefix, prefix + observed)
    visible = set().union(*(sources[j] for j in observed_indices))
    structural = set()
    for f in forks:
        if not structural.isdisjoint(f["cells"]) or not visible.isdisjoint(f["cells"]):
            return None
        structural |= f["cells"]
    occupied = structural | visible
    swept = set()
    for f in forks:
        if not f["active"]:
            continue
        j = f["member"]
        if xs[j] <= f["stop"]:
            return None
        final = {(r, c) for r in range(ys[j], ys[j] + heights[j])
                 for c in range(f["stop"], f["stop"] + lengths[j])}
        route = {(r, c) for r in range(ys[j], ys[j] + heights[j])
                 for c in range(f["stop"], xs[j] + lengths[j])}
        if not route.isdisjoint(occupied | swept):
            return None
        f.update(final=final, route=route)
        swept |= route
    # Include all traversed virtual sources in the domain, not only queries.
    extent = structural | set().union(*sources)
    ymin, ymax, xmin, xmax = _bounds(extent)
    height, width = ymax - ymin + 1, xmax - xmin + 1
    if max(height, width) > 30:
        return None
    top = int(rng.integers(min(2, 30 - height) + 1))
    bottom = int(rng.integers(min(2, 30 - height - top) + 1))
    left = int(rng.integers(min(2, 30 - width) + 1))
    right = int(rng.integers(min(2, 30 - width - left) + 1))
    shift = top - ymin, left - xmin

    def shifted(cells):
        return {(r + shift[0], c + shift[1]) for r, c in cells}

    for f in forks:
        for key in ("cells", "route", "final"):
            if key in f:
                f[key] = shifted(f[key])
    sources = [shifted(cells) for cells in sources]
    return dict(shape=(height + top + bottom, width + left + right),
                sources=sources, forks=forks, visible=list(observed_indices),
                observed=observed, initial=initial)


def _make_pair(rng, split, index, background, receiver_options, rotation, force_compact=False):
    geometry = _sample_geometry(rng, split, index, compact=True) if force_compact else None
    if not force_compact:
        for _ in range(240):
            geometry = _sample_geometry(rng, split, index)
            if geometry is not None:
                break
    if geometry is None:
        # A constructive <=27-by-14 layout of the same rule guarantees progress.
        geometry = _sample_geometry(rng, split, index, compact=True)
    assert geometry is not None, "compact construction must fit"
    nf = len(geometry["forks"])
    receiver_colors = list(map(int, rng.choice(receiver_options, size=nf, replace=False)))
    content_options = [c for c in range(10) if c != background and c not in receiver_colors]
    p = 2 if geometry["initial"] else int(rng.integers(1, geometry["observed"]))
    palette = list(map(int, rng.choice(content_options, size=p, replace=False)))
    grid = np.full(geometry["shape"], background, dtype=np.int8)
    for j in geometry["visible"]:
        _paint(grid, geometry["sources"][j], palette[j % p])
    for f, color in zip(geometry["forks"], receiver_colors):
        _paint(grid, f["cells"], color)
    output = grid.copy()
    for f, color in zip(geometry["forks"], receiver_colors):
        if f["active"]:
            _paint(output, f["route"], color)
            _paint(output, f["final"], palette[f["member"] % p])
    vals, counts = np.unique(grid, return_counts=True)
    assert int(vals[np.argmax(counts)]) == background and sum(counts == counts.max()) == 1
    if bool(rng.integers(2)):
        grid, output = grid[::-1].copy(), output[::-1].copy()
    grid, output = np.rot90(grid, rotation).copy(), np.rot90(output, rotation).copy()
    return {"input": grid, "output": output}, receiver_colors


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate a whole deterministic puzzle without writing artifacts."""
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    backgrounds = list(map(int, rng.permutation(10)))
    used_receiver_colors = {c: set() for c in range(10)}
    base_rotation = int(rng.integers(4))
    puzzle = {"train": [], "test": []}
    ordinal = 0
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            background = backgrounds[ordinal % 10]
            options = [c for c in range(10) if c != background and c not in used_receiver_colors[background]]
            rotation = ((base_rotation + index) % 4 if split == "train" and index < 2 else
                        (base_rotation + int(rng.integers(2))) % 4 if split == "test" else
                        int(rng.integers(4)))
            pair, receiver_colors = _make_pair(rng, split, index, background, options, rotation)
            used_receiver_colors[background].update(receiver_colors)
            puzzle[split].append(pair)
            ordinal += 1
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    """Check returned arrays, continuation, capture, ablations, and shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert grid.shape == output.shape and _key(grid) not in seen
            seen.add(_key(grid))
            a = _analyze(grid)
            assert np.array_equal(_render(a), output), "incorrect capture label"
            assert not np.array_equal(grid, output), "A-omission identity ablation survived"
            assert set(np.unique(output)) <= set(np.unique(grid)), "hidden color convention"
            for rect in a["rectangles"]:
                assert all(output[p] == rect["color"] for p in rect["cells"]), "changed observation"
            active_ids = {m["fi"] for m in a["matches"]}
            for fi, fork in enumerate(a["forks"]):
                assert min(b[2] for b in fork["bands"]) >= 1
                assert all(output[p] == fork["color"] for p in fork["cells"]), "changed receiver"
                if fi not in active_ids:
                    assert not any(p in m["route"] for p in fork["cells"] for m in a["matches"])
            no_docking = grid.copy()
            for match in a["matches"]:
                fork = a["forks"][match["fi"]]
                assert match["distance"] > 0 and match["route"] - match["final"]
                assert all(grid[p] == a["background"] for p in match["source"])
                assert all(output[p] == match["color"] for p in match["final"])
                assert all(output[p] == fork["color"] for p in match["route"] - match["final"])
                _paint(no_docking, match["source"], match["color"])
                i = match["index"]
                if split == "train":
                    assert i in (-1, len(a["rectangles"])), "training should query neighbors"
            assert not np.array_equal(no_docking, output), "B-omission source rendering survived"
            if split == "test":
                assert any(m["index"] >= len(a["rectangles"]) + 1 or m["index"] <= -2
                           for m in a["matches"]), "test needs skipped members"
            analyses[split].append(a)
    first, second = analyses["train"][:2]
    assert first["inactive"] == 1 and len(first["matches"]) == 1
    assert second["inactive"] == 0 and len(second["matches"]) == 2
    for a in (first, second):
        assert a["model"]["dh"] != 0 and len(a["model"]["palette"]) == 2
        visible_spans = {r["span"] for r in a["rectangles"]}
        assert all(m["geometry"][3] not in visible_spans for m in a["matches"]), "constant-copy A ablation survived"
        center_relations = [f["bands"][1][2] - max(f["bands"][0][2], f["bands"][2][2]) for f in a["forks"]]
        assert min(center_relations) < 0 < max(center_relations)


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _short_move_regression():
    grid = np.zeros((18, 20), dtype=np.int8)
    for y, span, length, color in ((2, 1, 3, 1), (4, 2, 4, 2), (7, 3, 5, 1)):
        grid[y:y + span, 12:12 + length] = color
    _paint(grid, _fork_cells(11, 4, 6, 1, 1, (1, 4, 1)), 9)
    output = grid.copy()
    output[11:15, 11:18] = 9
    output[11:15, 11:17] = 2
    assert np.array_equal(solve(grid), output)
    match = _analyze(grid)["matches"][0]
    assert match["distance"] == 1 < match["geometry"][2]


def self_test():
    """Bounded, streamed checks; returns compact scalar diversity evidence."""
    signatures, structures, shapes, directions, backgrounds = set(), set(), set(), set(), set()
    cycles, growth, axial_growth, drifts, gaps, spans, query_distances = (set() for _ in range(7))
    captures = inactive = pairs = backward = multi = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                pairs += 1
                shapes.add(pair["input"].shape)
                directions.add(a["out"])
                backgrounds.add(a["background"])
                m = a["model"]
                cycles.add(len(m["palette"]))
                growth.add(m["dh"])
                axial_growth.add(m["dl"])
                drifts.add(m["dk"])
                gaps.add(m["gap"])
                spans.update(f["span"] for f in a["forks"])
                captures += len(a["matches"])
                inactive += a["inactive"]
                multi += len(a["matches"]) > 1
                backward += any(match["index"] < 0 for match in a["matches"])
                for match in a["matches"]:
                    i, n = match["index"], m["observed"]
                    query_distances.add(-i - 1 if i < 0 else i - n)
                normalized, roles = [], {}
                for color in pair["input"].flat:
                    normalized.append(roles.setdefault(int(color), len(roles)))
                structures.add(hashlib.sha256(bytes(pair["input"].shape) + bytes(normalized)).hexdigest())
        if seed < 3:
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
            sample = puzzle["test"][0]
            for turns in range(4):
                assert np.array_equal(solve(np.rot90(sample["input"], turns)), np.rot90(sample["output"], turns))
            palette = np.random.default_rng(seed + 100).permutation(10)
            assert np.array_equal(solve(palette[sample["input"]]), palette[sample["output"]])
    assert len(signatures) == 42 and len(structures) >= 180 and len(shapes) >= 35
    assert len(directions) == 4 and len(backgrounds) == 10 and cycles == {1, 2, 3}
    assert {-1, 0, 1} <= growth and {-1, 0, 1} <= axial_growth
    assert {-2, -1, 0, 1, 2} <= drifts and gaps == {1, 2, 3}
    assert len(spans) >= 7 and {0, 1, 2} <= query_distances
    assert inactive >= 42 and multi >= 42 and backward >= 42
    legal_counts = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(91, nt, ne)
            validate_generated_puzzle(p)
            legal_counts += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None), ("4", 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    broken = generate_puzzle(77, 2, 1)
    a = _analyze(broken["test"][0]["input"])
    point = min(a["matches"][0]["final"])
    broken["test"][0]["output"][point] = a["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    _short_move_regression()
    # Exercise the normally unnecessary bounded fallback directly, including
    # every special demonstration and the farther test query.
    fallback = {"train": [], "test": []}
    for split, count in (("train", 2), ("test", 1)):
        for i in range(count):
            pair, _ = _make_pair(np.random.default_rng(400 + i), split, i, 0,
                                 list(range(1, 10)), i, force_compact=True)
            fallback[split].append(pair)
    validate_generated_puzzle(fallback)
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                normalized_structures=len(structures), input_shapes=len(shapes),
                directions=len(directions), backgrounds=len(backgrounds), cycles=sorted(cycles),
                transverse_increments=sorted(growth), axial_increments=sorted(axial_growth),
                axial_drifts=sorted(drifts), gaps=sorted(gaps), receiver_spans=sorted(spans),
                skipped_counts=sorted(query_distances), captures=captures, inactive=inactive,
                multiple_capture_pairs=multi, backward_query_pairs=backward,
                legal_count_combinations=legal_counts)
