"""Transparent detector lattices illuminated by physical southeast rays.

No parent code or data is imported.  Coordinates in the ray rule are input
pixels; only the final readout uses logical compartment coordinates.
"""

from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grids must be 2D integer arrays, sides 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _runs(mask):
    runs = []
    for i in np.flatnonzero(mask):
        i = int(i)
        if runs and runs[-1][1] == i:
            runs[-1] = (runs[-1][0], i + 1)
        else:
            runs.append((i, i + 1))
    return runs


def _connected(points):
    points = set(points)
    if not points:
        return False
    seen = {next(iter(points))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in points and q not in seen:
                stack.append(q)
                seen.add(q)
    return seen == points


def _parse_lattice(input_grid):
    """Find the unique full-band lattice and its inset monochromatic patches."""
    grid = _grid(input_grid)
    candidates = []
    for raw in np.unique(grid):
        separator = int(raw)
        rr, cc = np.where(grid == separator)
        top, bottom, left, right = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
        sub = grid[top:bottom + 1, left:right + 1]
        full_r, full_c = np.all(sub == separator, axis=1), np.all(sub == separator, axis=0)
        rb, cb = _runs(full_r), _runs(full_c)
        if len(rb) < 2 or len(cb) < 2:
            continue
        if rb[0][0] != 0 or rb[-1][1] != sub.shape[0] or cb[0][0] != 0 or cb[-1][1] != sub.shape[1]:
            continue
        if not np.array_equal(sub == separator, full_r[:, None] | full_c[None, :]):
            continue
        rows = [(rb[i][1] + top, rb[i + 1][0] + top) for i in range(len(rb) - 1)]
        cols = [(cb[i][1] + left, cb[i + 1][0] + left) for i in range(len(cb) - 1)]
        if any(b - a < 3 for a, b in rows + cols):
            continue
        canvas = int(grid[rows[0][0], cols[0][0]])
        if canvas == separator:
            continue
        outside = np.ones(grid.shape, dtype=bool)
        outside[top:bottom + 1, left:right + 1] = False
        if np.any(grid[outside] != canvas):
            continue
        patches, valid = [], True
        for i, (rs, re) in enumerate(rows):
            for j, (cs, ce) in enumerate(cols):
                block = grid[rs:re, cs:ce]
                if (np.any(block[0] != canvas) or np.any(block[-1] != canvas)
                        or np.any(block[:, 0] != canvas) or np.any(block[:, -1] != canvas)):
                    valid = False
                    break
                colors = set(map(int, np.unique(block))) - {canvas}
                if len(colors) > 1:
                    valid = False
                    break
                if colors:
                    color = next(iter(colors))
                    points = [(int(r) + rs, int(c) + cs) for r, c in np.argwhere(block == color)]
                    if not _connected(points):
                        valid = False
                        break
                    patches.append({"cell": (i, j), "color": color, "points": points})
            if not valid:
                break
        if valid and patches:
            candidates.append({"canvas": canvas, "separator": separator,
                               "rows": rows, "cols": cols, "patches": patches,
                               "row_bands": [(a + top, b + top) for a, b in rb],
                               "col_bands": [(a + left, b + left) for a, b in cb],
                               "bbox": (top, bottom + 1, left, right + 1),
                               "input_shape": grid.shape})
    if len(candidates) != 1:
        raise ValueError("input needs one unambiguous lattice with inset connected patches")
    return candidates[0]


def _footprint(points, rows, cols):
    """Explicit physical-pixel ray traversal, ignoring separator pixels."""
    row_at = {r: i for i, (a, b) in enumerate(rows) for r in range(a, b)}
    col_at = {c: j for j, (a, b) in enumerate(cols) for c in range(a, b)}
    hits = set()
    for r, c in points:
        while r < rows[-1][1] and c < cols[-1][1]:
            if r in row_at and c in col_at:
                hits.add((row_at[r], col_at[c]))
            r, c = r + 1, c + 1
    return hits


def _readout(meta, patches=None):
    patches = meta["patches"] if patches is None else patches
    result = np.full((len(meta["rows"]), len(meta["cols"])), meta["canvas"], dtype=np.int8)
    for patch in patches:
        for cell in _footprint(patch["points"], meta["rows"], meta["cols"]):
            if result[cell] not in (meta["canvas"], patch["color"]):
                raise ValueError("different colors illuminate the same compartment")
            result[cell] = patch["color"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve exclusively from the input's visible geometry and colors."""
    return _readout(_parse_lattice(input_grid))


def _interval_readout(meta):
    """Check ray/detector intersection algebraically, without tracing pixels."""
    out = np.full((len(meta["rows"]), len(meta["cols"])), meta["canvas"], dtype=np.int8)
    for i, (rs, re) in enumerate(meta["rows"]):
        for j, (cs, ce) in enumerate(meta["cols"]):
            colors = set()
            for patch in meta["patches"]:
                if any(max(0, rs - r, cs - c) <= min(re - 1 - r, ce - 1 - c)
                       for r, c in patch["points"]):
                    colors.add(patch["color"])
            _require(len(colors) <= 1, "detector has a color conflict")
            if colors:
                out[i, j] = next(iter(colors))
    return out


def _decoded(meta):
    out = np.full((len(meta["rows"]), len(meta["cols"])), meta["canvas"], dtype=np.int8)
    for p in meta["patches"]:
        out[p["cell"]] = p["color"]
    return out


def _logical_rays(meta):
    """Deliberately wrong shortcut: first compress, then use logical diagonals."""
    out = _decoded(meta)
    for p in meta["patches"]:
        r, c = p["cell"]
        while r < out.shape[0] and c < out.shape[1]:
            out[r, c] = p["color"]
            r, c = r + 1, c + 1
    return out


def _axis(rng, count, special=None):
    minimum = [3] * count
    if special is not None:
        minimum[special] = 5
    for _ in range(128):
        bands = ([int(rng.integers(1, 4))] * (count + 1) if rng.random() < .20
                 else list(map(int, rng.integers(1, 4, size=count + 1))))
        before, after = map(int, rng.integers(0, 3, size=2))
        budget = 30 - sum(bands) - before - after - sum(minimum)
        if budget >= 0:
            break
    else:
        bands, before, after = [1] * (count + 1), 0, 0
        budget = 30 - sum(bands) - sum(minimum)
    sizes = minimum.copy()
    extra = int(rng.integers(0, min(budget, sum(8 - x for x in sizes)) + 1))
    for _ in range(extra):
        choices = [i for i, n in enumerate(sizes) if n < 8]
        sizes[int(rng.choice(choices))] += 1
    br, gaps, cursor = [], [], before
    for i, band in enumerate(bands):
        br.append((cursor, cursor + band))
        cursor += band
        if i < count:
            gaps.append((cursor, cursor + sizes[i]))
            cursor += sizes[i]
    return cursor + after, br, gaps


def _base(rng, shape, roles, special=None):
    h, rb, rows = _axis(rng, shape[0], None if special is None else special[0])
    w, cb, cols = _axis(rng, shape[1], None if special is None else special[1])
    canvas, separator = roles
    grid = np.full((h, w), canvas, dtype=np.int8)
    for a, b in rb:
        grid[a:b, cb[0][0]:cb[-1][1]] = separator
    for a, b in cb:
        grid[rb[0][0]:rb[-1][1], a:b] = separator
    return grid, {"canvas": canvas, "separator": separator, "rows": rows, "cols": cols, "patches": []}


def _patch(rng, meta, cell, color, wide=False):
    rs, re = meta["rows"][cell[0]]
    cs, ce = meta["cols"][cell[1]]
    if wide:
        if rng.random() < .5:
            ph, pw = 1, int(rng.integers(2, ce - cs - 1))
        else:
            ph, pw = int(rng.integers(2, re - rs - 1)), 1
    else:
        ph = int(rng.integers(1, min(4, re - rs - 2) + 1))
        pw = int(rng.integers(1, min(4, ce - cs - 2) + 1))
    r = int(rng.integers(rs + 1, re - ph))
    c = int(rng.integers(cs + 1, ce - pw))
    return {"cell": cell, "color": color,
            "points": [(r + i, c + j) for i in range(ph) for j in range(pw)]}


def _paint(base, patches):
    grid = base.copy()
    for p in patches:
        for r, c in p["points"]:
            grid[r, c] = p["color"]
    return grid


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _center_only(meta):
    """Counterfactual point-source approximation, with upper/left center ties."""
    patches = []
    for patch in meta["patches"]:
        points = patch["points"]
        mr = sum(r for r, c in points) / len(points)
        mc = sum(c for r, c in points) / len(points)
        point = min(points, key=lambda q: ((q[0] - mr) ** 2 + (q[1] - mc) ** 2, q))
        patches.append({**patch, "points": [point]})
    return _readout(meta, patches)


def _contrast_pair(rng, palette):
    """Two inputs with identical compressed payloads but unequal ray readouts."""
    roles = tuple(palette[:2])
    for _ in range(1200):
        shape = (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
        cell = (int(rng.integers(1, shape[0] - 1)), int(rng.integers(1, shape[1] - 1)))
        base, meta = _base(rng, shape, roles, special=cell)
        main = _patch(rng, meta, cell, palette[2], wide=True)
        subset = {**main, "points": [main["points"][int(rng.integers(len(main["points"])))] ]}
        full_hits = _footprint(main["points"], meta["rows"], meta["cols"])
        small_hits = _footprint(subset["points"], meta["rows"], meta["cols"])
        if full_hits == small_hits or len(small_hits) < 2:
            continue
        if np.array_equal(_readout({**meta, "patches": [main]}),
                          _center_only({**meta, "patches": [main]})):
            continue
        # A multi-pixel source must teach ray thickness, and a physical ray
        # must teach that logical diagonal stepping is the wrong metric.
        if not any(j - cell[1] != i - cell[0] for i, j in full_hits):
            continue
        # Add a second, separately colored patch; favor a second extended ray.
        options = [(i, j) for i in range(shape[0]) for j in range(shape[1]) if (i, j) not in full_hits]
        rng.shuffle(options)
        secondary = None
        for other in options:
            for _ in range(3):
                candidate = _patch(rng, meta, other, palette[3])
                hits = _footprint(candidate["points"], meta["rows"], meta["cols"])
                if not hits & full_hits and (cell[0] - 1, cell[1] - 1) not in hits:
                    secondary = candidate
                    break
            if secondary is not None:
                break
        if secondary is None:
            continue
        result = []
        for primary in (main, subset):
            patches = [primary, secondary]
            candidate_meta = {**meta, "patches": patches}
            result.append({"input": _paint(base, patches), "output": _readout(candidate_meta)})
        if not np.array_equal(result[0]["output"], result[1]["output"]):
            if rng.random() < .5:
                result.reverse()
            return result
    raise RuntimeError("could not construct a ray-thickness training contrast")


def _random_pair(rng, shape, roles, payloads, used):
    for _ in range(160):
        base, meta = _base(rng, shape, roles)
        target = int(rng.integers(1, min(6, shape[0] * shape[1]) + 1))
        cells = [(i, j) for i in range(shape[0]) for j in range(shape[1])]
        rng.shuffle(cells)
        footprints = {}
        for cell in cells:
            if len(meta["patches"]) >= target:
                break
            for _ in range(8):
                color = int(rng.choice(payloads))
                patch = _patch(rng, meta, cell, color)
                hits = _footprint(patch["points"], meta["rows"], meta["cols"])
                if all(footprints.get(q, color) == color for q in hits):
                    meta["patches"].append(patch)
                    footprints.update({q: color for q in hits})
                    break
        if not meta["patches"]:
            continue
        output = _readout(meta)
        if np.array_equal(output, _decoded(meta)):
            continue
        grid = _paint(base, meta["patches"])
        if _key(grid) not in used:
            return {"input": grid, "output": output}
    raise RuntimeError("could not construct a distinct illuminated detector lattice")


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    train = _contrast_pair(rng, palette)
    used = {_key(p["input"]) for p in train}
    shapes = {p["output"].shape for p in train}
    puzzle = {"train": train, "test": []}
    for split, count in (("train", nt - 2), ("test", ne)):
        for _ in range(count):
            choices = [(r, c) for r in range(2, 7) for c in range(2, 7)
                       if split == "train" or (r, c) not in shapes]
            shape = choices[int(rng.integers(len(choices)))]
            # Reuse and swap role colors as well as drawing fresh roles.
            if rng.random() < .50:
                roles = tuple(palette[:2])
                if rng.random() < .25:
                    roles = roles[::-1]
            else:
                roles = tuple(map(int, rng.choice(10, size=2, replace=False)))
            available = [c for c in range(10) if c not in roles]
            pool = list(map(int, rng.choice(available, size=int(rng.integers(1, 6)), replace=False)))
            pair = _random_pair(rng, shape, roles, pool, used)
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
            if split == "train":
                shapes.add(shape)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "need exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, metas = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair needs input/output only")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in used, "duplicate input")
            used.add(_key(inp))
            meta = _parse_lattice(inp)
            metas[split].append(meta)
            _require(np.array_equal(out, _interval_readout(meta)), "wrong physical ray detector readout")
            _require(np.array_equal(out, solve(inp)), "input-only solve mismatch")
            _require(out.shape == (len(meta["rows"]), len(meta["cols"])), "one cell per detector required")
            for p in meta["patches"]:
                _require(out[p["cell"]] == p["color"], "source detector lost its own color")
            _require(not np.array_equal(out, _decoded(meta)), "pair must expose propagation beyond swatches")
            _require(meta["separator"] not in out, "separator leaked to readout")
    first, second = metas["train"][:2]
    _require(np.array_equal(_decoded(first), _decoded(second)), "first two examples must share logical payloads")
    _require(first["rows"] == second["rows"] and first["cols"] == second["cols"], "contrast geometry changed")
    _require(not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"]), "ray-thickness contrast missing")
    _require(any(not np.array_equal(_logical_rays(m), _readout(m)) for m in (first, second)), "logical-ray shortcut not distinguished")
    _require(any(not np.array_equal(_center_only(m), _readout(m)) for m in (first, second)), "point-source shortcut not distinguished")
    _require(any(p["cell"][0] > 0 and p["cell"][1] > 0
                 and _readout(m)[p["cell"][0] - 1, p["cell"][1] - 1] == m["canvas"]
                 for m in (first, second) for p in m["patches"]), "no forward-only witness")
    train_shapes = {(len(m["rows"]), len(m["cols"])) for m in metas["train"]}
    _require(all((len(m["rows"]), len(m["cols"])) not in train_shapes for m in metas["test"]), "test needs a new detector shape")


def self_test():
    """Bounded invariant, diversity, metamorphic, and count checks on 42 seeds."""
    # The same physical beam is read differently when a detector boundary
    # moves. This witnesses A's contribution without changing B's ray field.
    fixtures = []
    for divider in (6, 8):
        grid = np.zeros((13, 13), dtype=np.int8)
        grid[[0, 6, 12], :] = 5
        grid[:, [0, divider, 12]] = 5
        grid[3, 3] = 2
        fixtures.append(grid)
    _require(np.array_equal(solve(fixtures[0]), [[2, 0], [0, 2]]), "transparent separator fixture failed")
    _require(np.array_equal(solve(fixtures[1]), [[2, 0], [2, 2]]), "detector geometry has no effect")
    redundant = fixtures[0].copy()
    redundant[9, 9] = 2
    _require(np.array_equal(solve(redundant), solve(fixtures[0])), "same-color overlap must saturate")
    conflict = fixtures[0].copy()
    conflict[9, 10] = 3
    try:
        solve(conflict)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting detector colors accepted")
    late = fixtures[0].copy()
    late[3, 3], late[8, 8] = 0, 2
    _require(np.array_equal(solve(late), [[0, 0], [0, 2]]), "rays must not extend backward")
    shapes, sizes, structures, roles, counts = set(), set(), set(), set(), set()
    patch_areas, band_widths, features = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                meta = _parse_lattice(inp)
                shapes.add(out.shape)
                sizes.add(inp.shape)
                counts.add(len(meta["patches"]))
                roles.add((meta["canvas"], meta["separator"]))
                structures.add((inp.shape, tuple(meta["rows"]), tuple(meta["cols"]),
                                tuple((p["cell"], tuple(p["points"])) for p in meta["patches"])))
                patch_areas.update(len(p["points"]) for p in meta["patches"])
                band_widths.update(b - a for a, b in meta["row_bands"] + meta["col_bands"])
                if inp.shape[0] == inp.shape[1]:
                    features.add("square")
                colors = [p["color"] for p in meta["patches"]]
                if len(set(colors)) < len(colors):
                    features.add("repeated_color")
                if not np.array_equal(_logical_rays(meta), out):
                    features.add("physical_metric")
                top, bottom, left, right = meta["bbox"]
                if top == 0 or left == 0 or bottom == inp.shape[0] or right == inp.shape[1]:
                    features.add("zero_margin")
                if top or left or bottom < inp.shape[0] or right < inp.shape[1]:
                    features.add("visible_margin")
                # A bijective palette permutation cannot change the geometry.
                perm = np.array([(i + 3) % 10 for i in range(10)], dtype=np.int8)
                _require(np.array_equal(solve(perm[inp]), perm[out]), "color-role equivariance failed")
                # Adding canvas above/left shifts rays and windows together.
                if max(inp.shape) < 30:
                    padded = np.pad(inp, ((1, 0), (1, 0)), constant_values=meta["canvas"])
                    _require(np.array_equal(solve(padded), out), "translation invariance failed")
    _require(len(shapes) >= 18 and len(sizes) >= 50 and len(structures) >= 150, "insufficient structural diversity")
    _require(len(roles) >= 20 and len(counts) >= 5, "insufficient palette/count diversity")
    _require({1, 2, 3} <= band_widths and len(patch_areas) >= 7, "insufficient physical variation")
    _require({"square", "repeated_color", "physical_metric", "zero_margin", "visible_margin"} <= features, "missing geometry modes")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(91273, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(19013, 6, 3), generate_puzzle(19013, 6, 3)
    _require(all(_key(x[k]) == _key(y[k]) for s in ("train", "test")
                 for x, y in zip(a[s], b[s]) for k in ("input", "output")), "seed nondeterminism")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus explicit count cases)")
