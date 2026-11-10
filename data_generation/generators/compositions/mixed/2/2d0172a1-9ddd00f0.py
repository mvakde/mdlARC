"""Containment-normalized, coordinate-indexed cores in a compartment lattice.

The three colors have visible roles: full separator lines, compartment
background (the reserved final row/column), and the drawing color.  No puzzle
parameters are needed by solve.  See task_spec.md for the core/shell distinction.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             np.all((a >= 0) & (a <= 9)), "grid colors must be integers 0..9")
    return a


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        cells, stack = [], [(int(y), int(x))]
        seen[y, x] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= rr < h and 0 <= cc < w and mask[rr, cc]
                        and not seen[rr, cc]):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        part = np.zeros_like(mask, dtype=bool)
        rr, cc = zip(*cells)
        part[rr, cc] = True
        result.append(part)
    return result


def _enclosed(mask):
    """Actual enclosed space, not bounding-box containment."""
    wall = np.pad(mask, 1)
    seen = np.zeros_like(wall)
    stack = [(0, 0)]
    seen[0, 0] = True
    h, w = wall.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (0 <= rr < h and 0 <= cc < w and not wall[rr, cc]
                    and not seen[rr, cc]):
                seen[rr, cc] = True
                stack.append((rr, cc))
    return (~wall & ~seen)[1:-1, 1:-1]


def _bounds(mask):
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()+1), int(xs.max()+1)


def _canonical(n, depth, shape, bg, fg, hole):
    """Rectangular shells, their moats, and a separate n-by-n terminal core."""
    side = n + 4*depth
    _require(min(shape) >= side+1, "normalized contents do not fit")
    tile = np.full(shape, bg, dtype=np.int8)
    for level in range(depth):
        low, high = 2*level, side-2*level
        tile[low, low:high] = tile[high-1, low:high] = fg
        tile[low:high, low] = tile[low:high, high-1] = fg
    start = 2*depth
    tile[start:start+n, start:start+n] = fg
    if hole is not None:
        tile[start+hole[0], start+hole[1]] = bg
    return tile


def _inspect_tile(tile, n, i, j, bg, fg):
    """Recognize completed tiles before interpreting their core's little hole."""
    _require(np.all(tile[-1] == bg) and np.all(tile[:, -1] == bg),
             "each compartment needs a background final row and column")
    cap = min(4, (min(tile.shape)-1-n)//4)
    for depth in range(1, cap+1):
        if np.array_equal(tile, _canonical(n, depth, tile.shape, bg, fg, (i, j))):
            return dict(depth=depth, state="ready", nonrect=0,
                        blob_area=n*n-1, raw_core_origin=(2*depth, 2*depth))
    parts = _components(tile == fg)
    _require(parts, "entirely erased shells are not supported")
    labels = np.zeros(tile.shape, dtype=int)
    for index, part in enumerate(parts, 1):
        labels[part] = index
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]),
                 (labels[:-1, 1:], labels[1:, :-1])):
        _require(not np.any((a > 0) & (b > 0) & (a != b)), "touching objects")
    rings, blobs = [], []
    nonrect = 0
    for part in parts:
        inside = _enclosed(part)
        if inside.any():
            _require(len(_components(inside)) == 1, "a shell has multiple chambers")
            rings.append((part, inside))
            y, x, yy, xx = _bounds(part)
            local = part[y:yy, x:xx]
            nonrect += not (local[0].all() and local[-1].all()
                            and local[:, 0].all() and local[:, -1].all())
        else:
            blobs.append(part)
    depth = len(rings)
    _require(1 <= depth <= cap and len(blobs) <= 1,
             "expected a fitting containment chain and at most one terminal blob")
    rings.sort(key=lambda item: int(item[1].sum()), reverse=True)
    for index in range(depth-1):
        _require(np.all(rings[index][1][rings[index+1][0]]),
                 "shells must form one containment chain")
    if blobs:
        _require(np.all(rings[-1][1][blobs[0]]), "blob is outside the innermost shell")
        y, x, yy, xx = _bounds(blobs[0])
        _require(int(blobs[0].sum()) >= (yy-y)*(xx-x)/2, "terminal blob is not compact")
    y, x, _, _ = _bounds(rings[-1][0])
    return dict(depth=depth, state="blob" if blobs else "empty",
                nonrect=int(nonrect), blob_area=int(blobs[0].sum()) if blobs else 0,
                raw_core_origin=(y+2, x+2))


def _parse(input_grid):
    a = _grid(input_grid)
    colors = [int(c) for c in np.unique(a)]
    _require(len(colors) == 3, "expected exactly three geometric color roles")
    candidates = []
    for color in colors:
        rows = np.flatnonzero(np.all(a == color, axis=1)).tolist()
        cols = np.flatnonzero(np.all(a == color, axis=0)).tolist()
        if rows and cols:
            candidates.append((color, rows, cols))
    _require(len(candidates) == 1, "separator role is ambiguous")
    sep, rows, cols = candidates[0]
    n = len(rows)+1
    _require(n in (2, 3) and len(cols)+1 == n, "expected a 2x2 or 3x3 lattice")
    _require(rows[0] > 0 and rows[-1] < a.shape[0]-1 and
             cols[0] > 0 and cols[-1] < a.shape[1]-1, "outer separator")
    rb = list(zip([0]+[r+1 for r in rows], rows+[a.shape[0]]))
    cb = list(zip([0]+[c+1 for c in cols], cols+[a.shape[1]]))
    _require(all(b > t for t, b in rb+cb), "empty compartment")
    bg = int(a[rb[0][1]-1, cb[0][1]-1])
    _require(bg != sep, "background marker is a separator")
    fg = next(c for c in colors if c not in (bg, sep))
    records = []
    for i, (top, bottom) in enumerate(rb):
        for j, (left, right) in enumerate(cb):
            tile = a[top:bottom, left:right]
            _require(not np.any(tile == sep), "separator pigment inside a compartment")
            rec = _inspect_tile(tile, n, i, j, bg, fg)
            rec.update(i=i, j=j, bounds=(top, left, bottom, right), shape=tile.shape)
            records.append(rec)
    return dict(n=n, bg=bg, fg=fg, sep=sep, rows=rows, cols=cols,
                row_sizes=[b-t for t, b in rb], col_sizes=[r-l for l, r in cb],
                records=records, input=a)


def _answer(info, *, indexing="identity", flatten=False, no_holes=False):
    out = info["input"].astype(np.int8, copy=True)
    n = info["n"]
    for rec in info["records"]:
        i, j = rec["i"], rec["j"]
        hole = (i, j)
        if indexing == "shared":
            hole = (0, 0)
        elif indexing == "transpose":
            hole = (j, i)
        elif indexing == "reverse":
            hole = (n-1-i, n-1-j)
        elif indexing == "row_only":
            hole = (i, 0)
        elif indexing == "column_only":
            hole = (0, j)
        top, left, bottom, right = rec["bounds"]
        out[top:bottom, left:right] = _canonical(
            n, 1 if flatten else rec["depth"], rec["shape"],
            info["bg"], info["fg"], None if no_holes else hole)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all color roles, lattice coordinates and containment from input."""
    return _answer(_parse(input_grid))


def _frame(rng, child, pads, force_cut=False):
    top, bottom, left, right = pads
    h, w = child.shape
    shape = (h+top+bottom, w+left+right)
    filled = np.ones(shape, dtype=bool)
    yy, xx = np.indices(shape)
    distances = (yy+xx, yy+(shape[1]-1-xx),
                 (shape[0]-1-yy)+xx, (shape[0]-1-yy)+(shape[1]-1-xx))
    caps = (min(top, left)-2, min(top, right)-2,
            min(bottom, left)-2, min(bottom, right)-2)
    for k, (distance, cap) in enumerate(zip(distances, caps)):
        cut = int(rng.integers(min(3, cap)+1))
        if force_cut and k == 0:
            cut = max(1, cut)
        filled[distance < cut] = False
    # A bay lies in a side's safe corridor and cannot touch the child.
    side = int(rng.integers(4))
    if pads[side] >= 4 and rng.random() < .65:
        indent = int(rng.integers(1, pads[side]-2))
        if side < 2:
            x0, x1 = left, left+max(1, w//2)
            if side == 0:
                filled[:indent, x0:x1] = False
            else:
                filled[-indent:, x0:x1] = False
        else:
            y0, y1 = top, top+max(1, h//2)
            if side == 2:
                filled[y0:y1, :indent] = False
            else:
                filled[y0:y1, -indent:] = False
    padded = np.pad(filled, 1)
    eroded = np.ones(shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy+shape[0], dx:dx+shape[1]]
    result = filled & ~eroded
    result[top:top+h, left:left+w] |= child
    return result


def _raw_tile(rng, n, depth, shape, bg, fg, empty, force_distort=False):
    # The last row and column are a visible background key, never drawing area.
    limits = [shape[0]-1, shape[1]-1]
    max_core = [min(4, v-4*depth) for v in limits]
    _require(min(max_core) >= 1, "metric diagram does not fit")
    dims = [int(rng.integers(1, cap+1)) for cap in max_core]
    if force_distort:
        # Keep room for unequal clearances without fixing every test blob
        # to a dot. Terminal metric size never determines normalized core size.
        dims = [int(rng.integers(1, max(1, cap-1)+1)) for cap in max_core]
    child = np.zeros(dims, dtype=bool) if empty else np.ones(dims, dtype=bool)
    if not empty and min(dims) >= 2 and rng.random() < .65:
        # Boundary-corner peeling retains a connected, compact, hole-free blob.
        child[int(rng.choice((0, dims[0]-1))), int(rng.choice((0, dims[1]-1)))] = False
    pads = np.full((depth, 4), 2, dtype=int)
    for axis, sides in ((0, (0, 1)), (1, (2, 3))):
        budget = limits[axis]-dims[axis]-4*depth
        if force_distort and budget:
            pads[-1, sides[0]] += 1
            budget -= 1
        extras = int(rng.integers(budget+1))
        for _ in range(extras):
            pads[int(rng.integers(depth)), int(rng.choice(sides))] += 1
    for level in range(depth):
        force = (force_distort and level == depth-1 and
                 pads[level, 0] >= 3 and pads[level, 2] >= 3)
        child = _frame(rng, child, pads[level].tolist(), force_cut=force)
    top = int(rng.integers(limits[0]-child.shape[0]+1))
    left = int(rng.integers(limits[1]-child.shape[1]+1))
    tile = np.full(shape, bg, dtype=np.int8)
    tile[top:top+child.shape[0], left:left+child.shape[1]][child] = fg
    return tile


def _sizes(rng, n, depths, uniform=False):
    rows = [n+4*int(max(depths[i]))+1 for i in range(n)]
    cols = [n+4*int(max(depths[:, j]))+1 for j in range(n)]
    if uniform:
        # Equal metric panel sizes with different depths defeat a box-size rule.
        rows = [int(rng.integers(max(rows), 15))]*n
        cols = [int(rng.integers(max(cols), 15))]*n
    else:
        for sizes in (rows, cols):
            budget = 30-(sum(sizes)+n-1)
            _require(budget >= 0, "layout exceeds ARC bounds")
            for _ in range(int(rng.integers(budget+1))):
                sizes[int(rng.integers(n))] += 1
    return rows, cols


def _assemble(n, row_sizes, col_sizes, sep, bg):
    shape = (sum(row_sizes)+n-1, sum(col_sizes)+n-1)
    grid = np.full(shape, sep, dtype=np.int8)
    bounds = []
    top = 0
    for i, height in enumerate(row_sizes):
        left = 0
        for j, width in enumerate(col_sizes):
            grid[top:top+height, left:left+width] = bg
            bounds.append((i, j, top, left, top+height, left+width))
            left += width+1
        top += height+1
    return grid, bounds


def _make_pair(rng, n, test, palette, witness=False):
    bg, fg, sep = palette
    depths = np.ones((n, n), dtype=int)
    if witness:
        depths[0, 0] = depths[1, 1] = 2
        depths = np.rot90(depths, int(rng.integers(2)))
    elif test:
        depths[int(rng.integers(n)), int(rng.integers(n))] = (
            int(rng.choice((3, 4))) if n == 2 else 2)
        if n == 2 and depths.max() == 3 and rng.random() < .6:
            shallow = list(zip(*np.where(depths == 1)))
            y, x = shallow[int(rng.integers(len(shallow)))]
            depths[y, x] = 2
    elif n == 2:
        # Extra training examples still do not introduce test depths 3 and 4.
        depths = rng.integers(1, 3, size=(2, 2))
    rows, cols = _sizes(rng, n, depths, uniform=witness)
    inp, bounds = _assemble(n, rows, cols, sep, bg)
    out = inp.copy()
    states = np.full((n, n), "empty", dtype="U5")
    if witness:
        deep = list(zip(*np.where(depths == 2)))
        shallow = list(zip(*np.where(depths == 1)))
        states[deep[0]] = "blob"
        states[shallow[0]] = "ready"
        states[shallow[1]] = "blob"
    else:
        order = rng.permutation(n*n)
        if test:
            count = int(rng.integers(1, max(2, n*n//3+1)))
            states.flat[order[:count]] = "blob"
        else:
            ready = int(rng.integers(1, max(2, n*n//3+1)))
            blob = int(rng.integers(1, n*n-ready))
            states.flat[order[:ready]] = "ready"
            states.flat[order[ready:ready+blob]] = "blob"
    for i, j, top, left, bottom, right in bounds:
        shape, depth = (bottom-top, right-left), int(depths[i, j])
        target = _canonical(n, depth, shape, bg, fg, (i, j))
        out[top:bottom, left:right] = target
        if states[i, j] == "ready":
            source = target
        else:
            source = _raw_tile(rng, n, depth, shape, bg, fg,
                               empty=states[i, j] == "empty",
                               force_distort=witness or test)
        inp[top:bottom, left:right] = source
    return {"input": inp, "output": out}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and
             not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    palette_mode = int(rng.integers(3))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            n = (2 if index == 0 else 3 if index == 1 else int(rng.choice((2, 3))))
            if split == "test":
                n = int(rng.choice((2, 3)))
            if palette_mode == 0:
                palette = base
            elif palette_mode == 1:
                shift = (index+(split == "test")) % 3
                palette = base[shift:]+base[:shift]
            else:
                palette = tuple(int(c) for c in rng.choice(10, 3, replace=False))
            for _ in range(24):
                pair = _make_pair(rng, n, split == "test", palette,
                                  witness=split == "train" and index == 0)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input search exhausted")
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _count("num_train", a, 2, 10)
    _count("num_test", b, 1, 4)
    _require(a > b, "training must outnumber tests")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input/output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "separator canvas must be retained")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            _require(np.array_equal(_answer(info), out), "incorrect normalized/indexed output")
            n, bg, fg = info["n"], info["bg"], info["fg"]
            _require(np.array_equal(inp == info["sep"], out == info["sep"]),
                     "separator changed")
            for rec in info["records"]:
                i, j, depth = rec["i"], rec["j"], rec["depth"]
                top, left, bottom, right = rec["bounds"]
                tile = out[top:bottom, left:right]
                offset = 2*depth
                core = tile[offset:offset+n, offset:offset+n]
                missing = np.argwhere(core == bg)
                _require(missing.tolist() == [[i, j]] and int((core == fg).sum()) == n*n-1,
                         "core must have exactly its macro-coordinate missing cell")
                for level in range(depth):
                    low, high = 2*level, n+4*depth-2*level
                    _require(np.all(tile[low, low:high] == fg) and
                             np.all(tile[high-1, low:high] == fg) and
                             np.all(tile[low:high, low] == fg) and
                             np.all(tile[low:high, high-1] == fg), "frame is incomplete")
                    _require(np.all(tile[low+1, low+1:high-1] == bg) and
                             np.all(tile[high-2, low+1:high-1] == bg) and
                             np.all(tile[low+1:high-1, low+1] == bg) and
                             np.all(tile[low+1:high-1, high-2] == bg), "moat was filled")
                if rec["state"] == "ready":
                    _require(np.array_equal(inp[top:bottom, left:right], tile),
                             "completed normalized contents changed")
            _require(not np.array_equal(inp, out), "pair is an identity")
            infos[split].append(info)
    _require({q["n"] for q in infos["train"]} == {2, 3}, "both lattice sizes need demonstrations")
    witnesses = [q for q in infos["train"] if q["n"] == 2 and
                 len({r["shape"] for r in q["records"]}) == 1 and
                 {r["depth"] for r in q["records"]} == {1, 2} and
                 {r["state"] for r in q["records"]} == {"ready", "blob", "empty"}]
    _require(witnesses, "missing same-size/different-depth normalization witness")
    for q in infos["test"]:
        trained = max(r["depth"] for t in infos["train"] if t["n"] == q["n"]
                      for r in t["records"])
        _require(max(r["depth"] for r in q["records"]) > trained,
                 "test must recombine the rule with greater containment depth")
        _require(all(r["state"] != "ready" for r in q["records"]),
                 "test should require normalization throughout")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds, bounded redraws and explicit concept-removal checks."""
    rng = np.random.default_rng(196608)
    hashes, shapes, depth_plans, palettes, orders, depths, states = (set() for _ in range(7))
    nonrect = pairs = ready = empty = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        hashes.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                pairs += 1
                shapes.add(pair["input"].shape)
                orders.add(info["n"])
                palettes.add((info["bg"], info["fg"], info["sep"]))
                plan = tuple(r["depth"] for r in info["records"])
                depth_plans.add((info["n"], plan))
                for r in info["records"]:
                    depths.add(r["depth"])
                    states.add(r["state"])
                    nonrect += r["nonrect"]
                    ready += r["state"] == "ready"
                    empty += r["state"] == "empty"
                assert np.array_equal(solve(pair["output"]), pair["output"])
                assert not np.array_equal(_answer(info, no_holes=True), pair["output"])
                for mode in ("shared", "transpose", "reverse", "row_only", "column_only"):
                    assert not np.array_equal(_answer(info, indexing=mode), pair["output"])
        witness = next(p for p in puzzle["train"] if
                       _parse(p["input"])["n"] == 2 and
                       len({r["shape"] for r in _parse(p["input"])["records"]}) == 1 and
                       {r["depth"] for r in _parse(p["input"])["records"]} == {1, 2})
        assert not np.array_equal(_answer(_parse(witness["input"]), flatten=True), witness["output"])
        # Causal intervention: at the SAME macro coordinate and with the SAME
        # compartment dimensions, add a shell. A must move B's absolute hole
        # by (2,2). Palette, lattice, and all other compartments stay fixed.
        wi = _parse(witness["input"])
        rec = next(r for r in wi["records"] if r["depth"] == 1)
        top, left, bottom, right = rec["bounds"]
        intervention = witness["input"].copy()
        intervention[top:bottom, left:right] = _raw_tile(
            rng, 2, 2, rec["shape"], wi["bg"], wi["fg"], empty=False)
        changed = solve(intervention)
        old_hole = (top+2+rec["i"], left+2+rec["j"])
        new_hole = (old_hole[0]+2, old_hole[1]+2)
        assert changed[new_hole] == wi["bg"]
        assert not np.array_equal(changed, witness["output"])
        for other in wi["records"]:
            if (other["i"], other["j"]) != (rec["i"], rec["j"]):
                t, l, b, r = other["bounds"]
                assert np.array_equal(changed[t:b, l:r], witness["output"][t:b, l:r])
        test = puzzle["test"][0]
        info = _parse(test["input"])
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[test["input"]]), permutation[test["output"]])
        assert np.array_equal(solve(test["input"].T), test["output"].T)
        # Independently redraw the same topologies, changing blob sizes,
        # frame geometry, clearances and placement without changing the answer.
        redraw = test["input"].copy()
        for rec in info["records"]:
            top, left, bottom, right = rec["bounds"]
            redraw[top:bottom, left:right] = _raw_tile(
                rng, info["n"], rec["depth"], rec["shape"], info["bg"], info["fg"],
                empty=bool(rng.integers(2)))
        assert np.array_equal(solve(redraw), test["output"])
        if seed == 0:
            assert _digest(generate_puzzle(seed)) == _digest(puzzle)
            bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()}
                       for p in puzzle[s]] for s in puzzle}
            r = info["records"][0]
            y, x = r["bounds"][:2]
            bad["test"][0]["output"][y+2*r["depth"]+r["i"], x+2*r["depth"]+r["j"]] = info["fg"]
            try:
                validate_generated_puzzle(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("deleted coordinate hole accepted")
    assert len(hashes) == 42 and len(shapes) >= 12 and len(depth_plans) >= 12
    assert len(palettes) >= 30 and orders == {2, 3} and depths == {1, 2, 3, 4}
    assert states == {"ready", "blob", "empty"} and nonrect > 42
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(hashes),
                input_shapes=len(shapes), depth_plans=len(depth_plans),
                palettes=len(palettes), orders=sorted(orders), depths=sorted(depths),
                nonrectangular_shells=nonrect, ready_compartments=ready,
                empty_cores=empty, metamorphic_checks=126,
                shell_interventions=42, coordinate_alternatives=5,
                independent_solver=False)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
