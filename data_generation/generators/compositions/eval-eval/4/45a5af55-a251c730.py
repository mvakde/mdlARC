"""Keyed motifs grow shared, stripe-coded contours in a cropped target.

Only the input supplies the palette, layer code, dictionary, and placements.
No parent generator is imported. Labels use dilation; solve uses distances.
"""

from collections import Counter
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=False)


def _components(mask):
    todo = set(zip(*np.nonzero(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _D4:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(sorted(cells))
    return result


def _bounds(cells):
    rows, cols = zip(*[(p[0], p[1]) for p in cells])
    return min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _crop(grid, box):
    r, c, h, w = box
    return grid[r:r+h, c:c+w]


def _inside(p, box):
    r, c, h, w = box
    return r <= p[0] < r+h and c <= p[1] < c+w


def _parse(input_grid):
    """Identify the three geometric roles without consulting an output."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    if not (np.all(grid[0] == bg) and np.all(grid[-1] == bg)
            and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)):
        raise ValueError("the exterior rim must be uniform background")
    components = _components(grid != bg)
    frames = []
    for cells in components:
        box = _bounds(cells)
        r, c, h, w = box
        if h < 7 or w < 7 or len(cells) != 2*h + 2*w - 4:
            continue
        if len({int(grid[p]) for p in cells}) != 1:
            continue
        if all(y in (r, r+h-1) or x in (c, c+w-1) for y, x in cells):
            frames.append((box, cells))
    if len(frames) != 1:
        raise ValueError("expected one isolated rectangular target frame")
    box, frame_cells = frames[0]
    frame_set = set(frame_cells)
    anchors, outside = [], []
    for cells in components:
        if set(cells) == frame_set:
            continue
        if any(_inside(p, box) for p in cells):
            if not all(_inside(p, box) for p in cells) or len(cells) != 1:
                raise ValueError("the target contains only isolated anchors")
            r, c = cells[0]
            anchors.append((r-box[0], c-box[1], int(grid[r, c])))
        else:
            outside.append(cells)
    codes = []
    for cells in outside:
        cb = _bounds(cells)
        patch = _crop(grid, cb)
        if (cb[2] >= 2 and cb[3] >= 2 and len(cells) == patch.size
                and np.all(patch == patch[:, :1]) and len(np.unique(patch)) >= 2):
            codes.append((cb, cells))
    if len(codes) != 1:
        raise ValueError("expected one filled strip of uniform horizontal rows")
    code_box, code_cells = codes[0]
    code = tuple(int(c) for c in _crop(grid, code_box)[:, 0])
    code_set = set(code_cells)
    templates, repetitions = {}, Counter()
    for cells in outside:
        if set(cells) == code_set:
            continue
        counts = Counter(int(grid[p]) for p in cells)
        singletons = [c for c, n in counts.items() if n == 1]
        if len(cells) < 3 or len(singletons) != 1:
            raise ValueError("a reference motif needs exactly one singleton key color")
        key = singletons[0]
        ar, ac = next(p for p in cells if grid[p] == key)
        signature = tuple(sorted((r-ar, c-ac, int(grid[r, c])) for r, c in cells))
        if key in templates and templates[key] != signature:
            raise ValueError("conflicting definitions of one key")
        templates[key] = signature
        repetitions[key] += 1
    if not anchors or not templates or any(k not in templates for _, _, k in anchors):
        raise ValueError("every requested anchor needs a visible reference")
    if any(color in templates for t in templates.values() for dr, dc, color in t
           if (dr, dc) != (0, 0)):
        raise ValueError("decoration colors cannot be dictionary keys")
    target = _crop(grid, box).copy()
    base = np.zeros(target.shape, dtype=bool)
    painted = np.full(target.shape, bg, dtype=np.int8)
    for ar, ac, key in anchors:
        for dr, dc, color in templates[key]:
            r, c = ar+dr, ac+dc
            if not (1 <= r < target.shape[0]-1 and 1 <= c < target.shape[1]-1):
                raise ValueError("a completed motif may not cross the frame")
            if base[r, c]:
                raise ValueError("base motif footprints must be disjoint")
            if target[r, c] != (key if (dr, dc) == (0, 0) else bg):
                raise ValueError("a stamp would overwrite another anchor")
            base[r, c] = True
            painted[r, c] = color
    # Separation is part of this family; halo overlap is deliberately permitted.
    if len(_components(base)) != len(anchors):
        raise ValueError("base motifs must remain separated")
    return {"grid": grid, "bg": bg, "box": box, "target": target,
            "code": code, "code_box": code_box, "templates": templates,
            "repetitions": dict(repetitions), "anchors": anchors,
            "base": base, "painted": painted}


def _distance(mask, manhattan=False):
    rows, cols = np.indices(mask.shape)
    distance = np.full(mask.shape, 99, dtype=int)
    for r, c in zip(*np.nonzero(mask)):
        dr, dc = np.abs(rows-r), np.abs(cols-c)
        distance = np.minimum(distance, dr+dc if manhattan else np.maximum(dr, dc))
    return distance


def _from_distance(info, code=None, base=None, painted=None, manhattan=False):
    code = info["code"] if code is None else tuple(code)
    base = info["base"] if base is None else base
    painted = info["painted"] if painted is None else painted
    out = info["target"].copy()
    out[1:-1, 1:-1] = info["bg"]
    distance = _distance(base, manhattan)
    for d in range(1, len(code)+1):
        m = distance[1:-1, 1:-1] == d
        out[1:-1, 1:-1][m] = code[-d]
    out[base] = painted[base]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate keyed motifs, then color minimum square-distance shells."""
    return _from_distance(_parse(input_grid))


def _dilate(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    result = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            result |= padded[dr:dr+h, dc:dc+w]
    return result


def _label_by_growth(info):
    """Construct labels by simultaneous eight-neighbor growth, not distances."""
    out = info["target"].copy()
    reached = info["base"].copy()
    interior = np.zeros_like(reached)
    interior[1:-1, 1:-1] = True
    for color in reversed(info["code"]):
        enlarged = _dilate(reached) & interior
        out[enlarged & ~reached] = color
        reached = enlarged
    out[info["base"]] = info["painted"][info["base"]]
    return out


def _shortcut(info, kind):
    if kind == "omit_layers":
        out = info["target"].copy()
        out[info["base"]] = info["painted"][info["base"]]
        return out
    if kind == "anchors_only":
        base = np.zeros_like(info["base"])
        painted = np.full_like(info["painted"], info["bg"])
        for r, c, key in info["anchors"]:
            base[r, c], painted[r, c] = True, key
        return _from_distance(info, base=base, painted=painted)
    if kind == "one_template":
        base = np.zeros_like(info["base"])
        painted = np.full_like(info["painted"], info["bg"])
        template = info["templates"][info["anchors"][0][2]]
        for ar, ac, key in info["anchors"]:
            for dr, dc, color in template:
                r, c = ar+dr, ac+dc
                if 1 <= r < base.shape[0]-1 and 1 <= c < base.shape[1]-1:
                    base[r, c] = True
                    painted[r, c] = key if (dr, dc) == (0, 0) else color
        for r, c, key in info["anchors"]:
            painted[r, c] = key
        return _from_distance(info, base=base, painted=painted)
    if kind == "reverse":
        return _from_distance(info, code=info["code"][::-1])
    if kind == "collapse":
        code = [info["code"][0]]
        for color in info["code"][1:]:
            if color != code[-1]:
                code.append(color)
        return _from_distance(info, code=code)
    if kind == "manhattan":
        return _from_distance(info, manhattan=True)
    if kind == "boxes":
        base = np.zeros_like(info["base"])
        painted = info["painted"].copy()
        for ar, ac, key in info["anchors"]:
            lr, lc, h, w = _bounds(info["templates"][key])
            base[ar+lr:ar+lr+h, ac+lc:ac+lc+w] = True
        # Keep visible motifs but use the filled boxes to define growth.
        out = _from_distance(info, base=base, painted=painted)
        return out
    if kind == "overwrite":
        out = info["target"].copy()
        rows, cols = np.indices(out.shape)
        for ar, ac, key in info["anchors"]:
            mask = np.zeros_like(info["base"])
            for dr, dc, _ in info["templates"][key]:
                mask[ar+dr, ac+dc] = True
            dist = _distance(mask)
            for d, color in enumerate(reversed(info["code"]), 1):
                m = (dist == d) & (rows > 0) & (rows < out.shape[0]-1) & (cols > 0) & (cols < out.shape[1]-1)
                out[m] = color
        out[info["base"]] = info["painted"][info["base"]]
        return out
    raise ValueError(kind)


_ALTERNATIVES = ("omit_layers", "anchors_only", "one_template", "reverse", "collapse",
                 "manhattan", "boxes", "overwrite")


def _shape(rng):
    """Grow fresh polyominoes, keeping a nonrectangular silhouette."""
    for _ in range(80):
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        n = int(rng.integers(3, min(12, h*w)))
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        style = int(rng.integers(3))
        while len(cells) < n:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4
                               if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            neighbor_count = np.array([sum((r+dr, c+dc) in cells for dr, dc in _D4)
                                       for r, c in frontier], dtype=float)
            weights = (np.ones(len(frontier)), neighbor_count**2, 1/neighbor_count**2)[style]
            cells.add(frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))])
        _, _, bh, bw = _bounds(cells)
        if len(cells) == bh*bw:
            continue
        # Arbitrary occupied anchors, including extremal and internal ones.
        ar, ac = _pick(rng, sorted(cells))
        return tuple(sorted((r-ar, c-ac) for r, c in cells))
    return ((0, 0), (1, 0), (1, 1))


def _make_templates(rng, keys, decorations, require_different=False):
    templates = {}
    for key in keys:
        for _ in range(80):
            shape = _shape(rng)
            if not require_different or shape not in [tuple((r, c) for r, c, _ in t) for t in templates.values()]:
                break
        offsets = [p for p in shape if p != (0, 0)]
        count = int(rng.integers(1, min(3, len(decorations), len(offsets)//2)+1))
        colors = [int(c) for c in rng.choice(decorations, count, replace=False)]
        assignment = colors*2 + [_pick(rng, colors) for _ in range(len(offsets)-2*count)]
        rng.shuffle(assignment)
        templates[int(key)] = tuple(sorted([(0, 0, int(key))]
            + [(r, c, int(color)) for (r, c), color in zip(offsets, assignment)]))
    return templates


def _pack(items, width):
    r, c, row_h = 0, 0, 0
    positions = []
    for item in items:
        h, w = item["shape"]
        if w > width:
            return None
        if c+w > width:
            r, c, row_h = r+row_h+1, 0, 0
        positions.append((r, c))
        c, row_h = c+w+1, max(row_h, h)
    return r+row_h, positions


def _scene_geometry(rng, items, target_shape):
    th, tw = target_shape
    options = []
    for axis in (0, 1):
        for gap in (1, 2, 3):
            for width in range(2, 29):
                packed = _pack(items, width)
                if packed is None:
                    continue
                height, places = packed
                if height > 28:
                    continue
                required = [max(height, th)+2, max(width, tw)+2]
                required[axis] = (height+th if axis == 0 else width+tw)+gap+2
                if max(required) <= 30:
                    options.append((axis, gap, height, width, places, required))
    if not options:
        return None
    axis, gap, bh, bw, places, required = _pick(rng, options)
    canvas = [int(rng.integers(n, min(31, n+4))) for n in required]
    bank_shape = (bh, bw)
    positions = [[0, 0], [0, 0]]
    shapes = [bank_shape, target_shape]
    first = int(rng.integers(2))
    occupied_length = shapes[0][axis]+shapes[1][axis]+gap
    start = int(rng.integers(1, canvas[axis]-occupied_length))
    positions[first][axis] = start
    positions[1-first][axis] = start+shapes[first][axis]+gap
    for i in (0, 1):
        other = 1-axis
        positions[i][other] = int(rng.integers(1, canvas[other]-shapes[i][other]))
    bank = positions[0]
    target_box = tuple(positions[1]+list(target_shape))
    return tuple(canvas), [(r+bank[0], c+bank[1]) for r, c in places], target_box


def _placements(rng, templates, ids, shape, crowded):
    h, w = shape
    for _ in range(35):
        occupied, places = set(), []
        for key in ids:
            t = templates[key]
            lr, lc, bh, bw = _bounds(t)
            candidates = [(r, c) for r in range(1-lr, h-lr-bh)
                          for c in range(1-lc, w-lc-bw)]
            rng.shuffle(candidates)
            if crowded and occupied:
                # Randomized among nearby fits; no fixed anchor lattice.
                candidates.sort(key=lambda a: min(max(abs(a[0]-r), abs(a[1]-c)) for r, c in occupied))
            for ar, ac in candidates:
                footprint = {(ar+dr, ac+dc) for dr, dc, _ in t}
                if any((r+dr, c+dc) in occupied for r, c in footprint
                       for dr, dc in ((0, 0),)+_D4):
                    continue
                occupied.update(footprint)
                places.append((ar, ac, key))
                break
            else:
                break
        if len(places) == len(ids):
            return places
    return None


def _sample_code(rng, colors, length, evidence):
    if evidence:
        # Both traversal direction and retaining duplicate rows are observable.
        a, b = [int(c) for c in rng.choice(colors, 2, replace=False)]
        code = [a, a] + [b]*(length-2)
        if length >= 4 and rng.random() < .5:
            code[-1] = int(_pick(rng, [c for c in colors if c not in (a, b)]))
        return code
    count = int(rng.integers(2, min(5, length)+1))
    palette = [int(c) for c in rng.choice(colors, count, replace=False)]
    code = list(rng.permutation(palette))
    while len(code) < length:
        color = _pick(rng, palette)
        at = int(rng.integers(len(code)+1))
        code.insert(at, color)
    return [int(c) for c in code]


def _make_pair(rng, ntypes, length, target_style, evidence, seen):
    for attempt in range(160):
        palette = [int(c) for c in rng.permutation(10)]
        bg, keys = palette[0], palette[1:1+ntypes]
        decorations = palette[1+ntypes:]
        templates = _make_templates(rng, keys, decorations, require_different=evidence == 2)
        code = _sample_code(rng, palette[1:], length, evidence == 1)
        code_width = int(rng.integers(2, 6))
        ref_ids = [k for k in keys for _ in range(2 if evidence else int(rng.integers(1, 3)))]
        items = [{"kind": "code", "shape": (length, code_width)}]
        for key in ref_ids:
            _, _, h, w = _bounds(templates[key])
            items.append({"kind": "motif", "shape": (h, w), "key": key})
        rng.shuffle(items)
        if target_style == "elongated":
            shape = (int(rng.integers(9, 14)), int(rng.integers(19, 25)))
            if rng.random() < .5:
                shape = shape[::-1]
        elif target_style == "square":
            side = int(rng.integers(12, 22))
            shape = (side, side+int(rng.integers(-1, 2)))
        else:
            shape = (int(rng.integers(11, 24)), int(rng.integers(11, 24)))
        if evidence == 1 and target_style != "square":
            shape = (int(rng.integers(16, 23)), int(rng.integers(16, 23)))
        geometry = _scene_geometry(rng, items, shape)
        if geometry is None:
            # Space, not the split, determines feasible layouts.
            if target_style == "square":
                side = int(rng.integers(12, 16))
                shape = (side, side+int(rng.integers(-1, 2)))
            elif target_style == "elongated":
                shape = (int(rng.integers(9, 13)), int(rng.integers(19, 23)))
                if rng.random() < .5:
                    shape = shape[::-1]
            else:
                shape = (int(rng.integers(10, 14)), int(rng.integers(14, 22)))
            geometry = _scene_geometry(rng, items, shape)
        if geometry is None:
            continue
        canvas_shape, positions, box = geometry
        target_count = (2 if evidence == 1 else max(3, ntypes) if evidence == 2
                        else int(rng.integers(ntypes, min(7, ntypes+4))))
        ids = keys + [_pick(rng, keys) for _ in range(target_count-ntypes)]
        rng.shuffle(ids)
        places = _placements(rng, templates, ids, shape, evidence == 1 or rng.random() < .55)
        if places is None:
            continue
        grid = np.full(canvas_shape, bg, dtype=np.int8)
        target = _crop(grid, box)
        frame = int(_pick(rng, palette[1:]))
        target[0], target[-1], target[:, 0], target[:, -1] = frame, frame, frame, frame
        for (r, c), item in zip(positions, items):
            h, w = item["shape"]
            if item["kind"] == "code":
                grid[r:r+h, c:c+w] = np.asarray(code)[:, None]
            else:
                t = templates[item["key"]]
                lr, lc, _, _ = _bounds(t)
                for dr, dc, color in t:
                    grid[r+dr-lr, c+dc-lc] = color
        for r, c, key in places:
            target[r, c] = key
        signature = _grid_key(grid)
        if signature in seen:
            continue
        try:
            info = _parse(grid)
        except ValueError:
            continue
        out = _label_by_growth(info)
        if evidence == 1:
            if any(np.array_equal(_shortcut(info, k), out) for k in _ALTERNATIVES if k != "one_template"):
                continue
            # This demonstration shows the finite halo extent and the outer band.
            dist = _distance(info["base"])[1:-1, 1:-1]
            if dist.max() <= length or not all(np.any(dist == d) for d in range(1, length+1)):
                continue
        if evidence == 2 and len({tuple((r, c) for r, c, _ in t)
                                  for t in info["templates"].values()}) != ntypes:
            continue
        if evidence == 2 and np.array_equal(_shortcut(info, "one_template"), out):
            continue
        if not np.any(out[1:-1, 1:-1] != target[1:-1, 1:-1]):
            continue
        seen.add(signature)
        return {"input": grid, "output": out}
    raise RuntimeError("bounded scene construction could not satisfy evidence")


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count, test_count = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    # Regimes affect sampling, never the input-only transformation.
    train_lengths = (range(2, 4), range(4, 7), range(2, 7), range(2, 7))[regime]
    test_lengths = (range(4, 7), range(2, 4), range(2, 7), range(2, 7))[regime]
    seen, puzzle = set(), {"train": [], "test": []}
    for i in range(train_count):
        length = int(_pick(rng, list(train_lengths)))
        if i == 0:
            length = max(3, length)
        ntypes = 1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3 if regime == 2 else 4))
        style = "square" if regime == 3 else "free"
        puzzle["train"].append(_make_pair(rng, ntypes, length, style, i+1 if i < 2 else 0, seen))
    rng.shuffle(puzzle["train"])
    for i in range(test_count):
        ntypes = 3 if regime == 2 else int(rng.integers(1, 4))
        length = int(_pick(rng, list(test_lengths)))
        style = "elongated" if regime == 3 else "free"
        puzzle["test"].append(_make_pair(rng, ntypes, length, style, 0, seen))
    validate_generated_puzzle(puzzle)
    train_features = [pair_features(p) for p in puzzle["train"]]
    test_features = [pair_features(p) for p in puzzle["test"]]
    if regime == 0:
        assert min(f["layers"] for f in test_features) > max(f["layers"] for f in train_features)
    elif regime == 1:
        assert max(f["layers"] for f in test_features) < min(f["layers"] for f in train_features)
    elif regime == 2:
        assert min(f["types"] for f in test_features) > max(f["types"] for f in train_features)
    else:
        assert all(abs(f["output_shape"][0]-f["output_shape"][1]) <= 1 for f in train_features)
        assert all(max(f["output_shape"])/min(f["output_shape"]) >= 19/13 for f in test_features)
    return puzzle


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def pair_features(pair):
    """Task-specific measurements derived from visible grids."""
    info = _parse(pair["input"])
    distance = _distance(info["base"])
    templates = info["templates"]
    _, _, h, w = info["box"]
    geometries = tuple(sorted(tuple((r, c) for r, c, _ in t) for t in templates.values()))
    cb = info["code_box"]
    layout = "stacked" if cb[0]+cb[2] <= info["box"][0] or info["box"][0]+h <= cb[0] else "beside"
    return {"input_shape": info["grid"].shape, "output_shape": (h, w),
            "layers": len(info["code"]), "code_width": cb[3],
            "code_colors": len(set(info["code"])), "types": len(templates),
            "anchors": len(info["anchors"]), "repetitions": tuple(info["repetitions"].values()),
            "motif_sizes": tuple(len(t) for t in templates.values()), "geometries": geometries,
            "layout": layout, "background": info["bg"],
            "multicolor": any(len({color for r, c, color in t if (r, c) != (0, 0)}) > 1 for t in templates.values()),
            "repeated_rows": any(a == b for a, b in zip(info["code"], info["code"][1:])),
            "color_return": len([c for i, c in enumerate(info["code"]) if i == 0 or c != info["code"][i-1]]) > len(set(info["code"])),
            "clipped_halo": bool(np.any(distance[0] <= len(info["code"])) or np.any(distance[-1] <= len(info["code"]))
                                 or np.any(distance[:, 0] <= len(info["code"])) or np.any(distance[:, -1] <= len(info["code"]))),
            "merge_matters": not np.array_equal(_shortcut(info, "overwrite"), pair["output"])}


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("exactly train and test are required")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise AssertionError("splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence, train_info = set(), {k: False for k in _ALTERNATIVES}, []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise AssertionError("pairs must have exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _grid_key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse(inp)
            assert np.array_equal(out, _label_by_growth(info)), "incorrect contour growth or crop"
            assert np.array_equal(out, _from_distance(info)), "distance and growth disagree"
            assert np.array_equal(out[info["base"]], info["painted"][info["base"]]), "motif colors changed"
            assert np.array_equal(out[0], info["target"][0]) and np.array_equal(out[-1], info["target"][-1])
            assert np.array_equal(out[:, 0], info["target"][:, 0]) and np.array_equal(out[:, -1], info["target"][:, -1])
            assert set(np.unique(out)) <= set(np.unique(inp)), "invented color"
            assert all(not np.array_equal(_shortcut(info, k), out) for k in ("omit_layers", "anchors_only")), "a parent is inert"
            if split == "train":
                train_info.append(info)
                for alternative in evidence:
                    evidence[alternative] |= not np.array_equal(_shortcut(info, alternative), out)
    assert all(evidence.values()), "demonstrations fail to exclude a simpler alternative"
    assert any(len(i["templates"]) >= 2 for i in train_info), "key dispatch needs a multitype demonstration"
    assert len({tuple(sorted(i["templates"].values())) for i in train_info}) >= 2, "dictionary must be supplied afresh"
    return None


def self_test():
    """Bounded 42-seed coverage, counts, metamorphisms, and negative checks."""
    records, signatures, colors = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_grid_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"].copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, pair["input"]), "solve mutated its input"
                records.append(pair_features(pair))
                colors.update(int(c) for c in np.unique(inp))
    assert len(signatures) == 42
    assert colors == set(range(10))
    assert {r["layers"] for r in records} == set(range(2, 7))
    assert {r["types"] for r in records} == {1, 2, 3}
    assert {r["layout"] for r in records} == {"stacked", "beside"}
    assert {r["background"] for r in records} == set(range(10))
    assert len({r["input_shape"] for r in records}) >= 25
    assert len({r["output_shape"] for r in records}) >= 25
    assert len({g for r in records for g in r["geometries"]}) >= 100
    assert {2, 3, 4, 5} <= {r["code_width"] for r in records}
    assert {1, 2} <= {n for r in records for n in r["repetitions"]}
    assert set(range(3, 10)) <= {n for r in records for n in r["motif_sizes"]}
    for feature in ("multicolor", "repeated_rows", "color_return", "clipped_halo", "merge_matters"):
        assert sum(r[feature] for r in records) >= 5, feature
    assert len({r["anchors"] for r in records}) >= 4
    # All legal count requests and invalid numeric types are checked explicitly.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, again = generate_puzzle(7654321, 5, 2), generate_puzzle(7654321, 5, 2)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s]) for k in p)
    # Recoloring must not change semantic geometry.
    pair = first["train"][0]
    permutation = np.array([9, 7, 3, 0, 8, 2, 6, 1, 5, 4], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    info = _parse(pair["input"])
    cb = info["code_box"]
    # Changing one code row changes exactly its distance shell outside motifs.
    changed = pair["input"].copy()
    replacement = next(c for c in range(10) if c not in (info["bg"], info["code"][0]))
    changed[cb[0], cb[1]:cb[1]+cb[3]] = replacement
    predicted = solve(changed)
    d = _distance(info["base"])
    expected = pair["output"].copy()
    shell = d == len(info["code"])
    shell[[0, -1], :] = False
    shell[:, [0, -1]] = False
    expected[shell] = replacement
    assert np.array_equal(predicted, expected)
    # No tie-breaking state: distances depend only on the union of seed cells.
    mask = np.zeros((13, 15), dtype=bool)
    mask[5, 4], mask[5, 10] = True, True
    assert _distance(mask)[5, 7] == 3
    reach = mask.copy()
    for radius in range(1, 6):
        reach = _dilate(reach)
        assert np.array_equal(reach, _distance(mask) <= radius)
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Removing the strip makes the input invalid.
    malformed = pair["input"].copy()
    _crop(malformed, cb)[:] = info["bg"]
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("missing layer code accepted")
    return {"seeds": 42, "pairs": len(records),
            "input_shapes": len({r["input_shape"] for r in records}),
            "output_shapes": len({r["output_shape"] for r in records}),
            "motif_geometries": len({g for r in records for g in r["geometries"]}),
            "features": {k: sum(r[k] for r in records) for k in
                         ("multicolor", "repeated_rows", "color_return", "clipped_halo", "merge_matters")}}
