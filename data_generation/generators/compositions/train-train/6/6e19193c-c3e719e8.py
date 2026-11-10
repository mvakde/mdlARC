"""Self-stamping gap rays.

Every object in the input is a small solid rectangle (2x2, 3x3, 2x3 or 3x2;
one shape per grid) that is either complete ("inert") or missing exactly one
border cell ("marker").  The output zooms the grid by the object size: every
input cell becomes an object-sized block.  Each object cell becomes an exact
copy of its own object (self-stamping, from parent c3e719e8).  A marker's
missing cell stays a blank block and, like the L-triomino gap of parent
6e19193c, defines an outward direction (from the object's centre through the
gap); every macro cell beyond the gap along that direction, up to the grid
edge, also becomes a copy of the marker.  Everything else is background.
"""
from __future__ import annotations

import numpy as np


_SHAPES = ((2, 2), (3, 3), (2, 3), (3, 2))
_DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_ORTHOGONALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NAMES = {(-1, -1): "NW", (-1, 1): "NE", (1, -1): "SW", (1, 1): "SE",
          (-1, 0): "N", (1, 0): "S", (0, -1): "W", (0, 1): "E"}
_STYLES = ("accent", "rows", "cols", "random", "checker", "ring")


# ---------------------------------------------------------------------------
# Geometry shared by the solver, validator and sampler
# ---------------------------------------------------------------------------

def _gap_direction(shape, gap):
    """Outward direction from the object's centre through its missing cell."""
    kh, kw = shape
    return (int(np.sign(2 * gap[0] - (kh - 1))), int(np.sign(2 * gap[1] - (kw - 1))))


def _gap_for(shape, direction):
    """The unique border cell of a block of `shape` that points in `direction`."""
    kh, kw = shape
    for r in range(kh):
        for c in range(kw):
            if (r in (0, kh - 1) or c in (0, kw - 1)) and _gap_direction(shape, (r, c)) == direction:
                return (r, c)
    return None


def _ray(height, width, top, left, gap, direction):
    row, col = top + gap[0] + direction[0], left + gap[1] + direction[1]
    cells = []
    while 0 <= row < height and 0 <= col < width:
        cells.append((row, col))
        row += direction[0]
        col += direction[1]
    return cells


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if len(values) > 1 and counts[order[0]] == counts[order[1]]:
        raise ValueError("background colour is not uniquely most frequent")
    return int(values[order[0]])


def _components(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            result.append(sorted(cells))
    return result


def _parse(grid):
    """Recover background, object shape and objects from pixels only."""
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("grid must be 2D")
    height, width = grid.shape
    background = _background(grid)
    objects = []
    for cells in _components(grid != background):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        top, left = min(rows), min(cols)
        shape = (max(rows) - top + 1, max(cols) - left + 1)
        area = shape[0] * shape[1]
        pattern = grid[top:top + shape[0], left:left + shape[1]].copy()
        if len(cells) == area:
            gap = direction = None
            ray = []
        elif len(cells) == area - 1:
            missing = np.argwhere(pattern == background)
            gap = (int(missing[0][0]), int(missing[0][1]))
            if not (gap[0] in (0, shape[0] - 1) or gap[1] in (0, shape[1] - 1)):
                raise ValueError("missing cell is not on the object border")
            direction = _gap_direction(shape, gap)
            if direction == (0, 0):
                raise ValueError("missing cell has no outward direction")
            ray = _ray(height, width, top, left, gap, direction)
        else:
            raise ValueError("object is not a block missing at most one cell")
        objects.append({"top": top, "left": left, "shape": shape, "cells": cells,
                        "pattern": pattern, "gap": gap, "direction": direction,
                        "ray": ray})
    if not objects:
        raise ValueError("no objects")
    shapes = {o["shape"] for o in objects}
    if len(shapes) != 1:
        raise ValueError("objects in one grid must share a shape")
    shape = shapes.pop()
    if shape[0] * height > 30 or shape[1] * width > 30:
        raise ValueError("zoomed output would exceed 30x30")
    return background, shape, objects


def _render(height, width, background, shape, objects, with_rays=True):
    kh, kw = shape
    out = np.full((height * kh, width * kw), background, dtype=int)
    for obj in objects:
        targets = list(obj["cells"]) + (list(obj["ray"]) if with_rays else [])
        for r, c in targets:
            out[r * kh:(r + 1) * kh, c * kw:(c + 1) * kw] = obj["pattern"]
    return out


def _majority(pattern, background):
    values, counts = np.unique(pattern[pattern != background], return_counts=True)
    return int(values[int(np.argmax(counts))])


def _shortcut_outputs(input_grid):
    """Plausible rules that drop or distort one parent's contribution."""
    grid = np.asarray(input_grid, dtype=int)
    height, width = grid.shape
    background, shape, objects = _parse(grid)
    kh, kw = shape
    # B alone: every object is self-stamped but nothing is fired.
    no_rays = _render(height, width, background, shape, objects, with_rays=False)
    # A alone, then a plain zoom: rays in the marker's main colour, solid blocks.
    painted = grid.copy()
    for obj in objects:
        for r, c in obj["ray"]:
            painted[r, c] = _majority(obj["pattern"], background)
    solid_zoom = np.kron(painted, np.ones((kh, kw), dtype=int))
    # Ray that starts on the gap itself (fills the gap instead of skipping it).
    gap_start = []
    one_copy = []
    for obj in objects:
        if obj["gap"] is None:
            gap_start.append(obj)
            one_copy.append(obj)
            continue
        gap = (obj["top"] + obj["gap"][0], obj["left"] + obj["gap"][1])
        gap_start.append(dict(obj, ray=[gap] + list(obj["ray"])))
        one_copy.append(dict(obj, ray=list(obj["ray"])[:1]))
    return {"no_rays": no_rays, "solid_zoom": solid_zoom,
            "gap_start": _render(height, width, background, shape, gap_start),
            "one_copy": _render(height, width, background, shape, one_copy)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    background, shape, objects = _parse(grid)
    return _render(grid.shape[0], grid.shape[1], background, shape, objects)


# ---------------------------------------------------------------------------
# Strict family checks
# ---------------------------------------------------------------------------

def _bbox(obj):
    kh, kw = obj["shape"]
    return {(obj["top"] + r, obj["left"] + c) for r in range(kh) for c in range(kw)}


def _check_layout(grid):
    """Family invariants of one input; returns its parse."""
    background, shape, objects = _parse(grid)
    if shape not in _SHAPES:
        raise AssertionError(f"unsupported object shape {shape}")
    markers = [o for o in objects if o["gap"] is not None]
    if not markers:
        raise AssertionError("every input needs at least one marker")
    boxes = [_bbox(o) for o in objects]
    for i, obj in enumerate(objects):
        if obj["gap"] is not None and not obj["ray"]:
            raise AssertionError("marker gap points straight off the grid")
        for j, other in enumerate(objects):
            if i == j:
                continue
            if set(obj["ray"]) & boxes[j]:
                raise AssertionError("a ray passes through another object")
            if j > i and set(obj["ray"]) & set(other["ray"]):
                raise AssertionError("two rays share a cell")
            if j > i:
                ri = (obj["top"] - 1, obj["top"] + shape[0], obj["left"] - 1, obj["left"] + shape[1])
                if not (other["top"] > ri[1] or other["top"] + shape[0] - 1 < ri[0] or
                        other["left"] > ri[3] or other["left"] + shape[1] - 1 < ri[2]):
                    raise AssertionError("object boxes touch")
    for obj in objects:
        colors = set(int(v) for v in obj["pattern"].flat) - {background}
        if not colors:
            raise AssertionError("object without colour")
    return background, shape, objects


def _features(grid):
    background, shape, objects = _check_layout(grid)
    markers = [o for o in objects if o["gap"] is not None]
    return {
        "background": background,
        "shape": shape,
        "markers": len(markers),
        "inert": len(objects) - len(markers),
        "directions": {o["direction"] for o in markers},
        "orthogonal": any(0 in o["direction"] for o in markers),
        "multicolor": any(len(set(int(v) for v in o["pattern"].flat) - {background}) > 1
                          for o in objects),
        "ray_lengths": [len(o["ray"]) for o in markers],
        "colors": {int(v) for o in objects for v in o["pattern"].flat} - {background},
    }


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, tuple(int(v) for v in grid.flat)


def validate_generated_puzzle(puzzle: dict) -> None:
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("puzzle must have exactly train/test")
    train, test = puzzle["train"], puzzle["test"]
    if not (isinstance(train, list) and isinstance(test, list)):
        raise AssertionError("splits must be lists")
    if not (2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)):
        raise AssertionError("illegal pair counts")
    seen = set()
    feats = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise AssertionError("pair must have exactly input/output")
            grid_in = np.asarray(pair["input"])
            grid_out = np.asarray(pair["output"])
            for g in (grid_in, grid_out):
                if g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape):
                    raise AssertionError("grid size outside 1..30")
                if not np.issubdtype(g.dtype, np.integer) or g.min() < 0 or g.max() > 9:
                    raise AssertionError("grid colours outside 0..9")
            key = _key(grid_in)
            if key in seen:
                raise AssertionError("duplicate input")
            seen.add(key)
            f = _features(grid_in)
            kh, kw = f["shape"]
            if grid_out.shape != (grid_in.shape[0] * kh, grid_in.shape[1] * kw):
                raise AssertionError("output is not zoomed by the object shape")
            if not np.array_equal(solve(grid_in), grid_out):
                raise AssertionError("output disagrees with the rule")
            feats[split].append(f)
    tr, te = feats["train"], feats["test"]
    train_dirs = set().union(*(f["directions"] for f in tr))
    if len(train_dirs) < 2:
        raise AssertionError("training must show at least two ray directions")
    train_lengths = {n for f in tr for n in f["ray_lengths"]}
    if max(train_lengths) < 3 or len(train_lengths) < 2:
        raise AssertionError("training must show edge-reaching rays of several lengths")
    # Shortcut separation: every pair refutes "self-stamp without rays" (B
    # alone) and "zoom the A-output with solid blocks" (A without B).
    for split in ("train", "test"):
        for pair in puzzle[split]:
            alternatives = _shortcut_outputs(pair["input"])
            for name in ("no_rays", "solid_zoom", "gap_start"):
                if np.array_equal(alternatives[name], np.asarray(pair["output"])):
                    raise AssertionError(f"pair does not refute the {name} shortcut")
    if all(np.array_equal(_shortcut_outputs(p["input"])["one_copy"], np.asarray(p["output"]))
           for p in puzzle["train"]):
        raise AssertionError("training does not refute the single-copy shortcut")
    if any(f["multicolor"] for f in te) and not any(f["multicolor"] for f in tr):
        raise AssertionError("multicolour stamps in test are not demonstrated")
    if any(f["inert"] for f in te) and not any(f["inert"] for f in tr):
        raise AssertionError("inert blocks in test are not demonstrated")
    if any(f["orthogonal"] for f in te) and not any(f["orthogonal"] for f in tr):
        raise AssertionError("orthogonal gaps in test are not demonstrated")
    train_shapes = {f["shape"] for f in tr}
    if any(f["shape"] not in train_shapes for f in te) and len(train_shapes) < 2:
        raise AssertionError("a new test zoom needs at least two training zooms")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _size_limits(shape):
    return 30 // shape[0], 30 // shape[1]


def _pattern(rng, shape, gap, background, style, colors):
    """A kh x kw stamp; the gap (if any) is background."""
    kh, kw = shape
    cells = [(r, c) for r in range(kh) for c in range(kw) if (r, c) != gap]
    for _ in range(30):
        pattern = np.full(shape, background, dtype=int)
        if style == "mono":
            for r, c in cells:
                pattern[r, c] = colors[0]
            return pattern
        a, b = colors[0], colors[1]
        for r, c in cells:
            if style == "accent":
                value = a
            elif style == "rows":
                value = colors[r % len(colors)] if len(colors) > 2 and kh == 3 else (a if r % 2 == 0 else b)
            elif style == "cols":
                value = colors[c % len(colors)] if len(colors) > 2 and kw == 3 else (a if c % 2 == 0 else b)
            elif style == "checker":
                value = a if (r + c) % 2 == 0 else b
            elif style == "ring" and kh >= 3 and kw >= 3:
                value = b if (0 < r < kh - 1 and 0 < c < kw - 1) else a
            elif style == "ring":
                value = a if r in (0, kh - 1) and c in (0, kw - 1) else b
            else:  # random
                value = colors[int(rng.integers(len(colors)))]
            pattern[r, c] = value
        if style == "accent":
            r, c = cells[int(rng.integers(len(cells)))]
            pattern[r, c] = b
        used = set(int(v) for v in pattern.flat) - {background}
        if len(used) >= 2:
            return pattern
        style = "random"
    pattern = np.full(shape, background, dtype=int)
    for i, (r, c) in enumerate(cells):
        pattern[r, c] = colors[i % 2]
    return pattern


def _separated(top, left, shape, other):
    kh, kw = shape
    return (top > other["top"] + kh or top + kh < other["top"] or
            left > other["left"] + kw or left + kw < other["left"])


def _place(rng, height, width, shape, items, ray_profile):
    """Place objects so that the input parses uniquely and rays never collide."""
    kh, kw = shape
    placed, ray_cells, boxes = [], set(), set()
    for index, item in enumerate(items):
        gap = item.get("gap")
        direction = item.get("direction")
        candidates = []
        for top in range(height - kh + 1):
            for left in range(width - kw + 1):
                if any(not _separated(top, left, shape, other) for other in placed):
                    continue
                box = {(top + r, left + c) for r in range(kh) for c in range(kw)}
                if box & ray_cells:
                    continue
                ray = []
                if gap is not None:
                    ray = _ray(height, width, top, left, gap, direction)
                    if not ray or set(ray) & boxes or set(ray) & ray_cells:
                        continue
                candidates.append((top, left, ray, box))
        if not candidates:
            return None
        preferred = candidates
        if gap is not None:
            if ray_profile == "long" and index == 0:
                preferred = [c for c in candidates if len(c[2]) >= 3] or candidates
            elif ray_profile == "short" and index == 0:
                preferred = [c for c in candidates if len(c[2]) == 1] or candidates
            elif ray_profile == "mixed" and index < 2:
                want = (lambda n: n == 1) if index == 0 else (lambda n: n >= 3)
                preferred = [c for c in candidates if want(len(c[2]))] or candidates
        if gap is not None and preferred is candidates:
            # Favour informative rays: probability grows with visible length.
            weights = np.array([min(len(c[2]), 6) for c in candidates], dtype=float)
            choice = int(rng.choice(len(candidates), p=weights / weights.sum()))
        else:
            choice = int(rng.integers(len(preferred)))
        top, left, ray, box = preferred[choice]
        placed.append({"top": top, "left": left, "gap": gap, "direction": direction,
                       "ray": ray, "item": item})
        ray_cells.update(ray)
        boxes.update(box)
    return placed


def _build_pair(rng, spec):
    """spec: shape, size, background, items(kind/direction/style/colors)."""
    shape = spec["shape"]
    for _ in range(25):
        height, width = spec["size"]
        items = []
        for item in spec["items"]:
            entry = dict(item)
            if item["kind"] == "marker":
                entry["gap"] = _gap_for(shape, item["direction"])
            items.append(entry)
        order = [i for i in range(len(items)) if items[i]["kind"] == "marker"]
        order += [i for i in range(len(items)) if items[i]["kind"] != "marker"]
        placed = _place(rng, height, width, shape, [items[i] for i in order], spec["ray_profile"])
        if placed is None:
            # Grow the canvas a little (within the zoom limit) and retry.
            hmax, wmax = _size_limits(shape)
            spec = dict(spec, size=(min(hmax, height + int(rng.integers(0, 2))),
                                    min(wmax, width + int(rng.integers(0, 2)))))
            continue
        grid = np.full((height, width), spec["background"], dtype=int)
        for obj in placed:
            item = obj["item"]
            pattern = _pattern(rng, shape, obj["gap"], spec["background"], item["style"], item["colors"])
            kh, kw = shape
            region = grid[obj["top"]:obj["top"] + kh, obj["left"]:obj["left"] + kw]
            mask = pattern != spec["background"]
            region[mask] = pattern[mask]
        try:
            _check_layout(grid)
        except (AssertionError, ValueError):
            continue
        return grid, solve(grid)
    return None


def _plan(rng, num_train, num_test):
    """Puzzle-level conventions and per-pair specifications."""
    # Zoom / object shape plan.
    shape_mode = "fixed" if rng.random() < 0.5 else "mixed"
    weights = np.array([0.42, 0.24, 0.17, 0.17])
    if shape_mode == "fixed":
        base = _SHAPES[int(rng.choice(4, p=weights))]
        train_shapes = [base] * num_train
        pool = [base]
    else:
        size = 2 if rng.random() < 0.7 or num_train < 3 else 3
        pool = [_SHAPES[int(i)] for i in rng.choice(4, size=size, replace=False, p=weights)]
        train_shapes = [pool[i % size] for i in range(num_train)]
        if num_train > size:
            for i in range(size, num_train):
                train_shapes[i] = _pick(rng, pool)
        order = rng.permutation(num_train)
        train_shapes = [train_shapes[int(i)] for i in order]
    # Train/test shift.
    shifts = ["direction", "density", "sparse", "plain"]
    if shape_mode == "mixed":
        shifts += ["scale", "scale"]
    shift = _pick(rng, shifts)
    test_shapes = []
    for _ in range(num_test):
        if shift == "scale":
            novel = [s for s in _SHAPES if s not in pool]
            test_shapes.append(_pick(rng, novel) if novel and rng.random() < 0.7 else _pick(rng, pool))
        else:
            test_shapes.append(_pick(rng, pool))
    if shift == "scale" and all(s in train_shapes for s in test_shapes):
        # A redistributed training zoom still counts as a scale change if the
        # test uses the zoom least represented in training.
        counts = {s: train_shapes.count(s) for s in pool}
        test_shapes[0] = min(pool, key=lambda s: (counts[s], _SHAPES.index(s)))
    shapes = train_shapes + test_shapes
    # Edge (orthogonal) gaps require a 3-long side in some training pair.
    edges = rng.random() < 0.45 and any(_gap_for(s, d) for s in train_shapes for d in _ORTHOGONALS)
    # Colour conventions.
    color_mode = _pick(rng, ("mono", "multi", "some", "some"))
    uniform_pair_color = color_mode == "mono" and rng.random() < 0.35
    inert = rng.random() < 0.4
    base_bg = int(rng.integers(10)) if rng.random() < 0.6 else 0
    bg_varies = rng.random() < 0.35
    novel_palette = rng.random() < 0.3
    # Direction pools.
    if shift == "direction":
        train_diag = [_DIAGONALS[int(i)] for i in rng.choice(4, size=int(rng.integers(2, 4)), replace=False)]
    else:
        k = int(rng.choice((2, 3, 4), p=(0.25, 0.25, 0.5)))
        train_diag = [_DIAGONALS[int(i)] for i in rng.choice(4, size=k, replace=False)]
    supported = [d for d in _ORTHOGONALS if any(_gap_for(s, d) for s in train_shapes)]
    train_orth = [d for d in supported if rng.random() < 0.7] or ([_pick(rng, supported)] if supported else [])
    # Marker / inert counts.
    def capacity(shape, size):
        kh, kw = shape
        return int(np.clip(round(size[0] * size[1] / ((kh + 1) * (kw + 1) * 3.0)), 1, 6))

    def sample_size(shape, big=False, small=False):
        hmax, wmax = _size_limits(shape)
        if small:
            return int(rng.integers(5, max(6, min(hmax, 9)) + 1)), int(rng.integers(5, max(6, min(wmax, 9)) + 1))
        if big:
            return int(rng.integers(max(5, hmax - 3), hmax + 1)), int(rng.integers(max(5, wmax - 3), wmax + 1))
        if rng.random() < 0.15 and hmax >= 10 and wmax >= 10:
            return 10, 10
        return int(rng.integers(5, hmax + 1)), int(rng.integers(5, wmax + 1))

    specs = []
    for index, shape in enumerate(shapes):
        is_test = index >= num_train
        big = is_test and shift == "density"
        small = is_test and shift == "sparse"
        size = sample_size(shape, big=big, small=small)
        cap = capacity(shape, size)
        markers = int(rng.integers(1, cap + 1))
        if cap >= 2 and rng.random() < 0.55:
            markers = int(rng.integers(2, cap + 1))
        if small:
            markers = 1
        background = base_bg
        if bg_varies and rng.random() < 0.6:
            background = int(rng.integers(10))
        specs.append({"shape": shape, "size": size, "background": background,
                      "markers": markers, "inert": 0, "is_test": is_test, "cap": cap})
    if shift == "density":
        most = max(s["markers"] for s in specs[:num_train])
        for s in specs[num_train:]:
            s["markers"] = max(s["markers"], min(s["cap"] + 1, most + 1))
    if inert:
        for s in specs:
            if rng.random() < 0.55:
                s["inert"] = int(rng.integers(1, 3))
        if not any(s["inert"] for s in specs[:num_train]):
            specs[int(rng.integers(num_train))]["inert"] = 1
    # Direction assignment with training coverage.
    for s in specs:
        avail_orth = [d for d in _ORTHOGONALS if edges and _gap_for(s["shape"], d)]
        if s["is_test"]:
            s["dir_pool"] = list(_DIAGONALS) + avail_orth
        else:
            s["dir_pool"] = list(train_diag) + [d for d in avail_orth if d in train_orth]
        s["directions"] = [_pick(rng, s["dir_pool"]) for _ in range(s["markers"])]
    train_specs = specs[:num_train]
    required = list(train_diag[:2])
    if edges:
        orth_pairs = [s for s in train_specs if any(0 in d for d in s["dir_pool"])]
        host = _pick(rng, orth_pairs)
        host["directions"][int(rng.integers(len(host["directions"])))] = _pick(
            rng, [d for d in host["dir_pool"] if 0 in d])
    slots = [(i, j) for i, s in enumerate(train_specs) for j in range(len(s["directions"]))
             if 0 not in s["directions"][j]]
    if len(slots) < len(required):
        train_specs[0]["markers"] += 1
        train_specs[0]["directions"].append(required[0])
        slots.append((0, len(train_specs[0]["directions"]) - 1))
    for (i, j), d in zip([slots[int(k)] for k in rng.permutation(len(slots))], required):
        train_specs[i]["directions"][j] = d
    if shift == "direction":
        held = [d for d in _DIAGONALS if d not in train_diag]
        target = specs[num_train + int(rng.integers(num_test))]
        target["directions"][int(rng.integers(len(target["directions"])))] = _pick(rng, held)
    # Colours and stamp styles.
    bank = [int(c) for c in rng.permutation(10)]
    train_bank_size = int(rng.integers(3, 7))
    for s in specs:
        avail = [c for c in bank if c != s["background"]]
        if novel_palette and s["is_test"]:
            fresh = [c for c in avail if c not in bank[:train_bank_size]]
            avail = fresh + [c for c in avail if c not in fresh] if len(fresh) >= 2 else avail
        else:
            local = [c for c in bank[:train_bank_size] if c != s["background"]]
            avail = local if len(local) >= 3 else avail
        pair_color = _pick(rng, avail[:3])
        items = []
        kinds = ["marker"] * len(s["directions"]) + ["inert"] * s["inert"]
        for k, kind in enumerate(kinds):
            if color_mode == "mono":
                multi = False
            elif color_mode == "multi":
                multi = True
            else:
                multi = rng.random() < 0.5
            if multi:
                n = int(rng.integers(2, 4))
                colors = [int(c) for c in rng.choice(avail[:max(n + 1, 4)], size=n, replace=False)]
                style = _pick(rng, _STYLES)
            else:
                colors = [pair_color if uniform_pair_color else _pick(rng, avail[:5])]
                style = "mono"
            item = {"kind": kind, "colors": colors, "style": style}
            if kind == "marker":
                item["direction"] = s["directions"][k]
            items.append(item)
        s["items"] = items
        s["ray_profile"] = _pick(rng, ("any", "any", "any", "long", "long", "short", "mixed"))
    if color_mode != "mono" and not any(it["style"] != "mono" for s in train_specs for it in s["items"]):
        host = train_specs[int(rng.integers(num_train))]
        it = host["items"][0]
        avail = [c for c in bank if c != host["background"]]
        it["style"], it["colors"] = "accent", [int(c) for c in rng.choice(avail[:4], size=2, replace=False)]
    return specs, shift


def _attempt(rng, num_train, num_test):
    specs, shift = _plan(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, spec in enumerate(specs):
        split = "train" if index < num_train else "test"
        pair = None
        for _ in range(8):
            built = _build_pair(rng, spec)
            if built is None:
                spec = dict(spec, items=spec["items"][:max(1, len(spec["items"]) - 1)]
                            if len(spec["items"]) > 1 and rng.random() < 0.5 else spec["items"])
                continue
            if _key(built[0]) in seen:
                continue
            pair = built
            break
        if pair is None:
            return None
        seen.add(_key(pair[0]))
        puzzle[split].append({"input": pair[0], "output": pair[1]})
    return puzzle


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        puzzle = _attempt(rng, num_train, num_test)
        if puzzle is None:
            continue
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError("could not generate a valid puzzle")


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def _rows(*rows):
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


def self_test() -> None:
    # Hand-checked example: a 2x2 L pointing NW in a 5x5 grid.
    grid = _rows("00000", "00000", "00000", "00003", "00034")
    stamp = _rows("03", "34")
    expected = np.zeros((10, 10), dtype=int)
    for r, c in ((3, 4), (4, 3), (4, 4), (2, 2), (1, 1), (0, 0)):
        expected[2 * r:2 * r + 2, 2 * c:2 * c + 2] = stamp
    assert np.array_equal(solve(grid), expected)
    # Hand-checked 3x3 object with an edge gap: ray goes straight up.
    grid = _rows("5555555", "5555555", "5555555", "5515155", "5511155", "5511155", "5555555")
    out = solve(grid)
    assert out.shape == (21, 21)
    block = grid[3:6, 2:5]
    for r in range(3):
        assert np.array_equal(out[3 * r:3 * r + 3, 9:12], block)
    assert np.all(out[9:12, 9:12] == 5)          # the gap itself stays blank
    assert np.array_equal(out[9:12, 6:9], block)  # an object cell is a stamp
    # Inert block: self-stamped but no ray.
    grid = _rows("000000", "012000", "021000", "000000", "000300", "000330")
    out = solve(grid)
    assert np.array_equal(out[2:6, 2:6], np.kron(np.ones((2, 2), int), grid[1:3, 1:3]))
    # Counts.
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(5, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for a, b in ((2, 1), (10, 4), (5, 4)):
        p = generate_puzzle(11, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
    first = generate_puzzle(123)
    again = generate_puzzle(123)
    assert all(np.array_equal(x[k], y[k]) for s in ("train", "test")
               for x, y in zip(first[s], again[s]) for k in ("input", "output"))
    # Diversity and shortcut separation over 42 seeds.
    shapes, dirs, sizes, backgrounds, marker_counts, ray_lengths = set(), set(), set(), set(), set(), set()
    flags = {"multicolor": 0, "inert": 0, "orthogonal": 0, "mixed_zoom": 0,
             "novel_test_zoom": 0, "heldout_dir": 0, "short_ray": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        feats = {s: [_features(p["input"]) for p in puzzle[s]] for s in ("train", "test")}
        train_dirs = set().union(*(f["directions"] for f in feats["train"]))
        test_dirs = set().union(*(f["directions"] for f in feats["test"]))
        train_shapes = {f["shape"] for f in feats["train"]}
        flags["mixed_zoom"] += len(train_shapes) > 1
        flags["novel_test_zoom"] += any(f["shape"] not in train_shapes for f in feats["test"])
        flags["heldout_dir"] += bool(test_dirs - train_dirs)
        refuted = {"no_rays": False, "solid_zoom": False, "gap_start": False, "one_copy": False}
        for split in ("train", "test"):
            for pair, f in zip(puzzle[split], feats[split]):
                g = np.asarray(pair["input"])
                for name, alternative in _shortcut_outputs(g).items():
                    refuted[name] |= not np.array_equal(alternative, pair["output"])
                shapes.add(f["shape"])
                dirs.update(f["directions"])
                sizes.add(g.shape)
                backgrounds.add(f["background"])
                marker_counts.add(f["markers"])
                ray_lengths.update(f["ray_lengths"])
                flags["multicolor"] += f["multicolor"]
                flags["inert"] += f["inert"] > 0
                flags["orthogonal"] += f["orthogonal"]
                flags["short_ray"] += 1 in f["ray_lengths"]
        assert all(refuted.values()), refuted
    assert shapes == set(_SHAPES), shapes
    assert len(dirs) == 8, dirs
    assert len(sizes) >= 30 and len(backgrounds) >= 6
    assert max(marker_counts) >= 4 and 1 in marker_counts
    assert 1 in ray_lengths and max(ray_lengths) >= 8
    assert all(v > 0 for v in flags.values()), flags

