"""Ranked cavity shadows: topology selects the slots of a depth equalizer.

Only Python's standard library and NumPy are used. No files are read by the
generator or solver. self_test streams its seed sweep and writes no artifacts.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=False)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _geometry(mask):
    """Find enclosed negative-space components and their extreme-face contact."""
    holes, open_rows = [], set()
    h, w = mask.shape
    for cells in _components(~mask):
        rr, cc = zip(*cells)
        exterior = min(rr) == 0 or max(rr) == h - 1 or min(cc) == 0 or max(cc) == w - 1
        if exterior:
            open_rows.update(r for r, c in cells if c == w - 2)
            continue
        rows = set(rr)
        contacts = {r for r, c in cells if c == w - 2 and mask[r, w - 1]}
        active = bool(contacts)
        _require(not active or rows == contacts, "partially leading holes are outside this family")
        holes.append({"row": min(rr), "height": max(rr) - min(rr) + 1,
                      "depth": max(cc) - min(cc) + 1, "area": len(cells),
                      "active": active, "cells": cells})
    return holes, open_rows


def _canonical(grid):
    """Infer rail, blank projection side, background and absolute reading axis."""
    candidates = []
    for transpose in (False, True):
        a = grid.T if transpose else grid
        for flip in (False, True):
            g = np.fliplr(a) if flip else a
            h, w = g.shape
            for x in range(1, w - 1):
                rail, bg = int(g[0, x]), int(g[0, -1])
                if rail == bg or not np.all(g[:, x] == rail):
                    continue
                if np.count_nonzero(g == rail) != h:
                    continue
                if not np.all(g[:, x + 1:] == bg) or not np.any(g[:, :x] != bg):
                    continue
                candidates.append((g, transpose, flip, bg, rail, x))
    _require(len(candidates) == 1, "expected one exclusive full rail and one blank side")
    return candidates[0]


def _restore(grid, transpose, flip):
    a = np.fliplr(grid) if flip else grid
    return (a.T if transpose else a).copy()


def _parse(value):
    grid = _grid(value)
    g, transpose, flip, bg, rail, x = _canonical(grid)
    objects, slots = [], []
    previous_end = -2
    for cells in _components(g[:, :x] != bg):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        _require(r0 > previous_end + 1, "objects must have separated row bands")
        previous_end = r1
        colors = {int(g[r, c]) for r, c in cells}
        _require(len(colors) == 1, "each object must be monochrome")
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        holes, opened = _geometry(mask)
        color = colors.pop()
        obj = {"row": r0, "end": r1, "left": c0, "width": mask.shape[1],
               "color": color, "mask": mask, "holes": holes, "open_rows": opened,
               "gap_to_rail": x - c1 - 1}
        index = len(objects)
        for hole in holes:
            if hole["active"]:
                slots.append({**hole, "row": r0 + hole["row"],
                              "color": color, "object": index,
                              "object_width": mask.shape[1]})
        objects.append(obj)
    slots.sort(key=lambda s: s["row"])
    _require(all(a["row"] + a["height"] < b["row"] for a, b in zip(slots, slots[1:])),
             "aperture row bands must be separated")
    _require(all(s["depth"] <= g.shape[1] - x - 1 for s in slots), "shadow would exceed canvas")
    return {"grid": g, "transpose": transpose, "flip": flip, "bg": bg,
            "rail": rail, "x": x, "objects": objects, "slots": slots}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read one input only; transfer ranked depths into fixed aperture bands."""
    a = _parse(input_grid)
    out = a["grid"].copy()
    pool = [s["depth"] for s in a["slots"]]
    # Repeated selection is deliberately separate from the builder's sorted zip.
    for slot in a["slots"]:
        length = min(pool)
        pool.remove(length)
        y, h = slot["row"], slot["height"]
        out[y:y + h, a["x"] + 1:a["x"] + 1 + length] = slot["color"]
    return _restore(out, a["transpose"], a["flip"])


def _band(kind, depth, height=1, step=False, locked=False):
    return dict(kind=kind, depth=int(depth), height=int(height), step=step, locked=locked)


def _groups(bands, maximum):
    return [bands[i:i + maximum] for i in range(0, len(bands), maximum)]


def _plan(rng, witness, test):
    """Choose cavity composition, independently of palette and placement."""
    if witness == 1:
        base = int(rng.integers(1, 4))
        values = [base + 2, base, base + 1] if rng.random() < .5 else [base + 1, base + 2, base]
        active = [_band("active", d, 2 if d == base + 1 else 1,
                        locked=d == base + 2) for d in values]
        cut = int(rng.integers(1, 3))
        groups = [active[:cut], active[cut:]]
        contrast = [_band("open", int(rng.integers(1, 5))),
                    _band("recessed", int(rng.integers(1, 6)))]
        if rng.random() < .5:
            rng.shuffle(contrast)
            groups.insert(int(rng.integers(3)), contrast)
        else:
            for band in contrast:
                groups.insert(int(rng.integers(len(groups) + 1)), [band])
    elif witness == 2:
        base = int(rng.integers(1, 3))
        values = [base + 1, base, base + 1]
        if rng.random() < .5:
            values.append(base + 2)
        rng.shuffle(values)
        if values == sorted(values):
            values = values[1:] + values[:1]
        active = [_band("active", d) for d in values]
        chosen = next(b for b in active if b["depth"] >= 2)
        chosen.update(height=2, step=True)
        groups = _groups(active, 2)
        # An open channel shares an emitter; a separate thin tail recesses a ring.
        groups[int(rng.integers(len(groups)))].append(_band("open", 2))
        groups.insert(int(rng.integers(len(groups) + 1)), [_band("tail", int(rng.integers(1, 6)))])
    elif test:
        # A connectivity shift: 3-4 separately ranked apertures in one object.
        n = int(rng.integers(3, 5))
        groups = [[_band("active", int(rng.integers(1, 8)), step=rng.random() < .55)
                   for _ in range(n)]]
        if rng.random() < .55:
            groups.append([_band("active", int(rng.integers(1, 8)))])
        for _ in range(int(rng.integers(0, 3))):
            b = _band(str(rng.choice(["open", "recessed"])), int(rng.integers(1, 7)))
            if rng.random() < .55:
                groups[int(rng.integers(len(groups)))].append(b)
            else:
                groups.insert(int(rng.integers(len(groups) + 1)), [b])
    else:
        n = 0 if rng.random() < .18 else int(rng.integers(1, 6))
        active = [_band("active", int(rng.integers(1, 6)), step=rng.random() < .4)
                  for _ in range(n)]
        groups = _groups(active, int(rng.integers(1, 3)))
        for _ in range(int(rng.integers(1, 4)) if not n else int(rng.integers(0, 3))):
            kind = str(rng.choice(["open", "recessed", "solid", "tail"]))
            b = [] if kind == "solid" else [_band(kind, int(rng.integers(1, 6)))]
            groups.insert(int(rng.integers(len(groups) + 1)), b)
    # Minimal heights include one separating row between every cavity/object.
    def minimum(gs):
        return sum(2 + sum(b["height"] for b in bs) + max(0, len(bs) - 1)
                   for bs in gs) + len(gs) - 1 + 2
    # Only optional extra distractors can exceed the budget in large scenes.
    while minimum(groups) > 28:
        removable = [i for i, bs in enumerate(groups) if not any(b["kind"] == "active" for b in bs)]
        _require(bool(removable), "invalid construction budget")
        groups.pop(removable[-1])
    specs = [dict(bands=bs, top=1, bottom=1, separators=[1] * max(0, len(bs) - 1)) for bs in groups]
    gaps = [1] * (len(specs) - 1)
    margins = [1, 1]
    spare = 30 - minimum(groups)
    # Bounded extra space changes cavity heights, walls, spacing and canvas size.
    for _ in range(int(rng.integers(spare + 1))):
        choices = [("margin", i, None) for i in range(2) if margins[i] < 4]
        choices += [("gap", i, None) for i, g in enumerate(gaps) if g < 4]
        for i, s in enumerate(specs):
            choices += [(key, i, None) for key in ("top", "bottom") if s[key] < 3]
            choices += [("separator", i, j) for j, v in enumerate(s["separators"]) if v < 3]
            choices += [("height", i, j) for j, b in enumerate(s["bands"])
                        if not b["locked"] and b["height"] < 3]
        if not choices:
            break
        key, i, j = choices[int(rng.integers(len(choices)))]
        if key == "margin":
            margins[i] += 1
        elif key == "gap":
            gaps[i] += 1
        elif key in ("top", "bottom"):
            specs[i][key] += 1
        elif key == "separator":
            specs[i]["separators"][j] += 1
        else:
            specs[i]["bands"][j]["height"] += 1
    return specs, gaps, margins


def _object_mask(rng, spec):
    bands = spec["bands"]
    tail = bool(bands) and all(b["kind"] == "tail" for b in bands)
    width = max([2] + [b["depth"] + (3 if b["kind"] == "recessed" else 2) for b in bands])
    width += int(rng.integers(0, 3))
    h = spec["top"] + spec["bottom"] + sum(b["height"] for b in bands) + sum(spec["separators"])
    mask = np.ones((h, width), dtype=bool)
    y, active = spec["top"], []
    for i, band in enumerate(bands):
        kind, depth, bh = band["kind"], band["depth"], band["height"]
        right = width - (3 if kind == "recessed" else 2)
        left = right - depth + 1
        starts = [left] * bh
        if band["step"] and depth >= 2 and bh >= 2:
            starts = [int(rng.integers(left, right + 1)) for _ in range(bh)]
            starts[0], starts[-1] = left, left + 1
        for k in range(bh):
            mask[y + k, starts[k]:right + 1] = False
        if kind == "open":
            side = int(rng.integers(4 if len(bands) == 1 else 2))
            if side == 0:
                mask[y:y + bh, :right + 1] = False
            elif side == 1:
                mask[y:y + bh, left:] = False
            elif side == 2:
                mask[:y + 1, right] = False
            else:
                mask[y + bh - 1:, right] = False
        if kind == "active":
            active.append((y, bh, depth))
        y += bh + (spec["separators"][i] if i < len(bands) - 1 else 0)
    if tail:
        extension = int(rng.integers(1, 3))
        extended = np.zeros((h, width + extension), dtype=bool)
        extended[:, :width] = mask
        extended[0 if rng.random() < .5 else -1, width:] = True
        mask = extended
    _require(len(_components(mask)) == 1, "object construction disconnected")
    return mask, active


def _sample_pair(rng, witness, test, transpose, flip, fixed_palette):
    specs, gaps, margins = _plan(rng, witness, test)
    built = [_object_mask(rng, s) for s in specs]
    max_width = max(m.shape[1] for m, _ in built)
    max_depth = max([1] + [d for _, holes in built for _, _, d in holes])
    available = 29 - max_width - max_depth
    _require(available >= 1, "width budget exhausted")
    extra_work = int(rng.integers(min(6, available - 1) + 1))
    work = max_width + extra_work
    extra_band = int(rng.integers(1 if witness else 0, min(6, available - extra_work) + 1))
    band = max_depth + extra_band
    height = sum(m.shape[0] for m, _ in built) + sum(gaps) + sum(margins)
    palette = fixed_palette if fixed_palette is not None else tuple(map(int, rng.permutation(10)[:6]))
    bg, rail = palette[:2]
    colors = list(palette[2:])
    rng.shuffle(colors)
    inp = np.full((height, work + 1 + band), bg, dtype=np.int8)
    inp[:, work] = rail
    slots = []
    row = margins[0]
    for i, (mask, holes) in enumerate(built):
        color = colors[i % len(colors)]
        col = int(rng.integers(work - mask.shape[1] + 1))
        view = inp[row:row + mask.shape[0], col:col + mask.shape[1]]
        view[mask] = color
        slots.extend((row + y, h, d, color) for y, h, d in holes)
        row += mask.shape[0] + (gaps[i] if i < len(gaps) else 0)
    slots.sort()
    out = inp.copy()
    for (y, h, _, color), length in zip(slots, sorted(s[2] for s in slots)):
        out[y:y + h, work + 1:work + 1 + length] = color
    return {"input": _restore(inp, transpose, flip), "output": _restore(out, transpose, flip)}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _unique(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    # The 20 color-cycle/reflection maps give distinct images of >=3 colors.
    # At most 13 previous inputs are excluded; geometry is never regenerated.
    for sign in (1, -1):
        for shift in range(10):
            lookup = np.array([(sign * c + shift) % 10 for c in range(10)], dtype=np.int8)
            candidate = {k: lookup[v] for k, v in pair.items()}
            if _key(candidate["input"]) not in seen:
                return candidate
    raise AssertionError("finite color orbit exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    fixed = tuple(map(int, rng.permutation(10)[:6])) if rng.random() < .4 else None
    mixed = rng.random() < .6
    base = bool(rng.integers(2))
    axes = [False, True] if mixed else [base, base]
    axes += [bool(rng.integers(2)) if mixed else base for _ in range(nt - 2)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            transpose = axes[i] if split == "train" else bool(rng.integers(2)) if mixed else base
            pair = _sample_pair(rng, i + 1 if split == "train" and i < 2 else 0,
                                split == "test", transpose, bool(rng.integers(2)), fixed)
            pair = _unique(pair, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    slots = a["slots"]
    depths = [s["depth"] for s in slots]
    ranked = sorted(depths)
    local = []
    for i in range(len(a["objects"])):
        local.extend(sorted(s["depth"] for s in slots if s["object"] == i))
    return {"depths": tuple(depths), "heights": tuple(s["height"] for s in slots),
            "sort_matters": depths != ranked,
            "reverse_fails": depths[::-1] != ranked,
            "local_fails": local != ranked,
            "area_fails": [s["depth"] for s in sorted(slots, key=lambda s: s["area"])] != ranked,
            "open": any(o["open_rows"] for o in a["objects"]),
            "decoy": any(not h["active"] for o in a["objects"] for h in o["holes"]),
            "stepped": any(h["area"] < h["depth"] * h["height"] for o in a["objects"] for h in o["holes"]),
            "max_group": max(sum(h["active"] for h in o["holes"]) for o in a["objects"]),
            "count": len(a["objects"]), "slots": len(slots),
            "orientation": (a["transpose"], a["flip"]),
            "shape": tuple(a["grid"].shape), "bg": a["bg"], "rail": a["rail"],
            "band": a["grid"].shape[1] - a["x"] - 1,
            "object_widths": tuple(o["width"] for o in a["objects"]),
            "gaps": tuple(o["gap_to_rail"] for o in a["objects"])}


def _check_pair(pair):
    """Relational output check: no call to solve, builder, or shadow renderer."""
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "shape changed")
    a = _parse(inp)
    b = out.T if a["transpose"] else out
    b = np.fliplr(b) if a["flip"] else b
    x, bg = a["x"], a["bg"]
    _require(np.array_equal(a["grid"][:, :x + 1], b[:, :x + 1]), "objects or rail changed")
    covered = np.zeros(b[:, x + 1:].shape, dtype=bool)
    lengths = []
    for slot in a["slots"]:
        y, h = slot["row"], slot["height"]
        rows = b[y:y + h, x + 1:]
        first = rows[0]
        occupied = np.flatnonzero(first != bg)
        _require(len(occupied) > 0 and np.array_equal(occupied, np.arange(len(occupied))),
                 "each shadow must be a nonempty prefix")
        n = len(occupied)
        _require(np.all(rows[:, :n] == slot["color"]) and np.all(rows[:, n:] == bg),
                 "slot color, height or rectangle changed")
        covered[y:y + h, :n] = True
        lengths.append(n)
    _require(np.all(b[:, x + 1:][~covered] == bg), "open, recessed, or unrelated row emitted")
    _require(all(v <= w for v, w in zip(lengths, lengths[1:])), "shadows are not increasing")
    pool = [s["depth"] for s in a["slots"]]
    for n in lengths:
        _require(n in pool, "shadow depth multiset changed")
        pool.remove(n)
    _require(not pool, "missing shadow")
    return _features(a)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, measured = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            f = _check_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            measured[split].append(f)
    train, test = measured["train"], measured["test"]
    for name in ("sort_matters", "reverse_fails", "local_fails", "area_fails", "open", "decoy", "stepped"):
        _require(any(f[name] for f in train), f"missing training witness: {name}")
    demonstrated = {f["orientation"][0] for f in train if f["sort_matters"]}
    _require(all(f["orientation"][0] in demonstrated for f in test), "test reading axis not demonstrated")
    _require(max(f["max_group"] for f in train) <= 2, "training exceeds aperture grouping curriculum")
    _require(all(f["max_group"] >= 3 for f in test), "missing test connectivity shift")
    return None


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    signatures, structures, shapes, directions = set(), set(), set(), set()
    counts, depths, heights, backgrounds, rails = set(), set(), set(), set(), set()
    stepped = silent = ties = shared = zero_gap = far_gap = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _digest(puzzle)
        signatures.add(signature)
        if seed == 0:
            first = signature
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
                a = _parse(pair["input"])
                f = _features(a)
                shapes.add(tuple(pair["input"].shape))
                directions.add(f["orientation"])
                counts.add(f["count"])
                depths.update(f["depths"])
                heights.update(f["heights"])
                backgrounds.add(f["bg"])
                rails.add(f["rail"])
                stepped += f["stepped"]
                silent += not f["slots"]
                ties += len(set(f["depths"])) < len(f["depths"])
                shared += f["max_group"] >= 2
                zero_gap += 0 in f["gaps"]
                far_gap += any(g >= 3 for g in f["gaps"])
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair["input"].ravel())
                structures.add((pair["input"].shape, hashlib.sha256(normalized).hexdigest()))
        # A real incorrect depth and an illegal source mutation must be rejected.
        if seed == 0:
            for mutation in ("depth", "source"):
                p = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzle[s]]
                     for s in ("train", "test")}
                pair = p["test"][0]
                a = _parse(pair["input"])
                canonical = pair["output"].T if a["transpose"] else pair["output"]
                canonical = np.fliplr(canonical) if a["flip"] else canonical
                if mutation == "source":
                    canonical[0, a["x"]] = a["bg"]
                else:
                    canonical[a["slots"][0]["row"], a["x"] + 1] = a["bg"]
                try:
                    validate_generated_puzzle(p)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("validator accepted corruption")
    _require(len(signatures) == 42 and first == _digest(generate_puzzle(0)), "seed reproduction failed")
    _require(len(structures) >= 190 and len(shapes) >= 45, "geometry diversity collapsed")
    _require(directions == {(False, False), (False, True), (True, False), (True, True)}, "missing direction")
    _require(depths >= set(range(1, 8)) and heights >= {1, 2, 3}, "measurement ranges collapsed")
    _require(len(counts) >= 4 and stepped >= 42 and ties >= 30 and shared >= 100, "structure ranges collapsed")
    _require(silent >= 1 and zero_gap >= 10 and far_gap >= 10, "missing placement or silent variation")
    _require(backgrounds == set(range(10)) and rails == set(range(10)), "role colors collapsed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(12345 + nt * 10 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "input_structures": len(structures), "input_shapes": len(shapes),
            "object_counts": sorted(counts), "depths": sorted(depths), "slot_heights": sorted(heights),
            "stepped_pairs": stepped, "silent_pairs": silent, "tied_pairs": ties,
            "multi_aperture_pairs": shared, "aligned_source_pairs": zero_gap,
            "distant_source_pairs": far_gap}
