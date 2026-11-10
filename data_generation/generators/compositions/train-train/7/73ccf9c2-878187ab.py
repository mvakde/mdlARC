"""Mirror cancellation: turn the two colors of unpaired pixels into geometry.

Only standard-library modules and NumPy are used.  The solver has no learned or
puzzle-level parameters: every color role and output dimension is input-derived.
"""

import numpy as np


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    """Eight-connected components, irrespective of pigment."""
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    height, width = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, points = [(r, c)], []
        while todo:
            rr, cc = todo.pop()
            points.append((rr, cc))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = rr + dr, cc + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and mask[nr, nc] and not seen[nr, nc]):
                        seen[nr, nc] = True
                        todo.append((nr, nc))
        result.append(points)
    return result


def _analyze_input(value):
    """Read visible geometry; this routine never consults generated metadata."""
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 3 or 2 * int(counts.max()) <= a.size:
        raise ValueError("a majority background and two foreground colors are required")
    bg = int(colors[int(np.argmax(counts))])
    signals = tuple(int(c) for c in colors if c != bg)
    records = []
    for points in _components(a != bg):
        rows, cols = zip(*points)
        r0, r1 = min(rows), max(rows) + 1
        c0, c1 = min(cols), max(cols) + 1
        crop = a[r0:r1, c0:c1]
        mask = crop != bg
        if int(mask.sum()) != len(points):
            raise ValueError("one figure's box encloses another figure")
        # Occupied mirror partners cancel even when their pigments differ.
        residual = mask & ~mask[:, ::-1]
        records.append({"box": (r0, r1, c0, c1), "crop": crop,
                        "mask": mask, "residual": residual})
    if not 3 <= len(records) <= 5:
        raise ValueError("expected three to five separated figures")
    for i, first in enumerate(records):
        r0, r1, c0, c1 = first["box"]
        for other in records[i + 1:]:
            s0, s1, d0, d1 = other["box"]
            if not (r1 <= s0 or s1 <= r0 or c1 <= d0 or d1 <= c0):
                raise ValueError("figure boxes overlap")
    targets = [rec for rec in records if rec["residual"].any()]
    if len(targets) != 1:
        raise ValueError("expected a unique occupancy-symmetry outlier")
    target = targets[0]
    residual_counts = {c: int(np.count_nonzero(
        target["residual"] & (target["crop"] == c))) for c in signals}
    body, accent = sorted(signals, key=residual_counts.get)
    height, width = residual_counts[body], residual_counts[accent]
    if not (3 <= height < width and height <= a.shape[0] and width <= a.shape[1]):
        raise ValueError("distinct residual counts must fit the output canvas")
    return {"background": bg, "signals": signals, "records": records,
            "target": target, "counts": residual_counts, "body": body,
            "accent": accent, "height": height, "width": width}


def _draw(shape, height, width, background, body, accent):
    out = np.full(shape, background, dtype=np.int8)
    out[-height:, :width] = body
    for d in range(height):
        out[shape[0] - 1 - d, d] = accent
        out[shape[0] - 1 - d, width - 1 - d] = accent
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Cancel local mirror partners, count the survivors, draw their diagram."""
    a = _grid(input_grid)
    facts = _analyze_input(a)
    return _draw(a.shape, facts["height"], facts["width"],
                 facts["background"], facts["body"], facts["accent"])


def _adjacent(mask, r, c):
    return bool(mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any())


def _symmetric_core(rng, height, width, density):
    """An eight-connected mirrored scaffold with a tight, varied silhouette."""
    mask = np.zeros((height, width), dtype=bool)
    left = (width - 1) // 2
    mask[:, left:width - left] = True
    row = int(rng.integers(height))
    for col in range(left - 1, -1, -1):
        row = int(np.clip(row + int(rng.integers(-1, 2)), 0, height - 1))
        mask[row, col] = mask[row, width - 1 - col] = True
    if rng.random() < 0.18:
        # Preserve some simple interval-band silhouettes alongside branched ones.
        for r in range(height):
            extent = int(rng.integers(left + 1))
            mask[r, left - extent:width - left + extent] = True
    for _ in range(3 * height * width):
        if mask.mean() >= density:
            break
        r, c = int(rng.integers(height)), int(rng.integers((width + 1) // 2))
        if not mask[r, c] and _adjacent(mask, r, c):
            mask[r, c] = mask[r, width - 1 - c] = True
    return mask


def _target_mask(rng, total, excluded_shape=None):
    dimensions = []
    for h in range(7, 14):
        for w in range(7, 14):
            # Leave enough mirror pairs outside the mandatory spanning scaffold.
            capacity = h * (w // 2) - w // 2 - (h if w % 2 == 0 else 0)
            if capacity >= total + 4 and (h, w) != excluded_shape:
                dimensions.append((h, w))
    for _ in range(160):
        h, w = dimensions[int(rng.integers(len(dimensions)))]
        mask = _symmetric_core(rng, h, w, float(rng.uniform(0.18, 0.38)))
        available = [(r, c) for r in range(h) for c in range(w // 2)
                     if not mask[r, c]]
        if len(available) < total:
            continue
        for _ in range(total):
            choices = []
            for r, c in available:
                for cc in (c, w - 1 - c):
                    if _adjacent(mask, r, cc):
                        choices.append((r, c, cc))
            if not choices:
                break
            r, c, cc = choices[int(rng.integers(len(choices)))]
            mask[r, cc] = True
            available.remove((r, c))
        residual = mask & ~mask[:, ::-1]
        if (int(residual.sum()) == total
                and int(residual[:, :w // 2].sum()) >= 2
                and int(residual[:, (w + 1) // 2:].sum()) >= 2):
            return mask
    raise RuntimeError("could not construct the requested mirror residual")


def _objects(rng, total, count, large_control=False, excluded_shape=None):
    target = _target_mask(rng, total, excluded_shape)
    masks = [target]
    for i in range(count - 1):
        if large_control and i == 0:
            h, w = target.shape
        else:
            h, w = int(rng.integers(3, 10)), int(rng.integers(3, 10))
        masks.append(_symmetric_core(rng, h, w, float(rng.uniform(0.25, 0.65))))
    return masks


def _pack(rng, masks, shape, quadrant=None):
    """Pack irregular figures by their boxes, without a panel or slot template."""
    height, width = shape
    order = [0] + sorted(range(1, len(masks)), key=lambda i: -masks[i].size)
    for _ in range(100):
        positions, occupied = {}, []
        for i in order:
            h, w = masks[i].shape
            candidates = []
            for r in range(height - h + 1):
                for c in range(width - w + 1):
                    if i == 0 and quadrant is not None:
                        if ((2 * r + h < height) != quadrant[0]
                                or (2 * c + w < width) != quadrant[1]):
                            continue
                    if all(r + h < rr or rr + hh < r or c + w < cc or cc + ww < c
                           for rr, cc, hh, ww in occupied):
                        candidates.append((r, c))
            if not candidates:
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            positions[i] = (r, c)
            occupied.append((r, c, h, w))
        if len(positions) == len(masks):
            return [positions[i] for i in range(len(masks))]
    raise RuntimeError("could not pack separated figures")


def _paint(rng, masks, positions, shape, palette, dimensions, small_index,
           global_first_count=None):
    bg, color0, color1 = palette
    h, w = dimensions
    target = masks[0]
    residual = target & ~target[:, ::-1]
    lonely = [tuple(map(int, p)) for p in np.argwhere(residual)]
    paired = []
    for i, mask in enumerate(masks):
        for r, c in np.argwhere(mask):
            r, c = int(r), int(c)
            if i != 0 or not residual[r, c]:
                paired.append((i, r, c))
    total = len(lonely) + len(paired)
    n0 = h if small_index == 0 else w
    if global_first_count is None:
        global_first_count = total // 2 + int(rng.integers(-total // 8, total // 8 + 1))
    paired0 = global_first_count - n0
    if not 1 <= paired0 < len(paired):
        raise RuntimeError("incompatible pigment budget")
    for _ in range(120):
        chosen_lonely = set(int(i) for i in rng.choice(len(lonely), n0, replace=False))
        # Each residual pigment must be represented on both sides of the axis.
        sides = {color0: set(), color1: set()}
        for j, (_, c) in enumerate(lonely):
            sides[color0 if j in chosen_lonely else color1].add(c < target.shape[1] // 2)
        if any(len(v) != 2 for v in sides.values()):
            continue
        colored = [np.full(m.shape, bg, dtype=np.int8) for m in masks]
        chosen_paired = set(int(i) for i in rng.choice(len(paired), paired0, replace=False))
        for j, (i, r, c) in enumerate(paired):
            colored[i][r, c] = color0 if j in chosen_paired else color1
        for j, (r, c) in enumerate(lonely):
            colored[0][r, c] = color0 if j in chosen_lonely else color1
        paired_target = target & ~residual
        if any(not np.any(paired_target & (colored[0] == color))
               for color in (color0, color1)):
            continue
        if not all(not np.array_equal(crop, crop[:, ::-1]) for crop in colored[1:]):
            continue
        a = np.full(shape, bg, dtype=np.int8)
        for crop, mask, (r, c) in zip(colored, masks, positions):
            region = a[r:r + mask.shape[0], c:c + mask.shape[1]]
            region[mask] = crop[mask]
        if 2 * int(np.count_nonzero(a == bg)) <= a.size:
            raise RuntimeError("scene is too dense for a majority background")
        body = (color0, color1)[small_index]
        accent = (color1, color0)[small_index]
        return {"input": a, "output": _draw(shape, h, w, bg, body, accent)}
    raise RuntimeError("could not distribute pigments with the teaching controls")


def _fresh_pair(rng, dimensions, palette, small_index, shape=None, excluded_shape=None,
                test=False):
    for _ in range(60):
        count = int(rng.integers(3, 6))
        masks = _objects(rng, sum(dimensions), count, excluded_shape=excluded_shape)
        canvas = shape or (int(rng.integers(24 if test else 22, 31)),
                           int(rng.integers(24 if test else 22, 31)))
        try:
            positions = _pack(rng, masks, canvas)
            return _paint(rng, masks, positions, canvas, palette, dimensions, small_index)
        except RuntimeError:
            continue
    raise RuntimeError("could not build a fresh scene")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    palette = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    varied_palette = bool(rng.integers(2))
    shapes = [(h, w) for h in range(3, 8) for w in range(h + 1, 13)]
    controls = [(a, b) for a in shapes for b in shapes
                if sum(a) == sum(b) and a[0] <= a[1] // 2
                and b[0] > (b[1] + 1) // 2]
    first, second = controls[int(rng.integers(len(controls)))]
    if int(rng.integers(2)):
        first, second = second, first
    masks = _objects(rng, sum(first), 4, large_control=True)
    canvas = (int(rng.integers(27, 31)), int(rng.integers(27, 31)))
    quadrant = (bool(rng.integers(2)), bool(rng.integers(2)))
    small = int(rng.integers(2))
    pigment0 = sum(int(m.sum()) for m in masks) // 2 - int(rng.integers(1, 4))
    pair0 = _paint(rng, masks, _pack(rng, masks, canvas, quadrant), canvas,
                   palette, first, small, pigment0)
    pair1 = _paint(rng, masks, _pack(rng, masks, canvas, tuple(not v for v in quadrant)),
                   canvas, palette, second, 1 - small, pigment0)
    puzzle = {"train": [pair0, pair1], "test": []}
    seen = {_key(pair0["input"]), _key(pair1["input"])}
    for split, count in (("train", train - 2), ("test", test)):
        for index in range(count):
            if split == "train" and index == 0:
                dims = first  # same diagram, new geometry and pigment arrangement
                forced_canvas, excluded = canvas, masks[0].shape
            elif split == "train":
                dims = shapes[int(rng.integers(len(shapes)))]
                forced_canvas, excluded = None, None
            else:
                # Extrapolate both counters; the same cancellation rule applies.
                dims = (int(rng.integers(8, 12)), int(rng.integers(13, 17)))
                forced_canvas, excluded = None, None
            for _ in range(20):
                local_palette = (tuple(int(c) for c in rng.choice(10, 3, replace=False))
                                 if varied_palette else palette)
                pair = _fresh_pair(rng, dims, local_palette, int(rng.integers(2)),
                                   forced_canvas, excluded, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not create distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, the rule, and the deliberate minimum-demonstration controls."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    facts, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            f = _analyze_input(a)
            assert out.shape == a.shape
            assert np.array_equal(out, solve(a)), "incorrect cancellation diagram"
            assert set(map(int, np.unique(out))) == set(map(int, np.unique(a)))
            t = f["target"]
            paired = t["mask"] & ~t["residual"]
            assert all(np.any(paired & (t["crop"] == c)) for c in f["signals"])
            assert all(not np.array_equal(r["crop"], r["crop"][:, ::-1])
                       for r in f["records"] if r is not t)
            for color in f["signals"]:
                rc = np.argwhere(t["residual"] & (t["crop"] == color))[:, 1]
                assert np.any(rc < t["mask"].shape[1] // 2)
                assert np.any(rc >= (t["mask"].shape[1] + 1) // 2)
            # Counting either raw scene colors or all target pixels must be wrong.
            assert all(np.count_nonzero(a == c) > f["counts"][c] for c in f["signals"])
            if split == "train":
                assert 3 <= f["height"] <= 7 and f["width"] <= 12
            else:
                assert 8 <= f["height"] <= 11 and 13 <= f["width"] <= 16
            facts[split].append(f)
    a, b = facts["train"][:2]
    first, second = puzzle["train"][:2]
    assert np.array_equal(np.unique(first["input"], return_counts=True)[0],
                          np.unique(second["input"], return_counts=True)[0])
    assert np.array_equal(np.unique(first["input"], return_counts=True)[1],
                          np.unique(second["input"], return_counts=True)[1])
    assert np.array_equal(a["target"]["mask"], b["target"]["mask"])
    assert a["height"] + a["width"] == b["height"] + b["width"]
    assert (a["height"], a["width"]) != (b["height"], b["width"])
    assert a["body"] == b["accent"] and a["accent"] == b["body"]
    assert sorted(f["height"] <= f["width"] // 2 for f in (a, b)) == [False, True]
    for f in (a, b):
        assert len(f["records"]) == 4
        assert any(r is not f["target"] and r["mask"].shape == f["target"]["mask"].shape
                   for r in f["records"])
    centers = []
    for f, pair in zip((a, b), (first, second)):
        r0, r1, c0, c1 = f["target"]["box"]
        centers.append((r0 + r1 < pair["input"].shape[0],
                        c0 + c1 < pair["input"].shape[1]))
    assert all(x != y for x, y in zip(*centers))
    if len(puzzle["train"]) >= 3:
        c = facts["train"][2]
        assert (c["height"], c["width"]) == (a["height"], a["width"])
        assert c["target"]["mask"].shape != a["target"]["mask"].shape
        assert puzzle["train"][2]["input"].shape == first["input"].shape
    return None


def self_test():
    """Bounded structural, semantic and metamorphic checks over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    structures, canvases, target_shapes, output_rectangles, palettes = set(), set(), set(), set(), set()
    figure_counts, locations, parity, geometry = set(), set(), set(), set()
    edge_scenes = multi_run = diagonal_only = 0
    for puzzle in puzzles:
        for pair in puzzle["train"] + puzzle["test"]:
            a, out = pair["input"], pair["output"]
            f = _analyze_input(a)
            h, w = f["height"], f["width"]
            # A separate cell-by-cell rendering check, not an independent solver.
            expected = np.full(a.shape, f["background"], dtype=int)
            for r in range(a.shape[0]):
                for c in range(a.shape[1]):
                    d = a.shape[0] - 1 - r
                    if d < h and c < w:
                        expected[r, c] = f["accent"] if c in (d, w - 1 - d) else f["body"]
            assert np.array_equal(out, expected)
            assert np.array_equal(solve(a[:, ::-1]), out)
            assert np.array_equal(solve(a[::-1, :]), out)
            changed = a.copy()
            t = f["target"]
            r, c = map(int, np.argwhere(t["mask"] & ~t["residual"])[0])
            r += t["box"][0]
            c += t["box"][2]
            changed[r, c] = next(color for color in f["signals"] if color != changed[r, c])
            assert np.array_equal(solve(changed), out), "paired pigment must be irrelevant"
            if w - h >= 3:
                changed = a.copy()
                r, c = map(int, np.argwhere(t["residual"] & (t["crop"] == f["accent"]))[0])
                changed[r + t["box"][0], c + t["box"][2]] = f["body"]
                assert not np.array_equal(solve(changed), out), "residual pigment must matter"
            # Normalize pigments semantically to measure geometry beyond recoloring.
            normalized = np.zeros(a.shape, dtype=np.uint8)
            normalized[a == f["body"]] = 1
            normalized[a == f["accent"]] = 2
            structures.add(_key(normalized))
            canvases.add(a.shape)
            target_shapes.add(t["mask"].shape)
            output_rectangles.add((h, w))
            palettes.add((f["background"], f["body"], f["accent"]))
            figure_counts.add(len(f["records"]))
            r0, r1, c0, c1 = t["box"]
            locations.add((r0 + r1 < a.shape[0], c0 + c1 < a.shape[1]))
            parity.add(t["mask"].shape[1] % 2)
            geometry.add("separate" if 2 * (h - 1) < w - 1 else
                         "meet" if 2 * (h - 1) == w - 1 else "cross")
            edge_scenes += any(r["box"][0] == 0 or r["box"][1] == a.shape[0]
                               or r["box"][2] == 0 or r["box"][3] == a.shape[1]
                               for r in f["records"])
            for rec in f["records"]:
                mask = rec["mask"]
                multi_run += any(np.count_nonzero(np.diff(np.pad(row.astype(int), (1, 1))) == 1) > 1
                                 for row in mask)
                # Four-neighbor reachability, compared with guaranteed eight-connectivity.
                seen = {tuple(map(int, np.argwhere(mask)[0]))}
                todo = list(seen)
                while todo:
                    rr, cc = todo.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                                and mask[nr, nc] and (nr, nc) not in seen):
                            seen.add((nr, nc))
                            todo.append((nr, nc))
                diagonal_only += len(seen) < int(mask.sum())
    assert len(structures) == 210
    assert len(canvases) >= 30 and len(target_shapes) >= 20
    assert len(output_rectangles) >= 30 and len(palettes) >= 40
    assert figure_counts == {3, 4, 5} and len(locations) == 4 and parity == {0, 1}
    assert geometry == {"separate", "meet", "cross"}
    assert edge_scenes >= 25 and multi_run >= 100 and diagonal_only >= 20
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(9000 + 10 * train + test, train, test)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(1, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(123456, 5, 3), generate_puzzle(123456, 5, 3)
    assert all(_key(x[role]) == _key(y[role]) for split in ("train", "test")
               for x, y in zip(p[split], q[split]) for role in ("input", "output"))
    return {"seeds": 42, "pairs": 210, "input_structures": len(structures),
            "canvas_shapes": len(canvases), "target_shapes": len(target_shapes),
            "rectangle_shapes": len(output_rectangles), "palettes": len(palettes),
            "edge_scenes": edge_scenes, "multi_run_figures": multi_run,
            "diagonal_only_figures": diagonal_only}
