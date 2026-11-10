"""Register four sparse panel fragments, then repair their common color multiset.

Only NumPy and the standard library are used. The solver reads input pixels;
no examples, seed, construction coordinates, or saved files are consulted.

Registration fixes output coordinates, not missing color identities: translation
preserves histograms. The required A-then-B execution order is implemented here,
but repairing before translating would give the same completed output. See
composition.md for this limitation of the saved proposal's causal claim.
"""
from itertools import product
import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= grid) & (grid <= 9)), "colors must be 0..9")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _quadrants(grid):
    h, w = grid.shape[0] // 2, grid.shape[1] // 2
    return [grid[:h, :w], grid[:h, w+1:],
            grid[h+1:, :w], grid[h+1:, w+1:]]


def _join(panels, structural):
    h, w = panels[0].shape
    grid = np.full((2*h+1, 2*w+1), structural, dtype=np.int8)
    for destination, panel in zip(_quadrants(grid), panels):
        destination[:] = panel
    return grid


def _hist(grid):
    return np.bincount(np.asarray(grid, dtype=np.int64).ravel(), minlength=10)


def _neighbors(point):
    r, c = point
    return ((r-1, c), (r+1, c), (r, c-1), (r, c+1))


def _connected(points):
    remaining = set(points)
    if not remaining:
        return False
    stack = [min(remaining)]
    remaining.remove(stack[0])
    while stack:
        for point in _neighbors(stack.pop()):
            if point in remaining:
                remaining.remove(point)
                stack.append(point)
    return not remaining


def _patches(panel, background, structural):
    """Exact, fully colored 2x2 patches; no literal color is a key marker."""
    found = {}
    for r in range(panel.shape[0]-1):
        for c in range(panel.shape[1]-1):
            values = tuple(int(x) for x in panel[r:r+2, c:c+2].ravel())
            if background not in values and structural not in values and len(set(values)) >= 2:
                found.setdefault(values, []).append((r, c))
    return found


def _register(panels, background, structural):
    """Find the unique relative translation supported by common exact patches.

    Layers stay separate: differing payload colors at corresponding coordinates
    are legal. Their collective tight envelope fixes the otherwise free global
    translation. Histogram agreement is deliberately not used to select offsets.
    """
    h, w = panels[0].shape
    points = [np.argwhere(panel != background) for panel in panels]
    windows = [_patches(panel, background, structural) for panel in panels]
    common = set(windows[0]).intersection(*(set(x) for x in windows[1:]))
    placements = {}
    combinations = 0
    for patch in sorted(common):
        sizes = [len(window[patch]) for window in windows]
        combinations += int(np.prod(sizes))
        _require(combinations <= 4096, "too many competing signature occurrences")
        for anchors in product(*(window[patch] for window in windows)):
            local = [p - np.array(anchor) for p, anchor in zip(points, anchors)]
            lower = np.minimum.reduce([p.min(axis=0) for p in local])
            upper = np.maximum.reduce([p.max(axis=0) for p in local])
            if tuple(upper-lower+1) != (h, w):
                continue
            shifts = tuple(tuple(int(x) for x in -np.array(anchor)-lower)
                           for anchor in anchors)
            placements[shifts] = patch
            _require(len(placements) <= 1, "ambiguous exact-patch registration")
    _require(len(placements) == 1, "no unique registration spanning the panel envelope")
    shifts = next(iter(placements))
    registered = []
    for panel, coords, shift in zip(panels, points, shifts):
        target = coords + np.array(shift)
        result = np.full((h, w), background, dtype=np.int8)
        result[target[:, 0], target[:, 1]] = panel[coords[:, 0], coords[:, 1]]
        registered.append(result)
    return registered, shifts, placements[shifts]


def _common_histogram(panels, structural):
    observed = []
    for panel in panels:
        counts = _hist(panel)
        _require(counts[structural] == 1, "each panel needs one structural placeholder")
        counts[structural] = 0
        observed.append(counts)
    target = np.maximum.reduce(observed)
    _require(int(target.sum()) == panels[0].size,
             "common multiset is ambiguous or inconsistent")
    missing = []
    for counts in observed:
        delta = target-counts
        _require(np.all(delta >= 0) and int(delta.sum()) == 1
                 and np.count_nonzero(delta) == 1, "panel has no unique single deficit")
        missing.append(int(np.flatnonzero(delta)[0]))
    return target, tuple(missing)


def _decode(input_grid):
    grid = _grid(input_grid)
    rows, cols = grid.shape
    _require(rows >= 5 and cols >= 5 and rows % 2 == cols % 2 == 1,
             "expected odd dimensions and a central cross")
    structural = int(grid[rows//2, cols//2])
    _require(np.all(grid[rows//2, :] == structural)
             and np.all(grid[:, cols//2] == structural), "incomplete central cross")
    panels = [panel.copy() for panel in _quadrants(grid)]
    counts = _hist(np.stack(panels))
    background = int(counts.argmax())
    _require(background != structural and int(counts[background])*2 > 4*panels[0].size,
             "no dominant non-cross background")
    for panel in panels:
        _require(_connected(map(tuple, np.argwhere(panel != background))),
                 "a quadrant must contain one connected sparse fragment")
    registered, shifts, patch = _register(panels, background, structural)
    histogram, missing = _common_histogram(registered, structural)
    _require(histogram[background] > panels[0].size//2, "panels must be sparse")
    _require(all(value not in (background, structural) for value in missing),
             "placeholders replace foreground payload only")
    return grid, panels, registered, shifts, patch, background, structural, histogram, missing


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Align by exact signatures, normalize the common envelope, then repair."""
    _, _, registered, _, _, _, structural, _, missing = _decode(input_grid)
    completed = [panel.copy() for panel in registered]
    for panel, color in zip(completed, missing):
        panel[panel == structural] = color
    return _join(completed, structural)


def _grow_shape(rng, kh, kw, length, direction, count):
    """A key with an independently grown arm, bends, and side branches."""
    cells = {(r, c) for r in range(kh) for c in range(kw)}
    pad_a, pad_b = [int(x) for x in rng.integers(1, 3, size=2)]
    if direction == 0:  # north
        bounds = (-length, kh-1, -pad_a, kw-1+pad_b)
        start, end = (0, int(rng.integers(kw))), (-length, int(rng.integers(kw)))
    elif direction == 1:  # east
        bounds = (-pad_a, kh-1+pad_b, 0, kw-1+length)
        start, end = (int(rng.integers(kh)), kw-1), (int(rng.integers(kh)), kw-1+length)
    elif direction == 2:  # south
        bounds = (0, kh-1+length, -pad_a, kw-1+pad_b)
        start, end = (kh-1, int(rng.integers(kw))), (kh-1+length, int(rng.integers(kw)))
    else:  # west
        bounds = (-pad_a, kh-1+pad_b, -length, kw-1)
        start, end = (int(rng.integers(kh)), 0), (int(rng.integers(kh)), -length)
    point = start
    while point != end:
        options = []
        for axis in range(2):
            if point[axis] != end[axis]:
                nxt = list(point)
                nxt[axis] += 1 if point[axis] < end[axis] else -1
                options.append(tuple(nxt))
        point = options[int(rng.integers(len(options)))]
        cells.add(point)
    _require(len(cells) <= count, "arm exceeds the foreground budget")
    r0, r1, c0, c1 = bounds
    while len(cells) < count:
        frontier = sorted({p for q in cells for p in _neighbors(q)
                           if p not in cells and r0 <= p[0] <= r1 and c0 <= p[1] <= c1})
        _require(bool(frontier), "growth domain too small")
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _copy_shortcut_fails(registered, completed, structural):
    """At least one hole cannot be filled by ANY other layer at that position."""
    for i, panel in enumerate(registered):
        r, c = np.argwhere(panel == structural)[0]
        correct = int(completed[i][r, c])
        if all(int(registered[j][r, c]) != correct for j in range(4) if j != i):
            return True
    return False


def _make_pair(rng, index, is_test, fixed_palette):
    for _ in range(160):
        palette = fixed_palette if fixed_palette is not None else rng.permutation(10)
        background, structural = int(palette[0]), int(palette[1])
        if index == 0 and not is_test:
            kh, kw = 2, 2
        elif index == 1 and not is_test:
            kh, kw = ((2, 3), (3, 2))[int(rng.integers(2))]
        else:
            kh, kw = ((2, 2), (2, 3), (3, 2))[int(rng.integers(3))]
        if is_test:
            lengths = rng.integers(4, 6, size=4)
            lengths[int(rng.integers(4))] = 5
        else:
            lengths = rng.integers(2, 4 if index == 0 else 5 if index == 1 else 6, size=4)
        count = kh*kw + int(lengths.max()) + int(rng.integers(2, 6))
        extra_count = count-kh*kw
        support_size = int(rng.integers(4 if index == 1 and not is_test else 3,
                                        min(6, extra_count)+1))
        colors = np.asarray(palette[2:2+support_size], dtype=np.int8)
        key_values = rng.choice(colors, size=(kh, kw)).astype(np.int8)
        # A nonuniform 2x2 patch is always present, with richer keys common.
        if len(set(int(x) for x in key_values[:2, :2].ravel())) < 3:
            continue
        extra_values = np.concatenate((colors, rng.choice(colors, size=extra_count-support_size)))
        extra_values = extra_values.astype(np.int8)
        order = rng.permutation(4)
        shapes = [_grow_shape(rng, kh, kw, int(lengths[d]), int(d), count) for d in order]
        lower = tuple(min(p[a] for shape in shapes for p in shape) for a in range(2))
        upper = tuple(max(p[a] for shape in shapes for p in shape) for a in range(2))
        h, w = upper[0]-lower[0]+1, upper[1]-lower[1]+1
        if not (h <= 14 and w <= 14 and 2*count < h*w):
            continue
        key_cells = {(r, c) for r in range(kh) for c in range(kw)}
        completed = []
        extra_coords = []
        for shape in shapes:
            panel = np.full((h, w), background, dtype=np.int8)
            for r, c in key_cells:
                panel[r-lower[0], c-lower[1]] = key_values[r, c]
            positions = sorted(shape-key_cells)
            values = rng.permutation(extra_values)
            for (r, c), value in zip(positions, values):
                panel[r-lower[0], c-lower[1]] = value
            completed.append(panel)
            extra_coords.append([(r-lower[0], c-lower[1]) for r, c in positions])
        if index == 0 and not is_test:
            a, b = [int(x) for x in rng.choice(colors, size=2, replace=False)]
            missing = np.array([a, a, b, b], dtype=np.int8)
        elif index == 1 and not is_test:
            missing = rng.choice(colors, size=4, replace=False)
        else:
            distinct = int(rng.integers(2, min(4, support_size)+1))
            selected = rng.choice(colors, size=distinct, replace=False)
            missing = np.concatenate((selected, rng.choice(selected, size=4-distinct)))
        rng.shuffle(missing)
        registered = []
        for panel, positions, value in zip(completed, extra_coords, missing):
            damaged = panel.copy()
            choices = [p for p in positions if int(panel[p]) == int(value)]
            hole = choices[int(rng.integers(len(choices)))]
            damaged[hole] = structural
            registered.append(damaged)
        if not _copy_shortcut_fails(registered, completed, structural):
            continue
        # Translate each whole connected support within its original quadrant.
        # A different crop corner is not the registration convention.
        fragments = []
        all_shifts = []
        for panel in registered:
            occupied = np.argwhere(panel != background)
            lo, hi = occupied.min(axis=0), occupied.max(axis=0)
            bh, bw = hi-lo+1
            possible = [(r, c) for r in range(h-int(bh)+1) for c in range(w-int(bw)+1)
                        if (r, c) != tuple(lo)]
            if not possible:
                break
            top = np.array(possible[int(rng.integers(len(possible)))])
            translated = occupied-lo+top
            fragment = np.full((h, w), background, dtype=np.int8)
            fragment[translated[:, 0], translated[:, 1]] = panel[occupied[:, 0], occupied[:, 1]]
            fragments.append(fragment)
            all_shifts.append(tuple(int(x) for x in lo-top))
        if len(fragments) != 4 or not any(dr != 0 and dc != 0 for dr, dc in all_shifts):
            continue
        inp, out = _join(fragments, structural), _join(completed, structural)
        try:
            decoded = _decode(inp)
        except ValueError:
            continue
        if decoded[3] != tuple(all_shifts):
            continue
        # Construction supplies labels independently of the input-only solver.
        if not all(np.array_equal(a, b) for a, b in zip(decoded[2], registered)):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded sampling could not find an unambiguous registered pair")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # Some puzzles reuse role colors; others recolor locally. Both conventions
    # are solved identically because the cross and background are visible.
    fixed_palette = rng.permutation(10) if bool(rng.integers(2)) else None
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(8):
                pair = _make_pair(rng, index, split == "test", fixed_palette)
                key = _key(pair["input"])
                different_demo_size = (split != "train" or index != 1
                                       or pair["input"].shape != puzzle["train"][0]["input"].shape)
                if key not in seen and different_demo_size:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded sampling could not make a distinct input")
    return puzzle


def _analyze_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    decoded = _decode(inp)
    _, fragments, registered, shifts, patch, background, structural, target, missing = decoded
    _require(inp.shape == out.shape, "grid dimensions changed")
    rows, cols = inp.shape
    _require(np.array_equal(inp[rows//2, :], out[rows//2, :])
             and np.array_equal(inp[:, cols//2], out[:, cols//2]), "cross changed")
    completed = _quadrants(out)
    for before, after, color in zip(registered, completed, missing):
        holes = before == structural
        _require(np.array_equal(before[~holes], after[~holes]), "registered arrangement changed")
        _require(np.all(after[holes] == color), "wrong categorical repair")
        _require(np.array_equal(_hist(after), target), "completed multisets differ")
    _require(all(shift != (0, 0) for shift in shifts), "a generated fragment did not move")
    _require(any(dr != 0 and dc != 0 for dr, dc in shifts), "no two-axis registration")
    _require(len({_key(x) for x in completed}) == 4, "panel arrangements must differ")
    _require(_copy_shortcut_fails(registered, completed, structural), "copying explains all repairs")
    # Explicit ablations: B alone leaves pieces misplaced; A alone leaves holes.
    without_a = [panel.copy() for panel in fragments]
    for panel, color in zip(without_a, missing):
        panel[panel == structural] = color
    _require(not np.array_equal(_join(without_a, structural), out), "A is dispensable")
    _require(not np.array_equal(_join(registered, structural), out), "B is dispensable")
    # A common corner-cropping heuristic must also fail on the joint envelope.
    corner_cropped = []
    for panel in fragments:
        occupied = np.argwhere(panel != background)
        lo = occupied.min(axis=0)
        shifted = occupied-lo
        result = np.full(panel.shape, background, dtype=np.int8)
        result[shifted[:, 0], shifted[:, 1]] = panel[occupied[:, 0], occupied[:, 1]]
        corner_cropped.append(result)
    _require(any(not np.array_equal(a, b) for a, b in zip(corner_cropped, registered)),
             "bounding boxes alone explain registration")
    return {"shape": inp.shape, "background": background, "structural": structural,
            "histogram": tuple(int(x) for x in target), "missing": missing,
            "shifts": shifts, "patch": patch,
            "fragment_shapes": tuple(tuple(int(x) for x in np.ptp(np.argwhere(p != background), axis=0)+1)
                                     for p in fragments),
            "foreground_count": int(np.count_nonzero(completed[0] != background))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(nq, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp = _grid(pair["input"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            details[split].append(_analyze_pair(pair))
    _require(any(len(set(d["missing"])) < 4 for d in details["train"]),
             "training must show repeated missing colors")
    _require(any(len(set(d["missing"])) == 4 for d in details["train"]),
             "training must show four different missing colors")
    _require(len({d["shape"] for d in details["train"]}) >= 2, "training needs geometric variation")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, cells = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(cells)
    return digest.hexdigest()


def self_test():
    """Bounded, streamed checks: 42 seeds, no files or retained puzzle corpus."""
    signatures, shapes, fragments, histograms, patterns, shifts = [set() for _ in range(6)]
    backgrounds, structural, foreground_counts, repair_diversity = [set() for _ in range(4)]
    fixed, changing, pairs_checked = 0, 0, 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _digest(puzzle)
        _require(digest == _digest(generate_puzzle(seed)), "seed not reproducible")
        signatures.add(digest)
        local_roles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
                d = _analyze_pair(pair)
                shapes.add(d["shape"])
                fragments.update(d["fragment_shapes"])
                histograms.add(d["histogram"])
                patterns.add(d["patch"])
                shifts.update(d["shifts"])
                backgrounds.add(d["background"])
                structural.add(d["structural"])
                foreground_counts.add(d["foreground_count"])
                repair_diversity.add(len(set(d["missing"])))
                local_roles.add((d["background"], d["structural"]))
                pairs_checked += 1
        fixed += len(local_roles) == 1
        changing += len(local_roles) > 1
        if seed == 0:
            first = puzzle["train"][0]
    _require(len(signatures) == 42, "duplicate puzzles across seeds")
    _require(len(shapes) >= 20 and len(fragments) >= 30, "insufficient geometry variation")
    _require(len(histograms) >= 100 and len(patterns) >= 50, "insufficient categorical/key variation")
    _require(len(shifts) >= 50 and len(foreground_counts) >= 5, "insufficient displacement/count variation")
    _require(len(backgrounds) == len(structural) == 10, "palette roles do not cover all colors")
    _require({2, 3, 4} <= repair_diversity and fixed and changing, "missing sampling regimes")
    # Metamorphic checks do not reuse generation coordinates.
    mapping = np.array([4, 8, 2, 0, 9, 6, 1, 5, 3, 7], dtype=np.int8)
    for transform in (lambda x: x.T.copy(), lambda x: x[::-1, :].copy(),
                      lambda x: mapping[x]):
        _require(np.array_equal(solve(transform(first["input"])), transform(first["output"])),
                 "orientation or palette equivariance failed")
    probe = first["input"].copy()
    solve(probe)
    _require(np.array_equal(probe, first["input"]), "solver mutated its input")

    # The two uniqueness conditions are distinct. Equal observed histograms
    # with one unknown in every panel leave the missing color undetermined.
    decoded = _decode(first["input"])
    try:
        _common_histogram([decoded[2][0]]*4, decoded[6])
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous categorical completion accepted")

    # A repeated signature can support several full-envelope translations.
    # The matcher must reject these, rather than pick the first/minimum shift.
    repeated = np.zeros((2, 10), dtype=np.int8)
    motif = np.array([[1, 2], [3, 1]], dtype=np.int8)
    repeated[:, :2] = motif
    repeated[:, 4:6] = motif
    repeated[1, 2:4] = 2
    try:
        _register([repeated]*4, 0, 9)
    except ValueError as error:
        _require("ambiguous" in str(error), "wrong registration rejection")
    else:
        raise AssertionError("ambiguous registration accepted")

    wrong = {k: v.copy() for k, v in first.items()}
    hole = tuple(np.argwhere(decoded[2][0] == decoded[6])[0])
    _quadrants(wrong["output"])[0][hole] = decoded[5]
    try:
        _analyze_pair(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect payload repair accepted")
    # Erasing all exact registration evidence must fail, even when the cross
    # and color counts are still available. A palette-collapsed input does not
    # contain an eligible multicolor patch.
    bad = first["input"].copy()
    decoded = _decode(bad)
    bg, st = decoded[5], decoded[6]
    bad[(bad != bg) & (bad != st)] = next(c for c in range(10) if c not in (bg, st))
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("unregistered fragments accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(12345, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs_checked": pairs_checked, "puzzles": len(signatures),
            "grid_shapes": len(shapes), "fragment_shapes": len(fragments),
            "histograms": len(histograms), "signature_patterns": len(patterns),
            "translations": len(shifts), "foreground_counts": len(foreground_counts),
            "background_colors": len(backgrounds), "structural_colors": len(structural),
            "fixed_role_puzzles": fixed, "changing_role_puzzles": changing,
            "semantic_checks": "generator-owned; no independent solver or blind prediction"}
