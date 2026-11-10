"""Repair reflection orbits, then label a closed contour's two bend types.

Only NumPy and the standard library are used. There are no file reads, random
choices, generated examples, or mutable generation state in solve().
"""
from hashlib import sha256
import json

import numpy as np


ERASE, CONVEX, CONCAVE = 5, 4, 2
PAYLOAD = (0, 1, 3, 6, 7, 8, 9)
HOST_SEEDS = (
    0, 676823699, 883013329, 252690709, 448512719, 961537050,
    45703320, 789585538, 469690744, 718991991, 904628198, 24017483,
    222346662, 806222546, 462223878, 777507819, 605437235, 860779200,
    28561758, 738157917, 837097184, 914528367, 289794455, 472689736,
    84244440, 929890167, 853544104, 711412728, 124569925, 590622279,
    276626004, 343496586, 425667797, 88736573, 646526543, 413351295,
    928083283, 102069729, 852520544, 923287717, 264741733, 997512845,
)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid of colors 0..9")
    return a.astype(np.int8, copy=True)


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    i = int(np.argmax(counts))
    color = int(colors[i])
    if color not in PAYLOAD or int(counts[i]) <= grid.size // 2:
        raise ValueError("Background must be an allowed color and a strict majority")
    return color


def _axis_candidates(grid, background, dimension):
    """All nontrivial folds with >=4 visible, distinct contour-pair witnesses."""
    a = grid if dimension == 0 else grid.T
    n = a.shape[0]
    result = []
    for fold in range(1, 2 * n - 2):
        first = np.arange(max(0, fold - n + 1), (fold + 1) // 2)
        left, right = a[first], a[fold - first]
        visible = (left != ERASE) & (right != ERASE)
        support = visible & (left != background)
        if np.count_nonzero(support) >= 4 and np.all(left[visible] == right[visible]):
            result.append(fold)
    return result


def _infer_axes(grid, background):
    rows = _axis_candidates(grid, background, 0)
    cols = _axis_candidates(grid, background, 1)
    if len(rows) != 1 or len(cols) != 1:
        raise ValueError("Surviving colors must determine unique supported folds")
    return rows[0], cols[0]


def _orbit(r, c, h, w, row_sum, col_sum):
    rr, cc = row_sum - r, col_sum - c
    rows = sorted({r, rr}) if 0 <= rr < h else [r]
    cols = sorted({c, cc}) if 0 <= cc < w else [c]
    return tuple((y, x) for y in rows for x in cols)


def _orbits(shape, axes):
    h, w = shape
    seen = set()
    for r in range(h):
        for c in range(w):
            if (r, c) not in seen:
                orbit = _orbit(r, c, h, w, *axes)
                seen.update(orbit)
                yield orbit


def _restore(grid):
    if np.any((grid == CONVEX) | (grid == CONCAVE)):
        raise ValueError("The two output labels are reserved from inputs")
    background = _background(grid)
    axes = _infer_axes(grid, background)
    clean = grid.copy()
    for orbit in _orbits(grid.shape, axes):
        colors = {int(grid[p]) for p in orbit if grid[p] != ERASE}
        if len(colors) != 1:
            raise ValueError("An orbit has no witness or has conflicting witnesses")
        color = next(iter(colors))
        for p in orbit:
            if grid[p] == ERASE:
                clean[p] = color
    return clean, background, axes


def _neighbors(p, cells):
    r, c = p
    return sorted(q for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                  if q in cells)


def _contour(mask):
    """Trace a single cycle; classify turns using its oriented enclosed area."""
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if len(cells) < 8 or any(len(_neighbors(p, cells)) != 2 for p in cells):
        raise ValueError("The repaired contour must be a simple orthogonal cycle")
    start = min(cells)
    path, visited = [start], {start}
    previous, current = start, _neighbors(start, cells)[0]
    while current != start:
        if current in visited:
            raise ValueError("Self-intersecting contour")
        path.append(current)
        visited.add(current)
        following = next(q for q in _neighbors(current, cells) if q != previous)
        previous, current = current, following
    if visited != cells:
        raise ValueError("Exactly one repaired contour is required")
    area = sum(c * path[(i + 1) % len(path)][0]
               - path[(i + 1) % len(path)][1] * r
               for i, (r, c) in enumerate(path))
    if not area:
        raise ValueError("Contour must enclose an interior")
    kinds = np.zeros(mask.shape, dtype=np.int8)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % len(path)]
        cross = (c - pc) * (nr - r) - (r - pr) * (nc - c)
        if cross:
            kinds[r, c] = 1 if cross * area > 0 else 2
    if np.count_nonzero(kinds == 1) - np.count_nonzero(kinds == 2) != 4:
        raise ValueError("Invalid orthogonal turning number")
    return path, kinds


def _paint(clean, kinds):
    result = clean.copy()
    result[kinds == 1] = CONVEX
    result[kinds == 2] = CONCAVE
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive both folds, repair colors, and classify geometry from input only."""
    clean, background, _ = _restore(_grid(input_grid))
    _, kinds = _contour(clean != background)
    return _paint(clean, kinds)


def _rasterize(vertices, shape):
    mask = np.zeros(shape, dtype=bool)
    cells = set()
    for p, q in zip(vertices, vertices[1:] + vertices[:1]):
        dr, dc = q[0] - p[0], q[1] - p[1]
        if (dr == 0) == (dc == 0):
            raise ValueError("Polygon edges must be nonempty and orthogonal")
        length = abs(dr) + abs(dc)
        sr, sc = int(np.sign(dr)), int(np.sign(dc))
        for t in range(length):
            cell = (p[0] + t * sr, p[1] + t * sc)
            if cell in cells:
                raise ValueError("Polygon repeats a cell")
            cells.add(cell)
            mask[cell] = True
    _contour(mask)
    return mask


def _shape(rng, steps):
    """Build a contour, not a filled image: oscillating symmetric side profiles."""
    notch = bool(rng.random() < .38)
    gaps = [int(rng.integers(2, 4)) for _ in range(steps)]
    if notch and gaps:
        gaps[0] = 4
    levels = [0]
    for gap in gaps:
        levels.append(levels[-1] + gap)
    middle = int(rng.integers(2, 5)) if steps else int(rng.integers(6, 14))
    h = 2 * levels[-1] + middle + 1
    if h > 29:
        h = 29
    largest = int(rng.integers(4, 13))
    radii = [int(rng.integers(3 if notch else 2, largest + 1))]
    for _ in range(steps):
        options = [v for v in range(2, largest + 1) if v != radii[-1]]
        radii.append(int(rng.choice(options)))
    radius, parity = max(radii), int(rng.integers(2))
    w = 2 * radius + 1 + parity
    right = [radius + parity + r for r in radii]
    left = [radius - r for r in radii]
    vertices = [(0, left[0])]
    if notch:
        half = int(rng.integers(1, radii[0] - 1))
        nl, nr = radius - half, radius + parity + half
        max_depth = (gaps[0] - 2) if gaps else (h - 3) // 2
        depth = int(rng.integers(1, max_depth + 1))
        vertices.extend([(0, nl), (depth, nl), (depth, nr), (0, nr)])
    vertices.append((0, right[0]))
    for i in range(1, len(levels)):
        vertices.extend([(levels[i], right[i - 1]), (levels[i], right[i])])
    vertices.append((h - 1 - levels[-1], right[-1]))
    for i in range(len(levels) - 1, 0, -1):
        vertices.extend([(h - 1 - levels[i], right[i - 1]),
                         (h - 1 - levels[i - 1], right[i - 1])])
    if notch:
        vertices.extend([(h - 1, nr), (h - 1 - depth, nr),
                         (h - 1 - depth, nl), (h - 1, nl)])
    vertices.append((h - 1, left[0]))
    for i in range(1, len(levels)):
        vertices.extend([(h - 1 - levels[i], left[i - 1]),
                         (h - 1 - levels[i], left[i])])
    vertices.append((levels[-1], left[-1]))
    for i in range(len(levels) - 1, 0, -1):
        vertices.extend([(levels[i], left[i - 1]), (levels[i - 1], left[i - 1])])
    # Closing vertex is implicit, and consecutive collinear vertices are harmless.
    if vertices[-1] == vertices[0]:
        vertices.pop()
    mask = _rasterize(vertices, (h, w))
    assert np.array_equal(mask, mask[::-1]) and np.array_equal(mask, mask[:, ::-1])
    return np.ascontiguousarray(np.rot90(mask, int(rng.integers(4))))


def _raw_mask(rng, shape, test):
    h, w = shape
    family = int(rng.integers(5))
    mask = np.zeros(shape, dtype=bool)
    if family == 0:
        return rng.random(shape) < (.07 if test else .035)
    for _ in range(int(rng.integers(1, 4))):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        rh, cw = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        if family == 1:
            mask[r:min(h, r + rh), c:min(w, c + cw)] = True
        elif family == 2:
            if rng.random() < .5:
                rh, cw = 1, int(rng.integers(4, max(5, w // 2 + 1)))
            else:
                rh, cw = int(rng.integers(4, max(5, h // 2 + 1))), 1
            mask[r:min(h, r + rh), c:min(w, c + cw)] = True
        elif family == 3:
            for dy in range(min(rh, h - r)):
                mask[r + dy, c:min(w, c + 1 + dy)] = True
        else:
            for _ in range(12 if test else 7):
                mask[r, c] = True
                dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                r, c = int(np.clip(r + dr, 0, h - 1)), int(np.clip(c + dc, 0, w - 1))
    return mask


def _mask(rng, clean, background, axes, kinds, hard, test):
    mask = _raw_mask(rng, clean.shape, test)
    orbits = list(_orbits(clean.shape, axes))
    groups = {name: [] for name in ("convex", "concave", "straight", "background")}
    for orbit in orbits:
        p = orbit[0]
        name = ("background" if clean[p] == background else
                "convex" if kinds[p] == 1 else
                "concave" if kinds[p] == 2 else "straight")
        if len(orbit) == 4:
            groups[name].append(orbit)
    if any(not groups[name] for name in groups) or len(groups["straight"]) < 2:
        raise ValueError("Need off-axis witnesses of both bends, straight runs, and background")
    chosen = []
    for name in ("convex", "concave", "background"):
        group = groups[name]
        chosen.append((name, group[int(rng.integers(len(group)))]))
    straight_indices = rng.choice(len(groups["straight"]), size=2, replace=False)
    chosen.extend(("straight", groups["straight"][int(i)]) for i in straight_indices)
    forced = []
    for name, orbit in chosen:
        order = [orbit[int(i)] for i in rng.permutation(4)]
        count = 3 if hard and name != "background" else 1
        erased = set(order[:count])
        forced.append((orbit, erased))
        if name in ("convex", "concave"):
            # Erase across bends, rather than just placing isolated missing pixels.
            for r, c in erased:
                for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= q[0] < clean.shape[0] and 0 <= q[1] < clean.shape[1]
                            and clean[q] != background):
                        mask[q] = True
    for orbit in orbits:
        if all(mask[p] for p in orbit):
            mask[orbit[int(rng.integers(len(orbit)))]] = False
    for orbit, erased in forced:
        for p in orbit:
            mask[p] = p in erased
    return mask


def _nearest_fails(grid, clean, background, kinds):
    """A nearest-visible-contour copying shortcut fails regardless of tie choice."""
    witnesses = np.argwhere((grid != background) & (grid != ERASE))
    for r, c in np.argwhere((grid == ERASE) & (clean != background) & (kinds == 0)):
        distance = np.abs(witnesses - np.array([r, c])).sum(axis=1)
        closest = witnesses[distance == distance.min()]
        if all(grid[tuple(p)] != clean[r, c] for p in closest):
            return True
    return False


def _features(grid, clean, background, axes, kinds):
    mask = grid == ERASE
    h, w = grid.shape
    diagonal_bends = one_fold_fringe = axis_erased = 0
    for r, c in np.argwhere(mask):
        rr, cc = axes[0] - int(r), axes[1] - int(c)
        row_in, col_in = 0 <= rr < h, 0 <= cc < w
        if row_in and col_in and rr != r and cc != c:
            diagonal_bends += int(kinds[r, c] != 0 and mask[rr, c]
                                  and mask[r, cc] and not mask[rr, cc])
        one_fold_fringe += int(row_in != col_in)
        axis_erased += int(2 * r == axes[0] or 2 * c == axes[1])
    return {
        "convex": int(np.count_nonzero(kinds == 1)),
        "concave": int(np.count_nonzero(kinds == 2)),
        "erased_convex": int(np.count_nonzero(mask & (kinds == 1))),
        "erased_concave": int(np.count_nonzero(mask & (kinds == 2))),
        "erased_straight": int(np.count_nonzero(mask & (clean != background) & (kinds == 0))),
        "erased_background": int(np.count_nonzero(mask & (clean == background))),
        "diagonal_only_bends": diagonal_bends,
        "one_fold_fringe": one_fold_fringe,
        "axis_erased": axis_erased,
        "nearest_copy_fails": bool(_nearest_fails(grid, clean, background, kinds)),
    }


def _make_pair(rng, background, palette, test, steps, hard, min_concave=0,
               require_offcenter=False):
    for _ in range(100):
        try:
            small_mask = _shape(rng, steps)
            sh, sw = small_mask.shape
            h = sh + int(rng.integers(min(8, 30 - sh) + 1))
            w = sw + int(rng.integers(min(8, 30 - sw) + 1))
            top, left = int(rng.integers(h - sh + 1)), int(rng.integers(w - sw + 1))
            contour = np.zeros((h, w), dtype=bool)
            contour[top:top + sh, left:left + sw] = small_mask
            axes = (2 * top + sh - 1, 2 * left + sw - 1)
            if require_offcenter and axes == (h - 1, w - 1):
                continue
            _, kinds = _contour(contour)
            if np.count_nonzero(kinds == 2) < min_concave:
                continue
            k = int(rng.integers(2, len(palette) + 1))
            colors = [int(x) for x in rng.choice(palette, k, replace=False)]
            if test and not set(colors) & set(palette[-2:]):
                colors[-1] = int(rng.choice(palette[-2:]))
            clean = np.full((h, w), background, dtype=np.int8)
            contour_orbits = [o for o in _orbits(clean.shape, axes) if contour[o[0]]]
            if len(contour_orbits) < k:
                continue
            values = rng.integers(k, size=len(contour_orbits))
            values[rng.permutation(len(values))[:k]] = np.arange(k)
            for orbit, color_index in zip(contour_orbits, values):
                for p in orbit:
                    clean[p] = colors[int(color_index)]
            # Original payload does not determine curvature or recoloring.
            shared_colors = (set(map(int, clean[kinds == 1]))
                             & set(map(int, clean[kinds == 2]))
                             & set(map(int, clean[contour & (kinds == 0)])))
            if not shared_colors:
                continue
            for _ in range(8):
                missing = _mask(rng, clean, background, axes, kinds, hard, test)
                grid = clean.copy()
                grid[missing] = ERASE
                if np.count_nonzero(grid == background) <= grid.size // 2:
                    continue
                try:
                    repaired, inferred_background, inferred_axes = _restore(grid)
                except ValueError:
                    continue
                if (inferred_axes != axes or inferred_background != background
                        or not np.array_equal(repaired, clean)):
                    continue
                f = _features(grid, clean, background, axes, kinds)
                if (not f["nearest_copy_fails"] or f["erased_straight"] < 2
                        or not f["erased_convex"] or not f["erased_concave"]
                        or not f["erased_background"]):
                    continue
                if hard and not f["diagonal_only_bends"]:
                    continue
                if any(not np.any((kinds == kind) & ~missing) for kind in (1, 2)):
                    continue
                return {"input": grid, "output": _paint(clean, kinds)}
        except ValueError:
            continue
    raise RuntimeError("Bounded contour construction exhausted; no silent fallback")


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    pool = [int(x) for x in rng.permutation(PAYLOAD)]
    backgrounds, training_colors, test_colors = pool[:2], pool[2:5], pool[2:]
    puzzle, seen = {"train": [], "test": []}, set()
    maximum_train_concave = 0
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            test = split == "test"
            steps = int(rng.integers(3, 5)) if test else (
                index + 1 if index < 2 else int(rng.integers(1, 3)))
            hard = test or index == 1 or (index > 1 and rng.random() < .5)
            background = backgrounds[index % 2] if not test else int(rng.choice(backgrounds))
            for _ in range(12):
                pair = _make_pair(rng, background, test_colors if test else training_colors,
                                  test, steps, hard,
                                  maximum_train_concave + 1 if test else 1,
                                  require_offcenter=not test and index == 0)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if not test:
                        maximum_train_concave = max(maximum_train_concave,
                            int(np.count_nonzero(pair["output"] == CONCAVE)))
                    break
            else:
                raise RuntimeError("Unique input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _point_inside(path, row, col):
    """Ray crossing on the polygon centerline, used only for angle validation."""
    crossings = 0
    for (r1, c1), (r2, c2) in zip(path, path[1:] + path[:1]):
        if c1 == c2 and min(r1, r2) <= row < max(r1, r2) and c1 > col:
            crossings += 1
    return crossings % 2 == 1


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, train_colors, test_colors = set(), set(), set()
    train_concaves, test_concaves = [], []
    hard_training = offcenter_training = False
    training_backgrounds = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert grid.shape == output.shape
            key = _key(grid)
            assert key not in seen
            seen.add(key)
            clean, background, axes = _restore(grid)
            path, kinds = _contour(clean != background)
            assert np.array_equal(output, _paint(clean, kinds)), "Incorrect repair or corner label"
            assert not np.any(output == ERASE)
            missing = grid == ERASE
            assert missing.any()
            # Exact preservation, including every original straight-cell color.
            unchanged = ~missing & (kinds == 0)
            assert np.array_equal(grid[unchanged], output[unchanged])
            assert np.array_equal(output[clean == background], clean[clean == background])
            assert np.array_equal(output[(clean != background) & (kinds == 0)],
                                  clean[(clean != background) & (kinds == 0)])
            assert not np.array_equal(clean, output), "Omitting A must change the answer"
            assert np.any(missing & (kinds != 0)), "Omitting B must lose erased bends"
            assert any(_point_inside(path, r, c) for r, c in np.argwhere(clean == background))
            for r, c in np.argwhere(kinds != 0):
                quadrants = sum(_point_inside(path, r + dr, c + dc)
                                for dr in (-.25, .25) for dc in (-.25, .25))
                assert quadrants == (1 if kinds[r, c] == 1 else 3), "Wrong interior angle"
            for orbit in _orbits(grid.shape, axes):
                assert len({int(clean[p]) for p in orbit}) == 1
                assert any(not missing[p] for p in orbit), "No surviving witness"
            f = _features(grid, clean, background, axes, kinds)
            assert all(f[name] > 0 for name in ("erased_convex", "erased_concave", "erased_background"))
            assert f["erased_straight"] >= 2 and f["nearest_copy_fails"]
            assert all(np.any((kinds == kind) & ~missing) for kind in (1, 2))
            assert (set(map(int, clean[kinds == 1])) & set(map(int, clean[kinds == 2]))
                    & set(map(int, clean[(clean != background) & (kinds == 0)])))
            # Identical local neighbor directions can represent either bend type.
            local_types = {}
            for r, c in np.argwhere(kinds != 0):
                signature = tuple(sorted((q[0] - r, q[1] - c)
                                         for q in _neighbors((r, c), set(path))))
                local_types.setdefault(signature, set()).add(int(kinds[r, c]))
            assert any(types == {1, 2} for types in local_types.values())
            colors = {int(x) for x in np.unique(clean[clean != background])}
            assert len(colors) >= 2
            if split == "train":
                train_colors.update(colors)
                train_concaves.append(f["concave"])
                hard_training |= f["diagonal_only_bends"] > 0
                offcenter_training |= axes != (grid.shape[0] - 1, grid.shape[1] - 1)
                training_backgrounds.add(background)
            else:
                test_colors.update(colors)
                test_concaves.append(f["concave"])
                assert f["diagonal_only_bends"] > 0
    assert hard_training, "Training must expose use of both folds at a bend"
    assert offcenter_training and len(training_backgrounds) == 2
    assert min(test_concaves) > max(train_concaves), "Missing complexity shift"
    assert test_colors - train_colors, "Missing test-local novel contour payload"


def _signature(puzzle):
    value = {s: [{k: np.asarray(p[k]).tolist() for k in ("input", "output")}
                 for p in puzzle[s]] for s in ("train", "test")}
    return sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Stream 42 seeds, retain only hashes/counters, write no files or traces."""
    signatures, shapes, palettes, backgrounds, offsets, parities = (set() for _ in range(6))
    geometries, concaves, mask_sizes, normalized = (set() for _ in range(4))
    diagonal, fringe, axis, off_center = 0, 0, 0, 0
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                clean, background, axes = _restore(grid)
                path, kinds = _contour(clean != background)
                assert np.array_equal(solve(grid.copy()), pair["output"])
                assert np.array_equal(solve(grid.tolist()), pair["output"])
                pairs += 1
                shapes.add(grid.shape)
                backgrounds.add(background)
                palettes.add(len(np.unique(clean)) - 1)
                offsets.add(tuple(int(np.sign(a - n + 1)) for a, n in zip(axes, grid.shape)))
                parities.add(tuple(a % 2 for a in axes))
                geometries.add((grid.shape, sha256((clean != background).tobytes()).hexdigest()))
                concaves.add(int(np.count_nonzero(kinds == 2)))
                mask_sizes.add(int(np.count_nonzero(grid == ERASE)))
                f = _features(grid, clean, background, axes, kinds)
                diagonal += f["diagonal_only_bends"] > 0
                fringe += f["one_fold_fringe"] > 0
                axis += f["axis_erased"] > 0
                off_center += axes != (grid.shape[0] - 1, grid.shape[1] - 1)
                roles = {}
                norm = [[roles.setdefault(int(c), len(roles)) for c in row] for row in grid]
                normalized.add(sha256(json.dumps(norm, separators=(",", ":")).encode()).hexdigest())
    assert len(signatures) == 42 and len(normalized) >= 200
    assert len(shapes) >= 50 and len(geometries) >= 150
    assert len(backgrounds) >= 6 and len(palettes) >= 3
    assert len(offsets) == 9 and len(parities) == 4
    assert len(concaves) >= 4 and len(mask_sizes) >= 30
    assert diagonal >= 84 and fringe and axis and off_center >= 100
    a, b = generate_puzzle(123456, 5, 2), generate_puzzle(123456, 5, 2)
    assert _signature(a) == _signature(b)
    # Count semantics and rejection are checked before any sampling work.
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(123 + nt + ns, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Negative checks exercise both repaired and preserved outputs.
    for eligible in (a["test"][0]["input"] == ERASE, a["test"][0]["input"] != ERASE):
        bad = {s: [{k: q[k].copy() for k in q} for q in a[s]] for s in a}
        p = tuple(np.argwhere(eligible)[0])
        bad["test"][0]["output"][p] = (int(bad["test"][0]["output"][p]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError("Corrupted output passed validation")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "grid_shapes": len(shapes), "contour_geometries": len(geometries),
            "color_normalized_inputs": len(normalized), "background_colors": sorted(backgrounds),
            "contour_palette_sizes": sorted(palettes), "concave_counts": sorted(concaves),
            "offset_sign_combinations": len(offsets), "axis_parities": len(parities),
            "diagonal_only_bend_pairs": diagonal, "one_fold_fringe_pairs": fringe,
            "erased_axis_pairs": axis, "off_center_pairs": off_center,
            "independent_solver": False, "blind_prediction": False}
