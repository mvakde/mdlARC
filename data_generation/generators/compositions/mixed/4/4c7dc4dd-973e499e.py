"""A locally demonstrated color substitution whose tiles are global support masks.

Only the input is needed to solve. No parent code, files, cached examples, or
generation metadata participate in prediction. All saved diagnostics are bounded.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return grid.astype(np.int8, copy=False)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _digest(grid):
    return hashlib.sha256(str(grid.shape).encode() + grid.astype(np.int8).tobytes()).hexdigest()


def _draw_frame(grid, box, color):
    r, c, h, w = box[:4]
    grid[r, c:c+w] = grid[r+h-1, c:c+w] = color
    grid[r:r+h, c] = grid[r:r+h, c+w-1] = color


def _inside(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return y < r and x < c and r+h < y+bh and c+w < x+bw


def _disjoint(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return r+h <= y or y+bh <= r or c+w <= x or x+bw <= c


def _rings(grid):
    """Find isolated monochrome outline components, without known frame colors."""
    height, width = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    by_color = {}
    for row in range(height):
        for col in range(width):
            if visited[row, col]:
                continue
            color = int(grid[row, col])
            stack = [(row, col)]
            visited[row, col] = True
            points = []
            while stack:
                r, c = stack.pop()
                points.append((r, c))
                for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if (0 <= y < height and 0 <= x < width and not visited[y, x]
                            and grid[y, x] == color):
                        visited[y, x] = True
                        stack.append((y, x))
            top = min(p[0] for p in points)
            bottom = max(p[0] for p in points)
            left = min(p[1] for p in points)
            right = max(p[1] for p in points)
            h, w = bottom-top+1, right-left+1
            if (min(h, w) >= 4 and len(points) == 2*h+2*w-4
                    and all(r in (top, bottom) or c in (left, right) for r, c in points)):
                by_color.setdefault(color, []).append((top, left, h, w, color))
    return by_color


def _crop(grid, box):
    r, c, h, w = box[:4]
    return grid[r+1:r+h-1, c+1:c+w-1].copy()


def _extract_scene(grid):
    """Associate four panel frames through two enclosing outlines."""
    rings = _rings(grid)
    candidates = []
    for gc, groups in rings.items():
        if len(groups) != 2 or not _disjoint(*groups):
            continue
        for fc, frames in rings.items():
            if fc == gc or len(frames) != 4:
                continue
            members = [[p for p in frames if _inside(p, g)] for g in groups]
            if any(len(m) != 2 or not _disjoint(*m) for m in members):
                continue
            if len({_key(np.asarray(p[:4])) for m in members for p in m}) != 4:
                continue
            states = []
            for group, panels in zip(groups, members):
                small, large = sorted(panels, key=lambda p: p[2]*p[3])
                source, companion = _crop(grid, small), _crop(grid, large)
                n = source.shape[0]
                if not (source.shape == (n, n) and 2 <= n <= 4
                        and companion.shape == (n*n, n*n)
                        and len(np.unique(source)) >= 2):
                    break
                states.append(dict(source=source, companion=companion,
                                   small=small, large=large, group=group))
            if len(states) != 2:
                continue
            uniform = [i for i, state in enumerate(states)
                       if len(np.unique(state["companion"])) == 1]
            if len(uniform) != 1:
                continue
            missing = uniform[0]
            ref, target = states[1-missing], states[missing]
            blank = int(target["companion"][0, 0])
            if (blank in ref["source"] or blank in target["source"]
                    or set(map(int, np.unique(ref["source"])))
                    != set(map(int, np.unique(target["source"])))):
                continue
            structural = {gc, fc}
            if structural & (set(map(int, np.unique(ref["source"]))) | {blank}):
                continue
            candidates.append(dict(reference=ref, target=target, blank=blank,
                                   group_color=gc, frame_color=fc))
    if len(candidates) != 1:
        raise ValueError(f"expected one four-panel analogy; found {len(candidates)}")
    return candidates[0]


def _infer_mapping(scene):
    """Read the ink of each category's support, checking every reference tile."""
    source = scene["reference"]["source"]
    complete = scene["reference"]["companion"]
    n = source.shape[0]
    blank = scene["blank"]
    mapping = {}
    for r in range(n):
        for c in range(n):
            color = int(source[r, c])
            tile = complete[r*n:(r+1)*n, c*n:(c+1)*n]
            support = source == color
            ink = np.unique(tile[support])
            if (len(ink) != 1 or ink[0] == blank or np.any(tile[~support] != blank)
                    or (color in mapping and mapping[color] != int(ink[0]))):
                raise ValueError("complete pair does not demonstrate color-support substitution")
            mapping[color] = int(ink[0])
    if set(mapping) != set(mapping.values()):
        raise ValueError("demonstrated ink map must permute the categorical palette")
    return mapping


def _expand(source, blank, mapping):
    """Generation path: place one entire category support in every selector tile."""
    n = source.shape[0]
    out = np.full((n*n, n*n), blank, dtype=np.int8)
    for r in range(n):
        for c in range(n):
            color = int(source[r, c])
            out[r*n:(r+1)*n, c*n:(c+1)*n] = np.where(
                source == color, mapping[color], blank)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Rebuild just the missing companion, using only visible input cells."""
    grid = _grid(input_grid)
    scene = _extract_scene(grid)
    mapping = _infer_mapping(scene)
    target = scene["target"]["source"]
    n = target.shape[0]
    # A different construction from generation: compare every pair of positions.
    same = target[:, :, None, None] == target[None, None, :, :]
    inks = np.array([[mapping[int(c)] for c in row] for row in target], dtype=np.int8)
    four_axes = np.where(same, inks[:, :, None, None], scene["blank"])
    return four_axes.transpose(0, 2, 1, 3).reshape(n*n, n*n).copy()


def _asymmetric(labels):
    """Exclude every nonidentity square symmetry, so mask orientation is taught."""
    return all(not np.array_equal(labels, transformed)
               for k in range(4)
               for transformed in ((np.rot90(labels, k), np.rot90(labels, k)[:, ::-1])
                                   if k else (labels[:, ::-1],)))


def _labels(rng, n, count, style, *, anchor=False):
    """Bounded structural sampling, with a guaranteed general scatter fallback."""
    rr, cc = np.indices((n, n))
    for attempt in range(128):
        active = style if attempt < 64 else "scatter"
        if active == "scatter":
            flat = np.r_[np.arange(count), rng.integers(count, size=n*n-count)]
            rng.shuffle(flat)
            labels = flat.reshape(n, n)
        elif active == "bands":
            labels = ((rr if rng.integers(2) else cc) + int(rng.integers(count))) % count
        elif active == "checker":
            labels = (rr+cc) % count
        elif active == "cluster":
            centers = rng.choice(n*n, count, replace=False)
            distances = ((rr[None]-centers[:, None, None]//n)**2
                         + (cc[None]-centers[:, None, None] % n)**2)
            labels = (distances + rng.random(distances.shape)*.2).argmin(axis=0)
        elif active == "mirror":
            half = rng.integers(count, size=(n, (n+1)//2))
            labels = np.c_[half, half[:, :n//2][:, ::-1]]
        else:
            raise ValueError("unknown support layout")
        counts = np.bincount(labels.ravel(), minlength=count)
        if (np.any(counts == 0) or not np.any(counts >= 2)
                or (anchor and (np.any(counts < 2) or not _asymmetric(labels)))):
            continue
        return labels.astype(np.int8)
    # Anchor: n=3, count=2 or 3. Every role repeats and orientation is visible.
    if anchor:
        fallback = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.int8)
        if count == 3:
            fallback[1, 0] = fallback[2, 2] = 2
        return fallback
    # Each role appears, with a repeated role; not a different transformation.
    flat = np.arange(n*n) % count
    rng.shuffle(flat)
    return flat.reshape(n, n).astype(np.int8)


def _minimum_group(n, vertical):
    # Two panels: n+2 and n²+2, one-cell gap, enclosing outline on each side.
    dims = (n*n+4, n*n+n+7)
    return dims[::-1] if vertical else dims


def _layout(rng, sides):
    options = []
    for v0 in (False, True):
        for v1 in (False, True):
            dims = [_minimum_group(n, v) for n, v in zip(sides, (v0, v1))]
            for axis in (0, 1):
                if (sum(d[axis] for d in dims)+1 <= 30
                        and max(d[1-axis] for d in dims) <= 30):
                    options.append(((v0, v1), dims, axis))
    if not options:
        raise ValueError("these two expansion scales cannot fit in an ARC canvas")
    verticals, minima, axis = options[int(rng.integers(len(options)))]
    cross = 1-axis
    length = int(rng.integers(sum(d[axis] for d in minima)+1, 31))
    breadth = int(rng.integers(max(d[cross] for d in minima), 31))
    slack = length-sum(d[axis] for d in minima)-1
    allocation = rng.multinomial(slack, np.full(5, .2))
    groups, panels = [], []
    cursor = int(allocation[2])
    for i, (n, vertical, minimum) in enumerate(zip(sides, verticals, minima)):
        dims = list(minimum)
        dims[axis] += int(allocation[i])
        dims[cross] = int(rng.integers(minimum[cross], breadth+1))
        start = [0, 0]
        start[axis] = cursor
        start[cross] = int(rng.integers(breadth-dims[cross]+1))
        group = (*start, *dims)
        groups.append(group)
        cursor += dims[axis]+1+int(allocation[3])
        along = 0 if vertical else 1
        across = 1-along
        p, q = n+2, n*n+2
        gap = int(rng.integers(1, dims[along]-p-q-1))
        offset = int(rng.integers(1, dims[along]-p-q-gap))
        source_first = bool(rng.integers(2))
        order = [("small", p), ("large", q)] if source_first else [("large", q), ("small", p)]
        positions = {}
        for name, size in order:
            pos = list(start)
            pos[along] += offset
            pos[across] += int(rng.integers(1, dims[across]-size))
            positions[name] = (*pos, size, size)
            offset += size+gap
        panels.append(positions)
    shape = (length, breadth) if axis == 0 else (breadth, length)
    return shape, groups, panels


def _make_pair(rng, ref_n, target_n, count, *, anchor=False, style="scatter"):
    colors = rng.permutation(10).astype(np.int8)
    background, blank, group_color, frame_color = map(int, colors[:4])
    palette = colors[4:4+count]
    reference = palette[_labels(rng, ref_n, count, "scatter", anchor=anchor)]
    target = palette[_labels(rng, target_n, count, style)]
    # A changed support layout prevents reference-blueprint copying.
    for _ in range(32):
        if not np.array_equal(reference, target):
            break
        target = palette[_labels(rng, target_n, count, "scatter")]
    if np.array_equal(reference, target):
        target = target.copy()
        different = np.argwhere(target != target[0, 0])[0]
        p = tuple(map(int, different))
        target[0, 0], target[p] = target[p], target[0, 0]
    ink_order = rng.permutation(count)
    mapping = dict(zip(map(int, palette), map(int, palette[ink_order])))
    reference_output = _expand(reference, blank, mapping)
    output = _expand(target, blank, mapping)
    ref_slot = int(rng.integers(2))
    sides = [ref_n, target_n] if ref_slot == 0 else [target_n, ref_n]
    shape, groups, frames = _layout(rng, sides)
    canvas = np.full(shape, background, dtype=np.int8)
    if rng.integers(3) == 0:
        rr, cc = np.indices(shape)
        canvas[(rr+cc) % 2 == 0] = blank
    for i, (group, frame) in enumerate(zip(groups, frames)):
        _draw_frame(canvas, group, group_color)
        occupied = reference if i == ref_slot else target
        companion = reference_output if i == ref_slot else np.full(output.shape, blank, dtype=np.int8)
        for name, content in (("small", occupied), ("large", companion)):
            box = frame[name]
            _draw_frame(canvas, box, frame_color)
            r, c, h, w = box
            canvas[r+1:r+h-1, c+1:c+w-1] = content
    return {"input": canvas, "output": output}


def _reference_twin(pair):
    """Only the informative reference companion changes in this diagnostic twin."""
    canvas = pair["input"].copy()
    scene = _extract_scene(canvas)
    mapping = _infer_mapping(scene)
    categories = sorted(mapping)
    inks = [mapping[c] for c in categories]
    rotated = inks[1:]+inks[:1]
    changed_map = dict(zip(categories, rotated))
    ref = scene["reference"]
    content = _expand(ref["source"], scene["blank"], changed_map)
    r, c, h, w = ref["large"][:4]
    canvas[r+1:r+h-1, c+1:c+w-1] = content
    return {"input": canvas,
            "output": _expand(scene["target"]["source"], scene["blank"], changed_map)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return one coherent local-analogy family, with explicit legal counts."""
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    a = _count(num_train, "num_train", 2, 10)
    b = _count(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    anchor = _make_pair(rng, 3, int(rng.choice([2, 3])), int(rng.choice([2, 3])), anchor=True)
    puzzle = {"train": [anchor, _reference_twin(anchor)], "test": []}
    used = {_key(p["input"]) for p in puzzle["train"]}
    styles = ("scatter", "bands", "checker", "cluster", "mirror")
    for split, count in (("train", a-2), ("test", b)):
        for index in range(count):
            for _ in range(32):
                if split == "test" and index == 0:
                    # Genuine held-out expansion scale; a smaller internal
                    # worked example demonstrates the very same local rule.
                    ref_n, target_n = 2, 4
                else:
                    ref_n, target_n = int(rng.choice([2, 3])), int(rng.choice([2, 3]))
                maximum = min(6, ref_n*ref_n-1, target_n*target_n-1)
                categories = int(rng.integers(2, maximum+1))
                pair = _make_pair(rng, ref_n, target_n, categories,
                                  style=styles[int(rng.integers(len(styles)))])
                key = _key(pair["input"])
                if key not in used:
                    puzzle[split].append(pair)
                    used.add(key)
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    grid, out = _grid(pair["input"]), _grid(pair["output"])
    scene = _extract_scene(grid)
    mapping = _infer_mapping(scene)
    blank = scene["blank"]
    source = scene["target"]["source"]
    n = source.shape[0]
    _require(out.shape == (n*n, n*n), "wrong target expansion size")
    _require(not np.array_equal(source, scene["reference"]["source"]), "copied reference layout")
    # Exhaustive scalar equality invariant, separate from either array method.
    for r in range(n):
        for c in range(n):
            color = int(source[r, c])
            for u in range(n):
                for v in range(n):
                    expected = mapping[color] if source[u, v] == color else blank
                    _require(out[r*n+u, c*n+v] == expected, "incorrect target support-mask cell")
    _require(np.array_equal(solve(grid), out), "input-only solve disagrees")
    colors, counts = np.unique(source, return_counts=True)
    _require(np.any(counts > 1), "needs repeated selectors")
    _require(np.count_nonzero(out != blank) == int(np.sum(counts.astype(int)**2)),
             "category frequency-square invariant failed")
    # Dropping B: a solid tile for each category must produce a different answer.
    solid = np.repeat(np.repeat(np.array([[mapping[int(x)] for x in row] for row in source]), n, 0), n, 1)
    _require(not np.array_equal(solid, out), "whole-input support concept is vacuous")
    # Repeated selectors really yield identical non-solid tiles.
    for color in colors:
        tiles = [out[r*n:(r+1)*n, c*n:(c+1)*n] for r, c in np.argwhere(source == color)]
        _require(all(np.array_equal(t, tiles[0]) for t in tiles), "same-color tiles differ")
        _require(np.count_nonzero(tiles[0] != blank) == np.count_nonzero(source == color),
                 "tile does not contain the entire target support")
    return scene, mapping


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal example counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            scene, mapping = _check_pair(pair)
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append((scene, mapping))
    # The first two demonstrations are a controlled reference-dependence witness.
    first, second = analyses["train"][:2]
    _require(np.array_equal(first[0]["target"]["source"], second[0]["target"]["source"]),
             "reference-dependence witnesses must share a target")
    _require(first[1] != second[1] and not np.array_equal(
        puzzle["train"][0]["output"], puzzle["train"][1]["output"]),
        "complete pair must causally change the answer")
    left, right = [_grid(p["input"]).copy() for p in puzzle["train"][:2]]
    _require(left.shape == right.shape, "witness canvas shapes differ")
    r, c, h, w = first[0]["reference"]["large"][:4]
    left[r+1:r+h-1, c+1:c+w-1] = right[r+1:r+h-1, c+1:c+w-1]
    _require(np.array_equal(left, right), "witness changed something outside the reference companion")
    ref = first[0]["reference"]["source"]
    _require(_asymmetric(ref) and min(np.unique(ref, return_counts=True)[1]) >= 2,
             "anchor must expose whole supports and their orientation")
    train_sizes = {s["target"]["source"].shape[0] for s, _ in analyses["train"]}
    _require(train_sizes <= {2, 3} and analyses["test"][0][0]["target"]["source"].shape == (4, 4),
             "missing held-out scale shift")


def _diagnostic_test():
    """Bounded rule-closure checks beyond the main generation schedule."""
    source = np.array([[1, 1], [1, 2]], dtype=np.int8)
    mapping = {1: 2, 2: 1}
    expected = np.array([[2, 2, 2, 2], [2, 0, 2, 0],
                         [2, 2, 0, 0], [2, 0, 0, 1]], dtype=np.int8)
    _require(np.array_equal(_expand(source, 0, mapping), expected),
             "worked nonidentity support example failed")
    # Equal category frequencies cannot establish or confuse the ink map.
    tied = np.array([[1, 2], [2, 1]], dtype=np.int8)
    scene = {"reference": {"source": tied, "companion": _expand(tied, 0, mapping)},
             "blank": 0}
    _require(_infer_mapping(scene) == mapping, "tied-frequency map inference failed")
    # These are all scale combinations that fit the two-enclosure carrier.
    # In particular, n=4 is also supported on the reference side, even though
    # main generation uses it only for the held-out target scale.
    rng = np.random.default_rng(20261023)
    profiles = 0
    styles = ("scatter", "bands", "checker", "cluster", "mirror")
    last_pair = None
    for ref_n, target_n in ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)):
        maximum = min(6, ref_n*ref_n-1, target_n*target_n-1)
        for count in range(2, maximum+1):
            pair = _make_pair(rng, ref_n, target_n, count,
                              style=styles[profiles % len(styles)])
            _check_pair(pair)
            # Every global square symmetry preserves both coordinate levels.
            for k in range(4):
                grid, out = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
                _require(np.array_equal(solve(grid), out), "rotation closure failed")
                _require(np.array_equal(solve(grid[:, ::-1]), out[:, ::-1]),
                         "reflection closure failed")
            last_pair = pair
            profiles += 1
    # A contradictory worked example must be rejected, not silently transferred.
    grid = last_pair["input"].copy()
    scene = _extract_scene(grid)
    ref = scene["reference"]
    r, c = map(int, np.argwhere(ref["companion"] != scene["blank"])[0])
    top, left = ref["large"][:2]
    grid[top+1+r, left+1+c] = scene["blank"]
    try:
        solve(grid)
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent reference support accepted")
    return profiles


def self_test():
    """42 seeds, streaming summaries, metamorphic checks and rejection probes."""
    puzzles, shapes, supports, palettes, maps, layouts, sizes, blanks = (set() for _ in range(8))
    counts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        duplicate = generate_puzzle(seed)
        _require(all(np.array_equal(p[k], q[k]) for split in puzzle
                     for p, q in zip(puzzle[split], duplicate[split]) for k in ("input", "output")),
                 "seed did not reproduce the puzzle")
        puzzles.add(tuple(_digest(p["input"]) for split in puzzle for p in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _extract_scene(pair["input"])
                mapping = _infer_mapping(scene)
                source = scene["target"]["source"]
                cats = list(dict.fromkeys(map(int, source.flat)))
                normalized = np.array([[cats.index(int(x)) for x in row] for row in source], dtype=np.int8)
                supports.add(_key(normalized))
                shapes.add(pair["input"].shape)
                palettes.add(tuple(sorted(mapping)))
                maps.add(tuple(sorted(mapping.items())))
                blanks.add(scene["blank"])
                sizes.add((split, source.shape[0]))
                layouts.add(tuple(scene[s]["small"][:4] + scene[s]["large"][:4]
                                  for s in ("reference", "target")))
                _, frequencies = np.unique(source, return_counts=True)
                counts["pairs"] += 1
                counts["singletons"] += bool(np.any(frequencies == 1))
                counts["tied_frequencies"] += len(set(frequencies)) < len(frequencies)
                counts["four_plus_categories"] += len(cats) >= 4
                counts["nonidentity_maps"] += any(k != v for k, v in mapping.items())
                counts["symmetric_targets"] += np.array_equal(source, source.T) or np.array_equal(source, source[:, ::-1])
                # Rotation, reflection and arbitrary literal recoloring commute
                # with equality-mask expansion, even for n=4 inputs.
                for transformed, expected in ((pair["input"].T, pair["output"].T),
                                               (pair["input"][::-1], pair["output"][::-1])):
                    _require(np.array_equal(solve(transformed), expected), "geometry equivariance failed")
                recolor = np.random.default_rng(seed+10000).permutation(10)
                _require(np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]]),
                         "literal palette identities affected solving")
        # A test target keeps its own support; copying the reference blueprint
        # gives at least the wrong scale on the first test of every puzzle.
        test_scene = _extract_scene(puzzle["test"][0]["input"])
        _require(test_scene["reference"]["source"].shape == (2, 2), "shift reference size")
    _require(len(puzzles) == 42 and len(supports) >= 80 and len(shapes) >= 20,
             "insufficient structural diversity")
    _require(len(palettes) >= 20 and len(maps) >= 80 and len(layouts) >= 120 and len(blanks) == 10,
             "insufficient palette/layout diversity")
    _require({("train", 2), ("train", 3), ("test", 4)} <= sizes, "missing scale variation")
    _require(all(counts[k] for k in ("singletons", "tied_frequencies", "four_plus_categories",
                                   "nonidentity_maps", "symmetric_targets")), "missing support variation")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(314159, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1),
                 (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    closure_profiles = _diagnostic_test()
    return {"seeds": 42, **{k: int(v) for k, v in counts.items()}, "distinct_puzzles": len(puzzles),
            "canvas_shapes": len(shapes), "normalized_target_supports": len(supports),
            "palettes": len(palettes), "color_maps": len(maps), "layouts": len(layouts),
            "blank_colors": len(blanks), "legal_count_combinations": 30,
            "closure_size_category_profiles": closure_profiles,
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
