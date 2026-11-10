"""Extract, classify, and interleave bends of two concentric contours.

The specification is in task_spec.md. No parent code, example lookup, files,
network, or generation state is used by solve().
"""

from __future__ import annotations

import hashlib

import numpy as np


CONVEX = 4
CONCAVE = 2
INPUT_COLORS = (0, 1, 3, 5, 6, 7, 8, 9)
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grids must be rectangles with dimensions 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid cells must be integers")
    _require(np.all((0 <= grid) & (grid <= 9)), "grid colors must be 0..9")
    return grid


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _neighbors(cell, cells):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in _DIRECTIONS
            if (r + dr, c + dc) in cells]


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells = {start}
        stack = [start]
        while stack:
            p = stack.pop()
            for q in _neighbors(p, remaining):
                remaining.remove(q)
                cells.add(q)
                stack.append(q)
        result.append(cells)
    return result


def _inside(path, row, col):
    """Ray casting for points off the polygon; row/col may be quarter cells."""
    odd = False
    for (r1, c1), (r2, c2) in zip(path, path[1:] + path[:1]):
        if (r1 > row) != (r2 > row):
            crossing = c1 + (row - r1) * (c2 - c1) / (r2 - r1)
            if crossing > col:
                odd = not odd
    return odd


def _loop_info(cells):
    _require(len(cells) >= 8, "a loop needs at least eight cells")
    graph = {p: _neighbors(p, cells) for p in cells}
    _require(all(len(q) == 2 for q in graph.values()),
             "a contour must be an unbranched simple cycle")
    start = min(cells)
    path = [start]
    seen = {start}
    previous, current = start, min(graph[start])
    while current != start:
        _require(current not in seen, "contour revisits a cell")
        seen.add(current)
        path.append(current)
        following = [q for q in graph[current] if q != previous]
        _require(len(following) == 1, "ambiguous contour trace")
        previous, current = current, following[0]
    _require(seen == cells, "contour contains disconnected cycles")
    area = sum(c1 * r2 - c2 * r1
               for (r1, c1), (r2, c2) in zip(path, path[1:] + path[:1]))
    _require(area != 0, "zero-area contour")
    # Positive shoelace area in screen coordinates is clockwise.
    if area < 0:
        path = path[:1] + path[:0:-1]
    classes = {}
    bends, labels, straight = [], [], set()
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % len(path)]
        cross = (c - pc) * (nr - r) - (r - pr) * (nc - c)
        if cross:
            label = CONVEX if cross > 0 else CONCAVE
            bends.append((r, c))
            labels.append(label)
            classes[(r, c)] = label
        else:
            straight.add((r, c))
    _require(bends and bends[0] == start, "canonical start must be a bend")
    _require(labels.count(CONVEX) - labels.count(CONCAVE) == 4,
             "invalid orthogonal polygon turn total")
    rows, cols = zip(*cells)
    return {"cells": cells, "path": path, "bends": bends,
            "labels": labels, "classes": classes, "straight": straight,
            "bbox": (min(rows), min(cols), max(rows), max(cols))}


def _perimeter(center, radius):
    r, c = center
    lo_r, hi_r = r - radius, r + radius
    lo_c, hi_c = c - radius, c + radius
    return ([(lo_r, j) for j in range(lo_c, hi_c + 1)]
            + [(i, hi_c) for i in range(lo_r + 1, hi_r + 1)]
            + [(hi_r, j) for j in range(hi_c - 1, lo_c - 1, -1)]
            + [(i, lo_c) for i in range(hi_r - 1, lo_r, -1)])


def _weave(outer, inner):
    return [value for pair in zip(outer, inner) for value in pair]


def _analyze(input_grid):
    grid = _grid(input_grid)
    _require(not np.any((grid == CONVEX) | (grid == CONCAVE)),
             "input may not use reserved curvature labels 2 or 4")
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    _require(int(counts[index]) > grid.size // 2,
             "background must occupy strictly more than half the grid")
    background = int(colors[index])
    components = _components(grid != background)
    _require(len(components) == 3, "expected two contours and one singleton")
    singletons = [p for p in components if len(p) == 1]
    _require(len(singletons) == 1, "expected exactly one center marker")
    center = next(iter(singletons[0]))
    loops = []
    for cells in components:
        if len(cells) == 1:
            continue
        info = _loop_info(cells)
        shades = {int(grid[p]) for p in cells}
        _require(len(shades) == 1, "each source contour must be monochrome")
        info["color"] = shades.pop()
        a, b, d, e = info["bbox"]
        _require((a + d, b + e) == (2 * center[0], 2 * center[1]),
                 "both contour boxes must share the integer marker center")
        _require(_inside(info["path"], *center), "marker must be inside each loop")
        loops.append(info)
    loops.sort(key=lambda p: ((p["bbox"][2] - p["bbox"][0] + 1)
                             * (p["bbox"][3] - p["bbox"][1] + 1)), reverse=True)
    outer, inner = loops
    a, b, d, e = outer["bbox"]
    f, g, h, j = inner["bbox"]
    _require(a < f and b < g and h < d and j < e, "strictly nested boxes required")
    # A one-cell halo separates the contours, including diagonal proximity.
    halo = {(r + dr, c + dc) for r, c in outer["cells"]
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    _require(not (halo & inner["cells"]), "source contours must remain separate")
    _require(all(_inside(outer["path"], *p) for p in inner["cells"]),
             "inner contour must lie in the outer polygon")
    k = len(outer["bends"])
    _require(k == len(inner["bends"]) and k % 4 == 0,
             "equal bend counts divisible by four required")
    radius = k // 4
    cr, cc = center
    _require(0 <= cr - radius and cr + radius < grid.shape[0]
             and 0 <= cc - radius and cc + radius < grid.shape[1],
             "compact destination must fit the canvas")
    _require(all(abs(r - cr) > radius + 1 or abs(c - cc) > radius + 1
                 for r, c in inner["cells"]),
             "destination and one-cell halo must be clear of the inner loop")
    slots = _perimeter(center, radius)
    _require(len(slots) == 2 * k and len(set(slots)) == 2 * k,
             "destination perimeter has incorrect capacity")
    _require(all(grid[p] == background for p in slots),
             "destination must be empty")
    return {"background": background, "center": center, "loops": loops,
            "k": k, "radius": radius, "slots": slots}


def _render(grid, info, values=None):
    output = np.array(grid, dtype=np.int8, copy=True)
    outer, inner = info["loops"]
    for loop in info["loops"]:
        for p in loop["bends"]:
            output[p] = info["background"]
    if values is None:
        values = _weave(outer["labels"], inner["labels"])
    for p, value in zip(info["slots"], values):
        output[p] = value
    output[info["center"]] = grid[info["center"]]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the full output exclusively from the visible input."""
    grid = _grid(input_grid)
    return _render(grid, _analyze(grid))


def _alternatives(info):
    outer, inner = info["loops"]
    a, b = outer["labels"], inner["labels"]
    row_a = [outer["classes"][p] for p in sorted(outer["bends"])]
    row_b = [inner["classes"][p] for p in sorted(inner["bends"])]
    return {
        "source_colors": [value for _ in a for value in (outer["color"], inner["color"])],
        "all_convex": [CONVEX] * (2 * len(a)),
        "inner_is_hole": _weave(a, [CONCAVE if v == CONVEX else CONVEX for v in b]),
        "inner_first": _weave(b, a),
        "counterclockwise": _weave(a[:1] + a[:0:-1], b[:1] + b[:0:-1]),
        "concatenation": a + b,
        "row_major": _weave(row_a, row_b),
        "shifted_inner": _weave(a, b[1:] + b[:1]),
    }


def _witnesses(info):
    actual = _weave(*(p["labels"] for p in info["loops"]))
    return {name for name, candidate in _alternatives(info).items() if candidate != actual}


def _polygon_info(vertices, radii, protected):
    """Validate an edited polygon in coordinates relative to its center."""
    yr, xr = radii
    py, px = protected
    if len(set(vertices)) != len(vertices):
        return None
    cells = []
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        dr, dc = end[0] - start[0], end[1] - start[1]
        if (dr == 0) == (dc == 0):
            return None
        length = abs(dr) + abs(dc)
        sr, sc = int(np.sign(dr)), int(np.sign(dc))
        for step in range(length):
            r, c = start[0] + step * sr, start[1] + step * sc
            if abs(r) > yr or abs(c) > xr or (abs(r) <= py and abs(c) <= px):
                return None
            cells.append((r, c))
    if len(set(cells)) != len(cells):
        return None
    try:
        info = _loop_info(set(cells))
    except ValueError:
        return None
    if (info["bbox"] != (-yr, -xr, yr, xr)
            or not _inside(info["path"], 0, 0)):
        return None
    return info


def _corner_edit(vertices, rng):
    i = int(rng.integers(len(vertices)))
    previous, current, following = vertices[i - 1], vertices[i], vertices[(i + 1) % len(vertices)]
    l1 = abs(previous[0] - current[0]) + abs(previous[1] - current[1])
    l2 = abs(following[0] - current[0]) + abs(following[1] - current[1])
    if min(l1, l2) < 2:
        return None
    a = int(rng.integers(1, min(3, l1 - 1) + 1))
    b = int(rng.integers(1, min(3, l2 - 1) + 1))
    dr1, dc1 = (previous[0] - current[0]) // l1, (previous[1] - current[1]) // l1
    dr2, dc2 = (following[0] - current[0]) // l2, (following[1] - current[1]) // l2
    p = (current[0] + a * dr1, current[1] + a * dc1)
    q = (current[0] + b * dr2, current[1] + b * dc2)
    middle = (p[0] + b * dr2, p[1] + b * dc2)
    return vertices[:i] + [p, middle, q] + vertices[i + 1:]


def _edge_edit(vertices, rng):
    i = int(rng.integers(len(vertices)))
    start, end = vertices[i], vertices[(i + 1) % len(vertices)]
    length = abs(end[0] - start[0]) + abs(end[1] - start[1])
    if length < 4:
        return None
    u = int(rng.integers(1, length - 2))
    v = int(rng.integers(u + 2, length))
    depth = int(rng.integers(1, 4))
    sign = int(rng.choice((-1, 1)))
    dr, dc = (end[0] - start[0]) // length, (end[1] - start[1]) // length
    nr, nc = sign * dc, -sign * dr
    additions = [(start[0] + u * dr, start[1] + u * dc),
                 (start[0] + u * dr + depth * nr, start[1] + u * dc + depth * nc),
                 (start[0] + v * dr + depth * nr, start[1] + v * dc + depth * nc),
                 (start[0] + v * dr, start[1] + v * dc)]
    return vertices[:i + 1] + additions + vertices[i + 1:]


def _notched_fallback(rng, radii, protected, k):
    """Shallow construction for k<=28 with sufficient radii; no retry archive."""
    yr, xr = radii
    corners = [(-yr, -xr), (-yr, xr), (yr, xr), (yr, -xr)]
    counts = [0, 0, 0, 0]
    order = [int(i) for i in rng.permutation(4)]
    for i in range((k - 4) // 4):
        counts[order[i % 4]] += 1
    vertices = []
    for i, (start, end) in enumerate(zip(corners, corners[1:] + corners[:1])):
        vertices.append(start)
        length = abs(end[0] - start[0]) + abs(end[1] - start[1])
        dr, dc = (end[0] - start[0]) // length, (end[1] - start[1]) // length
        positions = ([] if counts[i] == 0 else [length // 2 - 1]
                     if counts[i] == 1 else [2, length - 4])
        for u in positions:
            v = u + 2
            vertices.extend([(start[0] + u * dr, start[1] + u * dc),
                             (start[0] + u * dr + dc, start[1] + u * dc - dr),
                             (start[0] + v * dr + dc, start[1] + v * dc - dr),
                             (start[0] + v * dr, start[1] + v * dc)])
    info = _polygon_info(vertices, radii, protected)
    _require(info is not None and len(info["bends"]) == k, "fallback construction failed")
    return info


def _sample_loop(rng, radii, protected, k):
    yr, xr = radii
    for _ in range(12):
        vertices = [(-yr, -xr), (-yr, xr), (yr, xr), (yr, -xr)]
        info = _polygon_info(vertices, radii, protected)
        # Even numbers of corner edits keep the final bend count divisible by four.
        cuts = 2 * int(rng.integers((k - 4) // 4 + 1))
        detours = (k - 4 - 2 * cuts) // 4
        plan = [2] * cuts + [4] * detours
        rng.shuffle(plan)
        completed = True
        for increment in plan:
            for _ in range(48):
                proposal = (_corner_edit(vertices, rng) if increment == 2
                            else _edge_edit(vertices, rng))
                if proposal is None:
                    continue
                candidate = _polygon_info(proposal, radii, protected)
                if candidate is not None and len(candidate["bends"]) == len(info["bends"]) + increment:
                    vertices, info = proposal, candidate
                    break
            else:
                completed = False
                break
        if completed and len(info["bends"]) == k:
            return info
    return _notched_fallback(rng, radii, protected, k)


def _input(rng, k, background):
    radius = k // 4
    inner_radii = tuple(int(v) for v in rng.integers(radius + 3, radius + 6, size=2))
    outer_radii = tuple(int(rng.integers(v + 3, min(v + 5, 14) + 1))
                        for v in inner_radii)
    inner = _sample_loop(rng, inner_radii, (radius + 1, radius + 1), k)
    outer = _sample_loop(rng, outer_radii, tuple(v + 1 for v in inner_radii), k)
    oy, ox = outer_radii
    margins = []
    for v in (oy, ox):
        space = 30 - (2 * v + 1)
        first = int(rng.integers(min(space, 3) + 1))
        second = int(rng.integers(min(space - first, 3) + 1))
        margins.append((first, second))
    (top, bottom), (left, right) = margins
    center = (top + oy, left + ox)
    grid = np.full((2 * oy + 1 + top + bottom, 2 * ox + 1 + left + right),
                   background, dtype=np.int8)
    choices = [v for v in INPUT_COLORS if v != background]
    outer_color = int(rng.choice(choices))
    inner_color = (outer_color if rng.random() < .3 else
                   int(rng.choice([v for v in choices if v != outer_color])))
    center_color = (int(rng.choice((outer_color, inner_color))) if rng.random() < .4 else
                    int(rng.choice([v for v in choices if v not in (outer_color, inner_color)])))
    for loop, color in ((outer, outer_color), (inner, inner_color)):
        for r, c in loop["cells"]:
            grid[r + center[0], c + center[1]] = color
    grid[center] = center_color
    return grid


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate a deterministic puzzle, with no retained state or file writes."""
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_background = int(rng.choice(INPUT_COLORS))
    for index in range(nt + nq):
        training = index < nt
        if index == 0:
            k = 8
        elif index == 1:
            k = 12
        elif training:
            k = int(rng.choice((4, 8, 12)))
        else:
            k = int(rng.choice((16, 20, 24)))
        background = (first_background if index == 0 else
                      int(rng.choice([v for v in INPUT_COLORS if v != first_background]))
                      if index == 1 else int(rng.choice(INPUT_COLORS)))
        for _ in range(48):
            grid = _input(rng, k, background)
            info = _analyze(grid)
            if index < 2 and len(_witnesses(info)) != len(_alternatives(info)):
                continue
            key = _key(grid)
            if key in seen:
                continue
            seen.add(key)
            puzzle["train" if training else "test"].append(
                {"input": grid, "output": _render(grid, info)})
            break
        else:
            raise RuntimeError("bounded generation could not find a distinct ordering witness")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pairs must have exactly input and output")
    grid, output = _grid(pair["input"]), _grid(pair["output"])
    _require(grid.shape == output.shape, "straight scaffolds require the same canvas")
    info = _analyze(grid)
    outer, inner = info["loops"]
    bg = info["background"]
    # This local interior-angle check does not use the signed-turn label test.
    for loop in info["loops"]:
        for (r, c), label in loop["classes"].items():
            quadrants = sum(_inside(loop["path"], r + dr, c + dc)
                            for dr in (-.25, .25) for dc in (-.25, .25))
            _require(quadrants == (1 if label == CONVEX else 3),
                     "curvature disagrees with bounded-interior angle")
        _require(loop["straight"], "each source loop must have straight cells")
        _require(all(output[p] == grid[p] for p in loop["straight"]),
                 "straight source cells changed")
        _require(all(output[p] == bg for p in loop["bends"]),
                 "original bends were not extracted")
    values = _weave(outer["labels"], inner["labels"])
    _require([int(output[p]) for p in info["slots"]] == values,
             "compact perimeter does not weave clockwise outer/inner curvature labels")
    _require(output[info["center"]] == grid[info["center"]], "center must be copied")
    altered = set(info["slots"]) | set(outer["bends"]) | set(inner["bends"])
    mask = np.ones(grid.shape, dtype=bool)
    for p in altered:
        mask[p] = False
    _require(np.array_equal(grid[mask], output[mask]), "unrelated source/background cells changed")
    _require(np.count_nonzero(output != bg) == np.count_nonzero(grid != bg),
             "foreground cell count must be conserved")
    _require(np.count_nonzero((output == CONVEX) | (output == CONCAVE)) == 2 * info["k"],
             "labels must occur exclusively on the new perimeter")
    _require(np.array_equal(output, solve(grid)), "input-only solver mismatch")
    # Explicit parent ablations, including A's original in-place alternative.
    _require(not np.array_equal(output, _render(grid, info, _alternatives(info)["source_colors"])),
             "omitting parent A unexpectedly preserves the answer")
    in_place = grid.copy()
    for loop in info["loops"]:
        for p, label in loop["classes"].items():
            in_place[p] = label
    _require(not np.array_equal(output, in_place), "omitting parent B preserves the answer")
    return info


def validate_generated_puzzle(puzzle):
    """Check geometry, preservation, transfer, ablations, and demonstration evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train and test must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input within puzzle")
            seen.add(key)
            infos[split].append(info)
    training = infos["train"]
    _require(len({p["k"] for p in training}) >= 2, "training must show multiple perimeter sizes")
    _require(len({p["background"] for p in training}) >= 2, "training backgrounds must vary")
    for info in training[:2]:
        _require(all(CONCAVE in p["labels"] for p in info["loops"]),
                 "initial demonstrations must expose both curvature labels")
        _require(len(_witnesses(info)) == len(_alternatives(info)),
                 "initial demonstration fails to distinguish a named ordering shortcut")
    _require(all(p["k"] > max(q["k"] for q in training) for p in infos["test"]),
             "tests must extend to a larger bend count")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes((len(puzzle[split]),)))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _canonical_turns(labels):
    variants = []
    for seq in (labels, labels[::-1]):
        variants.extend(tuple(seq[i:] + seq[:i]) for i in range(len(seq)))
    return min(variants)


def self_test():
    """Stream 42 seeds and small API checks; retain hashes/features, never grids."""
    shapes, counts, backgrounds, patterns, layouts, digests = (set() for _ in range(6))
    concave_counts, output_patterns, source_palettes = set(), set(), set()
    adjacent = rectangular = offset = edge = same_color = center_reuse = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        if seed == 0:
            _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)),
                     "seed does not reproduce the whole puzzle")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = _analyze(grid)
                shapes.add(grid.shape)
                counts.add(info["k"])
                backgrounds.add(info["background"])
                source_palettes.add(tuple(sorted(int(v) for v in np.unique(grid))))
                cr, cc = info["center"]
                a, b, d, e = info["loops"][0]["bbox"]
                offset += len({a, b, grid.shape[0] - 1 - d, grid.shape[1] - 1 - e}) > 1
                edge += a == 0 or b == 0 or d == grid.shape[0] - 1 or e == grid.shape[1] - 1
                same_color += info["loops"][0]["color"] == info["loops"][1]["color"]
                center_reuse += int(grid[cr, cc]) in {p["color"] for p in info["loops"]}
                bend_geometry = []
                for loop in info["loops"]:
                    patterns.add(_canonical_turns(loop["labels"]))
                    concave_counts.add(loop["labels"].count(CONCAVE))
                    box = loop["bbox"]
                    rectangular += box[2] - box[0] != box[3] - box[1]
                    bend_set = set(loop["bends"])
                    adjacent += any(_neighbors(p, bend_set) for p in bend_set)
                    bend_geometry.append(tuple((r - cr, c - cc) for r, c in loop["bends"]))
                layouts.add(tuple(bend_geometry))
                output_patterns.add(tuple(int(pair["output"][p]) for p in info["slots"]))
        # A role-preserving bijection of every input color cannot change the rule.
        pair = puzzle["test"][0]
        mapping = dict(zip(INPUT_COLORS, INPUT_COLORS[1:] + INPUT_COLORS[:1]))
        mapping.update({CONVEX: CONVEX, CONCAVE: CONCAVE})
        lookup = np.array([mapping[i] for i in range(10)], dtype=np.int8)
        _require(np.array_equal(solve(lookup[pair["input"]]), lookup[pair["output"]]),
                 "source palette identities affect the solution")
    _require(len(digests) == 42, "different seeds produced repeated puzzles")
    _require(counts == {4, 8, 12, 16, 20, 24}, "missing sampled bend counts")
    _require(backgrounds == set(INPUT_COLORS), "background role lacks palette diversity")
    _require(len(shapes) >= 35 and len(layouts) >= 150 and len(patterns) >= 20,
             "insufficient geometric or curvature-order diversity")
    _require(len(output_patterns) >= 100 and len(source_palettes) >= 25,
             "insufficient output-sequence or palette variation")
    _require(all(v > 0 for v in (adjacent, rectangular, offset, edge, same_color, center_reuse)),
             "missing compatible geometric or palette variation")
    _require(concave_counts == {0, 2, 4, 6, 8, 10}, "missing concavity complexity levels")
    # All legal explicit count combinations; only one puzzle is retained at a time.
    legal_counts = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(10000 + 10 * nt + nq, nt, nq)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == nq,
                     "explicit counts not honored")
            legal_counts += 1
    invalid = ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
               (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None))
    for nt, nq in invalid:
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_puzzle_digest(generate_puzzle(913, 7, 3)) ==
             _puzzle_digest(generate_puzzle(913, 7, 3)), "explicit-count nondeterminism")
    # A k=28 member lies outside the sampler's puzzle schedule but follows the rule.
    # Twelve corner cuts can be hard to pack; use six shallow detours instead.
    rng = np.random.default_rng(91028)
    inner = _notched_fallback(rng, (10, 10), (8, 8), 28)
    outer = _notched_fallback(rng, (13, 13), (11, 11), 28)
    novel = np.zeros((29, 29), dtype=np.int8)
    for loop, color in ((outer, 3), (inner, 5)):
        for r, c in loop["cells"]:
            novel[r + 14, c + 14] = color
    novel[14, 14] = 8
    _validate_pair({"input": novel, "output": solve(novel)})
    # Nested-list interchange and rejection of a materially corrupted answer.
    pair = generate_puzzle(250)["train"][0]
    _validate_pair({role: value.tolist() for role, value in pair.items()})
    bad = {"input": pair["input"], "output": pair["output"].copy()}
    position = _analyze(bad["input"])["slots"][0]
    bad["output"][position] = CONCAVE if bad["output"][position] == CONVEX else CONVEX
    try:
        _validate_pair(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted curvature answer was accepted")
    return {"seeds": 42, "distinct_puzzles": len(digests), "input_shapes": len(shapes),
            "bend_counts": sorted(counts), "backgrounds": sorted(backgrounds),
            "curvature_patterns_up_to_rotation_reflection": len(patterns),
            "bend_layouts": len(layouts), "perimeter_sequences": len(output_patterns),
            "input_palettes": len(source_palettes), "legal_count_combinations": legal_counts,
            "adjacent_bend_objects": adjacent, "rectangular_objects": rectangular,
            "offset_pairs": offset, "border_touching_pairs": edge,
            "same_color_contour_pairs": same_color, "center_color_reuse_pairs": center_reuse,
            "additional_bend_count_checked": 28,
            "independent_solver": False, "blind_prediction": False}
