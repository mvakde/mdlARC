"""Halo silhouettes: compare ideal unions of local stamps to a framed stencil.

The solver reads only the supplied grid. Generation uses no parent code or data.
"""

from __future__ import annotations

import json

import numpy as np


_DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _normalize(cells):
    cells = set(cells)
    if not cells:
        raise ValueError("empty shape")
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _footprint(seeds):
    # Deliberately unbounded: clipping and occupied cells do not change identity.
    return {(r + dr, c + dc) for r, c in seeds for dr, dc in _DIRECTIONS}


def _transform(cells, turns=0, reflect=False):
    cells = set(cells)
    if reflect:
        cells = {(r, -c) for r, c in cells}
    for _ in range(turns):
        cells = {(-c, r) for r, c in cells}
    return _normalize(cells)


def _components(cells, distance=1):
    remaining = set(cells)
    components = []
    offsets = ([(dr, dc) for dr in range(-distance, distance + 1)
                for dc in range(-distance, distance + 1) if dr or dc]
               if distance > 1 else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, found = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in offsets:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    found.add(point)
                    todo.append(point)
        components.append(tuple(sorted(found)))
    return components


def _perimeter(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}


def _frame_kind(cells):
    box = _box(cells)
    r0, c0, r1, c1 = box
    if min(r1 - r0, c1 - c0) < 6:
        return None
    actual, edge = set(cells), _perimeter(box)
    if actual == edge:
        return "closed"
    if not actual < edge:
        return None
    if len(edge - actual) == 1:
        return "gap"
    for row in (r0, r1):
        for col in (c0, c1):
            corner = {(row, c) for c in range(c0, c1 + 1)} | {
                (r, col) for r in range(r0, r1 + 1)}
            if actual == corner:
                return "corner"
    for side in ({(r0, c) for c in range(c0 + 1, c1)},
                 {(r1, c) for c in range(c0 + 1, c1)},
                 {(r, c0) for r in range(r0 + 1, r1)},
                 {(r, c1) for r in range(r0 + 1, r1)}):
        if edge - actual == side:
            return "three"
    return None


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "bad colors")
    return a.astype(int, copy=False)


def _parse(value):
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 4, "four semantic colors required")
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > grid.size // 2, "background needs a strict majority")
    components = {int(color): _components(map(tuple, np.argwhere(grid == color)))
                  for color in colors if color != bg}
    closed = [(color, comp) for color, comps in components.items() for comp in comps
              if _frame_kind(comp) == "closed"]
    _require(len(closed) == 1, "one closed frame required")
    frame, selected = closed[0]
    r0, c0, r1, c1 = _box(selected)
    inside = set(map(int, np.unique(grid[r0 + 1:r1, c0 + 1:c1]))) - {bg}
    _require(len(inside) == 1 and frame not in inside, "one stencil color inside frame")
    ink = inside.pop()
    seed = (set(map(int, colors)) - {bg, frame, ink}).pop()
    frame_boxes, stencils, styles, stones = [], [], [], set()
    selected_key = None
    covered_ink = set()
    for comp in components[frame]:
        kind = _frame_kind(comp)
        if kind is None:
            _require(len(comp) <= 3, "inert stones have at most three cells")
            stones.update(comp)
            continue
        box = _box(comp)
        a, b, d, e = box
        cells = {(r, c) for r in range(a + 1, d) for c in range(b + 1, e)
                 if grid[r, c] == ink}
        _require(cells, "empty stencil")
        lowr, lowc, highr, highc = _box(cells)
        _require(lowr >= a + 2 and lowc >= b + 2 and highr <= d - 2 and highc <= e - 2,
                 "stencil needs a background margin")
        _require(set(map(int, np.unique(grid[a + 1:d, b + 1:e]))) <= {bg, ink},
                 "unexpected cell inside stencil frame")
        key = _normalize(cells)
        if kind == "closed":
            selected_key = key
        else:
            stencils.append(key)
            styles.append(kind)
        frame_boxes.append(box)
        covered_ink.update(cells)
    _require(covered_ink == set(map(tuple, np.argwhere(grid == ink))),
             "all ink must belong to framed stencils")
    _require(all(key != selected_key for key in stencils), "decoy stencil repeats selected stencil")
    seeds = set(map(tuple, np.argwhere(grid == seed)))
    _require(seeds, "no exterior seeds")
    _require(all(not (a <= r <= d and b <= c <= e)
                 for r, c in seeds | stones for a, b, d, e in frame_boxes),
             "seeds or stones inside a frame box")
    groups = _components(seeds, distance=2)
    # A one-cell blank lane between ideal footprints makes group membership clear.
    _require(all(max(abs(r - u), abs(c - v)) >= 4
                 for i, group in enumerate(groups) for other in groups[i + 1:]
                 for r, c in group for u, v in other), "ambiguous intergroup spacing")
    footprints = [_footprint(group) for group in groups]
    keys = [_normalize(f) for f in footprints]
    matches = [i for i, key in enumerate(keys) if key == selected_key]
    return {"grid": grid, "roles": (bg, frame, seed, ink), "groups": groups,
            "footprints": footprints, "keys": keys, "target": selected_key,
            "matches": matches, "decoys": stencils, "styles": styles,
            "stones": stones, "frame_box": (r0, c0, r1, c1)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    grid = info["grid"]
    bg, frame, seed, ink = info["roles"]
    output = grid.copy()
    height, width = grid.shape
    for index in info["matches"]:
        for r, c in info["footprints"][index]:
            if 0 <= r < height and 0 <= c < width and grid[r, c] == bg:
                output[r, c] = ink
    return output


def _pattern(rng, category=None):
    if category is None:
        category = ("bar", "elbow", "step", "fork", "loop", "growth")[int(rng.integers(6))]
    if category == "bar":
        length = int(rng.integers(2, 5))
        cells = {(0, c) for c in range(0, length + 1, 2)} | {(0, length)}
    elif category in ("elbow", "anchor"):
        a = b = 2
        if category == "elbow":
            a, b = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        cells = ({(r, 0) for r in range(0, a + 1, 2)} | {(a, 0)} |
                 {(0, c) for c in range(0, b + 1, 2)} | {(0, b)})
    elif category == "step":
        cells = {(0, 0), (0, 2), (2, 2), (2, 4)}
        if rng.integers(2):
            cells.add((4, 4))
    elif category == "fork":
        cells = {(0, 0), (0, 2), (0, 4), (2, 2)}
        if rng.integers(2):
            cells.add((4, 2))
    elif category == "loop":
        cells = {(r, c) for r in (0, 2, 4) for c in (0, 2, 4)} - {(2, 2)}
        if rng.integers(2):
            cells.remove((4, 4))
    else:
        cells = {(2, 2)}
        for _ in range(int(rng.integers(2, 8))):
            choices = sorted({(r + dr, c + dc) for r, c in cells
                              for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0),
                                             (0, 2), (0, -2), (2, 0), (-2, 0))
                              if 0 <= r + dr <= 4 and 0 <= c + dc <= 4} - cells)
            cells.add(choices[int(rng.integers(len(choices)))])
    return _transform(cells, int(rng.integers(4)), bool(rng.integers(2)))


def _redundant(pattern, rng, dense=False):
    footprint = _footprint(pattern)
    a, b, d, e = _box(pattern)
    options = [(r, c) for r in range(a, d + 1) for c in range(b, e + 1)
               if (r, c) not in pattern and _footprint([(r, c)]) <= footprint]
    result = set(pattern)
    for point in options:
        if dense or rng.random() < 0.5:
            result.add(point)
    _require(_footprint(result) == footprint, "redundancy altered silhouette")
    return _normalize(result)


def _nonmatches(pattern, rng):
    target = _normalize(_footprint(pattern))
    choices, seen = [], {target}
    for turns, flip in ((1, False), (2, False), (3, False), (0, True), (1, True)):
        variant = _transform(pattern, turns, flip)
        key = _normalize(_footprint(variant))
        if key not in seen:
            seen.add(key)
            choices.append(variant)
    for _ in range(16):
        variant = _pattern(rng)
        key = _normalize(_footprint(variant))
        if key not in seen:
            seen.add(key)
            choices.append(variant)
    return choices


class _PackingError(Exception):
    pass


def _separate(box, occupied):
    a, b, d, e = box
    return all(d + 1 < r0 or r1 + 1 < a or e + 1 < c0 or c1 + 1 < b
               for r0, c0, r1, c1 in occupied)


def _place(rng, height, width, bh, bw, occupied, boundary=False):
    if boundary:
        positions = sorted({(-1, c) for c in range(-1, width - bw + 2)} |
                           {(height - bh + 1, c) for c in range(-1, width - bw + 2)} |
                           {(r, -1) for r in range(-1, height - bh + 2)} |
                           {(r, width - bw + 1) for r in range(-1, height - bh + 2)})
    else:
        positions = [(r, c) for r in range(height - bh + 1) for c in range(width - bw + 1)]
    for index in rng.permutation(len(positions)):
        r, c = positions[int(index)]
        visible = (max(0, r), max(0, c), min(height - 1, r + bh - 1),
                   min(width - 1, c + bw - 1))
        if _separate(visible, occupied):
            occupied.append(visible)
            return r, c
    raise _PackingError("no separated rectangle fits")


def _draw_frame(grid, r, c, bh, bw, frame, style, rng):
    box = (r, c, r + bh - 1, c + bw - 1)
    cells = _perimeter(box)
    a, b, d, e = box
    if style == "corner":
        row = a if rng.integers(2) else d
        col = b if rng.integers(2) else e
        cells = {(row, x) for x in range(b, e + 1)} | {(x, col) for x in range(a, d + 1)}
    elif style in ("three", "gap"):
        sides = [sorted({(a, x) for x in range(b + 1, e)}),
                 sorted({(d, x) for x in range(b + 1, e)}),
                 sorted({(x, b) for x in range(a + 1, d)}),
                 sorted({(x, e) for x in range(a + 1, d)})]
        side = sides[int(rng.integers(4))]
        cells -= set(side) if style == "three" else {side[int(rng.integers(len(side)))]}
    for point in cells:
        grid[point] = frame


def _construct(rng, pattern, roles, mode, size, decoy_count, group_count, match_count,
               boundary, blocked):
    height, width = size
    bg, frame, seed, ink = roles
    grid = np.full(size, bg, dtype=np.int8)
    occupied = []
    alternatives = _nonmatches(pattern, rng)
    decoy_patterns = alternatives[:decoy_count]
    # Place full stencils, but never place any seed inside them.
    for index, p in enumerate([pattern] + decoy_patterns):
        stencil = _normalize(_footprint(p))
        _, _, d, e = _box(stencil)
        bh, bw = d + 5 + int(rng.integers(2)), e + 5 + int(rng.integers(2))
        r, c = _place(rng, height, width, bh, bw, occupied)
        style = "closed" if index == 0 else ("corner", "three", "gap")[int(rng.integers(3))]
        _draw_frame(grid, r, c, bh, bw, frame, style, rng)
        top = r + int(rng.integers(2, bh - d - 2))
        left = c + int(rng.integers(2, bw - e - 2))
        for u, v in stencil:
            grid[top + u, left + v] = ink
    entries = []
    for i in range(match_count):
        variant = (_redundant(pattern, rng, dense=True) if mode == "bar" and i == 1
                   else pattern if i == 0 else _redundant(pattern, rng))
        entries.append((variant, True, boundary and i == 0))
    for p in decoy_patterns:
        entries.append((p, False, False))
    while len(entries) < group_count:
        p = alternatives[int(rng.integers(len(alternatives)))]
        entries.append((_redundant(p, rng), False, False))
    # Boundary copies go first, then larger footprints. Equal sizes are shuffled.
    rng.shuffle(entries)
    entries.sort(key=lambda item: (not item[2], -len(_footprint(item[0]))))
    active_footprints = []
    for p, match, edge in entries:
        _, _, d, e = _box(p)
        top, left = _place(rng, height, width, d + 3, e + 3, occupied, boundary=edge)
        centers = {(top + 1 + r, left + 1 + c) for r, c in p}
        for point in centers:
            grid[point] = seed
        if match:
            active_footprints.append(_footprint(centers))
    # Stones use the already identified frame color, so no new color key is needed.
    stone_options = sorted({p for f in active_footprints for p in f
                            if 0 <= p[0] < height and 0 <= p[1] < width and grid[p] == bg})
    if blocked:
        point = stone_options[int(rng.integers(len(stone_options)))]
        grid[point] = frame
    if rng.random() < 0.45:
        far = [(r, c) for r in range(height) for c in range(width)
               if grid[r, c] == bg and all(_separate((r, c, r, c), [box]) for box in occupied)]
        if far:
            grid[far[int(rng.integers(len(far)))]] = frame
    # Construct output from the known placements; solve reparses the actual input.
    output = grid.copy()
    for footprint in active_footprints:
        for point in footprint:
            if 0 <= point[0] < height and 0 <= point[1] < width and grid[point] == bg:
                output[point] = ink
    return {"input": grid, "output": output}


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    modes = ["bar", "anchor"]
    rng.shuffle(modes)
    modes += [None] * (num_train - 2)
    puzzle = {"train": [], "test": []}
    seen, train_keys, train_patterns = set(), set(), []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            mode = modes[index] if split == "train" else None
            if split == "test" and index > 0 and rng.random() < 0.4:
                pattern = train_patterns[int(rng.integers(len(train_patterns)))]
            else:
                for _ in range(200):
                    pattern = _pattern(rng, mode)
                    if split == "train" or index > 0 or _normalize(_footprint(pattern)) not in train_keys:
                        break
                else:
                    raise RuntimeError("could not sample an unfamiliar test silhouette")
            roles = tuple(map(int, rng.permutation(10)[:4]))
            if mode == "bar":
                decoys, matches, groups, boundary, blocked = 0, 2, 3, True, True
            elif mode == "anchor":
                decoys, matches, groups, boundary, blocked = 1, 1, 3, False, False
            else:
                decoys = int(rng.integers(2))
                matches = int(rng.integers(1, 4))
                groups = int(rng.integers(max(3, matches + 1), 6))
                boundary, blocked = bool(rng.integers(2)), bool(rng.integers(2))
            for attempt in range(120):
                # Size ranges overlap completely; packing needs, not split, set size.
                minimum = 22 if decoys else 18
                if attempt >= 30:
                    minimum = 27
                size = (int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31)))
                try:
                    pair = _construct(rng, pattern, roles, mode, size, decoys, groups,
                                      matches, boundary, blocked)
                    info = _analyze_pair(pair)
                    key = (size, pair["input"].tobytes())
                    if key in seen:
                        continue
                    _require(len(info["matches"]) == matches, "unexpected match count")
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
                except _PackingError:
                    continue
            else:
                raise RuntimeError("could not pack separated halo silhouettes")
            if split == "train":
                train_patterns.append(pattern)
                train_keys.add(_normalize(_footprint(pattern)))
    validate_generated_puzzle(puzzle)
    return puzzle


def _paint_mask(info, indices):
    """Array-based validation of coverage, separate from solve's painting loop."""
    grid = info["grid"]
    height, width = grid.shape
    covered = np.zeros(grid.shape, dtype=bool)
    for i in indices:
        seeds = np.zeros(grid.shape, dtype=bool)
        for point in info["groups"][i]:
            seeds[point] = True
        for dr, dc in _DIRECTIONS:
            r0, r1 = max(0, dr), min(height, height + dr)
            c0, c1 = max(0, dc), min(width, width + dc)
            covered[r0:r1, c0:c1] |= seeds[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    output = grid.copy()
    output[covered & (grid == info["roles"][0])] = info["roles"][3]
    return output


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
    info = _parse(pair["input"])
    inp, out = info["grid"], _grid(pair["output"])
    _require(inp.shape == out.shape, "output dimensions changed")
    _require(info["matches"] and len(info["matches"]) < len(info["groups"]),
             "positive and negative groups required")
    _require(np.array_equal(out, _paint_mask(info, info["matches"])), "incorrect halo output")
    _require(np.any(out != inp), "no visible action")
    _require(np.array_equal(inp[inp != info["roles"][0]], out[inp != info["roles"][0]]),
             "an original nonbackground cell changed")
    _require(not np.array_equal(out, _paint_mask(info, range(len(info["groups"])))),
             "ignoring shape selection does not change answer")
    height, width = inp.shape
    info["clipped"] = any(not (0 <= r < height and 0 <= c < width)
                          for i in info["matches"] for r, c in info["footprints"][i])
    info["blocked"] = any(info["stones"] & info["footprints"][i] for i in info["matches"])
    info["decoy_hit"] = any(key in info["decoys"] for key in info["keys"]
                            if key != info["target"])
    info["decompositions"] = len({_normalize(info["groups"][i]) for i in info["matches"]})
    info["count_contrast"] = len({len(info["groups"][i]) for i in info["matches"]}) > 1
    target = info["target"]
    info["concave"] = len(target) < (max(r for r, c in target) + 1) * (max(c for r, c in target) + 1)
    info["rotation_contrast"] = any(
        key != target and len(key) == len(target) and _box(key) == _box(target)
        and key in {_transform(target, k) for k in (1, 2, 3)} for key in info["keys"])
    info["overlap"] = any(sum(len(_footprint([p])) for p in info["groups"][i]) >
                          len(info["footprints"][i]) for i in info["matches"])
    return info


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = (info["grid"].shape, info["grid"].tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    for feature in ("clipped", "blocked", "decoy_hit", "count_contrast", "rotation_contrast", "overlap"):
        _require(any(i[feature] for i in train), f"missing training evidence: {feature}")
    _require(any(i["concave"] for i in train), "no nonrectangular union demonstration")
    _require(any(not i["decoys"] for i in train), "no simple selector demonstration")
    _require(len({i["target"] for i in train}) >= 2, "training silhouettes must vary")
    _require(infos["test"][0]["target"] not in {i["target"] for i in train},
             "first test must transfer to an unfamiliar silhouette")


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _counterfactual_checks(puzzle):
    # A's selector must matter even when the palette and exterior stay fixed.
    pair = next(p for p in puzzle["train"] if _analyze_pair(p)["decoy_hit"])
    info = _parse(pair["input"])
    altered = info["grid"].copy()
    bg, frame, seed, ink = info["roles"]
    frame_comps = _components(map(tuple, np.argwhere(altered == frame)))
    decoy = next(comp for comp in frame_comps if _frame_kind(comp) in ("corner", "gap", "three"))
    for point in _perimeter(_box(decoy)):
        altered[point] = frame
    r0, c0, r1, c1 = info["frame_box"]
    altered[r0, c0 + 1] = bg
    _require(not np.array_equal(solve(altered), pair["output"]), "selector swap had no effect")

    # B's union is insensitive to a redundant center, while keeping centers intact.
    pair = next(p for p in puzzle["train"] if _analyze_pair(p)["count_contrast"])
    info = _parse(pair["input"])
    for i in info["matches"]:
        group = set(info["groups"][i])
        for point in sorted(group):
            if (len(group) > 1 and len(_components(group - {point}, distance=2)) == 1
                    and _footprint(group - {point}) == _footprint(group)):
                altered = info["grid"].copy()
                altered[point] = info["roles"][0]
                expected = np.asarray(pair["output"]).copy()
                expected[point] = info["roles"][3]
                _require(np.array_equal(solve(altered), expected), "redundant-center invariance failed")
                # A whole-grid turn changes both query and candidates coherently.
                _require(np.array_equal(solve(np.rot90(info["grid"])), np.rot90(pair["output"])),
                         "whole-grid rotation failed")
                permutation = np.array([6, 2, 9, 0, 5, 8, 1, 4, 7, 3])
                _require(np.array_equal(solve(permutation[info["grid"]]), permutation[pair["output"]]),
                         "color equivariance failed")
                return
    raise AssertionError("no redundant matching center in anchor demonstration")


def self_test() -> dict:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    infos = {s: [_analyze_pair(pair) for p in puzzles for pair in p[s]] for s in ("train", "test")}
    for p in puzzles:
        validate_generated_puzzle(p)
        _counterfactual_checks(p)
        for split in ("train", "test"):
            for pair in p[split]:
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagreement")
    all_infos = infos["train"] + infos["test"]
    metrics = {"seeds": 42, "pairs": len(all_infos),
               "dimensions": len({i["grid"].shape for i in all_infos}),
               "stencils": len({i["target"] for i in all_infos}),
               "palettes": len({i["roles"] for i in all_infos}),
               "group_counts": sorted({len(i["groups"]) for i in all_infos}),
               "match_counts": sorted({len(i["matches"]) for i in all_infos}),
               "fragment_styles": sorted({s for i in all_infos for s in i["styles"]}),
               "test_clipped": sum(i["clipped"] for i in infos["test"]),
               "test_blocked": sum(i["blocked"] for i in infos["test"]),
               "test_concave": sum(i["concave"] for i in infos["test"]),
               "different_decompositions": sum(i["count_contrast"] for i in all_infos)}
    _require(metrics["dimensions"] >= 40 and metrics["stencils"] >= 25, "geometry collapsed")
    _require(metrics["palettes"] >= 100, "palette collapsed")
    _require(metrics["group_counts"] == [3, 4, 5] and metrics["match_counts"] == [1, 2, 3],
             "population diversity collapsed")
    _require(metrics["fragment_styles"] == ["corner", "gap", "three"], "fragment diversity collapsed")
    _require(min(metrics["test_clipped"], metrics["test_blocked"], metrics["test_concave"]) >= 8,
             "test behavior lacks variation")
    _require(_same(generate_puzzle(817, 6, 3), generate_puzzle(817, 6, 3)), "nondeterminism")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "ignored counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return metrics


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
