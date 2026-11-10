"""A worked square-inventory analogy chooses the layout of rank composites.

No parent code, data files, saved samples, or generation state are used by solve.
All diagnostics are bounded and run in memory; this module writes no files.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _components(grid, background):
    """Monochromatic 4-connected components, including complete edge objects."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == background:
                continue
            color = int(grid[r, c])
            todo, cells = [(r, c)], []
            seen[r, c] = True
            while todo:
                y, x = todo.pop()
                cells.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and grid[yy, xx] == color:
                        seen[yy, xx] = True
                        todo.append((yy, xx))
            ys, xs = zip(*cells)
            result.append((color, (min(ys), min(xs), max(ys)+1, max(xs)+1), cells))
    return result


def _inventory(panel, background):
    """Reject nests and frames: every nonbackground component must be solid."""
    classes = {}
    for color, (r, c, y, x), cells in _components(panel, background):
        side = y-r
        _require(side == x-c and 2 <= side <= 5 and len(cells) == side*side,
                 "inventory contains a nonsquare or fragment")
        if side in classes:
            _require(classes[side][0] == color, "same-size squares must share a color")
            classes[side] = (color, classes[side][1]+1)
        else:
            classes[side] = (color, 1)
    _require(2 <= len(classes) <= 4, "inventory must have 2..4 size classes")
    _require(len({color for color, _ in classes.values()}) == len(classes), "size colors must be distinct")
    _require(all(1 <= n <= 4 for _, n in classes.values()), "class counts must be 1..4")
    return dict(sorted(classes.items()))


def _rank_sizes(classes):
    return [sorted((s for s, (_, n) in classes.items() if n >= rank), reverse=True)
            for rank in range(1, max(n for _, n in classes.values())+1)]


def _strip_width(classes):
    return sum(active[0]+1 for active in _rank_sizes(classes))


def _render(classes, shape, background, corner, vertical):
    """Ranks increase left-to-right; corner/vertical are demonstrated locally."""
    h, w = shape
    tallest = max(classes)
    _require(h >= tallest+1 and w >= _strip_width(classes), "companion panel is too small")
    result = np.full(shape, background, dtype=np.int8)
    top = 0 if vertical == 0 else h-tallest-1
    cursor = 0
    for active in _rank_sizes(classes):
        side = active[0]
        for s in active:
            row = 0 if corner < 2 else side-s
            col = 0 if corner % 2 == 0 else side-s
            result[top+row:top+row+s, cursor+col:cursor+col+s] = classes[s][0]
        cursor += side+1
    return result


def _frame_panels(grid, background):
    """Find two joined double frames by geometry, never by literal color."""
    groups = []
    for color, (r, c, y, x), cells in _components(grid, background):
        h, w = y-r, x-c
        if h < 7 or w < 7:
            continue
        mask = grid[r:y, c:x] == color
        if not (mask[0].all() and mask[-1].all() and mask[:, 0].all() and mask[:, -1].all()):
            continue
        rows = [i for i in range(2, h-2) if mask[i].all()]
        cols = [i for i in range(2, w-2) if mask[:, i].all()]
        if len(rows)+len(cols) != 1:
            continue
        expected = np.zeros((h, w), dtype=bool)
        expected[[0, -1], :] = True
        expected[:, [0, -1]] = True
        if rows:
            d = rows[0]
            expected[d] = True
            boxes = [(r+1, c+1, r+d, x-1), (r+d+1, c+1, y-1, x-1)]
            orientation = "vertical"
        else:
            d = cols[0]
            expected[:, d] = True
            boxes = [(r+1, c+1, y-1, c+d), (r+1, c+d+1, y-1, x-1)]
            orientation = "horizontal"
        if len(cells) != int(expected.sum()) or not np.array_equal(mask, expected):
            continue
        panels = [grid[a:b, d:e].copy() for a, d, b, e in boxes]
        groups.append(dict(box=(r, c, y, x), boxes=boxes, panels=panels,
                           color=color, orientation=orientation))
    _require(len(groups) == 2, "expected exactly two divided grouping frames")
    _require(groups[0]["color"] != groups[1]["color"], "group frame colors must differ")
    a, b = [g["box"] for g in groups]
    _require(a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], "groups overlap")
    frame_colors = {g["color"] for g in groups}
    for g in groups:
        for panel in g["panels"]:
            _require(not (set(map(int, np.unique(panel))) & frame_colors), "frame colors cannot be payload")
    inside = np.zeros(grid.shape, dtype=bool)
    for g in groups:
        r, c, y, x = g["box"]
        inside[r:y, c:x] = True
    _require(np.all(grid[~inside] == background), "exterior must be neutral")
    return groups


def _analyze(input_grid):
    grid = _grid(input_grid)
    hist = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(hist))
    _require(np.count_nonzero(hist == hist[background]) == 1, "neutral must be uniquely dominant")
    groups = _frame_panels(grid, background)
    blanks = [(i, j) for i, g in enumerate(groups) for j, p in enumerate(g["panels"])
              if np.all(p == background)]
    _require(len(blanks) == 1, "expected exactly one blank companion")
    target_group, blank_slot = blanks[0]
    occupied = groups[target_group]["panels"][1-blank_slot]
    target = _inventory(occupied, background)
    blank_shape = groups[target_group]["panels"][blank_slot].shape
    reference = groups[1-target_group]["panels"]
    interpretations = []
    for slot in range(2):
        try:
            classes = _inventory(reference[slot], background)
        except ValueError:
            continue
        companion = reference[1-slot]
        # A complete pair must distinguish nesting and repeated-rank exhaustion.
        _require(len(set(n for _, n in classes.values())) > 1 and max(n for _, n in classes.values()) >= 2,
                 "reference must demonstrate unequal multiplicities")
        for corner in range(4):
            for vertical in range(2):
                try:
                    prediction = _render(classes, companion.shape, background, corner, vertical)
                except ValueError:
                    continue
                if np.array_equal(companion, prediction):
                    interpretations.append((classes, corner, vertical, slot))
    _require(len(interpretations) == 1, "reference must uniquely demonstrate the layout")
    classes, corner, vertical, reference_slot = interpretations[0]
    return dict(background=background, groups=groups, reference=classes, target=target,
                corner=corner, vertical=vertical, target_group=target_group,
                blank_slot=blank_slot, reference_slot=reference_slot,
                occupied=occupied, blank_shape=blank_shape)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the worked analogy and return only the missing framed interior."""
    facts = _analyze(input_grid)
    return _render(facts["target"], facts["blank_shape"], facts["background"],
                   facts["corner"], facts["vertical"])


def _pack(rng, shape, classes, background):
    """At most 28 independent tiny packing attempts; retain no failed grids."""
    h, w = shape
    items = [(s, color) for s, (color, n) in classes.items() for _ in range(n)]
    for _ in range(28):
        grid = np.full(shape, background, dtype=np.int8)
        blocked = np.zeros(shape, dtype=np.int16)
        order = sorted(items, key=lambda item: -item[0])
        for s, color in order:
            if s > min(shape):
                break
            prefix = np.pad(blocked.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            windows = prefix[s:, s:]-prefix[:-s, s:]-prefix[s:, :-s]+prefix[:-s, :-s]
            positions = np.argwhere(windows == 0)
            if not len(positions):
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            grid[r:r+s, c:c+s] = color
            blocked[max(0, r-1):min(h, r+s+1), max(0, c-1):min(w, c+s+1)] = 1
        else:
            return grid
    raise ValueError("inventory packing did not fit")


def _sample_classes(rng, colors, profile=None, complex_case=False, introductory=False):
    if profile is None:
        pool = np.arange(2, 5 if introductory else 6)
        k = int(rng.integers(2, min(4, len(pool))+1))
        sides = sorted(map(int, rng.choice(pool, size=k, replace=False)))
        count_limit = 4 if introductory else (3 if k == 4 else 5)
        counts = list(map(int, rng.integers(1, count_limit, size=k)))
        if len(set(counts)) == 1:
            counts[int(rng.integers(k))] = 1 if counts[0] > 1 else 2
        if complex_case and k < 4 and 5 not in sides and max(counts) < 4:
            counts[int(rng.integers(k))] = 4
    else:
        sides, counts = profile
    chosen = list(map(int, rng.choice(colors, len(sides), replace=False)))
    return {s: (color, n) for s, color, n in zip(sides, chosen, counts)}


def _draw_group(panels, color, orientation, reverse):
    a, b = panels[::-1] if reverse else panels
    if orientation == "horizontal":
        h = a.shape[0]+2
        grid = np.full((h, a.shape[1]+b.shape[1]+3), color, dtype=np.int8)
        grid[1:-1, 1:a.shape[1]+1] = a
        grid[1:-1, a.shape[1]+2:-1] = b
    else:
        w = a.shape[1]+2
        grid = np.full((a.shape[0]+b.shape[0]+3, w), color, dtype=np.int8)
        grid[1:a.shape[0]+1, 1:-1] = a
        grid[a.shape[0]+2:-1, 1:-1] = b
    return grid


def _make_scene(rng, palette, orientation, scaffold=False, shifted=False):
    for _ in range(120):
        background, frame1, frame2 = map(int, palette[:3])
        ref = _sample_classes(rng, palette[3:], ((2, 3, 4), (3, 1, 2)) if scaffold else None,
                              introductory=not shifted)
        target = _sample_classes(rng, palette[3:], ((2, 3, 4), (1, 3, 2)) if scaffold else None,
                                 complex_case=shifted, introductory=not shifted)
        if shifted and not (len(target) == 4 or 5 in target or max(n for _, n in target.values()) == 4):
            continue
        widths = [_strip_width(cl) for cl in (ref, target)]
        if max(widths) > 15:
            continue
        if orientation == "horizontal":
            inv_shapes, comp_shapes = [], []
            for cl, width in zip((ref, target), widths):
                height = 12 if scaffold else int(rng.integers(10, 13))
                max_inv_width = 27-width
                if max_inv_width < max(cl):
                    break
                inv_width = max_inv_width
                if not scaffold and max_inv_width >= 9:
                    inv_width = int(rng.integers(max(9, max(cl)), max_inv_width+1))
                extra = int(rng.integers(0, min(2, max_inv_width-inv_width)+1))
                inv_shapes.append((height, inv_width))
                comp_shapes.append((height, width+extra))
            if len(inv_shapes) != 2:
                continue
        else:
            # Side-by-side enclosures, each split into an inventory and a strip.
            if sum(widths) > 25:
                continue
            inv_shapes, comp_shapes = [], []
            remaining = 25-sum(widths)
            for cl, width in zip((ref, target), widths):
                extra = int(rng.integers(0, min(2, remaining)+1))
                remaining -= extra
                width += extra
                ch = max(cl)+int(rng.integers(2, 4))
                ih = int(rng.integers(11, 28-ch))
                inv_shapes.append((ih, width))
                comp_shapes.append((ch, width))
        try:
            inventories = [_pack(rng, shape, cl, background) for shape, cl in zip(inv_shapes, (ref, target))]
        except ValueError:
            continue
        corner, vertical = int(rng.integers(4)), int(rng.integers(2))
        reference_strip = _render(ref, comp_shapes[0], background, corner, vertical)
        blank = np.full(comp_shapes[1], background, dtype=np.int8)
        group1 = _draw_group([inventories[0], reference_strip], frame1, orientation, bool(rng.integers(2)))
        group2 = _draw_group([inventories[1], blank], frame2, orientation, bool(rng.integers(2)))
        groups = [group1, group2]
        rng.shuffle(groups)
        if orientation == "horizontal":
            min_h, min_w = sum(g.shape[0] for g in groups)+1, max(g.shape[1] for g in groups)
        else:
            min_h, min_w = max(g.shape[0] for g in groups), sum(g.shape[1] for g in groups)+1
        if min_h > 30 or min_w > 30:
            continue
        h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
        grid = np.full((h, w), background, dtype=np.int8)
        axis = 0 if orientation == "horizontal" else 1
        slack = (h if axis == 0 else w)-sum(g.shape[axis] for g in groups)-1
        margins = rng.multinomial(slack, [1/3]*3)
        along = int(margins[0])
        for index, group in enumerate(groups):
            cross = int(rng.integers((w if axis == 0 else h)-group.shape[1-axis]+1))
            r, c = (along, cross) if axis == 0 else (cross, along)
            grid[r:r+group.shape[0], c:c+group.shape[1]] = group
            along += group.shape[axis]+1+(int(margins[1]) if index == 0 else 0)
        expected = _render(target, comp_shapes[1], background, corner, vertical)
        try:
            facts = _analyze(grid)
        except ValueError:
            continue
        _require(facts["target"] == target and facts["reference"] == ref, "packing changed inventory")
        _require(np.array_equal(solve(grid), expected), "visible analogy disagrees with construction")
        return {"input": grid, "output": expected}
    raise RuntimeError("bounded scene construction exhausted")


def _replace_reference_layout(pair, corner, vertical):
    """Only the reference companion changes; target and every frame stay fixed."""
    facts = _analyze(pair["input"])
    grid = pair["input"].copy()
    group = facts["groups"][1-facts["target_group"]]
    slot = 1-facts["reference_slot"]
    r, c, y, x = group["boxes"][slot]
    grid[r:y, c:x] = _render(facts["reference"], (y-r, x-c), facts["background"], corner, vertical)
    out = _render(facts["target"], facts["blank_shape"], facts["background"], corner, vertical)
    return {"input": grid, "output": out}


def _reference_twin(pair, rng):
    facts = _analyze(pair["input"])
    corner = (facts["corner"]+int(rng.integers(1, 4))) % 4
    return _replace_reference_layout(pair, corner, 1-facts["vertical"])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Deterministic generation with explicit counts and no on-disk artifacts."""
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    stable = bool(rng.integers(2))
    first = _make_scene(rng, palette, "horizontal", scaffold=True)
    puzzle = {"train": [first, _reference_twin(first, rng)], "test": []}
    seen = {_key(pair["input"]) for pair in puzzle["train"]}
    for split, count in (("train", a-2), ("test", b)):
        for _ in range(count):
            for _attempt in range(20):
                local_palette = palette if stable else rng.permutation(10)
                orientation = "horizontal" if rng.integers(2) else "vertical"
                pair = _make_scene(rng, local_palette, orientation, shifted=split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input retries exhausted")
    return puzzle


def _certify_strip(strip, classes, background, corner, vertical):
    """Cellwise certificate: the smallest eligible covering size wins each cell."""
    h, w = strip.shape
    top = 0 if vertical == 0 else h-max(classes)-1
    expected_nonbg = np.zeros(strip.shape, dtype=bool)
    cursor = 0
    seen_by_class = {s: 0 for s in classes}
    for rank, active in enumerate(_rank_sizes(classes), 1):
        big = active[0]
        for s in active:
            seen_by_class[s] += 1
            _require(classes[s][1] >= rank, "rank threshold violated")
        for r in range(big):
            for c in range(big):
                covered = [s for s in active if (r < s if corner < 2 else r >= big-s)
                           and (c < s if corner % 2 == 0 else c >= big-s)]
                _require(bool(covered), "uncovered composite cell")
                _require(strip[top+r, cursor+c] == classes[min(covered)][0], "incorrect nested overwrite")
                expected_nonbg[top+r, cursor+c] = True
        _require(np.all(strip[:, cursor+big] == background), "missing rank separator")
        cursor += big+1
    _require(all(seen_by_class[s] == classes[s][1] for s in classes), "square inventory was not conserved")
    _require(np.all(strip[~expected_nonbg] == background), "nonneutral padding")
    _require(np.all(strip[-1] == background), "missing bottom neutral row")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, facts_by_split = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(grid)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            facts = _analyze(grid)
            _require(out.shape == facts["blank_shape"], "output must be the blank interior")
            _certify_strip(out, facts["target"], facts["background"], facts["corner"], facts["vertical"])
            group = facts["groups"][1-facts["target_group"]]
            _certify_strip(group["panels"][1-facts["reference_slot"]], facts["reference"],
                           facts["background"], facts["corner"], facts["vertical"])
            _require(np.array_equal(solve(grid), out), "input-only solver disagreement")
            _require(len(set(n for _, n in facts["target"].values())) > 1, "target must expose exhaustion")
            if split == "train":
                for classes in (facts["reference"], facts["target"]):
                    _require(set(classes) <= {2, 3, 4} and max(n for _, n in classes.values()) <= 3,
                             "training must reserve larger inventory features for testing")
            facts_by_split[split].append(facts)
    x, y = facts_by_split["train"][:2]
    _require(np.array_equal(x["occupied"], y["occupied"]) and x["blank_shape"] == y["blank_shape"],
             "first demonstrations must hold target evidence fixed")
    _require(x["corner"] != y["corner"] and x["vertical"] != y["vertical"], "reference layouts must change")
    _require(not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"]),
             "analogy must affect the answer")
    _require(tuple(n for _, n in x["reference"].values()) == (3, 1, 2), "missing reference exhaustion scaffold")
    _require(tuple(n for _, n in x["target"].values()) == (1, 3, 2), "missing target exhaustion scaffold")
    for f in facts_by_split["test"]:
        _require(len(f["target"]) == 4 or 5 in f["target"] or max(n for _, n in f["target"].values()) == 4,
                 "test must recombine a larger inventory feature")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """42 seeds, bounded summaries, causal probes, count and mutation checks."""
    # A handwritten count/exhaustion example anchors the pixel convention.
    fixture_classes = {2: (1, 1), 3: (2, 3), 4: (3, 2)}
    fixture = np.array([[int(c) for c in row] for row in (
        "11230222302220", "11230222302220", "22230222302220",
        "33330333300000", "00000000000000")], dtype=np.int8)
    _require(np.array_equal(_render(fixture_classes, fixture.shape, 0, 0, 0), fixture),
             "handwritten threshold/nesting fixture failed")
    _certify_strip(fixture, fixture_classes, 0, 0, 0)
    shapes, profiles, palettes, modes, layouts, signatures = (set() for _ in range(6))
    corners, leaders, class_counts, sizes = set(), set(), set(), set()
    edge_squares = 0
    orientations, reference_slots, target_slots, reference_groups = (set() for _ in range(4))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _puzzle_digest(puzzle)
        signatures.add(signature)
        _require(signature == _puzzle_digest(generate_puzzle(seed)), "seed not reproducible")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _analyze(pair["input"])
                shapes.add((pair["input"].shape, pair["output"].shape))
                profile = tuple((s, n) for s, (_, n) in f["target"].items())
                profiles.add(profile)
                palettes.add((f["background"], tuple(color for color, _ in f["target"].values())))
                corners.add(f["corner"])
                modes.add((f["corner"], f["vertical"]))
                layouts.add(tuple((g["orientation"], g["box"], tuple(g["boxes"])) for g in f["groups"]))
                max_count = max(n for _, n in f["target"].values())
                leaders.update(s for s, (_, n) in f["target"].items() if n == max_count)
                class_counts.add(len(f["target"]))
                sizes.update(f["target"])
                orientations.update(g["orientation"] for g in f["groups"])
                reference_slots.add(f["reference_slot"])
                target_slots.add(1-f["blank_slot"])
                reference_groups.add(1-f["target_group"])
                for _, (r, c, y, x), _ in _components(f["occupied"], f["background"]):
                    edge_squares += r == 0 or c == 0 or y == f["occupied"].shape[0] or x == f["occupied"].shape[1]
                # Ignoring multiplicities (all classes at all ranks) changes a tail.
                bad = {s: (color, max_count) for s, (color, _) in f["target"].items()}
                needed = _strip_width(bad)
                alternative = _render(bad, (pair["output"].shape[0], max(needed, pair["output"].shape[1])),
                                      f["background"], f["corner"], f["vertical"])
                alternative = alternative[:, :pair["output"].shape[1]]
                _require(not np.array_equal(alternative, pair["output"]),
                         "same-shape threshold ablation has no effect")
                once = {s: (color, 1) for s, (color, _) in f["target"].items()}
                _require(not np.array_equal(_render(once, pair["output"].shape, f["background"],
                                                    f["corner"], f["vertical"]), pair["output"]),
                         "ignoring repeated occurrences has no effect")
        # Full-canvas twin: changes are confined to the reference companion.
        a, b = puzzle["train"][:2]
        f = _analyze(a["input"])
        g = f["groups"][1-f["target_group"]]
        r, c, y, x = g["boxes"][1-f["reference_slot"]]
        allowed = np.zeros(a["input"].shape, dtype=bool)
        allowed[r:y, c:x] = True
        _require(np.all(a["input"][~allowed] == b["input"][~allowed]), "twin changed target evidence")
        # Palette equivariance verifies semantic roles, including a nonzero neutral.
        mapping = np.random.default_rng(seed+500).permutation(10)
        _require(np.array_equal(solve(mapping[a["input"]]), mapping[a["output"]]), "literal-color shortcut")
        original = a["input"].copy()
        _require(np.array_equal(solve(original), a["output"]) and np.array_equal(original, a["input"]),
                 "solve must not mutate its input")
    _require(len(signatures) == 42 and len(profiles) >= 35 and len(shapes) >= 40, "insufficient structural variation")
    _require(len(palettes) >= 40 and len(layouts) >= 100, "insufficient palette/layout variation")
    _require(corners == set(range(4)) and len(modes) == 8, "layout parameter coverage")
    _require(class_counts == {2, 3, 4} and sizes == {2, 3, 4, 5} and leaders == sizes, "inventory coverage")
    _require(edge_squares > 30, "complete panel-edge squares were not exercised")
    _require(orientations == {"horizontal", "vertical"}, "both enclosure layouts must occur")
    _require(reference_slots == target_slots == reference_groups == {0, 1}, "panel role/order coverage")
    # Isolate corner and vertical placement, rather than changing them together.
    base = generate_puzzle(0, 2, 1)["train"][0]
    answers = set()
    for corner in range(4):
        for vertical in range(2):
            probe = _replace_reference_layout(base, corner, vertical)
            _require(np.array_equal(solve(probe["input"]), probe["output"]), "reference dependence failed")
            answers.add(_key(probe["output"]))
    _require(len(answers) == 8, "same target must admit all eight reference-selected answers")
    # Independently permute each enclosure's member order. Content selects roles.
    facts = _analyze(base["input"])
    for reference_reverse in (False, True):
        for target_reverse in (False, True):
            pieces = []
            for index, g in enumerate(facts["groups"]):
                reverse = target_reverse if index == facts["target_group"] else reference_reverse
                pieces.append(_draw_group(g["panels"], g["color"], g["orientation"], reverse))
            h = sum(p.shape[0] for p in pieces)+1
            w = max(p.shape[1] for p in pieces)
            reordered = np.full((h, w), facts["background"], dtype=np.int8)
            top = 0
            for piece in pieces:
                reordered[top:top+piece.shape[0], :piece.shape[1]] = piece
                top += piece.shape[0]+1
            _require(np.array_equal(solve(reordered), base["output"]), "slot position changed semantics")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(100+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(9, 2, 1)
    damaged["test"][0]["output"][-1, -1] = (int(damaged["test"][0]["output"][-1, -1])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect label accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), shapes=len(shapes),
                target_profiles=len(profiles), palettes=len(palettes), layouts=len(layouts),
                corner_vertical_modes=len(modes), complete_edge_squares=edge_squares)
