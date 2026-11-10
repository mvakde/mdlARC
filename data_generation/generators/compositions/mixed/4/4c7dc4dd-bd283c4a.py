"""Framed inventory analogy, with frequency-ranked bottom-up column packing.

No files, seeds, training examples or generator state are consulted by solve.
All drawing happens after the panel inventories and their answers are built.
"""
from collections import Counter
from itertools import combinations

import numpy as np


HOST_SEEDS = (0, 453952628, 222310592, 748503916, 779083336, 733098103,
              137079055, 835387927, 955442709, 255067239, 26137113, 548370635,
              715749534, 129082798, 289145518, 758179334, 81742782, 329901213,
              713406300, 372860851, 917168465, 805016360, 3480444, 71940096,
              159112265, 263243381, 417677038, 650441205, 342677912, 596112413,
              561787410, 271599444, 702286713, 646632785, 590729951, 80876733,
              692953910, 966686307, 296459464, 487269570, 855232171, 488284437)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _hist(a):
    return Counter(map(int, a.flat))


def _rank(a):
    counts = _hist(a)
    if len(counts) < 3 or len(set(counts.values())) != len(counts):
        raise ValueError("Inventories need at least three colors and no frequency ties")
    return sorted(counts, key=lambda c: -counts[c])


def _raster(a, mode=0, order=None):
    """Eight traversals; mode 0 is columns left-to-right, cells bottom-to-top."""
    counts = _hist(a)
    colors = _rank(a) if order is None else order
    stream = np.array([c for c in colors for _ in range(counts[c])], dtype=np.int8)
    h, w = a.shape
    out = stream.reshape((w, h)).T if mode < 4 else stream.reshape((h, w))
    if mode % 4 in (0, 2):
        out = out[::-1, :]
    if mode % 4 in (2, 3):
        out = out[:, ::-1]
    return out.copy()


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _inside(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return y < r and x < c and r + h < y + bh and c + w < x + bw


def _disjoint(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return r + h <= y or y + bh <= r or c + w <= x or x + bw <= c


def _interior(a, box):
    r, c, h, w = box[:4]
    return a[r + 1:r + h - 1, c + 1:c + w - 1].copy()


def _rings(a):
    """Find isolated single-cell rectangular outline components in O(grid area)."""
    h, w = a.shape
    visited = np.zeros(a.shape, dtype=bool)
    rings = []
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            color = int(a[r, c])
            todo, cells = [(r, c)], []
            visited[r, c] = True
            while todo:
                y, x = todo.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= yy < h and 0 <= xx < w and not visited[yy, xx]
                            and a[yy, xx] == color):
                        visited[yy, xx] = True
                        todo.append((yy, xx))
            top, bottom = min(y for y, _ in cells), max(y for y, _ in cells)
            left, right = min(x for _, x in cells), max(x for _, x in cells)
            bh, bw = bottom - top + 1, right - left + 1
            if (bh >= 6 and bw >= 6 and len(cells) == 2 * (bh + bw) - 4
                    and all(y in (top, bottom) or x in (left, right) for y, x in cells)):
                box = (top, left, bh, bw, color)
                if not np.any(_interior(a, box) == color):
                    rings.append(box)
    return sorted(rings)


def _scene(a):
    rings = _rings(a)
    scenes = {}
    for color in sorted({b[4] for b in rings}):
        panels = [b for b in rings if b[4] == color and 6 <= b[2] <= 10 and 6 <= b[3] <= 10]
        if len(panels) != 4 or any(not _disjoint(x, y) for x, y in combinations(panels, 2)):
            continue
        groups = []
        for box in rings:
            if box[4] == color:
                continue
            members = tuple(i for i, panel in enumerate(panels) if _inside(panel, box))
            if len(members) == 2 and panels[members[0]][2:4] == panels[members[1]][2:4]:
                groups.append((box, members))
        for (box0, ids0), (box1, ids1) in combinations(groups, 2):
            if set(ids0) & set(ids1) or not _disjoint(box0, box1):
                continue
            boxes, ids = [box0, box1], [ids0, ids1]
            contents = [[_interior(a, panels[i]) for i in group] for group in ids]
            blank = [(g, s) for g in range(2) for s in range(2)
                     if np.all(contents[g][s] == contents[g][s][0, 0])]
            if len(blank) != 1:
                continue
            g, slot = blank[0]
            neutral = int(contents[g][slot][0, 0])
            structural = {color, box0[4], box1[4], neutral}
            if len(structural) != 4:
                continue
            occupied = contents[g][1 - slot]
            reference = contents[1 - g]
            if any(set(map(int, np.unique(p))) & structural for p in reference + [occupied]):
                continue
            association = tuple(sorted((ids0, ids1)))
            scenes[(tuple(panels), association)] = {
                "panels": panels, "boxes": boxes, "ids": ids,
                "contents": contents, "reference": reference, "target": occupied,
                "neutral": neutral, "target_group": g, "blank_slot": slot,
                "target_box": panels[ids[g][1 - slot]],
                "blank_box": panels[ids[g][slot]],
                "reference_boxes": [panels[i] for i in ids[1 - g]],
            }
    if len(scenes) != 1:
        raise ValueError(f"Expected one four-panel association, found {len(scenes)}")
    return next(iter(scenes.values()))


def _reference_mode(reference):
    """Infer traversal from pixels, considering both unordered reference members.

    The hypothesis set is broader than the generated family. Generation and
    validation require mode 0, but inference does not assume its answer. This
    makes the data flow genuinely reference-driven without claiming that a
    fixed-B family prevents a knowledgeable target-only shortcut.
    """
    first, second = reference
    if first.shape != second.shape or _hist(first) != _hist(second):
        raise ValueError("Reference pair must preserve its inventory and dimensions")
    matches = []
    for source, result in ((first, second), (second, first)):
        for mode in range(8):
            if not np.array_equal(source, result) and np.array_equal(_raster(source, mode), result):
                matches.append(mode)
    if len(matches) != 1:
        raise ValueError("Reference must uniquely demonstrate a frequency-ranked raster traversal")
    return matches[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read groups, infer the demonstrated raster, repack only the target inventory."""
    scene = _scene(_grid(input_grid))
    mode = _reference_mode(scene["reference"])
    return _raster(scene["target"], mode)


def _counts(rng, shape, k, profile, reveal=False):
    n = int(np.prod(shape))
    for _ in range(512):
        if profile == "tail":
            weights = rng.dirichlet(np.full(k - 1, 2.0))
            counts = np.r_[rng.multinomial(n - k, weights) + 1, 1]
        else:
            alpha = {"balanced": 12.0, "ordinary": 2.0, "skew": .6}[profile]
            weights = rng.dirichlet(np.full(k, alpha))
            counts = rng.multinomial(n - k, weights) + 1
        counts = np.sort(counts)[::-1]
        if len(set(map(int, counts))) != k:
            continue
        probe = np.repeat(np.arange(k), counts).reshape(shape)
        if reveal and len({_key(_raster(probe, m)) for m in range(8)}) != 8:
            continue
        return counts
    raise RuntimeError("Bounded frequency sampling exhausted")


def _scatter(rng, colors, counts, shape, style):
    n = int(np.prod(shape))
    stream = np.repeat(colors, counts).astype(np.int8)
    h, w = shape
    rr, cc = np.indices(shape)
    for _ in range(16):
        if style == "near":
            a = _raster(stream.reshape(shape)).ravel()
            for _ in range(max(3, n // 6)):
                x, y = rng.choice(n, 2, replace=False)
                a[x], a[y] = a[y], a[x]
        else:
            if style == "scatter":
                keys = rng.random(shape)
            elif style == "stripe":
                keys = ((rr + int(rng.integers(1, 4)) * cc) % int(rng.integers(2, 5))
                        + rng.random(shape) * .7)
            elif style == "patch":
                field = rng.random((2, 3))
                keys = field[rr * 2 // h, cc * 3 // w] + rng.random(shape) * .15
            elif style == "cluster":
                keys = ((rr - rng.uniform(0, h)) / h) ** 2 + ((cc - rng.uniform(0, w)) / w) ** 2
                keys += rng.random(shape) * .12
            else:
                raise ValueError("Unknown scattering style")
            a = np.empty(n, dtype=np.int8)
            a[np.argsort(keys.ravel(), kind="stable")] = stream
        # A position-based color ordering must visibly disagree with frequency rank.
        if a[0] == colors[0]:
            pos = int(rng.choice(np.flatnonzero(a != colors[0])))
            a[0], a[pos] = a[pos], a[0]
        a = a.reshape(shape)
        if all(not np.array_equal(a, _raster(a, m)) for m in range(8)):
            return a
    raise RuntimeError("Bounded scattering exhausted")


def _layout(rng, shapes, axes):
    sizes = [(h + 2, w + 2) for h, w in shapes]
    minima = [(h + 2, 2 * w + 3) if axis == 0 else (2 * h + 3, w + 2)
              for (h, w), axis in zip(sizes, axes)]
    feasible = [a for a in range(2) if sum(m[a] for m in minima) + 1 <= 30
                and max(m[1 - a] for m in minima) <= 30]
    if not feasible:
        raise ValueError("Panel arrangement does not fit")
    direction = int(rng.choice(feasible))
    along = int(rng.integers(sum(m[direction] for m in minima) + 1, 31))
    across = int(rng.integers(max(m[1 - direction] for m in minima), 31))
    slack = along - sum(m[direction] for m in minima) - 1
    growth = rng.multinomial(slack, np.full(5, .2))
    starts = [int(growth[2]), int(growth[2] + minima[0][direction] + growth[0] + 1 + growth[3])]
    boxes, panels = [], []
    for i, ((ph, pw), axis) in enumerate(zip(sizes, axes)):
        dims = [0, 0]
        dims[direction] = minima[i][direction] + int(growth[i])
        dims[1 - direction] = int(rng.integers(minima[i][1 - direction], across + 1))
        origin = [0, 0]
        origin[direction] = starts[i]
        origin[1 - direction] = int(rng.integers(across - dims[1 - direction] + 1))
        r, c = origin
        gh, gw = dims
        boxes.append((r, c, gh, gw))
        span, size = (gw, pw) if axis == 0 else (gh, ph)
        gap = int(rng.integers(1, span - 2 * size - 1))
        start = int(rng.integers(1, span - 2 * size - gap))
        positions = []
        for offset in (start, start + size + gap):
            y = r + int(rng.integers(1, gh - ph)) if axis == 0 else r + offset
            x = c + offset if axis == 0 else c + int(rng.integers(1, gw - pw))
            positions.append((y, x, ph, pw))
        panels.append(positions)
    canvas = (along, across) if direction == 0 else (across, along)
    return canvas, boxes, panels


def _outline(a, box, color):
    r, c, h, w = box[:4]
    a[r, c:c + w] = a[r + h - 1, c:c + w] = color
    a[r:r + h, c] = a[r:r + h, c + w - 1] = color


def _put(a, box, interior):
    r, c, h, w = box[:4]
    a[r + 1:r + h - 1, c + 1:c + w - 1] = interior


def _make_pair(rng, target_shape, target_colors, target_counts, style, axes,
               reference_group, ref_order, target_slot):
    # Structural colors are disjoint from all payload colors, but their identities vary.
    other = np.array([c for c in range(10) if c not in target_colors])
    neutral, frame, group0, group1 = map(int, rng.choice(other, 4, replace=False))
    available = np.array([c for c in range(10) if c not in (neutral, frame, group0, group1)])
    reference_shape = tuple(map(int, rng.integers(4, 7, size=2)))
    k = int(rng.integers(3, 6))
    ref_colors = rng.choice(available, k, replace=False)
    profiles = ("ordinary", "balanced", "skew", "tail")
    counts = _counts(rng, reference_shape, k, str(rng.choice(profiles)), reveal=True)
    reference_source = _scatter(rng, ref_colors, counts, reference_shape, str(rng.choice(
        ("scatter", "stripe", "cluster", "patch", "near"))))
    reference_result = _raster(reference_source)
    target = _scatter(rng, target_colors, target_counts, target_shape, style)
    answer = _raster(target)
    if _key(answer) == _key(reference_result):
        raise ValueError("Copying the reference answer would be uninformative")
    shapes = [None, None]
    shapes[reference_group], shapes[1 - reference_group] = reference_shape, target_shape
    canvas, boxes, panels = _layout(rng, shapes, axes)
    # Camouflage reuses payload/neutral colors; structural outlines stay isolated.
    wallpaper = rng.choice(np.r_[available, neutral], int(rng.integers(2, 5)), replace=False)
    rr, cc = np.indices(canvas)
    mode = int(rng.integers(3))
    if mode == 0:
        indices = (rr + cc * int(rng.integers(1, 4))) % len(wallpaper)
    elif mode == 1:
        indices = (rr // 2 + cc // 2) % len(wallpaper)
    else:
        tile = rng.integers(len(wallpaper), size=(3, 4))
        indices = tile[rr % 3, cc % 4]
    grid = wallpaper[indices].astype(np.int8)
    reference = [reference_source, reference_result]
    if ref_order:
        reference.reverse()
    target_pair = [np.full(target_shape, neutral, dtype=np.int8) for _ in range(2)]
    target_pair[target_slot] = target
    contents = [None, None]
    contents[reference_group], contents[1 - reference_group] = reference, target_pair
    for i in range(2):
        _outline(grid, boxes[i], (group0, group1)[i])
        for box, data in zip(panels[i], contents[i]):
            _outline(grid, box, frame)
            _put(grid, box, data)
    # Reading the pixels is separate from logical answer construction.
    if not np.array_equal(solve(grid), answer):
        raise ValueError("Scene did not read back correctly")
    return {"input": grid, "output": answer}


def _checked_count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _checked_count(num_train, 2, 10, "num_train")
    ne = _checked_count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shape0 = tuple(map(int, rng.integers(4, 7, size=2)))
    colors0 = rng.choice(10, int(rng.integers(3, 5)), replace=False)
    if list(colors0) == sorted(colors0):
        colors0 = np.roll(colors0, 1)
    counts0 = _counts(rng, shape0, len(colors0), "ordinary", reveal=True)
    counts1 = counts0
    for _ in range(128):
        counts1 = _counts(rng, shape0, len(colors0), "balanced", reveal=True)
        if not np.array_equal(counts1, counts0):
            break
    else:
        raise RuntimeError("Could not build a changed-inventory contrast")
    base_group, base_slot, base_order = map(int, rng.integers(2, size=3))
    styles = ("scatter", "stripe", "patch", "cluster", "near")
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt + ne):
        testing = i >= nt
        if i < 3:
            shape = shape0
            colors = np.roll(colors0, 1) if i == 1 else colors0
            counts = counts1 if i == 1 else counts0
            style = ("scatter", "cluster", "stripe")[i]
        else:
            shape = tuple(map(int, rng.integers(4, 7, size=2)))
            colors = rng.choice(10, int(rng.integers(3, 6)), replace=False)
            counts = _counts(rng, shape, len(colors), str(rng.choice(("ordinary", "skew", "tail"))))
            style = str(rng.choice(styles))
        if testing:
            # Larger target, independent reference scale, and a familiar-layout recombination.
            shape = (int(rng.integers(7, 9)), int(rng.integers(5, 9)))
            if rng.integers(2):
                shape = shape[::-1]
            colors = rng.choice(10, int(rng.integers(4, 7)), replace=False)
            counts = _counts(rng, shape, len(colors), str(rng.choice(("balanced", "ordinary", "skew", "tail"))))
            style = str(rng.choice(styles))
        if list(colors) == sorted(colors):
            colors = np.roll(colors, 1)
        axes = [i % 2, i % 2] if i < 2 else list(map(int, rng.integers(2, size=2)))
        if testing:
            axes = [0, 1] if rng.integers(2) else [1, 0]
        reference_group = base_group ^ (i % 2) if i < 2 else int(rng.integers(2))
        target_slot = base_slot ^ (i % 2) if i < 2 else int(rng.integers(2))
        ref_order = base_order ^ (i % 2) if i < 2 else int(rng.integers(2))
        for _ in range(32):
            try:
                pair = _make_pair(rng, shape, colors, counts, style, axes,
                                  reference_group, ref_order, target_slot)
            except ValueError:
                continue
            if i == 2 and not testing:
                first_target = _scene(puzzle["train"][0]["input"])["target"]
                if np.array_equal(_scene(pair["input"])["target"], first_target):
                    continue
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("Bounded scene construction exhausted")
        seen.add(_key(pair["input"]))
        puzzle["test" if testing else "train"].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("Expected exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("Splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _checked_count(nt, 2, 10, "num_train")
    _checked_count(ne, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    seen, scenes = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("Expected exactly input and output")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            if _key(a) in seen:
                raise ValueError("Duplicate input")
            seen.add(_key(a))
            scene = _scene(a)
            mode = _reference_mode(scene["reference"])
            if mode != 0:
                raise ValueError("The saved proposal requires bottom-up columns left-to-right")
            target = scene["target"]
            if (target.shape != b.shape or _hist(target) != _hist(b)
                    or not np.array_equal(_raster(target, mode), b)):
                raise ValueError("Answer must repack the target's exact inventory")
            if np.array_equal(target, b):
                raise ValueError("Target should need rearranging")
            if np.array_equal(_raster(target, order=list(dict.fromkeys(map(int, target.flat)))), b):
                raise ValueError("First-occurrence shortcut was not distinguished")
            if np.array_equal(_raster(target, order=sorted(_hist(target))), b):
                raise ValueError("Literal color-order shortcut was not distinguished")
            if _key(b) in {_key(x) for x in scene["reference"]}:
                raise ValueError("Copying a reference panel was not distinguished")
            scenes[split].append(scene)
    first, second = [s["target"] for s in scenes["train"][:2]]
    if (first.shape != second.shape or set(_hist(first)) != set(_hist(second))
            or _hist(first) == _hist(second) or _rank(first)[0] == _rank(second)[0]):
        raise ValueError("First two demonstrations need a changed-count/rank contrast")
    if nt >= 3:
        third = scenes["train"][2]["target"]
        if first.shape != third.shape or _hist(first) != _hist(third) or np.array_equal(first, third):
            raise ValueError("Third demonstration must contrast arrangement at fixed inventory")
    if any(max(s["target"].shape) <= 6 for s in scenes["test"]):
        raise ValueError("Tests should extend target size beyond training")
    return True


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in a)


def _direct_expected(target):
    """Simple loop implementation used only by checks, not by construction or solve."""
    counts = Counter(int(c) for row in target for c in row)
    stream = [c for c in sorted(counts, key=lambda c: -counts[c]) for _ in range(counts[c])]
    result = np.empty_like(target)
    p = 0
    for col in range(target.shape[1]):
        for row in range(target.shape[0] - 1, -1, -1):
            result[row, col] = stream[p]
            p += 1
    return result


def self_test():
    """42 puzzles, streaming diagnostics, small fixed metamorphic checks; no files."""
    shapes, canvases, ranks, layouts, palettes, fingerprints = set(), set(), set(), set(), set(), set()
    remainders, profiles = set(), set()
    near_rank = tails = aligned = partial = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                scene = _scene(pair["input"])
                target, answer = scene["target"], pair["output"]
                assert np.array_equal(_direct_expected(target), answer)
                shapes.add(target.shape)
                canvases.add(pair["input"].shape)
                ranks.add(tuple(_rank(target)))
                palettes.add(len(_hist(target)))
                boxes = [[scene["panels"][i] for i in ids] for ids in scene["ids"]]
                orientations = tuple(int(b[0][0] + b[0][2] <= b[1][0]
                                         or b[1][0] + b[1][2] <= b[0][0]) for b in boxes)
                layouts.add((orientations, scene["target_group"], scene["blank_slot"]))
                counts = sorted(_hist(target).values(), reverse=True)
                profiles.add((target.shape, tuple(counts)))
                boundaries = np.cumsum(counts)[:-1] % target.shape[0]
                remainders.update(map(int, boundaries))
                aligned += int(np.any(boundaries == 0))
                partial += int(np.any(boundaries != 0))
                near_rank += any(x - y == 1 for x, y in zip(counts, counts[1:]))
                tails += counts[-1] == 1
        # Both member orders and blank positions must be semantically irrelevant.
        pair = puzzle["train"][0]
        a, answer = pair["input"], pair["output"]
        scene = _scene(a)
        twin = a.copy()
        for box, data in zip(scene["reference_boxes"], scene["reference"][::-1]):
            _put(twin, box, data)
        assert np.array_equal(solve(twin), answer)
        twin = a.copy()
        _put(twin, scene["blank_box"], scene["target"])
        _put(twin, scene["target_box"], np.full(scene["target"].shape, scene["neutral"]))
        assert np.array_equal(solve(twin), answer)
        # Reference-dependent algorithm diagnostic: a different demonstrated
        # traversal changes the prediction for the same target. These twins
        # deliberately lie OUTSIDE the proposal's fixed-mode generated family;
        # they do not establish semantic necessity of A within that family.
        source = next(x for x in scene["reference"]
                      if not np.array_equal(x, _raster(x)))
        twin = a.copy()
        for box, data in zip(scene["reference_boxes"], (source, _raster(source, 1))):
            _put(twin, box, data)
        assert np.array_equal(solve(twin), _raster(scene["target"], 1))
        assert not np.array_equal(solve(twin), answer)
        # Every color is a local role, including zero; palette relabeling must
        # commute with the transformation.
        palette = np.random.default_rng(seed + 1).permutation(10).astype(np.int8)
        assert np.array_equal(solve(palette[a]), palette[answer])
        # Changing arrangement at fixed inventory cannot affect the answer.
        twin = a.copy()
        _put(twin, scene["target_box"], np.roll(scene["target"].ravel(), 1).reshape(scene["target"].shape))
        assert np.array_equal(solve(twin), answer)
        # Changing only the target's color frequencies/ranks must affect the answer.
        colors = _rank(scene["target"])
        changed = scene["target"].copy()
        changed[scene["target"] == colors[0]] = colors[1]
        changed[scene["target"] == colors[1]] = colors[0]
        twin = a.copy()
        _put(twin, scene["target_box"], changed)
        assert not np.array_equal(solve(twin), answer)
        # A false complete reference is rejected, rather than silently ignored.
        twin = a.copy()
        _put(twin, scene["reference_boxes"][1], scene["reference"][0])
        try:
            solve(twin)
        except ValueError:
            pass
        else:
            raise AssertionError("An invalid reference was ignored")
    assert len(fingerprints) == 42 and len(shapes) >= 15 and len(canvases) >= 30
    assert palettes == {3, 4, 5, 6} and len(ranks) >= 75 and len(profiles) >= 100
    assert len(layouts) >= 10 and len(remainders) >= 7
    assert min(near_rank, tails, aligned, partial) >= 5
    assert _same(generate_puzzle(0), generate_puzzle(0))
    # Full explicit count contract, including the short two-demonstration case.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(42 + nt * 5 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Validator must notice an inventory-preserving but spatially wrong output.
    broken = generate_puzzle(0, 2, 1)
    out = broken["test"][0]["output"]
    broken["test"][0]["output"] = out[::-1].copy()
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong output accepted")
    # The inference hypothesis set must not broaden the saved generated rule.
    broken = generate_puzzle(0, 2, 1)
    pair = broken["train"][0]
    scene = _scene(pair["input"])
    source = next(x for x in scene["reference"] if not np.array_equal(x, _raster(x)))
    for box, data in zip(scene["reference_boxes"], (source, _raster(source, 1))):
        _put(pair["input"], box, data)
    pair["output"] = _raster(scene["target"], 1)
    assert np.array_equal(solve(pair["input"]), pair["output"])
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("A different demonstrated traversal entered the fixed-mode family")
    return {"seeds_checked": len(HOST_SEEDS), "pairs_checked": len(HOST_SEEDS) * 5,
            "target_shapes": len(shapes), "canvas_shapes": len(canvases),
            "palette_sizes": sorted(palettes), "frequency_rank_orders": len(ranks),
            "layout_configurations": len(layouts), "frequency_profiles": len(profiles),
            "boundary_remainders": sorted(remainders), "near_rank_pairs": int(near_rank),
            "singleton_tail_pairs": int(tails), "aligned_boundary_pairs": int(aligned),
            "partial_boundary_pairs": int(partial), "explicit_count_combinations": 30,
            "reference_counterfactual_scope": "outside fixed-mode generated family",
            "independent_semantic_verification": False}


if __name__ == "__main__":
    self_test()
    print("self_test passed")
