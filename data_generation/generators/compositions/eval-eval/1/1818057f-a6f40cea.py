"""Hidden junction stencils: periodic rectangular contours make local pluses.

The input reader uses only pixels. Generation renders a complete contour layer
and computes its stamp before covering it; it does not obtain labels from solve.
"""
from __future__ import annotations

import itertools
import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=True)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _edge(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _bounds(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _halo(points):
    return {q for y, x in points for q in
            ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))}


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in _DIRECTIONS:
                q = y + dy, x + dx
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _frames(g, bg):
    result = []
    for color in np.unique(g):
        if color == bg:
            continue
        points = {tuple(map(int, p)) for p in np.argwhere(g == color)}
        for part in _components(points):
            box = _bounds(part)
            t, b, l, r = box
            if b - t < 2 or r - l < 2 or part != _edge(box):
                continue
            inside = g[t + 1:b, l + 1:r]
            if np.all(inside == inside[0, 0]) and inside[0, 0] != color:
                result.append((box, int(color), int(inside[0, 0])))
    return sorted(result, key=lambda f: (-len(_area(f[0])), f[0]))


def _complete_geometry(points, covered):
    """Pair up visible pieces; intersections are allowed only under the screen.

    Unlike disjoint-loop restoration, inferred perimeters may intersect, touch,
    share arms, or create new centers within the covered region.
    """
    parts = _components(points)
    assert 1 <= len(parts) <= 12, "missing or excessive contour fragments"
    choices = []
    for i in range(len(parts)):
        for j in range(i, len(parts)):
            visible = parts[i] | parts[j]
            box = _bounds(visible)
            t, b, l, r = box
            if b - t < 2 or r - l < 2:
                continue
            full = _edge(box)
            if full - covered == visible:
                choices.append((frozenset((i, j)), box, full))
    covers = []

    def search(unused, chosen):
        if len(covers) >= 2:
            return
        if not unused:
            covers.append(chosen)
            return
        first = min(unused)
        for ids, box, full in choices:
            if first in ids and ids <= unused:
                search(unused - ids, chosen + [(box, full)])

    search(set(range(len(parts))), [])
    assert len(covers) == 1, "rectangle completion must be unique"
    return covers[0]


def _complete_colors(g, visible, full):
    ordered = sorted(full)
    predictions, explanations = set(), []
    for sign in (1, -1):
        for period in range(1, 7):
            residues, counts = {}, [0] * period
            for y, x in sorted(visible):
                index, color = (y + sign * x) % period, int(g[y, x])
                if index in residues and residues[index] != color:
                    break
                residues[index] = color
                counts[index] += 1
            else:
                if min(counts) >= 2:
                    predictions.add(tuple(residues[(y + sign * x) % period]
                                          for y, x in ordered))
                    explanations.append((period, sign))
    assert len(predictions) == 1, "visible repetition must determine all hidden colors"
    return dict(zip(ordered, next(iter(predictions)))), explanations


def _read_scene(g):
    bg = int(g[0, 0])
    assert all(int(g[p]) == bg for p in
               ((0, -1), (-1, 0), (-1, -1))), "blank canvas corners required"
    frames = _frames(g, bg)
    assert frames, "no opaque screen"
    main, frame, fill = frames[0]
    assert len(frames) == 1 or len(_area(main)) > len(_area(frames[1][0])), "largest-frame tie"
    screens = [box for box, color, _ in frames if color == frame]
    covered = set().union(*(_area(box) for box in screens))
    assert sum(len(_area(box)) for box in screens) == len(covered), "screens overlap"
    visible = {tuple(map(int, p)) for p in np.argwhere(g != bg)} - covered
    ink_colors = {int(g[p]) for p in visible}
    assert ink_colors and not ink_colors & {bg, frame, fill}, "ink and screen roles must differ"
    assert len(ink_colors) <= 3
    outlines = _complete_geometry(visible, covered)
    full = set().union(*(cells for _, cells in outlines))
    for box in screens:
        if box != main:
            assert not full & _area(box), "small screens may not erase contours"
    # Explicitly check exterior segmentation, rather than trusting scene metadata.
    for (_, one), (_, two) in itertools.combinations(outlines, 2):
        assert not (one - covered) & _halo(two - covered), "exterior contours touch"
    colors, fits = _complete_colors(g, visible, full)
    layer = np.full(g.shape, bg, dtype=np.int8)
    for p, color in colors.items():
        layer[p] = color
    return dict(background=bg, frame=frame, fill=fill, main=main,
                screens=screens, outlines=outlines, layer=layer, fits=fits)


def _plus_union(layer, bg, monochrome=True):
    """Detect simultaneously before cropping, including centers on the frame."""
    centers = np.zeros(layer.shape, dtype=bool)
    a = layer[1:-1, 1:-1]
    if monochrome:
        centers[1:-1, 1:-1] = ((a != bg) & (a == layer[:-2, 1:-1])
            & (a == layer[2:, 1:-1]) & (a == layer[1:-1, :-2])
            & (a == layer[1:-1, 2:]))
    else:
        centers[1:-1, 1:-1] = ((a != bg) & (layer[:-2, 1:-1] != bg)
            & (layer[2:, 1:-1] != bg) & (layer[1:-1, :-2] != bg)
            & (layer[1:-1, 2:] != bg))
    union = centers.copy()
    union[1:] |= centers[:-1]
    union[:-1] |= centers[1:]
    union[:, 1:] |= centers[:, :-1]
    union[:, :-1] |= centers[:, 1:]
    return centers, union


def _stamp(layer, bg, frame, fill, box):
    t, b, l, r = box
    _, union = _plus_union(layer, bg)
    answer = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
    answer[union[t + 1:b, l + 1:r]] = frame
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _read_scene(_grid(input_grid))
    return _stamp(scene["layer"], scene["background"], scene["frame"],
                  scene["fill"], scene["main"])


def _sample_indices(rng, low, high, count, gap=1):
    # Sample separated coordinates without unbounded rejection.
    pool = np.arange(low, high + 1 - (gap - 1) * (count - 1))
    values = sorted(map(int, rng.choice(pool, count, replace=False)))
    return [v + (gap - 1) * i for i, v in enumerate(values)]


def _geometry(rng, layout, count, witness=False):
    first = 2 if witness else int(rng.integers(max(1, count - 3), min(3, count - 1) + 1))
    second = count - first
    margin = 2 * max(first, second) + 1
    # The two occupied exterior sides need not have equal free margins.
    max_inside = min(17, 28 - 2 * margin)
    min_inside = max(7, 4 * max(first, second) - 3) if layout == "span" else 7
    oh, ow = map(int, rng.integers(min_inside, max_inside + 1, size=2))
    h = int(rng.integers(oh + 2 + 2 * margin, 31))
    w = int(rng.integers(ow + 2 + 2 * margin, 31))
    t = int(rng.integers(margin, h - oh - 1 - margin))
    l = int(rng.integers(margin, w - ow - 1 - margin))
    b, r = t + oh + 1, l + ow + 1
    boxes = []
    if layout == "corner":
        # Opposite corner approaches allow intersections entirely under the mask.
        if witness:
            y = int(rng.integers(t + 4, b - 1))
            ends_y = [y + 2, y]
        else:
            ends_y = _sample_indices(rng, t + max(4, 2 * second), b, first, 2)[::-1]
        ends_x = _sample_indices(rng, l + max(4, 2 * second), r, first, 2)[::-1]
        starts_y = _sample_indices(rng, t, min(ends_y) - 2, second, 2)
        starts_x = _sample_indices(rng, l, min(ends_x) - 2, second, 2)
        if witness:
            starts_y[0] = t  # A plus center on the frame contributes an inside arm.
        for i in range(first):
            boxes.append((t - 2 * (first - i), ends_y[i],
                          l - 2 * (first - i), ends_x[i]))
        for i in range(second):
            boxes.append((starts_y[i], b + 2 * (second - i),
                          starts_x[i], r + 2 * (second - i)))
    else:
        # Horizontal loops and vertical loops span the mask and cross inside it.
        top_rows = _sample_indices(rng, t, t + (oh - 1) // 2, first, 2)
        bottom_rows = _sample_indices(rng, t + (oh + 3) // 2, b, first, 2)[::-1]
        left_cols = _sample_indices(rng, l, l + (ow - 1) // 2, second, 2)
        right_cols = _sample_indices(rng, l + (ow + 3) // 2, r, second, 2)[::-1]
        for i in range(first):
            boxes.append((top_rows[i], bottom_rows[i],
                          l - 2 * (first - i), r + 2 * (first - i)))
        for i in range(second):
            boxes.append((t - 2 * (second - i), b + 2 * (second - i),
                          left_cols[i], right_cols[i]))
    return (h, w), (t, b, l, r), boxes


def _orient(g, turns, flip):
    a = np.rot90(g, turns)
    return np.fliplr(a).copy() if flip else a.copy()


def _direct_union(layer, bg, destructive=False):
    """Bounded validation scanner; also a deliberately wrong sequential rule."""
    source = layer.copy()
    cells, centers = set(), set()
    for y in range(1, source.shape[0] - 1):
        for x in range(1, source.shape[1] - 1):
            points = [(y, x)] + [(y + dy, x + dx) for dy, dx in _DIRECTIONS]
            color = int(source[y, x])
            if color != bg and all(source[p] == color for p in points):
                centers.add((y, x))
                cells.update(points)
                if destructive:
                    for p in points:
                        source[p] = bg
    mask = np.zeros(layer.shape, dtype=bool)
    for p in cells:
        mask[p] = True
    return centers, mask


def _features(g):
    s = _read_scene(g)
    t, b, l, r = s["main"]
    crop = np.s_[t + 1:b, l + 1:r]
    centers, union = _direct_union(s["layer"], s["background"])
    _, destructive = _direct_union(s["layer"], s["background"], True)
    _, geometric = _plus_union(s["layer"], s["background"], False)
    _, cropped = _plus_union(s["layer"][crop], s["background"])
    return dict(scene=s, crop=crop, centers=centers, union=union,
                count=len(s["outlines"]), period=min(p for p, _ in s["fits"]),
                selective=bool(np.any((s["layer"][crop] != s["background"]) & ~union[crop])),
                overlap=not np.array_equal(union[crop], destructive[crop]),
                phase_rejection=bool(np.any(geometric[crop] & ~union[crop])),
                boundary=not np.array_equal(cropped, union[crop]))


def _make_pair(rng, plan, period, roles, seen):
    bg, frame, other_fill, a, b, c = roles
    for _ in range(180):
        shape, main, boxes = _geometry(rng, plan["layout"], plan["count"], plan["witness"])
        full = set().union(*(_edge(box) for box in boxes))
        fill = bg if plan["same_fill"] else other_fill
        if plan["pattern"]:
            word = [a] * 4 + ([b] if period == 5 else [b, int(rng.choice((b, c)))])
            phase, sign = int(rng.integers(period)), int(rng.choice((-1, 1)))
        else:
            word, phase, sign = [a], 0, 1
        layer = np.full(shape, bg, dtype=np.int8)
        for y, x in full:
            layer[y, x] = word[(y + sign * x + phase) % len(word)]
        answer = _stamp(layer, bg, frame, fill, main)
        grid = layer.copy()
        screens = [main]
        occupied = full | _area(main)
        # Enumerate clear 3x3/3x4 screens, rather than hiding unobservable objects.
        for _ in range(plan["extras"]):
            choices = [(y, y + 2, x, x + ww - 1)
                       for ww in (3, 4) for y in range(1, shape[0] - 3)
                       for x in range(1, shape[1] - ww)]
            rng.shuffle(choices)
            for box in choices:
                if not _area(box) & _halo(occupied):
                    screens.append(box)
                    occupied.update(_area(box))
                    break
            else:
                break
        if len(screens) != 1 + plan["extras"]:
            continue
        for box in screens:
            y0, y1, x0, x1 = box
            grid[y0 + 1:y1, x0 + 1:x1] = fill
            for p in _edge(box):
                grid[p] = frame
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        grid, answer = _orient(grid, turns, flip), _orient(answer, turns, flip)
        try:
            f = _features(grid)
            assert f["count"] == len(boxes)
            assert np.array_equal(answer, solve(grid))
            assert np.any(answer == frame) and np.any(answer == fill)
            assert f["selective"]
            if plan["witness"]:
                assert f["overlap"] and f["boundary"]
            if plan["pattern"]:
                assert f["period"] == period and f["phase_rejection"]
        except AssertionError:
            continue
        if _key(grid) not in seen:
            return {"input": grid, "output": answer}
    raise RuntimeError("bounded scene sampling failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    num_train, num_test = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if seed < 0 or not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("nonnegative seed and legal train/test counts required")
    rng = np.random.default_rng(seed)
    period = int(rng.integers(5, 7))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = split == "train" and i == 0
            periodic_demo = split == "train" and i == 1
            loops = 3 if witness else (2 if periodic_demo else
                    int(rng.integers(2, 5) if split == "train" else rng.integers(4, 6)))
            plan = dict(layout="corner" if witness else ("span" if periodic_demo else
                        str(rng.choice(("corner", "span")))), count=loops,
                        witness=witness, pattern=periodic_demo or (not witness and rng.random() < .75),
                        same_fill=witness or (not periodic_demo and rng.random() < .4),
                        extras=1 if periodic_demo else int(rng.integers(0, 3)))
            # Distinct pair-level roles; all colors can serve each role across seeds.
            roles = list(map(int, rng.choice(10, 6, replace=False)))
            pair = _make_pair(rng, plan, period, roles, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            f = _features(inp)
            s, crop = f["scene"], f["crop"]
            assert out.shape == f["union"][crop].shape
            expected = np.full(out.shape, s["fill"], dtype=np.int8)
            expected[f["union"][crop]] = s["frame"]
            assert np.array_equal(out, expected), "incorrect stencil"
            assert np.array_equal(out, solve(inp.copy()))
            assert np.any(out == s["frame"]) and np.any(out == s["fill"])
            # An input-only ablation of B finds no hidden pluses in uniform fill.
            _, visible_union = _plus_union(inp, s["background"])
            assert not np.array_equal(visible_union[crop], f["union"][crop])
            assert f["selective"], "a restored-contours shortcut must differ"
            features[split].append(f)
    train = features["train"]
    assert any(f["period"] == 1 and f["overlap"] and f["boundary"] for f in train)
    assert any(f["period"] > 1 and f["phase_rejection"] and len(f["scene"]["screens"]) > 1 for f in train)
    assert {f["scene"]["fill"] == f["scene"]["background"] for f in train} == {True, False}
    assert max(f["count"] for f in train) >= 3
    for f in features["test"]:
        assert 4 <= f["count"] <= 5
        assert f["period"] == 1 or any(t["period"] == f["period"] for t in train)


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    keys, shapes, output_shapes, structures, periods, ink_counts, counts = set(), set(), set(), set(), set(), set(), set()
    flags = {name: 0 for name in ("overlap", "boundary", "phase_rejection")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        keys.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for pair in puzzle["train"] + puzzle["test"]:
            g, out = pair["input"], pair["output"]
            f = _features(g)
            shapes.add(g.shape)
            output_shapes.add(out.shape)
            mapping = {}
            structures.add((g.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in g.flat)))
            periods.add(f["period"])
            ink_counts.add(len(set(map(int, f["scene"]["layer"].flat))) - 1)
            counts.add(f["count"])
            for name in flags:
                flags[name] += bool(f[name])
    assert len(keys) == 42 and len(shapes) >= 30 and len(output_shapes) >= 25
    assert len(structures) >= 180 and {1, 5, 6} <= periods
    assert {1, 2, 3} <= ink_counts and {2, 3, 4, 5} <= counts
    assert all(v >= 42 for v in flags.values())
    assert any(30 in shape for shape in shapes)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(100 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    q = generate_puzzle(17)
    assert tuple(_key(p["input"]) for p in q["train"] + q["test"]) == tuple(
        _key(p["input"]) for p in puzzles[17]["train"] + puzzles[17]["test"])
    # Small scanner fixtures verify overlaps, exclusion of the background,
    # color continuity, and clipping without involving rectangle sampling.
    layer = np.zeros((5, 6), dtype=np.int8)
    layer[1:4, 1:5] = 2
    centers, union = _plus_union(layer, 0)
    assert centers.sum() == 2 and union.sum() == 8
    assert _direct_union(layer, 0, True)[1].sum() == 5
    assert np.array_equal(union, _direct_union(layer, 0)[1])
    layer[2, 2] = 3
    assert not _plus_union(layer, 0)[1].any()
    assert _plus_union(layer, 0, False)[1].any()
    assert not _plus_union(np.zeros((3, 3), dtype=int), 0)[1].any()
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in q[s]] for s in q}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "output_shapes": len(output_shapes), "structures": len(structures),
            "periods": sorted(periods), "contour_counts": sorted(counts), **flags}
