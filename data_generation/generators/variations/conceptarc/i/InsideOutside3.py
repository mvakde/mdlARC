import math
import numpy as np


_SOURCE_INPUTS = frozenset((
    "0000000000/0000000000/0020202000/0004000000/0020002000/0000000000/0020202000/0000000000/0000000000/0000000000",
    "0000000000/0200200204/0000000000/0000000000/0200000200/0000000000/0000000000/0200200200/0000000000/0000000000",
    "0000000000/0000000000/0000200000/0000002000/0020000000/0000040020/2000000000/0020002000/0000000000/0000200000",
    "0000000000/0020200000/0000000000/2000002000/0000000000/0020200000/0000004000/0000000000/0000000000/0000000000",
    "0000000000/0000200000/0000000000/0020002000/0000000000/2020202020/0000000000/0000400000/0000000000/0000000000",
    "0000000000/0000202020/0000000000/0002000020/0000040000/0002000020/0000000000/0002020200/0000000000/0000000000",
    "0202020204/0000000000/0200000202/0000000000/0200000002/0000000000/0202020202/0000000000/0000000000/0000000000",
))

_RGB = ((0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
        (255, 220, 0), (170, 170, 170), (240, 18, 190),
        (255, 133, 27), (127, 219, 255), (135, 12, 37))
_COLOR_TRIPLES = tuple((a, b, c) for a in range(10) for b in range(10)
                       for c in range(10) if len({a, b, c}) == 3
                       and all(sum((u - v) ** 2 for u, v in zip(_RGB[i], _RGB[j]))
                               >= 150 ** 2 for i, j in ((a, b), (a, c), (b, c))))


def _cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _convex_hull(points):
    points = sorted(set(points))
    lower, upper = [], []
    for p in points:
        while len(lower) > 1 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) > 1 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _relation(point, hull):
    turns = [_cross(hull[i], hull[(i + 1) % len(hull)], point)
             for i in range(len(hull))]
    if any(t < 0 for t in turns):
        return "outside"
    return "boundary" if any(t == 0 for t in turns) else "inside"


def _on_segment(point, a, b):
    return (_cross(a, b, point) == 0
            and min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= point[1] <= max(a[1], b[1]))


def _edge_distance(point, hull):
    px, py = point
    best = float("inf")
    for a, b in zip(hull, hull[1:] + hull[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy)
                          / (dx * dx + dy * dy)))
        best = min(best, math.hypot(px - a[0] - t * dx,
                                    py - a[1] - t * dy))
    return best


def _polygon(rng, bounds, family):
    x0, y0, x1, y1 = bounds
    sx, sy = x1 - x0, y1 - y0
    if family == "triangle":
        mx = int(rng.integers(x0 + 1, x1))
        my = int(rng.integers(y0 + 1, y1))
        options = [
            [(mx, y0), (x1, y1), (x0, y1)],
            [(x0, y0), (x1, y0), (mx, y1)],
            [(x0, y0), (x1, my), (x0, y1)],
            [(x1, y0), (x1, y1), (x0, my)],
        ]
        return options[int(rng.integers(0, 4))]
    if family == "diamond":
        return [(int(rng.integers(x0 + 1, x1)), y0),
                (x1, int(rng.integers(y0 + 1, y1))),
                (int(rng.integers(x0 + 1, x1)), y1),
                (x0, int(rng.integers(y0 + 1, y1)))]
    if family == "skew":
        shift = int(rng.integers(1, max(2, sx // 2)))
        if rng.random() < 0.5:
            return [(x0 + shift, y0), (x1, y0),
                    (x1 - shift, y1), (x0, y1)]
        return [(x0, y0), (x1 - shift, y0),
                (x1, y1), (x0 + shift, y1)]
    cuts = {"rectangle": 0, "five_corner": 1, "six_corner": 2,
            "seven_corner": 3, "octagon": 4}[family]
    cut_corners = set(int(v) for v in rng.choice(4, size=cuts, replace=False))
    cx = [int(rng.integers(1, max(2, sx // 3 + 1))) for _ in range(4)]
    cy = [int(rng.integers(1, max(2, sy // 3 + 1))) for _ in range(4)]
    result = []
    corner_points = [
        ((x0 + cx[0], y0), (x0, y0 + cy[0]), (x0, y0)),
        ((x1 - cx[1], y0), (x1, y0 + cy[1]), (x1, y0)),
        ((x1, y1 - cy[2]), (x1 - cx[2], y1), (x1, y1)),
        ((x0 + cx[3], y1), (x0, y1 - cy[3]), (x0, y1)),
    ]
    for i, (a, b, corner) in enumerate(corner_points):
        result.extend((a, b) if i in cut_corners else (corner,))
    if 0 in cut_corners:
        result[0], result[1] = result[1], result[0]
    return result


def _dots(rng, hull, bounds, regime):
    dots = set(hull)
    candidates = set()
    for a, b in zip(hull, hull[1:] + hull[:1]):
        steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        edge_candidates = []
        for k in range(1, steps):
            p = (round(a[0] + (b[0] - a[0]) * k / steps),
                 round(a[1] + (b[1] - a[1]) * k / steps))
            if p not in dots and _relation(p, hull) != "outside":
                candidates.add(p)
                edge_candidates.append(p)
        # Long sides need visible clues all the way around the figure.
        for j in range(1, min(3, steps // 4) + 1):
            if edge_candidates:
                index = min(len(edge_candidates) - 1,
                            round((len(edge_candidates) - 1) * j
                                  / (min(3, steps // 4) + 1)))
                dots.add(edge_candidates[index])
    candidates = list(candidates)
    rng.shuffle(candidates)
    desired = max(8, len(dots),
                  int(rng.integers(7, 14 if regime == "source_train" else 19)))
    for p in candidates:
        if len(dots) >= desired:
            break
        if min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for q in dots) >= 2:
            dots.add(p)
    if len(dots) < 6:
        for p in candidates:
            dots.add(p)
            if len(dots) >= 6:
                break
    if rng.random() < 0.38:
        x0, y0, x1, y1 = bounds
        inset = [(x, y) for y in range(y0 + 1, y1)
                 for x in range(x0 + 1, x1)
                 if (x, y) not in dots and _relation((x, y), hull) == "inside"
                 and _edge_distance((x, y), hull) <= 1.45]
        rng.shuffle(inset)
        if inset:
            dots.add(inset[0])
    return dots


def _choose_marker(rng, hull, dots, bounds, shape, desired):
    height, width = shape
    x0, y0, x1, y1 = bounds
    candidates = [(x, y) for y in range(height) for x in range(width)
                  if (x, y) not in dots and _relation((x, y), hull) == desired]
    if not candidates:
        return None
    if desired == "inside":
        near = [p for p in candidates if _edge_distance(p, hull) <= 1.6]
        central = [p for p in candidates if _edge_distance(p, hull) >= 2.25]
        pool = near if rng.random() < 0.48 else central
    else:
        within = [p for p in candidates if x0 <= p[0] <= x1
                  and y0 <= p[1] <= y1]
        around = [p for p in candidates if x0 - 2 <= p[0] <= x1 + 2
                  and y0 - 2 <= p[1] <= y1 + 2 and p not in within]
        pool = within if rng.random() < 0.35 else around
    if not pool:
        pool = candidates
    return pool[int(rng.integers(0, len(pool)))]


_TRAIN_FAMILIES = ("triangle", "rectangle", "five_corner", "six_corner",
                   "skew", "diamond")
_TEST_FAMILIES = _TRAIN_FAMILIES + ("seven_corner", "octagon")


def _generate_pair(rng, regime, desired, colors):
    families = _TRAIN_FAMILIES if regime == "source_train" else _TEST_FAMILIES
    for _ in range(120):
        height = int(rng.integers(8, 19 if regime == "source_train" else 23))
        width = int(rng.integers(8, 19 if regime == "source_train" else 23))
        sx = int(rng.integers(4, min(width - 1, 15) + 1))
        sy = int(rng.integers(4, min(height - 1, 15) + 1))
        x0 = int(rng.integers(0, width - sx))
        y0 = int(rng.integers(0, height - sy))
        bounds = (x0, y0, x0 + sx, y0 + sy)
        vertices = _polygon(rng, bounds, families[int(rng.integers(0, len(families)))])
        hull = _convex_hull(vertices)
        if len(hull) != len(vertices) or len(hull) < 3:
            continue
        dots = _dots(rng, hull, bounds, regime)
        if len(dots) < 6:
            continue
        marker = _choose_marker(rng, hull, dots, bounds, (height, width), desired)
        if marker is None:
            continue
        background, dot_color, marker_color = colors
        grid = np.full((height, width), background, dtype=np.int8)
        for x, y in dots:
            grid[y, x] = dot_color
        grid[marker[1], marker[0]] = marker_color
        output = grid.copy()
        if desired == "inside":
            output[output == dot_color] = background
        else:
            output[marker[1], marker[0]] = background
        return {"input": grid, "output": output}
    raise RuntimeError("Could not construct a readable dotted figure.")


def _choose_counts(rng, num_train, num_test):
    for value, minimum, maximum in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, (int, np.integer))
                                  or not minimum <= value <= maximum):
            raise ValueError("Pair count is out of range.")
    options = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not options:
        raise ValueError("Train count must exceed test count.")
    return options[int(rng.integers(0, len(options)))]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    # A singleton is the marker; the repeated color is the dotted figure.
    base = _COLOR_TRIPLES[int(rng.integers(0, len(_COLOR_TRIPLES)))]
    colors = [base, (base[1], base[2], base[0])]
    for _ in range(train_count + test_count - 2):
        if rng.random() < 0.45:
            colors.append(tuple(int(c) for c in rng.permutation(base)))
        else:
            colors.append(_COLOR_TRIPLES[int(rng.integers(0, len(_COLOR_TRIPLES)))])
    labels = ["inside", "outside"] + [
        "inside" if rng.random() < 0.5 else "outside"
        for _ in range(train_count - 2)]
    rng.shuffle(labels)
    test_labels = ["inside" if rng.random() < 0.5 else "outside"
                   for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    first_train_shape = None
    pair_index = 0
    for name, regime, cases in (("train", "source_train", labels),
                                ("test", "target_test", test_labels)):
        for index, desired in enumerate(cases):
            for _ in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _generate_pair(pair_rng, regime, desired, colors[pair_index])
                key = (pair["input"].shape, pair["input"].tobytes())
                if key in seen:
                    continue
                if name == "train" and index == 1 and pair["input"].shape == first_train_shape:
                    continue
                seen.add(key)
                puzzle[name].append(pair)
                if name == "train" and index == 0:
                    first_train_shape = pair["input"].shape
                break
            else:
                raise RuntimeError("Could not generate distinct pairs.")
            pair_index += 1
    return puzzle


def _read_input(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3
    background = int(values[int(np.argmax(counts))])
    assert int(np.max(counts)) > int(np.partition(counts, -2)[-2])
    other = [(int(v), int(n)) for v, n in zip(values, counts) if v != background]
    markers = [v for v, n in other if n == 1]
    outlines = [v for v, n in other if n >= 6]
    assert len(markers) == len(outlines) == 1
    marker_color, dot_color = markers[0], outlines[0]
    ys, xs = np.where(grid == dot_color)
    dots = [(int(x), int(y)) for y, x in zip(ys, xs)]
    hull = _convex_hull(dots)
    assert 3 <= len(hull) <= 8
    assert all(_relation(dot, hull) != "outside" for dot in dots)
    my, mx = (int(v) for v in np.argwhere(grid == marker_color)[0])
    marker = (mx, my)
    relation = _relation(marker, hull)
    assert relation in ("inside", "outside")
    return {"roles": (background, dot_color, marker_color),
            "relation": relation, "hull": hull, "dots": dots,
            "marker": marker,
            "bounds": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}


def _inspect_pair(pair):
    grid, output = pair["input"], pair["output"]
    info = _read_input(grid)
    background, dot_color, marker_color = info["roles"]
    my, mx = info["marker"][1], info["marker"][0]
    expected = grid.copy()
    if info["relation"] == "inside":
        expected[expected == dot_color] = background
    else:
        expected[my, mx] = background
    assert np.array_equal(output, expected)
    return info


def _predict_from_train(train, test_input):
    learned = {}
    for pair in train:
        info = _read_input(pair["input"])
        background, dot_color, marker_color = info["roles"]
        output = pair["output"]
        if np.count_nonzero(output == marker_color) == 0:
            deleted = "marker"
            assert np.count_nonzero(output == dot_color) == len(info["dots"])
        else:
            deleted = "dots"
            assert np.count_nonzero(output == marker_color) == 1
            assert np.count_nonzero(output == dot_color) == 0
        previous = learned.setdefault(info["relation"], deleted)
        assert previous == deleted
    assert set(learned) == {"inside", "outside"}
    info = _read_input(test_input)
    background, dot_color, _ = info["roles"]
    predicted = test_input.copy()
    if learned[info["relation"]] == "dots":
        predicted[predicted == dot_color] = background
    else:
        x, y = info["marker"]
        predicted[y, x] = background
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    details = {"train": [], "test": []}
    for name in ("train", "test"):
        for pair in puzzle[name]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            grid, output = pair["input"], pair["output"]
            assert grid.shape == output.shape
            input_key = (grid.shape, grid.tobytes())
            pair_key = (input_key, output.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if grid.shape == (10, 10):
                key = "/".join("".join(str(int(v)) for v in row) for row in grid)
                assert key not in _SOURCE_INPUTS
            details[name].append(_inspect_pair(pair))
    assert {d["relation"] for d in details["train"]} == {"inside", "outside"}
    assert len({pair["input"].shape for pair in train}) >= 2
    assert len({d["roles"] for d in details["train"]}) >= 2
    for pair in test:
        assert np.array_equal(_predict_from_train(train, pair["input"].copy()),
                              pair["output"])


def _same_puzzle(a, b):
    return all(len(a[name]) == len(b[name])
               and all(np.array_equal(left[part], right[part])
                       for left, right in zip(a[name], b[name])
                       for part in ("input", "output"))
               for name in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, sizes, palettes, layouts = set(), set(), set(), set(), set()
    hull_sizes, test_first, train_first = set(), set(), set()
    off_edge = edge_touch = within_bbox_outside = near_inside = 0
    shared_hull_sizes = changed_roles = novel_test_hulls = 0
    small = large = elongated = off_edge_touch = 0
    marker_sides, dot_counts = set(), set()
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        for pair in puzzle["test"]:
            predicted = _predict_from_train(puzzle["train"], pair["input"].copy())
            assert np.array_equal(predicted, pair["output"])
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple((p["input"].shape, p["input"].tobytes())
                             for name in ("train", "test") for p in puzzle[name]))
        infos = {name: [_inspect_pair(p) for p in puzzle[name]]
                 for name in ("train", "test")}
        train_first.add(infos["train"][0]["relation"])
        test_first.add(infos["test"][0]["relation"])
        shared_hull_sizes += bool({len(d["hull"]) for d in infos["train"]}
                                  & {len(d["hull"]) for d in infos["test"]})
        novel_test_hulls += bool({len(d["hull"]) for d in infos["test"]}
                                 - {len(d["hull"]) for d in infos["train"]})
        changed_roles += len({d["roles"] for d in infos["train"]}) > 1
        for name in ("train", "test"):
            for pair, d in zip(puzzle[name], infos[name]):
                grid = pair["input"]
                sizes.add(grid.shape)
                palettes.add(d["roles"])
                hull_sizes.add(len(d["hull"]))
                dot_counts.add(len(d["dots"]))
                x0, y0, x1, y1 = d["bounds"]
                touching = x0 == 0 or y0 == 0 or x1 == grid.shape[1]-1 or y1 == grid.shape[0]-1
                edge_touch += touching
                inward = any(not any(_on_segment(p, d["hull"][i],
                                                   d["hull"][(i+1) % len(d["hull"])])
                                        for i in range(len(d["hull"]))) for p in d["dots"])
                off_edge += inward
                off_edge_touch += inward and touching
                small += x1 - x0 <= 5 and y1 - y0 <= 5
                large += max(x1 - x0, y1 - y0) >= 11
                elongated += abs((x1 - x0) - (y1 - y0)) >= 4
                if d["relation"] == "outside":
                    x, y = d["marker"]
                    within_bbox_outside += x0 <= x <= x1 and y0 <= y <= y1
                    marker_sides.add("left" if x < x0 else "right" if x > x1
                                     else "top" if y < y0 else "bottom" if y > y1
                                     else "slope")
                else:
                    near_inside += _edge_distance(d["marker"], d["hull"]) <= 1.6
                layouts.add(tuple(sorted((x-x0, y-y0) for x, y in d["dots"])))
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 35 and len(palettes) >= 60 and len(layouts) >= 100
    assert {3, 4, 5, 6, 7, 8} <= hull_sizes
    assert len(dot_counts) >= 10 and small >= 10 and large >= 60 and elongated >= 60
    assert train_first == test_first == {"inside", "outside"}
    assert off_edge >= 15 and edge_touch >= 30 and within_bbox_outside >= 10
    assert near_inside >= 10 and shared_hull_sizes >= 10 and novel_test_hulls >= 8
    assert off_edge_touch >= 10 and len(marker_sides) == 5 and changed_roles == 42
    for a, b in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(123, a, b)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
        assert _same_puzzle(puzzle, generate_puzzle(123, a, b))
    for a, b in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(234, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(345, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted.")
