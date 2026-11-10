import numpy as np


GUIDE = {"H": (0, 1), "V": (1, 0), "D": (1, 1), "A": (1, -1)}
PERP = {"H": (1, 0), "V": (0, 1), "D": (1, -1), "A": (1, 1)}


def _count(value, name, low, high):
    if value is not None and (isinstance(value, bool)
                              or not isinstance(value, (int, np.integer))
                              or not low <= value <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _interpret(grid):
    """Infer guide, seed, and result using only the input grid."""
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) >= 2 and counts.max() > grid.size // 2
    background = int(colors[np.argmax(counts)])
    marks = [(int(r), int(c)) for r, c in np.argwhere(grid != background)]
    candidates = []
    for orientation, (dr, dc) in GUIDE.items():
        groups = {}
        for r, c in marks:
            line = r if orientation == "H" else c if orientation == "V" else (
                r - c if orientation == "D" else r + c)
            groups.setdefault((int(grid[r, c]), line), []).append((r, c))
        for (guide_color, _), points in groups.items():
            if len(points) < 3 or len(points) != len(marks) - 1:
                continue
            points.sort(key=lambda p: p[1] if orientation == "H" else p[0])
            spacing = points[1][1] - points[0][1] if orientation == "H" else points[1][0] - points[0][0]
            if spacing not in (2, 3, 4):
                continue
            r0, c0 = points[0]
            if any((r, c) != (r0 + i * spacing * dr, c0 + i * spacing * dc)
                   for i, (r, c) in enumerate(points)):
                continue
            sr, sc = next(p for p in marks if p not in points)
            pr, pc = PERP[orientation]
            span = spacing * (len(points) - 1)
            for meeting in range(span + 1):
                mr, mc = r0 + meeting * dr, c0 + meeting * dc
                delta_r, delta_c = sr - mr, sc - mc
                signed = delta_r if pr else delta_c
                if (delta_r, delta_c) != (signed * pr, signed * pc):
                    continue
                distance = abs(signed)
                if distance < 2:
                    continue
                side = 1 if signed > 0 else -1
                if any(grid[sr - k * side * pr, sc - k * side * pc] != background
                       for k in range(1, distance)):
                    continue
                expected = grid.copy()
                seed_color = int(grid[sr, sc])
                for k in range(1, distance):
                    expected[sr - k * side * pr, sc - k * side * pc] = seed_color
                candidates.append({
                    "orientation": orientation, "spacing": spacing,
                    "dot_count": len(points), "meeting": meeting,
                    "crossing": "dot" if meeting % spacing == 0 else "gap",
                    "endpoint": meeting in (0, span),
                    "distance": distance, "side": side,
                    "background": background, "guide_color": guide_color,
                    "seed_color": seed_color, "seed": (sr, sc),
                    "shape": grid.shape, "start": points[0], "expected": expected,
                })
    return candidates


def _source_inputs():
    cases = [
        (12, 12, [(r, 1, 5) for r in range(0, 12, 2)], (5, 11, 4)),
        (9, 14, [(1, c, 7) for c in range(0, 14, 2)], (7, 8, 7)),
        (7, 11, [(2*i, 2*i, 3) for i in range(4)], (6, 0, 2)),
        (11, 12, [(2, c, 1) for c in range(0, 12, 2)], (10, 0, 2)),
        (19, 20, [(13, c, 4) for c in range(0, 16, 2)], (5, 7, 7)),
        (12, 12, [(7, 10, 6), (9, 8, 6), (11, 6, 6)], (10, 11, 8)),
    ]
    signatures = set()
    for height, width, dots, seed in cases:
        grid = np.zeros((height, width), dtype=np.int8)
        for r, c, color in dots + [seed]:
            grid[r, c] = color
        signatures.add((grid.shape, grid.tobytes()))
    return signatures


def _colors(rng, same):
    background = 0 if rng.random() < 0.55 else int(rng.integers(1, 10))
    choices = [c for c in range(10) if c != background]
    guide = int(rng.choice(choices))
    seed = guide if same else int(rng.choice([c for c in choices if c != guide]))
    return background, guide, seed


def _draw_pair(rng, orientation, spacing, crossing, same, size_band,
               favor_endpoint, favor_edge, minimum_distance, seen):
    dr, dc = GUIDE[orientation]
    pr, pc = PERP[orientation]
    low, high = {"small": (7, 13), "medium": (11, 21),
                 "large": (19, 31)}[size_band]
    original = _source_inputs()
    for _ in range(1200):
        height, width = (int(x) for x in rng.integers(low, high, size=2))
        maximum = min(9, (height - 1) // spacing + 1 if dr else 9,
                      (width - 1) // spacing + 1 if dc else 9)
        if maximum < 3:
            continue
        dot_count = int(rng.integers(3, maximum + 1))
        span = spacing * (dot_count - 1)
        r0 = int(rng.integers(0, height - span * dr))
        c0 = int(rng.integers(span if dc < 0 else 0,
                              width - span if dc > 0 else width))
        if crossing == "dot":
            meeting = int(rng.choice([0, span])) if favor_endpoint else (
                spacing * int(rng.integers(1, dot_count - 1)))
        else:
            meeting = int(rng.choice([i for i in range(1, span)
                                      if i % spacing != 0]))
        mr, mc = r0 + meeting * dr, c0 + meeting * dc
        side = int(rng.choice([-1, 1]))
        maximum_distance = 0
        while (0 <= mr + (maximum_distance + 1) * side * pr < height
               and 0 <= mc + (maximum_distance + 1) * side * pc < width):
            maximum_distance += 1
        if maximum_distance < minimum_distance:
            continue
        if favor_edge:
            if maximum_distance > 18:
                continue
            distance = maximum_distance
        elif rng.random() < 0.18:
            distance = minimum_distance
        else:
            distance = int(rng.integers(
                minimum_distance, min(maximum_distance, 14) + 1))
        sr, sc = mr + distance * side * pr, mc + distance * side * pc
        background, guide_color, seed_color = _colors(rng, same)
        inp = np.full((height, width), background, dtype=np.int8)
        for i in range(dot_count):
            inp[r0 + i * spacing * dr, c0 + i * spacing * dc] = guide_color
        if inp[sr, sc] != background:
            continue
        inp[sr, sc] = seed_color
        key = (inp.shape, inp.tobytes())
        if key in seen or key in original:
            continue
        interpretations = _interpret(inp)
        if len(interpretations) != 1:
            continue
        info = interpretations[0]
        if (info["orientation"], info["spacing"], info["meeting"],
            info["seed"]) != (orientation, spacing, meeting, (sr, sc)):
            continue
        seen.add(key)
        return {"input": inp, "output": info["expected"]}
    raise RuntimeError("Could not place an unambiguous guide and seed")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    _count(num_train, "num_train", 2, 10)
    _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        number = abs(int(seed))
        num_train = 2 + (number * 7) % 9
        num_test = 1 + (number // 9) % min(4, num_train - 1)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # The perpendicular-extension rule is the only puzzle-wide convention.
    # Show a chosen spacing in training before it can appear in testing.
    spacing_cap = int(rng.integers(2, 5))
    first_two = [str(rng.choice(["D", "A"])),
                 str(rng.choice(["H", "V"]))]
    rng.shuffle(first_two)
    same_first = bool(rng.integers(2))
    train_specs = [
        (first_two[0], spacing_cap, "dot", same_first),
        (first_two[1], 2, "gap", not same_first),
    ]
    for _ in range(2, num_train):
        train_specs.append((
            str(rng.choice(["H", "V", "D", "A"])),
            int(rng.integers(2, spacing_cap + 1)),
            str(rng.choice(["dot", "gap"])), bool(rng.integers(2)),
        ))
    rng.shuffle(train_specs)
    shown_spacings = sorted({spec[1] for spec in train_specs})
    unseen = [o for o in GUIDE if o not in {spec[0] for spec in train_specs}]
    test_specs = []
    for index in range(num_test):
        orientation = str(rng.choice(unseen if index == 0 and unseen
                                      and rng.random() < 0.65 else list(GUIDE)))
        test_specs.append((
            orientation, int(rng.choice(shown_spacings)),
            str(rng.choice(["dot", "gap"])), bool(rng.integers(2)),
        ))

    seen = set()
    def sample(spec, minimum_distance=2):
        band = str(rng.choice(["small", "medium", "large"],
                              p=[0.30, 0.42, 0.28]))
        endpoint = spec[2] == "dot" and rng.random() < 0.30
        edge = rng.random() < 0.28
        return _draw_pair(rng, *spec, band, endpoint, edge,
                          minimum_distance, seen)

    return {"train": [sample(spec, 4 if spec[2] == "gap" else 3)
                     for spec in train_specs],
            "test": [sample(spec) for spec in test_specs]}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    original = _source_inputs()
    seen_inputs, seen_pairs = set(), set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            pair_key = (key, out.shape, out.tobytes())
            assert key not in seen_inputs and key not in original
            assert pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            interpretations = _interpret(inp)
            assert len(interpretations) == 1, "Input geometry is ambiguous"
            info = interpretations[0]
            assert np.array_equal(out, info["expected"])
            metadata[split].append(info)
    training, tests = metadata["train"], metadata["test"]
    assert {"dot", "gap"} <= {m["crossing"] for m in training}
    assert {True, False} <= {m["guide_color"] == m["seed_color"] for m in training}
    assert any(m["orientation"] in ("D", "A") for m in training)
    assert any(m["orientation"] in ("H", "V") for m in training)
    assert {m["spacing"] for m in tests} <= {m["spacing"] for m in training}


def _same_puzzle(left, right):
    return (all(len(left[s]) == len(right[s]) for s in ("train", "test"))
            and all(np.array_equal(a[name], b[name])
                    for split in ("train", "test")
                    for a, b in zip(left[split], right[split])
                    for name in ("input", "output")))


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    shapes, layouts, backgrounds, colors, distances, dot_counts = (
        set(), set(), set(), set(), set(), set())
    orientations = {"train": set(), "test": set()}
    spacings = {"train": set(), "test": set()}
    crossings = {"train": set(), "test": set()}
    endpoints = edges = same_colors = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple((p["input"].shape, p["input"].tobytes())
                             for split in ("train", "test") for p in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _interpret(pair["input"])[0]
                assert np.array_equal(pair["output"], info["expected"])
                orientations[split].add(info["orientation"])
                spacings[split].add(info["spacing"])
                crossings[split].add(info["crossing"])
                backgrounds.add(info["background"])
                colors.update((info["guide_color"], info["seed_color"]))
                distances.add(info["distance"])
                dot_counts.add(info["dot_count"])
                shapes.add(info["shape"])
                layouts.add((info["orientation"], info["spacing"],
                             info["dot_count"], info["meeting"],
                             info["distance"], info["start"]))
                endpoints += info["endpoint"]
                r, c = info["seed"]
                h, w = info["shape"]
                edges += r in (0, h-1) or c in (0, w-1)
                same_colors += info["guide_color"] == info["seed_color"]
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 40 and len(shapes) >= 60 and len(layouts) >= 100
    assert orientations["train"] == orientations["test"] == set(GUIDE)
    assert spacings["train"] == spacings["test"] == {2, 3, 4}
    assert crossings["train"] == crossings["test"] == {"dot", "gap"}
    assert len(backgrounds) >= 7 and len(colors) >= 8
    assert min(distances) == 2 and max(distances) >= 10
    assert min(dot_counts) == 3 and max(dot_counts) >= 7
    assert endpoints >= 10 and edges >= 20 and same_colors >= 30
    for seed, ntrain, ntest in (
        (100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 5, 3),
        (104, None, 1), (105, None, 4), (106, 2, None), (107, 10, None),
    ):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed, ntrain, ntest))
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(200, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {ntrain}, {ntest}")
