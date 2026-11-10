import numpy as np


_SOURCE_INPUTS = {
    "00000000/00222000/01222110/01222110/01222110/01222110/01222110/00222000",
    "00000000/00000000/00444400/03444430/03444430/00444400/00000000/00000000",
    "00060000/77767000/77767000/77767000/77767000/00060000/00060000/00060000",
    "00000000/00050000/08858800/08858800/00050000/00000000/00000000/00000000",
    "0000000/0555000/0555000/7555777/0555000/0000000/0000000",
    "9000000000/0900000000/0555555000/0555555000/0555555000/0555555000/0000009000/0000000900/0000000090/0000000009",
}


def _key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _source_text(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _rectangle(mask):
    ys, xs = np.nonzero(mask)
    assert len(ys)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    assert len(ys) == (y1 - y0 + 1) * (x1 - x0 + 1)
    return y0, y1, x0, x1


def _stroke(mask):
    ys, xs = np.nonzero(mask)
    assert len(ys)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    h, w = y1 - y0 + 1, x1 - x0 + 1
    if len(ys) == h * w and w > h:
        return "horizontal", h, w, 0, (y0, y1, x0, x1)
    if len(ys) == h * w and h > w:
        return "vertical", w, h, 0, (y0, y1, x0, x1)
    if h == w == len(ys) and h >= 5:
        if all(mask[y0 + i, x0 + i] for i in range(h)):
            return "diagonal", 1, h, 1, (y0, y1, x0, x1)
        if all(mask[y0 + i, x1 - i] for i in range(h)):
            return "diagonal", 1, h, -1, (y0, y1, x0, x1)
    raise AssertionError("not one straight stroke")


def _visible_sides(stroke, occluder, orientation, box):
    oy0, oy1, ox0, ox1 = box
    visible = stroke & ~occluder
    if orientation == "horizontal":
        return bool(np.any(visible[:, :ox0]) and np.any(visible[:, ox1 + 1:]))
    if orientation == "vertical":
        return bool(np.any(visible[:oy0, :]) and np.any(visible[oy1 + 1:, :]))
    return bool(np.any(visible[:oy0, :]) and np.any(visible[oy1 + 1:, :]))


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    input_colors = set(int(v) for v in np.unique(inp))
    output_colors = set(int(v) for v in np.unique(out))
    assert len(input_colors) == 3 and len(output_colors) == 2
    assert output_colors < input_colors
    (occluder,) = tuple(input_colors - output_colors)
    occ = inp == occluder
    box = _rectangle(occ)
    assert np.array_equal(inp[~occ], out[~occ])
    possibilities = []
    for fg in output_colors:
        try:
            orientation, thickness, length, slope, stroke_box = _stroke(out == fg)
            possibilities.append((fg, orientation, thickness, length, slope, stroke_box))
        except AssertionError:
            pass
    assert len(possibilities) == 1
    fg, orientation, thickness, length, slope, stroke_box = possibilities[0]
    (bg,) = tuple(output_colors - {fg})
    stroke = out == fg
    assert np.any(occ & stroke) and np.any(occ & ~stroke)
    assert _visible_sides(stroke, occ, orientation, box)
    assert np.array_equal(inp == fg, stroke & ~occ)
    assert np.array_equal(inp == bg, ~stroke & ~occ)
    return {
        "background": bg, "foreground": fg, "occluder": occluder,
        "orientation": orientation, "thickness": thickness, "length": length,
        "slope": slope, "stroke_box": stroke_box, "occluder_box": box,
    }


def _reconstruct_from_input(inp):
    """Infer all straight-stroke completions using only the input's three colors."""
    colors = [int(v) for v in np.unique(inp)]
    if len(colors) != 3:
        return []
    answers = {}
    for occ_color in colors:
        occ = inp == occ_color
        try:
            box = _rectangle(occ)
        except AssertionError:
            continue
        for fg in colors:
            if fg == occ_color:
                continue
            bg = next(v for v in colors if v not in (fg, occ_color))
            visible = inp == fg
            ys, xs = np.nonzero(visible)
            if not len(ys):
                continue
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            h, w = y1 - y0 + 1, x1 - x0 + 1
            candidate_masks = []
            if w > h:
                mask = np.zeros(inp.shape, dtype=bool)
                mask[y0:y1 + 1, x0:x1 + 1] = True
                candidate_masks.append(mask)
            if h > w:
                mask = np.zeros(inp.shape, dtype=bool)
                mask[y0:y1 + 1, x0:x1 + 1] = True
                candidate_masks.append(mask)
            if h == w and h >= 5:
                for slope in (-1, 1):
                    mask = np.zeros(inp.shape, dtype=bool)
                    for i in range(h):
                        mask[y0 + i, x0 + i if slope == 1 else x1 - i] = True
                    candidate_masks.append(mask)
            for stroke in candidate_masks:
                if not np.array_equal(visible, stroke & ~occ):
                    continue
                try:
                    orientation = _stroke(stroke)[0]
                except AssertionError:
                    continue
                if not _visible_sides(stroke, occ, orientation, box):
                    continue
                if not (np.any(occ & stroke) and np.any(occ & ~stroke)):
                    continue
                out = np.full(inp.shape, bg, dtype=np.int8)
                out[stroke] = fg
                try:
                    _analyze_pair(inp, out)
                except AssertionError:
                    continue
                answers[_key(out)] = out
    return list(answers.values())


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _axis_pair(rng, shape, orientation, thickness, bg, fg, occ_color):
    h, w = shape
    along = w if orientation == "horizontal" else h
    cross = h if orientation == "horizontal" else w
    thick = min(thickness, cross - 1, along - 2)
    assert thick >= 1
    min_length = max(3, thick + 1)
    length = along if rng.random() < 0.30 else int(rng.integers(min_length, along + 1))
    start = int(rng.integers(0, along - length + 1))
    cross_start = int(rng.integers(0, cross - thick + 1))
    hidden = int(rng.integers(1, length - 1))
    before = int(rng.integers(1, length - hidden))
    occ_cross = int(rng.integers(thick + 1, cross + 1))
    low = max(0, cross_start + thick - occ_cross)
    high = min(cross_start, cross - occ_cross)
    occ_cross_start = int(rng.integers(low, high + 1))
    out = np.full(shape, bg, dtype=np.int8)
    inp = out.copy()
    if orientation == "horizontal":
        out[cross_start:cross_start + thick, start:start + length] = fg
        inp = out.copy()
        inp[occ_cross_start:occ_cross_start + occ_cross,
            start + before:start + before + hidden] = occ_color
    else:
        out[start:start + length, cross_start:cross_start + thick] = fg
        inp = out.copy()
        inp[start + before:start + before + hidden,
            occ_cross_start:occ_cross_start + occ_cross] = occ_color
    return inp, out


def _diagonal_pair(rng, shape, bg, fg, occ_color):
    h, w = shape
    length = min(h, w) if rng.random() < 0.35 else int(rng.integers(5, min(h, w) + 1))
    sy = int(rng.integers(0, h - length + 1))
    slope = _choose(rng, (-1, 1))
    sx = int(rng.integers(0, w - length + 1)) if slope == 1 else int(rng.integers(length - 1, w))
    hidden = int(rng.integers(1, min(6, length - 2) + 1))
    before = int(rng.integers(1, length - hidden))
    out = np.full(shape, bg, dtype=np.int8)
    for i in range(length):
        out[sy + i, sx + slope * i] = fg
    path = [(sy + i, sx + slope * i) for i in range(length)]
    target = list(range(before, before + hidden))
    rects = []
    for rh in range(hidden, min(h, hidden + 6) + 1):
        for rw in range(hidden, min(w, hidden + 6) + 1):
            if rh * rw <= hidden:
                continue
            for y0 in range(max(0, path[target[-1]][0] - rh + 1),
                            min(path[target[0]][0], h - rh) + 1):
                xs = [path[i][1] for i in target]
                for x0 in range(max(0, max(xs) - rw + 1),
                                min(min(xs), w - rw) + 1):
                    covered = [i for i, (y, x) in enumerate(path)
                               if y0 <= y < y0 + rh and x0 <= x < x0 + rw]
                    if covered == target:
                        rects.append((y0, x0, rh, rw))
    if not rects:
        # The strip of exactly the hidden rows always has a feasible placement.
        rh, rw = hidden, hidden + 1
        y0 = path[before][0]
        xs = [path[i][1] for i in target]
        x0 = max(0, max(xs) - rw + 1)
        rects.append((y0, x0, rh, rw))
    y0, x0, rh, rw = _choose(rng, rects)
    inp = out.copy()
    inp[y0:y0 + rh, x0:x0 + rw] = occ_color
    return inp, out


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} is outside {low}..{high}")
    return value


def _orientation_plan(rng, ntrain, ntest):
    # The puzzle has no hidden color or direction convention. This plan only
    # controls how much orientation generalization its examples demonstrate.
    mode = int(rng.integers(4))
    orientations = ["horizontal", "vertical", "diagonal"]
    if mode == 3:
        first = int(rng.integers(3))
        train = [orientations[(first + i) % 3] for i in range(ntrain)]
    else:
        train = [orientations[mode]] * ntrain
    test = []
    for _ in range(ntest):
        if mode < 3 and rng.random() < 0.55:
            test.append(_choose(rng, [o for o in orientations if o != train[0]]))
        else:
            test.append(_choose(rng, orientations))
    return train, test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    train_plan, test_plan = _orientation_plan(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, plan in (("train", train_plan), ("test", test_plan)):
        for slot, orientation in enumerate(plan):
            for attempt in range(120):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                if orientation == "diagonal":
                    upper = 15 if split == "train" else 20
                    shape = (int(pair_rng.integers(6, upper + 1)),
                             int(pair_rng.integers(6, upper + 1)))
                else:
                    upper = 16 if split == "train" else 20
                    shape = (int(pair_rng.integers(5, upper + 1)),
                             int(pair_rng.integers(5, upper + 1)))
                if pair_rng.random() < 0.24:
                    side = int(pair_rng.integers(6, upper + 1))
                    shape = (side, side)
                if split == "train" and slot > 0 and shape == puzzle["train"][0]["input"].shape:
                    continue
                bg, fg, occ = (int(v) for v in pair_rng.choice(10, size=3, replace=False))
                if split == "train" and slot > 0:
                    earlier_bg = _analyze_pair(puzzle["train"][0]["input"], puzzle["train"][0]["output"])["background"]
                    if bg == earlier_bg:
                        continue
                if orientation == "diagonal":
                    inp, out = _diagonal_pair(pair_rng, shape, bg, fg, occ)
                else:
                    thickness = int(pair_rng.integers(1, 7 if split == "train" else 8))
                    inp, out = _axis_pair(pair_rng, shape, orientation, thickness, bg, fg, occ)
                key = _key(inp)
                if key in seen or _source_text(inp) in _SOURCE_INPUTS:
                    continue
                try:
                    info = _analyze_pair(inp, out)
                    answers = _reconstruct_from_input(inp)
                    assert info["orientation"] == orientation
                    assert len(answers) == 1 and np.array_equal(answers[0], out)
                except AssertionError:
                    continue
                seen.add(key)
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("Could not construct an unambiguous distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    info_by_split = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key, output_key = _key(inp), _key(out)
            assert input_key not in seen_inputs
            assert (input_key, output_key) not in seen_pairs
            assert _source_text(inp) not in _SOURCE_INPUTS
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            info = _analyze_pair(inp, out)
            answers = _reconstruct_from_input(inp)
            assert len(answers) == 1 and np.array_equal(answers[0], out)
            info_by_split[split].append(info)
    # Local color roles are demonstrated by each pair; there is no latent map.
    assert len({pair["input"].shape for pair in puzzle["train"]}) >= 2
    assert len({info["background"] for info in info_by_split["train"]}) >= 2


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(pa["input"], pb["input"]) and
        np.array_equal(pa["output"], pb["output"])
        for pa, pb in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, orientations = set(), set(), set(), set()
    slopes, backgrounds, occ_aspects, thicknesses = set(), set(), set(), set()
    short_flanks, squareish, full_edge, thin_train = 0, 0, 0, 0
    train_orientations, test_orientations, signatures = set(), set(), set()
    layouts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        local_train, local_test = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _analyze_pair(inp, out)
                orientation = info["orientation"]
                (oy0, oy1, ox0, ox1) = info["occluder_box"]
                (sy0, sy1, sx0, sx1) = info["stroke_box"]
                shapes.add(inp.shape)
                orientations.add(orientation)
                backgrounds.add(info["background"])
                thicknesses.add(info["thickness"])
                occ_aspects.add(np.sign((ox1 - ox0) - (oy1 - oy0)))
                layouts.add((orientation, inp.shape, info["stroke_box"], info["occluder_box"]))
                if orientation == "diagonal":
                    slopes.add(info["slope"])
                    if sy0 == 0 and sy1 == inp.shape[0] - 1 and sx0 == 0 and sx1 == inp.shape[1] - 1:
                        full_edge += 1
                elif orientation == "horizontal":
                    short_flanks += int(ox0 - sx0 == 1 or sx1 - ox1 == 1)
                    squareish += int(info["length"] <= info["thickness"] + 1)
                    full_edge += int(sx0 == 0 and sx1 == inp.shape[1] - 1)
                else:
                    short_flanks += int(oy0 - sy0 == 1 or sy1 - oy1 == 1)
                    squareish += int(info["length"] <= info["thickness"] + 1)
                    full_edge += int(sy0 == 0 and sy1 == inp.shape[0] - 1)
                if split == "train":
                    local_train.append(orientation)
                    thin_train += int(info["thickness"] == 1)
                else:
                    local_test.append(orientation)
                assert len(_reconstruct_from_input(inp)) == 1
        train_orientations.add(tuple(local_train))
        test_orientations.add(tuple(local_test))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 40 and len(layouts) >= 120
    assert len(shapes) >= 30 and len(backgrounds) >= 8
    assert orientations == {"horizontal", "vertical", "diagonal"}
    assert slopes == {-1, 1} and occ_aspects == {-1, 0, 1}
    assert len(thicknesses) >= 5 and len(train_orientations) >= 5
    assert len(test_orientations) >= 5
    assert short_flanks >= 10 and squareish >= 2 and full_edge >= 10 and thin_train >= 5
    for a, b in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(900 + a * 10 + b, a, b)
        assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
        validate_generated_puzzle(puzzle)
    for a, b in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1200 + (a or 0) * 10 + (b or 0), a, b)
        if a is not None:
            assert len(puzzle["train"]) == a
        if b is not None:
            assert len(puzzle["test"]) == b
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 4}):
        try:
            generate_puzzle(77, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _same_puzzle(generate_puzzle(17), generate_puzzle(17))
    assert _same_puzzle(generate_puzzle(17, 5, 3), generate_puzzle(17, 5, 3))
    assert not _same_puzzle(generate_puzzle(17), generate_puzzle(18))
