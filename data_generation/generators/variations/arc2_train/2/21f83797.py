"""Generate the two-marker, extended-lines ARC rectangle family."""
import numpy as np

_SOURCE_MARKERS = (((4, 2), (8, 10)), ((3, 3), (9, 8)), ((1, 4), (10, 9)))


def _count(value, name, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return value


def _counts(seed, rng, num_train, num_test):
    train, test = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _output(shape, bg, marker, fill, top, bottom, left, right):
    out = np.full(shape, bg, dtype=np.int8)
    out[top + 1:bottom, left + 1:right] = fill
    out[top, :] = marker
    out[bottom, :] = marker
    out[:, left] = marker
    out[:, right] = marker
    return out


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _source_copy(pair):
    inp = pair["input"]
    if inp.shape != (13, 13):
        return False
    for coords in _SOURCE_MARKERS:
        source = np.zeros((13, 13), dtype=np.int8)
        for position in coords:
            source[position] = 2
        if np.array_equal(inp, source):
            return True
    return False


def _geometry(rng, cap_h, cap_w):
    kind = int(rng.integers(8))
    if kind == 0:
        return 1, 1
    if kind == 1:
        return 1, int(rng.integers(2, cap_w + 1))
    if kind == 2:
        return int(rng.integers(2, cap_h + 1)), 1
    if kind == 3:
        side = int(rng.integers(1, min(cap_h, cap_w) + 1))
        return side, side
    if kind == 4:
        width = int(rng.integers(2, cap_w + 1))
        return int(rng.integers(1, min(cap_h, width) + 1)), width
    if kind == 5:
        height = int(rng.integers(2, cap_h + 1))
        return height, int(rng.integers(1, min(cap_w, height) + 1))
    return int(rng.integers(1, cap_h + 1)), int(rng.integers(1, cap_w + 1))


def _test_geometry(rng, category, training, cap_h, cap_w):
    high_h = max(h for h, _ in training)
    high_w = max(w for _, w in training)
    if category == 0 and high_h < cap_h:
        return int(rng.integers(high_h + 1, cap_h + 1)), int(rng.integers(1, cap_w + 1))
    if category == 1 and high_w < cap_w:
        return int(rng.integers(1, cap_h + 1)), int(rng.integers(high_w + 1, cap_w + 1))
    if category == 2:
        h, w = training[int(rng.integers(len(training)))]
        return max(1, min(cap_h, h + int(rng.integers(-2, 3)))), max(1, min(cap_w, w + int(rng.integers(-2, 3))))
    if category == 3:
        return int(rng.integers(1, min(cap_h, high_h) + 1)), int(rng.integers(1, min(cap_w, high_w) + 1))
    return _geometry(rng, cap_h, cap_w)


def _axis(rng, interior, mode, inset=False, common=None):
    available = 28 - interior
    if common is not None:
        canvas = common
        slack = canvas - interior - 2
    elif inset:
        before = int(rng.integers(1, min(5, available - 1) + 1))
        after = int(rng.integers(1, min(5, available - before) + 1))
        return interior + before + after + 2, before
    else:
        if mode == "thirty":
            slack = available
        elif mode == "tight":
            slack = int(rng.integers(0, min(3, available) + 1))
        elif mode == "large":
            slack = int(rng.integers(min(15, available), available + 1))
        elif mode == "inset":
            slack = (int(rng.integers(2, min(12, available) + 1))
                     if available >= 2 else available)
        else:
            slack = int(rng.integers(0, available + 1))
        canvas = interior + slack + 2
    if inset or mode == "inset":
        before = int(rng.integers(1, slack)) if slack >= 2 else 0
    else:
        choice = int(rng.integers(4))
        before = 0 if choice == 0 else slack if choice == 1 else int(rng.integers(0, slack + 1))
    return canvas, before


def _palette(rng, fill, previous=None):
    choices = [c for c in range(10) if c != fill and (previous is None or c != previous[0])]
    bg = choices[int(rng.integers(len(choices)))]
    choices = [c for c in range(10) if c not in (fill, bg)]
    marker = choices[int(rng.integers(len(choices)))]
    return bg, marker


def _make_pair(rng, fill, palette, diagonal, geometry, common, ordinary=False, corner=False):
    h, w = geometry
    if corner:
        canvas_h, top, canvas_w, left = 3, 0, 3, 0
    else:
        joint = int(rng.integers(10))
        modes = ("tight", "inset", "large", "spread", "thirty")
        if joint < 3:
            mode_h = mode_w = ("thirty", "tight", "large")[joint]
        else:
            mode_h = modes[int(rng.integers(len(modes)))]
            mode_w = modes[int(rng.integers(len(modes)))]
        canvas_h, top = _axis(rng, h, mode_h, ordinary, None if common is None else common[0])
        canvas_w, left = _axis(rng, w, mode_w, ordinary, None if common is None else common[1])
    bottom, right = top + h + 1, left + w + 1
    bg, marker = palette
    inp = np.full((canvas_h, canvas_w), bg, dtype=np.int8)
    if diagonal == 1:
        inp[top, left], inp[bottom, right] = marker, marker
    else:
        inp[top, right], inp[bottom, left] = marker, marker
    return {"input": inp, "output": _output(inp.shape, bg, marker, fill, top, bottom, left, right)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    fill = int(rng.integers(10))
    change_palette, change_diagonal = bool(rng.integers(2)), bool(rng.integers(2))
    base_palette = _palette(rng, fill)
    base_diagonal = 1 if int(rng.integers(2)) == 0 else -1
    common = None
    if seed % 13 != 0 and int(rng.integers(4)) == 0:
        common = tuple(30 if int(rng.integers(5)) == 0 else int(rng.integers(11, 31)) for _ in range(2))
    cap_h, cap_w = (28, 28) if common is None else (common[0] - 2, common[1] - 2)
    geometries = []
    for index in range(train_count):
        if index == 0:
            geometry = (int(rng.integers(2, min(8, cap_h - 2) + 1)),
                        int(rng.integers(2, min(8, cap_w - 2) + 1)))
        else:
            geometry = _geometry(rng, min(15, cap_h), min(15, cap_w))
            if index == 1 and geometry == geometries[0]:
                geometry = (1, geometry[1]) if geometry[0] != 1 else (min(cap_h, 2), geometry[1])
        geometries.append(geometry)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train":
                geometry = geometries[index]
            else:
                category = (seed + index) % 5 if index == 0 else int(rng.integers(5))
                geometry = _test_geometry(rng, category, geometries, cap_h, cap_w)
            palette = (base_palette if split == "train" and index == 0 else
                       _palette(rng, fill, base_palette if split == "train" and index == 1 else None)
                       if change_palette else base_palette)
            if change_diagonal:
                diagonal = (base_diagonal if split == "train" and index == 0 else
                            -base_diagonal if split == "train" and index == 1
                            else 1 if int(rng.integers(2)) == 0 else -1)
            else:
                diagonal = base_diagonal
            ordinary = split == "train" and index == 0
            corner = split == "test" and index == 0 and seed % 13 == 0
            if corner:
                geometry = (1, 1)
            for attempt in range(80):
                pair = _make_pair(rng, fill, palette, diagonal, geometry, common, ordinary, corner)
                key = _grid_key(pair["input"])
                if key not in seen and not _source_copy(pair):
                    break
                if attempt == 30:
                    corner = False
                    geometry = _geometry(rng, cap_h, cap_w)
            else:
                raise RuntimeError("could not make a unique pair")
            seen.add(key)
            puzzle[split].append(pair)
    if train_count >= 3 and change_diagonal and seed % 4 == 1:
        original = puzzle["train"][0]
        twin_input = original["input"].copy()
        bg, marker = base_palette
        coords = np.argwhere(twin_input == marker)
        top, bottom = sorted(int(v) for v in coords[:, 0])
        left, right = sorted(int(v) for v in coords[:, 1])
        twin_input[coords[:, 0], coords[:, 1]] = bg
        if original["input"][top, left] == marker:
            twin_input[top, right] = twin_input[bottom, left] = marker
        else:
            twin_input[top, left] = twin_input[bottom, right] = marker
        twin = {"input": twin_input, "output": original["output"].copy()}
        if _grid_key(twin_input) not in seen and not _source_copy(twin):
            puzzle["train"][2] = twin
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert 3 <= grid.shape[0] <= 30 and 3 <= grid.shape[1] <= 30
        assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 2
    markers = [int(c) for c, n in zip(colors, counts) if int(n) == 2]
    assert len(markers) == 1
    marker = markers[0]
    bg = int(colors[0] if colors[1] == marker else colors[1])
    coords = np.argwhere(inp == marker)
    assert coords.shape == (2, 2)
    top, bottom = sorted(int(v) for v in coords[:, 0])
    left, right = sorted(int(v) for v in coords[:, 1])
    assert bottom - top >= 2 and right - left >= 2
    fill = int(out[top + 1, left + 1])
    assert fill not in (bg, marker)
    assert np.array_equal(out, _output(inp.shape, bg, marker, fill, top, bottom, left, right))
    assert not _source_copy(pair)
    return {"bg": bg, "marker": marker, "fill": fill,
            "diagonal": 1 if inp[top, left] == marker else -1,
            "geometry": (bottom - top - 1, right - left - 1),
            "shape": inp.shape, "bounds": (top, bottom, left, right),
            "margins": (top, inp.shape[0] - bottom - 1, left, inp.shape[1] - right - 1)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = _grid_key(pair["input"])
            assert key not in seen
            seen.add(key)
            details[split].append(info)
    info = details["train"] + details["test"]
    fills = {x["fill"] for x in info}
    assert len(fills) == 1
    fill = next(iter(fills))
    assert all(fill not in (x["bg"], x["marker"]) for x in info)
    assert any(all(m > 0 for m in x["margins"]) for x in details["train"])
    assert len({(x["shape"], x["bounds"]) for x in details["train"]}) >= 2


def _predict(train, test_input):
    """Use only training grids and the test input; never generator metadata."""
    fills = set()
    for pair in train:
        fills.update(set(int(c) for c in np.unique(pair["output"])) -
                     set(int(c) for c in np.unique(pair["input"])))
    assert len(fills) == 1
    fill = next(iter(fills))
    colors, counts = np.unique(test_input, return_counts=True)
    marker = int(colors[np.flatnonzero(counts == 2)[0]])
    bg = int(colors[0] if colors[1] == marker else colors[1])
    coords = np.argwhere(test_input == marker)
    top, bottom = sorted(int(v) for v in coords[:, 0])
    left, right = sorted(int(v) for v in coords[:, 1])
    answer = np.full(test_input.shape, bg, dtype=np.int8)
    for row in range(answer.shape[0]):
        for col in range(answer.shape[1]):
            if row == top or row == bottom or col == left or col == right:
                answer[row, col] = marker
            elif top < row < bottom and left < col < right:
                answer[row, col] = fill
    return answer


def self_test() -> None:
    def signature(puzzle):
        return tuple(_grid_key(pair[k]) for split in ("train", "test")
                     for pair in puzzle[split] for k in ("input", "output"))
    assert signature(generate_puzzle(8000)) == signature(generate_puzzle(8000))
    assert signature(generate_puzzle(8001, 7, 3)) == signature(generate_puzzle(8001, 7, 3))
    assert signature(generate_puzzle(8001, 7, 3)) != signature(generate_puzzle(8002, 7, 3))
    original = generate_puzzle(0)
    transposed = {split: [{key: pair[key].T.copy() for key in ("input", "output")}
                          for pair in original[split]] for split in ("train", "test")}
    validate_generated_puzzle(transposed)
    boundary_input = np.zeros((3, 3), dtype=np.int8)
    boundary_input[0, 0] = boundary_input[2, 2] = 2
    boundary_output = np.full((3, 3), 2, dtype=np.int8)
    boundary_output[1, 1] = 1
    _analyze_pair({"input": boundary_input, "output": boundary_output})
    for seed, train, test in ((8100, 2, 1), (8101, 10, 1), (8102, 10, 4),
                              (8200, None, 1), (8201, None, 4), (8202, 2, None), (8203, 10, None)):
        p = generate_puzzle(seed, train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
        validate_generated_puzzle(p)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(8300, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts, shapes, geometries, layouts, palettes = (set() for _ in range(6))
    diagonals = set()
    modes = [0, 0, 0, 0]
    coverage = [0] * 8
    review = {name: 0 for name in (
        "background_change", "marker_change", "mixed_training_diagonal",
        "small_large", "unequal_margins", "near_full", "full_30",
        "thin_horizontal", "thin_vertical", "same_size_other_position",
        "no_output_background", "taller_test", "wider_test",
        "smaller_test", "similar_test")}
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        ts = [_analyze_pair(pair) for pair in p["train"]]
        es = [_analyze_pair(pair) for pair in p["test"]]
        info = ts + es
        palette_set = {(x["bg"], x["marker"]) for x in info}
        diagonal_set = {x["diagonal"] for x in info}
        modes[0] += len(palette_set) > 1
        modes[1] += len(palette_set) == 1
        modes[2] += len(diagonal_set) > 1
        modes[3] += len(diagonal_set) == 1
        review["background_change"] += len({x["bg"] for x in info}) > 1
        review["marker_change"] += len({x["marker"] for x in info}) > 1
        review["mixed_training_diagonal"] += len({x["diagonal"] for x in ts}) > 1
        coverage[6] += len({x["shape"] for x in info}) == 1
        for a in range(len(p["train"])):
            for b in range(a + 1, len(p["train"])):
                coverage[7] += (np.array_equal(p["train"][a]["output"], p["train"][b]["output"])
                                and not np.array_equal(p["train"][a]["input"], p["train"][b]["input"]))
                ai, bi = ts[a], ts[b]
                review["same_size_other_position"] += (ai["geometry"] == bi["geometry"]
                                                        and ai["bounds"] != bi["bounds"])
        max_h, max_w = max(x["geometry"][0] for x in ts), max(x["geometry"][1] for x in ts)
        max_area = max(x["geometry"][0] * x["geometry"][1] for x in ts)
        for x in es:
            h, w = x["geometry"]
            review["taller_test"] += h > max_h
            review["wider_test"] += w > max_w
            review["smaller_test"] += h * w < max_area
            review["similar_test"] += min(abs(h * w - y["geometry"][0] * y["geometry"][1])
                                          for y in ts) <= 3
        for split in ("train", "test"):
            for pair in p[split]:
                x = _analyze_pair(pair)
                shapes.add(x["shape"])
                geometries.add(x["geometry"])
                layouts.add((x["shape"], x["bounds"], x["diagonal"]))
                palettes.add((x["bg"], x["marker"], x["fill"]))
                diagonals.add(x["diagonal"])
                h, w = x["geometry"]
                coverage[0] += 0 in x["margins"]
                coverage[1] += 30 in x["shape"]
                coverage[2] += (h, w) == (1, 1)
                coverage[3] += h == 1 or w == 1
                coverage[4] += h == w
                coverage[5] += w >= 2 * h
                review["small_large"] += h * w <= 9 and min(x["shape"]) >= 20
                review["unequal_margins"] += max(x["margins"]) - min(x["margins"]) >= 8
                review["near_full"] += h >= x["shape"][0] - 4 and w >= x["shape"][1] - 4
                review["full_30"] += x["shape"] == (30, 30)
                review["thin_horizontal"] += h == 1 and w > 1
                review["thin_vertical"] += w == 1 and h > 1
                review["no_output_background"] += x["bg"] not in np.unique(pair["output"])
                if split == "test":
                    assert np.array_equal(_predict(p["train"], pair["input"]), pair["output"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 80 and len(geometries) >= 60 and len(layouts) >= 170
    assert len(palettes) >= 50 and diagonals == {-1, 1}
    assert min(modes) >= 5 and all(coverage) and all(review.values())
