"""Generate square-size sorting puzzles with pair-local appearance and geometry."""

from itertools import combinations

import numpy as np


_SOURCE_INPUTS = tuple(
    np.array([[int(ch) for ch in row] for row in text.split("/")], dtype=np.int8)
    for text in (
        "01111104440333302222220/01111104440333302222220/"
        "01111104440333302222220/01111100000333302222220/"
        "01111100000000002222220/00000000000000002222220/"
        + "/".join(["00000000000000000000000"] * 6),
        "55555500000/" + "/".join(["50000500000"] * 4)
        + "/55555500000/00000000000/22220000000/"
        "20020000000/20020000000/22220000000/00000000000/"
        "77700000000/70700000000/77700000000",
        "8888880000000/8888880000000/8888880000000/"
        "8888880000666/8888880440666/8888880440666",
        "700000/000000/770000/770000/000000/"
        + "/".join(["777770"] * 5)
        + "/000000/777000/777000/777000",
        "2222000000/2002055500/2002050500/2222055506",
    )
)


def _is_source_input(grid):
    return any(grid.shape == old.shape and np.array_equal(grid, old)
               for old in _SOURCE_INPUTS)


def _runs(active):
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _parse_with_background(grid, background):
    row_runs = _runs(np.any(grid != background, axis=1))
    col_runs = _runs(np.any(grid != background, axis=0))
    if len(col_runs) >= 2 and len(row_runs) == 1:
        axis, bands = "horizontal", col_runs
    elif len(row_runs) >= 2 and len(col_runs) == 1:
        axis, bands = "vertical", row_runs
    else:
        raise AssertionError("Expected one separated sequence")
    assert 2 <= len(bands) <= 6
    gaps = [bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)]
    assert len(set(gaps)) == 1 and 1 <= gaps[0] <= 3

    objects = []
    for start, end in bands:
        slab = grid[:, start:end] if axis == "horizontal" else grid[start:end, :]
        ys, xs = np.nonzero(slab != background)
        assert len(ys)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        tile = slab[y0:y1, x0:x1].copy()
        size = end - start
        assert tile.shape == (size, size)
        border = np.r_[tile[0, :], tile[-1, :], tile[:, 0], tile[:, -1]]
        assert np.all(border == border[0]) and border[0] != background
        color = int(border[0])
        inner = tile[1:-1, 1:-1]
        if np.all(tile == color):
            style = "solid"
        elif size >= 3 and np.all(inner == background):
            style = "hollow"
        else:
            accents = inner[(inner != background) & (inner != color)]
            assert size >= 4 and len(accents) == 1
            assert np.count_nonzero(inner != background) == 1
            style = "marked"
        objects.append({
            "size": size, "color": color, "style": style, "tile": tile,
            "cross_start": y0 if axis == "horizontal" else x0,
            "cross_end": y1 if axis == "horizontal" else x1,
        })
    sizes = [obj["size"] for obj in objects]
    assert len(sizes) == len(set(sizes))
    starts = {obj["cross_start"] for obj in objects}
    ends = {obj["cross_end"] for obj in objects}
    assert len(starts) == 1 or len(ends) == 1
    if axis == "horizontal":
        align = "top" if len(starts) == 1 else "bottom"
    else:
        align = "left" if len(starts) == 1 else "right"
    return {"background": int(background), "axis": axis, "align": align,
            "bands": bands, "gap": gaps[0], "objects": objects}


def _parse(grid):
    # A complete separator row or column supplies the background even if it
    # is nonzero or a solid square occupies more cells than the background.
    candidates = set()
    for row in grid:
        if np.all(row == row[0]):
            candidates.add(int(row[0]))
    for column in grid.T:
        if np.all(column == column[0]):
            candidates.add(int(column[0]))
    matches = []
    for background in candidates:
        try:
            matches.append(_parse_with_background(grid, background))
        except (AssertionError, IndexError):
            pass
    assert len(matches) == 1, "Background or square segmentation is ambiguous"
    return matches[0]


def _transform(grid, info):
    result = np.full(grid.shape, info["background"], dtype=grid.dtype)
    cursor = info["bands"][0][0]
    for obj in sorted(info["objects"], key=lambda item: item["size"]):
        size = obj["size"]
        if info["axis"] == "horizontal":
            y = obj["cross_start"] if info["align"] == "top" else obj["cross_end"] - size
            result[y:y + size, cursor:cursor + size] = obj["tile"]
        else:
            x = obj["cross_start"] if info["align"] == "left" else obj["cross_end"] - size
            result[cursor:cursor + size, x:x + size] = obj["tile"]
        cursor += size + info["gap"]
    return result


def _count_value(value, name, low, high):
    if value is not None and (isinstance(value, (bool, np.bool_))
                              or not isinstance(value, (int, np.integer))
                              or not low <= value <= high):
        raise ValueError(name)


def _choose_counts(seed, rng, num_train, num_test):
    _count_value(num_train, "num_train", 2, 10)
    _count_value(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _choose_sizes(rng, first_training, force_mark):
    counts = (3, 4, 5) if first_training else (2, 3, 4, 5, 6)
    weights = (0.3, 0.4, 0.3) if first_training else (0.12, 0.28, 0.30, 0.22, 0.08)
    count = int(rng.choice(counts, p=weights))
    preferred_max = int(rng.choice((6, 7, 8, 9, 10),
                                   p=(0.35, 0.20, 0.15, 0.15, 0.15)))
    possible = []
    for gap in (1, 2, 3):
        choices = [sizes for sizes in combinations(range(1, preferred_max + 1), count)
                   if sum(sizes) + gap * (count - 1) <= 30
                   and (not force_mark or max(sizes) >= 4)]
        if choices:
            possible.append((gap, choices))
    gap_index = int(rng.choice(len(possible), p=np.array(
        [0.65, 0.24, 0.11][:len(possible)]) /
        sum([0.65, 0.24, 0.11][:len(possible)])))
    gap, choices = possible[gap_index]
    sizes = list(choices[int(rng.integers(len(choices)))])
    return sizes, gap


def _choose_colors(rng, background, count):
    available = [color for color in range(10) if color != background]
    color_count = int(rng.integers(1, min(count, 5) + 1))
    palette = [int(c) for c in rng.choice(available, size=color_count, replace=False)]
    colors = palette + [int(rng.choice(palette)) for _ in range(count - color_count)]
    rng.shuffle(colors)
    return colors


def _make_objects(rng, sizes, background, allow_mark, force_mark):
    count = len(sizes)
    colors = _choose_colors(rng, background, count)
    modes = ["solid", "mixed"]
    if min(sizes) >= 3:
        modes.append("hollow")
    mode = str(rng.choice(modes))
    styles = []
    for size in sizes:
        if size < 3 or mode == "solid":
            styles.append("solid")
        elif mode == "hollow":
            styles.append("hollow")
        else:
            styles.append(str(rng.choice(["solid", "hollow"])))
    if mode == "mixed" and any(size >= 3 for size in sizes):
        eligible = [i for i, size in enumerate(sizes) if size >= 3]
        styles[int(rng.choice(eligible))] = "hollow"
        if all(style == "hollow" for style in styles):
            styles[int(rng.choice(eligible))] = "solid"
    if allow_mark and (force_mark or rng.random() < 0.16):
        eligible = [i for i, size in enumerate(sizes) if size >= 4]
        if eligible:
            styles[int(rng.choice(eligible))] = "marked"

    objects = []
    for size, color, style in zip(sizes, colors, styles):
        tile = np.full((size, size), color, dtype=np.int8)
        if style in ("hollow", "marked"):
            tile[1:-1, 1:-1] = background
        if style == "marked":
            accents = [c for c in range(10) if c not in (background, color)]
            accent = int(rng.choice(accents))
            positions = [(y, x) for y in range(1, size - 1)
                         for x in range(1, size - 1)
                         if (y, x) != ((size - 1) // 2, (size - 1) // 2)]
            y, x = positions[int(rng.integers(len(positions)))]
            tile[y, x] = accent
        objects.append({"size": size, "tile": tile})
    order = rng.permutation(count).tolist()
    ordered = [objects[i] for i in order]
    if [obj["size"] for obj in ordered] == sizes:
        ordered[0], ordered[-1] = ordered[-1], ordered[0]
    return ordered


def _render(objects, axis, align, shape, first, gap, background, cross_lead, cross_tail):
    grid = np.full(shape, background, dtype=np.int8)
    cursor = first
    for obj in objects:
        size = obj["size"]
        if axis == "horizontal":
            y = cross_lead if align == "top" else shape[0] - cross_tail - size
            grid[y:y + size, cursor:cursor + size] = obj["tile"]
        else:
            x = cross_lead if align == "left" else shape[1] - cross_tail - size
            grid[cursor:cursor + size, x:x + size] = obj["tile"]
        cursor += size + gap
    return grid


def _make_pair(rng, regime, allow_mark, force_mark=False, first_training=False):
    sizes, gap = _choose_sizes(rng, first_training, force_mark)
    background = 0 if rng.random() < 0.82 else int(rng.integers(1, 10))
    objects = _make_objects(rng, sizes, background, allow_mark, force_mark)
    axis = str(rng.choice(["horizontal", "vertical"]))
    align = str(rng.choice(["top", "bottom"] if axis == "horizontal" else ["left", "right"]))
    extent = sum(sizes) + gap * (len(sizes) - 1)
    spare = 30 - extent
    first = int(rng.integers(0, min(6, spare) + 1))
    trailing = int(rng.integers(0, min(8, spare - first) + 1))
    cross_lead = int(rng.integers(0, min(5, 30 - max(sizes)) + 1))
    cross_tail = int(rng.integers(0, min(6, 30 - max(sizes) - cross_lead) + 1))
    along = first + extent + trailing
    across = cross_lead + max(sizes) + cross_tail
    shape = (across, along) if axis == "horizontal" else (along, across)
    input_grid = _render(objects, axis, align, shape, first, gap, background,
                         cross_lead, cross_tail)
    output_grid = _render(sorted(objects, key=lambda obj: obj["size"]), axis,
                          align, shape, first, gap, background,
                          cross_lead, cross_tail)
    return {"input": input_grid, "output": output_grid}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(seed, rng, num_train, num_test)
    # Sorting direction is the only puzzle-level convention. A marked interior
    # is introduced by training before it can appear in a test.
    marked_theme = bool(rng.random() < 0.35)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(40):
                pair = _make_pair(pair_rng,
                                  "source_train" if split == "train" else "target_test",
                                  marked_theme, marked_theme and split == "train" and index == 0,
                                  split == "train" and index == 0)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen and not _is_source_input(pair["input"]):
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Unable to make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict_from_demonstrations(train_pairs, test_input):
    # This deliberately receives no seed, metadata, or test output.
    assert len(train_pairs) >= 2
    for pair in train_pairs:
        parsed = _parse(pair["input"])
        assert np.array_equal(pair["output"], _transform(pair["input"], parsed))
    return _transform(test_input, _parse(test_input))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    training_has_mark = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            source, target = pair["input"], pair["output"]
            for grid in (source, target):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert source.shape == target.shape
            key = (source.shape, source.tobytes())
            assert key not in seen and not _is_source_input(source)
            seen.add(key)
            before, after = _parse(source), _parse(target)
            assert before["background"] == after["background"]
            assert before["axis"] == after["axis"]
            assert before["align"] == after["align"]
            assert before["gap"] == after["gap"]
            assert before["bands"][0][0] == after["bands"][0][0]
            assert [o["size"] for o in before["objects"]] != sorted(
                o["size"] for o in before["objects"])
            assert [o["size"] for o in after["objects"]] == sorted(
                o["size"] for o in before["objects"])
            assert np.array_equal(target, _transform(source, before))
            if split == "train":
                training_has_mark |= any(o["style"] == "marked" for o in before["objects"])
            else:
                assert training_has_mark or not any(
                    o["style"] == "marked" for o in before["objects"])
    for pair in puzzle["test"]:
        assert np.array_equal(pair["output"], _predict_from_demonstrations(
            [{"input": p["input"].copy(), "output": p["output"].copy()}
             for p in puzzle["train"]], pair["input"].copy()))


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(pa[key], pb[key])
        for pa, pb in zip(a[split], b[split]) for key in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    counts, shapes, layouts, colors, styles, gaps, margins = (
        set(), set(), set(), set(), set(), set(), set())
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                shapes.add(pair["input"].shape)
                layouts.add((info["axis"], info["align"],
                             tuple(o["size"] for o in info["objects"])))
                colors.add(len(set(np.unique(pair["input"])) - {info["background"]}))
                styles.update(o["style"] for o in info["objects"])
                gaps.add(info["gap"])
                margins.add((info["bands"][0][0],
                             pair["input"].shape[1 if info["axis"] == "horizontal" else 0]
                             - info["bands"][-1][1]))
    assert len(counts) >= 10 and len(shapes) >= 35 and len(layouts) >= 100
    assert {1, 2, 3, 4} <= colors and styles == {"solid", "hollow", "marked"}
    assert gaps == {1, 2, 3} and any(0 in pair for pair in margins)
    for requested in ((2, 1), (2, None), (10, 4), (None, 4), (10, None)):
        puzzle = generate_puzzle(1000, *requested)
        if requested[0] is not None:
            assert len(puzzle["train"]) == requested[0]
        if requested[1] is not None:
            assert len(puzzle["test"]) == requested[1]
    for illegal in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (4, 4)):
        try:
            generate_puzzle(1, *illegal)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts accepted")
