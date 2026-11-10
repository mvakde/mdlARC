"""Column-local upward propagation puzzles."""

import numpy as np


_SOURCE_INPUTS = {
    ("000001000", "000001000", "010001000", "010003001", "010001001",
     "010001001", "010001003", "030001001", "010001001"),
    ("040000000", "040000000", "070000000", "040400000", "040400000",
     "040404000", "040404000", "040407004", "040404004"),
    ("00050000", "00050000", "00060000", "00050005", "50050005",
     "60050006", "50050005", "50050005"),
    ("000000", "888800", "888800", "882800", "888800", "888800",
     "888800", "288800", "888800", "888200"),
    ("999999999999", "999999999999", "999999999999", "111111111111"),
}


def _count(value, low, high, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _columns(rng, width, n, style, isolated=False):
    if style == "full":
        return list(range(width))
    if style == "dense":
        left = int(rng.integers(0, width - n + 1))
        return list(range(left, left + n))
    if isolated:
        packed = np.sort(rng.choice(width - n + 1, n, replace=False))
        return [int(x + i) for i, x in enumerate(packed)]
    for _ in range(32):
        cols = sorted(int(x) for x in rng.choice(width, n, replace=False))
        gaps = np.diff(cols)
        if np.any(gaps > 1) and (style != "mixed" or np.any(gaps == 1)):
            return cols
    cols = [0, 1] if style == "mixed" else [0]
    while len(cols) < n:
        cols.append(cols[-1] + 2)
    return cols


def _layout(rng, style, split, isolated=False):
    if style == "full":
        height = 4 if int(rng.integers(0, 5)) == 0 else int(rng.integers(5, 17))
        width = int(rng.integers(4, 19))
        cols = list(range(width))
    elif style == "sparse":
        height = int(rng.integers(9, 19))
        width = int(rng.integers(12, 24))
        n = int(rng.integers(4, min(7, (width - 1) // 2) + 1))
        cols = _columns(rng, width, n, style, isolated)
    elif style == "dense":
        height = int(rng.integers(8, 20))
        width = int(rng.integers(5, 19))
        n = int(rng.integers(3, min(9, width) + 1))
        cols = _columns(rng, width, n, style)
    else:
        height = int(rng.integers(9, 21))
        width = int(rng.integers(12, 26))
        n = int(rng.integers(5, min(10, (width + 1) // 2) + 1))
        cols = _columns(rng, width, n, style)

    runs = {}
    rectangle = style == "dense" and int(rng.integers(0, 4)) == 0
    if style == "dense":
        common_top = int(rng.integers(0, height - 4))
        common_bottom = int(rng.integers(common_top + 4, height))
    for col in cols:
        if style == "full":
            top, bottom = 0, height - 1
        elif style == "dense":
            if rectangle:
                top, bottom = common_top, common_bottom
            else:
                top = max(0, min(height - 3, common_top + int(rng.integers(-3, 4))))
                bottom = max(top + 2, min(height - 1, common_bottom + int(rng.integers(-3, 4))))
        else:
            top = int(rng.integers(0, height - 2))
            if int(rng.integers(0, 7)) == 0:
                bottom = top + 1
            else:
                bottom = int(rng.integers(top + 2, height))
        runs[col] = [(top, bottom)]

    if split:
        col = int(rng.choice(cols))
        upper_top = int(rng.integers(0, height - 7))
        upper_bottom = upper_top + int(rng.integers(1, 3))
        lower_top = upper_bottom + int(rng.integers(2, 4))
        lower_bottom = int(rng.integers(lower_top + 2, height))
        runs[col] = [(upper_top, upper_bottom), (lower_top, lower_bottom)]
    else:
        col = None
    return height, width, runs, col


def _solve(grid, background):
    """Infer singleton seeds locally, stopping at each run boundary."""
    answer = grid.copy()
    height, width = grid.shape
    for col in range(width):
        row = 0
        while row < height:
            if grid[row, col] == background:
                row += 1
                continue
            top = row
            while row < height and grid[row, col] != background:
                row += 1
            colors, counts = np.unique(grid[top:row, col], return_counts=True)
            if len(colors) == 1:
                continue
            assert len(colors) == 2 and min(counts) == 1 and max(counts) >= 2
            marker = int(colors[np.argmin(counts)])
            seed = top + int(np.flatnonzero(grid[top:row, col] == marker)[0])
            assert seed > top and row - top >= 3
            answer[top:seed, col] = marker
    return answer


def _make_pair(rng, style, background, force_gap=False, force_evidence=False,
               force_multi=False, isolated=False):
    height, width, runs, split_col = _layout(rng, style, force_gap, isolated)
    cols = list(runs)
    n_marked = int(rng.integers(2 if force_evidence or force_multi else 1, len(cols)))
    marked = set(int(c) for c in rng.choice(cols, n_marked, replace=False))
    if split_col is not None and split_col not in marked:
        marked.remove(next(iter(marked)))
        marked.add(split_col)
    marked_order = sorted(marked)
    rng.shuffle(marked_order)

    if force_evidence:
        short_col, long_col = marked_order[:2]
        runs[short_col] = [(0, int(rng.integers(3, height)))]
        top = int(rng.integers(0, height - 5))
        runs[long_col] = [(top, int(rng.integers(top + 4, height)))]
    else:
        short_col = long_col = None

    colors = [int(x) for x in rng.permutation([x for x in range(10) if x != background])]
    palette = [(colors[0], colors[1]), (colors[2], colors[3])]
    multi = force_multi or int(rng.integers(0, 4)) == 0
    grid = np.full((height, width), background, dtype=np.int8)
    for col in cols:
        assignment = int(rng.integers(0, 2)) if multi else 0
        if force_multi and col in marked:
            assignment = marked_order.index(col) % 2
        base, marker = palette[assignment]
        for run_index, (top, bottom) in enumerate(runs[col]):
            if col in marked and (split_col != col or run_index == 1):
                if bottom - top + 1 < 3:
                    bottom = min(height - 1, top + 2)
                    top = max(0, bottom - 2)
                if col == short_col:
                    seed = top + 1
                elif col == long_col:
                    seed = top + 3
                else:
                    seed = int(rng.integers(top + 1, bottom + 1))
                grid[top:bottom + 1, col] = base
                grid[seed, col] = marker
            else:
                grid[top:bottom + 1, col] = base
    return {"input": grid, "output": _solve(grid, background)}


def _features(grid, output, background):
    assert grid.shape == output.shape
    assert np.array_equal(_solve(grid, background), output)
    height, width = grid.shape
    marked_cols = set()
    short = long = below = bottom_seed = split = stubs = 0
    palettes = set()
    runs_by_col = []
    seeds = {}
    for col in range(width):
        segments = []
        row = 0
        while row < height:
            if grid[row, col] == background:
                row += 1
                continue
            top = row
            while row < height and grid[row, col] != background:
                row += 1
            bottom = row - 1
            run = grid[top:row, col]
            colors, counts = np.unique(run, return_counts=True)
            assert len(colors) <= 2
            if len(colors) == 2:
                assert min(counts) == 1 and max(counts) >= 2
                marker = int(colors[np.argmin(counts)])
                base = int(colors[np.argmax(counts)])
                seed = top + int(np.flatnonzero(run == marker)[0])
                assert seed > top
                marked_cols.add(col)
                seeds[col] = (seed, bottom)
                short += seed - top == 1
                long += seed - top >= 3
                below += seed < bottom
                bottom_seed += seed == bottom
                palettes.add((base, marker))
                segments.append((top, bottom, True))
            else:
                stubs += bottom - top + 1 == 2
                segments.append((top, bottom, False))
        if len(segments) > 1:
            split += 1
            assert len(segments) == 2 and not segments[0][2] and segments[1][2]
            assert segments[0][1] + 1 < segments[1][0]
        if segments:
            runs_by_col.append((col, segments))
    cols = [col for col, _ in runs_by_col]
    assert marked_cols and set(cols) - marked_cols
    touching = any(b - a == 1 for a, b in zip(cols, cols[1:]))
    gaps = any(b - a > 1 for a, b in zip(cols, cols[1:]))
    stepped = any(
        b[0] - a[0] == 1 and
        (a[1][0][0] != b[1][0][0] or a[1][-1][1] != b[1][-1][1])
        for a, b in zip(runs_by_col, runs_by_col[1:])
    )
    occupied_rows, occupied_cols = np.where(grid != background)
    top_row, bottom_row = int(occupied_rows.min()), int(occupied_rows.max())
    rectangle = (
        np.all(grid[top_row:bottom_row + 1, cols[0]:cols[-1] + 1] != background)
        and len(occupied_rows) == (bottom_row - top_row + 1) * (cols[-1] - cols[0] + 1)
    )
    interior_unmarked_block = (
        rectangle and np.any(grid == background) and len(cols) >= 4
        and any(col not in marked_cols for col in cols[1:-1])
    )
    if not np.any(grid == background):
        style = "full"
    elif touching and gaps:
        style = "mixed"
    elif touching:
        style = "dense"
    else:
        style = "sparse"
    return {"style": style, "marked": len(marked_cols),
            "unmarked": len(set(cols) - marked_cols), "short": short,
            "long": long, "below": below, "split": split,
            "bottom_seed": bottom_seed, "stubs": stubs, "stepped": stepped,
            "touching": touching, "gaps": gaps, "palettes": palettes,
            "edge": cols[0] == 0 or cols[-1] == width - 1,
            "interior_unmarked_block": interior_unmarked_block,
            "leftmost_below": cols[0] in seeds and seeds[cols[0]][0] < seeds[cols[0]][1],
            "leftmost_bottom": cols[0] in seeds and seeds[cols[0]][0] == seeds[cols[0]][1],
            "height": height, "width": width, "cols": len(cols)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        if int(rng.integers(0, 5)) == 0:
            num_train, num_test = 2, 1
        else:
            num_test = int(rng.integers(1, 5))
            num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    background = int(rng.choice([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    source_like = num_train == 2 or int(rng.integers(0, 3)) == 0
    train_styles = ["sparse", "sparse" if source_like else "mixed"]
    train_styles += [str(rng.choice(["sparse", "mixed", "dense", "full"]))
                     for _ in range(num_train - 2)]
    test_styles = [str(rng.choice(["dense", "full"] if source_like else
                                  ["dense", "full", "mixed"]))]
    test_styles += [str(rng.choice(["sparse", "dense", "full", "mixed"]))
                    for _ in range(num_test - 1)]
    if num_test >= 3:
        test_styles[1] = "full"
        test_styles[2] = "sparse"
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, styles in (("train", train_styles), ("test", test_styles)):
        for index, style in enumerate(styles):
            for _ in range(40):
                pair = _make_pair(
                    rng, style, background,
                    force_gap=(split == "train" and index == 1) or
                              (split == "test" and style != "full" and int(rng.integers(0, 3)) == 0),
                    force_evidence=split == "train" and index == 0,
                    force_multi=split == "train" and index == 1,
                    isolated=split == "train" and index < 2 and source_like,
                )
                grid = pair["input"]
                key = (grid.shape, grid.tobytes())
                if key not in seen and tuple("".join(map(str, row)) for row in grid) not in _SOURCE_INPUTS:
                    seen.add(key)
                    break
            else:
                raise RuntimeError("distinct layout construction failed")
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
                assert np.issubdtype(array.dtype, np.integer)
                assert np.all((array >= 0) & (array <= 9))
            assert grid.shape == output.shape and not np.array_equal(grid, output)
            ikey = (grid.shape, grid.astype(np.uint8).tobytes())
            pkey = (ikey, output.astype(np.uint8).tobytes())
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            assert tuple("".join(map(str, row)) for row in grid) not in _SOURCE_INPUTS

    first = puzzle["train"][0]["input"]
    values, counts = np.unique(first, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert counts.max() > first.size // 2
    train = [_features(p["input"], p["output"], background) for p in puzzle["train"]]
    test = [_features(p["input"], p["output"], background) for p in puzzle["test"]]
    assert train[0]["style"] in ("sparse", "mixed")
    assert train[0]["marked"] >= 2 and train[0]["unmarked"] >= 1
    assert train[0]["short"] and train[0]["long"] and train[0]["below"]
    assert train[1]["split"] >= 1 and len(train[1]["palettes"]) >= 2
    assert all(f["marked"] >= 1 and f["unmarked"] >= 1 for f in train + test)
    assert all(f["split"] == 0 for f in train + test if f["style"] == "full")


def self_test() -> None:
    def signature(puzzle):
        return tuple((pair["input"].shape, pair["input"].tobytes(), pair["output"].tobytes())
                     for split in ("train", "test") for pair in puzzle[split])

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert all(signature(p) == signature(generate_puzzle(seed)) for seed, p in enumerate(puzzles))
    assert len({signature(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    features = []
    backgrounds = set()
    split_features = {"train": [], "test": []}
    source_like = 0
    combined = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        first = puzzle["train"][0]["input"]
        values, counts = np.unique(first, return_counts=True)
        background = int(values[np.argmax(counts)])
        backgrounds.add(background)
        local = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair["input"], pair["output"], background)
                features.append(f)
                split_features[split].append(f)
                local[split].append(f)
                combined += bool(f["split"] and f["touching"] and len(f["palettes"]) > 1)
        assert len({pair["input"].shape for split in ("train", "test")
                    for pair in puzzle[split]}) >= 2
        source_like += (
            len(puzzle["train"]) == 2
            and all(f["style"] == "sparse" for f in local["train"])
            and local["test"][0]["style"] in ("dense", "full")
        )
    assert len({(f["height"], f["width"]) for f in features}) >= 35
    assert {f["style"] for f in features} == {"sparse", "mixed", "dense", "full"}
    assert any(f["height"] == 4 and f["style"] == "full" for f in features)
    assert any(f["split"] for f in features)
    assert any(f["touching"] and f["gaps"] for f in features)
    assert any(f["unmarked"] >= 2 for f in features)
    assert len(backgrounds) >= 4
    assert source_like >= 4 and combined >= 10
    assert sum(bool(f["split"]) for f in split_features["test"]) >= 5
    assert sum(bool(f["touching"]) for f in split_features["train"]) >= 20
    assert sum(bool(f["stubs"]) for f in features) >= 20
    assert sum(bool(f["stepped"]) for f in features) >= 40
    assert sum(bool(f["short"]) for f in features) >= 40
    assert sum(bool(f["long"]) for f in features) >= 40
    assert sum(bool(f["below"]) for f in features) >= 40
    assert sum(bool(f["bottom_seed"]) for f in features) >= 40
    assert sum(len(f["palettes"]) > 1 for f in features) >= 20
    assert sum(f["style"] == "full" and f["height"] == 4 for f in features) >= 5
    assert sum(f["style"] == "dense" and f["edge"] for f in features) >= 10
    assert sum(bool(f["interior_unmarked_block"]) for f in features) >= 3
    assert sum(f["leftmost_below"] for f in features
               if f["style"] in ("dense", "full")) >= 10
    assert sum(f["leftmost_bottom"] for f in features
               if f["style"] in ("dense", "full")) >= 5
    assert {f["style"] for f in split_features["test"]} == {
        "sparse", "mixed", "dense", "full"
    }
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                          (2, None), (10, None)):
        p = generate_puzzle(900 + (ntrain or 0) * 10 + (ntest or 0), ntrain, ntest)
        validate_generated_puzzle(p)
        if ntrain is not None:
            assert len(p["train"]) == ntrain
        if ntest is not None:
            assert len(p["test"]) == ntest
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (2, 3), (3, 4)):
        try:
            generate_puzzle(11, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
