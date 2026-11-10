import numpy as np
from collections import Counter


_SOURCE_INPUTS = {
    tuple(block.splitlines())
    for block in (
        """0000000000000
0000000000000
0000000000000
0388333333330
0338333333330
0668666666660
0668666666660
0778887777770
0777787777770
0444488444440
0444448888440""",
        """33338888883
33338333333
22228888222
22222228222
77777778777
77777778777
77778888777
99988999999
11181111111
11188111111
11118881111""",
        """0000000000
0388833330
0333833330
0388833330
0383333330
0383333330
0188111110
0118111110
0118888110
0111118110
0000000000""",
        """333333883333
333333383333
333338883333
333338333333
777888777777
777877777777
777888777777
777778777777
444448888444
444444448444
444444448444""",
        """000000000000000
088333333333330
038333333333330
038333333333330
038888333333330
044448444444440
077778888887770
011111111181110
028888888882220
098999999999990
000000000000000""",
        """0000000000
0003833000
0003833000
0003883000
0005585500
0005885500
0005855500
0008822222
0008222222
0008222222
0018111100
0088111100
0081111100""",
    )
}


def _grid_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(cell)) for cell in row) for row in grid)


def _resolve_counts(seed, rng, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not lo <= value <= hi):
            raise ValueError("Illegal pair count")
    if num_train is None and num_test is None:
        num_test = 1 + int(seed) % 4
        num_train = num_test + 1 + (int(seed) // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _path(rng, rows, width, long_sweep):
    mark = np.zeros((rows, width), dtype=bool)
    x = int(rng.integers(1, width - 1))
    long_row = int(rng.integers(rows)) if long_sweep else -1
    for y in range(rows):
        if y == long_row:
            candidates = [v for v in range(1, width - 1) if abs(v - x) >= 5]
            nx = int(rng.choice(candidates)) if candidates else x
        elif rng.random() < .22:
            nx = x
        elif rng.random() < .7:
            nx = int(np.clip(x + int(rng.integers(-3, 4)), 1, width - 2))
        else:
            nx = int(rng.integers(1, width - 1))
        mark[y, min(x, nx):max(x, nx) + 1] = True
        x = nx
    return mark


def _random_extent(rng, width, path_slice, outside_side=None):
    xs = np.where(path_slice)[1]
    xmin, xmax = int(xs.min()), int(xs.max())
    if outside_side == "left":
        choices = [v for v in range(xmin + 1, min(xmax + 1, width - 4))]
        if choices:
            left = int(rng.choice(choices))
            right = int(rng.integers(max(left + 4, xmax), width))
            return left, right
    if outside_side == "right":
        choices = [v for v in range(max(xmin, 4), xmax)]
        if choices:
            right = int(rng.choice(choices))
            left = int(rng.integers(0, min(right - 4, xmin) + 1))
            return left, right
    for _ in range(30):
        length = int(rng.integers(5, width + 1))
        left = int(rng.integers(0, width - length + 1))
        right = left + length - 1
        if np.any((xs >= left) & (xs <= right)):
            return left, right
    return 0, width - 1


def _make_pair(rng, regime, background, tracer, band_count,
               forced_lower_color=None, forced_top_color=None,
               sparse=False, geometry=None, min_top=0, min_retained=0):
    if regime not in ("source_train", "target_test"):
        raise ValueError("Unknown regime")
    train = regime == "source_train"
    # The two regimes overlap. The target split gives somewhat more weight
    # to larger, staggered, and outside-crossing configurations.
    height = int(rng.integers(9, 19 if train else 24))
    width = int(rng.integers(9, 20 if train else 25))
    height = max(height, band_count + 4)
    if min_retained >= 8:
        height = max(height, 12)
    uniform = (geometry == "uniform" or
               (geometry != "staggered" and rng.random() < (.27 if train else .18)))
    want_outside = not sparse and rng.random() < (.32 if train else .58)
    want_narrow_top = not uniform and rng.random() < .32
    long_sweep = rng.random() < .42
    edge_touch = not uniform and rng.random() < .32
    for attempt in range(180):
        fallback = attempt >= 120
        # At least two rows in the top band and one in every other band.
        colored = (height - 1 if fallback and min_retained >= 8 else
                   int(rng.integers(max(band_count + 2, height - 4), height)))
        heights = [2] + [1] * (band_count - 1)
        for _ in range(colored - sum(heights)):
            heights[int(rng.integers(1 if fallback and min_retained >= 8
                                      else 0, band_count))] += 1
        top_margin = int(rng.integers(0, height - colored + 1))
        spans = []
        y = 0
        for size in heights:
            spans.append((y, y + size))
            y += size
        mark = _path(rng, colored, width, long_sweep and not sparse)
        if fallback:
            mark[:] = False
            x = int(rng.integers(2, width - 5))
            mark[:, x] = True
            if min_top >= 5:
                mark[0, x:x + 4] = True
            if geometry == "staggered":
                mark[spans[1][0], x + 1] = True
        if sparse:
            x = int(rng.choice(np.flatnonzero(mark[spans[0][1] - 1])))
            mark[spans[0][1]:] = False
            mark[spans[0][1]:, x] = True
        if long_sweep and not sparse and not fallback and max(int(row.sum()) for row in mark) < 6:
            continue
        if sparse and int(mark[spans[0][1]:].sum()) > max(6, colored // 2):
            continue
        if uniform or (fallback and geometry != "staggered"):
            xs = np.where(mark)[1]
            left = int(rng.integers(0, int(xs.min())))
            right = int(rng.integers(int(xs.max()) + 1, width))
            if right - left < 4:
                continue
            bounds = [(left, right)] * band_count
        else:
            bounds = []
            outside_index = int(rng.integers(1, band_count)) if want_outside else -1
            side = str(rng.choice(("left", "right")))
            for index, (start, end) in enumerate(spans):
                bounds.append(_random_extent(rng, width, mark[start:end],
                                side if index == outside_index else None))
            if fallback:
                x = int(np.where(mark[0])[0][0])
                bounds = [(0, width - 1)] * band_count
                bounds[1] = (x + 1, width - 1)
            if edge_touch and not fallback:
                index = int(rng.integers(band_count))
                left, right = bounds[index]
                bounds[index] = (0, right) if rng.random() < .5 else (left, width - 1)
            if want_narrow_top and not fallback:
                top_width = bounds[0][1] - bounds[0][0]
                if not any(r - l >= top_width + 2 for l, r in bounds[1:]):
                    continue
            if want_outside and not fallback:
                start, end = spans[outside_index]
                left, right = bounds[outside_index]
                xs = np.where(mark[start:end])[1]
                if not np.any((xs < left) | (xs > right)):
                    continue
            if len(set(bounds)) == 1:
                continue
        colors_available = [c for c in range(10) if c not in (background, tracer)]
        draw_pool = [c for c in colors_available
                     if c not in (forced_lower_color, forced_top_color)]
        colors = list(map(int, rng.choice(draw_pool, size=band_count, replace=False)))
        if forced_top_color is not None:
            colors[0] = forced_top_color
        if forced_lower_color is not None:
            lower = int(rng.integers(1, band_count))
            colors[lower] = forced_lower_color
        elif band_count >= 3 and rng.random() < .18:
            lower = int(rng.integers(2, band_count))
            colors[lower] = colors[0]
        grid = np.full((height, width), background, dtype=np.int8)
        for (start, end), (left, right), color in zip(spans, bounds, colors):
            grid[top_margin + start:top_margin + end, left:right + 1] = color
        ys, xs = np.where(mark)
        grid[top_margin + ys, xs] = tracer
        cutoff = top_margin + heights[0]
        output = np.where((grid == tracer) &
                          (np.arange(height)[:, None] >= cutoff),
                          tracer, background).astype(np.int8)
        if output.shape == grid.shape and not np.array_equal(output, grid):
            try:
                info = _analyze_pair(grid, output, background, tracer)
            except AssertionError:
                continue
            if (info["band_count"] == band_count and
                    info["top_tracer"] >= min_top and
                    info["retained"] >= min_retained and
                    (not sparse or info["retained"] <= 6)):
                return grid, output
    raise RuntimeError("Could not construct a valid band and tracer pair")


def _analyze_pair(grid, output, background, tracer):
    assert grid.shape == output.shape
    row_colors = []
    for row in grid:
        fills = np.unique(row[(row != background) & (row != tracer)])
        assert len(fills) <= 1
        row_colors.append(int(fills[0]) if len(fills) else None)
    colored = [y for y, color in enumerate(row_colors) if color is not None]
    assert colored and colored == list(range(colored[0], colored[-1] + 1))
    spans = []
    colors = []
    for y in colored:
        color = row_colors[y]
        if not colors or colors[-1] != color:
            colors.append(color)
            spans.append([y, y + 1])
        else:
            spans[-1][1] = y + 1
    assert 2 <= len(spans) <= 6
    assert len(spans[0]) == 2
    bounds = []
    outside_left = outside_right = False
    for index, ((start, end), color) in enumerate(zip(spans, colors)):
        fill = grid[start:end] == color
        assert np.all(fill.sum(axis=1) >= 3)
        xx = np.where(fill)[1]
        left, right = int(xx.min()), int(xx.max())
        assert np.all(grid[start:end, left:right + 1] != background)
        bounds.append((left, right))
        tracer_x = np.where(grid[start:end] == tracer)[1]
        assert len(tracer_x) > 0
        if index > 0:
            outside_left |= bool(np.any(tracer_x < left))
            outside_right |= bool(np.any(tracer_x > right))
    cells = set(zip(*np.where(grid == tracer)))
    assert cells
    frontier = [next(iter(cells))]
    seen = set(frontier)
    while frontier:
        y, x = frontier.pop()
        for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    assert seen == cells
    cutoff = spans[0][1]
    expected = np.where((grid == tracer) &
                        (np.arange(grid.shape[0])[:, None] >= cutoff),
                        tracer, background)
    assert np.array_equal(output, expected)
    assert np.any(grid[:cutoff] == tracer)
    assert np.any(output == tracer)
    return {"band_count": len(spans), "colors": tuple(colors),
            "bounds": tuple(bounds), "cutoff": cutoff,
            "outside_left": outside_left, "outside_right": outside_right,
            "max_run": max(int(np.sum(row == tracer)) for row in grid),
            "retained": int(np.sum(output == tracer)),
            "top_tracer": int(np.sum(grid[:cutoff] == tracer))}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    background = int(rng.integers(10))
    tracer = int(rng.choice([c for c in range(10) if c != background]))
    source_like_geometry = int(seed) % 11 == 0
    shared_fill = int(rng.choice([c for c in range(10)
                                  if c not in (background, tracer)]))
    top_example = int(rng.integers(train_count))
    lower_example = int(rng.choice([i for i in range(train_count)
                                     if i != top_example]))
    puzzle = {"train": [], "test": []}
    used = set()
    max_top = max_retained = 0
    for split, regime, count in (("train", "source_train", train_count),
                                 ("test", "target_test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            # Counts are drawn independently of pair position and grid size.
            band_count = int(pair_rng.integers(2, 7))
            last_training = split == "train" and index == train_count - 1
            need_top = 5 if last_training and max_top < 5 else 0
            need_retained = 8 if last_training and max_retained < 8 else 0
            for _ in range(80):
                grid, output = _make_pair(pair_rng, regime, background,
                    tracer, band_count,
                    shared_fill if split == "train" and index == lower_example else None,
                    shared_fill if split == "train" and index == top_example else None,
                    sparse=(split == "train" and index > 1 and band_count <= 3 and
                            need_retained == 0 and pair_rng.random() < .12),
                    geometry=("uniform" if split == "train" else "staggered")
                             if source_like_geometry else None,
                    min_top=need_top, min_retained=need_retained)
                key = _grid_key(grid)
                if key not in used and key not in _SOURCE_INPUTS:
                    break
            else:
                raise RuntimeError("Could not produce a unique pair")
            used.add(key)
            if split == "train":
                info = _analyze_pair(grid, output, background, tracer)
                max_top = max(max_top, info["top_tracer"])
                max_retained = max(max_retained, info["retained"])
            puzzle[split].append({"input": grid, "output": output})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    pairs = puzzle["train"] + puzzle["test"]
    seen = set()
    backgrounds = set()
    tracers = set()
    train_info = []
    for index, pair in enumerate(pairs):
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        grid, output = pair["input"], pair["output"]
        for arr in (grid, output):
            assert isinstance(arr, np.ndarray) and arr.ndim == 2
            assert np.issubdtype(arr.dtype, np.integer)
            assert all(1 <= n <= 30 for n in arr.shape)
            assert np.all((arr >= 0) & (arr <= 9))
        assert grid.shape == output.shape and not np.array_equal(grid, output)
        key = _grid_key(grid)
        assert key not in seen and key not in _SOURCE_INPUTS
        seen.add(key)
        values, counts = np.unique(output, return_counts=True)
        assert len(values) == 2
        bg = int(values[np.argmax(counts)])
        trace = int(values[np.argmin(counts)])
        assert bg != trace
        backgrounds.add(bg)
        tracers.add(trace)
        info = _analyze_pair(grid, output, bg, trace)
        if index < len(puzzle["train"]):
            train_info.append(info)
    assert len(backgrounds) == len(tracers) == 1
    assert any(
        first["colors"][0] in second["colors"][1:]
        for i, first in enumerate(train_info)
        for j, second in enumerate(train_info) if i != j
    )
    assert all(info["top_tracer"] >= 2 and info["retained"] >= 1
               for info in train_info)
    assert max(info["top_tracer"] for info in train_info) >= 5
    assert max(info["retained"] for info in train_info) >= 8


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k])
        for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    backgrounds, tracers, shapes, layouts = set(), set(), set(), set()
    train_outside = test_outside = train_long = test_long = 0
    train_uniform = test_uniform = narrow_top = reused = sparse = 0
    measured = {"train": Counter(), "test": Counter()}
    within = Counter()
    positions = {n: set() for n in range(2, 7)}
    contrast_positions = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_records = {"train": [], "test": []}
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                grid, output = pair["input"], pair["output"]
                vals, counts = np.unique(output, return_counts=True)
                bg = int(vals[np.argmax(counts)])
                tracer = int(vals[np.argmin(counts)])
                info = _analyze_pair(grid, output, bg, tracer)
                backgrounds.add(bg); tracers.add(tracer)
                shapes.add(grid.shape)
                layouts.add((grid.shape, info["band_count"], info["bounds"]))
                positions[info["band_count"]].add((split, index))
                outside = info["outside_left"] or info["outside_right"]
                uniform = len(set(info["bounds"])) == 1
                top_start = next(y for y, row in enumerate(grid)
                                 if np.any((row != bg) & (row != tracer)))
                top_x = np.where(grid[top_start:info["cutoff"]] == tracer)[1]
                top_left, top_right = info["bounds"][0]
                top_outside = bool(np.any((top_x < top_left) | (top_x > top_right)))
                compact = max(grid.shape) <= 12
                large = max(grid.shape) >= 18
                long = info["max_run"] >= 6
                narrow = any(r-l >= top_right-top_left+2
                             for l,r in info["bounds"][1:])
                puzzle_records[split].append((grid.shape, info["band_count"],
                                               uniform, outside, long))
                for feature, present in (
                    ("compact", compact), ("large", large),
                    ("outside_left", info["outside_left"]),
                    ("outside_right", info["outside_right"]),
                    ("top_outside", top_outside), ("narrow_top", narrow),
                    ("edge_touch", any(l == 0 or r == grid.shape[1]-1
                                       for l,r in info["bounds"])),
                    ("repeat_fill", len(set(info["colors"])) < len(info["colors"])),
                    ("sparse", info["retained"] <= 6),
                    ("combined", not uniform and outside and long),
                ):
                    measured[split][feature] += int(present)
                if split == "train":
                    train_outside += outside; train_long += info["max_run"] >= 6
                    train_uniform += uniform
                    sparse += info["retained"] <= 6
                else:
                    test_outside += outside; test_long += info["max_run"] >= 6
                    test_uniform += uniform
                narrow_top += any(r-l > info["bounds"][0][1]-info["bounds"][0][0]
                                  for l, r in info["bounds"][1:])
                reused += len(set(info["colors"])) < len(info["colors"])
        together = puzzle_records["train"] + puzzle_records["test"]
        within["height"] += len({r[0][0] for r in together}) > 1
        within["width"] += len({r[0][1] for r in together}) > 1
        within["band_count"] += len({r[1] for r in together}) > 1
        within["uniform_and_staggered"] += len({r[2] for r in together}) > 1
        within["inside_and_outside"] += len({r[3] for r in together}) > 1
        within["long_and_short"] += len({r[4] for r in together}) > 1
        within["source_like"] += (all(r[2] for r in puzzle_records["train"])
                                  and any(not r[2] for r in puzzle_records["test"]))
        train_colors = []
        for pair in puzzle["train"]:
            values, counts = np.unique(pair["output"], return_counts=True)
            bg = int(values[np.argmax(counts)])
            tracer = int(values[np.argmin(counts)])
            train_colors.append(_analyze_pair(pair["input"], pair["output"],
                                              bg, tracer)["colors"])
        contrast_positions.update(
            (i, j) for i, colors_a in enumerate(train_colors)
            for j, colors_b in enumerate(train_colors)
            if i != j and colors_a[0] in colors_b[1:]
        )
    assert len(backgrounds) >= 6 and len(tracers) >= 6
    assert len(shapes) >= 35 and len(layouts) >= 100
    assert min(train_outside, test_outside, train_long, test_long,
               train_uniform, test_uniform, narrow_top, reused) >= 2
    for split in ("train", "test"):
        assert all(measured[split][name] >= 2 for name in
                   ("compact", "large", "outside_left", "outside_right",
                    "top_outside", "narrow_top", "edge_touch",
                    "repeat_fill", "combined"))
    assert measured["train"]["sparse"] >= 2
    assert all(within[name] >= 4 for name in
               ("height", "width", "band_count", "uniform_and_staggered",
                "inside_and_outside", "long_and_short", "source_like"))
    assert any(len(v) >= 3 for v in positions.values())
    assert len(contrast_positions) >= 8
    for args in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        a = generate_puzzle(101, *args)
        b = generate_puzzle(101, *args)
        assert _same_puzzle(a, b)
    assert _same_puzzle(generate_puzzle(7), generate_puzzle(7))
    assert not _same_puzzle(generate_puzzle(7), generate_puzzle(8))
    for args in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(True,1)):
        try:
            generate_puzzle(5, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid pair count accepted")
