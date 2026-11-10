"""Solved grids for the horizontal-divider endpoint rule."""
import numpy as np

_SOURCE_INPUTS = (
    ("0000000000", "0022200222", "0000000000", "0000222000", "4444444444", "0000000000", "0000000000", "0222000000", "0000222000", "0000000000"),
    ("003330000000", "000000033300", "666666666666", "000000000000", "000000000000", "000000333333", "000000000000", "000000000000", "000000000000"),
    ("000000000000", "000333330000", "000000000000", "000000000000", "000000000333", "111111111111", "000000000000", "033333300000", "000000000333"),
    ("0000440000", "0000000000", "3333333333", "0000000000", "0000444444", "0000000000"),
    ("00000000000", "06666666666", "55555555555", "06666666666", "00000000000"),
)
_SOURCE_KEYS = {(len(rows), len(rows[0]), "".join(rows)) for rows in _SOURCE_INPUTS}


def _key(grid):
    return grid.shape, grid.tobytes()


def _source_key(grid):
    return grid.shape[0], grid.shape[1], "".join(str(int(v)) for v in grid.flat)


def _checked_count(value, low, high, name):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _choose_counts(rng, num_train, num_test):
    _checked_count(num_train, 2, 10, "num_train")
    _checked_count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _colors(rng, base_background):
    background = base_background if rng.random() < 0.62 else int(rng.integers(10))
    choices = [c for c in range(10) if c != background]
    line, run = (int(c) for c in rng.choice(choices, size=2, replace=False))
    return background, line, run


def _count(rng, rows):
    cap = min(5, 3 * len(rows))
    weights = np.array([5, 5, 4, 3, 2][:cap], dtype=float)
    return int(rng.choice(np.arange(1, cap + 1), p=weights / weights.sum()))


def _groups(rng, rows, count):
    occupied = int(rng.integers((count + 2) // 3, min(count, len(rows)) + 1))
    chosen = [int(v) for v in rng.choice(rows, size=occupied, replace=False)]
    groups = {row: 1 for row in chosen}
    for _ in range(count - occupied):
        eligible = [row for row in chosen if groups[row] < 3]
        groups[int(rng.choice(eligible))] += 1
    return groups


def _lengths(rng, width, count, single, regime):
    lengths = [1 if i == single else 2 for i in range(count)]
    if count == 1:
        if single == 0:
            return [1]
        draw = rng.random()
        if draw < 0.12:
            return [width - 1]
        if draw < (0.39 if regime == "target_test" else 0.34) and width >= 9:
            return [int(rng.integers(7, width))]
        return [int(rng.integers(2, min(6, width - 1) + 1))]
    capacity = width - count + 1 - sum(lengths)
    assert capacity >= 0
    limit = capacity if rng.random() < 0.18 else min(capacity, 4 * count)
    extra = int(rng.integers(0, limit + 1))
    eligible = [i for i in range(count) if i != single]
    allocation = rng.multinomial(extra, rng.dirichlet(np.ones(len(eligible))))
    for i, amount in zip(eligible, allocation):
        lengths[i] += int(amount)
    return lengths


def _place_row(grid, row, lengths, color, rng):
    width = grid.shape[1]
    slack = width - sum(lengths) - len(lengths) + 1
    assert slack >= 0
    if len(lengths) == 1:
        draw = rng.random()
        start = 0 if draw < 0.18 else width - lengths[0] if draw < 0.36 else int(rng.integers(slack + 1))
        grid[row, start:start + lengths[0]] = color
        return
    spaces = rng.multinomial(slack, rng.dirichlet(np.ones(len(lengths) + 1)))
    col = int(spaces[0])
    for i, length in enumerate(lengths):
        grid[row, col:col + length] = color
        col += length + 1 + int(spaces[i + 1])


def _make_output(grid, divider, run_color, line_color):
    out = grid.copy()
    height, width = grid.shape
    for row in range(height):
        if row == divider:
            continue
        col = 0
        while col < width:
            if grid[row, col] != run_color:
                col += 1
                continue
            start = col
            while col < width and grid[row, col] == run_color:
                col += 1
            out[row, start if row < divider else col - 1] = line_color
    return out


def _generate_pair(rng, regime, base_background):
    assert regime in ("source_train", "target_test")
    compact_chance = 0.30 if regime == "target_test" else 0.24
    height = int(rng.integers(5, 7) if rng.random() < compact_chance else rng.integers(7, 20))
    width = int(rng.integers(8, 25))
    divider = int(rng.choice(sorted({1, 2, height - 3, height - 2}))) if rng.random() < 0.35 else int(rng.integers(1, height - 1))
    sides = (list(range(divider)), list(range(divider + 1, height)))
    blank_options = [r for rows in sides if len(rows) >= 2 for r in rows]
    blank = int(rng.choice(blank_options))
    available = [[r for r in rows if r != blank] for rows in sides]
    background, line_color, run_color = _colors(rng, base_background)
    grid = np.full((height, width), background, dtype=np.int8)
    grid[divider, :] = line_color
    for rows in available:
        count = _count(rng, rows)
        groups = _groups(rng, rows, count)
        single = int(rng.integers(count)) if count >= 2 and rng.random() < 0.11 else -1
        index = 0
        for row, n in groups.items():
            local_single = single - index if index <= single < index + n else -1
            _place_row(grid, row, _lengths(rng, width, n, local_single, regime), run_color, rng)
            index += n
    return grid, _make_output(grid, divider, run_color, line_color)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    # Spatial direction is fixed by the source. The background preference is a
    # shared visual motif; all output-relevant roles are visible in each input.
    base_background = int(rng.integers(10))
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, regime, count in (("train", "source_train", train_count), ("test", "target_test", test_count)):
        for _ in range(count):
            for _attempt in range(100):
                grid, answer = _generate_pair(rng, regime, base_background)
                key = _key(grid)
                if key not in seen and _source_key(grid) not in _SOURCE_KEYS:
                    seen.add(key)
                    puzzle[section].append({"input": grid, "output": answer})
                    break
            else:
                raise RuntimeError("unable to generate a unique pair")
    return puzzle


def _inspect_pair(pair):
    grid, output = pair["input"], pair["output"]
    height, width = grid.shape
    colors = {int(v) for v in np.unique(grid)}
    assert len(colors) == 3
    full_rows = [(r, int(grid[r, 0])) for r in range(height) if np.all(grid[r] == grid[r, 0])]
    candidates = [(r, c) for r, c in full_rows if np.count_nonzero(grid == c) == width]
    assert len(candidates) == 1
    divider, line_color = candidates[0]
    assert 0 < divider < height - 1
    blank_colors = {c for _, c in full_rows if c != line_color}
    assert len(blank_colors) == 1
    background = next(iter(blank_colors))
    run_color = next(iter(colors - {background, line_color}))
    assert not np.any(grid[np.arange(height) != divider] == line_color)
    expected, runs = grid.copy(), []
    for row in range(height):
        if row == divider:
            continue
        assert np.all((grid[row] == background) | (grid[row] == run_color))
        col, row_runs = 0, 0
        while col < width:
            if grid[row, col] != run_color:
                col += 1
                continue
            start = col
            while col < width and grid[row, col] == run_color:
                col += 1
            end = col - 1
            assert end - start + 1 < width
            runs.append((row, start, end))
            row_runs += 1
            expected[row, start if row < divider else end] = line_color
        assert row_runs <= 3
    upper = [run for run in runs if run[0] < divider]
    lower = [run for run in runs if run[0] > divider]
    assert 1 <= len(upper) <= 5 and 1 <= len(lower) <= 5
    assert any(end > start for _, start, end in upper)
    assert any(end > start for _, start, end in lower)
    assert np.array_equal(output, expected) and not np.array_equal(output, grid)
    return {"background": background, "line_color": line_color, "run_color": run_color,
            "divider": divider, "upper": upper, "lower": lower, "runs": runs}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            key = _key(pair["input"])
            pair_key = (key, _key(pair["output"]))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _source_key(pair["input"]) not in _SOURCE_KEYS
            _inspect_pair(pair)


def _same_puzzle(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def _predict_from_demonstrations(train_pairs, test_input):
    """Derive the two endpoint choices from demonstrations, then use input only."""
    directions = {"upper": set(), "lower": set()}
    for pair in train_pairs:
        info = _inspect_pair(pair)
        changes = {tuple(int(v) for v in cell)
                   for cell in np.argwhere(pair["input"] != pair["output"])}
        for side in ("upper", "lower"):
            for row, start, end in info[side]:
                if start < end:
                    if (row, start) in changes:
                        directions[side].add("left")
                    if (row, end) in changes:
                        directions[side].add("right")
    assert directions == {"upper": {"left"}, "lower": {"right"}}

    grid = test_input
    height, width = grid.shape
    full_rows = [(r, int(grid[r, 0])) for r in range(height)
                 if np.all(grid[r] == grid[r, 0])]
    divider, line_color = next((r, c) for r, c in full_rows
                               if np.count_nonzero(grid == c) == width)
    background = next(c for _, c in full_rows if c != line_color)
    run_color = next(iter({int(v) for v in np.unique(grid)} -
                          {background, line_color}))
    answer = grid.copy()
    for row in range(height):
        if row == divider:
            continue
        col = 0
        while col < width:
            if grid[row, col] != run_color:
                col += 1
                continue
            start = col
            while col < width and grid[row, col] == run_color:
                col += 1
            endpoint = start if "left" in directions["upper" if row < divider else "lower"] else col - 1
            answer[row, endpoint] = line_color
    return answer


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, palettes, puzzles = set(), set(), set(), set(), set(), set()
    metrics = {k: 0 for k in ("lower_more_train", "upper_more_test", "upper_group", "lower_group", "test_group",
                              "triple", "train_long", "test_short", "near_full", "left", "right", "compact",
                              "outer", "boundary", "far", "adjacent", "test_gap", "single", "fresh_line",
                              "background_change", "non_swap", "count_change")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzles.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        infos = {s: [_inspect_pair(p) for p in puzzle[s]] for s in ("train", "test")}
        train_lines = {i["line_color"] for i in infos["train"]}
        train_roles = {(i["line_color"], i["run_color"]) for i in infos["train"]}
        all_infos = infos["train"] + infos["test"]
        metrics["fresh_line"] += any(i["line_color"] not in train_lines for i in infos["test"])
        metrics["background_change"] += len({i["background"] for i in all_infos}) > 1
        metrics["non_swap"] += any((i["run_color"], i["line_color"]) not in train_roles for i in infos["test"])
        metrics["count_change"] += len({(len(i["upper"]), len(i["lower"])) for i in all_infos}) > 1
        for pair in puzzle["test"]:
            predicted = _predict_from_demonstrations(puzzle["train"], pair["input"].copy())
            assert np.array_equal(predicted, pair["output"])
        for section in ("train", "test"):
            for pair, info in zip(puzzle[section], infos[section]):
                grid, out = pair["input"], pair["output"]
                h, w = grid.shape
                shapes.add((h, w))
                layouts.add((h, w, info["divider"], tuple(info["runs"])))
                palettes.add((info["background"], info["line_color"], info["run_color"]))
                upper, lower, runs = info["upper"], info["lower"], info["runs"]
                row_counts = [sum(x[0] == r for x in runs) for r, _, _ in runs]
                metrics["lower_more_train"] += section == "train" and len(lower) > len(upper)
                metrics["upper_more_test"] += section == "test" and len(upper) > len(lower)
                metrics["upper_group"] += any(sum(x[0] == r for x in upper) >= 2 for r, _, _ in upper)
                metrics["lower_group"] += any(sum(x[0] == r for x in lower) >= 2 for r, _, _ in lower)
                metrics["test_group"] += section == "test" and max(row_counts) >= 2
                metrics["triple"] += max(row_counts) >= 3
                metrics["train_long"] += section == "train" and any(e-s+1 >= 7 for _, s, e in runs)
                metrics["test_short"] += section == "test" and all(e-s+1 < 7 for _, s, e in runs)
                metrics["near_full"] += any(e-s+1 == w-1 for _, s, e in runs)
                metrics["left"] += any(s == 0 for _, s, _ in runs)
                metrics["right"] += any(e == w-1 for _, _, e in runs)
                metrics["compact"] += h <= 6
                metrics["outer"] += any(r in (0, h-1) for r, _, _ in runs)
                metrics["boundary"] += min(info["divider"], h-1-info["divider"]) <= 2
                metrics["far"] += any(abs(r-info["divider"]) >= 3 for r, _, _ in runs)
                metrics["adjacent"] += any(abs(r-info["divider"]) == 1 for r, _, _ in runs)
                metrics["test_gap"] += section == "test" and all(abs(r-info["divider"]) > 1 for r, _, _ in runs)
                metrics["single"] += any(s == e for _, s, e in runs)
                changed = {tuple(int(v) for v in cell) for cell in np.argwhere(grid != out)}
                endpoints = {(r, s if r < info["divider"] else e) for r, s, e in runs}
                assert changed == endpoints and all(int(out[r, c]) == info["line_color"] for r, c in changed)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzles) >= 40 and len(shapes) >= 30 and len(layouts) >= 120 and len(palettes) >= 60
    bounds = {"lower_more_train": 8, "upper_more_test": 3, "upper_group": 20, "lower_group": 20,
              "test_group": 5, "triple": 3, "train_long": 15, "test_short": 3, "near_full": 5,
              "left": 20, "right": 20, "compact": 15, "outer": 20, "boundary": 25,
              "far": 25, "adjacent": 25, "test_gap": 1, "single": 3, "fresh_line": 10,
              "background_change": 15, "non_swap": 20, "count_change": 30}
    assert all(metrics[k] >= v for k, v in bounds.items()), metrics
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(731, train, test)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(731, train, test))
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4), (True, None), (None, 1.5)):
        try:
            generate_puzzle(731, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    damaged = generate_puzzle(997, 2, 1)
    damaged["test"][0]["output"] = damaged["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an incorrect output")
