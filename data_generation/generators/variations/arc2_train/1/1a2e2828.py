import numpy as np


_SOURCE_PAIRS = {
    (
        (
            "000033005000",
            "222233225222",
            "000033005000",
            "111111111111",
            "111111111111",
            "000033005000",
            "444444445444",
            "444444445444",
            "000033005000",
            "000033005000",
            "000033005000",
        ),
        1,
    ),
    (
        (
            "00044000800",
            "00044000800",
            "33344333833",
            "00044000800",
            "00044000800",
            "66666666866",
            "66666666866",
            "00044000800",
            "00044000800",
        ),
        8,
    ),
    (
        (
            "02206008800",
            "12216118811",
            "12216118811",
            "12216118811",
            "02206008800",
            "02206008800",
            "44446444444",
            "44446444444",
            "02206008800",
            "02206008800",
            "02206008800",
        ),
        6,
    ),
    (
        (
            "010",
            "333",
            "010",
        ),
        3,
    ),
    (
        (
            "022000080000",
            "022000080000",
            "322333383333",
            "322333383333",
            "022000080000",
            "666666666666",
            "022000080000",
            "022000080000",
        ),
        6,
    ),
    (
        (
            "0003300007000",
            "0003300007000",
            "1113311117111",
            "1113311117111",
            "0003300007000",
            "0003300007000",
            "6666666667666",
            "0003300007000",
            "0003300007000",
            "0003300007000",
            "0003300007000",
        ),
        7,
    ),
}


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _normalize_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed: int, rng: np.random.Generator, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train_count = _normalize_count(num_train, "num_train")
    test_count = _normalize_count(num_test, "num_test")
    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        train_count = test_count + 1 + (seed // 4) % (10 - test_count)
    elif train_count is None:
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif test_count is None:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


def _sample_puzzle_parameters(rng: np.random.Generator, train_count: int) -> dict:
    backgrounds = [int(rng.integers(0, 10))]
    if int(rng.integers(0, 2)):
        backgrounds.append(int(rng.choice([c for c in range(10) if c != backgrounds[0]])))
    available = [c for c in range(10) if c not in backgrounds]
    rng.shuffle(available)
    ceiling = min(7 if len(backgrounds) == 1 else 6, len(available) - 2)
    if train_count == 2:
        ceiling = min(ceiling, 5)
    palette_size = int(rng.integers(5, ceiling + 1))
    return {"backgrounds": tuple(backgrounds), "train_palette": tuple(available[:palette_size])}


def _sample_layout(rng: np.random.Generator, count: int, regime: str) -> tuple[int, tuple[tuple[int, int], ...]]:
    # The last margin remains visible, so the lower-right corner exposes background.
    minimum = max(3, 2 * count)
    category = int(rng.choice(3, p=(0.42, 0.37, 0.21) if regime == "source_train" else (0.24, 0.37, 0.39)))
    lo, hi = ((minimum, min(12, 30)), (max(minimum, 9), 21), (max(minimum, 18), 30))[category]
    if lo > hi:
        lo, hi = minimum, 30
    dimension = int(rng.integers(lo, hi + 1))
    start_gap = int(rng.integers(0, 4) != 0) if dimension > 2 * count else 0
    minimum_gaps = count + start_gap  # one between neighbors, one after the last
    minimum_background = max(minimum_gaps, (dimension + 5) // 6)
    budget = dimension - minimum_background
    thickness = [int(rng.integers(1, 8)) for _ in range(count)]
    while sum(thickness) > budget:
        reducible = [i for i, value in enumerate(thickness) if value > 1]
        thickness[int(rng.choice(reducible))] -= 1
    gaps = [start_gap] + [1] * count
    remaining = dimension - sum(thickness) - sum(gaps)
    for _ in range(remaining):
        gaps[int(rng.integers(0, count + 1))] += 1
    ranges = []
    position = gaps[0]
    for index, size in enumerate(thickness):
        ranges.append((position, position + size))
        position += size + gaps[index + 1]
    _require(position == dimension, "layout accounting error")
    return dimension, tuple(ranges)


def _make_depth_order(rng: np.random.Generator, horizontal_count: int, vertical_count: int, top_orientation: str) -> list[int]:
    horizontal = list(range(horizontal_count))
    vertical = list(range(horizontal_count, horizontal_count + vertical_count))
    rng.shuffle(horizontal)
    rng.shuffle(vertical)
    first = horizontal.pop() if top_orientation == "h" else vertical.pop()
    second = vertical.pop() if top_orientation == "h" else horizontal.pop()
    remainder = horizontal + vertical
    rng.shuffle(remainder)
    return [first, second] + remainder


def _generate_pair(rng: np.random.Generator, regime: str, background: int, colors: tuple[int, ...], winner: int, horizontal_count: int, vertical_count: int, top_orientation: str | None) -> tuple[np.ndarray, np.ndarray]:
    total = horizontal_count + vertical_count
    _require(len(colors) == total and len(set(colors)) == total, "invalid stripe palette")
    _require(background not in colors and winner in colors, "invalid color roles")
    if top_orientation is None:
        top_orientation = "h" if int(rng.integers(0, 2)) == 0 else "v"
    order = _make_depth_order(rng, horizontal_count, vertical_count, top_orientation)
    remaining = [c for c in colors if c != winner]
    rng.shuffle(remaining)
    node_colors = {node: color for node, color in zip(order, [winner] + remaining)}
    height, horizontal_ranges = _sample_layout(rng, horizontal_count, regime)
    width, vertical_ranges = _sample_layout(rng, vertical_count, regime)
    grid = np.full((height, width), background, dtype=np.int8)
    for node in reversed(order):
        if node < horizontal_count:
            start, end = horizontal_ranges[node]
            grid[start:end, :] = node_colors[node]
        else:
            start, end = vertical_ranges[node - horizontal_count]
            grid[:, start:end] = node_colors[node]
    return grid, np.asarray([[winner]], dtype=np.int8)


def _runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    indices = np.flatnonzero(mask)
    if not indices.size:
        return ()
    result = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        index = int(raw)
        if index != previous + 1:
            result.append((start, previous + 1))
            start = index
        previous = index
    result.append((start, previous + 1))
    return tuple(result)


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(int(v) for v in grid.shape), tuple(int(v) for v in grid.flat))


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    if output_grid.shape != (1, 1):
        return False
    rows = tuple("".join(str(int(v)) for v in row) for row in input_grid)
    return (rows, int(output_grid[0, 0])) in _SOURCE_PAIRS


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    _require(output_grid.shape == (1, 1), "output must be 1 by 1")
    height, width = input_grid.shape
    background = int(input_grid[-1, -1])
    horizontal_runs = _runs(np.all(input_grid != background, axis=1))
    vertical_runs = _runs(np.all(input_grid != background, axis=0))
    _require(horizontal_runs and vertical_runs, "both stripe orientations are required")
    outside_h = [r for r in range(height) if not any(a <= r < b for a, b in horizontal_runs)]
    outside_v = [c for c in range(width) if not any(a <= c < b for a, b in vertical_runs)]
    _require(outside_h and outside_v, "background must remain exposed")
    hcolors = []
    for start, end in horizontal_runs:
        cells = input_grid[start:end, :][:, outside_v]
        color = int(cells.flat[0])
        _require(color != background and np.all(cells == color), "broken horizontal stripe")
        hcolors.append(color)
    vcolors = []
    for start, end in vertical_runs:
        cells = input_grid[outside_h, :][:, start:end]
        color = int(cells.flat[0])
        _require(color != background and np.all(cells == color), "broken vertical stripe")
        vcolors.append(color)
    colors = hcolors + vcolors
    _require(len(set(colors)) == len(colors), "stripe colors must be distinct")
    hmask = np.full(height, -1, dtype=np.int16)
    vmask = np.full(width, -1, dtype=np.int16)
    for i, (a, b) in enumerate(horizontal_runs):
        hmask[a:b] = i
    for i, (a, b) in enumerate(vertical_runs):
        vmask[a:b] = i
    adjacency = [set() for _ in colors]
    indegree = [0] * len(colors)
    for r in range(height):
        for c in range(width):
            hi, vi, value = int(hmask[r]), int(vmask[c]), int(input_grid[r, c])
            if hi < 0 and vi < 0:
                _require(value == background, "unexplained background cell")
            elif hi >= 0 and vi < 0:
                _require(value == hcolors[hi], "horizontal band is incomplete")
            elif hi < 0 and vi >= 0:
                _require(value == vcolors[vi], "vertical band is incomplete")
    for hi, (ra, rb) in enumerate(horizontal_runs):
        for vi, (ca, cb) in enumerate(vertical_runs):
            crossing = input_grid[ra:rb, ca:cb]
            visible = int(crossing.flat[0])
            _require(np.all(crossing == visible), "nonuniform crossing")
            if visible == hcolors[hi]:
                winner, loser = hi, len(hcolors) + vi
            elif visible == vcolors[vi]:
                winner, loser = len(hcolors) + vi, hi
            else:
                raise AssertionError("crossing uses neither stripe color")
            adjacency[winner].add(loser)
            indegree[loser] += 1
    work = indegree.copy()
    remaining = set(range(len(colors)))
    unique_order = []
    complete_order = True
    while remaining:
        sources = [node for node in remaining if work[node] == 0]
        _require(sources, "cyclic crossing relations")
        if len(sources) != 1:
            complete_order = False
        node = sources[0]
        remaining.remove(node)
        unique_order.append(node)
        for below in adjacency[node]:
            work[below] -= 1
    sources = [i for i, degree in enumerate(indegree) if degree == 0]
    _require(len(sources) == 1, "uppermost stripe is ambiguous")
    top = sources[0]
    reachable = {top}
    frontier = [top]
    while frontier:
        for below in adjacency[frontier.pop()]:
            if below not in reachable:
                reachable.add(below)
                frontier.append(below)
    _require(len(reachable) == len(colors), "top stripe does not dominate all others")
    _require(int(output_grid[0, 0]) == colors[top], "incorrect output color")
    _require(set(int(v) for v in np.unique(input_grid)) == {background, *colors}, "extra input color")
    return {"background": background, "height": height, "width": width,
            "horizontal_count": len(hcolors), "vertical_count": len(vcolors),
            "horizontal_runs": horizontal_runs, "vertical_runs": vertical_runs,
            "thicknesses": tuple(b-a for a,b in horizontal_runs + vertical_runs),
            "band_colors": tuple(colors), "top_color": colors[top],
            "top_orientation": "h" if top < len(hcolors) else "v",
            "complete_order": complete_order,
            "edge_band": horizontal_runs[0][0] == 0 or vertical_runs[0][0] == 0}


def _infer_uninterrupted(input_grid: np.ndarray) -> int:
    background = int(input_grid[-1, -1])
    full = set()
    for row in input_grid:
        if np.all(row == row[0]) and int(row[0]) != background:
            full.add(int(row[0]))
    for col in input_grid.T:
        if np.all(col == col[0]) and int(col[0]) != background:
            full.add(int(col[0]))
    _require(len(full) == 1, "input has no unique uninterrupted stripe")
    return full.pop()


def _validate_grid(grid: np.ndarray, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, f"{label} size outside 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must be integer")
    _require(0 <= int(grid.min()) and int(grid.max()) <= 9, f"{label} has illegal color")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp, f"{split}[{index}] input")
            _validate_grid(out, f"{split}[{index}] output")
            ikey, okey = _grid_key(inp), _grid_key(out)
            _require(ikey not in seen_inputs and (ikey, okey) not in seen_pairs, "duplicate pair")
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            _require(not _is_source_pair(inp, out), "exact source copy")
            analysis = _analyze_pair(inp, out)
            _require(_infer_uninterrupted(inp) == int(out[0, 0]), "direct visible-stripe prediction disagrees")
            analyses[split].append(analysis)
    train = analyses["train"]
    _require(any(a["horizontal_count"] == a["vertical_count"] == 1 for a in train), "training lacks simple crossing")
    _require(any(a["horizontal_count"] + a["vertical_count"] >= 3 for a in train), "training lacks multi-stripe crossing")
    _require({a["top_orientation"] for a in train} == {"h", "v"}, "training lacks both winner orientations")
    _require(len({a["top_color"] for a in train}) >= 2, "training has fixed answer color")
    train_winners = {a["top_color"] for a in train}
    train_losers = {color for a in train for color in a["band_colors"] if color != a["top_color"]}
    _require(bool(train_winners & train_losers), "training lacks a color in both depth roles")
    # The only convention is visible over-under selection, demonstrated in training.
    # Both splits obey the same complete-span geometry and unique-top requirement.


def _choose_counts(rng: np.random.Generator, maximum: int, regime: str, minimum: int = 2) -> tuple[int, int]:
    if regime == "source_train":
        total = int(rng.integers(2, maximum + 1))
    else:
        choices = np.arange(minimum, maximum + 1)
        weights = np.asarray([1.8 if n in (2, 4) else (1.2 if n >= 7 else 1.0) for n in choices], dtype=float)
        total = int(rng.choice(choices, p=weights / weights.sum()))
    if total == 4 and int(rng.integers(0, 3)) != 0:
        return 2, 2
    horizontal = int(rng.integers(1, total))
    return horizontal, total - horizontal


def _test_colors(rng: np.random.Generator, mode: int, background: int, train_input_colors: set[int], train_stripe_colors: set[int]) -> tuple[tuple[int, ...], int, int, int]:
    familiar = [c for c in sorted(train_input_colors) if c != background]
    familiar_winners = [c for c in sorted(train_stripe_colors) if c != background]
    novel = [c for c in range(10) if c not in train_input_colors and c != background]
    _require(len(familiar) >= 4 and len(novel) >= 2, "insufficient palette diversity")
    maximum = len(familiar) if mode == 0 else min(9, len(familiar) + (1 if mode in (1, 2) else len(novel)))
    if mode == 3:
        minimum = 3
    else:
        minimum = 2
    maximum = max(minimum, maximum)
    horizontal, vertical = _choose_counts(rng, maximum, "target_test", minimum)
    total = horizontal + vertical
    if mode == 0:
        chosen = list(rng.choice(familiar, size=total, replace=False))
        winner = int(rng.choice([c for c in chosen if c in familiar_winners]))
    elif mode == 1:
        winner = int(rng.choice(novel))
        chosen = [winner] + [int(c) for c in rng.choice(familiar, size=total - 1, replace=False)]
    elif mode == 2:
        winner = int(rng.choice(familiar_winners))
        chosen = [winner, int(rng.choice(novel))]
        choices = [c for c in familiar if c != winner]
        chosen += [int(c) for c in rng.choice(choices, size=total - 2, replace=False)]
    else:
        novel_count = int(rng.integers(max(2, total - len(familiar)), min(len(novel), total - 1) + 1))
        new_colors = [int(c) for c in rng.choice(novel, size=novel_count, replace=False)]
        old_colors = [int(c) for c in rng.choice(familiar, size=total - novel_count, replace=False)]
        chosen = new_colors + old_colors
        winner = int(rng.choice(new_colors if int(rng.integers(0, 2)) else [c for c in old_colors if c in familiar_winners] or old_colors))
    _require(len(chosen) == total and len(set(chosen)) == total, "test palette construction failed")
    return tuple(chosen), winner, horizontal, vertical


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng, train_count)
    backgrounds = parameters["backgrounds"]
    palette = parameters["train_palette"]
    winner_colors = list(rng.choice(palette, size=2, replace=False))
    first_top = "h" if int(rng.integers(0, 2)) == 0 else "v"
    # One simple and one multi-stripe example, shuffled among other demonstrations.
    demonstrations = [(1, 1, first_top, int(winner_colors[0]), int(winner_colors[1]))]
    demonstrations.append((2, 2, "v" if first_top == "h" else "h", int(winner_colors[1]), int(winner_colors[0])))
    for _ in range(train_count - 2):
        horizontal, vertical = _choose_counts(rng, len(palette), "source_train")
        top = "h" if int(rng.integers(0, 2)) == 0 else "v"
        winner = int(rng.choice(palette))
        demonstrations.append((horizontal, vertical, top, winner, None))
    rng.shuffle(demonstrations)
    if len(backgrounds) == 1:
        train_backgrounds = [backgrounds[0]] * train_count
    else:
        train_backgrounds = [backgrounds[i % 2] for i in range(train_count)]
        rng.shuffle(train_backgrounds)
    puzzle = {"train": [], "test": []}
    seen = set()
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=train_count + test_count, dtype=np.uint64)
    for index, (hc, vc, top_orientation, winner, required_loser) in enumerate(demonstrations):
        pair_rng = np.random.default_rng(pair_seeds[index])
        colors = [winner]
        if required_loser is not None:
            colors.append(required_loser)
        pool = [c for c in palette if c not in colors]
        colors += [int(c) for c in pair_rng.choice(pool, size=hc + vc - len(colors), replace=False)]
        for _ in range(20):
            inp, out = _generate_pair(pair_rng, "source_train", train_backgrounds[index], tuple(colors), winner, hc, vc, top_orientation)
            key = _grid_key(inp)
            if key not in seen and not _is_source_pair(inp, out):
                break
        else:
            raise ValueError("could not make distinct training pair")
        seen.add(key)
        puzzle["train"].append({"input": inp, "output": out})
    train_input_colors = {int(v) for pair in puzzle["train"] for v in np.unique(pair["input"])}
    train_stripe_colors = {int(v) for pair in puzzle["train"] for v in np.unique(pair["input"])} - set(backgrounds)
    modes = list(rng.permutation(4))[:test_count]
    for index, mode in enumerate(modes):
        pair_rng = np.random.default_rng(pair_seeds[train_count + index])
        background = int(pair_rng.choice(backgrounds))
        colors, winner, hc, vc = _test_colors(pair_rng, int(mode), background, train_input_colors, train_stripe_colors)
        for _ in range(20):
            inp, out = _generate_pair(pair_rng, "target_test", background, colors, winner, hc, vc, None)
            key = _grid_key(inp)
            if key not in seen and not _is_source_pair(inp, out):
                break
        else:
            raise ValueError("could not make distinct test pair")
        seen.add(key)
        puzzle["test"].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    _require(len({_puzzle_signature(p) for p in puzzles}) >= 40, "whole puzzle diversity")
    _require(_puzzle_signature(puzzles[17]) == _puzzle_signature(generate_puzzle(17)), "nondeterministic automatic call")
    _require(_puzzle_signature(generate_puzzle(903, 6, 3)) == _puzzle_signature(generate_puzzle(903, 6, 3)), "nondeterministic explicit call")
    _require(_puzzle_signature(generate_puzzle(903, 6, 3)) != _puzzle_signature(generate_puzzle(904, 6, 3)), "seed has no effect")
    for seed, train_count, test_count in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1101, None, 1), (1102, None, 4), (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "explicit training count")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "explicit test count")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1200, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
    observations = []
    mode_counts = {"familiar": 0, "novel_winner": 0,
                   "familiar_winner_novel_loser": 0, "multi_novel": 0}
    mixed_background_puzzles = 0
    smaller_tests = 0
    wider_tests = 0
    simple_positions = set()
    layout_signatures = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_colors = {int(v) for pair in puzzle["train"] for v in np.unique(pair["input"])}
        puzzle_features = []
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                a = _analyze_pair(pair["input"], pair["output"])
                a["split"] = split
                a["novel"] = len(set(a["band_colors"]) - train_colors) if split == "test" else 0
                a["familiar_winner"] = a["top_color"] in train_colors
                puzzle_features.append(a)
                observations.append(a)
                _require(_infer_uninterrupted(pair["input"]) == int(pair["output"][0, 0]), "independent prediction")
                layout_signatures.add((a["height"], a["width"], a["horizontal_runs"],
                                       a["vertical_runs"], a["top_orientation"]))
                if split == "train" and a["horizontal_count"] == a["vertical_count"] == 1:
                    simple_positions.add(index)
                if split == "test":
                    if a["novel"] == 0:
                        mode = "familiar"
                    elif a["novel"] == 1 and not a["familiar_winner"]:
                        mode = "novel_winner"
                    elif a["novel"] == 1:
                        mode = "familiar_winner_novel_loser"
                    else:
                        mode = "multi_novel"
                    mode_counts[mode] += 1
        _require(len({a["top_orientation"] for a in puzzle_features if a["split"] == "train"}) == 2, "training orientation coverage")
        if len({a["background"] for a in puzzle_features}) > 1:
            mixed_background_puzzles += 1
        train_features = [a for a in puzzle_features if a["split"] == "train"]
        for a in (item for item in puzzle_features if item["split"] == "test"):
            smaller_tests += any(a["height"] * a["width"] < t["height"] * t["width"] for t in train_features)
            wider_tests += a["width"] > max(t["width"] for t in train_features)
    tests = [a for a in observations if a["split"] == "test"]
    _require(all(value >= 10 for value in mode_counts.values()), "test palette modes lack coverage")
    _require(10 <= mixed_background_puzzles <= 32, "constant/mixed background modes lack coverage")
    _require(3 <= len(simple_positions) and min(simple_positions) == 0, "fixed training lesson placement")
    _require(smaller_tests >= 10 and wider_tests >= 10, "test scale is fixed relative to training")
    _require(len(layout_signatures) >= 180, "layout diversity is too low")
    _require({a["background"] for a in observations} == set(range(10)), "background palette lacks coverage")
    _require({a["top_orientation"] for a in tests} == {"h", "v"}, "test winner orientation lacks coverage")
    _require(sum(not a["complete_order"] for a in tests) >= 15, "few partially ordered tests")
    _require(any(a["horizontal_count"] == a["vertical_count"] == 1 for a in tests), "no simple test")
    _require(any(a["horizontal_count"] == a["vertical_count"] == 2 for a in tests), "no familiar four-stripe test")
    _require(any(min(a["horizontal_count"], a["vertical_count"]) == 1 and max(a["horizontal_count"], a["vertical_count"]) >= 4 for a in tests), "no one-against-many test")
    _require(any(a["horizontal_count"] + a["vertical_count"] >= 8 for a in tests), "no large stripe count")
    _require(any(a["height"] == 30 or a["width"] == 30 for a in tests), "no full-size test canvas")
    _require(any(max(a["thicknesses"]) >= 5 for a in tests), "no wide test stripe")
    _require(any(min(a["thicknesses"]) == 1 for a in tests), "no thin test stripe")
    _require(any(a["edge_band"] for a in tests), "no edge test stripe")
    _require(any(a["background"] == 0 for a in tests), "no black test background")
    _require(any(a["novel"] >= 2 and not a["complete_order"] and
                 a["horizontal_count"] + a["vertical_count"] >= 7 and
                 max(a["thicknesses"]) >= 4 and a["edge_band"] for a in tests),
             "compatible variation axes never combine")
    nine_band = generate_puzzle(45)["test"]
    _require(any(len(np.unique(pair["input"])) == 10 for pair in nine_band),
             "nine-stripe construction is not exercised")
