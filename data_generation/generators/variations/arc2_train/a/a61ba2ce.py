"""Generate scattered L-trominoes assembled around a four-cell center."""

import numpy as np


_ORIENTATIONS = (0, 1, 2, 3)
_DESTINATION = {3: (0, 0), 2: (0, 2), 1: (2, 0), 0: (2, 2)}


def _rows(*rows: str) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = (
    (
        _rows("0000000000000", "0000000088000", "0000000008000",
              "0011000000000", "0010000000000", "0000000020000",
              "0000000220000", "0000000000000", "0000000000000",
              "0000400000000", "0000440000000", "0000000000000",
              "0000000000000"),
        _rows("1188", "1008", "4002", "4422"),
    ),
    (
        _rows("0000000000000", "0000008800000", "0000008000000",
              "0220000000000", "0020000000000", "0000000000000",
              "0000000000000", "0000000010000", "0000000110000",
              "0003000000000", "0003300000000", "0000000000000",
              "0000000000000"),
        _rows("8822", "8002", "3001", "3311"),
    ),
    (
        _rows("0000000000000", "0000000000000", "0006000000880",
              "0066000000080", "0000000000000", "0000000000000",
              "0000001000000",
              "0000001100000", "0000000000000", "0033000000000",
              "0030000000000", "0000000000000", "0000000000000"),
        _rows("3388", "3008", "1006", "1166"),
    ),
)


def _resolve_counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return int(num_train), int(num_test)
    if num_train is None and num_test is None:
        # Short demonstrations are common, while the full legal range remains used.
        trains = np.arange(2, 11)
        weights = np.array([20, 19, 17, 12, 10, 8, 6, 5, 3], dtype=float)
        train = int(rng.choice(trains, p=weights / weights.sum()))
        return train, int(rng.integers(1, min(4, train - 1) + 1))
    if num_train is None:
        test = int(num_test)
        choices = np.arange(max(2, test + 1), 11)
        weights = 1.0 / choices
        return int(rng.choice(choices, p=weights / weights.sum())), test
    train = int(num_train)
    return train, int(rng.integers(1, min(4, train - 1) + 1))


def _sample_dimensions(rng):
    kind = int(rng.choice(5, p=[0.12, 0.25, 0.22, 0.22, 0.19]))
    if kind == 0:
        return (13, 13)
    if kind == 1:
        side = int(rng.integers(7, 23))
        return (side, side)
    if kind in (2, 3):
        short = int(rng.integers(7, 15))
        long = int(rng.integers(short + 3, 27))
        return (short, long) if kind == 2 else (long, short)
    return (int(rng.integers(7, 25)), int(rng.integers(7, 25)))


def _boxes_clear(a, b):
    # One background row or column suffices, including for equal colors.
    return abs(a[0] - b[0]) >= 3 or abs(a[1] - b[1]) >= 3


def _sample_positions(rng, height, width, style):
    all_positions = [(r, c) for r in range(height - 1)
                     for c in range(width - 1)]
    if style == "compact":
        span_h, span_w = min(height, 7), min(width, 7)
        top = int(rng.integers(0, height - span_h + 1))
        left = int(rng.integers(0, width - span_w + 1))
        preferred = [(r, c) for r, c in all_positions
                     if top <= r <= top + span_h - 2
                     and left <= c <= left + span_w - 2]
    elif style == "margin":
        inset_h = min(3, max(0, (height - 6) // 2))
        inset_w = min(3, max(0, (width - 6) // 2))
        preferred = [(r, c) for r, c in all_positions
                     if inset_h <= r <= height - 2 - inset_h
                     and inset_w <= c <= width - 2 - inset_w]
    elif style == "edge":
        preferred = [(r, c) for r, c in all_positions
                     if min(r, c, height - 2 - r, width - 2 - c) <= 1]
    else:
        preferred = all_positions
    for pool in (preferred, all_positions):
        for _ in range(35):
            chosen = []
            for index in rng.permutation(len(pool)):
                point = pool[int(index)]
                if all(_boxes_clear(point, old) for old in chosen):
                    chosen.append(point)
                    if len(chosen) == 4:
                        return chosen
    return [(0, 0), (0, width - 2), (height - 2, 0),
            (height - 2, width - 2)]


def _scatter_input(rng, shape, background, colors, style):
    height, width = shape
    grid = np.full(shape, background, dtype=np.int8)
    positions = _sample_positions(rng, height, width, style)
    orientation_positions = [None] * 4
    for orientation, position in zip(rng.permutation(4), positions):
        orientation = int(orientation)
        row, col = position
        orientation_positions[orientation] = position
        for k in _ORIENTATIONS:
            if k != orientation:
                grid[row + k // 2, col + k % 2] = colors[orientation]
    return grid, tuple(orientation_positions)


def _assemble_output(background, colors):
    output = np.full((4, 4), background, dtype=np.int8)
    for orientation, color in enumerate(colors):
        row, col = _DESTINATION[orientation]
        for k in _ORIENTATIONS:
            if k != orientation:
                output[row + k // 2, col + k % 2] = color
    return output


def _signature(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _make_pair(rng, shape, background, colors, style, seen, avoid_order=None):
    output = _assemble_output(background, colors)
    for _ in range(100):
        inp, positions = _scatter_input(rng, shape, background, colors, style)
        order = tuple(sorted(_ORIENTATIONS, key=lambda i: positions[i]))
        if order == avoid_order or _signature(inp) in seen:
            continue
        if any(np.array_equal(inp, a) and np.array_equal(output, b)
               for a, b in _SOURCE_PAIRS):
            continue
        return {"input": inp, "output": output.copy()}, order
    raise RuntimeError("could not sample a distinct pair")


def _analyze_input(grid):
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    maximum = int(counts.max())
    assert int(np.count_nonzero(counts == maximum)) == 1
    background = int(np.argmax(counts))
    mask = grid != background
    assert int(mask.sum()) == 12
    seen = np.zeros(grid.shape, dtype=bool)
    colors = [None] * 4
    positions = [None] * 4
    components = 0
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]
                            and mask[rr, cc] and not seen[rr, cc]):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            components += 1
            assert len(cells) == 3
            assert len({int(grid[r, c]) for r, c in cells}) == 1
            top = min(r for r, _ in cells)
            left = min(c for _, c in cells)
            assert max(r for r, _ in cells) == top + 1
            assert max(c for _, c in cells) == left + 1
            occupied = {(r - top, c - left) for r, c in cells}
            missing = {(0, 0), (0, 1), (1, 0), (1, 1)} - occupied
            assert len(missing) == 1
            mr, mc = missing.pop()
            orientation = 2 * mr + mc
            assert colors[orientation] is None
            colors[orientation] = int(grid[row, col])
            positions[orientation] = (top, left)
    assert components == 4 and all(color is not None for color in colors)
    assert all(_boxes_clear(positions[i], positions[j])
               for i in range(4) for j in range(i + 1, 4))
    return {"background": background, "colors": tuple(colors),
            "positions": tuple(positions)}


def _pick_background(rng, base, core, variable):
    if not variable:
        return base
    return int(rng.choice([base, core[0], core[1], core[2]],
                          p=[0.42, 0.28, 0.15, 0.15]))


def _sample_palette(rng, background, familiar, novelty, color_count):
    available = [c for c in range(10) if c != background]
    familiar_pool = [c for c in available if c in familiar]
    new_pool = [c for c in available if c not in familiar]
    if novelty == "familiar":
        new_count = 0
    elif novelty == "one":
        new_count = 1
    else:
        new_count = min(color_count, len(new_pool),
                        int(rng.integers(2, 4)))
    new_count = min(new_count, len(new_pool))
    new_count = max(new_count, color_count - len(familiar_pool))
    selected = [int(v) for v in rng.choice(new_pool, new_count, replace=False)]
    selected += [int(v) for v in rng.choice(
        familiar_pool, color_count - new_count, replace=False)]
    return tuple(selected)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    digits = [int(x) for x in rng.permutation(10)]
    base, core = digits[0], digits[1:6]
    variable_background = bool(rng.integers(0, 2))
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                             size=train_count + test_count)
    pair_rngs = [np.random.default_rng(int(s)) for s in pair_seeds]
    puzzle = {"train": [], "test": []}
    seen = set()
    train_piece_colors = set()
    first_mapping = None
    first_order = None
    train_shapes = []

    for index in range(train_count + test_count):
        pair_rng = pair_rngs[index]
        training = index < train_count
        if index == 0:
            background = base
            mapping = tuple(int(x) for x in pair_rng.permutation(core[:4]))
        elif index == 1:
            background = core[0] if variable_background else base
            selected = ([base] if variable_background else [core[0]]) + core[1:4]
            pivot = core[1]
            old_orientation = first_mapping.index(pivot)
            alternatives = [i for i in _ORIENTATIONS if i != old_orientation]
            new_orientation = int(pair_rng.choice(alternatives))
            remaining = [c for c in selected if c != pivot]
            pair_rng.shuffle(remaining)
            mapping_list = [None] * 4
            mapping_list[new_orientation] = pivot
            for i, color in zip([i for i in _ORIENTATIONS
                                 if i != new_orientation], remaining):
                mapping_list[i] = color
            mapping = tuple(mapping_list)
        else:
            background = _pick_background(
                pair_rng, base, core, variable_background)
            count = int(pair_rng.choice([1, 2, 3, 4],
                                        p=[0.06, 0.21, 0.25, 0.48]))
            if training:
                # Draw independently, while usually retaining familiar colors.
                novelty = str(pair_rng.choice(
                    ["familiar", "one", "several"],
                    p=[0.72, 0.23, 0.05]))
            else:
                novelty = str(pair_rng.choice(
                    ["familiar", "one", "several"],
                    p=[0.35, 0.40, 0.25]))
            if (count == 4 and background == base
                    and pair_rng.random() < 0.45):
                # A familiar mapping can recur after positions change.
                mapping = first_mapping
            else:
                selected = _sample_palette(
                    pair_rng, background, train_piece_colors,
                    novelty, count)
                # Ensure exactly the requested number of piece colors.
                mapping_list = list(selected)
                mapping_list += [int(v) for v in pair_rng.choice(
                    selected, 4 - count, replace=True)]
                pair_rng.shuffle(mapping_list)
                mapping = tuple(mapping_list)
        if index == 0:
            first_mapping = mapping

        if not training and train_shapes and pair_rng.random() < 0.30:
            shape = train_shapes[int(pair_rng.integers(0, len(train_shapes)))]
        else:
            shape = _sample_dimensions(pair_rng)
        if index == 1 and shape == train_shapes[0]:
            shape = (shape[0] + 1, shape[1])
        style = str(pair_rng.choice(
            ["scatter", "compact", "margin", "edge"],
            p=[0.38, 0.27, 0.17, 0.18]))
        pair, order = _make_pair(
            pair_rng, shape, background, mapping, style, seen,
            first_order if index == 1 else None)
        seen.add(_signature(pair["input"]))
        if training:
            puzzle["train"].append(pair)
            train_piece_colors.update(mapping)
            train_shapes.append(shape)
        else:
            puzzle["test"].append(pair)
        if index == 0:
            first_order = order
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            signature = _signature(inp)
            assert signature not in seen_inputs
            seen_inputs.add(signature)
            assert not any(np.array_equal(inp, a) and np.array_equal(out, b)
                           for a, b in _SOURCE_PAIRS)
            analysis = _analyze_input(inp)
            assert out.shape == (4, 4)
            assert np.array_equal(out, _assemble_output(
                analysis["background"], analysis["colors"]))
            assert np.all(out[1:3, 1:3] == analysis["background"])
            analyses[split].append(analysis)
    orders = {tuple(sorted(_ORIENTATIONS, key=lambda i: a["positions"][i]))
              for a in analyses["train"]}
    assert len(orders) >= 2
    history = {}
    for a in analyses["train"]:
        for orientation, color in enumerate(a["colors"]):
            history.setdefault(color, set()).add(orientation)
    assert any(len(orientations) >= 2 for orientations in history.values())
    # Recover the destination convention from the returned training grids.
    learned = {}
    for pair, analysis in zip(puzzle["train"], analyses["train"]):
        out = pair["output"]
        bg = analysis["background"]
        for orientation, color in enumerate(analysis["colors"]):
            candidates = []
            for row, col in ((0, 0), (0, 2), (2, 0), (2, 2)):
                tile = out[row:row + 2, col:col + 2]
                if tile[orientation // 2, orientation % 2] != bg:
                    continue
                if all(tile[k // 2, k % 2] == color for k in _ORIENTATIONS
                       if k != orientation):
                    candidates.append((row, col))
            assert len(candidates) == 1
            if orientation in learned:
                assert learned[orientation] == candidates[0]
            learned[orientation] = candidates[0]
    assert len(learned) == 4 and len(set(learned.values())) == 4
    for pair, analysis in zip(puzzle["test"], analyses["test"]):
        predicted = np.full((4, 4), analysis["background"], dtype=np.int8)
        for orientation, color in enumerate(analysis["colors"]):
            row, col = learned[orientation]
            for k in _ORIENTATIONS:
                if k != orientation:
                    predicted[row + k // 2, col + k % 2] = color
        assert np.array_equal(predicted, pair["output"])
    # The background and piece colors are locally visible. The training
    # witnesses rule out fixed scan-order and fixed color-to-corner mappings.


def _puzzles_equal(a, b):
    return all(
        np.array_equal(pa[key], pb[key])
        for split in ("train", "test")
        for pa, pb in zip(a[split], b[split])
        for key in ("input", "output")
    ) and all(len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    counts_train, counts_test = set(), set()
    signatures = set()
    shapes, palettes, repeated_colors, backgrounds = set(), set(), 0, set()
    novelty, color_counts = set(), set()
    metrics = {name: 0 for name in (
        "short", "equal_size", "square_13", "small", "large", "wide",
        "tall", "variable_background", "cross_role", "compact",
        "compact_repeated", "margin", "edge", "close", "scattered",
        "position_only", "position_and_palette", "stable_test",
        "reassigned_test")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        metrics["short"] += 2 <= len(puzzle["train"]) <= 4
        signatures.add(tuple(_signature(p["input"]) for split in
                             ("train", "test") for p in puzzle[split]))
        familiar = set()
        train_mappings = []
        for p in puzzle["train"]:
            a = _analyze_input(p["input"])
            familiar.update(a["colors"])
            train_mappings.append(a["colors"])
        all_pairs = puzzle["train"] + puzzle["test"]
        assert len({p["input"].shape for p in all_pairs}) >= 2
        train_shapes = {p["input"].shape for p in puzzle["train"]}
        metrics["equal_size"] += any(p["input"].shape in train_shapes
                                     for p in puzzle["test"])
        analyses = [_analyze_input(p["input"]) for p in all_pairs]
        local_backgrounds = {a["background"] for a in analyses}
        local_piece_colors = {c for a in analyses for c in a["colors"]}
        metrics["variable_background"] += len(local_backgrounds) > 1
        metrics["cross_role"] += bool(local_backgrounds & local_piece_colors)
        prior = []
        for split in ("train", "test"):
            for p in puzzle[split]:
                a = _analyze_input(p["input"])
                height, width = p["input"].shape
                shapes.add((height, width))
                palettes.add(tuple(sorted(set(a["colors"]))))
                backgrounds.add(a["background"])
                unique_colors = len(set(a["colors"]))
                color_counts.add(unique_colors)
                repeated_colors += unique_colors < 4
                metrics["square_13"] += (height, width) == (13, 13)
                metrics["small"] += min(height, width) <= 10
                metrics["large"] += max(height, width) >= 20
                metrics["wide"] += width >= height + 3
                metrics["tall"] += height >= width + 3
                positions = a["positions"]
                span_h = max(r for r, _ in positions) - min(
                    r for r, _ in positions) + 2
                span_w = max(c for _, c in positions) - min(
                    c for _, c in positions) + 2
                compact = span_h <= 8 and span_w <= 8
                metrics["compact"] += compact
                metrics["compact_repeated"] += compact and unique_colors < 4
                margins = [min(r, c, height - r - 2, width - c - 2)
                           for r, c in positions]
                metrics["margin"] += min(margins) >= 3
                metrics["edge"] += min(margins) == 0
                gaps = [max(abs(positions[i][0] - positions[j][0]) - 2,
                            abs(positions[i][1] - positions[j][1]) - 2)
                        for i in range(4) for j in range(i + 1, 4)]
                metrics["close"] += min(gaps) == 1
                metrics["scattered"] += min(gaps) >= 3
                mapping = a["colors"]
                metrics["position_only"] += any(
                    bg == a["background"] and old == mapping
                    and old_pos != positions for bg, old, old_pos in prior)
                metrics["position_and_palette"] += any(
                    old != mapping and old_pos != positions
                    for _, old, old_pos in prior)
                prior.append((a["background"], mapping, positions))
                if split == "test":
                    novelty.add(len(set(a["colors"]) - familiar))
                    metrics["stable_test"] += mapping in train_mappings
                    metrics["reassigned_test"] += any(
                        c in familiar and any(
                            old[i] != c and c in old
                            for old in train_mappings)
                        for i, c in enumerate(mapping))
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(palettes) >= 50
    assert repeated_colors >= 15 and color_counts == {1, 2, 3, 4}
    assert len(backgrounds) >= 8
    assert {0, 1, 2}.issubset(novelty)
    assert metrics["short"] >= 12 and metrics["equal_size"] >= 8
    assert all(metrics[name] >= 5 for name in (
        "square_13", "small", "large", "wide", "tall",
        "compact", "compact_repeated", "margin", "edge", "close",
        "scattered", "position_only", "position_and_palette",
        "stable_test", "reassigned_test"))
    assert metrics["variable_background"] >= 8
    assert metrics["cross_role"] >= 8
    for seed, n, m in ((61, 2, 1), (62, 10, 1), (63, 10, 4),
                       (64, 7, 3), (65, None, 1), (66, None, 4),
                       (67, 2, None), (68, 10, None)):
        p = generate_puzzle(seed, n, m)
        assert (n is None or len(p["train"]) == n)
        assert (m is None or len(p["test"]) == m)
        validate_generated_puzzle(p)
    for n, m in ((1, 1), (11, 1), (2, 0), (2, 5),
                 (2, 2), (3, 4)):
        try:
            generate_puzzle(99, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
