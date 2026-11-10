"""ARC rectangles recolored by the marker above each horizontal span."""

import numpy as np


_SOURCE_INPUT_ROWS = {
    ("0100040070", "0000000000", "5555000555", "5555000555",
     "5555000555", "5555000555", "0000000000", "0005555000",
     "0005555000", "0000000000"),
    ("0020060008", "0000000000", "0000555500", "0000555500",
     "0550555500", "0550555500", "0550000000", "0550000555",
     "0550000555", "0000000555"),
    ("0100060070", "0000000000", "0550000000", "0550000555",
     "0550000555", "0005550000", "0005550000", "0005550000",
     "0000000000", "0000000000"),
    ("3000600090", "0000000000", "0055555055", "0055555055",
     "0055555055", "0055555055", "0000000055", "5555000055",
     "5555000055", "0000000000"),
}


def _count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        lower = max(2, test + 1)
        train = lower + (seed // 4) % (11 - lower)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return grid.shape, grid.tobytes()


def _source_copy(grid):
    if grid.shape != (10, 10):
        return False
    rows = tuple("".join(str(int(x)) for x in row) for row in grid)
    return rows in _SOURCE_INPUT_ROWS


def _geometry(rng, count, *, compact=False, showcase=False,
              first_train=False, target_test=False, heldout=False):
    """Build disjoint column lanes; row intervals remain independently free."""
    for _ in range(500):
        if compact:
            legend = 0
            gap = 1
            widths = [int(rng.integers(2, 4)) for _ in range(count)]
            spaces = [1] * (count - 1)
            remaining = 10 - sum(widths) - sum(spaces)
            if remaining < 0:
                continue
            left_margin = int(rng.integers(0, remaining + 1))
            right_margin = remaining - left_margin
            canvas_width = 10
        else:
            legend = 0 if first_train else int(rng.integers(0, 3))
            if showcase:
                legend = int(rng.integers(1, 3))
            gap = 1 if first_train else int(rng.integers(1, 4))
            if showcase:
                gap = int(rng.integers(2, 4))
            widths = [int(rng.integers(2, 6)) for _ in range(count)]
            if showcase or rng.random() < 0.55:
                wide = int(rng.integers(count))
                widths[wide] = int(rng.integers(5, 9))
            if showcase:
                other = (wide + 1) % count
                widths[other] = max(3, widths[other])
            if heldout:
                heldout_index = int(rng.integers(count))
                widths[heldout_index] = 8
            spaces = [int(rng.integers(1, 4)) for _ in range(count - 1)]
            left_margin = int(rng.integers(0, 4))
            right_margin = int(rng.integers(0, 4))
            canvas_width = (left_margin + sum(widths) +
                            sum(spaces) + right_margin)
            if canvas_width > 30 or canvas_width < 9:
                continue

        lefts = []
        cursor = left_margin
        for index, object_width in enumerate(widths):
            lefts.append(cursor)
            cursor += object_width
            if index < count - 1:
                cursor += spaces[index]
        assert cursor + right_margin == canvas_width

        heights = [
            int(rng.integers(2, 5 if compact else 8))
            for _ in range(count)
        ]
        if heldout:
            heights[heldout_index] = 7
        if not target_test and any(
            width == 8 and object_height == 7
            for width, object_height in zip(widths, heights)
        ):
            continue
        start = legend + gap + 1
        if showcase:
            heights[0] = 5
            heights[1] = max(3, heights[1])
            mode = "stagger"
        elif compact:
            mode = "overlap" if rng.random() < 0.65 else "free"
        else:
            mode = ("overlap", "free", "spread", "ordered")[
                int(rng.integers(0, 4))
            ]

        if mode == "stagger":
            tops = [start, start + 1, start + 2]
            tops += [
                start + int(rng.integers(0, 5))
                for _ in range(count - 3)
            ]
        elif mode == "overlap":
            base = start + int(rng.integers(0, 3))
            tops = [
                base + int(rng.integers(0, 3))
                for _ in range(count)
            ]
        elif mode == "ordered":
            order = list(rng.permutation(count))
            tops = [0] * count
            position = start
            for index in order:
                tops[index] = position
                position += heights[index] + int(rng.integers(1, 3))
        else:
            reach = 5 if compact else (8 if mode == "free" else 13)
            tops = [
                start + int(rng.integers(0, reach))
                for _ in range(count)
            ]
        if first_train:
            tops[0] = start

        last_row = max(t + h for t, h in zip(tops, heights))
        if compact:
            canvas_height = 10
            if last_row > 10:
                continue
        else:
            canvas_height = max(
                9, last_row + int(rng.integers(0, 4))
            )
            if canvas_height > 30:
                continue

        offsets = [
            int(rng.integers(0, width)) for width in widths
        ]
        if showcase:
            offsets[wide] = 0 if rng.random() < 0.5 else widths[wide] - 1
            offsets[other] = 1
        markers = [left + offset for left, offset in zip(lefts, offsets)]
        rectangles = [
            (top, left, height, width)
            for top, left, height, width in
            zip(tops, lefts, heights, widths)
        ]
        return canvas_height, canvas_width, legend, markers, rectangles
    raise ValueError("could not construct geometry")


def _make_pair(rng, background, placeholder, count, *,
               compact=False, showcase=False, first_train=False,
               target_test=False, heldout=False):
    height, width, legend, markers, rectangles = _geometry(
        rng, count, compact=compact, showcase=showcase,
        first_train=first_train, target_test=target_test,
        heldout=heldout
    )
    choices = [c for c in range(10) if c not in (background, placeholder)]
    colors = [int(c) for c in rng.choice(choices, size=count, replace=False)]
    input_grid = np.full((height, width), background, dtype=np.int8)
    for column, color in zip(markers, colors):
        input_grid[legend, column] = color
    output_grid = input_grid.copy()
    for (top, left, object_height, object_width), color in zip(
        rectangles, colors
    ):
        region = np.s_[top:top + object_height, left:left + object_width]
        input_grid[region] = placeholder
        output_grid[region] = color
    return {"input": input_grid, "output": output_grid}


def _components(grid, color, first_object_row):
    mask = grid == color
    mask[:first_object_row, :] = False
    seen = np.zeros(grid.shape, dtype=bool)
    found = []
    for row in range(first_object_row, grid.shape[0]):
        for col in range(grid.shape[1]):
            if not mask[row, col] or seen[row, col]:
                continue
            todo = [(row, col)]
            seen[row, col] = True
            cells = []
            for r, c in todo:
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c),
                               (r, c - 1), (r, c + 1)):
                    if (0 <= nr < grid.shape[0] and
                        0 <= nc < grid.shape[1] and
                        mask[nr, nc] and not seen[nr, nc]):
                        seen[nr, nc] = True
                        todo.append((nr, nc))
            found.append(cells)
    return found


def _inspect(pair):
    source = pair["input"]
    target = pair["output"]
    assert source.shape == target.shape
    height, width = source.shape
    candidates = []
    for legend in range(min(3, height - 2)):
        background = int(source[legend + 1, 0])
        if not np.all(source[legend + 1] == background):
            continue
        if not np.all(source[:legend] == background):
            continue
        columns = np.flatnonzero(source[legend] != background)
        if not 2 <= len(columns) <= 5:
            continue
        if any(b - a < 2 for a, b in zip(columns, columns[1:])):
            continue
        candidates.append((legend, background, columns))
    assert len(candidates) == 1
    legend, background, columns = candidates[0]
    colors = [int(source[legend, column]) for column in columns]
    assert len(set(colors)) == len(colors)
    first_object_row = legend + 2
    assert np.all(source[legend + 1] == background)
    lower = source[first_object_row:]
    values = np.unique(lower[lower != background])
    assert len(values) == 1
    placeholder = int(values[0])
    assert placeholder not in colors
    objects = _components(source, placeholder, first_object_row)
    assert len(objects) == len(columns)
    expected = source.copy()
    records = []
    used = []
    for cells in objects:
        rows, cols = zip(*cells)
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        object_height = bottom - top + 1
        object_width = right - left + 1
        assert object_height >= 2 and object_width >= 2
        assert len(cells) == object_height * object_width
        assert np.all(source[top:bottom + 1, left:right + 1] == placeholder)
        aligned = [int(col) for col in columns if left <= col <= right]
        assert len(aligned) == 1
        marker = aligned[0]
        used.append(marker)
        color = int(source[legend, marker])
        expected[top:bottom + 1, left:right + 1] = color
        records.append((top, bottom, left, right, marker))
    assert set(used) == set(int(c) for c in columns)
    assert np.array_equal(target, expected)
    assert not np.array_equal(source, target)
    return {
        "shape": (height, width), "legend": legend,
        "gap": min(r[0] for r in records) - legend - 1,
        "background": background, "placeholder": placeholder,
        "count": len(columns), "records": sorted(records, key=lambda r: r[4]),
        "palette": tuple(colors),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    inputs, pairs, layouts = set(), set(), set()
    infos = {"train": [], "test": []}
    for group in ("train", "test"):
        for pair in puzzle[group]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            assert not _source_copy(pair["input"])
            info = _inspect(pair)
            layout = (info["shape"], info["legend"], tuple(info["records"]))
            assert layout not in layouts
            layouts.add(layout)
            infos[group].append(info)
    all_infos = infos["train"] + infos["test"]
    assert len({info["background"] for info in all_infos}) == 1
    assert infos["train"][0]["legend"] == 0
    assert infos["train"][0]["gap"] == 1
    assert infos["train"][0]["count"] == 3
    assert infos["train"][1]["count"] == 5
    assert infos["train"][1]["legend"] > 0
    assert infos["train"][1]["gap"] >= 2
    assert infos["train"][0]["placeholder"] != infos["train"][1]["placeholder"]
    assert any(
        r[4] in (r[2], r[3]) and r[3] - r[2] >= 2
        for info in infos["train"] for r in info["records"]
    )
    assert any(
        r[2] < r[4] < r[3]
        for info in infos["train"] for r in info["records"]
    )
    assert any(
        r[1] - r[0] + 1 >= 5
        for info in infos["train"] for r in info["records"]
    )
    assert any(
        a[0] <= b[1] and b[0] <= a[1]
        for info in infos["train"]
        for a in info["records"] for b in info["records"]
        if a != b
    )
    assert max(info["count"] for info in infos["train"]) >= max(
        info["count"] for info in infos["test"]
    )


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    background = int(rng.integers(0, 10))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    previous_placeholder = None
    for group, total in (("train", train_count), ("test", test_count)):
        for index in range(total):
            for _ in range(100):
                if group == "train" and index == 0:
                    count = 3
                    compact = seed % 3 == 0
                elif group == "train" and index == 1:
                    count = 5
                    compact = False
                elif group == "test" and index == 0 and seed % 3 == 0:
                    count = 3
                    compact = True
                else:
                    count = int(rng.integers(2, 6))
                    compact = False
                choices = [
                    c for c in range(10)
                    if c != background and
                    (not (group == "train" and index == 1) or
                     c != previous_placeholder)
                ]
                placeholder = int(rng.choice(choices))
                pair_rng = np.random.default_rng(
                    int(rng.integers(0, 2**32, dtype=np.uint64))
                )
                try:
                    pair = _make_pair(
                        pair_rng, background, placeholder, count,
                        compact=compact,
                        showcase=group == "train" and index == 1,
                        first_train=group == "train" and index == 0,
                        target_test=group == "test",
                        heldout=(group == "test" and index == 0 and
                                 seed % 3 != 0 and seed % 4 == 1),
                    )
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key in used_inputs or _source_copy(pair["input"]):
                    continue
                used_inputs.add(key)
                puzzle[group].append(pair)
                previous_placeholder = placeholder
                break
            else:
                raise ValueError("could not construct unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(a, b):
    return all(
        len(a[group]) == len(b[group]) and all(
            np.array_equal(x[name], y[name])
            for x, y in zip(a[group], b[group])
            for name in ("input", "output")
        )
        for group in ("train", "test")
    )


def self_test() -> None:
    assert _same_puzzle(generate_puzzle(17), generate_puzzle(17))
    assert _same_puzzle(
        generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3)
    )
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + index, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
    for index, counts in enumerate(((None, 1), (None, 4),
                                    (2, None), (10, None))):
        validate_generated_puzzle(generate_puzzle(3000 + index, *counts))
    for args in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 2.0}, {"num_test": True},
    ):
        try:
            generate_puzzle(4000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {args}")
    counts_train, counts_test = set(), set()
    shapes, layouts, palettes, placeholders, backgrounds = (
        set(), set(), set(), set(), set()
    )
    counts, gaps, legend_rows = set(), set(), set()
    overlaps, wide_train, wide_test, ten_tests = 0, 0, 0, 0
    train_combinations, test_combinations = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        for group in ("train", "test"):
            for pair in puzzle[group]:
                info = _inspect(pair)
                shapes.add(info["shape"])
                layouts.add((info["shape"], info["legend"],
                             tuple(info["records"])))
                palettes.add(info["palette"])
                placeholders.add(info["placeholder"])
                backgrounds.add(info["background"])
                counts.add(info["count"])
                gaps.add(info["gap"])
                legend_rows.add(info["legend"])
                for top, bottom, left, right, marker in info["records"]:
                    combination = (right - left + 1, bottom - top + 1)
                    if group == "train":
                        train_combinations.add(combination)
                    else:
                        test_combinations.add(combination)
                if any(a[0] <= b[1] and b[0] <= a[1]
                       for a in info["records"] for b in info["records"]
                       if a != b):
                    overlaps += 1
                if any(r[3] - r[2] + 1 >= 5 for r in info["records"]):
                    if group == "train":
                        wide_train += 1
                    else:
                        wide_test += 1
                if group == "test" and info["shape"] == (10, 10):
                    ten_tests += 1
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert counts == {2, 3, 4, 5}
    assert len(shapes) >= 25 and len(layouts) >= 150
    assert len(palettes) >= 100 and len(placeholders) >= 7
    assert len(backgrounds) >= 7
    assert gaps >= {1, 2, 3} and legend_rows == {0, 1, 2}
    assert overlaps >= 50
    assert wide_train >= 30 and wide_test >= 10 and ten_tests >= 10
    assert (8, 7) not in train_combinations
    assert (8, 7) in test_combinations
