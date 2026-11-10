"""Synthetic repeated-panel discrepancy puzzles."""

import hashlib
import numpy as np


_SOURCE_PAIR_DIGESTS = frozenset({
    "97221d60c89282801ddef2dd4e8081fd076afc53e93230e0b1113c7bcf9d7fb1",
    "30c0faca9f684b5ffed0e69efa56d4a7cb5ea77a07e565c28d5719d60b94d8cc",
    "79ba30db709411ff5fd4139fe03a7192190a1d298e99f2f3c91153e0c1b72eb4",
    "ef8489d48e179d336f0042da3a738b09f298797f7b8b25235f8c8901033332b5",
})
_GEOMETRIES = tuple(
    (height, width, rows, columns)
    for height in range(4, 7)
    for width in range(4, 7)
    for rows in range(3, 6)
    for columns in range(3, 6)
    if rows * height + rows - 1 <= 30
    and columns * width + columns - 1 <= 30
)
_CLASSIC = (5, 5, 5, 5)


def _checked_count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = None if num_train is None else _checked_count("num_train", num_train, 2, 10)
    test = None if num_test is None else _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        if rng.random() < 0.18:
            train, test = 3, 1
        else:
            test = int(rng.integers(1, 5))
            train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _neighbors(point, height, width):
    row, column = point
    return {
        (r, c)
        for r, c in ((row - 1, column), (row + 1, column),
                     (row, column - 1), (row, column + 1))
        if 0 <= r < height and 0 <= c < width
    }


def _motif_mask(rng, height, width, style):
    area = height * width
    density_band = int(rng.integers(3))
    bounds = ((0.18, 0.29), (0.30, 0.44), (0.45, 0.58))[density_band]
    count = max(4, min(area - 3, int(round(area * rng.uniform(*bounds)))))
    cells = [(r, c) for r in range(height) for c in range(width)]
    required = set()
    forbidden = set()
    color_group = []

    if style in ("cluster", "mixed"):
        starts = [(r, c) for r in range(height - 1) for c in range(width - 1)]
        r, c = starts[int(rng.integers(len(starts)))]
        cluster = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
        required.update(cluster)
        color_group = cluster[:3]
    if style == "run":
        horizontal = bool(rng.integers(2))
        length = int(rng.integers(3, min(5, width if horizontal else height) + 1))
        r = int(rng.integers(height if horizontal else height - length + 1))
        c = int(rng.integers(width - length + 1 if horizontal else width))
        run = [(r, c + i) if horizontal else (r + i, c) for i in range(length)]
        required.update(run)
        color_group = run[:3]
    if style in ("isolated", "mixed"):
        candidates = [
            point for point in cells
            if point not in required and not (_neighbors(point, height, width) & required)
            and area - len(_neighbors(point, height, width) - required) >= count
        ]
        if candidates:
            isolated = candidates[int(rng.integers(len(candidates)))]
            required.add(isolated)
            forbidden.update(_neighbors(isolated, height, width) - required)
    possible = [point for point in cells if point not in required and point not in forbidden]
    extra = min(count - len(required), len(possible))
    if extra > 0:
        indices = rng.choice(len(possible), size=extra, replace=False)
        required.update(possible[int(i)] for i in np.atleast_1d(indices))
    mask = np.zeros((height, width), dtype=bool)
    for row, column in required:
        mask[row, column] = True
    return mask, color_group


def _color_motif(rng, mask, color_group, background, available):
    positions = [tuple(map(int, point)) for point in np.argwhere(mask)]
    group = [point for point in color_group if point in positions]
    maximum = min(5, len(positions) - len(group) + (1 if group else 0))
    accent_count = int(rng.integers(2, maximum + 1))
    accents = [int(x) for x in rng.choice(available, size=accent_count, replace=False)]
    motif = np.full(mask.shape, background, dtype=np.int8)
    for point in group:
        motif[point] = accents[0]
    remaining = [point for point in positions if point not in group]
    rng.shuffle(remaining)
    assigned = {accent: len(group) if accent == accents[0] else 0 for accent in accents}
    for point in remaining:
        least = min(assigned.values())
        choices = [accent for accent in accents if assigned[accent] == least]
        label = choices[int(rng.integers(len(choices)))]
        motif[point] = label
        assigned[label] += 1
    return motif


def _choose_errors(rng, motif, background, layout, split):
    height, width = motif.shape
    occupied = [tuple(map(int, point)) for point in np.argwhere(motif != background)]
    if layout in ("row", "column"):
        axis = 0 if layout == "row" else 1
        groups = [
            [point for point in occupied if point[axis] == index]
            for index in range((height, width)[axis])
        ]
        candidates = [group for group in groups if 2 <= len(group) <= 6]
        if candidates:
            return candidates[int(rng.integers(len(candidates)))]
    if layout == "run":
        runs = []
        occupied_set = set(occupied)
        for r, c in occupied:
            for dr, dc in ((0, 1), (1, 0)):
                for length in (2, 3, 4):
                    run = [(r + dr * k, c + dc * k) for k in range(length)]
                    if all(point in occupied_set for point in run):
                        runs.append(run)
        if runs:
            return runs[int(rng.integers(len(runs)))]
    upper = min(4 if split == "train" else 5, len(occupied))
    count = int(rng.integers(2 if layout == "distributed" else 1, upper + 1))
    if layout == "distributed":
        rows = sorted({point[0] for point in occupied})
        if len(rows) >= 2:
            chosen_rows = rng.choice(rows, size=2, replace=False)
            selected = [
                [point for point in occupied if point[0] == int(row)][
                    int(rng.integers(sum(point[0] == int(row) for point in occupied)))
                ]
                for row in chosen_rows
            ]
            remainder = [point for point in occupied if point not in selected]
            if count > 2:
                indices = rng.choice(len(remainder), size=count - 2, replace=False)
                selected.extend(remainder[int(i)] for i in np.atleast_1d(indices))
            return selected
    indices = rng.choice(len(occupied), size=count, replace=False)
    return [occupied[int(i)] for i in np.atleast_1d(indices)]


def _assemble_pair(rng, geometry, roles, style, layout, split):
    height, width, rows, columns = geometry
    separator, background, marker = roles
    mask, color_group = _motif_mask(rng, height, width, style)
    available = [color for color in range(10) if color not in roles]
    motif = _color_motif(rng, mask, color_group, background, available)
    errors = _choose_errors(rng, motif, background, layout, split)
    defect_row = int(rng.integers(rows))
    defect_column = int(rng.integers(columns))
    grid = np.full((rows * (height + 1) - 1, columns * (width + 1) - 1),
                   separator, dtype=np.int8)
    for panel_row in range(rows):
        for panel_column in range(columns):
            panel = motif.copy()
            if (panel_row, panel_column) == (defect_row, defect_column):
                for point in errors:
                    panel[point] = background
            r, c = panel_row * (height + 1), panel_column * (width + 1)
            grid[r:r + height, c:c + width] = panel
    output = motif.copy()
    for point in errors:
        output[point] = marker
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    marker = int(rng.integers(10))
    other = [color for color in range(10) if color != marker]
    base = [int(x) for x in rng.choice(other, size=2, replace=False)]
    classic = bool(rng.random() < (0.65 if (train_count, test_count) == (3, 1) else 0.22))
    total = train_count + test_count
    pair_seeds = rng.integers(0, 2**63, size=total, dtype=np.int64)
    geometries = []
    for index in range(total):
        if classic:
            geometry = _CLASSIC
        else:
            choices = _GEOMETRIES
            if index == 1:
                choices = tuple(item for item in choices if item != geometries[0])
            geometry = choices[int(rng.integers(len(choices)))]
        geometries.append(geometry)
    puzzle = {"train": [], "test": []}
    styles = ("scattered", "isolated", "run", "cluster", "mixed")
    layouts = ("distributed", "row", "column", "run", "scattered")
    for index in range(total):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        split = "train" if index < train_count else "test"
        local_index = index if split == "train" else index - train_count
        if classic or index == 0:
            separator, background = base
        elif index == 1:
            separator, background = base[::-1]
        elif pair_rng.random() < 0.35:
            separator, background = base
        elif pair_rng.random() < 0.4:
            separator, background = base[::-1]
        else:
            separator, background = [int(x) for x in pair_rng.choice(other, size=2, replace=False)]
        style = styles[int(pair_rng.integers(len(styles)))]
        if split == "train" and local_index == 0:
            layout = "distributed"
        elif split == "train" and local_index == 1:
            layout = "row" if bool(pair_rng.integers(2)) else "column"
        elif split == "test" and local_index == 0 and pair_rng.random() < 0.42:
            layout = "row"
        else:
            layout = layouts[int(pair_rng.integers(len(layouts)))]
        puzzle[split].append(_assemble_pair(pair_rng, geometries[index],
                                             (separator, background, marker),
                                             style, layout, split))
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return grid.shape, tuple(int(value) for value in grid.flat)


def _source_pair_digest(pair):
    inp, out = pair["input"], pair["output"]
    return hashlib.sha256(np.asarray(inp, dtype=np.uint8).tobytes() +
                          np.asarray(out, dtype=np.uint8).tobytes()).hexdigest()


def _segments(length, lines):
    bounds = [-1] + lines + [length]
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]


def _analyze_input(input_grid):
    height, width = input_grid.shape
    row_lines = [r for r in range(height) if np.all(input_grid[r] == input_grid[r, 0])]
    col_lines = [c for c in range(width) if np.all(input_grid[:, c] == input_grid[0, c])]
    assert 2 <= len(row_lines) <= 4 and 2 <= len(col_lines) <= 4
    separator = int(input_grid[row_lines[0], 0])
    assert all(np.all(input_grid[r] == separator) for r in row_lines)
    assert all(np.all(input_grid[:, c] == separator) for c in col_lines)
    row_segments, col_segments = _segments(height, row_lines), _segments(width, col_lines)
    panel_heights = {b - a for a, b in row_segments}
    panel_widths = {b - a for a, b in col_segments}
    assert len(panel_heights) == len(panel_widths) == 1
    panel_height, panel_width = panel_heights.pop(), panel_widths.pop()
    assert 4 <= panel_height <= 6 and 4 <= panel_width <= 6
    assert all(row_segments[i + 1][0] == row_segments[i][1] + 1 for i in range(len(row_segments) - 1))
    assert all(col_segments[i + 1][0] == col_segments[i][1] + 1 for i in range(len(col_segments) - 1))
    panels = np.stack([input_grid[ra:rb, ca:cb]
                       for ra, rb in row_segments for ca, cb in col_segments])
    assert not np.any(panels == separator)
    canonical = np.empty((panel_height, panel_width), dtype=np.int8)
    for r in range(panel_height):
        for c in range(panel_width):
            counts = np.bincount(panels[:, r, c].astype(int), minlength=10)
            winner = int(np.argmax(counts))
            assert counts[winner] >= len(panels) - 1
            canonical[r, c] = winner
    counts = np.bincount(canonical.ravel().astype(int), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > np.partition(counts, -2)[-2]
    assert counts[background] >= 0.4 * canonical.size
    assert background != separator
    accents = set(int(x) for x in np.unique(canonical)) - {background}
    assert 2 <= len(accents) <= 5 and separator not in accents
    differences = panels != canonical
    bad = np.flatnonzero(np.any(differences, axis=(1, 2)))
    assert len(bad) == 1
    error_mask = differences[int(bad[0])]
    assert 1 <= int(error_mask.sum()) <= 6
    assert np.all(panels[int(bad[0])][error_mask] == background)
    assert np.all(canonical[error_mask] != background)
    assert np.all(np.sum(differences, axis=0)[error_mask] == 1)
    return {"canonical": canonical, "error_mask": error_mask,
            "background": background, "separator": separator,
            "defect": int(bad[0]), "copies": len(panels),
            "geometry": (panel_height, panel_width, len(row_segments), len(col_segments)),
            "accent_count": len(accents), "foreground": int(np.sum(canonical != background)),
            "error_count": int(error_mask.sum()),
            "error_rows": len(np.unique(np.argwhere(error_mask)[:, 0])),
            "error_columns": len(np.unique(np.argwhere(error_mask)[:, 1]))}


def _predict(input_grid, marker):
    analysis = _analyze_input(input_grid)
    result = analysis["canonical"].copy()
    result[analysis["error_mask"]] = marker
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    markers = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = input_key, _grid_key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if pair["input"].shape == (29, 29) and pair["output"].shape == (5, 5):
                assert _source_pair_digest(pair) not in _SOURCE_PAIR_DIGESTS
            analysis = _analyze_input(pair["input"])
            assert pair["output"].shape == analysis["canonical"].shape
            unchanged = ~analysis["error_mask"]
            assert np.array_equal(pair["output"][unchanged], analysis["canonical"][unchanged])
            marker_values = np.unique(pair["output"][analysis["error_mask"]])
            assert len(marker_values) == 1
            marker = int(marker_values[0])
            assert marker not in pair["input"]
            assert marker != analysis["background"] and marker != analysis["separator"]
            assert np.array_equal(_predict(pair["input"], marker), pair["output"])
            markers.add(marker)
            analyses[split].append(analysis)
    assert len(markers) == 1
    marker = next(iter(markers))
    assert all(np.any(pair["output"] == marker) for pair in puzzle["train"])
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(pair["input"], marker), pair["output"])


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[s], second[s])) for s in ("train", "test"))


def _motif_features(analysis):
    motif = analysis["canonical"]
    mask = motif != analysis["background"]
    h, w = mask.shape
    isolated = sum(bool(mask[r, c]) and not any(mask[p] for p in _neighbors((r, c), h, w))
                   for r in range(h) for c in range(w))
    same_run = any(motif[r, c] == motif[r, c + 1] == motif[r, c + 2]
                   and mask[r, c] and mask[r, c + 1] and mask[r, c + 2]
                   for r in range(h) for c in range(w - 2)) or any(
        motif[r, c] == motif[r + 1, c] == motif[r + 2, c]
        and mask[r, c] and mask[r + 1, c] and mask[r + 2, c]
        for r in range(h - 2) for c in range(w))
    block = any(np.all(mask[r:r + 2, c:c + 2])
                for r in range(h - 1) for c in range(w - 1))
    return isolated, same_run, block


def self_test() -> None:
    specifications = [(1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                      (1004, None, 1), (1005, None, 4),
                      (1006, 2, None), (1007, 10, None)]
    for seed, train, test in specifications:
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    a, b = generate_puzzle(123456, 7, 3), generate_puzzle(123456, 7, 3)
    assert _puzzles_equal(a, b)
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4},
                      {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(77, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
    train_counts, test_counts, geometries, roles, masks, structures = set(), set(), set(), set(), set(), set()
    layouts = {"train": set(), "test": set()}
    motif_features = {"isolated": 0, "run": 0, "block": 0}
    densities, accents, defect_locations = set(), set(), set()
    recurring_locations = 0
    pair_color_changes = 0
    mixed_geometries = 0
    separator_three = 0
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_geometries, local_roles, local_defects = set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_input(pair["input"])
                marker = int(pair["output"][analysis["error_mask"]][0])
                assert np.array_equal(_predict(pair["input"], marker), pair["output"])
                local_geometries.add(analysis["geometry"])
                local_roles.add((analysis["separator"], analysis["background"]))
                local_defects.add((analysis["geometry"], analysis["defect"]))
                geometries.add(analysis["geometry"])
                roles.add((analysis["separator"], analysis["background"], marker))
                defect_locations.add((analysis["geometry"], analysis["defect"]))
                if analysis["separator"] == 3:
                    separator_three += 1
                mask = analysis["canonical"] != analysis["background"]
                masks.add((mask.shape, tuple(bool(v) for v in mask.flat)))
                structures.add((analysis["geometry"], tuple(int(v) for v in analysis["canonical"].flat)))
                densities.add(round(analysis["foreground"] / mask.size, 1))
                accents.add(analysis["accent_count"])
                if analysis["error_rows"] == 1:
                    layouts[split].add("one_row")
                if analysis["error_columns"] == 1:
                    layouts[split].add("one_column")
                if analysis["error_rows"] >= 2:
                    layouts[split].add("distributed")
                layouts[split].add(analysis["error_count"])
                isolated, run, block = _motif_features(analysis)
                motif_features["isolated"] += bool(isolated)
                motif_features["run"] += bool(run)
                motif_features["block"] += bool(block)
        recurring_locations += len(puzzle["train"] + puzzle["test"]) - len(local_defects)
        pair_color_changes += len(local_roles) > 1
        mixed_geometries += len(local_geometries) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert sum(not _puzzles_equal(puzzles[0], p) for p in puzzles[1:]) >= 40
    assert len(geometries) >= 12 and len(roles) >= 30
    assert len(masks) >= 0.8 * sum(len(p["train"] + p["test"]) for p in puzzles)
    assert len(structures) > 200 and len(defect_locations) >= 30
    assert len(densities) >= 4 and len(accents) >= 3
    assert all({"one_row", "one_column", "distributed"} <= layouts[s] for s in layouts)
    assert all(len({v for v in layouts[s] if isinstance(v, int)}) >= 4 for s in layouts)
    assert all(count >= 20 for count in motif_features.values())
    assert recurring_locations >= 1 and pair_color_changes >= 10
    assert mixed_geometries >= 20 and separator_three >= 1
