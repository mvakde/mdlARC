"""Rectangular-panel ARC puzzles with local, shared-boundary votes."""

from collections import Counter

import numpy as np


_SOURCE_INPUT_ROWS = frozenset({
    ("2555155552", "0050000500", "0010000100", "0050000500", "0050000500", "5155525551", "0050000500", "0050000200", "0050000500", "1555255551"),
    ("1515525551", "0005000500", "0005000200", "0005000500", "5255555525", "0005000500", "0001000500", "0005000500", "0005000500", "5255551552"),
    ("1525255551", "0005002000", "0005005000", "5255515525", "0005002000", "0001005000", "2255155515", "0005001000", "0005005000", "5515515525"),
    ("5155555515", "0050050000", "0050010000", "5555155215", "0020020000", "0050050000", "2555255125", "0010020000", "0050050000", "5255551551"),
})


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _count_arg(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _count_arg(num_train, "num_train", 2, 10)
    test = _count_arg(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        # Cycling the legal ranges guarantees count diversity even in a short seed run.
        test = 1 + int(seed) % 4
        low = max(2, test + 1)
        train = low + (int(seed) // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _open_intervals(length, bars):
    occupied = set(bars)
    result = []
    start = None
    for position in range(length + 1):
        if position < length and position not in occupied:
            if start is None:
                start = position
        elif start is not None:
            result.append((start, position - 1))
            start = None
    return result


def _axis(rng, count, mode, edge_start, edge_end, enlarge):
    if mode == "compact":
        sizes = [int(x) for x in rng.integers(2, 4, size=count)]
    else:
        sizes = [int(x) for x in rng.integers(2, 6, size=count)]
    capacity = 30 - (count - 1) - int(edge_start) - int(edge_end)
    while sum(sizes) > capacity:
        largest = max(range(count), key=lambda index: sizes[index])
        sizes[largest] -= 1
    if mode == "stretched" and enlarge:
        index = int(rng.integers(count))
        extra = min(int(rng.integers(3, 10)), capacity - sum(sizes))
        sizes[index] += max(0, extra)
    if mode == "large" and enlarge:
        target = int(rng.integers(25, 31)) - (count - 1) - int(edge_start) - int(edge_end)
        target = min(capacity, max(sum(sizes), target))
        extra = target - sum(sizes)
        prominent = int(rng.integers(count))
        for _ in range(extra):
            if float(rng.random()) < 0.6:
                sizes[prominent] += 1
            else:
                sizes[int(rng.integers(count))] += 1
    return sizes


def _geometry(rng, regime, pattern):
    if regime == "source_train":
        count_weights = [0.32, 0.32, 0.24, 0.12]
        size_weights = [0.21, 0.51, 0.20, 0.08]
    else:
        count_weights = [0.21, 0.29, 0.29, 0.21]
        size_weights = [0.12, 0.37, 0.30, 0.21]
    rows = int(rng.choice([2, 3, 4, 5], p=count_weights))
    columns = int(rng.choice([2, 3, 4, 5], p=count_weights))
    mode = str(rng.choice(["compact", "ordinary", "stretched", "large"], p=size_weights))
    edges = [bool(rng.integers(2)) for _ in range(4)]
    if pattern == "checker":
        # Opposite winners need some private boundary cells to be realizable.
        if not (edges[0] or edges[1]):
            edges[int(rng.integers(2))] = True
        if not (edges[2] or edges[3]):
            edges[2 + int(rng.integers(2))] = True
    enlarged_axis = int(rng.integers(2))
    row_sizes = _axis(rng, rows, mode, edges[0], edges[1], enlarged_axis == 0 or mode == "large")
    column_sizes = _axis(rng, columns, mode, edges[2], edges[3], enlarged_axis == 1 or mode == "large")
    if bool(rng.integers(2)):
        row_sizes, column_sizes = column_sizes, row_sizes
        edges = [edges[2], edges[3], edges[0], edges[1]]
    return row_sizes, column_sizes, tuple(edges), mode


def _bar_positions(sizes, before, after):
    bars = []
    intervals = []
    cursor = 0
    if before:
        bars.append(cursor)
        cursor += 1
    for index, size in enumerate(sizes):
        intervals.append((cursor, cursor + size - 1))
        cursor += size
        if index + 1 < len(sizes) or after:
            bars.append(cursor)
            cursor += 1
    return bars, intervals, cursor


def _base_grid(row_sizes, column_sizes, edges, background, structure):
    hbars, row_intervals, height = _bar_positions(row_sizes, edges[0], edges[1])
    vbars, column_intervals, width = _bar_positions(column_sizes, edges[2], edges[3])
    grid = np.full((height, width), background, dtype=np.int8)
    grid[hbars, :] = structure
    grid[:, vbars] = structure
    panels = [(r0, r1, c0, c1) for r0, r1 in row_intervals for c0, c1 in column_intervals]
    return grid, hbars, vbars, panels


def _boundary(panel, shape):
    r0, r1, c0, c1 = panel
    height, width = shape
    cells = set()
    if r0:
        cells.update((r0 - 1, column) for column in range(c0, c1 + 1))
    if r1 + 1 < height:
        cells.update((r1 + 1, column) for column in range(c0, c1 + 1))
    if c0:
        cells.update((row, c0 - 1) for row in range(r0, r1 + 1))
    if c1 + 1 < width:
        cells.update((row, c1 + 1) for row in range(r0, r1 + 1))
    return tuple(sorted(cells))


def _votes(grid, boundary, labels):
    return tuple(sum(int(grid[cell] == label) for cell in boundary) for label in labels)


def _analyze_geometry(grid, background):
    mask = grid != background
    hbars = np.flatnonzero(np.all(mask, axis=1)).tolist()
    vbars = np.flatnonzero(np.all(mask, axis=0)).tolist()
    expected = np.zeros(grid.shape, dtype=bool)
    expected[hbars, :] = True
    expected[:, vbars] = True
    _require(np.array_equal(mask, expected), "input is not a union of complete one-cell bars")
    row_intervals = _open_intervals(grid.shape[0], hbars)
    column_intervals = _open_intervals(grid.shape[1], vbars)
    _require(2 <= len(row_intervals) <= 5 and 2 <= len(column_intervals) <= 5, "panel count out of range")
    _require(all(end - start + 1 >= 2 for start, end in row_intervals + column_intervals), "panel too thin")
    panels = [(r0, r1, c0, c1) for r0, r1 in row_intervals for c0, c1 in column_intervals]
    return {"mask": mask, "hbars": hbars, "vbars": vbars, "rows": row_intervals, "columns": column_intervals, "panels": panels}


def _winner_map(rng, rows, columns, pattern):
    if pattern in ("all0", "all1"):
        return [int(pattern[-1])] * (rows * columns)
    if pattern == "checker":
        phase = int(rng.integers(2))
        return [(row + column + phase) % 2 for row in range(rows) for column in range(columns)]
    if pattern == "cluster":
        axis = int(rng.integers(2))
        count = rows if axis == 0 else columns
        split = int(rng.integers(1, count))
        phase = int(rng.integers(2))
        return [phase ^ int((row if axis == 0 else column) >= split) for row in range(rows) for column in range(columns)]
    return [int(value) for value in rng.integers(0, 2, size=rows * columns)]


def _populate(rng, base, hbars, vbars, panels, labels, structure, desired, pattern, forced_tie, priority):
    boundaries = [_boundary(panel, base.shape) for panel in panels]
    owners = {}
    for index, boundary in enumerate(boundaries):
        for cell in boundary:
            owners.setdefault(cell, []).append(index)
    hset, vset = set(hbars), set(vbars)
    eligible = [cell for cell in owners if not (cell[0] in hset and cell[1] in vset)]
    if forced_tie:
        possible = [index for index, cells in enumerate(boundaries) if len(cells) >= 2]
        tie_index = int(rng.choice(possible))
        tie_size = int(rng.integers(1, min(3, len(boundaries[tie_index]) // 2) + 1))
    else:
        tie_index, tie_size = None, 0
    placement = str(rng.choice(["isolated", "grouped", "exterior", "shared", "side"], p=[0.21, 0.20, 0.19, 0.20, 0.20]))
    density = float(rng.uniform(0.12, 0.31))
    grid = base.copy()
    order = rng.permutation(len(eligible))
    selected = set()
    for candidate in order:
        cell = eligible[int(candidate)]
        local_density = density
        if placement == "exterior":
            local_density *= 1.65 if len(owners[cell]) == 1 else 0.45
        elif placement == "shared":
            local_density *= 1.65 if len(owners[cell]) == 2 else 0.45
        if float(rng.random()) >= local_density:
            continue
        if placement == "isolated" and any((cell[0] + dr, cell[1] + dc) in selected for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            continue
        selected.add(cell)
        if placement == "grouped" and float(rng.random()) < 0.65:
            neighbors = [neighbor for neighbor in ((cell[0] + 1, cell[1]), (cell[0] - 1, cell[1]), (cell[0], cell[1] + 1), (cell[0], cell[1] - 1)) if neighbor in owners and neighbor in eligible]
            if neighbors:
                selected.add(neighbors[int(rng.integers(len(neighbors)))])
    for cell in selected:
        nearby = owners[cell]
        score = sum(1 if desired[index] == 1 else -1 for index in nearby)
        if placement == "side":
            score += 1.4 if cell[0] in hset else -1.4
        probability_one = 0.5 + (0.28 if score > 0 else -0.28 if score < 0 else 0.0)
        grid[cell] = labels[int(float(rng.random()) < probability_one)]
    locked = set()
    if tie_index is not None:
        locked.update(boundaries[tie_index])
        for cell in locked:
            grid[cell] = structure
        cells = rng.choice(len(boundaries[tie_index]), size=2 * tie_size, replace=False)
        values = [labels[0]] * tie_size + [labels[1]] * tie_size
        rng.shuffle(values)
        for local, value in zip(cells, values):
            grid[boundaries[tie_index][int(local)]] = value
    # Every input exposes two competitors, even when all panels elect one.
    for label in labels:
        if not np.any(grid == label):
            candidates = [cell for cell in eligible if cell not in locked and grid[cell] == structure]
            if not candidates:
                return None
            grid[candidates[int(rng.integers(len(candidates)))]] = label
    for _ in range(3 * len(eligible) + 1):
        deficits = []
        for index, boundary in enumerate(boundaries):
            if index == tie_index:
                continue
            counts = _votes(grid, boundary, labels)
            target = desired[index]
            gap = counts[1 - target] - counts[target] + 1
            if gap > 0:
                deficits.append((gap, index))
        if not deficits:
            markers = sum(int(np.count_nonzero(grid == label)) for label in labels)
            if markers < int(np.count_nonzero(grid == structure)):
                return grid
            return None
        _, index = max(deficits)
        target = desired[index]
        candidates = [cell for cell in boundaries[index] if cell not in locked and grid[cell] == structure]
        if not candidates:
            return None
        scores = []
        for cell in candidates:
            score = sum(2 if desired[other] == target else -2 for other in owners[cell] if other != index)
            score += 1 if len(owners[cell]) == 1 else 0
            scores.append(score)
        best = max(scores)
        choices = [cell for cell, score in zip(candidates, scores) if score == best]
        grid[choices[int(rng.integers(len(choices)))]] = labels[target]
    return None


def _solve(input_grid, background, labels, tie_priority):
    geometry = _analyze_geometry(input_grid, background)
    output = input_grid.copy()
    for panel in geometry["panels"]:
        first, second = _votes(input_grid, _boundary(panel, input_grid.shape), labels)
        _require(first + second > 0, "panel has no visible boundary vote")
        winner = labels[0] if first > second else labels[1] if second > first else tie_priority
        r0, r1, c0, c1 = panel
        output[r0:r1 + 1, c0:c1 + 1] = winner
    return output


def _roles(rng, puzzle):
    if puzzle["tie_enabled"]:
        labels = puzzle["labels"]
        other = [color for color in range(10) if color not in labels]
        background, structure = (int(value) for value in rng.choice(other, size=2, replace=False))
    else:
        background, structure, *labels = (int(value) for value in rng.choice(10, size=4, replace=False))
        labels = tuple(sorted(labels))
    return background, structure, tuple(labels)


def _pair(rng, regime, puzzle, force_tie, forced_pattern=None, geometry_template=None):
    for _ in range(64):
        pattern = forced_pattern or str(rng.choice(["all0", "all1", "cluster", "checker", "irregular"], p=[0.13, 0.13, 0.24, 0.20, 0.30]))
        if force_tie and forced_pattern is None and pattern in ("all0", "all1"):
            pattern = "irregular"
        if geometry_template is not None and pattern == "checker":
            edges = geometry_template[2]
            if not (edges[0] or edges[1]) or not (edges[2] or edges[3]):
                pattern = "cluster"
        background, structure, labels = _roles(rng, puzzle)
        if geometry_template is None:
            row_sizes, column_sizes, edges, _ = _geometry(rng, regime, pattern)
        else:
            row_sizes, column_sizes, edges = geometry_template
        base, hbars, vbars, panels = _base_grid(row_sizes, column_sizes, edges, background, structure)
        desired = _winner_map(rng, len(row_sizes), len(column_sizes), pattern)
        marked = _populate(rng, base, hbars, vbars, panels, labels, structure, desired, pattern, force_tie, puzzle["priority"])
        if marked is None:
            continue
        output = _solve(marked, background, labels, puzzle["priority"] if puzzle["tie_enabled"] else labels[0])
        if _rows_signature(marked) in _SOURCE_INPUT_ROWS:
            continue
        return {"input": marked, "output": output}
    raise RuntimeError("could not construct a panel network with consistent votes")


def _array_key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes(order="C")


def _rows_signature(grid):
    if grid.shape != (10, 10):
        return None
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _puzzles_equal(first, second):
    return all(len(first[section]) == len(second[section]) and all(np.array_equal(a[side], b[side]) for a, b in zip(first[section], second[section]) for side in ("input", "output")) for section in ("train", "test"))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    tie_enabled = bool(float(rng.random()) < 0.26)
    if tie_enabled:
        labels = tuple(sorted(int(value) for value in rng.choice(10, size=2, replace=False)))
        priority = labels[int(rng.integers(2))]
    else:
        labels, priority = None, None
    conventions = {"tie_enabled": tie_enabled, "labels": labels, "priority": priority}
    result = {"train": [], "test": []}
    seen = set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            familiar_template = None
            if section == "test" and index == 0 and float(rng.random()) < 0.36:
                source_pair = result["train"][int(rng.integers(train_count))]
                familiar_template = _template_from_pair(source_pair)
            for _ in range(64):
                child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                force_tie = tie_enabled and ((section == "train" and index == 0) or (section == "train" and index > 0 and float(rng.random()) < 0.22) or (section == "test" and float(rng.random()) < 0.48))
                forced_pattern = None
                if section == "train" and index == count - 1:
                    wins = _training_winner_indices(result["train"])
                    if len(wins) == 1:
                        forced_pattern = "all1" if 0 in wins else "all0"
                        if force_tie and conventions["labels"] is not None and conventions["labels"].index(priority) != int(forced_pattern[-1]):
                            force_tie = False
                pair = _pair(child, "source_train" if section == "train" else "target_test", conventions, force_tie, forced_pattern, familiar_template)
                key = _array_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    result[section].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(result)
    return result


def _training_winner_indices(pairs):
    indices = set()
    for pair in pairs:
        input_grid, output_grid = pair["input"], pair["output"]
        background = next(iter(set(int(x) for x in np.unique(input_grid)) - set(int(x) for x in np.unique(output_grid))))
        geom = _analyze_geometry(input_grid, background)
        structure = max((int(x) for x in np.unique(input_grid[geom["mask"]])), key=lambda x: int(np.count_nonzero(input_grid[geom["mask"]] == x)))
        labels = sorted(set(int(x) for x in np.unique(input_grid)) - {background, structure})
        for panel in geom["panels"]:
            r0, _, c0, _ = panel
            indices.add(labels.index(int(output_grid[r0, c0])))
    return indices


def _template_from_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    background = next(iter(set(int(x) for x in np.unique(input_grid)) - set(int(x) for x in np.unique(output_grid))))
    geom = _analyze_geometry(input_grid, background)
    row_sizes = [end - start + 1 for start, end in geom["rows"]]
    column_sizes = [end - start + 1 for start, end in geom["columns"]]
    edges = (
        0 in geom["hbars"], input_grid.shape[0] - 1 in geom["hbars"],
        0 in geom["vbars"], input_grid.shape[1] - 1 in geom["vbars"],
    )
    return row_sizes, column_sizes, edges


def _validate_and_describe(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "sections must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_wins, train_ties, test_ties = set(), [], []
    pair_label_roles = []
    features = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer), "grid must be 2D integer array")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid size")
                _require(np.all((0 <= grid) & (grid <= 9)), "illegal ARC color")
            _require(input_grid.shape == output_grid.shape, "shape changed")
            input_key, output_key = _array_key(input_grid), _array_key(output_grid)
            _require(input_key not in seen_inputs and (input_key, output_key) not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            _require(_rows_signature(input_grid) not in _SOURCE_INPUT_ROWS, "copied source input")
            input_colors = set(int(x) for x in np.unique(input_grid))
            output_colors = set(int(x) for x in np.unique(output_grid))
            missing = input_colors - output_colors
            _require(len(input_colors) == 4 and len(missing) == 1, "color roles not visible")
            background = next(iter(missing))
            geom = _analyze_geometry(input_grid, background)
            mask = geom["mask"]
            bar_colors = {int(x): int(np.count_nonzero(input_grid[mask] == x)) for x in np.unique(input_grid[mask])}
            structure = max(bar_colors, key=bar_colors.get)
            labels = tuple(sorted(input_colors - {background, structure}))
            pair_label_roles.append(labels)
            _require(len(labels) == 2 and all(bar_colors.get(label, 0) > 0 for label in labels), "two marker roles required")
            marker_count = sum(bar_colors[label] for label in labels)
            _require(marker_count < bar_colors[structure], "markers must be sparse")
            for row in geom["hbars"]:
                for column in geom["vbars"]:
                    _require(input_grid[row, column] == structure, "bar intersection replaced")
            _require(np.array_equal(output_grid[mask], input_grid[mask]), "bar cell altered")
            _require(not np.any(output_grid == background), "blank cell left unfilled")
            local_wins = []
            for panel in geom["panels"]:
                r0, r1, c0, c1 = panel
                region = output_grid[r0:r1 + 1, c0:c1 + 1]
                values = np.unique(region)
                _require(len(values) == 1 and int(values[0]) in labels, "panel fill not uniform marker color")
                winner = int(values[0])
                index = labels.index(winner)
                local_wins.append(index)
                first, second = _votes(input_grid, _boundary(panel, input_grid.shape), labels)
                _require(first + second > 0, "panel has no vote")
                if first == second:
                    (train_ties if section == "train" else test_ties).append((labels, winner, first))
                else:
                    _require(index == int(second > first), "panel fails strict majority")
                if section == "train":
                    train_wins.add(index)
            features.append((section, len(geom["rows"]), len(geom["columns"]), tuple(int(0 in bars) + 2 * int(length - 1 in bars) for bars, length in ((geom["hbars"], input_grid.shape[0]), (geom["vbars"], input_grid.shape[1]))), tuple(local_wins), input_grid.shape))
    _require(train_wins == {0, 1}, "training must show both winning roles")
    if train_ties or test_ties:
        _require(train_ties, "test tie has no training demonstration")
        tie_labels = train_ties[0][0]
        tie_priority = train_ties[0][1]
        _require(all(labels == tie_labels for labels in pair_label_roles), "tie roles must stay recognizable across pairs")
        _require(all(labels == tie_labels and winner == tie_priority for labels, winner, _ in train_ties + test_ties), "inconsistent tie convention")
    return {"features": features, "train_ties": train_ties, "test_ties": test_ties}


def validate_generated_puzzle(puzzle: dict) -> None:
    _validate_and_describe(puzzle)


def _review_features(pair):
    grid, output = pair["input"], pair["output"]
    background = next(iter(set(int(x) for x in np.unique(grid)) - set(int(x) for x in np.unique(output))))
    geom = _analyze_geometry(grid, background)
    mask = geom["mask"]
    structure = max((int(x) for x in np.unique(grid[mask])), key=lambda color: int(np.count_nonzero(grid[mask] == color)))
    labels = tuple(sorted(set(int(x) for x in np.unique(grid)) - {background, structure}))
    row_sizes = [end - start + 1 for start, end in geom["rows"]]
    column_sizes = [end - start + 1 for start, end in geom["columns"]]
    edges = (0 in geom["hbars"], grid.shape[0] - 1 in geom["hbars"], 0 in geom["vbars"], grid.shape[1] - 1 in geom["vbars"])
    winners = []
    measures = Counter()
    owners = Counter(cell for panel in geom["panels"] for cell in _boundary(panel, grid.shape))
    marker_cells = {cell for cell in owners if int(grid[cell]) in labels}
    measures["isolated_markers"] = sum(not any((row + dr, column + dc) in marker_cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))) for row, column in marker_cells)
    measures["grouped_markers"] = len(marker_cells) - measures["isolated_markers"]
    measures["shared_markers"] = sum(owners[cell] == 2 for cell in marker_cells)
    measures["exterior_markers"] = sum(owners[cell] == 1 for cell in marker_cells)
    for axis in (row_sizes, column_sizes):
        if max(axis) >= 9:
            for position, size in enumerate(axis):
                if size == max(axis):
                    measures["wide_" + ("first" if position == 0 else "last" if position == len(axis) - 1 else "middle")] += 1
    for panel in geom["panels"]:
        r0, r1, c0, c1 = panel
        winner = int(output[r0, c0])
        winners.append(labels.index(winner))
        first, second = _votes(grid, _boundary(panel, grid.shape), labels)
        measures["margin_one"] += int(abs(first - second) == 1)
        measures["margin_three"] += int(abs(first - second) >= 3)
        measures["ties"] += int(first == second)
        side_choices = []
        side_cells = []
        if r0:
            side_cells.append([(r0 - 1, column) for column in range(c0, c1 + 1)])
        if r1 + 1 < grid.shape[0]:
            side_cells.append([(r1 + 1, column) for column in range(c0, c1 + 1)])
        if c0:
            side_cells.append([(row, c0 - 1) for row in range(r0, r1 + 1)])
        if c1 + 1 < grid.shape[1]:
            side_cells.append([(row, c1 + 1) for row in range(r0, r1 + 1)])
        for side in side_cells:
            votes = [sum(int(grid[cell] == label) for cell in side) for label in labels]
            if votes[0] != votes[1]:
                side_choices.append(int(votes[1] > votes[0]))
        measures["side_conflicts"] += int(len(set(side_choices)) > 1)
    rows, columns = len(row_sizes), len(column_sizes)
    measures["all_one"] = int(len(set(winners)) == 1)
    measures["alternating"] = int(all(winners[row * columns + col] != winners[row * columns + col + 1] for row in range(rows) for col in range(columns - 1)) and all(winners[row * columns + col] != winners[(row + 1) * columns + col] for row in range(rows - 1) for col in range(columns)))
    same_neighbors = sum(winners[row * columns + col] == winners[row * columns + col + 1] for row in range(rows) for col in range(columns - 1)) + sum(winners[row * columns + col] == winners[(row + 1) * columns + col] for row in range(rows - 1) for col in range(columns))
    measures["clustered"] = int(len(set(winners)) == 2 and same_neighbors >= 2)
    return measures, (rows, columns), (tuple(row_sizes), tuple(column_sizes)), edges, (background, structure, labels)


def self_test() -> None:
    a = generate_puzzle(4419, 6, 2)
    _require(_puzzles_equal(a, generate_puzzle(4419, 6, 2)), "nondeterministic puzzle")
    _require(not _puzzles_equal(a, generate_puzzle(4420, 6, 2)), "seeds did not differ")
    for offset, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(12000 + offset, train, test)
        _require(train is None or len(puzzle["train"]) == train, "explicit train count ignored")
        _require(test is None or len(puzzle["test"]) == test, "explicit test count ignored")
    for offset, args in enumerate(((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4), (2.5, 1), (True, 1))):
        try:
            generate_puzzle(14000 + offset, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {args}")
    train_counts, test_counts, shapes, layouts, framing = set(), set(), set(), set(), set()
    strict_puzzles = tie_puzzles = all_one_pairs = near_limit = compact = 0
    review_totals = Counter()
    familiar_tests = new_test_lattices = mixed_label_puzzles = mixed_framing_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        descriptor = _validate_and_describe(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        strict_puzzles += int(not descriptor["train_ties"] and not descriptor["test_ties"])
        tie_puzzles += int(bool(descriptor["train_ties"]))
        train_lattices, test_lattices, train_sizes, test_sizes = set(), set(), set(), set()
        pair_labels, pair_frames = set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                measures, lattice, sizes, edges, roles = _review_features(pair)
                review_totals.update(measures)
                (train_lattices if section == "train" else test_lattices).add(lattice)
                (train_sizes if section == "train" else test_sizes).add(sizes)
                pair_labels.add(roles[2])
                pair_frames.add(edges)
        familiar_tests += int(bool(train_sizes & test_sizes))
        new_test_lattices += int(bool(test_lattices - train_lattices))
        mixed_label_puzzles += int(len(pair_labels) > 1)
        mixed_framing_puzzles += int(len(pair_frames) > 1)
        for section, rows, columns, edges, winners, shape in descriptor["features"]:
            shapes.add(shape)
            layouts.add((rows, columns))
            framing.add(edges)
            all_one_pairs += int(len(set(winners)) == 1)
            near_limit += int(max(shape) >= 26)
            compact += int(max(shape) <= 10)
        for pair in puzzle["test"]:
            input_grid, expected = pair["input"], pair["output"]
            background = next(iter(set(int(x) for x in np.unique(input_grid)) - set(int(x) for x in np.unique(expected))))
            geom = _analyze_geometry(input_grid, background)
            colors = {int(x): int(np.count_nonzero(input_grid[geom["mask"]] == x)) for x in np.unique(input_grid[geom["mask"]])}
            structure = max(colors, key=colors.get)
            labels = tuple(sorted(set(int(x) for x in np.unique(input_grid)) - {background, structure}))
            priority = descriptor["train_ties"][0][1] if descriptor["train_ties"] else labels[0]
            _require(np.array_equal(_solve(input_grid, background, labels, priority), expected), "visible-vote reconstruction failed")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack range")
    _require(len(shapes) >= 35 and len(framing) >= 9, "dimensions or framing lack diversity")
    _require({(2, 2), (2, 4), (4, 2)} <= layouts and any(r >= 4 and c >= 4 for r, c in layouts), "lattice diversity missing")
    _require(strict_puzzles >= 20 and tie_puzzles >= 3, "strict/tie family balance missing")
    _require(all_one_pairs >= 5 and near_limit >= 3 and compact >= 3, "winner/scale diversity missing")
    _require(all(review_totals["wide_" + position] > 0 for position in ("first", "middle", "last")), "wide panels do not move")
    _require(review_totals["margin_one"] > 100 and review_totals["margin_three"] > 100, "vote margins lack range")
    _require(review_totals["shared_markers"] > 100 and review_totals["exterior_markers"] > 100, "marker locations lack range")
    _require(review_totals["isolated_markers"] > 100 and review_totals["grouped_markers"] > 100, "marker grouping lacks range")
    _require(review_totals["side_conflicts"] > 20, "contrasting panel sides missing")
    _require(review_totals["clustered"] >= 5 and review_totals["alternating"] >= 5, "winner maps lack range")
    _require(familiar_tests >= 3 and new_test_lattices >= 10, "test layouts lack familiar or novel cases")
    _require(mixed_label_puzzles >= 10 and mixed_framing_puzzles >= 20, "within-puzzle roles or framing lack range")
