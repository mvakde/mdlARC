import numpy as np


_SOURCE_RAW = (
    (
        (
            "00000000000000",
            "00000000000000",
            "00006000600000",
            "00666000600000",
            "06600006666000",
            "00666000060000",
            "00000000000000",
            "00030000020000",
            "00000000000000",
            "00000000000000",
            "00000000000000",
            "00000000000000",
        ),
        (
            "00030200",
            "03330200",
            "33002222",
            "03330020",
        ),
    ),
    (
        (
            "000000000",
            "006000000",
            "066660800",
            "000600000",
            "006600000",
            "000000000",
            "000000000",
            "000666000",
            "000600010",
            "006660000",
            "000060000",
            "000000000",
        ),
        (
            "0800",
            "8888",
            "0080",
            "0880",
            "0111",
            "0100",
            "1110",
            "0010",
        ),
    ),
    (
        (
            "000000000000000000",
            "006000000000000000",
            "066660000000060060",
            "006600000000066660",
            "006000006000006600",
            "000000066000000660",
            "001000006000000000",
            "000000066660000400",
            "000000000000000000",
            "000000003000000000",
            "000000000000000000",
            "000000000000000000",
        ),
        (
            "010003004004",
            "111133004444",
            "011003000440",
            "010033330044",
        ),
    ),
    (
        (
            "0000000000000",
            "0000006600000",
            "0000066000100",
            "0000006660000",
            "0000000060000",
            "0000000000000",
            "0000000000000",
            "0006000000000",
            "0006666070000",
            "0006000000000",
            "0006600000000",
            "0000000000000",
            "0000000000000",
            "0000006660000",
            "0000006066000",
            "0000006000030",
            "0000006660000",
            "0000000000000",
        ),
        (
            "0110",
            "1100",
            "0111",
            "0001",
            "7000",
            "7777",
            "7000",
            "7700",
            "3330",
            "3033",
            "3000",
            "3330",
        ),
    ),
)


def _rows_to_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (_rows_to_grid(input_rows), _rows_to_grid(output_rows))
    for input_rows, output_rows in _SOURCE_RAW
)


def _validate_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_train = int(num_train)
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError(
                "no legal test count for the requested train count"
            )
        num_test = int(rng.integers(1, maximum_test + 1))
    else:
        num_train = int(num_train)
        num_test = int(num_test)

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False

    start = tuple(int(value) for value in points[0])
    seen = {start}
    stack = [start]
    height, width = mask.shape

    while stack:
        row, column = stack.pop()
        for delta_row, delta_column in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            next_row = row + delta_row
            next_column = column + delta_column
            point = (next_row, next_column)

            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and mask[next_row, next_column]
                and point not in seen
            ):
                seen.add(point)
                stack.append(point)

    return len(seen) == int(mask.sum())


def _make_mask(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    area = height * width
    mode = float(rng.random())
    if mode < 0.035:
        return np.ones((height, width), dtype=bool)
    if mode < 0.54:
        low, high = 0.34, 0.57
    elif mode < 0.90:
        low, high = 0.57, 0.75
    else:
        low, high = 0.75, 0.91
    target = max(height + width - 1, int(round(float(rng.uniform(low, high)) * area)))
    target = min(target, area - 1)
    for _ in range(160):
        anchors = [
            (0, int(rng.integers(width))),
            (height - 1, int(rng.integers(width))),
            (int(rng.integers(height)), 0),
            (int(rng.integers(height)), width - 1),
        ]
        rng.shuffle(anchors)
        mask = np.zeros((height, width), dtype=bool)
        mask[anchors[0]] = True
        for endpoint in anchors[1:]:
            occupied = np.argwhere(mask)
            distances = np.abs(occupied - np.array(endpoint)).sum(axis=1)
            nearest = occupied[np.flatnonzero(distances == distances.min())]
            start = nearest[int(rng.integers(len(nearest)))]
            row, col = int(start[0]), int(start[1])
            while (row, col) != endpoint:
                moves = []
                if row != endpoint[0]:
                    moves.append((row + (1 if endpoint[0] > row else -1), col))
                if col != endpoint[1]:
                    moves.append((row, col + (1 if endpoint[1] > col else -1)))
                row, col = moves[int(rng.integers(len(moves)))]
                mask[row, col] = True
        if int(mask.sum()) > target:
            continue
        while int(mask.sum()) < target:
            frontier = set()
            for row, col in np.argwhere(mask):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = int(row) + dr, int(col) + dc
                    if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                        frontier.add((rr, cc))
            choices = sorted(frontier)
            mask[choices[int(rng.integers(len(choices)))]] = True
        return mask
    # A simple connected cross guarantees a tight sparse crop in the rare fallback.
    mask = np.zeros((height, width), dtype=bool)
    row, col = int(rng.integers(height)), int(rng.integers(width))
    mask[row, :] = True
    mask[:, col] = True
    return mask


def _components(binary: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = binary.shape
    seen = np.zeros((height, width), dtype=bool)
    result = []
    for r, c in np.argwhere(binary):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        component = []
        while stack:
            row, col = stack.pop()
            component.append((row, col))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width and binary[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(component)
    return result


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    values, counts = np.unique(input_grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    candidate_carriers = []
    for value in values:
        color = int(value)
        if color == background:
            continue
        parts = _components(input_grid == color)
        if len(parts) >= 2 and all(len(part) >= 5 for part in parts):
            candidate_carriers.append((color, parts))
    assert len(candidate_carriers) == 1
    carrier, components = candidate_carriers[0]
    assert 2 <= len(components) <= 5
    objects = []
    shape = None
    for part in components:
        rows = [p[0] for p in part]
        cols = [p[1] for p in part]
        top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
        crop = input_grid[top:bottom + 1, left:right + 1]
        assert np.all((crop == background) | (crop == carrier))
        mask = crop == carrier
        assert _connected(mask)
        assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
        if shape is None:
            shape = crop.shape
        assert crop.shape == shape
        objects.append(((top, bottom, left, right), mask))
    assert shape is not None and 3 <= shape[0] <= 5 and 3 <= shape[1] <= 5
    marker_locations = np.argwhere((input_grid != background) & (input_grid != carrier))
    assert len(marker_locations) == len(objects)
    markers = []
    for r, c in marker_locations:
        r, c = int(r), int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < input_grid.shape[0] and 0 <= cc < input_grid.shape[1]:
                    assert input_grid[rr, cc] == background
        markers.append((r, c, int(input_grid[r, c])))
    interpretations = []
    for side in ("above", "below", "left", "right"):
        assignment = []
        distances = []
        for bbox, _ in objects:
            top, bottom, left, right = bbox
            candidates = []
            for index, (r, c, _) in enumerate(markers):
                if side == "above" and left <= c <= right and 2 <= top - r <= 5:
                    candidates.append((index, top - r))
                elif side == "below" and left <= c <= right and 2 <= r - bottom <= 5:
                    candidates.append((index, r - bottom))
                elif side == "left" and top <= r <= bottom and 2 <= left - c <= 5:
                    candidates.append((index, left - c))
                elif side == "right" and top <= r <= bottom and 2 <= c - right <= 5:
                    candidates.append((index, c - right))
            if len(candidates) != 1:
                break
            assignment.append(candidates[0][0])
            distances.append(candidates[0][1])
        if len(assignment) == len(objects) and len(set(assignment)) == len(markers):
            interpretations.append((side, assignment, distances))
    assert len(interpretations) == 1
    side, assignment, distances = interpretations[0]
    orientation = "horizontal" if side in ("above", "below") else "vertical"
    order = sorted(range(len(objects)), key=lambda i: objects[i][0][2 if orientation == "horizontal" else 0])
    gaps = []
    for a, b in zip(order, order[1:]):
        box_a, box_b = objects[a][0], objects[b][0]
        gap = box_b[2] - box_a[3] - 1 if orientation == "horizontal" else box_b[0] - box_a[1] - 1
        assert gap >= 2
        gaps.append(gap)
    tiles = []
    colors = []
    bboxes = []
    masks = []
    densities = []
    for index in order:
        bbox, mask = objects[index]
        color = markers[assignment[index]][2]
        tiles.append(np.where(mask, color, background).astype(np.int8))
        colors.append(color)
        bboxes.append(bbox)
        masks.append(mask.tobytes())
        densities.append(float(mask.mean()))
    expected = np.concatenate(tiles, axis=1 if orientation == "horizontal" else 0)
    assert np.array_equal(output_grid, expected)
    output_colors = set(int(x) for x in np.unique(output_grid))
    assert set(colors) <= output_colors <= {background, *colors}
    assert carrier not in set(int(x) for x in np.unique(output_grid))
    stagger = [b[0] for b in bboxes] if orientation == "horizontal" else [b[2] for b in bboxes]
    return {
        "background": background, "carrier": carrier, "tile_shape": tuple(int(x) for x in shape),
        "side": side, "orientation": orientation, "object_count": len(objects),
        "marker_colors": tuple(colors), "bboxes": tuple(bboxes), "mask_signatures": tuple(masks),
        "densities": tuple(densities), "stagger": max(stagger) - min(stagger),
        "gaps": tuple(gaps), "marker_distances": tuple(distances),
    }


def _grid_signature(grid: np.ndarray) -> tuple:
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _pair_is_source_duplicate(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return any(np.array_equal(input_grid, a) and np.array_equal(output_grid, b) for a, b in _SOURCE_PAIRS)


def _make_pair(rng: np.random.Generator, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    height, width = spec["tile_shape"]
    count = spec["object_count"]
    orientation = spec["orientation"]
    side = spec["side"]
    background, carrier = spec["background"], spec["carrier"]
    colors = spec["marker_colors"]
    assert len(colors) == count
    masks = []
    for index in range(count):
        if index and spec.get("repeat_mask") and index == 1:
            masks.append(masks[0].copy())
        else:
            masks.append(_make_mask(rng, height, width))
    for _ in range(100):
        gap_high = 4 if count <= 3 else 3
        gaps = [int(rng.integers(2, gap_high + 1)) for _ in range(count - 1)]
        spread = int(rng.integers(3, 8))
        offsets = [int(rng.integers(spread + 1)) for _ in range(count)]
        origins = []
        markers = []
        cursor = 0
        for index in range(count):
            if orientation == "horizontal":
                row, col = offsets[index], cursor
                mr = row - int(rng.integers(2, 6)) if side == "above" else row + height - 1 + int(rng.integers(2, 6))
                mc = col + int(rng.integers(width))
                cursor += width
            else:
                row, col = cursor, offsets[index]
                mr = row + int(rng.integers(height))
                mc = col - int(rng.integers(2, 6)) if side == "left" else col + width - 1 + int(rng.integers(2, 6))
                cursor += height
            origins.append((row, col))
            markers.append((mr, mc))
            if index < count - 1:
                cursor += gaps[index]
        minrow = min([r for r, _ in origins] + [r for r, _ in markers])
        mincol = min([c for _, c in origins] + [c for _, c in markers])
        maxrow = max([r + height - 1 for r, _ in origins] + [r for r, _ in markers])
        maxcol = max([c + width - 1 for _, c in origins] + [c for _, c in markers])
        margins = [int(rng.integers(0, 5)) for _ in range(4)]
        top, bottom, left, right = margins
        grid_h = maxrow - minrow + 1 + top + bottom
        grid_w = maxcol - mincol + 1 + left + right
        if grid_h < spec.get("min_height", 0):
            extra = spec["min_height"] - grid_h
            bottom += extra
            grid_h += extra
        if not (grid_h <= 30 and grid_w <= 30):
            continue
        row_shift, col_shift = top - minrow, left - mincol
        input_grid = np.full((grid_h, grid_w), background, dtype=np.int8)
        for mask, (row, col) in zip(masks, origins):
            crop = input_grid[row + row_shift:row + row_shift + height, col + col_shift:col + col_shift + width]
            if not np.all(crop == background):
                break
            crop[mask] = carrier
        else:
            for color, (row, col) in zip(colors, markers):
                rr, cc = row + row_shift, col + col_shift
                if input_grid[rr, cc] != background:
                    break
                input_grid[rr, cc] = color
            else:
                tiles = [np.where(mask, color, background).astype(np.int8) for mask, color in zip(masks, colors)]
                output_grid = np.concatenate(tiles, axis=1 if orientation == "horizontal" else 0)
                try:
                    _analyze_pair(input_grid, output_grid)
                except AssertionError:
                    continue
                return input_grid, output_grid
    raise RuntimeError("could not place an unambiguous pair")


def _pick_colors(rng: np.random.Generator, pool: list[int], count: int,
                 repeated: bool = False, forced: int | None = None) -> tuple[int, ...]:
    choices = list(int(x) for x in pool)
    assert choices and (not repeated or count >= 2)
    if repeated:
        repeated_color = int(forced) if forced is not None else choices[int(rng.integers(len(choices)))]
        result = [repeated_color, repeated_color]
        remaining = [x for x in choices if x != repeated_color]
        while len(result) < count:
            result.append(int(remaining[int(rng.integers(len(remaining)))]) if remaining else repeated_color)
    else:
        if forced is not None:
            remaining = [x for x in choices if x != int(forced)]
            result = [int(forced)] + [int(x) for x in rng.choice(remaining, size=count - 1, replace=False)]
        else:
            result = [int(x) for x in rng.choice(choices, size=count, replace=False)]
    rng.shuffle(result)
    return tuple(result)


def _choose_shape(rng: np.random.Generator, count: int, orientation: str) -> tuple[int, int]:
    height, width = int(rng.integers(3, 6)), int(rng.integers(3, 6))
    if count >= 4:
        if orientation == "horizontal":
            width = int(rng.integers(3, 5)) if count == 4 else 3
        else:
            height = int(rng.integers(3, 5)) if count == 4 else 3
    return height, width


def _choose_count(rng: np.random.Generator, test: bool) -> int:
    probabilities = (0.18, 0.43, 0.29, 0.10) if test else (0.35, 0.37, 0.22, 0.06)
    return int(rng.choice(np.array([2, 3, 4, 5]), p=probabilities))


def _roles(rng: np.random.Generator, available: list[int]) -> tuple[int, int]:
    x = rng.choice(available, size=2, replace=False)
    return int(x[0]), int(x[1])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            signature = _grid_signature(pair["input"])
            pair_signature = (signature, _grid_signature(pair["output"]))
            assert signature not in inputs and pair_signature not in pairs
            inputs.add(signature)
            pairs.add(pair_signature)
            assert not np.array_equal(pair["input"], pair["output"])
            assert not _pair_is_source_duplicate(pair["input"], pair["output"])
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))
    train = analyses["train"]
    test = analyses["test"]
    # Both packing rules are visible in training, including the one used by each test.
    assert {a["orientation"] for a in train} == {"horizontal", "vertical"}
    train_sides = {a["side"] for a in train}
    assert all(a["side"] in train_sides for a in test)
    if any(len(set(a["marker_colors"])) < a["object_count"] for a in test):
        assert any(len(set(a["marker_colors"])) < a["object_count"] for a in train)
    train_input_colors = set().union(*(set(int(x) for x in np.unique(p["input"])) for p in puzzle["train"]))
    for a in test:
        assert a["object_count"] in (2, 3, 4, 5)
        assert a["orientation"] in {t["orientation"] for t in train}
        # New hues are allowed because training demonstrates direct marker-to-output transfer.
        assert a["background"] in range(10) and a["carrier"] in range(10)
    assert len(train_input_colors) >= 3


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    palette = [int(x) for x in rng.permutation(10)]
    reserved = palette[0]
    normal = palette[1:]
    base_roles = (normal[0], normal[1])
    base_shape = (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
    role_mode = "variable" if rng.random() < 0.60 else "fixed"
    shape_mode = "variable" if rng.random() < 0.63 else "fixed"
    regime = str(rng.choice(["classic", "local", "novel", "mixed"], p=[0.18, 0.26, 0.22, 0.34]))
    repeated_mode = regime != "classic" and rng.random() < 0.38
    orientation_order = ["horizontal", "vertical"]
    rng.shuffle(orientation_order)
    orientation_order += [str(rng.choice(["horizontal", "vertical"])) for _ in range(num_train - 2)]
    if regime == "classic":
        orientation_order = ["horizontal", "vertical"] + ["horizontal" if i == 2 else str(rng.choice(["horizontal", "vertical"])) for i in range(2, num_train)]
        role_mode = shape_mode = "fixed"
        repeated_mode = False
    side_seen = {"horizontal": [], "vertical": []}
    fixed_sides = {"horizontal": str(rng.choice(["above", "below"])),
                   "vertical": str(rng.choice(["left", "right"]))}
    if regime == "classic":
        fixed_sides = {"horizontal": "below", "vertical": "right"}
    puzzle = {"train": [], "test": []}
    specs = []
    used_inputs = set()
    for index, orientation in enumerate(orientation_order):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        count = (2 if index < 2 else 3 if index == 2 and orientation == "horizontal" else _choose_count(pair_rng, False)) if regime == "classic" else _choose_count(pair_rng, False)
        if role_mode == "fixed":
            background, carrier = base_roles
        elif index == 0:
            background, carrier = base_roles
        elif index == 1:
            background, carrier = base_roles[::-1]
        else:
            background, carrier = _roles(pair_rng, normal)
        if shape_mode == "fixed" and count <= 3:
            shape = base_shape
        elif shape_mode == "variable" and index == 0:
            shape = base_shape if count <= 3 else _choose_shape(pair_rng, count, orientation)
        elif shape_mode == "variable" and index == 1 and count <= 3:
            choices = [(h, w) for h in range(3, 6) for w in range(3, 6) if (h, w) != specs[0]["tile_shape"]]
            shape = choices[int(pair_rng.integers(len(choices)))]
        else:
            shape = _choose_shape(pair_rng, count, orientation)
        if count >= 4:
            shape = _choose_shape(pair_rng, count, orientation)
        if regime == "classic":
            side = fixed_sides[orientation]
        elif side_seen[orientation] and pair_rng.random() < 0.42:
            side = {"above": "below", "below": "above", "left": "right", "right": "left"}[side_seen[orientation][0]]
        else:
            side = fixed_sides[orientation]
        side_seen[orientation].append(side)
        pool = [x for x in normal if x not in (background, carrier)]
        repeated = repeated_mode and index == 0 or repeated_mode and index > 0 and pair_rng.random() < 0.25
        colors = _pick_colors(pair_rng, pool, count, repeated)
        spec = {"background": background, "carrier": carrier, "tile_shape": shape,
                "orientation": orientation, "side": side, "object_count": count,
                "marker_colors": colors, "repeat_mask": pair_rng.random() < 0.14}
        for _ in range(8):
            input_grid, output_grid = _make_pair(pair_rng, spec)
            signature = _grid_signature(input_grid)
            if signature not in used_inputs and not _pair_is_source_duplicate(input_grid, output_grid):
                break
        else:
            raise RuntimeError("could not make a unique training pair")
        used_inputs.add(signature)
        puzzle["train"].append({"input": input_grid, "output": output_grid})
        specs.append(spec)
    seen_marker_colors = {
        color
        for pair in puzzle["train"]
        for color in _analyze_pair(pair["input"], pair["output"])["marker_colors"]
    }
    for index in range(num_test):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        mode = ("classic" if regime == "classic" else regime if index == 0 else str(pair_rng.choice(["local", "novel", "mixed"])))
        if mode == "local":
            exemplar = specs[int(pair_rng.integers(len(specs)))]
            spec = {**exemplar, "repeat_mask": pair_rng.random() < 0.18}
        else:
            if mode == "classic":
                orientation, count = "vertical", 3
                exemplar = next(s for s in specs if s["orientation"] == "vertical")
                background, carrier = base_roles
                shape = base_shape
                side = exemplar["side"]
            else:
                orientation = str(pair_rng.choice(["horizontal", "vertical"]))
                count = _choose_count(pair_rng, True)
                exemplar = specs[int(pair_rng.integers(len(specs)))]
                if role_mode == "fixed" or pair_rng.random() < 0.45:
                    background, carrier = exemplar["background"], exemplar["carrier"]
                else:
                    background, carrier = _roles(pair_rng, normal)
                shape = _choose_shape(pair_rng, count, orientation) if shape_mode == "variable" or count >= 4 else base_shape
                side = str(pair_rng.choice(side_seen[orientation]))
            pool = [x for x in normal if x not in (background, carrier)]
            novel = mode == "classic" or mode == "novel" or mode == "mixed" and pair_rng.random() < 0.48
            if novel:
                pool.append(reserved)
            else:
                familiar = [x for x in pool if x in seen_marker_colors]
                if familiar:
                    pool = familiar + [x for x in pool if x not in familiar]
            repeated = repeated_mode and pair_rng.random() < 0.48
            forced = reserved if novel else (familiar[int(pair_rng.integers(len(familiar)))] if not repeated and familiar else None)
            colors = _pick_colors(pair_rng, pool, count, repeated, forced)
            spec = {"background": background, "carrier": carrier, "tile_shape": shape,
                    "orientation": orientation, "side": side, "object_count": count,
                    "marker_colors": colors, "repeat_mask": pair_rng.random() < 0.18}
            if mode == "classic":
                maximum_train_height = max(pair["input"].shape[0] for pair in puzzle["train"])
                if maximum_train_height < 30:
                    spec["min_height"] = maximum_train_height + 1
        for _ in range(8):
            input_grid, output_grid = _make_pair(pair_rng, spec)
            signature = _grid_signature(input_grid)
            if signature not in used_inputs and not _pair_is_source_duplicate(input_grid, output_grid):
                break
        else:
            raise RuntimeError("could not make a unique test pair")
        used_inputs.add(signature)
        puzzle["test"].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((split, tuple((_grid_signature(p["input"]), _grid_signature(p["output"])) for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    analyses = [[_analyze_pair(pair["input"], pair["output"]) for split in ("train", "test") for pair in p[split]] for p in puzzles]
    train_analyses = [[_analyze_pair(pair["input"], pair["output"]) for pair in p["train"]] for p in puzzles]
    test_analyses = [[_analyze_pair(pair["input"], pair["output"]) for pair in p["test"]] for p in puzzles]
    flat = [a for p in analyses for a in p]
    role_pairs = {(a["background"], a["carrier"]) for a in flat}
    shapes = {a["tile_shape"] for a in flat}
    layouts = {(a["orientation"], a["side"], a["object_count"], a["bboxes"]) for a in flat}
    silhouettes = {(a["tile_shape"], mask) for a in flat for mask in a["mask_signatures"]}
    assert len(role_pairs) >= 20 and len(shapes) >= 6
    assert len(layouts) >= 150 and len(silhouettes) >= 150
    assert {a["side"] for a in flat} == {"above", "below", "left", "right"}
    assert {a["orientation"] for p in test_analyses for a in p} == {"horizontal", "vertical"}
    assert {a["object_count"] for a in flat} == {2, 3, 4, 5}
    assert {distance for a in flat for distance in a["marker_distances"]} == {2, 3, 4, 5}
    assert {gap for a in flat for gap in a["gaps"]} >= {2, 3, 4}
    assert sum(a["stagger"] >= 3 for a in flat) >= 40
    assert sum(d <= 0.63 for a in flat for d in a["densities"]) >= 100
    assert sum(d > 0.75 for a in flat for d in a["densities"]) >= 40
    assert any(d == 1 for a in flat for d in a["densities"])
    assert sum(len(set(a["marker_colors"])) < a["object_count"] for p in train_analyses for a in p) >= 5
    assert sum(any(a["mask_signatures"][i] == a["mask_signatures"][j]
                   and a["marker_colors"][i] != a["marker_colors"][j]
                   for i in range(a["object_count"]) for j in range(i + 1, a["object_count"]))
               for a in flat) >= 5
    assert sum(len({(a["background"], a["carrier"]) for a in p}) > 1 for p in train_analyses) >= 5
    assert sum(len({a["tile_shape"] for a in p}) > 1 for p in train_analyses) >= 5
    assert any(len({a["side"] for a in p if a["orientation"] == "horizontal"}) > 1 for p in train_analyses)
    assert any(len({a["side"] for a in p if a["orientation"] == "vertical"}) > 1 for p in train_analyses)
    assert sum(a["orientation"] == "vertical" and a["object_count"] >= 3
               for p in train_analyses for a in p) >= 10
    assert sum(a["stagger"] >= 3 and any(d <= 0.63 for d in a["densities"])
               for a in flat) >= 20
    new_hue = familiar = same_factors = shorter_or_equal = taller = 0
    for puzzle, training, testing in zip(puzzles, train_analyses, test_analyses):
        training_colors = {color for a in training for color in a["marker_colors"]}
        maximum_training_height = max(pair["input"].shape[0] for pair in puzzle["train"])
        for pair, a in zip(puzzle["test"], testing):
            new_hue += any(color not in training_colors for color in a["marker_colors"])
            familiar += all(color in training_colors for color in a["marker_colors"])
            shorter_or_equal += pair["input"].shape[0] <= maximum_training_height
            taller += pair["input"].shape[0] > maximum_training_height
            same_factors += any(
                (a["background"], a["carrier"], a["tile_shape"], a["side"],
                 a["object_count"], a["marker_colors"])
                == (b["background"], b["carrier"], b["tile_shape"], b["side"],
                    b["object_count"], b["marker_colors"])
                for b in training
            )
    assert min(new_hue, familiar, same_factors, shorter_or_equal, taller) >= 3
    for seed, train_count, test_count in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                                          (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    assert _puzzle_signature(generate_puzzle(777, 7, 3)) == _puzzle_signature(generate_puzzle(777, 7, 3))
    assert _puzzle_signature(generate_puzzle(778, 7, 3)) != _puzzle_signature(generate_puzzle(779, 7, 3))
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(999, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {arguments}")
