import numpy as np


_SOURCE_PAIRS = (
    (
        (
            "11071011",
            "13011013",
            "00000000",
            "11011011",
            "11011011",
            "00000000",
            "11071011",
            "11011013",
        ),
        (
            "11071011",
            "13013013",
            "00000000",
            "11071011",
            "11011013",
            "00000000",
            "11071011",
            "11011013",
        ),
    ),
    (
        (
            "14011014",
            "11011011",
            "00000000",
            "11011011",
            "21011021",
            "00000000",
            "14011011",
            "11011011",
        ),
        (
            "14014014",
            "11011011",
            "00000000",
            "14011011",
            "21021021",
            "00000000",
            "14011011",
            "11011011",
        ),
    ),
    (
        (
            "11011011",
            "11011011",
            "00000000",
            "11011011",
            "31011031",
            "00000000",
            "11011011",
            "11011011",
        ),
        (
            "11011011",
            "11011011",
            "00000000",
            "11011011",
            "31031031",
            "00000000",
            "11011011",
            "11011011",
        ),
    ),
    (
        (
            "11011011",
            "18011018",
            "00000000",
            "11011011",
            "11011011",
            "00000000",
            "61011061",
            "11011018",
        ),
        (
            "11011011",
            "18018018",
            "00000000",
            "11011011",
            "11011018",
            "00000000",
            "61061061",
            "11011018",
        ),
    ),
)


def _checked_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _select_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(
            rng.integers(1, min(4, num_train - 1) + 1)
        )

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    geometry = _sample_geometry(rng)
    flexible_layout = bool(rng.integers(2))
    flexible_roles = bool(rng.integers(2))
    shuffled = [int(x) for x in rng.permutation(10)]
    roles = (tuple(shuffled[:2]), tuple(shuffled[2:4]))
    accent_pool = shuffled[4:] if flexible_roles else shuffled[2:]
    n_train_colors = int(rng.integers(2, 4))
    return {
        "geometry": geometry,
        "flexible_layout": flexible_layout,
        "flexible_roles": flexible_roles,
        "roles": roles,
        "train_colors": tuple(accent_pool[:n_train_colors]),
        "fresh_colors": tuple(accent_pool[n_train_colors:]),
    }


def _sample_geometry(rng: np.random.Generator) -> dict:
    if int(rng.integers(8)) == 0:
        return dict(panel_rows=3, panel_cols=3, panel_height=2,
                    panel_width=2, row_divider=1, col_divider=1)
    return {
        "panel_rows": int(rng.integers(3, 7)),
        "panel_cols": int(rng.integers(3, 7)),
        "panel_height": int(rng.integers(2, 4)),
        "panel_width": int(rng.integers(2, 4)),
        "row_divider": int(rng.integers(1, 3)),
        "col_divider": int(rng.integers(1, 3)),
    }


def _pair_parameters(
    rng: np.random.Generator, puzzle_params: dict,
    index: int, is_train: bool,
) -> dict:
    params = dict(puzzle_params["geometry"])
    if puzzle_params["flexible_layout"] and index != 0:
        for _ in range(20):
            candidate = _sample_geometry(rng)
            if candidate != puzzle_params["geometry"]:
                params = candidate
                break
    role_index = 0
    if puzzle_params["flexible_roles"] and index != 0:
        role_index = 1 if is_train and index == 1 else int(rng.integers(2))
    params["background"], params["carrier"] = puzzle_params["roles"][role_index]
    return params


def _separated_pair(
    size: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    choices = [
        (start, end)
        for start in range(size)
        for end in range(start + 2, size)
    ]
    interior = [pair for pair in choices if pair[0] > 0 and pair[1] < size - 1]
    if interior and int(rng.integers(3)) == 0:
        choices = interior
    return choices[int(rng.integers(len(choices)))]


def _component(
    rng: np.random.Generator, kind: str, params: dict,
    color: int, offset: tuple[int, int],
) -> dict:
    n_rows, n_cols = params["panel_rows"], params["panel_cols"]
    marks = set()
    filled = set()
    if kind in ("h", "pre_h", "corner", "cross"):
        left, right = _separated_pair(n_cols, rng)
        if kind == "pre_h" and n_cols >= 5:
            options = [(a, b) for a in range(n_cols) for b in range(a + 4, n_cols)]
            left, right = options[int(rng.integers(len(options)))]
        row = int(rng.integers(n_rows))
        if kind == "cross":
            row = int(rng.integers(1, n_rows - 1))
        marks.update(((row, left), (row, right)))
        filled.update((row, col) for col in range(left, right + 1))
        if kind == "pre_h" and right - left >= 4:
            marks.add((row, int(rng.integers(left + 2, right - 1))))
    if kind in ("v", "pre_v", "corner", "cross"):
        top, bottom = _separated_pair(n_rows, rng)
        if kind == "cross":
            choices = [(a, b) for a in range(row) for b in range(row + 1, n_rows)]
            top, bottom = choices[int(rng.integers(len(choices)))]
        if kind == "pre_v" and n_rows >= 5:
            options = [(a, b) for a in range(n_rows) for b in range(a + 4, n_rows)]
            top, bottom = options[int(rng.integers(len(options)))]
        col = int(rng.integers(n_cols))
        if kind == "corner":
            row = (top, bottom)[int(rng.integers(2))]
            col = (left, right)[int(rng.integers(2))]
            marks = {(row, left), (row, right), (top, col), (bottom, col)}
            filled = {(row, c) for c in range(left, right + 1)}
        elif kind == "cross":
            col = int(rng.integers(left + 1, right))
            marks.add((top, col))
            marks.add((bottom, col))
        else:
            marks.update(((top, col), (bottom, col)))
        filled.update((r, col) for r in range(top, bottom + 1))
        if kind == "pre_v" and bottom - top >= 4:
            marks.add((int(rng.integers(top + 2, bottom - 1)), col))
    if kind == "adjacent":
        if int(rng.integers(2)):
            row = int(rng.integers(n_rows))
            col = int(rng.integers(n_cols - 1))
            marks = {(row, col), (row, col + 1)}
        else:
            col = int(rng.integers(n_cols))
            row = int(rng.integers(n_rows - 1))
            marks = {(row, col), (row + 1, col)}
        filled = set(marks)
    return {"color": color, "offset": offset, "marks": marks, "filled": filled}


_MODES = {
    "h": ("h",), "v": ("v",), "corner": ("corner",),
    "cross": ("cross",), "pre_h": ("pre_h",), "pre_v": ("pre_v",),
    "corner_h": ("corner", "h"), "corner_v": ("corner", "v"),
    "two_h": ("h", "h"), "two_v": ("v", "v"),
    "mixed": ("h", "v"), "h_decoy": ("h", "adjacent"),
    "v_decoy": ("v", "adjacent"),
    "cross_h": ("cross", "h"), "cross_v": ("cross", "v"),
    "pre_h_v": ("pre_h", "v"), "pre_v_h": ("pre_v", "h"),
}


def _panel_runs(
    count: int,
    panel_size: int,
    divider_size: int,
) -> list[tuple[int, int]]:
    result = []
    start = 0

    for index in range(count):
        result.append((start, start + panel_size))
        start += panel_size
        if index + 1 < count:
            start += divider_size

    return result


def _render_pair(
    params: dict,
    components: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    rows = _panel_runs(
        params["panel_rows"],
        params["panel_height"],
        params["row_divider"],
    )
    cols = _panel_runs(
        params["panel_cols"],
        params["panel_width"],
        params["col_divider"],
    )

    input_grid = np.full(
        (rows[-1][1], cols[-1][1]),
        params["background"],
        dtype=np.int8,
    )

    for row_start, row_end in rows:
        for col_start, col_end in cols:
            input_grid[
                row_start:row_end,
                col_start:col_end,
            ] = params["carrier"]
    output_grid = input_grid.copy()
    requested = {}
    for component in components:
        color = component["color"]
        local_row, local_col = component["offset"]
        for panel_row, panel_col in component["filled"]:
            key = (panel_row, panel_col, local_row, local_col)
            if key in requested and requested[key] != color:
                raise ValueError("conflicting requested fills")
            requested[key] = color
            output_grid[rows[panel_row][0] + local_row,
                        cols[panel_col][0] + local_col] = color
    for component in components:
        color = component["color"]
        local_row, local_col = component["offset"]
        for panel_row, panel_col in component["marks"]:
            key = (panel_row, panel_col, local_row, local_col)
            if requested[key] != color:
                raise ValueError("conflicting input markers")
            input_grid[rows[panel_row][0] + local_row,
                       cols[panel_col][0] + local_col] = color
    if not np.any(input_grid != output_grid):
        raise ValueError("pair has no completion")
    if not np.array_equal(_solve_grid(input_grid), output_grid):
        raise ValueError("unintended marker connection")
    return input_grid, output_grid


def _runs(
    mask: np.ndarray,
    value: bool,
) -> list[tuple[int, int]]:
    result = []
    start = None

    for index, item in enumerate(mask):
        if bool(item) == value and start is None:
            start = index
        if bool(item) != value and start is not None:
            result.append((start, index))
            start = None

    if start is not None:
        result.append((start, len(mask)))

    return result


def _inspect_lattice(grid: np.ndarray) -> dict:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2

    candidates = []

    for value in np.unique(grid):
        row_mask = np.all(grid == value, axis=1)
        col_mask = np.all(grid == value, axis=0)

        if np.any(row_mask) and np.any(col_mask):
            candidates.append(
                (int(value), row_mask, col_mask)
            )

    assert len(candidates) == 1

    background, row_dividers, col_dividers = candidates[0]

    rows = _runs(row_dividers, False)
    cols = _runs(col_dividers, False)
    row_separators = _runs(row_dividers, True)
    col_separators = _runs(col_dividers, True)

    assert len(rows) >= 3
    assert len(cols) >= 3
    assert rows[0][0] == 0
    assert rows[-1][1] == grid.shape[0]
    assert cols[0][0] == 0
    assert cols[-1][1] == grid.shape[1]
    assert len(row_separators) == len(rows) - 1
    assert len(col_separators) == len(cols) - 1

    panel_heights = {
        end - start for start, end in rows
    }
    panel_widths = {
        end - start for start, end in cols
    }
    row_widths = {
        end - start for start, end in row_separators
    }
    col_widths = {
        end - start for start, end in col_separators
    }

    assert len(panel_heights) == 1
    assert len(panel_widths) == 1
    assert len(row_widths) == 1
    assert len(col_widths) == 1

    panel_height = next(iter(panel_heights))
    panel_width = next(iter(panel_widths))

    assert panel_height >= 2
    assert panel_width >= 2

    interior_values = np.concatenate(
        [
            grid[
                row_start:row_end,
                col_start:col_end,
            ].ravel()
            for row_start, row_end in rows
            for col_start, col_end in cols
        ]
    )

    assert not np.any(interior_values == background)

    values, counts = np.unique(
        interior_values,
        return_counts=True,
    )

    assert len(values) >= 2

    largest = np.max(counts)
    assert np.count_nonzero(counts == largest) == 1

    carrier = int(values[int(np.argmax(counts))])
    assert carrier != background

    signature = (
        len(rows),
        len(cols),
        panel_height,
        panel_width,
        next(iter(row_widths)),
        next(iter(col_widths)),
        background,
        carrier,
    )

    return {
        "rows": rows,
        "cols": cols,
        "background": background,
        "carrier": carrier,
        "signature": signature,
    }


def _marker_groups(
    grid: np.ndarray,
    info: dict,
) -> dict:
    groups = {}
    carrier = info["carrier"]

    for panel_row, row_bounds in enumerate(info["rows"]):
        row_start, row_end = row_bounds

        for panel_col, col_bounds in enumerate(
            info["cols"]
        ):
            col_start, col_end = col_bounds
            panel = grid[
                row_start:row_end,
                col_start:col_end,
            ]

            for local_row, local_col in np.argwhere(
                panel != carrier
            ):
                color = int(
                    panel[local_row, local_col]
                )
                assert color != info["background"]

                key = (
                    color,
                    int(local_row),
                    int(local_col),
                )
                groups.setdefault(key, []).append(
                    (panel_row, panel_col)
                )

    return groups


def _solve_grid(input_grid: np.ndarray) -> np.ndarray:
    info = _inspect_lattice(input_grid)
    groups = _marker_groups(input_grid, info)
    output = input_grid.copy()

    def paint(
        panel_row: int,
        panel_col: int,
        local_row: int,
        local_col: int,
        color: int,
    ) -> None:
        row = (
            info["rows"][panel_row][0] + local_row
        )
        col = (
            info["cols"][panel_col][0] + local_col
        )

        old = int(output[row, col])
        assert old in (info["carrier"], color)
        output[row, col] = color

    for key, positions in groups.items():
        color, local_row, local_col = key
        by_row = {}
        by_col = {}

        for panel_row, panel_col in positions:
            by_row.setdefault(
                panel_row, []
            ).append(panel_col)
            by_col.setdefault(
                panel_col, []
            ).append(panel_row)

        for panel_row, panel_cols in by_row.items():
            if len(set(panel_cols)) >= 2:
                for panel_col in range(
                    min(panel_cols),
                    max(panel_cols) + 1,
                ):
                    paint(
                        panel_row,
                        panel_col,
                        local_row,
                        local_col,
                        color,
                    )

        for panel_col, panel_rows in by_col.items():
            if len(set(panel_rows)) >= 2:
                for panel_row in range(
                    min(panel_rows),
                    max(panel_rows) + 1,
                ):
                    paint(
                        panel_row,
                        panel_col,
                        local_row,
                        local_col,
                        color,
                    )

    return output


def _independent_completion(input_grid: np.ndarray) -> np.ndarray:
    info = _inspect_lattice(input_grid)
    panels = np.stack([
        np.stack([
            input_grid[r0:r1, c0:c1]
            for c0, c1 in info["cols"]
        ])
        for r0, r1 in info["rows"]
    ])
    completed = panels.copy()
    for local_row in range(panels.shape[2]):
        for local_col in range(panels.shape[3]):
            cell_plane = panels[:, :, local_row, local_col]
            for color in np.unique(cell_plane):
                if int(color) == info["carrier"]:
                    continue
                for row in range(cell_plane.shape[0]):
                    marked = np.flatnonzero(cell_plane[row] == color)
                    if len(marked) >= 2:
                        completed[row, marked[0]:marked[-1] + 1, local_row, local_col] = color
                for col in range(cell_plane.shape[1]):
                    marked = np.flatnonzero(cell_plane[:, col] == color)
                    if len(marked) >= 2:
                        completed[marked[0]:marked[-1] + 1, col, local_row, local_col] = color
    result = input_grid.copy()
    for row, (r0, r1) in enumerate(info["rows"]):
        for col, (c0, c1) in enumerate(info["cols"]):
            result[r0:r1, c0:c1] = completed[row, col]
    return result


def _pair_features(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    info = _inspect_lattice(input_grid)
    groups = _marker_groups(input_grid, info)

    horizontal = 0
    vertical = 0
    horizontal_members = {}
    vertical_members = {}
    all_positions = set()
    participating = set()
    premarked = 0
    empty_crossings = 0
    interior_spans = 0
    adjacent_groups = 0

    for key, positions in groups.items():
        all_positions.update(
            (key, position) for position in positions
        )

        h_members = set()
        v_members = set()
        h_lines = []
        v_lines = []
        by_row = {}
        by_col = {}

        for panel_row, panel_col in positions:
            by_row.setdefault(
                panel_row, []
            ).append(panel_col)
            by_col.setdefault(
                panel_col, []
            ).append(panel_row)

        for panel_row, panel_cols in by_row.items():
            unique_cols = set(panel_cols)
            if (
                len(unique_cols) >= 2
                and max(unique_cols) - min(unique_cols) >= 2
            ):
                horizontal += 1
                h_lines.append((panel_row, min(unique_cols), max(unique_cols)))
                if len(unique_cols) >= 3 and max(unique_cols) - min(unique_cols) + 1 > len(unique_cols):
                    premarked += 1
                if min(unique_cols) > 0 and max(unique_cols) < len(info["cols"]) - 1:
                    interior_spans += 1
                h_members.update(
                    (panel_row, col)
                    for col in unique_cols
                )

        for panel_col, panel_rows in by_col.items():
            unique_rows = set(panel_rows)
            if (
                len(unique_rows) >= 2
                and max(unique_rows) - min(unique_rows) >= 2
            ):
                vertical += 1
                v_lines.append((panel_col, min(unique_rows), max(unique_rows)))
                if len(unique_rows) >= 3 and max(unique_rows) - min(unique_rows) + 1 > len(unique_rows):
                    premarked += 1
                if min(unique_rows) > 0 and max(unique_rows) < len(info["rows"]) - 1:
                    interior_spans += 1
                v_members.update(
                    (row, panel_col)
                    for row in unique_rows
                )

        horizontal_members[key] = h_members
        vertical_members[key] = v_members

        for row, left, right in h_lines:
            for col, top, bottom in v_lines:
                if top <= row <= bottom and left <= col <= right and (row, col) not in positions:
                    empty_crossings += 1
        if len(positions) == 2:
            (r0, c0), (r1, c1) = positions
            if (r0 == r1 and abs(c0 - c1) == 1) or (c0 == c1 and abs(r0 - r1) == 1):
                adjacent_groups += 1

        participating.update(
            (key, position)
            for position in h_members | v_members
        )

    corner = any(
        horizontal_members[key] & vertical_members[key]
        for key in groups
    )

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "corner": corner,
        "colors": {
            key[0] for key in groups
        },
        "offsets": {
            (key[1], key[2]) for key in groups
        },
        "added": int(
            np.count_nonzero(input_grid != output_grid)
        ),
        "all_participate": (
            participating == all_positions
        ),
        "premarked": premarked,
        "empty_crossings": empty_crossings,
        "interior_spans": interior_spans,
        "adjacent_groups": adjacent_groups,
    }


def _grid_key(grid: np.ndarray) -> tuple:
    normalized = np.asarray(grid, dtype=np.int64)
    return normalized.shape, normalized.tobytes()


def _matches_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    if (
        input_grid.shape != (8, 8)
        or output_grid.shape != (8, 8)
    ):
        return False

    input_text = tuple(
        "".join(str(int(value)) for value in row)
        for row in input_grid
    )
    output_text = tuple(
        "".join(str(int(value)) for value in row)
        for row in output_grid
    )

    return any(
        input_text == source_input
        and output_text == source_output
        for source_input, source_output in _SOURCE_PAIRS
    )


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    index: int,
    params: dict,
    puzzle_params: dict,
    used_train_offsets: set,
) -> tuple[np.ndarray, np.ndarray]:
    if regime == "source_train" and index < 2:
        mode = ("corner_v", "corner_h")[index]
    else:
        modes = (
            "h", "v", "corner", "cross", "pre_h", "pre_v",
            "corner_h", "corner_v", "two_h", "two_v", "mixed",
            "h_decoy", "v_decoy", "cross_h", "cross_v",
            "pre_h_v", "pre_v_h",
        )
        mode = modes[int(rng.integers(len(modes)))]
    kinds = _MODES[mode]
    possible_offsets = [
        (row, col)
        for row in range(params["panel_height"])
        for col in range(params["panel_width"])
    ]
    available = possible_offsets
    if regime == "source_train" and index < 2:
        unused = [offset for offset in possible_offsets if offset not in used_train_offsets]
        if len(unused) >= len(kinds):
            available = unused
    if regime == "target_test":
        familiar = [offset for offset in possible_offsets if offset in used_train_offsets]
        novel = [offset for offset in possible_offsets if offset not in used_train_offsets]
        if novel and len(used_train_offsets) >= 3 and int(rng.integers(3)) == 0:
            first = novel[int(rng.integers(len(novel)))]
            rest = [offset for offset in familiar if offset != first]
            if len(rest) < len(kinds) - 1:
                rest = [offset for offset in possible_offsets if offset != first]
            offsets = [first] + [rest[int(i)] for i in rng.choice(len(rest), size=len(kinds)-1, replace=False)]
        else:
            if len(familiar) >= len(kinds):
                available = familiar
            offsets = [available[int(i)] for i in rng.choice(len(available), size=len(kinds), replace=False)]
    else:
        offsets = [available[int(i)] for i in rng.choice(len(available), size=len(kinds), replace=False)]

    train_colors = puzzle_params["train_colors"]
    fresh_colors = puzzle_params["fresh_colors"]
    if regime == "source_train":
        palette = train_colors
    else:
        color_mode = int(rng.integers(3))
        palette = (train_colors, fresh_colors, train_colors + fresh_colors)[color_mode]
    if regime == "source_train" and index < 2:
        colors = [train_colors[int(i)] for i in rng.choice(len(train_colors), size=2, replace=False)]
    elif len(kinds) == 1 or int(rng.integers(4)) == 0:
        color = palette[int(rng.integers(len(palette)))]
        colors = [color] * len(kinds)
    else:
        colors = [palette[int(i)] for i in rng.choice(len(palette), size=len(kinds), replace=False)]
    components = [
        _component(rng, kind, params, int(color), offset)
        for kind, color, offset in zip(kinds, colors, offsets)
    ]
    return _render_pair(params, components)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)

    num_train, num_test = _select_counts(
        rng,
        num_train,
        num_test,
    )
    puzzle_params = _sample_puzzle_parameters(rng)

    pair_seeds = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=num_train + num_test,
        dtype=np.uint64,
    )

    puzzle = {
        "train": [],
        "test": [],
    }
    seen_inputs = set()
    used_train_offsets = set()

    for overall_index, pair_seed in enumerate(
        pair_seeds
    ):
        is_train = overall_index < num_train

        if is_train:
            regime = "source_train"
            pair_index = overall_index
            destination = puzzle["train"]
        else:
            regime = "target_test"
            pair_index = overall_index - num_train
            destination = puzzle["test"]

        pair_rng = np.random.default_rng(pair_seed)

        for _ in range(200):
            params = _pair_parameters(pair_rng, puzzle_params, overall_index, is_train)
            try:
                input_grid, output_grid = _generate_pair(
                    pair_rng, regime, pair_index, params,
                    puzzle_params, used_train_offsets,
                )
            except ValueError:
                continue
            input_key = _grid_key(input_grid)

            if (
                input_key not in seen_inputs
                and not _matches_source_pair(
                    input_grid,
                    output_grid,
                )
            ):
                seen_inputs.add(input_key)
                destination.append(
                    {
                        "input": input_grid,
                        "output": output_grid,
                    }
                )
                if is_train:
                    used_train_offsets.update(
                        _pair_features(input_grid, output_grid)["offsets"]
                    )
                break
        else:
            raise ValueError(
                "could not create enough distinct pairs"
            )

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(
    puzzle: dict,
) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_keys = set()
    pair_keys = set()
    train_features = []
    test_features = []

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(
                    grid.dtype,
                    np.integer,
                )
                assert np.all(
                    (0 <= grid) & (grid <= 9)
                )

            assert input_grid.shape == output_grid.shape

            input_key = _grid_key(input_grid)
            output_key = _grid_key(output_grid)

            assert input_key not in input_keys
            assert (
                input_key,
                output_key,
            ) not in pair_keys

            input_keys.add(input_key)
            pair_keys.add(
                (input_key, output_key)
            )

            assert not _matches_source_pair(
                input_grid,
                output_grid,
            )

            input_info = _inspect_lattice(input_grid)
            output_info = _inspect_lattice(output_grid)

            assert (
                input_info["signature"]
                == output_info["signature"]
            )

            expected = _solve_grid(input_grid)
            assert np.array_equal(
                output_grid,
                expected,
            )
            assert np.array_equal(
                output_grid,
                _independent_completion(input_grid),
            )

            features = _pair_features(
                input_grid,
                output_grid,
            )

            assert features["added"] > 0
            assert features["horizontal"] + features["vertical"] >= 1
            assert np.array_equal(
                input_grid[input_grid == input_info["background"]],
                output_grid[input_grid == input_info["background"]],
            )

            if split == "train":
                train_features.append(features)
            else:
                test_features.append(features)

    assert train_features[0]["horizontal"] >= 1
    assert train_features[0]["vertical"] >= 2
    assert train_features[0]["corner"]
    assert len(train_features[0]["colors"]) >= 2
    assert train_features[1]["horizontal"] >= 2
    assert train_features[1]["vertical"] >= 1
    assert train_features[1]["corner"]
    assert len(train_features[1]["colors"]) >= 2

    train_colors = set().union(
        *(features["colors"] for features in train_features)
    )
    train_offsets = set().union(
        *(features["offsets"] for features in train_features)
    )
    assert len(train_colors) >= 2
    assert len(train_offsets) >= 3
    for features in test_features:
        if (features["horizontal"] + features["vertical"]) >= 3:
            assert train_features[0]["vertical"] >= 2
            assert train_features[1]["horizontal"] >= 2


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first) != set(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False

            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def self_test() -> None:
    automatic = [
        generate_puzzle(seed)
        for seed in range(6)
    ]

    explicit_cases = [
        (101, 2, 1),
        (102, 10, 1),
        (103, 10, 4),
        (104, 6, 3),
    ]
    explicit = [
        generate_puzzle(seed, n_train, n_test)
        for seed, n_train, n_test in explicit_cases
    ]

    partial = [
        generate_puzzle(201, None, 1),
        generate_puzzle(202, None, 4),
        generate_puzzle(203, 2, None),
        generate_puzzle(204, 10, None),
    ]

    for puzzle in automatic + explicit + partial:
        validate_generated_puzzle(puzzle)

    assert len(partial[0]["test"]) == 1
    assert len(partial[1]["test"]) == 4
    assert len(partial[2]["train"]) == 2
    assert len(partial[2]["test"]) == 1
    assert len(partial[3]["train"]) == 10

    first = generate_puzzle(999, 7, 3)
    second = generate_puzzle(999, 7, 3)
    assert _puzzles_equal(first, second)

    different = sum(
        not _puzzles_equal(automatic[0], puzzle)
        for puzzle in automatic[1:]
    )
    assert different >= 4

    illegal = [
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 2.0, "num_test": 1},
    ]

    for kwargs in illegal:
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"illegal count arguments accepted: {kwargs}"
            )

    corpus = [
        generate_puzzle(1000 + seed)
        for seed in range(42)
    ]

    train_counts = {
        len(puzzle["train"])
        for puzzle in corpus
    }
    test_counts = {
        len(puzzle["test"])
        for puzzle in corpus
    }

    signatures = {
        _inspect_lattice(
            puzzle["train"][0]["input"]
        )["signature"]
        for puzzle in corpus
    }
    shapes = {
        puzzle["train"][0]["input"].shape
        for puzzle in corpus
    }
    geometries = {
        signature[:6]
        for signature in signatures
    }
    role_assignments = {
        signature[6:]
        for signature in signatures
    }

    offset_vocabularies = set()
    first_inputs = set()

    for puzzle in corpus:
        offsets = set()
        for pair in puzzle["train"]:
            features = _pair_features(
                pair["input"],
                pair["output"],
            )
            offsets.update(features["offsets"])

        offset_vocabularies.add(
            tuple(sorted(offsets))
        )
        first_inputs.add(
            _grid_key(
                puzzle["train"][0]["input"]
            )
        )

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(shapes) >= 8
    assert len(geometries) >= 20
    assert len(role_assignments) >= 20
    assert len(offset_vocabularies) >= 6
    assert len(first_inputs) >= 40

    for puzzle in corpus:
        validate_generated_puzzle(puzzle)

    for source_input, source_output in _SOURCE_PAIRS:
        input_grid = np.array([[int(cell) for cell in row] for row in source_input], dtype=np.int8)
        output_grid = np.array([[int(cell) for cell in row] for row in source_output], dtype=np.int8)
        assert np.array_equal(_independent_completion(input_grid), output_grid)

    measured = {
        name: 0 for name in (
            "layout_changes", "role_changes", "train_multicolor",
            "train_three", "test_one_color", "test_three",
            "test_reused_color", "test_new_color", "test_new_offset",
            "premarked", "empty_crossings", "adjacent_groups",
            "interior_spans", "asymmetric", "source_sized",
            "cross_multicolor", "pre_multicolor",
        )
    }
    for puzzle in corpus:
        pairs = puzzle["train"] + puzzle["test"]
        signatures = [_inspect_lattice(pair["input"])["signature"] for pair in pairs]
        measured["layout_changes"] += len({s[:6] for s in signatures}) > 1
        measured["role_changes"] += len({s[6:] for s in signatures}) > 1
        train_features = [_pair_features(pair["input"], pair["output"]) for pair in puzzle["train"]]
        test_features = [_pair_features(pair["input"], pair["output"]) for pair in puzzle["test"]]
        train_colors = set().union(*(f["colors"] for f in train_features))
        train_offsets = set().union(*(f["offsets"] for f in train_features))
        assert len(train_offsets) >= 3
        for split, split_pairs, split_features in (
            ("train", puzzle["train"], train_features),
            ("test", puzzle["test"], test_features),
        ):
            for pair, features in zip(split_pairs, split_features):
                line_count = features["horizontal"] + features["vertical"]
                if split == "train":
                    measured["train_multicolor"] += len(features["colors"]) >= 2
                    measured["train_three"] += line_count >= 3
                else:
                    measured["test_one_color"] += len(features["colors"]) == 1
                    measured["test_three"] += line_count >= 3
                    measured["test_reused_color"] += bool(features["colors"] & train_colors)
                    measured["test_new_color"] += bool(features["colors"] - train_colors)
                    measured["test_new_offset"] += bool(features["offsets"] - train_offsets)
                for name in ("premarked", "empty_crossings", "adjacent_groups", "interior_spans"):
                    measured[name] += features[name] > 0
                measured["cross_multicolor"] += features["empty_crossings"] > 0 and len(features["colors"]) > 1
                measured["pre_multicolor"] += features["premarked"] > 0 and len(features["colors"]) > 1
                signature = _inspect_lattice(pair["input"])["signature"]
                measured["asymmetric"] += signature[:2] in ((3, 6), (6, 3))
                measured["source_sized"] += pair["input"].shape == (8, 8)
    assert measured["layout_changes"] >= 8
    assert measured["role_changes"] >= 8
    assert measured["train_multicolor"] >= 80
    assert measured["train_three"] >= 50
    assert measured["test_one_color"] >= 20
    assert measured["test_three"] >= 8
    assert measured["test_reused_color"] >= 15
    assert measured["test_new_color"] >= 15
    assert measured["test_new_offset"] >= 2
    assert measured["premarked"] >= 4
    assert measured["empty_crossings"] >= 4
    assert measured["adjacent_groups"] >= 4
    assert measured["interior_spans"] >= 8
    assert measured["asymmetric"] >= 4
    assert measured["source_sized"] >= 4
    assert measured["cross_multicolor"] >= 1
    assert measured["pre_multicolor"] >= 1
