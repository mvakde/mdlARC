import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rows_signature(rows: tuple[str, ...]) -> tuple:
    return (
        len(rows),
        len(rows[0]),
        tuple(int(ch) for row in rows for ch in row),
    )


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(v) for v in grid.ravel()),
    )


# Rejecting an exact source input is stronger than merely rejecting an exact
# source input/output pair, since this task's output is deterministic.
_SOURCE_INPUT_SIGNATURES = frozenset(
    {
        _rows_signature(
            (
                "0002200000",
                "0322220300",
                "0000222000",
                "0000200000",
                "0002200000",
                "0000000000",
                "0000000000",
                "0300000300",
                "0000000000",
            )
        ),
        _rows_signature(
            (
                "000000000",
                "030000030",
                "002220000",
                "222000000",
                "202000000",
                "222200000",
                "000200000",
                "030000030",
                "000000000",
            )
        ),
        _rows_signature(
            (
                "30000030",
                "00020000",
                "02220000",
                "02000000",
                "22222000",
                "02000000",
                "30000030",
                "00000000",
                "00000000",
            )
        ),
        _rows_signature(
            (
                "3020003",
                "0220000",
                "2222200",
                "0200000",
                "0220000",
                "0000000",
                "3000003",
            )
        ),
        _rows_signature(
            (
                "00000000",
                "30000030",
                "20000000",
                "22000000",
                "22200000",
                "22222000",
                "20000000",
                "30000030",
                "00000000",
                "00000000",
            )
        ),
    }
)


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False

    start = min(coords)
    seen = {start}
    stack = [start]

    while stack:
        r, c = stack.pop()
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    return seen == coords


def _has_enclosed_hole(shape: tuple[int, int, tuple[tuple[int, int], ...]]) -> bool:
    height, width, cells_tuple = shape
    cells = set(cells_tuple)
    empty = {(r, c) for r in range(height) for c in range(width)} - cells
    exterior = {p for p in empty if p[0] in (0, height - 1) or p[1] in (0, width - 1)}
    stack = list(exterior)
    while stack:
        r, c = stack.pop()
        for next_cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if next_cell in empty and next_cell not in exterior:
                exterior.add(next_cell)
                stack.append(next_cell)
    return bool(empty - exterior)


def _frontier(
    cells: set[tuple[int, int]],
    height: int,
    width: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    result: set[tuple[int, int]] = set()

    for r, c in sorted(cells):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (
                0 <= nr < height
                and 0 <= nc < width
                and (nr, nc) not in cells
                and (nr, nc) not in forbidden
            ):
                result.add((nr, nc))

    return sorted(result)


def _touches_all_sides(
    cells: set[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    return (
        any(r == 0 for r, _ in cells)
        and any(r == height - 1 for r, _ in cells)
        and any(c == 0 for _, c in cells)
        and any(c == width - 1 for _, c in cells)
    )


def _sample_payload_shape(
    rng: np.random.Generator,
    height: int,
    width: int,
    forbidden_cells: set[tuple[int, int]],
    fixed_shape: tuple[tuple[int, int], ...] | None = None,
) -> tuple[tuple[int, int], ...] | None:
    if fixed_shape is not None:
        cells = set(fixed_shape)
        if (
            len(cells) >= 5
            and not cells & forbidden_cells
            and _is_connected(cells)
            and _touches_all_sides(cells, height, width)
        ):
            return fixed_shape
        return None

    allowed = {(r, c) for r in range(height) for c in range(width)} - forbidden_cells
    if len(allowed) < 5:
        return None
    mode = int(rng.integers(0, 10))
    for _ in range(40):
        if mode == 0 and not forbidden_cells:
            # A solid boundary around a small enclosed hole.
            cells = set(allowed)
            center = (int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1)))
            cells.remove(center)
        else:
            start = (int(rng.integers(0, height)), int(rng.integers(0, width)))
            if start not in allowed:
                continue
            cells = {start}
            targets = [
                (0, int(rng.integers(0, width))),
                (height - 1, int(rng.integers(0, width))),
                (int(rng.integers(0, height)), 0),
                (int(rng.integers(0, height)), width - 1),
            ]
            failed = False
            for target in targets:
                if target not in allowed:
                    failed = True
                    break
                here = min(cells, key=lambda p: abs(p[0] - target[0]) + abs(p[1] - target[1]))
                while here != target:
                    steps = []
                    if here[0] != target[0]:
                        steps.append((here[0] + (1 if target[0] > here[0] else -1), here[1]))
                    if here[1] != target[1]:
                        steps.append((here[0], here[1] + (1 if target[1] > here[1] else -1)))
                    steps = [p for p in steps if p in allowed]
                    if not steps:
                        failed = True
                        break
                    here = steps[int(rng.integers(0, len(steps)))]
                    cells.add(here)
                if failed:
                    break
            if failed:
                continue

            area = height * width
            if mode <= 5:
                target_size = int(rng.integers(max(5, int(area * 0.30)), max(6, int(area * 0.50)) + 1))
                if height == width == 5 and mode <= 3:
                    target_size = int(rng.integers(11, 14))
            else:
                target_size = int(rng.integers(max(5, int(area * 0.58)), max(6, int(area * 0.82)) + 1))
            target_size = min(target_size, len(allowed) - (0 if forbidden_cells else 1))
            while len(cells) < target_size:
                frontier = _frontier(cells, height, width, forbidden_cells)
                if not frontier:
                    break
                cells.add(frontier[int(rng.integers(0, len(frontier)))])

        if len(cells) >= 5 and len(cells) < height * width and _is_connected(cells) and _touches_all_sides(cells, height, width):
            return tuple(sorted(cells))
    return None


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    colors: tuple[int, int, int],
    displacement: tuple[int, int],
    size_mode: str = "general",
    fixed_destination: tuple[int, int, int, int] | None = None,
    fixed_shape: tuple[tuple[int, int], ...] | None = None,
    required_displacements: tuple[tuple[int, int], ...] = (),
) -> tuple[dict, dict]:
    background, marker, payload = colors
    delta_row, delta_col = displacement
    modes = {
        "source": ((10, 14), (7, 12)),
        "similar": ((7, 11), (7, 11)),
        "wide": ((7, 11), (12, 18)),
        "tall": ((12, 18), (7, 11)),
        "general": ((7, 18), (7, 18)),
    }
    if regime not in ("source_train", "target_test") or size_mode not in modes:
        raise ValueError("Unknown pair regime or size mode")
    if regime == "source_train":
        height_range, width_range = (7, 18), (7, 18)
    else:
        height_range, width_range = modes[size_mode]

    for _ in range(500):
        if fixed_destination is None:
            if fixed_shape is None:
                interior_height = int(rng.integers(3, 7 if regime == "source_train" else 8))
                interior_width = int(rng.integers(3, 7 if regime == "source_train" else 8))
            else:
                interior_height = max(r for r, _ in fixed_shape) + 1
                interior_width = max(c for _, c in fixed_shape) + 1
        else:
            _, _, interior_height, interior_width = fixed_destination

        height = int(rng.integers(height_range[0], height_range[1] + 1))
        width = int(rng.integers(width_range[0], width_range[1] + 1))
        displacements = (displacement,) + required_displacements
        min_top = max([0] + [dr - 1 for dr, _ in displacements])
        max_top = min([height - interior_height - 2] + [height - interior_height - 1 + dr for dr, _ in displacements])
        min_left = max([0] + [dc - 1 for _, dc in displacements])
        max_left = min([width - interior_width - 2] + [width - interior_width - 1 + dc for _, dc in displacements])
        if min_top > max_top or min_left > max_left:
            continue
        if fixed_destination is None:
            top = int(rng.integers(min_top, max_top + 1))
            left = int(rng.integers(min_left, max_left + 1))
        else:
            top, left = fixed_destination[:2]
            if not (min_top <= top <= max_top and min_left <= left <= max_left):
                continue
        bottom, right = top + interior_height + 1, left + interior_width + 1
        if size_mode == "source" and (height - bottom - 1 < 2 or height <= 9):
            continue
        source_row, source_col = top + 1 - delta_row, left + 1 - delta_col
        marker_positions = {(top, left), (top, right), (bottom, left), (bottom, right)}
        forbidden_relative = {
            (r - source_row, c - source_col)
            for r, c in marker_positions
            if source_row <= r < source_row + interior_height
            and source_col <= c < source_col + interior_width
        }
        shape = _sample_payload_shape(rng, interior_height, interior_width, forbidden_relative, fixed_shape)
        if shape is None:
            continue

        input_grid = np.full(
            (height, width), background, dtype=np.int8
        )
        output_grid = np.full(
            (height, width), background, dtype=np.int8
        )

        for r, c in marker_positions:
            input_grid[r, c] = marker
            output_grid[r, c] = marker

        for rr, cc in shape:
            input_grid[source_row + rr, source_col + cc] = payload
            output_grid[
                top + 1 + rr, left + 1 + cc
            ] = payload

        pair = {
            "input": input_grid,
            "output": output_grid,
        }
        metadata = {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "delta_row": delta_row,
            "delta_col": delta_col,
            "shape": shape,
        }
        return pair, metadata

    raise RuntimeError("Could not generate a feasible pair")


def _validate_optional_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> None:
    if value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not minimum <= int(value) <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    _require(
        input_grid.shape == output_grid.shape,
        "Input and output dimensions differ",
    )
    _require(
        not np.array_equal(input_grid, output_grid),
        "Generated transformation must move the payload",
    )

    input_colors = {int(v) for v in np.unique(input_grid)}
    output_colors = {int(v) for v in np.unique(output_grid)}
    _require(
        input_colors == output_colors and len(input_colors) == 3,
        "Each pair must use exactly three preserved semantic colors",
    )

    marker_candidates: list[tuple[int, set[tuple[int, int]]]] = []
    for color in sorted(input_colors):
        input_positions = {
            (int(r), int(c))
            for r, c in np.argwhere(input_grid == color)
        }
        output_positions = {
            (int(r), int(c))
            for r, c in np.argwhere(output_grid == color)
        }

        if len(input_positions) != 4 or input_positions != output_positions:
            continue

        rows = sorted({r for r, _ in input_positions})
        cols = sorted({c for _, c in input_positions})
        if len(rows) != 2 or len(cols) != 2:
            continue

        corners = {
            (rows[0], cols[0]),
            (rows[0], cols[1]),
            (rows[1], cols[0]),
            (rows[1], cols[1]),
        }
        if input_positions == corners:
            marker_candidates.append((color, input_positions))

    _require(
        len(marker_candidates) == 1,
        "There must be one uniquely identifiable landmark color",
    )

    marker_color, marker_positions = marker_candidates[0]
    marker_rows = sorted({r for r, _ in marker_positions})
    marker_cols = sorted({c for _, c in marker_positions})
    top, bottom = marker_rows
    left, right = marker_cols

    interior_height = bottom - top - 1
    interior_width = right - left - 1
    _require(
        interior_height >= 3 and interior_width >= 3,
        "Landmark interior is too small",
    )

    payload_candidates: list[tuple] = []
    for color in sorted(input_colors - {marker_color}):
        input_positions = {
            (int(r), int(c))
            for r, c in np.argwhere(input_grid == color)
        }
        output_positions = {
            (int(r), int(c))
            for r, c in np.argwhere(output_grid == color)
        }

        if not input_positions or len(input_positions) != len(output_positions):
            continue

        input_box = _bbox(input_positions)
        output_box = _bbox(output_positions)

        input_box_height = input_box[1] - input_box[0] + 1
        input_box_width = input_box[3] - input_box[2] + 1

        if (
            input_box_height != interior_height
            or input_box_width != interior_width
        ):
            continue

        expected_output_box = (
            top + 1,
            bottom - 1,
            left + 1,
            right - 1,
        )
        if output_box != expected_output_box:
            continue

        delta_row = output_box[0] - input_box[0]
        delta_col = output_box[2] - input_box[2]
        translated = {
            (r + delta_row, c + delta_col)
            for r, c in input_positions
        }
        if translated != output_positions:
            continue

        payload_candidates.append(
            (
                color,
                input_positions,
                output_positions,
                input_box,
                output_box,
                delta_row,
                delta_col,
            )
        )

    _require(
        len(payload_candidates) == 1,
        "There must be one rigidly translated payload",
    )

    (
        payload_color,
        input_payload,
        output_payload,
        input_box,
        output_box,
        delta_row,
        delta_col,
    ) = payload_candidates[0]

    background_candidates = input_colors - {
        marker_color,
        payload_color,
    }
    _require(
        len(background_candidates) == 1,
        "The background role is not unique",
    )
    background_color = next(iter(background_candidates))

    _require(
        len(input_payload) >= 5,
        "Payload is too small to be structurally informative",
    )
    _require(
        _is_connected(input_payload),
        "Input payload must be four-connected",
    )
    _require(
        _is_connected(output_payload),
        "Output payload must remain four-connected",
    )

    expected = input_grid.copy()
    expected[input_grid == payload_color] = background_color
    for r, c in output_payload:
        expected[r, c] = payload_color

    _require(
        np.array_equal(expected, output_grid),
        "Output changes something other than rigid payload relocation",
    )

    normalized_shape = tuple(
        sorted(
            (r - input_box[0], c - input_box[2])
            for r, c in input_payload
        )
    )
    shape_signature = (
        interior_height,
        interior_width,
        normalized_shape,
    )

    return {
        "height": int(input_grid.shape[0]),
        "width": int(input_grid.shape[1]),
        "background": background_color,
        "marker": marker_color,
        "payload": payload_color,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "delta_row": delta_row,
        "delta_col": delta_col,
        "shape_signature": shape_signature,
        "input_box": input_box,
        "output_box": output_box,
    }


def _choose_displacement(rng: np.random.Generator, axis: str) -> tuple[int, int]:
    magnitude = int(rng.choice((1, 1, 2, 2, 3, 4, 5, 6)))
    sign = -1 if rng.integers(0, 2) else 1
    if axis == "horizontal":
        return (0, sign * magnitude)
    if axis == "vertical":
        return (sign * magnitude, 0)
    return (sign * int(rng.integers(1, 4)),
            (-1 if rng.integers(0, 2) else 1) * int(rng.integers(1, 4)))


def _sample_colors(rng: np.random.Generator) -> tuple[int, int, int]:
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    rng = np.random.default_rng(seed)
    if num_train is None and num_test is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif num_test is None:
        train_count = int(num_train)
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        train_count, test_count = int(num_train), int(num_test)

    # The only puzzle-wide conventions are whether colors persist between pairs
    # and which displacement examples are demonstrated. Neither changes the rule.
    source_like = bool(rng.integers(0, 5) == 0)
    recolor = bool(rng.integers(0, 2))
    repeat_mode = ("destination", "shape", "none")[int(rng.integers(0, 3))]
    base_colors = _sample_colors(rng)
    if source_like:
        train_moves = [_choose_displacement(rng, "vertical"), (0, 1)]
    else:
        train_moves = [_choose_displacement(rng, "horizontal"),
                       _choose_displacement(rng, "vertical")]
    if train_count >= 3:
        train_moves.append(_choose_displacement(rng, "diagonal"))
    while len(train_moves) < train_count:
        move = _choose_displacement(rng, ("horizontal", "vertical", "diagonal")
                                    [int(rng.integers(0, 3))])
        train_moves.append(move)

    train_pairs: list[dict] = []
    train_meta: list[dict] = []
    used_inputs: set[tuple] = set()
    for index, move in enumerate(train_moves):
        if recolor:
            colors = _sample_colors(rng)
            if index == 1:
                while colors == train_meta[0]["colors"]:
                    colors = _sample_colors(rng)
        else:
            colors = base_colors
        fixed_destination = None
        fixed_shape = None
        required = ()
        if index == 0 and repeat_mode == "destination":
            required = (train_moves[1],)
        if index == 1:
            if repeat_mode == "destination":
                first = train_meta[0]
                fixed_destination = (first["top"], first["left"],
                                     first["interior_height"], first["interior_width"])
            elif repeat_mode == "shape":
                fixed_shape = train_meta[0]["shape"]
        for _ in range(30):
            pair, meta = _generate_pair(rng, "source_train", colors, move,
                                        fixed_destination=fixed_destination,
                                        fixed_shape=fixed_shape,
                                        required_displacements=required)
            signature = _grid_signature(pair["input"])
            if signature in used_inputs or signature in _SOURCE_INPUT_SIGNATURES:
                continue
            if index == 1 and repeat_mode == "shape":
                first = train_meta[0]
                if (meta["top"], meta["left"], meta["bottom"], meta["right"]) == (
                    first["top"], first["left"], first["bottom"], first["right"]):
                    continue
            if index == 1 and repeat_mode == "destination" and meta["shape"] == train_meta[0]["shape"]:
                continue
            break
        else:
            raise RuntimeError("Could not construct distinct training pair")
        meta["colors"] = colors
        meta["interior_height"] = meta["bottom"] - meta["top"] - 1
        meta["interior_width"] = meta["right"] - meta["left"] - 1
        train_pairs.append(pair)
        train_meta.append(meta)
        used_inputs.add(signature)

    unique_moves = list(dict.fromkeys(train_moves))
    rng.shuffle(unique_moves)
    test_pairs: list[dict] = []
    for index in range(test_count):
        move = (0, 1) if source_like and index == 0 else unique_moves[index % len(unique_moves)]
        colors = _sample_colors(rng) if recolor else base_colors
        size_mode = "source" if source_like and index == 0 else (
            "similar", "wide", "tall", "general"
        )[int(rng.integers(0, 4))]
        for _ in range(30):
            pair, meta = _generate_pair(rng, "target_test", colors, move,
                                        size_mode=size_mode)
            signature = _grid_signature(pair["input"])
            if signature not in used_inputs and signature not in _SOURCE_INPUT_SIGNATURES:
                break
        else:
            raise RuntimeError("Could not construct distinct test pair")
        test_pairs.append(pair)
        used_inputs.add(signature)

    puzzle = {"train": train_pairs, "test": test_pairs}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "Splits must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest,
             "Illegal pair counts")
    inputs: set[tuple] = set()
    pairs: set[tuple] = set()
    infos: dict[str, list[dict]] = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "Malformed pair")
            for grid in (pair["input"], pair["output"]):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "Grids must be two-dimensional arrays")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                         "Illegal grid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer), "Grid must use integers")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "Illegal ARC colors")
            input_sig = _grid_signature(pair["input"])
            pair_sig = (input_sig, _grid_signature(pair["output"]))
            _require(input_sig not in inputs and pair_sig not in pairs,
                     "Duplicate input or pair")
            _require(input_sig not in _SOURCE_INPUT_SIGNATURES, "Source input copied")
            inputs.add(input_sig)
            pairs.add(pair_sig)
            info = _analyze_pair(pair["input"], pair["output"])
            _require(7 <= info["height"] <= 18 and 7 <= info["width"] <= 18,
                     "Canvas outside supported regime")
            _require(abs(info["delta_row"]) <= 6 and abs(info["delta_col"]) <= 6,
                     "Displacement outside supported range")
            infos[split].append(info)

    train = infos["train"]
    test = infos["test"]
    _require(any(i["delta_row"] == 0 and i["delta_col"] != 0 for i in train),
             "Training lacks horizontal alignment evidence")
    _require(any(i["delta_col"] == 0 and i["delta_row"] != 0 for i in train),
             "Training lacks vertical alignment evidence")
    train_colors = {(i["background"], i["marker"], i["payload"]) for i in train}
    all_colors = train_colors | {(i["background"], i["marker"], i["payload"]) for i in test}
    _require(len(all_colors) == 1 or len(train_colors) >= 2,
             "Recoloring must be demonstrated in training")
    demonstrated_moves = {(i["delta_row"], i["delta_col"]) for i in train}
    for info in test:
        _require((info["delta_row"], info["delta_col"]) in demonstrated_moves,
                 "Test displacement is not demonstrated in training")


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((split, tuple((_grid_signature(pair["input"]),
                                _grid_signature(pair["output"]))
                               for pair in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(12001)
    _require(_puzzle_signature(a) == _puzzle_signature(generate_puzzle(12001)),
             "Automatic generation is nondeterministic")
    b = generate_puzzle(12002, 6, 3)
    _require(_puzzle_signature(b) == _puzzle_signature(generate_puzzle(12002, 6, 3)),
             "Explicit generation is nondeterministic")
    _require(_puzzle_signature(a) != _puzzle_signature(b), "Seeds did not differ")
    for index, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4),
                                      (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(13000 + index, nt, ns)
        validate_generated_puzzle(p)
        if nt is not None:
            _require(len(p["train"]) == nt, "Training count not honored")
        if ns is not None:
            _require(len(p["test"]) == ns, "Test count not honored")
    for index, (nt, ns) in enumerate(((1, None), (11, None), (None, 0),
                                      (None, 5), (2, 2), (2, 3), (4, 4))):
        try:
            generate_puzzle(14000 + index, nt, ns)
        except ValueError:
            continue
        raise AssertionError("Illegal count accepted")

    train_counts, test_counts, signatures = set(), set(), set()
    directions, sizes, shapes, palettes = set(), set(), set(), set()
    sparse = negative = long_moves = repeated_dest = repeated_shape = recolored = 0
    holes = sparse_five = fresh_anchors = little_bottom = test_negative = 0
    test_outside = right_one = multi_test_moves = combined = 0
    regimes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_infos = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_infos = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        all_infos = train_infos + test_infos
        train_dests = [(i["top"], i["bottom"], i["left"], i["right"]) for i in train_infos]
        train_shapes = [i["shape_signature"] for i in train_infos]
        repeated_dest += len(train_dests) != len(set(train_dests))
        repeated_shape += len(train_shapes) != len(set(train_shapes))
        role_tuples = {(i["background"], i["marker"], i["payload"]) for i in all_infos}
        recolored += len(role_tuples) > 1
        multi_test_moves += len({(i["delta_row"], i["delta_col"]) for i in test_infos}) > 1
        palettes |= role_tuples
        for info in all_infos:
            dr, dc = info["delta_row"], info["delta_col"]
            directions.add((0 if dr == 0 else (1 if dr > 0 else -1),
                            0 if dc == 0 else (1 if dc > 0 else -1)))
            sizes.add((info["height"], info["width"]))
            shapes.add(info["shape_signature"])
            occupancy = len(info["shape_signature"][2]) / (info["shape_signature"][0] * info["shape_signature"][1])
            sparse += occupancy <= 0.52
            negative += dr < 0 or dc < 0
            long_moves += abs(dr) >= info["shape_signature"][0] or abs(dc) >= info["shape_signature"][1]
            holes += _has_enclosed_hole(info["shape_signature"])
            sparse_five += (info["shape_signature"][0:2] == (5, 5)
                            and 11 <= len(info["shape_signature"][2]) <= 13)
        for info in test_infos:
            dr, dc = info["delta_row"], info["delta_col"]
            box_height, box_width = info["shape_signature"][:2]
            test_negative += dr < 0 or dc < 0
            test_outside += abs(dr) >= box_height or abs(dc) >= box_width
            right_one += (dr, dc) == (0, 1)
            little_bottom += info["height"] - info["bottom"] - 1 <= 1
            fresh_anchors += (info["top"] not in {t["top"] for t in train_infos}
                              or info["left"] not in {t["left"] for t in train_infos})
            combined += (len(role_tuples) > 1 and (dr < 0 or dc < 0)
                         and (abs(dr) >= box_height or abs(dc) >= box_width))
            if info["height"] >= 10 and info["height"] - info["bottom"] - 1 >= 2 and (info["delta_row"], info["delta_col"]) == (0, 1):
                regimes.add("source")
            if info["height"] <= 11 and info["width"] <= 11:
                regimes.add("similar")
            if info["width"] >= 12 and info["height"] <= 11:
                regimes.add("wide")
            if info["height"] >= 12 and info["width"] <= 11:
                regimes.add("tall")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "Pair counts lack diversity")
    _require(len(signatures) >= 40 and len(sizes) >= 40 and len(shapes) >= 100,
             "Structural diversity is too low")
    _require(len(directions) >= 6 and negative >= 20 and long_moves >= 5,
             "Displacement diversity is too low")
    _require(sparse >= 20 and repeated_dest >= 4 and repeated_shape >= 4,
             "Shape and repeated geometry coverage is too low")
    _require(recolored >= 8 and len(palettes) >= 30, "Recoloring diversity is too low")
    _require(holes >= 10 and sparse_five >= 3, "Holes or source-like sparse shapes are scarce")
    _require(test_negative >= 10 and test_outside >= 3 and right_one >= 3,
             "Test displacement regimes are too narrow")
    _require(fresh_anchors >= 10 and little_bottom >= 10 and multi_test_moves >= 8,
             "Test layout and within-puzzle moves are too narrow")
    _require(combined >= 1, "Color, reverse motion, and long move never combine")
    _require(regimes == {"source", "similar", "wide", "tall"},
             "Test canvas regimes lack diversity")
